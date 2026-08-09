#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kWarpsPerBlock = kThreadsPerBlock / device::kWarpThreads;

struct Gemma4QKVNormRopeParams {
  bf16_t* __restrict__ qkv;
  const bf16_t* __restrict__ q_weight;
  const bf16_t* __restrict__ k_weight;
  const float* __restrict__ cos_sin_cache;
  const void* __restrict__ positions;
  int64_t token_stride;
  uint32_t num_q_heads;
  uint32_t num_kv_heads;
  uint32_t num_tokens;
  float eps;
};

template <typename PositionType, int64_t kHeadDim, bool kUsePDL>
__global__ void gemma4_qkv_norm_rope_kernel(const Gemma4QKVNormRopeParams __grid_constant__ params) {
  using namespace device;

  static_assert(kHeadDim == 256, "Gemma 4 specialization requires head_dim=256");
  static_assert(kHeadDim % kWarpThreads == 0, "head_dim must be divisible by warp size");

  constexpr uint32_t kElementsPerThread = kHeadDim / kWarpThreads;
  constexpr uint32_t kPackedElementsPerThread = kElementsPerThread / 2;
  constexpr uint32_t kHalfHeadDim = kHeadDim / 2;
  constexpr uint32_t kHeadPairsPerWarp = kWarpThreads / 2;
  constexpr uint32_t kPaddedElementsPerPair = kElementsPerThread + 1;
  constexpr uint32_t kCacheHalfStride = kHeadPairsPerWarp * kPaddedElementsPerPair;
  using Packed = packed_t<bf16_t>;
  using Storage = AlignedVector<Packed, kPackedElementsPerThread>;
  __shared__ float padded_rope_cache[2 * kCacheHalfStride];

  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t first_block_work_id = blockIdx.x * kWarpsPerBlock;
  const uint32_t block_work_stride = gridDim.x * kWarpsPerBlock;
  const uint32_t heads_per_token = params.num_q_heads + 2 * params.num_kv_heads;
  const uint32_t work_count = heads_per_token * params.num_tokens;
  const uint32_t rope_head_count = params.num_q_heads + params.num_kv_heads;

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t block_work_id = first_block_work_id; block_work_id < work_count;
       block_work_id += block_work_stride) {
    const uint32_t token_id = block_work_id / heads_per_token;
    const uint32_t first_head_id = block_work_id % heads_per_token;
    const bool block_applies_rope = first_head_id < rope_head_count;

    if (block_applies_rope) {
      const auto position = static_cast<int64_t>(static_cast<const PositionType*>(params.positions)[token_id]);
      const float* source_cache = params.cos_sin_cache + position * kHeadDim;
      const uint32_t source_index = threadIdx.x;
      const uint32_t cache_half = source_index / kHalfHeadDim;
      const uint32_t frequency_index = source_index % kHalfHeadDim;
      const uint32_t pair_lane = frequency_index / kElementsPerThread;
      const uint32_t element_index = frequency_index % kElementsPerThread;
      const uint32_t padded_index = cache_half * kCacheHalfStride +
          pair_lane * kPaddedElementsPerPair + element_index;
      padded_rope_cache[padded_index] = source_cache[source_index];
      __syncthreads();
    }

    const uint32_t head_id = first_head_id + warp_id;
    const bool is_q = head_id < params.num_q_heads;
    const bool is_k = head_id >= params.num_q_heads && head_id < params.num_q_heads + params.num_kv_heads;
    const bool apply_rope = is_q || is_k;

    bf16_t* input = params.qkv + token_id * params.token_stride + head_id * kHeadDim;
    const bf16_t* weight = is_q ? params.q_weight : params.k_weight;
    auto input_vector = load_as<Storage>(input, lane_id);

    float elements[kElementsPerThread];
    float sum_of_squares = 0.0f;

#pragma unroll
    for (uint32_t packed_index = 0; packed_index < kPackedElementsPerThread; ++packed_index) {
      const auto [x0, x1] = cast<fp32x2_t>(input_vector[packed_index]);
      elements[2 * packed_index] = x0;
      elements[2 * packed_index + 1] = x1;
      sum_of_squares += x0 * x0 + x1 * x1;
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const float inverse_rms = math::rsqrt(sum_of_squares / static_cast<float>(kHeadDim) + params.eps);

    if (apply_rope) {
      const auto weight_vector = load_as<Storage>(weight, lane_id);
#pragma unroll
      for (uint32_t packed_index = 0; packed_index < kPackedElementsPerThread; ++packed_index) {
        const auto [w0, w1] = cast<fp32x2_t>(weight_vector[packed_index]);
        input_vector[packed_index] = cast<Packed, fp32x2_t>(
            {elements[2 * packed_index] * inverse_rms * w0,
             elements[2 * packed_index + 1] * inverse_rms * w1});
        const auto [rounded0, rounded1] = cast<fp32x2_t>(input_vector[packed_index]);
        elements[2 * packed_index] = rounded0;
        elements[2 * packed_index + 1] = rounded1;
      }

#pragma unroll
      for (uint32_t element_index = 0; element_index < kElementsPerThread; ++element_index) {
        const float paired = __shfl_xor_sync(0xffffffffu, elements[element_index], kWarpThreads / 2);
        const float rotated = lane_id < kWarpThreads / 2 ? -paired : paired;
        const uint32_t pair_lane = lane_id % kHeadPairsPerWarp;
        const uint32_t cache_index = pair_lane * kPaddedElementsPerPair + element_index;
        const float cosine = padded_rope_cache[cache_index];
        const float sine = padded_rope_cache[kCacheHalfStride + cache_index];
        elements[element_index] = elements[element_index] * cosine + rotated * sine;
      }

#pragma unroll
      for (uint32_t packed_index = 0; packed_index < kPackedElementsPerThread; ++packed_index) {
        input_vector[packed_index] = cast<Packed, fp32x2_t>(
            {elements[2 * packed_index], elements[2 * packed_index + 1]});
      }
    } else {
#pragma unroll
      for (uint32_t packed_index = 0; packed_index < kPackedElementsPerThread; ++packed_index) {
        input_vector[packed_index] = cast<Packed, fp32x2_t>(
            {elements[2 * packed_index] * inverse_rms,
             elements[2 * packed_index + 1] * inverse_rms});
      }
    }

    store_as<Storage>(input, input_vector, lane_id);
    if (block_applies_rope) {
      __syncthreads();
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename PositionType, int64_t kHeadDim, bool kUsePDL>
void gemma4_qkv_norm_rope(
    tvm::ffi::TensorView qkv,
    tvm::ffi::TensorView q_weight,
    tvm::ffi::TensorView k_weight,
    tvm::ffi::TensorView cos_sin_cache,
    tvm::ffi::TensorView positions,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    double eps) {
  using namespace host;

  auto num_tokens = SymbolicSize{"num_tokens"};
  auto qkv_width = SymbolicSize{"qkv_width"};
  auto token_stride = SymbolicSize{"token_stride"};
  auto device = SymbolicDevice{};
  constexpr auto head_dim = kHeadDim;
  device.set_options<kDLCUDA>();

  TensorMatcher({num_tokens, qkv_width})
      .with_strides({token_stride, 1})
      .with_dtype<bf16_t>()
      .with_device(device)
      .verify(qkv);
  TensorMatcher({head_dim})
      .with_strides({1})
      .with_dtype<bf16_t>()
      .with_device(device)
      .verify(q_weight)
      .verify(k_weight);
  TensorMatcher({-1, head_dim})
      .with_strides({head_dim, 1})
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(cos_sin_cache);
  TensorMatcher({num_tokens})
      .with_strides({1})
      .with_dtype<PositionType>()
      .with_device(device)
      .verify(positions);

  RuntimeCheck(num_q_heads > 0, "num_q_heads must be positive");
  RuntimeCheck(num_kv_heads > 0, "num_kv_heads must be positive");
  RuntimeCheck(
      (num_q_heads + 2 * num_kv_heads) % kWarpsPerBlock == 0,
      "the QKV head count must be divisible by the warps per block");
  RuntimeCheck(
      qkv_width.unwrap() == (num_q_heads + 2 * num_kv_heads) * head_dim,
      "qkv width must equal (num_q_heads + 2 * num_kv_heads) * head_dim");

  const auto token_count = static_cast<uint32_t>(num_tokens.unwrap());
  const auto q_head_count = static_cast<uint32_t>(num_q_heads);
  const auto kv_head_count = static_cast<uint32_t>(num_kv_heads);
  const auto work_count = token_count * (q_head_count + 2 * kv_head_count);
  if (work_count == 0) return;

  const auto params = Gemma4QKVNormRopeParams{
      .qkv = static_cast<bf16_t*>(qkv.data_ptr()),
      .q_weight = static_cast<const bf16_t*>(q_weight.data_ptr()),
      .k_weight = static_cast<const bf16_t*>(k_weight.data_ptr()),
      .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
      .positions = positions.data_ptr(),
      .token_stride = token_stride.unwrap(),
      .num_q_heads = q_head_count,
      .num_kv_heads = kv_head_count,
      .num_tokens = token_count,
      .eps = static_cast<float>(eps),
  };

  constexpr auto kernel = gemma4_qkv_norm_rope_kernel<PositionType, kHeadDim, kUsePDL>;
  const uint32_t sm_count = runtime::get_sm_count(device.unwrap().device_id);
  static const uint32_t blocks_per_sm = runtime::get_blocks_per_sm(kernel, kThreadsPerBlock);
  const uint32_t blocks_per_token = (q_head_count + 2 * kv_head_count) / kWarpsPerBlock;
  const uint32_t needed_blocks = div_ceil(work_count, kWarpsPerBlock);
  uint32_t block_count = std::min(blocks_per_sm * sm_count, needed_blocks);
  if (block_count < needed_blocks) {
    block_count -= block_count % blocks_per_token;
    block_count = std::max(block_count, blocks_per_token);
  }
  LaunchKernel(block_count, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
}

}  // namespace
