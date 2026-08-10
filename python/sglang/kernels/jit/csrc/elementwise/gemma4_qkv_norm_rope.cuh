#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
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
constexpr uint32_t kWarpsPerHead = 4;
constexpr uint32_t kThreadsPerHead = kWarpsPerHead * device::kWarpThreads;
constexpr uint32_t kHeadsPerBlock = kThreadsPerBlock / kThreadsPerHead;

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
  static_assert(kHeadDim == 2 * kThreadsPerHead, "each reduction thread must own one BF16 pair");

  constexpr uint32_t kHalfHeadDim = kHeadDim / 2;
  using Packed = packed_t<bf16_t>;
  __shared__ float rope_cache[kHeadDim];
  __shared__ float warp_sums[kHeadsPerBlock][kWarpsPerHead];
  __shared__ Packed normalized_pairs[kHeadsPerBlock][kThreadsPerHead];

  const uint32_t head_group = threadIdx.x / kThreadsPerHead;
  const uint32_t head_thread = threadIdx.x % kThreadsPerHead;
  const uint32_t warp_in_head = head_thread / kWarpThreads;
  const uint32_t lane_id = head_thread % kWarpThreads;
  const uint32_t heads_per_token = params.num_q_heads + 2 * params.num_kv_heads;
  const uint32_t rope_head_count = params.num_q_heads + params.num_kv_heads;

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t token_id = blockIdx.x; token_id < params.num_tokens; token_id += gridDim.x) {
    const auto position = static_cast<int64_t>(static_cast<const PositionType*>(params.positions)[token_id]);
    const float* source_cache = params.cos_sin_cache + position * kHeadDim;
    rope_cache[threadIdx.x] = source_cache[threadIdx.x];
    __syncthreads();

    for (uint32_t head_id = head_group; head_id < heads_per_token; head_id += kHeadsPerBlock) {
      const bool is_q = head_id < params.num_q_heads;
      const bool is_k = head_id >= params.num_q_heads && head_id < rope_head_count;
      const bool apply_rope = is_q || is_k;

      bf16_t* input = params.qkv + token_id * params.token_stride + head_id * kHeadDim;
      const bf16_t* weight = is_q ? params.q_weight : params.k_weight;
      const auto input_pair = load_as<Packed>(input, head_thread);
      const auto [x0, x1] = cast<fp32x2_t>(input_pair);
      float sum_of_squares = fmaf(x0, x0, x1 * x1);

#pragma unroll
      for (uint32_t offset = kWarpThreads / 2; offset > 0; offset /= 2) {
        sum_of_squares += __shfl_xor_sync(0xffffffffu, sum_of_squares, offset);
      }

      if (lane_id == 0) {
        warp_sums[head_group][warp_in_head] = sum_of_squares;
      }
      __syncthreads();

      if (warp_in_head == 0) {
        float head_sum = lane_id < kWarpsPerHead ? warp_sums[head_group][lane_id] : 0.0f;
        head_sum += __shfl_xor_sync(0xffffffffu, head_sum, 2);
        head_sum += __shfl_xor_sync(0xffffffffu, head_sum, 1);
        if (lane_id == 0) {
          warp_sums[head_group][0] = head_sum;
        }
      }
      __syncthreads();

      const float inverse_rms =
          math::rsqrt(fmaf(warp_sums[head_group][0], 1.0f / static_cast<float>(kHeadDim), params.eps));
      Packed normalized_pair;

      if (apply_rope) {
        const auto [w0, w1] = cast<fp32x2_t>(load_as<Packed>(weight, head_thread));
        normalized_pair = cast<Packed, fp32x2_t>({x0 * inverse_rms * w0, x1 * inverse_rms * w1});
      } else {
        normalized_pair = cast<Packed, fp32x2_t>({x0 * inverse_rms, x1 * inverse_rms});
      }

      normalized_pairs[head_group][head_thread] = normalized_pair;
      __syncthreads();

      if (apply_rope) {
        const auto [value0, value1] = cast<fp32x2_t>(normalized_pair);
        const auto [paired0, paired1] =
            cast<fp32x2_t>(normalized_pairs[head_group][head_thread ^ (kThreadsPerHead / 2)]);
        const uint32_t frequency_pair = (head_thread % (kThreadsPerHead / 2)) * 2;
        const float cosine0 = rope_cache[frequency_pair];
        const float cosine1 = rope_cache[frequency_pair + 1];
        const float sine0 = rope_cache[kHalfHeadDim + frequency_pair];
        const float sine1 = rope_cache[kHalfHeadDim + frequency_pair + 1];
        const float rotated0 = head_thread < kThreadsPerHead / 2 ? -paired0 : paired0;
        const float rotated1 = head_thread < kThreadsPerHead / 2 ? -paired1 : paired1;
        normalized_pair =
            cast<Packed, fp32x2_t>({value0 * cosine0 + rotated0 * sine0, value1 * cosine1 + rotated1 * sine1});
      }

      store_as<Packed>(input, normalized_pair, head_thread);
      __syncthreads();
    }

    __syncthreads();
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
  TensorMatcher({num_tokens}).with_strides({1}).with_dtype<PositionType>().with_device(device).verify(positions);

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
  if (token_count == 0) return;

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
  const uint32_t block_count = std::min(blocks_per_sm * sm_count, token_count);
  LaunchKernel(block_count, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
}

}  // namespace
