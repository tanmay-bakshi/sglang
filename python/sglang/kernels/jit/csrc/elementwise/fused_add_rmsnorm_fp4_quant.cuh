#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cooperative_groups/reduce.h>
#include <nv_internal/tensorrt_llm/kernels/quantization_utils.cuh>
#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_bf16.h>

namespace {

namespace cg = cooperative_groups;
namespace tk = tensorrt_llm::kernels;

constexpr uint32_t kElementsPerVector = 16;
constexpr uint32_t kPackedPairsPerVector = kElementsPerVector / 2;

struct FusedAddRMSNormFP4QuantParams {
  const __nv_bfloat16* __restrict__ input;
  __nv_bfloat16* __restrict__ residual;
  const __nv_bfloat16* __restrict__ weight;
  const float* __restrict__ global_scale;
  uint8_t* __restrict__ output;
  uint8_t* __restrict__ output_scale;
  uint32_t num_tokens;
  uint32_t hidden_size;
  float epsilon;
};

SGL_DEVICE uint64_t
fused_rmsnorm_fp4_scale_offset(const uint32_t row, const uint32_t scale_column, const uint32_t padded_scale_columns) {
  constexpr uint32_t kColumnsPerGroup = 4;
  constexpr uint32_t kRowsPerInnerGroup = 32;
  constexpr uint32_t kRowsPerGroup = 128;

  const uint32_t column_in_group = scale_column % kColumnsPerGroup;
  const uint32_t column_group = scale_column / kColumnsPerGroup;
  const uint32_t row_in_inner_group = row % kRowsPerInnerGroup;
  const uint32_t inner_row_group = (row % kRowsPerGroup) / kRowsPerInnerGroup;
  const uint32_t row_group = row / kRowsPerGroup;

  return static_cast<uint64_t>(column_in_group) + static_cast<uint64_t>(column_group) * 512 +
         static_cast<uint64_t>(row_in_inner_group) * 16 + static_cast<uint64_t>(inner_row_group) * 4 +
         static_cast<uint64_t>(row_group) * kRowsPerGroup * padded_scale_columns;
}

template <bool kUsePDL>
__global__ void fused_add_rmsnorm_fp4_quant_kernel(const __grid_constant__ FusedAddRMSNormFP4QuantParams params) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using NormVector = device::AlignedVector<bf16x2_t, kPackedPairsPerVector>;
  using QuantVector = tk::PackedVec<__nv_bfloat16, kElementsPerVector>;

  __shared__ float reduction_buffer[32];

  const uint32_t token_id = blockIdx.x;
  const uint32_t vector_column = threadIdx.x;
  const uint32_t vectors_per_row = params.hidden_size / kElementsPerVector;
  const uint32_t scale_columns = vectors_per_row;
  const uint32_t padded_scale_columns = (scale_columns + 3) / 4 * 4;
  const uint32_t padded_scale_rows = (params.num_tokens + 127) / 128 * 128;
  const bool is_active = vector_column < vectors_per_row;

  NormVector summed;
  float2 square_sum = make_float2(0.0f, 0.0f);

  device::PDLWaitPrimary<kUsePDL>();
  if (is_active) {
    const uint64_t vector_offset = static_cast<uint64_t>(token_id) * vectors_per_row + vector_column;
    const NormVector input = reinterpret_cast<const NormVector*>(params.input)[vector_offset];
    const NormVector residual = reinterpret_cast<const NormVector*>(params.residual)[vector_offset];

#pragma unroll
    for (uint32_t index = 0; index < kPackedPairsPerVector; ++index) {
      const float2 input_f32 = device::cast<fp32x2_t>(input[index]);
      const float2 residual_f32 = device::cast<fp32x2_t>(residual[index]);
      const float2 sum_f32 = make_float2(input_f32.x + residual_f32.x, input_f32.y + residual_f32.y);
      square_sum.x += sum_f32.x * sum_f32.x;
      square_sum.y += sum_f32.y * sum_f32.y;
      summed[index] = device::cast<bf16x2_t>(sum_f32);
    }
    reinterpret_cast<NormVector*>(params.residual)[vector_offset] = summed;
  }

  const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  const float warp_sum = cg::reduce(warp, square_sum.x + square_sum.y, cg::plus<float>());
  if (threadIdx.x % 32 == 0) {
    reduction_buffer[threadIdx.x / 32] = warp_sum;
  }

  __syncthreads();
  if (threadIdx.x < 32) {
    const float block_sum =
        cg::reduce(warp, threadIdx.x < blockDim.x / 32 ? reduction_buffer[threadIdx.x] : 0.0f, cg::plus<float>());
    reduction_buffer[threadIdx.x] =
        rsqrtf(params.epsilon + block_sum * (1.0f / static_cast<float>(params.hidden_size)));
  }
  __syncthreads();

  if (is_active) {
    const float inverse_rms = reduction_buffer[threadIdx.x / 32];
    const NormVector weight = reinterpret_cast<const NormVector*>(params.weight)[vector_column];
    QuantVector normalized;
#pragma unroll
    for (uint32_t index = 0; index < kPackedPairsPerVector; ++index) {
      const float2 value_f32 = device::cast<fp32x2_t>(summed[index]);
      const float2 weight_f32 = device::cast<fp32x2_t>(weight[index]);
      normalized.elts[index] = device::cast<bf16x2_t>(
          make_float2(value_f32.x * weight_f32.x * inverse_rms, value_f32.y * weight_f32.y * inverse_rms));
    }

    const uint64_t output_offset = static_cast<uint64_t>(token_id) * vectors_per_row + vector_column;
    uint8_t* scale_output =
        params.output_scale + fused_rmsnorm_fp4_scale_offset(token_id, vector_column, padded_scale_columns);
    reinterpret_cast<uint64_t*>(params.output)[output_offset] =
        tk::cvt_warp_fp16_to_fp4<__nv_bfloat16, kElementsPerVector, kElementsPerVector, false, false>(
            normalized, params.global_scale[0], scale_output);
  }

  for (uint32_t scale_column = scale_columns + threadIdx.x; scale_column < padded_scale_columns;
       scale_column += blockDim.x) {
    params.output_scale[fused_rmsnorm_fp4_scale_offset(token_id, scale_column, padded_scale_columns)] = 0;
  }

  if (blockIdx.x == 0) {
    const uint64_t padding_items = static_cast<uint64_t>(padded_scale_rows - params.num_tokens) * padded_scale_columns;
    for (uint64_t padding_index = threadIdx.x; padding_index < padding_items; padding_index += blockDim.x) {
      const uint32_t row = params.num_tokens + static_cast<uint32_t>(padding_index / padded_scale_columns);
      const uint32_t scale_column = static_cast<uint32_t>(padding_index % padded_scale_columns);
      params.output_scale[fused_rmsnorm_fp4_scale_offset(row, scale_column, padded_scale_columns)] = 0;
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
#endif
}

template <bool kUsePDL>
struct FusedAddRMSNormFP4QuantKernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView global_scale,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      float epsilon) {
    using namespace host;

    auto num_tokens = SymbolicSize{"num_tokens"};
    auto hidden_size = SymbolicSize{"hidden_size"};
    auto output_width = SymbolicSize{"output_width"};
    auto scale_rows = SymbolicSize{"scale_rows"};
    auto scale_columns = SymbolicSize{"scale_columns"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({num_tokens, hidden_size})
        .with_strides({hidden_size, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input)
        .verify(residual);
    TensorMatcher({hidden_size}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({num_tokens, output_width})
        .with_strides({output_width, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(output);
    TensorMatcher({scale_rows, scale_columns})
        .with_strides({scale_columns, 1})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(output_scale);

    RuntimeCheck(global_scale.numel() == 1, "global_scale must contain one element");
    RuntimeCheck(is_type<float>(global_scale.dtype()), "global_scale must have dtype float32");
    RuntimeCheck(global_scale.device().device_type == kDLCUDA, "global_scale must be on CUDA");
    RuntimeCheck(
        global_scale.device().device_id == device.unwrap().device_id, "all tensors must be on the same CUDA device");

    const uint64_t tokens = static_cast<uint64_t>(num_tokens.unwrap());
    const uint64_t hidden = static_cast<uint64_t>(hidden_size.unwrap());
    RuntimeCheck(tokens > 0, "num_tokens must be positive");
    RuntimeCheck(hidden > 0, "hidden_size must be positive");
    RuntimeCheck(hidden % kElementsPerVector == 0, "hidden_size must be divisible by 16");
    RuntimeCheck(hidden <= 16384, "hidden_size must not exceed one 1024-thread block");
    RuntimeCheck(
        static_cast<uint64_t>(output_width.unwrap()) * 2 == hidden,
        "packed FP4 output width must equal hidden_size / 2");

    const uint64_t expected_scale_rows = div_ceil(tokens, uint64_t{128}) * 128;
    const uint64_t expected_scale_columns = div_ceil(hidden / kElementsPerVector, uint64_t{4}) * 4;
    RuntimeCheck(
        static_cast<uint64_t>(scale_rows.unwrap()) == expected_scale_rows, "scale row count must be padded to 128");
    RuntimeCheck(
        static_cast<uint64_t>(scale_columns.unwrap()) == expected_scale_columns,
        "scale column count must be padded to 4");

    const uint32_t vectors_per_row = static_cast<uint32_t>(hidden / kElementsPerVector);
    const uint32_t threads = div_ceil(vectors_per_row, uint32_t{32}) * 32;
    const auto params = FusedAddRMSNormFP4QuantParams{
        .input = static_cast<const __nv_bfloat16*>(input.data_ptr()),
        .residual = static_cast<__nv_bfloat16*>(residual.data_ptr()),
        .weight = static_cast<const __nv_bfloat16*>(weight.data_ptr()),
        .global_scale = static_cast<const float*>(global_scale.data_ptr()),
        .output = static_cast<uint8_t*>(output.data_ptr()),
        .output_scale = static_cast<uint8_t*>(output_scale.data_ptr()),
        .num_tokens = static_cast<uint32_t>(tokens),
        .hidden_size = static_cast<uint32_t>(hidden),
        .epsilon = epsilon,
    };
    LaunchKernel(static_cast<uint32_t>(tokens), threads, device.unwrap())
        .enable_pdl(kUsePDL)(fused_add_rmsnorm_fp4_quant_kernel<kUsePDL>, params);
  }
};

}  // namespace
