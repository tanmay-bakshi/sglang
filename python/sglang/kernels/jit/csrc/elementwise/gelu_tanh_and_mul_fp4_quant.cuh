#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <nv_internal/tensorrt_llm/kernels/quantization.cuh>

#include <cuda_bf16.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

namespace tk = tensorrt_llm::kernels;

struct GeluTanhMulFP4QuantParams {
  const __nv_bfloat16* __restrict__ input;
  const float* __restrict__ global_scale;
  uint8_t* __restrict__ output;
  uint8_t* __restrict__ output_scale;
  uint32_t num_tokens;
  uint32_t hidden_size;
};

template <int kElementsPerThread>
SGL_DEVICE void gelu_tanh_and_mul(
    tk::PackedVec<__nv_bfloat16, kElementsPerThread>& gate,
    const tk::PackedVec<__nv_bfloat16, kElementsPerThread>& up) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;

#pragma unroll
  for (int index = 0; index < kElementsPerThread / 2; ++index) {
    const float2 gate_f32 = __bfloat1622float2(gate.elts[index]);
    const float2 up_f32 = __bfloat1622float2(up.elts[index]);
    const float gate_x_cdf =
        gate_f32.x *
        (0.5f *
         (1.0f + tanhf(kBeta * (gate_f32.x + kAlpha * gate_f32.x * gate_f32.x * gate_f32.x))));
    const float gate_y_cdf =
        gate_f32.y *
        (0.5f *
         (1.0f + tanhf(kBeta * (gate_f32.y + kAlpha * gate_f32.y * gate_f32.y * gate_f32.y))));
    gate.elts[index] = __float22bfloat162_rn(
        make_float2(gate_x_cdf * up_f32.x, gate_y_cdf * up_f32.y));
  }
}

template <bool kUsePDL>
__global__ void gelu_tanh_mul_fp4_quant_kernel(
    const __grid_constant__ GeluTanhMulFP4QuantParams params) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr int kScaleVectorSize = 16;
  constexpr int kElementsPerThread = tk::CVT_FP16_TO_FP4_ELTS_PER_THREAD;
  constexpr int kThreadsPerScale = kScaleVectorSize / kElementsPerThread;
  using InputVector = tk::PackedVec<__nv_bfloat16, kElementsPerThread>;
  using PackedOutput = std::conditional_t<kElementsPerThread == 16, uint64_t, uint32_t>;

  static_assert(kElementsPerThread == 8 || kElementsPerThread == 16);
  static_assert(kScaleVectorSize % kElementsPerThread == 0);

  const uint32_t vectors_per_row = params.hidden_size / kElementsPerThread;
  const uint64_t work_items = static_cast<uint64_t>(params.num_tokens) * vectors_per_row;
  const uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t thread_stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

  device::PDLWaitPrimary<kUsePDL>();
  for (uint64_t work_index = thread_id; work_index < work_items; work_index += thread_stride) {
    const uint32_t row = static_cast<uint32_t>(work_index / vectors_per_row);
    const uint32_t column = static_cast<uint32_t>(work_index % vectors_per_row);
    const uint64_t gate_offset =
        static_cast<uint64_t>(row) * vectors_per_row * 2 + column;

    InputVector gate;
    InputVector up;
    tk::loadPackedVec(
        gate, reinterpret_cast<const InputVector*>(params.input) + gate_offset);
    tk::loadPackedVec(
        up,
        reinterpret_cast<const InputVector*>(params.input) + gate_offset + vectors_per_row);
    gelu_tanh_and_mul(gate, up);

    auto* scale_output = tk::cvt_quant_to_fp4_get_sf_out_offset<
        uint32_t,
        kScaleVectorSize,
        kThreadsPerScale>(
        static_cast<int>(row),
        static_cast<int>(column),
        static_cast<int>(params.hidden_size),
        reinterpret_cast<uint32_t*>(params.output_scale));

    reinterpret_cast<PackedOutput*>(params.output)[work_index] =
        tk::cvt_warp_fp16_to_fp4<
            __nv_bfloat16,
            kScaleVectorSize,
            kElementsPerThread,
            false,
            false>(gate, params.global_scale[0], scale_output);
  }
  device::PDLTriggerSecondary<kUsePDL>();
#endif
}

template <bool kUsePDL>
struct GeluTanhMulFP4QuantKernel {
  static constexpr uint32_t kBlockSize = 512;

  static void run(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView global_scale,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale) {
    using namespace host;

    auto num_tokens = SymbolicSize{"num_tokens"};
    auto input_width = SymbolicSize{"input_width"};
    auto output_width = SymbolicSize{"output_width"};
    auto scale_rows = SymbolicSize{"scale_rows"};
    auto scale_columns = SymbolicSize{"scale_columns"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({num_tokens, input_width})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input);
    TensorMatcher({num_tokens, output_width})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(output);
    TensorMatcher({scale_rows, scale_columns})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(output_scale);

    RuntimeCheck(global_scale.numel() == 1, "global_scale must contain one element");
    RuntimeCheck(is_type<float>(global_scale.dtype()), "global_scale must have dtype float32");
    RuntimeCheck(global_scale.device().device_type == kDLCUDA, "global_scale must be on CUDA");
    RuntimeCheck(
        global_scale.device().device_id == device.unwrap().device_id,
        "all tensors must be on the same CUDA device");

    const uint64_t tokens = static_cast<uint64_t>(num_tokens.unwrap());
    const uint64_t fused_width = static_cast<uint64_t>(input_width.unwrap());
    RuntimeCheck(tokens > 0, "num_tokens must be positive");
    RuntimeCheck(fused_width % 2 == 0, "input width must contain equal gate and up halves");
    const uint64_t hidden_size = fused_width / 2;
    RuntimeCheck(hidden_size % 16 == 0, "hidden size must be divisible by 16");
    RuntimeCheck(
        static_cast<uint64_t>(output_width.unwrap()) * 2 == hidden_size,
        "packed FP4 output width must equal hidden_size / 2");

    const uint64_t expected_scale_rows = div_ceil(tokens, uint64_t{128}) * 128;
    const uint64_t scale_vectors = hidden_size / 16;
    const uint64_t expected_scale_columns = div_ceil(scale_vectors, uint64_t{4}) * 4;
    RuntimeCheck(
        static_cast<uint64_t>(scale_rows.unwrap()) == expected_scale_rows,
        "scale row count must be padded to 128");
    RuntimeCheck(
        static_cast<uint64_t>(scale_columns.unwrap()) == expected_scale_columns,
        "scale column count must be padded to 4");

    constexpr uint64_t kElementsPerThread = tk::CVT_FP16_TO_FP4_ELTS_PER_THREAD;
    RuntimeCheck(
        hidden_size % (kElementsPerThread * 32) == 0,
        "hidden size must keep every FP4 scale group inside one warp");
    const uint64_t total_work = tokens * (hidden_size / kElementsPerThread);
    RuntimeCheck(
        total_work <= std::numeric_limits<uint32_t>::max(),
        "fused GeGLU FP4 quantization exceeds 32-bit launch indexing");

    const auto kernel = gelu_tanh_mul_fp4_quant_kernel<kUsePDL>;
    const uint32_t blocks_per_sm = host::runtime::get_blocks_per_sm(kernel, kBlockSize);
    const uint32_t max_blocks =
        host::runtime::get_sm_count(device.unwrap().device_id) * blocks_per_sm;
    const uint32_t grid_size = std::min(
        div_ceil(static_cast<uint32_t>(total_work), kBlockSize), max_blocks);
    const auto params = GeluTanhMulFP4QuantParams{
        .input = static_cast<const __nv_bfloat16*>(input.data_ptr()),
        .global_scale = static_cast<const float*>(global_scale.data_ptr()),
        .output = static_cast<uint8_t*>(output.data_ptr()),
        .output_scale = static_cast<uint8_t*>(output_scale.data_ptr()),
        .num_tokens = static_cast<uint32_t>(tokens),
        .hidden_size = static_cast<uint32_t>(hidden_size),
    };
    LaunchKernel(grid_size, kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
