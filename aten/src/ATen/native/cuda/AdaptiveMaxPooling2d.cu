#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_max_pool2d_backward_native.h>
#include <ATen/ops/adaptive_max_pool2d_native.h>
#include <ATen/ops/empty.h>
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>


namespace at::native {

namespace {

__device__ inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (a / b) * c + ((a % b) * c) / b;
}

__device__ inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return 1 + ((a + 1) * c - 1) / b;
}

// 4d tensor B x D x H x W

/*
 * Description:
 *    this function adaptively maxpools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output, 4D argmax x and y
 */
 template <typename T>
__global__ void adaptivemaxpool(const T *input, T *output, int64_t *indices,
                        int isizeH, int isizeW,
                        int osizeH, int osizeW,
                        int64_t istrideD, int64_t istrideH, int64_t istrideW)
{
  // iterators
  int oh, ow;

  // compute offsets based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  const int ostepW = blockDim.x;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  const int ostepH = blockDim.y*gridDim.y;
  // select input/output plane
  output = output + o_plane*osizeH*osizeW;
  input = input + i_plane*istrideD;
  indices = indices + o_plane*osizeH*osizeW;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    int istartH = start_index(oh, osizeH, isizeH);
    int iendH   = end_index(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for(ow = ostartW; ow < oendW; ow += ostepW) {
      int istartW = start_index(ow, osizeW, isizeW);
      int iendW   = end_index(ow, osizeW, isizeW);

      int kW = iendW - istartW;

      // Compute the mean of the input image...
      const T *ptr_input = input + istartH*istrideH + istartW*istrideW;
      T *ptr_output = output + oh*osizeW + ow;
      int64_t *ptr_ind = indices + oh*osizeW + ow;
      int argmax = istartH * isizeW + istartW;
      T max = at::numeric_limits<T>::lower_bound(); // -Infinity
      int ih, iw;
      for(ih = 0; ih < kH; ih++) {
        for(iw = 0; iw < kW; iw++) {
          T val = ptr_input[iw*istrideW];
          if ((val > max) || at::_isnan(val)) {
            max = val;
            argmax = (ih+istartH)*isizeW + iw+istartW;
          }
        }
        ptr_input += istrideH; // next input line
      }
      // Update output and argmax
      *ptr_output = max;
      *ptr_ind = argmax;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
 template <typename T>
__global__ void adaptivemaxgradinput(T *gradInput, const T *gradOutput, const int64_t *indices,
                             int isizeH, int isizeW,
                             int osizeH, int osizeW)
{
  // iterators
  int oh, ow;

  // compute offsets based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;
  //int k = blockIdx.x % sizeD;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o_plane*osizeH*osizeW;
  gradInput = gradInput + i_plane*isizeH*isizeW;
  indices = indices + o_plane*osizeH*osizeW;

  // compute gradInput
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      const T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
      const int64_t *ptr_ind = indices + oh*osizeW + ow;
      T z = *ptr_gradOutput;

      int argmax = (*ptr_ind);

      gradInput[argmax] += z;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 *    when kH != dH or kW != dW (uses atomic add)
 */
 template <typename T>
__global__ void atomicadaptivemaxgradinput(
  T *gradInput, const T *gradOutput, const int64_t *indices,
  int isizeH, int isizeW, int osizeH, int osizeW
)
{
  // iterators
  int oh, ow;

  // compute offsets based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o_plane*osizeH*osizeW;
  gradInput = gradInput + i_plane*isizeH*isizeW;
  indices = indices + o_plane*osizeH*osizeW;

  // compute gradInput
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      const T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
      const int64_t *ptr_ind = indices + oh*osizeW + ow;
      T z = *ptr_gradOutput;

      int argmax = (*ptr_ind);

      // atomic add since different threads could update same variable
      fastAtomicAdd(gradInput, argmax, isizeH * isizeW, z, true);
    }
  }
}

/*
 * Description:
 *    Deterministic version of atomicadaptivemaxgradinput, used when
 *    torch.use_deterministic_algorithms(True) is set.
 *    Gather-style: one thread per gradInput element (plane, ih, iw). The
 *    output cells whose adaptive window contains ih are exactly
 *    [start_index(ih, isizeH, osizeH), end_index(ih, isizeH, osizeH)) (and the
 *    same for iw), so each thread visits that range in a fixed order and sums
 *    the gradOutput entries whose argmax points at (ih, iw). No atomics are
 *    used, so the result is bitwise reproducible.
 *    gradOutput, indices and gradInput must be contiguous; planes are the
 *    flattened batch and channel dimensions.
 */
template <typename T>
__global__ void deterministicadaptivemaxgradinput(
  T *gradInput, const T *gradOutput, const int64_t *indices,
  int64_t nplanes, int64_t isizeH, int64_t isizeW,
  int64_t osizeH, int64_t osizeW
)
{
  using acc_t = at::acc_type<T, /*is_cuda=*/true>;
  const int64_t isizeHW = isizeH * isizeW;
  const int64_t osizeHW = osizeH * osizeW;
  const int64_t numel = nplanes * isizeHW;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < numel; index += step) {
    const int64_t plane = index / isizeHW;
    const int64_t pos = index - plane * isizeHW;
    const int64_t ih = pos / isizeW;
    const int64_t iw = pos - ih * isizeW;

    const int64_t ohstart = start_index(ih, isizeH, osizeH);
    const int64_t ohend = ::min(end_index(ih, isizeH, osizeH), osizeH);
    const int64_t owstart = start_index(iw, isizeW, osizeW);
    const int64_t owend = ::min(end_index(iw, isizeW, osizeW), osizeW);

    const T *ptr_gradOutput = gradOutput + plane * osizeHW;
    const int64_t *ptr_ind = indices + plane * osizeHW;

    acc_t gradient = acc_t(0);
    for (int64_t oh = ohstart; oh < ohend; ++oh) {
      for (int64_t ow = owstart; ow < owend; ++ow) {
        const int64_t o = oh * osizeW + ow;
        if (ptr_ind[o] == pos) {
          gradient += static_cast<acc_t>(ptr_gradOutput[o]);
        }
      }
    }
    gradInput[index] = static_cast<T>(gradient);
  }
}
} // namespace

// 4d tensor B x D x H x W

TORCH_IMPL_FUNC(adaptive_max_pool2d_out_cuda)
(const Tensor& input,
IntArrayRef output_size,
const Tensor& output,
const Tensor& indices) {
  TensorArg output_arg{output, "output", 1};
  TensorArg indices_arg{indices, "indices", 2};
  TensorArg input_arg{input, "input", 3};

  checkAllSameGPU(
      __func__, {output_arg, indices_arg, input_arg});
  if (input.numel() == 0) {
    return;
  }

  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  const at::Tensor output_c = output.is_contiguous() ? output : at::empty(output.sizes(), output.options());
  const at::Tensor indices_c = indices.is_contiguous() ? indices : at::empty(indices.sizes(), indices.options());

  if (input.ndimension() == 3) {
    int64_t sizeD = input.size(0);
    int64_t isizeH = input.size(1);
    int64_t isizeW = input.size(2);

    int64_t istrideD = input.stride(0);
    int64_t istrideH = input.stride(1);
    int64_t istrideW = input.stride(2);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "adaptive_max_pool2d_cuda", [&] {
          const scalar_t* input_data = input.const_data_ptr<scalar_t>();
          scalar_t* output_data = output_c.mutable_data_ptr<scalar_t>();
          int64_t* indices_data = indices_c.mutable_data_ptr<int64_t>();

          // cuda blocks & threads:
          int blocksH = (int)(16L / sizeD);
          blocksH = blocksH < 1 ? 1 : blocksH;
          dim3 blocks(sizeD, blocksH);
          dim3 threads(32, 8);

          // run maxpool kernel
          adaptivemaxpool<<<
              blocks,
              threads,
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              input_data,
              output_data,
              indices_data,
              isizeH,
              isizeW,
              osizeH,
              osizeW,
              istrideD,
              istrideH,
              istrideW);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  } else {
    Tensor input_ = input.contiguous();
    int64_t sizeB = input_.size(0);
    int64_t sizeD = input_.size(1);
    int64_t isizeH = input_.size(2);
    int64_t isizeW = input_.size(3);

    // In the kernel, the batch and channel dimensions are treated as if they
    // are flattened and istrideD is used as the stride of this flattened dim
    // Handle the edge case where input_.size(1) == 1, where despite passing the
    // contiguity check the stride might not be H * W
    int64_t istrideD = isizeH * isizeW;
    int64_t istrideH = input_.stride(2);
    int64_t istrideW = input_.stride(3);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input_.scalar_type(),
        "adaptive_max_pool2d_cuda",
        [&] {
          const scalar_t* input_data = input_.const_data_ptr<scalar_t>();
          scalar_t* output_data = output_c.mutable_data_ptr<scalar_t>();
          int64_t* indices_data = indices_c.mutable_data_ptr<int64_t>();

          // cuda blocks & threads:
          int blocksH = (int)(16L / sizeD);
          blocksH = blocksH < 1 ? 1 : blocksH;
          dim3 blocks(sizeB * sizeD, blocksH);
          dim3 threads(32, 8);

          // run maxpool kernel
          adaptivemaxpool<<<
              blocks,
              threads,
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              input_data,
              output_data,
              indices_data,
              isizeH,
              isizeW,
              osizeH,
              osizeW,
              istrideD,
              istrideH,
              istrideW);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }

  if (!output.is_contiguous()) {
    output.copy_(output_c);
  }
  if (!indices.is_contiguous()) {
    indices.copy_(indices_c);
  }
}

TORCH_IMPL_FUNC(adaptive_max_pool2d_backward_out_cuda)
(const Tensor& gradOutput,
 const Tensor& input,
 const Tensor& indices,
 const Tensor& gradInput) {
  TensorArg grad_input_arg{gradInput, "gradInput", 1};
  TensorArg grad_output_arg{gradOutput, "gradOutput", 2};
  TensorArg input_arg{input, "input", 3};
  TensorArg indices_arg{indices, "indices", 4};

  checkAllSameGPU(
      __func__,
      {grad_input_arg, grad_output_arg, input_arg, indices_arg});

  if (gradOutput.numel() == 0) {
    return;
  }

  // See Note [Enabling Deterministic Operations]
  // The default kernels below scatter gradOutput into gradInput with atomic
  // adds, whose order (and thus floating point result) varies between runs
  // when adaptive windows overlap. In deterministic mode, gather instead.
  if (globalContext().deterministicAlgorithms()) {
    const at::Tensor gradOutput_ = gradOutput.contiguous();
    const at::Tensor indices_ = indices.contiguous();
    const at::Tensor gradInput_c = gradInput.is_contiguous() ? gradInput : at::empty(gradInput.sizes(), gradInput.options());

    const int64_t isizeH = input.size(-2);
    const int64_t isizeW = input.size(-1);
    const int64_t osizeH = gradOutput_.size(-2);
    const int64_t osizeW = gradOutput_.size(-1);
    const int64_t nplanes = gradOutput_.numel() / (osizeH * osizeW);
    const int64_t numel = nplanes * isizeH * isizeW;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "adaptive_max_pool2d_backward_cuda",
        [&] {
          constexpr int num_threads = 256;
          const int64_t max_blocks = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
          const int64_t num_blocks = std::min(
              at::ceil_div(numel, static_cast<int64_t>(num_threads)), max_blocks);
          deterministicadaptivemaxgradinput<<<
              static_cast<unsigned int>(num_blocks),
              num_threads,
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              gradInput_c.mutable_data_ptr<scalar_t>(),
              gradOutput_.const_data_ptr<scalar_t>(),
              indices_.const_data_ptr<int64_t>(),
              nplanes,
              isizeH,
              isizeW,
              osizeH,
              osizeW);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

    if (!gradInput.is_contiguous()) {
      gradInput.copy_(gradInput_c);
    }
    return;
  }

  bool atomic =
      true; // suboptimal, but without atomic it doesn't pass the tests

  const at::Tensor gradOutput_ = gradOutput.contiguous();
  const at::Tensor indices_ = indices.contiguous();
  const at::Tensor gradInput_c = gradInput.is_contiguous() ? gradInput : at::empty(gradInput.sizes(), gradInput.options());

  if (input.ndimension() == 3) {
    int64_t sizeD = input.size(0);
    int64_t isizeH = input.size(1);
    int64_t isizeW = input.size(2);

    int64_t osizeH = gradOutput_.size(1);
    int64_t osizeW = gradOutput_.size(2);

    // bool atomic = (isizeH%osizeH != 0) || (isizeW%osizeW != 0);

    gradInput_c.zero_();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "adaptive_max_pool2d_backward_cuda",
        [&] {
          scalar_t* gradInput_data = gradInput_c.mutable_data_ptr<scalar_t>();
          const scalar_t* gradOutput_data = gradOutput_.const_data_ptr<scalar_t>();
          const int64_t* indices_data = indices_.const_data_ptr<int64_t>();

          // cuda blocks & threads:
          int blocksH = (int)(16L / sizeD);
          blocksH = blocksH < 1 ? 1 : blocksH;
          dim3 blocks(sizeD, blocksH);
          dim3 threads(32, 8);

          if (atomic) {
            // run updateGradInput kernel, accumulate gradients atomically
            atomicadaptivemaxgradinput<<<
                blocks,
                threads,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                gradInput_data,
                gradOutput_data,
                indices_data,
                isizeH,
                isizeW,
                osizeH,
                osizeW);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            // run updateGradInput kernel
            adaptivemaxgradinput<<<
                blocks,
                threads,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                gradInput_data,
                gradOutput_data,
                indices_data,
                isizeH,
                isizeW,
                osizeH,
                osizeW);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        });
  } else {
    int64_t sizeB = input.size(0);
    int64_t sizeD = input.size(1);
    int64_t isizeH = input.size(2);
    int64_t isizeW = input.size(3);

    int64_t osizeH = gradOutput_.size(2);
    int64_t osizeW = gradOutput_.size(3);

    gradInput_c.zero_();

    // bool atomic = (isizeH%osizeH != 0) || (isizeW%osizeW != 0);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "adaptive_max_pool2d_backward_cuda",
        [&] {
          scalar_t* gradInput_data = gradInput_c.mutable_data_ptr<scalar_t>();
          const scalar_t* gradOutput_data = gradOutput_.const_data_ptr<scalar_t>();
          const int64_t* indices_data = indices_.const_data_ptr<int64_t>();

          // cuda blocks & threads:
          int blocksH = (int)(16L / sizeD);
          blocksH = blocksH < 1 ? 1 : blocksH;
          dim3 blocks(sizeB * sizeD, blocksH);
          dim3 threads(32, 8);

          if (atomic) {
            // run updateGradInput kernel, accumulate gradients atomically
            atomicadaptivemaxgradinput<<<
                blocks,
                threads,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                gradInput_data,
                gradOutput_data,
                indices_data,
                isizeH,
                isizeW,
                osizeH,
                osizeW);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            // run updateGradInput kernel, accumulate gradients atomically
            adaptivemaxgradinput<<<
                blocks,
                threads,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                gradInput_data,
                gradOutput_data,
                indices_data,
                isizeH,
                isizeW,
                osizeH,
                osizeW);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        });
  }

  if (!gradInput.is_contiguous()) {
    gradInput.copy_(gradInput_c);
  }
 }
} // namespace at::native
