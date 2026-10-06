#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

// Implicit im2col: A[M,K] times B[K,C_out], without writing A to global memory.
#define TILE_H 8
#define TILE_W 8
#define TILE_C 16

__global__ void gemm_gpu_o4_kernel(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ out, int C_in, int H, int W,
    int C_out, int KH, int KW, int stride, int pad,
    int out_h, int out_w) {
    extern __shared__ float shmem[];
    float* sh_x = shmem;
    float* sh_w = shmem + TILE_H * TILE_C;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;
    int n = blockIdx.z;
    int M = out_h * out_w;
    int m = blockIdx.x * TILE_H + ty;
    int co = blockIdx.y * TILE_W + tx;
    int K = C_in * KH * KW;
    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_C) {
        for (int idx = tid; idx < TILE_H * TILE_C; idx += num_threads) {
            int row = blockIdx.x * TILE_H + idx / TILE_C;
            int k = k0 + idx % TILE_C;
            float val = 0.0f;
            if (row < M && k < K) {
                int oh = row / out_w, ow = row % out_w;
                int ci = k / (KH * KW);
                int kh = (k % (KH * KW)) / KW, kw = k % KW;
                int ih = oh * stride - pad + kh, iw = ow * stride - pad + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    val = x[((n * C_in + ci) * H + ih) * W + iw];
            }
            sh_x[idx] = val;
        }
        for (int idx = tid; idx < TILE_C * TILE_W; idx += num_threads) {
            int k = k0 + idx / TILE_W;
            int channel = blockIdx.y * TILE_W + idx % TILE_W;
            sh_w[idx] = (k < K && channel < C_out) ? w[channel * K + k] : 0.0f;
        }
        __syncthreads();
        // Boundary threads must also reach both barriers.
        for (int k = 0; k < TILE_C; ++k)
            sum += sh_x[ty * TILE_C + k] * sh_w[k * TILE_W + tx];
        __syncthreads();
    }
    if (m < M && co < C_out)
        out[(n * C_out + co) * M + m] = sum;
}

torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w, int stride, int pad) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "input and weight must be CUDA tensors");
    TORCH_CHECK(x.device() == w.device(), "input and weight must use the same device");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && w.scalar_type() == torch::kFloat32,
                "only float32 is supported");
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4, "expected NCHW input and OIHW weight");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(x.size(1) == w.size(1), "input channels do not match");
    TORCH_CHECK(stride > 0 && pad >= 0, "invalid stride or padding");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && w.size(0) > 0 &&
                w.size(2) > 0 && w.size(3) > 0, "dimensions must be positive");
    TORCH_CHECK(x.size(2) + 2 * pad >= w.size(2) && x.size(3) + 2 * pad >= w.size(3),
                "filter is larger than padded input");
    c10::cuda::CUDAGuard device_guard(x.device());
    int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    int C_out = w.size(0), KH = w.size(2), KW = w.size(3);
    int out_h = (H + 2 * pad - KH) / stride + 1;
    int out_w = (W + 2 * pad - KW) / stride + 1;
    // Every valid output is written once; no separate zero-initialization kernel.
    auto out = torch::empty({N, C_out, out_h, out_w}, x.options());
    dim3 block(TILE_W, TILE_H);
    dim3 grid((out_h * out_w + TILE_H - 1) / TILE_H,
              (C_out + TILE_W - 1) / TILE_W, N);
    size_t shared_bytes = (TILE_H * TILE_C + TILE_C * TILE_W) * sizeof(float);
    gemm_gpu_o4_kernel<<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(),
        C_in, H, W, C_out, KH, KW, stride, pad, out_h, out_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_cuda", &conv_cuda, "Tiled Conv2D (CUDA, no bias)");
}
