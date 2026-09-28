#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// example
#define TILE_H 8   
#define TILE_W 8   
#define TILE_C 16  

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float* __restrict__ x,       // input: N x C x H x W
    const float* __restrict__ w,       // weights: C_out x C_in x KH x KW
    float* __restrict__ out,           // output: N x C x H x W
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w
) {
    extern __shared__ float shmem[];  // shared memory for partial sums
    
    float* sh_x = shmem;
		float* sh_w = shmem + TILE_H * TILE_C;

		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;

		int n = blockIdx.z;  // batch index

		int M = out_h * out_w;
		int m = blockIdx.x * TILE_H + ty;  
		int co = blockIdx.y * TILE_W + tx;
		int K = C_in * KH * KW;
		float sum = 0.0f;

		for (int k0 = 0; k0 < K; k0 += TILE_C) {
			  int A_tile_size = TILE_H * TILE_C;
				for (int idx = tid; idx < A_tile_size; idx += num_threads) {
					  int local_row = idx / TILE_C;
						int local_col = idx % TILE_C;
						int global_row = blockIdx.x * TILE_H + local_row;
						int global_col = k0 + local_col;
						float val = 0.0f;
						if (global_row < M && global_col < K) {
							  int oh = global_row / out_w;
								int ow = global_row % out_w;
								int kernel_size = KH * KW;
								int c_in = global_col / kernel_size;
								int kernel_offset = global_col % kernel_size;
								int kh = kernel_offset / KW;
								int kw = kernel_offset % KW;
								int ih = oh * stride - pad + kh;
								int iw = ow * stride - pad + kw;
								if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
									  int x_idx = n * (C_in * H * W) + c_in * (H * W) + ih * W + iw;
										val = x[x_idx];
								}
						}
						sh_x[idx] = val;
				}

				int B_tile_size = TILE_C * TILE_W;
				for (idx = tid; idx < B_tile_size; idx += num_threads) {
					  int local_row = idx / TILE_W;
						int local_col = idx % TILE_W;
						int global_row = k0 + local_row;
						int global_col = blockIdx.y * TILE_W + local_col;
						float val = 0.0f;
						if (global_row < K && global_col < C_out) {
								int w_idx = global_col * K + global_row;
								val = w[w_idx];
						}
						sh_w[idx] = val;
				}

				__syncthreads();

				if (m < M && co < C_out) {
					  for (int k = 0; k < TILE_C; k++) {
							  sum += sh_x[ty * TILE_C + k] * sh_w[k * TILE_W + tx];
						}
				}

				__syncthreads();

				if (m < M && co < C_out) {
					  int out_idx = n * (C_out * M) + co * M + m;
						out[out_idx] = sum;
				}
		}
}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w,
                          int stride, int pad) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = w.size(0);
    int KH = w.size(2);
    int KW = w.size(3);

    int out_h = (H + 2 * pad - KH) / stride + 1;
    int out_w = (W + 2 * pad - KW) / stride + 1;

    auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());

    dim3 block(8, 8);
    dim3 grid((out_w + block.x - 1)/block.x,
              (out_h + block.y - 1)/block.y,
              N);

    gemm_gpu_o4_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, KH, KW,
        stride, pad,
        out_h, out_w);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}
