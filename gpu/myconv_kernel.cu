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
    extern __shared__ float shmem[];

    int col = blockIdx.x * TILE_W + threadIdx.x;
    int row = blockIdx.y * TILE_H + threadIdx.y;
    int n = blockIdx.z;

    int tile_h = (TILE_H - 1) * stride + KH;
    int tile_w = (TILE_W - 1) * stride + KW;
    int tile_size = tile_h * tile_w;

    int tid = threadIdx.y * TILE_W + threadIdx.x;
    int num_threads = TILE_H * TILE_W;

    int row_origin = blockIdx.y * TILE_H * stride - pad;
    int col_origin = blockIdx.x * TILE_W * stride - pad;

    for (int oc = 0; oc < C_out; ++oc) {
        float sum = 0.0f;
        int num_tiles = (C_in + TILE_C - 1) / TILE_C;

        for (int t = 0; t < num_tiles; t++) {
            int c0 = t * TILE_C;
            int c_count = TILE_C;
            if (c_count > C_in - c0) {
                c_count = C_in - c0;
            }

            int load_count = c_count * tile_size;
            for (int idx = tid; idx < load_count; idx += num_threads) {
                int c_local = idx / tile_size;
                int spatial = idx % tile_size;
                int tile_row = spatial / tile_w;
                int tile_col = spatial % tile_w;
                int x_row = row_origin + tile_row;
                int x_col = col_origin + tile_col;

                if (x_row >= 0 && x_row < H && x_col >= 0 && x_col < W) {
                    shmem[idx] = x[((n * C_in + (c0 + c_local)) * H + x_row) * W + x_col];
                } else {
                    shmem[idx] = 0.0f;
                }
            }
            __syncthreads();

            if (row < out_h && col < out_w) {
                for (int c_local = 0; c_local < c_count; ++c_local) {
                    int c = c0 + c_local;
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int tile_row = threadIdx.y * stride + kh;
                            int tile_col = threadIdx.x * stride + kw;
                            float x_val = shmem[c_local * tile_size + tile_row * tile_w + tile_col];
                            float w_val = w[(((oc * C_in + c) * KH + kh) * KW) + kw];
                            sum += x_val * w_val;
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (row < out_h && col < out_w) {
            out[((n * C_out + oc) * out_h + row) * out_w + col] = sum;
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

    int tile_h = (TILE_H - 1) * stride + KH;
    int tile_w = (TILE_W - 1) * stride + KW;
    size_t smem = (size_t)TILE_C * tile_h * tile_w * sizeof(float);

    gemm_gpu_o4_kernel<<<grid, block, smem>>>(
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
