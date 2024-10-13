#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
#define BLOCK_SIZE 32

__global__ void unroll_Kernel(int C, int H, int W, int K, const float* X, float* X_unroll)
{
    int t = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;
    if (t < C * W_unroll) {
        int c = t / W_unroll;
        int s = t % W_unroll;
        int h_out = s / W_out;
        int w_out = s % W_out;
        int h_unroll = h_out * W_out + w_out;
        int w_base = c * K * K;
        for(int p = 0; p < K; p++) {
            for(int q = 0; q < K; q++) {
                int w_unroll = w_base + p * K + q;
                X_unroll[w_unroll * W_unroll + h_unroll] = X[(c) * (H * W) + (h_out + p) * (W) + w_out + q];
            }
        }
    }
}

__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;
    for (int m = 0; m < ((numAColumns - 1)/TILE_WIDTH + 1); ++m) {
        if ((Row < numARows) && (m*TILE_WIDTH+tx < numAColumns)) {
            subTileM[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
        } else {
            subTileM[ty][tx] = 0;
        }
        if ((m*TILE_WIDTH+ty < numBRows) && (Col < numBColumns)) {
            subTileN[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
        } else {
            subTileN[ty][tx] = 0;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += subTileM[ty][k] * subTileN[k][tx];
        }
        __syncthreads();
        }
        if ((Row < numCRows) && (Col < numCColumns)) {
        C[Row*numCColumns+Col] = Pvalue;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **) device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Channel * Map_out * K * K * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Channel * Map_out * K * K * sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    float *X_unroll;
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;
    int H_unroll = Channel * K * K;
    int W_unroll = H_out * W_out;
    cudaMalloc((void**) &X_unroll, H_unroll * W_unroll * sizeof(float));
    int num_blocks = ceil(((float) Channel * H_out * W_out) / BLOCK_SIZE);
    dim3 blockDim (TILE_WIDTH, TILE_WIDTH , 1);
    dim3 gridDim (ceil((float) W_unroll / TILE_WIDTH), ceil((float) Map_out / TILE_WIDTH), 1);
    for (int b = 0; b < Batch; b++) {
        unroll_Kernel<<<num_blocks, BLOCK_SIZE>>>(Channel, Height, Width, K, device_input, X_unroll);
        device_input += Channel * Height * Width;
        matrixMultiplyShared<<<gridDim, blockDim>>>(device_mask, X_unroll, device_output, Map_out, H_unroll, H_unroll, W_unroll, Map_out, W_unroll);
        device_output += Map_out * W_unroll;
    }
    cudaFree(X_unroll);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}