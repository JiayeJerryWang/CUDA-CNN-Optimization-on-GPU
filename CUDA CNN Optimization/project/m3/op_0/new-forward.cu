#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_grid = ceil((float) Width_out / TILE_WIDTH);
    int H_grid = ceil((float) Height_out / TILE_WIDTH);

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
    for (int c = 0; c < Channel; c++) { 
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
            }
        }
        __syncthreads();
    }
    if (b < Batch && m < Map_out && h < Height_out && w < Width_out) {
        out_4d(b, m, h, w) = acc;
    }
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int num_seg = 10;
    cudaStream_t stream[num_seg];
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaHostRegister((void *)host_input, (Batch * Channel * Height * Width * sizeof(float)), cudaHostRegisterDefault);
    cudaHostRegister((void *)host_output, (Batch * Map_out * Height_out * Width_out * sizeof(float)), cudaHostRegisterDefault);
    cudaMalloc((void **)device_input_ptr, (Batch * Channel * Height * Width * sizeof(float)));
    cudaMalloc((void **)device_output_ptr, (Batch * Map_out * Height_out * Width_out * sizeof(float)));
    cudaMalloc((void **)device_mask_ptr, (Channel * Map_out * K * K * sizeof(float)));
    int W_grid = ceil((float) Width_out / TILE_WIDTH);
    int H_grid = ceil((float) Height_out / TILE_WIDTH);
    int Y = H_grid * W_grid;
    int seg_in = Batch * Channel * Height * Width;
    int seg_out = Batch * Map_out * Height_out * Width_out;
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dim_grid(Map_out, Y, Batch / num_seg);
    for (int i = 0; i < num_seg; ++i) {
        cudaStreamCreate(&stream[i]);
    }
    cudaMemcpyAsync(*device_mask_ptr, host_mask, Channel * Map_out * K * K * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    for (int i = 0; i < num_seg; i++) {
        cudaMemcpyAsync((*device_input_ptr) + ((seg_in / num_seg) * i), host_input + ((seg_in / num_seg) * i), (seg_in / num_seg) * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        conv_forward_kernel<<<dim_grid, dimBlock, 0, stream[i]>>>((*device_output_ptr) + ((seg_out / num_seg) * i), (*device_input_ptr) + ((seg_in / num_seg) * i), 
            *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        cudaMemcpyAsync((float *)host_output + ((seg_out / num_seg) * i), (*device_output_ptr) + ((seg_out / num_seg) * i), (seg_out / num_seg) * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    for(int j = 0; j < 10; j++) {
        cudaStreamDestroy(stream[j]);
    }
    cudaFree(device_input_ptr);
    cudaFree(device_output_ptr);
    cudaFree(device_mask_ptr);
    cudaHostRegister((void *)host_input);
    cudaHostRegister((void *)host_output);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
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