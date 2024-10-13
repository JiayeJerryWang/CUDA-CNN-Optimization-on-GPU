// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, float *sum) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  int index = 2*blockIdx.x*blockDim.x + threadIdx.x;
  if (index < len) {
    T[threadIdx.x] = input[index];
  } 
  if (index+blockDim.x < len) {
    T[threadIdx.x+blockDim.x] = input[index+blockDim.x];
  } 

  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0) {
      T[index] += T[index-stride];
    }
    stride = stride*2;
  }

  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE) {
      T[index+stride] += T[index];
    }
    stride = stride / 2;
  }

  __syncthreads();
  if (threadIdx.x == BLOCK_SIZE-1) {
    sum[blockIdx.x] = T[2*BLOCK_SIZE-1];
  }
  
  __syncthreads();
  if(index < len) {
    output[index] = T[threadIdx.x];
  }
  if (index+blockDim.x <len) {
    output[index+blockDim.x] = T[threadIdx.x + blockDim.x];
  }
}

__global__ void AddSum(float *sum, float *output, int len) {
  if(len > 2*blockIdx.x*blockDim.x + threadIdx.x){
    if (blockIdx.x > 0 ) {
      output[(2*blockIdx.x*blockDim.x) + threadIdx.x] += sum[blockIdx.x - 1];
    }
  }
  if(len > 2*blockIdx.x*blockDim.x + threadIdx.x + blockDim.x){
    if (blockIdx.x > 0 ) {
      output[(2*blockIdx.x*blockDim.x) + threadIdx.x + blockDim.x] += sum[blockIdx.x - 1];
    } 
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceSum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceSum, ceil(numElements/(2.0*BLOCK_SIZE)) * sizeof(float)));

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements/(2.0*BLOCK_SIZE)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, deviceSum);
  scan<<<1, dimBlock>>>(deviceSum, deviceSum, ceil(numElements/(2.0*BLOCK_SIZE)), deviceInput);
  AddSum<<<dimGrid, dimBlock>>>(deviceSum, deviceOutput, numElements);
  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

