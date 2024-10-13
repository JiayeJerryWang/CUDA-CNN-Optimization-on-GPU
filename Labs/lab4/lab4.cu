#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 8
#define TILE_SIZE TILE_WIDTH + 2
//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];
// Implement a 3D convolution with a 3x3x3 kernel in constant memory and a 3D shared-memory tiling
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float tile[TILE_SIZE][TILE_SIZE][TILE_SIZE];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int col_o = bx * TILE_WIDTH + tx;
  int row_o = by * TILE_WIDTH + ty;
  int height_o = bz * TILE_WIDTH + tz;
  int col_i = col_o - 1;
  int row_i = row_o - 1;
  int height_i = height_o - 1;

  float Pvalue = 0.0f;
  if ((row_i >= 0) && (row_i < y_size) &&
      (col_i >= 0) && (col_i < x_size) &&
      (height_i >= 0) && (height_i < z_size)) {
    tile[tz][ty][tx] = input[height_i * y_size * x_size + row_i * x_size + col_i];
  } else {
    tile[tz][ty][tx] = 0.0f;
  }
  __syncthreads (); // wait for tile
  if (ty < TILE_WIDTH && tx < TILE_WIDTH && tz < TILE_WIDTH) {
    for(int i = 0; i < MASK_WIDTH; i++) {
      for(int j = 0; j < MASK_WIDTH; j++) {
        for(int k = 0; k < MASK_WIDTH; k++) {
          Pvalue += Mc[i][j][k] * tile[i+tz][j+ty][k+tx];
        }
      }
    }
    if (height_o < z_size && row_o < y_size && col_o < x_size) {
      output[height_o * y_size * x_size + row_o * x_size + col_o] = Pvalue;
    }
  } 
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;
  //@@ Initial deviceInput and deviceOutput here.

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength - 3) * sizeof(float));


  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength * sizeof(float));


  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(((float)x_size) / TILE_WIDTH), ceil(((float)y_size) / TILE_WIDTH), ceil(((float)z_size) / TILE_WIDTH));
  dim3 DimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

