// LAB 2 SP24

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  int Row = blockIdx.x * blockDim.x + threadIdx.x;
  int Col = blockIdx.y * blockDim.y + threadIdx.y;
  if ((Row < numCRows) && (Col < numCColumns)) {
    float Pvalue = 0;
    for (int k = 0; k < numAColumns; ++k) {
      Pvalue += A[Row * numAColumns + k] * B[k * numBColumns + Col];
    }
    C[Row * numCColumns + Col] = Pvalue;
  } 
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *A_d;
  float *B_d;
  float *C_d;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *) malloc(numCColumns * numCRows * sizeof(float));

  //@@ Allocate GPU memory here
  cudaMalloc((void **) &A_d, numAColumns * numARows * sizeof(float));
  cudaMalloc((void **) &B_d, numBColumns * numBRows * sizeof(float));
  cudaMalloc((void **) &C_d, numCColumns * numCRows * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(A_d, hostA, numAColumns * numARows * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, hostB, numBColumns * numBRows * sizeof(float), cudaMemcpyHostToDevice); 

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil((float) numCRows / 16), ceil((float) numCColumns / 16), 1);
  dim3 DimBlock(16, 16, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid, DimBlock>>>(A_d, B_d, C_d, numARows,
                               numAColumns, numBRows,
                               numBColumns, numCRows,
                               numCColumns);
  cudaDeviceSynchronize();
  
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, C_d, numCColumns * numCRows * sizeof(float), cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(A_d); 
  cudaFree(B_d); 
  cudaFree(C_d);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  //@@Free the hostC matrix
  free(hostC);
  return 0;
}

