// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
//@@ insert code here

// Cast the image from float to unsigned char
// Implement a kernel that casts the image from float * to unsigned char *.

// for ii from 0 to (width * height * channels) do
// 	ucharImage[ii] = (unsigned char) (255 * inputImage[ii])
// end
__global__ void float_to_char(float* inputImage, unsigned char* ucharImage, int width, int height, int channels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width * height * channels) {
    ucharImage[i] = (unsigned char) (255 * inputImage[i]);
  }
}

// Convert the image from RGB to GrayScale
// Implement a kernel that converts the RGB image to GrayScale. A sample sequential pseudo code is shown below. You will find one the lectures and one of the textbook chapters helpful.

// for ii from 0 to height do
// 	for jj from 0 to width do
// 		idx = ii * width + jj
// 		# here channels is 3
// 		r = ucharImage[3*idx]
// 		g = ucharImage[3*idx + 1]
// 		b = ucharImage[3*idx + 2]
// 		grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b)
// 	end
// end
__global__ void rgb_to_grayscale(unsigned char* ucharImage, unsigned char* grayImage, int width, int height) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width * height) {
    unsigned char r = ucharImage[3 * i];
    unsigned char g = ucharImage[3 * i + 1];
    unsigned char b = ucharImage[3 * i + 2];
    grayImage[i] = (unsigned char) (0.21f * r + 0.71f * g + 0.07f * b);
  }
}

// Compute the histogram of grayImage
// Implement a kernel that computes the histogram (like in the lectures) of the image. A sample pseudo code is shown below. You will find one of the lectures and one of the textbook chapters helpful.

// histogram = [0, ...., 0] # here len(histogram) = 256
// for ii from 0 to width * height do
// 	histogram[grayImage[ii]]++
// end
__global__ void histo_kernel(unsigned char *buffer, unsigned int *histo, int size) {
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
  if (threadIdx.x < HISTOGRAM_LENGTH) {
    histo_private[threadIdx.x] = 0;
  }
  __syncthreads();
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while (i < size) {
    atomicAdd(&(histo_private[buffer[i]]), 1);
    i += stride;
  }
  __syncthreads();
  if (threadIdx.x < HISTOGRAM_LENGTH) {
    atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
  }
}

// Compute the Cumulative Distribution Function of histogram
// This is a scan operation like you have done in the previous lab. A sample sequential pseudo code is shown below.

// cdf[0] = p(histogram[0])
// for ii from 1 to 256 do
// 	cdf[ii] = cdf[ii - 1] + p(histogram[ii])
// end
// Where p() calculates the probability of a pixel to be in a histogram bin

// def p(x):
// 	return x / (width * height)
// end
__global__ void scan(unsigned int* input, float* output, int size){
  __shared__ float T[HISTOGRAM_LENGTH*2];
  int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < HISTOGRAM_LENGTH){
    T[threadIdx.x] = input[index];
  }
  int stride = 1;
  while(stride < 2*HISTOGRAM_LENGTH) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*HISTOGRAM_LENGTH && (index-stride) >= 0) {
      T[index] += T[index-stride];
    }
    stride = stride*2;
  }
  stride = HISTOGRAM_LENGTH/2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*HISTOGRAM_LENGTH) {
      T[index+stride] += T[index];
    }
    stride = stride / 2;
  }
  __syncthreads();
  if(index < HISTOGRAM_LENGTH){
    output[index] = T[threadIdx.x] / size;
  }
}

// Compute the minimum value of the CDF. The maximal value of the CDF should be 1.0.
// Define the histogram equalization function
// The histogram equalization function (correct) remaps the cdf of the histogram of the image to a linear function and is defined as

// def correct_color(val) 
// 	return clamp(255*(cdf[val] - cdfmin)/(1.0 - cdfmin), 0, 255.0)
// end

// def clamp(x, start, end)
// 	return min(max(x, start), end)
// end
// Apply the histogram equalization function
// Once you have implemented all of the above, then you are ready to correct the input image. This can be done by writing a kernel to apply the correct_color() function to the RGB pixel values in parallel.

// for ii from 0 to (width * height * channels) do
// 	ucharImage[ii] = correct_color(ucharImage[ii])
// end
// Cast back to float
// for ii from 0 to (width * height * channels) do
// 	outputImage[ii] = (float) (ucharImage[ii]/255.0)
// end
// And you're done
__device__ int clamp(int x, int start, int end) {
    return min(max(x, start), end);
}

__device__ int correct_color(float* cdf, int val) {
    return clamp(255*(cdf[val] - cdf[0])/(1.0 - cdf[0]), 0, 255.0);
}

__global__ void histo_qualization(unsigned char* input, float* output, float* cdf, int size){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    int val = input[i];
    output[i] = (float) correct_color(cdf, val) / 255.0;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInput;
  float *deviceOutput;
  unsigned char *ucharImage;
  unsigned char *grayImage;
  unsigned int *histo;
  float* cdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  //@@ insert code here
  cudaMalloc((void **)&deviceInput, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&ucharImage, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&grayImage, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&histo, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemset(histo, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset(cdf, 0, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemset(deviceOutput, 0, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMemcpy(deviceInput, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid(((imageWidth * imageHeight * imageChannels) - 1) / HISTOGRAM_LENGTH + 1);
  dim3 dimBlock(HISTOGRAM_LENGTH);
  float_to_char<<<dimGrid, dimBlock>>>(deviceInput, ucharImage, imageWidth, imageHeight, imageChannels);
  rgb_to_grayscale<<<dimGrid, dimBlock>>>(ucharImage, grayImage, imageWidth, imageHeight);
  histo_kernel<<<dimGrid, dimBlock>>>(grayImage, histo, imageWidth * imageHeight);
  scan<<<dimGrid, dimBlock>>>(histo, cdf, imageWidth * imageHeight);
  histo_qualization<<<dimGrid, dimBlock>>>(ucharImage, deviceOutput, cdf, imageWidth * imageHeight * imageChannels);
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(ucharImage);
  cudaFree(grayImage);
  cudaFree(histo);
  cudaFree(cdf);
  wbSolution(args, outputImage);

  //@@ insert code here
  free(hostInputImageData);
  free(hostOutputImageData);
  return 0;
}

