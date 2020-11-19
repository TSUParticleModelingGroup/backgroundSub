// nvcc EthanPixels.cu -o temp -lm

#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime.h>

// size of vector
#define N 10
#define M 4

#define BLOCK 128

// Globals
int *Pixels_CPU, *AveragePixel_CPU;
int *Pixels_GPU, *AveragePixel_GPU, *logMean_GPU, *logStd_GPU;
dim3 dimBlock, dimGrid;

void AllocateMemory()
{
	Pixels_CPU = (int *)malloc(N*M*sizeof(int)); // Probably a short int
	cudaMalloc((void**)&Pixels_GPU,N*M*sizeof(int));  // Probably a short int
	AveragePixel_CPU = (int *)malloc(N*sizeof(int)); // Probably a short int
	logMean_CPU = (int *)malloc(N*sizeof(int)); // Probably a short int
	logStd_CPU = (int *)malloc(N*sizeof(int)); // Probably a short int
	cudaMalloc((void**)&AveragePixel_GPU,N*sizeof(int));  // Probably a short int
	cudaMalloc((void**)&logMean_GPU,N*sizeof(float));  // Probably a short int
	cudaMalloc((void**)&logStd_GPU,N*sizeof(float));  // Probably a short int
}

/* 
	However you get you 300,000 by 80 pixels loaded in here then CUDA will do the rest. 
	This is loading the big vector from 1st 300,000 then from 2nd 300,000 and so on until frame 80.
   	It may be faster to load the pixels the other way 80 first pixels then 80 second pixels and so on 300000 times.
   	Test it and see.
   	I just load (below) some small values to check that everything is working. 
   	M is the number of frames and N is the number of pixels per frame
*/
void loadPixels()
{
	for(int i = 0; i < M; i++)
	{
		for(int j = 0; j < N; j++)
		{
			Pixels_CPU[j +i*N] = i*5;
		}
	}
	for(int j = 0; j < N; j++)
	{
		AveragePixel_CPU[j] = 1;
	}
}

void SetUpCudaDevices()
{
	dimBlock.x = BLOCK;
	dimBlock.y = 1;
	dimBlock.z = 1;

	dimGrid.x = ((N-1)/BLOCK)+1;
	dimGrid.y = 1;
	dimGrid.z = 1;
}

void copyPixelsUp()
{
	//cudaMemcpy(Pixels_GPU, Pixels_CPU, N*M*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(Pixels_GPU, Pixels_CPU, N*M*sizeof(int), cudaMemcpyHostToDevice);
}

__global__ void pixelWork(int *averagePixel, int *totalPixels, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(pixel < pixelsPerFrame)
	{
		int sum = 0;
		for(int i = 0; i < frames; i++)
		{
			sum += totalPixels[pixel + pixelsPerFrame*i];
		}
		averagePixel[pixel] = sum/frames;
	}
}

__global__ void logNormal(int *averagePixel, int *totalPixels, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(pixel < pixelsPerFrame)
	{	
		for(int i = 0; i < frames; i++)
		{
			totalPixels[pixel] = totalPixels[pixel] -  averagePixel[pixel];
			totalPixels[pixel] = abs(totalPixels[pixel]);
			totalPixels[pixel] = log(totalPixels[pixel]);
		}
	}
}

__global__ void pixelWork2(float *logMean, int *totalPixels, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(pixel < pixelsPerFrame)
	{
		int sum = 0;
		for(int i = 0; i < frames; i++)
		{
			sum += totalPixels[pixel + pixelsPerFrame*i];
		}
		logMean[pixel] = sum/frames;
	}
}

__global__ void pixelWork3(float *logStd, float *logMean, int *totalPixels, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(pixel < pixelsPerFrame)
	{
		int sum = 0;
		for(int i = 0; i < frames; i++)
		{
			sum += totalPixels[pixel + pixelsPerFrame*i] - logMean[pixel + pixelsPerFrame*i];
		}
		logStd[pixel] = sqrt( (sum*sum)/(frames-1));
	}
}

void copyPixelsDown()
{
	cudaMemcpyAsync(AveragePixel_CPU, AveragePixel_GPU, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(logMean_CPU, logMean_GPU, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(logStd_CPU, logStd_GPU, N*sizeof(int), cudaMemcpyDeviceToHost);
}

void stats()
{
	for(int i = 0; i < N; i++)
	{
		printf("AveragePixel_CPU[%d] = %d \n", i, AveragePixel_CPU[i]);
	}
}

void cleanUp()
{
	free(Pixels_CPU);
	free(AveragePixel_CPU);

	cudaFree(Pixels_GPU);
	cudaFree(AveragePixel_GPU);
}

void errorCheck(const char *message)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: %s = %s\n", message, cudaGetErrorString(error));
		exit(0);
	}
}

int main()
{
	AllocateMemory();
	SetUpCudaDevices();
	loadPixels();
	copyPixelsUp();
	errorCheck("copyPixelsUp");
	cudaDeviceSynchronize();
	pixelWork<<<dimGrid,dimBlock>>>(AveragePixel_GPU, Pixels_GPU, N, M);
	errorCheck("pixelWork");
	logNormal(AveragePixel_GPU, Pixels_GPU, N, M);
	errorCheck("logNormal");
	pixelWork2<<<dimGrid,dimBlock>>>(logMean_GPU, Pixels_GPU, N, M);
	pixelWork3<<<dimGrid,dimBlock>>>(logStd_GPU,logMean_GPU, Pixels_GPU, N, M);
	copyPixelsDown();
	errorCheck("copyAveragePixelsDown");
	cudaDeviceSynchronize();
	stats();
	cleanUp();
	printf("\n DONE \n");
}

