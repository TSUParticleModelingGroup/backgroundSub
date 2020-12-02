// nvcc EthanPixels.cu -o temp -lm

#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

// size of vector
#define N 10
#define M 4

#define BLOCK 128

// Globals
int *BlockOfFrames_CPU, *MeanFrame_CPU;
float *MeanLogNormalFrame_CPU, *StdvLogNormalFrame_CPU;
int *BlockOfFrames_GPU, *BlockOfLogNormalFrames_GPU;
int *MeanFrame_GPU;
float *MeanLogNormalFrame_GPU, *StdvLogNormalFrame_GPU;
dim3 dimBlock, dimGrid;

void AllocateMemory()
{
	// This are the set of frames that will be used to generate the log normal frame
	// and the standard deviation frame
	BlockOfFrames_CPU = (int *)malloc(N*M*sizeof(int)); 
	cudaMalloc((void**)&BlockOfFrames_GPU,N*M*sizeof(int));
	cudaMalloc((void**)&BlockOfLogNormalFrames_GPU,N*M*sizeof(int));  
	
	// Will hold the log normal frame and the standard deviation of the frames minus the log normal
	MeanFrame_CPU = (int *)malloc(N*sizeof(int));
	MeanLogNormalFrame_CPU = (float *)malloc(N*sizeof(float));
	StdvLogNormalFrame_CPU = (float *)malloc(N*sizeof(float));
	cudaMalloc((void**)&MeanFrame_GPU,N*sizeof(int));
	cudaMalloc((void**)&MeanLogNormalFrame_GPU,N*sizeof(float));
	cudaMalloc((void**)&StdvLogNormalFrame_GPU,N*sizeof(float));
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
			BlockOfFrames_CPU[j +i*N] = i*5;
		}
	}
	for(int j = 0; j < N; j++)
	{
		MeanFrame_CPU[j] = -1;
		MeanLogNormalFrame_CPU[j] = -1;
		StdvLogNormalFrame_CPU[j] = -1;
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

void copyFramessUp()
{
	cudaMemcpyAsync(BlockOfFrames_GPU, BlockOfFrames_CPU, N*M*sizeof(int), cudaMemcpyHostToDevice);
}

__global__ void creatingMeanPixelFrame(int *meanFrame, int *allFrames, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	if(pixel < pixelsPerFrame)
	{
		int sum = 0;
		for(int i = 0; i < frames; i++)
		{
			sum += allFrames[pixel + pixelsPerFrame*i];
		}
		meanFrame[pixel] = sum/frames;
	}
}

__global__ void creatingLogNormalFrames(int *meanFrame, int *allFrames, int *allFramesLogNormal, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	if(pixel < pixelsPerFrame)
	{
		for(int i = 0; i < frames; i++)
		{
			allFramesLogNormal[pixel] = allFrames[pixel] -  meanFrame[pixel];
			allFramesLogNormal[pixel] = abs(allFramesLogNormal[pixel]);
			allFramesLogNormal[pixel] = __log(allFramesLogNormal[pixel]);
		}
	}
}

__global__ void creatingMeanLogNormalFrame(float *meanlogNormalFrame, int *allFramesLogNormal, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	if(pixel < pixelsPerFrame)
	{
		int sum = 0;
		for(int i = 0; i < frames; i++)
		{
			sum += allFramesLogNormal[pixel + pixelsPerFrame*i];
		}
		meanlogNormalFrame[pixel] = sum/frames;
	}
}

__global__ void creatingStdvLogNormalFrame(float *stdvLogNormalFrame, float *meanLogNormalFrame, int *allFramesLogNormal, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	if(pixel < pixelsPerFrame)
	{
		int sum = 0;
		for(int i = 0; i < frames; i++)
		{
			sum += allFramesLogNormal[pixel + pixelsPerFrame*i] - meanLogNormalFrame[pixel];
		}
		stdvLogNormalFrame[pixel] = __sqrt((sum*sum)/(frames-1));
	}
}

void copyFramesDown()
{
	cudaMemcpyAsync(MeanFrame_CPU, MeanFrame_GPU, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(MeanLogNormalFrame_CPU, MeanLogNormalFrame_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(StdvLogNormalFrame_CPU, StdvLogNormalFrame_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
}

void stats()
{
	for(int i = 0; i < N; i++)
	{
		printf("MeanFrame_CPU[%d] = %d MeanLogNormalFrame_CPU[%d] = %f StdvLogNormalFrame_CPU[%d] = %f\n", i, MeanFrame_CPU[i], i, MeanLogNormalFrame_CPU[i], i, StdvLogNormalFrame_CPU[i]);
	}
}

void cleanUp()
{
	free(BlockOfFrames_CPU);
	free(MeanFrame_CPU);
	free(MeanLogNormalFrame_CPU);
	free(StdvLogNormalFrame_CPU);

	cudaFree(BlockOfFrames_GPU);
	cudaFree(BlockOfLogNormalFrames_GPU);
	cudaFree(MeanFrame_GPU);
	cudaFree(MeanLogNormalFrame_GPU);
	cudaFree(StdvLogNormalFrame_GPU);
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
	copyFramessUp();
	errorCheck("copyFramessUp");
	
	cudaDeviceSynchronize();
	creatingMeanPixelFrame<<<dimGrid,dimBlock>>>(MeanFrame_GPU, BlockOfFrames_GPU, N, M);
	errorCheck("creatingMeanPixelFrame");
	
	creatingLogNormalFrames<<<dimGrid,dimBlock>>>(MeanFrame_GPU, BlockOfFrames_GPU, BlockOfLogNormalFrames_GPU, N, M);
	errorCheck("creatingLogNormalFrames");
	
	creatingMeanLogNormalFrame<<<dimGrid,dimBlock>>>(MeanLogNormalFrame_GPU, BlockOfLogNormalFrames_GPU, N, M);
	errorCheck("creatingMeanLogNormalFrame");
	
	creatingStdvLogNormalFrame<<<dimGrid,dimBlock>>>(StdvLogNormalFrame_GPU, MeanLogNormalFrame_GPU, BlockOfLogNormalFrames_GPU, N, M);
	errorCheck("creatingStdvLogNormalFrame");
	
	copyFramesDown();
	errorCheck("copyFramesDown");
	
	cudaDeviceSynchronize();
	
	stats();
	cleanUp();
	printf("\n DONE \n");
}
