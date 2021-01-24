// nvcc backGroundSubtraction.cu -o temp.exe -lm

#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

// size of vector
#define M 4	// Number of frames
#define N 10	// Number of pixels per frame

#define BLOCK 128  // Size of blocks, best if it is a power of 2.

// Globals
int *BlockOfFrames_CPU, *BlockOfFrames_GPU;
float *MeanFrame_CPU, *MeanFrame_GPU;
float *BlockOfLogNormalFrames_GPU;
float *MeanLogNormalFrame_CPU, *MeanLogNormalFrame_GPU;
float *MedianLogNormalFrame_CPU, *MedianLogNormalFrame_GPU;
float *StdvLogNormalFrame_CPU, *StdvLogNormalFrame_GPU;

dim3 dimBlock, dimGrid;

void AllocateMemory()
{
	// This are the set of frames that will be used to generate the log normal frame
	// and the standard deviation frame
	BlockOfFrames_CPU = (int *)malloc(N*M*sizeof(int)); 
	cudaMalloc((void**)&BlockOfFrames_GPU,N*M*sizeof(int));
	cudaMalloc((void**)&BlockOfLogNormalFrames_GPU,N*M*sizeof(float));  
	
	// Will hold the log normal frame and the standard deviation of the frames minus the log normal
	MeanFrame_CPU 			= (float *)malloc(N*sizeof(float));
	MeanLogNormalFrame_CPU 		= (float *)malloc(N*sizeof(float));
	MedianLogNormalFrame_CPU 	= (float *)malloc(N*sizeof(float));
	StdvLogNormalFrame_CPU 		= (float *)malloc(N*sizeof(float));
	cudaMalloc((void**)&MeanFrame_GPU,		N*sizeof(float));
	cudaMalloc((void**)&MeanLogNormalFrame_GPU,	N*sizeof(float));
	cudaMalloc((void**)&MedianLogNormalFrame_GPU,	N*sizeof(float));
	cudaMalloc((void**)&StdvLogNormalFrame_GPU,	N*sizeof(float));
}
	
/*
	However you get 300,000 by 80 pixels loaded in here then CUDA will do the rest.
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
			BlockOfFrames_CPU[j +i*N] = i;
			//if(i == M-1) BlockOfFrames_CPU[j +i*N] = 4000;
		}
	}
	for(int j = 0; j < N; j++)
	{
		MeanFrame_CPU[j] = -1.0;
		MeanLogNormalFrame_CPU[j] = -1.0;
		MedianLogNormalFrame_CPU[j] = -1.0;
		StdvLogNormalFrame_CPU[j] = -1.0;
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

__global__ void creatingMeanPixelFrame(float *meanFrame, int *allFrames, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	if(pixel < pixelsPerFrame)
	{
		double sum = 0.0;
		for(int i = 0; i < frames; i++)
		{
			sum += allFrames[pixel + pixelsPerFrame*i];
		}
		meanFrame[pixel] = sum/(float)frames;
	}
}

__global__ void creatingLogNormalFrames(float *meanFrame, int *allFrames, float *allFramesLogNormal, int pixelsPerFrame, int frames)
{
	int id;
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	if(pixel < pixelsPerFrame)
	{
		for(int i = 0; i < frames; i++)
		{
			//Same screen location (pixel) but moving through frames (i).
			id = pixel + pixelsPerFrame*i;
			
			allFramesLogNormal[id] = (float)allFrames[id] -  meanFrame[pixel];
			allFramesLogNormal[id] = abs(allFramesLogNormal[id]);
			
			//Can't take log of zero so to be safe check and move it off zero.
			if(allFramesLogNormal[id] == 0.0f) 
			{
				allFramesLogNormal[id] = 0.000001f;
			}
			
			allFramesLogNormal[id] = logf(allFramesLogNormal[id]);
			
			allFramesLogNormal[id] = (float)allFrames[id]; // Need to take this line out used it tot taet mean and median and stdv.
		}
	}
}

__global__ void creatingMeanLogNormalFrame(float *meanlogNormalFrame, float *allFramesLogNormal, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	if(pixel < pixelsPerFrame)
	{
		double sum = 0.0;
		for(int i = 0; i < frames; i++)
		{
			sum += allFramesLogNormal[pixel + pixelsPerFrame*i];
		}
		meanlogNormalFrame[pixel] = sum/(float)frames;
	}
}

__global__ void creatingMedianLogNormalFrame(float *medianlogNormalFrame, float *allFramesLogNormal, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	int used[M], index, count;
	float median = 0.0;
	float small;
	if(pixel < pixelsPerFrame)
	{
		small = 10000000.0f;  //Needs to be a number larger than anything you would get in a log of a pixel.
		for(int i = 0; i < frames; i++)
		{
			used[i] = 0;
		}
		if(frames%2 == 0)
		{
			int middle2 = frames/2;
			int middle1 = middle2 - 1;
			index = -1;
			count = 0;
			while(count <= middle2)
			{
				for(int i = 0; i < frames; i++)
				{
					//printf("\n all = %f id = %d", allFramesLogNormal[pixel + pixelsPerFrame*i], pixel + pixelsPerFrame*i);
					if(allFramesLogNormal[pixel + pixelsPerFrame*i] < small && used[i] == 0)
					{
						small = allFramesLogNormal[pixel + pixelsPerFrame*i];
						printf("\n small = %f", small);
						index = i;
					}
				}
				if(index == -1) printf("\nError no index found\n");
				used[index] = 1;
	
				if(count == middle1 || count == middle2) 
				{	
					median += allFramesLogNormal[pixel + pixelsPerFrame*index];
					printf("\nMedian = %f index = %d, middle1 = %d, middle2 = %d", median, index, middle1, middle2);
				}
				
				count++;
			}
			median /=2.0f;
		}
		else
		{
			int middle = frames/2;
			index = -1;
			count = 0;
			while(count <= middle)
			{
				for(int i = 0; i < frames; i++)
				{
					if(allFramesLogNormal[pixel + pixelsPerFrame*i] < small)
					{
						if(used[i] == 0)
						{
							small = allFramesLogNormal[pixel + pixelsPerFrame*i];
							index = i;
						}
					}
				}
				if(index == -1) printf("\nError no index found\n");
				used[index] = 1;
				
				if(count == middle) 
				{
					median += allFramesLogNormal[pixel + pixelsPerFrame*index];
				}
				
				count++;
			}
		}
		medianlogNormalFrame[pixel] = median;
	}
}

__global__ void creatingStdvLogNormalFrame(float *stdvLogNormalFrame, float *meanLogNormalFrame, float *allFramesLogNormal, int pixelsPerFrame, int frames)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	float temp;
	if(pixel < pixelsPerFrame)
	{
		double sum = 0.0;
		for(int i = 0; i < frames; i++)
		{
			temp = allFramesLogNormal[pixel + pixelsPerFrame*i] - meanLogNormalFrame[pixel];
			sum += temp*temp;
		}
		stdvLogNormalFrame[pixel] = sqrtf((sum*sum)/(float)(frames-1));
	}
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

void copyFramesDown()
{
	cudaMemcpyAsync(MeanFrame_CPU, MeanFrame_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck("copy Mean frame down");
	cudaMemcpyAsync(MeanLogNormalFrame_CPU, MeanLogNormalFrame_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck("copy MeanLogNormal frame down");
	cudaMemcpyAsync(MedianLogNormalFrame_CPU, MedianLogNormalFrame_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck("copy MedianLogNormal frame down");
	cudaMemcpyAsync(StdvLogNormalFrame_CPU, StdvLogNormalFrame_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck("copy StdvLogNormal frame down");
}

void stats()
{
	for(int i = 0; i < N; i++)
	{
		printf("MeanFrame[%d] = %f MeanLogNormalFrame[%d] = %f MedianLogNormalFrame[%d] = %f StdvLogNormalFrame[%d] = %f \n", i, MeanFrame_CPU[i], i, MeanLogNormalFrame_CPU[i], i, MedianLogNormalFrame_CPU[i], i, StdvLogNormalFrame_CPU[i]);
	}
}

void cleanUp()
{
	free(BlockOfFrames_CPU);
	free(MeanFrame_CPU);
	free(MeanLogNormalFrame_CPU);
	free(MedianLogNormalFrame_CPU);
	free(StdvLogNormalFrame_CPU);

	cudaFree(BlockOfFrames_GPU);
	cudaFree(BlockOfLogNormalFrames_GPU);
	cudaFree(MeanFrame_GPU);
	cudaFree(MeanLogNormalFrame_GPU);
	cudaFree(MedianLogNormalFrame_GPU);
	cudaFree(StdvLogNormalFrame_GPU);
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
	
	creatingMedianLogNormalFrame<<<dimGrid,dimBlock>>>(MedianLogNormalFrame_GPU, BlockOfLogNormalFrames_GPU, N, M);
	errorCheck("creatingMedianLogNormalFrame");
	
	creatingStdvLogNormalFrame<<<dimGrid,dimBlock>>>(StdvLogNormalFrame_GPU, MeanLogNormalFrame_GPU, BlockOfLogNormalFrames_GPU, N, M);
	errorCheck("creatingStdvLogNormalFrame");
	
	copyFramesDown();
	errorCheck("copyFramesDown");
	
	cudaDeviceSynchronize();
	
	stats();
	cleanUp();
	printf("\n DONE \n");
}
