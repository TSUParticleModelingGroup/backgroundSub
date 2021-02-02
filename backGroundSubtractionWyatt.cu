// nvcc backGroundSubtractionWyatt.cu -o temp.exe -lm

#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

// size of vector
#define FRAMES 5		// Number of frames
#define PIXELS_PER_FRAME 10	// Number of pixels per frame

#define BLOCK 128  // Size of blocks, best if it is a power of 2.

// Globals
int *BlockOfFrames_CPU, *BlockOfFrames_GPU;
float *MeanFrame_GPU;
float *BlockOfLogNormalFrames_GPU;
float *MeanLogNormalFrame_GPU;
float *MedianLogNormalFrame_GPU;
float *StdvLogNormalFrame_GPU;
int *NewFrame_CPU, *NewFrame_GPU;
int *BlackAndWhiteFrame_CPU, *BlackAndWhiteFrame_GPU;

// These globals can be removed after debugging.
float *BlockOfLogNormalFrames_CPU;
float *MeanFrame_CPU;
float *MeanLogNormalFrame_CPU;
float *MedianLogNormalFrame_CPU;
float *StdvLogNormalFrame_CPU;

dim3 dimBlock, dimGrid;

void AllocateMemory()
{
	// This are the set of frames that will be used to generate the log normal frame
	// and the standard deviation frame
	BlockOfFrames_CPU = (int *)malloc(FRAMES*PIXELS_PER_FRAME*sizeof(int));
	cudaMalloc((void**)&BlockOfFrames_GPU,FRAMES*PIXELS_PER_FRAME*sizeof(int));
	cudaMalloc((void**)&BlockOfLogNormalFrames_GPU,FRAMES*PIXELS_PER_FRAME*sizeof(float));  
	
	// Will hold the log normal frame and the standard deviation of the frames minus the log normal
	cudaMalloc((void**)&MeanFrame_GPU,		PIXELS_PER_FRAME*sizeof(float));
	cudaMalloc((void**)&MeanLogNormalFrame_GPU,	PIXELS_PER_FRAME*sizeof(float));
	cudaMalloc((void**)&MedianLogNormalFrame_GPU,	PIXELS_PER_FRAME*sizeof(float));
	cudaMalloc((void**)&StdvLogNormalFrame_GPU,	PIXELS_PER_FRAME*sizeof(float));
	
	NewFrame_CPU 			= (int *)malloc(PIXELS_PER_FRAME*sizeof(float));
	BlackAndWhiteFrame_CPU 		= (int *)malloc(PIXELS_PER_FRAME*sizeof(float));
	cudaMalloc((void**)&NewFrame_GPU,		PIXELS_PER_FRAME*sizeof(int));
	cudaMalloc((void**)&BlackAndWhiteFrame_GPU,	PIXELS_PER_FRAME*sizeof(int));
	
	// These all can be removed latter. I'm just using them for debuging
	BlockOfLogNormalFrames_CPU 	= (float *)malloc(FRAMES*PIXELS_PER_FRAME*sizeof(float));
	MeanFrame_CPU 			= (float *)malloc(PIXELS_PER_FRAME*sizeof(float));
	MeanLogNormalFrame_CPU 		= (float *)malloc(PIXELS_PER_FRAME*sizeof(float));
	MedianLogNormalFrame_CPU 	= (float *)malloc(PIXELS_PER_FRAME*sizeof(float));
	StdvLogNormalFrame_CPU 		= (float *)malloc(PIXELS_PER_FRAME*sizeof(float));
}
	
void loadPixels()
{
	/*
	However you get 300,000 by 80 pixels loaded in here then CUDA will do the rest.
	This is loading the big vector from 1st 300,000 then from 2nd 300,000 and so on until frame 80.
	It may be faster to load the pixels the other way 80 first pixels then 80 second pixels and so on 300000 times.
	Test it and see.
	I just load (below) some small values to check that everything is working.
	M is the number of frames and N is the number of pixels per frame
	*/
	for(int i = 0; i < FRAMES; i++)
	{
		for(int j = 0; j < PIXELS_PER_FRAME; j++)
		{
			BlockOfFrames_CPU[j +i*PIXELS_PER_FRAME] = i;
			if(i == 4) BlockOfFrames_CPU[j +i*PIXELS_PER_FRAME] = 12;
		}
	}
}

void loadNewFrame()
{
	//This is where you will load the image to be processed.
	for(int i = 0; i < PIXELS_PER_FRAME; i++)
	{
		NewFrame_CPU[i] = i*2;
	}
}

void SetUpCudaDevices()
{
	dimBlock.x = BLOCK;
	dimBlock.y = 1;
	dimBlock.z = 1;

	dimGrid.x = ((PIXELS_PER_FRAME - 1)/BLOCK)+1;
	dimGrid.y = 1;
	dimGrid.z = 1;
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
			
			//allFramesLogNormal[id] = (float)allFrames[id];  // Remove after debugging.
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
	int used[FRAMES], index, count;
	float median = 0.0;
	float small;
	if(pixel < pixelsPerFrame)
	{
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
				small = 10000000.0f;  //Needs to be a number larger than anything you would get in a log of a pixel.
				for(int i = 0; i < frames; i++)
				{
					if(allFramesLogNormal[pixel + pixelsPerFrame*i] < small && used[i] == 0)
					{
						small = allFramesLogNormal[pixel + pixelsPerFrame*i];
						index = i;
					}
				}
				if(index == -1) printf("\nError no index found\n");
				used[index] = 1;
	
				if(count == middle1 || count == middle2) 
				{	
					median += allFramesLogNormal[pixel + pixelsPerFrame*index];
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
				small = 10000000.0f;  //Needs to be a number larger than anything you would get in a log of a pixel.
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
		stdvLogNormalFrame[pixel] = sqrtf((sum)/(float)(frames-1));
	}
}

__global__ void CreateBlackAndWHiteFrame(int *BlackAndWhiteFrame_GPU, int *NewFrame_GPU, float *StdvLogNormalFrame_GPU, float *MeanLogNormalFrame_GPU, int pixelsPerFrame)
{
	int pixel = threadIdx.x + blockIdx.x*blockDim.x;
	
	float breakPoint = 2.0f; // ************** not sure what this value should be ??????????
	
	if(pixel < pixelsPerFrame)
	{
		float CDF = 0.5f + 0.5f*erff((logf((float)NewFrame_GPU[pixel]) - MeanLogNormalFrame_GPU[pixel])/sqrtf(2.0*StdvLogNormalFrame_GPU[pixel]));
		if(CDF < breakPoint)
		{
			BlackAndWhiteFrame_GPU[pixel] = 0;
		}
		else
		{
			BlackAndWhiteFrame_GPU[pixel] = 1;  //Can remove this if you do a memset before you use the data.
		}
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

void stats()
{

	cudaMemcpyAsync(BlockOfLogNormalFrames_CPU, BlockOfLogNormalFrames_GPU, PIXELS_PER_FRAME*FRAMES*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck("copy Mean frame down");
	cudaMemcpyAsync(MeanFrame_CPU, MeanFrame_GPU, PIXELS_PER_FRAME*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck("copy Mean frame down");
	cudaMemcpyAsync(MeanLogNormalFrame_CPU, MeanLogNormalFrame_GPU, PIXELS_PER_FRAME*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck("copy MeanLogNormal frame down");
	cudaMemcpyAsync(MedianLogNormalFrame_CPU, MedianLogNormalFrame_GPU, PIXELS_PER_FRAME*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck("copy MedianLogNormal frame down");
	cudaMemcpyAsync(StdvLogNormalFrame_CPU, StdvLogNormalFrame_GPU, PIXELS_PER_FRAME*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck("copy StdvLogNormal frame down");
	
	printf("\n\n");
	printf("frames");
	for(int j = 0; j < FRAMES; j++)
	{
		printf("\n");
		for(int i = 0; i < PIXELS_PER_FRAME; i++)
		{
			printf("%d ", BlockOfFrames_CPU[i + j*PIXELS_PER_FRAME]);
		}
	}
	
	printf("\n\n");
	printf("log normal frames");
	for(int j = 0; j < FRAMES; j++)
	{
		printf("\n");
		for(int i = 0; i < PIXELS_PER_FRAME; i++)
		{
			printf("%f ", BlockOfLogNormalFrames_CPU[i + j*PIXELS_PER_FRAME]);
		}
	}
	
	printf("\n\n");
	for(int i = 0; i < PIXELS_PER_FRAME; i++)
	{
		printf("MeanFrame[%d] = %f MeanLogNormalFrame[%d] = %f MedianLogNormalFrame[%d] = %f StdvLogNormalFrame[%d] = %f \n", i, MeanFrame_CPU[i], i, MeanLogNormalFrame_CPU[i], i, MedianLogNormalFrame_CPU[i], i, StdvLogNormalFrame_CPU[i]);
	}
	
	printf("\n");
	for(int i = 0; i < PIXELS_PER_FRAME; i++)
	{
		printf("NewFrame[%d] = %d blackAndWhiteFrame[%d] = %d \n", i, NewFrame_CPU[i], i, BlackAndWhiteFrame_CPU[i]);
	}
}

void cleanUp()
{
	free(BlockOfFrames_CPU);
	free(NewFrame_CPU);
	free(BlackAndWhiteFrame_CPU);

	cudaFree(BlockOfFrames_GPU);
	cudaFree(BlockOfLogNormalFrames_GPU);
	cudaFree(MeanFrame_GPU);
	cudaFree(MeanLogNormalFrame_GPU);
	cudaFree(MedianLogNormalFrame_GPU);
	cudaFree(StdvLogNormalFrame_GPU);
	cudaFree(NewFrame_GPU);
	cudaFree(BlackAndWhiteFrame_GPU);
	
	// These can be removed latter. I just used them for debuging.
	free(BlockOfLogNormalFrames_CPU);
	free(MeanFrame_CPU);
	free(MeanLogNormalFrame_CPU);
	free(MedianLogNormalFrame_CPU);
	free(StdvLogNormalFrame_CPU);
}

int main()
{
	AllocateMemory();
	SetUpCudaDevices();
	loadPixels();
	
	cudaMemcpyAsync(BlockOfFrames_GPU, BlockOfFrames_CPU, PIXELS_PER_FRAME*FRAMES*sizeof(int), cudaMemcpyHostToDevice);
	errorCheck("copyFramessUp");
	cudaDeviceSynchronize();
	creatingMeanPixelFrame<<<dimGrid,dimBlock>>>(MeanFrame_GPU, BlockOfFrames_GPU, PIXELS_PER_FRAME, FRAMES);
	errorCheck("creatingMeanPixelFrame");
	
	creatingLogNormalFrames<<<dimGrid,dimBlock>>>(MeanFrame_GPU, BlockOfFrames_GPU, BlockOfLogNormalFrames_GPU, PIXELS_PER_FRAME, FRAMES);
	errorCheck("creatingLogNormalFrames");
	
	creatingMeanLogNormalFrame<<<dimGrid,dimBlock>>>(MeanLogNormalFrame_GPU, BlockOfLogNormalFrames_GPU, PIXELS_PER_FRAME, FRAMES);
	errorCheck("creatingMeanLogNormalFrame");
	
	creatingMedianLogNormalFrame<<<dimGrid,dimBlock>>>(MedianLogNormalFrame_GPU, BlockOfLogNormalFrames_GPU, PIXELS_PER_FRAME, FRAMES);
	errorCheck("creatingMedianLogNormalFrame");
	
	creatingStdvLogNormalFrame<<<dimGrid,dimBlock>>>(StdvLogNormalFrame_GPU, MeanLogNormalFrame_GPU, BlockOfLogNormalFrames_GPU, PIXELS_PER_FRAME, FRAMES);
	errorCheck("creatingStdvLogNormalFrame");
	
	cudaDeviceSynchronize();
	
	loadNewFrame();
	cudaMemcpyAsync(NewFrame_GPU, NewFrame_CPU, PIXELS_PER_FRAME*sizeof(int), cudaMemcpyHostToDevice);
	errorCheck("copy New frame up");
	
	CreateBlackAndWHiteFrame<<<dimGrid,dimBlock>>>(BlackAndWhiteFrame_GPU, NewFrame_GPU, StdvLogNormalFrame_GPU, MeanLogNormalFrame_GPU, PIXELS_PER_FRAME);
	errorCheck("creatingStdvLogNormalFrame");
	cudaMemcpyAsync(BlackAndWhiteFrame_CPU, BlackAndWhiteFrame_GPU, PIXELS_PER_FRAME*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck("copy black and white frame down");
	
	//Do stuff with black and white frame
	
	stats();
	cleanUp();
	printf("\n DONE \n");
}
