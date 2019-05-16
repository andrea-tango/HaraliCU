#ifndef CUDAFUNCTIONS_H_
#define CUDAFUNCTIONS_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#include <iostream>
#include <assert.h>

#include "Window.h"
#include "WorkArea.h"
#include "WindowFeatureComputer.h"
#include "ImageData.h"

using namespace std;

// Error handling
/**
 * Exits the program if any of the CUDA runtime function invocation, controlled
 * by this method, returns a failure
 */
void cudaCheckError(cudaError_t err);
/**
 * Exits the program if the launch of the kernel of computation fails
 */
void checkKernelLaunchError();
/**
 * Prints how many blocks and how many threads per block will be computed the 
 * the kernel
 */
void printGPULaunchConfiguration(dim3 Grid, dim3 Blocks);
/*
 * Querying GPU info
 */
void queryGPUData();

// Block dimensioning
int getCudaBlockSideX();
int getCudaBlockSideY();
/*
 * Returns the dimension of each squared block of threads
 */
dim3 getBlockConfiguration();

//Grid dimensioning
/**
 * Returns the dimension of a grid obtained from the image physical dimension where
 * each thread computes just a window
 */
int getGridSide(int imageRows, int imageCols);
/**
 * Creates a grid from the image dimension where each thread  
 * computes just a window
 */
dim3 getGridFromImage(int imageRows, int imageCols);
/**
 * Method that generates the smallest computing grid that can fit in
 * the GPU memory
 * @param numberOfPairs: number of pixel pairs that belong to each window
 * @param featureSize: memory space required by the values that will be computed
 */
dim3 getGridFromAvailableMemory(int numberOfPairs,
 size_t featureSize);
/**
 * Method that generates the computing grid 
 * GPU allocable heap changes according to the defined grid
 * If any block cannot be launched ,the program will abort
 * @param numberOfPairsInWindow: number of pixel pairs that belong to each window
 * @param featureSize: memory space required by the values that will be computed
 * @param imgRows: number of the rows composing the image
 * @param imgCols: number of the columns composing the image
 * @param verbose: print extra info about the required memory
 */
dim3 getGrid(int numberOfPairsInWindow, size_t featureSize, int imgRows, 
	int imgCols, bool verbose);
/**
 * Method invoked by each thread to obtain the reference to their own memory,
 * entirely pre-allocated on the host, needed for their computation
 * @param globalWorkArea: reference to the global, allocated by the host, memory
 * that each thread will use to perform their job
 * @param threadId: unique thread id inside the launching configuration
 */
__device__ WorkArea adjustThreadWorkArea(WorkArea globalWorkArea, int threadId);
/**
 * The program aborts if any block of threads cannot be launched for 
 * insufficient memory (especially, in the case of obsolete GPUs)
 */
void handleInsufficientMemory();
/**
 * Method that checks if the proposed number of threads will have enough memory
 * @param numberOfPairs: number of pixel pairs that belong to each window
 * @param featureSize: memory space required by the values that will be computed
 * @param numberOfThreads: number of threads of the proposed grid
 * @param verbose: printing extra info on the memory consumed
 */
bool checkEnoughWorkingAreaForThreads(int numberOfPairs, int numberOfThreads,
 size_t featureSize, bool verbose);

/**
 * Kernel that computes all the features in each window of the image. Each
 * window will be computed by an autonomous thread of the grid
 * @param pixels: pixel intensities of the analyzed image
 * @param img: image metadata
 * @param globalWorkArea: class that stores the pointers to the pre-allocated 
 * space that will contain the arrays of representations used by each thread
 * to perform its computation
 */
__global__ void computeFeatures(unsigned int * pixels, 
	ImageData img, Window windowData, WorkArea globalWorkArea);
#endif 
