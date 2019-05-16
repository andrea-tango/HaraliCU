#include "CudaFunctions.h"
#include <cmath>

/* CUDA METHODS */

/**
 * Exits the program if any of the CUDA runtime function invocation, controlled
 * by this method, returns a failure
 */
void cudaCheckError(cudaError_t err){
	if( err != cudaSuccess ) {
		cerr << "ERROR: " << cudaGetErrorString(err) << endl;
		exit(-1);
	}
}

 
/**
/**
 * Exits the program if the launch of the kernel of computation fails
 */
void checkKernelLaunchError(){
	cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) // Detect configuration launch errors
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess) // Detect kernel execution errors
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

/**
 * Print how many blocks and how many threads per block will be computed 
 * by the kernel
 */
void printGPULaunchConfiguration(dim3 Grid, dim3 Blocks){
	cout << "\t- GPU Launch Configuration -" << endl;
	cout << "\t GRID\t rows: " << Grid.y << " x cols: " << Grid.x << endl;
	cout << "\t BLOCK\t rows: " << Blocks.y << " x cols: " << Blocks.x << endl;
}

/*
 * Querying and printing info on the gpu of the system
 */
void queryGPUData(){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cout << "\t- GPU DATA -" << endl;
	cout << "\tDevice name: " << prop.name << endl;
	cout << "\tNumber of multiprocessors: " << prop.multiProcessorCount << endl;
	size_t gpuMemory = prop.totalGlobalMem;
	cout << "\tTotalGlobalMemory: " << (gpuMemory / 1024 / 1024) << " MB" << endl;
}

 /* 
	Default CUDA block dimensioning 
	Square with side of 16
*/
int getCudaBlockSideX(){
	return 16;
}

// Square
int getCudaBlockSideY(){
	return 16;
}

/* 
 * The block is always fixed, only the grid changes 
 * according to gpu memory/image size 
*/
dim3 getBlockConfiguration()
{
	int ROWS = getCudaBlockSideY();
	int COLS = getCudaBlockSideX(); 
	dim3 configuration(ROWS, COLS);
	return configuration;
}

/**
 * Returns the dimension of a grid obtained from the image physical dimension where
 * each thread computes just a window
 */
int getGridSide(int imageRows, int imageCols){
	// Smallest side of a rectangular image will determine grid dimension
	int imageSmallestSide = imageRows;
	if(imageCols < imageSmallestSide)
		imageSmallestSide = imageCols;
   
	int blockSide = getCudaBlockSideX();
	// Check if image size is low enough to fit in maximum grid
	// round up division 
	int gridSide = (imageSmallestSide + blockSide -1) / blockSide;
	// Cannot exceed 65536 blocks in grid
	if(gridSide > 256){
		gridSide = 256;
	}
	return gridSide;
}

/**
 * Creates a grid from the image dimension where each thread  
 * computes just a window
 */
dim3 getGridFromImage(int imgRows, int imgCols)
{

	dim3 Blocks = getBlockConfiguration();

 	double x = 2;
 	int value = (int)ceil(imgRows*imgCols/(1.0*Blocks.x * Blocks.y));

 	while(true)
 	{

 		if(x*x >= value)
 		{
 			break;
 		}

 		else
 			x += 1;
 	}

 	int dim = (int) x;

	return dim3(dim, dim);
}


/**
 * Program aborts if not even 1 block of threads can be launched for 
 * insufficient memory (very obsolete gpu)
 */
void handleInsufficientMemory(){
	cerr << "FAILURE ! Gpu doesn't have enough memory \
	to hold the results and the space needed to threads" << endl;
	cerr << "Try lowering window side and/or symmetricity "<< endl;
	exit(-1);
}  

/**
 * Method that checks if the proposed number of threads will have enough memory
 * @param numberOfPairs: number of pixel pairs that belong to each window
 * @param featureSize: memory space required by the values that will be computed
 * @param numberOfThreads: the number of the threads in the proposed grid
 * @param verbose: printing extra info on the memory consumed
 */
bool checkEnoughWorkingAreaForThreads(int numberOfPairs, int numberOfThreads,
 size_t featureSize, bool verbose){
	// Getting the GPU mem size
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	size_t gpuMemory = prop.totalGlobalMem;
	// Computing the memory needed by a single thread
	size_t workAreaSpace = numberOfPairs * 
		( 4 * sizeof(AggregatedGrayPair) + sizeof(GrayPair));
	// Multiplying for the number of threads
	size_t totalWorkAreas = workAreaSpace * numberOfThreads;

	long long int difference = gpuMemory - (featureSize + totalWorkAreas);
	if(difference <= 0){
		return false;
	}
	else{
		if(verbose){
			cout << endl << "\tGPU threads space: (MB) " << totalWorkAreas / 1024 / 1024 << endl;
			size_t free, total;
			cudaMemGetInfo(&free,&total);
			cout << "\tGPU used memory: (MB) " << (((totalWorkAreas + featureSize) / 1024) / 1024) << endl;
		}
		return true;
	}
}

/**
 * Method that generates the smallest computing grid that can fit in
 *  the GPU memory
 * @param numberOfPairs: number of pixel pairs that belong to each window
 * @param featureSize: memory space required by the values that will be computed
 */
dim3 getGridFromAvailableMemory(int numberOfPairs,
 size_t featureSize){

	// Getting the GPU mem size
	size_t freeGpuMemory, total;
	cudaMemGetInfo(&freeGpuMemory,&total);

	// Computing the memory needed by a single thread
	size_t workAreaSpace = numberOfPairs * 
		( 4 * sizeof(AggregatedGrayPair) + 1 * sizeof(GrayPair));

	// How many thread fit in a single block
	int threadInBlock = getCudaBlockSideX() * getCudaBlockSideX();

	size_t singleBlockMemoryOccupation = workAreaSpace * threadInBlock;
	// Even one block can be launched
	if(freeGpuMemory <= singleBlockMemoryOccupation){
		handleInsufficientMemory(); // exit
	}

	cout << endl << "WARNING! Maximum available gpu memory consumed" << endl;
	// How many blocks can be launched
	int numberOfBlocks = freeGpuMemory / singleBlockMemoryOccupation;
	
	// Creating a 2D grid of blocks
	int gridSide = sqrt(numberOfBlocks);
	return dim3(gridSide, gridSide);
}


/**
 * Method that generates the computing grid 
 * If any block cannot be launched, the program will abort
 * @param numberOfPairsInWindow: number of pixel pairs that belongs to each window
 * @param featureSize: memory space required by the values that will be computed
 * @param imgRows: the number of the rows composing the image
 * @param imgCols: the number of the columns composing the image
 * @param verbose: print extra info on the memory consumed
 */
dim3 getGrid(int numberOfPairsInWindow, size_t featureSize, int imgRows, 
	int imgCols, bool verbose){
 	dim3 Blocks = getBlockConfiguration();

	// Generating the grid from image dimensions
	dim3 Grid = getGridFromImage(imgRows, imgCols);

	// Checking if there is enough space on the GPU to allocate the working areas
	int numberOfBlocks = Grid.x * Grid.y;
	int numberOfThreadsPerBlock = Blocks.x * Blocks.y;
	int numberOfThreads = numberOfThreadsPerBlock * numberOfBlocks;
	if(! checkEnoughWorkingAreaForThreads(numberOfPairsInWindow, 
		numberOfThreads, featureSize, verbose))
	{
		Grid = getGridFromAvailableMemory(numberOfPairsInWindow, featureSize);
		// Getting the total number of threads and checking if the gpu memory is sufficient
		numberOfBlocks = Grid.x * Grid.y;
		numberOfThreads = numberOfThreadsPerBlock * numberOfBlocks;
		checkEnoughWorkingAreaForThreads(numberOfPairsInWindow, 
			numberOfThreads, featureSize, verbose);
	}
	if(verbose)
		printGPULaunchConfiguration(Grid, Blocks);
	return Grid;
}

/**
 * Method called by each thread when it needs to know its own unique
 * index inside the kernel launching configuration
 */
__device__ int getGlobalIdx_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x)	+ threadIdx.x;
	return threadId;
}


/**
 * Method invoked by each thread to obtain the reference to their own memory,
 * entirely pre-allocated on the host, needed for their computation
 * @param globalWorkArea: reference to the global, allocated by host, memory
 * that each thread will use to do their job
 * @param threadId: unique thread id inside the launch configuration
 */
__device__ WorkArea adjustThreadWorkArea(WorkArea globalWorkArea, int threadId){
	// Each one of these data structures allows one thread to work
	GrayPair* grayPairs = globalWorkArea.grayPairs;
	AggregatedGrayPair* summedPairs = globalWorkArea.summedPairs;
	AggregatedGrayPair* subtractedPairs = globalWorkArea.subtractedPairs;
	AggregatedGrayPair* xMarginalPairs = globalWorkArea.xMarginalPairs;
	AggregatedGrayPair* yMarginalPairs = globalWorkArea.yMarginalPairs;

	// Thread shift for accessing its memory
	int pointerStride = globalWorkArea.numberOfElements * threadId;

	// Pointing to its own memory
	grayPairs += pointerStride;
	summedPairs += pointerStride;
	subtractedPairs += pointerStride;
	xMarginalPairs += pointerStride;
	yMarginalPairs += pointerStride;

	WorkArea wa(globalWorkArea.numberOfElements, grayPairs, summedPairs,
				subtractedPairs, xMarginalPairs, yMarginalPairs, 
				globalWorkArea.output);
	return wa;
}

/**
 * Kernel computes all the features in each window of the image. Each
 * window will be computed by an autonomous thread of the grid
 * @param pixels: pixel intensities of the analyzed image
 * @param img: image metadata
 * @param globalWorkArea: class that stores the pointers to the pre-allocated 
 * space that will contain the arrays of representations that each thread will
 * use to perform its computation
 */
__global__ void computeFeatures(unsigned int * pixels, 
	ImageData img, Window windowData,  
	WorkArea globalWorkArea){

	// Memory location on which the thread will work
	int threadUniqueId = getGlobalIdx_2D();
	WorkArea wa = adjustThreadWorkArea(globalWorkArea, threadUniqueId);

	// Getting X and Y starting coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// If a thread needs to compute more than one window 
	// How many shift right for reaching the next window to compute
	int colsStride =  gridDim.x * blockDim.x; 
	// How many shift down for reaching the next window to compute
	int rowsStride =  gridDim.y * blockDim.y;

	// Considering the borders and the consequent padding
	int appliedBorders = img.getBorderSize();
	int originalImageRows = img.getRows() - 2 * appliedBorders;
    int originalImageCols = img.getColumns() - 2 * appliedBorders;
	
	/* If no border is applied, the windows on the borders need to be excluded because
	no pixel pair is available (similarly to the MatLab graycomatrix) */
    if(appliedBorders == 0){
    	originalImageRows -= windowData.side;
    	originalImageCols -= windowData.side;
    }

	// Creating the local window information
	Window actualWindow {windowData.side, windowData.distance,
								 windowData.directionType, windowData.symmetric};
	for(int i = y; i < originalImageRows; i+= rowsStride){
		for(int j = x; j < originalImageCols ; j+= colsStride){
			// Providing the relative offset (starting point) of the window according to the image
			actualWindow.setSpatialOffsets(i + appliedBorders, j + appliedBorders);
			// Launching the computation of features on the window
			WindowFeatureComputer wfc(pixels, img, actualWindow, wa);
		}
	}
}

