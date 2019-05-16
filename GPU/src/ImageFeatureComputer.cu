#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include "CudaFunctions.h"
#include "ImageFeatureComputer.h"


ImageFeatureComputer::ImageFeatureComputer(const ProgramArguments& progArg)
:progArg(progArg){}

/**
 * Displays a set of information about the computation of the provided image
 * @param imgData
 * @param padding
 */
void ImageFeatureComputer::printInfo(const ImageData imgData, int border)
{
	cout << endl << "- Input image: " << progArg.imagePath;
	cout << endl << "- Output folder: " << progArg.outputFolder;
	int rows = imgData.getRows() - 2 * getAppliedBorders();
	int cols = imgData.getColumns() - 2 * getAppliedBorders();
	cout << endl << "- Rows: " << rows << " - Columns: " << cols << " - Pixel count: " << rows*cols;
	cout << endl << "- Min gray level: " << imgData.getMinGrayLevel();
	cout << endl << "- Max gray level: " << imgData.getMaxGrayLevel();
	cout << endl << "- Distance: " << progArg.distance;
	cout << endl << "- Window size: " << progArg.windowSize;
	if(progArg.symmetric)
		cout << endl << "- GLCM symmetry enabled";
	else
		cout << endl << "- GLCM symmetry disabled";
}

/**
 * Displays the memory space used while computing the features
 * @param imgData
 * @param padding
 */
void ImageFeatureComputer::printExtimatedSizes(const ImageData& img){
    int numberOfRows = img.getRows() - progArg.windowSize + 1;
    int numberOfColumns = img.getColumns() - progArg.windowSize + 1;
    int numberOfWindows = numberOfRows * numberOfColumns;
    int supportedFeatures = Features::getSupportedFeaturesCount();

    int featureNumber = numberOfWindows * supportedFeatures;
    cout << endl << "* Size estimation * " << endl;
    cout << "\tTotal features number: " << featureNumber << endl;
    int featureSize = (((featureNumber * sizeof(double))
                        /1024)/1024);
    cout << "\tTotal features weight: " <<  featureSize << " MB" << endl;
}

/**
 * Checks if all the options are coherent with the read image
 * @param progArg
 * @param img
 */
void checkOptionCompatibility(ProgramArguments& progArg, const Image img){
    int imageSmallestSide = img.getRows();
    if(img.getColumns() < imageSmallestSide)
        imageSmallestSide = img.getColumns();
    if(progArg.windowSize > imageSmallestSide){
        cout << "WARNING! The window side specified with the option -w"
                "exceeds the smallest dimension (" << imageSmallestSide << ") of the image read!" << endl;
        cout << "Window side is corrected to (" << imageSmallestSide << ")" << endl;
        progArg.windowSize = imageSmallestSide;
    }

}

/**
 * Utility method
 * @return applied border to the original image read
 */
int ImageFeatureComputer::getAppliedBorders(){
    int bordersToApply = 0;
    if(progArg.borderType != 0 )
        bordersToApply = progArg.windowSize;
    return bordersToApply;
}

/**
 * This method reads the image, computes the features, re-arranges the
 * results, and saves them in the file system
 */
void ImageFeatureComputer::compute(){
	bool verbose = progArg.verbose;

	// Image from imageLoader
	Image image = ImageLoader::readImage(progArg.imagePath, progArg.borderType,
                                         getAppliedBorders(), progArg.quantize,
                                         progArg.quantizationMax);
	ImageData imgData(image, getAppliedBorders());
	if(verbose)
    	cout << "* Image loaded * ";
    checkOptionCompatibility(progArg, image);
    // Print computation info to cout
	printInfo(imgData, progArg.distance);
	if(verbose) {
		// Additional info on memory occupation
		printExtimatedSizes(imgData);
	}

	int realImageRows = image.getRows() - 2 * getAppliedBorders();
    int realImageCols = image.getColumns() - 2 * getAppliedBorders();

	if(verbose)
		cout << "* COMPUTING features * " << endl;
	vector<vector<WindowFeatures>> fs= computeAllFeatures(image.getPixels().data(), imgData);
	vector<vector<FeatureValues>> formattedFeatures = getAllDirectionsAllFeatureValues(fs);
	if(verbose)
		cout << "* Features computed * " << endl;

	// Save result to file
	if(verbose)
		cout << "* Saving features to files *" << endl;
	saveFeaturesToFiles(realImageRows, realImageCols, formattedFeatures);

	// Save feature images
	if(progArg.createImages)
	{
		if(verbose)
			cout << "* Creating feature images *" << endl;
		// Compute how many features will be used for creating the image
        saveAllFeatureImages(realImageRows, realImageCols, formattedFeatures);
	}
	if(verbose)
		cout << "* DONE * " << endl;
}



/**
 * This method re-arranges (i.e., de-linearizes) all the feature values
 * computed per-window into a structure storing the
 * values of a feature for each window of the image, and for each distance and direction
 * @param featureValues: all the computed features, window-per-window
 * @param numberOfWindows: the number of the windows involved in the feature computation
 * @param featuresCount: the number of features computed in each window
 * @return structured array (windowFeatures[] wherein each cell has
 * directionFeatures[] composed of double[] = features)
 */
vector<vector<vector<double>>> formatOutputResults(const double* featureValues,
												   const int numberOfWindows, const int featuresCount){
	// For each window, an array of directions,
    // For each direction, an array of features
	vector<vector<vector<double>>> output(numberOfWindows,
										  vector<vector<double>>(1, vector<double> (featuresCount)));
	// How many double values fit into a window
	int windowResultsSize = featuresCount;

	for (int k = 0; k < numberOfWindows; ++k) {
		int windowOffset = k * windowResultsSize;
		const double* windowResultsStartingPoint = featureValues + windowOffset;

		// Copying each of the values
		vector<double> singleDirectionFeatures(windowResultsStartingPoint,
				windowResultsStartingPoint + windowResultsSize);
		output[k][0] = singleDirectionFeatures;
	}

	return output;
}

/**
 * This method allocates the memory working area according to numberOfPairs
 * and numberOfThreads.
 */
WorkArea generateGlobalWorkArea(int numberOfPairs, int numberOfThreads,
	double* d_featuresList){
	
	int totalNumberOfPairs = numberOfPairs * numberOfThreads;
	
	// Each one of these data structures allows one thread to work
	GrayPair* d_grayParis;
	AggregatedGrayPair* d_summedPairs;
	AggregatedGrayPair* d_subtractedPairs;
	AggregatedGrayPair* d_xMarginalPairs;
	AggregatedGrayPair* d_yMarginalPairs;

	cudaCheckError(cudaMalloc((void**) &d_grayParis, sizeof(GrayPair) * 
		totalNumberOfPairs));
	cudaCheckError(cudaMalloc((void**) &d_summedPairs, sizeof(AggregatedGrayPair) * 
		totalNumberOfPairs));
	cudaCheckError(cudaMalloc((void**) &d_subtractedPairs, sizeof(AggregatedGrayPair) * 
		totalNumberOfPairs));
	cudaCheckError(cudaMalloc((void**) &d_xMarginalPairs, sizeof(AggregatedGrayPair) * 
		totalNumberOfPairs));
	cudaCheckError(cudaMalloc((void**) &d_yMarginalPairs, sizeof(AggregatedGrayPair) * 
		totalNumberOfPairs));

	WorkArea wa(numberOfPairs, d_grayParis, d_summedPairs,
				d_subtractedPairs, d_xMarginalPairs, d_yMarginalPairs, d_featuresList);
	return wa;
}


/**
 * This method computes all the features for every window according to the
 * number of the provided directions
 * @param pixels: pixel intensities of the provided image
 * @param img: image metadata
 * @return array (one for each window) of array (one for each computed direction)
 * of array of double numbers (one for each feature)
 */
vector<vector<WindowFeatures>> ImageFeatureComputer::computeAllFeatures(unsigned int * pixels, const ImageData& img){
	bool verbose = progArg.verbose;
	if(verbose)
		queryGPUData();

	// Create window structure that will be given to threads
	Window windowData = Window(progArg.windowSize, progArg.distance, 
		progArg.directionType, progArg.symmetric);

	// Get dimensions of the original image without borders
  	int realImageRows = img.getRows() - 2 * getAppliedBorders();
    int realImageCols = img.getColumns() - 2 * getAppliedBorders();

	// How many windows need to be allocated
    int numberOfWindows = (realImageRows * realImageCols);
	// How many directions need to be allocated for each window
	short int numberOfDirs = 1;
	// How many feature values need to be allocated for each direction
	int featuresCount = Features::getSupportedFeaturesCount();

	// Pre-allocating the array that will contain features
	size_t featureSize = numberOfWindows * numberOfDirs * featuresCount * sizeof(double);
	double* featuresList = (double*) malloc(featureSize);
	if(featuresList == NULL){
		cerr << "FATAL ERROR! Not enough mallocable memory on the system" << endl;
		exit(3);
	}

	// Allocating the GPU space to store the results
	double* d_featuresList;
	cudaCheckError(cudaMalloc((void**) &d_featuresList, featureSize));

	// Compute how many elements will be stored in each thread working memory
	int extimatedWindowRows = windowData.side; // 0° has all rows
	int extimateWindowCols = windowData.side - (windowData.distance); // at least 1 column is absent
	int numberOfPairsInWindow = extimatedWindowRows * extimateWindowCols;

	// COPY the image pixels to the GPU
	unsigned int * d_pixels;
	cudaCheckError(cudaMalloc((void**) &d_pixels, sizeof(unsigned int) * img.getRows() * img.getColumns()));
	cudaCheckError(cudaMemcpy(d_pixels, pixels,
			sizeof(unsigned int) * img.getRows() * img.getColumns(),
			cudaMemcpyHostToDevice));

	// Try to achieve higher performance
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	// Get Grid and block configuration
	dim3 Blocks = getBlockConfiguration(); 
	dim3 Grid = getGrid(numberOfPairsInWindow, featureSize, realImageRows, realImageCols, verbose);

	int numberOfThreads = Grid.x * Grid.y * Blocks.x * Blocks.y;

	// CPU pre-allocation of the device memory consumed by threads
	WorkArea globalWorkArea = generateGlobalWorkArea(numberOfPairsInWindow, 
		numberOfThreads, d_featuresList);

	// Launching the kernel
	computeFeatures<<<Grid, Blocks>>>(d_pixels, img, windowData, 
			globalWorkArea);

	cudaDeviceSynchronize();

	// Checking if everything is correct
	checkKernelLaunchError();

	// Copying back the results from GPU
	cudaCheckError(cudaMemcpy(featuresList, d_featuresList,
			featureSize,
			cudaMemcpyDeviceToHost));

	// Yielding the data structure
	// Windows[] with Directions[] with feature values
	vector<vector<vector<double>>> output =
			formatOutputResults(featuresList, numberOfWindows, featuresCount);

	free(featuresList); // Releasing the CPU feature array
	globalWorkArea.release(); // Releasing device work memory
	cudaFree(d_featuresList); // Releasing Gpu feature array
	cudaFree(d_pixels); // Releasing the image on the GPU
	
	return output;
}


/**
 * This method will extract the results from each window
 * @param imageFeatures: array (1 for each window) of array (1 for each
 * computed direction) of array of doubles (1 for each feature)
 * @return array (1 for each direction) of array (1 for each feature) of all
 * the values computed of that feature
 * Es. <Entropy, (0.1, 0.2, 3, 4 , ...)>
 */
vector<vector<vector<double>>> ImageFeatureComputer::getAllDirectionsAllFeatureValues(const vector<vector<vector<double>>>& imageFeatures){
	vector<FeatureNames> supportedFeatures = Features::getAllSupportedFeatures();
	vector<vector<vector<double>>> output(1);

	// For each computed direction
	// one external vector cell for each of the 18 features
	// each cell has all the values of that feature
	vector<vector<double>> featuresInDirection(supportedFeatures.size());

	// for each computed window
	for (int i = 0; i < imageFeatures.size() ; ++i) {
		// for each supported feature
		for (int k = 0; k < supportedFeatures.size(); ++k) {
			FeatureNames currentFeature = supportedFeatures[k];
			// Push the value found in the output list for that direction
			featuresInDirection[currentFeature].push_back(imageFeatures.at(i).at(0).at(currentFeature));
		}

	}
	output[0] = featuresInDirection;

	return output;
}


/**
 * This method saves, in different folders, all the features values
 * computed for each directions of the image
 * @param imageFeatures
 */
void ImageFeatureComputer::saveFeaturesToFiles(const int rowNumber, const int colNumber, 
	const vector<vector<FeatureValues>>& imageFeatures)
{
    int dirType = progArg.directionType;

    string outFolder = progArg.outputFolder;
    Utils::createFolder(outFolder);
    string foldersPath[] ={ "/Values0/", "/Values45/", "/Values90/", "/Values135/"};

    // Firstly, the folder is created
    string outputDirectionPath = outFolder + foldersPath[dirType -1];
	Utils::createFolder(outputDirectionPath);
    saveDirectedFeaturesToFiles(rowNumber, colNumber, imageFeatures[0], outputDirectionPath);
}

/**
 * This method saves in the given folder, the values of all
 * the features computed for the given distance and orientation
 * @param imageDirectedFeatures: all the values computed for each feature
 * for a given distance and orientation
 * @param outputFolderPath
 */
void ImageFeatureComputer::saveDirectedFeaturesToFiles(const int rowNumber, const int colNumber,
	const vector<FeatureValues>& imageDirectedFeatures, const string& outputFolderPath)
{
	vector<string> fileDestinations = Features::getAllFeaturesFileNames();

	// for each feature
	for(int i = 0; i < imageDirectedFeatures.size(); i++) {
		string newFileName(outputFolderPath); // create the right file path
		pair<FeatureNames , FeatureValues> featurePair = make_pair((FeatureNames) i, imageDirectedFeatures[i]);
		saveFeatureToFile(rowNumber, colNumber, featurePair, newFileName.append(fileDestinations[i]));
	}
}

/**
 * This method saves in the given folder, all the values for one feature
 * computed for the given distance and orientation
 * @param featurePair: all the feature values for a given feature
 * @param filePath
 */
void ImageFeatureComputer::saveFeatureToFile(const int rowNumber, const int colNumber, const pair<FeatureNames,
	vector<double>>& featurePair, string filePath)
{
	// Open the file
	ofstream file;
	file.open(filePath.append(".txt"));
	if(file.is_open())
	{
		
		int idx = 0;

		for(int i=0; i < rowNumber; i++)
		{
			for(int j=0; j < colNumber; j++)
			{
				if(j < colNumber-1)
					file << featurePair.second[idx] << "\t";
				else
				{
					if(i < rowNumber-1)
						file << featurePair.second[idx] << "\n";
					else
						file << featurePair.second[idx];
				}

				idx++;
			}
		}

		file.close();
	} else{
		cerr << "Couldn't save the feature values to file" << endl;
	}

}

// IMAGING
/**
 * This method produces and saves all the images associated with each feature
 * for every direction
 * @param rowNumber: the number of rows of the output image
 * @param colNumber: the number of columns of the output image
 * @param imageFeatures
 */
void ImageFeatureComputer::saveAllFeatureImages(const int rowNumber,
		const int colNumber, const vector<vector<FeatureValues>>& imageFeatures){
    int dirType = progArg.directionType;

    string outFolder = progArg.outputFolder;
    string foldersPath[] ={ "/Images0/", "/Images45/", "/Images90/", "/Images135/"};
    string outputDirectionPath = outFolder + foldersPath[dirType -1];
	Utils::createFolder(outputDirectionPath);
    // For each direction computed
    saveAllFeatureDirectedImages(rowNumber, colNumber, imageFeatures[0],
                outputDirectionPath);
}

/**
 * This method will produce and save all the images associated with
 * each feature for a given distance and orientation
 * @param rowNumber: the number of rows of the output image
 * @param colNumber: the number of columns of the output image
 * @param imageFeatures: all the values computed for each feature of the image
 * @param outputFolderPath: output path for saving the image
 */
void ImageFeatureComputer::saveAllFeatureDirectedImages(const int rowNumber,
		const int colNumber, const vector<FeatureValues>& imageDirectedFeatures, const string& outputFolderPath){

	vector<string> fileDestinations = Features::getAllFeaturesFileNames();

	// For each feature
	for(int i = 0; i < imageDirectedFeatures.size(); i++) {
		string newFileName(outputFolderPath);
		saveFeatureImage(rowNumber, colNumber, imageDirectedFeatures[i], newFileName.append(fileDestinations[i]));
	}
}

/**
 * This method produces and saves in the filesystem the image associated with
 * a feature for a given distance and orientation
 * @param rowNumber: the number of rows of the output image
 * @param colNumber: the number of columns of the output image
 * @param featureValues: values representing the intensity values of the output image
 * @param outputFilePath: output path for saving the image
 */
void ImageFeatureComputer::saveFeatureImage(const int rowNumber,
		const int colNumber, const FeatureValues& featureValues,const string& filePath){
	typedef vector<WindowFeatures>::const_iterator VI;

	int imageSize = rowNumber * colNumber;

	// Check if dimensions are compatible
	if(featureValues.size() != imageSize){
		cerr << "Fatal Error! Couldn't create the image; size unexpected " << featureValues.size();
		exit(-2);
	}

	// Create]ing a 2D matrix of double grayPairs
	Mat_<double> imageFeature = ImageLoader::createDoubleMat(rowNumber, colNumber, featureValues);
    ImageLoader::saveImage(imageFeature, filePath);
}

