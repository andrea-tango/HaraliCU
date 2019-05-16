#include <iostream>
#include <fstream>

#include "ImageFeatureComputer.h"


ImageFeatureComputer::ImageFeatureComputer(const ProgramArguments& progArg)
:progArg(progArg){}

/**
 * Displays a set of info about the feature computation of the input image
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
    	cout << endl << "* Image loaded * ";
    checkOptionCompatibility(progArg, image);
    // Print computation info to cout
	printInfo(imgData, progArg.windowSize);
	if(verbose) {
		// Additional info on memory allocation
		printExtimatedSizes(imgData);
	}

	int realImageRows = image.getRows() - 2 * getAppliedBorders();
    int realImageCols = image.getColumns() - 2 * getAppliedBorders();
      
	// Computing every feature
	if(verbose)
		cout << "* COMPUTING features * " << endl;
	vector<vector<WindowFeatures>> fs= computeAllFeatures(image.getPixels().data(), imgData);
	vector<vector<FeatureValues>> formattedFeatures = getAllDirectionsAllFeatureValues(fs);
	if(verbose)
		cout << "* Features computed * " << endl;

	// Saving result to file
	if(verbose)
		cout << "* Saving features to files *" << endl;
	saveFeaturesToFiles(realImageRows, realImageCols, formattedFeatures);

	// Saving feature images
	if(progArg.createImages)
	{
		if(verbose)
			cout << "* Creating feature images *" << endl;
		// Computing how many features will be used for creating the image
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

        // Copying all the values
        vector<double> singleDirectionFeatures(windowResultsStartingPoint,
        		windowResultsStartingPoint + windowResultsSize);
		output[k][0] = singleDirectionFeatures;
	}

    return output;
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
	// Creating the metadata of each window that will be created
	Window windowData = Window(progArg.windowSize, progArg.distance, progArg.directionType, progArg.symmetric);

	// Get dimensions of the original image without borders
    int originalImageRows = img.getRows() - 2 * getAppliedBorders();
    int originalImageCols = img.getColumns() - 2 * getAppliedBorders();

    // Pre-Allocation of the working areas

	// How many windows are needed to be allocated
    int numberOfWindows = (originalImageRows * originalImageCols);
    // How many directions are needed to be allocated for each window
    short int numberOfDirs = 1;
    // How many feature values are needed to be allocated for each direction
    int featuresCount = Features::getSupportedFeaturesCount();

    // Pre-allocating the array that will store the features
    size_t featureSize = numberOfWindows * numberOfDirs * featuresCount * sizeof(double);
    double* featuresList = (double*) malloc(featureSize);
    if(featuresList == NULL){
        cerr << "FATAL ERROR! Not enough mallocable memory on the system" << endl;
        exit(3);
    }

    // 	Pre-allocation of the working area
    int extimatedWindowRows = windowData.side; // 0° has all rows
    int extimateWindowCols = windowData.side - (windowData.distance); // at least 1 column is lost
    int numberOfPairsInWindow = extimatedWindowRows * extimateWindowCols;

	GrayPair* elements = (GrayPair*) malloc(sizeof(GrayPair)
	        * numberOfPairsInWindow);
	AggregatedGrayPair* summedPairs = (AggregatedGrayPair*) malloc(sizeof(AggregatedGrayPair)
	        * numberOfPairsInWindow );
    AggregatedGrayPair* subtractedPairs = (AggregatedGrayPair*) malloc(sizeof(AggregatedGrayPair)
            * numberOfPairsInWindow);
    AggregatedGrayPair* xMarginalPairs = (AggregatedGrayPair*) malloc(sizeof(AggregatedGrayPair)
            * numberOfPairsInWindow);
    AggregatedGrayPair* yMarginalPairs = (AggregatedGrayPair*) malloc(sizeof(AggregatedGrayPair)
            * numberOfPairsInWindow);

    WorkArea wa(numberOfPairsInWindow, elements, summedPairs,
                subtractedPairs, xMarginalPairs, yMarginalPairs, featuresList);

    /* If no border is applied, the window on the borders need to be excluded because
		no pixel pair is available (according to the MatLab graycomatrix built-in function)*/
    if(progArg.borderType == 0){
    	originalImageRows -= windowData.side;
    	originalImageCols -= windowData.side;
    }

    // Sliding windows over the image
    for(int i = 0; i < originalImageRows ; i++)
    {
        for(int j = 0; j < originalImageCols ; j++)
        {
            // Creating the local window information
            Window currentWindow {windowData.side, windowData.distance,
                                 progArg.directionType, windowData.symmetric};
            // Seeting the relative offset (starting point) inside the image for the current sliding window
            currentWindow.setSpatialOffsets(i + getAppliedBorders(), j + getAppliedBorders());
            // Launching the computation of the features on the window
            WindowFeatureComputer wfc(pixels, img, currentWindow, wa);
        }

	}

	// Yielding the data structure
    vector<vector<vector<double>>> output =
            formatOutputResults(featuresList, numberOfWindows, featuresCount);

	free(featuresList);
	wa.release();
	return output;
}



/**
 * This method will extract the results from each window
 * @param imageFeatures: array (1 for each window) of array (1 for each
 * computed direction) of array of doubles (1 for each feature)
 * @return array (1 for each direction) of array (1 for each feature) of all
 * the values computed of that feature
 * Es. <Entropy  (0.1, 0.2, 3, 4 , ...)>
 */
vector<vector<FeatureValues>> ImageFeatureComputer::getAllDirectionsAllFeatureValues(const vector<vector<WindowFeatures>>& imageFeatures){
	vector<FeatureNames> supportedFeatures = Features::getAllSupportedFeatures();
	vector<vector<FeatureValues>> output(1);

	// for each computed direction
	// one external vector cell for each of the 18 features
	// each cell has all the values of that feature
	vector<FeatureValues> featuresInDirection(supportedFeatures.size());

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
    // For each computed direction
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

	// Checking if dimensions are compatible
	if(featureValues.size() != imageSize){
		cerr << "Fatal Error! Couldn't create the image; size unexpected " << featureValues.size();
		exit(-2);
	}

	// Creating a 2D matrix of double grayPairs
	Mat_<double> imageFeature = ImageLoader::createDoubleMat(rowNumber, colNumber, featureValues);
    ImageLoader::saveImage(imageFeature, filePath);
}





