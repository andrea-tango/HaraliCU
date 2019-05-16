
#ifndef IMAGEFEATURECOMPUTER_H_
#define IMAGEFEATURECOMPUTER_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include "ImageLoader.h"
#include "ProgramArguments.h"
#include "Utils.h"
#include "WindowFeatureComputer.h"

/**
 * Array of all the features that can be extracted simultaneously from a
 * window
 */
typedef vector<double> WindowFeatures;
/**
 * Array of all the features that can be extracted simultaneously from a
 * direction in a window
 */
typedef vector<double> FeatureValues;

using namespace cv;

/**
 * This class performs three main tasks:
 * - Reading and transforming the image according to the options provided
 * - Computing all the features in all the windows that can be created in the
 * image according to the provided options
 * - Organizing and saving the results
 */
class ImageFeatureComputer {
public:
	/**
	 * Initialize the class
	 * @param progArg: parameters of the current feature extraction problem
	 */
	ImageFeatureComputer(const ProgramArguments& progArg);
	/**
	 * This method reads the image, computes the features, re-arranges the
	 * results and saves them as need on the file system
	 */
	void compute();
	/**
     * This method computes all the features for every window for the
     * number of the provided directions
     * @param pixels: pixel intensities of the provided image
     * @param img: image metadata
     * @return array (one for each window) of array (one for each computed direction)
     * of array of doubles (one for each feature)
     */
    vector<vector<WindowFeatures>> computeAllFeatures(unsigned int * pixels, const ImageData& img);

    // EXTRACTING THE RESULTS
    /**
     * This method extracts the results from each window
     * @param imageFeatures: array (one for each window) of array (one for each
     * computed direction) of array of doubles (one for each feature)
     * @return array (one for each direction) of array (one for each feature) of all
     * the values computed of that feature
     * Es. <Entropy , (0.1, 0.2, 3, 4 , ...)>
     */
    vector<vector<vector<double>>> getAllDirectionsAllFeatureValues(const vector<vector<WindowFeatures>>& imageFeatures);

	// SAVING THE RESULTS ON FILES
	/**
	 * This method will save on different folders, all the features values
	 * computed for each directions of the image
	 * @param imageFeatures
	 */
	void saveFeaturesToFiles(const int rowNumber, const int colNumber, const vector<vector<vector<double>>>& imageFeatures);

    // IMAGING
    /**
     * This method produces and saves all the images associated with each feature
     * for each direction
     * @param rowNumber: how many rows each image will have
     * @param colNumber: how many columns each image will have
     * @param imageFeatures
     */
    void saveAllFeatureImages(int rowNumber,  int colNumber, const vector<vector<FeatureValues>>& imageFeatures);


private:
	ProgramArguments progArg;

	// SUPPORT FILESAVE methods
	/**
	 * This method saves in the given folder, all the values of all
	 * the features computed for 1  directions
	 * @param imageDirectedFeatures: all the values computed for each feature
	 * in one direction of the image
	 * @param outputFolderPath
	 */
	void saveDirectedFeaturesToFiles(const int rowNumber, const int colNumber, 
		const vector<vector<double>>& imageDirectedFeatures, const string& outputFolderPath);
	/**
	 * This method will save into the given folder, all the values for a feature
     * computed for one directions
	 * @param imageFeatures: all the feature values of a feature
	 * @param path
	 */
	void saveFeatureToFile(const int rowNumber, const int colNumber, const pair<FeatureNames,
		vector<double>>& imageFeatures, const string path);

	// SUPPORT IMAGING methods
	/**
	 * This method will produce and save all the images associated with
	 * each feature in one direction
	 * @param rowNumber: number of rows in the output image
	 * @param colNumber: number of columns in the output image
	 * @param imageFeatures: all the values computed for each feature of the image
	 * @param outputFolderPath: path for the output image
	 */
	void saveAllFeatureDirectedImages(int rowNumber,  int colNumber,
			const vector<vector<double>> &imageFeatures, const string& outputFolderPath);
	/**
	 * This method produces and saves in the filesystem the image associated with
	 * a feature in one direction
	 * @param rowNumber: number of rows in the output image
	 * @param colNumber: number of columns in the output image
	 * @param featureValues: values that will be the intensities values of the
	 * image
	 * @param outputFilePath: path for the output image
	 */
	void saveFeatureImage(int rowNumber,  int colNumber,
			const vector<double>& featureValues, const string& outputFilePath);

	/**
	 * Utility method
	 * @return applied border to the original image read
	 */
	int getAppliedBorders();
	/**
	 * Displays a set of information about the computation of the provided image
	 * @param imgData
	 * @param padding
	 */
    void printInfo(ImageData imgData, int padding);
	/**
	 * Displays the memory space used while computing the features
	 * @param imgData
	 * @param padding
	 */
	void printExtimatedSizes(const ImageData& img);
};



#endif /* IMAGEFEATURECOMPUTER_H_ */
