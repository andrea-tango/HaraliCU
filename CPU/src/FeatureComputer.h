#ifndef FEATUREEXTRACTOR_FEATURECOMPUTER_H
#define FEATUREEXTRACTOR_FEATURECOMPUTER_H

#include <vector>
#include "GLCM.h"
#include "Features.h"

/**
 * This class computes 18 Haralick features for a single sliding window, given a
 * particular distance and orientation
 */
 class FeatureComputer {
public:
    /**
     * Initializes the object and calculated the features of interest
     * @param pixels: pixels of the whole image
     * @param img: metadata about the image (matrix size, maxGrayLevel, borders)
     * @param shiftRows: shift on the y-axis to apply for denoting the neighbor
     * pixel
     * @param shiftColumns: shift on the x-axis to apply for denoting the neighbor
     * pixel
     * @param windowData: metadata about the window of interest (size, starting
     * point in the image, etc.)
     * @param wa: memory location where this object will create the arrays
     * needed for computing its features
     */
    FeatureComputer(const unsigned int * pixels, const ImageData& img,
            int shiftRows, int shiftColumns, const Window& windowData,
            WorkArea& wa);
private:
    // Initialize of the GLCM
    /**
     * Pixels composing the image
     */
    const unsigned int * pixels;
    /**
     * Metadata about the image (dimensions, maxGrayLevel)
     */
    ImageData image;
    /**
     * Window of interest wherein the features are computed
     */
    Window windowData;
    /**
     * Memory working area used for computing the requested features
     */
    WorkArea& workArea;
    /**
     * Output variable for the computed features
     */
    double * featureOutput;
    /**
     * Offset to identify the sliding window of interest for the
     * object; this information will be used for storing the results in the
     * correct memory location
     */
    int outputWindowOffset;
    /**
     * Computes the offset to identify the sliding window of interest for the
     * object; this information will be used for storing the results in the
     * correct memory location
     */
    void computeOutputWindowFeaturesIndex();

    /**
     * Launches the computation of all features supported
     */
    void computeDirectionalFeatures();
    /**
     * Computes the features that can be extracted from the GLCM of the image;
     * this method will store the results automatically
     * @param metaGLCM: object of class GLCM that will provide gray pairs
     * @param features: variable that stores the results; this pointer is obtained
     * according to the work area
     */
    void extractAutonomousFeatures(const GLCM& metaGLCM, double* features);
    /**
     * Computes the features that can be extracted from the AggregatedPairs
     * obtained by adding gray levels of the pixel pairs;
     * this method will store the results automatically
     * @param metaGLCM: object of the class GLCM that will provide gray pairs
     * @param features: storing variable; this pointer is obtained
     * according to the work area
     */
    void extractSumAggregatedFeatures(const GLCM& metaGLCM, double* features);
    /**
     * Computes the features that can be extracted from the AggregatedPairs
     * obtained by subtracting gray levels of the pixel pairs;
     * this method will store the results automatically
     * @param metaGLCM: object of the class GLCM that will provide gray pairs
     * @param features: storing variable; this pointer is obtained
     * according to the work area
     */
    void extractDiffAggregatedFeatures(const GLCM& metaGLCM, double* features);
    /**
     * Computes the features that can be extracted from the AggregatedPairs
     * obtained by computing the marginal frequency of the gray levels of the
     * reference/neighbor pixels;
     * this method will store the results automatically
     * @param metaGLCM: object of class GLCM that will provide gray pairs
     * @param features: storing variable; this pointer is obtained
     * according to the work area
     */
    void extractMarginalFeatures(const GLCM& metaGLCM, double* features);

};

#endif //FEATUREEXTRACTOR_FEATURECOMPUTER_H
