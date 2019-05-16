#ifndef FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
#define FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H

#include "FeatureComputer.h"
#include "Direction.h"

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

using namespace std;

/**
 * This class computes the features for a direction of the window of interest
 */
class WindowFeatureComputer {

public:
    /**
     * Constructs the class that computes the features for a window
     * @param pixels: composing the entire image
     * @param img: metadata about the image (image matrix dimensions,
     * minGrayLevel, maxGrayLevel, borders)
     * @param wd: metadata about the window of interest (size, starting
     * point in the image)
     * @param wa: memory location where this object will create the arrays of
     * representation needed for computing its features
     */
    WindowFeatureComputer(unsigned int * pixels, const ImageData& img, const Window& wd, WorkArea& wa);
    /**
     * Features computed in the specified direction
     */
    void computeWindowFeatures();

private:
    /**
     * Pixels composing the image
     */
    const unsigned int * pixels;
    /**
     * Metadata of the image (dimensions, minGrayLevel, maxGrayLevel)
     */
    ImageData image;
    /**
     * Window of interest where the glcm is computed
     */
    Window windowData;
    /**
     * Memory location used for computing the feature regarding the current window
     */
    WorkArea& workArea;
};


#endif //FEATUREEXTRACTOR_WINDOWFEATURECOMPUTER_H
