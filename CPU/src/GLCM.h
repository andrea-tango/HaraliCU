#ifndef FEATUREEXTRACTOR_GLCM_H
#define FEATUREEXTRACTOR_GLCM_H

#include "GrayPair.h"
#include "AggregatedGrayPair.h"
#include "Window.h"
#include "ImageData.h"
#include "WorkArea.h"


using namespace std;

/**
 * This class generates all the elements needed to compute the features
 * from the pixel pairs of the image.
 */
class GLCM {
public:
    /**
     * The class GLCM contains all the gray pairs occurring in the window of interest
     */
    GrayPair* grayPairs;
    /**
     * The number of different gray pairs occurring in the window of interest.
     * It is necessary since the grayPairs array is pre-allocated with the
     * worst case number of elements
     */
    int effectiveNumberOfGrayPairs;
    /**
     * Array of pairs <k, frequency> storing the sum of the occurrences
     * of the gray level pair k
     */
    AggregatedGrayPair* summedPairs;
    /**
     * The number of different added gray pairs found in the window of
     * interest.
     * It is necessary since the grayPairs array is pre-allocated with the
     * worst case number of elements
     */
    int numberOfSummedPairs;
    /**
     * Array of pairs <k, frequency> storing the difference of the occurrences
     * of the gray level pair k
     */
    AggregatedGrayPair* subtractedPairs;
    /**
     * The number of different subtracted gray pairs found in the window of
     * interest.
     * It is necessary since the grayPairs array is pre-allocated with the
     * maximum number of elements in the worst case scenario
     */
    int numberOfSubtractedPairs;
    /**
     * Array of pairs <k, frequency> where k is the gray level of the reference
     * pixel in the pair
     */
    AggregatedGrayPair* xMarginalPairs;
    /**
    * The number of different x-marginal gray pairs found in the window of
    * interest.
    * It is necessary since the grayPairs array is pre-allocated with the
    * maximum number of elements in the worst case
    */
    int numberOfxMarginalPairs;
    /**
     * Array of pairs <k, frequency> where k is the gray level of the neighbor
     * pixel in the pair
     */
    AggregatedGrayPair* yMarginalPairs;
    /**
    * The number of different y-marginal gray pairs found in the window of
    * interest.
    * It is necessary since the grayPairs array is pre-allocated with the
    * maximum number of number of elements in the worst case
    */
    int numberOfyMarginalPairs;

     /**
      * Constructor of the GLCM that launches also the methods to generate
      * all the elements needed for extracting the features from them
      * @param pixels: composing the entire image
      * @param image: metadata of the image (matrix size,
      * maxGrayLevel, borders)
      * @param windowData: metadata of this window of interest from which
      * the glcm is computed
      * @param wa: memory working area where this object creates the arrays of
     * representation needed for computing the features
      */
    GLCM(const unsigned int * pixels, const ImageData& image, Window& windowData, WorkArea& wa);
    ~GLCM();

    // Getter methods exposed for the FeatureComputer class
    /**
     * Getter. Returns the number of pairs that belongs to the GLCM; this value
     * is used for computing the probability from the frequency of each item
     * @return the number of pairs that belongs to the GLCM
     */
    int getNumberOfPairs() const;
    /**
     * Getter. Returns the maximum grayLevel of the image
     * @return
     */
    int getMaxGrayLevel() const;

    /**
     * DEBUG METHOD. Prints a complete representation of the GLCM object and
     * every one of its array of grayPairs / aggregatedGrayPairs
     */
    void printGLCM() const;

private:
    /**
      * Pixels composing the image
      */
    const unsigned int * pixels;
    /**
     * Metadata of the image (dimensions, maxGrayLevel)
     */
    ImageData image;
    /**
     * Window of interest where the glcm is computed
     */
    Window windowData;
    /**
     * Memory working area used for computing this window's feature
     */
    WorkArea& workArea;

    /**
     * number of pairs that belongs to the GLCM
     */
    int numberOfPairs;

    /**
     * Computes the shift to apply at the column for denoting the pixels of each
     * pair of the glcm; it affects only 135° orientation
     * @return d (distance) pixels need to be ignored
     */
    int computeWindowColumnOffset();
    /**
    * Computes the shift to apply at the row for locating the pixels of each
    * pair of the glcm; it does not affect only 0° orientation
    * @return d (distance) pixels need to be ignored
    */
    int computeWindowRowOffset();
    // Geometric limits in the windows where this GLCM is computed
    /**
     * Geometric limit of the sub-window
     * @return the number of rows of the window to be considered
     */
    int getWindowRowsBorder() const;
    /**
    * Geometric limit of the sub-window
    * @return how many columns of the window need to be considered
    */
    int getWindowColsBorder() const;
    // Addressing methods to get pixels in the pair
    /**
     * Methods to get the reference pixel in each pair of the glcm
     * @param row in the sub-window of the reference pixel
     * @param col in the sub-window of the reference pixel
     * @param initialRowOffset see computeWindowRowOffset
     * @param initialColumnOffset see computeWindowColOffset
     * @return the index of the pixel in the array of pixels (linearized) of
     * the window
     */
    int getReferenceIndex(int row, int col, int initialRowOffset, int initialColumnOffset);
    /**
     * Methods to get the neighbor pixel in each pair of the glcm
     * @param row in the sub-window of the neighbor pixel
     * @param col in the sub-window of the neighbor pixel
     * @param initialColumnOffset see computeWindowColOffset
     * @return the index of the pixel in the array of pixels (linearized) of
     * the window
     */
    int getNeighborIndex(int row, int col, int initialColumnOffset);
    // Methods to build the glcm from input pixel and directional data
    /**
     * Method that adds a GrayPair into the pre-allocated memory.
     * It uses the convention that GrayPair (i=0, j=0, frequency=0) indicates
     * the available memory
     */
    void insertElement(GrayPair* elements, GrayPair actualPair,
            uint& lastInsertionPosition, bool symmetry);
    /**
     * Method that inserts an AggregatedGrayPair in the pre-allocated memory.
     * It uses the convention that AggregateGrayPair (k=0, frequency=0) indicates
     * available memory
     */
    void insertElement(AggregatedGrayPair* elements,
            AggregatedGrayPair actualPair, uint& lastInsertionPosition);
    /**
     * This method creates the array of GrayPairs
     */
    void initializeGlcmElements();
    // Representations useful for aggregated features
    /**
     * This method produces the two arrays of AggregatedPairs <k, frequency>
     * where k is the sum or difference of both grayLevels of a GrayPair.
     * This representation is used in computeSumXXX() and computeDiffXXX() features
     */
    void codifyAggregatedPairs();
    // Representation useful for HXY
    /**
     * This method produces the two arrays of AggregatedPairs <k, frequency>
     * where k is one grayLevel of GLCM and frequency is the "marginal" frequency of that level
     * (i.e., how many times k is present in all GrayPair<k, ?>)
     * This representation is used for computing features HX, HXY, HXY1, imoc
     */
    void codifyMarginalPairs() ;

    // debug printing methods
    /**
     * DEBUG METHOD. Prints a complete representation of the GCLM class and
     * the content of all its arrays of elements
     */
    void printGLCMData() const;
    /**
     * DEBUG METHOD. Prints all the grayPairs of the GLCM
     */
    void printGLCMElements() const;
    /**
     * DEBUG METHOD. Prints all the summedGrayPairs and subtractedGrayPairs
     */
    void printAggregated() const;
    /**
     * DEBUG METHOD. Prints all the xmarginalGrayPairs and ymarginalGrayPairs
     */
    void printMarginalProbabilityElements() const;
    void printGLCMAggregatedElements(bool areSummed) const;

};


#endif //FEATUREEXTRACTOR_GLCM_H
