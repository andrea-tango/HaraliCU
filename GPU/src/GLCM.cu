#include <cmath>
#include <iostream>
#include <assert.h>
#include "GLCM.h"
#include "GrayPair.h"
#include "AggregatedGrayPair.h"

using namespace std;

// Constructors
__device__ GLCM::GLCM(const unsigned int * pixels, const ImageData& image,
        Window& windowData, WorkArea& wa): pixels(pixels), img(image),
        windowData(windowData),  workArea(wa) ,grayPairs(wa.grayPairs),
        summedPairs(wa.summedPairs), subtractedPairs(wa.subtractedPairs),
        xMarginalPairs(wa.xMarginalPairs), yMarginalPairs(wa.yMarginalPairs)
        {
    // Computing the number of pairs that need to be processed in this GLCM
    this->numberOfPairs = getWindowRowsBorder() * getWindowColsBorder();

    // Dealing with memory leaking
    workArea.cleanup();
    // Generating the elements of this GLCM
    initializeGlcmElements();}


__device__ GLCM::~GLCM(){

}

__device__ int GLCM::getNumberOfPairs() const {
        return numberOfPairs;
}

__device__ int GLCM::getMaxGrayLevel() const {
    return img.getMaxGrayLevel();
}

/**
 * Geometric limit of the sub-window
 * @return the number of rows of the window to be considered
 */
__device__ int GLCM::getWindowRowsBorder() const{
   return (windowData.side - (windowData.distance * abs(windowData.shiftRows)));
}

/**
 * Geometric limit of the sub-window
 * @return the number of the columns of the window to be considered
 */
__device__ int GLCM::getWindowColsBorder() const{
    return (windowData.side - (windowData.distance * abs(windowData.shiftColumns)));
}


/**
 * Computing the shift to apply at the column for locating the pixels of each
 * pair of the glcm; it affects only 135° orientation
 * @return d (distance) pixels to be ignored
 */
__device__ inline int GLCM::computeWindowColumnOffset()
{
    int initialColumnOffset = 0; // for 0°,45°,90°
    if((windowData.shiftRows * windowData.shiftColumns) > 0) // 135°
        initialColumnOffset = 1;
    return initialColumnOffset;
}

/**
 * Computes the shift to apply at the row for locating the pixels of each
 * pair of the glcm; it does not affect 0° orientation alone
 * @return d (distance) pixels need to be ignored
*/
__device__ inline int GLCM::computeWindowRowOffset()
{
    int initialRowOffset = 1; // for 45°,90°,135°
    if((windowData.shiftRows == 0) && (windowData.shiftColumns > 0))
        initialRowOffset = 0; // for 0°
    return initialRowOffset;
}

/**
 * Methods to obtain the reference pixel in each pair of the glcm
 * @param row in the sub-window of the reference pixel
 * @param col in the sub-window of the reference pixel
 * @param initialRowOffset see computeWindowRowOffset
 * @param initialColumnOffset see computeWindowColOffset
 * @return the index of the pixel in the array of pixels (linearized) of
 * the window
 */
__device__ inline int GLCM::getReferenceIndex(const int i, const int j,
                                   const int initialWindowRowOffset, const int initialWindowColumnOffset){
    int row = (i + windowData.imageRowsOffset) // starting point in the image
            + (initialWindowRowOffset * windowData.distance); // add direction eventual down-shift (45°, 90°, 135°)
    int col = (j + windowData.imageColumnsOffset) + // starting point in the image
            (initialWindowColumnOffset * windowData.distance); // add direction shift
    int index = ( row * img.getColumns()) + col;
    assert(index >= 0);
    return index;
}

/**
 * Methods to obtain the neighbor pixel in each pair of the glcm
 * @param row in the sub-window of the neighbor pixel
 * @param col in the sub-window of the neighbor pixel
 * @param initialColumnOffset see computeWindowColOffset
 * @return the index of the pixel in the array of pixels (linearized) of
 * the window
 */
__device__ inline int GLCM::getNeighborIndex(const int i, const int j,
                                  const int initialWindowColumnOffset){
    int row = (i + windowData.imageRowsOffset); // starting point in the image
    int col = (j + windowData.imageColumnsOffset) + // starting point in the image
              (initialWindowColumnOffset * windowData.distance) +  // add 135* right-shift
              (windowData.shiftColumns * windowData.distance); // add direction shift
    int index = (row * img.getColumns()) + col;
    assert(index >= 0);
    return index;
}

/**
 * Method that adds a GrayPair into the pre-allocated memory
 * It uses the convention that GrayPair (i=0, j=0, frequency=0) means
 * available memory
 */
__device__ inline void GLCM::insertElement(GrayPair* grayPairs, const GrayPair actualPair, 
    uint& lastInsertionPosition, bool symmetry){
    int position = 0;
    // Finding if the element was already inserted, and where
    while((!grayPairs[position].compareTo(actualPair, symmetry)) && (position < numberOfPairs))
        position++;
    // If found
    if((lastInsertionPosition > 0) // 0,0 as first element will increase insertion position
        && (position != numberOfPairs)){ // if the item was already inserted
        grayPairs[position].operator++();
        if((actualPair.getGrayLevelI() == 0) && (actualPair.getGrayLevelJ() == 0)
            && (grayPairs[position].getFrequency() == actualPair.getFrequency()))
            // Corner case, the inserted pair <0,0> that matches with every empty field
            lastInsertionPosition++;
    }
    else
    {
        grayPairs[lastInsertionPosition] = actualPair;
        lastInsertionPosition++;
    }
}

/**
 * This method creates the array of GrayPairs
*/
__device__ void GLCM::initializeGlcmElements() {
    // Defining the subBorders offset according to the orientation
    int initialWindowColumnOffset = computeWindowColumnOffset();
    int initialWindowRowOffset = computeWindowRowOffset();

    grayLevelType referenceGrayLevel;
    grayLevelType neighborGrayLevel;
    unsigned int lastInsertionPosition = 0;
    // Navigating the sub-window of interest
    for (int i = 0; i < getWindowRowsBorder() ; i++)
    {
        for (int j = 0; j < getWindowColsBorder(); j++)
        {
            // Extracting the two pixels in the pair
            int referenceIndex = getReferenceIndex(i, j,
                    initialWindowRowOffset, initialWindowColumnOffset);
            // Limit up to 2^16 gray levels
            referenceGrayLevel = pixels[referenceIndex]; // should be safe
            int neighborIndex = getNeighborIndex(i, j,
                    initialWindowColumnOffset);
            // Limit up to 2^16 gray levels
            neighborGrayLevel = pixels[neighborIndex];  // should be safe

            GrayPair actualPair;
            
            if((windowData.symmetric) && (neighborGrayLevel > referenceGrayLevel))
            {
                actualPair = GrayPair(neighborGrayLevel, referenceGrayLevel);
            }
            else
            {
                actualPair = GrayPair(referenceGrayLevel, neighborGrayLevel);
            }

            insertElement(grayPairs, actualPair, lastInsertionPosition, windowData.symmetric);
            
        }
    }
    effectiveNumberOfGrayPairs = lastInsertionPosition;
    codifyAggregatedPairs();
    codifyMarginalPairs();
}

/**
 * Method that adds an AggregatedGrayPair into the pre-allocated memory.
 * It uses the convention that AggregateGrayPair (k=0, frequency=0) means
 * available memory
 */
__device__ inline void GLCM::insertElement(AggregatedGrayPair* elements, const AggregatedGrayPair actualPair, uint& lastInsertionPosition){
    int position = 0;
    // Finding if the element was already inserted, and where
    while((!elements[position].compareTo(actualPair)) && (position < numberOfPairs))
        position++;
    // If found
    if((lastInsertionPosition > 0) && // corner case 0 as first element
        (position != numberOfPairs)){ // if the item was already inserted
            elements[position].increaseFrequency(actualPair.getFrequency());
        if((actualPair.getAggregatedGrayLevel() == 0) && // corner case 0 as regular element
        (elements[position].getFrequency() == actualPair.getFrequency()))
            // Corner case, inserted 0 that matches with every empty field
            lastInsertionPosition++;
    }
    else
    {
        elements[lastInsertionPosition] = actualPair;
        lastInsertionPosition++;
    }
}

/**
 * This method produces the two arrays of AggregatedPairs (k, frequency)
 * where k is the sum or difference of both grayLevels of a GrayPair.
 * This representation is used in computeSumXXX() and computeDiffXXX() features
*/
__device__ void GLCM::codifyAggregatedPairs() {
    unsigned int lastInsertPosition = 0;
    // Summed pairs, firstly
    for(int i = 0 ; i < effectiveNumberOfGrayPairs; i++){
        // Creating the summed pairs, firstly
        grayLevelType k= grayPairs[i].getGrayLevelI() + grayPairs[i].getGrayLevelJ();
        AggregatedGrayPair summedElement(k, grayPairs[i].getFrequency());

        insertElement(summedPairs, summedElement, lastInsertPosition);
    }
    numberOfSummedPairs = lastInsertPosition;

    // Diff pairs, secondly
    lastInsertPosition = 0;
    for(int i = 0 ; i < effectiveNumberOfGrayPairs; i++){
        int diff = grayPairs[i].getGrayLevelI() - grayPairs[i].getGrayLevelJ();
        grayLevelType k= static_cast<uint>(abs(diff));
        AggregatedGrayPair element(k, grayPairs[i].getFrequency());

        insertElement(subtractedPairs, element, lastInsertPosition);
    }
    numberOfSubtractedPairs = lastInsertPosition;
}

/**
 * This method produces the two arrays of AggregatedPairs (k, frequency)
 * where k is one grayLevel of GLCM and frequency is the "marginal" frequency of that level
 * (i.e., how many times k is present in all GrayPair<k, ?>)
 * This representation is used for computing features HX, HXY, HXY1, imoc
*/
__device__ void GLCM::codifyMarginalPairs() {
    unsigned int lastInsertPosition = 0;
    // xMarginalPairs, firstly
    for(int i = 0 ; i < effectiveNumberOfGrayPairs; i++){
        grayLevelType firstGrayLevel = grayPairs[i].getGrayLevelI();
        AggregatedGrayPair element(firstGrayLevel, grayPairs[i].getFrequency());

        insertElement(xMarginalPairs, element, lastInsertPosition);
    }
    numberOfxMarginalPairs = lastInsertPosition;

    // yMarginalPairs, secondly
    lastInsertPosition = 0;
    for(int i = 0 ; i < effectiveNumberOfGrayPairs; i++){
        grayLevelType secondGrayLevel = grayPairs[i].getGrayLevelJ();
        AggregatedGrayPair element(secondGrayLevel, grayPairs[i].getFrequency());

        insertElement(yMarginalPairs, element, lastInsertPosition);
    }
    numberOfyMarginalPairs = lastInsertPosition;
}

/* DEBUGGING METHODS */
__device__ void GLCM::printGLCM() const {
    printGLCMData();
    printGLCMElements();
    printAggregated();
    printMarginalProbabilityElements();
}

__device__ void GLCM::printGLCMData() const{
    printf("\n");
    printf("***\tGLCM data\t***\n");
    printf("Shift rows: %d \n", windowData.shiftRows);
    printf("Shift columns: %d \n", windowData.shiftColumns);
    printf("Sliding window side: %d \n", windowData.side);
    printf("Border rows: %d \n", getWindowRowsBorder());
    printf("Border columns: %d \n", getWindowColsBorder());
    printf("Symetric: ");
    if(windowData.symmetric){
    	printf("Yes\n");
    }
    else{
    	printf("No\n");
    }
    printf("\n");;
}

__device__ void GLCM::printGLCMElements() const{
    printf("* GrayPairs *\n");
    for (int i = 0; i < effectiveNumberOfGrayPairs; ++i) {
        grayPairs[i].printPair();;
    }
}

__device__ void GLCM::printAggregated() const{
    printGLCMAggregatedElements(true);
    printGLCMAggregatedElements(false);
}

__device__ void GLCM::printGLCMAggregatedElements(bool areSummed) const{
    printf("\n");
    if(areSummed) {
        printf("* Summed grayPairsMap *\n");
        for (int i = 0; i < numberOfSummedPairs; ++i) {
            summedPairs[i].printPair();
        }
    }
    else {
        printf("* Subtracted grayPairsMap *\n");
        for (int i = 0; i < numberOfSubtractedPairs; ++i) {
            subtractedPairs[i].printPair();
        }
    }
}



__device__ void GLCM::printMarginalProbabilityElements() const{
    printf("\n* xMarginal encoding\n");
    for (int i = 0; i < numberOfxMarginalPairs; ++i) {
        printf("(%d, X):\t%d\n", xMarginalPairs[i].getAggregatedGrayLevel(), xMarginalPairs[i].getFrequency());
    }
    printf("\n* yMarginal encoding\n");
    for (int i = 0; i <numberOfyMarginalPairs; ++i) {
        printf("(X, %d):\t%d\n", yMarginalPairs[i].getAggregatedGrayLevel(), yMarginalPairs[i].getFrequency());

    }

}


