#ifndef PRE_CUDA_WORKAREA_H
#define PRE_CUDA_WORKAREA_H

#include "GrayPair.h"
#include "AggregatedGrayPair.h"

using namespace std;

/**
 * This class handles memory locations used by the GLCM class to generate
 * glcm and the other 4 arrays from which features will be extracted as well as some
 * useful for the GLCM.
 *
 * Memory is allocated (by using malloc) externally to this class but the pointers are collected here.
*/
class WorkArea {
public:
    /**
     * Initialization
     * @param length: number of pairs of each window
     * @param grayPairs: memory space where the array of GrayPairs is created
     * for each window of the image
     * @param summedPairs: memory space where the array of summedGrayPairs
     * is created for each window of the image
     * @param subtractedPairs: memory space where the array of subtractedGrayPairs
     * is created for each window of the image
     * @param xMarginalPairs: memory space where the array of x-marginalGrayPairs
     * is created for each window of the image
     * @param yMarginalPairs: memory space where the array of y-marginalGrayPairs
     * is created for each window of the image
     * @param out: memory space where all the features values will be stored
     */
    WorkArea(int length,
            GrayPair* grayPairs,
            AggregatedGrayPair* summedPairs,
            AggregatedGrayPair* subtractedPairs,
            AggregatedGrayPair* xMarginalPairs,
            AggregatedGrayPair* yMarginalPairs,
            double* out):
            numberOfElements(length), grayPairs(grayPairs), summedPairs(summedPairs),
            subtractedPairs(subtractedPairs), xMarginalPairs(xMarginalPairs),
            yMarginalPairs(yMarginalPairs), output(out){};
    /**
     * Gets the arrays to initial state so another window can be processed
     */
    void cleanup();
    /**
     * Invocation of free on the pointers of all the meta-arrays of pairs
     */
    void release();
    /**
     * Where the GLCM are assembled
     */
    GrayPair* grayPairs;
    /**
     * Where the sum-aggregated representations are assembled
     */
    AggregatedGrayPair* summedPairs;
    /**
     * Where the diff-aggregated representations are assembled
     */
    AggregatedGrayPair* subtractedPairs;
    /**
     * Where the x-marginalPairs representations are assembled
     */
    AggregatedGrayPair* xMarginalPairs;
    /**
     * Where the y-marginalPairs representations are assembled
     */
    AggregatedGrayPair* yMarginalPairs;
    /**
     * Memory space where all the features values are stored
     */
    double* output;
    /**
     * Number of pairs contained in each window
     */
    int numberOfElements;

};


#endif //PRE_CUDA_WORKAREA_H
