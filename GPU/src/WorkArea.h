#ifndef WORKAREA_H_
#define WORKAREA_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

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
    CUDA_HOSTDEV WorkArea(int length,
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
    CUDA_DEV void cleanup();
    /**
     * Invocation of free on the pointers of all the meta-arrays of pairs
     */
    CUDA_HOST void release();
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

#endif /* WORKAREA_H_ */
