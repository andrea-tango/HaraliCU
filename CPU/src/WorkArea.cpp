#include <cstdlib>
#include "WorkArea.h"

/*
 * This method is necessary because, at the start of the computation,
 * some of the functions of GLCM class could mistake dirty cells in
 * the areas (allocated by malloc) as legit pixel-pairs generated previously
 * and not as available memory
*/
void WorkArea::cleanup() {
    // Representations that GLCM uses as "available" memory
    GrayPair voidElement; // 0 in each field
    AggregatedGrayPair voidAggregatedElement; // 0 in each field

    for (int i = 0; i < numberOfElements; ++i) {
        grayPairs[i] = voidElement;
        summedPairs[i] = voidAggregatedElement;
        subtractedPairs[i] = voidAggregatedElement;
        xMarginalPairs[i] = voidAggregatedElement;
        yMarginalPairs[i] = voidAggregatedElement;
    }
}

// Invoked externally when the workArea is not needed
void WorkArea::release(){
    free(grayPairs);
    free(summedPairs);
    free(subtractedPairs);
    free(xMarginalPairs);
    free(yMarginalPairs);
}