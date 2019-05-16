#include "WindowFeatureComputer.h"


__device__ WindowFeatureComputer::WindowFeatureComputer(unsigned int * pxls,
		const ImageData& img, const Window& wd, WorkArea& wa): pixels(pxls),
		image(img), windowData(wd), workArea(wa){
	computeWindowFeatures();
}

/*
 *	This method computes all the features for all directionType directions
 *	provided by a parameter to the program; the order is 0°, 45°, 90°, 135° ;
*/
__device__ void WindowFeatureComputer::computeWindowFeatures() {
  	// Getting shift vector for each direction of interest
    Direction actualDir = Direction(windowData.directionType);
    // Creating the autonomous thread of computation
    FeatureComputer fc(pixels, image, actualDir.shiftRows, actualDir.shiftColumns,
						   windowData, workArea);
}
