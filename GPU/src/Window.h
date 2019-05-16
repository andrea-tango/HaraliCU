#ifndef WINDOW_H_
#define WINDOW_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

/**
 * This class embeds all the necessary metadata used from GLCM class to locate
 * the pixel pairs that need to be processed in this window, in one direction
*/
class Window {
public:
    /**
     * Initialization
     * @param dimension: width of each squared window
     * @param distance: length of vector reference-neighbor pixel pair
     * @param directionType: direction of interest in this window
     * @param symmetric: symmetry of the gray levels in the window
     */
    CUDA_HOSTDEV Window(short int dimension, short int distance, short int directionType, bool symmetric = false);
    // Structural data uniform for all windows
    /**
     * Width of each squared window
     */
    short int side; 
    /**
     * Length of the vector reference-neighbor pixel pair
     */
    short int distance;
    /**
     *  Symmetry of the pixel pair
     */
    bool symmetric;
    /**
     * Attribute used by the WindowFeatureComputer.
     * Redundant information obtainable from the combination of shiftRows and
     * shiftColumns
     */
    short int directionType;
    
    // Direction shifts to locate the pixel pair <reference,neighbor>
    // The 4 possible combinations are imposed after the creation of the window
    /**
     * Shift on the y axis to denote the neighbor pixel
     */
    int shiftRows; 
    /**
     * Shift on the x axis to denote the neighbor pixel
     */
    int shiftColumns;
    /**
     * Acquires both shifts on the x and y axes to denote the neighbor pixel of
     * the pair
     */
    CUDA_HOSTDEV void setDirectionShifts(int shiftRows, int shiftColumns);

     // Offset to define the starting point of the window inside the entire image
    /**
     * First row of the image that belongs to this window
     */
    int imageRowsOffset;
    /**
     * First column of the image that belongs to this window
     */
    int imageColumnsOffset;
    /**
     * Acquires both shifts defining the starting point for the window in the image
     */
    CUDA_HOSTDEV void setSpatialOffsets(int rowOffset, int columnOffset);
};

#endif /* WINDOW_H_ */
