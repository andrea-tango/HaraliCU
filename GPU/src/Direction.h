#ifndef DIRECTION_H_
#define DIRECTION_H_

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
 * This class represents one of the supported directions;
 * it embeds values for locating reference-neighbor pixel pairs.
 * Supported directions with their associated eoncoding:
 * 0°[1], 45°[2], 90° [3], 135° [4]
*/

class Direction {
public:
    /**
     * Constructs an object of the Direction class
     * @param directionNumber: the encoding associated with the direction:
     * 0°[1], 45°[2], 90° [3], 135° [4]
     */
    CUDA_HOSTDEV Direction(int directionNumber);

    /**
     * Shows info about the direction
     * @param direction: the encoding associated with the direction:
     * 0°[1], 45°[2], 90° [3], 135° [4]
     */
    CUDA_HOSTDEV static void printDirectionLabel(const int direction);
    char label[21];
    /**
     * Shift on the y axis to denote the neighbor pixel
     */
    int shiftRows;
    /**
     * Shift on the x axis to denote the neighbor pixel
     */
    int shiftColumns;
};
#endif /* DIRECTION_H_ */
