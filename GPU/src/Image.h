
#ifndef IMAGE_H_
#define IMAGE_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

#include <iostream>
#include <vector>

using namespace std;

/**
 * This class represents the acquired image; it embeds:
 * - all the pixels (as unsigned int) composing the image
 * - matrix size dimensions (rows and columns)
 * - the minimum and maximum gray levels actually represented in in the image
 */
class Image {
public:
    /**
     * Constructor of the Image object
     * @param pixels: all pixels of the image transformed into unsigned int
     * @param rows
     * @param columns
     * @param minGrayLevel: minimum gray level represented in the image
     * @param maxGrayLevel: maximum gray level represented in the image
     * (it could depend on the image type and possible quantization strategy)
     */
    Image(vector<unsigned int> pixels, unsigned int rows, unsigned int columns,
          unsigned int minGrayLevel, unsigned int maxGrayLevel): 
          pixels(pixels), rows(rows), columns(columns), minGrayLevel(minGrayLevel),
          maxGrayLevel(maxGrayLevel){};
    // Getters
    /**
     * Getter
     * @return the pixels of the image
     */
    vector<unsigned int> getPixels() const; // Pixels must be moved to gpu with a plain pointer
    // Only pysical dimensions can be used in GPU
    /**
    * Getter
    * @return the number of rows of the image
    */
    CUDA_HOSTDEV unsigned int getRows() const;
    /**
    * Getter
    * @return the number of columns of the image
    */
    CUDA_HOSTDEV unsigned int getColumns() const;
    /**
     * @return the maximum gray level actually represented in in the image
     * (it could depend on the image type and possible quantization strategy)
     */
    CUDA_HOSTDEV unsigned int getMaxGrayLevel() const;
    /**
     * @return the minimum gray level actually represented in in the image
     * (it could depend on the image type and possible quantization strategy)
     */
    CUDA_HOSTDEV unsigned int getMinGrayLevel() const;
    /**
     * DEBUG METHOD. Print all the pixels of the image
     */
    CUDA_HOST void printElements() const;

private:
    vector<unsigned int> pixels;
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
    const unsigned int minGrayLevel;
};


#endif /* IMAGE_H_ */
