#ifndef PRE_CUDA_IMAGEALLOCATED_H
#define PRE_CUDA_IMAGEALLOCATED_H

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
    /**
     * Getter
     * @return the pixels of the image
     */
    vector<unsigned int> getPixels() const;
    /**
    * Getter
    * @return the number of rows of the image
    */
    unsigned int getRows() const;
    /**
    * Getter
    * @return the number of columns of the image
    */
    unsigned int getColumns() const;
    /**
     * @return the maximum gray level actually represented in in the image
     * (it could depend on the image type and possible quantization strategy)
     */
    unsigned int getMaxGrayLevel() const;
    /**
     * @return the minimum gray level actually represented in in the image
     * (it could depend on the image type and possible quantization strategy)
     */
    unsigned int getMinGrayLevel() const;
    /**
     * DEBUG METHOD. Print all the pixels of the image
     */
    void printElements() const;

private:
    vector<unsigned int> pixels;
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
    const unsigned int minGrayLevel;
};


#endif //PRE_CUDA_IMAGEALLOCATED_H
