
#ifndef FEATUREEXTRACTOR_IMAGE_H
#define FEATUREEXTRACTOR_IMAGE_H

#include "Image.h"

using namespace std;

/**
 * This class stores the metadata of the acquired image:
 * - matrix size dimensions (rows and columns)
 * - the minimum and maximum gray levels actually represented in in the image
 *
 * The "Image" class should be used only for the CPU version.
 * Still useful for the development and debugging of the GPU version
*/

class ImageData {
public:
    /**
     * USE THE OTHER CONSTRUCTOR
     * @param rows
     * @param columns
     * @param borders
     * @param mxGrayLevel
     */
    explicit ImageData(unsigned int rows, unsigned int columns, int borders,
            unsigned int mixGrayLevel, unsigned int maxGrayLevel)
            : rows(rows), columns(columns), appliedBorders(borders),
            minGrayLevel(minGrayLevel), maxGrayLevel(maxGrayLevel){};
     /**
      * Constructor that extracts the metadata from the "complete" Image class
      * @param whole image as well as the metadata
      * @param borders applied to each side the original image
    */
    explicit ImageData(const Image& img, int borders)
            : rows(img.getRows()), columns(img.getColumns()),
            appliedBorders(borders), 
            minGrayLevel(img.getMinGrayLevel()),
            maxGrayLevel(img.getMaxGrayLevel()){};
    // Getters
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
     * @return maximum gray level represented in the
     * image; it depends on the image type and eventual quantization applied
     */
    unsigned int getMaxGrayLevel() const;
    /**
     * @return minimum gray level represented in the
     * image; it depends on the image type and eventual quantization applied
     */
    unsigned int getMinGrayLevel() const;
    /**
     *
     * @return borders applied to each side of the original image
     */
    int getBorderSize() const;

private:
    const unsigned int rows;
    const unsigned int columns;
    const unsigned int maxGrayLevel;
    const unsigned int minGrayLevel;
    // Borders applied to each side of the original image
    int appliedBorders;
};


#endif //FEATUREEXTRACTOR_IMAGE_H
