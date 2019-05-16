#include "Image.h"

__host__ __device__ unsigned int Image::getRows() const{
    return rows;
}

__host__ __device__ unsigned int Image::getColumns() const{
    return columns;
}

vector<unsigned int> Image::getPixels() const{
    return pixels;
}

__host__ __device__ unsigned int Image::getMaxGrayLevel() const{
    return maxGrayLevel;
}

__host__ __device__ unsigned int Image::getMinGrayLevel() const{
    return minGrayLevel;
}

void Image::printElements() const {
    std::cout << "Img = " << std::endl;

    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < columns; j++) {
            std::cout << pixels[i * rows + j] << " ";
        }
        std::cout << std::endl;
    }
}
