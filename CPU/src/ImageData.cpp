#include <iostream>
#include <stdio.h>
#include <cstring>
#include "ImageData.h"

uint ImageData::getRows() const{
    return rows;
}

uint ImageData::getColumns() const{
    return columns;
}

int ImageData::getBorderSize() const {
    return appliedBorders;
}
uint ImageData::getMaxGrayLevel() const{
    return maxGrayLevel;
}
uint ImageData::getMinGrayLevel() const{
    return minGrayLevel;
}