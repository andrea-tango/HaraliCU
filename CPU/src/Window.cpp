#include "Window.h"

Window::Window(const short int dimension, const short int distance,
			   short int dirNumber, const bool symmetric){
	this->side = dimension;
	this->distance = distance;
	this->symmetric = symmetric;
	this->directionType = dirNumber;
}

void Window::setDirectionShifts(const int shiftRows, const int shiftColumns){
	this->shiftRows = shiftRows;
	this->shiftColumns = shiftColumns;
}

void Window::setSpatialOffsets(const int rowOffset, const int columnOffset){
	this->imageRowsOffset = rowOffset;
	this->imageColumnsOffset = columnOffset;
}
