#ifndef FEATUREEXTRACTOR_WINDOW_H
#define FEATUREEXTRACTOR_WINDOW_H

/**
 * This class embeds all the metadata used from the GLCM class to locate
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
    Window(short int dimension, short int distance, short int directionType, bool symmetric = false);
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
     * Symmetry of the pixel pair
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
     * Acquires both shifts on the x and y axis to denote the neighbor pixel of
     * the pair
     */
    void setDirectionShifts(int shiftRows, int shiftColumns);

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
     * Acquires both shift that tells the window starting point in the image
     */
    void setSpatialOffsets(int rowOffset, int columnOffset);
};


#endif //FEATUREEXTRACTOR_WINDOW_H
