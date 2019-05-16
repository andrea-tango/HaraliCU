
#ifndef FEATUREEXTRACTOR_DIRECTION_H
#define FEATUREEXTRACTOR_DIRECTION_H

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
    Direction(int directionNumber);

    /**
     * Shows info about the direction
     * @param direction: the encoding associated with the direction:
     * 0°[1], 45°[2], 90° [3], 135° [4]
     */
    static void printDirectionLabel(const int direction);
    char label[20];
    /**
     * Shift on the y axis to denote the neighbor pixel
     */
    int shiftRows;
    /**
     * Shift on the x axis to denote the neighbor pixel
     */
    int shiftColumns;
};


#endif //FEATUREEXTRACTOR_DIRECTION_H
