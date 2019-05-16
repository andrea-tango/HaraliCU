#ifndef FEATUREEXTRACTOR_GRAYPAIR_H
#define FEATUREEXTRACTOR_GRAYPAIR_H

// Custom types for code reusability
// Unsigned shorts halve the memory footprint of the application
typedef unsigned short grayLevelType;
typedef unsigned short frequencyType;

/**
 * This class represents the gray levels of a pixel pair
*/
class GrayPair {
public:
    /**
     * Constructor for initializing pre-allocated working areas
     */
    GrayPair();
    /**
     * Constructor for effective gray-tone pairs
     * @param i grayLevel of the reference pixel of the pair
     * @param j grayLevel of the neighbor pixel of the pair
     */
    GrayPair(grayLevelType i, grayLevelType j);
    /**
     * Getter
     * @return the gray level of the reference pixel of the pair
     */
    grayLevelType getGrayLevelI() const;
    /**
     * Getter
     * @return the gray level of the neighbor pixel of the pair
     */
    grayLevelType getGrayLevelJ() const;
    /**
     * Getter
     * @return the frequency of the pair of gray levels in the glcm
     */
    frequencyType getFrequency() const;

    /**
     * Method to determine the equality according to the gray tones of the pair
     * @param other: gray pair to compare
     * @param symmetry: true if the symmetric versionhas to be  considered
     * symmetry ('i == i' or 'i == j')
     * @return: true if both grayLevels of both items are the same
     */
    bool compareTo(GrayPair other, bool symmetry) const;
    // Setter
    /**
     * DEPRECATED Setter. Use the ++ operator instad
     */
    void frequencyIncrease(); // frequency can be incremented only by 1
    /**
     * Shows textual representation of the gray pair
     */
    void printPair() const;

    // Overloaded C++ operators inherited from implementation that uses STL
    GrayPair& operator++(){
        this->frequency +=1;
        return *this;
    }

    bool operator==(const GrayPair& other) const{
        if((grayLevelI == other.getGrayLevelI()) &&
            (grayLevelJ == other.getGrayLevelJ()))
            return true;
        else
            return false;

    }

    bool operator<(const GrayPair& other) const{
        if(grayLevelI != other.getGrayLevelI())
            return (grayLevelI < other.getGrayLevelI());
        else
            return (grayLevelJ < other.getGrayLevelJ());
    }
private:
    grayLevelType grayLevelI;
    grayLevelType grayLevelJ;
    frequencyType frequency;

};


#endif //FEATUREEXTRACTOR_GRAYPAIR_H
