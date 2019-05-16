#ifndef FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H
#define FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H

// Custom types for code reusability
// Unsigned shorts halve the memory footprint of the application
typedef unsigned short grayLevelType;
typedef unsigned short frequencyType;

/**
 * This class represents two possible types of elements:
 * - Elements obtained by summing or subtracting two gray levels of a pixel pair
 * - Elements representing the frequency of one of the two gray levels of the
 * pixel pairs (reference gray level or neighbor gray level)
*/

class AggregatedGrayPair {
public:
    /**
     * Constructor for initializing pre-allocated work areas
     */
    AggregatedGrayPair();
    /**
     * Constructor for effective gray-tone pairs
     * @param level: gray level of the object
     * @param frequency: frequency of the object
     */
    AggregatedGrayPair(grayLevelType level, frequencyType frequency);

    /**
     * Shows textual representation with level and frequency
     */
    void printPair() const;
    /**
     * Getter
     * @return the gray level of the object
     */
    grayLevelType getAggregatedGrayLevel() const;
    /**
     * Getter
     * @return the frequency of the object
     */
    frequencyType getFrequency() const;
    /**
     * Setter
     * @param amount that will increment the frequency
     */
    void increaseFrequency(frequencyType amount);
    
    /**
     * Method to compare two AggregatedGrayPair objects according to the
     * equality of gray levels
     * @param other: object of the same type
     * @return true if the two objects have the same gray level
     */
    bool compareTo(AggregatedGrayPair other) const;

    // Overloaded C++ operators inherited from implementation that uses STL
    bool operator==(const AggregatedGrayPair& other) const{
        return (grayLevel == other.getAggregatedGrayLevel());
    }

    bool operator<(const AggregatedGrayPair& other) const{
        return (grayLevel < other.getAggregatedGrayLevel());
    }

    AggregatedGrayPair& operator++(){
        this->frequency += 1;
        return *this;
    }
private:
    grayLevelType grayLevel;
    frequencyType frequency;

};


#endif //FEATUREEXTRACTOR_AGGREGATEDGRAYPAIR_H
