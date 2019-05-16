#ifndef GRAYPAIR_H_
#define GRAYPAIR_H_

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

// Custom types for code reusability
// Unsigned shorts halve the memory footprint of the application
typedef short unsigned grayLevelType;
typedef short unsigned frequencyType;

/**
 * This class represents the gray levels of a pixel pair
*/
class GrayPair{
public:
	/**
     * Constructor for initializing pre-allocated working areas
     */
    CUDA_DEV GrayPair();
	/**
     * Constructor for effective gray-tone pairs
     * @param i grayLevel of the reference pixel of the pair
     * @param j grayLevel of the neighbor pixel of the pair
     */
    CUDA_DEV GrayPair(grayLevelType i, grayLevelType j);
    // Getters
    /**
     * Getter
     * @return the gray level of the reference pixel of the pair
     */
	CUDA_DEV grayLevelType getGrayLevelI() const;
    /**
     * Getter
     * @return the gray level of the neighbor pixel of the pair
     */
	CUDA_DEV grayLevelType getGrayLevelJ() const;
    /**
     * Getter
     * @return the frequency of the pair of gray levels in the glcm
     */
	CUDA_DEV frequencyType getFrequency() const;
    /**
     * Setter. Increase the frequency of the pair by 1
     */
	CUDA_DEV void frequencyIncrease(); // frequency can be incremented only by 1
	/**
     * Method to determine the equality according to the gray tones of the pair
     * @param other: gray pair to compare
     * @param symmetry: true if the symmetric versionhas to be  considered
     * symmetry ('i == i' or 'i == j')
     * @return: true if both grayLevels of both items are the same
     */
    CUDA_DEV bool compareTo(GrayPair other, bool symmetry) const;
    /**
     * Shows textual representation of the gray pair
     */
	CUDA_DEV void printPair() const;

    // Overloaded C++ operators inherited from implementation that uses STL
	CUDA_DEV GrayPair& operator++(){
        this->frequency +=1;
        return *this;
    }

	CUDA_DEV bool operator==(const GrayPair& other) const{
        if((grayLevelI == other.getGrayLevelI()) &&
            (grayLevelJ == other.getGrayLevelJ()))
            return true;
        else
            return false;

    }

	CUDA_DEV bool operator<(const GrayPair& other) const{
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
#endif /* GRAYPAIR_H_ */

