#include <iostream>
#include "GrayPair.h"

/* Constructors */
GrayPair::GrayPair()
{
    grayLevelI = 0;
    grayLevelJ = 0;
    frequency = 0;
}

GrayPair::GrayPair (grayLevelType i, grayLevelType j) {
   grayLevelI = i;
   grayLevelJ = j;
   frequency = 1;
}

void GrayPair::printPair()const {
    std::cout << "i: "<< grayLevelI;
    std::cout << "\tj: " << grayLevelJ;
    std::cout << "\tmult: " << frequency;
    std::cout << std::endl;
}

void GrayPair::frequencyIncrease(){
    frequency += 1;
}

bool GrayPair::compareTo(GrayPair other, bool symmetry) const{
    bool pairsAreEquals;
    bool sameGrayLevels = (grayLevelI == other.getGrayLevelI())
                          && (grayLevelJ == other.getGrayLevelJ());

    bool symmetricGrayLevels = (grayLevelI == other.getGrayLevelJ())
         && (grayLevelJ == other.getGrayLevelI());

    if(symmetry){
        if(sameGrayLevels || symmetricGrayLevels)
            pairsAreEquals = true;
        else
            pairsAreEquals = false;
    }
    else{
        if(sameGrayLevels)
            pairsAreEquals = true;
        else
            pairsAreEquals = false;
    }
    return pairsAreEquals;

}

/* Extracting the pairs */
grayLevelType GrayPair::getGrayLevelI() const{
    return grayLevelI;
}

grayLevelType GrayPair::getGrayLevelJ() const{
    return grayLevelJ;
}

frequencyType GrayPair::getFrequency() const {
    return frequency;
}
