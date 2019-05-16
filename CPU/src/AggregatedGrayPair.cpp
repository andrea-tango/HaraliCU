#include <iostream>
#include "AggregatedGrayPair.h"

AggregatedGrayPair::AggregatedGrayPair() {
    grayLevel = 0;
    frequency = 0;
}

AggregatedGrayPair::AggregatedGrayPair(grayLevelType i, frequencyType freq){
    grayLevel = i;
    frequency = freq;
}

void AggregatedGrayPair::printPair() const {
    std::cout << "k: " << grayLevel;
    std::cout << "\freq: " << frequency;
    std::cout << std::endl;
}

/* Extracting the pairs */
grayLevelType AggregatedGrayPair::getAggregatedGrayLevel() const{
    return grayLevel;
}

frequencyType AggregatedGrayPair::getFrequency() const {
    return frequency;
}

bool AggregatedGrayPair::compareTo(AggregatedGrayPair other) const{
    return (grayLevel == other.getAggregatedGrayLevel());
}

void AggregatedGrayPair::increaseFrequency(frequencyType amount){
    frequency += amount;
}


