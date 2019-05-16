#include <stdio.h>
#include "AggregatedGrayPair.h"

__device__  AggregatedGrayPair::AggregatedGrayPair() {
    grayLevel = 0;
    frequency = 0;
}

__device__ AggregatedGrayPair::AggregatedGrayPair(grayLevelType i, frequencyType freq){
    grayLevel = i;
    frequency = freq;
}

__device__ void AggregatedGrayPair::printPair() const {
    printf("k: %d", grayLevel);
    printf("\tfreq: %d", frequency);
    printf("\n");
}

__device__ bool AggregatedGrayPair::compareTo(AggregatedGrayPair other) const{
    return (grayLevel == other.getAggregatedGrayLevel());
}

/* Extracting the pairs */
__device__ grayLevelType AggregatedGrayPair::getAggregatedGrayLevel() const{
    return grayLevel;
}

__device__ frequencyType AggregatedGrayPair::getFrequency() const {
    return frequency;
}

__device__ void AggregatedGrayPair::increaseFrequency(frequencyType amount){
    frequency += amount;
}
