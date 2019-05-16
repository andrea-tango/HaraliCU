#ifndef PRE_CUDA_PROGRAMARGUMENTS_H
#define PRE_CUDA_PROGRAMARGUMENTS_H

#include <string>
#include <iostream>
#include <getopt.h> // For options check

#include "Utils.h"

using namespace std;

/**
 * Class for keeping and checking all the possible parameters to the current GLCM problem
 */
class ProgramArguments {
public:
    /**
     * Size of each squared window that will be generated
     */
    short int windowSize;
    /**
     *  Optional quantization of gray levels to a fixed range
     */
    bool quantize;
    /**
     *  Maximum gray level when a quantization step is applied
     */
    int quantizationMax;
    /**
     * Type of padding applied to the orginal image:
     * 0 = no border
     * 1 = zero pixel border
     * 2 = symmetric border
     */
    short int borderType;
    /**
     * Symmetry condition of the gray level pairs in each GLCM
     */
    bool symmetric;
    /**
     * Magnitude of the vector connecting the reference to neighbor pixel
     */
    short int distance;
    /**
     * Selected orientation among 0°, 45°, 90°, 135°
     */
    short int directionType;
    /**
     * Number of directions computed for each window.
     * LIMITED to 1 at this release
     */
    short int directionsNumber;
    /**
     * Optional generation of the feature maps from the computed feature values
     */
    bool createImages;
    /**
     * Path/name of the image to process
     */
    string imagePath;
    /**
     * Output path for the results.
     * If none is provided, the name of the image will be used without any
     * path/extension
     */
    string outputFolder;
    /**
     * Prints additional information
     */
    bool verbose;

    /**
     * Constructor of the class that contains all the parameters of the problem
     * @param windowSize: size of each squared window that will be created
     * @param quantize: optional quantization of the gray levels in
     * @param symmetric: optional symmetry of gray levels in each pixels pair during the GLCM computation
     * @param distance: magnitude of the vector connecting the reference to neighbor pixel
     * @param dirType: selected orientation among 0°, 45°, 90°, 135°
     * @param dirNumber: number of directions computed for each window
     * @param createImages: optional generation of the feature maps from the computed feature values
     * computed
     * @param border: type of border applied to the orginal image
     * @param verbose: printing additional info
     * @param outFolder: output path for the results
     */
    ProgramArguments(short int windowSize = 5,
                     bool quantize = false,
                     bool symmetric = false,
                     short int distance = 1,
                     short int dirType = 1,
                     short int dirNumber = 1,
                     bool createImages = false,
                     short int border = 1,
                     bool verbose = false,
                     string outFolder = "output")
            : windowSize(windowSize), borderType(border), quantize(quantize), symmetric(symmetric), distance(distance),
              directionType(dirType), directionsNumber(dirNumber),
              createImages(createImages), outputFolder(outFolder),
              verbose(verbose){};
    /**
     * Shows the user how to use the program and its options
     */
    static void printProgramUsage();
    /**
     * Loads the options provided in the command line and checks them
     * @param argc
     * @param argv
     * @return
     */
    static ProgramArguments checkOptions(int argc, char* argv[]);
};


#endif //PRE_CUDA_PROGRAMARGUMENTS_H