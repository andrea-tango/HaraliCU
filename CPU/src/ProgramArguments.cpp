#include "ProgramArguments.h"

/**
 * Shows a visual helper to the user about the usage of the tool
 */
void ProgramArguments::printProgramUsage(){
    const char *help =
    "./HaraliCPU -i <value> path input folder\n"
    "            -o <value> path output folder               (default: output)\n"
    "            -w <value> window size                      (default: 5)\n"
    "            -p <value> padding                          (default: 1)\n"
    "            -d <value> distance                         (default: 1)\n"
    "            -t <value> orientation                      (default: 1)\n"
    "            -q <value> max gray level for quantization  (default: image max value)\n"
    "            -g GLCM pairs symmetric\n"
    "            -s saving the feature maps\n"
    "            -v verbose\n"
    "            -h help\n"
    "            -? help\n";

    cout << endl << help << endl;
    cout << endl << "*****************************************************************************************************" << endl;
    exit(2);
}

/**
 * Function for checking and loading the options provided by the user
 * into the class ProgramArgument 
 * @param argc number of provided options
 * @param argv value of the provided options
 * @return class ProgramArguments that stores all the parameters for the GLCM computation
 */
ProgramArguments ProgramArguments::checkOptions(int argc, char* argv[]){
    ProgramArguments progArg;
    int opt;
    while((opt = getopt(argc, argv, "gsw:d:n:ht:vo:i:q:p:")) != -1){
        switch (opt){
            case 'p':{
                // Choosing among no, zero or symmetric padding
                short int type = atoi(optarg);
                switch(type){
                    case 0:
                    case 1:
                    case 2:
                        progArg.borderType = type;
                        break;
                    default:
                        cerr << "ERROR! -b option must be a value between 0 and 2" << endl;
                        printProgramUsage();
                }
                break;
            }
            case 'q':{
                // Quantization step
                progArg.quantize = true;
                progArg.quantizationMax = atoi(optarg);
                break;
            }
            case 'g':{
                // Making the GLCM pairs symmetric
                progArg.symmetric = true;
                break;
            }
            case 's':{
                // Saving the feature maps
                progArg.createImages = true;
                break;
            }
            case 'i':{
                // Input folder
                progArg.imagePath = optarg;
                break;
            }
            case 'o':{
                // Output folder for the results
                progArg.outputFolder = optarg;
                break;
            }
            case 'v':{
                // Verbose mode
                progArg.verbose = true;
                break;
            }
            case 'd': {
                int distance = atoi(optarg);
                if (distance < 1) {
                    cerr << "ERROR! The distance between each pixel pair must be >= 1 ";
                    printProgramUsage();
                }
                progArg.distance = distance;
                break;
            }
            case 'w': {
                // Deciding the size of each sub-window of the image
                short int windowSize = atoi(optarg);
                if ((windowSize < 2) || (windowSize > 100)) {
                    cerr << "ERROR! The size of the sliding windows to be extracted. Option (-w) "
                            "must have a value between 2 and 100";
                    printProgramUsage();
                }
                progArg.windowSize = windowSize;
                break;
            }
            case 't':{
                // Deciding how many of the 4 directions will be computed
                short int dirType = atoi(optarg);
                if(dirType > 4 || dirType <1){
                    cerr << "ERROR ! The type of directions to be computed. "
                            "Option (-t) must be a value between 1 and 4" << endl;
                    printProgramUsage();
                }
                progArg.directionType = dirType;

                break;
            }
            case 'n':{
                short int dirNumber = atoi(optarg);
                if(dirNumber != 1){
                    cout << "Warning! In the current version, just one direction "
                            "can be be computed at each time" << endl;
                    printProgramUsage();
                }
                progArg.directionType = dirNumber;
                break;
            }
            case '?':
                // Unsupported options
                printProgramUsage();
            case 'h':
                // Help
                printProgramUsage();
                break;
            default:
                printProgramUsage();
        }


    }

    if(progArg.distance > progArg.windowSize){
        cout << "WARNING: distance cannot higher than the window size; distance value corrected to 1" << endl;
        progArg.distance = 1;
    }

    // No image provided
    if(progArg.imagePath.empty()) {
        cerr << "ERROR! Missing image path!" << endl;
        printProgramUsage();
    }

    // Option output folder was not used
    if(progArg.outputFolder.empty())
        progArg.outputFolder = Utils::removeExtension(Utils::basename(progArg.imagePath));

    return progArg;
}