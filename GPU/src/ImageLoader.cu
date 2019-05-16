#include "ImageLoader.h"

#define IMG16MAXGRAYLEVEL 65535
#define IMG8MAXGRAYLEVEL 255

Mat ImageLoader::readImage(string fileName){
    Mat inputImage;
    try
    {
        inputImage = imread(fileName, IMREAD_UNCHANGED);
    }
    catch (cv::Exception& e) {
        const char *err_msg = e.what();
        cerr << "Exception occurred: " << err_msg << endl;
    }
    if(! inputImage.data )  // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        exit(-1);
    }
    // If the input is not a gray-scale image, it is converted in a color image
    if((inputImage.depth() != CV_8UC1) && (inputImage.depth() != CV_16UC1))
    {
        // Reducing the number of color channels from 3 to 1
        cvtColor(inputImage, inputImage, CV_RGB2GRAY);
        inputImage.convertTo(inputImage, CV_8UC1);
    }

    return inputImage;
}


Mat ImageLoader::createDoubleMat(const int rows, const int cols,
                                 const vector<double>& input){
    Mat_<double> output = Mat(rows, cols, CV_64F);
    // Copying the values into the image
    memcpy(output.data, input.data(), rows * cols * sizeof(double));
    return output;
}

// Utility method to iterate on the pixels encoded as uchars
inline void readUchars(vector<uint>& output, Mat& img){
    typedef MatConstIterator_<uchar> MI;
    int address = 0;
    for(MI element = img.begin<uchar>() ; element != img.end<uchar>() ; element++)
    {
        output[address] = *element;
        address++;
    }
}

// Utility method to iterate on the pixels encoded as uint
inline void readUint(vector<uint>& output, Mat& img){
    typedef MatConstIterator_<ushort> MI;
    int address = 0;
    for(MI element = img.begin<ushort>() ; element != img.end<ushort>() ; element++)
    {
        output[address] = *element;
        address++;
    }
}

Image ImageLoader::readImage(const string fileName, short int borderType,
                             int borderSize, bool quantize, int quantizationMax)
{
    // Opening image from file system
    cout << "* Loading image ..." << endl;

    Mat imgRead = readImage(fileName);

    double minIM, maxIM;
    minMaxLoc(imgRead, &minIM, &maxIM);

    uint minLevelIn = (uint) minIM;
    uint maxLevelIn = (uint) maxIM;

    cout << "* Min gray level in the input image: " << minLevelIn << endl;
    cout << "* Max gray level in the input image: " << maxLevelIn << endl;

    if((quantize) && (imgRead.depth() == CV_16UC1) && (quantizationMax > IMG16MAXGRAYLEVEL)){
        cout << "Warning! Provided a quantization level > maximum gray level of the image";
        quantizationMax = IMG16MAXGRAYLEVEL;
    }
    if((quantize) && (imgRead.depth() == CV_8UC1) && (quantizationMax > IMG8MAXGRAYLEVEL)){
        cout << "Warning! Provided a quantization level > maximum gray level of the image";
        quantizationMax = IMG8MAXGRAYLEVEL;
    }
    if(quantize)
    {
        imgRead = quantizeImage(imgRead, quantizationMax, minLevelIn, maxLevelIn);

        minMaxLoc(imgRead, &minIM, &maxIM);
        minLevelIn = (uint) minIM;
        maxLevelIn = (uint) maxIM;

    }

    // Creating borders to the image
    addBorderToImage(imgRead, borderType, borderSize);

    // Getting the pixels from the image to a standard uint array
    vector<uint> pixels(imgRead.total());

    // readUint(pixels, imgRead);

    switch (imgRead.type())
    {
        case CV_16UC1:
            readUint(pixels, imgRead);
            break;
        case CV_8UC1:
            readUchars(pixels, imgRead);
            break;
        default:
            cerr << "ERROR! Unsupported depth type: " << imgRead.type();
            exit(-4);
    }

    // CREATE IMAGE abstraction structure
    Image image = Image(pixels, imgRead.rows, imgRead.cols, minLevelIn, maxLevelIn);
    return image;
}


// Debug method
void ImageLoader::showImagePaused(const Mat& img, const string& windowName){
    namedWindow(windowName, WINDOW_AUTOSIZE );// Create a window for display.
    imshow(windowName, img );                   // Show our image inside it.
    waitKey(0);
}

// GLCM can work only with grayscale images
Mat ImageLoader::convertToGrayScale(const Mat& inputImage) {
    // Converting the image to a 255 gray-scale
    Mat convertedImage = inputImage.clone();
    normalize(convertedImage, convertedImage, 0, 255, NORM_MINMAX, CV_8UC1);
    return convertedImage;
}

unsigned int quantizationStep(uint intensity, uint maxLevel, uint minLevelIn, uint maxLevelIn)
{
    // double value = ((intensity) * (maxLevel))/(double)(maxLevelIn - minLevelIn);
    double value = ((intensity - minLevelIn) * (maxLevel))/(double)(maxLevelIn - minLevelIn);

    return (uint)round(value);
}

Mat ImageLoader::quantizeImage(Mat& img, int maxLevel, int minLevelIn, int maxLevelIn)
{
    Mat convertedImage = img.clone();

    typedef MatIterator_<ushort> MI;

    for(MI element = convertedImage.begin<ushort>() ; element != convertedImage.end<ushort>() ; element++)
    {

        uint intensity = *element;
        uint newIntensity = quantizationStep(intensity, maxLevel, minLevelIn, maxLevelIn);

        *element = newIntensity;
    }

    return convertedImage;
}

void ImageLoader::addBorderToImage(Mat &img, short int borderType, int borderSize) {
    switch (borderType){
        case 0: // NO PADDING
            break;
        case 1: // 0 pixel padding
            copyMakeBorder(img, img, borderSize, borderSize, borderSize, borderSize, BORDER_CONSTANT, 0);
            break;
        case 2: // Reflect pixels at the borders
            copyMakeBorder(img, img, borderSize, borderSize, borderSize, borderSize, BORDER_REPLICATE);
            break;
    }

}

// Linear contrast stretching
Mat ImageLoader::stretchImage(const Mat& inputImage){
    Mat stretched;

    // The stretching can only be applied to gray scale CV_8U
    if(inputImage.type() != CV_8UC1){
        inputImage.convertTo(inputImage, CV_8U);
    }

    Ptr<CLAHE> clahe = createCLAHE(4);
    clahe->apply(inputImage, stretched);

    return stretched;
}

// Performing the needed transformation and saving the image
void ImageLoader::saveImage(const Mat &img, const string &fileName, bool stretch){
    // Transforming to a format that opencv can save with imwrite
    Mat convertedImage = ImageLoader::convertToGrayScale(img);

    if(stretch){
        Mat stretched = stretchImage(convertedImage);
        saveImageToFileSystem(stretched, fileName);
    }
    else
        saveImageToFileSystem(convertedImage, fileName);
}

void ImageLoader::saveImageToFileSystem(const Mat& img, const string& fileName){
    try {
        imwrite(fileName +".png", img);
    }catch (exception& e){
        cout << e.what() << '\n';
        cerr << "Fatal Error! Couldn't save the image";
        exit(-3);
    }
}