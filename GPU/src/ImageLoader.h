#ifndef IMAGELOADER_H_
#define IMAGELOADER_H_

#include <iostream>
#include "ImageData.h"
#include <opencv/cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;


/** This class uses OpenCV to read, transform and save images allowing HaraliCU
 * to work with every image (color or gray-scale) and format supported
 * by the OpenCV library
*/
class ImageLoader {
public:
    /**
     * Method called to construct an Image object
     * @param fileName: the path/name of the image to read
     * @param borderType: type of the border to apply to the image
     * @param borderSize: border to apply to each side of the image
     * @param quantize: quantization of grayLevels to apply to the image read
     * @param quantizationMax: maximum gray level when quantization is
     * applied to reduce the graylevels in [0, quantizationMax]
     * @return
     */
    static Image readImage(string fileName, short int borderType, int borderSize, bool quantize, int quantizationMax);
    /**
     * Method used when generating the feature maps with the computed features values
     * @param rows
     * @param cols
     * @param input: list of all the feature values used as intensity in the
     * output image
     * @return image obtained from the provided feature values
     */
    static Mat createDoubleMat(int rows, int cols, const vector<double>& input);
    /**
     * Stores the output feature map
     * @param image to save
     * @param fileName path where to save the image
     * @param linear stretch applied for enhancing the image
     */
    static void saveImage(const Mat &img, const string &fileName,
                          bool stretch = true);
    // DEBUG method
    static void showImagePaused(const Mat& img, const string& windowName);
private:
    /**
     * Invocation of the buily-in OpenCV reading method from filesystem
     * @param fileName
     * @return image matrix
     */
    static Mat readImage(string fileName);
    /**
     * Converting images with colors to gray-scale
     * @param inputImage
     * @return
     */
    static Mat convertToGrayScale(const Mat& inputImage);
    /**
     * Quantize gray levels in set [0, Max]
     * @param inputImage
     * @return
     */
    static Mat quantizeImage(Mat& inputImage, int maxLevel, int minLevelIn, int maxLevelIn);
    /**
     * Returns a stretched image for enhancing the image
     * @param inputImage
     * @return
     */
    static Mat stretchImage(const Mat& inputImage);
    /**
     * Stores the image in the filesystem
     * @param img
     * @param fileName
     */
    static void saveImageToFileSystem(const Mat& img, const string& fileName);
    /**
     * Add borders to the imported image
     * @param img
     * @param borderType
     * @param borderSize
     */
    static void addBorderToImage(Mat &img, short int borderType, int borderSize);

};


#endif /* IMAGELOADER_H_ */
