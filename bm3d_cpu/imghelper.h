#ifndef IMGHELPER_H
#include <cufft.h>
#include "opencv2/core/core.hpp"
using namespace cv;

#include <vector>
#endif // IMGHELPER_H

class ImgHelper
{
    public:
        ~ImgHelper()
        {}

        static void transform2D(Mat* image);
        static void transform2DCuda(Mat* image);
        static void writeMatToFile(cv::Mat& m, const char* filename, int x, int y);
        static void writeMatToFile(float* data, const char* filename, int x, int y);
        static void writeComplexMatToFile(cufftComplex* data, const char* filename, int x, int y);
        static void getWindowBuffer(int x, int y, float* buffer, Mat image, int wSize);

    private:
        ImgHelper();

};
