#ifndef IMGHELPER_H
#define IMGHELPER_H
#include <cufft.h>
#include "opencv2/core/core.hpp"
using namespace cv;

#include <vector>


class ImgHelper
{
    public:
        ImgHelper() : _debug(false)
        {}

        ~ImgHelper()
        {}

        void transform2D(Mat* image);
        void transform2DCuda(Mat* image);
        cufftComplex* fft(float* imageBuffer, int n1);
        float* ifft(cufftComplex* imageBuffer, int n1);
        void transform2DCuda(float* imageBuffer, int n1);
        void writeMatToFile(cv::Mat& m, const char* filename, int x, int y);
        void writeMatToFile(float* data, const char* filename, int x, int y);
        void writeComplexMatToFile(cufftComplex* data, const char* filename, int x, int y);
        void getWindowBuffer(int x, int y, float* buffer, Mat image, int wSize, int* outX, int* outY);
        void setDebugMode(bool debug);

    private:
        bool _debug;

};

#endif // IMGHELPER_H
