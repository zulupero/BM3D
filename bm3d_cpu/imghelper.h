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

        void Process3DHT(cufftComplex* imageBuffer, int n1);
        void transform2D(Mat* image);
        void transform2DCuda(Mat* image);
        cufftComplex* fft(float* imageBuffer, int n1);
        float* ifft(cufftComplex* imageBuffer, int n1);
        cufftComplex* fft3D(float* imageBuffer, int n1);
        float* ifft3D(cufftComplex* imageBuffer, int n1);
        void transform2DCuda(float* imageBuffer, int n1);
        void writeMatToFile(cv::Mat& m, const char* filename, int x, int y);
        void writeMatToFile(float* data, const char* filename, int x, int y);
        void writeComplexMatToFile(cufftComplex* data, const char* filename, int x, int y);
        float* getWindowBuffer(int x, int y,  Mat image, int wSize);
        float* getWindowBuffer(int x, int y, float* source, int wSize, int width, int height);
        void stackBlock(int x, int y, float* buffer, float* stackBlock, int bSize, int position);
        void setDebugMode(bool debug);

    private:
        bool _debug;

};

#endif // IMGHELPER_H
