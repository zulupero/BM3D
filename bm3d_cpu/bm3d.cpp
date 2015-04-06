#include "bm3d.h"

#include "imghelper.h"
#include <vector>


BM3D::BM3D() :
_bm(),
_imgHelper(),
_debug(false)
{
}

BM3D::~BM3D()
{
}

void BM3D::process(Mat* image)
{
    processBasicHT(image);
    processFinalWien(image);
}

void BM3D::processFinalWien(Mat* image)
{
}

void BM3D::processBasicHT(Mat* image)
{
    ///Block mathing (create 3D array)
    std::vector<Mat> planes;
    split(*image, planes);

    //vector<Mat> outplanes(planes.

    for(size_t i= 0; i<planes.size(); ++i)
    {
        planes[i].convertTo(planes[i], CV_32FC1);

        float* windowBuffer = (float*)malloc(BM3D::WINDOW_SIZE * BM3D::WINDOW_SIZE * sizeof(float));
        int outX, outY;
        _imgHelper.getWindowBuffer(0, 0, windowBuffer, planes[i], BM3D::WINDOW_SIZE, &outX, &outY);

        _bm.processBM(windowBuffer, BM3D::BLOCK_SIZE);

        free(windowBuffer);
    }


    ///3D FFT

    ///Filter (HT)

    ///Calculate basic estimates
}

void BM3D::setDebugMode(bool debug)
{
    _debug = debug;
    _bm.setDebugMode(debug);
    _imgHelper.setDebugMode(debug);
}
