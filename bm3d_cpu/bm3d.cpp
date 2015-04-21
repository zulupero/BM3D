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
    printf("\n-----START BM3D ----");
    processBasicHT(image);
    processFinalWien(image);
    printf("\n-----END BM3D ----\n");
}

void BM3D::processFinalWien(Mat* image)
{
    printf("\n>>> FINAL ESTIMATE (WIEN)");
}

void BM3D::processBasicHT(Mat* image)
{
    printf("\n>>> BASIC ESTIMATE (HT)");
    ///Block mathing (create 3D array)
    std::vector<Mat> planes;
    split(*image, planes);

    //vector<Mat> outplanes(planes.

    for(size_t i= 0; i<planes.size(); ++i)
    {
        planes[i].convertTo(planes[i], CV_32FC1);

        float* windowBuffer = (float*)malloc(BM3D::WINDOW_SIZE * BM3D::WINDOW_SIZE * sizeof(float));
        int outX, outY;

        printf("\nTreat window (0,0)");
        _imgHelper.getWindowBuffer(40, 40, windowBuffer, planes[i], BM3D::WINDOW_SIZE, &outX, &outY);

        float** stackedBlocks = _bm.processBM(windowBuffer, BM3D::WINDOW_SIZE);

        ///Filter (HT)

        ///Calculate basic estimates

        free(windowBuffer);
    }
}

void BM3D::setDebugMode(bool debug)
{
    _debug = debug;
    _bm.setDebugMode(debug);
    _imgHelper.setDebugMode(debug);
}
