#include "bm.h"

#include <vector>

BlockMatch::BlockMatch(int windowSize) :
_imgHelper(), _windowSize(windowSize), _debug(false)
{
    _imgHelper.setDebugMode(_debug);
}

BlockMatch::~BlockMatch()
{
}

void BlockMatch::setDebugMode(bool debug)
{
    _debug = debug;
    _imgHelper.setDebugMode(debug);
}

void BlockMatch::processWindowBM(cv::Mat* image)
{
    std::vector<Mat> planes;
    split(*image, planes);

    //vector<Mat> outplanes(planes.

    for(size_t i= 0; i<planes.size(); ++i)
    {
        planes[i].convertTo(planes[i], CV_32FC1);

        float* windowBuffer = (float*)malloc(_windowSize * _windowSize * sizeof(float));
        int outX, outY;
        _imgHelper.getWindowBuffer(9, 0, windowBuffer, planes[i], _windowSize, &outX, &outY);
        _imgHelper.transform2DCuda(windowBuffer, 8);
        free(windowBuffer);
    }
}
