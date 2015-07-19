#ifndef BM3D_H
#define BM3D_H
#include <cufft.h>
#include "opencv2/core/core.hpp"

using namespace cv;

#include "bm.h"

class BM3D
{
    public:
        BM3D();
        ~BM3D();

        void process(Mat* image);
        void setDebugMode(bool debug);

    private:
        BlockMatch _bm;
        ImgHelper _imgHelper;
        bool _debug;

        void processBasicHT(Mat* image);
        void processBasicHT2(Mat* image);
        void processFinalWien(Mat* image);

        const int WINDOW_SIZE = 40; //40x40 window
};

#endif // BM3D_H
