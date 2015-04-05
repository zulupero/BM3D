#ifndef BM_H
#define BM_H
#include <cufft.h>
#include "opencv2/core/core.hpp"

using namespace cv;

#include "imghelper.h"


class BlockMatch
{
    public:
        BlockMatch(int windowSize);
        ~BlockMatch();

        void processWindowBM(Mat* image);
        void setDebugMode(bool debug);

    private:
        ImgHelper _imgHelper;
        int _windowSize;
        bool _debug;
};

#endif // BM_H
