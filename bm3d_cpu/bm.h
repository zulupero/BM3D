#ifndef BM_H
#define BM_H
#include <cufft.h>
#include "opencv2/core/core.hpp"

using namespace cv;

#include "imghelper.h"

class Block
{
    public:
        Block(int x, int y, float* startPtr);
        ~Block();

        int getX() { return _x; }
        int getY() { return _y; }
        float* getStartPtr() { return _startPtr; }

    private:
        int _x;
        int _y;
        float* _startPtr;
};

class BlockMatch
{
    public:
        BlockMatch();
        ~BlockMatch();

        void processBM(float* imageBuffer, int blockSize);
        void setDebugMode(bool debug);

    private:
        ImgHelper _imgHelper;
        int _windowSize;
        bool _debug;
};

#endif // BM_H
