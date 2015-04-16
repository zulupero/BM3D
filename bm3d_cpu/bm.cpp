#include "bm.h"
#include "imghelpercuda.h"

#include <vector>

Block::Block(int x, int y, float* startPtr) :
_x(x), _y(y), _startPtr(startPtr)
{
}

Block::~Block()
{
}


BlockMatch::BlockMatch() :
_imgHelper(), _debug(false)
{
}

BlockMatch::~BlockMatch()
{
}

void BlockMatch::setDebugMode(bool debug)
{
    _debug = debug;
    _imgHelper.setDebugMode(debug);
}

void BlockMatch::processBM(float* imageBuffer, int windowSize)
{
    _imgHelper.writeMatToFile(imageBuffer, "in_1.txt", windowSize, windowSize);
    printf("\nprocess fft");
    cufftComplex* out = _imgHelper.fft(imageBuffer, windowSize);

    cufftComplex* temp = ImgHelperCuda::get(out, windowSize, windowSize);
    _imgHelper.writeComplexMatToFile(temp, "in_2.txt", windowSize, windowSize/2);
    free(temp);
    temp = 0;

    printf("\nprocess Block matching");
    ImgHelperCuda::ProcessBM(out, BlockMatch::GAMMA, windowSize, BlockMatch::BLOCK_SIZE);



    temp = ImgHelperCuda::get(out, windowSize, windowSize);
    _imgHelper.writeComplexMatToFile(temp, "in_3.txt", windowSize, windowSize/2);
    free(temp);
    temp = 0;

    printf("\nprocess iFFT (only for the test)");
    float* out2 = _imgHelper.ifft(out, windowSize);

    float size_ = windowSize * windowSize;
    for(int i = 0; i < size_; ++i)
        out2[i] = out2[i] / size_;
    _imgHelper.writeMatToFile(out2, "in_4.txt", windowSize, windowSize);
}
