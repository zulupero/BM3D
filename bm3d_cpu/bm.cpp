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

void BlockMatch::processBM(float* imageBuffer, int blockSize)
{
    _imgHelper.writeMatToFile(imageBuffer, "in_1.txt", 40, 40);
    cufftComplex* out = _imgHelper.fft(imageBuffer, 40);
    //_imgHelper.writeComplexMatToFile(out, )
    ImgHelperCuda::Process2DHT(out, 40);

    float* out2 = _imgHelper.ifft(out, 40);

    for(int i = 0; i < 40 * 40; ++i)
        out2[i] = out2[i] / 40;
    _imgHelper.writeMatToFile(out2, "in_3.txt", 40, 40 );

}
