#include "bm.h"

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
    _imgHelper.transform2DCuda(imageBuffer, blockSize);
}
