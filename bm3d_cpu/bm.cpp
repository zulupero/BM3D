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



float** BlockMatch::processBM(float* imageBuffer, int windowSize)
{
    _imgHelper.writeMatToFile(imageBuffer, "in_1.txt", windowSize, windowSize);
    printf("\nprocess fft (CUFFT)");
    cufftComplex* out = _imgHelper.fft(imageBuffer, windowSize);

    //cufftComplex* temp = ImgHelperCuda::get(out, windowSize, windowSize);
    //_imgHelper.writeComplexMatToFile(temp, "in_2.txt", windowSize, windowSize/2);
    //free(temp);
    //temp = 0;

    printf("\nprocess Block matching (GPU)");
    int16_t* matching = ImgHelperCuda::ProcessBM(out, BlockMatch::GAMMA, windowSize, BlockMatch::BLOCK_SIZE);

    printf("\n\n----- Matching blocks (TEST) ------\n");
    int sizeNormVector = windowSize / BlockMatch::BLOCK_SIZE;
    float** stackedBlocks = (float**)malloc(sizeNormVector * sizeNormVector * sizeof(float*));
    for(int i= 0; i < sizeNormVector * sizeNormVector; ++i)
    {
        printf("\nB%i:\n", i );
        float* stackBlock = (float*)malloc(BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE * sizeof(float));
        memset(stackBlock, 0, BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE * sizeof(float));
        for(int k = i * BlockMatch::BLOCK_SIZE; k < (i+1) * BlockMatch::BLOCK_SIZE; ++k)
        {
            if(matching[k] > -1)
            {
                printf("\n\tblock %i", matching[k]);
                int x = matching[k] * BlockMatch::BLOCK_SIZE;
                //_imgHelper.stackBlock(x, 0, imageBuffer, stackBlock, BlockMatch::BLOCK_SIZE, i * BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE);
            }
        }
        stackedBlocks[i] = stackBlock;
    }
    printf("\n");

    free(matching);
    return stackedBlocks;

    //temp = ImgHelperCuda::get(out, windowSize, windowSize);
    //_imgHelper.writeComplexMatToFile(temp, "in_3.txt", windowSize, windowSize/2);
    //free(temp);
    //temp = 0;

    //printf("\nprocess iFFT (only for the test)");
    //float* out2 = _imgHelper.ifft(out, windowSize);

    //float size_ = windowSize * windowSize;
    //for(int i = 0; i < size_; ++i)
     //   out2[i] = out2[i] / size_;
    //_imgHelper.writeMatToFile(out2, "in_4.txt", windowSize, windowSize);
}
