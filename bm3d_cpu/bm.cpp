#include "bm.h"
#include "imghelpercuda.h"

#include <vector>
#include "util.h"

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

float** BlockMatch::getBlocks(float* imageBuffer, int windowSize)
{
    int nbBlocks = windowSize / BlockMatch::BLOCK_SIZE;
    float** blocks = (float**)malloc(nbBlocks * nbBlocks * sizeof(float*));
    for(int i=0; i< nbBlocks * nbBlocks; ++i)
    {
        int x = i * BlockMatch::BLOCK_SIZE % windowSize;
        int y = int(i * BlockMatch::BLOCK_SIZE / windowSize) * BlockMatch::BLOCK_SIZE;
        blocks[i] = _imgHelper.getWindowBuffer(x,y,imageBuffer,BlockMatch::BLOCK_SIZE,windowSize, windowSize);
    }
    return blocks;
}

void BlockMatch::processBM2(float* imageBuffer)
{
}

float** BlockMatch::processBM(float* imageBuffer, float** blocks, int windowSize)
{
    //_imgHelper.writeMatToFile(imageBuffer, "in_1.txt", windowSize, windowSize);
    printf("\nprocess fft (CUFFT)");

    Timer::start();
    cufftComplex* out = _imgHelper.fft(imageBuffer, windowSize);
    Timer::add("[BM] FFT window buffer (GPU - no kernel)");


    /*
    cufftComplex* temp = ImgHelperCuda::get(out, windowSize, windowSize);
    _imgHelper.writeComplexMatToFile(temp, "in_2.txt", windowSize, windowSize/2);
    free(temp);
    temp = 0;
    */

    printf("\nprocess Block matching (GPU)");
    Timer::start();
    int16_t* matching = ImgHelperCuda::ProcessBM(out, BlockMatch::THRESHOLD, windowSize, BlockMatch::BLOCK_SIZE);
    Timer::add("[BM] Process BM (GPU)");

    printf("\n\n----- Matching blocks (TEST) ------\n");
    Timer::start();
    int sizeNormVector = windowSize / BlockMatch::BLOCK_SIZE;
    float** stackedBlocks = (float**)malloc(sizeNormVector * sizeNormVector * sizeof(float*));
    for(int i= 0; i < sizeNormVector * sizeNormVector; ++i)
    {
        //printf("\nB%i:\n", i );
        float* stackBlock = (float*)malloc(BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE * sizeof(float));
        memset(stackBlock, 0, BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE * sizeof(float));
        int offset = 0;
        for(int k = i * BlockMatch::BLOCK_SIZE; k < (i+1) * BlockMatch::BLOCK_SIZE; ++k)
        {
            if(matching[k] > -1)
            {
               // printf("\n\tblock %i", matching[k]);
                memcpy(&stackBlock[offset], &blocks[matching[k]][0], BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE * sizeof(float));

                /*if(i==24)
                {
                    printf(":\n\t\t");
                    for(int q= 0; q < BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE; q++)
                    {
                        printf("%f, ", blocks[matching[k]][q]);
                        if(q % BlockMatch::BLOCK_SIZE == BlockMatch::BLOCK_SIZE -1) printf("\n\t\t");
                    }
                }*/

                offset += BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE;
                //int x = matching[k] * BlockMatch::BLOCK_SIZE;
                //_imgHelper.stackBlock(x, 0, imageBuffer, stackBlock, BlockMatch::BLOCK_SIZE, i * BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE);
            }
        }
        if(i==24)
        {
            printf("\n\t\tStacked array:\n\t\t");
            int n = 0;
            printf("%d: ", n);
            for(int q= 0; q < BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE; q++)
            {
                printf("%f, ", stackBlock[q]);
                if(q % BlockMatch::BLOCK_SIZE == BlockMatch::BLOCK_SIZE -1 ) { ++n; printf("\n\t\t%d: ", n); }
            }
        }
        stackedBlocks[i] = stackBlock;
    }
    printf("\n");
    Timer::add("[BM] 3D Group creation (CPU)");

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
