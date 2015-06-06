#include "bm3d.h"

#include "imghelper.h"
#include "util.h"
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

        Size s = planes[i].size();
        int x = 0;
        int y = 0;
        while(y < s.height)
        {
            Timer::start();
            float* windowBuffer = _imgHelper.getWindowBuffer(x, y, planes[i], BM3D::WINDOW_SIZE);
            Timer::add("GetWindowBuffer");


            ///Block matching
            Timer::start();
            float** blocks = _bm.getBlocks(windowBuffer, BM3D::WINDOW_SIZE);
            Timer::add("GetBlocks");

            float** stackedBlocks = _bm.processBM(windowBuffer, blocks, BM3D::WINDOW_SIZE);

            ///3D transform + Filter (HT)
            Timer::start();
            int nbOfBlocks = BM3D::WINDOW_SIZE / BlockMatch::BLOCK_SIZE;
            for(int i=0; i< nbOfBlocks * nbOfBlocks; ++i)
            {
                cufftComplex* out = _imgHelper.fft3D(stackedBlocks[i], BlockMatch::BLOCK_SIZE);
                _imgHelper.Process3DHT(out, BlockMatch::BLOCK_SIZE);
                float* out2 = _imgHelper.ifft3D(out, BlockMatch::BLOCK_SIZE);

                int fSize = BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE * BlockMatch::BLOCK_SIZE;

                if(i==24)
                {
                    printf(":\n\t\t");
                    for(int q= 0; q <  fSize; ++q)
                    {
                        out2[q] = out2[q]/fSize;
                        printf("%f, ", (out2[q]/fSize));
                        if(q % BlockMatch::BLOCK_SIZE == BlockMatch::BLOCK_SIZE -1) printf("\n\t\t");
                    }
                }

            }
            Timer::add("3D Filter");

            ///Calculate basic estimates

            free(windowBuffer);


            x += BM3D::WINDOW_SIZE;
            if(x > s.width)
            {
                x = 0;
                y += BM3D::WINDOW_SIZE;
            }
        //}

        Timer::showResults();
    }
}

void BM3D::setDebugMode(bool debug)
{
    _debug = debug;
    _bm.setDebugMode(debug);
    _imgHelper.setDebugMode(debug);
}
