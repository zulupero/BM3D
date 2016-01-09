#include <stdio.h>

#include "bm3d.h"
#include "utilities.h"
#include "timeutil.h"

#include <string>

BM3D::BM3D_Context BM3D::context;

 __constant__ float kaiserWindow[] = { 0.1924,    0.2989,    0.3846 ,   0.4325 ,   0.4325   , 0.3846  ,  0.2989    ,0.1924,
        0.2989,    0.4642 ,   0.5974  ,  0.6717 ,   0.6717  ,  0.5974   , 0.4642 ,   0.2989,
        0.3846 ,   0.5974 ,   0.7688  ,  0.8644 ,   0.8644   , 0.7688  ,  0.5974 ,   0.3846,
        0.4325 ,   0.6717  ,  0.8644  ,  0.9718 ,   0.9718   , 0.8644 ,   0.6717  ,  0.4325,
        0.4325 ,   0.6717 ,   0.8644    ,0.9718   , 0.9718  ,  0.8644 ,   0.6717  ,  0.4325,
        0.3846  ,  0.5974,    0.7688 ,   0.8644 ,   0.8644 ,   0.7688  ,  0.5974  ,  0.3846,
        0.2989  ,  0.4642 ,   0.5974 ,   0.6717,    0.6717 ,   0.5974  ,  0.4642 ,   0.2989,
        0.1924 ,   0.2989  ,  0.3846  ,  0.4325 ,   0.4325 ,   0.3846 ,   0.2989  ,  0.1924};

__constant__ double DCTv8matrixT[] =
{
    0.353553390593274f,   0.353553390593274f,   0.353553390593274f,   0.353553390593274f,   0.353553390593274f,   0.353553390593274f,   0.353553390593274f,   0.353553390593274f,
   0.490392640201615f,   0.415734806151273f,   0.277785116509801f,   0.097545161008064f,  -0.097545161008064f,  -0.277785116509801f,  -0.415734806151273f,  -0.490392640201615f,
   0.461939766255643f,   0.191341716182545f,  -0.191341716182545f,  -0.461939766255643f,  -0.461939766255643f,  -0.191341716182545f,   0.191341716182545f,   0.461939766255643f,
   0.415734806151273f,  -0.097545161008064f,  -0.490392640201615f,  -0.277785116509801f,   0.277785116509801f,   0.490392640201615f,   0.097545161008064f,  -0.415734806151272f,
   0.353553390593274f,  -0.353553390593274f,  -0.353553390593274f,   0.353553390593274f,   0.353553390593274f,  -0.353553390593273f,  -0.353553390593274f,   0.353553390593273f,
   0.277785116509801f,  -0.490392640201615f,   0.097545161008064f,   0.415734806151273f,  -0.415734806151273f,  -0.097545161008065f,   0.490392640201615f,  -0.277785116509801f,
   0.191341716182545f,  -0.461939766255643f,   0.461939766255643f,  -0.191341716182545f,  -0.191341716182545f,   0.461939766255644f,  -0.461939766255644f,   0.191341716182543f,
   0.097545161008064f,  -0.277785116509801f,   0.415734806151273f,  -0.490392640201615f,   0.490392640201615f,  -0.415734806151272f,   0.277785116509802f,  -0.097545161008063
}; 

__constant__ double DCTv8matrix[] = 
{
    0.353553390593274f,   0.490392640201615f  , 0.461939766255643f,  0.415734806151273f,   0.353553390593274f, 0.277785116509801f ,  0.191341716182545f, 0.097545161008064f,
   0.353553390593274f,   0.415734806151273f,   0.191341716182545f,  -0.097545161008064f,  -0.353553390593274f,  -0.490392640201615f,  -0.461939766255643f,  -0.277785116509801f,
   0.353553390593274f,   0.277785116509801f,  -0.191341716182545f,  -0.490392640201615f,  -0.353553390593274f,   0.097545161008064f,   0.461939766255643f,   0.415734806151273f,
   0.353553390593274f,   0.097545161008064f,  -0.461939766255643f,  -0.277785116509801f,   0.353553390593274f,   0.415734806151273f,  -0.191341716182545f,  -0.490392640201615f,
   0.353553390593274f,  -0.097545161008064f,  -0.461939766255643f,   0.277785116509801f,   0.353553390593274f,  -0.415734806151273f,  -0.191341716182545f,   0.490392640201615f,
   0.353553390593274f,  -0.277785116509801f,  -0.191341716182545f,   0.490392640201615f,  -0.353553390593273f,  -0.097545161008065f,   0.461939766255644f,  -0.415734806151272f,
   0.353553390593274f,  -0.415734806151273f,   0.191341716182545f,   0.097545161008064f,  -0.353553390593274f,   0.490392640201615f,  -0.461939766255644f,   0.277785116509802f,
   0.353553390593274f,  -0.490392640201615f,   0.461939766255643f,  -0.415734806151272f,   0.353553390593273f,  -0.277785116509801f,   0.191341716182543f, -0.097545161008063f
};

__constant__ float coef_norm[] = 
{
0.031250, 0.044194, 0.044194, 0.044194, 0.044194, 0.044194, 0.044194, 0.044194, 
0.044194, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 
0.044194, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 
0.044194, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 
0.044194, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 
0.044194, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 
0.044194, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 
0.044194, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500
}; 

__constant__ float coef_norm_inv[] =
{
2.000000, 1.414214, 1.414214, 1.414214, 1.414214, 1.414214, 1.414214, 1.414214, 
1.414214, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
1.414214, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
1.414214, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
1.414214, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
1.414214, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
1.414214, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
1.414214, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void BM3D::gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void BM3D::BM3D_dispose()
{
}

void BM3D::BM3D_Initialize(BM3D::SourceImage img, BM3D::SourceImage imgOrig, int width, int height, int pHard, int hardLimit, int wienLimit, double hardThreshold, int sigma, int windowSize, bool debug)
{
    printf("\n--> Execution on Tesla K40c");
    if(cudaSuccess != cudaSetDevice(0)) printf("\n\tNo device 0 available");

    if(debug)
    {
        int sz = 1048576 * 1024;
        cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);
    }

    printf("\nBM3D context initialization");
    BM3D::context.hardLimit = hardLimit;
    BM3D::context.wienLimit = wienLimit;
    BM3D::context.hardThreshold = hardThreshold;
    BM3D::context.sigma = sigma;
    
    BM3D::context.halfWindowSize = windowSize/2;
    BM3D::context.offset = BM3D::context.halfWindowSize + 8;
    int w2 = width + 2 * BM3D::context.offset;
    int h2 = height + 2 * BM3D::context.offset;
    BM3D::context.nbBlocksPerWindow = int(windowSize/pHard) * int(windowSize/pHard);
    BM3D::context.nbBlocksIntern = (width / pHard) * (height /pHard);    
    BM3D::context.nbBlocks = BM3D::context.nbBlocksIntern * BM3D::context.nbBlocksPerWindow; 
    BM3D::context.widthBlocksIntern = (width / pHard);
    BM3D::context.widthBlocksWindow = int(windowSize/pHard);
    BM3D::context.windowSize = (BM3D::context.nbBlocksPerWindow * 4) + 4;
    BM3D::context.blockSize = (BM3D::context.nbBlocksPerWindow << 6) + 64;
    

    BM3D::context.img_widthOrig = width; 
    BM3D::context.img_heightOrig= height;
    BM3D::context.img_width = w2; 
    BM3D::context.img_height= h2;
    BM3D::context.pHard = pHard;
    BM3D::context.sourceImage = img;
    BM3D::context.origImage = imgOrig;

    printf("\n\tNumber of blocks          = %d", BM3D::context.nbBlocks);
    printf("\n\tNumber of blocks (line)   = %d", BM3D::context.widthBlocksIntern);
    printf("\n\tNumber of blocks (w line) = %d", BM3D::context.widthBlocksWindow);
    printf("\n\tNumber of blocks (window) = %d", BM3D::context.nbBlocksPerWindow);
    printf("\n\tOffset                    = %d", BM3D::context.offset);
    printf("\n\tWidth                     = %d", BM3D::context.img_width);
    printf("\n\tHeight                    = %d", BM3D::context.img_height);
    printf("\n\tDevice image              = %f Mb", (width * height * sizeof(float)/1024.00 / 1024.00));
    printf("\n\tNoisy image               = %f Mb", (w2 * h2 * sizeof(float)/1024.00 / 1024.00)); 
    printf("\n\tBasic image               = %f Mb", (w2 * h2 * sizeof(float)/1024.00 / 1024.00)); 
    printf("\n\tWindow map                = %f Mb", (BM3D::context.nbBlocksIntern * BM3D::context.windowSize * sizeof(int)/1024.00 / 1024.00));  
    printf("\n\tBlocks 3D                 = %f Mb", (BM3D::context.nbBlocksIntern * 32 * 66 * sizeof(double)/1024.00 / 1024.00)); //+ x,y 
    printf("\n\tBlocks 3D (orig)          = %f Mb", (BM3D::context.nbBlocksIntern * 32 * 66 * sizeof(double)/1024.00 / 1024.00)); //+ x,y
    printf("\n\tNP array                  = %f Mb", (BM3D::context.nbBlocksIntern * sizeof(int)/1024.00 / 1024.00));
    printf("\n\tWP array                  = %f Mb", (BM3D::context.nbBlocksIntern * sizeof(double)/1024.00 / 1024.00));
    printf("\n\tEstimates array           = %f Mb", (w2 * h2 * 2 * sizeof(float)/1024.00 / 1024.00));
    printf("\n\tSimilar blocks array      = %f Mb", (BM3D::context.nbBlocksIntern * sizeof(int)/1024.00 / 1024.00));

    gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, width * height * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.deviceImage, &img[0], width * height * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&BM3D::context.basicImage, w2 * h2 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.basicImage, 0, w2 * h2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.noisyImage, w2 * h2 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.noisyImage, 0, w2 * h2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.estimates, w2 * h2 * 2 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.estimates, 0, w2 * h2 * 2 * sizeof(float)));

    gpuErrchk(cudaMalloc(&BM3D::context.blocks, BM3D::context.nbBlocksIntern * BM3D::context.blockSize * sizeof(double)));
    gpuErrchk(cudaMalloc(&BM3D::context.windowMap, BM3D::context.nbBlocksIntern * BM3D::context.windowSize * sizeof(int)));
    gpuErrchk(cudaMemset(BM3D::context.windowMap, -1, BM3D::context.nbBlocksIntern * BM3D::context.windowSize * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.blocks3D, BM3D::context.nbBlocksIntern * 32 * 66 * sizeof(double)));
    gpuErrchk(cudaMemset(BM3D::context.blocks3D, 0, BM3D::context.nbBlocksIntern * 32 * 66 * sizeof(double)));
    gpuErrchk(cudaMalloc(&BM3D::context.blocks3DOrig, BM3D::context.nbBlocksIntern * 32 * 66 * sizeof(double)));
    gpuErrchk(cudaMemset(BM3D::context.blocks3DOrig, 0, BM3D::context.nbBlocksIntern * 32 * 66 * sizeof(double)));
    gpuErrchk(cudaMalloc(&BM3D::context.npArray, BM3D::context.nbBlocksIntern  * sizeof(int)));
    gpuErrchk(cudaMemset(BM3D::context.npArray, 0, BM3D::context.nbBlocksIntern  * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.wpArray, BM3D::context.nbBlocksIntern  * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.wpArray, 0, BM3D::context.nbBlocksIntern  * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.nbSimilarBlocks, BM3D::context.nbBlocksIntern  * sizeof(int)));
    gpuErrchk(cudaMemset(BM3D::context.nbSimilarBlocks, 0, BM3D::context.nbBlocksIntern  * sizeof(int)));
}

void BM3D::BM3D_Run()
{
    printf("\n\nRun BM3D"); 
    BM3D_BasicEstimate();
    BM3D_FinalEstimate();
}

void BM3D::BM3D_SaveImage(bool final)
{
    float* denoisedImage = (float*)malloc(BM3D::context.img_widthOrig * BM3D::context.img_heightOrig * sizeof(float));
    gpuErrchk(cudaMemcpy(&denoisedImage[0], BM3D::context.deviceImage, BM3D::context.img_widthOrig * BM3D::context.img_heightOrig * sizeof(float), cudaMemcpyDeviceToHost));
    std::string filename((final) ? "final.png" : "basic.png");
    //if(final)
    {
        double psrn = 0, rmse =0;
        compute_psnr(BM3D::context.origImage, denoisedImage, &psrn, &rmse);  
        printf("\n\t\tpsrn = %f, rmse = %f", psrn, rmse);
    }

    save_image(filename.c_str(), denoisedImage, BM3D::context.img_widthOrig, BM3D::context.img_heightOrig, 1);
}

void BM3D::BM3D_FinalEstimate()
{
    gpuErrchk(cudaMemset(BM3D::context.windowMap, -1, BM3D::context.nbBlocksIntern * BM3D::context.windowSize * sizeof(int))); 
    gpuErrchk(cudaMemset(BM3D::context.basicImage, 0, BM3D::context.img_width * BM3D::context.img_height * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.nbSimilarBlocks, 0, BM3D::context.nbBlocksIntern  * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.wpArray, 0, BM3D::context.nbBlocksIntern  * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.estimates, 0, BM3D::context.img_width * BM3D::context.img_height * 2 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.blocks3D, 0, BM3D::context.nbBlocksIntern * 32 * 66 * sizeof(double)));

    printf("\n\tFinal estimates (2 step)");
    Timer::start(); 
    BM3D_CreateWindow(true);    
    BM3D_BlockMatching(true);
    BM3D_ShowDistance(1000);
    BM3D_Create3DBlocks(true);
    BM3D_WienFilter();
    BM3D_Inverse3D(true);
    BM3D_Aggregation(true);
    BM3D_InverseShift();
    Timer::add("BM3D-Final estimates");
    BM3D_SaveImage(true);
}

void BM3D::BM3D_BasicEstimate()
{
    printf("\n\tBasic estimates (1 step)");
    Timer::start();
    BM3D_CreateWindow();
    BM3D_BlockMatching();
    BM3D_ShowDistance(1000);
    BM3D_Create3DBlocks();
    BM3D_HardThresholdFilter();
    BM3D_ShowBlock(1000);
    BM3D_Inverse3D();
    BM3D_Aggregation();
    BM3D_InverseShift();
    Timer::add("BM3D-Basic estimates");
    BM3D_SaveImage();
}


__global__
void WienFilter(double* blocks3D, double* blocks3DOrig, int size, int* similarBlocks, int sigma2, float* wpArray)
{    
    int block = (blockIdx.y * size) + blockIdx.x;
    if(blockIdx.z < similarBlocks[block])
    {
        float coef = 1.0f / (float)similarBlocks[block];
        int blockPixelIndex = (block * 2112) + (blockIdx.z * 66) + (threadIdx.y << 3) + threadIdx.x;
        float estimateValue = blocks3D[blockPixelIndex];
        float value = (estimateValue * estimateValue) * coef;
        value = value/ (value + (float)sigma2);
        blocks3D[blockPixelIndex] = blocks3DOrig[blockPixelIndex] * value * coef;
        //if(block==1000) printf("\nblock = %d, z = %d, val = %f, orig=%f, value = %f, coef = %f, calc val = %f", block, blockIdx.z, estimateValue, blocks3DOrig[blockPixelIndex], value, coef, blocks3D[blockPixelIndex] );
        atomicAdd(&wpArray[block], value);
    }
}

__global__
void CalculateFinalWP(int size, int sigma2, float* wpArray)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    wpArray[block] = 1.0f / (sigma2 * wpArray[block]);
}

void BM3D::BM3D_WienFilter()
{
    {    
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, 32);
        dim3 numThreads(8, 8);
        WienFilter<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.blocks3DOrig, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks, (BM3D::context.sigma * BM3D::context.sigma), BM3D::context.wpArray); 
        cudaDeviceSynchronize();
    }
    BM3D_ShowBlock(1000);
    {    
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(1);
        CalculateFinalWP<<<numBlocks,numThreads>>>(BM3D::context.widthBlocksIntern, (BM3D::context.sigma * BM3D::context.sigma), BM3D::context.wpArray); 
        cudaDeviceSynchronize();
    }   
}

__global__
void pre_aggregation(double* blocks3D, float* wpArray, int size, float* estimates, int width, int* similarBlocks)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    if(blockIdx.z < similarBlocks[block])
    {
        int block3DIndex = (block * 2112) + (blockIdx.z * 66);
        int xImg = (int)blocks3D[block3DIndex + 64];
        int yImg = (int)blocks3D[block3DIndex + 65];
        int xPixel = xImg + threadIdx.x;
        int yPixel = yImg + threadIdx.y;
        int estimateIndex = ((yPixel * width) + xPixel) << 1;
        int kaiserIndex = (threadIdx.y << 3) + threadIdx.x;
        if(xPixel == 279 && yPixel == 246) printf("\nblock = %d, x = %d, y = %d, val = %f, kval = %f, wp = %f, calc val = %f", block, xPixel, yPixel, blocks3D[block3DIndex + kaiserIndex],kaiserWindow[kaiserIndex], wpArray[block], kaiserWindow[kaiserIndex] * wpArray[block] * blocks3D[block3DIndex + kaiserIndex]);
        atomicAdd(&estimates[estimateIndex], (kaiserWindow[kaiserIndex] * wpArray[block] * blocks3D[block3DIndex + kaiserIndex]));
        atomicAdd(&estimates[estimateIndex+1], (kaiserWindow[kaiserIndex] * wpArray[block]));
    }
}

__global__
void aggregation(float* estimates, float* basicImage, int size, int offset)
{
    int basicImageIndex = (blockIdx.y * size) + blockIdx.x;
    int estimateIndex = (basicImageIndex << 1);
    if(estimates[estimateIndex+1] > 0)
    {
        basicImage[basicImageIndex] = estimates[estimateIndex]/estimates[estimateIndex+1];
        //if(basicImage[basicImageIndex] > 0 && basicImage[basicImageIndex] < 10 && blockIdx.y -offset > 250 && blockIdx.y - offset < 256) printf("\nx = %d, y = %d, val = %f, num = %f, den = %f", blockIdx.x, blockIdx.y, basicImage[basicImageIndex], estimates  [estimateIndex], estimates[estimateIndex+1]);
    }
}

void BM3D::BM3D_Aggregation(bool final)
{
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, (final) ? 32 : 16);
        dim3 numThreads(8, 8);
        pre_aggregation<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.wpArray, BM3D::context.widthBlocksIntern, BM3D::context.estimates, BM3D::context.img_width, BM3D::context.nbSimilarBlocks); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.img_width, BM3D::context.img_height);
        dim3 numThreads(1);
        aggregation<<<numBlocks,numThreads>>>(BM3D::context.estimates, BM3D::context.basicImage, BM3D::context.img_width, BM3D::context.offset); 
        cudaDeviceSynchronize();   
    }
}

__global__
void HadamarTransform32(double* blocks3D, int size, int* nbSimilarBlocks, int applycoef)
{
    int block = ((blockIdx.y * size) + blockIdx.x);
    int maxBlock = nbSimilarBlocks[block];
    float MUL = (applycoef == 1) ? (1.0f / (float)maxBlock ) : 1.0;
    int block3DIndex = block * 2112 + (threadIdx.y << 3) + threadIdx.x;
    //we can assume that the top-left corner of the basic image always has a pixel egal to 0 due to the shift 
    //of the image. 
    int index = block3DIndex;
    
    double a = blocks3D[index];
    index += 66;
    double b = blocks3D[index]; 
    index += 66;
    double c = blocks3D[index];
    index += 66;
    double d = blocks3D[index]; 
    index += 66;
    double e = blocks3D[index];
    index += 66;
    double f = blocks3D[index];
    index += 66;
    double g = blocks3D[index];
    index += 66;
    double h = blocks3D[index]; 
    index += 66;
    double i = blocks3D[index];
    index += 66;
    double j = blocks3D[index];
    index += 66;
    double k = blocks3D[index];
    index += 66;
    double l = blocks3D[index];
    index += 66;
    double m = blocks3D[index];
    index += 66;
    double n = blocks3D[index];
    index += 66;
    double o = blocks3D[index];
    index += 66;
    double p = blocks3D[index]; 
    index += 66;

    double a2 = blocks3D[index];
    index += 66;
    double b2 = blocks3D[index];
    index += 66;
    double c2 = blocks3D[index];
    index += 66;
    double d2 = blocks3D[index];
    index += 66;
    double e2 = blocks3D[index];
    index += 66;
    double f2 = blocks3D[index];
    index += 66;
    double g2 = blocks3D[index];
    index += 66;
    double h2 = blocks3D[index];
    index += 66;
    double i2 = blocks3D[index];
    index += 66;
    double j2 = blocks3D[index];
    index += 66;
    double k2 = blocks3D[index];
    index += 66;
    double l2 = blocks3D[index];
    index += 66;
    double m2 = blocks3D[index];
    index += 66;
    double n2 = blocks3D[index];
    index += 66;
    double o2 = blocks3D[index];
    index += 66;
    double p2 = blocks3D[index]; //32

    //if(block == 1000) printf("\n\n%f, %f, %f, %f, %f, %f,%f, %f\n%f, %f, %f, %f, %f, %f,%f, %f\n%f, %f, %f, %f, %f, %f,%f, %f\n%f, %f, %f, %f, %f, %f,%f, %f",
      //                      a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,a2,b2,c2,d2,e2,f2,g2,h2,i2,j2,k2,l2,m2,n2,o2,p2);

    index = block3DIndex;
    blocks3D[index] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p + a2+b2+c2+d2+e2+f2+g2+h2+i2+j2+k2+l2+m2+n2+o2+p2)*MUL;
    index += 66;
    blocks3D[index] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p + a2-b2+c2-d2+e2-f2+g2-h2+i2-j2+k2-l2+m2-n2+o2-p2)*MUL; //2
    if(maxBlock > 2)
    {
        index += 66;
        blocks3D[index] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p + a2+b2-c2-d2+e2+f2-g2-h2+i2+j2-k2-l2+m2+n2-o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p + a2-b2-c2+d2+e2-f2-g2+h2+i2-j2-k2+l2+m2-n2-o2+p2)*MUL; //4
    }
    if(maxBlock > 4)
    {
        index += 66;
        blocks3D[index] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p + a2+b2+c2+d2-e2-f2-g2-h2+i2+j2+k2+l2-m2-n2-o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p + a2-b2+c2-d2-e2+f2-g2+h2+i2-j2+k2-l2-m2+n2-o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p + a2+b2-c2-d2-e2-f2+g2+h2+i2+j2-k2-l2-m2-n2+o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p + a2-b2-c2+d2-e2+f2+g2-h2+i2-j2-k2+l2-m2+n2+o2-p2)*MUL; //8
    }
    if(maxBlock > 8)
    {
        index += 66;
        blocks3D[index] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p + a2+b2+c2+d2+e2+f2+g2+h2-i2-j2-k2-l2-m2-n2-o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p + a2-b2+c2-d2+e2-f2+g2-h2-i2+j2-k2+l2-m2+n2-o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p + a2+b2-c2-d2+e2+f2-g2-h2-i2-j2+k2+l2-m2-n2+o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p + a2-b2-c2+d2+e2-f2-g2+h2-i2+j2+k2-l2-m2+n2+o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p + a2+b2+c2+d2-e2-f2-g2-h2-i2-j2-k2-l2+m2+n2+o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p + a2-b2+c2-d2-e2+f2-g2+h2-i2+j2-k2+l2+m2-n2+o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p + a2+b2-c2-d2-e2-f2+g2+h2-i2-j2+k2+l2+m2+n2-o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p + a2-b2-c2+d2-e2+f2+g2-h2-i2+j2+k2-l2+m2-n2-o2+p2)*MUL; //16
    }
    if(maxBlock > 16)    
    {
        index += 66;
        blocks3D[index] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p - a2-b2-c2-d2-e2-f2-g2-h2-i2-j2-k2-l2-m2-n2-o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p - a2+b2-c2+d2-e2+f2-g2+h2-i2+j2-k2+l2-m2+n2-o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p - a2-b2+c2+d2-e2-f2+g2+h2-i2-j2+k2+l2-m2-n2+o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p - a2+b2+c2-d2-e2+f2+g2-h2-i2+j2+k2-l2-m2+n2+o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p - a2-b2-c2-d2+e2+f2+g2+h2-i2-j2-k2-l2+m2+n2+o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p - a2+b2-c2+d2+e2-f2+g2-h2-i2+j2-k2+l2+m2-n2+o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p - a2-b2+c2+d2+e2+f2-g2-h2-i2-j2+k2+l2+m2+n2-o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p - a2+b2+c2-d2+e2-f2-g2+h2-i2+j2+k2-l2+m2-n2-o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p - a2-b2-c2-d2-e2-f2-g2-h2+i2+j2+k2+l2+m2+n2+o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p - a2+b2-c2+d2-e2+f2-g2+h2+i2-j2+k2-l2+m2-n2+o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p - a2-b2+c2+d2-e2-f2+g2+h2+i2+j2-k2-l2+m2+n2-o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p - a2+b2+c2-d2-e2+f2+g2-h2+i2-j2-k2+l2+m2-n2-o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p - a2-b2-c2-d2+e2+f2+g2+h2+i2+j2+k2+l2-m2-n2-o2-p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p - a2+b2-c2+d2+e2-f2+g2-h2+i2-j2+k2-l2-m2+n2-o2+p2)*MUL;  
        index += 66;
        blocks3D[index] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p - a2-b2+c2+d2+e2+f2-g2-h2+i2+j2-k2-l2-m2-n2+o2+p2)*MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p - a2+b2+c2-d2+e2-f2-g2+h2+i2-j2-k2+l2-m2+n2+o2-p2)*MUL; //32
    }
  
}


__global__
void HadamarTransform16(double* blocks3D, int size, int* nbSimilarBlocks, int applycoef)
{
    int block = ((blockIdx.y * size) + blockIdx.x);
    int maxBlock = nbSimilarBlocks[block];
    float MUL = (applycoef == 1) ? (1.0f / (float)maxBlock ) : 1.0;
    int block3DIndex = block * 2112 + (threadIdx.y << 3) + threadIdx.x;
    //we can assume that the top-left corner of the basic image always has a pixel egal to 0 due to the shift 
    //of the image. 
    int index = block3DIndex;
    double a = blocks3D[index];
    index += 66;
    double b = blocks3D[index];
    index += 66;
    double c = blocks3D[index];
    index += 66;
    double d = blocks3D[index]; 
    index += 66;
    double e = blocks3D[index];
    index += 66;
    double f = blocks3D[index];
    index += 66;
    double g = blocks3D[index];
    index += 66;
    double h = blocks3D[index];
    index += 66;
    double i = blocks3D[index];
    index += 66;
    double j = blocks3D[index];
    index += 66;
    double k = blocks3D[index];
    index += 66;
    double l = blocks3D[index];
    index += 66;
    double m = blocks3D[index];
    index += 66;
    double n = blocks3D[index];
    index += 66;
    double o = blocks3D[index];
    index += 66;
    double p = blocks3D[index];

    //if((blockIdx.y * size) + blockIdx.x == 1000) printf("\n\n%f, %f, %f, %f, %f, %f,%f, %f\n%f, %f, %f, %f, %f, %f,%f, %f",                            a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p);

    index = block3DIndex;
    blocks3D[index] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p) * MUL;
    index += 66;
    blocks3D[index] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p) * MUL;
    if(maxBlock > 2)
    {
        index += 66;
        blocks3D[index] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p) * MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p) * MUL;
    }
    if(maxBlock > 4)
    {
        index += 66;
        blocks3D[index] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p) * MUL;
        index += 66;
        blocks3D[index] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p) * MUL;
        index += 66;
        blocks3D[index] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p) * MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p) * MUL;
    }
    if(maxBlock > 8)
    {
        index += 66;
        blocks3D[index] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p) * MUL;
        index += 66;
        blocks3D[index] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p) * MUL;
        index += 66;
        blocks3D[index] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p) * MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p) * MUL;
        index += 66;
        blocks3D[index] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p) * MUL;
        index += 66;
        blocks3D[index] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p) * MUL;
        index += 66;
        blocks3D[index] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p) * MUL;
        index += 66;
        blocks3D[index] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p) * MUL;
    }
}

__device__ void Hadamar8(double* inputs, double MUL)
{
    double a = inputs[0];
    double b = inputs[1];
    double c = inputs[2];
    double d = inputs[3];
    double e = inputs[4];  
    double f = inputs[5];
    double g = inputs[6];
    double h = inputs[7];
    
    inputs[0] = (a+b+c+d+e+f+g+h)*MUL;
    inputs[1] = (a-b+c-d+e-f+g-h)*MUL;
    inputs[2] = (a+b-c-d+e+f-g-h)*MUL;
    inputs[3] = (a-b-c+d+e-f-g+h)*MUL;
    inputs[4] = (a+b+c+d-e-f-g-h)*MUL;
    inputs[5] = (a-b+c-d-e+f-g+h)*MUL;
    inputs[6] = (a+b-c-d-e-f+g+h)*MUL;
    inputs[7] = (a-b-c+d-e+f+g-h)*MUL;
}

__global__
void inverseTransform2D_row(double* blocks3D, int size, double MUL)
{
    int blockIndex = (((blockIdx.y * size) + blockIdx.x) *  2112) + (threadIdx.y * 66) + (threadIdx.x << 3);
    double inputs[8];
    inputs[0] = blocks3D[blockIndex];
    inputs[1] = blocks3D[blockIndex+1];
    inputs[2] = blocks3D[blockIndex+2];
    inputs[3] = blocks3D[blockIndex+3];
    inputs[4] = blocks3D[blockIndex+4];
    inputs[5] = blocks3D[blockIndex+5];
    inputs[6] = blocks3D[blockIndex+6];
    inputs[7] = blocks3D[blockIndex+7];
    Hadamar8(inputs, MUL);
    blocks3D[blockIndex] = inputs[0];
    blocks3D[blockIndex+1] = inputs[1];
    blocks3D[blockIndex+2] = inputs[2];
    blocks3D[blockIndex+3] = inputs[3];
    blocks3D[blockIndex+4] = inputs[4];
    blocks3D[blockIndex+5] = inputs[5];
    blocks3D[blockIndex+6] = inputs[6];
    blocks3D[blockIndex+7] = inputs[7];
}

__global__
void inverseTransform2D_col(double* blocks3D, int size, double MUL)
{
    int blockIndex = (((blockIdx.y * size) + blockIdx.x) * 2112) + (threadIdx.y * 66) + threadIdx.x;
    double inputs[8];
    int index = blockIndex;
    inputs[0] = blocks3D[index];
    index += 8;
    inputs[1] = blocks3D[index];
    index += 8;
    inputs[2] = blocks3D[index];
    index += 8;
    inputs[3] = blocks3D[index];
    index += 8;
    inputs[4] = blocks3D[index];
    index += 8;
    inputs[5] = blocks3D[index];
    index += 8;
    inputs[6] = blocks3D[index];
    index += 8;
    inputs[7] = blocks3D[index];
    Hadamar8(inputs, MUL);
    index = blockIndex;
    blocks3D[index] = inputs[0];
    index += 8;
    blocks3D[index] = inputs[1];
    index += 8;
    blocks3D[index] = inputs[2];
    index += 8;
    blocks3D[index] = inputs[3];
    index += 8;
    blocks3D[index] = inputs[4];
    index += 8;
    blocks3D[index] = inputs[5];
    index += 8;
    blocks3D[index] = inputs[6];
    index += 8;
    blocks3D[index] = inputs[7];
}

__global__
void ApplyCoefBasicEstimate(double* blocks3D, int size, int* nbSimilarBlocks)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    int blockPixelIndex = (block * 2112) + (blockIdx.z * 66) + (threadIdx.y << 3) + threadIdx.x;
    float coef = 1.0f / nbSimilarBlocks[block];
    blocks3D[blockPixelIndex] = blocks3D[blockPixelIndex] * coef;  
    //if(block == 2000) printf("\nblock = %d, num = %f", block, nbSimilarBlocks[block]);
}

__global__
void invTransform2D(double* blocks3D, int size)
{
    
    double iDct[64];
    int blockIndex = (((blockIdx.y * size) + blockIdx.x) * 2112) + (threadIdx.x * 66);
    //for(int i=0;i < 64;i++) blocks3D[blockIndex + i] = blocks3D[blockIndex + i] * coef_norm_inv[i];
    for(int y=0; y< 8; ++y)
    {
        for(int x =0; x< 8; ++x)
        {
            int pixelIndex = (y << 3) + x;
            double sum = 0;
            for(int i=0; i<8; ++i)
            {
                sum += DCTv8matrix[(y<<3) + i] * blocks3D[blockIndex + x + (i << 3)];
            }
            iDct[pixelIndex] = sum;
        }
    }
    for(int y=0; y< 8; ++y)
    {
        for(int x =0; x< 8; ++x)
        {
            int pixelIndex = (y << 3) + x;
            double sum = 0;
            for(int i=0; i<8; ++i)
            {
                sum += iDct[(y<<3) + i] * DCTv8matrixT[x + (i<<3)];
            }
            blocks3D[blockIndex + pixelIndex] = sum;
        }
    }
}

__global__
void ShowPosition(int block, double* blocks3D)
{
    printf("\n\n");    
    for(int i=0;i<16;i++)
        printf("\nblock = %d, x = %f, y = %f", block, blocks3D[block * 2112 + i * 66 + 64], blocks3D[block * 2112 + i * 66 + 65]);
}

void BM3D::BM3D_ShowPosition(int block)
{
    dim3 numBlocks(1);
    dim3 numThreads(1);
    ShowPosition<<<numBlocks,numThreads>>>(block, BM3D::context.blocks3D);
    cudaDeviceSynchronize();
}

void BM3D::BM3D_Inverse3D(bool final)
{    
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(8, 8);
        if(final)
        {
            HadamarTransform32<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks, 0);
        }
        else
        {
            HadamarTransform16<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks, 1);
        }        
        cudaDeviceSynchronize();   
    }
    BM3D_ShowBlock(1000);
    //double MUL = 1.0f/sqrt(8);
    /*if(!final)
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, 16);
        dim3 numThreads(8, 8);
        ApplyCoefBasicEstimate<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks); 
        cudaDeviceSynchronize();   
    }*/
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads((final) ? 32 : 16);
        invTransform2D<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern); 
        cudaDeviceSynchronize();   
    }
    BM3D_ShowBlock(1000);
    /*
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(8, (final) ? 32 : 16);
        inverseTransform2D_col<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, MUL); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(8, (final) ? 32 : 16);
        inverseTransform2D_row<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, MUL); 
        cudaDeviceSynchronize();   
    }*/
}

__global__
void HardThresholdFilter(double* blocks3D, double threshold, int size, int* nbSimilarBlocks)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    int blockPixelIndex = (block * 2112) + (blockIdx.z * 66) + (threadIdx.y << 3) + threadIdx.x;
    float coef_norm = sqrtf((float)nbSimilarBlocks[block]);
    if(fabs(blocks3D[blockPixelIndex]) <= (threshold * coef_norm)) blocks3D[blockPixelIndex] = 0; 
}

__global__
void CalculateNP(double* blocks3D, int* npArray, int size)
{
    int block = ((blockIdx.y * size) + blockIdx.x);
    int blockIndex = (block * 2112) + (blockIdx.z * 66) + (threadIdx.y << 3) + threadIdx.x;
    if(fabs(blocks3D[blockIndex]) > 0) atomicAdd(&npArray[block], 1);  
}


__global__
void CalculateWP(float* wpArray, int* npArray, int size, int sigma)
{
    int block = ((blockIdx.y * size) + blockIdx.x);
    wpArray[block] = (npArray[block] > 0) ? (1.0 / (sigma * sigma * npArray[block])) : 1.0;
}

void BM3D::BM3D_HardThresholdFilter()
{
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, 16);
        dim3 numThreads(8, 8);
        HardThresholdFilter<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.hardThreshold, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, 16);
        dim3 numThreads(8, 8);
        CalculateNP<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.npArray, BM3D::context.widthBlocksIntern); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(1);
        CalculateWP<<<numBlocks,numThreads>>>(BM3D::context.wpArray, BM3D::context.npArray, BM3D::context.widthBlocksIntern, BM3D::context.sigma); 
        cudaDeviceSynchronize();   
    }
}

__global__
void BM_CreateBlocks(int* windowMap, int size, int windowSize, int width, float* image, double* blocks, int blockSize, int offset)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    int blockIndex = (block * blockSize) + (threadIdx.x << 6);
    double blockArray[64];

    int windowMapIndex = (block * windowSize) + (threadIdx.x * 4);
    int refX = windowMap[windowMapIndex];
    int refY = windowMap[windowMapIndex+1];       

    for(int y=0; y< 8; y++)
    {
        for(int x=0; x< 8; x++)
        {
            blockArray[(y << 3) + x] = image[((refY + y) * width) + refX + x];
        }
    }

    /*if(block == 1000 && threadIdx.x == 0)
    {
        printf("\nBlock 1000: %d, %d, x= %d, y =%d\n", windowMapIndex, windowSize, (refX-offset), (refY-offset));
        for(int i=0; i<8; i++)
        {
            for(int j=0;j<8;j++)
            {
                printf("%f, ", blockArray[(i << 3) + j]);
            }
            printf("\n");
        }
    }*/

    double dct[64];
    for(int y=0; y< 8; ++y)
    {
        for(int x =0; x< 8; ++x)
        {
            int pixelIndex = (y << 3) + x;
            double sum = 0;
            for(int i=0; i<8; ++i)
            {
                sum += (DCTv8matrixT[(y<<3) + i] * blockArray[x + (i << 3)]);
            }
            dct[pixelIndex] = sum;
        }
    }
    for(int y=0; y< 8; ++y)
    {
        for(int x =0; x< 8; ++x)
        {
            int pixelIndex = (y << 3) + x;
            double sum = 0;
            for(int i=0; i<8; ++i)
            {
                sum += (dct[(y<<3) + i] * DCTv8matrix[x + (i<<3)]);
            }
            blocks[blockIndex + pixelIndex] = sum; 
        }
    }

    /*if(block == 1000 && threadIdx.x == 0)
    {
        printf("\nBlock 1000:\n");
        for(int i=0; i<8; i++)
        {
            for(int j=0;j<8;j++)
            {
                printf("%f, ", blocks[blockIndex + (i<<3) + j]);
            }
            printf("\n");
        }
    }*/
    
}

__global__
void BM_CalculateDistance(int* windowMap, int size, int windowSize, int width, float* image, double* blocks, int blockSize)
{
    __shared__ float distances[8];
    for(int i=0; i<16; ++i) distances[i] = 0;
    __syncthreads();

    int block = (blockIdx.y * size) + blockIdx.x;
    int blockIndex = (block * blockSize) + 64 + (blockIdx.z << 6) + (threadIdx.x << 3);
    int blockRefIndex = block * blockSize + (threadIdx.x << 3);

    int windowMapRefIndex = block * windowSize;
    int refX = windowMap[windowMapRefIndex];
    int refY = windowMap[windowMapRefIndex+1] + threadIdx.x;    

    int windowMapBlockIndex = windowMapRefIndex + 4 + blockIdx.z * 4;
    int blockX = windowMap[windowMapBlockIndex];
    int blockY = windowMap[windowMapBlockIndex+1] + threadIdx.x;

    double distance = 0;
    for(int i=0; i<8; ++i)
    {
        double valueRef = blocks[blockRefIndex + i];
        //float valueRef = image[refY * width + refX + i];
        double valueBlock = blocks[blockIndex + i];
        //float valueBlock = image[blockY * width + blockX + i];
        distance += (valueRef - valueBlock) * (valueRef - valueBlock);
    }
    distances[threadIdx.x] = distance;
    __syncthreads();

    if(threadIdx.x == 0)
    {
        windowMap[windowMapBlockIndex+2] = int(distances[0] + distances[1] + distances[2] + distances[3] + distances[4] + distances[5] + distances[6] + distances[7]);
        //if(block == 1000) printf("\nblock %d, distance = %d", blockIdx.z, windowMap[windowMapBlockIndex+2]);
    }
}

__global__
void BM_DistanceThreshold(int* windowMap, int size, int windowSize, int limit)
{
    int windowMapIndex = (((blockIdx.y * size) + blockIdx.x) * windowSize) + 4 + (threadIdx.x * 4);
    //if((blockIdx.y * size) + blockIdx.x == 1000) printf("\nthreashold: %d, %d", limit, windowMap[windowMapIndex+2]);
    if(windowMap[windowMapIndex+2] >= limit) windowMap[windowMapIndex+2] = -1;
    //if((blockIdx.y * size) + blockIdx.x == 1000) printf("\nafter threashold: %d, %d", limit, windowMap[windowMapIndex+2]);
}
    

__global__
void BM_Sort(int* windowMap, int size, int windowSize, int nbBlocksPerWindow, int maxBlock, int* nbSimilarBlocks)
{
    int windowMapIndex = (((blockIdx.y * size) + blockIdx.x) * windowSize) + 4;
    int index = 1;   
    int indexes[32];
    for(int n = 0; n< maxBlock; n++)
    {
        int currentDistance = 9999999;
        int foundIndex = -1;
        for(int i=0; i< nbBlocksPerWindow; i++)
        {                
            if(windowMap[windowMapIndex + i * 4 + 2] > -1  && windowMap[windowMapIndex + i * 4 + 2] < currentDistance) { foundIndex = i; currentDistance = windowMap[windowMapIndex + i * 4 + 2]; }
        }   
        if(foundIndex > -1)
        {
            windowMap[windowMapIndex + foundIndex * 4 + 2] = -1 * windowMap[windowMapIndex + foundIndex * 4 + 2];
            windowMap[windowMapIndex + foundIndex * 4 + 3] = index;
            indexes[index-1] = foundIndex;
            if((blockIdx.y * size) + blockIdx.x == 1000) printf("\nset index %d to %d", index, foundIndex);
            index++;            
        }
    }
    
    unsigned r = 1;
    while (r * 2 <= index)
        r *= 2;

    int baseIndex = (((blockIdx.y * size) + blockIdx.x) * windowSize);
    if(r == 1) //avoid problem
    {
        windowMap[windowMapIndex ] = windowMap[baseIndex];
        windowMap[windowMapIndex + 1 ] = windowMap[baseIndex + 1];
        windowMap[windowMapIndex + 2 ] = windowMap[baseIndex + 2];
        windowMap[windowMapIndex + 3 ] = r;
        r++;
    }

    for(int i= r; i < index-1; i++)
    {
        windowMap[windowMapIndex + indexes[i] * 4 + 3] = -1;
        if((blockIdx.y * size) + blockIdx.x == 1000) printf("\nreset %d", i);
    }

    nbSimilarBlocks[(blockIdx.y * size) + blockIdx.x] = r;


    
    //printf("\nblock = %d, nb = %d", (blockIdx.y * size) + blockIdx.x, nbSimilarBlocks[(blockIdx.y * size) + blockIdx.x]);
}


__global__
void ShowDistance(int block, int* windowMap, int windowSize, int maxBlock, int blockPerWindow)
{
    int windowMapIndex = block * windowSize;
    for(int i=0; i< maxBlock; i++)
    {
        for(int k=0; k< blockPerWindow; k++)
        {
            if(windowMap[windowMapIndex + k * 4 + 3] == i)
            {
                printf("\nBlock %d, distance = %d, index = %d, x= %d, y= %d", i, windowMap[windowMapIndex + k * 4 + 2], windowMap[windowMapIndex + k * 4 + 3], windowMap[windowMapIndex + i * 4], windowMap[windowMapIndex + k * 4 + 1] );
                break;
            }
        }
    }
}

void BM3D::BM3D_ShowDistance(int block)
{
    dim3 val(1);
    ShowDistance<<<val,val>>>(block, BM3D::context.windowMap, BM3D::context.windowSize, 32, BM3D::context.nbBlocksPerWindow);
    cudaDeviceSynchronize();
}

void BM3D::BM3D_BlockMatching(bool final)
{
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(BM3D::context.nbBlocksPerWindow + 1);
        BM_CreateBlocks<<<numBlocks,numThreads>>>(BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, BM3D::context.img_width, BM3D::context.basicImage, BM3D::context.blocks, BM3D::context.blockSize, BM3D::context.offset ); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, BM3D::context.nbBlocksPerWindow);
        dim3 numThreads(8);
        BM_CalculateDistance<<<numBlocks,numThreads>>>(BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, BM3D::context.img_width, BM3D::context.basicImage, BM3D::context.blocks, BM3D::context.blockSize ); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(BM3D::context.nbBlocksPerWindow);
        BM_DistanceThreshold<<<numBlocks,numThreads>>>(BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, (final) ? BM3D::context.wienLimit : BM3D::context.hardLimit); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(1);
        //dim3 numThreads(BM3D::context.nbBlocksPerWindow);
        BM_Sort<<<numBlocks,numThreads>>>(BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, BM3D::context.nbBlocksPerWindow, (final) ? 32: 16, BM3D::context.nbSimilarBlocks); 
        cudaDeviceSynchronize();   
    }
}

__global__
void Create3DBlock(float* image, double* blocks3D, int* windowMap, int size, int windowSize, int width, int* similarBlock)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    int windowMapIndex = (block * windowSize) + (blockIdx.z * 4);
    if(windowMap[windowMapIndex+3] < similarBlock[block] && windowMap[windowMapIndex+3] > -1)
    { 
        int block3DIndex = (((blockIdx.y * size) + blockIdx.x) * 2112) + (windowMap[windowMapIndex+3] * 66) +  (threadIdx.y << 3) + threadIdx.x;
        int xImg = windowMap[windowMapIndex] + threadIdx.x;
        int yImg = windowMap[windowMapIndex+1] + threadIdx.y;
        blocks3D[block3DIndex] = image[yImg * width + xImg];       
    }
}

__global__
void AddBlockPosition(double* blocks3D, int* windowMap, int size, int windowSize, int* similarBlock)
{
    int block = ((blockIdx.y * size) + blockIdx.x);
    int windowMapIndex = block * windowSize + (threadIdx.x * 4);
    if(windowMap[windowMapIndex+3] < similarBlock[block] && windowMap[windowMapIndex+3] > -1)
    { 
        int block3DIndex = (((blockIdx.y * size) + blockIdx.x) * 2112) + (windowMap[windowMapIndex+3] * 66);
        blocks3D[block3DIndex+64] = windowMap[windowMapIndex];
        blocks3D[block3DIndex+65] = windowMap[windowMapIndex+1];
    }
}

__global__
void Transform2D_row(double* blocks3D, int size, double MUL)
{
    int block3DIndex = (((blockIdx.y * size) + blockIdx.x) * 2112) + (threadIdx.x * 66) + (threadIdx.y << 3);
    double inputs[8];
    inputs[0] = blocks3D[block3DIndex];
    inputs[1] = blocks3D[block3DIndex+1];
    inputs[2] = blocks3D[block3DIndex+2];
    inputs[3] = blocks3D[block3DIndex+3];
    inputs[4] = blocks3D[block3DIndex+4];
    inputs[5] = blocks3D[block3DIndex+5];
    inputs[6] = blocks3D[block3DIndex+6];
    inputs[7] = blocks3D[block3DIndex+7];
    Hadamar8(inputs, MUL);
    blocks3D[block3DIndex] = inputs[0];
    blocks3D[block3DIndex+1] = inputs[1];
    blocks3D[block3DIndex+2] = inputs[2];
    blocks3D[block3DIndex+3] = inputs[3];
    blocks3D[block3DIndex+4] = inputs[4];
    blocks3D[block3DIndex+5] = inputs[5];
    blocks3D[block3DIndex+6] = inputs[6];
    blocks3D[block3DIndex+7] = inputs[7];
}

__global__
void Transform2D_col(double* blocks3D, int size, double MUL)
{
    int block3DIndex = (((blockIdx.y * size) + blockIdx.x) * 2112) + (threadIdx.x * 66) + threadIdx.y;
    double inputs[8];
    int index = block3DIndex;
    inputs[0] = blocks3D[index];
    index += 8;
    inputs[1] = blocks3D[index];
    index += 8;
    inputs[2] = blocks3D[index];
    index += 8;
    inputs[3] = blocks3D[index];
    index += 8;
    inputs[4] = blocks3D[index];
    index += 8;
    inputs[5] = blocks3D[index];
    index += 8;
    inputs[6] = blocks3D[index];
    index += 8;
    inputs[7] = blocks3D[index];
    Hadamar8(inputs, MUL);
    index = block3DIndex;
    blocks3D[index] = inputs[0];
    index += 8;
    blocks3D[index] = inputs[1];
    index += 8;
    blocks3D[index] = inputs[2];
    index += 8;
    blocks3D[index] = inputs[3];
    index += 8;
    blocks3D[index] = inputs[4];
    index += 8;
    blocks3D[index] = inputs[5];
    index += 8;
    blocks3D[index] = inputs[6];
    index += 8;
    blocks3D[index] = inputs[7];
}

__global__
void ShowBlock(int block, double* blocks3D)
{
    printf("\n\nBlock %d, %d\n", block, block * 2112);
    for(int i=0; i<64; i++) 
    {
        if(i%8 ==0) printf("\n");
        printf("%f, ", blocks3D[block * 2112 + i]);
    }
}

__global__
void Transform2D(double* blocks3D, int size, int* similarBlocks)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    if(threadIdx.x < similarBlocks[block])
    {
        int blockIndex = (block * 2112) + (threadIdx.x * 66);
        double dct[64];
        for(int y=0; y< 8; ++y)
        {
            for(int x =0; x< 8; ++x)
            {
                int pixelIndex = (y << 3) + x;
                double sum = 0;
                for(int i=0; i<8; ++i)
                {
                    sum += (DCTv8matrixT[(y<<3) + i] * blocks3D[blockIndex + x + (i << 3)]);
                }
                dct[pixelIndex] = sum;
            }
        }
        for(int y=0; y< 8; ++y)
        {
            for(int x =0; x< 8; ++x)
            {
                int pixelIndex = (y << 3) + x;
                double sum = 0;
                for(int i=0; i<8; ++i)
                {
                    sum += (dct[(y<<3) + i] * DCTv8matrix[x + (i<<3)]);
                }
                blocks3D[blockIndex + pixelIndex] = sum; 
            }
        }
    }
}

void BM3D::BM3D_ShowBlock(int block)
{
    dim3 val(1);
    ShowBlock<<<val,val>>>(block, BM3D::context.blocks3D);
}

void BM3D::BM3D_Create3DBlocks(bool final)
{
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, BM3D::context.nbBlocksPerWindow + 1);
        dim3 numThreads(8, 8);
        Create3DBlock<<<numBlocks,numThreads>>>(BM3D::context.basicImage, BM3D::context.blocks3D, BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, BM3D::context.img_width, BM3D::context.nbSimilarBlocks);
        cudaDeviceSynchronize(); 
    }
    BM3D_ShowBlock(1000);
    if(final)
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, BM3D::context.nbBlocksPerWindow + 1);
        dim3 numThreads(8, 8);
        Create3DBlock<<<numBlocks,numThreads>>>(BM3D::context.noisyImage, BM3D::context.blocks3DOrig, BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, BM3D::context.img_width, BM3D::context.nbSimilarBlocks);
        cudaDeviceSynchronize(); 
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(BM3D::context.nbBlocksPerWindow + 1);
        AddBlockPosition<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, BM3D::context.nbSimilarBlocks);
        cudaDeviceSynchronize(); 
    }
    BM3D_ShowBlock(1000);
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads((final) ? 32 : 16);
        Transform2D<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks); 
        cudaDeviceSynchronize();   
    }
    /*float MUL = 1.0f / sqrt(8);
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads((final) ? 32 : 16, 8);
        Transform2D_row<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, MUL);
        cudaDeviceSynchronize(); 
        Transform2D_col<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, MUL);
        cudaDeviceSynchronize();
    }*/
    if(final)
    {
        /*dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads((final) ? 32 : 16, 8);
        Transform2D_row<<<numBlocks,numThreads>>>(BM3D::context.blocks3DOrig, BM3D::context.widthBlocksIntern, MUL);
        cudaDeviceSynchronize(); 
        Transform2D_col<<<numBlocks,numThreads>>>(BM3D::context.blocks3DOrig, BM3D::context.widthBlocksIntern, MUL);
        cudaDeviceSynchronize();*/

        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads((final) ? 32 : 16);
        Transform2D<<<numBlocks,numThreads>>>(BM3D::context.blocks3DOrig, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks); 
        cudaDeviceSynchronize();
    }
    BM3D_ShowBlock(1000);
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(8, 8);
        if(final)
        {
            //float MUL = 1.0f;///sqrt(32);
            HadamarTransform32<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks, 0);
            HadamarTransform32<<<numBlocks,numThreads>>>(BM3D::context.blocks3DOrig, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks, 0);
        }
        else
        {
            //float MUL = 1.0f;
            HadamarTransform16<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks, 0);
        }
        cudaDeviceSynchronize();    
    }
    BM3D_ShowBlock(1000);
}


/*** 
*Create windows
***/

__global__
void ShiftImage(float* originalImage, float* basicImage, int widthOrig, int width, int offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int originalImageIndex = (y * widthOrig) + x;
    int basicImageIndex = ((y+offset) * width) + (x+offset);
    basicImage[basicImageIndex] = originalImage[originalImageIndex];
}


__global__
void InverseShiftImage(float* originalImage, float* basicImage, int widthOrig, int width, int offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int originalImageIndex = (y * widthOrig) + x;
    int basicImageIndex = ((y+offset) * width) + (x+offset);
    originalImage[originalImageIndex] = basicImage[basicImageIndex];
}

void BM3D::BM3D_InverseShift()
{
   {
        dim3 numBlocks(BM3D::context.img_widthOrig/8, BM3D::context.img_heightOrig/8);
        dim3 numThreads(8,8);
        InverseShiftImage<<<numBlocks,numThreads>>>(BM3D::context.deviceImage, BM3D::context.basicImage, BM3D::context.img_widthOrig, BM3D::context.img_width, BM3D::context.offset); 
        cudaDeviceSynchronize();   
   } 
}

__global__
void CreateWindowBlocks(int* windowMap, int widthBlockIntern, int p, int windowSize, int offset, int halfWindowSize, int widthBlocksWindow)
{
    int windowMapIndex = ((blockIdx.y * widthBlockIntern) + blockIdx.x) * windowSize;
    int refX = windowMap[windowMapIndex];
    int refY = windowMap[windowMapIndex+1];
    int blockX = refX - halfWindowSize + threadIdx.x * p;
    int blockY = refY - halfWindowSize + threadIdx.y * p;
    int windowBlockIndex = windowMapIndex + 4 + (((threadIdx.y * widthBlocksWindow) + threadIdx.x) * 4);
    windowMap[windowBlockIndex] = blockX;
    windowMap[windowBlockIndex+1] = blockY;
}

__global__
void CreateRefBlock(int* windowMap, int size, int windowSize, int p, int offset)
{
    int refX = blockIdx.x * p + offset;
    int refY = blockIdx.y * p + offset;
    int windowMapIndex = ((blockIdx.y * size) + blockIdx.x) * windowSize;
    windowMap[windowMapIndex] = refX;
    windowMap[windowMapIndex+1] = refY;
    windowMap[windowMapIndex+2] = 0;
    windowMap[windowMapIndex+3] = 0;
}

__global__
void CopyImage(float* basicImage, float* noisyImage, int width)
{
    int pixelIndex = blockIdx.y * width + blockIdx.x;
    noisyImage[pixelIndex] = basicImage[pixelIndex];
}

void BM3D::BM3D_CreateWindow(bool final)
{
    {
        dim3 numBlocks(BM3D::context.img_widthOrig/8, BM3D::context.img_heightOrig/8);
        dim3 numThreads(8,8);
        ShiftImage<<<numBlocks,numThreads>>>(BM3D::context.deviceImage, BM3D::context.basicImage, BM3D::context.img_widthOrig, BM3D::context.img_width, BM3D::context.offset); 
        cudaDeviceSynchronize();   
    }
    if(!final)
    {
        dim3 numBlocks(BM3D::context.img_width, BM3D::context.img_height);
        dim3 numThreads(1);
        CopyImage<<<numBlocks,numThreads>>>(BM3D::context.basicImage, BM3D::context.noisyImage, BM3D::context.img_width); 
        cudaDeviceSynchronize();
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(1);
        CreateRefBlock<<<numBlocks,numThreads>>>(BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, BM3D::context.pHard, BM3D::context.offset); 
        cudaDeviceSynchronize();   
    } 
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(BM3D::context.widthBlocksWindow, BM3D::context.widthBlocksWindow);
        CreateWindowBlocks<<<numBlocks,numThreads>>>(BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.pHard, BM3D::context.windowSize, BM3D::context.offset, BM3D::context.halfWindowSize, BM3D::context.widthBlocksWindow); 
        cudaDeviceSynchronize();   
    }  
}	
