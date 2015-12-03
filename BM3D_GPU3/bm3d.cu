#include <stdio.h>

#include "bm3d.h"
#include "utilities.h"
#include "timeutil.h"

#include <string>

BM3D::BM3D_Context BM3D::context;

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

void BM3D::BM3D_Initialize(BM3D::SourceImage img, BM3D::SourceImage imgOrig, int width, int height, int pHard, int hardLimit, int wienLimit, double hardThreshold, int sigma, bool debug)
{
    printf("\n--> Execution on Tesla K40c");
    if(cudaSuccess != cudaSetDevice(0)) printf("\n\tNo device 0 available");

    if(debug)
    {
        int sz = 1048576 * 1024;
        cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);
    }

    printf("\nBM3D context initialization");
    BM3D::context.debugPixel = 1792;
    BM3D::context.debugBlock = 15508;

    BM3D::context.hardLimit = hardLimit;
    BM3D::context.wienLimit = wienLimit;
    BM3D::context.hardThreshold = hardThreshold;
    BM3D::context.sigma = sigma;

    int w2 = width - (width % pHard) + (10 * pHard);
    int h2 = height - (height % pHard) + (10 * pHard);
    
    BM3D::context.nbBlocksIntern = (width / pHard) * (height /pHard);
    BM3D::context.nbBlocks = (w2 / pHard) * (h2 /pHard);
    BM3D::context.widthBlocks = (w2 / pHard);
    BM3D::context.widthBlocksIntern = (width / pHard);

    w2 += 8;  //nHard = 8
    h2 += 8;  //nHard = 8

    Timer::start(); 
    float integralMap[w2 * h2];
    memset(integralMap, 0, w2 * h2 * sizeof(float));
    int offset = 5 * pHard;
    for(int y =0;y < width; ++y)
    {
        for(int x=0; x< height; ++x)
        {
            integralMap[(y+offset) * width + (x+offset)] = img[y * width + x] + integralMap[(y+offset-1) * width + (x+offset)] + integralMap[(y+offset) * width + (x+offset-1)];
        }
    }
    Timer::add("Initialize integral image map");
    
    BM3D::context.img_widthOrig = width; 
    BM3D::context.img_heightOrig= height;
    BM3D::context.img_width = w2; 
    BM3D::context.img_height= h2;
    BM3D::context.pHard = pHard;
    BM3D::context.sourceImage = img;
    BM3D::context.origImage = imgOrig;

    printf("\n\tNumber of blocks          = %d", BM3D::context.nbBlocks);
    printf("\n\tNumber of blocks (intern) = %d", BM3D::context.nbBlocksIntern);
    printf("\n\tWidth blocks (intern)     = %d", BM3D::context.widthBlocksIntern);
    printf("\n\tWidth blocks              = %d", BM3D::context.widthBlocks);
    printf("\n\tWidth                     = %d", BM3D::context.img_width);
    printf("\n\tHeight                    = %d", BM3D::context.img_height);
    printf("\n\tDevice image              = %f Mb", (width * height * sizeof(float)/1024.00 / 1024.00));  
    printf("\n\tBasic image               = %f Mb", (w2 * h2 * sizeof(float)/1024.00 / 1024.00));
    printf("\n\tIntegral image            = %f Mb", (w2 * h2 * sizeof(float)/1024.00 / 1024.00));
    printf("\n\tBlocks array              = %f Mb", (BM3D::context.nbBlocks * 66 * sizeof(double)/1024.00 / 1024.00));  
    printf("\n\tBlocks array (orig)       = %f Mb", (BM3D::context.nbBlocks * 66 * sizeof(double)/1024.00 / 1024.00));  
    printf("\n\tBlocks map                = %f Mb", (BM3D::context.nbBlocks * 100 * 10 * sizeof(int)/1024.00 / 1024.00));  
    printf("\n\tBM vectors                = %f Mb", (BM3D::context.nbBlocksIntern * 32 * sizeof(int)/1024.00 / 1024.00)); 
    printf("\n\tBlocks 3D                 = %f Mb", (BM3D::context.nbBlocksIntern * 32 * 64 * sizeof(double)/1024.00 / 1024.00));  
    printf("\n\tBlocks 3D (orig)          = %f Mb", (BM3D::context.nbBlocksIntern * 32 * 64 * sizeof(double)/1024.00 / 1024.00));
    printf("\n\tNP array                  = %f Mb", (BM3D::context.nbBlocksIntern  * sizeof(int)/1024.00 / 1024.00));
    printf("\n\tWP array                  = %f Mb", (BM3D::context.nbBlocksIntern  * sizeof(double)/1024.00 / 1024.00));
    printf("\n\tEstimates array           = %f Mb", (w2 * h2 * 2 * sizeof(float)/1024.00 / 1024.00));
    printf("\n\tSimilar blocks array      = %f Mb", (BM3D::context.nbBlocksIntern  * sizeof(float)/1024.00 / 1024.00));

    gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, width * height * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.deviceImage, &img[0], width * height * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&BM3D::context.basicImage, w2 * h2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.integralImageMap, w2 * h2 * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.integralImageMap, &integralMap[0], w2 * h2 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(BM3D::context.basicImage, 0, w2 * h2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.estimates, w2 * h2 * 2 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.estimates, 0, w2 * h2 * 2 * sizeof(float)));
    
    //! First quarter of the matrix
    float kaiserWindow[64];
    int kHW = 8;
    kaiserWindow[0 + kHW * 0] = 0.1924f; kaiserWindow[0 + kHW * 1] = 0.2989f; kaiserWindow[0 + kHW * 2] = 0.3846f; kaiserWindow[0 + kHW * 3] = 0.4325f;
    kaiserWindow[1 + kHW * 0] = 0.2989f; kaiserWindow[1 + kHW * 1] = 0.4642f; kaiserWindow[1 + kHW * 2] = 0.5974f; kaiserWindow[1 + kHW * 3] = 0.6717f;
    kaiserWindow[2 + kHW * 0] = 0.3846f; kaiserWindow[2 + kHW * 1] = 0.5974f; kaiserWindow[2 + kHW * 2] = 0.7688f; kaiserWindow[2 + kHW * 3] = 0.8644f;
    kaiserWindow[3 + kHW * 0] = 0.4325f; kaiserWindow[3 + kHW * 1] = 0.6717f; kaiserWindow[3 + kHW * 2] = 0.8644f; kaiserWindow[3 + kHW * 3] = 0.9718f;

    //! Completing the rest of the matrix by symmetry
    for(unsigned i = 0; i < kHW / 2; i++)
        for (unsigned j = kHW / 2; j < kHW; j++)
            kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)];

    for (unsigned i = kHW / 2; i < kHW; i++)
        for (unsigned j = 0; j < kHW; j++)
            kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j];

    gpuErrchk(cudaMalloc(&BM3D::context.kaiserWindowCoef, 64 * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.kaiserWindowCoef, kaiserWindow, 64 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&BM3D::context.blockMap, BM3D::context.nbBlocksIntern * 100 * 10 * sizeof(int)));
    gpuErrchk(cudaMemset(BM3D::context.blockMap, 0, BM3D::context.nbBlocksIntern * 100 * 10 * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.blocks, BM3D::context.nbBlocks * 66 * sizeof(double)));
    gpuErrchk(cudaMemset(BM3D::context.blocks, 0, BM3D::context.nbBlocks * 66 * sizeof(double)));
    gpuErrchk(cudaMalloc(&BM3D::context.blocksOrig, BM3D::context.nbBlocks * 66 * sizeof(double)));
    gpuErrchk(cudaMalloc(&BM3D::context.bmVectors, BM3D::context.nbBlocksIntern * 32 * sizeof(int)));
    gpuErrchk(cudaMemset(BM3D::context.bmVectors, 0, BM3D::context.nbBlocksIntern * 32 * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.blocks3D, BM3D::context.nbBlocksIntern * 32 * 64 * sizeof(double)));
    gpuErrchk(cudaMalloc(&BM3D::context.blocks3DOrig, BM3D::context.nbBlocksIntern * 32 * 64 * sizeof(double)));
    gpuErrchk(cudaMalloc(&BM3D::context.npArray, BM3D::context.nbBlocksIntern  * sizeof(int)));
    gpuErrchk(cudaMemset(BM3D::context.npArray, 0, BM3D::context.nbBlocksIntern  * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.wpArray, BM3D::context.nbBlocksIntern  * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.nbSimilarBlocks, BM3D::context.nbBlocksIntern  * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.nbSimilarBlocks, 0, BM3D::context.nbBlocksIntern  * sizeof(float)));

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

__global__
void InitBlocksArray(double* blocks, int size)
{
    int blockIndex = ((blockIdx.y * size) + blockIdx.x) * 66 + (threadIdx.y << 3) + threadIdx.x;
    blocks[blockIndex] = 0;
}

void BM3D::BM3D_FinalEstimate()
{
    //gpuErrchk(cudaMemset(BM3D::context.wpArray, 0, BM3D::context.nbBlocksIntern  * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.bmVectors, 0, BM3D::context.nbBlocksIntern * 32 * sizeof(int)));
    gpuErrchk(cudaMemset(BM3D::context.blockMap, 0, BM3D::context.nbBlocksIntern * 100 * 10 * sizeof(int))); 
    gpuErrchk(cudaMemset(BM3D::context.basicImage, 0, BM3D::context.img_width * BM3D::context.img_height * sizeof(float)));
    //gpuErrchk(cudaMemset(BM3D::context.blocks, 0, BM3D::context.nbBlocks * 66 * sizeof(double)));

    //BM3D_ShowBlock(0);
    {
        //init blocks array
        dim3 numBlocks(BM3D::context.widthBlocks, BM3D::context.widthBlocks);
        dim3 numThreads(8,8);
        InitBlocksArray<<<numBlocks,numThreads>>>(BM3D::context.blocks, BM3D::context.widthBlocks); 
        cudaDeviceSynchronize();
    }
    //BM3D_ShowBlock(0);

    printf("\n\tFinal estimates (2 step)");
    Timer::start(); 
    BM3D_CreateBlock();
    //BM3D_ShowBlock(0);
    BM3D_2DTransform(true);
    BM3D_BlockMatching(true);
    //BM3D_ShowBlock3D(6515);
    BM3D_WienFilter();
    //BM3D_ShowBlock3D(6515);
    BM3D_Inverse3D(true);
    //BM3D_ShowBlock3D(6515);
    BM3D_Aggregation(true);
    BM3D_InverseShift();
    Timer::add("BM3D-Final estimates");
    BM3D_SaveImage(true);
}

void BM3D::BM3D_BasicEstimate()
{
    printf("\n\tBasic estimates (1 step)");
    Timer::start();
    BM3D_CreateBlock();
    BM3D_2DTransform();
    BM3D_BlockMatching();
    //BM3D_ShowDistance(0);
    //BM3D_ShowDistance(0, true);
    BM3D_HardThresholdFilter();
    //BM3D_ShowBlock3D(6515);
    BM3D_Inverse3D();
    //BM3D_ShowBlock3D(6515);
    BM3D_Aggregation();
    BM3D_InverseShift();
    Timer::add("BM3D-Basic estimates");
    BM3D_SaveImage();
}


__global__
void WienFilter(double* blocks3D, double* blocks3DOrig, int size, float* similarBlocks, int sigma, float* wpArray)
{    
    int block = (blockIdx.y * size) + blockIdx.x;
    float coef = 1.0 / similarBlocks[block];    
    int blockPixelIndex = (block << 11) + (blockIdx.z << 6) + (threadIdx.y << 3) + threadIdx.x;
    float estimateValue = blocks3D[blockPixelIndex];
    float value = estimateValue * estimateValue;// * coef;
    value /= (value + (sigma * sigma));
    blocks3D[blockPixelIndex] = blocks3DOrig[blockPixelIndex] * value;// * coef;
    atomicAdd(&wpArray[block], value);
}

__global__
void CalculateFinalWP(int size, int sigma, float* wpArray)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    wpArray[block] = (wpArray[block] > 0) ? (1.0f / (sigma * sigma * wpArray[block])) : 1.0f;
}

void BM3D::BM3D_WienFilter()
{
    {    
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, 32);
        dim3 numThreads(8, 8);
        WienFilter<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.blocks3DOrig, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks, BM3D::context.sigma, BM3D::context.wpArray); 
        cudaDeviceSynchronize();
    }
    {    
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(1);
        CalculateFinalWP<<<numBlocks,numThreads>>>(BM3D::context.widthBlocksIntern, BM3D::context.sigma, BM3D::context.wpArray); 
        cudaDeviceSynchronize();
    }   
}

__global__
void pre_aggregation(double* blocks3D, float* wpArray, double* blocks, int* bmVectors, int size, float* kaiserCoef, float* estimates, int width)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    int bmVectorIndex = (block << 5) + blockIdx.z;
    int blockIndex = bmVectors[bmVectorIndex];
    if(blockIndex > 0)
    {
        int xImg = (int)blocks[blockIndex + 64];
        int yImg = (int)blocks[blockIndex + 65];
        int xPixel = xImg + threadIdx.x;
        int yPixel = yImg + threadIdx.y;
        int estimateIndex = ((yPixel * width) + xPixel) << 1;
        int kaiserIndex = (threadIdx.y << 3) + threadIdx.x;
        int block3DIndex = (block << 11) + (blockIdx.z << 6) + kaiserIndex;
        atomicAdd(&estimates[estimateIndex], (kaiserCoef[kaiserIndex] * wpArray[block] * blocks3D[block3DIndex]));
        atomicAdd(&estimates[estimateIndex+1], (kaiserCoef[kaiserIndex] * wpArray[block]));
    }
}

__global__
void aggregation(float* estimates, float* basicImage, int size)
{
    int basicImageIndex = (((blockIdx.y * size) + blockIdx.x) << 6) + (threadIdx.y << 3) + threadIdx.x;
    int estimateIndex = (basicImageIndex << 1);
    basicImage[basicImageIndex] = estimates[estimateIndex]/estimates[estimateIndex+1];
}

void BM3D::BM3D_Aggregation(bool final)
{
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, (final) ? 32 : 16);
        dim3 numThreads(8, 8);
        pre_aggregation<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.wpArray, BM3D::context.blocks, BM3D::context.bmVectors, BM3D::context.widthBlocksIntern, BM3D::context.kaiserWindowCoef, BM3D::context.estimates, BM3D::context.img_width); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.img_width/8, BM3D::context.img_height/8);
        dim3 numThreads(8, 8);
        aggregation<<<numBlocks,numThreads>>>(BM3D::context.estimates, BM3D::context.basicImage, BM3D::context.img_height/8); 
        cudaDeviceSynchronize();   
    }
}

__global__
void inverse3D16(double* blocks3D, int size)
{
    int block3DIndex = (((blockIdx.y * size) + blockIdx.x) << 11) + (threadIdx.y << 3) + threadIdx.x;

    double a = blocks3D[block3DIndex];
    double b = blocks3D[block3DIndex+64];
    double c = blocks3D[block3DIndex+128];
    double d = blocks3D[block3DIndex+192];
    double e = blocks3D[block3DIndex+256];
    double f = blocks3D[block3DIndex+320];
    double g = blocks3D[block3DIndex+384];
    double h = blocks3D[block3DIndex+448];
    double i = blocks3D[block3DIndex+512];
    double j = blocks3D[block3DIndex+576];
    double k = blocks3D[block3DIndex+640];
    double l = blocks3D[block3DIndex+704];
    double m = blocks3D[block3DIndex+768];
    double n = blocks3D[block3DIndex+832];
    double o = blocks3D[block3DIndex+896];
    double p = blocks3D[block3DIndex+960];

    blocks3D[block3DIndex] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p) / 4.0;
    blocks3D[block3DIndex+64] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p) / 4.0;
    blocks3D[block3DIndex+128] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p) / 4.0;
    blocks3D[block3DIndex+192] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p) / 4.0;
    blocks3D[block3DIndex+256] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p) / 4.0;
    blocks3D[block3DIndex+320] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p) / 4.0;
    blocks3D[block3DIndex+384] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p) / 4.0;
    blocks3D[block3DIndex+448] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p) / 4.0;
    blocks3D[block3DIndex+512] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p) / 4.0;
    blocks3D[block3DIndex+576] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p) / 4.0;
    blocks3D[block3DIndex+640] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p) / 4.0;
    blocks3D[block3DIndex+704] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p) / 4.0;
    blocks3D[block3DIndex+768] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p) / 4.0;
    blocks3D[block3DIndex+832] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p) / 4.0;
    blocks3D[block3DIndex+896] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p) / 4.0;
    blocks3D[block3DIndex+960] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p) / 4.0;
}

__global__
void inverse3D32(double* blocks3D, int size, float DIVISOR)
{
    int block3DIndex = (((blockIdx.y * size) + blockIdx.x) << 11) + (threadIdx.y << 3) + threadIdx.x;

    double a = blocks3D[block3DIndex];
    double b = blocks3D[block3DIndex+64];
    double c = blocks3D[block3DIndex+128];
    double d = blocks3D[block3DIndex+192];
    double e = blocks3D[block3DIndex+256];
    double f = blocks3D[block3DIndex+320];
    double g = blocks3D[block3DIndex+384];
    double h = blocks3D[block3DIndex+448];
    double i = blocks3D[block3DIndex+512];
    double j = blocks3D[block3DIndex+576];
    double k = blocks3D[block3DIndex+640];
    double l = blocks3D[block3DIndex+704];
    double m = blocks3D[block3DIndex+768];
    double n = blocks3D[block3DIndex+832];
    double o = blocks3D[block3DIndex+896];
    double p = blocks3D[block3DIndex+960];

    double a2 = blocks3D[block3DIndex+1024];
    double b2 = blocks3D[block3DIndex+1088];
    double c2 = blocks3D[block3DIndex+1152];
    double d2 = blocks3D[block3DIndex+1216];
    double e2 = blocks3D[block3DIndex+1280];
    double f2 = blocks3D[block3DIndex+1344];
    double g2 = blocks3D[block3DIndex+1408];
    double h2 = blocks3D[block3DIndex+1472];
    double i2 = blocks3D[block3DIndex+1536];
    double j2 = blocks3D[block3DIndex+1600];
    double k2 = blocks3D[block3DIndex+1664];
    double l2 = blocks3D[block3DIndex+1728];
    double m2 = blocks3D[block3DIndex+1792];
    double n2 = blocks3D[block3DIndex+1856];
    double o2 = blocks3D[block3DIndex+1920];
    double p2 = blocks3D[block3DIndex+1984];

    blocks3D[block3DIndex] = ((a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p) + (a2+b2+c2+d2+e2+f2+g2+h2+i2+j2+k2+l2+m2+n2+o2+p2)) / DIVISOR;
    blocks3D[block3DIndex+64] = ((a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p) + (a2-b2+c2-d2+e2-f2+g2-h2+i2-j2+k2-l2+m2-n2+o2-p2)) / DIVISOR;
    blocks3D[block3DIndex+128] = ((a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p) + (a2+b2-c2-d2+e2+f2-g2-h2+i2+j2-k2-l2+m2+n2-o2-p2)) / DIVISOR;
    blocks3D[block3DIndex+192] = ((a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p) + (a2-b2-c2+d2+e2-f2-g2+h2+i2-j2-k2+l2+m2-n2-o2+p2)) / DIVISOR;
    blocks3D[block3DIndex+256] = ((a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p) + (a2+b2+c2+d2-e2-f2-g2-h2+i2+j2+k2+l2-m2-n2-o2-p2)) / DIVISOR;
    blocks3D[block3DIndex+320] = ((a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p) + (a2-b2+c2-d2-e2+f2-g2+h2+i2-j2+k2-l2-m2+n2-o2+p2)) / DIVISOR;
    blocks3D[block3DIndex+384] = ((a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p) + (a2+b2-c2-d2-e2-f2+g2+h2+i2+j2-k2-l2-m2-n2+o2+p2)) / DIVISOR;
    blocks3D[block3DIndex+448] = ((a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p) + (a2-b2-c2+d2-e2+f2+g2-h2+i2-j2-k2+l2-m2+n2+o2-p2)) / DIVISOR;
    blocks3D[block3DIndex+512] = ((a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p) + (a2+b2+c2+d2+e2+f2+g2+h2-i2-j2-k2-l2-m2-n2-o2-p2)) / DIVISOR;
    blocks3D[block3DIndex+576] = ((a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p) + (a2-b2+c2-d2+e2-f2+g2-h2-i2+j2-k2+l2-m2+n2-o2+p2)) / DIVISOR;
    blocks3D[block3DIndex+640] = ((a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p) + (a2+b2-c2-d2+e2+f2-g2-h2-i2-j2+k2+l2-m2-n2+o2+p2)) / DIVISOR;
    blocks3D[block3DIndex+704] = ((a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p) + (a2-b2-c2+d2+e2-f2-g2+h2-i2+j2+k2-l2-m2+n2+o2-p2)) / DIVISOR;
    blocks3D[block3DIndex+768] = ((a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p) + (a2+b2+c2+d2-e2-f2-g2-h2-i2-j2-k2-l2+m2+n2+o2+p2)) / DIVISOR;
    blocks3D[block3DIndex+832] = ((a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p) + (a2-b2+c2-d2-e2+f2-g2+h2-i2+j2-k2+l2+m2-n2+o2-p2)) / DIVISOR;
    blocks3D[block3DIndex+896] = ((a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p) + (a2+b2-c2-d2-e2-f2+g2+h2-i2-j2+k2+l2+m2+n2-o2-p2)) / DIVISOR;
    blocks3D[block3DIndex+960] = ((a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p) + (a2-b2-c2+d2-e2+f2+g2-h2-i2+j2+k2-l2+m2-n2-o2+p2)) / DIVISOR;

    blocks3D[block3DIndex+1024] = ((a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p) - a2-b2-c2-d2-e2-f2-g2-h2-i2-j2-k2-l2-m2-n2-o2-p2) / DIVISOR;
    blocks3D[block3DIndex+1088] = ((a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p) - a2+b2-c2+d2-e2+f2-g2+h2-i2+j2-k2+l2-m2+n2-o2+p2) / DIVISOR;
    blocks3D[block3DIndex+1152] = ((a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p) - a2-b2+c2+d2-e2-f2+g2+h2-i2-j2+k2+l2-m2-n2+o2+p2) / DIVISOR;
    blocks3D[block3DIndex+1216] = ((a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p) - a2+b2+c2-d2-e2+f2+g2-h2-i2+j2+k2-l2-m2+n2+o2-p2) / DIVISOR;
    blocks3D[block3DIndex+1280] = ((a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p) - a2-b2-c2-d2+e2+f2+g2+h2-i2-j2-k2-l2+m2+n2+o2+p2) / DIVISOR;
    blocks3D[block3DIndex+1344] = ((a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p) - a2+b2-c2+d2+e2-f2+g2-h2-i2+j2-k2+l2+m2-n2+o2-p2) / DIVISOR;
    blocks3D[block3DIndex+1408] = ((a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p) - a2-b2+c2+d2+e2+f2-g2-h2-i2-j2+k2+l2+m2+n2-o2-p2) / DIVISOR;
    blocks3D[block3DIndex+1472] = ((a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p) - a2+b2+c2-d2+e2-f2-g2+h2-i2+j2+k2-l2+m2-n2-o2+p2) / DIVISOR;
    blocks3D[block3DIndex+1536] = ((a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p) - a2-b2-c2-d2-e2-f2-g2-h2+i2+j2+k2+l2+m2+n2+o2+p2) / DIVISOR;
    blocks3D[block3DIndex+1600] = ((a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p) - a2+b2-c2+d2-e2+f2-g2+h2+i2-j2+k2-l2+m2-n2+o2-p2) / DIVISOR;
    blocks3D[block3DIndex+1664] = ((a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p) - a2-b2+c2+d2-e2-f2+g2+h2+i2+j2-k2-l2+m2+n2-o2-p2) / DIVISOR;
    blocks3D[block3DIndex+1728] = ((a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p) - a2+b2+c2-d2-e2+f2+g2-h2+i2-j2-k2+l2+m2-n2-o2+p2) / DIVISOR;
    blocks3D[block3DIndex+1792] = ((a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p) - a2-b2-c2-d2+e2+f2+g2+h2+i2+j2+k2+l2-m2-n2-o2-p2) / DIVISOR;
    blocks3D[block3DIndex+1856] = ((a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p) - a2+b2-c2+d2+e2-f2+g2-h2+i2-j2+k2-l2-m2+n2-o2+p2) / DIVISOR;
    blocks3D[block3DIndex+1920] = ((a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p) - a2-b2+c2+d2+e2+f2-g2-h2+i2+j2-k2-l2-m2-n2+o2+p2) / DIVISOR;
    blocks3D[block3DIndex+1984] = ((a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p) - a2+b2+c2-d2+e2-f2-g2+h2+i2-j2-k2+l2-m2+n2+o2-p2) / DIVISOR;
}


__device__ void Hadamar8(double* inputs, double DIVISOR)
{
    double a = inputs[0];
    double b = inputs[1];
    double c = inputs[2];
    double d = inputs[3];
    double e = inputs[4];  
    double f = inputs[5];
    double g = inputs[6];
    double h = inputs[7];
    
    inputs[0] = (a+b+c+d+e+f+g+h)/DIVISOR;
    inputs[1] = (a-b+c-d+e-f+g-h)/DIVISOR;
    inputs[2] = (a+b-c-d+e+f-g-h)/DIVISOR;
    inputs[3] = (a-b-c+d+e-f-g+h)/DIVISOR;
    inputs[4] = (a+b+c+d-e-f-g-h)/DIVISOR;
    inputs[5] = (a-b+c-d-e+f-g+h)/DIVISOR;
    inputs[6] = (a+b-c-d-e-f+g+h)/DIVISOR;
    inputs[7] = (a-b-c+d-e+f+g-h)/DIVISOR;
}

__global__
void inverseTransform2D_row(double* blocks3D, int size, double DIVISOR)
{
    int blockIndex = (((blockIdx.y * size) + blockIdx.x) << 11) + (threadIdx.y << 6) + (threadIdx.x << 3);
    double inputs[8];
    inputs[0] = blocks3D[blockIndex];
    inputs[1] = blocks3D[blockIndex+1];
    inputs[2] = blocks3D[blockIndex+2];
    inputs[3] = blocks3D[blockIndex+3];
    inputs[4] = blocks3D[blockIndex+4];
    inputs[5] = blocks3D[blockIndex+5];
    inputs[6] = blocks3D[blockIndex+6];
    inputs[7] = blocks3D[blockIndex+7];
    Hadamar8(inputs, DIVISOR);
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
void inverseTransform2D_col(double* blocks3D, int size, double DIVISOR)
{
    int blockIndex = (((blockIdx.y * size) + blockIdx.x) << 11) + (threadIdx.y << 6) + threadIdx.x;
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
    Hadamar8(inputs, DIVISOR);
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
void ApplyCoefBasicEstimate(double* blocks3D, double threshold, int size, float* nbSimilarBlocks)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    int blockPixelIndex = (block << 11) + (blockIdx.z << 6) + (threadIdx.y << 3) + threadIdx.x;
    float coef = 1.0f / nbSimilarBlocks[block];
    blocks3D[blockPixelIndex] = blocks3D[blockPixelIndex] * coef;  
}

void BM3D::BM3D_Inverse3D(bool final)
{
    double DIVISOR = sqrt(8);
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(8, 8);
        if(final)
        {
            float DIVISOR = sqrt(32);
            inverse3D32<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, DIVISOR); 
        }
        else
        {
            inverse3D16<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern);
        }        
        cudaDeviceSynchronize();   
    }
    /*if(!final)
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, 16);
        dim3 numThreads(8, 8);
        ApplyCoefBasicEstimate<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.hardThreshold, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks); 
        cudaDeviceSynchronize();   
    }*/
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(8, (final) ? 32 : 16);
        inverseTransform2D_col<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, DIVISOR); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(8, (final) ? 32 : 16);
        inverseTransform2D_row<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, DIVISOR); 
        cudaDeviceSynchronize();   
    }
}

__global__
void HardThresholdFilter(double* blocks3D, double threshold, int size, float* nbSimilarBlocks)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    int blockPixelIndex = (block << 11) + (blockIdx.z << 6) + (threadIdx.y << 3) + threadIdx.x;
    float coef_norm = sqrtf(nbSimilarBlocks[block]);
    if(blocks3D[blockPixelIndex] <= (threshold * coef_norm)) blocks3D[blockPixelIndex] = 0;  
}

__global__
void CalculateNP(double* blocks3D, int* npArray, int size)
{
    int block = ((blockIdx.y * size) + blockIdx.x);
    int blockIndex = (block << 11) + (blockIdx.z << 6) + (threadIdx.y << 3) + threadIdx.x;
    if(blocks3D[blockIndex] > 0) atomicAdd(&npArray[block], 1);  
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
void showDistance(int block, int size, int* blockMap, double* blocks, int* bmVectors, bool vectorOnly)
{
    int index = block * 1000;
    int bmVectorIndex = block << 5;
    printf("\n");
    printf("\nBM Vector = ");

    if(vectorOnly)
        for(int i=0; i<32; ++i) printf(" %d, ", (bmVectors[bmVectorIndex + i] / 66));

    if(!vectorOnly)
    {
        for(int i= 0; i < 100; i++)
        {
            //if(blockMap[index + (i * 10) + 9] < 99999999)
            {
                int blockIndex = blockMap[index + (i* 10)];
                printf("\ncmp block %d", i);
                printf("\nindex: %d", (blockIndex/ 66));
                printf("\nblock x=%d, y=%d", (int)blocks[blockIndex+64], (int)blocks[blockIndex+65]);
                for(int j = 0; j< 9; ++j)
                    printf("\n\t%d: distance %d", (j+1), blockMap[index + (i * 10) + 1 + j]);
            }
        }
    }
}

void BM3D::BM3D_ShowDistance(int block, bool vectorOnly)
{
   {
        dim3 numBlocks(1);
        dim3 numThreads(1);
        showDistance<<<numBlocks,numThreads>>>(block, BM3D::context.widthBlocksIntern, BM3D::context.blockMap, BM3D::context.blocks, BM3D::context.bmVectors, vectorOnly); 
        cudaDeviceSynchronize();   
   }     
}

__global__
void ShowPixel(int x, int y, int nbBlocks, double* blocks, float* deviceImage, float* basicImage, int widthOrig, int width)
{
    for(int i=0; i < nbBlocks; ++i)
    {
        float xImg = blocks[(i * 66) + 64];
        float yImg = blocks[(i * 66) + 65];
        if(x >= xImg && x <= (xImg + 8) && y >= yImg && y <= (yImg + 8))
            printf("\nblock = %d, x = %d, y = %d, xImg = %f, yImg = %f", i, x, y, xImg, yImg );
    }
    
    int origImageIndex = (y * widthOrig) +x;
    int basicImageIndex = ((y+15) * width) +(x+15);
    printf("\nOrig value = %f", deviceImage[origImageIndex]);
    printf("\nestimate value = %f", basicImage[basicImageIndex]);
        
}

void BM3D::BM3D_ShowPixel(int x, int y)
{
   {
        dim3 numBlocks(1);
        dim3 numThreads(1);
        ShowPixel<<<numBlocks,numThreads>>>(x, y, BM3D::context.nbBlocks, BM3D::context.blocks, BM3D::context.deviceImage, BM3D::context.basicImage,  BM3D::context.img_widthOrig, BM3D::context.img_width); 
        cudaDeviceSynchronize();   
   }     
}

__global__
void BM_CalculateDistance(int* blockMap, double* blocks, int size)
{
    int blockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 1000) + ((threadIdx.y * 10 + threadIdx.x) * 10);
    int cmpBlockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 1000) + 550;
    int cmpBlockIndex = blockMap[cmpBlockMapIndex] + (threadIdx.z << 3);
    int blockIndex =  blockMap[blockMapIndex] + (threadIdx.z << 3);
   

    blockMap[blockMapIndex + 1 + threadIdx.z] = int(
          
                                ((blocks[cmpBlockIndex] - blocks[blockIndex]) * (blocks[cmpBlockIndex] - blocks[blockIndex])) +
                                ((blocks[cmpBlockIndex+1] - blocks[blockIndex+1]) * (blocks[cmpBlockIndex+1] - blocks[blockIndex+1])) +
                                ((blocks[cmpBlockIndex+2] - blocks[blockIndex+2]) * (blocks[cmpBlockIndex+2] - blocks[blockIndex+2])) +
                                ((blocks[cmpBlockIndex+3] - blocks[blockIndex+3]) * (blocks[cmpBlockIndex+3] - blocks[blockIndex+3])) +
                                ((blocks[cmpBlockIndex+4] - blocks[blockIndex+4]) * (blocks[cmpBlockIndex+4] - blocks[blockIndex+4])) +
                                ((blocks[cmpBlockIndex+5] - blocks[blockIndex+5]) * (blocks[cmpBlockIndex+5] - blocks[blockIndex+5])) +
                                ((blocks[cmpBlockIndex+6] - blocks[blockIndex+6]) * (blocks[cmpBlockIndex+6] - blocks[blockIndex+6])) +
                                ((blocks[cmpBlockIndex+7] - blocks[blockIndex+7]) * (blocks[cmpBlockIndex+7] - blocks[blockIndex+7])));
}

/*__global__
void BM_CalculateDistance2(int* blockMap, double* blocks, int size, float* integralImage, int width)
{
    int blockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 1000) + ((threadIdx.y * 10 + threadIdx.x) * 10);
    int cmpBlockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 1000) + 550;
    int cmpBlockIndex = blockMap[cmpBlockMapIndex];
    int blockIndex =  blockMap[blockMapIndex];
    int xRefBlock = int(blocks[blockIndex + 64]);
    int yRefBlock = int(blocks[blockIndex + 65]);
    int xCmpBlock = int(blocks[cmpBlockIndex + 64]);
    int yCmpBlock = int(blocks[cmpBlockIndex + 65]);
    float A = integralImage[

    blockMap[blockMapIndex + 1 + threadIdx.z] = int(
          
                                ((blocks[cmpBlockIndex] - blocks[blockIndex]) * (blocks[cmpBlockIndex] - blocks[blockIndex])) +
                                ((blocks[cmpBlockIndex+1] - blocks[blockIndex+1]) * (blocks[cmpBlockIndex+1] - blocks[blockIndex+1])) +
                                ((blocks[cmpBlockIndex+2] - blocks[blockIndex+2]) * (blocks[cmpBlockIndex+2] - blocks[blockIndex+2])) +
                                ((blocks[cmpBlockIndex+3] - blocks[blockIndex+3]) * (blocks[cmpBlockIndex+3] - blocks[blockIndex+3])) +
                                ((blocks[cmpBlockIndex+4] - blocks[blockIndex+4]) * (blocks[cmpBlockIndex+4] - blocks[blockIndex+4])) +
                                ((blocks[cmpBlockIndex+5] - blocks[blockIndex+5]) * (blocks[cmpBlockIndex+5] - blocks[blockIndex+5])) +
                                ((blocks[cmpBlockIndex+6] - blocks[blockIndex+6]) * (blocks[cmpBlockIndex+6] - blocks[blockIndex+6])) +
                                ((blocks[cmpBlockIndex+7] - blocks[blockIndex+7]) * (blocks[cmpBlockIndex+7] - blocks[blockIndex+7])));
}*/

__global__
void BM_AddAndLimit(int* blockMap, int size, int limit)
{
    int blockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 1000) + (threadIdx.y * 10 + threadIdx.x) * 10;
    int sum = blockMap[blockMapIndex + 1] + blockMap[blockMapIndex + 2] + blockMap[blockMapIndex + 3] + blockMap[blockMapIndex + 4] + blockMap[blockMapIndex + 5] + blockMap[blockMapIndex + 6] + blockMap[blockMapIndex + 7] + blockMap[blockMapIndex + 8];
    if(sum == 0 && threadIdx.x != 5 && threadIdx.y != 5) sum = limit;
    blockMap[blockMapIndex + 9] = (sum < limit) ? sum : 99999999;
}


__global__
void BM_Sort(int* blockMap, int size)
{
    int blockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 1000);
    int currentBlockIndex = blockMapIndex + (threadIdx.y * 100) + (threadIdx.x *10);
    int distance = blockMap[currentBlockIndex+9];

    if(distance < 99999999)
    {
        int index = 0;
        for(int i=0; i<100; ++i)
        {
            if(distance > blockMap[blockMapIndex + i * 10 + 9]) index++;
        }
        blockMap[currentBlockIndex+1] = index;
    }
}

__global__
void BM_CreateBmVector(int* blockMap, int* bmVectors, int size, float* nbSimilarBlocks, int sizeBMVector)
{
    int currentBlockIndex = (((blockIdx.y * size) + blockIdx.x) * 1000) + (threadIdx.y * 100) + (threadIdx.x *10);
    int bmVectorIndex = (((blockIdx.y * size) + blockIdx.x) << 5);
    if(blockMap[currentBlockIndex+9] < 99999999 && blockMap[currentBlockIndex+1] < sizeBMVector)
    {
        bmVectors[bmVectorIndex + blockMap[currentBlockIndex+1]] = blockMap[currentBlockIndex];
        atomicAdd(&nbSimilarBlocks[(blockIdx.y * size) + blockIdx.x], 1);
    }
}

__global__
void Create3DBlocks16(double* blocks, double* blocks3D, int* bmVectors, int size)
{
    int index = (blockIdx.y * size) + blockIdx.x;
    int bmVectorIndex = index << 5;
    {
        
        int pixelIndex = (threadIdx.y << 3) + threadIdx.x;
        int block3DIndex = (index << 11) + pixelIndex;
        //we can assume that the top-left corner of the basic image always has a pixel egal to 0 due to the shift 
        //of the image. 

        double a = blocks[bmVectors[bmVectorIndex] + pixelIndex];
        double b = blocks[bmVectors[bmVectorIndex+1] + pixelIndex];
        double c = blocks[bmVectors[bmVectorIndex+2] + pixelIndex];
        double d = blocks[bmVectors[bmVectorIndex+3] + pixelIndex];
        double e = blocks[bmVectors[bmVectorIndex+4] + pixelIndex];
        double f = blocks[bmVectors[bmVectorIndex+5] + pixelIndex];
        double g = blocks[bmVectors[bmVectorIndex+6] + pixelIndex];
        double h = blocks[bmVectors[bmVectorIndex+7] + pixelIndex];
        double i = blocks[bmVectors[bmVectorIndex+8] + pixelIndex];
        double j = blocks[bmVectors[bmVectorIndex+9] + pixelIndex];
        double k = blocks[bmVectors[bmVectorIndex+10] + pixelIndex];
        double l = blocks[bmVectors[bmVectorIndex+11] + pixelIndex];
        double m = blocks[bmVectors[bmVectorIndex+12] + pixelIndex];
        double n = blocks[bmVectors[bmVectorIndex+13] + pixelIndex];
        double o = blocks[bmVectors[bmVectorIndex+14] + pixelIndex];
        double p = blocks[bmVectors[bmVectorIndex+15] + pixelIndex];

        
        blocks3D[block3DIndex] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p) / 4.0;
        blocks3D[block3DIndex+64] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p) / 4.0;
        blocks3D[block3DIndex+128] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p) / 4.0;
        blocks3D[block3DIndex+192] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p) / 4.0;
        blocks3D[block3DIndex+256] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p) / 4.0;
        blocks3D[block3DIndex+320] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p) / 4.0;
        blocks3D[block3DIndex+384] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p) / 4.0;
        blocks3D[block3DIndex+448] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p) / 4.0;
        blocks3D[block3DIndex+512] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p) / 4.0;
        blocks3D[block3DIndex+576] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p) / 4.0;
        blocks3D[block3DIndex+640] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p) / 4.0;
        blocks3D[block3DIndex+704] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p) / 4.0;
        blocks3D[block3DIndex+768] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p) / 4.0;
        blocks3D[block3DIndex+832] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p) / 4.0;
        blocks3D[block3DIndex+896] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p) / 4.0;
        blocks3D[block3DIndex+960] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p) / 4.0;

    }
}

__global__
void Create3DBlocks32(double* blocks, double* blocks3D, int* bmVectors, int size, float DIVISOR)
{
    int index = (blockIdx.y * size) + blockIdx.x;
    int bmVectorIndex = index << 5;
    {
        
        int pixelIndex = (threadIdx.y << 3) + threadIdx.x;
        int block3DIndex = (index << 11) + pixelIndex;
        //we can assume that the top-left corner of the basic image always has a pixel egal to 0 due to the shift 
        //of the image. 

        double a = blocks[bmVectors[bmVectorIndex] + pixelIndex];
        double b = blocks[bmVectors[bmVectorIndex+1] + pixelIndex];
        double c = blocks[bmVectors[bmVectorIndex+2] + pixelIndex];
        double d = blocks[bmVectors[bmVectorIndex+3] + pixelIndex];
        double e = blocks[bmVectors[bmVectorIndex+4] + pixelIndex];
        double f = blocks[bmVectors[bmVectorIndex+5] + pixelIndex];
        double g = blocks[bmVectors[bmVectorIndex+6] + pixelIndex];
        double h = blocks[bmVectors[bmVectorIndex+7] + pixelIndex];
        double i = blocks[bmVectors[bmVectorIndex+8] + pixelIndex];
        double j = blocks[bmVectors[bmVectorIndex+9] + pixelIndex];
        double k = blocks[bmVectors[bmVectorIndex+10] + pixelIndex];
        double l = blocks[bmVectors[bmVectorIndex+11] + pixelIndex];
        double m = blocks[bmVectors[bmVectorIndex+12] + pixelIndex];
        double n = blocks[bmVectors[bmVectorIndex+13] + pixelIndex];
        double o = blocks[bmVectors[bmVectorIndex+14] + pixelIndex];
        double p = blocks[bmVectors[bmVectorIndex+15] + pixelIndex];

        double a2 = blocks[bmVectors[bmVectorIndex+16] + pixelIndex];
        double b2 = blocks[bmVectors[bmVectorIndex+17] + pixelIndex];
        double c2 = blocks[bmVectors[bmVectorIndex+18] + pixelIndex];
        double d2 = blocks[bmVectors[bmVectorIndex+19] + pixelIndex];
        double e2 = blocks[bmVectors[bmVectorIndex+20] + pixelIndex];
        double f2 = blocks[bmVectors[bmVectorIndex+21] + pixelIndex];
        double g2 = blocks[bmVectors[bmVectorIndex+22] + pixelIndex];
        double h2 = blocks[bmVectors[bmVectorIndex+23] + pixelIndex];
        double i2 = blocks[bmVectors[bmVectorIndex+24] + pixelIndex];
        double j2 = blocks[bmVectors[bmVectorIndex+25] + pixelIndex];
        double k2 = blocks[bmVectors[bmVectorIndex+26] + pixelIndex];
        double l2 = blocks[bmVectors[bmVectorIndex+27] + pixelIndex];
        double m2 = blocks[bmVectors[bmVectorIndex+28] + pixelIndex];
        double n2 = blocks[bmVectors[bmVectorIndex+29] + pixelIndex];
        double o2 = blocks[bmVectors[bmVectorIndex+30] + pixelIndex];
        double p2 = blocks[bmVectors[bmVectorIndex+31] + pixelIndex];

        //if(index == 6515)
          //  printf("\n%d, %d, %d, %d, %f", block3DIndex, index, pixelIndex, bmVectors[bmVectorIndex+30], blocks[0]);
            //printf("\n%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f:%d:%d,%f:%d:%d,%f:%d, %d, %d, %d",
                //a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,a2,b2,c2,d2,e2,f2,g2,h2,i2,j2,k2,l2,m2,n2, bmVectors[bmVectorIndex+29], bmVectorIndex,o2,bmVectors[bmVectorIndex+30], bmVectorIndex,p2, bmVectors[bmVectorIndex+31], block3DIndex, index, pixelIndex);

        
        blocks3D[block3DIndex] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p + a2+b2+c2+d2+e2+f2+g2+h2+i2+j2+k2+l2+m2+n2+o2+p2) / DIVISOR;
        blocks3D[block3DIndex+64] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p + a2-b2+c2-d2+e2-f2+g2-h2+i2-j2+k2-l2+m2-n2+o2-p2) / DIVISOR;
        blocks3D[block3DIndex+128] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p + a2+b2-c2-d2+e2+f2-g2-h2+i2+j2-k2-l2+m2+n2-o2-p2) / DIVISOR;
        blocks3D[block3DIndex+192] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p + a2-b2-c2+d2+e2-f2-g2+h2+i2-j2-k2+l2+m2-n2-o2+p2) / DIVISOR;
        blocks3D[block3DIndex+256] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p + a2+b2+c2+d2-e2-f2-g2-h2+i2+j2+k2+l2-m2-n2-o2-p2) / DIVISOR;
        blocks3D[block3DIndex+320] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p + a2-b2+c2-d2-e2+f2-g2+h2+i2-j2+k2-l2-m2+n2-o2+p2) / DIVISOR;
        blocks3D[block3DIndex+384] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p + a2+b2-c2-d2-e2-f2+g2+h2+i2+j2-k2-l2-m2-n2+o2+p2) / DIVISOR;
        blocks3D[block3DIndex+448] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p + a2-b2-c2+d2-e2+f2+g2-h2+i2-j2-k2+l2-m2+n2+o2-p2) / DIVISOR;
        blocks3D[block3DIndex+512] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p + a2+b2+c2+d2+e2+f2+g2+h2-i2-j2-k2-l2-m2-n2-o2-p2) / DIVISOR;
        blocks3D[block3DIndex+576] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p + a2-b2+c2-d2+e2-f2+g2-h2-i2+j2-k2+l2-m2+n2-o2+p2) / DIVISOR;
        blocks3D[block3DIndex+640] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p + a2+b2-c2-d2+e2+f2-g2-h2-i2-j2+k2+l2-m2-n2+o2+p2) / DIVISOR;
        blocks3D[block3DIndex+704] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p + a2-b2-c2+d2+e2-f2-g2+h2-i2+j2+k2-l2-m2+n2+o2-p2) / DIVISOR;
        blocks3D[block3DIndex+768] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p + a2+b2+c2+d2-e2-f2-g2-h2-i2-j2-k2-l2+m2+n2+o2+p2) / DIVISOR;
        blocks3D[block3DIndex+832] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p + a2-b2+c2-d2-e2+f2-g2+h2-i2+j2-k2+l2+m2-n2+o2-p2) / DIVISOR;
        blocks3D[block3DIndex+896] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p + a2+b2-c2-d2-e2-f2+g2+h2-i2-j2+k2+l2+m2+n2-o2-p2) / DIVISOR;
        blocks3D[block3DIndex+960] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p + a2-b2-c2+d2-e2+f2+g2-h2-i2+j2+k2-l2+m2-n2-o2+p2) / DIVISOR;

        blocks3D[block3DIndex+1024] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p - a2-b2-c2-d2-e2-f2-g2-h2-i2-j2-k2-l2-m2-n2-o2-p2) / DIVISOR;
        blocks3D[block3DIndex+1088] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p - a2+b2-c2+d2-e2+f2-g2+h2-i2+j2-k2+l2-m2+n2-o2+p2) / DIVISOR;
        blocks3D[block3DIndex+1152] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p - a2-b2+c2+d2-e2-f2+g2+h2-i2-j2+k2+l2-m2-n2+o2+p2) / DIVISOR;
        blocks3D[block3DIndex+1216] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p - a2+b2+c2-d2-e2+f2+g2-h2-i2+j2+k2-l2-m2+n2+o2-p2) / DIVISOR;
        blocks3D[block3DIndex+1280] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p - a2-b2-c2-d2+e2+f2+g2+h2-i2-j2-k2-l2+m2+n2+o2+p2) / DIVISOR;
        blocks3D[block3DIndex+1344] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p - a2+b2-c2+d2+e2-f2+g2-h2-i2+j2-k2+l2+m2-n2+o2-p2) / DIVISOR;
        blocks3D[block3DIndex+1408] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p - a2-b2+c2+d2+e2+f2-g2-h2-i2-j2+k2+l2+m2+n2-o2-p2) / DIVISOR;
        blocks3D[block3DIndex+1472] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p - a2+b2+c2-d2+e2-f2-g2+h2-i2+j2+k2-l2+m2-n2-o2+p2) / DIVISOR;
        blocks3D[block3DIndex+1536] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p - a2-b2-c2-d2-e2-f2-g2-h2+i2+j2+k2+l2+m2+n2+o2+p2) / DIVISOR;
        blocks3D[block3DIndex+1600] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p - a2+b2-c2+d2-e2+f2-g2+h2+i2-j2+k2-l2+m2-n2+o2-p2) / DIVISOR;
        blocks3D[block3DIndex+1664] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p - a2-b2+c2+d2-e2-f2+g2+h2+i2+j2-k2-l2+m2+n2-o2-p2) / DIVISOR;
        blocks3D[block3DIndex+1728] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p - a2+b2+c2-d2-e2+f2+g2-h2+i2-j2-k2+l2+m2-n2-o2+p2) / DIVISOR;
        blocks3D[block3DIndex+1792] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p - a2-b2-c2-d2+e2+f2+g2+h2+i2+j2+k2+l2-m2-n2-o2-p2) / DIVISOR;
        blocks3D[block3DIndex+1856] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p - a2+b2-c2+d2+e2-f2+g2-h2+i2-j2+k2-l2-m2+n2-o2+p2) / DIVISOR;
        blocks3D[block3DIndex+1920] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p - a2-b2+c2+d2+e2+f2-g2-h2+i2+j2-k2-l2-m2-n2+o2+p2) / DIVISOR;
        blocks3D[block3DIndex+1984] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p - a2+b2+c2-d2+e2-f2-g2+h2+i2-j2-k2+l2-m2+n2+o2-p2) / DIVISOR;
    }
}

void BM3D::BM3D_BlockMatching(bool final)
{

    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(10, 10, 8);
        BM_CalculateDistance<<<numBlocks,numThreads>>>(BM3D::context.blockMap, BM3D::context.blocks, BM3D::context.widthBlocksIntern); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(10, 10);
        BM_AddAndLimit<<<numBlocks,numThreads>>>(BM3D::context.blockMap, BM3D::context.widthBlocksIntern, (final) ? BM3D::context.wienLimit : BM3D::context.hardLimit); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(10, 10);
        BM_Sort<<<numBlocks,numThreads>>>(BM3D::context.blockMap, BM3D::context.widthBlocksIntern); 
        cudaDeviceSynchronize();
        BM_CreateBmVector<<<numBlocks,numThreads>>>(BM3D::context.blockMap, BM3D::context.bmVectors, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks, (final) ? 32 : 16); 
        cudaDeviceSynchronize();
    }
    //BM3D_ShowDistance(6515, true);
    {

        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(8, 8);
        if(final)
        {
            float DIVISOR = sqrt(32);
            //BM3D_ShowBlock(100);
            Create3DBlocks32<<<numBlocks,numThreads>>>(BM3D::context.blocks, BM3D::context.blocks3D, BM3D::context.bmVectors, BM3D::context.widthBlocksIntern, DIVISOR); 
            cudaDeviceSynchronize();
            Create3DBlocks32<<<numBlocks,numThreads>>>(BM3D::context.blocksOrig, BM3D::context.blocks3DOrig, BM3D::context.bmVectors, BM3D::context.widthBlocksIntern, DIVISOR); 
            cudaDeviceSynchronize(); 
        }
        else
        {
            //BM3D_ShowBlock(100);
            Create3DBlocks16<<<numBlocks,numThreads>>>(BM3D::context.blocks, BM3D::context.blocks3D, BM3D::context.bmVectors, BM3D::context.widthBlocksIntern); 
            cudaDeviceSynchronize();
        }
    }
}


__global__
void ShowBlock(int block, int size, double* blocks)
{
    int index = block * 66;
    printf("\n\n");
    for(int i = 0; i < 64; i++) printf("%f, ", blocks[index+i]);
    printf("\nx = %f, y = %f", blocks[index+64], blocks[index+65]);
}

void BM3D::BM3D_ShowBlock(int block)
{
   {
        dim3 numBlocks(1);
        dim3 numThreads(1);
        ShowBlock<<<numBlocks,numThreads>>>(block, BM3D::context.widthBlocks, BM3D::context.blocks); 
        cudaDeviceSynchronize();   
   }
}

__global__
void ShowBlock3D(int block, int size, double* blocks3D)
{
    int index = block << 11;
    printf("\n\n");
    for(int i = 0; i < 64; i++) printf("%f, ", blocks3D[index+i]);
}

void BM3D::BM3D_ShowBlock3D(int block, bool orig)
{
   {
        dim3 numBlocks(1);
        dim3 numThreads(1);
        ShowBlock3D<<<numBlocks,numThreads>>>(block, BM3D::context.widthBlocks, (orig) ? BM3D::context.blocks3DOrig : BM3D::context.blocks3D); 
        cudaDeviceSynchronize();   
   }
}

__global__
void Transform2D_row(double* blocks, int size, double DIVISOR)
{
    int blockIndex = (((blockIdx.y * size) + blockIdx.x) * 66) + (threadIdx.x << 3);
    double inputs[8];
    inputs[0] = blocks[blockIndex];
    inputs[1] = blocks[blockIndex+1];
    inputs[2] = blocks[blockIndex+2];
    inputs[3] = blocks[blockIndex+3];
    inputs[4] = blocks[blockIndex+4];
    inputs[5] = blocks[blockIndex+5];
    inputs[6] = blocks[blockIndex+6];
    inputs[7] = blocks[blockIndex+7];
    Hadamar8(inputs, DIVISOR);
    blocks[blockIndex] = inputs[0];
    blocks[blockIndex+1] = inputs[1];
    blocks[blockIndex+2] = inputs[2];
    blocks[blockIndex+3] = inputs[3];
    blocks[blockIndex+4] = inputs[4];
    blocks[blockIndex+5] = inputs[5];
    blocks[blockIndex+6] = inputs[6];
    blocks[blockIndex+7] = inputs[7];
}

__global__
void Transform2D_col(double* blocks, int size, double DIVISOR)
{
    int blockIndex = (((blockIdx.y * size) + blockIdx.x) * 66) + threadIdx.x;
    double inputs[8];
    int index = blockIndex;
    inputs[0] = blocks[index];
    index += 8;
    inputs[1] = blocks[index];
    index += 8;
    inputs[2] = blocks[index];
    index += 8;
    inputs[3] = blocks[index];
    index += 8;
    inputs[4] = blocks[index];
    index += 8;
    inputs[5] = blocks[index];
    index += 8;
    inputs[6] = blocks[index];
    index += 8;
    inputs[7] = blocks[index];
    Hadamar8(inputs, DIVISOR);
    index = blockIndex;
    blocks[index] = inputs[0];
    index += 8;
    blocks[index] = inputs[1];
    index += 8;
    blocks[index] = inputs[2];
    index += 8;
    blocks[index] = inputs[3];
    index += 8;
    blocks[index] = inputs[4];
    index += 8;
    blocks[index] = inputs[5];
    index += 8;
    blocks[index] = inputs[6];
    index += 8;
    blocks[index] = inputs[7];
}

__global__
void CopyBlocks(double* blocks, double* blocksOrig, int size)
{
    int blockIndex = ((blockIdx.y * size) + blockIdx.x) * 66 + (threadIdx.y << 3) + threadIdx.x;
    blocksOrig[blockIndex] = blocks[blockIndex];
}

void BM3D::BM3D_2DTransform(bool final)
{
   double DIVISOR = sqrt(8);
   {
        dim3 numBlocks(BM3D::context.widthBlocks, BM3D::context.widthBlocks);
        dim3 numThreads(8);
        Transform2D_row<<<numBlocks,numThreads>>>(BM3D::context.blocks, BM3D::context.widthBlocks, DIVISOR); 
        cudaDeviceSynchronize();   
   }
   {
        dim3 numBlocks(BM3D::context.widthBlocks, BM3D::context.widthBlocks);
        dim3 numThreads(8);
        Transform2D_col<<<numBlocks,numThreads>>>(BM3D::context.blocks, BM3D::context.widthBlocks, DIVISOR); 
        cudaDeviceSynchronize();   
   }
   if(!final)
   {
        dim3 numBlocks(BM3D::context.widthBlocks, BM3D::context.widthBlocks);
        dim3 numThreads(8,8);
        CopyBlocks<<<numBlocks,numThreads>>>(BM3D::context.blocks, BM3D::context.blocksOrig, BM3D::context.widthBlocks); 
        cudaDeviceSynchronize();
   }
}

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
void CreateBlocks(float* basicImage, double* blocks, int size, int width, int pHard)
{
    int blockPixelIndex = (((blockIdx.y * size) + blockIdx.x) * 66) + (threadIdx.y << 3) + threadIdx.x;
    int imgIndex = ((blockIdx.y * pHard + threadIdx.y) * width) + (blockIdx.x * pHard + threadIdx.x);
    blocks[blockPixelIndex] = basicImage[imgIndex];  
}

__global__
void CreateBlocksMap(int* blockMap, int sizeIntern, int size, int pHard)
{
    int blockMapIndex = (((blockIdx.y * sizeIntern) + blockIdx.x) * 1000) + ((threadIdx.y * 10) + threadIdx.x) * 10;
    blockMap[blockMapIndex] = (((blockIdx.y + threadIdx.y) * size) + (blockIdx.x + threadIdx.x)) * 66; //block index
}

__global__
void SetBlockPosition(double* blocks, int size, int pHard)
{
    int blockIndex = ((blockIdx.y * size) + blockIdx.x) * 66;
    blocks[blockIndex+64] = blockIdx.x * pHard;
    blocks[blockIndex+65] = blockIdx.y * pHard;
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
   int offset = 5 * BM3D::context.pHard;
   {
        dim3 numBlocks(BM3D::context.img_widthOrig/8, BM3D::context.img_heightOrig/8);
        dim3 numThreads(8,8);
        InverseShiftImage<<<numBlocks,numThreads>>>(BM3D::context.deviceImage, BM3D::context.basicImage, BM3D::context.img_widthOrig, BM3D::context.img_width, offset); 
        cudaDeviceSynchronize();   
   } 
}

void BM3D::BM3D_CreateBlock()
{
    int offset = 5 * BM3D::context.pHard;
    {
        dim3 numBlocks(BM3D::context.img_widthOrig/8, BM3D::context.img_heightOrig/8);
        dim3 numThreads(8,8);
        ShiftImage<<<numBlocks,numThreads>>>(BM3D::context.deviceImage, BM3D::context.basicImage, BM3D::context.img_widthOrig, BM3D::context.img_width, offset); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocks, BM3D::context.widthBlocks);
        dim3 numThreads(8,8);
        CreateBlocks<<<numBlocks,numThreads>>>(BM3D::context.basicImage, BM3D::context.blocks, BM3D::context.widthBlocks, BM3D::context.img_width, BM3D::context.pHard); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocks, BM3D::context.widthBlocks);
        dim3 numThreads(1);
        SetBlockPosition<<<numBlocks,numThreads>>>(BM3D::context.blocks, BM3D::context.widthBlocks, BM3D::context.pHard); 
        cudaDeviceSynchronize();
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(10,10);
        CreateBlocksMap<<<numBlocks,numThreads>>>(BM3D::context.blockMap, BM3D::context.widthBlocksIntern, BM3D::context.widthBlocks, BM3D::context.pHard); 
        cudaDeviceSynchronize();   
    }  
}








	
