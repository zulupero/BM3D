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

__constant__ float DCTv8matrix[] =
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

__constant__ float DCTv8matrixT[] = 
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
    printf("\n\tWidth                     = %d", BM3D::context.img_width);
    printf("\n\tHeight                    = %d", BM3D::context.img_height);
    printf("\n\tDevice image              = %f Mb", (width * height * sizeof(float)/1024.00 / 1024.00));  
    printf("\n\tBasic image               = %f Mb", (w2 * h2 * sizeof(float)/1024.00 / 1024.00));
    //printf("\n\tBlocks array              = %f Mb", (BM3D::context.nbBlocks * 66 * sizeof(double)/1024.00 / 1024.00));  
    //printf("\n\tBlocks array (orig)       = %f Mb", (BM3D::context.nbBlocks * 66 * sizeof(double)/1024.00 / 1024.00));  
    printf("\n\tWindow map                = %f Mb", (BM3D::context.nbBlocksIntern * BM3D::context.windowSize * sizeof(int)/1024.00 / 1024.00));  
    //printf("\n\tBM vectors                = %f Mb", (BM3D::context.nbBlocksIntern * 32 * sizeof(int)/1024.00 / 1024.00)); 
    printf("\n\tBlocks 3D                 = %f Mb", (BM3D::context.nbBlocksIntern * 32 * 66 * sizeof(double)/1024.00 / 1024.00)); //+ x,y 
    printf("\n\tBlocks 3D (orig)          = %f Mb", (BM3D::context.nbBlocksIntern * 32 * 66 * sizeof(double)/1024.00 / 1024.00)); //+ x,y
    printf("\n\tNP array                  = %f Mb", (BM3D::context.nbBlocksIntern * sizeof(int)/1024.00 / 1024.00));
    printf("\n\tWP array                  = %f Mb", (BM3D::context.nbBlocksIntern * sizeof(double)/1024.00 / 1024.00));
    printf("\n\tEstimates array           = %f Mb", (w2 * h2 * 2 * sizeof(float)/1024.00 / 1024.00));
    printf("\n\tSimilar blocks array      = %f Mb", (BM3D::context.nbBlocksIntern * sizeof(int)/1024.00 / 1024.00));
    //printf("\n\tTinv +Tfor ma             = %f Mb", (2 * 64  * sizeof(float)/1024.00 / 1024.00));
    //printf("\n\tMean values               = %f Mb", (BM3D::context.nbBlocks * sizeof(double)/1024.00 / 1024.00));    

    gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, width * height * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.deviceImage, &img[0], width * height * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&BM3D::context.basicImage, w2 * h2 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.basicImage, 0, w2 * h2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.estimates, w2 * h2 * 2 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.estimates, 0, w2 * h2 * 2 * sizeof(float)));

    /*float Tforward[64] = {1,1,1,1,1,1,1,1,
                          1,-1,1,-1,1,-1,1,-1,
                          1,1,-1,-1,1,1,-1,-1,
                          1,-1,-1,1,1,-1,-1,1,
                          1,1,1,1,-1,-1,-1,-1,
                          1,-1,1,-1,-1,1,-1,1,
                          1,1,-1,-1,-1,-1,1,1,
                          1,-1,-1,1,-1,1,1,-1};*/
        

    

/*    float Tforward[64] = {0.353553390593274, 0.353553390593274, 0.35355339059327, 0.35355339059327, 0.353553390593274, 0.353553390593274, 0.353553390593274,   0.353553390593274,
       0.490392640201615, 0.415734806151273, 0.277785116509801, 0.097545161008064, -0.097545161008064, -0.277785116509801,-0.415734806151273, -0.490392640201615,
       0.461939766255643,   0.191341716182545 , -0.191341716182545 , -0.461939766255643 , -0.461939766255643  ,-0.191341716182545 ,  0.191341716182545 ,  0.461939766255643,
       0.415734806151273,  -0.097545161008064,  -0.490392640201615 , -0.277785116509801 ,  0.277785116509801 ,  0.490392640201615 ,  0.097545161008064 , -0.415734806151273,
       0.353553390593274,  -0.353553390593274,  -0.353553390593274,   0.353553390593274 ,  0.353553390593274  ,-0.353553390593274 , -0.353553390593274 ,  0.353553390593274,
       0.277785116509801,  -0.490392640201615 ,  0.097545161008064 ,  0.415734806151273 , -0.415734806151273 , -0.097545161008064 ,  0.490392640201615 , -0.277785116509801,
       0.191341716182545 , -0.461939766255643,   0.461939766255643 , -0.191341716182545 , -0.191341716182545  , 0.461939766255643 , -0.461939766255643 ,  0.191341716182545,
       0.097545161008064,  -0.277785116509801 ,  0.415734806151273 , -0.490392640201615 ,  0.490392640201615 , -0.415734806151273 ,  0.277785116509801 , -0.097545161008064};
   
    float TInverse[64] = {0.353553390593274,   0.490392640201615,   0.461939766255644,   0.415734806151273,   0.353553390593273 ,  0.277785116509801 ,  0.191341716182545,   0.097545161008064,
   0.353553390593273,   0.415734806151273 ,  0.191341716182545 , -0.097545161008064 , -0.353553390593274 , -0.490392640201615 , -0.461939766255644 , -0.277785116509801,
   0.353553390593274,   0.277785116509801,  -0.191341716182545,  -0.490392640201615 , -0.353553390593274 ,  0.097545161008064 ,  0.461939766255644 ,  0.415734806151273,
   0.353553390593274,   0.097545161008065,  -0.461939766255644,  -0.277785116509801,   0.353553390593273 ,  0.415734806151273 , -0.191341716182545 , -0.490392640201615,
   0.353553390593274,  -0.097545161008064,  -0.461939766255644,   0.277785116509801,   0.353553390593274 , -0.415734806151273 , -0.191341716182545 ,  0.490392640201615,
   0.353553390593274,  -0.277785116509801,  -0.191341716182545,   0.490392640201615 , -0.353553390593274 , -0.097545161008064,   0.461939766255644 , -0.415734806151273,
   0.353553390593274,  -0.415734806151273,   0.191341716182545,   0.097545161008065,  -0.353553390593274 ,  0.490392640201615,  -0.461939766255644 ,  0.277785116509801,
   0.353553390593274,  -0.490392640201615,   0.461939766255644,  -0.415734806151273,  0.353553390593273 , -0.277785116509801 ,  0.191341716182545 , -0.097545161008064};*/

    /*gpuErrchk(cudaMalloc(&BM3D::context.Tforward, 64 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.Tinverse, 64 * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.Tforward, &Tforward[0], 64 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(BM3D::context.Tinverse, &TInverse[0], 64 * sizeof(float), cudaMemcpyHostToDevice));*/

    //! First quarter of the matrix
   

    /*float kaiserWindow[64] = { 0.6387, 0.6665, 0.6943, 0.7219, 0.7219, 0.6943, 0.6665, 0.6387, 
                               0.8559, 0.8813, 0.9062, 0.9304, 0.9304, 0.9062, 0.9813, 0.8559,
                               0.10397, 0.10588, 0.10768, 0.10938, 0.10938, 0.10768, 0.10588, 0.10397,
                               0.11609, 0.11706, 0.11789, 0.11858, 0.11858, 0.11789, 0.11706, 0.11609,
                               0.11609, 0.11706, 0.11789, 0.11858, 0.11858, 0.11789, 0.11706, 0.11609,
                               0.10397, 0.10588, 0.10768, 0.10938, 0.10938, 0.10768, 0.10588, 0.10397,
                               0.8559, 0.8813, 0.9062, 0.9304, 0.9304, 0.9062, 0.9813, 0.8559,
                               0.6387, 0.6665, 0.6943, 0.7219, 0.7219, 0.6943, 0.6665, 0.6387 };*/

    //gpuErrchk(cudaMalloc(&BM3D::context.kaiserWindowCoef, 64 * sizeof(float)));
    //gpuErrchk(cudaMemcpy(BM3D::context.kaiserWindowCoef, kaiserWindow, 64 * sizeof(float), cudaMemcpyHostToDevice));
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
    //BM3D_FinalEstimate();
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
    gpuErrchk(cudaMemset(BM3D::context.blocks3DOrig, 0, BM3D::context.nbBlocksIntern * 32 * 66 * sizeof(double)));

    printf("\n\tFinal estimates (2 step)");
    Timer::start(); 
    BM3D_CreateWindow();    
    BM3D_BlockMatching(true);
    //BM3D_2DTransform2(true);
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
    //BM3D_2DTransform2();
    BM3D_Create3DBlocks();
    BM3D_HardThresholdFilter();
    BM3D_Inverse3D();
    BM3D_Aggregation();
    //BM3D_InverseShift();
    Timer::add("BM3D-Basic estimates");
    //BM3D_SaveImage();
}


__global__
void WienFilter(double* blocks3D, double* blocks3DOrig, int size, int* similarBlocks, int sigma, float* wpArray)
{    
    int block = (blockIdx.y * size) + blockIdx.x;
    if(blockIdx.z < similarBlocks[block])
    {
        float coef = 1.0f / similarBlocks[block];
        int blockPixelIndex = (block << 11) + (blockIdx.z << 6) + (threadIdx.y << 3) + threadIdx.x;
        float estimateValue = blocks3D[blockPixelIndex];
        float value = estimateValue * estimateValue * coef;
        value /= (value + (sigma * sigma));
        blocks3D[blockPixelIndex] = blocks3DOrig[blockPixelIndex] * value * coef;
        atomicAdd(&wpArray[block], value);
    }
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
void pre_aggregation(double* blocks3D, float* wpArray, int size, float* estimates, int width)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    int block3DIndex = (block * 2112) + (blockIdx.z * 66);
    int xImg = (int)blocks3D[block3DIndex + 64];
    int yImg = (int)blocks3D[block3DIndex + 65];
    int xPixel = xImg + threadIdx.x;
    int yPixel = yImg + threadIdx.y;
    int estimateIndex = ((yPixel * width) + xPixel) << 1;
    int kaiserIndex = (threadIdx.y << 3) + threadIdx.x;
    if(block==1000) printf("\nblock = %d, xImg = %d, yImg = %d, xPixel = %d, yPixel = %d", block, xImg, yImg, xPixel, yPixel);
    //atomicAdd(&estimates[estimateIndex], (kaiserWindow[kaiserIndex] * wpArray[block] * blocks3D[block3DIndex + kaiserIndex]));
    //atomicAdd(&estimates[estimateIndex+1], (kaiserWindow[kaiserIndex] * wpArray[block]));
}

__global__
void aggregation(float* estimates, float* basicImage, int size)
{
    int basicImageIndex = (blockIdx.y * size) + blockIdx.x;
    int estimateIndex = (basicImageIndex << 1);
    basicImage[basicImageIndex] = int(estimates[estimateIndex]/estimates[estimateIndex+1]);
}

void BM3D::BM3D_Aggregation(bool final)
{
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, (final) ? 32 : 16);
        dim3 numThreads(8, 8);
        pre_aggregation<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.wpArray, BM3D::context.widthBlocksIntern, BM3D::context.estimates, BM3D::context.img_width); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.img_width, BM3D::context.img_height);
        dim3 numThreads(1);
        aggregation<<<numBlocks,numThreads>>>(BM3D::context.estimates, BM3D::context.basicImage, BM3D::context.img_width); 
        cudaDeviceSynchronize();   
    }
}

__global__
void inverse3D16(double* blocks3D, int size)
{
    int block3DIndex = (((blockIdx.y * size) + blockIdx.x) * 2112) + (threadIdx.y << 3) + threadIdx.x;

    double a = blocks3D[block3DIndex];
    double b = blocks3D[block3DIndex+66];
    double c = blocks3D[block3DIndex+130];
    double d = blocks3D[block3DIndex+194];
    double e = blocks3D[block3DIndex+258];
    double f = blocks3D[block3DIndex+322];
    double g = blocks3D[block3DIndex+386];
    double h = blocks3D[block3DIndex+450];
    double i = blocks3D[block3DIndex+514];
    double j = blocks3D[block3DIndex+578];
    double k = blocks3D[block3DIndex+642];
    double l = blocks3D[block3DIndex+706];
    double m = blocks3D[block3DIndex+770];
    double n = blocks3D[block3DIndex+834];
    double o = blocks3D[block3DIndex+898];
    double p = blocks3D[block3DIndex+962];

    blocks3D[block3DIndex] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p);// / 4.0;
    blocks3D[block3DIndex+66] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p);// / 4.0;
    blocks3D[block3DIndex+130] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p);// / 4.0;
    blocks3D[block3DIndex+194] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p);// / 4.0;
    blocks3D[block3DIndex+258] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p);// / 4.0;
    blocks3D[block3DIndex+322] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p);// / 4.0;
    blocks3D[block3DIndex+386] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p);// / 4.0;
    blocks3D[block3DIndex+450] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p);// / 4.0;
    blocks3D[block3DIndex+514] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p);// / 4.0;
    blocks3D[block3DIndex+578] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p);// / 4.0;
    blocks3D[block3DIndex+642] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p);// / 4.0;
    blocks3D[block3DIndex+706] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p);// / 4.0;
    blocks3D[block3DIndex+770] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p);// / 4.0;
    blocks3D[block3DIndex+834] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p);// / 4.0;
    blocks3D[block3DIndex+898] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p);// / 4.0;
    blocks3D[block3DIndex+962] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p);// / 4.0;
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

    blocks3D[block3DIndex] = ((a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p) + (a2+b2+c2+d2+e2+f2+g2+h2+i2+j2+k2+l2+m2+n2+o2+p2));// / DIVISOR;
    blocks3D[block3DIndex+64] = ((a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p) + (a2-b2+c2-d2+e2-f2+g2-h2+i2-j2+k2-l2+m2-n2+o2-p2));// / DIVISOR;
    blocks3D[block3DIndex+128] = ((a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p) + (a2+b2-c2-d2+e2+f2-g2-h2+i2+j2-k2-l2+m2+n2-o2-p2));// / DIVISOR;
    blocks3D[block3DIndex+192] = ((a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p) + (a2-b2-c2+d2+e2-f2-g2+h2+i2-j2-k2+l2+m2-n2-o2+p2));// / DIVISOR;
    blocks3D[block3DIndex+256] = ((a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p) + (a2+b2+c2+d2-e2-f2-g2-h2+i2+j2+k2+l2-m2-n2-o2-p2));// / DIVISOR;
    blocks3D[block3DIndex+320] = ((a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p) + (a2-b2+c2-d2-e2+f2-g2+h2+i2-j2+k2-l2-m2+n2-o2+p2));// / DIVISOR;
    blocks3D[block3DIndex+384] = ((a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p) + (a2+b2-c2-d2-e2-f2+g2+h2+i2+j2-k2-l2-m2-n2+o2+p2));// / DIVISOR;
    blocks3D[block3DIndex+448] = ((a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p) + (a2-b2-c2+d2-e2+f2+g2-h2+i2-j2-k2+l2-m2+n2+o2-p2));// / DIVISOR;
    blocks3D[block3DIndex+512] = ((a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p) + (a2+b2+c2+d2+e2+f2+g2+h2-i2-j2-k2-l2-m2-n2-o2-p2));// / DIVISOR;
    blocks3D[block3DIndex+576] = ((a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p) + (a2-b2+c2-d2+e2-f2+g2-h2-i2+j2-k2+l2-m2+n2-o2+p2));// / DIVISOR;
    blocks3D[block3DIndex+640] = ((a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p) + (a2+b2-c2-d2+e2+f2-g2-h2-i2-j2+k2+l2-m2-n2+o2+p2));// / DIVISOR;
    blocks3D[block3DIndex+704] = ((a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p) + (a2-b2-c2+d2+e2-f2-g2+h2-i2+j2+k2-l2-m2+n2+o2-p2));// / DIVISOR;
    blocks3D[block3DIndex+768] = ((a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p) + (a2+b2+c2+d2-e2-f2-g2-h2-i2-j2-k2-l2+m2+n2+o2+p2));// / DIVISOR;
    blocks3D[block3DIndex+832] = ((a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p) + (a2-b2+c2-d2-e2+f2-g2+h2-i2+j2-k2+l2+m2-n2+o2-p2));// / DIVISOR;
    blocks3D[block3DIndex+896] = ((a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p) + (a2+b2-c2-d2-e2-f2+g2+h2-i2-j2+k2+l2+m2+n2-o2-p2));// / DIVISOR;
    blocks3D[block3DIndex+960] = ((a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p) + (a2-b2-c2+d2-e2+f2+g2-h2-i2+j2+k2-l2+m2-n2-o2+p2));// / DIVISOR;

    blocks3D[block3DIndex+1024] = ((a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p) - a2-b2-c2-d2-e2-f2-g2-h2-i2-j2-k2-l2-m2-n2-o2-p2);// / DIVISOR;
    blocks3D[block3DIndex+1088] = ((a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p) - a2+b2-c2+d2-e2+f2-g2+h2-i2+j2-k2+l2-m2+n2-o2+p2);// / DIVISOR;
    blocks3D[block3DIndex+1152] = ((a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p) - a2-b2+c2+d2-e2-f2+g2+h2-i2-j2+k2+l2-m2-n2+o2+p2);// / DIVISOR;
    blocks3D[block3DIndex+1216] = ((a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p) - a2+b2+c2-d2-e2+f2+g2-h2-i2+j2+k2-l2-m2+n2+o2-p2);// / DIVISOR;
    blocks3D[block3DIndex+1280] = ((a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p) - a2-b2-c2-d2+e2+f2+g2+h2-i2-j2-k2-l2+m2+n2+o2+p2);// / DIVISOR;
    blocks3D[block3DIndex+1344] = ((a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p) - a2+b2-c2+d2+e2-f2+g2-h2-i2+j2-k2+l2+m2-n2+o2-p2);// / DIVISOR;
    blocks3D[block3DIndex+1408] = ((a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p) - a2-b2+c2+d2+e2+f2-g2-h2-i2-j2+k2+l2+m2+n2-o2-p2);// / DIVISOR;
    blocks3D[block3DIndex+1472] = ((a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p) - a2+b2+c2-d2+e2-f2-g2+h2-i2+j2+k2-l2+m2-n2-o2+p2);// / DIVISOR;
    blocks3D[block3DIndex+1536] = ((a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p) - a2-b2-c2-d2-e2-f2-g2-h2+i2+j2+k2+l2+m2+n2+o2+p2);// / DIVISOR;
    blocks3D[block3DIndex+1600] = ((a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p) - a2+b2-c2+d2-e2+f2-g2+h2+i2-j2+k2-l2+m2-n2+o2-p2);// / DIVISOR;
    blocks3D[block3DIndex+1664] = ((a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p) - a2-b2+c2+d2-e2-f2+g2+h2+i2+j2-k2-l2+m2+n2-o2-p2);// / DIVISOR;
    blocks3D[block3DIndex+1728] = ((a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p) - a2+b2+c2-d2-e2+f2+g2-h2+i2-j2-k2+l2+m2-n2-o2+p2);// / DIVISOR;
    blocks3D[block3DIndex+1792] = ((a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p) - a2-b2-c2-d2+e2+f2+g2+h2+i2+j2+k2+l2-m2-n2-o2-p2);// / DIVISOR;
    blocks3D[block3DIndex+1856] = ((a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p) - a2+b2-c2+d2+e2-f2+g2-h2+i2-j2+k2-l2-m2+n2-o2+p2);// / DIVISOR;
    blocks3D[block3DIndex+1920] = ((a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p) - a2-b2+c2+d2+e2+f2-g2-h2+i2+j2-k2-l2-m2-n2+o2+p2);// / DIVISOR;
    blocks3D[block3DIndex+1984] = ((a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p) - a2+b2+c2-d2+e2-f2-g2+h2+i2-j2-k2+l2-m2+n2+o2-p2);// / DIVISOR;
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
    int blockIndex = (((blockIdx.y * size) + blockIdx.x) << 11) + (blockIdx.z << 6);
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

void BM3D::BM3D_Inverse3D(bool final)
{    
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

    double DIVISOR = sqrt(8);
    if(!final)
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, 16);
        dim3 numThreads(8, 8);
        ApplyCoefBasicEstimate<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks); 
        cudaDeviceSynchronize();   
    }
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

void BM3D::BM3D_Inverse3D2(bool final)
{
    //double DIVISOR = sqrt(8);
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
    if(!final)
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, 16);
        dim3 numThreads(8, 8);
        ApplyCoefBasicEstimate<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, BM3D::context.nbSimilarBlocks); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, (final) ? 32 : 16);
        dim3 numThreads(1);
        invTransform2D<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern); 
        cudaDeviceSynchronize();   
    }
}


__global__
void HardThresholdFilter(double* blocks3D, double threshold, int size, int* nbSimilarBlocks)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    int blockPixelIndex = (block * 2112) + (blockIdx.z * 66) + (threadIdx.y << 3) + threadIdx.x;
    float coef_norm = sqrtf(float(nbSimilarBlocks[block]));
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
void BM_CalculateDistance(int* windowMap, int size, int windowSize, int width, float* image)
{
    __shared__ float distances[8];
    for(int i=0; i<16; ++i) distances[i] = 0;
    __syncthreads();

    int windowMapIndex = ((blockIdx.y * size) + blockIdx.x) * windowSize;
    int refX = windowMap[windowMapIndex];
    int refY = windowMap[windowMapIndex+1] + threadIdx.x;    

    int blockIndex = windowMapIndex + 4 + blockIdx.z * 4;
    int blockX = windowMap[blockIndex];
    int blockY = windowMap[blockIndex+1] + threadIdx.x;

    float distance = 0;
    for(int i=0; i<8; ++i)
    {
        float valueRef = image[refY * width + refX + i];
        float valueBlock = image[blockY * width + blockX + i];
        distance += (valueBlock - valueRef) * (valueBlock - valueRef);
    }
    distances[threadIdx.x] = distance;
    __syncthreads();

    if(threadIdx.x == 0)
    {
        windowMap[blockIndex+2] = int(distances[0] + distances[1] + distances[2] + distances[3] + distances[4] + distances[5] + distances[6] + distances[7]);
    }
}

__global__
void BM_DistanceThreshold(int* windowMap, int size, int windowSize, int limit)
{
    int windowMapIndex = (((blockIdx.y * size) + blockIdx.x) * windowSize) + 4 + (threadIdx.x * 4);
    if(windowMap[windowMapIndex+3] >= limit) windowMap[windowMapIndex+3] = -1;
}

__global__
void BM_Sort(int* windowMap, int size, int windowSize, int nbBlocksPerWindow)
{
    int windowMapIndex = (((blockIdx.y * size) + blockIdx.x) * windowSize) + 4;
    int blockIndex = windowMapIndex + (threadIdx.x * 4);
    int currentDistance = windowMap[blockIndex+2];
    int index = 1;
    for(int i=0; i< nbBlocksPerWindow; i++)
    {
        if(currentDistance > windowMap[windowMapIndex + i * 4 + 2]) ++index;
    }
    windowMap[blockIndex+3] = index;
}


__global__
void HadamarTransform16(double* blocks3D, int size)
{
    int block3DIndex = (blockIdx.y * size) + blockIdx.x + (threadIdx.y << 3) + threadIdx.x;
    //we can assume that the top-left corner of the basic image always has a pixel egal to 0 due to the shift 
    //of the image. 

    double a = blocks3D[block3DIndex];
    double b = blocks3D[block3DIndex+66];
    double c = blocks3D[block3DIndex+130];
    double d = blocks3D[block3DIndex+194];
    double e = blocks3D[block3DIndex+258];
    double f = blocks3D[block3DIndex+322];
    double g = blocks3D[block3DIndex+386];
    double h = blocks3D[block3DIndex+450];
    double i = blocks3D[block3DIndex+514];
    double j = blocks3D[block3DIndex+578];
    double k = blocks3D[block3DIndex+642];
    double l = blocks3D[block3DIndex+706];
    double m = blocks3D[block3DIndex+770];
    double n = blocks3D[block3DIndex+834];
    double o = blocks3D[block3DIndex+898];
    double p = blocks3D[block3DIndex+962];

    blocks3D[block3DIndex] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p);// / 4.0;
    blocks3D[block3DIndex+66] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p);// / 4.0;
    blocks3D[block3DIndex+130] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p);// / 4.0;
    blocks3D[block3DIndex+194] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p);// / 4.0;
    blocks3D[block3DIndex+258] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p);// / 4.0;
    blocks3D[block3DIndex+322] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p);// / 4.0;
    blocks3D[block3DIndex+386] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p);// / 4.0;
    blocks3D[block3DIndex+450] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p);// / 4.0;
    blocks3D[block3DIndex+514] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p);// / 4.0;
    blocks3D[block3DIndex+578] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p);// / 4.0;
    blocks3D[block3DIndex+642] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p);// / 4.0;
    blocks3D[block3DIndex+706] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p);// / 4.0;
    blocks3D[block3DIndex+770] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p);// / 4.0;
    blocks3D[block3DIndex+834] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p);// / 4.0;
    blocks3D[block3DIndex+898] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p);// / 4.0;
    blocks3D[block3DIndex+962] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p);// / 4.0;
}

__global__
void HadamarTransform32(double* blocks3D, int size)
{
    /*int index = (blockIdx.y * size) + blockIdx.x;
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

        
        blocks3D[block3DIndex] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p + a2+b2+c2+d2+e2+f2+g2+h2+i2+j2+k2+l2+m2+n2+o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+64] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p + a2-b2+c2-d2+e2-f2+g2-h2+i2-j2+k2-l2+m2-n2+o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+128] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p + a2+b2-c2-d2+e2+f2-g2-h2+i2+j2-k2-l2+m2+n2-o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+192] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p + a2-b2-c2+d2+e2-f2-g2+h2+i2-j2-k2+l2+m2-n2-o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+256] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p + a2+b2+c2+d2-e2-f2-g2-h2+i2+j2+k2+l2-m2-n2-o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+320] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p + a2-b2+c2-d2-e2+f2-g2+h2+i2-j2+k2-l2-m2+n2-o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+384] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p + a2+b2-c2-d2-e2-f2+g2+h2+i2+j2-k2-l2-m2-n2+o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+448] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p + a2-b2-c2+d2-e2+f2+g2-h2+i2-j2-k2+l2-m2+n2+o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+512] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p + a2+b2+c2+d2+e2+f2+g2+h2-i2-j2-k2-l2-m2-n2-o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+576] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p + a2-b2+c2-d2+e2-f2+g2-h2-i2+j2-k2+l2-m2+n2-o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+640] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p + a2+b2-c2-d2+e2+f2-g2-h2-i2-j2+k2+l2-m2-n2+o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+704] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p + a2-b2-c2+d2+e2-f2-g2+h2-i2+j2+k2-l2-m2+n2+o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+768] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p + a2+b2+c2+d2-e2-f2-g2-h2-i2-j2-k2-l2+m2+n2+o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+832] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p + a2-b2+c2-d2-e2+f2-g2+h2-i2+j2-k2+l2+m2-n2+o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+896] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p + a2+b2-c2-d2-e2-f2+g2+h2-i2-j2+k2+l2+m2+n2-o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+960] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p + a2-b2-c2+d2-e2+f2+g2-h2-i2+j2+k2-l2+m2-n2-o2+p2);// / DIVISOR;

        blocks3D[block3DIndex+1024] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p - a2-b2-c2-d2-e2-f2-g2-h2-i2-j2-k2-l2-m2-n2-o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+1088] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p - a2+b2-c2+d2-e2+f2-g2+h2-i2+j2-k2+l2-m2+n2-o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+1152] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p - a2-b2+c2+d2-e2-f2+g2+h2-i2-j2+k2+l2-m2-n2+o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+1216] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p - a2+b2+c2-d2-e2+f2+g2-h2-i2+j2+k2-l2-m2+n2+o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+1280] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p - a2-b2-c2-d2+e2+f2+g2+h2-i2-j2-k2-l2+m2+n2+o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+1344] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p - a2+b2-c2+d2+e2-f2+g2-h2-i2+j2-k2+l2+m2-n2+o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+1408] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p - a2-b2+c2+d2+e2+f2-g2-h2-i2-j2+k2+l2+m2+n2-o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+1472] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p - a2+b2+c2-d2+e2-f2-g2+h2-i2+j2+k2-l2+m2-n2-o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+1536] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p - a2-b2-c2-d2-e2-f2-g2-h2+i2+j2+k2+l2+m2+n2+o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+1600] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p - a2+b2-c2+d2-e2+f2-g2+h2+i2-j2+k2-l2+m2-n2+o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+1664] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p - a2-b2+c2+d2-e2-f2+g2+h2+i2+j2-k2-l2+m2+n2-o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+1728] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p - a2+b2+c2-d2-e2+f2+g2-h2+i2-j2-k2+l2+m2-n2-o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+1792] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p - a2-b2-c2-d2+e2+f2+g2+h2+i2+j2+k2+l2-m2-n2-o2-p2);// / DIVISOR;
        blocks3D[block3DIndex+1856] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p - a2+b2-c2+d2+e2-f2+g2-h2+i2-j2+k2-l2-m2+n2-o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+1920] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p - a2-b2+c2+d2+e2+f2-g2-h2+i2+j2-k2-l2-m2-n2+o2+p2);// / DIVISOR;
        blocks3D[block3DIndex+1984] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p - a2+b2+c2-d2+e2-f2-g2+h2+i2-j2-k2+l2-m2+n2+o2-p2);// / DIVISOR;
    }*/
}

void BM3D::BM3D_BlockMatching(bool final)
{
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, BM3D::context.nbBlocksPerWindow);
        dim3 numThreads(8);
        BM_CalculateDistance<<<numBlocks,numThreads>>>(BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, BM3D::context.img_width, BM3D::context.basicImage ); 
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
        dim3 numThreads(BM3D::context.nbBlocksPerWindow);
        BM_Sort<<<numBlocks,numThreads>>>(BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, BM3D::context.nbBlocksPerWindow); 
        cudaDeviceSynchronize();   
    }
}

__global__
void Create3DBlock(float* image, double* blocks3D, int* windowMap, int size, int windowSize, int width, int limit)
{
    int windowMapIndex = ((blockIdx.y * size) + blockIdx.x) * windowSize + (blockIdx.z * 4);
    if(windowMap[windowMapIndex+3] < limit)
    { 
        int block3DIndex = (((blockIdx.y * size) + blockIdx.x) * 2112) + (windowMap[windowMapIndex+3] * 66) +  (threadIdx.y << 3) + threadIdx.x;
        int xImg = windowMap[windowMapIndex] + threadIdx.x;
        int yImg = windowMap[windowMapIndex+1] + threadIdx.y;
        blocks3D[block3DIndex] = image[yImg * width + xImg];    
    }
}

__global__
void AddBlockPosition(double* blocks3D, int* windowMap, int size, int windowSize, int limit)
{
    int windowMapIndex = ((blockIdx.y * size) + blockIdx.x) * windowSize + (threadIdx.x * 4);
    if(windowMap[windowMapIndex+3] < limit)
    { 
        int block3DIndex = (((blockIdx.y * size) + blockIdx.x) * 2112) + (windowMap[windowMapIndex+3] * 66);
        blocks3D[block3DIndex+64] = windowMap[windowMapIndex];
        blocks3D[block3DIndex+65] = windowMap[windowMapIndex+1];
        if((blockIdx.y * size) + blockIdx.x == 1000) printf("\nx = %d, y = %d", blocks3D[block3DIndex+64], blocks3D[block3DIndex+65]);
    }
}

__global__
void Transform2D_row(double* blocks3D, int size, double DIVISOR)
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
    Hadamar8(inputs, DIVISOR);
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
void Transform2D_col(double* blocks3D, int size, double DIVISOR)
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
    Hadamar8(inputs, DIVISOR);
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

void BM3D::BM3D_Create3DBlocks(bool final)
{
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern, BM3D::context.nbBlocksPerWindow + 1);
        dim3 numThreads(8, 8);
        Create3DBlock<<<numBlocks,numThreads>>>(BM3D::context.basicImage, BM3D::context.blocks3D, BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, BM3D::context.img_width, (final) ? 32 : 16);
        cudaDeviceSynchronize(); 
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(BM3D::context.nbBlocksPerWindow + 1);
        AddBlockPosition<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.windowMap, BM3D::context.widthBlocksIntern, BM3D::context.windowSize, (final) ? 32 : 16);
        cudaDeviceSynchronize(); 
    }
    {
        float DIVISOR = sqrt(8);
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads((final) ? 32 : 16, 8);
        Transform2D_row<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, DIVISOR);
        cudaDeviceSynchronize(); 
        Transform2D_col<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern, DIVISOR);
        cudaDeviceSynchronize();
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(8, 8);
        if(final)
        {
            HadamarTransform32<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern);
            //HadamarTransform32<<<numBlocks,numThreads>>>(BM3D::context.blocks3DOrig, BM3D::context.widthBlocksIntern);
        }
        else
        {
            HadamarTransform16<<<numBlocks,numThreads>>>(BM3D::context.blocks3D, BM3D::context.widthBlocksIntern);
        }
        cudaDeviceSynchronize();    
    }
}

/*__global__
void Transform2D(double* blocks, int size)
{
    int blockIndex = (((blockIdx.y * size) + blockIdx.x) * 66);
    double dct[64];
    for(int y=0; y< 8; ++y)
    {
        for(int x =0; x< 8; ++x)
        {
            int pixelIndex = (y << 3) + x;
            double sum = 0;
            for(int i=0; i<8; ++i)
            {
                sum += (DCTv8matrixT[(y<<3) + i] * blocks[blockIndex + x + (i << 3)]);
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
            blocks[blockIndex + pixelIndex] = sum; //* coef_norm[pixelIndex];
        }
    }
}*/

/*void BM3D::BM3D_2DTransform2(bool final)
{
   {
        dim3 numBlocks(BM3D::context.widthBlocks, BM3D::context.widthBlocks);
        dim3 numThreads(1);
        Transform2D<<<numBlocks,numThreads>>>(BM3D::context.blocks, BM3D::context.widthBlocks); 
        cudaDeviceSynchronize();
   }
   if(!final)
   {
        dim3 numBlocks(BM3D::context.widthBlocks, BM3D::context.widthBlocks);
        dim3 numThreads(8,8);
        CopyBlocks<<<numBlocks,numThreads>>>(BM3D::context.blocks, BM3D::context.blocksOrig, BM3D::context.widthBlocks); 
        cudaDeviceSynchronize();
   }
}*/

/*void BM3D::BM3D_2DTransform(bool final)
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
}*/


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

void BM3D::BM3D_CreateWindow()
{
    {
        dim3 numBlocks(BM3D::context.img_widthOrig/8, BM3D::context.img_heightOrig/8);
        dim3 numThreads(8,8);
        ShiftImage<<<numBlocks,numThreads>>>(BM3D::context.deviceImage, BM3D::context.basicImage, BM3D::context.img_widthOrig, BM3D::context.img_width, BM3D::context.offset); 
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
