#include <stdio.h>

#include "bm3d.h"
#include "utilities.h"
#include "timeutil.h"

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

void BM3D::BM3D_Initialize(BM3D::SourceImage img, int width, int height, int pHard, bool debug)
{
//    Timer::startCuda();
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

    int w2 = width - (width % pHard) + (10 * pHard);
    int h2 = height - (height % pHard) + (10 * pHard);
    
    BM3D::context.nbBlocksIntern = (width / pHard) * (height /pHard);
    BM3D::context.nbBlocks = (w2 / pHard) * (h2 /pHard);
    BM3D::context.widthBlocks = (w2 / pHard);
    BM3D::context.widthBlocksIntern = (width / pHard);

    w2 += 8;  //nHard = 8
    h2 += 8;  //nHard = 8
    
    BM3D::context.img_widthOrig = width; 
    BM3D::context.img_heightOrig= height;
    BM3D::context.img_width = w2; 
    BM3D::context.img_height= h2;
    BM3D::context.pHard = pHard;
    BM3D::context.sourceImage = img;

    gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, BM3D::context.img_width * BM3D::context.img_height * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.deviceImage, &img[0], width * height * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&BM3D::context.basicImage, w2 * h2 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.basicImage, 0, w2 * h2 * sizeof(float)));
    //Kaiser-window coef
    float kaiserWindow[64] = {  0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924,
                                0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989,
                                0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846,
                                0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325,
                                0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325,
                                0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846,
                                0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989,
                                0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924};

    gpuErrchk(cudaMalloc(&BM3D::context.kaiserWindowCoef, 64 * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.kaiserWindowCoef, kaiserWindow, 64 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&BM3D::context.blockMap, BM3D::context.nbBlocksIntern * 100 * 4 * sizeof(double)));
    gpuErrchk(cudaMemset(BM3D::context.blockMap, -1, BM3D::context.nbBlocksIntern * 100 * 4 * sizeof(double)));
    gpuErrchk(cudaMalloc(&BM3D::context.blocks, BM3D::context.nbBlocks * 64 * sizeof(double)));

    printf("\n\tNumber of blocks          = %d", BM3D::context.nbBlocks);
    printf("\n\tNumber of blocks (intern) = %d", BM3D::context.nbBlocksIntern);
    printf("\n\tWidth blocks (intern)     = %d", BM3D::context.widthBlocksIntern);
    printf("\n\tWidth blocks              = %d", BM3D::context.widthBlocks);
    printf("\n\tWidth                     = %d", BM3D::context.img_width);
    printf("\n\tHeight                    = %d", BM3D::context.img_height);
    printf("\n\tSize Image array          = %f Mb", (BM3D::context.img_widthOrig * BM3D::context.img_heightOrig * sizeof(float)/1024.00 / 1024.00));  
    printf("\n\tBasic Image array         = %f Mb", (BM3D::context.img_width * BM3D::context.img_height * sizeof(float)/1024.00 / 1024.00));
    printf("\n\tBlocks array              = %f Mb", (BM3D::context.nbBlocks * 64 * sizeof(double)/1024.00 / 1024.00));  
    printf("\n\tBlocks Map                = %f Mb", (BM3D::context.nbBlocks * 100 * 4 * sizeof(double)/1024.00 / 1024.00));  
}

void BM3D::BM3D_Run()
{
    printf("\n\nRun BM3D");    
    BM3D_BasicEstimate();
    
}

void BM3D::BM3D_SaveBasicImage()
{
    /*float* basicImage = (float*)malloc(BM3D::context.img_width * BM3D::context.img_height * sizeof(float));
    gpuErrchk(cudaMemcpy(&basicImage[0], BM3D::context.basicImage, BM3D::context.img_width * BM3D::context.img_height * sizeof(float), cudaMemcpyDeviceToHost));
    char* filename = "test.png";
    save_image(filename, basicImage, BM3D::context.img_width, BM3D::context.img_height, 1);*/
}

void BM3D::BM3D_BasicEstimate()
{
    printf("\n\tBasic estimates (1 step)");
    BM3D_CreateBlock();
    //BM3D_ShowBlock(30, 30);
    BM3D_2DTransform();
    //BM3D_ShowBlock(30, 30);
    //BM3D_2DTransform();
    //BM3D_ShowBlock(30, 30);
    BM3D_BlockMatching();
    BM3D_ShowDistance(30,30);
    
}

__global__
void showDistance(int x, int y, int size, double* blockMap)
{
    int index = ((y * size) + x) * 400;
    printf("\n");
    for(int i= 0; i < 100; i++)
    {
        printf("\ncmp block %d", i);
        printf("\n\tdistance %f", blockMap[index + 3]);
    }
}

void BM3D::BM3D_ShowDistance(int x, int y)
{
   {
        dim3 numBlocks(1);
        dim3 numThreads(1);
        showDistance<<<numBlocks,numThreads>>>(x, y, BM3D::context.widthBlocksIntern, BM3D::context.blockMap); 
        cudaThreadSynchronize();   
   }     
}

__device__
void addRow(double* blockRowValues, double* cmpRowValues)
{
    blockRowValues[0] = ((cmpRowValues[0] - blockRowValues[0]) * (cmpRowValues[0] - blockRowValues[0])) +
                        ((cmpRowValues[1] - blockRowValues[1]) * (cmpRowValues[1] - blockRowValues[1])) +
                        ((cmpRowValues[2] - blockRowValues[2]) * (cmpRowValues[2] - blockRowValues[2])) +
                        ((cmpRowValues[3] - blockRowValues[3]) * (cmpRowValues[3] - blockRowValues[3])) +
                        ((cmpRowValues[4] - blockRowValues[4]) * (cmpRowValues[4] - blockRowValues[4])) +
                        ((cmpRowValues[5] - blockRowValues[5]) * (cmpRowValues[5] - blockRowValues[5])) +
                        ((cmpRowValues[6] - blockRowValues[6]) * (cmpRowValues[6] - blockRowValues[6])) +
                        ((cmpRowValues[7] - blockRowValues[7]) * (cmpRowValues[7] - blockRowValues[7]));
}

__global__
void CalculateDistance(double* blockMap, double* blocks, int size)
{
    int blockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 400) + (threadIdx.y * 10 + threadIdx.x) * 4;
    int cmpBlockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 400) + 220;
    int blockIndex = (int)blockMap[blockMapIndex];// + (threadIdx.z << 3);
    int cmpBlockIndex = (int)blockMap[cmpBlockMapIndex];// + (threadIdx.z << 3);

    double blockValues[8];
    double cmpValues[8];    
    double sum = 0;
    for(int i = 0; i < 8; ++i)
    {   
        blockIndex += (i << 3); 
        blockValues[0] = blocks[blockIndex];
        blockValues[1] = blocks[blockIndex+1];
        blockValues[2] = blocks[blockIndex+2]; 
        blockValues[3] = blocks[blockIndex+3];
        blockValues[4] = blocks[blockIndex+4];
        blockValues[5] = blocks[blockIndex+5];
        blockValues[6] = blocks[blockIndex+6];
        blockValues[7] = blocks[blockIndex+7];

        cmpBlockIndex += (i << 3);
        cmpValues[0] = blocks[cmpBlockIndex];
        cmpValues[1] = blocks[cmpBlockIndex+1];
        cmpValues[2] = blocks[cmpBlockIndex+2]; 
        cmpValues[3] = blocks[cmpBlockIndex+3];
        cmpValues[4] = blocks[cmpBlockIndex+4];
        cmpValues[5] = blocks[cmpBlockIndex+5];
        cmpValues[6] = blocks[cmpBlockIndex+6];
        cmpValues[7] = blocks[cmpBlockIndex+7];
        
        addRow(blockValues, cmpValues);
        sum += blockValues[0];
    }

    blockMap[blockMapIndex +3] = sum;

    /*blockMap[blockMapIndex+3 + threadIdx.z] = ((blocks[cmpBlockIndex] - blocks[blockIndex]) * (blocks[cmpBlockIndex] - blocks[blockIndex])) +
                                ((blocks[cmpBlockIndex+1] - blocks[blockIndex]+1) * (blocks[cmpBlockIndex+1] - blocks[blockIndex+1])) +
                                ((blocks[cmpBlockIndex+2] - blocks[blockIndex]+2) * (blocks[cmpBlockIndex+2] - blocks[blockIndex+2])) +
                                ((blocks[cmpBlockIndex+3] - blocks[blockIndex]+3) * (blocks[cmpBlockIndex+3] - blocks[blockIndex+3])) +
                                ((blocks[cmpBlockIndex+4] - blocks[blockIndex]+4) * (blocks[cmpBlockIndex+4] - blocks[blockIndex+4])) +
                                ((blocks[cmpBlockIndex+5] - blocks[blockIndex]+5) * (blocks[cmpBlockIndex+5] - blocks[blockIndex+5])) +
                                ((blocks[cmpBlockIndex+6] - blocks[blockIndex]+6) * (blocks[cmpBlockIndex+6] - blocks[blockIndex+6])) +
                                ((blocks[cmpBlockIndex+7] - blocks[blockIndex]+7) * (blocks[cmpBlockIndex+7] - blocks[blockIndex+7]));*/
}


__global__
void SelectBlock(double* blockMap, int size)
{
    //int blockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 1200) + (threadIdx.y * 10 + threadIdx.x) * 12;
}


void BM3D::BM3D_BlockMatching()
{
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(10,10);
        CalculateDistance<<<numBlocks,numThreads>>>(BM3D::context.blockMap, BM3D::context.blocks, BM3D::context.widthBlocksIntern); 
        cudaThreadSynchronize ();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(10,10);
        SelectBlock<<<numBlocks,numThreads>>>(BM3D::context.blockMap, BM3D::context.widthBlocksIntern); 
        cudaThreadSynchronize ();   
    }  
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
void ShowBlock(int x, int y, int size, double* blocks)
{
    int index = ((y * size) + x) << 6;
    printf("\n\n");
    for(int i = 0; i < 64; i++) printf("%f, ", blocks[index+i]);
}

void BM3D::BM3D_ShowBlock(int x, int y)
{
   {
        dim3 numBlocks(1);
        dim3 numThreads(1);
        ShowBlock<<<numBlocks,numThreads>>>(x, y, BM3D::context.widthBlocks, BM3D::context.blocks); 
        cudaThreadSynchronize();   
   }
}

__global__
void Transform2D_row(double* blocks, int size, double DIVISOR)
{
    int blockIndex = (((blockIdx.y * size) + blockIdx.x) << 6) + (threadIdx.x << 3);
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
    int blockIndex = (((blockIdx.y * size) + blockIdx.x) << 6) + threadIdx.x;
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

void BM3D::BM3D_2DTransform()
{
   double DIVISOR = sqrt(8);
   {
        dim3 numBlocks(BM3D::context.widthBlocks, BM3D::context.widthBlocks);
        dim3 numThreads(8);
        Transform2D_row<<<numBlocks,numThreads>>>(BM3D::context.blocks, BM3D::context.widthBlocks, DIVISOR); 
        cudaThreadSynchronize();   
   }
   {
        dim3 numBlocks(BM3D::context.widthBlocks, BM3D::context.widthBlocks);
        dim3 numThreads(8);
        Transform2D_col<<<numBlocks,numThreads>>>(BM3D::context.blocks, BM3D::context.widthBlocks, DIVISOR); 
        cudaThreadSynchronize();   
   }
}

__global__
void CreateBasicImage(float* basicImage, float* originalImage, int size, int widthOrig, int width, int offset)
{
    int x = blockIdx.x * size + threadIdx.x;
    int y = blockIdx.y * size + threadIdx.y;
    int originalImageIndex = (y * widthOrig) + x;
    int basicImageIndex = ((y+offset) * width) + (x+offset);
    basicImage[basicImageIndex] = originalImage[originalImageIndex];
}

__global__
void CreateBlocks(float* basicImage, double* blocks, int size, int width, int pHard)
{
    int block = (blockIdx.y * size) + blockIdx.x;
    int imgX = blockIdx.x * pHard + threadIdx.x;
    int imgY = blockIdx.y * pHard + threadIdx.y; 
    int blockIndex = (block << 6) + (threadIdx.y << 3) + threadIdx.x;
    int imgIndex = imgY * width + imgX;
    blocks[blockIndex] = basicImage[imgIndex];  
    //if(blockIdx.x == 30 && blockIdx.y == 30) printf("\nblock = %d, imgX = %d, imgY = %d, img value = %f, block value = %f", block, imgX, imgY, basicImage[imgIndex], blocks[blockIndex]);
}

__global__
void CreateBlocksMap(double* blockMap, int size, int pHard)
{
    int blockX = (blockIdx.x + threadIdx.x) * pHard;
    int blockY = (blockIdx.y + threadIdx.y) * pHard;
    
    int blockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 400) + ((threadIdx.y * 10) + threadIdx.x) * 4;
    blockMap[blockMapIndex] = (((blockIdx.y + threadIdx.y) * size) + (blockIdx.x + threadIdx.x)) << 6; //block index
    blockMap[blockMapIndex+1] = blockX;
    blockMap[blockMapIndex+2] = blockY;
}

void BM3D::BM3D_CreateBlock()
{
    int offset = 5 * BM3D::context.pHard;
    {
        dim3 numBlocks(BM3D::context.img_widthOrig/8, BM3D::context.img_heightOrig/8);
        dim3 numThreads(8,8);
        CreateBasicImage<<<numBlocks,numThreads>>>(BM3D::context.basicImage, BM3D::context.deviceImage, BM3D::context.img_width/8, BM3D::context.img_widthOrig, BM3D::context.img_width, offset); 
        cudaThreadSynchronize ();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocks, BM3D::context.widthBlocks);
        dim3 numThreads(8,8);
        CreateBlocks<<<numBlocks,numThreads>>>(BM3D::context.basicImage, BM3D::context.blocks, BM3D::context.widthBlocks, BM3D::context.img_width, BM3D::context.pHard); 
        cudaThreadSynchronize ();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(10,10);
        CreateBlocksMap<<<numBlocks,numThreads>>>(BM3D::context.blockMap, BM3D::context.widthBlocksIntern, BM3D::context.pHard); 
        cudaThreadSynchronize ();   
    }  
}

__device__ void HadamarTransform16(float* inputs, float* outputs)
{
    double a = inputs[0];
    double b = inputs[1];
    double c = inputs[2];
    double d = inputs[3];
    double e = inputs[4];
    double f = inputs[5];
    double g = inputs[6];
    double h = inputs[7];
    double i = inputs[8];
    double j = inputs[9];
    double k = inputs[10];
    double l = inputs[11];
    double m = inputs[12];
    double n = inputs[13];
    double o = inputs[14];
    double p = inputs[15];

    outputs[0] = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p) / 4.0;
    outputs[1] = (a-b+c-d+e-f+g-h+i-j+k-l+m-n+o-p) / 4.0;
    outputs[2] = (a+b-c-d+e+f-g-h+i+j-k-l+m+n-o-p) / 4.0;
    outputs[3] = (a-b-c+d+e-f-g+h+i-j-k+l+m-n-o+p) / 4.0;
    outputs[4] = (a+b+c+d-e-f-g-h+i+j+k+l-m-n-o-p) / 4.0;
    outputs[5] = (a-b+c-d-e+f-g+h+i-j+k-l-m+n-o+p) / 4.0;
    outputs[6] = (a+b-c-d-e-f+g+h+i+j-k-l-m-n+o+p) / 4.0;
    outputs[7] = (a-b-c+d-e+f+g-h+i-j-k+l-m+n+o-p) / 4.0;
    outputs[8] = (a+b+c+d+e+f+g+h-i-j-k-l-m-n-o-p) / 4.0;
    outputs[9] = (a-b+c-d+e-f+g-h-i+j-k+l-m+n-o+p) / 4.0;
    outputs[10] = (a+b-c-d+e+f-g-h-i-j+k+l-m-n+o+p) / 4.0;
    outputs[11] = (a-b-c+d+e-f-g+h-i+j+k-l-m+n+o-p) / 4.0;
    outputs[12] = (a+b+c+d-e-f-g-h-i-j-k-l+m+n+o+p) / 4.0;
    outputs[13] = (a-b+c-d-e+f-g+h-i+j-k+l+m-n+o-p) / 4.0;
    outputs[14] = (a+b-c-d-e-f+g+h-i-j+k+l+m+n-o-p) / 4.0;
    outputs[15] = (a-b-c+d-e+f+g-h-i+j+k-l+m-n-o+p) / 4.0;
}






	
