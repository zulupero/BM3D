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

void BM3D::BM3D_Initialize(BM3D::SourceImage img, int width, int height, int pHard, int hardLimit, bool debug)
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
    gpuErrchk(cudaMalloc(&BM3D::context.blockMap, BM3D::context.nbBlocksIntern * 100 * 10 * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.blocks, BM3D::context.nbBlocks * 66 * sizeof(double)));
    gpuErrchk(cudaMalloc(&BM3D::context.bmVectors, BM3D::context.nbBlocksIntern * 16 * sizeof(int)));
    gpuErrchk(cudaMemset(BM3D::context.bmVectors, -1, BM3D::context.nbBlocksIntern * 16 * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.blocks3D, BM3D::context.nbBlocksIntern * 16 * 64 * sizeof(double)));
    gpuErrchk(cudaMemset(BM3D::context.blocks3D, 0, BM3D::context.nbBlocksIntern * 16 * 64 * sizeof(double)));

    printf("\n\tNumber of blocks          = %d", BM3D::context.nbBlocks);
    printf("\n\tNumber of blocks (intern) = %d", BM3D::context.nbBlocksIntern);
    printf("\n\tWidth blocks (intern)     = %d", BM3D::context.widthBlocksIntern);
    printf("\n\tWidth blocks              = %d", BM3D::context.widthBlocks);
    printf("\n\tWidth                     = %d", BM3D::context.img_width);
    printf("\n\tHeight                    = %d", BM3D::context.img_height);
    printf("\n\tSize Image array          = %f Mb", (BM3D::context.img_widthOrig * BM3D::context.img_heightOrig * sizeof(float)/1024.00 / 1024.00));  
    printf("\n\tBasic Image array         = %f Mb", (BM3D::context.img_width * BM3D::context.img_height * sizeof(float)/1024.00 / 1024.00));
    printf("\n\tBlocks array              = %f Mb", (BM3D::context.nbBlocks * 66 * sizeof(double)/1024.00 / 1024.00));  
    printf("\n\tBlocks Map                = %f Mb", (BM3D::context.nbBlocks * 100 * 10 * sizeof(int)/1024.00 / 1024.00));  
    printf("\n\tBM Vectors                = %f Mb", (BM3D::context.nbBlocksIntern * 16 * sizeof(int)/1024.00 / 1024.00)); 
    printf("\n\tBlocks 3D                 = %f Mb", (BM3D::context.nbBlocksIntern * 16 * 64 * sizeof(double)/1024.00 / 1024.00));  
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
    //BM3D_ShowBlock(84,0);
    BM3D_2DTransform();
    //BM3D_ShowBlock(84,0);
    BM3D_BlockMatching();
    //BM3D_ShowDistance(84,0);
    
}

__global__
void showDistance(int x, int y, int size, int* blockMap, double* blocks, int* bmVectors)
{
    //int index = ((y * size) + x) * 1000;
    int bmVectorIndex = ((y * size) + x) << 4;
    printf("\n");
    printf("\nBM Vector = ");
    for(int i=0; i<16; ++i) printf(" %d, ", bmVectors[bmVectorIndex + i]);
    /*for(int i= 0; i < 100; i++)
    {
        //if(blockMap[index + (i * 10) + 9] < 99999999)
        {
            int blockIndex = blockMap[index + (i* 10)];
            printf("\ncmp block %d", i);
            printf("\nindex: %d", blockIndex);
            printf("\nblock x=%d, y=%d", (int)blocks[blockIndex+64], (int)blocks[blockIndex+65]);
            for(int j = 0; j< 9; ++j)
                printf("\n\t%d: distance %d", j, blockMap[index + (i * 10) + 1 + j]);
        }
    }*/
}

void BM3D::BM3D_ShowDistance(int x, int y)
{
   {
        dim3 numBlocks(1);
        dim3 numThreads(1);
        showDistance<<<numBlocks,numThreads>>>(x, y, BM3D::context.widthBlocksIntern, BM3D::context.blockMap, BM3D::context.blocks, BM3D::context.bmVectors); 
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

__global__
void BM_AddAndLimit(int* blockMap, int size, int limit)
{
    int blockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 1000) + (threadIdx.y * 10 + threadIdx.x) * 10;
    int sum = blockMap[blockMapIndex + 1] + blockMap[blockMapIndex + 2] + blockMap[blockMapIndex + 3] + blockMap[blockMapIndex + 4] + blockMap[blockMapIndex + 5] + blockMap[blockMapIndex + 6] + blockMap[blockMapIndex + 7] + blockMap[blockMapIndex + 8];
    blockMap[blockMapIndex + 9] = (sum <= limit) ? sum : 99999999;
}


__global__
void BM_Sort(int* blockMap, int size)
{
    int blockMapIndex = (((blockIdx.y * size) + blockIdx.x) * 1000);
    int currentBlockIndex = blockMapIndex + (threadIdx.y * 100) + (threadIdx.x *10);
    int currentD = blockMap[currentBlockIndex+9];
    if(currentD < 99999999)
    {
        int index = 0;
        for(int i=0; i<100; ++i)
        {
            if(currentD > blockMap[blockMapIndex + i * 10 + 9]) index++;
        }
        blockMap[currentBlockIndex+1] = index;
    }
}

__global__
void BM_CreateBmVector(int* blockMap, int* bmVectors, int size)
{
    int currentBlockIndex = (((blockIdx.y * size) + blockIdx.x) * 1000) + (threadIdx.y * 100) + (threadIdx.x *10);
    int bmVectorIndex = (((blockIdx.y * size) + blockIdx.x) << 4);
    if(blockMap[currentBlockIndex+9] < 99999999)
    {
        bmVectors[bmVectorIndex + blockMap[currentBlockIndex+1]] = blockMap[currentBlockIndex];
    }
}

void BM3D::BM3D_BlockMatching()
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
        BM_AddAndLimit<<<numBlocks,numThreads>>>(BM3D::context.blockMap, BM3D::context.widthBlocksIntern, BM3D::context.hardLimit); 
        cudaDeviceSynchronize();   
    }
    {
        dim3 numBlocks(BM3D::context.widthBlocksIntern, BM3D::context.widthBlocksIntern);
        dim3 numThreads(10, 10);
        BM_Sort<<<numBlocks,numThreads>>>(BM3D::context.blockMap, BM3D::context.widthBlocksIntern); 
        cudaDeviceSynchronize();    
        BM_CreateBmVector<<<numBlocks,numThreads>>>(BM3D::context.blockMap, BM3D::context.bmVectors, BM3D::context.widthBlocksIntern); 
        cudaDeviceSynchronize();
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
    int index = ((y * size) + x) * 66;
    printf("\n\n");
    for(int i = 0; i < 64; i++) printf("%f, ", blocks[index+i]);
    printf("\nx = %f, y = %f", blocks[index+64], blocks[index+65]);
}

void BM3D::BM3D_ShowBlock(int x, int y)
{
   {
        dim3 numBlocks(1);
        dim3 numThreads(1);
        ShowBlock<<<numBlocks,numThreads>>>(x, y, BM3D::context.widthBlocks, BM3D::context.blocks); 
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

void BM3D::BM3D_2DTransform()
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
}

__global__
void ShiftImage(float* basicImage, float* originalImage, int widthOrig, int width, int offset)
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

void BM3D::BM3D_CreateBlock()
{
    int offset = 5 * BM3D::context.pHard;
    {
        dim3 numBlocks(BM3D::context.img_widthOrig/8, BM3D::context.img_heightOrig/8);
        dim3 numThreads(8,8);
        ShiftImage<<<numBlocks,numThreads>>>(BM3D::context.basicImage, BM3D::context.deviceImage, BM3D::context.img_widthOrig, BM3D::context.img_width, offset); 
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






	
