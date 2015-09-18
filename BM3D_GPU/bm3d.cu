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

__global__ void BlocksInitialization(float** blocks, float** dctBlocks, int nHard)
{
    int block = blockIdx.x;
    blocks[block] = (float*)malloc(nHard * nHard * sizeof(float));
    memset(blocks[block], 0, nHard * nHard * sizeof(float));
    dctBlocks[block] = (float*)malloc(nHard * nHard * sizeof(float));
    memset(dctBlocks[block], 0, nHard * nHard * sizeof(float));
}

void BM3D::BM3D_PrepareDCT(float* cosParam1, float* cosParam2)
{
    for(int v = 0; v < 8; ++v)
    {
        for(int u = 0; u < 8; ++u)
        {
            for(int y= 0; y < 8; ++y)
            {
                for(int x = 0; x < 8; ++x)
                {
                    cosParam1[(u * 8) + x] = cos((((2.0 * x) + 1) * u * M_PI) / 16.0);
                    cosParam2[(v * 8) + y] = cos((((2.0 * y) + 1) * v * M_PI) / 16.0);
                }
            }
        }
    } 
}


void BM3D::BM3D_PrepareiDCT(float* cosParam1, float* cosParam2)
{
    for(int y = 0; y < 8; ++y)
    {
        for(int x = 0; x < 8; ++x)
        {
            for(int v= 0; v < 8; ++v)
            {
                for(int u = 0; u < 8; ++u)
                {
                    cosParam1[(x * 8) + u] = cos((((2.0 * x) + 1) * u * M_PI) / 16.0);
                    cosParam2[(y * 8) + v] = cos((((2.0 * y) + 1) * v * M_PI) / 16.0);
                }
            }
        }
    } 
}

void BM3D::BM3D_Initialize(BM3D::SourceImage img, int width, int height, int pHard, int nHard, bool debug)
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
    int widthOffset = width % pHard;
    int heightOffset = height % pHard;
    BM3D::context.img_widthWithBorder = width - widthOffset + 47; //+ search window
    BM3D::context.img_heightWithBorder = height - heightOffset + 47; //+ search window
    BM3D::context.nbBlocks = (width / 2) * (height /2);
    BM3D::context.nbBlocks_total = BM3D::context.nbBlocks;
    BM3D::context.nbBlocksPerLine = (width / 2);
    BM3D::context.nbBlocksPerLine_total = BM3D::context.nbBlocksPerLine;

    BM3D::context.img_width = width; 
    BM3D::context.img_height= height;
    BM3D::context.pHard = pHard;
    BM3D::context.nHard = nHard;
    BM3D::context.sourceImage = img;

    gpuErrchk(cudaMalloc(&BM3D::context.pixelMap, width * height * sizeof(int))); //Pixel Map
    gpuErrchk(cudaMalloc(&BM3D::context.blocks, BM3D::context.nbBlocks * nHard * nHard * sizeof(float))); //Blocks array
    gpuErrchk(cudaMalloc(&BM3D::context.dctBlocks, BM3D::context.nbBlocks * nHard * nHard * sizeof(float))); //DCT Blocks array
    gpuErrchk(cudaMalloc(&BM3D::context.blockMap, BM3D::context.nbBlocks * 8 * sizeof(int))); //store x;y;vX;vY;zX;zY;rX;rY of each block (Block Map)
    gpuErrchk(cudaMalloc(&BM3D::context.distanceArray, BM3D::context.nbBlocks * 169 * sizeof(int))); //distance array (for each block)
    gpuErrchk(cudaMalloc(&BM3D::context.bmVectors, BM3D::context.nbBlocks * 16 * sizeof(int))); //BM vectors
    gpuErrchk(cudaMemset(BM3D::context.bmVectors, 0, BM3D::context.nbBlocks * 16 * sizeof(int))); //BM vectors
    gpuErrchk(cudaMalloc(&BM3D::context.bmVectorsComplete, BM3D::context.nbBlocks * 169 * sizeof(int))); //BM vectors
    gpuErrchk(cudaMemset(BM3D::context.bmVectorsComplete, 0, BM3D::context.nbBlocks * 169 * sizeof(int))); //BM vectors
    gpuErrchk(cudaMalloc(&BM3D::context.deviceBlocks3D, BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.deviceBlocks3D, 0, BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, BM3D::context.img_width * BM3D::context.img_height * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.deviceImage, &img[0], width * height * sizeof(float), cudaMemcpyHostToDevice));

    float* cosParam1 = (float*)malloc(64 * sizeof(float));
    float* cosParam2 = (float*)malloc(64 * sizeof(float));
    float* icosParam1 = (float*)malloc(64 * sizeof(float));
    float* icosParam2 = (float*)malloc(64 * sizeof(float));
    float* cArray  = (float*)malloc(8 * sizeof(float));
    BM3D_PrepareDCT(cosParam1, cosParam2);
    BM3D_PrepareiDCT(icosParam1, icosParam2);
    BM3D_PrepareCArray(cArray);
    gpuErrchk(cudaMalloc(&BM3D::context.dctCosParam1, 64 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.dctCosParam2, 64 * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.dctCosParam1, cosParam1, 64 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(BM3D::context.dctCosParam2, cosParam2, 64 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&BM3D::context.idctCosParam1, 64 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.idctCosParam2, 64 * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.idctCosParam1, icosParam1, 64 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(BM3D::context.idctCosParam2, icosParam2, 64 * sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&BM3D::context.cArray, 8 * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.cArray, cArray, 8 * sizeof(float), cudaMemcpyHostToDevice));

    free(cArray);
    free(cosParam1);
    free(cosParam2);

    printf("\n\tNumber of blocks = %d", BM3D::context.nbBlocks);
    printf("\n\tBlock per line= %d", BM3D::context.nbBlocksPerLine);
    printf("\n\tSize Image array= %f Mb", (BM3D::context.img_width * BM3D::context.img_height * sizeof(float)/1024.00 / 1024.00));  
    printf("\n\tSize 3D array = %f Mb", (BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float) /1024.00 / 1024.00));  
    printf("\n\tSize Block Matching vectors = %f Mb", (BM3D::context.nbBlocks * 16 * sizeof(int) /1024.00 / 1024.00));  
    printf("\n\tSize Block Matching vectors (complete) = %f Mb", (BM3D::context.nbBlocks * 169 * sizeof(int) /1024.00 / 1024.00));      
    printf("\n\tSize Block Map array = %f Mb", (BM3D::context.nbBlocks * 8 * sizeof(int) /1024.00 / 1024.00));  
    printf("\n\tSize Blocks array = %f Mb", (BM3D::context.nbBlocks * nHard * nHard * sizeof(float) /1024.00 / 1024.00));  
    printf("\n\tSize Distance array = %f Mb", (BM3D::context.nbBlocks * 169 * sizeof(int) /1024.00 / 1024.00));
}

void BM3D::BM3D_PrepareCArray(float* cArray)
{
    float VAL = 1.0 / sqrt(2.0);
    for(int u = 0; u < 8; ++u) {
        if(u == 0) cArray[u] = VAL;
        else cArray[u] = 1.0;
    } 
}

void BM3D::BM3D_Run()
{
    printf("\n\nRun BM3D");    
    BM3D_BasicEstimate();

//    Timer::showResults();
}

void BM3D::BM3D_BasicEstimate()
{
    printf("\n\tBasic estimates (1 step)");
    BM3D_CreateBlockMap();
    BM3D_2DDCT();
    BM3D_BlockMatching();
    BM3D_HTFilter();
    //call inverse 3D transform
    BM3D_2DiDCT();
    //call aggregation method
}

__global__ void applyHTFilter(float* blocks3D, float limit, int blockSize)
{
    int block = (blockIdx.y * blockSize) + blockIdx.x;
    int blockOffset = (threadIdx.y * blockDim.x) + threadIdx.x;
    int block3DIndex = (block * 1024) + (blockOffset * 64) + threadIdx.z;
    int val = int(blocks3D[block3DIndex]) - limit;
    int mul = (((val >> 31) | -(-val >> 31)) + 1) >> 1; 
    blocks3D[block3DIndex] *= mul;
}

void BM3D::BM3D_HTFilter()
{ 
    dim3 threadsPerBlock(8,8,16); 
    int blockXY = sqrt(BM3D::context.nbBlocks); 
    dim3 numBlocks(blockXY, blockXY);
    applyHTFilter<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks3D, (2.7 * 30), blockXY); // 30 = sigma
    cudaThreadSynchronize();
}

__global__ void create3DArray(int* bmVectors, float* blocks3D, int blockSize, float* dctBlocks)
{
    int block = (blockIdx.y  * blockSize) + blockIdx.x;
    int blockVectorIndex = (block << 4);
    int block3DIndex = (block * 1024);
    int blockOffset = (threadIdx.y * blockDim.x) + threadIdx.x;
    
    int index = (bmVectors[blockVectorIndex] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex]);
    int a = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+1] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+1]);
    int b = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+2] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+2]);
    int c = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+3] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+3]);
    int d = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+4] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+4]);
    int e = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+5] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+5]);
    int f = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+6] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+6]);
    int g = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+7] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+7]);
    int h = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+8] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+8]);
    int i = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+9] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+9]);
    int j = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+10] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+10]);
    int k = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+11] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+11]);
    int l = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+12] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+12]);
    int m = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+13] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+13]);
    int n = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+14] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+14]);
    int o = dctBlocks[index];
    index = (bmVectors[blockVectorIndex+15] << 6) + blockOffset;
    //if(index >= 1048576) printf("\nblock = %d, index=%d, bmVector val=%d,", block, index, bmVectors[blockVectorIndex+15]);
    int p = dctBlocks[index];

    int x1_1 = a+b+c+d+e+f+g+h; 
    int x1_2 = i+j+k+l+m+n+o+p;
    int x1_3 = i-j-k-l-m-n-o-p;
    int x1 = x1_1 + x1_2;
    int x2_1 = a-b+c-d+e-f+g-h;
    int x2_2 = i-j+k-l+m-n+o-p; 
    int x2_3 = i+j-k+l-m+n-o+p;
    int x2 = x2_1 + x2_2;
    int x3_1 = a-b-c-d+e+f-g-h;  
    int x3_2 = i+j-k-l+m+n-o-p;
    int x3_3 = i-j+k+l-m-n+o+p;
    int x3 = x3_1 + x3_2;
    int x4_1 = a-b-c+d+e-f-g+h; 
    int x4_2 = i-j+k+l+m-n-o+p;
    int x4_3 = i+j-k-l-m+n+o-p;
    int x4 = x4_1 + x4_2;
    int x5_1 = a+b+c+d-e-f-g-h; 
    int x5_2 = i+j+k+l-m-n-o-p;
    int x5_3 = i-j-k-l+m+n+o+p;
    int x5 = x5_1 + x5_2;
    int x6_1 = a-b+c-d-e+f-g+h; 
    int x6_2 = i-j+k-l-m+n-o+p;
    int x6_3 = i+j-k+l+m-n+o-p;
    int x6 = x6_1 + x6_2;
    int x7_1 = a-b-c-d-e-f+g+h; 
    int x7_2 = i+j-k-l-m-n+o+p;
    int x7_3 = i-j+k+l+m+n-o-p;
    int x7 = x7_1 + x7_2;
    int x8_1 = a-b-c+d-e+f+g-h;
    int x8_2 = i-j+k+l-m+n+o-p;
    int x8_3 = i+j-k-l+m-n-o+p;
    int x8 = x8_1 + x8_2;
    
    int x9 = x1_1 - x1_3;
    int x10 = x2_1 - x2_3;
    int x11 = x3_1 - x3_3;
    int x12 = x4_1 - x4_3;
    int x13 = x5_1 - x5_3;
    int x14 = x6_1 - x6_3;
    int x15 = x7_1 - x7_3;
    int x16 = x8_1 - x8_3;

    block3DIndex += blockOffset;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block); 
    blocks3D[block3DIndex] = (x1 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x2 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x3 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x4 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x5 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x6 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x7 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x8 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x9 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x10 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x11 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x12 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x13 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x14 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x15 >> 6);
    block3DIndex += 64;
    //if(block3DIndex >= 16777216) printf("\nblock3DIndex = %d, block = %d,", block3DIndex, block);
    blocks3D[block3DIndex] = (x16 >> 6);
}

__global__ void CalculateDistances(int* distanceArray, int* blockMap, float* dctBlocks, int blockSize, int* bmVectorsComplete)
{
    int block = (blockIdx.y * blockSize) + blockIdx.x;
    int blockIndex = (block << 6);
    int blockMapIndex = (block << 3);
    int rx = blockMap[blockMapIndex+6];
    int ry = blockMap[blockMapIndex+7];
    int compareBlock = ((blockIdx.y + (ry * threadIdx.y)) * blockSize) + (rx * threadIdx.x);
    int compareBlockIndex = (compareBlock << 6);
    int distanceArrayIndex = (block * 169) + (threadIdx.y * 13) + threadIdx.x;
    float distance = 0, diff = 0;
 
    bmVectorsComplete[distanceArrayIndex] = compareBlock;
    //TODO: Perf bottleneck!!!
    for(int i =0; i< 64; ++i)
    {
        diff =  dctBlocks[compareBlockIndex + i] - dctBlocks[blockIndex + i];
        distance = distance + (diff * diff);
    }
    int d = int(distance);
    d = (d >> 6); //divide by nHardxnHard (8x8)
    distanceArray[distanceArrayIndex] = d;
}

__global__ void ApplyDistanceThreshold(int* distanceArray, int limit, int blockSize, int* bmVectorsComplete)
{
    int distanceArrayIndex = (((blockIdx.y * blockSize) + blockIdx.x) * 169) + (threadIdx.y * 13) + threadIdx.x;
    int distance = distanceArray[distanceArrayIndex] - limit;
    int mul = (-((distance >> 31) | -(-distance >> 31)) + 1) >> 1;
    bmVectorsComplete[distanceArrayIndex] *= mul;
    //printf("\ndistance = %d, mul = %d, distance = %d ", distance, mul, distanceArray[distanceArrayIndex]);
}

__global__ void ShrinkBmVectors(int* bmVectorsComplete, int* bmVectors)
{
    int block = (((blockIdx.y * 16) + blockIdx.x) * 64) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int bmVectorCIndex = block * 169;
    int bmVectorIndex = block * 16;
    int i = 0;
    for(int index = 0; index < 169; ++index)
    {
        int val = bmVectorsComplete[bmVectorCIndex + index];
        if( val > 0)
        {
            bmVectors[bmVectorIndex + i] = val;
            ++i;
            if(i >= 16) break;
        }
    }

    /*if(block == 16383)
    {
        printf("\n");
        for(int i =0; i< 16; ++i)
        {
            printf("%d, ", bmVectors[bmVectorIndex + i]); 
        }
        printf("\n");
        for(int i =0; i< 169; ++i)
        {
            printf("%d, ", bmVectorsComplete[bmVectorCIndex + i]); 
        }
    }*/
}

void BM3D::BM3D_BlockMatching()
{
    {    
        dim3 threadsPerBlock(13,13); //sliding window of 13by13 patches of 8x8 pixels
        int blockXY = sqrt(BM3D::context.nbBlocks); 
        dim3 numBlocks(blockXY, blockXY);
        CalculateDistances<<<numBlocks,threadsPerBlock>>>(BM3D::context.distanceArray, BM3D::context.blockMap, BM3D::context.dctBlocks, blockXY, BM3D::context.bmVectorsComplete );
        cudaThreadSynchronize();

        ApplyDistanceThreshold<<<numBlocks,threadsPerBlock>>>(BM3D::context.distanceArray, 2500, blockXY, BM3D::context.bmVectorsComplete);
        cudaThreadSynchronize();
    }
    
    {
        dim3 threadsPerBlock(8,8); 
        dim3 numBlocks(16, 16); //image 256x256 -> 16384 blocks (blocks: 16x16 / threads: 8x8)
        ShrinkBmVectors<<<numBlocks,threadsPerBlock>>>(BM3D::context.bmVectorsComplete, BM3D::context.bmVectors);
        cudaThreadSynchronize();    
    }
    
    {
        dim3 threadsPerBlock(8,8); 
        int blockXY = sqrt(BM3D::context.nbBlocks); 
        dim3 numBlocks(blockXY, blockXY);
        create3DArray<<<numBlocks,threadsPerBlock>>>(BM3D::context.bmVectors, BM3D::context.deviceBlocks3D, blockXY, BM3D::context.dctBlocks);
        cudaThreadSynchronize();
    }
}

__global__ void iDCT2D8x8(float* blocks, float* dctBlocks, float* dctCosParam1, float* dctCosParam2, float* cArray)
{
    int size = blockDim.x * blockDim.y;
    int block = (((blockIdx.y << 4 ) + blockIdx.x)  * size) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //printf("\nb= %d, bY = %d, bX = %d, tY = %d, tX = %d", block, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x);    
    int blockIndex = block * size;
    
        
    int x, y, u, v;
    for(y = 0; y < 8; ++y)
    {
        for(x = 0; x < 8; ++x)
        {
            float sum = 0.0;
            for(v = 0; v < 8; ++v)
            {
                for(u = 0; u < 8; ++u)
                {
                    sum += cArray[u] * cArray[v] * dctBlocks[blockIndex + (v << 3) + u] * dctCosParam1[(x << 3) + u] * dctCosParam2[(y << 3) + v];
                }
            }
    
            blocks[blockIndex + (y << 3) +  x] =  fabs(round(0.25 * sum));
        }
    }

    /*if(block == 16383)
    {
        printf("\nBlock = %d\n\n", block);
        for(int i= 0; i< size; i++)
        {
            printf("%f, ", blocks[blockIndex + i]);   
        } 
    }*/
}


__global__ void DCT2D8x8(float* blocks, float* dctBlocks, float* dctCosParam1, float* dctCosParam2, float* cArray)
{
    int size = blockDim.x * blockDim.y;
    int block = (((blockIdx.y << 4 ) + blockIdx.x)  * size) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //printf("\nb= %d, bY = %d, bX = %d, tY = %d, tX = %d", block, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x);    
    int blockIndex = block * size;

    /*if(block == 16383)
    {
        printf("\nDBlock = %d\n", block);
        for(int i= 0; i< size; i++)
        {
            printf("%f, ", blocks[blockIndex + i]);   
        } 
    }*/

    int x, y, u, v;
    for(v= 0; v < 8; ++v)
    {
        for(u = 0; u < 8; ++u)
        {
            float sum = 0.0;
            for(y = 0; y < 8; ++y)
            {
                for(x = 0; x< 8; ++x)
                {
                    sum += blocks[blockIndex + (y << 3) + x] * dctCosParam1[(u << 3) + x] * dctCosParam2[(v << 3) + y];
                }
            }
    
            dctBlocks[blockIndex + (v << 3) + u] =  0.25 * cArray[u] * cArray[v] * sum;
        }
    } 

    /*if(block == 16383)
    {
        printf("\nDBlock = %d\n", block);
        for(int i= 0; i< size; i++)
        {
            printf("%f, ", dctBlocks[blockIndex + i]);   
        } 
    }*/
}

void BM3D::BM3D_2DiDCT()
{
    dim3 threadsPerBlock(8,8);
    int blockXY = sqrt(BM3D::context.nbBlocks >> 6); 
    dim3 numBlocks(blockXY, blockXY);
    iDCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.blocks, BM3D::context.dctBlocks, BM3D::context.idctCosParam1, BM3D::context.idctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
}

void BM3D::BM3D_2DDCT()
{
    dim3 threadsPerBlock(8,8);
    int blockXY = sqrt(BM3D::context.nbBlocks >> 6); 
    dim3 numBlocks(blockXY, blockXY);
    DCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.blocks, BM3D::context.dctBlocks, BM3D::context.dctCosParam1, BM3D::context.dctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
}

__global__ void CreateBlocks_Zone(int* blockMap, int blockIndexOffset, int offsetX, int offsetY, int vX, int vY)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int block = ((y << 6) + x) + blockIndexOffset;   //mul by 64  
    int blockMapIndex = (block << 3);
    
    int posX = x << 1;
    int posY = y << 1;    
    int xImg = offsetX + (vX * posX);
    int yImg = offsetY + (vY * posY);
    blockMap[blockMapIndex] = xImg;
    blockMap[blockMapIndex+1] = yImg;
    blockMap[blockMapIndex+2] = vX;
    blockMap[blockMapIndex+3] = vY;
    blockMap[blockMapIndex+4] = x;
    blockMap[blockMapIndex+5] = y;
    int x2 = (x - 32) + 1;
    int y2 = (y - 32) + 1; 
    int rx = -((x2 >> 31) | -(-x2 >> 31));    
    int ry = -((y2 >> 31) | -(-y2 >> 31));
    blockMap[blockMapIndex+6] = rx;
    blockMap[blockMapIndex+7] = ry;

    //if(blockMapIndex >= 65536)
    //printf("\nblock = %d, xImg = %d, yImg = %d, x = %d, y = %d, vx = %d, vy = %d, blockMapIndex = %d,rx =%d, ry=%d", block, blockMap[blockMapIndex], blockMap[blockMapIndex+1], x, y, blockMap[blockMapIndex+2], blockMap[blockMapIndex+3], blockMapIndex, rx, ry); 
}

__global__ void CreateBlocks(float* img, float* blocks, int* blockMap, int width)
{
    int block = (blockIdx.y << 7) + blockIdx.x;
    int blockMapIndex = block << 3;
    int blockIndex = block << 6;
   
    int xImg = blockMap[blockMapIndex];
    int yImg = blockMap[blockMapIndex+1];
    int vX = blockMap[blockMapIndex+2];
    int vY = blockMap[blockMapIndex+3];

    int blockPixelIndex = blockIndex + (threadIdx.y * blockDim.x) + threadIdx.x;
    int xPos = xImg + (vX * threadIdx.x);
    int yPos = yImg + (vY * threadIdx.y);
    int imgIndex = (yPos * width) + xPos;
    //if(blockPixelIndex >= 1048576 || imgIndex >= 65536)
        //printf("\nblock = %d, blockMapIndex = %d, imgIndex = %d, pixelIndex = %d, xPos = %d, yPos = %d, vx = %d, vy = %d", block, blockMapIndex, imgIndex, blockPixelIndex, xImg, yImg, vX, vY);
    blocks[blockPixelIndex] = img[imgIndex];
}

void BM3D::BM3D_CreateBlockMap()
{
    dim3 threadsPerBlock(8, 8);
    int width = (BM3D::context.img_width >> 2); //64 blocks per zone
    int numBlockXY = (width >> 3);  //devision by 8, number of blocks X
    dim3 numBlocks(numBlockXY, numBlockXY); //(8x8) (8x8) = 4096 blocks for each zone (1,2,3,4)
    int blockIndexOffset = width * width;
    
    CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockMap, 0, 0, 0, 1, 1);
    cudaThreadSynchronize ();     
    CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockMap, blockIndexOffset, (BM3D::context.img_width - 1), 0, -1, 1);
    cudaThreadSynchronize (); 
    CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockMap, (blockIndexOffset * 2), 0, (BM3D::context.img_height -1 ), 1, -1);
    cudaThreadSynchronize (); 
    CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockMap, (blockIndexOffset * 3), (BM3D::context.img_width-1), (BM3D::context.img_height-1), -1, -1);
    cudaThreadSynchronize (); 
        
    int blockXY = sqrt(BM3D::context.nbBlocks);
    dim3 numBlocks2(blockXY, blockXY);
    CreateBlocks<<<numBlocks2, threadsPerBlock>>>(BM3D::context.deviceImage, BM3D::context.blocks, BM3D::context.blockMap, BM3D::context.img_width); 
    cudaThreadSynchronize ();
}




	
