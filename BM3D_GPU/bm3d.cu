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
    /*BM3D::context.nbBlocks = ((width - widthOffset) / pHard) * ((height - heightOffset) / pHard);
    BM3D::context.nbBlocks_total = ((BM3D::context.img_widthWithBorder - nHard)/ pHard) * ((BM3D::context.img_heightWithBorder - nHard) / pHard);
    BM3D::context.nbBlocksPerLine = ((width - widthOffset) / pHard);
    BM3D::context.nbBlocksPerLine_total = ((BM3D::context.img_widthWithBorder - nHard) / pHard);*/
    BM3D::context.nbBlocks = (width / 2) * (height/2);
    BM3D::context.nbBlocks_total = BM3D::context.nbBlocks;
    BM3D::context.nbBlocksPerLine = width / 2;
    BM3D::context.nbBlocksPerLine_total = BM3D::context.nbBlocksPerLine;

    BM3D::context.img_width = width; 
    BM3D::context.img_height= height;
    BM3D::context.pHard = pHard;
    BM3D::context.nHard = nHard;
    BM3D::context.sourceImage = img;

    gpuErrchk(cudaMalloc(&BM3D::context.blocks, BM3D::context.nbBlocks * nHard * nHard * sizeof(float))); //Blocks array
    gpuErrchk(cudaMalloc(&BM3D::context.blockIndexMapping, BM3D::context.nbBlocks * 4 * sizeof(int))); //store x;y of each block
    //gpuErrchk(cudaMemset(BM3D::context.blockIndexMapping, 0, BM3D::context.nbBlocks * 2 * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.bmIndexArray, BM3D::context.nbBlocks * 16 * sizeof(int)));
    //gpuErrchk(cudaMemset(BM3D::context.bmIndexArray, 0, BM3D::context.nbBlocks * 16 * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.deviceBlocks3D, BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float)));
    //gpuErrchk(cudaMemset(BM3D::context.deviceBlocks3D, 0, BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float)));
    //gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, BM3D::context.img_width * BM3D::context.img_height * sizeof(float)));
    //gpuErrchk(cudaMemset(BM3D::context.deviceImage, 0, BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.deviceImage, &img[0], width * height * sizeof(float), cudaMemcpyHostToDevice));

    /*cudaMalloc((float**)&BM3D::context.deviceBlocks, BM3D::context.nbBlocks * sizeof(float*));
    cudaMalloc((float**)&BM3D::context.deviceBlocksDCT, BM3D::context.nbBlocks * sizeof(float*));
    dim3 threadsPerBlock(1);
    dim3 numBlocks(BM3D::context.nbBlocks);
    BlocksInitialization<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks, BM3D::context.deviceBlocksDCT, nHard);*/

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
    //cudaThreadSynchronize ();

    free(cArray);
    free(cosParam1);
    free(cosParam2);

//    Timer::addCuda("Cuda initialization");

    //printf("\n\tBorder width (pixel) = %d", (-1 * widthOffset + nHard + 47));
    //printf("\n\tBorder height (pixel) = %d", (-1 * heightOffset + nHard + 47));
    //printf("\n\tImg width (border) = %d", BM3D::context.img_widthWithBorder);
    //printf("\n\tImg height (border) = %d", BM3D::context.img_heightWithBorder);
    printf("\n\tNumber of blocks = %d", BM3D::context.nbBlocks);
    //printf("\n\tNumber of blocks (total) = %d", BM3D::context.nbBlocks_total);
    //printf("\n\tSize blocks array = %f Mb", ((BM3D::context.nbBlocks * nHard * nHard * sizeof(float))/1024.00/1024));
    printf("\n\tBlock per line= %d", BM3D::context.nbBlocksPerLine);
    //printf("\n\tBlock per line (Total)= %d", BM3D::context.nbBlocksPerLine_total);
    //printf("\n\tSize block array= %f Mb", (BM3D::context.nbBlocks * nHard * nHard/1024.00 / 1024.00));
    //printf("\n\tSize Image array= %f Mb", (BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder/1024.00 / 1024.00));  
    printf("\n\tSize Image array= %f Mb", (BM3D::context.img_width * BM3D::context.img_height * sizeof(float)/1024.00 / 1024.00));  
    printf("\n\tSize 3D array = %f Mb", (BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float) /1024.00 / 1024.00));  
    printf("\n\tSize Block Matching vectors = %f Mb", (BM3D::context.nbBlocks * 16 * sizeof(int) /1024.00 / 1024.00));  
    printf("\n\tSize Block Mapping array = %f Mb", (BM3D::context.nbBlocks * 4 * sizeof(int) /1024.00 / 1024.00));  
    printf("\n\tSize Blocks array = %f Mb", (BM3D::context.nbBlocks * nHard * nHard * sizeof(float) /1024.00 / 1024.00));  
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
    //BM3D_CreateBlocks2();
    //BM3D_CreateBlocks();
    //BM3D_2DDCT();
    //BM3D_BlockMatching();
    //BM3D_2DiDCT();
}

__global__ void blockMatching(float** dctBlocks, int* bmIndexArray, int* distanceArray, int blocksPerLine, int sizeNhard)
{
    //To Stephane    
    //!!!Number of blocks (add blocks at the end of the images, dummy blocks)
    int block = blockIdx.x;
    int compareBlock = block + (threadIdx.y * blocksPerLine) + threadIdx.x;
    if(compareBlock < 7225)
    {
        float distance = 0, diff = 0;
        /*for(int i=0; i< sizeNhard; ++i)
        {
            diff = (dctBlocks[block][i] - dctBlocks[compareBlock][i]);
            distance += (diff * diff);                
        }*/
        
        int distanceInt = int(distance);
        distanceInt = (distanceInt >> 6) - 2500;
        int offset = ((distanceInt >> 31) | -(-distanceInt >> 31)) + 1; //return 1 or -1

        int idx = (block * 507) + (39 * threadIdx.y) + (3 * threadIdx.x) + offset;
        //bmIndexArray[idx] = compareBlock;
    }
}

__global__ void create3DArray(int* bmIndexArray, int* distanceArray)
{
    /*if(blockIdx.x == 100)
    {
        printf("\n");
        for(int i=0; i< 3 *169; ++i)
        {
            int idx = (blockIdx.x * 3 * 169) +  (3 * i);
            printf("%d|%d|%d, ", bmIndexArray[idx], bmIndexArray[idx+1], bmIndexArray[idx+2]);
        }
        printf("\n");
    }*/
}

void BM3D::BM3D_BlockMatching()
{
//    Timer::startCuda();
    dim3 threadsPerBlock(13, 13);
    dim3 numBlocks(BM3D::context.nbBlocks);
    int sizeNhard = BM3D::context.nHard * BM3D::context.nHard;
    blockMatching<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocksDCT, BM3D::context.bmIndexArray, BM3D::context.distanceArray, BM3D::context.nbBlocksPerLine, sizeNhard);
//    Timer::addCuda("Basic estimate - Block matching (HT)");
    cudaThreadSynchronize();
    Timer::startCuda();
    dim3 threadsPerBlock2(1);
    dim3 numBlocks2(BM3D::context.nbBlocks);
    create3DArray<<<numBlocks2,threadsPerBlock2>>>(BM3D::context.bmIndexArray, BM3D::context.distanceArray);
//    Timer::addCuda("Basic estimate - Create 3D block (HT)");
}

__global__ void iDCT2D8x8(float** blocks, float** dctBlocks, float* dctCosParam1, float* dctCosParam2, float* cArray)
{
    int block = blockIdx.x;
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
                    sum += cArray[u] * cArray[v] * dctBlocks[block][(v * 8) + u] * dctCosParam1[(x * 8) + u] * dctCosParam2[(y * 8) + v];
                }
            }
    
            blocks[block][(y * 8) +  x] =  fabs(round(0.25 * sum));
        }
    } 


    /*if(block == 2)
    {
       printf("\n inverse DCT block 2");       
       printf("\n");
       for(int i= 0; i < 8; ++i)
       {
            for(int j=0; j< 8; ++j)
            {
                printf("%f, ", dctCosParam1[i*j]);
            }
            printf("\n");
       }
       printf("\n");
       for(int i= 0; i < 8; ++i)
       {
            for(int j=0; j< 8; ++j)
            {
                printf("%f, ", dctCosParam2[i*j]);
            }
            printf("\n");
       }
       printf("\n");
       for(int i= 0; i < 8; ++i)
       {
            for(int j=0; j< 8; ++j)
            {
                printf("%f, ", blocks[block][i*j]);
            }
            printf("\n");
       }
       printf("\n");
       for(int i= 0; i < 8; ++i)
       {
            for(int j=0; j< 8; ++j)
            {
                printf("%f, ", dctBlocks[block][i*j]);
            }
            printf("\n");
       }
    }*/
}

__global__ void DCT2D8x8(float** blocks, float** dctBlocks, float* dctCosParam1, float* dctCosParam2, float* cArray)
{
    int block = blockIdx.x;
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
                    sum += blocks[block][(y * 8) + x] * dctCosParam1[(u * 8) + x] * dctCosParam2[(v * 8) + y];
                }
            }
    
            dctBlocks[block][(v * 8) + u] =  0.25 * cArray[u] * cArray[v] * sum;
        }
    } 

    /*if(block == 2)
    {
       printf("\nDCT block 2");       
       printf("\n");
       for(int i= 0; i < 8; ++i)
       {
            for(int j=0; j< 8; ++j)
            {
                printf("%f, ", dctCosParam1[i*j]);
            }
            printf("\n");
       }
       printf("\n");
       for(int i= 0; i < 8; ++i)
       {
            for(int j=0; j< 8; ++j)
            {
                printf("%f, ", dctCosParam2[i*j]);
            }
            printf("\n");
       }
       printf("\n");
       for(int i= 0; i < 8; ++i)
       {
            printf("%f, ", cArray[i]);
       }
       printf("\n");
       for(int i= 0; i < 8; ++i)
       {
            for(int j=0; j< 8; ++j)
            {
                printf("%f, ", blocks[block][i*j]);
            }
            printf("\n");
       }
       printf("\n");
       for(int i= 0; i < 8; ++i)
       {
            for(int j=0; j< 8; ++j)
            {
                printf("%f, ", dctBlocks[block][i*j]);
            }
            printf("\n");
       }
    }*/
//       printf("\n%d block: dct[%d][0]= %f, block[%d][0] = %f", block, block, dctBlocks[block][0], block, blocks[block][0]);
}

void BM3D::BM3D_2DiDCT()
{
//    Timer::startCuda();
    dim3 threadsPerBlock(1);
    dim3 numBlocks(BM3D::context.nbBlocks);
    iDCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks, BM3D::context.deviceBlocksDCT, BM3D::context.idctCosParam1, BM3D::context.idctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
//    Timer::addCuda("Basic estimate - 2D iDCT (Blocks)");
}

void BM3D::BM3D_2DDCT()
{
//    Timer::startCuda();
    dim3 threadsPerBlock(1);
    dim3 numBlocks(BM3D::context.nbBlocks);
    DCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks, BM3D::context.deviceBlocksDCT, BM3D::context.dctCosParam1, BM3D::context.dctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
//    Timer::addCuda("Basic estimate - 2D DCT (Blocks)");
}

__global__ void CreateBlocks_Intern(float* img, float** blocks, const int blocksPerLine, const int pHard, const int nHard, const int width, int sizeBlockArray, int sizeImgArray)
{
    int block = blockIdx.x;
    int x = threadIdx.x;
    int y = threadIdx.y;
    
    int img_y = int(block/(blocksPerLine+1)) * pHard;
    int img_x = (block % blocksPerLine) * pHard;

    //int offsetBlock = (block * nHard * nHard) + (y * nHard) + x;
    int offsetBlock = (y * nHard) + x;
    int offsetImg = ((img_y + y) * width) + img_x + x;

    

    //blocks[offsetBlock] = img[offsetImg];
    blocks[block][offsetBlock] = img[offsetImg];

    //bool checkBlockOffset = (offsetBlock >= sizeBlockArray);
    //bool checkImgOffset = (offsetImg >= sizeImgArray);
    //printf("\nblock %d, line = %d, phard = %d, nhard = %d, width = %d, offsetBlock = %d, offsetImg = %d, img_x = %d, img_y = %d, x = %d, y = %d, b= %s, i=%s, blocks[block][offsetBlock] = %f ", block, blocksPerLine, pHard, nHard, width, offsetBlock, offsetImg, img_x, img_y, x, y, (checkBlockOffset) ? "OUT" : "IN", (checkImgOffset) ? "OUT" : "IN", blocks[block][offsetBlock] );
}

__global__ void CreateBlocks_Intern2(float* img, float* blocks, int* blockIndexMapping, int nHard, int width)
{
    int block = blockIdx.x;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int indexBlock = (block * nHard * nHard) + (y * nHard) + x;
    int mappingIndex = (block * 4); 
    int imgIndex = (blockIndexMapping[mappingIndex+1] * width) + blockIndexMapping[mappingIndex];
    blocks[indexBlock] = img[imgIndex];

    printf("\nindex block = %d, mapping index = %d, img index = %d, blocks[indexBlock] = %d", indexBlock, mappingIndex, imgIndex, img[imgIndex]);
}

__global__ void CreateBlocks_Zone1(int* blockIndexMapping, float* img, float* blocks, int nHard, int width)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int block = ((y * 64) + x);
    int blockIndex = block * 4;    

    int xImg = (x << 1);
    int yImg = (y << 2);
    blockIndexMapping[blockIndex] = xImg;
    blockIndexMapping[blockIndex+1] = yImg;
    blockIndexMapping[blockIndex+2] = 1;
    blockIndexMapping[blockIndex+3] = 1;

    for(int xOffset=0; xOffset< nHard; ++xOffset)
        for(int yOffset=0; yOffset< nHard; ++yOffset)
            blocks[block + (yOffset * nHard) + xOffset] = img[((yImg + yOffset) * width) + (xImg + xOffset)];
/*
    int offset1 = block;
    int offset2 = (yImg * width) + xImg;
    blocks[offset1 + 0] = img[offset2 + 0];
    blocks[offset1 + 1] = img[offset2 + 1];   
    blocks[offset1 + 2] = img[offset2 + 2];
    blocks[offset1 + 3] = img[offset2 + 3];
    blocks[offset1 + 4] = img[offset2 + 4];
    blocks[offset1 + 5] = img[offset2 + 5];   
    blocks[offset1 + 6] = img[offset2 + 6];
    blocks[offset1 + 7] = img[offset2 + 7];
    
    offset1 = block + nHard;
    offset2 = ((yImg + 1) * width) + xImg;
    blocks[offset1 + 0] = img[offset2 + 0];
    blocks[offset1 + 1] = img[offset2 + 1];   
    blocks[offset1 + 2] = img[offset2 + 2];
    blocks[offset1 + 3] = img[offset2 + 3];
    blocks[offset1 + 4] = img[offset2 + 4];
    blocks[offset1 + 5] = img[offset2 + 5];   
    blocks[offset1 + 6] = img[offset2 + 6];
    blocks[offset1 + 7] = img[offset2 + 7];

    offset1 = block + (2 * nHard);
    offset2 = ((yImg + 2) * width) + xImg;
    blocks[offset1 + 0] = img[offset2 + 0];
    blocks[offset1 + 1] = img[offset2 + 1];   
    blocks[offset1 + 2] = img[offset2 + 2];
    blocks[offset1 + 3] = img[offset2 + 3];
    blocks[offset1 + 4] = img[offset2 + 4];
    blocks[offset1 + 5] = img[offset2 + 5];   
    blocks[offset1 + 6] = img[offset2 + 6];
    blocks[offset1 + 7] = img[offset2 + 7];

    offset1 = block + (3 * nHard);
    offset2 = ((yImg + 3) * width) + xImg;
    blocks[offset1 + 0] = img[offset2 + 0];
    blocks[offset1 + 1] = img[offset2 + 1];   
    blocks[offset1 + 2] = img[offset2 + 2];
    blocks[offset1 + 3] = img[offset2 + 3];
    blocks[offset1 + 4] = img[offset2 + 4];
    blocks[offset1 + 5] = img[offset2 + 5];   
    blocks[offset1 + 6] = img[offset2 + 6];
    blocks[offset1 + 7] = img[offset2 + 7];

    offset1 = block + (4 * nHard);
    offset2 = ((yImg + 4) * width) + xImg;
    blocks[offset1 + 0] = img[offset2 + 0];
    blocks[offset1 + 1] = img[offset2 + 1];   
    blocks[offset1 + 2] = img[offset2 + 2];
    blocks[offset1 + 3] = img[offset2 + 3];
    blocks[offset1 + 4] = img[offset2 + 4];
    blocks[offset1 + 5] = img[offset2 + 5];   
    blocks[offset1 + 6] = img[offset2 + 6];
    blocks[offset1 + 7] = img[offset2 + 7];

    offset1 = block + (5 * nHard);
    offset2 = ((yImg + 5) * width) + xImg;
    blocks[offset1 + 0] = img[offset2 + 0];
    blocks[offset1 + 1] = img[offset2 + 1];   
    blocks[offset1 + 2] = img[offset2 + 2];
    blocks[offset1 + 3] = img[offset2 + 3];
    blocks[offset1 + 4] = img[offset2 + 4];
    blocks[offset1 + 5] = img[offset2 + 5];   
    blocks[offset1 + 6] = img[offset2 + 6];
    blocks[offset1 + 7] = img[offset2 + 7];

    offset1 = block + (6 * nHard);
    offset2 = ((yImg + 6) * width) + xImg;
    blocks[offset1 + 0] = img[offset2 + 0];
    blocks[offset1 + 1] = img[offset2 + 1];   
    blocks[offset1 + 2] = img[offset2 + 2];
    blocks[offset1 + 3] = img[offset2 + 3];
    blocks[offset1 + 4] = img[offset2 + 4];
    blocks[offset1 + 5] = img[offset2 + 5];   
    blocks[offset1 + 6] = img[offset2 + 6];
    blocks[offset1 + 7] = img[offset2 + 7];

    offset1 = block + (7 * nHard);
    offset2 = ((yImg + 7) * width) + xImg;
    blocks[offset1 + 0] = img[offset2 + 0];
    blocks[offset1 + 1] = img[offset2 + 1];   
    blocks[offset1 + 2] = img[offset2 + 2];
    blocks[offset1 + 3] = img[offset2 + 3];
    blocks[offset1 + 4] = img[offset2 + 4];
    blocks[offset1 + 5] = img[offset2 + 5];   
    blocks[offset1 + 6] = img[offset2 + 6];
    blocks[offset1 + 7] = img[offset2 + 7];

*/
    /*if(block == 100)
    {
        printf("\n");
        for(int i= 0; i< 64; i++)
        {
            printf("%f, ", blocks[block + i]);   
        } 
    }
    */
//    if(blockIndex == 16383)
        //printf("\n block index %d, x = %d, y = %d, x= %d, y = %d, blockIdxX = %d, blockDimX = %d, thX= %d, blockIdxY = %d, blockDimY = %d, thY = %d", blockIndex, (x << 1), (y << 1), x, y, blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y, blockDim.y, threadIdx.y); 
        //printf("\nx = %d, y = %d", x, y); 
}

__global__ void CreateBlocks_Zone2(int* blockIndexMapping, int offsetIndex, int offsetX, int offsetY)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int blockIndex = (((y * 64) + x) * 4) + offsetIndex;
    blockIndexMapping[blockIndex] = offsetX - (x << 1);
    blockIndexMapping[blockIndex+1] = (y << 1);
    blockIndexMapping[blockIndex+2] = -1;
    blockIndexMapping[blockIndex+3] = 1;
//    if(blockIndex == 16383)
        //printf("\n block index %d, x = %d, y = %d, x= %d, y = %d, blockIdxX = %d, blockDimX = %d, thX= %d, blockIdxY = %d, blockDimY = %d, thY = %d", blockIndex, offsetX - (x << 1), (y << 1), x, y, blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y, blockDim.y, threadIdx.y); 
        //printf("\nx = %d, y = %d", x, y); 
}

__global__ void CreateBlocks_Zone3(int* blockIndexMapping, int offsetIndex, int offsetX, int offsetY)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int blockIndex = (((y * 64) + x) * 4) + offsetIndex;
    blockIndexMapping[blockIndex] = (x << 1);
    blockIndexMapping[blockIndex+1] = offsetY - (y << 1);
    blockIndexMapping[blockIndex+2] = 1;
    blockIndexMapping[blockIndex+3] = -1;
//    if(blockIndex == 16383)
        //printf("\n block index %d, x = %d, y = %d, x= %d, y = %d, blockIdxX = %d, blockDimX = %d, thX= %d, blockIdxY = %d, blockDimY = %d, thY = %d", blockIndex, (x << 1), offsetY - (y << 1), x, y, blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y, blockDim.y, threadIdx.y); 
        //printf("\nx = %d, y = %d", x, y); 
}

__global__ void CreateBlocks_Zone4(int* blockIndexMapping, int offsetIndex, int offsetX, int offsetY)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int blockIndex = (((y * 64) + x) * 4) + offsetIndex;
    blockIndexMapping[blockIndex] = offsetX - (x << 1);
    blockIndexMapping[blockIndex+1] = offsetY - (y << 1);
    blockIndexMapping[blockIndex+2] = -1;
    blockIndexMapping[blockIndex+3] = -1;    
//    if(blockIndex == 16383)
        //printf("\n block index %d, x = %d, y = %d, x= %d, y = %d, blockIdxX = %d, blockDimX = %d, thX= %d, blockIdxY = %d, blockDimY = %d, thY = %d", blockIndex, offsetX - (x << 1), offsetY - (y << 1), x, y, blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y, blockDim.y, threadIdx.y); 
        //printf("\nx = %d, y = %d", x, y); 
}

void BM3D::BM3D_CreateBlockMap()
{
    dim3 threadsPerBlock(8, 8);
    int width = (BM3D::context.img_width >> 2); //64 blocks per zone
    int height = (BM3D::context.img_height >> 2);
    int numBlockX = (width >> 3);  //devision by 8, number of threads X
    int numBlockY = (height >> 3); //devision by 8, number of threads Y
    dim3 numBlocks(numBlockX, numBlockY);
    
    int index = (width * height * 4);
    CreateBlocks_Zone1<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockIndexMapping, BM3D::context.deviceImage, BM3D::context.blocks, BM3D::context.nHard, BM3D::context.img_width);
    //cudaThreadSynchronize ();  
    CreateBlocks_Zone2<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockIndexMapping, index, BM3D::context.img_width, 0);
    CreateBlocks_Zone3<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockIndexMapping, (index * 2), 0, BM3D::context.img_height );
    CreateBlocks_Zone4<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockIndexMapping, (index * 3), BM3D::context.img_width, BM3D::context.img_height);  
    cudaThreadSynchronize ();  
}

void BM3D::BM3D_CreateBlocks2()
{
    dim3 threadsPerBlock(BM3D::context.nHard, BM3D::context.nHard);
    dim3 numBlocks(BM3D::context.nbBlocks);
    CreateBlocks_Intern2<<<numBlocks,threadsPerBlock>>>( BM3D::context.deviceImage, 
                                                        BM3D::context.blocks,                                                        
                                                        BM3D::context.blockIndexMapping, 
                                                        BM3D::context.nHard,
                                                        BM3D::context.img_width);
    cudaThreadSynchronize ();
}

void BM3D::BM3D_CreateBlocks()
{
//    Timer::startCuda();
    dim3 threadsPerBlock(BM3D::context.nHard, BM3D::context.nHard);
    dim3 numBlocks(BM3D::context.nbBlocks);
    CreateBlocks_Intern<<<numBlocks,threadsPerBlock>>>( BM3D::context.deviceImage, 
                                                        BM3D::context.deviceBlocks, 
                                                        BM3D::context.nbBlocksPerLine, 
                                                        BM3D::context.pHard, 
                                                        BM3D::context.nHard, 
                                                        BM3D::context.img_widthWithBorder,
                                                        BM3D::context.nbBlocks * BM3D::context.nHard * BM3D::context.nHard,
                                                        BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder);
    cudaThreadSynchronize ();

//    Timer::addCuda("Basic estimate - create blocks");
    //int size = BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder;
    //saveDeviceArray(BM3D::context.deviceImage, size, BM3D::context.img_widthWithBorder, "outputs/img.txt");
    //saveDeviceArray(BM3D::context.deviceBlocks, (BM3D::context.nbBlocks * BM3D::context.nHard * BM3D::context.nHard), (BM3D::context.nHard * BM3D::context.nHard), "outputs/blocks.txt");
}




	
