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
    gpuErrchk(cudaMalloc(&BM3D::context.blockIndexMapping, BM3D::context.nbBlocks * 4 * sizeof(int))); //store x;y of each block (Block Map)
    //gpuErrchk(cudaMemset(BM3D::context.blockIndexMapping, 0, BM3D::context.nbBlocks * 2 * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.bmIndexArray, BM3D::context.nbBlocks * 16 * sizeof(int))); //BM vectors
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
    BM3D_2DDCT2();
    BM3D_BlockMatching2();
    BM3D_2DiDCT2();
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
    
}

__global__ void BlockMatching2(int* bmVectors, float* dctBlocks, int* blockMap, int blockPerLine, int blockSize, int* pixelMap, int nHard, int width)
{
    int xBlock = (blockIdx.x * blockDim.x) + threadIdx.x;
    int yBlock = (blockIdx.y * blockDim.y) + threadIdx.y;
    int block = (yBlock * blockPerLine) + xBlock;
    int blockIndex = (block * blockSize);
    int mappingIndex = (blockIndex << 2);
    int xImg = blockMap[mappingIndex];
    int yImg = blockMap[mappingIndex+1];
    int vX = blockMap[mappingIndex+2];
    int vY = blockMap[mappingIndex+3];
    
    int cBlockX=0, cBlockY=0, compareBlock;
    //create some threads and blocks!!! instead of a o(n3) loop --> nX & nY
    for(int nX = 0; nX < 4; ++nX)
    {
        for(int nY = 0; nY < 4; ++nY)
        {
            cBlockX = xImg + vX * (nX << 8);
            cBlockY = yImg + vY * (nY << 8);
            compareBlock = pixelMap[(cBlockY * width) + cBlockX];
            printf("\ncB = %d, b = %d", compareBlock, block);            

            float distance = 0, diff = 0;
            for(int i = 0; i< blockSize; ++i)
            {
                diff = dctBlocks[block + i] - dctBlocks[compareBlock + i];
                distance += (diff * diff);      
            }              
        }
    }
}

void BM3D::BM3D_BlockMatching2()
{
    dim3 threadsPerBlock(8,8);
    int blockXY = sqrt((BM3D::context.nbBlocks >> 6)); //divided by 64  
    dim3 numBlocks(blockXY, blockXY);
    BlockMatching2<<<numBlocks,threadsPerBlock>>>(BM3D::context.bmIndexArray, BM3D::context.dctBlocks, BM3D::context.blockIndexMapping, BM3D::context.nbBlocksPerLine, (BM3D::context.nHard * BM3D::context.nHard), BM3D::context.pixelMap, BM3D::context.nHard, BM3D::context.img_width);
    cudaThreadSynchronize ();
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

__global__ void iDCT2D8x8_3(float* blocks, float* dctBlocks, int* blockIndexMapping, int blockPerLine, int blockSize, float* dctCosParam1, float* dctCosParam2, float* cArray)
{
    int xBlock = (blockIdx.x * blockDim.x) + threadIdx.x;
    int yBlock = (blockIdx.y * blockDim.y) + threadIdx.y;
    int block = (yBlock * blockPerLine) + xBlock;
    int blockIndex = (block * blockSize);
    
        
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
                    sum += cArray[u] * cArray[v] * dctBlocks[blockIndex + (v << 8) + u] * dctCosParam1[(x << 8) + u] * dctCosParam2[(y << 8) + v];
                }
            }
    
            blocks[blockIndex + (y << 8) +  x] =  fabs(round(0.25 * sum));
        }
    }

    /*if(block == 0)
    {
        printf("\nBlock = %d\n\n", block);
        for(int i= 0; i< blockSize; i++)
        {
            printf("%f, ", blocks[blockIndex + i]);   
        } 
    }*/
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

__global__ void DCT2D8x8_3(float* blocks, float* dctBlocks, int* blockIndexMapping, int blockPerLine, int blockSize, float* dctCosParam1, float* dctCosParam2, float* cArray)
{
    int xBlock = (blockIdx.x * blockDim.x) + threadIdx.x;
    int yBlock = (blockIdx.y * blockDim.y) + threadIdx.y;
    int block = (yBlock * blockPerLine) + xBlock;
    int blockIndex = (block * blockSize);

    /*if(block == 0)
    {
        printf("\nBlock = %d\n\n", block);
        for(int i= 0; i< blockSize; i++)
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
    
            dctBlocks[blockIndex + (v << 8) + u] =  0.25 * cArray[u] * cArray[v] * sum;
        }
    } 

    /*if(block == 0)
    {
        printf("\nBlock = %d\n\n", block);
        for(int i= 0; i< blockSize; i++)
        {
            printf("%f, ", dctBlocks[blockIndex + i]);   
        } 
    }*/

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
    dim3 threadsPerBlock(1);
    dim3 numBlocks(BM3D::context.nbBlocks);
    iDCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks, BM3D::context.deviceBlocksDCT, BM3D::context.idctCosParam1, BM3D::context.idctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
}

void BM3D::BM3D_2DDCT()
{
    dim3 threadsPerBlock(1);
    dim3 numBlocks(BM3D::context.nbBlocks);
    DCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks, BM3D::context.deviceBlocksDCT, BM3D::context.dctCosParam1, BM3D::context.dctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
}

 //The AAN (Arai/Agui/Nakajima) algorithm is one of the fastest known 1D DCTs.  http://unix4lyfe.org/dct/
__global__ void DCT2D8x8_2(float* blocks, int* blockIndexMapping, int blockPerLine, int blockSize)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int block = (y * blockPerLine) + x;
    int blockIndex = (block * blockSize);
    
    if(block == 0)
    {
        printf("\nBlock = %d, block index = %d\n\n", block, blockIndex);
        for(int i= 0; i< blockSize; i++)
        {
            printf("%f, ", blocks[blockIndex + i]);   
        } 
    }

    int i;
	int rows[8][8];

	int	c1=1004 /* cos(pi/16) << 10 */,
				s1=200 /* sin(pi/16) */,
				c3=851 /* cos(3pi/16) << 10 */,
				s3=569 /* sin(3pi/16) << 10 */,
				r2c6=554 /* sqrt(2)*cos(6pi/16) << 10 */,
				r2s6=1337 /* sqrt(2)*sin(6pi/16) << 10 */,
				r2=181; /* sqrt(2) << 7*/

	int x0,x1,x2,x3,x4,x5,x6,x7,x8;

    /* transform rows */
    
	for (i=0; i<8; i++)
    {
        x0 = blocks[blockIndex + (i * 8) + 0];        
		x1 = blocks[blockIndex + (i * 8) + 1];
        x2 = blocks[blockIndex + (i * 8) + 2];
        x3 = blocks[blockIndex + (i * 8) + 3];
        x4 = blocks[blockIndex + (i * 8) + 4];
        x5 = blocks[blockIndex + (i * 8) + 5];
        x6 = blocks[blockIndex + (i * 8) + 6];
        x7 = blocks[blockIndex + (i * 8) + 7];

		/* Stage 1 */
		x8=x7+x0;
		x0-=x7;
		x7=x1+x6;
		x1-=x6;
		x6=x2+x5;
		x2-=x5;
		x5=x3+x4;
		x3-=x4;

		/* Stage 2 */
		x4=x8+x5;
		x8-=x5;
		x5=x7+x6;
		x7-=x6;
		x6=c1*(x1+x2);
		x2=(-s1-c1)*x2+x6;
		x1=(s1-c1)*x1+x6;
		x6=c3*(x0+x3);
		x3=(-s3-c3)*x3+x6;
		x0=(s3-c3)*x0+x6;

		/* Stage 3 */
		x6=x4+x5;
		x4-=x5;
		x5=r2c6*(x7+x8);
		x7=(-r2s6-r2c6)*x7+x5;
		x8=(r2s6-r2c6)*x8+x5;
		x5=x0+x2;
		x0-=x2;
		x2=x3+x1;
		x3-=x1;

		/* Stage 4 and output */
		rows[i][0]=x6;
		rows[i][4]=x4;
		rows[i][2]=x8>>10;
		rows[i][6]=x7>>10;
		rows[i][7]=(x2-x5)>>10;
		rows[i][1]=(x2+x5)>>10;
		rows[i][3]=(x3*r2)>>17;
		rows[i][5]=(x0*r2)>>17;
	}

	/* transform columns */
	for (i=0; i<8; i++)
	{
		x0 = rows[0][i];
		x1 = rows[1][i];
		x2 = rows[2][i];
		x3 = rows[3][i];
		x4 = rows[4][i];
		x5 = rows[5][i];
		x6 = rows[6][i];
		x7 = rows[7][i];

		/* Stage 1 */
		x8=x7+x0;
		x0-=x7;
		x7=x1+x6;
		x1-=x6;
		x6=x2+x5;
		x2-=x5;
		x5=x3+x4;
		x3-=x4;

		/* Stage 2 */
		x4=x8+x5;
		x8-=x5;
		x5=x7+x6;
		x7-=x6;
		x6=c1*(x1+x2);
		x2=(-s1-c1)*x2+x6;
		x1=(s1-c1)*x1+x6;
		x6=c3*(x0+x3);
		x3=(-s3-c3)*x3+x6;
		x0=(s3-c3)*x0+x6;

		/* Stage 3 */
		x6=x4+x5;
		x4-=x5;
		x5=r2c6*(x7+x8);
		x7=(-r2s6-r2c6)*x7+x5;
		x8=(r2s6-r2c6)*x8+x5;
		x5=x0+x2;
		x0-=x2;
		x2=x3+x1;
		x3-=x1;

		/* Stage 4 and output */
        /*blocks[blockIndex + (0 << 3) + i]=(float)((x6+16)>>3);
		blocks[blockIndex + (4 << 3) + i]=(float)((x4+16)>>3);
		blocks[blockIndex + (2 << 3) + i]=(float)((x8+16384)>>13);
		blocks[blockIndex + (6 << 3) + i]=(float)((x7+16384)>>13);
		blocks[blockIndex + (7 << 3) + i]=(float)((x2-x5+16384)>>13);
		blocks[blockIndex + (1 << 3) + i]=(float)((x2+x5+16384)>>13);
		blocks[blockIndex + (3 << 3) + i]=(float)(((x3>>8)*r2+8192)>>12);
		blocks[blockIndex + (5 << 3) + i]=(float)(((x0>>8)*r2+8192)>>12);*/

		blocks[blockIndex + (i * 8) + 0]=(float)((x6+16)>>3);
		blocks[blockIndex + (i * 8) + 4]=(float)((x4+16)>>3);
		blocks[blockIndex + (i * 8) + 2]=(float)((x8+16384)>>13);
		blocks[blockIndex + (i * 8) + 6]=(float)((x7+16384)>>13);
		blocks[blockIndex + (i * 8) + 7]=(float)((x2-x5+16384)>>13);
		blocks[blockIndex + (i * 8) + 1]=(float)((x2+x5+16384)>>13);
		blocks[blockIndex + (i * 8) + 3]=(float)(((x3>>8)*r2+8192)>>12);
		blocks[blockIndex + (i * 8) + 5]=(float)(((x0>>8)*r2+8192)>>12);
	}

    /*if(block == 0)
    {
        printf("\nBlock = %d\n\n", block);
        for(int i= 0; i< blockSize; i++)
        {
            printf("%f, ", blocks[blockIndex + i]);   
        } 
    }*/
}

__global__ void iDCT8x8_2(float* blocks, int* blockIndexMapping, int blockPerLine, int blockSize, float* dctCosParam1, float* dctCosParam2, float* cArray)
{
    int xBlock = (blockIdx.x * blockDim.x) + threadIdx.x;
    int yBlock = (blockIdx.y * blockDim.y) + threadIdx.y;
    int block = (yBlock * blockPerLine) + xBlock;
    int blockIndex = (block * blockSize);

    if(block == 0)
    {
        printf("\nBlock = %d\n\n", block);
        for(int i= 0; i< blockSize; i++)
        {
            printf("%f, ", blocks[blockIndex + i]);   
        } 
    }

    int u,v,x,y;

	/* iDCT */
	for (y=0; y<8; y++)
	for (x=0; x<8; x++)
	{
		double z = 0.0;

		for (v=0; v<8; v++)
		for (u=0; u<8; u++)
		{
			z += cArray[u] * cArray[v] * blocks[blockIndex + (v << 3) + u] * 
                cos((double)(2*x+1) * (double)u * M_PI/16.0) *
				cos((double)(2*y+1) * (double)v * M_PI/16.0);
				//dctCosParam1[(x << 3) + u] * dctCosParam2[(y << 3) + v];
		}

		z *= 0.25;
		//if (z > 255.0) z = 255.0;
		//if (z < 0) z = 0.0;

        blocks[blockIndex + (y << 3) + x] = z;
	}

    if(block == 0)
    {
        printf("\nBlock = %d\n\n", block);
        for(int i= 0; i< blockSize; i++)
        {
            printf("%f, ", blocks[blockIndex + i]);   
        } 
    }
}

void BM3D::BM3D_2DiDCT2()
{
    dim3 threadsPerBlock(8,8);
    int blockXY = sqrt((BM3D::context.nbBlocks >> 6)); //divided by 64  
    dim3 numBlocks(blockXY, blockXY);
    iDCT2D8x8_3<<<numBlocks,threadsPerBlock>>>(BM3D::context.blocks, BM3D::context.dctBlocks, BM3D::context.blockIndexMapping, BM3D::context.nbBlocksPerLine, (BM3D::context.nHard * BM3D::context.nHard), BM3D::context.idctCosParam1, BM3D::context.idctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
}

void BM3D::BM3D_2DDCT2()
{
    dim3 threadsPerBlock(8,8);
    int blockXY = sqrt((BM3D::context.nbBlocks >> 6)); //divided by 64  
    dim3 numBlocks(blockXY, blockXY);
    DCT2D8x8_3<<<numBlocks,threadsPerBlock>>>(BM3D::context.blocks, BM3D::context.dctBlocks, BM3D::context.blockIndexMapping, BM3D::context.nbBlocksPerLine, (BM3D::context.nHard * BM3D::context.nHard), BM3D::context.dctCosParam1, BM3D::context.dctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
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

__global__ void CreateBlocks_Zone(int* blockIndexMapping, float* img, int offsetIndex, int offsetIndexBlock, int offsetX, int offsetY, int dirX, int dirY, float* blocks, int nHard, int width, int* pixelMap)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int block = ((y << 6) + x) + offsetIndexBlock;   //mul by 64  
    int blockIndex = (block << 2) + offsetIndex;
    
    int xImg = offsetX + (dirX * (x << 1));
    int yImg = offsetY + (dirY * (y << 1));
    blockIndexMapping[blockIndex] = xImg;
    blockIndexMapping[blockIndex+1] = yImg;
    blockIndexMapping[blockIndex+2] = dirX;
    blockIndexMapping[blockIndex+3] = dirY;

    int blockNumber = 0;
    int imgPos = 0;
    for(int xOffset=0; xOffset< nHard; ++xOffset)
    {
        for(int yOffset=0; yOffset< nHard; ++yOffset)
        {
            blockNumber = block + (yOffset * nHard) + xOffset;
            imgPos = ((yImg + (dirY * yOffset)) * width) + (xImg + (dirY * xOffset));
            blocks[blockNumber] = img[imgPos];
            pixelMap[imgPos] = blockNumber;
        }
    }

    //if(blockIndex == 0)
        //printf("\n block index %d, x = %d, y = %d, x= %d, y = %d, blockIdxX = %d, blockDimX = %d, thX= %d, blockIdxY = %d, blockDimY = %d, thY = %d", blockIndex, offsetX - (x << 1), (y << 1), x, y, blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y, blockDim.y, threadIdx.y); 
        //printf("\nx = %d, y = %d", x, y); 

    /*if(block == 0 || block == 16383)
    {
        printf("\nBlock = %d\n\n", block);
        for(int i= 0; i< 64; i++)
        {
            printf("%f, ", blocks[block + i]);   
        } 
    }*/
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
    int indexBlock = (width * height);
    
    CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockIndexMapping, BM3D::context.deviceImage, 0, 0, 0, 0, 1, 1, BM3D::context.blocks, BM3D::context.nHard, BM3D::context.img_width, BM3D::context.pixelMap);
    CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockIndexMapping, BM3D::context.deviceImage, index, indexBlock, BM3D::context.img_width, 0, -1, 1, BM3D::context.blocks, BM3D::context.nHard, BM3D::context.img_width, BM3D::context.pixelMap);
    CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockIndexMapping, BM3D::context.deviceImage, (index * 2), (indexBlock * 2), 0, BM3D::context.img_height, 1, -1, BM3D::context.blocks, BM3D::context.nHard, BM3D::context.img_width, BM3D::context.pixelMap);
    CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockIndexMapping, BM3D::context.deviceImage, (index * 3), (indexBlock * 3), BM3D::context.img_width, BM3D::context.img_height, -1, -1, BM3D::context.blocks, BM3D::context.nHard, BM3D::context.img_width, BM3D::context.pixelMap);
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




	
