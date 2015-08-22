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

void BM3D::BM3D_Initialize(BM3D::SourceImage img, int width, int height, int pHard, int nHard)
{
    Timer::startCuda();
    printf("\n--> Execution on Tesla K40c");
    if(cudaSuccess != cudaSetDevice(0)) printf("\n\tNo device 0 available");

    int sz = 1048576 * 500;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);

    printf("\nBM3D context initialization");
    int widthOffset = width % pHard;
    int heightOffset = height % pHard;
    BM3D::context.img_widthWithBorder = width - widthOffset + nHard + 47; //+ search window
    BM3D::context.img_heightWithBorder = height - heightOffset + nHard + 47; //+ search window
    BM3D::context.nbBlocks = ((width - widthOffset) / pHard) * ((height - heightOffset) / pHard);
    BM3D::context.nbBlocksPerLine = ((width - widthOffset) / pHard);

    BM3D::context.img_width = width; 
    BM3D::context.img_height= height;
    BM3D::context.pHard = pHard;
    BM3D::context.nHard = nHard;
    BM3D::context.sourceImage = img;

    gpuErrchk(cudaMalloc(&BM3D::context.distanceArray, BM3D::context.nbBlocks * 169 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.distanceArray, 0, BM3D::context.nbBlocks * 169 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.bmIndexArray, BM3D::context.nbBlocks * 169 * sizeof(int)));
    gpuErrchk(cudaMemset(BM3D::context.bmIndexArray, 0, BM3D::context.nbBlocks * 169 * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.deviceBlocks3D, BM3D::context.nbBlocks * nHard * nHard * 169 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.deviceBlocks3D, 0, BM3D::context.nbBlocks * nHard * nHard * 169 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.deviceImage, 0, BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.deviceImage, &img[0], width * height * sizeof(float), cudaMemcpyHostToDevice));
    cudaThreadSynchronize ();

    cudaMalloc((float**)&BM3D::context.deviceBlocks, BM3D::context.nbBlocks * sizeof(float*));
    cudaMalloc((float**)&BM3D::context.deviceBlocksDCT, BM3D::context.nbBlocks * sizeof(float*));
    dim3 threadsPerBlock(1);
    dim3 numBlocks(BM3D::context.nbBlocks);
    BlocksInitialization<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks, BM3D::context.deviceBlocksDCT, nHard);
    cudaThreadSynchronize ();

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
    cudaThreadSynchronize ();

    free(cArray);
    free(cosParam1);
    free(cosParam2);

    Timer::addCuda("Cuda initialization");

    printf("\n\tBorder width (pixel) = %d", (-1 * widthOffset + nHard + 47));
    printf("\n\tBorder height (pixel) = %d", (-1 * heightOffset + nHard + 47));
    printf("\n\tImg width (border) = %d", BM3D::context.img_widthWithBorder);
    printf("\n\tImg height (border) = %d", BM3D::context.img_heightWithBorder);
    printf("\n\tNumber of blocks = %d", BM3D::context.nbBlocks);
    printf("\n\tSize blocks array = %f Mb", ((BM3D::context.nbBlocks * nHard * nHard * sizeof(float))/1024.00/1024));
    printf("\n\tBlock per line= %d", BM3D::context.nbBlocksPerLine);
    printf("\n\tSize block array= %f Mb", (BM3D::context.nbBlocks * nHard * nHard/1024.00 / 1024.00));
    printf("\n\tSize Image array= %f Mb", (BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder/1024.00 / 1024.00));  
    printf("\n\tSize 3D array = %f Mb", (BM3D::context.nbBlocks * nHard * nHard * 169 * sizeof(float) /1024.00 / 1024.00));  
    printf("\n\tSize Distance array = %f Mb", (BM3D::context.nbBlocks * 169 * sizeof(float) /1024.00 / 1024.00));  
    printf("\n\tSize BM index array = %f Mb", (BM3D::context.nbBlocks * 169 * sizeof(int) /1024.00 / 1024.00));  
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

    Timer::showResults();
}

void BM3D::BM3D_BasicEstimate()
{
    printf("\n\tBasic estimates (1 step)");
    BM3D_CreateBlocks();
    BM3D_2DDCT();
    BM3D_BlockMatching();
    BM3D_2DiDCT();
}

__global__ void blockMatching(float** dctBlocks, float* blocks3D, int* bmIndexArray, float* distanceArray, int blocksPerLine, int sizeNhard)
{
    int block = blockIdx.x;
    int compareBlock = block + (threadIdx.y * blocksPerLine) + threadIdx.x;
    
    float distance = 0.0;    
    for(int i=0; i< sizeNhard; ++i)
    {
        float diff = dctBlocks[block][i] - dctBlocks[compareBlock][i];
        distance += fabs(diff * diff);
    }
    distance /= sizeNhard;

    //int index = (block * 169) + (threadIdx.y * 13) + threadIdx.x;
    //bmIndexArray[index] = int(distance);
    //distanceArray[index] = distance;
       
    //bmIndexArray[(block * 169) + ((threadIdx.y * 13) + threadIdx.x)] = compareBlock;
    if(distance < 2500.0) 
    {
        int index3Dblock = (blockIdx.x * 64 * 169) + (((threadIdx.y * 13) + threadIdx.x) * 64);
        //memcpy(&blocks3D[index3Dblock], &dctBlocks[compareBlock][0], 64 * sizeof(float));
        //memcpy(&blocks3D[0], &dctBlocks[compareBlock][0], 64 * sizeof(float));
        for(int i=0; i< 64; ++i)
        {
            blocks3D[index3Dblock + i] = dctBlocks[compareBlock][i];
        }
    }
    /*
        int index3Dblock = (blockIdx.x * 8 * 8 * 169) + (((threadIdx.y * 13) + threadIdx.x) * 8 * 8);
        memcpy(&blocks3D[index3Dblock], &dctBlocks[compareBlock][0], 64 * sizeof(float));
        if(blockIdx.x == 100)
        {
            for(int k=0;  k<169; ++k)
            {    
                printf("\nBlock %d\n", k);     
                for(int i=0;i<8;++i)
                {   
                    for(int j=0; j< 8; ++j)
                    {   
                        int index = (blockIdx.x * 8 * 8 * 16) + (k * 8 * 8);
                        printf("%f, ",  deviceBlocks3D[index + (i * 8) + j]);
                    }
                    printf("\n");
                }
            }
        }
    }*/
    //else
    //{
        //nothing
    //}
    //distanceArray[(block * 169) + (threadIdx.y * 13) + threadIdx.x] = sum;
    //bmIndexArray[(block * 169) + (threadIdx.y * 13) + threadIdx.x] = compareBlock;
    //if(sum < 2500.0) blocksMatchingIndex[(block * 169) + (threadIdx.y * 13) + threadIdx.x] = compareBlock;
    //printf("\n block =%d, block compare = %d, value= %f", block, compareBlock, sum);    
}

__global__ void create3DArray(int* bmIndexArray)
{
    if(blockIdx.x == 100)
    {
        printf("\n");
        for(int i=0; i< 169; ++i)
        {
            printf("%d,", bmIndexArray[(blockIdx.x * 169) + i]);
        }
        printf("\n");
    }

    /*int indexBlock = blockIdx.x * 169;
    int numSimilarBlocks = 0;
    for(int i=0; i< 169; ++i)
    {
        if(distanceArray[indexBlock + i] < 2500)
        {
            int index3Dblock = (blockIdx.x * 8 * 8 * 16) + (numSimilarBlocks * 8 * 8);  
            memcpy(&deviceBlocks3D[index3Dblock], &dctBlocks[bmIndexArray[indexBlock + i]][0], 64 * sizeof(float));
            numSimilarBlocks = (numSimilarBlocks + 1) % 16;
        }
        else
        {
            //nothing   
        }
    }*/

    /*if(blockIdx.x == 100)
    {
        for(int k=0;  k<169; ++k)
        {    
            printf("\nBlock %d\n", k);     
            for(int i=0;i<8;++i)
            {   
                for(int j=0; j< 8; ++j)
                {   
                    int index = (blockIdx.x * 8 * 8 * 16) + (k * 8 * 8);
                    printf("%f, ",  deviceBlocks3D[index + (i * 8) + j]);
                }
                printf("\n");
            }
        }
    }*/
}

void BM3D::BM3D_BlockMatching()
{
    Timer::startCuda();
    dim3 threadsPerBlock(13, 13);
    dim3 numBlocks(BM3D::context.nbBlocks);
    int sizeNhard = BM3D::context.nHard * BM3D::context.nHard;
    blockMatching<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocksDCT, BM3D::context.deviceBlocks3D, BM3D::context.bmIndexArray, BM3D::context.distanceArray, BM3D::context.nbBlocksPerLine, sizeNhard);
    Timer::addCuda("Basic estimate - Block matching (HT)");
    cudaThreadSynchronize();
    Timer::startCuda();
    dim3 threadsPerBlock2(1);
    dim3 numBlocks2(BM3D::context.nbBlocks);
    create3DArray<<<numBlocks2,threadsPerBlock2>>>(BM3D::context.bmIndexArray);
    Timer::addCuda("Basic estimate - Create 3D block (HT)");
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
    Timer::startCuda();
    dim3 threadsPerBlock(1);
    dim3 numBlocks(BM3D::context.nbBlocks);
    iDCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks, BM3D::context.deviceBlocksDCT, BM3D::context.idctCosParam1, BM3D::context.idctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
    Timer::addCuda("Basic estimate - 2D iDCT (Blocks)");
}

void BM3D::BM3D_2DDCT()
{
    Timer::startCuda();
    dim3 threadsPerBlock(1);
    dim3 numBlocks(BM3D::context.nbBlocks);
    DCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks, BM3D::context.deviceBlocksDCT, BM3D::context.dctCosParam1, BM3D::context.dctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
    Timer::addCuda("Basic estimate - 2D DCT (Blocks)");
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

void BM3D::BM3D_CreateBlocks()
{
    Timer::startCuda();
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

    Timer::addCuda("Basic estimate - create blocks");
    //int size = BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder;
    //saveDeviceArray(BM3D::context.deviceImage, size, BM3D::context.img_widthWithBorder, "outputs/img.txt");
    //saveDeviceArray(BM3D::context.deviceBlocks, (BM3D::context.nbBlocks * BM3D::context.nHard * BM3D::context.nHard), (BM3D::context.nHard * BM3D::context.nHard), "outputs/blocks.txt");
}




	
