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
    BM3D::context.nbBlocks = (width / 2) * (height /2);
    BM3D::context.nbBlocks_total = BM3D::context.nbBlocks;
    BM3D::context.nbBlocksPerLine = (width / 2);
    BM3D::context.nbBlocksPerLine_total = BM3D::context.nbBlocksPerLine;

    BM3D::context.img_width = width; 
    BM3D::context.img_height= height;
    BM3D::context.pHard = pHard;
    BM3D::context.nHard = nHard;
    BM3D::context.sourceImage = img;

    gpuErrchk(cudaMalloc(&BM3D::context.pixelMap, width * height * 64 * sizeof(int))); //Pixel Map
    gpuErrchk(cudaMemset(BM3D::context.pixelMap, -1, width * height * 64 * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.pixelMapIndex, width * height * sizeof(int))); //Pixel Map Index
    gpuErrchk(cudaMemset(BM3D::context.pixelMapIndex, 0, width * height * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.blocks, BM3D::context.nbBlocks * nHard * nHard * sizeof(float))); //Blocks array
    gpuErrchk(cudaMalloc(&BM3D::context.wpArray, BM3D::context.nbBlocks * sizeof(float))); //WP array (for each block)
    gpuErrchk(cudaMalloc(&BM3D::context.npArray, BM3D::context.nbBlocks * sizeof(int))); //NP array (for each block)
    gpuErrchk(cudaMalloc(&BM3D::context.dctBlocks, BM3D::context.nbBlocks * nHard * nHard * sizeof(float))); //DCT Blocks array
    gpuErrchk(cudaMalloc(&BM3D::context.blockMap, BM3D::context.nbBlocks * 8 * sizeof(int))); //store x;y;vX;vY;zX;zY;rX;rY of each block (Block Map)
    gpuErrchk(cudaMalloc(&BM3D::context.distanceArray, BM3D::context.nbBlocks * 169 * sizeof(int))); //distance array (for each block)
    gpuErrchk(cudaMalloc(&BM3D::context.blockGroupMap, BM3D::context.nbBlocks * 128 * sizeof(int))); //BlockGroup Map
    gpuErrchk(cudaMemset(BM3D::context.blockGroupMap, -1, BM3D::context.nbBlocks * 128 * sizeof(int))); 
    gpuErrchk(cudaMalloc(&BM3D::context.blockGroupIndex, BM3D::context.nbBlocks * sizeof(int))); //BlockGroup Map Index
    gpuErrchk(cudaMemset(BM3D::context.blockGroupIndex, 0, BM3D::context.nbBlocks * sizeof(int))); 
    gpuErrchk(cudaMalloc(&BM3D::context.bmVectors, BM3D::context.nbBlocks * 16 * sizeof(int))); //BM vectors
    gpuErrchk(cudaMemset(BM3D::context.bmVectors, -1, BM3D::context.nbBlocks * 16 * sizeof(int))); 
    gpuErrchk(cudaMalloc(&BM3D::context.bmVectorsComplete, BM3D::context.nbBlocks * 169 * sizeof(int))); //BM vectors
    gpuErrchk(cudaMemset(BM3D::context.bmVectorsComplete, -1, BM3D::context.nbBlocks * 169 * sizeof(int))); //BM vectors
    gpuErrchk(cudaMalloc(&BM3D::context.deviceBlocks3D, BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.deviceBlocks3D, 0, BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.finalBlocks3D, BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.finalBlocks3D, 0, BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, BM3D::context.img_width * BM3D::context.img_height * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.deviceImage, &img[0], width * height * sizeof(float), cudaMemcpyHostToDevice));

    //TODO: replace by Hadamard8 function
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
    printf("\n\tSize 3D array (final) = %f Mb", (BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float) /1024.00 / 1024.00));      
    printf("\n\tSize Block Matching vectors = %f Mb", (BM3D::context.nbBlocks * 16 * sizeof(int) /1024.00 / 1024.00));  
    printf("\n\tSize Block Matching vectors (complete) = %f Mb", (BM3D::context.nbBlocks * 169 * sizeof(int) /1024.00 / 1024.00));      
    printf("\n\tSize Block Map array = %f Mb", (BM3D::context.nbBlocks * 8 * sizeof(int) /1024.00 / 1024.00));  
    printf("\n\tSize Blocks array = %f Mb", (BM3D::context.nbBlocks * nHard * nHard * sizeof(float) /1024.00 / 1024.00));  
    printf("\n\tSize Distance array = %f Mb", (BM3D::context.nbBlocks * 169 * sizeof(int) /1024.00 / 1024.00));
    printf("\n\tSize Pixel map= %f Mb", (width * height * 64 * sizeof(int) /1024.00 / 1024.00));
    printf("\n\tSize Pixel map Index= %f Mb", (width * height * sizeof(int) /1024.00 / 1024.00));    
    printf("\n\tSize Block group map = %f Mb", (BM3D::context.nbBlocks * 128 * sizeof(int) /1024.00 / 1024.00));
    printf("\n\tSize Block group Index = %f Mb", (BM3D::context.nbBlocks * sizeof(int) /1024.00 / 1024.00));
    printf("\n\tSize WP array = %f Mb", (BM3D::context.nbBlocks * sizeof(float) /1024.00 / 1024.00));
    printf("\n\tSize NP array = %f Mb", (BM3D::context.nbBlocks * sizeof(int) /1024.00 / 1024.00));
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
    BM3D_CalculateWPArray();
    BM3D_InverseTransform();
    BM3D_Aggregation();
}

void BM3D::BM3D_Aggregation()
{
    //TODO
}

__global__ void calculateNPArray(float* blocks3D, int blockSize, int* npArray)
{
    int block = (blockIdx.y * blockSize) + blockIdx.x;    
    int block3DIndex = (block << 10) + (threadIdx.z << 6) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if(blocks3D[block3DIndex] > 0) atomicAdd(&npArray[block], 1);
}

__global__ void calculateWPArray(int* npArray, float* wpArray)
{
    int block = (((blockIdx.y << 5) + blockIdx.x) << 4) + (threadIdx.y << 2) + threadIdx.x;
    wpArray[block] = (npArray[block] >= 1) ? (1.0 / npArray[block]) : 1; 
}

void BM3D::BM3D_CalculateWPArray()
{
    {
        dim3 threadsPerBlock(8,8,16); 
        int blockXY = sqrt(BM3D::context.nbBlocks); 
        dim3 numBlocks(blockXY, blockXY);
        calculateNPArray<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks3D, blockXY, BM3D::context.npArray); 
        cudaThreadSynchronize();    
    }
    {
        dim3 threadsPerBlock(4,4); 
        dim3 numBlocks(32, 32);
        calculateWPArray<<<numBlocks,threadsPerBlock>>>(BM3D::context.npArray, BM3D::context.wpArray); 
        cudaThreadSynchronize();    
    }
}

__global__ void applyHTFilter(float* blocks3D, int limit, int blockSize)
{
    int block = (blockIdx.y << 7) + blockIdx.x;
    int blockPixel = (threadIdx.y << 3) + threadIdx.x;
    int block3DIndex = (block << 10) + (threadIdx.z << 6) + blockPixel;
    int val = int(blocks3D[block3DIndex]);
    int mul = ((((val -limit) >> 31) | -(-(val-limit) >> 31)) + 1) >> 1; 
    blocks3D[block3DIndex] = val * mul;
}

void BM3D::BM3D_HTFilter()
{ 
    dim3 threadsPerBlock(8,8,16); 
    int blockXY = sqrt(BM3D::context.nbBlocks); 
    dim3 numBlocks(blockXY, blockXY);
    applyHTFilter<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks3D, 90, blockXY); // limmit = 3 * 30 (sigma) , BM3D IPOL = 2.7 * sigma
    cudaThreadSynchronize();
}

__device__ void HadamarTransform16(float* inputs, float* outputs)
{
    float a = inputs[0];
    float b = inputs[1];
    float c = inputs[2];
    float d = inputs[3];
    float e = inputs[4];
    float f = inputs[5];
    float g = inputs[6];
    float h = inputs[7];
    float i = inputs[8];
    float j = inputs[9];
    float k = inputs[10];
    float l = inputs[11];
    float m = inputs[12];
    float n = inputs[13];
    float o = inputs[14];
    float p = inputs[15];

    float x1_1 = a+b+c+d+e+f+g+h; 
    float x1_2 = i+j+k+l+m+n+o+p;
    float x1_3 = i-j-k-l-m-n-o-p;
    float x1 = x1_1 + x1_2;
    float x2_1 = a-b+c-d+e-f+g-h;
    float x2_2 = i-j+k-l+m-n+o-p; 
    float x2_3 = i+j-k+l-m+n-o+p;
    float x2 = x2_1 + x2_2;
    float x3_1 = a+b-c-d+e+f-g-h;  
    float x3_2 = i+j-k-l+m+n-o-p;
    float x3_3 = i-j+k+l-m-n+o+p;
    float x3 = x3_1 + x3_2;
    float x4_1 = a-b-c+d+e-f-g+h; 
    float x4_2 = i-j-k+l+m-n-o+p;
    float x4_3 = i+j+k-l-m+n+o-p;
    float x4 = x4_1 + x4_2;
    float x5_1 = a+b+c+d-e-f-g-h; 
    float x5_2 = i+j+k+l-m-n-o-p;
    float x5_3 = i-j-k-l+m+n+o+p;
    float x5 = x5_1 + x5_2;
    float x6_1 = a-b+c-d-e+f-g+h; 
    float x6_2 = i-j+k-l-m+n-o+p;
    float x6_3 = i+j-k+l+m-n+o-p;
    float x6 = x6_1 + x6_2;
    float x7_1 = a+b-c-d-e-f+g+h; 
    float x7_2 = i+j-k-l-m-n+o+p;
    float x7_3 = i-j+k+l+m+n-o-p;
    float x7 = x7_1 + x7_2;
    float x8_1 = a-b-c+d-e+f+g-h;
    float x8_2 = i-j-k+l-m+n+o-p;
    float x8_3 = i+j+k-l+m-n-o+p;
    float x8 = x8_1 + x8_2;
    
    float x9 = x1_1 - x1_3;
    float x10 = x2_1 - x2_3;
    float x11 = x3_1 - x3_3;
    float x12 = x4_1 - x4_3;
    float x13 = x5_1 - x5_3;
    float x14 = x6_1 - x6_3;
    float x15 = x7_1 - x7_3;
    float x16 = x8_1 - x8_3;

    outputs[0] = (x1 / 4);
    outputs[1] = (x2 / 4);
    outputs[2] = (x3 / 4);
    outputs[3] = (x4 / 4);
    outputs[4] = (x5 / 4);
    outputs[5] = (x6 / 4);
    outputs[6] = (x7 / 4);
    outputs[7] = (x8 / 4);
    outputs[8] = (x9 / 4);
    outputs[9] = (x10 / 4);
    outputs[10] = (x11 / 4);
    outputs[11] = (x12 / 4);
    outputs[12] = (x13 / 4);
    outputs[13] = (x14 / 4);
    outputs[14] = (x15 / 4);
    outputs[15] = (x16 / 4);
}

__global__ void create3DArray(int* bmVectors, float* blocks3D, int blockSize, float* dctBlocks)
{
    int block = (blockIdx.y  * blockSize) + blockIdx.x;
    int blockVectorIndex = (block << 4);
    int block3DIndex = (block * 1024);
    int blockOffset = (threadIdx.y * blockDim.x) + threadIdx.x;
    
//TODO: if bmVectors[<index>] == 0 !!!
    int index = (bmVectors[blockVectorIndex] << 6) + blockOffset;
    float a = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+1] << 6) + blockOffset;
    float b = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+2] << 6) + blockOffset;
    float c = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+3] << 6) + blockOffset;
    float d = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+4] << 6) + blockOffset;
    float e = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+5] << 6) + blockOffset;
    float f = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+6] << 6) + blockOffset;
    float g = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+7] << 6) + blockOffset;
    float h = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+8] << 6) + blockOffset;
    float i = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+9] << 6) + blockOffset;
    float j = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+10] << 6) + blockOffset;
    float k = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+11] << 6) + blockOffset;
    float l = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+12] << 6) + blockOffset;
    float m = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+13] << 6) + blockOffset;
    float n = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+14] << 6) + blockOffset;
    float o = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+15] << 6) + blockOffset;
    float p = (index > -1) ? dctBlocks[index] : 0;

    float inputs[16] = {a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p};
    float outputs[16];
    HadamarTransform16(inputs, outputs);

    block3DIndex += blockOffset;
    blocks3D[block3DIndex] = outputs[0];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[1];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[2];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[3];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[4];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[5];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[6];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[7];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[8];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[9];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[10];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[11];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[12];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[13];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[14];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[15];
}

__global__ void InverseHadamarTransform16(float* blocks3D, int blockSize)
{
    int block = (blockIdx.y  * blockSize) + blockIdx.x;
    int block3DIndex = (block * 1024);
    int blockPixel = (threadIdx.y * blockDim.x) + threadIdx.x;
    
    int index = block3DIndex + blockPixel;
    float a = blocks3D[index];
    index += 64;
    float b = blocks3D[index];
    index += 64;
    float c = blocks3D[index];
    index += 64;
    float d = blocks3D[index];
    index += 64;
    float e = blocks3D[index];
    index += 64;
    float f = blocks3D[index];
    index += 64;
    float g = blocks3D[index];
    index += 64;
    float h = blocks3D[index];
    index += 64;
    float i = blocks3D[index];
    index += 64;
    float j = blocks3D[index];
    index += 64;
    float k = blocks3D[index];
    index += 64;
    float l = blocks3D[index];
    index += 64;
    float m = blocks3D[index];
    index += 64;
    float n = blocks3D[index];
    index += 64;
    float o = blocks3D[index];
    index += 64;
    float p = blocks3D[index];

    float inputs[16] = {a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p};
    float outputs[16];
    HadamarTransform16(inputs, outputs);

    block3DIndex = block3DIndex + blockPixel;
    blocks3D[block3DIndex] = outputs[0];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[1];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[2];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[3];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[4];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[5];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[6];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[7];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[8];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[9];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[10];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[11];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[12];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[13];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[14];
    block3DIndex += 64;
    blocks3D[block3DIndex] = outputs[15];
}

void BM3D::BM3D_InverseTransform()
{
    //inverse Z transform 
    dim3 threadsPerBlock(8,8); 
    int blockXY = sqrt(BM3D::context.nbBlocks); 
    dim3 numBlocks(blockXY, blockXY);
    InverseHadamarTransform16<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks3D, blockXY);
    cudaThreadSynchronize();    
    
    BM3D_2DiDCT();
}

__global__ void CalculateDistances(int* distanceArray, int* blockMap, float* dctBlocks, int blockSize, int* bmVectorsComplete)
{
    int block = (blockIdx.y * blockSize) + blockIdx.x;
    int blockIndex = (block << 6);
    int blockMapIndex = (block << 3);
    int rx = blockMap[blockMapIndex+6];
    int ry = blockMap[blockMapIndex+7];
    int compareBlock = ((blockIdx.y + (ry * threadIdx.y)) * blockSize) + (rx * threadIdx.x) + blockIdx.x;
    int compareBlockIndex = (compareBlock << 6);
    int distanceArrayIndex = (block * 169) + (threadIdx.y * 13) + threadIdx.x;
    float distance = 0, diff = 0;
 
    bmVectorsComplete[distanceArrayIndex] = compareBlock+1;
    /*if(block == 16383)
    {
        printf("\n [%d] compare block = %d", (threadIdx.y *13 + threadIdx.x), compareBlock);
    }*/
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
}

__global__ void ShrinkBmVectors(int* bmVectorsComplete, int* bmVectors, int* blockGroupMap, int* blockGroupIndex)
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
            int offset = atomicAdd(&blockGroupIndex[val-1], 1);
            int index2 = ((val-1) << 7) + (offset << 1);    
            blockGroupMap[index2] = block;
            blockGroupMap[index2+1] = i;
            bmVectors[bmVectorIndex + i] = val-1;
            ++i;
            if(i >= 16) break;
        }
    }
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
        ShrinkBmVectors<<<numBlocks,threadsPerBlock>>>(BM3D::context.bmVectorsComplete, BM3D::context.bmVectors, BM3D::context.blockGroupMap, BM3D::context.blockGroupIndex);
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

__global__ void iDCT2D8x8(float* blocks3D, float* finalBlocks3D, int blockSize, float* dctCosParam1, float* dctCosParam2, float* cArray)
{
    int block = (blockIdx.y * blockSize) + blockIdx.x;
    int blocks3DIndex = (block << 10) + (threadIdx.y << 2) + threadIdx.x;
    
        
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
                    sum += cArray[u] * cArray[v] * blocks3D[blocks3DIndex + (v << 3) + u] * dctCosParam1[(x << 3) + u] * dctCosParam2[(y << 3) + v];
                }
            }
    
            finalBlocks3D[blocks3DIndex + (y << 3) +  x] =  fabs(round(0.25 * sum));
        }
    }

    /*if(block == 100 && threadIdx.y == 0 && threadIdx.x == 0)
    {
        int blockIndex = (block << 7);
        for(int i= 0; i< 128; i += 2)
        {
            printf("[%d,%d], ", blockGroupMap[blockIndex+ i], blockGroupMap[blockIndex+ i + 1]);   
        }
    }*/

    /*if(block == 16383 && threadIdx.y == 0 && threadIdx.x == 0)
    {
        printf("\nBlock = %d\n\n", block);
        for(int i= 0; i< 64; i++)
        {
            printf("%f, ", finalBlocks3D[blocks3DIndex + i]);   
        } 
    }*/
}

/*__global__ void iDCT2D8x8(float* blocks, float* dctBlocks, float* dctCosParam1, float* dctCosParam2, float* cArray)
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
    }
}
*/

__global__ void DCT2D8x8(float* blocks, float* dctBlocks, float* dctCosParam1, float* dctCosParam2, float* cArray)
{
    int size = blockDim.x * blockDim.y;
    int block = (((blockIdx.y << 4 ) + blockIdx.x)  * size) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //printf("\nb= %d, bY = %d, bX = %d, tY = %d, tX = %d", block, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x);    
    int blockIndex = block * size;

    /*if(block == 100)
    {
        printf("\nDBlock = %d\n", block);
        for(int i= 0; i< size * 2; i += 2)
        {
            printf("[%d,%d], ", blockGroupMap[(blockIndex << 1) + i], blockGroupMap[(blockIndex << 1) + i + 1]);   
        }
        printf("\nPixel = 1978\n", block);
        for(int i= 0; i< size; i++)
        {
            printf("%d, ", pixelMap[126592 + i]);   
        }
    }
    */

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

/*void BM3D::BM3D_2DiDCT()
{
    dim3 threadsPerBlock(8,8);
    int blockXY = sqrt(BM3D::context.nbBlocks >> 6); 
    dim3 numBlocks(blockXY, blockXY);
    iDCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.blocks, BM3D::context.dctBlocks, BM3D::context.idctCosParam1, BM3D::context.idctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
}
*/

__device__ void Hadamar8(float* inputs, float* outputs, float DIVISOR)
{
    float a = inputs[0];
    float b = inputs[1];
    float c = inputs[2];
    float d = inputs[3];
    float e = inputs[4];  
    float f = inputs[5];
    float g = inputs[6];
    float h = inputs[7];
    
    outputs[0] = (a+b+c+d+e+f+g+h)/DIVISOR;
    outputs[1] = (a-b+c-d+e-f+g-h)/DIVISOR;
    outputs[2] = (a+b-c-d+e+f-g-h)/DIVISOR;
    outputs[3] = (a-b-c+d+e-f-g+h)/DIVISOR;
    outputs[4] = (a+b+c+d-e-f-g-h)/DIVISOR;
    outputs[5] = (a-b+c-d-e+f-g+h)/DIVISOR;
    outputs[6] = (a+b-c-d-e-f+g+h)/DIVISOR;
    outputs[7] = (a-b-c+d-e+f+g-h)/DIVISOR;
}

__global__ void Hadamar2D_Row(float* blocks, float* dctBlocks, int blockSize, float DIVISOR)
{
    int block = (blockIdx.y * blockSize) + blockIdx.x;
    int blockIndex = (block << 6);
    
    printf("\nblock = %d", block);
    if(block == 16383)
    {
        printf("\nDBlock = %d\n", block);
        for(int i= 0; i< 64; i++)
        {
            printf("%f, ", blocks[blockIndex + i]);   
        } 
    }

    float inputs[8];
    float outputs[8];
    int i = threadIdx.x;
    int offset1 = blockIndex + (i << 3);
    //for(int i=0; i< 8; ++i)
    //{
        inputs[0] = blocks[offset1];
        inputs[1] = blocks[offset1 + 1];
        inputs[2] = blocks[offset1 + 2];
        inputs[3] = blocks[offset1 + 3];
        inputs[4] = blocks[offset1 + 4];
        inputs[5] = blocks[offset1 + 5];  
        inputs[6] = blocks[offset1 + 6];
        inputs[7] = blocks[offset1 + 7];

        Hadamar8(inputs, outputs, DIVISOR);

        dctBlocks[offset1] = outputs[0];
        dctBlocks[offset1 + 1] = outputs[1];
        dctBlocks[offset1 + 2] = outputs[2];
        dctBlocks[offset1 + 3] = outputs[3];
        dctBlocks[offset1 + 4] = outputs[4];
        dctBlocks[offset1 + 5] = outputs[5];
        dctBlocks[offset1 + 6] = outputs[6];
        dctBlocks[offset1 + 7] = outputs[7];
    //}

    //for(int i=0; i< 8; ++i)
    //{
       /* inputs[0] = dctBlocks[blockIndex + i];
        inputs[1] = dctBlocks[blockIndex + 9];
        inputs[2] = dctBlocks[blockIndex + 18];
        inputs[3] = dctBlocks[blockIndex + 27];
        inputs[4] = dctBlocks[blockIndex + 36];
        inputs[5] = dctBlocks[blockIndex + 45];  
        inputs[6] = dctBlocks[blockIndex + 54];
        inputs[7] = dctBlocks[blockIndex + 63];

        Hadamar8(inputs, outputs, DIVISOR);

        dctBlocks[blockIndex + i] = outputs[0];
        dctBlocks[blockIndex + 8 + i] = outputs[1];
        dctBlocks[blockIndex + 16 + i] = outputs[2];
        dctBlocks[blockIndex + 24 + i] = outputs[3];
        dctBlocks[blockIndex + 32 + i] = outputs[4];
        dctBlocks[blockIndex + 40 + i] = outputs[5];
        dctBlocks[blockIndex + 48 + i] = outputs[6];
        dctBlocks[blockIndex + 56 + i] = outputs[7];
*/
    //}

    /*if(block == 16383)
    {
        printf("\nDBlock = %d\n", block);
        for(int i= 0; i< 64; i++)
        {
            printf("%f, ", dctBlocks[blockIndex + i]);   
        } 
    }*/

}

__global__ void Hadamar2D_Col(float* blocks, float* dctBlocks, int blockSize, float DIVISOR)
{
    int block = (blockIdx.y * blockSize) + blockIdx.x;
    int blockIndex = (block << 6);
    
    /*printf("\nblock = %d", block);
    if(block == 16383)
    {
        printf("\nDBlock = %d\n", block);
        for(int i= 0; i< 64; i++)
        {
            printf("%f, ", blocks[blockIndex + i]);   
        } 
    }*/

    float inputs[8];
    float outputs[8];
    int i = threadIdx.x;
    //int offset1 = blockIndex + (i << 3);
    //for(int i=0; i< 8; ++i)
    //{
        /*inputs[0] = blocks[offset1];
        inputs[1] = blocks[offset1 + 1];
        inputs[2] = blocks[offset1 + 2];
        inputs[3] = blocks[offset1 + 3];
        inputs[4] = blocks[offset1 + 4];
        inputs[5] = blocks[offset1 + 5];  
        inputs[6] = blocks[offset1 + 6];
        inputs[7] = blocks[offset1 + 7];

        Hadamar8(inputs, outputs, DIVISOR);

        dctBlocks[offset1] = outputs[0];
        dctBlocks[offset1 + 1] = outputs[1];
        dctBlocks[offset1 + 2] = outputs[2];
        dctBlocks[offset1 + 3] = outputs[3];
        dctBlocks[offset1 + 4] = outputs[4];
        dctBlocks[offset1 + 5] = outputs[5];
        dctBlocks[offset1 + 6] = outputs[6];
        dctBlocks[offset1 + 7] = outputs[7];*/
    //}

    //for(int i=0; i< 8; ++i)
    //{
        inputs[0] = dctBlocks[blockIndex + i];
        inputs[1] = dctBlocks[blockIndex + 9 + i];
        inputs[2] = dctBlocks[blockIndex + 18 + i];
        inputs[3] = dctBlocks[blockIndex + 27 + i];
        inputs[4] = dctBlocks[blockIndex + 36 + i];
        inputs[5] = dctBlocks[blockIndex + 45 + i];  
        inputs[6] = dctBlocks[blockIndex + 54 + i];
        inputs[7] = dctBlocks[blockIndex + 63 + i];

        Hadamar8(inputs, outputs, DIVISOR);

        dctBlocks[blockIndex + i] = outputs[0];
        dctBlocks[blockIndex + 9 + i] = outputs[1];
        dctBlocks[blockIndex + 18 + i] = outputs[2];
        dctBlocks[blockIndex + 27 + i] = outputs[3];
        dctBlocks[blockIndex + 36 + i] = outputs[4];
        dctBlocks[blockIndex + 45 + i] = outputs[5];
        dctBlocks[blockIndex + 54 + i] = outputs[6];
        dctBlocks[blockIndex + 63 + i] = outputs[7];

    //}

    if(block == 16383)
    {
        printf("\nDBlock = %d\n", block);
        for(int i= 0; i< 64; i++)
        {
            printf("%f, ", dctBlocks[blockIndex + i]);   
        } 
    }

}

void BM3D::BM3D_2DiDCT()
{
    dim3 threadsPerBlock(4,4);
    int blockXY = sqrt(BM3D::context.nbBlocks); 
    dim3 numBlocks(blockXY, blockXY);
    iDCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks3D, BM3D::context.finalBlocks3D, blockXY, BM3D::context.idctCosParam1, BM3D::context.idctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
}

void BM3D::BM3D_2DDCT()
{
    dim3 threadsPerBlock(8,8);
    int blockXY = sqrt(BM3D::context.nbBlocks >> 6); 
    dim3 numBlocks(blockXY, blockXY);
    DCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.blocks, BM3D::context.dctBlocks, BM3D::context.dctCosParam1, BM3D::context.dctCosParam2, BM3D::context.cArray);
    cudaThreadSynchronize ();
    /*dim3 threadsPerBlock(8);
    int blockXY = sqrt(BM3D::context.nbBlocks); 
    dim3 numBlocks(blockXY, blockXY);
    float DIVISOR = sqrt(8);
    Hadamar2D_Row<<<numBlocks,threadsPerBlock>>>(BM3D::context.blocks, BM3D::context.dctBlocks, blockXY, DIVISOR);
    cudaThreadSynchronize();
    Hadamar2D_Col<<<numBlocks,threadsPerBlock>>>(BM3D::context.blocks, BM3D::context.dctBlocks, blockXY, DIVISOR);
    cudaThreadSynchronize();*/
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

__global__ void CreateBlocks(float* img, float* blocks, int* blockMap, int width, int* pixelMap, int* pixelMapIndex)
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

    //int index = (imgIndex << 6) + (threadIdx.y << 3) + threadIdx.x;
    int offset = atomicAdd(&pixelMapIndex[imgIndex], 1); 
    int index = (imgIndex << 6) + offset;
    pixelMap[index] = block; 
    /*if(block == 16383)
    {
        printf("\nblock = %d, imgIndex = %d, x =%d, y = %d", block, imgIndex, threadIdx.x, threadIdx.y);
    }
    if(imgIndex == 53970)
    {
        printf("\nblock = %d, imgIndex = %d, x =%d, y = %d", block, imgIndex, threadIdx.x, threadIdx.y);
    }*/
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
    CreateBlocks<<<numBlocks2, threadsPerBlock>>>(BM3D::context.deviceImage, BM3D::context.blocks, BM3D::context.blockMap, BM3D::context.img_width, BM3D::context.pixelMap, BM3D::context.pixelMapIndex); 
    cudaThreadSynchronize ();
}




	
