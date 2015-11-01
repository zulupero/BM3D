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

void BM3D::BM3D_dispose()
{
    /*cudaFree(BM3D::context.pixelMap); //Pixel Map
    cudaFree(BM3D::context.pixelMapIndex); //Pixel Map Index
    cudaFree(BM3D::context.blocks); //Blocks array
    cudaFree(BM3D::context.wpArray); //WP array (for each block)
    cudaFree(BM3D::context.npArray); //NP array (for each block)
    cudaFree(BM3D::context.dctBlocks); //DCT Blocks array
    cudaFree(BM3D::context.blockMap); //store x;y;vX;vY;zX;zY;rX;rY of each block (Block Map)
    cudaFree(BM3D::context.distanceArray); //distance array (for each block)
    cudaFree(BM3D::context.blockGroupMap); //BlockGroup Map
    cudaFree(BM3D::context.blockGroupIndex); //BlockGroup Map Index
    cudaFree(BM3D::context.bmVectors); //BM vectors
    cudaFree(BM3D::context.bmVectorsComplete); //BM vectors
    cudaFree(BM3D::context.deviceBlocks3D);
    cudaFree(BM3D::context.finalBlocks3D);
    cudaFree(BM3D::context.deviceImage);
    cudaFree(BM3D::context.basicImage);
    cudaFree(BM3D::context.basicNumDen);

    cudaFree(BM3D::context.dctCosParam1);
    cudaFree(BM3D::context.dctCosParam2);
    cudaFree(BM3D::context.idctCosParam1);
    cudaFree(BM3D::context.idctCosParam2);

    cudaFree(BM3D::context.cArray);*/
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
    BM3D::context.debugPixel = 1792;//255;
    BM3D::context.debugBlock = 15508;//124;

    BM3D::context.nbBlocks = (width / 2) * (height /2);
    BM3D::context.nbBlocks_total = BM3D::context.nbBlocks;
    BM3D::context.nbBlocksPerLine = (width / 2);
    BM3D::context.nbBlocksPerLine_total = BM3D::context.nbBlocksPerLine;

    BM3D::context.img_width = width; 
    BM3D::context.img_height= height;
    BM3D::context.pHard = pHard;
    BM3D::context.nHard = nHard;
    BM3D::context.sourceImage = img;

    gpuErrchk(cudaMalloc(&BM3D::context.pixelMap, width * height * 64 * 3 * sizeof(int))); //Pixel Map
    gpuErrchk(cudaMemset(BM3D::context.pixelMap, -1, width * height * 64 * 3 * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.pixelMapIndex, width * height * sizeof(int))); //Pixel Map Index
    gpuErrchk(cudaMemset(BM3D::context.pixelMapIndex, 0, width * height * sizeof(int)));
    gpuErrchk(cudaMalloc(&BM3D::context.blocks, BM3D::context.nbBlocks * nHard * nHard * sizeof(float))); //Blocks array
    gpuErrchk(cudaMemset(BM3D::context.blocks, 0, BM3D::context.nbBlocks * nHard * nHard * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.wpArray, BM3D::context.nbBlocks * sizeof(float))); //WP array (for each block)
    gpuErrchk(cudaMalloc(&BM3D::context.npArray, BM3D::context.nbBlocks * sizeof(int))); //NP array (for each block)
    gpuErrchk(cudaMemset(BM3D::context.npArray, 0, BM3D::context.nbBlocks * sizeof(int))); 
    gpuErrchk(cudaMalloc(&BM3D::context.dctBlocks, BM3D::context.nbBlocks * nHard * nHard * sizeof(float))); //DCT Blocks array
    gpuErrchk(cudaMemset(BM3D::context.dctBlocks, 0, BM3D::context.nbBlocks * nHard * nHard * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.blockMap, BM3D::context.nbBlocks * 8 * sizeof(int))); //store x;y;vX;vY;zX;zY;rX;rY of each block (Block Map)
    gpuErrchk(cudaMalloc(&BM3D::context.distanceArray, BM3D::context.nbBlocks * 169 * sizeof(float))); //distance array (for each block)
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
    gpuErrchk(cudaMalloc(&BM3D::context.basicImage, BM3D::context.img_width * BM3D::context.img_height * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.basicImage, 0, BM3D::context.img_width * BM3D::context.img_height * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.basicValues, BM3D::context.img_width * BM3D::context.img_height * 2 * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.basicValues, 0, BM3D::context.img_width * BM3D::context.img_height * 2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&BM3D::context.numberOfSimilarPatches, BM3D::context.nbBlocks * sizeof(int)));
    gpuErrchk(cudaMemset(BM3D::context.numberOfSimilarPatches, 0, BM3D::context.nbBlocks * sizeof(int)));

    //Kaiser-window coef
    float kaiserWindow[64] = {0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924,
                          0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989,
                          0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846,
                          0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325,
                          0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325,
                          0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846,
                          0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989,
                          0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924};

    gpuErrchk(cudaMalloc(&BM3D::context.kaiserWindowCoef, 64 * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.kaiserWindowCoef, kaiserWindow, 64 * sizeof(float), cudaMemcpyHostToDevice));

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
    printf("\n\tSize Basic Image array= %f Mb", (BM3D::context.img_width * BM3D::context.img_height * sizeof(float)/1024.00 / 1024.00));      
    printf("\n\tSize Basic values array= %f Mb", (BM3D::context.img_width * BM3D::context.img_height * 2 * sizeof(float)/1024.00 / 1024.00));
    printf("\n\tSize 3D array = %f Mb", (BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float) /1024.00 / 1024.00));  
    printf("\n\tSize 3D array (final) = %f Mb", (BM3D::context.nbBlocks * nHard * nHard * 16 * sizeof(float) /1024.00 / 1024.00));      
    printf("\n\tSize Block Matching vectors = %f Mb", (BM3D::context.nbBlocks * 16 * sizeof(int) /1024.00 / 1024.00));  
    printf("\n\tSize Block Matching vectors (complete) = %f Mb", (BM3D::context.nbBlocks * 169 * sizeof(int) /1024.00 / 1024.00));      
    printf("\n\tSize Block Map array = %f Mb", (BM3D::context.nbBlocks * 8 * sizeof(int) /1024.00 / 1024.00));  
    printf("\n\tSize Blocks array = %f Mb", (BM3D::context.nbBlocks * nHard * nHard * sizeof(float) /1024.00 / 1024.00));  
    printf("\n\tSize Distance array = %f Mb", (BM3D::context.nbBlocks * 169 * sizeof(float) /1024.00 / 1024.00));
    printf("\n\tSize Pixel map= %f Mb", (width * height * 64 * 3 * sizeof(int) /1024.00 / 1024.00));
    printf("\n\tSize Pixel map Index= %f Mb", (width * height * sizeof(int) /1024.00 / 1024.00));    
    printf("\n\tSize Block group map = %f Mb", (BM3D::context.nbBlocks * 128 * sizeof(int) /1024.00 / 1024.00));
    printf("\n\tSize Block group Index = %f Mb", (BM3D::context.nbBlocks * sizeof(int) /1024.00 / 1024.00));
    printf("\n\tSize WP array = %f Mb", (BM3D::context.nbBlocks * sizeof(float) /1024.00 / 1024.00));
    printf("\n\tSize NP array = %f Mb", (BM3D::context.nbBlocks * sizeof(int) /1024.00 / 1024.00));
    printf("\n\tSize Number of patches = %f Mb", (BM3D::context.nbBlocks * sizeof(int) /1024.00 / 1024.00));
}

__global__ void ShowImage(float* image, float* basicImage, int* pixelMap, int* blockGroupMap, int* npArray, float* wpArray, int debugBlock, int debugPixel, int* numberOfSimilarPatches)
{
    printf("\nTest PixelMap \n");
    for(int i= 0; i<64; ++i)
    {
        printf("[%d, %d:%d], ", pixelMap[(debugPixel * 192) + (i * 3)], pixelMap[(debugPixel * 192) + (i * 3) + 1], pixelMap[(debugPixel * 192) + (i * 3) + 2]);      
    }    
    
    printf("\nTest BlockGroup \n");
    for(int i= 0; i<64; ++i)
    {
        printf("[%d, %d], ", blockGroupMap[(debugBlock << 7) + (i<<1)], blockGroupMap[(debugBlock << 7) + (i<<1)+ 1]);      
    }
       
    
    printf("\nTEST Image\n");
    for(int i=0;i<512; i++)
    {
        printf("%f, ", image[i]);
    }   
    printf("\nTEST basic Image\n");
    for(int i=0;i<65536; i++)
    {
        if(basicImage[i] > 250) printf("%d: %f, ", i, basicImage[i]);
    }

    printf("\n WP array \n");
    for(int i=0;i<200; i++)
    {
        printf("%f, ", wpArray[i]);
    }
    
    printf("\n NP array \n");
    for(int i=0;i<200; i++)
    {
        printf("%d, ", npArray[i]);
    }

    printf("\n Similar patches array \n");
    for(int i=0;i<16384; i++)
    {
        if(numberOfSimilarPatches[i] != 16) printf("[%d, %d], ", i, numberOfSimilarPatches[i]);
    }
}

__global__ void ShowBlock(int StartIndex, float* array, int* vectorArray, int index)
{
    printf("\nTEST Block---");
    for(int i= 0; i< 64; i++)
    {
        printf("%f, ", array[StartIndex + i]);   
    } 

    printf("\nTEST Bm vector---");
    for(int i= 0; i< 16; i++)
    {
        printf("%d, ", vectorArray[index + i]);   
    } 
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
}

void BM3D::BM3D_SaveBasicImage()
{
    float* basicImage = (float*)malloc(BM3D::context.img_width * BM3D::context.img_height * sizeof(float));
    gpuErrchk(cudaMemcpy(&basicImage[0], BM3D::context.basicImage, BM3D::context.img_width * BM3D::context.img_height * sizeof(float), cudaMemcpyDeviceToHost));
    char* filename = "test.png";
    save_image(filename, basicImage, BM3D::context.img_width, BM3D::context.img_height, 1);
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
    BM3D_SaveBasicImage();
    BM3D_dispose();
}

__global__ void calculateNumeratorDenominator_basic(float* wpArray, float* basicValues, int* pixelMap, int* blockGroupMap, float* blocks3D, float* kWindowCoef, int debugPixel)
{
    int pixelIndex = (((blockIdx.y << 5) + blockIdx.x) << 6) + (threadIdx.y << 3) + threadIdx.x;
    int pixelMapIndex = pixelIndex * 192;
    
    //float values[16384 * 2];
    //for(int i= 0; i< 16384 * 2; ++i) values[i] = 0;
    //memset(&values[0], -1, 16384 * 2 * sizeof(float));
    int basicValuesIndex = (pixelIndex << 1);

    for(int i=0; i<64; ++i)
    {
        int index = i * 3;
        int block = pixelMap[pixelMapIndex + index];
        if(block == -1) break;

        int x = pixelMap[pixelMapIndex+index+1];
        int y = pixelMap[pixelMapIndex+index+2];
        for(int j=0; j<64; ++j)
        {
            int index2 = (block << 7) + (j * 2);
            int group = blockGroupMap[index2];
            if(group == -1) break;

            int block3DPos = blockGroupMap[index2+1];
            int block3DIndex = (group << 10) + (block3DPos << 6) + (y << 3) + x;
            float val = blocks3D[block3DIndex];
            //int indexValue = group * 2;
            int kWindowIndex = (y << 3) + x;
            //if(val > 0)
            {
                basicValues[basicValuesIndex] += (kWindowCoef[kWindowIndex] * wpArray[block] * val);
                basicValues[basicValuesIndex+1] += (kWindowCoef[kWindowIndex] * wpArray[block]);    
                //basicValues[basicValuesIndex] += (wpArray[block] * val);
                //basicValues[basicValuesIndex+1] += (wpArray[block]);    
            }
            if(pixelIndex == debugPixel) printf("\nval = %f, block = %d, group = %d, gIndex = %d, x = %d, y= %d, index = %d", val, block, group, block3DPos, x, y, block3DIndex);
            //if(val == 0) printf("\nval = %f, block = %d, group = %d, gIndex = %d, x = %d, y= %d, index = %d, pixel = %d", val, block, group, block3DPos, x, y, block3DIndex, pixelIndex);         
                        
            //basicValues[basicValuesIndex] += val;
            //basicValues[basicValuesIndex+1] += //wpArray[group];
        }
    } 
    /*int basicValuesIndex = (pixelIndex << 1);
    for(int i=0; i< 16384; ++i)
    {
        int index = i * 2; 
        //if(pixelIndex == 1000) printf("\nval = %f,", values[index]);       
        //if(values[index] > -1)
        //{
            basicValues[basicValuesIndex] += values[index];
            basicValues[basicValuesIndex+1] += values[index+1];
            //if(pixelIndex == 1000) printf("\nval1 = %f, val2 = %f,", values[index], values[index+1]);
        //}
    }*/
}

__global__ void aggregation(float* basicValues, float* basicImage, int debugPixel)
{
    int pixelIndex = (((blockIdx.y << 5) + blockIdx.x) << 6) + (threadIdx.y << 3) + threadIdx.x;
    int basicValuesIndex = (pixelIndex << 1);
    if(pixelIndex == debugPixel) printf("\nnum = %f, den = %f, ", basicValues[basicValuesIndex],  basicValues[basicValuesIndex+1]);
    basicImage[pixelIndex] = basicValues[basicValuesIndex] / basicValues[basicValuesIndex+1];
}

void BM3D::BM3D_Aggregation()
{
    {
        //dim3 threadsPerBlock(8,8,8); 
        //dim3 numBlocks(BM3D::context.img_width, BM3D::context.img_height,8);
        dim3 threadsPerBlock(8,8); 
        dim3 numBlocks(32, 32);
        calculateNumeratorDenominator_basic<<<numBlocks,threadsPerBlock>>>(BM3D::context.wpArray, BM3D::context.basicValues, BM3D::context.pixelMap, BM3D::context.blockGroupMap, BM3D::context.finalBlocks3D, BM3D::context.kaiserWindowCoef, BM3D::context.debugPixel); 
        cudaThreadSynchronize();    
    }

    {
        dim3 threadsPerBlock(8,8); 
        dim3 numBlocks(32, 32);
        aggregation<<<numBlocks,threadsPerBlock>>>(BM3D::context.basicValues, BM3D::context.basicImage, BM3D::context.debugPixel); 
        cudaThreadSynchronize();    
    }
    {
        dim3 testThreads(1);
        dim3 testBlocks(1);
        ShowImage<<<testBlocks,testThreads>>>(BM3D::context.deviceImage, BM3D::context.basicImage, BM3D::context.pixelMap, BM3D::context.blockGroupMap, BM3D::context.npArray, BM3D::context.wpArray, BM3D::context.debugBlock, BM3D::context.debugPixel, BM3D::context.numberOfSimilarPatches);
        cudaThreadSynchronize();    
    }
}

__global__ void calculateNPArray(float* blocks3D, int* npArray, int* numberOfSimilarPatches)
{
    int block = (((blockIdx.y << 5) + blockIdx.x) << 4) + (threadIdx.y << 2) + threadIdx.x;   
    int block3DIndex = (block << 10);
    int sum = 0;    
    //int size = (numberOfSimilarPatches[block] << 6);
    for(int i=0; i< 1024; ++i)
    {
        if(fabs(blocks3D[block3DIndex+i]) > 0) ++sum;
    }
    npArray[block] = sum;
}

__global__ void calculateWPArray(int* npArray, float* wpArray)
{
    int block = (((blockIdx.y << 5) + blockIdx.x) << 4) + (threadIdx.y << 2) + threadIdx.x;
    wpArray[block] = (npArray[block] > 1) ? (1.0 / (900 * npArray[block])) : 1;  //sigma squared = 900 / sigma = 30
}

void BM3D::BM3D_CalculateWPArray()
{
    {
        dim3 threadsPerBlock(4,4); 
        dim3 numBlocks(32, 32);
        printf("\n");
        calculateNPArray<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks3D, BM3D::context.npArray, BM3D::context.numberOfSimilarPatches); 
        cudaThreadSynchronize();    
    }
    {
        dim3 threadsPerBlock(4,4); 
        dim3 numBlocks(32, 32);
        calculateWPArray<<<numBlocks,threadsPerBlock>>>(BM3D::context.npArray, BM3D::context.wpArray); 
        cudaThreadSynchronize();    
    }
}

__global__ void applyHTFilter(float* blocks3D, float limit, int blockSize, int* numberOfSimilarPatches)
{
    int block = (blockIdx.y * blockSize) + blockIdx.x;
    int blockPixel = (threadIdx.y << 3) + threadIdx.x;
    int block3DIndex = (block << 10) + (threadIdx.z << 6) + blockPixel;
    float cmpVal = limit ; //sqrtf((float)numberOfSimilarPatches[block]);
    //int val = int(blocks3D[block3DIndex]);
    //int mul = ((((val -limit) >> 31) | -(-(val-limit) >> 31)) + 1) >> 1; 
    if(fabs(blocks3D[block3DIndex]) <= cmpVal) blocks3D[block3DIndex] = 0;
    //blocks3D[block3DIndex] = blocks3D[block3DIndex] * mul;
}

void BM3D::BM3D_HTFilter()
{ 
    dim3 threadsPerBlock(8,8,16); 
    int blockXY = sqrt(BM3D::context.nbBlocks); 
    //dim3 numBlocks(blockXY, blockXY);
    dim3 numBlocks(125, 125);
    applyHTFilter<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks3D, 2.7 * 30, /*blockXY*/124, BM3D::context.numberOfSimilarPatches); // limit = 2.7 * 30 (sigma) , BM3D IPOL = 2.7 * sigma, coef = sqrtf(#number of stacked blocks) := 4
    cudaThreadSynchronize();

    dim3 testThreads(1);
    dim3 testBlocks(1);
    ShowBlock<<<testBlocks, testThreads>>>((BM3D::context.debugBlock << 10) + (2 << 6), BM3D::context.deviceBlocks3D, BM3D::context.bmVectors, BM3D::context.debugBlock << 4);

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

__global__ void create3DArray(int* bmVectors, float* blocks3D, int blockSize, float* dctBlocks)
{
    int block = (blockIdx.y  * blockSize) + blockIdx.x;
    int blockVectorIndex = (block << 4);
    int block3DIndex = (block << 10);
    int blockOffset = (threadIdx.y * blockDim.x) + threadIdx.x;
    
    int index = (bmVectors[blockVectorIndex] > -1) ? (bmVectors[blockVectorIndex] << 6) + blockOffset : -1;
    float a = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+1] > -1 ) ? (bmVectors[blockVectorIndex+1] << 6) + blockOffset : -1;
    float b = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+2] > -1 ) ? (bmVectors[blockVectorIndex+2] << 6) + blockOffset : -1;
    float c = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+3] > -1 ) ? (bmVectors[blockVectorIndex+3] << 6) + blockOffset : -1;
    float d = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+4] > -1 ) ? (bmVectors[blockVectorIndex+4] << 6) + blockOffset : -1;
    float e = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+5] > -1 ) ? (bmVectors[blockVectorIndex+5] << 6) + blockOffset : -1;
    float f = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+6] > -1 ) ? (bmVectors[blockVectorIndex+6] << 6) + blockOffset : -1;
    float g = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+7] > -1 ) ? (bmVectors[blockVectorIndex+7] << 6) + blockOffset : -1;
    float h = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+8] > -1 ) ? (bmVectors[blockVectorIndex+8] << 6) + blockOffset : -1;
    float i = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+9] > -1 ) ? (bmVectors[blockVectorIndex+9] << 6) + blockOffset : -1;
    float j = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+10] > -1 ) ? (bmVectors[blockVectorIndex+10] << 6) + blockOffset : -1;
    float k = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+11] > -1 ) ? (bmVectors[blockVectorIndex+11] << 6) + blockOffset : -1;
    float l = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+12] > -1 ) ? (bmVectors[blockVectorIndex+12] << 6) + blockOffset : -1;
    float m = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+13] > -1 ) ? (bmVectors[blockVectorIndex+13] << 6) + blockOffset : -1;
    float n = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+14] > -1 ) ? (bmVectors[blockVectorIndex+14] << 6) + blockOffset : -1;
    float o = (index > -1) ? dctBlocks[index] : 0;
    index = (bmVectors[blockVectorIndex+15] > -1 ) ? (bmVectors[blockVectorIndex+15] << 6) + blockOffset : -1;        
    float p = (index > -1) ? dctBlocks[index] : 0;

    float inputs[16] = {a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p};
    float outputs[16];
    HadamarTransform16(inputs, outputs);

    block3DIndex += blockOffset;    
    blocks3D[block3DIndex] = outputs[0];
    /*if(block == 16383 && threadIdx.x == 0 && threadIdx.y == 0) 
    {
        printf("\na = %f,b = %f,c = %f,d = %f,e = %f,f = %f,g = %f,h = %f,i = %f,j = %f,k = %f,l = %f,m = %f,n = %f,o = %f,p = %f,",
                a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p);
        printf("\nTEST: val = %f", blocks3D[block3DIndex]);    
    
    }*/
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
    //dim3 numBlocks(blockXY, blockXY);
    dim3 numBlocks(125, 125);
    InverseHadamarTransform16<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks3D, /*blockXY*/124);
    cudaThreadSynchronize();    
    
    BM3D_2DiDCT();
}

__global__ void CalculateDistances(float* distanceArray, int* blockMap, float* dctBlocks, int blockSize, int* bmVector, int debugBlock)
{
    int block = (blockIdx.y * blockSize) + blockIdx.x;
    int blockIndex = (block << 6);
    //int blockMapIndex = (block << 3);
    //int rx = blockMap[blockMapIndex+6];
    //int ry = blockMap[blockMapIndex+7];
    //int compareBlock = 0;
    
    int rx = 0;
    int ry = 0;
    if(blockIdx.x > 6 && blockIdx.x < 118 && blockIdx.y > 6 && blockIdx.y < 118)
    { 
        rx = (threadIdx.x >= 6) ? 1: -1;
        ry = (threadIdx.y >= 6) ? 1: -1;    
    }
    else
    {
        rx = (blockIdx.x >= 64) ? -1: 1;
        ry = (blockIdx.y >= 64) ? -1: 1;
    }

    
    int distanceArrayIndex = (block * 169) + (threadIdx.y * 13) + threadIdx.x;
    float distance = 0, diff = 0;

    //if(blockIdx.x > 6 && blockIdx.x < 118 && blockIdx.y > 6 && blockIdx.y < 118)
    //{         
    int compareBlock = ((blockIdx.y + (ry * threadIdx.y)) * blockSize) + (rx * threadIdx.x) + blockIdx.x;
    bmVector[distanceArrayIndex] = compareBlock;
    int compareBlockIndex = (compareBlock << 6);
    //}
    //else
    //{
    //    compareBlock = block;
    //    bmVector[(block * 169)] = compareBlock;
    //}
    //int compareBlock = ((blockIdx.y + (ry * threadIdx.y)) * blockSize) + (rx * threadIdx.x) + blockIdx.x;
    
    
    //TODO: Perf bottleneck!!!
    for(int i =0; i< 64; ++i)
    {
        diff =  dctBlocks[compareBlockIndex + i] - dctBlocks[blockIndex + i];
        distance = distance + (diff * diff);
    }
    //int d = int(distance);
    //d = (d >> 6); //divide by nHardxnHard (8x8)
    //distanceArray[distanceArrayIndex] = d;
    distanceArray[distanceArrayIndex] = distance;
    if(block == debugBlock) printf("\nblock %d, cmp Block %d, distance = %f, bx = %d, by= %d, rx = %d, ry = %d, tx = %d, ty = %d", block, compareBlock, distance, blockIdx.x, blockIdx.y, rx, ry, threadIdx.x, threadIdx.y);
}

__global__ void ApplyDistanceThreshold(float* distanceArray, int limit, int blockSize, int* bmVectors)
{
    int distanceArrayIndex = (((blockIdx.y * blockSize) + blockIdx.x) * 169) + (threadIdx.y * 13) + threadIdx.x;
    //int distance = distanceArray[distanceArrayIndex] - limit;
    //int mul = (-((distance >> 31) | -(-distance >> 31)) + 1) >> 1;
    if(distanceArray[distanceArrayIndex] >= limit)
      bmVectors[distanceArrayIndex] = -1;  
    //bmVectorsComplete[distanceArrayIndex] *= mul;
}

__global__ void CreateBlockGroupMap(float* distanceArray, int* bmVectorsComplete, int* bmVectors, int* blockGroupMap, int* blockGroupIndex, int debugBlock, int* numberOfSimilarPatches)
{
    int block = (((blockIdx.y << 4) + blockIdx.x) << 6) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int bmVectorIndex = (block << 4);
    int bmVectorIndexC = (block * 169);

    for(int i=0; i<169; ++i)
    {
        float m = 99999999;
        int index = 0;
        for(int j=i;j<64; ++j)
        {
            if(distanceArray[j] < m) 
            {
                m = distanceArray[bmVectorIndexC + j];
                index = j;
            }
        }
        
        float tmp = distanceArray[bmVectorIndexC + index];
        int tmpIndex = bmVectorsComplete[bmVectorIndexC + index];
        distanceArray[bmVectorIndexC + index] = distanceArray[bmVectorIndexC + i];
        bmVectorsComplete[bmVectorIndexC + index] = bmVectorsComplete[bmVectorIndexC + i];
        distanceArray[bmVectorIndexC + i] = tmp;
        bmVectorsComplete[bmVectorIndexC + i] = tmpIndex;
    }

    int tot = 0;
    for(int index = 0; index < 169; ++index)
    {
        int cmpBlock = bmVectorsComplete[bmVectorIndexC + index];
        //if(block == debugBlock) printf("\nBlock = %d, cmp Block = %d, distance = %f", block, cmpBlock,distanceArray[bmVectorIndexC + index] );
        if( cmpBlock > -1  && distanceArray[index] < 360000)
        {
            int offset = atomicAdd(&blockGroupIndex[cmpBlock], 1);
            int index2 = ((cmpBlock) << 7) + (offset << 1);    
            blockGroupMap[index2] = block;
            blockGroupMap[index2+1] = tot;
            atomicAdd(&numberOfSimilarPatches[block], 1);
            //numberOfSimilarPatches[block]++;
            bmVectors[bmVectorIndex + tot] = cmpBlock;
            ++tot;
            if(tot >= 16) break;
        }
    }
}

void BM3D::BM3D_BlockMatching()
{
    {    
        dim3 threadsPerBlock(13,13); //sliding window of 13by13 patches of 8x8 pixels
        int blockXY = sqrt(BM3D::context.nbBlocks); 
        //dim3 numBlocks(blockXY, blockXY);
        dim3 numBlocks(125, 125);
        CalculateDistances<<<numBlocks,threadsPerBlock>>>(BM3D::context.distanceArray, BM3D::context.blockMap, BM3D::context.dctBlocks, /*blockXY*/124, BM3D::context.bmVectorsComplete, BM3D::context.debugBlock );
        cudaThreadSynchronize();

        //ApplyDistanceThreshold<<<numBlocks,threadsPerBlock>>>(BM3D::context.distanceArray, 2500 * 64 * 2, /*blockXY*/124, BM3D::context.bmVectorsComplete);
        //cudaThreadSynchronize();
    }
    
    {
        dim3 threadsPerBlock(8,8); 
        dim3 numBlocks(16, 16); //image 256x256 -> 16384 blocks (blocks: 16x16 / threads: 8x8)
        CreateBlockGroupMap<<<numBlocks,threadsPerBlock>>>(BM3D::context.distanceArray, BM3D::context.bmVectorsComplete, BM3D::context.bmVectors, BM3D::context.blockGroupMap, BM3D::context.blockGroupIndex, BM3D::context.debugBlock, BM3D::context.numberOfSimilarPatches);
        cudaThreadSynchronize();    
    }
    
    {
        dim3 threadsPerBlock(8,8); 
        int blockXY = sqrt(BM3D::context.nbBlocks); 
        //dim3 numBlocks(blockXY, blockXY);
        dim3 numBlocks(125, 125);
        create3DArray<<<numBlocks,threadsPerBlock>>>(BM3D::context.bmVectors, BM3D::context.deviceBlocks3D, /*blockXY*/124, BM3D::context.dctBlocks);
        cudaThreadSynchronize();

        dim3 testThreads(1);
        dim3 testBlocks(1);
        ShowBlock<<<testBlocks, testThreads>>>((BM3D::context.debugBlock << 10) + (2 << 6), BM3D::context.deviceBlocks3D, BM3D::context.bmVectors, BM3D::context.debugBlock << 4);        
        
    }
}

__global__ void iDCT2D8x8(float* blocks3D, float* finalBlocks3D, int blockSize, float* dctCosParam1, float* dctCosParam2, float* cArray, int debugBlock)
{
    int block = (blockIdx.y * blockSize) + blockIdx.x;
    int blocks3DIndex = (block << 10) + (((threadIdx.y << 2) + threadIdx.x) << 6);
    
    if(block == debugBlock && threadIdx.y == 0 && threadIdx.x == 0)
    {
        printf("\nBlock = %d\n\n", block);
        for(int i= 0; i< 64; i++)
        {
            printf("%f, ", blocks3D[blocks3DIndex + i]);   
        } 
    }
        
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

    if(block == debugBlock && threadIdx.y == 0 && threadIdx.x == 0)
    {
        printf("\nBlock = %d\n\n", block);
        for(int i= 0; i< 64; i++)
        {
            printf("%f, ", finalBlocks3D[blocks3DIndex + i]);   
        } 
    }
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

__global__ void DCT2D8x8(float* blocks, float* dctBlocks, float* dctCosParam1, float* dctCosParam2, float* cArray, int debugBlock)
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

    if(block == debugBlock)
    {
        printf("\nDBlock = %d\n", block);
        for(int i= 0; i< size; i++)
        {
            printf("%f, ", blocks[blockIndex + i]);   
        } 
    }

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

    if(block == debugBlock)
    {
        printf("\nDBlock = %d\n", block);
        for(int i= 0; i< size; i++)
        {
            printf("%f, ", dctBlocks[blockIndex + i]);   
        } 
    }
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
    
 
    if(block == 9357)
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

    if(block == 4104)
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
    //dim3 numBlocks(blockXY, blockXY);
    dim3 numBlocks(125, 125);
    iDCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks3D, BM3D::context.finalBlocks3D, /*blockXY*/124, BM3D::context.idctCosParam1, BM3D::context.idctCosParam2, BM3D::context.cArray, BM3D::context.debugBlock);
    cudaThreadSynchronize ();
}

void BM3D::BM3D_2DDCT()
{
    dim3 threadsPerBlock(8,8);
    int blockXY = sqrt(BM3D::context.nbBlocks >> 6); 
    dim3 numBlocks(blockXY, blockXY);
    //dim3 numBlocks(125, 125);
    DCT2D8x8<<<numBlocks,threadsPerBlock>>>(BM3D::context.blocks, BM3D::context.dctBlocks, BM3D::context.dctCosParam1, BM3D::context.dctCosParam2, BM3D::context.cArray, BM3D::context.debugBlock);
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

__global__ void CreateBlocks(float* img, float* blocks, int* blockMap, int width, int* pixelMap, int* pixelMapIndex, int debugBlock)
{
    int block = (blockIdx.y * 124) + blockIdx.x;
    //int blockMapIndex = block << 3;
    int blockIndex = block << 6;
   
    //int xImg = blockMap[blockMapIndex];
    //int yImg = blockMap[blockMapIndex+1];
    //int vX = blockMap[blockMapIndex+2];
    //int vY = blockMap[blockMapIndex+3];
    int xImg = (blockIdx.x << 1);
    int yImg = (blockIdx.y << 1);

    //int blockX = (vX == 1) ? threadIdx.x : (7 - threadIdx.x); 
    //int blockY = (vY == 1) ? threadIdx.y : (7 - threadIdx.y);
    //int blockPixelIndex = blockIndex + (blockY << 3) + blockX;
    int blockPixelIndex = blockIndex + (threadIdx.y << 3) + threadIdx.x;
    //int xPos = xImg + (vX * threadIdx.x);
    //int yPos = yImg + (vY * threadIdx.y);
    int xPos = xImg + threadIdx.x;
    int yPos = yImg + threadIdx.y;
    int imgIndex = (yPos * width) + xPos;
    //if(blockPixelIndex >= 1048576 || imgIndex >= 65536)
        //printf("\nblock = %d, blockMapIndex = %d, imgIndex = %d, pixelIndex = %d, xPos = %d, yPos = %d, vx = %d, vy = %d", block, blockMapIndex, imgIndex, blockPixelIndex, xImg, yImg, vX, vY);
    blocks[blockPixelIndex] = img[imgIndex];

    //int index = (imgIndex << 6) + (threadIdx.y << 3) + threadIdx.x;
    int offset = atomicAdd(&pixelMapIndex[imgIndex], 1); 
    int index = (imgIndex * 192) + (offset * 3);
    pixelMap[index] = block; 
    //pixelMap[index+1] = blockX; 
    //pixelMap[index+2] = blockY; 
    pixelMap[index+1] = threadIdx.x; 
    pixelMap[index+2] = threadIdx.y;
    if(block == debugBlock)
    {
        printf("\nblock = %d, imgIndex = %d, x =%d, y = %d, img = %f, block val = %f", block, imgIndex, xPos, yPos, img[imgIndex], blocks[blockPixelIndex]);
    }
    /*if(imgIndex == 53970)
    {
        printf("\nblock = %d, imgIndex = %d, x =%d, y = %d", block, imgIndex, threadIdx.x, threadIdx.y);
    }*/
}

void BM3D::BM3D_CreateBlockMap()
{
    dim3 threadsPerBlock(8, 8);
    //int width = (BM3D::context.img_width >> 2); //64 blocks per zone
    //int numBlockXY = (width >> 3);  //devision by 8, number of blocks X
    //dim3 numBlocks(numBlockXY, numBlockXY); //(8x8) (8x8) = 4096 blocks for each zone (1,2,3,4)
    //int blockIndexOffset = width * width;
    
    //CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockMap, 0, 0, 0, 1, 1);
    //cudaThreadSynchronize ();     
    //CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockMap, blockIndexOffset, (BM3D::context.img_width - 1), 0, -1, 1);
    //cudaThreadSynchronize (); 
    //CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockMap, (blockIndexOffset * 2), 0, (BM3D::context.img_height -1 ), 1, -1);
    //cudaThreadSynchronize (); 
    //CreateBlocks_Zone<<<numBlocks,threadsPerBlock>>>( BM3D::context.blockMap, (blockIndexOffset * 3), (BM3D::context.img_width-1), (BM3D::context.img_height-1), -1, -1);
    //cudaThreadSynchronize (); 
        
    //int blockXY = sqrt(BM3D::context.nbBlocks);
    //dim3 numBlocks2(blockXY, blockXY);
    dim3 numBlocks2(125, 125);
    CreateBlocks<<<numBlocks2, threadsPerBlock>>>(BM3D::context.deviceImage, BM3D::context.blocks, BM3D::context.blockMap, BM3D::context.img_width, BM3D::context.pixelMap, BM3D::context.pixelMapIndex, BM3D::context.debugBlock); 
    cudaThreadSynchronize ();
}




	
