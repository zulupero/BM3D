#include <stdio.h>

#include "bm3d.h"
#include "utilities.h"

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


void BM3D::BM3D_Initialize(BM3D::SourceImage img, int width, int height, int pHard, int nHard)
{
    printf("\nBM3D context initialization");
    BM3D::context.img_width = width; 
    BM3D::context.img_height= height;
    BM3D::context.pHard = pHard;
    BM3D::context.nHard = nHard;
    BM3D::context.sourceImage = img;

    gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, width * height * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.deviceImage, &img[0], width * height * sizeof(float), cudaMemcpyHostToDevice));
  
    int nbBlocks = 2 * ((int((width - nHard)/pHard)) * (int((height - nHard)/pHard)));  
    gpuErrchk(cudaMalloc(&BM3D::context.deviceBlocks, nbBlocks * nHard * nHard * sizeof(float)));
    printf("\n\tNumber of blocks = %d", nbBlocks);
    printf("\n\tSize blocks array = %u", (nbBlocks * nHard * nHard * sizeof(float)));
}

void BM3D::BM3D_Run()
{
    printf("\nExecute on Tesla K40c");
    if(cudaSuccess != cudaSetDevice(0)) printf("\nNo device 0 available");

    printf("\n\nRun BM3D");
    BM3D_BasicEstimate();
}

void BM3D::BM3D_BasicEstimate()
{
    printf("\nBasic estimates (1 step)");
    BM3D_CreateBlocks();
}

void BM3D::BM3D_CreateBlocks()
{
    printf("\n\tCreate blocks");
        
}

__global__
void CreateBlocks(float* img)
{
}


	
