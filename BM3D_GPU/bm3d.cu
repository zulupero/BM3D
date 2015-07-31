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

    int widthOffset = width % pHard;
    int heightOffset = height % pHard;
    BM3D::context.img_widthWithBorder = width - widthOffset + nHard;
    BM3D::context.img_heightWithBorder = height - heightOffset + nHard;
    BM3D::context.nbBlocks = ((width - widthOffset) / pHard) * ((height - heightOffset) / pHard);

    BM3D::context.img_width = width; 
    BM3D::context.img_height= height;
    BM3D::context.pHard = pHard;
    BM3D::context.nHard = nHard;
    BM3D::context.sourceImage = img;

    gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.deviceImage, &img[0], width * height * sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&BM3D::context.deviceBlocks, BM3D::context.nbBlocks * nHard * nHard * sizeof(float)));
    printf("\n\tBorder width (pixel) = %d", (-1 * widthOffset + nHard));
    printf("\n\tBorder height (pixel) = %d", (-1 * heightOffset + nHard));
    printf("\n\tImg width (border) = %d", BM3D::context.img_widthWithBorder);
    printf("\n\tImg height (border) = %d", BM3D::context.img_heightWithBorder);
    printf("\n\tNumber of blocks = %d", BM3D::context.nbBlocks);
    printf("\n\tSize blocks array = %u bytes", (BM3D::context.nbBlocks * nHard * nHard * sizeof(float)));
}

void BM3D::BM3D_Run()
{
    printf("\n\nRun BM3D");    
    printf("\n\t--> Execution on Tesla K40c");
    if(cudaSuccess != cudaSetDevice(0)) printf("\n\tNo device 0 available");

    
    BM3D_BasicEstimate();
}

void BM3D::BM3D_BasicEstimate()
{
    printf("\n\tBasic estimates (1 step)");
    BM3D_CreateBlocks();
}

__global__
void CreateBlocks_Intern(float* img, float* blocks)
{
    int n = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    //printf("\nThread: %d", n);
}

void BM3D::BM3D_CreateBlocks()
{
    printf("\n\t\tCreate blocks");
    dim3 threadsPerBlock(1);
    dim3 numBlocks(BM3D::context.nbBlocks);
    CreateBlocks_Intern<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceImage, BM3D::context.deviceBlocks);
    cudaThreadSynchronize();
}




	
