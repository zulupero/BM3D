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

__global__
void BlocksInitialization(float** blocks, cufftComplex** dctBlocks, int nHard)
{
    int block = blockIdx.x;
    blocks[block] = (float*)malloc(nHard * nHard * sizeof(float));
    memset(blocks[block], 0, nHard * nHard * sizeof(float));
    dctBlocks[block] = (cufftComplex*)malloc(nHard * (nHard/2.0) * sizeof(cufftComplex));
    memset(dctBlocks[block], 0, nHard * (nHard/2.0) * sizeof(cufftComplex));
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
    BM3D::context.img_widthWithBorder = width - widthOffset + nHard;
    BM3D::context.img_heightWithBorder = height - heightOffset + nHard;
    BM3D::context.nbBlocks = ((width - widthOffset) / pHard) * ((height - heightOffset) / pHard);
    BM3D::context.nbBlocksPerLine = ((width - widthOffset) / pHard);

    BM3D::context.img_width = width; 
    BM3D::context.img_height= height;
    BM3D::context.pHard = pHard;
    BM3D::context.nHard = nHard;
    BM3D::context.sourceImage = img;

    gpuErrchk(cudaMalloc(&BM3D::context.deviceImage, BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder * sizeof(float)));
    gpuErrchk(cudaMemset(BM3D::context.deviceImage, 0, BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder * sizeof(float)));
    gpuErrchk(cudaMemcpy(BM3D::context.deviceImage, &img[0], width * height * sizeof(float), cudaMemcpyHostToDevice));
   
    //gpuErrchk(cudaMalloc(&BM3D::context.deviceBlocks, BM3D::context.nbBlocks * nHard * nHard * sizeof(float)));
    //gpuErrchk(cudaMemset(BM3D::context.deviceBlocks, 0, BM3D::context.nbBlocks * nHard * nHard * sizeof(float)));

    cudaMalloc((float**)&BM3D::context.deviceBlocks2, BM3D::context.nbBlocks * sizeof(float*));
    cudaMalloc((cufftComplex**)&BM3D::context.deviceBlocksDCT, BM3D::context.nbBlocks * sizeof(cufftComplex*));
    dim3 threadsPerBlock(1);
    dim3 numBlocks(BM3D::context.nbBlocks);
    BlocksInitialization<<<numBlocks,threadsPerBlock>>>(BM3D::context.deviceBlocks2, BM3D::context.deviceBlocksDCT, nHard);
    cudaThreadSynchronize ();

    Timer::addCuda("Cuda initialization");

    printf("\n\tBorder width (pixel) = %d", (-1 * widthOffset + nHard));
    printf("\n\tBorder height (pixel) = %d", (-1 * heightOffset + nHard));
    printf("\n\tImg width (border) = %d", BM3D::context.img_widthWithBorder);
    printf("\n\tImg height (border) = %d", BM3D::context.img_heightWithBorder);
    printf("\n\tNumber of blocks = %d", BM3D::context.nbBlocks);
    printf("\n\tSize blocks array = %u bytes", (BM3D::context.nbBlocks * nHard * nHard * sizeof(float)));
    printf("\n\tBlock per line= %d", BM3D::context.nbBlocksPerLine);
    printf("\n\tSize block array= %d", BM3D::context.nbBlocks * nHard * nHard);
    printf("\n\tSize Image array= %d", BM3D::context.img_widthWithBorder * BM3D::context.img_heightWithBorder);    
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
    //BM3D_2DDCT();
}

void BM3D::BM3D_2DDCT()
{
    Timer::startCuda();
    for(int i=0; i< BM3D::context.nbBlocks; ++i)
    {
        cufftHandle handle;
        cufftResult r = cufftPlan2d(&handle,BM3D::context.nHard,BM3D::context.nHard,CUFFT_R2C);
        CheckCufftError(r, "cufftPlan2d");

        r = cufftExecR2C(handle,BM3D::context.deviceBlocks2[i], BM3D::context.deviceBlocksDCT[i]);
        CheckCufftError(r, "cufftExecR2C");
        cudaThreadSynchronize ();

        r = cufftDestroy(handle);
        CheckCufftError(r, "cufftDestroy");  
    }
    Timer::addCuda("Basic estimate - 2D DCT (Blocks)");
}

__global__
void CreateBlocks_Intern(float* img, float** blocks, const int blocksPerLine, const int pHard, const int nHard, const int width, int sizeBlockArray, int sizeImgArray)
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

    bool checkBlockOffset = (offsetBlock >= sizeBlockArray);
    bool checkImgOffset = (offsetImg >= sizeImgArray);
    printf("\nblock %d, line = %d, phard = %d, nhard = %d, width = %d, offsetBlock = %d, offsetImg = %d, img_x = %d, img_y = %d, x = %d, y = %d, b= %s, i=%s, blocks[block][offsetBlock] = %f ", block, blocksPerLine, pHard, nHard, width, offsetBlock, offsetImg, img_x, img_y, x, y, (checkBlockOffset) ? "OUT" : "IN", (checkImgOffset) ? "OUT" : "IN", blocks[block][offsetBlock] );
}

void BM3D::BM3D_CreateBlocks()
{
    Timer::startCuda();
    dim3 threadsPerBlock(BM3D::context.nHard, BM3D::context.nHard);
    dim3 numBlocks(BM3D::context.nbBlocks);
    CreateBlocks_Intern<<<numBlocks,threadsPerBlock>>>( BM3D::context.deviceImage, 
                                                        BM3D::context.deviceBlocks2, 
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




	
