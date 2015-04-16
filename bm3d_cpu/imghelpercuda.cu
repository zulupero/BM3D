#include "imghelpercuda.h"
#include <vector>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void ImgHelperCuda::gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void ImgHelperCuda::CheckCufftError(cufftResult result, const char* method)
{
    switch(result)
    {
        case CUFFT_SUCCESS:
            //printf("\n CUFFT (%s): SUCCESS", method);
            break;
        case CUFFT_ALLOC_FAILED:
            printf("\n CUFFT (%s): Allocation failed", method);
            break;
        case CUFFT_INVALID_VALUE:
            printf("\n CUFFT (%s): Invalid value", method);
            break;
        case CUFFT_INTERNAL_ERROR:
            printf("\n CUFFT (%s): Internal error", method);
            break;
        case CUFFT_SETUP_FAILED:
            printf("\n CUFFT (%s): Setup failed", method);
            break;
        case CUFFT_INVALID_SIZE:
            printf("\n CUFFT (%s): Invalid size", method);
            break;
        default:
            printf("\n CUFFT (%s): unkown error", method);
    };
}

void ImgHelperCuda::fft(float* src, cufftComplex* dst, int width, int height)
{
    float* plainSrc;
    cufftComplex* plainDst;

    gpuErrchk(cudaMalloc(&plainSrc, width * height * sizeof(float)));
    gpuErrchk(cudaMalloc(&plainDst,  width * ((height/2) + 1) * sizeof(cufftComplex)));
    gpuErrchk(cudaMemcpy(plainSrc,src,width * height * sizeof(float),cudaMemcpyHostToDevice));

    cufftHandle handle;
    cufftResult r = cufftPlan2d(&handle,width,height,CUFFT_R2C);
    CheckCufftError(r, "cufftPlan2d");

    r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    CheckCufftError(r, "cufftSetCompatibilityMode");

    r = cufftExecR2C(handle,plainSrc,plainDst);
    CheckCufftError(r, "cufftExecR2C");
    cudaThreadSynchronize ();

    gpuErrchk(cudaMemcpy(dst, plainDst, width * ((height/2) + 1) * sizeof(cufftComplex),cudaMemcpyDeviceToHost));

    r = cufftDestroy(handle);
    CheckCufftError(r, "cufftDestroy");

    cudaFree(plainSrc);
    cudaFree(plainDst);
}

cufftComplex* ImgHelperCuda::fft2(float* src, int width, int height)
{
    float* plainSrc;
    cufftComplex* plainDst;

    gpuErrchk(cudaMalloc(&plainSrc, width * height * sizeof(float)));
    gpuErrchk(cudaMalloc(&plainDst,  width * ((height/2) + 1) * sizeof(cufftComplex)));
    gpuErrchk(cudaMemcpy(plainSrc,src,width * height * sizeof(float),cudaMemcpyHostToDevice));

    cufftHandle handle;
    cufftResult r = cufftPlan2d(&handle,width,height,CUFFT_R2C);
    CheckCufftError(r, "cufftPlan2d");

    r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    CheckCufftError(r, "cufftSetCompatibilityMode");

    r = cufftExecR2C(handle,plainSrc,plainDst);
    CheckCufftError(r, "cufftExecR2C");
    cudaThreadSynchronize ();

    //gpuErrchk(cudaMemcpy(dst, plainDst, width * ((height/2) + 1) * sizeof(cufftComplex),cudaMemcpyDeviceToHost));

    r = cufftDestroy(handle);
    CheckCufftError(r, "cufftDestroy");

    //cudaFree(plainSrc);
    //cudaFree(plainDst);
    return plainDst;
}

void ImgHelperCuda::ifft(cufftComplex* src, float* dst, int width, int height)
{
    cufftComplex* plainSrc;
    float* plainDst;

    gpuErrchk(cudaMalloc(&plainSrc, width * ((height/2) + 1) * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc(&plainDst, width * height * sizeof(float)));
    gpuErrchk(cudaMemcpy(plainSrc,src,width * ((height/2) + 1) * sizeof(cufftComplex),cudaMemcpyHostToDevice));

    cufftHandle handle;
    cufftResult r = cufftPlan2d(&handle,width,height,CUFFT_C2R);
    CheckCufftError(r, "cufftPlan2d");

    r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    CheckCufftError(r, "cufftSetCompatibilityMode");

    r = cufftExecC2R(handle,plainSrc,plainDst);
    CheckCufftError(r, "cufftExecZ2D");
    cudaThreadSynchronize ();

    gpuErrchk(cudaMemcpy(dst,plainDst,width * height * sizeof(float),cudaMemcpyDeviceToHost));

    r = cufftDestroy(handle);
    CheckCufftError(r, "cufftDestroy");

    cudaFree(plainSrc);
    cudaFree(plainDst);
}

float* ImgHelperCuda::ifft2(cufftComplex* src, int width, int height)
{
    //cufftComplex* plainSrc;
    float* plainDst;

    //gpuErrchk(cudaMalloc(&plainSrc, width * ((height/2) + 1) * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc(&plainDst, width * height * sizeof(float)));
    //gpuErrchk(cudaMemcpy(plainSrc,src,width * ((height/2) + 1) * sizeof(cufftComplex),cudaMemcpyHostToDevice));

    cufftHandle handle;
    cufftResult r = cufftPlan2d(&handle,width,height,CUFFT_C2R);
    CheckCufftError(r, "cufftPlan2d");

    r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    CheckCufftError(r, "cufftSetCompatibilityMode");

    r = cufftExecC2R(handle,src,plainDst);
    CheckCufftError(r, "cufftExecZ2D");
    cudaThreadSynchronize ();

    float* dst = (float*)malloc(width * height * sizeof(float));
    gpuErrchk(cudaMemcpy(dst,plainDst,width * height * sizeof(float),cudaMemcpyDeviceToHost));

    r = cufftDestroy(handle);
    CheckCufftError(r, "cufftDestroy");

    //cudaFree(plainSrc);
    //cudaFree(plainDst);

    return dst;
}


//this call could be avoided if we add this logic into the method ""ProcessNorm_intern"
__global__
void Process2DHT_intern(cufftComplex* src, int gamma, int windowSize)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int pos = i * windowSize + j;
    if(pos < windowSize * windowSize)
    {
        //avoid if (perf)!!! - GPU branching!!!
        if(src[pos].x < 0 && (src[pos].x * -1) < gamma ) { src[pos].x = 0; src[pos].y = 0; }
        if(src[pos].x > 0 && src[pos].x < gamma) { src[pos].x = 0; src[pos].y = 0; }
    }
}

__global__
void ProcessNorm_intern(cufftComplex* src, float* normVector, int windowSize, int blockSize)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int pos = i * windowSize + j;
    if((pos < (windowSize * windowSize)) && (pos % blockSize == 0))
    {
        int outIndex = (pos /blockSize) -1;
        //perf O(n2)!!!!
        float norm = 0;
        for(int k =0; i < blockSize; ++i)
        {
            for(int n=0; k < blockSize; ++k)
            {
                int i2 = i + k;
                int j2 = j + n;
                int pos2 = i2 * windowSize + j2;

                //Verify the formula!!!
                if(pos2 < (windowSize * windowSize)) norm += (src[pos2].x * src[pos2].x);
            }
        }
        normVector[outIndex] = norm;
    }
}

cufftComplex* ImgHelperCuda::get(cufftComplex* src, int width, int height)
{
    cufftComplex* dst = (cufftComplex*)malloc(width * (height/2) * sizeof(cufftComplex));
    gpuErrchk(cudaMemcpy(dst,src,width * height * sizeof(float),cudaMemcpyDeviceToHost));
    return dst;
}

float* ImgHelperCuda::get(float* src, int width, int height)
{
    float* dst = (float*)malloc(width * height * sizeof(float));
    gpuErrchk(cudaMemcpy(dst,src,width * height * sizeof(float),cudaMemcpyDeviceToHost));
    return dst;
}

void ImgHelperCuda::ProcessBM(cufftComplex* src, int gamma, int windowSize, int blockSize)
{
    dim3 threadsPerBlock(ImgHelperCuda::HT_2D_THREADS, ImgHelperCuda::HT_2D_THREADS);
    dim3 numBlocks(windowSize/threadsPerBlock.x, windowSize/threadsPerBlock.y);

    printf("\n\tprocess 2D HT");
    Process2DHT_intern<<<numBlocks,threadsPerBlock>>>(src, gamma, windowSize);
    cudaThreadSynchronize();

    printf("\n\tComputes blocks value");
    float* normVector_d;
    int sizeNormVector = (windowSize / blockSize) ;
    gpuErrchk(cudaMalloc(&normVector_d, sizeNormVector * sizeNormVector * sizeof(float)));

    ///--> we have to reduce the number of blocks and threads!!!!
    ProcessNorm_intern<<<numBlocks,threadsPerBlock>>>(src, normVector_d, windowSize, blockSize);
    cudaThreadSynchronize();

    //Only for testing
    float* normVector_h = (float*)malloc(sizeNormVector * sizeNormVector * sizeof(float));
    gpuErrchk(cudaMemcpy(normVector_h,normVector_d, sizeNormVector * sizeNormVector * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\n\n----- BLOCKS VALUE (TEST) ------\n");
    for(int i= 0; i < sizeNormVector * sizeNormVector; ++i)
    {
        printf("B%i: %f\n", (i+1), normVector_h[i] );
    }
    printf("\n");

    printf("\n\tMatching - 3D groups");
    printf("\n");
}




