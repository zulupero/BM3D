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
    cufftComplex* plainSrc;
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

__global__
void Process2DHT_intern(cufftComplex* src, int gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y;
    if(src[i].x <= gamma) src[i].x = 0;
}


void ImgHelperCuda::Process2DHT(cufftComplex* src, int gamma)
{
    dim3 blocks(5,5);
    dim3 threads(8,8);
    Process2DHT_intern<<<blocks,threads>>>(src, gamma);
    cudaThreadSynchronize ();
}




