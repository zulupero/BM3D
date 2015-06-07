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

    //r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    //CheckCufftError(r, "cufftSetCompatibilityMode");

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

cufftComplex* ImgHelperCuda::fft3D(float* src, int x, int y, int z)
{
    float* plainSrc;
    cufftComplex* plainDst;

    gpuErrchk(cudaMalloc(&plainSrc, x * y * z * sizeof(float)));
    gpuErrchk(cudaMalloc(&plainDst,  x * y * ((z/2) + 1) * sizeof(cufftComplex)));
    gpuErrchk(cudaMemcpy(plainSrc,src, x * y * z * sizeof(float), cudaMemcpyHostToDevice));

    cufftHandle handle;
    cufftResult r = cufftPlan3d(&handle,x,y,z,CUFFT_R2C);
    CheckCufftError(r, "cufftPlan3d");

    r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    CheckCufftError(r, "cufftSetCompatibilityMode");

    r = cufftExecR2C(handle,plainSrc,plainDst);
    CheckCufftError(r, "cufftExecR2C");
    cudaThreadSynchronize ();

    //gpuErrchk(cudaMemcpy(dst, plainDst, x * y * ((z/2) + 1) * sizeof(cufftComplex),cudaMemcpyDeviceToHost));

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

float* ImgHelperCuda::ifft3D(cufftComplex* src, int x, int y, int z)
{
    //cufftComplex* plainSrc;
    float* plainDst;

    //gpuErrchk(cudaMalloc(&plainSrc, x * y * ((z/2) + 1) * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc(&plainDst, x * y * z * sizeof(float)));
    //gpuErrchk(cudaMemcpy(plainSrc,src,x * y * ((z/2) + 1) * sizeof(cufftComplex),cudaMemcpyHostToDevice));

    cufftHandle handle;
    cufftResult r = cufftPlan3d(&handle,x,y,z,CUFFT_C2R);
    CheckCufftError(r, "cufftPlan3d");

    r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    CheckCufftError(r, "cufftSetCompatibilityMode");

    r = cufftExecC2R(handle,src,plainDst);
    CheckCufftError(r, "cufftExecZ2D");
    cudaThreadSynchronize ();

    float* dst = (float*)malloc(x * y * z * sizeof(float));
    gpuErrchk(cudaMemcpy(dst,plainDst,x * y * z * sizeof(float),cudaMemcpyDeviceToHost));

    r = cufftDestroy(handle);
    CheckCufftError(r, "cufftDestroy");

    //cudaFree(plainSrc);
    //cudaFree(plainDst);

    return dst;
}


//this call could be avoided if we add this logic into the method ""ProcessNorm_intern"
__global__
void Process2DHT_intern(cufftComplex* src, int threshold, int windowSize)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int pos = i * windowSize + j;
    if(pos < windowSize * windowSize)
    {
        //avoid if (perf)!!! - GPU branching!!!
        if(src[pos].x < 0 && (src[pos].x * -1) < threshold ) { src[pos].x = 0; src[pos].y = 0; }
        if(src[pos].x > 0 && src[pos].x < threshold) { src[pos].x = 0; src[pos].y = 0; }
    }
}

__global__
void ProcessNorm_intern(cufftComplex* src, float* normVector, int size, int windowSize, int blockSize)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < size * size)
    {
        int hBlock = blockSize / 2;
        int pos = ((i % size) * blockSize) + (int(i/size) * size * windowSize);

        float norm = 0;
        for(int k =0; k < hBlock; ++k)
        {
            for(int n=0; n < blockSize; ++n)
            {
                norm += src[pos].x * src[pos].x + src[pos].y + src[pos].y;
                pos += n;
            }
            pos += windowSize;
        }
        normVector[i] = norm;
    }
}

__global__
void ProcessMatching_intern(cufftComplex* src, int16_t* matching, int size, int windowSize, int blockSize, int threshold)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < size * size)
    {
        int hBlock = blockSize / 2;
        int val1 = (i % size) * blockSize;
        int val2 = int(i/size) * (size - 1) * windowSize;
        int pos = val1 + val2;

        int matchingIndex = i * blockSize;
        int matchingOffset = 0;
        matching[matchingIndex+matchingOffset] = i;
        ++matchingOffset;
        int divisor = blockSize * blockSize;
        int loop = size * size;
        for(int k= 0; k < loop; ++k)
        {
            if(k != i)
            {
                int pos2 = ((k % size) * blockSize) + (int(k/size) * (size - 1)  * windowSize);
                int pos2s = pos2;
                int pos1 = pos;
                double norm = 0;
                for(int m =0; m < hBlock -1; ++m)
                {
                    for(int n=0; n < blockSize; ++n)
                    {
                        double diff = fabs(src[pos1].x) - fabs(src[pos2].x);
                        norm += (diff * diff);
                        pos1 += n;
                        pos2 += n;
                    }
                    pos1 += windowSize;
                    pos2 += windowSize;
                }
                norm = sqrt(norm);

                double distance = norm / divisor;
                if(distance < threshold)
                {
                    matching[matchingIndex+matchingOffset] = k;
                    if(matchingOffset < blockSize -1) ++matchingOffset;
                }
                else
                {
                }
                /*
                int x1 = pos % windowSize;
                int y1 = int(pos / windowSize);
                int x2 = pos2s % windowSize;
                int y2 = int(pos2s / windowSize);
                printf("\n\tCPB %d,%d, pos1 = %d, pos2 = %d, (%d,%d), (%d,%d), norm = %f, distance = %f, val1 = %d, val2 = %d",
                        i,k, pos, pos2s, x1, y1, x2, y2, norm, distance, val1, val2);
                */
            }
        }
    }
}

__global__
void Process3DHT_intern(cufftComplex* src, int threshold, int windowSize)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int k = (blockIdx.z * blockDim.z) + threadIdx.z;
    int pos = j * windowSize + k * windowSize + i;
    if(pos < windowSize * windowSize * ((windowSize/2) +1))
    {
        /*if(pos == 0)
        {
            int size = windowSize * windowSize * ((windowSize/2) + 1);
            int line = 0;
            printf("\n%d: ", line);
            for(int i = 0; i< size; ++i)
            {
                printf("%f, ", src[i].x);
                if(i % windowSize == windowSize -1) { ++line; printf("\n%d: ", line); }
            }
        }*/
        if(fabs(src[pos].x) < threshold ) { src[pos].x = 0; src[pos].y = 0; }
        else {}
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

void ImgHelperCuda::Process3DHT(cufftComplex* src, int windowSize)
{

    dim3 threadsPerBlock(ImgHelperCuda::HT_THREADS, ImgHelperCuda::HT_THREADS, ImgHelperCuda::HT_THREADS);
    dim3 numBlocks(windowSize/threadsPerBlock.x, windowSize/threadsPerBlock.y, windowSize/threadsPerBlock.z);

    //printf("\n\tprocess 3D HT");
    Process3DHT_intern<<<numBlocks,threadsPerBlock>>>(src, ImgHelperCuda::HT_3D_THRESHOLD, windowSize);
    cudaThreadSynchronize();
}

int16_t* ImgHelperCuda::ProcessBM(cufftComplex* src, int threshold, int windowSize, int blockSize)
{
    //dim3 threadsPerBlock(ImgHelperCuda::HT_THREADS, ImgHelperCuda::HT_THREADS);
    //dim3 numBlocks(windowSize/threadsPerBlock.x, windowSize/threadsPerBlock.y);

    //printf("\n\tprocess 2D HT");
    //Process2DHT_intern<<<numBlocks,threadsPerBlock>>>(src, threshold, windowSize);
    //cudaThreadSynchronize();

    int sizeNormVector = (windowSize / blockSize) ;

    /*printf("\n\tComputes blocks value");
    float* normVector_d;
    int sizeNormVector = (windowSize / blockSize) ;
    dim3 threadsPerBlockMatching(sizeNormVector * sizeNormVector);
    dim3 numBlocksMatching(1);
    gpuErrchk(cudaMalloc(&normVector_d, sizeNormVector * sizeNormVector * sizeof(float)));
    ProcessNorm_intern<<<numBlocksMatching,threadsPerBlockMatching>>>(src, normVector_d, sizeNormVector, windowSize, blockSize);
    cudaThreadSynchronize();

    //Only for testing-----
    float* normVector_h = (float*)malloc(sizeNormVector * sizeNormVector * sizeof(float));
    gpuErrchk(cudaMemcpy(normVector_h,normVector_d, sizeNormVector * sizeNormVector * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\n\n----- BLOCKS VALUE (TEST) ------\n");
    for(int i= 0; i < sizeNormVector * sizeNormVector; ++i)
    {
        printf("B%i: %f\n", i, normVector_h[i] );
    }
    printf("\n");
    //---------------------
    */

    dim3 threadsPerBlockMatching(sizeNormVector * sizeNormVector);
    dim3 numBlocksMatching(1);

    printf("\n\tMatching - 3D groups");
    int16_t* matching_d;
    gpuErrchk(cudaMalloc(&matching_d, sizeNormVector * sizeNormVector * blockSize * sizeof(int16_t)));
    gpuErrchk(cudaMemset(matching_d, -1, sizeNormVector * sizeNormVector * blockSize * sizeof(int16_t)));
    ProcessMatching_intern<<<numBlocksMatching,threadsPerBlockMatching>>>(src, matching_d, sizeNormVector, windowSize, blockSize, threshold);
    cudaThreadSynchronize();

    int16_t* matching = (int16_t*)malloc(sizeNormVector * sizeNormVector * blockSize * sizeof(int16_t));
    gpuErrchk(cudaMemcpy(matching,matching_d, sizeNormVector * sizeNormVector * blockSize * sizeof(int16_t), cudaMemcpyDeviceToHost));

    return matching;
}




