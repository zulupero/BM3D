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

void ImgHelperCuda::fft_device_double(double* src, cufftDoubleComplex* dst, int width, int height)
{
    double* plainSrc;
    cufftDoubleComplex* plainDst;

    gpuErrchk(cudaMalloc(&plainSrc, width * height * sizeof(double)));
    gpuErrchk(cudaMalloc(&plainDst,  width * ((height/2) + 1) * sizeof(cufftDoubleComplex)));
    gpuErrchk(cudaMemcpy(plainSrc,src,width * height * sizeof(double),cudaMemcpyHostToDevice));

    cufftHandle handle;
    cufftResult r = cufftPlan2d(&handle,width,height,CUFFT_D2Z);
    CheckCufftError(r, "cufftPlan2d");

    r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    CheckCufftError(r, "cufftSetCompatibilityMode");

    r = cufftExecD2Z(handle,plainSrc,plainDst);
    CheckCufftError(r, "cufftExecD2Z");
    cudaThreadSynchronize ();

    gpuErrchk(cudaMemcpy(dst, plainDst, width * ((height/2) + 1) * sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost));

    r = cufftDestroy(handle);
    CheckCufftError(r, "cufftDestroy");

    cudaFree(plainSrc);
    cudaFree(plainDst);
}

void ImgHelperCuda::fft_inverse_device_double(cufftDoubleComplex* src, double* dst, int width, int height)
{
    cufftDoubleComplex* plainSrc;
    double* plainDst;

    gpuErrchk(cudaMalloc(&plainSrc, width * ((height/2) + 1) * sizeof(cufftDoubleComplex)));
    gpuErrchk(cudaMalloc(&plainDst, width * height * sizeof(double)));
    gpuErrchk(cudaMemcpy(plainSrc,src,width * ((height/2) + 1) * sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice));

    cufftHandle handle;
    cufftResult r = cufftPlan2d(&handle,width,height,CUFFT_Z2D);
    CheckCufftError(r, "cufftPlan2d");

    r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    CheckCufftError(r, "cufftSetCompatibilityMode");

    r = cufftExecZ2D(handle,plainSrc,plainDst);
    CheckCufftError(r, "cufftExecZ2D");
    cudaThreadSynchronize ();

    gpuErrchk(cudaMemcpy(dst,plainDst,width * height * sizeof(double),cudaMemcpyDeviceToHost));

    r = cufftDestroy(handle);
    CheckCufftError(r, "cufftDestroy");

    cudaFree(plainSrc);
    cudaFree(plainDst);
}

void ImgHelperCuda::fft_device(float* src, cufftComplex* dst, int width, int height)
{
    //src and dst are device pointers allocated with cudaMallocPitch

    //Convert them to plain pointers. No padding of rows.
    for(int i = 0; i < 10; ++i)
    {
        printf("%f,", src[i]);
    }
    printf("\n");

    float* plainSrc;
    cufftComplex* plainDst;

    gpuErrchk(cudaMalloc<float>(&plainSrc,width * height * sizeof(float)));
    gpuErrchk(cudaMalloc<cufftComplex>(&plainDst, width * height * sizeof(cufftComplex)));
    gpuErrchk(cudaMemcpy(plainSrc,src,width * height * sizeof(float),cudaMemcpyHostToDevice));

    cufftHandle handle;
    cufftResult r = cufftPlan2d(&handle,width,height,CUFFT_R2C);
    CheckCufftError(r, "cufftPlan2d");

    r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    CheckCufftError(r, "cufftSetCompatibilityMode");

    r = cufftExecR2C(handle,plainSrc,plainDst);
    CheckCufftError(r, "cufftExecR2C");
    cudaThreadSynchronize ();

    gpuErrchk(cudaMemcpy(dst,plainDst,width * height * sizeof(cufftComplex),cudaMemcpyDeviceToHost));

    r = cufftDestroy(handle);
    CheckCufftError(r, "cufftDestroy");

    cudaFree(plainSrc);
    cudaFree(plainDst);

    for(int i = 0; i < 10; ++i)
    {
        printf("%f,", dst[i].x);
    }
    printf("\n");
}

void ImgHelperCuda::fft_inverse_device(cufftComplex* src, float* dst, int width, int height)
{
    //src and dst are device pointers allocated with cudaMallocPitch

    //Convert them to plain pointers. No padding of rows.
    for(int i = 0; i < 10; ++i)
    {
        printf("%f,", src[i].x);
    }
    printf("\n");

    cufftComplex* plainSrc;
    float* plainDst;

    gpuErrchk(cudaMalloc<cufftComplex>(&plainSrc, width * height * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc<float>(&plainDst, width * height * sizeof(float)));
    gpuErrchk(cudaMemcpy(plainSrc,src,width * height * sizeof(cufftComplex),cudaMemcpyHostToDevice));

    cufftHandle handle;
    cufftResult r = cufftPlan2d(&handle,width,height,CUFFT_C2R);
    CheckCufftError(r, "cufftPlan2d");

    r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    CheckCufftError(r, "cufftSetCompatibilityMode");

    r = cufftExecC2R(handle,plainSrc,plainDst);
    CheckCufftError(r, "cufftExecC2R");
    cudaThreadSynchronize ();

    gpuErrchk(cudaMemcpy(dst,plainDst,width * height * sizeof(float),cudaMemcpyDeviceToHost));

    r = cufftDestroy(handle);
    CheckCufftError(r, "cufftDestroy");

    cudaFree(plainSrc);
    cudaFree(plainDst);

    for(int i = 0; i < 10; ++i)
    {
        printf("%f,", dst[i]);
    }
    printf("\n");
}

cufftReal* ImgHelperCuda::InverseTransform2DTest(cufftComplex* data, int x, int y)
{
    for(int i = 0; i < 5; ++i)
    {
        printf("%f,", data[i].x);
    }
    printf("\n");

    cufftReal* mid_h= (cufftReal*)malloc( x * y *sizeof(cufftReal));
    cufftComplex* in_d;
    cufftReal* mid_d;
    gpuErrchk(cudaMalloc((void**) &in_d, x * y * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc((void**)&mid_d, x * y * sizeof(cufftReal)));
    gpuErrchk(cudaMemcpy((cufftComplex*)in_d, (cufftComplex*)data, x * y * sizeof(cufftComplex),cudaMemcpyHostToDevice));

    cufftHandle handle;
    int rank = 2; // 2D fft
    int n[] = {x, y};    // Size of the Fourier transform
    int istride = 1, ostride = 1; // Stride lengths
    int idist = 1, odist = 1;     // Distance between batches
    int inembed[] = {x, y}; // Input size with pitch
    int onembed[] = {x, y}; // Output size with pitch
    int batch = 1;
    cufftResult r = cufftPlanMany(&handle, rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_C2R, batch);
    CheckCufftError(r, "cufftPlanMany");

    r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    CheckCufftError(r, "cufftSetCompatibilityMode");

    r = cufftExecC2R(handle,data,mid_d);
    CheckCufftError(r, "cufftExecC2R");

    gpuErrchk(cudaMemcpy((cufftReal*)mid_h, (cufftReal*)mid_d, x * y * sizeof(cufftReal), cudaMemcpyDeviceToHost));
    for(int i = 0; i < 5; ++i)
    {
        printf("%f,", mid_h[i]);
    }
    printf("\n");

    r = cufftDestroy(handle);
    CheckCufftError(r, "cufftDestroy");

    cudaFree(in_d);
    cudaFree(mid_d);

    return mid_h;
}

cufftComplex* ImgHelperCuda::Transform2DTest(cufftReal* data, int x, int y)
{
    for(int i = 0; i < 5; ++i)
    {
        printf("%f,", data[i]);
    }
    printf("\n");

    size_t hostSize = x * y * sizeof(cufftComplex);
    cufftComplex* mid_h= (cufftComplex*)malloc( hostSize );
    cufftReal* in_d;
    cufftComplex* mid_d;
    gpuErrchk(cudaMalloc((void**) &in_d, x * y * sizeof(cufftReal)));
    gpuErrchk(cudaMalloc((void**)&mid_d, x * y * sizeof(cufftComplex)));
    gpuErrchk(cudaMemcpy((cufftReal*)in_d, (cufftReal*)data, x * y * sizeof(cufftReal),cudaMemcpyHostToDevice));

    cufftHandle handle;
    int rank = 2; // 2D fft
    int n[] = {x, y};    // Size of the Fourier transform
    int istride = 1, ostride = 1; // Stride lengths
    int idist = 1, odist = 1;     // Distance between batches
    int inembed[] = {0}; // Input size with pitch
    int onembed[] = {0}; // Output size with pitch
    int batch = 1;
    cufftResult r;
    /*
    r = cufftPlanMany(&handle, rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_R2C, batch);
    CheckCufftError(r, "cufftPlanMany");
    */

    cufftPlan2d(&handle, x , y, CUFFT_R2C);

    //r = cufftSetCompatibilityMode(handle,CUFFT_COMPATIBILITY_NATIVE);
    //CheckCufftError(r, "cufftSetCompatibility");

    r = cufftExecR2C(handle, data, mid_d);
    CheckCufftError(r, "cufftExecR2C");

    gpuErrchk(cudaMemcpy((cufftComplex*)mid_h, (cufftComplex*)mid_d, x * y * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    for(int i = 0; i < 5; ++i)
    {
        printf("%f,", mid_h[i].x);
    }
    printf("\n");

    r = cufftDestroy(handle);
    CheckCufftError(r, "cufftDestroy");

    cudaFree(in_d);
    cudaFree(mid_d);

    return mid_h;
}

cufftReal* ImgHelperCuda::Inversetransform2D(cufftComplex* data, int x, int y, int* outX, int* outY)
{
    //*outX = x;
    //*outY = y;
    *outX = x;
    *outY = (y -1) * 2;
    cufftComplex *in_d;
    cufftReal *mid_d, *mid_h;
    cufftHandle plan;

    for(int i = 0; i < 5; ++i)
    {
        printf("%f,", data[i].x);
    }
    printf("\n");

    mid_h= (cufftReal*)malloc( (*outX) * (*outY) *sizeof(cufftReal));

    cudaMalloc((void**) &in_d, x * y * sizeof(cufftComplex));
    cudaMalloc((void**)&mid_d, (*outX) * (*outY) * sizeof(cufftReal));

    cufftPlan2d(&plan, x , y, CUFFT_C2R);

    cudaMemcpy((cufftComplex*)in_d, (cufftComplex*)data, x * y * sizeof(cufftComplex),cudaMemcpyHostToDevice);

    cufftExecC2R(plan, (cufftComplex*)in_d, (cufftReal*)mid_d);

    cudaMemcpy((cufftReal*)mid_h, (cufftReal*)mid_d, (*outX) * (*outY) * sizeof(cufftReal), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5; ++i)
    {
        printf("%f,", mid_h[i]);
    }
    printf("\n");

    cufftDestroy(plan);
    cudaFree(in_d);
    cudaFree(mid_d);

    return mid_h;
}

cufftComplex* ImgHelperCuda::Transform2D(cufftReal* data, int x, int y, int* outX, int* outY)
{
    //*outX = x;
    //*outY = y;
    *outX = x;
    *outY = y/2 + 1;
    cufftReal* in_d;
    cufftComplex* mid_d, *mid_h;
    cufftHandle plan;

    for(int i = 0; i < 5; ++i)
    {
        printf("%f,", data[i]);
    }
    printf("\n");

    mid_h= (cufftComplex*)malloc( (*outX) * (*outY) *sizeof(cufftComplex));

    cudaMalloc((void**) &in_d, x * y * sizeof(cufftReal));
    cudaMalloc((void**)&mid_d, (*outX) * (*outY) * sizeof(cufftComplex));

    cufftPlan2d(&plan, x , y, CUFFT_R2C);

    cudaMemcpy((cufftReal*)in_d, (cufftReal*)data, x * y * sizeof(cufftReal),cudaMemcpyHostToDevice);

    cufftExecR2C(plan, (cufftReal*)in_d, (cufftComplex*)mid_d);

    cudaMemcpy((cufftComplex*)mid_h, (cufftComplex*)mid_d, (*outX) * (*outY) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5; ++i)
    {
        printf("%f,", mid_h[i].x);
    }
    printf("\n");

    cufftDestroy(plan);
    cudaFree(in_d);
    cudaFree(mid_d);

    return mid_h;
}
