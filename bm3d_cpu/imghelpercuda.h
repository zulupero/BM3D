#ifndef IMGHELPERCUDA_H
#include <cufft.h>
#endif // IMGHELPERCUDA_H

class ImgHelperCuda
{
    public:
        ~ImgHelperCuda()
        {}

        static cufftComplex* Transform2D(cufftReal* data, int x, int y, int* outX, int* outY);
        static cufftReal* Inversetransform2D(cufftComplex* data, int x, int y, int* outX, int* outY);
        static void ApplyHardThresholdFilter(cufftComplex* data);
        static cufftComplex* Transform2DTest(cufftReal* data, int x, int y);
        static cufftReal* InverseTransform2DTest(cufftComplex* data, int x, int y);
        static void fft_device_double(double* src, cufftDoubleComplex* dst, int width, int height, int srcPitch, int dstPitch);
        static void fft_inverse_device_double(cufftDoubleComplex* src, double* dst, int width, int height, int srcPitch, int dstPitch);
        static void fft_device(float* src, cufftComplex* dst, int width, int height, int srcPitch, int dstPitch);
        static void fft_inverse_device(cufftComplex* src, float* dst, int width, int height, int srcPitch, int dstPitch);

    private:
        ImgHelperCuda();

        static void CheckCufftError(cufftResult result, const char* method);
        static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
};
