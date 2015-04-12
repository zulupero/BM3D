#ifndef IMGHELPERCUDA_H
#define IMGHELPERCUDA_H
#include <cufft.h>


class ImgHelperCuda
{
    public:
        ~ImgHelperCuda()
        {}

        static void Process2DHT(cufftComplex* src, int gamma);

        static void fft(float* src, cufftComplex* dst, int width, int height);
        static cufftComplex* fft2(float* src, int width, int height);
        static void ifft(cufftComplex* src, float* dst, int width, int height);
        static float* ifft2(cufftComplex* src, int width, int height);

    private:
        ImgHelperCuda();

        static void CheckCufftError(cufftResult result, const char* method);
        static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
};
#endif // IMGHELPERCUDA_H
