#ifndef IMGHELPERCUDA_H
#define IMGHELPERCUDA_H
#include <cufft.h>


class ImgHelperCuda
{
    public:
        ~ImgHelperCuda()
        {}

        static int16_t* ProcessBM(cufftComplex* src, int threshold, int windowSize, int blockSize);
        static void Process3DHT(cufftComplex* src, int windowSize);
        static void fft(float* src, cufftComplex* dst, int width, int height);
        static cufftComplex* fft2(float* src, int width, int height);
        static void ifft(cufftComplex* src, float* dst, int width, int height);
        static float* ifft2(cufftComplex* src, int width, int height);
        static cufftComplex* get(cufftComplex* src, int width, int height);
        static float* get(float* src, int width, int height);
        static cufftComplex* fft3D(float* src, int x, int y, int z);
        static float* ifft3D(cufftComplex* src, int x, int y, int z);

    private:
        ImgHelperCuda();

        static void CheckCufftError(cufftResult result, const char* method);
        static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

        static const int HT_THREADS = 8;
        static const int HT_3D_THRESHOLD = 243; ///2.7 * gamma (here gamma is equal to 90)
};
#endif // IMGHELPERCUDA_H
