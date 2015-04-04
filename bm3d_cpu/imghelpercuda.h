#ifndef IMGHELPERCUDA_H
#include <cufft.h>
#endif // IMGHELPERCUDA_H

class ImgHelperCuda
{
    public:
        ~ImgHelperCuda()
        {}

        static void fft(float* src, cufftComplex* dst, int width, int height);
        static void ifft(cufftComplex* src, float* dst, int width, int height);

    private:
        ImgHelperCuda();

        static void CheckCufftError(cufftResult result, const char* method);
        static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
};
