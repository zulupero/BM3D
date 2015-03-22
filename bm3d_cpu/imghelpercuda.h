#ifndef IMGHELPERCUDA_H
#include <cufft.h>
#endif // IMGHELPERCUDA_H

class ImgHelperCuda
{
    public:
        ~ImgHelperCuda()
        {}

        static void transform2D(cufftReal* image, int x, int y);

    private:
        ImgHelperCuda();

};
