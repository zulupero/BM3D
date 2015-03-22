#include "imghelpercuda.h"
#include <vector>
#include <stdio.h>

void ImgHelperCuda::transform2D(cufftReal* image, int x, int y)
{
    cufftReal *in_h, *in_d;
    cufftComplex *mid_d, *mid_h;
    cufftHandle pF;

    //in_h = (cufftReal*) malloc(x* y * sizeof(cufftReal));
    //memcpy(in_h, image, x * y * sizeof(cufftReal));
    in_h = image;


    //out_h= (cufftReal*) malloc(x * y * sizeof(cufftReal));
    mid_h= (cufftComplex*)malloc(x * y *sizeof(cufftComplex));

    cudaMalloc((void**) &in_d, x * y * sizeof(cufftReal));
    cudaMalloc((void**)&mid_d, x * y * sizeof(cufftComplex));

    cufftPlan2d(&pF, x , y, CUFFT_R2C);

    cudaMemcpy((cufftReal*)in_d, (cufftReal*)in_h, x * y * sizeof(cufftReal),cudaMemcpyHostToDevice);

    cufftExecR2C(pF, (cufftReal*)in_d, (cufftComplex*)mid_d);

    cudaMemcpy((cufftComplex*)mid_h, (cufftComplex*)mid_d, y * x * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for(int i=0; i< x * y; ++i)
    {
        image[i] = mid_h[i].x;
        //printf("%f,", image[i]);
    }
}
