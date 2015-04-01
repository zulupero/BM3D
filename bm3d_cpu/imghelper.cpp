#include "imghelper.h"
#include "imghelpercuda.h"
#include <vector>
#include <iostream>
#include <cufft.h>
using namespace std;

void ImgHelper::transform2D(Mat* image)
{

    vector<Mat> planes;
    split(*image, planes);

    vector<Mat> outplanes(planes.size());
    for(size_t i= 0; i<planes.size(); ++i)
    {
        Size s = planes[i].size();
        cout << "NORMAL: "<<s.width << "," << s.height << std::endl;
        planes[i].convertTo(planes[i], CV_32FC1);
        for(int k = 0; k< 5; ++k) cout << planes[i].at<float>(0,k) <<",";
        cout << endl;
        dct(planes[i], outplanes[i]);
        for(int k = 0; k< 5; ++k) cout << outplanes[i].at<float>(0,k) <<",";
        cout << endl;
    }

    merge(outplanes, *image);
}

void ImgHelper::transform2DCuda(Mat* image)
{
    vector<Mat> planes;
    split(*image, planes);

    vector<Mat> outplanes(planes.size());
    for(size_t i= 0; i<planes.size(); ++i)
    {
        Size s = planes[i].size();
        cout << "CUDA: " << s.width << "," << s.height << std::endl;
        planes[i].convertTo(planes[i], CV_32F);


        int N1 = 64;
        //cufftReal* data =  (cufftReal*)malloc(N1 * N1 * sizeof(cufftReal));
        float* data;//(cufftReal*)malloc(N1 * N1 * sizeof(float));
        std::vector<float> array;
        array.assign((float*)planes[i].datastart, (float*)planes[i].dataend);
        //data = (cufftReal*)&array[0];
        data = &array[0];

        //int outX = 0;
        //int outY = 0;
        cout << "FORWARD" << endl;
        //cufftComplex* out = ImgHelperCuda::Transform2D(data, N1, N1, &outX, &outY);
        //cufftComplex* out = ImgHelperCuda::Transform2DTest(data, N1, N1);
        cufftComplex* out= (cufftComplex*)malloc( N1 * N1 * sizeof(cufftComplex));
        ImgHelperCuda::fft_device(data, out, N1, N1, N1 * sizeof(float), N1 * sizeof(cufftComplex));
        cout << endl << endl << "INVERSE" << endl;
        //cufftReal* out2 = ImgHelperCuda::Inversetransform2D(out, outX, outY, &outX, &outY);
        //cufftReal* out2 = ImgHelperCuda::InverseTransform2DTest(out, N1, N1);
        float* out2 = (float*)malloc( N1 * N1 * sizeof(double));
        ImgHelperCuda::fft_inverse_device(out, out2, N1, N1, N1 * sizeof(cufftComplex), N1 * sizeof(float));

        float* out3 = (float*)malloc( N1 * N1 * sizeof(float));
        for(int i=0;i< N1; ++i)
            for(int k=0; k<N1;++k)
                out3[i*N1+k]=out[i*N1+k].x;

        int sizes[] = {N1, N1};
        outplanes[i] = Mat(2, sizes, CV_32F, (float*)out3);

        free(out2);
        free(out3);
        free(out);

    }
    merge(outplanes, *image);
}


