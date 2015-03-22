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
        dct(planes[i], outplanes[i]);
    }

    merge(outplanes, *image);
}

void ImgHelper::transform2DCuda(Mat* image)
{
    /*for(int i=0; i<10; ++i)
    {
        for(int k =0; k<10; ++k)
        {
            cout << image->at<float>(i,k) << ",";
        }
        cout << endl;
    }
    cout << endl;*/

    //vector<cufftReal> array;
    //array.assign(image->datastart, image->dataend);
    //for(int i=0; i<10; ++i) cout << array[i] << ",";
    //cout << endl;
    //Size s = image->size();
    //ImgHelperCuda::transform2D((cufftReal*)image->data, s.width, s.height);
    //Mat* m = new Mat(array);
    //m->copyTo(*image);
    //delete m;

    for(int i=0; i<10; ++i)
    {
        for(int k =0; k<10; ++k)
        {
            cout << image->at<float>(i,k) << ",";
        }
        cout << endl;
    }
    cout << endl;


    vector<Mat> planes;
    split(*image, planes);

    vector<Mat> outplanes(planes.size());
    for(size_t i= 0; i<planes.size(); ++i)
    {
        Size s = planes[i].size();
        cout << "CUDA: " << s.width << "," << s.height << std::endl;
        planes[i].convertTo(planes[i], CV_32FC1);
        //vector<cufftReal> array;
        //array.assign(planes[i].datastart, planes[i].dataend);
        //for(int i=0; i<10; ++i) cout << array[i] << ",";
        //cout << endl;
        //ImgHelperCuda::transform2D(&array[0], s.width, s.height);

        cufftReal* data =  (cufftReal*)malloc(s.width * s.height *sizeof(cufftReal));
        for(int n = 0; n < s.width; ++n)
        {
            for(int k= 0; k < s.height; ++k)
            {
                data[n*s.width+ k] = planes[i].at<float>(n,k);
            }
        }

        ImgHelperCuda::transform2D(data, s.width, s.height);
        //planes[i] = Mat(2, s.width * s.height, CV_32FC1, (float*)&array[0]);

        for(int i=0; i<10; ++i)
        {
            cout << data[i] << ",";
        }
        cout << endl;

        for(int n = 0; n < s.width; ++n)
        {
            for(int k= 0; k < s.height; ++k)
            {
                planes[i].at<float>(n,k) = data[n*s.width+ k];
            }
        }

        free(data);

    }

    for(int i=0; i<10; ++i)
    {
        for(int k =0; k<10; ++k)
        {
            cout << image->at<float>(i,k) << ",";
        }
        cout << endl;
    }
    cout << endl;

    merge(outplanes, *image);
}


