#include "imghelper.h"
#include "imghelpercuda.h"
#include <iostream>

#include <fstream>
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

void ImgHelper::getWindowBuffer(int x, int y, float* buffer, Mat image, int wSize)
{
    int offsetY = y;
    int offsetX = x;
    for(int i=0; i< wSize * wSize; ++i)
    {
        int mod = i % wSize;
        offsetX += mod;
        if(mod == 0) { ++offsetY; }
        offsetX += x;
        buffer[i] = image.at<float>(x+offsetX, offsetY);
    }
}

void ImgHelper::transform2DCuda(Mat* image)
{
    vector<Mat> planes;
    split(*image, planes);

    //vector<Mat> outplanes(planes.size());
    for(size_t i= 0; i<planes.size(); ++i)
    {
        //Size s = planes[i].size();
        planes[i].convertTo(planes[i], CV_32FC1);

        int WINDOW_SIZE = 40;
        float* windowBuffer = (float*)malloc(WINDOW_SIZE * WINDOW_SIZE * sizeof(float));
        //getWindowBuffer(0, 0, windowBuffer, planes[i], WINDOW_SIZE);
        //writeMatToFile(windowBuffer, "in_5.txt", WINDOW_SIZE, WINDOW_SIZE);
        free(windowBuffer);

        int N1 = 8;
        cout << "CUFFT: " << N1 << "," << N1 << std::endl;
        writeMatToFile(planes[i], "in.txt", N1, N1);

        float* data = (float*)malloc( N1 * N1 * sizeof(float));
        memset(data, 0, N1 * N1 * sizeof(float));
        for(int j=0; j< N1; ++j)
            for(int k=0; k< N1; ++k)
                data[j* N1 + k] = planes[i].at<float>(j, k);

        writeMatToFile(data, "in_2.txt", N1, N1);

        cout << "FORWARD..." << endl;
        cufftComplex* out= (cufftComplex*)malloc( N1 * ((N1/2)+1) * sizeof(cufftComplex));
        ImgHelperCuda::fft(data, out, N1, N1);

        writeComplexMatToFile(out, "in_3.txt", N1, ((N1/2) + 1));

        /*outplanes[i] = Mat(N1, N1, CV_32FC1);
        for(int j=0; j< N1; ++j)
            for(int k=0; k< N1; ++k)
                outplanes[i].at<float>(j, k) = out[j*N1 + k].x;
        */

        cout << "INVERSE..." << endl;
        float* out2 = (float*)malloc( N1 * N1 * sizeof(float));
        ImgHelperCuda::ifft(out, out2, N1, N1);

        float divisor = N1 * N1;
        for(int j=0; j< N1 * N1; ++j)
            out2[j] = out2[j] / divisor;

        writeMatToFile(out2, "in_4.txt", N1, N1);
        free(out2);
        free(out);
    }
    //merge(outplanes, *image);
}

void ImgHelper::writeMatToFile(cv::Mat& m, const char* filename, int x, int y)
{
    ofstream fout(filename);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<x; i++)
    {
        for(int j=0; j<y; j++)
        {
            fout<<m.at<double>(i,j)<<"\t";
        }
        fout<<endl;
    }

    fout.close();
}

void ImgHelper::writeMatToFile(float* data, const char* filename, int x, int y)
{
    ofstream fout(filename);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<x; i++)
    {
        for(int j=0; j<y; j++)
        {
            fout<<data[i*x+j]<<"\t";
        }
        fout<<endl;
    }

    fout.close();
}

void ImgHelper::writeComplexMatToFile(cufftComplex* data, const char* filename, int x, int y)
{
    ofstream fout(filename);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<y; i++)
    {
        for(int j=0; j<x; j++)
        {
            fout<<data[i*x+j].x<<"\t";
        }
        fout<<endl;
    }

    fout.close();
}


