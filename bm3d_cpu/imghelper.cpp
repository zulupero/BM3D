#include "imghelper.h"
#include "imghelpercuda.h"
#include <iostream>

#include <fstream>
using namespace std;

void ImgHelper::setDebugMode(bool debug)
{
    _debug = debug;
}

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

void ImgHelper::getWindowBuffer(int x, int y, float* buffer, Mat image, int wSize, int* outX, int* outY)
{
    Size s = image.size();
    int offsetY = y;
    int offsetX = x;
    int oldOffsetX = 0;
    for(int i=0; i< wSize * wSize; ++i)
    {
        buffer[i] = image.at<float>(offsetY, offsetX);
        ++offsetX;
        if(offsetX == wSize || offsetX == s.width)
        {
            oldOffsetX = offsetX;
            if(offsetX == wSize) offsetX -= wSize;
            else offsetX -= s.width;
            ++offsetY;
            if(offsetY == s.height) break;
        }
    }
    (*outX) = oldOffsetX;
    (*outY) = offsetY;
}

cufftComplex* ImgHelper::fft(float* imageBuffer, int n1)
{
    return ImgHelperCuda::fft2(imageBuffer, n1, n1);
}

float* ImgHelper::ifft(cufftComplex* imageBuffer, int n1)
{
    return ImgHelperCuda::ifft2(imageBuffer, n1, n1);
}

void ImgHelper::transform2DCuda(float* imageBuffer, int n1)
{
    if(_debug) cout << "CUFFT: " << n1 << "," << n1 << std::endl;
    if(_debug) writeMatToFile(imageBuffer, "in.txt", n1, n1);

    if(_debug) cout << "FORWARD..." << endl;
    cufftComplex* out= (cufftComplex*)malloc( n1 * ((n1/2)+1) * sizeof(cufftComplex));
    ImgHelperCuda::fft(imageBuffer, out, n1, n1);

    if(_debug) writeComplexMatToFile(out, "in_2.txt", n1, ((n1/2) + 1));

    if(_debug) cout << "INVERSE..." << endl;
    ImgHelperCuda::ifft(out, imageBuffer, n1, n1);

    float divisor = n1 * n1;
    for(int i=0; i< n1 * n1; ++i)
        imageBuffer[i] = imageBuffer[i] / divisor;

    if(_debug) writeMatToFile(imageBuffer, "in_4.txt", n1, n1);
    free(out);
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
            fout<<m.at<float>(i,j)<<"\t";
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


