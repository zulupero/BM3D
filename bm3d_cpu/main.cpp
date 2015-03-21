#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv/cv.h"
//#include "opencv/highgui.h"

#include "bm.h"

#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    Mat output;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    vector<Mat> planes;
    split(image, planes);

    vector<Mat> outplanes(planes.size());
    cout <<"input planes type = "<< planes[0].type() << std::endl;

    for(size_t i= 0; i<planes.size(); ++i)
    {
        planes[i].convertTo(planes[i], CV_32FC1);
        dct(planes[i], outplanes[i]);
    }

    Mat merged;
    merge(outplanes, merged);

    imwrite("../BM3D_images/test.jpg", merged);

    cout << "success" << std::endl;                                         // Wait for a keystroke in the window
    return 0;
}
