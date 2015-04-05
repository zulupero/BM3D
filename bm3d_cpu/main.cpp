#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "bm.h"
#include "imghelper.h"

#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc < 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    bool debug = false;
    if(argc >= 3)
    {
        debug = true;
    }

    int WINDOW_SIZE = 40;

    BlockMatch bm(WINDOW_SIZE);
    bm.setDebugMode(debug);
    bm.processWindowBM(&image);

    //TEST
    //ImgHelper::transform2DCuda(&image);

    //imwrite("../BM3D_images/test.jpg", image);
    return 0;
}
