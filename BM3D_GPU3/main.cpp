#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>

#include "utilities.h"
#include "timeutil.h"
#include "bm3d.h"

#include <string>

int main(int argc, char **argv)
{
    BM3D::SourceImage img;
    BM3D::SourceImage imgOrig;
    unsigned width, height, chnls;

    if(argc < 8)
    {
	printf("\ntoo few arguments\n");
        return EXIT_FAILURE;
    }

    //! Load image
    printf("Orig Image: %s\n", argv[1]);
    printf("Noised Image: %s\n", argv[2]);
    printf("Sigma: %s\n", argv[3]);
    printf("Window size: %s blocks\n", argv[8]);
    int sigma = atoi(argv[3]);
    float HTThreshold = atof(argv[4]);
    int distanceWienThreshold = atoi(argv[5]);
    int distanceHTThreshold = atoi(argv[6]);
    int step = atoi(argv[7]);
    int windowSize = atoi(argv[8]);
    
    if(load_image(argv[2], img, &width, &height, &chnls) != EXIT_SUCCESS) return EXIT_FAILURE;
    //Hardacoded for test purposes
    //std::string imagePath("../BM3D_images/ImNoisy.png");
    //if(load_image(imagePath.c_str(), img, &width, &height, &chnls) != EXIT_SUCCESS) return EXIT_FAILURE;
    //std::string imageOrig("../BM3D_images/house.png");
    //if(load_image(imageOrig.c_str(), imgOrig, &width, &height, &chnls) != EXIT_SUCCESS) return EXIT_FAILURE;
    if(load_image(argv[1], imgOrig, &width, &height, &chnls) != EXIT_SUCCESS) return EXIT_FAILURE;
    
    Timer::start();
    //sigma = 25;
    
    BM3D::BM3D_Initialize(img, imgOrig, width, height, step, (step * distanceHTThreshold * 64), (3 *distanceWienThreshold * 64), (sigma * HTThreshold), sigma, windowSize, false);  
    //phard = 3, hard limit = 2500 * 64, Wien limit = 400 * 64,  hard threshol = sigma * 2.7

    Timer::add("BM3D-Initialization");
    BM3D::BM3D_Run();
    Timer::showResults();

    printf("\n");
    return EXIT_SUCCESS;
}
