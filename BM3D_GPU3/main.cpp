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
    unsigned width, height, chnls;

    if(argc < 2)
    {
	printf("\ntoo few arguments\n");
        return EXIT_FAILURE;
    }

    //! Load image
    printf("Image: %s\n", argv[1]);
    
    //if(load_image(argv[1], img, &width, &height, &chnls) != EXIT_SUCCESS) return EXIT_FAILURE;
    //Hardacoded for test purposes
    std::string imagePath("../BM3D_images/ImNoisy.png");

    if(load_image(imagePath.c_str(), img, &width, &height, &chnls) != EXIT_SUCCESS) return EXIT_FAILURE;
    
    Timer::start();
    BM3D::BM3D_Initialize(img, width, height, 3, (2500 * 64), (30 * 2.7), 30, false);  //phard = 3, hard limit = 2500 * 64, hard threshol = sigma * 2.7, sigma = 30
    Timer::add("BM3D-Initialization");
    BM3D::BM3D_Run();
    Timer::showResults();

    printf("\n");
    return EXIT_SUCCESS;
}