#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>

#include "utilities.h"
#include "timeutil.h"
#include "bm3d.h"


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
    if(load_image("../BM3D_images/ImNoisy.png", img, &width, &height, &chnls) != EXIT_SUCCESS) return EXIT_FAILURE;

    BM3D::BM3D_Initialize(img, width, height, 3, false);  //phard = 3
    Timer::start();
    BM3D::BM3D_Run();
    Timer::add("BM3D");
    Timer::showResults();

    printf("\n");
    return EXIT_SUCCESS;
}
