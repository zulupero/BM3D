#!/bin/sh
rm ../BM3D_images/test.jpg
./bm3d_cpu ../BM3D_images/image_F16_512rgb.png 
ls -l ../BM3D_images/test*.*

