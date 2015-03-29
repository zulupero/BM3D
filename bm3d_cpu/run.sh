#!/bin/sh
clear
rm ../BM3D_images/test.jpg
rm ../BM3D_images/test2.jpg
make
./bm3d_cpu ../BM3D_images/couple.png 
ls -l ../BM3D_images/test*.*

