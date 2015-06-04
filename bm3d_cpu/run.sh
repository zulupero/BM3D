#!/bin/sh
clear
rm in*.txt
#rm ../BM3D_images/test.jpg
#rm ../BM3D_images/test2.jpg
./bm3d_cpu ../BM3D_images/Lena512_noi_s90.png $1 
#ls -l ../BM3D_images/test*.*


