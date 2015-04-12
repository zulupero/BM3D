#!/bin/sh
clear
rm in*.txt
#rm ../BM3D_images/test.jpg
#rm ../BM3D_images/test2.jpg
./bm3d_cpu ../BM3D_images/couple.png $1 
#ls -l ../BM3D_images/test*.*
echo "before FFT:"
head in_1.txt
echo "FFT:"
head in_2.txt
echo "2D HT(filter):"
head in_3.txt

