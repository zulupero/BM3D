#!/bin/sh
rm test.png
 ./bm3d_gpu ../BM3D_images/ImNoisy.png > out.txt
less out.txt
