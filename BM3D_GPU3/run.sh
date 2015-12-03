#!/bin/sh
./bm3d_gpu ../BM3D_images/house.png ../BM3D_images/ImNoisy.png $1 $2 $3 $4
mv basic.png outputs/house_basic.png
mv final.png outputs/house_final.png
./bm3d_gpu ../BM3D_images/Lena512.png ../BM3D_images/Lena512_noi_s30.png $1 $2 $3 $4
mv basic.png outputs/lena_basic.png
mv final.png outputs/lena_final.png

