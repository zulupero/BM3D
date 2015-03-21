cmake_minimum_required(VERSION 2.8)
project (BM3D_CPU)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x -Wall -Wextra -Werror")
set(OPENCV "/home/cuenat/BM3D/install_opencv")
set(INCLUDE_OPENCV "${OPENCV}/include")
include_directories ( "${INCLUDE_OPENCV}")

set(bm3d_cpu 
    main.cpp )

add_executable (bm3d_cpu ${bm3d_cpu})

target_link_libraries(bm3d_cpu
		      ${OPENCV}/lib/libopencv_highgui.so
		      ${OPENCV}/lib/libopencv_core.so
		      ${OPENCV}/lib/libopencv_imgcodecs.so
		     )






