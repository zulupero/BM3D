CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(bm3d_gpu)

SET(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-7.0/")
FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(/usr/local/cuda/include)
#set(OPENCV "/home/cuenat/BM3D/BM3D/install_opencv")
#set(INCLUDE_OPENCV "${OPENCV}/include")
#include_directories ( "${INCLUDE_OPENCV}")
#set(FFTW "/home/cuenat/BM3D/BM3D/BM3D_IPOL/fftw-3.3.4/bin")
#set(INCLUDE_FFTW "${FFTW}/include")
#include_directories ( "${INCLUDE_FFTW}")


FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c" "*.h")
CUDA_ADD_EXECUTABLE(bm3d_gpu ${SOURCES})
#ADD_EXECUTABLE(bm3d_cpu ${SOURCES})

LIST(APPEND CMAKE_CXX_FLAGS "-g -o0 -std=c++0x  -ffast-math -Wall") 
#-O3

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -g -G -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

#TARGET_LINK_LIBRARIES(bm3d_cpu 
#		      /usr/local/cuda-6.5/lib64/libcudart.so
#		      /usr/local/cuda-6.5/lib64/libcufft.so
#		      ${OPENCV}/lib/libopencv_highgui.so
#		      ${OPENCV}/lib/libopencv_core.so
#		      ${OPENCV}/lib/libopencv_imgcodecs.so
#		     )

TARGET_LINK_LIBRARIES(bm3d_gpu 
		      -lpng
		      /usr/local/cuda-7.0/lib64/libcufft.so
		     )





