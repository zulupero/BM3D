#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int main()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for(int i=0; i < nDevices; ++i)
    {
	cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);
        printf("Device number: %d\n", i);
        printf("Device name: %s\n", prop.name);
        printf("\tMemory clock rate: %d\n", prop.memoryClockRate);
        printf("\tMemory bus with (bits): %d\n", prop.memoryBusWidth);
        printf("\tPeak memory bandwith (Gb/s): %f\n", 2.0 * prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	printf("\tmax threads per block: %d\n", prop.maxThreadsPerBlock);
	printf("\ttotal Global Memory: %u\n", prop.totalGlobalMem);
	printf("\tWarp size: %d", prop.warpSize); 
    }
}
