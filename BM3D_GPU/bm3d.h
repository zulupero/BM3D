#ifndef BM3D_H
#define BM3D_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

class BM3D
{
public:
	typedef std::vector<float> SourceImage;

	class BM3D_Context 
	{
	   public: 
		int img_width;
	  	int img_height;
	   	int pHard;
		int nHard; 
	   	SourceImage sourceImage;
		float* deviceImage;
		float* deviceBlocks;
	};

private:	
	static BM3D_Context context;	

public:
	static void BM3D_Initialize(SourceImage img, int width, int height, int pHard, int nhard);	
	static void BM3D_Run();
	~BM3D()
	{}

private:
	static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false);
	static void BM3D_CreateBlocks();
	static void BM3D_BasicEstimate();
	

	BM3D();
	BM3D operator=(BM3D& bm3d);
};

#endif
