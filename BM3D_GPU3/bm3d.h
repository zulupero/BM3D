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
        int             debugPixel;
        int             debugBlock;
		int             img_widthOrig;
	  	int             img_heightOrig;
        int             img_width;
	  	int             img_height;
	   	int             pHard;
	   	SourceImage     sourceImage;		
        int             nbBlocksIntern;        
        int             nbBlocks;
        int             widthBlocksIntern;
        int             widthBlocks;
        bool            debugMode; 
        int             hardLimit;
        
        //Device (member-variables):
        float*      deviceImage;
        float*      basicImage;     
        float*      kaiserWindowCoef;
        int*        blockMap;
        double*     blocks;
        int*        bmVectors;
        double*     blocks3D;
	};

private:	
	static BM3D_Context context;	

public:
	static void BM3D_Initialize(SourceImage img, int width, int height, int pHard, int hardLimit, bool debug = false);	
	static void BM3D_Run();
	~BM3D()
	{}

private:
	static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
    static void BM3D_dispose();
    static void BM3D_SaveBasicImage();
    static void BM3D_BasicEstimate();
    static void BM3D_CreateBlock();
    static void BM3D_2DTransform();
    static void BM3D_BlockMatching();

    static void BM3D_ShowBlock(int x, int y);
    static void BM3D_ShowDistance(int x, int y);

	BM3D();
	BM3D operator=(BM3D& bm3d);
};

#endif
