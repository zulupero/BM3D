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
        SourceImage     origImage;		        
        int             nbBlocks;
        int             nbBlocksIntern;
        int             nbBlocksPerWindow;
        int             widthBlocksIntern;
        int             widthBlocksWindow;
        bool            debugMode; 
        int             hardLimit;
        int             wienLimit;
        double          hardThreshold;
        int             sigma;
        int             windowSize;
        int             offset;
        int             halfWindowSize;
        int             windowSize;
        
        //Device (member-variables):
        float*      deviceImage;
        float*      basicImage;     
        int*        windowMap;
        double*     blocks3D;
        double*     blocks3DOrig;
        int*        npArray;
        float*      wpArray;
        float*      estimates;
        int*        nbSimilarBlocks;
	};

private:	
	static BM3D_Context context;	

public:
	static void BM3D_Initialize(SourceImage img, SourceImage imgOrig, int width, int height, int pHard, int hardLimit, int wienLimit, double hardThreshold, int sigma, int windowSize, bool debug = false);	
	static void BM3D_Run();
	~BM3D()
	{}

private:
	static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
    static void BM3D_dispose();
    static void BM3D_SaveImage(bool final = false);
    static void BM3D_BasicEstimate();
    static void BM3D_FinalEstimate();
    static void BM3D_CreateWindow();
    static void BM3D_2DTransform(bool final = false);
    static void BM3D_2DTransform2(bool final = false);
    static void BM3D_BlockMatching(bool final = false);
    static void BM3D_HardThresholdFilter();
    static void BM3D_WienFilter();
    static void BM3D_Inverse3D(bool final=false);
    static void BM3D_Inverse3D2(bool final = false);
    static void BM3D_Aggregation(bool final=false);
    static void BM3D_InverseShift();
    static void BM3D_Create3DBlocks(bool final=false);

    static void BM3D_ShowBlock(int block);
    static void BM3D_ShowDistance(int block, bool vectorOnly = false);
    static void BM3D_ShowPixel(int x, int y);
    static void BM3D_ShowBlock3D(int block, bool orig = false);

	BM3D();
	BM3D operator=(BM3D& bm3d);
};

#endif
