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
       //TODO: replace "float" by "double"
	   public: 
		int img_width;
	  	int img_height;
	   	int pHard;
		int nHard; 
	   	SourceImage sourceImage;		
		int nbBlocks_total;
        int nbBlocks;
        int nbBlocksPerLine;
        int nbBlocksPerLine_total;        
        bool debugMode; 
        
        //Device (member-variables):
        float* deviceImage;     
        float* dctCosParam1;    
        float* dctCosParam2;    
        float* idctCosParam1;   
        float* idctCosParam2;   
        float* cArray;          
        float* deviceBlocks3D;  
        float* finalBlocks3D;   
        int* bmVectorsComplete; 
        int* bmVectors;         
        int* blockMap;          
        float* blocks;          
        float* dctBlocks;       
        int* pixelMap;          
        int* pixelMapIndex;     
        int* blockGroupMap;     
        int* blockGroupIndex;   
        float* wpArray;         
        int* distanceArray;  
        int* npArray;   
        float* basicImage;
        float* basicValues;
        float* kaiserWindowCoef;
	};

private:	
	static BM3D_Context context;	

public:
	static void BM3D_Initialize(SourceImage img, int width, int height, int pHard, int nhard, bool debug = false);	
	static void BM3D_Run();
	~BM3D()
	{}

private:
	static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
	static void BM3D_CreateBlocks();
    static void BM3D_CreateBlockMap();
    static void BM3D_2DDCT();
    static void BM3D_2DDCT2();
    static void BM3D_2DiDCT();
    static void BM3D_2DiDCT2();
	static void BM3D_BasicEstimate();
    static void BM3D_PrepareDCT(float* cosParam1, float* cosParam2);
    static void BM3D_PrepareiDCT(float* cosParam1, float* cosParam2);
    static void BM3D_PrepareCArray(float* cArray);
    static void BM3D_BlockMatching();
    static void BM3D_BlockMatching2();
    static void BM3D_HTFilter();
	static void BM3D_InverseTransform();
    static void BM3D_CalculateWPArray();
    static void BM3D_Aggregation();
    static void BM3D_dispose();
    static void BM3D_SaveBasicImage();

	BM3D();
	BM3D operator=(BM3D& bm3d);
};

#endif
