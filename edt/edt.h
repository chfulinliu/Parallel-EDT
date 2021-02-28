//by fulin.liu@hotmail.com
#pragma once
#include <vector_types.h>
#include <cuda_runtime.h>
class edt {
public:
    using idx_t = short;
    using idx2_t = short2;
    void setup(int maxWidth, int maxHeight);
    void transform(const char* input_dev, int inputStride, idx2_t* closestIdx_dev, int idxStride, float* distance, int distStride, int width, int height, cudaStream_t _st);
    void transform(const char* input_dev, int inputStride, idx2_t* closestIdx_dev, int idxStride, int width, int height, cudaStream_t _st);
    void transform(const char* input_dev, int inputStride, float* distance, int distStride, int width, int height, cudaStream_t _st);
    void release();
    ~edt();
    // returns required memory size to process input image of size (maxWidth, maxHeight)
    static size_t requiredMemSize(int maxWidth, int maxHeight);
    // returns the max size of image can be processed, assuming infinite GPU memory, idx2_t.x->width, idx2_t.y->height
    static idx2_t maxAllowedSize(int cudaDevice);
public:
    idx_t* rowSites_dev = nullptr;
    idx2_t* stkNodes_dev = nullptr;
    idx2_t* stkNodeTr_closest2dTr_dev = nullptr;
    cudaStream_t st = 0;
    int rowSiteStride, stkNodeStride, closestTrStride;
    int inputWidth, inputHeight;
    int bufWidth, bufHeight; 
    int wideInputThreshold, highInputThreshold;
    cudaDeviceProp devProp;
private:
    void prepareInternalVariables(int width, int height, cudaStream_t _st);
    void findClosest(const char* input_dev, int inputStride);
    void find1DSites_narrow(const char* input_dev, int inputStride);
    void find1DSites_middle(const char* input_dev, int inputStride);
    void find1DSites_wide(const char* input_dev, int inputStride);
    void makeStacks(int stackSz);  //sz: 4,8,16,32
    void find2DSites_low();
    void find2DSites_middle();
    void find2DSites_high();
    void writeClosestAndDistance(idx2_t* closestIdx_dev, int idxStride, float* distance, int distStride);
    void writeClosest(idx2_t* closestIdx_dev, int idxStride);
    void writeDistance(float* distance, int distStride);
};

