/*
Author: Fulin Liu (fulin.liu@hotmail.com)

File Name: edt.h

============================================================================
MIT License

Copyright(c) 2021 Fulin Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this softwareand associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright noticeand this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
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
protected:
    idx_t* rowSites_dev = nullptr;
    idx2_t* stkNodes_dev = nullptr;
    idx2_t* stkNodeTr_closest2dTr_dev = nullptr;
    cudaStream_t st = 0;
    int rowSiteStride, stkNodeStride, closestTrStride;
    int inputWidth, inputHeight;
    int bufWidth, bufHeight; 
    int wideInputThreshold, highInputThreshold;
    cudaDeviceProp devProp;
protected:
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

