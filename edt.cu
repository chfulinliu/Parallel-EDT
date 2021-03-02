/*
Author: Fulin Liu 

File Name: edt.cu

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

#include <cuda_runtime.h>
#include <algorithm>
#include "edt.h"
#include "edt_kernels.cuh"


#ifndef __CUDACC__
//<<<gridDim, blockDim, smemSize=0, stream4Region=0>>>
#define cudaExecCFG(...)
#else
#define cudaExecCFG(...) <<< __VA_ARGS__ >>>
#endif // __CUDACC__


template<int den>
static constexpr int div_ceil(int num) {
    static_assert(den > 0, "Den <=0 ");
    return (num + den - 1) / den;
}

static int div_ceil(int num, int den) {
    return (num + den - 1) / den;
}

template<int grain>
static constexpr int rnd_up(int val) {
    static_assert(grain > 0, "Grain <= 0");
    return (val + grain - 1) / grain * grain;
}

static constexpr int rnd_up(int val, int grain) {
    return (val + grain - 1) / grain * grain;
}

template<int grain>
static constexpr int rnd_down(int val) {
    static_assert(grain > 0, "Grain <= 0");
    return val / grain * grain;
}

static constexpr int rnd_down(int val, int grain) {
    return val / grain * grain;
}

template<typename T>
static void reallocDev(T& ptr, size_t newSz) {
    if (ptr)
        cudaFree(ptr);
    cudaMalloc(&ptr, newSz);
}

template<typename T>
static void freeDev(T& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

size_t edt::requiredMemSize(int maxWidth, int maxHeight) {
    size_t bufPixCnt = size_t(rnd_up<32>(maxWidth)) * rnd_up<32>(maxHeight);
    return bufPixCnt * (sizeof(idx_t) + sizeof(idx2_t) + sizeof(idx2_t));
}

idx2_t edt::maxAllowedSize(int cudaDevice) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cudaDevice);
    int x = std::min((int)(prop.sharedMemPerBlock / sizeof(idx_t)), SHRT_MAX);
    int y = std::min((int)(prop.sharedMemPerBlock / sizeof(idx_t)), SHRT_MAX);
    return idx2_t{ (idx_t)x, (idx_t)y };
}

void edt::setup(int width, int height) {
    int dev;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&devProp, dev);
    // buf sizes are aligned to 32, but we do not stick to this alignment in following process

    bufWidth = rnd_up<32>(width);
    bufHeight = rnd_up<32>(height);

    wideInputThreshold = int(devProp.sharedMemPerBlock / sizeof(idx2_t));
    highInputThreshold = int(devProp.sharedMemPerBlock / sizeof(idx2_t));

    size_t bufPixCnt = size_t(bufWidth) * bufHeight;
    reallocDev(rowSites_dev, bufPixCnt * sizeof(idx_t));
    reallocDev(stkNodes_dev, bufPixCnt * sizeof(idx2_t));
    reallocDev(stkNodeTr_closest2dTr_dev, bufPixCnt * sizeof(idx2_t));
}

void edt::release() {
    freeDev(rowSites_dev);
    freeDev(stkNodes_dev);
    freeDev(stkNodeTr_closest2dTr_dev);
}

edt::~edt() {
    release();
}

void edt::transform(const char* input_dev, int inputStride,
    idx2_t* closestIdx_dev, int idxStride, float* distance, int distStride,
    int width, int height, cudaStream_t _st) {
    prepareInternalVariables(width, height, _st);
    findClosest(input_dev, inputStride);
    writeClosestAndDistance(closestIdx_dev, idxStride, distance, distStride);
}

void edt::transform(const char* input_dev, int inputStride, idx2_t* closestIdx_dev, int idxStride, int width, int height, cudaStream_t _st) {
    prepareInternalVariables(width, height, _st);
    findClosest(input_dev, inputStride);
    writeClosest(closestIdx_dev, idxStride);
}

void edt::transform(const char* input_dev, int inputStride, float* distance, int distStride, int width, int height, cudaStream_t _st) {
    prepareInternalVariables(width, height, _st);
    findClosest(input_dev, inputStride);
    writeDistance(distance, distStride);
}

void edt::prepareInternalVariables(int width, int height, cudaStream_t _st) {
    inputWidth = width;
    inputHeight = height;

    rowSiteStride = rnd_up<32>(inputWidth) * sizeof(idx_t);
    stkNodeStride = rnd_up<32>(inputWidth) * sizeof(idx2_t);
    closestTrStride = rnd_up<32>(inputHeight) * sizeof(idx2_t);

    st = _st;   // save stream for later use
}

void edt::findClosest(const char* input_dev, int inputStride) {
    if (inputWidth > wideInputThreshold)
        find1DSites_wide(input_dev, inputStride);
    else if (inputWidth > 256)
        find1DSites_middle(input_dev, inputStride);
    else
        find1DSites_narrow(input_dev, inputStride);

    makeStacks(32);

    if (inputHeight > highInputThreshold)
        find2DSites_high();
    else if (inputHeight > 128)
        find2DSites_middle();
    else
        find2DSites_low();
}

void edt::find1DSites_narrow(const char* input_dev, int inputStride) {
    dim3 blkDim(rnd_up<32>(inputWidth));
    dim3 grdDim(1, inputHeight);
    edt_findClosest1D_narrow cudaExecCFG(grdDim, blkDim, 0, st) (input_dev, inputStride, rowSites_dev, rowSiteStride, inputWidth);
}

void edt::find1DSites_middle(const char* input_dev, int inputStride) {
    int minThreadCntPerBlock = std::max((inputWidth + 31) / 32, 64);
    int smemSize = inputWidth * sizeof(idx2_t);
    int maxBlockCntPerSM = (int)devProp.sharedMemPerBlock / smemSize;
    int recMinThreadCntPerBlock = devProp.maxThreadsPerBlock / maxBlockCntPerSM;
    int blockDim = std::max(minThreadCntPerBlock, recMinThreadCntPerBlock);
    dim3 blkDim(rnd_up<32>(blockDim));
    dim3 grdDim(1, inputHeight);
    edt_findClosest1D_middle cudaExecCFG(grdDim, blkDim, smemSize, st) (input_dev, inputStride, rowSites_dev, rowSiteStride, inputWidth);
}

void edt::find1DSites_wide(const char* input_dev, int inputStride) {
    int minThreadCntPerBlock = std::max((inputWidth + 31) / 32, 64);
    int smemSize = inputWidth * sizeof(idx_t);
    int maxBlockCntPerSM = (int)devProp.sharedMemPerBlock / smemSize;
    int recMinThreadCntPerBlock = devProp.maxThreadsPerBlock / maxBlockCntPerSM;
    int blockDim = std::max(minThreadCntPerBlock, recMinThreadCntPerBlock);
    dim3 blkDim(rnd_up<32>(blockDim));
    dim3 grdDim(1, inputHeight);

    edt_findClosest1D_wide_findNearestRightPos cudaExecCFG(grdDim, blkDim, smemSize, st) (input_dev, inputStride, rowSites_dev, rowSiteStride, inputWidth);
    edt_findClosest1D_wide_findNearestPos cudaExecCFG(grdDim, blkDim, smemSize, st) (input_dev, inputStride, rowSites_dev, rowSiteStride, inputWidth);
}

void edt::makeStacks(int initialStackSize) {
    constexpr int tileSize = 32;    // this value must be same as TS in cuda kernal function
    dim3 createBlkDim(tileSize, tileSize / initialStackSize);
    dim3 createGrdDim(std::max(1, div_ceil<tileSize>(inputWidth)), std::max(1, div_ceil<tileSize>(inputHeight)));
    // Merging row stacks is much slower than merging column stacks. So we create and merge column stacks and then transpose it.
    /*
    int numStacks = div_ceil(inputHeight, initialStackSize);
    dim3 mergeBlkDim(std::max(1, numStacks / 2), 1);
    dim3 mergeGrdDim(1, std::max(1, div_ceil(inputWidth, mergeBlkDim.y)));
    edt_mergeRowStacks cudaExecCFG(mergeGrdDim, mergeBlkDim, 0, st) (stkNodes_dev, stkNodeStride, inputHeight, inputWidth, rowSites_dev, rowSiteStride, initialStackSize);
    edt_creatRowStacks cudaExecCFG(createGrdDim, createBlkDim, 0, st) (rowSites_dev, rowSiteStride, stkNodes_dev, stkNodeStride, initialStackSize, inputWidth, inputHeight);
    */
    edt_creatColStacks cudaExecCFG(createGrdDim, createBlkDim, 0, st) (rowSites_dev, rowSiteStride, stkNodes_dev, stkNodeStride, initialStackSize, inputWidth, inputHeight);

    int numStacks = div_ceil(inputHeight, initialStackSize);
    dim3 mergeBlkDim(32, 1);
    dim3 mergeGrdDim(div_ceil(inputWidth, mergeBlkDim.x), numStacks / 2);
    int currentStkSize = initialStackSize;
    constexpr int switchThreshold = 1;
    while (numStacks > switchThreshold) {
        edt_mergeColStacks cudaExecCFG(mergeGrdDim, mergeBlkDim, 0, st) (stkNodes_dev, stkNodeStride, inputWidth, inputHeight, rowSites_dev, rowSiteStride, currentStkSize);
        currentStkSize += currentStkSize;
        numStacks = div_ceil(inputHeight, currentStkSize);
        mergeGrdDim.y = numStacks / 2;
    }
    // This part is designed to decrease kernal call count, but it consumes more GPU time when image height exceeds 1024, so it is disabled by set switchThreshold to 1;
    if (numStacks > 1) {
        mergeBlkDim = dim3(32, numStacks / 2);
        mergeGrdDim = dim3(div_ceil(inputWidth, mergeBlkDim.x), 1);
        edt_mergeColStacks_low cudaExecCFG(mergeGrdDim, mergeBlkDim, 0, st) (stkNodes_dev, stkNodeStride, inputWidth, inputHeight, rowSites_dev, rowSiteStride, currentStkSize);
    }
    dim3 blkDim(16, 16);
    dim3 grdDim(div_ceil<16>(inputWidth), div_ceil<16>(inputHeight));
    // A fast transpose with out bank conflict, but it is always a little bit slower than NPP implementation. Here we still use this version to avoid NPP dependency.
    edt_transpose32bit cudaExecCFG(grdDim, blkDim, 0, st)((int32_t*)stkNodes_dev, stkNodeStride, (int32_t*)stkNodeTr_closest2dTr_dev, closestTrStride, inputWidth, inputHeight);
}

void edt::find2DSites_low() {
    dim3 blkDim(rnd_up<32>(inputHeight));
    dim3 grdDim(1, inputWidth);
    int fcSmemSz = blkDim.x * sizeof(idx2_t);
    edt_findClosest2D_low cudaExecCFG(grdDim, blkDim, fcSmemSz, st) (stkNodeTr_closest2dTr_dev, closestTrStride, rowSites_dev, rowSiteStride, inputWidth, inputHeight);
}

void edt::find2DSites_middle() {
    int minThreadCntPerBlock = std::max((inputHeight + 31) / 32, 64);
    int smemSize = inputHeight * sizeof(idx2_t);
    int maxBlockCntPerSM = (int)devProp.sharedMemPerBlock / smemSize;
    int recMinThreadCntPerBlock = devProp.maxThreadsPerBlock / maxBlockCntPerSM;
    int blockDim = std::max(minThreadCntPerBlock, recMinThreadCntPerBlock);
    dim3 blkDim(rnd_up<32>(blockDim));
    dim3 grdDim(1, inputWidth);
    edt_findClosest2D_middle cudaExecCFG(grdDim, blkDim, smemSize, st) (stkNodeTr_closest2dTr_dev, closestTrStride, rowSites_dev, rowSiteStride, inputWidth, inputHeight);
}

void edt::find2DSites_high() {
    int minThreadCntPerBlock = std::max((inputHeight + 31) / 32, 64);
    int smemSize = inputHeight * sizeof(idx_t);
    int maxBlockCntPerSM = (int)devProp.sharedMemPerBlock / smemSize;
    int recMinThreadCntPerBlock = devProp.maxThreadsPerBlock / maxBlockCntPerSM;
    int blockDim = std::max(minThreadCntPerBlock, recMinThreadCntPerBlock);
    dim3 blkDim(rnd_up<32>(blockDim));
    dim3 grdDim(1, inputWidth);
    edt_findClosest2D_high cudaExecCFG(grdDim, blkDim, smemSize, st) (stkNodeTr_closest2dTr_dev, closestTrStride, rowSites_dev, rowSiteStride, inputWidth, inputHeight);
}

void edt::writeClosestAndDistance(idx2_t* closestIdx_dev, int idxStride, float* distance, int distStride) {
    dim3 blkDim(16, 16);
    dim3 grdDim(std::max(1, div_ceil<16>(inputHeight)), std::max(1, div_ceil<16>(inputWidth)));
    edt_writeClosestAndDistance cudaExecCFG(grdDim, blkDim, 0, st) (stkNodeTr_closest2dTr_dev, closestTrStride, closestIdx_dev, idxStride, distance, distStride, inputWidth, inputHeight);
}

void edt::writeClosest(idx2_t* closestIdx_dev, int idxStride) {
    dim3 blkDim(16, 16);
    dim3 grdDim(std::max(1, div_ceil<16>(inputHeight)), std::max(1, div_ceil<16>(inputWidth)));
    edt_writeClosest cudaExecCFG(grdDim, blkDim, 0, st) (stkNodeTr_closest2dTr_dev, closestTrStride, closestIdx_dev, idxStride, inputWidth, inputHeight);
}

void edt::writeDistance(float* distance, int distStride) {
    dim3 blkDim(16, 16);
    dim3 grdDim(std::max(1, div_ceil<16>(inputHeight)), std::max(1, div_ceil<16>(inputWidth)));
    edt_writeDistance cudaExecCFG(grdDim, blkDim, 0, st) (stkNodeTr_closest2dTr_dev, closestTrStride, distance, distStride, inputWidth, inputHeight);
}

