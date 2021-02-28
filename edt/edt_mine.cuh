//by fulin.liu@hotmail.com
#pragma once
#include <cinttypes>
#include <vector_types.h>
#include <cuda_runtime.h>
using idx_t = short;
using idx2_t = short2;
using stk_node_t = idx2_t;


__global__ void edt_findClosest1D_narrow(const char* _input, int inStride, idx_t* _output, int outStride, int srcWidth);
__global__ void edt_findClosest1D_middle(const char* _input, int inStride, idx_t* _output, int outStride, int srcWidth);
__global__ void edt_findClosest1D_wide_findNearestRightPos(const char* _input, int inStride, short* _outNearestPosRightSide, int outStride, int srcWidth);
__global__ void edt_findClosest1D_wide_findNearestPos(const char* _input, int inStride, short* inNearestRight_outNearest, int outStride, int srcWidth);

__global__ void edt_creatColStacks(const idx_t* closestPerRow, int closestPerRowStride, stk_node_t* stk_nodes, int stk_node_stride, int maxStkSz, int srcImgWidth, int srcImgHeight);
__global__ void edt_mergeColStacks(stk_node_t* stkNodes, int stkNodeStride, int stkNodeWidth, int stkNodeHeight, const idx_t* sites, int siteStride, int single_stk_sz);
__global__ void edt_mergeColStacks_low(stk_node_t* stkNodes, int stkNodeStride, int stkNodeWidth, int stkNodeHeight, const idx_t* sites, int siteStride, int single_stk_sz);

__global__ void edt_transpose32bit(const int32_t* in, int inStride, int32_t* out, int outStride, const int width_in, const int height_in);

__global__ void edt_findClosest2D_low(idx2_t* in_nodes_out_closest, int nodeStride, const idx_t* rowSites, int rowSiteStride, int srcImgWidth, int srcImgHeight);
__global__ void edt_findClosest2D_middle(idx2_t* in_nodes_out_closest, int nodeStride, const idx_t* rowSites, int rowSiteStride, int srcImgWidth, int srcImgHeight);
__global__ void edt_findClosest2D_high(idx2_t* in_nodes_out_closest, int nodeStride, const idx_t* rowSites, int rowSiteStride, int srcImgWidth, int srcImgHeight);

__global__ void edt_writeClosestAndDistance(const idx2_t* closestTr, int closestTrStride, idx2_t* closest, int closestStride, float* distances, int distStride, const int dstWidth, const int dstHeight);
__global__ void edt_writeClosest(const idx2_t* closestTr, int closestTrStride, idx2_t* closest, int closestStride, const int dstWidth, const int dstHeight);
__global__ void edt_writeDistance(const idx2_t* closestTr, int closestTrStride, float* distances, int distStride, const int dstWidth, const int dstHeight);


// depreated functions
#if 0
__global__ void edt_creatRowStacks(const idx_t* rowSites, int rowSiteStride, stk_node_t* stk_nodes, int stk_node_stride, int innerBlkHeight, int srcImgWidth, int srcImgHeight);
__global__ void edt_mergeRowStacks(stk_node_t* stkNodes, int stkNodeStride, int stkNodeWidth, int stkNodeHeight, const idx_t* sites, int siteStride, int single_stk_sz);
#endif
