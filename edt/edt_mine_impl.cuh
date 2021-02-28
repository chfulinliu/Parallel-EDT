//by fulin.liu@hotmail.com
#include "edt_mine.cuh"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <float.h>
#ifdef max
#undef max
#endif // max
#ifdef min
#undef min
#endif // min

#ifndef __CUDACC__
//some function definations to evict error hints of visual studio
template<typename T, typename Tt, typename Tx, typename Ty>
T tex2D(Tt tex, Tx x, Ty y) { return T(); }
template<typename T, typename Tt, typename Tx, typename Ty, typename Tm = int>
T surf2Dread(Tt tex, Tx x, Ty y, Tm m = 0) { return T(); }
template<typename T>
T atomicAdd(T* pt, T v) { return T(); }
template<typename T>
T atomicMax(T* pt, T v) { return T(); }
template<typename T>
T atomicMin(T* pt, T v) { return T(); }
template<typename T>
T __ldg(const T* p) { return T(); }
template<typename Ta, typename Tb>
Ta max(Ta a, Tb b) { return Ta(); }
template<typename Ta, typename Tb>
Ta min(Ta a, Tb b) { return Ta(); }
template<typename T>
T abs(T a) { return T(); }
template<typename T>
T sqrt(T a) { return T(); }
template<typename T>
int __clz(T v) { return T(); }
template<typename T>
int __ffs(T v) { return T(); }
template<typename Tm, typename Tp>
int __ballot_sync(Tm, Tp p) { return 0; }
template<typename Tm, typename Tv, typename Ti>
Tv __shfl_sync(Tm m, Tv v, Ti i) { return Tv(); }
void __syncthreads() {}
void __syncwarp() {}
#endif


/*
* Although valid row sites are all positive, we can not define INVALID_ROW_SITE as arbitrary negative number.
* INVALID_ROW_SITE will fill positions where we cannot find a nearby active pixel on its left/right.
* So, this value is involved in the edt_chooseClosest function.
* Only SHRT_MIN can guarantee a proper position site is chosen.
* Because distances between any positive short numbers are less than that between positive short number and SHRT_MIN.
*/
#define INVALID_ROW_SITE SHRT_MIN
/*
* INVALID_NODE_PTR_NO_SITE and INVALID_NODE_PTR_IS_SITE can be arbitrarily chosen from negative short numbers, as long as they are not equal.
*/
//#define INVALID_NODE_PTR_NO_SITE (-2)
#define INVALID_NODE_PTR_NO_SITE SHRT_MIN
#define INVALID_NODE_PTR_IS_SITE (-1)
#define IS_PTR_INVALID(a) ((a)<0)
#define IS_PTR_VALID(a) ((a)>=0)

#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32

template<typename T>
__device__ T* ptrAt(int row, int col, T* data, int strideInBytes) {
	return (T*)((char*)data + row * strideInBytes) + col;
}
template<typename T>
__device__ T* ptrAt(int row, T* data, int strideInBytes) {
	return (T*)((char*)data + row * strideInBytes);
}
template<typename T>
__device__ T* ptrNxtRow(T* data, int strideInBytes) {
	return (T*)((char*)data + strideInBytes);
}