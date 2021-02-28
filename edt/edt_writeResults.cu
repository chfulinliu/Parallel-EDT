/*
Author: Fulin Liu 

File Name: edt_writeResults.cu

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

#include "edt_kernels_impl.cuh"

//blockDim.x MUST equasl to 16
//blockDim.y MUST equals to 16
//a block processes a tile

__global__ void edt_writeClosestAndDistance(const idx2_t* closestTr, int closestTrStride, idx2_t* closest_idx, int closestStride, float* distances, int distStride, const int dstWidth, const int dstHeight) {
	constexpr int blkSz = 16;
	__shared__ idx2_t tile[blkSz][blkSz + 1];
	int inRow = blockIdx.y * blockDim.y + threadIdx.y;
	int inCol = blockIdx.x * blockDim.x + threadIdx.x;
	if (inRow < dstWidth && inCol < dstHeight)
		tile[threadIdx.y][threadIdx.x] = *ptrAt(inRow, inCol, closestTr, closestTrStride);
	__syncthreads();
	int outRow = blockIdx.x * blockDim.x + threadIdx.y;
	int outCol = blockIdx.y * blockDim.y + threadIdx.x;
	if (outRow < dstHeight && outCol < dstWidth) {
		idx2_t site = tile[threadIdx.x][threadIdx.y];
		float dist;
		if (site.x >= 0) {
			float dx = site.x - outCol;
			float dy = site.y - outRow;
			dist = sqrt(dx * dx + dy * dy);
		}
		else
			dist = FLT_MAX;
		*ptrAt(outRow, outCol, closest_idx, closestStride) = site;
		*ptrAt(outRow, outCol, distances, distStride) = dist;
	}
}

__global__ void edt_writeClosest(const idx2_t* closestTr, int closestTrStride, idx2_t* closest_idx, int closestStride, const int dstWidth, const int dstHeight) {
	constexpr int blkSz = 16;
	__shared__ idx2_t tile[blkSz][blkSz + 1];
	int inRow = blockIdx.y * blockDim.y + threadIdx.y;
	int inCol = blockIdx.x * blockDim.x + threadIdx.x;
	if (inRow < dstWidth && inCol < dstHeight)
		tile[threadIdx.y][threadIdx.x] = *ptrAt(inRow, inCol, closestTr, closestTrStride);
	__syncthreads();
	int outRow = blockIdx.x * blockDim.x + threadIdx.y;
	int outCol = blockIdx.y * blockDim.y + threadIdx.x;
	if (outRow < dstHeight && outCol < dstWidth) {
		*ptrAt(outRow, outCol, closest_idx, closestStride) = tile[threadIdx.x][threadIdx.y];
	}
}

__global__ void edt_writeDistance(const idx2_t* closestTr, int closestTrStride, float* distances, int distStride, const int dstWidth, const int dstHeight) {
	constexpr int blkSz = 16;
	__shared__ idx2_t tile[blkSz][blkSz + 1];
	int inRow = blockIdx.y * blockDim.y + threadIdx.y;
	int inCol = blockIdx.x * blockDim.x + threadIdx.x;
	if (inRow < dstWidth && inCol < dstHeight)
		tile[threadIdx.y][threadIdx.x] = *ptrAt(inRow, inCol, closestTr, closestTrStride);
	__syncthreads();
	int outRow = blockIdx.x * blockDim.x + threadIdx.y;
	int outCol = blockIdx.y * blockDim.y + threadIdx.x;
	if (outRow < dstHeight && outCol < dstWidth) {
		idx2_t site = tile[threadIdx.x][threadIdx.y];
		float dist;
		if (site.x >= 0) {
			float dx = site.x - outCol;
			float dy = site.y - outRow;
			dist = sqrt(dx * dx + dy * dy);
		}
		else
			dist = FLT_MAX;
		*ptrAt(outRow, outCol, distances, distStride) = dist;
	}
}
