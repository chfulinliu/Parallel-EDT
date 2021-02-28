#include "edt_mine_impl.cuh"

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
