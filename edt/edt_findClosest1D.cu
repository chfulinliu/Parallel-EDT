#include "edt_mine_impl.cuh"


__device__ inline short2 edt_findClosestLR_core32(int val, int pred) {
	const int lane = threadIdx.x & 0x1F;
	const unsigned int mask = 1 << lane;
	const unsigned int mask_left = mask - 1 + mask;
	const unsigned int mask_right = FULL_MASK - (mask - 1);
	int voted_x = __ballot_sync(FULL_MASK, pred);
	int masked_idx_left = mask_left & voted_x;
	int masked_idx_right = mask_right & voted_x;
	int count_zeros_left = __clz(masked_idx_left);
	int first_one_right = __ffs(masked_idx_right);
	int closest_p_left = count_zeros_left < WARP_SIZE ?
		WARP_SIZE - 1 - count_zeros_left : 0;
	int closest_p_right = first_one_right > 0 ?
		first_one_right - 1 : WARP_SIZE - 1;
	int closest_idx_left = __shfl_sync(FULL_MASK, val, closest_p_left);
	int closest_idx_right = __shfl_sync(FULL_MASK, val, closest_p_right);
	return make_short2(closest_idx_left, closest_idx_right);
}
__device__ inline int edt_findClosestL_core32(int val, int pred) {
	const int lane = threadIdx.x & 0x1F;
	const unsigned int mask = 1 << lane;
	const unsigned int mask_left = mask - 1 + mask;
	int voted_x = __ballot_sync(FULL_MASK, pred);
	int masked_idx_left = mask_left & voted_x;
	int count_zeros_left = __clz(masked_idx_left);
	int closest_p_left = count_zeros_left < WARP_SIZE ?
		WARP_SIZE - 1 - count_zeros_left : 0;
	int closest_idx_left = __shfl_sync(FULL_MASK, val, closest_p_left);
	return closest_idx_left;
}

__device__ inline int edt_findClosestL_core32(int val) {
	return edt_findClosestL_core32(val, val != INVALID_ROW_SITE);
}

__device__ inline int edt_findClosestR_core32(int val, int pred) {
	const int lane = threadIdx.x & 0x1F;
	const unsigned int mask = 1 << lane;
	const unsigned int mask_right = FULL_MASK - (mask - 1);
	int voted_x = __ballot_sync(FULL_MASK, pred);
	int masked_idx_right = mask_right & voted_x;
	int first_one_right = __ffs(masked_idx_right);
	int closest_p_right = first_one_right > 0 ?
		first_one_right - 1 : WARP_SIZE - 1;
	int closest_idx_right = __shfl_sync(FULL_MASK, val, closest_p_right);
	return closest_idx_right;
}

__device__ inline int edt_findClosestR_core32(int val) {
	return edt_findClosestR_core32(val, val != INVALID_ROW_SITE);
}

__device__ inline short2 edt_findClosestLR_core32(short2 myLR) {
	const int lane = threadIdx.x & 0x1F;
	const unsigned int mask = 1 << lane;
	const unsigned int mask_left = mask - 1 + mask;
	const unsigned int mask_right = FULL_MASK - (mask - 1);
	int idx_left = myLR.x;
	int idx_right = myLR.y;
	int voted_idx_left = __ballot_sync(FULL_MASK, idx_left != INVALID_ROW_SITE);
	int voted_idx_right = __ballot_sync(FULL_MASK, idx_right != INVALID_ROW_SITE);
	int masked_idx_left = mask_left & voted_idx_left;
	int masked_idx_right = mask_right & voted_idx_right;
	int count_zeros_left = __clz(masked_idx_left);
	int first_one_right = __ffs(masked_idx_right);
	int closest_p_left = count_zeros_left < WARP_SIZE ?
		WARP_SIZE - 1 - count_zeros_left : 0;
	int closest_p_right = first_one_right > 0 ?
		first_one_right - 1 : WARP_SIZE - 1;
	int closest_idx_left = __shfl_sync(FULL_MASK, idx_left, closest_p_left);
	int closest_idx_right = __shfl_sync(FULL_MASK, idx_right, closest_p_right);
	return short2{ (short)closest_idx_left,(short)closest_idx_right };
}
__device__ inline short edt_chooseClosest(short2 closestLR, int myIdx) {
	return abs(closestLR.x - myIdx) < abs(closestLR.y - myIdx) ? closestLR.x : closestLR.y;
}

__device__ inline int edt_chooseClosest(int L, int R, int myIdx) {
	return abs(L - myIdx) < abs(R - myIdx) ? L : R;
}

// output width is padded to be n*32
__global__ void edt_findClosest1D_narrow(const char* _input, int inStride, idx_t* _output, int outStride, int srcWidth) {
	const int lane = threadIdx.x & 0x1F;
	constexpr short2 negLR{ INVALID_ROW_SITE, INVALID_ROW_SITE };
	__shared__ short2 sharedLR[WARP_SIZE];
	if (threadIdx.x < WARP_SIZE) {
		sharedLR[threadIdx.x] = negLR;
	}
	__syncthreads();
	const char* input = ptrAt(blockIdx.y, _input, inStride);
	idx_t* output = ptrAt(blockIdx.y, _output, outStride);
	/////////////////////////////////////////////////////////////////////
	int index = threadIdx.x;
	int isFore = (index < srcWidth&& input[index] != 0);
	int x = isFore ? index : INVALID_ROW_SITE;
	short2 closestLR = edt_findClosestLR_core32(x, isFore);
	if (lane == WARP_SIZE - 1)
		sharedLR[threadIdx.x / WARP_SIZE].x = closestLR.x;
	if (lane == 0)
		sharedLR[threadIdx.x / WARP_SIZE].y = closestLR.y;
	__syncthreads();
	if (threadIdx.x < warpSize)
		sharedLR[threadIdx.x] = edt_findClosestLR_core32(sharedLR[threadIdx.x]);
	__syncthreads();
	int sIdx = threadIdx.x / WARP_SIZE;
	if (sIdx > 0 && closestLR.x == INVALID_ROW_SITE)
		closestLR.x = sharedLR[sIdx - 1].x;
	if (sIdx < WARP_SIZE - 1 && closestLR.y == INVALID_ROW_SITE)
		closestLR.y = sharedLR[sIdx + 1].y;
	if (index < srcWidth)
		output[index] = edt_chooseClosest(closestLR, index);
}


//blockdim.x must be a multiple of 32
__global__ void edt_findClosest1D_middle(const char* _input, int inStride, short* _output, int outStride, int srcWidth) {
	constexpr short2 negLR{ INVALID_ROW_SITE, INVALID_ROW_SITE };
	extern __shared__ short2 closestLR[];
	const char* input = ptrAt(blockIdx.y, _input, inStride);
	short* output = ptrAt(blockIdx.y, _output, outStride);
	for (int bias = 0; bias < srcWidth; bias += blockDim.x) {
		int index = threadIdx.x + bias;
		int inRange = index < srcWidth;
		int isFore = (inRange && input[index] != 0);
		int x = isFore ? index : INVALID_ROW_SITE;
		short2 cLR = edt_findClosestLR_core32(x, isFore);
		if (inRange)
			closestLR[index] = cLR;
	}
	__syncthreads();
	int actThreadCnt = (srcWidth + 31) >> 5;//=(srcWidth+31)/32
	int alvThreadCnt = (actThreadCnt + 31) & 0xFFFFFFE0;	//alive thread count
	if (threadIdx.x < alvThreadCnt) {
		short2 cLR = negLR;
		short rx, ly;
		int idx_left_end, idx_right_end;
		bool active = threadIdx.x < actThreadCnt;
		if (active) {
			idx_left_end = threadIdx.x * 32;
			idx_right_end = min(idx_left_end + 31, srcWidth - 1);
			rx = closestLR[idx_right_end].x;
			ly = closestLR[idx_left_end].y;
			cLR.x = rx;
			cLR.y = ly;
		}
		cLR = edt_findClosestLR_core32(cLR);
		if (active) {
			if (rx == INVALID_ROW_SITE)
				closestLR[idx_right_end].x = cLR.x;
			if (ly == INVALID_ROW_SITE)
				closestLR[idx_left_end].y = cLR.y;
		}
	}
	__syncthreads();
	int innActThreadCnt = (srcWidth + 1023) >> 10;
	if (innActThreadCnt > 1) {
		int innAliThreadCnt = (innActThreadCnt + 31) & 0xFFFFFFE0;
		if (threadIdx.x < innAliThreadCnt) {
			bool active = threadIdx.x < innActThreadCnt;
			short2 cLR = negLR;
			short rx, ly;
			int idx_left_end, idx_right_end;
			if (active) {
				idx_left_end = threadIdx.x * 1024;
				idx_right_end = min(idx_left_end + 1024 - 1, srcWidth - 1);
				rx = closestLR[idx_right_end].x;
				ly = closestLR[idx_left_end].y;
				cLR.x = rx;
				cLR.y = ly;
			}
			cLR = edt_findClosestLR_core32(cLR);
			if (active) {
				if (rx == INVALID_ROW_SITE)
					closestLR[idx_right_end].x = cLR.x;
				if (ly == INVALID_ROW_SITE)
					closestLR[idx_left_end].y = cLR.y;
			}
		}
		__syncthreads();
		if (threadIdx.x < actThreadCnt) {
			int idx_left_end = threadIdx.x * 32;
			int idx_right_end = min(idx_left_end + 31, srcWidth - 1);
			int src_idx_left_side = (idx_left_end & 0xFFFFFC00) - 1;
			int src_idx_right_side = (idx_left_end & 0xFFFFFC00) + 1024;
			if (src_idx_left_side >= 0 && closestLR[idx_right_end].x == INVALID_ROW_SITE)
				closestLR[idx_right_end].x = closestLR[src_idx_left_side].x;
			if (src_idx_right_side < srcWidth && closestLR[idx_left_end].y == INVALID_ROW_SITE)
				closestLR[idx_left_end].y = closestLR[src_idx_right_side].y;
		}
		__syncthreads();
	}
	for (int bias = 0; bias < srcWidth; bias += blockDim.x) {
		int index = threadIdx.x + bias;
		if (index < srcWidth) {
			int src_idx_left_side = (index & 0xFFFFFFE0) - 1;
			int src_idx_right_side = (index & 0xFFFFFFE0) + 32;
			if (src_idx_left_side >= 0 && closestLR[index].x == INVALID_ROW_SITE)
				closestLR[index].x = closestLR[src_idx_left_side].x;
			if (src_idx_right_side < srcWidth && closestLR[index].y == INVALID_ROW_SITE)
				closestLR[index].y = closestLR[src_idx_right_side].y;
			output[index] = edt_chooseClosest(closestLR[index], index);
		}
	}
}


//blockdim.x must be a multiple of 32
__global__ void edt_findClosest1D_wide_findNearestRightPos(const char* _input, int inStride, short* _outNearestPosRightSide, int outStride, int srcWidth) {
	constexpr short negR{ INVALID_ROW_SITE };
	extern __shared__ short closestR[];
	const char* input = ptrAt(blockIdx.y, _input, inStride);
	short* output = ptrAt(blockIdx.y, _outNearestPosRightSide, outStride);
	for (int bias = 0; bias < srcWidth; bias += blockDim.x) {
		int index = threadIdx.x + bias;
		int inRange = index < srcWidth;
		int isFore = (inRange && input[index] != 0);
		int x = isFore ? index : INVALID_ROW_SITE;
		int cR = edt_findClosestR_core32(x, isFore);
		if (inRange)
			closestR[index] = cR;
	}
	__syncthreads();
	int actThreadCnt = (srcWidth + 31) >> 5;//=(srcWidth+31)/32
	int alvThreadCnt = (actThreadCnt + 31) & 0xFFFFFFE0;	//alive thread count
	if (threadIdx.x < alvThreadCnt) {
		short cR = negR;
		short ly;
		int idx_left_end;
		bool active = threadIdx.x < actThreadCnt;
		if (active) {
			idx_left_end = threadIdx.x * 32;
			ly = closestR[idx_left_end];
			cR = ly;
		}
		cR = edt_findClosestR_core32(cR);
		if (active) {
			if (ly == INVALID_ROW_SITE)
				closestR[idx_left_end] = cR;
		}
	}
	__syncthreads();
	int innActThreadCnt = (srcWidth + 1023) >> 10;
	if (innActThreadCnt > 1) {
		int innAliThreadCnt = (innActThreadCnt + 31) & 0xFFFFFFE0;
		if (threadIdx.x < innAliThreadCnt) {
			bool active = threadIdx.x < innActThreadCnt;
			short cR = negR;
			short ly;
			int idx_left_end;
			if (active) {
				idx_left_end = threadIdx.x * 1024;
				ly = closestR[idx_left_end];
				cR = ly;
			}
			cR = edt_findClosestR_core32(cR);
			if (active) {
				if (ly == INVALID_ROW_SITE)
					closestR[idx_left_end] = cR;
			}
		}
		__syncthreads();
		if (threadIdx.x < actThreadCnt) {
			int idx_left_end = threadIdx.x * 32;
			int src_idx_right_side = (idx_left_end & 0xFFFFFC00) + 1024;
			if (src_idx_right_side < srcWidth && closestR[idx_left_end] == INVALID_ROW_SITE)
				closestR[idx_left_end] = closestR[src_idx_right_side];
		}
		__syncthreads();
	}
	for (int bias = 0; bias < srcWidth; bias += blockDim.x) {
		int index = threadIdx.x + bias;
		if (index < srcWidth) {
			int src_idx_right_side = (index & 0xFFFFFFE0) + 32;
			int myR = closestR[index];
			if (src_idx_right_side < srcWidth && myR == INVALID_ROW_SITE)
				myR = closestR[src_idx_right_side];
			output[index] = myR;
		}
	}
}

//blockdim.x must be a multiple of 32
__global__ void edt_findClosest1D_wide_findNearestPos(const char* _input, int inStride, short* inNearestRight_outNearest, int outStride, int srcWidth) {
	constexpr short negL{ INVALID_ROW_SITE };
	extern __shared__ short closestL[];
	const char* input = ptrAt(blockIdx.y, _input, inStride);
	short* inout = ptrAt(blockIdx.y, inNearestRight_outNearest, outStride);
	for (int bias = 0; bias < srcWidth; bias += blockDim.x) {
		int index = threadIdx.x + bias;
		int inRange = index < srcWidth;
		int isFore = (inRange && input[index] != 0);
		int x = isFore ? index : INVALID_ROW_SITE;
		short cL = edt_findClosestL_core32(x, isFore);
		if (inRange)
			closestL[index] = cL;
	}
	__syncthreads();
	int actThreadCnt = (srcWidth + 31) >> 5;//=(srcWidth+31)/32
	int alvThreadCnt = (actThreadCnt + 31) & 0xFFFFFFE0;	//alive thread count
	if (threadIdx.x < alvThreadCnt) {
		short cL = negL;
		short rx;
		int idx_left_end, idx_right_end;
		bool active = threadIdx.x < actThreadCnt;
		if (active) {
			idx_left_end = threadIdx.x * 32;
			idx_right_end = min(idx_left_end + 31, srcWidth - 1);
			rx = closestL[idx_right_end];
			cL = rx;
		}
		cL = edt_findClosestL_core32(cL);
		if (active) {
			if (rx == INVALID_ROW_SITE)
				closestL[idx_right_end] = cL;
		}
	}
	__syncthreads();
	int innActThreadCnt = (srcWidth + 1023) >> 10;
	if (innActThreadCnt > 1) {
		int innAliThreadCnt = (innActThreadCnt + 31) & 0xFFFFFFE0;
		if (threadIdx.x < innAliThreadCnt) {
			bool active = threadIdx.x < innActThreadCnt;
			short cL = negL;
			short rx/*, ly*/;
			int idx_left_end, idx_right_end;
			if (active) {
				idx_left_end = threadIdx.x * 1024;
				idx_right_end = min(idx_left_end + 1024 - 1, srcWidth - 1);
				rx = closestL[idx_right_end];
				cL = rx;
			}
			cL = edt_findClosestL_core32(cL);
			if (active) {
				if (rx == INVALID_ROW_SITE)
					closestL[idx_right_end] = cL;
			}
		}
		__syncthreads();
		if (threadIdx.x < actThreadCnt) {
			int idx_left_end = threadIdx.x * 32;
			int idx_right_end = min(idx_left_end + 31, srcWidth - 1);
			int src_idx_left_side = (idx_left_end & 0xFFFFFC00) - 1;
			if (src_idx_left_side >= 0 && closestL[idx_right_end] == INVALID_ROW_SITE)
				closestL[idx_right_end] = closestL[src_idx_left_side];
		}
		__syncthreads();
	}
	for (int bias = 0; bias < srcWidth; bias += blockDim.x) {
		int index = threadIdx.x + bias;
		if (index < srcWidth) {
			int myL = closestL[index];
			int myR = inout[index];
			int src_idx_left_side = (index & 0xFFFFFFE0) - 1;
			if (src_idx_left_side >= 0 && myL == INVALID_ROW_SITE)
				myL = closestL[src_idx_left_side];
			inout[index] = edt_chooseClosest(myL, myR, index);
		}
	}
}
