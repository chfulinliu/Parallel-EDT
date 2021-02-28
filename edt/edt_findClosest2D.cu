#include "edt_mine_impl.cuh"

__device__ inline int ybscross(int xa, int ya, int xb, int yb, int x) {
	int xbma = xb - xa;
	int xbpa = xb + xa;
	int xpx = x + x;
	int ybpa = yb + ya;
	int ybma = yb - ya;
	int num = xbma * (xbpa - xpx) + ybpa * ybma - 1;
	int den = ybma + ybma;
	return num / den;
}

//blockDim.y MUST be 1
__global__ void edt_findClosest2D_low(idx2_t* in_nodes_out_closest, int nodeStride, const idx_t* sites, int siteStride, int srcImgWidth, int srcImgHeight) {
	constexpr idx2_t invalidSite{ -1, -1 };
	const int node_row = blockIdx.y;
	const int node_col = threadIdx.x;
	const int site_col = node_row;
	const int site_row = node_col;
	__shared__ extern idx2_t nearest[];
	__shared__ int shared_right[WARP_SIZE];
	if (threadIdx.x < WARP_SIZE) {
		shared_right[threadIdx.x] = -1;
	}
	nearest[node_col] = invalidSite;
	__syncthreads();
	stk_node_t lstSite{ -1,-1 };
	if (node_col < srcImgHeight) {
		stk_node_t node = *ptrAt(node_row, node_col, in_nodes_out_closest, nodeStride);
		if (node.x != INVALID_NODE_PTR_NO_SITE && node.y != INVALID_NODE_PTR_NO_SITE) {
			int domiIdx = INT_MAX;	// use the largest integer, so that if current site don't have next site, it could dominate the right part of current line
			int myx = *ptrAt(site_row, site_col, sites, siteStride);
			int myy = node_col;
			if (node.x > 0) //have next site
			{
				int nxtx = *ptrAt(node.x, site_col, sites, siteStride);
				int nxty = node.x;
				domiIdx = ybscross(myx, myy, nxtx, nxty, site_col);
			}
			if (domiIdx >= srcImgHeight) {
				lstSite = make_short2(myx, myy);
				if (node.y > 0) // have previous site
				{
					int prevx = *ptrAt(node.y, site_col, sites, siteStride);
					int prevy = node.y;
					int prevDomiIdx = ybscross(myx, myy, prevx, prevy, site_col);
					if (prevDomiIdx >= srcImgHeight)
						lstSite = make_short2(-1, -1);
				}
			}
			else if (domiIdx >= 0)
			{
				nearest[domiIdx] = make_short2(myx, myy);
			}
		}
	}
	__syncthreads();
	//x stores right most dominant index of myself, y stores right most dominant index of my previous site
	if (lstSite.y != -1 && nearest[srcImgHeight - 1].x < 0)
		nearest[srcImgHeight - 1] = lstSite;
	__syncthreads();
	/////////////////////////////////////////////////////////////////////
	const int index = node_col;
	const int lane = threadIdx.x & 0x1F;
	const bool isFore = nearest[index].x != -1;
	const int x = isFore ? index : -1;
	const int voted_x = __ballot_sync(FULL_MASK, isFore);
	unsigned int mask_right = FULL_MASK - ((1 << lane) - 1);
	int masked_x_right = mask_right & voted_x;
	int first_one_right = __ffs(masked_x_right);
	int closest_p_right = first_one_right > 0 ?
		first_one_right - 1 : WARP_SIZE - 1;
	int closest_index_right = __shfl_sync(FULL_MASK, x, closest_p_right);
	if (lane == 0)
		shared_right[threadIdx.x / WARP_SIZE] = closest_index_right;
	__syncthreads();

	if (threadIdx.x < warpSize) {
		int x_right = shared_right[threadIdx.x];
		int voted_x_right = __ballot_sync(FULL_MASK, x_right != -1);
		masked_x_right = mask_right & voted_x_right;
		first_one_right = __ffs(masked_x_right);
		closest_p_right = first_one_right > 0 ?
			first_one_right - 1 : WARP_SIZE - 1;
		shared_right[threadIdx.x] = __shfl_sync(FULL_MASK, x_right, closest_p_right);
	}
	__syncthreads();

	int sIdx = threadIdx.x / WARP_SIZE;
	if (sIdx < WARP_SIZE - 1 && closest_index_right == -1)
		closest_index_right = shared_right[sIdx + 1];
	if (node_col < srcImgHeight)
		*ptrAt(node_row, node_col, in_nodes_out_closest, nodeStride) = closest_index_right >= 0 ?
		nearest[closest_index_right] : invalidSite;
}

__device__ inline int edt_findClosest_middle_prop_core32(int val) {
	const int lane = threadIdx.x & 0x1F;
	const unsigned int mask_right = FULL_MASK - ((1 << lane) - 1);
	int voted_x = __ballot_sync(FULL_MASK, val != -1);
	int masked_idx_right = mask_right & voted_x;
	int first_one_right = __ffs(masked_idx_right);
	int closest_p_right = first_one_right > 0 ?
		first_one_right - 1 : WARP_SIZE - 1;
	int closest_right = __shfl_sync(FULL_MASK, val, closest_p_right);
	return closest_right;
}

__global__ void edt_findClosest2D_middle(idx2_t* in_nodes_out_closest, int nodeStride, const idx_t* sites, int siteStride, int srcImgWidth, int srcImgHeight) {
	const int node_row = blockIdx.y;
	const int site_col = node_row;
	__shared__ extern idx2_t psClosest[];
	idx2_t* pLeftNode = ptrAt(node_row, in_nodes_out_closest, nodeStride);
	for (int bias = 0; bias < srcImgHeight; bias += blockDim.x) {
		int index = bias + threadIdx.x;
		if (index < srcImgHeight) {
			((int*)psClosest)[index] = -1;
		}
	}
	__syncthreads();
	stk_node_t lstSite{ -1,-1 };
	for (int bias = 0; bias < srcImgHeight; bias += blockDim.x) {
		int node_col = threadIdx.x + bias;
		const int site_row = node_col;
		if (node_col < srcImgHeight) {
			stk_node_t node = pLeftNode[node_col];
			if (node.x != INVALID_NODE_PTR_NO_SITE && node.y != INVALID_NODE_PTR_NO_SITE)// this site valid
			{
				int domiIdx = INT_MAX;
				int myx = *ptrAt(site_row, site_col, sites, siteStride);
				int myy = node_col;
				if (node.x > 0) //have next site
				{
					int nxtx = *ptrAt(node.x, site_col, sites, siteStride);
					int nxty = node.x;
					domiIdx = ybscross(myx, myy, nxtx, nxty, site_col);
				}
				if (domiIdx >= srcImgHeight) {
					lstSite = make_short2(myx, myy);
					if (node.y > 0) // have previous site
					{
						int prevx = *ptrAt(node.y, site_col, sites, siteStride);
						int prevy = node.y;
						int prevDomiIdx = ybscross(myx, myy, prevx, prevy, site_col);
						if (prevDomiIdx >= srcImgHeight)
							lstSite = make_short2(-1, -1);
					}
				}
				else if (domiIdx >= 0)
				{
					psClosest[domiIdx] = make_short2(myx, myy);
				}
			}
		}
	}
	__syncthreads();
	if (lstSite.y != -1 && psClosest[srcImgHeight - 1].x < 0)
		psClosest[srcImgHeight - 1] = lstSite;
	__syncthreads();
	const int bundleWidth = srcImgHeight;
	int* closestRight = (int*)psClosest;
	int* bundle = (int*)pLeftNode;
	for (int bias = 0; bias < bundleWidth; bias += blockDim.x) {
		int index = threadIdx.x + bias;
		int inRange = index < bundleWidth;
		int nR = inRange ? closestRight[index] : -1;
		nR = edt_findClosest_middle_prop_core32(nR);
		if (inRange)
			closestRight[index] = nR;
	}
	__syncthreads();
	int actThreadCnt = (bundleWidth + 31) >> 5;//=(srcWidth+31)/32
	int alvThreadCnt = (actThreadCnt + 31) & 0xFFFFFFE0;	//alive thread count
	if (threadIdx.x < alvThreadCnt) {
		int nR = -1;
		int idx_left_end;
		bool active = threadIdx.x < actThreadCnt;
		if (active) {
			idx_left_end = threadIdx.x * 32;
			nR = closestRight[idx_left_end];
		}
		int nnR = edt_findClosest_middle_prop_core32(nR);
		if (active) {
			if (nR == -1)
				closestRight[idx_left_end] = nnR;
		}
	}
	__syncthreads();
	int innActThreadCnt = (bundleWidth + 1023) >> 10;
	if (innActThreadCnt > 1) {
		int innAliThreadCnt = (innActThreadCnt + 31) & 0xFFFFFFE0;
		if (threadIdx.x < innAliThreadCnt) {
			bool active = threadIdx.x < innActThreadCnt;
			int nR = -1;
			int idx_left_end;
			if (active) {
				idx_left_end = threadIdx.x * 1024;
				nR = closestRight[idx_left_end];
			}
			int nnR = edt_findClosest_middle_prop_core32(nR);
			if (active) {
				if (nR == -1)
					closestRight[idx_left_end] = nnR;
			}
		}
		__syncthreads();
		if (threadIdx.x < actThreadCnt) {
			int idx_left_end = threadIdx.x * 32;
			int src_idx_right_side = (idx_left_end & 0xFFFFFC00) + 1024;
			if (src_idx_right_side < bundleWidth && closestRight[idx_left_end] == -1)
				closestRight[idx_left_end] = closestRight[src_idx_right_side];
		}
		__syncthreads();
	}
	for (int bias = 0; bias < bundleWidth; bias += blockDim.x) {
		int index = threadIdx.x + bias;
		if (index < bundleWidth) {
			int src_idx_right_side = (index & 0xFFFFFFE0) + 32;
			if (src_idx_right_side < bundleWidth && closestRight[index] == -1)
				closestRight[index] = closestRight[src_idx_right_side];
			bundle[index] = closestRight[index];
		}
	}
}

__global__ void edt_findClosest2D_high(idx2_t* in_nodes_out_closest, int nodeStride, const idx_t* sites, int siteStride, int srcImgWidth, int srcImgHeight) {
	const int node_row = blockIdx.y;
	const int site_col = node_row;
	__shared__ extern idx_t psClosestR[];
	idx2_t* pLeftNode = ptrAt(node_row, in_nodes_out_closest, nodeStride);
	for (int bias = 0; bias < srcImgHeight; bias += blockDim.x) {
		int index = bias + threadIdx.x;
		if (index < srcImgHeight) {
			psClosestR[index] = -1;
		}
	}
	__syncthreads();
	stk_node_t lstSite{ -1,-1 };
	for (int bias = 0; bias < srcImgHeight; bias += blockDim.x) {
		int node_col = threadIdx.x + bias;
		const int site_row = node_col;
		if (node_col < srcImgHeight) {
			stk_node_t node = pLeftNode[node_col];
			if (node.x != INVALID_NODE_PTR_NO_SITE && node.y != INVALID_NODE_PTR_NO_SITE)// this site valid
			{
				int domiIdx = INT_MAX;
				int myx = *ptrAt(site_row, site_col, sites, siteStride);
				int myy = node_col;
				if (node.x > 0) //have next site
				{
					int nxtx = *ptrAt(node.x, site_col, sites, siteStride);
					int nxty = node.x;
					domiIdx = ybscross(myx, myy, nxtx, nxty, site_col);
				}
				if (domiIdx >= srcImgHeight) {
					lstSite = make_short2(myx, myy);
					if (node.y > 0) // have previous site
					{
						int prevx = *ptrAt(node.y, site_col, sites, siteStride);
						int prevy = node.y;
						int prevDomiIdx = ybscross(myx, myy, prevx, prevy, site_col);
						if (prevDomiIdx >= srcImgHeight)
							lstSite = make_short2(-1, -1);
					}
				}
				else if (domiIdx >= 0)
				{
					psClosestR[domiIdx] = myy;
				}
			}
		}
	}
	__syncthreads();
	if (lstSite.y != -1 && psClosestR[srcImgHeight - 1] < 0)
		psClosestR[srcImgHeight - 1] = lstSite.y;
	__syncthreads();
	const int bundleWidth = srcImgHeight;
	for (int bias = 0; bias < bundleWidth; bias += blockDim.x) {
		int index = threadIdx.x + bias;
		int inRange = index < bundleWidth;
		int nR = inRange ? psClosestR[index] : -1;
		nR = edt_findClosest_middle_prop_core32(nR);
		if (inRange)
			psClosestR[index] = nR;
	}
	__syncthreads();
	int actThreadCnt = (bundleWidth + 31) >> 5;//=(srcWidth+31)/32
	int alvThreadCnt = (actThreadCnt + 31) & 0xFFFFFFE0;	//alive thread count
	if (threadIdx.x < alvThreadCnt) {
		int nR = -1;
		int idx_left_end;
		bool active = threadIdx.x < actThreadCnt;
		if (active) {
			idx_left_end = threadIdx.x * 32;
			nR = psClosestR[idx_left_end];
		}
		int nnR = edt_findClosest_middle_prop_core32(nR);
		if (active) {
			if (nR == -1)
				psClosestR[idx_left_end] = nnR;
		}
	}
	__syncthreads();
	int innActThreadCnt = (bundleWidth + 1023) >> 10;
	if (innActThreadCnt > 1) {
		int innAliThreadCnt = (innActThreadCnt + 31) & 0xFFFFFFE0;
		if (threadIdx.x < innAliThreadCnt) {
			bool active = threadIdx.x < innActThreadCnt;
			int nR = -1;
			int idx_left_end;
			if (active) {
				idx_left_end = threadIdx.x * 1024;
				nR = psClosestR[idx_left_end];
			}
			int nnR = edt_findClosest_middle_prop_core32(nR);
			if (active) {
				if (nR == -1)
					psClosestR[idx_left_end] = nnR;
			}
		}
		__syncthreads();
		if (threadIdx.x < actThreadCnt) {
			int idx_left_end = threadIdx.x * 32;
			int src_idx_right_side = (idx_left_end & 0xFFFFFC00) + 1024;
			if (src_idx_right_side < bundleWidth && psClosestR[idx_left_end] == -1)
				psClosestR[idx_left_end] = psClosestR[src_idx_right_side];
		}
		__syncthreads();
	}
	for (int bias = 0; bias < bundleWidth; bias += blockDim.x) {
		int index = threadIdx.x + bias;
		if (index < bundleWidth) {
			int src_idx_right_side = (index & 0xFFFFFFE0) + 32;
			if (src_idx_right_side < bundleWidth && psClosestR[index] == -1)
				psClosestR[index] = psClosestR[src_idx_right_side];
			int x = -1;
			int y = psClosestR[index];
			if (y >= 0)
				x = *ptrAt(y, site_col, sites, siteStride);
			pLeftNode[index].x = x;
			pLeftNode[index].y = y;
		}
	}
}
