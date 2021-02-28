/*
Author: Fulin Liu 

File Name: edt_utilities.cu

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

__global__ void edt_transpose32bit(const int32_t* in, int inStride, int32_t* out, int outStride, const int width_in, const int height_in) {
	constexpr int blkSz = 16;
	__shared__ int tile[blkSz][blkSz + 1];
	const int by = blockIdx.y * blockDim.y;
	const int bx = blockIdx.x * blockDim.x;
	const int inRow = by + threadIdx.y;
	const int inCol = bx + threadIdx.x;
	if (inRow < height_in && inCol < width_in)
		tile[threadIdx.y][threadIdx.x] = *ptrAt(inRow, inCol, in, inStride);
	__syncthreads();
	const int outRow = bx + threadIdx.y;
	const int outCol = by + threadIdx.x;
	if (outRow < width_in && outCol < height_in)
		*ptrAt(outRow, outCol, out, outStride) = tile[threadIdx.x][threadIdx.y];
}

// depreated functions
#if 0
#define EMPTY_MAK SHRT_MIN
#define MY_NEG_INT_INF SHRT_MIN
#define MY_POS_INT_INF SHRT_MAX


//maxStkSz should ALWAYS equals to WARP_SIZE/blockDim.y
//every block processes a WARP_SIZE by WARP_SIZE patch
//stk and stkPtrs are TRANSPOSED w.r.t. origional image
//blockDim.x should ALWAYS equals to WARP_SIZE
__global__ void edt_creatRowStacks(const idx_t* closestPerRow, int closestPerRowStride, stk_node_t* stk_nodes, int stk_node_stride, int maxStkSz, int srcImgWidth, int srcImgHeight) {
	constexpr int TS = WARP_SIZE;		//tile size(width equals height)
	constexpr int BH = TS;		//buf height
	constexpr int BW = TS + 1;	//buf width
	__shared__ stk_node_t stk_node_buf[BH][BW];
	const int stkptrs_row_idx = blockIdx.x * blockDim.x + threadIdx.x;	//transposed
	const int stkptrs_col_idx = blockIdx.y * blockDim.y + threadIdx.y;	//transposed
	const int closestPerRow_idx_col = stkptrs_row_idx;							//transpose back
	const int x = closestPerRow_idx_col;
	const int closest_per_row_idx_row_origin = stkptrs_col_idx * maxStkSz;			//transpose back
	const int rowIdxLimit = min(srcImgHeight - closest_per_row_idx_row_origin, maxStkSz);
	if (x < srcImgWidth && rowIdxLimit > 0) {
		const idx_t* prowSites = ptrAt(closest_per_row_idx_row_origin, closestPerRow_idx_col, closestPerRow, closestPerRowStride);
		stk_node_t(*nodes)[BW] = stk_node_buf + threadIdx.y * maxStkSz;
		int rb = 0;		//current row bias
		stk_node_t last_a, last_b;
		int curr_x, curr_y;
		int last_y = INVALID_NODE_PTR_IS_SITE;
		last_a.y = INVALID_NODE_PTR_IS_SITE; last_b.y = INVALID_NODE_PTR_IS_SITE;
		constexpr stk_node_t invalid_node{ INVALID_NODE_PTR_NO_SITE, INVALID_NODE_PTR_NO_SITE };
		//create stack and backward pointers
		while (rb < rowIdxLimit) {
			nodes[rb][threadIdx.x] = invalid_node;
			curr_x = *prowSites;
			curr_y = rb;
			if (curr_x != INVALID_ROW_SITE) {
				while (last_b.y >= 0) {
					if (!dominate(last_a.x, last_b.y, last_b.x, last_y, curr_x, curr_y, x))
						break;
					nodes[last_y][threadIdx.x] = invalid_node;
					last_y = last_b.y;
					last_b = last_a;
					if (last_a.y >= 0)
						last_a = nodes[last_a.y][threadIdx.x];
				}
				last_a = last_b;
				last_b = make_short2(curr_x, last_y);
				last_y = curr_y;
				nodes[rb][threadIdx.x] = last_b;
			}
			rb++;
			prowSites = ptrNxtRow(prowSites, closestPerRowStride);
		}
		const int lst_idx = rowIdxLimit - 1;
		if (last_y != lst_idx)
			nodes[lst_idx][threadIdx.x] = make_short2(INVALID_NODE_PTR_NO_SITE, last_y);

		//create forward pointers
		last_y = INVALID_NODE_PTR_IS_SITE;
		int next_y;
		if (nodes[lst_idx][threadIdx.x].x == INVALID_NODE_PTR_NO_SITE)
			next_y = nodes[lst_idx][threadIdx.x].y;
		else
			next_y = lst_idx;
		while (next_y >= 0) {
			nodes[next_y][threadIdx.x].x = last_y;
			last_y = next_y;
			next_y = nodes[next_y][threadIdx.x].y;
		}
		if (last_y != 0)
			nodes[0][threadIdx.x] = make_short2(last_y, INVALID_NODE_PTR_NO_SITE);
	}
	__syncthreads();
	int row_col_bias = maxStkSz * threadIdx.y;
	int dst_col = blockIdx.y * TS + threadIdx.x;
	if (dst_col < srcImgHeight) {
		const int dstTopRow = blockIdx.x * TS + row_col_bias;
		stk_node_t* pNodes = ptrAt(dstTopRow, blockIdx.y * TS + threadIdx.x, stk_nodes, stk_node_stride);
		int bias = blockIdx.y * TS + threadIdx.x / maxStkSz * maxStkSz;
		int dstRowIdxLimit = min(srcImgWidth - dstTopRow, maxStkSz);
		for (int bufCol_dstRow = 0; bufCol_dstRow < dstRowIdxLimit; ++bufCol_dstRow) {
			stk_node_t n = stk_node_buf[threadIdx.x][bufCol_dstRow + row_col_bias];
			if (n.x >= 0) n.x += bias;
			if (n.y >= 0) n.y += bias;
			*pNodes = n;
			pNodes = ptrNxtRow(pNodes, stk_node_stride);
		}
	}
}

__device__ inline void edt_mergeRowStack_core(stk_node_t* pStk, const int stk_1_begin, const int stk_1_end_2_begin, const int stk_2_end, const idx_t* pxsites, const int xsitestride, const int x) {
	constexpr stk_node_t invalid_node{ EMPTY_MAK, EMPTY_MAK };
	int firsty = -1, lasty = -1;
	stk_node_t last1 = invalid_node;
	stk_node_t last2 = invalid_node;
	stk_node_t current = invalid_node;
	// last_a and last_b: x component store the x coordinate of the site, 
	// y component store the backward pointer
	// current: y component store the x coordinate of the site, 
	// x component store the forward pointer

	lasty = stk_1_end_2_begin - 1;
	last2 = make_short2(*ptrAt(lasty, pxsites, xsitestride), pStk[lasty].y);

	//last position of stack is invalid(only serves as a pointer)
	if (last2.x == MY_NEG_INT_INF) {
		lasty = last2.y;
		if (lasty >= 0)
			last2 = make_short2(*ptrAt(lasty, pxsites, xsitestride), pStk[lasty].y);
		else
			last2 = make_short2(MY_NEG_INT_INF, EMPTY_MAK);
	}

	if (last2.y >= 0) {
		// Second item at the top of the stack
		last1 = make_short2(*ptrAt(last2.y, pxsites, xsitestride), pStk[last2.y].y);
	}

	// Get the first item of the second band
	firsty = stk_1_end_2_begin;
	current = make_short2(pStk[firsty].x, *ptrAt(firsty, pxsites, xsitestride));
	if (current.y == MY_NEG_INT_INF) {
		firsty = current.x;
		if (firsty >= 0)
			current = make_short2(pStk[firsty].x, *ptrAt(firsty, pxsites, xsitestride));
		else
			current = make_short2(EMPTY_MAK, MY_NEG_INT_INF);
	}
	// Count the number of item in the second band that survive so far. 
	// Once it reaches 2, we can stop. 
	int top = 0;
	while (top < 2 && current.y != MY_NEG_INT_INF) {
		// While there's still something on the left
		while (last2.y >= 0) {
			if (!dominate(last1.x, last2.y, last2.x, lasty, current.y, firsty, x))
				break;
			pStk[lasty] = invalid_node;
			lasty = last2.y;
			last2 = last1;
			top--;
			if (last2.y >= 0)
				last1 = make_short2(*ptrAt(last1.y, pxsites, xsitestride), pStk[last1.y].y);
		}
		// Update the current pointer 
		pStk[firsty] = make_short2(current.x, lasty);
		if (lasty >= 0)
			pStk[lasty] = make_short2(firsty, last2.y);
		last1 = last2;
		last2 = make_short2(current.y, lasty);
		lasty = firsty;
		firsty = current.x;
		top = max(1, top + 1);
		// Advance the current pointer to the next one
		if (firsty >= 0)
			current = make_short2(pStk[firsty].x, *ptrAt(firsty, pxsites, xsitestride));
		else
			current = make_short2(EMPTY_MAK, MY_NEG_INT_INF);
	}

	// Update the head and tail pointer. 
	firsty = stk_1_begin;
	lasty = stk_1_end_2_begin;
	current = pStk[firsty];
	if (current.y == EMPTY_MAK && current.x < 0) {	// No head?
		last1 = pStk[lasty];
		if (last1.y == EMPTY_MAK)
			current.x = last1.x;
		else
			current.x = lasty;
		pStk[firsty] = current;
	}
	firsty = stk_1_end_2_begin - 1;
	lasty = stk_2_end - 1;
	current = pStk[lasty];
	if (current.x == EMPTY_MAK && current.y < 0) {	// No tail?
		last1 = pStk[firsty];
		if (last1.x == EMPTY_MAK)
			current.y = last1.y;
		else
			current.y = firsty;
		pStk[lasty] = current;
	}
}

//blockDim.x should ALWAYS equals to (stkHeadersWidth+1)/2
__global__ void edt_mergeRowStacks(stk_node_t* stkNodes, int stkNodeStride, int stkNodeWidth, int stkNodeHeight, const idx_t* sites, int siteStride, int single_stk_sz) {
	const int stk_row = blockDim.y * blockIdx.y + threadIdx.y;
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int site_col = stk_row;
	int sw = single_stk_sz;
	bool inRange = stk_row < stkNodeHeight;
	while (sw < stkNodeWidth) {
		int workThreadCnt = (stkNodeWidth + sw - 1) / sw / 2;
		if (inRange && tidx < workThreadCnt) {
			int stk_1_begin = tidx * 2 * sw;
			int stk_1_end_2_begin = stk_1_begin + sw;
			int stk_2_end = stk_1_end_2_begin + sw;
			if (stk_2_end > stkNodeWidth)
				stk_2_end = stkNodeWidth;
			stk_node_t* pStk = ptrAt(stk_row, stkNodes, stkNodeStride);
			edt_mergeRowStack_core(pStk, stk_1_begin, stk_1_end_2_begin, stk_2_end, sites + site_col, siteStride, site_col);
		}
		sw = sw + sw;
		__syncthreads();
	}
}

#endif

