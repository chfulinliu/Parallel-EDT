/*
Author: Fulin Liu 

File Name: edt_makeStacks.cu

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


__device__ inline bool dominate(int ax, int ay, int bx, int by, int cx, int cy, int x) {
	int xpx = x + x;
	int x_bpa = bx + ax, y_bpa = by + ay;
	int x_bma = bx - ax, y_bma = by - ay;
	int den_ab = y_bma + y_bma;
	int num_ab = (x_bpa - xpx) * x_bma + y_bpa * y_bma - 1;
	int x_cpb = cx + bx, y_cpb = cy + by;
	int x_cmb = cx - bx, y_cmb = cy - by;
	int den_bc = y_cmb + y_cmb;
	int num_bc = (x_cpb - xpx) * x_cmb + y_cpb * y_cmb - 1;
	return num_ab / den_ab >= num_bc / den_bc;
}


//maxStkSz should ALWAYS equals to WARP_SIZE/blockDim.y
//every block processes a WARP_SIZE by WARP_SIZE patch
//stk and stkPtrs are TRANSPOSED w.r.t. origional image
//blockDim.x should ALWAYS equals to WARP_SIZE
__global__ void edt_creatColStacks(const idx_t* closestPerRow, int closestPerRowStride, stk_node_t* stk_nodes, int stk_node_stride, int maxStkSz, int srcImgWidth, int srcImgHeight) {
	constexpr int TS = WARP_SIZE;		//tile size(width equals height)
	constexpr int BH = TS;		//buf height
	constexpr int BW = TS + 1;	//buf width, BW is 1 more than BH to avoid bank confilct of shared memory
	__shared__ stk_node_t stk_node_buf[BH][BW];
	const int x = blockIdx.x * blockDim.x + threadIdx.x;	//transposed
	const int y_begin = (blockIdx.y * blockDim.y + threadIdx.y) * maxStkSz;			//transpose back
	const int rowIdxLimit = min(srcImgHeight - y_begin, maxStkSz);
	if (x < srcImgWidth && rowIdxLimit > 0) {
		const idx_t* prowSites = ptrAt(y_begin, x, closestPerRow, closestPerRowStride);
		stk_node_t(*nodes)[BW] = stk_node_buf + threadIdx.y * maxStkSz;
		int rb = 0;		//current row bias
		stk_node_t last_a, last_b;
		int curr_x, curr_y, last_y;
		last_y = INVALID_NODE_PTR_IS_SITE;
		last_a.y = INVALID_NODE_PTR_IS_SITE;
		last_b.y = INVALID_NODE_PTR_IS_SITE;
		constexpr stk_node_t invalid_node{ INVALID_NODE_PTR_NO_SITE, INVALID_NODE_PTR_NO_SITE };
		//create stack and backward pointers, the backword pointer is stored in node.y, node.x saves the correspnding row site of current node
		while (rb < rowIdxLimit) {
			nodes[rb][threadIdx.x] = invalid_node;
			curr_x = *prowSites;
			curr_y = rb;
			if (curr_x != INVALID_ROW_SITE) {
				while (IS_PTR_VALID(last_b.y)) {
					if (!dominate(last_a.x, last_b.y, last_b.x, last_y, curr_x, curr_y, x))
						break;
					nodes[last_y][threadIdx.x] = invalid_node;
					last_y = last_b.y;
					last_b = last_a;
					if (IS_PTR_VALID(last_a.y))
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
		if (last_y != lst_idx) // last node is not valid i.e. its row site is invalid
			nodes[lst_idx][threadIdx.x] = make_short2(INVALID_NODE_PTR_NO_SITE, last_y);

		//create forward pointers
		last_y = INVALID_NODE_PTR_IS_SITE;
		int next_y;
		if (nodes[lst_idx][threadIdx.x].x == INVALID_NODE_PTR_NO_SITE) // last node is not valid i.e. its row site is invalid
			next_y = nodes[lst_idx][threadIdx.x].y;
		else
			next_y = lst_idx;
		// here, next_y is the index of the last valid node
		while (IS_PTR_VALID(next_y)) {
			// when the following line is executed, last_y is the node just after next_y
			nodes[next_y][threadIdx.x].x = last_y; // when this line is executed the first time, next_y is the last valid node, it do not have next node, thus the index is INVALID_NODE_PTR_IS_SITE
			last_y = next_y;
			// move next_y to the head of the linked list
			next_y = nodes[next_y][threadIdx.x].y;
		}
		// when last code block finished execution, next_y is invalid, thus last_y should be the first valid node in the stack
		// the first valid node is not nodes[0][threadIdx.x], update the head of the stack
		if (last_y != 0)
			nodes[0][threadIdx.x] = make_short2(last_y, INVALID_NODE_PTR_NO_SITE);
		// The stack is a double linked list, values stored in each node serve as pointers to previous(node.y) or next(node.x) nodes, negative value is invalid pointer value
		// the pointers stored in the list also indicate the validity of corresponding node, if any of the pointer stored in a node is INVALID_NODE_PTR_NO_SITE, the node is invalid (dominated by other nodes)
		stk_node_t* pDst = ptrAt(y_begin, x, stk_nodes, stk_node_stride);
		for (int rb = 0; rb < rowIdxLimit; ++rb) {
			stk_node_t n = nodes[rb][threadIdx.x];
			if (n.x >= 0) n.x += y_begin;
			if (n.y >= 0) n.y += y_begin;
			*pDst = n;
			pDst = ptrNxtRow(pDst, stk_node_stride);
		}
	}
}

__device__ inline void edt_mergeColStack_core(stk_node_t* pStk, const int stk_1_begin, const int stk_1_end_2_begin, const int stk_2_end, const int stkstride, const idx_t* pxsites, const int xsitestride, const int x) {
	constexpr stk_node_t invalid_node{ INVALID_NODE_PTR_NO_SITE, INVALID_NODE_PTR_NO_SITE };
	int firsty = INVALID_NODE_PTR_IS_SITE, lasty = INVALID_NODE_PTR_IS_SITE;
	stk_node_t last_a = { INVALID_ROW_SITE, INVALID_NODE_PTR_NO_SITE };
	stk_node_t last_b = { INVALID_ROW_SITE, INVALID_NODE_PTR_NO_SITE };
	stk_node_t current = { INVALID_NODE_PTR_NO_SITE, INVALID_ROW_SITE };
	// last_a and last_b: x component store the x coordinate of the site, 
	// y component store the backward pointer
	// current: y component store the x coordinate of the site, 
	// x component store the forward pointer

	lasty = stk_1_end_2_begin - 1;
	last_b = make_short2(*ptrAt(lasty, pxsites, xsitestride), ptrAt(lasty, pStk, stkstride)->y);

	//last position of stack is invalid(only serves as a pointer)
	if (last_b.x == INVALID_NODE_PTR_NO_SITE) {
		lasty = last_b.y;
		if (IS_PTR_VALID(lasty))
			last_b = make_short2(*ptrAt(lasty, pxsites, xsitestride), ptrAt(lasty, pStk, stkstride)->y);
		else
			last_b = make_short2(INVALID_ROW_SITE, INVALID_NODE_PTR_NO_SITE);
	}

	if (IS_PTR_VALID(last_b.y)) {
		// Second item at the top of the stack
		last_a = make_short2(*ptrAt(last_b.y, pxsites, xsitestride), ptrAt(last_b.y, pStk, stkstride)->y);
	}

	// Get the first item of the second band
	firsty = stk_1_end_2_begin;
	current = make_short2(ptrAt(firsty, pStk, stkstride)->x, *ptrAt(firsty, pxsites, xsitestride));
	//the node occupying the first place in stack will not be dominated by others and will be kept in the list, unless it has no closest row site
	if (current.y == INVALID_ROW_SITE) {
		firsty = current.x;
		if (IS_PTR_VALID(firsty))
			current = make_short2(ptrAt(firsty, pStk, stkstride)->x, *ptrAt(firsty, pxsites, xsitestride));
		else
			current = make_short2(INVALID_NODE_PTR_NO_SITE, INVALID_ROW_SITE);
	}
	// Count the number of item in the second band that survive so far. 
	// Once it reaches 2, we can stop. 
	int top = 0;
	while (top < 2 && current.y != INVALID_ROW_SITE) {
		// While there's still something on the left
		while (IS_PTR_VALID(last_b.y)) {
			if (!dominate(last_a.x, last_b.y, last_b.x, lasty, current.y, firsty, x))
				break;
			*ptrAt(lasty, pStk, stkstride) = invalid_node;
			lasty = last_b.y;
			last_b = last_a;
			top--;
			if (IS_PTR_VALID(last_b.y))
				last_a = make_short2(*ptrAt(last_a.y, pxsites, xsitestride), ptrAt(last_a.y, pStk, stkstride)->y);
		}
		// Update the current pointer 
		*ptrAt(firsty, pStk, stkstride) = make_short2(current.x, lasty);
		if (IS_PTR_VALID(lasty))
			*ptrAt(lasty, pStk, stkstride) = make_short2(firsty, last_b.y);
		last_a = last_b;
		last_b = make_short2(current.y, lasty);
		lasty = firsty;
		firsty = current.x;
		top = max(1, top + 1);
		// Advance the current pointer to the next one
		if (IS_PTR_VALID(firsty))
			current = make_short2(ptrAt(firsty, pStk, stkstride)->x, *ptrAt(firsty, pxsites, xsitestride));
		else
			current = make_short2(INVALID_NODE_PTR_NO_SITE, INVALID_ROW_SITE);
	}

	// Update the head and tail pointer. 
	firsty = stk_1_begin;
	lasty = stk_1_end_2_begin;
	current = *ptrAt(firsty, pStk, stkstride);
	if (current.y == INVALID_NODE_PTR_NO_SITE && IS_PTR_INVALID(current.x)) {	// No head?
		last_a = *ptrAt(lasty, pStk, stkstride);
		if (last_a.y == INVALID_NODE_PTR_NO_SITE)
			current.x = last_a.x;
		else
			current.x = lasty;
		*ptrAt(firsty, pStk, stkstride) = current;
	}
	firsty = stk_1_end_2_begin - 1;
	lasty = stk_2_end - 1;
	current = *ptrAt(lasty, pStk, stkstride);
	if (current.x == INVALID_NODE_PTR_NO_SITE && IS_PTR_INVALID(current.y)) {	// No tail?
		last_a = *ptrAt(firsty, pStk, stkstride);
		if (last_a.x == INVALID_NODE_PTR_NO_SITE)
			current.y = last_a.y;
		else
			current.y = firsty;
		*ptrAt(lasty, pStk, stkstride) = current;
	}
}

__global__ void edt_mergeColStacks(stk_node_t* stkNodes, int stkNodeStride, int stkNodeWidth, int stkNodeHeight, const idx_t* sites, int siteStride, int single_stk_sz) {
	const int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (col < stkNodeWidth) {
		const int stk_1_begin = blockIdx.y * single_stk_sz * 2;
		const int stk_1_end_2_begin = stk_1_begin + single_stk_sz;
		const int stk_2_end = min(stk_1_end_2_begin + single_stk_sz, stkNodeHeight);
		edt_mergeColStack_core(stkNodes + col, stk_1_begin, stk_1_end_2_begin, stk_2_end, stkNodeStride, sites + col, siteStride, col);
	}
}

__global__ void edt_mergeColStacks_low(stk_node_t* stkNodes, int stkNodeStride, int stkNodeWidth, int stkNodeHeight, const idx_t* sites, int siteStride, int single_stk_sz) {
	const int stk_col = blockDim.x * blockIdx.x + threadIdx.x;
	const int site_col = stk_col;
	bool inRange = stk_col < stkNodeWidth;
	while (single_stk_sz < stkNodeHeight) {
		int workThreadCnt = (stkNodeHeight + single_stk_sz - 1) / single_stk_sz / 2;
		if (inRange && threadIdx.y < workThreadCnt) {
			int stk_1_begin = threadIdx.y * single_stk_sz * 2;
			int stk_1_end_2_begin = stk_1_begin + single_stk_sz;
			int stk_2_end = min(stk_1_end_2_begin + single_stk_sz, stkNodeHeight);
			edt_mergeColStack_core(stkNodes + stk_col, stk_1_begin, stk_1_end_2_begin, stk_2_end, stkNodeStride, sites + site_col, siteStride, site_col);
		}
		single_stk_sz = single_stk_sz + single_stk_sz;
		__syncthreads();
	}
}
