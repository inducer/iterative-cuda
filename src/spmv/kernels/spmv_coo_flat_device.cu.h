/* Copyright 2008 NVIDIA Corporation.  All Rights Reserved */

#pragma once

#include "sparse_formats.h"
#include "utils.h"
#include "texture.h"
#include "kernels/spmv_coo_serial_device.cu.h"
#include "kernels/spmv_common_device.cu.h"

//////////////////////////////////////////////////////////////////////////////
// COO SpMV kernel which flattens data irregularity (segmented reduction)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_coo_flat_device
//   The input coo_matrix must be sorted by row.  Columns within each row
//   may appear in any order and duplicate entries are also acceptable.
//   This sorted COO format is easily obtained by expanding the row pointer
//   of a CSR matrix (csr.Ap) into proper row indices and then copying 
//   the arrays containing the CSR column indices (csr.Aj) and nonzero values
//   (csr.Ax) verbatim.
//
// spmv_coo_flat_tex_device
//   Same as spmv_coo_flat_device, except that the texture cache is 
//   used for accessing the x vector.

template <typename IndexType, typename ValueType, unsigned int BLOCK_SIZE, bool UseCache>
__global__ void
spmv_coo_flat_kernel(const IndexType num_nonzeros,
                     const IndexType interval_size,
                     const IndexType * I, 
                     const IndexType * J, 
                     const ValueType * V, 
                     const ValueType * x, 
                           ValueType * y)
{
    __shared__ IndexType idx[BLOCK_SIZE];
    __shared__ ValueType val[BLOCK_SIZE];
    __shared__ IndexType carry_idx[BLOCK_SIZE / 32];
    __shared__ ValueType carry_val[BLOCK_SIZE / 32];

    const IndexType thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;     // global thread index
    const IndexType thread_lane = threadIdx.x & (WARP_SIZE-1);               // thread index within the warp
    const IndexType warp_id     = thread_id   / WARP_SIZE;                   // global warp index
    const IndexType warp_lane   = threadIdx.x / WARP_SIZE;                   // warp index within the CTA

    const IndexType begin = warp_id * interval_size + thread_lane;           // thread's offset into I,J,V
    const IndexType end   = min(begin + interval_size, num_nonzeros);        // end of thread's work

    if(begin >= end) return;                                                 // warp has no work to do

    const IndexType first_idx = I[warp_id * interval_size];                  // first row of this warp's interval

    if (thread_lane == 0){
        carry_idx[warp_lane] = first_idx; 
        carry_val[warp_lane] = 0;
    }
    
    for(IndexType n = begin; n < end; n += WARP_SIZE){
        idx[threadIdx.x] = I[n];                                             // row index
        val[threadIdx.x] = V[n] * fetch_x<UseCache>(J[n], x);                // val = A[row,col] * x[col] 

        if (thread_lane == 0){
            if(idx[threadIdx.x] == carry_idx[warp_lane])
                val[threadIdx.x] += carry_val[warp_lane];                    // row continues into this warp's span
            else if(carry_idx[warp_lane] != first_idx)
                y[carry_idx[warp_lane]] += carry_val[warp_lane];             // row terminated, does not span boundary
            else
                myAtomicAdd(y + carry_idx[warp_lane], carry_val[warp_lane]);   // row terminated, but spans iter-warp boundary
        }

        // segmented reduction in shared memory
        if( thread_lane >=  1 && idx[threadIdx.x] == idx[threadIdx.x - 1] ) { val[threadIdx.x] += val[threadIdx.x -  1]; EMUSYNC; } 
        if( thread_lane >=  2 && idx[threadIdx.x] == idx[threadIdx.x - 2] ) { val[threadIdx.x] += val[threadIdx.x -  2]; EMUSYNC; }
        if( thread_lane >=  4 && idx[threadIdx.x] == idx[threadIdx.x - 4] ) { val[threadIdx.x] += val[threadIdx.x -  4]; EMUSYNC; }
        if( thread_lane >=  8 && idx[threadIdx.x] == idx[threadIdx.x - 8] ) { val[threadIdx.x] += val[threadIdx.x -  8]; EMUSYNC; }
        if( thread_lane >= 16 && idx[threadIdx.x] == idx[threadIdx.x -16] ) { val[threadIdx.x] += val[threadIdx.x - 16]; EMUSYNC; }

        if( thread_lane == 31 ) {
            carry_idx[warp_lane] = idx[threadIdx.x];                         // last thread in warp saves its results
            carry_val[warp_lane] = val[threadIdx.x];
        }
        else if ( idx[threadIdx.x] != idx[threadIdx.x+1] ) {                 // row terminates here
            if(idx[threadIdx.x] != first_idx)
                y[idx[threadIdx.x]] += val[threadIdx.x];                     // row terminated, does not span inter-warp boundary
            else
                myAtomicAdd(y + idx[threadIdx.x], val[threadIdx.x]);           // row terminated, but spans iter-warp boundary
        }
        
    }

    // final carry
    if(thread_lane == 31){
        myAtomicAdd(y + carry_idx[warp_lane], carry_val[warp_lane]); 
    }
}


template <typename IndexType, typename ValueType, bool UseCache>
void __spmv_coo_flat_device(const coo_matrix<IndexType,ValueType>& d_coo, 
                            const ValueType * d_x, 
                                  ValueType * d_y)
{
    if(d_coo.num_nonzeros == 0)
        return; //empty matrix

    const unsigned int BLOCK_SIZE      = 128;
    const unsigned int MAX_BLOCKS      = 4*MAX_THREADS / BLOCK_SIZE; // empirically  better on test cases
    const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

    const unsigned int num_units  = d_coo.num_nonzeros / WARP_SIZE; 
    const unsigned int num_warps  = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
    const unsigned int num_blocks = DIVIDE_INTO(num_warps, WARPS_PER_BLOCK);
    const unsigned int num_iters  = DIVIDE_INTO(num_units, num_warps);
    
    const unsigned int interval_size = WARP_SIZE * num_iters;

    const IndexType tail = num_units * WARP_SIZE; // do the last few nonzeros separately

    if (UseCache)
        bind_x(d_x);

    spmv_coo_flat_kernel<IndexType, ValueType, BLOCK_SIZE, UseCache> <<<num_blocks, BLOCK_SIZE>>>
        (tail, interval_size, d_coo.I, d_coo.J, d_coo.V, d_x, d_y);

    spmv_coo_serial_kernel<IndexType,ValueType> <<<1,1>>>
        (d_coo.num_nonzeros - tail, d_coo.I + tail, d_coo.J + tail, d_coo.V + tail, d_x, d_y);

    if (UseCache)
        unbind_x(d_x);
}

template <typename IndexType, typename ValueType>
void spmv_coo_flat_device(const coo_matrix<IndexType,ValueType>& d_coo, 
                          const ValueType * d_x, 
                                ValueType * d_y)
{ 
    __spmv_coo_flat_device<IndexType, ValueType, false>(d_coo, d_x, d_y);
}


template <typename IndexType, typename ValueType>
void spmv_coo_flat_tex_device(const coo_matrix<IndexType,ValueType>& d_coo, 
                              const ValueType * d_x, 
                                    ValueType * d_y)
{ 
    __spmv_coo_flat_device<IndexType, ValueType, true>(d_coo, d_x, d_y);
}
