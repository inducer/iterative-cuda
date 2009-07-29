/* Copyright 2008 NVIDIA Corporation.  All Rights Reserved */

#pragma once

#include "sparse_formats.h"

// COO format SpMV kernel that uses only one thread
// This is incredibly slow, so it is only useful for testing purposes,
// *extremely* small matrices, or a few elements at the end of a 
// larger matrix

template <typename IndexType, typename ValueType>
__global__ void
spmv_coo_serial_kernel(const IndexType num_nonzeros,
                       const IndexType * I, 
                       const IndexType * J, 
                       const ValueType * V, 
                       const ValueType * x, 
                             ValueType * y)
{
    for(IndexType n = 0; n < num_nonzeros; n++){
        y[I[n]] += V[n] * x[J[n]];
    }
}


template <typename IndexType, typename ValueType>
void spmv_coo_serial_device(const coo_matrix<IndexType,ValueType>& d_coo, 
                            const ValueType * d_x, 
                                  ValueType * d_y)
{
    spmv_coo_serial_kernel<IndexType,ValueType> <<<1,1>>>
        (d_coo.num_nonzeros, d_coo.I, d_coo.J, d_coo.V, d_x, d_y);
}
