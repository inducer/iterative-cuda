/*
Iterative CUDA is licensed to you under the MIT/X Consortium license:

Copyright (c) 2009 Andreas Kloeckner.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the Software), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/




#ifndef _AAFADFJ_ITERATIVE_CUDA_GPU_VECTOR_HPP_SEEN
#define _AAFADFJ_ITERATIVE_CUDA_GPU_VECTOR_HPP_SEEN



#include <iterative-cuda.hpp>
#include "helpers.hpp"
#include "partition.hpp"
#include "csr_to_pk.hpp"




namespace iterative_cuda
{
  typedef std::uint32_t packed_index_type;




  template <typename ValueType, typename IndexType>
  struct gpu_sparse_pkt_matrix_pimpl
  {
    // these are GPU pointers
    IndexType *permute_old_to_new;  
    IndexType *permute_new_to_old;

    packed_index_type *index_array;
    ValueType *data_array;

    IndexType *pos_start;
    IndexType *pos_end;

    IndexType *coo_i;
    IndexType *coo_j;
    ValueType *coo_v;
  }




  template <typename VT, typename IT>
  gpu_sparse_pkt_matrix<VT, IT>::gpu_sparse_pkt_matrix(
      index_type row_count,
      index_type column_count,
      const index_type *csr_row_pointers,
      const index_type *csr_column_indices,
      const value_type *csr_nonzeros)
  : pimpl(new gpu_sparse_pkt_matrix_pimpl)
  {
    csr_matrix<index_type, value_type> csr_mat;
    csr_mat.Ap = csr_row_pointers;
    csr_mat.Aj = csr_column_indices;
    csr_mat.Ax = csr_nonzeros;

    index_type rows_per_packet = 
      (SHARED_MEM_AMOUNT - 100)
      / (2*sizeof(value_type));

    index_type block_count = divide_into(h, rows_per_packet);

    std::vector<IndexType> partition;
    partition_csr(csr_mat, block_count, partition, /*Kway*/ true);

    pkt_matrix<index_type, value_type> pkt =
      csr_to_pkt(csr_mat, partition.data());


  }
}




#endif
