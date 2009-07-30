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




#ifndef AAFADFJ_ITERATIVE_CUDA_GPU_SPARSE_MATRIX_HPP_SEEN
#define AAFADFJ_ITERATIVE_CUDA_GPU_SPARSE_MATRIX_HPP_SEEN



#include <iterative-cuda.hpp>
#include <stdint.h>
#include <iostream>
#include "helpers.hpp"
#include "partition.h"
#include "csr_to_pkt.h"
#include "utils.h"
#include "kernels/spmv_pkt_device.cu.h"
#include "cpu-sparse-matrix.hpp"




namespace iterative_cuda
{
  typedef uint32_t packed_index_type;




  template <typename ValueType, typename IndexType>
  struct gpu_sparse_pkt_matrix_pimpl
  {
    pkt_matrix<IndexType, ValueType> matrix;
  };




  template <typename VT, typename IT>
  inline gpu_sparse_pkt_matrix<VT, IT>::gpu_sparse_pkt_matrix(
      cpu_sparse_csr_matrix<VT, IT> const &csr_mat)
  : pimpl(new gpu_sparse_pkt_matrix_pimpl<VT, IT>)
  {
    index_type rows_per_packet =
      (SHARED_MEM_BYTES - 100)
      / (2*sizeof(value_type));

    index_type block_count = ICUDA_DIVIDE_INTO(
        csr_mat.row_count(), rows_per_packet);

    std::vector<index_type> partition;

    bool partition_ok;
    partition.resize(csr_mat.row_count());
    do
    {
      partition_csr(csr_mat.pimpl->matrix, block_count, partition, /*Kway*/ true);

      std::vector<index_type> block_occupancy(block_count, 0);
      for (index_type i = 0; i < csr_mat.row_count(); ++i)
        ++block_occupancy[partition[i]];

      partition_ok = true;
      for (index_type i = 0; i < block_count; ++i)
        if (block_occupancy[i] > rows_per_packet)
        {
          std::cerr << "Metis partition invalid, retrying..." << std::endl;
          partition_ok = false;
          block_count += 2 + int(1.02*block_count);
          break;
        }
    }
    while (!partition_ok);

    pkt_matrix<index_type, value_type> host_matrix =
      csr_to_pkt(csr_mat.pimpl->matrix, partition.data());

    pimpl->matrix = copy_matrix_to_device(host_matrix);
    delete_pkt_matrix(host_matrix, HOST_MEMORY);
  }




  template <typename VT, typename IT>
  inline gpu_sparse_pkt_matrix<VT, IT>::~gpu_sparse_pkt_matrix()
  {
    delete_pkt_matrix(pimpl->matrix, DEVICE_MEMORY);
  }




  template <typename VT, typename IT>
  inline IT gpu_sparse_pkt_matrix<VT, IT>::row_count() const
  {
    return pimpl->matrix.num_rows;
  }




  template <typename VT, typename IT>
  inline IT gpu_sparse_pkt_matrix<VT, IT>::column_count() const
  {
    return pimpl->matrix.num_cols;
  }




  template <typename VT, typename IT>
  inline void gpu_sparse_pkt_matrix<VT, IT>::permute(
      vector_type &dest,
      vector_type const &src) const
  {
    gather_device(dest.ptr(), src.ptr(), 
        pimpl->matrix.permute_old_to_new, row_count());
  }




  template <typename VT, typename IT>
  inline void gpu_sparse_pkt_matrix<VT, IT>::unpermute(
      vector_type &dest,
      vector_type const &src) const
  {
    gather_device(dest.ptr(), src.ptr(), 
        pimpl->matrix.permute_new_to_old, row_count());
  }





  template <typename VT, typename IT>
  inline void gpu_sparse_pkt_matrix<VT, IT>::operator()(
      vector_type &dest, vector_type const &src) const
  {
    spmv_pkt_device(pimpl->matrix, src.ptr(), dest.ptr());
  }
}




#endif
