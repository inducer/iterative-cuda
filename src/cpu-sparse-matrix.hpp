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




#ifndef AAFADFJ_ITERATIVE_CUDA_CPU_SPARSE_MATRIX_HPP_SEEN
#define AAFADFJ_ITERATIVE_CUDA_CPU_SPARSE_MATRIX_HPP_SEEN




#include <iterative-cuda.hpp>
#include "sparse_io.h"




namespace iterative_cuda
{
  template <typename ValueType, typename IndexType>
  struct cpu_sparse_csr_matrix_pimpl
  {
    csr_matrix<IndexType, ValueType> matrix;
    bool own_matrix;
  };




  template <typename VT, typename IT>
  cpu_sparse_csr_matrix<VT, IT>::cpu_sparse_csr_matrix(
      index_type row_count,
      index_type column_count,
      index_type nonzero_count,
      const index_type *csr_row_pointers,
      const index_type *csr_column_indices,
      const value_type *csr_nonzeros,
      bool take_ownership)
  : pimpl(new cpu_sparse_csr_matrix_pimpl<VT, IT>)
  {
    pimpl->matrix.num_rows = row_count;
    pimpl->matrix.num_cols = column_count;
    pimpl->matrix.num_nonzeros = nonzero_count;
    pimpl->matrix.Ap = const_cast<index_type *>(csr_row_pointers);
    pimpl->matrix.Aj = const_cast<index_type *>(csr_column_indices);
    pimpl->matrix.Ax = const_cast<value_type *>(csr_nonzeros);
    pimpl->own_matrix = take_ownership;
  }




  template <typename VT, typename IT>
  cpu_sparse_csr_matrix<VT, IT>::~cpu_sparse_csr_matrix()
  {
    if (pimpl->own_matrix)
      delete_csr_matrix(pimpl->matrix, HOST_MEMORY);
  }




  template <typename VT, typename IT>
  IT cpu_sparse_csr_matrix<VT, IT>::row_count() const
  {
    return pimpl->matrix.num_rows;
  }




  template <typename VT, typename IT>
  IT cpu_sparse_csr_matrix<VT, IT>::column_count() const
  {
    return pimpl->matrix.num_cols;
  }




  template <typename VT, typename IT>
  void cpu_sparse_csr_matrix<VT, IT>::operator()(
      value_type *y, value_type const *x) const
  {
    csr_matrix<index_type, value_type> const &mat(pimpl->matrix);
    for (index_type i = 0; i < mat.num_rows; ++i)
    {
      const index_type row_start = mat.Ap[i];
      const index_type row_end = mat.Ap[i+1];

      value_type sum = y[i];
      for (index_type jj = row_start; jj < row_end; jj++) 
      {
        const index_type j = mat.Aj[jj];
        sum += x[j] * mat.Ax[jj];
      }
      y[i] = sum;
    }
  }




  template <typename VT, typename IT>
  void cpu_sparse_csr_matrix<VT, IT>::extract_diagonal(value_type *d) const
  {
    csr_matrix<index_type, value_type> const &mat(pimpl->matrix);
    for (index_type i = 0; i < mat.num_rows; ++i)
    {
      d[i] = 0;
      const index_type row_start = mat.Ap[i];
      const index_type row_end = mat.Ap[i+1];

      for (index_type jj = row_start; jj < row_end; jj++) 
      {
        const index_type j = mat.Aj[jj];
        if (i == j)
        {
          d[i] += mat.Ax[jj];
        }
      }
    }
  }




  template <class ValueType, class IndexType>
  cpu_sparse_csr_matrix<ValueType, IndexType> *
  cpu_sparse_csr_matrix<ValueType, IndexType>::read_matrix_market_file(
      const char *fn)
  {
    csr_matrix<IndexType, ValueType> csr_mat =
      read_csr_matrix<IndexType, ValueType>(fn);

    typedef cpu_sparse_csr_matrix<ValueType, IndexType> mat_tp;
    std::auto_ptr<mat_tp> result(new mat_tp(
          csr_mat.num_rows, csr_mat.num_cols, csr_mat.num_nonzeros,
          csr_mat.Ap, csr_mat.Aj, csr_mat.Ax));

    return result.release();
  }
}




#endif
