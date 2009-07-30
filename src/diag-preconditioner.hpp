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



#ifndef AAFADFJ_ITERATIVE_CUDA_DIAG_PRECONDITIONER_HPP_SEEN
#define AAFADFJ_ITERATIVE_CUDA_DIAG_PRECONDITIONER_HPP_SEEN




#include <iterative-cuda.hpp>
#include "elementwise.hpp"




namespace iterative_cuda
{
  template <typename GpuVector>
  struct diagonal_preconditioner_pimpl
  {
    GpuVector const *vec;
  };





  template <class GpuVector>
  inline diagonal_preconditioner<GpuVector>::
  diagonal_preconditioner(gpu_vector_type const &vec)
    : pimpl(new diagonal_preconditioner_pimpl<GpuVector>)
  {
    pimpl->vec = &vec;
  }




  template <class GpuVector>
  inline void diagonal_preconditioner<GpuVector>::operator()(
      gpu_vector_type &result, gpu_vector_type const &op)
  {
    multiply(result, op, *pimpl->vec);
  }
}




#endif
