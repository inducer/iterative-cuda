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




namespace iterative_cuda
{
  template <typename ValueType, typename IndexType>
  struct gpu_vector_pimpl
  {
    ValueType *gpu_data;
    IndexType size;
  };




  template <typename VT, typename IT>
  gpu_vector<VT, IT>::gpu_vector(index_type size)
  : pimpl(new gpu_vector_pimpl<VT, IT>)
  {
    ICUDA_CHK(cudaMalloc, ((void **) &pimpl->gpu_data, size*sizeof(value_type)));
    pimpl->size = size;
  }




  template <typename VT, typename IT>
  gpu_vector<VT, IT>::gpu_vector(gpu_vector const &src)
  : pimpl(new gpu_vector_pimpl<VT, IT>)
  {
    ICUDA_CHK(cudaMalloc, ((void **) &pimpl->gpu_data, src.size()*sizeof(value_type)));
    pimpl->size = src.size();
    ICUDA_CHK(cudaMemcpy, (pimpl->gpu_data, src.pimpl->gpu_data, 
          src.size()*sizeof(value_type),
          cudaMemcpyDeviceToDevice));
  }




  template <typename VT, typename IT>
  gpu_vector<VT, IT>::~gpu_vector()
  {
    ICUDA_CHK(cudaFree, (pimpl->gpu_data));
  }




  template <typename VT, typename IT>
  IT gpu_vector<VT, IT>::size() const
  { return pimpl->size; }




  template <typename VT, typename IT>
  void gpu_vector<VT, IT>::from_cpu(value_type *cpu)
  {
    ICUDA_CHK(cudaMemcpy, (pimpl->gpu_data, cpu, 
          size()*sizeof(value_type),
          cudaMemcpyHostToDevice));
  }




  template <typename VT, typename IT>
  void gpu_vector<VT, IT>::to_cpu(value_type *cpu)
  {
    ICUDA_CHK(cudaMemcpy, (cpu, pimpl->gpu_data,
          size()*sizeof(value_type),
          cudaMemcpyDeviceToHost));
  }




  template <typename VT, typename IT>
  gpu_vector<VT, IT>::value_type *gpu_vector<VT, IT>::ptr()
  { return pimpl->gpu_data; }




  template <typename VT, typename IT>
  const gpu_vector<VT, IT>::value_type *gpu_vector<VT, IT>::ptr() const
  { return pimpl->gpu_data; }
}




#endif
