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
#include "reduction.hpp"
#include "elementwise.hpp"
#include "mempool.hpp"




namespace iterative_cuda
{
  class cuda_allocator
  {
    public:
      typedef void *pointer_type;
      typedef unsigned long size_type;

      pointer_type allocate(size_type s)
      {
        void *result;
        cudaError_t cuda_err_code = cudaMalloc(&result, s);
        if (cuda_err_code == cudaErrorMemoryAllocation)
          throw std::bad_alloc();
        else if (cuda_err_code != cudaSuccess)
        {
          printf("cudaMalloc failed with code %d\n", cuda_err_code);
          abort();
        }
        return result;
      }

      void free(pointer_type p)
      {
        ICUDA_CHK(cudaFree, (p));
      }

      void try_release_blocks()
      { }
  };



  typedef memory_pool<cuda_allocator> cuda_mempool;

  cuda_mempool &get_cuda_mempool();





  template <typename ValueType, typename IndexType>
  struct gpu_vector_pimpl
  {
    ValueType *gpu_data;
    IndexType size;
  };




  template <typename VT, typename IT>
  inline gpu_vector<VT, IT>::gpu_vector(index_type size)
  : pimpl(new gpu_vector_pimpl<VT, IT>)
  {
    pimpl->gpu_data = (value_type *) get_cuda_mempool().allocate(
        size*sizeof(value_type));
    pimpl->size = size;
  }




  template <typename VT, typename IT>
  inline gpu_vector<VT, IT>::gpu_vector(value_type *cpu, index_type size)
  : pimpl(new gpu_vector_pimpl<VT, IT>)
  {
    pimpl->gpu_data = (value_type *) get_cuda_mempool().allocate(
        size*sizeof(value_type));
    pimpl->size = size;
    from_cpu(cpu);
  }




  template <typename VT, typename IT>
  inline gpu_vector<VT, IT>::gpu_vector(gpu_vector const &src)
  : pimpl(new gpu_vector_pimpl<VT, IT>)
  {
    pimpl->gpu_data = (value_type *) get_cuda_mempool().allocate(
        src.size()*sizeof(value_type));
    pimpl->size = src.size();
    ICUDA_CHK(cudaMemcpy, (ptr(), src.ptr(), 
          src.size()*sizeof(value_type),
          cudaMemcpyDeviceToDevice));
  }




  template <typename VT, typename IT>
  inline gpu_vector<VT, IT>::~gpu_vector()
  {
    get_cuda_mempool().free(pimpl->gpu_data, pimpl->size*sizeof(value_type));
  }




  template <typename VT, typename IT>
  inline IT gpu_vector<VT, IT>::size() const
  { return pimpl->size; }




  template <typename VT, typename IT>
  inline void gpu_vector<VT, IT>::from_cpu(value_type *cpu)
  {
    ICUDA_CHK(cudaMemcpy, (ptr(), cpu, 
          size()*sizeof(value_type),
          cudaMemcpyHostToDevice));
  }




  template <typename VT, typename IT>
  inline void gpu_vector<VT, IT>::to_cpu(value_type *cpu)
  {
    ICUDA_CHK(cudaMemcpy, (cpu, ptr(),
          size()*sizeof(value_type),
          cudaMemcpyDeviceToHost));
  }




  template <typename VT, typename IT>
  inline gpu_vector<VT, IT>::value_type *gpu_vector<VT, IT>::ptr()
  { return (value_type *) pimpl->gpu_data; }




  template <typename VT, typename IT>
  inline const gpu_vector<VT, IT>::value_type *gpu_vector<VT, IT>::ptr() const
  { return (const value_type *) pimpl->gpu_data; }




  template <typename VT, typename IT>
  inline void gpu_vector<VT, IT>::set_to_linear_combination(
      value_type a,
      gpu_vector const &x,
      value_type b,
      gpu_vector const &y)
  {
    lc2(a, x, b, y, *this);
  }




  template <typename VT, typename IT>
  inline gpu_vector<VT, IT> *gpu_vector<VT, IT>::dot(gpu_vector const &b) const
  {
    return inner_product(*this, b);
  }
}




#endif
