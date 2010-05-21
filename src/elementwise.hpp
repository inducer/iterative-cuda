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




#ifndef AAFADFJ_ITERATIVE_CUDA_ELEMENTWISE_HPP_SEEN
#define AAFADFJ_ITERATIVE_CUDA_ELEMENTWISE_HPP_SEEN




#include "helpers.hpp"




namespace iterative_cuda
{
  static bool dev_props_loaded = false;
  static cudaDeviceProp dev_props;

  static inline void fill_dev_props()
  {
    if (dev_props_loaded)
      return;

    int my_dev;
    ICUDA_CHK(cudaGetDevice, (&my_dev));

    ICUDA_CHK(cudaGetDeviceProperties, (&dev_props, my_dev));
  }




  inline void splay(unsigned n, dim3 &grid, dim3 &block)
  {
    fill_dev_props();

    unsigned min_threads = dev_props.warpSize;
    unsigned max_threads = 128;
    unsigned max_blocks = 4 * 8 * dev_props.multiProcessorCount;

    unsigned block_count, threads_per_block;
    if (n < min_threads)
    {
      block_count = 1;
      threads_per_block = min_threads;
    }
    else if (n < (max_blocks * min_threads))
    {
      block_count = (n + min_threads - 1) / min_threads;
      threads_per_block = min_threads;
    }
    else if ( n < (max_blocks * max_threads))
    {
      block_count = max_blocks;
      unsigned grp = (n + min_threads - 1) / min_threads;
      threads_per_block = ((grp + max_blocks -1) / max_blocks) * min_threads;
    }
    else
    {
      block_count = max_blocks;
      threads_per_block = max_threads;
    }

    grid = block_count;
    block = threads_per_block;
  }




  template <class ValueType>
  __global__ void lc2_kernel(
      ValueType *z,
      ValueType a, ValueType const *x,
      ValueType b, ValueType const *y,
      unsigned n)
  {
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x*blockDim.x;
    unsigned cta_start = blockDim.x*blockIdx.x;
    unsigned i;

    for (i = cta_start + tid; i < n; i += total_threads) 
      z[i] = a*x[i] + b*y[i];
  }




  template <class ValueType>
  __global__ void lc2p_kernel(
      ValueType *z,
      ValueType a, ValueType const *x,
      ValueType b0, ValueType const *b1_ptr, ValueType const *y,
      unsigned n)
  {
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x*blockDim.x;
    unsigned cta_start = blockDim.x*blockIdx.x;
    unsigned i;

    ValueType b1 = *b1_ptr;

    for (i = cta_start + tid; i < n; i += total_threads) 
      z[i] = a*x[i] + b0*b1*y[i];
  }




  template <class VT, class IT>
  void lc2(
      gpu_vector<VT, IT> &z,
      VT a, gpu_vector<VT, IT> const &x,
      VT b, gpu_vector<VT, IT> const &y
      )
  {
    dim3 grid, block;
    splay(x.size(), grid, block);
    lc2_kernel<VT><<<grid, block>>>(
        z.ptr(), a, x.ptr(), b, y.ptr(), x.size());
  }




  template <class VT, class IT>
  void lc2(
      gpu_vector<VT, IT> &z,
      VT a, gpu_vector<VT, IT> const &x,
      VT b0, gpu_vector<VT, IT> const &b1, gpu_vector<VT, IT> const &y
      )
  {
    dim3 grid, block;
    splay(x.size(), grid, block);
    lc2p_kernel<VT><<<grid, block>>>(
        z.ptr(), a, x.ptr(), b0, b1.ptr(), y.ptr(), x.size());
  }




  template <class ValueType>
  __global__ void multiply_kernel(
      ValueType *z, ValueType const *x, ValueType const *y, unsigned n)
  {
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x*blockDim.x;
    unsigned cta_start = blockDim.x*blockIdx.x;
    unsigned i;

    for (i = cta_start + tid; i < n; i += total_threads) 
      z[i] = x[i]*y[i];
  }




  template <class VT, class IT>
  void multiply(
      gpu_vector<VT, IT> &z,
      gpu_vector<VT, IT> const &x,
      gpu_vector<VT, IT> const &y
      )
  {
    dim3 grid, block;
    splay(x.size(), grid, block);
    multiply_kernel<VT><<<grid, block>>>(
        z.ptr(), x.ptr(), y.ptr(), x.size());
  }




  template <class ValueType>
  __global__ void divide_kernel(
      ValueType *z, ValueType const *x, ValueType const *y, unsigned n)
  {
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x*blockDim.x;
    unsigned cta_start = blockDim.x*blockIdx.x;
    unsigned i;

    for (i = cta_start + tid; i < n; i += total_threads) 
      z[i] = x[i]/y[i];
  }




  template <class VT, class IT>
  void divide(
      gpu_vector<VT, IT> &z,
      gpu_vector<VT, IT> const &x,
      gpu_vector<VT, IT> const &y
      )
  {
    dim3 grid, block;
    splay(x.size(), grid, block);
    divide_kernel<VT><<<grid, block>>>(
        z.ptr(), x.ptr(), y.ptr(), x.size());
  }




  template <class ValueType>
  __global__ void guarded_divide_kernel(
      ValueType *z, ValueType const *x, ValueType const *y, unsigned n)
  {
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x*blockDim.x;
    unsigned cta_start = blockDim.x*blockIdx.x;
    unsigned i;

    for (i = cta_start + tid; i < n; i += total_threads) 
      z[i] = y[i] == 0 ? 0 : (x[i]/y[i]);
  }




  template <class VT, class IT>
  void guarded_divide(
      gpu_vector<VT, IT> &z,
      gpu_vector<VT, IT> const &x,
      gpu_vector<VT, IT> const &y
      )
  {
    dim3 grid, block;
    splay(x.size(), grid, block);
    guarded_divide_kernel<VT><<<grid, block>>>(
        z.ptr(), x.ptr(), y.ptr(), x.size());
  }




  template <class ValueType>
  __global__ void fill_kernel(
      ValueType *z, ValueType x, unsigned n)
  {
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x*blockDim.x;
    unsigned cta_start = blockDim.x*blockIdx.x;
    unsigned i;

    for (i = cta_start + tid; i < n; i += total_threads) 
      z[i] = x;
  }




  template <class VT, class IT>
  void fill(gpu_vector<VT, IT> &z, VT x)
  {
    dim3 grid, block;
    splay(z.size(), grid, block);
    fill_kernel<VT><<<grid, block>>>(z.ptr(), x, z.size());
  }
}




#endif
