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

Based on code by Mark Harris at Nvidia.
*/




#ifndef AAFADFJ_ITERATIVE_CUDA_REDUCTION_HPP_SEEN
#define AAFADFJ_ITERATIVE_CUDA_REDUCTION_HPP_SEEN




#include <memory>




namespace iterative_cuda
{
#define MAKE_REDUCTION_KERNEL(KERNEL_NAME, ARG_DECL, READ_AND_MAP, REDUCE) \
  template <class ValueType, unsigned BlockSize> \
  __global__ void KERNEL_NAME##_kernel(ValueType *out, \
      ARG_DECL, unsigned int seq_count, unsigned int n) \
  { \
    __shared__ ValueType sdata[BlockSize]; \
    \
    unsigned int tid = threadIdx.x; \
    unsigned int i = blockIdx.x*BlockSize*seq_count + tid; \
    \
    ValueType acc = 0; \
    for (unsigned s = 0; s < seq_count; ++s) \
    {  \
      if (i >= n) \
        break; \
      acc = REDUCE(acc, READ_AND_MAP(i));  \
      \
      i += BlockSize;  \
    } \
    \
    sdata[tid] = acc; \
    \
    __syncthreads(); \
    \
    if (BlockSize >= 512) \
    { \
      if (tid < 256) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 256]); } \
      __syncthreads(); \
    } \
    \
    if (BlockSize >= 256)  \
    { \
      if (tid < 128) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 128]); }  \
      __syncthreads();  \
    } \
    \
    if (BlockSize >= 128)  \
    { \
      if (tid < 64) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 64]); }  \
      __syncthreads();  \
    } \
    \
    if (tid < 32)  \
    { \
      if (BlockSize >= 64) sdata[tid] = REDUCE(sdata[tid], sdata[tid + 32]); \
      if (BlockSize >= 32) sdata[tid] = REDUCE(sdata[tid], sdata[tid + 16]); \
      if (BlockSize >= 16) sdata[tid] = REDUCE(sdata[tid], sdata[tid + 8]); \
      if (BlockSize >= 8)  sdata[tid] = REDUCE(sdata[tid], sdata[tid + 4]); \
      if (BlockSize >= 4)  sdata[tid] = REDUCE(sdata[tid], sdata[tid + 2]); \
      if (BlockSize >= 2)  sdata[tid] = REDUCE(sdata[tid], sdata[tid + 1]); \
    } \
    \
    if (tid == 0) out[blockIdx.x] = sdata[0]; \
  } \




  #define INNER_PRODUCT_STAGE1_ARGS ValueType const *a, ValueType const *b
  #define INNER_PRODUCT_READ_AND_MAP(i) a[i]*b[i]
  #define INNER_PRODUCT_STAGE2_ARGS ValueType const *a
  #define STRAIGHT_READ(i) a[i]
  #define SUM_REDUCE(a, b) a+b
  
  MAKE_REDUCTION_KERNEL(inner_product_stage1, 
      INNER_PRODUCT_STAGE1_ARGS, 
      INNER_PRODUCT_READ_AND_MAP, SUM_REDUCE);
  MAKE_REDUCTION_KERNEL(inner_product_stage2, 
      INNER_PRODUCT_STAGE2_ARGS, 
      STRAIGHT_READ, SUM_REDUCE);




  template <class VT, class IT>
  gpu_vector<VT, IT> *inner_product(
      gpu_vector<VT, IT> const &a,
      gpu_vector<VT, IT> const &b)
  {
    const unsigned BLOCK_SIZE = 512;
    const unsigned MAX_BLOCK_COUNT = 1024;
    const unsigned SMALL_SEQ_COUNT = 4;

    typedef gpu_vector<VT, IT> vec_t;

    std::auto_ptr<vec_t> result;

    while (true)
    {
      unsigned sz;
      if (!result.get())
        // stage 1
        sz = a.size();
      else
        // stage 2..n
        sz = result->size();

      unsigned block_count, seq_count;
      if (sz <= BLOCK_SIZE*SMALL_SEQ_COUNT*MAX_BLOCK_COUNT)
      {
        unsigned total_block_size = SMALL_SEQ_COUNT*BLOCK_SIZE;
        block_count = (sz + total_block_size - 1) / total_block_size;
        seq_count = SMALL_SEQ_COUNT;
      }
      else
      {
        block_count = MAX_BLOCK_COUNT;
        unsigned macroblock_size = block_count*BLOCK_SIZE;
        seq_count = (sz + macroblock_size - 1) / macroblock_size;
      }

      if (!result.get())
      {
        // stage 1
        result = std::auto_ptr<vec_t>(new vec_t(block_count));

        inner_product_stage1_kernel<VT, BLOCK_SIZE>
          <<<block_count, BLOCK_SIZE>>>(result->ptr(), a.ptr(), b.ptr(), 
              seq_count, sz);
      }
      else
      {
        // stage 2..n
        std::auto_ptr<vec_t> new_result(new vec_t(block_count));

        inner_product_stage2_kernel<VT, BLOCK_SIZE>
          <<<block_count, BLOCK_SIZE>>>(new_result->ptr(), result->ptr(),
              seq_count, sz);

        result = new_result;
      }
      if (block_count == 1)
        return result.release();
    }
  }
}




#endif
