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




#include <iterative-cuda.hpp>
#include "gpu-vector.hpp"
#include "cpu-sparse-matrix.hpp"
#include "gpu-sparse-matrix.hpp"
#include "diag-preconditioner.hpp"
#include "cg.hpp"




using namespace iterative_cuda;




#define INSTANTIATE(TP) \
  template class gpu_vector<TP>; \
  template class cpu_sparse_csr_matrix<TP>; \
  template class gpu_sparse_pkt_matrix<TP>; \
  template class diagonal_preconditioner<gpu_vector<TP> >; \
  void gpu_cg( \
      const gpu_sparse_pkt_matrix<TP> &a, \
      const identity_preconditioner<gpu_vector<TP> > m_inv, \
      gpu_vector<TP> const &x, \
      gpu_vector<TP> const &b, \
      TP tol, \
      unsigned max_iterations); \
  void gpu_cg( \
      const gpu_sparse_pkt_matrix<TP> &a, \
      const diagonal_preconditioner<gpu_vector<TP> > m_inv, \
      gpu_vector<TP> const &x, \
      gpu_vector<TP> const &b, \
      TP tol, \
      unsigned max_iterations);
  


INSTANTIATE(float);
INSTANTIATE(double);
