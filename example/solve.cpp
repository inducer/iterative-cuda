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
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>




using namespace iterative_cuda;




double get_time()
{
  const double TICKS_PER_SECOND = 1000*1000;
  struct timeval tv;
  struct timezone tz;
  ::gettimeofday(&tv, &tz);
  return double(tv.tv_sec) + double(tv.tv_usec)/TICKS_PER_SECOND;
}




int main(int argc, char **argv)
{
  srand48(49583344);

  if (argc != 2)
  {
    std::cerr << "usage: " << argv[0] << " matrix.mtx" << std::endl;
    return 1;
  }
#if 0
  typedef float value_type;
#else
  typedef double value_type;
#endif
  typedef cpu_sparse_csr_matrix<value_type> cpu_mat_type;
  typedef gpu_sparse_pkt_matrix<value_type> gpu_mat_type;
  typedef gpu_vector<value_type> gvec_t;

  std::auto_ptr<cpu_mat_type> cpu_mat(
      cpu_mat_type::read_matrix_market_file(argv[1]));

  unsigned n = cpu_mat->row_count();

  value_type *diag = new value_type[n];
  value_type *inv_diag = new value_type[n];
  cpu_mat->extract_diagonal(diag);
  for (int i = 0; i < n; ++i)
  {
    inv_diag[i] = 1./diag[i];
  }
  gvec_t inv_diag_gpu(inv_diag, n);
  delete[] inv_diag;
  delete[] diag;

  gpu_mat_type gpu_mat(*cpu_mat);

  gvec_t perm_inv_diag_gpu(n);
  gpu_mat.permute(perm_inv_diag_gpu, inv_diag_gpu);
  diagonal_preconditioner<gvec_t> diag_pre(perm_inv_diag_gpu);

  // build host vectors
  value_type *b = new value_type[n];

  for (int i = 0; i < n; ++i)
    b[i] = drand48();

  // transfer vectors to gpu
  std::cout << "begin solve" << std::endl;

  double start = get_time();
  gvec_t b_gpu(b, n);
  gvec_t b_perm_gpu(n);
  gpu_mat.permute(b_perm_gpu, b_gpu);

  gvec_t x_perm_gpu(n);
  x_perm_gpu.fill(0);

  // perform solve
  value_type tol;
  if (sizeof(value_type) < sizeof(double))
    tol = 1e-4;
  else
    tol = 1e-12;

  unsigned it_count;
  gpu_cg(gpu_mat, 
      // identity_preconditioner<gvec_t>(),
      diag_pre,
      x_perm_gpu, b_perm_gpu, tol, /*max_iterations*/ 0, &it_count);

  gvec_t x_gpu(n);
  gpu_mat.unpermute(x_gpu, x_perm_gpu);

  // compute residual on cpu
  value_type *x = new value_type[n];
  x_gpu.to_cpu(x);
  double stop = get_time();

  value_type *b2 = new value_type[n];
  for (int i = 0; i < n; ++i)
    b2[i] = 0;
  (*cpu_mat)(b2, x);

  value_type error = 0;
  value_type norm = 0;

  for (int i = 0; i < gpu_mat.row_count(); ++i)
  {
    error += (b[i]-b2[i])*(b[i]-b2[i]);
    norm += b[i]*b[i];
  }

  std::cerr << "residual norm " << sqrt(error/norm) << std::endl;
  std::cout << "iterations " << it_count << std::endl;
  std::cout << "elapsed seconds " << stop-start << std::endl;
  std::cout << "elapsed seconds per iteration " << (stop-start)/it_count << std::endl;

  delete[] b;
  delete[] x;
  delete[] b2;

  return 0;
}
