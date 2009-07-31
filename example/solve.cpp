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




using namespace iterative_cuda;

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr << "usage: " << argv[0] << " matrix.mtx" << std::endl;
    return 1;
  }
  typedef float value_type;
  typedef cpu_sparse_csr_matrix<value_type> cpu_mat_type;
  typedef gpu_sparse_pkt_matrix<value_type> gpu_mat_type;
  std::auto_ptr<cpu_mat_type> cpu_mat(
      cpu_mat_type::read_matrix_market_file(argv[1]));

  gpu_mat_type gpu_mat(*cpu_mat);

  // build host vectors
  unsigned n = gpu_mat.row_count();
  value_type *b = new value_type[n];
  value_type *x = new value_type[n];
  value_type *b2 = new value_type[n];

  for (int i = 0; i < gpu_mat.column_count(); ++i)
  {
    b[i] = drand48();
  }

  // transfer vectors to gpu
  typedef gpu_vector<value_type> vec_t;
  vec_t b_gpu(b, n);
  vec_t x_gpu(n);
  vec_t b2_gpu(n);

  x_gpu.fill(0);
  b2_gpu.fill(0);

  std::cout << "begin solve" << std::endl;

  value_type tol;
  if (sizeof(value_type) < 8) // ick
    tol = 1e-4;
  else
    tol = 1e-8;

  unsigned it_count;
  gpu_cg(gpu_mat, 
      identity_preconditioner<vec_t>(),
      x_gpu, b_gpu, tol, /*max_iterations*/ 0, &it_count);

  gpu_mat(b2_gpu, x_gpu);

  vec_t residual_gpu(n);
  residual_gpu.set_to_linear_combination(1, b2_gpu, -1, b_gpu);

  std::auto_ptr<vec_t> res_norm_gpu(residual_gpu.dot(residual_gpu));
  value_type res_norm;
  res_norm_gpu->to_cpu(&res_norm);

  std::cout << "residual norm " << res_norm << std::endl;
  std::cout << "iterations " << it_count << std::endl;

  return 0;
}


