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
  value_type *x = new value_type[gpu_mat.column_count()];
  value_type *y1 = new value_type[gpu_mat.row_count()];
  value_type *y2 = new value_type[gpu_mat.row_count()];

  for (int i = 0; i < gpu_mat.column_count(); ++i)
    x[i] = drand48();
  for (int i = 0; i < gpu_mat.row_count(); ++i)
  {
    y1[i] = 0;
  }

  // do gpu matrix multiply
  gpu_vector<value_type> x_gpu(x, gpu_mat.column_count());
  gpu_vector<value_type> y_gpu(gpu_mat.row_count());

  gpu_vector<value_type> x_perm_gpu(gpu_mat.column_count());
  gpu_vector<value_type> y_perm_gpu(gpu_mat.row_count());
  gpu_mat.permute(x_perm_gpu, x_gpu);

  y_perm_gpu.fill(0);
  gpu_mat(y_perm_gpu, x_perm_gpu);

  gpu_mat.unpermute(y_gpu, y_perm_gpu);

  y_gpu.to_cpu(y2);
  synchronize_gpu();

  // compute error
  (*cpu_mat)(y1, x);

  value_type error = 0;
  value_type norm = 0;

  for (int i = 0; i < gpu_mat.row_count(); ++i)
  {
    error += (y1[i]-y2[i])*(y1[i]-y2[i]);
    norm += y1[i]*y1[i];
  }
  std::cerr << sqrt(error/norm) << std::endl;

  delete[] x;
  delete[] y1;
  delete[] y2;

  return 0;
}
