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
  typedef float entry_type;
  typedef cpu_sparse_csr_matrix<entry_type> cpu_mat_type;
  typedef gpu_sparse_pkt_matrix<entry_type> gpu_mat_type;
  std::auto_ptr<cpu_mat_type> cpu_mat(
      cpu_mat_type::read_matrix_market_file(argv[1]));

  gpu_mat_type gpu_mat(*cpu_mat);

  // build host vectors
  entry_type *x = new entry_type[gpu_mat.column_count()];
  entry_type *y1 = new entry_type[gpu_mat.row_count()];
  entry_type *y2 = new entry_type[gpu_mat.row_count()];

  for (int i = 0; i < gpu_mat.column_count(); ++i)
    x[i] = drand48();
  for (int i = 0; i < gpu_mat.row_count(); ++i)
  {
    y1[i] = 0;
    y2[i] = 0;
  }

  // do gpu matrix multiply
  gpu_vector<entry_type> x_gpu(gpu_mat.column_count());
  gpu_vector<entry_type> y_gpu(gpu_mat.row_count());

  x_gpu.from_cpu(x);
  y_gpu.from_cpu(y2);

  gpu_mat(y_gpu, x_gpu);

  y_gpu.to_cpu(y2);
  synchronize_gpu();

  // compute error
  (*cpu_mat)(y1, x);

  entry_type error = 0;
  entry_type norm = 0;

  for (int i = 0; i < gpu_mat.row_count(); ++i)
  {
    error += (y1[i]-y2[i])*(y1[i]-y2[i]);
    norm += x[i]*x[i];
  }
  std::cerr << error/norm << std::endl;

  delete[] x;
  delete[] y1;
  delete[] y2;

  return 0;
}
