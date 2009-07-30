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
  typedef gpu_sparse_pkt_matrix<entry_type> mat_type;
  std::auto_ptr<mat_type> mat(
      mat_type::read_matrix_market_file(argv[1]));

  // build host vectors
  entry_type *x = new entry_type[mat->column_count()];
  entry_type *y = new entry_type[mat->row_count()];

  for (int i = 0; i < mat->column_count(); ++i)
    x[i] = drand48();
  for (int i = 0; i < mat->row_count(); ++i)
    y[i] = 0;

  gpu_vector<entry_type> x_gpu(mat->column_count());
  gpu_vector<entry_type> y_gpu(mat->row_count());

  x_gpu.from_cpu(x);
  y_gpu.from_cpu(y);

  (*mat)(y_gpu, x_gpu);

  y_gpu.to_cpu(y);
  synchronize_gpu();

  delete[] x;
  delete[] y;

  return 0;
}
