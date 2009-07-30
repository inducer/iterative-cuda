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
  typedef float value_type;

  const unsigned n = 20000000;
  value_type *x = new value_type[n];
  value_type *y = new value_type[n];

  for (int i = 0; i < n; ++i)
  {
    x[i] = drand48();
    y[i] = drand48();
  }

  double ref_dot = 0;

  for (int i = 0; i < n; ++i)
    ref_dot += double(x[i])*double(y[i]);

  typedef gpu_vector<value_type> vec_t;
  vec_t x_gpu(x, n);
  vec_t y_gpu(y, n);

  std::auto_ptr<vec_t> result(x_gpu.dot(y_gpu));

  value_type gpu_dot;
  result->to_cpu(&gpu_dot);

  std::cerr << (ref_dot-gpu_dot)/ref_dot << std::endl;

  return 0;
}

