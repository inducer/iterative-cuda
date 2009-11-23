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



#ifndef AAFADFJ_ITERATIVE_CUDA_CG_HPP_SEEN
#define AAFADFJ_ITERATIVE_CUDA_CG_HPP_SEEN




#include <iterative-cuda.hpp>
#include "reduction.hpp"




namespace iterative_cuda
{
  template <typename ValueType, typename IndexType, typename Operator, typename Preconditioner>
  inline void gpu_cg(
      const Operator &a,
      const Preconditioner &m_inv,
      gpu_vector<ValueType, IndexType> &x,
      gpu_vector<ValueType, IndexType> const &b,
      ValueType tol,
      unsigned max_iterations,
      unsigned *it_count)
  {
    typedef gpu_vector<ValueType, IndexType> vector_t;
    typedef vector_t gpu_scalar_t; // such is life. :/
    typedef typename vector_t::value_type scalar_t;

    if (a.row_count() != a.column_count())
      throw std::runtime_error("gpu_cg: A is not quadratic");

    unsigned n = a.row_count();

    std::auto_ptr<vector_t> norm_b_squared_gpu(b.dot(b));
    scalar_t norm_b_squared;
    norm_b_squared_gpu->to_cpu(&norm_b_squared);

    if (norm_b_squared == 0)
    {
      x = b;
      return;
    }

    if (max_iterations == 0)
      max_iterations = n;

    // typed up from J.R. Shewchuk, 
    // An Introduction to the Conjugate Gradient Method
    // Without the Agonizing Pain, Edition 1 1/4 [8/1994]
    // Appendix B3

    unsigned iterations = 0;
    vector_t ax(n);
    ax.fill(0);
    a(ax, x);

    vector_t residual(n);
    residual.set_to_linear_combination(1, b, -1, ax);

    vector_t d(n);
    m_inv(d, residual);

    std::auto_ptr<gpu_scalar_t> delta_new(residual.dot(d));

    scalar_t delta_0;
    delta_new->to_cpu(&delta_0);

    while (iterations < max_iterations)
    {
      vector_t q(n);
      q.fill(0);
      a(q, d);

      gpu_scalar_t alpha(1);
      std::auto_ptr<gpu_scalar_t> d_dot_q(d.dot(q));
      divide(alpha, *delta_new, *d_dot_q);

      x.set_to_linear_combination(1, x, 1, alpha, d);

      bool calculate_real_residual = iterations % 50 == 0;

      if (calculate_real_residual)
      {
        ax.fill(0);
        a(ax, x);
        residual.set_to_linear_combination(1, b, -1, ax);
      }
      else
        residual.set_to_linear_combination(1, residual, -1, alpha, q);

      vector_t s(n);
      m_inv(s, residual);

      std::auto_ptr<gpu_scalar_t> delta_old(delta_new);
      delta_new = std::auto_ptr<gpu_scalar_t>(residual.dot(s));

      if (calculate_real_residual)
      {
	// Only terminate the loop on the basis of a "real" residual.

        scalar_t delta_new_host;
        delta_new->to_cpu(&delta_new_host);
        if (std::abs(delta_new_host) < tol*tol * std::abs(delta_0))
          break;
      }

      gpu_scalar_t beta(1);
      divide(beta, *delta_new, *delta_old);

      d.set_to_linear_combination(1, s, 1, beta, d);

      iterations++;
    }

    if (iterations == max_iterations)
      throw std::runtime_error("cg failed to converge");

    if (it_count)
      *it_count = iterations;
  }
}




#endif
