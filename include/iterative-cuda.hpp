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




#ifndef AAFADFJ_ITERATIVE_CUDA_HPP_SEEN
#define AAFADFJ_ITERATIVE_CUDA_HPP_SEEN




#include <utility>
#include <memory>
#include <stdint.h>




namespace iterative_cuda
{
  class noncopyable
  {
   protected:
      noncopyable() {}
      ~noncopyable() {}
   private:
      noncopyable( const noncopyable& );
      const noncopyable& operator=( const noncopyable& );
  };




  template <typename ValueType, typename IndexType>
  class gpu_vector_pimpl;

  template <typename ValueType, typename IndexType=int>
  class gpu_vector// : noncopyable
  {
    public:
      typedef IndexType index_type;
      typedef ValueType value_type;

    private:
      std::auto_ptr<gpu_vector_pimpl<value_type, index_type> > pimpl;

    public:
      explicit gpu_vector(index_type size);
      gpu_vector(value_type *cpu, index_type size);
      gpu_vector(gpu_vector const &src);
      ~gpu_vector();

      gpu_vector &operator=(gpu_vector const &src);

      index_type size() const;

      void from_cpu(value_type const *cpu);
      void to_cpu(value_type *cpu) const;

      value_type *ptr();
      const value_type *ptr() const;

      void fill(value_type x);
      void set_to_product(
          gpu_vector const &x,
          gpu_vector const &y);
      void set_to_quotient(
          gpu_vector const &x,
          gpu_vector const &y);
      void set_to_linear_combination(
          value_type a,
          gpu_vector const &x,
          value_type b,
          gpu_vector const &y);

      void set_to_linear_combination(
          value_type a,
          gpu_vector const &x,
          value_type b0,
          gpu_vector const &b1,
          gpu_vector const &y);

      gpu_vector *dot(gpu_vector const &b) const;
  };




  template <typename ValueType, typename IndexType=int>
  class gpu_sparse_pkt_matrix;

  template <typename ValueType, typename IndexType>
  class cpu_sparse_csr_matrix_pimpl;

  template <typename ValueType, typename IndexType=int>
  class cpu_sparse_csr_matrix
  {
    public:
      typedef IndexType index_type;
      typedef ValueType value_type;

    private:
      std::auto_ptr<
        cpu_sparse_csr_matrix_pimpl<value_type, index_type>
        > pimpl;

    public:
      cpu_sparse_csr_matrix(
          index_type row_count,
          index_type column_count,
          index_type nonzero_count,
          const index_type *csr_row_pointers,
          const index_type *csr_column_indices,
          const value_type *csr_nonzeros,
          bool take_ownership=true);
      ~cpu_sparse_csr_matrix();

      index_type row_count() const;
      index_type column_count() const;

      void operator()(value_type *y, value_type const *x) const;
      void extract_diagonal(value_type *d) const;

      static cpu_sparse_csr_matrix *read_matrix_market_file(const char *fn);

      friend class gpu_sparse_pkt_matrix<value_type, index_type>;
  };




  template <typename ValueType, typename IndexType>
  class gpu_sparse_pkt_matrix_pimpl;

  template <typename ValueType, typename IndexType>
  class gpu_sparse_pkt_matrix// : noncopyable
  {
    public:
      typedef IndexType index_type;
      typedef ValueType value_type;
      typedef gpu_vector<value_type, index_type> vector_type;

    private:
      std::auto_ptr<
        gpu_sparse_pkt_matrix_pimpl<value_type, index_type>
        > pimpl;

    public:
      gpu_sparse_pkt_matrix(
          cpu_sparse_csr_matrix<value_type, index_type> const &csr_mat);
      ~gpu_sparse_pkt_matrix();

      index_type row_count() const;
      index_type column_count() const;

      void permute(vector_type &dest, vector_type const &src) const;
      void unpermute(vector_type &dest, vector_type const &src) const;

      void operator()(vector_type &dest, vector_type const &src) const;
  };




  template <typename GpuVector>
  class diagonal_preconditioner_pimpl;




  template <class GpuVector>
  class identity_preconditioner : private noncopyable
  {
    public:
      typedef GpuVector gpu_vector_type;

      void operator()(gpu_vector_type &result, gpu_vector_type const &op) const
      {
        result = op;
      }
  };




  template <class GpuVector>
  class diagonal_preconditioner : private noncopyable
  {
    public:
      typedef GpuVector gpu_vector_type;
      typedef typename gpu_vector_type::index_type index_type;
      typedef typename gpu_vector_type::value_type value_type;

    private:
      std::auto_ptr<
        diagonal_preconditioner_pimpl<gpu_vector_type>
        > pimpl;

    public:
      // keeps a reference to vec
      diagonal_preconditioner(gpu_vector_type const &vec);
      ~diagonal_preconditioner();

      void operator()(gpu_vector_type &result, gpu_vector_type const &op) const;
  };





  void synchronize_gpu();

  template <typename ValueType, typename IndexType, typename Operator, typename Preconditioner>
  void gpu_cg(
      const Operator &a,
      const Preconditioner &m_inv,
      gpu_vector<ValueType, IndexType> &x,
      gpu_vector<ValueType, IndexType> const &b,
      ValueType tol=1e-8,
      unsigned max_iterations=0,
      unsigned *itcount=0);
}




#endif
