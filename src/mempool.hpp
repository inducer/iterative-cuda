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




// Abstract memory pool implementation




#ifndef AFJDFJSDFSD_ITERATIVE_CUDA_HEADER_SEEN_MEMPOOL_HPP
#define AFJDFJSDFSD_ITERATIVE_PYCUDA_HEADER_SEEN_MEMPOOL_HPP




#include <iterative-cuda.hpp>
#include <map>
#include <map>
#include <cassert>
#include <vector>
#include <stdexcept>
#include "bitlog.hpp"




namespace iterative_cuda
{
  template <class T>
  inline T signed_left_shift(T x, signed shift_amount)
  {
    if (shift_amount < 0)
      return x >> -shift_amount;
    else
      return x << shift_amount;
  }




  template <class T>
  inline T signed_right_shift(T x, signed shift_amount)
  {
    if (shift_amount < 0)
      return x << -shift_amount;
    else
      return x >> shift_amount;
  }




  template<class Allocator>
  class memory_pool
  {
    public:
      typedef typename Allocator::pointer_type pointer_type;
      typedef typename Allocator::size_type size_type;

    private:
      typedef uint32_t bin_nr_t;
      typedef std::vector<pointer_type> bin_t;

      typedef std::map<bin_nr_t, bin_t *> container_t;
      container_t m_container;
      typedef typename container_t::value_type bin_pair_t;

      Allocator m_allocator;

      // A held block is one that's been released by the application, but that
      // we are keeping around to dish out again.
      unsigned m_held_blocks;

      // An active block is one that is in use by the application.
      unsigned m_active_blocks;

      bool m_stop_holding;

    public:
      memory_pool(Allocator const &alloc=Allocator())
        : m_allocator(alloc), m_held_blocks(0), m_active_blocks(0), m_stop_holding(false)
      {
      }
      
      ~memory_pool()
      { 
        free_held(); 

        typename container_t::iterator 
          first = m_container.begin(),
          last = m_container.end();

        while (first != last)
          delete (first++)->second;
        m_container.clear();
      }

      static const unsigned mantissa_bits = 2;
      static const unsigned mantissa_mask = (1 << mantissa_bits) - 1;

      static bin_nr_t bin_number(size_type size)
      {
        signed l = bitlog2(size);
        size_type shifted = signed_right_shift(size, l-signed(mantissa_bits));
        if (size && (shifted & (1 << mantissa_bits)) == 0)
          throw std::runtime_error("memory_pool::bin_number: bitlog2 fault");
        size_type chopped = shifted & mantissa_mask;
        return l << mantissa_bits | chopped;
      }

      static size_type alloc_size(bin_nr_t bin)
      {
        bin_nr_t exponent = bin >> mantissa_bits;
        bin_nr_t mantissa = bin & mantissa_mask;

        size_type ones = signed_left_shift(1, 
            signed(exponent)-signed(mantissa_bits)
            );
        if (ones) ones -= 1;

        size_type head = signed_left_shift(
           (1<<mantissa_bits) | mantissa, 
            signed(exponent)-signed(mantissa_bits));
        if (ones & head)
          throw std::runtime_error("memory_pool::alloc_size: bit-counting fault");
        return head | ones;
      }

    protected:
      bin_t &get_bin(bin_nr_t bin_nr)
      {
        typename container_t::iterator it = m_container.find(bin_nr);
        if (it == m_container.end())
        {
          bin_t *new_bin = new bin_t;
          m_container.insert(std::make_pair(bin_nr, new_bin));
          return *new_bin;
        }
        else
          return *it->second;
      }

      void inc_held_blocks()
      {
        if (m_held_blocks == 0)
          start_holding_blocks();
        ++m_held_blocks;
      }

      void dec_held_blocks()
      {
        --m_held_blocks;
        if (m_held_blocks == 0)
          stop_holding_blocks();
      }

      virtual void start_holding_blocks()
      { }

      virtual void stop_holding_blocks()
      { }

    public:
      pointer_type allocate(size_type size)
      {
        bin_nr_t bin_nr = bin_number(size);
        bin_t &bin = get_bin(bin_nr);
        
        if (bin.size())
          return pop_block_from_bin(bin, size);

        size_type alloc_sz = alloc_size(bin_nr);

        assert(bin_number(alloc_sz) == bin_nr);

        try { return get_from_allocator(alloc_sz); }
        catch (std::bad_alloc) { } // OOM? ok, free memory

        m_allocator.try_release_blocks();
        if (bin.size())
          return pop_block_from_bin(bin, size);

        while (try_to_free_memory())
        {
          try { return get_from_allocator(alloc_sz); }
          catch (std::bad_alloc) { } // still OOM? free more
        }

        throw std::bad_alloc();
      }

      void free(pointer_type p, size_type size)
      {
        --m_active_blocks;

        if (!m_stop_holding)
        {
          inc_held_blocks();
          get_bin(bin_number(size)).push_back(p);
        }
        else
          m_allocator.free(p);
      }

      void free_held()
      {
        typename container_t::iterator 
          first = m_container.begin(),
          last = m_container.end();

        while (first != last)
        {
          bin_pair_t bin_pair = *first;
          bin_t &bin = *bin_pair.second;

          while (bin.size())
          {
            m_allocator.free(bin.back());
            bin.pop_back();
            
            dec_held_blocks();
          }

          ++first;
        }

        assert(m_held_blocks == 0);
      }

      void stop_holding()
      {
        m_stop_holding = true;
        free_held();
      }

      unsigned active_blocks()
      { return m_active_blocks; }

      unsigned held_blocks()
      { return m_held_blocks; }

      bool try_to_free_memory()
      {
        typename container_t::reverse_iterator 
          first = m_container.rbegin(),
          last = m_container.rend();
        while (first != last)
        {
          bin_pair_t bin_pair = *first;
          bin_t &bin = *bin_pair.second;

          if (bin.size())
          {
            m_allocator.free(bin.back());
            bin.pop_back();

            dec_held_blocks();

            return true;
          }

          ++first;
        }

        return false;
      }

    private:
      pointer_type get_from_allocator(size_type alloc_sz)
      {
        pointer_type result = m_allocator.allocate(alloc_sz);
        ++m_active_blocks;

        return result;
      }

      pointer_type pop_block_from_bin(bin_t &bin, size_type size)
      {
        pointer_type result = bin.back();
        bin.pop_back();

        dec_held_blocks();
        ++m_active_blocks;

        return result;
      }
  };
}




#endif
