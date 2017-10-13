#ifndef YNRUQBUQAOTNDIIPOKUNOMTSTLXBOTMNSDPWULLJXNLLFCMXYLMOFVMLTNETGIEQPGCRCGCPK
#define YNRUQBUQAOTNDIIPOKUNOMTSTLXBOTMNSDPWULLJXNLLFCMXYLMOFVMLTNETGIEQPGCRCGCPK

#include <f/stride_iterator/stride_iterator.hpp>

#include <cstddef>
#include <iterator>

namespace f
{

    template< typename T >
    struct stride_array
    {
        typedef stride_array                                    self_type;
        typedef T                                               value_type;
        typedef value_type&                                     reference;
        typedef const value_type&                               const_reference;
        typedef value_type*                                     pointer;
        typedef const value_type*                               const_pointer;
        typedef stride_iterator<pointer>                        iterator;
        typedef stride_iterator<const_pointer>                  const_iterator;
        typedef std::reverse_iterator<iterator>                 reverse_iterator;
        typedef std::reverse_iterator<const_iterator>           const_reverse_iterator;
        typedef std::size_t                                     size_type;
        typedef std::ptrdiff_t                                  difference_type;

        pointer         p_; //
        size_type       n_; //number of elements
        difference_type s_; //stride step

        stride_array() : p_( nullptr ), n_( 0 ), s_( 1 ) {}

        stride_array( pointer p, size_type n, difference_type s = 1 ) : p_( p ), n_( n ), s_( s ) {}

        stride_array( self_type const& ) = default;
        stride_array( self_type&& ) = default;
        self_type& operator = ( self_type const& ) = default;
        self_type& operator = ( self_type&& ) = default;

        iterator begin() { return iterator( p_, s_ ); }
        iterator end() { return iterator( p_, s_ ) + n; }

        const_iterator begin() const { return const_iterator( p_, s_ ); }
        const_iterator end() const { return const_iterator( p_, s_ ) + n; }

        const_iterator cbegin() const { return const_iterator( p_, s_ ); }
        const_iterator cend() const { return const_iterator( p_, s_ ) + n; }


        reverse_iterator rbegin() { return reverse_iterator(end()); }
        reverse_iterator rend() { return reverse_iterator(begin()); }

        const_reverse_iterator rbegin() const { return const_reverse_iterator( end() ); }
        const_reverse_iterator rend() const { return const_reverse_iterator( begin() ); }

        const_reverse_iterator crbegin() const { return const_reverse_iterator( end() ); }
        const_reverse_iterator crend() const { return const_reverse_iterator( begin() ); }

    };//struct stride_array

    //make_stride_array
    //swap
    //operators ...

}//namespace f

#endif//YNRUQBUQAOTNDIIPOKUNOMTSTLXBOTMNSDPWULLJXNLLFCMXYLMOFVMLTNETGIEQPGCRCGCPK

