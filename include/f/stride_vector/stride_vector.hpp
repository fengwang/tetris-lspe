#ifndef YNRUQBUQAOTNDIIPOKUNOMTSTLXBOTMNSDPWULLJXNLLFCMXYLMOFVMLTNETGIEQPGCRCGCPK
#define YNRUQBUQAOTNDIIPOKUNOMTSTLXBOTMNSDPWULLJXNLLFCMXYLMOFVMLTNETGIEQPGCRCGCPK

//TODO:
//      arithmetic operators 

#include <f/stride_iterator/stride_iterator.hpp>
#include <f/algorithm/all_of.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <cassert>

namespace f
{

    template<typename T>
    struct stride_vector
    {
        typedef stride_vector                                    self_type;
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

        stride_vector() : p_( nullptr ), n_( 0 ), s_( 1 ) {}
        stride_vector( pointer p, size_type n, difference_type s = 1 ) : p_( p ), n_( n ), s_( s ) {}

        stride_vector( self_type const& )            = default;
        stride_vector( self_type&& )                 = default;
        self_type& operator = ( self_type const& )  = default;
        self_type& operator = ( self_type && )      = default;

        iterator begin()                            { return iterator( p_, s_ ); }
        iterator end()                              { return iterator( p_, s_ ) + n_; }

        const_iterator begin() const                { return const_iterator( p_, s_ ); }
        const_iterator end() const                  { return const_iterator( p_, s_ ) + n_; }

        reverse_iterator rbegin()                   { return reverse_iterator( end() ); }
        reverse_iterator rend()                     { return reverse_iterator( begin() ); }

        const_reverse_iterator rbegin() const       { return const_reverse_iterator( end() ); }
        const_reverse_iterator rend() const         { return const_reverse_iterator( begin() ); }

        const_iterator cbegin() const               { return const_iterator( p_, s_ ); }
        const_iterator cend() const                 { return const_iterator( p_, s_ ) + n_; }

        const_reverse_iterator crbegin() const      { return const_reverse_iterator( cend() ); }
        const_reverse_iterator crend() const        { return const_reverse_iterator( cbegin() ); }

        size_type size() const                      { return n_; }

        difference_type step() const                { return s_; }

        reference operator[] ( size_type n )        { return p_[n*s_]; }

        const_reference operator[] ( size_type n ) const { return p_[n*s_]; }

        reference front()                           { return p_[0]; }

        const_reference front() const               { return p_[0]; }

        reference back()                            { return p_[s_*n_-n_]; }

        const_reference back() const                { return p_[s_*n_-n_]; }

        pointer data()                              { return p_; }

        const_pointer data() const                  { return p_; }

        void assign( value_type const& v )          { fill( v ); }

        void fill( value_type const& v )            { std::fill( begin(), end(), v ); }

        reference at ( size_type n )
        {
            if ( n >= n_ ) assert( !"stride_vector: at out_of_range" );

            return p_[n*s_];
        }

        const_reference at ( size_type n ) const
        {
            if ( n >= n_ ) assert( !"stride_vector: at out_of_range" );

            return p_[n*s_];
        }

        void swap( self_type& other )
        {
            std::swap( p_, other.p_ );
            std::swap( n_, other.n_ );
            std::swap( s_, other.s_ );
        }

    };//struct stride_vector

    template< typename T >
    bool operator == ( stride_vector<T> const& lhs, stride_vector<T> const& rhs )
    {
        if ( lhs.size() == rhs.size() ) return all_of( lhs.begin(), lhs.end(), rhs.begin(), []( T const& x, T const& y ) {  return x == y; } );

        return false;
    }

    template< typename T >
    bool operator != ( stride_vector<T> const& lhs, stride_vector<T> const& rhs )
    {
        if ( lhs.size() == rhs.size() ) return !all_of( lhs.begin(), lhs.end(), rhs.begin(), []( T const& x, T const& y ) {  return x == y; } );
        
        return true;
    }

    template< typename T >
    bool operator < ( stride_vector<T> const& lhs, stride_vector<T> const& rhs )
    {
        if ( lhs.size() == rhs.size() ) return all_of( lhs.begin(), lhs.end(), rhs.begin(), []( T const& x, T const& y ) {  return x < y; } );

        return false;
    }

    template< typename T >
    bool operator <= ( stride_vector<T> const& lhs, stride_vector<T> const& rhs )
    {
        if ( lhs.size() == rhs.size() ) return all_of( lhs.begin(), lhs.end(), rhs.begin(), []( T const& x, T const& y ) {  return x <= y; } );

        return false;
    }

    template< typename T >
    bool operator > ( stride_vector<T> const& lhs, stride_vector<T> const& rhs )
    {
        if ( lhs.size() == rhs.size() ) return all_of( lhs.begin(), lhs.end(), rhs.begin(), []( T const& x, T const& y ) {  return x > y; } );

        return false;
    }

    template< typename T >
    bool operator >= ( stride_vector<T> const& lhs, stride_vector<T> const& rhs )
    {
        if ( lhs.size() == rhs.size() ) return all_of( lhs.begin(), lhs.end(), rhs.begin(), []( T const& x, T const& y ) {  return x >= y; } );

        return false;
    }

    template< typename T >
    void swap( stride_vector<T>& lhs, stride_vector<T>& rhs )
    {
        lhs.swap( rhs );
    }

    template< typename T >
    stride_vector<T> const make_stride_vector( T* p, std::size_t n, std::ptrdiff_t s = 1 )
    {
        return stride_vector<T>{ p, n, s };
    }

}//namespace f

#endif//YNRUQBUQAOTNDIIPOKUNOMTSTLXBOTMNSDPWULLJXNLLFCMXYLMOFVMLTNETGIEQPGCRCGCPK

