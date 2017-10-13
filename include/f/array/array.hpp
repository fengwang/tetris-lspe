#ifndef MARRAY_HPP_INCLUDED_SDOPINASLFDKJH43O9HU8AFSDLKJH4O8USAFDKJABSDFKAMBDSFO
#define MARRAY_HPP_INCLUDED_SDOPINASLFDKJH43O9HU8AFSDLKJH4O8USAFDKJABSDFKAMBDSFO

#include <f/algorithm/for_each.hpp>

#include <type_traits>
#include <utility>
#include <iterator>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <cstddef>

namespace f
{

    template <class Type, std::size_t Size>
    struct  array
    {
        typedef array                                 self_type;
        typedef Type                                  value_type;
        typedef value_type&                           reference;
        typedef const value_type&                     const_reference;
        typedef value_type*                           iterator;
        typedef const value_type*                     const_iterator;
        typedef value_type*                           pointer;
        typedef const value_type*                     const_pointer;
        typedef std::size_t                           size_type;
        typedef ptrdiff_t                             difference_type;
        typedef std::reverse_iterator<iterator>       reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

        value_type element_data_[Size > 0 ? Size : 1];

        void fill( value_type const& value )
        {
            std::fill_n( element_data_, Size, value );
        }

        void assign( value_type const& value )
        {
            fill( value );
        }

        void swap( self_type& other )
        {
            std::swap_ranges( begin(), end(), other.begin() );
        }

        // iterators:
        iterator begin()
        {
            return iterator( element_data_ );
        }

        const_iterator begin() const
        {
            return const_iterator( element_data_ );
        }

        iterator end()
        {
            return iterator( element_data_ + Size );
        }

        const_iterator end() const
        {
            return const_iterator( element_data_ + Size );
        }

        reverse_iterator rbegin()
        {
            return reverse_iterator( end() );
        }

        const_reverse_iterator rbegin() const
        {
            return const_reverse_iterator( end() );
        }

        reverse_iterator rend()
        {
            return reverse_iterator( begin() );
        }

        const_reverse_iterator rend() const
        {
            return const_reverse_iterator( begin() );
        }

        const_iterator cbegin() const
        {
            return begin();
        }

        const_iterator cend() const
        {
            return end();
        }

        const_reverse_iterator crbegin() const
        {
            return rbegin();
        }

        const_reverse_iterator crend() const
        {
            return rend();
        }

        // capacity:
        constexpr size_type size() const
        {
            return Size;
        }

        constexpr size_type max_size() const
        {
            return Size;
        }

        constexpr bool empty() const
        {
            return Size == 0;
        }

        // element access:
        reference operator[]( size_type position )
        {
            return element_data_[position];
        }
        const_reference operator[]( size_type position ) const
        {
            return element_data_[position];
        }

        reference at( size_type position )
        {
            if ( position > Size ) assert( !"array::at -- out of range!" );
            return element_data_[position];
        }

        const_reference at( size_type position ) const
        {
            if ( position > Size ) assert( !"array::at -- out of range!" );
            return element_data_[position];
        }

        reference front()
        {
            return element_data_[0];
        }

        const_reference front() const
        {
            return element_data_[0];
        }

        reference back()
        {
            return element_data_[Size > 0 ? Size - 1 : 0];
        }

        const_reference back() const
        {
            return element_data_[Size > 0 ? Size - 1 : 0];
        }

        pointer data()
        {
            return element_data_;
        }

        const_pointer data() const
        {
            return element_data_;
        }

        //alrithmetic operators
        self_type& operator += ( value_type const& value )
        {
            std::for_each( begin(), end(), [&value]( value_type & x ) { x += value; } );
            return *this;
        }

        self_type& operator += ( self_type const& other )
        {
            for_each( begin(), end(), other.cbegin(), []( value_type & x, value_type const & y ) { x += y; } );
            return *this;
        }

        self_type& operator -= ( value_type const& value )
        {
            std::for_each( begin(), end(), [&value]( value_type & x ) { x -= value; } );
            return *this;
        }

        self_type& operator -= ( self_type const& other )
        {
            for_each( begin(), end(), other.cbegin(), []( value_type & x, value_type const & y ) { x -= y; } );
            return *this;
        }

        self_type& operator *= ( value_type const& value )
        {
            std::for_each( begin(), end(), [&value]( value_type & x ) { x *= value; } );
            return *this;
        }

        self_type& operator /= ( value_type const& value )
        {
            std::for_each( begin(), end(), [&value]( value_type & x ) { x /= value; } );
            return *this;
        }

        self_type const operator +() const
        {
            return *this;
        }

        self_type const operator -() const
        {
            self_type ans( *this );
            std::for_each( ans.begin(), ans.end(), []( value_type & x ) { x = -x; } );
            return ans;
        }

        //friend alrithmetic operators
        friend self_type const operator + ( self_type const& lhs, self_type const& rhs )
        {
            self_type ans( lhs );
            ans += rhs;
            return ans;
        }

        friend self_type const operator + ( value_type const& lhs, self_type const& rhs )
        {
            self_type ans( rhs );
            ans += lhs;
            return ans;
        }

        friend self_type const operator + ( self_type const& lhs, value_type const& rhs )
        {
            return rhs + lhs;
        }

        friend self_type const operator - ( value_type const& lhs, self_type const& rhs )
        {
            return lhs + ( -rhs );
        }

        friend self_type const operator - ( self_type const& lhs, value_type const& rhs )
        {
            self_type ans( lhs );
            ans -= rhs;
            return ans;
        }

        friend self_type const operator - ( self_type const& lhs, self_type const& rhs )
        {
            self_type ans( lhs );
            ans -= rhs;
            return ans;
        }

        friend self_type const operator * ( self_type const& lhs, value_type const& rhs )
        {
            self_type ans( lhs );
            ans *= rhs;
            return ans;
        }

        friend self_type const operator * ( value_type const& lhs, self_type const& rhs )
        {
            return rhs * lhs;
        }

        friend self_type const operator / ( self_type const& lhs, value_type const& rhs )
        {
            self_type ans( lhs );
            ans /= rhs;
            return ans;
        }

        //friend relation operators
        friend bool operator == ( self_type const& lhs, self_type const& rhs )
        {
            return std::equal( lhs.cbegin(), lhs.cend(), rhs.cbegin() );
        }

        friend bool operator != ( self_type const& lhs, self_type const& rhs )
        {
            return !( lhs == rhs );
        }

        friend bool operator < ( self_type const& lhs, self_type const& rhs )
        {
            return std::lexicographical_compare( lhs.cbegin(), lhs.cend(), rhs.cbegin() );
        }

        friend bool operator > ( self_type const& lhs, self_type const& rhs )
        {
            return rhs < lhs;
        }

        friend bool operator >= ( self_type const& lhs, self_type const& rhs )
        {
            return !( lhs < rhs );
        }

        friend bool operator <= ( self_type const& lhs, self_type const& rhs )
        {
            return !( rhs < lhs );
        }

    }; //struct array


    template <class Type, std::size_t Size>
    void swap( const array<Type, Size>& lhs, const array<Type, Size>& rhs )
    {
        lhs.swap( rhs );
    }

    template <std::size_t Position, class Type, std::size_t Size>
    Type& get( array<Type, Size>& the_one )
    {
        return the_one[Position];
    }

    template <std::size_t Position, class Type, std::size_t Size>
    const Type& get( const array<Type, Size>& the_one )
    {
        return the_one[Position];
    }

    template <std::size_t Position, class Type, std::size_t Size>
    Type&& get( array<Type, Size>&& the_one )
    {
        return std::move( the_one[Position] );
    }

    template <typename T, typename... Types>
    array<T, sizeof...(Types)+1> const make_array( T const& arg1, Types const& ... argn )
    {
        return array<T, sizeof...(Types)+1>{ arg1, argn... };
    }

}//namespace f

#endif//_ARRAY_HPP_INCLUDED_SDOPINASLFDKJH43O9HU8AFSDLKJH4O8USAFDKJABSDFKAMBDSFO

