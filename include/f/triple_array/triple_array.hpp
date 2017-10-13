#ifndef MARRAY_HPP_INCLUDED_SDFIOJ9438USLKFDJDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDSFD8E3I
#define MARRAY_HPP_INCLUDED_SDFIOJ9438USLKFDJDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDSFD8E3I

#include <f/matrix/matrix.hpp>

#include <array>
#include <algorithm>
#include <iosfwd>
#include <cassert>
#include <iostream>

namespace f
{

    template<typename T>
    struct triple_array// : std::array<T,3>
    {
        typedef T                   value_type;
        typedef triple_array        self_type;
        typedef value_type*         pointer;
        typedef const value_type*   const_pointer;

        value_type data_[3];

        pointer operator[]( const unsigned long ) 
        {
            return &(data_[0]);
        }

        const_pointer operator[]( const unsigned ) const
        {
            return &(data_[0]);
        }

        pointer begin()
        {
            return &(data_[0]);
        }

        const_pointer begin() const
        {
            return &(data_[0]);
        }

        pointer end()
        {
            return &(data_[3]);
        }

        const_pointer end() const
        {
            return &(data_[3]);
        }

        value_type x() const { return data_[0]; }
        value_type y() const { return data_[1]; }
        value_type z() const { return data_[2]; }

        value_type& x() { return data_[0]; }
        value_type& y() { return data_[1]; }
        value_type& z() { return data_[2]; }

        explicit triple_array ( const value_type a = 0, const value_type b = 0, const value_type c = 0 )
        {
            data_[0] = a; data_[1] = b; data_[2] = c;
        }
        
        value_type norm() const 
        {
            const value_type m = std::max(std::abs(data_[0]), std::max(std::abs(data_[1]), std::abs(data_[2])));
            if ( value_type() == m ) return value_type();
            const value_type x = data_[0]/m;
            const value_type y = data_[1]/m;
            const value_type z = data_[2]/m;
            return m * std::sqrt(x*x +y*y +z*z);
        }

        const self_type operator - () const 
        {
            return self_type(-data_[0], -data_[1], -data_[2]);
        }

        friend std::ostream& operator << ( std::ostream& os, const self_type& self )
        {
            return os << " ("  << self[0] << ", " << self[1] << ", " << self[2] << ") " ;
        }

        template<std::size_t D, typename A>
        friend const self_type operator * ( const f::matrix<T,D,A>& lhs, const self_type& rhs )
        {
            assert( 3 == lhs.row() );
            assert( 3 == lhs.col() );
            return self_type(   std::inner_product( lhs.row_begin(0), lhs.row_end(0), rhs.begin(), value_type() ),
                                std::inner_product( lhs.row_begin(1), lhs.row_end(1), rhs.begin(), value_type() ),
                                std::inner_product( lhs.row_begin(2), lhs.row_end(2), rhs.begin(), value_type() ));
        }

        template<std::size_t D, typename A>
        friend const self_type operator * ( const self_type& lhs, const f::matrix<T,D,A>& rhs )
        {
            assert( 3 == rhs.row() ); 
            assert( 3 == rhs.col() ); 
            return self_type (  std::inner_product( lhs.begin(), lhs.end(), rhs.col_begin(0), value_type() ),
                                std::inner_product( lhs.begin(), lhs.end(), rhs.col_begin(1), value_type() ),
                                std::inner_product( lhs.begin(), lhs.end(), rhs.col_begin(2), value_type() ));
        }
        
        friend const self_type operator + ( const self_type& lhs, const self_type rhs )
        {
            return self_type(lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2]);
        }
        
        friend const self_type operator + ( const self_type& lhs, const value_type rhs )
        {
            return self_type(lhs[0]+rhs, lhs[1]+rhs, lhs[2]+rhs);
        }
        
        friend const self_type operator + ( const value_type lhs, const self_type& rhs )
        {
            return rhs + lhs;
        }
        
        friend const self_type operator - ( const self_type& lhs, const value_type rhs )
        {
            return self_type(lhs[0]-rhs, lhs[1]-rhs, lhs[2]-rhs);
        }
        
        friend const self_type operator - ( const self_type& lhs, const self_type rhs )
        {
            return self_type(lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2]);
        }

        friend const self_type operator * ( const self_type& lhs, const value_type rhs )
        {
            return self_type(lhs[0]*rhs, lhs[1]*rhs, lhs[2]*rhs);
        }
        
        friend const self_type operator * ( const value_type lhs, const self_type& rhs )
        {
            return rhs * lhs;
        }
        
        friend const self_type operator / ( const self_type& lhs, const value_type rhs )
        {
            return self_type(lhs[0]/rhs, lhs[1]/rhs, lhs[2]/rhs);
        }
        
        friend bool operator == ( const self_type& lhs, const self_type& rhs )
        {
            return  lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
        }
        
        friend bool operator != ( const self_type& lhs, const self_type& rhs )
        {
            return  lhs[0] != rhs[0] || lhs[1] != rhs[1] || lhs[2] != rhs[2];
        }
        
        friend bool operator < ( const self_type& lhs, const self_type& rhs )
        {
            if ( lhs[0] < rhs[0] ) return true;
            if ( lhs[0] > rhs[0] ) return false;
            if ( lhs[1] < rhs[1] ) return true;
            if ( lhs[1] > rhs[1] ) return false;
            if ( lhs[2] < rhs[2] ) return true;
            return false;
        }
        
        friend bool operator <= ( const self_type& lhs, const self_type& rhs )
        {
            return (lhs < rhs) || (lhs == rhs);
        }
        
        friend bool operator > ( const self_type& lhs, const self_type& rhs )
        {
            if ( lhs[0] > rhs[0] ) return true;
            if ( lhs[0] < rhs[0] ) return false;
            if ( lhs[1] > rhs[1] ) return true;
            if ( lhs[1] < rhs[1] ) return false;
            if ( lhs[2] > rhs[2] ) return true;
            return false;
        }
        
        friend bool operator >= ( const self_type& lhs, const self_type& rhs )
        {
            return (lhs > rhs) || (lhs == rhs);
        }

    };//struct triple_array

    template<typename T>
    triple_array<T> const scale_multiply( const triple_array<T>& lhs, const triple_array<T>& rhs )
    {
        return triple_array<T>( lhs[0]*rhs[0], lhs[1]*rhs[1], lhs[2]*rhs[2] );
    }

    template<typename T>
    T inner_product( const triple_array<T>& lhs, const triple_array<T>& rhs )
    {
        return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), T(0));
    }

    template<typename T>
    const triple_array<T> cross_product( const triple_array<T>& lhs, const triple_array<T>& rhs )
    {
        return triple_array<T>( lhs[1]*rhs[2] - rhs[1]*lhs[2], lhs[2]*rhs[0] - rhs[2]*lhs[0], lhs[0]*rhs[1] - rhs[0]*lhs[1] );
    }

    template<typename T>
    T included_angle( const triple_array<T>& lhs, const triple_array<T>& rhs )
    {
        const T ln = lhs.norm();
        const T rn = rhs.norm();
        const T lr = inner_product(lhs, rhs);
        if ( T() == lr ) return T();
        return std::acos(lr/(ln*rn));
    }

};//namespace f`

#endif//_ARRAY_HPP_INCLUDED_SDFIOJ9438USLKFDJDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDSFD8E3I

