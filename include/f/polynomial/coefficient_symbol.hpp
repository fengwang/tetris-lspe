#ifndef YGRLTDGOQKGJECAXUJVLYJERYAOFHCCXXAJKHJPEQFUIKASOMEYNQMRAIQCJVVXMTBNURYFFV
#define YGRLTDGOQKGJECAXUJVLYJERYAOFHCCXXAJKHJPEQFUIKASOMEYNQMRAIQCJVVXMTBNURYFFV

#include <f/polynomial/symbol.hpp>
#include <f/polynomial/complex_symbol.hpp>
#include <f/singleton/singleton.hpp>

#include <iostream>

namespace f
{
    enum coefficient_symbol_tag
    {
        A,
        C1,
        C2
    };

    template<typename T>
    struct coefficient_symbol : symbol<T, coefficient_symbol<T>>
    {
        typedef T                                              value_type;
        typedef unsigned long                                  size_type;
        typedef coefficient_symbol                             self_type;
        typedef complex_symbol<value_type>                     complex_symbol_type;

        complex_symbol_type                                    c_symbol;
        coefficient_symbol_tag                                 tag;
        value_type                                             val;

        coefficient_symbol( complex_symbol_type const& c_symbol_, coefficient_symbol_tag tag_, value_type const val_ ) : c_symbol( c_symbol_ ), tag( tag_ ), val( val_ ) {}

        coefficient_symbol( self_type const& other ) : c_symbol( other.c_symbol ), tag( other.tag ), val( other.val ) { }

        value_type eval() const
        {
            if ( A == tag ) return eval( c_symbol );

            return val;
        }

        friend bool operator < ( self_type const& lhs, self_type const& rhs )
        {
            if ( lhs.c_symbol < rhs.c_symbol ) return true;
            if ( lhs.c_symbol > rhs.c_symbol ) return false;
            if ( lhs.tag < rhs.tag ) return true;
            if ( lhs.tag > rhs.tag ) return false;
            return lhs.val < rhs.val;
        }

        friend bool operator == ( self_type const& lhs, self_type const& rhs )
        {
            return lhs.c_symbol == rhs.c_symbol && lhs.tag == rhs.tag && lhs.val == rhs.val;
        }

        friend std::ostream& operator << ( std::ostream& os, self_type const& rhs )
        {
            os << "(";

            if ( rhs.tag == A )
                return os << "A(" << rhs.c_symbol << ")";

            if ( rhs.tag == C1 )
                return os << "C1(" << rhs.c_symbol << ")";

            return os << "C2(" << rhs.c_symbol << ")";
        }

    };

    template<typename T>
    coefficient_symbol<T> const make_coefficient_symbol( complex_symbol<T> const& c_symbol_, coefficient_symbol_tag tag_, T val_ = T{} )
    {
        return coefficient_symbol<T> { c_symbol_, tag_, val_ };
    }

    //A
    template<typename T>
    coefficient_symbol<T> const make_a_radius_symbol( const T& ref, unsigned long index )
    {
        return make_coefficient_symbol( make_symbol( ref, index, radius ), A, T{} );
    }

    template<typename T>
    coefficient_symbol<T> const make_a_cosine_symbol( const T& ref, unsigned long index )
    {
        return make_coefficient_symbol( make_symbol( ref, index, cosine ), A, T{} );
    }

    template<typename T>
    coefficient_symbol<T> const make_a_sine_symbol( const T& ref, unsigned long index )
    {
        return make_coefficient_symbol( make_symbol( ref, index, sine ), A, T{} );
    }

    //C1
    template<typename T>
    coefficient_symbol<T> const make_c1_radius_symbol( unsigned long index1, unsigned long index2, T value )
    {
        T const& ref = singleton<T>::instance();
        return make_coefficient_symbol( make_symbol( ref, index1*10000000000+index2, radius ), C1, value );
    }

    template<typename T>
    coefficient_symbol<T> const make_c1_cosine_symbol( unsigned long index1, unsigned long index2, T value )
    {
        T const& ref = singleton<T>::instance();
        return make_coefficient_symbol( make_symbol( ref, index1*10000000000+index2, cosine ), C1, value );
    }

    template<typename T>
    coefficient_symbol<T> const make_c1_sine_symbol( unsigned long index1, unsigned long index2, T value )
    {
        T const& ref = singleton<T>::instance();
        return make_coefficient_symbol( make_symbol( ref, index1*10000000000+index2, sine ), C1, value );
    }

    //C2
    template<typename T>
    coefficient_symbol<T> const make_c2_radius_symbol( unsigned long index1, unsigned long index2, unsigned long index3, T value )
    {
        T const& ref = singleton<T>::instance();
        return make_coefficient_symbol( make_symbol( ref, index1*1000000000000+index2*1000000+index3, radius ), C2, value );
    }

    template<typename T>
    coefficient_symbol<T> const make_c2_cosine_symbol( unsigned long index1, unsigned long index2, unsigned long index3, T value )
    {
        T const& ref = singleton<T>::instance();
        return make_coefficient_symbol( make_symbol( ref, index1*1000000000000+index2*1000000+index3, cosine ), C2, value );
    }

    template<typename T>
    coefficient_symbol<T> const make_c2_sine_symbol( unsigned long index1, unsigned long index2, unsigned long index3, T value )
    {
        T const& ref = singleton<T>::instance();
        return make_coefficient_symbol( make_symbol( ref, index1*1000000000000+index2*1000000+index3, sine ), C2, value );
    }

}//namespace f

#endif//YGRLTDGOQKGJECAXUJVLYJERYAOFHCCXXAJKHJPEQFUIKASOMEYNQMRAIQCJVVXMTBNURYFFV

