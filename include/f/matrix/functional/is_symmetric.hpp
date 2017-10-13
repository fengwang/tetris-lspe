#ifndef MIS_SYMMETRIC_HPP_INCLUDED_SOFDIJDDDDDDDD093ULFKJVMCNALKJSFDCNCNJDSHSDOIUOEIFDSFDA389FFFDFF 
#define MIS_SYMMETRIC_HPP_INCLUDED_SOFDIJDDDDDDDD093ULFKJVMCNALKJSFDCNCNJDSHSDOIUOEIFDSFDA389FFFDFF

#include <f/matrix/matrix.hpp>

#include <algorithm>
#include <cstddef>
#include <cassert>

namespace f
{

    template< typename T, std::size_t N, typename A, typename F >
    bool is_symmetric( const matrix<T,N,A>& m, F f )
    {
        if ( m.row() != m.col() ) return false;
        
        for ( std::size_t i = 1; i != m.row(); ++i )
            if ( ! std::equal( m.upper_diag_cbegin(i), m.upper_diag_cend(i), m.lower_diag_cbegin(i), f ) )
                return false;

       return true; 
    }

    template< typename T, std::size_t N, typename A >
    bool is_symmetric( const matrix<T,N,A>& m )
    {
        return is_symmetric( m, []( const T v1, const T v2 ) { return v1 == v2; } );
    }
}//namespace f

#endif//_IS_SYMMETRIC_HPP_INCLUDED_SOFDIJDDDDDDDD093ULFKJVMCNALKJSFDCNCNJDSHSDOIUOEIFDSFDA389FFFDFF

