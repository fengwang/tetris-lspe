#ifndef MREPMAT_HPP_INCLUDED_FDSOIJ4398USFLAKJLSFKJLVCSJFDF89IU4098UAKSFJDLJFSKS2
#define MREPMAT_HPP_INCLUDED_FDSOIJ4398USFLAKJLSFKJLVCSJFDF89IU4098UAKSFJDLJFSKS2

#include <f/matrix/matrix.hpp>
#include <f/matrix/operators.hpp>

#include <cstddef>
#include <cassert>

namespace f
{
    template<typename T, std::size_t N, typename A>
    const matrix<T,N,A>
    repmat( const matrix<T,N,A>& m, const std::size_t r, const std::size_t c )
    {
        assert( r );
        assert( c );
        
        if ( 1 == r  && 1 == c ) return m;

        if ( 1 == r ) return repmat( m, 1, c-1 ) || m;

        if ( 1 == c ) return repmat( m, r-1, 1 ) && m;

        return repmat( repmat( m, 1, c), r, 1 );
    }

}//namespace f

#endif//_REPMAT_HPP_INCLUDED_FDSOIJ4398USFLAKJLSFKJLVCSJFDF89IU4098UAKSFJDLJFSKS2

