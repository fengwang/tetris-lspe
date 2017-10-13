#ifndef MHILB_HPP_INCLUDED_SFIOJ8398FDVJKZXM3984444444444444444444444444444444444444444444444444HHHH
#define MHILB_HPP_INCLUDED_SFIOJ8398FDVJKZXM3984444444444444444444444444444444444444444444444444HHHH

#include <f/matrix/matrix.hpp>

#include <algorithm>
#include <cstddef>

namespace f
{

    template<typename T,
             std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const hilb( const std::size_t n )
    {
        matrix<T,D,A> ans( n, n );
        for ( std::size_t i = 0; i < n; ++i )
            for ( std::size_t j = i; j < n; ++j )
            {
                ans[i][j] = T(1) /(i+j+1);
                ans[j][i] = ans[i][j];
            }

        return ans;
    }

    template<typename T,
             std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const hilbert( const std::size_t n )
    {
        return hilb<T,D,A>(n);
    }

    template<typename Matrix>
    Matrix const hilb( const std::size_t n, const Matrix& )
    {
        typedef typename Matrix::value_type value_type;
        Matrix ans( n, n );
        for ( std::size_t i = 0; i < n; ++i )
            for ( std::size_t j = i; j < n; ++j )
            {
                ans[i][j] = value_type(1) /(i+j+1);
                ans[j][i] = ans[i][j];
            }

        return ans;
    }

    template<typename Matrix>
    Matrix const hilbert( const std::size_t n, const Matrix& m )
    {
        return hilb( n, m );
    }

}//namespace f

#endif//_HILB_HPP_INCLUDED_SFIOJ8398FDVJKZXM3984444444444444444444444444444444444444444444444444HHHH

