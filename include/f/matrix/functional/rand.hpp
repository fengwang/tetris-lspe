#ifndef MRAND_HPP_INCOUDED_SOFI4398UAFSDIJSLFDKJLVCKXJSAFDLSFDIJ4EIULASFDIJOISFDJAOSFIDJOSFDIU4EUSAL
#define MRAND_HPP_INCOUDED_SOFI4398UAFSDIJSLFDKJLVCKXJSAFDLSFDIJ4EIULASFDIJOISFDJAOSFIDJOSFDIU4EUSAL

#include <f/matrix/matrix.hpp>

#include <algorithm>
#include <cstddef>
#include <ctime>
#include <cstdlib>

namespace f
{
    // Parameters:
    //              r   ----    row of the return matrix
    //              c   ----    col of the return matrix
    // Return
    //          a matrix with size (r,c), and every element s.t. U(0,1)
    template<typename T = double, std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const rand( const std::size_t r, const std::size_t c )
    {
        matrix<T> ans{ r, c };
        std::srand(static_cast<unsigned int>(static_cast<std::size_t>(std::time(nullptr)) + reinterpret_cast<std::size_t>(&ans)));
        auto const generator = [](){ return T(std::rand())/T(RAND_MAX); };
        //std::fill( ans.diag_begin(), ans.diag_end(), T(1) );
        std::generate( ans.begin(), ans.end(), generator );
        return ans;
    }

    template<typename T = double, std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const rand( const std::size_t n )
    {
        return rand<T,D,A>( n, n );
    }

    template<typename T = double, std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const rand( const matrix<T,D,A>& m )
    {
        return rand<T,D,A>( m.row(), m.col() );
    }

    template<typename T = double, std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const ran( const std::size_t r, const std::size_t c )
    {
        return rand<T, D, A>( r, c );
    }

    template<typename T = double, std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const ran( const std::size_t n )
    {
        return rand<T, D, A>( n );
    }

    template<typename T = double, std::size_t D = 256,
             typename A = std::allocator<typename remove_const<typename remove_reference<T>::result_type>::result_type> >
    matrix<T,D,A> const ran( const matrix<T,D,A>& m )
    {
        return rand<T,D,A>( m.row(), m.col() );
    }

}//namespace f

#endif//_RAND_HPP_INCOUDED_SOFI4398UAFSDIJSLFDKJLVCKXJSAFDLSFDIJ4EIULASFDIJOISFDJAOSFIDJOSFDIU4EUSAL


