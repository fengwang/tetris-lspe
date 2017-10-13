#ifndef MLINEAR_LEAST_SQUARE_FIT_HPP_INCLUDED_DOSIFUJ4OEIUSDLFKHJEOIUSDLFKJ4EOIRULFSDKJALSKFJDOEIRLSFDKJFDSKJFDSKLJSFDKJF
#define MLINEAR_LEAST_SQUARE_FIT_HPP_INCLUDED_DOSIFUJ4OEIUSDLFKHJEOIUSDLFKJ4EOIRULFSDKJALSKFJDOEIRLSFDKJFDSKJFDSKLJSFDKJF

#include <f/matrix/matrix.hpp>
#include <f/matrix/numeric/singular_value_decomposition.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <iterator>
#include <valarray>
#include <iterator>


//TODO:
//      fix the bug in llsf3

namespace f
{
    ///////////////////////////////////////////////////////////////////////////////
    // Forward Declaration
    ///////////////////////////////////////////////////////////////////////////////

    // the model to fit :
    //                  y(x) = \sum_{k=0}^{M} a_{k} f_{k}(x)
    //                  y(x) = a_0 * f_0(x) + a_1 * f_1(x) + ... + a_{m-1} * f_{m-1}(x)
    // with weigh array :
    //                  w
    // where x[n], y[n], w[n], f[m] are known to us.
    /*
     * Inputs:
     *          x_      begin positon of array x[n]
     *          _x      end position of array x[n]
     *          y_      begin position of array y[n]
     *          w_      begin position of array w[n] (this parameter is optional)
     *          f_      begin position of function array f[m]
     *          _f      end position of funciton array f[m]
     *          a_      begin positon of the fitted parameters array a[m]
     */
    template< typename II1, typename II2, typename II3, typename II4, typename OI >
    OI
    linear_least_square_fit( II1 x_, II1 _x, // x[n]
                             II2 y_,         // y[n]
                             II3 w_,         // w[n]
                             II4 f_, II4 _f, // f[m]
                             OI  a_ );       // a[m]

    // the model to fit is :
    //                  y(x) = \sum_{k=0}^{M} a_{k} f_{k}(x)
    // where x[n], y[n], f[m] are known to us.
    /*
     * Inputs:
     *          x_      begin positon of array x[n]
     *          _x      end position of array x[n]
     *          y_      begin position of array y[n]
     *          f_      begin position of function array f[m]
     *          _f      end position of funciton array f[m]
     *          a_      begin positon of the fitted parameters array a[m]
     */
    template< typename II1, typename II2, typename II3, typename OI >
    OI
    linear_least_square_fit( II1 x_, II1 _x, // x[n]
                             II2 y_,         // y[n]
                             II3 f_, II3 _f, // f[m]
                             OI  a_ );       // a[m]

    ///////////////////////////////////////////////////////////////////////////////
    // Implementation
    ///////////////////////////////////////////////////////////////////////////////

    template< typename II1, typename II2, typename II3, typename II4, typename OI >
    OI
    linear_least_square_fit( II1 x_, II1 _x, // x[n]
                             II2 y_,          // y[n]
                             II3 w_,          // w[n]
                             II4 f_, II4 _f,  // f[m]
                             OI  a_ )         // a[m]
    {
        typedef typename std::iterator_traits<II1>::value_type value_type;
        auto const n = std::distance( x_, _x );
        auto const m = std::distance( f_, _f );
        assert( n >= m );

        matrix<value_type> fx( m, n );
        for ( auto i = 0; i < m; ++i )
            for ( auto j = 0; j < n; ++j )
                fx[i][j] = (*(f_+i))(*(x_+j));

        //\alpha_{i,j} = \sum^{n-1}_{k=0} w_k^2 fx_{i,k} * fx_{j,k}
        matrix<value_type> alpha( m, m );
        for ( auto i = 0; i < m; ++i )
            for ( auto j = 0; j < m; ++j )
                for ( auto k = 0; k < n; ++k )
                    alpha[i][j] += w_[k] * w_[k] * fx[i][k] * fx[j][k];

        //beta_{i} = \sum_{j=0]^{n-1} w_j^2 * y_j * fx_{i,j}
        std::valarray<value_type> beta( m );
        for ( auto i = 0; i < m; ++i )
            for ( auto j = 0; j < n; ++j )
                beta[i] += w_[j] * w_[j] * y_[j] * fx[i][j];

        // solve \alpha b = \beta
        //      using SVD
        //              \alpha      = u w v^{T}
        //              \alpha^{-1} = v w^{-1} u^{T}
        //      get
        //              b = v * w^{-1} * u^{T} * beta
        matrix<value_type> u;
        matrix<value_type> w;
        matrix<value_type> v;
        singular_value_decomposition( alpha, u, w, v );
        std::for_each( w.diag_begin(), w.diag_end(), [](value_type& v){ v = value_type(1)/v; } );
        auto const ans_b = v * w * u.transpose() * beta;
        return std::copy( ans_b.begin(), ans_b.end(), a_ );
    }

    template< typename II1, typename II2, typename II3, typename OI >
    OI
    linear_least_square_fit( II1 x_, II1 _x, // x[n]
                             II2 y_,         // y[n]
                             II3 f_, II3 _f, // f[m]
                             OI  a_ )        // a[m]
    {
        typedef typename std::iterator_traits<II1>::value_type value_type;
        auto const n = std::distance( x_, _x );
        auto const m = std::distance( f_, _f );
        assert( n >= m );

        matrix<value_type> fx( m, n );
        for ( auto i = 0; i < m; ++i )
            for ( auto j = 0; j < n; ++j )
                fx[i][j] = (*(f_+i))(*(x_+j));

        //\alpha_{i,j} = \sum^{n-1}_{k=0} fx_{i,k} * fx_{j,k}
        matrix<value_type> alpha( m, m );
        for ( auto i = 0; i < m; ++i )
            for ( auto j = 0; j < m; ++j )
                alpha[i][j] = std::inner_product( fx.row_begin(i), fx.row_end(i), 
                                                  fx.row_begin(j), value_type() );

        //beta_{i} = \sum_{j=0]^{n-1} y_j * fx_{i,j}
        std::valarray<value_type> beta( m );
        for ( auto i = 0; i < m; ++i )
            beta[i] = std::inner_product( fx.row_begin(i), fx.row_end(i), y_, value_type() );

        /*
        // solve \alpha b = \beta
        //      using SVD
        //              \alpha      = u w v^{T}
        //              \alpha^{-1} = v w^{-1} u^{T}
        //      get
        //              b = v * w^{-1} * u^{T} * beta
        matrix<value_type> u;
        matrix<value_type> w;
        matrix<value_type> v;
        singular_value_decomposition( alpha, u, w, v );
        std::for_each( w.diag_begin(), w.diag_end(), [](value_type& v){ v = value_type(1)/v; } );
        auto const ans_b = v * w * u.transpose() * beta;
        */
        auto const ans_b = alpha.inverse() * beta;
        return std::copy( ans_b.begin(), ans_b.end(), a_ );
    }

    // the model to fit
    //  y = a x
    template< typename T, std::size_t D, typename A, typename II, typename OI >
    OI
    linear_least_square_fit( const matrix<T,D,A>& x, // x[n][m]
                             II y_,                  // y[n]
                             OI a_ )                 // a[m]
    {
        typedef T value_type;
        const std::size_t n = x.row();
        const std::size_t m = x.col();
        assert( n >= m );

        const matrix<value_type> fx = transpose(x);

        //\alpha_{i,j} = \sum^{n-1}_{k=0} fx_{i,k} * fx_{j,k}
        matrix<value_type> alpha( m, m );
        for ( std::size_t i = 0; i < m; ++i )
            for ( std::size_t j = 0; j < m; ++j )
                alpha[i][j] = std::inner_product( fx.row_begin(i), fx.row_end(i), fx.row_begin(j), value_type() );

        //beta_{i} = \sum_{j=0]^{n-1} y_j * fx_{i,j}
        std::valarray<value_type> beta( m );
        for ( std::size_t i = 0; i < m; ++i )
            beta[i] = std::inner_product( fx.row_begin(i), fx.row_end(i), y_, value_type() );

#if 1
        // solve \alpha b = \beta
        //      using SVD
        //              \alpha      = u w v^{T}
        //              \alpha^{-1} = v w^{-1} u^{T}
        //      get
        //              b = v * w^{-1} * u^{T} * beta
        matrix<value_type> u;
        matrix<value_type> w;
        matrix<value_type> v;
        singular_value_decomposition( alpha, u, w, v );
        //std::for_each( w.diag_begin(), w.diag_end(), [](value_type& v){ v = value_type(1)/v; } );
        std::for_each( w.diag_begin(), w.diag_end(), [](value_type& v){ if ( std::abs(v) < value_type{1.0e-20} ) v = value_type{}; else v = value_type{1}/v; } );
        auto const& ans_b = v * w * u.transpose() * beta;
#endif
#if 0
        auto const& ans_b = alpha.inverse() * beta;
#endif
        return std::copy( ans_b.begin(), ans_b.end(), a_ );
    }

}//namespace f

#endif//_LINEAR_LEAST_SQUARE_FIT_HPP_INCLUDED_DOSIFUJ4OEIUSDLFKHJEOIUSDLFKJ4EOIRULFSDKJALSKFJDOEIRLSFDKJFDSKJFDSKLJSFDKJF

