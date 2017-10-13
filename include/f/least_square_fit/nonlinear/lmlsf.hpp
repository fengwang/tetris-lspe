#ifndef FJABCPAXVSDAWGOGSWVHNLSCINBNUDPMWROIABNVYXDWGFRXCVUPDPLSLILTPXJATEWAAJNTT
#define FJABCPAXVSDAWGOGSWVHNLSCINBNUDPMWROIABNVYXDWGFRXCVUPDPLSLILTPXJATEWAAJNTT

#include <f/derivative/derivative.hpp>
#include <f/matrix/matrix.hpp>
//#include <f/matrix/numeric/lu_solver.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numeric>

#include <iostream>

namespace f
{
    //f::levenberg_marquardt_least_square_fit( sincos(), 2, vx.row_begin(0), vx.row_end(0), vy.begin(), a_fit.begin() );
    // the model to fit is 
    //                      y = f( x, a );
    namespace levenberg_marquardt_least_square_fit_private
    {
        
        template< typename T, typename Function >
        struct levenberg_marquardt_impl
        {
            typedef T                           value_type;
            typedef f::matrix<value_type>       matrix_type;
            typedef std::size_t                 size_type;
            typedef Function                    function_type;

            size_type       n; //size of examples
            size_type       l; //length of array x
            size_type       m; //length of array a
            function_type   fxa; //the fitting function -- usage: y = fxa(x)(a); 

            matrix_type     x;                      //[n,l]
            matrix_type     y;                      //[n,1]

            //randomly generate initial value for parameter a
            void random_initialize_a( matrix_type& a ) const
            {
                std::copy( x.begin(), x.begin()+a.size(), a.begin() );
            }

            //calculate \delta y from given parameter a
            // \delta y = y - fxa(a;x)
            void a__delta_y( const matrix_type& a, matrix_type& delta_y ) const
            {
                assert( a.row() == m );
                assert( a.col() == 1 );
                assert( delta_y.row() == n );
                assert( delta_y.col() == 1 );

                for ( size_type i = 0; i != n; ++i )
                    //delta_y[i][0] = fxa( &(x[i][0]) )( &(a[0][0]) ) - y[i][0];
                    delta_y[i][0] = fxa( x.data() )( a.data() ) - y[i][0];
            }

            //calculate chi^2 from \delta y
            // \chi^2 = \sum \delta_y ^2
            void delta_y__chi_chi( const matrix_type& delta_y, value_type& chi_chi ) const
            {
                assert( delta_y.row() == n );
                assert( delta_y.col() == 1 );

                chi_chi = std::inner_product( delta_y.col_begin(0), delta_y.col_end(0), delta_y.col_begin(0), value_type() );
            }

            //calculate derivatives \partial y / \partial a for every x
            void a__partial_y_partial_a( matrix_type& a, matrix_type& partial_y_partial_a ) const
            {
                assert( a.row() == m );
                assert( a.col() == 1 );
                assert( partial_y_partial_a.row() == n );
                assert( partial_y_partial_a.col() == m );

                for ( size_type r = 0; r != n; ++r )
                {
                    auto const& fxa_x = fxa(&x[r][0]);
                    for ( size_type c = 0; c != m; ++c )
                    {
                        //auto const& ff = make_derivative( fxa(&x[r][0]), c );
                        auto const& ff = make_derivative( fxa_x, c );
                        //partial_y_partial_a[r][c] = ff( &a[0][0] );
                        partial_y_partial_a[r][c] = ff( a.data() );
                    }
                }
            }
            
            //beta[k] is the inner product of \delta y and k-th column of \partial y / \partial a 
            void partial_y_partial_a_delta_y__beta( const matrix_type& partial_y_partial_a, const matrix_type& delta_y, matrix_type& beta ) const
            {
                assert( partial_y_partial_a.row() == n );
                assert( partial_y_partial_a.col() == m );
                assert( delta_y.row() == n );
                assert( delta_y.col() == 1 );
                assert( beta.row() == m );
                assert( beta.col() == 1 );

                for ( size_type i = 0; i != m; ++i )
                    beta[i][0] = std::inner_product( partial_y_partial_a.col_begin(i), partial_y_partial_a.col_end(i), delta_y.col_begin(0), value_type() );
            }

            //alpha[i][j] is the inner_product of the i-th and the j-th column of \partial y / \partial a 
            void partial_y_partial_a__alpha( const matrix_type& partial_y_partial_a, matrix_type& alpha ) const
            {
                assert( partial_y_partial_a.row() == n );
                assert( partial_y_partial_a.col() == m );
                assert( alpha.row() == m );
                assert( alpha.col() == m );

                for ( size_type r = 0; r != m; ++r )
                    for ( size_type c = 0; c != m; ++c )
                        alpha[r][c] = std::inner_product( partial_y_partial_a.col_begin(r), partial_y_partial_a.col_end(r), partial_y_partial_a.col_begin(c), value_type() );
            }

            //alpha'[i][j] = alpha[i][j]                    if i != j
            //alpha'[i][j] = alpha[i][j] * (l+\lambda)      if i == j
            void alpha_lambda__alpha_pri( const matrix_type& alpha, const value_type lambda, matrix_type& alpha_pri ) const
            {
                assert( alpha.row() == m );
                assert( alpha.col() == m );
                assert( alpha_pri.row() == m );
                assert( alpha_pri.col() == m );

                std::copy( alpha.begin(), alpha.end(), alpha_pri.begin() );
                std::transform( alpha.diag_begin(), alpha.diag_end(), alpha_pri.diag_begin(), [lambda](value_type const x){ return x + x*lambda; } );
            }

            //Solve a linear equation to get \delta a
            // \alpha' \delta a = \beta
            void alpha_pri_beta__delta_a( const matrix_type& alpha_pri, const matrix_type& beta, matrix_type& delta_a ) const
            {
                assert( alpha_pri.row() == m );
                assert( alpha_pri.col() == m );
                assert( beta.row() == m );
                assert( beta.col() == 1 );
                assert( delta_a.row() == m );
                assert( delta_a.col() == 1 );

                delta_a = alpha_pri.inverse() * beta;

                //assert( 0 == lu_solver( alpha_pri, delta_a, beta ) );
            }

            void iterate( matrix_type&         a,
                          matrix_type&         delta_y,
                          value_type&          chi_chi, 
                          matrix_type&         partial_y_partial_a,
                          matrix_type&         beta,
                          matrix_type&         delta_a,
                          matrix_type&         alpha,
                          matrix_type&         alpha_pri,
                          value_type&          lambda,
                          value_type const&    lambda_factor,
                          matrix_type&         new_a,
                          matrix_type&         new_delta_y,
                          value_type&          new_chi_chi,
                          value_type&          difference,
                          value_type const&    threshold )
            {
                alpha_lambda__alpha_pri( alpha, lambda, alpha_pri );
                alpha_pri_beta__delta_a( alpha_pri, beta, delta_a );
                new_a = a - delta_a;
                a__delta_y( new_a, new_delta_y );
                delta_y__chi_chi( new_delta_y, new_chi_chi );
                difference = std::abs( chi_chi - new_chi_chi );

                if ( std::isnan(new_chi_chi) || std::isinf(new_chi_chi) )
                {
                    assert( !"Unknown error occurs in levenberg_marquardt_impl::operator()!" );
                }

                if ( new_chi_chi >= chi_chi ) //not finding a better solution, increase lambda and search again
                {
                    lambda *= (lambda_factor+1.0);
                }
                else                          //a better solution found, update a with a+\delta a, decrease lambda and search again
                {
                    lambda /= lambda_factor;
                    a.swap( new_a );
                    delta_y.swap( new_delta_y );
                    std::swap( chi_chi, new_chi_chi );

                    a__partial_y_partial_a( a, partial_y_partial_a );
                    partial_y_partial_a_delta_y__beta( partial_y_partial_a, delta_y, beta );
                    partial_y_partial_a__alpha( partial_y_partial_a, alpha );
                }
            }


            template<typename X_In_Iterator, typename Y_In_Iterator>
            levenberg_marquardt_impl( const function_type& f_, const size_type m_,
                                      X_In_Iterator x_first, X_In_Iterator x_last, 
                                      Y_In_Iterator y_first, Y_In_Iterator y_last )
            {
                const size_type nl = std::distance( x_first, x_last );
                n = std::distance( y_first, y_last );
                m = m_;
                l = nl / n;
                assert( l * n == nl ); //dimension check

                x.resize( n, l );
                y.resize( n, 1 );

                std::copy( x_first, x_last, x.begin() );
                std::copy( y_first, y_last, y.begin() );

                fxa = f_;
            }

            template<typename A_Out_Iterator>
            int operator()( A_Out_Iterator a_first ) const 
            {
                matrix_type         a( m, 1 );                      //[m,1]
                matrix_type         delta_y( n, 1 );                //[n,1]
                value_type          chi_chi; 
                matrix_type         partial_y_partial_a( n, m );    //[n,m]
                matrix_type         beta( m, 1 );                   //[m,1]
                matrix_type         delta_a( m, 1 );                //[m,1]
                matrix_type         alpha( m, m );                  //[m,m]
                matrix_type         alpha_pri( m, m );              //[m,m]
                value_type          lambda = 1.6180339887498948482e-5;
                //value_type const    lambda_factor = 1.6180339887498948482;
                value_type const    lambda_factor = 10.0;
                matrix_type         new_a( m, 1 );                  //[m,1]
                matrix_type         new_delta_y( n, 1 );            //[n,1]
                value_type          new_chi_chi;
                value_type          difference;
                value_type const    threshold = 1.0e-15;

                random_initialize_a( a );
                a__delta_y( a, delta_y );
                delta_y__chi_chi( delta_y, chi_chi );

                a__partial_y_partial_a( a, partial_y_partial_a );
                partial_y_partial_a_delta_y__beta( partial_y_partial_a, delta_y, beta );
                partial_y_partial_a__alpha( partial_y_partial_a, alpha );

                for (;;)
                {
                    alpha_lambda__alpha_pri( alpha, lambda, alpha_pri );
                    alpha_pri_beta__delta_a( alpha_pri, beta, delta_a );
                    new_a = a - delta_a;
                    a__delta_y( new_a, new_delta_y );
                    delta_y__chi_chi( new_delta_y, new_chi_chi );
                    difference = std::abs( chi_chi - new_chi_chi );

                    //if ( std::isinf(lambda) || std::isnan(lambda) || std::isnan(new_chi_chi) || std::isinf(new_chi_chi) )
                    if ( std::isnan(new_chi_chi) || std::isinf(new_chi_chi) )
                    {
                        assert( !"Unknown error occurs in levenberg_marquardt_impl::operator()!" );
                        return 1;
                    }

                    //if current solution is good engough, exit here
                    //if ( difference < threshold * std::min(chi_chi, new_chi_chi)  )
                    if ( std::isinf(lambda) || std::isnan(lambda) )
                    {

                        if ( chi_chi < new_chi_chi )
                            std::copy( a.col_begin(0), a.col_end(0), a_first );
                        else
                            std::copy( new_a.col_begin(0), new_a.col_end(0), a_first );

                        return 0;
                    }
                    
                    if ( new_chi_chi >= chi_chi ) //not finding a better solution, increase lambda and search again
                    {
                        lambda *= (lambda_factor+1.0);
                    }
                    else                          //a better solution found, update a with a+\delta a, decrease lambda and search again
                    {
                        lambda /= lambda_factor;
                        //BUG: swap 
                        //a.swap( new_a );
                        a = new_a;
                        //delta_y.swap( new_delta_y );
                        delta_y = new_delta_y;
                        std::swap( chi_chi, new_chi_chi );

                        a__partial_y_partial_a( a, partial_y_partial_a );
                        partial_y_partial_a_delta_y__beta( partial_y_partial_a, delta_y, beta );
                        partial_y_partial_a__alpha( partial_y_partial_a, alpha );
                    }
                    
                }
                
                return 1;//should never reach here
            }

        };//struct levenberg_marquardt_impl

    }//namespace levenberg_marquardt_least_square_fit_private 

    /*
     *      Input:
     *              f           the function to be fitted, namely     y = f(x;a)  -- y = f(x)(a)
     */
    template<typename Function, typename X_In_Iterator, typename Y_In_Iterator, typename A_Out_Iterator>
    int levenberg_marquardt_least_square_fit( Function const& f, 
                                              std::size_t parameters_to_fit, 
                                              X_In_Iterator x_first, X_In_Iterator x_last, 
                                              Y_In_Iterator y_first, Y_In_Iterator y_last,
                                              A_Out_Iterator fitted_parameters_first )
    {
        typedef typename std::iterator_traits<X_In_Iterator>::value_type    value_type; 
        typedef Function                                                    function_type;

        using namespace levenberg_marquardt_least_square_fit_private;
        levenberg_marquardt_impl<value_type, function_type> lm( f, parameters_to_fit, x_first, x_last, y_first, y_last );

        return lm( fitted_parameters_first );
    }

}//namespace f

#endif//FJABCPAXVSDAWGOGSWVHNLSCINBNUDPMWROIABNVYXDWGFRXCVUPDPLSLILTPXJATEWAAJNTT

