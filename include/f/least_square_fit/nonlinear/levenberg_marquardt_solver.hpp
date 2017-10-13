#ifndef MASFDPISDFKLJ489VNSAJFDKALKJALJKD32098AUOIY3KDAHFLKJ2OUHAFDLKJAHFLKJAHF8
#define MASFDPISDFKLJ489VNSAJFDKALKJALJKD32098AUOIY3KDAHFLKJ2OUHAFDLKJAHFLKJAHF8

#include <f/algorithm/max.hpp>
#include <f/algorithm/any_of.hpp>
#include <f/algorithm/all_of.hpp>
#include <f/matrix/matrix.hpp>
#include <f/derivative/derivative.hpp>
#include <f/derivative/second_derivative.hpp>
#include <f/matrix/numeric/lu_solver.hpp>
#include <f/variate_generator/variate_generator.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <functional>

namespace f
{
    namespace levenberg_marquardt_solver__private_dsfpoinaslfkj43iouh
    {

        template< typename T >
        struct levenberg_marquardt_solver_impl
        {
            typedef T                                           value_type;
            typedef value_type*                                 pointer;
            typedef const value_type*                           const_pointer;
            typedef matrix<value_type>                          matrix_type;
            typedef std::size_t                                 size_type;
            typedef std::function<value_type(pointer)>          function_type;
            typedef matrix<function_type>                       function_matrix_type;

            function_type           X;
            size_type               n;

            function_matrix_type    f_alpha;
            function_matrix_type    f_beta;

            template< typename Chi_Function >
            levenberg_marquardt_solver_impl( Chi_Function const& X_, const size_type n_ ) : X(X_), n(n_), f_alpha(n, n), f_beta(n, 1)
            {
                for ( size_type r = 0; r != n; ++r )
                    for ( size_type c = 0; c != n; ++ c )
                    {
                        f_alpha[r][c] = make_second_derivative( X, r, c );
                        assert( f_alpha[r][c] );
                    }

                for ( size_type r = 0; r != n; ++r )
                {
                    f_beta[r][0] = make_derivative( X, r );
                    assert( f_beta[r][0] );
                }
            }

            value_type chi_chi( matrix_type& a ) const
            {
                return X(a.data());
            }

            //randomly generate initial value for parameter a
            /*
            void random_initialize_a( matrix_type& a ) const
            {
                std::fill( a.begin(), a.end(), value_type(0.5) );
            }
            */
            void random_initialize_a( matrix_type& a, value_type x = value_type(0.5) ) const
            {
                value_type const central = x;
                value_type const variance = value_type(1) + std::abs(x);
                variate_generator<value_type, gaussian> vg( central, variance );
                std::generate( a.begin(), a.end(), vg );
                //std::fill( a.begin(), a.end(), x );
            }

            //beta[k] is the inner product of \delta y and k-th column of \partial y / \partial a
            void a__beta( matrix_type& a, matrix_type& beta ) const
            {
                //std::cout << "\nlm solver -- updating beta\n";
                assert( a.row() == n );
                assert( a.col() == 1 );
                assert( beta.row() == n );
                assert( beta.col() == 1 );

                for ( size_type i = 0; i != n; ++i )
                    beta[i][0] = (f_beta[i][0])( a.data() );

                //std::cout << "\nbeta is calculated as:\n" << beta << "\n";

                beta *= value_type(-0.5);
            }

            //alpha[i][j] is the inner_product of the i-th and the j-th column of \partial y / \partial a
            void a__alpha( matrix_type& a, matrix_type& alpha ) const
            {
                //std::cout << "\nlm solver -- updating alpha\n";
                assert( a.row() == n );
                assert( a.col() == 1 );
                assert( alpha.row() == n );
                assert( alpha.col() == n );

                for ( size_type r = 0; r != n; ++r )
                    for ( size_type c = 0; c <= r; ++c )
                    {
                        alpha[r][c] = (f_alpha[r][c])(a.data());
                        alpha[c][r] = alpha[r][c];
                    }

                //std::cout << "\nalpha matrix is calculated as\n" << alpha << "\n";

                alpha *= value_type(0.5);
            }

            //alpha'[i][j] = alpha[i][j]                    if i != j
            //alpha'[i][j] = alpha[i][j] * (l+\lambda)      if i == j
            void alpha_lambda__alpha_pri( const matrix_type& alpha, const value_type lambda, matrix_type& alpha_pri ) const
            {
                //std::cout << "\nlm solver -- updating alpha_pri\n";
                assert( alpha.row() == n );
                assert( alpha.col() == n );
                assert( alpha_pri.row() == n );
                assert( alpha_pri.col() == n );

                std::copy( alpha.begin(), alpha.end(), alpha_pri.begin() );
                //auto const& max_one = *std::max_element( alpha.begin(), alpha.end(), [](value_type lhs, value_type rhs){ return std::abs(lhs) < std::abs(rhs); } );
                std::transform( alpha.diag_begin(), alpha.diag_end(), alpha_pri.diag_begin(), [lambda](value_type const x){ return x + x*lambda; } );
                //std::transform( alpha.diag_begin(), alpha.diag_end(), alpha_pri.diag_begin(),
                //                [lambda, max_one](value_type const x){ if ( value_type() == x ) return std::abs(max_one) + x + x*lambda; else return x + x*lambda; } );

                //std::cout << "\nalpha_pri matrix is calculated as\n" << alpha_pri << "\n";
            }

            //Solve a linear equation to get \delta a
            // \alpha' \delta a = \beta
            void alpha_pri_beta__delta_a( const matrix_type& alpha_pri, const matrix_type& beta, matrix_type& delta_a ) const
            {
                //std::cout << "\nlm solver -- updating delta_a\n";
                assert( alpha_pri.row() == n );
                assert( alpha_pri.col() == n );
                assert( beta.row() == n );
                assert( beta.col() == 1 );
                assert( delta_a.row() == n );
                assert( delta_a.col() == 1 );

                if ( lu_solver(alpha_pri, delta_a, beta ) )
                {
                    auto alpha_pri_ = alpha_pri;
                    auto const& max_one = std::abs( *std::max_element( alpha_pri.begin(), alpha_pri.end(), []( value_type lhs, value_type rhs ) { return std::abs(lhs)<std::abs(rhs); } ) );
                    std::transform( alpha_pri.diag_begin(), alpha_pri.diag_end(), alpha_pri_.diag_begin(),
                                    [max_one]( value_type x ) { if ( x > value_type(0) ) return max_one + x; return -max_one+x; } );

                    if ( all_of( alpha_pri_.begin(), alpha_pri_.end(), []( value_type x ) { return x == value_type(0); } ) )
                        std::fill( alpha_pri_.diag_begin(), alpha_pri_.diag_end(), value_type(1) );

                    if ( lu_solver(alpha_pri_, delta_a, beta ) )
                    {
                        /*
                        std::cout << "\nalpha_pri_=\n" << alpha_pri_ << "\n";
                        std::cout << "\nalpha_pri=\n" << alpha_pri << "\n";
                        std::cout << "\nbeta=\n" << beta << "\n";
                        std::cout << "\ndeltai_a = \n" << delta_a << std::endl;
                        */
                        assert( !"levenberg_marquardt_impl::alpha_pri_beta__delta_a() : failed to solve the linear equation using LU decomposition!" );
                    }
                }

                //std::cout << "\ndelta_a matrix is calculated as\n" << delta_a << "\n";
            }

            template<typename A_Out_Iterator>
            value_type operator()( A_Out_Iterator a_first, value_type initial_guess = 0.5 ) const
            {
                matrix_type         a( n, 1 );                      //[m,1]
                value_type          residual;
                matrix_type         beta( n, 1 );                   //[m,1]
                matrix_type         delta_a( n, 1 );                //[m,1]
                matrix_type         alpha( n, n );                  //[m,m]
                matrix_type         alpha_pri( n, n );              //[m,m]
                value_type          lambda = 1.6180339887498948482e-5;
                value_type          lambda_ratio = 1.6180339887498948482;
                matrix_type         new_a( n, 1 );                  //[m,1]
                matrix_type         new_delta_y( n, 1 );            //[n,1]
                value_type          new_residual;

                random_initialize_a( a, initial_guess );
                residual = chi_chi( a );

                a__beta( a, beta );
                a__alpha( a, alpha );
                //variate_generator<value_type> lambda_factor( std::sqrt(lambda_ratio), lambda_ratio*lambda_ratio );
                variate_generator<value_type> lambda_factor( lambda_ratio, lambda_ratio*lambda_ratio );

                for (;;)
                {
                    alpha_lambda__alpha_pri( alpha, lambda, alpha_pri );
                    alpha_pri_beta__delta_a( alpha_pri, beta, delta_a );
                    new_a = a + delta_a;
                    new_residual = chi_chi( new_a );

                    std::cout << "\ncurrent residual is \t" << residual;
                    std::cout << "\nnew     residual is \t" << new_residual << "\n";
                    std::cout << "\nlambda           is \t" << lambda << "\n";

                    if ( std::isnan(new_residual) || std::isinf(new_residual) )
                    {
                        std::cout << "\nlm solver: new_residual is nan, halting......\n";
                        std::copy( a.col_begin(0), a.col_end(0), a_first );
                        return residual;

                        /*
                        assert( !"Unknown error occurs in levenberg_marquardt_impl::operator()!" );
                        return new_residual;
                        */
                    }

                    //if current solution is good engough, exit here
                    //TODO fix the bug in any of
                    //if ( std::isinf(lambda) || std::isnan(lambda) || any_of( alpha_pri.begin(), alpha_pri.end(), []( value_type x ) { return std::isinf(x)||std::isnan(x); } ) )
                    if ( std::isinf(lambda) || std::isnan(lambda) ) //|| any_of( alpha_pri.begin(), alpha_pri.end(), []( value_type x ) { return std::isinf(x)||std::isnan(x); } ) )
                    /*
                    if ( std::isinf(lambda) ||
                         std::isnan(lambda) ||
                         any_of( alpha_pri.begin(), alpha_pri.end(), []( value_type x ) { return std::isinf(x)||std::isnan(x); } )  ||
                         any_of( new_a.begin(), new_a.end(), []( value_type x ) { return std::isinf(x)||std::isnan(x); } )
                       )
                    */
                    {
                        if ( residual < new_residual )
                            std::copy( a.col_begin(0), a.col_end(0), a_first );
                        else
                            std::copy( new_a.col_begin(0), new_a.col_end(0), a_first );

                        return std::min(residual, new_residual);
                    }

                    if ( new_residual >= residual ) //not finding a better solution, increase lambda and search again
                    {
                        lambda *= lambda_factor();
                    }
                    else                          //a better solution found, update a with a+\delta a, decrease lambda and search again
                    {
                        std::cout << "\nlm: a new mininum found with residual of " << new_residual << "\n";
                        lambda /= lambda_factor();
                        a.swap( new_a );
                        std::swap( residual, new_residual );

                        a__beta( a, beta );
                        a__alpha( a, alpha );
                    }
                }

                assert(!"levenberg_marquardt_impl::operator() -- should never reach here!");

                return residual;//should never reach here
            }

        };//struct levenberg_marquardt_impl

    }//namespace levenberg_marquardt_solver__private_dsfpoinaslfkj43iouh

    template<typename Merit_Function, typename Output_Iterator>
    typename std::iterator_traits<Output_Iterator>::value_type levenberg_marquardt_solver( Merit_Function const& X, std::size_t n, Output_Iterator out, typename std::iterator_traits<Output_Iterator>::value_type const initial_guess = 0.5 )
    {
        typedef typename std::iterator_traits<Output_Iterator>::value_type value_type;
        levenberg_marquardt_solver__private_dsfpoinaslfkj43iouh::levenberg_marquardt_solver_impl<value_type> const solver( X, n );
        return solver( out, initial_guess );
    }

}//namespace f

#endif//_ASFDPISDFKLJ489VNSAJFDKALKJALJKD32098AUOIY3KDAHFLKJ2OUHAFDLKJAHFLKJAHF8

