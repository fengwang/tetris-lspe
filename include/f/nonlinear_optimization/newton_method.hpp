#ifndef JPFRAWBIPMOWNMHIXOMYPHHGWIGXVRCPEWLGVBAFAOSXQVDYJCNTNTVSLHGTBYIKMYDPVKFXE
#define JPFRAWBIPMOWNMHIXOMYPHHGWIGXVRCPEWLGVBAFAOSXQVDYJCNTNTVSLHGTBYIKMYDPVKFXE

#include <f/matrix/matrix.hpp>
#include <f/matrix/numeric/lu_solver.hpp>
#include <f/matrix/numeric/singular_value_decomposition.hpp>
#include <f/algorithm/any_of.hpp>

#include <functional>
#include <algorithm>
#include <numeric>

#include <iostream>

namespace f
{
#if 0
Usage:
    struct my_own_nonlinear_problem : newton_method< double, my_own_nonlinear_problem >
    {
        typdef double value_type;
        typdef std::function<double(double*)> function_type;

        function_type const derivative( const unsigned long n ) const;           //must
        function_type const second_derivative( const unsigned long n ) const;    //must
        unsigned long unknown_variable_size() const;                            //must

        void generate_initial_guess( double* )const;                            //optional
        function_type const operator()( double* ) const;                        //optional -- return the optimization function, \Chi^2
        unsigned long loops() const;                                            //optional
        value_type gamma() const;                                               //optional
    };

    my_own_nonlinear_problem mp;
    double* ans;
    mp( ans );

Proposed Test Case:

    \Chi^2 = (x+1)^4

    Rosenbrock function

    http://www.uni-graz.at/imawww/kuntsevich/solvopt/results/moreset.html

#endif

    template< typename T, typename D >
    struct newton_method
    {
        typedef T                                       value_type;
        typedef matrix<value_type>                      matrix_type;
        typedef matrix<value_type>                      array_type;
        typedef T*                                      pointer;
        typedef std::function<value_type(pointer)>      function_type;
        typedef unsigned long                           size_type;
        typedef D                                       zen_type;

        mutable matrix_type                             array_x;

        function_type const derivative( const unsigned long m_ ) const
        {
            return function_type{nullptr};
        }
        function_type const second_derivative( const unsigned long m_, const unsigned long n_  ) const
        {
            return function_type{nullptr};
        }

        value_type calculate_derivative( const unsigned long m_, pointer p_ ) const  //optional
        {
            auto const& zen = static_cast< zen_type const& >( *this );

            return (zen.derivative( m_ ))( p_ );
        }

        value_type calculate_second_derivative( const unsigned long m_, const unsigned long n_, pointer p_ ) const 
        {
            auto const& zen = static_cast< zen_type const& >( *this );

            return (zen.second_derivative( m_, n_ ))( p_ );
        }

        //initial value
        void generate_initial_guess( pointer x_ ) const
        {
            auto const& zen = static_cast< zen_type const& >( *this );

            size_type const n = zen.unknown_variable_size();
            std::fill_n( x_, n, value_type() );
        }

        //iteration
        size_type loops() const
        {
            return 100;
        }

        value_type gamma() const
        {
            return 0.0618;
        }

        value_type eps() const
        {
            return value_type(1.0e-40);
        }

        void on_fitting_start()
        {}

        void on_fitting_end()
        {}

        void on_iteration_start( unsigned long )
        {}

        void on_iteration_end( unsigned long )
        {}

        bool stop_here()
        {
            return false;
        }

        template< typename Output_Iterator >
        void operator()( Output_Iterator itor )
        {
            auto& zen = static_cast< zen_type& >( *this );

            zen.on_fitting_start();

            size_type const n = zen.unknown_variable_size();

            matrix_type local_hessian( n, n );
            matrix_type local_jacobi( n, 1 );
            matrix_type& local_x = array_x;
            local_x.resize( n, 1 );

            zen.generate_initial_guess( local_x.data() );

            matrix_type U, S, V;

            size_type iteration = zen.loops();
            for ( size_type i = 0; i != iteration; ++i )
            {
                zen.on_iteration_start( i );

                //calc local_hessian
                for ( size_type r = 0; r != n; ++r )
                    for ( size_type c = 0; c <= r; ++c )
                    {
                        local_hessian[r][c] = zen.calculate_second_derivative( r, c, local_x.data() );

                        if ( c == r ) continue;

                        local_hessian[c][r] = local_hessian[r][c];
                    }
                //calc local_jacobi
                for ( size_type r = 0; r != n; ++r )
                    local_jacobi[r][0] = zen.calculate_derivative( r,  local_x.data() );
                //iterate
                //TODO: use more stable method here

                singular_value_decomposition( local_hessian, U, S, V );

                //patch for non-positive definite hessian matrix
                /*
                value_type const minimum_element = *std::min_element( S.diag_begin(), S.diag_end() );
                value_type const fix_eps = 1.0;
                if ( minimum_element < fix_eps )
                {
                    value_type diag_fix = std::abs(minimum_element) + fix_eps;
                    std::for_each( S.diag_begin(), S.diag_end(), [diag_fix]( value_type & x ) { x += diag_fix; } );
                }
                */

                std::for_each( S.diag_begin(), S.diag_end(), []( value_type& v ) { if ( std::abs(v) > 1.0e-10 ) v = 1/v; else v = 0.0; } );
                //std::for_each( S.diag_begin(), S.diag_end(), []( value_type& v ) { if ( v < value_type{} ) v = 0.0; } );

                auto const& diff =  V * S * U.transpose() * local_jacobi * zen.gamma();

                if ( std::any_of( diff.begin(), diff.end(), []( value_type x ) { return std::isnan( x ) || std::isinf( x ) ; } ) )
                    break;

                local_x -= diff;

                //local_x -= V * S * U.transpose() * local_jacobi * zen.gamma();
                //local_x -= local_hessian.inverse() * local_jacobi * zen.gamma();

                //if ||jacobi||^2 is small enough, quit
                value_type const residual = std::inner_product( local_jacobi.begin(), local_jacobi.end(), local_jacobi.begin(), value_type() );

                zen.on_iteration_end( i );

                if ( std::isnan(residual) || std::isinf(residual) ) break;

                if ( zen.eps() > residual ) break;

                if ( zen.stop_here() ) break;
            }

            std::copy( local_x.begin(), local_x.end(), itor );

            zen.on_fitting_end();
        }



    };//newton_method

}//namespace f

#endif//JPFRAWBIPMOWNMHIXOMYPHHGWIGXVRCPEWLGVBAFAOSXQVDYJCNTNTVSLHGTBYIKMYDPVKFXE
