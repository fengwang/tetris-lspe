#ifndef MIQYTETMJURCQAJATRRIBYNGGUPHJNEVIEKCDPFAIJIHPGNMXUUNSDIPAMNRBDGIMISJJCLHT
#define MIQYTETMJURCQAJATRRIBYNGGUPHJNEVIEKCDPFAIJIHPGNMXUUNSDIPAMNRBDGIMISJJCLHT

#include <f/matrix/matrix.hpp>
#include <f/matrix/numeric/lu_solver.hpp>
#include <f/matrix/numeric/singular_value_decomposition.hpp>
#include <f/algorithm/any_of.hpp>

#include <algorithm>
#include <numeric>

#include <iostream>

namespace f
{
#if 0
Usage:
    struct my_own_nonlinear_problem : sp_newton_method< double, my_own_nonlinear_problem >
    {
        typdef double value_type;

        //methods required
        value_type calcualte_derivate( unsigned long );
        value_type calculate_second_derivative( unsigned long, unsigned long );
        value_type residual();                                                      //calculate the current residual
        unsigned long unknown_variable_size() const;
    };
#endif

    template< typename T, typename D >
    struct sp_newton_method
    {
        typedef T                                       value_type;
        typedef matrix<value_type>                      matrix_type;
        typedef matrix<value_type>                      array_type;
        typedef T*                                      pointer;
        typedef unsigned long                           size_type;
        typedef D                                       zen_type;

        matrix_type                                     array_x;            //the unknowns to be optimized
        matrix_type                                     pk;                 //the derivative at current step
        matrix_type                                     hessian;            //the hessian matrix at current step
        matrix_type                                     jacobi;             //the jacobi matrix at current step

        matrix_type                                     U, S, V; //for SVD only

        void generate_initial_guess()
        {
            auto& zen = static_cast< zen_type& >( *this );

            size_type const n = zen.unknown_variable_size();
            std::fill_n( zen.array_x.begin(), n, value_type() );
        }

        //iteration
        size_type loops()
        {
            return 100;
        }

        value_type gamma( unsigned long )
        {
            return 0.618;
            //return 0.02148379396344907986; //\log(pi)/e^{pi}
        }

        value_type eps( unsigned long )
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

        bool stop_here( unsigned long )
        {
            return false;
        }

        /*
         *  return
         *              0           success
         *              1           failed
         */
        int iterate( unsigned long loop_index )
        {
            auto& zen = static_cast< zen_type& >( *this );

            ////!!
            zen.on_iteration_start( loop_index );

            //Generate Hessian
            size_type const n = zen.unknown_variable_size();
            for ( size_type r = 0; r != n; ++r )
                for ( size_type c = 0; c <= r; ++c )
                {
                    hessian[r][c] = zen.calculate_second_derivative( r, c );

                    if ( c == r ) continue;

                    hessian[c][r] = hessian[r][c];
                }

            //Generate Jacobi
            for ( size_type r = 0; r != n; ++r )
                jacobi[r][0] = zen.calculate_derivative( r );

            singular_value_decomposition( hessian, U, S, V );

            std::for_each( S.diag_begin(), S.diag_end(), []( value_type& v ) { if ( std::abs(v) > 1.0e-10 ) v = 1/v; else v = 0.0; } );

            //Generate Diff
            auto const& diff =  V * S * U.transpose() * jacobi;

            if ( std::any_of( diff.begin(), diff.end(), []( value_type x ) { return std::isnan( x ) || std::isinf( x ) ; } ) )
                return 1;

            //Update X
            array_x -= diff * zen.gamma( loop_index );;

            ////!!
            zen.on_iteration_end( loop_index );

            //Check termination condition
            value_type const residual = std::inner_product( jacobi.begin(), jacobi.end(), jacobi.begin(), value_type{} );
            if ( std::isnan(residual) || std::isinf(residual) ) return 1;

            if ( zen.eps( loop_index ) > residual ) return 1;

            if ( zen.stop_here( loop_index ) ) return 1;

            return 0;
        }

        void operator()()
        {
            auto& zen = static_cast< zen_type& >( *this );

            zen.on_fitting_start();

            size_type const n = zen.unknown_variable_size();

            hessian.resize( n, n );
            jacobi.resize( n, 1 );
            array_x.resize( n, 1 );

            zen.generate_initial_guess();

            for ( size_type i = 0; i != zen.loops(); ++i )
                if ( iterate( i ) ) break;

            zen.on_fitting_end();
        }

    };//newton_method

}//namespace f

#endif//MIQYTETMJURCQAJATRRIBYNGGUPHJNEVIEKCDPFAIJIHPGNMXUUNSDIPAMNRBDGIMISJJCLHT
