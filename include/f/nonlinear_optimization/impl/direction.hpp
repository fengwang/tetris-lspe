#ifndef ELWFFRISVNOGPSAEACTBTVRAQRRGQFMUINBIDCOVONLPDYVQIHLRNREJDJDWNFEAYATAPAIIS
#define ELWFFRISVNOGPSAEACTBTVRAQRRGQFMUINBIDCOVONLPDYVQIHLRNREJDJDWNFEAYATAPAIIS

#include <f/matrix/numeric/conjugate_gradient_squared.hpp>
#include <f/matrix/numeric/lu_solver.hpp>

#include <cstddef>
#include <cmath>
#include <algorithm>

namespace f
{
    template< typename T, typename Zen >
    struct direction
    {
        typedef T                                       value_type;
        typedef Zen                                     zen_type;
        typedef std::size_t                             size_type;

        int make_direction()
        {
            auto& zen = static_cast<zen_type&>(*this);
            return zen.setup_direction();
        }

        /*
         *  return
         *      0           ----        success
         *      1           ----        failed
         */
        int setup_direction()
        {
            auto& zen = static_cast<zen_type&>(*this);

            zen.make_jacobian();
            zen.make_hessian();

            size_type const n = zen.unknown_variables;
            size_type const loops = std::max( 20, std::ceil(std::sqrt(n)) );

            //TODO:
            //          more linear solver here
            if ( conjugate_gradient_squared( zen.hessian_matrix, zen.fitting_array, zen.jacobian_array, loops ) )
                return ( lu_solver( zen.hessian_matrix, zen.fitting_array, zen.jacobian_array ) )

            return 0;
        }
    };

    template< typename T, typename Zen >
    struct levenberg_marquardt_direction : direction<T, levenberg_marquardt_direction<T, Zen> >
    {
        typedef T                                       value_type;
        typedef Zen                                     zen_type;
        typedef std::size_t                             size_type;

        value_type                                      lambda;
        value_type                                      lambda_factor;

        void make_lambda_factor( value_type lambda_factor_ = value_type{1.618} )
        {
            auto& zen = static_cast<zen_type&>(*this);
            zen.setup_lambda_factor(lambda_factor_);
        }

        void make_lambda( value_type lambda_ = value_type{1.618e-5} )
        {
            auto& zen = static_cast<zen_type&>(*this);
            zen.setup_lambda( lambda_ );
        }

        void setup_lambda_factor( value_type lambda_factor_ = value_type{1.618} )
        {
            auto& zen = static_cast<zen_type&>(*this);
            zen.lambda_factor = lambda_factor;
        }

        void setup_lambda( value_type lambda_ = value_type{1.618e-5} )
        {
            auto& zen = static_cast<zen_type&>(*this);
            zen.lambda = lambda_; 
        }

        int setup_direction()
        {
            auto& zen = static_cast<zen_type&>(*this);

            zen.make_jacobian();
            zen.make_hessian();
           
            for (;;)
            {    
                if ( std::isnan(lambda) ) return 1;
                if(zen.setup_new_direction() ) return 1;

                //calculate chi_square
                value_type const old_chi_square = zen.chi_square;
                zen.make_chi_square();
                value_type const new_chi_square = zen.chi_square;

                //if chi_square is good
                if ( new_chi_square < old_chi_square )
                {
                    lambda *= lambda_factor_;
                    break;
                }

                //if chi_square is bad
                lambda /= lambda_factor_;
                zen.make_chi_square( old_chi_square );
            }

            return 0;
        }

        int setup_new_direction()
        {
            auto& zen = static_cast<zen_type&>(*this);

            size_type const n = zen.unknown_variables;
            size_type const loops = std::max( 100, std::ceil(std::sqrt(n)) );

            //modify hessian
            std::for_each( zen.hessian_matrix.diag_begin(), zen.hessian_matrix.diag_end(), [&](value_type x) { x *= value_type{1}+zen.lambda; } );

            if ( conjugate_gradient_squared( zen.hessian_matrix, zen.fitting_array, zen.jacobian_array, loops ) )
            {
                if ( lu_solver( zen.hessian_matrix, zen.fitting_array, zen.jacobian_array ) )
                    return 1;
            }

            return 0;
        }
    };//

}//namespace f

#endif//ELWFFRISVNOGPSAEACTBTVRAQRRGQFMUINBIDCOVONLPDYVQIHLRNREJDJDWNFEAYATAPAIIS

