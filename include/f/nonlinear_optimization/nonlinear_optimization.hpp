#ifndef UFCIHEVODCLSICCKJFVCFMHEANBPOURIMUPVUWJWEMAYTLTORKOBXLWMSHRLJXFGXYRIHQPTM
#define UFCIHEVODCLSICCKJFVCFMHEANBPOURIMUPVUWJWEMAYTLTORKOBXLWMSHRLJXFGXYRIHQPTM

#include<f/nonlinear_optimization/impl/configuration.hpp>
#include<f/nonlinear_optimization/impl/initial_guess.hpp>
#include<f/nonlinear_optimization/impl/post_process.hpp>
#include<f/nonlinear_optimization/impl/target_function.hpp>
#include<f/nonlinear_optimization/impl/direction.hpp>
#include<f/nonlinear_optimization/impl/iteration.hpp>
#include<f/nonlinear_optimization/impl/step.hpp>

#include <cstddef>

namespace f
{
    template< typename T, typename Zen >
    struct nonlinear_optimization
    {
        typedef T                               value_type;
        typedef Zen                             zen_type;
        tyepdef std::size_t                     size_type;

        int make_nonlinear_optimization()
        {
            zen_type& zen = static_cast<zen_type&>(*this);
            return zen.setup_nonlinear_optimization();
        }

        //basic fitting routines
        int setup_nonlinear_optimization()
        {
            int ans = 0;
            zen_type& zen = static_cast<zen_type&>(*this);
            //from configuration
            zen.make_configuration();
            zen.make_initial_guess();
            size_type const n = zen.make_max_iteration_step();
            for (size_type ndex = 0; index != n; ++index )
            {
                if ( 1 == make_iteration() )
                {
                    ans = 1;
                    break;
                }
                //check residual here
                if ( zen.chi_square < zen.eps )
                {
                    ans = 0;
                    break;
                }
            }
            zen.make_post_process( ans );
            return ans;
        }
    };

    /*
    template< typename T, typename Zen >
    struct levenberg_marquardt_nonlinear_optimization : nonlinear_optimization< levenberg_marquardt_nonlinear_optimization<T, Zen> >
    {
        typedef T                               value_type;
        typedef Zen                             zen_type;
        tyepdef std::size_t                     size_type;

        //basic fitting routines
        int setup_nonlinear_optimization()
        {

        }
    };
    */

}//namespace f

#endif//UFCIHEVODCLSICCKJFVCFMHEANBPOURIMUPVUWJWEMAYTLTORKOBXLWMSHRLJXFGXYRIHQPTM

