#ifndef RJQCMXRAIWENPJTJRPNNUNJVYRBENRUUKSELATYPOKXIGIFXRGNCAEFWXXXBWXARHTPGDLETW
#define RJQCMXRAIWENPJTJRPNNUNJVYRBENRUUKSELATYPOKXIGIFXRGNCAEFWXXXBWXARHTPGDLETW

#include <cstddef>
#include <cmath>
#include <algorithm>

#include <f/matrix/matrix.hpp>

namespace f
{

    template<typename T, typename Zen>
    struct iteration
    {
        typedef T                               value_type;
        typedef matrix<value_type>              matrix_type;
        typedef Zen                             zen_type;
        typedef std::size_t                     size_type;

        size_type make_max_iteration_step() const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return zen.setup_max_iteration_step();
        }

        //returns
        //      1       ----        failed to find a better solution
        //      0       ----        successful
        int make_iteration()
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return zen.setup_iteration();
        }

        size_type setup_max_iteration_step()
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            size_type const sqr_u = std::ceil( std::sqrt( zen.unknown_variables ) );
            return std::max( 1000, sqr_u );
        }

        int setup_iteration()
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            //calculate direction da
            if( zen.make_direction() ) return 1;
            //TODO: -- stop conditions here
            //calculate step size
            value_type alpha = zen.make_step();
            //updata current a
            zen.fitting_array += alpha * zen.direction_array;
            //update chi_quare
            zen.make_chi_square();
            //increase current_step
            ++zen.current_step;
            return 0;
        }
    };
/*
    template<typename T, typename Zen>
    struct levenberg_marquardt_iteration : iteration< T, levenberg_marquardt_iteration<T, Zen> >
    {
        typedef T                               value_type;
        typedef matrix<value_type>              matrix_type;
        typedef Zen                             zen_type;
        typedef std::size_t                     size_type;

        int setup_iteration()
        {
            zen_type& zen = static_cast<zen_type&>( *this );




            return 0;
        }

    };//
*/

}//namespace f

#endif//RJQCMXRAIWENPJTJRPNNUNJVYRBENRUUKSELATYPOKXIGIFXRGNCAEFWXXXBWXARHTPGDLETW

