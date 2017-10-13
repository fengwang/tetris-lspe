#ifndef AGUOIXMPUXLBMHSKADKFJWEMKKXFXUVQDPLGCRTJWWRNFJJFIRGHILCJCFDCVUULBNJXBIBUU
#define AGUOIXMPUXLBMHSKADKFJWEMKKXFXUVQDPLGCRTJWWRNFJJFIRGHILCJCFDCVUULBNJXBIBUU

#include <f/matrix/matrix.hpp>

#include <cstddef>

namespace f
{
    template< typename T, typename Zen >
    struct configuration
    {
        typedef Zen                             zen_type;
        typedef T                               value_type;
        typedef matrix<value_type>              matrix_type;
        typedef std::size_t                     size_type;

        size_type   unknown_variables;
        size_type   current_step;
        matrix_type fitting_array;              //1D
        matrix_type direction_array;            //1D
        matrix_type jacobian_array;             //1D
        matrix_type hessian_matrix;             //2D
        value_type  chi_square;
        value_type  eps;

        void make_configuration()
        {
            zen_type& zen = static_cast<zen_type&>(*this);
            zen.setup_configuration();
        }

        void setup_configuration()
        {
            zen_type& zen = static_cast<zen_type&>(*this);
            unknown_variables = zen.setup_unknown_variables();
            current_step = 0;
            fitting_array.resize(unknown_variables, 1);
            direction_array.resize(unknown_variables, 1);
            jacobian_array.resize(unknown_variables, 1);
            hessian_matrix.resize(unknown_variables, n);
            zen.setup_eps();
        }

        //this method must be implemented
        //
        //size_type setup_unknown_variables()
        //
        void setup_eps( value_type eps_ = value_type{1.0e-10} )
        {
            zen_type& zen = static_cast<zen_type&>(*this);
            zen.eps = eps_;
        }

    };

}//namespace f

#endif//AGUOIXMPUXLBMHSKADKFJWEMKKXFXUVQDPLGCRTJWWRNFJJFIRGHILCJCFDCVUULBNJXBIBUU

