#ifndef EAWGJNFQJUGQECKUVDTWJHGAORYWLIRBMIDWROVQIVBWYTHMYOKECSRDMODMUQUSMJPGDQNIE
#define EAWGJNFQJUGQECKUVDTWJHGAORYWLIRBMIDWROVQIVBWYTHMYOKECSRDMODMUQUSMJPGDQNIE

#include <f/matrix/matrix.hpp>
#include <f/derivative/derivative.hpp>
#include <f/derivative/second_derivative.hpp>

#include <cassert>

namespace f
{
    template<typename T, typename Zen>
    struct target_function
    {
        typedef T                                           value_type;
        typedef value_type*                                 pointer;
        typedef matrix<value_type>                          matrix_type;
        typedef Zen                                         zen_type;
        typedef std::size_t                                 size_type;

        //Input:
        //      a vector representing the parameter to be fitted
        //Return:
        //      modify the default chi_square to a new value
        void make_chi_square(pointer a)
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            chi_square = zen.setup_chi_square(a);
        }

        void make_chi_square()
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            make_chi_square( &(zen.fitting_array[0][0]) );
        }

        void make_chi_square( value_type const chi_square_ )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            chi_square = chi_square_;
        }

        //value_type setup_chi_square()

    };//struct target_function

    //the model to fit is a function  
    //      y = f( x, a )
    //  where a[m] is the parameter array to be fitter,  x[?] is known vector set,
    //  and there are n set of experimental data{ [y_0, x_0[?]], [y_1, x_1[?]], ... [y_{n-1}, x_{n-1}[?]] ]}, 
    //  i.e.
    //      y_0 = f( x_0, a )
    //      y_1 = f( x_1, a )
    //      ......
    //      y_{n-1} = f( x_{n-1}, a )
    //                          
    template<typename T, typename Zen>
    struct levenberg_marquardt_target_function : target_function< T, levenberg_marquardt_target_function< T, Zen > >
    {
        typedef T                                           value_type;
        typedef value_type*                                 pointer;
        typedef matrix<value_type>                          matrix_type;
        typedef Zen                                         zen_type;
        typedef std::size_t                                 size_type;

        matrix_type                                         y;//[m][1]
        matrix_type                                         x;//[m][n]

        void make_y_array( matrix_type const& y_ )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            zen.setup_y_array( y_ );
        }

        void make_x_matrix( matrix_type const& x_ )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            zen.setup_x_matrix( x_ );
        }

        value_type make_chi_square(pointer a)
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return zen.setup_chi_square( a );
        }

        value_type setup_chi_square(pointer a)
        {
            zen_type& zen = static_cast<zen_type&>( *this );

            assert( y.row() );
            assert( y.row() == x .row() );

            value_type ans{0};

            size_type const m = (zen.y).row();

            for ( size_type r = 0; r != m; ++r )
            {
                value_type diff = (zen.y)[r] - zen.setup_model( &((zen.x)[r][0]), a );
                ans += diff * diff;
            }
            return ans;
        }

        //value_type setup_model( pointer x, pointer a )

        void setup_y_array( matrix_type const& y_ )
        {
            auto& zen = static_cast<zen_type&>( *this );
            zen.y = y_;
        }

        void setup_x_matrix( matrix_type const& x_ )
        {
            auto& zen = static_cast<zen_type&>( *this );
            zen.x = x_;
        }

    };//levenberg_marquardt_target_function

}//namespace f

#endif//EAWGJNFQJUGQECKUVDTWJHGAORYWLIRBMIDWROVQIVBWYTHMYOKECSRDMODMUQUSMJPGDQNIE

