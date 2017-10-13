#ifndef CTSGPCYWEIBIIREUARVHOTEPUVMOJPUMJEKIHFTJOIXRNPATUTJAOXMOKNVGUSDTTRKQENUFV
#define CTSGPCYWEIBIIREUARVHOTEPUVMOJPUMJEKIHFTJOIXRNPATUTJAOXMOKNVGUSDTTRKQENUFV

#include <iostream>

namespace f
{
    template< typename T, typename Zen >
    struct post_process
    {
        typedef T               value_type;
        typedef Zen             zen_type;

        void make_post_process( int result )
        {
            zen_type& zen = static_cast<zen_type&>(*this);
            zen.setup_post_process( result );
        }

        void setup_post_process( int result )
        {
            zen_type& zen = static_cast<zen_type&>(*this);
            if ( 0 == result )
                std::cout << "\nSuccessfully fitted the problem.\n";
            else
                std::cout << "\nFailed to fit the problem.\n";
            std::cout << "\nthe residual is " << zen.chi_square
                      << "\nthe fitted parameters are:\n" << zen.fitting_function << "\n";
        }
    };

}//namespace f

#endif//CTSGPCYWEIBIIREUARVHOTEPUVMOJPUMJEKIHFTJOIXRNPATUTJAOXMOKNVGUSDTTRKQENUFV

