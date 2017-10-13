#ifndef CBJUSOGALAGWAIAKJHQOWGNRDVSYJQHVFSDRKJMWHCVVVABSCEPPCHVFVQRBEVNOSDADLAJPE
#define CBJUSOGALAGWAIAKJHQOWGNRDVSYJQHVFSDRKJMWHCVVVABSCEPPCHVFVQRBEVNOSDADLAJPE

#include <f/matrix/matrix.hpp>

namespace f
{
    template< typename Refinement >
    struct tilt_matrix
    {
        typedef Refinement  zen_type;
/*
 *  Format for the tilt matirx:
 *
 *  [index] [tilt_x-ideal-mrad] [tilt_y-ideal-mrad] [tilt_x-ac] [tilt_y-ac] [tilt_x-mrad] [tilt_y-mrad] [tilt_x-nm] [tilt_y-nm] [intensity]
 *
 */
        matrix<double>      the_tilt_matrix;

        void load_tilt_matrix()
        {
            auto& zen = static_cast<zen_type&>(*this); 
            the_tilt_matrix.load( zen.the_configuration.tilt_path );
        }
    };

}//namespace f

#endif//CBJUSOGALAGWAIAKJHQOWGNRDVSYJQHVFSDRKJMWHCVVVABSCEPPCHVFVQRBEVNOSDADLAJPE

