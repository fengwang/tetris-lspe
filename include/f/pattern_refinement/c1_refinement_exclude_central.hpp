#ifndef XIVKGUIXCPSHRJYHRDWSPCKAOTTRBFQKHDLIBXGJUDQPXOTEYUFTLFFMBSYXTJRRTOQBKJMME
#define XIVKGUIXCPSHRJYHRDWSPCKAOTTRBFQKHDLIBXGJUDQPXOTEYUFTLFFMBSYXTJRRTOQBKJMME

#include <f/pattern_refinement/c1_pattern_exclude_central.hpp>
#include <f/pattern_refinement/pattern_refinement_configuration.hpp>

namespace f
{

    struct c1_refinement
    {
        pattern_refinement_configuration    prc;
        c1_pattern                          cp;

        c1_refinement( pattern_refinement_configuration const& prc_ ) : prc( prc_ ), cp( prc_ ) {}

        unsigned long unknowns()
        {
            return cp.simulated_intensity.row() + 8;
        }

        auto make_merit_function( )
        {
            return [&]( double* x )
            {
                (*this).prc.estimated_thickness = x[0];
                (*this).prc.estimated_tilt_rotation = x[1];
                (*this).prc.kx_factor = {{ x[2], x[3] }};
                (*this).prc.ky_factor = {{ x[4], x[5] }};
                (*this).prc.intensity_factor = {{ x[6], x[7] }};

                double* x_ = x + 8;

                return (*this).cp.make_square_difference( x_, (*this).prc );
            };
        }
    
    };

}//namespace f

#endif//XIVKGUIXCPSHRJYHRDWSPCKAOTTRBFQKHDLIBXGJUDQPXOTEYUFTLFFMBSYXTJRRTOQBKJMME

