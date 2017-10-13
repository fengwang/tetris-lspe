#ifndef NWFWDFJOUCKWERTUAQTWOPTIEQPXTXGFGSHLAASNLJELUITBRHFADDJIPLOQFJUHPASRWKBYX
#define NWFWDFJOUCKWERTUAQTWOPTIEQPXTXGFGSHLAASNLJELUITBRHFADDJIPLOQFJUHPASRWKBYX

#include <f/pattern_refinement/c1_pattern_exclude_central.hpp>
#include <f/pattern_refinement/pattern_refinement_configuration.hpp>

#include <string>

namespace f
{

    struct c1_refinement
    {
        pattern_refinement_configuration    prc;
        c1_pattern                          cp;

        c1_refinement( pattern_refinement_configuration const& prc_ ) : prc( prc_ ), cp( prc_ ) 
        {
        }

        unsigned long unknowns()
        {
            return cp.simulated_intensity.row() + 18;
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

                (*this).prc.high_tension = x[8];
                (*this).prc.unit_cell[0] = x[9];   (*this).prc.unit_cell[1] = x[10]; (*this).prc.unit_cell[2] = x[11]; 
                (*this).prc.unit_cell[3] = x[12];  (*this).prc.unit_cell[4] = x[13]; (*this).prc.unit_cell[5] = x[14]; 
                (*this).prc.unit_cell[6] = x[15];  (*this).prc.unit_cell[7] = x[16]; (*this).prc.unit_cell[8] = x[17]; 

                double* x_ = x + 18;

                return (*this).cp.make_square_difference( x_, (*this).prc );
            };
        }

        void dump()
        {
            cp.dump();
        }
    
    };

}//namespace f

#endif//NWFWDFJOUCKWERTUAQTWOPTIEQPXTXGFGSHLAASNLJELUITBRHFADDJIPLOQFJUHPASRWKBYX

