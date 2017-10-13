#ifndef SKAJXPMSWIOBFPIBUAFBCYTRFBJWJBYQCGDKRGTNOLFTEXVSIJCKIFBCITGMWGYJSCVLKJEBQ
#define SKAJXPMSWIOBFPIBUAFBCYTRFBJWJBYQCGDKRGTNOLFTEXVSIJCKIFBCITGMWGYJSCVLKJEBQ

#include <f/pattern_refinement/pattern_refinement_configuration.hpp>
#include <f/pattern_refinement/cpu_space_pattern.hpp>

#include <f/wave_length/wave_length.hpp>

#include <f/matrix/matrix.hpp>
#include <vector>
#include <cmath>
#include <complex>
#include <numeric>
#include <string>

namespace f
{

    struct c1_pattern : cpu_space_pattern
    {
        typedef double                      value_type;
        typedef matrix<value_type>          matrix_type;

        matrix_type                         simulated_intensity;
        matrix_type                         differential_intensity;

        double                              total_squared_intensity;
        unsigned long                       offset;
        
        c1_pattern( pattern_refinement_configuration const& config ) : cpu_space_pattern{ config }
        {

            simulated_intensity.resize( ((*this).intensity).row(), ((*this).intensity).col() );
            std::fill( simulated_intensity.row_begin(0), simulated_intensity.row_end(0), 1.0 );
            differential_intensity.resize( ((*this).intensity).row(), ((*this).intensity).col() );

            offset = simulated_intensity.col();
            total_squared_intensity = std::inner_product( (*this).intensity.begin()+offset, (*this).intensity.end(), (*this).intensity.begin()+offset, 0.0 );
        }

        value_type calculate_c1_intensity( value_type pi_lambda_t, value_type diag, value_type ug )
        {
            std::complex<value_type> const T{ 0.0, pi_lambda_t};
            if ( std::abs(diag) < 1.0e-10 ) 
                return std::norm( T * std::exp( T * diag ) * ug );
            return std::norm( ( std::exp( T * diag ) - 1.0 ) / diag ) * ug * ug;
        }

        value_type make_square_difference( value_type* abs_ug, pattern_refinement_configuration const& config )
        {
            (*this).initialize( config );

            value_type const pi_lambda_t = wave_length( config.high_tension ) * 3.14159265358979323846 * config.estimated_thickness;

            for ( unsigned long r = 1; r != simulated_intensity.row(); ++r )
                for ( unsigned long c = 0; c != simulated_intensity.col(); ++c )
                   simulated_intensity[r][c] = calculate_c1_intensity( pi_lambda_t, (*this).diag[r][c], abs_ug[r] ); 

            std::fill( simulated_intensity.row_begin(0), simulated_intensity.row_end(0), 0.0 );

            differential_intensity = (*this).intensity - (config.intensity_factor)[0] * simulated_intensity - (config.intensity_factor)[1];
            std::fill( differential_intensity.row_begin(0), differential_intensity.row_end(0), 0.0 );

            double const square_res = std::inner_product( differential_intensity.begin()+offset, differential_intensity.end(), differential_intensity.begin()+offset, 0.0 );

            return square_res / total_squared_intensity;
        }

        void dump( std::string const& prefix = std::string{"c1_pattern_"}, std::string const& suffix = std::string{".txt"} )
        {
            cpu_space_pattern::dump( prefix, suffix );
            simulated_intensity.save_as( prefix + std::string{"simulated_intensity"} + suffix );
            differential_intensity.save_as( prefix + std::string{"differential_intensity"} + suffix );
        }
    };

}//namespace f

#endif//SKAJXPMSWIOBFPIBUAFBCYTRFBJWJBYQCGDKRGTNOLFTEXVSIJCKIFBCITGMWGYJSCVLKJEBQ

