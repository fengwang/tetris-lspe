#ifndef ONGVGWDOARWHOOVUBAEBSQLSIRBBBMJLBCYEGMRHHYRBKWMRKLVVTTHCLDVBQPVSKWONDXHJP
#define ONGVGWDOARWHOOVUBAEBSQLSIRBBBMJLBCYEGMRHHYRBKWMRKLVVTTHCLDVBQPVSKWONDXHJP

#include <f/pattern_refinement/pattern_refinement_configuration.hpp>

#include <f/matrix/matrix.hpp>
#include <f/beam/beam.hpp>
#include <f/wave_length/wave_length.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <string>

namespace f
{

    struct cpu_space_pattern
    {
        typedef double              value_type;
        typedef matrix<value_type>  matrix_type;

        matrix_type                 intensity;
        matrix_type                 beam;  // [index, kx, ky, kz]
        matrix_type                 tilt;  //

        matrix<int>                 ar;    //
        matrix_type                 new_beam; // [index, kx, ky, kz] 
        matrix_type                 kt;    // [kt_x, kt_y ] -- for all tilts
        matrix_type                 kt_orig;    // [kt_x, kt_y ] -- for all tilts without rotation
        matrix_type                 gvec;  // [g_x, g_y] -- for all beams
        matrix_type                 diag;

        virtual ~cpu_space_pattern() {}

        void normalize( matrix_type& mat )
        {
#if 0
            value_type const total = std::accumulate( mat.begin(), mat.end(), value_type{} );
            value_type const divider = total / static_cast<value_type>( mat.col() );

            mat /= divider;
#else
            std::for_each( mat.begin(), mat.end(), []( auto& x){ x = std::max( 0.0, x ); } );
            for ( unsigned long c = 0; c != mat.col(); ++c )
            {
                value_type const divider = std::accumulate( mat.col_begin(c), mat.col_end(c), value_type{} );
                std::for_each( mat.col_begin(c), mat.col_end(c), [divider]( value_type& x ){ x /= divider; } );
            }
#endif
        }

        cpu_space_pattern( pattern_refinement_configuration const& config )
        {
            intensity.load( config.intensity_path );
            normalize( intensity );
            beam.load( config.beam_path );
            tilt.load( config.tilt_path );

            make_full_beam_list( beam, ar, new_beam );            

            initialize( config );
        }

        void initialize( pattern_refinement_configuration const& config )
        {
            refine_tilt( config.high_tension, config.estimated_tilt_rotation, config.kx_factor, config.ky_factor );
            construct_gvec( config.zone, config.gx, config.unit_cell );
            construct_diag();
        }


        void construct_gvec( std::array<value_type, 3> const& gz, std::array<value_type, 3> const& gx, std::array<value_type, 9> const& unit_cell )
        {
            matrix_type Mm{ 3, 3 };
            std::copy( unit_cell.begin(), unit_cell.end(), Mm.begin() );

            value_type z_norm = std::sqrt( std::inner_product( gz.begin(), gz.end(), gz.begin(), value_type{} ) );
            value_type x_norm = std::sqrt( std::inner_product( gx.begin(), gx.end(), gx.begin(), value_type{} ) );

            auto kz = gz;
            auto kx = gx;
            auto ky = gz;
            std::for_each( kz.begin(), kz.end(), [z_norm]( value_type& z ){ z /= z_norm; } );
            std::for_each( kx.begin(), kx.end(), [x_norm]( value_type& x ){ x /= x_norm; } );
            ky[0] = kz[1] * kx[2] - kz[2] * kx[1];
            ky[1] = kz[2] * kx[0] - kz[0] * kx[2];
            ky[2] = kz[0] * kx[1] - kz[1] * kx[0];
            matrix_type gxy{ 3, 2 };
            std::copy( kx.begin(), kx.end(), gxy.col_begin(0) );
            std::copy( ky.begin(), ky.end(), gxy.col_begin(1) );

            matrix_type exact_beam{ new_beam.row(), 3 };
            std::copy( new_beam.col_begin(1), new_beam.col_end(1), exact_beam.col_begin(0) ); 
            std::copy( new_beam.col_begin(2), new_beam.col_end(2), exact_beam.col_begin(1) ); 
            std::copy( new_beam.col_begin(3), new_beam.col_end(3), exact_beam.col_begin(2) ); 

            gvec = exact_beam / Mm * gxy;
        }

        // Tilt --> shrink
        void refine_tilt( value_type high_tension, value_type estimated_tilt_rotation, std::array<value_type,2> const& kx_factor, std::array<value_type,2> const& ky_factor )
        {
            value_type const lambda = wave_length( high_tension );
            // [index  t_ideal_mrad(2)  tilt_DACs(2)  t_actual_mrad(2) t_actual_1/nm(2)  intensity]
            kt.resize( tilt.row(), 2 );

            auto converter = []( value_type lambda, value_type scaler, value_type offset )
            {
                return [=]( value_type x )
                {
                    return scaler * std::sin( x * value_type{0.001} ) / lambda + offset;
                };
            };

            std::transform( tilt.col_begin( 5 ), tilt.col_end( 5 ), kt.col_begin( 0 ), converter( lambda, kx_factor[0], kx_factor[1] ) );
            std::transform( tilt.col_begin( 6 ), tilt.col_end( 6 ), kt.col_begin( 1 ), converter( lambda, ky_factor[0], ky_factor[1] ) );

            kt_orig = kt;
            std::copy( tilt.col_begin( 5 ), tilt.col_end( 5 ), kt_orig.col_begin( 0 ) );
            std::copy( tilt.col_begin( 6 ), tilt.col_end( 6 ), kt_orig.col_begin( 1 ) );
            kt_orig *= 0.001;

            value_type const theta = estimated_tilt_rotation * 3.1415926535897932384626433;
            matrix_type rot{ 2, 2 };
            rot[0][0] = std::cos(theta); rot[0][1] = -std::sin(theta);
            rot[1][0] = -rot[0][1];      rot[1][1] = rot[0][0];

            kt *= rot;
        }

        void construct_diag()
        {
            diag.resize( intensity.row(), intensity.col() );

            for ( unsigned long r = 0; r != diag.row(); ++r )
            {
                value_type const gx = gvec[r][0];
                value_type const gy = gvec[r][1];
                for ( unsigned long c = 0; c != diag.col(); ++c )
                {
                    value_type const kx = kt[c][0];
                    value_type const ky = kt[c][1];

                    diag[r][c] = - 2.0* (kx*gx + ky*gy) - gx*gx - gy*gy;
                }
            }
        }

        void dump( std::string const& prefix= std::string{"cpu_space_pattern_"}, std::string const& suffix = std::string{".txt"} )
        {
            intensity.save_as( prefix + std::string{"intensity"} + suffix );
            beam.save_as( prefix + std::string{"beam"} + suffix );
            tilt.save_as( prefix + std::string{"tilt"} + suffix );
            ar.save_as( prefix + std::string{"ar"} + suffix );
            new_beam.save_as( prefix + std::string{"new_beam"} + suffix );
            kt.save_as( prefix + std::string{"kt_rotated"} + suffix );
            kt_orig.save_as( prefix + std::string{"kt"} + suffix );
            gvec.save_as( prefix + std::string{"gvec"} + suffix );
            diag.save_as( prefix + std::string{"diag"} + suffix );
        }
    };

}//namespace f

#endif//ONGVGWDOARWHOOVUBAEBSQLSIRBBBMJLBCYEGMRHHYRBKWMRKLVVTTHCLDVBQPVSKWONDXHJP

