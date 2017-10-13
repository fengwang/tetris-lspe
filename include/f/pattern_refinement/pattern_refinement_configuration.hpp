#ifndef UKVKIHMMSLUYBXLBXEVAMKNTCTOBVAJPAVMWTTYQBPXQMNNNCILBEVTFKAXKMRNCJFBJLXNJC
#define UKVKIHMMSLUYBXLBXEVAMKNTCTOBVAJPAVMWTTYQBPXQMNNNCILBEVTFKAXKMRNCJFBJLXNJC

#ifdef CEREAL_XML_STRING_VALUE
#undef CEREAL_XML_STRING_VALUE
#endif

#define CEREAL_XML_STRING_VALUE "pattern_refinement_configuration"

#include <cereal/cereal.hpp>

#include <f/matrix/matrix.hpp>

#include <vector>
#include <string>
#include <array>
#include <fstream>
#include <cassert>

namespace f
{

    struct pattern_refinement_configuration
    {
        typedef double value_type;

        std::vector<int>            gpu_id;
        value_type                  high_tension;               //Kev
        value_type                  estimated_thickness;        //A
        value_type                  estimated_tilt_rotation;    //radius
        std::array<value_type,2>    kx_factor;                  //[scaler -- 1.0, offset -- 0.0]
        std::array<value_type,2>    ky_factor;                  //[scaler -- 1.0, offset -- 0.0]
        std::array<value_type,2>    intensity_factor;           //[scaler -- 1.0, offset -- 0.0]

        std::array<value_type, 3>   zone;
        std::array<value_type, 3>   gx;
        std::array<value_type, 9>   unit_cell;
        std::string                 intensity_path;
        std::string                 beam_path;
        std::string                 tilt_path;
        std::string                 diag_path;

        std::vector<double>         ug_modulus;

        void save_as( std::string const& path ) const
        {
            std::ofstream ofs( path.c_str() ); 
            assert( ofs );
            cereal::XMLOutputArchive archive(ofs);

            archive( CEREAL_NVP( gpu_id ),
                     CEREAL_NVP( high_tension ),
                     CEREAL_NVP( estimated_thickness ),
                     CEREAL_NVP( estimated_tilt_rotation ),
                     CEREAL_NVP( kx_factor ),
                     CEREAL_NVP( ky_factor ),
                     CEREAL_NVP( intensity_factor ),
                     CEREAL_NVP( zone ),
                     CEREAL_NVP( gx ),
                     CEREAL_NVP( unit_cell ),
                     CEREAL_NVP( intensity_path ),
                     CEREAL_NVP( beam_path ),
                     CEREAL_NVP( tilt_path ),
                     CEREAL_NVP( diag_path ),
                     CEREAL_NVP( ug_modulus ) 
                     );
        }

        void import_from( std::string const& path )
        {
            std::ifstream ifs( path.c_str() );
            assert( ifs );
            cereal::XMLInputArchive archive( ifs );

            archive( CEREAL_NVP( gpu_id ),
                     CEREAL_NVP( high_tension ),
                     CEREAL_NVP( estimated_thickness ),
                     CEREAL_NVP( estimated_tilt_rotation ),
                     CEREAL_NVP( kx_factor ),
                     CEREAL_NVP( ky_factor ),
                     CEREAL_NVP( intensity_factor ),
                     CEREAL_NVP( zone ),
                     CEREAL_NVP( gx ),
                     CEREAL_NVP( unit_cell ),
                     CEREAL_NVP( intensity_path ),
                     CEREAL_NVP( beam_path ),
                     CEREAL_NVP( tilt_path ),
                     CEREAL_NVP( diag_path ),
                     CEREAL_NVP( ug_modulus ) 
                     );
        }

        void load( std::string const& path )
        {
            import_from( path );
        }

    };

}//namespace f

#endif//UKVKIHMMSLUYBXLBXEVAMKNTCTOBVAJPAVMWTTYQBPXQMNNNCILBEVTFKAXKMRNCJFBJLXNJC

