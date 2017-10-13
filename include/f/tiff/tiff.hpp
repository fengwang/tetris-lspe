#ifndef MCYILLNFLVFRERLGXAYMFABAOSDFOGDQGMRVKWWOMCEIRJGQGPQJGRDUULQPAQHWICNUACAHS
#define MCYILLNFLVFRERLGXAYMFABAOSDFOGDQGMRVKWWOMCEIRJGQGPQJGRDUULQPAQHWICNUACAHS

#include <f/matrix/matrix.hpp>

#include <cstdint>
#include <vector>
#include <string>
#include <cassert>
#include <cstddef>

#define cimg_display 0
#define cimg_use_tiff

#include "../../3rdparty/CImg.h"

//bool tiff_load_from( char const*, std::vector<f::matrix<std::uint16_t>>& );
//bool tiff_save_as( char const*, std::vector<f::matrix<std::uint16_t>> const& );

namespace f
{
    struct tiff
    {
        //typedef std::uint16_t               value_type;
        typedef std::size_t                 size_type;
        typedef float                       value_type;
        typedef matrix<value_type>          matrix_type;
        typedef std::vector<matrix_type>    data_type;

        data_type                           tiff_data;

        void load( std::string const& path_ )
        {
            load( path_.c_str() );
        }

        void load( char const* path_ )
        {
            using namespace cimg_library;

            CImgList<value_type> img_list;
            img_list.load_tiff( path_ );

            for ( auto const& img : img_list )
            {
                matrix_type mat{ static_cast<size_type>(img.width()), static_cast<size_type>(img.height()) };

                if ( 1 == img.spectrum() )
                {
                    std::copy( img.begin(), img.end(), mat.begin() );
                    tiff_data.emplace_back( mat.transpose() );
                    continue;
                }

                auto const& new_img = (img.get_RGBtoYCbCr()).get_channel(0);
                std::copy( new_img.begin(), new_img.end(), mat.begin() );
                tiff_data.emplace_back( mat.transpose() );
            }
        }

/*
        void save_as( std::string const& path_ ) const
        {
            save( path_.c_str() );
        }

        void save_as( char const* path_ ) const
        {

            if ( !tiff_save_as( path_, tiff_data ) )
                assert( !"Failed to save TIFF images!" );
        }
*/
    };

}//namespace f

#endif//MCYILLNFLVFRERLGXAYMFABAOSDFOGDQGMRVKWWOMCEIRJGQGPQJGRDUULQPAQHWICNUACAHS

