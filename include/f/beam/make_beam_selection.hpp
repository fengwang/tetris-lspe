#ifndef VVAXUWQDWJPVUUTQCOKANIDWXWRINGHQWJOUXCWSVADYWCEJJCASXNBEYQJYXOFDNPRVWRLRF
#define VVAXUWQDWJPVUUTQCOKANIDWXWRINGHQWJOUXCWSVADYWCEJJCASXNBEYQJYXOFDNPRVWRLRF

#include <f/matrix/matrix.hpp>

#include "make_ar.hpp"
#include "beam_list.hpp"

#include <string>
#include <cassert>
#include <vector>

namespace f
{

    inline void make_beam_selection( std::string const& intensity_path,
                                     std::string const& diag_path,
                                     std::string const& beam_path,
                                     std::string const& output_folder,
                                     double const alpha = 1.0e-3 ) noexcept
    {
        matrix<double> intensity;
        intensity.load( intensity_path );

        matrix<double> diag;
        diag.load( diag_path );

        assert( intensity.row() == diag.row() );
        assert( intensity.col() == diag.col() );

        auto bl = make_beam_list( beam_path );

        //auto ar = make_ar( bl, _1, 1_ );

        std::vector<int> var;
        std::vector<double> vint;
        std::vector<double> vdia;

        for ( unsigned long tilt_index = 0; tilt_index != intensity.col(); ++tilt_index )
        {
            var.clear();
            vint.clear();
            vdia.clear();

            for ( unsigned long row_index = 0; row_index != intensity.row(); ++row_index )
            {
                if ( intensity[row_index][tilt_index] >= alpha )
                {
                    var.push_back( row_index );
                    vint.push_back( intensity[row_index][tilt_index] );
                    vdia.push_back( diag[row_index][tilt_index] );
                }
            }

            auto ar = make_ar( bl, var.begin(), var.end() );

            matrix<double> int_( vint.size(), 1 );
            std::copy( vint.begin(), vint.end(), int_.begin() );

            matrix<double> dia_( vdia.size(), 1 );
            std::copy( vdia.begin(), vdia.end(), dia_.begin() );

            std::string num = std::to_string( tilt_index );

            ar.save_as( output_folder + std::string{"/Ar_"} + num + std::string{".txt"} );
            int_.save_as( output_folder + std::string{"/Intensities_"} + num + std::string{".txt"} );
            dia_.save_as( output_folder + std::string{"/Diag_"} + num + std::string{".txt"} );

        }
    }

    inline void make_beam_selection( std::string const& intensity_path,
                                     std::string const& diag_path,
                                     std::string const& beam_path,
                                     std::string const& output_folder,
                                     unsigned long const selected ) noexcept
    {
        matrix<double> intensity;
        intensity.load( intensity_path );

        matrix<double> diag;
        diag.load( diag_path );

        assert( intensity.row() == diag.row() );
        assert( intensity.col() == diag.col() );

        auto bl = make_beam_list( beam_path );

        //auto ar = make_ar( bl, _1, 1_ );

        std::vector<int> var;
        std::vector<double> vint;
        std::vector<double> vdia;
        std::vector<double> all( intensity.row() );
        unsigned long const sel_index = intensity.row() - selected;

        for ( unsigned long tilt_index = 0; tilt_index != intensity.col(); ++tilt_index )
        {
            var.clear();
            vint.clear();
            vdia.clear();

            all.resize( intensity.row() );
            std::copy( intensity.col_begin(tilt_index), intensity.col_end(tilt_index), all.begin() );
            std::sort( all.begin(), all.end() );//ok as small length
            double const alpha = all[sel_index];

            for ( unsigned long row_index = 0; row_index != intensity.row(); ++row_index )
            {
                if ( intensity[row_index][tilt_index] >= alpha )
                {
                    var.push_back( row_index );
                    vint.push_back( intensity[row_index][tilt_index] );
                    vdia.push_back( diag[row_index][tilt_index] );
                }
            }

            auto ar = make_ar( bl, var.begin(), var.end() );

            matrix<double> int_( vint.size(), 1 );
            std::copy( vint.begin(), vint.end(), int_.begin() );
            int_ /= std::accumulate( int_.begin(), int_.end(), double{0} );

            matrix<double> dia_( vdia.size(), 1 );
            std::copy( vdia.begin(), vdia.end(), dia_.begin() );

            std::string num = std::to_string( tilt_index );

            ar.save_as( output_folder + std::string{"/Ar_"} + num + std::string{".txt"} );
            int_.save_as( output_folder + std::string{"/Intensities_"} + num + std::string{".txt"} );
            dia_.save_as( output_folder + std::string{"/Diag_"} + num + std::string{".txt"} );

        }
    }


}//namespace f

#endif//VVAXUWQDWJPVUUTQCOKANIDWXWRINGHQWJOUXCWSVADYWCEJJCASXNBEYQJYXOFDNPRVWRLRF

