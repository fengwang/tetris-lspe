#ifndef JEUNAVVDCHHDKDQIBMEVLYGHRSNFQJVKMAMXJKVJINGVDNSPLFONXCUYRUBBLKGRXSWNACOLA
#define JEUNAVVDCHHDKDQIBMEVLYGHRSNFQJVKMAMXJKVJINGVDNSPLFONXCUYRUBBLKGRXSWNACOLA

#include <f/matrix/matrix.hpp>
#include <f/lexical_cast/lexical_cast.hpp>

#include <string>
#include <cassert>

namespace f
{
    inline void generate_classic_pattern(   std::string const& ar_path,
                                            std::string const& diag_path,
                                            std::string const& intensity_path,
                                            std::string const& pattern_folder )
    {
        matrix<unsigned long> ar;
        ar.load( ar_path );

        matrix<double> diag;
        diag.load( diag_path );

        matrix<double> intensity;
        intensity.load( intensity_path );
        
        assert( diag.row() == intensity.row() );
        assert( diag.col() == intensity.col() );
        assert( ar.row() == diag.row() );

        std::string pattern_folder_{ pattern_folder };
        if ( *pattern_folder.rbegin() != '/' )
            pattern_folder_ += std::string{"/"};

        matrix<double> diag_{ diag.row(), 1 };
        matrix<double> intensity_{ intensity.row(), 1 };

        for ( unsigned long c = 0; c != diag.col(); ++c )
        {
            std::string num = lexical_cast<std::string>( c );
            std::copy( diag.col_begin(c), diag.col_end(c), diag_.begin() ); 
            std::copy( intensity.col_begin(c), intensity.col_end(c), intensity_.begin() );
            
            ar.save_as( pattern_folder_ + std::string{"Ar_"} + num + std::string{".txt"} );
            diag_.save_as( pattern_folder_ + std::string{"Diag_"} + num + std::string{".txt"} );

            std::for_each( intensity_.begin(), intensity_.end(), [](auto& x){ x = std::max(x, 0.0); } );
            auto sum = std::accumulate( intensity_.begin(), intensity_.end(), double{0} );
            std::for_each( intensity_.begin(), intensity_.end(), [sum](auto& x){ x /= sum; } );

            intensity_.save_as( pattern_folder_ + std::string{"Intensities_"} + num + std::string{".txt"} );
        }
    }

}//namespace f

#endif//JEUNAVVDCHHDKDQIBMEVLYGHRSNFQJVKMAMXJKVJINGVDNSPLFONXCUYRUBBLKGRXSWNACOLA

