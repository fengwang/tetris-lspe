#ifndef YTLQNXXHFCNVDBATNEIXJJVFYORQJJCVLHYHSNPSWJGCSDKKLQPCNPECHDQGEKQMEDOAHVICS
#define YTLQNXXHFCNVDBATNEIXJJVFYORQJJCVLHYHSNPSWJGCSDKKLQPCNPECHDQGEKQMEDOAHVICS

#include <f/matrix/matrix.hpp>

#include <cassert>
#include <numeric>

namespace f
{
    template< typename Refinement >
    struct experimental_intensity_matrix
    {
        typedef Refinement          zen_type;
        matrix<double>              the_experimental_matrix;

        void load_experimental_intensity_matrix()
        {
            auto& zen = static_cast<zen_type&>(*this);
            the_experimental_matrix.load( zen.the_configuration.intensity_path );

            assert( the_experimental_matrix.row() == zen.the_beam_matrix.row() );
            assert( the_experimental_matrix.col() == zen.the_tilt_matrix.row() );
        }

        void normalize_experimental_intensity_matrix()
        {
            assert( the_experimental_matrix.row() );
            assert( the_experimental_matrix.col() );
            for ( unsigned long c = 0; c != the_experimental_matrix.col(); ++c )
            {
                std::for_each( the_experimental_matrix.col_begin(c), the_experimental_matrix.col_end(c), [](double& x){ x = std::max( x, 0.0 ); } );
                double const sum = std::accumulate( the_experimental_matrix.col_begin(c), the_experimental_matrix.col_end(c), 0.0 );
                std::for_each( the_experimental_matrix.col_begin(c), the_experimental_matrix.col_end(c), [sum](double& x){ x /= sum; } );
            }
        }

    };
}//namespace f

#endif//YTLQNXXHFCNVDBATNEIXJJVFYORQJJCVLHYHSNPSWJGCSDKKLQPCNPECHDQGEKQMEDOAHVICS

