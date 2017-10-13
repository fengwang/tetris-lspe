#ifndef KNVCMSPLNNNCSFJFJSJBRKIEAAGCYOPKSYCWOECTWLFADGKDWVWUGQJATVYODLISPCWRBDTGU
#define KNVCMSPLNNNCSFJFJSJBRKIEAAGCYOPKSYCWOECTWLFADGKDWVWUGQJATVYODLISPCWRBDTGU

#include <f/matrix/matrix.hpp>

#include <cassert>
#include <complex>
#include <string>
#include <cmath>
#include <cassert>

namespace f
{

    inline void make_real_potential( matrix<std::complex<double> > const& ug, matrix<int> const& beam, std::string const& path )
    {
        assert( ug.row() );
        assert( 1 == ug.col() );
        assert( 3 == beam.col() );
        assert( beam.row() >= ug.row() );

        int const max_offset = *std::max_element( beam.begin(), beam.end(), []( auto const& x, auto const& y ) { return std::abs(x) < std::abs(y); } );
        int const most_sig_bit = static_cast<int>( std::log2( static_cast<double>(max_offset) ) + 1 ); 
        int const plot_size = 1 << most_sig_bit;
        int const plot_mid = plot_size >> 1;

        matrix<std::complex<double> > plot_map;
        plot_map.resize( plot_size, plot_size );
        std::fill( plot_map.begin(), plot_map.end(), std::complex<double>{ 0.0, 0.0 } );

        for ( unsigned long index = 0; index != ug.row(); ++index )
        {
            int const gx = beam[index][0];
            int const gy = beam[index][1];
            plot_map[plot_mid+gy+gy][plot_mid+gx+gx] = ug[index][0];
        }

        auto const& complex_map = ifft( ifftshift(plot_map) * static_cast<double>(plot_map.size() ) );
        auto const& real_map = real( complex_map );
        real_map.save_as_bmp( path );
    }

    inline void make_imag_potential( matrix<std::complex<double> > const& ug, matrix<int> const& beam, std::string const& path )
    {
        assert( ug.row() );
        assert( 1 == ug.col() );
        assert( 3 == beam.col() );
        assert( beam.row() >= ug.row() );

        int const max_offset = *std::max_element( beam.begin(), beam.end(), []( auto const& x, auto const& y ) { return std::abs(x) < std::abs(y); } );
        int const most_sig_bit = static_cast<int>( std::log2( static_cast<double>(max_offset) ) + 1 ); 
        int const plot_size = 1 << most_sig_bit;
        int const plot_mid = plot_size >> 1;

        matrix<std::complex<double> > plot_map;
        plot_map.resize( plot_size, plot_size );
        std::fill( plot_map.begin(), plot_map.end(), std::complex<double>{ 0.0, 0.0 } );

        for ( unsigned long index = 0; index != ug.row(); ++index )
        {
            int const gx = beam[index][0];
            int const gy = beam[index][1];
            plot_map[plot_mid+gy+gy][plot_mid+gx+gx] = ug[index][0];
        }

        auto const& complex_map = ifft( ifftshift(plot_map) * static_cast<double>(plot_map.size() ) );
        auto const& imag_map = imag( complex_map );
        imag_map.save_as_bmp( path );
    }

    inline void make_potential( matrix<std::complex<double> > const& ug, matrix<int> const& beam, std::string const& path )
    {
        assert( ug.row() );
        assert( 1 == ug.col() );
        assert( 3 == beam.col() );
        assert( beam.row() >= ug.row() );

        //int const max_offset = *std::max_element( beam.begin(), beam.end(), []( auto const& x, auto const& y ) { return std::abs(x) < std::abs(y); } );
        int const max_offset_0 = *std::max_element( beam.col_begin(0), beam.col_begin(0)+ug.row(), []( auto const& x, auto const& y ) { return std::abs(x) < std::abs(y); } );
        int const max_offset_1 = *std::max_element( beam.col_begin(1), beam.col_begin(1)+ug.row(), []( auto const& x, auto const& y ) { return std::abs(x) < std::abs(y); } );
        int const max_offset = std::max( max_offset_0, max_offset_1 );
        int const most_sig_bit = static_cast<int>( std::log2( static_cast<double>(max_offset) ) + 1 ) + 1; 
        int const plot_size = std::max( 64, 1 << most_sig_bit );
        int const plot_mid = plot_size >> 1;

        matrix<std::complex<double> > plot_map;
        plot_map.resize( plot_size, plot_size );
        std::fill( plot_map.begin(), plot_map.end(), std::complex<double>{ 0.0, 0.0 } );

        for ( unsigned long index = 0; index != ug.row(); ++index )
        {
            int const gx = beam[index][0];
            int const gy = beam[index][1];
            plot_map[plot_mid+gy+gy][plot_mid+gx+gx] = ug[index][0];
        }

        auto const& complex_map = ifft( ifftshift(plot_map) * static_cast<double>(plot_map.size() ) );
        auto const& imag_map = imag( complex_map );
        imag_map.save_as_bmp( std::string{"imag_"} + path );

        auto const& real_map = real( complex_map );
        real_map.save_as_bmp( std::string{"real_"} + path );
    }

}//namespace f

#endif//KNVCMSPLNNNCSFJFJSJBRKIEAAGCYOPKSYCWOECTWLFADGKDWVWUGQJATVYODLISPCWRBDTGU
