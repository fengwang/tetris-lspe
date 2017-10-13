#ifndef RSWTEIYIJSTNXKXDDTIMITRPSGENTGSKBPFXYVISWGILPTYBHKJNYSBGQGOGICYQUMEMUDEBM
#define RSWTEIYIJSTNXKXDDTIMITRPSGENTGSKBPFXYVISWGILPTYBHKJNYSBGQGOGICYQUMEMUDEBM

#include <f/matrix/matrix.hpp>
#include <f/algorithm/for_each.hpp>
#include <f/wave_length/wave_length.hpp>

#include <algorithm>
#include <string>
#include <cassert>
#include <complex>

namespace f
{
    inline void simulate_intensity_c( matrix<unsigned long> const& ar, matrix<std::complex<double> > const& ug, matrix<double> const& diag, double const thickness, std::string const& output )
    {
        assert( ug.col() == 1 );
        unsigned long const max_ug = ug.row();

        matrix<std::complex<double> > A( ar.row(), ar.col() );
        for_each( A.begin(), A.end(), ar.begin(), [&]( std::complex<double>& c, unsigned long idx )
                { 
                    if ( idx >= ug.row() ) 
                        std::cerr << "Error loading idx " << idx << "\t" << ug.row() << "\n";
                    assert( idx < max_ug ); 
                    c = ug[idx][0]; 
                } 
                );

        matrix<double> I( diag.row(), diag.col() );

        for ( unsigned long index = 0; index != diag.col(); ++index )
        {
            std::copy( diag.col_begin(index), diag.col_end(index), A.diag_begin() );
            matrix<std::complex<double> > const& S = expm( A * std::complex<double>(0.0, thickness) );
            for_each( S.col_begin(0), S.col_end(0), I.col_begin(index), []( std::complex<double> const& c, double& x ){ x = std::norm(c); } );
        }

        I.save_as( output );
    }

    inline void simulate_intensity_1( matrix<unsigned long> const& ar, matrix<double> const& ug, matrix<double> const& diag, double const thickness, std::string const& output, unsigned long index )
    {
        if ( (index == 0) || (index > ug.row()) ) 
            index = ug.row();

        matrix<double> dug( (ug.col() >> 1) - 1, 2 );
        std::copy( ug.row_begin(index-1)+1, ug.row_end(index-1)-1, dug.begin() );

        matrix<std::complex<double> > cug( dug.row(), 1 );
        for ( unsigned long index = 0; index != cug.row(); ++index ) cug[index][0] = std::complex<double>( dug[index][0], dug[index][1] );

        simulate_intensity_c( ar, cug, diag, thickness, output );
    }

    //with column ug
    inline void simulate_intensity_2( matrix<unsigned long> const& ar, matrix<double> const& ug, matrix<double> const& diag, double const thickness, std::string const& output )
    {
        matrix<std::complex<double> > cug( ug.row(), 1 );
        for ( unsigned long index = 0; index != cug.row(); ++index ) cug[index][0] = std::complex<double>( ug[index][0], ug[index][1] );

        simulate_intensity_c( ar, cug, diag, thickness, output );
    }

    inline void simulate_intensity( matrix<unsigned long> const& ar, matrix<double> const& ug, matrix<double> const& diag, double const thickness, std::string const& output, unsigned long index = 0 )
    {
        assert( ar.row() == ar.col() );
        assert( diag.row() == ar.row() );
        assert( diag.col() );
        assert( ug.size() );
        assert( thickness > 0.0 );
        assert( output.size() );

        if ( 2 == ug.col() )
            simulate_intensity_2( ar, ug, diag, thickness, output );
        else
            simulate_intensity_1( ar, ug, diag, thickness, output, index );
    }

    inline void simulate_intensity( std::string const& ar, std::string const& ug, std::string const& diag, double const thickness, std::string const& output, unsigned long index = 0 )
    {
        matrix<unsigned long> mar;
        mar.load( ar );
        matrix<double> mug; 
        mug.load( ug );
        matrix<double> mdiag;
        mdiag.load( diag );
        simulate_intensity( mar, mug, mdiag, thickness, output, index );
    }

    inline void simulate_intensity( char const* const ar, char const* const ug, char const* const diag, double const thickness, char const* const output, unsigned long index = 0 )
    {
        simulate_intensity( std::string{ar}, std::string{ug}, std::string{diag}, thickness, std::string{output}, index );
    }


    inline void simulate_intensity_with_tilt( char const* const beam, char const* const ar, char const* const ug, char const* const tilt, double const thickness, char const* const output, double radius, unsigned long index = 0, double const ev = 120.0 )
    {
        matrix<double> mbeam;
        mbeam.load( beam );
        assert( mbeam.row() );
        assert( mbeam.col() == 4 );

        matrix<unsigned long> mar;
        mar.load( ar );
        matrix<double> mug; 
        mug.load( ug );

        matrix<double> mtilt;
        mtilt.load( tilt );
        assert( mtilt.row() );
        assert( mtilt.col() == 2 );

        double const lambda = wave_length( ev );
        matrix<double> mdiag( mar.row(), mtilt.row() );;

        matrix<double> beams{ mbeam.row(), 3 };
        std::copy( mbeam.col_begin(1), mbeam.col_end(1), beams.col_begin(0) );
        std::copy( mbeam.col_begin(2), mbeam.col_end(2), beams.col_begin(1) );
        std::copy( mbeam.col_begin(3), mbeam.col_end(3), beams.col_begin(2) );
        matrix<double> cell{3, 3};
        cell[0][0] = 3.905; cell[0][1] = 0.0; cell[0][2] = 0.0;
        cell[1][0] = 0.0; cell[1][1] = 3.905; cell[1][2] = 0.0;
        cell[2][0] = 0.0; cell[2][1] = 0.0; cell[2][2] = 3.905;
        beams /= cell;

        for ( unsigned long c = 0; c != mdiag.col(); ++c )
        {
            double const t_x = radius * mtilt[c][0] / 1000.0;
            double const t_y = radius * mtilt[c][1] / 1000.0;
            double const kt_x = std::sin(t_x)/lambda;
            double const kt_y = std::sin(t_y)/lambda;
            for ( unsigned long r = 0; r != mdiag.row(); ++r )
            {
                double const gx = beams[r][0];
                double const gy = beams[r][1];
                mdiag[r][c] = -gx*gx -gy*gy - 2.0*gx*kt_x - 2.0*gy*kt_y;
            }
        }

        simulate_intensity( mar, mug, mdiag, thickness, output, index );
    }

}//namespace f

#endif//RSWTEIYIJSTNXKXDDTIMITRPSGENTGSKBPFXYVISWGILPTYBHKJNYSBGQGOGICYQUMEMUDEBM

