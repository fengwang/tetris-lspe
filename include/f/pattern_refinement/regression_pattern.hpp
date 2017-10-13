#ifndef SQYOFBOVXMWNPDWMJCVXUGOMQOLCYBEJJGRSKPTNUJITJLGXSUISWMTJGSWLQYBJQMEUTLGKI
#define SQYOFBOVXMWNPDWMJCVXUGOMQOLCYBEJJGRSKPTNUJITJLGXSUISWMTJGSWLQYBJQMEUTLGKI

#include <f/matrix/matrix.hpp>
#include <f/algorithm/for_each.hpp>
#include <f/wave_length/wave_length.hpp>

#include <complex>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace f
{
    
    struct regression_pattern
    {
        typedef matrix<double>  matrix_type;
        /* 0 -- 1008*/
        unsigned long index;            // 0-1008
        /* 145 A */
        double thickness;
        double lambda;
        /* 120 */
        double high_tension;
        /* -1.6 */
        double rotation_angle;
        /* 1.0 0.0 0.0 1.0 */
        matrix_type kt_ac_dc;           // 2X2 
                                        // Kx_ac Kx_dc
                                        // Ky_ac Ky_dc
        /* 1.0 0.0 */                                       
        matrix_type I_ac_dc;            // 1X2  -- [ac dc]
        /*
         * 3.905 0.000 0.000
         * 0.000 3.905 0.000
         * 0.000 0.000 3.905
         */
        matrix_type unit_cell;          // 3X3

        matrix<unsigned long> Ar;       // 61X61
        matrix_type ug;                 // MX2
        matrix_type intensity;          // 61X1009
        matrix_type beams;              // RX3
        matrix_type tilt;               // 1009X2
        matrix<std::complex<double> >   A;

        // cache
        matrix<std::complex<double> >   D;
        matrix<std::complex<double> >   S;
        matrix<double>                  I_diff;
        matrix<double>                  rot; // [cos -sin]
                                             // [sin cos ]
        matrix<double>                  kt;  // 1 X 2
        matrix<double>                  gvec;// [R X 3]

        auto make_r_function( unsigned long idx )
        {
            index = idx;
            lambda = wave_length( 120.0 );
            unit_cell.resize( 3, 3 );
            unit_cell[0][0] = 3.905; unit_cell[0][1] = 0.0; unit_cell[0][2] = 0.0;
            unit_cell[1][0] = 0.0; unit_cell[1][1] = 3.905; unit_cell[1][2] = 0.0;
            unit_cell[2][0] = 0.0; unit_cell[2][1] = 0.0; unit_cell[2][2] = 3.905;
            return [this]( double* x ) // 
            {
                (*this).thickness = 145.0 * ( 1.0+x[0] );
                (*this).rotation_angle = -1.6 * ( 1.0+x[1] );

                (*this).kt_ac_dc.resize(2, 2);
                (*this).kt_ac_dc[0][0] = 1.0 + x[2]; (*this).kt_ac_dc[0][1] = x[3];
                (*this).kt_ac_dc[1][0] = 1.0 + x[4]; (*this).kt_ac_dc[1][1] = x[5];

                (*this).I_ac_dc.resize( 1, 2 );
                (*this).I_ac_dc[0][0] = 1.0 + x[6]; (*this).I_ac_dc[0][1] = x[7];
#if 0
                (*this).unit_cell.resize( 3, 3 );
                /*
                (*this).unit_cell[0][0] = x[9]; (*this).unit_cell[0][1] = x[10]; (*this).unit_cell[0][2] = x[11];
                (*this).unit_cell[1][0] = x[12]; (*this).unit_cell[1][1] = x[13]; (*this).unit_cell[1][2] = x[14];
                (*this).unit_cell[2][0] = x[15]; (*this).unit_cell[2][1] = x[16]; (*this).unit_cell[2][2] = x[17];
                */
                /*
                (*this).unit_cell[0][0] = x[9]+3.905; (*this).unit_cell[0][1] = x[10]; (*this).unit_cell[0][2] = x[11];
                (*this).unit_cell[1][0] = x[12]; (*this).unit_cell[1][1] = x[13]+3.905; (*this).unit_cell[1][2] = x[14];
                (*this).unit_cell[2][0] = x[15]; (*this).unit_cell[2][1] = x[16]; (*this).unit_cell[2][2] = x[17]+3.905;
                */
                //(*this).unit_cell[0][0] = (1.0+x[9])*3.905; (*this).unit_cell[0][1] = x[10]; (*this).unit_cell[0][2] = x[11];
                //(*this).unit_cell[1][0] = x[12]; (*this).unit_cell[1][1] = (1.0+x[13])*3.905; (*this).unit_cell[1][2] = x[14];
                //(*this).unit_cell[2][0] = x[15]; (*this).unit_cell[2][1] = x[16]; (*this).unit_cell[2][2] = (1.0+x[17])*3.905;
                (*this).unit_cell[0][0] = (1.0+x[9])*3.905; (*this).unit_cell[0][1] = x[10]; (*this).unit_cell[0][2] = 0.0;
                (*this).unit_cell[1][0] = x[11]; (*this).unit_cell[1][1] = (1.0+x[12])*3.905; (*this).unit_cell[1][2] = 0.0;
                (*this).unit_cell[2][0] = 0.0; (*this).unit_cell[2][1] = 0.0; (*this).unit_cell[2][2] = 3.905;
#endif
                return (*this).make_diff_r();
            };
        }

        double make_diff_square()
        {
            //make gv
            make_gvec();
            //make rot
            make_rot();
            //make kt
            make_kt();
            //make diag
            update_A_diag();

            D = std::complex<double>{ 0.0, thickness*3.14159265358979323846*lambda } * A;
            S = expm( D );
            I_diff.resize( S.row(), 1 );
            for_each( S.col_begin(0), S.col_end(0), intensity.col_begin(index), I_diff.col_begin(0),
                      [this]( std::complex<double> const& c, double i_exp, double& i_diff )
                      { i_diff = std::norm(c) - i_exp*((*this).I_ac_dc)[0][0] - ((*this).I_ac_dc)[0][1]; } 
                    );
            return std::inner_product( I_diff.col_begin(0)+1, I_diff.col_end(0), I_diff.col_begin(0)+1, 0.0 );
        }

        void make_gvec()
        {
           //std::cerr << "\nunit_cell is\n" << unit_cell << "\n";
           // modify here if gx gy gz changed
           gvec = beams;
           gvec /= unit_cell;

           //std::cerr << "\nGvec is\n" << gvec << "\n";
        }

        void make_rot()
        {
            rot.resize( 2, 2 );
            rot[0][0] = std::cos( rotation_angle ); rot[0][1] = -std::sin( rotation_angle );
            rot[1][0] = -rot[0][1];                 rot[1][1] = rot[0][0];

            //std::cerr << "\nrot is\n" << rot << "\n";
        }

        void make_kt()
        {
            kt.resize( 1, 2 ); //kx, ky 
            kt[0][0] = kt_ac_dc[0][0] * std::sin( tilt[index][0] ) / lambda + kt_ac_dc[0][1];
            kt[0][1] = kt_ac_dc[1][0] * std::sin( tilt[index][1] ) / lambda + kt_ac_dc[1][1];
            kt *= rot;

            //std::cerr << "\nkt is\n" << kt << "\n";
        }

        void update_A_diag()
        {
            for_each( A.diag_begin(), A.diag_end(), gvec.col_begin(0), gvec.col_begin(1),
                     [this]( std::complex<double>& d, double gx, double gy )
                     {
                        double const kx = ((*this).kt)[0][0];
                        double const ky = ((*this).kt)[0][1];
                        d = std::complex<double>{ -2.0 * ( kx*gx + ky*gy ) - gx*gx - gy*gy, 0.0 };
                      }
                    );
        }

        void update_A()
        {
            A.resize( Ar.row(), Ar.col() );
            for_each( A.begin(), A.end(), Ar.begin(),
                      [this]( std::complex<double>& a, unsigned long index )
                      {
                        a = std::complex<double>{ ((*this).ug)[index][0], ((*this).ug)[index][1] };
                      }
                    );
        }

        double make_diff_r()
        {
            //make gv
            make_gvec();
            //make rot
            make_rot();
            //make kt
            make_kt();
            //make diag
            update_A_diag();

            D = std::complex<double>{ 0.0, thickness*3.14159265358979323846*lambda } * A;
            S = expm( D );
            I_diff.resize( S.row(), 1 );
            for_each( S.col_begin(0), S.col_end(0), intensity.col_begin(index), I_diff.col_begin(0),
                      [this]( std::complex<double> const& c, double i_exp, double& i_diff )
                      { i_diff = std::norm(c) - i_exp*((*this).I_ac_dc)[0][0] - ((*this).I_ac_dc)[0][1]; } 
                    );
<<<<<<< HEAD
            //return std::accumulate( I_diff.col_begin(0), I_diff.col_end(0), 0.0, [](double ini, double x){ double X = x*x; return ini + std::sqrt(X+1.0e-10); } );
=======
>>>>>>> 6873d9b10dde494a87dea760870a245eaece7689
            return std::accumulate( I_diff.col_begin(0)+1, I_diff.col_end(0), 0.0, [](double ini, double x){ double X = x*x; return ini + std::sqrt(X+1.0e-10); } );
        }

        double make_diff_quad()
        {
            //make gv
            make_gvec();
            //make rot
            make_rot();
            //make kt
            make_kt();
            //make diag
            update_A_diag();

            D = std::complex<double>{ 0.0, thickness*3.14159265358979323846*lambda } * A;
            S = expm( D );
            I_diff.resize( S.row(), 1 );
            for_each( S.col_begin(0), S.col_end(0), intensity.col_begin(index), I_diff.col_begin(0),
                      [this]( std::complex<double> const& c, double i_exp, double& i_diff )
                      { i_diff = std::norm(c) - i_exp*((*this).I_ac_dc)[0][0] - ((*this).I_ac_dc)[0][1]; } 
                    );
            return std::accumulate( I_diff.col_begin(0), I_diff.col_end(0), 0.0, [](double ini, double x){ double X = x*x; return ini + X*X; } );
            //return std::inner_product( I_diff.col_begin(0)+1, I_diff.col_end(0), I_diff.col_begin(0)+1, 0.0 );
        }
    };

}//namespace f

#endif//SQYOFBOVXMWNPDWMJCVXUGOMQOLCYBEJJGRSKPTNUJITJLGXSUISWMTJGSWLQYBJQMEUTLGKI

