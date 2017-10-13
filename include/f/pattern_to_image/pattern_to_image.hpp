#ifndef NEODYRPIWCYSRXIXJSBUAFTXAVFXDCDESHDODGOBJBRJCYJGYUXBQVGJLGAJNITUSXAHUJNCS
#define NEODYRPIWCYSRXIXJSBUAFTXAVFXDCDESHDODGOBJBRJCYJGYUXBQVGJLGAJNITUSXAHUJNCS

#include <f/matrix/matrix.hpp>
#include <f/lexical_cast/lexical_cast.hpp>

#include <string>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace f
{

    void pattern_to_image( matrix<unsigned long> const& tilt_mat, matrix<double> const& intensities, matrix<double>& single_image, unsigned long const index, unsigned long block_dim )
    {
        single_image.resize( block_dim, block_dim );
        std::fill( single_image.begin(), single_image.end(), -1.0 );

        unsigned long const central = block_dim >> 1;
        double const radius = static_cast<double>(central);

        // copy measured/simulated intensity
        for ( unsigned long jndex = 0; jndex != tilt_mat.row(); ++jndex )
            single_image[tilt_mat[jndex][0]][tilt_mat[jndex][1]] = intensities[index][jndex];

        matrix<double> square_distance{ tilt_mat.row(), 1 };
        matrix<double> square_distance_se{ tilt_mat.row(), 1 };

        for ( unsigned long r = 0; r != block_dim; ++r )
            for ( unsigned long c = 0; c != block_dim; ++c )
            {
                //skip measured intensities
                if ( single_image[r][c] >= 0.0 ) continue;

                // skip margins
                double offset_r = static_cast<double>(r) - central;
                double offset_c = static_cast<double>(c) - central;
                if ( offset_r * offset_r + offset_c * offset_c >= radius * radius )
                {
                    single_image[r][c] = 0.0;
                    continue;
                }

                // potential field fitting
                // fill-in square_disance matrix
                for ( unsigned long jndex = 0; jndex != square_distance.row(); ++jndex )
                {
                    unsigned long diff_r = r - tilt_mat[jndex][0];
                    unsigned long diff_c = c - tilt_mat[jndex][1];
                    square_distance[jndex][0] = 1.0 / static_cast<double>( diff_r*diff_r + diff_c*diff_c );
                }

                unsigned long use_all = 0;

                if ( use_all )
                {
                    double const w_vf = std::inner_product( intensities.row_begin(index), intensities.row_end(index), square_distance.col_begin(0), 0.0 );
                    double const w_sm = std::accumulate( square_distance.col_begin(0), square_distance.col_end(0), 0.0 );
                    single_image[r][c] = w_vf / w_sm;
                }
                else
                {
                    unsigned long const reference_points_used = 4;
                    square_distance_se = square_distance;
                    std::nth_element( square_distance_se.begin(), square_distance_se.begin()+reference_points_used, square_distance_se.end(), [](double x, double y){ return x > y; } );
                    double const threshold = *(square_distance_se.begin()+reference_points_used-1);

                    double w_vf = 0.0;
                    double w_sm = 0.0;
                    for ( unsigned long jndex = 0; jndex != tilt_mat.row(); ++jndex )
                    {
                        if ( square_distance[jndex][0] >= threshold )
                        {
                            w_vf += square_distance[jndex][0] * intensities[index][jndex];
                            w_sm += square_distance[jndex][0];
                        }
                    }
                    single_image[r][c] = w_vf / w_sm;
                    if ( std::isnan(single_image[r][c]) || std::isinf(single_image[r][c]) || (single_image[r][c] < 0.0) )
                        single_image[r][c] = 0.0;
                }
            }

        //debug:
        //
        //std::string const image_path = lexical_cast<std::string>(index) + std::string{"_image.txt"};
        //single_image.save_as( image_path );
    }

    void pattern_to_image( matrix<long> const& beams, matrix<double> const& tilts, matrix<double> const& intensities, matrix<double>& image, double const rotation )
    {
        long const beam_offset_r = *std::min_element(beams.col_begin(1), beams.col_end(1));
        long const beam_offset_c = *std::min_element(beams.col_begin(2), beams.col_end(2));
        long const beam_offset_rm = *std::max_element(beams.col_begin(1), beams.col_end(1));
        long const beam_offset_cm = *std::max_element(beams.col_begin(2), beams.col_end(2));

        long const grid_dim = std::max( beam_offset_cm, beam_offset_rm ) - std::min( beam_offset_c, beam_offset_r ) + 1;

        //unsigned long const total_patterns = intensities.row();

        unsigned long tilt_x = 7;
        unsigned long tilt_y = 8;
        if ( tilts.col() == 2 )
        {
            tilt_x = 0;
            tilt_y = 1;
        }

        double const size_x = *std::max_element( tilts.col_begin(tilt_x), tilts.col_end(tilt_x) ) - *std::min( tilts.col_begin(tilt_x), tilts.col_end(tilt_x) );
        double const size_y = *std::max_element( tilts.col_begin(tilt_y), tilts.col_end(tilt_y) ) - *std::min( tilts.col_begin(tilt_y), tilts.col_end(tilt_y) );
        unsigned long const block_dim = static_cast<unsigned long>( std::ceil( std::max(size_x, size_y) * 10.0 ) ) | 0x1; // make block_dim odd

        //tilt matrix stores the coordinates of measured intensities in form of [ x, y ], size is [N][2]
        matrix<unsigned long> tilt_mat{ tilts.row(), 2 };
        unsigned long const central = block_dim >> 1;

        auto transfer = [central]( double x ) { return central + static_cast<unsigned long>( std::ceil( x * 10.0 ) ); };
        if ( 1 )
        {
            matrix<double> tilt_mat_{ 2, tilts.row() };
            std::copy( tilts.col_begin(tilt_x), tilts.col_end(tilt_x), tilt_mat_.row_begin(0) );
            std::copy( tilts.col_begin(tilt_y), tilts.col_end(tilt_y), tilt_mat_.row_begin(1) );
            matrix<double> rot_mat{ 2, 2 };
            rot_mat[0][0] = std::cos(rotation); rot_mat[0][1] = std::sin(rotation);
            rot_mat[1][0] =-std::sin(rotation); rot_mat[1][1] = std::cos(rotation);
            //!!!! TODO
            matrix<double> rotated_mat = rot_mat * tilt_mat_;
            std::transform( rotated_mat.row_begin(0), rotated_mat.row_end(0), tilt_mat.col_begin(0), transfer );
            std::transform( rotated_mat.row_begin(1), rotated_mat.row_end(1), tilt_mat.col_begin(1), transfer );
        }
        else
        {
            std::transform( tilts.col_begin(tilt_x), tilts.col_end(tilt_x), tilt_mat.col_begin(0), transfer );
            std::transform( tilts.col_begin(tilt_y), tilts.col_end(tilt_y), tilt_mat.col_begin(1), transfer );
        }

        image.resize( grid_dim*block_dim, grid_dim*block_dim );
        std::fill( image.begin(), image.end(), 0.0 );

        matrix<double> single_image{block_dim, block_dim};

        if(1)
        {
            for ( unsigned long index = 0; index != beams.row(); ++index )
            {
                //std::cerr << "Constructing Beam " << beams[index][1] << " " << beams[index][2] << "\n";
                pattern_to_image( tilt_mat, intensities, single_image, index, block_dim );
                {
                    //debug
                    /*
                    {
                        std::string file_name = lexical_cast<std::string>( index ) + std::string{".bmp"};
                        single_image.save_as_bmp( file_name );
                    }
                    */
#if 1
                    long r_off = beams[index][1] - beam_offset_r;
                    long c_off = beams[index][2] - beam_offset_c;
#else
                    long c_off = beams[index][1] - beam_offset_r;
                    long r_off = beams[index][2] - beam_offset_c;
#endif

                    long row_offset = r_off * block_dim;
                    long col_offset = c_off * block_dim;
                    assert( row_offset < static_cast<long>(image.row()) );
                    assert( col_offset < static_cast<long>(image.col()) );

                    for ( unsigned long row = 0; row != block_dim; ++row )
                        std::copy( single_image.row_begin(row), single_image.row_end(row), image.row_begin( row_offset+row ) + col_offset );
                }
            }
        }
        else
        {
            for ( long r = 0; r != grid_dim; ++r )
                for ( long c = 0; c != grid_dim; ++c )
                {
                    unsigned long const index = r * grid_dim + c;
                    pattern_to_image( tilt_mat, intensities, single_image, index, block_dim );
                    {
                        long r_off = beams[index][1] - beam_offset_r;
                        long c_off = beams[index][2] - beam_offset_c;
                        /*
                        long c_off = beams[index][1] - beam_offset_r;
                        long r_off = beams[index][2] - beam_offset_c;
                        */
                        long row_offset = r_off * block_dim;
                        long col_offset = c_off * block_dim;


                        for ( unsigned long row = 0; row != block_dim; ++row )
                        {
                            std::copy( single_image.row_begin(row), single_image.row_end(row), image.row_begin( row_offset+row ) + col_offset );
                        }
                    }
                }
        }
    }

    void pattern_to_image( std::string const& beams_path, std::string const& tilts_path, std::string const& intensities_path, std::string const& image_path, double const rotation = 0.0 )
    {
        std::cout << "Pattern to Image - 1: generating " << image_path << std::endl;


        matrix<double> tilts;
        matrix<double> intensities;
        matrix<double> image;
        matrix<long> beam;

        tilts.load( tilts_path );
        intensities.load( intensities_path );
        beam.load(beams_path);
#if 1
        for ( unsigned long c = 0; c != intensities.col(); ++c )
        {
            double const sum = std::accumulate( intensities.col_begin(c), intensities.col_end(c), 0.0 ) + 1.0e-100;
            std::for_each( intensities.col_begin(c), intensities.col_end(c), [sum]( double&x ){ x /= sum; } );
        }
#endif

        pattern_to_image( beam, tilts, intensities, image, rotation );

#if 0
        {
            double const alpha = 2.71828182845904523536;
            std::for_each( image.begin(), image.end(), [alpha]( double& x ) { x = std::log(1.0+alpha*x); } );
        }
#endif
        {
            double const gap = *std::max_element( image.begin(), image.end() ) - *std::min_element( image.begin(), image.end() );
            std::for_each( image.begin(), image.end(), [gap]( double& x ) { if ( std::abs(x) > 1.0e-10 ) x += gap / 750.0; } );
        }

        //image.save_as_bmp( image_path, std::string{"hotblue"}, std::string{"log"} );
        //image.save_as_bmp( image_path, std::string{"hotblue"}, std::string{"log1"} );
        //image.save_as_bmp( image_path, std::string{"hotblue"}, std::string{"logpi"} );
        //image.save_as_bmp( image_path, std::string{"hotblue"}, std::string{"log3"} );
        //image.save_as_bmp( image_path, std::string{"hotblue"}, std::string{"log4"} );
        image.save_as_bmp( image_path, std::string{"hotblue"}, std::string{"logx"} );
        //image.save_as_bmp( image_path, std::string{"jet"}, std::string{"logx"} );
        //image.save_as_bmp( image_path );
        //
        //image.save_as( image_path + std::string{".txt"} );

        {
            // convert image_path+ '.bmp '  -transparent white image_path+ '.eps'
            // eeps image_path+ '.eps'
            std::string cmd_1 = std::string{"/usr/local/bin/convert "} + image_path + std::string{".bmp -transparent white "} + image_path + std::string{".eps"};
            std::string cmd_2 = std::string{"/Users/feng/bin/eeps "} + image_path + std::string{".eps"};
            std::system( cmd_1.c_str() );
            std::system( cmd_2.c_str() );
        }
    }

    void pattern_to_image( std::string const& beams_path, std::string const& tilts_path, std::string const& intensities_path, std::string const& image_path, double const rotation, unsigned long tilts_involved )
    {
        std::cout << "Pattern to Image - 2: generating " << image_path << std::endl;

        matrix<double> tilts;
        matrix<double> intensities;
        matrix<double> image;
        matrix<long> beam;

        tilts.load( tilts_path );
        intensities.load( intensities_path );
        {
            if ( intensities.col() > tilts_involved )
            {
                //double const mx = *std::max_element( intensities.begin(), intensities.end() );
                //double const mn = *std::min_element( intensities.begin(), intensities.end() );
                for ( unsigned long c = tilts_involved; c != intensities.col(); ++c )
                    std::fill( intensities.col_begin(c), intensities.col_end(c), 1.0e-100 );
                //intensities[0][0] = mx;
            }
        }
        beam.load(beams_path);

        for ( unsigned long c = 0; c != intensities.col(); ++c )
        {
            double const sum = std::accumulate( intensities.col_begin(c), intensities.col_end(c), 0.0 );
            if ( sum > 1.0e-10 )
                std::for_each( intensities.col_begin(c), intensities.col_end(c), [sum]( double&x ){ x /= sum; } );
        }

        pattern_to_image( beam, tilts, intensities, image, rotation );

        {
            double const gap = *std::max_element( image.begin(), image.end() ) - *std::min_element( image.begin(), image.end() );
            std::for_each( image.begin(), image.end(), [gap]( double& x ) { if ( std::abs(x) > 1.0e-10 ) x += gap / 750.0; } );
        }

        //image.save_as_bmp( image_path );
        image.save_as_bmp( image_path, std::string{"hotblue"}, std::string{"logx"} );

        {
            // convert image_path+ '.bmp '  -transparent white image_path+ '.eps'
            // eeps image_path+ '.eps'
            std::string cmd_1 = std::string{"/usr/local/bin/convert "} + image_path + std::string{".bmp -transparent white "} + image_path + std::string{".eps"};
            std::string cmd_2 = std::string{"/Users/feng/bin/eeps "} + image_path + std::string{".eps"};
            std::system( cmd_1.c_str() );
            std::system( cmd_2.c_str() );
        }
    }

}//namespace f

#endif//NEODYRPIWCYSRXIXJSBUAFTXAVFXDCDESHDODGOBJBRJCYJGYUXBQVGJLGAJNITUSXAHUJNCS

