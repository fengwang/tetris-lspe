#ifndef IPDHADYCTHXOCNHMNTCGSAKTYPVFPKVFSFKBYBUMFAWDFBEUBBYTUVODVWWYEXGHBWQYIGISS
#define IPDHADYCTHXOCNHMNTCGSAKTYPVFPKVFSFKBYBUMFAWDFBEUBBYTUVODVWWYEXGHBWQYIGISS

#include <f/matrix/matrix.hpp>

#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <functional>

namespace f
{

    struct reflection_residual
    {
        matrix<int> beam;           //[index, h, k, l]
        matrix<double> intensity;   //[.....]
        unsigned long unknown_size; // ug*2

        matrix<double> accu;        //[index_hash, intensity_accumulation]

        std::vector<matrix<double> > recs; //[[intensity,index][intensity, index]] ...

        std::vector<std::function<double(double*)> > merits;

        reflection_residual( std::string const& beam_path, std::string const& intensity_path, unsigned long beams )
        {
            beam.load( beam_path );
            intensity.load( intensity_path );
            assert( beam.row() >= intensity.row() );
            assert( beam.col() == 4 );

            std::for_each( intensity.begin(), intensity.end(), []( double& x ){ if ( x < 0.0 ) x = 0.0; } );

            unknown_size = beams + beams;


            //Hash
            accu.resize( intensity.row(), 2 );

            auto const& hash_fun = []( int h, int k, int l ) 
            { 
                double const dh_ = std::abs(h);
                double const dk_ = std::abs(k);
                double const dl_ = std::abs(l);
                double const dh = std::max( dh_, std::max( dk_, dl_ ) );
                double const dl = std::min( dh_, std::min( dk_, dl_ ) );
                double const dk = dh_ + dk_ + dl_ - dh - dl;

                return 1000000 * dh + 1000 * dk + dl; 
            };

            for ( unsigned long r = 0; r != accu.row(); ++r )
            {
                accu[r][0] = hash_fun( beam[r][1], beam[r][2], beam[r][3] );
                accu[r][1] = std::accumulate( intensity.row_begin(r), intensity.row_end(r), 0.0 );
            }


            //Unique
            std::vector<double> weighs;
            weighs.resize( accu.row() );
            std::copy( accu.col_begin(0), accu.col_end(0), weighs.begin() );
            std::sort( weighs.begin(), weighs.end() );
            weighs.erase( std::unique( weighs.begin(), weighs.end(), [](double x, double y){ return std::abs(x-y) < 0.1; } ), weighs.end() );

            //Construct reflections
            for ( unsigned long index = 1; index != weighs.size(); ++index )
            {
                double const key = weighs[index];
                unsigned long const elems = std::count_if( accu.col_begin(0), accu.col_end(0), [key](double x){ return std::abs(x-key) < 0.1; } );
                matrix<double> mt{ elems, 3 };
                unsigned long kndex = 0;
                for ( unsigned long jndex = 1; jndex != accu.row(); ++jndex )
                {
                    if ( std::abs( accu[jndex][0] - key ) < 0.1 )
                    {
                        mt[kndex][0] = jndex;
                        mt[kndex][1] = accu[jndex][1];
                        kndex++;
                    }
                }
                recs.push_back( mt );
            }
            if ( 1 )
            {
                std::cout << "\nThe size of recs is " << recs.size() << "\n";
                for ( auto& mat : recs )
                    std::cout << mat << "\n\n";
            }

            // construct merits
            double const total_energy = intensity.col();
            double const total_residual = total_energy - accu[0][1];
            double const total_beam = intensity.row();
            double const threshold = total_residual / ( 2.0 * ( total_beam - 1 ) ); //2.78 for experimental
            for ( unsigned long index = 0; index != recs.size(); ++index )
            {
                //merits
                auto& mat = recs[index];
                double const average = std::accumulate( mat.col_begin(1), mat.col_end(1), 0.0 ) / static_cast<double>(mat.row());
                if ( average >= threshold )
                {
                    double sig = 0.0;
                    std::for_each( mat.col_begin(1), mat.col_end(1), [&sig,average]( double x ){ sig += (x-average)*(x-average); } );
                    sig /= static_cast<double>(mat.row());
                    sig = std::sqrt( sig );
                    if ( sig >= average * 0.1 ) 
                    {
                        //std::cout << "\nAt index " << index << ", the sig is " <<  sig << ", and the average is " << average << ", ignoring.\n";
                        //std::cout << "\nthe stored matrix is \n" << mat << "\n";
                        continue; // noops -- no symmetry
                    }

                    //std::cout << "\nAppending NONZERO mat:\n" << mat << "\n";

                    merits.emplace_back( 
                                        [this, index, average, total_residual](double* x)
                                        {
                                            auto& mat = ((*this).recs)[index];
                                            assert( mat.row() );
                                            assert( mat.col() );

                                            for ( unsigned long jndex = 0; jndex != mat.row(); ++jndex )
                                            {
                                                unsigned long const offset = static_cast<unsigned long>( mat[jndex][0] );
                                                mat[jndex][2] = x[offset+offset] * x[offset+offset] + x[offset+offset+1] * x[offset+offset+1];
                                            }

                                            double res = 0.0;
                                            double const mean = std::accumulate( mat.col_begin(2), mat.col_end(2), 0.0 ) / static_cast<double>( mat.row() );
                                            std::for_each( mat.col_begin(2), mat.col_end(2), [mean, &res]( double x ){ res += (mean-x) * (mean-x); } );
                                            res /= static_cast<double>( mat.row() );

                                            return total_residual * res * average;
                                        }
                                    );
                }
                else
                {

                    std::cout << "\nAssuming zero mat:\n" << mat << "\n";

                    merits.emplace_back( 
                                        [this, index, average, total_residual](double* x)
                                        {
                                            double res = 0.0;

                                            auto& mat = ((*this).recs)[index];
                                            assert( mat.row() );
                                            assert( mat.col() );

                                            for ( unsigned long jndex = 0; jndex != mat.row(); ++jndex )
                                            {
                                                unsigned long const offset = static_cast<unsigned long>( mat[jndex][0] );
                                                res += x[offset+offset] * x[offset+offset];
                                                res += x[offset+offset+1] * x[offset+offset+1];
                                            }
                                            return  total_residual * res / average;
                                        }
                                    );
                }
            }






        }


        std::function<double(double*)> make_normal_residual()
        {
            return [this]( double* x )
            {
                double res = 0.0;

                for ( auto& func : (*this).merits )
                    res += func( x );

                return res;
            };
        }
        


        
    };


}//namespace f

#endif//IPDHADYCTHXOCNHMNTCGSAKTYPVFPKVFSFKBYBUMFAWDFBEUBBYTUVODVWWYEXGHBWQYIGISS

