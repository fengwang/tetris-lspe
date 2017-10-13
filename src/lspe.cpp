//#ifdef _MSC_VER
//    #include <io.h>
//#else
//    #include <unistd.h>
//#endif

#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <algorithm>


#include <lspe.hpp>

using namespace std;

void train_lspe( std::vector<double>& weights_, long training_set_, double lambda_ );
void play_lspe( std::vector<double>& weights_ );

int main( int argc, char const* argv[] )
{
    srand( 0 );
    std::vector<double> weights;

    train_lspe( weights, 100000, 0.001 );

    srand( 0 );
    play_lspe( weights );

    return 0;
}

void play_lspe( std::vector<double>& weights_ )
{
    lspe lat;
    tetris t;

    while ( !t.gameover )
    {
        auto const& ba = lat.select_action( t, weights_.begin() );
        t.play_action( ba, true );
    }
}

void train_lspe( std::vector<double>& weights_, long training_set_, double lambda_ )
{
    lspe lat( training_set_, lambda_ );
    lat();

    weights_.resize( lat.theta.size() );
    std::copy( lat.theta.begin(), lat.theta.end(), weights_.begin() );
}

