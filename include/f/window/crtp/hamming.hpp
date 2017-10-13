#ifndef SYRPQJEKLOVOGXDNGRMYYBGRXCAGQLODPVLOVWSTAUAORQKTDBRYHMSPIUEBNHRROAORFMDMO
#define SYRPQJEKLOVOGXDNGRMYYBGRXCAGQLODPVLOVWSTAUAORQKTDBRYHMSPIUEBNHRROAORFMDMO

#include <f/window/crtp/window_crtp.hpp>

#include <cstddef>
#include <cmath>

namespace f
{

    template< typename T >
    struct hamming_window : window_crtp< hamming_window< T >, T >
    {
        typedef T               value_type;    
        typedef std::size_t     size_type;
        typedef window_crtp< hamming_window< T >, T > host_type;

        hamming_window( size_type n ) : host_type( n ) {}

        value_type impl( const size_type n ) const
        {
            const value_type pi = 3.1415926535897932384626433;
            const value_type v = pi * 2 * n / static_cast<host_type const&>(*this).size();
            return 0.54 - 0.46 * std::cos(v);
        }

    };//struct hamming_window

}//namespace f

#endif//SYRPQJEKLOVOGXDNGRMYYBGRXCAGQLODPVLOVWSTAUAORQKTDBRYHMSPIUEBNHRROAORFMDMO
