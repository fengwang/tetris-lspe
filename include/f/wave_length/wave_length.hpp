#ifndef ECEYULPHATNYPPEPUMTTDFCMQHOORLCIXQSNXQKHQDOIDMWSBOJKWFGUDMGGNYVRDJKGLRRCL
#define ECEYULPHATNYPPEPUMTTDFCMQHOORLCIXQSNXQKHQDOIDMWSBOJKWFGUDMGGNYVRDJKGLRRCL

#include <cmath>

namespace f
{

    //returns the relativistic electron wavelength for an input voltage in keV
    template< typename T >
    T wave_length( T v0 )
    {
        v0 = std::abs(v0);
        T const emass = 510.99906; // electron rest mass in keV
        T const hc =  12.3984244; // h * c
        return hc / std::sqrt( v0 * ( v0 + emass+emass ) );
    }

}//namespace f

#endif//ECEYULPHATNYPPEPUMTTDFCMQHOORLCIXQSNXQKHQDOIDMWSBOJKWFGUDMGGNYVRDJKGLRRCL

