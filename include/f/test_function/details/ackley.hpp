#ifndef UHYHUMRXTTCOBUQYEKPIDJLBUBCNEMCTDPYINIOGBIEFJASIMBCHDLQBEJHLXJJNRSKVLPFUX
#define UHYHUMRXTTCOBUQYEKPIDJLBUBCNEMCTDPYINIOGBIEFJASIMBCHDLQBEJHLXJJNRSKVLPFUX

#include <cmath>

namespace f
{

    // minimal 0 at ( 0, 0, ..., 0 )
    template< typename T = double >
    auto make_ackley( unsigned long n, T a = 20, T b = 0.2, T c = 6.28318530717958647692 ) noexcept
    {
        return [=]( T* x ) noexcept
        {
            T term1{0};
            T term2{0};
            for ( unsigned long index = 0; index != n; ++index )
            {
                term1 += x[index] * x[index];
                term2 += std::cos( c * x[index] );
            }

            return - a * std::exp( -b * std::sqrt( term1/static_cast<T>(n) ) ) - std::exp( term2/static_cast<T>(n) ) + a + 2.71828182845904523536;
        };
    }

}//namespace f

#endif//UHYHUMRXTTCOBUQYEKPIDJLBUBCNEMCTDPYINIOGBIEFJASIMBCHDLQBEJHLXJJNRSKVLPFUX

