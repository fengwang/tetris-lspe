#ifndef XRKBOSVQSOACLJXAJXNOHTJOKPFPFPCLIEKYFAIJFOFWQPCRTMGKSLYLJGQPNUHJJKWNRHXQE
#define XRKBOSVQSOACLJXAJXNOHTJOKPFPFPCLIEKYFAIJFOFWQPCRTMGKSLYLJGQPNUHJJKWNRHXQE

#include <cmath>

namespace f
{
    template< typename T = double >
    auto make_drop_wave() noexcept
    {
        return []( T* x ) noexcept
        {
            T const nm = x[0]*x[0] + x[1]*x[1];
            return - ( T{1} + std::cos(T{12}*std::sqrt(nm)) ) / ( T{0.5}*nm + T{2} );
        };
    }

}//namespace f

#endif//XRKBOSVQSOACLJXAJXNOHTJOKPFPFPCLIEKYFAIJFOFWQPCRTMGKSLYLJGQPNUHJJKWNRHXQE

