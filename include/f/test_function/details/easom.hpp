#ifndef EVFPBJUSMCMHLHKWHVMHFJYFAQAIGBFAHMQLONXFLVXADJDLVTIFCUDDNRYOTHTHPQJKDBSUY
#define EVFPBJUSMCMHLHKWHVMHFJYFAQAIGBFAHMQLONXFLVXADJDLVTIFCUDDNRYOTHTHPQJKDBSUY

#include <cmath>

namespace f
{
    template< typename T = double >
    auto make_easom() noexcept
    {
        return []( T* x ) noexcept
        {
            T const pi = 3.1415926535897932384626433;
            return -std::cos( x[0] ) * std::cos( x[1] ) * std::exp( -(x[0]-pi)*(x[0]-pi) - (x[1]-pi)*(x[1]-pi) );
        };
    }

}//namespace f

#endif//EVFPBJUSMCMHLHKWHVMHFJYFAQAIGBFAHMQLONXFLVXADJDLVTIFCUDDNRYOTHTHPQJKDBSUY

