#ifndef VVCSQQCESCQTQRKQCOYFJMWPLKLXPUWHQMDTTFMQGUQUAPBOXJWMLMBNEPOVVWRCVUDVRTPMQ
#define VVCSQQCESCQTQRKQCOYFJMWPLKLXPUWHQMDTTFMQGUQUAPBOXJWMLMBNEPOVVWRCVUDVRTPMQ

#include <cmath>

namespace f
{
    template< typename T = double >
    auto make_brain( T a = 1.0, T b = 0.40584510488433310621, T c = 1.59154943091895335769, T d = 6, T e = 10, T f = 0.03978873577297383394 ) noexcept
    {
        return [=]( T* x ) noexcept
        {
            T const term1 =  x[1] - b*x[0]*x[0] + c*x[0] - d;
            return a * term1 * term1 + e * ( T{1} - f ) * std::cos(x[0]) + e;
        };
    }

}//namespace f

#endif//VVCSQQCESCQTQRKQCOYFJMWPLKLXPUWHQMDTTFMQGUQUAPBOXJWMLMBNEPOVVWRCVUDVRTPMQ

