#ifndef VMKJHDEKWXOTECOQRITCPCMIYXUXWCYWCDSWBNQERVCGONANKGCNDXQYKHQKBQTSFRQXMAPDX
#define VMKJHDEKWXOTECOQRITCPCMIYXUXWCYWCDSWBNQERVCGONANKGCNDXQYKHQKBQTSFRQXMAPDX

#include <cmath>

namespace f
{
    template< typename T = double >
    auto make_schwelfel( unsigned long n ) noexcept
    {
        return [n]( T* x) noexcept 
        {
            T ans{0};
            for ( unsigned long index = 0; index != n; ++index )
                ans -= x[index] * std::sin( std::abs(x[index]) );
            return ans;
        };
    }

}//namespace f

#endif//VMKJHDEKWXOTECOQRITCPCMIYXUXWCYWCDSWBNQERVCGONANKGCNDXQYKHQKBQTSFRQXMAPDX

