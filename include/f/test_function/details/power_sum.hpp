#ifndef YLREJJBOXNJGQHXRCVJCXSFAYFXWOQLTNMUKOJCPRUHBYXCMHBMTLLTFXEIYUQMHUOYOHAPON
#define YLREJJBOXNJGQHXRCVJCXSFAYFXWOQLTNMUKOJCPRUHBYXCMHBMTLLTFXEIYUQMHUOYOHAPON

#include <cmath>

namespace f
{
    namespace power_sum_private
    {
        template< typename T >
        T power( T x, unsigned long n )
        {
            if ( n == 1 ) return x;

            if ( n == 0 ) return T{1};

            if ( n & 1 ) return x * power( x, n-1 );
            
            T const power_ = power( x, n << 1 );
            return power_ * power_;
        }
    }

    template< typename T = double >
    auto make_power_sum( unsigned long n ) noexcept
    {
        return [n]( T* x ) noexcept 
        {
            T ans{0};

            for ( unsigned long index = 0; index != n; ++index )
                ans += power_sum_private::power( std::abs( x[index] ), index+2 );

            return ans;
        };
    }

}//namespace f

#endif//YLREJJBOXNJGQHXRCVJCXSFAYFXWOQLTNMUKOJCPRUHBYXCMHBMTLLTFXEIYUQMHUOYOHAPON

