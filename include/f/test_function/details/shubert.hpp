#ifndef DWJJSUFDPQUWYQVNKYJEMLTVOCGWCPAAAREVAGBHIUDFHSHOBPFQTVUDVQFWTYCIHYTSJDGUX
#define DWJJSUFDPQUWYQVNKYJEMLTVOCGWCPAAAREVAGBHIUDFHSHOBPFQTVUDVQFWTYCIHYTSJDGUX

#include <cmath>

namespace f
{
    template< typename T = double >
    auto make_shubert() noexcept
    {
        return []( T*x ) noexcept
        {
            T term1{0};
            T term2{0};

            for ( unsigned long index = 0; index != 5; ++index )
            {
                term1 += T{index+1} * std::cos( T{index+2}*x[0] + T{1} );
                term2 += T{index+1} * std::cos( T{index+2}*x[1] + T{1} );
            }

            return - term1 * term2;
        };
    }

}//namespace f

#endif//DWJJSUFDPQUWYQVNKYJEMLTVOCGWCPAAAREVAGBHIUDFHSHOBPFQTVUDVQFWTYCIHYTSJDGUX

