#ifndef MMLNFYKDCPWMBCNJBIFKRSOPRBEUIWMHRNDQUOWNFKDIFEYRYVVYPXHEUXJNDUGOJLJSVJKJJ
#define MMLNFYKDCPWMBCNJBIFKRSOPRBEUIWMHRNDQUOWNFKDIFEYRYVVYPXHEUXJNDUGOJLJSVJKJJ

#include <numeric>

namespace f
{

    template< typename T = double >
    auto make_de_jong( unsigned long n ) noexcept
    {
        return [n]( T* x ) noexcept
        {
            return std::inner_product( x, x+n, x, T{0} );
        };
    }

}//namespace f

#endif//MMLNFYKDCPWMBCNJBIFKRSOPRBEUIWMHRNDQUOWNFKDIFEYRYVVYPXHEUXJNDUGOJLJSVJKJJ

