#ifndef SJUEDGDPBRTVXFWJYIRHUWHUFBWXQCWSQBCBNEIPFKDDMYKHRPWFAKCASTBBRYGKLIRTRXYQP
#define SJUEDGDPBRTVXFWJYIRHUWHUFBWXQCWSQBCBNEIPFKDDMYKHRPWFAKCASTBBRYGKLIRTRXYQP

#include <cmath>

namespace f
{
    template< typename T = double >
    auto make_griewank( unsigned long n ) noexcept
    {
       return [n]( T* x ) noexcept 
       {
           T term1{0};
           T term2{1};
           for ( unsigned long index = 0; index != n; ++index )
           {
                term1 += x[index]*x[index];
                term2 *= std::cos( x[index] / std::sqrt( static_cast<T>(index+1) ) );
           }
           return static_cast<T>(1) + term1 + term2;
       };
    }

}//namespace f

#endif//SJUEDGDPBRTVXFWJYIRHUWHUFBWXQCWSQBCBNEIPFKDDMYKHRPWFAKCASTBBRYGKLIRTRXYQP

