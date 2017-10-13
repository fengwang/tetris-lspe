#ifndef JAFMREKQDIKDDVJUUWOMXPXCMYWEXHCPUMKRITAXLLDDUQIOBGLAJITTESCYVQPEWYYABRLGV
#define JAFMREKQDIKDDVJUUWOMXPXCMYWEXHCPUMKRITAXLLDDUQIOBGLAJITTESCYVQPEWYYABRLGV

namespace f
{
    template< typename T = double >
    auto make_camel_back() noexcept
    {
        return []( T* x ) noexcept
        {
            T const x1 = x[0];
            T const x2 = x[1];
            T const x11 = x1*x1;
            T const x12 = x1*x2;
            T const x22 = x2*x2;

            return ( T{4} - T{2.1}*x11 + x11*x11/T{3} ) * x22 + x12 + ( T{-4} + T{4}*x22 ) *x22;
        };
    }

}//namespace f

#endif//JAFMREKQDIKDDVJUUWOMXPXCMYWEXHCPUMKRITAXLLDDUQIOBGLAJITTESCYVQPEWYYABRLGV
