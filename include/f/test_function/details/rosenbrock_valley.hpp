#ifndef PUEHKLRWTPCKXBOFVTDQSNGCTSPBWWAECKTXFDYNBXLROOMUSEBFIFQLRTMMFOWFNEXFMASBB
#define PUEHKLRWTPCKXBOFVTDQSNGCTSPBWWAECKTXFDYNBXLROOMUSEBFIFQLRTMMFOWFNEXFMASBB

namespace f
{

    template< typename T = double >
    auto make_rosenbrock_valley( unsigned long n ) noexcept
    {
        return [n]( T* x ) noexcept
        {
            T ans{0};
            for ( unsigned long index = 0; index != n-1; ++index )
            {
                T const tmp = static_cast<T>( 1 ) - x[index];
                T const tmp2 = x[index+1] - x[index]*x[index];
                ans += static_cast<T>( 100 ) * tmp2*tmp2 +  tmp * tmp;
            }
            return ans;
        };
    }

}//namespace f

#endif//PUEHKLRWTPCKXBOFVTDQSNGCTSPBWWAECKTXFDYNBXLROOMUSEBFIFQLRTMMFOWFNEXFMASBB

