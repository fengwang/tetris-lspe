#ifndef PESJLPOQVSVWXMMYBRTIAAWVCPRDRQMAUQMXAHXIVGLVQHUSEOUWAMRPJCWMPGSNWOKCVEDIN
#define PESJLPOQVSVWXMMYBRTIAAWVCPRDRQMAUQMXAHXIVGLVQHUSEOUWAMRPJCWMPGSNWOKCVEDIN

#include <cmath>

namespace f
{
    template< typename T = double >
    auto make_rastrigin( unsigned long n ) noexcept
    {
        return [n]( T* x ) noexcept
        {
            T ans{10*n};
            for ( unsigned long index = 0; index != n; ++index )
                ans += x[index]*x[index] + static_cast<T>(10) * std::cos( 6.28318530717958647692 * x[index] );
            return ans;
        };
    }

}//namespace f

#endif//PESJLPOQVSVWXMMYBRTIAAWVCPRDRQMAUQMXAHXIVGLVQHUSEOUWAMRPJCWMPGSNWOKCVEDIN
