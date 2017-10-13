#ifndef SWKJSAUKXLRWXSRHPECKHMIDGNWWAPLCEKTKQQPYPBUCLLCTJKVSGIRQSAOHIDTYVPCUGFFVU
#define SWKJSAUKXLRWXSRHPECKHMIDGNWWAPLCEKTKQQPYPBUCLLCTJKVSGIRQSAOHIDTYVPCUGFFVU

#include <f/window/crtp/window_crtp.hpp>

#include <cstddef>
#include <cmath>

namespace f
{

    template< typename T >
    struct exact_blackman_window : window_crtp< exact_blackman_window< T >, T >
    {
        typedef T               value_type;    
        typedef std::size_t     size_type;
        typedef window_crtp< exact_blackman_window< T >, T > host_type;

        exact_blackman_window( size_type n ) : host_type( n ) {}

        value_type impl( const size_type n ) const
        {
            const value_type pi = 3.1415926535897932384626433;
            const value_type v = pi * 2 * n / static_cast<host_type const&>(*this).size();
            return 0.42659071367153912296 - 0.49656061908856405847 * std::cos(v) + 0.07684866723989681857 * std::cos(v+v);
        }

    };//struct exact_blackman_window

}//namespace f

#endif//SWKJSAUKXLRWXSRHPECKHMIDGNWWAPLCEKTKQQPYPBUCLLCTJKVSGIRQSAOHIDTYVPCUGFFVU
