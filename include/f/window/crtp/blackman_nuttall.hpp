#ifndef VHRSDSKOHNUUNNUKOWWICXACGJYHBPOHMXMEMTPYERFXTAMWCXRDAACOUSGXFPYHJDOOTUWQU
#define VHRSDSKOHNUUNNUKOWWICXACGJYHBPOHMXMEMTPYERFXTAMWCXRDAACOUSGXFPYHJDOOTUWQU

#include <f/window/crtp/window_crtp.hpp>

#include <cstddef>
#include <cmath>

namespace f
{

    template< typename T >
    struct blackman_nuttall_window : window_crtp< blackman_nuttall_window< T >, T >
    {
        typedef T               value_type;    
        typedef std::size_t     size_type;
        typedef window_crtp< blackman_nuttall_window< T >, T > host_type;

        blackman_nuttall_window( size_type n ) : host_type( n ) {}

        value_type impl( const size_type n ) const
        {
            const value_type pi = 3.1415926535897932384626433;
            const value_type v = pi * 2 * n / static_cast<host_type const&>(*this).size();
            return 0.3635819 - 0.4891775 * std::cos(v) + 0.1365995 * std::cos(v+v) - 0.0106411 * std::cos(v+v+v);
        }

    };//struct blackman_nuttall_window

}//namespace f

#endif//VHRSDSKOHNUUNNUKOWWICXACGJYHBPOHMXMEMTPYERFXTAMWCXRDAACOUSGXFPYHJDOOTUWQU
