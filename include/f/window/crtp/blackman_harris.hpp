#ifndef PDTYCXMWWEFJTFTGPINMPIDRDFJCEDEGOTJHLNFMFCIVNMPJUEVFSGUINYPTLXQJXCBWQGPGM
#define PDTYCXMWWEFJTFTGPINMPIDRDFJCEDEGOTJHLNFMFCIVNMPJUEVFSGUINYPTLXQJXCBWQGPGM

#include <f/window/crtp/window_crtp.hpp>

#include <cstddef>
#include <cmath>

namespace f
{

    template< typename T >
    struct blackman_harris_window : window_crtp< blackman_harris_window< T >, T >
    {
        typedef T               value_type;    
        typedef std::size_t     size_type;
        typedef window_crtp< blackman_harris_window< T >, T > host_type;

        blackman_harris_window( size_type n ) : host_type( n ) {}

        value_type impl( const size_type n ) const
        {
            const value_type pi = 3.1415926535897932384626433;
            const value_type v = pi * 2 * n / static_cast<host_type const&>(*this).size();
            return 0.35875 - 0.48829 * std::cos(v) + 0.14128 * std::cos(v+v) - 0.01168 * std::cos(v+v+v);
        }

    };//struct blackman_harris_window

}//namespace f

#endif//PDTYCXMWWEFJTFTGPINMPIDRDFJCEDEGOTJHLNFMFCIVNMPJUEVFSGUINYPTLXQJXCBWQGPGM
