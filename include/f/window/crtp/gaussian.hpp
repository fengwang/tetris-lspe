#ifndef MJBOMUPUMQJBMKGDVNPCVDLQOFTVAFWQGNUIDMXYUWVGXKFMQREVPDLUCNTDRCWNVMFGLUXJV
#define MJBOMUPUMQJBMKGDVNPCVDLQOFTVAFWQGNUIDMXYUWVGXKFMQREVPDLUCNTDRCWNVMFGLUXJV

#include <f/window/crtp/window_crtp.hpp>

#include <cstddef>
#include <cmath>

namespace f
{

    template< typename T >
    struct gaussian_window : window_crtp< gaussian_window< T >, T >
    {
        typedef T               value_type;    
        typedef std::size_t     size_type;
        typedef window_crtp< gaussian_window< T >, T > host_type;

        value_type alpha_;

        gaussian_window( size_type n, const value_type alpha ) : host_type( n ), alpha_(alpha) {}

        value_type impl( const size_type n ) const
        {
            const value_type N = static_cast<host_type const&>(*this).size();
            const value_type p = alpha_ * ( (n+n) / N - 1 );

            return std::exp( - 0.5 * p * p );
        }

    };//struct gaussian_window

}//namespace f

#endif//MJBOMUPUMQJBMKGDVNPCVDLQOFTVAFWQGNUIDMXYUWVGXKFMQREVPDLUCNTDRCWNVMFGLUXJV
