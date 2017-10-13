#ifndef CVFJFHOKEMQILTTELCJQFWNRYXVYFIYBMNYMAVOJMIMLCVXMDRNTWEIBIRJSGRVTAIVYRXFYY
#define CVFJFHOKEMQILTTELCJQFWNRYXVYFIYBMNYMAVOJMIMLCVXMDRNTWEIBIRJSGRVTAIVYRXFYY

#include <f/window/crtp/window_crtp.hpp>

#include <cstddef>

namespace f
{

    template< typename T >
    struct triangle_window : window_crtp< triangle_window< T >, T >
    {
        typedef T               value_type;    
        typedef std::size_t     size_type;
        typedef window_crtp< triangle_window< T >, T > host_type;

        triangle_window( const size_type n ) : host_type( n ) {}

        value_type impl( const size_type n ) const
        {
            const value_type N = static_cast<host_type const&>(*this).size();
            const value_type N_2 = N / 2.0;

            if ( n < N_2 )
                return n / N_2;

            return (N-n) / N_2;
        }

    };//struct triangle_window

}//namespace f

#endif//CVFJFHOKEMQILTTELCJQFWNRYXVYFIYBMNYMAVOJMIMLCVXMDRNTWEIBIRJSGRVTAIVYRXFYY
