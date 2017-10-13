#ifndef SAUAOUSQFVGGRMUVMCIENMROUUSIVNJCKGQSQSHUDQEBHJPAPMSMTVXPPNIUDYJJSPCQEKFMM
#define SAUAOUSQFVGGRMUVMCIENMROUUSIVNJCKGQSQSHUDQEBHJPAPMSMTVXPPNIUDYJJSPCQEKFMM

#include <f/window/crtp/window_crtp.hpp>

#include <cstddef>
#include <cmath>

namespace f
{

    template< typename T >
    struct flat_top_window : window_crtp< flat_top_window< T >, T >
    {
        typedef T               value_type;    
        typedef std::size_t     size_type;
        typedef window_crtp< flat_top_window< T >, T > host_type;

        flat_top_window( size_type n ) : host_type( n ) {}

        value_type impl( const size_type n ) const
        {
            const value_type pi = 3.1415926535897932384626433;
            const value_type v = pi * 2 * n / static_cast<host_type const&>(*this).size();
            return  0.21557895 - 0.41663158 * std::cos(v) + 0.277263158 * std::cos(v+v) - 0.083578947 * std::cos(v+v+v) + 0.006947368 * std::cos(v+v+v+v); 
        }

    };//struct flat_top_window

}//namespace f

#endif//SAUAOUSQFVGGRMUVMCIENMROUUSIVNJCKGQSQSHUDQEBHJPAPMSMTVXPPNIUDYJJSPCQEKFMM
