#ifndef FNODSFABEHCCPHOCKMEDIIQRRSFMPGIOVHENSXENHTWYXRDHOEPGNKEQQHYELBVQYIEQJWYDW
#define FNODSFABEHCCPHOCKMEDIIQRRSFMPGIOVHENSXENHTWYXRDHOEPGNKEQQHYELBVQYIEQJWYDW

#include <f/window/crtp/window_crtp.hpp>

#include <cstddef>

namespace f
{

    template< typename T >
    struct rectangle_window : window_crtp< rectangle_window< T >, T >
    {
        typedef T               value_type;    
        typedef std::size_t     size_type;
        typedef window_crtp< rectangle_window< T >, T > host_type;

        rectangle_window( size_type n ) : host_type( n ) {}

        value_type impl( const size_type ) const
        {
            return value_type(1.0);
        }

    };//struct rectangle_window

}//namespace f

#endif//FNODSFABEHCCPHOCKMEDIIQRRSFMPGIOVHENSXENHTWYXRDHOEPGNKEQQHYELBVQYIEQJWYDW
