#ifndef BNECSIFBTMYBXJXHMXDNSSVFOBUEBFJMVUYGRBAOSVXTXIDHKCJVASAORJCLPDBFSDDEKODFN
#define BNECSIFBTMYBXJXHMXDNSSVFOBUEBFJMVUYGRBAOSVXTXIDHKCJVASAORJCLPDBFSDDEKODFN

#include <thread>
#include <vector>
#include <cassert>
#include <algorithm>

namespace f
{

#if 0

Usage:

    std::vector<double> x( 1000000 );

    auto func = [&x]( unsigned long index ){ x[index] = std::exp(x[index]-1.0); };

    parallel_map( func, 0, x.size(), 1, 32 );

#endif 

    template< typename Function >
    void parallel_map( Function func, unsigned long from, unsigned long to, unsigned long stride, unsigned long total_threads ) noexcept
    {
        assert( to >= from );
        assert( stride );

        if (  total_threads <= 1 )
        {
            for ( unsigned long index = 0; index < to; index += stride )
                func( index );
        }

        std::vector<std::thread> v_threads;
        unsigned long const total_entry = ( to - from + stride - 1 ) / stride;
        unsigned long const entry_per_thread = ( total_entry + total_threads - 1 ) / total_threads;

        unsigned long begin = from;
        unsigned long end = from + entry_per_thread * stride;

        auto working_function = [&]()
        {
            for ( unsigned long index = begin; index < end; index += stride )
                func( index );
        };

        for ( unsigned long index = 0; index != total_threads - 1; ++index )
        {
            v_threads.push_back( std::thread( working_function ) );
            begin = end;
            end += entry_per_thread * stride;
        }

        working_function();
        
        for ( auto& the_thread : v_threads )
            the_thread.join();
    }

}//namespace f

#endif//BNECSIFBTMYBXJXHMXDNSSVFOBUEBFJMVUYGRBAOSVXTXIDHKCJVASAORJCLPDBFSDDEKODFN

