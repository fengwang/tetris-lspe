#ifndef HOMTFKWXMCEYIIXBYOWJJNTXKIEQGXJAOEJMUMJIXUWRTACTUKXGKGIHAIJSOJNEEICPNWXES
#define HOMTFKWXMCEYIIXBYOWJJNTXKIEQGXJAOEJMUMJIXUWRTACTUKXGKGIHAIJSOJNEEICPNWXES

#include <thread>
#include <vector>
#include <cassert>
#include <algorithm>

namespace f
{

    namespace parallel_private
    {
        template< unsigned long N >
        struct parallel_impler;

        template<>
        struct parallel_impler<1>
        {
            std::thread the_thread;

            template< typename Function, typename List >
            parallel_impler( Function function, List list ) noexcept : the_thread{ [function, list]() noexcept { list(function); } } {} 

            ~parallel_impler(){ the_thread.join(); }
        };

        template< unsigned long N >
        struct parallel_impler
        {
            parallel_impler<N-1> prev_impler;
            std::thread the_thread;

            template< typename Function, typename List, typename ... Lists >
            parallel_impler( Function function, List list, Lists ... lists ) noexcept : prev_impler{ function, lists... },  the_thread{ [function, list]() noexcept { list(function); } } {} 

            ~parallel_impler(){ the_thread.join(); }
        };
    }

    template< typename... Args >
    auto make_arg( Args... args ) noexcept
    {
        return [=]( auto func ) noexcept { return func( args... ); }; 
    }

    template< typename Function >
    void make_parallel( Function ) noexcept
    {
    }

    //cons:
    //      not inlinely expanded during compliation
    template< typename Function, typename List, typename... Lists >
    void make_parallel( Function func, List list, Lists... lists ) noexcept
    {
#if 1
        std::thread the_thread{ [=]() noexcept { list(func); } };
        make_parallel( func, lists... );
        the_thread.join();
#else
        parallel_private::parallel_impler<1+sizeof...(lists)> pi{ func, list, lists... };
#endif
    }



#if 0
    proposed usage:

        std::vector< int > vi( 4096 );
        std::vector< int > sum( 4 );

        make_parallel( []( auto result_itor, auto begin_itor, auto end_itor ) { *result_itor = std::accumulate( begin_itor, end_itor, 0 ); },
                       make_arg( sum.begin()+0, vi.begin()+0, vi.begin()+1024 ), 
                       make_arg( sum.begin()+1, vi.begin()+1024, vi.begin()+2048 ), 
                       make_arg( sum.begin()+2, vi.begin()+2048, vi.begin()+3072 ), 
                       make_arg( sum.begin()+3, vi.begin()+3072, vi.begin()+4096 )  
                     );

        int total = sum[0] + sum[1] + sum[2] + sum[3];

#endif

    template< typename Function >
    void parallel( Function func, unsigned long from, unsigned long to, unsigned long stride, unsigned long total_threads ) noexcept
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

        //main thread
        begin = end;
        end += entry_per_thread * stride;
        working_function();
        
        for ( auto& the_thread : v_threads )
            the_thread.join();
    }

#if 0

    std::vector<double> x( 1000000 );

    auto func = [&x]( unsigned long index ){ x[index] = std::exp(x[index]-1.0); };

    parallel( func, 0, x.size(), 1, 32 );

#endif 

}//namespace f

#endif//HOMTFKWXMCEYIIXBYOWJJNTXKIEQGXJAOEJMUMJIXUWRTACTUKXGKGIHAIJSOJNEEICPNWXES

