#ifndef MTIME_MANAGER_HPP_INCLUDED_SDOFJ3498YASFOIUHASFO8UH4O87AGHSFDIUAHSFHTUAWIFOUHASFIUHASFIUHSFI
#define MTIME_MANAGER_HPP_INCLUDED_SDOFJ3498YASFOIUHASFO8UH4O87AGHSFDIUAHSFHTUAWIFOUHASFIUHASFIUHSFI

#include <chrono>
#include <cstdint>
#include <cstddef>

namespace f
{

    namespace ga
    {

        //usage:
        // auto& tm = singleton<time_manager>::instance();
        // tm.set_planning_time( 978987987 ); //within ga manager
        // auto et = tm.elapse_time();
        // auto pt = tm.planning_time();
        // if ( tm.is_timeout() ) ...
        struct time_manager
        {
            typedef std::size_t                                 size_type;

            std::chrono::time_point<std::chrono::system_clock>  start;
            size_type                                           planning_time_in_second;
            size_type                                           elapse_time_in_second;

            time_manager( const size_type planning_time_in_second_ = 1000 ) :
                start( std::chrono::system_clock::now() ),
                planning_time_in_second( planning_time_in_second_ ),
                elapse_time_in_second( 0 )
            {}

            void operator()( const size_type planning_time_in_second_ )
            {
                set_planning_time( planning_time_in_second_ );
            }

            //set planning time to another value
            void set_planning_time( const size_type planning_time_in_second_ = 1000 )
            {
                planning_time_in_second = planning_time_in_second_;
                elapse_time_in_second = 0;
            }

            //return time elapsed in second
            size_type elapse_time()
            {
                auto const current_time = std::chrono::system_clock::now();
                elapse_time_in_second = std::chrono::duration_cast<std::chrono::seconds> ( current_time - start ).count();
                return elapse_time_in_second;
            }

            size_type planning_time() const
            {
                return planning_time_in_second;
            }

            bool is_timeout()
            {
                //if ( !(elapse_time_in_second & 0x7f) )
                //    std::cerr << "\ntime manger: " << elapse_time_in_second << " of " << planning_time_in_second;
                return elapse_time() > planning_time_in_second;
            }

        };//time_manager

    }//namespace ga

}//namespace f

#endif//_TIME_MANAGER_HPP_INCLUDED_SDOFJ3498YASFOIUHASFO8UH4O87AGHSFDIUAHSFHTUAWIFOUHASFIUHASFIUHSFI

