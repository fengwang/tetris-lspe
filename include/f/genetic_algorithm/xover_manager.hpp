#ifndef MXOVER_MANAGER_HPP_INCLUDED_SDFOIJROTIHASFO8498YASFIUHVBKJ9348T7YHSDKGJN34987UHSDGKJNBIUH489
#define MXOVER_MANAGER_HPP_INCLUDED_SDFOIJROTIHASFO8498YASFIUHVBKJ9348T7YHSDKGJN34987UHSDGKJNBIUH489

#include <f/genetic_algorithm/gray.hpp>
#include <f/singleton/singleton.hpp>
#include <f/variate_generator/variate_generator.hpp>

#include <cmath>
#include <vector>
#include <cstdint>

namespace f
{

    namespace ga
    {
        //      returns an id that is selected
        // usage:
        //      auto& cs = singleton<xover_selection_manager>::instance();
        //      cs.initialize(1080); //should only in ga manager
        //      auto const selected_chromosome = cs();
        struct xover_selection_manager
        {
            std::vector<double> weigh_array;

            xover_selection_manager( const std::size_t n = 196 )
            {
                initialize( n );
            }

            void initialize( const std::size_t n = 196 )
            {
                weigh_array.resize( n );
                const double alpha = std::log( n ) / ( n + n - 2 );
                const double factor = std::exp( -alpha );
                double current = factor;
                for ( std::size_t i = 0; i != n; ++i )
                {
                    weigh_array[i] = current;//exp(-(i+1)\alpha)
                    current *= factor;
                }
                // weigh array elements should look like this
                // { [a], [a+a^2], ..., [a+a^2+...+a^N] }
                for ( std::size_t i = 1; i != n; ++i )
                {
                    weigh_array[i] += weigh_array[i - 1];
                }
                auto const acc = *( weigh_array.rbegin() );
                // weigh array elements should look like this
                // { [x], [xx], ..., [1] }
                std::for_each( weigh_array.begin(), weigh_array.end(), [acc]( double & v )
                {
                    v /= acc;
                } );
            }

            std::size_t operator()() const
            {
                // a random number U[0,1]
                auto& vg = singleton<variate_generator<double>>::instance();
                auto const p = vg();
                return std::distance( weigh_array.begin(), std::upper_bound( weigh_array.begin(), weigh_array.end(), p ) );
            }
        };//struct xover_manager


        struct uniform_binary_xover
        {
            typedef uniform_binary_xover self_type;
            typedef unsigned long uint_type;
            void operator()( const uint_type father, const uint_type mother, uint_type& son, uint_type& daughter ) const
            {
                uint_type const f = ulong_to_gray()( father );
                uint_type const m = ulong_to_gray()( mother );
                uint_type const lower_mask = 0x5555555555555555UL;
                uint_type const upper_mask = 0xaaaaaaaaaaaaaaaaUL;
                son      = gray_to_ulong()( ( f & lower_mask ) | ( m & upper_mask ) );
                daughter = gray_to_ulong()( ( m & lower_mask ) | ( f & upper_mask ) );
            }

            void operator()( uint_type& father, uint_type& mother ) const
            {
                uint_type son, daughter;
                self_type()( father, mother, son, daughter );
                father = son;
                mother = daughter;
            }
        };//struct binary_xover

        struct single_point_xover
        {
            typedef single_point_xover self_type;
            void operator()( unsigned long const father, unsigned long const mother, unsigned long& s, unsigned long& d ) const
            {
                const std::size_t length = sizeof( unsigned long ) << 3;
                auto& vg = singleton<variate_generator<double>>::instance();
                const std::size_t right_pos = static_cast<std::size_t>(vg() * length);
                const std::size_t left_pos = length - right_pos;
                unsigned long const f = ulong_to_gray()( father );
                unsigned long const m = ulong_to_gray()( mother );
                s = ( ( f >> right_pos ) << right_pos ) | ( ( m << left_pos ) >> left_pos );
                d = ( ( m >> right_pos ) << right_pos ) | ( ( f << left_pos ) >> left_pos );
                if ( f == m )
                {
                    s = f;
                    //const unsigned long bitmask = (unsigned long)( 1 ) << right_pos;
                    const unsigned long bitmask = 1ULL < right_pos;
                    d = f ^ bitmask;
                }
                s = gray_to_ulong()( s );
                d = gray_to_ulong()( d );
            }

            void operator()( unsigned long& father, unsigned long& mother ) const
            {
                unsigned long son, daughter;
                self_type()( father, mother, son, daughter );
                father = son;
                mother = daughter;
            }

        };//struct single_point_xover

    }//namespace ga

};//namespace f

#endif//_XOVER_MANAGER_HPP_INCLUDED_SDFOIJROTIHASFO8498YASFIUHVBKJ9348T7YHSDKGJN34987UHSDGKJNBIUH489
