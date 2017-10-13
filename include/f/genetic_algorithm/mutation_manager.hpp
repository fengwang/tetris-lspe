#ifndef MMUATATION_MANAGER_HPP_INCLUDED_SDOI4E98UAFSKLJSAFOIHJASFKJHASFKHIUYHASFKJHYUHSFDKJHAIUHRFSD
#define MMUATATION_MANAGER_HPP_INCLUDED_SDOI4E98UAFSKLJSAFOIHJASFKJHASFKHIUYHASFKJHYUHSFDKJHAIUHRFSD

#include <f/variate_generator/variate_generator.hpp>

#include <f/singleton/singleton.hpp>
#include <f/genetic_algorithm/gray.hpp>

#include <cstddef>
#include <cstdint>

namespace f
{
    namespace ga
    {
        // usage:
        //          auto& mm = singleton<mutation_manager>::instance();
        //          mm.initialize( 1234 );
        //          std::size_t the_mutation_chromosome_id = mm();
        struct mutation_manager
        {
            std::size_t n;

            mutation_manager( const std::size_t n_ = 196 ) : n( n_ ) {}

            void initialize( const std::size_t n_ )
            {
                n = n_;
            }

            std::size_t operator()() const
            {
                auto& vg = singleton<variate_generator<double>>::instance();
                return static_cast<std::size_t>(vg() * n);
            }

            std::size_t operator()( const std::size_t n_ ) const
            {
                auto& vg = singleton<variate_generator<double>>::instance();
                return static_cast<std::size_t>(vg() * n_);
            }
        };//struct mutation_manager

        struct binary_mutation
        {
            typedef unsigned long uint_type;
            // 1) select a bit randomly
            // 2) flip the bit
            void operator()( uint_type& u ) const
            {
                auto& vg = singleton<variate_generator<double>>::instance();
                uint_type const total_pos = sizeof( uint_type ) << 3;
                // 1)
                uint_type const mask_pos = static_cast<uint_type>(total_pos * vg());
                uint_type const mask = 1 << mask_pos;
                u = ulong_to_gray()( u );
                // 2)
                u ^= mask;
                u = gray_to_ulong()( u );
            }
        };//struct binary_mutation

    }//namespace ga

}//namespace f

#endif//_MUATATION_MANAGER_HPP_INCLUDED_SDOI4E98UAFSKLJSAFOIHJASFKJHASFKHIUYHASFKJHYUHSFDKJHAIUHRFSD

