#ifndef MGRAY_TO_INT64_HPP_INCLUDED_DSOFIJR98AFSIHJ98R3YAOWI3498YAOISFUHASKFDHJASIFUH3HU7AIUHFDKJVKJ
#define MGRAY_TO_INT64_HPP_INCLUDED_DSOFIJR98AFSIHJ98R3YAOWI3498YAOISFUHASKFDHJASIFUH3HU7AIUHFDKJVKJ

#include <cstdint>

namespace f
{

    namespace ga
    {

        // usage:
        //          ulong_t n;
        //          auto i = gray_to_int_64()(n); //convert n to normal int
        //
        //          matrix<ulong_t> m;
        //          feng::for_each( m.begin(), m.end(), [](ulong_t& v){ v = gray_to_ulong()(v); } );
        //          std::transform( m.begin(), m.end(), m.begin(), gray_to_ulong() );
        struct gray_to_ulong
        {
            typedef unsigned long value_type;

            value_type operator()( const value_type v ) const
            {
                value_type num = v;
                unsigned int const numBits = 64;
                for ( unsigned int shift = 1; shift < numBits; shift *= 2 )
                    num ^= num >> shift;
                return num;
            }

        };//gray_to_ulong
    };//namespace ga

    namespace ga
    {
        // usage:
        //      ulong_t n;
        //      auto g = ulong_to_gray()(n); //convert n to gray code
        struct ulong_to_gray
        {
            typedef unsigned long value_type;

            value_type operator()( const value_type v ) const
            {
                return v ^ ( v >> 1 );
            }

        };//ulong_to_gray
    };//namespace ga

}//namespace f

#endif//_INT64_TO_GRAY_HPP_INCLUDED_FSOIJ3498YUAOFSIUHOUHWAFI8HUEFOHU8ASDVKJBVDIUHAWFGIUYWEFHIUASFIU

