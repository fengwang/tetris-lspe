#ifndef MTYPE_AT_HPP_INCLUDED_DSPOFIHSADLFKJASFDJKLAS9483IASDFUHAKSDFJHDFSJVKJNF
#define MTYPE_AT_HPP_INCLUDED_DSPOFIHSADLFKJASFDJKLAS9483IASDFUHAKSDFJHDFSJVKJNF

#include <cstddef>

namespace f
{

    template< std::size_t N, typename T, typename... Types >
    struct type_at
    {
        static_assert( N < sizeof...(Types)+1, "dim size exceeds limitation!" );
        typedef typename type_at<N-1, Types...>::result_type result_type;
    };

    template<typename T, typename... Types>
    struct type_at< 0, T, Types...>
    {
        typedef T result_type;
    };

}//namespace f

#endif//_TYPE_AT_HPP_INCLUDED_DSPOFIHSADLFKJASFDJKLAS9483IASDFUHAKSDFJHDFSJVKJNF

