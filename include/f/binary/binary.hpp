#ifndef MBINARY_HPP_INCLUDED_FSDOIJAPOI4EHP89ASFHIOASFHOUASFDKADFHUIOUHASFDKLJSDFAHSADFHJFDIHFDJFFFFOSIJFSDDD
#define MBINARY_HPP_INCLUDED_FSDOIJAPOI4EHP89ASFHIOASFHOUASFDKADFHUIOUHASFDKLJSDFAHSADFHJFDIHFDJFFFFOSIJFSDDD

namespace binary_private_afdsiuoh4879yasfdkhjlawf89y74rliuhasf8ghsf
{
    typedef long value_type;

    template<char..._0_1_series>
    struct binary_impl;

    template<>
    struct binary_impl<'0'>
    {
        static const value_type value = 0;
    };

    template<>
    struct binary_impl<'1'>
    {
        static const value_type value = 1;
    };

    template<char _0_or_1, char...rest_0_1_series>
    struct binary_impl<_0_or_1, rest_0_1_series...>
    {
        static const value_type value = ( binary_impl<_0_or_1>::value << sizeof...( rest_0_1_series ) ) + binary_impl<rest_0_1_series...>::value;
    };

}//namespace binary_private_afdsiuoh4879yasfdkhjlawf89y74rliuhasf8ghsf

template <char ..._0_1_series>
constexpr binary_private_afdsiuoh4879yasfdkhjlawf89y74rliuhasf8ghsf::value_type operator "" _b()
{
    static_assert ( sizeof...( _0_1_series ) < (sizeof(binary_private_afdsiuoh4879yasfdkhjlawf89y74rliuhasf8ghsf::value_type) << 3), "Error: too long a binary integer series." );
    return binary_private_afdsiuoh4879yasfdkhjlawf89y74rliuhasf8ghsf::binary_impl<_0_1_series...>::value;
}

#endif//_BINARY_HPP_INCLUDED_FSDOIJAPOI4EHP89ASFHIOASFHOUASFDKADFHUIOUHASFDKLJSDFAHSADFHJFDIHFDJFFFFOSIJFSDDD

