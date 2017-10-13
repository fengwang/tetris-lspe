#ifndef NHEOFATBTBPTFKEIWRGYDOGFGSVPSFJDSDALOTVMITEDAJPAUHAFVQQJOBNTKAEGSUOBJTSVV
#define NHEOFATBTBPTFKEIWRGYDOGFGSVPSFJDSDALOTVMITEDAJPAUHAFVQQJOBNTKAEGSUOBJTSVV

#include <f/stride_iterator/stride_iterator.hpp>

namespace f
{
    template<typename Matrix, typename T>
    struct parasite_bracket_operator
    {
        typedef Matrix                                                          zen_type;
        typedef unsigned long                                                   size_type;
        typedef T                                                               value_type;
        typedef value_type*                                                     pointer;
        typedef stride_iterator<pointer>                                        row_type;
        typedef stride_iterator<const pointer>                                  const_row_type;

        row_type operator[]( const size_type index )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return zen.row_begin( index );
        }

        const_row_type operator[]( const size_type index ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return zen.row_begin( index );
        }

        value_type& operator()( const size_type r = 0, const size_type c = 0 )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return *( zen.row_begin( r ) + c );
        }

        value_type operator()( const size_type r = 0, const size_type c = 0 ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return *( zen.row_cbegin( r ) + c );
        }
    };

}

#endif//NHEOFATBTBPTFKEIWRGYDOGFGSVPSFJDSDALOTVMITEDAJPAUHAFVQQJOBNTKAEGSUOBJTSVV

