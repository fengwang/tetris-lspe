#ifndef LVQONDSFWDOOFEMRQMWHCACSVOABYGGPRMPFYYJQNRBYGTTTDGNYNFFVDDXHDSVGPVMGXSXVJ
#define LVQONDSFWDOOFEMRQMWHCACSVOABYGGPRMPFYYJQNRBYGTTTDGNYNFFVDDXHDSVGPVMGXSXVJ

#include <f/stride_iterator/stride_iterator.hpp>
#include <iterator>

namespace f
{
    template<typename Matrix, typename T>
    struct parasite_col_iterator
    {
        typedef Matrix                                                          zen_type;
        typedef T                                                               value_type;
        typedef unsigned long                                                   size_type;
        typedef value_type*                                                     pointer;
        typedef stride_iterator<pointer>                                        col_type;
        typedef stride_iterator<const value_type*>                                  const_col_type;
        typedef std::reverse_iterator<col_type>                                 reverse_col_type;
        typedef std::reverse_iterator<const_col_type>                           const_reverse_col_type;

        col_type col_begin( const size_type index = 0 )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return col_type( zen.data + index * zen.row_stride, zen.col_stride );
        }

        col_type col_end( const size_type index = 0 )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return col_begin( index ) + zen.row;
        }

        const_col_type col_begin( const size_type index = 0 ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_col_type( zen.data + index * zen.row_stride, zen.col_stride );
        }

        const_col_type col_end( const size_type index = 0 ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return col_begin( index ) + zen.row;
        }

        const_col_type col_cbegin( const size_type index = 0 ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_col_type( zen.data + index * zen.row_stride, zen.col_stride);
        }

        const_col_type col_cend( const size_type index = 0 ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return col_begin( index ) + zen.row;
        }

        reverse_col_type col_rbegin( const size_type index = 0 )
        {
            return reverse_col_type( col_end( index ) );
        }

        reverse_col_type col_rend( const size_type index = 0 )
        {
            return reverse_col_type( col_begin( index ) );
        }

        const_reverse_col_type col_rbegin( const size_type index = 0 ) const
        {
            return const_reverse_col_type( col_end( index ) );
        }

        const_reverse_col_type col_rend( const size_type index = 0 ) const
        {
            return const_reverse_col_type( col_begin( index ) );
        }

        const_reverse_col_type col_crbegin( const size_type index = 0 ) const
        {
            return const_reverse_col_type( col_end( index ) );
        }

        const_reverse_col_type col_crend( const size_type index = 0 ) const
        {
            return const_reverse_col_type( col_begin( index ) );
        }

    };//struct

}

#endif//LVQONDSFWDOOFEMRQMWHCACSVOABYGGPRMPFYYJQNRBYGTTTDGNYNFFVDDXHDSVGPVMGXSXVJ

