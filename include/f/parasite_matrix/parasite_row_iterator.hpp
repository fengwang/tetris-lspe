#ifndef RFLKOCSSPVNBGFMOEEFDQQREDJQQPRGYCSBJXIESNXOBWPEAREPAFOGSDSIFEJNVIOLEIWKEM
#define RFLKOCSSPVNBGFMOEEFDQQREDJQQPRGYCSBJXIESNXOBWPEAREPAFOGSDSIFEJNVIOLEIWKEM

#include <f/stride_iterator/stride_iterator.hpp>
#include <iterator>

namespace f
{
    template<typename Matrix, typename T>
    struct parasite_row_iterator
    {
        typedef Matrix                                                          zen_type;
        typedef T                                                               value_type;
        typedef unsigned long                                                   size_type;
        typedef value_type*                                                     pointer;
        typedef stride_iterator<pointer>                                        row_type;
        typedef stride_iterator<const value_type*>                              const_row_type;
        typedef std::reverse_iterator<row_type>                                 reverse_row_type;
        typedef std::reverse_iterator<const_row_type>                           const_reverse_row_type;

        row_type row_begin( const size_type index = 0 )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return row_type( zen.data + index * zen.col_stride, zen.row_stride );
        }

        row_type row_end( const size_type index = 0 )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            return row_begin( index ) + zen.col;
        }

        const_row_type row_begin( const size_type index = 0 ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_row_type( zen.data + index * zen.col_stride, zen.row_stride );
        }

        const_row_type row_end( const size_type index = 0 ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return row_begin( index ) + zen.col;
        }

        const_row_type row_cbegin( const size_type index = 0 ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return const_row_type( zen.data + index * zen.col_stride, zen.row_stride);
        }

        const_row_type row_cend( const size_type index = 0 ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return row_begin( index ) + zen.col;
        }

        reverse_row_type row_rbegin( const size_type index = 0 )
        {
            return reverse_row_type( row_end( index ) );
        }

        reverse_row_type row_rend( const size_type index = 0 )
        {
            return reverse_row_type( row_begin( index ) );
        }

        const_reverse_row_type row_rbegin( const size_type index = 0 ) const
        {
            return const_reverse_row_type( row_end( index ) );
        }

        const_reverse_row_type row_rend( const size_type index = 0 ) const
        {
            return const_reverse_row_type( row_begin( index ) );
        }

        const_reverse_row_type row_crbegin( const size_type index = 0 ) const
        {
            return const_reverse_row_type( row_end( index ) );
        }

        const_reverse_row_type row_crend( const size_type index = 0 ) const
        {
            return const_reverse_row_type( row_begin( index ) );
        }

    };//struct

}

#endif//RFLKOCSSPVNBGFMOEEFDQQREDJQQPRGYCSBJXIESNXOBWPEAREPAFOGSDSIFEJNVIOLEIWKEM

