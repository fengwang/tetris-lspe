#ifndef MROW_COL_SIZE_HPP_INCLUDED_SDOPNSAL430JAFLKJNADJKNVOSFADOIJH439O8JUHAFFS
#define MROW_COL_SIZE_HPP_INCLUDED_SDOPNSAL430JAFLKJNADJKNVOSFADOIJH439O8JUHAFFS

#include <f/matrix/details/crtp/typedef.hpp>

namespace f
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_row_col_size
    {
        typedef Matrix                                                          zen_type;
        typedef crtp_typedef<Type, Allocator>                                   proxy_type;
        typedef typename proxy_type::size_type                                  size_type;

        size_type row() const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return zen.row_;
        }

        size_type rows() const
        {
            return row();
        }

        size_type size1() const
        {
            return row();
        }

        size_type col() const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            return zen.col_;
        }

        size_type cols() const
        {
            return col();
        }

        size_type size2() const
        {
            return col();
        }

        size_type size() const
        {
            return row() * col();
        }

    };//struct  crtp_row_col_size
}

#endif//_ROW_COL_SIZE_HPP_INCLUDED_SDOPNSAL430JAFLKJNADJKNVOSFADOIJH439O8JUHAFFS

