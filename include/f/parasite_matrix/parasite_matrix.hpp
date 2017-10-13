#ifndef XNKDJMDRDXHOSWYLXUKSLSDEYTFSOIASKFMLRYWEEWYHBAUKJOWVDFPFCEQXTETYTKOKVYPMK
#define XNKDJMDRDXHOSWYLXUKSLSDEYTFSOIASKFMLRYWEEWYHBAUKJOWVDFPFCEQXTETYTKOKVYPMK

#include "./parasite_bracket_operator.hpp"
#include "./parasite_col_iterator.hpp"
#include "./parasite_output_operator.hpp"
#include "./parasite_row_iterator.hpp"

namespace f
{
    template< typename T >
    struct parasite_matrix:
        public parasite_bracket_operator<parasite_matrix<T>, T>,
        public parasite_col_iterator<parasite_matrix<T>, T>,
        public parasite_output_operator<parasite_matrix<T>, T>,
        public parasite_row_iterator<parasite_matrix<T>, T>
    {
        typedef T                   value_type;
        typedef unsigned long       size_type;
        typedef value_type*         pointer;

        size_type                   row;
        size_type                   col;
        size_type                   row_stride;
        size_type                   col_stride;
        pointer                     data;

        parasite_matrix( size_type row_, size_type col_, pointer data_ ): row( row_ ), col( col_ ), row_stride(1), col_stride( col_ ), data( data_ ) {}
        parasite_matrix( size_type row_, size_type row_stride_, size_type col_, size_type col_stride_, pointer data_ ): row( row_ ), col( col_ ), row_stride(row_stride_), col_stride( col_stride_ ), data( data_ ) {}
        parasite_matrix( parasite_matrix const& ) = default;
        parasite_matrix( parasite_matrix&& ) = default;
        parasite_matrix& operator = ( parasite_matrix const& ) = default;
        parasite_matrix& operator = ( parasite_matrix && ) = default;
        ~parasite_matrix() = default;
    };

}//namespace f

#endif//XNKDJMDRDXHOSWYLXUKSLSDEYTFSOIASKFMLRYWEEWYHBAUKJOWVDFPFCEQXTETYTKOKVYPMK

