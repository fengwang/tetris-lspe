#ifndef RNDYREKLOXUDWSSDGKNYHEDDCGJVQOFLNWWDMGJMQQOPSACAIDAXSRAGAGCORDBCGDOHARBFB
#define RNDYREKLOXUDWSSDGKNYHEDDCGJVQOFLNWWDMGJMQQOPSACAIDAXSRAGAGCORDBCGDOHARBFB

#include <array>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <memory>

namespace f
{

    template< typename Iterator, unsigned long Step >
    struct static_matrix_stride_iterator
    {
        typedef typename std::iterator_traits<Iterator>::value_type         value_type;
        typedef typename std::iterator_traits<Iterator>::reference          reference;
        typedef typename std::iterator_traits<Iterator>::difference_type    difference_type;
        typedef typename std::iterator_traits<Iterator>::pointer            pointer;
        typedef std::random_access_iterator_tag                             iterator_category;
        typedef static_matrix_stride_iterator                               self_type;

        Iterator            iterator_;

        static_matrix_stride_iterator() : iterator_( 0 ) {}

        static_matrix_stride_iterator( Iterator it ) : iterator_( it )
        {
        }

        self_type& operator ++()
        {
            iterator_ += Step;
            return *this;
        }

        self_type const operator ++( int )
        {
            self_type ans{ *this };
            (*this).operator++();
            return ans;
        }

        self_type& operator --()
        {
            iterator_ -= Step;
            return *this;
        }

        self_type& operator += ( difference_type dt )
        {
            iterator_ += Step * dt;
            return *this;
        }

        self_type& operator -= ( difference_type dt )
        {
            iterator_ -= Step * dt;
            return *this;
        }

        reference operator[]( difference_type dt )
        {
            return iterator_[dt*Step];
        }

        const reference operator[]( difference_type dt ) const
        {
            return iterator_[dt*Step];
        }

        reference operator *()
        {
            return iterator_[0];
        }

        const reference operator *() const
        {
            return iterator_[0];
        }

        friend bool operator == ( self_type const& lhs, self_type const& rhs )
        {
            return lhs.iterator_ == rhs.iterator_;
        }

        friend bool operator != ( self_type const& lhs, self_type const& rhs )
        {
            return lhs.iterator_ != rhs.iterator_;
        }

        friend bool operator < ( self_type const& lhs, self_type const& rhs )
        {
            return lhs.iterator_ < rhs.iterator_;
        }

        friend bool operator <= ( self_type const& lhs, self_type const& rhs )
        {
            return lhs.iterator_ <= rhs.iterator_;
        }

        friend bool operator > ( self_type const& lhs, self_type const& rhs )
        {
            return lhs.iterator_ > rhs.iterator_;
        }

        friend bool operator >= ( self_type const& lhs, self_type const& rhs )
        {
            return lhs.iterator_ >= rhs.iterator_;
        }

        friend difference_type operator - ( self_type const& lhs, self_type const& rhs )
        {
            return (lhs.iterator_ - rhs.iterator_) / Step;
        }

        friend difference_type operator - ( self_type const& lhs, difference_type rhs )
        {
            self_type ans{ lhs };
            ans -= rhs;
            return ans;
        }

        friend self_type operator + ( self_type const& lhs, difference_type rhs )
        {
            self_type ans{ lhs };
            ans += rhs;
            return ans;
        }

        friend self_type operator + ( difference_type lhs, self_type const& rhs )
        {
            return rhs + lhs;
        }
    };


    template< typename T, unsigned long Row, unsigned long Col >
    struct static_matrix
    {
        typedef T                                                           value_type;
        typedef value_type*                                                 reference;
        typedef const value_type*                                           const_reference;
        typedef value_type*                                                 row_type;
        typedef const value_type*                                           const_row_type;
        typedef static_matrix_stride_iterator<value_type*, Col>             col_type;
        typedef static_matrix_stride_iterator<const value_type*, Col>       const_col_type;
        typedef std::array<value_type, Row*Col>                             storage_type;
        typedef static_matrix                                               self_type;

        storage_type                                storage;

        constexpr unsigned long row() const
        {
            return Row;
        }

        constexpr unsigned long col() const
        {
            return Col;
        }

        auto begin()
        {
            return std::addressof( storage[0] );
            //return storage.begin();
        }

        auto end()
        {
            return std::addressof( storage[Row*Col] );
            //return storage.end();
        }

        auto begin() const
        {
            return std::addressof( storage[0] );
            //return storage.begin();
        }

        auto end() const
        {
            return std::addressof( storage[Row*Col] );
            //return storage.end();
        }

        row_type row_begin( unsigned long index )
        {
            return begin() + index * Col;
        }

        const_row_type row_begin( unsigned long index ) const
        {
            return begin() + index * Col;
        }

        row_type row_end( unsigned long index )
        {
            return begin() + index * Col + Col;
        }

        const_row_type row_end( unsigned long index ) const
        {
            return begin() + index * Col + Col;
        }

        auto row_rbegin( unsigned long index )
        {
            return std::reverse_iterator<decltype(row_begin(index))>{ row_end(index) };
        }

        auto row_rbegin( unsigned long index ) const
        {
            return std::reverse_iterator<decltype(row_begin(index))>{ row_end(index) };
        }

        auto row_rend( unsigned long index )
        {
            return std::reverse_iterator<decltype(row_begin(index))>{ row_begin(index) };
        }

        auto row_rend( unsigned long index ) const
        {
            return std::reverse_iterator<decltype(row_begin(index))>{ row_begin(index) };
        }

        col_type col_begin( unsigned long index )
        {
            return col_type{ begin()+index };
        }

        col_type col_end( unsigned long index )
        {
            return col_begin(index) + Row;
        }

        const_col_type col_begin( unsigned long index ) const
        {
            return const_col_type{ begin()+index };
        }

        const_col_type col_end( unsigned long index ) const
        {
            return const_col_type{ begin()+index+Row*Col };
        }

        auto col_rbegin( unsigned long index )
        {
            return std::reverse_iterator<decltype(col_begin(index))>{ col_end(index) };
        }

        auto col_rbegin( unsigned long index ) const
        {
            return std::reverse_iterator<decltype(col_begin(index))>{ col_end(index) };
        }

        auto col_rend( unsigned long index )
        {
            return std::reverse_iterator<decltype(col_begin(index))>{ col_begin(index) };
        }

        auto col_rend( unsigned long index ) const
        {
            return std::reverse_iterator<decltype(col_begin(index))>{ col_begin(index) };
        }

        reference operator[]( unsigned long index )
        {
            return std::addressof( storage[index*Col] );
            //return row_begin( index );
        }

        const_reference operator[]( unsigned long index ) const
        {
            return std::addressof( storage[index*Col] );
            //return row_begin( index );
        }

        void swap( self_type& other )
        {
            std::swap( storage, other.storage );
        }

        auto data()
        {
            return storage.data();
        }

        auto data() const
        {
            return storage.data();
        }

        constexpr unsigned long size() const
        {
            return Row*Col;
        }

        static_matrix() = default;
        static_matrix( static_matrix const& ) = default;
        static_matrix( static_matrix&& ) = default;
        static_matrix& operator = ( static_matrix const& ) = default;
        static_matrix& operator = ( static_matrix && ) = default;

        friend std::ostream& operator << ( std::ostream& os, static_matrix const& sm )
        {
            for ( unsigned long r = 0; r != sm.row(); ++r )
            {
                for ( unsigned long c = 0; c != sm.col(); ++c )
                    os << sm[r][c] << "\t";
                os << "\n";
            }
            return os;
        }

    };


}//namespace f

#endif//RNDYREKLOXUDWSSDGKNYHEDDCGJVQOFLNWWDMGJMQQOPSACAIDAXSRAGAGCORDBCGDOHARBFB

