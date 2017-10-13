#ifndef TFSANIUKVQKTWBPUPELEVCLBLJKLBSQIEANWLWOBHMAONPBVQLQKINAKBSLPWWUUIXSHMATTN
#define TFSANIUKVQKTWBPUPELEVCLBLJKLBSQIEANWLWOBHMAONPBVQLQKINAKBSLPWWUUIXSHMATTN

#include <cstddef>
#include <functional>
#include <iterator>

namespace f
{
    
    template< typename Window, typename T >
    struct window_crtp
    {
        typedef Window                          zen_type;
        typedef T                               value_type;
        //typedef typename zen_type::value_type   value_type;
        typedef std::size_t                     size_type;
        typedef std::function<value_type(size_type)>    function_type;

        size_type size_;

        window_crtp( const size_type n ) : size_( n ) {}

        size_type size() const 
        {
            return size_;
        }

        struct iterator
        {
            typedef iterator                                self_type;
            typedef window_crtp::value_type                 value_type;
            typedef std::size_t                             size_type;
            typedef std::input_iterator_tag                 iterator_category;
            typedef std::ptrdiff_t                          difference_type;
            typedef void                                    pointer;
            typedef void                                    reference;
            typedef std::function<value_type(size_type)>    function_type;

            size_type pos_;
            function_type f_;

            iterator( const size_type position, const function_type& f ) : pos_( position ), f_( f ) {}

            iterator( const self_type& other ) = default;
            iterator( self_type&& other ) = default;
            self_type& operator = ( const self_type& other ) = default;
            self_type& operator = ( self_type&& other ) = default;

            value_type operator* () const 
            {
                return f_(pos_);
            }

            self_type& operator ++() 
            {
                ++pos_;
                return *this;
            }

            const self_type operator ++(int)
            {
                self_type ans( *this );
                ++(*this);
                return ans;
            }

            self_type& operator --()
            {
                --pos_;
                return *this;
            }

            const self_type operator --(int)
            {
                self_type ans( *this );
                --( *this );
                return ans;
            }

            friend const self_type operator + ( const self_type& lhs, const size_type n )
            {
                const self_type ans( lhs.pos_ + n, lhs.f_ );
                return ans;
            }

            friend const self_type operator + ( const size_type n, const self_type& rhs )
            {
                return rhs + n;
            }

            friend const self_type operator - ( const self_type& lhs, const size_type n )
            {
                const self_type ans( lhs.pos_ - n, lhs.f_ );
                return ans;
            }

            friend bool operator == ( const self_type& lhs, const self_type& rhs )
            {
                return lhs.pos_ == rhs.pos_;//&& lhs.f_ == rhs.f_;
            }

            friend bool operator != ( const self_type& lhs, const self_type& rhs )
            {
                return lhs.pos_ != rhs.pos_;//|| lhs.f_ != rhs.f_;
            }
        };

        typedef std::reverse_iterator<iterator>     reverse_iterator;

        iterator begin() const
        {
            auto const& zen = static_cast<zen_type const&>(*this);
            auto const& f = [&zen]( size_type n ) { return zen.impl(n); };
            return iterator( 0, f );
            //return iterator( 0, std::bind( &zen_type::impl, zen ) );
        }

        iterator cbegin() const
        {
            return begin();
        }

        iterator end() const
        {
            auto const& zen = static_cast<zen_type const&>(*this);
            //function_type const& f = std::bind( &zen_type::impl, zen );
            auto const& f = [&zen]( size_type n ) { return zen.impl(n); };
            return iterator( size(), f );
        }

        iterator cend() const
        {
            return end();
        }

        reverse_iterator rbegin() const
        {
            return reverse_iterator( end() );
        }

        reverse_iterator crbegin() const
        {
            return rbegin();
        }

        reverse_iterator rend() const
        {
            return reverse_iterator( begin() );
        }

        reverse_iterator crend() const
        {
            return rend();
        }

        value_type operator[]( const size_type i ) const
        {
            return static_cast<zen_type const&>(*this).impl( i );
        }
        
    };

    /*
        template< typename T >
        struct Hamming : window_crtp< Hamming< T > >
        {
            typedef T value_type;
            typedef std::size_t total;

            Hamming( const std::size_t n ) : total( n ) {}

            value_type impl( const size_type index ) const
            {}
        };
     */

}//namespace f

#endif//TFSANIUKVQKTWBPUPELEVCLBLJKLBSQIEANWLWOBHMAONPBVQLQKINAKBSLPWWUUIXSHMATTN
