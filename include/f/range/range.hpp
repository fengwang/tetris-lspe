#ifndef TRCQLGHQOYIFVAIUFMXNUWPITKLXLWDGVVXYMHUFMSQPOFJAWTLUIRVMVGRMTAUIMONGLCLFP
#define TRCQLGHQOYIFVAIUFMXNUWPITKLXLWDGVVXYMHUFMSQPOFJAWTLUIRVMVGRMTAUIMONGLCLFP

#include <iterator>
#include <cstddef>
#include <functional>

namespace f
{

    namespace range_detail
    {
        template <typename T>
        struct range
        {
            typedef T                                                 value_type;
            typedef range                                             self_type;
            typedef std::function<value_type( value_type )>           step_function_type;
            typedef std::function<bool( value_type, value_type )>     equality_check_function_type;

            value_type                                                first;
            value_type                                                last;
            step_function_type                                        step_function;
            equality_check_function_type                              equality_check_function;

            struct iterator;

            //[first, last)
            template<typename Step_Function, typename Equality_Check_Function>
            range( value_type const& first_, value_type const& last_, Step_Function const& sf_, Equality_Check_Function const& ecf_ ) : first( first_ ), last( last_ ), step_function( sf_ ), equality_check_function( ecf_ ) {}

            template<typename Step_Function>
            range( value_type const& first_, value_type const& last_, Step_Function const& sf_ ) : first( first_ ), last( last_ ), step_function( sf_ ),  equality_check_function( []( value_type const& lhs, value_type const& rhs ) { return lhs == rhs || lhs > rhs; } ) {}

            range( value_type const& first_, value_type const& last_ ) : first( first_ ), last( last_ ), step_function( []( value_type x ) { return ++x; } ), equality_check_function( []( value_type const& lhs, value_type const& rhs ) { return lhs == rhs || lhs > rhs; } ) {}

            iterator begin() const
            {
                return iterator{ first, step_function, equality_check_function };
            }

            iterator end() const
            {
                return iterator{ last, step_function, equality_check_function };
            }

        };//struct range

        template <typename T >
        struct range<T>::iterator
        {
            typedef iterator                                            self_type;
            typedef T                                                   value_type;
            typedef void                                                pointer;
            typedef void                                                reference;
            typedef std::size_t                                         size_type;
            typedef std::ptrdiff_t                                      difference_type;
            typedef std::input_iterator_tag                             iterator_category;
            typedef std::function<value_type( value_type )>             step_function_type;
            typedef std::function<bool( value_type, value_type )>       equality_check_function_type;

            value_type                                                  value;
            step_function_type                                          step_function;
            equality_check_function_type                                equality_check_function;

            iterator( value_type const& value_, step_function_type const& step_function_, equality_check_function_type const& equality_check_function_ ) : value( value_ ), step_function( step_function_ ), equality_check_function( equality_check_function_ ) {}

            value_type operator *() const
            {
                return value;
            }

            self_type& operator ++()
            {
                value = step_function(value);
                return *this;
            }

            self_type const operator ++(int)
            {
                self_type ans{*this};
                ++(*this);
                return ans;
            }

            self_type& operator +=( unsigned long  n )
            {
                while ( n--  )
                    ++(*this);
                return *this;
            }

            friend self_type const operator + ( self_type const& lhs, unsigned long rhs )
            {
                self_type ans{ lhs };
                ans += rhs;
                return ans;
            }

            friend self_type const operator + ( unsigned long lhs, self_type const& rhs )
            {
                return rhs + lhs;
            }

            friend bool operator == ( self_type const& lhs, self_type const& rhs )
            {
                return  lhs.equality_check_function( lhs.value, rhs.value );
            }

            friend bool operator != ( self_type const& lhs, self_type const& rhs )
            {
                return !( lhs == rhs );
            }
        };//struct iterator

        //make_range( first, last, stepper, equality_check );

        template< typename T, typename Step_Function, typename Equality_Check_Function >
        range<T> const make_range( T const& first_, T const& last_, Step_Function const& sf_, Equality_Check_Function const& ecf_ )
        {
            return range<T>{ first_, last_, sf_, ecf_ };
        }

        template< typename T, typename Step_Function >
        range<T> const make_range( T const& first_, T const& last_, Step_Function const& sf_ )
        {
            return range<T>{ first_, last_, sf_ };
        }

        template< typename T >
        range<T> const make_range( T const& first_, T const& last_ )
        {
            return range<T>{ first_, last_ };
        }

    }//namespace range_detail

    template< typename T >
    auto range( T first, T step, T last )
    {
        if ( step > T{0} )
            return range_detail::make_range( first, last, [step]( T x ){ return x+step; } );

        return range_detail::make_range( first, last, [step]( T x ){ return x+step; }, []( T lhs, T rhs ){ return lhs == rhs || lhs < rhs; } );
    }

    template< typename T >
    auto range( T first, T last )
    {
        return range( first, T{1}, last );
    }

    template< typename T >
    auto range( T last )
    {
        return range( T{0}, last );
    }

    // make_iterator //
    //for_each( make_iterator{}, make_iterator{}, containter.begin(), ..., function );

}//namespace f

#endif//TRCQLGHQOYIFVAIUFMXNUWPITKLXLWDGVVXYMHUFMSQPOFJAWTLUIRVMVGRMTAUIMONGLCLFP

