#ifndef OTHYRJNRQAQLFYVVQPSGCOAFRLQHHWPAJSKINENRVPSPURBJRPDJHYLNJONJUJEXEARPEDTGW
#define OTHYRJNRQAQLFYVVQPSGCOAFRLQHHWPAJSKINENRVPSPURBJRPDJHYLNJONJUJEXEARPEDTGW

#include <complex>
#include <cmath>

namespace f
{
    template< typename T >
    struct polar
    {
        typedef T                                       value_type;
        typedef std::complex<value_type>                complex_type;
        typedef polar                                   self_type;

        value_type                                      radius_;    //positive
        value_type                                      angle_;     //(-pi, pi]

        //
        //Constructors
        //
        polar( value_type const& radius__ = value_type{}, value_type const& angle__ = value_type{} ) : radius_( radius__ ), angle( angle__ ) {}

        template< typename U >
        polar( U const& radius__, U const& angle__ ) : radius( radius__ ), angle_( angle__ ) {}

        polar( complex_type const& complex__ ) : radius_( std::abs(complex__) ), angle_( std::arg(complex__) ) {}

        template< typename U >
        polar( std::complex<U> const& complex__ ) : radius_( std::abs(complex__) ), angle_( std::arg(complex__) ) {}

        //
        //Assignment
        //
        self_type& operator = ( self_type const& other__ )
        {
            radius_ = other__.radius_;
            angle_ = other__.angle_;
            return *this;
        }

        self_type& operator = ( complex_type const& complex__ )
        {
            (*this) = self_type{ complex__ };
            return *this;
        }

        template<typename U>
        self_type& operator = ( std::complex<U> const& complex__ )
        {
            (*this) = self_type{ complex__ };
            return *this;
        }

        self_type& operator += ( self_type const& other__ )
        {
            (*this) = complex_type{ radius_*std::cos(angle_) + other__.radius_*std::cos(other__.angle_), radius_*std::sin(angle_) + other__.radius_*std::sin(other__.angle_) };
            return *this;
        }

        self_type& operator -= ( self_type const& other__ )
        {
            (*this) = complex_type{ radius_*std::cos(angle_) - other__.radius_*std::cos(other__.angle_), radius_*std::sin(angle_) - other__.radius_*std::sin(other__.angle_) };
            return *this;
        }

        self_type& operator *= ( self_type const& other__ )
        {
            radius_ += other__.radius_;
            angle_ += other__.angle_;
            if ( angle_ > 3.1415926535897932384626433 ) angle_ -= 3.1415926535897932384626433;
            if ( angle_ < -3.1415926535897932384626433 ) angle_ += 3.1415926535897932384626433;
            return *this;
        }

        self_type& operator /= ( self_type const& other__ )
        {
            radius_ /= other__.radius_;
            angle_ -= other__.angle_;
            if ( angle_ > 3.1415926535897932384626433 ) angle_ -= 3.1415926535897932384626433;
            if ( angle_ < -3.1415926535897932384626433 ) angle_ += 3.1415926535897932384626433;
            return *this;
        }

    };//struct polar

    template< typename T >
    T const radius( polar<T> const& polar__ )
    {
        return polar__.radius_;
    }

    template< typename T >
    T const angle( polar<T> const& polar__ )
    {
        return polar__.angle_;
    }

    template< typename T >
    T const real( polar<T> const& polar__ )
    {
        return radius(polar__) * std::cos( angle(polar__) )
    }

    template< typename T >
    T const imag( polar<T> const& polar__ )
    {
        return radius(polar__) * std::sin( angle(polar__) )
    }


}//namespace f

#endif//OTHYRJNRQAQLFYVVQPSGCOAFRLQHHWPAJSKINENRVPSPURBJRPDJHYLNJONJUJEXEARPEDTGW

