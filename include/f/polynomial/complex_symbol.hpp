#ifndef MRYNVWTWFXSLLVWJPCJYVDQHMCBNGEBJYJIDHGHMEWGAAOGFGXBPYRFCSPFSONPRTUELVHCKU
#define MRYNVWTWFXSLLVWJPCJYVDQHMCBNGEBJYJIDHGHMEWGAAOGFGXBPYRFCSPFSONPRTUELVHCKU

#include <f/polynomial/symbol.hpp>

#include <iostream>

namespace f
{
    enum complex_symbol_tag
    {
        radius,
        cosine,
        sine,
        variable
    };

    template< typename T >
    struct complex_symbol : symbol< T, complex_symbol< T > >
    {
         typedef T                                              value_type;
         typedef unsigned long                                  size_type;
         typedef complex_symbol                                 self_type;

         value_type const&                                      ref;
         size_type                                              index;
         complex_symbol_tag                                     tag;

         complex_symbol( value_type const& ref_, size_type index_, complex_symbol_tag tag_ ) : ref( ref_ ), index( index_ ), tag( tag_ ) {}

         complex_symbol( self_type const& other ) : ref( other.ref ), index( other.index ), tag( other.tag ) { }

         value_type eval() const
         {
            return ref;
         }

         friend bool operator < ( self_type const& lhs, self_type const& rhs )
         {
             if ( lhs.index < rhs.index ) return true;
             if ( lhs.index > rhs.index ) return false;
             if ( lhs.tag < rhs.tag ) return true;
             return false;
         }

         friend bool operator == ( self_type const& lhs, self_type const& rhs )
         {
            return lhs.index == rhs.index && lhs.tag == rhs.tag;
         }

         friend std::ostream& operator << ( std::ostream& os, self_type const& rhs )
         {
            os << "(";

            if ( rhs.tag == radius )
                os << "radius";
            if ( rhs.tag == cosine )
                os << "cosine";
            if ( rhs.tag == sine )
                os << "sine";
            if ( rhs.tag == variable )
                os << "variable";

            os << ",";

            return os << rhs.index << ")";
         }
    };

    template<typename T>
    complex_symbol<T> const make_symbol( const T& ref, unsigned long index, complex_symbol_tag tag )
    {
        return complex_symbol<T>{ ref, index, tag };
    }

    //-----------------------------------------------------------------------------------------------
    template<typename T>
    complex_symbol<T> const make_radius_symbol( const T& ref, unsigned long index )
    {
        return make_symbol( ref, index, radius );
    }

    template<typename T>
    complex_symbol<T> const make_cosine_symbol( const T& ref, unsigned long index )
    {
        return make_symbol( ref, index, cosine );
    }

    template<typename T>
    complex_symbol<T> const make_sine_symbol( const T& ref, unsigned long index )
    {
        return make_symbol( ref, index, sine );
    }

    template<typename T>
    complex_symbol<T> const make_variable_symbol( const T& ref, unsigned long index )
    {
        return make_symbol( ref, index, variable );
    }
    //-----------------------------------------------------------------------------------------------

}//namespace f

#endif//MRYNVWTWFXSLLVWJPCJYVDQHMCBNGEBJYJIDHGHMEWGAAOGFGXBPYRFCSPFSONPRTUELVHCKU

