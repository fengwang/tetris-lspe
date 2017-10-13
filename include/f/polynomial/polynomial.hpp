#ifndef QLITBFTBEPRNEEALKYKBTHNNPRAXOQKKLWVPOWJNNXVOKIXMRIMRRAGGAKXKPVQQBTQPSRMWG
#define QLITBFTBEPRNEEALKYKBTHNNPRAXOQKKLWVPOWJNNXVOKIXMRIMRRAGGAKXKPVQQBTQPSRMWG

#include <f/polynomial/term.hpp>

#include <set>
#include <unordered_set>
#include <iostream>

namespace f
{

    //
    //polynomial is a set of terms
    //should take care of '+' '-' and '*' operators
    //with value, symbol, term and polynomial
    //
    template< typename T, typename Symbol_T >
    struct polynomial
    {
        typedef polynomial                                          self_type;
        typedef T                                                   value_type;
        typedef Symbol_T                                            symbol_type;
        typedef term<T, Symbol_T>                                   term_type;
        //typedef std::unordered_set<term_type>                       collection_type;
        typedef std::set<term_type>                                 collection_type;
        typedef typename collection_type::iterator                  iterator;
        typedef typename collection_type::const_iterator            const_iterator;
        typedef typename collection_type::reverse_iterator          reverse_iterator;
        typedef typename collection_type::const_reverse_iterator    const_reverse_iterator;

        collection_type                                             collection;

        polynomial(){}

        polynomial( collection_type const& collection_ ) : collection( collection_ ) {}

        //polynomial( self_type const& self_ ) : collection( self_.collection ) {}

        void trim( value_type v_ = value_type{ 1.0e-3 } )
        {
            collection_type collection_;
            for ( auto const& c_ : collection )
                if ( std::abs(c_.coefficient) > std::abs(v_) )
                    collection_.insert( c_ );
            collection.swap( collection_ );
        }

        value_type eval() const
        {
            value_type ans{};

            for ( auto const& the_term: collection )
                ans += the_term.eval();

            return ans;
        }

        void clear()
        {
            collection.clear();
        }

        self_type const abs() const
        {
            self_type ans;

            for ( auto const& the_term: collection )
                ans += the_term.abs();

            return ans;
        }

        polynomial( term_type const& term_ )
        {
            if ( term_.coefficient != value_type{} )
                collection.insert( term_ );
        }

        polynomial( value_type value_ )
        {
            if ( value_ != value_type{} )
                collection.insert( term_type{ value_ } );
        }

        polynomial( symbol_type const& symbol_ )
        {
            collection.insert( term_type{ symbol_ } );
        }

        self_type const operator - () const
        {
            self_type ans;
            for ( auto const& the_term : collection )
                ans -= the_term;
            return ans;
        }

        self_type& operator += ( term_type const& term_ )
        {
            //return if null term
            if ( value_type{} == term_.coefficient )
                return *this;

            //find duplicated record in collection
            auto itor = collection.find( term_ );
            if ( itor != collection.end() )
            {
                //calculate new term
                auto const new_coefficient = (*itor).coefficient + term_.coefficient;
                //erase old term
                collection.erase( itor );
                //if new term is null, directly return
                if ( value_type{} == new_coefficient )
                    return *this;

                //construct new term, insert and return
                term_type const new_term{ new_coefficient, term_.record };
                collection.insert( new_term );
                return *this;
            }

            //no duplicated record in collection, simply insert and return
            collection.insert( term_ );
            return *this;
        }

        friend self_type const operator + ( self_type const& lhs, term_type const& rhs )
        {
            self_type ans{ lhs };
            ans += rhs;
            return ans;
        }

        friend self_type const operator + ( term_type const& lhs, self_type const& rhs )
        {
            return rhs + lhs;
        }

        self_type& operator += ( value_type const& value_ )
        {
            operator += ( term_type{ value_ } );
            return *this;
        }

        friend self_type const operator + ( self_type const& lhs, value_type const& rhs )
        {
            self_type ans{ lhs };
            ans += rhs;
            return ans;
        }

        friend self_type const operator + ( value_type const& lhs, self_type const& rhs )
        {
            return rhs + lhs;
        }

        self_type& operator += ( symbol_type const& symbol_ )
        {
            operator += ( term_type{ symbol_ } );
            return *this;
        }

        friend self_type const operator + ( self_type const& lhs, symbol_type const& rhs )
        {
            self_type ans{ lhs };
            ans += rhs;
            return ans;
        }

        friend self_type const operator + ( symbol_type const& lhs, self_type const& rhs )
        {
            return rhs + lhs;
        }

        self_type& operator += ( self_type const& self_ )
        {
            for ( auto const& the_term : self_.collection )
                operator += ( the_term );
            return *this;
        }

        friend self_type const operator + ( self_type const& lhs, self_type const& rhs )
        {
            self_type ans{ lhs };
            ans += rhs;
            return ans;
        }

        // operator -
        self_type& operator -= ( value_type const& value_ )
        {
            operator += ( term_type{ -value_ } );
            return *this;
        }

        friend self_type const operator - ( self_type const& lhs, value_type const& rhs )
        {
            self_type ans{ lhs };
            ans -= rhs;
            return ans;
        }

        friend self_type const operator - ( value_type const& lhs, self_type const& rhs )
        {
            return -rhs + lhs;
        }

        self_type& operator -= ( symbol_type const& symbol_ )
        {
            operator += ( - term_type{ symbol_ } );
            return *this;
        }

        friend self_type const operator - ( self_type const& lhs, symbol_type const& rhs )
        {
            self_type ans{ lhs };
            ans -= rhs;
            return ans;
        }

        friend self_type const operator - ( symbol_type const& lhs, self_type const& rhs )
        {
            self_type const m_lhs{ lhs };
            return -m_lhs + rhs;
        }

        self_type& operator -= ( term_type const& term_ )
        {
            operator += ( -term_ );
            return *this;
        }

        friend self_type const operator - ( self_type const& lhs, term_type const& rhs )
        {
            self_type ans{ lhs };
            ans -= rhs;
            return ans;
        }

        friend self_type const operator - ( term_type const& lhs, self_type const& rhs )
        {
            self_type const m_lhs{ lhs };
            return -m_lhs + rhs;
        }

        self_type& operator -= ( self_type const& self_ )
        {
            for ( auto const& the_term : self_.collection )
                operator -= ( the_term );
            return *this;
        }

        friend self_type const operator - ( self_type const& lhs, self_type const& rhs )
        {
            self_type ans{ lhs };
            ans -= rhs;
            return ans;
        }

        // operator *
        self_type& operator *= ( value_type const value_ )
        {
            if ( value_type{} == value_ )
            {
                collection.clear();
                return *this;
            }

            self_type other;
            for ( auto const& the_term: collection )
                other += the_term * value_;

            std::swap( other.collection, collection );

            return *this;
        }

        friend self_type const operator * ( self_type const& lhs, value_type const rhs )
        {
            self_type ans( lhs );
            ans *= rhs;
            return ans;
        }

        friend self_type const operator * ( value_type const& lhs, self_type const rhs )
        {
            return rhs * lhs;
        }

        self_type& operator *= ( symbol_type const& symbol_ )
        {
            self_type ans;

            for ( auto const& the_term : collection )
                ans += the_term * symbol_;
            std::swap( ans.collection, collection );

            return *this;
        }

        friend self_type const operator * ( self_type const& lhs, symbol_type const rhs )
        {
            self_type ans( lhs );
            ans *= rhs;
            return ans;
        }

        friend self_type const operator * ( symbol_type const& lhs, self_type const rhs )
        {
            return rhs * lhs;
        }

        self_type& operator *= ( term_type const& term_ )
        {
            if ( value_type{} == term_.coefficient )
            {
                collection.clear();
                return *this;
            }

            self_type ans;
            for ( auto const& the_term: collection )
                ans += the_term * term_;

            std::swap( ans.collection, collection );

            return *this;
        }

        friend self_type const operator * ( self_type const& lhs, term_type const& rhs )
        {
            self_type ans( lhs );
            ans *= rhs;
            return ans;
        }

        friend self_type const operator * ( term_type const& lhs, self_type const rhs )
        {
            return rhs * lhs;
        }

        self_type& operator *= ( self_type const& self_ )
        {
            self_type ans;

            for ( auto const& term1: collection )
                for ( auto const& term2 : self_.collection )
                    ans += term1 * term2;

            std::swap( ans.collection, collection );

            return *this;
        }

        friend self_type const operator * ( self_type const& lhs, self_type const& rhs )
        {
            self_type ans{ lhs };
            ans *= rhs;
            return ans;
        }

        friend std::ostream& operator << ( std::ostream& os, self_type const& rhs )
        {
            if ( rhs.collection.empty() )
                return os << 0;

            for ( auto const& the_term: rhs.collection )
                os << the_term;

            return os;
        }

        std::ostream& print_to_code( std::ostream& os ) const
        {
            if ( collection.empty() )
            {
                os << "    return 0";
                return os;
            }

            if ( collection.size() == 1 )
            {
                os << "    return " << *(collection.begin());
                return os;
            }

            auto itor = collection.begin();
            //os << "    double tmp = " << *itor << ";\n";
            os << "    auto tmp = " << *itor << ";\n";

            while ( ++itor != collection.end() )
                os << "    tmp += " << *itor << ";\n";

            os << "    return tmp";

            return os;
        }

        unsigned long size() const
        {
            return collection.size();
        }

    };//struct polynomial

    //
    // term - value
    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator + ( term<T, Symbol_T> const& lhs, T const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans += term<T, Symbol_T>{ rhs };
        return ans;
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator - ( term<T, Symbol_T> const& lhs, T const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans -= term<T, Symbol_T>{ rhs };
        return ans;
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator * ( term<T, Symbol_T> const& lhs, T const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans *= term<T, Symbol_T>{ rhs };
        return ans;
    }

    // value - term
    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator + ( T const& lhs, term<T, Symbol_T> const& rhs )
    {
        return rhs + lhs;
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator - ( T const& lhs, term<T, Symbol_T> const& rhs )
    {
        return - rhs + lhs;
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator * ( T const& lhs, term<T, Symbol_T> const& rhs )
    {
        return rhs * lhs;
    }

    //term-term symbol-symbol symbol-term  term-symbol
    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator + ( term<T, Symbol_T> const& lhs, term<T, Symbol_T> const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans += rhs;
        return ans;
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator - ( term<T, Symbol_T> const& lhs, term<T, Symbol_T> const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans -= rhs;
        return ans;
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator * ( term<T, Symbol_T> const& lhs, term<T, Symbol_T> const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans *= rhs;
        return ans;
    }

    //symbol-term
    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator + ( Symbol_T const& lhs, term<T, Symbol_T> const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans += rhs;
        return ans;
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator - ( Symbol_T const& lhs, term<T, Symbol_T> const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans -= rhs;
        return ans;
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator * ( Symbol_T const& lhs, term<T, Symbol_T> const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans *= rhs;
        return ans;
    }

    //term-symbol
    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator + ( term<T, Symbol_T> const& lhs, Symbol_T const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans += rhs;
        return ans;
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator - ( term<T, Symbol_T> const& lhs, Symbol_T const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans -= rhs;
        return ans;
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const operator * ( term<T, Symbol_T> const& lhs, Symbol_T const& rhs )
    {
        polynomial<T, Symbol_T> ans{ lhs };
        ans *= rhs;
        return ans;
    }

    //make_derive and make_polynomial here

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const make_polynomial_derivative( polynomial<T, Symbol_T> const& polynomial_, Symbol_T const& symbol_ )
    {
        polynomial<T, Symbol_T> derivative;

        for ( auto const& term_ : polynomial_.collection )
            derivative += make_term_derivative( term_, symbol_ );

        return derivative;
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const make_second_polynomial_derivative( polynomial<T, Symbol_T> const& polynomial_, Symbol_T const& symbol_, Symbol_T const& symbol__ )
    {
        return make_polynomial_derivative( make_polynomial_derivative( polynomial_, symbol_ ), symbol__ );
    }

    template< typename T, typename Symbol_T >
    T eval( polynomial<T, Symbol_T> const& polynomial_ )
    {
        return polynomial_.eval();
    }

    template< typename T, typename Symbol_T >
    polynomial<T, Symbol_T> const abs( polynomial<T, Symbol_T> const& polynomial_ )
    {
        return polynomial_.abs();
    }


}//namespace f

#endif//QLITBFTBEPRNEEALKYKBTHNNPRAXOQKKLWVPOWJNNXVOKIXMRIMRRAGGAKXKPVQQBTQPSRMWG

