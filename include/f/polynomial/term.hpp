#ifndef DTXMRABYEANDNEQBTXNYJILLTFNWIJFJQTEAXIYEPHCPBLCUOEBGYGXTDEHQCRAYKFBXDFXYJ
#define DTXMRABYEANDNEQBTXNYJILLTFNWIJFJQTEAXIYEPHCPBLCUOEBGYGXTDEHQCRAYKFBXDFXYJ

#include <f/polynomial/symbol.hpp>

#include <cstddef>
#include <map>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <set>

namespace f
{
    //TODO:
    //      employ flyweight pattern for map
    template<typename T, typename Symbol_T>
    struct term
    {
        typedef term                                self_type;
        typedef T                                   value_type;
        typedef Symbol_T                            symbol_type;
        typedef std::size_t                         size_type;
        typedef std::map<symbol_type, size_type>    symbol_power_associate_type;
        typedef std::multiset<symbol_type>          multi_symbol_type;

        //----------------------------------------------------------------------
        // Here we try to keep both
        // the power-form of symbols   --   record  -- maps the [symbol - order]
        // as well as
        // the original symbols        --   symbols -- keeps all the [symbols]
        // of a term.
        //----------------------------------------------------------------------
        value_type                                  coefficient;
        symbol_power_associate_type                 record;
        multi_symbol_type                           symbols;

        explicit term( const value_type coefficient_ = value_type(1) ) : coefficient( coefficient_ ) {}

        term( const value_type coefficient_, const symbol_power_associate_type& record_ ) : coefficient( coefficient_ ), record( record_ )
        {
            update_symbols();
        }

        explicit term( const symbol_type& symbol_ ): coefficient(1)
        {
            record[symbol_] = 1;
            update_symbols();
        }

        size_type order() const
        {
            size_type ans = 0;
            for ( auto const& element : record )
                ans += element.second;
            return ans;
        }

        term( self_type const& other ) : coefficient( other.coefficient ), record( other.record )
        {
            update_symbols();
        }

        //record -->> symbols
        void update_symbols()
        {
            symbols.clear();
            for ( auto const& element : record )
            {
                size_type n = element.second;
                while ( n-- )
                    symbols.insert( element.first );
            }
        }

        self_type const operator - () const
        {
            return self_type{ -coefficient, record };
        }

        //this requires Symbol_T have an [operator value_type() const] method defined
        //operator value_type() const
        value_type eval() const
        {
            value_type ans = coefficient;

            for ( auto const& kv : record )
            {
                value_type const tmp = (kv.first).eval();//eval(kv.first);
                ans *= std::pow( tmp, kv.second );
            }

            return ans;
        }

        self_type& operator *= ( const value_type val )
        {
            if ( value_type{} == val )
            {
                clear();
                return *this;
            }

            coefficient *= val;
            return *this;
        }

        self_type& operator *= ( symbol_type const& other )
        {
            if ( coefficient != value_type{} )
            {
                ++record[other];
                update_symbols();
            }

            return *this;
        }

        self_type& operator *= ( self_type const& other )
        {
            if ( ( coefficient == value_type{} ) || ( other.coefficient == value_type{} ) )
            {
                clear();
                return *this;
            }

            operator *= ( other.coefficient );

            for ( auto const& kv : other.record )
                record[kv.first] += kv.second;

            update_symbols();
            return *this;
        }

        friend self_type const operator * ( value_type const lhs, self_type const& rhs )
        {
            self_type ans( rhs );
            ans *= lhs;
            return ans;
        }

        friend self_type const operator * ( self_type const& lhs, value_type const rhs )
        {
            return rhs * lhs;
        }

        friend self_type const operator * ( symbol_type const& lhs, self_type const& rhs )
        {
            self_type ans( rhs );
            ans *= lhs;
            return ans;
        }

        friend self_type const operator * ( self_type const& lhs, symbol_type const rhs )
        {
            return rhs * lhs;
        }

        friend self_type const operator * ( self_type const& lhs, self_type const& rhs )
        {
            self_type ans( rhs );
            ans *= lhs;
            return ans;
        }
#if 0
        friend self_type const operator * ( symbol_type const& lhs, symbol_type const& rhs )
        {
            self_type ans( rhs );
            ans *= lhs;
            return ans;
        }
#endif

        friend std::ostream& operator << ( std::ostream& os, self_type const& rhs )
        {

            os << "  ";

            os << std::setiosflags( std::ios_base::showpos ) << rhs.coefficient;
#if 1
            for ( auto const& kv : rhs.record )
            {
                os << " * ";

                os << kv.first;

                if ( kv.second > 1 )
                    os << "^" << kv.second;

/*
                if ( kv.second > 1 )
                {
                    auto power = kv.second;
                    while( power != 1 )
                    {
                        os << kv.first << " * ";
                        --power;
                    }
                }

                os << kv.first;
*/
            }
#endif
#if 0
            os << " ";
            std::copy( rhs.symbols.begin(), rhs.symbols.end(), std::ostream_iterator<symbol_type>(os, "") );
#endif

            return os;
        }

        friend bool operator < ( const self_type& lhs, const self_type& rhs )
        {
            if ( std::abs(lhs.coefficient) < std::abs(rhs.coefficient) ) return true;
            if ( std::abs(lhs.coefficient) == std::abs(rhs.coefficient) )
            {
                if ( lhs.order() < rhs.order() ) return true;
                if ( lhs.order() > rhs.order() ) return false;
                return lhs.symbols < rhs.symbols;
            }
            return false;
        }

        friend bool operator > ( const self_type& lhs, const self_type& rhs )
        {
            return std::abs(lhs.coefficient) > std::abs(rhs.coefficient);
            //if ( lhs.order() > rhs.order() ) return true;
            //if ( lhs.order() < rhs.order() ) return false;
            //return lhs.symbols > rhs.symbols;
        }

        friend bool operator == ( const self_type& lhs, const self_type& rhs )
        {
            return lhs.symbols == rhs.symbols;
            //return lhs.record == rhs.record;
        }

        self_type const abs() const
        {
            if ( coefficient < value_type{} )
                return term{ -coefficient, record };

            return term{ coefficient, record };
        }

        size_type size() const
        {
            return record.size();
            //return symbols.size();
        }

        void clear()
        {
            coefficient = value_type{};
            record.clear();
            symbols.clear();
        }

    };//term

    //unary case
    template<typename T, typename Symbol_T>
    term<T, Symbol_T> const make_term( T const coefficient_ )
    {
        term<T, Symbol_T> term1{ coefficient_ };
        return term1;
    }

    template<typename T, typename Symbol_T>
    term<T, Symbol_T> const make_term( Symbol_T const& symbol_ )
    {
        term<T, Symbol_T> term1{ symbol_ };
        return term1;
    }

    template<typename T, typename Symbol_T>
    term<T, Symbol_T> const make_term( term<T, Symbol_T> const& term_ )
    {
        return term_;
    }

    //binary case
    //// value -- ?
#if 0
    template<typename T, typename Symbol_T> term<T, Symbol_T> const
    make_term( T const coefficient1_, T const coefficient2_ )
    {
        return make_term<T, Symbol_T>( coefficient1_ * coefficient2_ );
    }
#endif

    template<typename T, typename Symbol_T> term<T, Symbol_T> const
    make_term( T const coefficient_, Symbol_T const& symbol_ )
    {
        return term<T, Symbol_T>{coefficient_} * term<T, Symbol_T>{symbol_};
    }

    template<typename T, typename Symbol_T> term<T, Symbol_T> const
    make_term( T const coefficient_, term<T, Symbol_T> const& term_ )
    {
        return term<T, Symbol_T>{coefficient_} * term_;
    }

    //// symbol -- ?
#if 0
    template<typename T, typename Symbol_T> term<T, Symbol_T> const
    make_term( Symbol_T const& symbol_, T const coefficient_ )
    {
        return make_term( coefficient_, symbol_ );
    }
#endif

    template<typename T, typename Symbol_T> term<T, Symbol_T> const
    make_term( Symbol_T const& symbol1_, Symbol_T const& symbol2_ )
    {
        return term<T, Symbol_T>{ symbol1_ } * term<T, Symbol_T>{ symbol2_ };
    }

    template<typename T, typename Symbol_T> term<T, Symbol_T> const
    make_term( Symbol_T const& symbol_, term<T, Symbol_T> const& term_ )
    {
        return term<T, Symbol_T>{ symbol_ } * term_;
    }

    //// term ---- ?
#if 0
    template<typename T, typename Symbol_T> term<T, Symbol_T> const
    make_term( term<T, Symbol_T> const& term_, T const coefficient_ )
    {
        return term_ * term<T, Symbol_T>{coefficient_};
    }
#endif

    template<typename T, typename Symbol_T> term<T, Symbol_T> const
    make_term( term<T, Symbol_T> const& term_, Symbol_T const& symbol_ )
    {
        return term_ * term<T, Symbol_T>{ symbol_ };
    }

    template<typename T, typename Symbol_T> term<T, Symbol_T> const
    make_term( term<T, Symbol_T> const& term1_, term<T, Symbol_T> const& term2_ )
    {
        return term1_ * term2_;
    }

    //variadic case
    //// value -- ?
#if 0
    template<typename T, typename Symbol_T, typename ... More> term<T, Symbol_T> const
    make_term( T const coefficient1_, T const coefficient2_, More const& ... more_ )
    {
        return make_term( make_term<T, Symbol_T>( coefficient1_ * coefficient2_ ), more_... );
    }
#endif

    template<typename T, typename Symbol_T, typename ... More> term<T, Symbol_T> const
    make_term( T const coefficient_, Symbol_T const& symbol_, More const& ... more_ )
    {
        return make_term( term<T, Symbol_T>{coefficient_} * term<T, Symbol_T>{symbol_}, more_... );
    }

    template<typename T, typename Symbol_T, typename ... More> term<T, Symbol_T> const
    make_term( T const coefficient_, term<T, Symbol_T> const& term_, More const& ... more_ )
    {
        return make_term( term<T, Symbol_T>{coefficient_} * term_, more_... );
    }

    //// symbol -- ?
#if 0
    template<typename T, typename Symbol_T, typename ... More> term<T, Symbol_T> const
    make_term( Symbol_T const& symbol_, T const coefficient_, More const& ... more_ )
    {
        return make_term( make_term( coefficient_, symbol_ ), more_... );
    }
#endif

    template<typename T, typename Symbol_T, typename ... More> term<T, Symbol_T> const
    make_term( Symbol_T const& symbol1_, Symbol_T const& symbol2_, More const& ... more_ )
    {
        return make_term( term<T, Symbol_T>{ symbol1_ } * term<T, Symbol_T>{ symbol2_ }, more_... );
    }

    template<typename T, typename Symbol_T, typename ... More> term<T, Symbol_T> const
    make_term( Symbol_T const& symbol_, term<T, Symbol_T> const& term_, More const& ... more_ )
    {
        return make_term( term<T, Symbol_T>{ symbol_ } * term_, more_... );
    }

    //// term ---- ?
#if 0
    template<typename T, typename Symbol_T, typename ... More> term<T, Symbol_T> const
    make_term( term<T, Symbol_T> const& term_, T const coefficient_, More const& ... more_ )
    {
        return make_term( term_ * term<T, Symbol_T>{coefficient_}, more_... );
    }
#endif

    template<typename T, typename Symbol_T, typename ... More> term<T, Symbol_T> const
    make_term( term<T, Symbol_T> const& term_, Symbol_T const& symbol_, More const& ... more_ )
    {
        return make_term( term_ * term<T, Symbol_T>{ symbol_ }, more_... );
    }

    template<typename T, typename Symbol_T, typename ... More> term<T, Symbol_T> const
    make_term( term<T, Symbol_T> const& term1_, term<T, Symbol_T> const& term2_, More const& ... more_ )
    {
        return make_term( term1_ * term2_, more_... );
    }


#if 0
    template<typename T, typename Symbol_T>
    term<T, Symbol_T> const make_term( term<T, Symbol_T> const& term1_, term<T, Symbol_T> const& term2_ )
    {
        return term1_ * term2_;
    }

    template<typename T, typename Symbol_T>
    term<T, Symbol_T> const make_term( term<T, Symbol_T> const& term_, Symbol_T const& symbol_ )
    {
        return term_ * symbol_;
    }

    template<typename T, typename Symbol_T, typename Term_Or_Symbol, typename... Symbol_S>
    term<T, Symbol_T> const make_term( T coefficient_, Term_Or_Symbol const& tos_, Symbol_S const& ... symbols_ )
    {
        auto const& term1 = make_term( coefficient_, tos_ );
        return make_term( term1, symbols_...);
    }

    template<typename T, typename Symbol_T, typename Term_Or_Symbol, typename... Symbol_S>
    term<T, Symbol_T> const make_term( term<T, Symbol_T> const& term_, Term_Or_Symbol const& tos_, Symbol_S const& ... symbols_ )
    {
        auto const& term1 = make_term( term_, tos_ );
        return make_term( term1, symbols_...);
    }

    template<typename T, typename Symbol_T, typename... Symbol_S>
    term<T, Symbol_T> const make_term( term<T, Symbol_T> const& term_, Symbol_T const& symbol_, Symbol_S const& ... symbols_ )
    {
        auto const& term1 = make_term( term_, symbol_ );
        return make_term( term1, symbols_...);
    }

    template<typename T, typename Symbol_T>
    term<T, Symbol_T> const make_term( T const coefficient_, Symbol_T const& symbol_ )
    {
        term<T, Symbol_T> const term1{ coefficient_ };
        return term1 * symbol_;
    }

    template<typename T, typename Symbol_T>
    term<T, Symbol_T> const make_term( T const coefficient_, term<T, Symbol_T> const& term_ )
    {
        return term_ * coefficient_;
    }

    template<typename T, typename Symbol_T>
    term<T, Symbol_T> const make_term( T const coefficient_ )
    {
        term<T, Symbol_T> const term1{ coefficient_ };
        return term1;
    }

    template<typename T, typename Symbol_T>
    term<T, Symbol_T> const make_term( Symbol_T const& symbol_ )
    {
        return make_term( T(1), symbol_ );
    }

    template<typename T, typename Symbol_T, typename... Symbol_S>
    term<T, Symbol_T> const make_term( T const coefficient_, Symbol_T const& symbol_, Symbol_S const& ... symbols_ )
    {
        term<T, Symbol_T> const term1 = make_term( coefficient_, symbol_ );
        return make_term( term1, symbols_...);
    }

#endif

    template<typename T, typename Symbol_T>
    term<T, Symbol_T> const make_term_derivative( term<T, Symbol_T> const& term_, Symbol_T const& symbol_ )
    {
        if ( term_.record.find( symbol_ ) != term_.record.end() )
        {
            auto ans = term_;

            if ( 1 == ans.record[symbol_] )
            {
                ans.record.erase( symbol_ );
                return ans;
            }

            ans.coefficient *= ans.record[symbol_]--;
            return ans;
        }

        return term<T, Symbol_T>{ T{} };
    }

    //make_term( val, symbol );
    //make_term( val, symbol, symbol... );
    //

    template< typename T, typename Symbol_T >
    T eval( term<T, Symbol_T> const& term_ )
    {
        return term_.eval();
    }

    template< typename T, typename Symbol_T >
    T abs( term<T, Symbol_T> const& term_ )
    {
        return term_.abs();
    }

}//namespace f

#endif//DTXMRABYEANDNEQBTXNYJILLTFNWIJFJQTEAXIYEPHCPBLCUOEBGYGXTDEHQCRAYKFBXDFXYJ

