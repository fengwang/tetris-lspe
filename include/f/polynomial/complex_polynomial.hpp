#ifndef ODFKFBNJBANTJRVEVTQPBBVQVQFDGSGJPKOEIKSGFPICCRDPOALDHYWFMUUASCQTTWKGPDAFE
#define ODFKFBNJBANTJRVEVTQPBBVQVQFDGSGJPKOEIKSGFPICCRDPOALDHYWFMUUASCQTTWKGPDAFE

#include <complex>
#include <algorithm>

#include "./polynomial.hpp"

namespace f
{

    template< typename T, typename Symbol_T >
    struct polynomial< std::complex<T>, Symbol_T >
    {
        typedef polynomial<std::complex<T>, Symbol_T>   self_type;

        typedef polynomial<T, Symbol_T>                 polynomial_type;
        typedef term<T, Symbol_T>                       term_type;
        typedef Symbol_T                                symbol_type;
        typedef T                                       value_type;
        typedef std::complex<T>                         complex_type;

        polynomial_type                                 real_polynomial;
        polynomial_type                                 imag_polynomial;

        polynomial(){}
        polynomial( complex_type const& complex_ ) : real_polynomial{ std::real(complex_) }, imag_polynomial{ std::imag(complex_) } {}
        polynomial( value_type real_value_, value_type imag_value_ = value_type{} ) : real_polynomial{ real_value_ }, imag_polynomial{ imag_value_ } {}
        polynomial( symbol_type real_symbol_, symbol_type imag_symbol_ = symbol_type{} ) : real_polynomial{ real_symbol_ }, imag_polynomial{ imag_symbol_ } {}
        polynomial( term_type const& real_term_, term_type const& imag_term_ = term_type{} ) : real_polynomial{ real_term_ }, imag_polynomial{ imag_term_ } {}
        polynomial( polynomial_type const& real_polynomial_, polynomial_type const& imag_polynomial_ = polynomial_type{} ) : real_polynomial{ real_polynomial_ }, imag_polynomial{ imag_polynomial_ } {}
        polynomial( self_type const& ) = default;
        polynomial( self_type&& ) = default;
        self_type& operator = ( self_type const& ) = default;
        self_type& operator = ( self_type&& ) = default;

        self_type& operator += ( self_type const& other_ )
        {
            real_polynomial += other_.real_polynomial;
            imag_polynomial += other_.imag_polynomial;
            return *this;
        }

        self_type& operator *= ( self_type const& other_ )
        {
            auto const& nr = real_polynomial * other_.real_polynomial - imag_polynomial * other_.imag_polynomial;
            auto const& ni = real_polynomial * other_.imag_polynomial + imag_polynomial * other_.real_polynomial;
            real_polynomial = nr;
            imag_polynomial = ni;
            return *this;
        }

        void trim( value_type const v_ = value_type{1.0e-3} )
        {
            real_polynomial.trim( v_ );
            imag_polynomial.trim( v_ );
        }

        polynomial_type norm() const
        {
            return real_polynomial * real_polynomial + imag_polynomial * imag_polynomial;
        }

        polynomial_type real() const
        {
            return real_polynomial;
        }

        polynomial_type imag() const
        {
            return imag_polynomial;
        }

        self_type const conj() const
        {
            return self_type{ real_polynomial, -imag_polynomial };
        }

        friend self_type const operator + ( self_type const& lhs, self_type const& rhs )
        {
            self_type ans{ lhs };
            ans += rhs;
            return ans;
        }

        friend self_type const operator * ( self_type const& lhs, self_type const& rhs )
        {
            self_type ans{ lhs };
            ans *= rhs;
            return ans;
        }

        friend std::ostream& operator << ( std::ostream& os_, self_type const& rhs )
        {
            os_ << "(" << rhs.real_polynomial << ") + i(" << rhs.imag_polynomial << ")";
            return os_;
        }
    };

    template< typename T, typename Symbol_T >
    polynomial<std::complex<T>, Symbol_T> const make_polynomial_derivative( polynomial<std::complex<T>, Symbol_T> const& polynomial_, Symbol_T const& symbol_ )
    {
        return polynomial<std::complex<T>, Symbol_T>{ make_polynomial_derivative( polynomial_.real_polynomial, symbol_ ), make_polynomial_derivative( polynomial_.imag_polynomial, symbol_ )  };
    }

    template< typename T, typename Symbol_T >
    polynomial<std::complex<T>, Symbol_T> const make_polynomial( term<T, Symbol_T> const& rt_, term<T, Symbol_T> const& it_ )
    {
        return polynomial<std::complex<T>, Symbol_T>{ rt_, it_ };
    }

}//namespace f

#endif//ODFKFBNJBANTJRVEVTQPBBVQVQFDGSGJPKOEIKSGFPICCRDPOALDHYWFMUUASCQTTWKGPDAFE

