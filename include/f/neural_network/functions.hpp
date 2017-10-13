#ifndef LCOBEQYMKVFNCCDMEXRJUQBCQYAKOILOGTJLERGSGMNCXOPQTBMANKPVVTRSLHFRPSAADNVNE
#define LCOBEQYMKVFNCCDMEXRJUQBCQYAKOILOGTJLERGSGMNCXOPQTBMANKPVVTRSLHFRPSAADNVNE

#include <f/overloader/overloader.hpp>
#include <f/matrix/matrix.hpp>
#include <f/algorithm/for_each.hpp>
#include <f/algorithm/all_of.hpp>

#include <memory>
#include <algorithm>
#include <vector>
#include <utility>
#include <cmath>
#include <array>
#include <type_traits>
#include <cassert>

#include <iostream>

namespace f
{

    namespace nn
    {
        //TODO:
        //      check input parameters

        template<typename T>
        auto logistic() noexcept
        {
            typedef T value_type;
            return std::make_pair(  [=]( value_type x ) noexcept { return 1.0 / ( 1.0 + std::exp(-x) ); },
                                    [=]( value_type x ) noexcept { auto const fx = 1.0 / ( 1.0 + std::exp(-x) ); return fx * ( 1.0 - fx );} );
                                    //[=]( value_type x ) noexcept { return x * ( 1.0 - x );} );
        };

        template<typename T>
        inline auto gaussian() noexcept
        {
            return std::make_pair(  [=]( T x ) noexcept { return T{std::exp(-x*x)}; },
                                    [=]( T x ) noexcept { return T{-2.0 * x * std::exp(-x*x)}; } );
        }
#if 0
        template< typename T > struct type_converter
        {
            typedef T result_type;
            template< typename U >
            result_type operator()( U u ) const noexcept
            {
                return static_cast<result_type>(u);
            }
        };

        inline auto logistic() noexcept
        {
            return []( auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { return converter(1.0 / ( 1.0 + std::exp(-x) )); },
                                        [=]( auto x ) noexcept { auto const fx = 1.0 / ( 1.0 + std::exp(-x) ); return converter(fx * ( 1.0 - fx ) );} );
            };
        }

        inline auto prelu() noexcept
        {
            return []( auto alpha, auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { if ( x < 0 ) return converter(alpha*x); return converter(x); },
                                        [=]( auto x ) noexcept { if ( x < 0 ) return converter(alpha); return converter(1.0); } );
            };
        }

        inline auto identity() noexcept
        {
            return []( auto converter ) noexcept
            {
                return std::make_pair( [=]( auto x ) noexcept { return converter(x); }, [=]( auto ) noexcept { return converter(1.0); } );
            };
        }

        inline auto soft_step() noexcept
        {
            return logistic();
        }

        inline auto tanh() noexcept
        {
            return []( auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { return converter(std::tanh(x)); },
                                        [=]( auto x ) noexcept { auto const fx = std::tanh(x); return converter( 1.0 - fx*fx ); } );
            };
        }

        inline auto arctan() noexcept
        {
            return []( auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { return converter(std::atan(x)); },
                                        [=]( auto x ) noexcept { return converter(1.0 / ( 1.0 + x*x )); } );
            };
        }

        inline auto soft_sign() noexcept
        {
            return []( auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { return converter(x / ( 1.0 + std::abs(x)) ); },
                                        [=]( auto x ) noexcept { auto tm = 1.0 + std::abs(x); return converter(1.0 / ( tm * tm )); } );
            };
        }

        inline auto rectifier() noexcept
        {
            return []( auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { if ( x < 0 ) return converter(0); return converter(x); },
                                        [=]( auto x ) noexcept { if ( x < 0 ) return converter(0); return converter(1.0); } );
            };
        }

        inline auto parameteric_rectifield_linear_unit() noexcept
        {
            return prelu();
        }

        inline auto elu() noexcept
        {
            return []( auto alpha, auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { if ( x < 0 ) return converter(alpha * ( std::exp(x) - 1.0 )); return converter(x); },
                                        [=]( auto x ) noexcept { if ( x < 0 ) return converter(alpha * std::exp(x)); return converter(1.0); } );
            };
        }

        inline auto exponential_linear_unit() noexcept
        {
            return elu();
        }

        inline auto soft_plus() noexcept
        {
            return []( auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { return converter(std::log( 1.0 + std::exp(x) )); },
                                        [=]( auto x ) noexcept { return converter(1.0 / ( 1.0 + std::exp(-x) )); } );
            };
        }

        inline auto bent_identity() noexcept
        {
            return []( auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { return converter(x + ( std::sqrt( 1.0 + x*x ) - 1.0 ) / 2.0); },
                                        [=]( auto x ) noexcept { return converter(1.0 + x / ( 2.0 * std::sqrt( 1.0 + x*x ) )); } );
            };
        }

        inline auto soft_exponential() noexcept
        {
            return []( auto alpha, auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { if (alpha < 0 ) return converter(- std::log(1.0-alpha*(x+alpha)) / alpha); if (alpha > 0 ) return converter( (std::exp(alpha*x) - 1.0) / alpha + alpha ); return converter(x); },
                                        [=]( auto x ) noexcept { if (alpha < 0 ) return converter( 1.0 / ( 1.0 - alpha*(alpha+x) )); return converter(std::exp(alpha*x)); } );
            };
        }

        inline auto sinsoid() noexcept
        {
            return []( auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { return converter(std::sin(x)); },
                                        [=]( auto x ) noexcept { return converter(std::cos(x)); } );
            };
        }

        inline auto sinc() noexcept
        {
            return []( auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { if ( std::abs(x) > 1.0e-5 ) return converter(std::sin(x)/x);  return converter(1.0); },
                                        [=]( auto x ) noexcept { if ( std::abs(x) > 1.0e-5 ) return converter(std::cos(x)/x - std::sin(x)/(x*x)); return converter(0); } );
            };
        }

        inline auto square() noexcept
        {
            return []( auto converter ) noexcept
            {
                return std::make_pair(  [=]( auto x ) noexcept { return converter(x*x); },
                                        [=]( auto x ) noexcept { return converter(2.0 * x); } );
            };
        }

#endif

    }//namespace nn

}//namespace f

#endif//LCOBEQYMKVFNCCDMEXRJUQBCQYAKOILOGTJLERGSGMNCXOPQTBMANKPVVTRSLHFRPSAADNVNE
