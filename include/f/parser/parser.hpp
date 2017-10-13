#ifndef LBGFRQVUVNDGCURGOHUTPFAJDBUFCDTXYCHJNKPJKVQPUHSVSDAWQVLNCDWICKFLCGPLYKNSN
#define LBGFRQVUVNDGCURGOHUTPFAJDBUFCDTXYCHJNKPJKVQPUHSVSDAWQVLNCDWICKFLCGPLYKNSN

#include <sstream>
#include <string>
#include <cstddef>
#include <functional>
#include <algorithm>
#include <iterator>
#include <vector>
#include <cstring>

#include <iostream>


namespace f
{

    template< typename T > struct tag
    {
        typedef T   value_type;
    };

    template< unsigned long N >
    struct integer
    {
        static const unsigned long value = N;
    };

    template< typename ... Types >
    auto option( Types ... args )
    {
        return [=]( auto invoker ) { return invoker( args ... ); };
    }    

    template< typename T, typename Triggered_Action, typename Silent_Action >
    auto make_option( std::string token_, Triggered_Action triggered_action_, Silent_Action silent_action_ )
    {
        return option( token_, triggered_action_, silent_action_, tag<T>{} );
    }

    template< typename T, unsigned long N, typename Triggered_Action, typename Silent_Action >
    auto make_option( std::string token_, char delimiter_, Triggered_Action triggered_action_, Silent_Action silent_action_ )
    {
        return option( token_, delimiter_, triggered_action_, silent_action_, tag<T>{}, integer<N>{} );
    }

    template< typename T, typename Triggered_Action >
    auto make_option( std::string token_, Triggered_Action triggered_action_ )
    {
        return make_option<T>( token_, triggered_action_, [](){} );
    }

    template< typename T, unsigned long N, typename Triggered_Action >
    auto make_option( std::string token_, char delimiter_, Triggered_Action triggered_action_ )
    {
        return make_option<T, N>( token_, delimiter_, triggered_action_, [](){} );
    }

    template< typename Arg, typename ... Args >
    struct overloader : Arg, overloader<Args...>
    {
        overloader( Arg arg_, Args ... args_ ) : Arg(arg_), overloader<Args...>( args_... ) {}
    };

    template< typename Arg >
    struct overloader<Arg> : Arg
    {
        overloader( Arg arg_ ) : Arg(arg_){}
    };

    template< typename ... Args >
    auto make_parser( Args ... args_ )
    {
        return overloader<Args...>{ args_... }; 
    }

    template< typename Option, typename ... Options >
    auto parse( int argc_, char** argv_, Option option_, Options ... options_ )
    {
        auto parser = make_parser(      [=]( std::string token_, char delimiter_, auto triggered_action_, auto silent_action_, auto tag_, auto integer_ ) 
                                        {
                                            for ( int index = 1; index != argc_; ++index )
                                                if ( 0 == std::strcmp( argv_[index], token_.c_str() ) )
                                                {
                                                    if ( index+1 == argc_ ) return 1;
                                                    std::string str_{ argv_[index+1] };
                                                    if ( integer_.value-1 != std::count( str_.begin(), str_.end(), delimiter_ ) ) return 1;
                                                    for( auto& ch : str_ ) ch = ( ch == delimiter_ ) ? ' ' : ch;
                                                    std::istringstream iss{ str_ };
                                                    typedef typename decltype(tag_)::value_type value_type;
                                                    std::vector<value_type> vals( integer_.value, 0.0 );
                                                    std::copy( std::istream_iterator<value_type>( iss ), std::istream_iterator<value_type>(), vals.begin() );
                                                    if ( iss.bad() ) return 1;
                                                    triggered_action_( vals.data() );
                                                    return 0;
                                                }

                                            silent_action_();
                                            return 0;
                                        },
                                        [=]( std::string token_, auto triggered_action_, auto silent_action_, auto tag_ )
                                        {
                                            for ( int index = 1; index != argc_; ++index )
                                                if ( 0 == std::strcmp( argv_[index], token_.c_str() ) )
                                                {
                                                    typedef typename decltype(tag_)::value_type value_type;
                                                    if ( index+1 == argc_ ) return 1;
                                                    value_type val;
                                                    std::istringstream iss{ std::string{ argv_[index+1] } };
                                                    iss >> val;

                                                    if ( iss.bad() ) return 1;

                                                    triggered_action_( val );
                                                    return 0;
                                                }

                                            silent_action_();
                                            return 0;
                                        },
                                        [=]( std::string token_, auto triggered_action_, auto silent_action_, tag<void> )
                                        {
                                            for ( int index = 1; index != argc_; ++index )
                                            {
                                                if ( 0 == std::strcmp( argv_[index], token_.c_str() ) )
                                                {
                                                    triggered_action_();
                                                    return 0;
                                                }
                                            }

                                            silent_action_();
                                            return 0;
                                        }
                              );

        return option( option_(parser), options_(parser)... );
    }

}//namespace f

#endif//LBGFRQVUVNDGCURGOHUTPFAJDBUFCDTXYCHJNKPJKVQPUHSVSDAWQVLNCDWICKFLCGPLYKNSN

