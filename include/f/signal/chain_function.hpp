#ifndef MCHAIN_FUNCTION_HPP_INCLUDED_SOFIHJ498UFSDKLJHKJFSDLKJSFDNMSIOHJ4EOIUSFDKJHEIUHSFDKJHEIUHSDKFJHSAKJFCKJHDKJHSFDDS
#define MCHAIN_FUNCTION_HPP_INCLUDED_SOFIHJ498UFSDKLJHKJFSDLKJSFDNMSIOHJ4EOIUSFDKJHEIUHSFDKJHEIUHSDKFJHSAKJFCKJHDKJHSFDDS

#include <functional>

namespace f
{
  namespace signal_private
  {
    template< typename R, typename... Args >
    struct chain_function
    {
        typedef R return_type;
        typedef std::function<R(Args...)> function_type;

        template<typename F>
        const function_type
        operator()( const F& f_ ) const
        { return f_; }

        template<typename F, typename... Fs>
        const function_type
        operator()( const F& f_, const Fs&... fs_ ) const
        {
            const auto f = chain_function<R, Args...>()(f_);
            const auto g = chain_function<R, Args...>()(fs_...);
            return [f, g](Args... args) { f(args...); return g(args...); };
        }
    };
  }//namespace signal0x_private
}//namespace signal0x

#endif//_CHAIN_FUNCTION_HPP_INCLUDED_SOFIHJ498UFSDKLJHKJFSDLKJSFDNMSIOHJ4EOIUSFDKJHEIUHSFDKJHEIUHSDKFJHSAKJFCKJHDKJHSFDDS

