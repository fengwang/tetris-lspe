#ifndef VUYYGKACXAECCNXWCRDJDBWSFDXEOMYEGTBPXMPRFIUCSYWDMIHLQQLRNEXQLMLGUVGATXAYE
#define VUYYGKACXAECCNXWCRDJDBWSFDXEOMYEGTBPXMPRFIUCSYWDMIHLQQLRNEXQLMLGUVGATXAYE

#include <memory>

namespace f
{

    template <class F, class R, class... Args>
    struct recursive_lambda;

    template <class F, class R, class... Args>
    struct recursive_lambda<R( Args... ), F> : public std::enable_shared_from_this<recursive_lambda<R( Args... ), F>>
    {
        typedef recursive_lambda<R( Args... ), F> self_type;

        F f_;

        R operator()( Args... args ) const
        {
            return f_( *this, args... );
        }

        std::weak_ptr<self_type> weak_shared_from_this()
        {
            return std::weak_ptr<self_type>( this->shared_from_this() );
        }

        recursive_lambda( F f ) : f_( f ) {}
    };

    template <class R, class... Args, class F>
    recursive_lambda<R( Args... ), F> make_recursive_lambda( F&& f )
    {
        return recursive_lambda<R( Args... ), F>( std::forward<F>( f ) );
    }

}//namespace f

#endif//VUYYGKACXAECCNXWCRDJDBWSFDXEOMYEGTBPXMPRFIUCSYWDMIHLQQLRNEXQLMLGUVGATXAYE
