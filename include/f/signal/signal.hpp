#ifndef MSIGNAL_HPP_INCLUDED_SOIU498ASDFKLSFDLKJXCVKJDSLKFJASLKJ438721SDP230SFAJHEIOUFDJLKSAFLKSAFOIU3498YSFDUIWEIUOWOIUE
#define MSIGNAL_HPP_INCLUDED_SOIU498ASDFKLSFDLKJXCVKJDSLKFJASLKJ438721SDP230SFAJHEIOUFDJLKSAFLKSAFOIU3498YSFDUIWEIUOWOIUE

#include <f/singleton/singleton.hpp>        // for f::singleton
#include <f/signal/chain_function.hpp>      // for f::signal_private::chain_function

#include <cstddef>      // for std::ptrdiff_t
#include <functional>   // for std::function
#include <utility>      // for std::forward, std::move
#include <algorithm>    // for std::swap
#include <map>          // for std::map
#include <utility>      // for std::make_pair
#include <limits>       // for std::numeric_limits<T>::max()
#include <mutex>        // for std::mutex etc.

namespace f
{
    typedef std::ptrdiff_t                                                      connection_type;

    template< typename R, typename... Args >
    struct signal
    {
        typedef std::ptrdiff_t                                                  connection_type;
        typedef connection_type                                                 connection;
        typedef int                                                             priority_type;
        typedef signal                                                          self_type;
        typedef self_type                                                       publisher_type;
        typedef std::function<R( Args... )>                                     function_type;
        typedef function_type                                                   subscriber_type;
        typedef function_type                                                   slot_type;
        typedef std::map<connection_type, subscriber_type>                      associate_connection_subscriber_type;
        typedef std::map<priority_type, associate_connection_subscriber_type>   priority_connection_subscriber_type;
        typedef std::mutex                                                      mutex_type;
        typedef std::lock_guard<mutex_type>                                     lock_guard_type;

    private://lock to get a consistent snapshot of the source
        signal( const self_type& other, const lock_guard_type& ): registered_subscriber_blocked_( other.registered_subscriber_blocked_ ), registered_subscriber_( other.registered_subscriber_ ) {}
        signal( self_type&& other, const lock_guard_type& ): registered_subscriber_blocked_( std::move(other.registered_subscriber_blocked_) ), registered_subscriber_( std::move(other.registered_subscriber_) ) {}
    public:
        signal( const self_type& other ) : signal( other, lock_gurad_type(other.mutex_ ) ) {}
        signal( self_type&& other ) : signal( other, lock_gurad_type(other.mutex_ ) ) {}
        signal() {}

        self_type& operator = ( const self_type& other )
        {   // 1)  lock both mutexes safely
            std::lock( mutex_, other.mutex_ );
            // 2)  adopt the ownership into the std::lock_guard instances to ensure the locks are released safely at the end of the function.
            lock_guard_type l1( mutex_, std::adopt_lock );
            lock_guard_type l2( other.mutex_, std::adopt_lock );
            registered_subscriber_blocked_ = other.registered_subscriber_blocked_;
            registered_subscriber_ = other.registered_subscriber_;
            return *this;
        }

        self_type& operator = ( self_type&& other )
        {
            lock_guard_type l( mutex_ );
            registered_subscriber_blocked_ = std::move(other.registered_subscriber_blocked_);
            registered_subscriber_ = std::move(other.registered_subscriber_);
            return *this;
        }

        template< typename... F >
        connection_type connect( const priority_type w, const F&... f )
        {   //is c thread safe??
            auto ff =  signal_private::chain_function<R, Args...>()(f...);    //generate a function chain
            auto& c =  f::singleton<connection_type>::instance();//a global id generator to do connection representation
            lock_guard_type l( mutex_ );
            (registered_subscriber_[w]).insert( std::make_pair( c, ff ) );      //save connection id and chain function to the table
            return c++;                                                         //increase global id generator
        }

        template< typename... F >
        connection_type connect(  const F&... f ) //if priority parameter not given, set it to the lowest
        { return connect( std::numeric_limits<priority_type>::max(), f... ); }

        void disconnect( const connection_type& c )
        {
            lock_guard_type l( mutex_ );
            for ( auto& i : registered_subscriber_ )
                    if ( i.second.erase( c ) ) { if(!i.second.size()) registered_subscriber_.erase(i.first); return; }
            for ( auto& i : registered_subscriber_blocked_ )
                    if ( i.second.erase( c ) ) { if(!i.second.size()) registered_subscriber_blocked_.erase(i.first); return; }
        }

        template< typename... Connections >
        void disconnect( const connection_type& con,  Connections ... cons )
        {
            disconnect( con );
            disconnect( cons... );
        }

        void disconnect_all()
        {
            lock_guard_type l( mutex_ );
            registered_subscriber_.clear();
            registered_subscriber_blocked_.clear();
        }

        void operator()( Args... args ) const
        {
            lock_guard_type l( mutex_ );
            for ( auto const & i : registered_subscriber_ ) //invoking functions according to their priorities
                for ( auto const & j : i.second ) //if of same priority, invoking randomly
                    try { (j.second)( args... ); } //in case of bad function call, such like null ptr, just skip
                    catch( std::bad_function_call& bfc ) {}
        }

        void emit( Args... args ) const
        { operator()(args...); }

        template<typename Output_Iterator>
        void operator() ( Output_Iterator o, Args... args ) const
        {
            lock_guard_type l( mutex_ );
            for ( auto const & i : registered_subscriber_ )
                for ( auto const & j : i.second )
                    try { *o++ = (j.second)( args... ); }//redirect the return values to a stream
                    catch( std::bad_function_call& bfc ) {}
        }

        template<typename Output_Iterator>
        void emit ( Output_Iterator o, Args... args ) const
        { operator()(o, args...); }

        //block one connection
        bool block( const connection_type con )
        {
            lock_guard_type l( mutex_ );
            for ( auto i : registered_subscriber_ )
                for ( auto j : i.second )
                {
                    if ( j.first != con ) continue;
                    registered_subscriber_blocked_[i.first].insert( j ); //copy the connection to the blocked group
                    registered_subscriber_[i.first].erase( con );        //erase the connection from the normal group
                    return true;
                }
            return false;
        }

        template< typename ... Connections >
        bool block( const connection_type con, Connections ... cons )
        {
            bool flag1 = block( con );
            bool flag2 = block( cons ... );
            if ( flag1 && flag2 ) return true;
            return false;
        }

        void block_all()
        {
            lock_guard_type l( mutex_ );
            registered_subscriber_blocked_.insert( registered_subscriber_.begin(), registered_subscriber_.end() );
            registered_subscriber_.clear();
        }

        bool unblock( const connection_type con )
        {
            lock_guard_type l( mutex_ );
            for ( auto i : registered_subscriber_blocked_ )
                for ( auto j : i.second )
                {
                    if ( j.first != con ) continue;
                    registered_subscriber_[i.first].insert( j );         //copy the connection to the normal group
                    registered_subscriber_blocked_[i.first].erase( con );//erase the connection from the blocked group
                    return true;
                }
            return false;
        }

        template< typename ... Connections >
        bool unblock( const connection_type con, Connections ... cons )
        {
            bool flag1 = unblock( con );
            bool flag2 = unblock( cons ... );
            if ( flag1 && flag2 ) return true;
            return false;
        }

        void unblock_all()
        {
            lock_guard_type l( mutex_ );
            registered_subscriber_.insert( registered_subscriber_blocked_.begin(), registered_subscriber_blocked_.end() );
            registered_subscriber_blocked_.clear();
        }

        void swap( self_type& other )
        {
            std::lock( mutex_, other.mutex_ );
            lock_guard_type l1( mutex_, std::adopt_lock );
            lock_guard_type l2( other.mutex_, std::adopt_lock );
            std::swap( registered_subscriber_, other.registered_subscriber_ );                //swap the normal group members
            std::swap( registered_subscriber_blocked_, other.registered_subscriber_blocked_ );//swap the blocked group members
        }

    private:
        priority_connection_subscriber_type                     registered_subscriber_blocked_; //the blocked group
        priority_connection_subscriber_type                     registered_subscriber_;         //the normal group
        mutable mutex_type                                      mutex_;                         //mutex
    };//struct signal

    template< typename R, typename... Args >
    struct scope_connection
    {   //can accept more slots?
        typedef scope_connection                        self_type;
        typedef signal<R, Args...>                      signal_type;
        typedef typename signal_type::connection_type   connection_type;
        typedef typename signal_type::priority_type     priority_type;
        typedef typename signal_type::function_type     function_type;

        template<typename... F>
        scope_connection( const priority_type w, signal_type& sig,  const F&... f ) : sig_(sig)
        { con_ = sig_.connect( w, f...); }

        template<typename... F>
        scope_connection( signal_type& sig,  const F&... f ) : sig_(sig)
        { con_ = sig_.connect( std::numeric_limits<priority_type>::max(), f...); }

        const connection_type connection() const { return con_; }

        ~scope_connection() { sig_.disconnect(con_); }          //disconnect on leaving scope

        scope_connection( const self_type& ) = delete;          //no copy ctor
        scope_connection( self_type&& ) = delete;               //no copy ctor
        self_type& operator = ( const self_type& ) = delete;    //no assignment operator
        self_type& operator = ( self_type&& ) = delete;         //no assignment operator

        private:
        signal_type& sig_;
        connection_type con_;
    };

    template< typename R, typename... Arg >
    void swap( signal<R,Arg...>& lhs, signal<R,Arg...>& rhs )
    { lhs.swap( rhs ); }

    template< typename R, typename... Arg, typename... Connections >
    void disconnect( signal<R, Arg...>& sig, Connections... cons )
    { sig.disconnect( cons... ); }

    template<typename S, typename... F>
    connection_type connect( const typename S::priotity_type w, S& s, const F&... f )
    { return s.connect( w, f... ); }

    template<typename S, typename... F>
    connection_type connect(  S& s, const F&... f )
    { return connect( std::numeric_limits<typename S::priority_type>::max(), s, f... ); }

    template< typename R, typename... Args, typename... F >
    scope_connection<R, Args...> const scope_connect( signal<R, Args...>& sig, const F&... f )
    { return scope_connection<R,Args...>( sig, f... ); }

    template< typename R, typename... Args, typename... F >
    scope_connection<R, Args...> const scope_connect( typename signal<R,Args...>::priority_type const w, signal<R, Args...>& sig, const F&... f )
    { return scope_connection<R,Args...>( w, sig, f... ); }

}//namespace f

#endif//_SIGNAL_HPP_INCLUDED_SOIU498ASDFKLSFDLKJXCVKJDSLKFJASLKJ438721SDP230SFAJHEIOUFDJLKSAFLKSAFOIU3498YSFDUIWEIUOWOIUE

