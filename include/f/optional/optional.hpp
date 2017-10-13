#ifndef OPTIONAL_HPP_INCLUDED_DSOIJALSKFJ349A8HFSUDLKAJSFDH4OIUHAFLKJHASDLKJAHF
#define OPTIONAL_HPP_INCLUDED_DSOIJALSKFJ349A8HFSUDLKAJSFDH4OIUHAFLKJHASDLKJAHF

#include <optional>


namespace f
{

	struct nothing_t
	{
		template< typename T >
		operator std::optional< T >() const noexcept
		{
			return std::optional< T >{};
		}
	};

	constexpr nothing_t nothing;

	template< typename T >
	auto just( T const& t ) noexcept
	{
		return std::optional< std::remove_cv_t< T > >{ t };
	}

	template< typename T >
	auto just( T&& t ) noexcept
	{
		return std::optional< std::remove_cv_t< T > >{std::forward< T&& >(t)};
	}

	template< typename T >
	auto just( std::optional<T> const& t ) noexcept
	{
		return t;
	}

	template< typename T >
	auto just( std::optional<T>&& t ) noexcept
	{
		return std::forward<std::optional<T>&&>( t );
	}

	template< typename Option, typename Function >
	auto operator | ( Option&& o, Function&& f ) noexcept
	-> decltype(std::forward< Function&& >(f)(*std::forward< Option&& >(o)))
	{
		if (o)
			return std::forward< Function&& >(f)(*std::forward< Option&& >(o) );

		return nothing;
	}

} //namespace f



#if 0

# include <utility>
# include <type_traits>
# include <initializer_list>
# include <cassert>
# include <functional>
# include <string>
# include <stdexcept>

namespace f
{
    template <class T> class optional;

    template <class T> class optional<T&>;

    template <class T> inline constexpr T&& constexpr_forward( typename std::remove_reference<T>::type& t ) noexcept
    {
        return static_cast< T&& >( t );
    }

    template <class T> inline constexpr T&& constexpr_forward( typename std::remove_reference<T>::type&& t ) noexcept
    {
        static_assert( !std::is_lvalue_reference<T>::value, "!!" );
        return static_cast< T&& >( t );
    }

    template <class T> inline constexpr typename std::remove_reference<T>::type&& constexpr_move( T&& t ) noexcept
    {
        return static_cast< typename std::remove_reference<T>::type&& >( t );
    }

    namespace detail_
    {

        template <typename T>
        struct has_overloaded_addressof
        {
            template <class X>
            constexpr static bool has_overload( ... ) { return false; }

            template <class X, size_t S = sizeof( std::declval<X&>().operator&() )>
            constexpr static bool has_overload( bool ) { return true; }

            constexpr static bool value = has_overload<T>( true );
        };

        template < typename T, typename std::enable_if < !has_overloaded_addressof<T>::value, bool >::type = false >
        constexpr T * static_addressof( T& ref )
        {
            return &ref;
        }

        template <typename T, typename std::enable_if<has_overloaded_addressof<T>::value, bool>::type = false>
        T * static_addressof( T& ref )
        {
            return std::addressof( ref );
        }

        template <class U>
        constexpr U convert( U v ) { return v; }

    }

    constexpr struct trivial_init_t {} trivial_init{};

    constexpr struct in_place_t {} in_place{};

    struct nullopt_t
    {
        struct init {};
        constexpr explicit nullopt_t( init ) {}
    };

    constexpr nullopt_t nullopt{nullopt_t::init()};

    class bad_optional_access : public std::logic_error
    {
        public:
            explicit bad_optional_access( const std::string& what_arg ) : std::logic_error{what_arg} {}
            explicit bad_optional_access( const char* what_arg ) : std::logic_error{what_arg} {}
    };

    template <class T>
    union storage_t
    {
        unsigned char dummy_;

        T value_;

        constexpr storage_t( trivial_init_t ) noexcept : dummy_() {}

        template <class... Args>
        constexpr storage_t( Args&& ... args ) : value_( constexpr_forward<Args>( args )... ) {}

        ~storage_t() {}
    };


    template <class T>
    union constexpr_storage_t
    {
        unsigned char dummy_;

        T value_;

        constexpr constexpr_storage_t( trivial_init_t ) noexcept : dummy_() {}

        template <class... Args>
        constexpr constexpr_storage_t( Args&& ... args ) : value_( constexpr_forward<Args>( args )... ) {}

        ~constexpr_storage_t() = default;
    };


    template <class T>
    struct optional_base
    {
        bool init_;

        storage_t<T> storage_;

        constexpr optional_base() noexcept : init_( false ), storage_( trivial_init ) {}

        explicit constexpr optional_base( const T& v ) : init_( true ), storage_( v ) {}

        explicit constexpr optional_base( T&& v ) : init_( true ), storage_( constexpr_move( v ) ) {}

        template <class... Args>
        explicit optional_base( in_place_t, Args&& ... args ) : init_( true ), storage_( constexpr_forward<Args>( args )... ) {}

        template <class U, class... Args, typename std::enable_if<std::is_constructible<T, std::initializer_list<U>>::value, bool>::type = false>
        explicit optional_base( in_place_t, std::initializer_list<U> il, Args && ... args ) : init_( true ), storage_( il, std::forward<Args>( args )... ) {}

        ~optional_base() { if ( init_ ) storage_.value_.T::~T(); }
    };


    template <class T>
    struct constexpr_optional_base
    {
        bool init_;

        constexpr_storage_t<T> storage_;

        constexpr constexpr_optional_base() noexcept : init_( false ), storage_( trivial_init ) {}

        explicit constexpr constexpr_optional_base( const T& v ) : init_( true ), storage_( v ) {}

        explicit constexpr constexpr_optional_base( T&& v ) : init_( true ), storage_( constexpr_move( v ) ) {}

        template <class... Args>
        explicit constexpr constexpr_optional_base( in_place_t, Args&& ... args ) : init_( true ), storage_( constexpr_forward<Args>( args )... ) {}

        template <class U, class... Args, typename std::enable_if<std::is_constructible<T, std::initializer_list<U>>::value, bool>::type = false>
        constexpr explicit constexpr_optional_base( in_place_t, std::initializer_list<U> il, Args && ... args ) : init_( true ), storage_( il, std::forward<Args>( args )... ) {}

        ~constexpr_optional_base() = default;
    };

    template <class T>
    using OptionalBase = typename std::conditional < std::is_trivially_destructible<T>::value, constexpr_optional_base<typename std::remove_const<T>::type>, optional_base<typename std::remove_const<T>::type> >::type;

    template <class T>
    class optional : private OptionalBase<T>
    {
            static_assert( !std::is_same<typename std::decay<T>::type, nullopt_t>::value, "bad T" );
            static_assert( !std::is_same<typename std::decay<T>::type, in_place_t>::value, "bad T" );

            constexpr bool initialized() const noexcept { return OptionalBase<T>::init_; }
            typename std::remove_const<T>::type* dataptr() { return std::addressof( OptionalBase<T>::storage_.value_ ); }
            constexpr const T* dataptr() const { return detail_::static_addressof( OptionalBase<T>::storage_.value_ ); }

            constexpr const T& contained_val() const& { return OptionalBase<T>::storage_.value_; }

            T& contained_val()& { return OptionalBase<T>::storage_.value_; }
            T&& contained_val()&& { return std::move( OptionalBase<T>::storage_.value_ ); }

            void clear() noexcept
            {
                if ( initialized() ) dataptr()->T::~T();

                OptionalBase<T>::init_ = false;
            }

            template <class... Args>
            void initialize( Args&& ... args ) noexcept( noexcept( T( std::forward<Args>( args )... ) ) )
            {
                assert( !OptionalBase<T>::init_ );
                ::new ( static_cast<void*>( dataptr() ) ) T( std::forward<Args>( args )... );
                OptionalBase<T>::init_ = true;
            }

            template <class U, class... Args>
            void initialize( std::initializer_list<U> il, Args&& ... args ) noexcept( noexcept( T( il, std::forward<Args>( args )... ) ) )
            {
                assert( !OptionalBase<T>::init_ );
                ::new ( static_cast<void*>( dataptr() ) ) T( il, std::forward<Args>( args )... );
                OptionalBase<T>::init_ = true;
            }

        public:
            typedef T value_type;

            constexpr optional() noexcept : OptionalBase<T>() {}

            constexpr optional( nullopt_t ) noexcept : OptionalBase<T>() {}

            optional( const optional& rhs ) : OptionalBase<T>()
            {
                if ( rhs.initialized() )
                {
                    ::new ( static_cast<void*>( dataptr() ) ) T( *rhs );
                    OptionalBase<T>::init_ = true;
                }
            }

            optional( optional&& rhs ) noexcept( std::is_nothrow_move_constructible<T>::value ) : OptionalBase<T>()
            {
                if ( rhs.initialized() )
                {
                    ::new ( static_cast<void*>( dataptr() ) ) T( std::move( *rhs ) );
                    OptionalBase<T>::init_ = true;
                }
            }

            constexpr optional( const T& v ) : OptionalBase<T>( v ) {}

            constexpr optional( T&& v ) : OptionalBase<T>( constexpr_move( v ) ) {}

            template <class... Args>
            explicit constexpr optional( in_place_t, Args&& ... args ) : OptionalBase<T>( in_place_t{}, constexpr_forward<Args>( args )... ) {}

            template <class U, class... Args, typename std::enable_if<std::is_constructible<T, std::initializer_list<U>>::value, bool>::type = false>
            constexpr explicit optional( in_place_t, std::initializer_list<U> il, Args && ... args ) : OptionalBase<T>( in_place_t{}, il, constexpr_forward<Args>( args )... ) {}


            ~optional() = default;

            optional& operator=( nullopt_t ) noexcept
            {
                clear();
                return *this;
            }

            optional& operator=( const optional& rhs )
            {
                if ( initialized() == true && rhs.initialized() == false ) clear();
                else if ( initialized() == false && rhs.initialized() == true ) initialize( *rhs );
                else if ( initialized() == true && rhs.initialized() == true ) contained_val() = *rhs;

                return *this;
            }

            optional& operator=( optional&& rhs )
            noexcept( std::is_nothrow_move_assignable<T>::value&& std::is_nothrow_move_constructible<T>::value )
            {
                if ( initialized() == true && rhs.initialized() == false ) clear();
                else if ( initialized() == false && rhs.initialized() == true ) initialize( std::move( *rhs ) );
                else if ( initialized() == true && rhs.initialized() == true ) contained_val() = std::move( *rhs );

                return *this;
            }

            template <class U>
            auto operator=( U&& v ) -> typename std::enable_if < std::is_same<typename std::decay<U>::type, T>::value, optional& >::type
            {
                if ( initialized() ) { contained_val() = std::forward<U>( v ); }
                else { initialize( std::forward<U>( v ) ); }
                return *this;
            }


            template <class... Args>
            void emplace( Args&& ... args )
            {
                clear();
                initialize( std::forward<Args>( args )... );
            }

            template <class U, class... Args>
            void emplace( std::initializer_list<U> il, Args&& ... args )
            {
                clear();
                initialize<U, Args...>( il, std::forward<Args>( args )... );
            }

            void swap( optional<T>& rhs ) noexcept( std::is_nothrow_move_constructible<T>::value&& noexcept( swap( std::declval<T&>(), std::declval<T&>() ) ) )
            {
                if ( initialized() == true && rhs.initialized() == false ) { rhs.initialize( std::move( **this ) ); clear(); }
                else if ( initialized() == false && rhs.initialized() == true ) { initialize( std::move( *rhs ) ); rhs.clear(); }
                else if ( initialized() == true && rhs.initialized() == true ) { using std::swap; swap( **this, *rhs ); }
            }

            explicit constexpr operator bool() const noexcept { return initialized(); }

            constexpr T const* operator ->() const
            {
                return ( ( initialized() ) ? ( dataptr() ) : ( [] {assert( !"initialized()" );}(), ( dataptr() ) ) );
            }

            T* operator ->()
            {
                assert ( initialized() );
                return dataptr();
            }

            constexpr T const& operator *() const
            {
                return ( ( initialized() ) ? ( contained_val() ) : ( [] {assert( !"initialized()" );}(), ( contained_val() ) ) );
            }

            T& operator *()
            {
                assert ( initialized() );
                return contained_val();
            }

            constexpr T const& value() const
            {
                return initialized() ? contained_val() : ( throw bad_optional_access( "bad optional access" ), contained_val() );
            }

            T& value()
            {
                return initialized() ? contained_val() : ( throw bad_optional_access( "bad optional access" ), contained_val() );
            }

            template <class V>
            constexpr T value_or( V&& v ) const&
            {
                return *this ? **this : detail_::convert<T>( constexpr_forward<V>( v ) );
            }

            template <class V>
            T value_or( V&& v )&&
            {
                return *this ? constexpr_move( const_cast<optional<T>&>( *this ).contained_val() ) : detail_::convert<T>( constexpr_forward<V>( v ) );
            }
    };


    template <class T>
    class optional<T&>
    {
            static_assert( !std::is_same<T, nullopt_t>::value, "bad T" );
            static_assert( !std::is_same<T, in_place_t>::value, "bad T" );
            T* ref;

        public:

            constexpr optional() noexcept : ref( nullptr ) {}

            constexpr optional( nullopt_t ) noexcept : ref( nullptr ) {}

            constexpr optional( T& v ) noexcept : ref( detail_::static_addressof( v ) ) {}

            optional( T&& ) = delete;

            constexpr optional( const optional& rhs ) noexcept : ref( rhs.ref ) {}

            explicit constexpr optional( in_place_t, T& v ) noexcept : ref( detail_::static_addressof( v ) ) {}

            explicit optional( in_place_t, T&& ) = delete;

            ~optional() = default;

            optional& operator=( nullopt_t ) noexcept
            {
                ref = nullptr;
                return *this;
            }

            template <typename U>
            auto operator=( U&& rhs ) noexcept -> typename std::enable_if < std::is_same<typename std::decay<U>::type, optional<T&>>::value, optional& >::type
            {
                ref = rhs.ref;
                return *this;
            }

            template <typename U>
            auto operator=( U&& rhs ) noexcept -> typename std::enable_if < !std::is_same<typename std::decay<U>::type, optional<T&>>::value, optional& >::type = delete;

            void emplace( T& v ) noexcept
            {
                ref = detail_::static_addressof( v );
            }

            void emplace( T&& ) = delete;


            void swap( optional<T&>& rhs ) noexcept
            {
                std::swap( ref, rhs.ref );
            }


            constexpr T* operator->() const
            {
                return ( ( ref ) ? ( ref ) : ( [] {assert( !"ref" );}(), ( ref ) ) );
            }

            constexpr T& operator*() const
            {
                return ( ( ref ) ? ( *ref ) : ( [] {assert( !"ref" );}(), ( *ref ) ) );
            }

            constexpr T& value() const
            {
                return ref ? *ref : ( throw bad_optional_access( "bad optional access" ), *ref );
            }

            explicit constexpr operator bool() const noexcept
            {
                return ref != nullptr;
            }

            template <class V>
            constexpr typename std::decay<T>::type value_or( V&& v ) const
            {
                return *this ? **this : detail_::convert<typename std::decay<T>::type>( constexpr_forward<V>( v ) );
            }
    };


    template <class T>
    class optional < T&& >
    {
            static_assert( sizeof( T ) == 0, "optional rvalue references disallowed" );
    };

    template <class T> constexpr bool operator==( const optional<T>& x, const optional<T>& y )
    {
        return bool( x ) != bool( y ) ? false : bool( x ) == false ? true : *x == *y;
    }

    template <class T> constexpr bool operator!=( const optional<T>& x, const optional<T>& y )
    {
        return !( x == y );
    }

    template <class T> constexpr bool operator<( const optional<T>& x, const optional<T>& y )
    {
        return ( !y ) ? false : ( !x ) ? true : *x < *y;
    }

    template <class T> constexpr bool operator>( const optional<T>& x, const optional<T>& y )
    {
        return ( y < x );
    }

    template <class T> constexpr bool operator<=( const optional<T>& x, const optional<T>& y )
    {
        return !( y < x );
    }

    template <class T> constexpr bool operator>=( const optional<T>& x, const optional<T>& y )
    {
        return !( x < y );
    }

    template <class T> constexpr bool operator==( const optional<T>& x, nullopt_t ) noexcept
    {
        return ( !x );
    }

    template <class T> constexpr bool operator==( nullopt_t, const optional<T>& x ) noexcept
    {
        return ( !x );
    }

    template <class T> constexpr bool operator!=( const optional<T>& x, nullopt_t ) noexcept
    {
        return bool( x );
    }

    template <class T> constexpr bool operator!=( nullopt_t, const optional<T>& x ) noexcept
    {
        return bool( x );
    }

    template <class T> constexpr bool operator<( const optional<T>&, nullopt_t ) noexcept
    {
        return false;
    }

    template <class T> constexpr bool operator<( nullopt_t, const optional<T>& x ) noexcept
    {
        return bool( x );
    }

    template <class T> constexpr bool operator<=( const optional<T>& x, nullopt_t ) noexcept
    {
        return ( !x );
    }

    template <class T> constexpr bool operator<=( nullopt_t, const optional<T>& ) noexcept
    {
        return true;
    }

    template <class T> constexpr bool operator>( const optional<T>& x, nullopt_t ) noexcept
    {
        return bool( x );
    }

    template <class T> constexpr bool operator>( nullopt_t, const optional<T>& ) noexcept
    {
        return false;
    }

    template <class T> constexpr bool operator>=( const optional<T>&, nullopt_t ) noexcept
    {
        return true;
    }

    template <class T> constexpr bool operator>=( nullopt_t, const optional<T>& x ) noexcept
    {
        return ( !x );
    }

    template <class T> constexpr bool operator==( const optional<T>& x, const T& v )
    {
        return bool( x ) ? *x == v : false;
    }

    template <class T> constexpr bool operator==( const T& v, const optional<T>& x )
    {
        return bool( x ) ? v == *x : false;
    }

    template <class T> constexpr bool operator!=( const optional<T>& x, const T& v )
    {
        return bool( x ) ? *x != v : true;
    }

    template <class T> constexpr bool operator!=( const T& v, const optional<T>& x )
    {
        return bool( x ) ? v != *x : true;
    }

    template <class T> constexpr bool operator<( const optional<T>& x, const T& v )
    {
        return bool( x ) ? *x < v : true;
    }

    template <class T> constexpr bool operator>( const T& v, const optional<T>& x )
    {
        return bool( x ) ? v > *x : true;
    }

    template <class T> constexpr bool operator>( const optional<T>& x, const T& v )
    {
        return bool( x ) ? *x > v : false;
    }

    template <class T> constexpr bool operator<( const T& v, const optional<T>& x )
    {
        return bool( x ) ? v < *x : false;
    }

    template <class T> constexpr bool operator>=( const optional<T>& x, const T& v )
    {
        return bool( x ) ? *x >= v : false;
    }

    template <class T> constexpr bool operator<=( const T& v, const optional<T>& x )
    {
        return bool( x ) ? v <= *x : false;
    }

    template <class T> constexpr bool operator<=( const optional<T>& x, const T& v )
    {
        return bool( x ) ? *x <= v : true;
    }

    template <class T> constexpr bool operator>=( const T& v, const optional<T>& x )
    {
        return bool( x ) ? v >= *x : true;
    }

    template <class T> constexpr bool operator==( const optional<T&>& x, const T& v )
    {
        return bool( x ) ? *x == v : false;
    }

    template <class T> constexpr bool operator==( const T& v, const optional<T&>& x )
    {
        return bool( x ) ? v == *x : false;
    }

    template <class T> constexpr bool operator!=( const optional<T&>& x, const T& v )
    {
        return bool( x ) ? *x != v : true;
    }

    template <class T> constexpr bool operator!=( const T& v, const optional<T&>& x )
    {
        return bool( x ) ? v != *x : true;
    }

    template <class T> constexpr bool operator<( const optional<T&>& x, const T& v )
    {
        return bool( x ) ? *x < v : true;
    }

    template <class T> constexpr bool operator>( const T& v, const optional<T&>& x )
    {
        return bool( x ) ? v > *x : true;
    }

    template <class T> constexpr bool operator>( const optional<T&>& x, const T& v )
    {
        return bool( x ) ? *x > v : false;
    }

    template <class T> constexpr bool operator<( const T& v, const optional<T&>& x )
    {
        return bool( x ) ? v < *x : false;
    }

    template <class T> constexpr bool operator>=( const optional<T&>& x, const T& v )
    {
        return bool( x ) ? *x >= v : false;
    }

    template <class T> constexpr bool operator<=( const T& v, const optional<T&>& x )
    {
        return bool( x ) ? v <= *x : false;
    }

    template <class T> constexpr bool operator<=( const optional<T&>& x, const T& v )
    {
        return bool( x ) ? *x <= v : true;
    }

    template <class T> constexpr bool operator>=( const T& v, const optional<T&>& x )
    {
        return bool( x ) ? v >= *x : true;
    }

    template <class T> constexpr bool operator==( const optional<const T&>& x, const T& v )
    {
        return bool( x ) ? *x == v : false;
    }

    template <class T> constexpr bool operator==( const T& v, const optional<const T&>& x )
    {
        return bool( x ) ? v == *x : false;
    }

    template <class T> constexpr bool operator!=( const optional<const T&>& x, const T& v )
    {
        return bool( x ) ? *x != v : true;
    }

    template <class T> constexpr bool operator!=( const T& v, const optional<const T&>& x )
    {
        return bool( x ) ? v != *x : true;
    }

    template <class T> constexpr bool operator<( const optional<const T&>& x, const T& v )
    {
        return bool( x ) ? *x < v : true;
    }

    template <class T> constexpr bool operator>( const T& v, const optional<const T&>& x )
    {
        return bool( x ) ? v > *x : true;
    }

    template <class T> constexpr bool operator>( const optional<const T&>& x, const T& v )
    {
        return bool( x ) ? *x > v : false;
    }

    template <class T> constexpr bool operator<( const T& v, const optional<const T&>& x )
    {
        return bool( x ) ? v < *x : false;
    }

    template <class T> constexpr bool operator>=( const optional<const T&>& x, const T& v )
    {
        return bool( x ) ? *x >= v : false;
    }

    template <class T> constexpr bool operator<=( const T& v, const optional<const T&>& x )
    {
        return bool( x ) ? v <= *x : false;
    }

    template <class T> constexpr bool operator<=( const optional<const T&>& x, const T& v )
    {
        return bool( x ) ? *x <= v : true;
    }

    template <class T> constexpr bool operator>=( const T& v, const optional<const T&>& x )
    {
        return bool( x ) ? v >= *x : true;
    }

    template <class T>
    void swap( optional<T>& x, optional<T>& y ) noexcept( noexcept( x.swap( y ) ) )
    {
        x.swap( y );
    }

    template <class T>
    constexpr optional<typename std::decay<T>::type> make_optional( T&& v )
    {
        return optional<typename std::decay<T>::type>( constexpr_forward<T>( v ) );
    }

    template <class X>
    constexpr optional<X&> make_optional( std::reference_wrapper<X> v )
    {
        return optional<X&>( v.get() );
    }

}

namespace std
{
    template <typename T>
    struct hash<f::optional<T>>
    {
        typedef typename hash<T>::result_type result_type;
        typedef f::optional<T> argument_type;

        constexpr result_type operator()( argument_type const& arg ) const
        {
            return arg ? std::hash<T> {}( *arg ) : result_type{};
        }
    };

    template <typename T>
    struct hash<f::optional<T&>>
    {
        typedef typename hash<T>::result_type result_type;
        typedef f::optional<T&> argument_type;

        constexpr result_type operator()( argument_type const& arg ) const
        {
            return arg ? std::hash<T> {}( *arg ) : result_type{};
        }
    };
}

#endif

#endif//OPTIONAL_HPP_INCLUDED_DSOIJALSKFJ349A8HFSUDLKAJSFDH4OIUHAFLKJHASDLKJAHF

