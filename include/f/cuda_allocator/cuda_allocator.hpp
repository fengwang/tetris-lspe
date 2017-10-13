#ifndef PRDWNTBBOFDAMMJGMXURUYWPHYBPVCLMCWEYOXIPAQNVXMSPNMCECTRCPGRNEDSXEMLPNWMVJ
#define PRDWNTBBOFDAMMJGMXURUYWPHYBPVCLMCWEYOXIPAQNVXMSPNMCECTRCPGRNEDSXEMLPNWMVJ

#include <f/cuda/cuda.hpp>

#include <cstddef>
#include <type_traits>
#include <memory>
#include <utility>
#include <optional>

namespace f
{

    template< typename T >
    struct cuda_allocator;

    template<>
    struct cuda_allocator<void>
    {
        typedef void value_type;
        typedef void* pointer;
        typedef const void* const_pointer;
        template< typename U >
        struct rebind
        {
            typedef cuda_allocator<U> other;
        };
    };

    template< typename T >
    struct cuda_allocator
    {
        typedef std::size_t             size_type;
        typedef std::ptrdiff_t          difference_type;
        typedef T                       value_type;
        typedef value_type*             pointer;
        typedef const value_type*       const_pointer;
        typedef value_type&             reference;
        typedef const value_type&       const_reference;

        template< typename U >
        struct rebind
        {
            typedef cuda_allocator<T> other;
        };

        typedef std::true_type          propagate_on_container_move_assignment;
        typedef std::true_type          is_always_equal;

        typedef cuda_allocator          self_type;
        typedef std::optional<int>      device_id_type;

        device_id_type                  device_id;

        cuda_allocator() noexcept {}
        cuda_allocator( int device_id_ ) noexcept : device_id( device_id_ ) {}

        template< typename U >
        cuda_allocator( cuda_allocator<U> const& other ) noexcept : device_id( other.device_id ) {}

        pointer allocate( size_type const size, cuda_allocator<void>::const_pointer hint = 0 )
        {
            if ( device_id )
                cuda::set( device_id.value() );

            return cuda::allocate<value_type>( size );
        }

        void deallocate( pointer p, size_type  )
        {
            if ( device_id )
                cuda::set( device_id.value() );

            cuda::deallocate( p );
        }

        // ---- methods below are deprecated

        size_type max_size() const noexcept
        {
            return -1;
        }

        void* operator new( size_type length, void* ptr )
        {
            return ptr;
        }

        template< typename U, typename ... Args >
        void construct( U* p, Args&&... args )
        {
            new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
        }

        template< typename U >
        void destroy( U* p )
        {
            p -> ~U();
        }

        pointer address( reference x ) const noexcept
        {
            return std::addressof( x );
        }

        const_pointer address( const_reference x ) const noexcept
        {
            return std::addressof( x );
        }
    };

}//namespace f

#endif//PRDWNTBBOFDAMMJGMXURUYWPHYBPVCLMCWEYOXIPAQNVXMSPNMCECTRCPGRNEDSXEMLPNWMVJ

