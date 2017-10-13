#ifndef DWJUWOBYHOTAGIAAHOTWEGPBLECMWYVTUVGJIDGRUCQRVQEVWSGTVWIICNONGWSPSRIOMJUKQ
#define DWJUWOBYHOTAGIAAHOTWEGPBLECMWYVTUVGJIDGRUCQRVQEVWSGTVWIICNONGWSPSRIOMJUKQ

namespace f
{

    namespace cuda
    {

        int get()
        {
            int get_device();
            return get_device();
        }

        void set( int id )
        {
            void set_device( int );

            int current_id = get();
            if ( current_id != id )
                set_device( id );
        }

        void reset()
        {
            void reset_device();
            reset_device();
        }

        void synchronize()
        {
            void synchronize_device();
            synchronize_device();
        }

        template< typename T >
        T* allocate( unsigned long n )
        {
            void cuda_allocate( void** p, unsigned long n );
            T* ans;
            cuda_allocate( reinterpret_cast<void**>(&ans), n*sizeof(T) );
            return ans;
        }

        template< typename T >
        void deallocate( T* ptr )
        {
            void cuda_deallocate( void* p );
            cuda_deallocate( reinterpret_cast<void*>(ptr) );
        }

        template< typename T >
        void host_to_device( T* host_begin, T* host_end, T* device_begin )
        {
            void cuda_memcopy_host_to_device( const void* src, unsigned long n, void* dst );
            unsigned long const n = sizeof(T)*(host_end-host_begin);
            cuda_memcopy_host_to_device( reinterpret_cast<const void*>(host_begin), n, reinterpret_cast<void*>(device_begin) );
        }

        template< typename T >
        void device_to_host( T* device_begin, T* device_end, T* host_begin )
        {
            void cuda_memcopy_device_to_host( const void* src, unsigned long n, void* dst );
            unsigned long const n = sizeof(T)*(device_end-device_begin);
            cuda_memcopy_device_to_host( reinterpret_cast<const void*>(device_begin), n, reinterpret_cast<void*>(host_begin) );
        }

    }

}//namespace f

#endif//DWJUWOBYHOTAGIAAHOTWEGPBLECMWYVTUVGJIDGRUCQRVQEVWSGTVWIICNONGWSPSRIOMJUKQ

