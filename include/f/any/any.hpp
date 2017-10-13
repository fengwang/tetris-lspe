#ifndef SXYCGHVGISDPEWRVMPSFINDGELAQCQWKNRMSQTRPBROTESISAUEKQNTHJHONQCOSQKBJRXMSO
#define SXYCGHVGISDPEWRVMPSFINDGELAQCQWKNRMSQTRPBROTESISAUEKQNTHJHONQCOSQKBJRXMSO

#include <type_traits>
#include <typeinfo>
#include <utility>
#include <memory>

namespace f
{

    namespace any_private_xxxlsdf
    {
        struct container_base
        {
            virtual ~container_base() {}
        };

        template<typename T>
        struct container : container_base
        {
            template<typename U>
            container( U&& value ) : value( std::forward<U>( value ) ) {}
            T value;
        };
    }

    struct any
    {
        using base_type = any_private_xxxlsdf::container_base;

        template<typename T>
        using storage_type = typename std::remove_cv<typename std::decay<T>::type>::type;

        template<typename T>
        using container_type = any_private_xxxlsdf::container<storage_type<T>>;

        template<typename U>
        any( U&& val ) : ptr( new container_type<U>{ std::forward<U>( val ) } )  {}

        template<typename U>
        inline bool is() const
        {
            auto container = std::dynamic_pointer_cast<container_type<U>>( ptr );
            if ( container ) return true;
            return false;
        }

        template<typename U>
        inline storage_type<U>& as() const
        {
            auto container = std::dynamic_pointer_cast<container_type<U>>( ptr );
            if ( !container ) throw std::bad_cast();
            return container->value;
        }

        template<typename  U>
        inline storage_type<U>& value() const
        {
            return as<storage_type<U>>();
        }

        std::shared_ptr<base_type> ptr;
    };

}//namespace f

#endif//SXYCGHVGISDPEWRVMPSFINDGELAQCQWKNRMSQTRPBROTESISAUEKQNTHJHONQCOSQKBJRXMSO

