#ifndef SWUIANQRNHQBKMVKMQNGXUATRVYEULBSABVRQKVBSNQTPKVWAQCQKGKYWTLHVGVMOCKPVLNPQ
#define SWUIANQRNHQBKMVKMQNGXUATRVYEULBSABVRQKVBSNQTPKVWAQCQKGKYWTLHVGVMOCKPVLNPQ

namespace f
{

    template<typename Result_Type, typename ...Args>
    struct abstract_function
    {
        virtual Result_Type operator()( Args... args ) = 0;
        virtual abstract_function* clone() const = 0;
        virtual ~abstract_function() = default;
    };

    template<typename Function, typename Result_Type, typename ...Args>
    struct concrete_function: public abstract_function<Result_Type, Args...>
    {
        Function func;

        concrete_function( const Function& x ) : func( x ) {}

        Result_Type operator()( Args... args ) override
        {
            return func( args... );
        }

        concrete_function* clone() const override
        {
            return new concrete_function{func};
        }
    };

    template<typename Function>
    struct func_filter
    {
        typedef Function type;
    };

    template<typename Result_Type, typename ...Args>
    struct func_filter<Result_Type( Args... )>
    {
        typedef Result_Type ( *type )( Args... );
    };

    template<typename Dummy>
    struct function;

    template<typename Result_Type, typename ...Args>
    struct function<Result_Type( Args... )>
    {
        typedef function                                    self_type;
        typedef abstract_function<Result_Type, Args...>*    abstract_function_pointer_type;;

        abstract_function_pointer_type f_ptr;

        function() : f_ptr( nullptr ) {}

        template<typename Function>
        function( const Function& x ) : f_ptr( new concrete_function<typename func_filter<Function>::type, Result_Type, Args...>( x ) ) {}

        function( const self_type& rhs ) : f_ptr( rhs.f_ptr ? rhs.f_ptr->clone() : nullptr ) {}

        self_type& operator=( const self_type& rhs )
        {
            if ( ( &rhs != this ) && ( rhs.f_ptr ) )
            {
                if ( f_ptr ) 
                    delete f_ptr;

                f_ptr = rhs.f_ptr->clone();
            }
            return *this;
        }

        template<typename Function>
        self_type& operator=( const Function& x )
        {
            if ( f_ptr ) delete f_ptr;
            f_ptr = new concrete_function<typename func_filter<Function>::type, Result_Type, Args...>( x );
            return *this;
        }

        Result_Type operator()( Args... args )
        {
            if ( f_ptr ) 
                return ( *f_ptr )( args... );

            //return Result_Type{};
        }

        void swap( self_type& other )
        {
            abstract_function_pointer_type f_cache = other.f_ptr;
            other.f_ptr = ( *this ).f_ptr;
            ( *this ).f_ptr = f_cache;
        }

        explicit operator bool() const
        {
            if ( ( *this ).f_ptr )
                return true;
            return false;
        }

        ~function() { delete f_ptr; }
        
    };//struct function

    template< typename Result_Type, typename... Arguments_Type >
    void swap( function<Result_Type(Arguments_Type...)>& lhs, function<Result_Type(Arguments_Type...)>& rhs )
    {
        lhs.swap( rhs );
    }

    template< typename Result_Type, typename... Arguments_Type >
    bool operator == ( function<Result_Type(Arguments_Type...)>& lhs, function<Result_Type(Arguments_Type...)>& rhs )
    {
        return lhs.f_ptr == rhs.f_ptr;
    }

    template< typename Result_Type, typename... Arguments_Type >
    bool operator != ( function<Result_Type(Arguments_Type...)>& lhs, function<Result_Type(Arguments_Type...)>& rhs )
    {
        return !(lhs == rhs);
    }

}//namespace f

#endif//SWUIANQRNHQBKMVKMQNGXUATRVYEULBSABVRQKVBSNQTPKVWAQCQKGKYWTLHVGVMOCKPVLNPQ
