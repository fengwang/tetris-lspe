#ifndef HTRDDDOSWSPJFSFQNPQQCWJPFEMIVMCHORKFCWVJOGHSFXLEFCDOBUKAEYSWBQVMDICMNGEIJ
#define HTRDDDOSWSPJFSFQNPQQCWJPFEMIVMCHORKFCWVJOGHSFXLEFCDOBUKAEYSWBQVMDICMNGEIJ

#include <iostream>
#include <iomanip>

namespace f
{
    template< typename T, typename Derived >
    struct symbol
    {
        typedef symbol                  self_type;
        typedef Derived                 zen_type;
        typedef T                       value_type;

        //operator T() const

        value_type eval() const
        {
            auto const& zen = static_cast<zen_type const&>(*this);
            return zen.eval();
        }

        friend bool operator < ( self_type const& lhs, self_type const& rhs )
        {
            auto const& llhs = static_cast<zen_type const&>(lhs);
            auto const& rrhs = static_cast<zen_type const&>(rhs);
            return llhs < rrhs;
        }

        friend bool operator == ( self_type const& lhs, self_type const& rhs )
        {
            auto const& llhs = static_cast<zen_type const&>(lhs);
            auto const& rrhs = static_cast<zen_type const&>(rhs);
            return llhs == rrhs;
        }

        friend std::ostream& operator << ( std::ostream& os, self_type const& rhs )
        {
            auto const& rrhs = static_cast<zen_type const&>(rhs);
            return os << rrhs;
        }

    };//struct symbol
/*
    //tiny_symbol is a kind of symbols with only a key[identity]
    template< typename T, typename IT >
    struct tiny_symbol : symbol<T, tiny_symbol<T, IT> >
    {
        typedef tiny_symbol<T, IT>      self_type;
        typedef IT                      identity_type;

        identity_type                   id_;

        tiny_symbol( identity_type const& id ) : id_(id) {}

        friend bool operator < ( self_type const& lhs, self_type const& rhs )
        {
            return lhs.id_ < rhs.id_;
        }

        friend bool operator == ( self_type const& lhs, self_type const& rhs )
        {
            return lhs.id_ == rhs.id_;
        }

        friend std::ostream& operator << ( std::ostream& os, self_type const& rhs )
        {
            return os << "[" << rhs.id_ << ")]";
        }

        ~tiny_symbol()
        {
        }

    };//struct tiny_symbol
*/

    //simple_symbol is a kind of symbols with key-value[id-value]
    template< typename T, typename IT >
    struct simple_symbol : symbol<T, simple_symbol<T, IT> >
    {
        typedef simple_symbol<T, IT>    self_type;
        typedef T                       value_type;
        typedef IT                      identity_type;

        T                               value_;
        identity_type                   id_;

        simple_symbol( value_type const value, identity_type const& id ) : value_(value), id_(id) {}

        value_type eval() const
        {
            return value_;
        }

        friend bool operator < ( self_type const& lhs, self_type const& rhs )
        {
            return lhs.id_ < rhs.id_;
        }

        friend bool operator == ( self_type const& lhs, self_type const& rhs )
        {
            return lhs.id_ == rhs.id_;
        }

        friend std::ostream& operator << ( std::ostream& os, self_type const& rhs )
        {
            return os << "[" << std::hex << std::showbase << rhs.id_ << "(" << rhs.value_ << ")]";
        }

        virtual ~simple_symbol()
        {
            //std::cout << "--" << id_ << "\t" << value_ << "\n";
        }

    };//struct simple_symbol

    //user defined symbol should have
    //
    //              value_type
    //              eval (optional)
    //              operator << (optional)
    //              is_less_than
    //              is_equal_to
    //
    //  defined.

}//namespace f

#endif//HTRDDDOSWSPJFSFQNPQQCWJPFEMIVMCHORKFCWVJOGHSFXLEFCDOBUKAEYSWBQVMDICMNGEIJ
