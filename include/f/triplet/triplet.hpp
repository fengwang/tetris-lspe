#ifndef EGLAGEFNUIPKVQJHFIYMNUYHIWFMRIWNDVDWEJBMVYMEAKOKWGKEYBVPDOWJYVMILQWQPXRTL
#define EGLAGEFNUIPKVQJHFIYMNUYHIWFMRIWNDVDWEJBMVYMEAKOKWGKEYBVPDOWJYVMILQWQPXRTL

#include <iostream>

namespace f
{
    template<typename T1, typename T2, typename T3>
    struct triplet
    {
        typedef T1      first_type;
        typedef T2      second_type;
        typedef T3      third_type;

        first_type      first;
        second_type     second;
        third_type      third;
    };

    template<typename T1, typename T2, typename T3>
    std::ostream& operator << ( std::ostream& os, triplet<T1, T2, T3> const& rhs )
    {
        return os << '(' << rhs.first << ',' << rhs.second << ',' << rhs.third << ')';
    }

    template<typename CharT, typename Traits, typename T1, typename T2, typename T3>
    std::basic_ostream<CharT, Traits>& operator << ( std::basic_ostream<CharT, Traits>& os, triplet<T1, T2, T3> const& rhs )
    {
        std::basic_ostream<CharT, Traits> os_;
        os_.flags( os.flags() );
        os_.imbue( os.getloc() );
        os_.precision( os.precision() );
        os_ << '(' << rhs.first << ',' << rhs.second << ',' << rhs.third << ')';
        return os << os_.str();
    }

    template<typename CharT, typename Traits, typename T1, typename T2, typename T3>
    std::basic_istream<CharT, Traits>& operator >> ( std::basic_istream<CharT, Traits>& is, triplet<T1, T2, T3>& rhs )
    {
        T1 first_;
        T2 second_;
        T3 third_;
        CharT ch;

        is >> ch;
        if ( ch == '(' )
        {
            is >> first_ >> ch;
            if ( ch == ',' )
            {
                is >> second_ >> ch;
                if ( ch == ',' )
                {
                    is >> third_ >> ch;
                    if ( ch == ')' )
                    {   //everything's ok, put to result
                        rhs.first = first_;
                        rhs.second = second_;
                        rhs.third = third_;
                        return is; //success case
                    }
                }
            }
        }

        is.setstate( std::ios_base::failbit );
        return is; //failure case
    }

    template<typename T1, typename T2, typename T3>
    triplet<T1, T2, T3> const make_triplet( T1 const& first_, T2 const& second_, T3 const& third_ )
    {
        return triplet<T1, T2, T3> { first_, second_, third_ };
    }

    template<typename T1, typename T2, typename T3>
    bool operator == ( const triplet<T1, T2, T3>& lhs, const triplet<T1, T2, T3>& rhs )
    {
        return lhs.first == rhs.first && lhs.second == rhs.second && lhs.third == rhs.third;
    }

    template<typename T1, typename T2, typename T3>
    bool operator != ( const triplet<T1, T2, T3>& lhs, const triplet<T1, T2, T3>& rhs )
    {
        return lhs.first != rhs.first || lhs.second != rhs.second || lhs.third != rhs.third;
    }

    template<typename T1, typename T2, typename T3>
    bool operator > ( const triplet<T1, T2, T3>& lhs, const triplet<T1, T2, T3>& rhs )
    {
        if ( lhs.first > rhs.first ) return true;
        if ( lhs.first < rhs.first ) return false;
        if ( lhs.second > rhs.second ) return true;
        if ( lhs.second < rhs.second ) return false;
        if ( lhs.third > rhs.third ) return true;
        return false;
    }

    template<typename T1, typename T2, typename T3>
    bool operator < ( const triplet<T1, T2, T3>& lhs, const triplet<T1, T2, T3>& rhs )
    {
        if ( lhs.first < rhs.first ) return true;
        if ( lhs.first > rhs.first ) return false;
        if ( lhs.second < rhs.second ) return true;
        if ( lhs.second > rhs.second ) return false;
        if ( lhs.third < rhs.third ) return true;
        return false;
    }

    template<typename T1, typename T2, typename T3>
    bool operator >= ( const triplet<T1, T2, T3>& lhs, const triplet<T1, T2, T3>& rhs )
    {
        return lhs > rhs || lhs == rhs;
    }

    template<typename T1, typename T2, typename T3>
    bool operator <= ( const triplet<T1, T2, T3>& lhs, const triplet<T1, T2, T3>& rhs )
    {
        return lhs < rhs || lhs == rhs;
    }

}//namespace f

#endif//EGLAGEFNUIPKVQJHFIYMNUYHIWFMRIWNDVDWEJBMVYMEAKOKWGKEYBVPDOWJYVMILQWQPXRTL
