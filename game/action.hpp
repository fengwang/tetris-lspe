#ifndef WHFLBROFJWNXWGWOBLXXRGPYHVRYXLXGSYGVSBIBUNUIPVPAGRIIXRUPNGFFYMWHBYDOQAYBE
#define WHFLBROFJWNXWGWOBLXXRGPYHVRYXLXGSYGVSBIBUNUIPVPAGRIIXRUPNGFFYMWHBYDOQAYBE

#include "./rotation.hpp"

#include <cassert>
#include <iostream>

struct action
{
    rotation    the_rotation;
    int         column;

    action( rotation r_, int col_ ) : the_rotation(r_), column(col_)
    {
        assert( (col_ >= 0) && "column cannot be negative!" );
    }

    action() : the_rotation( NONE ), column{ -1 } {}
};

bool operator == ( action const& lhs, action const& rhs )
{
    return (lhs.the_rotation == rhs.the_rotation) && (lhs.column == rhs.column);
}

std::ostream& operator << ( std::ostream& os, action const& a )
{
    os << "( ";

    switch (a.the_rotation)
    {
        case NONE:
            os << "None";
            break;
        case CLOCKWISE:
            os << "Clockwise";
            break;
        case COUNTER_CLOCKWISE:
            os << "Counter Clockwise";
            break;
        case FLIP:
            os << "Flip";
            break;
        default:
            os << "Unknown";
    }

    os << ", " << a.column << " )";

    return os;
}

#endif//WHFLBROFJWNXWGWOBLXXRGPYHVRYXLXGSYGVSBIBUNUIPVPAGRIIXRUPNGFFYMWHBYDOQAYBE

