#ifndef JOQCPHIALIWLUPTFVGTNNRUGSVWRQEQJOGEDUSSHTGUBETTTSMHDPHOYAJWMLOGEOPQSLAKCM
#define JOQCPHIALIWLUPTFVGTNNRUGSVWRQEQJOGEDUSSHTGUBETTTSMHDPHOYAJWMLOGEOPQSLAKCM

#include <ostream>
#include <sstream>
#include <iterator>
#include <iomanip>

namespace f
{
    template<typename Matrix, typename T>
    struct parasite_output_operator
    {
        typedef Matrix                                                          zen_type;
        typedef T                                                               value_type;
        typedef unsigned long                                                   size_type;

        template< typename Char, typename Traits >
        friend std::basic_ostream<Char, Traits>& operator << ( std::basic_ostream<Char, Traits>& lhs, zen_type const& rhs )
        {
            std::basic_ostringstream<Char, Traits> bos;
            bos.flags( lhs.flags() );
            bos.imbue( lhs.getloc() );
            bos.precision( lhs.precision() );

            for ( size_type i = 0; i < rhs.row; ++i )
            {
                std::copy( rhs.row_begin( i ), rhs.row_end( i ), std::ostream_iterator<value_type> ( bos, "\t" ) );
                bos << "\n";
            }
            return lhs << bos.str();

        }

    };//struct

}

#endif//JOQCPHIALIWLUPTFVGTNNRUGSVWRQEQJOGEDUSSHTGUBETTTSMHDPHOYAJWMLOGEOPQSLAKCM

