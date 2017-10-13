#ifndef IPHTDLJWCFDTMIGYLJASWCVHAABLUXLOXRPIXOFBCJYMSRYLIMQXPPNSQPDUNURBDTUAPOYYA
#define IPHTDLJWCFDTMIGYLJASWCVHAABLUXLOXRPIXOFBCJYMSRYLIMQXPPNSQPDUNURBDTUAPOYYA

#include <f/matrix/matrix.hpp>
#include <f/derivative/derivative.hpp>
#include <f/derivative/second_derivative.hpp>

namespace f
{
    template<typename T, typename Zen>
    struct hessian_function
    {
        typedef T                                           value_type;
        typedef value_type*                                 pointer;
        typedef matrix<value_type>                          matrix_type;
        typedef Zen                                         zen_type;
        typedef std::size_t                                 size_type;

        void make_hessian()
        {
            zen_type & zen = static_cast<zen_type&>( *this );
            size_type const n = zen.unknown_variables;
            H.resize( n, n );
            for ( size_type r = 0; r != n; ++r )
                for ( size_type c = 0; c != n; ++c )
                {
                    H[r][c] = zen.setup_hessian( r, c );
                    H[c][r] = H[r][c];
                }
        }

        //the default hessian evaluator
        value_type setup_hessian( size_type offset_1, size_type offset_2 ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            auto const& ddf = make_second_derivative( zen, offset_1, offset_2 );
            auto const& ddf = make_second_derivative( [&](pointer x){ return zen.make_chi_square(x);}, offset_1, offset_2);
            return ddf( &((zen.fitting_array)[0]) );
        }

    };//struct hessian_function

    template<typename T, typename Zen>
    struct levenberg_marquardt_hessian_function : hessian_function< T, levenberg_marquardt_hessian_function<T, Zen> >
    {
        typedef T                                           value_type;
        typedef value_type*                                 pointer;
        typedef matrix<value_type>                          matrix_type;
        typedef Zen                                         zen_type;
        typedef std::size_t                                 size_type;

        value_type setup_hessian( size_type offset_1, size_type offset_2 ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            //TODO here
        }

    };//levenberg_marquardt_hessian_function

}//namespace f

#endif//IPHTDLJWCFDTMIGYLJASWCVHAABLUXLOXRPIXOFBCJYMSRYLIMQXPPNSQPDUNURBDTUAPOYYA

