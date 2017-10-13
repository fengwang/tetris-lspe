#ifndef BMHYCHTEBWBJGOARRVRREEMFBHMFQBHURKPVIJVLTCSSOBUEDIWXHQTFUVJGYKHOCXQMAYVMS
#define BMHYCHTEBWBJGOARRVRREEMFBHMFQBHURKPVIJVLTCSSOBUEDIWXHQTFUVJGYKHOCXQMAYVMS

#include <algorithm>

namespace f
{
    template< typename T, typename Zen >
    struct initial_guess
    {
        typedef T                                   value_type;
        typedef Zen                                 zen_type;

        void make_initial_guess()
        {
            auto& zen = static_cast<zen_type&>(*this);
            zen.setup_initial_guess();
        }

        void setup_initial_guess()
        {
            auto& zen = static_cast<zen_type&>(*this);
            std::fill( zen.fitting_array.begin(), zen.fitting_array.end(), value_type{} );
        }
    };

}//namespace f

#endif//BMHYCHTEBWBJGOARRVRREEMFBHMFQBHURKPVIJVLTCSSOBUEDIWXHQTFUVJGYKHOCXQMAYVMS

