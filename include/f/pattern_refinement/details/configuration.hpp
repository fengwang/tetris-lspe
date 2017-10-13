#ifndef TVWTGFKSBJIDYIYTEHMLNBOFRTEQNEJHWITQAOWXGARGEWBCXWQTYFBLXNAPFVUCCHLSDBRCI
#define TVWTGFKSBJIDYIYTEHMLNBOFRTEQNEJHWITQAOWXGARGEWBCXWQTYFBLXNAPFVUCCHLSDBRCI

#include "../pattern_refinement_configuration.hpp"

#include <string>

namespace f
{

    template< typename Refinement >
    struct configuration
    {

        pattern_refinement_configuration the_configuration;

        void load_configuration( std::string const& path )
        {
            the_configuration.load( path );
        }

        void save_configuration( std::string const& path )
        {
            the_configuration.save_as( path );
        }
    
    };

}//namespace f

#endif//TVWTGFKSBJIDYIYTEHMLNBOFRTEQNEJHWITQAOWXGARGEWBCXWQTYFBLXNAPFVUCCHLSDBRCI

