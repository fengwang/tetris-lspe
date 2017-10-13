#ifndef IYBDUUXBPGIVBOUGHVCMBYDSUWFLBOMWDUVBVQLLHXSVUXVSFJXTRQEKUFGTYGPDYKDFTWJOQ
#define IYBDUUXBPGIVBOUGHVCMBYDSUWFLBOMWDUVBVQLLHXSVUXVSFJXTRQEKUFGTYGPDYKDFTWJOQ

#include <f/beam/make_ar.hpp>

#include <vector>
#include <numeric>

namespace f
{

    template< typename Refinement >
    struct ar_matrix
    {
        typedef Refinement      zen_type;
        matrix<unsigned long>   the_ar_matrix;

        void make_ar_matrix()
        {
            auto& zen = static_cast<zen_type&>(*this);
            
            std::vector<unsigned long> arr( zen.the_beam_matrix.row() );
            std::iota( arr.begin(), arr.end(), 0 );

            the_ar_matrix = make_ar( zen.the_beam_list, arr.begin(), arr.end() ):
        }
    
    };

}//namespace f

#endif//IYBDUUXBPGIVBOUGHVCMBYDSUWFLBOMWDUVBVQLLHXSVUXVSFJXTRQEKUFGTYGPDYKDFTWJOQ

