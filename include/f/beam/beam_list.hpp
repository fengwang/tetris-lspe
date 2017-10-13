#ifndef YFFVVBCAEMBHPMNKUOBXIMOWJHUXJYNUXQHDMTDDBUJQBPAWEFUMGCWEGNNSNHVAKGXYKXJXJ
#define YFFVVBCAEMBHPMNKUOBXIMOWJHUXJYNUXQHDMTDDBUJQBPAWEFUMGCWEGNNSNHVAKGXYKXJXJ

#include "beam.hpp"

#include <f/matrix/matrix.hpp>

#include <map>
#include <string>
#include <iterator>
#include <algorithm>
#include <cassert>

namespace f
{

    struct beam_list
    {
        std::map<beam, unsigned long> beam_index_map;
        std::map<unsigned long, beam> index_beam_map;
    };


    //only 500+ beams, no threshold here
    // auto bl = make_beam_list( "testdata/working/new_beam.txt" );
    inline beam_list make_beam_list( std::string const& beam_path )
    {
        matrix<int> beam_mat;

        beam_mat.load( beam_path );

        assert( beam_mat.row() );
        assert( beam_mat.col() >= 4 );

        beam_list bl;

        for ( unsigned long r = 0; r != beam_mat.row(); ++r )
        {
            unsigned long const the_index = beam_mat[r][0];
            beam const the_beam{ beam_mat[r][1], beam_mat[r][2], beam_mat[r][3] };
            (bl.beam_index_map)[the_beam] = the_index;
            (bl.index_beam_map)[the_index] = the_beam;
        }

        assert( bl.beam_index_map.size() == bl.index_beam_map.size() );

        return bl;
    }
/*
    template< typename Itor >
    inline matrix<unsigned long> const make_ar( beam_list const& bl, Itor first, Itor last )
    {
        unsigned long dim = std::distance( first, last );
        assert( dim );

        matrix<unsigned long> ar{ dim, dim };
        std::fill( ar.begin(), ar.end(), 0 );        

        unsigned long const max_in_beam = *std::max_element(first, last);
        assert( max_in_beam <= bl.beam_index_map.size() );

        std::copy( first, last, ar.col_begin(0) );

        for ( unsigned long r= 0; r != ar.row(); ++r )
            for ( unsigned long c=1; c != ar.col(); ++c )
            {
                beam const new_beam = bl.index_beam_map[ar[r][0]] - bl.index_beam_map[ar[c][0]]; 
                assert( bl.beam_index_map.find(new_beam) != bl.beam_index_map.end() );
                ar[r][c] = bl.beam_index_map[new_beam];
            }

        return ar;
    }
*/

}//namespace f

#endif//YFFVVBCAEMBHPMNKUOBXIMOWJHUXJYNUXQHDMTDDBUJQBPAWEFUMGCWEGNNSNHVAKGXYKXJXJ

