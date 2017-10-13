#ifndef JVCIIOOILOJOWAOMPVEROMIMRXPVNPPLPXMRTNERCNBREVHGBDJIQWUXGEUPUBDTLUNEVRFJU
#define JVCIIOOILOJOWAOMPVEROMIMRXPVNPPLPXMRTNERCNBREVHGBDJIQWUXGEUPUBDTLUNEVRFJU

#include "beam_list.hpp"

namespace f
{

    template< typename Itor >
    inline matrix<unsigned long> const make_ar( beam_list& bl, Itor first, Itor last )
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
                if( bl.beam_index_map.find(new_beam) != bl.beam_index_map.end() )
                {
                    ar[r][c] = bl.beam_index_map[new_beam];
                    continue;
                }

                assert( bl.beam_index_map.size() == bl.index_beam_map.size() );

                unsigned long new_index = bl.beam_index_map.size();

                bl.beam_index_map[new_beam] = new_index;
                bl.index_beam_map[new_index] = new_beam;

                ar[r][c] = new_index;
            }

        return ar;
    }

}//namespace f

#endif//JVCIIOOILOJOWAOMPVEROMIMRXPVNPPLPXMRTNERCNBREVHGBDJIQWUXGEUPUBDTLUNEVRFJU

