#ifndef TXKSSVXKNXYDUJXLILCEOGPLWQBYKRUNOSUGCUNYQMULTBSWNOPKSTNGIRWLSDWOHMIJOVSLW
#define TXKSSVXKNXYDUJXLILCEOGPLWQBYKRUNOSUGCUNYQMULTBSWNOPKSTNGIRWLSDWOHMIJOVSLW

#include "beam_list.hpp"
#include "make_ar.hpp"
#include <vector>

namespace f
{

    inline void make_beam( std::string const& old_beam_path, std::string const& new_beam_path )
    {
        auto bl = make_beam_list( old_beam_path );
        unsigned long const n = bl.index_beam_map.size();
        
        std::vector<unsigned long> vl;
        vl.resize(n);
        for ( unsigned long index = 0; index != n; ++index )
            vl[index] = index;

        auto ar = make_ar( bl, vl.begin(), vl.end() );

        unsigned long const N = bl.index_beam_map.size();

        matrix<long int> bm{ N, 4 };

        for ( unsigned long index = 0; index != N; ++index )
        {
            bm[index][0] = index;
            bm[index][1] = bl.index_beam_map[index].mh;
            bm[index][2] = bl.index_beam_map[index].mk;
            bm[index][3] = bl.index_beam_map[index].ml;
        }

        bm.save_as( new_beam_path );
    }

}//namespace f
#endif//TXKSSVXKNXYDUJXLILCEOGPLWQBYKRUNOSUGCUNYQMULTBSWNOPKSTNGIRWLSDWOHMIJOVSLW
