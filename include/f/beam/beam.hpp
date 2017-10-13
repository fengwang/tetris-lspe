#ifndef SKQISQBUIMEUDLCUXUVIHXHHCNQUBIFTYJIAJSABMADOAFONBNIUPNSXKHASHHTHADQNLWFKS
#define SKQISQBUIMEUDLCUXUVIHXHHCNQUBIFTYJIAJSABMADOAFONBNIUPNSXKHASHHTHADQNLWFKS

#include <f/matrix/matrix.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <cassert>

namespace f
{

    struct beam
    {
        int mh;
        int mk;
        int ml;
    };

    bool operator == ( beam const& lhs, beam const& rhs )
    {
        return lhs.mh == rhs.mh &&
               lhs.mk == rhs.mk &&
               lhs.ml == rhs.ml;
    }

    bool operator < ( beam const& lhs, beam const& rhs )
    {
        if ( lhs.mh < rhs.mh ) return true;
        if ( lhs.mh > rhs.mh ) return false;
        if ( lhs.mk < rhs.mk ) return true;
        if ( lhs.mk > rhs.mk ) return false;
        if ( lhs.ml < rhs.ml ) return true;
        return false;
    }

    inline beam const operator - ( beam const& lhs, beam const& rhs )
    {
        return beam{ lhs.mh - rhs.mh, lhs.mk - rhs.mk, lhs.ml - rhs.ml };
    }

    inline std::ostream& operator << ( std::ostream& os, beam const& rhs )
    {
        os << "B["<<rhs.mh << "][" << rhs.mk << "][" << rhs.ml << "]";
        return os;
    }

    inline void make_full_beam_list( matrix<double> const& old_beams, matrix<int>& ar, matrix<double>& new_beams )
    {
        std::map<beam, unsigned long> beam_index_map;
        std::map<unsigned long, beam> index_beam_map;

        unsigned long const total_beams = old_beams.row();

        for ( unsigned long index = 0; index != total_beams; ++index )
        {
            assert( old_beams[index][0] == index );
            int mh = old_beams[index][1];
            int mk = old_beams[index][2];
            int ml = old_beams[index][3];
            beam const the_beam{ mh, mk, ml };
            beam_index_map[the_beam] = index;
            index_beam_map[index] = the_beam;
        }

        ar.resize( total_beams, total_beams );

        for ( unsigned long index = 0; index != total_beams; ++index )
            ar[index][0] = index;

        for ( unsigned long row_index = 0; row_index != ar.row(); ++row_index )
            for ( unsigned long col_index = 1; col_index != ar.col(); ++col_index )
            {
                beam const new_beam = index_beam_map[ar[row_index][0]] - index_beam_map[ar[col_index][0]];
                if ( beam_index_map.find(new_beam) != beam_index_map.end() )
                {
                    ar[row_index][col_index] = beam_index_map[new_beam];
                    continue;
                }

                assert( beam_index_map.size() == index_beam_map.size() );
                unsigned long new_index = beam_index_map.size();
                beam_index_map[new_beam] = new_index;
                index_beam_map[new_index] = new_beam;
                ar[row_index][col_index] = new_index;
            }

        new_beams.resize( index_beam_map.size(), 4 );
        for ( auto const& elem : index_beam_map )
        {
            unsigned long const index = elem.first;
            int const h = elem.second.mh;
            int const k = elem.second.mk;
            int const l = elem.second.ml;

            new_beams[index][0] = index;
            new_beams[index][1] = h;
            new_beams[index][2] = k;
            new_beams[index][3] = l;
        }
    }

    inline void make_full_beam_list( std::string const& beams_file_path = "beams.txt", std::string const& ar_file_path = "Ar.txt", std::string const& new_beams_file_path = "new_beams.txt" )
    {
#if 0
        std::map<beam, unsigned long> beam_index_map;
        std::map<unsigned long, beam> index_beam_map;

        matrix<double> old_beams;
        old_beams.load( beams_file_path );

        unsigned long const total_beams = old_beams.row();

        for ( unsigned long index = 0; index != total_beams; ++index )
        {
            assert( old_beams[index][0] == index );
            int mh = old_beams[index][1];
            int mk = old_beams[index][2];
            int ml = old_beams[index][3];
            beam const the_beam{ mh, mk, ml };
            beam_index_map[the_beam] = index;
            index_beam_map[index] = the_beam;
        }

        matrix<int> ar{ total_beams, total_beams };
        for ( unsigned long index = 0; index != total_beams; ++index ) ar[index][0] = index;

        for ( unsigned long row_index = 0; row_index != ar.row(); ++row_index )
            for ( unsigned long col_index = 1; col_index != ar.col(); ++col_index )
            {
                beam const new_beam = index_beam_map[ar[row_index][0]] - index_beam_map[ar[col_index][0]];
                if ( beam_index_map.find(new_beam) != beam_index_map.end() )
                {
                    ar[row_index][col_index] = beam_index_map[new_beam];
                    continue;
                }

                assert( beam_index_map.size() == index_beam_map.size() );
                unsigned long new_index = beam_index_map.size();
                beam_index_map[new_beam] = new_index;
                index_beam_map[new_index] = new_beam;
                ar[row_index][col_index] = new_index;
            }
        ar.save_as( ar_file_path );

        matrix<int> new_beam_matrix{ index_beam_map.size(), 4 };
        for ( auto const& elem : index_beam_map )
        {
            unsigned long const index = elem.first;
            int const h = elem.second.mh;
            int const k = elem.second.mk;
            int const l = elem.second.ml;

            new_beam_matrix[index][0] = index;
            new_beam_matrix[index][1] = h;
            new_beam_matrix[index][2] = k;
            new_beam_matrix[index][3] = l;
        }
        new_beam_matrix.save_as( new_beams_file_path );
#else
        matrix<double> old_beams;
        old_beams.load( beams_file_path );
        matrix<int> ar;
        matrix<double> new_beams;
        make_full_beam_list( old_beams, ar, new_beams );
        ar.save_as( ar_file_path );
        new_beams.save_as( new_beams_file_path );
#endif

    }

}//namespace f

#endif//SKQISQBUIMEUDLCUXUVIHXHHCNQUBIFTYJIAJSABMADOAFONBNIUPNSXKHASHHTHADQNLWFKS

