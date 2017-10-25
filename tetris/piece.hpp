#ifndef OHQHRBBODDSTUGFLQCYACXKFVERNEHTHDQGXTHSPEYSFLBFIJTLPJCYGDYBTHSVNYSBOUEFIS
#define OHQHRBBODDSTUGFLQCYACXKFVERNEHTHDQGXTHSPEYSFLBFIJTLPJCYGDYBTHSVNYSBOUEFIS

#include "./static_matrix.hpp"
#include "./rotation.hpp"

#include <cassert>
#include <algorithm>
#include <iostream>

struct piece
{
    typedef f::static_matrix<int, 4, 4>     matrix_type;
    matrix_type                             piece_;
    int                                     id_;

    matrix_type const copy_piece() //
    {
        return piece_;
    }

    unsigned long piece_width() const
    {
        for ( auto c = piece_.col()-1; c != 0; --c )
            if ( std::any_of( piece_.col_begin(c), piece_.col_end(c), [](int id){ return id != 0; } ) )
                return c;

        return 0;
    }

    void replace_piece( matrix_type const& mat )
    {
        piece_ = mat;
    }

    bool empty_left() const
    {
        if ( std::any_of( piece_.col_begin(0), piece_.col_end(0), []( int id ){ return id != 0; } ) )
            return false;
        return true;
    }

    bool empty_bottom() const
    {
        if ( std::any_of( piece_.row_begin(3), piece_.row_end(3), []( int id ){ return id != 0; } ) )
            return false;
        return true;
    }

    void pull_left()
    {
        while( empty_left() )
        {
            std::copy( piece_.col_begin(1), piece_.col_end(1), piece_.col_begin(0) );
            std::copy( piece_.col_begin(2), piece_.col_end(2), piece_.col_begin(1) );
            std::copy( piece_.col_begin(3), piece_.col_end(3), piece_.col_begin(2) );
            std::fill( piece_.col_begin(3), piece_.col_end(3), 0 );
        }
    }

    void pull_down()
    {
        while (empty_bottom() )
        {
            std::copy( piece_.row_begin(2), piece_.row_end(2), piece_.row_begin(3) );
            std::copy( piece_.row_begin(1), piece_.row_end(1), piece_.row_begin(2) );
            std::copy( piece_.row_begin(0), piece_.row_end(0), piece_.row_begin(1) );
            std::fill( piece_.row_begin(0), piece_.row_end(0), 0 );
        }
    }

    void rotate( rotation rot )
    {
        auto buf = piece_;

        switch ( static_cast<long>(rot) )
        {
            case NONE:
                break;

            case CLOCKWISE:
                std::copy( piece_.row_begin(0), piece_.row_end(0), buf.col_begin(3) );
                std::copy( piece_.row_begin(1), piece_.row_end(1), buf.col_begin(2) );
                std::copy( piece_.row_begin(2), piece_.row_end(2), buf.col_begin(1) );
                std::copy( piece_.row_begin(3), piece_.row_end(3), buf.col_begin(0) );
                break;


            case COUNTER_CLOCKWISE:
                std::copy( piece_.row_begin(0), piece_.row_end(0), buf.col_begin(0) );
                std::copy( piece_.row_begin(1), piece_.row_end(1), buf.col_begin(1) );
                std::copy( piece_.row_begin(2), piece_.row_end(2), buf.col_begin(2) );
                std::copy( piece_.row_begin(3), piece_.row_end(3), buf.col_begin(3) );

                std::reverse( buf.col_begin(0), buf.col_end(0) );
                std::reverse( buf.col_begin(1), buf.col_end(1) );
                std::reverse( buf.col_begin(2), buf.col_end(2) );
                std::reverse( buf.col_begin(3), buf.col_end(3) );
                break;

            case FLIP:
                std::reverse( buf.col_begin(0), buf.col_end(0) );
                std::reverse( buf.col_begin(1), buf.col_end(1) );
                std::reverse( buf.col_begin(2), buf.col_end(2) );
                std::reverse( buf.col_begin(3), buf.col_end(3) );

                std::reverse( buf.row_begin(0), buf.row_end(0) );
                std::reverse( buf.row_begin(1), buf.row_end(1) );
                std::reverse( buf.row_begin(2), buf.row_end(2) );
                std::reverse( buf.row_begin(3), buf.row_end(3) );
                break;

            default:
                assert( !"Never reach here!" );
        }
        piece_.swap(buf);
        pull_left();
        pull_down();
    }

    piece( int id = 0 )
    {
        if ( id == 0 )
            id = std::rand() % 7 + 1;

        id_ = id;

        assert( id > 0 && "piece(int) -- id must positive" );
        assert( id < 8 && "piece(int) -- id must less than 8" );

        std::fill( piece_.begin(), piece_.end(), 0 );

	   	switch ( id )
    	{
        case 1:
            // . . . .
            // . . . .
            // X X . .
            // X X . .
            //
            //
            piece_[2][0] = 1; piece_[2][1] = 1;
            piece_[3][0] = 1; piece_[3][1] = 1;
            break;

        case 2:
            // . . . .
            // . . . .
            // X . . .
            // X X X .
            piece_[2][0] = 1;
            piece_[3][0] = 1; piece_[3][1] = 1; piece_[3][2] = 1;
            break;

        case 3:
            // . . . .
            // . . . .
            // . . X .
            // X X X .
            //
            //
            piece_[2][0] = 0; piece_[2][1] = 0; piece_[2][2] = 1;
            piece_[3][0] = 1; piece_[3][1] = 1; piece_[3][2] = 1;
            break;

        case 4:
            // . . . .
            // . . . .
            // . . . .
            // X X X X
            //
            //
            //
            piece_[3][0] = 1; piece_[3][1] = 1; piece_[3][2] = 1; piece_[3][3] = 1;
            break;

        case 5:
            // . . . .
            // . . . .
            // . X X .
            // X X . .
            //
            //
            piece_[2][0] = 0; piece_[2][1] = 1; piece_[2][2] = 1; piece_[2][3] = 0;
            piece_[3][0] = 1; piece_[3][1] = 1; piece_[3][2] = 0; piece_[3][3] = 0;
            break;

        case 6:
            // . . . .
            // . . . .
            // X X . .
            // . X X .
            //
            //
            piece_[2][0] = 1; piece_[2][1] = 1; piece_[2][2] = 0; piece_[2][3] = 0;
            piece_[3][0] = 0; piece_[3][1] = 1; piece_[3][2] = 1; piece_[3][3] = 0;
            break;

        case 7:
            // . . . .
            // . . . .
            // . X . .
            // X X X .
            //
            //
            piece_[2][0] = 0; piece_[2][1] = 1; piece_[2][2] = 0; piece_[2][3] = 0;
            piece_[3][0] = 1; piece_[3][1] = 1; piece_[3][2] = 1; piece_[3][3] = 0;
            break;

        default:
			assert( !"Never reach here!" );
        }
    }



};


std::ostream& operator << ( std::ostream& os, piece const& p )
{
    for ( auto r = 0UL; r != p.piece_.row(); ++r )
    {
        for ( auto c = 0UL; c != p.piece_.col(); ++c )
            if ( (p.piece_)[r][c] )
                os << static_cast<char>(27) << "[1;37;" << 40 + p.id_ << "m  " << static_cast<char>(27)<< "[0m";
            else
                os << ". ";
        os << "\n";
    }

#if 0
    for ( auto c = 0UL; c != p.piece_.col(); ++c )
    {
        if ( p.piece_.col_begin(c) != p.piece_.col_end(c) )
            std::cout << "!Equal\n";
        else
            std::cout << "Equal\n";
        std::cout << "size " << std::distance( p.piece_.col_begin(c), p.piece_.col_end(c) ) << "\n";
        std::cout << "diff " << p.piece_.col_begin(c) - p.piece_.col_end(c) << "\n";
        std::copy( p.piece_.col_begin(c), p.piece_.col_end(c), std::ostream_iterator<int>( os, " " ) );
        os << "\n";
    }
#endif

    return os;
}


#endif//OHQHRBBODDSTUGFLQCYACXKFVERNEHTHDQGXTHSPEYSFLBFIJTLPJCYGDYBTHSVNYSBOUEFIS

