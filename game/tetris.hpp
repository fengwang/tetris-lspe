#ifndef QPLDMVPLGQNFVYQGCOWUSAGGTGRFETJGNYACYXLXGMBUHGUTCYCOVMCKCMCKWRAHKAEBQJLHL
#define QPLDMVPLGQNFVYQGCOWUSAGGTGRFETJGNYACYXLXGMBUHGUTCYCOVMCKCMCKWRAHKAEBQJLHL

#include "./rotation.hpp"
#include "./action.hpp"
#include "./piece.hpp"
#include "./static_matrix.hpp"

#include <numeric>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <iterator>

constexpr unsigned long board_rows = 20;
constexpr unsigned long board_cols = 10;

struct tetris
{
    typedef f::static_matrix<int, board_rows, board_cols>   matrix_type;

    int                                     gameover;
    unsigned long                           lines_cleared;
    matrix_type                             board;
    piece                                   current_piece;
    piece                                   next_piece;

    void reset_current_piece( int id )
    {
        assert( id > 0 && "reset_current_piece: id must positive!" );
        assert( id < 8 && "reset_current_piece: id must less than 8!" );
        current_piece = piece{ id };
    }

    tetris( int gameover_ = 0 ) : gameover( gameover_ ), lines_cleared( 0 )
    {
        std::fill( board.begin(), board.end(), 0 );
    }

    void clear_board()
    {
        std::fill( board.begin(), board.end(), 0 );
    }

    int highest_valid_column() const
    {
        //return board_cols - current_piece.piece_width() - 1;
        return board_cols - current_piece.piece_width();
    }

    //void place_piece( long drop_column, long drop_row )
    void place_piece( long drop_row, long drop_column )
    {
        auto const& cp = current_piece.piece_;

        for ( auto r = 0L; r != cp.row(); ++r )
            for ( auto c = 0L; c != cp.col(); ++c )
            {
                if ( 0 == cp[r][c] ) continue;
                if ( r+drop_row < 0 )
                {
                    gameover = true;
                }
                else
                {
                    assert( (0 == board[r+drop_row][c+drop_column]) && "Overlap happens!" );
                    board[r+drop_row][c+drop_column] = current_piece.id_;
                }
            }

        current_piece = next_piece;
        next_piece = piece{};
    }

    bool collision( int drop_row, int drop_column )
    {
        auto const& p = current_piece.piece_;

        for ( long r = 0; r != p.row(); ++r )
            for ( long c = 0; c != p.col(); ++c )
            {
                //if ( r + drop_row < 0 ) continue;

                if ( !( p[r][c] && r+drop_row >= 0 ) ) continue;


                if ( c + drop_column >= board_cols ) return true;
                if ( r + drop_row >= board_rows ) return true;
                if ( board[r+drop_row][c+drop_column] != 0 ) return true;
            }

        return false;
    }

    void clear_line( unsigned long r )
    {
        assert( r && "clear_line : argument must be positive." );
        assert( r < board_rows  && "clear_line : argument exceed maximum row." );
        assert( std::all_of( board.row_begin(r), board.row_end(r), []( int val ){ return val != 0; } ) && "This line is not fullfilled." );

        for ( auto r_ = r; r_ != 0; --r_ )
        {
            std::copy( board.row_begin(r_-1), board.row_end(r_-1), board.row_begin(r_) );
        }
        std::fill( board.row_begin(0), board.row_end(0), 0 );
    }

    void clear_lines()
    {
        for ( auto r = 1UL; r != board.row(); ++r )
            if ( std::all_of( board.row_begin(r), board.row_end(r), [](int val){ return val != 0; } ) )
            {
                clear_line( r );
                ++ lines_cleared;
            }
    }

    void drop_column( int column )
    {
        if( column > highest_valid_column() || column < 0 )
        {
            std::cerr << "Error in droping column " << column << "\n";
            std::cerr << "highest validd column is " << highest_valid_column() << "\n";
            std::cerr << "The game board is\n"  << *this << "\n";
        }

        assert( column <= highest_valid_column() && "Beat highest column!" );
        assert( column >= 0 && "column cannot be negative!" );

        int drop_row = -4;

        while( !collision( drop_row, column ) )
        {
            ++drop_row;
        }
        --drop_row;

        place_piece( drop_row, column );
        clear_lines();
    }

    void play_action( action const& a, bool animated )
    {
        if ( a.column < 0 || a.column > 10 )
        {
            std::cerr << "Error play_action at column " << a.column << "\n";;
        }
        current_piece.rotate( a.the_rotation );
        drop_column( a.column );

        if ( animated )
            std::cout << (*this) << "\n";
    }


    friend std::ostream& operator << ( std::ostream& os, tetris const& rhs ) //printboard
    {
        for ( auto r = 0UL; r != rhs.board.row(); ++r )
        {
            for ( auto c = 0UL; c != rhs.board.col(); ++c )
            {
                if ( (rhs.board)[r][c] != 0 )
                    os << static_cast<char>(27) << "[1;37;" << (rhs.board)[r][c]+40 << "m  " << static_cast<char>(27) << "[0m";
                else
                    os << " .";
            }
            os << "\n";
        }

        os << "Height\n";
        for ( unsigned long idx = 0; idx != 10; ++ idx )
            os << rhs.height(idx) << " ";
        os << "\n";

        os << "Empties\n";
        for ( unsigned long idx = 0; idx != 10; ++ idx )
            os << rhs.empties(idx) << " ";
        os << "\n";

        os << "Current Piece\n" << rhs.current_piece << std::endl;
        os << "Next Piece\n" << rhs.next_piece << std::endl;
        os << "Lines Cleared: " << rhs.lines_cleared << std::endl;

        std::cerr << "Height: " << rhs.height() << std::endl;
        std::cerr << "Aggragate Height: " << rhs.aggragate_height() << std::endl;
        std::cerr << "holes: " << rhs.holes() << std::endl;
        std::cerr << "blocked spaces: " << rhs.blocked_spaces() << std::endl;
        std::cerr << "Aggragate blocked pieces: " << rhs.aggragate_blocked_pieces() << std::endl;
        std::cerr << "Roughness: " << rhs.roughness() << std::endl;
        std::cerr << "Aggragate roughness: " << rhs.aggragate_roughness() << std::endl;

        return os;
    }

    // features

    // highest piece
    unsigned long height() const
    {
        for ( auto r = 0UL; r != board_rows; ++r )
            if ( std::any_of( board.row_begin(r), board.row_end(r), []( int val ){ return val != 0; } ) )
                return board_rows - r;

        return 0;
    }

    // sum of the highest of every piece
    unsigned long aggragate_height() const
    {
        auto ans = 0UL;

        for ( auto r = 0UL; r != board.row(); ++r )
            for ( auto c = 0UL; c != board.col(); ++c )
            {
                if ( board[r][c] != 0 )
                {
                    ans += board_rows - r;
                    //break;
                }
            }

        return ans;
    }

    // fully enclosed section of empty spaces
    // -- NOT necessarily very precious
    unsigned long holes() const
    {
        auto board_copy = board;

        //mark the first row
        std::for_each( board_copy.row_begin(0), board_copy.row_end(0), []( int& val ){ if ( 0 == val ) val = -1; } );

        // Top -> Down
        auto top_down_mark = [&board_copy]()
        {
            for ( auto r = 1UL; r != board_rows; ++ r )
                for ( auto c = 0UL; c != board_cols; ++c )
                    if ( (board_copy[r-1][c] == -1) && (board_copy[r][c] == 0) )
                        board_copy[r][c] = -1;
        };

        // Left -> Right
        auto left_right_mark = [&board_copy]()
        {
            for ( auto r = 1UL; r != board_rows; ++ r )
                for ( auto c = 1UL; c != board_cols; ++c )
                    if ( (board_copy[r][c-1] == -1) && (board_copy[r][c] == 0) )
                        board_copy[r][c] = -1;
        };

        // Right -> Left
        auto right_left_mark = [&board_copy]()
        {
            for ( auto r = 1UL; r != board_rows; ++ r )
                for ( auto c = 0UL; c != board_cols-1; ++c )
                    if ( (board_copy[r][c+1] == -1) && (board_copy[r][c] == 0) )
                        board_copy[r][c] = -1;
        };

        // bottum up makr ignored

        constexpr unsigned long loops = 3;
        for ( unsigned long index = 0; index != loops; ++index )
        {
            top_down_mark();
            left_right_mark();
            right_left_mark();
        }

        return std::count_if( board_copy.begin(), board_copy.end(), []( int val ){ return val == 0; } );
    }

    // empty space with pieces above
    unsigned long blocked_spaces() const
    {
        auto ans = 0UL;

        for ( auto c = 0; c != board_cols; ++c )
        {
            auto itor = std::find_if( board.col_begin(c), board.col_end(c), []( int val ){ return val != 0; } );

            if ( itor == board.col_end(c) ) continue;

            ans += std::count_if( itor, board.col_end(c), []( int val ) { return val == 0; } );
        }

        return ans;
    }

    // sum of number of pieces
    unsigned long aggragate_blocked_pieces() const
    {
        return std::count_if( board.begin(), board.end(), [](int val){ return val != 0; } );
    }


    // Feature to use
    unsigned long aggragate_roughness() const
    {
        // height of each column
        std::array<long, board_cols> heights;
        for ( auto c = 0UL; c != board_cols; ++c )
            heights[c] = board.col_end(c) - std::find_if( board.col_begin(c), board.col_end(c), [](int val){ return val != 0; } );

        // height difference of adjacent columns
        std::array<long, board_cols> heights_diff;
        std::adjacent_difference( heights.begin(), heights.end(), heights_diff.begin() );
        std::for_each( heights_diff.begin(), heights_diff.end(), []( long& val ){ if ( val < 0 ) val = -val; } );

        long ans = std::accumulate( heights_diff.begin()+1, heights_diff.end(), 0L );

        assert( ans >= 0 && "Roughness cannot be negative!" );
        return static_cast<unsigned long>(ans);
    }

    // Feature to use
    unsigned long roughness() const
    {
        // height of each column
        std::array<long, board_cols> heights;
        for ( auto c = 0UL; c != board_cols; ++c )
            heights[c] = board.col_end(c) - std::find_if( board.col_begin(c), board.col_end(c), [](int val){ return val != 0; } );

        // height difference of adjacent columns
        std::array<long, board_cols> heights_diff;
        std::adjacent_difference( heights.begin(), heights.end(), heights_diff.begin() );
        std::for_each( heights_diff.begin(), heights_diff.end(), []( long& val ){ if ( val < 0 ) val = -val; } );

        long ans = *std::max_element( heights_diff.begin()+1, heights_diff.end() );

        assert( ans >= 0 && "Roughness cannot be negative!" );
        return static_cast<unsigned long>(ans);
    }

    // returns the height of the selected column of the board
    unsigned long height( unsigned long column_index ) const
    {
        assert( column_index < board_cols && "tetris::height -- exceed board width!" );

        auto itor = std::find_if( board.col_begin(column_index), board.col_end(column_index), []( int val ){ return val != 0; } );
        return board.col_end(column_index) - itor;
    }

    unsigned long empties( unsigned long column_index ) const
    {
        assert( column_index < board_cols && "tetris::empties -- exceed board width!" );

        return std::count_if( board.col_begin(column_index) + (board_rows - height(column_index)), board.col_end(column_index), [](int val){ return val == 0; } );
    }
};

#endif//QPLDMVPLGQNFVYQGCOWUSAGGTGRFETJGNYACYXLXGMBUHGUTCYCOVMCKCMCKWRAHKAEBQJLHL

