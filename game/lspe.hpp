#ifndef _LSTD_HPP_INCLUDED
#define _LSTD_HPP_INCLUDED

#include "./tetris.hpp"

#include <f/matrix/matrix.hpp>
#include <f/matrix/numeric/singular_value_decomposition.hpp>

#include <numeric>
#include <utility>
#include <array>
#include <tuple>
#include <iterator>
#include <algorithm>

constexpr unsigned long features_to_use = 33;

struct lspe
{
    typedef double              value_type;
    typedef tetris              state_type;
    typedef action              action_type;
    typedef piece               piece_type;
    typedef f::matrix<double>   matrix_type;
    typedef unsigned long       size_type;

    matrix_type                 A;
    matrix_type                 theta;
    matrix_type                 e;
    matrix_type                 e_new;
    matrix_type                 b;
    matrix_type                 psi;
    matrix_type                 psi_new;
    matrix_type                 psi_diff;

    // for matrix svd
    matrix_type                 A_u;
    matrix_type                 A_v;
    matrix_type                 A_w;

    state_type                  game;
    size_type                   training_set;
    value_type                  lambda;
    value_type                  delta;

    // returns a column vector of all the features
    matrix_type const features( state_type const& s_ ) const
    {
        matrix_type     ans( features_to_use, 1 );
        std::array<unsigned long, 10> hets{ s_.height(0), s_.height(1), s_.height(2), s_.height(3), s_.height(4), s_.height(5), s_.height(6), s_.height(7), s_.height(8), s_.height(9) };
        std::array<unsigned long, 10> emps{ s_.empties(0), s_.empties(1), s_.empties(2), s_.empties(3), s_.empties(4), s_.empties(5), s_.empties(6), s_.empties(7), s_.empties(8), s_.empties(9) };
        // bias                                 // 1
        ans[0][0] = 1;
        //height of every column                // 10
        ans[1][0] = hets[ 0 ];
        ans[2][0] = hets[ 1 ];
        ans[3][0] = hets[ 2 ];
        ans[4][0] = hets[ 3 ];
        ans[5][0] = hets[ 4 ];
        ans[6][0] = hets[ 5 ];
        ans[7][0] = hets[ 6 ];
        ans[8][0] = hets[ 7 ];
        ans[9][0] = hets[ 8 ];
        ans[10][0] = hets[ 9 ];
        //empty spaces of every column          // 10
        ans[11][0] = emps[ 0 ];
        ans[12][0] = emps[ 1 ];
        ans[13][0] = emps[ 2 ];
        ans[14][0] = emps[ 3 ];
        ans[15][0] = emps[ 4 ];
        ans[16][0] = emps[ 5 ];
        ans[17][0] = emps[ 6 ];
        ans[18][0] = emps[ 7 ];
        ans[19][0] = emps[ 8 ];
        ans[20][0] = emps[ 9 ];
        //height difference of adjancent columns// 9 -- cliffs
        ans[21][0] = std::abs(ans[1][0] - ans[2][0]);
        ans[22][0] = std::abs(ans[2][0] - ans[3][0]);
        ans[23][0] = std::abs(ans[3][0] - ans[4][0]);
        ans[24][0] = std::abs(ans[4][0] - ans[5][0]);
        ans[25][0] = std::abs(ans[5][0] - ans[6][0]);
        ans[26][0] = std::abs(ans[6][0] - ans[7][0]);
        ans[27][0] = std::abs(ans[7][0] - ans[8][0]);
        ans[28][0] = std::abs(ans[8][0] - ans[9][0]);
        ans[29][0] = std::abs(ans[9][0] - ans[10][0]);
        //holes                                 // 1
        ans[30][0] = s_.holes();
        //highest column                        // 1
        ans[31][0] = *std::max_element( ans.begin()+1, ans.begin()+11 ); // highest column

        //more features
        ans[32][0] = *std::max_element( ans.begin()+21, ans.begin()+30 ); //max cliffs

        return ans/20.0;
    }

    lspe( unsigned long training_set_ = 10000, value_type lambda_ = 0.002 ) : training_set( training_set_ ), lambda( lambda_ )
    {
        A.resize( features_to_use, features_to_use );
        e.resize( features_to_use, 1 );
        e_new.resize( features_to_use, 1 );
        b.resize( features_to_use, 1 );
        psi.resize( features_to_use, 1 );
        psi_new.resize( features_to_use, 1 );
        psi_diff.resize( features_to_use, 1 );

        theta.resize( features_to_use, 1 );
        theta.resize( features_to_use, 1 );
        std::fill( theta.begin(), theta.end(), -0.010 );
    }

    void operator()()
    {
        for ( size_type idx = 0; idx != training_set; ++idx )
        {
            std::cerr << "Training " << idx+1 << " of " << training_set << "\n\n";

            e = features( game );
            psi = e;
            delta = 0.0;
            std::fill( A.begin(), A.end(), 0 );
            std::fill( b.begin(), b.end(), 0 );

            for (;;)
            {
                if ( game.gameover ) break;

                step_in();
            }

            std::cerr << "Training result is\n" << theta << "\n";
            std::cerr << game.lines_cleared << " Lines Cleared.\n";

            std::ofstream ofs{ "./lspe_output.txt", std::ofstream::out | std::ofstream::app };
            ofs << game.lines_cleared << "\t";
            for ( auto const x : theta )
                ofs << x << "\t";
            ofs << "\n";
            ofs.close();


            update_theta();

            srand(0);
            game = state_type{};
        }

        std::cout << "Training result is\n" << theta << "\n";
    }

    // returns a tuple of reward and succeding state
    auto const reward( state_type const& s_, action_type const& a_ ) const
    {
        state_type s = s_;
        s.play_action( a_, false );

        return std::make_pair( s.lines_cleared - s_.lines_cleared, s );
    }

    void step_in()
    {
#if 0
        //auto const& ba = select_action( game, theta.begin() ); // return best_score best_action next_state
        //auto const& best_action = std::get<1>( ba );
        auto const& best_action = select_action( game, theta.begin() );
        auto const& best_reward = reward( game, best_action );
        double const best_action_reward = best_reward.first;
        game.play_action( best_action, false );

        psi_new = features( game );
        psi_diff = psi - psi_new;

        A += e * psi_diff.transpose();

        b += best_action_reward * e;

        e *= lambda;
        e += psi_new;

        psi = psi_new;
#else
        auto const& best_action = select_action( game, theta.begin() );
        auto const& best_reward = reward( game, best_action );
        double const best_action_reward = best_reward.first;
        game.play_action( best_action, false );
        //game.play_action( best_action, true );
        psi_new = features( game );
        double const v = std::inner_product( psi.begin(), psi.end(), theta.begin(), 0.0 );
        double const gamma = 0.99;
        //double const delta = gamma * lambda * delta + ( best_action_reward + gamma * std::inner_product( theta.begin(), theta.end(), psi_new.begin(), 0.0 ) - v );
        double const delta = gamma * gamma * delta + ( best_action_reward + gamma * std::inner_product( theta.begin(), theta.end(), psi_new.begin(), 0.0 ) - v );
        b += (v+delta) * psi;
        A += psi * psi.transpose();

        psi = psi_new;
#endif

    }

    void update_theta()
    {
        double const b_norm = std::inner_product( b.begin(), b.end(), b.begin(), 0.0 );
        if ( b_norm < 1.0e-10 )
        {
            std::cerr << "Failed with very small b norm\n";
            return;
        }

        //update theta

        auto svd_return = f::singular_value_decomposition( A, A_u, A_v, A_w );
        if ( svd_return )
        {
            std::cerr << "Failed when applying SVD to matrix\n";
            std::cout << A << "\n";
            std::cerr << game << "\n";
            assert(!"Breaks with bad svd." );
            return;
        }
        //assert( svd_return == 0 && "Failed to execute SVD" );
        std::for_each( A_v.begin(), A_v.end(), []( auto& val ){ if (std::abs(val) > 1.0e-10) val = 1.0/val; else val = 0.0; } );
        theta += lambda * A_w * A_v * A_u.transpose() * b;
    }

    template< typename Itor >
    value_type state_value( state_type const& s_, Itor it_ ) const // it_ is the first itor of weight
    {
        auto const& phi = features( s_ );
        return std::inner_product( phi.begin(), phi.end(), it_, value_type{0} );
    }

    // select best action for state s_ with piece p_ and weight itor it_
    template< typename Itor >
    auto const select_action( state_type const& s_, Itor it_, piece_type const& p_ ) const
    {
        // for all possible actions
        action_type best_action{ NONE, 0 };
        state_type best_state;
        value_type best_score = -99999999999999.0;

        std::array<rotation, 4> all_rotations{ NONE, CLOCKWISE, COUNTER_CLOCKWISE, FLIP };
        state_type s = s_;
        s.current_piece = p_;

        for ( auto const& r : all_rotations ) // for all possible rotations
        {
            piece_type p = p_;
            p.rotate( r );
            int const left_most_column = 0;
            int const right_most_column = static_cast<int>(board_cols - p.piece_width()) - 1;
            for ( int c = left_most_column; c < right_most_column; ++c ) // for all possible columns
            {
                // estimation method -- state, action, weight, piece
                auto const& rw = reward( s, action_type{ r, c } );
                auto val = state_value( rw.second, it_ );
                double const current_score = rw.first + val;

                if ( current_score > best_score )
                {
                    best_action = action_type{ r, c };
                    best_state = rw.second;
                    best_score = current_score;
                }
            }
        }

        assert( best_action.column >= 0 && best_action.column < 10  && "Best action column is not valid" );

        return std::make_tuple( best_score, best_action, best_state );
    }

    // select best action with an expection of next piece
    template< typename Itor >
    action_type const select_action( state_type const& s_, Itor it_ ) const
    {
        // enumerate all actions
        // enumerate all possible succeeding pieces
        // select the one with best score
        // for all possible actions
        //action_type best_action;
        action_type best_action{ NONE, 0 };
        state_type best_state;
        value_type best_score = -99999999999999.0;

        std::array<rotation, 4> const all_rotations{ NONE, CLOCKWISE, COUNTER_CLOCKWISE, FLIP };
        std::array<piece_type, 7> const all_pieces{ piece_type{1}, piece_type{2}, piece_type{3}, piece_type{4}, piece_type{5}, piece_type{6}, piece_type{7} };

        for ( auto const& r : all_rotations ) // for all possible rotations
        {
            piece_type p = game.current_piece;
            p.rotate( r );
            int const left_most_column = 0;
            int const right_most_column = static_cast<int>(board_cols - p.piece_width());
            for ( int c = left_most_column; c < right_most_column; ++c ) // for all possible columns
            {
                // estimation method -- state, action, weight, piece
                auto const& rw = reward( s_, action_type{ r, c } );
                double val = 0.0;
                for ( auto const& next_step_piece : all_pieces )
                {
                    auto const& best_action_at_next_step = select_action( rw.second, it_, next_step_piece );
                    val += std::get<0>( best_action_at_next_step );
                }
                val /= value_type{ 7.0 };
                //auto val = state_value( rw.second, it_ );
                //
                val += state_value( rw.second, it_ ); //...
                double const current_score = rw.first + val;

                if ( current_score > best_score )
                {
                    best_action = action_type{ r, c };
                    best_state = rw.second;
                    best_score = current_score;
                }
            }
        }

        assert( best_action.column >= 0 && best_action.column < 10  && "Best action column is not valid" );

        return best_action;
    }

};

#endif//_LSTD_HPP_INCLUDED

