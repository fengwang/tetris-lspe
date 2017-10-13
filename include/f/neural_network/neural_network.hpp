//
// NeurAl NetwOrk  -- NANO
//
#ifndef LUSWRWFRPSGJRKBTGSQGFECEROJNVUMASEKWRHUXCQKSGLCHSAASFLTLCNLUFWVQQPXYDPQXA
#define LUSWRWFRPSGJRKBTGSQGFECEROJNVUMASEKWRHUXCQKSGLCHSAASFLTLCNLUFWVQQPXYDPQXA

#include <f/overloader/overloader.hpp>
#include <f/matrix/matrix.hpp>
#include <f/algorithm/for_each.hpp>
#include <f/algorithm/all_of.hpp>
#include <f/variate_generator/variate_generator.hpp>

#include "./functions.hpp"

#include <memory>
#include <algorithm>
#include <vector>
#include <utility>
#include <cmath>
#include <array>
#include <type_traits>
#include <cassert>

#include <iostream>

namespace f
{

    template< typename T >
    struct neural_network_config
    {
        typedef T                                       value_type;
        typedef std::function<value_type(value_type)>   function_type;
        typedef std::vector<function_type>              function_vector_type;
        typedef unsigned long                           size_type;
        typedef std::vector<size_type>                  size_vector_type;
        typedef matrix<value_type>                      matrix_type;

        size_vector_type                                layer_dims;             // [ 10, 15, 20, 15, 2 ]
        function_vector_type                            activation_functions;   // [ sigmoid, sigmoid, ... ]
        function_vector_type                            derivative_functions;   // [ sigmoid', sigmoid', ... ]
        value_type                                      learning_rate;          // can be differrent for different connections, in range [10^-1, 10^-5]
        value_type                                      momentum_constant;      // [0, 1)
    };

    template< typename T >
    void assert_config( neural_network_config<T> const& cfg_ )
    {
        assert( cfg_.layer_dims.size() );
        assert( cfg_.layer_dims.size() == cfg_.activation_functions.size() );
        assert( cfg_.layer_dims.size() == cfg_.derivative_functions.size() );
        assert( std::all_of( cfg_.layer_dims.begin(), cfg_.layer_dims.end(), []( auto u_ ){ return u_ != 0; } ) );
        assert( std::all_of( cfg_.activation_functions.begin(), cfg_.activation_functions.end(), []( auto const& f_ ){ if ( f_ ) return true; return false; } ) );
        assert( std::all_of( cfg_.derivative_functions.begin(), cfg_.derivative_functions.end(), []( auto const& f_ ){ if ( f_ ) return true; return false; } ) );
        assert( cfg_.learning_rate > T{0} );
        assert( cfg_.momentum_constant >= T{0} );
        assert( cfg_.momentum_constant < T{1} );
    }

    template< typename T, typename Concrete_Neural_Network >
    struct neural_network
    {
        typedef T                                   value_type;
        typedef value_type*                         pointer;

        neural_network_config<value_type>           config;

        void initialize( neural_network_config<T> const& config_ )
        {
            assert_config( config_ );
            config = config_;
            static_cast<Concrete_Neural_Network&>(*this).do_initialization( config_ );
        }

        void feed( pointer input_, pointer expected_output_ )
        {
            static_cast<Concrete_Neural_Network&>(*this).do_feeding( input_, expected_output_ );
        }

        value_type validate( pointer input_, pointer output_ )
        {
            return static_cast<Concrete_Neural_Network&>(*this).do_validation( input_, output_ );
        }

        void predict( pointer input_, pointer output_ )
        {
            static_cast<Concrete_Neural_Network&>(*this).do_prediction( input_, output_ );
        }

    };

    template< typename Type >
    struct default_neural_network : neural_network< Type, default_neural_network<Type> >
    {
        typedef Type                            value_type;
        typedef value_type*                     pointer;
        typedef matrix<value_type>              matrix_type;
        typedef std::vector<matrix_type>        matrix_vector_type;
        typedef unsigned long                   size_type;

        size_type                               layers;
        matrix_vector_type                      W;
        matrix_vector_type                      NI;     //input of a neuron
        matrix_vector_type                      N;      //output of a neuron
        matrix_vector_type                      E;
        matrix_type                             T;

        matrix_vector_type                      dW;

        unsigned long                           mini_batch;
        unsigned long                           current_iteration;

        friend std::ostream& operator << ( std::ostream& os, default_neural_network<Type> const& dnn )
        {
            os << "Layers:\n";
            std::copy( dnn.config.layer_dims.begin(), dnn.config.layer_dims.end(), std::ostream_iterator<size_type>(os, " ") );
            os<< "\n";
            os << "\nW:\n";
            for ( auto const& w : dnn.W )
              os << w << "\n";
            os << "\nN:\n";
            for ( auto const& n : dnn.N )
              os << n << "\n";
            os << "\nNI:\n";
            for ( auto const& n : dnn.NI )
              os << n << "\n";
            os << "\nE:\n";
            os << "\nE:\n";
            for ( auto const& e : dnn.E )
              os << e << "\n";
            os << "\nT:\n" << dnn.T << "\n\n";
            return os;
        }

        void do_initialization( neural_network_config<Type> const& config_ )
        {
            assert( config_.layer_dims.size() );

            layers = config_.layer_dims.size();
            auto const& L = config_.layer_dims;

            W.resize( layers-1 );
            for ( auto index = 0UL; index != W.size(); ++index )
                W[index].resize( L[index]+1, L[index+1] );

            variate_generator<double> vg{ value_type{-1.0}, value_type{1.0} };
            for ( auto index = 0UL; index != W.size(); ++index )
                std::generate( W[index].begin(), W[index].end(), vg );

            N.resize( layers );
            for ( auto index = 0UL; index != N.size(); ++index )
                N[index].resize( L[index]+1, 1 );

            NI.resize( layers );
            for ( auto index = 0UL; index != N.size(); ++index )
                NI[index].resize( L[index]+1, 1 );

            for ( auto index = 0UL; index != N.size(); ++index )
                N[index][0][0] = value_type{1.0};

            E.resize( layers );
            //for ( auto index = 0UL; index != E.size(); ++index )
            for ( auto index = 1UL; index != E.size(); ++index )
                E[index].resize( L[index], 1 );

            T.resize( *(L.rbegin()), 1 );

            dW.resize( layers-1 );
            for ( auto index = 0UL; index != dW.size(); ++index )
            {
                dW[index].resize( L[index]+1, L[index+1] );
                std::fill( dW[index].begin(), dW[index].end(), value_type{0} );
            }

            mini_batch = 128;
            current_iteration = 0;
        }

        template< typename Iterator >
        void do_feeding( Iterator input_itor_, Iterator expected_output_itor_ )
        {
            std::copy( input_itor_, input_itor_+N[0].size()-1, N[0].begin()+1 );
            std::copy( expected_output_itor_, expected_output_itor_+T.size(), T.begin() );
        }

        void do_feeding( pointer input_, pointer expected_output_ )
        {
            std::copy( input_, input_+N[0].size()-1, N[0].begin()+1 );
            std::copy( expected_output_, expected_output_+T.size(), T.begin() );
/*
            forward();
            backward();
            update();
*/
        }

        void forward() // This is fine
        {
            for ( auto i = 1UL; i != layers; ++i )      //layer loop
                for ( auto r = 1UL; r != N[i].row(); ++r )  //neuron loop
                {
                    // N[i] of size [ L[i]+1, 1 ]
                    // N[i-1] of size [ L[i-1]+1, 1 ]
                    // W[i-1] of size [ L[i-1]+1, L[i] ]
                    // N[i] is set to inner product of N[i-1] and each column of W[i-1]
                    NI[i][r][0] = std::inner_product( N[i-1].begin(), N[i-1].end(), W[i-1].col_begin(r-1), value_type{0.0} );
                    N[i][r][0] = (*this).config.activation_functions[i]( NI[i][r][0] );
                }
        }

        void backward()
        {
            // The last layer
            {
                auto& E_ = *(E.rbegin());
                auto& N_ = *(N.rbegin());
                auto const & f_ = *((*this).config.derivative_functions.rbegin());  //the last derivation function
                for ( auto i = 0UL; i != E_.size(); ++i )
                {
                    E_[i][0] = (T[i][0] - N_[i+1][0]) * f_(N_[i+1][0]);
                    //std::cout << E_[i][0] << " = " << T[i][0] - N_[i+1][0] << " * " << f_(N_[i+1][0]) << "\n";
                }
            }
            {
                for ( auto i = layers-2; i != 0; --i )
                {
                    auto const& f_ = (*this).config.derivative_functions[i];
                    // E[i] of size [ L[i], 1 ]
                    // E[i+1] of size [ L[i+1], 1 ]
                    // W[i] of size [ L[i]+1, L[i+1] ]
                    for ( auto r = 0UL; r != E[i].row(); ++r )
                    {
                        E[i][r][0] = std::inner_product( E[i+1].begin(), E[i+1].end(), W[i].row_begin(r+1), value_type{0.0} );
                        //E[i][r][0] *= f_( N[i][r+1][0] );
                        E[i][r][0] *= f_( NI[i][r+1][0] );
                    }
                }
            }
        }

        void update()
        {
            for ( auto i = 0UL; i != W.size(); ++i )
            {
                /*
                std::cout << "Update of W matrix at index " << i << "\n";
                std::cout << "W[] = \n" << W[i] << "\n";
                std::cout << "N[] = \n" << N[i] << "\n";
                std::cout << "E[] = \n" << E[i+1] << "\n" << std::endl;
                */
                for ( auto r = 0UL; r != W[i].row(); ++r )
                    for ( auto c = 0UL; c != W[i].col(); ++c )
                    {
                        assert( r < N[i].row() );
                        assert( c < E[i+1].row() );
                        //W[i][r][c] += ((*this).config.learning_rate) * N[i][r][0] * E[i+1][c][0];
                        dW[i][r][c] += ((*this).config.learning_rate) * N[i][r][0] * E[i+1][c][0] / static_cast<value_type>(mini_batch);
                    }


            }

            if ( current_iteration++ > mini_batch )
            {
                for ( auto i = 0UL; i != W.size(); ++i )
                {
                    W[i] += dW[i];
                    std::fill( dW[i].begin(), dW[i].end(), value_type{0} );
                }
                current_iteration = 0;
            }
        }

        void do_prediction( pointer input_, pointer output_ )
        {
          std::copy( input_, input_+N[0].size()-1, N[0].begin()+1 );
          //std::copy( output_, output_+T.size(), T.begin() );

          forward();

          auto const& n = *(N.rbegin());
          std::copy( n.begin()+1, n.end(), output_ );
        }

        value_type do_validation( pointer input_, pointer output_ )
        {
          std::copy( input_, input_+N[0].size()-1, N[0].begin()+1 );
          std::copy( output_, output_+T.size(), T.begin() );

          forward();

          auto const& error = *(N.rbegin());
          unsigned long const output_length = T.size();

          value_type err = value_type{0};
          value_type sum = value_type{1.0e-10};
          for ( auto i = 0UL; i != output_length; ++i )
          {
              value_type df = error[i+1][0] - output_[i];
              err += df * df;
              sum += output_[i] * output_[i];
          }
          return err / sum;
        }

        void train( matrix_type& input_matrix_, matrix_type& output_matrix_ )
        {
            assert( input_matrix_.row() == output_matrix_.row() && "Training size should match!" );
            assert( input_matrix_.col() == (*this).config.layer_dims[0] && "Input size should match!" );
            assert( output_matrix_.col() == *((*this).config.layer_dims.rbegin()) && "Output size should match!" );

            for ( auto idx = 0UL; idx != input_matrix_.row(); ++idx )
            {
                do_feeding( input_matrix_.row_begin(idx), output_matrix_.row_begin(idx) );
                forward();
                backward();
                update();
            }
        }
    };

}//namespace f

#endif//LUSWRWFRPSGJRKBTGSQGFECEROJNVUMASEKWRHUXCQKSGLCHSAASFLTLCNLUFWVQQPXYDPQXA
