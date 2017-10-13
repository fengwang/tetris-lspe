#ifndef LWJFYMMSNAJWAXFNVPOFRIXAWVMQOCFLQWSJNDHJFARFSDUSKCAVOODAFMMOYJLDTIEUFBSCE
#define LWJFYMMSNAJWAXFNVPOFRIXAWVMQOCFLQWSJNDHJFARFSDUSKCAVOODAFMMOYJLDTIEUFBSCE

#include "./neural_network.hpp"

#include <f/variate_generator/variate_generator.hpp>

namespace f
{

    template< typename T >
    struct default_neural_network : neural_network_config< T, default_neural_network<T> >
    {
        typedef unsigned long                   size_type;
        typedef T                               value_type;
        typedef value_type*                     pointer;
        typedef matrix<value_type>              matrix_type;
        typedef std::vector<matrix_type>        matrix_vector_type;

        matrix_vector_type                      w;  // synaptic weight of neurons
        matrix_vector_type                      v;  // the induced local field
        matrix_vector_type                      y;  // output signal at each layer
        matrix_vector_type                      delta; // local gradients
        matrix_type                             e;  // error signal at the output
        matrix_type                             x;  // traing example at input layer in the epoch
        matrix_type                             d;  // traing example desired response in the epoch

        void do_forward_computation( pointer input_, pointer output_ )
        {
            std::copy( input_, input_ + config.layer_dims[0], y[0].begin() );
            std::copy( output_, output_ + (*(config.layer_dims.rbegin())), d.begin() );
            for ( size_type index = 1; index != config.layer_dims.size(); ++index )
            {
                for ( size_type jndex = 0; jndex != config.layer_dims[index]; ++jndex )
                    v[index] = y[index-1] * w[index-1]; // TODO: optimize here

                std::for_each( y[index].begin(), y[index].end(), v[index].begin(), [this, index]( value_type& y_, value_type const v_ ){ y_ = (*this).config.activation_functions[index-1](v_); } );
            }
            e = d - (*(y.rbegin())); // TODO: optimize here
        }

        void do_backward_computation()
        {
        }

        void do_feeding( pointer input_, pointer output_ )
        {
            do_forward_computation( input_, output_ );
            do_backward_computation();
        }

        void do_validation( pointer input_, pointer output_ )
        {
        }

        void do_initialization( neural_network_config<T> const& config_ )
        {
            w.resize( config_.layer_dims.size()-1 );
            variate_generator<value_type> vg{ value_type{-1}, value_type{1} };

            for ( size_type index = 0; index != config_.layer_dims.size()-1; ++index )
            {
                w[index].resize( layer_dims[index], layer_dims[index+1] );
                std::generate( w[index].begin(), w[index].end(), vg );
            }

            v.resize( config_.layer_dims.size() );
            y.resize( config_.layer_dims.size() );
            delta.resize( config_.layer_dims.size() );
            for ( size_type index = 0; index != config_.layer_dims.size(); ++index )
            {
                v[index].resize( 1, layer_dims[index] );
                y[index].resize( 1, layer_dims[index] );
                delta[index].resize( 1, layer_dims[index] );
            }

            x.resize( 1, *(layer_dims.begin()) );
            e.resize( 1, *(layer_dims.rbegin()) );
            d.resize( 1, *(layer_dims.rbegin()) );
        }
    };

}//namespace f

#endif//LWJFYMMSNAJWAXFNVPOFRIXAWVMQOCFLQWSJNDHJFARFSDUSKCAVOODAFMMOYJLDTIEUFBSCE

