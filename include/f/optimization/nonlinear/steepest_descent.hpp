#ifndef GLIPSTSAKNNRLWNNQKATRLDJOVFFQJDPYMRNUYWXADHJNUXPYUEXKJWQLVXFXTCLNJKBISPAK
#define GLIPSTSAKNNRLWNNQKATRLDJOVFFQJDPYMRNUYWXADHJNUXPYUEXKJWQLVXFXTCLNJKBISPAK

#include <f/matrix/matrix.hpp>
#include <f/derivative/derivative.hpp>

#include <algorithm>
#include <functional>
#include <cstddef>

namespace f
{

    template< typename T, typename Concret_Descent >
    struct default_steepest_descent
    {
        typedef T                                               value_type;
        typedef Concret_Descent                                 zen_type;
        typedef value_type*                                     pointer;
        typedef std::function<value_type(pointer)>              function_type;
        typedef std::size_t                                     size_type;

        function_type                                           merit_function;
        matrix<function_type>                                   derivative_function; //\delta f -- [n][1]
        size_type                                               variable_length;
        size_type                                               total_step;
        size_type                                               current_step;
        matrix<value_type>                                      direction;
        matrix<value_type>                                      gradient_k;
        matrix<value_type>                                      gradient_k_1;
        matrix<value_type>                                      initial_guess;      // [n][1]
        matrix<value_type>                                      current_solution;   // [n][1]
        matrix<value_type>                                      trial_solution;   // [n][1]
        value_type                                              step_size;
        value_type                                              eps;
        value_type                                              current_residual;
        matrix<value_type>                                      last_solution;   // [n][1]
        value_type                                              threshold;

        std::function<void(value_type, pointer)>                iteration_over_function;

        void make_step_size() //default
        {
            //std::cerr << "making step size.\n";
#if 1
            //value_type const max_step_size = 1.6180339887498948482;
            value_type const max_step_size = 3.1415926535897932384626433;
            //value_type const max_step_size = 3.1415926535897932384626433 * 1024.0;
#endif
#if 0
            value_type const nm = std::sqrt( std::inner_product( direction.begin(), direction.end(), direction.begin(), value_type{0} ) / static_cast<value_type>(variable_length) );
            value_type const max_step_size = std::min( 0.00186744273170798882 / nm, step_size * 6.28318530717958647692 );
#endif
            value_type const decrease_ratio = 0.6180339887498948482;
            step_size = max_step_size;
            unsigned long const max_steps = 97;

            for ( unsigned long index = 0; index != max_steps; ++index )
            {
                /*
                std::cerr << "\tbegin step size looping index " << index << " and step size " << step_size << "\n";
                std::cerr << "direction is \n" << transpose( direction ) << "\n";
                std::cerr << "current_solution is \n" << transpose( current_solution ) << "\n";
                */
                trial_solution = current_solution + step_size * direction;
                //std::cerr << "trial_solution is \n" << transpose( trial_solution ) << "\n";
                value_type const new_residual = merit_function( trial_solution.data() );
                //std::cerr << "\tnew residual evaluated as " << new_residual << "\n";
                //if ( new_residual < current_residual || step_size < eps )
                if ( new_residual < current_residual )
                {
                    current_residual = new_residual;
                    //std::cerr << "\t\tbreaking with new residual\n";
                    break;
                }
                if ( step_size < eps )
                {
                    step_size = 0.0;
                    //std::cerr << "\t\tbreaking with size less than eps\n";
                    break;
                }
                step_size *= decrease_ratio;

                //std::cerr << "\tstep index " << index << " to " << step_size <<  " from " << current_residual << " to " << new_residual << "\n";
            }
            //std::cerr << "End step size.\n";

        }

        template< typename Function >
        default_steepest_descent( Function const& func_, size_type len_, size_type loop_ = 1000 ) : merit_function{ func_ }, variable_length{ len_ }
        {
            //default derivatives
            derivative_function.resize( variable_length, 1 );

            for ( size_type index = 0; index != variable_length; ++index )
                derivative_function[index][0] = make_derivative( merit_function, index );

            current_step = 0;
            total_step = loop_;

            initial_guess.resize( variable_length, 1 );
            std::fill( initial_guess.begin(), initial_guess.end(), value_type{} );

            current_solution = initial_guess;
            last_solution = initial_guess;

            direction.resize( variable_length, 1 );

            eps = 1.0e-10;

            gradient_k.resize( variable_length, 1 );
            gradient_k_1.resize( variable_length, 1 );

            iteration_over_function = [](value_type,pointer){};

            step_size = 1.6180339887498948482;

            threshold = T{0};
        }

        template< typename F >
        void config_iteration_function( F func )
        {
            iteration_over_function = func;
        }

        void iterate()
        {
            //std::cerr << "Iterating\n";

            auto& zen = static_cast<zen_type&>(*this);
            zen.make_direction(); //
            zen.make_step_size(); //

            if ( step_size < eps ) return;

            current_solution += step_size * direction;
            ++current_step;

            iteration_over_function( current_residual, current_solution.data() );
#if 0
            std::cout.precision( 15 );
            std::cout << "\n\nstep is " << step_size;
            std::cout << "\ndirection is " << direction.transpose();
            std::cout << "\nresidual is " << current_residual;
            //std::cout << "\ncurrent_solution is " << current_solution.transpose();
            std::cout << "\n";
            //std::ofstream ofs( "solution.noisy.stro3_select_7_5.dat", std::fstream::app );
            //std::ofstream ofs( "solution.sto_accr.12.2.dat", std::fstream::app );
            //std::ofstream ofs( "solution.sto_accr.12.2.se.dat", std::fstream::app );
            //std::ofstream ofs( "solution.sto_accr.12.2.exp.dat", std::fstream::app );
            //std::ofstream ofs( "solution.9beam.12.2.dat", std::fstream::app );
            //std::ofstream ofs( "./solution.sto_accr.12.2.dat", std::fstream::app );
            //std::ofstream ofs( "./solution.sto_accr.12.2_abs_dec_1.dat", std::fstream::app );
            std::ofstream ofs( "./steepest.dump.dat", std::fstream::app );
            //std::ofstream ofs( "solution.9beam.12.2.dat", std::fstream::app );
            //std::ofstream ofs( "new.solution.9beam.12.2.dat", std::fstream::app );
            //std::ofstream ofs( "solution.71.simple.derivative.10.0.dat", std::fstream::app );
            //std::ofstream ofs( "solution.9beams.derivative.10.0.dat", std::fstream::app );
            //std::ofstream ofs( "solution.9beams.derivative.10.0.dat", std::fstream::app );
            ofs << current_residual << "\t\t" << current_solution.transpose() << "\n";
            ofs.close();
#endif
            //std::cerr << "Iterating End\n";
        }

        template< typename Otor >
        value_type operator()( Otor oi_ )
        {
            auto& zen = static_cast<zen_type&>(*this);

            current_solution = initial_guess;
            current_residual = merit_function( current_solution.data() );

            while ( current_step != total_step )
            {
                auto last_residual = merit_function( current_solution.data() );

                if ( last_residual < threshold )
                {
                    std::cout << "Solver breaks with small threshold." << std::endl;
                    break;
                }

                zen.iterate();
                value_type const direction_norm = std::inner_product( direction.begin(), direction.end(), direction.begin(), value_type{} );

                //if ( std::isnan(direction_norm) || std::isinf(direction_norm) ) break;
                if ( std::isnan(direction_norm) || std::isinf(direction_norm) )
                {
                    std::swap( current_solution, last_solution );
                    //std::cerr << "Steepest Descent terminate with nan/inf\n";
                    std::cout << "Solver breaks with nan/inf." << std::endl;
                    break;
                }

                //if ( direction_norm < eps*eps )
                if ( direction_norm < eps )
                {
                    //std::cerr << "Steepest Descent terminate with small direction norm " << direction_norm << "\n";
                    std::cout << "Solver breaks with small direction norm." << std::endl;
                    break;
                }

                last_solution = current_solution;

                auto new_residual = merit_function( last_solution.data() );

                if ( new_residual > last_residual )
                {
                    std::cout << "Solver breaks with larger new residual." << std::endl;
                    break;
                }

                //if ( std::abs( new_residual - last_residual ) < std::abs( last_residual * eps * eps) )
                if ( std::abs( new_residual - last_residual ) < std::abs( last_residual * eps) )
                {
                    std::cout << "Solver breaks with small gain." << std::endl;
                    break;
                }

                if ( step_size < eps )
                {
                    std::cout << "Solver breaks with small step size." << std::endl;
                    break;
                }
            }

            //if ( current_step >= total_step )
                //std::cerr << "Steepest Descent terminate with total step " << current_step << "\n";

            std::copy( current_solution.begin(), current_solution.end(), oi_ );
            return merit_function( &(current_solution[0][0]) );
        }

        void make_gradient()
        {
            for ( size_type i = 0; i != variable_length; ++i )
                gradient_k_1[i][0] = derivative_function[i][0](current_solution.data() );
        }

        void make_direction() //default
        {
            //std::cerr << "starting making direction \n";
            /*
            for ( size_type i = 0; i != variable_length; ++i )
                direction[i][0] = -derivative_function[i][0]( current_solution.data() );
            */

            gradient_k.swap( gradient_k_1 );

            auto& zen = static_cast<Concret_Descent&>(*this);
            zen.make_gradient();
            /*
            for ( size_type i = 0; i != variable_length; ++i )
                gradient_k_1[i][0] = derivative_function[i][0](current_solution.data() );
            */

            if ( 0 == current_step )
            {
                direction = -gradient_k_1;
                return;
            }

            value_type const beta_k1_k1 = std::inner_product( gradient_k_1.begin(), gradient_k_1.end(), gradient_k_1.begin(), value_type{} );
            value_type const beta_k_p = std::inner_product( gradient_k.begin(), gradient_k.end(), direction.begin(), value_type{} );
            value_type const beta_k1_p = std::inner_product( gradient_k_1.begin(), gradient_k_1.end(), direction.begin(), value_type{} );
            value_type const beta = beta_k1_k1 / ( beta_k1_p - beta_k_p );

            direction *= beta;
            direction -= gradient_k_1;

            //std::cerr << "End making direction \n";
        }

        void config_threshold( T threshold_ )
        {
            threshold = threshold_;
        }

        template< typename Function >
        void config_derivative_function( Function const& df_, size_type index_ )
        {
            assert( index_ < variable_length );
            derivative_function[index_][0] = df_;
        }

        template< typename Itor >
        void config_initial_guess( Itor begin_ )
        {
            for ( size_type index = 0; index != variable_length; ++index )
            {
                initial_guess[index][0] = *begin_;
                ++begin_;
            }

            current_step = 0;
            current_solution = initial_guess;
        }

        void config_initial_guess( value_type x )
        {
            initial_guess[0][0] = x;
            current_step = 0;
            current_solution = initial_guess;
        }

        void config_total_steps( size_type loop_ )
        {
            total_step = loop_;
        }

        void config_eps( value_type const eps_ )
        {
            eps = eps_;
        }

    };//struct default_steepest_descent

    template< typename T >
    struct steepest_descent : default_steepest_descent< T, steepest_descent<T> >
    {
        typedef std::size_t                                     size_type;

        template< typename Function >
        steepest_descent( Function const& func_, size_type len_, size_type loop_ = 1000 ) : default_steepest_descent< T, steepest_descent<T> >( func_, len_, loop_ ) {}
    };//struct steepest_descent

    template< typename T >
    //struct direct_steepest_descent : default_steepest_descent< T, steepest_descent<T> >
    struct direct_steepest_descent : default_steepest_descent< T, direct_steepest_descent<T> >
    {
        typedef std::size_t                                     size_type;
        typedef T                                               value_type;
        typedef value_type*                                     pointer;
        matrix<value_type>                                      ug_tmp{ (*this).variable_length, 1 };

        template< typename Function >
        //direct_steepest_descent( Function const& func_, size_type len_, size_type loop_ = 1000 ) : default_steepest_descent< T, steepest_descent<T> >( func_, len_, loop_ )
        direct_steepest_descent( Function const& func_, size_type len_, size_type loop_ = 1000 ) : default_steepest_descent< T, direct_steepest_descent<T> >( func_, len_, loop_ )
        {
            ug_tmp.resize( len_, 1 );
#if 0
            for ( size_type index = 0; index != (*this).variable_length; ++index )
                (*this).derivative_function[index][0] = [this, index]( pointer x )
                {
                    size_type n = (*this).variable_length;
                    value_type res = (*this).current_residual;
                    value_type step = value_type{ 1.0e-6 };
                    std::copy( x, x+n, (*this).ug_tmp.begin() );
                    (*this).ug_tmp[index][0] += step;
                    value_type new_res = (*this).merit_function( (*this).ug_tmp.data() );
                    return (new_res - (*this).current_residual) / step;
                };
#endif
        }

    };//struct direct_steepest_descent

    template< typename T >
    struct simple_steepest_descent : default_steepest_descent< T, simple_steepest_descent<T> >
    {
        typedef std::size_t                                     size_type;
        typedef T                                               value_type;
        typedef value_type*                                     pointer;
        typedef std::function<value_type(pointer)>              function_type;

        template< typename Function >
        simple_steepest_descent( Function const& func_, size_type len_, size_type loop_ = 1000 )
        : default_steepest_descent< T, simple_steepest_descent<T> >( func_, len_, loop_ ) {}

        void make_gradient()
        {
            auto const fxyz = (*this).merit_function( (*this).current_solution.data() );
            T const eps = 1.0e-7;

            for ( size_type i = 0; i != (*this).variable_length; ++i )
            {
                (*this).current_solution[i][0] -= eps;
                auto const fxyz_ = (*this).merit_function( (*this).current_solution.data() );

                (*this).gradient_k_1[i][0] = (fxyz - fxyz_) / eps;

                (*this).current_solution[i][0] += eps;
            }
        }

    };//struct simple_steepest_descent

}//namespace f

#endif//GLIPSTSAKNNRLWNNQKATRLDJOVFFQJDPYMRNUYWXADHJNUXPYUEXKJWQLVXFXTCLNJKBISPAK

