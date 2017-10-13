#ifndef MGA_HPP_INCLUDED_SDPI9JH4389YFASLKJNVKLJNFADIUH439OY8HSFDLJUH498YHALFDKJ
#define MGA_HPP_INCLUDED_SDPI9JH4389YFASLKJNVKLJNFADIUH439OY8HSFDLJUH498YHALFDKJ

#include <f/genetic_algorithm/chromosome.hpp>
#include <f/genetic_algorithm/mutation_manager.hpp>
#include <f/genetic_algorithm/xover_manager.hpp>
#include <f/genetic_algorithm/probability_manager.hpp>

#include <f/algorithm/for_each.hpp>
#include <f/singleton/singleton.hpp>

#include <cstdlib>
#include <algorithm>
#include <limits>
#include <thread>

namespace f
{
/*
    typical usage:

    struct xx : public f::genetic_algorithm<xx>
    {
        std::size_t runtime() const // how long will the ga solver run, in seconds
        {
            return 100;//100s
        }

        std::size_t chromosome_length() const // how many genes are there in a single chromosome(individual), i.e. using how many unsigned long variable to represent a chromosome
        {
            return 2;
        }

        std::size_t population_size() const // how many individuals will be used during the evaluation
        {
            return 256;
        }

        double evaluate( unsigned long* v ) const   // given an array of unsigned long, evaluate its fitness, the better it fits the problem, the smaller the value should be returned
        {
            //...
        }

        void on_elite( unsigned long* v, double fitness ) const // what to do with the best fitted individual for the current generation?
        {
            //...
        }

    };//struct xx
*/
    template<typename Custom_GA>
    struct genetic_algorithm
    {
            typedef double              fitness_type;
            typedef unsigned long       value_type;
            typedef std::size_t         size_type;
            typedef Custom_GA           zen_type;

            void operator()( value_type* output ) const
            {
                zen_type const& cg = static_cast<zen_type const&>( *this );
                return cg.impl( output );
            }

            void on_elite( value_type* val, double fitness ) const
            {
            }

            void on_loop_over() const
            {}

            struct parallel_evaluator
            {
                template<typename Host, typename Iterator>
                void operator()(Host host, Iterator first, Iterator last) const
                {
                    for ( auto pch =first; pch != last; ++pch )
                        if ( (*pch).is_modified() )
                                (*pch).assign_fitness( host.evaluate( (*pch).data() ) );
                }
            };//struct parallel_evaluator

            //genetic algorithm here, as default implementation
            //will store the elite one to the output
            void impl( value_type* output) const
            {

                using namespace ga;

                std::srand ( unsigned ( std::time(0) ) );//for random_shuffle

                zen_type const& cg = static_cast<zen_type const&>( *this );
                //total individual number of a population
                const std::size_t population_per_generation = cg.population_size();
                //how long will the ga engine run
                const std::size_t run_time = cg.runtime();
                //the length of chromosome
                const std::size_t ch_length = cg.chromosome_length();

                //initialize time manager
                auto& t_manager = singleton<time_manager>::instance();
                t_manager( run_time );
                //initialize crossover manager
                auto& xs_manager = singleton<xover_selection_manager>::instance();
                xs_manager.initialize( population_per_generation );
                //initialize mutation manager
                auto& m_manager = singleton<mutation_manager>::instance();
                m_manager.initialize( population_per_generation );

                std::vector<chromosome> current_population( population_per_generation );
                std::vector<chromosome> selected_population_for_xover_father( population_per_generation/2 );
                std::vector<chromosome> selected_population_for_xover_mother( population_per_generation/2 );

                //random generate using resize method
                for ( size_type i = 0; i != population_per_generation; ++i )
                    current_population[i].resize( ch_length );

                //we can import population set from file cache
                //on_start();


                for ( size_type i = 0; i != population_per_generation/2; ++i )
                {
                    selected_population_for_xover_father[i].resize( ch_length );
                    selected_population_for_xover_mother[i].resize( ch_length );
                }

                fitness_type elite_fitness = std::numeric_limits<fitness_type>::max();
                std::vector<unsigned long> current_elite( ch_length );

                for ( ;; )
                {
#if 1 
                    //evaluate all the individuals in current_population
                    //TODO:
                    //     main thread should also do some calculation to make acceleration
                    size_type const NN = current_population.size();
                    std::thread t0( parallel_evaluator(), cg, current_population.begin(), current_population.begin()+NN/4 );
                    std::thread t1( parallel_evaluator(), cg, current_population.begin()+NN/4, current_population.begin()+NN/2 );
                    std::thread t2( parallel_evaluator(), cg, current_population.begin()+NN/2, current_population.begin()+NN/4+NN/2 );
                    //std::thread t3( parallel_evaluator(), cg, current_population.begin()+NN/2+NN/4, current_population.begin()+NN );
                    parallel_evaluator{}( cg, current_population.begin()+NN/2+NN/4, current_population.begin()+NN );
                    t0.join();
                    t1.join();
                    t2.join();
                    //t3.join();
#endif
#if 0
                    //single thread
                    parallel_evaluator()( cg, current_population.begin(), current_population.end() );
#endif

                    //sort
                    std::sort( current_population.begin(), current_population.end() );

                    //elite stuff
                    if ( current_population[0].fitness() < elite_fitness )
                    {
                        elite_fitness = current_population[0].fitness();
                        std::copy( current_population[0].begin(), current_population[0].end(), current_elite.begin() );
                        cg.on_elite( &(current_elite[0]), elite_fitness );
                    }

                    //if timeout, then return with the elite one
                    auto& t_manager = singleton<time_manager>::instance();
                    if ( t_manager.is_timeout() )
                    {
                        //std::copy( elite_individual.begin(), elite_individual.end(), output );
                        std::copy( current_elite.begin(), current_elite.end(), output );
                        return;
                    }

                    //unique
                    const unsigned long pos_of_unique = std::distance( current_population.begin(), std::unique( current_population.begin(), current_population.end() ) );

                    //if more than wanted, then shuffle and resize
                    if ( pos_of_unique > population_per_generation )
                    {
                        std::random_shuffle( current_population.begin(), current_population.begin()+pos_of_unique );
                        current_population.resize( population_per_generation );
                    }
                    //otherwise, randomly generate new individuals and unique
                    else
                    {
                        for ( unsigned long i = pos_of_unique; i != population_per_generation; ++i )
                        {
                            current_population[i].resize( ch_length );
                            cg.evaluate( current_population[i].data() );
                            (current_population[i]).mark_as_evaluated();
                        }
                    }

                    //sort again
                    std::sort( current_population.begin(), current_population.end() );

                    //select, all selected individual are stored in selected_population_for_xover_father and selected_population_for_xover_mother
                    auto& xpm = singleton<xover_probability_manager>::instance();
                    std::size_t const selection_number = xpm(population_per_generation); //parents to be selected
                    auto& xs_manager = singleton<xover_selection_manager>::instance();

                    selected_population_for_xover_father.clear();
                    selected_population_for_xover_mother.clear();
                    for ( std::size_t i = 0; i != selection_number; ++i )
                    {
                        const std::size_t select1 = xs_manager();
                        const std::size_t select2 = xs_manager();
                        selected_population_for_xover_father.push_back(current_population[select1]);
                        selected_population_for_xover_mother.push_back(current_population[select2]);
                    }

                    //xover, directly put new generation at the end of current_population
                    for ( std::size_t i = 0; i != selection_number; ++i )
                    {
                        chromosome son(ch_length);
                        chromosome daughter(ch_length);

                        for_each( selected_population_for_xover_father[i].begin(), selected_population_for_xover_father[i].end(),
                                  selected_population_for_xover_mother[i].begin(),
                                  son.begin(),
                                  daughter.begin(),
                                  single_point_xover()
                                );

                        current_population.push_back( son );
                        current_population.push_back( daughter );
                    }

                    //mutation
                    auto& mpm = singleton<mutation_probability_manager>::instance();
                    mpm.reset();
                    for ( auto& chromosome : current_population )
                        if ( mpm.should_mutate_current_gene() )
                        {
                            for ( auto& gene : chromosome )
                                binary_mutation()(gene);
                            //makr muated ones as not evaluated
                            chromosome.mark_as_modified();
                        }
                    //next loops

                    //store or analysis current chrosomeomse?
                    on_loop_over();
                }

                assert( !"should never reach here!" );
            }

    };//genetic_algorithm

}//namespace f

#endif//_GA_HPP_INCLUDED_SDPI9JH4389YFASLKJNVKLJNFADIUH439OY8HSFDLJUH498YHALFDKJ

