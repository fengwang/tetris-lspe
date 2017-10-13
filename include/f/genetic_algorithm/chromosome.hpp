#ifndef MCHROMOSOME_HPP_INCLUDED_SFPOINASFLKJH4EP9UHAFLDKJHASDFLKAHJSFLASKJDHDSA
#define MCHROMOSOME_HPP_INCLUDED_SFPOINASFLKJH4EP9UHAFLDKJHASDFLKAHJSFLASKJDHDSA 

#include <f/singleton/singleton.hpp>
#include <f/variate_generator/variate_generator.hpp>

#include <f/buffered_allocator/buffered_allocator.hpp>

#include <vector>
#include <memory>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iterator>

namespace f
{

    namespace ga
    {
        struct chromosome
        {
            typedef double                              value_type;
            //typedef std::vector<unsigned long, buffered_allocator<unsigned long, 2> >          gene_type;
            typedef std::vector<unsigned long >         gene_type;
            typedef gene_type::iterator                 iterator;

            std::shared_ptr<gene_type>                  chrome_;
            value_type                                  fit_;       //fitness, the smaller the better
            bool                                        modified_;  //modified flag after last fitness evaluation

            chromosome( const unsigned long n = 1 ) : chrome_(std::make_shared<gene_type>(n)), modified_(true)
            {
                //chrome_ = std::make_shared<gene_type>(n);
                //modified_ = true;
            }

            void resize( const unsigned long n )
            {
                chrome_ = std::make_shared<gene_type>(n);

                auto& vg = singleton<variate_generator<long double> >::instance();

                for ( unsigned long i = 0; i != n; ++i )
                {
                    //const long double val = vg() * static_cast<long double>(std::numeric_limits<unsigned long>::max());
                    const long double val = vg() * static_cast<long double>( 100.0 );
                    (*chrome_)[i] = static_cast<unsigned long>(val);
                }

                modified_ = true;

            }

            iterator begin()
            {
                return (*chrome_).begin();
            }

            iterator end()
            {
                return (*chrome_).end();
            }

            unsigned long * data()
            {
                return (*chrome_).data();
            }

            value_type fitness() const
            {
                assert( !is_modified() );
                return fit_;
            }

            void assign_fitness( const double fit )
            {
                fit_ = fit;
                modified_ = false;
            }

            bool is_modified() const
            {
                return modified_;
            }

            void mark_as_modified()
            {
                modified_ = true;
            }

            void mark_as_evaluated()
            {
                modified_ = false;
            }

            friend bool operator > ( const chromosome& lhs, const chromosome& rhs )
            {
                assert( !lhs.is_modified() );
                assert( !rhs.is_modified() );
                return lhs.fit_ > rhs.fit_;
            }

            friend bool operator < ( const chromosome& lhs, const chromosome& rhs )
            {
                assert( !lhs.is_modified() );
                assert( !rhs.is_modified() );
                return lhs.fit_ < rhs.fit_;
            }

            friend bool operator == ( const chromosome& lhs, const chromosome& rhs )
            {
                assert( !lhs.is_modified() );
                assert( !rhs.is_modified() );
                return lhs.fit_ == rhs.fit_ && lhs.chrome_ == rhs.chrome_;
            }

        }; //struct chromosome

    }//namemspace ga

}//namespace f

#endif//_CHROMOSOME_HPP_INCLUDED_SFPOINASFLKJH4EP9UHAFLDKJHASDFLKAHJSFLASKJDHDSA 

