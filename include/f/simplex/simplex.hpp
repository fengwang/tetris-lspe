#ifndef NCQAIIGXYFCOJAANXNGVXKFNQOACKFRGJMTKCFQYLLMIUMROPLQYCTCGRDFWFBGATQBBFNUIX
#define NCQAIIGXYFCOJAANXNGVXKFNQOACKFRGJMTKCFQYLLMIUMROPLQYCTCGRDFWFBGATQBBFNUIX

#include <f/matrix/matrix.hpp>

#include <memory>
#include <map>
#include <functional>
#include <cassert>

namespace f
{
    template< typename T >
    struct simplex
    {
        typedef T                                               value_type;
        typedef value_type*                                     pointer;
        typedef std::function<value_type(pointer)>              function_type;
        typedef unsigned long                                   size_type;
        typedef matrix<value_type>                              matrix_type;
        typedef std::shared_ptr<matrix_type>                    matrix_pointer;
        typedef std::multimap<value_type, matrix_pointer>       solution_space_type;

        function_type           target_function;
        size_type               unknowns;
        matrix_pointer          centroid;                       
        matrix_pointer          reflect;                       
        matrix_pointer          expand;                       
        matrix_pointer          contract;                    
        solution_space_type     solution_space;
        value_type              alpha;
        value_type              gamma;
        value_type              rho;
        value_type              sigma;

        value_type              f_xr;                           //residual at reflection
        value_type              f_xc;                           //residual at constrain
        value_type              f_xe;                           //residual at expansion

        size_type               loops;

        template< typename Output_Iterator >
        void operator()( Output_Iterator oter )
        {
            for( size_type loop = 0; loop != loops; ++loop )
            {
                update_centroid();
                update_reflect();

                if ( f_xr >= (*(solution_space.begin())).first && f_xr < (*(++(solution_space.rbegin()))).first )
                {
                    pickup_relfect();
                    continue;
                }

                if ( f_xr < (*(solution_space.begin())).first )
                {
                    update_expand();

                    if ( f_xe < f_xr )
                    {
                        pickup_expand();
                        continue;
                    }

                    pickup_relfect();
                    continue;
                }

                update_contract();

                if ( f_xc <= (*(solution_space.rbegin())).first )
                {
                    pickup_contract();
                    continue;
                }

                shrink();
            }

            auto& best_solution = (*(solution_space.begin())).second;
            *oter++ = (*(solution_space.begin())).first;
            std::copy( (*best_solution).begin(), (*best_solution).end(), oter );
        }

        void update_centroid()
        {
            assert( unknowns + 1 == solution_space.size() );

            std::fill( (*centroid).begin(), (*centroid).end(), value_type{0} );

            for ( auto& elem : solution_space )
                *centroid += *(elem.second);

            *centroid -= *((*solution_space.rbegin()).second);

            if ( solution_space.size() )
                *centroid /= static_cast<value_type>( solution_space.size() );
        }

        void update_reflect()
        {
            reflect = std::make_shared<matrix_type>( 1, unknowns );
            *reflect = (*centroid) + alpha * ( (*centroid) - (*((*(solution_space.rbegin())).second) ) );
            f_xr = target_function( (*reflect).data() );
        }

        void update_expand()
        {
            expand = std::make_shared<matrix_type>( 1, unknowns );
            *expand = (*centroid) + gamma * ( (*centroid) - (*((*(solution_space.rbegin())).second) ) );
            f_xe = target_function( (*expand).data() );
        }

        void update_contract()
        {
            contract = std::make_shared<matrix_type>( 1, unknowns );
            *contract = (*centroid) + rho * ( (*centroid) - (*((*(solution_space.rbegin())).second) ) );
            f_xc = target_function( (*contract).data() );
        }

        void pickup_relfect()
        {
            assert( solution_space.size() == unknowns+1 );
            solution_space.insert( std::make_pair( f_xr, reflect ) );
            //solution_space[f_xr] = reflect;
            solution_space.erase( std::next(solution_space.rbegin()).base() );
        }

        void pickup_expand()
        {
            assert( solution_space.size() == unknowns+1 );
            solution_space.insert( std::make_pair( f_xe, expand ) );
            //solution_space[f_xe] = expand;
            solution_space.erase( std::next(solution_space.rbegin()).base() );
        }

        void pickup_contract()
        {
            assert( solution_space.size() == unknowns+1 );
            solution_space.insert( std::make_pair( f_xc, contract ) );
            //solution_space[f_xc] = contract;
            solution_space.erase( std::next(solution_space.rbegin()).base() );
        }

        void shrink()
        {
            solution_space_type solution_space_;

            auto mat_1 = (*(solution_space.begin())).second;
           
            for ( auto& elem : solution_space )
            {
                auto mat = std::make_shared<matrix_type>( 1, unknowns );
                *mat = (*mat_1) + sigma * ( (*(elem.second)) - (*mat_1) );
                //solution_space_[target_function((*mat).data())] = mat;
                solution_space_.insert( std::make_pair(target_function((*mat).data()), mat) );
            }

            solution_space.swap( solution_space_ );
        }

        template< typename Function > 
        simplex( Function target_function_, size_type unknowns_, size_type loops_, value_type const boundary_ ) : target_function( target_function_ ), unknowns( unknowns_ ), loops( loops_ )
        {
            alpha = value_type{1};
            gamma = value_type{2};
            rho = value_type{-0.5};
            sigma = value_type{ 0.5 };

            initialize_soluation_space( boundary_ ); 
        }

        void initialize_soluation_space( value_type const boundary )
        {
            for ( size_type index = 0; index != unknowns; ++index )
            {
                matrix_pointer p = std::make_shared<matrix_type>( 1, unknowns );
                std::fill( (*p).begin(), (*p).end(), 0.0 );
                (*p)[0][index] = boundary;
                //solution_space[target_function( (*p).data() )] = p;
                solution_space.insert(std::make_pair(target_function( (*p).data() ), p) );
            }
            {
                matrix_pointer p = std::make_shared<matrix_type>( 1, unknowns );
                std::fill( (*p).begin(), (*p).end(), boundary / value_type{2} );
                solution_space.insert( std::make_pair( target_function( (*p).data() ), p ) );
                //solution_space[target_function( (*p).data() )] = p;
            }

            centroid = std::make_shared<matrix_type>( 1, unknowns );
            reflect = std::make_shared<matrix_type>( 1, unknowns );
            expand = std::make_shared<matrix_type>( 1, unknowns );
            contract = std::make_shared<matrix_type>( 1, unknowns );
        }

    };

    template< typename Func, typename T >
    auto make_simplex( Func&& target_function_, unsigned long unknowns_, unsigned long loops_, T boundary_ )
    {
        return simplex<T>{ target_function_, unknowns_, loops_, boundary_ };
    }

}//namespace f

#endif//NCQAIIGXYFCOJAANXNGVXKFNQOACKFRGJMTKCFQYLLMIUMROPLQYCTCGRDFWFBGATQBBFNUIX

