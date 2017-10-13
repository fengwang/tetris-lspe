#ifndef MBLKDIAG_HPP_INCLUDED_SOFIDJ4398YUAKFJH4UIHAKLFHSAKFJV4398UYAFSKHREOIUAOFIJWEORIUFDJKSHSFDV8
#define MBLKDIAG_HPP_INCLUDED_SOFIDJ4398YUAKFJH4UIHAKLFHSAKFJV4398UYAFSKHREOIUAOFIJWEORIUFDJKSHSFDV8

#include <f/matrix/matrix.hpp>
#include <f/matrix/functional/zeros.hpp>

#include <cstddef>

#if 0
        Function:
                    blkdiag
                    blk_diag
                    block_diag
        Comment:
                construct a block diagonal matrix
        Inputs:
                a series of matrices: A, B, C, D, ...
        Output:
                a combined block diagonal matrix
                [ A, 0, 0, 0, ... ]
                [ 0, B, 0, 0, ... ]
                [ 0, 0, C, 0, ... ]
                [ 0, 0, 0, D, ... ]
                [ .  .  .  .  ... ]
                [ .  .  .  .  ... ]
                [ .  .  .  .  ... ]
#endif

namespace f
{

    template<typename T1, std::size_t D1, typename A1,
             typename T2, std::size_t D2, typename A2 >
    matrix<T1, D1, A1> const blkdiag( const matrix<T1, D1, A1>& m1, const matrix<T2,D2,A2>& m2 )
    {
        return  ( m1                            || zeros(m1, m1.row(), m2.col()) ) &&
                ( zeros(m1, m2.row(), m1.col()) || m2                            );
    }


    template<typename T1, std::size_t D1, typename A1,
             typename T2, std::size_t D2, typename A2,
             typename ... Matrices>
    matrix<T1, D1, A1> const blkdiag( const matrix<T1, D1, A1>& m1, const matrix<T2,D2,A2>& m2, const Matrices& ... matrices )
    {
        return blkdiag( blkdiag(m1, m2), matrices... );
    }

    template<typename T, std::size_t D, typename A, typename ... Matrices>
    matrix<T, D, A> const blk_diag( const matrix<T,D,A>& m, const Matrices& ... matrices )
    {
        return blkdiag(m, matrices...);
    }

    template<typename T, std::size_t D, typename A, typename ... Matrices>
    matrix<T, D, A> const block_diag( const matrix<T,D,A>& m, const Matrices& ... matrices )
    {
        return blkdiag(m, matrices...);
    }

}//namesapce f

#endif//_BLKDIAG_HPP_INCLUDED_SOFIDJ4398YUAKFJH4UIHAKLFHSAKFJV4398UYAFSKHREOIUAOFIJWEORIUFDJKSHSFDV8

