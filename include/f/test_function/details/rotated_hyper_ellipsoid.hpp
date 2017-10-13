#ifndef ANWUQOSUIIBYTKKAGMTCCWPPDCNCTDMHUBECKAUDFTQRFGUOUUDYRJHRSUQODSYEDGHBEEFRQ
#define ANWUQOSUIIBYTKKAGMTCCWPPDCNCTDMHUBECKAUDFTQRFGUOUUDYRJHRSUQODSYEDGHBEEFRQ

namespace f
{

    template< typename T = double >
    auto make_rotated_hyper_ellipsoid( unsigned long n ) noexcept
    {
        return [n]( T* x ) noexcept
        {
            T ans{0};
            for ( unsigned long index = 0; index != n; ++index )
                for ( unsigned long jndex = 0; jndex != index; ++jndex )
                    ans += x[index]*x[index]; 

            return ans;
        };
    }

}//namespace f

#endif//ANWUQOSUIIBYTKKAGMTCCWPPDCNCTDMHUBECKAUDFTQRFGUOUUDYRJHRSUQODSYEDGHBEEFRQ

