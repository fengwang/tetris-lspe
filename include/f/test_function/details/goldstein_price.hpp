#ifndef ILNNEOSEOOMMODKYELUPWXBQHNHPRXKOMPNSDHUHXGXUSCNSPDQLOMCSYXYKLKIGQFOTKILLV
#define ILNNEOSEOOMMODKYELUPWXBQHNHPRXKOMPNSDHUHXGXUSCNSPDQLOMCSYXYKLKIGQFOTKILLV

namespace f
{
    template< typename T = double >
    auto make_goldstein_price() noexcept
    {
        return []( T* x ) noexcept
        {
            T const x1 = x[0];
            T const x2 = x[1];
            T const x11 = x1*x1;
            T const x12 = x1*x2;
            T const x22 = x2*x2;
            T const term1 = x1 + x2 + T{1};
            T const term2 = x1+x1-x2-x2-x2;
            return ( T{1} + term1*term1*( T{19} - T{14}*x1 + T{3}*x11 - T{14}*x2 + T{6}*x12 + T{3}*x22) ) *
                   ( T{30} + term2*term2*( T{18} - T{32}*x1 + T{12}*x11 + T{48}*x2 - T{36}*x12 + T{27}*x22 ) );
        };
    }

}//namespace f

#endif//ILNNEOSEOOMMODKYELUPWXBQHNHPRXKOMPNSDHUHXGXUSCNSPDQLOMCSYXYKLKIGQFOTKILLV

