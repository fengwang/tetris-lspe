#ifndef BUWEEPUJRBLGGHNFEIRAADIEPERBPTVLSDHGTMMTIBHAVJMQFBUGYIUOCSVWVHWXRKXFKESMC
#define BUWEEPUJRBLGGHNFEIRAADIEPERBPTVLSDHGTMMTIBHAVJMQFBUGYIUOCSVWVHWXRKXFKESMC

namespace f
{

    template< typename T = double >
    auto make_axis_parallel_hyper_ellipsoid( unsigned long n ) noexcept
    {
        return [n]( T* x ) noexcept
        {
            T ans{0};
            for ( unsigned long index = 0; index != n; ++index )
                ans += static_cast<T>( index+1 ) * x[index] * x[index];
            return ans;
        };
    }

}//namespace f

#endif//BUWEEPUJRBLGGHNFEIRAADIEPERBPTVLSDHGTMMTIBHAVJMQFBUGYIUOCSVWVHWXRKXFKESMC

