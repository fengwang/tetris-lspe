#ifndef OJHCDSOFSDMBWQPPWAQXJSAHLOKYKASXCLSYYPCXTDHIESHTRCGBEAXPGVIDTXDUJISNLWHCH
#define OJHCDSOFSDMBWQPPWAQXJSAHLOKYKASXCLSYYPCXTDHIESHTRCGBEAXPGVIDTXDUJISNLWHCH

namespace f
{
    namespace fifth_de_jong_private
    {
        template<typename T>
        T power_6( T x ) noexcept
        {
            T const xx = x * x;
            return xx * xx * xx;
        }
    }
    template<typename T = double>
    auto make_fifth_de_jong()
    {
        return []( T * x ) noexcept
        {
            T const a[2][25] = {  { -32, -16,   0,  16,  32, -32, -16,   0,  16,  32, -32, -16, 0, 16, 32, -32, -16,  0, 16, 32, -32, -16,  0, 16, 32 },
                                  { -32, -32, -32, -32, -32, -16, -16, -16, -16, -16,   0,   0, 0,  0,  0,  16,  16, 16, 16, 16,  32,  32, 32, 32, 32 } };
            T ans{ 0.002 };
            for ( unsigned long index = 0; index != 25; ++index )
            {
               T const tmp = T{index+1} + power_6( x[0] - a[0][index] ) + power_6( x[1] - a[1][index] );
               ans += T{1} / tmp;
            }

            return T{1} / ans;
        };
    }

}//namespace f

#endif//OJHCDSOFSDMBWQPPWAQXJSAHLOKYKASXCLSYYPCXTDHIESHTRCGBEAXPGVIDTXDUJISNLWHCH

