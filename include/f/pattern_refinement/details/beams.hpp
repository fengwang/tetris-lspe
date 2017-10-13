#ifndef FBDPVKNQARTENPUTHAGPFVQRXMISLKSBHNLAPTPFRVVHNPPYHFOYHYTDTPEVTTBTYXIFBEDHR
#define FBDPVKNQARTENPUTHAGPFVQRXMISLKSBHNLAPTPFRVVHNPPYHFOYHYTDTPEVTTBTYXIFBEDHR

#include <f/beam/beam_list.hpp>

namespace f
{

    template< typename Refinement >
    struct beams
    {
        typedef Refinement  zen_type;

        matrix<long>        the_beam_matrix;
        beam_list           the_beam_list;

        void load_beams()
        {
            auto& zen = static_cast<zen_type&>(*this);

            the_beam_matrix.load( zen.the_configuration.beam_path );
            the_beam_list = make_beam_list( zen.the_configuration.beam_path );
        }
    };

}//namespace f

#endif//FBDPVKNQARTENPUTHAGPFVQRXMISLKSBHNLAPTPFRVVHNPPYHFOYHYTDTPEVTTBTYXIFBEDHR

