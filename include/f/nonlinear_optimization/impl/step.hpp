#ifndef CKPVHFFCSFEDOERVBGDHHEFFBPIXBTGIFXUMGWUQTKWDELVIEJMKQYWTFRBUYIGBLTGKAGYRB
#define CKPVHFFCSFEDOERVBGDHHEFFBPIXBTGIFXUMGWUQTKWDELVIEJMKQYWTFRBUYIGBLTGKAGYRB

namespace f
{

    template<typename T, typename Zen>
    struct step
    {
        typedef T                               value_type;
        typedef Zen                             zen_type;

        value_type make_step()
        {
            auto& zen = static_cast<zen_type&>(*this);
            return zen.setup_step();
        }

        value_type setup_step()
        {
            return value_type{1.0};
        }
    };//struct step

}//namespace f

#endif//CKPVHFFCSFEDOERVBGDHHEFFBPIXBTGIFXUMGWUQTKWDELVIEJMKQYWTFRBUYIGBLTGKAGYRB

