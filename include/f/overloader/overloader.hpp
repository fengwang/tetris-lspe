#ifndef S8IJ498UALKAJSFDP4089UAFLDKJASDF098443I8ASLKJFDLKJASF908UJ4LIASF0934JFLKF
#define S8IJ498UALKAJSFDP4089UAFLDKJASDF098443I8ASLKJFDLKJASF908UJ4LIASF0934JFLKF

namespace f
{
    namespace overloader_imple_private_dsponjsa9oy8
    {

        template <typename Arg, typename... Args>
        struct overloader : overloader<Arg>, overloader<Args...>
        {
            overloader( Arg a_, Args... b_ ) noexcept : overloader<Arg>( a_ ), overloader<Args...>( b_... ) {}
        };

        template <typename Arg>
        struct overloader<Arg> : Arg
        {
            overloader( Arg a_ ) noexcept : Arg( a_ ) {}
        };

    }

    template <typename ... Overloaders>
    auto make_overloader( Overloaders ... overloader_ ) noexcept
    {
        return overloader_imple_private_dsponjsa9oy8::overloader<Overloaders...>( overloader_... );
    }

}//namespace scheme

#endif//S8IJ498UALKAJSFDP4089UAFLDKJASDF098443I8ASLKJFDLKJASF908UJ4LIASF0934JFLKF

