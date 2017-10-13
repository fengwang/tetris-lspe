#ifndef MSTATIC_CHECK_HPP_INCLUDED
#define MSTATIC_CHECK_HPP_INCLUDED


namespace f
{
    template<int> struct CompileTimeError;
    template<> struct CompileTimeError<true> {};
}
#define STATIC_CHECK(expr, msg) \
    { feng::CompileTimeError<((expr) != 0)> ERROR_##msg; (void)ERROR_##msg; }


#endif // _STATIC_CHECK_HPP_INCLUDED


