#ifndef URKKSSSOARGWXOFXFIWARWLJEFFPERMMTUPCIKEFFLBGUCYYWQBNHQUHXOUPKHKTLTUQTDXFK
#define URKKSSSOARGWXOFXFIWARWLJEFFPERMMTUPCIKEFFLBGUCYYWQBNHQUHXOUPKHKTLTUQTDXFK

#include <ctime>
#include <iomanip>
#include <iostream>

namespace f
{
    struct timer
    {
        typedef unsigned long clock_type;
        typedef float float_type;
        typedef double double_type;
        typedef long double long_double_type;
        std::clock_t t;

        template<typename T >
        timer( const T& val )
        {
            std::cout  << "[[ " << val << " ]]\t\t";
            t = std::clock();
        }

        timer()
        {
            t = std::clock();
        }

        operator clock_type () const
        {
            const std::clock_t _t = std::clock();
            const std::clock_t  d = _t - t;
            return d;
        }

        operator float_type () const
        {
            const std::clock_t _t = std::clock();
            const std::clock_t  d = _t - t;
            return  float_type(d) / CLOCKS_PER_SEC;
        }

        operator double_type () const
        {
            const std::clock_t _t = std::clock();
            const std::clock_t  d = _t - t;
            return  double_type(d) / CLOCKS_PER_SEC;
        }

        operator long_double_type () const
        {
            const std::clock_t _t = std::clock();
            const std::clock_t  d = _t - t;
            return  long_double_type(d) / CLOCKS_PER_SEC;
        }

        ~timer()
        {
            const std::clock_t _t = std::clock();
            const std::clock_t  d = _t - t;
            std::cout.precision(17);
            std::cout << std::setw(25);
            std::cout << static_cast<double>(d) << " clocks -- ";
            std::cout.precision(17);
            std::cout << std::setw(25);
            std::cout << static_cast<double>(d)/CLOCKS_PER_SEC << " second(s)" << std::endl;
        }

    };

}//namespace f

#endif//URKKSSSOARGWXOFXFIWARWLJEFFPERMMTUPCIKEFFLBGUCYYWQBNHQUHXOUPKHKTLTUQTDXFK

