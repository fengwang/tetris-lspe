#ifndef UHBTSNMDSNJTVHKYSLLTQYUYXTDRTGKORYQGOIWPFAOAXTPBPJQFYWRFPBDMYDGDHIFUJPJQU
#define UHBTSNMDSNJTVHKYSLLTQYUYXTDRTGKORYQGOIWPFAOAXTPBPJQFYWRFPBDMYDGDHIFUJPJQU

#include <atomic>

namespace f
{

    struct spin_lock
    {
        void lock() 
        {
            while (locked.test_and_set(std::memory_order_acquire)) { true; }
        }

        void unlock() 
        {
            locked.clear(std::memory_order_release);
        }

        std::atomic_flag locked = ATOMIC_FLAG_INIT;;
    };

}//namespace f

#endif//UHBTSNMDSNJTVHKYSLLTQYUYXTDRTGKORYQGOIWPFAOAXTPBPJQFYWRFPBDMYDGDHIFUJPJQU

