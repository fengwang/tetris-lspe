#ifndef DSFJVUVFFGXWSMNHABGNWFIRAMGMLNDPMUVBVBRGLNGDLARKVGJLINGNFJMTYUMXTJQOBNPOF
#define DSFJVUVFFGXWSMNHABGNWFIRAMGMLNDPMUVBVBRGLNGDLARKVGJLINGNFJMTYUMXTJQOBNPOF

#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>
#include <cstdlib>

namespace cuda_assert_private
{

    struct cuda_result_assert
    {
        void operator()( const cudaError_t& result, const char* const file, const unsigned long line ) const
        {
            if ( 0 == result ) { return; }
            fprintf( stderr, "%s:%lu: cudaError occured:\n[[ERROR]]: %s\n", file, line, cudaGetErrorString(result) );
            abort();
        }

        void operator()( const CUresult& result, const char* const file, const unsigned long line ) const
        {
            if ( 0 == result ) { return; }

            const char* msg;
            cuGetErrorString( result, &msg );
            const char* name;
            cuGetErrorName( result, &name );

            fprintf( stderr, "%s:%lu: CUresult error occured:\n[[ERROR]]: %s --- %s\n", file, line, name, msg );
            abort();
        }

        void operator()( const nvrtcResult& result, const char* const file, const unsigned long line ) const
        {
            if ( 0 == result ) { return; }

            fprintf( stderr, "%s:%lu: nvrtcResult error occured:\n[[ERROR]]: %s\n", file, line, nvrtcGetErrorString(result) );
            abort();
        }

        void operator()( const cufftResult& result, const char* const file, const unsigned long line ) const
        {
            if ( 0 == result ) { return; }
            //no GetErrorString thing
            char const* cufft_error_string[] =
            {
                "Success",
                "CUFFT was passed an invalid plan handle",
                "CUFFT failed to allocate GPU or CPU memory",
                "No longer used",
                "User specified an invalid pointer or parameter",
                "Driver or internal CUFFT library error",
                "Failed to execute an FFT on the GPU",
                "The CUFFT library failed to initialize",
                "User specified an invalid transform size",
                "No longer used"
            };
            fprintf( stderr, "%s:%lu: nvrtcResult error occured:\n[[ERROR]]: %s\n", file, line, cufft_error_string[result] );
            abort();
        }
    };

}

#ifdef cuda_assert
#undef cuda_assert
#endif

#define cuda_assert(result) cuda_assert_private::cuda_result_assert{}(result, __FILE__, __LINE__)


#endif//DSFJVUVFFGXWSMNHABGNWFIRAMGMLNDPMUVBVBRGLNGDLARKVGJLINGNFJMTYUMXTJQOBNPOF

