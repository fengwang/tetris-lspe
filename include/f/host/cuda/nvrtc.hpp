#ifndef NMKBPLSNAKLPWPGELNXHNIYOOADWRQEEYYOALJXVOYBGQJXINQMLXQPIMKLPYJPDVLOPYWFDA
#define NMKBPLSNAKLPWPGELNXHNIYOOADWRQEEYYOALJXVOYBGQJXINQMLXQPIMKLPYJPDVLOPYWFDA

extern "C"
{

    typedef enum
    {
        NVRTC_SUCCESS = 0,
        NVRTC_ERROR_OUT_OF_MEMORY = 1,
        NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
        NVRTC_ERROR_INVALID_INPUT = 3,
        NVRTC_ERROR_INVALID_PROGRAM = 4,
        NVRTC_ERROR_INVALID_OPTION = 5,
        NVRTC_ERROR_COMPILATION = 6,
        NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
        NVRTC_ERROR_INTERNAL_ERROR = 11
    } nvrtcResult;

    const char* nvrtcGetErrorString( nvrtcResult result );

    nvrtcResult nvrtcVersion( int* major, int* minor );

    typedef struct _nvrtcProgram* nvrtcProgram;

    nvrtcResult nvrtcCreateProgram( nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char* const* headers, const char* const* includeNames );

    nvrtcResult nvrtcDestroyProgram( nvrtcProgram* prog );

    nvrtcResult nvrtcCompileProgram( nvrtcProgram prog, int numOptions, const char* const* options );

    nvrtcResult nvrtcGetPTXSize( nvrtcProgram prog, size_t* ptxSizeRet );

    nvrtcResult nvrtcGetPTX( nvrtcProgram prog, char* ptx );

    nvrtcResult nvrtcGetProgramLogSize( nvrtcProgram prog, size_t* logSizeRet );

    nvrtcResult nvrtcGetProgramLog( nvrtcProgram prog, char* log );

    nvrtcResult nvrtcAddNameExpression( nvrtcProgram prog, const char* const name_expression );

    nvrtcResult nvrtcGetLoweredName( nvrtcProgram prog, const char* const name_expression, const char** lowered_name );

}

extern "C" int printf( const char* __restrict, ... );
extern "C" void abort (void);

#ifdef nvrtc_assert
#undef nvrtc_assert
#endif

struct nvrtc_result_assert
{
    void operator()( nvrtcResult const result, const char* const file, const unsigned long line ) const
    {
        if ( result != NVRTC_SUCCESS )
        {
            printf( "%s:%lu: NVRTC error occured:\n[[Error]]: %s\n", file, line, nvrtcGetErrorString(result) );
        }
    }
};

#define nvrtc_assert(result) nvrtc_result_assert{}( result, __FILE__, __LINE__ )

#endif//NMKBPLSNAKLPWPGELNXHNIYOOADWRQEEYYOALJXVOYBGQJXINQMLXQPIMKLPYJPDVLOPYWFDA

