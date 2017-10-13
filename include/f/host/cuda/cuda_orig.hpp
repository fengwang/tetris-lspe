#ifndef XCEGHXRSDHDENNCPKYAXKLKPNNYYCYDOHUVUXBXHBHJEWBYEODXMWNJJTIJMYPTVFUTIPBQFJ
#define XCEGHXRSDHDENNCPKYAXKLKPNNYYCYDOHUVUXBXHBHJEWBYEODXMWNJJTIJMYPTVFUTIPBQFJ

extern "C"
{

	typedef int CUdevice;                                     /**< CUDA device */
	typedef struct CUctx_st *CUcontext;                       /**< CUDA context */
	typedef struct CUmod_st *CUmodule;                        /**< CUDA module */
	typedef struct CUfunc_st *CUfunction;                     /**< CUDA function */
	typedef struct CUarray_st *CUarray;                       /**< CUDA array */
	typedef struct CUmipmappedArray_st *CUmipmappedArray;     /**< CUDA mipmapped array */
	typedef struct CUtexref_st *CUtexref;                     /**< CUDA texture reference */
	typedef struct CUsurfref_st *CUsurfref;                   /**< CUDA surface reference */
	typedef struct CUevent_st *CUevent;                       /**< CUDA event */
	typedef struct CUstream_st *CUstream;                     /**< CUDA stream */
	typedef struct CUgraphicsResource_st *CUgraphicsResource; /**< CUDA graphics interop resource */
	typedef unsigned long long CUtexObject;                   /**< An opaque value that represents a CUDA texture object */
	typedef unsigned long long CUsurfObject;                  /**< An opaque value that represents a CUDA surface object */

	typedef struct CUuuid_st {                                /**< CUDA definition of UUID */
		char bytes[16];
	} CUuuid;

}

#endif//XCEGHXRSDHDENNCPKYAXKLKPNNYYCYDOHUVUXBXHBHJEWBYEODXMWNJJTIJMYPTVFUTIPBQFJ

