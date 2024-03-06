
#ifndef CUDAPCG_KERNELS_UTILS_H_INCLUDED
#define CUDAPCG_KERNELS_UTILS_H_INCLUDED

// Predefined num of threads on each block (controlled in compiler)
#ifdef CUDAPCG_BLOCKDIM_1024
  #define THREADS_PER_BLOCK 1024
#elif CUDAPCG_BLOCKDIM_512
  #define THREADS_PER_BLOCK 512
#elif CUCUDAPCG_BLOCKDIM_256
  #define THREADS_PER_BLOCK 256
#elif CUDAPCG_BLOCKDIM_64
  #define THREADS_PER_BLOCK 64
#elif CUDAPCG_BLOCKDIM_32
  #define THREADS_PER_BLOCK 32
#else //default is 128
  #define THREADS_PER_BLOCK 128 // by choice. multiple of 32, up to 1024
#endif

//---------------------------------
///////////////////////////////////
//////// DOUBLE PRECISION /////////
/////////// ATOMIC ADD ////////////
///////////////////////////////////
//---------------------------------

/*
  CUDA did not implement atomicAdd for double precision IEEE754 floats
  prior to devices with cc 6.0

  The following conditional macro overloads the atomicAdd function to
  handle doubles when (__CUDA_ARCH__ < 600).

  Based on these stackoverflow discussions:
    [https://stackoverflow.com/questions/12626096/why-has-atomicadd-not-been-implemented-for-doubles]
    [https://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error]

  The function implemented below was adapted from this CUDA programming guide:
    [https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/CUDA_C_Programming_Guide.pdf]
    (Page 97)
*/

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#elif defined(CUDAPCG_VAR_64BIT)
__device__ double atomicAdd(double *at, double x){
    unsigned long long int* at_as_ull = (unsigned long long int*)at;
    unsigned long long int val = *at_as_ull, aux;
    do {
        aux = val;
        val = atomicCAS(at_as_ull, aux,__double_as_longlong(x+__longlong_as_double(aux)));
    } while (val != aux);
    return __longlong_as_double(val);
}
#endif

//---------------------------------
///////////////////////////////////
/////////// AUX MACROS ////////////
///////////////////////////////////
//---------------------------------

/*
    These macros are used in the Aprod kernels
    to navigate the structured grid, considering
    the adopted DOF numbering system (line major).
    
    Obs.: WALK_RIGHT and WALK_LEFT work with local
          indexes within a layer (3D).
*/
#define WALK_UP(id, nrows) (id-1+nrows*(!(id%nrows)))
#define WALK_DOWN(id, nrows) (id+1-nrows*(!((id+1)%nrows)))
#define WALK_RIGHT(id, nrows, nrowscols) ((id+nrows)%nrowscols)
#define WALK_LEFT(id, nrows, nrowscols) ((id+(nrowscols-nrows))%nrowscols)
#define WALK_FAR(id, nrowscols, nrowscolslayers) ((id+nrowscols)%nrowscolslayers)
#define WALK_NEAR(id, nrowscols, nrowscolslayers) ((id+(nrowscolslayers-nrowscols))%nrowscolslayers)

/*
    These macros reproduce the adopted DOF numbering
    system (line major), from [row,col,layer] indexes.
*/
#define PERIODICNUM_2D(row,col,nrows,ncols) ((row+nrows)%nrows+((col+ncols)%ncols)*nrows)
#define PERIODICNUM_3D(row,col,layer,nrows,ncols,nlayers) ((row+nrows)%nrows+((col+ncols)%ncols)*nrows+((layer+nlayers)%nlayers)*nrows*ncols)


#endif // CUDAPCG_KERNELS_UTILS_H_INCLUDED
