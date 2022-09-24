
#ifndef CUDAPCG_KERNELS_UTILS_H_INCLUDED
#define CUDAPCG_KERNELS_UTILS_H_INCLUDED

// Predefined num of threads on each block (controlled in compiler)
#ifdef CUDAPCG_BLOCKDIM_1024
  #define THREADS_PER_BLOCK 1024
#elif CUCUDAPCG_BLOCKDIM_256
  #define THREADS_PER_BLOCK 256
#elif CUDAPCG_BLOCKDIM_128
  #define THREADS_PER_BLOCK 128
#elif CUDAPCG_BLOCKDIM_64
  #define THREADS_PER_BLOCK 64
#elif CUDAPCG_BLOCKDIM_32
  #define THREADS_PER_BLOCK 32
#else //default is 512
  #define THREADS_PER_BLOCK 512 // by choice. multiple of 32, up to 1024
#endif

//---------------------------------
///////////////////////////////////
/////////// AUX MACROS ////////////
///////////////////////////////////
//---------------------------------

/*
    These macros are used in the Aprod kernels
    to navigate the structured grid, considering
    the adopted DOF numbering system (ANDREASEN).
*/
#define WALK_UP(from, n_y) (from-1+n_y*(!(from%n_y)))
#define WALK_DOWN(from, n_y) (from+1-n_y*(!((from+1)%n_y)))
#define WALK_RIGHT(from, n_y, total) ((from+n_y)%total)
#define WALK_LEFT(from, n_x, n_y, total) ((from+(n_x-1)*n_y)%total)
#define WALK_FAR(from, n_xy, total) ((from+n_xy)%total)
#define WALK_NEAR(from, n_xy, n_z, total) ((from+(n_z-1)*n_xy)%total)


#endif // CUDAPCG_KERNELS_UTILS_H_INCLUDED
