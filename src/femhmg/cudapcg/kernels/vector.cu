#include "vector.h"
#include "utils.h"

//---------------------------------
///////////////////////////////////
////////// CUDA KERNELS ///////////
////////// (VECTOR OPS) ///////////
///////////////////////////////////
//---------------------------------

/*
    Relatively standard kernels, to perform
    vector operations. These are all generic
    (no compromises with the image-based
    matrix-free approach)
*/

//------------------------------------------------------------------------------
// Kernel to copy data from an array to another
template<typename T>
__global__ void kernel_arrcpy(T *v, unsigned int n, T *res){
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<n) res[i] = v[i];
}
template __global__ void kernel_arrcpy<float>(float *v, unsigned int n, float *res);
template __global__ void kernel_arrcpy<double>(double *v, unsigned int n, double *res);
//------------------------------------------------------------------------------
// Kernel to fill a vector with zeros
template<typename T>
__global__ void kernel_zeros(T * v, unsigned int n){
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<n) v[i] = 0.0;
}
template __global__ void kernel_zeros<float>(float *v, unsigned int n);
template __global__ void kernel_zeros<double>(double *v, unsigned int n);
//------------------------------------------------------------------------------
// Kernel to perform term-by-term multiplication between two vectors
template<typename T>
__global__ void kernel_termbytermmul(T * v1, T * v2, unsigned int n, T * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<n) res[i] = v1[i]*v2[i];
}
template __global__ void kernel_termbytermmul<float>(float *v1, float *v2, unsigned int n, float *res);
template __global__ void kernel_termbytermmul<double>(double *v1, double *v2, unsigned int n, double *res);
//------------------------------------------------------------------------------
// Kernel to perform term-by-term division between two vectors
template<typename T>
__global__ void kernel_termbytermdiv(T * v1, T * v2, unsigned int n, T * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<n) res[i] = v1[i]/v2[i];
}
template __global__ void kernel_termbytermdiv<float>(float *v1, float *v2, unsigned int n, float *res);
template __global__ void kernel_termbytermdiv<double>(double *v1, double *v2, unsigned int n, double *res);
//------------------------------------------------------------------------------
// Kernel to perform term-by-term inversion of a vector
template<typename T>
__global__ void kernel_termbyterminv(T * v, unsigned int n){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<n) v[i] = 1.0/v[i];
}
template __global__ void kernel_termbyterminv<float>(float *v, unsigned int n);
template __global__ void kernel_termbyterminv<double>(double *v, unsigned int n);
//------------------------------------------------------------------------------
// Kernel to sum two vectors, considering a scalar multiplier. res = y + a*x
template<typename T>
__global__ void kernel_saxpy(T * y, T * x, T a, unsigned int n, T * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<n) res[i] = y[i] + a*x[i];
}
template __global__ void kernel_saxpy<float>(float * y, float * x, float a, unsigned int n, float * res);
template __global__ void kernel_saxpy<double>(double * y, double * x, double a, unsigned int n, double * res);
//------------------------------------------------------------------------------
// Kernel to sum two vectors, considering a scalar multiplier.
// Result is stored into first array -> y += a*x
template<typename T>
__global__ void kernel_saxpy_iny(T * y, T * x, T a, unsigned int n){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<n) y[i] += a*x[i];
}
template __global__ void kernel_saxpy_iny<float>(float * y, float * x, float a, unsigned int n);
template __global__ void kernel_saxpy_iny<double>(double * y, double * x, double a, unsigned int n);
//------------------------------------------------------------------------------
#define REDUCE_WARP(cache,li)\
  if (li<16) cache[li] += cache[li+16]; __syncthreads();\
  if (li<8)  cache[li] +=  cache[li+8]; __syncthreads();\
  if (li<4)  cache[li] +=  cache[li+4]; __syncthreads();\
  if (li<2)  cache[li] +=  cache[li+2]; __syncthreads();\
  if (li<1)  cache[li] +=  cache[li+1]; __syncthreads();
//------------------------------------------------------------------------------
// Kernel to perform reduction of a vector, using shared mem
template<typename T>
__global__ void kernel_reduce(T *v, unsigned int n, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < n)
    cache[li] = (double)v[gi];
  else
    cache[li] = 0.0;
  __syncthreads();
  #if THREADS_PER_BLOCK >= 1024
    if (li<512) cache[li] += cache[li+512]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 512
    if (li<256) cache[li] += cache[li+256]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 256
    if (li<128) cache[li] += cache[li+128]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 128
    if (li<64) cache[li] += cache[li+64]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 64
    if (li<32) cache[li] += cache[li+32]; __syncthreads();
  #endif
  REDUCE_WARP(cache,li);
  if (li == 0)
    res[blockIdx.x] = cache[0];
}
template __global__ void kernel_reduce<float>(float *v, unsigned int n, double *res);
template __global__ void kernel_reduce<double>(double *v, unsigned int n, double *res);
//------------------------------------------------------------------------------
// Kernel to perform reduction of the abolute values in a vector, using shared mem
template<typename T>
__global__ void kernel_absreduce(T *v, unsigned int n, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < n)
    cache[li] = fabs((double)v[gi]);
  else
    cache[li] = 0.0;
  __syncthreads();
  #if THREADS_PER_BLOCK >= 1024
    if (li<512) cache[li] += cache[li+512]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 512
    if (li<256) cache[li] += cache[li+256]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 256
    if (li<128) cache[li] += cache[li+128]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 128
    if (li<64) cache[li] += cache[li+64]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 64
    if (li<32) cache[li] += cache[li+32]; __syncthreads();
  #endif
  REDUCE_WARP(cache,li);
  if (li == 0)
    res[blockIdx.x] = cache[0];
}
template __global__ void kernel_absreduce<float>(float *v, unsigned int n, double *res);
template __global__ void kernel_absreduce<double>(double *v, unsigned int n, double *res);
//------------------------------------------------------------------------------
// Kernel to perform dot product between two vectors, using shared mem
template<typename T>
__global__ void kernel_dotprod(T *v1, T *v2, unsigned int n, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < n)
    cache[li] = (double) (v1[gi]*v2[gi]);
  else
    cache[li] = 0.0;
  __syncthreads();
  #if THREADS_PER_BLOCK >= 1024
    if (li<512) cache[li] += cache[li+512]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 512
    if (li<256) cache[li] += cache[li+256]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 256
    if (li<128) cache[li] += cache[li+128]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 128
    if (li<64) cache[li] += cache[li+64]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 64
    if (li<32) cache[li] += cache[li+32]; __syncthreads();
  #endif
  REDUCE_WARP(cache,li);
  if (li == 0)
    res[blockIdx.x] = cache[0];
}
template __global__ void kernel_dotprod<float>(float *v1, float *v2, unsigned int n, double *res);
template __global__ void kernel_dotprod<double>(double *v1, double *v2, unsigned int n, double *res);
//------------------------------------------------------------------------------
#define REDUCE_WARP_MAX(cache,li)\
  if (li<16) cache[li] = cache[li+16] > cache[li] ? cache[li+16] : cache[li]; __syncthreads();\
  if (li<8)  cache[li] =  cache[li+8] > cache[li] ?  cache[li+8] : cache[li]; __syncthreads();\
  if (li<4)  cache[li] =  cache[li+4] > cache[li] ?  cache[li+4] : cache[li]; __syncthreads();\
  if (li<2)  cache[li] =  cache[li+2] > cache[li] ?  cache[li+2] : cache[li]; __syncthreads();\
  if (li<1)  cache[li] =  cache[li+1] > cache[li] ?  cache[li+1] : cache[li]; __syncthreads();
//------------------------------------------------------------------------------
// Kernel to get max value within an array in gpu
template<typename T>
__global__ void kernel_max(T *v, unsigned int n, double *res){
  // Get local and global thread index
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  // shmem cache
  __shared__ double cache[THREADS_PER_BLOCK];
  // Fill cache
  if (gi<n)
    cache[li] = (double)v[gi];
  else
    cache[li] = 0.0; // for safety
  __syncthreads();
  #if THREADS_PER_BLOCK >= 1024
    if (li<512) cache[li] = cache[li+512] > cache[li] ? cache[li+512] : cache[li]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 512
    if (li<256) cache[li] = cache[li+256] > cache[li] ? cache[li+256] : cache[li]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 256
    if (li<128) cache[li] = cache[li+128] > cache[li] ? cache[li+128] : cache[li]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 128
    if (li<64)  cache[li] =  cache[li+64] > cache[li] ?  cache[li+64] : cache[li]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 64
    if (li<32)  cache[li] =  cache[li+32] > cache[li] ?  cache[li+32] : cache[li]; __syncthreads();
  #endif
  REDUCE_WARP_MAX(cache,li);
  // Put cache results into result global mem arr
  if (li==0)
    res[blockIdx.x] = cache[0];
}
template __global__ void kernel_max<float>(float *v, unsigned int n, double *res);
template __global__ void kernel_max<double>(double *v, unsigned int n, double *res);
//------------------------------------------------------------------------------
// Kernel to get max absolute value within an array in gpu
template<typename T>
__global__ void kernel_absmax(T *v, unsigned int n, double *res){
  // Get local and global thread index
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  // shmem cache
  __shared__ double cache[THREADS_PER_BLOCK];
  // Fill cache
  if (gi<n)
    cache[li] = fabs((double)v[gi]);
  else
    cache[li] = 0.0; // for safety
  __syncthreads();
  #if THREADS_PER_BLOCK >= 1024
    if (li<512) cache[li] = cache[li+512] > cache[li] ? cache[li+512] : cache[li]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 512
    if (li<256) cache[li] = cache[li+256] > cache[li] ? cache[li+256] : cache[li]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 256
    if (li<128) cache[li] = cache[li+128] > cache[li] ? cache[li+128] : cache[li]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 128
    if (li<64)  cache[li] =  cache[li+64] > cache[li] ?  cache[li+64] : cache[li]; __syncthreads();
  #endif
  #if THREADS_PER_BLOCK >= 64
    if (li<32)  cache[li] =  cache[li+32] > cache[li] ?  cache[li+32] : cache[li]; __syncthreads();
  #endif
  REDUCE_WARP_MAX(cache,li);
  // Put cache results into result global mem arr
  if (li==0)
    res[blockIdx.x] = cache[0];
}
template __global__ void kernel_absmax<float>(float *v, unsigned int n, double *res);
template __global__ void kernel_absmax<double>(double *v, unsigned int n, double *res);
//------------------------------------------------------------------------------
