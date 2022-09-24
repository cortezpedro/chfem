/*

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>

#ifndef INCLUDES_CUDAPCG_H_INCLUDED
#define INCLUDES_CUDAPCG_H_INCLUDED

#define CUDAPCG_TRUE 1
#define CUDAPCG_FALSE 0

#define CUDAPCG_TOLERANCE 0.00001
#define CUDAPCG_MAX_ITERATIONS 100000

#define CUDAPCG_THERMAL_2D 0
#define CUDAPCG_THERMAL_3D 1
#define CUDAPCG_ELASTIC_2D 2
#define CUDAPCG_ELASTIC_3D 3

#define CUDAPCG_DEFAULT_SOLVER 0 // allocates x,r,d,q and a material map (image) on the GPU

#ifdef CUDAPCG_VAR_64BIT
  typedef double cudapcgVar_t; // 64bit IEEE754
  typedef double cudapcgTol_t; // 64bit IEEE754
#else // default is 32bit float
  typedef float cudapcgVar_t; // 32bit IEEE754
  typedef float cudapcgTol_t; // 32bit IEEE754
#endif

#ifdef CUDAPCG_MATKEY_64BIT
  typedef unsigned long int cudapcgMap_t;
  #define MATKEY_BITSTEP_2D 16
  #define MATKEY_BITSTEP_3D 8
  #define MATKEY_BITSTEP_RANGE_2D 65535
  #define MATKEY_BITSTEP_RANGE_3D 255
#elif CUDAPCG_MATKEY_32BIT
  typedef unsigned int cudapcgMap_t;
  #define MATKEY_BITSTEP_2D 8
  #define MATKEY_BITSTEP_3D 4
  #define MATKEY_BITSTEP_RANGE_2D 255
  #define MATKEY_BITSTEP_RANGE_3D 15
#elif CUDAPCG_MATKEY_8BIT
  typedef unsigned char cudapcgMap_t;
  #define MATKEY_BITSTEP_2D 2
  #define MATKEY_BITSTEP_3D 1
  #define MATKEY_BITSTEP_RANGE_2D 3
  #define MATKEY_BITSTEP_RANGE_3D 1
#else // default is 16bit
  typedef unsigned short int cudapcgMap_t;
  #define MATKEY_BITSTEP_2D 4
  #define MATKEY_BITSTEP_3D 2
  #define MATKEY_BITSTEP_RANGE_2D 15
  #define MATKEY_BITSTEP_RANGE_3D 3
#endif

// Ceil unsigned int division - for cuda block dimensioning
#define CEIL(num, div) ((num-1)/div+1)

typedef unsigned char cudapcgFlag_t; // 8bit
typedef unsigned int cudapcgIdMap_t; // 32bit

typedef struct _pcgmodel{

    cudapcgFlag_t freeAllowed_flag;

    unsigned int nrows;
    unsigned int ncols;
    unsigned int nlayers;
    
    unsigned int nelem;
    unsigned int nvars;
    
    unsigned int nkeys;
    unsigned int nlocalvars;
    
    cudapcgMap_t * image;

} cudapcgModel_t;

#endif // INCLUDES_CUDAPCG_H_INCLUDED
