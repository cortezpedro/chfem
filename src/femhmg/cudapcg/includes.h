#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>

#ifndef INCLUDES_CUDAPCG_H_INCLUDED
#define INCLUDES_CUDAPCG_H_INCLUDED

#define CUDAPCG_TRUE 1
#define CUDAPCG_FALSE 0

#define CUDAPCG_NBN 0
#define CUDAPCG_EBE 1

#define CUDAPCG_POREMAP_IMG 0
#define CUDAPCG_POREMAP_NUM 1

#define CUDAPCG_TOLERANCE 0.00001
#define CUDAPCG_MAX_ITERATIONS 100000

#define CUDAPCG_L2_NORM 0
#define CUDAPCG_INF_NORM 1
#define CUDAPCG_ERROR_NORM 2

#define CUDAPCG_THERMAL_2D 0
#define CUDAPCG_THERMAL_3D 1
#define CUDAPCG_ELASTIC_2D 2
#define CUDAPCG_ELASTIC_3D 3
#define CUDAPCG_FLUID_2D 4
#define CUDAPCG_FLUID_3D 5

#define CUDAPCG_DEFAULT_SOLVER 0
#define CUDAPCG_NOJACOBI_SOLVER 1
#define CUDAPCG_MINRES_SOLVER 2
// More solvers soon

#ifdef CUDAPCG_VAR_32BIT
  typedef float cudapcgVar_t; // 32bit IEEE754
  typedef float cudapcgTol_t; // 32bit IEEE754
  #define CUDA_ABS(x) fabsf(x)
#else // default is CUDAPCG_VAR_64BIT
  #ifndef CUDAPCG_VAR_64BIT
    #define CUDAPCG_VAR_64BIT
  #endif
  typedef double cudapcgVar_t; // 64bit IEEE754
  typedef double cudapcgTol_t; // 64bit IEEE754
  #define CUDA_ABS(x) fabs(x)
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

    char *name;

    cudapcgFlag_t freeAllowed_flag;
    cudapcgFlag_t parStrategy_flag;
    cudapcgFlag_t poremap_flag;

    unsigned int nrows;
    unsigned int ncols;
    unsigned int nlayers;

    unsigned int nelem;
    unsigned int nvars;

    unsigned int nkeys;
    unsigned int nlocalvars;
    unsigned int nporenodes;
    unsigned int nbordernodes;
    unsigned int nhmgvars;

    // gpu arrays to characterize the model
    cudapcgMap_t *image;
    
    cudapcgFlag_t *pore_map;          // used in fluid simulations
    cudapcgFlag_t *border_pore_map;   // used in fluid simulations
    
    cudapcgIdMap_t *periodic2DOF_map; // used in fluid simulations
    cudapcgIdMap_t *DOF2periodic_map; // used in fluid simulations

} cudapcgModel_t;

#endif // INCLUDES_CUDAPCG_H_INCLUDED
