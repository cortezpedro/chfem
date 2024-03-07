/*
  =====================================================================
  Universidade Federal Fluminense (UFF) - Niteroi, Brazil
  Institute of Computing
  Authors: Cortez Lopes, P., Pereira., A.
  contact: pedrocortez@id.uff.br

  [cudapcg]
  
  History:
    * v1.0 (nov/2020) [CORTEZ] -> CUDA, PCG in GPU
    * v1.1 (sep/2022) [CORTEZ] -> Added permeability, MINRES.
                                  atomicAdd for EBE.
                                  refactoring of kernels for readability.
  
  Pre-history:
    Initially developed as a final work for the graduate course "Arquitetura
    e Programacao de GPUs", at the Institute of Computing, UFF.

  API for solving linear systems associated to FEM models with an matrix-free
  solvers, using CUDA. All global matrix operations involve "assembly on-the-fly"

  THERMAL CONDUCTIVITY, LINEAR ELASTICITY, ABSOLUTE PERMEABILITY.
  
  The NBN and EBE kernels are implemented here. These should be invoked
  via the wrapper host functions in "cudapcg_kernels_wrappers.cu".

  =====================================================================
*/

#include "matrix_free.h"
#include "utils.h"
#include "../error_handling.h"

#if defined CUDAPCG_MATKEY_8BIT
  __constant__ cudapcgVar_t K[1152];// max size of K is 576*2 (2 materials on elastic_3D)
#elif !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT
  __constant__ cudapcgVar_t K[2304];// max size of K is 576*4 (4 materials on elastic_3D)
#endif

void setConstantLocalK(cudapcgVar_t * lclK, unsigned long int size){
  #if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT
    HANDLE_ERROR(cudaMemcpyToSymbol(K,lclK,size));
  #else
    printf("WARNING: Storing local matrices in constant memory cache is not supported for material keys with 32bits or 64bits.\n");
  #endif
  return;
}

// Macro for shmem warp reduce
#define REDUCE_WARP(cache,li)\
  if (li<16) cache[li] += cache[li+16]; __syncthreads();\
  if (li<8)  cache[li] +=  cache[li+8]; __syncthreads();\
  if (li<4)  cache[li] +=  cache[li+4]; __syncthreads();\
  if (li<2)  cache[li] +=  cache[li+2]; __syncthreads();\
  if (li<1)  cache[li] +=  cache[li+1]; __syncthreads();

//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
////////////// MATRIX-VECTOR PRODDS AND PRECONDITIONER ///////////////
///////////////////////// (DEVICE FUNCTIONS) /////////////////////////
//////////////////////////////////////////////////////////////////////
//--------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
////////// NODE-BY-NODE ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
__device__ void Preconditioner_thermal_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t matkey, cudapcgVar_t *q){
  // Local var to store index in local matrix
  unsigned int mat;
  // Local var to store result
  cudapcgVar_t res=0.0;

  mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_2D))*16;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16+5;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16+10;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16+15;
  res += K[mat];

  *q = 1.0/res;
}
//------------------------------------------------------------------------------
__device__ void Aprod_thermal_2D_NodeByNode(unsigned int i,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res=0.0;
  // Local var to store index in local matrix
  unsigned int mat;
  // Get row and col ids
  int row = i%rows, col = i/rows;
  // Local var to store neighbor dof indexes
  unsigned int dof;
  // Go through neighbor elems
  #pragma unroll
  for (int e=0; e<4; e++){
      row += (e==2);
      col += (e==3)-(e==1);
      mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_2D))*16 + e*4;
      dof = PERIODICNUM_2D(row,col,rows,cols);     // [row,col]
      res += K[mat] * v[dof];
      dof = PERIODICNUM_2D(row,col+1,rows,cols);   // [row,col] -> WALK_RIGHT
      res += K[mat+1] * v[dof];
      dof = PERIODICNUM_2D(row-1,col+1,rows,cols); // [row,col] -> WALK_RIGHT, WALK_UP
      res += K[mat+2] * v[dof];
      dof = PERIODICNUM_2D(row-1,col,rows,cols);   // [row,col] -> WALK_UP
      res += K[mat+3] * v[dof];
      matkey >>= MATKEY_BITSTEP_2D;
  }
  *q = res;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_thermal_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t matkey, cudapcgVar_t *q){
  // Local var to store index in local matrix
  unsigned int mat;
  // Local var to store result
  cudapcgVar_t res=0.0;

  mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*64;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*64+9;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*64+18;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*64+27;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*64+36;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*64+45;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*64+54;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*64+63;
  res += K[mat];
  
  *q = 1.0/res;
}
//------------------------------------------------------------------------------
__device__ void Aprod_thermal_3D_NodeByNode(unsigned int i,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q){
  // Local vars to store result
  cudapcgVar_t res=0.0;
  // Local var to store index in local matrix
  unsigned int mat;
  // Get row and col ids
  int row = i%rows, col = (i%(rows*cols))/rows, layer = i/(rows*cols);
  // Local var to store neighbor dof indexes
  unsigned int dof;
  // Go through neighbor elems
  #pragma unroll
  for (int e=0; e<8; e++){
      row += (e==2)-(e==4)+(e==6);
      col += (e==3)-(e==1)+(e==7)-(e==5);
      layer -= (e==4);
      mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*64 + e*8;
      dof = PERIODICNUM_3D(row,col,layer,rows,cols,layers);       // [row,col,layer]
      res +=   K[mat]*v[dof];
      dof = PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);     // [row,col,layer] -> WALK_RIGHT
      res += K[mat+1]*v[dof];
      dof = PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers);   // [row,col,layer] -> WALK_RIGHT, WALK_UP
      res += K[mat+2]*v[dof];
      dof = PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);     // [row,col,layer] -> WALK_UP
      res += K[mat+3]*v[dof];
      dof = PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);     // [row,col,layer] -> WALK_FAR
      res += K[mat+4]*v[dof];
      dof = PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers);   // [row,col,layer] -> WALK_FAR, WALK_RIGHT
      res += K[mat+5]*v[dof];
      dof = PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers); // [row,col,layer] -> WALK_FAR, WALK_RIGHT, WALK_UP
      res += K[mat+6]*v[dof];
      dof = PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers);   // [row,col,layer] -> WALK_FAR, WALK_UP
      res += K[mat+7]*v[dof];
      matkey >>= MATKEY_BITSTEP_3D;
  }
  *q = res;
}
//------------------------------------------------------------------------------
// Considers hard-coded analytical solution for Q1 elem
__device__ void Preconditioner_thermal_3D_NodeByNode_SparseQ1(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t matkey, cudapcgVar_t *q){
  // Local var to store index in local matrix
  unsigned int mat;
  // Local var to store result
  cudapcgVar_t res=0.0;

  mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*40;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+5;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+11;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+16;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+23;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+28;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+34;
  res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+39;
  res += K[mat];
  
  *q = 1.0/res;
}
//------------------------------------------------------------------------------
// Considers hard-coded analytical solution for Q1 elem
__device__ void Aprod_thermal_3D_NodeByNode_SparseQ1(unsigned int i,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q){
  /*
    ATTENTION: Analytical solution is cosidered, zeros are not stored.

    k_el = material_prop * {
      {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
      {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
      { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
      {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
      {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
      { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
      { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
      { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
    }
  */
  // Local vars to store result
  cudapcgVar_t res=0.0;
  // Local var to store index in local matrix
  unsigned int mat;
  // Get row and col ids
  int row = i%rows, col = (i%(rows*cols))/rows, layer = i/(rows*cols);
  // Local var to store neighbor dof indexes
  unsigned int dof;
  // Iteration on first neighbor elem
  mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*40;
  dof = i;                                                    // [row,col,layer]
  res +=   K[mat] * v[dof];
  dof = PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers);   // [row,col,layer] -> WALK_RIGHT, WALK_UP
  res += K[mat+1] * v[dof];
  dof = PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers);   // [row,col,layer] -> WALK_FAR, WALK_RIGHT
  res += K[mat+2] * v[dof];
  dof = PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers); // [row,col,layer] -> WALK_FAR, WALK_RIGHT, WALK_UP
  res += K[mat+3] * v[dof];
  dof = PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers);   // [row,col,layer] -> WALK_FAR, WALK_UP
  res += K[mat+4] * v[dof];
  // Iteration on second neighbor elem
  mat = ((unsigned int) ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D))*40;
  col-=1;
  dof = i;                                                    // [row,col,layer] -> WALK_RIGHT
  res += K[mat+5] * v[dof];
  dof = PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);     // [row,col,layer] -> WALK_UP
  res += K[mat+6] * v[dof];
  dof = PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);     // [row,col,layer] -> WALK_FAR
  res += K[mat+7] * v[dof];
  dof = PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers); // [row,col,layer] -> WALK_FAR, WALK_RIGHT, WALK_UP
  res += K[mat+8] * v[dof];
  dof = PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers);   // [row,col,layer] -> WALK_FAR, WALK_UP
  res += K[mat+9] * v[dof];
  // Iteration on third neighbor elem
  mat = ((unsigned int) ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D))*40;
  row+=1;
  dof = PERIODICNUM_3D(row,col,layer,rows,cols,layers);     // [row,col,layer]
  res += K[mat+10] * v[dof];
  dof = i;                                                  // [row,col,layer] -> WALK_RIGHT, WALK_UP
  res += K[mat+11] * v[dof];
  dof = PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);   // [row,col,layer] -> WALK_FAR
  res += K[mat+12] * v[dof];
  dof = PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers); // [row,col,layer] -> WALK_FAR, WALK_RIGHT
  res += K[mat+13] * v[dof];
  dof = PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers); // [row,col,layer] -> WALK_FAR, WALK_UP
  res += K[mat+14] * v[dof];
  // Iteration on fourth neighbor elem
  mat = ((unsigned int) ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D))*40;
  col+=1;
  dof = PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);     // [row,col,layer] -> WALK_RIGHT
  res += K[mat+15] * v[dof];
  dof = i;                                                    // [row,col,layer] -> WALK_UP
  res += K[mat+16] * v[dof];
  dof = PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);     // [row,col,layer] -> WALK_FAR
  res += K[mat+17] * v[dof];
  dof = PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers);   // [row,col,layer] -> WALK_FAR, WALK_RIGHT
  res += K[mat+18] * v[dof];
  dof = PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers); // [row,col,layer] -> WALK_FAR, WALK_RIGHT, WALK_UP
  res += K[mat+19] * v[dof];
  // Iteration on fifth neighbor elem
  mat = ((unsigned int) ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D))*40;
  row-=1; layer-=1;
  dof = PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);     // [row,col,layer] -> WALK_RIGHT, WALK_UP
  res += K[mat+20] * v[dof];
  dof = PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers);   // [row,col,layer] -> WALK_RIGHT, WALK_UP
  res += K[mat+21] * v[dof];
  dof = PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);     // [row,col,layer] -> WALK_UP
  res += K[mat+22] * v[dof];
  dof = i;                                                    // [row,col,layer] -> WALK_FAR
  res += K[mat+23] * v[dof];
  dof = PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers); // [row,col,layer] -> WALK_FAR, WALK_RIGHT, WALK_UP
  res += K[mat+24] * v[dof];
  // Iteration on sixth neighbor elem
  mat = ((unsigned int) ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D))*40;
  col-=1;
  dof = PERIODICNUM_3D(row,col,layer,rows,cols,layers);     // [row,col,layer]
  res += K[mat+25] * v[dof];
  dof = PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers); // [row,col,layer] -> WALK_RIGHT, WALK_UP
  res += K[mat+26] * v[dof];
  dof = PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);   // [row,col,layer] -> WALK_UP
  res += K[mat+27] * v[dof];
  dof = i;                                                  // [row,col,layer] -> WALK_FAR, WALK_RIGHT
  res += K[mat+28] * v[dof];
  dof = PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers); // [row,col,layer] -> WALK_FAR, WALK_UP
  res += K[mat+29] * v[dof];
  // Iteration on seventh neighbor elem
  mat = ((unsigned int) ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D))*40;
  row+=1;
  dof = PERIODICNUM_3D(row,col,layer,rows,cols,layers);   // [row,col,layer]
  res += K[mat+30] * v[dof];
  dof = PERIODICNUM_3D(row,col+1,layer,rows,cols,layers); // [row,col,layer] -> WALK_RIGHT
  res += K[mat+31] * v[dof];
  dof = PERIODICNUM_3D(row-1,col,layer,rows,cols,layers); // [row,col,layer] -> WALK_UP
  res += K[mat+32] * v[dof];
  dof = PERIODICNUM_3D(row,col,layer+1,rows,cols,layers); // [row,col,layer] -> WALK_FAR
  res += K[mat+33] * v[dof];
  dof = i;                                                // [row,col,layer] -> WALK_FAR, WALK_RIGHT, WALK_UP
  res += K[mat+34] * v[dof];
  // Iteration on eighth neighbor elem
  mat = ((unsigned int) ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D))*40;
  col+=1;
  dof = PERIODICNUM_3D(row,col,layer,rows,cols,layers);     // [row,col,layer]
  res += K[mat+35] * v[dof];
  dof = PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);   // [row,col,layer] -> WALK_RIGHT
  res += K[mat+36] * v[dof];
  dof = PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers); // [row,col,layer] -> WALK_RIGHT, WALK_UP
  res += K[mat+37] * v[dof];
  dof = PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers); // [row,col,layer] -> WALK_FAR, WALK_RIGHT
  res += K[mat+38] * v[dof];
  dof = i;                                                  // [row,col,layer] -> WALK_FAR, WALK_UP
  res += K[mat+39] * v[dof];

  *q = res;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_elastic_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t matkey, cudapcgVar_t *q){
  // Local var to store index in local matrix
  unsigned int mat;
  // Local var to store result
  cudapcgVar_t res_x=0.0, res_y=0.0;

  mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_2D))*64;
  res_x =  K[mat]; res_y =  K[mat+9];
  mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64+18;
  res_x += K[mat]; res_y += K[mat+9];
  mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64+36;
  res_x += K[mat]; res_y += K[mat+9];
  mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64+54;
  res_x += K[mat]; res_y += K[mat+9];

  *q     = 1.0/res_x;
  *(++q) = 1.0/res_y;
}
//------------------------------------------------------------------------------
#define APROD_ELASTIC_2D_NODE(N)\
  v_x = v[dof]; v_y = v[dof+1];\
  res_x +=   K[mat+(N*2)] * v_x + K[mat+1+(N*2)] * v_y;\
  res_y += K[mat+8+(N*2)] * v_x + K[mat+9+(N*2)] * v_y;\
//------------------------------------------------------------------------------
__device__ void Aprod_elastic_2D_NodeByNode(unsigned int i,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, cudapcgVar_t *q){
  // Local vars to store entries of v to be used on computations
  cudapcgVar_t v_x, v_y;
  // Local vars to store result
  cudapcgVar_t res_x=0.0, res_y=0.0;
  // Local var to store index in local matrix
  unsigned int mat;
  // Get row and col ids
  int row = i%rows, col = i/rows;
  // Local var to store neighbor dof indexes
  unsigned int dof;
  // Go through neighbor elems
  #pragma unroll
  for (int e=0; e<4; e++){
      row += (e==2);
      col += (e==3)-(e==1);
      mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_2D))*64 + e*16;
      dof = 2*PERIODICNUM_2D(row,col,rows,cols);      // [row,col]
      APROD_ELASTIC_2D_NODE(0);
      dof = 2*PERIODICNUM_2D(row,col+1,rows,cols);    // [row,col] -> WALK_RIGHT
      APROD_ELASTIC_2D_NODE(1);
      dof = 2*PERIODICNUM_2D(row-1,col+1,rows,cols);  // [row,col] -> WALK_RIGHT, WALK_UP
      APROD_ELASTIC_2D_NODE(2);
      dof = 2*PERIODICNUM_2D(row-1,col,rows,cols);    // [row,col] -> WALK_UP
      APROD_ELASTIC_2D_NODE(3);
      matkey >>= MATKEY_BITSTEP_2D;
  }
  *q     = res_x;
  *(++q) = res_y;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_elastic_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t matkey, cudapcgVar_t *q){
  // Local var to store index in local matrix
  unsigned int mat;
  // Local var to store result
  cudapcgVar_t res_x=0.0, res_y=0.0, res_z=0.0;
  
  #pragma unroll
  for (int e=0; e<8; e++){
    mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*576+(e*75);
    res_x += K[mat]; res_y += K[mat+25]; res_z += K[mat+50];
    matkey >>= MATKEY_BITSTEP_3D;
  }

  *q     = 1.0/res_x;
  *(++q) = 1.0/res_y;
  *(++q) = 1.0/res_z;
}
//------------------------------------------------------------------------------
#define APROD_ELASTIC_3D_NODE(N)\
  v_x = v[dof]; v_y = v[dof+1]; v_z = v[dof+2];\
  res_x +=    K[mat+(N*3)] * v_x +  K[mat+1+(N*3)] * v_y +  K[mat+2+(N*3)] * v_z;\
  res_y += K[mat+24+(N*3)] * v_x + K[mat+25+(N*3)] * v_y + K[mat+26+(N*3)] * v_z;\
  res_z += K[mat+48+(N*3)] * v_x + K[mat+49+(N*3)] * v_y + K[mat+50+(N*3)] * v_z;\
//------------------------------------------------------------------------------
__device__ void Aprod_elastic_3D_NodeByNode(unsigned int i,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q){
  // Local vars to store entries of v to be used on computations
  cudapcgVar_t v_x, v_y, v_z;
  // Local vars to store result
  cudapcgVar_t res_x=0.0, res_y=0.0, res_z=0.0;
  // Local var to store index in local matrix
  unsigned int mat;
  // Get row and col ids
  int row = i%rows, col = (i%(rows*cols))/rows, layer = i/(rows*cols);
  // Local var to store neighbor dof indexes
  unsigned int dof;
  // Go through neighbor elems
  #pragma unroll
  for (int e=0; e<8; e++){
      row += (e==2)-(e==4)+(e==6);
      col += (e==3)-(e==1)+(e==7)-(e==5);
      layer -= (e==4);
      mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*576 + e*72;
      dof = 3*PERIODICNUM_3D(row,col,layer,rows,cols,layers);       // [row,col,layer]
      APROD_ELASTIC_3D_NODE(0);
      dof = 3*PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);     // [row,col,layer] -> WALK_RIGHT
      APROD_ELASTIC_3D_NODE(1);
      dof = 3*PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers);   // [row,col,layer] -> WALK_RIGHT, WALK_UP
      APROD_ELASTIC_3D_NODE(2);
      dof = 3*PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);     // [row,col,layer] -> WALK_UP
      APROD_ELASTIC_3D_NODE(3);
      dof = 3*PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);     // [row,col,layer] -> WALK_FAR
      APROD_ELASTIC_3D_NODE(4);
      dof = 3*PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers);   // [row,col,layer] -> WALK_FAR, WALK_RIGHT
      APROD_ELASTIC_3D_NODE(5);
      dof = 3*PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers); // [row,col,layer] -> WALK_FAR, WALK_RIGHT, WALK_UP
      APROD_ELASTIC_3D_NODE(6);
      dof = 3*PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers);   // [row,col,layer] -> WALK_FAR, WALK_UP
      APROD_ELASTIC_3D_NODE(7);
      matkey >>= MATKEY_BITSTEP_3D;
  }
  *q     = res_x;
  *(++q) = res_y;
  *(++q) = res_z;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_fluid_2D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *q){
  *q     = 1.0/(   K[0] +  K[26] +  K[52] +  K[78]);
  *(++q) = 1.0/(  K[13] +  K[39] +  K[65] +  K[91]);
  *(++q) = 1.0/(-K[104] - K[117] - K[130] - K[143]);
}
//------------------------------------------------------------------------------
#define APROD_FLUID_2D_PORE_NODE(N)\
  if (dof<nVelocityNodes){\
    dof<<=1;\
    v_x = v[dof]; v_y = v[dof+1];;\
    res_x +=    K[mat_v+(N*2)]*v_x +  K[mat_v+1+(N*2)]*v_y;\
    res_y += K[mat_v+12+(N*2)]*v_x + K[mat_v+13+(N*2)]*v_y;\
    res_p += K[mat_p+96+(N*2)]*v_x + K[mat_p+97+(N*2)]*v_y;\
  }\
  v_p = v[dof_p];\
  res_x +=   K[mat_v+8+N]*v_p;\
  res_y +=  K[mat_v+20+N]*v_p;\
  res_p -= K[mat_p+104+N]*v_p;\
//------------------------------------------------------------------------------
__device__ void Aprod_fluid_2D_NodeByNode_Pore(cudapcgIdMap_t id,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int nVelocityNodes, cudapcgIdMap_t *DOFMap, unsigned int rows, unsigned int cols, cudapcgVar_t *q){
  // Local vars to store entries of v to be used on computations
  cudapcgVar_t v_x, v_y, v_p;
  // Local vars to store result
  cudapcgVar_t res_x=0.0, res_y=0.0, res_p=0.0;
  // Get row and col ids
  int row = id%rows, col = id/rows;
  // Local var to store neighbor dof indexes
  unsigned int dof, dof_p;
  // Local var to store index in local matrix
  unsigned int mat_v, mat_p;
  // Go through neighbor elems
  #pragma unroll
  for (int e=0; e<4; e++){
      row += (e==2);
      col += (e==3)-(e==1);
      mat_v = e*24; mat_p = e*12;
      dof = DOFMap[PERIODICNUM_2D(row,col,rows,cols)];     dof_p = dof+nVelocityNodes*2;
      APROD_FLUID_2D_PORE_NODE(0);
      dof = DOFMap[PERIODICNUM_2D(row,col+1,rows,cols)];   dof_p = dof+nVelocityNodes*2;
      APROD_FLUID_2D_PORE_NODE(1);
      dof = DOFMap[PERIODICNUM_2D(row-1,col+1,rows,cols)]; dof_p = dof+nVelocityNodes*2;
      APROD_FLUID_2D_PORE_NODE(2);
      dof = DOFMap[PERIODICNUM_2D(row-1,col,rows,cols)];   dof_p = dof+nVelocityNodes*2;
      APROD_FLUID_2D_PORE_NODE(3);
  }
  *q     = res_x;
  *(++q) = res_y;
  *(++q) = res_p;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_fluid_2D_StencilByStencil_Pore(
    cudapcgVar_t *q){
  *q     = 0.25;
  *(++q) = 0.25;
  *(++q) = -2.25;
}
//------------------------------------------------------------------------------
static __constant__ char StencilSigns_Fluid2D_Corner[4][6]={
  {  1,  1,  1, -1, -1,  1 },
  { -1,  1, -1,  1, -1, -1 },
  { -1, -1, -1, -1,  1,  1 },
  {  1, -1,  1,  1,  1, -1 }
};
//------------------------------------------------------------------------------
__device__ void Aprod_fluid_2D_StencilByStencil_Pore(cudapcgIdMap_t id,
    cudapcgVar_t * v, unsigned int nVelocityNodes, cudapcgIdMap_t *DOFMap, unsigned int rows, unsigned int cols, cudapcgVar_t *q){
  // Local vars to store entries of v to be used on computations
  cudapcgVar_t v_x, v_y, v_p;
  // Local vars to store result
  cudapcgVar_t res_x=0.0, res_y=0.0, res_p=0.0;
  // Get row and col ids
  int row = id%rows, col = id/rows;
  // Local var to store neighbor dof indexes
  unsigned int dof, dof_p;

  // dof cache for stencil corners
  cudapcgIdMap_t corner_dofs[4] = {
    DOFMap[PERIODICNUM_2D(row-1,col-1,rows,cols)],
    DOFMap[PERIODICNUM_2D(row+1,col-1,rows,cols)],
    DOFMap[PERIODICNUM_2D(row-1,col+1,rows,cols)],
    DOFMap[PERIODICNUM_2D(row+1,col+1,rows,cols)]
  };

  // center
  dof = DOFMap[id]; dof_p = dof+nVelocityNodes*2; dof<<=1;
  res_x += 4.0*v[dof];
  res_y += 4.0*v[dof+1];
  res_p += -(4.0/9.0)*v[dof_p];

  // corners
  #pragma unroll
  for (int j=0; j<4; j++){
    dof = corner_dofs[j]; dof_p = dof+nVelocityNodes*2;
    if (dof<nVelocityNodes){
      dof<<=1;
      v_x = v[dof]; v_y = v[dof+1];
      res_x += -0.5*v_x + 0.25*StencilSigns_Fluid2D_Corner[j][0]*v_y;
      res_y += -0.5*v_y + 0.25*StencilSigns_Fluid2D_Corner[j][2]*v_x;
      res_p += (1.0/12.0)*(StencilSigns_Fluid2D_Corner[j][4]*v_x + StencilSigns_Fluid2D_Corner[j][5]*v_y);
    }
    v_p = v[dof_p];
    res_x += (1.0/12.0)*StencilSigns_Fluid2D_Corner[j][1]*v_p;
    res_y += (1.0/12.0)*StencilSigns_Fluid2D_Corner[j][3]*v_p;
    res_p += (1.0/18.0)*v_p;
  }

  // imediate neighbors (left,right,up,down)
  #pragma unroll
  for (int j=-1; j<=1; j+=2){
    dof = DOFMap[PERIODICNUM_2D(row,col+j,rows,cols)]; dof_p = dof+nVelocityNodes*2;
    if (dof<nVelocityNodes){
      dof<<=1;
      v_x = v[dof];
      res_x += -v_x;
      res_p += (j*1.0/3.0)*v_x;
    }
    v_p = v[dof_p];
    res_x += (-j*1.0/3.0)*v_p;
    res_p += (1.0/18.0)*v_p;
  }
  #pragma unroll
  for (int j=-1; j<=1; j+=2){
    dof = DOFMap[PERIODICNUM_2D(row+j,col,rows,cols)]; dof_p = dof+nVelocityNodes*2;
    if (dof<nVelocityNodes){
      dof<<=1;
      v_y = v[dof+1];
      res_y += -v_y;
      res_p += (-j*1.0/3.0)*v_y;
    }
    v_p = v[dof_p];
    res_y += (j*1.0/3.0)*v_p;
    res_p += (1.0/18.0)*v_p;
  }
  
  *q     = res_x;
  *(++q) = res_y;
  *(++q) = res_p;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_fluid_2D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgFlag_t fluidkey, cudapcgVar_t *q){
  *q = 1.0/( -K[104]*(fluidkey&1)
             -K[117]*((fluidkey>>1)&1)
             -K[130]*((fluidkey>>2)&1)
             -K[143]*((fluidkey>>3)&1) );
}
//------------------------------------------------------------------------------
#define APROD_FLUID_2D_BORDER_NODE(N)\
  if (dof<nVelocityNodes){\
    dof<<=1;\
    v_x = v[dof]; v_y = v[dof+1];\
    res_p += K[mat+96+(N*2)]*v_x + K[mat+97+(N*2)]*v_y;\
  }\
  v_p = v[dof_p];\
  res_p -= K[mat+104+N]*v_p;\
//------------------------------------------------------------------------------
__device__ void Aprod_fluid_2D_NodeByNode_Border(cudapcgIdMap_t id,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int nVelocityNodes, cudapcgFlag_t fluidkey, cudapcgIdMap_t *DOFMap, unsigned int rows, unsigned int cols, cudapcgVar_t *q){
  // Local vars to store entries of v to be used on computations
  cudapcgVar_t v_x, v_y, v_p;
  // Local var to store result
  cudapcgVar_t res_p=0.0;
  // Get row and col ids
  int row = id%rows, col = id/rows;
  // Local var to store neighbor dof indexes
  unsigned int dof, dof_p;
  // Local var to store index in local matrix
  unsigned int mat;
  // Go through neighbor elems
  #pragma unroll
  for (int e=0; e<4; e++){
      row += (e==2);
      col += (e==3)-(e==1);
      if ((fluidkey>>e)&1){
        mat = e*12;
        dof = DOFMap[PERIODICNUM_2D(row,col,rows,cols)];     dof_p = dof+nVelocityNodes*2;
        APROD_FLUID_2D_BORDER_NODE(0);
        dof = DOFMap[PERIODICNUM_2D(row,col+1,rows,cols)];   dof_p = dof+nVelocityNodes*2;
        APROD_FLUID_2D_BORDER_NODE(1);
        dof = DOFMap[PERIODICNUM_2D(row-1,col+1,rows,cols)]; dof_p = dof+nVelocityNodes*2;
        APROD_FLUID_2D_BORDER_NODE(2);
        dof = DOFMap[PERIODICNUM_2D(row-1,col,rows,cols)];   dof_p = dof+nVelocityNodes*2;
        APROD_FLUID_2D_BORDER_NODE(3);
      }
  }
  *q = res_p;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_fluid_3D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *q){
  *q     = 1.0/(   K[0] +  K[99] + K[198] + K[297] + K[396] + K[495] + K[594] +  K[693]);
  *(++q) = 1.0/(  K[33] + K[132] + K[231] + K[330] + K[429] + K[528] + K[627] +  K[726]);
  *(++q) = 1.0/(  K[66] + K[165] + K[264] + K[363] + K[462] + K[561] + K[660] +  K[759]);
  *(++q) = 1.0/(-K[792] - K[825] - K[858] - K[891] - K[924] - K[957] - K[990] - K[1023]);
}
//------------------------------------------------------------------------------
#define APROD_FLUID_3D_PORE_NODE(N)\
  if (dof<nVelocityNodes){\
    dof*=3;\
    v_x = v[dof]; v_y = v[dof+1]; v_z = v[dof+2];\
    res_x +=     K[mat_v+(N*3)]*v_x +   K[mat_v+1+(N*3)]*v_y +   K[mat_v+2+(N*3)]*v_z;\
    res_y +=  K[mat_v+32+(N*3)]*v_x +  K[mat_v+33+(N*3)]*v_y +  K[mat_v+34+(N*3)]*v_z;\
    res_z +=  K[mat_v+64+(N*3)]*v_x +  K[mat_v+65+(N*3)]*v_y +  K[mat_v+66+(N*3)]*v_z;\
    res_p += K[mat_p+768+(N*3)]*v_x + K[mat_p+769+(N*3)]*v_y + K[mat_p+770+(N*3)]*v_z;\
  }\
  v_p = v[dof_p];\
  res_x +=  K[mat_v+24+N]*v_p;\
  res_y +=  K[mat_v+56+N]*v_p;\
  res_z +=  K[mat_v+88+N]*v_p;\
  res_p -= K[mat_p+792+N]*v_p;\
//------------------------------------------------------------------------------ 
__device__ void Aprod_fluid_3D_NodeByNode_Pore(cudapcgIdMap_t id,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int nVelocityNodes, cudapcgIdMap_t *DOFMap, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t * q){
  // Local vars to store entries of v to be used on computations
  cudapcgVar_t v_x, v_y, v_z, v_p;
  // Local vars to store result
  cudapcgVar_t res_x=0.0, res_y=0.0, res_z=0.0, res_p=0.0;
  // Get row and col ids
  int row = id%rows, col = (id%(rows*cols))/rows, layer = id/(rows*cols);
  // Local var to store neighbor dof indexes
  unsigned int dof, dof_p;
  // Local var to store indexes in local matrix
  int mat_v, mat_p;
  // Go through neighbor elems
  //#pragma unroll // TODO: explicit unroll directive seems to not be working for this kernel.
  for (int e=0; e<8; e++){
      row += (e==2)-(e==4)+(e==6);
      col += (e==3)-(e==1)+(e==7)-(e==5);
      layer -= (e==4);
      mat_v = e*96; mat_p = e*32;
      dof = DOFMap[PERIODICNUM_3D(row,col,layer,rows,cols,layers)];       dof_p = dof+nVelocityNodes*3;
      APROD_FLUID_3D_PORE_NODE(0);
      dof = DOFMap[PERIODICNUM_3D(row,col+1,layer,rows,cols,layers)];     dof_p = dof+nVelocityNodes*3;
      APROD_FLUID_3D_PORE_NODE(1);
      dof = DOFMap[PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers)];   dof_p = dof+nVelocityNodes*3;
      APROD_FLUID_3D_PORE_NODE(2);
      dof = DOFMap[PERIODICNUM_3D(row-1,col,layer,rows,cols,layers)];     dof_p = dof+nVelocityNodes*3;
      APROD_FLUID_3D_PORE_NODE(3);
      dof = DOFMap[PERIODICNUM_3D(row,col,layer+1,rows,cols,layers)];     dof_p = dof+nVelocityNodes*3;
      APROD_FLUID_3D_PORE_NODE(4);
      dof = DOFMap[PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers)];   dof_p = dof+nVelocityNodes*3;
      APROD_FLUID_3D_PORE_NODE(5);
      dof = DOFMap[PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers)]; dof_p = dof+nVelocityNodes*3;
      APROD_FLUID_3D_PORE_NODE(6);
      dof = DOFMap[PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers)];   dof_p = dof+nVelocityNodes*3;
      APROD_FLUID_3D_PORE_NODE(7);
  }
  *q     = res_x;
  *(++q) = res_y;
  *(++q) = res_z;
  *(++q) = res_p;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_fluid_3D_StencilByStencil_Pore(
    cudapcgVar_t *q){
  *q     = 0.28125;
  *(++q) = 0.28125;
  *(++q) = 0.28125;
  *(++q) = -2.25;
}
//------------------------------------------------------------------------------
static __constant__ char StencilSigns_Fluid3D_Corner[8][12]={
  { 1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1, -1},
  {-1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1, -1},
  {-1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1, -1},
  { 1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1},
  { 1,  1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1},
  {-1,  1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1},
  {-1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1},
  { 1, -1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1}
};
//------------------------------------------------------------------------------ 
__device__ void Aprod_fluid_3D_StencilByStencil_Pore(cudapcgIdMap_t id,
    cudapcgVar_t * v, unsigned int nVelocityNodes, cudapcgIdMap_t *DOFMap, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t * q){
  // Local vars to store entries of v to be used on computations
  cudapcgVar_t v_x, v_y, v_z, v_p;
  // Local vars to store result
  cudapcgVar_t res_x=0.0, res_y=0.0, res_z=0.0, res_p=0.0;
  // Get row and col ids
  int row = id%rows, col = (id%(rows*cols))/rows, layer = id/(rows*cols);
  // Local var to store neighbor dof indexes
  unsigned int dof, dof_p;
  
  // dof cache for stencil corners
  cudapcgIdMap_t corner_dofs[8] = {
    DOFMap[PERIODICNUM_3D(row-1,col-1,layer-1,rows,cols,layers)],
    DOFMap[PERIODICNUM_3D(row+1,col-1,layer-1,rows,cols,layers)],
    DOFMap[PERIODICNUM_3D(row-1,col+1,layer-1,rows,cols,layers)],
    DOFMap[PERIODICNUM_3D(row+1,col+1,layer-1,rows,cols,layers)],
    DOFMap[PERIODICNUM_3D(row-1,col-1,layer+1,rows,cols,layers)],
    DOFMap[PERIODICNUM_3D(row+1,col-1,layer+1,rows,cols,layers)],
    DOFMap[PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers)],
    DOFMap[PERIODICNUM_3D(row+1,col+1,layer+1,rows,cols,layers)]
  };

  // center
  dof = DOFMap[id]; dof_p = dof+nVelocityNodes*3; dof*=3;
  res_x += (32.0/9.0)*v[dof];
  res_y += (32.0/9.0)*v[dof+1];
  res_z += (32.0/9.0)*v[dof+2];
  res_p += -(4.0/9.0)*v[dof_p];

  // corners
  #pragma unroll
  for (int j=0; j<8; j++){
    dof = corner_dofs[j]; dof_p = dof+nVelocityNodes*3;
    if (dof<nVelocityNodes){
      dof*=3;
      v_x = v[dof]; v_y = v[dof+1]; v_z = v[dof+2];
      res_x += -(1.0/9.0)*v_x + (1.0/24.0)*(StencilSigns_Fluid3D_Corner[j][0]*v_y + StencilSigns_Fluid3D_Corner[j][1]*v_z);
      res_y += -(1.0/9.0)*v_y + (1.0/24.0)*(StencilSigns_Fluid3D_Corner[j][3]*v_x + StencilSigns_Fluid3D_Corner[j][4]*v_z);
      res_z += -(1.0/9.0)*v_z + (1.0/24.0)*(StencilSigns_Fluid3D_Corner[j][6]*v_x + StencilSigns_Fluid3D_Corner[j][7]*v_y);
      res_p += (1.0/72.0)*(StencilSigns_Fluid3D_Corner[j][9]*v_x + StencilSigns_Fluid3D_Corner[j][10]*v_y + StencilSigns_Fluid3D_Corner[j][11]*v_z);
    }
    v_p = v[dof_p];
    res_x += (1.0/72.0)*StencilSigns_Fluid3D_Corner[j][2]*v_p;
    res_y += (1.0/72.0)*StencilSigns_Fluid3D_Corner[j][5]*v_p;
    res_z += (1.0/72.0)*StencilSigns_Fluid3D_Corner[j][8]*v_p;
    res_p += (1.0/72.0)*v_p;
  }

  // imediate neighbors (near,far,left,right,up,down)
  #pragma unroll
  for (int j=-1; j<=1; j+=2){
    dof = DOFMap[PERIODICNUM_3D(row,col,layer+j,rows,cols,layers)]; dof_p = dof+nVelocityNodes*3;
    if (dof<nVelocityNodes){
      dof*=3;
      v_x = v[dof]; v_y = v[dof+1]; v_z = v[dof+2];
      res_x +=  (2.0/9.0)*v_x;
      res_y +=  (2.0/9.0)*v_y;
      res_z += -(4.0/9.0)*v_z;
      res_p += (j*2.0/9.0)*v_z;
    }
    v_p = v[dof_p];
    res_z += (-j*2.0/9.0)*v_p;
  }
  #pragma unroll
  for (int j=-1; j<=1; j+=2){
    dof = DOFMap[PERIODICNUM_3D(row,col+j,layer,rows,cols,layers)]; dof_p = dof+nVelocityNodes*3;
    if (dof<nVelocityNodes){
      dof*=3;
      v_x = v[dof]; v_y = v[dof+1]; v_z = v[dof+2];
      res_x += -(4.0/9.0)*v_x;
      res_y +=  (2.0/9.0)*v_y;
      res_z +=  (2.0/9.0)*v_z;
      res_p += (j*2.0/9.0)*v_x;
    }
    v_p = v[dof_p];
    res_x += (-j*2.0/9.0)*v_p;
  }
  #pragma unroll
  for (int j=-1; j<=1; j+=2){
    dof = DOFMap[PERIODICNUM_3D(row+j,col,layer,rows,cols,layers)]; dof_p = dof+nVelocityNodes*3;
    if (dof<nVelocityNodes){
      dof*=3;
      v_x = v[dof]; v_y = v[dof+1]; v_z = v[dof+2];
      res_x +=  (2.0/9.0)*v_x;
      res_y += -(4.0/9.0)*v_y;
      res_z +=  (2.0/9.0)*v_z;
      res_p += (-j*2.0/9.0)*v_y;
    }
    v_p = v[dof_p];
    res_y += (j*2.0/9.0)*v_p;
  }

  // x=0
  for (int k=-1; k<=1; k+=2){
    #pragma unroll
    for (int j=-1; j<=1; j+=2){
      dof = DOFMap[PERIODICNUM_3D(row+j,col,layer+k,rows,cols,layers)]; dof_p = dof+nVelocityNodes*3;
      if (dof<nVelocityNodes){
        dof*=3;
        v_x = v[dof]; v_y = v[dof+1]; v_z = v[dof+2];
        res_x += -(1.0/9.0)*v_x;
        res_y += -(1.0/3.6)*v_y + (k*j)*(1.0/6.0)*v_z;
        res_z += -(1.0/3.6)*v_z + (k*j)*(1.0/6.0)*v_y;
        res_p += (-j*1.0/18.0)*v_y + (k*1.0/18.0)*v_z;
      }
      v_p = v[dof_p];
      res_y +=  (j*1.0/18.0)*v_p;
      res_z += (-k*1.0/18.0)*v_p;
      res_p +=    (1.0/36.0)*v_p;
    }
  }

  // y=0
  for (int k=-1; k<=1; k+=2){
    #pragma unroll
    for (int j=-1; j<=1; j+=2){
      dof = DOFMap[PERIODICNUM_3D(row,col+j,layer+k,rows,cols,layers)]; dof_p = dof+nVelocityNodes*3;
      if (dof<nVelocityNodes){
        dof*=3;
        v_x = v[dof]; v_y = v[dof+1]; v_z = v[dof+2];
        res_x += -(1.0/3.6)*v_x + (-k*j)*(1.0/6.0)*v_z;
        res_y += -(1.0/9.0)*v_y;
        res_z += -(1.0/3.6)*v_z + (-k*j)*(1.0/6.0)*v_x;
        res_p += (j*1.0/18.0)*v_x + (k*1.0/18.0)*v_z;
      }
      v_p = v[dof_p];
      res_x += (-j*1.0/18.0)*v_p;
      res_z += (-k*1.0/18.0)*v_p;
      res_p +=    (1.0/36.0)*v_p;
    }
  }

  // z=0
  for (int k=-1; k<=1; k+=2){
    #pragma unroll
    for (int j=-1; j<=1; j+=2){
      dof = DOFMap[PERIODICNUM_3D(row+j,col+k,layer,rows,cols,layers)]; dof_p = dof+nVelocityNodes*3;
      if (dof<nVelocityNodes){
        dof*=3;
        v_x = v[dof]; v_y = v[dof+1]; v_z = v[dof+2];
        res_x += -(1.0/3.6)*v_x + (k*j)*(1.0/6.0)*v_y;
        res_y += -(1.0/3.6)*v_y + (k*j)*(1.0/6.0)*v_x;
        res_z += -(1.0/9.0)*v_z;
        res_p += (k*1.0/18.0)*v_x + (-j*1.0/18.0)*v_y;
      }
      v_p = v[dof_p];
      res_x += (-k*1.0/18.0)*v_p;
      res_y +=  (j*1.0/18.0)*v_p;
      res_p +=    (1.0/36.0)*v_p;
    }
  }

  *q     = res_x;
  *(++q) = res_y;
  *(++q) = res_z;
  *(++q) = res_p;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_fluid_3D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgFlag_t fluidkey, cudapcgVar_t *q){
  *q = 1.0/( -K[792]*(fluidkey&1)
             -K[825]*((fluidkey>>1)&1)
             -K[858]*((fluidkey>>2)&1)
             -K[891]*((fluidkey>>3)&1)
             -K[924]*((fluidkey>>4)&1)
             -K[957]*((fluidkey>>5)&1)
             -K[990]*((fluidkey>>6)&1)
            -K[1023]*((fluidkey>>7)&1) );
}
//------------------------------------------------------------------------------
#define APROD_FLUID_3D_BORDER_NODE(N)\
  if (dof<nVelocityNodes){\
    dof*=3;\
    v_x = v[dof]; v_y = v[dof+1]; v_z = v[dof+2];\
    res_p += K[mat+768+(N*3)]*v_x + K[mat+769+(N*3)]*v_y + K[mat+770+(N*3)]*v_z;\
  }\
  v_p = v[dof_p];\
  res_p -= K[mat+792+N]*v_p;\
//------------------------------------------------------------------------------
__device__ void Aprod_fluid_3D_NodeByNode_Border(cudapcgIdMap_t id,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int nVelocityNodes, cudapcgFlag_t fluidkey, cudapcgIdMap_t *DOFMap, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t * q){
  // Local vars to store entries of v to be used on computations
  cudapcgVar_t v_x, v_y, v_z, v_p;
  // Local vars to store result
  cudapcgVar_t res_p=0.0;
  int row = id%rows, col = (id%(rows*cols))/rows, layer = id/(rows*cols);
  // Local var to store neighbor dof indexes
  unsigned int dof, dof_p;
  // Local var to store indexes in local matrix
  unsigned int mat;
  // Go through neighbor elems
  //#pragma unroll // TODO: explicit unroll directive seems to not be working for this kernel.
  for (int e=0; e<8; e++){
      row += (e==2)-(e==4)+(e==6);
      col += (e==3)-(e==1)+(e==7)-(e==5);
      layer -= (e==4);
      if ((fluidkey>>e)&1){
        mat = e*32;
        dof = DOFMap[PERIODICNUM_3D(row,col,layer,rows,cols,layers)];       dof_p = dof+nVelocityNodes*3;
        APROD_FLUID_3D_BORDER_NODE(0);
        dof = DOFMap[PERIODICNUM_3D(row,col+1,layer,rows,cols,layers)];     dof_p = dof+nVelocityNodes*3;
        APROD_FLUID_3D_BORDER_NODE(1);
        dof = DOFMap[PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers)];   dof_p = dof+nVelocityNodes*3;
        APROD_FLUID_3D_BORDER_NODE(2);
        dof = DOFMap[PERIODICNUM_3D(row-1,col,layer,rows,cols,layers)];     dof_p = dof+nVelocityNodes*3;
        APROD_FLUID_3D_BORDER_NODE(3);
        dof = DOFMap[PERIODICNUM_3D(row,col,layer+1,rows,cols,layers)];     dof_p = dof+nVelocityNodes*3;
        APROD_FLUID_3D_BORDER_NODE(4);
        dof = DOFMap[PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers)];   dof_p = dof+nVelocityNodes*3;
        APROD_FLUID_3D_BORDER_NODE(5);
        dof = DOFMap[PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers)]; dof_p = dof+nVelocityNodes*3;
        APROD_FLUID_3D_BORDER_NODE(6);
        dof = DOFMap[PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers)];   dof_p = dof+nVelocityNodes*3;
        APROD_FLUID_3D_BORDER_NODE(7);
      }
  }
  *q = res_p;
}
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
__device__ void Preconditioner_thermal_2D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t *materials, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res=0.0;

  #pragma unroll
  for (int e=0; e<4; e++){
    res += K[materials[e]*16+(e*5)];
  }

  *q = 1.0/res;
}
//------------------------------------------------------------------------------
__device__ void Aprod_thermal_2D_ElemByElem(unsigned int i, unsigned int n,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res=0.0;
  // Local var to store index in local matrix
  unsigned int mat;
  // Get row and col ids
  int row = (i+1)%rows, col = i/rows;
  // Local var to store neighbor dof indexes
  unsigned int dof;

  mat = matkey*16 + n*4;
  dof = PERIODICNUM_2D(row,col,rows,cols);
  res += K[mat] * v[dof];
  dof = PERIODICNUM_2D(row,col+1,rows,cols);
  res += K[mat+1] * v[dof];
  dof = PERIODICNUM_2D(row-1,col+1,rows,cols);
  res += K[mat+2] * v[dof];
  dof = PERIODICNUM_2D(row-1,col,rows,cols);
  res += K[mat+3] * v[dof];

  *q = res;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_thermal_3D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t *materials, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res=0.0;

  #pragma unroll
  for (int e=0; e<8; e++){
    res += K[materials[e]*64+(e*9)];
  }

  *q = 1.0/res;
}
//------------------------------------------------------------------------------
__device__ void Aprod_thermal_3D_ElemByElem(unsigned int i, unsigned int n,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q){
  // Local vars to store result
  cudapcgVar_t res=0.0;
  // Local var to store index in local matrix
  unsigned int mat;
  // Get row and col ids
  int row = (i+1)%rows, col = (i%(rows*cols))/rows, layer = i/(rows*cols);
  // Local var to store neighbor dof indexes
  unsigned int dof;
  
  mat = matkey*64 + n*8;
  dof = PERIODICNUM_3D(row,col,layer,rows,cols,layers);
  res +=   K[mat]*v[dof];
  dof = PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);
  res += K[mat+1]*v[dof];
  dof = PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers);
  res += K[mat+2]*v[dof];
  dof = PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);
  res += K[mat+3]*v[dof];
  dof = PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);
  res += K[mat+4]*v[dof];
  dof = PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers);
  res += K[mat+5]*v[dof];
  dof = PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers);
  res += K[mat+6]*v[dof];
  dof = PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers);
  res += K[mat+7]*v[dof];

  *q = res;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_thermal_3D_ElemByElem_SparseQ1(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t *materials, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res=0.0;

  #pragma unroll
  for (int e=0; e<8; e++){
     res += K[materials[e]*40]; // using always first diagonal term
  }

  *q = 1.0/res;
}
//------------------------------------------------------------------------------
template <int N>
__device__ void Aprod_thermal_3D_ElemByElem_SparseQ1(unsigned int i,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q){
  // Local vars to store result
  cudapcgVar_t res=0.0;
  // Local var to store index in local matrix
  unsigned int mat = matkey*40;
  // Get row and col ids
  int row = (i+1)%rows, col = (i%(rows*cols))/rows, layer = i/(rows*cols);
  // Local var to store neighbor dof indexes
  unsigned int dof;
  
  if (N==0){
    dof = PERIODICNUM_3D(row,col,layer,rows,cols,layers);;
    res +=   K[mat] * v[dof];
    dof = PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers);
    res += K[mat+1] * v[dof];
    dof = PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers);
    res += K[mat+2] * v[dof];
    dof = PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers);
    res += K[mat+3] * v[dof];
    dof = PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers);
    res += K[mat+4] * v[dof];
  } else if (N==1){
    dof = PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);
    res += K[mat+5] * v[dof];
    dof = PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);
    res += K[mat+6] * v[dof];
    dof = PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);
    res += K[mat+7] * v[dof];
    dof = PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers);
    res += K[mat+8] * v[dof];
    dof = PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers);
    res += K[mat+9] * v[dof];
  } else if (N==2){
    dof = PERIODICNUM_3D(row,col,layer,rows,cols,layers);
    res += K[mat+10] * v[dof];
    dof = PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers);
    res += K[mat+11] * v[dof];
    dof = PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);
    res += K[mat+12] * v[dof];
    dof = PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers);
    res += K[mat+13] * v[dof];
    dof = PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers);
    res += K[mat+14] * v[dof];
  } else if (N==3){
    dof = PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);
    res += K[mat+15] * v[dof];
    dof = PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);
    res += K[mat+16] * v[dof];
    dof = PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);
    res += K[mat+17] * v[dof];
    dof = PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers);
    res += K[mat+18] * v[dof];
    dof = PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers);
    res += K[mat+19] * v[dof];
  } else if (N==4){
    dof = PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);
    res += K[mat+20] * v[dof];
    dof = PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers);
    res += K[mat+21] * v[dof];
    dof = PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);
    res += K[mat+22] * v[dof];
    dof = PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);
    res += K[mat+23] * v[dof];
    dof = PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers);
    res += K[mat+24] * v[dof];
  } else if (N==5){
    dof = PERIODICNUM_3D(row,col,layer,rows,cols,layers);
    res += K[mat+25] * v[dof];
    dof = PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers);
    res += K[mat+26] * v[dof];
    dof = PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);
    res += K[mat+27] * v[dof];
    dof = PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers);
    res += K[mat+28] * v[dof];
    dof = PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers);
    res += K[mat+29] * v[dof];
  } else if (N==6){
    dof = PERIODICNUM_3D(row,col,layer,rows,cols,layers);
    res += K[mat+30] * v[dof];
    dof = PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);
    res += K[mat+31] * v[dof];
    dof = PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);
    res += K[mat+32] * v[dof];
    dof = PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);
    res += K[mat+33] * v[dof];
    dof = PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers);
    res += K[mat+34] * v[dof];
  } else if (N==7){
    dof = PERIODICNUM_3D(row,col,layer,rows,cols,layers);
    res += K[mat+35] * v[dof];
    dof = PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);
    res += K[mat+36] * v[dof];
    dof = PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers);
    res += K[mat+37] * v[dof];
    dof = PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers);
    res += K[mat+38] * v[dof];
    dof = PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers);
    res += K[mat+39] * v[dof];
  }

  *q = res;
}
#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<0>(unsigned int i, cudapcgVar_t * K, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<1>(unsigned int i, cudapcgVar_t * K, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<2>(unsigned int i, cudapcgVar_t * K, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<3>(unsigned int i, cudapcgVar_t * K, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<4>(unsigned int i, cudapcgVar_t * K, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<5>(unsigned int i, cudapcgVar_t * K, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<6>(unsigned int i, cudapcgVar_t * K, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<7>(unsigned int i, cudapcgVar_t * K, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
#else
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<0>(unsigned int i, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<1>(unsigned int i, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<2>(unsigned int i, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<3>(unsigned int i, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<4>(unsigned int i, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<5>(unsigned int i, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<6>(unsigned int i, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
template __device__ void Aprod_thermal_3D_ElemByElem_SparseQ1<7>(unsigned int i, cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q);
#endif
//------------------------------------------------------------------------------
__device__ void Preconditioner_elastic_2D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t *materials, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res_x=0.0, res_y=0.0;
  // Local var to store indexes in local matrix
  unsigned int mat;

  #pragma unroll
  for (int e=0; e<4; e++){
    mat = materials[e]*64+(e*18);
    res_x += K[mat];
    res_y += K[mat+9];
  }

  *q     = 1.0/res_x;
  *(++q) = 1.0/res_y;
}
//------------------------------------------------------------------------------
__device__ void Aprod_elastic_2D_ElemByElem(unsigned int i, unsigned int n,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, cudapcgVar_t *q){
  // Local vars to store entries of v to be used on computations
  cudapcgVar_t v_x, v_y;
  // Local var to store result
  cudapcgVar_t res_x=0.0, res_y=0.0;
  // Local var to store index in local matrix
  unsigned int mat;
  // Get row and col ids
  int row = (i+1)%rows, col = i/rows;
  // Local var to store neighbor dof indexes
  unsigned int dof;

  mat = matkey*64 + n*16;
  dof = 2*PERIODICNUM_2D(row,col,rows,cols);
  APROD_ELASTIC_2D_NODE(0);
  dof = 2*PERIODICNUM_2D(row,col+1,rows,cols);
  APROD_ELASTIC_2D_NODE(1);
  dof = 2*PERIODICNUM_2D(row-1,col+1,rows,cols);
  APROD_ELASTIC_2D_NODE(2);
  dof = 2*PERIODICNUM_2D(row-1,col,rows,cols);
  APROD_ELASTIC_2D_NODE(3);

  *q     = res_x;
  *(++q) = res_y;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_elastic_3D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t *materials, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res_x=0.0, res_y=0.0, res_z=0.0;
  // Local var to store indexes in local matrix
  unsigned int mat;

  #pragma unroll
  for (int e=0; e<8; e++){
    mat = ((unsigned int)materials[e])*576+(e*75);
    res_x += K[mat];
    res_y += K[mat+25];
    res_z += K[mat+50];
  }

  *q     = 1.0/res_x;
  *(++q) = 1.0/res_y;
  *(++q) = 1.0/res_z;
}
//------------------------------------------------------------------------------
__device__ void Aprod_elastic_3D_ElemByElem(unsigned int i, unsigned int n,
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, cudapcgMap_t matkey, unsigned int rows, unsigned int cols, unsigned int layers, cudapcgVar_t *q){
  // Local vars to store entries of v to be used on computations
  cudapcgVar_t v_x, v_y, v_z;
  // Local vars to store result
  cudapcgVar_t res_x=0.0, res_y=0.0, res_z=0.0;
  // Local var to store index in local matrix
  unsigned int mat;
  // Get row and col ids
  int row = (i+1)%rows, col = (i%(rows*cols))/rows, layer = i/(rows*cols);
  // Local var to store neighbor dof indexes
  unsigned int dof;
  
  mat = ((unsigned int)matkey)*576 + n*72;
  dof = 3*PERIODICNUM_3D(row,col,layer,rows,cols,layers);
  APROD_ELASTIC_3D_NODE(0);
  dof = 3*PERIODICNUM_3D(row,col+1,layer,rows,cols,layers);
  APROD_ELASTIC_3D_NODE(1);
  dof = 3*PERIODICNUM_3D(row-1,col+1,layer,rows,cols,layers);
  APROD_ELASTIC_3D_NODE(2);
  dof = 3*PERIODICNUM_3D(row-1,col,layer,rows,cols,layers);
  APROD_ELASTIC_3D_NODE(3);
  dof = 3*PERIODICNUM_3D(row,col,layer+1,rows,cols,layers);
  APROD_ELASTIC_3D_NODE(4);
  dof = 3*PERIODICNUM_3D(row,col+1,layer+1,rows,cols,layers);
  APROD_ELASTIC_3D_NODE(5);
  dof = 3*PERIODICNUM_3D(row-1,col+1,layer+1,rows,cols,layers);
  APROD_ELASTIC_3D_NODE(6);
  dof = 3*PERIODICNUM_3D(row-1,col,layer+1,rows,cols,layers);
  APROD_ELASTIC_3D_NODE(7);

  *q     = res_x;
  *(++q) = res_y;
  *(++q) = res_z;
}
//------------------------------------------------------------------------------

/*
  ATTENTION: NO EBE FOR FLUIDS
*/

//---------------------------------
///////////////////////////////////
////// SCALAR DENSITY FIELDS //////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
__device__ void Preconditioner_thermal_2D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *conductivities, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res=0.0;

  #pragma unroll
  for (int e=0; e<4; e++){
    res += conductivities[e]*K[e*5];
  }

  *q = 1.0/res;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_thermal_3D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *conductivities, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res=0.0;

  #pragma unroll
  for (int e=0; e<8; e++){
    res += conductivities[e]*K[e*9];
  }

  *q = 1.0/res;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_thermal_3D_ElemByElem_ScalarDensityField_SparseQ1(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *conductivities, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res=0.0;

  #pragma unroll
  for (int e=0; e<8; e++){
     res += conductivities[e]*K[0]; // using always first diagonal term
  }

  *q = 1.0/res;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_elastic_2D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t *materials, cudapcgVar_t *elasticmoduli, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res_x=0.0, res_y=0.0;
  // Local var to store indexes in local matrix
  unsigned int mat;

  #pragma unroll
  for (int e=0; e<4; e++){
    mat = materials[e]*64+(e*18);
    res_x += elasticmoduli[e]*K[mat];
    res_y += elasticmoduli[e]*K[mat+9];
  }

  *q     = 1.0/res_x;
  *(++q) = 1.0/res_y;
}
//------------------------------------------------------------------------------
__device__ void Preconditioner_elastic_3D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgMap_t *materials, cudapcgVar_t *elasticmoduli, cudapcgVar_t *q){
  // Local var to store result
  cudapcgVar_t res_x=0.0, res_y=0.0, res_z=0.0;
  // Local var to store indexes in local matrix
  unsigned int mat;

  #pragma unroll
  for (int e=0; e<8; e++){
    mat = ((unsigned int)materials[e])*576+(e*75);
    res_x += elasticmoduli[e]*K[mat];
    res_y += elasticmoduli[e]*K[mat+25];
    res_z += elasticmoduli[e]*K[mat+50];
  }

  *q     = 1.0/res_x;
  *(++q) = 1.0/res_y;
  *(++q) = 1.0/res_z;
}
//------------------------------------------------------------------------------

/*
  ATTENTION: NO SCALAR DENSITY FIELD FOR FLUIDS
*/

//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
///////////////////////// CUDA KERNELS ///////////////////////////////
//////////////////////////////////////////////////////////////////////
//--------------------------------------------------------------------

//-----------------------------------------
///////////////////////////////////////////
/////////////// THERMAL 2D ////////////////
///////////////////////////////////////////
//-----------------------------------------

//---------------------------------
///////////////////////////////////
////////// NODE-BY-NODE ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 2D thermal analysis
__global__ void kernel_assemblePreConditioner_thermal_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t q;
    Preconditioner_thermal_2D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      material[i], &q);
    M[i] = q;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D thermal conductivity analysis
__global__ void kernel_applyPreConditioner_thermal_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t q;
    Preconditioner_thermal_2D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      material[i], &q);
    res[i] = scl1*v1[i]*q + (v2 == NULL ? 0.0 : scl2*v2[i]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D thermal analysis
__global__ void kernel_Aprod_thermal_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    // Local var to store result
    cudapcgVar_t res=0.0;

    Aprod_thermal_2D_NodeByNode( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[i], ny, nx, &res);

    // Put final result on global array
    q[i] = scl*res + q[i]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D thermal analysis
__global__ void kernel_PreConditionerAprod_thermal_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgMap_t matkey = material[i];

    cudapcgVar_t res;
    Aprod_thermal_2D_NodeByNode( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res);

    cudapcgVar_t p;
    Preconditioner_thermal_2D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      matkey, &p);

    // Put final result on global array
    res*=p;
    q[i] = scl*res + q[i]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_thermal_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q=0.0, vi= v1[gi];
    vi*=(v2 == NULL ? vi : v2[gi]);
    Preconditioner_thermal_2D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      material[gi], &q);
      cache[li] = (double) (vi*q);
  } else
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_thermal_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q=0.0;
    Aprod_thermal_2D_NodeByNode( gi,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[gi], ny, nx, &q);
      cache[li] = (double) (v[gi]*q);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_thermal_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q=0.0;
    Aprod_thermal_2D_NodeByNode( gi,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[gi], ny, nx, &q);
      cache[li] = (double) (q*q);
  } else
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
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 2D thermal analysis
__global__ void kernel_assemblePreConditioner_thermal_2D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int nxy=nx*ny;
    cudapcgMap_t materials_around_node[4] = {
      material[WALK_UP(i,ny)],
      material[WALK_LEFT(WALK_UP(i,ny),ny,nxy)],
      material[WALK_LEFT(i,ny,nxy)],
      material[i]
    };

    cudapcgVar_t q;
    Preconditioner_thermal_2D_ElemByElem(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &materials_around_node[0], &q);

    M[i] = q;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D thermal conductivity analysis
__global__ void kernel_applyPreConditioner_thermal_2D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int nxy=nx*ny;
    cudapcgMap_t materials_around_node[4] = {
      material[WALK_UP(i,ny)],
      material[WALK_LEFT(WALK_UP(i,ny),ny,nxy)],
      material[WALK_LEFT(i,ny,nxy)],
      material[i]
    };

    cudapcgVar_t q;
    Preconditioner_thermal_2D_ElemByElem(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &materials_around_node[0], &q);
    
    res[i] = scl1*v1[i]*q + (v2 == NULL ? 0.0 : scl2*v2[i]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D thermal analysis
__global__ void kernel_Aprod_thermal_2D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    unsigned int dof, nxy=nx*ny;
    cudapcgMap_t matkey = material[i];
    cudapcgVar_t res;

    Aprod_thermal_2D_ElemByElem( i, 0,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res);
    dof = WALK_DOWN(i,ny);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_2D_ElemByElem( i, 1,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res);
    dof = WALK_RIGHT(dof,ny,nxy);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_2D_ElemByElem( i, 2,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res);
    dof = WALK_UP(dof,ny);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_2D_ElemByElem( i, 3,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res);
    dof = i;
    atomicAdd(&q[dof],scl*res);

  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D thermal conductivity analysis
__global__ void kernel_applyPreConditioner_thermal_2D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t *res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int nxy=nx*ny;
    cudapcgVar_t densities_around_node[4] = {
      (cudapcgVar_t)field[WALK_UP(i,ny)],
      (cudapcgVar_t)field[WALK_LEFT(WALK_UP(i,ny),ny,nxy)],
      (cudapcgVar_t)field[WALK_LEFT(i,ny,nxy)],
      (cudapcgVar_t)field[i]
    };

    #pragma unroll
    for (int j=0; j<4; j++) { densities_around_node[j] *= (1.0/65535.0)*(fmax-fmin); densities_around_node[j] += fmin; }

    cudapcgVar_t q;
    Preconditioner_thermal_2D_ElemByElem_ScalarDensityField(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &densities_around_node[0], &q);
    
    res[i] = scl1*v1[i]*q + (v2 == NULL ? 0.0 : scl2*v2[i]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D thermal analysis
__global__ void kernel_Aprod_thermal_2D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    unsigned int dof, nxy=nx*ny;
    scl *= (cudapcgVar_t)(fmin + (fmax-fmin)*field[i]*(1.0/65535.0));
    cudapcgVar_t res;

    Aprod_thermal_2D_ElemByElem( i, 0,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, &res);
    dof = WALK_DOWN(i,ny);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_2D_ElemByElem( i, 1,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, &res);
    dof = WALK_RIGHT(dof,ny,nxy);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_2D_ElemByElem( i, 2,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, &res);
    dof = WALK_UP(dof,ny);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_2D_ElemByElem( i, 3,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, &res);
    dof = i;
    atomicAdd(&q[dof],scl*res);

  }
}
//------------------------------------------------------------------------------

//-----------------------------------------
///////////////////////////////////////////
/////////////// THERMAL 3D ////////////////
///////////////////////////////////////////
//-----------------------------------------

//---------------------------------
///////////////////////////////////
////////// NODE-BY-NODE ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 3D thermal analysis
__global__ void kernel_assemblePreConditioner_thermal_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t q;
    Preconditioner_thermal_3D_NodeByNode_SparseQ1(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    material[i], &q);
    M[i]=q;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 3D thermal conductivity analysis
__global__ void kernel_applyPreConditioner_thermal_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t q;
    Preconditioner_thermal_3D_NodeByNode_SparseQ1(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    material[i], &q);
    res[i] = scl1*v1[i]*q + (v2 == NULL ? 0.0 : scl2*v2[i]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    // Local vars to store result
    cudapcgVar_t res;
    Aprod_thermal_3D_NodeByNode_SparseQ1( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[i], ny, nx, nz, &res);
    q[i] = scl*res + q[i]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_PreConditionerAprod_thermal_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgMap_t matkey = material[i];

    cudapcgVar_t res;
    Aprod_thermal_3D_NodeByNode_SparseQ1( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res);
    cudapcgVar_t p;
    Preconditioner_thermal_3D_NodeByNode_SparseQ1(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    matkey, &p);

    res*=p;
    q[i] = scl*res + q[i]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_thermal_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q=0.0, vi=v1[gi];
    vi*=(v2 == NULL ? vi : v2[gi]);
    Preconditioner_thermal_3D_NodeByNode_SparseQ1(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      material[gi], &q);
      cache[li] = (double) (vi*q);
  } else
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_thermal_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q=0.0;
    Aprod_thermal_3D_NodeByNode_SparseQ1( gi,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[gi], ny, nx, nz, &q);
      cache[li] = (double) (v[gi]*q);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_thermal_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q=0.0;
    Aprod_thermal_3D_NodeByNode_SparseQ1( gi,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[gi], ny, nx, nz, &q);
      cache[li] = (double) (q*q);
  } else
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
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 3D thermal analysis
__global__ void kernel_assemblePreConditioner_thermal_3D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    // Get row and col ids
    int row = i%ny, col = (i%(nx*ny))/ny, layer = i/(nx*ny);

    cudapcgMap_t materials_around_node[8] = {
      material[PERIODICNUM_3D(row-1,col,layer,ny,nx,nz)],
      material[PERIODICNUM_3D(row-1,col-1,layer,ny,nx,nz)],
      material[PERIODICNUM_3D(row,col-1,layer,ny,nx,nz)],
      material[i],
      material[PERIODICNUM_3D(row-1,col,layer-1,ny,nx,nz)],
      material[PERIODICNUM_3D(row-1,col-1,layer-1,ny,nx,nz)],
      material[PERIODICNUM_3D(row,col-1,layer-1,ny,nx,nz)],
      material[WALK_NEAR(i,(nx*ny),(nx*ny*nz))]
    };

    cudapcgVar_t q;
    Preconditioner_thermal_3D_ElemByElem_SparseQ1(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &materials_around_node[0], &q);

    M[i] = q;
  }
}
//------------------------------------------------------------------------------
__global__ void kernel_applyPreConditioner_thermal_3D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz,  cudapcgVar_t *res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    // Get row and col ids
    int row = i%ny, col = (i%(nx*ny))/ny, layer = i/(nx*ny);

    cudapcgMap_t materials_around_node[8] = {
      material[PERIODICNUM_3D(row-1,col,layer,ny,nx,nz)],
      material[PERIODICNUM_3D(row-1,col-1,layer,ny,nx,nz)],
      material[PERIODICNUM_3D(row,col-1,layer,ny,nx,nz)],
      material[i],
      material[PERIODICNUM_3D(row-1,col,layer-1,ny,nx,nz)],
      material[PERIODICNUM_3D(row-1,col-1,layer-1,ny,nx,nz)],
      material[PERIODICNUM_3D(row,col-1,layer-1,ny,nx,nz)],
      material[WALK_NEAR(i,(nx*ny),(nx*ny*nz))]
    };

    cudapcgVar_t q;
    Preconditioner_thermal_3D_ElemByElem_SparseQ1(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &materials_around_node[0], &q);

    res[i] = scl1*v1[i]*q + (v2 == NULL ? 0.0 : scl2*v2[i]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    // Get row and col ids
    int row = (i+1)%ny, col = (i%(nx*ny))/ny, layer = i/(nx*ny);
    unsigned int dof;

    cudapcgMap_t matkey = material[i];
    cudapcgVar_t res;

    Aprod_thermal_3D_ElemByElem_SparseQ1<0>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row,col,layer,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<1>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row,col+1,layer,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<2>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row-1,col+1,layer,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<3>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res);
    dof = i; //PERIODICNUM_3D(row-1,col,layer,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<4>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row,col,layer+1,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<5>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row,col+1,layer+1,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<6>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row-1,col+1,layer+1,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<7>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row-1,col,layer+1,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);
  }
}
//------------------------------------------------------------------------------
__global__ void kernel_applyPreConditioner_thermal_3D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz,  cudapcgVar_t *res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    // Get row and col ids
    int row = i%ny, col = (i%(nx*ny))/ny, layer = i/(nx*ny);

    cudapcgVar_t densities_around_node[8] = {
      (cudapcgVar_t)field[PERIODICNUM_3D(row-1,col,layer,ny,nx,nz)],
      (cudapcgVar_t)field[PERIODICNUM_3D(row-1,col-1,layer,ny,nx,nz)],
      (cudapcgVar_t)field[PERIODICNUM_3D(row,col-1,layer,ny,nx,nz)],
      (cudapcgVar_t)field[i],
      (cudapcgVar_t)field[PERIODICNUM_3D(row-1,col,layer-1,ny,nx,nz)],
      (cudapcgVar_t)field[PERIODICNUM_3D(row-1,col-1,layer-1,ny,nx,nz)],
      (cudapcgVar_t)field[PERIODICNUM_3D(row,col-1,layer-1,ny,nx,nz)],
      (cudapcgVar_t)field[WALK_NEAR(i,(nx*ny),(nx*ny*nz))]
    };

    #pragma unroll
    for (int j=0; j<8; j++) { densities_around_node[j] *= (1.0/65535.0)*(fmax-fmin); densities_around_node[j] += fmin; }

    cudapcgVar_t q;
    Preconditioner_thermal_3D_ElemByElem_ScalarDensityField_SparseQ1(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &densities_around_node[0], &q);

    res[i] = scl1*v1[i]*q + (v2 == NULL ? 0.0 : scl2*v2[i]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    // Get row and col ids
    int row = (i+1)%ny, col = (i%(nx*ny))/ny, layer = i/(nx*ny);
    unsigned int dof;

    scl *= (cudapcgVar_t)(fmin + (fmax-fmin)*field[i]*(1.0/65535.0));
    cudapcgVar_t res;

    Aprod_thermal_3D_ElemByElem_SparseQ1<0>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row,col,layer,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<1>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row,col+1,layer,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<2>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row-1,col+1,layer,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<3>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, nz, &res);
    dof = i; //PERIODICNUM_3D(row-1,col,layer,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<4>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row,col,layer+1,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<5>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row,col+1,layer+1,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<6>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row-1,col+1,layer+1,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);

    Aprod_thermal_3D_ElemByElem_SparseQ1<7>( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, 0, ny, nx, nz, &res);
    dof = PERIODICNUM_3D(row-1,col,layer+1,ny,nx,nz);
    atomicAdd(&q[dof],scl*res);
  }
}
//------------------------------------------------------------------------------

//-----------------------------------------
///////////////////////////////////////////
/////////////// ELASTIC 2D ////////////////
///////////////////////////////////////////
//-----------------------------------------

//---------------------------------
///////////////////////////////////
////////// NODE-BY-NODE ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 2D elasticity analysis
__global__ void kernel_assemblePreConditioner_elastic_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/2){ // 2DOFs per node
    // Local var to store result
    cudapcgVar_t q[2];
    Preconditioner_elastic_2D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      material[i], &q[0]);
    M[2*i]   = q[0];
    M[2*i+1] = q[1];
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D elasticity analysis
__global__ void kernel_applyPreConditioner_elastic_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/2){ // 2DOFs per node
    // Local var to store result
    cudapcgVar_t q[2];
    Preconditioner_elastic_2D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      material[i], &q[0]);

    res[2*i]   = scl1*  v1[2*i]*q[0] + (v2 == NULL ? 0.0 : scl2*  v2[2*i]);
    res[2*i+1] = scl1*v1[2*i+1]*q[1] + (v2 == NULL ? 0.0 : scl2*v2[2*i+1]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D elasticity analysis
__global__ void kernel_Aprod_elastic_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/2){
    // Local var to store result
    cudapcgVar_t res[2];
    Aprod_elastic_2D_NodeByNode( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[i], ny, nx, &res[0]);
    q[2*i]   = scl*res[0] +   q[2*i]*scl_prev;
    q[2*i+1] = scl*res[1] + q[2*i+1]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D elasticity analysis
__global__ void kernel_PreConditionerAprod_elastic_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/2){
    cudapcgMap_t matkey = material[i];

    cudapcgVar_t res[2];
    Aprod_elastic_2D_NodeByNode( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res[0]);
    cudapcgVar_t p[2];
    Preconditioner_elastic_2D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      matkey, &p[0]);

    res[0]*=p[0];
    res[1]*=p[1]; 
    q[2*i]   = scl*res[0] +   q[2*i]*scl_prev;
    q[2*i+1] = scl*res[1] + q[2*i+1]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_elastic_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim/2){
    cudapcgVar_t q[2], vi[2]={v1[2*gi],v1[2*gi+1]};
    vi[0]*=(v2 == NULL ? vi[0] : v2[2*gi]); vi[1]*=(v2 == NULL ? vi[1] : v2[2*gi+1]);
    Preconditioner_elastic_2D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      material[gi], &q[0]);
      cache[li] = (double) (vi[0]*q[0]+vi[1]*q[1]);
  } else
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_elastic_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim/2){
    cudapcgVar_t q[2];
    Aprod_elastic_2D_NodeByNode( gi,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[gi], ny, nx, &q[0]);
      cache[li] = (double) (v[2*gi]*q[0] + v[2*gi+1]*q[1]);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_elastic_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim/2){
    cudapcgVar_t q[2];
    Aprod_elastic_2D_NodeByNode( gi,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[gi], ny, nx, &q[0]);
      cache[li] = (double) (q[0]*q[0] + q[1]*q[1]);
  } else
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
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 2D elasticity analysis
__global__ void kernel_assemblePreConditioner_elastic_2D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){ 
    unsigned int nxy=nx*ny;
    cudapcgMap_t materials_around_node[4] = {
      material[WALK_UP(i,ny)],
      material[WALK_LEFT(WALK_UP(i,ny),ny,nxy)],
      material[WALK_LEFT(i,ny,nxy)],
      material[i]
    };

    cudapcgVar_t q[2];
    Preconditioner_elastic_2D_ElemByElem(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &materials_around_node[0], &q[0]);

    M[2*i]   = q[0];
    M[2*i+1] = q[1];
  }
}
//------------------------------------------------------------------------------
__global__ void kernel_applyPreConditioner_elastic_2D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){ 
    unsigned int nxy=nx*ny;
    cudapcgMap_t materials_around_node[4] = {
      material[WALK_UP(i,ny)],
      material[WALK_LEFT(WALK_UP(i,ny),ny,nxy)],
      material[WALK_LEFT(i,ny,nxy)],
      material[i]
    };

    cudapcgVar_t q[2];
    Preconditioner_elastic_2D_ElemByElem(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &materials_around_node[0], &q[0]);

    res[2*i]   = scl1*  v1[2*i]*q[0] + (v2 == NULL ? 0.0 : scl2*  v2[2*i]);
    res[2*i+1] = scl1*v1[2*i+1]*q[1] + (v2 == NULL ? 0.0 : scl2*v2[2*i+1]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D elasticity analysis
__global__ void kernel_Aprod_elastic_2D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    unsigned int dof, nxy=nx*ny;

    cudapcgMap_t matkey = material[i];

    cudapcgVar_t res[2];

    Aprod_elastic_2D_ElemByElem( i, 0,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res[0]);
    dof = WALK_DOWN(i,ny);
    atomicAdd(  &q[2*dof],scl*res[0]);
    atomicAdd(&q[2*dof+1],scl*res[1]);

    Aprod_elastic_2D_ElemByElem( i, 1,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res[0]);
    dof = WALK_RIGHT(dof,ny,nxy);
    atomicAdd(  &q[2*dof],scl*res[0]);
    atomicAdd(&q[2*dof+1],scl*res[1]);

    Aprod_elastic_2D_ElemByElem( i, 2,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res[0]);
    dof = WALK_UP(dof,ny);
    atomicAdd(  &q[2*dof],scl*res[0]);
    atomicAdd(&q[2*dof+1],scl*res[1]);

    Aprod_elastic_2D_ElemByElem( i, 3,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res[0]);
    dof = i;
    atomicAdd(  &q[2*dof],scl*res[0]);
    atomicAdd(&q[2*dof+1],scl*res[1]);
  }
}
//------------------------------------------------------------------------------
__global__ void kernel_applyPreConditioner_elastic_2D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t *res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){ 
    unsigned int nxy=nx*ny;
    cudapcgMap_t materials_around_node[4] = {
      material[WALK_UP(i,ny)],
      material[WALK_LEFT(WALK_UP(i,ny),ny,nxy)],
      material[WALK_LEFT(i,ny,nxy)],
      material[i]
    };

    cudapcgVar_t densities_around_node[4] = {
      (cudapcgVar_t)field[WALK_UP(i,ny)],
      (cudapcgVar_t)field[WALK_LEFT(WALK_UP(i,ny),ny,nxy)],
      (cudapcgVar_t)field[WALK_LEFT(i,ny,nxy)],
      (cudapcgVar_t)field[i]
    };

    #pragma unroll
    for (int j=0; j<4; j++) { densities_around_node[j] *= (1.0/65535.0)*(fmax-fmin); densities_around_node[j] += fmin; }

    cudapcgVar_t q[2];
    Preconditioner_elastic_2D_ElemByElem_ScalarDensityField(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &materials_around_node[0], &densities_around_node[0], &q[0]);

    res[2*i]   = scl1*  v1[2*i]*q[0] + (v2 == NULL ? 0.0 : scl2*  v2[2*i]);
    res[2*i+1] = scl1*v1[2*i+1]*q[1] + (v2 == NULL ? 0.0 : scl2*v2[2*i+1]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D elasticity analysis
__global__ void kernel_Aprod_elastic_2D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    unsigned int dof, nxy=nx*ny;

    cudapcgMap_t matkey = material[i];
    scl *= (cudapcgVar_t)(fmin + (fmax-fmin)*field[i]*(1.0/65535.0));

    cudapcgVar_t res[2];

    Aprod_elastic_2D_ElemByElem( i, 0,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res[0]);
    dof = WALK_DOWN(i,ny);
    atomicAdd(  &q[2*dof],scl*res[0]);
    atomicAdd(&q[2*dof+1],scl*res[1]);

    Aprod_elastic_2D_ElemByElem( i, 1,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res[0]);
    dof = WALK_RIGHT(dof,ny,nxy);
    atomicAdd(  &q[2*dof],scl*res[0]);
    atomicAdd(&q[2*dof+1],scl*res[1]);

    Aprod_elastic_2D_ElemByElem( i, 2,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res[0]);
    dof = WALK_UP(dof,ny);
    atomicAdd(  &q[2*dof],scl*res[0]);
    atomicAdd(&q[2*dof+1],scl*res[1]);

    Aprod_elastic_2D_ElemByElem( i, 3,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, &res[0]);
    dof = i;
    atomicAdd(  &q[2*dof],scl*res[0]);
    atomicAdd(&q[2*dof+1],scl*res[1]);
  }
}
//------------------------------------------------------------------------------

//-----------------------------------------
///////////////////////////////////////////
/////////////// ELASTIC 3D ////////////////
///////////////////////////////////////////
//-----------------------------------------

//---------------------------------
///////////////////////////////////
////////// NODE-BY-NODE ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 3D elasticity analysis
__global__ void kernel_assemblePreConditioner_elastic_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/3){ // 3DOFs per node
    // Local var to store result
    cudapcgVar_t q[3];
    Preconditioner_elastic_3D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      material[i], &q[0]);
    M[3*i]   = q[0];
    M[3*i+1] = q[1];
    M[3*i+2] = q[2];
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D elasticity analysis
__global__ void kernel_applyPreConditioner_elastic_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/3){ // 3DOFs per node
    // Local var to store result
    cudapcgVar_t q[3];
    Preconditioner_elastic_3D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      material[i], &q[0]);

    res[3*i]   = scl1*  v1[3*i]*q[0] + (v2 == NULL ? 0.0 : scl2*  v2[3*i]);
    res[3*i+1] = scl1*v1[3*i+1]*q[1] + (v2 == NULL ? 0.0 : scl2*v2[3*i+1]);
    res[3*i+2] = scl1*v1[3*i+2]*q[2] + (v2 == NULL ? 0.0 : scl2*v2[3*i+2]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D elasticity analysis
__global__ void kernel_Aprod_elastic_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/3){
    // Local var to store result
    cudapcgVar_t res[3];
    Aprod_elastic_3D_NodeByNode( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[i], ny, nx, nz, &res[0]);
    q[3*i]   = scl*res[0] +   q[3*i]*scl_prev;
    q[3*i+1] = scl*res[1] + q[3*i+1]*scl_prev;
    q[3*i+2] = scl*res[2] + q[3*i+2]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D elasticity analysis
__global__ void kernel_PreConditionerAprod_elastic_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/3){
    cudapcgMap_t matkey = material[i];

    cudapcgVar_t res[3];
    Aprod_elastic_3D_NodeByNode( i,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    cudapcgVar_t p[3];
    Preconditioner_elastic_3D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      matkey, &p[0]);

    res[0]*=p[0];
    res[1]*=p[1];
    res[2]*=p[2];
    q[3*i]   = scl*res[0] +   q[3*i]*scl_prev;
    q[3*i+1] = scl*res[1] + q[3*i+1]*scl_prev;
    q[3*i+2] = scl*res[2] + q[3*i+2]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_elastic_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim/3){
    cudapcgVar_t q[3], vi[3]={v1[3*gi],v1[3*gi+1],v1[3*gi+2]};
    vi[0]*=(v2 == NULL ? vi[0] : v2[3*gi]); vi[1]*=(v2 == NULL ? vi[1] : v2[3*gi+1]); vi[2]*=(v2 == NULL ? vi[2] : v2[3*gi+2]);
    Preconditioner_elastic_3D_NodeByNode(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      material[gi], &q[0]);
      cache[li] = (double) (vi[0]*q[0]+vi[1]*q[1]+vi[2]*q[2]);
  } else
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_elastic_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim/3){
    cudapcgVar_t q[3];
    Aprod_elastic_3D_NodeByNode( gi,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[gi], ny, nx, nz, &q[0]);
      cache[li] = (double) (v[3*gi]*q[0] + v[3*gi+1]*q[1] + v[3*gi+2]*q[2]);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_elastic_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim/3){
    cudapcgVar_t q[3];
    Aprod_elastic_3D_NodeByNode( gi,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, material[gi], ny, nx, nz, &q[0]);
      cache[li] = (double) (q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
  } else
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
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 3D elasticity analysis
__global__ void kernel_assemblePreConditioner_elastic_3D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    // Get row and col ids
    int row = i%ny, col = (i%(nx*ny))/ny, layer = i/(nx*ny);

    cudapcgMap_t materials_around_node[8] = {
      material[PERIODICNUM_3D(row-1,col,layer,ny,nx,nz)],
      material[PERIODICNUM_3D(row-1,col-1,layer,ny,nx,nz)],
      material[PERIODICNUM_3D(row,col-1,layer,ny,nx,nz)],
      material[i],
      material[PERIODICNUM_3D(row-1,col,layer-1,ny,nx,nz)],
      material[PERIODICNUM_3D(row-1,col-1,layer-1,ny,nx,nz)],
      material[PERIODICNUM_3D(row,col-1,layer-1,ny,nx,nz)],
      material[WALK_NEAR(i,(nx*ny),(nx*ny*nz))]
    };

    cudapcgVar_t q[3];
    Preconditioner_elastic_3D_ElemByElem(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &materials_around_node[0], &q[0]);

    M[3*i]   = q[0];
    M[3*i+1] = q[1];
    M[3*i+2] = q[2];
  }
}
//------------------------------------------------------------------------------
__global__ void kernel_applyPreConditioner_elastic_3D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){    
    // Get row and col ids
    int row = i%ny, col = (i%(nx*ny))/ny, layer = i/(nx*ny);

    cudapcgMap_t materials_around_node[8] = {
      material[PERIODICNUM_3D(row-1,col,layer,ny,nx,nz)],
      material[PERIODICNUM_3D(row-1,col-1,layer,ny,nx,nz)],
      material[PERIODICNUM_3D(row,col-1,layer,ny,nx,nz)],
      material[i],
      material[PERIODICNUM_3D(row-1,col,layer-1,ny,nx,nz)],
      material[PERIODICNUM_3D(row-1,col-1,layer-1,ny,nx,nz)],
      material[PERIODICNUM_3D(row,col-1,layer-1,ny,nx,nz)],
      material[WALK_NEAR(i,(nx*ny),(nx*ny*nz))]
    };
    
    cudapcgVar_t q[3];
    Preconditioner_elastic_3D_ElemByElem(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &materials_around_node[0], &q[0]);

    res[3*i]   = scl1*  v1[3*i]*q[0] + (v2 == NULL ? 0.0 : scl2*  v2[3*i]);
    res[3*i+1] = scl1*v1[3*i+1]*q[1] + (v2 == NULL ? 0.0 : scl2*v2[3*i+1]);
    res[3*i+2] = scl1*v1[3*i+2]*q[2] + (v2 == NULL ? 0.0 : scl2*v2[3*i+2]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D elasticity analysis
__global__ void kernel_Aprod_elastic_3D_ElemByElem(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    // Get row and col ids
    int row = (i+1)%ny, col = (i%(nx*ny))/ny, layer = i/(nx*ny);
    unsigned int dof;

    cudapcgMap_t matkey = material[i];

    cudapcgVar_t res[3];

    Aprod_elastic_3D_ElemByElem( i, 0,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row,col,layer,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 1,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row,col+1,layer,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 2,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row-1,col+1,layer,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 3,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = i; //PERIODICNUM_3D(row-1,col,layer,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 4,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row,col,layer+1,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 5,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row,col+1,layer+1,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 6,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row-1,col+1,layer+1,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 7,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row-1,col,layer+1,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);
  }
}
//------------------------------------------------------------------------------
__global__ void kernel_applyPreConditioner_elastic_3D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){    
    // Get row and col ids
    int row = i%ny, col = (i%(nx*ny))/ny, layer = i/(nx*ny);

    cudapcgMap_t materials_around_node[8] = {
      material[PERIODICNUM_3D(row-1,col,layer,ny,nx,nz)],
      material[PERIODICNUM_3D(row-1,col-1,layer,ny,nx,nz)],
      material[PERIODICNUM_3D(row,col-1,layer,ny,nx,nz)],
      material[i],
      material[PERIODICNUM_3D(row-1,col,layer-1,ny,nx,nz)],
      material[PERIODICNUM_3D(row-1,col-1,layer-1,ny,nx,nz)],
      material[PERIODICNUM_3D(row,col-1,layer-1,ny,nx,nz)],
      material[WALK_NEAR(i,(nx*ny),(nx*ny*nz))]
    };

    cudapcgVar_t densities_around_node[8] = {
      (cudapcgVar_t)field[PERIODICNUM_3D(row-1,col,layer,ny,nx,nz)],
      (cudapcgVar_t)field[PERIODICNUM_3D(row-1,col-1,layer,ny,nx,nz)],
      (cudapcgVar_t)field[PERIODICNUM_3D(row,col-1,layer,ny,nx,nz)],
      (cudapcgVar_t)field[i],
      (cudapcgVar_t)field[PERIODICNUM_3D(row-1,col,layer-1,ny,nx,nz)],
      (cudapcgVar_t)field[PERIODICNUM_3D(row-1,col-1,layer-1,ny,nx,nz)],
      (cudapcgVar_t)field[PERIODICNUM_3D(row,col-1,layer-1,ny,nx,nz)],
      (cudapcgVar_t)field[WALK_NEAR(i,(nx*ny),(nx*ny*nz))]
    };

    #pragma unroll
    for (int j=0; j<8; j++) { densities_around_node[j] *= (1.0/65535.0)*(fmax-fmin); densities_around_node[j] += fmin; }
    
    cudapcgVar_t q[3];
    Preconditioner_elastic_3D_ElemByElem_ScalarDensityField(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &materials_around_node[0], &densities_around_node[0], &q[0]);

    res[3*i]   = scl1*  v1[3*i]*q[0] + (v2 == NULL ? 0.0 : scl2*  v2[3*i]);
    res[3*i+1] = scl1*v1[3*i+1]*q[1] + (v2 == NULL ? 0.0 : scl2*v2[3*i+1]);
    res[3*i+2] = scl1*v1[3*i+2]*q[2] + (v2 == NULL ? 0.0 : scl2*v2[3*i+2]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D elasticity analysis
__global__ void kernel_Aprod_elastic_3D_ElemByElem_ScalarDensityField(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    // Get row and col ids
    int row = (i+1)%ny, col = (i%(nx*ny))/ny, layer = i/(nx*ny);
    unsigned int dof;

    cudapcgMap_t matkey = material[i];
    scl *= (cudapcgVar_t)(fmin + (fmax-fmin)*field[i]*(1.0/65535.0));

    cudapcgVar_t res[3];

    Aprod_elastic_3D_ElemByElem( i, 0,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row,col,layer,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 1,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row,col+1,layer,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 2,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row-1,col+1,layer,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 3,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = i; //PERIODICNUM_3D(row-1,col,layer,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 4,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row,col,layer+1,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 5,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row,col+1,layer+1,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 6,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row-1,col+1,layer+1,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);

    Aprod_elastic_3D_ElemByElem( i, 7,
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, matkey, ny, nx, nz, &res[0]);
    dof = PERIODICNUM_3D(row-1,col,layer+1,ny,nx,nz);
    atomicAdd(  &q[3*dof],scl*res[0]);
    atomicAdd(&q[3*dof+1],scl*res[1]);
    atomicAdd(&q[3*dof+2],scl*res[2]);
  }
}
//------------------------------------------------------------------------------

//-----------------------------------------
///////////////////////////////////////////
//////////////// FLUID 2D /////////////////
///////////////////////////////////////////
//-----------------------------------------

//---------------------------------
///////////////////////////////////
////////// NODE-BY-NODE ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    if (fluidkey){
      cudapcgVar_t q[3];
      cudapcgIdMap_t id = DOFMap[i];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Preconditioner_fluid_2D_NodeByNode_Pore(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          &q[0]);
        res[2*id]   = scl1*  v1[2*id]*q[0] + (v2 == NULL ? 0.0 : scl2*  v2[2*id]);
        res[2*id+1] = scl1*v1[2*id+1]*q[1] + (v2 == NULL ? 0.0 : scl2*v2[2*id+1]);
      } else {
        Preconditioner_fluid_2D_NodeByNode_Border(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          fluidkey,&q[2]);
      }
      res[id+nVelocityNodes*2] = scl1*v1[id+nVelocityNodes*2]*q[2] + (v2 == NULL ? 0.0 : scl2*v2[id+nVelocityNodes*2]);
    }
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D permeability analysis
__global__ void kernel_Aprod_fluid_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    if (fluidkey){
      cudapcgVar_t res[3];
      cudapcgIdMap_t id = DOFMap[i];
      // Velocity DOFs
      if (id < nVelocityNodes){
        
        Aprod_fluid_2D_NodeByNode_Pore( i,
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v, nVelocityNodes, DOFMap, ny, nx, &res[0]);
        
        q[2*id]   = scl*res[0] +   q[2*id]*scl_prev;
        q[2*id+1] = scl*res[1] + q[2*id+1]*scl_prev;
      } else {

        Aprod_fluid_2D_NodeByNode_Border( i,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, &res[2]);
      }
      
      q[id+nVelocityNodes*2] = scl*res[2] + q[id+nVelocityNodes*2]*scl_prev;
    }
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q[3], vi;
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Preconditioner_fluid_2D_NodeByNode_Pore(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          &q[0]);
        vi = v1[2*id];
        vi*=(v2 == NULL ? vi : v2[2*id]);
        cache[li]  = (double) (vi*q[0]);
        vi = v1[2*id+1];
        vi*=(v2 == NULL ? vi : v2[2*id+1]);
        cache[li] += (double) (vi*q[1]);
      } else {
        Preconditioner_fluid_2D_NodeByNode_Border(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          fluidkey,&q[2]);
      }
      vi = v1[id+nVelocityNodes*2];
      vi*=(v2 == NULL ? vi : v2[id+nVelocityNodes*2]);
      cache[li] += (double) (vi*q[2]);
    }
  }
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_fluid_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q[3];
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Aprod_fluid_2D_NodeByNode_Pore( gi,
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v, nVelocityNodes, DOFMap, ny, nx, &q[0]);
        
        cache[li] = (double) (v[2*id]*q[0] + v[2*id+1]*q[1]);
      } else {

        Aprod_fluid_2D_NodeByNode_Border( gi,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, &q[2]);
      }
      cache[li] += (double) (v[id+nVelocityNodes*2]*q[2]);
    }
  }
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_fluid_2D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q[3];
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Aprod_fluid_2D_NodeByNode_Pore( gi,
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v, nVelocityNodes, DOFMap, ny, nx, &q[0]);
        
        cache[li] = (double) (q[0]*q[0] + q[1]*q[1]);
      } else {

        Aprod_fluid_2D_NodeByNode_Border( gi,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, &q[2]);
      }
      cache[li] += (double) (q[2]*q[2]);
    }
  }
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
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t q[3];
    Preconditioner_fluid_2D_NodeByNode_Pore(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &q[0]);
    res[2*i]     = scl1*    v1[2*i]*q[0] + (v2 == NULL ? 0.0 : scl2*    v2[2*i]);
    res[2*i+1]   = scl1*  v1[2*i+1]*q[1] + (v2 == NULL ? 0.0 : scl2*  v2[2*i+1]);
    res[i+dim*2] = scl1*v1[i+dim*2]*q[2] + (v2 == NULL ? 0.0 : scl2*v2[i+dim*2]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t q;
    Preconditioner_fluid_2D_NodeByNode_Border(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      FluidMap[i],&q);
    
    res[i+nVelocityNodes*3] = scl1*v1[i+nVelocityNodes*3]*q + (v2 == NULL ? 0.0 : scl2*v2[i+nVelocityNodes*3]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D permeability analysis
__global__ void kernel_Aprod_fluid_2D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

      cudapcgVar_t res[3];

      Aprod_fluid_2D_NodeByNode_Pore( NodeMap[i],
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, dim, DOFMap, ny, nx, &res[0]);

      q[2*i]     = scl*res[0] +     q[2*i]*scl_prev;
      q[2*i+1]   = scl*res[1] +   q[2*i+1]*scl_prev;
      q[i+dim*2] = scl*res[2] + q[i+dim*2]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D permeability analysis
__global__ void kernel_Aprod_fluid_2D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

      cudapcgVar_t res;

      Aprod_fluid_2D_NodeByNode_Border( NodeMap[i+nVelocityNodes],
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, FluidMap[i], DOFMap, ny, nx, &res);

      q[i+nVelocityNodes*3] = scl*res + q[i+nVelocityNodes*3]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D permeability analysis
__global__ void kernel_PreConditionerAprod_fluid_2D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

      cudapcgVar_t res[3];

      Aprod_fluid_2D_NodeByNode_Pore( NodeMap[i],
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, dim, DOFMap, ny, nx, &res[0]);

      cudapcgVar_t p[3];
      Preconditioner_fluid_2D_NodeByNode_Pore(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        &p[0]);

      res[0]*=p[0];
      res[1]*=p[1];
      res[2]*=p[2];
      q[2*i]     = scl*res[0] +     q[2*i]*scl_prev;
      q[2*i+1]   = scl*res[1] +   q[2*i+1]*scl_prev;
      q[i+dim*2] = scl*res[2] + q[i+dim*2]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D permeability analysis
__global__ void kernel_PreConditionerAprod_fluid_2D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
      cudapcgFlag_t fluidkey = FluidMap[i];

      cudapcgVar_t res;
      Aprod_fluid_2D_NodeByNode_Border( NodeMap[i+nVelocityNodes],
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, &res);

    cudapcgVar_t p;
    Preconditioner_fluid_2D_NodeByNode_Border(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      fluidkey,&p);

      res*=p;
      q[i+nVelocityNodes*3] = scl*res + q[i+nVelocityNodes*3]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_2D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[3], vi;
    Preconditioner_fluid_2D_NodeByNode_Pore(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &q[0]);
    vi = v1[2*gi];
    vi*=(v2 == NULL ? vi : v2[2*gi]);
    cache[li]  = (double) (vi*q[0]);
    vi = v1[2*gi+1];
    vi*=(v2 == NULL ? vi : v2[2*gi+1]);
    cache[li] += (double) (vi*q[1]);
    vi = v1[gi+dim*2];
    vi*=(v2 == NULL ? vi : v2[gi+dim*2]);
    cache[li] += (double) (vi*q[2]); 
  } else
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
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_2D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q, vi;
    Preconditioner_fluid_2D_NodeByNode_Border(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      FluidMap[gi],&q);
    vi = v1[gi+nVelocityNodes*3];
    vi*=(v2 == NULL ? vi : v2[gi+nVelocityNodes*3]);
    cache[li] = (double) (vi*q); 
  } else
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_fluid_2D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[3];
    Aprod_fluid_2D_NodeByNode_Pore( NodeMap[gi],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, dim, DOFMap, ny, nx, &q[0]);
    cache[li] = (double) (v[2*gi]*q[0] + v[2*gi+1]*q[1] + v[gi+dim*2]*q[2]);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_fluid_2D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[3];
    Aprod_fluid_2D_NodeByNode_Pore( NodeMap[gi],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, dim, DOFMap, ny, nx, &q[0]);
    cache[li] = (double) (q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
  } else
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_fluid_2D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q;
    Aprod_fluid_2D_NodeByNode_Border( NodeMap[gi+nVelocityNodes],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, nVelocityNodes, FluidMap[gi], DOFMap, ny, nx, &q);
      cache[li] = (double) (v[gi+nVelocityNodes*3]*q);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_fluid_2D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q;
    Aprod_fluid_2D_NodeByNode_Border( NodeMap[gi+nVelocityNodes],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, nVelocityNodes, FluidMap[gi], DOFMap, ny, nx, &q);
      cache[li] = (double) (q*q);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,M^-1*A*v)
__global__ void kernel_dotPreConditionerA2prod_fluid_2D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    cudapcgVar_t q;
    Aprod_fluid_2D_NodeByNode_Border( NodeMap[gi+nVelocityNodes],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, nVelocityNodes, fluidkey, DOFMap, ny, nx, &q);
    cudapcgVar_t p;
    Preconditioner_fluid_2D_NodeByNode_Border(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      fluidkey,&p);
      cache[li] = (double) (q*p*q);
  } else
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
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
/////// STENCIL-BY-STENCIL ////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_2D_StencilByStencil(
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    if (fluidkey){
      cudapcgVar_t q;
      cudapcgIdMap_t id = DOFMap[i];
      // Velocity DOFs
      if (id < nVelocityNodes){
        res[2*id]   = scl1*  v1[2*id]*0.25 + (v2 == NULL ? 0.0 : scl2*  v2[2*id]);
        res[2*id+1] = scl1*v1[2*id+1]*0.25 + (v2 == NULL ? 0.0 : scl2*v2[2*id+1]);
        q=-2.25;
      } else {
        q = -9.0 / (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) );
      }
      res[id+nVelocityNodes*2] = scl1*v1[id+nVelocityNodes*2]*q + (v2 == NULL ? 0.0 : scl2*v2[id+nVelocityNodes*2]);
    }
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D permeability analysis
__global__ void kernel_Aprod_fluid_2D_StencilByStencil(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    if (fluidkey){
      cudapcgVar_t res[3];
      cudapcgIdMap_t id = DOFMap[i];
      // Velocity DOFs
      if (id < nVelocityNodes){
        
        Aprod_fluid_2D_StencilByStencil_Pore( i, v, dim, DOFMap, ny, nx, &res[0]);
        
        q[2*id]   = scl*res[0] +   q[2*id]*scl_prev;
        q[2*id+1] = scl*res[1] + q[2*id+1]*scl_prev;
      } else {

        Aprod_fluid_2D_NodeByNode_Border( i,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, &res[2]);
      }
      
      q[id+nVelocityNodes*2] = scl*res[2] + q[id+nVelocityNodes*2]*scl_prev;
    }
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_2D_StencilByStencil(
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q, vi;
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        vi = v1[2*id];
        vi*=(v2 == NULL ? vi : v2[2*id]);
        cache[li]  = (double) (vi*0.25);
        vi = v1[2*id+1];
        vi*=(v2 == NULL ? vi : v2[2*id+1]);
        cache[li] += (double) (vi*0.25);
        q = -2.25;
      } else {
        q = -9.0 / (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) );
      }
      vi = v1[id+nVelocityNodes*2];
      vi*=(v2 == NULL ? vi : v2[id+nVelocityNodes*2]);
      cache[li] += (double) (vi*q);
    }
  }
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_fluid_2D_StencilByStencil(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q[3];
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Aprod_fluid_2D_StencilByStencil_Pore( gi, v, dim, DOFMap, ny, nx, &q[0]);
        
        cache[li] = (double) (v[2*id]*q[0] + v[2*id+1]*q[1]);
      } else {

        Aprod_fluid_2D_NodeByNode_Border( gi,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, &q[2]);
      }
      cache[li] += (double) (v[id+nVelocityNodes*2]*q[2]);
    }
  }
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_fluid_2D_StencilByStencil(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q[3];
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Aprod_fluid_2D_StencilByStencil_Pore( gi, v, dim, DOFMap, ny, nx, &q[0]);
        
        cache[li] = (double) (q[0]*q[0] + q[1]*q[1]);
      } else {

        Aprod_fluid_2D_NodeByNode_Border( gi,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, &q[2]);
      }
      cache[li] += (double) (q[2]*q[2]);
    }
  }
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
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_2D_StencilByStencil_Pore(
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    res[2*i]     = scl1*    v1[2*i]* 0.25 + (v2 == NULL ? 0.0 : scl2*    v2[2*i]);
    res[2*i+1]   = scl1*  v1[2*i+1]* 0.25 + (v2 == NULL ? 0.0 : scl2*  v2[2*i+1]);
    res[i+dim*2] = scl1*v1[i+dim*2]*-2.25 + (v2 == NULL ? 0.0 : scl2*v2[i+dim*2]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_2D_StencilByStencil_Border(
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    cudapcgVar_t q = -9.0 / (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) );
    res[i+nVelocityNodes*3] = scl1*v1[i+nVelocityNodes*3]*q + (v2 == NULL ? 0.0 : scl2*v2[i+nVelocityNodes*3]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D permeability analysis
__global__ void kernel_applyinvPreConditioner_fluid_2D_StencilByStencil_Pore(
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    res[2*i]     = scl1*    v1[2*i]*4.0         + (v2 == NULL ? 0.0 : scl2*    v2[2*i]);
    res[2*i+1]   = scl1*  v1[2*i+1]*4.0         + (v2 == NULL ? 0.0 : scl2*  v2[2*i+1]);
    res[i+dim*2] = scl1*v1[i+dim*2]*(1.0/-2.25) + (v2 == NULL ? 0.0 : scl2*v2[i+dim*2]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D permeability analysis
__global__ void kernel_applyinvPreConditioner_fluid_2D_StencilByStencil_Border(
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    cudapcgVar_t q = (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) ) / -9.0;
    res[i+nVelocityNodes*3] = scl1*v1[i+nVelocityNodes*3]*q + (v2 == NULL ? 0.0 : scl2*v2[i+nVelocityNodes*3]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D permeability analysis
__global__ void kernel_Aprod_fluid_2D_StencilByStencil_Pore(
    cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

      cudapcgVar_t res[3];
      Aprod_fluid_2D_StencilByStencil_Pore( NodeMap[i], v, dim, DOFMap, ny, nx, &res[0]);

      q[2*i]     = scl*res[0] +     q[2*i]*scl_prev;
      q[2*i+1]   = scl*res[1] +   q[2*i+1]*scl_prev;
      q[i+dim*2] = scl*res[2] + q[i+dim*2]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D permeability analysis
__global__ void kernel_PreConditionerAprod_fluid_2D_StencilByStencil_Pore(
    cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

      cudapcgVar_t res[3];
      Aprod_fluid_2D_StencilByStencil_Pore( NodeMap[i], v, dim, DOFMap, ny, nx, &res[0]);

      res[0]*= 0.25;
      res[1]*= 0.25;
      res[2]*=-2.25;
      q[2*i]     = scl*res[0] +     q[2*i]*scl_prev;
      q[2*i+1]   = scl*res[1] +   q[2*i+1]*scl_prev;
      q[i+dim*2] = scl*res[2] + q[i+dim*2]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_2D_StencilByStencil_Pore(
  cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t vi, lcl;
    vi = v1[2*gi];
    vi*=(v2 == NULL ? vi : v2[2*gi]);
    lcl  = (double) (vi* 0.25);
    vi = v1[2*gi+1];
    vi*=(v2 == NULL ? vi : v2[2*gi+1]);
    lcl += (double) (vi* 0.25);
    vi = v1[gi+dim*2];
    vi*=(v2 == NULL ? vi : v2[gi+dim*2]);
    lcl += (double) (vi*-2.25); 
    cache[li] = lcl;
  } else
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
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_2D_StencilByStencil_Border(
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    cudapcgVar_t vi, q = -9.0 / (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) );
    vi = v1[gi+nVelocityNodes*3];
    vi*=(v2 == NULL ? vi : v2[gi+nVelocityNodes*3]);
    cache[li] = (double) (vi*q); 
  } else
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
//------------------------------------------------------------------------------
// dot(v,M*v)
__global__ void kernel_dotinvPreConditioner_fluid_2D_StencilByStencil_Pore(
  cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t vi, lcl;
    vi = v1[2*gi];
    vi*=(v2 == NULL ? vi : v2[2*gi]);
    lcl  = (double) (vi* 4.0);
    vi = v1[2*gi+1];
    vi*=(v2 == NULL ? vi : v2[2*gi+1]);
    lcl += (double) (vi* 4.0);
    vi = v1[gi+dim*2];
    vi*=(v2 == NULL ? vi : v2[gi+dim*2]);
    lcl += (double) (vi*(4.0/9.0)); 
    cache[li] = lcl;
  } else
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
//------------------------------------------------------------------------------
// dot(v,M*v)
__global__ void kernel_dotinvPreConditioner_fluid_2D_StencilByStencil_Border(
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    cudapcgVar_t vi, q = -9.0 / (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) );
    vi = v1[gi+nVelocityNodes*3];
    vi*=(v2 == NULL ? vi : v2[gi+nVelocityNodes*3]);
    cache[li] = (double) (vi/q); 
  } else
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_fluid_2D_StencilByStencil_Pore(
    cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[3];
    Aprod_fluid_2D_StencilByStencil_Pore( NodeMap[gi], v, dim, DOFMap, ny, nx, &q[0]);
    cache[li] = (double) (v[2*gi]*q[0] + v[2*gi+1]*q[1] + v[gi+dim*2]*q[2]);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_fluid_2D_StencilByStencil_Pore(
    cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[3];
    Aprod_fluid_2D_StencilByStencil_Pore( NodeMap[gi], v, dim, DOFMap, ny, nx, &q[0]);
    cache[li] = (double) (q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,M^-1*A*v)
__global__ void kernel_dotPreConditionerA2prod_fluid_2D_StencilByStencil_Pore(
    cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[3];
    Aprod_fluid_2D_StencilByStencil_Pore( NodeMap[gi], v, dim, DOFMap, ny, nx, &q[0]);
    cache[li] = (double) (q[0]*q[0]*0.25 + q[1]*q[1]*0.25 + q[2]*q[2]*-2.25);
  } else
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
//------------------------------------------------------------------------------

//-----------------------------------------
///////////////////////////////////////////
//////////////// FLUID 3D /////////////////
///////////////////////////////////////////
//-----------------------------------------

//---------------------------------
///////////////////////////////////
////////// NODE-BY-NODE ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 3D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    if (fluidkey){
      cudapcgVar_t q[4];
      cudapcgIdMap_t id = DOFMap[i];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Preconditioner_fluid_3D_NodeByNode_Pore(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          &q[0]);
        res[3*id]   = scl1*  v1[3*id]*q[0] + (v2 == NULL ? 0.0 : scl2*  v2[3*id]);
        res[3*id+1] = scl1*v1[3*id+1]*q[1] + (v2 == NULL ? 0.0 : scl2*v2[3*id+1]);
        res[3*id+2] = scl1*v1[3*id+2]*q[2] + (v2 == NULL ? 0.0 : scl2*v2[3*id+2]);
      } else {
        Preconditioner_fluid_3D_NodeByNode_Border(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          fluidkey,&q[3]);
      }
      res[id+nVelocityNodes*3] = scl1*v1[id+nVelocityNodes*3]*q[3] + (v2 == NULL ? 0.0 : scl2*v2[id+nVelocityNodes*3]);
    }
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D permeability analysis
__global__ void kernel_Aprod_fluid_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    if (fluidkey){
      cudapcgVar_t res[4];
      cudapcgIdMap_t id = DOFMap[i];
      // Velocity DOFs
      if (id < nVelocityNodes){
        
        Aprod_fluid_3D_NodeByNode_Pore( i,
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v, nVelocityNodes, DOFMap, ny, nx, nz, &res[0]);
        
        q[3*id]   = scl*res[0] +   q[3*id]*scl_prev;
        q[3*id+1] = scl*res[1] + q[3*id+1]*scl_prev;
        q[3*id+2] = scl*res[2] + q[3*id+2]*scl_prev;
        
      } else {

        Aprod_fluid_3D_NodeByNode_Border( i,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, nz, &res[3]);
      }
      
      q[id+nVelocityNodes*3] = scl*res[3] + q[id+nVelocityNodes*3]*scl_prev;
    }
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q[4], vi;
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Preconditioner_fluid_3D_NodeByNode_Pore(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          &q[0]);
        vi = v1[3*id];
        vi*=(v2 == NULL ? vi : v2[3*id]);
        cache[li]  = (double) (vi*q[0]);
        vi = v1[3*id+1];
        vi*=(v2 == NULL ? vi : v2[3*id+1]);
        cache[li] += (double) (vi*q[1]);
        vi = v1[3*id+2];
        vi*=(v2 == NULL ? vi : v2[3*id+2]);
        cache[li] += (double) (vi*q[2]);
      } else {
        Preconditioner_fluid_3D_NodeByNode_Border(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          fluidkey,&q[3]);
      }
      vi = v1[id+nVelocityNodes*3];
      vi*=(v2 == NULL ? vi : v2[id+nVelocityNodes*3]);
      cache[li] += (double) (vi*q[3]);
    }
  }
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_fluid_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q[4];
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Aprod_fluid_3D_NodeByNode_Pore( gi,
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v, nVelocityNodes, DOFMap, ny, nx, nz, &q[0]);
        
        cache[li] = (double) (v[3*id]*q[0] + v[3*id+1]*q[1] + v[3*id+2]*q[2]);
      } else {

        Aprod_fluid_3D_NodeByNode_Border( gi,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, nz, &q[3]);
      }
      cache[li] += (double) (v[id+nVelocityNodes*3]*q[3]);
    }
  }
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_fluid_3D_NodeByNode(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q[4];
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Aprod_fluid_3D_NodeByNode_Pore( gi,
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v, nVelocityNodes, DOFMap, ny, nx, nz, &q[0]);
        
        cache[li] = (double) (q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
      } else {

        Aprod_fluid_3D_NodeByNode_Border( gi,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, nz, &q[3]);
      }
      cache[li] += (double) (q[3]*q[3]);
    }
  }
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
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 3D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t q[4];
    Preconditioner_fluid_3D_NodeByNode_Pore(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &q[0]);
    res[3*i]     = scl1*    v1[3*i]*q[0] + (v2 == NULL ? 0.0 : scl2*    v2[3*i]);
    res[3*i+1]   = scl1*  v1[3*i+1]*q[1] + (v2 == NULL ? 0.0 : scl2*  v2[3*i+1]);
    res[3*i+2]   = scl1*  v1[3*i+2]*q[2] + (v2 == NULL ? 0.0 : scl2*  v2[3*i+2]);
    res[i+dim*3] = scl1*v1[i+dim*3]*q[3] + (v2 == NULL ? 0.0 : scl2*v2[i+dim*3]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 3D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    cudapcgVar_t q;
    Preconditioner_fluid_3D_NodeByNode_Border(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      fluidkey,&q);
    res[i+nVelocityNodes*4] = scl1*v1[i+nVelocityNodes*4]*q + (v2 == NULL ? 0.0 : scl2*v2[i+nVelocityNodes*4]);
  }
}
//------------------------------------------------------------------------------ 
__global__ void kernel_Aprod_fluid_3D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t res[4];

    Aprod_fluid_3D_NodeByNode_Pore( NodeMap[i],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, dim, DOFMap, ny, nx, nz, &res[0]);

    q[3*i]     = scl*res[0] +     q[3*i]*scl_prev;
    q[3*i+1]   = scl*res[1] +   q[3*i+1]*scl_prev;
    q[3*i+2]   = scl*res[2] +   q[3*i+2]*scl_prev;
    q[i+dim*3] = scl*res[3] + q[i+dim*3]*scl_prev;
  }
}
//------------------------------------------------------------------------------ 
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D permeability analysis
__global__ void kernel_Aprod_fluid_3D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t res;

    Aprod_fluid_3D_NodeByNode_Border( NodeMap[i+nVelocityNodes],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, nVelocityNodes, FluidMap[i], DOFMap, ny, nx, nz, &res);

    q[i+nVelocityNodes*4] = scl*res + q[i+nVelocityNodes*4]*scl_prev;
  }
}
//------------------------------------------------------------------------------ 
__global__ void kernel_PreConditionerAprod_fluid_3D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){

    cudapcgVar_t res[4];
    Aprod_fluid_3D_NodeByNode_Pore( NodeMap[i],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, dim, DOFMap, ny, nx, nz, &res[0]);

    cudapcgVar_t p[4];
    Preconditioner_fluid_3D_NodeByNode_Pore(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &p[0]);

    res[0]*=p[0];
    res[1]*=p[1];
    res[2]*=p[2];
    res[3]*=p[3];
    q[3*i]     = scl*res[0] +     q[3*i]*scl_prev;
    q[3*i+1]   = scl*res[1] +   q[3*i+1]*scl_prev;
    q[3*i+2]   = scl*res[2] +   q[3*i+2]*scl_prev;
    q[i+dim*3] = scl*res[3] + q[i+dim*3]*scl_prev;
  }
}
//------------------------------------------------------------------------------ 
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D permeability analysis
__global__ void kernel_PreConditionerAprod_fluid_3D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];

    cudapcgVar_t res;
    Aprod_fluid_3D_NodeByNode_Border( NodeMap[i+nVelocityNodes],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, nVelocityNodes, fluidkey, DOFMap, ny, nx, nz, &res);

    cudapcgVar_t p;
    Preconditioner_fluid_3D_NodeByNode_Border(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      fluidkey,&p);
    
    res*=p;
    q[i+nVelocityNodes*4] = scl*res + q[i+nVelocityNodes*4]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_3D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[4], vi;
    Preconditioner_fluid_3D_NodeByNode_Pore(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      &q[0]);
    vi = v1[3*gi];
    vi*=(v2 == NULL ? vi : v2[3*gi]);
    cache[li]  = (double) (vi*q[0]);
    vi = v1[3*gi+1];
    vi*=(v2 == NULL ? vi : v2[3*gi+1]);
    cache[li] += (double) (vi*q[1]);
    vi = v1[3*gi+2];
    vi*=(v2 == NULL ? vi : v2[3*gi+2]);
    cache[li] += (double) (vi*q[2]);
    vi = v1[gi+dim*3];
    vi*=(v2 == NULL ? vi : v2[gi+dim*3]);
    cache[li] += (double) (vi*q[3]); 
  } else
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
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_3D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q, vi;
    Preconditioner_fluid_3D_NodeByNode_Border(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      FluidMap[gi],&q);
    vi = v1[gi+nVelocityNodes*4];
    vi*=(v2 == NULL ? vi : v2[gi+nVelocityNodes*4]);
    cache[li] = (double) (vi*q); 
  } else
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_fluid_3D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[4];
    Aprod_fluid_3D_NodeByNode_Pore( NodeMap[gi],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, dim, DOFMap, ny, nx, nz, &q[0]);
      cache[li] = (double) (v[3*gi]*q[0] + v[3*gi+1]*q[1] + v[3*gi+2]*q[2] + v[gi+dim*3]*q[3]);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_fluid_3D_NodeByNode_Pore(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[4];
    Aprod_fluid_3D_NodeByNode_Pore( NodeMap[gi],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, dim, DOFMap, ny, nx, nz, &q[0]);
      cache[li] = (double) (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  } else
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_fluid_3D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q;
    Aprod_fluid_3D_NodeByNode_Border( NodeMap[gi+nVelocityNodes],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, nVelocityNodes, FluidMap[gi], DOFMap, ny, nx, nz, &q);
      cache[li] = (double) (v[gi+nVelocityNodes*4]*q);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_fluid_3D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q;
    Aprod_fluid_3D_NodeByNode_Border( NodeMap[gi+nVelocityNodes],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, nVelocityNodes, FluidMap[gi], DOFMap, ny, nx, nz, &q);
      cache[li] = (double) (q*q);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,M^-1*A*v)
__global__ void kernel_dotPreConditionerA2prod_fluid_3D_NodeByNode_Border(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    cudapcgVar_t q;
    Aprod_fluid_3D_NodeByNode_Border( NodeMap[gi+nVelocityNodes],
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v, nVelocityNodes, fluidkey, DOFMap, ny, nx, nz, &q);
    cudapcgVar_t p;
    Preconditioner_fluid_3D_NodeByNode_Border(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      fluidkey,&p);
    cache[li] = (double) (q*p*q);
  } else
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
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
/////// STENCIL-BY-STENCIL ////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 3D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_3D_StencilByStencil(
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    if (fluidkey){
      cudapcgIdMap_t id = DOFMap[i];
      // Velocity DOFs
      if (id < nVelocityNodes){
        res[3*id]   = scl1*  v1[3*id]* 0.28125 + (v2 == NULL ? 0.0 : scl2*  v2[3*id]);
        res[3*id+1] = scl1*v1[3*id+1]* 0.28125 + (v2 == NULL ? 0.0 : scl2*v2[3*id+1]);
        res[3*id+2] = scl1*v1[3*id+2]* 0.28125 + (v2 == NULL ? 0.0 : scl2*v2[3*id+2]);
        res[id+nVelocityNodes*3] = scl1*v1[id+nVelocityNodes*3]*-2.25 + (v2 == NULL ? 0.0 : scl2*v2[id+nVelocityNodes*3]);
      } else {
        cudapcgVar_t q = -18.0 / (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) + ((fluidkey>>4)&1) + ((fluidkey>>5)&1) + ((fluidkey>>6)&1) + ((fluidkey>>7)&1) );
        res[id+nVelocityNodes*3] = scl1*v1[id+nVelocityNodes*3]*q + (v2 == NULL ? 0.0 : scl2*v2[id+nVelocityNodes*3]);
      }
    }
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D permeability analysis
__global__ void kernel_Aprod_fluid_3D_StencilByStencil(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    if (fluidkey){
      cudapcgVar_t res[4];
      cudapcgIdMap_t id = DOFMap[i];
      // Velocity DOFs
      if (id < nVelocityNodes){
        
        Aprod_fluid_3D_StencilByStencil_Pore( i, v, nVelocityNodes, DOFMap, ny, nx, nz, &res[0]);
        
        q[3*id]   = scl*res[0] +   q[3*id]*scl_prev;
        q[3*id+1] = scl*res[1] + q[3*id+1]*scl_prev;
        q[3*id+2] = scl*res[2] + q[3*id+2]*scl_prev;
        
      } else {

        Aprod_fluid_3D_NodeByNode_Border( i,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, nz, &res[3]);
      }
      
      q[id+nVelocityNodes*3] = scl*res[3] + q[id+nVelocityNodes*3]*scl_prev;
    }
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_3D_StencilByStencil(
  cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q, vi;
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        vi = v1[3*id];
        vi*=(v2 == NULL ? vi : v2[3*id]);
        cache[li]  = (double) (vi*0.28125);
        vi = v1[3*id+1];
        vi*=(v2 == NULL ? vi : v2[3*id+1]);
        cache[li] += (double) (vi*0.28125);
        vi = v1[3*id+2];
        vi*=(v2 == NULL ? vi : v2[3*id+2]);
        cache[li] += (double) (vi*0.28125);
        q = -2.25;
      } else {
        q = -18.0 / (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) + ((fluidkey>>4)&1) + ((fluidkey>>5)&1) + ((fluidkey>>6)&1) + ((fluidkey>>7)&1) );
      }
      vi = v1[id+nVelocityNodes*3];
      vi*=(v2 == NULL ? vi : v2[id+nVelocityNodes*3]);
      cache[li] += (double) (vi*q);
    }
  }
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_fluid_3D_StencilByStencil(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q[4];
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Aprod_fluid_3D_StencilByStencil_Pore( gi, v, nVelocityNodes, DOFMap, ny, nx, nz, &q[0]);
        
        cache[li] = (double) (v[3*id]*q[0] + v[3*id+1]*q[1] + v[3*id+2]*q[2]);
      } else {

        Aprod_fluid_3D_NodeByNode_Border( gi,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, nz, &q[3]);
      }
      cache[li] += (double) (v[id+nVelocityNodes*3]*q[3]);
    }
  }
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_fluid_3D_StencilByStencil(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      cudapcgVar_t *K,
    #endif
    cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  cache[li] = 0.0;
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    if (fluidkey){
      cudapcgVar_t q[4];
      cudapcgIdMap_t id = DOFMap[gi];
      // Velocity DOFs
      if (id < nVelocityNodes){
        Aprod_fluid_3D_StencilByStencil_Pore( gi, v, nVelocityNodes, DOFMap, ny, nx, nz, &q[0]);
        
        cache[li] = (double) (q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
      } else {

        Aprod_fluid_3D_NodeByNode_Border( gi,
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v, nVelocityNodes, fluidkey, DOFMap, ny, nx, nz, &q[3]);
      }
      cache[li] += (double) (q[3]*q[3]);
    }
  }
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
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 3D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_3D_StencilByStencil_Pore(
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    res[3*i]     = scl1*    v1[3*i]* 0.28125 + (v2 == NULL ? 0.0 : scl2*    v2[3*i]);
    res[3*i+1]   = scl1*  v1[3*i+1]* 0.28125 + (v2 == NULL ? 0.0 : scl2*  v2[3*i+1]);
    res[3*i+2]   = scl1*  v1[3*i+2]* 0.28125 + (v2 == NULL ? 0.0 : scl2*  v2[3*i+2]);
    res[i+dim*3] = scl1*v1[i+dim*3]*-2.25    + (v2 == NULL ? 0.0 : scl2*v2[i+dim*3]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 3D permeability analysis
__global__ void kernel_applyPreConditioner_fluid_3D_StencilByStencil_Border(
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    cudapcgVar_t q = -18.0 / (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) + ((fluidkey>>4)&1) + ((fluidkey>>5)&1) + ((fluidkey>>6)&1) + ((fluidkey>>7)&1) );
    res[i+nVelocityNodes*4] = scl1*v1[i+nVelocityNodes*4]*q + (v2 == NULL ? 0.0 : scl2*v2[i+nVelocityNodes*4]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 3D permeability analysis
__global__ void kernel_applyinvPreConditioner_fluid_3D_StencilByStencil_Pore(
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    res[3*i]     = scl1*    v1[3*i]*(1.0/0.28125) + (v2 == NULL ? 0.0 : scl2*    v2[3*i]);
    res[3*i+1]   = scl1*  v1[3*i+1]*(1.0/0.28125) + (v2 == NULL ? 0.0 : scl2*  v2[3*i+1]);
    res[3*i+2]   = scl1*  v1[3*i+2]*(1.0/0.28125) + (v2 == NULL ? 0.0 : scl2*  v2[3*i+2]);
    res[i+dim*3] = scl1*v1[i+dim*3]*(1.0/-2.25)   + (v2 == NULL ? 0.0 : scl2*v2[i+dim*3]);
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 3D permeability analysis
__global__ void kernel_applyinvPreConditioner_fluid_3D_StencilByStencil_Border(
    cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgFlag_t fluidkey = FluidMap[i];
    cudapcgVar_t q =  (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) + ((fluidkey>>4)&1) + ((fluidkey>>5)&1) + ((fluidkey>>6)&1) + ((fluidkey>>7)&1) ) / -18.0;
    res[i+nVelocityNodes*4] = scl1*v1[i+nVelocityNodes*4]*q + (v2 == NULL ? 0.0 : scl2*v2[i+nVelocityNodes*4]);
  }
}
//------------------------------------------------------------------------------ 
__global__ void kernel_Aprod_fluid_3D_StencilByStencil_Pore(
    cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t res[4];
    Aprod_fluid_3D_StencilByStencil_Pore( NodeMap[i], v, dim, DOFMap, ny, nx, nz, &res[0]);

    q[3*i]     = scl*res[0] +     q[3*i]*scl_prev;
    q[3*i+1]   = scl*res[1] +   q[3*i+1]*scl_prev;
    q[3*i+2]   = scl*res[2] +   q[3*i+2]*scl_prev;
    q[i+dim*3] = scl*res[3] + q[i+dim*3]*scl_prev;
  }
}
//------------------------------------------------------------------------------ 
__global__ void kernel_PreConditionerAprod_fluid_3D_StencilByStencil_Pore(
    cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t res[4];
    Aprod_fluid_3D_StencilByStencil_Pore( NodeMap[i], v, dim, DOFMap, ny, nx, nz, &res[0]);

    res[0]*= 0.28125;
    res[1]*= 0.28125;
    res[2]*= 0.28125;
    res[3]*=-2.25;
    q[3*i]     = scl*res[0] +     q[3*i]*scl_prev;
    q[3*i+1]   = scl*res[1] +   q[3*i+1]*scl_prev;
    q[3*i+2]   = scl*res[2] +   q[3*i+2]*scl_prev;
    q[i+dim*3] = scl*res[3] + q[i+dim*3]*scl_prev;
  }
}
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_3D_StencilByStencil_Pore(
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t vi, lcl;
    vi = v1[3*gi];
    vi*=(v2 == NULL ? vi : v2[3*gi]);
    lcl  = (double) (vi* 0.28125);
    vi = v1[3*gi+1];
    vi*=(v2 == NULL ? vi : v2[3*gi+1]);
    lcl += (double) (vi* 0.28125);
    vi = v1[3*gi+2];
    vi*=(v2 == NULL ? vi : v2[3*gi+2]);
    lcl += (double) (vi* 0.28125);
    vi = v1[gi+dim*3];
    vi*=(v2 == NULL ? vi : v2[gi+dim*3]);
    lcl += (double) (vi*-2.25); 
    cache[li] = lcl;
  } else
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
//------------------------------------------------------------------------------
// dot(v,M^-1*v)
__global__ void kernel_dotPreConditioner_fluid_3D_StencilByStencil_Border(
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    cudapcgVar_t q = -18.0 / (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) + ((fluidkey>>4)&1) + ((fluidkey>>5)&1) + ((fluidkey>>6)&1) + ((fluidkey>>7)&1) );
    cudapcgVar_t vi = v1[gi+nVelocityNodes*4];
    vi*=(v2 == NULL ? vi : v2[gi+nVelocityNodes*4]);
    cache[li] = (double) (vi*q); 
  } else
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
//------------------------------------------------------------------------------
// dot(v,M*v)
__global__ void kernel_dotinvPreConditioner_fluid_3D_StencilByStencil_Pore(
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t vi, lcl;
    vi = v1[3*gi];
    vi*=(v2 == NULL ? vi : v2[3*gi]);
    lcl  = (double) (vi*(64.0/18.0));
    vi = v1[3*gi+1];
    vi*=(v2 == NULL ? vi : v2[3*gi+1]);
    lcl += (double) (vi*(64.0/18.0));
    vi = v1[3*gi+2];
    vi*=(v2 == NULL ? vi : v2[3*gi+2]);
    lcl += (double) (vi*(64.0/18.0));
    vi = v1[gi+dim*3];
    vi*=(v2 == NULL ? vi : v2[gi+dim*3]);
    lcl += (double) (vi*(4.0/9.0)); 
    cache[li] = lcl;
  } else
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
//------------------------------------------------------------------------------
// dot(v,M*v)
__global__ void kernel_dotinvPreConditioner_fluid_3D_StencilByStencil_Border(
    cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgFlag_t fluidkey = FluidMap[gi];
    cudapcgVar_t q = -18.0 / (cudapcgVar_t) ( (fluidkey&1) + ((fluidkey>>1)&1) + ((fluidkey>>2)&1) + ((fluidkey>>3)&1) + ((fluidkey>>4)&1) + ((fluidkey>>5)&1) + ((fluidkey>>6)&1) + ((fluidkey>>7)&1) );
    cudapcgVar_t vi = v1[gi+nVelocityNodes*4];
    vi*=(v2 == NULL ? vi : v2[gi+nVelocityNodes*4]);
    cache[li] = (double) (vi/q); 
  } else
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
//------------------------------------------------------------------------------
// dot(v,A*v)
__global__ void kernel_dotAprod_fluid_3D_StencilByStencil_Pore(
  cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[4];
    Aprod_fluid_3D_StencilByStencil_Pore( NodeMap[gi], v, dim, DOFMap, ny, nx, nz, &q[0]);
    cache[li] = (double) (v[3*gi]*q[0] + v[3*gi+1]*q[1] + v[3*gi+2]*q[2] + v[gi+dim*3]*q[3]);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,A*v)
__global__ void kernel_dotA2prod_fluid_3D_StencilByStencil_Pore(
  cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[4];
    Aprod_fluid_3D_StencilByStencil_Pore( NodeMap[gi], v, dim, DOFMap, ny, nx, nz, &q[0]);
    cache[li] = (double) (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  } else
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
//------------------------------------------------------------------------------
// dot(A*v,M^-1*A*v)
__global__ void kernel_dotPreConditionerA2prod_fluid_3D_StencilByStencil_Pore(
  cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res){
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  __shared__ double cache[THREADS_PER_BLOCK];
  if (gi < dim){
    cudapcgVar_t q[4];
    Aprod_fluid_3D_StencilByStencil_Pore( NodeMap[gi], v, dim, DOFMap, ny, nx, nz, &q[0]);
    cache[li] = (double) (q[0]*q[0]*0.28125 + q[1]*q[1]*0.28125 + q[2]*q[2]*0.28125 + q[3]*q[3]*-2.25);
  } else
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
//------------------------------------------------------------------------------