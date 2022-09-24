#include "../includes.h"

#ifndef CUDAPCG_KERNELS_SOURCE_H_INCLUDED
#define CUDAPCG_KERNELS_SOURCE_H_INCLUDED

// Auxiliary host function to memcpy local matrices to constant symbol
void setConstantLocalK(cudapcgVar_t * lclK, unsigned long int size);

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

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_thermal_2D_NodeByNode(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_thermal_2D_NodeByNode(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_2D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#else

__global__ void kernel_assemblePreConditioner_thermal_2D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_thermal_2D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_2D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#endif

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_thermal_2D_ElemByElem(unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int ny);

__global__ void kernel_Aprod_thermal_2D_ElemByElem(unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int ny, cudapcgVar_t *q);

#else

__global__ void kernel_assemblePreConditioner_thermal_2D_ElemByElem(cudapcgVar_t *K, unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int ny);

__global__ void kernel_Aprod_thermal_2D_ElemByElem(cudapcgVar_t *K, unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int ny, cudapcgVar_t *q);

#endif

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

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_thermal_3D_NodeByNode(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_thermal_3D_NodeByNode(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_3D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#else

__global__ void kernel_assemblePreConditioner_thermal_3D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_thermal_3D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_3D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#endif

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_thermal_3D_ElemByElem(unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_Aprod_thermal_3D_ElemByElem(unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);

__global__ void kernel_Aprod_thermal_3D_ElemByElem_n0(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n1(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n2(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n3(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n4(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n5(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n6(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n7(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);

#else

__global__ void kernel_assemblePreConditioner_thermal_3D_ElemByElem(cudapcgVar_t *K, unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_Aprod_thermal_3D_ElemByElem(cudapcgVar_t *K, unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);

__global__ void kernel_Aprod_thermal_3D_ElemByElem_n0(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n1(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n2(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n3(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n4(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n5(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n6(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n7(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);

#endif

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

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_elastic_2D_NodeByNode(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_elastic_2D_NodeByNode(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#else

__global__ void kernel_assemblePreConditioner_elastic_2D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_elastic_2D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#endif

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_elastic_2D_ElemByElem(unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int ny);

__global__ void kernel_Aprod_elastic_2D_ElemByElem(unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int ny, cudapcgVar_t *q);

#else

__global__ void kernel_assemblePreConditioner_elastic_2D_ElemByElem(cudapcgVar_t *K, unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int ny);

__global__ void kernel_Aprod_elastic_2D_ElemByElem(cudapcgVar_t *K, unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int ny, cudapcgVar_t *q);

#endif

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

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_elastic_3D_NodeByNode(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_elastic_3D_NodeByNode(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_3D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#else

__global__ void kernel_assemblePreConditioner_elastic_3D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_elastic_3D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_3D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#endif

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_elastic_3D_ElemByElem(unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_Aprod_elastic_3D_ElemByElem(unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);

#else

__global__ void kernel_assemblePreConditioner_elastic_3D_ElemByElem(cudapcgVar_t *K, unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_Aprod_elastic_3D_ElemByElem(cudapcgVar_t *K, unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q);

#endif

#endif //CUDAPCG_KERNELS_SOURCE_H_INCLUDED
