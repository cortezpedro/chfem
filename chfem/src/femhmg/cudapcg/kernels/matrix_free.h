/*
  =====================================================================
  Universidade Federal Fluminense (UFF) - Niteroi, Brazil
  Institute of Computing
  Authors: Cortez Lopes, P., Pereira., A.
  contact: pedrocortez@id.uff.br
  =====================================================================
*/

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

__global__ void kernel_applyPreConditioner_thermal_2D_NodeByNode(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_2D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_thermal_2D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_thermal_2D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotA2prod_thermal_2D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotPreConditioner_thermal_2D_NodeByNode(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res);

#else

__global__ void kernel_assemblePreConditioner_thermal_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_thermal_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_thermal_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_thermal_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotA2prod_thermal_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotPreConditioner_thermal_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res);

#endif

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_thermal_2D_ElemByElem(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_applyPreConditioner_thermal_2D_ElemByElem(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_2D_ElemByElem(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_thermal_2D_ElemByElem_ScalarDensityField(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_2D_ElemByElem_ScalarDensityField(cudapcgVar_t * v, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

#else

__global__ void kernel_assemblePreConditioner_thermal_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_applyPreConditioner_thermal_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_thermal_2D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_2D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

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

__global__ void kernel_applyPreConditioner_thermal_3D_NodeByNode(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_3D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_thermal_3D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_thermal_3D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned nz, double *res);

__global__ void kernel_dotA2prod_thermal_3D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned nz, double *res);

__global__ void kernel_dotPreConditioner_thermal_3D_NodeByNode(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res);

#else

__global__ void kernel_assemblePreConditioner_thermal_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_thermal_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_thermal_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_thermal_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned nz, double *res);

__global__ void kernel_dotA2prod_thermal_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned nz, double *res);

__global__ void kernel_dotPreConditioner_thermal_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res);

#endif

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_thermal_3D_ElemByElem(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz);

__global__ void kernel_applyPreConditioner_thermal_3D_ElemByElem(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_3D_ElemByElem(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_thermal_3D_ElemByElem_ScalarDensityField(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_3D_ElemByElem_ScalarDensityField(cudapcgVar_t * v, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

#else

__global__ void kernel_assemblePreConditioner_thermal_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz);

__global__ void kernel_applyPreConditioner_thermal_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_thermal_3D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_3D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

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

__global__ void kernel_applyPreConditioner_elastic_2D_NodeByNode(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_elastic_2D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_elastic_2D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotA2prod_elastic_2D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotPreConditioner_elastic_2D_NodeByNode(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res);

#else

__global__ void kernel_assemblePreConditioner_elastic_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_elastic_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_elastic_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_elastic_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotA2prod_elastic_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotPreConditioner_elastic_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res);

#endif

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_elastic_2D_ElemByElem(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_applyPreConditioner_elastic_2D_ElemByElem(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_ElemByElem(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_elastic_2D_ElemByElem_ScalarDensityField(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_ElemByElem_ScalarDensityField(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

#else

__global__ void kernel_assemblePreConditioner_elastic_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_applyPreConditioner_elastic_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_elastic_2D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

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

__global__ void kernel_applyPreConditioner_elastic_3D_NodeByNode(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_3D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_elastic_3D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_elastic_3D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned nz, double *res);

__global__ void kernel_dotA2prod_elastic_3D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned nz, double *res);

__global__ void kernel_dotPreConditioner_elastic_3D_NodeByNode(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res);

#else

__global__ void kernel_assemblePreConditioner_elastic_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_elastic_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_elastic_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_elastic_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned nz, double *res);

__global__ void kernel_dotA2prod_elastic_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned nz, double *res);

__global__ void kernel_dotPreConditioner_elastic_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgMap_t *material, double *res);

#endif

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_elastic_3D_ElemByElem(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz);

__global__ void kernel_applyPreConditioner_elastic_3D_ElemByElem(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_3D_ElemByElem(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_elastic_3D_ElemByElem_ScalarDensityField(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_3D_ElemByElem_ScalarDensityField(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

#else

__global__ void kernel_assemblePreConditioner_elastic_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz);

__global__ void kernel_applyPreConditioner_elastic_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_elastic_3D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_3D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

#endif

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

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_2D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_fluid_2D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotA2prod_fluid_2D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotPreConditioner_fluid_2D_NodeByNode(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, double *res);

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode_Pore(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res);

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode_Border(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_2D_NodeByNode_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_Aprod_fluid_2D_NodeByNode_Border(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_fluid_2D_NodeByNode_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_fluid_2D_NodeByNode_Border(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_fluid_2D_NodeByNode_Pore(cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotAprod_fluid_2D_NodeByNode_Border(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotA2prod_fluid_2D_NodeByNode_Pore(cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotA2prod_fluid_2D_NodeByNode_Border(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotPreConditionerA2prod_fluid_2D_NodeByNode_Border(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotPreConditioner_fluid_2D_NodeByNode_Pore(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, double *res);

__global__ void kernel_dotPreConditioner_fluid_2D_NodeByNode_Border(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res);

#else

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_fluid_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotA2prod_fluid_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotPreConditioner_fluid_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, double *res);

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res);

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_2D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_Aprod_fluid_2D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_fluid_2D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_fluid_2D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_fluid_2D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotAprod_fluid_2D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotA2prod_fluid_2D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotA2prod_fluid_2D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotPreConditionerA2prod_fluid_2D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

__global__ void kernel_dotPreConditioner_fluid_2D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, double *res);

__global__ void kernel_dotPreConditioner_fluid_2D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res);

#endif

//---------------------------------
///////////////////////////////////
/////// STENCIL-BY-STENCIL ////////
///////////////////////////////////
//---------------------------------

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_Aprod_fluid_2D_StencilByStencil(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);
__global__ void kernel_dotAprod_fluid_2D_StencilByStencil(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);
__global__ void kernel_dotA2prod_fluid_2D_StencilByStencil(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

#else

__global__ void kernel_Aprod_fluid_2D_StencilByStencil(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);
__global__ void kernel_dotAprod_fluid_2D_StencilByStencil(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);
__global__ void kernel_dotA2prod_fluid_2D_StencilByStencil(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, double *res);

#endif

__global__ void kernel_applyPreConditioner_fluid_2D_StencilByStencil(cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res);
__global__ void kernel_dotPreConditioner_fluid_2D_StencilByStencil(cudapcgVar_t *v1, cudapcgVar_t *v2,unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, double *res);
__global__ void kernel_applyPreConditioner_fluid_2D_StencilByStencil_Pore(cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res);
__global__ void kernel_applyPreConditioner_fluid_2D_StencilByStencil_Border(cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);
__global__ void kernel_applyinvPreConditioner_fluid_2D_StencilByStencil_Pore(cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res);
__global__ void kernel_applyinvPreConditioner_fluid_2D_StencilByStencil_Border(cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);
__global__ void kernel_Aprod_fluid_2D_StencilByStencil_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);
__global__ void kernel_PreConditionerAprod_fluid_2D_StencilByStencil_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);
__global__ void kernel_dotPreConditioner_fluid_2D_StencilByStencil_Pore(cudapcgVar_t *v1, cudapcgVar_t *v2,unsigned int dim, double *res);
__global__ void kernel_dotPreConditioner_fluid_2D_StencilByStencil_Border(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res);
__global__ void kernel_dotinvPreConditioner_fluid_2D_StencilByStencil_Pore(cudapcgVar_t *v1, cudapcgVar_t *v2,unsigned int dim, double *res);
__global__ void kernel_dotinvPreConditioner_fluid_2D_StencilByStencil_Border(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res);
__global__ void kernel_dotAprod_fluid_2D_StencilByStencil_Pore(cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res);
__global__ void kernel_dotA2prod_fluid_2D_StencilByStencil_Pore(cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res);
__global__ void kernel_dotPreConditionerA2prod_fluid_2D_StencilByStencil_Pore(cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, double *res);

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

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_3D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_fluid_3D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotA2prod_fluid_3D_NodeByNode(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotPreConditioner_fluid_3D_NodeByNode(cudapcgVar_t *v1, cudapcgVar_t *v2,unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, double *res);

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode_Pore(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res);

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode_Border(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_3D_NodeByNode_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_Aprod_fluid_3D_NodeByNode_Border(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_fluid_3D_NodeByNode_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_fluid_3D_NodeByNode_Border(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_fluid_3D_NodeByNode_Pore(cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotAprod_fluid_3D_NodeByNode_Border(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotA2prod_fluid_3D_NodeByNode_Pore(cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotA2prod_fluid_3D_NodeByNode_Border(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotPreConditionerA2prod_fluid_3D_NodeByNode_Border(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotPreConditioner_fluid_3D_NodeByNode_Pore(cudapcgVar_t *v1, cudapcgVar_t *v2,unsigned int dim, double *res);

__global__ void kernel_dotPreConditioner_fluid_3D_NodeByNode_Border(cudapcgVar_t *v1, cudapcgVar_t *v2,unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res);

#else

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_fluid_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotA2prod_fluid_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotPreConditioner_fluid_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t *v1, cudapcgVar_t *v2,unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, double *res);

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res);

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_3D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_Aprod_fluid_3D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_fluid_3D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_PreConditionerAprod_fluid_3D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);

__global__ void kernel_dotAprod_fluid_3D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotAprod_fluid_3D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dot2Aprod_fluid_3D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotA2prod_fluid_3D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotPreConditionerA2prod_fluid_3D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

__global__ void kernel_dotPreConditioner_fluid_3D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, double *res);

__global__ void kernel_dotPreConditioner_fluid_3D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res);

#endif

//---------------------------------
///////////////////////////////////
/////// STENCIL-BY-STENCIL ////////
///////////////////////////////////
//---------------------------------

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_Aprod_fluid_3D_StencilByStencil(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);
__global__ void kernel_dotAprod_fluid_3D_StencilByStencil(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);
__global__ void kernel_dotA2prod_fluid_3D_StencilByStencil(cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

#else

__global__ void kernel_Aprod_fluid_3D_StencilByStencil(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);
__global__ void kernel_dotAprod_fluid_3D_StencilByStencil(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);
__global__ void kernel_dotA2prod_fluid_3D_StencilByStencil(cudapcgVar_t *K, cudapcgVar_t *v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

#endif

__global__ void kernel_applyPreConditioner_fluid_3D_StencilByStencil(cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res);
__global__ void kernel_dotPreConditioner_fluid_3D_StencilByStencil(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, double *res);
__global__ void kernel_applyPreConditioner_fluid_3D_StencilByStencil_Pore(cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res);
__global__ void kernel_applyPreConditioner_fluid_3D_StencilByStencil_Border(cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);
__global__ void kernel_applyinvPreConditioner_fluid_3D_StencilByStencil_Pore(cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgVar_t * res);
__global__ void kernel_applyinvPreConditioner_fluid_3D_StencilByStencil_Border(cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);
__global__ void kernel_Aprod_fluid_3D_StencilByStencil_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);
__global__ void kernel_PreConditionerAprod_fluid_3D_StencilByStencil_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgVar_t scl_prev);
__global__ void kernel_dotPreConditioner_fluid_3D_StencilByStencil_Pore(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, double *res);
__global__ void kernel_dotPreConditioner_fluid_3D_StencilByStencil_Border(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res);
__global__ void kernel_dotinvPreConditioner_fluid_3D_StencilByStencil_Pore(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, double *res);
__global__ void kernel_dotinvPreConditioner_fluid_3D_StencilByStencil_Border(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, double *res);
__global__ void kernel_dotAprod_fluid_3D_StencilByStencil_Pore(cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res);
__global__ void kernel_dotA2prod_fluid_3D_StencilByStencil_Pore(cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res);
__global__ void kernel_dotPreConditionerA2prod_fluid_3D_StencilByStencil_Pore(cudapcgVar_t *v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, double *res);

#endif //CUDAPCG_KERNELS_SOURCE_H_INCLUDED
