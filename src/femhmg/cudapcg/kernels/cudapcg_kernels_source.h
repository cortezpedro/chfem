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

__global__ void kernel_assemblePreConditioner_thermal_2D_ElemByElem(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_applyPreConditioner_thermal_2D_ElemByElem(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_2D_ElemByElem(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_thermal_2D_ElemByElem_ScalarDensityField(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_2D_ElemByElem_ScalarDensityField(cudapcgVar_t * v, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

#else

__global__ void kernel_assemblePreConditioner_thermal_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_applyPreConditioner_thermal_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_thermal_2D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

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

__global__ void kernel_assemblePreConditioner_thermal_3D_ElemByElem(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz);

__global__ void kernel_applyPreConditioner_thermal_3D_ElemByElem(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_3D_ElemByElem(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_thermal_3D_ElemByElem_ScalarDensityField(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_3D_ElemByElem_ScalarDensityField(cudapcgVar_t * v, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

#else

__global__ void kernel_assemblePreConditioner_thermal_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz);

__global__ void kernel_applyPreConditioner_thermal_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_thermal_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_thermal_3D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

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

__global__ void kernel_applyPreConditioner_elastic_2D_NodeByNode(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#else

__global__ void kernel_assemblePreConditioner_elastic_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material);

__global__ void kernel_applyPreConditioner_elastic_2D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_NodeByNode(cudapcgVar_t *K,cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#endif

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT

__global__ void kernel_assemblePreConditioner_elastic_2D_ElemByElem(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_applyPreConditioner_elastic_2D_ElemByElem(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_ElemByElem(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_elastic_2D_ElemByElem_ScalarDensityField(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_ElemByElem_ScalarDensityField(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

#else

__global__ void kernel_assemblePreConditioner_elastic_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny);

__global__ void kernel_applyPreConditioner_elastic_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_2D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_elastic_2D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, cudapcgVar_t * res);

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

__global__ void kernel_assemblePreConditioner_elastic_3D_ElemByElem(cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz);

__global__ void kernel_applyPreConditioner_elastic_3D_ElemByElem(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_3D_ElemByElem(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_elastic_3D_ElemByElem_ScalarDensityField(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_3D_ElemByElem_ScalarDensityField(cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

#else

__global__ void kernel_assemblePreConditioner_elastic_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz);

__global__ void kernel_applyPreConditioner_elastic_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

__global__ void kernel_Aprod_elastic_3D_ElemByElem(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl);

__global__ void kernel_applyPreConditioner_elastic_3D_ElemByElem_ScalarDensityField(cudapcgVar_t *K, cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, parametricScalarField_t *field, double fmin, double fmax, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * res);

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

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_2D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgVar_t * res);

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode_Border(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_2D_NodeByNode_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

__global__ void kernel_Aprod_fluid_2D_NodeByNode_Border(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#else

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_2D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgVar_t * res);

__global__ void kernel_applyPreConditioner_fluid_2D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_2D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

__global__ void kernel_Aprod_fluid_2D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#endif


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

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_3D_NodeByNode(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgVar_t * res);

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode_Border(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_3D_NodeByNode_Pore(cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

__global__ void kernel_Aprod_fluid_3D_NodeByNode_Border(cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#else

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_3D_NodeByNode(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgVar_t * res);

__global__ void kernel_applyPreConditioner_fluid_3D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, unsigned int nVelocityNodes, cudapcgVar_t * res);

__global__ void kernel_Aprod_fluid_3D_NodeByNode_Pore(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

__global__ void kernel_Aprod_fluid_3D_NodeByNode_Border(cudapcgVar_t *K, cudapcgVar_t * v, unsigned int dim, cudapcgFlag_t *FluidMap, cudapcgIdMap_t *NodeMap, cudapcgIdMap_t *DOFMap, unsigned int nVelocityNodes, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t * q, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#endif

#endif //CUDAPCG_KERNELS_SOURCE_H_INCLUDED
