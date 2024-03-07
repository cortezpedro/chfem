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
  
  General purpose kernels are implemented here.
  Host wrappers for NBN and EBE kernels are also implemented here.

  =====================================================================
*/

#include "../includes.h"

#ifndef CUDAPCG_KERNELS_WRAPPERS_H_INCLUDED
#define CUDAPCG_KERNELS_WRAPPERS_H_INCLUDED

//---------------------------
/////////////////////////////
///// MEMORY MANAGEMENT /////
/////////////////////////////
//---------------------------

void allocDotProdArrs(unsigned int dim);
void freeDotProdArrs();

void allocLocalK(unsigned long int size);
void setLocalK(cudapcgVar_t * lclK, unsigned long int size);
void freeLocalK();

// By default, no preconditioner is allocated
// These functions are only called if specifically requested.
void allocPreConditioner(cudapcgModel_t *m);
void freePreConditioner();

//---------------------------
/////////////////////////////
///// VECTOR OPERATIONS /////
/////////////////////////////
//---------------------------

void zeros(cudapcgVar_t * v, unsigned int dim);                                                        // v = 0
void scale(cudapcgVar_t * v, cudapcgVar_t scl, unsigned int dim);                                      // v *= scl
void arrcpy(cudapcgVar_t * v, unsigned int dim, cudapcgVar_t * res);                                   // res = v
cudapcgVar_t max(cudapcgVar_t *v, unsigned int dim);                                                   // max(v)
cudapcgVar_t absmax(cudapcgVar_t *v, unsigned int dim);                                                // max(abs(v))
cudapcgVar_t absmax_signed(cudapcgVar_t *v, unsigned int dim);                                         // (+ or -) max(abs(v))
double reduce(cudapcgVar_t *v, unsigned int dim);                                                      // sum(v)
double absreduce(cudapcgVar_t *v, unsigned int dim);                                                   // sum(abs(v))
double reduce_with_stride(cudapcgVar_t *v, unsigned int dim, unsigned int stride, unsigned int shift); // res = sum(v[shift:stride:end])
double reduce_with_stride_and_scale(cudapcgVar_t *v, unsigned int dim, unsigned int stride, unsigned int shift, cudapcgVar_t scl); // res = sum(scl*v[shift:stride:end])
double dotprod(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim);                                  // dot(v1,v2)
void termbytermmul(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res);        // res = v1 .* v2
void termbytermdiv(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res);        // res = v1 ./ v2
void termbyterminv(cudapcgVar_t * v, unsigned int dim);                                                // v = 1 ./ v 
void axpy(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim, cudapcgVar_t * res);   // res = a*x + y
void axpy_iny(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim);                   // y = a*x + y
void axpy_iny_with_stride(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim, unsigned int stride, unsigned int shift); // y = a*x[shift:stride:end] + y

//---------------------------
/////////////////////////////
///// IMAGE OPERATIONS //////
/////////////////////////////
//---------------------------

// Interpolates field v from [rows/2,cols/2,layers/2] mesh to [rows,cols,layers] mesh
// Obs.: stride => number of variables per node, i.e. scalar or vector field 
void interpl2(cudapcgVar_t *v, unsigned int rows, unsigned int cols, unsigned int layers, unsigned int stride, cudapcgVar_t *res);

//---------------------------
/////////////////////////////
/// MATRIXFREE OPERATIONS ///
/////////////////////////////
//---------------------------

void assemblePreConditioner_thermal_2D(cudapcgModel_t *m); // not being used
void assemblePreConditioner_thermal_3D(cudapcgModel_t *m); // not being used
void assemblePreConditioner_elastic_2D(cudapcgModel_t *m); // not being used
void assemblePreConditioner_elastic_3D(cudapcgModel_t *m); // not being used
//void assemblePreConditioner_fluid_2D(cudapcgModel_t *m); // has not been implemented
//void assemblePreConditioner_fluid_3D(cudapcgModel_t *m); // has not been implemented

// res = scl1*(M^-1*v1) + scl2*v2
// obs.: v2 can be NULL
void applyPreConditioner_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);
void applyPreConditioner_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);
void applyPreConditioner_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);
void applyPreConditioner_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);
void applyPreConditioner_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);
void applyPreConditioner_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);

// res = scl1*(M*v1) + scl2*v2
// obs.: v2 can be NULL
void applyinvPreConditioner_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);
void applyinvPreConditioner_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);
void applyinvPreConditioner_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);
void applyinvPreConditioner_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);
void applyinvPreConditioner_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);
void applyinvPreConditioner_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res);

// res = scl*(A*v) + scl_prev*res
void Aprod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);
void Aprod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);
void Aprod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);
void Aprod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);
void Aprod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);
void Aprod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);

// res = scl*(M^-1*A*v) + scl_prev*res
void PreConditionerAprod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);
void PreConditionerAprod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);
void PreConditionerAprod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);
void PreConditionerAprod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);
void PreConditionerAprod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);
void PreConditionerAprod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res);

// scl*dot(v2,M^-1*v1)
double dotPreConditioner_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);
double dotPreConditioner_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);
double dotPreConditioner_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);
double dotPreConditioner_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);
double dotPreConditioner_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);
double dotPreConditioner_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);

// scl*dot(v2,M*v1)
double dotinvPreConditioner_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);
double dotinvPreConditioner_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);
double dotinvPreConditioner_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);
double dotinvPreConditioner_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);
double dotinvPreConditioner_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);
double dotinvPreConditioner_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl);

// scl*dot(v,A*v)
double dotAprod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotAprod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotAprod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotAprod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotAprod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotAprod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);

// scl*dot(A*v,A*v)
double dotA2prod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotA2prod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotA2prod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotA2prod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotA2prod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotA2prod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);

// scl*dot(A*v,M^-1*A*v)
double dotPreConditionerA2prod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotPreConditionerA2prod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotPreConditionerA2prod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotPreConditionerA2prod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotPreConditionerA2prod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);
double dotPreConditionerA2prod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl);

#endif // CUDAPCG_KERNELS_WRAPPERS_H_INCLUDED
