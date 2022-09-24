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

// Auxiliary API functions
void allocDotProdArrs(unsigned int dim);
void freeDotProdArrs();

void allocLocalK(unsigned long int size);
void setLocalK(cudapcgVar_t * lclK, unsigned long int size);
void freeLocalK();

//-------------------------------
/////////////////////////////////
//////////// GENERAL ////////////
//////////// PURPOSE ////////////
/////////////////////////////////
//-------------------------------

void arrcpy(cudapcgVar_t * res, cudapcgVar_t * v, unsigned int dim);
void zeros(cudapcgVar_t * v, unsigned int dim);
cudapcgVar_t absmax(cudapcgVar_t *v, unsigned int dim);
double reduce(cudapcgVar_t *v, unsigned int dim);
double absreduce(cudapcgVar_t *v, unsigned int dim);
double dotprod(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim);
void termbytermmul(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res);
void termbytermdiv(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res);
void termbyterminv(cudapcgVar_t * v, unsigned int dim);
void saxpy(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim, cudapcgVar_t * res);
void saxpy_iny(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim);
void interpl2(cudapcgVar_t *res, cudapcgVar_t *v, unsigned int dim_x, unsigned int dim_y, unsigned int dim_z, unsigned int stride);


//-------------------------------
/////////////////////////////////
////////// MATRIX-FREE //////////
//////////// KERNELS ////////////
/////////////////////////////////
//-------------------------------

// By default, no preconditioner is allocated
// These functions are only called if specifically requested.
void allocPreConditioner(cudapcgModel_t *m);
void freePreConditioner();
void assemblePreConditioner_thermal_2D(cudapcgModel_t *m);
void assemblePreConditioner_thermal_3D(cudapcgModel_t *m);
void assemblePreConditioner_elastic_2D(cudapcgModel_t *m);
void assemblePreConditioner_elastic_3D(cudapcgModel_t *m);
//void assemblePreConditioner_fluid_2D(cudapcgModel_t *m); // not used
//void assemblePreConditioner_fluid_3D(cudapcgModel_t *m); // not used

// res = M^-1 * v1 + scl*v2
/*
    ATTENTION: FLUID KERNELS -> res = M^-1 * v1
                                (v2 is ignored)
*/
void applyPreConditioner_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res);
void applyPreConditioner_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res);
void applyPreConditioner_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res);
void applyPreConditioner_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res);
void applyPreConditioner_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res);
void applyPreConditioner_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res);

// res = scl*(A*v) + isIncrement*res
void Aprod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t *res);
void Aprod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t *res);
void Aprod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t *res);
void Aprod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t *res);
void Aprod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t *res);
void Aprod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t *res);

#endif // CUDAPCG_KERNELS_WRAPPERS_H_INCLUDED
