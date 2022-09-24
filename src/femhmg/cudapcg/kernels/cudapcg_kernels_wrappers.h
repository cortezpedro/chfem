/*
    Universidade Federal Fluminense (UFF) - Niteroi, Brazil
    Institute of Computing
    Author: Cortez, P.
    History: v1.0 (november/2020)

    API with CPU functions to call CUDA kernels used in cudapcg.h

    All kernels are implemented in cudapcg_kernels.c

    History:
        Initially developed as a final work for the graduate course "Arquitetura
        e Programacao de GPUs", at the Institute of Computing, UFF.

    ATTENTION.1:
        Considers a structured regular mesh of quad elements (2D)
        or hexahedron elements (3D).

    ATTENTION.2:
        As it is, this API is not generic, for any Ax = b. Linear systems must
        be associated to FEM homogenization problems, for Potential or
        Elasticity, both 2D or 3D.
*/

#include "../includes.h"

#ifndef CUDAPCG_KERNELS_WRAPPERS_H_INCLUDED
#define CUDAPCG_KERNELS_WRAPPERS_H_INCLUDED

// Auxiliary API functions
void setParallelAssemblyFlag(cudapcgFlag_t flag);
cudapcgFlag_t getParallelAssemblyFlag();

void allocDotProdArrs(unsigned int dim);
void freeDotProdArrs();

void allocLocalK(unsigned long int size);
void setLocalK(cudapcgVar_t * lclK, unsigned long int size);
void freeLocalK();

//-------------------------------
/////////////////////////////////
//////////// SINGLE /////////////
//////////// STREAM /////////////
/////////////////////////////////
//-------------------------------

void arrcpy(cudapcgVar_t * res, cudapcgVar_t * v, unsigned int dim);
void zeros(cudapcgVar_t * v, unsigned int dim);
cudapcgVar_t dotprod(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim);
void termbytermmul(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res);
void termbytermdiv(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res);
void termbyterminv(cudapcgVar_t * v, unsigned int dim);
void sumVec(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgVar_t * res);
void sumVecIntoFirst(cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim);
void interpl2(cudapcgVar_t *res, cudapcgVar_t *v, unsigned int dim_x, unsigned int dim_y, unsigned int dim_z, unsigned int stride);

void allocPreConditioner(cudapcgModel_t *m);
void freePreConditioner();
void assemblePreConditioner_thermal_2D(cudapcgModel_t *m);
void assemblePreConditioner_thermal_3D(cudapcgModel_t *m);
void assemblePreConditioner_elastic_2D(cudapcgModel_t *m);
void assemblePreConditioner_elastic_3D(cudapcgModel_t *m);

// res = M^-1 * v1 + scl*v2
void applyPreConditioner_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res);
void applyPreConditioner_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res);
void applyPreConditioner_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res);
void applyPreConditioner_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res);

// res = scl*(A*v) + isIncrement*res
void Aprod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t *res, cudapcgVar_t scl, cudapcgFlag_t isIncrement);
void Aprod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t *res, cudapcgVar_t scl, cudapcgFlag_t isIncrement);
void Aprod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t *res, cudapcgVar_t scl, cudapcgFlag_t isIncrement);
void Aprod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t *res, cudapcgVar_t scl, cudapcgFlag_t isIncrement);

#endif // CUDAPCG_KERNELS_WRAPPERS_H_INCLUDED
