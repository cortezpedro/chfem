/*
    Universidade Federal Fluminense (UFF) - Niteroi, Brazil
    Institute of Computing
    Author: Cortez, P.
    History: v1.0 (november/2020)

    API for solving linear systems associated to
    FEM models with an assembly-free PCG
    method, using CUDA.
    All global matrix operations involve "assembly on-the-fly"

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

#include "includes.h"

#ifndef CUDAPCG_H_INCLUDED
#define CUDAPCG_H_INCLUDED

#ifdef __cplusplus // nvcc defaults .cu files to c++
extern "C" {
#endif

  cudapcgFlag_t cudapcgInit(cudapcgFlag_t analysis_flag, cudapcgFlag_t parAssembly_flag);
  cudapcgFlag_t cudapcgEnd();

  cudapcgFlag_t cudapcgSetModelConstructorFcn(cudapcgFlag_t (*fcn)(cudapcgModel_t **, const void *));
  cudapcgFlag_t cudapcgBuildModel(const void *data);

  cudapcgModel_t * cudapcgNewModel(void);
  cudapcgFlag_t cudapcgSetModel(cudapcgModel_t *model);

  cudapcgFlag_t cudapcgSetNumTol(cudapcgTol_t t);
  cudapcgFlag_t cudapcgSetMaxIterations(unsigned int n);

  cudapcgFlag_t cudapcgAllocateArrays();
  cudapcgFlag_t cudapcgFreeArrays();

  cudapcgFlag_t cudapcgSetRHS(cudapcgVar_t * RHS);
  cudapcgFlag_t cudapcgSetX0(cudapcgVar_t * x0, cudapcgFlag_t mustInterpolate);
  cudapcgFlag_t cudapcgSetImage(cudapcgMap_t *img);
  cudapcgFlag_t cudapcgSetLclMtxs(cudapcgVar_t * LclMtxs);

  cudapcgFlag_t cudapcgSolve(cudapcgVar_t * x);

  cudapcgFlag_t cudapcgSetHeaderString(char *header);
  cudapcgFlag_t cudapcgPrintSolverReport();
  cudapcgFlag_t cudapcgPrintSolverReport2(char *dest);
  cudapcgFlag_t cudapcgPrintSolverMetrics();
  cudapcgFlag_t cudapcgPrintSolverMetrics2(char *dest);

  unsigned int cudapcgGetNumIterations();
  unsigned int cudapcgGetMaxNumIterations();
  cudapcgVar_t cudapcgGetResidual();

#ifdef __cplusplus
}
#endif

#endif // CUDAPCG_H_INCLUDED
