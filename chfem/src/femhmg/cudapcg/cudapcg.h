/*
  =====================================================================
  Universidade Federal Fluminense (UFF) - Niteroi, Brazil
  Institute of Computing
  Authors: Cortez Lopes, P., Pereira., A.
  contact: pedrocortez@id.uff.br
  =====================================================================
*/

#include "includes.h"

#ifndef CUDAPCG_H_INCLUDED
#define CUDAPCG_H_INCLUDED

#ifdef __cplusplus // nvcc defaults .cu files to c++
extern "C" {
#endif

  cudapcgFlag_t cudapcgInit(cudapcgFlag_t analysis_flag, cudapcgFlag_t parStrategy_flag);
  cudapcgFlag_t cudapcgEnd();

  cudapcgFlag_t cudapcgSetModelConstructorFcn(cudapcgFlag_t (*fcn)(cudapcgModel_t **, const void *));
  cudapcgFlag_t cudapcgBuildModel(const void *data);

  cudapcgModel_t * cudapcgNewModel(void);
  cudapcgFlag_t cudapcgSetModel(cudapcgModel_t *model);

  cudapcgFlag_t cudapcgSetNumTol(cudapcgTol_t t);
  cudapcgFlag_t cudapcgSetMaxIterations(unsigned int n);

  cudapcgFlag_t cudapcgSetXReduceMode(cudapcgFlag_t flag);
  cudapcgFlag_t cudapcgSetXReduceShift(unsigned int shift);
  cudapcgFlag_t cudapcgSetXReduceScale(double scl);
  cudapcgFlag_t cudapcgSetReduceStabFactor(double scl);

  cudapcgFlag_t cudapcgSetSolver(cudapcgFlag_t flag);
  cudapcgFlag_t cudapcgSetPreconditioner(cudapcgFlag_t flag);
  cudapcgFlag_t cudapcgSetResNorm(cudapcgFlag_t flag);
  
  cudapcgFlag_t cudapcgAllocateArrays();
  cudapcgFlag_t cudapcgFreeArrays();

  cudapcgFlag_t cudapcgSetRHS(cudapcgVar_t * RHS);
  cudapcgFlag_t cudapcgSetX0(cudapcgVar_t * x0, cudapcgFlag_t mustInterpolate);
  cudapcgFlag_t cudapcgSetImage(cudapcgMap_t *img);
  cudapcgFlag_t cudapcgSetParametricDensityField(parametricScalarField_t *field, double fmin, double fmax);
  cudapcgFlag_t cudapcgSetPoreMap(cudapcgFlag_t *pores);
  cudapcgFlag_t cudapcgSetPeriodic2DOFMap(cudapcgIdMap_t *vars);
  cudapcgFlag_t cudapcgSetDOF2PeriodicMap(cudapcgIdMap_t *nodes);
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
