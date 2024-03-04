
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_PMINRES2_H_INCLUDED
#define CUDAPCG_PMINRES2_H_INCLUDED

cudapcgFlag_t setX0_pminres2(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_pminres2(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_pminres2(cudapcgSolver_t *solver);
cudapcgFlag_t solve_pminres2(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_PMINRES2_H_INCLUDED
