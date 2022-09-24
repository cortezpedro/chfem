
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_DEFAULT_H_INCLUDED
#define CUDAPCG_DEFAULT_H_INCLUDED

cudapcgFlag_t setX0_default(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_default(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_default(cudapcgSolver_t *solver);
cudapcgFlag_t solve_default(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_DEFAULT_H_INCLUDED
