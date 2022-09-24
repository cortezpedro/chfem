
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_RD_H_INCLUDED
#define CUDAPCG_RD_H_INCLUDED

cudapcgFlag_t setX0_RD(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_RD(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_RD(cudapcgSolver_t *solver);
cudapcgFlag_t solve_RD(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_RD_H_INCLUDED
