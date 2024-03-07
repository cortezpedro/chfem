
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_CG_H_INCLUDED
#define CUDAPCG_CG_H_INCLUDED

cudapcgFlag_t setX0_cg(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_cg(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_cg(cudapcgSolver_t *solver);
cudapcgFlag_t solve_cg(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_CG_H_INCLUDED
