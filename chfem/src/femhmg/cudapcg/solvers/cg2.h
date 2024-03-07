
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_CG2_H_INCLUDED
#define CUDAPCG_CG2_H_INCLUDED

cudapcgFlag_t setX0_cg2(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_cg2(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_cg2(cudapcgSolver_t *solver);
cudapcgFlag_t solve_cg2(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_CG2_H_INCLUDED
