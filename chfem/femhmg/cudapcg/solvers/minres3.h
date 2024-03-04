
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_MINRES3_H_INCLUDED
#define CUDAPCG_MINRES3_H_INCLUDED

cudapcgFlag_t setX0_minres3(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_minres3(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_minres3(cudapcgSolver_t *solver);
cudapcgFlag_t solve_minres3(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_MINRES3_H_INCLUDED
