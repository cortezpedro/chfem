
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_MINRES_H_INCLUDED
#define CUDAPCG_MINRES_H_INCLUDED

cudapcgFlag_t setX0_minres(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_minres(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_minres(cudapcgSolver_t *solver);
cudapcgFlag_t solve_minres(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_MINRES_H_INCLUDED
