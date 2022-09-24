
#include "../includes.h"
#include "cudapcg_default.h"
#include "solvers.h"

#ifndef CUDAPCG_NOJACOBI_H_INCLUDED
#define CUDAPCG_NOJACOBI_H_INCLUDED

cudapcgFlag_t setX0_nojacobi(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_nojacobi(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_nojacobi(cudapcgSolver_t *solver);
cudapcgFlag_t solve_nojacobi(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_NOJACOBI_H_INCLUDED
