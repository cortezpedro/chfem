
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_PCG2_H_INCLUDED
#define CUDAPCG_PCG2_H_INCLUDED

cudapcgFlag_t setX0_pcg2(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_pcg2(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_pcg2(cudapcgSolver_t *solver);
cudapcgFlag_t solve_pcg2(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_PCG2_H_INCLUDED
