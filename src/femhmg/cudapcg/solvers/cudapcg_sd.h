
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_SD_H_INCLUDED
#define CUDAPCG_SD_H_INCLUDED

cudapcgFlag_t setX0_SD(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_SD(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_SD(cudapcgSolver_t *solver);
cudapcgFlag_t solve_SD(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_SD_H_INCLUDED
