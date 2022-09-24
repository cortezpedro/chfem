
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_XRD_H_INCLUDED
#define CUDAPCG_XRD_H_INCLUDED

cudapcgFlag_t setX0_XRD(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_XRD(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_XRD(cudapcgSolver_t *solver);
cudapcgFlag_t solve_XRD(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_XRD_H_INCLUDED
