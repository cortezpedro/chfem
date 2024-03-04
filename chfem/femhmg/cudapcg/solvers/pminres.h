
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_PMINRES_H_INCLUDED
#define CUDAPCG_PMINRES_H_INCLUDED

cudapcgFlag_t setX0_pminres(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_pminres(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_pminres(cudapcgSolver_t *solver);
cudapcgFlag_t solve_pminres(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_PMINRES_H_INCLUDED
