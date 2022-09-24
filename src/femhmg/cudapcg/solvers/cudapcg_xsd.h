
#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_XSD_H_INCLUDED
#define CUDAPCG_XSD_H_INCLUDED

cudapcgFlag_t setX0_XSD(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate);
cudapcgFlag_t allocDeviceArrays_XSD(cudapcgSolver_t *solver);
cudapcgFlag_t freeDeviceArrays_XSD(cudapcgSolver_t *solver);
cudapcgFlag_t solve_XSD(cudapcgSolver_t *solver, cudapcgVar_t *res_x);

#endif // CUDAPCG_XSD_H_INCLUDED
