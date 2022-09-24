/*
	Universidade Federal Fluminense (UFF) - Niteroi, Brazil
	Institute of Computing
	Authors: Cortez, P., Vianna, R., Sapucaia, V., Pereira., A.
	History: 
		* v0.0 (jul/2020) [ALL]    -> OpenMp, parallelization of CPU code
		* v1.0 (nov/2020) [CORTEZ] -> CUDA, PCG on GPU. (substituted parpcg.h with cudapcg.h)

	API for the handling of FEM models used in
    material homogenization.
    Makes use of cudapcg.h to solve linear system
    of equations.

    ATTENTION.1:
        Considers a structured regular mesh of quad elements (2D)
        or hexahedron elements (3D).

    ATTENTION.2:
        As it is, works for Potential or Elasticity, both 2D or 3D.
*/

#include "../includes.h"

#ifndef FEMHMG_THERMAL_2D_H_INCLUDED
#define FEMHMG_THERMAL_2D_H_INCLUDED

logical initModel_thermal_2D(hmgModel_t *model);
void assembleLocalMtxs_thermal_2D(hmgModel_t *model);
void assembleRHS_thermal_2D(hmgModel_t *model);
void updateC_thermal_2D(hmgModel_t *model, cudapcgVar_t * T);
void printC_thermal_2D(hmgModel_t *model, char *dest);

#endif // FEMHMG_THERMAL_2D_H_INCLUDED
