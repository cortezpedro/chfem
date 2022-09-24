/*
	Universidade Federal Fluminense (UFF) - Niteroi, Brazil
	Institute of Computing
	Authors: Cortez, P., Vianna, R., Sapucaia, V., Pereira., A.
	History: 
		* v1.0 (jul/2020) [ALL]    -> OpenMp, parallelization of CPU code
		* v1.1 (nov/2020) [CORTEZ] -> CUDA, PCG on GPU. (substituted parpcg.h with cudapcg.h)

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

#include "includes.h"

#ifndef FEMHMG_H_INCLUDED
#define FEMHMG_H_INCLUDED

logical hmgInit(char * data_filename, char * elem_filename);
logical hmgEnd();

void hmgSetParallelStrategyFlag(cudapcgFlag_t flag);

void hmgSetHomogenizationFlag(hmgFlag_t flag);
void hmgFindInitialGuesses(unsigned int nlevels);
void hmgSolveHomogenization();

void hmgPrintModelData();
void hmgPrintConstitutiveMtx();
logical hmgSaveConstitutiveMtx(const char * filename);

void hmgKeepTrackOfAnalysisReport(reportFlag_t flag);
reportFlag_t hmgPrintReport(const char *filename);

#endif // FEMHMG_H_INCLUDED
