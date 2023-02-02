/*
	=====================================================================
  Universidade Federal Fluminense (UFF) - Niteroi, Brazil
  Institute of Computing
  Authors: Cortez Lopes, P., Vianna, R., Sapucaia, V., Pereira., A.
  contact: pedrocortez@id.uff.br

  Image-based Computational Homogenization with the FEM in GPU.
             (C)           (H)                     (FEM)  (GPU)
  [chfem_gpu]
  
  History:
    * v0.0 (jul/2020) [ALL]    -> OpenMp, parallelization of CPU code
    * v1.0 (nov/2020) [CORTEZ] -> CUDA, PCG in GPU
    * v1.1 (sep/2022) [CORTEZ] -> Added permeability, MINRES.
                                  atomicAdd for EBE.
                                  refactoring of kernels for readability.

  API for the handling of FEM models used in material homogenization.
  Makes use of cudapcg.h to solve linear systems of equations.
  
  THERMAL CONDUCTIVITY, LINEAR ELASTICITY, ABSOLUTE PERMEABILITY.

  =====================================================================
*/

#include "includes.h"

#ifndef FEMHMG_H_INCLUDED
#define FEMHMG_H_INCLUDED

logical hmgInit(char * data_filename, char * elem_filename, char *sdf_filename);
logical hmgEnd();

void hmgSetParallelStrategyFlag(cudapcgFlag_t flag);
void hmgSetSolverFlag(cudapcgFlag_t flag);
void hmgSetStoppingCriteria(cudapcgFlag_t flag);
logical hmgSetPoreMappingStrategy(cudapcgFlag_t flag);

void hmgFindInitialGuesses(unsigned int nlevels);
void hmgExportX(logical flag);
void hmgImportX(char *file);

void hmgSetHomogenizationFlag(hmgFlag_t flag);
void hmgSolveHomogenization();

void hmgPrintModelData();
var *hmgGetConstitutiveMtx();
void hmgPrintConstitutiveMtx();
logical hmgSaveConstitutiveMtx(const char * filename);

void hmgSaveFields(logical mustExport_flag, logical byElems_flag);

void hmgKeepTrackOfAnalysisReport(reportFlag_t flag);
reportFlag_t hmgPrintReport(const char *filename);

#endif // FEMHMG_H_INCLUDED
