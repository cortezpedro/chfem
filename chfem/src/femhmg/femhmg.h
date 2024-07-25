#include "includes.h"

#ifndef FEMHMG_H_INCLUDED
#define FEMHMG_H_INCLUDED

logical hmgInit(char * data_filename, char * elem_filename, char * sdf_filename, uint8_t* data);
logical hmgEnd();

void hmgSetXReduceFlag(cudapcgFlag_t flag);
void hmgSetParallelStrategyFlag(cudapcgFlag_t flag);
void hmgSetSolverFlag(cudapcgFlag_t flag);
void hmgSetPreConditionerFlag(cudapcgFlag_t flag);
void hmgSetStoppingCriteria(cudapcgFlag_t flag);
logical hmgSetPoreMappingStrategy(cudapcgFlag_t flag);

void hmgFindInitialGuesses(unsigned int nlevels);
void hmgExportX(logical flag);
void hmgImportX(char *file);

void hmgSetHomogenizationFlag(hmgFlag_t flag);
void hmgSolveHomogenization();

void hmgPrintModelData();
void hmgPrintConstitutiveMtx();

var *hmgGetConstitutiveMtx();
unsigned int hmgGetConstitutiveMtxDim();
var *hmgGetThermalExpansion();
unsigned int hmgGetThermalExpansionDim();

logical hmgSaveConstitutiveMtx(const char * filename);

void hmgSaveFields(logical mustExport_flag, logical byElems_flag);

void hmgKeepTrackOfAnalysisReport(reportFlag_t flag);
reportFlag_t hmgPrintReport(const char *filename);

#endif // FEMHMG_H_INCLUDED
