#include "../includes.h"

#ifndef FEMHMG_ELASTIC_3D_H_INCLUDED
#define FEMHMG_ELASTIC_3D_H_INCLUDED

logical initModel_elastic_3D(hmgModel_t *model);
void assembleLocalMtxs_elastic_3D(hmgModel_t *model);
void assembleRHS_elastic_3D(hmgModel_t *model);
void updateC_elastic_3D(hmgModel_t *model, cudapcgVar_t * D);
void printC_elastic_3D(hmgModel_t *model, char *dest);
void saveFields_elastic_3D(hmgModel_t *model, cudapcgVar_t * D);

void assembleRHS_elastic_3D_ScalarDensityField(hmgModel_t *model);
void updateC_elastic_3D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * D);
void saveFields_elastic_3D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * D);

#endif // FEMHMG_ELASTIC_3D_H_INCLUDED
