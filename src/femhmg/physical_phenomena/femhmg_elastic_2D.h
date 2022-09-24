#include "../includes.h"

#ifndef FEMHMG_ELASTIC_2D_H_INCLUDED
#define FEMHMG_ELASTIC_2D_H_INCLUDED

logical initModel_elastic_2D(hmgModel_t *model);
void assembleLocalMtxs_elastic_2D_PlaneStrain(hmgModel_t *model);
void assembleLocalMtxs_elastic_2D_PlaneStress(hmgModel_t *model);
void assembleRHS_elastic_2D(hmgModel_t *model);
void updateC_elastic_2D(hmgModel_t *model, cudapcgVar_t * D);
void printC_elastic_2D(hmgModel_t *model, char *dest);
void saveFields_elastic_2D(hmgModel_t *model, cudapcgVar_t * D);

#endif // FEMHMG_ELASTIC_2D_H_INCLUDED
