#include "../includes.h"

#ifndef FEMHMG_FLUID_3D_H_INCLUDED
#define FEMHMG_FLUID_3D_H_INCLUDED

logical initModel_fluid_3D(hmgModel_t *model);
void assembleLocalMtxs_fluid_3D(hmgModel_t *model);
void assembleRHS_fluid_3D(hmgModel_t *model);
void updateC_fluid_3D(hmgModel_t *model, cudapcgVar_t * V);
void printC_fluid_3D(hmgModel_t *model, char *dest);
void saveFields_fluid_3D(hmgModel_t *model, cudapcgVar_t * V);

#endif // FEMHMG_FLUID_3D_H_INCLUDED
