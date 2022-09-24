#include "../includes.h"

#ifndef FEMHMG_FLUID_2D_H_INCLUDED
#define FEMHMG_FLUID_2D_H_INCLUDED

logical initModel_fluid_2D(hmgModel_t *model);
void assembleLocalMtxs_fluid_2D(hmgModel_t *model);
void assembleRHS_fluid_2D(hmgModel_t *model);
void updateC_fluid_2D(hmgModel_t *model, cudapcgVar_t * V);
void printC_fluid_2D(hmgModel_t *model, char *dest);
void saveFields_fluid_2D(hmgModel_t *model, cudapcgVar_t * V);

void assembleRHS_PressureGrad_fluid_2D(hmgModel_t *model);
void laplace2stokes_fluid_2D(hmgModel_t *model);

#endif // FEMHMG_FLUID_2D_H_INCLUDED
