#include "../includes.h"

#ifndef FEMHMG_THERMAL_3D_H_INCLUDED
#define FEMHMG_THERMAL_3D_H_INCLUDED

logical initModel_thermal_3D(hmgModel_t *model);
void assembleLocalMtxs_thermal_3D(hmgModel_t *model);
void assembleRHS_thermal_3D(hmgModel_t *model);
void updateC_thermal_3D(hmgModel_t *model, cudapcgVar_t * T);
void printC_thermal_3D(hmgModel_t *model, char *dest);
void saveFields_thermal_3D(hmgModel_t *model, cudapcgVar_t * T);

void assembleRHS_thermal_3D_ScalarDensityField(hmgModel_t *model);
void updateC_thermal_3D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * T);
void saveFields_thermal_3D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * T);

#endif // FEMHMG_THERMAL_3D_H_INCLUDED
