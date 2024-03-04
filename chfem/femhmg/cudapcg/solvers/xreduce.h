#include "../includes.h"
#include "solvers.h"

#ifndef CUDAPCG_XREDUCE_H_INCLUDED
#define CUDAPCG_XREDUCE_H_INCLUDED

void update_xreduce(cudapcgModel_t *m, unsigned int it, cudapcgFlag_t flag, unsigned int shift, double stab, double a, cudapcgVar_t *d, cudapcgVar_t *x, double **res_per_it);
void save_xreduce(char *filename, cudapcgModel_t *m, unsigned int it, cudapcgFlag_t flag, unsigned int shift, double scl, double **res_per_it);

//void update_xreduce(cudapcgModel_t *m, unsigned int it, cudapcgFlag_t flag, unsigned int shift, double stab, double a, cudapcgVar_t *d, cudapcgVar_t *x, double **res_per_it, double **norms);

#endif // CUDAPCG_XREDUCE_H_INCLUDED
