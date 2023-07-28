/*

*/

#include "../error_handling.h"
#include "stopping_criteria.h"

#ifndef INCLUDES_CUDAPCG_SOLVERS_H_INCLUDED
#define INCLUDES_CUDAPCG_SOLVERS_H_INCLUDED

typedef struct _pcgsolver{

    unsigned long int count; // counts calls to solver->solve()

    char *header_str;

    cudapcgModel_t *model;

    // gpu arrays
    cudapcgVar_t *x;
    cudapcgVar_t *r;
    cudapcgVar_t *d;
    cudapcgVar_t *q;
    cudapcgVar_t *s; // used in minres

    cudapcgFlag_t userAllocatedArrays_flag;
    cudapcgFlag_t x0_hasBeenSet_flag;
    
    cudapcgFlag_t preconditioner_flag;

    unsigned int analysis_flag;
    unsigned int parallelStrategy_flag;
    unsigned int solver_flag;

    cudapcgFlag_t resnorm_flag;

    cudapcgTol_t num_tol;
    unsigned int max_iterations;

    cudapcgFlag_t xreduce_flag;
    unsigned int xreduce_shift;
    double reduce_stab_factor;
    double xreduce_scale;

    cudapcgVar_t residual;
    unsigned int iteration;
    double total_time;
    double mean_time_per_iteration;
    cudapcgFlag_t foundSolution_flag;

    cudapcgFlag_t mustAssemblePreConditioner;
    void (*assemblePreConditioner)(cudapcgModel_t *);
    void (*applyPreConditioner)(cudapcgModel_t * , cudapcgVar_t *, cudapcgVar_t *, cudapcgVar_t, cudapcgVar_t, cudapcgVar_t *);
    void (*applyinvPreConditioner)(cudapcgModel_t * , cudapcgVar_t *, cudapcgVar_t *, cudapcgVar_t, cudapcgVar_t, cudapcgVar_t *);
    void (*Aprod)(cudapcgModel_t * , cudapcgVar_t *, cudapcgVar_t, cudapcgVar_t , cudapcgVar_t *);
    void (*PreConditionerAprod)(cudapcgModel_t * , cudapcgVar_t *, cudapcgVar_t, cudapcgVar_t , cudapcgVar_t *);
    double (*dotPreConditioner)(cudapcgModel_t * , cudapcgVar_t *, cudapcgVar_t *, cudapcgVar_t);
    double (*dotinvPreConditioner)(cudapcgModel_t * , cudapcgVar_t *, cudapcgVar_t *, cudapcgVar_t);
    double (*dotAprod)(cudapcgModel_t * , cudapcgVar_t *, cudapcgVar_t);
    double (*dotA2prod)(cudapcgModel_t * , cudapcgVar_t *, cudapcgVar_t);
    double (*dotPreConditionerA2prod)(cudapcgModel_t * , cudapcgVar_t *, cudapcgVar_t);

    cudapcgFlag_t (*solve)(struct _pcgsolver *, cudapcgVar_t *);
    cudapcgFlag_t (*setX0)(struct _pcgsolver *, cudapcgVar_t *, cudapcgFlag_t);
    cudapcgFlag_t (*allocDeviceArrays)(struct _pcgsolver *);
    cudapcgFlag_t (*freeDeviceArrays)(struct _pcgsolver *);

} cudapcgSolver_t;

#endif // INCLUDES_CUDAPCG_SOLVERS_H_INCLUDED
