/*
*/

#include "stopping_criteria.h"

//------------------------------------------------------------------------------
cudapcgFlag_t isResidualAboveTol(cudapcgVar_t delta, cudapcgVar_t delta_0, cudapcgTol_t num_tol){
    return (ABS(delta) > num_tol*num_tol*ABS(delta_0));
}
//------------------------------------------------------------------------------
cudapcgVar_t evalResidual(cudapcgVar_t delta, cudapcgVar_t delta_0){
    return SQRT(ABS(delta)/ABS(delta_0));
}
//------------------------------------------------------------------------------
