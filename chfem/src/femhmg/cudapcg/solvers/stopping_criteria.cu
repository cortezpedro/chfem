/*
*/

#include "stopping_criteria.h"

//------------------------------------------------------------------------------
cudapcgFlag_t isResidualAboveTol(double delta, double delta_0, cudapcgTol_t num_tol){
    double tol = (double) num_tol; // for safety
    return (abs_double(delta) > tol*tol*abs_double(delta_0));
}
//------------------------------------------------------------------------------
double evalResidual(double delta, double delta_0){
    return sqrt_double(abs_double(delta)/abs_double(delta_0));
}
//------------------------------------------------------------------------------
