/*

*/

#include "../includes.h"
#include "solvers_utils.h"

#ifndef INCLUDES_CUDAPCG_SOLVERS_STOP_H_INCLUDED
#define INCLUDES_CUDAPCG_SOLVERS_STOP_H_INCLUDED

//------------------------------------------------------------------------------
cudapcgFlag_t isResidualAboveTol(double delta, double delta_0, cudapcgTol_t num_tol);
double evalResidual(double delta, double delta_0);
//------------------------------------------------------------------------------


#endif // INCLUDES_CUDAPCG_SOLVERS_STOP_H_INCLUDED
