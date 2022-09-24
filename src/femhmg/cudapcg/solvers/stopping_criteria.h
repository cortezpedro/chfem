/*

*/

#include "../includes.h"
#include "solvers_utils.h"

#ifndef INCLUDES_CUDAPCG_SOLVERS_STOP_H_INCLUDED
#define INCLUDES_CUDAPCG_SOLVERS_STOP_H_INCLUDED

//------------------------------------------------------------------------------
cudapcgFlag_t isResidualAboveTol(cudapcgVar_t delta, cudapcgVar_t delta_0, cudapcgTol_t num_tol);
cudapcgVar_t evalResidual(cudapcgVar_t delta, cudapcgVar_t delta_0);
//------------------------------------------------------------------------------


#endif // INCLUDES_CUDAPCG_SOLVERS_STOP_H_INCLUDED
