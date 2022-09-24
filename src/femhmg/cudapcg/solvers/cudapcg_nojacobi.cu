/*

*/

#include "cudapcg_nojacobi.h"
#include "../kernels/cudapcg_kernels_wrappers.h"

//---------------------------------
///////////////////////////////////
//////// PUBLIC FUNCTIONS /////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
cudapcgFlag_t setX0_nojacobi(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate){
  return setX0_default(solver,x0,mustInterpolate);
}
//------------------------------------------------------------------------------
cudapcgFlag_t allocDeviceArrays_nojacobi(cudapcgSolver_t *solver){
  return allocDeviceArrays_default(solver);
}
//------------------------------------------------------------------------------
cudapcgFlag_t freeDeviceArrays_nojacobi(cudapcgSolver_t *solver){
  return freeDeviceArrays_default(solver);
}
//------------------------------------------------------------------------------
cudapcgFlag_t solve_nojacobi(cudapcgSolver_t *solver, cudapcgVar_t *res_x){
    if (solver == NULL)
        return CUDAPCG_FALSE;

    solver->count++;

    cudapcgVar_t *x = solver->x;
    cudapcgVar_t *r = solver->r;
    cudapcgVar_t *d = solver->d;
    cudapcgVar_t *q = solver->q;

    unsigned int n = solver->model->nvars;
    unsigned int n_stopping_criteria = solver->model->nhmgvars;

    #ifdef CUDAPCG_TRACK_STOPCRIT
    cudapcgVar_t *stopcrit_metrics = (double *)malloc(sizeof(double)*(solver->max_iterations+1));
    #endif

    cudapcgModel_t *model = solver->model;

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    float time, mean_time=0.0;
    HANDLE_ERROR(cudaEventRecord(start,0));

    if (!solver->x0_hasBeenSet_flag)
      zeros(x,n);
    zeros(q,n);

    double a, delta, delta_0, delta_old, stop_metric, res_0;
    unsigned char mustContinueIterating = 1;

    solver->iteration = 0;

    // residual parameters for x0=[0]
    delta_0 = dotprod(r,r,n);                      // delta = r*r

    if (solver->resnorm_flag == CUDAPCG_INF_NORM)
      res_0 = (double) absmax(r,n);
    else
      res_0 = delta_0;

    // Check if x0=[0] has already satisfied an absolute tolerance
    // This is a safety check. As we perform dimensionless residual evaluation, with
    // respect to delta_0, numerical trouble might occur if this value is too small.
    if (abs_double(res_0) < 0.000000000001){ // 1e-12
        solver->residual = 0.0;
        // Copy result back to cpu
        HANDLE_ERROR(cudaMemcpy(res_x,x,n*sizeof(cudapcgVar_t),cudaMemcpyDeviceToHost));
        printf("%sNull solution satisfied PCG.\n",solver->header_str);
        return CUDAPCG_TRUE;
    }

    // check if an initial guess was provided
    if (solver->x0_hasBeenSet_flag){
      // recalculate resiudals considering initial guess
      solver->Aprod(model,x,1.0,CUDAPCG_FALSE,q);    // q = A*x
      saxpy_iny(r,q,-1.0,n);                         // r += -q
      //arrcpy(d,r,n);
      delta = dotprod(r,r,n);                        // delta = r*r
      // Check if initial guess has already satisfied dimensionless tolerance
      if (!isResidualAboveTol(delta,delta_0,solver->num_tol)){
          solver->residual = evalResidual(delta,delta_0);
          // Copy result back to cpu
          HANDLE_ERROR(cudaMemcpy(res_x,x,n*sizeof(cudapcgVar_t),cudaMemcpyDeviceToHost));
          printf("%sInitial guess satisfied PCG.\n",solver->header_str);
          return CUDAPCG_TRUE;
      }
    } else {
      // assume that x0=[0] is initial guess
      delta = delta_0;
    }
    delta_old = delta;

    #ifdef CUDAPCG_TRACK_STOPCRIT
    switch (solver->resnorm_flag){
      case CUDAPCG_L2_NORM:
        stop_metric = evalResidual(delta,delta_0);
        break;
      case CUDAPCG_INF_NORM:
        stop_metric = (double) absmax(r,n)/res_0;
        break;
      case CUDAPCG_ERROR_NORM:
        stop_metric = 1.0;
        break;
    }
    stopcrit_metrics[solver->iteration] = stop_metric;
    #endif

    // First iteration outside of while loop
    solver->iteration++;
    arrcpy(d,r,n);
    solver->Aprod(model,d,1.0,CUDAPCG_FALSE,q);    // q = A*d
    a = delta / dotprod(d,q,n);                    // a = delta/(d*q)
    saxpy_iny(x,d,a,n);                            // x += a*d
    saxpy_iny(r,q,-a,n);                           // r += -a*q
    delta = dotprod(r,r,n);                        // delta = r*r

    switch (solver->resnorm_flag){
      case CUDAPCG_L2_NORM:
        stop_metric = evalResidual(delta,delta_0);
        mustContinueIterating = isResidualAboveTol(delta,delta_0,solver->num_tol);
        break;
      case CUDAPCG_INF_NORM:
        stop_metric = ((double) absmax(r,n))/res_0;
        mustContinueIterating = stop_metric > solver->num_tol;
        break;
      case CUDAPCG_ERROR_NORM:
        stop_metric = abs_double(a)*((double)absmax(d,n_stopping_criteria))/((double)absmax(x,n_stopping_criteria));
        mustContinueIterating = !((stop_metric <= solver->num_tol) || (!isResidualAboveTol(delta,delta_0,solver->num_tol)));
        break;
    }
    #ifdef CUDAPCG_TRACK_STOPCRIT
    stopcrit_metrics[solver->iteration] = stop_metric;
    #endif

    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&time,start,stop));
    mean_time = ((solver->iteration-1)*mean_time + time) / solver->iteration;

    #ifndef CUDAPCG_QUIET_ITERATIONS
      if (solver->resnorm_flag != CUDAPCG_ERROR_NORM)
        printf("\r%siteration: %d, residual: %.3e",solver->header_str,solver->iteration,stop_metric);
      else
        printf("\r%siteration: %d, residual: %.3e, error: %.3e",solver->header_str,solver->iteration,evalResidual(delta,delta_0),stop_metric);
    #endif

    while (mustContinueIterating && solver->iteration < solver->max_iterations){
        HANDLE_ERROR(cudaEventRecord(start,0));

        solver->iteration++;
        saxpy(r,d,delta/delta_old,n,d);               // d = s+(delta/delta_old)*d
        solver->Aprod(model,d,1.0,CUDAPCG_FALSE,q);   // q = A*d
        a = delta / dotprod(d,q,n);                   // a = delta/(d*q)
        saxpy_iny(x,d,a,n);                           // x += a*d
        saxpy_iny(r,q,-a,n);                          // r += -a*q
        delta_old = delta;
        delta = dotprod(r,r,n);                       // delta = r*r

        switch (solver->resnorm_flag){
          case CUDAPCG_L2_NORM:
            stop_metric = evalResidual(delta,delta_0);
            mustContinueIterating = isResidualAboveTol(delta,delta_0,solver->num_tol);
            break;
          case CUDAPCG_INF_NORM:
            stop_metric = ((double) absmax(r,n))/res_0;
            mustContinueIterating = stop_metric > solver->num_tol;
            break;
          case CUDAPCG_ERROR_NORM:
            stop_metric = abs_double(a)*((double)absmax(d,n_stopping_criteria))/((double)absmax(x,n_stopping_criteria));
            mustContinueIterating = !((stop_metric <= solver->num_tol) || (!isResidualAboveTol(delta,delta_0,solver->num_tol)));
            break;
        }
        #ifdef CUDAPCG_TRACK_STOPCRIT
        stopcrit_metrics[solver->iteration] = stop_metric;
        #endif

        HANDLE_ERROR(cudaEventRecord(stop,0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        HANDLE_ERROR(cudaEventElapsedTime(&time,start,stop));
        mean_time = ((solver->iteration-1)*mean_time + time) / solver->iteration;

        #ifndef CUDAPCG_QUIET_ITERATIONS
          if (solver->resnorm_flag != CUDAPCG_ERROR_NORM)
            printf("\r%siteration: %d, residual: %.3e",solver->header_str,solver->iteration,stop_metric);
          else
            printf("\r%siteration: %d, residual: %.3e, error: %.3e",solver->header_str,solver->iteration,evalResidual(delta,delta_0),stop_metric);
        #endif
    }

    #ifdef CUDAPCG_QUIET_ITERATIONS
      if (solver->resnorm_flag != CUDAPCG_ERROR_NORM)
        printf("%siteration: %d, residual: %.3e",solver->header_str,solver->iteration,stop_metric);
      else
        printf("%siteration: %d, residual: %.3e, error: %.3e",solver->header_str,solver->iteration,evalResidual(delta,delta_0),stop_metric);
    #endif
    printf("\n");

    // Evaluate dimensionless residual
    if (solver->resnorm_flag == CUDAPCG_L2_NORM){
      solver->residual = evalResidual(delta,delta_0);
      solver->foundSolution_flag = !isResidualAboveTol(delta,delta_0,solver->num_tol);
    } else {
      solver->residual = stop_metric;
      solver->foundSolution_flag = solver->residual <= solver->num_tol;
    }

    // Copy result back to cpu
    HANDLE_ERROR(cudaMemcpy(res_x,x,n*sizeof(cudapcgVar_t),cudaMemcpyDeviceToHost));

    solver->mean_time_per_iteration = mean_time*0.001; // value is in ms

    #ifdef CUDAPCG_TRACK_STOPCRIT
    FILE * file = fopen("stopcrit_metrics.bin","wb");
    if (file)
      fwrite(stopcrit_metrics,sizeof(double)*(solver->iteration+1),1,file);
    fclose(file);
    free(stopcrit_metrics);
    #endif

    return solver->foundSolution_flag;
}
//------------------------------------------------------------------------------
