/*

*/

#include "minres2.h"
#include "xreduce.h"
#include "../kernels/wrappers.h"

//---------------------------------
///////////////////////////////////
//////// PUBLIC FUNCTIONS /////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
cudapcgFlag_t setX0_minres2(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate){
  size_t var_sz = sizeof(cudapcgVar_t)*((size_t)solver->model->nvars);
  if (solver->d == NULL) HANDLE_ERROR(cudaMalloc(&solver->d,var_sz));
  if (mustInterpolate){
      unsigned int nodal_dofs = solver->model->nvars/solver->model->nelem;
      size_t coarse_var_sz = sizeof(cudapcgVar_t)*nodal_dofs*((solver->model->ncols)/2)*((solver->model->nrows)/2)*((solver->model->nlayers)/2+(solver->model->nlayers<2));
      cudapcgVar_t *temp=NULL;
      HANDLE_ERROR(cudaMalloc(&temp,coarse_var_sz));
      HANDLE_ERROR(cudaMemcpy(temp,x0,coarse_var_sz,cudaMemcpyHostToDevice));
      interpl2(temp,solver->model->nrows,solver->model->ncols,solver->model->nlayers,solver->model->nvars/solver->model->nelem,solver->d);
      HANDLE_ERROR(cudaFree(temp));
  } else {
      HANDLE_ERROR(cudaMemcpy(solver->d,x0,var_sz,cudaMemcpyHostToDevice));
  }
  solver->x0_hasBeenSet_flag = CUDAPCG_TRUE;
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t allocDeviceArrays_minres2(cudapcgSolver_t *solver){
  size_t sz = sizeof(cudapcgVar_t)*solver->model->nvars;
  if (!solver->x0_hasBeenSet_flag) HANDLE_ERROR(cudaMalloc(&solver->d,sz));
  // allocate arrays that will be used to store dotprod kernel results (within cudapcg_kernels.h)
  allocDotProdArrs(solver->model->nvars);
  solver->userAllocatedArrays_flag = CUDAPCG_TRUE;
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t freeDeviceArrays_minres2(cudapcgSolver_t *solver){
  if (!(solver->x0_hasBeenSet_flag)){
    if (solver->d!=NULL) HANDLE_ERROR(cudaFree(solver->d));
    solver->d = NULL;
  }
  freeDotProdArrs();
  solver->userAllocatedArrays_flag = CUDAPCG_FALSE;
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t solve_minres2(cudapcgSolver_t *solver, cudapcgVar_t *res_x){
    if (solver == NULL)
        return CUDAPCG_FALSE;

    solver->count++;

    cudapcgVar_t *r = solver->r;
    cudapcgVar_t *d = solver->d;

    unsigned int n = solver->model->nvars;
    unsigned int n_stopping_criteria = solver->model->nhmgvars;

    #ifdef CUDAPCG_TRACK_STOPCRIT
    cudapcgVar_t *stopcrit_metrics = (double *)malloc(sizeof(double)*(solver->max_iterations+1));
    #endif
    
    cudapcgModel_t *model = solver->model;

    double **res_per_it = (double **)malloc(sizeof(double*)*model->nvarspernode);
    res_per_it[0] = (double *)malloc(sizeof(double)*(solver->max_iterations+1)*model->nvarspernode);
    for (int ii=1; ii<model->nvarspernode; ii++) res_per_it[ii] = &(res_per_it[0][ii*(solver->max_iterations+1)]);

    // double **norms = NULL;
    // if (solver->xreduce_flag == CUDAPCG_XREDUCE_NONE){
    //   norms = (double **)malloc(sizeof(double *)*4);
    //   norms[0] = (double *)malloc(sizeof(double)*(solver->max_iterations+1)*4);
    //   norms[1] = &(norms[0][1*(solver->max_iterations+1)]);
    //   norms[2] = &(norms[0][2*(solver->max_iterations+1)]);
    //   norms[3] = &(norms[0][3*(solver->max_iterations+1)]);
    //   norms[0][0]=0.0;
    //   norms[1][0]=0.0;
    //   norms[2][0]=0.0;
    //   norms[3][0]=0.0;
    // }

    cudapcgVar_t *x=NULL;
    if (solver->xreduce_flag == CUDAPCG_XREDUCE_ONLYDIR){
      HANDLE_ERROR(cudaMalloc(&x,sizeof(cudapcgVar_t)*model->nhmgvars/model->nvarspernode));
      zeros(x,model->nhmgvars/model->nvarspernode);
    } else if (solver->xreduce_flag == CUDAPCG_XREDUCE_FULL){
      HANDLE_ERROR(cudaMalloc(&x,sizeof(cudapcgVar_t)*model->nhmgvars));
      zeros(x,model->nhmgvars);
    }

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    float time, mean_time=0.0;
    HANDLE_ERROR(cudaEventRecord(start,0));

    double a, b, delta, delta_0, delta_old, stop_metric, res_0;
    unsigned char mustContinueIterating = 1;

    solver->iteration = 0;

    // Init resulting effective fields
    for (int ii=0; ii<model->nvarspernode; ii++) res_per_it[ii][0]=0.0;

    // residual parameters for x0=[0]
    delta_0 = solver->dotAprod(model,r,1.0);  // delta = dot(r,A*r)

    if (solver->resnorm_flag == CUDAPCG_INF_NORM)
      res_0 = (double) absmax(r,n);//(scl_factor*scl_factor)*((double) absmax(r,n));
    else
      res_0 = dotprod(r,r,n);//(scl_factor*scl_factor)*dotprod(r,r,n); //delta_0;

    // Check if x0=[0] has already satisfied an absolute tolerance
    // This is a safety check. As we perform dimensionless residual evaluation, with
    // respect to delta_0, numerical trouble might occur if this value is too small.
    if (abs_double(res_0) < 0.000000000001){ // 1e-12
        solver->residual = 0.0;
        printf("%sNull solution satisfied MINRES.\n",solver->header_str);
        return CUDAPCG_TRUE;
    }

    // check if an initial guess was provided
    if (solver->x0_hasBeenSet_flag){
      // recalculate resiudals considering initial guess
      solver->Aprod(model,d,-1.0,1.0,r);             // r += -1.0*A*x (d is being used to store x0)
      delta = solver->dotAprod(model,r,1.0);         // delta = dot(r,A*r)
      switch (solver->xreduce_flag){
        case CUDAPCG_XREDUCE_ONLYDIR:
          axpy_iny_with_stride(x,d,1.0,model->nhmgvars,model->nvarspernode,solver->xreduce_shift);
          break;
        case CUDAPCG_XREDUCE_FULL:
          arrcpy(d,model->nhmgvars,x);
          break;
      }
      // update effective fields
      for (int ii=0; ii<model->nvarspernode; ii++) res_per_it[ii][0]=reduce_with_stride(d,model->nhmgvars,model->nvarspernode,ii);
      // Check if initial guess has already satisfied dimensionless tolerance
      if (!isResidualAboveTol(delta,delta_0,solver->num_tol)){
          solver->residual = evalResidual(delta,delta_0);
          printf("%sInitial guess satisfied MINRES.\n",solver->header_str);
          return CUDAPCG_TRUE;
      }
    } else {
      // assume that x0=[0] is initial guess
      delta = delta_0;
    }
    delta_old = delta;

    arrcpy(r,n,d);                       // d = r

    #ifdef CUDAPCG_TRACK_STOPCRIT
    switch (solver->resnorm_flag){
      case CUDAPCG_L2_NORM:
        stop_metric = dotprod(r,r,n);
        stop_metric = evalResidual(stop_metric,res_0);
        break;
      case CUDAPCG_INF_NORM:
        stop_metric = ((double) absmax(r,n))/res_0;
        break;
      case CUDAPCG_ERROR_NORM:
        stop_metric = 1.0;
        break;
    }
    stopcrit_metrics[solver->iteration] = stop_metric;
    #endif

    // First iteration outside of while loop
    solver->iteration++;
    a = delta / solver->dotA2prod(model,d,1.0);    // a = delta/(Ad*Ad)
    update_xreduce(model,solver->iteration,solver->xreduce_flag,solver->xreduce_shift,solver->reduce_stab_factor,a,d,x,res_per_it);
    solver->Aprod(model,d,-a,1.0,r);               // r = -a*A*d + r
    delta = solver->dotAprod(model,r,1.0);         // delta = dot(r,A*r)

    switch (solver->resnorm_flag){
      case CUDAPCG_L2_NORM:
        stop_metric = dotprod(r,r,n);
        mustContinueIterating = isResidualAboveTol(stop_metric,res_0,solver->num_tol);
        stop_metric = evalResidual(stop_metric,res_0);
        break;
      case CUDAPCG_INF_NORM:
        stop_metric = ((double) absmax(r,n))/res_0;
        mustContinueIterating = stop_metric > solver->num_tol;
        break;
      case CUDAPCG_ERROR_NORM:
        stop_metric = abs_double(a)*((double)absmax(d,n_stopping_criteria))/1.0;//((double)absmax(x,n_stopping_criteria));
        mustContinueIterating = stop_metric > solver->num_tol;
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
        b = delta/delta_old;
        axpy(r,d,b,n,d);                              // d = r+b*d
        a = delta / solver->dotA2prod(model,d,1.0);   // a = delta/(Ad*Ad)
        update_xreduce(model,solver->iteration,solver->xreduce_flag,solver->xreduce_shift,solver->reduce_stab_factor,a,d,x,res_per_it);
        solver->Aprod(model,d,-a,1.0,r);              // r = -a*A*d + r
        delta_old = delta;
        delta = solver->dotAprod(model,r,1.0);        // delta = dot(r,A*r)

        switch (solver->resnorm_flag){
          case CUDAPCG_L2_NORM:
            stop_metric = dotprod(r,r,n);
            mustContinueIterating = isResidualAboveTol(stop_metric,res_0,solver->num_tol);
            stop_metric = evalResidual(stop_metric,res_0);
            break;
          case CUDAPCG_INF_NORM:
            stop_metric = ((double) absmax(r,n))/res_0;
            mustContinueIterating = stop_metric > solver->num_tol;
            break;
          case CUDAPCG_ERROR_NORM:
            stop_metric = abs_double(a)*((double)absmax(d,n_stopping_criteria))/1.0;//((double)absmax(x,n_stopping_criteria));
            mustContinueIterating = stop_metric > solver->num_tol;
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
    solver->residual = stop_metric;
    solver->foundSolution_flag = solver->residual <= solver->num_tol;

    solver->mean_time_per_iteration = mean_time*0.001; // value is in ms

    char filename_buffer[2048];
    sprintf(filename_buffer,"%s_xreduce_metrics_%lu.bin",model->name,solver->count);
    save_xreduce(&filename_buffer[0],model,solver->iteration,solver->xreduce_flag,solver->xreduce_shift,solver->xreduce_scale,res_per_it);
    
    double *ptr=NULL;
    for (int ii=0; ii<model->nvarspernode; ii++){
      ptr = res_per_it[ii];
      switch (solver->xreduce_flag){
        case CUDAPCG_XREDUCE_NONE:
          for (int jj=solver->iteration; jj>0; jj--) ptr[jj-1]+=ptr[jj];
          res_x[ii] = *ptr;
          break;
        case CUDAPCG_XREDUCE_ONLYDIR:
          res_x[ii] = solver->xreduce_shift == ii ? ptr[solver->iteration] : 0.0;
          break;
        case CUDAPCG_XREDUCE_FULL:
          res_x[ii] = ptr[solver->iteration];
          break;
      }
    }
    free(res_per_it[0]);
    free(res_per_it);

    // FILE * file = NULL;
    // if (norms){
    //   sprintf(filename_buffer,"%s_absmax_metrics_%lu.bin",model->name,solver->count);
    //   file = fopen(filename_buffer,"wb");
    //   if (file) fwrite(norms[0],sizeof(double)*(solver->iteration+1),1,file);
    //   fclose(file);

    //   sprintf(filename_buffer,"%s_l2norm_metrics_%lu.bin",model->name,solver->count);
    //   file = fopen(filename_buffer,"wb");
    //   if (file) fwrite(norms[1],sizeof(double)*(solver->iteration+1),1,file);
    //   fclose(file);

    //   sprintf(filename_buffer,"%s_alpha_metrics_%lu.bin",model->name,solver->count);
    //   file = fopen(filename_buffer,"wb");
    //   if (file) fwrite(norms[2],sizeof(double)*(solver->iteration+1),1,file);
    //   fclose(file);

    //   sprintf(filename_buffer,"%s_dreduce_metrics_%lu.bin",model->name,solver->count);
    //   file = fopen(filename_buffer,"wb");
    //   if (file) fwrite(norms[3],sizeof(double)*(solver->iteration+1),1,file);
    //   fclose(file);

    //   free(norms[0]);
    //   free(norms);
    //   norms=NULL;
    // }

    if (solver->xreduce_flag > CUDAPCG_XREDUCE_NONE) HANDLE_ERROR(cudaFree(x)); x = NULL;

    #ifdef CUDAPCG_TRACK_STOPCRIT
    sprintf(filename_buffer,"%s_stopcrit_metrics_%lu.bin",model->name,solver->count);
    FILE * file = fopen(filename_buffer,"wb");
    if (file)
      fwrite(stopcrit_metrics,sizeof(double)*(solver->iteration+1),1,file);
    fclose(file);
    free(stopcrit_metrics);
    #endif

    return solver->foundSolution_flag;
}
//------------------------------------------------------------------------------
