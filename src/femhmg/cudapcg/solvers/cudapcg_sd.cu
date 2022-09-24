/*

*/

#include "cudapcg_sd.h"
#include "../kernels/cudapcg_kernels_wrappers.h"

//---------------------------------
///////////////////////////////////
//////// PUBLIC FUNCTIONS /////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
cudapcgFlag_t setX0_SD(cudapcgSolver_t *solver, cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate){
  unsigned int var_sz = sizeof(cudapcgVar_t)*solver->model->nvars;
  if (solver->d == NULL)
      HANDLE_ERROR(cudaMalloc(&solver->d,var_sz));
  if (mustInterpolate){
      unsigned int nodal_dofs = solver->model->nvars/solver->model->nelem;
      unsigned int coarse_var_sz = sizeof(cudapcgVar_t)*nodal_dofs*((solver->model->ncols)/2)*((solver->model->nrows)/2)*((solver->model->nlayers)/2+(solver->model->nlayers<2));
      if (solver->q == NULL)
          HANDLE_ERROR(cudaMalloc(&solver->q,coarse_var_sz));
      HANDLE_ERROR(cudaMemcpy(solver->q,x0,coarse_var_sz,cudaMemcpyHostToDevice));
      interpl2(solver->d,solver->q,solver->model->ncols,solver->model->nrows,solver->model->nlayers,solver->model->nvars/solver->model->nelem);
      if (solver->q!=NULL) HANDLE_ERROR(cudaFree(solver->q)); solver->q = NULL;
  } else {
      HANDLE_ERROR(cudaMemcpy(solver->d,x0,var_sz,cudaMemcpyHostToDevice));
  }
  solver->x0_hasBeenSet_flag = CUDAPCG_TRUE;
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t allocDeviceArrays_SD(cudapcgSolver_t *solver){
  unsigned int sz = sizeof(cudapcgVar_t)*solver->model->nvars;
  if (!solver->x0_hasBeenSet_flag)
      HANDLE_ERROR(cudaMalloc(&solver->d,sz));
  HANDLE_ERROR(cudaHostAlloc(&solver->h_d,sz,cudaHostAllocDefault));
  // allocate arrays that will be used to store dotprod kernel results (within cudapcg_kernels.h)
  allocDotProdArrs(solver->model->nvars);
  solver->userAllocatedArrays_flag = CUDAPCG_TRUE;
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t freeDeviceArrays_SD(cudapcgSolver_t *solver){
  if (!(solver->x0_hasBeenSet_flag)){
      if (solver->d!=NULL) HANDLE_ERROR(cudaFree(solver->d));
      solver->d = NULL;
  }
  if (solver->h_d!=NULL) HANDLE_ERROR(cudaFreeHost(solver->h_d)); solver->h_d = NULL;
  freeDotProdArrs();
  solver->userAllocatedArrays_flag = CUDAPCG_FALSE;
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t solve_SD(cudapcgSolver_t *solver, cudapcgVar_t *res_x){
    if (solver == NULL)
        return CUDAPCG_FALSE;
        
    cudapcgVar_t *s = solver->r;
    cudapcgVar_t *d = solver->d;
    
    unsigned int n = solver->model->nvars;
    
    cudapcgModel_t *model = solver->model;
    
    // Create streams for mem transfer and GPU computing
    cudaStream_t kernels_stream;
    HANDLE_ERROR(cudaStreamCreate(&kernels_stream));
    
    unsigned int nStreams = solver->nstreams;
    unsigned int memtransfer_streamDim[nStreams]; memtransfer_streamDim[0] = CEIL(n,nStreams);
    unsigned int memtransfer_streamSize[nStreams];
    cudaStream_t memtransfer_streams[nStreams];
    for (unsigned int i=0; i<nStreams; i++){
      HANDLE_ERROR(cudaStreamCreate(&memtransfer_streams[i]));
      if (i){
        if (i!=(nStreams-1))
          memtransfer_streamDim[i] = memtransfer_streamDim[0];
        else
          memtransfer_streamDim[i] = (n-i*memtransfer_streamDim[0]);
      }
      memtransfer_streamSize[i] = memtransfer_streamDim[i]*sizeof(cudapcgVar_t);
    }
        
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    float time, mean_time=0.0;
    HANDLE_ERROR(cudaEventRecord(start,0));
    
    // Initialize x values on CPU
    if (!(solver->x0_hasBeenSet_flag))
      zeros(d,n);
    cudaMemcpy(res_x,d,n*sizeof(cudapcgVar_t),cudaMemcpyDeviceToHost); // d is used to store x0 in GPU
      
    cudapcgVar_t a, delta, delta_0, delta_old;
    cudapcgVar_t * d_ptr_dotres;
    
    solver->iteration = 0;

    // residual parameters for x0=[0]
    solver->applyPreConditioner_stream(model,s,s,0.0,s,kernels_stream);    // s = M^-1*r + 0.0*r
    d_ptr_dotres = solver->dotprod_precond_stream(model,s,kernels_stream); // delta = (M*s)*s
    HANDLE_ERROR(cudaMemcpyAsync(&delta_0,d_ptr_dotres,sizeof(cudapcgVar_t),cudaMemcpyDeviceToHost,kernels_stream));
    HANDLE_ERROR(cudaStreamSynchronize(kernels_stream));
    
    // Check if x0=[0] has already satisfied an absolute tolerance
    // This is a safety check. As we perform dimensionless residual evaluation,
    // in respect to delta_0, numerical trouble might occur if this value is too small.
    if (ABS(delta_0) < 0.000000000001){ // 1e-12
        solver->residual = 0.0;
        printf("%sNull solution satisfied PCG.\n",solver->header_str);
        //HANDLE_ERROR(cudaStreamSynchronize(memtransfer_stream));
        for (unsigned int i=0; i<nStreams; i++){
          HANDLE_ERROR(cudaStreamSynchronize(memtransfer_streams[i]));
          HANDLE_ERROR(cudaStreamDestroy(memtransfer_streams[i]));
        }
        //HANDLE_ERROR(cudaStreamDestroy(memtransfer_stream));
        HANDLE_ERROR(cudaStreamDestroy(kernels_stream));
        return CUDAPCG_TRUE;
    }

    // check if an initial guess was provided
    if (solver->x0_hasBeenSet_flag){
      // recalculate resiudals considering initial guess
      solver->Aprod_stream(model,d,s,-1.0,CUDAPCG_TRUE,kernels_stream);       // s += -(M^-1*A*x) (using d to store initial guess for x)
      d_ptr_dotres = solver->dotprod_precond_stream(model,s,kernels_stream);  // delta = (M*s)*s
      HANDLE_ERROR(cudaMemcpyAsync(&delta,d_ptr_dotres,sizeof(cudapcgVar_t),cudaMemcpyDeviceToHost,kernels_stream));
      HANDLE_ERROR(cudaStreamSynchronize(kernels_stream));
      // Check if initial guess has already satisfied dimensionless tolerance
      if (!isResidualAboveTol(delta,delta_0,solver->num_tol)){
          solver->residual = evalResidual(delta,delta_0);
          printf("%sInitial guess satisfied PCG.\n",solver->header_str);
          for (unsigned int i=0; i<nStreams; i++){
            HANDLE_ERROR(cudaStreamSynchronize(memtransfer_streams[i]));
            HANDLE_ERROR(cudaStreamDestroy(memtransfer_streams[i]));
          }
          HANDLE_ERROR(cudaStreamDestroy(kernels_stream));
          return CUDAPCG_TRUE;
      }
    } else {
      // assume that x0=[0] is initial guess
      delta = delta_0;
    }
    delta_old = delta;
    arrcpy_stream(d,s,n,kernels_stream);     // d = s
    HANDLE_ERROR(cudaStreamSynchronize(kernels_stream));
    
    unsigned int offset = 0;
    for (unsigned int i=0; i<nStreams; i++){
      HANDLE_ERROR(cudaMemcpyAsync(&solver->h_d[offset],&d[offset],memtransfer_streamSize[i],cudaMemcpyDeviceToHost,memtransfer_streams[i]));
      offset += memtransfer_streamDim[i];
    }

    // First iteration outside of while loop
    solver->iteration++;
    a = delta / solver->dotprod_Aprod_stream(model,d,kernels_stream);      // a = delta/(d*(A*d))
    solver->Aprod_stream(model,d,s,-a,CUDAPCG_TRUE,kernels_stream);        // s += -a*(M^-1*A*d)
    d_ptr_dotres = solver->dotprod_precond_stream(model,s,kernels_stream); // delta = (M*s)*s
    // x += a*d
    offset = 0;
    for (unsigned int i=0; i<nStreams; i++){
      HANDLE_ERROR(cudaStreamSynchronize(memtransfer_streams[i]));
      #pragma omp parallel for
      for (unsigned int ii=offset; ii<(offset+memtransfer_streamDim[i]); ii++){
        res_x[ii] += a*solver->h_d[ii];
      }
      offset += memtransfer_streamDim[i];
    }
    HANDLE_ERROR(cudaMemcpyAsync(&delta,d_ptr_dotres,sizeof(cudapcgVar_t),cudaMemcpyDeviceToHost,kernels_stream));
    HANDLE_ERROR(cudaStreamSynchronize(kernels_stream));
    
    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&time,start,stop));
    mean_time = ((solver->iteration-1)*mean_time + time) / solver->iteration;
    
    #ifndef CUDAPCG_QUIET_ITERATIONS
      printf("\r%siteration: %d, residual: %.3e",solver->header_str,solver->iteration,evalResidual(delta,delta_0));
    #endif

    while (isResidualAboveTol(delta,delta_0,solver->num_tol) && solver->iteration < solver->max_iterations){
        HANDLE_ERROR(cudaEventRecord(start,0));
        
        solver->iteration++;
        sumVec_stream(s,d,delta/delta_old,n,d,kernels_stream); // d = s+(delta/delta_old)*d
        HANDLE_ERROR(cudaStreamSynchronize(kernels_stream));
        offset = 0;
        for (unsigned int i=0; i<nStreams; i++){
          HANDLE_ERROR(cudaMemcpyAsync(&solver->h_d[offset],&d[offset],memtransfer_streamSize[i],cudaMemcpyDeviceToHost,memtransfer_streams[i]));
          offset += memtransfer_streamDim[i];
        }
        a = delta / solver->dotprod_Aprod_stream(model,d,kernels_stream);       // a = delta/(d*(A*d))
        solver->Aprod_stream(model,d,s,-a,CUDAPCG_TRUE,kernels_stream);         // s += -a*(M^-1*A*d)
        d_ptr_dotres = solver->dotprod_precond_stream(model,s,kernels_stream);  // delta = (M*s)*s
        // x += a*d
        offset = 0;
        for (unsigned int i=0; i<nStreams; i++){
          HANDLE_ERROR(cudaStreamSynchronize(memtransfer_streams[i]));
          #pragma omp parallel for
          for (unsigned int ii=offset; ii<(offset+memtransfer_streamDim[i]); ii++){
            res_x[ii] += a*solver->h_d[ii];
          }
          offset += memtransfer_streamDim[i];
        }
        delta_old = delta;
        HANDLE_ERROR(cudaMemcpyAsync(&delta,d_ptr_dotres,sizeof(cudapcgVar_t),cudaMemcpyDeviceToHost,kernels_stream));
        HANDLE_ERROR(cudaStreamSynchronize(kernels_stream));
        
        HANDLE_ERROR(cudaEventRecord(stop,0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        HANDLE_ERROR(cudaEventElapsedTime(&time,start,stop));
        mean_time = ((solver->iteration-1)*mean_time + time) / solver->iteration;
        
        #ifndef CUDAPCG_QUIET_ITERATIONS
          printf("\r%siteration: %d, residual: %.3e",solver->header_str,solver->iteration,evalResidual(delta,delta_0));
        #endif
    }
    
    #ifdef CUDAPCG_QUIET_ITERATIONS
      printf("%siteration: %d, residual: %.3e",solver->header_str,solver->iteration,evalResidual(delta,delta_0));
    #endif
    printf("\n");
    
    // Evaluate dimensionless residual
    solver->residual = evalResidual(delta,delta_0);
    
    // For safety, destroy streams
    for (unsigned int i=0; i<nStreams; i++)
      HANDLE_ERROR(cudaStreamDestroy(memtransfer_streams[i]));
    HANDLE_ERROR(cudaStreamDestroy(kernels_stream));
    
    solver->mean_time_per_iteration = mean_time*0.001; // value is in ms
    
    solver->foundSolution_flag = !isResidualAboveTol(delta,delta_0,solver->num_tol);

    return solver->foundSolution_flag;
}
//------------------------------------------------------------------------------
