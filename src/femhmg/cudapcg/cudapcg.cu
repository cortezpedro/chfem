/*
    Universidade Federal Fluminense (UFF) - Niteroi, Brazil
    Institute of Computing
    Author: Cortez, P.
    History: v1.0 (november/2020)

    API for solving linear systems associated to
    FEM models with an assembly-free PCG
    method, using CUDA.
    All global matrix operations involve "assembly on-the-fly"

    History:
        Initially developed as a final work for the graduate course "Arquitetura
        e Programacao de GPUs", at the Institute of Computing, UFF.

    ATTENTION.1:
        Considers a structured regular mesh of quad elements (2D)
        or hexahedron elements (3D).

    ATTENTION.2:
        As it is, this API is not generic, for any Ax = b. Linear systems must
        be associated to FEM homogenization problems, for Potential or
        Elasticity, both 2D or 3D.
*/

#include "cudapcg.h"
#include "error_handling.h"
#include "kernels/cudapcg_kernels_wrappers.h"
#include "solvers/cudapcg_default.h"

//---------------------------------
///////////////////////////////////
//////////// GLOBALS //////////////
///// (FOR INTERNAL USE ONLY) /////
///////////////////////////////////
//---------------------------------

cudapcgSolver_t *solver = NULL;

//---------------------------------
///////////////////////////////////
//////// PRIVATE FUNCTIONS ////////
////////// (DECLARATIONS) /////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
cudapcgFlag_t setAnalysis();
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
//////// PRIVATE FUNCTIONS ////////
//////////  (AUXILIARY)  //////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
cudapcgFlag_t setModelStruct_simple(cudapcgModel_t **ptr, const void *data){
  if (data==NULL) return CUDAPCG_FALSE;
  *ptr = (cudapcgModel_t *)data;
  return CUDAPCG_TRUE;
}
cudapcgFlag_t (*setModelStruct)(cudapcgModel_t **, const void *) = setModelStruct_simple;
//------------------------------------------------------------------------------
void freeModelStruct(cudapcgModel_t *model){
    if (solver->model->image!=NULL)           HANDLE_ERROR(cudaFree(solver->model->image));           solver->model->image=NULL;
    free(solver->model); solver->model=NULL;
    return;
}
//------------------------------------------------------------------------------
cudapcgFlag_t isModelValid(cudapcgSolver_t *_solver){
  if (_solver==NULL) return CUDAPCG_FALSE;
  if (_solver->model==NULL) return CUDAPCG_FALSE;
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
//////// PUBLIC FUNCTIONS /////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Setters and getters
cudapcgFlag_t cudapcgSetNumTol(cudapcgTol_t t){ if(solver==NULL) return CUDAPCG_FALSE; solver->num_tol = t; return CUDAPCG_TRUE; }
cudapcgFlag_t cudapcgSetMaxIterations(unsigned int n){ if(solver==NULL) return CUDAPCG_FALSE; solver->max_iterations = n; return CUDAPCG_TRUE; }
unsigned int cudapcgGetNumIterations(){ if(solver==NULL) return 0; return solver->iteration; }
unsigned int cudapcgGetMaxNumIterations(){ if(solver==NULL) return 0; return solver->max_iterations; }
cudapcgVar_t cudapcgGetResidual(){ if(solver==NULL) return 0; return solver->residual; }
//---------------------------------------------------------------------------------

cudapcgFlag_t cudapcgInit(cudapcgFlag_t analysis_flag, cudapcgFlag_t parAssembly_flag){

    // Make sure that no previously allocated data messes with this init -- for safety!
    cudapcgEnd();

    // Initial checks for potential inconsistencies in provided model
    if (analysis_flag != CUDAPCG_THERMAL_2D && analysis_flag != CUDAPCG_THERMAL_3D &&
        analysis_flag != CUDAPCG_ELASTIC_2D && analysis_flag != CUDAPCG_ELASTIC_3D)
        return CUDAPCG_FALSE;
    
    parAssembly_flag = parAssembly_flag ? 1 : 0; // EBE (1) vs NBN (0)
    
    solver = (cudapcgSolver_t *)malloc(sizeof(cudapcgSolver_t));
    if (solver == NULL){
        printf("ERROR: Memory allocation for solver struct has failed.\n");
        return CUDAPCG_FALSE;
    }
    
    solver->header_str = (char *)malloc(sizeof(char));
    strcpy(solver->header_str,"");
    
    solver->model = NULL;
    
    solver->analysis_flag = analysis_flag;
    solver->parallelStrategy_flag = parAssembly_flag;
    setParallelAssemblyFlag(parAssembly_flag); // api for kernels
    
    solver->residual = 1.0;
    solver->iteration = 0;
    solver->total_time = 0.0;
    solver->mean_time_per_iteration = 0.0;
    solver->foundSolution_flag = CUDAPCG_FALSE;
    
    solver->num_tol = CUDAPCG_TOLERANCE;
    solver->max_iterations = CUDAPCG_MAX_ITERATIONS;
    
    solver->userAllocatedArrays_flag = CUDAPCG_FALSE;
    solver->x0_hasBeenSet_flag = CUDAPCG_FALSE;
    
    if (!setAnalysis())
      return CUDAPCG_FALSE;

    // Initialize as default solver (4 arrays in the device)
    solver->solve = solve_default;
    solver->setX0 = setX0_default;
    solver->allocDeviceArrays = allocDeviceArrays_default;
    solver->freeDeviceArrays = freeDeviceArrays_default;
    
    // Initialize arrays pointing to NULL
    solver->x = NULL;
    solver->r = NULL;
    solver->d = NULL;
    solver->q = NULL;
    solver->h_d = NULL;
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgEnd(){
    if (solver == NULL)
        return CUDAPCG_FALSE;
    if (solver->x0_hasBeenSet_flag){
        if (solver->x!=NULL) HANDLE_ERROR(cudaFree(solver->x)); solver->x=NULL;
        if (solver->d!=NULL) HANDLE_ERROR(cudaFree(solver->d)); solver->d=NULL;
    }
    free(solver->header_str);
    if (solver->model!=NULL && solver->model->freeAllowed_flag)
      freeModelStruct(solver->model);
    if (solver->userAllocatedArrays_flag)
      cudapcgFreeArrays();
    if (solver->r!=NULL) HANDLE_ERROR(cudaFree(solver->r)); solver->r=NULL;
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      freeLocalK();
    #endif
    free(solver);
    solver = NULL;
    // Reset device (call from cuda API) -- for safety!
    cudaDeviceReset();
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgModel_t * cudapcgNewModel(void){
  cudapcgModel_t *new_model = (cudapcgModel_t *)malloc(sizeof(cudapcgModel_t));
  if (!new_model) return NULL;
  new_model->freeAllowed_flag = CUDAPCG_TRUE;
  new_model->nrows = 0;
  new_model->ncols = 0;
  new_model->nlayers = 0;
  new_model->nelem = 0;
  new_model->nvars = 0;
  new_model->nkeys = 0;
  new_model->nlocalvars = 0;
  new_model->image = NULL;
  return new_model;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetModel(cudapcgModel_t *model){
  if (solver==NULL) return CUDAPCG_FALSE;
  if (solver->model!=NULL && solver->model->freeAllowed_flag)
    freeModelStruct(solver->model);
  solver->model = model;
  cudapcgFreeArrays();
  unsigned int var_sz = sizeof(cudapcgVar_t)*solver->model->nvars;
  if (solver->r!=NULL) HANDLE_ERROR(cudaFree(solver->r)); solver->r=NULL;
  HANDLE_ERROR(cudaMalloc(&(solver->r),var_sz));
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetModelConstructorFcn(cudapcgFlag_t (*fcn)(cudapcgModel_t **, const void *)){
  if (fcn==NULL) return CUDAPCG_FALSE;
  setModelStruct = fcn;
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgBuildModel(const void *data){
  if (solver==NULL) return CUDAPCG_FALSE;
  if (solver->model==NULL){
    solver->model = cudapcgNewModel();
  } else if (solver->model->freeAllowed_flag){
    freeModelStruct(solver->model);
    solver->model = cudapcgNewModel();
  }
  if (!setModelStruct(&(solver->model),data)) return CUDAPCG_FALSE;
  cudapcgFreeArrays();
  unsigned int var_sz = sizeof(cudapcgVar_t)*solver->model->nvars;
  if (solver->r!=NULL) HANDLE_ERROR(cudaFree(solver->r)); solver->r=NULL;
  HANDLE_ERROR(cudaMalloc(&(solver->r),var_sz));
  if (solver->model->image==NULL)
    HANDLE_ERROR(cudaMalloc(&(solver->model->image),sizeof(cudapcgMap_t)*solver->model->nelem));
  
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetX0(cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate){
    if (!isModelValid(solver))
        return CUDAPCG_FALSE;
    // Check if an array was provided
    if (x0 != NULL)
        return solver->setX0(solver,x0,mustInterpolate);
    // Recieved null pointer for x0
    if (!solver->x0_hasBeenSet_flag)
        return CUDAPCG_TRUE;
    if (solver->x == NULL){
        solver->x0_hasBeenSet_flag = CUDAPCG_FALSE;
        return CUDAPCG_TRUE;
    }
    if (solver->userAllocatedArrays_flag){
        zeros(solver->x,solver->model->nvars);
        return CUDAPCG_TRUE;
    }
    if (solver->x!=NULL) HANDLE_ERROR(cudaFree(solver->x)); solver->x = NULL;
    solver->x0_hasBeenSet_flag = CUDAPCG_FALSE;
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetRHS(cudapcgVar_t * RHS){
    if (!isModelValid(solver))
        return CUDAPCG_FALSE;
    HANDLE_ERROR(cudaMemcpy(solver->r,RHS,solver->model->nvars*sizeof(cudapcgVar_t),cudaMemcpyHostToDevice));
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetImage(cudapcgMap_t *img){ 
    if (!isModelValid(solver))
        return CUDAPCG_FALSE;
    // Obs.: Arr size always works with solver->model->nelem because it is numerically equivalent to (valid_nodes/dof_per_node)
    HANDLE_ERROR(cudaMemcpy(solver->model->image,img,solver->model->nelem*sizeof(cudapcgMap_t),cudaMemcpyHostToDevice));
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetLclMtxs(cudapcgVar_t * LclMtxs){
    if (!isModelValid(solver))
        return CUDAPCG_FALSE;
    unsigned long int sz = solver->model->nkeys*solver->model->nlocalvars*sizeof(cudapcgVar_t);
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      freeLocalK(); // for safety
      allocLocalK(sz);
    #endif
    setLocalK(LclMtxs,sz);
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgAllocateArrays(){
    if (!isModelValid(solver))
        return CUDAPCG_FALSE;
    return solver->allocDeviceArrays(solver);
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgFreeArrays(){
    if (solver == NULL)
        return CUDAPCG_FALSE;
    return solver->freeDeviceArrays(solver);
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSolve(cudapcgVar_t *res_x){
    if (!isModelValid(solver) || res_x==NULL)
        return CUDAPCG_FALSE;

    double t = omp_get_wtime();

    cudapcgFlag_t mustFreeAfterPCG = CUDAPCG_TRUE;
    if (solver->userAllocatedArrays_flag)
        mustFreeAfterPCG = CUDAPCG_FALSE;
    else
        solver->allocDeviceArrays(solver);
    
    solver->solve(solver,res_x);

    if (mustFreeAfterPCG)
        solver->freeDeviceArrays(solver);
        
    solver->total_time = omp_get_wtime()-t;

    return solver->foundSolution_flag;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetHeaderString(char *header){
    if (solver==NULL)
      return CUDAPCG_FALSE;
    
    free(solver->header_str);
    
    if (header==NULL){
      solver->header_str = (char *)malloc(sizeof(char));
      strcpy(solver->header_str,"");
      return CUDAPCG_TRUE;
    }
    
    solver->header_str = (char *)malloc(sizeof(char)*(strlen(header)+1));
    if (solver->header_str==NULL){
      solver->header_str = (char *)malloc(sizeof(char));
      strcpy(solver->header_str,"");
      return CUDAPCG_FALSE;
    }
    strcpy(solver->header_str,header);
    
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgPrintSolverReport(){
    if (solver==NULL)
      return CUDAPCG_FALSE;
    
    if (solver->header_str) printf("%s",solver->header_str);
    if (solver->foundSolution_flag)
        printf("PCG solver found a solution in %d iterations.\n",solver->iteration);
    else
        printf("PCG solver did not converge within provided max number of iterations (%d).\n",solver->max_iterations);
    
    if (solver->header_str) printf("%s",solver->header_str);
    printf("Residual: %e\n",solver->residual);
    
    return cudapcgPrintSolverMetrics();
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgPrintSolverReport2(char *dest){
    if (dest==NULL)
      return CUDAPCG_FALSE;
    
    int sz = sprintf(
      dest,
      "%sPCG solver ran through %d iterations.\n"\
      "%sResidual: %e\n"\
      "%sMean time per iteration: %.2e s\n"\
      "%sElapsed time: %.2e s\n",
      solver->header_str,solver->iteration,
      solver->header_str,solver->residual,
      solver->header_str,solver->mean_time_per_iteration,
      solver->header_str,solver->total_time
    );
    
    if (sz<0) return CUDAPCG_FALSE;
    
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgPrintSolverMetrics(){
    if (solver==NULL)
      return CUDAPCG_FALSE;
    
    if (solver->header_str) printf("%s",solver->header_str);
    printf("Mean time per iteration: %.2e s\n",solver->mean_time_per_iteration);
    
    if (solver->header_str) printf("%s",solver->header_str);
    printf("Elapsed time: %.2e s\n",solver->total_time);
    
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgPrintSolverMetrics2(char *dest){
    if (dest==NULL)
      return CUDAPCG_FALSE;
    
    int sz = sprintf(
      dest,
      "%sMean time per iteration: %.2e s\n"\
      "%sElapsed time: %.2e s\n",
      solver->header_str,solver->mean_time_per_iteration,
      solver->header_str,solver->total_time
    );
    
    if (sz<0) return CUDAPCG_FALSE;
    
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
//////// PRIVATE FUNCTIONS ////////
//////// (IMPLEMENTATIONS) ////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
cudapcgFlag_t setAnalysis(){

  if (solver == NULL) return CUDAPCG_FALSE;

  if (solver->analysis_flag == CUDAPCG_THERMAL_2D){

    solver->Aprod = Aprod_thermal_2D;
    if (solver->parallelStrategy_flag==1) { // elem-by-elem
      solver->mustAssemblePreConditioner = CUDAPCG_TRUE;
      solver->assemblePreConditioner = assemblePreConditioner_thermal_2D;
    } else {
      solver->mustAssemblePreConditioner = CUDAPCG_FALSE;
    }
    solver->applyPreConditioner = applyPreConditioner_thermal_2D;

  } else if (solver->analysis_flag == CUDAPCG_THERMAL_3D){

    solver->Aprod = Aprod_thermal_3D;
    if (solver->parallelStrategy_flag==1) { // elem-by-elem
      solver->mustAssemblePreConditioner = CUDAPCG_TRUE;
      solver->assemblePreConditioner = assemblePreConditioner_thermal_3D;
    } else {
      solver->mustAssemblePreConditioner = CUDAPCG_FALSE;
    }
    solver->applyPreConditioner = applyPreConditioner_thermal_3D;

  } else if (solver->analysis_flag == CUDAPCG_ELASTIC_2D){

    solver->Aprod = Aprod_elastic_2D;
    if (solver->parallelStrategy_flag==1) { // elem-by-elem
      solver->mustAssemblePreConditioner = CUDAPCG_TRUE;
      solver->assemblePreConditioner = assemblePreConditioner_elastic_2D;
    } else {
      solver->mustAssemblePreConditioner = CUDAPCG_FALSE;
    }
    solver->applyPreConditioner = applyPreConditioner_elastic_2D;

  } else if (solver->analysis_flag == CUDAPCG_ELASTIC_3D){

    solver->Aprod = Aprod_elastic_3D;
    if (solver->parallelStrategy_flag==1){ // elem-by-elem
      solver->mustAssemblePreConditioner = CUDAPCG_TRUE;
      solver->assemblePreConditioner = assemblePreConditioner_elastic_3D;
    } else {
      solver->mustAssemblePreConditioner = CUDAPCG_FALSE;
    }
    solver->applyPreConditioner = applyPreConditioner_elastic_3D;

  } else
    return CUDAPCG_FALSE;
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------

