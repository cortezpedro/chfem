/*
  =====================================================================
  Universidade Federal Fluminense (UFF) - Niteroi, Brazil
  Institute of Computing
  Authors: Cortez Lopes, P., Pereira., A.
  contact: pedrocortez@id.uff.br
  =====================================================================
*/

#include "cudapcg.h"
#include "error_handling.h"
#include "kernels/wrappers.h"

#include "solvers/cg.h"
#include "solvers/pcg.h"
#include "solvers/cg3.h"
#include "solvers/pcg3.h"

#include "solvers/minres.h"
#include "solvers/pminres.h"
#include "solvers/minres3.h"
#include "solvers/pminres3.h"

// Only available for permeability analysis for now.
#include "solvers/cg2.h"
#include "solvers/pcg2.h"
#include "solvers/minres2.h"
#include "solvers/pminres2.h"

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
cudapcgFlag_t setSolver(unsigned int flag);
cudapcgFlag_t setResNorm(cudapcgFlag_t flag);
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
    if (solver->model->name !=NULL)            {free(solver->model->name); solver->model->name = NULL;}
    if (solver->model->image!=NULL)            {HANDLE_ERROR(cudaFree(solver->model->image));            solver->model->image=NULL;}
    if (solver->model->parametric_density_field!=NULL){HANDLE_ERROR(cudaFree(solver->model->parametric_density_field)); solver->model->parametric_density_field=NULL;}
    if (solver->model->pore_map!=NULL)         {HANDLE_ERROR(cudaFree(solver->model->pore_map));         solver->model->pore_map=NULL;}
    if (solver->model->border_pore_map!=NULL)  {HANDLE_ERROR(cudaFree(solver->model->border_pore_map));  solver->model->border_pore_map=NULL;}    
    if (solver->model->periodic2DOF_map!=NULL) {HANDLE_ERROR(cudaFree(solver->model->periodic2DOF_map)); solver->model->periodic2DOF_map=NULL;}
    if (solver->model->DOF2periodic_map!=NULL) {HANDLE_ERROR(cudaFree(solver->model->DOF2periodic_map)); solver->model->DOF2periodic_map=NULL;}
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
cudapcgFlag_t cudapcgSetXReduceMode(cudapcgFlag_t flag){ if(solver==NULL) return CUDAPCG_FALSE; solver->xreduce_flag = flag <= CUDAPCG_XREDUCE_FULL ? flag : CUDAPCG_XREDUCE_FULL; return CUDAPCG_TRUE; }
cudapcgFlag_t cudapcgSetXReduceShift(unsigned int shift){ if(solver==NULL) return CUDAPCG_FALSE; solver->xreduce_shift = shift; return CUDAPCG_TRUE; }
cudapcgFlag_t cudapcgSetXReduceScale(double scl){ if(solver==NULL) return CUDAPCG_FALSE; solver->xreduce_scale = scl; return CUDAPCG_TRUE; }
cudapcgFlag_t cudapcgSetReduceStabFactor(double scl){ if(solver==NULL) return CUDAPCG_FALSE; solver->reduce_stab_factor = scl; return CUDAPCG_TRUE; }
cudapcgFlag_t cudapcgSetSolver(cudapcgFlag_t flag){ if(solver==NULL) return CUDAPCG_FALSE; return setSolver(flag);}
cudapcgFlag_t cudapcgSetPreconditioner(cudapcgFlag_t flag){ if(solver==NULL) return CUDAPCG_FALSE; solver->preconditioner_flag = (flag>0) ? CUDAPCG_TRUE : CUDAPCG_FALSE; solver->solver_flag++; return setSolver(solver->solver_flag-1);}
cudapcgFlag_t cudapcgSetResNorm(cudapcgFlag_t flag){ if(solver==NULL) return CUDAPCG_FALSE; return setResNorm(flag);}
unsigned int cudapcgGetNumIterations(){ if(solver==NULL) return 0; return solver->iteration; }
unsigned int cudapcgGetMaxNumIterations(){ if(solver==NULL) return 0; return solver->max_iterations; }
cudapcgVar_t cudapcgGetResidual(){ if(solver==NULL) return 0; return solver->residual; }
//---------------------------------------------------------------------------------

cudapcgFlag_t cudapcgInit(cudapcgFlag_t analysis_flag, cudapcgFlag_t parStrategy_flag){

    // Make sure that no previously allocated data messes with this init -- for safety!
    cudapcgEnd();

    // Initial checks for potential inconsistencies in provided model
    if (analysis_flag != CUDAPCG_THERMAL_2D           && analysis_flag != CUDAPCG_THERMAL_3D &&
        analysis_flag != CUDAPCG_ELASTIC_2D           && analysis_flag != CUDAPCG_ELASTIC_3D &&
        analysis_flag != CUDAPCG_FLUID_2D             && analysis_flag != CUDAPCG_FLUID_3D)
        return CUDAPCG_FALSE;

    // Check if user provided flag for unsupported functionalities
    if ((analysis_flag == CUDAPCG_FLUID_2D || analysis_flag == CUDAPCG_FLUID_3D) && parStrategy_flag != CUDAPCG_NBN){
      if (parStrategy_flag == CUDAPCG_EBE){ // ElemByElem
        printf("ERROR: EBE solver is unavailable for FLUID analysis.\n");
      } else {
        printf("ERROR: Invalid \"parStrategy_flag\" (%d) for FLUID analysis.\n",parStrategy_flag);
      }
      return CUDAPCG_FALSE;
    }

    solver = (cudapcgSolver_t *)malloc(sizeof(cudapcgSolver_t));
    if (solver == NULL){
        printf("ERROR: Memory allocation for solver struct has failed.\n");
        return CUDAPCG_FALSE;
    }

    solver->count = 0;

    solver->header_str = (char *)malloc(sizeof(char));
    strcpy(solver->header_str,"");

    solver->model = NULL;

    solver->analysis_flag = analysis_flag;
    solver->parallelStrategy_flag = parStrategy_flag;

    solver->residual = 1.0;
    solver->iteration = 0;
    solver->total_time = 0.0;
    solver->mean_time_per_iteration = 0.0;
    solver->foundSolution_flag = CUDAPCG_FALSE;

    solver->num_tol = CUDAPCG_TOLERANCE;
    solver->max_iterations = CUDAPCG_MAX_ITERATIONS;

    solver->xreduce_flag = CUDAPCG_XREDUCE_FULL;
    solver->xreduce_shift = 0;
    solver->xreduce_scale = 1.0;
    solver->reduce_stab_factor = 0.0001;

    solver->resnorm_flag = CUDAPCG_L2_NORM;

    solver->userAllocatedArrays_flag = CUDAPCG_FALSE;
    solver->x0_hasBeenSet_flag = CUDAPCG_FALSE;
    solver->mustAssemblePreConditioner = CUDAPCG_FALSE;

    if (!setAnalysis())
      return CUDAPCG_FALSE;

    // Initialize solver
    solver->solver_flag = CUDAPCG_CG_SOLVER;
    solver->preconditioner_flag = CUDAPCG_TRUE;
    solver->solve = solve_pcg;
    solver->setX0 = setX0_cg;
    solver->allocDeviceArrays = allocDeviceArrays_cg;
    solver->freeDeviceArrays = freeDeviceArrays_cg;

    // Initialize arrays pointing to NULL
    solver->x = NULL;
    solver->r = NULL;
    solver->d = NULL;
    solver->q = NULL;
    solver->s = NULL; // used in minres
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgEnd(){
    if (solver == NULL) return CUDAPCG_FALSE;
    free(solver->header_str);
    if (solver->model!=NULL && solver->model->freeAllowed_flag) freeModelStruct(solver->model);
    if (solver->userAllocatedArrays_flag) cudapcgFreeArrays();
    if (solver->x!=NULL) HANDLE_ERROR(cudaFree(solver->x)); solver->x=NULL;
    if (solver->r!=NULL) HANDLE_ERROR(cudaFree(solver->r)); solver->r=NULL;
    if (solver->d!=NULL) HANDLE_ERROR(cudaFree(solver->d)); solver->d=NULL;
    if (solver->q!=NULL) HANDLE_ERROR(cudaFree(solver->q)); solver->q=NULL;
    if (solver->s!=NULL) HANDLE_ERROR(cudaFree(solver->s)); solver->s=NULL;
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
  new_model->name = NULL;
  new_model->freeAllowed_flag = CUDAPCG_TRUE;
  new_model->parStrategy_flag = CUDAPCG_NBN;
  new_model->poremap_flag     = CUDAPCG_POREMAP_NUM;
  new_model->parametric_density_field_flag = CUDAPCG_FALSE;
  #ifdef TESTING_STENCIL
  new_model->SBS_flag = CUDAPCG_TRUE;
  #else
  new_model->SBS_flag = CUDAPCG_FALSE;
  #endif
  new_model->nrows = 0;
  new_model->ncols = 0;
  new_model->nlayers = 0;
  new_model->nelem = 0;
  new_model->nvars = 0;
  new_model->nkeys = 0;
  new_model->localmtxdim = 0;
  new_model->nporenodes = 0;
  new_model->nbordernodes = 0;
  new_model->image = NULL;
  new_model->parametric_density_field = NULL;
  new_model->limits_density_field[0]=0.0;
  new_model->limits_density_field[1]=0.0;
  new_model->pore_map = NULL;
  new_model->border_pore_map = NULL;
  new_model->periodic2DOF_map = NULL;
  new_model->DOF2periodic_map = NULL;
  return new_model;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetModel(cudapcgModel_t *model){
  if (solver==NULL) return CUDAPCG_FALSE;
  if (solver->model!=NULL && solver->model->freeAllowed_flag)
    freeModelStruct(solver->model);
  solver->model = model;
  cudapcgFreeArrays();
  size_t var_sz = sizeof(cudapcgVar_t)*((size_t)solver->model->nvars);
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
  size_t var_sz = sizeof(cudapcgVar_t)*((size_t)solver->model->nvars);
  if (solver->r!=NULL) HANDLE_ERROR(cudaFree(solver->r)); solver->r=NULL;
  HANDLE_ERROR(cudaMalloc(&(solver->r),var_sz));
  if (solver->analysis_flag < CUDAPCG_FLUID_2D){
    if (solver->model->parametric_density_field_flag == CUDAPCG_FALSE){
      if (solver->model->image==NULL)
        HANDLE_ERROR(cudaMalloc(&(solver->model->image),sizeof(cudapcgMap_t)*solver->model->nelem));
    } else {
      if (solver->analysis_flag >= CUDAPCG_ELASTIC_2D){ // not required for thermal analysis
        HANDLE_ERROR(cudaMalloc(&(solver->model->image),sizeof(cudapcgMap_t)*solver->model->nelem));
      }
      HANDLE_ERROR(cudaMalloc(&(solver->model->parametric_density_field),sizeof(parametricScalarField_t)*solver->model->nelem));
    }
  } else {
    if (solver->model->periodic2DOF_map==NULL)
      HANDLE_ERROR(cudaMalloc(&(solver->model->periodic2DOF_map),sizeof(cudapcgIdMap_t)*solver->model->nelem));
    if (solver->model->poremap_flag == CUDAPCG_POREMAP_IMG){
        if (solver->model->pore_map==NULL)
          HANDLE_ERROR(cudaMalloc(&(solver->model->pore_map),sizeof(cudapcgFlag_t)*solver->model->nelem));
    } else if (solver->model->poremap_flag == CUDAPCG_POREMAP_NUM){
        if (solver->model->border_pore_map==NULL)
          HANDLE_ERROR(cudaMalloc(&(solver->model->border_pore_map),sizeof(cudapcgFlag_t)*solver->model->nbordernodes));
        if (solver->model->DOF2periodic_map==NULL)
          HANDLE_ERROR(cudaMalloc(&(solver->model->DOF2periodic_map),sizeof(cudapcgIdMap_t)*(solver->model->nporenodes+solver->model->nbordernodes)));
    }
  }
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetX0(cudapcgVar_t *x0, cudapcgFlag_t mustInterpolate){
    if (!isModelValid(solver)) return CUDAPCG_FALSE;
    // Check if an array was provided
    if (x0 != NULL) return solver->setX0(solver,x0,mustInterpolate);
    // Recieved null pointer for x0
    if (!solver->x0_hasBeenSet_flag) return CUDAPCG_TRUE;
    solver->x0_hasBeenSet_flag = CUDAPCG_FALSE;
    cudapcgVar_t **ptr_to_device_x0 = solver->solver_flag < CUDAPCG_CG2_SOLVER ? &(solver->x) : &(solver->d);
    if (*ptr_to_device_x0 == NULL) return CUDAPCG_TRUE;
    if (solver->userAllocatedArrays_flag){
        zeros(*ptr_to_device_x0,solver->model->nvars);
        return CUDAPCG_TRUE;
    }
    if (*ptr_to_device_x0!=NULL) HANDLE_ERROR(cudaFree(*ptr_to_device_x0)); *ptr_to_device_x0 = NULL;
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
cudapcgFlag_t cudapcgSetParametricDensityField(parametricScalarField_t *field, double fmin, double fmax){
    if (!isModelValid(solver))
        return CUDAPCG_FALSE;
    HANDLE_ERROR(cudaMemcpy(solver->model->parametric_density_field,field,solver->model->nelem*sizeof(parametricScalarField_t),cudaMemcpyHostToDevice));
    solver->model->limits_density_field[0] = fmin;
    solver->model->limits_density_field[1] = fmax;
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetPoreMap(cudapcgFlag_t *pores){
    if (!isModelValid(solver))
        return CUDAPCG_FALSE;
    if (solver->model->poremap_flag == CUDAPCG_POREMAP_IMG){
    // Obs.: Arr size always works with solver->model->nelem because it is numerically equivalent to (valid_nodes/dof_per_node)
    HANDLE_ERROR(cudaMemcpy(solver->model->pore_map,pores,solver->model->nelem*sizeof(cudapcgFlag_t),cudaMemcpyHostToDevice));
    } else if (solver->model->poremap_flag == CUDAPCG_POREMAP_NUM){
    HANDLE_ERROR(cudaMemcpy(solver->model->border_pore_map,pores,solver->model->nbordernodes*sizeof(cudapcgFlag_t),cudaMemcpyHostToDevice));
    } else return CUDAPCG_FALSE;
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetPeriodic2DOFMap(cudapcgIdMap_t *vars){
    if (!isModelValid(solver))
        return CUDAPCG_FALSE;
    // if (solver->model->poremap_flag != CUDAPCG_POREMAP_NUM) return CUDAPCG_FALSE;
    // Obs.: Arr size always works with solver->model->nelem because it is numerically equivalent to (valid_nodes/dof_per_node)
    HANDLE_ERROR(cudaMemcpy(solver->model->periodic2DOF_map,vars,solver->model->nelem*sizeof(cudapcgIdMap_t),cudaMemcpyHostToDevice));
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetDOF2PeriodicMap(cudapcgIdMap_t *nodes){
    if (!isModelValid(solver))
        return CUDAPCG_FALSE;
    if (solver->model->poremap_flag != CUDAPCG_POREMAP_NUM) return CUDAPCG_FALSE;
    HANDLE_ERROR(cudaMemcpy(solver->model->DOF2periodic_map,nodes,(solver->model->nporenodes+solver->model->nbordernodes)*sizeof(cudapcgIdMap_t),cudaMemcpyHostToDevice));
    return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgSetLclMtxs(cudapcgVar_t * LclMtxs){
    if (!isModelValid(solver))
        return CUDAPCG_FALSE;
    size_t sz = solver->model->nkeys*solver->model->localmtxdim*sizeof(cudapcgVar_t);
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

    solver->assemblePreConditioner = assemblePreConditioner_thermal_2D;
    solver->applyPreConditioner = applyPreConditioner_thermal_2D;
    solver->applyinvPreConditioner = applyinvPreConditioner_thermal_2D;
    solver->Aprod = Aprod_thermal_2D;
    solver->PreConditionerAprod = PreConditionerAprod_thermal_2D;
    solver->dotPreConditioner = dotPreConditioner_thermal_2D;
    solver->dotinvPreConditioner = dotinvPreConditioner_thermal_2D;
    solver->dotAprod = dotAprod_thermal_2D;
    solver->dotA2prod = dotA2prod_thermal_2D;
    solver->dotPreConditionerA2prod = dotPreConditionerA2prod_thermal_2D;

  } else if (solver->analysis_flag == CUDAPCG_THERMAL_3D){

    solver->assemblePreConditioner = assemblePreConditioner_thermal_3D;
    solver->applyPreConditioner = applyPreConditioner_thermal_3D;
    solver->applyinvPreConditioner = applyinvPreConditioner_thermal_3D;
    solver->Aprod = Aprod_thermal_3D;
    solver->PreConditionerAprod = PreConditionerAprod_thermal_3D;
    solver->dotPreConditioner = dotPreConditioner_thermal_3D;
    solver->dotinvPreConditioner = dotinvPreConditioner_thermal_3D;
    solver->dotAprod = dotAprod_thermal_3D;
    solver->dotA2prod = dotA2prod_thermal_3D;
    solver->dotPreConditionerA2prod = dotPreConditionerA2prod_thermal_3D;

  } else if (solver->analysis_flag == CUDAPCG_ELASTIC_2D){

    solver->assemblePreConditioner = assemblePreConditioner_elastic_2D;
    solver->applyPreConditioner = applyPreConditioner_elastic_2D;
    solver->applyinvPreConditioner = applyinvPreConditioner_elastic_2D;
    solver->Aprod = Aprod_elastic_2D;
    solver->PreConditionerAprod = PreConditionerAprod_elastic_2D;
    solver->dotPreConditioner = dotPreConditioner_elastic_2D;
    solver->dotinvPreConditioner = dotinvPreConditioner_elastic_2D;
    solver->dotAprod = dotAprod_elastic_2D;
    solver->dotA2prod = dotA2prod_elastic_2D;
    solver->dotPreConditionerA2prod = dotPreConditionerA2prod_elastic_2D;

  } else if (solver->analysis_flag == CUDAPCG_ELASTIC_3D){

    solver->assemblePreConditioner = assemblePreConditioner_elastic_3D;
    solver->applyPreConditioner = applyPreConditioner_elastic_3D;
    solver->applyinvPreConditioner = applyinvPreConditioner_elastic_3D;
    solver->Aprod = Aprod_elastic_3D;
    solver->PreConditionerAprod = PreConditionerAprod_elastic_3D;
    solver->dotPreConditioner = dotPreConditioner_elastic_3D;
    solver->dotinvPreConditioner = dotinvPreConditioner_elastic_3D;
    solver->dotAprod = dotAprod_elastic_3D;
    solver->dotA2prod = dotA2prod_elastic_3D;
    solver->dotPreConditionerA2prod = dotPreConditionerA2prod_elastic_3D;

  } else if (solver->analysis_flag == CUDAPCG_FLUID_2D){

    solver->assemblePreConditioner = NULL;
    solver->applyPreConditioner = applyPreConditioner_fluid_2D;
    solver->applyinvPreConditioner = applyinvPreConditioner_fluid_2D;
    solver->Aprod = Aprod_fluid_2D;
    solver->PreConditionerAprod = PreConditionerAprod_fluid_2D;
    solver->dotPreConditioner = dotPreConditioner_fluid_2D;
    solver->dotinvPreConditioner = dotinvPreConditioner_fluid_2D;
    solver->dotAprod = dotAprod_fluid_2D;
    solver->dotA2prod = dotA2prod_fluid_2D;
    solver->dotPreConditionerA2prod = dotPreConditionerA2prod_fluid_2D;

  } else if (solver->analysis_flag == CUDAPCG_FLUID_3D){

    solver->assemblePreConditioner = NULL;
    solver->applyPreConditioner = applyPreConditioner_fluid_3D;
    solver->applyinvPreConditioner = applyinvPreConditioner_fluid_3D;
    solver->Aprod = Aprod_fluid_3D;
    solver->PreConditionerAprod = PreConditionerAprod_fluid_3D;
    solver->dotPreConditioner = dotPreConditioner_fluid_3D;
    solver->dotinvPreConditioner = dotinvPreConditioner_fluid_3D;
    solver->dotAprod = dotAprod_fluid_3D;
    solver->dotA2prod = dotA2prod_fluid_3D;
    solver->dotPreConditionerA2prod = dotPreConditionerA2prod_fluid_3D;

  } else return CUDAPCG_FALSE;

  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
cudapcgFlag_t setSolver(unsigned int flag){

  if (solver == NULL) return CUDAPCG_FALSE;

  if (flag > CUDAPCG_MINRES2_SOLVER){
    printf("ERROR: Invalid \"solverFlag\" (%d).\n",flag);
    return CUDAPCG_FALSE;
  }

  if (solver->solver_flag == flag) return CUDAPCG_TRUE;

  if (flag == CUDAPCG_CG_SOLVER){
    if (solver->preconditioner_flag == CUDAPCG_TRUE)
      solver->solve = solve_pcg;
    else
      solver->solve = solve_cg;
    solver->setX0 = setX0_cg;
    solver->allocDeviceArrays = allocDeviceArrays_cg;
    solver->freeDeviceArrays = freeDeviceArrays_cg;

  } else if (flag == CUDAPCG_MINRES_SOLVER){
    if (solver->preconditioner_flag == CUDAPCG_TRUE)
      solver->solve = solve_pminres;
    else
      solver->solve = solve_minres;
    solver->setX0 = setX0_minres;
    solver->allocDeviceArrays = allocDeviceArrays_minres;
    solver->freeDeviceArrays = freeDeviceArrays_minres;

  } else if (flag == CUDAPCG_CG3_SOLVER){
    if (solver->preconditioner_flag == CUDAPCG_TRUE)
      solver->solve = solve_pcg3;
    else
      solver->solve = solve_cg3;
    solver->setX0 = setX0_cg3;
    solver->allocDeviceArrays = allocDeviceArrays_cg3;
    solver->freeDeviceArrays = freeDeviceArrays_cg3;

  } else if (flag == CUDAPCG_MINRES3_SOLVER){
    if (solver->preconditioner_flag == CUDAPCG_TRUE)
      solver->solve = solve_pminres3;
    else
      solver->solve = solve_minres3;
    solver->setX0 = setX0_minres3;
    solver->allocDeviceArrays = allocDeviceArrays_minres3;
    solver->freeDeviceArrays = freeDeviceArrays_minres3;

  } else if (flag == CUDAPCG_CG2_SOLVER){
    if (solver->preconditioner_flag == CUDAPCG_TRUE)
      solver->solve = solve_pcg2;
    else
      solver->solve = solve_cg2;
    solver->setX0 = setX0_cg2;
    solver->allocDeviceArrays = allocDeviceArrays_cg2;
    solver->freeDeviceArrays = freeDeviceArrays_cg2;

  } else if (flag == CUDAPCG_MINRES2_SOLVER){
    if (solver->preconditioner_flag == CUDAPCG_TRUE)
      solver->solve = solve_pminres2;
    else
      solver->solve = solve_minres2;
    solver->setX0 = setX0_minres2;
    solver->allocDeviceArrays = allocDeviceArrays_minres2;
    solver->freeDeviceArrays = freeDeviceArrays_minres2;
  }

  if (solver->s != NULL) { HANDLE_ERROR(cudaFree(solver->s)); solver->s = NULL; }
  if (solver->userAllocatedArrays_flag && flag == CUDAPCG_MINRES_SOLVER){
    HANDLE_ERROR(cudaMalloc(&(solver->s),solver->model->nvars*sizeof(cudapcgVar_t)));
  }

  solver->solver_flag = flag;
  return CUDAPCG_TRUE;
  
}
//------------------------------------------------------------------------------
cudapcgFlag_t setResNorm(cudapcgFlag_t flag){
  if (solver == NULL) return CUDAPCG_FALSE;
  solver->resnorm_flag = flag > CUDAPCG_ERROR_NORM ? CUDAPCG_L2_NORM : flag; // defaults to L2
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
