/*
	Universidade Federal Fluminense (UFF) - Niteroi, Brazil
	Institute of Computing
	Authors: Cortez, P., Vianna, R., Sapucaia, V., Pereira., A.
	History: 
		* v0.0 (jul/2020) [ALL]    -> OpenMp, parallelization of CPU code
		* v1.0 (nov/2020) [CORTEZ] -> CUDA, PCG on GPU. (substituted parpcg.h with cudapcg.h)

	API for the handling of FEM models used in
    material homogenization.
    Makes use of cudapcg.h to solve linear system
    of equations.

    ATTENTION.1:
        Considers a structured regular mesh of quad elements (2D)
        or hexahedron elements (3D).

    ATTENTION.2:
        As it is, works for Potential or Elasticity, both 2D or 3D.
*/

#include "femhmg.h"
#include "physical_phenomena/femhmg_thermal_2D.h"
#include "physical_phenomena/femhmg_thermal_3D.h"
#include "physical_phenomena/femhmg_elastic_2D.h"
#include "physical_phenomena/femhmg_elastic_3D.h"

//---------------------------------
///////////////////////////////////
//////////// GLOBALS //////////////
///// (FOR INTERNAL USE ONLY) /////
///////////////////////////////////
//---------------------------------

hmgModel_t * hmgModel = NULL;

char report_buffer[1024]; // 1 KB (should always be enough)

double time_total;

// These are auxiliary. Might organize in a struct later.
unsigned int assemblyfree_strategy_flag=0;

//---------------------------------
///////////////////////////////////
//////// PRIVATE FUNCTIONS ////////
////////// (DECLARATIONS) /////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
logical readData(char * filename);
logical readMaterialMap(char * filename);
logical readMaterialMapNF(char * filename);
logical readMaterialMapRAW(char * filename);
logical setAnalysisType();
void free_model_data(hmgModel_t * model);
void assemble_coarse_copy(hmgModel_t * coarse_model, hmgModel_t * original_model);
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
//////// PRIVATE FUNCTIONS ////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
cudapcgFlag_t cudapcgModel_constructor(cudapcgModel_t **ptr, const void *data){
  if (data==NULL) return CUDAPCG_FALSE;
  
  hmgModel_t *model = (hmgModel_t *)data;
  
  if (model->m_nmat==0 || model->m_nelem==0) return CUDAPCG_FALSE;
  if (model->m_nz==0 && model->m_dim_flag==HMG_3D) return CUDAPCG_FALSE;
  
  (*ptr)->nrows = model->m_ny-1;
  (*ptr)->ncols = model->m_nx-1;
  (*ptr)->nlayers = (model->m_nz-1)*(model->m_nz>0);
  
  (*ptr)->nelem = model->m_nelem;
  (*ptr)->nvars = model->m_ndof;
  
  (*ptr)->nkeys = model->m_nmat;
  (*ptr)->nlocalvars = model->m_lclMtx_dim;
  
  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
//////// PUBLIC FUNCTIONS /////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
logical hmgInit(char * data_filename, char * elem_filename){

	// Initialize time metric
	time_total = omp_get_wtime();
	
	printf("Initializing femhmg...\n");

	// Allocate model struct
	hmgModel = (hmgModel_t *)malloc(sizeof(hmgModel_t));
  if (hmgModel == NULL){
      printf("ERROR: Memory allocation for model struct has failed.\n");
      return HMG_FALSE;
  }
  
  // Feed filename strings to struct
  hmgModel->neutralFile = data_filename;
  hmgModel->imageFile = elem_filename;
  
	// Init auxiliary flags
	hmgModel->m_hmg_flag_was_set = HMG_FALSE;
	hmgModel->m_hmg_flag = HOMOGENIZE_ALL;
	hmgModel->m_using_x0_flag = HMG_FALSE;

	// Read data input file
	if (!readData(data_filename))
		return HMG_FALSE;
  
  unsigned int step_count=0, num_of_steps=2;
  printf("\r    Building model struct...[%3d%%]",(step_count*100)/num_of_steps);
  
	// Set analysis type
	if (!setAnalysisType())
		return HMG_FALSE;
	printf("\r    Building model struct...[%3d%%]",((++step_count)*100)/num_of_steps);
  
	// Initialize model parameters
	if (!hmgModel->initModel(hmgModel))
		return HMG_FALSE;
	printf("\r    Building model struct...[%3d%%]",((++step_count)*100)/num_of_steps);
		
	printf("\n");

	// Allocate material map array
	hmgModel->elem_material_map = (cudapcgMap_t *) malloc(sizeof(cudapcgMap_t)*hmgModel->m_nelem);
	if (hmgModel->elem_material_map == NULL){
      printf("ERROR: Memory allocation for image array has failed.\n");
      free(hmgModel);
      return HMG_FALSE;
  }

	// Read elem map input file
	if (!readMaterialMap(elem_filename)){
		free(hmgModel->elem_material_map);
		free(hmgModel);
		return HMG_FALSE;
	}
	
	step_count=0, num_of_steps=3;
	printf("\r    Assembling maps and local matrices...[%3d%%]",(step_count*100)/num_of_steps);

	// Allocate hmgModel->node_dof_map array
	hmgModel->node_dof_map = (unsigned int *) malloc(sizeof(unsigned int)*hmgModel->m_nnode);
	if (hmgModel->node_dof_map == NULL){
      printf("ERROR: Memory allocation for DOF map has failed.\n");
      free(hmgModel->elem_material_map);
		  free(hmgModel);
      return HMG_FALSE;
  }

	// Assemble hmgModel->node_dof_map
	hmgModel->assembleNodeDofMap(hmgModel);
	printf("\r    Assembling maps and local matrices...[%3d%%]",((++step_count)*100)/num_of_steps);

	// Allocate hmgModel->dof_material_map array
  hmgModel->dof_material_map = (cudapcgMap_t *) malloc(sizeof(cudapcgMap_t)*hmgModel->m_ndof/hmgModel->m_nnodedof);
  if (hmgModel->dof_material_map == NULL){
      printf("ERROR: Memory allocation for DOF material map has failed.\n");
      free(hmgModel->node_dof_map);
      free(hmgModel->elem_material_map);
	    free(hmgModel);
      return HMG_FALSE;
  }

	// Assemble hmgModel->dof_material_map
	hmgModel->assembleDofMaterialMap(hmgModel);
	printf("\r    Assembling maps and local matrices...[%3d%%]",((++step_count)*100)/num_of_steps);

	// Allocate local FEM matrices array
	hmgModel->Mtxs = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*hmgModel->m_lclMtx_dim*hmgModel->m_nmat);
	hmgModel->CB   = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*hmgModel->m_lclCB_dim*hmgModel->m_nmat);
	
	if (hmgModel->Mtxs == NULL || hmgModel->CB == NULL){
      printf("ERROR: Memory allocation for local matrices has failed.\n");
      free(hmgModel->dof_material_map);
      free(hmgModel->node_dof_map);
      free(hmgModel->elem_material_map);
	    free(hmgModel);
      return HMG_FALSE;
  }

	// Assemble local FE matrices
	hmgModel->assembleLocalMtxs(hmgModel);
	printf("\r    Assembling maps and local matrices...[%3d%%]",((++step_count)*100)/num_of_steps);
  printf("\n");

	// Allocate constitutive matrix array
	hmgModel->C = (var *) malloc(sizeof(var)*hmgModel->m_C_dim);
	if (hmgModel->Mtxs == NULL){
      printf("ERROR: Memory allocation for matrix representation of homogenized tensor has failed.\n");
      free(hmgModel->Mtxs);
      free(hmgModel->CB);
      free(hmgModel->dof_material_map);
      free(hmgModel->node_dof_map);
      free(hmgModel->elem_material_map);
	    free(hmgModel);
      return HMG_FALSE;
  }
	
	// Initialize pointer to array of strings to store analysis report
	hmgModel->report = reportCreate(NULL);
	hmgModel->m_report_flag = HMG_FALSE;
	
	// Print feedback that process went ok
	double time_init = omp_get_wtime()-time_total;
	printf("Done.(%.2e s)\n",time_init);
	
  return HMG_TRUE;
}
//------------------------------------------------------------------------------
logical hmgEnd(){
  if (hmgModel == NULL) return HMG_FALSE;

	// Check if x0 matrix should be freed
	if (hmgModel->m_using_x0_flag){
		for (unsigned int i=0; i<6; i++){
			if (hmgModel->x0[i] != NULL)
				free(hmgModel->x0[i]);
		}
		free(hmgModel->x0);
	}

	// Free dynamic arrays from memory
	free(hmgModel->elem_material_map);
	free(hmgModel->node_dof_map);
	if (hmgModel->dof_material_map) free(hmgModel->dof_material_map);
	free(hmgModel->Mtxs);
	free(hmgModel->CB);
	free(hmgModel->C);
	
	// free report
	reportFree(hmgModel->report,REPORT_TRUE);

	// Free model struct
	free(hmgModel);
	hmgModel = NULL;

	// Compute total elapsed time
	time_total = omp_get_wtime()-time_total;
	printf("Elapsed time (total): %.2e s\n",time_total);

  return HMG_TRUE;
}
//------------------------------------------------------------------------------
void hmgSetParallelStrategyFlag(cudapcgFlag_t flag){
  assemblyfree_strategy_flag = flag ? 1 : 0;
  return;
}
//------------------------------------------------------------------------------
void hmgSetHomogenizationFlag(hmgFlag_t flag){
  if (hmgModel == NULL) return;
  
  if (flag > HOMOGENIZE_XY){
    hmgModel->m_hmg_flag_was_set = HMG_FALSE;
    return;
  }
  if (hmgModel->m_analysis_flag == HMG_THERMAL){
    if (hmgModel->m_dim_flag == HMG_2D){
      if (flag > HOMOGENIZE_Y){
        hmgModel->m_hmg_flag_was_set = HMG_FALSE;
        return;
      }
    } else {
      if (flag > HOMOGENIZE_Z){
        hmgModel->m_hmg_flag_was_set = HMG_FALSE;
        return;
      }
    } 
  } else {//if (hmgModel->m_analysis_flag == HMG_ELASTIC)
    if (hmgModel->m_dim_flag == HMG_2D){
      if (flag > HOMOGENIZE_Y && flag != HOMOGENIZE_XY){
        hmgModel->m_hmg_flag_was_set = HMG_FALSE;
        return;
      }
    }
  }
  hmgModel->m_hmg_flag_was_set = HMG_TRUE;
  hmgModel->m_hmg_flag = flag;
  return;
}
//------------------------------------------------------------------------------
void hmgFindInitialGuesses(unsigned int nlevels){
  if (!nlevels || hmgModel==NULL)
    return;
  
  // Append text report
  if (hmgModel->m_report_flag)
    reportAppend(hmgModel->report, "Finding initial guesses...\n");
  
  // Buffer variable to measure wall time
  double t;
  
  // Measuring time to build coarse meshes
  t = omp_get_wtime();
  printf("\rBuilding coarse meshes...[%d/%d]",0,nlevels);
  
  // Allocate array of pointers to models
  hmgModel_t ** model_arr = (hmgModel_t **) malloc(sizeof(hmgModel_t *)*(nlevels+1));
  if (model_arr == NULL){
      printf(" ERROR: Memory allocation for coarse meshes array has failed.\n");
      exit(0);
      return;
  }
  
  // Set current model as head of array
  model_arr[0] = hmgModel;  
  
  // Go through all levels, building coarse material maps
  for (unsigned int i=0; i<nlevels; i++){
    // Allocate aray of pointers to initial guesses on previous (refined) model
    model_arr[i]->x0 = (cudapcgVar_t **) malloc(sizeof(cudapcgVar_t *)*6);
    if (model_arr[i]->x0 == NULL){
        printf(" ERROR: Memory allocation for list of initial guess arrays has failed.\n");
        exit(0);
        return;
    }
    model_arr[i]->x0[HOMOGENIZE_X]  = NULL;
	  model_arr[i]->x0[HOMOGENIZE_Y]  = NULL;
	  model_arr[i]->x0[HOMOGENIZE_Z]  = NULL;
	  model_arr[i]->x0[HOMOGENIZE_YZ] = NULL;
	  model_arr[i]->x0[HOMOGENIZE_XZ] = NULL;
	  model_arr[i]->x0[HOMOGENIZE_XY] = NULL;
	  model_arr[i]->m_using_x0_flag = HMG_TRUE;
	  // Create coarse model
	  model_arr[i+1] = (hmgModel_t *)malloc(sizeof(hmgModel_t));
    if (model_arr[i+1] == NULL){
        printf(" ERROR: Memory allocation for coarse mesh %d has failed.\n",i+1);
	      exit(0);
        return;
    }
	  model_arr[i+1]->m_using_x0_flag = HMG_FALSE;
	  assemble_coarse_copy(model_arr[i+1],model_arr[i]);
	  printf("\rBuilding coarse meshes...[%d/%d]",i+1,nlevels);
  }
  t = omp_get_wtime()-t;
  printf(" (%.2e s)\n",t);
  
  // Auxiliary simulation id static arrays
  const char sim_str[6][11] = { {"X"},{"Y"},{"Z"},{"YZ (shear)"},{"XZ (shear)"},{"XY (shear)"} };
	unsigned int sim_id[6] = {0,0,0,0,0,0};
	unsigned int sim_sz;
	if (hmgModel->m_hmg_flag_was_set){
		sim_sz = 1;
		sim_id[0] = hmgModel->m_hmg_flag;
	} else if (hmgModel->m_analysis_flag == HMG_THERMAL){
		if (hmgModel->m_dim_flag == HMG_2D){
			sim_sz = 2;
			sim_id[0] = HOMOGENIZE_X;
			sim_id[1] = HOMOGENIZE_Y;
		} else { //if (hmgModel->m_dim_flag == HMG_3D)
			sim_sz = 3;
			sim_id[0] = HOMOGENIZE_X;
			sim_id[1] = HOMOGENIZE_Y;
			sim_id[2] = HOMOGENIZE_Z;
		}
	} else {//if (hmgModel->m_analysis_flag == HMG_ELASTIC)
		if (hmgModel->m_dim_flag == HMG_2D){
			sim_sz = 3;
			sim_id[0] = HOMOGENIZE_X;
			sim_id[1] = HOMOGENIZE_Y;
			sim_id[2] = HOMOGENIZE_XY;
		} else { //if (hmgModel->m_dim_flag == HMG_3D)
			sim_sz = 6;
			sim_id[0] = HOMOGENIZE_X;
			sim_id[1] = HOMOGENIZE_Y;
			sim_id[2] = HOMOGENIZE_Z;
			sim_id[3] = HOMOGENIZE_YZ;
			sim_id[4] = HOMOGENIZE_XZ;
			sim_id[5] = HOMOGENIZE_XY;
		}
	}
  
  // Allocate x array
  cudapcgVar_t * x   = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*model_arr[1]->m_ndof);
  if (x==NULL){
    printf(" ERROR: Memory allocation for solution array has failed.\n");
    exit(0);
    return;
  }
  
  // Buffer variable to measure time per level
  double t_init_guess, t_level;
  
  // Auxiliary string to ensure proper indentation when printing solver report
  char str_indentation[] = "    | ";
  
  // Solve systems from back up, recursively finding initial guesses
  for (unsigned int i=nlevels; i>0; i--){
    t_level = omp_get_wtime();
    printf("Computing initial guesses in coarse mesh %d...\n",i);
	  t = omp_get_wtime();
    printf("    Initializing solver...");
	  // Init cudapcg api
	  if(!cudapcgInit(model_arr[i]->m_pcg_flag, assemblyfree_strategy_flag)){
	    printf("Failed.\n");
		  model_arr[i-1]->m_using_x0_flag = HMG_FALSE;
	    for (unsigned int j=0; j<6; j++){
	      if (model_arr[i-1]->x0[j] != NULL){
	        free(model_arr[i-1]->x0[j]);
	      }
	    }
	    free(model_arr[i-1]->x0);
	  } else {
	    // Set model constructor function for cudapcg API and build model
	    cudapcgSetModelConstructorFcn(cudapcgModel_constructor);
	    cudapcgBuildModel(model_arr[i]);
	    // Set indentation header to print metrics
	    cudapcgSetHeaderString(str_indentation);
	    // Assemble dof maps
      hmgModel = model_arr[i];
	    hmgModel->assembleNodeDofMap(hmgModel);
	    hmgModel->assembleDofMaterialMap(hmgModel);
	    hmgModel = model_arr[0];
	    // Allocate RHS
	    model_arr[i]->RHS = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*model_arr[i]->m_ndof);
	    if (model_arr[i]->RHS==NULL){
        printf(" ERROR: Memory allocation for right-hand side array has failed.\n");
        exit(0);
        return;
      }
	    // Set numeric tol and max num of iterations
	    cudapcgSetNumTol(model_arr[i]->m_num_tol);
	    cudapcgSetMaxIterations(model_arr[i]->m_max_iterations);
	    // Provide material map
	    if (assemblyfree_strategy_flag==0)
		    cudapcgSetImage(model_arr[i]->dof_material_map);
	    else
		    cudapcgSetImage(model_arr[i]->elem_material_map);
	    // Provide local FEM matrices
	    cudapcgSetLclMtxs(model_arr[i]->Mtxs);
	    // Allocate PCG arrays
	    cudapcgAllocateArrays();
	    // Print feedback regarding solver initialization
	    t = omp_get_wtime()-t;
	    printf("Done.(%.2e s)\n",t);
	    // Find initial guesses
	    hmgModel = model_arr[i];
	    for (unsigned int j=0; j<sim_sz; j++){
		    model_arr[i]->m_hmg_flag = sim_id[j];
		    sprintf(
		      report_buffer,
		      "    ------------------------------------------------------\n"\
		      "    Finding initial guess on %s:\n"\
		      "    Mesh dimension: [%i,%i,%i] elements\n",
		      sim_str[model_arr[i]->m_hmg_flag],model_arr[i]->m_nx-1,model_arr[i]->m_ny-1,(model_arr[i]->m_nz-1)*(model_arr[i]->m_nz>0)
		    );
		    if (model_arr[0]->m_report_flag)
          reportAppend(model_arr[0]->report, report_buffer);
        printf("%s",report_buffer);
		    // Provide RHS of the linear system of eqs to be solved
		    t = omp_get_wtime();
	      printf("    Assembling RHS...");
		    model_arr[i]->assembleRHS(model_arr[i]);
		    cudapcgSetRHS(model_arr[i]->RHS);
		    t = omp_get_wtime()-t;
	      printf("Done.(%.2e s)\n",t);
		    // Solve system of eqs
		    t = omp_get_wtime();
		    strcpy(report_buffer,"    Solving system of equations...\n");
        printf("%s",report_buffer);
        if (model_arr[0]->m_report_flag)
          reportAppend(model_arr[0]->report, report_buffer);
		    if (model_arr[i]->m_using_x0_flag){
		      t_init_guess = omp_get_wtime();
		      printf("%sInterpolating initial guess...",str_indentation);
		      cudapcgSetX0(model_arr[i]->x0[model_arr[i]->m_hmg_flag],CUDAPCG_TRUE);
		      free(model_arr[i]->x0[model_arr[i]->m_hmg_flag]);
		      model_arr[i]->x0[model_arr[i]->m_hmg_flag] = NULL;
		      t_init_guess = omp_get_wtime()-t_init_guess;
		      printf("Done.(%.2e s)\n",t_init_guess);
		    }
		    cudapcgFlag_t foundSolution = cudapcgSolve(x);
		    t = omp_get_wtime()-t;
        cudapcgPrintSolverMetrics();
		    if (model_arr[0]->m_report_flag){
		      cudapcgPrintSolverReport2(report_buffer);
		      reportAppend(model_arr[0]->report, report_buffer);
		    }
        printf("    Done.(%.2e s)\n",t);
        if (foundSolution){
          t = omp_get_wtime();
          printf("    Storing initial guess...");
          model_arr[i-1]->x0[model_arr[i]->m_hmg_flag] = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*model_arr[i]->m_ndof);
          if (model_arr[i]->RHS==NULL){
            printf(" ERROR: Memory allocation for initial guess array has failed.\n");
            exit(0);
            return;
          }
			    cudapcgVar_t *ptr_x0 = model_arr[i-1]->x0[model_arr[i]->m_hmg_flag];
			    unsigned int dim = model_arr[i]->m_ndof;
			    #pragma omp parallel for private(dim)
			    for (unsigned int n=0; n<dim; n++){
			      ptr_x0[n] = x[n];
			    }
			    t = omp_get_wtime()-t;
			    printf("Done.(%.2e s)\n",t);
        }
	    }
	    strcpy(report_buffer,"    ------------------------------------------------------\n"\
	                         "    ******************************************************\n");
      printf("%s",report_buffer);
      if (model_arr[0]->m_report_flag)
        reportAppend(model_arr[0]->report, report_buffer);
	    hmgModel = model_arr[0];
	    // End cudapcg
	    cudapcgFreeArrays();
	    cudapcgSetX0(NULL,CUDAPCG_FALSE);
	    cudapcgEnd();
	    // Free RHS
	    free(model_arr[i]->RHS);
	  }
	  free_model_data(model_arr[i]);
	  if (model_arr[i]->m_using_x0_flag){
	    for (unsigned int j=0; j<6; j++){
	      if (model_arr[i]->x0[j] != NULL){
	        free(model_arr[i]->x0[j]);
	      }
	    }
	    free(model_arr[i]->x0);
	  }
	  free(model_arr[i]);
	  model_arr[i]=NULL;
	  //print feedback regarding this level
	  t_level = omp_get_wtime()-t_level;
	  printf("Done.(%.2e s)\n",t_level);
  }
  
  // Free dynamic array from memory
  free(x);
  
  // Free array of pointers to models
  for (unsigned int i=1; i<=nlevels; i++){
    if (model_arr[i]!=NULL)
      free(model_arr[i]);
  }
  free(model_arr);
  
  if (hmgModel->m_report_flag)
    reportAppend(hmgModel->report, "Done.\n");
  return;
}
//------------------------------------------------------------------------------
void hmgSolveHomogenization(){
  if (hmgModel == NULL) return;
  
  // Append text report
  if (hmgModel->m_report_flag)
    reportAppend(hmgModel->report, "Running homogenization...\n");

  // Buffer variables to measure wall time
  double t, t_init_guess, t_hmg;
  
  t_hmg = omp_get_wtime(); 
  printf("Running homogenization...\n");
  
  // Initialize C with zeros
	for (unsigned int i = 0; i<hmgModel->m_C_dim; i++)
		hmgModel->C[i] = 0.0;

  t = omp_get_wtime(); 
  printf("    Initializing solver...");
  
	// Init cudapcg api
	if(!cudapcgInit(hmgModel->m_pcg_flag, assemblyfree_strategy_flag)){
		return;
	}
	
	// Set model constructor fcn cudapcg API and build model
	cudapcgSetModelConstructorFcn(cudapcgModel_constructor);	
	cudapcgBuildModel(hmgModel);

	// Set numeric tol and max num of iterations
	cudapcgSetNumTol(hmgModel->m_num_tol);
	cudapcgSetMaxIterations(hmgModel->m_max_iterations);

	// Provide material map
  if (assemblyfree_strategy_flag==0)
	  cudapcgSetImage(hmgModel->dof_material_map);
  else
	  cudapcgSetImage(hmgModel->elem_material_map);

	// Provide local FEM matrices
	cudapcgSetLclMtxs(hmgModel->Mtxs);

	// Allocate RHS array
	hmgModel->RHS = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*hmgModel->m_ndof);
	if (hmgModel->RHS==NULL){
    printf(" ERROR: Memory allocation for right-hand side array has failed.\n");
    exit(0);
    return;
  }

	// Allocate x array
	cudapcgVar_t *x = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*hmgModel->m_ndof);
	if (x==NULL){
    printf(" ERROR: Memory allocation for solution array has failed.\n");
    exit(0);
    return;
  }

	// Allocate PCG arrays
	cudapcgAllocateArrays();
	
	// Measuirng time for solver initialization
	t = omp_get_wtime()-t;
	printf("Done.(%.2e s)\n",t);
  
	// Auxiliary simulation id static arrays
	const char sim_str[6][11] = { {"X"},{"Y"},{"Z"},{"YZ (shear)"},{"XZ (shear)"},{"XY (shear)"} };
	unsigned int sim_id[6] = {0,0,0,0,0,0};
	unsigned int sim_sz;
	if (hmgModel->m_hmg_flag_was_set){
		sim_sz = 1;
		sim_id[0] = hmgModel->m_hmg_flag;
	} else if (hmgModel->m_analysis_flag == HMG_THERMAL){
		if (hmgModel->m_dim_flag == HMG_2D){
			sim_sz = 2;
			sim_id[0] = HOMOGENIZE_X;
			sim_id[1] = HOMOGENIZE_Y;
		} else { //if (hmgModel->m_dim_flag == HMG_3D)
			sim_sz = 3;
			sim_id[0] = HOMOGENIZE_X;
			sim_id[1] = HOMOGENIZE_Y;
			sim_id[2] = HOMOGENIZE_Z;
		}
	} else {//if (hmgModel->m_analysis_flag == HMG_ELASTIC)
		if (hmgModel->m_dim_flag == HMG_2D){
			sim_sz = 3;
			sim_id[0] = HOMOGENIZE_X;
			sim_id[1] = HOMOGENIZE_Y;
			sim_id[2] = HOMOGENIZE_XY;
		} else { //if (hmgModel->m_dim_flag == HMG_3D)
			sim_sz = 6;
			sim_id[0] = HOMOGENIZE_X;
			sim_id[1] = HOMOGENIZE_Y;
			sim_id[2] = HOMOGENIZE_Z;
			sim_id[3] = HOMOGENIZE_YZ;
			sim_id[4] = HOMOGENIZE_XZ;
			sim_id[5] = HOMOGENIZE_XY;
		}
	}
	
	// Auxiliary string to ensure proper indentation when printing solver report
  char str_indentation[] = "    | ";
  
  // Set indentation header to print metrics
	cudapcgSetHeaderString(str_indentation);
  
  // Loop to call simulation on each direction
  for (unsigned int i=0; i<sim_sz; i++){
  
    // Set homogenization direction flag
    hmgModel->m_hmg_flag = sim_id[i];
    
    // Print header
	  sprintf(
      report_buffer,
      "    ------------------------------------------------------\n"\
      "    Analysis on %s:\n",
      sim_str[hmgModel->m_hmg_flag]
    );
    if (hmgModel->m_report_flag)
      reportAppend(hmgModel->report, report_buffer);
    printf("%s",report_buffer);
	
	  // Provide RHS of the linear system of eqs to be solved
	  t = omp_get_wtime();
	  printf("    Assembling RHS...");
	  hmgModel->assembleRHS(hmgModel);
	  cudapcgSetRHS(hmgModel->RHS);
	  t = omp_get_wtime()-t;
	  printf("Done.(%.2e s)\n",t);

	  // Solve system of eqs
	  t = omp_get_wtime();
	  strcpy(report_buffer,"    Solving system of equations...\n");
	  if (hmgModel->m_report_flag)
      reportAppend(hmgModel->report, report_buffer);
    printf("%s",report_buffer);
	  if (hmgModel->m_using_x0_flag){
	      t_init_guess = omp_get_wtime();
	      printf("%sInterpolating initial guess...",str_indentation);
	  	  cudapcgSetX0(hmgModel->x0[hmgModel->m_hmg_flag],CUDAPCG_TRUE);
	  	  t_init_guess = omp_get_wtime()-t_init_guess;
	  	  printf("Done.(%.2e s)\n",t_init_guess);
	  }
	  cudapcgSolve(x); t = omp_get_wtime()-t;
	  cudapcgPrintSolverMetrics();
	  if (hmgModel->m_report_flag){
      cudapcgPrintSolverReport2(report_buffer);
      reportAppend(hmgModel->report, report_buffer);
    }
	  printf("    Done.(%.2e s)\n",t);

	  // Update C
	  t = omp_get_wtime();
	  printf("    Updating homogenized properties...");
	  hmgModel->updateC(hmgModel,x);
	  t = omp_get_wtime()-t;
	  printf("Done.(%.2e s)\n",t);
	  if (hmgModel->m_report_flag){
	    sprintf(
        report_buffer,
        "    Updating homogenized properties...\n"\
        "%sElapsed time: %.2e s\n",
        str_indentation, t
      );
      reportAppend(hmgModel->report, report_buffer);
    }
  }
  
  strcpy(report_buffer,"    ------------------------------------------------------\n");
  if (hmgModel->m_report_flag)
    reportAppend(hmgModel->report, report_buffer);
  printf("%s",report_buffer);

	// Free dynamic arrays from memory
	cudapcgFreeArrays();
	cudapcgSetX0(NULL,CUDAPCG_FALSE);
	free(hmgModel->RHS);
	free(x);

	// Finish cudapcg api
	cudapcgEnd();
	
	// Print feedback that homogenization ended
	t_hmg = omp_get_wtime()-t_hmg;
	printf("Done.(%.2e s)\n",t_hmg);
	
	// Append text report
  if (hmgModel->m_report_flag)
    reportAppend(hmgModel->report, "Done.\n");
    
  return;
}
//------------------------------------------------------------------------------
void hmgPrintModelData(){
  if (hmgModel == NULL) return;

  char str_sim[11];
  if (hmgModel->m_analysis_flag == HMG_THERMAL){
    if (hmgModel->m_dim_flag == HMG_2D)
      strcpy(str_sim,"THERMAL_2D");
    else //if (hmgModel->m_dim_flag == HMG_3D)
      strcpy(str_sim,"THERMAL_3D");
  } else if (hmgModel->m_analysis_flag == HMG_ELASTIC){
    if (hmgModel->m_dim_flag == HMG_2D)
      strcpy(str_sim,"ELASTIC_2D");
    else //if (hmgModel->m_dim_flag == HMG_3D)
      strcpy(str_sim,"ELASTIC_3D");
  }
  
  const char str_hmg[7][11] = { {"X"},{"Y"},{"Z"},{"YZ (shear)"},{"XZ (shear)"},{"XY (shear)"},{"ALL"} };

  sprintf(
    report_buffer,
    "*******************************************************\n"\
    "MODEL DATA:\n"\
    "Analysis: %s\n"\
    "Homogenization on direction: %s\n"\
    "Number of elements (x,y,z): [%d,%d,%d]\n"\
    "Number of elements (total): %d\n"\
    "Number of nodes: %d\n"\
    "Number of DOFs: %d\n"\
    "Number of materials: %d\n"\
    "Tolerance for PCG: %e\n"\
    "Max iterations for PCG: %d\n"\
    "*******************************************************\n",
    str_sim,str_hmg[hmgModel->m_hmg_flag],hmgModel->m_nx-1,hmgModel->m_ny-1,hmgModel->m_nz-(hmgModel->m_nz>0),
    hmgModel->m_nelem,hmgModel->m_nnode,hmgModel->m_ndof,hmgModel->m_nmat,hmgModel->m_num_tol,hmgModel->m_max_iterations
  );
  
  if (hmgModel->m_report_flag)
    reportAppend(hmgModel->report, report_buffer);
  printf("%s",report_buffer);
  
  unsigned int flag = assemblyfree_strategy_flag;
  printf("Parallel assembly on-the-fly strategy: ");
	if (flag == 0)
	  printf("Node-by-node (%d)\n",flag);
	else if (flag == 1)
	  printf("Elem-by-elem (local serialization) (%d)\n",flag);
	printf("*******************************************************\n");
  return;
}
//------------------------------------------------------------------------------
void hmgPrintConstitutiveMtx(){
  if (hmgModel == NULL) return;
	hmgModel->printC(hmgModel,report_buffer);
	printf("%s",report_buffer);
	if (hmgModel->m_report_flag)
	  reportAppend(hmgModel->report, report_buffer);
	return;
}
//------------------------------------------------------------------------------
logical hmgSaveConstitutiveMtx(const char * filename){
  if (hmgModel == NULL) return HMG_FALSE;
  FILE *file;
  file = fopen(filename,"wb");
  if (file)
    fwrite(hmgModel->C,sizeof(var)*hmgModel->m_C_dim,1,file);
  fclose(file);
  return HMG_TRUE;
}
//------------------------------------------------------------------------------
void hmgKeepTrackOfAnalysisReport(reportFlag_t flag){
  if (hmgModel==NULL) return;
  hmgModel->m_report_flag = flag;
  if (hmgModel->m_report_flag){
    sprintf(
      report_buffer,
      "Neutral file: %s\n"\
      "Image: %s\n",
      hmgModel->neutralFile,
      hmgModel->imageFile
    );
    reportAppend(hmgModel->report,report_buffer);
  }
  return;
}
//------------------------------------------------------------------------------
reportFlag_t hmgPrintReport(const char *filename){
  if (hmgModel==NULL) return REPORT_FALSE;
  if (hmgModel->report==NULL || hmgModel->m_report_flag==REPORT_FALSE) return REPORT_FALSE;
  if (filename==NULL){
    reportPrint(hmgModel->report);
    return REPORT_TRUE;
  }
  return reportPrint2File(filename,hmgModel->report);
}
//------------------------------------------------------------------------------


//---------------------------------
///////////////////////////////////
//////// PRIVATE FUNCTIONS ////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
logical setAnalysisType(){
	// Switch between thermal 2D/3D, elasticity 2D/3D and permeability 2D/3D
	if (hmgModel->m_analysis_flag == HMG_THERMAL){
		if (hmgModel->m_dim_flag == HMG_2D){
			hmgModel->m_pcg_flag = CUDAPCG_THERMAL_2D;
			hmgModel->initModel = initModel_thermal_2D;
		} else if (hmgModel->m_dim_flag == HMG_3D){
			hmgModel->m_pcg_flag = CUDAPCG_THERMAL_3D;
			hmgModel->initModel = initModel_thermal_3D;
		} else {
		  printf("ERROR: Bad image dimensions. Model is neither 2D nor 3D.\n");
			return HMG_FALSE;
	  }
	} else if (hmgModel->m_analysis_flag == HMG_ELASTIC){
		if (hmgModel->m_dim_flag == HMG_2D){
			hmgModel->m_pcg_flag = CUDAPCG_ELASTIC_2D;
			hmgModel->initModel = initModel_elastic_2D;
		} else if (hmgModel->m_dim_flag == HMG_3D){
			hmgModel->m_pcg_flag = CUDAPCG_ELASTIC_3D;
			hmgModel->initModel = initModel_elastic_3D;
		} else {
		  printf("ERROR: Bad image dimensions. Model is neither 2D nor 3D.\n");
			return HMG_FALSE;
	  }
	} else {
	  printf("ERROR: Analysis flag does not match any possible value.\n");
		return HMG_FALSE;
  }
	return HMG_TRUE;
}
//------------------------------------------------------------------------------
logical readData(char * filename){
	char str[STR_BUFFER_SIZE];
	unsigned char mat_id; // 8bit
	FILE * file;
	file = fopen( filename , "r");
	if (!file) return HMG_FALSE;
	var tol_buffer;
	unsigned int count=0, num_of_info=8;
	printf("\r    Scanning through neutral file...[%3d%%]",(count*100)/num_of_info);
  while (fscanf(file, "%s", str)!=EOF && count<num_of_info){
	  if (!strcmp(str,"%type_of_analysis")){
		  fscanf(file, "%hu", &(hmgModel->m_analysis_flag));
		  printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
		  if (hmgModel->m_analysis_flag != HMG_THERMAL &&
		      hmgModel->m_analysis_flag != HMG_ELASTIC ){
			  fclose(file);
			  printf("ERROR: Provided analysis flag does not match any possible value.\n");
			  return HMG_FALSE;
		  }
		  
	  //} else if (!strcmp(str,"%type_of_solver")){

	  //} else if (!strcmp(str,"%type_of_rhs")){

	  } else if (!strcmp(str,"%voxel_size")){
		  fscanf(file, "%lf", &(hmgModel->m_elem_size));
		  printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
	  } else if (!strcmp(str,"%solver_tolerance")){
		  fscanf(file, "%lf", &tol_buffer);
		  hmgModel->m_num_tol = (cudapcgTol_t) tol_buffer;
		  printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
	  } else if (!strcmp(str,"%number_of_iterations")){
		  fscanf(file, "%i", &(hmgModel->m_max_iterations));
		  printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
	  } else if (!strcmp(str,"%image_dimensions")){
      fscanf(file, "%i %i %i", &(hmgModel->m_nx), &(hmgModel->m_ny), &(hmgModel->m_nz));
      printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
		  hmgModel->m_nx++;hmgModel->m_ny++;
		  if (hmgModel->m_nz){
			  hmgModel->m_dim_flag = HMG_3D;
			  hmgModel->m_nz++;
		  } else {
			  hmgModel->m_dim_flag = HMG_2D;
		  }
	  } else if (!strcmp(str,"%refinement")){
		  fscanf(file, "%i", &(hmgModel->m_mesh_refinement));
		  printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
		  if (hmgModel->m_mesh_refinement < 1) return HMG_FALSE;
	  } else if (!strcmp(str,"%number_of_materials")){
		  fscanf(file, "%hhu", &(hmgModel->m_nmat));
		  printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
		  if (hmgModel->m_nmat > MAX_COLORNUM){
			  fclose(file);
			  printf("ERROR: Number of different materials (%d) is greater than max (%d).\n",hmgModel->m_nmat,MAX_COLORKEY);
			  return HMG_FALSE;
		  }
	  } else if (!strcmp(str,"%properties_of_materials")){
		  for (unsigned char i=0; i<hmgModel->m_nmat; i++){
			  if (hmgModel->m_analysis_flag == HMG_ELASTIC){
				  fscanf(file, "%hhu %lf %lf", &mat_id, &(hmgModel->props[2*i]), &(hmgModel->props[2*i+1]));
			  } else if (hmgModel->m_analysis_flag == HMG_THERMAL){
				  fscanf(file, "%hhu %lf", &mat_id, &(hmgModel->props[i]));
			  }
			  printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
			    
			  if (mat_id > MAX_COLORKEY){
				  fclose(file);
				  printf("ERROR: Material %d has color key (%d) greater than max (%d).\n",i,mat_id,MAX_COLORKEY);
				  return HMG_FALSE;
			  }
			  hmgModel->props_keys[mat_id] = i;
		  }
		  
	  //} else if (!strcmp(str,"%volume_fraction")){

	  //} else if (!strcmp(str,"%data_type")){

	  }
  }
	fclose(file);
	printf("\r    Scanning through neutral file...[%3d%%]",100);
	printf("\n");
	// check if number of materials is within bitstep range (for NbN and DbD strategies)
	if (hmgModel->m_nmat > (MATKEY_BITSTEP_RANGE_2D+1) && hmgModel->m_dim_flag == HMG_2D){
	  printf("WARNING: Number of specified materials (%d) is greater than what can be represented with MATKEY_BITSTEP (%d). ",hmgModel->m_nmat,MATKEY_BITSTEP_2D);
	  printf("Either avoid Node-by-Node, or recompile with greater MATKEY size.\n");
	} else if (hmgModel->m_nmat > (MATKEY_BITSTEP_RANGE_3D+1) && hmgModel->m_dim_flag == HMG_3D){
	  printf("WARNING: Number of specified materials (%d) is greater than what can be represented with MATKEY_BITSTEP (%d). ",hmgModel->m_nmat,MATKEY_BITSTEP_3D);
	  printf("Either avoid Node-by-Node, or recompile with greater MATKEY size.\n");
	}
	if (hmgModel->m_mesh_refinement > 1){
	  hmgModel->m_elem_size /= hmgModel->m_mesh_refinement;
	  hmgModel->m_nx = (hmgModel->m_nx-1)*hmgModel->m_mesh_refinement + 1;
	  hmgModel->m_ny = (hmgModel->m_ny-1)*hmgModel->m_mesh_refinement + 1;
	  if (hmgModel->m_dim_flag == HMG_3D)
	    hmgModel->m_nz = (hmgModel->m_nz-1)*hmgModel->m_mesh_refinement + 1;
	//} else if (hmgModel->m_mesh_refinement != 1){
	//  return HMG_FALSE;
	}
	return HMG_TRUE;
}
//------------------------------------------------------------------------------
logical readMaterialMap(char * filename){
	// Check file format before reading
	unsigned long int str_len = strlen(filename);
	if (str_len < 3)
		return HMG_FALSE;
	if (!strcmp(&filename[str_len-3],".nf"))
		return readMaterialMapNF(filename);
	if (!strcmp(&filename[str_len-4],".raw"))
		return readMaterialMapRAW(filename);
	printf("ERROR: file %s is not in a valid format.\n",filename);
	return HMG_FALSE;
}
//------------------------------------------------------------------------------
logical readMaterialMapNF(char * filename){
  // Check if a mesh refinement was set (not supported for this type of input)
  if (hmgModel->m_mesh_refinement != 1){
    printf("ERROR: Material map neutral file (.nf) does not support mesh refinement. Use a RAW binary file (.raw) instead.\n");
    return HMG_FALSE;
  }
	unsigned char buffer;
	unsigned int i;
	char str[STR_BUFFER_SIZE];
	unsigned int count=0, num_of_voxels=hmgModel->m_nelem;
	printf("\r    Getting image from .nf...[%3d%%]",(count*100)/num_of_voxels);
	FILE * file;
	file = fopen(filename,"r");
	if (file) {
    while (fscanf(file, "%s", str)!=EOF){
      if (!strcmp(str,"%ELEMENTS")){
        for (i=0;i<hmgModel->m_nelem;i++){
          if (fscanf(file, "%hhu", &buffer)!=EOF){; // data is 8bit
            printf("\r    Getting image from .nf...[%3d%%]",(++count*100)/num_of_voxels);
	          hmgModel->elem_material_map[i] = (cudapcgMap_t) hmgModel->props_keys[buffer];
	        } else {
	           // If reached EOF before expected, return false
					  fclose(file);
					  printf(" ERROR: Reached end of file before expected.\n");
					  return HMG_FALSE;
	        }
	      }
	      fclose(file);
        printf("\n");
        return HMG_TRUE;
	    }
    }
    fclose(file);
    printf(" ERROR: Failed to read image data.\n");
    return HMG_FALSE;
	}
	printf(" ERROR: Failed to open file.\n");
	return HMG_FALSE;
}
//------------------------------------------------------------------------------
logical readMaterialMapRAW(char * filename){
	unsigned char buffer; // 8bit
	unsigned int i, j, k, ii, jj, kk;
	unsigned int rows = (hmgModel->m_ny-1) / hmgModel->m_mesh_refinement;
	unsigned int cols = (hmgModel->m_nx-1) / hmgModel->m_mesh_refinement;
	unsigned int rows_ref = hmgModel->m_ny-1;
	unsigned int cols_ref = hmgModel->m_nx-1;
	unsigned int slices;
	if (hmgModel->m_dim_flag == HMG_3D)
		slices = (hmgModel->m_nz-1) / hmgModel->m_mesh_refinement;
	else
		slices = 1;
	printf("\r    Getting image from .raw...[%3d%%]",0);
	FILE * file;
	file = fopen(filename,"rb");
	if (file) {
	  // Loops to transpose data. Raw file is line by line, our indexing is
	  // column by column. First slice runs out of loop (2D).
	  k=0;
	  for (i = 0; i<rows; i++){
		  for (j = 0; j<cols; j++){
			  if (fread(&buffer,sizeof(unsigned char),1,file)!=EOF){ //data is 8bit
			    for (kk = hmgModel->m_mesh_refinement*k; kk<(hmgModel->m_mesh_refinement*(k+1)*(hmgModel->m_nz>0)+(hmgModel->m_nz<1)); kk++){
			      for (ii = hmgModel->m_mesh_refinement*i; ii<hmgModel->m_mesh_refinement*(i+1); ii++){
			        for (jj = hmgModel->m_mesh_refinement*j; jj<hmgModel->m_mesh_refinement*(j+1); jj++){
				        hmgModel->elem_material_map[ii+jj*rows_ref+kk*rows_ref*cols_ref] = (cudapcgMap_t) hmgModel->props_keys[buffer];
				      }
				    }
				  }
			  } else {
				  // If reached EOF before expected, return false
				  fclose(file);
				  printf(" ERROR: Reached end of file before expected.\n");
				  return HMG_FALSE;
			  }
		  }
		  printf("\r    Getting image from .raw...[%3d%%]",((i+1)*100)/(slices*rows));
	  }
	  for (k = 1; k<slices; k++){
		  for (i = 0; i<rows; i++){
			  for (j = 0; j<cols; j++){
				  if (fread(&buffer,sizeof(unsigned char),1,file)!=EOF){ //data is 8bit
				    for (kk = hmgModel->m_mesh_refinement*k; kk<(hmgModel->m_mesh_refinement*(k+1)*(hmgModel->m_nz>0)+(hmgModel->m_nz<1)); kk++){
				      for (ii = hmgModel->m_mesh_refinement*i; ii<hmgModel->m_mesh_refinement*(i+1); ii++){
				        for (jj = hmgModel->m_mesh_refinement*j; jj<hmgModel->m_mesh_refinement*(j+1); jj++){
					        hmgModel->elem_material_map[ii+jj*rows_ref+kk*rows_ref*cols_ref] = (cudapcgMap_t) hmgModel->props_keys[buffer];
					      }
					    }
					  }
				  } else {
					  // If reached EOF before expected, return false
					  fclose(file);
					  printf(" ERROR: Reached end of file before expected.\n");
					  return HMG_FALSE;
				  }
			  }
		  }
		  printf("\r    Getting image from .raw...[%3d%%]",((k+1)*100)/slices);
	  }
    fclose(file);
    printf("\n");
		return HMG_TRUE;
	}
	printf(" ERROR: Failed to open file.\n");
	return HMG_FALSE;
}
//------------------------------------------------------------------------------
void free_model_data(hmgModel_t * model){
  free(model->elem_material_map);
	free(model->node_dof_map);
	free(model->dof_material_map);
  return;
}
//------------------------------------------------------------------------------
void assemble_coarse_copy(hmgModel_t * coarse_model, hmgModel_t * original_model){
  // copy data from original to coarse model
	coarse_model->m_dim_flag             = original_model->m_dim_flag;
  coarse_model->m_analysis_flag        = original_model->m_analysis_flag;
  coarse_model->m_pcg_flag             = original_model->m_pcg_flag;
  coarse_model->m_hmg_flag             = original_model->m_hmg_flag;
  coarse_model->m_hmg_flag_was_set     = original_model->m_hmg_flag_was_set;
  coarse_model->m_elem_size            = original_model->m_elem_size;
  coarse_model->m_mesh_refinement      = original_model->m_mesh_refinement;
  coarse_model->m_C_dim                = original_model->m_C_dim;
  coarse_model->m_lclMtx_dim           = original_model->m_lclMtx_dim;
  coarse_model->m_lclCB_dim            = original_model->m_lclCB_dim;
	coarse_model->m_num_tol              = original_model->m_num_tol;
  coarse_model->m_max_iterations       = original_model->m_max_iterations;
  coarse_model->initModel              = original_model->initModel;
  coarse_model->assembleLocalMtxs      = original_model->assembleLocalMtxs;
  coarse_model->assembleNodeDofMap     = original_model->assembleNodeDofMap;
  coarse_model->assembleDofMaterialMap = original_model->assembleDofMaterialMap;
  coarse_model->assembleRHS            = original_model->assembleRHS;
  coarse_model->Mtxs                   = original_model->Mtxs;
  coarse_model->CB                     = original_model->CB;

	// Set coarse model dimension parameters
	coarse_model->m_nx       = (original_model->m_nx-1)/2+1;
  coarse_model->m_ny       = (original_model->m_ny-1)/2+1;
  coarse_model->m_nz       = ((original_model->m_nz-1)/2+1)*(original_model->m_nz>0);
  coarse_model->m_nnode    = coarse_model->m_nx * coarse_model->m_ny * (coarse_model->m_nz + (coarse_model->m_nz<1));
  coarse_model->m_nnodedof = original_model->m_nnodedof;
  coarse_model->m_nelem    = (coarse_model->m_nx-1) * (coarse_model->m_ny-1) * ((coarse_model->m_nz-1)*(coarse_model->m_nz>0) + (coarse_model->m_nz<1));
  coarse_model->m_nelemdof = original_model->m_nelemdof;
  coarse_model->m_ndof     = coarse_model->m_nelem * coarse_model->m_nnodedof;
  coarse_model->m_nmat     = original_model->m_nmat;

	// Allocate maps for coarse model
	coarse_model->elem_material_map = (cudapcgMap_t *) malloc(sizeof(cudapcgMap_t)*coarse_model->m_nelem);
	if (coarse_model->elem_material_map == NULL){
      printf(" ERROR: Memory allocation for coarse image-based mesh has failed.\n");
      exit(0);
      return;
  }
  coarse_model->node_dof_map      = (unsigned int *) malloc(sizeof(unsigned int)*coarse_model->m_nnode);
  if (coarse_model->node_dof_map == NULL){
      printf(" ERROR: Memory allocation for coarse DOF map has failed.\n");
      exit(0);
      return;
  }
  coarse_model->dof_material_map  = (cudapcgMap_t *) malloc(sizeof(cudapcgMap_t)*coarse_model->m_nelem);
  if (coarse_model->dof_material_map == NULL){
      printf(" ERROR: Memory allocation for coarse image-based mesh has failed.\n");
      exit(0);
      return;
  }

	// Assemble coarse element material map from original model
	unsigned int idx, idx0, j, k;
	#pragma omp parallel for private(j, k, idx, idx0)
	for (unsigned int i=0; i<(coarse_model->m_ny-1); i++){
		for (j=0; j<(coarse_model->m_nx-1); j++){
			for (k=0; k<((coarse_model->m_nz-1)*(coarse_model->m_nz>0)+(coarse_model->m_nz<1)); k++){
				idx  = (2*i)+(2*j)*(original_model->m_ny-1)+(2*k)*(original_model->m_nx-1)*(original_model->m_ny-1);
				idx0 = i+j*(coarse_model->m_ny-1)+k*(coarse_model->m_nx-1)*(coarse_model->m_ny-1);
				coarse_model->elem_material_map[idx0] = original_model->elem_material_map[idx];
			}
		}
	}
	
	return;
}
//------------------------------------------------------------------------------
