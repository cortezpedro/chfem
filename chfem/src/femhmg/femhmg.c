#include "femhmg.h"
#include "physical_phenomena/femhmg_thermal_2D.h"
#include "physical_phenomena/femhmg_thermal_3D.h"
#include "physical_phenomena/femhmg_elastic_2D.h"
#include "physical_phenomena/femhmg_elastic_3D.h"
#include "physical_phenomena/femhmg_fluid_2D.h"
#include "physical_phenomena/femhmg_fluid_3D.h"

//---------------------------------
///////////////////////////////////
//////////// GLOBALS //////////////
///// (FOR INTERNAL USE ONLY) /////
///////////////////////////////////
//---------------------------------

hmgModel_t * hmgModel = NULL;

char report_buffer[2048]; // 2 KB (should always be enough)

double time_total;

// These are auxiliary. Might organize in a struct later.
unsigned int assemblyfree_strategy_flag=CUDAPCG_NBN, solver_flag=CUDAPCG_CG_SOLVER, preconditioner_flag=CUDAPCG_TRUE, default_poremap_flag=CUDAPCG_POREMAP_NUM, xreduce_flag = CUDAPCG_XREDUCE_FULL;
logical stopCrit_flag=CUDAPCG_L2_NORM;
double xreduce_scale_factor=0.0001;

//---------------------------------
///////////////////////////////////
//////// PRIVATE FUNCTIONS ////////
////////// (DECLARATIONS) /////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
logical readData(char * filename);
logical readMaterialMap(char * filename, uint8_t* npdata);
logical readMaterialMapNF(char * filename);
logical readMaterialMapRAW(char * filename);
logical readMaterialMapNumpy(uint8_t* npdata);
logical readScalarDensityFieldMap(char * filename);
logical setAnalysisType();
void free_model_arrs(hmgModel_t * model);
void assemble_coarse_copy(hmgModel_t * coarse_model, hmgModel_t * original_model);
void saveFields();
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

  (*ptr)->name = (char *)malloc(sizeof(char)*(strlen(model->neutralFile_noExt)+1));
  strcpy((*ptr)->name,model->neutralFile_noExt);

  (*ptr)->parStrategy_flag = assemblyfree_strategy_flag;
  (*ptr)->poremap_flag = model->poremap_flag;
  (*ptr)->parametric_density_field_flag = ( model->sdfFile == NULL || model->m_analysis_flag == HMG_FLUID ) ? CUDAPCG_FALSE : CUDAPCG_TRUE;

  (*ptr)->nrows = model->m_ny-1;
  (*ptr)->ncols = model->m_nx-1;
  (*ptr)->nlayers = (model->m_nz-1)*(model->m_nz>0);

  (*ptr)->nelem = model->m_nelem;
  (*ptr)->nvars = model->m_ndof;
  (*ptr)->nvarspernode = HMG_FLUID ? model->m_dim_flag : (model->m_ndof / model->m_nelem);

  (*ptr)->nkeys = model->m_analysis_flag == HMG_FLUID ? 1 : model->m_nmat;
  (*ptr)->localmtxdim = model->m_lclMtx_dim;
  (*ptr)->nporenodes = model->m_nVelocityNodes;
  (*ptr)->nbordernodes = model->m_nBorderNodes;

  (*ptr)->nhmgvars = model->m_analysis_flag == HMG_FLUID ? model->m_nVelocityNodes*model->m_dim_flag : model->m_ndof;

  return CUDAPCG_TRUE;
}
//------------------------------------------------------------------------------
void estimate_memory(hmgModel_t *model, size_t *mem_maps, size_t *mem_solver){
  if (model==NULL || mem_maps==NULL || mem_solver==NULL) return;
  
  *mem_maps=0;
  *mem_solver=0;
  
  // map requirements
  if (model->m_analysis_flag == HMG_THERMAL || model->m_analysis_flag == HMG_ELASTIC){
   if (model->sdfFile==NULL) *mem_maps += model->m_nelem*sizeof(cudapcgMap_t);
   else                      *mem_maps += model->m_nelem*sizeof(parametricScalarField_t);
  } else if (model->m_analysis_flag == HMG_FLUID) {
    *mem_maps += model->m_nelem*sizeof(cudapcgIdMap_t); // image-to-mesh map
    if (model->poremap_flag == CUDAPCG_POREMAP_NUM){
      *mem_maps += (model->m_nVelocityNodes+model->m_nBorderNodes)*sizeof(cudapcgIdMap_t); // mesh-to-image map
      *mem_maps += model->m_nBorderNodes*sizeof(uint8_t); // borderkeys
    } else *mem_maps += model->m_nelem*sizeof(cudapcgFlag_t);
  } else return; // invalid analysis flag
  
  // solver requirements: vectors
  unsigned int n_vectors;
  if      (solver_flag == CUDAPCG_CG_SOLVER)      n_vectors=4;
  else if (solver_flag == CUDAPCG_MINRES_SOLVER)  n_vectors=5;
  else if (solver_flag <= CUDAPCG_MINRES3_SOLVER) n_vectors=3;
  else if (solver_flag <= CUDAPCG_MINRES2_SOLVER) n_vectors=2;
  else return; // invalid solver_flag
  *mem_solver = n_vectors*model->m_ndof*sizeof(cudapcgVar_t);
  
  // check if xreduce has to be considered
  if (model->m_analysis_flag == HMG_FLUID && solver_flag >= CUDAPCG_CG2_SOLVER && xreduce_flag != CUDAPCG_XREDUCE_NONE){
    *mem_solver += (xreduce_flag == CUDAPCG_XREDUCE_FULL ? model->m_dim_flag : 1)
                    * model->m_nVelocityNodes*sizeof(cudapcgVar_t);
  }
  
  // auxiliary reduce vector (admitting 128 threads per block)
  // 1% extra to account for second vector (approximation)
  *mem_solver += (size_t)(1.01*((double)(CEIL(model->m_ndof,128)*sizeof(cudapcgVar_t))));
  
  return;
}
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
//////// PUBLIC FUNCTIONS /////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
logical hmgInit(char * data_filename, char * elem_filename, char * sdf_filename, uint8_t* npdata){

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
  hmgModel->sdfFile = sdf_filename;

  // Store filename for header .nf without extension
  unsigned long nf_strlen = strlen(data_filename);
  hmgModel->neutralFile_noExt = (char *)malloc(sizeof(char)*(nf_strlen+1));
  strcpy(hmgModel->neutralFile_noExt,data_filename);
  hmgModel->neutralFile_noExt[nf_strlen-3] = '\0';

  // Init auxiliary flags
  hmgModel->m_hmg_flag_was_set = HMG_FALSE;
  hmgModel->m_hmg_flag = HOMOGENIZE_ALL;
  hmgModel->m_using_x0_flag = HMG_FALSE;
  hmgModel->m_exportX_flag = HMG_FALSE;
  hmgModel->m_hmg_thermal_expansion_flag = HMG_FALSE;
  
  hmgModel->m_saveFields_flag = HMG_FALSE;
  hmgModel->m_fieldsByElem_flag = 0;

  hmgModel->m_scalar_field_data_type_flag = HMG_FLOAT32;

  hmgModel->poremap_flag = default_poremap_flag;
  
  hmgModel->x0File = NULL;
  hmgModel->x0 = NULL;
  
  hmgModel->node_dof_map = NULL;
  hmgModel->elem_material_map = NULL;
  hmgModel->dof_material_map = NULL;
  hmgModel->density_map = NULL;
  hmgModel->dof_id_map = NULL;
  hmgModel->dof_fluid_map = NULL;
  hmgModel->Mtxs = NULL;
  hmgModel->CB = NULL;
  hmgModel->RHS = NULL;
  hmgModel->C = NULL;
  hmgModel->thermal_expansion = NULL;
  hmgModel->pore_border_fluidkeys = NULL;
  hmgModel->pore_dof2node_map = NULL;
  hmgModel->report = NULL;

  // Read data input file
  if (!readData(data_filename))
    return HMG_FALSE;
    
  // Check if user provided scalar density field (.bin) and avoid unsupported functionalities
  if (hmgModel->sdfFile!=NULL){
    if (hmgModel->m_analysis_flag == HMG_FLUID){
      printf("WARNING: Scalar density field input is not supported for permeability analysis.\n");
      hmgModel->sdfFile = NULL;
    } else {
      assemblyfree_strategy_flag = CUDAPCG_EBE;
    }
  }

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
  if (!readMaterialMap(elem_filename, npdata)){
    free(hmgModel->elem_material_map);
    free(hmgModel);
    return HMG_FALSE;
  }
  
  // Deal with scalar density field
  if (hmgModel->sdfFile != NULL){
  
    // Allocate scalar density field array
    hmgModel->density_map = (parametricScalarField_t *)malloc(sizeof(parametricScalarField_t)*hmgModel->m_nelem);
    if (hmgModel->density_map == NULL){
        printf("ERROR: Memory allocation for scalar density field array has failed.\n");
        free(hmgModel->elem_material_map);
        free(hmgModel);
        return HMG_FALSE;
    }
    
    // Read sdf map input file
    if (!readScalarDensityFieldMap(sdf_filename)){
      free(hmgModel->elem_material_map);
      free(hmgModel->density_map);
      free(hmgModel);
      return HMG_FALSE;
    }
    
    // Reset normalized properties (obs: invalid if FLUID)
    if (hmgModel->m_analysis_flag == HMG_THERMAL){
      hmgModel->m_nmat=1; // scalar field with varying normalized conductivity matrix
      hmgModel->props[0]=1.0;
    } else {// if (hmgModel->m_analysis_flag == HMG_ELASTIC){
      for (int j=0; j<hmgModel->m_nmat; j++) hmgModel->props[2*j]=1.0; // normalize E. Still might have different poisson ratios
    }
    
  } else {
    hmgModel->density_map = NULL; // to be sure
  }

  step_count=0, num_of_steps=3*(hmgModel->m_analysis_flag != HMG_FLUID)+4*(hmgModel->m_analysis_flag == HMG_FLUID);
  printf("\r    Assembling maps and local matrices...[%3d%%]",(step_count*100)/num_of_steps);

  // Allocate hmgModel->node_dof_map array
  hmgModel->node_dof_map = (unsigned int *) malloc(sizeof(unsigned int)*hmgModel->m_nnode);
  if (hmgModel->node_dof_map == NULL){
      printf("ERROR: Memory allocation for DOF map has failed.\n");
      free(hmgModel->elem_material_map);
      if (hmgModel->density_map)  free(hmgModel->density_map);
      free(hmgModel);
      return HMG_FALSE;
  }

  // Assemble hmgModel->node_dof_map
  hmgModel->assembleNodeDofMap(hmgModel);
  printf("\r    Assembling maps and local matrices...[%3d%%]",((++step_count)*100)/num_of_steps);

  // Allocate hmgModel->dof_material_map array
  if (hmgModel->sdfFile==NULL){
    if (hmgModel->m_analysis_flag != HMG_FLUID){
      hmgModel->dof_material_map = (cudapcgMap_t *) malloc(sizeof(cudapcgMap_t)*hmgModel->m_ndof/hmgModel->m_nnodedof);
      hmgModel->dof_fluid_map = NULL;
      if (hmgModel->dof_material_map == NULL){
          printf("ERROR: Memory allocation for DOF material map has failed.\n");
          free(hmgModel->node_dof_map);
          free(hmgModel->elem_material_map);
          if (hmgModel->density_map) free(hmgModel->density_map);
          free(hmgModel);
          return HMG_FALSE;
      }
    } else {
      hmgModel->dof_material_map = NULL;
      hmgModel->dof_fluid_map = (cudapcgFlag_t *) malloc(sizeof(cudapcgFlag_t)*hmgModel->m_nelem);
      if (hmgModel->dof_fluid_map == NULL){
          printf("ERROR: Memory allocation for pore map has failed.\n");
          free(hmgModel->node_dof_map);
          free(hmgModel->elem_material_map);
          if (hmgModel->density_map) free(hmgModel->density_map);
          free(hmgModel);
          return HMG_FALSE;
      }
    }
    
    // Assemble hmgModel->dof_material_map
    hmgModel->assembleDofMaterialMap(hmgModel);
    
  } else {
    hmgModel->dof_material_map = NULL;
    hmgModel->dof_fluid_map = NULL;
  }
    
  printf("\r    Assembling maps and local matrices...[%3d%%]",((++step_count)*100)/num_of_steps);

  // Assmble dof_id_map (used in permeability analysis)
  if (hmgModel->m_analysis_flag == HMG_FLUID){
    hmgModel->dof_id_map = (cudapcgIdMap_t *) malloc(sizeof(cudapcgIdMap_t)*hmgModel->m_nelem);
    if (hmgModel->dof_id_map == NULL){
        printf("ERROR: Memory allocation for pore DOF id map has failed.\n");
        free(hmgModel->node_dof_map);
        free(hmgModel->elem_material_map);
        if (hmgModel->dof_material_map)  free(hmgModel->dof_material_map);
        if (hmgModel->dof_fluid_map)     free(hmgModel->dof_fluid_map);
        if (hmgModel->density_map)       free(hmgModel->density_map);
        free(hmgModel);
        return HMG_FALSE;
    }
    hmgModel->assembleDofIdMap(hmgModel);
    printf("\r    Assembling maps and local matrices...[%3d%%]",((++step_count)*100)/num_of_steps);
  } else {
    hmgModel->dof_id_map = NULL;
  }

  // Allocate local FEM matrices array
  if (hmgModel->m_analysis_flag != HMG_FLUID){
    hmgModel->Mtxs = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*hmgModel->m_lclMtx_dim*hmgModel->m_nmat);
    hmgModel->CB   = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*hmgModel->m_lclCB_dim*hmgModel->m_nmat);
  } else {
    hmgModel->Mtxs = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*hmgModel->m_lclMtx_dim);
    hmgModel->CB   = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*hmgModel->m_lclCB_dim);
  }
  if (hmgModel->Mtxs == NULL || hmgModel->CB == NULL){
      printf("ERROR: Memory allocation for local matrices has failed.\n");
      free(hmgModel->dof_id_map);
      free(hmgModel->node_dof_map);
      free(hmgModel->elem_material_map);
      if (hmgModel->dof_material_map)  free(hmgModel->dof_material_map);
      if (hmgModel->dof_fluid_map)     free(hmgModel->dof_fluid_map);
      if (hmgModel->density_map)       free(hmgModel->density_map);
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
      free(hmgModel->dof_id_map);
      free(hmgModel->node_dof_map);
      free(hmgModel->elem_material_map);
      if (hmgModel->dof_material_map)  free(hmgModel->dof_material_map);
      if (hmgModel->dof_fluid_map)     free(hmgModel->dof_fluid_map);
      if (hmgModel->density_map)       free(hmgModel->density_map);
      free(hmgModel);
      return HMG_FALSE;
  }
  
  if (hmgModel->m_hmg_thermal_expansion_flag == HMG_TRUE){
    hmgModel->thermal_expansion = (var *)malloc(sizeof(var)*6); // largest case (3D)
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
logical hmgEnd(var **eff_coeff){
  if (hmgModel == NULL) return HMG_FALSE;

  // Check if x0 matrix should be freed
  if (hmgModel->m_using_x0_flag && hmgModel->x0 != NULL){
    for (unsigned int i=0; i<6; i++){
      if (hmgModel->x0[i] != NULL){
        free(hmgModel->x0[i]);
        hmgModel->x0[i] = NULL;
      }
    }
    free(hmgModel->x0);
    hmgModel->x0 = NULL;
  }

  if (eff_coeff != NULL && hmgModel->C != NULL) {
    *eff_coeff = hmgModel->C; // Save the address of C before freeing
  }

  // Free dynamic arrays from memory
  if (hmgModel->elem_material_map)
    free(hmgModel->elem_material_map);  hmgModel->elem_material_map = NULL;
  free(hmgModel->node_dof_map);       hmgModel->node_dof_map = NULL;
  if (hmgModel->dof_material_map)
    free(hmgModel->dof_material_map); hmgModel->dof_material_map = NULL;
  if (hmgModel->dof_fluid_map)
    free(hmgModel->dof_fluid_map);    hmgModel->dof_fluid_map = NULL;
  if (hmgModel->dof_id_map)
    free(hmgModel->dof_id_map);       hmgModel->dof_id_map = NULL;
  if (hmgModel->density_map)
    free(hmgModel->density_map);      hmgModel->density_map = NULL;
  free(hmgModel->Mtxs);               hmgModel->Mtxs = NULL;
  free(hmgModel->CB);                 hmgModel->CB = NULL;
  hmgModel->C = NULL; // free(hmgModel->C); this deletes first element of eff_coeff
  
  if (hmgModel->thermal_expansion)
  free(hmgModel->thermal_expansion);  hmgModel->thermal_expansion = NULL;

  if (hmgModel->pore_border_fluidkeys)
  	free(hmgModel->pore_border_fluidkeys); hmgModel->pore_border_fluidkeys = NULL;
  if (hmgModel->pore_dof2node_map)
  	free(hmgModel->pore_dof2node_map);     hmgModel->pore_dof2node_map = NULL;

  // free report
  reportFree(hmgModel->report,REPORT_TRUE); hmgModel->report = NULL;

  // free dynamic string
  free(hmgModel->neutralFile_noExt); hmgModel->neutralFile_noExt = NULL;

  // Free model struct
  free(hmgModel);
  hmgModel = NULL;

  // Compute total elapsed time
  time_total = omp_get_wtime()-time_total;
  printf("Elapsed time (total): %.2e s\n",time_total);

  return HMG_TRUE;
}
//------------------------------------------------------------------------------
void hmgSetXReduceFlag(cudapcgFlag_t flag){
  xreduce_flag = flag <= CUDAPCG_XREDUCE_FULL ? flag : CUDAPCG_XREDUCE_FULL;
  return;
}
//------------------------------------------------------------------------------
void hmgSetParallelStrategyFlag(cudapcgFlag_t flag){
  assemblyfree_strategy_flag = hmgModel->sdfFile==NULL ? flag : CUDAPCG_EBE; // safety measure for now
  return;
}
//------------------------------------------------------------------------------
void hmgSetSolverFlag(cudapcgFlag_t flag){
  solver_flag = flag;
  return;
}
//------------------------------------------------------------------------------
void hmgSetPreConditionerFlag(cudapcgFlag_t flag){
  preconditioner_flag = (flag > 0) ? CUDAPCG_TRUE : CUDAPCG_FALSE;
  return;
}
//------------------------------------------------------------------------------
void hmgSetStoppingCriteria(cudapcgFlag_t flag){
  stopCrit_flag = flag;
  return;
}
//------------------------------------------------------------------------------
logical hmgSetPoreMappingStrategy(cudapcgFlag_t flag){

  if (hmgModel == NULL){
    default_poremap_flag = flag;
    return HMG_TRUE;
  }

  if (flag == hmgModel->poremap_flag || flag > CUDAPCG_POREMAP_NUM) return HMG_TRUE;

  if (hmgModel->m_analysis_flag != HMG_FLUID){
    hmgModel->poremap_flag = flag;
    return HMG_TRUE;
  }

  if (flag == CUDAPCG_POREMAP_NUM){
    free(hmgModel->elem_material_map);
    hmgModel->elem_material_map = NULL;
    hmgModel->assembleDofIdMap(hmgModel);
  } else { // if (flag == CUDAPCG_POREMAP_IMG)
    free(hmgModel->pore_dof2node_map);
    hmgModel->pore_dof2node_map = NULL;
    free(hmgModel->pore_border_fluidkeys);
    hmgModel->pore_border_fluidkeys = NULL;
    if (hmgModel->elem_material_map == NULL)
      hmgModel->elem_material_map = (cudapcgMap_t *) malloc(sizeof(cudapcgMap_t)*hmgModel->m_nelem);
    if (hmgModel->elem_material_map == NULL){
      printf("ERROR: Failed to allocate memory for image.\n");
      return HMG_FALSE;
    }
    if (!readMaterialMap(hmgModel->imageFile, NULL)){
      printf("ERROR: Failed to read image from %s.\n",hmgModel->imageFile);
      return HMG_FALSE;
    }
  }

  hmgModel->poremap_flag = flag;
  return HMG_TRUE;
}
//------------------------------------------------------------------------------
void hmgSetHomogenizationFlag(hmgFlag_t flag){
  if (hmgModel == NULL) return;

  if (flag > HOMOGENIZE_XY){
    hmgModel->m_hmg_flag_was_set = HMG_FALSE;
    return;
  }
  if (hmgModel->m_analysis_flag == HMG_THERMAL || hmgModel->m_analysis_flag == HMG_FLUID){
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
void hmgExportX(logical flag){
  if (hmgModel == NULL) return;
  hmgModel->m_exportX_flag = flag > 0 ? HMG_TRUE : HMG_FALSE;
  return;
}
//------------------------------------------------------------------------------
void hmgImportX(char *file){
  if (hmgModel == NULL) return;
  hmgModel->x0File = file;
  hmgModel->m_using_x0_flag = HMG_TRUE;
  return;
}
//------------------------------------------------------------------------------
void hmgFindInitialGuesses(unsigned int nlevels){
  if (!nlevels || hmgModel==NULL)
    return;

  // Check if analysis is of permeability (not supported for now)
  if (hmgModel->m_analysis_flag == HMG_FLUID){
    printf("WARNING: Initial guess estimation is not yet available for permeability analysis.\n");
    printf("         Null initial guess will be considered (x0 = [0]).\n");
    return;
  }

  // Check if performing analysis with scalar density field input (not supported for now)
  if (hmgModel->sdfFile != NULL){
    printf("WARNING: Initial guess estimation is not yet available for analyses with a scalar field input (.bin).\n");
    printf("         Null initial guess will be considered (x0 = [0]).\n");
    return;
  }

  // Check if solver is for permeability only
  if (solver_flag > CUDAPCG_MINRES3_SOLVER){
    printf("WARNING: CG2 and MINRES2 are only available for FLUID.\n");
    printf("         Defaulting to %s.\n", solver_flag == CUDAPCG_CG2_SOLVER ? "CG3" :  "MINRES3" );
    solver_flag = solver_flag == CUDAPCG_CG2_SOLVER ? CUDAPCG_CG3_SOLVER :  CUDAPCG_MINRES3_SOLVER;
  }

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
  } else if (hmgModel->m_analysis_flag == HMG_THERMAL){ // || hmgModel->m_analysis_flag == HMG_FLUID
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
  cudapcgVar_t * x  = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*model_arr[1]->m_ndof);
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
      // Set solver flag (ternary operator to switch between MParPCG or MINRES)
      cudapcgSetSolver( (solver_flag == CUDAPCG_MINRES_SOLVER || solver_flag == CUDAPCG_MINRES3_SOLVER) ? CUDAPCG_MINRES_SOLVER : CUDAPCG_CG_SOLVER);
      cudapcgSetPreconditioner(preconditioner_flag);
      // Set residual norm flag
      cudapcgSetResNorm(0); // L2
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
      if (assemblyfree_strategy_flag==CUDAPCG_NBN)
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
          #pragma omp parallel for
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
    free_model_arrs(model_arr[i]);
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

  // Check if analysis is not of permeability and solver is for permeability only
  if (hmgModel->m_analysis_flag != HMG_FLUID && solver_flag > CUDAPCG_MINRES3_SOLVER){
    printf("WARNING: CG2 and MINRES2 are only available for FLUID.\n");
    printf("         Defaulting to %s.\n", solver_flag == CUDAPCG_CG2_SOLVER ? "CG3" :  "MINRES3" );
    solver_flag = solver_flag == CUDAPCG_CG2_SOLVER ? CUDAPCG_CG3_SOLVER :  CUDAPCG_MINRES3_SOLVER;
  }

  // Buffer variables to measure wall time
  double t, t_init_guess, t_hmg;

  // Auxiliary string to ensure proper indentation when printing solver report
  char str_indentation[] = "    | ";

  // Auxiliary simulation id static arrays
  const char sim_str[6][11] = { {"X"},{"Y"},{"Z"},{"YZ (shear)"},{"XZ (shear)"},{"XY (shear)"} };
  unsigned int sim_id[6] = {0,0,0,0,0,0};
  unsigned int sim_sz;
  if (hmgModel->m_hmg_flag_was_set){
    sim_sz = 1;
    sim_id[0] = hmgModel->m_hmg_flag;
  } else if (hmgModel->m_analysis_flag == HMG_THERMAL || hmgModel->m_analysis_flag == HMG_FLUID){
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

  // Initialize C with zeros
  for (unsigned int i = 0; i<hmgModel->m_C_dim; i++)
    hmgModel->C[i] = 0.0;


  // Append text report
  if (hmgModel->m_report_flag)
    reportAppend(hmgModel->report, "Running homogenization...\n");

  t_hmg = omp_get_wtime();
  printf("Running homogenization...\n");

  t = omp_get_wtime();
  printf("    Initializing solver...");

  // Init cudapcg api
  if(!cudapcgInit(hmgModel->m_pcg_flag, assemblyfree_strategy_flag)){
    return;
  }

  // Set xreduce flag
  cudapcgSetXReduceMode(xreduce_flag);
  
  // Set xreduce scale
  cudapcgSetXReduceScale( (hmgModel->m_elem_size*hmgModel->m_elem_size) / (double) (hmgModel->m_nelem) );

  // Set xreduce stabilization factor (default=0.0001)
  cudapcgSetReduceStabFactor(xreduce_scale_factor);

  // Set indentation header to print metrics
  cudapcgSetHeaderString(str_indentation);

  // Set model constructor fcn cudapcg API and build model
  cudapcgSetModelConstructorFcn(cudapcgModel_constructor);
  cudapcgBuildModel(hmgModel);

  // Set solver flag
  if(!cudapcgSetSolver(solver_flag)){
    cudapcgEnd();
    return;
  }
  if(!cudapcgSetPreconditioner(preconditioner_flag)){
    cudapcgEnd();
    return;
  }

  // Set residual norm flag
  if(!cudapcgSetResNorm(stopCrit_flag)){
    cudapcgEnd();
    return;
  }

  // Set numeric tol and max num of iterations
  cudapcgSetNumTol(hmgModel->m_num_tol);
  cudapcgSetMaxIterations(hmgModel->m_max_iterations);

  // Provide material map
  if (hmgModel->sdfFile==NULL){
    if (hmgModel->m_analysis_flag != HMG_FLUID){
      if (assemblyfree_strategy_flag==CUDAPCG_NBN)
        cudapcgSetImage(hmgModel->dof_material_map);
      else
        cudapcgSetImage(hmgModel->elem_material_map);
    } else {
        cudapcgSetPeriodic2DOFMap(hmgModel->dof_id_map);
        if (hmgModel->poremap_flag == CUDAPCG_POREMAP_IMG)
          cudapcgSetPoreMap(hmgModel->dof_fluid_map);
        else if (hmgModel->poremap_flag == CUDAPCG_POREMAP_NUM){
          cudapcgSetPoreMap(hmgModel->pore_border_fluidkeys);
          cudapcgSetDOF2PeriodicMap(hmgModel->pore_dof2node_map);
        }
    }
  } else {
    if (hmgModel->m_analysis_flag == HMG_THERMAL){
      cudapcgSetParametricDensityField(hmgModel->density_map,hmgModel->density_min,hmgModel->density_max);
    } else if (hmgModel->m_analysis_flag == HMG_ELASTIC){
      cudapcgSetParametricDensityField(hmgModel->density_map,hmgModel->density_min,hmgModel->density_max);
      cudapcgSetImage(hmgModel->elem_material_map);
    }
  }

  // Provide local FEM matrices
  cudapcgSetLclMtxs(hmgModel->Mtxs);

  // Allocate RHS array
  hmgModel->RHS = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*hmgModel->m_ndof);
  if (hmgModel->RHS==NULL){
    printf(" ERROR: Memory allocation for right-hand side array has failed.\n");
    cudapcgEnd();
    exit(0);
    return;
  }

  // Allocate x array
  cudapcgVar_t *x = (cudapcgVar_t *) malloc(sizeof(cudapcgVar_t)*(solver_flag < CUDAPCG_CG2_SOLVER ? hmgModel->m_ndof : hmgModel->m_C_dim));
  if (x==NULL){
    printf(" ERROR: Memory allocation for solution array has failed.\n");
    cudapcgEnd();
    exit(0);
    return;
  }

  // Allocate PCG arrays
  cudapcgAllocateArrays();

  // Initialize pointer to files
  FILE *file;
  char file_buffer[1024];
  cudapcgVar_t * x0_fromfile = NULL;
  if (hmgModel->m_using_x0_flag && hmgModel->x0File != NULL){
    x0_fromfile = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*hmgModel->m_ndof);
  }

  // Measuirng time for solver initialization
  t = omp_get_wtime()-t;
  printf("Done.(%.2e s)\n",t);

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
      if (hmgModel->x0File == NULL){
          printf("%sInterpolating initial guess...",str_indentation);
          cudapcgSetX0(hmgModel->x0[hmgModel->m_hmg_flag],CUDAPCG_TRUE);
      } else {
      sprintf(file_buffer,"%s_%d.bin",hmgModel->x0File,hmgModel->m_hmg_flag);
      printf("%sReading initial guess from %s...",str_indentation,file_buffer);
      file = fopen(file_buffer,"rb");
      if (!file){
        printf("Failed!...");
        cudapcgSetX0(NULL,CUDAPCG_FALSE);
      } else {
        if (fread(x0_fromfile,sizeof(cudapcgVar_t),hmgModel->m_ndof,file) < hmgModel->m_ndof) printf("WARNING: Failed to read all %d items from %s...",hmgModel->m_ndof,file_buffer);
        cudapcgSetX0(x0_fromfile,CUDAPCG_FALSE);
      }
      fclose(file);
      }
      t_init_guess = omp_get_wtime()-t_init_guess;
      printf("Done.(%.2e s)\n",t_init_guess);
    }
    if (solver_flag > CUDAPCG_MINRES3_SOLVER) cudapcgSetXReduceShift(hmgModel->m_hmg_flag <= HMG_DIR_Z ? hmgModel->m_hmg_flag : HMG_DIR_X);
    cudapcgSolve(x); t = omp_get_wtime()-t;
    cudapcgPrintSolverMetrics();
    if (hmgModel->m_report_flag){
      cudapcgPrintSolverReport2(report_buffer);
      reportAppend(hmgModel->report, report_buffer);
    }
    printf("    Done.(%.2e s)\n",t);

    if (hmgModel->m_exportX_flag){
      sprintf(file_buffer,"%s_X_%d.bin",hmgModel->neutralFile_noExt,hmgModel->m_hmg_flag);
      file = fopen(file_buffer,"wb");
      if (file){
        fwrite(x,sizeof(cudapcgVar_t)*hmgModel->m_ndof,1,file);
      }
      fclose(file);
    }
    
    // Update C
    if (solver_flag >= CUDAPCG_CG2_SOLVER){
      printf("    Effective fields:\n");
      for (int jj=0; jj<hmgModel->m_dim_flag; jj++){
        //x[jj] *= (hmgModel->m_elem_size * hmgModel->m_elem_size) / (double) hmgModel->m_nelem;
        printf("        %.8e\n",x[jj]);
        hmgModel->C[jj*hmgModel->m_dim_flag+hmgModel->m_hmg_flag] = x[jj];
      }
    } else {
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

    // Output fields from simulation (if required)
    if (hmgModel->m_saveFields_flag){
      t = omp_get_wtime();
      printf("    Exporting simulation results...");
      hmgModel->saveFields(hmgModel,x);
      t = omp_get_wtime()-t;
      printf("Done.(%.2e s)\n",t);
    }
  }

  if (hmgModel->m_hmg_thermal_expansion_flag == HMG_TRUE){
    if(hmgModel->m_analysis_flag == HMG_ELASTIC){
      hmgModel->m_hmg_flag = HOMOGENIZE_THERMAL_EXPANSION;
      
      sprintf(
        report_buffer,
        "    ------------------------------------------------------\n"\
        "    Thermal expansion analysis:\n"
      );
      if (hmgModel->m_report_flag)
        reportAppend(hmgModel->report, report_buffer);
      printf("%s",report_buffer);
      
      t = omp_get_wtime();
      printf("    Assembling RHS...");
      hmgModel->assembleRHS(hmgModel);
      cudapcgSetRHS(hmgModel->RHS);
      t = omp_get_wtime()-t;
      printf("Done.(%.2e s)\n",t);
      
      t = omp_get_wtime();
      strcpy(report_buffer,"    Solving system of equations...\n");
      if (hmgModel->m_report_flag)
        reportAppend(hmgModel->report, report_buffer);
      printf("%s",report_buffer);
      cudapcgSolve(x); t = omp_get_wtime()-t;
      cudapcgPrintSolverMetrics();
      if (hmgModel->m_report_flag){
        cudapcgPrintSolverReport2(report_buffer);
        reportAppend(hmgModel->report, report_buffer);
      }
      printf("    Done.(%.2e s)\n",t);
      
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
    } else {
      printf("    WARNING: Thermal expansion analysis only available for ELASTIC.\n");
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

  if (x0_fromfile) free(x0_fromfile);

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
  } else { // if (hmgModel->m_analysis_flag == HMG_FLUID)
      if (hmgModel->m_dim_flag == HMG_2D)
      strcpy(str_sim,"FLUID_2D");
    else //if (hmgModel->m_dim_flag == HMG_3D)
      strcpy(str_sim,"FLUID_3D");
  }

  const char str_hmg[7][11] = { {"X"},{"Y"},{"Z"},{"YZ (shear)"},{"XZ (shear)"},{"XY (shear)"},{"ALL"} };
  const char str_solver[6][8] = { {"CG"},{"MINRES"},{"CG3"},{"MINRES3"},{"CG2"},{"MINRES2"} };
  char xreduce_str_buffer[32];
  sprintf(xreduce_str_buffer,"xreduce stab factor: %.2e\n",xreduce_scale_factor);
  char porosity_str_buffer[128];
  if (hmgModel->m_analysis_flag == HMG_FLUID){
    sprintf(
      porosity_str_buffer,
      "  | porosity: %05.2f%%\n",
      100.0*(((double) hmgModel->m_nporeelem) / hmgModel->m_nelem)
    );
  }
  char fluidnodes_str_buffer[256];
  if (hmgModel->m_analysis_flag == HMG_FLUID){
    sprintf(
      fluidnodes_str_buffer,
      "  | within pores: %u\n"\
      "  | at interfaces: %u\n",
      hmgModel->m_nVelocityNodes,
      hmgModel->m_nBorderNodes
    );
  }

  sprintf(
    report_buffer,
    "*******************************************************\n"\
    "MODEL DATA:\n"\
    "Analysis: %s\n"\
    "Homogenization on direction: %s\n"\
    "Number of voxels (x,y,z): [%u,%u,%u]\n"\
    "Number of voxels (total): %u\n"\
    "Number of vertices: %u\n"\
    "Number of elements: %u\n"\
    "%s"\
    "Number of nodes: %u\n"\
    "%s"\
    "Number of DOFs: %u\n"\
    "Number of materials: %u\n"\
    "Numerical tolerance: %.3e\n"\
    "Max iterations: %u\n"\
    "Solver: %s%s\n"\
    "%s"\
    "*******************************************************\n",
    str_sim,
    str_hmg[hmgModel->m_hmg_flag],
    hmgModel->m_nx-1,hmgModel->m_ny-1,hmgModel->m_nz-(hmgModel->m_nz>0),
    hmgModel->m_nelem, // number of voxels   (at this point, m_nelem assumes the mesh is monolithic)
    hmgModel->m_nnode, // number of vertices (at this point, m_nnode assumes the mesh is monolithic)
    hmgModel->m_analysis_flag != HMG_FLUID ? hmgModel->m_nelem : hmgModel->m_nporeelem,
    hmgModel->m_analysis_flag != HMG_FLUID ? "" : porosity_str_buffer,
    hmgModel->m_analysis_flag != HMG_FLUID ? (hmgModel->m_ndof / hmgModel->m_nnodedof) : (hmgModel->m_nVelocityNodes+hmgModel->m_nBorderNodes),
    hmgModel->m_analysis_flag != HMG_FLUID ? "" : fluidnodes_str_buffer,
    hmgModel->m_ndof,
    hmgModel->m_nmat,
    hmgModel->m_num_tol,
    hmgModel->m_max_iterations,
    preconditioner_flag ? "P" : "", str_solver[solver_flag],
    solver_flag < CUDAPCG_CG2_SOLVER ? "" : xreduce_str_buffer 
  );

  if (hmgModel->m_report_flag)
    reportAppend(hmgModel->report, report_buffer);
  printf("%s",report_buffer);
  
  size_t size_of_maps_MB=0, size_of_solver_MB=0;
  estimate_memory(hmgModel,&size_of_maps_MB,&size_of_solver_MB);
  
  size_of_maps_MB   /= 1000000;
  size_of_solver_MB /= 1000000;
  
  // memory estimates
  sprintf(
    report_buffer,
    "GPU MEMORY REQUIREMENTS:\n"\
    "  + %zuMB [maps]\n"\
    "  + %zuMB [solver]\n"\
    "  + ~100MB [O(1) overhead]\n"\
    "--------------------------\n"\
    "  %.3lfGB\n"\
    "*******************************************************\n",
    size_of_maps_MB,
    size_of_solver_MB,
    ((double) (size_of_maps_MB+size_of_solver_MB+100))*0.001
  );

  if (hmgModel->m_report_flag)
    reportAppend(hmgModel->report, report_buffer);
  printf("%s",report_buffer);

  unsigned int flag = assemblyfree_strategy_flag;
  if (hmgModel->m_analysis_flag == HMG_FLUID && flag != 0){
     printf("WARNING: Permeability analysis is only available with Node-by-node strategy. Provided flag was ignored.\n");
     flag = 0;
  }
  printf("Parallel assembly on-the-fly strategy: ");
  if (flag == CUDAPCG_NBN)
    printf("Node-by-node (%d)\n",flag);
  else if (flag == CUDAPCG_EBE)
    printf("Elem-by-elem (%d)\n",flag);
  else
  	printf("Invalid flag (%d)\n", flag);
  printf("*******************************************************\n");
  return;
}
//------------------------------------------------------------------------------
var *hmgGetConstitutiveMtx(){
  if (hmgModel == NULL) return NULL;
  return hmgModel->C;
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
void hmgSaveFields(logical mustExport_flag, logical byElems_flag){

  if (hmgModel==NULL) return;

  hmgModel->m_saveFields_flag = mustExport_flag;
  hmgModel->m_fieldsByElem_flag = byElems_flag;

  return;
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
  } else if (hmgModel->m_analysis_flag == HMG_FLUID){
    if (hmgModel->m_dim_flag == HMG_2D){
      hmgModel->m_pcg_flag = CUDAPCG_FLUID_2D;
      hmgModel->initModel = initModel_fluid_2D;
    } else if (hmgModel->m_dim_flag == HMG_3D){
      hmgModel->m_pcg_flag = CUDAPCG_FLUID_3D;
      hmgModel->initModel = initModel_fluid_3D;
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
  unsigned int count=0, num_of_info=12;
  logical was_xreduce_stabfactor_set = HMG_FALSE;
  printf("\r    Scanning through neutral file...[%3d%%]",(count*100)/num_of_info);
  while (fscanf(file, "%s", str)!=EOF && count<num_of_info){
    if (!strcmp(str,"%type_of_analysis")){
      if (fscanf(file, "%hu", &(hmgModel->m_analysis_flag))==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
      printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
      if (hmgModel->m_analysis_flag != HMG_THERMAL &&
          hmgModel->m_analysis_flag != HMG_ELASTIC &&
          hmgModel->m_analysis_flag != HMG_FLUID ){
        fclose(file);
        printf("ERROR: Provided analysis flag does not match any possible value.\n");
        return HMG_FALSE;
      }

    } else if (!strcmp(str,"%type_of_solver")){
      // using mat_id as auxiliary variable to read from file. unsigned 8bit
      if(fscanf(file, "%hhu", &mat_id)==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
      hmgSetSolverFlag(mat_id);
      printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
      
    //} else if (!strcmp(str,"%type_of_rhs")){

    } else if (!strcmp(str,"%voxel_size")){
      if(fscanf(file, "%lf", &(hmgModel->m_elem_size))==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
      printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
    } else if (!strcmp(str,"%solver_tolerance")){
      if(fscanf(file, "%lf", &tol_buffer)==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
      hmgModel->m_num_tol = (cudapcgTol_t) tol_buffer;
      printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
    } else if (!strcmp(str,"%number_of_iterations")){
      if(fscanf(file, "%i", &(hmgModel->m_max_iterations))==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
      printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
    } else if (!strcmp(str,"%image_dimensions")){
      if(fscanf(file, "%i %i %i", &(hmgModel->m_nx), &(hmgModel->m_ny), &(hmgModel->m_nz))==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
      printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
      hmgModel->m_nx++;hmgModel->m_ny++;
      if (hmgModel->m_nz){
        hmgModel->m_dim_flag = HMG_3D;
        hmgModel->m_nz++;
      } else {
        hmgModel->m_dim_flag = HMG_2D;
      }
    } else if (!strcmp(str,"%refinement")){
      if(fscanf(file, "%i", &(hmgModel->m_mesh_refinement))==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
      printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
      if (hmgModel->m_mesh_refinement < 1) return HMG_FALSE;
    } else if (!strcmp(str,"%number_of_materials")){
      if(fscanf(file, "%hhu", &(hmgModel->m_nmat))==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
      printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
      if (hmgModel->m_nmat > MAX_COLORNUM){
        fclose(file);
        printf("ERROR: Number of different materials (%d) is greater than max (%d).\n",hmgModel->m_nmat,MAX_COLORKEY);
        return HMG_FALSE;
      }
    } else if (!strcmp(str,"%properties_of_materials")){
      for (unsigned char i=0; i<hmgModel->m_nmat; i++){
        if (hmgModel->m_analysis_flag == HMG_ELASTIC){
          if(fscanf(file, "%hhu %lf %lf", &mat_id, &(hmgModel->props[2*i]), &(hmgModel->props[2*i+1]))==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
        } else if (hmgModel->m_analysis_flag == HMG_THERMAL){
          if(fscanf(file, "%hhu %lf", &mat_id, &(hmgModel->props[i]))==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
        } else
          if(fscanf(file, "%hhu", &mat_id)==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
        printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);

        if (mat_id > MAX_COLORKEY){
          fclose(file);
          printf("ERROR: Material %d has color key (%d) greater than max (%d).\n",i,mat_id,MAX_COLORKEY);
          return HMG_FALSE;
        }
        hmgModel->props_keys[mat_id] = i;
      }
      
    } else if (!strcmp(str,"%thermal_expansion")){
      if (hmgModel->m_analysis_flag == HMG_ELASTIC){
        for (unsigned char i=0; i<hmgModel->m_nmat; i++){
          if(fscanf(file, "%lf", &(hmgModel->alpha[i]))==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
        }
        hmgModel->m_hmg_thermal_expansion_flag = HMG_TRUE;
      }

    //} else if (!strcmp(str,"%volume_fraction")){

    } else if (!strcmp(str,"%data_type")){
      if(fscanf(file, "%s", str)==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
      if (!strcmp(str,"float32") || !strcmp(str,"float") || !strcmp(str,"single")){
        hmgModel->m_scalar_field_data_type_flag = HMG_FLOAT32;
      } else if (!strcmp(str,"float64") || !strcmp(str,"double")){
        hmgModel->m_scalar_field_data_type_flag = HMG_FLOAT64;
      } else if (hmgModel->sdfFile!=NULL){
        printf("WARNING: %%data_type was set as \"%s\", which is invalid for the scalar density field input. Will attempt to read it as float32.",str);
        hmgModel->m_scalar_field_data_type_flag = HMG_FLOAT32;
      }
      printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
    } else if(!strcmp(str,"%xreduce_scale_factor")){
      if(fscanf(file, "%lf", &xreduce_scale_factor)==EOF){ printf("WARNING: Reached unexpected EOF when parsing %s\n",filename); break; }
      was_xreduce_stabfactor_set = HMG_TRUE;
      printf("\r    Scanning through neutral file...[%3d%%]",((++count)*100)/num_of_info);
    }
  }
  fclose(file);
  printf("\r    Scanning through neutral file...[%3d%%]",100);
  printf("\n");
  // check if number of materials is within bitstep range (for NbN and DbD strategies)
  if (hmgModel->m_nmat > (MATKEY_BITSTEP_RANGE_2D+1) && hmgModel->m_dim_flag == HMG_2D){
    printf("WARNING: Number of specified materials (%d) is greater than what can be represented with MATKEY_BITSTEP (%d). ",hmgModel->m_nmat,MATKEY_BITSTEP_2D);
    printf("Either avoid Node-by-Node and DOF-by-DOF, or recompile with greater MATKEY size.\n");
  } else if (hmgModel->m_nmat > (MATKEY_BITSTEP_RANGE_3D+1) && hmgModel->m_dim_flag == HMG_3D){
    printf("WARNING: Number of specified materials (%d) is greater than what can be represented with MATKEY_BITSTEP (%d). ",hmgModel->m_nmat,MATKEY_BITSTEP_3D);
    printf("Either avoid Node-by-Node and DOF-by-DOF, or recompile with greater MATKEY size.\n");
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
  // check xreduce stab factor
  if (!was_xreduce_stabfactor_set) xreduce_scale_factor = hmgModel->m_num_tol >= 0.0000005 ? (hmgModel->m_num_tol*hmgModel->m_num_tol) : 0.000000000001; // defaults to 1e-12 
  return HMG_TRUE;
}
//------------------------------------------------------------------------------
logical readMaterialMap(char * filename, uint8_t* npdata){
  if (filename == NULL && npdata != NULL){
    return readMaterialMapNumpy(npdata);
  } else {
    // Check file format before reading
    unsigned long int str_len = strlen(filename);
    if (!strcmp(&filename[str_len-3],".nf"))
      return readMaterialMapNF(filename);
    if (!strcmp(&filename[str_len-4],".raw")){
      return readMaterialMapRAW(filename);
    }
  }
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
logical readMaterialMapNumpy(uint8_t* npdata){
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
  // Loops to transpose data. Raw file is line by line, our indexing is
  // column by column. First slice runs out of loop (2D).
  k=0;
  unsigned int dataIndex = 0;
  uint8_t buffer;
  for (i = 0; i<rows; i++){
    for (j = 0; j<cols; j++){
      buffer = npdata[dataIndex++];
      for (kk = hmgModel->m_mesh_refinement*k; kk<(hmgModel->m_mesh_refinement*(k+1)*(hmgModel->m_nz>0)+(hmgModel->m_nz<1)); kk++){
        for (ii = hmgModel->m_mesh_refinement*i; ii<hmgModel->m_mesh_refinement*(i+1); ii++){
          for (jj = hmgModel->m_mesh_refinement*j; jj<hmgModel->m_mesh_refinement*(j+1); jj++){
            hmgModel->elem_material_map[ii+jj*rows_ref+kk*rows_ref*cols_ref] = (cudapcgMap_t) hmgModel->props_keys[buffer];
          }
        }
      }
    }
    printf("\r    Getting image from numpy...[%3d%%]",((i+1)*100)/(slices*rows));
  }
  for (k = 1; k<slices; k++){
    for (i = 0; i<rows; i++){
      for (j = 0; j<cols; j++){
        buffer = npdata[dataIndex++];
        for (kk = hmgModel->m_mesh_refinement*k; kk<(hmgModel->m_mesh_refinement*(k+1)*(hmgModel->m_nz>0)+(hmgModel->m_nz<1)); kk++){
          for (ii = hmgModel->m_mesh_refinement*i; ii<hmgModel->m_mesh_refinement*(i+1); ii++){
            for (jj = hmgModel->m_mesh_refinement*j; jj<hmgModel->m_mesh_refinement*(j+1); jj++){
              hmgModel->elem_material_map[ii+jj*rows_ref+kk*rows_ref*cols_ref] = (cudapcgMap_t) hmgModel->props_keys[buffer];
            }
          }
        }
      }
    }
    printf("\r    Getting image from numpy...[%3d%%]",((k+1)*100)/slices);
  }
  printf("\n");
  return HMG_TRUE;
}
//------------------------------------------------------------------------------
logical readScalarDensityFieldMap(char * filename){
  double buffer;
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
  if (hmgModel->m_scalar_field_data_type_flag > HMG_FLOAT64){
    printf("    ERROR: Invalid data type for the scalar field to be read from %s. Must be float32 or float64.\n",filename);
    return HMG_FALSE;
  }
  logical isF32 = (hmgModel->m_scalar_field_data_type_flag == HMG_FLOAT32);
  void *field = NULL;
  unsigned int sz = (isF32 ? sizeof(float) : sizeof(double));
  field = malloc(sz*hmgModel->m_nelem);
  if (field==NULL){
    printf("    ERROR: Failed to allocate memory to read scalar field from %s.\n",filename);
    return HMG_FALSE;
  }
  printf("    Getting density field from .bin...");
  FILE *file=NULL;
  file = fopen(filename,"rb");
  if (file==NULL){
    printf(" ERROR: Failed to open %s.\n",filename);
    free(field);
    return HMG_FALSE;
  }
  size_t managed_to_read = fread(field,sz,hmgModel->m_nelem,file);
  fclose(file);
  if (managed_to_read <  hmgModel->m_nelem){
    printf(" ERROR: Failed to read all %e MB from %s.\n",(float)(sz*hmgModel->m_nelem)/1024.0/1024.0,filename);
    free(field);
    return HMG_FALSE;
  }
  printf("Done.\n");
  unsigned int props_stride = hmgModel->m_analysis_flag==HMG_ELASTIC ? 2 : 1;
  double Emax=(double) (isF32 ? VOID_AS_FLOAT(field,0) : VOID_AS_DOUBLE(field,0));
  Emax *= hmgModel->props[hmgModel->elem_material_map[0]*props_stride];
  double Emin=Emax;
  double thisE;
  for (unsigned int i=1; i<(rows*cols*slices); i++){
    thisE = (double) (isF32 ? VOID_AS_FLOAT(field,i) : VOID_AS_DOUBLE(field,i));
    thisE *= hmgModel->props[hmgModel->elem_material_map[i]*props_stride];
    Emax = thisE>Emax ? thisE : Emax;
    Emin = thisE<Emin ? thisE : Emin;
  }
  if (Emax==Emin) Emin=Emax-1.0; // for safety
  double dE = Emax-Emin;
  hmgModel->density_max=Emax;
  hmgModel->density_min=Emin;
  printf("\r    Building parametric field...[%3d%%]",0);
  for (k = 0; k<slices; k++){
    for (i = 0; i<rows; i++){
      for (j = 0; j<cols; j++){
        buffer = (double) (isF32 ? VOID_AS_FLOAT(field,j+i*cols+k*rows*cols) : VOID_AS_DOUBLE(field,j+i*cols+k*rows*cols));
        buffer *= hmgModel->props[hmgModel->elem_material_map[(i+j*rows_ref+k*rows_ref*cols_ref)*hmgModel->m_mesh_refinement]*props_stride];
        for (kk = hmgModel->m_mesh_refinement*k; kk<(hmgModel->m_mesh_refinement*(k+1)*(hmgModel->m_nz>0)+(hmgModel->m_nz<1)); kk++){
          for (ii = hmgModel->m_mesh_refinement*i; ii<hmgModel->m_mesh_refinement*(i+1); ii++){
            for (jj = hmgModel->m_mesh_refinement*j; jj<hmgModel->m_mesh_refinement*(j+1); jj++){
              hmgModel->density_map[ii+jj*rows_ref+kk*rows_ref*cols_ref] = (parametricScalarField_t)(65535*(buffer-Emin)/dE); // uint16
            }
          }
        }
      }
    }
    printf("\r    Building parametric field...[%3d%%]",((k+1)*100)/slices);
  }
  printf("\n");
  free(field);
  return HMG_TRUE;
}
//------------------------------------------------------------------------------
void free_model_arrs(hmgModel_t * model){
  free(model->elem_material_map);
  free(model->node_dof_map);
  free(model->dof_material_map);
  return;
}
//------------------------------------------------------------------------------
void assemble_coarse_copy(hmgModel_t * coarse_model, hmgModel_t * original_model){
  // copy data from original to coarse model
  coarse_model->neutralFile            = original_model->neutralFile;
  coarse_model->neutralFile_noExt      = original_model->neutralFile_noExt;
  coarse_model->imageFile              = original_model->imageFile;
  coarse_model->sdfFile                = original_model->sdfFile;
  
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
