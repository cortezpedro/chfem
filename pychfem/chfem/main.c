/*
  =====================================================================
  Universidade Federal Fluminense (UFF) - Niteroi, Brazil
  Institute of Computing
  Authors: Cortez Lopes, P., Vianna, R., Sapucaia, V., Pereira., A.
  contact: pedrocortez@id.uff.br

  Image-based Computational Homogenization with the FEM in GPU.
             (C)           (H)                     (FEM)  (GPU)
  [chfem_gpu]
  
  History:
    * v0.0 (jul/2020) [ALL]    -> OpenMp, parallelization of CPU code
    * v1.0 (nov/2020) [CORTEZ] -> CUDA, PCG in GPU
    * v1.1 (sep/2022) [CORTEZ] -> Added permeability, MINRES.
                                  atomicAdd for EBE.
                                  refactoring of kernels for readability.

  Computes effective properties of heterogeneous periodic media
  characterized by raster images. Takes raw uint8 grayscale matrices
  as input.
  
  REQUIRES NVIDIA GRAPHICS CARDS AND DRIVERS.
  
  Developed in Debian-based Linux distros, for Linux.
  Has been tested (and ran fine) in the following OS:
    - Mint 19.3
    - Mint 20.3
    - Ubuntu 20.04
    - CentOS
    - Windows 10
  
  =====================================================================
  To compile:
  
    ~[root]$ cd compile
    ~[root]/compile$ python3 compile_chfem_gpu.py <options>
    ~[root]/compile$ cd ..
    ~[root]$ ls
    _________
    chfem_gpu compile src test 
    *********
        ^
   (executable)
  =====================================================================
  To run:
  
    ~[root]$ ./chfem_gpu [NF_FILE] [RAW_FILE] <options>
    
  =====================================================================
  To run tests:
  
    ~[root]$ cd test
    ~[root]/test$ python3 compile_test.py <options>
    ~[root]/test$ ./test
    
  =====================================================================
  OBS.1: FOR FURTHER INFO IN REGARDS TO <options> IN ALL OF THE CASES
         LISTED ABOVE, USE THE "-h" FLAG.
        
  OBS.2: AUXILIARY PYTHON SCRIPTS "compile_*.py" INVOKE nvcc TO GENERATE
         EXECUTABLE BINARIES. "python3" MAY BE REPLACED BY WHATEVER NAME
         OF YOUR ENVIROMENT VARIABLE FOR Python3.0+.
*/

#include "includes.h"

//------------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    
  // Initialize input struct
  chfemgpuInput_t * user_input = (chfemgpuInput_t *) malloc(sizeof(chfemgpuInput_t));
  initDefaultInput(user_input);

  // Check input
  unsigned char readInput_flag = readInput(argv,argc,user_input);
  if (!readInput_flag){
    free(user_input);
    printf("Failed to parse arguments.\nProcess aborted.\n");
    return 0;
  } else if (readInput_flag==2){ // found -h
    free(user_input);
    return 0;
  }

  printf("#######################################################\n");

  // Initialize FEM model for homogenization
  if (!hmgInit(user_input->neutral_file,user_input->raw_file,user_input->sdf_bin_file, NULL)){
    printf("Failed to properly read input files.\nProcess aborted.\n");
    printf("#######################################################\n");
    free(user_input);
    return -1;
  }

  return runAnalysis(user_input);
}

int runAnalysis(chfemgpuInput_t * user_input){
  // Keep track of metrics in a report file
  if (user_input->writeReport_flag)
    hmgKeepTrackOfAnalysisReport(REPORT_TRUE);

  // Set flag for exportation of fields from simulation
  hmgSaveFields(user_input->exportFields_flag,user_input->fieldsByElem_flag);

  // Set parallel strategy flag
  hmgSetParallelStrategyFlag(user_input->parallel_flag);

  // Set pore mapping strategy flag (used in permeability analysis)
  if (!hmgSetPoreMappingStrategy(user_input->poremap_flag)){
    printf("ERROR: Failed to set \"poremap_flag\"=%d.\nProcess aborted.\n",user_input->poremap_flag);
    hmgEnd();
    free(user_input);
    return -1;
  }

  // Set solver flag
  hmgSetSolverFlag(user_input->solver_flag);

  // Set preconditioner flag
  hmgSetPreConditionerFlag(user_input->preconditioner_flag);

  // Set xreduce flag
  hmgSetXReduceFlag(user_input->xreduce_flag);

  // Set homogenization direction flag (default is HOMOGENIZE_ALL)
  hmgSetHomogenizationFlag(user_input->hmg_direction_flag);

  // Print basic model data report
  hmgPrintModelData();

  // Specify stopping criteria for the pcg solver
  hmgSetStoppingCriteria(user_input->stopcrit_flag);

  // Initial guesses for PCG solver
  if (user_input->importX_flag){
    // Read initial guesses from binary file
    hmgImportX(user_input->x0_file);
  } else {
    // Compute initial guesses from coarse meshes
    hmgFindInitialGuesses(user_input->num_of_recursions_initguess);
  }

  // Set flag to store PCG results
  hmgExportX(user_input->exportX_flag);

  // Solve for effective homogenized properties
  hmgSolveHomogenization();

  // Print homogenized constitutive matrix
  hmgPrintConstitutiveMtx();

  // Write report file (will only activate if hmgKeepTrackOfAnalysisReport was called)
  hmgPrintReport(user_input->report_file);

  // Save result to binary file
  if (user_input->save2binary_flag)
    hmgSaveConstitutiveMtx(user_input->binary_file);

  // Finish femhmg API. (ATTENTION: Will free dynamic arrays from memory)
  hmgEnd();

  // Free input struct
  free(user_input);

  printf("#######################################################\n");

  return 0;
}

//------------------------------------------------------------------------------
void printHelp(){
  printf("\nComputational Homogenization with the image-based FEM in GPU - v1.2 - (2020-2023) - LCC UFF\n\n");

  printf("./chfem_gpu [.nf] [.raw] ... (options)\n\nor\n\n./chfem_gpu (options) ... -i [.nf] [.raw] ... (options)\n\n");
  printf("\t-b: Save results in a binary file. Must be followed by a string with a filename.\n");
  printf("\t-c: Stopping criteria for the PCG method: 0 - L2 (default), 1 - Inf, 2 - L2+Error.\n");
  printf("\t-d: Target direction: 0 - X, 1 - Y, 2 - Z, 3 - YZ, 4 - XZ, 5 - XY, 6 - ALL (default).\n");
  printf("\t-e: Export fields from simulation (by nodes).\n");
  printf("\t-f: Input scalar density field. Must be followed by a [.bin] file.\n");
  printf("\t-h: Print this help info and exit.\n");
  printf("\t-i: Input files. Must be followed by: [.nf] [.raw].\n");
  printf("\t-j: Jacobi preconditioning: 0 - no, 1 - yes (default).\n");
  printf("\t-m: Write metrics report. Must be followed by a string with a filename.\n");
  printf("\t-p: Parallel matrix-free strategy: 0 - NBN (default), 1 - EBE.\n");
  printf("\t-pm: Pore mapping strategy: 0 - image, 1 - DOF number (default).\n");
  printf("\t-r: Number of recursive searches for initial guesses.\n");
  printf("\t-s: Solver: 0 - CG (default), 1 - MINRES, 2 - CG3, 3 - MINRES3, 4 - CG2, 5 - MINRES2.\n");
  printf("\t-u: Reduce strategy for velocity fields (FLUID): 0 - on the fly, 1 - only diag, 2 - full.\n");
  printf("\t-xi: Import initial guesses for PCG solver from binary files. Must be followed by a string with a filename.\n");
  printf("\t-xo: Export result vector (x) from the PCG solver.\n\n");

  return;
}
//------------------------------------------------------------------------------
unsigned int findFlag(const char *flag, char *arr[], unsigned int sz){
  unsigned int index = 0;
  for (unsigned int i=1; i<sz; i++){
    if (!strcmp(arr[i],flag)){
      index = i;
      break;
    }
  }
  return index;
}
//------------------------------------------------------------------------------
unsigned char readInput(char *arr[], unsigned int sz, chfemgpuInput_t * user_input){
  // Initial checks
  if (sz<2){
    printf("Invalid input.\nCall -h for help with input directives.\n");
    return 0;
  }
  // Search for -h flag
  if (findFlag("-h",arr,sz)){
    printHelp();
    return 2;
  }
  if (sz==2){
    printf("Invalid input ignored: %s.\nCall -h for help with input directives.\n",arr[1]);
    return 0;
  }
  // Find strings with input files
  unsigned int id = findFlag("-i",arr,sz);
  user_input->neutral_file  = READ_ENTRY(id+1,arr,sz);
  user_input->raw_file      = READ_ENTRY(id+2,arr,sz);
  // Iterate through input array
  unsigned int i=1+(!id)*2;
  char * str_buffer=NULL;
  unsigned int * ptr_prop=NULL;
  unsigned char valid_input_flag=0;
  while(i<sz){
    if (i!=id){
      if (!strcmp(arr[i],"-r")){
        ptr_prop = &(user_input->num_of_recursions_initguess);
      } else if (!strcmp(arr[i],"-j")){
        ptr_prop = &(user_input->preconditioner_flag);
      } else if (!strcmp(arr[i],"-u")){
        ptr_prop = &(user_input->xreduce_flag);
      } else if (!strcmp(arr[i],"-s")){
        ptr_prop = &(user_input->solver_flag);
      } else if (!strcmp(arr[i],"-p")){
        ptr_prop = &(user_input->parallel_flag);
      } else if (!strcmp(arr[i],"-pm")){
        ptr_prop = &(user_input->poremap_flag);
      } else if (!strcmp(arr[i],"-d")){
        ptr_prop = &(user_input->hmg_direction_flag);
      } else if (!strcmp(arr[i],"-c")){
        ptr_prop = &(user_input->stopcrit_flag);
      } else if (!strcmp(arr[i],"-e")){
        //str_buffer = READ_ENTRY(++i,arr,sz); if (str_buffer==NULL) return 0;
        user_input->exportFields_flag = 1;
        user_input->fieldsByElem_flag = 0;//atoi(str_buffer) > 0;
        ptr_prop = NULL;
        valid_input_flag=1;
      } else if (!strcmp(arr[i],"-b")){
        str_buffer = READ_ENTRY(++i,arr,sz); if (str_buffer==NULL) return 0;
        user_input->save2binary_flag = 1;
        user_input->binary_file = str_buffer;
        ptr_prop = NULL;
        valid_input_flag=1;
      } else if (!strcmp(arr[i],"-f")){
        str_buffer = READ_ENTRY(++i,arr,sz); if (str_buffer==NULL) return 0;
        user_input->sdf_bin_file = str_buffer;
        ptr_prop = NULL;
        valid_input_flag=1;
      } else if (!strcmp(arr[i],"-m")){
        str_buffer = READ_ENTRY(++i,arr,sz); if (str_buffer==NULL) return 0;
        user_input->writeReport_flag = 1;
        user_input->report_file = str_buffer;
        ptr_prop = NULL;
        valid_input_flag=1;
      } else if (!strcmp(arr[i],"-xi")){
        str_buffer = READ_ENTRY(++i,arr,sz); if (str_buffer==NULL) return 0;
        user_input->importX_flag = 1;
        user_input->x0_file = str_buffer;
        ptr_prop = NULL;
        valid_input_flag=1;
      } else if (!strcmp(arr[i],"-xo")){
        user_input->exportX_flag = 1;
        ptr_prop = NULL;
        valid_input_flag=1;
      }
      if (ptr_prop!=NULL){
        str_buffer = READ_ENTRY(++i,arr,sz); if (str_buffer==NULL) return 0;
        *ptr_prop = atoi(str_buffer); // maybe generalize later
        ptr_prop=NULL;
      } else {
        if (!valid_input_flag)
          printf("Invalid input ignored: %s.\n",arr[i]);
        valid_input_flag=0;
      }
    } else i+=2;
    i++;
  }
  // Check for potential problems
  unsigned char success = 1;
  if (user_input->neutral_file==NULL || user_input->raw_file==NULL){
    printf("Invalid input, .nf and .raw files are required to characterize model.\n");
    success = 0;
  }
  if (user_input->xreduce_flag>2){
    printf("Invalid reduce strategy flag.\n");
    success = 0;
  }
  if (user_input->solver_flag>5){
    printf("Invalid solver flag.\n");
    success = 0;
  }
  if (user_input->parallel_flag>1){
    printf("Invalid parallel strategy flag.\n");
    success = 0;
  }
  if (!success) printf("Call -h for help with input directives.\n");
  return success;
}
//------------------------------------------------------------------------------
void initDefaultInput(chfemgpuInput_t * user_input){
  user_input->neutral_file = NULL;
  user_input->raw_file = NULL;
  user_input->sdf_bin_file = NULL;
  user_input->hmg_direction_flag = HOMOGENIZE_ALL;
  user_input->solver_flag = 0;
  user_input->preconditioner_flag = 1;
  user_input->parallel_flag = 0;
  user_input->num_of_recursions_initguess = 0;
  user_input->save2binary_flag = 0;
  user_input->binary_file = NULL;
  user_input->writeReport_flag = 0;
  user_input->report_file = NULL;
  user_input->exportFields_flag = 0;
  user_input->fieldsByElem_flag = 0;
  user_input->exportX_flag = 0;
  user_input->importX_flag = 0;
  user_input->x0_file = NULL;
  user_input->stopcrit_flag = 0;
  user_input->poremap_flag = 1;
  user_input->xreduce_flag = 2;
  return;
}
//------------------------------------------------------------------------------
