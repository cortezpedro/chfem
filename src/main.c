/*
	Universidade Federal Fluminense (UFF) - Niteroi, Brazil
	Institute of Computing
	Authors: Cortez, P., Vianna, R., Sapucaia, V., Pereira., A.
	History: 
		* v0.0 (jul/2020) [ALL]    -> OpenMp, parallelization of CPU code
		* v1.0 (nov/2020) [CORTEZ] -> CUDA, PCG on GPU. (substituted parpcg.h with cudapcg.h)

	Homogenization of thermal conductivity properties with FEM.

    ATTENTION.1:
        Considers a structured regular mesh of quad elements (2D)
        or hexahedron elements (3D).

	ATTENTION.2:
        As it is, works for Potential or Elasticity, both 2D or 3D.

	-> Compiling directives with nvcc (Linux):
		$ nvcc -c cudapcg_kernels.cu
		$ nvcc -c cudapcg.cu
		$ nvcc -c femhmg.cu
		$ nvcc -c main.cu
		$ nvcc -o chfem_gpu *.o -Xcompiler -fopenmp
		$ rm *.o
	-> Run executable (Linux):
		$ ./chfemgpu [DATA.nf] [IMAGE.raw or ELEMMATMAP.nf]
*/

#include "includes.h"

//------------------------------------------------------------------------------
void printHelp();
unsigned int findFlag(const char *flag, char *arr[], unsigned int sz);
unsigned char readInput(char *arr[], unsigned int sz, chfemgpuInput_t * user_input);
void initDefaultInput(chfemgpuInput_t * user_input);

//------------------------------------------------------------------------------
int main(int argc, char * argv[]){

  // Initialize input struct
  chfemgpuInput_t * user_input = (chfemgpuInput_t *) malloc(sizeof(chfemgpuInput_t));
  initDefaultInput(user_input);

	// Check input
	if (!readInput(argv,argc,user_input)){
	  free(user_input);
	  return 0;
	}

	printf("#######################################################\n");

	// Initialize FEM model for homogenization
	if (!hmgInit(user_input->neutral_file,user_input->raw_file)){
	  printf("Failed to properly read input files.\nProcess aborted.\n");
		printf("#######################################################\n");
		free(user_input);
		return -1;
	}
	
	// Keep track of metrics in a report file
	if (user_input->writeReport_flag)
	  hmgKeepTrackOfAnalysisReport(REPORT_TRUE);
	
	// Set parallel strategy flag
	hmgSetParallelStrategyFlag(user_input->parallel_flag);
	
	// Set homogenization direction flag (default is HOMOGENIZE_ALL)
	hmgSetHomogenizationFlag(user_input->hmg_direction_flag);
	
	// Print basic model data report
	hmgPrintModelData();

	// Compute initial guesses from coarse meshes
	hmgFindInitialGuesses(user_input->num_of_recursions_initguess);
	
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
  printf("\nComputational Homogenization with the image-based FEM in GPU - v1.0 - (2020-2021) - LCC UFF\n\n");
  
  printf("./chfem_gpu [.nf] [.raw] ... (options)\n\nor\n\n./chfem_gpu (options) ... -i [.nf] [.raw] ... (options)\n\n");
  printf("\t-b: Save results in a binary file. Must be followed by a string with a filename.\n");
  printf("\t-d: Target direction: 0-X, 1-Y, 2-Z, 3-YZ, 4-XZ, 5-XY. 6 for all directions.\n");
  printf("\t-h: Print this help info and exit.\n");
  printf("\t-i: Input files. Must be followed by: [.nf] [.raw].\n");
  printf("\t-m: Write metrics report. Must be followed by a string with a filename.\n");
  printf("\t-p: Parallel assembly on-the-fly strategy: 0 (NbN), 1 (EbE).\n");
  printf("\t-r: Number of recursive searches for initial guesses.\n\n");

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
  if (sz==2){
    if (!strcmp(arr[1],"-h"))
      printHelp();
    else
      printf("Invalid input ignored: %s.\nCall -h for help with input directives.\n",arr[1]);
    return 0;
  }
  // Search for -h flag
  if (findFlag("-h",arr,sz)){
    printHelp();
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
      } else if (!strcmp(arr[i],"-p")){
        ptr_prop = &(user_input->parallel_flag);
      } else if (!strcmp(arr[i],"-d")){
        ptr_prop = &(user_input->hmg_direction_flag);
      } else if (!strcmp(arr[i],"-b")){
        str_buffer = READ_ENTRY(++i,arr,sz); if (str_buffer==NULL) return 0;
        user_input->save2binary_flag = 1;
        user_input->binary_file = str_buffer;
        ptr_prop = NULL;
        valid_input_flag=1;
      } else if (!strcmp(arr[i],"-m")){
        str_buffer = READ_ENTRY(++i,arr,sz); if (str_buffer==NULL) return 0;
        user_input->writeReport_flag = 1;
        user_input->report_file = str_buffer;
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
  user_input->hmg_direction_flag = HOMOGENIZE_ALL;
  user_input->parallel_flag = 0;
  user_input->num_of_recursions_initguess = 0;
  user_input->save2binary_flag = 0;
  user_input->binary_file = NULL;
  user_input->writeReport_flag = 0;
  user_input->report_file = NULL;
  return;
}
//------------------------------------------------------------------------------
