#include "../src/includes.h"
#include <math.h>

#define NUM_TESTS 14
#define TOL 0.0001
#define ZERO 0.0000000000000001

typedef chfemgpuInput_t input_t;

typedef struct __res{
  unsigned int sz;
  char *bin_file;
  double max_diff;
} result_t;


unsigned char WENT_OK[NUM_TESTS];
double DIFF[NUM_TESTS];

//------------------------------------------------------------------------------
char NF_FILES[NUM_TESTS][1024] = {
  "input/2D/thermal/100x100.nf",
  "input/2D/thermal/100x100.nf",
  "input/2D/thermal/100x100_bin.nf",
  "input/2D/elastic/100x100.nf",
  "input/2D/elastic/100x100.nf",
  "input/2D/elastic/100x100_bin.nf",
  "input/2D/fluid/100x100.nf",
  "input/3D/thermal/ggg40_100.nf",
  "input/3D/thermal/ggg40_100.nf",
  "input/3D/thermal/ggg40_100_bin.nf",
  "input/3D/elastic/ggg40_100.nf",
  "input/3D/elastic/ggg40_100.nf",
  "input/3D/elastic/ggg40_100_bin.nf",
  "input/3D/fluid/fibers_100x100x10.nf"
};
char RAW_FILES[NUM_TESTS][1024] = {
  "input/2D/100x100.raw",
  "input/2D/100x100.raw",
  "input/2D/100x100.raw",
  "input/2D/100x100.raw",
  "input/2D/100x100.raw",
  "input/2D/100x100.raw",
  "input/2D/100x100.raw",
  "input/3D/ggg40_100.raw",
  "input/3D/ggg40_100.raw",
  "input/3D/ggg40_100.raw",
  "input/3D/ggg40_100.raw",
  "input/3D/ggg40_100.raw",
  "input/3D/ggg40_100.raw",
  "input/3D/fibers_100x100x10.raw"
};
char SDF_FILES[NUM_TESTS][1024] = {
  "",
  "",
  "input/2D/100x100.bin",
  "",
  "",
  "input/2D/100x100.bin",
  "",
  "",
  "",
  "input/3D/ggg40_100_thermal.bin",
  "",
  "",
  "input/3D/ggg40_100_elastic.bin",
  ""
};
char RES_FILES[NUM_TESTS][1024] = {
  "input/2D/thermal/100x100_result.bin",
  "input/2D/thermal/100x100_result.bin",
  "input/2D/thermal/100x100_result.bin",
  "input/2D/elastic/100x100_result.bin",
  "input/2D/elastic/100x100_result.bin",
  "input/2D/elastic/100x100_result.bin",
  "input/2D/fluid/100x100_result.bin",
  "input/3D/thermal/ggg40_100_result.bin",
  "input/3D/thermal/ggg40_100_result.bin",
  "input/3D/thermal/ggg40_100_result.bin",
  "input/3D/elastic/ggg40_100_result.bin",
  "input/3D/elastic/ggg40_100_result.bin",
  "input/3D/elastic/ggg40_100_result.bin",
  "input/3D/fluid/fibers_100x100x10_result.bin"
};
unsigned int RES_SIZES[NUM_TESTS] = {
  4,
  4,
  4,
  9,
  9,
  9,
  4,
  9,
  9,
  9,
  36,
  36,
  36,
  9
};
//                        nf  raw  sdf -e -b bin -m txt xo xi bin
#define DEFAULT_IO_FLAGS NULL,NULL,NULL,0,0,0,NULL,0,NULL,0,0,NULL
input_t INPUTS[NUM_TESTS] = {
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,         // -c
    CUDAPCG_POREMAP_NUM,     // -pm
    HOMOGENIZE_ALL,          // -d
    CUDAPCG_CG_SOLVER,       // -s
    CUDAPCG_NBN,             // -p
    1,                       // -r
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL // -j, -u
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_ALL,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_EBE,
    1,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_ALL,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_EBE,
    1,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_ALL,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_NBN,
    1,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_ALL,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_EBE,
    1,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_ALL,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_EBE,
    1,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_ALL,
    CUDAPCG_MINRES_SOLVER,
    CUDAPCG_NBN,
    0,
    CUDAPCG_FALSE, CUDAPCG_XREDUCE_FULL // no jacobi
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_X,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_NBN,
    1,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_X,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_EBE,
    1,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_X,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_EBE,
    1,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_Y,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_NBN,
    1,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_Y,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_EBE,
    1,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_Y,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_EBE,
    1,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  },
  { 
    DEFAULT_IO_FLAGS,
    CUDAPCG_L2_NORM,
    CUDAPCG_POREMAP_NUM,
    HOMOGENIZE_Z,
    CUDAPCG_CG_SOLVER,
    CUDAPCG_NBN,
    0,
    CUDAPCG_TRUE, CUDAPCG_XREDUCE_FULL
  }
};
//------------------------------------------------------------------------------
int runTest(input_t *input, result_t *res);
//------------------------------------------------------------------------------
int main(void){
  
  result_t res = {0,NULL,0.0};
  int flag;
  
  for (int i=0; i<NUM_TESTS; i++){
    
    INPUTS[i].neutral_file = &( NF_FILES[i][0]);
    INPUTS[i].raw_file     = &(RAW_FILES[i][0]);
    INPUTS[i].sdf_bin_file = SDF_FILES[i][0]=='\0' ? NULL : &(SDF_FILES[i][0]);
  
    res.sz = RES_SIZES[i];
    res.bin_file = &(RES_FILES[i][0]);
    res.max_diff = 0.0;
    
    WENT_OK[i] = 0;
    
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\nTest %d...\n",i);
    flag = runTest(&INPUTS[i],&res);
    
    DIFF[i] = res.max_diff;
    
    if (!flag){
      printf("ERROR: Test %d failed.\n",i);
    }else if (res.max_diff>TOL || res.max_diff<(-TOL)){
      printf("WARNING: Test %d had \"max_diff\" greater than \"TOL\" (%.6e>%.6e)\n",i,res.max_diff,TOL);
      WENT_OK[i] = 2;
    }else
      WENT_OK[i] = 1;
  }
  
  printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
  printf("SUMMARY:\n");
  for (int i=0; i<NUM_TESTS; i++){
    printf("Test %d (%s): max_diff=%.3e -> ",i,NF_FILES[i],DIFF[i]);
    WENT_OK[i] ? ( WENT_OK[i] > 1 ? printf("[DIFF ABOVE TOL]\n") : printf("[OK]\n") ): printf("[FAILED]\n");
  }
  printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

  return 0;
}
//------------------------------------------------------------------------------
int runTest(input_t *input, result_t *res){

  // Initialize FEM model for homogenization
  if (!hmgInit(input->neutral_file,input->raw_file,input->sdf_bin_file)){ res->max_diff=INFINITY; return 0; }

  // Set parallel strategy flag
  hmgSetParallelStrategyFlag(input->parallel_flag);

  // Set pore mapping strategy flag (used in permeability analysis)
  if (!hmgSetPoreMappingStrategy(input->poremap_flag)){ res->max_diff=INFINITY; return 0; }

  // Set solver flag
  hmgSetSolverFlag(input->solver_flag);

  // Set preconditioner flag
  hmgSetPreConditionerFlag(input->preconditioner_flag);

  // Set xreduce flag
  hmgSetXReduceFlag(input->xreduce_flag);

  // Set homogenization direction flag (default is HOMOGENIZE_ALL)
  hmgSetHomogenizationFlag(input->hmg_direction_flag);
  
  // Print basic model data report
  hmgPrintModelData();

  // Initial guesses for PCG solver
  // Compute initial guesses from coarse meshes
  hmgFindInitialGuesses(input->num_of_recursions_initguess);

  // Solve for effective homogenized properties
  hmgSolveHomogenization();
  
  // Print homogenized constitutive matrix
  hmgPrintConstitutiveMtx();

  // Get homogenized constitutive matrix
  double *C = hmgGetConstitutiveMtx();
  
  // Check results
  double C_ref[36];
  FILE *file;
  file = fopen(res->bin_file,"rb");
  if (file){
    if (fread(&C_ref[0],sizeof(double),res->sz,file) < res->sz) printf("WARNING: Failed to read all %d items from %s\n",res->sz,res->bin_file);
  }
  fclose(file);
  
  res->max_diff = 0.0;
  double diff, max_c=0.0;
  for (int i=0;i<res->sz;i++){
    if (isnan(C[i])){
      res->max_diff = INFINITY;
      break;
    }
    diff = C[i]-C_ref[i];
    res->max_diff = fabs(res->max_diff) > fabs(diff) ? res->max_diff : diff;
    max_c = fabs(max_c)>fabs(C[i]) ? max_c : C[i];
  }
  res->max_diff /= fabs(max_c)>ZERO ? max_c : 1.0;

  // Finish femhmg API. (ATTENTION: Will free dynamic arrays from memory)
  hmgEnd();
  
  return 1;
}
//------------------------------------------------------------------------------
