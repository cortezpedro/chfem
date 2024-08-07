#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "femhmg/femhmg.h"

#ifndef INCLUDES_H_INCLUDED
#define INCLUDES_H_INCLUDED

// Defines struct type for input info
typedef struct _input{

  char * neutral_file;
  char * raw_file;
  char * sdf_bin_file;
  
  unsigned char exportFields_flag;
  unsigned char fieldsByElem_flag;

  unsigned char save2binary_flag;
  char * binary_file;

  unsigned char writeReport_flag;
  char * report_file;

  unsigned char exportX_flag;
  unsigned char importX_flag;
  char * x0_file;

  unsigned int stopcrit_flag;

  unsigned int poremap_flag;

  unsigned int hmg_direction_flag; //hmgFlag_t
  unsigned int solver_flag; //cudapcgFlag_t
  unsigned int parallel_flag; //cudapcgFlag_t
  unsigned int num_of_recursions_initguess;

  unsigned int preconditioner_flag;

  unsigned int xreduce_flag; //cudapcgFlag_t

  var eff_coeff[42];  // to return in python API
  // obs: largest size = 6x6 elasticity + 6 thermal_expansion
  
  unsigned int num_of_thermal_expansion_coeffs;

} chfemgpuInput_t;

void printHelp();
unsigned int findFlag(const char *flag, char *arr[], unsigned int sz);
unsigned char readInput(char *arr[], unsigned int sz, chfemgpuInput_t * user_input);
void initDefaultInput(chfemgpuInput_t * user_input);
int runAnalysis(chfemgpuInput_t * user_input);

//------------------------------------------------------------------------------
// Error handling when reading input
#define READ_ENTRY(id,list,size) readEntry(id,list,size,__FILE__,__LINE__)
static inline char * readEntry(unsigned int i, char *arr[], unsigned int sz, const char * filename, unsigned int line){
    if (i>=sz){
      printf("Invalid input after %s.\nCall -h for help with input directives.\n",arr[sz-1]);
      return NULL;
    }
    return arr[i];
}
//------------------------------------------------------------------------------

#endif // INCLUDES_H_INCLUDED
