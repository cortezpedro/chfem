/*
    Universidade Federal Fluminense (UFF) - Niteroi, Brazil
    Institute of Computing
    Authors: Cortez, P., Vianna, R.
    History: 
		* v0.0 (jul/2020) [ALL]    -> OpenMp, parallelization of CPU code
		* v1.0 (nov/2020) [CORTEZ] -> CUDA, PCG on GPU

    Includes header for FEM homogenization of physical properties from micro-CT
    images, in binary grayscale representation.

    Meant for "femhmg" and "cudapcg".
*/

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
  
  unsigned char save2binary_flag;
  char * binary_file;
  
  unsigned char writeReport_flag;
  char * report_file;
  
  unsigned int hmg_direction_flag; //hmgFlag_t
  unsigned int parallel_flag; //cudapcgFlag_t
  unsigned int num_of_recursions_initguess;
  
} chfemgpuInput_t;

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
