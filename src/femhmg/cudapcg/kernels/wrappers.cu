/*
  =====================================================================
  Universidade Federal Fluminense (UFF) - Niteroi, Brazil
  Institute of Computing
  Authors: Cortez Lopes, P., Pereira., A.
  contact: pedrocortez@id.uff.br

  [cudapcg]
  
  History:
    * v1.0 (nov/2020) [CORTEZ] -> CUDA, PCG in GPU
    * v1.1 (sep/2022) [CORTEZ] -> Added permeability, MINRES.
                                  atomicAdd for EBE.
                                  refactoring of kernels for readability.
  
  Pre-history:
    Initially developed as a final work for the graduate course "Arquitetura
    e Programacao de GPUs", at the Institute of Computing, UFF.

  API for solving linear systems associated to FEM models with an matrix-free
  solvers, using CUDA. All global matrix operations involve "assembly on-the-fly"

  THERMAL CONDUCTIVITY, LINEAR ELASTICITY, ABSOLUTE PERMEABILITY.
  
  General purpose kernels are implemented here.
  Host wrappers for NBN and EBE kernels are also implemented here.

  =====================================================================
*/

#include "wrappers.h"
#include "vector.h"
#include "image.h"
#include "matrix_free.h"
#include "utils.h"
#include "../error_handling.h"

//---------------------------------
///////////////////////////////////
//////////// GLOBALS //////////////
///////////////////////////////////
//---------------------------------

#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
  cudapcgVar_t *K = NULL; // for larger ranges of matkeys, store local matrices in device global memory
#endif

cudapcgVar_t *M=NULL; // array for jacobi preconditioner (stored in GPU). Obs.: Only allocated if explictly requested.

double *dot_res_1=NULL, *dot_res_2=NULL; // needed for reduction on dotprod

cudapcgFlag_t flag_PreConditionerWasAssembled = CUDAPCG_FALSE;

//---------------------------------
///////////////////////////////////
//////// PRIVATE FUNCTIONS ////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
double reduce_dot_res_1(unsigned int dim){ // dim: dimension of the vector that has been reduced to dot_res_1
    unsigned int blockDim = THREADS_PER_BLOCK;
    unsigned int gridDim = CEIL(dim,blockDim);
    unsigned int isRes1or2 = 1;
    while(dim > THREADS_PER_BLOCK){
      dim = gridDim;
      gridDim = CEIL(dim,blockDim);
      if (isRes1or2 == 1)
        kernel_reduce<double><<<gridDim,blockDim>>>(dot_res_1,dim,dot_res_2);
      else
        kernel_reduce<double><<<gridDim,blockDim>>>(dot_res_2,dim,dot_res_1);
      isRes1or2 = (isRes1or2+1)%2;
    }
    double res;
    if (isRes1or2 == 1){
      HANDLE_ERROR(cudaMemcpy(&res,dot_res_1,sizeof(double),cudaMemcpyDeviceToHost));
    } else {
      HANDLE_ERROR(cudaMemcpy(&res,dot_res_2,sizeof(double),cudaMemcpyDeviceToHost));
    }
    return res;
}
//------------------------------------------------------------------------------
double max_dot_res_1(unsigned int dim){ // dim: dimension of the vector that has been reduced to dot_res_1
    unsigned int blockDim = THREADS_PER_BLOCK;
    unsigned int gridDim = CEIL(dim,blockDim);
    unsigned int isRes1or2 = 1;
    while(dim > THREADS_PER_BLOCK){
      dim = gridDim;
      gridDim = CEIL(dim,blockDim);
      if (isRes1or2 == 1)
        kernel_max<double><<<gridDim,blockDim>>>(dot_res_1,dim,dot_res_2);
      else
        kernel_max<double><<<gridDim,blockDim>>>(dot_res_2,dim,dot_res_1);
      isRes1or2 = (isRes1or2+1)%2;
    }
    double res;
    if (isRes1or2 == 1){
      HANDLE_ERROR(cudaMemcpy(&res,dot_res_1,sizeof(double),cudaMemcpyDeviceToHost));
    } else {
      HANDLE_ERROR(cudaMemcpy(&res,dot_res_2,sizeof(double),cudaMemcpyDeviceToHost));
    }
    return res;
}
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
//////// PUBLIC FUNCTIONS /////////
///////////////////////////////////
//---------------------------------

//---------------------------
/////////////////////////////
///// MEMORY MANAGEMENT /////
/////////////////////////////
//---------------------------

//------------------------------------------------------------------------------
void allocDotProdArrs(unsigned int dim){
    unsigned int sz = sizeof(double)*CEIL(dim,THREADS_PER_BLOCK);
    HANDLE_ERROR(cudaMalloc(&dot_res_1,sz));
    sz/=sizeof(double);
    sz = sizeof(double)*CEIL(sz,THREADS_PER_BLOCK);
    HANDLE_ERROR(cudaMalloc(&dot_res_2,sz));
    return;
}
//------------------------------------------------------------------------------
void freeDotProdArrs(){
    if (dot_res_1!=NULL) HANDLE_ERROR(cudaFree(dot_res_1)); dot_res_1=NULL;
    if (dot_res_2!=NULL) HANDLE_ERROR(cudaFree(dot_res_2)); dot_res_2=NULL;
    return;
}
//------------------------------------------------------------------------------
void allocLocalK(unsigned long int size){
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    HANDLE_ERROR(cudaMalloc(&K,size));
  #endif
  return;
}
//------------------------------------------------------------------------------
void setLocalK(cudapcgVar_t * lclK, unsigned long int size){
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      HANDLE_ERROR(cudaMemcpy(K,lclK,size,cudaMemcpyHostToDevice));
    #else // 16bit or 8bit matkeys
      //HANDLE_ERROR(cudaMemcpyToSymbol(K,lclK,size));
      setConstantLocalK(lclK,size);
    #endif
    return;
}
//------------------------------------------------------------------------------
void freeLocalK(){
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    if (K!=NULL){
      HANDLE_ERROR(cudaFree(K));
      K = NULL;
    }
  #endif
  return;
}
//------------------------------------------------------------------------------
void allocPreConditioner(cudapcgModel_t *m){
    HANDLE_ERROR(cudaMalloc(&M,m->nvars*sizeof(cudapcgVar_t)));
    return;
}
//------------------------------------------------------------------------------
void freePreConditioner(){
    if (M!=NULL) HANDLE_ERROR(cudaFree(M)); M=NULL;
    flag_PreConditionerWasAssembled = CUDAPCG_FALSE;
    return;
}
//------------------------------------------------------------------------------

//---------------------------
/////////////////////////////
///// VECTOR OPERATIONS /////
/////////////////////////////
//---------------------------

//------------------------------------------------------------------------------
void zeros(cudapcgVar_t * v, unsigned int dim){
    kernel_zeros<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim);
    return;
}
//------------------------------------------------------------------------------
void arrcpy(cudapcgVar_t * v, unsigned int dim, cudapcgVar_t * res){
    kernel_arrcpy<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim,res);
    return;
}
//------------------------------------------------------------------------------
cudapcgVar_t max(cudapcgVar_t *v, unsigned int dim){
    kernel_max<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim,dot_res_1);
    return max_dot_res_1(dim);
}
//------------------------------------------------------------------------------
cudapcgVar_t absmax(cudapcgVar_t *v, unsigned int dim){
    kernel_absmax<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim,dot_res_1);
    return max_dot_res_1(dim);
}
//------------------------------------------------------------------------------
double reduce(cudapcgVar_t *v, unsigned int dim){
    kernel_reduce<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim,dot_res_1);
    return reduce_dot_res_1(dim);
}
//------------------------------------------------------------------------------
double absreduce(cudapcgVar_t *v, unsigned int dim){
    kernel_absreduce<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim,dot_res_1);
    return reduce_dot_res_1(dim);
}
//------------------------------------------------------------------------------
double dotprod(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim){
    kernel_dotprod<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v1,v2,dim,dot_res_1);
    return reduce_dot_res_1(dim);
}
//------------------------------------------------------------------------------
void termbytermmul(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res){
    kernel_termbytermmul<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v1,v2,dim,res);
    return;
}
//------------------------------------------------------------------------------
void termbytermdiv(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res){
    kernel_termbytermdiv<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v1,v2,dim,res);
    return;
}
//------------------------------------------------------------------------------
void termbyterminv(cudapcgVar_t * v, unsigned int dim){
    kernel_termbyterminv<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim);
    return;
}
//------------------------------------------------------------------------------
void saxpy(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim, cudapcgVar_t * res){
    kernel_saxpy<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(y,x,a,dim,res);
    return;
}
//------------------------------------------------------------------------------
void saxpy_iny(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim){
    kernel_saxpy_iny<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(y,x,a,dim);
    return;
}
//------------------------------------------------------------------------------

//---------------------------
/////////////////////////////
///// IMAGE OPERATIONS //////
/////////////////////////////
//---------------------------

//------------------------------------------------------------------------------
void interpl2(cudapcgVar_t *v, unsigned int rows, unsigned int cols, unsigned int layers, unsigned int stride, cudapcgVar_t *res){
    uint8_t is2D = layers==0;
    layers += layers<1; // if 0 (2D), work with a single layer
    size_t voxels = rows*cols*layers;
    size_t n = stride*voxels;
    size_t voxels_halfres = (rows/2)*(cols/2)*(layers > 1 ? layers/2 : 1);
    kernel_zeros<cudapcgVar_t><<<CEIL(n,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,n);
    kernel_project2<cudapcgVar_t><<<CEIL(voxels_halfres,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,v,cols,rows,layers,stride);
    kernel_interpl_rows<cudapcgVar_t><<<CEIL(voxels,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,cols,rows,layers,stride);
    kernel_interpl_cols<cudapcgVar_t><<<CEIL(voxels,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,cols,rows,layers,stride);
    if (!is2D)
      kernel_interpl_layers<cudapcgVar_t><<<CEIL(voxels,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,cols,rows,layers,stride);
    return;
}
//------------------------------------------------------------------------------

//---------------------------
/////////////////////////////
/// MATRIXFREE OPERATIONS ///
/////////////////////////////
//---------------------------

//------------------------------------------------------------------------------
void assemblePreConditioner_thermal_2D(cudapcgModel_t *m){
    if (m->parametric_density_field_flag == CUDAPCG_TRUE){
      flag_PreConditionerWasAssembled = CUDAPCG_FALSE;
      return;
    }
    if (m->parStrategy_flag == CUDAPCG_NBN){
      // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
      kernel_assemblePreConditioner_thermal_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        M,m->nvars,m->image);
    } else if (m->parStrategy_flag == CUDAPCG_EBE){
        kernel_assemblePreConditioner_thermal_2D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        M,m->nelem,m->image,m->ncols,m->nrows);
    }
    flag_PreConditionerWasAssembled = CUDAPCG_TRUE;
    return;
}
//------------------------------------------------------------------------------
void assemblePreConditioner_thermal_3D(cudapcgModel_t *m){
    if (m->parametric_density_field_flag == CUDAPCG_TRUE){
      flag_PreConditionerWasAssembled = CUDAPCG_FALSE;
      return;
    }
    if (m->parStrategy_flag == CUDAPCG_NBN){
      // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
      kernel_assemblePreConditioner_thermal_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        M,m->nvars,m->image);
    } else if (m->parStrategy_flag == CUDAPCG_EBE){
        kernel_assemblePreConditioner_thermal_3D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        M,m->nelem,m->image,m->ncols,m->nrows,m->nlayers);
    }
    flag_PreConditionerWasAssembled = CUDAPCG_TRUE;
    return;
}
//------------------------------------------------------------------------------
void assemblePreConditioner_elastic_2D(cudapcgModel_t *m){
    if (m->parametric_density_field_flag == CUDAPCG_TRUE){
      flag_PreConditionerWasAssembled = CUDAPCG_FALSE;
      return;
    }
    if (m->parStrategy_flag == CUDAPCG_NBN){
      // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
      kernel_assemblePreConditioner_elastic_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        M,m->nvars,m->image);
    } else if (m->parStrategy_flag == CUDAPCG_EBE){
      kernel_assemblePreConditioner_elastic_2D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        M,m->nelem,m->image,m->ncols,m->nrows);
    }
    flag_PreConditionerWasAssembled = CUDAPCG_TRUE;
    return;
}
//------------------------------------------------------------------------------
void assemblePreConditioner_elastic_3D(cudapcgModel_t *m){
    if (m->parametric_density_field_flag == CUDAPCG_TRUE){
      flag_PreConditionerWasAssembled = CUDAPCG_FALSE;
      return;
    }
    if (m->parStrategy_flag == CUDAPCG_NBN){
      // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
      kernel_assemblePreConditioner_elastic_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        M,m->nvars,m->image);
    } else if (m->parStrategy_flag == CUDAPCG_EBE){
      kernel_assemblePreConditioner_elastic_3D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        M,m->nelem,m->image,m->ncols,m->nrows,m->nlayers);
    }
    flag_PreConditionerWasAssembled = CUDAPCG_TRUE;
    return;
}
//------------------------------------------------------------------------------
void applyPreConditioner_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res){
    if (flag_PreConditionerWasAssembled){
        kernel_termbytermmul<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(M,v1,m->nvars,res);
        if (v2 != NULL && scl != 0.0)
          kernel_saxpy_iny<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,v2,scl,m->nvars);
        return;
    }
    if (v2==NULL) v2=v1;
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_applyPreConditioner_thermal_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl,m->nvars,m->image,res);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        kernel_applyPreConditioner_thermal_2D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl,m->nelem,m->image,m->ncols,m->nrows,res);
      }
    } else {
      kernel_applyPreConditioner_thermal_2D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v1,v2,scl,m->nelem,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,res);
    }
    return;
}
//------------------------------------------------------------------------------
void applyPreConditioner_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res){
    if (flag_PreConditionerWasAssembled){
        kernel_termbytermmul<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(M,v1,m->nvars,res);
        if (v2 != NULL && scl != 0.0)
          kernel_saxpy_iny<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,v2,scl,m->nvars);
        return;
    }
    if (v2==NULL) v2=v1;
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_applyPreConditioner_thermal_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl,m->nvars,m->image,res);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        kernel_applyPreConditioner_thermal_3D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl,m->nelem,m->image,m->ncols,m->nrows,m->nlayers,res);
      }
    } else {
      kernel_applyPreConditioner_thermal_3D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v1,v2,scl,m->nelem,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,m->nlayers,res);
    }
    return;
}
//------------------------------------------------------------------------------
void applyPreConditioner_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res){
    if (flag_PreConditionerWasAssembled){
        kernel_termbytermmul<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(M,v1,m->nvars,res);
        if (v2 != NULL && scl != 0.0)
          kernel_saxpy_iny<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,v2,scl,m->nvars);
        return;
    }
    if (v2==NULL) v2=v1;
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_applyPreConditioner_elastic_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl,m->nvars,m->image,res);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        kernel_applyPreConditioner_elastic_2D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl,m->nelem,m->image,m->ncols,m->nrows,res);
      }
    } else {
      kernel_applyPreConditioner_elastic_2D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v1,v2,scl,m->nelem,m->image,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,res);
    }
    return;
}
//------------------------------------------------------------------------------
void applyPreConditioner_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res){
    if (flag_PreConditionerWasAssembled){
        kernel_termbytermmul<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(M,v1,m->nvars,res);
        if (v2 != NULL && scl != 0.0)
          kernel_saxpy_iny<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,v2,scl,m->nvars);
        return;
    }
    if (v2==NULL) v2=v1;
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
          // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
          kernel_applyPreConditioner_elastic_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl,m->nvars,m->image,res);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
          kernel_applyPreConditioner_elastic_3D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl,m->nelem,m->image,m->ncols,m->nrows,m->nlayers,res);
      }
    } else {
      kernel_applyPreConditioner_elastic_3D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v1,v2,scl,m->nelem,m->image,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,m->nlayers,res);
    }
    return;
}
//------------------------------------------------------------------------------
void applyPreConditioner_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res){

    //if (v2==NULL) v2=v1;

    if (m->parStrategy_flag != CUDAPCG_NBN) return;

    if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

      // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
      kernel_applyPreConditioner_fluid_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,res);

    } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

      kernel_applyPreConditioner_fluid_2D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,m->nporenodes,res);

      kernel_applyPreConditioner_fluid_2D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,m->nbordernodes,m->border_pore_map,m->nporenodes,res);

    }
    return;
}
//------------------------------------------------------------------------------
void applyPreConditioner_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl, cudapcgVar_t *res){

    //if (v2==NULL) v2=v1;

    if (m->parStrategy_flag != CUDAPCG_NBN) return;
  
    if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

      // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
      kernel_applyPreConditioner_fluid_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,res);

    } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

      kernel_applyPreConditioner_fluid_3D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,m->nporenodes,res);

      kernel_applyPreConditioner_fluid_3D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,m->nbordernodes,m->border_pore_map,m->nporenodes,res);

    }
    return;
}
//------------------------------------------------------------------------------
void Aprod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t * res){
    
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_thermal_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nvars,m->image,m->ncols,m->nrows,res,scl,isIncrement);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        if (!isIncrement) kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
        kernel_Aprod_thermal_2D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->image,m->ncols,m->nrows,res,scl);
      }
    } else {
      if (!isIncrement) kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
      kernel_Aprod_thermal_2D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nelem,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,res,scl);
    }

    return;
}
//------------------------------------------------------------------------------
void Aprod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t * res){
    
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_thermal_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nvars,m->image,m->ncols,m->nrows,m->nlayers,res,scl,isIncrement);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        if (!isIncrement) kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
        kernel_Aprod_thermal_3D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->image,m->ncols,m->nrows,m->nlayers,res,scl);
      }
    } else {
      if (!isIncrement) kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
      kernel_Aprod_thermal_3D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nelem,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,m->nlayers,res,scl);
    }

    return;
}
//------------------------------------------------------------------------------
void Aprod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t * res){
    
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_elastic_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nvars,m->image,m->ncols,m->nrows,res,scl,isIncrement);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        if (!isIncrement) kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
        kernel_Aprod_elastic_2D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->image,m->ncols,m->nrows,res,scl);
      }
    } else {
      if (!isIncrement) kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
      kernel_Aprod_elastic_2D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nelem,m->image,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,res,scl);
    }

    return;
}
//------------------------------------------------------------------------------
void Aprod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t * res){
    
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_elastic_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nvars,m->image,m->ncols,m->nrows,m->nlayers,res,scl,isIncrement);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        if (!isIncrement) kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
        kernel_Aprod_elastic_3D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->image,m->ncols,m->nrows,m->nlayers,res,scl);
      }
    } else {
      if (!isIncrement) kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
      kernel_Aprod_elastic_3D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nelem,m->image,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,m->nlayers,res,scl);
    }

    return;
}
//------------------------------------------------------------------------------
void Aprod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t * res){

    if (m->parStrategy_flag != CUDAPCG_NBN) return;

    if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

      // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
      kernel_Aprod_fluid_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,res,scl,isIncrement);

    } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

      kernel_Aprod_fluid_2D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,res,scl,isIncrement);

      kernel_Aprod_fluid_2D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,res,scl,isIncrement);


    }
    return;
}
//------------------------------------------------------------------------------
void Aprod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgFlag_t isIncrement, cudapcgVar_t * res){

    if (m->parStrategy_flag != CUDAPCG_NBN) return;

    if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

      // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
      kernel_Aprod_fluid_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,res,scl,isIncrement);

    } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

      kernel_Aprod_fluid_3D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,m->nlayers,res,scl,isIncrement);

      kernel_Aprod_fluid_3D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,res,scl,isIncrement);

    }
    return;
}
//------------------------------------------------------------------------------
