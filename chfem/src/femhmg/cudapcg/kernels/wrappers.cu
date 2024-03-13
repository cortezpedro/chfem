/*
  =====================================================================
  Universidade Federal Fluminense (UFF) - Niteroi, Brazil
  Institute of Computing
  Authors: Cortez Lopes, P., Pereira., A.
  contact: pedrocortez@id.uff.br
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
double absmax_signed_dot_res_1(unsigned int dim){ // dim: dimension of the vector that has been reduced to dot_res_1
    unsigned int blockDim = THREADS_PER_BLOCK;
    unsigned int gridDim = CEIL(dim,blockDim);
    unsigned int isRes1or2 = 1;
    while(dim > THREADS_PER_BLOCK){
      dim = gridDim;
      gridDim = CEIL(dim,blockDim);
      if (isRes1or2 == 1)
        kernel_absmax_signed<double><<<gridDim,blockDim>>>(dot_res_1,dim,dot_res_2);
      else
        kernel_absmax_signed<double><<<gridDim,blockDim>>>(dot_res_2,dim,dot_res_1);
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
    sz =sizeof(double)*CEIL(sz,THREADS_PER_BLOCK);
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
void scale(cudapcgVar_t * v, cudapcgVar_t scl, unsigned int dim){
    kernel_scale<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,scl,dim);
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
    return (cudapcgVar_t) max_dot_res_1(dim);
}
//------------------------------------------------------------------------------
cudapcgVar_t absmax(cudapcgVar_t *v, unsigned int dim){
    kernel_absmax<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim,dot_res_1);
    return (cudapcgVar_t) max_dot_res_1(dim);
}
//------------------------------------------------------------------------------
cudapcgVar_t absmax_signed(cudapcgVar_t *v, unsigned int dim){
    kernel_absmax_signed<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim,dot_res_1);
    return (cudapcgVar_t) absmax_signed_dot_res_1(dim);
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
double reduce_with_stride(cudapcgVar_t *v, unsigned int dim, unsigned int stride, unsigned int shift){
  #ifdef NO_XREDUCE_STAB
  kernel_reduce_with_stride<cudapcgVar_t><<<CEIL(dim/stride,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim/stride,stride,shift,dot_res_1);
  return reduce_dot_res_1(dim);
  #else
  kernel_reduce_positive_values_with_stride<cudapcgVar_t><<<CEIL(dim/stride,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim/stride,stride,shift,dot_res_1);
  double result = reduce_dot_res_1(dim);
  kernel_reduce_negative_values_with_stride<cudapcgVar_t><<<CEIL(dim/stride,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim/stride,stride,shift,dot_res_1);
  result += reduce_dot_res_1(dim);
  return result;
  #endif
}
//------------------------------------------------------------------------------
double reduce_with_stride_and_scale(cudapcgVar_t *v, unsigned int dim, unsigned int stride, unsigned int shift, cudapcgVar_t scl){
  #ifdef NO_XREDUCE_STAB
  kernel_reduce_with_stride_and_scale<cudapcgVar_t><<<CEIL(dim/stride,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim/stride,stride,shift,scl,dot_res_1);
  return reduce_dot_res_1(dim);
  #else
  kernel_reduce_positive_values_with_stride_and_scale<cudapcgVar_t><<<CEIL(dim/stride,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim/stride,stride,shift,scl,dot_res_1);
  double result = reduce_dot_res_1(dim);
  kernel_reduce_negative_values_with_stride_and_scale<cudapcgVar_t><<<CEIL(dim/stride,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim/stride,stride,shift,scl,dot_res_1);
  result += reduce_dot_res_1(dim);
  return result;
  #endif
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
void axpy(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim, cudapcgVar_t * res){
    kernel_axpy<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(y,x,a,dim,res);
    return;
}
//------------------------------------------------------------------------------
void axpy_iny(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim){
    kernel_axpy_iny<cudapcgVar_t><<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(y,x,a,dim);
    return;
}
//------------------------------------------------------------------------------
void axpy_iny_with_stride(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim, unsigned int stride, unsigned int shift){
    kernel_axpy_iny_with_stride<cudapcgVar_t><<<CEIL(dim/stride,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(y,x,a,dim/stride,stride,shift);
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
    size_t voxels_halfres = (rows > 1 ? rows/2 : 1)*(cols > 1 ? cols/2 : 1)*(layers > 1 ? layers/2 : 1);
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
void applyPreConditioner_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){
    if (flag_PreConditionerWasAssembled){
        kernel_termbytermmul<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(M,v1,m->nvars,res);
        kernel_scale<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl1,m->nvars);
        if (v2 != NULL && scl2 != 0.0)
          kernel_axpy_iny<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,v2,scl2,m->nvars);
        return;
    }
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_applyPreConditioner_thermal_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl1,scl2,m->nvars,m->image,res);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        kernel_applyPreConditioner_thermal_2D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl1,scl2,m->nelem,m->image,m->ncols,m->nrows,res);
      }
    } else {
      kernel_applyPreConditioner_thermal_2D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v1,v2,scl1,scl2,m->nelem,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,res);
    }
    return;
}
//------------------------------------------------------------------------------
void applyPreConditioner_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){
    if (flag_PreConditionerWasAssembled){
        kernel_termbytermmul<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(M,v1,m->nvars,res);
        kernel_scale<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl1,m->nvars);
        if (v2 != NULL && scl2 != 0.0)
          kernel_axpy_iny<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,v2,scl2,m->nvars);
        return;
    }
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_applyPreConditioner_thermal_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl1,scl2,m->nvars,m->image,res);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        kernel_applyPreConditioner_thermal_3D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl1,scl2,m->nelem,m->image,m->ncols,m->nrows,m->nlayers,res);
      }
    } else {
      kernel_applyPreConditioner_thermal_3D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v1,v2,scl1,scl2,m->nelem,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,m->nlayers,res);
    }
    return;
}
//------------------------------------------------------------------------------
void applyPreConditioner_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){
    if (flag_PreConditionerWasAssembled){
        kernel_termbytermmul<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(M,v1,m->nvars,res);
        kernel_scale<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl1,m->nvars);
        if (v2 != NULL && scl2 != 0.0)
          kernel_axpy_iny<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,v2,scl2,m->nvars);
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
          v1,v2,scl1,scl2,m->nvars,m->image,res);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        kernel_applyPreConditioner_elastic_2D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl1,scl2,m->nelem,m->image,m->ncols,m->nrows,res);
      }
    } else {
      kernel_applyPreConditioner_elastic_2D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v1,v2,scl1,scl2,m->nelem,m->image,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,res);
    }
    return;
}
//------------------------------------------------------------------------------
void applyPreConditioner_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){
    if (flag_PreConditionerWasAssembled){
        kernel_termbytermmul<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(M,v1,m->nvars,res);
        kernel_scale<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl1,m->nvars);
        if (v2 != NULL && scl2 != 0.0)
          kernel_axpy_iny<cudapcgVar_t><<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,v2,scl2,m->nvars);
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
          v1,v2,scl1,scl2,m->nvars,m->image,res);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
          kernel_applyPreConditioner_elastic_3D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl1,scl2,m->nelem,m->image,m->ncols,m->nrows,m->nlayers,res);
      }
    } else {
      kernel_applyPreConditioner_elastic_3D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v1,v2,scl1,scl2,m->nelem,m->image,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,m->nlayers,res);
    }
    return;
}
//------------------------------------------------------------------------------
void applyPreConditioner_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){

    if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented for FLUID

    if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

      if(m->SBS_flag){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_applyPreConditioner_fluid_2D_StencilByStencil<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          v1,v2,scl1,scl2,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,res);
      } else {
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_applyPreConditioner_fluid_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl1,scl2,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,res);
      }
      

    } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

      if(m->SBS_flag){
        kernel_applyPreConditioner_fluid_2D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
            v1,v2,scl1,scl2,m->nporenodes,res);

        kernel_applyPreConditioner_fluid_2D_StencilByStencil_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
            v1,v2,scl1,scl2,m->nbordernodes,m->border_pore_map,m->nporenodes,res);
      } else {
        kernel_applyPreConditioner_fluid_2D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
            #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
              K,
            #endif
            v1,v2,scl1,scl2,m->nporenodes,res);

        kernel_applyPreConditioner_fluid_2D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
            #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
              K,
            #endif
            v1,v2,scl1,scl2,m->nbordernodes,m->border_pore_map,m->nporenodes,res);
      }

    }
    return;
}
//------------------------------------------------------------------------------
void applyPreConditioner_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){

    if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented for FLUID
  
    if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

      if(m->SBS_flag){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_applyPreConditioner_fluid_3D_StencilByStencil<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          v1,v2,scl1,scl2,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,res);
      } else {
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_applyPreConditioner_fluid_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl1,scl2,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,res);
      }
      
    } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

      if(m->SBS_flag){
        kernel_applyPreConditioner_fluid_3D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          v1,v2,scl1,scl2,m->nporenodes,res);

        kernel_applyPreConditioner_fluid_3D_StencilByStencil_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          v1,v2,scl1,scl2,m->nbordernodes,m->border_pore_map,m->nporenodes,res);

      } else {
        kernel_applyPreConditioner_fluid_3D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl1,scl2,m->nporenodes,res);

        kernel_applyPreConditioner_fluid_3D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v1,v2,scl1,scl2,m->nbordernodes,m->border_pore_map,m->nporenodes,res);
      }
    }
    return;
}
//------------------------------------------------------------------------------
void applyinvPreConditioner_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){
  printf("WARNING: \"applyinvPreConditioner\" has not been implemented for THERMAL_2D yet. Try another solver or remove preconditioning.\n");
  return;
}
//------------------------------------------------------------------------------
void applyinvPreConditioner_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){
  printf("WARNING: \"applyinvPreConditioner\" has not been implemented for THERMAL_3D yet. Try another solver or remove preconditioning.\n");
  return;
}
//------------------------------------------------------------------------------
void applyinvPreConditioner_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){
  printf("WARNING: \"applyinvPreConditioner\" has not been implemented for ELASTIC_2D yet. Try another solver or remove preconditioning.\n");
  return;
}
//------------------------------------------------------------------------------
void applyinvPreConditioner_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){
  printf("WARNING: \"applyinvPreConditioner\" has not been implemented for ELASTIC_3D yet. Try another solver or remove preconditioning.\n");
  return;
}
//------------------------------------------------------------------------------
void applyinvPreConditioner_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){

    if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented for FLUID
    if (m->poremap_flag != CUDAPCG_POREMAP_NUM) return; // not implemented

    kernel_applyinvPreConditioner_fluid_2D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        v1,v2,scl1,scl2,m->nporenodes,res);

    kernel_applyinvPreConditioner_fluid_2D_StencilByStencil_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        v1,v2,scl1,scl2,m->nbordernodes,m->border_pore_map,m->nporenodes,res);
    return;
}
//------------------------------------------------------------------------------
void applyinvPreConditioner_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl1, cudapcgVar_t scl2, cudapcgVar_t *res){
    if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented for FLUID
    if (m->poremap_flag != CUDAPCG_POREMAP_NUM) return; // not implemented

    kernel_applyinvPreConditioner_fluid_3D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        v1,v2,scl1,scl2,m->nporenodes,res);

    kernel_applyinvPreConditioner_fluid_3D_StencilByStencil_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        v1,v2,scl1,scl2,m->nbordernodes,m->border_pore_map,m->nporenodes,res);
    return;
}
//------------------------------------------------------------------------------
void Aprod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t * res){
    
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_thermal_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nvars,m->image,m->ncols,m->nrows,res,scl,scl_prev);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        if (scl_prev==0.0) 
          kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
        else
          kernel_scale<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl_prev,m->nvars);
        kernel_Aprod_thermal_2D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->image,m->ncols,m->nrows,res,scl);
      }
    } else {
      if (scl_prev==0.0) 
        kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
      else
        kernel_scale<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl_prev,m->nvars);
      kernel_Aprod_thermal_2D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nelem,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,res,scl);
    }

    return;
}
//------------------------------------------------------------------------------
void Aprod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t * res){
    
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_thermal_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nvars,m->image,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        if (scl_prev==0.0) 
          kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
        else
          kernel_scale<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl_prev,m->nvars);
        kernel_Aprod_thermal_3D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->image,m->ncols,m->nrows,m->nlayers,res,scl);
      }
    } else {
      if (scl_prev==0.0) 
        kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
      else
        kernel_scale<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl_prev,m->nvars);
      kernel_Aprod_thermal_3D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nelem,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,m->nlayers,res,scl);
    }

    return;
}
//------------------------------------------------------------------------------
void Aprod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t * res){
    
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_elastic_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nvars,m->image,m->ncols,m->nrows,res,scl,scl_prev);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        if (scl_prev==0.0) 
          kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
        else
          kernel_scale<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl_prev,m->nvars);
        kernel_Aprod_elastic_2D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->image,m->ncols,m->nrows,res,scl);
      }
    } else {
      if (scl_prev==0.0) 
        kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
      else
        kernel_scale<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl_prev,m->nvars);
      kernel_Aprod_elastic_2D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nelem,m->image,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,res,scl);
    }

    return;
}
//------------------------------------------------------------------------------
void Aprod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t * res){
    
    if (m->parametric_density_field_flag == CUDAPCG_FALSE){
      if (m->parStrategy_flag == CUDAPCG_NBN){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_elastic_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nvars,m->image,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);
      } else if (m->parStrategy_flag == CUDAPCG_EBE){
        if (scl_prev==0.0) 
          kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
        else
          kernel_scale<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl_prev,m->nvars);
        kernel_Aprod_elastic_3D_ElemByElem<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->image,m->ncols,m->nrows,m->nlayers,res,scl);
      }
    } else {
      if (scl_prev==0.0) 
        kernel_zeros<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,m->nvars);
      else
        kernel_scale<<<CEIL(m->nvars,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,scl_prev,m->nvars);
      kernel_Aprod_elastic_3D_ElemByElem_ScalarDensityField<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nelem,m->image,m->parametric_density_field,m->limits_density_field[0],m->limits_density_field[1],m->ncols,m->nrows,m->nlayers,res,scl);
    }

    return;
}
//------------------------------------------------------------------------------
void Aprod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t * res){

    if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented for FLUID

    if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

      if(m->SBS_flag){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_fluid_2D_StencilByStencil<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,res,scl,scl_prev);
      } else {
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_fluid_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,res,scl,scl_prev);
      }

    } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

      if(m->SBS_flag){
        kernel_Aprod_fluid_2D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,res,scl,scl_prev);
      } else {
        kernel_Aprod_fluid_2D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,res,scl,scl_prev);
      }

      kernel_Aprod_fluid_2D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
        #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
          K,
        #endif
        v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,res,scl,scl_prev);
    }
    return;
}
//------------------------------------------------------------------------------
void Aprod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t * v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t * res){

    if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented for FLUID

    if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

      if(m->SBS_flag){
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_fluid_3D_StencilByStencil<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);
      } else {
        // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
        kernel_Aprod_fluid_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);
      }

    } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

      if(m->SBS_flag){
        kernel_Aprod_fluid_3D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);
      } else {
        kernel_Aprod_fluid_3D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);
      }

      kernel_Aprod_fluid_3D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
          #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
            K,
          #endif
          v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);

    }
    return;
}
//------------------------------------------------------------------------------
void PreConditionerAprod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res){
  if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented
  if (m->parametric_density_field_flag != CUDAPCG_FALSE) return; // not implemented
  kernel_PreConditionerAprod_thermal_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    v,m->nvars,m->image,m->ncols,m->nrows,res,scl,scl_prev);
  return;
}
//------------------------------------------------------------------------------
void PreConditionerAprod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res){
  if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented
  if (m->parametric_density_field_flag != CUDAPCG_FALSE) return; // not implemented
  kernel_PreConditionerAprod_thermal_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    v,m->nvars,m->image,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);
  return;
}
//------------------------------------------------------------------------------
void PreConditionerAprod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res){
  if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented
  if (m->parametric_density_field_flag != CUDAPCG_FALSE) return; // not implemented
  kernel_PreConditionerAprod_elastic_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    v,m->nvars,m->image,m->ncols,m->nrows,res,scl,scl_prev);
  return;
}
//------------------------------------------------------------------------------
void PreConditionerAprod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res){
  if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented
  if (m->parametric_density_field_flag != CUDAPCG_FALSE) return; // not implemented
  kernel_PreConditionerAprod_elastic_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    v,m->nvars,m->image,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);
  return;
}
//------------------------------------------------------------------------------
void PreConditionerAprod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res){
  if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented
  if (m->poremap_flag != CUDAPCG_POREMAP_NUM) return; // not implemented
  if(m->SBS_flag){
    kernel_PreConditionerAprod_fluid_2D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,res,scl,scl_prev);
  } else {
    kernel_PreConditionerAprod_fluid_2D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,res,scl,scl_prev);
  }

  kernel_PreConditionerAprod_fluid_2D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,res,scl,scl_prev);
  return;
}
//------------------------------------------------------------------------------
void PreConditionerAprod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl, cudapcgVar_t scl_prev, cudapcgVar_t *res){
  if (m->parStrategy_flag != CUDAPCG_NBN) return; // EBE has not been implemented
  if (m->poremap_flag != CUDAPCG_POREMAP_NUM) return; // not implemented
  if(m->SBS_flag){
    kernel_PreConditionerAprod_fluid_3D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);
  } else {
    kernel_PreConditionerAprod_fluid_3D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);
  }

  kernel_PreConditionerAprod_fluid_3D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,res,scl,scl_prev);
  
  return;
}
//------------------------------------------------------------------------------
double dotPreConditioner_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  kernel_dotPreConditioner_thermal_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v1,v2,m->nvars,m->image,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotPreConditioner_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  kernel_dotPreConditioner_thermal_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v1,v2,m->nvars,m->image,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotPreConditioner_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  kernel_dotPreConditioner_elastic_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v1,v2,m->nvars,m->image,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotPreConditioner_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  kernel_dotPreConditioner_elastic_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v1,v2,m->nvars,m->image,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotPreConditioner_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  double res=0.0;
  if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

    if(m->SBS_flag){
      kernel_dotPreConditioner_fluid_2D_StencilByStencil<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v1,v2,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,dot_res_1);
    } else {
      kernel_dotPreConditioner_fluid_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v1,v2,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,dot_res_1);
    }
    
    res = ((double)scl)*reduce_dot_res_1(m->nelem);

  } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

    if(m->SBS_flag){
      kernel_dotPreConditioner_fluid_2D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v1,v2,m->nporenodes,dot_res_1);

      res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

      kernel_dotPreConditioner_fluid_2D_StencilByStencil_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v1,v2,m->nbordernodes,m->border_pore_map,m->nporenodes,dot_res_1);
    } else {
      kernel_dotPreConditioner_fluid_2D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v1,v2,m->nporenodes,dot_res_1);

      res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

      kernel_dotPreConditioner_fluid_2D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v1,v2,m->nbordernodes,m->border_pore_map,m->nporenodes,dot_res_1);
    }

    res += ((double)scl)*reduce_dot_res_1(m->nbordernodes);

  }
  return res;
}
//------------------------------------------------------------------------------
double dotPreConditioner_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  double res=0.0;
  if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

    if(m->SBS_flag){
      kernel_dotPreConditioner_fluid_3D_StencilByStencil<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v1,v2,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,dot_res_1);
    } else {
      kernel_dotPreConditioner_fluid_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v1,v2,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,dot_res_1);
    }
    
    res = ((double)scl)*reduce_dot_res_1(m->nelem);

  } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

    if(m->SBS_flag){
      kernel_dotPreConditioner_fluid_3D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v1,v2,m->nporenodes,dot_res_1);

      res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

      kernel_dotPreConditioner_fluid_3D_StencilByStencil_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v1,v2,m->nbordernodes,m->border_pore_map,m->nporenodes,dot_res_1);
    } else {
      kernel_dotPreConditioner_fluid_3D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v1,v2,m->nporenodes,dot_res_1);

      res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

      kernel_dotPreConditioner_fluid_3D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v1,v2,m->nbordernodes,m->border_pore_map,m->nporenodes,dot_res_1);
    }

    res += ((double)scl)*reduce_dot_res_1(m->nbordernodes);

  }
  return res;
}
//------------------------------------------------------------------------------
double dotinvPreConditioner_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  printf("WARNING: \"dotinvPreConditioner\" has not been implemented for THERMAL_2D yet. Try another solver or remove preconditioning.\n");
  return 0.0;
}
//------------------------------------------------------------------------------
double dotinvPreConditioner_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  printf("WARNING: \"dotinvPreConditioner\" has not been implemented for THERMAL_3D yet. Try another solver or remove preconditioning.\n");
  return 0.0;
}
//------------------------------------------------------------------------------
double dotinvPreConditioner_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  printf("WARNING: \"dotinvPreConditioner\" has not been implemented for ELASTIC_2D yet. Try another solver or remove preconditioning.\n");
  return 0.0;
}
//------------------------------------------------------------------------------
double dotinvPreConditioner_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  printf("WARNING: \"dotinvPreConditioner\" has not been implemented for ELASTIC_3D yet. Try another solver or remove preconditioning.\n");
  return 0.0;
}
//------------------------------------------------------------------------------
double dotinvPreConditioner_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  double res=0.0;
  
  kernel_dotinvPreConditioner_fluid_2D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  v1,v2,m->nporenodes,dot_res_1);

  res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

  kernel_dotinvPreConditioner_fluid_2D_StencilByStencil_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  v1,v2,m->nbordernodes,m->border_pore_map,m->nporenodes,dot_res_1);

  res += ((double)scl)*reduce_dot_res_1(m->nbordernodes);

  return res;
}
//------------------------------------------------------------------------------
double dotinvPreConditioner_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v1, cudapcgVar_t *v2, cudapcgVar_t scl){
  double res=0.0;
  
  kernel_dotinvPreConditioner_fluid_3D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  v1,v2,m->nporenodes,dot_res_1);

  res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

  kernel_dotinvPreConditioner_fluid_3D_StencilByStencil_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  v1,v2,m->nbordernodes,m->border_pore_map,m->nporenodes,dot_res_1);

  res += ((double)scl)*reduce_dot_res_1(m->nbordernodes);

  return res;
}
//------------------------------------------------------------------------------
double dotAprod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
  kernel_dotAprod_thermal_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v,m->nvars,m->image,m->ncols,m->nrows,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotAprod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
  kernel_dotAprod_thermal_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v,m->nvars,m->image,m->ncols,m->nrows,m->nlayers,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotAprod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
  kernel_dotAprod_elastic_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v,m->nvars,m->image,m->ncols,m->nrows,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotAprod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
  kernel_dotAprod_elastic_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v,m->nvars,m->image,m->ncols,m->nrows,m->nlayers,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotAprod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  double res=0.0;
  if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

    if(m->SBS_flag){
      kernel_dotAprod_fluid_2D_StencilByStencil<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,dot_res_1);
    } else {
      kernel_dotAprod_fluid_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,dot_res_1);
    }

    res = ((double)scl)*reduce_dot_res_1(m->nelem);

  } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

    if(m->SBS_flag){
      kernel_dotAprod_fluid_2D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,dot_res_1);
    } else {
      kernel_dotAprod_fluid_2D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,dot_res_1);
    }
    res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

    kernel_dotAprod_fluid_2D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,dot_res_1);

    res += ((double)scl)*reduce_dot_res_1(m->nbordernodes);

  }
  return res;
}
//------------------------------------------------------------------------------
double dotAprod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  double res=0.0;
  if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

    if(m->SBS_flag){
      kernel_dotAprod_fluid_3D_StencilByStencil<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,dot_res_1);
    } else {
      kernel_dotAprod_fluid_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,dot_res_1);
    }
    
    res = ((double)scl)*reduce_dot_res_1(m->nelem);

  } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

    if(m->SBS_flag){
      kernel_dotAprod_fluid_3D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,m->nlayers,dot_res_1);
    } else {
      kernel_dotAprod_fluid_3D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,m->nlayers,dot_res_1);
    }
    
    res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

    kernel_dotAprod_fluid_3D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,dot_res_1);

    res += ((double)scl)*reduce_dot_res_1(m->nbordernodes);

  }
  return res;
}
//------------------------------------------------------------------------------
double dotA2prod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
  kernel_dotA2prod_thermal_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v,m->nvars,m->image,m->ncols,m->nrows,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotA2prod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
  kernel_dotA2prod_thermal_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v,m->nvars,m->image,m->ncols,m->nrows,m->nlayers,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotA2prod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
  kernel_dotA2prod_elastic_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v,m->nvars,m->image,m->ncols,m->nrows,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotA2prod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  if (m->parametric_density_field_flag == CUDAPCG_TRUE) return 0.0; // has not been implemented
  // Obs.: Grid is dimensioned with model->nelem because it is equivalent numerically to (valid_nodes/dof_per_node)
  kernel_dotA2prod_elastic_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v,m->nvars,m->image,m->ncols,m->nrows,m->nlayers,dot_res_1);
  return ((double)scl)*reduce_dot_res_1(m->nelem);
}
//------------------------------------------------------------------------------
double dotA2prod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  double res=0.0;
  if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

    if(m->SBS_flag){
      kernel_dotA2prod_fluid_2D_StencilByStencil<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,dot_res_1);
    } else {
      kernel_dotA2prod_fluid_2D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,dot_res_1);
    }

    res = ((double)scl)*reduce_dot_res_1(m->nelem);

  } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

    if(m->SBS_flag){
      kernel_dotA2prod_fluid_2D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,dot_res_1);
    } else {
      kernel_dotA2prod_fluid_2D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,dot_res_1);
    }
    res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

    kernel_dotA2prod_fluid_2D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,dot_res_1);

    res += ((double)scl)*reduce_dot_res_1(m->nbordernodes);

  }
  return res;
}
//------------------------------------------------------------------------------
double dotA2prod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN) return 0.0;               // has not been implemented
  double res=0.0;
  if (m->poremap_flag == CUDAPCG_POREMAP_IMG){

    if(m->SBS_flag){
      kernel_dotA2prod_fluid_3D_StencilByStencil<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,dot_res_1);
    } else {
      kernel_dotA2prod_fluid_3D_NodeByNode<<<CEIL(m->nelem,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nelem,m->pore_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,dot_res_1);
    }
    
    res = ((double)scl)*reduce_dot_res_1(m->nelem);

  } else if (m->poremap_flag == CUDAPCG_POREMAP_NUM){

    if(m->SBS_flag){
      kernel_dotA2prod_fluid_3D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,m->nlayers,dot_res_1);
    } else {
      kernel_dotA2prod_fluid_3D_NodeByNode_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
      #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
        K,
      #endif
      v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,m->nlayers,dot_res_1);
    }
    
    res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

    kernel_dotA2prod_fluid_3D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
      K,
    #endif
    v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,dot_res_1);

    res += ((double)scl)*reduce_dot_res_1(m->nbordernodes);

  }
  return res;
}
//------------------------------------------------------------------------------
double dotPreConditionerA2prod_thermal_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  printf("WARNING: \"dotPreConditionerA2prod\" has not been implemented for THERMAL_2D yet. Try another solver or remove preconditioning.\n");
  return 0.0;
}
//------------------------------------------------------------------------------
double dotPreConditionerA2prod_thermal_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  printf("WARNING: \"dotPreConditionerA2prod\" has not been implemented for THERMAL_3D yet. Try another solver or remove preconditioning.\n");
  return 0.0;
}
//------------------------------------------------------------------------------
double dotPreConditionerA2prod_elastic_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  printf("WARNING: \"dotPreConditionerA2prod\" has not been implemented for ELASTIC_2D yet. Try another solver or remove preconditioning.\n");
  return 0.0;
}
//------------------------------------------------------------------------------
double dotPreConditionerA2prod_elastic_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  printf("WARNING: \"dotPreConditionerA2prod\" has not been implemented for ELASTIC_3D yet. Try another solver or remove preconditioning.\n");
  return 0.0;
}
//------------------------------------------------------------------------------
double dotPreConditionerA2prod_fluid_2D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN || m->poremap_flag != CUDAPCG_POREMAP_NUM){
    printf("WARNING: \"dotPreConditionerA2prod\" has only been implemented with NBN and POREMAP_NUM for FLUID_2D. Try another solver or remove preconditioning.\n");
    return 0.0;
  }
  double res=0.0;
  kernel_dotPreConditionerA2prod_fluid_2D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,dot_res_1);
  
  res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

  kernel_dotPreConditionerA2prod_fluid_2D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,dot_res_1);

  res += ((double)scl)*reduce_dot_res_1(m->nbordernodes);
  return res;
}
//------------------------------------------------------------------------------
double dotPreConditionerA2prod_fluid_3D(cudapcgModel_t *m, cudapcgVar_t *v, cudapcgVar_t scl){
  if (m->parStrategy_flag != CUDAPCG_NBN || m->poremap_flag != CUDAPCG_POREMAP_NUM){
    printf("WARNING: \"dotPreConditionerA2prod\" has only been implemented with NBN and POREMAP_NUM for FLUID_3D. Try another solver or remove preconditioning.\n");
    return 0.0;
  }
  double res=0.0;
  kernel_dotPreConditionerA2prod_fluid_3D_StencilByStencil_Pore<<<CEIL(m->nporenodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
    v,m->nporenodes,m->DOF2periodic_map,m->periodic2DOF_map,m->ncols,m->nrows,m->nlayers,dot_res_1);
  
  res = ((double)scl)*reduce_dot_res_1(m->nporenodes);

  kernel_dotPreConditionerA2prod_fluid_3D_NodeByNode_Border<<<CEIL(m->nbordernodes,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(
  #if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
    K,
  #endif
  v,m->nbordernodes,m->border_pore_map,m->DOF2periodic_map,m->periodic2DOF_map,m->nporenodes,m->ncols,m->nrows,m->nlayers,dot_res_1);

  res += ((double)scl)*reduce_dot_res_1(m->nbordernodes);
  return res;
}
//------------------------------------------------------------------------------
