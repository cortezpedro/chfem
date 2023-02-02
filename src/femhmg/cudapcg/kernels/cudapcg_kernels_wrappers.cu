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

#include "cudapcg_kernels_wrappers.h"
#include "cudapcg_kernels_source.h"
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
////////// CUDA KERNELS ///////////
////////// (VECTOR OPS) ///////////
///////////////////////////////////
//---------------------------------

/*
    Relatively standard kernels, to perform
    vector operations. These are all generic
    (no compromises with the image-based
    matrix-free approach)
*/

//------------------------------------------------------------------------------
// Kernel to copy data from an array to another
__global__ void kernel_arrcpy(cudapcgVar_t *v, cudapcgVar_t *res, unsigned int dim){
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<dim)
        res[i] = v[i];
}
//------------------------------------------------------------------------------
// Kernel to fill a vector with zeros
__global__ void kernel_zeros(cudapcgVar_t * v, unsigned int dim){
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<dim)
        v[i] = 0.0;
}
//------------------------------------------------------------------------------
// Kernel to get max absolute value within an array in gpu
__global__ void kernel_absmax(cudapcgVar_t *v, unsigned int dim, double *res){
  // Get local and global thread index
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  // shmem cache
  __shared__ cudapcgVar_t cache[THREADS_PER_BLOCK];
  // Fill cache
  if (gi<dim)
    cache[li] = CUDA_ABS(v[gi]);
  else
    cache[li] = 0.0; // for safety
  __syncthreads();
  // Init stride var
  unsigned int stride = THREADS_PER_BLOCK/2;
  // Keep going until stride is 0 (finished adding 2by2)
  while(stride){
    if (li < stride)
      cache[li] = cache[li+stride] > cache[li] ? cache[li+stride] : cache[li];
    __syncthreads();
    stride/=2;
  }
  // Put cache results into result global mem arr
  if (li==0)
    res[blockIdx.x] = (double)cache[0];
}
//------------------------------------------------------------------------------
// Kernel to get max absolute value within an array in gpu
__global__ void kernel_absmax_double(double *v, unsigned int dim, double *res){
  // Get local and global thread index
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  // shmem cache
  __shared__ double cache[THREADS_PER_BLOCK];
  // Fill cache
  if (gi<dim)
    cache[li] = fabs(v[gi]);
  else
    cache[li] = 0.0; // for safety
  __syncthreads();
  // Init stride var
  unsigned int stride = THREADS_PER_BLOCK/2;
  // Keep going until stride is 0 (finished adding 2by2)
  while(stride){
    if (li < stride)
      cache[li] = cache[li+stride] > cache[li] ? cache[li+stride] : cache[li];
    __syncthreads();
    stride/=2;
  }
  // Put cache results into result global mem arr
  if (li==0)
    res[blockIdx.x] = cache[0];
}
//------------------------------------------------------------------------------
// Kernel to perform reduction of a vector, using shared mem
__global__ void kernel_reduce(cudapcgVar_t * v, unsigned int dim, double * res){
    // Get local and global thread index
    unsigned int li = threadIdx.x;
    unsigned int gi = li + blockIdx.x * blockDim.x;
    // shmem cache
    __shared__ double cache[THREADS_PER_BLOCK];
    // Fill cache
    if (gi<dim)
      cache[li] = (double) v[gi];
    else
      cache[li] = 0.0; // for safety
    __syncthreads();
    // Init stride cudapcgVar_t
    unsigned int stride = THREADS_PER_BLOCK/2;
    // Keep going until stride is 0 (finished adding 2by2)
    while(stride){
      if (li < stride)
        cache[li]+=cache[li+stride];
      __syncthreads();
      stride/=2;
    }
    // Put cache results into result global mem arr
    if (li==0)
      res[blockIdx.x] = cache[0];
}
//------------------------------------------------------------------------------
// Kernel to perform reduction of a vector, using shared mem
__global__ void kernel_reduce_double(double * v, unsigned int dim, double * res){
    // Get local and global thread index
    unsigned int li = threadIdx.x;
    unsigned int gi = li + blockIdx.x * blockDim.x;
    // shmem cache
    __shared__ double cache[THREADS_PER_BLOCK];
    // Fill cache
    if (gi<dim)
      cache[li] = v[gi];
    else
      cache[li] = 0.0; // for safety
    __syncthreads();
    // Init stride cudapcgVar_t
    unsigned int stride = THREADS_PER_BLOCK/2;
    // Keep going until stride is 0 (finished adding 2by2)
    while(stride){
      if (li < stride)
        cache[li]+=cache[li+stride];
      __syncthreads();
      stride/=2;
    }
    // Put cache results into result global mem arr
    if (li==0)
      res[blockIdx.x] = cache[0];
}
//------------------------------------------------------------------------------
// Kernel to perform reduction of a vector, using shared mem
__global__ void kernel_absreduce(cudapcgVar_t * v, unsigned int dim, double * res){
    // Get local and global thread index
    unsigned int li = threadIdx.x;
    unsigned int gi = li + blockIdx.x * blockDim.x;
    // shmem cache
    __shared__ double cache[THREADS_PER_BLOCK];
    // Fill cache
    if (gi<dim)
      cache[li] = (double) CUDA_ABS(v[gi]);
    else
      cache[li] = 0.0; // for safety
    __syncthreads();
    // Init stride cudapcgVar_t
    unsigned int stride = THREADS_PER_BLOCK/2;
    // Keep going until stride is 0 (finished adding 2by2)
    while(stride){
      if (li < stride)
        cache[li]+=cache[li+stride];
      __syncthreads();
      stride/=2;
    }
    // Put cache results into result global mem arr
    if (li==0)
      res[blockIdx.x] = cache[0];
}
//------------------------------------------------------------------------------
// Kernel to perform dot product between two vectors, using shared mem
__global__ void kernel_dotprod(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, double * res){
  // Get local and global thread index
  unsigned int li = threadIdx.x;
  unsigned int gi = li + blockIdx.x * blockDim.x;
  // shmem cache
  __shared__ double cache[THREADS_PER_BLOCK];
  // Fill cache
  if (gi<dim)
    cache[li] = (double) (v1[gi]*v2[gi]);
  else
    cache[li] = 0.0; // for safety
  __syncthreads();
  // Init stride var
  unsigned int stride = THREADS_PER_BLOCK/2;
  // Keep going until stride is 0 (finished adding 2by2)
  while(stride){
    if (li < stride)
      cache[li]+=cache[li+stride];
    __syncthreads();
    stride/=2;
  }
  // Put cache results into result global mem arr
  if (li==0)
    res[blockIdx.x] = cache[0];
}
//------------------------------------------------------------------------------
// Kernel to perform term-by-term multiplication between two vectors
__global__ void kernel_termbytermmul(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim)
    res[i] = v1[i]*v2[i];
}
//------------------------------------------------------------------------------
// Kernel to perform term-by-term division between two vectors
__global__ void kernel_termbytermdiv(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim)
    res[i] = v1[i]/v2[i];
}
//------------------------------------------------------------------------------
// Kernel to perform term-by-term inversion of a vector
__global__ void kernel_termbyterminv(cudapcgVar_t * v, unsigned int dim){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim)
    v[i] = 1/v[i];
}
//------------------------------------------------------------------------------
// Kernel to sum two vectors, considering a scalar multiplier. res = y + a*x
__global__ void kernel_saxpy(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim)
    res[i] = y[i] + a*x[i];
}
//------------------------------------------------------------------------------
// Kernel to sum two vectors, considering a scalar multiplier.
// Result is stored into first array -> y += a*x
__global__ void kernel_saxpy_iny(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim)
    y[i] += a*x[i];
}
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
////////// CUDA KERNELS ///////////
/// (COARSENING/REFINEMENT OPS) ///
///////////////////////////////////
//---------------------------------

/*
    Kernels to project arrays from a coarse to a refined mesh,
    and vice-versa.
    Follows DOF numbering scheme.
*/
//------------------------------------------------------------------------------
// Kernels that projects vector v, from a coarse mesh, to res, on a refined mesh
// It is assumed that coarse mesh dimensions are [nx/2,ny/2,nz/2]
__global__ void kernel_project2(cudapcgVar_t *res, cudapcgVar_t *v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Compute coarse mesh dimensions
  unsigned int nx_coarse = nx/2;
  unsigned int ny_coarse = ny/2;
  unsigned int nz_coarse = nz/2;
  // Check if thread should work
  if (i<(nx_coarse*ny_coarse*(nz_coarse+(nz_coarse<1)))){
    // Find 3D indexes on coarse mesh
    unsigned int ii = i%(ny_coarse);
    unsigned int jj = (i%(nx_coarse*ny_coarse))/ny_coarse;
    unsigned int kk = i/(nx_coarse*ny_coarse);
    // Index on refined mesh
    unsigned int idx = (2*ii) + (2*jj)*ny + (2*kk)*nx*ny;
    // Project to refined mesh
    // Multiply by 2.0 to account for difference in model dimension
    for (unsigned int j=0; j<nLocalDOFs; j++)
      res[nLocalDOFs*idx+j] = 2.0*v[nLocalDOFs*i+j];
  }
}
//------------------------------------------------------------------------------
// Kernels that interpolate entries of vector v from coarse mesh to refined mesh
// It is assumed that coarse mesh dimensions are [nx/2,ny/2,nz/2]
__global__ void kernel_interpl_rows(cudapcgVar_t * v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if thread should work
  if (i<(nx*ny*(nz+(nz<1)))){
    // Find 3D indexes on refined mesh
    unsigned int ii = i%(ny);
    unsigned int jj = (i%(nx*ny))/ny;
    unsigned int kk = i/(nx*ny);
    // Index variable to navigate mesh
    unsigned int idx;
    // Check if must interpolate rows
    if ((ii&1) && !(jj&1) && !(kk&1)){
      idx = WALK_UP(i,ny);
      for (unsigned int j=0; j<nLocalDOFs; j++)
        v[nLocalDOFs*i+j]  = 0.5*v[nLocalDOFs*idx+j];
      idx = WALK_DOWN(i,ny);
      for (unsigned int j=0; j<nLocalDOFs; j++)
        v[nLocalDOFs*i+j] += 0.5*v[nLocalDOFs*idx+j];
    }
  }
}
//------------------------------------------------------------------------------
// Kernels that interpolate entries of vector v from coarse mesh to refined mesh
// It is assumed that coarse mesh dimensions are [nx/2,ny/2,nz/2]
__global__ void kernel_interpl_cols(cudapcgVar_t * v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if thread should work
  if (i<(nx*ny*(nz+(nz<1)))){
    // Find column and layer indexes on refined mesh
    unsigned int jj = (i%(nx*ny))/ny;
    unsigned int kk = i/(nx*ny);
    // Index variable to navigate mesh
    unsigned int idx;
    // Check if must interpolate cols
    if ((jj&1) && !(kk&1)){
      idx = WALK_LEFT((i%(nx*ny)),ny,(nx*ny)) + kk*(nx*ny);
      for (unsigned int j=0; j<nLocalDOFs; j++)
        v[nLocalDOFs*i+j]  = 0.5*v[nLocalDOFs*idx+j];
      idx = WALK_RIGHT((i%(nx*ny)),ny,(nx*ny)) + kk*(nx*ny);
      for (unsigned int j=0; j<nLocalDOFs; j++)
        v[nLocalDOFs*i+j] += 0.5*v[nLocalDOFs*idx+j];
    }
  }
}
//------------------------------------------------------------------------------
// Kernels that interpolate entries of vector v from coarse mesh to refined mesh
// It is assumed that coarse mesh dimensions are [nx/2,ny/2,nz/2]
__global__ void kernel_interpl_layers(cudapcgVar_t * v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if thread should work
  if (i<(nx*ny*(nz+(nz<1)))){
    // Find layer index on refined mesh
    unsigned int kk = i/(nx*ny);
    // Index variable to navigate mesh
    unsigned int idx;
    // Check if must interpolate layers
    if (kk&1){
      idx = WALK_NEAR(i,(nx*ny),(nx*ny*nz));
      for (unsigned int j=0; j<nLocalDOFs; j++)
        v[nLocalDOFs*i+j]  = 0.5*v[nLocalDOFs*idx+j];
      idx = WALK_FAR(i,(nx*ny),(nx*ny*nz));
      for (unsigned int j=0; j<nLocalDOFs; j++)
        v[nLocalDOFs*i+j] += 0.5*v[nLocalDOFs*idx+j];
    }
  }
}
//------------------------------------------------------------------------------

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
        kernel_reduce_double<<<gridDim,blockDim>>>(dot_res_1,dim,dot_res_2);
      else
        kernel_reduce_double<<<gridDim,blockDim>>>(dot_res_2,dim,dot_res_1);
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

//---------------------------------
///////////////////////////////////
//////// PUBLIC FUNCTIONS /////////
///////////////////////////////////
//---------------------------------

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
void arrcpy(cudapcgVar_t * res, cudapcgVar_t * v, unsigned int dim){
    kernel_arrcpy<<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,res,dim);
    return;
}
//------------------------------------------------------------------------------
void zeros(cudapcgVar_t * v, unsigned int dim){
    kernel_zeros<<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim);
    return;
}
//------------------------------------------------------------------------------
cudapcgVar_t absmax(cudapcgVar_t *v, unsigned int dim){
    unsigned int blockDim = THREADS_PER_BLOCK;
    unsigned int gridDim = CEIL(dim,blockDim);
    kernel_absmax<<<gridDim,blockDim>>>(v,dim,dot_res_1);
    unsigned int isRes1or2 = 1;
    while(dim > THREADS_PER_BLOCK){
      dim = gridDim;
      gridDim = CEIL(dim,blockDim);
      if (isRes1or2 == 1)
        kernel_absmax_double<<<gridDim,blockDim>>>(dot_res_1,dim,dot_res_2);
      else
        kernel_absmax_double<<<gridDim,blockDim>>>(dot_res_2,dim,dot_res_1);
      isRes1or2 = (isRes1or2+1)%2;
    }
    double res;
    if (isRes1or2 == 1){
      HANDLE_ERROR(cudaMemcpy(&res,dot_res_1,sizeof(double),cudaMemcpyDeviceToHost));
    } else {
      HANDLE_ERROR(cudaMemcpy(&res,dot_res_2,sizeof(double),cudaMemcpyDeviceToHost));
    }
    return (cudapcgVar_t)res;
}
//------------------------------------------------------------------------------
double reduce(cudapcgVar_t *v, unsigned int dim){
    kernel_reduce<<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim,dot_res_1);
    return reduce_dot_res_1(dim);
}
//------------------------------------------------------------------------------
double absreduce(cudapcgVar_t *v, unsigned int dim){
    kernel_absreduce<<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim,dot_res_1);
    return reduce_dot_res_1(dim);
}
//------------------------------------------------------------------------------
double dotprod(cudapcgVar_t *v1, cudapcgVar_t *v2, unsigned int dim){
    kernel_dotprod<<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v1,v2,dim,dot_res_1);
    return reduce_dot_res_1(dim);
}
//------------------------------------------------------------------------------
void termbytermmul(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res){
    kernel_termbytermmul<<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v1,v2,dim,res);
    return;
}
//------------------------------------------------------------------------------
void termbytermdiv(cudapcgVar_t * v1, cudapcgVar_t * v2, unsigned int dim, cudapcgVar_t * res){
    kernel_termbytermdiv<<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v1,v2,dim,res);
    return;
}
//------------------------------------------------------------------------------
void termbyterminv(cudapcgVar_t * v, unsigned int dim){
    kernel_termbyterminv<<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(v,dim);
    return;
}
//------------------------------------------------------------------------------
void saxpy(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim, cudapcgVar_t * res){
    kernel_saxpy<<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(y,x,a,dim,res);
    return;
}
//------------------------------------------------------------------------------
void saxpy_iny(cudapcgVar_t * y, cudapcgVar_t * x, cudapcgVar_t a, unsigned int dim){
    kernel_saxpy_iny<<<CEIL(dim,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(y,x,a,dim);
    return;
}
//------------------------------------------------------------------------------
void interpl2(cudapcgVar_t *res, cudapcgVar_t *v, unsigned int dim_x, unsigned int dim_y, unsigned int dim_z, unsigned int stride){
    kernel_zeros<<<CEIL(stride*dim_x*dim_y*(dim_z+(dim_z<1)),THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,stride*dim_x*dim_y*(dim_z+(dim_z<1)));
    kernel_project2<<<CEIL((dim_x/2)*(dim_y/2)*((dim_z/2)+((dim_z/2)<1)),THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,v,dim_x,dim_y,dim_z,stride);
    kernel_interpl_rows<<<CEIL(dim_x*dim_y*(dim_z+(dim_z<1)),THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,dim_x,dim_y,dim_z,stride);
    kernel_interpl_cols<<<CEIL(dim_x*dim_y*(dim_z+(dim_z<1)),THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,dim_x,dim_y,dim_z,stride);
    if (dim_z)
      kernel_interpl_layers<<<CEIL(dim_x*dim_y*dim_z,THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(res,dim_x,dim_y,dim_z,stride);
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
