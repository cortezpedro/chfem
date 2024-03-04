#include "image.h"
#include "utils.h"

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
template <typename T>
__global__ void kernel_project2(T *res, T *v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs){
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
template __global__ void kernel_project2<float>(float *res, float *v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);
template __global__ void kernel_project2<double>(double *res, double *v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);
//------------------------------------------------------------------------------
// Kernels that interpolate entries of vector v from coarse mesh to refined mesh
// It is assumed that coarse mesh dimensions are [nx/2,ny/2,nz/2]
template <typename T>
__global__ void kernel_interpl_rows(T * v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs){
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
template __global__ void kernel_interpl_rows<float>(float *v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);
template __global__ void kernel_interpl_rows<double>(double *v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);
//------------------------------------------------------------------------------
// Kernels that interpolate entries of vector v from coarse mesh to refined mesh
// It is assumed that coarse mesh dimensions are [nx/2,ny/2,nz/2]
template <typename T>
__global__ void kernel_interpl_cols(T * v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs){
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
template __global__ void kernel_interpl_cols<float>(float *v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);
template __global__ void kernel_interpl_cols<double>(double *v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);
//------------------------------------------------------------------------------
// Kernels that interpolate entries of vector v from coarse mesh to refined mesh
// It is assumed that coarse mesh dimensions are [nx/2,ny/2,nz/2]
template <typename T>
__global__ void kernel_interpl_layers(T * v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs){
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
template __global__ void kernel_interpl_layers<float>(float *v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);
template __global__ void kernel_interpl_layers<double>(double *v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);
//------------------------------------------------------------------------------
