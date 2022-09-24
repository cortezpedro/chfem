/*
    Universidade Federal Fluminense (UFF) - Niteroi, Brazil
    Institute of Computing
    Author: Cortez, P.
    History: v1.0 (november/2020)

    API with wrapper CPU functions to call CUDA kernels
    used by cudapcg.h
    
    CUDA kernels are implemented here.

    History:
        Initially developed as a final work for the graduate course "Arquitetura
        e Programacao de GPUs", at the Institute of Computing, UFF.

    ATTENTION.1:
        Considers a structured regular mesh of quad elements (2D)
        or hexahedron elements (3D).

    ATTENTION.2:
        As it is, this API is not generic, for any Ax = b. Linear systems must
        be associated to FEM homogenization problems, for Potential or
        Elasticity, both 2D or 3D.
*/

#include "cudapcg_kernels_source.h"
#include "utils.h"
#include "../error_handling.h"

#if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT
  __constant__ cudapcgVar_t K[2304];// max size of K is 576*4 (4 materials on elastic_3D)
#endif

void setConstantLocalK(cudapcgVar_t * lclK, unsigned long int size){
  #if !defined CUDAPCG_MATKEY_32BIT && !defined CUDAPCG_MATKEY_64BIT
    HANDLE_ERROR(cudaMemcpyToSymbol(K,lclK,size));
  #else
    printf("WARNING: Storing local matrices in constant memory cache is not supported for material keys with 32bits or 64bits.\n");
  #endif
  return;
}

//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
///////////////////////// CUDA KERNELS ///////////////////////////////
//////////////////////////////////////////////////////////////////////
//--------------------------------------------------------------------

//-----------------------------------------
///////////////////////////////////////////
/////////////// THERMAL 2D ////////////////
///////////////////////////////////////////
//-----------------------------------------

//---------------------------------
///////////////////////////////////
////////// NODE-BY-NODE ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 2D thermal analysis
__global__ void kernel_assemblePreConditioner_thermal_2D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    // Local var to store result
    cudapcgVar_t res;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_2D))*16;
    res =  K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16+5;
    res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16+10;
    res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16+15;
    res += K[mat];
    M[i] = 1.0/res;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D thermal conductivity analysis
__global__ void kernel_applyPreConditioner_thermal_2D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    // Local var to store result
    cudapcgVar_t precond;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_2D))*16;
    precond =  K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16+5;
    precond += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16+10;
    precond += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16+15;
    precond += K[mat];
    res[i] = v1[i]/precond + scl*v2[i];
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D thermal analysis
__global__ void kernel_Aprod_thermal_2D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    // Local var to store result
    cudapcgVar_t res;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_2D))*16;
    // Compute total number of dofs
    unsigned int ndofs = nx*ny;
    // Local var to store neighbor dof indexes
    unsigned int dof;
    // Iteration on first neighbor elem
    res  = K[mat]  *  v[i]; 
    dof = WALK_RIGHT(i,ny,ndofs);
    res += K[mat+1]*v[dof];
    dof = WALK_UP(dof,ny);
    res += K[mat+2]*v[dof];
    dof = WALK_UP(i,ny);
    res += K[mat+3]*v[dof];
    // Iteration on second neighbor elem
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16;
    dof = WALK_LEFT(i,nx,ny,ndofs);
    res += K[mat+4]*v[dof];
    res += K[mat+5]*  v[i];
    dof = WALK_UP(i,ny);
    res += K[mat+6]*v[dof];
    dof = WALK_LEFT(dof,nx,ny,ndofs);
    res += K[mat+7]*v[dof];
    // Iteration on third neighbor elem
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16;
    dof = WALK_DOWN(i,ny); dof = WALK_LEFT(dof,nx,ny,ndofs);
    res += K[mat+8] *v[dof];
    dof = WALK_RIGHT(dof,ny,ndofs);
    res += K[mat+9] *v[dof];
    res += K[mat+10]*  v[i];
    dof = WALK_LEFT(i,nx,ny,ndofs);
    res += K[mat+11]*v[dof];
    // Iteration on fourth neighbor elem
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*16;
    dof = WALK_DOWN(i,ny);
    res += K[mat+12]*v[dof];
    dof = WALK_RIGHT(dof,ny,ndofs);
    res += K[mat+13]*v[dof];
    dof = WALK_RIGHT(i,ny,ndofs);
    res += K[mat+14]*v[dof];
    res += K[mat+15]*  v[i];
    // Put final result on global array
    q[i] *= isIncrement;
    q[i] += scl*res;
  }
}
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 2D thermal analysis
__global__ void kernel_assemblePreConditioner_thermal_2D_ElemByElem(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int ny){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t contrib = K[(unsigned int)material[i]*16+n*5];
    unsigned int dof = i;
    if (n==2) dof = WALK_RIGHT(dof,ny,dim); // make use of dim (nelem==ndofs)
    else if (n==1) {dof = WALK_RIGHT(dof,ny,dim); dof = WALK_DOWN(dof,ny); }
    else if (n==0) dof = WALK_DOWN(dof,ny);
    M[dof] += contrib;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D thermal analysis
__global__ void kernel_Aprod_thermal_2D_ElemByElem(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int ny, cudapcgVar_t *q){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int row = (unsigned int) material[i]*16 + n*4;
    unsigned int dof;
    unsigned int id = 0;
    cudapcgVar_t contrib = 0.0;
    // 0
    dof = WALK_DOWN(i,ny);
    contrib += K[row]*v[dof];
    id += dof*(n==0);
    // 1
    dof = WALK_RIGHT(dof,ny,dim); // make use of dim (nelem==ndofs)
    contrib += K[row+1]*v[dof];
    id += dof*(n==1);
    // 2
    dof = WALK_RIGHT(i,ny,dim); 
    contrib += K[row+2]*v[dof];
    id += dof*(n==2);
    // 3
    contrib += K[row+3]*v[i];
    id += i*(n==3);

    q[id] += contrib;
  }
}
//------------------------------------------------------------------------------

//-----------------------------------------
///////////////////////////////////////////
/////////////// THERMAL 3D ////////////////
///////////////////////////////////////////
//-----------------------------------------

//---------------------------------
///////////////////////////////////
////////// NODE-BY-NODE ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 3D thermal analysis
__global__ void kernel_assemblePreConditioner_thermal_3D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    /*
       ATTENTION: Analytical solution is cosidered, zeros are not stored.

      k_el = material_prop * {
        {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
        {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
        { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
        {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
        {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
        { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
        { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
        { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
      }
    */
    // Local var to store result
    cudapcgVar_t res;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*40;
    res =  K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+5;
    res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+11;
    res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+16;
    res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+23;
    res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+28;
    res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+34;
    res += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+39;
    res += K[mat];
    M[i] = 1.0/res;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 3D thermal conductivity analysis
__global__ void kernel_applyPreConditioner_thermal_3D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    /*
       ATTENTION: Analytical solution is cosidered, zeros are not stored.

      k_el = material_prop * {
        {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
        {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
        { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
        {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
        {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
        { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
        { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
        { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
      }
    */
    // Local var to store result
    cudapcgVar_t precond;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*40;
    precond =  K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+5;
    precond += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+11;
    precond += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+16;
    precond += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+23;
    precond += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+28;
    precond += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+34;
    precond += K[mat]; mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40+39;
    precond += K[mat];
    res[i] = v1[i]/precond + scl*v2[i];
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    /*
      ATTENTION: Analytical solution is cosidered, zeros are not stored.

      k_el = material_prop * {
        {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
        {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
        { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
        {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
        {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
        { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
        { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
        { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
      }
    */
    // Local var to store result
    cudapcgVar_t res;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*40;
    // Compute number of dofs per layer
    unsigned int ndofs_xy = nx*ny;
    // Compute total number of dofs
    unsigned int ndofs = ndofs_xy*nz;
    // Find which layer thread i is on
    unsigned int layer = (i/ndofs_xy)*ndofs_xy;
    // Find local index, within layer
    unsigned int li = i%ndofs_xy;
    // Local var to store neighbor dof indexes
    unsigned int dof;
    // Iteration on first neighbor elem
    res  = K[mat]  *        v[i]; 
    dof = WALK_RIGHT(li,ny,ndofs_xy); dof = WALK_UP(dof,ny);
    res += K[mat+1]*v[dof+layer];
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    res += K[mat+2]*v[dof+layer];
    dof = WALK_UP(dof,ny);
    res += K[mat+3]*v[dof+layer];
    dof = WALK_UP(li,ny);
    res += K[mat+4]*v[dof+layer];
    // Iteration on second neighbor elem
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40;
    res += K[mat+5]*        v[i];
    dof = WALK_UP(li,ny); dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    res += K[mat+6]*v[dof+layer];
    layer = WALK_FAR(layer,ndofs_xy,ndofs); 
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    res += K[mat+7]*v[dof+layer]; 
    dof = WALK_UP(li,ny);
    res += K[mat+8]*v[dof+layer];
    dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    res += K[mat+9]*v[dof+layer];
    // Iteration on third neighbor elem
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40;
    dof = WALK_DOWN(li,ny); dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    res += K[mat+10]*v[dof+layer];
    res += K[mat+11]*        v[i];
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    dof = WALK_DOWN(li,ny); dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    res += K[mat+12]*v[dof+layer]; 
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    res += K[mat+13]*v[dof+layer];
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    res += K[mat+14]*v[dof+layer];
    // Iteration on fourth neighbor elem
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40;
    dof = WALK_DOWN(li,ny); dof = WALK_RIGHT(dof,ny,ndofs_xy);
    res += K[mat+15]*v[dof+layer];
    res += K[mat+16]*        v[i];
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    dof = WALK_DOWN(li,ny);
    res += K[mat+17]*v[dof+layer];
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    res += K[mat+18]*v[dof+layer];
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    res += K[mat+19]*v[dof+layer];
    // Iteration on fifth neighbor element
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs); // this is correct, we need to go back a full layer
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40;
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    res += K[mat+20]*v[dof+layer];
    dof = WALK_UP(dof,ny);
    res += K[mat+21]*v[dof+layer];
    dof = WALK_UP(li,ny);
    res += K[mat+22]*v[dof+layer];
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    res += K[mat+23]*        v[i];
    dof = WALK_UP(li,ny); dof = WALK_RIGHT(dof,ny,ndofs_xy);
    res += K[mat+24]*v[dof+layer];
    // Iteration on sixth neighbor element
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40;
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    res += K[mat+25]*v[dof+layer];
    dof = WALK_UP(li,ny);
    res += K[mat+26]*v[dof+layer];
    dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    res += K[mat+27]*v[dof+layer];
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    res += K[mat+28]*        v[i];
    dof = WALK_UP(li,ny); dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    res += K[mat+29]*v[dof+layer];
    // Iteration on seventh neighbor element
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40;
    dof = WALK_DOWN(li,ny); dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    res += K[mat+30]*v[dof+layer];
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    res += K[mat+31]*v[dof+layer];
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    res += K[mat+32]*v[dof+layer];
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    dof = WALK_DOWN(li,ny); dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    res += K[mat+33]*v[dof+layer];
    res += K[mat+34]*        v[i];
    // Iteration on eighth neighbor element
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*40;
    dof = WALK_DOWN(li,ny);
    res += K[mat+35]*v[dof+layer];
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    res += K[mat+36]*v[dof+layer];
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    res += K[mat+37]*v[dof+layer];
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    dof = WALK_DOWN(li,ny); dof = WALK_RIGHT(dof,ny,ndofs_xy);
    res += K[mat+38]*v[dof+layer];
    res += K[mat+39]*        v[i];
    // Put final result on global array
    q[i] *= isIncrement;
    q[i] += scl*res;
  }
}
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 3D thermal analysis
__global__ void kernel_assemblePreConditioner_thermal_3D_ElemByElem(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny){
  /*
    ATTENTION: Analytical solution is cosidered, zeros are not stored.

    k_el = material_prop * {
      {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
      {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
      { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
      {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
      {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
      { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
      { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
      { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
    }
  */
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    cudapcgVar_t contrib = K[(unsigned int)material[i]*40]; // using first diag entry, as it is the same for all (simple, but not generic)
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = i%ndofs_xy;
    unsigned int layer = ((i/ndofs_xy + (n>3))*ndofs_xy)%dim;
    if (n%4==2)dof = WALK_RIGHT(dof,ny,ndofs_xy);
    else if (n%4==1){ dof = WALK_RIGHT(dof,ny,ndofs_xy); dof = WALK_DOWN(dof,ny); }
    else if (n%4==0) dof = WALK_DOWN(dof,ny);
    M[dof+layer] += contrib;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
// ATTENTION: NOT BEING USED ANYMORE!
__global__ void kernel_Aprod_thermal_3D_ElemByElem(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q){
  /*
    ATTENTION: Analytical solution is cosidered, zeros are not stored.

    k_el = material_prop * {
      {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
      {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
      { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
      {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
      {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
      { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
      { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
      { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
    }
  */
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int row = (unsigned int)material[i]*40 + n*5;
    unsigned int col = 0;
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = i%ndofs_xy;
    unsigned int layer = (i/ndofs_xy)*ndofs_xy;
    unsigned int id=0;
    cudapcgVar_t contrib = 0.0;
    // 0
    dof = WALK_DOWN(dof,ny);
    if (n!=1 && n!=3 && n!=4){
      contrib += K[row+col]*v[dof+layer];
      col++;
      if (n==0) id = dof+layer;
    }
    // 1
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    if (n!=0 && n!=2 && n!=5){
      contrib += K[row+col]*v[dof+layer];
      col++;
      if (n==1) id = dof+layer;
    }
    // 2
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    if (n!=3 && n!=1 && n!=6){
      contrib += K[row+col]*v[dof+layer];
      col++;
      if (n==2) id = dof+layer;
    }
    // 3
    dof = i%ndofs_xy;
    if (n!=2 && n!=0 && n!=7){
      contrib += K[row+col]*v[dof+layer];
      col++;
      if (n==3) id = dof+layer;
    }
    // 4
    layer = WALK_FAR(layer,ndofs_xy,dim);
    dof = WALK_DOWN(dof,ny);
    if (n!=5 && n!=7 && n!=0){
      contrib += K[row+col]*v[dof+layer];
      col++;
      if (n==4) id = dof+layer;
    }
    // 5
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    if (n!=4 && n!=6 && n!=1){
      contrib += K[row+col]*v[dof+layer];
      col++;
      if (n==5) id = dof+layer;
    }
    // 6
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    if (n!=7 && n!=5 && n!=2){
      contrib += K[row+col]*v[dof+layer];
      col++;
      if (n==6) id = dof+layer;
    }
    // 7
    dof = i%ndofs_xy;
    if (n!=6 && n!=4 && n!=3){
      contrib += K[row+col]*v[dof+layer];
      //col++;
      if (n==7) id = dof+layer;
    }

    q[id] += contrib;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n0(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q){
  /*
    ATTENTION: Analytical solution is cosidered, zeros are not stored.

    k_el = material_prop * {
      {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
      {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
      { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
      {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
      {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
      { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
      { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
      { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
    }
  */
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int mat = (unsigned int)material[i]*40;
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = i%ndofs_xy;
    unsigned int layer = (i/ndofs_xy)*ndofs_xy;
    unsigned int id;
    cudapcgVar_t contrib = 0.0;
    // 0
    dof = WALK_DOWN(dof,ny); id = dof+layer;
    contrib += K[mat]*v[dof+layer];
    // 1 - no contrib
    dof = WALK_RIGHT(dof,ny,ndofs_xy);  
    // 2
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    contrib += K[mat+1]*v[dof+layer];
    // 3 - no contrib
    dof = i%ndofs_xy;
    // 4 - no contrib
    layer = WALK_FAR(layer,ndofs_xy,dim);
    dof = WALK_DOWN(dof,ny);
    // 5
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    contrib += K[mat+2]*v[dof+layer];
    // 6
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    contrib += K[mat+3]*v[dof+layer];
    // 7
    dof = i%ndofs_xy;
    contrib += K[mat+4]*v[dof+layer];

    q[id] = contrib;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n1(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q){
  /*
    ATTENTION: Analytical solution is cosidered, zeros are not stored.

    k_el = material_prop * {
      {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
      {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
      { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
      {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
      {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
      { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
      { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
      { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
    }
  */
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int mat = (unsigned int)material[i]*40+5;
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = i%ndofs_xy;
    unsigned int layer = (i/ndofs_xy)*ndofs_xy;
    unsigned int id;
    cudapcgVar_t contrib = 0.0;
    // 0 - no contrib
    dof = WALK_DOWN(dof,ny);
    // 1
    dof = WALK_RIGHT(dof,ny,ndofs_xy); id = dof+layer;
    contrib += K[mat]*v[dof+layer];
    // 2 - no contrib
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    // 3
    dof = i%ndofs_xy;
    contrib += K[mat+1]*v[dof+layer];
    // 4
    layer = WALK_FAR(layer,ndofs_xy,dim);
    dof = WALK_DOWN(dof,ny);
    contrib += K[mat+2]*v[dof+layer];
    // 5 - no contrib
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    // 6
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    contrib += K[mat+3]*v[dof+layer];
    // 7
    dof = i%ndofs_xy;
    contrib += K[mat+4]*v[dof+layer];

    q[id] += contrib;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n2(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q){
  /*
    ATTENTION: Analytical solution is cosidered, zeros are not stored.

    k_el = material_prop * {
      {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
      {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
      { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
      {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
      {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
      { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
      { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
      { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
    }
  */
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int mat = (unsigned int)material[i]*40+10;
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = i%ndofs_xy;
    unsigned int layer = (i/ndofs_xy)*ndofs_xy;
    unsigned int id;
    cudapcgVar_t contrib = 0.0;
    // 0
    dof = WALK_DOWN(dof,ny);
    contrib += K[mat]*v[dof+layer];
    // 1 - no contrib
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    // 2
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy); id = dof+layer;
    contrib += K[mat+1]*v[dof+layer];
    // 3 - no contrib
    dof = i%ndofs_xy;
    // 4
    layer = WALK_FAR(layer,ndofs_xy,dim);
    dof = WALK_DOWN(dof,ny);
    contrib += K[mat+2]*v[dof+layer];
    // 5
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    contrib += K[mat+3]*v[dof+layer];
    // 6 - no contrib
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    // 7
    dof = i%ndofs_xy;
    contrib += K[mat+4]*v[dof+layer];

    q[id] += contrib;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n3(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q){
  /*
    ATTENTION: Analytical solution is cosidered, zeros are not stored.

    k_el = material_prop * {
      {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
      {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
      { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
      {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
      {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
      { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
      { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
      { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
    }
  */
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int mat = (unsigned int)material[i]*40+15;
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = i%ndofs_xy;
    unsigned int layer = (i/ndofs_xy)*ndofs_xy;
    unsigned int id;
    cudapcgVar_t contrib = 0.0;
    // 0 - no contrib
    dof = WALK_DOWN(dof,ny);
    // 1
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    contrib += K[mat]*v[dof+layer];
    // 2 - no contrib
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    // 3
    dof = i%ndofs_xy; id = dof+layer;
    contrib += K[mat+1]*v[dof+layer];
    // 4
    layer = WALK_FAR(layer,ndofs_xy,dim);
    dof = WALK_DOWN(dof,ny);
    contrib += K[mat+2]*v[dof+layer];
    // 5
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    contrib += K[mat+3]*v[dof+layer];
    // 6
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    contrib += K[mat+4]*v[dof+layer];
    // 7 - no contrib
    // dof = i%ndofs_xy;

    q[id] += contrib;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n4(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q){
  /*
    ATTENTION: Analytical solution is cosidered, zeros are not stored.

    k_el = material_prop * {
      {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
      {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
      { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
      {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
      {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
      { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
      { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
      { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
    }
  */
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int mat = (unsigned int)material[i]*40+20;
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = i%ndofs_xy;
    unsigned int layer = (i/ndofs_xy)*ndofs_xy;
    unsigned int id;
    cudapcgVar_t contrib = 0.0;
    // 0 - no contrib
    dof = WALK_DOWN(dof,ny);
    // 1
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    contrib += K[mat]*v[dof+layer];
    // 2
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    contrib += K[mat+1]*v[dof+layer];
    // 3
    dof = i%ndofs_xy;
    contrib += K[mat+2]*v[dof+layer];
    // 4
    layer = WALK_FAR(layer,ndofs_xy,dim);
    dof = WALK_DOWN(dof,ny); id = dof+layer;
    contrib += K[mat+3]*v[dof+layer];
    // 5 - no contrib
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    // 6
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    contrib += K[mat+4]*v[dof+layer];
    // 7 - no contrib
    // dof = i%ndofs_xy;

    q[id] += contrib;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n5(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q){
  /*
    ATTENTION: Analytical solution is cosidered, zeros are not stored.

    k_el = material_prop * {
      {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
      {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
      { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
      {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
      {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
      { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
      { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
      { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
    }
  */
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int mat = (unsigned int)material[i]*40+25;
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = i%ndofs_xy;
    unsigned int layer = (i/ndofs_xy)*ndofs_xy;
    unsigned int id;
    cudapcgVar_t contrib = 0.0;
    // 0
    dof = WALK_DOWN(dof,ny);
    contrib += K[mat]*v[dof+layer];
    // 1 - no contrib
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    // 2
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    contrib += K[mat+1]*v[dof+layer];
    // 3
    dof = i%ndofs_xy;
    contrib += K[mat+2]*v[dof+layer];
    // 4 - no contrib
    layer = WALK_FAR(layer,ndofs_xy,dim);
    dof = WALK_DOWN(dof,ny);
    // 5
    dof = WALK_RIGHT(dof,ny,ndofs_xy); id = dof+layer;
    contrib += K[mat+3]*v[dof+layer];
    // 6 - no contrib
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    // 7
    dof = i%ndofs_xy;
    contrib += K[mat+4]*v[dof+layer];

    q[id] += contrib;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n6(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q){
  /*
    ATTENTION: Analytical solution is cosidered, zeros are not stored.

    k_el = material_prop * {
      {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
      {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
      { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
      {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
      {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
      { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
      { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
      { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
    }
  */
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int mat = (unsigned int)material[i]*40+30;
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = i%ndofs_xy;
    unsigned int layer = (i/ndofs_xy)*ndofs_xy;
    unsigned int id;
    cudapcgVar_t contrib = 0.0;
    // 0
    dof = WALK_DOWN(dof,ny);
    contrib += K[mat]*v[dof+layer];
    // 1
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    contrib += K[mat+1]*v[dof+layer];
    // 2 - no contrib
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    // 3
    dof = i%ndofs_xy;
    contrib += K[mat+2]*v[dof+layer];
    // 4
    layer = WALK_FAR(layer,ndofs_xy,dim);
    dof = WALK_DOWN(dof,ny);
    contrib += K[mat+3]*v[dof+layer];
    // 5 - no contrib
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    // 6
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy); id = dof+layer;
    contrib += K[mat+4]*v[dof+layer];
    // 7 - no contrib
    //dof = i%ndofs_xy;

    q[id] += contrib;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D thermal analysis
__global__ void kernel_Aprod_thermal_3D_ElemByElem_n7(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q){
  /*
    ATTENTION: Analytical solution is cosidered, zeros are not stored.

    k_el = material_prop * {
      {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
      {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
      { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
      {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
      {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
      { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
      { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
      { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
    }
  */
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim){
    unsigned int mat = (unsigned int)material[i]*40+35;
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = i%ndofs_xy;
    unsigned int layer = (i/ndofs_xy)*ndofs_xy;
    unsigned int id;
    cudapcgVar_t contrib = 0.0;
    // 0
    dof = WALK_DOWN(dof,ny);
    contrib += K[mat]*v[dof+layer];
    // 1
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    contrib += K[mat+1]*v[dof+layer];
    // 2
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    contrib += K[mat+2]*v[dof+layer];
    // 3 - no contrib
    dof = i%ndofs_xy;
    // 4 - no contrib
    layer = WALK_FAR(layer,ndofs_xy,dim);
    dof = WALK_DOWN(dof,ny);
    // 5
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    contrib += K[mat+3]*v[dof+layer];
    // 6 - no contrib
    dof = WALK_RIGHT((i%ndofs_xy),ny,ndofs_xy);
    // 7
    dof = i%ndofs_xy; id = dof+layer;
    contrib += K[mat+4]*v[dof+layer];

    q[id] += contrib;
  }
}
//------------------------------------------------------------------------------

//-----------------------------------------
///////////////////////////////////////////
/////////////// ELASTIC 2D ////////////////
///////////////////////////////////////////
//-----------------------------------------

//---------------------------------
///////////////////////////////////
////////// NODE-BY-NODE ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 2D elasticity analysis
__global__ void kernel_assemblePreConditioner_elastic_2D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/2){ // 2DOFs per node
    // Local var to store result
    cudapcgVar_t res_x, res_y;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_2D))*64;
    res_x =  K[mat];
    res_y =  K[mat+9]; 
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64;
    res_x += K[mat+18];
    res_y += K[mat+27];
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64;
    res_x += K[mat+36];
    res_y += K[mat+54];
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64;
    res_x += K[mat+45];
    res_y += K[mat+63];
    M[2*i]   = 1.0/res_x;
    M[2*i+1] = 1.0/res_y;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D elasticity analysis
__global__ void kernel_applyPreConditioner_elastic_2D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/2){ // 2DOFs per node
    // Local var to store result
    cudapcgVar_t precond_x, precond_y;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_2D))*64;
    precond_x =  K[mat];
    precond_y =  K[mat+9]; 
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64;
    precond_x += K[mat+18];
    precond_y += K[mat+27];
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64;
    precond_x += K[mat+36];
    precond_y += K[mat+54];
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64;
    precond_x += K[mat+45];
    precond_y += K[mat+63];
    
    res[2*i]   = v1[2*i]  /precond_x + scl*v2[2*i];
    res[2*i+1] = v1[2*i+1]/precond_y + scl*v2[2*i+1];
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D elasticity analysis
__global__ void kernel_Aprod_elastic_2D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/2){ // 2 DOFs per node
    // Compute total number of dofs
    unsigned int ndofs = nx*ny;
    // Local var to store neighbor dof indexes
    unsigned int dof;
    // Local vars to store entries of v to be used on computations
    cudapcgVar_t v_x, v_y;
    // Local vars to store result
    cudapcgVar_t res_x, res_y;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_2D))*64;
    // Iteration on first neighbor elem
    v_x = v[2*i]; v_y = v[2*i+1];
    res_x  = K[mat]   *v_x + K[mat+1] *v_y;
    res_y  = K[mat+8] *v_x + K[mat+9] *v_y;
    dof = WALK_RIGHT(i,ny,ndofs);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+2] *v_x + K[mat+3] *v_y;
    res_y += K[mat+10]*v_x + K[mat+11]*v_y;
    dof = WALK_UP(dof,ny);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+4] *v_x + K[mat+5] *v_y;
    res_y += K[mat+12]*v_x + K[mat+13]*v_y;
    dof = WALK_UP(i,ny);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+6] *v_x + K[mat+7] *v_y;
    res_y += K[mat+14]*v_x + K[mat+15]*v_y;
    // Iteration on second neighbor elem
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64;
    dof = WALK_LEFT(i,nx,ny,ndofs);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+16]*v_x + K[mat+17]*v_y;
    res_y += K[mat+24]*v_x + K[mat+25]*v_y;
    v_x = v[2*i]; v_y = v[2*i+1];
    res_x += K[mat+18]*v_x + K[mat+19]*v_y;
    res_y += K[mat+26]*v_x + K[mat+27]*v_y;
    dof = WALK_UP(i,ny);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+20]*v_x + K[mat+21]*v_y;
    res_y += K[mat+28]*v_x + K[mat+29]*v_y;
    dof = WALK_LEFT(dof,nx,ny,ndofs);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+22]*v_x + K[mat+23]*v_y;
    res_y += K[mat+30]*v_x + K[mat+31]*v_y;
    // Iteration on third neighbor elem
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64;
    dof = WALK_DOWN(i,ny); dof = WALK_LEFT(dof,nx,ny,ndofs);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+32]*v_x + K[mat+33]*v_y;
    res_y += K[mat+40]*v_x + K[mat+41]*v_y;
    dof = WALK_RIGHT(dof,ny,ndofs);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+34]*v_x + K[mat+35]*v_y;
    res_y += K[mat+42]*v_x + K[mat+43]*v_y;
    v_x = v[2*i]; v_y = v[2*i+1];
    res_x += K[mat+36]*v_x + K[mat+37]*v_y;
    res_y += K[mat+44]*v_x + K[mat+45]*v_y;
    dof = WALK_LEFT(i,nx,ny,ndofs);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+38]*v_x + K[mat+39]*v_y;
    res_y += K[mat+46]*v_x + K[mat+47]*v_y;
    // Iteration on fourth neighbor elem
    mat = ((matkey >>= MATKEY_BITSTEP_2D) & MATKEY_BITSTEP_RANGE_2D)*64;
    dof = WALK_DOWN(i,ny);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+48]*v_x + K[mat+49]*v_y;
    res_y += K[mat+56]*v_x + K[mat+57]*v_y;
    dof = WALK_RIGHT(dof,ny,ndofs);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+50]*v_x + K[mat+51]*v_y;
    res_y += K[mat+58]*v_x + K[mat+59]*v_y;
    dof = WALK_RIGHT(i,ny,ndofs);
    v_x = v[2*dof]; v_y = v[2*dof+1];
    res_x += K[mat+52]*v_x + K[mat+53]*v_y;
    res_y += K[mat+60]*v_x + K[mat+61]*v_y;
    v_x = v[2*i]; v_y = v[2*i+1];
    res_x += K[mat+54]*v_x + K[mat+55]*v_y;
    res_y += K[mat+62]*v_x + K[mat+63]*v_y;
    // Put final results on global array
    q[2*i]   *= isIncrement;
    q[2*i]   += scl*res_x;
    q[2*i+1] *= isIncrement;
    q[2*i+1] += scl*res_y;
  }
}
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 2D elasticity analysis
__global__ void kernel_assemblePreConditioner_elastic_2D_ElemByElem(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int ny){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<2*dim){ // 2 DOFs per node
    cudapcgVar_t contrib = K[(unsigned int)material[i/2]*64+n*18+(i%2)*9];
    unsigned int dof = i/2; // 2 DOFs per node
    if (n==2) dof = WALK_RIGHT(dof,ny,dim); // make use of dim (nelem==ndofs)
    else if (n==1) {dof = WALK_RIGHT(dof,ny,dim); dof = WALK_DOWN(dof,ny); }
    else if (n==0) dof = WALK_DOWN(dof,ny);
    M[2*dof+(i%2)] += contrib;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 2D elasticity analysis
__global__ void kernel_Aprod_elastic_2D_ElemByElem(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int ny, cudapcgVar_t *q){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<2*dim){ // 2 DOFs per node
    unsigned int e = i/2;
    unsigned int row = (unsigned int) material[e]*64 + n*16 + (i%2)*8;
    unsigned int dof;
    unsigned int id = 0;
    cudapcgVar_t contrib = 0.0;
    // 0
    dof = WALK_DOWN(e,ny);
    contrib += K[row]*v[2*dof] + K[row+1]*v[2*dof+1];
    id += (2*dof+(i%2))*(n==0);
    // 1
    dof = WALK_RIGHT(dof,ny,dim); // make use of dim (nelem==ndofs)
    contrib += K[row+2]*v[2*dof] + K[row+3]*v[2*dof+1];
    id += (2*dof+(i%2))*(n==1);
    // 2
    dof = WALK_RIGHT(e,ny,dim); 
    contrib += K[row+4]*v[2*dof] + K[row+5]*v[2*dof+1];
    id += (2*dof+(i%2))*(n==2);
    // 3
    contrib += K[row+6]*v[2*e] + K[row+7]*v[2*e+1];
    id += i*(n==3);
    q[id] += contrib;
  }
}
//------------------------------------------------------------------------------

//-----------------------------------------
///////////////////////////////////////////
/////////////// ELASTIC 3D ////////////////
///////////////////////////////////////////
//-----------------------------------------

//---------------------------------
///////////////////////////////////
////////// NODE-BY-NODE ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 3D elasticity analysis
__global__ void kernel_assemblePreConditioner_elastic_3D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/3){ // 3DOFs per node
    // Local var to store result
    cudapcgVar_t res_x, res_y, res_z;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*576;
    res_x =  K[mat];
    res_y =  K[mat+25];
    res_z =  K[mat+50];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    res_x += K[mat+75];
    res_y += K[mat+100];
    res_z += K[mat+125];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    res_x += K[mat+150];
    res_y += K[mat+175];
    res_z += K[mat+200];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    res_x += K[mat+225];
    res_y += K[mat+250];
    res_z += K[mat+275];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    res_x += K[mat+300];
    res_y += K[mat+325];
    res_z += K[mat+350];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    res_x += K[mat+375];
    res_y += K[mat+400];
    res_z += K[mat+425];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    res_x += K[mat+450];
    res_y += K[mat+475];
    res_z += K[mat+500];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    res_x += K[mat+525];
    res_y += K[mat+550];
    res_z += K[mat+575];
    M[3*i]   = 1.0/res_x;
    M[3*i+1] = 1.0/res_y;
    M[3*i+2] = 1.0/res_z;
  }
}
//------------------------------------------------------------------------------
// Kernel to perform assembly-free preconditioner multiplication for 2D elasticity analysis
__global__ void kernel_applyPreConditioner_elastic_3D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v1, cudapcgVar_t * v2, cudapcgVar_t scl, unsigned int dim, cudapcgMap_t *material, cudapcgVar_t * res){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/3){ // 3DOFs per node
    // Local var to store result
    cudapcgVar_t precond_x, precond_y, precond_z;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*576;
    precond_x =  K[mat];
    precond_y =  K[mat+25];
    precond_z =  K[mat+50];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    precond_x += K[mat+75];
    precond_y += K[mat+100];
    precond_z += K[mat+125];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    precond_x += K[mat+150];
    precond_y += K[mat+175];
    precond_z += K[mat+200];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    precond_x += K[mat+225];
    precond_y += K[mat+250];
    precond_z += K[mat+275];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    precond_x += K[mat+300];
    precond_y += K[mat+325];
    precond_z += K[mat+350];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    precond_x += K[mat+375];
    precond_y += K[mat+400];
    precond_z += K[mat+425];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    precond_x += K[mat+450];
    precond_y += K[mat+475];
    precond_z += K[mat+500];
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    precond_x += K[mat+525];
    precond_y += K[mat+550];
    precond_z += K[mat+575];
    res[3*i]   = v1[3*i]  /precond_x + scl*v2[3*i];
    res[3*i+1] = v1[3*i+1]/precond_y + scl*v2[3*i+1];
    res[3*i+2] = v1[3*i+2]/precond_z + scl*v2[3*i+2];
  }
}
//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D elasticity analysis
__global__ void kernel_Aprod_elastic_3D_NodeByNode(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, unsigned int nz, cudapcgVar_t *q, cudapcgVar_t scl, cudapcgFlag_t isIncrement){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<dim/3){ // 3 DOFs per node
    // Compute number of dofs per layer
    unsigned int ndofs_xy = nx*ny;
    // Compute total number of dofs
    unsigned int ndofs = ndofs_xy*nz;
    // Find which layer thread i is on
    unsigned int layer = (i/ndofs_xy)*ndofs_xy;
    // Find local index, within layer
    unsigned int li = i%ndofs_xy;
    // Local var to store neighbor dof indexes
    unsigned int dof;
    // Local vars to store entries of v to be used on computations
    cudapcgVar_t v_x, v_y, v_z;
    // Local vars to store result
    cudapcgVar_t res_x, res_y, res_z;
    // Material map value for this dof
    cudapcgMap_t matkey = material[i];
    unsigned int mat = ((unsigned int) (matkey & MATKEY_BITSTEP_RANGE_3D))*576;
    // Iteration on first neighbor elem
    v_x = v[3*i]; v_y = v[3*i+1]; v_z = v[3*i+2];
    res_x = K[mat]   *v_x + K[mat+1] *v_y + K[mat+2] *v_z;
    res_y = K[mat+24]*v_x + K[mat+25]*v_y + K[mat+26]*v_z;
    res_z = K[mat+48]*v_x + K[mat+49]*v_y + K[mat+50]*v_z;
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+3] *v_x + K[mat+4] *v_y + K[mat+5] *v_z;
    res_y += K[mat+27]*v_x + K[mat+28]*v_y + K[mat+29]*v_z;
    res_z += K[mat+51]*v_x + K[mat+52]*v_y + K[mat+53]*v_z;
    dof = WALK_UP(dof,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+6] *v_x + K[mat+7] *v_y + K[mat+8] *v_z;
    res_y += K[mat+30]*v_x + K[mat+31]*v_y + K[mat+32]*v_z;
    res_z += K[mat+54]*v_x + K[mat+55]*v_y + K[mat+56]*v_z;
    dof = WALK_UP(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+9] *v_x + K[mat+10]*v_y + K[mat+11]*v_z;
    res_y += K[mat+33]*v_x + K[mat+34]*v_y + K[mat+35]*v_z;
    res_z += K[mat+57]*v_x + K[mat+58]*v_y + K[mat+59]*v_z;
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    v_x = v[3*(li+layer)]; v_y = v[3*(li+layer)+1]; v_z = v[3*(li+layer)+2];
    res_x += K[mat+12]*v_x + K[mat+13]*v_y + K[mat+14]*v_z;
    res_y += K[mat+36]*v_x + K[mat+37]*v_y + K[mat+38]*v_z;
    res_z += K[mat+60]*v_x + K[mat+61]*v_y + K[mat+62]*v_z;
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+15]*v_x + K[mat+16]*v_y + K[mat+17]*v_z;
    res_y += K[mat+39]*v_x + K[mat+40]*v_y + K[mat+41]*v_z;
    res_z += K[mat+63]*v_x + K[mat+64]*v_y + K[mat+65]*v_z;
    dof = WALK_UP(dof,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+18]*v_x + K[mat+19]*v_y + K[mat+20]*v_z;
    res_y += K[mat+42]*v_x + K[mat+43]*v_y + K[mat+44]*v_z;
    res_z += K[mat+66]*v_x + K[mat+67]*v_y + K[mat+68]*v_z;
    dof = WALK_UP(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+21]*v_x + K[mat+22]*v_y + K[mat+23]*v_z;
    res_y += K[mat+45]*v_x + K[mat+46]*v_y + K[mat+47]*v_z;
    res_z += K[mat+69]*v_x + K[mat+70]*v_y + K[mat+71]*v_z;
    // Iteration on second neighbor elem
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+72] *v_x + K[mat+73] *v_y + K[mat+74] *v_z;
    res_y += K[mat+96] *v_x + K[mat+97] *v_y + K[mat+98] *v_z;
    res_z += K[mat+120]*v_x + K[mat+121]*v_y + K[mat+122]*v_z;
    v_x = v[3*i]; v_y = v[3*i+1]; v_z = v[3*i+2];
    res_x += K[mat+75] *v_x + K[mat+76] *v_y + K[mat+77] *v_z;
    res_y += K[mat+99] *v_x + K[mat+100]*v_y + K[mat+101]*v_z;
    res_z += K[mat+123]*v_x + K[mat+124]*v_y + K[mat+125]*v_z;
    dof = WALK_UP(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+78] *v_x + K[mat+79] *v_y + K[mat+80] *v_z;
    res_y += K[mat+102]*v_x + K[mat+103]*v_y + K[mat+104]*v_z;
    res_z += K[mat+126]*v_x + K[mat+127]*v_y + K[mat+128]*v_z;
    dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+81] *v_x + K[mat+82] *v_y + K[mat+83] *v_z;
    res_y += K[mat+105]*v_x + K[mat+106]*v_y + K[mat+107]*v_z;
    res_z += K[mat+129]*v_x + K[mat+130]*v_y + K[mat+131]*v_z;
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+84] *v_x + K[mat+85] *v_y + K[mat+86] *v_z;
    res_y += K[mat+108]*v_x + K[mat+109]*v_y + K[mat+110]*v_z;
    res_z += K[mat+132]*v_x + K[mat+133]*v_y + K[mat+134]*v_z;
    v_x = v[3*(li+layer)]; v_y = v[3*(li+layer)+1]; v_z = v[3*(li+layer)+2];
    res_x += K[mat+87] *v_x + K[mat+88] *v_y + K[mat+89] *v_z;
    res_y += K[mat+111]*v_x + K[mat+112]*v_y + K[mat+113]*v_z;
    res_z += K[mat+135]*v_x + K[mat+136]*v_y + K[mat+137]*v_z;
    dof = WALK_UP(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+90] *v_x + K[mat+91] *v_y + K[mat+92] *v_z;
    res_y += K[mat+114]*v_x + K[mat+115]*v_y + K[mat+116]*v_z;
    res_z += K[mat+138]*v_x + K[mat+139]*v_y + K[mat+140]*v_z;
    dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+93] *v_x + K[mat+94] *v_y + K[mat+95] *v_z;
    res_y += K[mat+117]*v_x + K[mat+118]*v_y + K[mat+119]*v_z;
    res_z += K[mat+141]*v_x + K[mat+142]*v_y + K[mat+143]*v_z;
    // Iteration on third neighbor elem
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    dof = WALK_DOWN(li,ny); dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+144]*v_x + K[mat+145]*v_y + K[mat+146]*v_z;
    res_y += K[mat+168]*v_x + K[mat+169]*v_y + K[mat+170]*v_z;
    res_z += K[mat+192]*v_x + K[mat+193]*v_y + K[mat+194]*v_z;
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+147]*v_x + K[mat+148]*v_y + K[mat+149]*v_z;
    res_y += K[mat+171]*v_x + K[mat+172]*v_y + K[mat+173]*v_z;
    res_z += K[mat+195]*v_x + K[mat+196]*v_y + K[mat+197]*v_z;
    v_x = v[3*i]; v_y = v[3*i+1]; v_z = v[3*i+2];
    res_x += K[mat+150]*v_x + K[mat+151]*v_y + K[mat+152]*v_z;
    res_y += K[mat+174]*v_x + K[mat+175]*v_y + K[mat+176]*v_z;
    res_z += K[mat+198]*v_x + K[mat+199]*v_y + K[mat+200]*v_z;
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+153]*v_x + K[mat+154]*v_y + K[mat+155]*v_z;
    res_y += K[mat+177]*v_x + K[mat+178]*v_y + K[mat+179]*v_z;
    res_z += K[mat+201]*v_x + K[mat+202]*v_y + K[mat+203]*v_z;
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    dof = WALK_DOWN(li,ny); dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+156]*v_x + K[mat+157]*v_y + K[mat+158]*v_z;
    res_y += K[mat+180]*v_x + K[mat+181]*v_y + K[mat+182]*v_z;
    res_z += K[mat+204]*v_x + K[mat+205]*v_y + K[mat+206]*v_z;
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+159]*v_x + K[mat+160]*v_y + K[mat+161]*v_z;
    res_y += K[mat+183]*v_x + K[mat+184]*v_y + K[mat+185]*v_z;
    res_z += K[mat+207]*v_x + K[mat+208]*v_y + K[mat+209]*v_z;
    v_x = v[3*(li+layer)]; v_y = v[3*(li+layer)+1]; v_z = v[3*(li+layer)+2];
    res_x += K[mat+162]*v_x + K[mat+163]*v_y + K[mat+164]*v_z;
    res_y += K[mat+186]*v_x + K[mat+187]*v_y + K[mat+188]*v_z;
    res_z += K[mat+210]*v_x + K[mat+211]*v_y + K[mat+212]*v_z;
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+165]*v_x + K[mat+166]*v_y + K[mat+167]*v_z;
    res_y += K[mat+189]*v_x + K[mat+190]*v_y + K[mat+191]*v_z;
    res_z += K[mat+213]*v_x + K[mat+214]*v_y + K[mat+215]*v_z;
    // Iteration on fourth neighbor elem
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    dof = WALK_DOWN(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+216]*v_x + K[mat+217]*v_y + K[mat+218]*v_z;
    res_y += K[mat+240]*v_x + K[mat+241]*v_y + K[mat+242]*v_z;
    res_z += K[mat+264]*v_x + K[mat+265]*v_y + K[mat+266]*v_z;
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+219]*v_x + K[mat+220]*v_y + K[mat+221]*v_z;
    res_y += K[mat+243]*v_x + K[mat+244]*v_y + K[mat+245]*v_z;
    res_z += K[mat+267]*v_x + K[mat+268]*v_y + K[mat+269]*v_z;
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+222]*v_x + K[mat+223]*v_y + K[mat+224]*v_z;
    res_y += K[mat+246]*v_x + K[mat+247]*v_y + K[mat+248]*v_z;
    res_z += K[mat+270]*v_x + K[mat+271]*v_y + K[mat+272]*v_z;
    v_x = v[3*i]; v_y = v[3*i+1]; v_z = v[3*i+2];
    res_x += K[mat+225]*v_x + K[mat+226]*v_y + K[mat+227]*v_z;
    res_y += K[mat+249]*v_x + K[mat+250]*v_y + K[mat+251]*v_z;
    res_z += K[mat+273]*v_x + K[mat+274]*v_y + K[mat+275]*v_z;
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    dof = WALK_DOWN(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+228]*v_x + K[mat+229]*v_y + K[mat+230]*v_z;
    res_y += K[mat+252]*v_x + K[mat+253]*v_y + K[mat+254]*v_z;
    res_z += K[mat+276]*v_x + K[mat+277]*v_y + K[mat+278]*v_z;
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+231]*v_x + K[mat+232]*v_y + K[mat+233]*v_z;
    res_y += K[mat+255]*v_x + K[mat+256]*v_y + K[mat+257]*v_z;
    res_z += K[mat+279]*v_x + K[mat+280]*v_y + K[mat+281]*v_z;
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+234]*v_x + K[mat+235]*v_y + K[mat+236]*v_z;
    res_y += K[mat+258]*v_x + K[mat+259]*v_y + K[mat+260]*v_z;
    res_z += K[mat+282]*v_x + K[mat+283]*v_y + K[mat+284]*v_z;
    v_x = v[3*(li+layer)]; v_y = v[3*(li+layer)+1]; v_z = v[3*(li+layer)+2];
    res_x += K[mat+237]*v_x + K[mat+238]*v_y + K[mat+239]*v_z;
    res_y += K[mat+261]*v_x + K[mat+262]*v_y + K[mat+263]*v_z;
    res_z += K[mat+285]*v_x + K[mat+286]*v_y + K[mat+287]*v_z;
    // Iteration on fifth neighbor elem
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs); // this is correct, we need to go back a full layer
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    v_x = v[3*(li+layer)]; v_y = v[3*(li+layer)+1]; v_z = v[3*(li+layer)+2];
    res_x += K[mat+288]*v_x + K[mat+289]*v_y + K[mat+290]*v_z;
    res_y += K[mat+312]*v_x + K[mat+313]*v_y + K[mat+314]*v_z;
    res_z += K[mat+336]*v_x + K[mat+337]*v_y + K[mat+338]*v_z;
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+291]*v_x + K[mat+292]*v_y + K[mat+293]*v_z;
    res_y += K[mat+315]*v_x + K[mat+316]*v_y + K[mat+317]*v_z;
    res_z += K[mat+339]*v_x + K[mat+340]*v_y + K[mat+341]*v_z;
    dof = WALK_UP(dof,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+294]*v_x + K[mat+295]*v_y + K[mat+296]*v_z;
    res_y += K[mat+318]*v_x + K[mat+319]*v_y + K[mat+320]*v_z;
    res_z += K[mat+342]*v_x + K[mat+343]*v_y + K[mat+344]*v_z;
    dof = WALK_UP(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+297]*v_x + K[mat+298]*v_y + K[mat+299]*v_z;
    res_y += K[mat+321]*v_x + K[mat+322]*v_y + K[mat+323]*v_z;
    res_z += K[mat+345]*v_x + K[mat+346]*v_y + K[mat+347]*v_z;
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    v_x = v[3*i]; v_y = v[3*i+1]; v_z = v[3*i+2];
    res_x += K[mat+300]*v_x + K[mat+301]*v_y + K[mat+302]*v_z;
    res_y += K[mat+324]*v_x + K[mat+325]*v_y + K[mat+326]*v_z;
    res_z += K[mat+348]*v_x + K[mat+349]*v_y + K[mat+350]*v_z;
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+303]*v_x + K[mat+304]*v_y + K[mat+305]*v_z;
    res_y += K[mat+327]*v_x + K[mat+328]*v_y + K[mat+329]*v_z;
    res_z += K[mat+351]*v_x + K[mat+352]*v_y + K[mat+353]*v_z;
    dof = WALK_UP(dof,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+306]*v_x + K[mat+307]*v_y + K[mat+308]*v_z;
    res_y += K[mat+330]*v_x + K[mat+331]*v_y + K[mat+332]*v_z;
    res_z += K[mat+354]*v_x + K[mat+355]*v_y + K[mat+356]*v_z;
    dof = WALK_UP(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+309]*v_x + K[mat+310]*v_y + K[mat+311]*v_z;
    res_y += K[mat+333]*v_x + K[mat+334]*v_y + K[mat+335]*v_z;
    res_z += K[mat+357]*v_x + K[mat+358]*v_y + K[mat+359]*v_z;
    // Iteration on sixth neighbor elem
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+360]*v_x + K[mat+361]*v_y + K[mat+362]*v_z;
    res_y += K[mat+384]*v_x + K[mat+385]*v_y + K[mat+386]*v_z;
    res_z += K[mat+408]*v_x + K[mat+409]*v_y + K[mat+410]*v_z;
    v_x = v[3*(li+layer)]; v_y = v[3*(li+layer)+1]; v_z = v[3*(li+layer)+2];
    res_x += K[mat+363]*v_x + K[mat+364]*v_y + K[mat+365]*v_z;
    res_y += K[mat+387]*v_x + K[mat+388]*v_y + K[mat+389]*v_z;
    res_z += K[mat+411]*v_x + K[mat+412]*v_y + K[mat+413]*v_z;
    dof = WALK_UP(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+366]*v_x + K[mat+367]*v_y + K[mat+368]*v_z;
    res_y += K[mat+390]*v_x + K[mat+391]*v_y + K[mat+392]*v_z;
    res_z += K[mat+414]*v_x + K[mat+415]*v_y + K[mat+416]*v_z;
    dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+369]*v_x + K[mat+370]*v_y + K[mat+371]*v_z;
    res_y += K[mat+393]*v_x + K[mat+394]*v_y + K[mat+395]*v_z;
    res_z += K[mat+417]*v_x + K[mat+418]*v_y + K[mat+419]*v_z;
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+372]*v_x + K[mat+373]*v_y + K[mat+374]*v_z;
    res_y += K[mat+396]*v_x + K[mat+397]*v_y + K[mat+398]*v_z;
    res_z += K[mat+420]*v_x + K[mat+421]*v_y + K[mat+422]*v_z;
    v_x = v[3*i]; v_y = v[3*i+1]; v_z = v[3*i+2];
    res_x += K[mat+375]*v_x + K[mat+376]*v_y + K[mat+377]*v_z;
    res_y += K[mat+399]*v_x + K[mat+400]*v_y + K[mat+401]*v_z;
    res_z += K[mat+423]*v_x + K[mat+424]*v_y + K[mat+425]*v_z;
    dof = WALK_UP(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+378]*v_x + K[mat+379]*v_y + K[mat+380]*v_z;
    res_y += K[mat+402]*v_x + K[mat+403]*v_y + K[mat+404]*v_z;
    res_z += K[mat+426]*v_x + K[mat+427]*v_y + K[mat+428]*v_z;
    dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+381]*v_x + K[mat+382]*v_y + K[mat+383]*v_z;
    res_y += K[mat+405]*v_x + K[mat+406]*v_y + K[mat+407]*v_z;
    res_z += K[mat+429]*v_x + K[mat+430]*v_y + K[mat+431]*v_z;
    // Iteration on seventh neighbor elem
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    dof = WALK_DOWN(li,ny); dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+432]*v_x + K[mat+433]*v_y + K[mat+434]*v_z;
    res_y += K[mat+456]*v_x + K[mat+457]*v_y + K[mat+458]*v_z;
    res_z += K[mat+480]*v_x + K[mat+481]*v_y + K[mat+482]*v_z;
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+435]*v_x + K[mat+436]*v_y + K[mat+437]*v_z;
    res_y += K[mat+459]*v_x + K[mat+460]*v_y + K[mat+461]*v_z;
    res_z += K[mat+483]*v_x + K[mat+484]*v_y + K[mat+485]*v_z;
    v_x = v[3*(li+layer)]; v_y = v[3*(li+layer)+1]; v_z = v[3*(li+layer)+2];
    res_x += K[mat+438]*v_x + K[mat+439]*v_y + K[mat+440]*v_z;
    res_y += K[mat+462]*v_x + K[mat+463]*v_y + K[mat+464]*v_z;
    res_z += K[mat+486]*v_x + K[mat+487]*v_y + K[mat+488]*v_z;
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+441]*v_x + K[mat+442]*v_y + K[mat+443]*v_z;
    res_y += K[mat+465]*v_x + K[mat+466]*v_y + K[mat+467]*v_z;
    res_z += K[mat+489]*v_x + K[mat+490]*v_y + K[mat+491]*v_z;
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    dof = WALK_DOWN(li,ny); dof = WALK_LEFT(dof,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+444]*v_x + K[mat+445]*v_y + K[mat+446]*v_z;
    res_y += K[mat+468]*v_x + K[mat+469]*v_y + K[mat+470]*v_z;
    res_z += K[mat+492]*v_x + K[mat+493]*v_y + K[mat+494]*v_z;
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+447]*v_x + K[mat+448]*v_y + K[mat+449]*v_z;
    res_y += K[mat+471]*v_x + K[mat+472]*v_y + K[mat+473]*v_z;
    res_z += K[mat+495]*v_x + K[mat+496]*v_y + K[mat+497]*v_z;
    v_x = v[3*i]; v_y = v[3*i+1]; v_z = v[3*i+2];
    res_x += K[mat+450]*v_x + K[mat+451]*v_y + K[mat+452]*v_z;
    res_y += K[mat+474]*v_x + K[mat+475]*v_y + K[mat+476]*v_z;
    res_z += K[mat+498]*v_x + K[mat+499]*v_y + K[mat+500]*v_z;
    dof = WALK_LEFT(li,nx,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+453]*v_x + K[mat+454]*v_y + K[mat+455]*v_z;
    res_y += K[mat+477]*v_x + K[mat+478]*v_y + K[mat+479]*v_z;
    res_z += K[mat+501]*v_x + K[mat+502]*v_y + K[mat+503]*v_z;
    // Iteration on eighth neighbor elem
    layer = WALK_NEAR(layer,ndofs_xy,nz,ndofs);
    mat = ((matkey >>= MATKEY_BITSTEP_3D) & MATKEY_BITSTEP_RANGE_3D)*576;
    dof = WALK_DOWN(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+504]*v_x + K[mat+505]*v_y + K[mat+506]*v_z;
    res_y += K[mat+528]*v_x + K[mat+529]*v_y + K[mat+530]*v_z;
    res_z += K[mat+552]*v_x + K[mat+553]*v_y + K[mat+554]*v_z;
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+507]*v_x + K[mat+508]*v_y + K[mat+509]*v_z;
    res_y += K[mat+531]*v_x + K[mat+532]*v_y + K[mat+533]*v_z;
    res_z += K[mat+555]*v_x + K[mat+556]*v_y + K[mat+557]*v_z;
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+510]*v_x + K[mat+511]*v_y + K[mat+512]*v_z;
    res_y += K[mat+534]*v_x + K[mat+535]*v_y + K[mat+536]*v_z;
    res_z += K[mat+558]*v_x + K[mat+559]*v_y + K[mat+560]*v_z;
    v_x = v[3*(li+layer)]; v_y = v[3*(li+layer)+1]; v_z = v[3*(li+layer)+2];
    res_x += K[mat+513]*v_x + K[mat+514]*v_y + K[mat+515]*v_z;
    res_y += K[mat+537]*v_x + K[mat+538]*v_y + K[mat+539]*v_z;
    res_z += K[mat+561]*v_x + K[mat+562]*v_y + K[mat+563]*v_z;
    layer = WALK_FAR(layer,ndofs_xy,ndofs);
    dof = WALK_DOWN(li,ny);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+516]*v_x + K[mat+517]*v_y + K[mat+518]*v_z;
    res_y += K[mat+540]*v_x + K[mat+541]*v_y + K[mat+542]*v_z;
    res_z += K[mat+564]*v_x + K[mat+565]*v_y + K[mat+566]*v_z;
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+519]*v_x + K[mat+520]*v_y + K[mat+521]*v_z;
    res_y += K[mat+543]*v_x + K[mat+544]*v_y + K[mat+545]*v_z;
    res_z += K[mat+567]*v_x + K[mat+568]*v_y + K[mat+569]*v_z;
    dof = WALK_RIGHT(li,ny,ndofs_xy);
    v_x = v[3*(dof+layer)]; v_y = v[3*(dof+layer)+1]; v_z = v[3*(dof+layer)+2];
    res_x += K[mat+522]*v_x + K[mat+523]*v_y + K[mat+524]*v_z;
    res_y += K[mat+546]*v_x + K[mat+547]*v_y + K[mat+548]*v_z;
    res_z += K[mat+570]*v_x + K[mat+571]*v_y + K[mat+572]*v_z;
    v_x = v[3*i]; v_y = v[3*i+1]; v_z = v[3*i+2];
    res_x += K[mat+525]*v_x + K[mat+526]*v_y + K[mat+527]*v_z;
    res_y += K[mat+549]*v_x + K[mat+550]*v_y + K[mat+551]*v_z;
    res_z += K[mat+573]*v_x + K[mat+574]*v_y + K[mat+575]*v_z;
    // Put final results on global array
    q[3*i]   *= isIncrement;
    q[3*i]   += scl*res_x;
    q[3*i+1] *= isIncrement;
    q[3*i+1] += scl*res_y;
    q[3*i+2] *= isIncrement;
    q[3*i+2] += scl*res_z;
  }
}
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
////////// ELEM-BY-ELEM ///////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
// Kernel to perform assembly of jacobi preconditioner for 3D elasticity analysis
__global__ void kernel_assemblePreConditioner_elastic_3D_ElemByElem(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		unsigned int n, cudapcgVar_t * M, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<3*dim){ // 3 DOFs per node
    cudapcgVar_t contrib = K[(unsigned int)material[i/3]*576+n*75+(i%3)*25];
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = (i/3)%ndofs_xy; // 3 DOFs per node
    unsigned int layer = (((i/3)/ndofs_xy + (n>3))*ndofs_xy)%dim;
    if (n%4==2) dof = WALK_RIGHT(dof,ny,ndofs_xy); // make use of dim (nelem==ndofs)
    else if (n%4==1) {dof = WALK_RIGHT(dof,ny,ndofs_xy); dof = WALK_DOWN(dof,ny); }
    else if (n%4==0) dof = WALK_DOWN(dof,ny);
    M[3*(dof+layer)+(i%3)] += contrib;
  }
}

//------------------------------------------------------------------------------
// Kernel to perform "assembly on-the-fly" matrix-vector product, for 3D elasticity analysis
__global__ void kernel_Aprod_elastic_3D_ElemByElem(
		#if defined CUDAPCG_MATKEY_32BIT || defined CUDAPCG_MATKEY_64BIT
			cudapcgVar_t *K,
		#endif
		unsigned int n, cudapcgVar_t * v, unsigned int dim, cudapcgMap_t *material, unsigned int nx, unsigned int ny, cudapcgVar_t *q){
  // Get global thread index
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Check if this thread must work
  if (i<3*dim){ // 3 DOFs per node
    unsigned int e = i/3;
    unsigned int row = (unsigned int)material[e]*576 + n*72 + (i%3)*24;
    unsigned int ndofs_xy = nx*ny;
    unsigned int dof = e%ndofs_xy;
    unsigned int layer = (e/ndofs_xy)*ndofs_xy;
    unsigned int id=0;
    cudapcgVar_t contrib = 0.0;
    // 0
    dof = WALK_DOWN(dof,ny);
    contrib += K[row]*v[3*(dof+layer)] + K[row+1]*v[3*(dof+layer)+1] + K[row+2]*v[3*(dof+layer)+2];
    id += (3*(dof+layer)+(i%3))*(n==0);
    // 1
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    contrib += K[row+3]*v[3*(dof+layer)] + K[row+4]*v[3*(dof+layer)+1] + K[row+5]*v[3*(dof+layer)+2];
    id += (3*(dof+layer)+(i%3))*(n==1);
    // 2
    dof = WALK_RIGHT((e%ndofs_xy),ny,ndofs_xy);
    contrib += K[row+6]*v[3*(dof+layer)] + K[row+7]*v[3*(dof+layer)+1] + K[row+8]*v[3*(dof+layer)+2];
    id += (3*(dof+layer)+(i%3))*(n==2);
    // 3
    dof = e%ndofs_xy;
    contrib += K[row+9]*v[3*(dof+layer)] + K[row+10]*v[3*(dof+layer)+1] + K[row+11]*v[3*(dof+layer)+2];
    id += (3*(dof+layer)+(i%3))*(n==3);
    // 4
    layer = WALK_FAR(layer,ndofs_xy,dim);
    dof = WALK_DOWN(dof,ny);
    contrib += K[row+12]*v[3*(dof+layer)] + K[row+13]*v[3*(dof+layer)+1] + K[row+14]*v[3*(dof+layer)+2];
    id += (3*(dof+layer)+(i%3))*(n==4);
    // 5
    dof = WALK_RIGHT(dof,ny,ndofs_xy);
    contrib += K[row+15]*v[3*(dof+layer)] + K[row+16]*v[3*(dof+layer)+1] + K[row+17]*v[3*(dof+layer)+2];
    id += (3*(dof+layer)+(i%3))*(n==5);
    // 6
    dof = WALK_RIGHT((e%ndofs_xy),ny,ndofs_xy);
    contrib += K[row+18]*v[3*(dof+layer)] + K[row+19]*v[3*(dof+layer)+1] + K[row+20]*v[3*(dof+layer)+2];
    id += (3*(dof+layer)+(i%3))*(n==6);
    // 7
    dof = e%ndofs_xy;
    contrib += K[row+21]*v[3*(dof+layer)] + K[row+22]*v[3*(dof+layer)+1] + K[row+23]*v[3*(dof+layer)+2];
    id += (3*(dof+layer)+(i%3))*(n==7);

    q[id] += contrib;
  }
}
//------------------------------------------------------------------------------
