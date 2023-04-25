#include "../includes.h"

#ifndef CUDAPCG_KERNELS_IMGOPS_H_INCLUDED
#define CUDAPCG_KERNELS_IMGOPS_H_INCLUDED

// ATTENTION: all kernels in this file are templates, meant to be reusable for single and double precision.
//            i.e. "typename T" can either be float or double

template <typename T> __global__ void kernel_project2(T *res, T *v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);
template <typename T> __global__ void kernel_interpl_rows(T * v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);
template <typename T> __global__ void kernel_interpl_cols(T * v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);
template <typename T> __global__ void kernel_interpl_layers(T * v, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nLocalDOFs);

#endif// CUDAPCG_KERNELS_IMGOPS_H_INCLUDED
