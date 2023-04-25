#include "../includes.h"

#ifndef CUDAPCG_KERNELS_VECOPS_H_INCLUDED
#define CUDAPCG_KERNELS_VECOPS_H_INCLUDED

// ATTENTION: all kernels in this file are templates, meant to be reusable for single and double precision.
//            i.e. "typename T" can either be float or double

template<typename T> __global__ void kernel_zeros(T * v, unsigned int n);                            // v = 0
template<typename T> __global__ void kernel_arrcpy(T *v, unsigned int n, T *res);                    // res = v
template<typename T> __global__ void kernel_termbytermmul(T * v1, T * v2, unsigned int n, T * res);  // res = v1 .* v2
template<typename T> __global__ void kernel_termbytermdiv(T * v1, T * v2, unsigned int n, T * res);  // res = v1 ./ v2
template<typename T> __global__ void kernel_termbyterminv(T * v, unsigned int n);                    // v = 1 ./ v
template<typename T> __global__ void kernel_saxpy(T * y, T * x, T a, unsigned int n, T * res);       // res = a*x + y
template<typename T> __global__ void kernel_saxpy_iny(T * y, T * x, T a, unsigned int n);            // y = a*x + y
template<typename T> __global__ void kernel_reduce(T *v, unsigned int n, double *res);               // res = sum(v)
template<typename T> __global__ void kernel_absreduce(T *v, unsigned int n, double *res);            // res = sum(abs(v))
template<typename T> __global__ void kernel_dotprod(T *v1, T *v2, unsigned int n, double *res);      // res = dot(v1,v2)
template<typename T> __global__ void kernel_max(T *v, unsigned int n, double *res);                  // res = max(v)
template<typename T> __global__ void kernel_absmax(T *v, unsigned int n, double *res);               // res = max(abs(v))

#endif// CUDAPCG_KERNELS_VECOPS_H_INCLUDED
