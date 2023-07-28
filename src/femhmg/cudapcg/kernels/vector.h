#include "../includes.h"

#ifndef CUDAPCG_KERNELS_VECOPS_H_INCLUDED
#define CUDAPCG_KERNELS_VECOPS_H_INCLUDED

// ATTENTION: all kernels in this file are templates, meant to be reusable for single and double precision.
//            i.e. "typename T" can either be float or double

// Kernels:
//
// zeros --------------------------------------------- v = 0
// scale --------------------------------------------- v *= scl
// arrcpy -------------------------------------------- res = v
// termbytermmul ------------------------------------- res = v1 .* v2
// termbytermdiv ------------------------------------- res = v1 ./ v2
// termbyterminv ------------------------------------- v = 1 ./ v
// axpy ---------------------------------------------- res = a*x + y
// axpy_iny ------------------------------------------ y = a*x + y
// axpy_iny_with_stride ------------------------------ y = a*x[shift:stride:end] + y
// reduce -------------------------------------------- res = sum(v)
// absreduce ----------------------------------------- res = sum(abs(v))
// reduce_with_stride -------------------------------- res = sum(v[shift:stride:end])
// reduce_with_stride_and_scale ---------------------- res = sum(scl*v[shift:stride:end])
// reduce_positive_values_with_stride ---------------- res = sum(v[v>0])
// reduce_negative_values_with_stride ---------------- res = sum(v[v<0])
// reduce_positive_values_with_stride_and_scale ------ res = sum(scl*v[v>0])
// reduce_negative_values_with_stride_and_scale ------ res = sum(scl*v[v<0])
// dotprod ------------------------------------------- res = dot(v1,v2)
// max ----------------------------------------------- res = max(v)
// absmax -------------------------------------------- res = max(abs(v))
// absmax_signed ------------------------------------- res = (+ or -) max(abs(v))

template<typename T> __global__ void kernel_zeros(T * v, unsigned int n);                                                 // v = 0
template<typename T> __global__ void kernel_scale(T * v, T scl, unsigned int n);                                          // v *= scl
template<typename T> __global__ void kernel_arrcpy(T *v, unsigned int n, T *res);                                         // res = v
template<typename T> __global__ void kernel_termbytermmul(T * v1, T * v2, unsigned int n, T * res);                       // res = v1 .* v2
template<typename T> __global__ void kernel_termbytermdiv(T * v1, T * v2, unsigned int n, T * res);                       // res = v1 ./ v2
template<typename T> __global__ void kernel_termbyterminv(T * v, unsigned int n);                                         // v = 1 ./ v
template<typename T> __global__ void kernel_axpy(T * y, T * x, T a, unsigned int n, T * res);                             // res = a*x + y
template<typename T> __global__ void kernel_axpy_iny(T * y, T * x, T a, unsigned int n);                                  // y = a*x + y
template<typename T> __global__ void kernel_axpy_iny_with_stride(T * y, T * x, T a, unsigned int n, unsigned int stride, unsigned int shift); // y = a*x[shift:stride:end] + y
template<typename T> __global__ void kernel_reduce(T *v, unsigned int n, double *res);                                    // res = sum(v)
template<typename T> __global__ void kernel_absreduce(T *v, unsigned int n, double *res);                                 // res = sum(abs(v))
template<typename T> __global__ void kernel_reduce_with_stride(T *v, unsigned int n, unsigned int stride, unsigned int shift, double *res);   // res = sum(v[shift:stride:end])
template<typename T> __global__ void kernel_reduce_with_stride_and_scale(T *v, unsigned int n, unsigned int stride, unsigned int shift, T scl, double *res);   // res = sum(scl*v[shift:stride:end])
template<typename T> __global__ void kernel_reduce_positive_values_with_stride(T *v, unsigned int n, unsigned int stride, unsigned int shift, double *res);    // res = sum(v[v>0])
template<typename T> __global__ void kernel_reduce_negative_values_with_stride(T *v, unsigned int n, unsigned int stride, unsigned int shift, double *res);    // res = sum(v[v<0])
template<typename T> __global__ void kernel_reduce_positive_values_with_stride_and_scale(T *v, unsigned int n, unsigned int stride, unsigned int shift, T scl, double *res);   // res = sum(scl*v[v>0])
template<typename T> __global__ void kernel_reduce_negative_values_with_stride_and_scale(T *v, unsigned int n, unsigned int stride, unsigned int shift, T scl, double *res);   // res = sum(scl*v[v<0])
template<typename T> __global__ void kernel_dotprod(T *v1, T *v2, unsigned int n, double *res);                           // res = dot(v1,v2)
template<typename T> __global__ void kernel_max(T *v, unsigned int n, double *res);                                       // res = max(v)
template<typename T> __global__ void kernel_absmax(T *v, unsigned int n, double *res);                                    // res = max(abs(v))
template<typename T> __global__ void kernel_absmax_signed(T *v, unsigned int n, double *res);                             // res = (+ or -) max(abs(v))

#endif// CUDAPCG_KERNELS_VECOPS_H_INCLUDED
