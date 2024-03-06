/*

*/

#ifndef INCLUDES_CUDAPCG_SOLVERS_UTILS_H_INCLUDED
#define INCLUDES_CUDAPCG_SOLVERS_UTILS_H_INCLUDED

#ifdef CUDAPCG_VAR_64BIT
  #define ABS(x) (abs_double((x)))
  #define SQRT(x) (sqrt_double((x)))
#else // default is 32bit float
  #define ABS(x) (abs_float((x)))
  #define SQRT(x) (sqrt_float((x)))
#endif

//------------------------------------------------------------------------------
float abs_float(float x);
float sqrt_float(float x);
double abs_double(double x);
double sqrt_double(double x);
//------------------------------------------------------------------------------


#endif // INCLUDES_CUDAPCG_SOLVERS_UTILS_H_INCLUDED
