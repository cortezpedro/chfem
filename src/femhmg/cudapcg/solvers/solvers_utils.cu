/*
*/

#include "solvers_utils.h"

//------------------------------------------------------------------------------
float abs_float(float x){
  long i = (*(long *) &x) & ~(1UL<<31); // convert x from float to long without losing bits and set the 32nd bit (sign) as 0 (positive)
  return (*(float *) &i);               // convert back to float and return
}
//------------------------------------------------------------------------------
// Function that uses bitwise operations on floats (considering IEEE 754 format)
// to find approximate values for sqrt(x), where x>=0.
// This is used to find a residual estimate, without the need to include math.h
float sqrt_float(float x){
  // Get bits of 1.0f as a long
  const float onef = 1.0f;
  const long onefAsInt = *(long*) &onef;
  
  // Compute approximation for result
  long i = *(long*) &x;                          // Read bits of x as long
  i = ((i>>1) + (onefAsInt>>1)) & ~(1UL<<31);    // Bitwise operations to find an approximate value for sqrt(x)
  float y = *(float*) &i;                        // Read approximation as a float
  
  // 3 Newton-Raphson iterations to improve approximation
  y = 0.5*(y+x/y);
  y = 0.5*(y+x/y);
  return 0.5*(y+x/y);
}
//------------------------------------------------------------------------------
double abs_double(double x){
  long long i = (*(long long *) &x) & ~(1ULL<<63); // convert x from double to long long without losing bits and set the 64th bit (sign) as 0 (positive)
  return (*(double *) &i);                         // convert back to double and return
}
//------------------------------------------------------------------------------
// Function that uses bitwise operations on doubles (considering IEEE 754 format)
// to find approximate values for sqrt(x), where x>=0.
// This is used to find a residual estimate, without the need to include math.h
double sqrt_double(double x){
  // Get bits of 1.0f as a long long
  const double onef = 1.0f;
  const long long onefAsInt = *(long long*) &onef;
  
  // Compute approximation for result
  long long i = *(long long*) &x;                // Read bits of x as long long
  i = ((i>>1) + (onefAsInt>>1)) & ~(1ULL<<63);   // Bitwise operations to find an approximate value for sqrt(x)
  double y = *(double*) &i;                      // Read approximation as a double
  
  // 3 Newton-Raphson iterations to improve approximation
  y = 0.5*(y+x/y);
  y = 0.5*(y+x/y);
  return 0.5*(y+x/y);
}
//------------------------------------------------------------------------------
