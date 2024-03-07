#include <cuda.h>

#ifndef CUDA_ERROR_HANDLING_H_INCLUDED
#define CUDA_ERROR_HANDLING_H_INCLUDED

#define HANDLE_ERROR(status) { cudaEvalStatus((status),__FILE__,__LINE__); }

static inline void cudaEvalStatus(cudaError_t status, const char * filename, unsigned int line){
    if (status != cudaSuccess){
        printf("CUDA error in %s (line %i): %s\n",filename,line,cudaGetErrorString(status));
        cudaDeviceReset();
        exit(0);
    }
}

#endif //CUDA_ERROR_HANDLING_H_INCLUDED
