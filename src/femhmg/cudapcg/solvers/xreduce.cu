#include "../kernels/wrappers.h"

//--------------------------------------------------------------------------------------------------
void update_xreduce(cudapcgModel_t *m, unsigned int it, cudapcgFlag_t flag, unsigned int shift, double stab, double a, cudapcgVar_t *d, cudapcgVar_t *x, double **res_per_it){//, double **norms){
  double scl_factor;
  switch (flag){
    case CUDAPCG_XREDUCE_NONE: // x is ignored
      #ifdef NO_XREDUCE_STAB
      scl_factor = 1.0;
      for (int ii=0; ii<m->nvarspernode; ii++) res_per_it[ii][it] = (a * scl_factor) * reduce_with_stride(d,m->nhmgvars,m->nvarspernode,ii);
      #else
      scl_factor = absmax_signed(d,m->nhmgvars)*stab;
      for (int ii=0; ii<m->nvarspernode; ii++) res_per_it[ii][it] = (a * scl_factor) * reduce_with_stride_and_scale(d,m->nhmgvars,m->nvarspernode,ii,1.0/scl_factor);
      #endif
      break;
    case CUDAPCG_XREDUCE_ONLYDIR:
      axpy_iny_with_stride(x,d,a,m->nhmgvars,m->nvarspernode,shift);
      res_per_it[shift][it] = reduce(x,m->nhmgvars/m->nvarspernode);// - (it > 0 ? res_per_it[shift][it-1] : 0.0);
      break;
    case CUDAPCG_XREDUCE_FULL:
      axpy_iny(x,d,a,m->nhmgvars);
      for (int ii=0; ii<m->nvarspernode; ii++) res_per_it[ii][it] = reduce_with_stride(x,m->nhmgvars,m->nvarspernode,ii);// - (it > 0 ? res_per_it[ii][it-1] : 0.0);
      break;
  }
  return;
}
//--------------------------------------------------------------------------------------------------
void save_xreduce(char *filename, cudapcgModel_t *m, unsigned int it, cudapcgFlag_t flag, unsigned int shift, double scl, double **res_per_it){
  FILE * file = fopen(filename,"wb");
  double *ptr=NULL;
  if (file){
    if (flag != CUDAPCG_XREDUCE_ONLYDIR){
      for (int ii=0; ii<m->nvarspernode; ii++){
        ptr = res_per_it[ii];
        #pragma omp parallel for
        for (size_t jj=0; jj<=it; jj++) ptr[jj]*=scl;
        fwrite(ptr,sizeof(double),it+1,file);
      }
    } else {
      ptr = res_per_it[shift];
      #pragma omp parallel for
      for (size_t jj=0; jj<=it; jj++) ptr[jj]*=scl;
      fwrite(ptr,sizeof(double),it+1,file);
    } 
  }
  fclose(file);
  return;
}
//--------------------------------------------------------------------------------------------------
