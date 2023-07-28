#include "femhmg_2D.h"
#include "femhmg_fluid_2D.h"

//------------------------------------------------------------------------------
void assembleDofIdMap_fluid_2D(hmgModel_t *model);
void assembleDofMaterialMap_fluid_2D(hmgModel_t *model);
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
logical initModel_fluid_2D(hmgModel_t *model){
  if ((model->m_nx-1) <= 0 || (model->m_ny-1) <= 0 || model->m_nmat <= 0){
    printf("ERROR: Failed to initialize model due to bad input parameters.\nrows:%d\ncols:%d\nmaterials:%d\n",(model->m_ny-1),(model->m_nx-1),model->m_nmat);
    return HMG_FALSE;
  }

  model->m_C_dim = 4;
  model->m_lclMtx_dim = 144;
  model->m_lclCB_dim = 20;

  model->m_nnodedof = 1; // actually, it's 3. kept as 1 for now because it is not used and it lets me reuse dof map functions
  model->m_nelemdof = 24;
  model->m_nnode = model->m_nx * model->m_ny;
  model->m_nelem = (model->m_nx-1) * (model->m_ny-1);
  model->m_ndof = model->m_nelem*model->m_nnodedof; // needs to be updated

  model->assembleLocalMtxs = assembleLocalMtxs_fluid_2D;
  model->assembleRHS = assembleRHS_fluid_2D;
  model->updateC = updateC_fluid_2D;
  model->printC = printC_fluid_2D;
  model->saveFields = saveFields_fluid_2D;
  
  model->assembleNodeDofMap = assembleNodeDofMap_2D;
  model->assembleDofIdMap = assembleDofIdMap_fluid_2D;
  model->assembleDofMaterialMap = assembleDofMaterialMap_fluid_2D;

  return HMG_TRUE;
}
//------------------------------------------------------------------------------
void assembleLocalMtxs_fluid_2D(hmgModel_t *model){

  // Hardcoded solution for 1x1 water element
  // Takes into account a stabilizer: stab = (dx^2+dy^2)/(12*L) = 1/6
  
  cudapcgVar_t c1 = 1.0/6.0;
  cudapcgVar_t c2 = 1.0/12.0;
  cudapcgVar_t c3 = 1.0/9.0;
  cudapcgVar_t c4 = 1.0/36.0;
  cudapcgVar_t c5 = 1.0/18.0;
  
  // lclK = [ K    G
  //          G' Stab ]
  cudapcgVar_t lclK[] = {  1.0,  0.25,  -0.5, -0.25,  -0.5, -0.25,   0.0,  0.25,  -c1, -c1, -c2, -c2,
                 0.25,   1.0,  0.25,   0.0, -0.25,  -0.5, -0.25,  -0.5,  -c1, -c2, -c2, -c1,
                 -0.5,  0.25,   1.0, -0.25,   0.0, -0.25,  -0.5,  0.25,   c1,  c1,  c2,  c2,
                -0.25,   0.0, -0.25,   1.0,  0.25,  -0.5,  0.25,  -0.5,  -c2, -c1, -c1, -c2,
                 -0.5, -0.25,   0.0,  0.25,   1.0,  0.25,  -0.5, -0.25,   c2,  c2,  c1,  c1,
                -0.25,  -0.5, -0.25,  -0.5,  0.25,   1.0,  0.25,   0.0,   c2,  c1,  c1,  c2,
                  0.0, -0.25,  -0.5,  0.25,  -0.5,  0.25,   1.0, -0.25,  -c2, -c2, -c1, -c1,
                 0.25,  -0.5,  0.25,  -0.5, -0.25,   0.0, -0.25,   1.0,   c1,  c2,  c2,  c1,
                  -c1,   -c1,    c1,   -c2,    c2,    c2,   -c2,    c1,   c3, -c4, -c5, -c4,
                  -c1,   -c2,    c1,   -c1,    c2,    c1,   -c2,    c2,  -c4,  c3, -c4, -c5,
                  -c2,   -c2,    c2,   -c1,    c1,    c1,   -c1,    c2,  -c5, -c4,  c3, -c4,
                  -c2,   -c1,    c2,   -c2,    c1,    c2,   -c1,    c1,  -c4, -c5, -c4,  c3 };
  
  #pragma omp parallel for
  for (unsigned int j=0;j<model->m_lclMtx_dim;j++){
   model->Mtxs[j] = lclK[j];
  }
  
  // lclCB = [ SN 
  //           F  ];
  cudapcgVar_t lclCB[] = { 0.25,  0.0, 0.25,  0.0, 0.25,  0.0, 0.25,  0.0,
                   0.0, 0.25,  0.0, 0.25,  0.0, 0.25,  0.0, 0.25,
                  0.25, 0.25, 0.25, 0.25 };
                  
  #pragma omp parallel for
  for (unsigned int j=0;j<model->m_lclCB_dim;j++){
   model->CB[j] = lclCB[j];
  }

  return;
}
//------------------------------------------------------------------------------
void assembleRHS_fluid_2D(hmgModel_t *model){

  #pragma omp parallel for
  for (unsigned int i=0; i<model->m_ndof; i++)
    model->RHS[i] = 0.0;
    
  unsigned int n, dof, dof_stride;
  if (model->m_hmg_flag == HOMOGENIZE_X){
    dof_stride = 0;
  } else {
    dof_stride = 1;
  }
  // Apply velocity field on domain
  for (unsigned int e=0; e<model->m_nelem; e++){
    // Check if this element is fluid
    if ((model->dof_fluid_map[e]>>3)&1){

      n = e+1+(e/(model->m_ny-1));
      dof = model->dof_id_map[model->node_dof_map[n]];
      if (dof < model->m_nVelocityNodes)
        model->RHS[2*dof+dof_stride] += model->CB[16]; // fe[1]

      n += model->m_ny;
      dof = model->dof_id_map[model->node_dof_map[n]];
      if (dof < model->m_nVelocityNodes)
        model->RHS[2*dof+dof_stride] += model->CB[17]; // fe[2]

      n -= 1;
      dof = model->dof_id_map[model->node_dof_map[n]];
      if (dof < model->m_nVelocityNodes)
        model->RHS[2*dof+dof_stride] += model->CB[18]; // fe[3]

      n -= model->m_ny;
      dof = model->dof_id_map[model->node_dof_map[n]];
      if (dof < model->m_nVelocityNodes)
        model->RHS[2*dof+dof_stride] += model->CB[19]; // fe[4]
    }
  }

  return;
}
//------------------------------------------------------------------------------
/*
void assembleRHS_PressureGrad_fluid_2D(hmgModel_t *model){

  #pragma omp parallel for
  for (unsigned int i=0; i<model->m_nVelocityNodes*2; i++)
    model->RHS[i] = 0.0;

  char str_buffer[1024];

  sprintf(str_buffer,"%s_temperature_%d.bin",model->neutralFile_noExt,model->m_hmg_flag);
  FILE *file = fopen(str_buffer,"rb");
  if (!file) {return;}
  cudapcgVar_t *p_fromfile = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*model->m_nelem);
  fread(p_fromfile,sizeof(cudapcgVar_t)*model->m_nelem,1,file);
  fclose(file);

  unsigned int rows = model->m_ny-1;
  unsigned int cols = model->m_nx-1;
  unsigned int nn = rows*cols;
  
  cudapcgIdMap_t id;
  cudapcgFlag_t fluidkey;
  cudapcgVar_t rhs_x, rhs_y;
  unsigned int n_neighbor;
  unsigned int nInterfaceNodes = model->m_ndof - 3*model->m_nVelocityNodes;
  //printf("fluidkeys = [\n");
  for (unsigned int n=0; n<nn; n++){
    fluidkey = model->dof_fluid_map[n];
    if ((fluidkey&15)==15){
      rhs_x = 0.0;
      rhs_y = 0.0;
      //printf("  %hu,\n",fluidkey);
      id = model->dof_id_map[n];
      rhs_x -= model->Mtxs[8]*p_fromfile[n];
      rhs_y -= model->Mtxs[20]*p_fromfile[n];
      n_neighbor = (n+rows)%nn;
      rhs_x -= model->Mtxs[9]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[21]*p_fromfile[n_neighbor];
      n_neighbor = n_neighbor-1+rows*(!(n_neighbor%rows));
      rhs_x -= model->Mtxs[10]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[22]*p_fromfile[n_neighbor];
      n_neighbor = n-1+rows*(!(n%rows));
      rhs_x -= model->Mtxs[11]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[23]*p_fromfile[n_neighbor];
      n_neighbor = (n+(cols-1)*rows)%nn;
      rhs_x -= model->Mtxs[32]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[44]*p_fromfile[n_neighbor];
      rhs_x -= model->Mtxs[33]*p_fromfile[n];
      rhs_y -= model->Mtxs[45]*p_fromfile[n];
      n_neighbor = n-1+rows*(!(n%rows));
      rhs_x -= model->Mtxs[34]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[46]*p_fromfile[n_neighbor];
      n_neighbor = (n_neighbor+(cols-1)*rows)%nn;
      rhs_x -= model->Mtxs[35]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[47]*p_fromfile[n_neighbor];
      n_neighbor = n+1-rows*(!((n+1)%rows));
      n_neighbor = (n_neighbor+(cols-1)*rows)%nn;
      rhs_x -= model->Mtxs[56]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[68]*p_fromfile[n_neighbor];
      n_neighbor = (n_neighbor+rows)%nn;
      rhs_x -= model->Mtxs[57]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[69]*p_fromfile[n_neighbor];
      rhs_x -= model->Mtxs[58]*p_fromfile[n];
      rhs_y -= model->Mtxs[70]*p_fromfile[n];
      n_neighbor = (n+(cols-1)*rows)%nn;
      rhs_x -= model->Mtxs[59]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[71]*p_fromfile[n_neighbor];
      n_neighbor = n+1-rows*(!((n+1)%rows));
      rhs_x -= model->Mtxs[80]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[92]*p_fromfile[n_neighbor];
      n_neighbor = (n_neighbor+rows)%nn;
      rhs_x -= model->Mtxs[81]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[93]*p_fromfile[n_neighbor];
      n_neighbor = (n+rows)%nn;
      rhs_x -= model->Mtxs[82]*p_fromfile[n_neighbor];
      rhs_y -= model->Mtxs[94]*p_fromfile[n_neighbor];
      rhs_x -= model->Mtxs[83]*p_fromfile[n];
      rhs_y -= model->Mtxs[95]*p_fromfile[n];
      model->RHS[2*id] += rhs_x/((cudapcgVar_t) cols);
      model->RHS[2*id+1] += rhs_y/((cudapcgVar_t) cols);
    }
  }
  //printf("]\n");
  
//  printf("RHS = [\n");
//  for (unsigned int n=0; n<model->m_nelem;n++){//}m_nVelocityNodes*2; n++){
//    printf("  %.4e,\n",p_fromfile[n]);//model->RHS[n]);
//  }
//  printf("]\n");

  free(p_fromfile);

  return;
}
*/
//------------------------------------------------------------------------------
void updateC_fluid_2D(hmgModel_t *model, cudapcgVar_t * V){
  var C_i=0.0,C_j=0.0;
  unsigned int i,j;
  if (model->m_hmg_flag == HOMOGENIZE_X){
    i = 0; j = 2;
  } else if (model->m_hmg_flag == HOMOGENIZE_Y){
    i = 1; j = 3;
  }
  unsigned int n, dof;
  #pragma omp parallel for private(n,dof,C_i,C_j)
  for (unsigned int e=0; e<model->m_nelem; e++){
    // Check if this element is fluid
    if ((model->dof_fluid_map[e]>>3)&1){
    
      C_i = 0.0;
      C_j = 0.0;

      n = e+1+(e/(model->m_ny-1));
      dof = model->dof_id_map[model->node_dof_map[n]];
      if (dof < model->m_nVelocityNodes){
        C_i += model->CB[0] * V[2*dof];
        C_j += model->CB[9] * V[2*dof+1];
      }

      n += model->m_ny;
      dof = model->dof_id_map[model->node_dof_map[n]];
      if (dof < model->m_nVelocityNodes){
        C_i += model->CB[2]  * V[2*dof];
        C_j += model->CB[11] * V[2*dof+1];
      }

      n -= 1;
      dof = model->dof_id_map[model->node_dof_map[n]];
      if (dof < model->m_nVelocityNodes){
        C_i += model->CB[4]  * V[2*dof];
        C_j += model->CB[13] * V[2*dof+1];
      }

      n -= model->m_ny;
      dof = model->dof_id_map[model->node_dof_map[n]];
      if (dof < model->m_nVelocityNodes){
        C_i += model->CB[6]  * V[2*dof];
        C_j += model->CB[15] * V[2*dof+1];
      }
      
      #pragma omp critical
      {
        model->C[i] += C_i; model->C[j] += C_j;
      }
    }
  }
  
  var div = model->m_nelem;
  var mul = model->m_elem_size * model->m_elem_size;
  
  model->C[i] /= div; model->C[j] /= div;
  model->C[i] *= mul; model->C[j] *= mul;
  return;
}
//------------------------------------------------------------------------------
void printC_fluid_2D(hmgModel_t *model, char *dest){
  if (dest==NULL){
    printf("-------------------------------------------------------\n");
    printf("Homogenized Constitutive Matrix (Permeability):\n");
    printf("  %.8e  ", model->C[0]); printf("%.8e\n", model->C[1]);
    printf("  %.8e  ", model->C[2]); printf("%.8e\n", model->C[3]);
    printf("-------------------------------------------------------\n");
  } else {
    sprintf(
      dest,
      "-------------------------------------------------------\n"\
      "Homogenized Constitutive Matrix (Permeability):\n"\
      "  %.8e  %.8e\n"\
      "  %.8e  %.8e\n"\
      "-------------------------------------------------------\n",
      model->C[0], model->C[1],
      model->C[2], model->C[3]
    );
  }
  return;
}
//------------------------------------------------------------------------------
void saveFields_fluid_2D(hmgModel_t *model, cudapcgVar_t * V){

  unsigned int rows = model->m_ny-1;
  unsigned int cols = model->m_nx-1;
  unsigned int nn = rows*cols;

  cudapcgVar_t * u = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*nn*2);
  cudapcgVar_t * p = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*nn);
  
  cudapcgIdMap_t id;
  cudapcgFlag_t fluidkey;
  unsigned int nInterfaceNodes = model->m_ndof - 3*model->m_nVelocityNodes;
  for (unsigned int n=0; n<nn; n++){
    u[2*n] = 0.0;
    u[2*n+1] = 0.0;
    p[n] = 0.0;
    id = model->dof_id_map[n];
    fluidkey = model->dof_fluid_map[n];
    if ((fluidkey&15)==15){
      u[2*n] += V[2*id];
      u[2*n+1] += V[2*id+1];
    }
    if (fluidkey&15){
      p[n] += V[2*model->m_nVelocityNodes+id];
    }
  }
  
  // Save arrays to binary files
  char str_buffer[1024];
  sprintf(str_buffer,"%s_velocity_%d.bin",model->neutralFile_noExt,model->m_hmg_flag);
  FILE * file = fopen(str_buffer,"wb");
  if (file)
    fwrite(u,sizeof(cudapcgVar_t)*nn*2,1,file);
  fclose(file);
  
  sprintf(str_buffer,"%s_pressure_%d.bin",model->neutralFile_noExt,model->m_hmg_flag);
  file = fopen(str_buffer,"wb");
  if (file)
    fwrite(p,sizeof(cudapcgVar_t)*nn,1,file);
  fclose(file);

  free(u);
  free(p);

  return;
}
//------------------------------------------------------------------------------
/*
void laplace2stokes_fluid_2D(hmgModel_t *model){

  char str_buffer[1024];

  //sprintf(str_buffer,"%s_flux_%d.bin",model->neutralFile_noExt,model->m_hmg_flag);
  //FILE *file = fopen(str_buffer,"rb");
  //if (!file) return;
  //cudapcgVar_t *x0_fromfile = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*model->m_nelem*3); //qx,qy,t
  //fread(x0_fromfile,sizeof(cudapcgVar_t)*model->m_nelem*2,1,file);
  //fclose(file);  

  sprintf(str_buffer,"%s_temperature_%d.bin",model->neutralFile_noExt,model->m_hmg_flag);
  FILE *file = fopen(str_buffer,"rb");
  if (!file) {return;}//{free(x0_fromfile); return;}
  cudapcgVar_t *x0_fromfile = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*model->m_nelem);
  fread(x0_fromfile,sizeof(cudapcgVar_t)*model->m_nelem,1,file);
  //fread(&x0_fromfile[model->m_nelem*2],sizeof(cudapcgVar_t)*model->m_nelem,1,file);
  fclose(file);

  cudapcgVar_t *x0 = model->x0[model->m_hmg_flag];

  unsigned int rows = model->m_ny-1;
  unsigned int cols = model->m_nx-1;
  unsigned int nn = rows*cols;
  
  cudapcgIdMap_t id;
  cudapcgFlag_t fluidkey;
  unsigned int nInterfaceNodes = model->m_ndof - 3*model->m_nVelocityNodes;
  for (unsigned int n=0; n<nn; n++){
    id = model->dof_id_map[n];
    fluidkey = model->dof_fluid_map[n];
    
    //if ((fluidkey&15)==15){
    //  x0[2*id] = -x0_fromfile[2*n];
    //  x0[2*id+1] = -x0_fromfile[2*n+1];
    //}
    
    if (fluidkey&15){
      x0[2*model->m_nVelocityNodes+id] = x0_fromfile[n]/((cudapcgVar_t) cols);//[2*model->m_nelem+n];
    }
  }

  free(x0_fromfile);

  return;
}
*/
//------------------------------------------------------------------------------

//---------------------------------
///////////////////////////////////
//////// PRIVATE FUNCTIONS ////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
void assembleDofIdMap_fluid_2D(hmgModel_t *model){
  unsigned int dof_count = 0;
  // Start by setting dof ids to nodes within fluid
  for (unsigned int i=0; i<model->m_nelem; i++){ // not really going through elems, but through nodes with diff dofs
    if ((model->dof_fluid_map[i]&15)==15){
      model->dof_id_map[i] = dof_count;
      dof_count++;
    } else {
      model->dof_id_map[i] = 0;
    }
  }
  model->m_nVelocityNodes = dof_count;
  // Now number solid-fluid interface nodes
  for (unsigned int i=0; i<model->m_nelem; i++){ // not really going through elems, but through nodes with diff dofs
    if ((model->dof_fluid_map[i]&15)!=15 && (model->dof_fluid_map[i]&15)!=0){
      model->dof_id_map[i] = dof_count;
      dof_count++;
    }
  }
  unsigned int nInterfaceNodes = dof_count - model->m_nVelocityNodes;
  model->m_ndof = 3*model->m_nVelocityNodes + nInterfaceNodes;
  
  model->m_nBorderNodes = nInterfaceNodes;

  if (model->poremap_flag == CUDAPCG_POREMAP_NUM){
  
    model->pore_dof2node_map = (cudapcgIdMap_t *)malloc(sizeof(cudapcgIdMap_t)*(model->m_nVelocityNodes+model->m_nBorderNodes));
    cudapcgIdMap_t buffer;
    for (unsigned int i=0; i<model->m_nelem; i++){
      if (model->dof_fluid_map[i]){
        buffer = model->dof_id_map[i];
        if (buffer < (model->m_nVelocityNodes+model->m_nBorderNodes))
          model->pore_dof2node_map[buffer] = i;
      }
    }
    
    model->pore_border_fluidkeys = (cudapcgFlag_t *)malloc(sizeof(cudapcgFlag_t)*model->m_nBorderNodes);
    #pragma omp parallel for
    for (unsigned int i=0; i<model->m_nBorderNodes; i++)
      model->pore_border_fluidkeys[i] = 0;
    unsigned char fluidkey;
    unsigned int n;
    
    #pragma omp parallel for private(fluidkey)
    for (unsigned int i=0; i<model->m_nelem; i++){
      fluidkey = model->dof_fluid_map[i];
      if (fluidkey && fluidkey!=15)
        model->pore_border_fluidkeys[model->dof_id_map[i]-model->m_nVelocityNodes] = fluidkey;
    }
  }
  
  return;
}
//------------------------------------------------------------------------------
void assembleDofMaterialMap_fluid_2D(hmgModel_t *model){
  #pragma omp parallel for
  for (unsigned int i=0; i<model->m_nelem; i++) // not really going through elems, but through nodes with diff dofs
    model->dof_fluid_map[i] = 0;
  unsigned int n;
  unsigned int dim_y = model->m_ny-1;
  // node 0 (left,bottom)
  #pragma omp parallel for private(n)
  for (unsigned int e=0;e<model->m_nelem;e++){
    n = e+1+(e/dim_y);
    model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1);
  }
  // node 1 (right,bottom)
  #pragma omp parallel for private(n)
  for (unsigned int e=0;e<model->m_nelem;e++){
    n = e+1+(e/dim_y)+model->m_ny;
    model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1)<<1;
  }
  // node 2 (right,top)
  #pragma omp parallel for private(n)
  for (unsigned int e=0;e<model->m_nelem;e++){
    n = e+(e/dim_y)+model->m_ny;
    model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1)<<2;
  }
  // node 3 (left,top)
  #pragma omp parallel for private(n)
  for (unsigned int e=0;e<model->m_nelem;e++){
    n = e+(e/dim_y);
    model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1)<<3;
  }
  
  if (model->poremap_flag == CUDAPCG_POREMAP_NUM){
    free(model->elem_material_map);
    model->elem_material_map = NULL;
  }
  return;
}
//------------------------------------------------------------------------------
