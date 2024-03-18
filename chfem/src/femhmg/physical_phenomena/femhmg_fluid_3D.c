#include "femhmg_3D.h"
#include "femhmg_fluid_3D.h"

//------------------------------------------------------------------------------
void assembleDofIdMap_fluid_3D(hmgModel_t *model);
void assembleDofMaterialMap_fluid_3D(hmgModel_t *model);
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
logical initModel_fluid_3D(hmgModel_t *model){
	if ((model->m_nx-1) <= 0 || (model->m_ny-1) <= 0 || (model->m_nz-1) <= 0 || model->m_nmat <= 0){
	  printf("ERROR: Failed to initialize model due to bad input parameters.\nrows:%d\ncols:%d\nslices:%d\nmaterials:%d\n",(model->m_ny-1),(model->m_nx-1),(model->m_nz-1),model->m_nmat);
		return HMG_FALSE;
	}

	model->m_C_dim = 9;
	model->m_lclMtx_dim = 1024;
	model->m_lclCB_dim = 80;

	model->m_nnodedof = 1; // actually, it's 4. kept as 1 for now because it is not used and it lets me reuse dof map functions
	model->m_nelemdof = 32;
	model->m_nnode = model->m_nx * model->m_ny * model->m_nz;
	model->m_nelem = (model->m_nx-1) * (model->m_ny-1) * (model->m_nz-1);
	model->m_nporeelem = model->m_nelem; // needs to be updated afterwards (by calling poremapping functions)
	model->m_ndof = model->m_nelem*model->m_nnodedof; // needs to be updated afterwards (by calling poremapping functions)
	

	model->assembleLocalMtxs = assembleLocalMtxs_fluid_3D;
	model->assembleRHS = assembleRHS_fluid_3D;
	model->updateC = updateC_fluid_3D;
	model->printC = printC_fluid_3D;
	model->saveFields = saveFields_fluid_3D;
	
	model->assembleNodeDofMap = assembleNodeDofMap_3D;
	model->assembleDofIdMap = assembleDofIdMap_fluid_3D;
	model->assembleDofMaterialMap = assembleDofMaterialMap_fluid_3D;

	return HMG_TRUE;
}
//------------------------------------------------------------------------------
void assembleLocalMtxs_fluid_3D(hmgModel_t *model){

	// Hardcoded solution for 1x1x1 water element
  // Takes into account a stabilizer: stab = (dx^2+dy^2)/(12*L) = 1/6
  
  cudapcgVar_t c1 = 4.0/9.0;
  cudapcgVar_t c2 = 1.0/12.0;
  cudapcgVar_t c3 = 1.0/9.0;
  cudapcgVar_t c4 = 1.0/18.0;
  cudapcgVar_t c5 = 1.0/7.2;
  cudapcgVar_t c6 = 1.0/24.0;
  cudapcgVar_t c7 = 1.0/36.0;
  cudapcgVar_t c8 = 1.0/72.0;
  
  // lclK = [ K    G
  //          G' Stab ]
  cudapcgVar_t lclK[] = { c1, c2, c2, -c3, -c2, -c2, -c5, -c2, -c6, c4, c2, c6, c4, c6, c2, -c5, -c6, -c2, -c3, -c6, -c6, -c4, c6, c6, -c4, -c4, -c7, -c7, -c7, -c7, -c8, -c8,
                c2, c1, c2, c2, c4, c6, -c2, -c5, -c6, -c2, -c3, -c2, c6, c4, c2, c6, -c4, c6, -c6, -c3, -c6, -c6, -c5, -c2, -c4, -c7, -c7, -c4, -c7, -c8, -c8, -c7,
                c2, c2, c1, c2, c6, c4, c6, c6, -c4, c6, c2, c4, -c2, -c2, -c3, -c2, -c6, -c5, -c6, -c6, -c3, -c6, -c2, -c5, -c4, -c7, -c8, -c7, -c4, -c7, -c8, -c7,
                -c3, c2, c2, c1, -c2, -c2, c4, -c2, -c6, -c5, c2, c6, -c5, c6, c2, c4, -c6, -c2, -c4, -c6, -c6, -c3, c6, c6, c4, c4, c7, c7, c7, c7, c8, c8,
                -c2, c4, c6, -c2, c1, c2, c2, -c3, -c2, c2, -c5, -c6, -c6, -c4, c6, -c6, c4, c2, c6, -c5, -c2, c6, -c3, -c6, -c7, -c4, -c4, -c7, -c8, -c7, -c7, -c8,
                -c2, c6, c4, -c2, c2, c1, -c6, c2, c4, -c6, c6, -c4, c2, -c6, -c5, c2, -c2, -c3, c6, -c2, -c5, c6, -c6, -c3, -c7, -c4, -c7, -c8, -c7, -c4, -c7, -c8,
                -c5, -c2, c6, c4, c2, -c6, c1, c2, -c2, -c3, -c2, c2, -c3, -c6, c6, -c4, c6, -c6, c4, c6, -c2, -c5, -c6, c2, c7, c7, c4, c4, c8, c8, c7, c7,
                -c2, -c5, c6, -c2, -c3, c2, c2, c1, -c2, c2, c4, -c6, -c6, -c3, c6, -c6, -c5, c2, c6, c4, -c2, c6, -c4, -c6, c7, c4, c4, c7, c8, c7, c7, c8,
                -c6, -c6, -c4, -c6, -c2, c4, -c2, -c2, c1, -c2, -c6, c4, c6, c6, -c3, c6, c2, -c5, c2, c2, -c3, c2, c6, -c5, -c8, -c7, -c4, -c7, -c8, -c7, -c4, -c7,
                c4, -c2, c6, -c5, c2, -c6, -c3, c2, -c2, c1, -c2, c2, -c4, -c6, c6, -c3, c6, -c6, -c5, c6, -c2, c4, -c6, c2, -c7, -c7, -c4, -c4, -c8, -c8, -c7, -c7,
                c2, -c3, c2, c2, -c5, c6, -c2, c4, -c6, -c2, c1, -c2, c6, -c5, c2, c6, -c3, c6, -c6, -c4, -c6, -c6, c4, -c2, c4, c7, c7, c4, c7, c8, c8, c7,
                c6, -c2, c4, c6, -c6, -c4, c2, -c6, c4, c2, -c2, c1, -c6, c2, -c5, -c6, c6, -c3, -c2, c6, -c5, -c2, c2, -c3, -c7, -c8, -c7, -c4, -c7, -c8, -c7, -c4,
                c4, c6, -c2, -c5, -c6, c2, -c3, -c6, c6, -c4, c6, -c6, c1, c2, -c2, -c3, -c2, c2, -c5, -c2, c6, c4, c2, -c6, -c7, -c7, -c8, -c8, -c4, -c4, -c7, -c7,
                c6, c4, -c2, c6, -c4, -c6, -c6, -c3, c6, -c6, -c5, c2, c2, c1, -c2, c2, c4, -c6, -c2, -c5, c6, -c2, -c3, c2, -c7, -c8, -c8, -c7, -c4, -c7, -c7, -c4,
                c2, c2, -c3, c2, c6, -c5, c6, c6, -c3, c6, c2, -c5, -c2, -c2, c1, -c2, -c6, c4, -c6, -c6, -c4, -c6, -c2, c4, c4, c7, c8, c7, c4, c7, c8, c7,
                -c5, c6, -c2, c4, -c6, c2, -c4, -c6, c6, -c3, c6, -c6, -c3, c2, -c2, c1, -c2, c2, c4, -c2, c6, -c5, c2, -c6, c7, c7, c8, c8, c4, c4, c7, c7,
                -c6, -c4, -c6, -c6, c4, -c2, c6, -c5, c2, c6, -c3, c6, -c2, c4, -c6, -c2, c1, -c2, c2, -c3, c2, c2, -c5, c6, -c8, -c7, -c7, -c8, -c7, -c4, -c4, -c7,
                -c2, c6, -c5, -c2, c2, -c3, -c6, c2, -c5, -c6, c6, -c3, c2, -c6, c4, c2, -c2, c1, c6, -c2, c4, c6, -c6, -c4, c7, c4, c7, c8, c7, c4, c7, c8,
                -c3, -c6, -c6, -c4, c6, c6, c4, c6, c2, -c5, -c6, -c2, -c5, -c2, -c6, c4, c2, c6, c1, c2, c2, -c3, -c2, -c2, c8, c8, c7, c7, c7, c7, c4, c4,
                -c6, -c3, -c6, -c6, -c5, -c2, c6, c4, c2, c6, -c4, c6, -c2, -c5, -c6, -c2, -c3, -c2, c2, c1, c2, c2, c4, c6, c8, c7, c7, c8, c7, c4, c4, c7,
                -c6, -c6, -c3, -c6, -c2, -c5, -c2, -c2, -c3, -c2, -c6, -c5, c6, c6, -c4, c6, c2, c4, c2, c2, c1, c2, c6, c4, c8, c7, c4, c7, c8, c7, c4, c7,
                -c4, -c6, -c6, -c3, c6, c6, -c5, c6, c2, c4, -c6, -c2, c4, -c2, -c6, -c5, c2, c6, -c3, c2, c2, c1, -c2, -c2, -c8, -c8, -c7, -c7, -c7, -c7, -c4, -c4,
                c6, -c5, -c2, c6, -c3, -c6, -c6, -c4, c6, -c6, c4, c2, c2, -c3, -c2, c2, -c5, -c6, -c2, c4, c6, -c2, c1, c2, c7, c8, c8, c7, c4, c7, c7, c4,
                c6, -c2, -c5, c6, -c6, -c3, c2, -c6, -c5, c2, -c2, -c3, -c6, c2, c4, -c6, c6, -c4, -c2, c6, c4, -c2, c2, c1, c7, c8, c7, c4, c7, c8, c7, c4,
                -c4, -c4, -c4, c4, -c7, -c7, c7, c7, -c8, -c7, c4, -c7, -c7, -c7, c4, c7, -c8, c7, c8, c8, c8, -c8, c7, c7, c4, 0.0, -c8, 0.0, 0.0, -c8, -c8, -c8,
                -c4, -c7, -c7, c4, -c4, -c4, c7, c4, -c7, -c7, c7, -c8, -c7, -c8, c7, c7, -c7, c4, c8, c7, c7, -c8, c8, c8, 0.0, c4, 0.0, -c8, -c8, 0.0, -c8, -c8,
                -c7, -c7, -c8, c7, -c4, -c7, c4, c4, -c4, -c4, c7, -c7, -c8, -c8, c8, c8, -c7, c7, c7, c7, c4, -c7, c8, c7, -c8, 0.0, c4, 0.0, -c8, -c8, 0.0, -c8,
                -c7, -c4, -c7, c7, -c7, -c8, c4, c7, -c7, -c4, c4, -c4, -c8, -c7, c7, c8, -c8, c8, c7, c8, c7, -c7, c7, c4, 0.0, -c8, 0.0, c4, -c8, -c8, -c8, 0.0,
                -c7, -c7, -c4, c7, -c8, -c7, c8, c8, -c8, -c8, c7, -c7, -c4, -c4, c4, c4, -c7, c7, c7, c7, c8, -c7, c4, c7, 0.0, -c8, -c8, -c8, c4, 0.0, -c8, 0.0,
                -c7, -c8, -c7, c7, -c7, -c4, c8, c7, -c7, -c8, c8, -c8, -c4, -c7, c7, c4, -c4, c4, c7, c4, c7, -c7, c7, c8, -c8, 0.0, -c8, -c8, 0.0, c4, 0.0, -c8,
                -c8, -c8, -c8, c8, -c7, -c7, c7, c7, -c4, -c7, c8, -c7, -c7, -c7, c8, c7, -c4, c7, c4, c4, c4, -c4, c7, c7, -c8, -c8, 0.0, -c8, -c8, 0.0, c4, 0.0,
                -c8, -c7, -c7, c8, -c8, -c8, c7, c8, -c7, -c7, c7, -c4, -c7, -c4, c7, c7, -c7, c8, c4, c7, c7, -c4, c4, c4, -c8, -c8, -c8, 0.0, 0.0, -c8, 0.0, c4 };

  
  #pragma omp parallel for
  for (unsigned int j=0;j<model->m_lclMtx_dim;j++){
   model->Mtxs[j] = lclK[j];
  }
  
  // lclCB = [ SN 
  //           F  ];
  cudapcgVar_t lclCB[] = { 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0,
                    0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0,
                    0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125,
                  0.125,0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125 };

                  
  #pragma omp parallel for
  for (unsigned int j=0;j<model->m_lclCB_dim;j++){
   model->CB[j] = lclCB[j];
  }

  return;
}
//------------------------------------------------------------------------------
void assembleRHS_fluid_3D(hmgModel_t *model){

	#pragma omp parallel for
	for (unsigned int i=0; i<model->m_ndof; i++)
		model->RHS[i] = 0.0;
	
	unsigned int dim_y  = model->m_ny-1;
	unsigned int dim_xy = (model->m_nx-1)*dim_y;
  unsigned int n, dof, dof_stride;
  if (model->m_hmg_flag == HOMOGENIZE_X){
    dof_stride = 0;
  } else if (model->m_hmg_flag == HOMOGENIZE_Y) {
    dof_stride = 1;
  } else {
    dof_stride = 2;
  }

  // Apply velocity field on domain
  for (unsigned int e=0; e<model->m_nelem; e++){
    // Check if this element is fluid
    if ((model->dof_fluid_map[e]>>3)&1){

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes)
			  model->RHS[3*dof+dof_stride] += model->CB[72]; // fe[1]

			n += model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes)
			  model->RHS[3*dof+dof_stride] += model->CB[73]; // fe[2]

			n -= 1;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes)
			  model->RHS[3*dof+dof_stride] += model->CB[74]; // fe[3]

			n -= model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes)
			  model->RHS[3*dof+dof_stride] += model->CB[75]; // fe[4]
			
			n += 1+model->m_nx*model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes)
			  model->RHS[3*dof+dof_stride] += model->CB[76]; // fe[5]

			n += model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes)
			  model->RHS[3*dof+dof_stride] += model->CB[77]; // fe[6]

			n -= 1;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes)
			  model->RHS[3*dof+dof_stride] += model->CB[78]; // fe[7]

			n -= model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes)
			  model->RHS[3*dof+dof_stride] += model->CB[79]; // fe[8]
    }
  }
  
  /*printf("RHS: [\n");
  for (unsigned int i=0; i<model->m_ndof; i++) printf("%.4e\n",model->RHS[i]);
  printf("]\n");*/

  return;
}

//------------------------------------------------------------------------------
void updateC_fluid_3D(hmgModel_t *model, cudapcgVar_t * V){
	var C_i=0.0,C_j=0.0,C_k=0.0;
  unsigned int dim_y = model->m_ny-1;
  unsigned int dim_xy = (model->m_nx-1)*dim_y;
  unsigned int i,j,k;
	if (model->m_hmg_flag == HOMOGENIZE_X){
		i = 0; j = 3, k = 6;
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		i = 1; j = 4, k = 7;
	} else if (model->m_hmg_flag == HOMOGENIZE_Z){
		i = 2; j = 5, k = 8;
	}
  unsigned int n, dof;
  #pragma omp parallel for private(n,dof,C_i,C_j,C_k)
  for (unsigned int e=0; e<model->m_nelem; e++){
    // Check if this element is fluid
    if ((model->dof_fluid_map[e]>>3)&1){
    
      C_i = 0.0;
      C_j = 0.0;
      C_k = 0.0;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes){
			  C_i += model->CB[0]  * V[3*dof];
			  C_j += model->CB[25] * V[3*dof+1];
			  C_k += model->CB[50] * V[3*dof+2];
			}

			n += model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes){
			  C_i += model->CB[3]  * V[3*dof];
			  C_j += model->CB[28] * V[3*dof+1];
			  C_k += model->CB[53] * V[3*dof+2];
			}

			n -= 1;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes){
			  C_i += model->CB[6]  * V[3*dof];
			  C_j += model->CB[31] * V[3*dof+1];
			  C_k += model->CB[56] * V[3*dof+2];
			}

			n -= model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes){
			  C_i += model->CB[9]  * V[3*dof];
			  C_j += model->CB[34] * V[3*dof+1];
			  C_k += model->CB[59] * V[3*dof+2];
			}
			
			n += 1+model->m_nx*model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes){
			  C_i += model->CB[12] * V[3*dof];
			  C_j += model->CB[37] * V[3*dof+1];
			  C_k += model->CB[62] * V[3*dof+2];
			}

			n += model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes){
			  C_i += model->CB[15] * V[3*dof];
			  C_j += model->CB[40] * V[3*dof+1];
			  C_k += model->CB[65] * V[3*dof+2];
			}

			n -= 1;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes){
			  C_i += model->CB[18] * V[3*dof];
			  C_j += model->CB[43] * V[3*dof+1];
			  C_k += model->CB[68] * V[3*dof+2];
			}

			n -= model->m_ny;
			dof = model->dof_id_map[model->node_dof_map[n]];
			if (dof < model->m_nVelocityNodes){
			  C_i += model->CB[21] * V[3*dof];
			  C_j += model->CB[46] * V[3*dof+1];
			  C_k += model->CB[71] * V[3*dof+2];
			}
			
			#pragma omp critical
		  {
			  model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
		  }
    }
  }
  
  var div = model->m_nelem;
  var mul = model->m_elem_size * model->m_elem_size; // not sure about this yet
  
  model->C[i] /= div; model->C[j] /= div; model->C[k] /= div;
  model->C[i] *= mul; model->C[j] *= mul; model->C[k] *= mul;
  return;
}
//------------------------------------------------------------------------------
void printC_fluid_3D(hmgModel_t *model, char *dest){
	if (dest==NULL){
	  printf("-------------------------------------------------------\n");
	  printf("Homogenized Constitutive Matrix (Permeability):\n");
	  printf("  %.8e  ", model->C[0]); printf("%.8e  ", model->C[1]); printf("%.8e\n", model->C[2]);
	  printf("  %.8e  ", model->C[3]); printf("%.8e  ", model->C[4]); printf("%.8e\n", model->C[5]);
	  printf("  %.8e  ", model->C[6]); printf("%.8e  ", model->C[7]); printf("%.8e\n", model->C[8]);
	  printf("-------------------------------------------------------\n");
	} else {
	  sprintf(
      dest,
      "-------------------------------------------------------\n"\
      "Homogenized Constitutive Matrix (Permeability):\n"\
      "  %.8e  %.8e  %.8e\n"\
      "  %.8e  %.8e  %.8e\n"\
      "  %.8e  %.8e  %.8e\n"\
      "-------------------------------------------------------\n",
      model->C[0], model->C[1], model->C[2],
      model->C[3], model->C[4], model->C[5],
      model->C[6], model->C[7], model->C[8]
    );
  }
	return;
}
//------------------------------------------------------------------------------
void saveFields_fluid_3D(hmgModel_t *model, cudapcgVar_t * V){

  unsigned int rows = model->m_ny-1;
  unsigned int cols = model->m_nx-1;
  unsigned int lays = model->m_nz-1;
  unsigned int nn = rows*cols*lays;

  cudapcgVar_t * u = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*nn*3);
  cudapcgVar_t * p = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*nn);
  
  cudapcgIdMap_t id;
  cudapcgFlag_t fluidkey;
  unsigned int nInterfaceNodes = model->m_ndof - 4*model->m_nVelocityNodes;
  for (unsigned int n=0; n<nn; n++){
    u[3*n] = 0.0;
    u[3*n+1] = 0.0;
    u[3*n+2] = 0.0;
    p[n] = 0.0;
    id = model->dof_id_map[n];
    fluidkey = model->dof_fluid_map[n];
    if (fluidkey==255){
      u[3*n] += V[3*id];
      u[3*n+1] += V[3*id+1];
      u[3*n+2] += V[3*id+2];
    }
    if (fluidkey){
      p[n] += V[3*model->m_nVelocityNodes+id];
    }
  }
  
  // Save arrays to binary files
  char str_buffer[1024];
  sprintf(str_buffer,"%s_velocity_%d.bin",model->neutralFile_noExt,model->m_hmg_flag);
  FILE * file = fopen(str_buffer,"wb");
  if (file)
    fwrite(u,sizeof(cudapcgVar_t)*nn*3,1,file);
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

//---------------------------------
///////////////////////////////////
//////// PRIVATE FUNCTIONS ////////
///////////////////////////////////
//---------------------------------

//------------------------------------------------------------------------------
void assembleDofIdMap_fluid_3D(hmgModel_t *model){	
  unsigned int dof_count = 0;
  // Start by setting dof ids to nodes within fluid
	for (unsigned int i=0; i<model->m_nelem; i++){ // not really going through elems, but through nodes with diff dofs
		if (model->dof_fluid_map[i]==255){
		  model->dof_id_map[i] = dof_count;
		  dof_count++;
		} else {
		  model->dof_id_map[i] = 0;
		}
	}
	model->m_nVelocityNodes = dof_count;
	// Now number solid-fluid interface nodes
	for (unsigned int i=0; i<model->m_nelem; i++){ // not really going through elems, but through nodes with diff dofs
		if (model->dof_fluid_map[i]!=255 && model->dof_fluid_map[i]!=0){
		  model->dof_id_map[i] = dof_count;
		  dof_count++;
		}
	}
	unsigned int nInterfaceNodes = dof_count - model->m_nVelocityNodes;
	model->m_ndof = 4*model->m_nVelocityNodes + nInterfaceNodes;
	
	
	
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
	    if (fluidkey && fluidkey!=255)
	      model->pore_border_fluidkeys[model->dof_id_map[i]-model->m_nVelocityNodes] = fluidkey;
	  }
	}
	
	return;
}
//------------------------------------------------------------------------------
void assembleDofMaterialMap_fluid_3D(hmgModel_t *model){
  // count porosity
  unsigned int pore_count = 0;
	for (unsigned int i=0; i<model->m_nelem; i++) if (model->elem_material_map[i]==0) pore_count++;
	model->m_nporeelem = pore_count;
	
  #pragma omp parallel for
	for (unsigned int i=0; i<model->m_nelem; i++) // not really going through elems, but through nodes with diff dofs
		model->dof_fluid_map[i] = 0;
	unsigned int n;
	unsigned int dim_y = model->m_ny-1;
	unsigned int dim_xy = (model->m_nx-1)*dim_y;
	// node 0 (left,bottom,near)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1);
	}
	// node 1 (right,bottom,near)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = model->m_ny+1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1)<<1;
	}
	// node 2 (right,top,near)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = model->m_ny+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1)<<2;
	}
	// node 3 (left,top,near)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = (e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1)<<3;
	}
	// node 4 (left,bottom,far)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy+1)*model->m_nx*model->m_ny;
		model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1)<<4;
	}
	// node 5 (right,bottom,far)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = model->m_ny+1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy+1)*model->m_nx*model->m_ny;
		model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1)<<5;
	}
	// node 6 (right,top,far)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = model->m_ny+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy+1)*model->m_nx*model->m_ny;
		model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1)<<6;
	}
	// node 7 (left,top,far)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = (e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy+1)*model->m_nx*model->m_ny;
		model->dof_fluid_map[model->node_dof_map[n]]+=(model->elem_material_map[e]<1)<<7;
	}
	
	if (model->poremap_flag == CUDAPCG_POREMAP_NUM){
    free(model->elem_material_map);
    model->elem_material_map = NULL;
  }
}
//------------------------------------------------------------------------------
