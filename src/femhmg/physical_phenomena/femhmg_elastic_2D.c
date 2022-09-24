/*
	Universidade Federal Fluminense (UFF) - Niteroi, Brazil
	Institute of Computing
	Authors: Cortez, P., Vianna, R., Sapucaia, V., Pereira., A.
	History: 
		* v0.0 (jul/2020) [ALL]    -> OpenMp, parallelization of CPU code
		* v1.0 (nov/2020) [CORTEZ] -> CUDA, PCG on GPU. (substituted parpcg.h with cudapcg.h)

	API for the handling of FEM models used in
    material homogenization.
    Makes use of cudapcg.h to solve linear system
    of equations.

    ATTENTION.1:
        Considers a structured regular mesh of quad elements (2D)
        or hexahedron elements (3D).

    ATTENTION.2:
        As it is, works for Potential or Elasticity, both 2D or 3D.
*/

#include "femhmg_2D.h"
#include "femhmg_elastic_2D.h"

//------------------------------------------------------------------------------
logical initModel_elastic_2D(hmgModel_t *model){
	if ((model->m_nx-1) <= 0 || (model->m_ny-1) <= 0 || model->m_nmat <= 0){
	  printf("ERROR: Failed to initialize model due to bad input parameters.\nrows:%d\ncols:%d\nmaterials:%d\n",(model->m_ny-1),(model->m_nx-1),model->m_nmat);
		return HMG_FALSE;
	}

	model->m_C_dim = 9;
	model->m_lclMtx_dim = 64;
	model->m_lclCB_dim = 24;

	model->m_nnodedof = 2;
	model->m_nelemdof = 8;
	model->m_nnode = model->m_nx * model->m_ny;
	model->m_nelem = (model->m_nx-1) * (model->m_ny-1);
	model->m_ndof = model->m_nelem*model->m_nnodedof;

  #ifdef ELASTIC_2D_PLANESTRESS
	  model->assembleLocalMtxs = assembleLocalMtxs_elastic_2D_PlaneStress;
	#else // ELASTIC_2D_PLANESTRAIN
	  model->assembleLocalMtxs = assembleLocalMtxs_elastic_2D_PlaneStrain;
	#endif
	model->assembleRHS = assembleRHS_elastic_2D;
	model->updateC = updateC_elastic_2D;
	model->printC = printC_elastic_2D;
	
	model->assembleNodeDofMap = assembleNodeDofMap_2D;
	model->assembleDofMaterialMap = assembleDofMaterialMap_2D;

	return HMG_TRUE;
}
//------------------------------------------------------------------------------
void assembleLocalMtxs_elastic_2D_PlaneStrain(hmgModel_t *model){

	cudapcgVar_t E, v;// coef;
	cudapcgVar_t c_1, c_2, c_3, c_4, c_5, c_6, c_7;

	unsigned int i, mat;
	cudapcgVar_t * thisK;
	for (i=0;i<model->m_nmat;i++){

		E = model->props[i*2];
		v = model->props[i*2+1];
		/*
		coef = E / ((1+v)*(1-2*v));
		
		c_1 = coef*(0.5 - 2*v/3.0);
		c_2 = coef*(0.125 - 0.5*v);
		c_3 = coef*(v/6.0);
		c_4 = coef*(0.125);
		c_5 = coef*(-0.25 + v/3.0);
		c_6 = coef*(-0.25 + v/6.0);
		*/

		c_1 = (E*(4*v - 3))/(6*(2*v - 1)*(v + 1));
		c_2 = E/(9*(2*v - 1)) - (5*E)/(36*(v + 1));
		c_3 = E/(8*(2*v - 1)*(v + 1));
		c_4 = (E*(4*v - 1))/(8*(2*v - 1)*(v + 1));
		c_5 = -(E*(2*v - 3))/(12*(2*v - 1)*(v + 1));
		c_6 = -(E*(4*v - 3))/(12*(2*v - 1)*(v + 1));
		c_7 = -(E*v)/(6*(2*v - 1)*(v + 1));

		/* Analytical solution for local K:
			[  c_1, -c_3,  c_2, -c_4,  c_6,  c_3,  c_7,  c_4 ]
			[ -c_3,  c_1,  c_4,  c_7,  c_3,  c_6, -c_4,  c_5 ]
			[  c_2,  c_4,  c_1,  c_3,  c_7, -c_4,  c_6, -c_3 ]
			[ -c_4,  c_7,  c_3,  c_1,  c_4,  c_5, -c_3,  c_6 ]
			[  c_6,  c_3,  c_7,  c_4,  c_1, -c_3,  c_5, -c_4 ]
			[  c_3,  c_6, -c_4,  c_5, -c_3,  c_1,  c_4,  c_7 ]
			[  c_7, -c_4,  c_6, -c_3,  c_5,  c_4,  c_1,  c_3 ]
			[  c_4,  c_5, -c_3,  c_6, -c_4,  c_7,  c_3,  c_1 ]
		*/

		thisK = &(model->Mtxs[i*model->m_lclMtx_dim]);

		thisK[0] = c_1; thisK[1] = -c_3; thisK[2] = c_2; thisK[3] = -c_4;
		thisK[4] = c_6; thisK[5] =  c_3; thisK[6] = c_7; thisK[7] =  c_4;

		thisK[8]  = -c_3; thisK[9]  = c_1; thisK[10] =  c_4; thisK[11] = c_7;
		thisK[12] =  c_3; thisK[13] = c_6; thisK[14] = -c_4; thisK[15] = c_5;

		thisK[16] = c_2; thisK[17] =  c_4; thisK[18] = c_1; thisK[19] =  c_3;
		thisK[20] = c_7; thisK[21] = -c_4; thisK[22] = c_6; thisK[23] = -c_3;

		thisK[24] = -c_4; thisK[25] = c_7; thisK[26] =  c_3; thisK[27] = c_1;
		thisK[28] =  c_4; thisK[29] = c_5; thisK[30] = -c_3; thisK[31] = c_6;

		thisK[32] = c_6; thisK[33] =  c_3; thisK[34] = c_7; thisK[35] =  c_4;
		thisK[36] = c_1; thisK[37] = -c_3; thisK[38] = c_5; thisK[39] = -c_4;

		thisK[40] =  c_3; thisK[41] = c_6; thisK[42] = -c_4; thisK[43] = c_5;
		thisK[44] = -c_3; thisK[45] = c_1; thisK[46] =  c_4; thisK[47] = c_7;

		thisK[48] = c_7; thisK[49] = -c_4; thisK[50] = c_6; thisK[51] = -c_3;
		thisK[52] = c_5; thisK[53] =  c_4; thisK[54] = c_1; thisK[55] =  c_3;

		thisK[56] =  c_4; thisK[57] = c_5; thisK[58] = -c_3; thisK[59] = c_6;
		thisK[60] = -c_4; thisK[61] = c_7; thisK[62] =  c_3; thisK[63] = c_1;

		c_1 = (E*(v - 1))/(2*(2*v - 1)*(v + 1));
		c_2 = (E*v)/(2*(2*v - 1)*(v + 1));
		c_3 = E/(4*(v + 1));

		/* Analytical solution for local model->C*B:
			[ -c_1,  c_2,  c_1,  c_2,  c_1, -c_2, -c_1, -c_2 ]
			[  c_2, -c_1, -c_2, -c_1, -c_2,  c_1,  c_2,  c_1 ]
			[ -c_3, -c_3, -c_3,  c_3,  c_3,  c_3,  c_3, -c_3 ]
		*/

		mat = i*model->m_lclCB_dim;

		model->CB[mat]   = -c_1; model->CB[mat+1] =  c_2; model->CB[mat+2] =  c_1; model->CB[mat+3] =  c_2;
		model->CB[mat+4] =  c_1; model->CB[mat+5] = -c_2; model->CB[mat+6] = -c_1; model->CB[mat+7] = -c_2;

		model->CB[mat+8]  =  c_2; model->CB[mat+9]  = -c_1; model->CB[mat+10] = -c_2; model->CB[mat+11] = -c_1;
		model->CB[mat+12] = -c_2; model->CB[mat+13] =  c_1; model->CB[mat+14] =  c_2; model->CB[mat+15] =  c_1;

		model->CB[mat+16] = -c_3; model->CB[mat+17] = -c_3; model->CB[mat+18] = -c_3; model->CB[mat+19] =  c_3;
		model->CB[mat+20] =  c_3; model->CB[mat+21] =  c_3; model->CB[mat+22] =  c_3; model->CB[mat+23] = -c_3;
	}

	return;
}
//------------------------------------------------------------------------------
void assembleLocalMtxs_elastic_2D_PlaneStress(hmgModel_t *model){

	cudapcgVar_t E, v, coef;
	cudapcgVar_t c1, c2, c3, c4, c5, c6;

	unsigned int i, j, mat;
	//cudapcgVar_t * thisK;
	
	for (i=0;i<model->m_nmat;i++){

		E = model->props[i*2];
		v = model->props[i*2+1];
		
		coef = E/(1-(v*v));

    c1 = coef * (0.5 - (v/6));
    c2 = coef * (v/6);
    c3 = coef * ((v/12) - 0.25);
    c4 = coef * ((v/12) + 0.25);
    c5 = coef * ((3*v)*0.125 - 0.125);
    c6 = coef * (v*0.125 + 0.125);
    
    cudapcgVar_t lclK[64] = {  c1,  c6, -c4,  c5,  c3, -c6,  c2, -c5,
                               c6,  c1, -c5,  c2, -c6,  c3,  c5, -c4,
                              -c4, -c5,  c1, -c6,  c2,  c5,  c3,  c6,
                               c5,  c2, -c6,  c1, -c5, -c4,  c6,  c3,
                               c3, -c6,  c2, -c5,  c1,  c6, -c4,  c5,
                              -c6,  c3,  c5, -c4,  c6,  c1, -c5,  c2,
                               c2,  c5,  c3,  c6, -c4, -c5,  c1, -c6,
                              -c5, -c4,  c6,  c3,  c5,  c2, -c6,  c1 };
                              
    mat = i*64;
    for (j=0;j<64;j++){
     model->Mtxs[mat+j] = lclK[j];
    }
    
    c1 = coef*(0.5);
    c2 = coef*(0.5*v);
    c3 = coef*(0.25-(0.25*v));
    
    cudapcgVar_t lclCB[24] = { -c1, -c2,  c1, -c2,  c1,  c2, -c1,  c2,
                               -c2, -c1,  c2, -c1,  c2,  c1, -c2,  c1,
                               -c3, -c3, -c3,  c3,  c3,  c3,  c3, -c3 };

    mat = i*24;
		for (j=0;j<24;j++){
		 model->CB[mat+j] = lclCB[j];
	  }
	}

	return;
}
//------------------------------------------------------------------------------
void assembleRHS_elastic_2D(hmgModel_t *model){

	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;

	unsigned int i;
    #pragma omp parallel for
    for (i=0; i<model->m_ndof; i++)
        model->RHS[i] = 0.0;

	unsigned int e,n;
	cudapcgVar_t * thisK;

	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (i=0; i<dim_y; i++){
			e = (model->m_nx-2)*dim_y+i;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);

			n = e+1+(e/dim_y);
			model->RHS[model->node_dof_map[n]]   -= (thisK[2]+thisK[4])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[10]+thisK[12])*dim_x;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= (thisK[18]+thisK[20])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[26]+thisK[28])*dim_x;

			n -= 1;
			model->RHS[model->node_dof_map[n]]   -= (thisK[34]+thisK[36])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[42]+thisK[44])*dim_x;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= (thisK[50]+thisK[52])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[58]+thisK[60])*dim_x;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (i=0; i<dim_x; i++){
			e = i*dim_y;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);

			n = e+1+(e/dim_y);
			model->RHS[model->node_dof_map[n]]   -= (thisK[5]+thisK[7])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[13]+thisK[15])*dim_y;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= (thisK[21]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[29]+thisK[31])*dim_y;

			n -= 1;
			model->RHS[model->node_dof_map[n]]   -= (thisK[37]+thisK[39])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[45]+thisK[47])*dim_y;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= (thisK[53]+thisK[55])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[61]+thisK[63])*dim_y;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_XY){
		for (i=0; i<dim_x; i++){
			e = i*dim_y;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);

			n = e+1+(e/dim_y);
			model->RHS[model->node_dof_map[n]]   -= (thisK[4]+thisK[6])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[12]+thisK[14])*dim_y;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= (thisK[20]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[28]+thisK[30])*dim_y;

			n -= 1;
			model->RHS[model->node_dof_map[n]]   -= (thisK[36]+thisK[38])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[44]+thisK[46])*dim_y;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= (thisK[52]+thisK[54])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[60]+thisK[62])*dim_y;
		}
	}

	return;
}

//------------------------------------------------------------------------------
void updateC_elastic_2D(hmgModel_t *model, cudapcgVar_t * D){
	unsigned int n;
	unsigned int e;
	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	var C_i, C_j, C_k;
	var d;
	cudapcgVar_t * thisCB;

	unsigned int i,j,k;
	if (model->m_hmg_flag == HOMOGENIZE_X){
		i = 0; j = 3; k = 6;
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		i = 1; j = 4; k = 7;
	} else if (model->m_hmg_flag == HOMOGENIZE_XY){
		i = 2; j = 5; k = 8;
	}

	C_i = 0.0; C_j = 0.0; C_k = 0.0;

	if (model->m_hmg_flag == HOMOGENIZE_X){

		#pragma omp parallel for private(C_i,C_j,C_k,thisCB)
		for (e=model->m_nelem-model->m_ny+1;e<model->m_nelem;e++){

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 1 (right,bottom)
			C_i = thisCB[2]; C_j = thisCB[10]; C_k = thisCB[18];

			// node 2 (right,top)
			C_i += thisCB[4]; C_j += thisCB[12]; C_k += thisCB[20];

			C_i *= dim_x; C_j *= dim_x; C_k *= dim_x;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
			}
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_Y){

		#pragma omp parallel for private(C_i,C_j,C_k,thisCB)
		for (e=0;e<model->m_nelem;e+=dim_y){

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 2 (right,top)
			C_i = thisCB[5]; C_j = thisCB[13]; C_k = thisCB[21];

			// node 3 (left,top)
			C_i += thisCB[7]; C_j += thisCB[15]; C_k += thisCB[23];

			C_i *= dim_y; C_j *= dim_y; C_k *= dim_y;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
			}
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_XY){

		#pragma omp parallel for private(C_i,C_j,C_k,thisCB)
		for (e=0;e<model->m_nelem;e+=dim_y){

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 2 (right,top)
			C_i = thisCB[4]; C_j = thisCB[12]; C_k = thisCB[20];

			// node 3 (left,top)
			C_i += thisCB[6]; C_j += thisCB[14]; C_k += thisCB[22];

			C_i *= dim_y; C_j *= dim_y; C_k *= dim_y;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
			}
		}
	}

	#pragma omp parallel for private(C_i,C_j,C_k,d,thisCB,n)
	for (e=0;e<model->m_nelem;e++){

		thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

		// node 0 (left,bottom)
		n = e+1+(e/dim_y);
		d = D[model->node_dof_map[n]];
		C_i = thisCB[0]*d; C_j = thisCB[8]*d; C_k = thisCB[16]*d;
		d = D[model->node_dof_map[n]+1];
		C_i += thisCB[1]*d; C_j += thisCB[9]*d; C_k += thisCB[17]*d;

		// node 1 (right,bottom)
		n+=model->m_ny;
		d = D[model->node_dof_map[n]];
		C_i += thisCB[2]*d; C_j += thisCB[10]*d; C_k += thisCB[18]*d;
		d = D[model->node_dof_map[n]+1];
		C_i += thisCB[3]*d; C_j += thisCB[11]*d; C_k += thisCB[19]*d;

		// node 2 (right,top)
		n-=1;
		d = D[model->node_dof_map[n]];
		C_i += thisCB[4]*d; C_j += thisCB[12]*d; C_k += thisCB[20]*d;
		d = D[model->node_dof_map[n]+1];
		C_i += thisCB[5]*d; C_j += thisCB[13]*d; C_k += thisCB[21]*d;

		// node 3 (left,top)
		n-=model->m_ny;
		d = D[model->node_dof_map[n]];
		C_i += thisCB[6]*d; C_j += thisCB[14]*d; C_k += thisCB[22]*d;
		d = D[model->node_dof_map[n]+1];
		C_i += thisCB[7]*d; C_j += thisCB[15]*d; C_k += thisCB[23]*d;

		#pragma omp critical
		{
			model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
		}
	}

	model->C[i] /= model->m_nelem; model->C[j] /= model->m_nelem; model->C[k] /= model->m_nelem;

	return;
}
//------------------------------------------------------------------------------
void printC_elastic_2D(hmgModel_t *model, char *dest){
  if (dest==NULL){
	  printf("-------------------------------------------------------\n");
	  printf("Homogenized Constitutive Matrix (Elasticity):\n");
	  printf("  %.8e  ", model->C[0]); printf("%.8e  ", model->C[1]); printf("%.8e\n", model->C[2]);
	  printf("  %.8e  ", model->C[3]); printf("%.8e  ", model->C[4]); printf("%.8e\n", model->C[5]);
	  printf("  %.8e  ", model->C[6]); printf("%.8e  ", model->C[7]); printf("%.8e\n", model->C[8]);
	  printf("-------------------------------------------------------\n");
	} else {
	  sprintf(
      dest,
      "-------------------------------------------------------\n"\
      "Homogenized Constitutive Matrix (Elasticity):\n"\
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
