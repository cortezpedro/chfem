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

#include "femhmg_3D.h"
#include "femhmg_elastic_3D.h"

//------------------------------------------------------------------------------
logical initModel_elastic_3D(hmgModel_t *model){
	if ((model->m_nx-1) <= 0 || (model->m_ny-1) <= 0 || (model->m_nz-1) <= 0 || model->m_nmat <= 0){
	  printf("ERROR: Failed to initialize model due to bad input parameters.\nrows:%d\ncols:%d\nslices:%d\nmaterials:%d\n",(model->m_ny-1),(model->m_nx-1),(model->m_nz-1),model->m_nmat);
		return HMG_FALSE;
	}

	model->m_C_dim = 36;
	model->m_lclMtx_dim = 576;
	model->m_lclCB_dim = 144;

	model->m_nnodedof = 3;
	model->m_nelemdof = 24;
	model->m_nnode = model->m_nx * model->m_ny * model->m_nz;
	model->m_nelem = (model->m_nx-1) * (model->m_ny-1) * (model->m_nz-1);
	model->m_ndof = model->m_nelem*model->m_nnodedof;

	model->assembleLocalMtxs = assembleLocalMtxs_elastic_3D;
	model->assembleRHS = assembleRHS_elastic_3D;
	model->updateC = updateC_elastic_3D;
	model->printC = printC_elastic_3D;
	
	model->assembleNodeDofMap = assembleNodeDofMap_3D;
	model->assembleDofMaterialMap = assembleDofMaterialMap_3D;
	
	return HMG_TRUE;
}
//------------------------------------------------------------------------------
void assembleLocalMtxs_elastic_3D(hmgModel_t *model){

	cudapcgVar_t E, v;
	cudapcgVar_t c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12;

	unsigned int i, j, mat;
	for (i=0;i<model->m_nmat;i++){

		E = model->props[i*2];
		v = model->props[i*2+1];

		c_1 = (E*(4*v - 1))/(24*(2*v - 1)*(v + 1));
		c_2 = (E*(4*v - 1))/(48*(2*v - 1)*(v + 1));
		c_3 = E/(18*(2*v - 1)*(v + 1));
		c_4 = E/(24*(2*v - 1)*(v + 1));
		c_5 = E/(36*(2*v - 1)*(v + 1));
		c_6 = E/(48*(2*v - 1)*(v + 1));
		c_7 = (E*(3*v - 2))/(9*(2*v - 1)*(v + 1));
		c_8 = (E*(3*v - 1))/(36*(2*v - 1)*(v + 1));
		c_9 = (E*(3*v - 2))/(36*(2*v - 1)*(v + 1));
		c_10 = (E*(6*v - 5))/(72*(2*v - 1)*(v + 1));
		c_11 = E/(36*(2*v - 1)) + (5*E)/(72*(v + 1));
		c_12 = E/(72*(2*v - 1)) + (5*E)/(144*(v + 1));

		// Analytical solution for local K
		cudapcgVar_t lclK[] = { c_7,-c_4,c_4,c_3,-c_1, c_1,-c_10,c_4, c_2,-c_5,c_1, c_6,-c_5,-c_6,-c_1,-c_10,-c_2,-c_4,-c_9, c_6,-c_6,-c_8, c_2, -c_2,
		-c_4,c_7,c_4,c_1,-c_5, c_6,c_4,-c_10, c_2,-c_1,c_3, c_1,-c_6,-c_5,-c_1, c_2,-c_8, -c_2, c_6,-c_9,-c_6,-c_2,-c_10,-c_4,
		 c_4, c_4,c_7,-c_1, c_6,-c_5,-c_2,-c_2,-c_8, c_6,-c_1,-c_5,c_11,c_11, c_3,-c_4, c_12,-c_10,-c_6,-c_6,-c_9, c_12,-c_4,-c_10,
		 c_3, c_1,-c_1,c_7, c_4,-c_4,-c_5,-c_1,-c_6,-c_10,-c_4,-c_2,-c_10,c_2, c_4,-c_5,c_6,c_11,-c_8,-c_2, c_12,-c_9,-c_6, c_6,
		-c_1,-c_5,c_6,c_4,c_7, c_4,c_1,c_3, c_1,-c_4,-c_10, c_2,-c_2,-c_8, -c_2, c_6,-c_5,-c_1, c_2,-c_10,-c_4,-c_6,-c_9,-c_6,
		 c_1, c_6,-c_5,-c_4, c_4,c_7,-c_6,-c_1,-c_5, c_2,-c_2,-c_8,c_4, c_12,-c_10,-c_1,c_11, c_3,-c_2,-c_4,-c_10,c_6,-c_6,-c_9,
		-c_10, c_4,-c_2,-c_5, c_1,-c_6,c_7,-c_4,-c_4, c_3,-c_1,-c_1,-c_9,c_6, c_6,-c_8,c_2,c_12,-c_5,-c_6,c_11,-c_10,-c_2, c_4,
		 c_4,-c_10,-c_2,-c_1, c_3,-c_1,-c_4,c_7,-c_4, c_1,-c_5,-c_6,c_6,-c_9, c_6,-c_2,-c_10, c_4,-c_6,-c_5,c_11,c_2,-c_8,c_12,
		 c_2, c_2,-c_8,-c_6, c_1,-c_5,-c_4,-c_4,c_7, c_1,-c_6,-c_5,c_6,c_6,-c_9, -c_2,c_4,-c_10,-c_1,-c_1,c_3,c_4, -c_2,-c_10,
		-c_5,-c_1,c_6,-c_10,-c_4, c_2,c_3,c_1, c_1,c_7,c_4, c_4,-c_8,-c_2, -c_2,-c_9,-c_6,-c_6,-c_10, c_2,-c_4,-c_5, c_6,-c_1,
		 c_1, c_3,-c_1,-c_4,-c_10,-c_2,-c_1,-c_5,-c_6, c_4,c_7,-c_4,c_2,-c_10, c_4,-c_6,-c_9, c_6,-c_2,-c_8, c_12,c_6,-c_5,c_11,
		 c_6, c_1,-c_5,-c_2, c_2,-c_8,-c_1,-c_6,-c_5, c_4,-c_4,c_7, c_12,c_4,-c_10,-c_6,c_6,-c_9,-c_4, -c_2,-c_10,c_11,-c_1, c_3,
		-c_5,-c_6,c_11,-c_10,-c_2, c_4,-c_9,c_6, c_6,-c_8,c_2,c_12,c_7,-c_4,-c_4, c_3,-c_1,-c_1,-c_10, c_4,-c_2,-c_5, c_1,-c_6,
		-c_6,-c_5,c_11,c_2,-c_8,c_12,c_6,-c_9, c_6,-c_2,-c_10, c_4,-c_4,c_7,-c_4, c_1,-c_5,-c_6, c_4,-c_10,-c_2,-c_1, c_3,-c_1,
		-c_1,-c_1,c_3,c_4, -c_2,-c_10,c_6,c_6,-c_9, -c_2,c_4,-c_10,-c_4,-c_4,c_7, c_1,-c_6,-c_5, c_2, c_2,-c_8,-c_6, c_1,-c_5,
		-c_10, c_2,-c_4,-c_5, c_6,-c_1,-c_8,-c_2, -c_2,-c_9,-c_6,-c_6,c_3,c_1, c_1,c_7,c_4, c_4,-c_5,-c_1,c_6,-c_10,-c_4, c_2,
		-c_2,-c_8, c_12,c_6,-c_5,c_11,c_2,-c_10, c_4,-c_6,-c_9, c_6,-c_1,-c_5,-c_6, c_4,c_7,-c_4, c_1, c_3,-c_1,-c_4,-c_10,-c_2,
		-c_4, -c_2,-c_10,c_11,-c_1, c_3, c_12,c_4,-c_10,-c_6,c_6,-c_9,-c_1,-c_6,-c_5, c_4,-c_4,c_7, c_6, c_1,-c_5,-c_2, c_2,-c_8,
		-c_9, c_6,-c_6,-c_8, c_2, -c_2,-c_5,-c_6,-c_1,-c_10,-c_2,-c_4,-c_10,c_4, c_2,-c_5,c_1, c_6,c_7,-c_4,c_4,c_3,-c_1, c_1,
		 c_6,-c_9,-c_6,-c_2,-c_10,-c_4,-c_6,-c_5,-c_1, c_2,-c_8, -c_2,c_4,-c_10, c_2,-c_1,c_3, c_1,-c_4,c_7,c_4,c_1,-c_5, c_6,
		-c_6,-c_6,-c_9, c_12,-c_4,-c_10,c_11,c_11, c_3,-c_4, c_12,-c_10,-c_2,-c_2,-c_8, c_6,-c_1,-c_5, c_4, c_4,c_7,-c_1, c_6,-c_5,
		-c_8,-c_2, c_12,-c_9,-c_6, c_6,-c_10,c_2, c_4,-c_5,c_6,c_11,-c_5,-c_1,-c_6,-c_10,-c_4,-c_2, c_3, c_1,-c_1,c_7, c_4,-c_4,
		 c_2,-c_10,-c_4,-c_6,-c_9,-c_6,-c_2,-c_8, -c_2, c_6,-c_5,-c_1,c_1,c_3, c_1,-c_4,-c_10, c_2,-c_1,-c_5,c_6,c_4,c_7, c_4,
		 -c_2,-c_4,-c_10,c_6,-c_6,-c_9,c_4, c_12,-c_10,-c_1,c_11, c_3,-c_6,-c_1,-c_5, c_2,-c_2,-c_8, c_1, c_6,-c_5,-c_4, c_4,c_7 };


		 mat = i*model->m_lclMtx_dim;
		 #pragma omp parallel for
		 for (j=0;j<model->m_lclMtx_dim;j++){
			 model->Mtxs[mat+j] = lclK[j];
		 }

		 c_1=E/(8*(v+1));
		 c_2=(E*(v-1))/(4*(2*v-1)*(v+1));
		 c_3=(E*v)/(4*(2*v-1)*(v+1));

		 // Analytical solution for local model->CB
		 cudapcgVar_t lclCB[] = {-c_2,c_3,-c_3,c_2,c_3,-c_3,c_2,-c_3,-c_3,-c_2,-c_3,-c_3,-c_2,c_3,c_3,c_2,c_3,c_3,c_2,-c_3,c_3,-c_2,-c_3,c_3,
		 c_3,-c_2,-c_3,-c_3,-c_2,-c_3,-c_3,c_2,-c_3,c_3,c_2,-c_3,c_3,-c_2,c_3,-c_3,-c_2,c_3,-c_3,c_2,c_3,c_3,c_2,c_3,
		 c_3,c_3,c_2,-c_3,c_3,c_2,-c_3,-c_3,c_2,c_3,-c_3,c_2,c_3,c_3,-c_2,-c_3,c_3,-c_2,-c_3,-c_3,-c_2,c_3,-c_3,-c_2,
		 0.0,c_1,-c_1,0.0,c_1,-c_1,0.0,c_1,c_1,0.0,c_1,c_1,0.0,-c_1,-c_1,0.0,-c_1,-c_1,0.0,-c_1,c_1,0.0,-c_1,c_1,
		 c_1,0.0,-c_1,c_1,0.0,c_1,c_1,0.0,c_1,c_1,0.0,-c_1,-c_1,0.0,-c_1,-c_1,0.0,c_1,-c_1,0.0,c_1,-c_1,0.0,-c_1,
		 -c_1,-c_1,0.0,-c_1,c_1,0.0,c_1,c_1,0.0,c_1,-c_1,0.0,-c_1,-c_1,0.0,-c_1,c_1,0.0,c_1,c_1,0.0,c_1,-c_1,0.0 };

		 mat = i*model->m_lclCB_dim;
		 #pragma omp parallel for
		 for (j=0;j<model->m_lclCB_dim;j++){
			 model->CB[mat+j] = lclCB[j];
		 }
 	}

	return;
}
//------------------------------------------------------------------------------
void assembleRHS_elastic_3D(hmgModel_t *model){

	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	unsigned int dim_z = model->m_nz-1;
	unsigned int dim_xy = dim_x*dim_y;
	unsigned int dim_xz = dim_x*dim_z;
	unsigned int dim_yz = dim_y*dim_z;

	unsigned int i;
	#pragma omp parallel for
	for (i=0; i<model->m_ndof; i++)
		model->RHS[i] = 0.0;

	unsigned int e,n;
	cudapcgVar_t * thisK;

	/*
		ATTENTION: Zeros are not stored in local FEM matrices.
		"thisK" indexes consider that. No calculations with zeros are made.
	*/

	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (i=0; i<dim_yz; i++){
			e = (model->m_nx-2)*dim_y+i%dim_y+(i/dim_y)*dim_xy;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (i=0; i<dim_xz; i++){
			e = (i%dim_x)*dim_y+(i/dim_x)*dim_xy;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Z){
		for (i=0; i<dim_xy; i++){
			e = i;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_YZ){
		for (i=0; i<dim_xz; i++){
			e = (i%dim_x)*dim_y+(i/dim_x)*dim_xy;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_XZ){
		for (i=0; i<dim_xy; i++){
			e = i;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_XY){
		for (i=0; i<dim_xz; i++){
			e = (i%dim_x)*dim_y+(i/dim_x)*dim_xy;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= ( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= (thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= (thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;
		}
	}
	return;
}

//------------------------------------------------------------------------------
void updateC_elastic_3D(hmgModel_t *model, cudapcgVar_t * D){
	unsigned int e, ei, n;
	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	unsigned int dim_z = model->m_nz-1;
	unsigned int dim_xy = dim_x*dim_y;
	unsigned int dim_xz = dim_x*dim_z;
	unsigned int dim_yz = dim_y*dim_z;

	var C_i, C_j, C_k, C_x, C_y, C_z;
	var d;
	cudapcgVar_t * thisCB;

	unsigned int i,j,k,x,y,z;
	if (model->m_hmg_flag == HOMOGENIZE_X){
		i = 0; j = 6; k = 12; x = 18; y = 24; z = 30;
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		i = 1; j = 7; k = 13; x = 19; y = 25; z = 31;
	} else if (model->m_hmg_flag == HOMOGENIZE_Z){
		i = 2; j = 8; k = 14; x = 20; y = 26; z = 32;
	} else if (model->m_hmg_flag == HOMOGENIZE_YZ){
		i = 3; j = 9; k = 15; x = 21; y = 27; z = 33;
	} else if (model->m_hmg_flag == HOMOGENIZE_XZ){
		i = 4; j = 10; k = 16; x = 22; y = 28; z = 34;
	} else if (model->m_hmg_flag == HOMOGENIZE_XY){
		i = 5; j = 11; k = 17; x = 23; y = 29; z = 35;
	}

	//C_i = 0.0; C_j = 0.0; C_k = 0.0; C_x = 0.0; C_y = 0.0; C_z = 0.0;

	if (model->m_hmg_flag == HOMOGENIZE_X){

		#pragma omp parallel for private(e,C_i,C_j,C_k,C_x,C_y,C_z,thisCB)
		for (ei=0; ei<dim_yz; ei++){
			e = (model->m_nx-2)*dim_y+ei%dim_y+(ei/dim_y)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 1 (right,bottom,near)
			C_i = thisCB[3];  C_j = thisCB[27]; C_k = thisCB[51];
			C_x = thisCB[75]; C_y = thisCB[99]; C_z = thisCB[123];

			// node 2 (right,top,near)
			C_i += thisCB[6];  C_j += thisCB[30];  C_k += thisCB[54];
			C_x += thisCB[78]; C_y += thisCB[102]; C_z += thisCB[126];

			// node 5 (right,bottom,far)
			C_i += thisCB[15]; C_j += thisCB[39];  C_k += thisCB[63];
			C_x += thisCB[87]; C_y += thisCB[111]; C_z += thisCB[135];

			// node 6 (right,top,far)
			C_i += thisCB[18]; C_j += thisCB[42];  C_k += thisCB[66];
			C_x += thisCB[90]; C_y += thisCB[114]; C_z += thisCB[138];

			C_i *= dim_x; C_j *= dim_x; C_k *= dim_x;
			C_x *= dim_x; C_y *= dim_x; C_z *= dim_x;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
				model->C[x] += C_x; model->C[y] += C_y; model->C[z] += C_z;
			}
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_Y){

		#pragma omp parallel for private(e,C_i,C_j,C_k,C_x,C_y,C_z,thisCB)
		for (ei=0; ei<dim_xz; ei++){
			e = (ei%dim_x)*dim_y+(ei/dim_x)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 2 (right,top,near)
			C_i = thisCB[7];  C_j = thisCB[31];  C_k = thisCB[55];
			C_x = thisCB[79]; C_y = thisCB[103]; C_z = thisCB[127];

			// node 3 (left,top,near)
			C_i += thisCB[10]; C_j += thisCB[34];  C_k += thisCB[58];
			C_x += thisCB[82]; C_y += thisCB[106]; C_z += thisCB[130];

			// node 5 (right,top,far)
			C_i += thisCB[19]; C_j += thisCB[43];  C_k += thisCB[67];
			C_x += thisCB[91]; C_y += thisCB[115]; C_z += thisCB[139];

			// node 7 (left,top,far)
			C_i += thisCB[22]; C_j += thisCB[46];  C_k += thisCB[70];
			C_x += thisCB[94]; C_y += thisCB[118]; C_z += thisCB[142];

			C_i *= dim_y; C_j *= dim_y; C_k *= dim_y;
			C_x *= dim_y; C_y *= dim_y; C_z *= dim_y;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
				model->C[x] += C_x; model->C[y] += C_y; model->C[z] += C_z;
			}
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_Z){

		#pragma omp parallel for private(e,C_i,C_j,C_k,C_x,C_y,C_z,thisCB)
		for (ei=0; ei<dim_xy; ei++){
			e = ei;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 0 (left,bottom,near)
			C_i = thisCB[2];  C_j = thisCB[26]; C_k = thisCB[50];
			C_x = thisCB[74]; C_y = thisCB[98]; C_z = thisCB[122];

			// node 1 (right,bottom,near)
			C_i += thisCB[5];  C_j += thisCB[29];  C_k += thisCB[53];
			C_x += thisCB[77]; C_y += thisCB[101]; C_z += thisCB[125];

			// node 2 (right,top,near)
			C_i += thisCB[8];  C_j += thisCB[32];  C_k += thisCB[56];
			C_x += thisCB[80]; C_y += thisCB[104]; C_z += thisCB[128];

			// node 3 (left,top,near)
			C_i += thisCB[11]; C_j += thisCB[35];  C_k += thisCB[59];
			C_x += thisCB[83]; C_y += thisCB[107]; C_z += thisCB[131];

			C_i *= dim_z; C_j *= dim_z; C_k *= dim_z;
			C_x *= dim_z; C_y *= dim_z; C_z *= dim_z;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
				model->C[x] += C_x; model->C[y] += C_y; model->C[z] += C_z;
			}
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_YZ){

		#pragma omp parallel for private(e,C_i,C_j,C_k,C_x,C_y,C_z,thisCB)
		for (ei=0; ei<dim_xz; ei++){
			e = (ei%dim_x)*dim_y+(ei/dim_x)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 2 (right,top,near)
			C_i = thisCB[8];  C_j = thisCB[32];  C_k = thisCB[56];
			C_x = thisCB[80]; C_y = thisCB[104]; C_z = thisCB[128];

			// node 3 (left,top,near)
			C_i += thisCB[11]; C_j += thisCB[35];  C_k += thisCB[59];
			C_x += thisCB[83]; C_y += thisCB[107]; C_z += thisCB[131];

			// node 5 (right,top,far)
			C_i += thisCB[20]; C_j += thisCB[44];  C_k += thisCB[68];
			C_x += thisCB[92]; C_y += thisCB[116]; C_z += thisCB[140];

			// node 7 (left,top,far)
			C_i += thisCB[23]; C_j += thisCB[47];  C_k += thisCB[71];
			C_x += thisCB[95]; C_y += thisCB[119]; C_z += thisCB[143];

			C_i *= dim_y; C_j *= dim_y; C_k *= dim_y;
			C_x *= dim_y; C_y *= dim_y; C_z *= dim_y;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
				model->C[x] += C_x; model->C[y] += C_y; model->C[z] += C_z;
			}
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_XZ){

		#pragma omp parallel for private(e,C_i,C_j,C_k,C_x,C_y,C_z,thisCB)
		for (ei=0; ei<dim_xy; ei++){
			e = ei;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 0 (left,bottom,near)
			C_i = thisCB[0];  C_j = thisCB[24]; C_k = thisCB[48];
			C_x = thisCB[72]; C_y = thisCB[96]; C_z = thisCB[120];

			// node 1 (right,bottom,near)
			C_i += thisCB[3];  C_j += thisCB[27]; C_k += thisCB[51];
			C_x += thisCB[75]; C_y += thisCB[99]; C_z += thisCB[123];

			// node 2 (right,top,near)
			C_i += thisCB[6];  C_j += thisCB[30];  C_k += thisCB[54];
			C_x += thisCB[78]; C_y += thisCB[102]; C_z += thisCB[126];

			// node 3 (left,top,near)
			C_i += thisCB[9];  C_j += thisCB[33];  C_k += thisCB[57];
			C_x += thisCB[81]; C_y += thisCB[105]; C_z += thisCB[129];

			C_i *= dim_z; C_j *= dim_z; C_k *= dim_z;
			C_x *= dim_z; C_y *= dim_z; C_z *= dim_z;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
				model->C[x] += C_x; model->C[y] += C_y; model->C[z] += C_z;
			}
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_XY){

		#pragma omp parallel for private(e,C_i,C_j,C_k,C_x,C_y,C_z,thisCB)
		for (ei=0; ei<dim_xz; ei++){
			e = (ei%dim_x)*dim_y+(ei/dim_x)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 2 (right,top,near)
			C_i = thisCB[6];  C_j = thisCB[30];  C_k = thisCB[54];
			C_x = thisCB[78]; C_y = thisCB[102]; C_z = thisCB[126];

			// node 3 (left,top,near)
			C_i += thisCB[9];  C_j += thisCB[33];  C_k += thisCB[57];
			C_x += thisCB[81]; C_y += thisCB[105]; C_z += thisCB[129];

			// node 5 (right,top,far)
			C_i += thisCB[18]; C_j += thisCB[42];  C_k += thisCB[66];
			C_x += thisCB[90]; C_y += thisCB[114]; C_z += thisCB[138];

			// node 7 (left,top,far)
			C_i += thisCB[21]; C_j += thisCB[45];  C_k += thisCB[69];
			C_x += thisCB[93]; C_y += thisCB[117]; C_z += thisCB[141];

			C_i *= dim_y; C_j *= dim_y; C_k *= dim_y;
			C_x *= dim_y; C_y *= dim_y; C_z *= dim_y;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
				model->C[x] += C_x; model->C[y] += C_y; model->C[z] += C_z;
			}
		}
	}

	unsigned int ii;

	#pragma omp parallel for private(ii,C_i,C_j,C_k,C_x,C_y,C_z,d,thisCB,n)
	for (e=0;e<model->m_nelem;e++){

		thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

		C_i = 0.0; C_j = 0.0; C_k = 0.0; C_x = 0.0; C_y = 0.0; C_z = 0.0;

		// node 0 (left,bottom,near)
		n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		for (ii=0;ii<3;ii++){
			d = D[model->node_dof_map[n]+ii];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 1 (right,bottom,near)
		n+=model->m_ny;
		for (ii=3;ii<6;ii++){
			d = D[model->node_dof_map[n]+ii-3];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 2 (right,top,near)
		n-=1;
		for (ii=6;ii<9;ii++){
			d = D[model->node_dof_map[n]+ii-6];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 3 (left,top,near)
		n-=model->m_ny;
		for (ii=9;ii<12;ii++){
			d = D[model->node_dof_map[n]+ii-9];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 4 (left,bottom,far)
		n+=1+model->m_nx*model->m_ny;
		for (ii=12;ii<15;ii++){
			d = D[model->node_dof_map[n]+ii-12];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 5 (right,bottom,far)
		n+=model->m_ny;
		for (ii=15;ii<18;ii++){
			d = D[model->node_dof_map[n]+ii-15];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 6 (right,top,far)
		n-=1;
		for (ii=18;ii<21;ii++){
			d = D[model->node_dof_map[n]+ii-18];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 7 (left,top,far)
		n-=model->m_ny;
		for (ii=21;ii<24;ii++){
			d = D[model->node_dof_map[n]+ii-21];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		#pragma omp critical
		{
			model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
			model->C[x] += C_x; model->C[y] += C_y; model->C[z] += C_z;
		}
	}

	model->C[i] /= model->m_nelem; model->C[j] /= model->m_nelem; model->C[k] /= model->m_nelem;
	model->C[x] /= model->m_nelem; model->C[y] /= model->m_nelem; model->C[z] /= model->m_nelem;

	return;
}
//------------------------------------------------------------------------------
void printC_elastic_3D(hmgModel_t *model, char *dest){
	if (dest==NULL){
	  printf("-------------------------------------------------------\n");
	  printf("Homogenized Constitutive Matrix (Elasticity):\n");
	  for (unsigned int i = 0; i<model->m_C_dim; i++){
		  printf("  %.8e", model->C[i]);
		  if (!((i+1)%6))
			  printf("\n");
	  }
	  printf("-------------------------------------------------------\n");
	} else {
	  sprintf(
      dest,
      "-------------------------------------------------------\n"\
      "Homogenized Constitutive Matrix (Elasticity):\n"\
      "  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n"\
      "  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n"\
      "  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n"\
      "  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n"\
      "  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n"\
      "  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n"\
      "-------------------------------------------------------\n",
      model->C[0] , model->C[1] , model->C[2] , model->C[3] , model->C[4] , model->C[5],
      model->C[6] , model->C[7] , model->C[8] , model->C[9] , model->C[10], model->C[11],
      model->C[12], model->C[13], model->C[14], model->C[15], model->C[16], model->C[17],
      model->C[18], model->C[19], model->C[20], model->C[21], model->C[22], model->C[23],
      model->C[24], model->C[25], model->C[26], model->C[27], model->C[28], model->C[29],
      model->C[30], model->C[31], model->C[32], model->C[33], model->C[34], model->C[35]
    );
  }
  return;
}
//------------------------------------------------------------------------------
