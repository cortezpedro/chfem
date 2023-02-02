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
	if (model->sdfFile==NULL){
	  model->assembleRHS = assembleRHS_elastic_3D;
	  model->updateC = updateC_elastic_3D;
	  model->saveFields = saveFields_elastic_3D;
	} else {
	  model->assembleRHS = assembleRHS_elastic_3D_ScalarDensityField;
	  model->updateC = updateC_elastic_3D_ScalarDensityField;
	  model->saveFields = saveFields_elastic_3D_ScalarDensityField;
	}
	model->printC = printC_elastic_3D;

	model->assembleNodeDofMap = assembleNodeDofMap_3D;
	model->assembleDofIdMap = NULL;
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
	cudapcgVar_t * thisK=NULL;

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
void assembleRHS_elastic_3D_ScalarDensityField(hmgModel_t *model){

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
	cudapcgVar_t * thisK=NULL;
	cudapcgVar_t scl=1.0;

	/*
		ATTENTION: Zeros are not stored in local FEM matrices.
		"thisK" indexes consider that. No calculations with zeros are made.
	*/

	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (i=0; i<dim_yz; i++){
			e = (model->m_nx-2)*dim_y+i%dim_y+(i/dim_y)*dim_xy;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);
			
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[3]+ thisK[6]+thisK[15]+thisK[18])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[27]+thisK[30]+thisK[39]+thisK[42])*dim_x;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[51]+thisK[54]+thisK[63]+thisK[66])*dim_x;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (i=0; i<dim_xz; i++){
			e = (i%dim_x)*dim_y+(i/dim_x)*dim_xy;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);
			
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[7]+thisK[10]+thisK[19]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[31]+thisK[34]+thisK[43]+thisK[46])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[55]+thisK[58]+thisK[67]+thisK[70])*dim_y;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Z){
		for (i=0; i<dim_xy; i++){
			e = i;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);
			
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[2]+ thisK[5]+ thisK[8]+thisK[11])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[26]+thisK[29]+thisK[32]+thisK[35])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[50]+thisK[53]+thisK[56]+thisK[59])*dim_z;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_YZ){
		for (i=0; i<dim_xz; i++){
			e = (i%dim_x)*dim_y+(i/dim_x)*dim_xy;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);
			
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[8]+thisK[11]+thisK[20]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[32]+thisK[35]+thisK[44]+thisK[47])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[56]+thisK[59]+thisK[68]+thisK[71])*dim_y;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_XZ){
		for (i=0; i<dim_xy; i++){
			e = i;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);
			
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[0]+ thisK[3]+ thisK[6]+ thisK[9])*dim_z;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[24]+thisK[27]+thisK[30]+thisK[33])*dim_z;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[48]+thisK[51]+thisK[54]+thisK[57])*dim_z;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_XY){
		for (i=0; i<dim_xz; i++){
			e = (i%dim_x)*dim_y+(i/dim_x)*dim_xy;

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);
			
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+3*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+6*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+9*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n += 1+model->m_nx*model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+12*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n += model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+15*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n -= 1;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+18*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;

			n -= model->m_ny;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim+21*24]);
			model->RHS[model->node_dof_map[n]]   -= scl*( thisK[6]+ thisK[9]+thisK[18]+thisK[21])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[30]+thisK[33]+thisK[42]+thisK[45])*dim_y;
			model->RHS[model->node_dof_map[n]+2] -= scl*(thisK[54]+thisK[57]+thisK[66]+thisK[69])*dim_y;
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
	cudapcgVar_t * thisCB = NULL;

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
void updateC_elastic_3D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * D){
	unsigned int e, ei, n;
	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	unsigned int dim_z = model->m_nz-1;
	unsigned int dim_xy = dim_x*dim_y;
	unsigned int dim_xz = dim_x*dim_z;
	unsigned int dim_yz = dim_y*dim_z;

	var C_i, C_j, C_k, C_x, C_y, C_z;
	var d;
	cudapcgVar_t * thisCB = NULL;
	cudapcgVar_t scl=1.0;

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
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

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

			C_i *= scl*dim_x; C_j *= scl*dim_x; C_k *= scl*dim_x;
			C_x *= scl*dim_x; C_y *= scl*dim_x; C_z *= scl*dim_x;

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
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

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

			C_i *= scl*dim_y; C_j *= scl*dim_y; C_k *= scl*dim_y;
			C_x *= scl*dim_y; C_y *= scl*dim_y; C_z *= scl*dim_y;

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
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

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

			C_i *= scl*dim_z; C_j *= scl*dim_z; C_k *= scl*dim_z;
			C_x *= scl*dim_z; C_y *= scl*dim_z; C_z *= scl*dim_z;

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
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

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

			C_i *= scl*dim_y; C_j *= scl*dim_y; C_k *= scl*dim_y;
			C_x *= scl*dim_y; C_y *= scl*dim_y; C_z *= scl*dim_y;

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
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

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

			C_i *= scl*dim_z; C_j *= scl*dim_z; C_k *= scl*dim_z;
			C_x *= scl*dim_z; C_y *= scl*dim_z; C_z *= scl*dim_z;

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
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

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

			C_i *= scl*dim_y; C_j *= scl*dim_y; C_k *= scl*dim_y;
			C_x *= scl*dim_y; C_y *= scl*dim_y; C_z *= scl*dim_y;

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
		scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

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
		
		C_i *= scl; C_j *= scl; C_k *= scl;
	  C_x *= scl; C_y *= scl; C_z *= scl;

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
void stressFromDispl_elastic_3D(var *s, var *d, var E, var v, var x, var y, var z){

	var coeff = E/ ((1.0+v)*(1.0-2.0*v));
	var C[36] = { coeff*(1.0-v), coeff*v, coeff*v, 0.0, 0.0, 0.0,
						 	  coeff*v, coeff*(1.0-v), coeff*v, 0.0, 0.0, 0.0,
								coeff*v, coeff*v, coeff*(1.0-v), 0.0, 0.0, 0.0,
							  0.0, 0.0, 0.0, coeff*0.5*(1.0-2.0*v), 0.0, 0.0,
							  0.0, 0.0, 0.0, 0.0, coeff*0.5*(1.0-2.0*v), 0.0,
							  0.0, 0.0, 0.0, 0.0, 0.0, coeff*0.5*(1.0-2.0*v) };

	var x1 = (1.0-x), y1 = (1.0-y), z1 = (1.0-z);
	var B[144] = { -y1*z, 0.0, 0.0, y1*z, 0.0, 0.0, y*z, 0.0, 0.0, -y*z, 0.0, 0.0, -y1*z1, 0.0, 0.0, y1*z1, 0.0, 0.0, y*z1, 0.0, 0.0, -y*z1, 0.0, 0.0,
	               0.0, -x1*z, 0.0, 0.0, -x*z, 0.0, 0.0, x*z, 0.0, 0.0, x1*z, 0.0, 0.0, -x1*z1, 0.0, 0.0, -x*z1, 0.0, 0.0, x*z1, 0.0, 0.0, x1*z1, 0.0,
								 0.0, 0.0, x1*y1, 0.0, 0.0, x*y1, 0.0, 0.0, x*y, 0.0, 0.0, x1*y, 0.0, 0.0, -x1*y1, 0.0, 0.0, -x*y1, 0.0, 0.0, -x*y, 0.0, 0.0, -x1*y,
								 0.0, x1*y1, -x1*z, 0.0, x*y1, -x*z, 0.0, x*y, x*z, 0.0, x1*y, x1*z, 0.0, -x1*y1, -x1*z1, 0.0, -x*y1, -x*z1, 0.0, -x*y, x*z1, 0.0, -x1*y, x1*z1,
								 x1*y1, 0.0, -y1*z, x*y1, 0.0, y1*z, x*y, 0.0, y*z, x1*y, 0.0, -y*z, -x1*y1, 0.0, -y1*z1, -x*y1, 0.0, y1*z1, -x*y, 0.0, y*z1, -x1*y, 0.0, -y*z1,
								 -x1*z, -y1*z, 0.0, -x*z, y1*z, 0.0, x*z, y*z, 0.0, x1*z, -y*z, 0.0, -x1*z1, -y1*z1, 0.0, -x*z1, y1*z1, 0.0, x*z1, y*z1, 0.0, x1*z1, -y*z1, 0.0 };

	var c_x, c_y, c_z, c_yz, c_xz, c_xy;
	for (unsigned int j=0; j<6; j++){
		c_x  = 0.0; c_y  = 0.0; c_z  = 0.0;
		c_yz = 0.0; c_xz = 0.0; c_xy = 0.0;
		for (unsigned int i=0; i<24; i++){
			c_x  +=     B[i] * d[i];
			c_y  +=  B[i+24] * d[i];
			c_z  +=  B[i+48] * d[i];
			c_yz +=  B[i+72] * d[i];
			c_xz +=  B[i+96] * d[i];
			c_xy += B[i+120] * d[i];
		}
		s[j] = C[j*6]*c_x + C[j*6+1]*c_y + C[j*6+2]*c_z + C[j*6+3]*c_yz + C[j*6+4]*c_xz + C[j*6+5]*c_xy;
	}

	return;
}
//------------------------------------------------------------------------------
void saveFields_elastic_3D(hmgModel_t *model, cudapcgVar_t * D){

  cudapcgVar_t * S = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*(model->m_ndof/3)*6);

  #pragma omp parallel for
  for (unsigned int i=0; i<(model->m_ndof/3)*6; i++){
    S[i] = 0.0;
  }

  unsigned int rows = model->m_ny-1;
  unsigned int cols = model->m_nx-1;
	unsigned int lays = model->m_nz-1;
  unsigned int rowscols = rows*cols;

  //cudapcgVar_t * thisCB;
  //var d, C_i, C_j, C_k, C_x, C_y, C_z;
  var E, v;
  unsigned int n,ii,dof;

  var * local_S = (var *)malloc(sizeof(var)*6);
  var * local_d = (var *)malloc(sizeof(var)*24);

  for (unsigned int e=0;e<model->m_nelem;e++){

		E = model->props[2*model->elem_material_map[e]];
		v = model->props[2*model->elem_material_map[e]+1];

		// node 0 (left,bottom,near)
		n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
		dof = model->node_dof_map[n];
		local_d[0] = (var) D[dof];
		local_d[1] = (var) D[dof+1];
		local_d[2] = (var) D[dof+2];

		// node 1 (right,bottom,near)
		n+=model->m_ny;
		dof = model->node_dof_map[n];
		local_d[3] = (var) D[dof];
		local_d[4] = (var) D[dof+1];
		local_d[5] = (var) D[dof+2];

		// node 2 (right,top,near)
		n-=1;
		dof = model->node_dof_map[n];
		local_d[6] = (var) D[dof];
		local_d[7] = (var) D[dof+1];
		local_d[8] = (var) D[dof+2];

		// node 3 (left,top,near)
		n-=model->m_ny;
		dof = model->node_dof_map[n];
		local_d[9]  = (var) D[dof];
		local_d[10] = (var) D[dof+1];
		local_d[11] = (var) D[dof+2];

		// node 4 (left,bottom,far)
		n+=1+model->m_nx*model->m_ny;
		dof = model->node_dof_map[n];
		local_d[12] = (var) D[dof];
		local_d[13] = (var) D[dof+1];
		local_d[14] = (var) D[dof+2];

		// node 5 (right,bottom,far)
		n+=model->m_ny;
		dof = model->node_dof_map[n];
		local_d[15] = (var) D[dof];
		local_d[16] = (var) D[dof+1];
		local_d[17] = (var) D[dof+2];

		// node 6 (right,top,far)
		n-=1;
		dof = model->node_dof_map[n];
		local_d[18] = (var) D[dof];
		local_d[19] = (var) D[dof+1];
		local_d[20] = (var) D[dof+2];

		// node 7 (left,top,far)
		n-=model->m_ny;
		dof = model->node_dof_map[n];
		local_d[21] = (var) D[dof];
		local_d[22] = (var) D[dof+1];
		local_d[23] = (var) D[dof+2];

		dof /= 3;
		stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,0.0);
		S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
		S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
		S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
		S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
		S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
		S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

		// node 0 (left,bottom,near)
		n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
		dof = model->node_dof_map[n]/3;
		stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,1.0);
		S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
		S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
		S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
		S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
		S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
		S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

		// node 1 (right,bottom,near)
		n+=model->m_ny;
		dof = model->node_dof_map[n]/3;
		stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,1.0);
		S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
		S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
		S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
		S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
		S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
		S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

		// node 2 (right,top,near)
		n-=1;
		dof = model->node_dof_map[n]/3;
		stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,1.0);
		S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
		S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
		S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
		S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
		S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
		S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

		// node 3 (left,top,near)
		n-=model->m_ny;
		dof = model->node_dof_map[n]/3;
		stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,1.0);
		S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
		S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
		S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
		S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
		S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
		S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

		// node 4 (left,bottom,far)
		n+=1+model->m_nx*model->m_ny;
		dof = model->node_dof_map[n]/3;
		stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,0.0);
		S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
		S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
		S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
		S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
		S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
		S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

		// node 5 (right,bottom,far)
		n+=model->m_ny;
		dof = model->node_dof_map[n]/3;
		stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,0.0);
		S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
		S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
		S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
		S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
		S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
		S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

		// node 6 (right,top,far)
		n-=1;
		dof = model->node_dof_map[n]/3;
		stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,0.0);
		S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
		S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
		S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
		S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
		S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
		S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

	}

	// Compensate for periodic borders
	unsigned int e;
	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (unsigned int l=0; l<lays; l++){
			for (unsigned int r=0; r<rows; r++){
				e = r + (rowscols-rows) + l*rowscols;

				E = model->props[2*model->elem_material_map[e]];
				v = model->props[2*model->elem_material_map[e]+1];

				for (unsigned int i=0; i<24; i++){
					local_d[i] = 0.0;
				}
				local_d[3]  = (var) cols;
				local_d[6]  = (var) cols;
				local_d[15] = (var) cols;
				local_d[18] = (var) cols;

				// node 0 (left,bottom,near)
				n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 3 (left,top,near)
				n-=1;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 4 (left,bottom,far)
				n+=1+model->m_nx*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 7 (left,top,far)
				n-=1;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				local_d[0]  = -((var) cols); local_d[3]  = 0.0;
				local_d[9]  = -((var) cols); local_d[6]  = 0.0;
				local_d[12] = -((var) cols); local_d[15] = 0.0;
				local_d[21] = -((var) cols); local_d[18] = 0.0;

				// node 1 (right,bottom,near)
				n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny + model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 2 (right,top,near)
				n-=1;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 5 (right,bottom,far)
				n+=1+model->m_nx*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 6 (right,top,far)
				n-=1;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];
			}
		}

  } else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (unsigned int l=0; l<lays; l++){
			for (unsigned int c=0; c<cols; c++){
				e = c*rows + l*rowscols;

				E = model->props[2*model->elem_material_map[e]];
				v = model->props[2*model->elem_material_map[e]+1];

				for (unsigned int i=0; i<24; i++){
					local_d[i] = 0.0;
				}
				local_d[7]  = (var) rows;
				local_d[10] = (var) rows;
				local_d[19] = (var) rows;
				local_d[22] = (var) rows;

				// node 0 (left,bottom,near)
				n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 1 (right,bottom,near)
				n+=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 4 (left,bottom,far)
				n+=(model->m_nx-1)*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 5 (right,bottom,far)
				n+=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				local_d[1]  = -((var) rows); local_d[7]  = 0.0;
				local_d[4]  = -((var) rows); local_d[10] = 0.0;
				local_d[13] = -((var) rows); local_d[19] = 0.0;
				local_d[16] = -((var) rows); local_d[22] = 0.0;

				// node 2 (right,top,near)
				n -= 1+model->m_nx*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 3 (left,top,near)
				n-=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 6 (right,top,far)
				n+=(model->m_nx+1)*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 7 (left,top,far)
				n-=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];
			}
		}
  }  else if (model->m_hmg_flag == HOMOGENIZE_Z){
		for (e=0; e<rowscols; e++){

			E = model->props[2*model->elem_material_map[e]];
			v = model->props[2*model->elem_material_map[e]+1];

			for (unsigned int i=0; i<24; i++){
				local_d[i] = 0.0;
			}
			local_d[2]  = (var) lays;
			local_d[5]  = (var) lays;
			local_d[8]  = (var) lays;
			local_d[11] = (var) lays;

			// node 4 (left,bottom,far)
			n = 1+(e%rowscols)+((e%rowscols)/rows)+(1+e/rowscols)*model->m_nx*model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,0.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 5 (right,bottom,far)
			n+=model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,0.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 6 (right,top,far)
			n-=1;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,0.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 7 (left,top,far)
			n-=model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,0.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			local_d[14] = -((var) lays); local_d[2]  = 0.0;
			local_d[17] = -((var) lays); local_d[5]  = 0.0;
			local_d[20] = -((var) lays); local_d[8]  = 0.0;
			local_d[23] = -((var) lays); local_d[11] = 0.0;

			// node 0 (left,bottom,near)
			n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,1.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 1 (right,bottom,near)
			n+=model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,1.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 2 (right,top,near)
			n-=1;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,1.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 3 (left,top,near)
			n-=model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,1.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];
		}
  } else if (model->m_hmg_flag == HOMOGENIZE_YZ){
		for (unsigned int l=0; l<lays; l++){
			for (unsigned int c=0; c<cols; c++){
				e = c*rows + l*rowscols;

				E = model->props[2*model->elem_material_map[e]];
				v = model->props[2*model->elem_material_map[e]+1];

				for (unsigned int i=0; i<24; i++){
					local_d[i] = 0.0;
				}
				local_d[8]  = (var) rows;
				local_d[11] = (var) rows;
				local_d[20] = (var) rows;
				local_d[23] = (var) rows;

				// node 0 (left,bottom,near)
				n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 1 (right,bottom,near)
				n+=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 4 (left,bottom,far)
				n+=(model->m_nx-1)*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 5 (right,bottom,far)
				n+=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				local_d[2]  = -((var) rows); local_d[8]  = 0.0;
				local_d[5]  = -((var) rows); local_d[11] = 0.0;
				local_d[14] = -((var) rows); local_d[20] = 0.0;
				local_d[17] = -((var) rows); local_d[23] = 0.0;

				// node 2 (right,top,near)
				n -= 1+model->m_nx*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 3 (left,top,near)
				n-=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 6 (right,top,far)
				n+=(model->m_nx+1)*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 7 (left,top,far)
				n-=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];
			}
		}
  }  else if (model->m_hmg_flag == HOMOGENIZE_XZ){
		for (e=0; e<rowscols; e++){

			E = model->props[2*model->elem_material_map[e]];
			v = model->props[2*model->elem_material_map[e]+1];

			for (unsigned int i=0; i<24; i++){
				local_d[i] = 0.0;
			}
			local_d[0] = (var) lays;
			local_d[3] = (var) lays;
			local_d[6] = (var) lays;
			local_d[9] = (var) lays;

			// node 4 (left,bottom,far)
			n = 1+(e%rowscols)+((e%rowscols)/rows)+(1+e/rowscols)*model->m_nx*model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,0.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 5 (right,bottom,far)
			n+=model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,0.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 6 (right,top,far)
			n-=1;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,0.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 7 (left,top,far)
			n-=model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,0.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			local_d[12] = -((var) lays); local_d[0] = 0.0;
			local_d[15] = -((var) lays); local_d[3] = 0.0;
			local_d[18] = -((var) lays); local_d[6] = 0.0;
			local_d[21] = -((var) lays); local_d[9] = 0.0;

			// node 0 (left,bottom,near)
			n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,1.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 1 (right,bottom,near)
			n+=model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,1.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 2 (right,top,near)
			n-=1;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,1.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

			// node 3 (left,top,near)
			n-=model->m_ny;
			dof = model->node_dof_map[n]/3;
			stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,1.0);
			S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
			S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
			S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
			S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
			S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
			S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];
		}
  } else if (model->m_hmg_flag == HOMOGENIZE_XY){
		for (unsigned int l=0; l<lays; l++){
			for (unsigned int c=0; c<cols; c++){
				e = c*rows + l*rowscols;

				E = model->props[2*model->elem_material_map[e]];
				v = model->props[2*model->elem_material_map[e]+1];

				for (unsigned int i=0; i<24; i++){
					local_d[i] = 0.0;
				}
				local_d[6]  = (var) rows;
				local_d[9]  = (var) rows;
				local_d[18] = (var) rows;
				local_d[21] = (var) rows;

				// node 0 (left,bottom,near)
				n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 1 (right,bottom,near)
				n+=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 4 (left,bottom,far)
				n+=(model->m_nx-1)*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,0.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 5 (right,bottom,far)
				n+=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,0.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				local_d[0]  = -((var) rows); local_d[6]  = 0.0;
				local_d[3]  = -((var) rows); local_d[9]  = 0.0;
				local_d[12] = -((var) rows); local_d[18] = 0.0;
				local_d[15] = -((var) rows); local_d[21] = 0.0;

				// node 2 (right,top,near)
				n -= 1+model->m_nx*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 3 (left,top,near)
				n-=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,1.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 6 (right,top,far)
				n+=(model->m_nx+1)*model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,1.0,1.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];

				// node 7 (left,top,far)
				n-=model->m_ny;
				dof = model->node_dof_map[n]/3;
				stressFromDispl_elastic_3D(local_S,local_d,E,v,0.0,1.0,0.0);
				S[6*dof]   += (cudapcgVar_t) 0.125*local_S[0];
				S[6*dof+1] += (cudapcgVar_t) 0.125*local_S[1];
				S[6*dof+2] += (cudapcgVar_t) 0.125*local_S[2];
				S[6*dof+3] += (cudapcgVar_t) 0.125*local_S[3];
				S[6*dof+4] += (cudapcgVar_t) 0.125*local_S[4];
				S[6*dof+5] += (cudapcgVar_t) 0.125*local_S[5];
			}
		}
  }

  // Save arrays to binary files
  char str_buffer[1024];
  sprintf(str_buffer,"%s_displacement_%d.bin",model->neutralFile_noExt,model->m_hmg_flag);
  FILE * file = fopen(str_buffer,"wb");
  if (file)
    fwrite(D,sizeof(cudapcgVar_t)*model->m_ndof,1,file);
  fclose(file);

  sprintf(str_buffer,"%s_stress_%d.bin",model->neutralFile_noExt,model->m_hmg_flag);
  file = fopen(str_buffer,"wb");
  if (file)
    fwrite(S,sizeof(cudapcgVar_t)*(model->m_ndof/3)*6,1,file);
  fclose(file);

  free(local_S);
  free(local_d);
  free(S);

  return;
}
//------------------------------------------------------------------------------
void saveFields_elastic_3D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * D){
  printf("WARNING: Field exportation not supported for scalar field input (.bin) yet.\n");
  return;
}
//------------------------------------------------------------------------------
