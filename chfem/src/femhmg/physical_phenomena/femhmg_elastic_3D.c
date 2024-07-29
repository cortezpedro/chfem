#include "femhmg_3D.h"
#include "femhmg_elastic_3D.h"

//------------------------------------------------------------------------------
void strainFromDispl_elastic_3D(var *s, var *d, var x, var y, var z);
//------------------------------------------------------------------------------
void stressFromDispl_elastic_3D(var *s, var *d, var E, var v, var x, var y, var z);
//------------------------------------------------------------------------------
void stressFromAlpha_elastic_3D(var *s, var *a, var E, var v, var x, var y, var z);
//------------------------------------------------------------------------------
void forceFromStrain_elastic_3D(var *f, var *a, var E, var v, var x, var y, var z);
//------------------------------------------------------------------------------

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
	model->m_nporeelem = model->m_nelem; // pore elems comprehend the pore space, the domain. as mesh is monolithic, porelem=elem.
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
	}  else if (model->m_hmg_flag == HOMOGENIZE_THERMAL_EXPANSION){
	  var alpha[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
	  var local_f[24] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
	  for (uint32_t e=0; e < model->m_nelem; e++){
	    alpha[0] = model->alpha[model->elem_material_map[e]];
	    alpha[1] = alpha[0];
		alpha[2] = alpha[0];
	    alpha[3] = 0.0;
		alpha[4] = 0.0;
		alpha[5] = 0.0;
	    forceFromStrain_elastic_3D(&local_f[0],&alpha[0],model->props[2*model->elem_material_map[e]],model->props[2*model->elem_material_map[e]+1],0.5,0.5,0.5);
	    n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		model->RHS[model->node_dof_map[n]]   += local_f[0];
		model->RHS[model->node_dof_map[n]+1] += local_f[1];
		model->RHS[model->node_dof_map[n]+2] += local_f[2];
		n += model->m_ny;
		model->RHS[model->node_dof_map[n]]   += local_f[3];
		model->RHS[model->node_dof_map[n]+1] += local_f[4];
		model->RHS[model->node_dof_map[n]+2] += local_f[5];
		n -= 1;
		model->RHS[model->node_dof_map[n]]   += local_f[6];
		model->RHS[model->node_dof_map[n]+1] += local_f[7];
		model->RHS[model->node_dof_map[n]+2] += local_f[8];
		n -= model->m_ny;
		model->RHS[model->node_dof_map[n]]   += local_f[9];
		model->RHS[model->node_dof_map[n]+1] += local_f[10];
		model->RHS[model->node_dof_map[n]+2] += local_f[11];
		n += 1+model->m_nx*model->m_ny;
		model->RHS[model->node_dof_map[n]]   += local_f[12];
		model->RHS[model->node_dof_map[n]+1] += local_f[13];
		model->RHS[model->node_dof_map[n]+2] += local_f[14];
		n += model->m_ny;
		model->RHS[model->node_dof_map[n]]   += local_f[15];
		model->RHS[model->node_dof_map[n]+1] += local_f[16];
		model->RHS[model->node_dof_map[n]+2] += local_f[17];
		n -= 1;
		model->RHS[model->node_dof_map[n]]   += local_f[18];
		model->RHS[model->node_dof_map[n]+1] += local_f[19];
		model->RHS[model->node_dof_map[n]+2] += local_f[20];
		n -= model->m_ny;
		model->RHS[model->node_dof_map[n]]   += local_f[21];
		model->RHS[model->node_dof_map[n]+1] += local_f[22];
		model->RHS[model->node_dof_map[n]+2] += local_f[23];
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
void updateC_elastic_3D_thermal_expansion(hmgModel_t *model, cudapcgVar_t * D){
    unsigned int e, n, ii;
	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	unsigned int dim_z = model->m_nz-1;
	unsigned int dim_xy = dim_x*dim_y;
	unsigned int dim_xz = dim_x*dim_z;
	unsigned int dim_yz = dim_y*dim_z;
	var d;
	cudapcgVar_t * thisCB=NULL;

	var stress[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
	var strain[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
	var local_d[24] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
	var local_stress[6]={0.0,0.0,0.0,0.0,0.0,0.0};
	var local_strain[6]={0.0,0.0,0.0,0.0,0.0,0.0};
	var local_alpha[6] = {0.0,0.0,0.0,0.0,0.0,0.0};

  var C_i=0.0, C_j=0.0, C_k=0.0, C_x=0.0, C_y=0.0, C_z=0.0;
  
  // Compute effective stress (similar to updateC)
	#pragma omp parallel for private(ii,thisCB,d,n) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
	for (e=0;e<model->m_nelem;e++){

		thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

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
	}
	stress[0] = C_i;
	stress[1] = C_j;
	stress[2] = C_k;
	stress[3] = C_x;
	stress[4] = C_y;
	stress[5] = C_z;

	// Compute effective strain and add thermal expansion contribution to effective stress
	for (e=0;e<model->m_nelem;e++){
		n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		local_d[0] = D[model->node_dof_map[n]];
		local_d[1] = D[model->node_dof_map[n]+1];
		local_d[2] = D[model->node_dof_map[n]+2];
		n+=model->m_ny;
		local_d[3] = D[model->node_dof_map[n]];
		local_d[4] = D[model->node_dof_map[n]+1];
		local_d[5] = D[model->node_dof_map[n]+2];
		n-=1;
		local_d[6] = D[model->node_dof_map[n]];
		local_d[7] = D[model->node_dof_map[n]+1];
		local_d[8] = D[model->node_dof_map[n]+2];
		n-=model->m_ny;
		local_d[9] = D[model->node_dof_map[n]];
		local_d[10] = D[model->node_dof_map[n]+1];
		local_d[11] = D[model->node_dof_map[n]+2];
		n+=1+model->m_nx*model->m_ny;
		local_d[12] = D[model->node_dof_map[n]];
		local_d[13] = D[model->node_dof_map[n]+1];
		local_d[14] = D[model->node_dof_map[n]+2];
		n+=model->m_ny;
		local_d[15] = D[model->node_dof_map[n]];
		local_d[16] = D[model->node_dof_map[n]+1];
		local_d[17] = D[model->node_dof_map[n]+2];
		n-=1;
		local_d[18] = D[model->node_dof_map[n]];
		local_d[19] = D[model->node_dof_map[n]+1];
		local_d[20] = D[model->node_dof_map[n]+2];
		n-=model->m_ny;
		local_d[21] = D[model->node_dof_map[n]];
		local_d[22] = D[model->node_dof_map[n]+1];
		local_d[23] = D[model->node_dof_map[n]+2];

		local_strain[0] = 0.0;
		local_strain[1] = 0.0;
		local_strain[2] = 0.0;
		local_strain[3] = 0.0;
		local_strain[4] = 0.0;
		local_strain[5] = 0.0;

		strainFromDispl_elastic_3D(&local_strain[0],&local_d[0],0.5,0.5,0.5);
		strain[0] += local_strain[0];
		strain[1] += local_strain[1];
		strain[2] += local_strain[2];
		strain[3] += local_strain[3];
		strain[4] += local_strain[4];
		strain[5] += local_strain[5];
		
		local_alpha[0] = model->alpha[model->elem_material_map[e]];
		local_alpha[1] = local_alpha[0];
		local_alpha[2] = local_alpha[0];
		local_alpha[3] = 0.0;
		local_alpha[4] = 0.0;
		local_alpha[5] = 0.0;
		
		local_stress[0] = 0.0;
		local_stress[1] = 0.0;
		local_stress[2] = 0.0;
		local_stress[3] = 0.0;
		local_stress[4] = 0.0;
		local_stress[5] = 0.0;
		
		stressFromAlpha_elastic_3D(&local_stress[0],&local_alpha[0],model->props[2*model->elem_material_map[e]],model->props[2*model->elem_material_map[e]+1],0.5,0.5,0.5);
		stress[0] -= local_stress[0];
		stress[1] -= local_stress[1];
		stress[2] -= local_stress[2];
		stress[3] -= local_stress[3];
		stress[4] -= local_stress[4];
		stress[5] -= local_stress[5];
	}
	for (unsigned int k=0; k<6; k++){
		strain[k] /= (double) model->m_nelem;
		stress[k] /= (double) model->m_nelem;
	}

	var beta[6];
	beta[0] = -stress[0] +  model->C[0]*strain[0] +  model->C[1]*strain[1] +  model->C[2]*strain[2] +  model->C[3]*strain[3] +  model->C[4]*strain[4] +  model->C[5]*strain[5];
	beta[1] = -stress[1] +  model->C[6]*strain[0] +  model->C[7]*strain[1] +  model->C[8]*strain[2] +  model->C[9]*strain[3] + model->C[10]*strain[4] + model->C[11]*strain[5];
	beta[2] = -stress[2] + model->C[12]*strain[0] + model->C[13]*strain[1] + model->C[14]*strain[2] + model->C[15]*strain[3] + model->C[16]*strain[4] + model->C[17]*strain[5];
	beta[3] = -stress[3] + model->C[18]*strain[0] + model->C[19]*strain[1] + model->C[20]*strain[2] + model->C[21]*strain[3] + model->C[22]*strain[4] + model->C[23]*strain[5];
	beta[4] = -stress[4] + model->C[24]*strain[0] + model->C[25]*strain[1] + model->C[26]*strain[2] + model->C[27]*strain[3] + model->C[28]*strain[4] + model->C[29]*strain[5];
	beta[5] = -stress[5] + model->C[30]*strain[0] + model->C[31]*strain[1] + model->C[32]*strain[2] + model->C[33]*strain[3] + model->C[34]*strain[4] + model->C[35]*strain[5];

	// Solve 6x6 system with simple hardcoded gauss elim: alpha = C^-1 *beta
	var alpha[6], scl;
	var M[36];
	for (unsigned int j=0; j<36; j++) M[j] = model->C[j];

	for (unsigned int i=0; i<5; i++){
		for (unsigned int j=i+1; j<6; j++){
			scl = M[j*6+i]/M[7*i];
			for (unsigned int k=i+1; k<6; k++) M[j*6+k] -= M[i*6+k]*scl;
			beta[j] -= beta[i]*scl;
			M[j*6+i] = 0.0;
		}
	}
	alpha[5] = beta[5]/M[35];
	for (int i=4; i>=0; i--){
		alpha[i] = beta[i];
		for (int j=5; j>i; j--) alpha[i] -= M[6*i+j]*alpha[j];
		alpha[i] /= M[7*i];
	}
	
	model->thermal_expansion[0] = alpha[0];
	model->thermal_expansion[1] = alpha[1];
	model->thermal_expansion[2] = alpha[2];
	model->thermal_expansion[3] = alpha[3];
	model->thermal_expansion[4] = alpha[4];
	model->thermal_expansion[5] = alpha[5];

  return;
}
//------------------------------------------------------------------------------
void updateC_elastic_3D(hmgModel_t *model, cudapcgVar_t * D){

	if (model->m_hmg_flag == HOMOGENIZE_THERMAL_EXPANSION){
		updateC_elastic_3D_thermal_expansion(model,D);
		return;
	}

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

	C_i = 0.0; C_j = 0.0; C_k = 0.0; C_x = 0.0; C_y = 0.0; C_z = 0.0;

	if (model->m_hmg_flag == HOMOGENIZE_X){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_yz; ei++){
			e = (model->m_nx-2)*dim_y+ei%dim_y+(ei/dim_y)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 1 (right,bottom,near)
			C_i += thisCB[3]*dim_x;  C_j += thisCB[27]*dim_x; C_k += thisCB[51]*dim_x;
			C_x += thisCB[75]*dim_x; C_y += thisCB[99]*dim_x; C_z += thisCB[123]*dim_x;

			// node 2 (right,top,near)
			C_i += thisCB[6]*dim_x;  C_j += thisCB[30]*dim_x;  C_k += thisCB[54];
			C_x += thisCB[78]*dim_x; C_y += thisCB[102]*dim_x; C_z += thisCB[126];

			// node 5 (right,bottom,far)
			C_i += thisCB[15]; C_j += thisCB[39];  C_k += thisCB[63]*dim_x;
			C_x += thisCB[87]; C_y += thisCB[111]; C_z += thisCB[135]*dim_x;

			// node 6 (right,top,far)
			C_i += thisCB[18]*dim_x; C_j += thisCB[42]*dim_x;  C_k += thisCB[66]*dim_x;
			C_x += thisCB[90]*dim_x; C_y += thisCB[114]*dim_x; C_z += thisCB[138]*dim_x;
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_Y){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_xz; ei++){
			e = (ei%dim_x)*dim_y+(ei/dim_x)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 2 (right,top,near)
			C_i += thisCB[7]*dim_y;  C_j += thisCB[31]*dim_y;  C_k += thisCB[55]*dim_y;
			C_x += thisCB[79]*dim_y; C_y += thisCB[103]*dim_y; C_z += thisCB[127]*dim_y;

			// node 3 (left,top,near)
			C_i += thisCB[10]*dim_y; C_j += thisCB[34]*dim_y;  C_k += thisCB[58]*dim_y;
			C_x += thisCB[82]*dim_y; C_y += thisCB[106]*dim_y; C_z += thisCB[130]*dim_y;

			// node 5 (right,top,far)
			C_i += thisCB[19]*dim_y; C_j += thisCB[43]*dim_y;  C_k += thisCB[67]*dim_y;
			C_x += thisCB[91]*dim_y; C_y += thisCB[115]*dim_y; C_z += thisCB[139]*dim_y;

			// node 7 (left,top,far)
			C_i += thisCB[22]*dim_y; C_j += thisCB[46]*dim_y;  C_k += thisCB[70]*dim_y;
			C_x += thisCB[94]*dim_y; C_y += thisCB[118]*dim_y; C_z += thisCB[142]*dim_y;
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_Z){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_xy; ei++){
			e = ei;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 0 (left,bottom,near)
			C_i += thisCB[2]*dim_z;  C_j += thisCB[26]*dim_z; C_k += thisCB[50]*dim_z;
			C_x += thisCB[74]*dim_z; C_y += thisCB[98]*dim_z; C_z += thisCB[122]*dim_z;

			// node 1 (right,bottom,near)
			C_i += thisCB[5]*dim_z;  C_j += thisCB[29]*dim_z;  C_k += thisCB[53]*dim_z;
			C_x += thisCB[77]*dim_z; C_y += thisCB[101]*dim_z; C_z += thisCB[125]*dim_z;

			// node 2 (right,top,near)
			C_i += thisCB[8]*dim_z;  C_j += thisCB[32]*dim_z;  C_k += thisCB[56]*dim_z;
			C_x += thisCB[80]*dim_z; C_y += thisCB[104]*dim_z; C_z += thisCB[128]*dim_z;

			// node 3 (left,top,near)
			C_i += thisCB[11]*dim_z; C_j += thisCB[35]*dim_z;  C_k += thisCB[59]*dim_z;
			C_x += thisCB[83]*dim_z; C_y += thisCB[107]*dim_z; C_z += thisCB[131]*dim_z;
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_YZ){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_xz; ei++){
			e = (ei%dim_x)*dim_y+(ei/dim_x)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 2 (right,top,near)
			C_i += thisCB[8]*dim_y;  C_j += thisCB[32]*dim_y;  C_k += thisCB[56]*dim_y;
			C_x += thisCB[80]*dim_y; C_y += thisCB[104]*dim_y; C_z += thisCB[128]*dim_y;

			// node 3 (left,top,near)
			C_i += thisCB[11]*dim_y; C_j += thisCB[35]*dim_y;  C_k += thisCB[59]*dim_y;
			C_x += thisCB[83]*dim_y; C_y += thisCB[107]*dim_y; C_z += thisCB[131]*dim_y;

			// node 5 (right,top,far)
			C_i += thisCB[20]*dim_y; C_j += thisCB[44]*dim_y;  C_k += thisCB[68]*dim_y;
			C_x += thisCB[92]*dim_y; C_y += thisCB[116]*dim_y; C_z += thisCB[140]*dim_y;

			// node 7 (left,top,far)
			C_i += thisCB[23]*dim_y; C_j += thisCB[47]*dim_y;  C_k += thisCB[71]*dim_y;
			C_x += thisCB[95]*dim_y; C_y += thisCB[119]*dim_y; C_z += thisCB[143]*dim_y;
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_XZ){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_xy; ei++){
			e = ei;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 0 (left,bottom,near)
			C_i += thisCB[0]*dim_z;  C_j += thisCB[24]*dim_z; C_k += thisCB[48]*dim_z;
			C_x += thisCB[72]*dim_z; C_y += thisCB[96]*dim_z; C_z += thisCB[120]*dim_z;

			// node 1 (right,bottom,near)
			C_i += thisCB[3]*dim_z;  C_j += thisCB[27]*dim_z; C_k += thisCB[51]*dim_z;
			C_x += thisCB[75]*dim_z; C_y += thisCB[99]*dim_z; C_z += thisCB[123]*dim_z;

			// node 2 (right,top,near)
			C_i += thisCB[6]*dim_z;  C_j += thisCB[30]*dim_z;  C_k += thisCB[54]*dim_z;
			C_x += thisCB[78]*dim_z; C_y += thisCB[102]*dim_z; C_z += thisCB[126]*dim_z;

			// node 3 (left,top,near)
			C_i += thisCB[9]*dim_z;  C_j += thisCB[33]*dim_z;  C_k += thisCB[57]*dim_z;
			C_x += thisCB[81]*dim_z; C_y += thisCB[105]*dim_z; C_z += thisCB[129]*dim_z;
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_XY){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_xz; ei++){
			e = (ei%dim_x)*dim_y+(ei/dim_x)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

			// node 2 (right,top,near)
			C_i += thisCB[6]*dim_y;  C_j += thisCB[30]*dim_y;  C_k += thisCB[54]*dim_y;
			C_x += thisCB[78]*dim_y; C_y += thisCB[102]*dim_y; C_z += thisCB[126]*dim_y;

			// node 3 (left,top,near)
			C_i += thisCB[9]*dim_y;  C_j += thisCB[33]*dim_y;  C_k += thisCB[57]*dim_y;
			C_x += thisCB[81]*dim_y; C_y += thisCB[105]*dim_y; C_z += thisCB[129]*dim_y;

			// node 5 (right,top,far)
			C_i += thisCB[18]*dim_y; C_j += thisCB[42]*dim_y;  C_k += thisCB[66]*dim_y;
			C_x += thisCB[90]*dim_y; C_y += thisCB[114]*dim_y; C_z += thisCB[138]*dim_y;

			// node 7 (left,top,far)
			C_i += thisCB[21]*dim_y; C_j += thisCB[45]*dim_y;  C_k += thisCB[69]*dim_y;
			C_x += thisCB[93]*dim_y; C_y += thisCB[117]*dim_y; C_z += thisCB[141]*dim_y;
		}
	}

	unsigned int ii;

	#pragma omp parallel for private(ii,thisCB,d,n) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
	for (e=0;e<model->m_nelem;e++){

		thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);

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
	}
	
	model->C[i] = C_i / model->m_nelem;
	model->C[j] = C_j / model->m_nelem;
	model->C[k] = C_k / model->m_nelem;
	model->C[x] = C_x / model->m_nelem;
	model->C[y] = C_y / model->m_nelem;
	model->C[z] = C_z / model->m_nelem;

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

	C_i = 0.0; C_j = 0.0; C_k = 0.0; C_x = 0.0; C_y = 0.0; C_z = 0.0;

	if (model->m_hmg_flag == HOMOGENIZE_X){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_yz; ei++){
			e = (model->m_nx-2)*dim_y+ei%dim_y+(ei/dim_y)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			// node 1 (right,bottom,near)
			C_i += thisCB[3]*scl*dim_x;  C_j += thisCB[27]*scl*dim_x; C_k += thisCB[51]*scl*dim_x;
			C_x += thisCB[75]*scl*dim_x; C_y += thisCB[99]*scl*dim_x; C_z += thisCB[123]*scl*dim_x;

			// node 2 (right,top,near)
			C_i += thisCB[6]*scl*dim_x;  C_j += thisCB[30]*scl*dim_x;  C_k += thisCB[54];
			C_x += thisCB[78]*scl*dim_x; C_y += thisCB[102]*scl*dim_x; C_z += thisCB[126];

			// node 5 (right,bottom,far)
			C_i += thisCB[15]; C_j += thisCB[39];  C_k += thisCB[63]*scl*dim_x;
			C_x += thisCB[87]; C_y += thisCB[111]; C_z += thisCB[135]*scl*dim_x;

			// node 6 (right,top,far)
			C_i += thisCB[18]*scl*dim_x; C_j += thisCB[42]*scl*dim_x;  C_k += thisCB[66]*scl*dim_x;
			C_x += thisCB[90]*scl*dim_x; C_y += thisCB[114]*scl*dim_x; C_z += thisCB[138]*scl*dim_x;
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_Y){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_xz; ei++){
			e = (ei%dim_x)*dim_y+(ei/dim_x)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			// node 2 (right,top,near)
			C_i += thisCB[7]*scl*dim_y;  C_j += thisCB[31]*scl*dim_y;  C_k += thisCB[55]*scl*dim_y;
			C_x += thisCB[79]*scl*dim_y; C_y += thisCB[103]*scl*dim_y; C_z += thisCB[127]*scl*dim_y;

			// node 3 (left,top,near)
			C_i += thisCB[10]*scl*dim_y; C_j += thisCB[34]*scl*dim_y;  C_k += thisCB[58]*scl*dim_y;
			C_x += thisCB[82]*scl*dim_y; C_y += thisCB[106]*scl*dim_y; C_z += thisCB[130]*scl*dim_y;

			// node 5 (right,top,far)
			C_i += thisCB[19]*scl*dim_y; C_j += thisCB[43]*scl*dim_y;  C_k += thisCB[67]*scl*dim_y;
			C_x += thisCB[91]*scl*dim_y; C_y += thisCB[115]*scl*dim_y; C_z += thisCB[139]*scl*dim_y;

			// node 7 (left,top,far)
			C_i += thisCB[22]*scl*dim_y; C_j += thisCB[46]*scl*dim_y;  C_k += thisCB[70]*scl*dim_y;
			C_x += thisCB[94]*scl*dim_y; C_y += thisCB[118]*scl*dim_y; C_z += thisCB[142]*scl*dim_y;
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_Z){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_xy; ei++){
			e = ei;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			// node 0 (left,bottom,near)
			C_i += thisCB[2]*scl*dim_z;  C_j += thisCB[26]*scl*dim_z; C_k += thisCB[50]*scl*dim_z;
			C_x += thisCB[74]*scl*dim_z; C_y += thisCB[98]*scl*dim_z; C_z += thisCB[122]*scl*dim_z;

			// node 1 (right,bottom,near)
			C_i += thisCB[5]*scl*dim_z;  C_j += thisCB[29]*scl*dim_z;  C_k += thisCB[53]*scl*dim_z;
			C_x += thisCB[77]*scl*dim_z; C_y += thisCB[101]*scl*dim_z; C_z += thisCB[125]*scl*dim_z;

			// node 2 (right,top,near)
			C_i += thisCB[8]*scl*dim_z;  C_j += thisCB[32]*scl*dim_z;  C_k += thisCB[56]*scl*dim_z;
			C_x += thisCB[80]*scl*dim_z; C_y += thisCB[104]*scl*dim_z; C_z += thisCB[128]*scl*dim_z;

			// node 3 (left,top,near)
			C_i += thisCB[11]*scl*dim_z; C_j += thisCB[35]*scl*dim_z;  C_k += thisCB[59]*scl*dim_z;
			C_x += thisCB[83]*scl*dim_z; C_y += thisCB[107]*scl*dim_z; C_z += thisCB[131]*scl*dim_z;
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_YZ){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_xz; ei++){
			e = (ei%dim_x)*dim_y+(ei/dim_x)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			// node 2 (right,top,near)
			C_i += thisCB[8]*scl*dim_y;  C_j += thisCB[32]*scl*dim_y;  C_k += thisCB[56]*scl*dim_y;
			C_x += thisCB[80]*scl*dim_y; C_y += thisCB[104]*scl*dim_y; C_z += thisCB[128]*scl*dim_y;

			// node 3 (left,top,near)
			C_i += thisCB[11]*scl*dim_y; C_j += thisCB[35]*scl*dim_y;  C_k += thisCB[59]*scl*dim_y;
			C_x += thisCB[83]*scl*dim_y; C_y += thisCB[107]*scl*dim_y; C_z += thisCB[131]*scl*dim_y;

			// node 5 (right,top,far)
			C_i += thisCB[20]*scl*dim_y; C_j += thisCB[44]*scl*dim_y;  C_k += thisCB[68]*scl*dim_y;
			C_x += thisCB[92]*scl*dim_y; C_y += thisCB[116]*scl*dim_y; C_z += thisCB[140]*scl*dim_y;

			// node 7 (left,top,far)
			C_i += thisCB[23]*scl*dim_y; C_j += thisCB[47]*scl*dim_y;  C_k += thisCB[71]*scl*dim_y;
			C_x += thisCB[95]*scl*dim_y; C_y += thisCB[119]*scl*dim_y; C_z += thisCB[143]*scl*dim_y;
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_XZ){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_xy; ei++){
			e = ei;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			// node 0 (left,bottom,near)
			C_i += thisCB[0]*scl*dim_z;  C_j += thisCB[24]*scl*dim_z; C_k += thisCB[48]*scl*dim_z;
			C_x += thisCB[72]*scl*dim_z; C_y += thisCB[96]*scl*dim_z; C_z += thisCB[120]*scl*dim_z;

			// node 1 (right,bottom,near)
			C_i += thisCB[3]*scl*dim_z;  C_j += thisCB[27]*scl*dim_z; C_k += thisCB[51]*scl*dim_z;
			C_x += thisCB[75]*scl*dim_z; C_y += thisCB[99]*scl*dim_z; C_z += thisCB[123]*scl*dim_z;

			// node 2 (right,top,near)
			C_i += thisCB[6]*scl*dim_z;  C_j += thisCB[30]*scl*dim_z;  C_k += thisCB[54]*scl*dim_z;
			C_x += thisCB[78]*scl*dim_z; C_y += thisCB[102]*scl*dim_z; C_z += thisCB[126]*scl*dim_z;

			// node 3 (left,top,near)
			C_i += thisCB[9]*scl*dim_z;  C_j += thisCB[33]*scl*dim_z;  C_k += thisCB[57]*scl*dim_z;
			C_x += thisCB[81]*scl*dim_z; C_y += thisCB[105]*scl*dim_z; C_z += thisCB[129]*scl*dim_z;
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_XY){

		#pragma omp parallel for private(e,thisCB) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
		for (ei=0; ei<dim_xz; ei++){
			e = (ei%dim_x)*dim_y+(ei/dim_x)*dim_xy;

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			// node 2 (right,top,near)
			C_i += thisCB[6]*scl*dim_y;  C_j += thisCB[30]*scl*dim_y;  C_k += thisCB[54]*scl*dim_y;
			C_x += thisCB[78]*scl*dim_y; C_y += thisCB[102]*scl*dim_y; C_z += thisCB[126]*scl*dim_y;

			// node 3 (left,top,near)
			C_i += thisCB[9]*scl*dim_y;  C_j += thisCB[33]*scl*dim_y;  C_k += thisCB[57]*scl*dim_y;
			C_x += thisCB[81]*scl*dim_y; C_y += thisCB[105]*scl*dim_y; C_z += thisCB[129]*scl*dim_y;

			// node 5 (right,top,far)
			C_i += thisCB[18]*scl*dim_y; C_j += thisCB[42]*scl*dim_y;  C_k += thisCB[66]*scl*dim_y;
			C_x += thisCB[90]*scl*dim_y; C_y += thisCB[114]*scl*dim_y; C_z += thisCB[138]*scl*dim_y;

			// node 7 (left,top,far)
			C_i += thisCB[21]*scl*dim_y; C_j += thisCB[45]*scl*dim_y;  C_k += thisCB[69]*scl*dim_y;
			C_x += thisCB[93]*scl*dim_y; C_y += thisCB[117]*scl*dim_y; C_z += thisCB[141]*scl*dim_y;
		}
	}

	unsigned int ii;

	#pragma omp parallel for private(ii,thisCB,d,n) reduction(+:C_i,C_j,C_k,C_x,C_y,C_z)
	for (e=0;e<model->m_nelem;e++){

		thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);
		scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

		// node 0 (left,bottom,near)
		n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		for (ii=0;ii<3;ii++){
			d = scl*D[model->node_dof_map[n]+ii];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 1 (right,bottom,near)
		n+=model->m_ny;
		for (ii=3;ii<6;ii++){
			d = scl*D[model->node_dof_map[n]+ii-3];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 2 (right,top,near)
		n-=1;
		for (ii=6;ii<9;ii++){
			d = scl*D[model->node_dof_map[n]+ii-6];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 3 (left,top,near)
		n-=model->m_ny;
		for (ii=9;ii<12;ii++){
			d = scl*D[model->node_dof_map[n]+ii-9];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 4 (left,bottom,far)
		n+=1+model->m_nx*model->m_ny;
		for (ii=12;ii<15;ii++){
			d = scl*D[model->node_dof_map[n]+ii-12];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 5 (right,bottom,far)
		n+=model->m_ny;
		for (ii=15;ii<18;ii++){
			d = scl*D[model->node_dof_map[n]+ii-15];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 6 (right,top,far)
		n-=1;
		for (ii=18;ii<21;ii++){
			d = scl*D[model->node_dof_map[n]+ii-18];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}

		// node 7 (left,top,far)
		n-=model->m_ny;
		for (ii=21;ii<24;ii++){
			d = scl*D[model->node_dof_map[n]+ii-21];
			C_i += thisCB[ii]*d;    C_j += thisCB[ii+24]*d; C_k += thisCB[ii+48]*d;
			C_x += thisCB[ii+72]*d; C_y += thisCB[ii+96]*d; C_z += thisCB[ii+120]*d;
		}
	}
	
	model->C[i] = C_i / model->m_nelem;
	model->C[j] = C_j / model->m_nelem;
	model->C[k] = C_k / model->m_nelem;
	model->C[x] = C_x / model->m_nelem;
	model->C[y] = C_y / model->m_nelem;
	model->C[z] = C_z / model->m_nelem;

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
	  if (model->m_hmg_thermal_expansion_flag == HMG_TRUE){
	    printf("Effective thermal expansion:\n");
	    printf("  %.8e  %.8e  %.8e\n",model->thermal_expansion[0],model->thermal_expansion[5],model->thermal_expansion[4]);
	    printf("  %.8e  %.8e  %.8e\n",model->thermal_expansion[5],model->thermal_expansion[1],model->thermal_expansion[3]);
		printf("  %.8e  %.8e  %.8e\n",model->thermal_expansion[4],model->thermal_expansion[3],model->thermal_expansion[2]);
	    printf("-------------------------------------------------------\n");
	  }
	} else {
	  if (model->m_hmg_thermal_expansion_flag == HMG_FALSE){
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
		"-------------------------------------------------------\n"\
        "Effective thermal expansion:\n"\
        "  %.8e  %.8e  %.8e\n"\
        "  %.8e  %.8e  %.8e\n"\
		"  %.8e  %.8e  %.8e\n"\
		"-------------------------------------------------------\n",
		model->C[0] , model->C[1] , model->C[2] , model->C[3] , model->C[4] , model->C[5],
		model->C[6] , model->C[7] , model->C[8] , model->C[9] , model->C[10], model->C[11],
		model->C[12], model->C[13], model->C[14], model->C[15], model->C[16], model->C[17],
		model->C[18], model->C[19], model->C[20], model->C[21], model->C[22], model->C[23],
		model->C[24], model->C[25], model->C[26], model->C[27], model->C[28], model->C[29],
		model->C[30], model->C[31], model->C[32], model->C[33], model->C[34], model->C[35],
		model->thermal_expansion[0],model->thermal_expansion[5],model->thermal_expansion[4],
		model->thermal_expansion[5],model->thermal_expansion[1],model->thermal_expansion[3],
		model->thermal_expansion[4],model->thermal_expansion[3],model->thermal_expansion[2]
		);
	  }
  }
  return;
}
//------------------------------------------------------------------------------}
void strainFromDispl_elastic_3D(var *s, var *d, var x, var y, var z){
	var x1 = (1.0-x), y1 = (1.0-y), z1 = (1.0-z);
	var B[144] = { -y1*z, 0.0, 0.0, y1*z, 0.0, 0.0, y*z, 0.0, 0.0, -y*z, 0.0, 0.0, -y1*z1, 0.0, 0.0, y1*z1, 0.0, 0.0, y*z1, 0.0, 0.0, -y*z1, 0.0, 0.0,
	               0.0, -x1*z, 0.0, 0.0, -x*z, 0.0, 0.0, x*z, 0.0, 0.0, x1*z, 0.0, 0.0, -x1*z1, 0.0, 0.0, -x*z1, 0.0, 0.0, x*z1, 0.0, 0.0, x1*z1, 0.0,
								 0.0, 0.0, x1*y1, 0.0, 0.0, x*y1, 0.0, 0.0, x*y, 0.0, 0.0, x1*y, 0.0, 0.0, -x1*y1, 0.0, 0.0, -x*y1, 0.0, 0.0, -x*y, 0.0, 0.0, -x1*y,
								 0.0, x1*y1, -x1*z, 0.0, x*y1, -x*z, 0.0, x*y, x*z, 0.0, x1*y, x1*z, 0.0, -x1*y1, -x1*z1, 0.0, -x*y1, -x*z1, 0.0, -x*y, x*z1, 0.0, -x1*y, x1*z1,
								 x1*y1, 0.0, -y1*z, x*y1, 0.0, y1*z, x*y, 0.0, y*z, x1*y, 0.0, -y*z, -x1*y1, 0.0, -y1*z1, -x*y1, 0.0, y1*z1, -x*y, 0.0, y*z1, -x1*y, 0.0, -y*z1,
								 -x1*z, -y1*z, 0.0, -x*z, y1*z, 0.0, x*z, y*z, 0.0, x1*z, -y*z, 0.0, -x1*z1, -y1*z1, 0.0, -x*z1, y1*z1, 0.0, x*z1, y*z1, 0.0, x1*z1, -y*z1, 0.0 };

	s[0] = 0.0;
	s[1] = 0.0;
	s[2] = 0.0;
	s[3] = 0.0;
	s[4] = 0.0;
	s[5] = 0.0;
	for (unsigned int i=0; i<24; i++){
		s[0] +=     B[i] * d[i];
		s[1] +=  B[i+24] * d[i];
		s[2] +=  B[i+48] * d[i];
		s[3] +=  B[i+72] * d[i];
		s[4] +=  B[i+96] * d[i];
		s[5] += B[i+120] * d[i];
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
void stressFromAlpha_elastic_3D(var *s, var *a, var E, var v, var x, var y, var z){

	var coeff = E/ ((1.0+v)*(1.0-2.0*v));
	var C[36] = { coeff*(1.0-v), coeff*v, coeff*v, 0.0, 0.0, 0.0,
						 	  coeff*v, coeff*(1.0-v), coeff*v, 0.0, 0.0, 0.0,
								coeff*v, coeff*v, coeff*(1.0-v), 0.0, 0.0, 0.0,
							  0.0, 0.0, 0.0, coeff*0.5*(1.0-2.0*v), 0.0, 0.0,
							  0.0, 0.0, 0.0, 0.0, coeff*0.5*(1.0-2.0*v), 0.0,
							  0.0, 0.0, 0.0, 0.0, 0.0, coeff*0.5*(1.0-2.0*v) };

	for (unsigned int j=0; j<6; j++){
		s[j] = 0.0;
		for (unsigned int k=0; k<6; k++) s[j] += C[j*6+k]*a[k];
	}
	return;
}
//------------------------------------------------------------------------------
void forceFromStrain_elastic_3D(var *f, var *a, var E, var v, var x, var y, var z){

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
				 
	var S[144];
	var *ptr;

	for (unsigned int i=0; i<24; i++){
		for (unsigned int j=0; j<6; j++){
			S[6*i+j] = 0.0;
			ptr = &S[6*i+j];
			for (unsigned int k=0; k<6; k++) *ptr += B[i+24*k]*C[j+6*k];
		}
	}
	
	for (unsigned int i=0; i<24; i++){
		f[i] = 0.0;
		for (unsigned int k=0; k<6; k++) f[i] += S[6*i+k]*a[k];
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
