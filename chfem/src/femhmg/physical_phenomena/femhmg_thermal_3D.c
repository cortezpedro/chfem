#include "femhmg_3D.h"
#include "femhmg_thermal_3D.h"

//------------------------------------------------------------------------------
logical initModel_thermal_3D(hmgModel_t *model){
	if ((model->m_nx-1) <= 0 || (model->m_ny-1) <= 0 || (model->m_nz-1) <= 0 || model->m_nmat <= 0){
	  printf("ERROR: Failed to initialize model due to bad input parameters.\nrows:%d\ncols:%d\nslices:%d\nmaterials:%d\n",(model->m_ny-1),(model->m_nx-1),(model->m_nz-1),model->m_nmat);
		return HMG_FALSE;
	}

	model->m_C_dim = 9;
	model->m_lclMtx_dim = 40;
	model->m_lclCB_dim = 0;

	model->m_nnodedof = 1;
	model->m_nelemdof = 8;
	model->m_nnode = model->m_nx * model->m_ny * model->m_nz;
	model->m_nelem = (model->m_nx-1) * (model->m_ny-1) * (model->m_nz-1);
	model->m_nporeelem = model->m_nelem; // pore elems comprehend the pore space, the domain. as mesh is monolithic, porelem=elem.
	model->m_ndof = model->m_nelem;

	if (model->sdfFile==NULL){
	  model->assembleRHS = assembleRHS_thermal_3D;
	  model->updateC = updateC_thermal_3D;
	  model->saveFields = saveFields_thermal_3D;
	} else {
	  model->assembleRHS = assembleRHS_thermal_3D_ScalarDensityField;
	  model->updateC = updateC_thermal_3D_ScalarDensityField;
	  model->saveFields = saveFields_thermal_3D_ScalarDensityField;
	}
	
	model->assembleLocalMtxs = assembleLocalMtxs_thermal_3D;
	model->printC = printC_thermal_3D;

	model->assembleNodeDofMap = assembleNodeDofMap_3D;
	model->assembleDofIdMap = NULL;
	model->assembleDofMaterialMap = assembleDofMaterialMap_3D;

	return HMG_TRUE;
}
//------------------------------------------------------------------------------
void assembleLocalMtxs_thermal_3D(hmgModel_t *model){
  /*
	  As mesh is voxel-based, all elements have a similar analytical solution
	  for their local conductivity matrix.

	  ATTENTION: Zeros are not stored.

	  k_el = material_prop * {
	    {   1/3,     0, -1/12,     0,     0, -1/12, -1/12, -1/12 },
	    {     0,   1/3,     0, -1/12, -1/12,     0, -1/12, -1/12 },
	    { -1/12,     0,   1/3,     0, -1/12, -1/12,     0, -1/12 },
	    {     0, -1/12,     0,   1/3, -1/12, -1/12, -1/12,     0 },
	    {     0, -1/12, -1/12, -1/12,   1/3,     0, -1/12,     0 },
	    { -1/12,     0, -1/12, -1/12,     0,   1/3,     0, -1/12 },
	    { -1/12, -1/12,     0, -1/12, -1/12,     0,   1/3,     0 },
	    { -1/12, -1/12, -1/12,     0,     0, -1/12,     0,   1/3 }
	  }
	*/

	cudapcgVar_t diag = 1.0/3.0;
	cudapcgVar_t out_of_diag = -1.0/12.0;
	cudapcgVar_t thisProp;

	unsigned int i,mat;
	for (i=0;i<model->m_nmat;i++){

		thisProp = model->props[i];
		mat = i * model->m_lclMtx_dim;

		model->Mtxs[mat]    = thisProp * diag;
		model->Mtxs[mat+1]  = thisProp * out_of_diag;
		model->Mtxs[mat+2]  = thisProp * out_of_diag;
		model->Mtxs[mat+3]  = thisProp * out_of_diag;
		model->Mtxs[mat+4]  = thisProp * out_of_diag;

		model->Mtxs[mat+5]  = thisProp * diag;
		model->Mtxs[mat+6]  = thisProp * out_of_diag;
		model->Mtxs[mat+7]  = thisProp * out_of_diag;
		model->Mtxs[mat+8]  = thisProp * out_of_diag;
		model->Mtxs[mat+9]  = thisProp * out_of_diag;

		model->Mtxs[mat+10] = thisProp * out_of_diag;
		model->Mtxs[mat+11] = thisProp * diag;
		model->Mtxs[mat+12] = thisProp * out_of_diag;
		model->Mtxs[mat+13] = thisProp * out_of_diag;
		model->Mtxs[mat+14] = thisProp * out_of_diag;

		model->Mtxs[mat+15] = thisProp * out_of_diag;
		model->Mtxs[mat+16] = thisProp * diag;
		model->Mtxs[mat+17] = thisProp * out_of_diag;
		model->Mtxs[mat+18] = thisProp * out_of_diag;
		model->Mtxs[mat+19] = thisProp * out_of_diag;

		model->Mtxs[mat+20] = thisProp * out_of_diag;
		model->Mtxs[mat+21] = thisProp * out_of_diag;
		model->Mtxs[mat+22] = thisProp * out_of_diag;
		model->Mtxs[mat+23] = thisProp * diag;
		model->Mtxs[mat+24] = thisProp * out_of_diag;

		model->Mtxs[mat+25] = thisProp * out_of_diag;
		model->Mtxs[mat+26] = thisProp * out_of_diag;
		model->Mtxs[mat+27] = thisProp * out_of_diag;
		model->Mtxs[mat+28] = thisProp * diag;
		model->Mtxs[mat+29] = thisProp * out_of_diag;

		model->Mtxs[mat+30] = thisProp * out_of_diag;
		model->Mtxs[mat+31] = thisProp * out_of_diag;
		model->Mtxs[mat+32] = thisProp * out_of_diag;
		model->Mtxs[mat+33] = thisProp * out_of_diag;
		model->Mtxs[mat+34] = thisProp * diag;

		model->Mtxs[mat+35] = thisProp * out_of_diag;
		model->Mtxs[mat+36] = thisProp * out_of_diag;
		model->Mtxs[mat+37] = thisProp * out_of_diag;
		model->Mtxs[mat+38] = thisProp * out_of_diag;
		model->Mtxs[mat+39] = thisProp * diag;
	}
	return;
}
//------------------------------------------------------------------------------
void assembleRHS_thermal_3D(hmgModel_t *model){

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
	cudapcgVar_t * thisK  = NULL;

	/*
		ATTENTION: Zeros are not stored in local FEM matrices.
		"thisK" indexes consider that. No calculations with zeros are made.
	*/

	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (i=0; i<dim_yz; i++){
			e = (model->m_nx-2)*dim_y+i%dim_y+(i/dim_y)*dim_xy;

			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[1]+thisK[2]+thisK[3])*dim_x;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[5]+thisK[8])*dim_x;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= (thisK[11]+thisK[13])*dim_x;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[15]+thisK[18]+thisK[19])*dim_x;

			n += 1+model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[20]+thisK[21]+thisK[24])*dim_x;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[26]+thisK[28])*dim_x;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= (thisK[31]+thisK[34])*dim_x;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[36]+thisK[37]+thisK[38])*dim_x;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (i=0; i<dim_xz; i++){
			e = (i%dim_x)*dim_y+(i/dim_x)*dim_xy;

			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[1]+thisK[3]+thisK[4])*dim_y;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[6]+thisK[8]+thisK[9])*dim_y;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= (thisK[11]+thisK[14])*dim_y;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[16]+thisK[19])*dim_y;

			n += 1+model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[21]+thisK[22]+thisK[24])*dim_y;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[26]+thisK[27]+thisK[29])*dim_y;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= (thisK[32]+thisK[34])*dim_y;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[37]+thisK[39])*dim_y;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Z){
		for (i=0; i<dim_xy; i++){
			e = i;

			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[0]+thisK[1])*dim_z;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[5]+thisK[6])*dim_z;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= (thisK[10]+thisK[11])*dim_z;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[15]+thisK[16])*dim_z;

			n += 1+model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[20]+thisK[21]+thisK[22])*dim_z;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[25]+thisK[26]+thisK[27])*dim_z;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= (thisK[30]+thisK[31]+thisK[32])*dim_z;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[35]+thisK[36]+thisK[37])*dim_z;
		}
	}
	return;
}
//------------------------------------------------------------------------------
void assembleRHS_thermal_3D_ScalarDensityField(hmgModel_t *model){

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
	cudapcgVar_t * thisK  = &(model->Mtxs[0]);
	cudapcgVar_t scl=1.0;

	/*
		ATTENTION: Zeros are not stored in local FEM matrices.
		"thisK" indexes consider that. No calculations with zeros are made.
	*/

	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (i=0; i<dim_yz; i++){
			e = (model->m_nx-2)*dim_y+i%dim_y+(i/dim_y)*dim_xy;

			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[1]+thisK[2]+thisK[3])*dim_x;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[5]+thisK[8])*dim_x;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[11]+thisK[13])*dim_x;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[15]+thisK[18]+thisK[19])*dim_x;

			n += 1+model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[20]+thisK[21]+thisK[24])*dim_x;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[26]+thisK[28])*dim_x;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[31]+thisK[34])*dim_x;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[36]+thisK[37]+thisK[38])*dim_x;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (i=0; i<dim_xz; i++){
			e = (i%dim_x)*dim_y+(i/dim_x)*dim_xy;

			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[1]+thisK[3]+thisK[4])*dim_y;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[6]+thisK[8]+thisK[9])*dim_y;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[11]+thisK[14])*dim_y;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[16]+thisK[19])*dim_y;

			n += 1+model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[21]+thisK[22]+thisK[24])*dim_y;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[26]+thisK[27]+thisK[29])*dim_y;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[32]+thisK[34])*dim_y;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[37]+thisK[39])*dim_y;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Z){
		for (i=0; i<dim_xy; i++){
			e = i;

			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[0]+thisK[1])*dim_z;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[5]+thisK[6])*dim_z;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[10]+thisK[11])*dim_z;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[15]+thisK[16])*dim_z;

			n += 1+model->m_nx*model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[20]+thisK[21]+thisK[22])*dim_z;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[25]+thisK[26]+thisK[27])*dim_z;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[30]+thisK[31]+thisK[32])*dim_z;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[35]+thisK[36]+thisK[37])*dim_z;
		}
	}
	return;
}
//------------------------------------------------------------------------------
void updateC_thermal_3D(hmgModel_t *model, cudapcgVar_t * T){
	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	unsigned int dim_z = model->m_nz-1;
	unsigned int dim_xy = dim_x*dim_y;
	unsigned int dim_xz = dim_x*dim_z;
	unsigned int dim_yz = dim_y*dim_z;

	unsigned int e, e_id, n;
	var coeff, C_lcl, C_i, C_j, C_k;

	unsigned int i,j,k;
	if (model->m_hmg_flag == HOMOGENIZE_X){
		i = 0; j = 3; k = 6;
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		i = 1; j = 4; k = 7;
	} else if (model->m_hmg_flag == HOMOGENIZE_Z){
		i = 2; j = 5; k = 8;
	}

	C_i = 0.0; C_j = 0.0; C_k = 0.0;
	C_lcl = 0.0;

	/*
		ATTENTION:
		The coefficients 0.25 and -0.25 used on the operations to compute model->C
		come from the analytical solution for the flux matrix of an hexahedron
		element on a regular voxel-based mesh.
	*/

	if (model->m_hmg_flag == HOMOGENIZE_X){
		// This loop does the same as
		/*
		// Compute local component to add in model->C
		C_lcl = model->props[model->elem_material_map[e]]*0.25*(model->m_nx-1);

		// node 1 (right,bottom,near)
		C_i = C_lcl; C_j = -C_lcl; C_k = C_lcl;

		// node 2 (right,top,near)
		C_i += C_lcl; C_j += C_lcl; C_k += C_lcl;

		// node 5 (right,bottom,far)
		C_i += C_lcl; C_j -= C_lcl; C_k -= C_lcl;

		// node 6 (right,top,far)
		C_i += C_lcl; C_j += C_lcl; C_k -= C_lcl;

		#pragma omp critical
		{
			model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
		}
		*/
		#pragma omp parallel for private(e_id) reduction(+:C_lcl)
		for (e=0;e<dim_yz;e++){
			e_id = (model->m_nx-2)*dim_y+e%dim_y+(e/dim_y)*dim_xy;
			C_lcl += model->props[model->elem_material_map[e_id]]*(model->m_nx-1);
		}
		model->C[i] += C_lcl;

	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		// Analogous to HOMOGENIZE_X  (uses nodes 2, 3, 6 and 7)
		#pragma omp parallel for private(e_id) reduction(+:C_lcl)
		for (e=0;e<dim_xz;e++){
			e_id = (e%dim_x)*dim_y+(e/dim_x)*dim_xy;
			C_lcl += model->props[model->elem_material_map[e_id]]*(model->m_ny-1);
		}
		model->C[j] += C_lcl;
	} else if (model->m_hmg_flag == HOMOGENIZE_Z){
		// Analogous to HOMOGENIZE_X  (uses nodes 0, 1, 2 and 3)
		#pragma omp parallel for reduction(+:C_lcl)
		for (e=0;e<dim_xy;e++){
			C_lcl += model->props[model->elem_material_map[e]]*(model->m_nz-1);
		}
		model->C[k] += C_lcl;
	}

	#pragma omp parallel for private(C_i,C_j,C_k,n,coeff,C_lcl)
	for (e=0;e<model->m_nelem;e++){

		// Compute coefficient of this element's flux matrix
		coeff = model->props[model->elem_material_map[e]]*0.25;

		// node 0 (left,bottom,near)
		n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i = -C_lcl; C_j = -C_lcl; C_k = C_lcl;

		// node 1 (right,bottom,near)
		n+=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j -= C_lcl; C_k += C_lcl;

		// node 2 (right,top,near)
		n-=1;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j += C_lcl; C_k += C_lcl;

		// node 3 (left,top,near)
		n-=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i -= C_lcl; C_j += C_lcl; C_k += C_lcl;

		// node 4 (left,bottom,far)
		n+=1+model->m_nx*model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i -= C_lcl; C_j -= C_lcl; C_k -= C_lcl;

		// node 5 (right,bottom,far)
		n+=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j -= C_lcl; C_k -= C_lcl;

		// node 6 (right,top,far)
		n-=1;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j += C_lcl; C_k -= C_lcl;

		// node 7 (left,top,far)
		n-=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i -= C_lcl; C_j += C_lcl; C_k -= C_lcl;

		#pragma omp critical
		{
			model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
		}
	}

	model->C[i] /= model->m_nelem; model->C[j] /= model->m_nelem; model->C[k] /= model->m_nelem;

	return;
}
//------------------------------------------------------------------------------
void updateC_thermal_3D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * T){
	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	unsigned int dim_z = model->m_nz-1;
	unsigned int dim_xy = dim_x*dim_y;
	unsigned int dim_xz = dim_x*dim_z;
	unsigned int dim_yz = dim_y*dim_z;

	unsigned int e, e_id, n;
	var coeff, C_lcl, C_i, C_j, C_k;

	unsigned int i,j,k;
	if (model->m_hmg_flag == HOMOGENIZE_X){
		i = 0; j = 3; k = 6;
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		i = 1; j = 4; k = 7;
	} else if (model->m_hmg_flag == HOMOGENIZE_Z){
		i = 2; j = 5; k = 8;
	}

	C_i = 0.0; C_j = 0.0; C_k = 0.0;
	C_lcl = 0.0;

	/*
		ATTENTION:
		The coefficients 0.25 and -0.25 used on the operations to compute model->C
		come from the analytical solution for the flux matrix of an hexahedron
		element on a regular voxel-based mesh.
	*/

	if (model->m_hmg_flag == HOMOGENIZE_X){
		// This loop does the same as
		/*
		// Compute local component to add in model->C
		C_lcl = model->props[model->elem_material_map[e]]*0.25*(model->m_nx-1);

		// node 1 (right,bottom,near)
		C_i = C_lcl; C_j = -C_lcl; C_k = C_lcl;

		// node 2 (right,top,near)
		C_i += C_lcl; C_j += C_lcl; C_k += C_lcl;

		// node 5 (right,bottom,far)
		C_i += C_lcl; C_j -= C_lcl; C_k -= C_lcl;

		// node 6 (right,top,far)
		C_i += C_lcl; C_j += C_lcl; C_k -= C_lcl;

		#pragma omp critical
		{
			model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
		}
		*/
		#pragma omp parallel for private(e_id) reduction(+:C_lcl)
		for (e=0;e<dim_yz;e++){
			e_id = (model->m_nx-2)*dim_y+e%dim_y+(e/dim_y)*dim_xy;
			C_lcl += (model->density_map[e_id]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min)*(model->m_nx-1);
		}
		model->C[i] += C_lcl;

	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		// Analogous to HOMOGENIZE_X  (uses nodes 2, 3, 6 and 7)
		#pragma omp parallel for private(e_id) reduction(+:C_lcl)
		for (e=0;e<dim_xz;e++){
			e_id = (e%dim_x)*dim_y+(e/dim_x)*dim_xy;
			C_lcl += (model->density_map[e_id]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min)*(model->m_ny-1);
		}
		model->C[j] += C_lcl;
	} else if (model->m_hmg_flag == HOMOGENIZE_Z){
		// Analogous to HOMOGENIZE_X  (uses nodes 0, 1, 2 and 3)
		#pragma omp parallel for reduction(+:C_lcl)
		for (e=0;e<dim_xy;e++){
			C_lcl += (model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min)*(model->m_nz-1);
		}
		model->C[k] += C_lcl;
	}

	#pragma omp parallel for private(C_i,C_j,C_k,n,coeff,C_lcl)
	for (e=0;e<model->m_nelem;e++){

		// Compute coefficient of this element's flux matrix
		coeff = (model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min)*0.25;

		// node 0 (left,bottom,near)
		n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i = -C_lcl; C_j = -C_lcl; C_k = C_lcl;

		// node 1 (right,bottom,near)
		n+=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j -= C_lcl; C_k += C_lcl;

		// node 2 (right,top,near)
		n-=1;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j += C_lcl; C_k += C_lcl;

		// node 3 (left,top,near)
		n-=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i -= C_lcl; C_j += C_lcl; C_k += C_lcl;

		// node 4 (left,bottom,far)
		n+=1+model->m_nx*model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i -= C_lcl; C_j -= C_lcl; C_k -= C_lcl;

		// node 5 (right,bottom,far)
		n+=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j -= C_lcl; C_k -= C_lcl;

		// node 6 (right,top,far)
		n-=1;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j += C_lcl; C_k -= C_lcl;

		// node 7 (left,top,far)
		n-=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i -= C_lcl; C_j += C_lcl; C_k -= C_lcl;

		#pragma omp critical
		{
			model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
		}
	}

	model->C[i] /= model->m_nelem; model->C[j] /= model->m_nelem; model->C[k] /= model->m_nelem;

	return;
}
//------------------------------------------------------------------------------
void printC_thermal_3D(hmgModel_t *model, char *dest){
	if (dest==NULL){
	  printf("-------------------------------------------------------\n");
	  printf("Homogenized Constitutive Matrix (Thermal Conductivity):\n");
	  printf("  %.8e  ", model->C[0]); printf("%.8e  ", model->C[1]); printf("%.8e\n", model->C[2]);
	  printf("  %.8e  ", model->C[3]); printf("%.8e  ", model->C[4]); printf("%.8e\n", model->C[5]);
	  printf("  %.8e  ", model->C[6]); printf("%.8e  ", model->C[7]); printf("%.8e\n", model->C[8]);
	  printf("-------------------------------------------------------\n");
	} else {
	  sprintf(
      dest,
      "-------------------------------------------------------\n"\
      "Homogenized Constitutive Matrix (Thermal Conductivity):\n"\
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
//------------------------------------------------------------------------------}
void fluxFromTemp_thermal_3D(var *q, var *t, var k, var x, var y, var z){
	var x1 = (1.0-x), y1 = (1.0-y), z1 = (1.0-z);
	var B[24] = { -y1*z, y1*z, y*z, -y*z, -y1*z1, y1*z1, y*z1, -y*z1,
	              -x1*z, -x*z, x*z, x1*z, -x1*z1, -x*z1, x*z1, x1*z1,
		       x1*y1, x*y1, x*y, x1*y, -x1*y1, -x*y1, -x*y, -x1*y };

	q[0] = 0.0;
	q[1] = 0.0;
	q[2] = 0.0;
	for (unsigned int i=0; i<8; i++){
		q[0] +=     B[i] * t[i];
		q[1] +=  B[i+ 8] * t[i];
		q[2] +=  B[i+16] * t[i];
	}
	q[0] *= k;
	q[1] *= k;
	q[2] *= k;
	return;
}
//------------------------------------------------------------------------------
void saveFields_thermal_3D(hmgModel_t *model, cudapcgVar_t * T){

  cudapcgVar_t * Q = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*model->m_ndof*3);
  for (unsigned int i=0; i<model->m_ndof*3; i++){
    Q[i] = 0.0;
  }

  unsigned int rows = model->m_ny-1;
  unsigned int cols = model->m_nx-1;
  unsigned int lays = model->m_nz-1;
  unsigned int rowscols = rows*cols;
  
  unsigned int dof, n, e;

  var local_Q[3];
  var local_T[8];
  
  var k;
  
  #pragma omp parallel for private(local_Q,local_T,n,dof,k)
  for (unsigned int e=0;e<model->m_nelem; e++){

		k = model->props[model->elem_material_map[e]];
		
		// node 0 (left,bottom,near)
		n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
		dof = model->node_dof_map[n];
		local_T[0] = (var) T[dof];

		// node 1 (right,bottom,near)
		n+=model->m_ny;
		dof = model->node_dof_map[n];
		local_T[1] = (var) T[dof];

		// node 2 (right,top,near)
		n-=1;
		dof = model->node_dof_map[n];
		local_T[2] = (var) T[dof];

		// node 3 (left,top,near)
		n-=model->m_ny;
		dof = model->node_dof_map[n];
		local_T[3] = (var) T[dof];

		// node 4 (left,bottom,far)
		n+=1+model->m_nx*model->m_ny;
		dof = model->node_dof_map[n];
		local_T[4] = (var) T[dof];

		// node 5 (right,bottom,far)
		n+=model->m_ny;
		dof = model->node_dof_map[n];
		local_T[5] = (var) T[dof];

		// node 6 (right,top,far)
		n-=1;
		dof = model->node_dof_map[n];
		local_T[6] = (var) T[dof];

		// node 7 (left,top,far)
		n-=model->m_ny;
		dof = model->node_dof_map[n];
		local_T[7] = (var) T[dof];
		
		fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,1.0,0.0);
		Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];
		
		// node 0 (left,bottom,near)
		n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
		dof = model->node_dof_map[n];
		
		fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,0.0,1.0);
		Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		// node 1 (right,bottom,near)
		n+=model->m_ny;
		dof = model->node_dof_map[n];
		
		fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,0.0,1.0);
		Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		// node 2 (right,top,near)
		n-=1;
		dof = model->node_dof_map[n];
		
		fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,1.0,1.0);
		Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		// node 3 (left,top,near)
		n-=model->m_ny;
		dof = model->node_dof_map[n];
		
		fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,1.0,1.0);
		Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		// node 4 (left,bottom,far)
		n+=1+model->m_nx*model->m_ny;
		dof = model->node_dof_map[n];
		
		fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,0.0,0.0);
		Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		// node 5 (right,bottom,far)
		n+=model->m_ny;
		dof = model->node_dof_map[n];
		
		fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,0.0,0.0);
		Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		// node 6 (right,top,far)
		n-=1;
		dof = model->node_dof_map[n];
		
		fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,1.0,0.0);
		Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];
	}
	
	// Compensate for periodic borders
	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (unsigned int l=0; l<lays; l++){
			for (unsigned int r=0; r<rows; r++){
			e = r + (rowscols-rows) + l*rowscols;
			k = model->props[model->elem_material_map[e]];

			local_T[0] = 0.0;
			local_T[1] = (var) cols;
			local_T[2] = (var) cols;
			local_T[3] = 0.0;
			local_T[4] = 0.0;
			local_T[5] = (var) cols;
			local_T[6] = (var) cols;
			local_T[7] = 0.0;

			// node 0 (left,bottom,near)
			n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,0.0,1.0);
			Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		  Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		  Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			// node 3 (left,top,near)
			n-=1;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,1.0,1.0);
			Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		  Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		  Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			// node 4 (left,bottom,far)
			n+=1+model->m_nx*model->m_ny;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,0.0,0.0);
			Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		  Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		  Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			// node 7 (left,top,far)
			n-=1;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,1.0,0.0);
			Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		  Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		  Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];
			
			local_T[0] = -((var) cols);
			local_T[1] = 0.0;
			local_T[2] = 0.0;
			local_T[3] = -((var) cols);
			local_T[4] = -((var) cols);
			local_T[5] = 0.0;
			local_T[6] = 0.0;
			local_T[7] = -((var) cols);
			
			// node 1 (right,bottom,near)
			n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny + model->m_ny;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,0.0,1.0);
			Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		  Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		  Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			// node 2 (right,top,near)
			n-=1;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,1.0,1.0);
			Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		  Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		  Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			// node 5 (right,bottom,far)
			n+=1+model->m_nx*model->m_ny;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,0.0,0.0);
			Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		  Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		  Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			// node 6 (right,top,far)
			n-=1;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,1.0,0.0);
			Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		  Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		  Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];
		}
		}
  } else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (unsigned int l=0; l<lays; l++){
			for (unsigned int c=0; c<cols; c++){
				e = c*rows + l*rowscols;
			  k = model->props[model->elem_material_map[e]];

			  local_T[0] = 0.0;
			  local_T[1] = 0.0;
			  local_T[2] = (var) rows;
			  local_T[3] = (var) rows;
			  local_T[4] = 0.0;
			  local_T[5] = 0.0;
			  local_T[6] = (var) rows;
			  local_T[7] = (var) rows;

			  // node 0 (left,bottom,near)
			  n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
			  dof = model->node_dof_map[n];
			  fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,0.0,1.0);
			  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			  // node 1 (right,bottom,near)
				n+=model->m_ny;
			  dof = model->node_dof_map[n];
			  fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,0.0,1.0);
			  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			  // node 4 (left,bottom,far)
			  n+=(model->m_nx-1)*model->m_ny;
			  dof = model->node_dof_map[n];
			  fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,0.0,0.0);
			  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			  // node 5 (right,bottom,far)
				n+=model->m_ny;
			  dof = model->node_dof_map[n];
			  fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,0.0,0.0);
			  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];
			  
			  local_T[0] = -((var) rows);
			  local_T[1] = -((var) rows);
			  local_T[2] = 0.0;
			  local_T[3] = 0.0;
			  local_T[4] = -((var) rows);
			  local_T[5] = -((var) rows);
			  local_T[6] = 0.0;
			  local_T[7] = 0.0;
			  
			  // node 2 (right,top,near)
				n -= 1+model->m_nx*model->m_ny;
			  dof = model->node_dof_map[n];
			  fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,1.0,1.0);
			  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			  // node 3 (left,top,near)
				n-=model->m_ny;
			  dof = model->node_dof_map[n];
			  fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,1.0,1.0);
			  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			  // node 6 (right,top,far)
				n+=(model->m_nx+1)*model->m_ny;
			  dof = model->node_dof_map[n];
			  fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,1.0,0.0);
			  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

			  // node 7 (left,top,far)
				n-=model->m_ny;
			  dof = model->node_dof_map[n];
			  fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,1.0,0.0);
			  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
		    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
		    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];
		}
		}
  }  else if (model->m_hmg_flag == HOMOGENIZE_Z){
		for (e=0; e<rowscols; e++){
			k = model->props[model->elem_material_map[e]];

			local_T[0] = (var) lays;
			local_T[1] = (var) lays;
			local_T[2] = (var) lays;
			local_T[3] = (var) lays;
			local_T[4] = 0.0;
			local_T[5] = 0.0;
			local_T[6] = 0.0;
			local_T[7] = 0.0;

			// node 4 (left,bottom,far)
			n = 1+(e%rowscols)+((e%rowscols)/rows)+(1+e/rowscols)*model->m_nx*model->m_ny;
		  dof = model->node_dof_map[n];
		  fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,0.0,0.0);
		  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
	    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
	    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		  // node 5 (right,bottom,far)
			n+=model->m_ny;
		  dof = model->node_dof_map[n];
		  fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,0.0,0.0);
		  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
	    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
	    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		  // node 6 (right,top,far)
			n-=1;
		  dof = model->node_dof_map[n];
		  fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,1.0,0.0);
		  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
	    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
	    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		  // node 7 (left,top,far)
			n-=model->m_ny;
		  dof = model->node_dof_map[n];
		  fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,1.0,0.0);
		  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
	    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
	    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];
			
			local_T[0] = 0.0;
			local_T[1] = 0.0;
			local_T[2] = 0.0;
			local_T[3] = 0.0;
			local_T[4] = -((var) lays);
			local_T[5] = -((var) lays);
			local_T[6] = -((var) lays);
			local_T[7] = -((var) lays);
			
			// node 0 (left,bottom,near)
			n = 1+(e%rowscols)+((e%rowscols)/rows)+(e/rowscols)*model->m_nx*model->m_ny;
		  dof = model->node_dof_map[n];
		  fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,0.0,1.0);
		  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
	    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
	    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		  // node 1 (right,bottom,near)
			n+=model->m_ny;
		  dof = model->node_dof_map[n];
		  fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,0.0,1.0);
		  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
	    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
	    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		  // node 2 (right,top,near)
			n-=1;
		  dof = model->node_dof_map[n];
		  fluxFromTemp_thermal_3D(local_Q,local_T,k,1.0,1.0,1.0);
		  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
	    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
	    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];

		  // node 3 (left,top,near)
			n-=model->m_ny;
		  dof = model->node_dof_map[n];
		  fluxFromTemp_thermal_3D(local_Q,local_T,k,0.0,1.0,1.0);
		  Q[3*dof]   += (cudapcgVar_t) 0.125*local_Q[0];
	    Q[3*dof+1] += (cudapcgVar_t) 0.125*local_Q[1];
	    Q[3*dof+2] += (cudapcgVar_t) 0.125*local_Q[2];
		}
  }

  // Save arrays to binary files
  char str_buffer[1024];
  sprintf(str_buffer,"%s_temperature_%d.bin",model->neutralFile_noExt,model->m_hmg_flag);
  FILE * file = fopen(str_buffer,"wb");
  if (file)
    fwrite(T,sizeof(cudapcgVar_t)*model->m_ndof,1,file);
  fclose(file);

  sprintf(str_buffer,"%s_flux_%d.bin",model->neutralFile_noExt,model->m_hmg_flag);
  file = fopen(str_buffer,"wb");
  if (file)
    fwrite(Q,sizeof(cudapcgVar_t)*model->m_ndof*3,1,file);
  fclose(file);

  free(Q);

  return;
}
//------------------------------------------------------------------------------
void saveFields_thermal_3D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * T){
  printf("WARNING: Field exportation not supported for scalar field input (.bin) yet.\n");
  return;
}
//------------------------------------------------------------------------------
