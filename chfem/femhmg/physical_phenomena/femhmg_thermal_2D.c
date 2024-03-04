#include "femhmg_2D.h"
#include "femhmg_thermal_2D.h"

//------------------------------------------------------------------------------
logical initModel_thermal_2D(hmgModel_t *model){
	if ((model->m_nx-1) <= 0 || (model->m_ny-1) <= 0 || model->m_nmat <= 0){
	  printf("ERROR: Failed to initialize model due to bad input parameters.\nrows:%d\ncols:%d\nmaterials:%d\n",(model->m_ny-1),(model->m_nx-1),model->m_nmat);
		return HMG_FALSE;
	}

	model->m_C_dim = 4;
	model->m_lclMtx_dim = 16;
	model->m_lclCB_dim = 0;

	model->m_nnodedof = 1;
	model->m_nelemdof = 4;
	model->m_nnode = model->m_nx * model->m_ny;
	model->m_nelem = (model->m_nx-1) * (model->m_ny-1);
	model->m_ndof = model->m_nelem;

  if (model->sdfFile==NULL){
	  model->assembleRHS = assembleRHS_thermal_2D;
	  model->updateC = updateC_thermal_2D;
	  model->saveFields = saveFields_thermal_2D;
	} else {
	  model->assembleRHS = assembleRHS_thermal_2D_ScalarDensityField;
	  model->updateC = updateC_thermal_2D_ScalarDensityField;
	  model->saveFields = saveFields_thermal_2D_ScalarDensityField;
	}
	
	model->assembleLocalMtxs = assembleLocalMtxs_thermal_2D;
	model->printC = printC_thermal_2D;

	model->assembleNodeDofMap = assembleNodeDofMap_2D;
	model->assembleDofIdMap = NULL;
	model->assembleDofMaterialMap = assembleDofMaterialMap_2D;

	return HMG_TRUE;
}
//------------------------------------------------------------------------------
void assembleLocalMtxs_thermal_2D(hmgModel_t *model){

	/*
	  Analytical solution for pixel-based mesh:

	  k_el = material_prop * {
	    {  2/3, -1/6, -1/3, -1/6 },
	    { -1/6,  2/3, -1/6, -1/3 },
	    { -1/3, -1/6,  2/3, -1/6 },
	    { -1/6, -1/3, -1/6,  2/3 }
	  }
	*/

	cudapcgVar_t diag = 2.0/3.0;
	cudapcgVar_t out_of_diag_16 = -1.0/6.0;
	cudapcgVar_t out_of_diag_13 = -1.0/3.0;
	cudapcgVar_t thisProp;

	unsigned int i,mat;
	for (i=0;i<model->m_nmat;i++){

		thisProp = model->props[i];
		mat = i * model->m_lclMtx_dim;

		model->Mtxs[mat]    = thisProp * diag;
		model->Mtxs[mat+1]  = thisProp * out_of_diag_16;
		model->Mtxs[mat+2]  = thisProp * out_of_diag_13;
		model->Mtxs[mat+3]  = thisProp * out_of_diag_16;

		model->Mtxs[mat+4]  = thisProp * out_of_diag_16;
		model->Mtxs[mat+5]  = thisProp * diag;
		model->Mtxs[mat+6]  = thisProp * out_of_diag_16;
		model->Mtxs[mat+7]  = thisProp * out_of_diag_13;

		model->Mtxs[mat+8]  = thisProp * out_of_diag_13;
		model->Mtxs[mat+9]  = thisProp * out_of_diag_16;
		model->Mtxs[mat+10] = thisProp * diag;
		model->Mtxs[mat+11] = thisProp * out_of_diag_16;

		model->Mtxs[mat+12] = thisProp * out_of_diag_16;
		model->Mtxs[mat+13] = thisProp * out_of_diag_13;
		model->Mtxs[mat+14] = thisProp * out_of_diag_16;
		model->Mtxs[mat+15] = thisProp * diag;
	}
	return;
}
//------------------------------------------------------------------------------
void assembleRHS_thermal_2D(hmgModel_t *model){

	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	unsigned int dim_xy = dim_x*dim_y;

	unsigned int i;
  #pragma omp parallel for
  for (i=0; i<dim_xy; i++)
      model->RHS[i] = 0.0;

	unsigned int e,n;
	cudapcgVar_t * thisK  = NULL;

	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (i=0; i<dim_y; i++){
			e = (model->m_nx-2)*dim_y+i;

		  thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			
			n = e+1+(e/dim_y);
			model->RHS[model->node_dof_map[n]] -= (thisK[1]+thisK[2])*dim_x;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[5]+thisK[6])*dim_x;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= (thisK[9]+thisK[10])*dim_x;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[13]+thisK[14])*dim_x;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (i=0; i<dim_x; i++){
			e = i*dim_y;

			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);

			n = e+1+(e/dim_y);
			model->RHS[model->node_dof_map[n]] -= (thisK[2]+thisK[3])*dim_y;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[6]+thisK[7])*dim_y;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= (thisK[10]+thisK[11])*dim_y;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= (thisK[14]+thisK[15])*dim_y;
		}
	}

	return;
}

//------------------------------------------------------------------------------
void assembleRHS_thermal_2D_ScalarDensityField(hmgModel_t *model){

	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	unsigned int dim_xy = dim_x*dim_y;

	unsigned int i;
  #pragma omp parallel for
  for (i=0; i<dim_xy; i++)
      model->RHS[i] = 0.0;

	unsigned int e,n;
	cudapcgVar_t * thisK  = &(model->Mtxs[0]);
	cudapcgVar_t scl=1.0;

	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (i=0; i<dim_y; i++){
			e = (model->m_nx-2)*dim_y+i;

			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			n = e+1+(e/dim_y);
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[1]+thisK[2])*dim_x;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[5]+thisK[6])*dim_x;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[9]+thisK[10])*dim_x;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[13]+thisK[14])*dim_x;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (i=0; i<dim_x; i++){
			e = i*dim_y;

			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			n = e+1+(e/dim_y);
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[2]+thisK[3])*dim_y;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[6]+thisK[7])*dim_y;

			n -= 1;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[10]+thisK[11])*dim_y;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]] -= scl*(thisK[14]+thisK[15])*dim_y;
		}
	}

	return;
}

//------------------------------------------------------------------------------
void updateC_thermal_2D(hmgModel_t *model, cudapcgVar_t * T){
	unsigned int n;
	unsigned int e;
	unsigned int dim_y = model->m_ny-1;
	var coeff;
	var C_lcl, C_i, C_j;

	unsigned int i,j;
	if (model->m_hmg_flag == HOMOGENIZE_X){
		i = 0; j = 2;
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		i = 1; j = 3;
	}

	C_i = 0.0; C_j = 0.0;
	C_lcl = 0.0;

	/*
		ATTENTION:
		The coefficients 0.5 and -0.5 used on the operations to compute model->C
		come from the analytical solution for the flux matrix of a quad
		element on a regular pixel-based mesh.
	*/

	if (model->m_hmg_flag == HOMOGENIZE_X){
		// This loop does the same as
		/*
		// Compute local component to add in model->C
		C_lcl = model->props[model->elem_material_map[e]]*0.5*(model->m_nx-1);

		// node 1 (right,bottom)
		C_i = C_lcl; C_j = -C_lcl;

		// node 2 (right,top)
		C_i += C_lcl; C_j += C_lcl;

		#pragma omp critical
		{
			model->C[i] += C_i; model->C[j] += C_j;
		}
		*/
	  #pragma omp parallel for reduction(+:C_lcl)
	  for (e=model->m_nelem-model->m_ny+1;e<model->m_nelem;e++){
		  C_lcl += model->props[model->elem_material_map[e]]*(model->m_nx-1);
	  }
		model->C[i] += C_lcl;

	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		// Analogous to HOMOGENIZE_X (uses nodes 2 and 3)
	  #pragma omp parallel for reduction(+:C_lcl)
	  for (e=0;e<model->m_nelem;e+=(model->m_ny-1)){
		  C_lcl += model->props[model->elem_material_map[e]]*(model->m_ny-1);
	  }
		model->C[j] += C_lcl;
	}

	#pragma omp parallel for private(C_i,C_j,n,coeff,C_lcl)
	for (e=0;e<model->m_nelem;e++){

		// Compute coefficient of this element's flux matrix
		coeff = model->props[model->elem_material_map[e]]*0.5;

		// node 0 (left,bottom)
		n = e+1+(e/dim_y);
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i = -C_lcl; C_j = -C_lcl;

		// node 1 (right,bottom)
		n+=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j -= C_lcl;

		// node 2 (right,top)
		n-=1;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j += C_lcl;

		// node 3 (left,top)
		n-=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i -= C_lcl; C_j += C_lcl;

		#pragma omp critical
		{
			model->C[i] += C_i; model->C[j] += C_j;
		}
	}

	model->C[i] /= model->m_nelem; model->C[j] /= model->m_nelem;

	return;
}
//------------------------------------------------------------------------------
void updateC_thermal_2D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * T){
	unsigned int n;
	unsigned int e;
	unsigned int dim_y = model->m_ny-1;
	var coeff;
	var C_lcl, C_i, C_j;

	unsigned int i,j;
	if (model->m_hmg_flag == HOMOGENIZE_X){
		i = 0; j = 2;
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		i = 1; j = 3;
	}

	C_i = 0.0; C_j = 0.0;
	C_lcl = 0.0;

	/*
		ATTENTION:
		The coefficients 0.5 and -0.5 used on the operations to compute model->C
		come from the analytical solution for the flux matrix of a quad
		element on a regular pixel-based mesh.
	*/

	if (model->m_hmg_flag == HOMOGENIZE_X){
		// This loop does the same as
		/*
		// Compute local component to add in model->C
		C_lcl = model->props[model->elem_material_map[e]]*0.5*(model->m_nx-1);

		// node 1 (right,bottom)
		C_i = C_lcl; C_j = -C_lcl;

		// node 2 (right,top)
		C_i += C_lcl; C_j += C_lcl;

		#pragma omp critical
		{
			model->C[i] += C_i; model->C[j] += C_j;
		}
		*/
	  #pragma omp parallel for reduction(+:C_lcl)
	  for (e=model->m_nelem-model->m_ny+1;e<model->m_nelem;e++){
		  C_lcl += (model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min)*(model->m_nx-1);
	  }
		model->C[i] += C_lcl;

	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		// Analogous to HOMOGENIZE_X (uses nodes 2 and 3)
	  #pragma omp parallel for reduction(+:C_lcl)
	  for (e=0;e<model->m_nelem;e+=(model->m_ny-1)){
		  C_lcl += (model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min)*(model->m_ny-1);
	  }
		model->C[j] += C_lcl;
	}

	#pragma omp parallel for private(C_i,C_j,n,coeff,C_lcl)
	for (e=0;e<model->m_nelem;e++){

		// Compute coefficient of this element's flux matrix
		coeff = (model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min)*0.5;

		// node 0 (left,bottom)
		n = e+1+(e/dim_y);
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i = -C_lcl; C_j = -C_lcl;

		// node 1 (right,bottom)
		n+=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j -= C_lcl;

		// node 2 (right,top)
		n-=1;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i += C_lcl; C_j += C_lcl;

		// node 3 (left,top)
		n-=model->m_ny;
		C_lcl = coeff*T[model->node_dof_map[n]];
		C_i -= C_lcl; C_j += C_lcl;

		#pragma omp critical
		{
			model->C[i] += C_i; model->C[j] += C_j;
		}
	}

	model->C[i] /= model->m_nelem; model->C[j] /= model->m_nelem;

	return;
}
//------------------------------------------------------------------------------
void printC_thermal_2D(hmgModel_t *model, char *dest){
	if (dest==NULL){
	  printf("-------------------------------------------------------\n");
	  printf("Homogenized Constitutive Matrix (Thermal Conductivity):\n");
	  printf("  %.8e  ", model->C[0]); printf("%.8e\n", model->C[1]);
	  printf("  %.8e  ", model->C[2]); printf("%.8e\n", model->C[3]);
	  printf("-------------------------------------------------------\n");
	} else {
	  sprintf(
      dest,
      "-------------------------------------------------------\n"\
      "Homogenized Constitutive Matrix (Thermal Conductivity):\n"\
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
void saveFields_thermal_2D(hmgModel_t *model, cudapcgVar_t * T){

  cudapcgVar_t * Q = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*model->m_ndof*2);

  unsigned int rows = model->m_ny-1;
  unsigned int cols = model->m_nx-1;
  unsigned int rowscols = rows*cols;

  unsigned int n_front, n_back;

  // Grad(T) X
  #pragma omp parallel for private(n_front,n_back)
  for (unsigned int n=0; n<model->m_ndof; n++){
    n_front = (n+rows)%rowscols; // WALK_RIGHT
    n_back  = (n+(cols-1)*rows)%rowscols; // WALK_LEFT
    Q[2*n] = 0.5*(T[n_front]-T[n_back])/model->m_elem_size;
  }

  // Grad(T) Y
  #pragma omp parallel for private(n_front,n_back)
  for (unsigned int n=0; n<model->m_ndof; n++){
    n_front = n+(-1+rows*(!(n%rows))); // WALK_UP
    n_back  = n+( 1-rows*(!((n+1)%rows))); // WALK_DOWN
    Q[2*n+1] = 0.5*(T[n_front]-T[n_back])/model->m_elem_size;
  }

	// Compensate for periodic borders
  if (model->m_hmg_flag == HOMOGENIZE_X){
		for (unsigned int n=0; n<rows; n++){
			Q[2*n] += 0.5*cols/model->m_elem_size;
		}
		for (unsigned int n=(rowscols-rows); n<rowscols; n++){
			Q[2*n] += 0.5*cols/model->m_elem_size;
		}
  } else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (unsigned int n=0; n<(rowscols-rows+1); n+=rows){
			Q[2*n+1] += 0.5*rows/model->m_elem_size;
		}
		for (unsigned int n=(rows-1); n<rowscols; n+=rows){
			Q[2*n+1] += 0.5*rows/model->m_elem_size;
		}
  }

  // Q = -kappa * Grad(T)
  cudapcgMap_t matkey;
  var kappa;
  #pragma omp parallel for private(matkey,kappa)
  for (unsigned int n=0; n<model->m_ndof; n++){
    matkey = model->dof_material_map[n];
    kappa  = model->props[matkey&MATKEY_BITSTEP_RANGE_2D];
    kappa += model->props[(matkey>>=MATKEY_BITSTEP_2D)&MATKEY_BITSTEP_RANGE_2D];
    kappa += model->props[(matkey>>=MATKEY_BITSTEP_2D)&MATKEY_BITSTEP_RANGE_2D];
    kappa += model->props[(matkey>>=MATKEY_BITSTEP_2D)&MATKEY_BITSTEP_RANGE_2D];
    Q[2*n]   *= -0.25*kappa;
    Q[2*n+1] *= -0.25*kappa;
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
    fwrite(Q,sizeof(cudapcgVar_t)*model->m_ndof*2,1,file);
  fclose(file);

  free(Q);

  return;
}
//------------------------------------------------------------------------------
void saveFields_thermal_2D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * T){
  printf("WARNING: Field exportation not supported for scalar field input (.bin) yet.\n");
  return;
}
//------------------------------------------------------------------------------
