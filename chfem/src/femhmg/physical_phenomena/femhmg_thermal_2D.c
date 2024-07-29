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
	model->m_nporeelem = model->m_nelem; // pore elems comprehend the pore space, the domain. as mesh is monolithic, porelem=elem.
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
	var lcl_C=0.0, C_i=0.0, C_j=0.0;

	unsigned int i,j;
	if (model->m_hmg_flag == HOMOGENIZE_X){
		i = 0; j = 2;
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		i = 1; j = 3;
	}

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
		lcl_C = model->props[model->elem_material_map[e]]*0.5*(model->m_nx-1);

		// node 1 (right,bottom)
		C_i = lcl_C; C_j = -lcl_C;

		// node 2 (right,top)
		C_i += lcl_C; C_j += lcl_C;

		#pragma omp critical
		{
			model->C[i] += lcl_C; model->C[j] += lcl_C;
		}
		*/
	  #pragma omp parallel for reduction(+:lcl_C)
	  for (e=model->m_nelem-model->m_ny+1;e<model->m_nelem;e++){
		  lcl_C += model->props[model->elem_material_map[e]]*(model->m_nx-1);
	  }
		C_i += lcl_C;

	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		// Analogous to HOMOGENIZE_X (uses nodes 2 and 3)
	  #pragma omp parallel for reduction(+:lcl_C)
	  for (e=0;e<model->m_nelem;e+=(model->m_ny-1)){
		  lcl_C += model->props[model->elem_material_map[e]]*(model->m_ny-1);
	  }
		C_j += lcl_C;
	}

	#pragma omp parallel for private(n,coeff,lcl_C) reduction(+:C_i,C_j)
	for (e=0;e<model->m_nelem;e++){

		// Compute coefficient of this element's flux matrix
		coeff = model->props[model->elem_material_map[e]]*0.5;

		// node 0 (left,bottom)
		n = e+1+(e/dim_y);
		lcl_C = coeff*T[model->node_dof_map[n]];
		C_i += -lcl_C;
		C_j += -lcl_C;

		// node 1 (right,bottom)
		n+=model->m_ny;
		lcl_C = coeff*T[model->node_dof_map[n]];
		C_i +=  lcl_C;
		C_j += -lcl_C;

		// node 2 (right,top)
		n-=1;
		lcl_C = coeff*T[model->node_dof_map[n]];
		C_i += lcl_C;
		C_j += lcl_C;

		// node 3 (left,top)
		n-=model->m_ny;
		lcl_C = coeff*T[model->node_dof_map[n]];
		C_i += -lcl_C;
		C_j +=  lcl_C;
	}
	
	model->C[i] = C_i; model->C[j] = C_j;

	model->C[i] /= model->m_nelem; model->C[j] /= model->m_nelem;

	return;
}
//------------------------------------------------------------------------------
void updateC_thermal_2D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * T){
	unsigned int n;
	unsigned int e;
	unsigned int dim_y = model->m_ny-1;
	var coeff;
	var lcl_C=0.0, C_i=0.0, C_j=0.0;

	unsigned int i,j;
	if (model->m_hmg_flag == HOMOGENIZE_X){
		i = 0; j = 2;
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		i = 1; j = 3;
	}

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
		lcl_C = model->props[model->elem_material_map[e]]*0.5*(model->m_nx-1);

		// node 1 (right,bottom)
		C_i = lcl_C; C_j = -lcl_C;

		// node 2 (right,top)
		C_i += lcl_C; C_j += lcl_C;

		#pragma omp critical
		{
			model->C[i] += lcl_C; model->C[j] += lcl_C;
		}
		*/
	  #pragma omp parallel for reduction(+:lcl_C)
	  for (e=model->m_nelem-model->m_ny+1;e<model->m_nelem;e++){
		  lcl_C += (model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min)*(model->m_nx-1);
	  }
		C_i += lcl_C;

	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		// Analogous to HOMOGENIZE_X (uses nodes 2 and 3)
	  #pragma omp parallel for reduction(+:lcl_C)
	  for (e=0;e<model->m_nelem;e+=(model->m_ny-1)){
		  lcl_C += (model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min)*(model->m_ny-1);
	  }
		C_j += lcl_C;
	}

	#pragma omp parallel for private(n,coeff,lcl_C) reduction(+:C_i,C_j)
	for (e=0;e<model->m_nelem;e++){

		// Compute coefficient of this element's flux matrix
		coeff = (model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min)*0.5;

		// node 0 (left,bottom)
		n = e+1+(e/dim_y);
		lcl_C = coeff*T[model->node_dof_map[n]];
		C_i += -lcl_C;
		C_j += -lcl_C;

		// node 1 (right,bottom)
		n+=model->m_ny;
		lcl_C = coeff*T[model->node_dof_map[n]];
		C_i +=  lcl_C;
		C_j += -lcl_C;

		// node 2 (right,top)
		n-=1;
		lcl_C = coeff*T[model->node_dof_map[n]];
		C_i += lcl_C;
		C_j += lcl_C;

		// node 3 (left,top)
		n-=model->m_ny;
		lcl_C = coeff*T[model->node_dof_map[n]];
		C_i += -lcl_C;
		C_j +=  lcl_C;
	}
	
	model->C[i] = C_i; model->C[j] = C_j;

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
void fluxFromTemp_thermal_2D(var *q, var *t, var k, var x, var y){

	var B[8] = { -(1.0-y), (1.0-y), y,     -y ,
	             -(1.0-x),     -x , x, (1.0-x) };

	q[0] = 0.0;
	q[1] = 0.0;
	for (unsigned int i=0; i<4; i++){
		q[0] +=    B[i] * t[i];
		q[1] +=  B[i+4] * t[i];
	}
	q[0] *= k;
	q[1] *= k;

	return;
}
//------------------------------------------------------------------------------
void saveFields_thermal_2D(hmgModel_t *model, cudapcgVar_t * T){

  cudapcgVar_t * Q = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*model->m_ndof*2);
  for (unsigned int i=0; i<model->m_ndof*2; i++){
    Q[i] = 0.0;
  }

  unsigned int rows = model->m_ny-1;
  unsigned int cols = model->m_nx-1;
  unsigned int rowscols = rows*cols;
  
  unsigned int dof, n;

  var local_Q[2];
  var local_T[4];
  
  var k;
  
  #pragma omp parallel for private(local_Q,local_T,n,dof,k)
  for (unsigned int e=0;e<model->m_nelem; e++){

		k = model->props[model->elem_material_map[e]];

		// node 0 (left,bottom)
		n = e+1+(e/rows);
		dof = model->node_dof_map[n];
		local_T[0] = (var) T[dof];

		// node 1 (right,bottom)
		n+=model->m_ny;
		dof = model->node_dof_map[n];
		local_T[1] = (var) T[dof];

		// node 2 (right,top)
		n-=1;
		dof = model->node_dof_map[n];
		local_T[2] = (var) T[dof];

		// node 3 (left,top)
		n-=model->m_ny;
		dof = model->node_dof_map[n];
		local_T[3] = (var) T[dof];

		fluxFromTemp_thermal_2D(local_Q,local_T,k,0.0,1.0);
		Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];

		// node 0 (left,bottom)
		n = e+1+(e/rows);
		dof = model->node_dof_map[n];
		fluxFromTemp_thermal_2D(local_Q,local_T,k,0.0,0.0);
		Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];

		// node 1 (right,bottom)
		n+=model->m_ny;
		dof = model->node_dof_map[n];
		fluxFromTemp_thermal_2D(local_Q,local_T,k,1.0,0.0);
		Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];

		// node 2 (right,top)
		n-=1;
		dof = model->node_dof_map[n];
		fluxFromTemp_thermal_2D(local_Q,local_T,k,1.0,1.0);
		Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];
	}
	
	// Compensate for periodic borders
	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (unsigned int e=(rowscols-rows); e<rowscols; e++){
			k = model->props[model->elem_material_map[e]];

			local_T[0] = 0.0;
			local_T[1] = (var) cols;
			local_T[2] = (var) cols;
			local_T[3] = 0.0;

			// node 0 (left,bottom)
			n = e+1+(e/rows);
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_2D(local_Q,local_T,k,0.0,0.0);
		  Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		  Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];

			// node 3 (left,top)
			n-=1;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_2D(local_Q,local_T,k,0.0,1.0);
		  Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		  Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];
		  
		  local_T[0] = -((var) cols);
			local_T[1] = 0.0;
			local_T[2] = 0.0;;
			local_T[3] = -((var) cols);

			// node 1 (right,bottom)
			n = e+1+(e/rows) + model->m_ny;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_2D(local_Q,local_T,k,1.0,0.0);
		  Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		  Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];

			// node 2 (right,top)
			n-=1;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_2D(local_Q,local_T,k,1.0,1.0);
		  Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		  Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];
		}
  } else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (unsigned int e=0; e<rowscols; e+=rows){//(rows-1)
			k = model->props[model->elem_material_map[e]];

			local_T[0] = 0.0;
			local_T[1] = 0.0;
			local_T[2] = (var) rows;
			local_T[3] = (var) rows;

			// node 0 (left,bottom)
			n = e+1+(e/rows);
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_2D(local_Q,local_T,k,0.0,0.0);
		  Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		  Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];

			// node 1 (right,bottom)
			n+=model->m_ny;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_2D(local_Q,local_T,k,1.0,0.0);
		  Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		  Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];

			local_T[0] = -(var) rows;
			local_T[1] = -(var) rows;
			local_T[2] = 0.0;
			local_T[3] = 0.0;

			// node 3 (left,top)
			n = e+(e/rows);
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_2D(local_Q,local_T,k,0.0,1.0);
		  Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		  Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];

			// node 2 (right,top)
			n+=model->m_ny;
			dof = model->node_dof_map[n];
			fluxFromTemp_thermal_2D(local_Q,local_T,k,1.0,1.0);
		  Q[2*dof]   += (cudapcgVar_t) 0.25*local_Q[0];
		  Q[2*dof+1] += (cudapcgVar_t) 0.25*local_Q[1];
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
