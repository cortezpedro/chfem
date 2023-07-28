#include "femhmg_2D.h"
#include "femhmg_elastic_2D.h"

//------------------------------------------------------------------------------
void strainFromDispl_elastic_2D(var *s, var *d, var x, var y);
//------------------------------------------------------------------------------
void stressFromDispl_elastic_2D(var *s, var *d, var E, var v, var x, var y);
//------------------------------------------------------------------------------
void stressFromAlpha_elastic_2D(var *s, var *a, var E, var v, var x, var y);
//------------------------------------------------------------------------------
void forceFromStrain_elastic_2D(var *f, var *a, var E, var v, var x, var y);
//------------------------------------------------------------------------------

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
	
	if (model->sdfFile==NULL){
	  model->assembleRHS = assembleRHS_elastic_2D;
	  model->updateC = updateC_elastic_2D;
	  model->saveFields = saveFields_elastic_2D;
	} else {
	  model->assembleRHS = assembleRHS_elastic_2D_ScalarDensityField;
	  model->updateC = updateC_elastic_2D_ScalarDensityField;
	  model->saveFields = saveFields_elastic_2D_ScalarDensityField;
	}
	model->printC = printC_elastic_2D;

	model->assembleNodeDofMap = assembleNodeDofMap_2D;
	model->assembleDofIdMap = NULL;
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
	cudapcgVar_t * thisK=NULL;

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
	} else if (model->m_hmg_flag == HOMOGENIZE_THERMAL_EXPANSION){
	  var alpha[3] = {0.0,0.0,0.0};
	  var local_f[8] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
	  for (uint32_t e=0; e < model->m_nelem; e++){
	    alpha[0] = model->alpha[model->elem_material_map[e]];
	    alpha[1] = alpha[0];
	    alpha[2] = 0.0;
	    forceFromStrain_elastic_2D(&local_f[0],&alpha[0],model->props[2*model->elem_material_map[e]],model->props[2*model->elem_material_map[e]+1],0.5,0.5);
	    n = e+1+(e/dim_y);
			model->RHS[model->node_dof_map[n]]   += local_f[0];
			model->RHS[model->node_dof_map[n]+1] += local_f[1];
			n += model->m_ny;
			model->RHS[model->node_dof_map[n]]   += local_f[2];
			model->RHS[model->node_dof_map[n]+1] += local_f[3];
			n -= 1;
			model->RHS[model->node_dof_map[n]]   += local_f[4];
			model->RHS[model->node_dof_map[n]+1] += local_f[5];
			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]]   += local_f[6];
			model->RHS[model->node_dof_map[n]+1] += local_f[7];
	  }
	}

	return;
}
//------------------------------------------------------------------------------
void assembleRHS_elastic_2D_ScalarDensityField(hmgModel_t *model){

	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;

	unsigned int i;
    #pragma omp parallel for
    for (i=0; i<model->m_ndof; i++)
        model->RHS[i] = 0.0;

	unsigned int e,n;
	cudapcgVar_t * thisK = &(model->Mtxs[0]);
	cudapcgVar_t scl=1.0;

	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (i=0; i<dim_y; i++){
			e = (model->m_nx-2)*dim_y+i;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			n = e+1+(e/dim_y);
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[2]+thisK[4])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[10]+thisK[12])*dim_x;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[18]+thisK[20])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[26]+thisK[28])*dim_x;

			n -= 1;
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[34]+thisK[36])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[42]+thisK[44])*dim_x;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[50]+thisK[52])*dim_x;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[58]+thisK[60])*dim_x;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (i=0; i<dim_x; i++){
			e = i*dim_y;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			n = e+1+(e/dim_y);
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[5]+thisK[7])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[13]+thisK[15])*dim_y;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[21]+thisK[23])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[29]+thisK[31])*dim_y;

			n -= 1;
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[37]+thisK[39])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[45]+thisK[47])*dim_y;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[53]+thisK[55])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[61]+thisK[63])*dim_y;
		}
	} else if (model->m_hmg_flag == HOMOGENIZE_XY){
		for (i=0; i<dim_x; i++){
			e = i*dim_y;
			thisK = &(model->Mtxs[model->elem_material_map[e]*model->m_lclMtx_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			n = e+1+(e/dim_y);
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[4]+thisK[6])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[12]+thisK[14])*dim_y;

			n += model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[20]+thisK[22])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[28]+thisK[30])*dim_y;

			n -= 1;
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[36]+thisK[38])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[44]+thisK[46])*dim_y;

			n -= model->m_ny;
			model->RHS[model->node_dof_map[n]]   -= scl*(thisK[52]+thisK[54])*dim_y;
			model->RHS[model->node_dof_map[n]+1] -= scl*(thisK[60]+thisK[62])*dim_y;
		}
	}

	return;
}
//------------------------------------------------------------------------------
void updateC_elastic_2D_thermal_expansion(hmgModel_t *model, cudapcgVar_t * D){
  unsigned int n;
	unsigned int e;
	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	var d;
	cudapcgVar_t * thisCB=NULL;

	var stress[3] = {0.0,0.0,0.0};
	var strain[3] = {0.0,0.0,0.0};
	var local_d[8] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
	var local_stress[3]={0.0,0.0,0.0};
	var local_strain[3]={0.0,0.0,0.0};
	var local_alpha[3] = {0.0,0.0,0.0};

  var C_i=0.0, C_j=0.0, C_k=0.0;
  var *pC_i=&stress[0], *pC_j=&stress[1], *pC_k=&stress[2];
  
  // Compute effective stress (similar to updateC)
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

		#pragma omp atomic
		*pC_i += C_i;

		#pragma omp atomic
		*pC_j += C_j;

		#pragma omp atomic
		*pC_k += C_k;
	}

	// Compute effective strain and add thermal expansion contribution to effective stress
	for (e=0;e<model->m_nelem;e++){
		// node 0 (left,bottom)
		n = e+1+(e/dim_y);
		local_d[0] = D[model->node_dof_map[n]];
		local_d[1] = D[model->node_dof_map[n]+1];

		// node 1 (right,bottom)
		n+=model->m_ny;
		local_d[2] = D[model->node_dof_map[n]];
		local_d[3] = D[model->node_dof_map[n]+1];

		// node 2 (right,top)
		n-=1;
		local_d[4] = D[model->node_dof_map[n]];
		local_d[5] = D[model->node_dof_map[n]+1];

		// node 3 (left,top)
		n-=model->m_ny;
		local_d[6] = D[model->node_dof_map[n]];
		local_d[7] = D[model->node_dof_map[n]+1];

		local_strain[0] = 0.0;
		local_strain[1] = 0.0;
		local_strain[2] = 0.0;

		strainFromDispl_elastic_2D(&local_strain[0],&local_d[0],0.5,0.5);
		strain[0] += local_strain[0];
		strain[1] += local_strain[1];
		strain[2] += local_strain[2];
		
		local_alpha[0] = model->alpha[model->elem_material_map[e]];
	local_alpha[1] = local_alpha[0];
	local_alpha[2] = 0.0;
		
		local_stress[0] = 0.0;
		local_stress[1] = 0.0;
		local_stress[2] = 0.0;
		
		stressFromAlpha_elastic_2D(&local_stress[0],&local_alpha[0],model->props[2*model->elem_material_map[e]],model->props[2*model->elem_material_map[e]+1],0.5,0.5);
		stress[0] -= local_stress[0];
		stress[1] -= local_stress[1];
		stress[2] -= local_stress[2];
	}
	strain[0] /= (double) model->m_nelem;
	strain[1] /= (double) model->m_nelem;
	strain[2] /= (double) model->m_nelem;
	stress[0] /= (double) model->m_nelem;
	stress[1] /= (double) model->m_nelem;
	stress[2] /= (double) model->m_nelem;

	var beta[3];
	beta[0] = -stress[0] + model->C[0]*strain[0] + model->C[1]*strain[1] + model->C[2]*strain[2];
	beta[1] = -stress[1] + model->C[3]*strain[0] + model->C[4]*strain[1] + model->C[5]*strain[2];
	beta[2] = -stress[2] + model->C[6]*strain[0] + model->C[7]*strain[1] + model->C[8]*strain[2];

	// Solve 3x3 system with simple hardcoded gauss elim: alpha = C^-1 *beta
	var alpha[3], scl;
	var M[9] = { model->C[0],model->C[1],model->C[2],
					model->C[3],model->C[4],model->C[5],
					model->C[6],model->C[7],model->C[8] };
  
  scl = M[3]/M[0];
  M[4] -= M[1]*scl;
  M[5] -= M[2]*scl;
  beta[1] -= beta[0]*scl;
  scl = M[6]/M[0];
  M[7] -= M[1]*scl;
  M[8] -= M[2]*scl;
  beta[2] -= beta[0]*scl;
  M[3] = 0.0;
  M[6] = 0.0;
  scl = M[7]/M[4];
  M[8] -= M[5]*scl;
  beta[2] -= beta[1]*scl;
  M[7] = 0.0;
  
  alpha[2] = beta[2]/M[8];
  alpha[1] = (beta[1]-M[5]*alpha[2])/M[4];
  alpha[0] = (beta[0]-M[1]*alpha[1]-M[2]*alpha[2])/M[0];
	
	model->thermal_expansion[0] = alpha[0];
	model->thermal_expansion[1] = alpha[1];
	model->thermal_expansion[2] = alpha[2];

  return;
}
//------------------------------------------------------------------------------
void updateC_elastic_2D(hmgModel_t *model, cudapcgVar_t * D){

  if (model->m_hmg_flag == HOMOGENIZE_THERMAL_EXPANSION){
    updateC_elastic_2D_thermal_expansion(model,D);
    return;
  }
  
	unsigned int n;
	unsigned int e;
	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	var C_i, C_j, C_k;
	var d;
	cudapcgVar_t * thisCB=NULL;

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

	var *pC_i=&(model->C[i]), *pC_j=&(model->C[j]), *pC_k=&(model->C[k]);

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

		#pragma omp atomic
		*pC_i += C_i;

		#pragma omp atomic
		*pC_j += C_j;

		#pragma omp atomic
		*pC_k += C_k;
	}

	*pC_i /= model->m_nelem; *pC_j /= model->m_nelem; *pC_k /= model->m_nelem;

	return;
}
//------------------------------------------------------------------------------
void updateC_elastic_2D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * D){
	unsigned int n;
	unsigned int e;
	unsigned int dim_x = model->m_nx-1;
	unsigned int dim_y = model->m_ny-1;
	var C_i, C_j, C_k;
	var d;
	cudapcgVar_t * thisCB=NULL;
	cudapcgVar_t scl=1.0;

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

		#pragma omp parallel for private(C_i,C_j,C_k,thisCB,scl)
		for (e=model->m_nelem-model->m_ny+1;e<model->m_nelem;e++){

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			// node 1 (right,bottom)
			C_i = thisCB[2]; C_j = thisCB[10]; C_k = thisCB[18];

			// node 2 (right,top)
			C_i += thisCB[4]; C_j += thisCB[12]; C_k += thisCB[20];

			C_i *= scl*dim_x; C_j *= scl*dim_x; C_k *= scl*dim_x;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
			}
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_Y){

		#pragma omp parallel for private(C_i,C_j,C_k,thisCB,scl)
		for (e=0;e<model->m_nelem;e+=dim_y){

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			// node 2 (right,top)
			C_i = thisCB[5]; C_j = thisCB[13]; C_k = thisCB[21];

			// node 3 (left,top)
			C_i += thisCB[7]; C_j += thisCB[15]; C_k += thisCB[23];

			C_i *= scl*dim_y; C_j *= scl*dim_y; C_k *= scl*dim_y;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
			}
		}

	} else if (model->m_hmg_flag == HOMOGENIZE_XY){

		#pragma omp parallel for private(C_i,C_j,C_k,thisCB,scl)
		for (e=0;e<model->m_nelem;e+=dim_y){

			thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);
			scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

			// node 2 (right,top)
			C_i = thisCB[4]; C_j = thisCB[12]; C_k = thisCB[20];

			// node 3 (left,top)
			C_i += thisCB[6]; C_j += thisCB[14]; C_k += thisCB[22];

			C_i *= scl*dim_y; C_j *= scl*dim_y; C_k *= scl*dim_y;

			#pragma omp critical
			{
				model->C[i] += C_i; model->C[j] += C_j; model->C[k] += C_k;
			}
		}
	}

	#pragma omp parallel for private(C_i,C_j,C_k,d,thisCB,n,scl)
	for (e=0;e<model->m_nelem;e++){

		thisCB = &(model->CB[model->elem_material_map[e]*model->m_lclCB_dim]);
		scl = (cudapcgVar_t)(model->density_map[e]*(1.0/65535.0)*(model->density_max-model->density_min) + model->density_min);

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
		
		C_i *= scl; C_j *= scl; C_k *= scl;

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
	  if (model->m_hmg_thermal_expansion_flag == HMG_TRUE){
	    printf("Effective thermal expansion:\n");
	    printf("  %.8e  %.8e\n",model->thermal_expansion[0],model->thermal_expansion[2]);
	    printf("  %.8e  %.8e\n",model->thermal_expansion[2],model->thermal_expansion[1]);
	    printf("-------------------------------------------------------\n");
	  }
	} else {
	  if (model->m_hmg_thermal_expansion_flag == HMG_FALSE){
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
	  } else {
	    sprintf(
        dest,
        "-------------------------------------------------------\n"\
        "Homogenized Constitutive Matrix (Elasticity):\n"\
        "  %.8e  %.8e  %.8e\n"\
        "  %.8e  %.8e  %.8e\n"\
        "  %.8e  %.8e  %.8e\n"\
        "-------------------------------------------------------\n"\
        "Effective thermal expansion:\n"\
        "  %.8e  %.8e\n"\
        "  %.8e  %.8e\n"\
        "-------------------------------------------------------\n",
        model->C[0], model->C[1], model->C[2],
        model->C[3], model->C[4], model->C[5],
        model->C[6], model->C[7], model->C[8],
        model->thermal_expansion[0],model->thermal_expansion[2],
        model->thermal_expansion[2],model->thermal_expansion[1]
      );
	  }
	  
  }
	return;
}
//------------------------------------------------------------------------------
void strainFromDispl_elastic_2D(var *s, var *d, var x, var y){

	var B[24] = { -(1.0-y), 0.0, (1.0-y), 0.0, y, 0.0, -y, 0.0,
	              0.0, -(1.0-x), 0.0, -x, 0.0, x, 0.0, (1.0-x),
				 -(1.0-x), -(1.0-y), -x, (1.0-y), x, y, (1.0-x), -y };

	s[0] = 0.0;
	s[1] = 0.0;
	s[2] = 0.0;
	for (unsigned int i=0; i<8; i++){
		s[0] +=    B[i] * d[i];
		s[1] +=  B[i+8] * d[i];
		s[2] += B[i+16] * d[i];
	}

	return;
}
//------------------------------------------------------------------------------
void stressFromDispl_elastic_2D(var *s, var *d, var E, var v, var x, var y){

	#ifdef ELASTIC_2D_PLANESTRESS
        var coeff = E/ (1.0-v);
	    var C[9] = { coeff, coeff*v, 0.0,
		             coeff*v, coeff, 0.0,
					  0.0, 0.0, coeff*0.5*(1.0-v) };
	#else // ELASTIC_2D_PLANESTRAIN
		var coeff = E/ ((1.0+v)*(1.0-2.0*v));
		var C[9] = { coeff*(1.0-v), coeff*v, 0.0,
					 coeff*v, coeff*(1.0-v), 0.0,
					 0.0, 0.0, coeff*0.5*(1.0-2.0*v) };
	#endif

	var B[24] = { -(1.0-y), 0.0, (1.0-y), 0.0, y, 0.0, -y, 0.0,
	              0.0, -(1.0-x), 0.0, -x, 0.0, x, 0.0, (1.0-x),
				 -(1.0-x), -(1.0-y), -x, (1.0-y), x, y, (1.0-x), -y };

	var c_x, c_y, c_xy;
	for (unsigned int j=0; j<3; j++){
		c_x  = 0.0;
		c_y  = 0.0;
		c_xy = 0.0;
		for (unsigned int i=0; i<8; i++){
			c_x  +=    B[i] * d[i];
			c_y  +=  B[i+8] * d[i];
			c_xy += B[i+16] * d[i];
		}
		s[j] = C[j*3]*c_x + C[j*3+1]*c_y + C[j*3+2]*c_xy;
	}

	return;
}
//------------------------------------------------------------------------------
void stressFromAlpha_elastic_2D(var *s, var *a, var E, var v, var x, var y){
  #ifdef ELASTIC_2D_PLANESTRESS
        var coeff = E/ (1.0-v);
	    var C[9] = { coeff, coeff*v, 0.0,
		             coeff*v, coeff, 0.0,
					  0.0, 0.0, coeff*0.5*(1.0-v) };
	#else // ELASTIC_2D_PLANESTRAIN
		var coeff = E/ ((1.0+v)*(1.0-2.0*v));
		var C[9] = { coeff*(1.0-v), coeff*v, 0.0,
					 coeff*v, coeff*(1.0-v), 0.0,
					 0.0, 0.0, coeff*0.5*(1.0-2.0*v) };
	#endif

	for (unsigned int j=0; j<3; j++){
		s[j] = C[j*3]*a[0] + C[j*3+1]*a[1] + C[j*3+2]*a[2];
	}
  return;
}
//------------------------------------------------------------------------------
void forceFromStrain_elastic_2D(var *f, var *a, var E, var v, var x, var y){

  #ifdef ELASTIC_2D_PLANESTRESS
      var coeff = E/ (1.0-v);
	    var C[9] = { coeff, coeff*v, 0.0,
		             coeff*v, coeff, 0.0,
					  0.0, 0.0, coeff*0.5*(1.0-v) };
	#else // ELASTIC_2D_PLANESTRAIN
		var coeff = E/ ((1.0+v)*(1.0-2.0*v));
		var C[9] = { coeff*(1.0-v), coeff*v, 0.0,
					 coeff*v, coeff*(1.0-v), 0.0,
					 0.0, 0.0, coeff*0.5*(1.0-2.0*v) };
	#endif

	var B[24] = { -(1.0-y), 0.0, (1.0-y), 0.0, y, 0.0, -y, 0.0,
	              0.0, -(1.0-x), 0.0, -x, 0.0, x, 0.0, (1.0-x),
				    -(1.0-x), -(1.0-y), -x, (1.0-y), x, y, (1.0-x), -y };
				 
	var S[24] = { 0.0,0.0,0.0,
	              0.0,0.0,0.0,
	              0.0,0.0,0.0,
	              0.0,0.0,0.0,
	              0.0,0.0,0.0,
	              0.0,0.0,0.0,
	              0.0,0.0,0.0,
	              0.0,0.0,0.0 };

	for (unsigned int i=0; i<8; i++){
		S[3*i]   = B[i] * C[0] + B[i+8]*C[3] + B[i+16]*C[6];
		S[3*i+1] = B[i] * C[1] + B[i+8]*C[4] + B[i+16]*C[7];
		S[3*i+2] = B[i] * C[2] + B[i+8]*C[5] + B[i+16]*C[8];
	}
	
	for (unsigned int i=0; i<8; i++){
		f[i] = S[3*i]*a[0] + S[3*i+1]*a[1] + S[3*i+2]*a[2];
	}
		
  return;
}
//------------------------------------------------------------------------------
void saveFields_elastic_2D(hmgModel_t *model, cudapcgVar_t * D){

  cudapcgVar_t * S = (cudapcgVar_t *)malloc(sizeof(cudapcgVar_t)*(model->m_ndof/2)*3);

  #pragma omp parallel for
  for (unsigned int i=0; i<(model->m_ndof/2)*3; i++){
    S[i] = 0.0;
  }

  unsigned int rows = model->m_ny-1;
  unsigned int cols = model->m_nx-1;
  unsigned int rowscols = rows*cols;

  // cudapcgVar_t * thisCB;
  // var d, C_i, C_j, C_k;
  var E, v;
  unsigned int n, dof;

  var * local_S = (var *)malloc(sizeof(var)*3);
  var * local_d = (var *)malloc(sizeof(var)*8);

  for (unsigned int e=0;e<model->m_nelem;e++){

		E = model->props[2*model->elem_material_map[e]];
		v = model->props[2*model->elem_material_map[e]+1];

		// node 0 (left,bottom)
		n = e+1+(e/rows);
		dof = model->node_dof_map[n];
		local_d[0] = (var) D[dof];
		local_d[1] = (var) D[dof+1];

		// node 1 (right,bottom)
		n+=model->m_ny;
		dof = model->node_dof_map[n];
		local_d[2] = (var) D[dof];
		local_d[3] = (var) D[dof+1];

		// node 2 (right,top)
		n-=1;
		dof = model->node_dof_map[n];
		local_d[4] = (var) D[dof];
		local_d[5] = (var) D[dof+1];

		// node 3 (left,top)
		n-=model->m_ny;
		dof = model->node_dof_map[n];
		local_d[6] = (var) D[dof];
		local_d[7] = (var) D[dof+1];

		stressFromDispl_elastic_2D(local_S,local_d,E,v,0.0,1.0);
		S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
		S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
		S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

		// node 0 (left,bottom)
		n = e+1+(e/rows);
		dof = model->node_dof_map[n];
		stressFromDispl_elastic_2D(local_S,local_d,E,v,0.0,0.0);
		S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
		S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
		S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

		// node 1 (right,bottom)
		n+=model->m_ny;
		dof = model->node_dof_map[n];
		stressFromDispl_elastic_2D(local_S,local_d,E,v,1.0,0.0);
		S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
		S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
		S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

		// node 2 (right,top)
		n-=1;
		dof = model->node_dof_map[n];
		stressFromDispl_elastic_2D(local_S,local_d,E,v,1.0,1.0);
		S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
		S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
		S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];
	}

	// Compensate for periodic borders
	if (model->m_hmg_flag == HOMOGENIZE_X){
		for (unsigned int e=(rowscols-rows); e<rowscols; e++){
			E = model->props[2*model->elem_material_map[e]];
			v = model->props[2*model->elem_material_map[e]+1];

			local_d[0] = 0.0;
			local_d[1] = 0.0;
			local_d[2] = (var) cols;
			local_d[3] = 0.0;
			local_d[4] = (var) cols;
			local_d[5] = 0.0;
			local_d[6] = 0.0;
			local_d[7] = 0.0;

			// node 0 (left,bottom)
			n = e+1+(e/rows);
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,0.0,0.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

			// node 3 (left,top)
			n-=1;
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,0.0,1.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

			local_d[0] = -((var) cols);
			local_d[1] = 0.0;
			local_d[2] = 0.0;
			local_d[3] = 0.0;
			local_d[4] = 0.0;
			local_d[5] = 0.0;
			local_d[6] = -((var) cols);
			local_d[7] = 0.0;

			// node 1 (right,bottom)
			n = e+1+(e/rows) + model->m_ny;
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,1.0,0.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

			// node 2 (right,top)
			n-=1;
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,1.0,1.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];
		}
  } else if (model->m_hmg_flag == HOMOGENIZE_Y){
		for (unsigned int e=0; e<rowscols; e+=rows){//(rows-1)
			E = model->props[2*model->elem_material_map[e]];
			v = model->props[2*model->elem_material_map[e]+1];

			local_d[0] = 0.0;
			local_d[1] = 0.0;
			local_d[2] = 0.0;
			local_d[3] = 0.0;
			local_d[4] = 0.0;
			local_d[5] = (var) rows;
			local_d[6] = 0.0;
			local_d[7] = (var) rows;

			// node 0 (left,bottom)
			n = e+1+(e/rows);
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,0.0,0.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

			// node 1 (right,bottom)
			n+=model->m_ny;
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,1.0,0.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

			local_d[0] = 0.0;
			local_d[1] = -((var) rows);
			local_d[2] = 0.0;
			local_d[3] = -((var) rows);
			local_d[4] = 0.0;
			local_d[5] = 0.0;
			local_d[6] = 0.0;
			local_d[7] = 0.0;

			// node 3 (left,top)
			n = e+(e/rows);
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,0.0,1.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

			// node 2 (right,top)
			n+=model->m_ny;
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,1.0,1.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];
		}
  }  else if (model->m_hmg_flag == HOMOGENIZE_XY){
		for (unsigned int e=0; e<rowscols; e+=rows){//(rows-1)
			E = model->props[2*model->elem_material_map[e]];
			v = model->props[2*model->elem_material_map[e]+1];

			local_d[0] = 0.0;
			local_d[1] = 0.0;
			local_d[2] = 0.0;
			local_d[3] = 0.0;
			local_d[4] = (var) rows;
			local_d[5] = 0.0;
			local_d[6] = (var) rows;
			local_d[7] = 0.0;

			// node 0 (left,bottom)
			n = e+1+(e/rows);
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,0.0,0.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

			// node 1 (right,bottom)
			n+=model->m_ny;
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,1.0,0.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

			local_d[0] = -((var) rows);
			local_d[1] = 0.0;
			local_d[2] = -((var) rows);
			local_d[3] = 0.0;
			local_d[4] = 0.0;
			local_d[5] = 0.0;
			local_d[6] = 0.0;
			local_d[7] = 0.0;

			// node 3 (left,top)
			n = e+(e/rows);
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,0.0,1.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];

			// node 2 (right,top)
			n+=model->m_ny;
			dof = model->node_dof_map[n];
			stressFromDispl_elastic_2D(local_S,local_d,E,v,1.0,1.0);
			S[3*(dof/2)]   += (cudapcgVar_t) 0.25*local_S[0];
			S[3*(dof/2)+1] += (cudapcgVar_t) 0.25*local_S[1];
			S[3*(dof/2)+2] += (cudapcgVar_t) 0.25*local_S[2];
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
    fwrite(S,sizeof(cudapcgVar_t)*(model->m_ndof/2)*3,1,file);
  fclose(file);

	free(local_S);
	free(local_d);
  free(S);

  return;
}
//------------------------------------------------------------------------------
void saveFields_elastic_2D_ScalarDensityField(hmgModel_t *model, cudapcgVar_t * D){
  printf("WARNING: Field exportation not supported for scalar field input (.bin) yet.\n");
  return;
}
//------------------------------------------------------------------------------
