#include "femhmg_2D.h"

//------------------------------------------------------------------------------
void assembleNodeDofMap_2D(hmgModel_t *model){
  unsigned int n;
	unsigned int dim_y = model->m_ny-1;
	#pragma omp parallel for
	for (n=0;n<model->m_nnode;n++){
		model->node_dof_map[n] = (n - n/model->m_ny - dim_y*((n%model->m_ny)/dim_y))%(model->m_ndof/model->m_nnodedof);
		model->node_dof_map[n] *= model->m_nnodedof;
	}
  return;
}
//------------------------------------------------------------------------------
void assembleDofMaterialMap_2D(hmgModel_t *model){
  #pragma omp parallel for
	for (unsigned int i=0; i<model->m_nelem; i++) // not really going through elems, but through nodes with diff dofs
		model->dof_material_map[i] = 0;
	unsigned int n;
	unsigned int dim_y = model->m_ny-1;
	// node 0 (left,bottom)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = e+1+(e/dim_y);
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_2D;
	}
	// node 1 (right,bottom)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = e+1+(e/dim_y)+model->m_ny;
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=(model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_2D)<<(1*MATKEY_BITSTEP_2D);
	}
	// node 2 (right,top)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = e+(e/dim_y)+model->m_ny;
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=(model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_2D)<<(2*MATKEY_BITSTEP_2D);
	}
	// node 3 (left,top)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = e+(e/dim_y);
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=(model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_2D)<<(3*MATKEY_BITSTEP_2D);
	}
  return;
}
//------------------------------------------------------------------------------
