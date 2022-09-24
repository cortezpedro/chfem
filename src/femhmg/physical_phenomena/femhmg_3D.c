#include "femhmg_3D.h"

//------------------------------------------------------------------------------
void assembleNodeDofMap_3D(hmgModel_t *model){
  unsigned int n, i;
	unsigned int nodes_xy = model->m_nx*model->m_ny;
	unsigned int ndof_xy = (model->m_nx-1)*(model->m_ny-1);
	unsigned int dim_y = model->m_ny-1;
	unsigned int dim_z = model->m_nz-1;
	#pragma omp parallel for private(i)
	for (n=0;n<model->m_nnode;n++){
		i = n%nodes_xy;
		model->node_dof_map[n] = (i - i/model->m_ny - dim_y*((i%model->m_ny)/dim_y))%ndof_xy+((n/nodes_xy)%dim_z)*ndof_xy;
		model->node_dof_map[n] *= model->m_nnodedof;
	}
  return;
}
//------------------------------------------------------------------------------
void assembleDofMaterialMap_3D(hmgModel_t *model){
  #pragma omp parallel for
	for (unsigned int i=0; i<model->m_nelem; i++) // not really going through elems, but through nodes with diff dofs
		model->dof_material_map[i] = 0;
	unsigned int n;
	unsigned int dim_y = model->m_ny-1;
	unsigned int dim_xy = (model->m_nx-1)*dim_y;
	// node 0 (left,bottom,near)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_3D;
	}
	// node 1 (right,bottom,near)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = model->m_ny+1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=(model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_3D)<<(1*MATKEY_BITSTEP_3D);
	}
	// node 2 (right,top,near)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = model->m_ny+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=(model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_3D)<<(2*MATKEY_BITSTEP_3D);
	}
	// node 3 (left,top,near)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = (e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy)*model->m_nx*model->m_ny;
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=(model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_3D)<<(3*MATKEY_BITSTEP_3D);
	}
	// node 4 (left,bottom,far)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = 1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy+1)*model->m_nx*model->m_ny;
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=(model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_3D)<<(4*MATKEY_BITSTEP_3D);
	}
	// node 5 (right,bottom,far)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = model->m_ny+1+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy+1)*model->m_nx*model->m_ny;
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=(model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_3D)<<(5*MATKEY_BITSTEP_3D);
	}
	// node 6 (right,top,far)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = model->m_ny+(e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy+1)*model->m_nx*model->m_ny;
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=(model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_3D)<<(6*MATKEY_BITSTEP_3D);
	}
	// node 7 (left,top,far)
	#pragma omp parallel for private(n)
	for (unsigned int e=0;e<model->m_nelem;e++){
		n = (e%dim_xy)+((e%dim_xy)/dim_y)+(e/dim_xy+1)*model->m_nx*model->m_ny;
		model->dof_material_map[model->node_dof_map[n]/model->m_nnodedof]+=(model->elem_material_map[e]&MATKEY_BITSTEP_RANGE_3D)<<(7*MATKEY_BITSTEP_3D);
	}
  return;
}
//------------------------------------------------------------------------------
