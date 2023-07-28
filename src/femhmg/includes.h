#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include "cudapcg/cudapcg.h"
#include "report/report.h"

#ifndef INCLUDES_FEMHMG_H_INCLUDED
#define INCLUDES_FEMHMG_H_INCLUDED

//------------------------------------------------------------------------------

#define MAX_COLORKEY 255
#define MAX_COLORNUM 256
#define PROPS_ARRAY_SIZE 512
#define STR_BUFFER_SIZE 256
#define DEFAULT_NUM_OF_STREAMS 8

#define MAX(A,B) (B>A?B:A)
#define VOID_AS_FLOAT(p,id) (((float *)p)[id])
#define VOID_AS_DOUBLE(p,id) (((double *)p)[id])

#define HMG_THERMAL 0
#define HMG_ELASTIC 1
#define HMG_FLUID 2

#define HMG_2D 2
#define HMG_3D 3

#define HMG_TRUE 1
#define HMG_FALSE 0

#define HMG_DIR_X 0
#define HMG_DIR_Y 1
#define HMG_DIR_Z 2

#define HOMOGENIZE_X 0
#define HOMOGENIZE_Y 1
#define HOMOGENIZE_Z 2
#define HOMOGENIZE_YZ 3
#define HOMOGENIZE_XZ 4
#define HOMOGENIZE_XY 5
#define HOMOGENIZE_ALL 6
#define HOMOGENIZE_THERMAL_EXPANSION 7

#define HMG_FLOAT32 0
#define HMG_FLOAT64 1

typedef double var;
typedef uint8_t logical;
typedef uint16_t hmgFlag_t;

typedef struct _hmgmodel{


    char *neutralFile;
    char *neutralFile_noExt;
    char *imageFile;
    char *sdfFile;

    report_t *report;
    reportFlag_t m_report_flag;

    logical m_exportX_flag;
    char *x0File;

    logical m_saveFields_flag;
    logical m_fieldsByElem_flag; 

    hmgFlag_t m_dim_flag;
    hmgFlag_t m_analysis_flag;
    hmgFlag_t m_hmg_flag;
    hmgFlag_t m_scalar_field_data_type_flag;
    cudapcgFlag_t m_pcg_flag;
    logical m_hmg_flag_was_set;
    logical m_using_x0_flag;
    logical m_hmg_thermal_expansion_flag;

    cudapcgFlag_t poremap_flag;

    var m_elem_size;
    unsigned int m_mesh_refinement;

    unsigned int m_C_dim;
    unsigned int m_lclMtx_dim;
    unsigned int m_lclCB_dim;

    var props[PROPS_ARRAY_SIZE];
    var alpha[MAX_COLORNUM];
    unsigned char props_keys[MAX_COLORNUM]; // data is 8bit

    double density_max;
    double density_min;

    unsigned int * node_dof_map;
    cudapcgMap_t * elem_material_map;
    cudapcgMap_t * dof_material_map;
    parametricScalarField_t * density_map;
    cudapcgIdMap_t  * dof_id_map; // used for permeability analysis
    cudapcgFlag_t * dof_fluid_map; // used for permeability analysis
    cudapcgVar_t * Mtxs;
    cudapcgVar_t * CB;
    cudapcgVar_t * RHS;
    cudapcgVar_t ** x0;
    var * C;
    var * thermal_expansion;
    cudapcgFlag_t * pore_border_fluidkeys;
    cudapcgIdMap_t * pore_dof2node_map;

    unsigned int m_nx;
    unsigned int m_ny;
    unsigned int m_nz;
    unsigned int m_nnode;
    unsigned int m_nnodedof;
    unsigned int m_nelem;
    unsigned int m_nelemdof;
    unsigned int m_ndof;
    unsigned int m_nVelocityNodes; // used for permeability analysis
    unsigned int m_nBorderNodes;
    
    unsigned char m_nmat; // keys are 8bit

    cudapcgTol_t m_num_tol;
    unsigned int m_max_iterations;

    logical (*initModel)(struct _hmgmodel *);
    void (*assembleLocalMtxs)(struct _hmgmodel *);
    void (*assembleNodeDofMap)(struct _hmgmodel *);
    void (*assembleDofIdMap)(struct _hmgmodel *);
    void (*assembleDofMaterialMap)(struct _hmgmodel *);
    void (*assembleRHS)(struct _hmgmodel *);
    void (*updateC)(struct _hmgmodel *, cudapcgVar_t *);
    void (*printC)(struct _hmgmodel *, char *);
    
    void (*saveFields)(struct _hmgmodel *, cudapcgVar_t *);

} hmgModel_t;

//-------------------------------------------------------------------------------

#endif // INCLUDES_FEMHMG_H_INCLUDED
