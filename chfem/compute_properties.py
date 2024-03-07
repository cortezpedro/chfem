from chfem.wrapper import run
from chfem.io import export_for_chfem
import numpy as np
import tempfile
import os

def compute_conductivity(array, mat_props, voxel_size=1e-6, solver_type='minres', solver_tolerance=1e-6, solver_maxiter=10000,
                         precondition=True, type_of_rhs=0, refinement=1, direction='all', output_fields=None):
    
    return parse_and_run(analysis_type=0, array=array, mat_props=mat_props, voxel_size=voxel_size, 
                         solver_type=solver_type, solver_tolerance=solver_tolerance, solver_maxiter=solver_maxiter,
                         precondition=precondition, rhs_type=type_of_rhs, refinement=refinement, direction=direction, output_fields=output_fields)

def compute_elasticity(array, mat_props, voxel_size=1e-6, solver_type='minres', solver_tolerance=1e-6, solver_maxiter=10000,
                       precondition=True, type_of_rhs=0, refinement=1, direction='all', output_fields=None):
    
    return parse_and_run(analysis_type=1, array=array, mat_props=mat_props, voxel_size=voxel_size, 
                         solver_type=solver_type, solver_tolerance=solver_tolerance, solver_maxiter=solver_maxiter,
                         precondition=precondition, rhs_type=type_of_rhs, refinement=refinement, direction=direction, output_fields=output_fields)

def compute_permeability(array, voxel_size=1e-6, solver_type='minres', solver_tolerance=1e-6, solver_maxiter=10000,
                         precondition=True, type_of_rhs=0, refinement=1, direction='all', output_fields=None):
    
    return parse_and_run(analysis_type=2, array=array, voxel_size=voxel_size, 
                         solver_type=solver_type, solver_tolerance=solver_tolerance, solver_maxiter=solver_maxiter,
                         precondition=precondition, rhs_type=type_of_rhs, refinement=refinement, direction=direction, output_fields=output_fields)


def check_solver(solver_type, output_fields):
    solvers = {'cg3': 2, 'minres3': 3, 'cg2': 4, 'minres2': 5}
    if solver_type not in ['cg', 'minres']:
        raise ValueError(f"Invalid solver type: {solver_type}")

    if output_fields:
        return solvers[solver_type + '3']
    return solvers[solver_type + '2']
    
def check_direction(direction):
    directions = {'x': 0, 'y': 1, 'z': 2, 'yz': 3, 'xz': 4, 'xy': 5, 'all': 6}
    if direction not in directions:
        raise ValueError(f"Invalid direction: {direction}")
    return directions[direction]

def parse_and_run(analysis_type, array, mat_props=None, voxel_size=1e-6, solver_type='minres', solver_tolerance=1e-6, solver_maxiter=10000,
                  precondition=True, rhs_type=0, refinement=1, direction='all', output_fields=None):
    if array.dtype != np.uint8:
        raise ValueError("Domain must be uint8 dtype.")
    elif array.ndim != 3:
        raise ValueError("Domain must be 3D.")
    
    unique_ids = np.unique(array)
    if 0 not in unique_ids:
        raise ValueError("No fluid phase found in the domain.")

    if output_fields is None:
        output_fields_flag = 0
        tmp_nf_file = tempfile.NamedTemporaryFile(delete=False)
    else:
        path, _ = os.path.split(output_fields)
        if path == '' or os.path.exists(path):
            output_fields_flag = 1
            tmp_nf_file = open(output_fields + ".nf", 'wb')
        else:
            raise ValueError(f"Output fields directory {path} does not exist.")

    export_for_chfem(None, array, analysis_type=analysis_type, mat_props=mat_props,
                     voxel_size=voxel_size, solver_type=check_solver(solver_type, output_fields_flag), 
                     rhs_type=rhs_type, refinement=refinement,
                     export_raw=False, export_nf=True, solver_tolerance=solver_tolerance, 
                     solver_maxiter=solver_maxiter, tmp_nf_file=tmp_nf_file)
    # print(tmp_nf_file.read().decode('utf-8'))

    print("Calling chfem wrapper")
    eff_coeff = run(array, tmp_nf_file.name, analysis_type, check_direction(direction), precondition, output_fields_flag)
    
    tmp_nf_file.close()
    os.remove(tmp_nf_file.name)
    
    return eff_coeff
