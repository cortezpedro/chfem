from chfem.wrapper import run
from chfem.io import export_for_chfem
import numpy as np
import tempfile
import os

def compute_property(property, array, mat_props=None, voxel_size=1e-6, solver=None, solver_tolerance=1e-6, solver_maxiter=10000,
                     precondition=True, type_of_rhs=0, refinement=1, xreduce='full', direction='all', output_fields=None, nf_filepath=None):
    """Computes the effective property of a material using chfem.

    This function calculates the effective thermal conductivity, linear elasticity,
    or permeability of a material sample characterized by a 3D domain. The domain is
    discretized into voxels, each assigned a material phase with specific properties.
    It interfaces with the chfem cuda/c library to perform image-based finite element analysis.

    :param property: The physical property to be computed ('conductivity', 'elasticity', 'permeability').
    :type property: str
    :param array: 3D numpy array representing the material domain, where each voxel's value is the material phase.
    :type array: np.ndarray
    :param mat_props: Material properties corresponding to each phase in the domain, provided as {phase_id: cond} for conductivity and {phase_id: (young, poisson)} or {phase_id: (young, poisson, alpha)} for elasticity, None for permeability
    :type mat_props: dict
    :param voxel_size: The edge length of each voxel in the domain, defaults to 1e-6 meters.
    :type voxel_size: float, optional
    :param solver: The type of solver to use ('cg' for Conjugate Gradient, 'minres' for MINimal RESidual). Defaults to 'cg' for conductivity and elasticity, and 'minres' for permeability. Options: 'cg', 'minres' (5 vectors implementations, faster but more memory), 'cg3', 'minres3' (3 vectors, slower but less memory), 'cg2', 'minres2' (2 vectors, less memory but not fields output).
    :type solver: str, optional
    :param solver_tolerance: The tolerance for the solver convergence criterion, defaults to 1e-6.
    :type solver_tolerance: float, optional
    :param solver_maxiter: The maximum number of iterations for the solver, defaults to 10000.
    :type solver_maxiter: int, optional
    :param precondition: Flag to use preconditioning in the solver, defaults to True.
    :type precondition: bool, optional
    :param type_of_rhs: Type of the right-hand side in the system of equations, defaults to 0.
    :type type_of_rhs: int, optional
    :param refinement: Refinement level for the domain discretization, defaults to 1.
    :type refinement: int, optional
    :param xreduce: Reduction strategy for 2-vector solvers ('full', 'diag', float: stablization factor ), defaults to 'full'.
    :type xreduce: str or float, optional
    :param direction: Direction for calculating the property ('x', 'y', 'z', 'yz', 'xz', 'xy', 'all'), defaults to 'all'.
    :type direction: str, optional
    :param output_fields: File path to output the fields (e.g., displacement, temperature), defaults to None.
    :type output_fields: str, optional
    :param nf_filepath: Provide the path to a Neutral File (.nf) to specify the simulation settings.
    :type nf_filepath: str, optional
    :return: The effective property coefficient.
    :rtype: list
    
    :Example:
    >>> import chfem
    >>> array = np.zeros((80, 90, 100), dtype=np.uint8)
    >>> array[20:60, 30:70, 30:70] = 255
    >>> keff = chfem.compute_property('conductivity', array, mat_props={255: 1, 0: 0.1}, direction='x', output_fields="cubes")
    >>> Ceff = chfem.compute_property('elasticity', array, mat_props={255: (200, 0.2), 0: (100, 0.3)}, direction='x', output_fields="cubes") 
    >>> Keff = chfem.compute_property('permeability', array, direction='x', output_fields="cubes")
    """
    
    # Error checks
    properties = {'conductivity': 0, 'elasticity': 1, 'permeability': 2}
    if property not in properties:
        raise ValueError(f"Invalid property: {property}")
    analysis_type = properties[property]
    
    directions = {'x': 0, 'y': 1, 'z': 2, 'yz': 3, 'xz': 4, 'xy': 5, 'all': 6}
    if direction not in directions:
        raise ValueError(f"Invalid direction: {direction}")
    direction_int = directions[direction]
    
    thermal_expansion_flag = False
    if mat_props is not None and (analysis_type == 1): # elasticity
      for _, props in mat_props.items():
        if len(props) == 3:
          thermal_expansion_flag = True
          break
      
    if thermal_expansion_flag:
      if direction_int < 6:
        raise ValueError(f"Invalid direction for thermal expansion analysis: {direction}. Must be \"all\".")
      for mat, props in mat_props.items():
        if len(props) < 3:
          raise ValueError(f"Invalid properties {props} for material {mat}. Expected thermal expansion coefficient.")
    
    output_fields_flag = 1 if output_fields is not None else 0

    if solver is None:
        solver = 'minres' if property == 'permeability' else 'cg'
    solvers = {'cg': 0, 'minres': 1, 'cg3': 2, 'minres3': 3, 'cg2': 4, 'minres2': 5}
    if solver not in solvers:
        raise ValueError(f"Invalid solver type: {solver}")
    if solver in ['cg2','minres2'] and ( output_fields_flag or not property == 'permeability'):
        raise ValueError(f"Invalid solver type: {solver}. Two-vectors solvers do not produce fields to be exported, nor are implemented for conductivy or elasticity analyses.")
    solver_type = solvers[solver]
    
    xreduce_flag=2
    xreduce_flags = { 'full': 2,  'diag': 1 }
    if xreduce in xreduce_flags:
      xreduce_flag = xreduce_flags[xreduce]
      xreduce=1.e-14 # default value, will not be used in this scenario
    else:
      if 'float' not in str(type(xreduce)):
        raise ValueError(f"Invalid xreduce parameter: {xreduce}. Expected 'full', 'diag', or a float.")
      xreduce_flag = 0
      xreduce = float(xreduce) # make sure it is a simple float type (might be numpy)
      

    if nf_filepath is not None:
        if os.path.exists(nf_filepath):
            nf_filename = nf_filepath
        else:
            raise ValueError(f"The file {nf_filepath} does not exist.")
        
    else:
        if mat_props is None and property != 'permeability':
            raise ValueError("Material properties must be provided, e.g. [(phase_id, cond),] for conductivity or [(phase_id, young, poisson),] for elasticity.")
            
        if array.dtype != np.uint8:
            raise ValueError("Domain must be uint8 dtype.")
        #elif array.ndim != 3:
        #    raise ValueError("Domain must be 3D.")
        
        unique_ids = np.unique(array)
        if 0 not in unique_ids:
            raise ValueError("No fluid phase found in the domain.")

        if output_fields is None:
            tmp_nf_file = tempfile.NamedTemporaryFile(delete=False)
        else:
            path, _ = os.path.split(output_fields)
            if path == '' or os.path.exists(path):
                tmp_nf_file = open(output_fields + ".nf", 'wb')
            else:
                raise ValueError(f"Output fields directory {path} does not exist.")
        
        # Create temporary .nf file with properties
        export_for_chfem(None, array, analysis_type=analysis_type, mat_props=mat_props,
                        voxel_size=voxel_size, solver_type=solver_type, 
                        rhs_type=type_of_rhs, refinement=refinement,
                        export_raw=False, export_nf=True, solver_tolerance=solver_tolerance, 
                        solver_maxiter=solver_maxiter,
                        xreduce_scale_factor=xreduce,
                        tmp_nf_file=tmp_nf_file)
        
        # print(tmp_nf_file.read().decode('utf-8'))
        nf_filename = tmp_nf_file.name

    # convert array to contiguous C order
    #array = np.ascontiguousarray(array.transpose(2, 1, 0))
    array = np.ascontiguousarray(np.reshape(array,(array.size)))

    print("Calling chfem wrapper")
    eff_coeff = run(array, nf_filename, analysis_type, direction_int, solver_type, precondition, output_fields_flag, xreduce_flag)
    
    if nf_filepath is None:
        tmp_nf_file.close(); os.remove(tmp_nf_file.name) # removing temporary .nf file
    
    return eff_coeff
