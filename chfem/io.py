import numpy as np

def import_nf(filename):
    """ Reads a .nf file (input for chfem_exec) into a python dict.

        :param filename: The path to the .nf file.
        :type filename: str
        :return dictionary with .nf data
        :rtype dict
    """
    # Ensure the filename ends with .nf
    if not filename.endswith('.nf'):
        raise ValueError("Filename does not end with '.nf'")
    
    lines = [] 
    with open(filename) as f:
        lines.extend( [line.rstrip() for line in f] )
    
    data = {}
    data["type_of_analysis"] = None
    data["type_of_solver"] = None
    data["type_of_rhs"] = None
    data["number_of_iterations"] = None
    data["refinement"] = None
    data["number_of_materials"] = None
    
    for key in data:
        ii = lines.index(f'%{key}')
        data[key] = int(lines[ii+1])
    
    ii = lines.index('%voxel_size')
    data["voxel_size"] = float(lines[ii+1])
    
    ii = lines.index('%solver_tolerance')
    data["solver_tolerance"] = float(lines[ii+1])
    
    ii = lines.index('%image_dimensions')
    nx, ny, nz = tuple(map(int,lines[ii+1].split()))
    data["image_dimensions"] = ( nz, ny, nx ) # for numpy 3D indexing of contiguous row-major array
    
    ii = lines.index('%properties_of_materials')
    data["properties_of_materials"] = []
    for jj in range(1,data["number_of_materials"]+1):
        props_str = lines[ii+jj].split()
        clr = int(props_str[0])
        props = []
        for p in props_str[1:]:
            props.append( float(p) )
        data["properties_of_materials"].append( ( clr, *props ) )
    
    data["volume_fraction"] = None
    try: # volume_fraction might not be in .nf file
        ii = lines.index('%volume_fraction')
        data["volume_fraction"] = tuple(map(float,lines[ii+1].split()))
    except Exception as ex:
        print(ex)
        
    data["data_type"] = None
    try: # data_type might not be in .nf file
        ii = lines.index('%data_type')
        data["data_type"] = lines[ii+1]
    except Exception as ex:
        print(ex)
    
    return data

def import_raw(filename, shape=None, dtype=np.uint8):
    """ Reads a .raw file (input for chfem_exec) into a numpy array.

    :param filename: The path to the .raw file.
    :type filename: str
    :param shape: The shape of the numpy array. Should be a tuple (z, y, x) for 3D data, (y, x) for 2D data.
    :type shape: tuple(int, int, int)
    :param dtype: The data type of the array. Defaults to np.uint8.
    :type dtype: np.dtype or string

    :return: A numpy array with the specified shape and dtype, containing the data from the raw file.
    :rtype: np.ndarray
    """
    # Ensure the filename ends with .raw
    if not filename.endswith('.raw'):
        raise ValueError("Filename does not end with '.raw'")
    
    # Read the raw data from the file
    with open(filename, "rb") as file_raw:
        raw_data = file_raw.read()

    if not shape is None:
        # Calculate the expected size of the file in bytes
        expected_size = np.prod(shape) * np.dtype(dtype).itemsize    

        # Check if the file size matches the expected size
        if len(raw_data) != expected_size:
            raise ValueError(f"File size ({len(raw_data)}) does not match expected size ({expected_size})")

    # Convert the raw data to a numpy array
    data_array = np.frombuffer(raw_data, dtype=dtype)

    if not shape is None:
        # Reshape the array to the specified shape
        data_array = data_array.reshape(shape)

    return data_array

def import_scalar_field_from_chfem(filename, domain_shape, rotate_domain=False, dtype=np.float64):
    """ Import scalar field (e.g. temperature, pressure) output from chfem

        :param filename: file path and name of .bin file
        :type filename: string
        :param domain_shape: shape of domain for which the scalar field was generated
        :type domain_shape: (int, int, int)
        :param rotate_domain: rotate the domain to be in (x,y,z) format. default is (z,y,x), same as export.
        :type rotate_domain: bool
        :param dtype: The data type of the array. Defaults to np.uint8.
        :type dtype: np.dtype or string
        :return: scalar field
        :rtype: np.ndarray
    """
    
    # 2D
    if len(domain_shape) == 2:
        domain = np.fromfile(filename, dtype=dtype).reshape( (domain_shape[1],domain_shape[0]) ) # [ x, y ] (fields from simulation are transposed xy)
        if not rotate_domain:
            domain = domain.transpose() # [ y, x ] == [ row, col ]
        return domain
    
    # 3D
    converted_shape = (domain_shape[0], domain_shape[2], domain_shape[1]) # [ z, x, y ] (fields from simulation are transposed xy)
    domain = np.fromfile(filename, dtype=dtype).reshape(converted_shape)
    if rotate_domain:
        domain = domain.transpose(1,2,0) # [ x, y, z ]
    else:
        domain = domain.transpose(0,2,1) # [ z, y, x ] == [ slice, row, col ]
    return domain

def import_vector_field_from_chfem(filename, domain_shape, rotate_domain=False, dtype=np.float64):
    """ Import vector field (e.g. heat flux, displacement, velocity) output from chfem

        :param filename: file path and name of .bin file
        :type filename: string
        :param domain_shape: shape of domain for which the scalar field was generated
        :type domain_shape: (int, int, int)
        :param rotate_domain: rotate the domain to be in (x,y,z) format. default is (z,y,x), same as export.
        :type rotate_domain: bool
        :param dtype: The data type of the array. Defaults to np.uint8.
        :type dtype: np.dtype or string
        :return: vector field ( len(domain_shape), *domain_shape )
        :rtype: np.ndarray
    """
    # 2D
    if len(domain_shape) == 2:
        domain = np.fromfile(filename, dtype=dtype).reshape( (domain_shape[1],domain_shape[0],2) ) # [ x, y, vector_field ] (fields from simulation are transposed xy)
        if rotate_domain:
            domain = domain.transpose(2,0,1) # [ vector_field, x, y ]
        else:
            domain = domain.transpose(2,1,0) # [ vector_field, y, x ]
        return domain
    
    # 3D
    converted_shape = (domain_shape[0], domain_shape[2], domain_shape[1]) # [ z, x, y ] (fields from simulation are transposed xy)
    domain = np.fromfile(filename, dtype=dtype)
    domain = np.reshape( domain, ( *converted_shape, 3) ) # [ z, x, y, vector_field ]
    if rotate_domain:
        domain = domain.transpose(3,1,2,0) # [ vector_field, x, y, z ]
    else:
        domain = domain.transpose(3,0,2,1) # [ vector_field, z, y, x ]
    return domain

def import_stress_field_from_chfem(filename, domain_shape, rotate_domain=False, dtype=np.float64):
    """ Import stress fields output from chfem

        :param filename: file path and name of .bin file
        :type filename: string
        :param domain_shape: shape of domain for which the scalar field was generated
        :type domain_shape: (int, int, int)
        :param rotate_domain: rotate the domain to be in (x,y,z) format. default is (z,y,x), same as export.
        :type rotate_domain: bool
        :param dtype: The data type of the array. Defaults to np.uint8.
        :type dtype: np.dtype or string
        :return: vector field ( stress_field, *domain_shape )  [ stress_field=3 (2D) or stress_field=6 (3D) ]
        :rtype: (np.ndarray, np.ndarray)
    """
    # 2D
    if len(domain_shape) == 2:
        domain = np.fromfile(filename, dtype=dtype).reshape( (domain_shape[1],domain_shape[0],3) ) # [ x, y, stress_field ] (fields from simulation are transposed xy)
        if rotate_domain:
            domain = domain.transpose(2,0,1) # [ stress_field, x, y ]
        else:
            domain = domain.transpose(2,1,0) # [ stress_field, y, x ]
        return domain
    
    # 3D
    converted_shape = (domain_shape[0], domain_shape[2], domain_shape[1]) # [ z, x, y ] (fields from simulation are transposed xy)
    domain = np.fromfile(filename, dtype=dtype)
    domain = np.reshape( domain, ( *converted_shape, 6) ) # [ z, x, y, stress_field ]
    if rotate_domain:
        domain = domain.transpose(3,1,2,0) # [ stress_field, x, y, z ]
    else:
        domain = domain.transpose(3,0,2,1) # [ stress_field, z, y, x ]
    return domain

def export_for_chfem(filename, array, analysis_type, mat_props=None,
                     voxel_size=1, solver_type=0, rhs_type=0, refinement=1,
                     export_raw=True, export_nf=True, 
                     solver_tolerance=1.e-6, solver_maxiter=10000, tmp_nf_file=None):
    """ Export a numpy array to run an analysis in chfem

        :param filename: filepath and name
        :type filename: string
        :param array: array to be exported. shape = (z,y,x)==[slice,row,col]
        :type array: np.array
        :param analysis_type: 0 = conductivity, 1 = elasticity, 2 = permeability
        :type analysis_type: int
        :param mat_props: material properties for each phase as a dictionary. For conductivity, use {phase_id: cond}. For elasticity, use {phase_id: (young, poisson)}. For permeability, use None.
        :type mat_props: dict
        :param voxel_size: voxel size
        :type voxel_size: float
        :param solver_type:  0 = pcg (default), 1 = cg, 2 = minres
        :type solver_type: int
        :param rhs_type: type of right hand side (0 or 1)
        :type rhs_type: int
        :param export_raw: export .raw file from numpy array
        :type export_raw: bool
        :param export_nf: export .nf file with simulations inputs for chfem
        :type export_nf: bool
        :param solver_tolerance: solver tolerance for simulation
        :type solver_tolerance: float
        :param solver_maxiter: maximum number of iterations
        :type solver_maxiter: int
        :param tmp_nf_file: only for use within the python API
        :type tmp_nf_file: file
        
        :return dictionary with .nf data
        :rtype dict

        :Example:
        >>> export_for_chfem('200_fiberform', array, 2, solver_tolerance=1e-6, solver_maxiter=100000)
    """

    domain = array.astype(np.uint8)
            
    mat_i, cmat_i = np.unique(domain, return_counts=True)
    materials = dict(zip(mat_i, cmat_i))
    materials = dict(sorted(materials.items(), key=lambda x: x[0]))

    # Check if mat_props IDs match array's unique values
    if mat_props is not None:
        mat_props_ids = set(mat_props.keys())
        array_material_ids = set(materials.keys())

        if mat_props_ids != array_material_ids:
            missing_ids_in_props = array_material_ids - mat_props_ids
            missing_ids_in_array = mat_props_ids - array_material_ids
            error_message = []
            if missing_ids_in_props:
                error_message.append(f"Missing material IDs in 'mat_props' that are present in the array: {missing_ids_in_props}.")
            if missing_ids_in_array:
                error_message.append(f"Extra material IDs in 'mat_props' that are not present in the array: {missing_ids_in_array}.")
            raise ValueError(' '.join(error_message))

    # Prepare material properties for export
    properties_of_materials = []
    if analysis_type == 0:  # conductivity
        for mat_id, cond in mat_props.items():
            properties_of_materials.append(f"{mat_id} {cond}")
    elif analysis_type == 1:  # elasticity
        for mat_id, (young, poisson) in mat_props.items():
            properties_of_materials.append(f"{mat_id} {young} {poisson}")
    elif analysis_type == 2:  # permeability
        properties_of_materials = [f"{mat_id}" for mat_id in materials.keys()]
    else:
        raise ValueError("Invalid analysis type.")
    
    volume_fractions = np.array(list(materials.values())) * 100. / np.prod(domain.shape)
    
    jdata = {}
    jdata["type_of_analysis"] = analysis_type
    jdata["type_of_solver"] = solver_type
    jdata["type_of_rhs"] = rhs_type
    jdata["voxel_size"] = voxel_size
    jdata["solver_tolerance"] = solver_tolerance
    jdata["number_of_iterations"] = solver_maxiter
    if len(domain.shape) == 2:
        jdata["image_dimensions"] = [domain.shape[1],domain.shape[0],0] # [ x, y, 0 ]
    else:
        jdata["image_dimensions"] = [domain.shape[2],domain.shape[1],domain.shape[0]] # [ x, y, z ]
    jdata["refinement"] = refinement
    jdata["number_of_materials"] = len(materials)
    jdata["properties_of_materials"] = properties_of_materials
    jdata["volume_fraction"] = list(np.around(volume_fractions, 2))
    jdata["data_type"] = "uint8"

    if export_nf:  # for chfem
        sText = ''
        for k, v in jdata.items():
            sText += '%' + str(k) + '\n'
            if k == 'properties_of_materials':
                for prop in v:
                    # Directly append the property string since it's already formatted
                    sText += prop + '\n'
            else:
                sText += str(v).replace('], ', '\n').replace('[', '').replace(']', '').replace(',', '') + '\n'
        sText = sText.strip()  # Remove the trailing newline for cleaner output

        # Write to temporary file or .nf file
        if tmp_nf_file is not None:
            tmp_nf_file.write(sText.encode('utf-8'))
            tmp_nf_file.flush()
            tmp_nf_file.seek(0)  # rewind for printing
        else:
            with open(filename[:len(filename) - 4] + ".nf", 'w') as file_nf:
                file_nf.write(sText)

    if export_raw:
        if filename[-4:] != '.raw':
            filename = filename + '.raw'
        domain.tofile(filename)
    
    return jdata
