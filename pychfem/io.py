import numpy as np
import json

def import_raw(filename, shape, dtype=np.uint8):
    """
    Reads a .raw file into a numpy array.

    Parameters:
    - filename: The path to the .raw file.
    - shape: The shape of the numpy array. Should be a tuple (z, y, x) for 3D data.
    - dtype: The data type of the array. Defaults to np.uint8.

    Returns:
    - A numpy array with the specified shape and dtype, containing the data from the raw file.
    """
    # Ensure the filename ends with .raw
    if not filename.endswith('.raw'):
        raise ValueError("Filename does not end with '.raw'")

    # Calculate the expected size of the file in bytes
    expected_size = np.prod(shape) * np.dtype(dtype).itemsize

    # Read the raw data from the file
    with open(filename, "rb") as file_raw:
        raw_data = file_raw.read()

    # Check if the file size matches the expected size
    if len(raw_data) != expected_size:
        raise ValueError(f"File size ({len(raw_data)}) does not match expected size ({expected_size})")

    # Convert the raw data to a numpy array
    data_array = np.frombuffer(raw_data, dtype=dtype)

    # Reshape the array to the specified shape
    data_array = data_array.reshape(shape)

    return data_array

def import_scalar_field_from_chfem(filename, domain_shape, rotate_domain=True):
    """ Import scalar field (e.g. temperature, pressure) output from 
        CHFEM_GPU CUDA kernels (https://gitlab.com/cortezpedro/chfem_gpu)

        :param filename: file path and name of .bin file
        :type filename: string
        :param domain_shape: shape of domain for which the scalar field was generated
        :type domain_shape: (int, int, int)
        :param rotate_domain: rotate the domain to be in the same format as export
        :type rotate_domain: bool
        :return: scalar field (x,y,z)
        :rtype: np.ndarray
    """
    converted_shape = (domain_shape[2], domain_shape[0], domain_shape[1])
    chpack_domain = np.fromfile(filename).reshape(converted_shape)
    if rotate_domain:
        chpack_domain = np.rot90(chpack_domain, axes=(0, 1))
        chpack_domain = np.rot90(chpack_domain, axes=(1, 2))
        chpack_domain = np.rot90(chpack_domain, axes=(0, 1))
        chpack_domain = np.rot90(chpack_domain, axes=(0, 1))

    return chpack_domain

def import_vector_field_from_chfem(filename, domain_shape, rotate_domain=True, correct_direction=None):
    """ Import vector field (e.g. heat flux, displacement, velocity) output from 
        CHFEM_GPU CUDA kernels (https://gitlab.com/cortezpedro/chfem_gpu)

        :param filename: file path and name of .bin file
        :type filename: string
        :param domain_shape: shape of domain for which the scalar field was generated
        :type domain_shape: (int, int, int)
        :param rotate_domain: rotate the domain to be in the same format as export
        :type rotate_domain: bool
        :param correct_direction: correct orientation field according to simulation direction, expects 'x', 'y', or 'z'
        :type correct_direction: str
        :return: vector field (x,y,z,3)
        :rtype: np.ndarray
    """
    chpack_domain = np.fromfile(filename, dtype=float)
    converted_shape = (domain_shape[2], domain_shape[0], domain_shape[1])

    orientation = np.zeros(converted_shape + (3,))
    orientation[:, :, :, 0] = chpack_domain[0::3].reshape(converted_shape)
    orientation[:, :, :, 1] = chpack_domain[1::3].reshape(converted_shape)
    orientation[:, :, :, 2] = chpack_domain[2::3].reshape(converted_shape)
    
    if correct_direction is not None:
        if correct_direction == 'x':
            orientation[:, :, :, 0] =   orientation[:, :, :, 0]
            orientation[:, :, :, 1] = - orientation[:, :, :, 1]
            orientation[:, :, :, 2] = - orientation[:, :, :, 2]
        elif correct_direction == 'y':
            orientation[:, :, :, 0] = - orientation[:, :, :, 0]
            orientation[:, :, :, 1] =   orientation[:, :, :, 1]
            orientation[:, :, :, 2] =   orientation[:, :, :, 2]
        elif correct_direction == 'z':
            orientation[:, :, :, 0] = - orientation[:, :, :, 0]
            orientation[:, :, :, 1] =   orientation[:, :, :, 1]
            orientation[:, :, :, 2] =   orientation[:, :, :, 2]
    
    if rotate_domain:
        orientation = np.rot90(orientation, axes=(0, 1))
        orientation = np.rot90(orientation, axes=(1, 2))
        orientation = np.rot90(orientation, axes=(0, 1))
        orientation = np.rot90(orientation, axes=(0, 1))

    return orientation

def import_stress_field_from_chfem(filename, domain_shape, rotate_domain=True):
    """ Import stress fields output from 
        CHFEM_GPU CUDA kernels (https://gitlab.com/cortezpedro/chfem_gpu)

        :param filename: file path and name of .bin file
        :type filename: string
        :param domain_shape: shape of domain for which the scalar field was generated
        :type domain_shape: (int, int, int)
        :param rotate_domain: rotate the domain to be in the same format as export
        :type rotate_domain: bool
        :return: direct stresses (x,y,z,3) and shear stresses (x,y,z,3)
        :rtype: (np.ndarray, np.ndarray)
    """
    chpack_domain = np.fromfile(filename, dtype=float)
    converted_shape = (domain_shape[2], domain_shape[0], domain_shape[1])

    sigma = np.zeros(converted_shape + (3,))
    tau = np.zeros(converted_shape + (3,))
    sigma[:, :, :, 0] = chpack_domain[0::6].reshape(converted_shape)
    sigma[:, :, :, 1] = chpack_domain[1::6].reshape(converted_shape)
    sigma[:, :, :, 2] = chpack_domain[2::6].reshape(converted_shape)
    tau[:, :, :, 0] = chpack_domain[3::6].reshape(converted_shape)
    tau[:, :, :, 1] = chpack_domain[4::6].reshape(converted_shape)
    tau[:, :, :, 2] = chpack_domain[5::6].reshape(converted_shape)

    if rotate_domain:
        sigma = np.rot90(sigma, axes=(0, 1))
        sigma = np.rot90(sigma, axes=(1, 2))
        sigma = np.rot90(sigma, axes=(0, 1))
        sigma = np.rot90(sigma, axes=(0, 1))
        tau = np.rot90(tau, axes=(0, 1))
        tau = np.rot90(tau, axes=(1, 2))
        tau = np.rot90(tau, axes=(0, 1))
        tau = np.rot90(tau, axes=(0, 1))

    return sigma, tau

def export_for_chfem(filename, ws, analysis, voxel_length=1, solver=0, export_nf=True, export_json=True, tol=1.e-6, max_iter=10000):
    """ Export a puma.Workspace to run an analysis in
        CHFEM_GPU CUDA kernels (https://gitlab.com/cortezpedro/chfem_gpu) or 
        CHPACK Julia package (https://gitlab.com/lcc-uff/Chpack.jl)

        :param filename: filepath and name
        :type filename: string
        :param ws: to be exported
        :type ws: pumapy.Workspace
        :param type_of_analysis: 0 = conductivity, 1 = elasticity, 2 = permeability
        :type type_of_analysis: int
        :param type_of_solver:  0 = pcg (default), 1 = cg, 2 = minres
        :type type_of_solver: int
        :param export_nf: export .nf file with simulations inputs for CHFEM_GPU
        :type export_nf: bool
        :param export_json: export .json file with simulations inputs for CHFPACK
        :type export_json: bool
        :param tol: solver tolerance for simulation
        :type tol: float
        :param max_iter: maximum number of iterations
        :type max_iter: int

        :Example:
        >>> import pumapy as puma
        >>> ws = puma.import_3Dtiff(puma.path_to_example_file("200_fiberform.tif"), 1.3e-6)
        Importing ...
        >>> puma.experimental.export_for_chfem('200_fiberform', ws, 2, tol=1e-6, max_iter=100000)
    """

    domain = ws.astype(np.uint8)

    if filename[-4:] != '.raw':
        filename = filename + '.raw'

    materials = {}
    # Save image data in RAW format
    with open(filename, "bw") as file_raw:
        for k in range(domain.shape[2]):
            mat_i, cmat_i = np.unique(domain[:, :, k], return_counts=True)
            for i in range(len(mat_i)):
                if mat_i[i] in materials:
                    materials[mat_i[i]] += cmat_i[i]
                else:
                    materials[mat_i[i]] = cmat_i[i]
            (domain[:, :, k].T).tofile(file_raw)

    materials = dict(sorted(materials.items(), key=lambda x: x[0]))
    
    mat = np.array(list(materials.keys()))
    if analysis != 2:
        mat = np.vstack((mat, np.zeros((mat.shape[0]), dtype=int))).T
    else:
        mat = np.expand_dims(mat, axis=1)
        
    cmat = np.array(list(materials.values()))
    cmat = cmat * 100. / np.prod(domain.shape)
    
    jdata = {}
    jdata["type_of_analysis"] = analysis
    jdata["type_of_solver"] = solver
    jdata["type_of_rhs"] = 0
    jdata["voxel_size"] = voxel_length
    jdata["solver_tolerance"] = tol
    jdata["number_of_iterations"] = max_iter
    jdata["image_dimensions"] = list(domain.shape)
    jdata["refinement"] = 1
    jdata["number_of_materials"] = mat.shape[0]
    jdata["properties_of_materials"] = mat.tolist()
    jdata["volume_fraction"] = list(np.around(cmat, 2))
    jdata["data_type"] = "uint8"

    if export_json:
        # Save image data in JSON format
        with open(filename[:len(filename) - 4] + ".json", 'w') as file_json:
            json.dump(jdata, file_json, sort_keys=False, indent=4, separators=(',', ': '))

    if export_nf:
        # Save image data in NF format
        with open(filename[:len(filename) - 4] + ".nf", 'w') as file_nf:
            sText = ''
            for k, v in jdata.items():
                sText += '%' + str(k) + '\n' + str(v) + '\n\n'
            sText = sText.replace('], ', '\n')
            sText = sText.replace('[', '')
            sText = sText.replace(']', '')
            sText = sText.replace(',', '')
            file_nf.write(sText)
