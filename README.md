[![license](https://img.shields.io/badge/license-MIT-green.svg?style=flat)](https://gitlab.com/cortezpedro/chfemgpu/-/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/chfem/badge/?version=latest)](https://chfem.readthedocs.io/en/latest/?badge=latest)

# chfem

`chfem`, which stands for _Computational Homogenization with the image-based Finite Element Method_, is a software written in C and CUDA C, wrapped in Python, for the computational homogenization of material samples characterized via $\mu$-CT. As it is, the effective properties that can be evaluated are:

+ Thermal conductivity $\rightarrow$ $\kappa\nabla^{2}u=0$
+ Linear elasticity $\rightarrow$ $\nabla\cdot\mathbf{C}\nabla\mathbf{u}=0$
+ Permeability $\rightarrow$ $\mu\nabla^{2}\mathbf{u}-\nabla p+\mathbf{b}=0$ ; $\nabla\cdot\mathbf{u} -\tau p=0$

The program follows a lightweight matrix-free approach to image-based finite element analysis, exploring GPU resources with CUDA to achieve significant performance gains. Our end goal is to be able to run large problems ($10^9$ DOFs) with relatively acessible graphics cards. Following is a visualization of the output from a permeability simulation.

<p align="center">
  <img src="https://github.com/cortezpedro/chfem/raw/dev/docs/source/chfem_example.png" width="100%"></img>
</p>

If you use `chfem` in your research, please use the following BibTeX entries to cite 
[our paper](https://doi.org/10.1016/j.commatsci.2023.112021):

```Bibtex
@article{toolkit2023,
    title = {Simulation toolkit for digital material characterization of large image-based microstructures},
    author = {Pedro C.F. Lopes and Rafael S. Vianna and Victor W. Sapucaia and Federico Semeraro and Ricardo Leiderman and André M.B. Pereira},
    journal = {Computational Materials Science},
    volume = {219},
    pages = {112021},
    year = {2023},
    publisher={Elsevier}
}
```
Developed at the _Laboratório de Computação Científica, Universidade Federal Fluminense_ (`LCC-UFF`). Niterói, Brazil.


## How to get started

There are two main ways to access the functionalities of `chfem`: through a python API or via an executable. We describe the installation of both next.
For beginner users, the python option is recommended. The executable is more flexible and potentially more optimizable, but it requires more setup and background knowledge,
so it is recommended for advanced users only.

### Prerequisites

You need an NVIDIA GPU to run `chfem`, either locally or remotely (e.g. Colab, AWS etc). It is also recommended to have a
[Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) or
[Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) installation and to work inside a conda environment:

```bash
conda create --name chfem python
conda activate chfem
```

If you have `nvcc` already installed in your system, you can skip to the next step. Otherwise, you can install it by using:

```bash
conda install conda-forge::cudatoolkit-dev
```

### Python Package

#### Installation

You can now install the `chfem` python package using:

```bash
pip install git+https://github.com/cortezpedro/chfem.git
```

#### Tutorial

You can follow [this Colab tutorial](https://colab.research.google.com/github/cortezpedro/chfem/blob/dev/tutorial.ipynb) to get started with `chfem`'s python API.

### Executable 

#### Installation

You can install and run the `chfem_exec` executable using:

```bash
git clone https://github.com/cortezpedro/chfem.git
cd chfem && mkdir -p build && cd build && cmake .. && make -j && cd ..
./chfem_exec -h
```

This will print the help information, confirming that the compilation was successfull. 
Next we describe the details of the executable input/outputs, as well as command line arguments and optional compiler flags. 

#### Run

You can run `chfem_exec` using:

```bash
./chfem [NEUTRAL_FILE] [RAW_IMAGE_FILE] <options>
```

Optional parameters:

```
-b: Save results in a binary file. Must be followed by a string with a filename.
-c: Stopping criteria for the PCG method: 0 - L2 (default), 1 - Inf, 2 - L2+Error.
-d: Target direction: 0 - X, 1 - Y, 2 - Z, 3 - YZ, 4 - XZ, 5 - XY, 6 - ALL (default).
-e: Export fields from simulation (by nodes).
-f: Input scalar density field. Must be followed by a [.bin] file.
-h: Print this help info and exit.
-i: Input files. Must be followed by: [.nf] [.raw].
-j: Jacobi preconditioning: 0 - no, 1 - yes (default).
-m: Write metrics report. Must be followed by a string with a filename.
-p: Parallel matrix-free strategy: 0 - NBN (default), 1 - EBE.
-pm: Pore mapping strategy: 0 - image, 1 - DOF number (default).
-r: Number of recursive searches for initial guesses.
-s: Solver: 0 - CG (default), 1 - MINRES, 2 - CG3, 3 - MINRES3, 4 - CG2, 5 - MINRES2.
-u: Reduce strategy for velocity fields (FLUID): 0 - on the fly, 1 - only diag, 2 - full.
-xi: Import initial guesses for PCG solver from binary files. Must be followed by a string with a filename.
-xo: Export result vector (x) from the PCG solver.
```

#### Input

As input, `chfem` expects a neutral file (`.nf`) [like this](https://github.com/cortezpedro/chfem/blob/dev/test/input/2D/thermal/100x100.nf) 
that contains textual descriptive info regarding analysis parameters, and a RAW file (`.raw`) containing an 8bit (0-255) grayscale value for each voxel (a raster grayscale image).

##### Neutral file

A textual file that sets analyses parameters and provides information about the model. Template `.nf`'s can be generated with [pyTomoviewer](https://github.com/LCC-UFF/pytomoviewer). 
The file must contain the following parameters (in this order):

```
%type_of_analysis
0
<uint8>
[0=THERMAL,1=ELASTIC,2=FLUID]

%voxel_size
1.0
<double>
[must be greater than 0.0]

%solver_tolerance
1e-06
<double>
[must be greater than 0.0]

%number_of_iterations
1000
<uint32>

%image_dimensions
100 100 0
<uint32> <uint32> <uint32>
[nx=COLS,ny=ROWS,nz=LAYERS]
[OBS.: nz=0 --> 2D]

%refinement
1
<uint32>

%number_of_materials
2
<uint8>

%properties_of_materials
0 1.0
255 10.0
<uint8> (<double>...)
[THERMAL: color, conductivity]                        [Example: 0 1.0]
[ELASTICITY: color, Young's modulus, Poisson's ratio] [Example: 0 1.0 0.3]
[FLUID: color (first one represents the pores)]       [Example: 0]

%data_type
float32
[single or double precison for the optional scalar density field input]
[float32 or float64]
```

##### Raw file

A binary representation of a raster image, with an 8-bit (uint8, 0-255) grayscale value per voxel. Each unique color value is taken as a material phase identifier in the given microstructure. In essence, the image is represented by an array, which can be generated from TIFF files with the [pyTomoviewer](https://github.com/LCC-UFF/pytomoviewer) application, or even it can be straightforwardly created with NumPy, for example.

Obs.: The provided image's voxels must be numbered from <u>left to right</u>, then from <u>top to bottom</u>, then from <u>near to far</u>.

#### Optional compiler flags:

+ Quiet iterations

\[DEFAULT=Not defined\] If defined, disables dynamic report of solver convergence.

```bash
-DCUDAPCG_QUIET_ITERATIONS
```

+ Floating point precision in the GPU

\[DEFAULT=CUDAPCG\_VAR\_64BIT\] Variable size for the GPU arrays.

```bash
-DCUDAPCG_VAR_32BIT
-DCUDAPCG_VAR_64BIT
```

+ Material key size

\[DEFAULT=CUDAPCG\_MATKEY\_16BIT\] Size of material keys. _Tip: if only interested in permeability, use 8BIT keys_.

```bash
-DCUDAPCG_MATKEY_8BIT
-DCUDAPCG_MATKEY_16BIT
-DCUDAPCG_MATKEY_32BIT
-DCUDAPCG_MATKEY_64BIT
```

+ Threads per block

\[DEFAULT=CUDAPCG\_BLOCKDIM\_128\] Number of threads per block for CUDA kernels. _Tip: Your hardware might benefit from lower block dimensions_.

```bash
-DCUDAPCG_BLOCKDIM_32
-DCUDAPCG_BLOCKDIM_64
-DCUDAPCG_BLOCKDIM_128
-DCUDAPCG_BLOCKDIM_256
-DCUDAPCG_BLOCKDIM_512
-DCUDAPCG_BLOCKDIM_1024
```

## Considerations

### Memory

| Analysis | Voxels  | DOFs \[$\times 10^6$\]    | CPU RAM \[GB\] | GPU RAM \[GB\]           |
|  :---:   | :---:   |        ---:               |     :---:      |     :---:                |
| THERMAL  | $400^3$ |       64                  |      1.5       | <32BIT> 1.1, <64BIT> 2.2 | 
| ELASTIC  | $400^3$ |      192                  |      4.0       | <32BIT> 3.2, <64BIT> 6.3 |
|  FLUID   | $400^3$ | $\approx 4\phi\times$ 64  |      1.3       | <32BIT> 1.3, <64BIT> 2.1 |

Obs.: These are memory **estimates**, for the sake of reference.

Obs.2: Porosity $\phi=0.25$ admitted for the FLUID analysis shown above.

Obs.3: Lower memory requirements can be achieved with the 3 and 2 vector solvers (CG3, CG2, MINRES3, MINRES2).

### Methods

+ Node-by-node
+ Element-by-element
+ Periodic boundary conditions
+ Linear shape functions for finite elements
+ Numerical stabilization for stokes flow

### Useful Resources

The following resources are also recommended to aid the analysis:
+ [pyTomoviewer](https://github.com/LCC-UFF/pytomoviewer), a GUI-based tool for viewing $\mu$-CT data depicted by stacks of TIFF files and generating `.nf` and `.raw` files that can serve as input for `chfem`.
+ [chpack](https://gitlab.com/lcc-uff/Chpack.jl), a bundle of Julia programs that employ similar matrix-free approaches in CPU, with script-like syntax.
+ Explanation of the matrix-free PCG solver in GPU for image-based FEM problems in periodic media
([link to paper](https://doi.org/10.1016/j.cma.2022.115276)).
```Bibtex
@article{lopes2022,
    title = {A GPU implementation of the PCG method for large-scale image-based finite element analysis in heterogeneous periodic media},
    author = {Pedro C.F. Lopes and André M.B. Pereira and Esteban Clua and Ricardo Leiderman},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {399},
    pages = {115276},
    year = {2022},
    publisher={Elsevier}
}
```
## FAQs

Obs.: Note that `nvcc` might not work out-of-the-box in Windows. A common issue when running our "compiler script" in Windows is:
```bash
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```
To solve this, we suggest installing [Microsoft Visual Studio Community](https://visualstudio.microsoft.com/vs/community/) with its C/C++ dependencies, then either adding `cl.exe`'s directory to your `PATH` or passing it to the compiler with the `-ccbin` flag. Alternatively, you may compile and run in MSVC.
