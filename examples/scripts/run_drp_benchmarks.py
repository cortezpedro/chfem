import chfem # pip install git+https://github.com/cortezpedro/chfem.git@dev
import numpy as np
from utils import img_utils as iu # requires numpy and pyvista
import urllib.request
import os
import gzip

# Input arguments
#########################################################################################################
SAMPLE = 'fontainebleau'
#SAMPLE = 'berea'
#SAMPLE = 'grosmont'
#SAMPLE = 'spherepack'

SOLVER    = 'minres'
#SOLVER    = 'minres3'
#SOLVER    = 'minres2'
TOLERANCE = 1e-04
MAXIT     = 10_000

VIEW_SAMPLE_FLAG = True
VIEW_FIELDS_FLAG = (SOLVER != 'minres2')
#########################################################################################################

# Sample look-up table
#########################################################################################################
SAMPLE_TABLE = { 
  'fontainebleau': { 'raw': 'segmented-exxon.raw.gz',       'voxel_size': 7.50e-06, 'shape': ( 300, 288, 288), 'preprocess_image': None},
  'berea':         { 'raw': 'segmented-vsg.raw.gz',         'voxel_size': 0.74e-06, 'shape': (1024,1024,1024), 'preprocess_image': lambda x: x[:,150:-150,150:-150]},
  'grosmont':      { 'raw': 'segmented-vsg.raw.gz',         'voxel_size': 2.02e-06, 'shape': (1024,1024,1024), 'preprocess_image': lambda x: x[:512]},
  'spherepack':    { 'raw': 'segmented-788x791x793.raw.gz', 'voxel_size': 7.00e-06, 'shape': ( 793, 791, 788), 'preprocess_image': lambda x: x[:550]}
}

if SAMPLE not in SAMPLE_TABLE:
  raise ValueError(f'Invalid sample name: {SAMPLE}\nPossible values: {SAMPLE_TABLE.keys()}')

SAMPLE_DATA = SAMPLE_TABLE[ SAMPLE ]

SHAPE  = SAMPLE_DATA['shape']
VOXEL  = SAMPLE_DATA['voxel_size']
RAW_GZ = SAMPLE_DATA['raw']

URL = f'https://raw.githubusercontent.com/fkrzikalla/drp-benchmarks/master/images/{SAMPLE}/{RAW_GZ}'
#########################################################################################################

# Run simulations
#########################################################################################################
cwd = os.getcwd()
filename = f'{cwd}/samples/{SAMPLE}_{RAW_GZ}'

# Check if samples directory exists
if not os.path.exists(f'{cwd}/samples'):
  os.makedirs(f'{cwd}/samples')

# Check if file has already been downloaded:
if not os.path.isfile(filename):
  print(f'Downloading image from {URL} to {filename}...')
  urllib.request.urlretrieve(URL,filename)
  print('Done')

print(f'Extracting image from {filename} ...')
with gzip.open(filename, 'rb') as f:
  raw_data = f.read()
print('Done')

img = np.frombuffer(raw_data, dtype='u1').reshape(SHAPE)

if SAMPLE_DATA['preprocess_image'] is not None:
  img = SAMPLE_DATA['preprocess_image'](img)
  SHAPE = img.shape

if VIEW_SAMPLE_FLAG:
  print('Rendering image ...')
  pore_color = 0
  solid_color = np.min(img[img>pore_color])
  my_cmap = lambda x: np.array( [ (0.,0.5,0.8,1.0) if d < solid_color/255 else (0.7,0.7,0.7,1.0) for d in x ]  )
  iu.pv_plot3D( img, flip_z=True, cmap=my_cmap, opacity=1, label=f'voxels ({SAMPLE})')
  print('Done')

out_files = None
if VIEW_FIELDS_FLAG:
  out_files = f'{cwd}/samples/{SAMPLE}'

Keff = chfem.compute_property( 'permeability', img, voxel_size=VOXEL, direction='x',
                               solver=SOLVER, solver_tolerance=TOLERANCE, solver_maxiter=MAXIT,
                               output_fields=out_files )
                               
Keff = np.array( Keff ) * ( 1e15 / 0.9869233 ) # conversion from m^2 to mD
print('Permeability tensor, in milliDarcy [mD]:')
print(Keff)

if VIEW_FIELDS_FLAG:
  print('Rendering velocity magnitude field ...')
  v = np.linalg.norm( chfem.import_vector_field_from_chfem(f'{out_files}_velocity_0.bin',SHAPE), axis=0 )
  v *= VOXEL**2 # quadratic scaling by VOXEL size. domain of simulations is normalized
  mu = np.mean(v)
  sigma = np.std(v)
  mask = v > mu+2*sigma
  v[np.logical_not(mask)] = 0.
  iu.pv_plot3D(v, flip_z=True, cmap='jet', opacity='linear', label='velocity magnitude [m/s]')
  print('Done')
#########################################################################################################
