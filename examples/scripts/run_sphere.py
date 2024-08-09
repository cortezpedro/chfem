import chfem # pip install git+https://github.com/cortezpedro/chfem.git@dev
import numpy as np
from utils import img_utils as iu # requires numpy and pyvista
import os

# Input arguments
#########################################################################################################
SAMPLE = 'spheres'

DIMENSION = 256
SHAPE     = ( DIMENSION, DIMENSION, DIMENSION )
VOXEL     = 1e-03 / DIMENSION # volume = 1mm^3

SOLVER    = 'minres'
#SOLVER    = 'minres3'
#SOLVER    = 'minres2'
TOLERANCE = 1e-04
MAXIT     = 20_000

VIEW_SAMPLE_FLAG = True
VIEW_FIELDS_FLAG = (SOLVER != 'minres2')
#########################################################################################################

# Function to create sphere model
#########################################################################################################
def create_sphere( shape, position=None, radius=None, color=1, img=None, dtype='uint8' ):
  if position is None:
    position = tuple( d/2 for d in shape )
  if radius is None:
    radius = np.min(shape) / 2
  if img is None:
    img = np.zeros( shape, dtype=dtype )
  x = np.zeros( shape[1:], dtype='float32' )
  y = np.zeros( shape[1:], dtype='float32' ).transpose()
  x[0,:] = np.linspace(0.5,shape[2]-0.5,shape[2],dtype='float32')
  x[:,:] = x[0,:]
  y[0,:] = np.linspace(shape[1]-0.5,0.5,shape[1],dtype='float32')
  y[:,:] = y[0,:]
  y = y.transpose()
  for page in range(shape[0]):
    z = np.float32(page + 0.5)
    d = np.sqrt( (x-position[2])**2 + (y-position[1])**2 + (z-position[0])**2 )
    img[ page, d<=radius ] = color
  return img
#########################################################################################################

# Run simulations
#########################################################################################################
cwd = os.getcwd()

# Check if samples directory exists
if not os.path.exists(f'{cwd}/samples'):
  os.makedirs(f'{cwd}/samples')

print('Generating sphere model ...')
img = create_sphere(SHAPE)
print('Done')

if VIEW_SAMPLE_FLAG:
  print('Rendering image ...')
  pore_color = 0
  solid_color = np.min(img[img>pore_color])
  my_cmap = lambda x: np.array( [ (0.,0.5,0.8,1.0) if d < solid_color/255 else (0.7,0.7,0.7,1.0) for d in x ]  )
  iu.pv_plot3D( img, cmap=my_cmap, opacity='foreground', label=f'voxels ({SAMPLE})')
  print('Done')

out_files = None
if VIEW_FIELDS_FLAG:
  out_files = f'{cwd}/samples/{SAMPLE}'

Keff = chfem.compute_property( 'permeability', img, voxel_size=VOXEL, direction='x',
                               solver=SOLVER, solver_tolerance=TOLERANCE, solver_maxiter=MAXIT,
                               output_fields=out_files )
                               
Keff = np.array( Keff ) * 1e12 # conversion from m^2 to um^2
print('Permeability tensor, in um^2:')
print(Keff)

if VIEW_FIELDS_FLAG:
  print('Rendering velocity magnitude field ...')
  v = np.linalg.norm( chfem.import_vector_field_from_chfem(f'{out_files}_velocity_0.bin',SHAPE), axis=0 )
  v *= VOXEL**2 # quadratic scaling by VOXEL size. domain of simulations is normalized
  iu.pv_plot3D(v, cmap='jet', opacity='linear', label='velocity magnitude [m/s]')
  print('Done')
#########################################################################################################
