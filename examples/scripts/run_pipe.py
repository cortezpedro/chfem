import chfem # pip install git+https://github.com/cortezpedro/chfem.git@dev
import numpy as np
from utils import img_utils as iu # requires numpy and pyvista
import os

# Input arguments
#########################################################################################################
SAMPLE = 'pipes'

DIMENSION = 256
THICKNESS = 10
DIRECTION = 'x'
VOXEL     = 1e-03 / DIMENSION # volume = 1mm^3
RADIUS    = DIMENSION / 4

SOLVER    = 'minres'
#SOLVER    = 'minres3'
#SOLVER    = 'minres2'
TOLERANCE = 1e-04
MAXIT     = 20_000

VIEW_SAMPLE_FLAG = True
VIEW_FIELDS_FLAG = (SOLVER != 'minres2')
#########################################################################################################

# Shape look-up table
#########################################################################################################
DIRECTION = DIRECTION.lower()

SHAPE_PER_DIRECTION_TABLE = { 'x': ( DIMENSION, DIMENSION, THICKNESS ),
                              'y': ( DIMENSION, THICKNESS, DIMENSION ),
                              'z': ( THICKNESS, DIMENSION, DIMENSION ) }

if DIRECTION not in SHAPE_PER_DIRECTION_TABLE:
  raise ValueError(f'Invalid direction: {DIRECTION}\nPossible values: {SHAPE_PER_DIRECTION_TABLE.keys()}')
SHAPE = SHAPE_PER_DIRECTION_TABLE[ DIRECTION ]
#########################################################################################################

# Function to create pipe model
#########################################################################################################
def create_cylinder(shape, position=None, radius=None, direction='z', color=1, img=None, dtype='uint8'):
  if img is None:
    img = np.zeros( shape, dtype=dtype )
  if direction.lower() == 'x':
    img = img.transpose(2,0,1) # x z y
    return create_cylinder(img.shape,position,radius,'z',color,img).transpose(1,2,0)
  elif direction.lower() == 'y':
    img = img.transpose(1,2,0) # y x z
    return create_cylinder(img.shape,position,radius,'z',color,img).transpose(2,0,1)
  if not (direction.lower() == 'z'):
    raise ValueError(f'unexpected direction: {direction}')
  if position is None:
    position = tuple( d/2 for d in shape[1:] )
  if radius is None:
    radius = np.min(shape) / 2
  x = np.zeros( shape[1:], dtype='float32' )
  y = np.zeros( shape[1:], dtype='float32' ).transpose()
  x[0,:] = np.linspace(0.5,shape[2]-0.5,shape[2],dtype='float32')
  x[:,:] = x[0,:]
  y[0,:] = np.linspace(shape[1]-0.5,0.5,shape[1],dtype='float32')
  y[:,:] = y[0,:]
  y = y.transpose()
  d = np.sqrt( (x-position[1])**2 + (y-position[0])**2 )
  img[ 0, d<=radius ] = color
  img[1:,:,:] = img[0,:,:]
  return img
#########################################################################################################

# Run simulations
#########################################################################################################
cwd = os.getcwd()

# Check if samples directory exists
if not os.path.exists(f'{cwd}/samples'):
  os.makedirs(f'{cwd}/samples')

print('Generating pipes model ...')
img = np.ones(SHAPE,dtype='u1')
img = create_cylinder(SHAPE, radius=RADIUS, direction=DIRECTION, color=0, img=img)
print('Done')

if VIEW_SAMPLE_FLAG:
  print('Rendering image ...')
  pore_color = 0
  solid_color = np.min(img[img>pore_color])
  my_cmap = lambda x: np.array( [ (0.,0.5,0.8,1.0) if d < solid_color/255 else (0.7,0.7,0.7,1.0) for d in x ]  )
  iu.pv_plot3D( img, cmap=my_cmap, opacity=1, label=f'voxels ({SAMPLE})')
  print('Done')

out_files = None
if VIEW_FIELDS_FLAG:
  out_files = f'{cwd}/samples/{SAMPLE}'

Keff = chfem.compute_property( 'permeability', img, voxel_size=VOXEL, direction=DIRECTION,
                               solver=SOLVER, solver_tolerance=TOLERANCE, solver_maxiter=MAXIT,
                               output_fields=out_files )
                               
Keff = np.array( Keff ) * 1e12 # conversion from m^2 to um^2
print('Permeability tensor, in um^2:')
print(Keff)

if VIEW_FIELDS_FLAG:
  print('Rendering velocity magnitude field ...')
  dir_int = 0 if DIRECTION=='x' else 1 if DIRECTION=='y' else 2 # if DIRECTION=='z'
  v = np.linalg.norm( chfem.import_vector_field_from_chfem(f'{out_files}_velocity_{dir_int}.bin',SHAPE), axis=0 )
  v *= VOXEL**2 # quadratic scaling by VOXEL size. domain of simulations is normalized
  iu.pv_plot3D(v, cmap='jet', opacity='linear', label='velocity magnitude [m/s]')
  print('Done')
#########################################################################################################
