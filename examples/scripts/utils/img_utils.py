import numpy as np
import pyvista as pv

def periodic_index(shape, kk, ii, jj):
    slices, rows, cols = shape
    return ((jj + cols) % cols + ((ii + rows) % rows) * cols + ((kk + slices) % slices) * rows * cols)

def parse_nf(filename, out='all'):
  with open(filename) as f:
    lines = [line.rstrip() for line in f]
  ii = lines.index('%refinement')
  ref = int(lines[ii+1])
  ii = lines.index('%voxel_size')
  voxel_size = float(lines[ii+1])/ref
  ii = lines.index('%properties_of_materials')
  pores = int(lines[ii+1])
  ii = lines.index('%image_dimensions')
  dim_list = lines[ii+1].split()
  shape = ( int(dim_list[2]) , int(dim_list[1]) , int(dim_list[0]) )
  if out.lower() == 'shape':
    return shape
  if out.lower() == 'voxel_size':
    return voxel_size
  if out.lower() == 'pore_color':
    return pores
  if out.lower() == 'refinement':
    return ref
  return shape, voxel_size, pores, ref

def copy_nf(nf_original, nf_temp, shape):
  with open(nf_original) as f:
    lines = [line.rstrip() for line in f]
  slices, rows, cols = shape
  ii = lines.index('%image_dimensions')
  lines[ii+1] = f'{rows} {cols} {slices}'
  with open(nf_temp, 'w') as f:
      for line in lines:
          f.write(f"{line}\n")
    
def load_raw(shape, filename):
  return np.reshape( np.fromfile(filename, dtype='uint8'), shape )

def pv_plot3D(img, flip_z=False, cmap='Greys', opacity=None, label='voxels', outline=True, scalar_bar_args=None, render_now=True):
  grid = pv.ImageData()
  xyz = ( img.shape[2], img.shape[1], img.shape[0] )
  grid.dimensions = xyz 
  grid.point_data[label] = np.reshape( img, (img.size) )
  if not flip_z:
    grid.point_data[label] = np.reshape( img, (img.size) )
  else:
    grid.point_data[label] = np.reshape( img[-1::-1,-1::-1,:], (img.size) )
  plotter = pv.Plotter()
  if opacity is None:
    plotter.add_volume(grid, scalars=label, cmap=cmap, scalar_bar_args=scalar_bar_args)
  else:
    plotter.add_volume(grid, scalars=label, cmap=cmap, opacity=opacity, scalar_bar_args=scalar_bar_args)
  if outline:
    plotter.add_mesh(grid.outline(), color='k')
  ax = plotter.add_axes(interactive=True)
  if flip_z:
    plotter.camera_position = [(xyz[0]*2.8, -xyz[1]*1.8, -xyz[0]*1.2),
                               (xyz[0]*0.8, xyz[1]*0.4, xyz[2]*0.4),
                               (-0.29,0.28,-0.9144)]
  if not render_now:
    return plotter
  plotter.show()
  return None

def gaussian_opacity(img, sigma_multiplier=1., n=101):
  mu = np.mean(img)
  sigma = np.std(img)
  sigma *= sigma_multiplier
  return 1. - np.exp( -( np.linspace(np.min(img),np.max(img),n) -mu  )**2 / (2*sigma*sigma) )
