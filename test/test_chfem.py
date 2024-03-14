import chfem
import numpy as np

np.set_printoptions(linewidth=200)

def test_conductivity():
    array = chfem.import_raw("test/input/3D/ggg40_100.raw", (100, 100, 100))
    array = array.transpose(2, 1, 0)
    coeff = chfem.compute_property('conductivity', array, direction='x', nf_filepath="test/input/3D/thermal/ggg40_100.nf")
    coeff_ref = np.fromfile("test/input/3D/thermal/ggg40_100_result.bin", dtype=np.double, count=36).reshape(3, 3)
    np.testing.assert_almost_equal(coeff, coeff_ref, decimal=4)

def test_elasticity():
    array = chfem.import_raw("test/input/3D/ggg40_100.raw", (100, 100, 100))
    array = array.transpose(2, 1, 0)
    coeff = chfem.compute_property('elasticity', array, direction='y', nf_filepath="test/input/3D/elastic/ggg40_100.nf")
    coeff_ref = np.fromfile("test/input/3D/elastic/ggg40_100_result.bin", dtype=np.double, count=36).reshape(6, 6)
    np.testing.assert_almost_equal(coeff, coeff_ref, decimal=3)

 def test_permeability():
    array = chfem.import_raw("test/input/3D/fibers_100x100x10.raw", (10, 100, 100))
    array = array.transpose(2, 1, 0)
    coeff = chfem.compute_property('permeability', array, direction='z', nf_filepath="test/input/3D/fluid/fibers_100x100x10.nf")
    coeff_ref = np.fromfile("test/input/3D/fluid/fibers_100x100x10_result.bin", dtype=np.double, count=36).reshape(3, 3)
    np.testing.assert_almost_equal(coeff, coeff_ref, decimal=14)
