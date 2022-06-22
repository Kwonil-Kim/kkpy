import kkpy
from numpy.testing import assert_allclose, assert_almost_equal

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import datetime

def test_get_intersect():
    arr1 = np.array([0,1,2,4,5,6])
    arr2 = np.array([0,1,3,4,6])
    assert_allclose(
        kkpy.util.get_intersect([arr1, arr2]),
        [0,1,4,6]
    )

def test_to_lower_resolution():
    arr = np.random.rand(250,150)
    assert kkpy.util.to_lower_resolution(arr,10,10).shape == (25,15)
    assert kkpy.util.to_lower_resolution(arr,5,10).shape == (50,15)

def test_derivative():
    xarr = np.arange(100)*0.1
    yarr = np.exp(xarr)
    dydx = kkpy.util.derivative(yarr, 5, pixelsize=0.1)
    assert dydx.shape == (100,)
    assert_almost_equal(dydx[0], 1.10701379)
    assert_almost_equal(dydx[-1], 18063.81620107)

    np.random.seed(0)
    arr2d = np.random.rand(100,100)
    dzdx = kkpy.util.derivative(arr2d, 11, pixelsize=3000, axis=1)
    assert dzdx.shape == (100, 100)
    assert_almost_equal(dzdx[0,0], 6.472040e-06)
    assert_almost_equal(dzdx[-1,-1], 5.204856e-06)

