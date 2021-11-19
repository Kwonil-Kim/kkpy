import kkpy
from numpy.testing import assert_allclose, assert_almost_equal

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import datetime

def test_read_vertix():
    list_fname = [
        '/disk/STORAGE/OBS/VERTIX/ICN/DATA/NC/202108/03/VERTIX_ICN_202108030400.nc',
        [
            '/disk/STORAGE/OBS/VERTIX/ICN/DATA/NC/202108/03/VERTIX_ICN_202108030400.nc',
            '/disk/STORAGE/OBS/VERTIX/ICN/DATA/NC/202108/03/VERTIX_ICN_202108030500.nc'
        ]
    ]
    
    expected = [
        [310, 167, 128, 11, (310,167)],
        [310, 334, 128, 22, (310,334)]
    ]
    
    for i_test, fname in enumerate(list_fname):
        ds = kkpy.io.read_vertix(fname)
        assert isinstance(ds, xr.core.dataset.Dataset)
        assert ds.ngates.size == expected[i_test][0]
        assert ds.time_dwell.size == expected[i_test][1]
        assert ds.spec_x.size == expected[i_test][2]
        assert ds.time_spec.size == expected[i_test][3]
        assert isinstance(ds['reflectivity'], xr.core.dataarray.DataArray)


def test_read_vet():
    list_fname = [
        '/disk/STORAGE/OBS/Radar/MAPLE/202003/15/rdr_hsr_vel_202003151000.bin',
        [
            '/disk/STORAGE/OBS/Radar/MAPLE/202003/15/rdr_hsr_vel_202003151000.bin',
            '/disk/STORAGE/OBS/Radar/MAPLE/202003/15/rdr_hsr_vel_202003151010.bin'
        ]
    ]
    
    expected = [
        [25, 25, 1, (1,25,25), (1,25,25)],
        [25, 25, 2, (2,25,25), (2,25,25)]
    ]
    
    for i_test, fname in enumerate(list_fname):
        ds = kkpy.io.read_vet(fname)
        assert isinstance(ds, xr.core.dataset.Dataset)
        assert ds.x.size == expected[i_test][0]
        assert ds.y.size == expected[i_test][1]
        assert ds.t.size == expected[i_test][2]
        assert ds.u.shape == expected[i_test][3]
        assert ds.v.shape == expected[i_test][4]
        assert isinstance(ds['u'], xr.core.dataarray.DataArray)
        assert isinstance(ds['v'], xr.core.dataarray.DataArray)
        assert isinstance(ds['t'].values[0], (datetime.date,np.datetime64))
        assert isinstance(ds.projection, ccrs.LambertConformal)

def test_read_hsr():
    list_fname = [
        '/disk/STORAGE/OBS/Radar/HSR/COMP_KMA/202007/05/RDR_CMP_HSR_EXT_202007050010.bin.gz',
        [
            '/disk/STORAGE/OBS/Radar/HSR/COMP_KMA/202007/05/RDR_CMP_HSR_EXT_202007050010.bin.gz',
            '/disk/STORAGE/OBS/Radar/HSR/COMP_KMA/202007/05/RDR_CMP_HSR_EXT_202007050020.bin.gz'
        ]
    ]
    
    expected = [
        [2305, 2881, 1, (1,2881,2305)],
        [2305, 2881, 2, (2,2881,2305)]
    ]
    
    for i_test, fname in enumerate(list_fname):
        ds = kkpy.io.read_hsr(fname)
        assert isinstance(ds, xr.core.dataset.Dataset)
        assert ds.x.size == expected[i_test][0]
        assert ds.y.size == expected[i_test][1]
        assert ds.t.size == expected[i_test][2]
        assert ds.reflectivity.shape == expected[i_test][3]
        assert isinstance(ds['reflectivity'], xr.core.dataarray.DataArray)
        assert isinstance(ds['t'].values[0], (datetime.date,np.datetime64))
        assert pd.to_datetime(ds['t'].values[0]) == datetime.datetime(2020,7,5,0,10)
        assert isinstance(ds.projection, ccrs.LambertConformal)
        
def test_read_sounding():
    list_fname = [
        '/disk/STORAGE/OBS/SONDE/KMA/202108/20/UPP_RAW_47102_2021082000.txt',
        [
            '/disk/STORAGE/OBS/SONDE/KMA/202108/20/UPP_RAW_47102_2021082000.txt',
            '/disk/STORAGE/OBS/SONDE/KMA/201802/28/UPP_RAW_47102_2018022800.txt',
            '/disk/STORAGE/OBS/SONDE/JEJU_MINGUN/GRAW/KMA_FORMAT/202108/17/UPP_RAW_47500_20210817180000.txt',
            '/disk/STORAGE/OBS/SONDE/KMA/201807/10/UPP_RAW_47155_2018071012.txt'
        ]
    ]
    
    expected = [
        [5.81322172, (6637, 15), datetime.datetime(2021,8,19,23,18,24), -3.736667, -7.263500, datetime.datetime(2021,8,19,23,18,24)],
        [5.81322172, (22432, 15), datetime.datetime(2021,8,19,23,18,24), -3.736667, -7.263500, datetime.datetime(2018,7,10,11,20,35)]
    ]
    
    for i_test, fname in enumerate(list_fname):
        df = kkpy.io.read_sounding(fname)
        assert isinstance(df, pd.core.frame.DataFrame)
        assert_almost_equal(df.WS[0], expected[i_test][0])
        assert df.shape == expected[i_test][1]
        assert df.time[0] == expected[i_test][2]
        assert_almost_equal(df.U[0], expected[i_test][3])
        assert_almost_equal(df.Uknot[0], expected[i_test][4])
        assert df.time.iloc[-1] == expected[i_test][5]
        assert np.count_nonzero(
            np.isin(
                df.columns,
                ['P', 'T', 'RH', 'WS', 'WD', 'Lon', 'Lat', 'Alt', 'Geo', 'Dew', 'U', 'V', 'Uknot', 'Vknot', 'time']
            )) == 15

def test_read_lidar_wind():
    list_fname = [
        '/disk/STORAGE/OBS/LIDAR/BOSUNG/VAD/QC/201903/04/LDR_BSO_20190304120207.dat',
        [
            '/disk/STORAGE/OBS/LIDAR/BOSUNG/VAD/QC/201903/04/LDR_BSO_20190304120207.dat',
            '/disk/STORAGE/OBS/LIDAR/BOSUNG/VAD/QC/201903/04/LDR_BSO_20190304121144.dat',
        ]
    ]
    list_ftime = [
        datetime.datetime(2019,3,4,12,2,7),
        [
            datetime.datetime(2019,3,4,12,2,7), datetime.datetime(2019,3,4,12,11,44)
        ]
    ]
    
    expected = [
        [4.72, 7.00, 9.17494679697, (37, 10), datetime.datetime(2019,3,4,12,2,7), datetime.datetime(2019,3,4,12,2,7)],
        [4.72, 7.00, 9.17494679697, (73, 10), datetime.datetime(2019,3,4,12,2,7), datetime.datetime(2019,3,4,12,11,44)]
    ]
    
    for i_test, fname in enumerate(list_fname):
        df = kkpy.io.read_lidar_wind(fname, list_ftime[i_test])
        assert isinstance(df, pd.core.frame.DataFrame)
        assert_almost_equal(df.U[0], expected[i_test][0])
        assert_almost_equal(df.WS[0], expected[i_test][1])
        assert_almost_equal(df.Uknot[0], expected[i_test][2])
        assert df.shape == expected[i_test][3]
        assert df.time[0] == expected[i_test][4]
        assert df.time.iloc[-1] == expected[i_test][5]
        print(df.columns)
        assert np.count_nonzero(
            np.isin(
                df.columns,
                ['Alt', 'U', 'V', 'W', 'WS', 'WD', 'Valid', 'Uknot', 'Vknot', 'time']
            )) == 10
    
    # check if dropna=False works
    df = kkpy.io.read_lidar_wind(list_fname[1], list_ftime[1], dropna=False)
    assert df.shape == (688,10)

def test_read_kma_wpr():
    list_fname = [
        '/disk/STORAGE/OBS/WPR/KMA_API/202101/01/LWPU_47130_202101010600.DAT',
        [
            '/disk/STORAGE/OBS/WPR/KMA_API/202101/01/LWPU_47130_202101010600.DAT',
            '/disk/STORAGE/OBS/WPR/KMA_API/202101/01/LWPU_47130_202101010610.DAT',
            '/disk/STORAGE/OBS/WPR/KMA_API/202101/01/HWPU_47130_202101010610.DAT',
        ]
    ]
    list_ftime = [
        datetime.datetime(2021,1,1,6,0),
        [
            datetime.datetime(2021,1,1,6,0), datetime.datetime(2021,1,1,6,10), datetime.datetime(2021,1,1,6,10)
        ]
    ]
    
    expected = [
        [(71, 8), datetime.datetime(2021,1,1,6,0,0), datetime.datetime(2021,1,1,6,0,0), -0.05, 0.13, -0.09719223],
        [(213, 8), datetime.datetime(2021,1,1,6,0,0), datetime.datetime(2021,1,1,6,10,0), -0.05, 0.13, -0.09719223]
    ]
    
    for i_test, fname in enumerate(list_fname):
        df = kkpy.io.read_wpr_kma(fname, list_ftime[i_test])
        assert isinstance(df, pd.core.frame.DataFrame)
        assert df.shape == expected[i_test][0]
        assert df.time[0] == expected[i_test][1]
        assert df.time.iloc[-1] == expected[i_test][2]
        assert_almost_equal(df.U[0], expected[i_test][3])
        assert_almost_equal(df.WS[0], expected[i_test][4])
        assert_almost_equal(df.Uknot[0], expected[i_test][5])
        assert np.count_nonzero(
            np.isin(
                df.columns,
                ['Alt', 'WD', 'WS', 'U', 'V', 'Uknot', 'Vknot', 'time']
            )) == 8

def test_read_pluvio_raw():
    list_fname = [
        '/disk/STORAGE/OBS/PLUVIO/KNU/KNU01/ICN/202110/KNU01_pluvio200_20211010.txt',
        [
            '/disk/STORAGE/OBS/PLUVIO/KNU/KNU01/ICN/202110/KNU01_pluvio200_20211010.txt',
            '/disk/STORAGE/OBS/PLUVIO/KNU/KNU01/ICN/202110/KNU01_pluvio200_20211011.txt',
        ]
    ]
    
    expected = [
        [(14401, 8), datetime.datetime(2021,10,10,0,0,0), datetime.datetime(2021,10,10,23,59,54), 278.48, 787.09, 24.9],
        [(28802, 8), datetime.datetime(2021,10,10,0,0,0), datetime.datetime(2021,10,11,23,59,54), 278.48, 787.09, 24.9]
    ]
    
    for i_test, fname in enumerate(list_fname):
        df = kkpy.io.read_pluvio_raw(fname)
        assert isinstance(df, pd.core.frame.DataFrame)
        assert df.shape == expected[i_test][0]
        assert df.time[0] == expected[i_test][1]
        assert df.time.iloc[-1] == expected[i_test][2]
        assert_almost_equal(df.PATNRT[0], expected[i_test][3])
        assert_almost_equal(df.BRT[0], expected[i_test][4])
        assert_almost_equal(df['T'][0], expected[i_test][5])
        assert np.count_nonzero(
            np.isin(
                df.columns,
                ['PRT', 'PA', 'PNRT', 'PATNRT', 'BRT', 'BNRT', 'T', 'time']
            )) == 8
