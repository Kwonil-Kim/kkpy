"""
kkpy.io
========================

Functions to read and write files

.. currentmodule:: io

.. autosummary::
    kkpy.io.read_aws
    kkpy.io.read_2dvd_rho
    kkpy.io.read_mxpol_rhi_with_hc
    kkpy.io.read_dem

"""
import numpy as np
import pandas as pd
import datetime
import glob
import os
import sys

def read_aws(time, date_range=True, datadir='/disk/STORAGE/OBS/AWS/', stnid=None, dask=True):
    """
    Read AWS_MIN files into dataframe.
    
    Examples
    ---------
    >>> import datetime
    >>> df_aws = kkpy.io.read_aws(time=datetime.datetime(2018,2,28,6,0))
    
    >>> df_aws = kkpy.io.read_aws(time=[datetime.datetime(2018,2,28,6,0),datetime.datetime(2018,3,1,12,0)], datadir='/path/to/aws/files/')
    
    Parameters
    ----------
    time : datetime or array_like of datetime
        Datetime of the data you want to read.
        If this is array of two elements, it will read all data within two datetimes by default.
        If this is array of elements and keyword *date_range* is False, it will read the data of specific time of each element.
    date_range : bool, optional
        False if argument *time* contains element of specific time you want to read.
    datadir : str, optional
        Directory of data.
    stnid : list, optional
        List of station id you want to read. Read all site if None.
    dask : boolean, optional
        Return a dask dataframe if True, otherwise return a pandas dataframe.
        
    Returns
    ---------
    df_aws : dataframe
        Return dataframe of aws data.
    """
    import dask.dataframe as dd
    
    if time is None:
        sys.exit(f'{__name__}: Check time argument')
    
    if len(time) == 1:
        date_range = False
    
    if date_range:
        if len(time) != 2:
            sys.exit(f'{__name__}: Check time and date_range arguments')
        if time[0] >= time[1]:
            sys.exit(f'{__name__}: time[1] must be greater than time[0]')
        
        dt_start = datetime.datetime(time[0].year, time[0].month, time[0].day, time[0].hour, time[0].minute)
        dt_finis = datetime.datetime(time[1].year, time[1].month, time[1].day, time[1].hour, time[1].minute)
        
        # Get file list
        filearr = np.array([])
        _dt = dt_start
        while _dt <= dt_finis:
            _filearr = np.sort(glob.glob(f'{datadir}/{_dt:%Y%m}/{_dt:%d}/AWS_MIN_{_dt:%Y%m%d%H%M}'))
            filearr = np.append(filearr, _filearr)
            _dt = _dt + datetime.timedelta(minutes=1)
        yyyy_filearr = [np.int(os.path.basename(x)[-12:-8]) for x in filearr]
        mm_filearr = [np.int(os.path.basename(x)[-8:-6]) for x in filearr]
        dd_filearr = [np.int(os.path.basename(x)[-6:-4]) for x in filearr]
        hh_filearr = [np.int(os.path.basename(x)[-4:-2]) for x in filearr]
        ii_filearr = [np.int(os.path.basename(x)[-2:]) for x in filearr]
        dt_filearr = np.array([datetime.datetime(yyyy,mm,dd,hh,ii) for (yyyy,mm,dd,hh,ii) in zip(yyyy_filearr, mm_filearr, dd_filearr, hh_filearr, ii_filearr)])

        filearr = filearr[(dt_filearr >= dt_start) & (dt_filearr <= dt_finis)]
        dt_filearr = dt_filearr[(dt_filearr >= dt_start) & (dt_filearr <= dt_finis)]
        
    else:
        list_dt_yyyymmddhhii = np.unique(np.array([datetime.datetime(_time.year, _time.month, _time.day, _time.hour, _time.minute) for _time in time]))
        
        filearr = np.array([f'{datadir}/{_dt:%Y%m}/{_dt:%d}/AWS_MIN_{_dt:%Y%m%d%H%M}' for _dt in list_dt_yyyymmddhhii])
        dt_filearr = list_dt_yyyymmddhhii
    
    if len(filearr) == 0:
        sys.exit(f'{__name__}: No matched data for the given time period')
    
    
    df_list = []
    names = ['ID', 'YMDHI', 'LON', 'LAT', 'HGT',
             'WD', 'WS', 'T', 'RH',
             'PA', 'PS', 'RE',
             'R60mAcc', 'R1d', 'R15m', 'R60m',
             'WDS', 'WSS', 'dummy']
    
    df_aws = dd.read_csv(filearr.tolist(), delimiter='#', names=names, header=None, na_values=[-999,-997])
    df_aws = df_aws.drop('dummy', axis=1)
    df_aws.WD      = df_aws.WD/10.
    df_aws.WS      = df_aws.WS/10.
    df_aws.T       = df_aws['T']/10.
    df_aws.RH      = df_aws.RH/10.
    df_aws.PA      = df_aws.PA/10.
    df_aws.PS      = df_aws.PS/10.
    df_aws.RE      = df_aws.RE/10.
    df_aws.R60mAcc = df_aws.R60mAcc/10.
    df_aws.R1d     = df_aws.R1d/10.
    df_aws.R15m    = df_aws.R15m/10.
    df_aws.R60m    = df_aws.R60m/10.
    df_aws.WDS     = df_aws.WDS/10.
    df_aws.WSS     = df_aws.WSS/10.
    if stnid:
        df_aws = df_aws[df_aws['ID'].isin(stnid)]

    df_aws = df_aws.set_index(dd.to_datetime(df_aws['YMDHI'], format='%Y%m%d%H%M'))
    df_aws = df_aws.drop('YMDHI', axis=1)
    
    if dask:
        return df_aws
    else:
        return df_aws.compute()


def read_2dvd_rho(time, date_range=True, datadir='/disk/common/kwonil_rainy/RHO_2DVD/', filename='2DVD_Dapp_v_rho_201*Deq.txt'):
    """
    Read 2DVD density files into dataframe.
    
    Examples
    ---------
    >>> import datetime
    >>> df_2dvd_drop = kkpy.io.read_2dvd_rho(time=datetime.datetime(2018,2,28)) # automatically date_range=False
    
    >>> df_2dvd_drop = kkpy.io.read_2dvd_rho(time=[datetime.datetime(2018,2,28,6),datetime.datetime(2018,3,1,12)], datadir='/path/to/2dvd/files/')
    
    >>> df_2dvd_drop = kkpy.io.read_2dvd_rho(time=list_of_many_datetimes, date_range=False)
    
    >>> df_2dvd_drop = kkpy.io.read_2dvd_rho(time=datetime.datetime(2018,2,28), filename='2DVD_rho_test_*.txt')
    
    Parameters
    ----------
    time : datetime or array_like of datetime
        Datetime of the data you want to read.
        If this is array of two elements, it will read all data within two datetimes by default.
        If this is array of elements and keyword *date_range* is False, it will read the data of specific time of each element.
    date_range : bool, optional
        False if argument *time* contains element of specific time you want to read.
    datadir : str, optional
        Directory of data.
    filename : str, optional
        File naming of data.
        
    Returns
    ---------
    df_2dvd_drop : dataframe
        Return dataframe of 2dvd data.
    """
    # Get file list
    filearr = np.array(np.sort(glob.glob(f'{datadir}/**/{filename}', recursive=True)))
    yyyy_filearr = [np.int(os.path.basename(x)[-27:-23]) for x in filearr]
    mm_filearr = [np.int(os.path.basename(x)[-23:-21]) for x in filearr]
    dd_filearr = [np.int(os.path.basename(x)[-21:-19]) for x in filearr]
    dt_filearr = np.array([datetime.datetime(yyyy,mm,dd) for (yyyy, mm, dd) in zip(yyyy_filearr, mm_filearr, dd_filearr)])

    if time is None:
        sys.exit(f'{__name__}: Check time argument')
    
    if len(time) == 1:
        date_range = False
    
    if date_range:
        if len(time) != 2:
            sys.exit(f'{__name__}: Check time and date_range arguments')
        if time[0] >= time[1]:
            sys.exit(f'{__name__}: time[1] must be greater than time[0]')
        
        dt_start = datetime.datetime(time[0].year, time[0].month, time[0].day)
        dt_finis = datetime.datetime(time[1].year, time[1].month, time[1].day)
        
        
        filearr = filearr[(dt_filearr >= dt_start) & (dt_filearr <= dt_finis)]
        dt_filearr = dt_filearr[(dt_filearr >= dt_start) & (dt_filearr <= dt_finis)]
    else:
        list_dt_yyyymmdd = np.unique(np.array([datetime.datetime(_time.year, _time.month, _time.day) for _time in time]))
        
        filearr = filearr[np.isin(dt_filearr, list_dt_yyyymmdd)]
        dt_filearr = dt_filearr[np.isin(dt_filearr, list_dt_yyyymmdd)]
    
    if len(filearr) == 0:
        sys.exit(f'{__name__}: No matched data for the given time period')
        
    # # READ DATA
    columns = ['hhmm', 'Dapp', 'VEL', 'RHO', 'AREA', 'WA', 'HA', 'WB', 'HB', 'Deq']
    dflist = []
    for i_file, (file, dt) in enumerate(zip(filearr, dt_filearr)):
        _df = pd.read_csv(file, skiprows=1, names=columns, header=None, delim_whitespace=True)
        _df['year'] = dt.year
        _df['month'] = dt.month
        _df['day'] = dt.day
        _df['hour'] = np.int_(_df['hhmm'] / 100)
        _df['minute'] = _df['hhmm'] % 100
        _df['jultime'] = pd.to_datetime(_df[['year','month','day','hour','minute']])
        _df = _df.drop(['hhmm','year','month','day','hour','minute'], axis=1)
        dflist.append(_df)
        print(i_file+1, filearr.size, file)

    df_2dvd_drop = pd.concat(dflist, sort=False, ignore_index=True)
    df_2dvd_drop.set_index('jultime', inplace=True)

    if date_range:
        if np.sum([np.sum([_time.hour, _time.minute, _time.second]) for _time in time]) != 0:
            df_2dvd_drop = df_2dvd_drop.loc[time[0]:time[1]]
    
    return df_2dvd_drop


def read_mxpol_rhi_with_hc(rhifile_nc, hcfile_mat):
    """
    Read MXPOL RHI with hydrometeor classification into py-ART radar object.
    
    Examples
    ---------
    >>> rhifile = '/disk/WORKSPACE/kwonil/MXPOL/RAW/2018/02/28/MXPol-polar-20180228-065130-RHI-225_8.nc'
    >>> hidfile = '/disk/WORKSPACE/kwonil/MXPOL/HID/2018/02/28/MXPol-polar-20180228-065130-RHI-225_8_zdrcorr_demix.mat'
    >>> radar_mxp = kkpy.io.read_mxpol_rhi_with_hc(rhifile, hcfile)
    
    Parameters
    ----------
    rhifile_nc : str or array_like of str
        Filepath of RHI data to read.
        The number and the order of elements should match with `hcfile_mat`.
    hcfile_mat : str or array_like of str
        Filepath of hydrometeor classification file to read.
        The number and the order of elements should match with `rhifile_nc`.
        
    Returns
    ---------
    radar : py-ART radar object
        Return py-ART radar object.
    """
    os.environ['PYART_QUIET'] = "True"
    import pyart
    import scipy.io
    from netCDF4 import Dataset
    
    # HC file
    HC_proportion = scipy.io.loadmat(hcfile_mat)
    # RHI file
    mxpol = Dataset(rhifile_nc,'r')
    El = mxpol.variables['Elevation'][:]
    wh_hc = np.logical_and(El>5,El<175)
    El = El[wh_hc]
    R = mxpol.variables['Range'][:]

    radar = pyart.testing.make_empty_rhi_radar(HC_proportion['AG'].shape[1], HC_proportion['AG'].shape[0], 1)

    ######## HIDs ########
    # find most probable habit
    for i, _HC in HC_proportion.items():
        if '_' in i: continue
            
        if i in 'AG':
            HC3d_proportion = np.array(HC_proportion[i])
        else:
            HC3d_proportion = np.dstack([HC3d_proportion, HC_proportion[i]])
    HC = np.float_(np.argmax(HC3d_proportion, axis=2))
    HC[np.isnan(HC3d_proportion[:,:,0])] = np.nan
    
    # add to PYART radar fields
    list_str = [
        'AG', 'CR', 'IH',
        'LR', 'MH', 'RN',
        'RP', 'WS']
    list_standard = [
        'Aggregation', 'Crystal', 'Ice hail / Graupel',
        'Light rain', 'Melting hail', 'Rain',
        'Rimed particles', 'Wet snow']
    for _str, _standard in zip(list_str, list_standard):
        mask_dict = {
            'data':HC_proportion[_str], 'unit':'-',
            'long_name':f'Proportion of the {_str}',
            '_FillValue':-9999, 'standard_name':_standard}
        radar.add_field(_str, mask_dict, replace_existing=True)
    
    radar.add_field('HC',
                    {'data':HC, 'unit':'-',
                     'long_name':f'Most probable habit. AG(0), CR(1), IH(2), LR(3), MH(4), RN(5), RP(6), WS(7)',
                     '_FillValue':-9999, 'standard_name':'Hydrometeor classification'},
                    replace_existing=True)
    
    ######## Radar variables ########
    ZDR = mxpol.variables['Zdr'][:].T[wh_hc]
    Z   = mxpol.variables['Zh'][:].T[wh_hc]
    KDP = mxpol.variables['Kdp'][:].T[wh_hc]

    mask_dict = {
        'data':KDP, 'unit':'deg/km',
        'long_name': 'differential phase shift',
        '_FillValue':-9999, 'standard_name':'KDP'
    }
    radar.add_field('KDP', mask_dict)

    mask_dict = {
        'data':ZDR-4.5, 'unit':'dB',
        'long_name': 'differential reflectivity',
        '_FillValue':-9999, 'standard_name':'ZDR'
    }
    radar.add_field('ZDR', mask_dict)

    mask_dict = {
        'data':Z, 'unit':'dBZ',
        'long_name': 'horizontal reflectivity',
        '_FillValue':-9999, 'standard_name':'ZHH'
    }
    radar.add_field('ZHH', mask_dict)


    radar.range['data'] = R
    radar.elevation['data'] = El
    azimuth = np.array(mxpol['Azimuth'][:][wh_hc])
    if azimuth[0] < 0: azimuth += 360
    radar.azimuth['data'] = azimuth
    radar.fixed_angle['data'] = azimuth
    radar.time['data'] = np.array(mxpol.variables['Time'][:])
    radar.time['units'] = "seconds since 1970-01-01T00:00:00Z"
    radar.longitude['data'] = np.array([mxpol.getncattr('Longitude-value')])
    radar.latitude['data'] = np.array([mxpol.getncattr('Latitude-value')])
    radar.metadata['instrument_name'] = 'MXPol'
    radar.altitude['data'] = np.array([mxpol.getncattr('Altitude-value')])
    
    return radar

def read_dem(file=None, area='pyeongchang'):
    """
    Read NASA SRTM 3-arcsec (90 meters) digital elevation model in South Korea.
    
    Examples
    ---------
    >>> dem, lon_dem, lat_dem, proj_dem = kkpy.io.read_dem(area='pyeongchang')
    >>> ax = plt.subplot(projection=ccrs.PlateCarree())
    >>> pm = ax.pcolormesh(lon_dem, lat_dem, dem.T, cmap=cmap, vmin=0, transform=ccrs.PlateCarree())
    
    >>> dem, lon_dem, lat_dem, proj_dem = kkpy.io.read_dem(area='korea')
    
    >>> dem, lon_dem, lat_dem, proj_dem = kkpy.io.read_dem(file='./pyeongchang_90m.tif')
    
    Parameters
    ----------
    file : str, optional
        Filepath of .tif DEM file to read.
    area : str, optional
        Region of interest. Possible options are 'pyeongchang' and 'korea'. Default is 'pyeongchang'.
        
    Returns
    ---------
    dem : float 2D array
        Return DEM elevation.
    lon_dem : float 2D array
        Return longitude of each DEM pixel.
    lat_dem : float 2D array
        Return latitude of each DEM pixel.
    proj_dem : osr object
        Spatial reference system of the used coordinates.
    """
    
    import wradlib as wrl
    
    if file is not None:
        ds = wrl.io.open_raster(file)
    else:
        if area in 'pyeongchang':
            ds = wrl.io.open_raster('/disk/WORKSPACE/kwonil/SRTM3_V2.1/TIF/pyeongchang_90m.tif')
        elif area in 'korea':
            ds = wrl.io.open_raster('/disk/WORKSPACE/kwonil/SRTM3_V2.1/TIF/korea_90m.tif')
        else:
            print('Please check area argument')
            
    dem, coord, proj_dem = wrl.georef.extract_raster_dataset(ds)
    lon_dem = coord[:,:,0]
    lat_dem = coord[:,:,1]
    dem = dem.astype(float)
    dem[dem <= 0] = np.nan
    dem = dem.T

    return dem, lon_dem, lat_dem, proj_dem