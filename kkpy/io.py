"""
kkpy.io
========================

Functions to read and write files

.. currentmodule:: io

.. autosummary::
    kkpy.io.get_fname
    kkpy.io.read_aws
    kkpy.io.read_2dvd_rho
    kkpy.io.read_mxpol_rhi_with_hc
    kkpy.io.read_dem
    kkpy.io.read_wissdom
    kkpy.io.read_vertix
    kkpy.io.read_vet
    kkpy.io.read_hsr
    kkpy.io.read_d3d
    kkpy.io.read_r3d
    kkpy.io.read_sounding
    kkpy.io.read_lidar_wind
    kkpy.io.read_wpr_kma
    kkpy.io.read_pluvio_raw
    kkpy.io.read_2dvd
    kkpy.io.read_wxt520

"""
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import glob
import os
import sys

def read_aws(time, date_range=True, datadir='/disk/STORAGE/OBS/AWS/', stnid=None, dask=True):
    """
    Read AWS (AWS_MIN).
    
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
    
    Warnings
    ---------
    This routine will be updated to receive argument for only filenames. The syntax will be likely changed in the near future.
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
        yyyy_filearr = [int(os.path.basename(x)[-12:-8]) for x in filearr]
        mm_filearr = [int(os.path.basename(x)[-8:-6]) for x in filearr]
        dd_filearr = [int(os.path.basename(x)[-6:-4]) for x in filearr]
        hh_filearr = [int(os.path.basename(x)[-4:-2]) for x in filearr]
        ii_filearr = [int(os.path.basename(x)[-2:]) for x in filearr]
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
    yyyy_filearr = [int(os.path.basename(x)[-27:-23]) for x in filearr]
    mm_filearr = [int(os.path.basename(x)[-23:-21]) for x in filearr]
    dd_filearr = [int(os.path.basename(x)[-21:-19]) for x in filearr]
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
    >>> pm = ax.pcolormesh(lon_dem, lat_dem, dem, cmap=cmap, vmin=0, transform=ccrs.PlateCarree())
    
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
    import cartopy.crs as ccrs
    
    if area in 'pyeongchang':
        dem = np.load('/disk/WORKSPACE/kwonil/SRTM3_V2.1/NPY/pyeongchang_90m_dem.npy')
        lon_dem = np.load('/disk/WORKSPACE/kwonil/SRTM3_V2.1/NPY/pyeongchang_90m_lon.npy')
        lat_dem = np.load('/disk/WORKSPACE/kwonil/SRTM3_V2.1/NPY/pyeongchang_90m_lat.npy')
    elif area in 'korea':
        dem = np.load('/disk/WORKSPACE/kwonil/SRTM3_V2.1/NPY/korea_90m_dem.npy')
        lon_dem = np.load('/disk/WORKSPACE/kwonil/SRTM3_V2.1/NPY/korea_90m_lon.npy')
        lat_dem = np.load('/disk/WORKSPACE/kwonil/SRTM3_V2.1/NPY/korea_90m_lat.npy')
    else:
        sys.exit(f'{__name__}: Check area argument')
    
    proj_dem = ccrs.PlateCarree()
    
    return dem, lon_dem, lat_dem, proj_dem

def get_fname(indir, pattern, dt, date_range=True, verbose=True):
    """
    Get filename corresponding to the given datetime(s) and format.
    
    Examples
    ---------
    >>> # Get radar filename
    >>> fname, fdatetime = get_fname('/disk/STORAGE/OBS/Radar/ICE-POP/PPI/NOQC/KST/',
    >>>                              '%Y%m/%d/RDR_GNG_%Y%m%d%H%M.uf',
    >>>                              [datetime.datetime(2018,2,28,15,0), datetime.datetime(2018,3,2,16,0)])

    >>> # Get AWS filename (no extension)
    >>> fname, fdatetime = get_fname('/disk/STORAGE/OBS/AWS/',
    >>>                              '%Y%m/%d/AWS_MIN_%Y%m%d%H%M',
    >>>                              [datetime.datetime(2018,1,22,5,30), datetime.datetime(2018,1,23,4,28)])
    
    >>> # Get MRR filename (duplicate format - %m)
    >>> fname, fdatetime = get_fname('/disk/STORAGE/OBS/MRR/AveData/',
    >>>                               '%Y%m/%m%d.ave',
    >>>                               [datetime.datetime(2015,8,15,5,30), datetime.datetime(2015,8,17,4,28)])

    >>> # Get 2DVD filename (one datetime, the use of DOY - %j)
    >>> fname, fdatetime = get_fname('/disk/STORAGE/OBS/2DVD/2dvddata/hyd/',
    >>>                              'V%y%j_1.txt',
    >>>                              datetime.datetime(2012,2,3))
    
    >>> # Get MRR-PRO filename (multiple datetimes with pandas)
    >>> import pandas as pd
    >>> fname, fdatetime = get_fname('/disk/STORAGE/OBS/MRR-PRO/',
    >>>                              '%Y%m/%Y%m%d/%Y%m%d_%H%M%S.nc',
    >>>                              pd.date_range(start='2020-06-01', end='2020-08-31', freq='1D'),
    >>>                              date_range=False,
    >>>                              verbose=True)

    Parameters
    ----------
    indir : str
        Path of the root directory. This should **not** have any format string.
    pattern : str
        Datetime pattern to match. The directory can be formatted here (eg. %Y%m/%d/sitename/data_%Y%m%d%H%M%S.csv).
        See format code description: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes.
    dt : datetime or list of datetime
        Datetime of interest to match.
        If a datetime object, find one matched file.
        If list of datetime objects, find matched files for specific datetimes if **`date_range` is False**.
        If list of two datetime objects and **`date_range` is True**, find all matched files within two datetimes.
    date_range : boolean, optional
        True if find all matched files within two datetimes. The number of dt should be two if `date_range` is True.
        Return matched files for specific datetimes of `dt` if False. Default is True.
    verbose : boolean, optional
        If True, print warnings 'File does not exist' if **`date_range` is False**. 
        
    Returns
    ---------
    fname : str or list of str
        Return filename of matched files.
    fdt : datetime or list of datetime
        Return datetime of matched files.
    """
    import warnings
    
    # check indir
    if indir.find('%') != -1:
        raise UserWarning('indir should not have datetime format')
    
    # check pattern
    if pattern.find('%') == -1:
        raise UserWarning('pattern should have at least one datetime format (eg. %Y%m%d)')
    
    # check dt
    is_dt_list = isinstance(dt, list) | isinstance(dt, np.ndarray)
    if not is_dt_list:
        if isinstance(dt, pd.DatetimeIndex):
            dt = dt.to_pydatetime()
    else:
        if isinstance(dt[0], pd.DatetimeIndex):
            dt = dt.to_pydatetime()
    
    # check if dt is single variable or list
    dt = np.array(dt)
    is_dt_single = dt.size == 1
    is_dt_two = dt.size == 2
    is_dt_multiple = dt.size > 2
    
    # check date_range
    if is_dt_single or is_dt_multiple:
        date_range = False
    if date_range and dt[0] >= dt[1]:
        raise UserWarning('dt[0] should be earlier than dt[1] if date_range is True')
    
    # split into the patterns of filename and directory
    pattern_fn = pattern.split('/')[-1]
    pattern_dir = '/'.join(pattern.split('/')[:-1])
    
    if is_dt_single:
        # the easiest case
        fname = [f'{indir}/{dt:{pattern}}']
    else:
        if not date_range:
            # the second easiest case
            fname = [f'{indir}/{_dt:{pattern}}' for _dt in dt]
        else:
            # check if only the last part of fmt contains format string
            wh_fmt = ['%' in x for x in pattern.split('/')]
            cnt_fmtstr = np.sum(wh_fmt)
            is_dir_clean = cnt_fmtstr == 1
            
            if is_dir_clean:
                # if directory doesn't contain the format string
                # find all files in the directory
                candidate_fn = np.array(glob.glob(f'{indir}/{_pattern_as_asterisk(pattern)}'))
                
                # filename to datetime
                candidate_dt = _fname2dt(candidate_fn, f'{indir}/{pattern}')
            else:
                # if directory contain the format string
                # find candidate paths
                dtlist = pd.date_range(start=dt[0], end=dt[1], freq='1H')
                candidate_paths = np.unique([t.strftime(f'{indir}{pattern_dir}') for t in dtlist])
                
                # 
                candidate_fn = []
                for candidate_path in candidate_paths:
                    candidate_fn.append(glob.glob(f'{candidate_path}/{_pattern_as_asterisk(pattern_fn)}'))
                
                # flatten list (the fastest way)
                candidate_fn = np.array(sum(candidate_fn, []))
                
                # filename to datetime
                candidate_dt = _fname2dt(candidate_fn, f'{indir}/{pattern}')
            
            # check if any file found
            if len(candidate_dt) == 0:
                raise UserWarning('No matched file found')
            
            # match with the datetime range
            lowest_dtfmt = _get_lowest_order_datetimeformat(pattern_fn)
            one_order_lower = {
                '%Y': '%m',
                '%m': '%d',
                '%d': '%H%M%S',
                '%j': '%H%M%S',
                '%H': '%M%S',
                '%M': '%S',
                '%S': '%f'
            }
            default_dtvalue = {
                '%m': 1,
                '%d': 1,
                '%H%M%S': 0,
                '%M%S': 0,
                '%S': 0,
                '%f': 0
            }
            lower_dtfmt = one_order_lower[lowest_dtfmt]
            candidate_dt = np.array(candidate_dt)
            if int(datetime.datetime.strftime(dt[0], lower_dtfmt)) != default_dtvalue[lower_dtfmt]:
                # Truncate unnecessary start time (eg. filename: 20180228, start_datetime: 20180228 13:00 --> 20180228 00:00)
                dt0_trunc = _truncate_unnecessary_datetime(dt[0], lower_dtfmt)
                wh = np.where(np.logical_and(candidate_dt >= dt0_trunc, candidate_dt <= dt[1]))[0]
            else:
                wh = np.where(np.logical_and(candidate_dt >= dt[0], candidate_dt <= dt[1]))[0]
            
            # check if any file found
            if wh.size == 0:
                raise UserWarning('No matched file found')
            
            # store matched files only
            fname = candidate_fn[wh]

        # get rid of duplicated files
        fname = np.unique(fname)
    
    # check if file exists
    fname_exist = []
    for _fname in fname:
        if os.path.isfile(_fname):
            fname_exist.append(_fname)
        else:
            if verbose:
                warnings.warn(f'File does not exist: {_fname}')
    fname = np.array(fname_exist)
    
    # prepare a return
    if fname.size == 0:
        raise UserWarning('No matched file found')
    fdt = _fname2dt(fname, f'{indir}/{pattern}')
    if is_dt_single:
        fname = fname[0]
    
    return fname, fdt

def _fname2dt(fnames, pattern):
    """
    Get datetime from the filename.
    """
    import re
    import parse
    
    fnames = np.array(fnames)
    
    dt = []
    clean_pattern = re.sub('//', '/', f'{pattern}')
    dtfmt = re.findall('(\\%\D)', clean_pattern)

    # for duplicated datetime format
    is_duplicate = len(set(dtfmt)) != len(dtfmt)
    replace = {
        '(%Y)': '{Y:04d}',
        '(%y)': '{y:02d}',
        '(%m)': '{m:02d}',
        '(%d)': '{d:02d}',
        '(%j)': '{j:03d}',
        '(%H)': '{H:02d}',
        '(%M)': '{M:02d}',
        '(%S)': '{S:02d}',
    }
    default = {
        'Y': 2020,
        'm': 1,
        'd': 1,
        'H': 0,
        'M': 0,
        'S': 0,
    }
    
    for fname in fnames:
        clean_fname = re.sub('//', '/', f'{fname}')
        
        if not is_duplicate:
            _dt = datetime.datetime.strptime(clean_fname, clean_pattern)
        else:
            parse_pattern = clean_pattern
            for key in replace.keys():
                parse_pattern = re.sub(key, replace[key], parse_pattern)
            parsed = parse.parse(parse_pattern, clean_fname)
            if 'j' not in parsed.named.keys():
                # set default value if no datetime key is found
                for key in default.keys():
                    if key not in parsed.named.keys():
                        parsed.named[key] = default[key]
                _dt = datetime.datetime(parsed['Y'], parsed['m'], parsed['d'], parsed['H'], parsed['M'], parsed['S'])
            else:
                # set default value if no datetime key is found
                for key in ['Y', 'H', 'M', 'S']:
                    if key not in parsed.named.keys():
                        parsed.named[key] = default[key]
                _dt = datetime.datetime(parsed['Y'], 1, 1, parsed['H'], parsed['M'], parsed['S']) + datetime.timedelta(parsed['j'] - 1)
        dt.append(_dt)

    dt = np.array(dt)
    if fnames.size == 1:
        dt = dt[0]

    return dt

def _pattern_as_asterisk(pattern):
    """
    Replace recurring %D (D: character) in the datetime pattern to asterisk.
    """
    import re
    
    return re.sub("(\\%\D)+", "*", pattern)

def _get_lowest_order_datetimeformat(pattern):
    """
    Get the lowest order of datetime format in the pattern.
    """
    import re
    
    dtfmts = re.findall('(\\%\D)', pattern)
    fmt2new = {
        '%Y': '%Y', # year
        '%y': '%Y',
        '%G': '%Y',
        '%m': '%m', # month
        '%B': '%m',
        '%b': '%m',
        '%d': '%d', # day
        '%j': '%j', # day of the year
        '%H': '%H', # hour
        '%I': '%H',
        '%M': '%M', # minute
        '%S': '%S', # second
    }
    for _fmt in dtfmts:
        lowest_fmt = fmt2new[_fmt]
    
    return lowest_fmt

def _truncate_unnecessary_datetime(dt, highest_fmt_unnecessary):
    """
    Truncate unnecessary datetime.
    """
    if '%m' in highest_fmt_unnecessary:
        dt_trunc = dt.replace(month=0, day=0, hour=0, minute=0, second=0, microsecond=0)
    elif '%d' in highest_fmt_unnecessary:
        dt_trunc = dt.replace(day=0, hour=0, minute=0, second=0, microsecond=0)
    elif '%H%M%S' in highest_fmt_unnecessary:
        dt_trunc = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif '%M%S' in highest_fmt_unnecessary:
        dt_trunc = dt.replace(minute=0, second=0, microsecond=0)
    elif '%S' in highest_fmt_unnecessary:
        dt_trunc = dt.replace(second=0, microsecond=0)
    elif '%f' in highest_fmt_unnecessary:
        dt_trunc = dt.replace(microsecond=0)
    else:
        raise UserWarning('Something wrong!! Please report an issue to GitHub with error log')
    
    return dt_trunc

def _get_proj_from_KNUwissdom(ds):
    import cartopy.crs as ccrs
    
    lcc = ds['lcc_projection']
    clon = lcc.longitude_of_central_meridian
    clat = lcc.latitude_of_projection_origin
    false_easting = lcc.false_easting
    false_northing = lcc.false_northing
    standard_parallel = lcc.standard_parallel
    
    proj = ccrs.LambertConformal(
        central_longitude=clon,
        central_latitude=clat,
        standard_parallels=standard_parallel,
        false_easting=false_easting,
        false_northing=false_northing,
        globe=ccrs.Globe(
            ellipse=None,
            semimajor_axis=6371008.77,
            semiminor_axis=6371008.77)
    )
    
    return proj

def _get_proj_from_KMAwissdom():
    import cartopy.crs as ccrs

    proj = ccrs.LambertConformal(
        central_longitude=126.0,
        central_latitude=38.0,
        standard_parallels=(30,60),
        false_easting=440000,
        false_northing=700000,
        globe=ccrs.Globe(
        ellipse=None,
        semimajor_axis=6371008.77,
        semiminor_axis=6371008.77)
    )
    
    return proj

def _read_wissdom_KNU1(fnames, degree='essential'):
    import xarray as xr
    ds = xr.open_mfdataset(fnames, concat_dim='NT', combine='nested')
        
    ds['proj'] = _get_proj_from_KNUwissdom(ds)
    ds = ds.rename(
        {'uu2':'u',
         'vv2':'v',
         'ww2':'w'
        }
    )
    if degree in ['extensive', 'debug']:
        ds = ds.rename(
            {'rvort2':'vor',
             'rdivg2':'div'
            }
        )
    
    if degree in ['essential', 'extensive']:
        ds.attrs={}
    
    list_vars = []
    for x in ds.variables.__iter__():
        list_vars.append(x)
    
    if degree in ['essential']:
        for var in ['u', 'v', 'w', 'x', 'y', 'lev', 'proj']:
            list_vars.remove(var)
        ds = ds.drop(list_vars)
    
    if degree in ['extensive']:
        for var in ['u', 'v', 'w', 'vor', 'div', 'x', 'y', 'lev', 'proj']:
            list_vars.remove(var)
        ds = ds.drop(list_vars)
    
    ds = ds.rename_dims({'X':'nx', 'Y':'ny', 'lev':'nz'})
    
    return ds

def _read_wissdom_KMAnc(fnames, degree='essential'):
    import xarray as xr
    ds = xr.open_mfdataset(fnames, concat_dim='NT', combine='nested')

    ds['proj'] = _get_proj_from_KMAwissdom()
    dataminus = ds.data_minus
    datascale = ds.data_scale
    dataout = ds.data_out

    ds['x'] = ds['nx'].values * ds.grid_size
    ds['y'] = ds['ny'].values * ds.grid_size
    ds['height'] = ds['height'][0,:]
    ds = ds.set_coords(("height")) # variable to coord

    ds = ds.rename(
        {'u_component':'u',
         'v_component':'v',
         'w_component':'w',
         'height':'lev'
        }
    )
    for f in ['u', 'v', 'w']:
        ds[f] = xr.where(
            ds[f] == dataout,
            np.nan,
            ds[f]
        )
        ds[f] = (ds[f]-dataminus)/datascale

    if degree in ['extensive', 'debug']:
        ds = ds.rename(
            {'vertical_vorticity':'vor',
             'divergence':'div'
            }
        )
        for f in ['div', 'vor']:
            ds[f] = xr.where(
                ds[f] == dataout,
                np.nan,
                ds[f]
            )
            ds[f] = xr.where(
                ds[f] <= 0,
                10**((ds[f]-dataminus)/datascale),
                -10**(-(ds[f]-dataminus)/datascale)
            )
    if degree in ['debug']:
        ds = ds.rename(
            {'vertical_velocity':'vt',
             'reflectivity':'z'
            }
        )
        for f in ['vt', 'z']:
            ds[f] = xr.where(
                ds[f] == dataout,
                np.nan,
                ds[f]
            )
            ds[f] = (ds[f]-dataminus)/datascale
    
    if degree in ['essential']:
        ds = ds.drop_vars([
            'vertical_vorticity',
            'divergence',
            'vertical_velocity',
            'reflectivity'
        ])
    if degree in ['extensive']:
        ds = ds.drop_vars([
            'vertical_velocity',
            'reflectivity'
        ])
    
    if degree in ['essential', 'extensive']:
        ds.attrs={}
        
    ds = ds.transpose('NT', 'ny', 'nx', 'nz', 'x', 'y')
    
    
    ds['x'] = ds['x'].rename({'x':'nx'})
    ds['y'] = ds['y'].rename({'y':'ny'})
    ds = ds.drop(['nx', 'ny'])
    
    return ds

def _read_wissdom_KMAbin(fname, degree='essential'):
    import xarray as xr
    import gzip
    from numba import njit
    
    @njit(fastmath=True)
    def fastpow(value):
        return 10.0**value

    @njit(fastmath=True)
    def invert_scaling1(arr, data_minus, data_scale):
        res = np.empty(arr.shape)
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                for z in range(arr.shape[2]):
                    res[x,y,z] = (arr[x,y,z]-data_minus)/data_scale
        return res

    import math
    @njit(fastmath=True)
    def invert_scaling2(arr, data_minus, data_scale):
        res = np.empty(arr.shape)
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                for z in range(arr.shape[2]):
                    if arr[x,y,z] <= 0:
                        res[x,y,z] = fastpow((arr[x,y,z]-data_minus)/data_scale)
                    else:
                        res[x,y,z] = -fastpow(-(arr[x,y,z]-data_minus)/data_scale)
        return res

    def bin2str(binary):
        return [ord(c) for c in binary.decode('latin-1')] ######## why not ascii ?????????

    def timestr2dt(file):
        yy = np.frombuffer(file.read(2), dtype=np.int16)[0]
        mm = ord(file.read(1))
        dd = ord(file.read(1))
        hh = ord(file.read(1))
        mi = ord(file.read(1))
        ss = ord(file.read(1))
        try:
            return datetime.datetime(yy,mm,dd,hh,mi,ss)
        except:
            return -1

    with gzip.open(fname,'rb') as f:
        version    = ord(f.read(1)) # char
        ptype      = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        tm         = timestr2dt(f) # struct
        tm_in      = timestr2dt(f) # struct
        num_stn    = ord(f.read(1)) # char
        map_code   = ord(f.read(1)) # char
        map_etc    = ord(f.read(1)) # char
        nx         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        ny         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        nz         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        dxy        = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        dz         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        z_min      = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        num_data   = ord(f.read(1)) # char
        dz2        = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        z_min2     = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_out   = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_in    = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_min   = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_minus = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_scale = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_unit  = ord(f.read(1)) # char
        etc        = np.frombuffer(f.read(16), dtype=np.int16) # short

        u = np.frombuffer(f.read(2*nx*ny*nz), dtype=np.int16).copy().reshape(nz,ny,nx)
        v = np.frombuffer(f.read(2*nx*ny*nz), dtype=np.int16).copy().reshape(nz,ny,nx)
        w = np.frombuffer(f.read(2*nx*ny*nz), dtype=np.int16).copy().reshape(nz,ny,nx)
        if degree in ['extensive','debug']:
            div = np.frombuffer(f.read(2*nx*ny*nz), dtype=np.int16).copy().reshape(nz,ny,nx)
            vor = np.frombuffer(f.read(2*nx*ny*nz), dtype=np.int16).copy().reshape(nz,ny,nx)
        if degree in ['debug']:
            dbz = np.frombuffer(f.read(2*nx*ny*nz), dtype=np.int16).copy().reshape(nz,ny,nx)
            vt = np.frombuffer(f.read(2*nx*ny*nz), dtype=np.int16).copy().reshape(nz,ny,nx)

    mask_u = u == data_out
    mask_v = v == data_out
    mask_w = w == data_out
    if degree in ['extensive','debug']:
        mask_div = div == data_out
        mask_vor = vor == data_out
    if degree in ['debug']:
        mask_dbz = dbz == data_out
        mask_vt = vt == data_out

    u = invert_scaling1(u, data_minus, data_scale)
    v = invert_scaling1(v, data_minus, data_scale)
    w = invert_scaling1(w, data_minus, data_scale)
    if degree in ['extensive','debug']:
        div = invert_scaling2(div, data_minus, data_scale)
        vor = invert_scaling2(vor, data_minus, data_scale)
    if degree in ['debug']:
        dbz = invert_scaling1(dbz, data_minus, data_scale)
        vt = invert_scaling1(vt, data_minus, data_scale)

    u[mask_u] = np.nan
    v[mask_v] = np.nan
    w[mask_w] = np.nan
    if degree in ['extensive','debug']:
        div[mask_div] = np.nan
        vor[mask_vor] = np.nan
    if degree in ['debug']:
        dbz[mask_dbz] = np.nan
        vt[mask_vt] = np.nan
    
    lev = np.arange(nz)*dz
    x = np.arange(nx)*dxy
    y = np.arange(ny)*dxy
    
    ds = xr.Dataset(
        {
            'u': (["NT","ny","nx","nz"], np.expand_dims(np.swapaxes(np.swapaxes(u,0,2),0,1),0)),
            'v': (["NT","ny","nx","nz"], np.expand_dims(np.swapaxes(np.swapaxes(v,0,2),0,1),0)),
            'w': (["NT","ny","nx","nz"], np.expand_dims(np.swapaxes(np.swapaxes(w,0,2),0,1),0)),
        },
        coords={
            'lev': (["nz"],
                (lev)),
            'x': (["nx"],
                (x)),
            'y': (["ny"],
                (y)),
        }
    )
    if degree in ['extensive', 'debug']:
        ds['div'] = (["NT","ny","nx","nz"], np.expand_dims(np.swapaxes(np.swapaxes(div,0,2),0,1),0))
        ds['vor'] = (["NT","ny","nx","nz"], np.expand_dims(np.swapaxes(np.swapaxes(vor,0,2),0,1),0))
    if degree in ['debug']:
        pass
    
    ds['proj'] = _get_proj_from_KMAwissdom()
    
    return ds

def read_wissdom(fnames, kind='KNUv2', degree='essential'):
    """
    Read WISSDOM wind field.
    
    Examples
    ---------
    >>> ds_wissdom = kkpy.io.read_wissdom('WISSDOM_VAR_201802280600.nc')
    
    >>> ds_wissdom = kkpy.io.read_wissdom('RDR_R3D_KMA_WD_201802280600.bin.gz', kind='KMAbin')
    
    >>> ds_wissdom = kkpy.io.read_wissdom('RDR_R3D_KMA_WD_201802280600.nc', kind='KMAnc', degree='extensive')
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of WISSDOM to read.
    kind : str, optional
        Data format. Possible options are 'KNUv2', 'KMAnc', and 'KMAbin'. Default is 'KNUv2'.
    degree : str, optional
        Degree of variable type to read. Possible options are 'essential', 'extensive', and 'debug'. Default is 'essential'.
        'essential' includes u, v, and w, while 'extensive' further includes divergence and vorticity. 'debug' returns all available variables.
        Note that the time efficiency for 'extensive' is low when kind='KMAbin'.
    
    Returns
    ---------
    ds : xarray dataset object
        Return WISSDOM wind field.
    """
    import xarray as xr
    
    if kind in ['KNUv2']:
        ds = _read_wissdom_KNU1(fnames, degree=degree)
        
    elif kind in ['KMAnc']:
        ds = _read_wissdom_KMAnc(fnames, degree=degree)
        
    elif kind in ['KMAbin']:
        if isinstance(fnames, (list,np.ndarray)):
            dslist = []
            for fname in fnames:
                dslist.append(_read_wissdom_KMAbin(fname, degree=degree))
            ds = xr.combine_nested(dslist, 'NT')
        else:
            ds = _read_wissdom_KMAbin(fnames, degree=degree)
    
    else:
        raise UserWarning(f'Not supported: kind={kind}')
    
    return ds

def read_vertix(fnames):
    """
    Read VertiX (nc).
    
    Examples
    ---------
    >>> ds_vtx = kkpy.io.read_vertix(fnames)
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of VertiX to read.
    
    Returns
    ---------
    ds : xarray dataset object
        Return VertiX dataset.
    """
    from cftime import num2date
    import xarray as xr
    
    # read data
    ds = xr.open_mfdataset(
        fnames,
        decode_times=False,
        combine='nested'
    )
    
    # calculate datetime
    for time_var in ['time_dwell', 'time_spec']:
        ds[time_var] = xr.DataArray(
            num2date(
                ds[time_var].data,
                'seconds since 1970-01-01 00:00:00.0',
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True
            ),
            dims=[time_var]
        )
    
    # variable to coordinate
    ds = ds.set_coords('range')
    ds = ds.set_coords('spec_vms')
    
    # replace nodata to NaN
    for var in list(ds.keys()):
        ds[var] = ds[var].where(ds[var] > ds.nodata)
        
    return ds

def read_vet(fnames):
    """
    Read WRC VET motion vector.
    
    Examples
    ---------
    >>> ds_vet = kkpy.io.read_vet(fnames)
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of VET motion vector to read.
    
    Returns
    ---------
    ds : xarray dataset object
        Return WRC VET motion vector dataset.
    """
    import xarray as xr
    
    dss = []
    
    if isinstance(fnames, (list, np.ndarray)):
        for i_f, fname in enumerate(fnames):
            _ds = _read_vet(fname)
            dss.append(_ds)
        ds = xr.concat(dss, dim='t')
    else:
        ds = _read_vet(fnames)
    
    return ds
    

def _read_vet(fname):
    import cartopy.crs as ccrs
    
    nx_vet = 25
    ny_vet = 25
    dxy_vet = 32000 # meter
    border_vet = 112
    clon_vet = 126.0
    clat_vet = 38.0
    fe_vet = 440000
    fn_vet = 770000

    proj_vet = ccrs.LambertConformal(
        central_longitude=clon_vet,
        central_latitude=clat_vet,
        false_easting=fe_vet,
        false_northing=fn_vet,
        standard_parallels=(30,60)
    )
    
    # READ DATA
    dt = np.dtype([
        ('t','<6i2'),
        ('pixel_res_km','<f4'),
        ('border_pix','<i4'),
        ('ew_arsize','<i4'),
        ('sn_arsize','<i4'),
        ('velf_ew_size','<i4'),
        ('velf_sn_size','<i4'),
        ('uv', '(25,25,2)f4')]
    )
    data = np.fromfile(fname, dtype=dt)
    
    # Resolution of vector data, assuming that N-S and E-W grids are same
    cnt_vec = data['velf_ew_size'][0]
    cnt_border = data['border_pix'][0]
    cnt_pixel = data['ew_arsize'][0]
    res_vec = (cnt_pixel - 2*cnt_border)/cnt_vec
    
    # DEFINE DATASET
    ds = xr.Dataset(
        {
            'u': (["t","y","x"], data['uv'][0,:,:,0][None,:,:]),
            'v': (["t","y","x"], data['uv'][0,:,:,1][None,:,:]),
        },
        coords={
            'x': (["x"],
                (np.arange(cnt_vec)*res_vec + res_vec/2. + cnt_border)*1e3),
            'y': (["y"],
                (np.arange(cnt_vec)*res_vec + res_vec/2. + cnt_border)*1e3),
            't': (["t"], [pd.Timestamp(os.path.basename(fname)[-16:-4])]),
        },
        attrs={
            'title': 'Output of variational echo tracking by KMA',
            'time_unit': 'KST',
            'vector_resolution': f'{res_vec} km',
            'crs': proj_vet.proj4_init,
            'projection': proj_vet,
        }
    )
    
    return ds

def read_hsr(fnames):
    """
    Read WRC HSR reflectivity.
    
    Examples
    ---------
    >>> ds_hsr = kkpy.io.read_hsr(fnames)
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of HSR reflectivity to read.
    
    Returns
    ---------
    ds : xarray dataset object
        Return WRC HSR reflectivity dataset.
    """
    import xarray as xr
    
    dss = []
    
    if isinstance(fnames, (list, np.ndarray)):
        for i_f, fname in enumerate(fnames):
            _ds = _read_hsr(fname)
            dss.append(_ds)
        ds = xr.concat(dss, dim='t')
    else:
        ds = _read_hsr(fnames)
    
    return ds

def _read_hsr(file):
    import gzip
    import cartopy.crs as ccrs
    
    def bin2str(binary):
        return [ord(c) for c in binary.decode('ascii')]

    def timestr2dt(file):
        yy = np.frombuffer(file.read(2), dtype=np.int16)[0]
        mm = ord(file.read(1))
        dd = ord(file.read(1))
        hh = ord(file.read(1))
        mi = ord(file.read(1))
        ss = ord(file.read(1))
        try:
            return datetime.datetime(yy,mm,dd,hh,mi,ss)
        except:
            return -1

    with gzip.open(file,'rb') as f:
        version    = ord(f.read(1)) # char
        ptype      = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        tm         = timestr2dt(f) # struct
        tm_in      = timestr2dt(f) # struct
        num_stn    = ord(f.read(1)) # char
        map_code   = ord(f.read(1)) # char
        map_etc    = ord(f.read(1)) # char
        nx         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        ny         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        nz         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        dxy        = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        dz         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        z_min      = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        num_data   = ord(f.read(1)) # char
        data_code  = bin2str(f.read(16)) # char
        etc        = bin2str(f.read(15)) # char
        
        for ii in range(48):
            stn_cd = bin2str(f.read(6)) # char
            _tm    = timestr2dt(f) # struct
            _tm_in = timestr2dt(f) # struct
        
        refl       = np.frombuffer(f.read(2*nx*ny), dtype=np.int16)
        
    refl = refl.copy().reshape(ny,nx) / 100
    refl[refl == -300] = -9999 # outside radar observable range
    refl[refl == -250] = -9998 # no echo
    refl[(refl < -100) & (refl > -9000)] = -9997 # null
    
    # hsr = {}
    # hsr['reflectivity'] = refl
    # hsr['nx'] = nx
    # hsr['ny'] = ny
    # hsr['dxy'] = dxy # meter
    # hsr['clon'] = clon_hsr # degN
    # hsr['clat'] = clat_hsr # degE
    # hsr['false_easting'] = fe_hsr # meter
    # hsr['false_northing'] = fn_hsr # meter
    # hsr['cnt_radar'] = num_stn
    # hsr['time'] = tm # datetime object
    
    clon_hsr = 126.0
    clat_hsr = 38.0
    fe_hsr = 560500 # meter
    fn_hsr = 840500 # meter
    proj_hsr = ccrs.LambertConformal(
        central_longitude=clon_hsr,
        central_latitude=clat_hsr,
        false_easting=fe_hsr,
        false_northing=fn_hsr,
        standard_parallels=(30,60)
    )
    
    # DEFINE DATASET
    ds = xr.Dataset(
        {
            'reflectivity': (["t","y","x"], refl[None,:,:]),
        },
        coords={
            'x': (["x"],
                np.arange(0,np.int64(nx)*dxy, dxy)),
            'y': (["y"],
                np.arange(0,np.int64(ny)*dxy, dxy)),
            't': (["t"], [pd.Timestamp(os.path.basename(file)[-19:-7])]),
        },
        attrs={
            'title': 'HSR product generated by KMA',
            'time_unit': 'KST',
            'cnt_radar': num_stn,
            'outside_radar_observable_range': -9999,
            'no_echo': -9998,
            'null': -9997,
            'crs': proj_hsr.proj4_init,
            'projection': proj_hsr,
        }
    )
    
    return ds

def read_d3d(fnames):
    """
    Read WRC D3D product.
    
    Examples
    ---------
    >>> ds_d3d = kkpy.io.read_d3d(fnames)
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of D3D product to read. Variables in all files must be of the same type.
    
    Returns
    ---------
    ds : xarray dataset object
        Return WRC D3D product dataset.
    """
    import xarray as xr
    
    dss = []
    
    if isinstance(fnames, (list, np.ndarray)):
        for i_f, fname in enumerate(fnames):
            _ds = _read_d3d(fname)
            dss.append(_ds)
        ds = xr.concat(dss, dim='t')
    else:
        ds = _read_d3d(fnames)
    
    return ds

def _read_d3d(fname):
    import gzip
    import cartopy.crs as ccrs
    
    def bin2str(binary):
        return [ord(c) for c in binary.decode('ascii')]

    def timestr2dt(file):
        yy = np.frombuffer(file.read(2), dtype=np.int16)[0]
        mm = ord(file.read(1))
        dd = ord(file.read(1))
        hh = ord(file.read(1))
        mi = ord(file.read(1))
        ss = ord(file.read(1))
        try:
            return datetime.datetime(yy,mm,dd,hh,mi,ss)
        except:
            return -1

    if 'ta' in os.path.basename(fname):
        varname = 'air_temperature'
    elif 'td' in os.path.basename(fname):
        varname = 'dew_point_temperature'
    elif 'pa':
        varname = 'air_pressure'
    else:
        raise UserWarning('Variable type is not recognized')
        return -1
    
    with gzip.open(fname,'rb') as f:
        version    = ord(f.read(1)) # char
        ptype      = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        tm         = timestr2dt(f) # struct
        tm_in      = timestr2dt(f) # struct
        num_stn    = ord(f.read(1)) # char
        map_code   = ord(f.read(1)) # char
        map_etc    = ord(f.read(1)) # char
        nx         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        ny         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        nz         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        dxy        = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        dz         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        z_min      = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        num_data   = ord(f.read(1)) # char
        dz2        = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        z_min2     = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_out   = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_in    = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_min   = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_minus = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_scale = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_unit  = ord(f.read(1)) # char
        etc        = np.frombuffer(f.read(16), dtype=np.int16) # short

        var3d      = (np.frombuffer(f.read(2*nz*ny*nx), dtype=np.int16)-data_minus)/data_scale
        var3d      = var3d.reshape(nz,ny,nx)

    proj = ccrs.LambertConformal(
        central_longitude=126.0,
        central_latitude=38.0,
        false_easting=444000,
        false_northing=772000,
        standard_parallels=(30,60)
    )
    
    # DEFINE DATASET
    ds = xr.Dataset(
        {
            varname: (["t","z","y","x"], var3d[None,:,:]),
        },
        coords={
            'x': (["x"],
                np.arange(0,np.int64(nx)*dxy, dxy)),
            'y': (["y"],
                np.arange(0,np.int64(ny)*dxy, dxy)),
            'z': (["z"],
                np.append(np.arange(0,2000,100), np.arange(2000,10001,200))),
            't': (["t"], [pd.Timestamp(os.path.basename(fname)[-19:-7])]),
        },
        attrs={
            'title': 'D3D product generated by KMA',
            'time_unit': 'KST',
            'crs': proj.proj4_init,
            'projection': proj,
        }
    )
    
    return ds

def read_r3d(fnames, kind='nc'):
    """
    Read WRC R3D product.

    Examples
    ---------
    >>> ds_r3d = kkpy.io.read_r3d(fnames)

    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of R3D product to read. Variables in all files must be of the same type.
    kind : str, optional
        Data format. Possible options are 'nc' and 'bin'. Default is 'nc'.

    ---------
    >>> ds_r3d = kkpy.io.read_r3d('RDR_R3D_EXT_KD_202301011740.nc')
    
    >>> ds_r3d = kkpy.io.read_r3d('RDR_R3D_EXT_HCI_202007010950.bin.gz', kind='bin')
    
    >>> ds_r3d = kkpy.io.read_r3d(['RDR_R3D_EXT_CZ_202007010740.bin.gz','RDR_R3D_EXT_CZ_202007010745.bin.gz'], kind='bin')
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of WISSDOM to read.
        
    Returns
    ---------
    ds : xarray dataset object
        Return WRC R3D product dataset.
    """
    import xarray as xr

    if kind in ['nc']:
        ds = _read_r3d_nc(fnames)
        
    elif kind in ['bin']:
        if isinstance(fnames, (list,np.ndarray)):
            dss = []
            for fname in fnames:
                dss.append(_read_r3d_bin(fname))
            ds = xr.combine_nested(dss, 't')
        else:
            ds = _read_r3d_bin(fnames)
    
    else:
        raise UserWarning(f'Not supported: kind={kind}')
    
    return ds

def _read_r3d_nc(fname):
    import xarray as xr
    import cartopy.crs as ccrs
    
    ds = xr.open_mfdataset(fname, concat_dim='t', combine='nested')
    
    fnames = None
    if isinstance(fname, (list,np.ndarray)):
        fnames = fname
        fname = fnames[0]
    
    if '_CZ' in os.path.basename(fname):
        varname = 'reflectivity'
    elif '_DR' in os.path.basename(fname):
        varname = 'differential_reflectivity'
    elif '_KD' in os.path.basename(fname):
        varname = 'specific_differential_phase'
    elif '_RH' in os.path.basename(fname):
        varname = 'CopolarCorrelation'
    elif '_HCI' in os.path.basename(fname):
        varname = 'radar_echo_classification'
    else:
        raise UserWarning('Variable type is not recognized')
        return -1
    
    proj = ccrs.LambertConformal(
        central_longitude=126.0,
        central_latitude=38.0,
        false_easting=440500,
        false_northing=770500,
        standard_parallels=(30,60)
    )
    dataminus = ds.data.data_minus
    datascale = ds.data.data_scale
    dataout = ds.data.data_out

    # ds['x'] = ds.grid_nx * ds.grid_size
    # ds['y'] = ds.grid_ny * ds.grid_size
    # ds = ds.set_coords(("height")) # variable to coord

    ds = ds.rename(
        {'data':varname}
    )
    
    ds[varname] = xr.where(
        ds[varname] == dataout,
        np.nan,
        ds[varname]
    )
    ds[varname] = (ds[varname]-dataminus)/datascale
    
    if fnames is not None:
        times = [pd.Timestamp(os.path.basename(fn)[-15:-3]) for fn in fnames]
    else:
        times = [pd.Timestamp(os.path.basename(fname)[-15:-3])]
    
    ds = ds.rename_dims({'nx':'x','ny':'y','nz':'z'})
    ds = ds.assign_coords({
        'x': (["x"],
            np.arange(0, np.int64(ds.grid_nx)*ds.grid_size, ds.grid_size)),
        'y': (["y"],
            np.arange(0, np.int64(ds.grid_ny)*ds.grid_size, ds.grid_size)),
        'z': (["z"],
            ds.height.values[0,:]),
        't': (["t"], times),
    })
    ds = ds.drop_vars('height')
    
    ds.attrs = {
        'title': 'R3D product generated by KMA',
        'time_unit': 'KST',
        'crs': proj.proj4_init,
        'projection': proj,
    }
            
    return ds

def _read_r3d_bin(fname):
    import cartopy.crs as ccrs
    import gzip
    
    def bin2str(binary):
        return [ord(c) for c in binary.decode('ascii')]

    def timestr2dt(fname):
        yy = np.frombuffer(fname.read(2), dtype=np.int16)[0]
        mm = ord(fname.read(1))
        dd = ord(fname.read(1))
        hh = ord(fname.read(1))
        mi = ord(fname.read(1))
        ss = ord(fname.read(1))
        try:
            return datetime.datetime(yy,mm,dd,hh,mi,ss)
        except:
            return -1
    
    if '_CZ' in os.path.basename(fname):
        varname = 'reflectivity'
    elif '_DR' in os.path.basename(fname):
        varname = 'differential_reflectivity'
    elif '_KD' in os.path.basename(fname):
        varname = 'specific_differential_phase'
    elif '_RH' in os.path.basename(fname):
        varname = 'CopolarCorrelation'
    elif '_HCI' in os.path.basename(fname):
        varname = 'radar_echo_classification'
    else:
        raise UserWarning('Variable type is not recognized')
        return -1

    with gzip.open(fname,'rb') as f:
        version    = ord(f.read(1)) # char
        ptype      = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        tm         = timestr2dt(f) # struct
        tm_in      = timestr2dt(f) # struct
        num_stn    = ord(f.read(1)) # char
        map_code   = ord(f.read(1)) # char
        map_etc    = ord(f.read(1)) # char
        nx         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        ny         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        nz         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        dxy        = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        dz         = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        z_min      = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        num_data   = ord(f.read(1)) # char
        dz2        = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        z_min2     = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_out   = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_in    = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_min   = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_minus = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_scale = np.frombuffer(f.read(2), dtype=np.int16)[0] # short
        data_unit  = ord(f.read(1)) # char
        etc        = np.frombuffer(f.read(16), dtype=np.int16) # short
        
        for ii in range(48):
            stn_cd = bin2str(f.read(6)) # char
            _tm    = timestr2dt(f) # struct
            _tm_in = timestr2dt(f) # struct
        
        data       = np.frombuffer(f.read(2*nx*ny*nz), dtype=np.int16)

    r3d_1d_scaled   = (data & 0x07FF)
    mask_outside_1d = (r3d_1d_scaled == data_out)
    mask_qced_1d    = (r3d_1d_scaled == data_in)
    r3d_1d = (r3d_1d_scaled - data_minus)/data_scale

    _r3d = r3d_1d.reshape(nz,ny,nx)
    mask_outside      = mask_outside_1d.reshape(nz,ny,nx)
    mask_qced         = mask_qced_1d.reshape(nz,ny,nx)
    _r3d[mask_outside] = -9999
    _r3d[mask_qced]    = -9998

    proj = ccrs.LambertConformal(
        central_longitude=126.0,
        central_latitude=38.0,
        false_easting=440500,
        false_northing=770500,
        standard_parallels=(30,60)
    )

    # DEFINE DATASET
    ds = xr.Dataset(
        {
            varname: (["t","z","y","x"], _r3d[None,:,:]),
        },
        coords={
            'x': (["x"],
                np.arange(0,np.int64(nx)*dxy, dxy)),
            'y': (["y"],
                np.arange(0,np.int64(ny)*dxy, dxy)),
            'z': (["z"],
                np.arange(0,np.int64(nz)*dz, dz)),
            't': (["t"], [pd.Timestamp(os.path.basename(fname)[-19:-7])]),
        },
        attrs={
            'title': 'R3D product generated by KMA',
            'time_unit': 'KST',
            'crs': proj.proj4_init,
            'projection': proj,
        }
    )

    return ds

def read_sounding(fnames):
    """
    Read sounding.
    
    Examples
    ---------
    >>> df = kkpy.io.read_sounding(fnames)
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of sounding to read.
    
    Returns
    ---------
    df : pandas dataframe object
        Return sounding dataframe.
    
    Notes
    ---------
    Columns:
        - 'P': pressure [hPa]
        - 'T': temperature [degC]
        - 'RH': relative humidity [%]
        - 'WS': wind speed [m s-1]
        - 'WD': wind direction [deg]
        - 'Lon': longitude [oE]
        - 'Lat': latitude [oN]
        - 'Alt': altitude from mean sea level [m]
        - 'Geo': geopotential height [gpm]
        - 'Dew': dew-point temperature [degC]
        - 'U': u-wind [m s-1]
        - 'V': v-wind [m s-1]
        - 'Uknot': u-wind [knot]
        - 'Vknot': v-wind [knot]
        - 'time': launch time [file-dependent, generally UTC]
    """
    import pandas as pd
    
    dfs = []
    
    if isinstance(fnames, (list, np.ndarray)):
        for i_f, fname in enumerate(fnames):
            _df = _read_sounding(fname)
            dfs.append(_df)
        df = pd.concat(dfs).reset_index(drop=True)
    else:
        df = _read_sounding(fnames)
    
    return df

def _read_sounding(filename):
    from . import util

    def read_sounding_v1(filename):
        df = pd.read_csv(filename,
                         delim_whitespace=True,
                         skiprows=10,
                         on_bad_lines='skip',
                         encoding='euc-kr'
                        )
        df = df.drop(['Time(min:sec)', 'Asc(m/m)'], axis=1)
        df.columns = ['P', 'T', 'RH', 'WS', 'WD', 'Lon', 'Lat', 'Alt','Geo', 'Dew']
        df['WS'] = util.knot2ms(df['WS'])
        return df

    def read_sounding_v2(filename):
        df = pd.read_csv(filename,
                         delim_whitespace=True,
                         skiprows=5,
                         on_bad_lines='skip',
                         encoding='ISO-8859-1'
                        )
        df = df.drop(['min', 's', 'm/s.1'], axis=1)
        df.columns = ['P', 'T', 'RH', 'WS', 'WD', 'Alt', 'Geo', 'Dew', 'Lat', 'Lon']
        # df['WS'] = util.ms2knot(df['WS'])
        return df

    def read_sounding_v3(filename):
        df = pd.read_csv(filename,
                         delim_whitespace=True,
                         skiprows=10,
                         on_bad_lines='skip',
                         encoding='euc-kr'
                        )
        df = df.drop(['Time(mm:ss)', 'Asc(m/m)'], axis=1)
        df.columns = ['P', 'T', 'RH', 'WS', 'WD', 'Lon', 'Lat', 'Alt','Geo', 'Dew']
        df['WS'] = util.knot2ms(df['WS'])
        return df
    
    def clean_df(df):
        def to_float(x):
            try:
                return float(x)
            except ValueError:
                return None

        list_cols = ['P', 'T', 'RH', 'WS', 'WD', 'Alt','Geo', 'Dew', 'Lat', 'Lon']

        _df = df
        for col in list_cols:
            valid_index = _df[col].apply(to_float).dropna().index
            _df = _df.loc[valid_index]

        for col in list_cols:
            _df[col] = _df[col].astype('float64')

        return _df
    
    def get_launch_time(fname):
        try:
            with open(fname, encoding='CP949') as f:
                hdr = f.readline()
                launch_time = datetime.datetime.strptime(hdr[31:50], '%Y-%m-%d %H:%M:%S')
        except:
            try:
                with open(fname, encoding='CP949') as f:
                    hdr = f.readline()
                    launch_time = datetime.datetime.strptime(hdr[29:48], '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    with open(fname, encoding='ISO-8859-1') as f:
                        mdy = f.readline()[-9:-1]
                        his = f.readline()[-9:-1]
                        launch_time = pd.to_datetime(f'{mdy} {his}', format='%d/%m/%y %H:%M:%S')
                except:
                    raise UserWarning(f'Failed reading launch time : {fname}')
        return launch_time
    
    try:
        df = read_sounding_v1(filename)
    except:
        try:
            df = read_sounding_v2(filename)
        except:
            try:
                df = read_sounding_v3(filename)
            except:
                raise UserWarning(f'Cannot read this kind of format: {filename}')

    df = clean_df(df)
    df['U'], df['V'] = util.wind2uv(wd=df['WD'], ws=df['WS'])
    df['Uknot'], df['Vknot'] = util.ms2knot(df['U']), util.ms2knot(df['V'])
    df['time'] = get_launch_time(filename)
    
    return df

def _read_lidar_wind(fname):
    from . import util
    
    df = pd.read_csv(
        fname,
        skiprows=2,
        delim_whitespace=True,
        names=['Alt', 'U', 'V', 'W', 'WS', 'WD', 'Valid'],
        na_values=-999,
    )
    df['Uknot'], df['Vknot'] = util.ms2knot(df['U']), util.ms2knot(df['V'])
    
    return df

def read_lidar_wind(fnames, ftimes, dropna=True):
    """
    Read lidar wind profile.
    
    Examples
    ---------
    >>> df = kkpy.io.read_lidar_wind(fnames)
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of lidar wind profile to read.
    ftimes : str or array_like
        Datetime(s) of lidar wind profile to read (see: kkpy.io.get_fname).
    dropna : boolean (optional)
        True if drop rows containing NaN values.
    
    Returns
    ---------
    df : pandas dataframe object
        Return lidar wind profile dataframe.
    
    Notes
    ---------
    Columns:
        - 'Alt': altitude from lidar [m]
        - 'U': u-wind [m s-1]
        - 'V': v-wind [m s-1]
        - 'W': w-wind [m s-1]
        - 'WS': wind speed [m s-1]
        - 'WD': wind direction [deg]
        - 'Valid': Proportion of the number of data available for wind calculation along the azimuth direction [%]
        - 'Uknot': u-wind [knot]
        - 'Vknot': v-wind [knot]
        - 'time': measurement time [file-dependent, generally KST]
    """

    dfs = []
    
    if isinstance(fnames, (list, np.ndarray)):
        for i_f, fname in enumerate(fnames):
            _df = _read_lidar_wind(fname)
            _df['time'] = ftimes[i_f]
            dfs.append(_df)
        df = pd.concat(dfs)
    else:
        df = _read_lidar_wind(fnames)
        df['time'] = ftimes
    
    if dropna:
        df = df.dropna()
    
    df = df.reset_index(drop=True)
    
    return df

def _read_wpr_kma(fname):
    from . import util
    
    with open(fname) as f:
        _ = f.readline()
        data = f.readline()
    split = data.split('#')
    
    df = pd.DataFrame({
        'Alt': np.float_(split[2:-1:7]),
        'WD': np.float_(split[3::7]),
        'WS': np.float_(split[4::7]),
        'U': np.float_(split[5::7]),
        'V': np.float_(split[6::7]),
        'Uknot': util.ms2knot(np.float_(split[5::7])),
        'Vknot': util.ms2knot(np.float_(split[6::7])),
        'time': None
    })
    
    return df

def read_wpr_kma(fnames, ftimes, dropna=True):
    """
    Read KMA wind profiler.
    
    Examples
    ---------
    >>> df = kkpy.io.read_wpr_kma(fnames)
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of KMA wind profiler data to read.
    ftimes : str or array_like
        Datetime(s) of KMA wind profiler data to read (see: kkpy.io.get_fname).
    dropna : boolean (optional)
        True if drop rows containing NaN values.
    
    Returns
    ---------
    df : pandas dataframe object
        Return KMA wind profiler data dataframe.
    
    Notes
    ---------
    Columns:
        - 'Alt': altitude from wind profiler [m]
        - 'WD': wind direction [deg]
        - 'WS': wind speed [m s-1]
        - 'U': u-wind [m s-1]
        - 'V': v-wind [m s-1]
        - 'Uknot': u-wind [knot]
        - 'Vknot': v-wind [knot]
        - 'time': measurement time [file-dependent, generally KST]
    """
    dfs = []
    
    if isinstance(fnames, (list, np.ndarray)):
        for i_f, fname in enumerate(fnames):
            _df = _read_wpr_kma(fname)
            _df['time'] = ftimes[i_f]
            dfs.append(_df)
        df = pd.concat(dfs)
    else:
        df = _read_wpr_kma(fnames)
        df['time'] = ftimes
    
    if dropna:
        df = df.dropna()
    
    df = df.reset_index(drop=True)
    
    return df

def _read_pluvio_raw(fname):
    df = pd.read_csv(
        fname,
        skiprows=26,
        names=[
            'Date', 'Time', 'STIN',
            'PRT', 'PA', 'PNRT',
            'PATNRT', 'BRT', 'BNRT',
            'T', 'HStatus', 'Status',
            'dummy1', 'dummy2', 'dummy3'],
        delim_whitespace=True,
    )
    df['time'] = pd.to_datetime(
        df['Date']+df['Time'],
        format='%Y/%m/%d%H:%M:%S'
    )
    df = df.drop([
        'STIN','HStatus','Status',
        'Date','Time',
        'dummy1','dummy2','dummy3'],
        axis=1)
    return df

def read_pluvio_raw(fnames, dropna=True):
    """
    Read pluvio raw data.
    
    Examples
    ---------
    >>> df = kkpy.io.read_pluvio(fnames)
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of pluvio raw data to read.
    dropna : boolean (optional)
        True if drop rows containing NaN values.
    
    Returns
    ---------
    df : pandas dataframe object
        Return pluvio raw data dataframe.
    
    Notes
    ---------
    Columns:
        - 'PRT': Intensity RT [mm hr-1]
        - 'PA': Accu RT-NRT [mm]
        - 'PNRT': Accu NRT [mm]
        - 'PATNRT': Accu total NRT [mm]
        - 'BRT': Bucket RT [mm]
        - 'BNRT': Bucket NRT [mm]
        - 'T': Temperature of load cell [degC]
        - 'time': measurement time [file-dependent, generally UTC]
    
    See details at OTT manual (pluvio operating instructions). The variable names are based on the document number 70.020.000.B.E 04-0515.
    """
    dfs = []
    
    if isinstance(fnames, (list, np.ndarray)):
        for i_f, fname in enumerate(fnames):
            _df = _read_pluvio_raw(fname)
            dfs.append(_df)
        df = pd.concat(dfs)
    else:
        df = _read_pluvio_raw(fnames)

    if dropna:
        df = df.dropna()
    
    df = df.reset_index(drop=True)
    
    return df

def _read_2dvd(filename):
    def _yydoy2dt(yydoy):
        return datetime.datetime.strptime(yydoy, '%y%j')

    def _2dvd_set_index_dt1(df, fname):
        yydoy = os.path.basename(fname)[1:6]
        dt_date = _yydoy2dt(yydoy)
        df['year'] = dt_date.year
        df['month'] = dt_date.month
        df['day'] = dt_date.day

        columns_date = ['year','month','day','hour','minute','second','millisecond']
        df = df.set_index(pd.to_datetime(df[columns_date]))

        return df.drop(columns_date, axis=1)

    def _2dvd_set_index_dt2(df, fname):
        dt_date = datetime.datetime.strptime(os.path.basename(fname)[16:24], '%Y%m%d')
        df['year'] = dt_date.year
        df['month'] = dt_date.month
        df['day'] = dt_date.day
        df['hour'] = df['HHMM'] // 100
        df['minute'] = df['HHMM'] % 100
        df = df.drop('HHMM', axis=1)

        columns_date = ['year','month','day','hour','minute']
        df = df.set_index(pd.to_datetime(df[columns_date]))

        return df.drop(columns_date, axis=1)
    
    def read_2dvd_asc1(fname):
        """
        P R I N T O U T   O F   .\DATASET7\V20231_1.hyd 
        TIME STAMP       DIAM    VOL     VEL     OBL    AREA     A>=  A<=  B>=  B<=
        hr mi sc msc     mm      mm3     m/s            mm2
        03 20 36 563     0.40    0.03    0.67    0.27  10907.40  457  461  104  111 
        """
        df = pd.read_csv(
            fname,
            header=None, delim_whitespace=True,
            skiprows=3,
            names=['hour','minute','second','millisecond','D_mm',
                   'VOL_mm3','VEL_ms','OBL','AREA_mm2',
                   'A1','A2','B1','B2'],
            engine='python',
        )
        if isinstance(df.index, pd.MultiIndex):
            raise UserWarning('Failed to read the format')
        df = _2dvd_set_index_dt1(df, fname)
        return df
    
    def read_2dvd_asc2(fname):
        """
        ^[[2J

        ====================================================================
        |                                                                  |
        |         T H E   2 D - V I D E O - D I S T R O M E T E R          |
        |                                                                  |
        |      HYD2ASC                                                     |
        |      sample how to print some hydrometeor-parameters             |
        |      to an ASCII file                                            |
        |                                                                  |
        |      (Feb. 09, 2004, internal version no. 7.002)                 |
        |                                                                  |
        |                                                                  |
        |                                 JOANNEUM RESEARCH, GRAZ/AUSTRIA  |
        ====================================================================



        P R I N T O U T   O F   C:\2dvddata\before_201806\hyd\V18100_1.hyd

        09 51 59 787     0.39    0.03    3.99    0.89  10977.31  540  544  319  322
        """
        df = pd.read_csv(
            fname,
            header=None, delim_whitespace=True,
            skiprows=20, skipfooter=1,
            names=['hour','minute','second','millisecond','D_mm',
                   'VOL_mm3','VEL_ms','OBL','AREA_mm2',
                   'A1','A2','B1','B2'],
            engine='python',
        )
        if isinstance(df.index, pd.MultiIndex):
            raise UserWarning('Failed to read the format')
        df = _2dvd_set_index_dt1(df, fname)
        return df

    def read_2dvd_asc3(fname):
        """
        TYPE_SNO printout of V17340_1.sno 
        TIME STAMP       DIAM    VOL     VEL     OBL    AREA     A>=  A<=  B>=  B<=  wid_A  obl_A  wid_B  obl_B
        hr mi sc msc     mm      mm3     m/s            mm2
        00 01 42 891     3.97   32.7610    0.92    0.00   9819.97  343  380   11   49   4.36   0.76   5.86   0.59
        """
        df = pd.read_csv(
            fname,
            header=None, delim_whitespace=True,
            skiprows=3,
            names=['hour','minute','second','millisecond','D_mm',
                   'VOL_mm3','VEL_ms','OBL','AREA_mm2',
                   'A1','A2','B1','B2','WA','OA','WB','OB'],
            engine='python',
        )
        if isinstance(df.index, pd.MultiIndex):
            raise UserWarning('Failed to read the format')
        df = _2dvd_set_index_dt1(df, fname)
        return df

    def read_2dvd_rho1(fname):
        """
        HOUR  MINUTE SEC MSEC [UTC] APPARENT_DIAMETER VELOCIY DENSITY WA HA WB HB Deq
        0139      0.4257243      1.1390000      0.7007125      0.0109990      0.5010000      0.3910000      0.5010000      0.4330000      0.4250000
        """
        df = pd.read_csv(
            fname,
            delim_whitespace=True,
            names=['HHMM', 'Dapp_mm', 'VEL_ms', 'Rho_gcm3', 'AREA_m2', 'WA', 'HA', 'WB', 'HB', 'D_mm'],
            skiprows=1
        )
        if isinstance(df.index, pd.MultiIndex):
            raise UserWarning('Failed to read the format')
        df['AREA_mm2'] = df['AREA_m2'] * 1e6
        df = df.drop('AREA_m2', axis=1)

        df = _2dvd_set_index_dt2(df, fname)
        return df

    try:
        df = read_2dvd_asc1(filename)
    except:
        try:
            df = read_2dvd_asc2(filename)
        except:
            try:
                df = read_2dvd_asc3(filename)
            except:
                try:
                    df = read_2dvd_rho1(filename)
                except:
                    raise UserWarning(f'Cannot read this kind of format: {filename}')
    
    return df
    
def read_2dvd(fnames, verbose=False):
    """
    Read 2DVD ascii file.
    
    Examples
    ---------
    >>> df_2dvd = kkpy.io.read_2dvd(fnames)
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of 2DVD to read.
    verbose : bool, optional
        True if print reading status
    
    Returns
    ---------
    df : pandas dataframe object
        Return 2dvd dataframe.
    """
    import pandas as pd
    
    dfs = []
    
    if isinstance(fnames, (list, np.ndarray)):
        for i_f, fname in enumerate(fnames):
            if verbose:
                print(i_f, fname)
            _df = _read_2dvd(fname)
            dfs.append(_df)
        df = pd.concat(dfs)
    else:
        df = _read_2dvd(fnames)
    
    return df

def _read_wxt520(fname):
    import pandas as pd
    from . import util
    
    df = pd.read_csv(fname)
    df = df.set_index(pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M:%S.%f'))
    df = df.drop(['Timestamp','TZ','Hail Accum. (hits/cm2)'], axis=1)
    df.columns = ['WD', 'WS', 'MWS', 'T', 'RH', 'P', 'R_ACC']
    df['U'], df['V'] = util.wind2uv(wd=df['WD'], ws=df['WS'])
    return df

def read_wxt520(fnames, verbose=False):
    """
    Read WXT520 ascii file.
    
    Examples
    ---------
    >>> df_wxt = kkpy.io.read_wxt520(fnames)
    
    Parameters
    ----------
    fnames : str or array_like
        Filename(s) of WXT520 to read.
    
    Returns
    ---------
    df : pandas dataframe object
        Return wxt520 dataframe.
    """
    import pandas as pd
    
    dfs = []
    
    if isinstance(fnames, (list, np.ndarray)):
        for i_f, fname in enumerate(fnames):
            _df = _read_wxt520(fname)
            dfs.append(_df)
        df = pd.concat(dfs)
    else:
        df = _read_wxt520(fnames)
    
    return df
