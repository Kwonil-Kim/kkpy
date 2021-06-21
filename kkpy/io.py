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
    Get filename matching the datetime and format.
    
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
            print(pattern)
            print(f'{dt[0]:{pattern}}')
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
            if int(datetime.datetime.strftime(dt[0], lower_dtfmt)) != default_dtvalue[lower_dtfmt]:
                # Truncate unnecessary start time (eg. filename: 20180228, start_datetime: 20180228 13:00 --> 20180228 00:00)
                dt0_trunc = _truncate_unnecessary_datetime(dt[0], lower_dtfmt)
                wh = np.where(np.logical_and(candidate_dt >= dt0_trunc, candidate_dt <= dt[1]))[0]
            else:
                wh = np.where(np.logical_and(candidate_dt >= dt[0], candidate_dt <= dt[1]))[0]
            
            # check if any file found
            if wh.size == 0:
                print(candidate_dt)
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
