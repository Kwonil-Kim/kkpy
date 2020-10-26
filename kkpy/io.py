"""
kkpy.io
========================

Functions to read and write files

.. currentmodule:: io

.. autosummary::
    kkpy.io.read_aws
    kkpy.io.read_2dvd_rho

"""
import numpy as np
import datetime
import glob

def read_aws(time=None, datadir='/disk/STORAGE/OBS/AWS/', date_range=True):
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
        If this is array of elements and keyword *range* is False, it will read the data of specific time of each element.
    datadir : str, optional
        Directory of the data.
    date_range : bool, optional
        False if argument *time* contains element of specific time you want to read.
        
    Returns
    ---------
    df_aws : dataframe
        Return dataframe of aws data.
    """
    
    list_aws = []
    
    filearr_1d = np.sort(glob.glob(aws_dir+YYYYMMDD[:6]+'/'+YYYYMMDD[6:8]+'/'+'AWS_MIN_'+YYYYMMDD+'*'))
    for file in filearr_1d:
        data = pd.read_csv(file, delimiter='#', names=names)
        data.wd         = data.wd/10.
        data.ws         = data.ws/10.
        data.t          = data.t/10.
        data.rh         = data.rh/10.
        data.pa         = data.pa/10.
        data.ps         = data.ps/10.
        data.rn60m_acc  = data.rn60m_acc/10.
        data.rn1d       = data.rn1d/10.
        data.rn15m      = data.rn15m/10.
        data.rn60m      = data.rn60m/10.
        data.wds        = data.wds/10.
        data.wss        = data.wss/10.
        list_aws.append(data[data.stn == 100].values.tolist()[0])

    df_aws = pd.DataFrame(list_aws, columns=names)
    
    
    return df_aws



def read_2dvd_rho(time=None, datadir='/disk/STORAGE/OBS/AWS/', date_range=True):
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
        If this is array of elements and keyword *range* is False, it will read the data of specific time of each element.
    datadir : str, optional
        Directory of the data.
    date_range : bool, optional
        False if argument *time* contains element of specific time you want to read.
        
    Returns
    ---------
    df_aws : dataframe
        Return dataframe of aws data.
    """
    # Get file list
    filearr = np.array(np.sort(glob.glob(f'{datadir}/**/2DVD_Dapp_v_rho_201*Deq.txt', recursive=True)))
    yyyy_filearr = [np.int(os.path.basename(x)[-27:-23]) for x in filearr]
    mm_filearr = [np.int(os.path.basename(x)[-23:-21]) for x in filearr]
    dd_filearr = [np.int(os.path.basename(x)[-21:-19]) for x in filearr]
    dt_filearr = np.array([datetime.datetime(yyyy,mm,dd) for (yyyy, mm, dd) in zip(yyyy_filearr, mm_filearr, dd_filearr)])

    # Select file within time range
    if ((len(time) == 2) & date_range):
        dt_start = time[0]
        dt_finis = time[1]
        # dt_start = datetime.datetime(2017,12,24)
        # dt_finis = datetime.datetime(2017,12,25)
        filearr = filearr[(dt_filearr >= dt_start) & (dt_filearr < dt_finis)]
        dt_filearr = dt_filearr[(dt_filearr >= dt_start) & (dt_filearr < dt_finis)]
    
    
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
        _df['jultime_minute'] = _df['jultime']
        _df = _df.drop(['hhmm','year','month','day','hour','minute'], axis=1)
        _df = _df.drop(['Dapp', 'RHO', 'WA', 'HA', 'WB', 'HB'], axis=1)
        dflist.append(_df)
        print(i_file+1, filearr.size, file)

    df_2dvd_drop = pd.concat(dflist, sort=False, ignore_index=True)
    df_2dvd_drop.set_index('jultime', inplace=True)

    return df_2dvd_drop
