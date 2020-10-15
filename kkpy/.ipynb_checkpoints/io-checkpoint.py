"""
kkpy.io
========================

Functions to read and write files

.. currentmodule:: io

.. autosummary::
    kkpy.io.read_aws

"""
import numpy as np
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
