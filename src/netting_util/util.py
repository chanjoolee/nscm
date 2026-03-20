'''
'''
import os
import psutil
import datetime

import numpy as np
import pandas as pd

def convert_dt_to_week(series: pd.Series) -> pd.Series:
    '''
    pandas datetime 타입의 series를, object 타입의 series로 변환한다.
    이때 변환된 week의 형식은 YYYYWW (ex: 202507)이다.
    '''
    isocalendar = series.dt.isocalendar()
    return isocalendar.year.astype(str) + isocalendar.week.astype(str).str.zfill(2)

def overrange_to_datetime(series: pd.Series) -> pd.Series:
    # object가 아니면 pass
    if series.dtype != np.dtype('O'):
        return series
    
    nan_val_num = -11111111111
    nan_date = np.datetime64(nan_val_num, 'D')
    nat = np.datetime64('NaT')
    pd_date_min = np.datetime64('1677-09-22', 'D')
    pd_date_max = np.datetime64('2262-04-11', 'D')
    
    np_date = pd.to_datetime(series.fillna(nan_val_num), errors='coerce').to_numpy().astype('datetime64[D]')
    np_date = np.where(np_date == nan_date, nat, np_date)

    clipped_date = np.clip(np_date, pd_date_min, pd_date_max)

    return pd.to_datetime(clipped_date)

def memory_exception():
    ''' Memory 사용량을 확인 후, 90프로를 초과할 경우 exception을 던진다. '''
    pMemory = round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 2)
    aMemory = round(psutil.virtual_memory().available / (1024 ** 2), 2)
    pRatio = round((pMemory / aMemory) * 100, 0)

    if pRatio > 90:
        raise Exception('Memory utilization exceeded 90%')

def convert_to_iyyyiw(date_str) -> str:
    '''
    Convert a date string in 'YYYY-MM-DD' format to IYYYIW format.
    
    Parameters:
    date_str (str): The date string to convert.
    
    Returns:
    str: The date in IYYYIW format.
    '''
    # 문자열을 날짜로 변환
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # ISO 연도와 주 번호 가져오기
    iso_year, iso_week, _ = date_obj.isocalendar()
    
    # IYYYIW 형식으로 변환
    return f'{iso_year}{iso_week:02d}'

def make_sequential_id(prefix: str, digit: int, dataframe: pd.DataFrame) -> pd.Series:
    return prefix + (dataframe.index + 1).astype(str).str.zfill(digit)
