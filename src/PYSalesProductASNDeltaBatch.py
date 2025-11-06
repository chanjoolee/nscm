from re import X
import os,sys,json,shutil,io,zipfile
import time
import datetime
import inspect
import traceback
import pandas as pd
from pandas.core.resample import T
from NSCMCommon import NSCMCommon as common
from NSCMCommon import VDCommon as vdCommon
# from typing_extensions import Literal
import glob
import numpy as np
from typing import Collection, Tuple,Union,Dict
import re
import gc
import functools
# import rbql
# import duckdb

########################################################################################################################
# Local 개발 시에 필요한 공통 변수 선언
########################################################################################################################
# o9에 저장된 instanceName
is_local = common.gfn_get_isLocal()
str_instance = 'PYSalesProductASNDeltaBatch'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"

is_print = True
flag_csv = True
flag_exception = True

# ======================================================
# 컬럼 상수 (Columns)
# ======================================================
COL_VERSION                  = 'Version.[Version Name]'# 공통 키
COL_SHIP_TO                  = 'Sales Domain.[Ship To]'
COL_ITEM                     = 'Item.[Item]'
COL_LOCATION                 = 'Location.[Location]'

# 시간 (current version 기준)
COL_PWEEK                    = 'Time.[Partial Week]'   # Step 3-1, 3-2, 3-3-2 모두 Partial Week 사용

# 세일즈 도메인 층위
COL_STD1                     = 'Sales Domain.[Sales Std1]'
COL_STD2                     = 'Sales Domain.[Sales Std2]'
COL_STD3                     = 'Sales Domain.[Sales Std3]'
COL_STD4                     = 'Sales Domain.[Sales Std4]'
COL_STD5                     = 'Sales Domain.[Sales Std5]'
COL_STD6                     = 'Sales Domain.[Sales Std6]'

# ASN
COL_SALES_PRODUCT_ASN        = 'Sales Product ASN'
COL_SALES_PRODUCT_ASN_DELTA  = 'Sales Product ASN Delta'

# Assortment (S/In)
COL_SIN_ASSORT_AP1           = 'S/In FCST(GI) Assortment_AP1'
COL_SIN_ASSORT_AP2           = 'S/In FCST(GI) Assortment_AP2'
COL_SIN_ASSORT_GC            = 'S/In FCST(GI) Assortment_GC'
COL_SIN_ASSORT_LOCAL         = 'S/In FCST(GI) Assortment_Local'

# Assortment (S/Out)
COL_SOUT_ASSORT_AP1          = 'S/Out FCST Assortment_AP1'
COL_SOUT_ASSORT_AP2          = 'S/Out FCST Assortment_AP2'
COL_SOUT_ASSORT_GC           = 'S/Out FCST Assortment_GC'
COL_SOUT_ASSORT_LOCAL        = 'S/Out FCST Assortment_Local'

# Dummy Measures (S/In)
COL_SIN_DUMMY_AP1            = 'S/In FCST(GI) Dummy_AP1'
COL_SIN_DUMMY_AP2            = 'S/In FCST(GI) Dummy_AP2'
COL_SIN_DUMMY_GC             = 'S/In FCST(GI) Dummy_GC'
COL_SIN_DUMMY_LOCAL          = 'S/In FCST(GI) Dummy_Local'

# Dummy Measures (S/Out)
COL_SOUT_DUMMY_AP1           = 'S/Out FCST Dummy_AP1'
COL_SOUT_DUMMY_AP2           = 'S/Out FCST Dummy_AP2'
COL_SOUT_DUMMY_GC            = 'S/Out FCST Dummy_GC'
COL_SOUT_DUMMY_LOCAL         = 'S/Out FCST Dummy_Local'

# Flooring Dummy
COL_FLOORING_DUMMY           = 'Flooring FCST Dummy'

# 기타 참조
COL_ITEM_GBM                 = 'Item.[Item GBM]'
COL_ITEM_STD1                = 'Item.[Item Std1]'
COL_ITEM_STD3                = 'Item.[Item Std3]'
COL_PG                       = 'Item.[Product Group]'
COL_SOUT_MASTER_STATUS       = 'S/Out Master Status'

# Forecast-Rule
COL_FRULE_GC_FCST               = 'FORECAST_RULE GC FCST'
COL_FRULE_AP2_FCST              = 'FORECAST_RULE AP2 FCST'
COL_FRULE_AP1_FCST              = 'FORECAST_RULE AP1 FCST'
COL_FRULE_AP0_FCST              = 'FORECAST_RULE AP0 FCST'
COL_FRULE_ISVALID               = 'FORECAST_RULE ISVALID'


# ======================================================
# 데이터프레임 상수 (Input)
# ======================================================
DF_IN_SALES_DOMAIN_DIMENSION = 'df_in_Sales_Domain_Dimension'
DF_IN_TIME                   = 'df_in_Time'                     # Partial Week만 포함 (current version)
DF_IN_SALES_DOMAIN_ESTORE    = 'df_in_Sales_Domain_Estore'
DF_IN_FORECAST_RULE          = 'df_in_Forecast_Rule'
DF_IN_SALES_PRODUCT_ASN_DELTA= 'df_in_Sales_Product_ASN_Delta'
DF_IN_SALES_PRODUCT_ASN      = 'df_in_Sales_Product_ASN'
DF_IN_SIN_ASSORTMENT         = 'df_in_SIn_Assortment'
DF_IN_SOUT_ASSORTMENT        = 'df_in_SOut_Assortment'
DF_IN_SOUT_SIMUL_MASTER      = 'df_in_Sell_Out_Simul_Master'    # (TO-BE : 7LV)
DF_IN_ITEM_MASTER            = 'df_in_item'                      # (VD 구분/참조)


# ======================================================
# 데이터프레임 상수 (Output)
# ======================================================

# Step 1
OUT_SALES_PRODUCT_ASN        = 'Output_Sales_Product_ASN'
OUT_SALES_PRODUCT_ASN_DELTA  = 'Output_Sales_Product_ASN_Delta'

# Step 2: Assortment (8개)
OUT_SIN_ASSORTMENT_GC        = 'Output_SIn_Assortment_GC'
OUT_SIN_ASSORTMENT_AP2       = 'Output_SIn_Assortment_AP2'
OUT_SIN_ASSORTMENT_AP1       = 'Output_SIn_Assortment_AP1'
OUT_SIN_ASSORTMENT_LOCAL     = 'Output_SIn_Assortment_Local'
OUT_SOUT_ASSORTMENT_GC       = 'Output_SOut_Assortment_GC'
OUT_SOUT_ASSORTMENT_AP2      = 'Output_SOut_Assortment_AP2'
OUT_SOUT_ASSORTMENT_AP1      = 'Output_SOut_Assortment_AP1'
OUT_SOUT_ASSORTMENT_LOCAL    = 'Output_SOut_Assortment_Local'

# Step 3: FCST Dummy (9개, 모두 Partial Week 축)
OUT_SIN_DUMMY_GC             = 'Output_SIn_Dummy_GC'
OUT_SIN_DUMMY_AP2            = 'Output_SIn_Dummy_AP2'
OUT_SIN_DUMMY_AP1            = 'Output_SIn_Dummy_AP1'
OUT_SIN_DUMMY_LOCAL          = 'Output_SIn_Dummy_Local'
OUT_SOUT_DUMMY_GC            = 'Output_SOut_Dummy_GC'
OUT_SOUT_DUMMY_AP2           = 'Output_SOut_Dummy_AP2'
OUT_SOUT_DUMMY_AP1           = 'Output_SOut_Dummy_AP1'
OUT_SOUT_DUMMY_LOCAL         = 'Output_SOut_Dummy_Local'
OUT_FLOORING_DUMMY           = 'Output_Flooring_Dummy'          # Step 3-3-2도 Partial Week 사용

########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################
logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
# fn_set_local_logfile()
LOG_LEVEL = common.G_log_level

def fn_log_dataframe(df_p_source: pd.DataFrame, str_p_source_name: str,int_p_row_num: int = 20) -> None:
    """
    Dataframe 로그 출력 조건 지정 함수
    :param df_p_source: 로그로 찍을 Dataframe
    :param str_p_source_name: 로그로 찍을 Dataframe 명
    :return: None
    """
    is_output = False
    if str_p_source_name.startswith('out_'):
        is_output = True

    if is_print:
        logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=int_p_row_num)
        # if is_local and not df_p_source.empty and flag_csv:
        if is_local and flag_csv:
            # 로컬 Debugging 시 csv 파일 출력
            df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)
    else:
        # 최종 Output 테이블인 경우에는 무조건 로그 출력
        if is_output:
            logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=20)
            # if is_local and not df_p_source.empty:
            if is_local:
                # 로컬 Debugging 시 csv 파일 출력
                df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)



def _decoration_(func):
    """
    1. 소스 내 함수 실행 시 반복되는 코드를 데코레이터로 변형하여 소스 라인을 줄일 수 있도록 함.
    2. 각 Step을 함수로 실행하는 경우 해당 함수에 뒤따르는 Step log 및 DF 로그, DF 로컬 출력을 데코레이터로 항상 출력하게 함.
    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        # 함수 시작 시각
        tm_start = time.time()
        # 함수 실행
        result = func(*args)
        # 함수 종료 시각
        tm_end = time.time()
        # 함수 실행 시간 로그
        logger.Note(p_note=f'[{func.__name__}] Total time is {tm_end - tm_start:.5f} sec.',
                    p_log_level=LOG_LEVEL.debug())
        # Step log 및 DF 로컬 출력 등을 위한 Keywords 변수 확인
        # Step No
        _step_no = kwargs.get('p_step_no')
        _step_desc = kwargs.get('p_step_desc')
        vdCommon.gfn_pyLog_detail(_step_desc)
        _df_name = kwargs.get('p_df_name')
        _warn_desc = kwargs.get('p_warn_desc')
        _exception_flag = kwargs.get('p_exception_flag')
        # Step log 관련 변수가 입력된 경우 Step log 출력
        if _step_no is not None and _step_desc is not None:
            logger.Step(p_step_no=_step_no, p_step_desc=_step_desc)
        # Warning 메시지가 있는 경우
        if _warn_desc is not None:
            # 함수 실행 결과가 DF이면서 해당 DF가 비어 있는 경우
            if type(result) == pd.DataFrame and result.empty:
                # Exception flag가 확인되고
                if _exception_flag is not None:
                    # Exception flag가 0이면 Warning 로그 출력, 1이면 Exception 발생시킴
                    if _exception_flag == 0:
                        logger.Note(p_note=_warn_desc, p_log_level=LOG_LEVEL.warning())
                    elif _exception_flag == 1:
                        raise Exception(_warn_desc)
        # DF 명이 있는 경우 로그 및 로컬 출력
        if _df_name is not None:
            fn_log_dataframe(result, _df_name)
        return result
    return wrapper


def fn_check_input_table(df_p_source: pd.DataFrame, str_p_source_name: str, str_p_cond: str) -> None:
    """
    Input Table을 체크한 결과를 로그 또는 Exception으로 표시한다.
    :param df_p_source: Input table
    :param str_p_source_name: Name of Input table
    :param str_p_cond: '0' - Exception, '1' - Warning Log
    :return: None
    """
    # Input Table 로그 출력
    logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1)

    if df_p_source.empty:
        if str_p_cond == '0':
            # 테이블이 비어 있는 경우 raise Exception
            raise Exception(f'[Exception] Input table({str_p_source_name}) is empty.')
        else:
            # 테이블이 비어 있는 경우 Warning log
            logger.Note(p_note=f'Input table({str_p_source_name}) is empty.', p_log_level=LOG_LEVEL.warning())


def fn_get_week(list_p_weeks: list, p_row: any) -> list:
    """
    in_Demand의 행과 Time.[Week] 목록을 받아 Time.[Week] - W Demand Build Ahead Limit<= t < Time.[Week]인 t의 목록을 찾아 리턴
    :param list_p_weeks:
    :param p_row:
    :return:
    """
    int_end = int(list_p_weeks.index(p_row['Time.[Week]']))
    int_start = int_end - int(p_row['W Demand Build Ahead Limit'])
    if int_start < 0:
        int_start = 0

    return list_p_weeks[int_start:int_end]

def fn_use_x_after_join(df_source: pd.DataFrame):
    """
    When join , there is 
    """
    df_source.columns = [col.replace('_x', '') if '_x' in col else col for col in df_source.columns]
    # Drop columns with '_y' suffix
    df_source.drop(columns=[col for col in df_source.columns if '_y' in col], inplace=True)
    # df_source = df_source.loc[:, ~df_source.columns.str.endswith('_y')]

def fn_use_y_after_join(df_source: pd.DataFrame):
    """
    When join , there is 
    """
    df_source.columns = [col.replace('_y', '') if '_y' in col else col for col in df_source.columns]
    # Drop columns with '_y' suffix
    df_source.drop(columns=[col for col in df_source.columns if '_x' in col], inplace=True)

# Remove '_x' and '_y' suffixes, keeping '_x' for specified columns
def customize_column_names(df_source: pd.DataFrame, column_use_y: list):
    # Replace '_y' with '' for columns not in column_use_y
    for col in df_source.columns:
        if '_y' in col:
            for col_y in column_use_y:
                if col_y in col:
                    df_source = df_source.rename(columns={col: col.replace('_y', '')})

    # Drop columns with '_x' suffix
    columns_x_to_drop = []
    for col in df_source.columns:
        if '_x' in col:
            for col_y in column_use_y:
                if col_y in col:
                    columns_x_to_drop.append(col)

    df_source.drop(columns=columns_x_to_drop, inplace=True)
    fn_use_x_after_join(df_source)


@_decoration_
def fn_make_week_list(df_p_source: pd.DataFrame) -> list:
    """
    전처리 - in_Time 테이블에서 Time.[Week]을 오름차순으로 정렬하여 리스트로 변환 후리턴
    :param df_p_source: in_Time
    :return: DataFrame
    """
    # 함수명
    str_my_name = inspect.stack()[0][3]
    
    # 입력 파라미터가 비어 있는 경우 비어 있는 DataFrame을 리턴
    if df_p_source.empty:
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(df_p_source)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    # 오름차순 정렬 후 'Time.[Week]'를 리스트로 변환
    list_return = df_p_source.sort_values(by='Time.[Week]')['Time.[Week]'].to_list()
    
    return list_return

def normalize_week(week_str):
    """Convert a week string with potential suffixes to an integer for comparison."""
    # Remove any non-digit characters (e.g., 'A' or 'B') and convert to integer
    return ''.join(filter(str.isdigit, week_str))

def is_within(current_week, start_week, end_week):
    """
    Check if the current week is within the range defined by start and end weeks.
    """
    return start_week <= current_week <= end_week

def fn_convert_type(df: pd.DataFrame, startWith: str, type):
    for column in df.columns:
        if column.startswith(startWith):
            df[column] = df[column].astype(type,errors='ignore')

# ───── Ship-To → Level LUT ───────────────────────
def build_shipto_level_lut(df_dim: pd.DataFrame):
    """
    Return (pd.Index, np.ndarray[int32], dict) for fast level lookup.
    """
    COL_LVS = [

        (COL_STD6 ,7),
        (COL_STD5, 6), (COL_STD4, 5), (COL_STD3, 4),
        (COL_STD2, 3), (COL_STD1, 2)
    ]
    lut = {}
    for col, lv in COL_LVS:
        lut.update({code: lv for code in df_dim[col].dropna().unique()})
    idx = pd.Index(lut.keys(), dtype=object)
    arr = np.fromiter(lut.values(), dtype='int32')
    return idx, arr, lut

def build_shipto_dim_arrays(df_dim: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
    """
    Returns
    -------
    dim_idx : Index(level-7 ShipTo)
    lv_arrs : ndarray shape(n,6) [LV2 … LV7]
              (컬럼순 : 2,3,4,5,6,7)
    """
    dim_idx = df_dim.set_index(COL_SHIP_TO)
    lv_cols = [COL_STD1, COL_STD2, COL_STD3,
               COL_STD4, COL_STD5, COL_STD6]
    lv_arrs = dim_idx[lv_cols].to_numpy(dtype=object)
    return dim_idx.index, lv_arrs
# -------------------------------------------------


def round_half_up_to_2(series: pd.Series) -> pd.Series:
    """
    반올림 규칙: Half-Up (3번째 자리에서 5 이상이면 올림).
    벡터화 연산으로 빠르게 처리. NaN은 그대로 유지.
    """
    s = pd.to_numeric(series, errors='coerce')  # 문자열/빈값 → NaN
    mask = s.notna()
    s2 = s.copy()
    abs_s = np.abs(s2[mask].to_numpy(dtype='float64'))
    # Half-Up: sign * floor(|x|*100 + 0.5) / 100
    rounded = (np.sign(s2[mask]) * np.floor(abs_s * 100.0 + 0.5)) / 100.0
    s2.loc[mask] = rounded
    return s2

################################################################################################################──────────
#  공통 타입 변환  (❌ `global` 사용 금지)
#  호출 측에서 `input_dataframes` 를 인자로 넘겨준다.
################################################################################################################──────────
def _fn_prepare_input_types(dict_dfs: dict) -> None:
    """
    dict_dfs :  { <df_name> : pandas.DataFrame, ... }
    
    • object  → str → category                (숫자·문자 혼재 대비)
    • float/int → fillna(0) → int32           (값이 실수면 round 후 변환)    **주의** : dict 내부의 DataFrame 을 *제자리*에서 변환하므로 반환값은 없다.
    """
    if not dict_dfs:        # 빈 dict 방어
        return

    for df_name, df in dict_dfs.items():
        if df.empty:
            continue

        # 1) object 컬럼 : str → category
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df[col] = df[col].astype(str).astype("category")

        # 2) numeric 컬럼 : fillna → int32
        num_cols = df.select_dtypes(
            include=["float64", "float32", "int64", "int32","int"]
        ).columns
        for col in num_cols:
            df[col].fillna(0, inplace=True)
            try:
                df[col] = df[col].astype("int32")
            except ValueError:
                df[col] = df[col].round().astype("int32")

def _fn_prepare_input_type(df: pd.DataFrame) -> None:
    if df.empty:
        return

    # 1) object 컬럼 : str → category
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).astype("category")

    # 2) numeric 컬럼 : fillna → int32
    num_cols = df.select_dtypes(
        include=["float64", "float32", "int64", "int32","int"]
    ).columns
    for col in num_cols:
        df[col].fillna(0, inplace=True)
        try:
            df[col] = df[col].astype("int32")
        except ValueError:
            df[col] = df[col].round().astype("int32")

def fn_prepare_input_types(dict_dfs: dict) -> None:
    if not dict_dfs:
        return    

    for df_name, df in dict_dfs.items():
        if df.empty:
            continue

        # 1) object → str → category
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df[col] = df[col].astype(str).astype("category")

        # 2) 정수만 int32로, 실수는 유지
        int_cols = df.select_dtypes(include=["int64", "int32", "Int64", "Int32", "int"]).columns
        for col in int_cols:
            df[col].fillna(0, inplace=True)
            df[col] = df[col].astype("int32")

        float_cols = df.select_dtypes(include=["float64", "float32"]).columns
        # 필요 시 공통 결측 처리만. 반올림/형변환은 각 도메인 함수(가격 등)에서 수행
        for col in float_cols:
            df[col].fillna(np.nan, inplace=True)

@_decoration_
def fn_process_in_df_mst():

    if is_local: 
        # 로컬인 경우 Output 폴더를 정리한다.
        for file in os.scandir(str_output_dir):
            os.remove(file.path)

        # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
        file_pattern = f"{os.getcwd()}/{str_input_dir}/*.csv" 
        csv_files = glob.glob(file_pattern)


        file_to_df_mapping = {
            f'{DF_IN_SALES_DOMAIN_DIMENSION 	}.csv' : 	DF_IN_SALES_DOMAIN_DIMENSION   			,
            f'{DF_IN_TIME                   	}.csv' :    DF_IN_TIME                              ,
            f'{DF_IN_SALES_DOMAIN_ESTORE    	}.csv' :    DF_IN_SALES_DOMAIN_ESTORE               ,
            f'{DF_IN_FORECAST_RULE          	}.csv' :    DF_IN_FORECAST_RULE                     ,
            f'{DF_IN_SALES_PRODUCT_ASN_DELTA	}.csv' :    DF_IN_SALES_PRODUCT_ASN_DELTA           ,
            f'{DF_IN_SALES_PRODUCT_ASN      	}.csv' :    DF_IN_SALES_PRODUCT_ASN                 ,
            f'{DF_IN_SIN_ASSORTMENT         	}.csv' :    DF_IN_SIN_ASSORTMENT                    ,
            f'{DF_IN_SOUT_ASSORTMENT        	}.csv' :    DF_IN_SOUT_ASSORTMENT                   ,
            f'{DF_IN_SOUT_SIMUL_MASTER          }.csv' :    DF_IN_SOUT_SIMUL_MASTER                 ,
            f'{DF_IN_ITEM_MASTER            	}.csv' :    DF_IN_ITEM_MASTER                       
              
        }

        def read_csv_with_fallback(filepath):
            encodings = ['utf-8-sig', 'utf-8', 'cp949']
            
            for enc in encodings:
                try:
                    return pd.read_csv(filepath, encoding=enc)
                except UnicodeDecodeError:
                    continue
            
            raise ValueError(f"Unable to read file {filepath} with tried encodings.")

        # Read all CSV files into a dictionary of DataFrames
        for file in csv_files:
            df = read_csv_with_fallback(file)
            file_name = file.split("/")[-1].split("\\")[-1].split(".")[0]
            # df['SourceFile'] = file_name
            # df.set_index('SourceFile',inplace=True)
            mapped = False
            for keyword, frame_name in file_to_df_mapping.items():
                # if file_name.startswith(keyword.split('.')[0]):
                if file_name == keyword.split('.')[0]:
                    input_dataframes[frame_name] = df
                    mapped = True
                    break
    else:
        # o9 에서 
        input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION 	] = df_in_Sales_Domain_Dimension         	
        input_dataframes[DF_IN_TIME                     ] = df_in_Time                           	
        input_dataframes[DF_IN_SALES_DOMAIN_ESTORE      ] = df_in_Sales_Domain_Estore            	
        input_dataframes[DF_IN_FORECAST_RULE            ] = df_in_Forecast_Rule                  	
        input_dataframes[DF_IN_SALES_PRODUCT_ASN_DELTA  ] = df_in_Sales_Product_ASN_Delta        	
        input_dataframes[DF_IN_SALES_PRODUCT_ASN        ] = df_in_Sales_Product_ASN              	
        input_dataframes[DF_IN_SIN_ASSORTMENT           ] = df_in_SIn_Assortment                 	
        input_dataframes[DF_IN_SOUT_ASSORTMENT          ] = df_in_SOut_Assortment                	
        input_dataframes[DF_IN_SOUT_SIMUL_MASTER        ] = df_in_Sell_Out_Simul_Master 
        input_dataframes[DF_IN_ITEM_MASTER              ] = df_in_item              


    # type convert : str ==> category, int ==> int32
    fn_convert_type(input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION], 'Sales Domain', str)    

    fn_convert_type(input_dataframes[DF_IN_TIME], 'Time.', str)  

    fn_convert_type(input_dataframes[DF_IN_SALES_DOMAIN_ESTORE], 'Sales Domain.', str)  

    fn_convert_type(input_dataframes[DF_IN_FORECAST_RULE], 'Sales Domain', str) 
    fn_convert_type(input_dataframes[DF_IN_FORECAST_RULE], 'FORECAST_RULE', 'int32') 
    # _fn_prepare_input_types({f'DF_IN_FORECAST_RULE':input_dataframes[DF_IN_FORECAST_RULE]})
    _fn_prepare_input_type(input_dataframes[DF_IN_FORECAST_RULE])
    #
    fn_convert_type(input_dataframes[DF_IN_SALES_PRODUCT_ASN_DELTA], 'Sales Domain', str)  
    fn_convert_type(input_dataframes[DF_IN_SALES_PRODUCT_ASN], 'Sales Domain', str)     

    fn_convert_type(input_dataframes[DF_IN_SIN_ASSORTMENT], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_ASSORTMENT], 'S/In FCST(GI)', 'int32') 
    _fn_prepare_input_type(input_dataframes[DF_IN_SIN_ASSORTMENT])

    fn_convert_type(input_dataframes[DF_IN_SOUT_ASSORTMENT], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_ASSORTMENT], 'S/Out FCST Assortment', 'int32') 
    _fn_prepare_input_type(input_dataframes[DF_IN_SOUT_ASSORTMENT])
    
    # # 가격 관련 컬럼명 상수 세트

    # df_in_Sell_Out_Simul_Master
    fn_convert_type(input_dataframes[DF_IN_SOUT_SIMUL_MASTER], 'Sales Domain', str)
    fn_prepare_input_types(input_dataframes)


# 1. Sanitize date string
# def sanitize_date_string(x):
#     if pd.isna(x):
#         return ''
#     x = str(x).strip()
#     for token in ['PM', 'AM', '오전', '오후']:
#         if token in x:
#             x = x.split(token)[0].strip()
#     return x[:10]  # Keep only 'YYYY/MM/DD'

def sanitize_date_string(x: object) -> str:
    """
    * 입력 예
        12/4/2020 12:00:00 AM
        2025-02-03 12:00:00 AM
        2019-09-16
        ''
    * 처리
        ① 공백 앞(= time 부분) 제거  
        ② `-` → `/` 통일  
        ③ 자리수에 따라  
            - YYYY/MM/DD  → 그대로  
            - M/D/YYYY    → 0-padding 후 YYYY/MM/DD 로 변환  
        ④ 실패 시 '' 리턴
    """
    if pd.isna(x) or str(x).strip() == '':
        return ''

    s = str(x).strip()

    # ① 공백(혹은 T) 이후 time 문자열 제거
    s = re.split(r'\s+|T', s, maxsplit=1)[0]

    # ② 구분자 통일
    s = s.replace('-', '/')

    # ③ 날짜 포맷 판별·정규화
    parts = s.split('/')
    try:
        if len(parts) == 3:
            # case-A : YYYY/MM/DD
            if len(parts[0]) == 4:
                y, m, d = parts
            # case-B : M/D/YYYY  또는  MM/DD/YYYY
            else:
                m, d, y = parts
            dt_obj = datetime.datetime(int(y), int(m), int(d))    # 유효성 체크
            return dt_obj.strftime('%Y/%m/%d')              # zero-padding 포함
    except Exception:
        pass       # fall-through → 실패 처리
    return ''       # 파싱 실패

# 벡터라이즈 버전
v_sanitize_date_string = np.vectorize(sanitize_date_string, otypes=[object])

# v_sanitize_date_string = np.vectorize(sanitize_date_string)

# 2. Validate date
@np.vectorize
def is_valid_date(x):
    try:
        if pd.isna(x) or x == '':
            return True
        datetime.datetime.strptime(str(x), '%Y/%m/%d')
        return True
    except:
        return False

# 3. Convert to datetime
@np.vectorize
def safe_strptime(x):
    try:
        return datetime.datetime.strptime(str(x), '%Y/%m/%d') if pd.notna(x) and x != '' else None
    except:
        return None

# 4. Convert to partial week with error-checking
@np.vectorize
def to_partial_week(item,shipto,x):
    try:
        if x is not None and x != '':
            # If x is not already a Python datetime, try to convert it
            if not isinstance(x, datetime.datetime):
                # This conversion uses pandas to ensure we get a proper Python datetime
                x = pd.to_datetime(x).to_pydatetime()
            # Convert Python datetime to numpy.datetime64 with seconds precision
            np_dt = np.datetime64(x, 's')
            seconds = (np_dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
            dt_utc = datetime.datetime.utcfromtimestamp(seconds)
            return common.gfn_get_partial_week(dt_utc, True)
        else: 
            return ''
    except Exception as e:
        print("Error in to_partial_week with value:", item, shipto, x, "Error:", e)
        return ''


def to_partial_week_datetime(x: Union[str, datetime.date, datetime.datetime]) -> str:
    """
    Robust date-string → 'YYYYWWA/B' converter.
    1. try ``pandas.to_datetime`` (handles *most* inputs fast, incl. numpy64)
    2. fallback to explicit ``strptime`` with the four formats above
    3. log & *raise* if none succeed
    Returns empty-string for ``None`` / '' / NaN.
    """
    _DATE_FMTS = (
        '%Y/%m/%d',   # ① 2025/04/16
        '%Y-%m-%d',   # ② 2025-04-16
        '%m-%d-%Y',   # ③ 04-16-2025
        '%m/%d/%Y',   # ④ 04/16/2025
    )

    if x is None or (isinstance(x, str) and not x.strip()) or pd.isna(x):
        return ''
    # ---------- 1) pandas fast-path ----------
    try:
        dt = pd.to_datetime(x, errors='raise').to_pydatetime()
        return common.gfn_get_partial_week(dt, True)
    except Exception as e_fast:        # noqa: BLE001
        last_exc = e_fast   # remember last exception for logging
    # ---------- 2) explicit strptime fallbacks ----------
    x_str = str(x).strip()
    for fmt in _DATE_FMTS:
        try:
            dt = datetime.datetime.strptime(x_str, fmt)
            return common.gfn_get_partial_week(dt, True)
        except ValueError as exc:
            last_exc = exc              # keep most recent for message
            continue
    # ---------- 3) give up ----------
    msg = f"[to_partial_week_datetime] un-parsable date: {x!r} – last error: {last_exc}"
    logger.Note(p_note=msg, p_log_level=LOG_LEVEL.error())   # or logger.error(...)
    raise ValueError(msg)



def to_add_week(row):
    try:
        if x is not None and x != '':
            # If x is not already a Python datetime, try to convert it
            dt = common.gfn_add_week(x, -1)
            return common.gfn_get_partial_week(dt, True)
        else: 
            return ''
    except Exception as e:
        print("Error in to_partial_week with value:", item, shipto, x, "Error:", e)
        return ''

@np.vectorize
def is_valid_add_week(x):
    try:
        dt = common.gfn_add_week(x, -1)
        return True
    except:
        return False



    
# ──────────────────────────────────────────────────────────────────────────────
# Ultra-fast, NumPy-only groupby (generalised) — ndarray/Series 안전 변환 패치
# ------------------------------------------------------------------------------
# • 목표: 거대한 DF에서 pandas.groupby 대신 **완전 벡터라이즈**로 집계
# • 키 컬럼 개수 제한 없음 (2, 3, 4…)
# • 지원 집계: 'sum', 'max', 'min', 'any', 'all', 'first', 'last', 'count'
#     - 'max'/'min'은 bool 또는 'Y'/'N' (문자) 플래그에도 안전하게 동작
# • 반환: key + 집계결과 컬럼들
# • 팁: 반환 후 qty=1 고정, Location='-' 등은 호출부에서 .assign 로 추가
# ──────────────────────────────────────────────────────────────────────────────
# • pd.to_numeric(...) 가 numpy.ndarray 를 반환할 때 .to_numpy() 가 없어
#   AttributeError 가 나는 이슈를 해결했습니다.
# • dtype 분기(숫자/불리언/문자) 명확화 + 변환 경로 최소화.
# ──────────────────────────────────────────────────────────────────────────────
def ultra_fast_groupby_numpy_general(
    df: pd.DataFrame,
    key_cols: list[str],
    aggs: dict[str, str],
    *,
    cast_key_to_category: bool = True,
    treat_YN_as_bool: bool = True,
) -> pd.DataFrame:
    """
    Ultra-fast groupby using NumPy (lexsort + reduceat pattern)

    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터.
    key_cols : list[str]
        그룹 키 컬럼 리스트 (예: [COL_SHIP_TO, COL_ITEM] 또는 [COL_SHIP_TO, COL_ITEM, COL_LOC]).
    aggs : dict[str, str]
        {대상컬럼: 집계함수명} 매핑.
        지원 집계: 'sum','max','min','any','all','first','last','count'
        - 'count' 의 대상컬럼 값은 무시되며 group size 반환.
    cast_key_to_category : bool, default True
        결과의 키 컬럼을 category 로 캐스팅(메모리 절감).
    treat_YN_as_bool : bool, default True
        문자열 'Y'/'N' 컬럼을 bool 처럼 취급하여 any/max, all/min 에서 기대대로 동작.

    Ultra-fast groupby using NumPy (lexsort + reduceat pattern)
    - FIX: reduceat 는 first_idx 가 오름차순이어야 하므로,
           집계는 정렬 인덱스(first_idx_sorted)로 수행하고,
           최종 결과만 '원본 등장 순서'로 재배열.
    Returns
    -------
    pd.DataFrame
        key_cols + list(aggs.keys()) 순서의 DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[*key_cols, *aggs.keys()])    # ── 0) 키 → 카테고리 codes ───────────────────────────────────────────
    key_vals = []
    key_codes = []
    for col in key_cols:
        s = df[col]
        if s.dtype.name != 'category':
            s = s.astype('category')
        key_vals.append(s.to_numpy())                    # 복원용 원본 값
        key_codes.append(s.cat.codes.to_numpy(np.int64)) # -1 포함(NA)

    # ── 1) lexsort 로 정렬 & 그룹 경계 ────────────────────────────────────
    order = np.lexsort(tuple(key_codes[::-1]))
    if order.size == 0:
        return pd.DataFrame(columns=[*key_cols, *aggs.keys()])

    codes_sorted = np.vstack([kc[order] for kc in key_codes]).T  # (N,K)
    change = np.any(np.diff(codes_sorted, axis=0) != 0, axis=1)
    first_idx_sorted = np.concatenate(([0], np.flatnonzero(change) + 1))
    end_idx_sorted   = np.empty_like(first_idx_sorted)
    end_idx_sorted[:-1] = first_idx_sorted[1:]
    end_idx_sorted[-1]  = codes_sorted.shape[0]

    # ★ 결과 표시 순서(원본 등장 순서) 계산 — 집계 인덱스는 건드리지 않음
    rep_rows_sorted = order[first_idx_sorted]                 # 각 그룹 대표행(원본 인덱스)
    rep_order = np.argsort(rep_rows_sorted, kind='mergesort') # 원본 등장 순서로의 재배열 인덱스
    rep_rows  = rep_rows_sorted[rep_order]

    # 결과 키 복원 (최종 출력 순서 = 원본 등장 순서)
    result = pd.DataFrame({col: key_vals[i][rep_rows] for i, col in enumerate(key_cols)})

    # ── 2) dtype 헬퍼 ────────────────────────────────────────────────────
    def _as_ndarray(x):
        return x.to_numpy() if isinstance(x, pd.Series) else np.asarray(x)

    def _is_numeric_dtype(dt) -> bool:
        return (np.issubdtype(dt, np.integer) or
                np.issubdtype(dt, np.unsignedinteger) or
                np.issubdtype(dt, np.floating) or
                np.issubdtype(dt, np.bool_))

    def _is_YN_array(arr: np.ndarray) -> bool:
        if arr.dtype == object or arr.dtype.kind in ('U', 'S'):
            u = pd.unique(arr)
            if u.size == 0:
                return False
            u = u[pd.notna(u)]
            if u.size == 0:
                return False
            return set(map(str, u)).issubset({'Y', 'N'})
        return False

    # ── 3) 집계 계산 (★ reduceat 는 오름차순 first_idx_sorted 사용) ───────
    for tgt_col, how in aggs.items():
        how_l = how.lower()
        col_sorted = _as_ndarray(df[tgt_col])[order]

        restore_YN = False
        if how_l in ('any', 'all', 'max', 'min'):
            if np.issubdtype(col_sorted.dtype, np.bool_):
                arr = col_sorted.view(np.int8)
            elif treat_YN_as_bool and _is_YN_array(col_sorted):
                arr = (col_sorted == 'Y').astype(np.int8, copy=False)
                # 원래 동작 유지: 'max'/'min' 에 한해 Y/N 복구
                restore_YN = (how_l in ('max', 'min'))
            elif _is_numeric_dtype(col_sorted.dtype):
                arr = col_sorted
            else:
                arr = pd.to_numeric(pd.Series(col_sorted), errors='raise').to_numpy()
        elif how_l in ('sum', 'first', 'last', 'count'):
            if how_l == 'sum':
                if np.issubdtype(col_sorted.dtype, np.bool_):
                    arr = col_sorted.view(np.int8)
                elif _is_numeric_dtype(col_sorted.dtype):
                    arr = col_sorted
                else:
                    arr = pd.to_numeric(pd.Series(col_sorted), errors='raise').to_numpy()
            else:
                arr = col_sorted
        else:
            raise ValueError(f"[ultra_fast_groupby_numpy_general] Unsupported agg: {how}")

        # 그룹별 집계 (오름차순 인덱스)
        if how_l == 'sum':
            out_vals_sorted = np.add.reduceat(arr, first_idx_sorted)
        elif how_l in ('any', 'max'):
            out_vals_sorted = np.maximum.reduceat(arr, first_idx_sorted)
        elif how_l in ('all', 'min'):
            out_vals_sorted = np.minimum.reduceat(arr, first_idx_sorted)
        elif how_l == 'first':
            out_vals_sorted = arr[first_idx_sorted]
        elif how_l == 'last':
            out_vals_sorted = arr[end_idx_sorted - 1]
        elif how_l == 'count':
            out_vals_sorted = (end_idx_sorted - first_idx_sorted).astype(np.int64)
        else:
            raise ValueError(f"[ultra_fast_groupby_numpy_general] Unsupported agg: {how}")

        # ★ 최종 출력 순서(원본 등장 순서)로 재배열
        out_vals = out_vals_sorted[rep_order]

        # 'Y'/'N' 복구 (0/1 → 'N'/'Y') — max/min 한정
        if restore_YN:
            out_vals = np.where(out_vals >= 1, 'Y', 'N')

        result[tgt_col] = out_vals

    # ── 4) 키 컬럼 category 캐스팅 (옵션) ────────────────────────────────
    if cast_key_to_category and not result.empty:
        result[key_cols] = result[key_cols].astype('category')

    # 안전장치: reduceat 인덱스 정렬 위반 방지용 (개발 시 검증)
    # assert np.all(first_idx_sorted[:-1] < first_idx_sorted[1:]), "first_idx must be strictly increasing"

    return result

################################################################################################################
#################### Start Step Functions  ##########
################################################################################################################
# To do : 여기 아래에 Step Function 들을 구현.
########################################################################################################################
# Step 1-1 : Sales Product ASN Delta 전처리
########################################################################################################################
@_decoration_
def fn_step01_01_preprocess_sales_product_asn_delta(
        df_asn_delta: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 1-1) Sales Product ASN Delta 전처리
    ----------------------------------------------------------
    • 입력 미존재 시 즉시 Exception 발생(프로그램 종료)
    • Version 컬럼(COL_VERSION) 제거
    • 'Sales Product ASN Delta' → 'Sales Product ASN' 컬럼명 변경
    • 반환 컬럼 순서 : [Ship To, Item, Location, Sales Product ASN]
    • dtype : 모두 category 로 캐스팅
    """    
    # ── 0) 입력 방어 : 공백 데이터면 즉시 종료 ───────────────────────────────
    if df_asn_delta is None or df_asn_delta.empty:
        raise Exception('[Step 1-1] Input 6 (df_in_Sales_Product_ASN_Delta) is empty. Program terminated.')

    # ── 1) 필요한 컬럼 존재 확인 ────────────────────────────────────────────
    REQ_COLS = [
        COL_SHIP_TO,         # 'Sales Domain.[Ship To]'
        COL_ITEM,            # 'Item.[Item]'
        COL_LOCATION,        # 'Location.[Location]'
        COL_SALES_PRODUCT_ASN_DELTA  # 'Sales Product ASN Delta'
    ]
    missing = [c for c in REQ_COLS if c not in df_asn_delta.columns]
    if missing:
        raise KeyError(f"[Step 1-1] Required columns missing in df_asn_delta: {missing}")

    # ── 2) 사본 생성 & 최소 컬럼만 선별 ──────────────────────────────────────
    use_cols = [COL_VERSION] + REQ_COLS if COL_VERSION in df_asn_delta.columns else REQ_COLS
    df = df_asn_delta.loc[:, use_cols].copy(deep=False)

    # ── 3) Version 컬럼 제거 ────────────────────────────────────────────────
    if COL_VERSION in df.columns:
        df.drop(columns=[COL_VERSION], inplace=True)

    # ── 4) 컬럼명 변경 : Delta → ASN ────────────────────────────────────────
    df.rename(columns={COL_SALES_PRODUCT_ASN_DELTA: COL_SALES_PRODUCT_ASN}, inplace=True)

    # ── 5) dtype 캐스팅 (메모리 절감) ───────────────────────────────────────
    #     전부 category 로 맞춰 대용량 처리 대비
    cast_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
    for c in cast_cols:
        # 이미 category 여도 재캐스팅 비용은 작고, object → category 로 바꾸면 메모리 절감 효과 큼
        df[c] = df[c].astype('category')

    # ── 6) 컬럼 순서 정렬 ───────────────────────────────────────────────────
    df = df[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]]

    # ── 7) 원본 즉시 해제(대용량 대비) ──────────────────────────────────────
    del df_asn_delta
    gc.collect()

    return df

########################################################################################################################
# Step 1-2 : Sales Product ASN 전처리  (기본 ASN에서 Delta와 겹치는 (ShipTo, Item, Location) 제거)
########################################################################################################################
@_decoration_
def fn_step01_02_preprocess_sales_product_asn(
        df_asn_base: pd.DataFrame,              # Input 7 : df_in_Sales_Product_ASN (원본)
        df_step01_01_delta: pd.DataFrame        # Step 1-1 결과 : (ShipTo, Item, Location, Sales Product ASN)
) -> pd.DataFrame:
    """
    Step 1-2) Sales Product ASN 전처리
    ----------------------------------------------------------
    • Version 컬럼(COL_VERSION) 삭제 (기본 ASN 표에서만)
    • 기본 ASN 에서 ASN Delta(1-1 결과)와 (ShipTo, Item, Location) 키가 겹치는 행 제거
      - 목적 : 1-3에서 Delta와 합칠 때 Y→N, Y→Y 충돌 제거
    • 반환 스키마 : [Sales Domain.[Ship To], Item.[Item], Location.[Location], Sales Product ASN]
    • 모든 컬럼 dtype : category
    • for-loop 없이 벡터화, 대용량/메모리 고려
    """    
    # ── 0) 입력 방어 ────────────────────────────────────────────────────────
    if df_asn_base is None or df_asn_base.empty:
        raise Exception('[Step 1-2] Input 7 (df_in_Sales_Product_ASN) is empty.')
    if df_step01_01_delta is None or df_step01_01_delta.empty:
        # 1-1에서 이미 빈 값이면 종료하도록 했지만, 재확인
        raise Exception('[Step 1-2] Step 1-1 result is empty. Program terminated.')

    # ── 1) 필수 컬럼 확인 ───────────────────────────────────────────────────
    REQ_BASE = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
    missing_base = [c for c in REQ_BASE if c not in df_asn_base.columns and c != COL_VERSION]
    if missing_base:
        raise KeyError(f"[Step 1-2] Required columns missing in df_asn_base: {missing_base}")

    REQ_DELTA_KEYS = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    missing_delta = [c for c in REQ_DELTA_KEYS if c not in df_step01_01_delta.columns]
    if missing_delta:
        raise KeyError(f"[Step 1-2] Required key columns missing in Step 1-1 result: {missing_delta}")

    # ── 2) 기본 ASN 사본 생성 및 Version 제거 ───────────────────────────────
    use_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
    # Version 컬럼이 있으면 드롭
    if COL_VERSION in df_asn_base.columns:
        df_asn = df_asn_base.drop(columns=[COL_VERSION]).loc[:, use_cols].copy(deep=False)
    else:
        df_asn = df_asn_base.loc[:, use_cols].copy(deep=False)

    # ── 3) (Item) 키로 Filter ) ─
    # 카테고리 간 카테고리-세트 불일치로 인한 merge 비용/오류를 피하기 위해
    # 문자열 키로 벡터화 비교(mask) 수행 (Delta 쪽 고유키 set만 생성 → 메모리 최소화)
    # left key
    k2 = df_asn[COL_ITEM].astype('object').astype(str)
    key_left = k2

    # right unique key set
    r2 = df_step01_01_delta[COL_ITEM].astype('object').astype(str)
    right_keys = (r2).unique()
    right_key_set = set(right_keys)   # Delta는 상대적으로 작다는 전제에서 set 사용

    # 조인 마스크
    mask_keep = key_left.isin(right_key_set)
    df_asn = df_asn.loc[mask_keep, use_cols].copy(deep=False)


    # ── 3) (ShipTo, Item, Location) 키로 안티조인 (Delta와 겹치는 기본 ASN 제거) ─
    # 카테고리 간 카테고리-세트 불일치로 인한 merge 비용/오류를 피하기 위해
    # 문자열 키로 벡터화 비교(mask) 수행 (Delta 쪽 고유키 set만 생성 → 메모리 최소화)
    # left key
    k1 = df_asn[COL_SHIP_TO].astype('object').astype(str)
    k2 = df_asn[COL_ITEM].astype('object').astype(str)
    k3 = df_asn[COL_LOCATION].astype('object').astype(str)
    key_left = k1 + '|' + k2 + '|' + k3

    # right unique key set
    r1 = df_step01_01_delta[COL_SHIP_TO].astype('object').astype(str)
    r2 = df_step01_01_delta[COL_ITEM].astype('object').astype(str)
    r3 = df_step01_01_delta[COL_LOCATION].astype('object').astype(str)
    right_keys = (r1 + '|' + r2 + '|' + r3).unique()
    right_key_set = set(right_keys)   # Delta는 상대적으로 작다는 전제에서 set 사용

    # 안티조인 마스크
    mask_keep = ~key_left.isin(right_key_set)
    df_out = df_asn.loc[mask_keep, use_cols].copy(deep=False)

    # ── 4) dtype 카테고리 캐스팅 ────────────────────────────────────────────
    for c in use_cols:
        df_out[c] = df_out[c].astype('category')

    # ── 5) 중간 변수 및 원본 참조 해제 ───────────────────────────────────────
    del (df_asn_base, df_step01_01_delta, df_asn,
         k1, k2, k3, key_left, r1, r2, r3, right_keys, right_key_set)
    gc.collect()

    return df_out


########################################################################################################################
# Step 1-3 : Sales Product ASN 구성 (ShipTo × Item × Location)
########################################################################################################################
@_decoration_
def fn_step01_03_build_sales_product_asn(
        df_step01_01_delta: pd.DataFrame,          # Step 1-1 결과 : [ShipTo, Item, Location, Sales Product ASN]
        df_step01_02_base_filtered: pd.DataFrame   # Step 1-2 결과 : [ShipTo, Item, Location, Sales Product ASN]
) -> pd.DataFrame:
    """
    Step 1-3) Sales Product ASN 구성 (ShipTo × Item × Location)
    ----------------------------------------------------------
    • 입력
        - df_step01_01_delta        : 1-1 결과 (Delta 전처리, 'Sales Product ASN' 으로 정규화, Y/N 포함)
        - df_step01_02_base_filtered: 1-2 결과 (기본 ASN 중 Delta와 겹치지 않는 행만 남김, 주로 Y)
    • 처리
        - 두 DataFrame을 같은 스키마로 정렬하여 유니온(concat)
        - (안전) 중복 키(ShipTo,Item,Location) 제거 (원칙적으로 1-2에서 제거되어 없어야 함)
        - 모든 컬럼을 category로 캐스팅
    • 반환 스키마(Version 없이)
        [Sales Domain.[Ship To], Item.[Item], Location.[Location], Sales Product ASN]
    • 성능/메모리
        - for-loop 없음, concat + drop_duplicates 사용
        - concat 중 카테고리 범주가 다르면 일시적으로 object로 오를 수 있으므로, 마지막에 재-cast(category)
    """    
    
    # ── 0) 방어 코드 ───────────────────────────────────────────────────────
    if df_step01_01_delta is None or df_step01_01_delta.empty:
        # 설계상 1-1에서 빈 경우 프로그램 종료이므로 여기서도 방어적으로 예외 처리
        raise Exception('[Step 1-3] Step 1-1 result (ASN Delta) is empty. Program terminated.')

    # 1-2 결과는 비어 있어도 유니온의 의미상 문제 없음(전부 Delta로만 구성)
    if df_step01_02_base_filtered is None:
        df_step01_02_base_filtered = pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN])

    # ── 1) 공통 스키마 정렬 ────────────────────────────────────────────────
    USE_COLS = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]

    missing_11 = [c for c in USE_COLS if c not in df_step01_01_delta.columns]
    if missing_11:
        raise KeyError(f"[Step 1-3] Step 1-1 result missing columns: {missing_11}")

    missing_12 = [c for c in USE_COLS if c not in df_step01_02_base_filtered.columns]
    if missing_12:
        raise KeyError(f"[Step 1-3] Step 1-2 result missing columns: {missing_12}")

    df_delta = df_step01_01_delta.loc[:, USE_COLS].copy(deep=False)
    df_base  = df_step01_02_base_filtered.loc[:, USE_COLS].copy(deep=False)

    # ── 2) 유니온(행 결합) ────────────────────────────────────────────────
    #  - 서로 다른 카테고리 범주가 있으면 일시적으로 object로 승격될 수 있음 → 아래에서 재-cast
    df_union = pd.concat([df_delta, df_base], axis=0, ignore_index=True, copy=False)

    # ── 3) (안전) 중복 키 제거 ─────────────────────────────────────────────
    #  원칙적으로 1-2에서 Delta와 겹치는 키를 제거했으므로 중복이 없어야 함.
    #  혹시 모를 중복 대비로 subset 키 기준 고유화.
    df_union.drop_duplicates(subset=[COL_SHIP_TO, COL_ITEM, COL_LOCATION], keep='last', inplace=True)

    # ── 4) 값 정규화 & dtype 캐스팅 ───────────────────────────────────────
    #  'Sales Product ASN' 값은 {Y,N}만 허용(스펙). 혹시 다른 값이 있으면 문자열로 변환 후 카테고리화.
    if df_union[COL_SALES_PRODUCT_ASN].dtype.name != 'category':
        df_union[COL_SALES_PRODUCT_ASN] = df_union[COL_SALES_PRODUCT_ASN].astype('object').astype(str)
    # 최종 컬럼을 category로 캐스팅
    for c in USE_COLS:
        df_union[c] = df_union[c].astype('category')

    # ── 5) 메모리 정리 ────────────────────────────────────────────────────
    del (df_step01_01_delta, df_step01_02_base_filtered, df_delta, df_base)
    gc.collect()

    return df_union


########################################################################################################################
# Step 1-4 : Sales Product ASN output  (from Input 6: df_in_Sales_Product_ASN_Delta)
# 기존 1-6
########################################################################################################################
@_decoration_
def fn_step01_04_output_sales_product_asn(
        df_in_asn_delta: pd.DataFrame,   # Input 6 : [Version, ShipTo, Item, Location, Sales Product ASN Delta (Y/N)]
        version: str                     # 예: 'CWV_DP'
) -> pd.DataFrame:
    """
    Step 1-6) Sales Product ASN output
    ----------------------------------------------------------
    • 입력  : df_in_Sales_Product_ASN_Delta (Input 6)
      - 컬럼 : [Version, ShipTo, Item, Location, Sales Product ASN Delta]
    • 처리  :
      - 컬럼명 변경 : 'Sales Product ASN Delta' → 'Sales Product ASN'
      - 값 변환     : 'N' → ''(빈값), 그 외는 'Y'만 유지
      - Version 값  : 함수 인자(version)로 덮어씀
      - 출력 스키마 : [Version, ShipTo, Item, Location, Sales Product ASN]
      - dtype       : category
    • 반환  : Output_Sales_Product_ASN 스키마의 DataFrame
    """
    # ── 0) 방어 코드 ───────────────────────────────────────────────────────
    REQ_COLS = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN_DELTA]
    missing = [c for c in REQ_COLS if c not in df_in_asn_delta.columns]
    if missing:
        raise KeyError(f"[Step 1-6] Input 6 missing columns: {missing}")    
    
    if df_in_asn_delta.empty:
        cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
        return pd.DataFrame(columns=cols)

    # ── 1) 필요한 컬럼만 복사 ──────────────────────────────────────────────
    df = df_in_asn_delta.loc[:, REQ_COLS].copy(deep=False)

    # ── 2) 컬럼명 변경 : Delta → 표준 컬럼 ─────────────────────────────────
    df.rename(columns={COL_SALES_PRODUCT_ASN_DELTA: COL_SALES_PRODUCT_ASN}, inplace=True)

    # ── 3) 값 변환 : 'Y' 유지, 나머지는 ''(빈값) ───────────────────────────
    norm = (
        df[COL_SALES_PRODUCT_ASN]
        .astype('object').astype(str).str.strip().str.upper()
    )
    df[COL_SALES_PRODUCT_ASN] = norm.where(norm.eq('Y'), '')

    # ── 4) Version 덮어쓰기 & 출력 스키마 정렬 ─────────────────────────────
    df[COL_VERSION] = version
    df_out = df[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]]

    # ── 5) dtype 정리 (메모리 절감: category) ─────────────────────────────
    for c in [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]:
        df_out[c] = df_out[c].astype('category')

    # ── 6) 메모리 정리 ────────────────────────────────────────────────────
    del df_in_asn_delta, df, norm
    gc.collect()

    return df_out

########################################################################################################################
# Step 1-5 : Sales Product ASN Delta output  (from Input 6: df_in_Sales_Product_ASN_Delta)
# 기존 1-7
########################################################################################################################
@_decoration_
def fn_step01_05_output_sales_product_asn_delta(
        df_in_asn_delta: pd.DataFrame,   # Input 6 : [Version, ShipTo, Item, Location, Sales Product ASN Delta (Y/N)]
        version: str                     # 예: 'CWV_DP'
) -> pd.DataFrame:
    """
    Step 1-7) Sales Product ASN Delta output
    ----------------------------------------------------------
    • 입력  : df_in_Sales_Product_ASN_Delta (Input 6)
      - 컬럼 : [Version, ShipTo, Item, Location, Sales Product ASN Delta]
    • 처리  :
      - 값 변환     : 'Sales Product ASN Delta' 컬럼을 전체 ''(빈값)로 설정
      - Version 값  : 함수 인자(version)로 덮어씀
      - 출력 스키마 : [Version, ShipTo, Item, Location, Sales Product ASN Delta]
      - dtype       : category
    • 반환  : Output_Sales_Product_ASN_Delta 스키마의 DataFrame
    """
    # ── 0) 방어 코드 ───────────────────────────────────────────────────────
    REQ_COLS = [ COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN_DELTA]
    missing = [c for c in REQ_COLS if c not in df_in_asn_delta.columns]
    if missing:
        raise KeyError(f"[Step 1-7] Input 6 missing columns: {missing}")    
        
    if df_in_asn_delta.empty:
        # 빈 DF를 지정 스키마로 반환
        df_empty = pd.DataFrame(columns=REQ_COLS)
        for c in REQ_COLS:
            df_empty[c] = df_empty[c].astype('category')
        return df_empty

    # ── 1) 필요한 컬럼만 복사 ──────────────────────────────────────────────
    df = df_in_asn_delta.loc[:, REQ_COLS].copy(deep=False)

    # ── 2) 값 변환 : 전체 ''(빈값)으로 설정 ────────────────────────────────
    #  ※ 스펙 '전체 Null 적용'을 카테고리 호환을 위해 ''로 대체(이후 시스템에서 Null로 처리될 수 있음)
    df[COL_SALES_PRODUCT_ASN_DELTA] = ''

    # ── 3) Version 덮어쓰기 & 출력 스키마 정렬 ─────────────────────────────
    df[COL_VERSION] = version
    df_out = df[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN_DELTA]]

    # ── 4) dtype 정리 (메모리 절감: category) ─────────────────────────────
    for c in [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN_DELTA]:
        df_out[c] = df_out[c].astype('category')

    # ── 5) 메모리 정리 ────────────────────────────────────────────────────
    del df_in_asn_delta, df
    gc.collect()

    return df_out


########################################################################################################################
# Step 2-1) Assortment 전처리
########################################################################################################################
@_decoration_
def fn_step02_01_preprocess_assortment(
        df_in_sin_assort: pd.DataFrame,          # Input 8 : df_in_SIn_Assortment
        df_in_sout_assort: pd.DataFrame,         # Input 9 : df_in_SOut_Assortment
        df_step01_01_asn_delta: pd.DataFrame     # Step 1-1 결과 : [ShipTo, Item, Location, Sales Product ASN]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 2-1) Assortment 전처리
    ----------------------------------------------------------
    • 필터 기준                          : Step 1-1 결과에 존재하는 Item만 사용 (ShipTo/Loc 무관, Item 단독 기준)
    • 공통 처리                          :
        - Version.[Version Name] 컬럼 삭제
        - 필요한 컬럼만 유지 (ShipTo, Item, Location, 각 Assortment Measure 4종)
        - 차원 컬럼(dtype) 최소화 (category 캐스팅)
    • 반환                               : (S/In 전처리 DF, S/Out 전처리 DF)
    • 스키마
        - S/In : [ShipTo, Item, Location, S/In FCST(GI) Assortment_AP1, _AP2, _GC, _Local]
        - S/Out: [ShipTo, Item, Location, S/Out FCST Assortment_AP1, _AP2, _GC, _Local]
    """    
    # ── 0) 방어 코드 ───────────────────────────────────────────────────────
    if df_step01_01_asn_delta is None or df_step01_01_asn_delta.empty:
        # 1-1 단계에서 이미 종료하도록 되어 있으나, 안전상 빈 DF 반환
        empty_si = pd.DataFrame(columns=[
            COL_SHIP_TO, COL_ITEM, COL_LOCATION,
            COL_SIN_ASSORT_AP1, COL_SIN_ASSORT_AP2, COL_SIN_ASSORT_GC, COL_SIN_ASSORT_LOCAL
        ])
        empty_so = pd.DataFrame(columns=[
            COL_SHIP_TO, COL_ITEM, COL_LOCATION,
            COL_SOUT_ASSORT_AP1, COL_SOUT_ASSORT_AP2, COL_SOUT_ASSORT_GC, COL_SOUT_ASSORT_LOCAL
        ])
        for c in [COL_SHIP_TO, COL_ITEM, COL_LOCATION]:
            empty_si[c] = empty_si[c].astype('category')
            empty_so[c] = empty_so[c].astype('category')
        return empty_si, empty_so

    # Step 1-1 에서 존재하는 Item 목록 (Item 기준 필터)
    items_from_delta = pd.Index(df_step01_01_asn_delta[COL_ITEM].astype('object'))

    # 각 테이블에서 반드시 있어야 할 컬럼
    NEED_DIM = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    NEED_SIN_MEAS  = [COL_SIN_ASSORT_AP1,  COL_SIN_ASSORT_AP2,  COL_SIN_ASSORT_GC,  COL_SIN_ASSORT_LOCAL]
    NEED_SOUT_MEAS = [COL_SOUT_ASSORT_AP1, COL_SOUT_ASSORT_AP2, COL_SOUT_ASSORT_GC, COL_SOUT_ASSORT_LOCAL]

    # 컬럼 체크 유틸
    def _check_cols(df: pd.DataFrame, need_cols: list, tag: str):
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise KeyError(f"[Step 2-1] {tag} missing columns: {missing}")

    _check_cols(df_in_sin_assort,  NEED_DIM + NEED_SIN_MEAS,  "S/In Assortment")
    _check_cols(df_in_sout_assort, NEED_DIM + NEED_SOUT_MEAS, "S/Out Assortment")

    # 공통 전처리 유틸
    def _prep(df_src: pd.DataFrame, meas_cols: list, tag: str) -> pd.DataFrame:
        use_cols = NEED_DIM + meas_cols
        # 필요한 컬럼만 슬라이싱 (+Version은 있으면 제거)
        df = df_src.loc[:, ([COL_VERSION] if COL_VERSION in df_src.columns else []) + use_cols].copy(deep=False)

        # Version 삭제
        if COL_VERSION in df.columns:
            df.drop(columns=[COL_VERSION], inplace=True)

        # Item 기준 필터링 (벡터연산)
        df = df[df[COL_ITEM].isin(items_from_delta)]

        # 차원 컬럼 category 캐스팅 (메모리 절감)
        for c in NEED_DIM:
            df[c] = df[c].astype('category')

        return df

    # ── 1) S/In 전처리 ───────────────────────────────────────────────────
    df_si = _prep(df_in_sin_assort, NEED_SIN_MEAS,  "S/In")

    # ── 2) S/Out 전처리 ──────────────────────────────────────────────────
    df_so = _prep(df_in_sout_assort, NEED_SOUT_MEAS, "S/Out")

    # ── 3) 메모리 정리 ───────────────────────────────────────────────────
    del df_in_sin_assort, df_in_sout_assort, df_step01_01_asn_delta
    gc.collect()

    return df_si, df_so


########################################################################################################################
# 🔧 Alias 상수  (Step 2-2에서 참조되는 미정의 상수 보완)
########################################################################################################################
# S/In Assortment 컬럼
COL_SIN_ASS_GC     = COL_SIN_ASSORT_GC
COL_SIN_ASS_AP2    = COL_SIN_ASSORT_AP2
COL_SIN_ASS_AP1    = COL_SIN_ASSORT_AP1
COL_SIN_ASS_LOCAL  = COL_SIN_ASSORT_LOCAL

# S/Out Assortment 컬럼
COL_SOUT_ASS_GC    = COL_SOUT_ASSORT_GC
COL_SOUT_ASS_AP2   = COL_SOUT_ASSORT_AP2
COL_SOUT_ASS_AP1   = COL_SOUT_ASSORT_AP1
COL_SOUT_ASS_LOCAL = COL_SOUT_ASSORT_LOCAL

# 반환 dict용 키 (Step 2 계열은 OUT_* 이름을 그대로 키로 사용해 충돌 방지)
STR_DF_OUT_SIN_GC     = OUT_SIN_ASSORTMENT_GC
STR_DF_OUT_SIN_AP2    = OUT_SIN_ASSORTMENT_AP2
STR_DF_OUT_SIN_AP1    = OUT_SIN_ASSORTMENT_AP1
STR_DF_OUT_SIN_LOCAL  = OUT_SIN_ASSORTMENT_LOCAL
STR_DF_OUT_SOUT_GC    = OUT_SOUT_ASSORTMENT_GC
STR_DF_OUT_SOUT_AP2   = OUT_SOUT_ASSORTMENT_AP2
STR_DF_OUT_SOUT_AP1   = OUT_SOUT_ASSORTMENT_AP1
STR_DF_OUT_SOUT_LOCAL = OUT_SOUT_ASSORTMENT_LOCAL

########################################################################################################################
# Step 2-2) Sales Product ASN(1-3) 로 Assortment Measure 구성 (GC/AP2/AP1/Local)
#  - Vectorised 방식 (fn_step05_make_actual_and_inv_fcst_level 참조)
########################################################################################################################
@_decoration_
def fn_step02_02_build_assortments(
        df_step01_03_asn_all      : pd.DataFrame,  # [ShipTo, Item, Location, Sales Product ASN (Y/N)]
        df_in_Forecast_Rule       : pd.DataFrame,  # [PG, ShipTo, RULE_GC/AP2/AP1/AP0, ISVALID]
        df_in_Item_Master         : pd.DataFrame,  # [Item, Item Std1]  ※ Item Std1 ↔ Product Group 매칭
        df_in_Sales_Domain_Dimension: pd.DataFrame,# [ShipTo, Sales Std1~6]
        df_in_Sales_Domain_Estore : pd.DataFrame   # [ShipTo]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    반환 : (sin_dict, sout_dict)
      • sin_dict : { OUT_SIn_Assortment_GC/AP2/AP1/Local : DF }
      • sout_dict: { OUT_SOut_Assortment_GC/AP2/AP1/Local: DF }  (기본 '-' + E-Store 상세 결합)
    컬럼 규칙
      • S/In : [ShipTo, Item, Location, Sales Product ASN, S/In FCST(GI) Assortment_XX]
      • S/Out: [ShipTo, Item, Location, Sales Product ASN, S/Out FCST Assortment_XX]
      • ASN='N' → Assortment_XX = NaN
    """    
    # ────────────────────────────────────────────────────────────────────────
    # 0) 방어 및 준비
    # ────────────────────────────────────────────────────────────────────────
    need_asn_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
    miss = [c for c in need_asn_cols if c not in df_step01_03_asn_all.columns]
    if miss:
        raise KeyError(f"[Step 2-2] Step 1-3 result missing columns: {miss}")

    # Forecast-Rule 필수
    need_rule = [COL_PG, COL_SHIP_TO, COL_FRULE_GC_FCST, COL_FRULE_AP2_FCST,
                 COL_FRULE_AP1_FCST, COL_FRULE_AP0_FCST]
    miss = [c for c in need_rule if c not in df_in_Forecast_Rule.columns]
    if miss:
        raise KeyError(f"[Step 2-2] Forecast Rule missing columns: {miss}")

    # Sales-Domain Dimension 필수
    need_dim = [COL_SHIP_TO, COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6]
    miss = [c for c in need_dim if c not in df_in_Sales_Domain_Dimension.columns]
    if miss:
        raise KeyError(f"[Step 2-2] Sales-Domain Dimension missing columns: {miss}")

    # Item Master 필수
    need_im = [
        COL_ITEM 
        ,COL_ITEM_STD1
    ]
    miss = [c for c in need_im if c not in df_in_Item_Master.columns]
    if miss:
        raise KeyError(f"[Step 2-2] Item Master missing columns: {miss}")

    # # E-Store 목록
    # if COL_SHIP_TO not in df_in_Sales_Domain_Estore.columns:
    #     raise KeyError(f"[Step 2-2] E-Store missing column: {COL_SHIP_TO}")

    # ────────────────────────────────────────────────────────────────────────
    # 1) ASN + PG 매핑 (Item.Std1 ↔ ForecastRule.PG)
    # ────────────────────────────────────────────────────────────────────────
    df_asn = df_step01_03_asn_all[need_asn_cols].copy(deep=False)
    pg_map = df_in_Item_Master.set_index(COL_ITEM)[COL_ITEM_STD1].astype(str).to_dict()
    df_asn[COL_PG] = df_asn[COL_ITEM].map(pg_map)

    # 유효 PG 만 남김
    df_asn = df_asn[~df_asn[COL_PG].isna()].copy()
    for c in [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PG, COL_SALES_PRODUCT_ASN]:
        df_asn[c] = df_asn[c].astype('category')

    # ────────────────────────────────────────────────────────────────────────
    # 2) RULE / DIM vector 준비 (fn_step05_make_actual_and_inv_fcst_level 방식)
    # ────────────────────────────────────────────────────────────────────────
    # Rule 유효값만
    df_rule = df_in_Forecast_Rule.copy()
    if COL_FRULE_ISVALID in df_rule.columns:
        df_rule = df_rule[df_rule[COL_FRULE_ISVALID].astype(str).str.upper().eq('Y')].copy()

    for col in (COL_FRULE_GC_FCST, COL_FRULE_AP2_FCST, COL_FRULE_AP1_FCST, COL_FRULE_AP0_FCST):
        df_rule[col] = df_rule[col].fillna(0).astype('int8')

    RULE: dict[tuple[str, str], tuple[int, int, int, int]] = {
        (str(pg), str(ship)): (gc, ap2, ap1, ap0)
        for pg, ship, gc, ap2, ap1, ap0 in zip(
            df_rule[COL_PG].astype(str),
            df_rule[COL_SHIP_TO].astype(str),
            df_rule[COL_FRULE_GC_FCST],
            df_rule[COL_FRULE_AP2_FCST],
            df_rule[COL_FRULE_AP1_FCST],
            df_rule[COL_FRULE_AP0_FCST],
        )
    }

    dim = df_in_Sales_Domain_Dimension.copy()
    dim_idx = dim.set_index(COL_SHIP_TO)
    LV_MAP = {
        lv: dim_idx[col].astype(str).to_dict()
        for lv, col in enumerate([COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6], start=1)
    }

    # ship_level : Ship코드가 처음 등장한(가장 상위) 레벨 (Lv-2 … Lv-7 ↔ Std1 … Std6)
    ship_level: dict[str, int] = {}
    for lv, col in enumerate([COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6], start=2):
        for code in dim[col].dropna().astype(str).unique():
            ship_level.setdefault(code, lv)

    def parent_of(arr: np.ndarray, lv: int) -> np.ndarray:
        return np.vectorize(LV_MAP[lv].get, otypes=[object])(arr)

    # ────────────────────────────────────────────────────────────────────────
    # 3) Core: vectorised 룰 매칭 → 4개 Tag buffer 생성 (S/In용)
    # ────────────────────────────────────────────────────────────────────────
    ship_np = df_asn[COL_SHIP_TO].astype(str).to_numpy()
    item_np = df_asn[COL_ITEM].to_numpy()
    loc_np  = df_asn[COL_LOCATION].to_numpy()
    pg_np   = df_asn[COL_PG].astype(str).to_numpy()
    asn_np  = df_asn[COL_SALES_PRODUCT_ASN].astype(str).str.upper().to_numpy()

    TAGS = ('GC', 'AP2', 'AP1', 'Local')
    tag_lvmat = {t: np.zeros_like(ship_np, dtype='int8') for t in TAGS}

    for lv in range(1, 7):  # Std1..Std6
        parent = parent_of(ship_np, lv)              # Ship → Std-lv 코드
        mask = parent != None
        if not mask.any():
            continue

        idx = np.flatnonzero(mask)
        rule_vec = np.array(
            [RULE.get((pg_np[i], parent[i]), (0, 0, 0, 0)) for i in idx],
            dtype='int8'
        )
        gc_lv, ap2_lv, ap1_lv, ap0_lv = rule_vec.T
        tag_lvmat['GC']   [idx] = np.where(gc_lv  != 0, gc_lv,  tag_lvmat['GC']   [idx])
        tag_lvmat['AP2']  [idx] = np.where(ap2_lv != 0, ap2_lv, tag_lvmat['AP2']  [idx])
        tag_lvmat['AP1']  [idx] = np.where(ap1_lv != 0, ap1_lv, tag_lvmat['AP1']  [idx])
        tag_lvmat['Local'][idx] = np.where(ap0_lv != 0, ap0_lv, tag_lvmat['Local'][idx])

    ship_lv_arr = np.fromiter((ship_level.get(s, 99) for s in ship_np), dtype='int8')

    def _build_sin(tag: str, qty_col: str) -> pd.DataFrame:
        lv_arr = tag_lvmat[tag]
        valid  = (lv_arr >= 2) & (lv_arr <= 7) & (lv_arr <= ship_lv_arr)
        if not valid.any():
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])

        tgt_ship = np.fromiter(
            (LV_MAP[l-1].get(s) if 2 <= l <= 7 else None
             for s, l in zip(ship_np[valid], lv_arr[valid])),
            dtype=object
        )
        ok = tgt_ship != None
        if not ok.any():
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])

        rows = {
            COL_SHIP_TO           : tgt_ship[ok],
            COL_ITEM              : item_np[valid][ok],
            COL_LOCATION          : loc_np [valid][ok],
            COL_SALES_PRODUCT_ASN : asn_np [valid][ok],
            qty_col               : np.ones(ok.sum(), dtype='int8')
        }
        df = pd.DataFrame(rows)

        df = ultra_fast_groupby_numpy_general(
            df=df,
            key_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION],
            aggs={COL_SALES_PRODUCT_ASN: 'max'}
        )
        df = df.assign(**{qty_col: 1})


        # ASN='N' → measure NaN
        df[qty_col] = np.where(df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y'), 1, np.nan)

        # dtype 최적화
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')
        return df

    sin_gc    = _build_sin('GC',    COL_SIN_ASS_GC)
    sin_ap2   = _build_sin('AP2',   COL_SIN_ASS_AP2)
    sin_ap1   = _build_sin('AP1',   COL_SIN_ASS_AP1)
    sin_local = _build_sin('Local', COL_SIN_ASS_LOCAL)

    sin_dict: dict[str, pd.DataFrame] = {
        STR_DF_OUT_SIN_GC    : sin_gc,
        STR_DF_OUT_SIN_AP2   : sin_ap2,
        STR_DF_OUT_SIN_AP1   : sin_ap1,
        STR_DF_OUT_SIN_LOCAL : sin_local,
    }

    # ────────────────────────────────────────────────────────────────────────
    # 4) Core: vectorised 룰 매칭 → 4개 Tag buffer 생성 (S/Out용)
    # ────────────────────────────────────────────────────────────────────────
    def _to_sout_base(df_in: pd.DataFrame, sin_col: str, sout_col: str) -> pd.DataFrame:
        if df_in.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                        COL_SALES_PRODUCT_ASN, sout_col])    
        # 보조 플래그: Y→1, 그 외→0
        df_tmp = df_in[[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN, sin_col]].copy()
        df_tmp['_asn_flag'] = (
            df_tmp[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')
        ).astype('int8')

        # ★ observed=True 로 '실제로 존재하는 그룹'만 집계 (카테고리 데카르트 곱 방지)
        df = ultra_fast_groupby_numpy_general(
            df=df_tmp,
            key_cols=[COL_SHIP_TO, COL_ITEM],
            aggs={sin_col: 'sum', '_asn_flag': 'max'}
        )

        # ASN 복원 및 수량 처리
        df[COL_SALES_PRODUCT_ASN] = np.where(df['_asn_flag'] == 1, 'Y', 'N')
        df.drop(columns=['_asn_flag'], inplace=True)

        df[sout_col] = 1
        df[COL_LOCATION] = '-'
        df[sout_col] = np.where(
            df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y'), 1, np.nan
        )

        # dtype 정리
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')

        return df[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN, sout_col]]  

    sout_base_gc    = _to_sout_base(sin_gc,    COL_SIN_ASS_GC,    COL_SOUT_ASS_GC)
    sout_base_ap2   = _to_sout_base(sin_ap2,   COL_SIN_ASS_AP2,   COL_SOUT_ASS_AP2)
    sout_base_ap1   = _to_sout_base(sin_ap1,   COL_SIN_ASS_AP1,   COL_SOUT_ASS_AP1)
    sout_base_local = _to_sout_base(sin_local, COL_SIN_ASS_LOCAL, COL_SOUT_ASS_LOCAL)

    # ────────────────────────────────────────────────────────────────────────
    # 5) S/Out – E-Store 상세(Location 유지) : df_asn 중 E-Store ShipTo만 재빌드
    # ────────────────────────────────────────────────────────────────────────
    est_set = set(df_in_Sales_Domain_Estore[COL_SHIP_TO].astype(str))
    if est_set:
        mask_es = df_asn[COL_SHIP_TO].astype(str).isin(est_set).to_numpy()
    else:
        mask_es = np.zeros(len(df_asn), dtype=bool)

    def _build_sout_est(tag: str, qty_col: str) -> pd.DataFrame:
        if not mask_es.any():
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])

        lv_arr = tag_lvmat[tag][mask_es]
        ship_v = ship_np[mask_es]; item_v = item_np[mask_es]
        loc_v  = loc_np [mask_es]; asn_v  = asn_np [mask_es]

        ship_lv_v = np.fromiter((ship_level.get(s, 99) for s in ship_v), dtype='int8')
        valid = (lv_arr >= 2) & (lv_arr <= 7) & (lv_arr <= ship_lv_v)
        if not valid.any():
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])

        tgt_ship = np.fromiter(
            (LV_MAP[l-1].get(s) if 2 <= l <= 7 else None
             for s, l in zip(ship_v[valid], lv_arr[valid])),
            dtype=object
        )
        ok = tgt_ship != None
        if not ok.any():
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])

        rows = {
            COL_SHIP_TO           : tgt_ship[ok],
            COL_ITEM              : item_v[valid][ok],
            COL_LOCATION          : loc_v [valid][ok],
            COL_SALES_PRODUCT_ASN : asn_v [valid][ok],
            qty_col               : np.ones(ok.sum(), dtype='int8')
        }
        df = pd.DataFrame(rows)        
        df = ultra_fast_groupby_numpy_general(
            df=df,
            key_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION],
            aggs={COL_SALES_PRODUCT_ASN: 'max'}
        )
        df = df.assign(**{qty_col: 1})
        

        df[qty_col] = np.where(df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y'), 1, np.nan)
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')
        return df

    sout_es_gc    = _build_sout_est('GC',    COL_SOUT_ASS_GC)
    sout_es_ap2   = _build_sout_est('AP2',   COL_SOUT_ASS_AP2)
    sout_es_ap1   = _build_sout_est('AP1',   COL_SOUT_ASS_AP1)
    sout_es_local = _build_sout_est('Local', COL_SOUT_ASS_LOCAL)

    # ────────────────────────────────────────────────────────────────────────
    # 6) S/Out 최종: 기본('-') + E-Store 상세 concat (중복 키 제거)
    # ────────────────────────────────────────────────────────────────────────
    def _concat_sout(base: pd.DataFrame, est: pd.DataFrame, qty_col: str) -> pd.DataFrame:
        if base.empty and est.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])
        gcols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
        df = pd.concat([base, est], ignore_index=True)
        df = df.drop_duplicates(subset=gcols, keep='first')
        # 보수적으로 다시 ASN 기준 NaN 적용
        df[qty_col] = np.where(df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y'), 1, np.nan)
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')
        return df

    sout_gc    = _concat_sout(sout_base_gc,    sout_es_gc,    COL_SOUT_ASS_GC)
    sout_ap2   = _concat_sout(sout_base_ap2,   sout_es_ap2,   COL_SOUT_ASS_AP2)
    sout_ap1   = _concat_sout(sout_base_ap1,   sout_es_ap1,   COL_SOUT_ASS_AP1)
    sout_local = _concat_sout(sout_base_local, sout_es_local, COL_SOUT_ASS_LOCAL)

    sout_dict: dict[str, pd.DataFrame] = {
        STR_DF_OUT_SOUT_GC    : sout_gc,
        STR_DF_OUT_SOUT_AP2   : sout_ap2,
        STR_DF_OUT_SOUT_AP1   : sout_ap1,
        STR_DF_OUT_SOUT_LOCAL : sout_local,
    }

    # ────────────────────────────────────────────────────────────────────────
    # 7) 메모리 정리
    # ────────────────────────────────────────────────────────────────────────
    del (df_step01_03_asn_all, df_in_Forecast_Rule, df_in_Item_Master,
         df_in_Sales_Domain_Dimension, df_in_Sales_Domain_Estore,
         df_asn, df_rule, dim, dim_idx,
         ship_np, item_np, loc_np, pg_np, asn_np,
         sin_gc, sin_ap2, sin_ap1, sin_local,
         sout_base_gc, sout_base_ap2, sout_base_ap1, sout_base_local,
         sout_es_gc, sout_es_ap2, sout_es_ap1, sout_es_local)
    gc.collect()

    return sin_dict, sout_dict

########################################################################################################################
# Step 2-3 : S/In Assortment Measure (GC/AP2/AP1/Local) 비교
########################################################################################################################
@_decoration_
def fn_step02_03_compare_sin_assortments(
        df_in_sin_assort: pd.DataFrame,                # Step 2-1 결과: S/In Assortment (Input 8 전처리본)
        sin_dict: dict[str, pd.DataFrame]              # Step 2-2 결과: {STR_DF_OUT_SIN_XXX: df, ...}
) -> dict[str, pd.DataFrame]:
    """
    Step 2-3) S/In Assortment Measure (GC/AP2/AP1/Local) 비교
    ----------------------------------------------------------------
    • 입력
      - df_in_sin_assort : (ShipTo, Item, Location, S/In FCST(GI) Assortment_[GC|AP2|AP1|Local])
      - sin_dict         : Step2-2에서 생성된 4개 S/In dict
                           (각 DF는 [ShipTo, Item, Location, Sales Product ASN, Assortment_xx] 스키마)
    • 처리
      - 키(ShipTo, Item, Location) 기준으로 df_in_sin_assort에 **이미 값이 있는 행**은
        ASN='Y' 라면 삭제(중복 방지)
      - ASN='N' 인 행은 해당 Assortment 컬럼을 NaN으로 강제
    • 반환
      - 동일 키의 dict 4종 (GC/AP2/AP1/Local)
    """    
    # ────────────────────────────────────────────────────────────────────────
    # 0) 안전 체크 & 준비
    # ────────────────────────────────────────────────────────────────────────
    if df_in_sin_assort is None:
        df_in_sin_assort = pd.DataFrame()

    TAGS_INFO = [
        (STR_DF_OUT_SIN_GC,    COL_SIN_ASSORT_GC),
        (STR_DF_OUT_SIN_AP2,   COL_SIN_ASSORT_AP2),
        (STR_DF_OUT_SIN_AP1,   COL_SIN_ASSORT_AP1),
        (STR_DF_OUT_SIN_LOCAL, COL_SIN_ASSORT_LOCAL),
    ]

    out_dict: dict[str, pd.DataFrame] = {}

    # ────────────────────────────────────────────────────────────────────────
    # 1) 각 태그별로: 기존 df_in_sin_assort에서 "이미 값 있는 키" 뽑기
    # ────────────────────────────────────────────────────────────────────────
    exist_key_map: dict[str, set[tuple]] = {}
    for _, col_ass in TAGS_INFO:
        if (not df_in_sin_assort.empty) and (col_ass in df_in_sin_assort.columns):
            exist_keys = (
                df_in_sin_assort.loc[df_in_sin_assort[col_ass].notna(),
                                     [COL_SHIP_TO, COL_ITEM, COL_LOCATION]]
                              .drop_duplicates()
                              .to_records(index=False)
            )
            exist_key_map[col_ass] = set(map(tuple, exist_keys))
        else:
            exist_key_map[col_ass] = set()

    # ────────────────────────────────────────────────────────────────────────
    # 2) 필터 로직 (ASN='Y' & 키가 기존에 존재 → drop, ASN='N' → NaN 강제)
    # ────────────────────────────────────────────────────────────────────────
    def _filter_one(df_in: pd.DataFrame, ass_col: str) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, ass_col])

        # 스키마 방어
        need = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN, ass_col]
        miss = [c for c in need if c not in df_in.columns]
        if miss:
            raise KeyError(f"[Step 2-3] S/In DF에 필수 컬럼이 없습니다: {miss}")

        df = df_in[need].copy(deep=False)

        # ASN 정상화 (object→str→upper)
        asn_y = df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')

        # 기존 값 있는 키 집합
        keyset = exist_key_map.get(ass_col, set())
        if keyset:
            keys = list(zip(df[COL_SHIP_TO].astype(str),
                            df[COL_ITEM].astype(str),
                            df[COL_LOCATION].astype(str)))
            has_old = pd.Series([k in keyset for k in keys], index=df.index)

            # ASN='Y' 이고 기존 값 존재 → 제거
            drop_mask = asn_y & has_old
            if drop_mask.any():
                df = df.loc[~drop_mask].copy()

        # ASN='N' → 해당 Assortment 컬럼 NaN 강제
        asn_y = df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')  # 재평가(필터 후)
        df[ass_col] = np.where(asn_y, 1, np.nan)

        # dtype 정리 (메모리 절감)
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')

        return df

    # ────────────────────────────────────────────────────────────────────────
    # 3) 태그별 처리 & out_dict 구성
    # ────────────────────────────────────────────────────────────────────────
    for name, col_ass in TAGS_INFO:
        df_base = sin_dict.get(name, pd.DataFrame())
        out_dict[name] = _filter_one(df_base, col_ass)

    # ────────────────────────────────────────────────────────────────────────
    # 4) 메모리 정리
    # ────────────────────────────────────────────────────────────────────────
    del df_in_sin_assort, sin_dict
    gc.collect()

    return out_dict

########################################################################################################################
# Step 2-4 : S/Out Assortment Measure (GC/AP2/AP1/Local) 비교
########################################################################################################################
@_decoration_
def fn_step02_04_compare_sout_assortments(
        df_in_sout_assort: pd.DataFrame,            # Step 2-1 결과: S/Out Assortment 전처리본
        sout_dict: dict[str, pd.DataFrame]          # Step 2-2 결과: {STR_DF_OUT_SOUT_xxx: df, ...}
) -> dict[str, pd.DataFrame]:
    """
    Step 2-4) S/Out Assortment Measure (GC/AP2/AP1/Local) 비교
    ----------------------------------------------------------------
    • 입력
      - df_in_sout_assort : (ShipTo, Item, Location, S/Out FCST Assortment_[GC|AP2|AP1|Local])
      - sout_dict         : Step2-2에서 생성된 4개 S/Out dict
                            (각 DF는 [ShipTo, Item, Location, Sales Product ASN, Assortment_xx] 스키마)
    • 처리 (키 = ShipTo, Item, Location)
      - ASN='Y' 이고, df_in_sout_assort에 해당 Assortment 값이 **이미 존재**하면 해당 row 삭제
      - ASN='N' 인 행은 해당 Assortment 컬럼을 NaN 으로 강제
    • 반환
      - 동일 키의 dict 4종 (GC/AP2/AP1/Local)
    """    
    
    if df_in_sout_assort is None:
        df_in_sout_assort = pd.DataFrame()

    TAGS_INFO = [
        (STR_DF_OUT_SOUT_GC,    COL_SOUT_ASSORT_GC),
        (STR_DF_OUT_SOUT_AP2,   COL_SOUT_ASSORT_AP2),
        (STR_DF_OUT_SOUT_AP1,   COL_SOUT_ASSORT_AP1),
        (STR_DF_OUT_SOUT_LOCAL, COL_SOUT_ASSORT_LOCAL),
    ]

    out_dict: dict[str, pd.DataFrame] = {}

    # ────────────────────────────────────────────────────────────────────────
    # 1) 태그별로 "기존 값이 있는 키(ShipTo, Item, Loc)" 집합 준비
    # ────────────────────────────────────────────────────────────────────────
    exist_key_map: dict[str, set[tuple]] = {}
    for _, col_ass in TAGS_INFO:
        if (not df_in_sout_assort.empty) and (col_ass in df_in_sout_assort.columns):
            exist_keys = (
                df_in_sout_assort.loc[df_in_sout_assort[col_ass].notna(),
                                      [COL_SHIP_TO, COL_ITEM, COL_LOCATION]]
                              .drop_duplicates()
                              .to_records(index=False)
            )
            exist_key_map[col_ass] = set(map(tuple, exist_keys))
        else:
            exist_key_map[col_ass] = set()

    # ────────────────────────────────────────────────────────────────────────
    # 2) 필터 로직: ASN='Y' & 기존 값 존재 → drop / ASN='N' → NaN 강제
    # ────────────────────────────────────────────────────────────────────────
    def _filter_one(df_in: pd.DataFrame, ass_col: str) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, ass_col])

        need = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN, ass_col]
        miss = [c for c in need if c not in df_in.columns]
        if miss:
            raise KeyError(f"[Step 2-4] S/Out DF에 필수 컬럼이 없습니다: {miss}")

        # 필요한 컬럼만 (얕은 복사) – 카테고리 dtype 유지
        df = df_in[need].copy(deep=False)

        # ASN='Y' 판단
        asn_y = df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')

        # 기존 값 존재하는 키셋
        keyset = exist_key_map.get(ass_col, set())
        if keyset:
            keys = list(zip(df[COL_SHIP_TO].astype(str),
                            df[COL_ITEM].astype(str),
                            df[COL_LOCATION].astype(str)))
            has_old = pd.Series([k in keyset for k in keys], index=df.index)

            # ASN='Y' & 기존값 존재 → drop
            drop_mask = asn_y & has_old
            if drop_mask.any():
                df = df.loc[~drop_mask].copy()

        # 남은 행들에 대해: ASN='N' → NaN, ASN='Y' → 1
        asn_y = df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')  # 재계산(필터 후)
        df[ass_col] = np.where(asn_y, 1, np.nan)

        # 메모리 절감: 카테고리 캐스팅 (마지막에만)
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')

        return df

    # ────────────────────────────────────────────────────────────────────────
    # 3) 태그별 처리 & 결과 dict 구성
    # ────────────────────────────────────────────────────────────────────────
    for df_name, col_ass in TAGS_INFO:
        base_df = sout_dict.get(df_name, pd.DataFrame())
        out_dict[df_name] = _filter_one(base_df, col_ass)

    # ────────────────────────────────────────────────────────────────────────
    # 4) 메모리 정리
    # ────────────────────────────────────────────────────────────────────────
    del df_in_sout_assort, sout_dict
    gc.collect()

    return out_dict


########################################################################################################################
# Step 2-5 : Assortment Measure Output 구성 (8개)
########################################################################################################################
@_decoration_
def fn_step02_05_format_assortment_outputs(
        sin_dict_in : dict[str, pd.DataFrame],   # ← Step 2-3 결과 dict (S/In: GC/AP2/AP1/Local)
        sout_dict_in: dict[str, pd.DataFrame],   # ← Step 2-4 결과 dict (S/Out: GC/AP2/AP1/Local)
        out_version : str                        # ← Version (예: 'CWV_DP')
) -> dict[str, pd.DataFrame]:
    """
    Step 2-5) Assortment Measure Output 구성
    ----------------------------------------------------------
    • Version.[Version Name] = out_version 추가 (하드코딩 금지)
    • Sales Product ASN 컬럼 제거
    • 8개 Output DF 반환 (o9 상위 호출 명세와 동일한 이름)

    반환 키:
      - Output_SIn_Assortment_GC / _AP2 / _AP1 / _Local
      - Output_SOut_Assortment_GC / _AP2 / _AP1 / _Local
    """

    # 0) 입력 딕셔너리 키 → (출력 키, 측정치 컬럼) 매핑
    SIN_MAP = {
        STR_DF_OUT_SIN_GC   : (OUT_SIN_ASSORTMENT_GC,    COL_SIN_ASSORT_GC),
        STR_DF_OUT_SIN_AP2  : (OUT_SIN_ASSORTMENT_AP2,   COL_SIN_ASSORT_AP2),
        STR_DF_OUT_SIN_AP1  : (OUT_SIN_ASSORTMENT_AP1,   COL_SIN_ASSORT_AP1),
        STR_DF_OUT_SIN_LOCAL: (OUT_SIN_ASSORTMENT_LOCAL, COL_SIN_ASSORT_LOCAL),
    }
    SOUT_MAP = {
        STR_DF_OUT_SOUT_GC   : (OUT_SOUT_ASSORTMENT_GC,    COL_SOUT_ASSORT_GC),
        STR_DF_OUT_SOUT_AP2  : (OUT_SOUT_ASSORTMENT_AP2,   COL_SOUT_ASSORT_AP2),
        STR_DF_OUT_SOUT_AP1  : (OUT_SOUT_ASSORTMENT_AP1,   COL_SOUT_ASSORT_AP1),
        STR_DF_OUT_SOUT_LOCAL: (OUT_SOUT_ASSORTMENT_LOCAL, COL_SOUT_ASSORT_LOCAL),
    }

    # 1) 공통 포맷터
    def _format(df_in: pd.DataFrame, meas_col: str) -> pd.DataFrame:
        """
        최종 스키마: [Version, ShipTo, Item, Location, meas_col]
        - ASN 컬럼 제거
        - Version 주입 및 컬럼 순서 고정
        - dtype: dims → category, measure → float32
        """
        col_order = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, meas_col]

        if df_in is None or df_in.empty:
            df_out = pd.DataFrame(columns=col_order)
            for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION):
                df_out[c] = df_out[c].astype('category')
            df_out[meas_col] = df_out[meas_col].astype('float32')
            return df_out

        need = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, meas_col]
        miss = [c for c in need if c not in df_in.columns]
        if miss:
            raise KeyError(f"[Step 2-5] 입력 DF에 필수 컬럼 누락: {miss}")

        # 얕은 복사 + ASN 제거
        use_cols = need + ([COL_SALES_PRODUCT_ASN] if COL_SALES_PRODUCT_ASN in df_in.columns else [])
        df = df_in[use_cols].copy(deep=False)
        if COL_SALES_PRODUCT_ASN in df.columns:
            df.drop(columns=[COL_SALES_PRODUCT_ASN], inplace=True)

        # Version 추가 + 순서 고정
        df[COL_VERSION] = out_version
        df = df[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, meas_col]]

        # dtype 정리
        for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION):
            # 이미 category 라면 유지, 아니라면 변환
            if df[c].dtype.name != 'category':
                df[c] = df[c].astype('category')
        # measure → float32 (1 또는 NaN)
        df[meas_col] = pd.to_numeric(df[meas_col], errors='coerce').astype('float32')

        return df

    # 2) S/In 변환
    out_dict: dict[str, pd.DataFrame] = {}
    for in_key, (out_key, meas_col) in SIN_MAP.items():
        df_src = sin_dict_in.get(in_key, pd.DataFrame())
        out_dict[out_key] = _format(df_src, meas_col)

    # 3) S/Out 변환
    for in_key, (out_key, meas_col) in SOUT_MAP.items():
        df_src = sout_dict_in.get(in_key, pd.DataFrame())
        out_dict[out_key] = _format(df_src, meas_col)

    # 4) 메모리 정리
    del sin_dict_in, sout_dict_in
    gc.collect()

    return out_dict

########################################################################################################################
# Step 3-공통) 현재 주차의 Partial Week 목록 추출 (current version: Partial Week만 제공)
########################################################################################################################
def _get_current_partial_weeks(df_time: pd.DataFrame) -> list[str]:
    """
    • df_in_Time 에서 '당주주차'에 해당하는 Partial Week들만 추출
    • 규칙: 첫 행의 숫자부(예: '202522')와 동일한 Partial Week (A/B 모두)만 선택
    """
    if df_time is None or df_time.empty or (COL_PWEEK not in df_time.columns):
        return []
    s = df_time[COL_PWEEK].astype(str)
    base = s.iloc[0][:-1] if len(s.iloc[0]) >= 1 else s.iloc[0]
    # 현재 주차의 A/B 모두 포함
    cur = s[s.str[:-1] == base].tolist()
    # 방어: 정렬(A 먼저, B 나중) – 관례적
    cur.sort()
    return cur

########################################################################################################################
# Step 3-공통) Y/N → 1/NaN 변환
########################################################################################################################
def _yn_to_dummy(series_asn: pd.Series) -> np.ndarray:
    asn = series_asn.astype(str).str.upper().to_numpy()
    return np.where(asn == 'Y', 1.0, np.nan)

########################################################################################################################
# Step 3-공통) 행 복제하여 Partial Week 붙이기 (벡터화)
########################################################################################################################
def _expand_with_partial_weeks(df_in: pd.DataFrame, partial_weeks: list[str], measure_col: str) -> pd.DataFrame:
    """
    입력: [ShipTo, Item, Location(있을 수도/없을 수도), measure_col, Sales Product ASN(있을 수도)]
    동작: 각 행을 partial_weeks 수만큼 복제하여 COL_PWEEK 열을 추가
    반환: [ShipTo, Item, Location, Time.[Partial Week], measure_col]
    """
    if df_in is None or df_in.empty or not partial_weeks:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, measure_col])

    # 필요한 차원 보정 (S/Out은 Location='-'가 이미 존재)
    if COL_LOCATION not in df_in.columns:
        df = df_in.assign(**{COL_LOCATION: '-'})
    else:
        df = df_in

    n = len(df)
    k = len(partial_weeks)
    rep_idx = np.repeat(np.arange(n), k)
    pw_col  = np.tile(np.array(partial_weeks, dtype=object), n)

    out = pd.DataFrame({
        COL_SHIP_TO : df[COL_SHIP_TO].to_numpy()[rep_idx],
        COL_ITEM    : df[COL_ITEM].to_numpy()[rep_idx],
        COL_LOCATION: df[COL_LOCATION].to_numpy()[rep_idx],
        COL_PWEEK   : pw_col,
        measure_col : pd.to_numeric(df[measure_col], errors='coerce').to_numpy()[rep_idx],
    })

    # dtype 정리
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK):
        out[c] = out[c].astype('category')

    return out[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, measure_col]]

########################################################################################################################
# Step 3-1) S/In Assortment → Dummy (현재 Partial Week로 확장, groupby 없음)
########################################################################################################################
@_decoration_
def fn_step03_01_build_sin_dummy_current_pw(
    df_step02_03_sin_dict: dict[str, pd.DataFrame],  # Step 2-3 결과 dict (S/In 비교 후)
    df_in_Time: pd.DataFrame                          # Input 2 (current version, Partial Week만)
) -> dict[str, pd.DataFrame]:
    """
    반환: {
      OUT_SIN_DUMMY_GC, OUT_SIN_DUMMY_AP2, OUT_SIN_DUMMY_AP1, OUT_SIN_DUMMY_LOCAL
    }
    • Sales Product ASN: Y→1, N→NaN
    • 각 행에 현재 주차 Partial Week(A/B 존재 시 둘 다) 부착
    • groupby(집계) 없음
    """
    partial_weeks = _get_current_partial_weeks(df_in_Time)

    TAGS = [
        (STR_DF_OUT_SIN_GC,    COL_SIN_DUMMY_GC),
        (STR_DF_OUT_SIN_AP2,   COL_SIN_DUMMY_AP2),
        (STR_DF_OUT_SIN_AP1,   COL_SIN_DUMMY_AP1),
        (STR_DF_OUT_SIN_LOCAL, COL_SIN_DUMMY_LOCAL),
    ]

    out: dict[str, pd.DataFrame] = {}
    for base_key, dummy_col in TAGS:
        df_base = df_step02_03_sin_dict.get(base_key, pd.DataFrame())
        
        # 빈값을 반환
        if df_base is None or df_base.empty:
            out_name = {
                COL_SIN_DUMMY_GC:    OUT_SIN_DUMMY_GC,
                COL_SIN_DUMMY_AP2:   OUT_SIN_DUMMY_AP2,
                COL_SIN_DUMMY_AP1:   OUT_SIN_DUMMY_AP1,
                COL_SIN_DUMMY_LOCAL: OUT_SIN_DUMMY_LOCAL,
            }[dummy_col]
            out[out_name] = pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, dummy_col])
            continue

        # Y/N → Dummy
        df_tmp = df_base[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]].copy(deep=False)
        df_tmp[dummy_col] = _yn_to_dummy(df_tmp[COL_SALES_PRODUCT_ASN])
        df_tmp.drop(columns=[COL_SALES_PRODUCT_ASN], inplace=True)

        # Partial Week 확장
        df_out = _expand_with_partial_weeks(df_tmp, partial_weeks, dummy_col)

        # 결과 저장
        out_name = {
            COL_SIN_DUMMY_GC:    OUT_SIN_DUMMY_GC,
            COL_SIN_DUMMY_AP2:   OUT_SIN_DUMMY_AP2,
            COL_SIN_DUMMY_AP1:   OUT_SIN_DUMMY_AP1,
            COL_SIN_DUMMY_LOCAL: OUT_SIN_DUMMY_LOCAL,
        }[dummy_col]
        out[out_name] = df_out

    return out

########################################################################################################################
# Step 3-2) S/Out Assortment → Dummy (현재 Partial Week로 확장, groupby 없음)
########################################################################################################################
@_decoration_
def fn_step03_02_build_sout_dummy_current_pw(
    df_step02_04_sout_dict: dict[str, pd.DataFrame], # Step 2-4 결과 dict (S/Out 비교 후)
    df_in_Time: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    반환: {
      OUT_SOUT_DUMMY_GC, OUT_SOUT_DUMMY_AP2, OUT_SOUT_DUMMY_AP1, OUT_SOUT_DUMMY_LOCAL
    }
    • Sales Product ASN: Y→1, N→NaN
    • 각 행에 현재 주차 Partial Week(A/B 존재 시 둘 다) 부착
    • groupby(집계) 없음
    """
    partial_weeks = _get_current_partial_weeks(df_in_Time)

    TAGS = [
        (STR_DF_OUT_SOUT_GC,    COL_SOUT_DUMMY_GC),
        (STR_DF_OUT_SOUT_AP2,   COL_SOUT_DUMMY_AP2),
        (STR_DF_OUT_SOUT_AP1,   COL_SOUT_DUMMY_AP1),
        (STR_DF_OUT_SOUT_LOCAL, COL_SOUT_DUMMY_LOCAL),
    ]

    out: dict[str, pd.DataFrame] = {}
    for base_key, dummy_col in TAGS:
        df_base = df_step02_04_sout_dict.get(base_key, pd.DataFrame())
        if df_base is None or df_base.empty:
            out_name = {
                COL_SOUT_DUMMY_GC:    OUT_SOUT_DUMMY_GC,
                COL_SOUT_DUMMY_AP2:   OUT_SOUT_DUMMY_AP2,
                COL_SOUT_DUMMY_AP1:   OUT_SOUT_DUMMY_AP1,
                COL_SOUT_DUMMY_LOCAL: OUT_SOUT_DUMMY_LOCAL,
            }[dummy_col]
            out[out_name] = pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, dummy_col])
            continue

        # Y/N → Dummy
        df_tmp = df_base[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]].copy(deep=False)
        df_tmp[dummy_col] = _yn_to_dummy(df_tmp[COL_SALES_PRODUCT_ASN])
        df_tmp.drop(columns=[COL_SALES_PRODUCT_ASN], inplace=True)

        # Partial Week 확장
        df_out = _expand_with_partial_weeks(df_tmp, partial_weeks, dummy_col)

        out_name = {
            COL_SOUT_DUMMY_GC:    OUT_SOUT_DUMMY_GC,
            COL_SOUT_DUMMY_AP2:   OUT_SOUT_DUMMY_AP2,
            COL_SOUT_DUMMY_AP1:   OUT_SOUT_DUMMY_AP1,
            COL_SOUT_DUMMY_LOCAL: OUT_SOUT_DUMMY_LOCAL,
        }[dummy_col]
        out[out_name] = df_out

    return out

########################################################################################################################
# Step 3-3-1) Flooring FCST Dummy 모수 산정 (VD Item만, Std5 단위 집계)
########################################################################################################################
@_decoration_
def fn_step03_03_01_flooring_population(
    df_step01_01_asn_delta: pd.DataFrame,             # Step 1-1 결과 (Delta, Y/N)
    df_asn_base: pd.DataFrame,                        # input 7
    df_in_item: pd.DataFrame,                         # Input 11 (Item GBM, Item Std1, Item)
    df_in_Sales_Domain_Dimension: pd.DataFrame        # Input 1 (ShipTo → Std1..Std6)
) -> pd.DataFrame:
    """
    반환 스키마(중간): [Sales Domain.[Ship To](=Std5), Item.[Item], Sales Product ASN]  ※ ShipTo는 Std5 코드
    처리:
      1) df_step01_01_asn_delta 에서 Item GBM == 'VD' 인 Item만 남김 (df_in_item 조인)
      2) df_step01_03_asn_all 과 (ShipTo, Item, Location, Sales Product ASN) 완전일치 행은 제거 (중복 제거)
      3) ShipTo → Std5 로 매핑
      4) (Std5, Item) 기준 'Sales Product ASN' 집계 (any/최대값: Y가 하나라도 있으면 Y)
    """
    # ── 0) 방어/필수 컬럼 체크
    need_asn = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
    for name, df in {
        'Step1-1': df_step01_01_asn_delta,
        'input 7': df_asn_base,
        'Item'   : df_in_item,
        'Dim'    : df_in_Sales_Domain_Dimension,
    }.items():
        if df is None or df.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN])
    
    if COL_ITEM_GBM not in df_in_item.columns:
        raise KeyError("[Step 3-3-1] df_in_item must include COL_ITEM_GBM")
    for c in need_asn:
        if c not in df_step01_01_asn_delta.columns or c not in df_asn_base.columns:
            raise KeyError(f"[Step 3-3-1] Missing ASN columns: {c}")
    for c in (COL_SHIP_TO, COL_STD5):
        if c not in df_in_Sales_Domain_Dimension.columns:
            raise KeyError(f"[Step 3-3-1] Sales-Domain Dimension missing: {c}")

    # ── 1) Delta 중 VD Item만
    vd_items = set(
        df_in_item.loc[df_in_item[COL_ITEM_GBM].astype(str).str.upper().eq('VD'), COL_ITEM].astype(str)
    )
    if not vd_items:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN])

    df_delta = df_step01_01_asn_delta.loc[
        df_step01_01_asn_delta[COL_ITEM].astype(str).isin(vd_items),
        need_asn
    ].copy(deep=False)

    if df_delta.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN])

    # ── 2) Delta에서 ASN All 과 완전 일치하는 행 제거 (안티조인: ShipTo+Item+Loc+ASN)
    key_all   = (df_asn_base[COL_SHIP_TO].astype(str) + '|' +
                 df_asn_base[COL_ITEM].astype(str)    + '|' +
                 df_asn_base[COL_LOCATION].astype(str)+ '|' +
                 df_asn_base[COL_SALES_PRODUCT_ASN].astype(str)).unique()
    dup_set = set(key_all)

    key_delta = (df_delta[COL_SHIP_TO].astype(str) + '|' +
                 df_delta[COL_ITEM].astype(str)    + '|' +
                 df_delta[COL_LOCATION].astype(str)+ '|' +
                 df_delta[COL_SALES_PRODUCT_ASN].astype(str))
    df_delta = df_delta.loc[~key_delta.isin(dup_set)].copy(deep=False)

    if df_delta.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN])

    # ── 3) ShipTo → Std5 매핑
    std5_map = df_in_Sales_Domain_Dimension.set_index(COL_SHIP_TO)[COL_STD5].astype(str).to_dict()
    df_delta[COL_SHIP_TO] = df_delta[COL_SHIP_TO].astype(str).map(std5_map)
    df_delta = df_delta[~df_delta[COL_SHIP_TO].isna()].copy()
    if df_delta.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN])

    # ── 4) (Std5, Item) 집계: Y 있으면 Y, 아니면 N
    #     ultra_fast_groupby_numpy_general 사용 (Y/N → 0/1 → max → 0/1 → Y/N 복원)
    tmp = df_delta[[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN]].copy(deep=False)
    # 안전 처리: Y/N → bool(1/0)로 임시 치환
    tmp['_yn'] = tmp[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y').astype(np.int8)

    gb = ultra_fast_groupby_numpy_general(
        df=tmp,
        key_cols=[COL_SHIP_TO, COL_ITEM],
        aggs={'_yn': 'max'}        # any와 동일
    )
    gb[COL_SALES_PRODUCT_ASN] = np.where(gb['_yn'] >= 1, 'Y', 'N')
    gb.drop(columns=['_yn'], inplace=True)

    # dtype 정리
    for c in (COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN):
        gb[c] = gb[c].astype('category')

    return gb[[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN]]

########################################################################################################################
# Step 3-3-2) Flooring FCST Dummy 생성 (현재 Partial Week 확장)
########################################################################################################################
@_decoration_
def fn_step03_03_02_flooring_dummy_expand_pw(
    df_flooring_pop: pd.DataFrame,   # Step 3-3-1 결과: [Std5(as ShipTo), Item, Sales Product ASN]
    df_in_Time: pd.DataFrame
) -> pd.DataFrame:
    """
    반환: [ShipTo(=Std5), Item, Time.[Partial Week], Flooring FCST Dummy]
    • Y→1, N→NaN
    • 현재 주차 Partial Week(A/B)로 확장
    """
    partial_weeks = _get_current_partial_weeks(df_in_Time)
    if df_flooring_pop is None or df_flooring_pop.empty or not partial_weeks:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_FLOORING_DUMMY])

    df_tmp = df_flooring_pop[[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN]].copy(deep=False)
    df_tmp[COL_FLOORING_DUMMY] = _yn_to_dummy(df_tmp[COL_SALES_PRODUCT_ASN])
    df_tmp.drop(columns=[COL_SALES_PRODUCT_ASN], inplace=True)

    # Flooring 은 Location 차원이 없음 → 내부에서 '-' 로 부여 후 삭제
    df_tmp[COL_LOCATION] = '-'
    df_out = _expand_with_partial_weeks(df_tmp, partial_weeks, COL_FLOORING_DUMMY)
    df_out.drop(columns=[COL_LOCATION], inplace=True)

    return df_out[[COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_FLOORING_DUMMY]]

########################################################################################################################
# Step 3-4) FCST Dummy Output 정리 (Version 주입, 컬럼 정렬)
########################################################################################################################
@_decoration_
def fn_step03_04_format_dummy_outputs(
    sin_dummy_dict: dict[str, pd.DataFrame],
    sout_dummy_dict: dict[str, pd.DataFrame],
    df_flooring_dummy: pd.DataFrame,
    out_version: str
) -> dict[str, pd.DataFrame]:
    """
    반환 dict (9개):
      Output_SIn_Dummy_GC / _AP2 / _AP1 / _Local
      Output_SOut_Dummy_GC / _AP2 / _AP1 / _Local
      Output_Flooring_Dummy
    """
    def _fmt(df_in: pd.DataFrame, meas_col: str, with_loc: bool = True) -> pd.DataFrame:
        # 스키마 정리 + Version 주입
        if df_in is None or df_in.empty:
            cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if with_loc else []) + [COL_PWEEK, meas_col]
            df = pd.DataFrame(columns=cols)
        else:
            df = df_in.copy(deep=False)
            for c in (COL_SHIP_TO, COL_ITEM):
                if c in df.columns and df[c].dtype.name != 'category':
                    df[c] = df[c].astype('category')
            if with_loc:
                if COL_LOCATION not in df.columns:
                    df[COL_LOCATION] = '-'
                else:
                    df[COL_LOCATION] = df[COL_LOCATION].astype('category')
            df[COL_PWEEK] = df[COL_PWEEK].astype('category')
            df[meas_col] = pd.to_numeric(df[meas_col], errors='coerce').astype('float32')
        if not COL_VERSION in df.columns:
            df.insert(0, COL_VERSION, out_version)
        cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if with_loc else []) + [COL_PWEEK, meas_col]
        return df[cols]

    out: dict[str, pd.DataFrame] = {}

    # S/In
    out[OUT_SIN_DUMMY_GC]    = _fmt(sin_dummy_dict.get(OUT_SIN_DUMMY_GC),    COL_SIN_DUMMY_GC,    with_loc=True)
    out[OUT_SIN_DUMMY_AP2]   = _fmt(sin_dummy_dict.get(OUT_SIN_DUMMY_AP2),   COL_SIN_DUMMY_AP2,   with_loc=True)
    out[OUT_SIN_DUMMY_AP1]   = _fmt(sin_dummy_dict.get(OUT_SIN_DUMMY_AP1),   COL_SIN_DUMMY_AP1,   with_loc=True)
    out[OUT_SIN_DUMMY_LOCAL] = _fmt(sin_dummy_dict.get(OUT_SIN_DUMMY_LOCAL), COL_SIN_DUMMY_LOCAL, with_loc=True)

    # S/Out
    out[OUT_SOUT_DUMMY_GC]    = _fmt(sout_dummy_dict.get(OUT_SOUT_DUMMY_GC),    COL_SOUT_DUMMY_GC,    with_loc=True)
    out[OUT_SOUT_DUMMY_AP2]   = _fmt(sout_dummy_dict.get(OUT_SOUT_DUMMY_AP2),   COL_SOUT_DUMMY_AP2,   with_loc=True)
    out[OUT_SOUT_DUMMY_AP1]   = _fmt(sout_dummy_dict.get(OUT_SOUT_DUMMY_AP1),   COL_SOUT_DUMMY_AP1,   with_loc=True)
    out[OUT_SOUT_DUMMY_LOCAL] = _fmt(sout_dummy_dict.get(OUT_SOUT_DUMMY_LOCAL), COL_SOUT_DUMMY_LOCAL, with_loc=True)

    # Flooring (Location 없음)
    if df_flooring_dummy is None:
        df_flooring_dummy = pd.DataFrame()
    out[OUT_FLOORING_DUMMY] = _fmt(df_flooring_dummy, COL_FLOORING_DUMMY, with_loc=False)

    return out

########################################################################################################################
# Step 3) Orchestrator — FCST Dummy 전체 생성 (서브스텝 호출 일괄)
########################################################################################################################
@_decoration_
def fn_step03_build_fcst_dummy_all(
    df_step02_03_sin_dict: dict[str, pd.DataFrame],   # Step 2-3
    df_step02_04_sout_dict: dict[str, pd.DataFrame],  # Step 2-4
    df_step01_01_asn_delta: pd.DataFrame,             # Step 1-1
    df_asn_base: pd.DataFrame,                        # input 7
    df_in_item: pd.DataFrame,                         # Input 11
    df_in_Sales_Domain_Dimension: pd.DataFrame,       # Input 1
    df_in_Time: pd.DataFrame,                         # Input 2
    out_version: str
) -> dict[str, pd.DataFrame]:
    """
    반환 dict (9개 Output)
    """
    # 3-1
    sin_dummy_dict = fn_step03_01_build_sin_dummy_current_pw(
        df_step02_03_sin_dict, df_in_Time
    )
    # 3-2
    sout_dummy_dict = fn_step03_02_build_sout_dummy_current_pw(
        df_step02_04_sout_dict, df_in_Time
    )
    # 3-3-1
    df_flooring_pop = fn_step03_03_01_flooring_population(
        df_step01_01_asn_delta,
        df_asn_base,
        df_in_item,
        df_in_Sales_Domain_Dimension
    )
    # 3-3-2
    df_flooring_dummy = fn_step03_03_02_flooring_dummy_expand_pw(
        df_flooring_pop, df_in_Time
    )
    # 3-4
    out_dict = fn_step03_04_format_dummy_outputs(
        sin_dummy_dict, sout_dummy_dict, df_flooring_dummy, out_version
    )
    return out_dict

####################################
############ Start Main  ###########
####################################
if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    input_dataframes = {}
    try:
        ################################################################################################################
        # 전처리 : 모듈 내에서 사용될 데이터에 대한 정합성 체크 및 데이터 선 가공
        ################################################################################################################
        
        if is_local:
            Version = 'CWV_DP'
            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------

            # input_folder_name  = str_instance           
            # output_folder_name = str_instance
            input_folder_name  = "PYSalesProductASNDelta"           
            output_folder_name = str_instance
            
            # ------
            str_input_dir = f'Input/{input_folder_name}/batch_0925_v2'
            # ------
            str_output_dir = f'Output/{output_folder_name}'
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y%m%d_%H_%M")
            str_output_dir = f"{str_output_dir}_{formatted_time}"
            # ------
            os.makedirs(str_input_dir, exist_ok=True)
            os.makedirs(str_output_dir, exist_ok=True)


            # ----------------------------------------------------
            # Week
            # ----------------------------------------------------
            # CurrentPartialWeek = '202447A'

        # --------------------------------------------------------------------------    
        # vdLog 초기화
        # --------------------------------------------------------------------------
        log_path = os.path.dirname(__file__) if is_local else ""
        vdCommon.gfn_pyLog_start(Version, str_instance, logger, is_local, log_path)
        
        # --------------------------------------------------------------------------
        # df_input 체크 시작
        # --------------------------------------------------------------------------
        logger.Note(p_note='df_input 체크 시작', p_log_level=LOG_LEVEL.debug())
        fn_process_in_df_mst()
        for in_df in input_dataframes:
            # 로그출력
            fn_log_dataframe(input_dataframes[in_df], in_df)


        logger.Note(p_note=f'Parameter Check', p_log_level=LOG_LEVEL.debug())
        logger.Note(p_note=f'Version            : {Version}', p_log_level=LOG_LEVEL.debug())
        # logger.Note(p_note=f'CurrentPartialWeek : {CurrentPartialWeek}', p_log_level=LOG_LEVEL.debug())


        ############ To do : 여기 아래에 Step Function 들을 Call 하는 코드 구현. ########
        # 예시
        # ################################################################################################################
        # # Step 00 – Ship-To 차원 LUT 구축
        # ################################################################################################################
        # dict_log = {
        #     'p_step_no': 00,
        #     'p_step_desc': 'Step 00 – load Ship-To dimension LUT'
        # }
        # df_fn_shipto_dim = step00_load_shipto_dimension(
        #     input_dataframes[STR_DF_DIM],
        #     **dict_log
        # )
        # fn_log_dataframe(df_fn_shipto_dim, f'step00_df_fn_shipto_dim')

        ################################################################################################################
        # Step 1-1) Sales Product ASN Delta 전처리
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 11,
            'p_step_desc': 'Step 1-1) Sales Product ASN Delta 전처리',
            'p_df_name'  : None
        }
        df_step01_01_asn_delta = fn_step01_01_preprocess_sales_product_asn_delta(
            input_dataframes[DF_IN_SALES_PRODUCT_ASN_DELTA],
            **dict_log
        )
        # 필요 시 중간결과 로그/CSV 출력은 데코레이터와 fn_log_dataframe 에서 처리됩니다.
        fn_log_dataframe(df_step01_01_asn_delta, 'df_step01_01_asn_delta')

        ################################################################################################################
        # Step 1-2) Sales Product ASN 전처리
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 12,
            'p_step_desc': 'Step 1-2) Sales Product ASN 전처리',
            'p_df_name'  : None
        }
        df_step01_02_asn_base_filtered = fn_step01_02_preprocess_sales_product_asn(
            input_dataframes[DF_IN_SALES_PRODUCT_ASN],     # 기본 ASN (Input 7)
            df_step01_01_asn_delta,                        # Step 1-1 결과
            **dict_log
        )
        fn_log_dataframe(df_step01_02_asn_base_filtered, 'df_step01_02_asn_base_filtered')

        ################################################################################################################
        # Step 1-3) Sales Product ASN 구성 (ShipTo × Item × Location)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 13,
            'p_step_desc': 'Step 1-3) Sales Product ASN 구성 (ShipTo×Item×Location)',
            'p_df_name'  : None
        }
        df_step01_03_asn_all = fn_step01_03_build_sales_product_asn(
            df_step01_01_asn_delta,                 # 1-1 결과
            df_step01_02_asn_base_filtered,         # 1-2 결과
            **dict_log
        )
        fn_log_dataframe(df_step01_03_asn_all, 'df_step01_03_asn_all')

        
        ################################################################################################################
        # Step 1-4) Sales Product ASN output. 기존 1-6
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 14,
            'p_step_desc': 'Step 1-4) Sales Product ASN output',
            'p_df_name'  : None
        }
        Output_Sales_Product_ASN = fn_step01_04_output_sales_product_asn(
            input_dataframes[DF_IN_SALES_PRODUCT_ASN_DELTA],    # Input6 => input5
            Version,                                            # 예: 'CWV_DP'
            **dict_log
        )
        fn_log_dataframe(Output_Sales_Product_ASN, f'df_step01_04_{OUT_SALES_PRODUCT_ASN}')  # 필요시 CSV도 저장
        
        ################################################################################################################
        # Step 1-5) Sales Product ASN Delta output. 기존 1-7
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 15,
            'p_step_desc': 'Step 1-5) Sales Product ASN Delta output',
            'p_df_name'  : None
        }
        Output_Sales_Product_ASN_Delta = fn_step01_05_output_sales_product_asn_delta(
            input_dataframes[DF_IN_SALES_PRODUCT_ASN_DELTA],  # Input 6 => input5
            Version,                                          # 예: 'CWV_DP'
            **dict_log
        )

        # 필요 시 중간결과 로그/CSV 출력은 데코레이터와 fn_log_dataframe 에서 처리됩니다.
        fn_log_dataframe(Output_Sales_Product_ASN_Delta, f'df_step01_05_{OUT_SALES_PRODUCT_ASN_DELTA}')

        ##############################################################################################################
        # Step 2-1) Assortment 전처리
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 21,
            'p_step_desc': 'Step 2-1) Assortment 전처리',
            'p_df_name'  : None
        }
        df_step02_01_sin_assort, df_step02_01_sout_assort = fn_step02_01_preprocess_assortment(
            input_dataframes[DF_IN_SIN_ASSORTMENT],      # Input 8
            input_dataframes[DF_IN_SOUT_ASSORTMENT],     # Input 9
            df_step01_01_asn_delta,                      # Step 1-1 결과
            **dict_log
        )
        fn_log_dataframe(df_step02_01_sin_assort,  'df_step02_01_sin_assort')
        fn_log_dataframe(df_step02_01_sout_assort, 'df_step02_01_sout_assort')

        ################################################################################################################
        # Step 2-2) Assortment Measure 구성 (GC/AP2/AP1/Local)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 22,
            'p_step_desc': 'Step 2-2) Assortment Measure 구성 (GC/AP2/AP1/Local)',
            'p_df_name'  : None
        }
        sin_dict, sout_dict = fn_step02_02_build_assortments(
            df_step01_03_asn_all,                                # Step 1-3 결과
            input_dataframes[DF_IN_FORECAST_RULE],               # Forecast-Rule
            input_dataframes[DF_IN_ITEM_MASTER],                 # Item Master (Item Std1 ↔ Product Group)
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],      # Sales-Domain Dimension
            input_dataframes[DF_IN_SALES_DOMAIN_ESTORE],         # E-Store Ship-To
            **dict_log
        )
        # 로그/CSV (데코레이터 외 필요 시)
        for name, df in sin_dict.items():
            fn_log_dataframe(df, f'step02_02_{name}')
        for name, df in sout_dict.items():
            fn_log_dataframe(df, f'step02_02_{name}')

        # (선택) 변수로 펼치기 — 이후 단계에서 개별 접근이 필요할 경우
        df_step02_02_SIn_Assortment_GC    = sin_dict[STR_DF_OUT_SIN_GC]
        df_step02_02_SIn_Assortment_AP2   = sin_dict[STR_DF_OUT_SIN_AP2]
        df_step02_02_SIn_Assortment_AP1   = sin_dict[STR_DF_OUT_SIN_AP1]
        df_step02_02_SIn_Assortment_Local = sin_dict[STR_DF_OUT_SIN_LOCAL]

        df_step02_02_SOut_Assortment_GC    = sout_dict[STR_DF_OUT_SOUT_GC]
        df_step02_02_SOut_Assortment_AP2   = sout_dict[STR_DF_OUT_SOUT_AP2]
        df_step02_02_SOut_Assortment_AP1   = sout_dict[STR_DF_OUT_SOUT_AP1]
        df_step02_02_SOut_Assortment_Local = sout_dict[STR_DF_OUT_SOUT_LOCAL]

        ################################################################################################################
        # Step 2-3) S/In Assortment Measure 비교 (GC/AP2/AP1/Local)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 23,
            'p_step_desc': 'Step 2-3) S/In Assortment Measure 비교',
            'p_df_name'  : None
        }
        # Step 2-1 결과 (S/In 전처리) + Step 2-2 결과(S/In dict)를 사용
        df_step02_03_sin_dict = fn_step02_03_compare_sin_assortments(
            df_step02_01_sin_assort,    # ← fn_step02_01_preprocess_assortment 반환 1번째 (S/In)
            sin_dict,   # ← fn_step02_02_build_assortments 반환 dict 중 S/In dict
            **dict_log
        )
        # 필요시 로그
        for k, df_tmp in df_step02_03_sin_dict.items():
            fn_log_dataframe(df_tmp, f'step02_03_{k}')

        ################################################################################################################
        # Step 2-4) S/Out Assortment Measure 비교 (GC/AP2/AP1/Local)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 24,
            'p_step_desc': 'Step 2-4) S/Out Assortment Measure 비교',
            'p_df_name'  : None
        }
        df_step02_04_sout_dict = fn_step02_04_compare_sout_assortments(
            df_step02_01_sout_assort,   # ← fn_step02_01_preprocess_assortment 반환 2번째 (S/Out)
            sout_dict,                  # ← fn_step02_02_build_assortments 반환 dict 중 S/Out dict
            **dict_log
        )

        # 필요 시 로깅/CSV
        for k, df_tmp in df_step02_04_sout_dict.items():
            fn_log_dataframe(df_tmp, f'step02_04_{k}')

        ################################################################################################################
        # Step 2-5) Assortment Measure Output 구성
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 25,
            'p_step_desc': 'Step 2-5) Assortment Measure Output 구성',
            'p_df_name'  : None
        }
        df_step02_05_out = fn_step02_05_format_assortment_outputs(
            df_step02_03_sin_dict,   # Step 2-3 결과 dict
            df_step02_04_sout_dict,  # Step 2-4 결과 dict
            Version,                 # 전역 Version (예: 'CWV_DP')
            **dict_log
        )
        # 8개 Output으로 펼치기
        Output_SIn_Assortment_GC     = df_step02_05_out[OUT_SIN_ASSORTMENT_GC]
        Output_SIn_Assortment_AP2    = df_step02_05_out[OUT_SIN_ASSORTMENT_AP2]
        Output_SIn_Assortment_AP1    = df_step02_05_out[OUT_SIN_ASSORTMENT_AP1]
        Output_SIn_Assortment_Local  = df_step02_05_out[OUT_SIN_ASSORTMENT_LOCAL]
        Output_SOut_Assortment_GC    = df_step02_05_out[OUT_SOUT_ASSORTMENT_GC]
        Output_SOut_Assortment_AP2   = df_step02_05_out[OUT_SOUT_ASSORTMENT_AP2]
        Output_SOut_Assortment_AP1   = df_step02_05_out[OUT_SOUT_ASSORTMENT_AP1]
        Output_SOut_Assortment_Local = df_step02_05_out[OUT_SOUT_ASSORTMENT_LOCAL]

        # 로깅
        for name, df_out in df_step02_05_out.items():
            fn_log_dataframe(df_out, f'step02_05_{name}')           

        ################################################################################################################
        # Step 3) FCST Dummy 값 생성 (Y/N) — (3-1 ~ 3-4 일괄)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 31,
            'p_step_desc': 'Step 3) FCST Dummy 값 생성 (Y/N)',
            'p_df_name'  : None
        }
        df_step03_dummy_dict = fn_step03_build_fcst_dummy_all(
            df_step02_03_sin_dict,                      # ← Step 2-3 결과
            df_step02_04_sout_dict,                     # ← Step 2-4 결과
            df_step01_01_asn_delta,                     # ← Step 1-1 결과
            # df_step01_03_asn_all,                     # ← Step 1-3 결과
            input_dataframes[DF_IN_SALES_PRODUCT_ASN],  # 기본 ASN (Input 7)
            input_dataframes[DF_IN_ITEM_MASTER],        # ← Input 11 (Item GBM/Std1/Item)
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],  # ← Input 1
            input_dataframes[DF_IN_TIME],               # ← Input 2 (Partial Week only)
            Version,                                    # ← 예: 'CWV_DP'
            **dict_log
        )
        
        # 9개 Output으로 펼치기
        Output_SIn_Dummy_GC    = df_step03_dummy_dict[OUT_SIN_DUMMY_GC]
        Output_SIn_Dummy_AP2   = df_step03_dummy_dict[OUT_SIN_DUMMY_AP2]
        Output_SIn_Dummy_AP1   = df_step03_dummy_dict[OUT_SIN_DUMMY_AP1]
        Output_SIn_Dummy_Local = df_step03_dummy_dict[OUT_SIN_DUMMY_LOCAL]
        Output_SOut_Dummy_GC   = df_step03_dummy_dict[OUT_SOUT_DUMMY_GC]
        Output_SOut_Dummy_AP2  = df_step03_dummy_dict[OUT_SOUT_DUMMY_AP2]
        Output_SOut_Dummy_AP1  = df_step03_dummy_dict[OUT_SOUT_DUMMY_AP1]
        Output_SOut_Dummy_Local= df_step03_dummy_dict[OUT_SOUT_DUMMY_LOCAL]
        Output_Flooring_Dummy  = df_step03_dummy_dict[OUT_FLOORING_DUMMY]

        # 로깅
        for name, df_out in df_step03_dummy_dict.items():
            fn_log_dataframe(df_out, f'step03_{name}')

    
    except Exception as e:
        trace_msg = traceback.format_exc()
        logger.Note(p_note=trace_msg, p_log_level=LOG_LEVEL.debug())
        logger.Error()
        if flag_exception:
            raise Exception(e)
        else:
            logger.info(f'{str_instance} exit - {time.strftime("%Y-%m-%d - %H:%M:%S")}')


    finally:
        
        if is_local:
            log_file_name = common.G_PROGRAM_NAME.replace('py', 'log')
            log_file_name = f'log/{log_file_name}'

            shutil.copyfile(log_file_name, os.path.join(str_output_dir, os.path.basename(log_file_name)))

            # prografile copy
            program_path = f"{os.getcwd()}/NSCM_DP_UI_Develop/{str_instance}.py"
            shutil.copyfile(program_path, os.path.join(str_output_dir, os.path.basename(program_path)))


        logger.Finish()
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::') # 25.05.12 need warning Log by Logger Issue
        