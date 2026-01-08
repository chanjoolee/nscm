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
from typing import Collection, Tuple,Union,Dict, Set
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
str_instance = 'PYSalesProductASNDeltaB2C'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"

is_print = True
flag_csv = True
flag_exception = True

# ======================================================
# 컬럼 상수 (Columns)
# ======================================================
# 공통 차원
COL_VERSION                 = 'Version.[Version Name]'
COL_SHIP_TO                 = 'Sales Domain.[Ship To]'
COL_STD1                    = 'Sales Domain.[Sales Std1]'
COL_STD2                    = 'Sales Domain.[Sales Std2]'
COL_STD3                    = 'Sales Domain.[Sales Std3]'
COL_STD4                    = 'Sales Domain.[Sales Std4]'
COL_STD5                    = 'Sales Domain.[Sales Std5]'
COL_STD6                    = 'Sales Domain.[Sales Std6]'
COL_ITEM_GBM                = 'Item.[Item GBM]'
COL_ITEM_STD1               = 'Item.[Item Std1]'
COL_ITEM_STD2               = 'Item.[Item Std2]'
COL_ITEM_STD3               = 'Item.[Item Std3]'
COL_ITEM_STD4               = 'Item.[Item Std4]'
COL_ITEM                    = 'Item.[Item]'

COL_LOCATION                = 'Location.[Location]'

# 시간
COL_PW                      = 'Time.[Partial Week]'
COL_WEEK                    = 'Time.[Week]'
COL_MONTH                   = 'Time.[Month]'

# Dummy (S/In)
COL_SIN_DUMMY_AP1           = 'S/In FCST(GI) Dummy_AP1'
COL_SIN_DUMMY_AP2           = 'S/In FCST(GI) Dummy_AP2'
COL_SIN_DUMMY_GC            = 'S/In FCST(GI) Dummy_GC'
COL_SIN_DUMMY_LOCAL         = 'S/In FCST(GI) Dummy_Local'

# Dummy (S/Out)
COL_SOUT_DUMMY_AP1          = 'S/Out FCST Dummy_AP1'
COL_SOUT_DUMMY_AP2          = 'S/Out FCST Dummy_AP2'
COL_SOUT_DUMMY_GC           = 'S/Out FCST Dummy_GC'
COL_SOUT_DUMMY_LOCAL        = 'S/Out FCST Dummy_Local'

# Flooring Dummy/FCST
COL_FLOORING_DUMMY          = 'Flooring FCST Dummy'
COL_FLOORING_FCST           = 'Flooring FCST'
COL_FLOORING_ASSORT         = 'Flooring FCST Assortment'

# Assortment (S/In)
COL_SIN_ASSORT_AP1          = 'S/In FCST(GI) Assortment_AP1'
COL_SIN_ASSORT_AP2          = 'S/In FCST(GI) Assortment_AP2'
COL_SIN_ASSORT_GC           = 'S/In FCST(GI) Assortment_GC'
COL_SIN_ASSORT_LOCAL        = 'S/In FCST(GI) Assortment_Local'

# Assortment (S/Out)
COL_SOUT_ASSORT_AP1         = 'S/Out FCST Assortment_AP1'
COL_SOUT_ASSORT_AP2         = 'S/Out FCST Assortment_AP2'
COL_SOUT_ASSORT_GC          = 'S/Out FCST Assortment_GC'
COL_SOUT_ASSORT_LOCAL       = 'S/Out FCST Assortment_Local'

# FCST (S/In)
COL_SIN_GI_AP1              = 'S/In FCST(GI)_AP1'
COL_SIN_BL_AP1              = 'S/In FCST(BL)_AP1'
COL_SIN_NEW_MODEL           = 'S/In FCST(GI) New Model'
COL_SIN_GI_AP2              = 'S/In FCST(GI)_AP2'
COL_SIN_BL_AP2              = 'S/In FCST(BL)_AP2'
COL_SIN_GC                  = 'S/In FCST(GI)_GC'
COL_SIN_BL_GC               = 'S/In FCST(BL)_GC'
COL_SIN_LOCAL               = 'S/In FCST(GI)_Local'
COL_SIN_BL_LOCAL            = 'S/In FCST(BL)_Local'

# FCST (S/Out)
COL_SOUT_AP1                = 'S/Out FCST_AP1'
COL_SOUT_AP2                = 'S/Out FCST_AP2'
COL_SOUT_GC                 = 'S/Out FCST_GC'
COL_SOUT_LOCAL              = 'S/Out FCST_Local'

# Estimated Price
COL_EST_PRICE_MOD_LOCAL     = 'Estimated Price Modify_Local'
COL_EST_PRICE_LOCAL         = 'Estimated Price_Local'
COL_AP_PRICE_USD            = 'Action Plan Price_USD'
COL_EXRATE_LOCAL            = 'Exchange Rate_Local'
COL_EP_STD2_LOCAL           = 'Estimated Price Item Std2_Local'
COL_EP_STD3_LOCAL           = 'Estimated Price Item Std3_Local'
COL_EP_STD4_LOCAL           = 'Estimated Price Item Std4_Local'

# Split Ratio (S/In)
COL_SIN_SR_AP1              = 'S/In FCST(GI) Split Ratio_AP1'
COL_SIN_SR_AP2              = 'S/In FCST(GI) Split Ratio_AP2'
COL_SIN_SR_GC               = 'S/In FCST(GI) Split Ratio_GC'
COL_SIN_SR_LOCAL            = 'S/In FCST(GI) Split Ratio_Local'

# Split Ratio (S/Out)
COL_SOUT_SR_AP1             = 'S/Out FCST Split Ratio_AP1'
COL_SOUT_SR_AP2             = 'S/Out FCST Split Ratio_AP2'
COL_SOUT_SR_GC              = 'S/Out FCST Split Ratio_GC'
COL_SOUT_SR_LOCAL           = 'S/Out FCST Split Ratio_Local'


# Estimated Price 신규 컬럼. 2025.11.07
COL_EST_PRICE_COLOR       = 'Estimated Price Color'

# 문자열 오타 방지용 토큰. 2025.11.07
MEASURE_LV_AP1   = 'ap1'
MEASURE_LV_AP2   = 'ap2'
MEASURE_LV_GC    = 'gc'
MEASURE_LV_LOCAL = 'local'
SIL_PAIR_SEP     = '^'
SIL_PART_SEP     = ':'

# ======================================================
# 내부 사용 상수: 컬럼 템플릿/매핑
# ======================================================
MEASURE_MAP = {'ap1': 'AP1', 'ap2': 'AP2', 'gc': 'GC', 'local': 'Local'}

# Dummy column name templates
COL_SIN_DUMMY_PREFIX   = 'S/In FCST(GI) Dummy_'
COL_SOUT_DUMMY_PREFIX  = 'S/Out FCST Dummy_'
COL_FLOORING_DUMMY     = 'Flooring FCST Dummy'   # Flooring은 단일 컬럼(주차 기준)

# 현행 FCST 비교 컬럼 (가정: 실제 FCST도 동일한 네이밍 패턴)
COL_SIN_FCST_PREFIX    = 'S/In FCST(GI)_'
COL_SOUT_FCST_PREFIX   = 'S/Out FCST_'
COL_FLOORING_FCST      = 'Flooring FCST'

# ※ 이미 다른 곳에서 정의했다면 중복 선언 금지
# 25.11.17 추가: Sales Std2 및 신규 Estimated Price 컬럼명
COL_EP_SALES_STD4_LOCAL      = 'Estimated Price Sales Std2 Item Std4_Local'
COL_EP_SALES_STD3_LOCAL      = 'Estimated Price Sales Std2 Item Std3_Local'
COL_EP_SALES_STD2_LOCAL      = 'Estimated Price Sales Std2 Item Std2_Local'

# ======================================================
# 데이터프레임 상수
# ======================================================
# ---------- INPUT DF KEYS ----------
DF_IN_SIN_DUMMY                 = 'df_in_sin_fcst_dummy'
DF_IN_SOUT_DUMMY                = 'df_in_sout_fcst_dummy'
DF_IN_FLOORING_DUMMY            = 'df_in_flooring_fcst_dummy'
DF_IN_SDD                       = 'df_in_Sales_Domain_Dimension'
DF_IN_TIME_PW                   = 'df_in_Time_pw'
DF_IN_TIME_W                    = 'df_in_Time_w'
DF_IN_ITEM_MST                  = 'df_in_Item_Master'
DF_IN_ESTORE                    = 'df_in_Sales_Domain_Estore'

DF_IN_EST_PRICE                 = 'df_in_Estimated_Price'
DF_IN_AP_PRICE                  = 'df_in_Action_Plan_Price'
DF_IN_EXRATE_LOCAL              = 'df_in_Exchange_Rate_Local'
DF_IN_EP_STD2_LOCAL             = 'df_in_Estimated_Price_Item_Std2_Local'
DF_IN_EP_STD3_LOCAL             = 'df_in_Estimated_Price_Item_Std3_Local'
DF_IN_EP_STD4_LOCAL             = 'df_in_Estimated_Price_Item_Std4_Local'

DF_IN_SIN_SR_AP1                = 'df_in_Sell_In_FCST_GI_Split_Ratio_AP1'
DF_IN_SIN_SR_AP2                = 'df_in_Sell_In_FCST_GI_Split_Ratio_AP2'
DF_IN_SIN_SR_GC                 = 'df_in_Sell_In_FCST_GI_Split_Ratio_GC'
DF_IN_SIN_SR_LOCAL              = 'df_in_Sell_In_FCST_GI_Split_Ratio_Local'

DF_IN_SOUT_SR_AP1               = 'df_in_Sell_Out_FCST_Split_Ratio_AP1'
DF_IN_SOUT_SR_AP2               = 'df_in_Sell_Out_FCST_Split_Ratio_AP2'
DF_IN_SOUT_SR_GC                = 'df_in_Sell_Out_FCST_Split_Ratio_GC'
DF_IN_SOUT_SR_LOCAL             = 'df_in_Sell_Out_FCST_Split_Ratio_Local'


# ---------- OUTPUT DF KEYS ----------
DF_OUT_SIN_DUMMY                = 'Output_SIn_Dummy'
DF_OUT_SOUT_DUMMY               = 'Output_SOut_Dummy'
DF_OUT_FLOORING_DUMMY           = 'Output_Flooring_Dummy'

DF_OUT_SIN_ASSORT               = 'Output_SIn_Assortment'
DF_OUT_SOUT_ASSORT              = 'Output_SOut_Assortment'
DF_OUT_FLOORING_ASSORT          = 'Output_Flooring_Assortment'      # (Output 2-3) Flooring용 Assortment

DF_OUT_SIN_GI_AP1               = 'df_output_Sell_In_FCST_GI_AP1'
DF_OUT_SIN_GI_AP2               = 'df_output_Sell_In_FCST_GI_AP2'
DF_OUT_SIN_GI_GC                = 'df_output_Sell_In_FCST_GI_GC'
DF_OUT_SIN_GI_LOCAL             = 'df_output_Sell_In_FCST_GI_Local'

DF_OUT_SOUT_AP1                 = 'df_output_Sell_Out_FCST_AP1'
DF_OUT_SOUT_AP2                 = 'df_output_Sell_Out_FCST_AP2'
DF_OUT_SOUT_GC                  = 'df_output_Sell_Out_FCST_GC'
DF_OUT_SOUT_LOCAL               = 'df_output_Sell_Out_FCST_Local'

DF_OUT_FLOORING_FCST            = 'df_output_Flooring_FCST'         # (Output 3-3) Flooring FCST (VD)
DF_OUT_BO_FCST                  = 'df_output_BO_FCST'  # (추후 스펙)

DF_OUT_EST_PRICE_LOCAL          = 'df_output_Estimated_Price_Local'

DF_OUT_SIN_SR_AP1               = 'df_output_Sell_In_FCST_GI_Split_Ratio_AP1'
DF_OUT_SIN_SR_AP2               = 'df_output_Sell_In_FCST_GI_Split_Ratio_AP2'
DF_OUT_SIN_SR_GC                = 'df_output_Sell_In_FCST_GI_Split_Ratio_GC'
DF_OUT_SIN_SR_LOCAL             = 'df_output_Sell_In_FCST_GI_Split_Ratio_Local'

DF_OUT_SOUT_SR_AP1              = 'df_output_Sell_Out_FCST_Split_Ratio_AP1'
DF_OUT_SOUT_SR_AP2              = 'df_output_Sell_Out_FCST_Split_Ratio_AP2'
DF_OUT_SOUT_SR_GC               = 'df_output_Sell_Out_FCST_Split_Ratio_GC'
DF_OUT_SOUT_SR_LOCAL            = 'df_output_Sell_Out_FCST_Split_Ratio_Local'


# ========================
# NEW: Input DataFrames. 2025.11.07
# ========================
DF_IN_SIN_FCST    = 'df_in_sin_fcst'     # (Input 23) S/In FCST(GI) 현행값 비교용
DF_IN_SOUT_FCST   = 'df_in_sout_fcst'    # (Input 24) S/Out FCST 현행값 비교용
DF_IN_FLOOR_FCST  = 'df_in_floor_fcst'   # (Input 25) Flooring FCST 현행값 비교용


# ========================
# NEW: Input DataFrames. 2025.11.17
# ========================
DF_IN_EP_SALES_STD2_ITEM_STD4_LOCAL = 'df_in_Estimated_Price_Sales_Std2_Item_Std4_Local'
DF_IN_EP_SALES_STD2_ITEM_STD3_LOCAL = 'df_in_Estimated_Price_Sales_Std2_Item_Std3_Local'
DF_IN_EP_SALES_STD2_ITEM_STD2_LOCAL = 'df_in_Estimated_Price_Sales_Std2_Item_Std2_Local'



# 25.11.24 추가 상수 (이미 있으면 중복 정의 금지)
COL_PROD_GROUP          = 'Item.[Product Group]'
COL_SOUT_MASTER_STATUS  = 'S/Out Master Status'
# COL_STD5                = 'Sales Domain.[Sales Std5]'   # SDD 안에 있을 것으로 가정

DF_IN_SOUT_SIMUL_MASTER = 'df_in_Sell_Out_Simul_Master'

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
    """
    PYSalesProductASNDeltaB2C: 입력 DF 적재 + 타입 표준화
      - 차원(prefix): "Version.", "Sales Domain", "Item.", "Location.", "Time." → category
      - 측정치(더미/가격/환율/Ratio/AP 가격): float32 (NaN 보존)
      - 진짜 정수 코드값만 int32
    전역 dict: input_dataframes 에 적재
    """    
    
    # -----------------------------
    # 0) 파일명 ↔ DF_KEY 매핑
    #    (로컬에서 CSV를 읽을 때 파일명은 아래 키와 동일해야 함: <키>.csv)
    # -----------------------------

    file_to_df_mapping = {
        f'{DF_IN_SIN_DUMMY     }.csv': DF_IN_SIN_DUMMY,
        f'{DF_IN_SOUT_DUMMY    }.csv': DF_IN_SOUT_DUMMY,
        f'{DF_IN_FLOORING_DUMMY}.csv': DF_IN_FLOORING_DUMMY,

        f'{DF_IN_SDD       }.csv': DF_IN_SDD,
        f'{DF_IN_TIME_PW   }.csv': DF_IN_TIME_PW,
        f'{DF_IN_TIME_W    }.csv': DF_IN_TIME_W,
        f'{DF_IN_ITEM_MST  }.csv': DF_IN_ITEM_MST,
        f'{DF_IN_ESTORE    }.csv': DF_IN_ESTORE,

        f'{DF_IN_EST_PRICE    }.csv': DF_IN_EST_PRICE,
        f'{DF_IN_AP_PRICE     }.csv': DF_IN_AP_PRICE,
        f'{DF_IN_EXRATE_LOCAL }.csv': DF_IN_EXRATE_LOCAL,
        f'{DF_IN_EP_STD2_LOCAL}.csv': DF_IN_EP_STD2_LOCAL,
        f'{DF_IN_EP_STD3_LOCAL}.csv': DF_IN_EP_STD3_LOCAL,
        f'{DF_IN_EP_STD4_LOCAL}.csv': DF_IN_EP_STD4_LOCAL,

        f'{DF_IN_SIN_SR_AP1  }.csv': DF_IN_SIN_SR_AP1,
        f'{DF_IN_SIN_SR_AP2  }.csv': DF_IN_SIN_SR_AP2,
        f'{DF_IN_SIN_SR_GC   }.csv': DF_IN_SIN_SR_GC,
        f'{DF_IN_SIN_SR_LOCAL}.csv': DF_IN_SIN_SR_LOCAL,

        f'{DF_IN_SOUT_SR_AP1  }.csv': DF_IN_SOUT_SR_AP1,
        f'{DF_IN_SOUT_SR_AP2  }.csv': DF_IN_SOUT_SR_AP2,
        f'{DF_IN_SOUT_SR_GC   }.csv': DF_IN_SOUT_SR_GC,
        f'{DF_IN_SOUT_SR_LOCAL}.csv': DF_IN_SOUT_SR_LOCAL,

        # 2025.11.07
        f'{DF_IN_SIN_FCST  }.csv': DF_IN_SIN_FCST,
        f'{DF_IN_SOUT_FCST }.csv': DF_IN_SOUT_FCST,
        f'{DF_IN_FLOOR_FCST}.csv': DF_IN_FLOOR_FCST,
        
        # 2025.11.17
        f'{DF_IN_EP_SALES_STD2_ITEM_STD4_LOCAL  }.csv': DF_IN_EP_SALES_STD2_ITEM_STD4_LOCAL,
        f'{DF_IN_EP_SALES_STD2_ITEM_STD3_LOCAL  }.csv': DF_IN_EP_SALES_STD2_ITEM_STD3_LOCAL,
        f'{DF_IN_EP_SALES_STD2_ITEM_STD2_LOCAL  }.csv': DF_IN_EP_SALES_STD2_ITEM_STD2_LOCAL,
    
        # 2025.11.24
        f'{DF_IN_SOUT_SIMUL_MASTER  }.csv': DF_IN_SOUT_SIMUL_MASTER
        
    }

    def read_csv_with_fallback(filepath: str) -> pd.DataFrame:
        for enc in ('utf-8-sig', 'utf-8', 'cp949'):
            try:
                return pd.read_csv(filepath, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to read file {filepath} with tried encodings.")

    # -----------------------------
    # 1) 로컬 / o9 분기
    # -----------------------------
    if is_local:
        # 출력 폴더 정리
        for file in os.scandir(str_output_dir):
            try:
                os.remove(file.path)
            except Exception:
                pass

        # 입력 CSV 적재
        for file in glob.glob(f"{os.getcwd()}/{str_input_dir}/*.csv"):
            df = read_csv_with_fallback(file)
            file_name = os.path.splitext(os.path.basename(file))[0]
            for fname, df_key in file_to_df_mapping.items():
                if file_name == os.path.splitext(fname)[0]:
                    input_dataframes[df_key] = df
                    break
    else:
        # o9 런타임: 외부에서 주입된 변수 바인딩
        input_dataframes[DF_IN_SIN_DUMMY     ] = df_in_sin_fcst_dummy
        input_dataframes[DF_IN_SOUT_DUMMY    ] = df_in_sout_fcst_dummy
        input_dataframes[DF_IN_FLOORING_DUMMY] = df_in_flooring_fcst_dummy

        input_dataframes[DF_IN_SDD      ] = df_in_Sales_Domain_Dimension
        input_dataframes[DF_IN_TIME_PW  ] = df_in_Time_pw
        input_dataframes[DF_IN_TIME_W   ] = df_in_Time_w
        input_dataframes[DF_IN_ITEM_MST ] = df_in_Item_Master
        input_dataframes[DF_IN_ESTORE   ] = df_in_Sales_Domain_Estore

        input_dataframes[DF_IN_EST_PRICE    ] = df_in_Estimated_Price
        input_dataframes[DF_IN_AP_PRICE     ] = df_in_Action_Plan_Price
        input_dataframes[DF_IN_EXRATE_LOCAL ] = df_in_Exchange_Rate_Local
        input_dataframes[DF_IN_EP_STD2_LOCAL] = df_in_Estimated_Price_Item_Std2_Local
        input_dataframes[DF_IN_EP_STD3_LOCAL] = df_in_Estimated_Price_Item_Std3_Local
        input_dataframes[DF_IN_EP_STD4_LOCAL] = df_in_Estimated_Price_Item_Std4_Local

        input_dataframes[DF_IN_SIN_SR_AP1  ] = df_in_Sell_In_FCST_GI_Split_Ratio_AP1
        input_dataframes[DF_IN_SIN_SR_AP2  ] = df_in_Sell_In_FCST_GI_Split_Ratio_AP2
        input_dataframes[DF_IN_SIN_SR_GC   ] = df_in_Sell_In_FCST_GI_Split_Ratio_GC
        input_dataframes[DF_IN_SIN_SR_LOCAL] = df_in_Sell_In_FCST_GI_Split_Ratio_Local

        input_dataframes[DF_IN_SOUT_SR_AP1  ] = df_in_Sell_Out_FCST_Split_Ratio_AP1
        input_dataframes[DF_IN_SOUT_SR_AP2  ] = df_in_Sell_Out_FCST_Split_Ratio_AP2
        input_dataframes[DF_IN_SOUT_SR_GC   ] = df_in_Sell_Out_FCST_Split_Ratio_GC
        input_dataframes[DF_IN_SOUT_SR_LOCAL] = df_in_Sell_Out_FCST_Split_Ratio_Local

        # 2025.11.07
        input_dataframes[DF_IN_SIN_FCST  ] = df_in_sin_fcst
        input_dataframes[DF_IN_SOUT_FCST ] = df_in_sout_fcst
        input_dataframes[DF_IN_FLOOR_FCST] = df_in_floor_fcst

        # 2025.11.17
        input_dataframes[DF_IN_EP_SALES_STD2_ITEM_STD4_LOCAL] = df_in_Estimated_Price_Sales_Std2_Item_Std4_Local
        input_dataframes[DF_IN_EP_SALES_STD2_ITEM_STD3_LOCAL] = df_in_Estimated_Price_Sales_Std2_Item_Std3_Local
        input_dataframes[DF_IN_EP_SALES_STD2_ITEM_STD2_LOCAL] = df_in_Estimated_Price_Sales_Std2_Item_Std2_Local

        # 2025.11.24
        input_dataframes[DF_IN_SOUT_SIMUL_MASTER] = df_in_Sell_Out_Simul_Master

    # -----------------------------
    # 2) 차원 컬럼: category 로 통일
    # -----------------------------
    dim_prefixes = ("Version.", "Sales Domain", "Item.", "Location.", "Time.")
    for key, df in list(input_dataframes.items()):
        if df is None or df.empty:
            continue
        # prefix 매칭되는 컬럼은 우선 str로 만든 뒤 category 로
        for p in dim_prefixes:
            fn_convert_type(df, p, str)
        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = df[c].astype(str).astype("category")

    # -----------------------------
    # 3) 측정치 컬럼: float32 로 통일(결측 보존)
    # -----------------------------
    # 더미/가격/환율/AP가격/Ratio 들을 커버하는 시작 문자열들
    meas_starts = (
        "S/In FCST",              # e.g. S/In FCST(GI) Dummy_*, S/In FCST(GI)_AP1 ...
        "S/Out FCST",             # e.g. S/Out FCST Dummy_*, S/Out FCST_AP1 ...
        "Flooring FCST",          # e.g. Flooring FCST Dummy
        "Estimated Price",        # e.g. Estimated Price Modify_Local, Estimated Price_Local
        "Exchange Rate",          # e.g. Exchange Rate_Local
        "Action Plan Price",      # e.g. Action Plan Price_USD
        "Split Ratio"             # 안전빵: 혹시 접두가 'Split Ratio' 로만 오는 경우
    )

    def _cast_measures_to_float32(df: pd.DataFrame):
        if df is None or df.empty:
            return
        cols = [c for c in df.columns if c.startswith(meas_starts)]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    for key, df in list(input_dataframes.items()):
        _cast_measures_to_float32(df)

    # -----------------------------
    # 4) 정수 코드값만 int32, float 는 유지
    #    (공통 유틸: object→category, int→int32, float는 NaN 보존)
    # -----------------------------
    fn_prepare_input_types(input_dataframes)

    # -----------------------------
    # 5) 주요 DF 간단 로그(선택)
    # -----------------------------


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


# ======================================================
# 공용: 차원형(category) 캐스팅 + 컬럼 순서 정리 유틸
# ======================================================
def _cast_dims_category(df: pd.DataFrame, include_location: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = [COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if include_location and (COL_LOCATION in df.columns) else [])
    for c in cols:
        if c in df.columns:
            try:
                if not pd.api.types.is_categorical_dtype(df[c]):
                    df[c] = df[c].astype('category')
            except Exception:
                df[c] = df[c].astype('string')
    return df
# ======================================================
# 유틸: salesItemLocation 파싱
# ------------------------------------------------------
#   "400001:RF29BB8600QLAA^400002:RF29BB8600QLAA"
#   "400001:RF29BB8600QLAA:S001^400002:RF29BB8600QLAA:S001"
# ======================================================
def gfn_parse_sales_item_location(salesItemLocation: str) -> Tuple[pd.DataFrame, bool]:
    rows, has_loc = [], False
    if not isinstance(salesItemLocation, str) or not salesItemLocation.strip():
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM]), False

    for token in salesItemLocation.split('^'):
        parts = token.strip().split(':')
        if len(parts) == 2:
            ship, item = parts
            rows.append({COL_SHIP_TO: ship, COL_ITEM: item})
        elif len(parts) == 3:
            ship, item, loc = parts
            rows.append({COL_SHIP_TO: ship, COL_ITEM: item, COL_LOCATION: loc})
            has_loc = True
        # 그 외 토큰은 무시

    df = pd.DataFrame(rows)
    if df.empty:
        return df, False
    df = _cast_dims_category(df, has_loc)
    return df, has_loc


#####################################################
#################### Start Step Functions  ##########
#####################################################
# ======================================================
# Step 01: 생성할 Dummy 선별 (+ 삭제용 Output 구성)
# ======================================================

# ---------- 유틸 ----------
def _coerce_dims(
    df: pd.DataFrame,
    cols: list[str]
) -> None:
    for c in cols:
        if c in df.columns and df[c].dtype.name != 'category':
            df[c] = df[c].astype('category')

def _to_float32(
    df: pd.DataFrame,
    cols: list[str]
) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')


# ======================================================
# Step 1-0) Sales 선정 (salesItemLocation → eStore 제거 → SDD 유효 ShipTo 필터)
#   반환: (df_pairs, pairs_have_loc)
#   Main에서: df_step01_00_sales_pairs, pairs_have_loc = fn_step01_00_select_sales(...)
# ======================================================
@_decoration_
def fn_step01_00_select_sales(
    df_in_sdd: pd.DataFrame,
    df_in_estore: pd.DataFrame,
    salesItemLocation: str,
    **kwargs
) -> Tuple[pd.DataFrame, bool]:

    df_pairs, has_loc = gfn_parse_sales_item_location(salesItemLocation)

    # eStore ShipTo 제거
    if df_in_estore is not None and not df_in_estore.empty and (COL_SHIP_TO in df_in_estore.columns):
        estore_set = set(df_in_estore[COL_SHIP_TO].astype(str).unique())
        df_pairs = df_pairs[~df_pairs[COL_SHIP_TO].astype(str).isin(estore_set)].copy()

    # SDD에 존재하는 ShipTo만 남김(유효성)
    if df_in_sdd is not None and not df_in_sdd.empty and (COL_SHIP_TO in df_in_sdd.columns):
        sdd_set = set(df_in_sdd[COL_SHIP_TO].astype(str).unique())
        df_pairs = df_pairs[df_pairs[COL_SHIP_TO].astype(str).isin(sdd_set)].copy()
    
    df_pairs = _cast_dims_category(df_pairs, has_loc)
    return df_pairs, has_loc

# ======================================================
# 유틸: salesItemLocation 페어 기반 더미 필터 (Vectorized, Semi-Join)
#   - df_dummy     : 더미 원본 DF (S/In, S/Out, Flooring 공통)
#   - pairs_df     : Step 1-0 결과 (COL_SHIP_TO, COL_ITEM, [COL_LOCATION?])
#   - pairs_have_loc: salesItemLocation에 Location 포함 여부
#   - note: Location은 "필터링 키"로만 사용, 반환 DF에서 제거하지 않음
# ======================================================
def _filter_dummy_by_pairs(
    df_dummy: pd.DataFrame,
    pairs_df: pd.DataFrame,
    pairs_have_loc: bool
) -> pd.DataFrame:

    if df_dummy is None or df_dummy.empty:
        return df_dummy if df_dummy is not None else pd.DataFrame()
    if pairs_df is None or pairs_df.empty:
        # 입력이 없으면 결과는 공집합(규격 유지)
        return df_dummy.iloc[0:0].copy()

    # --------- 조인 키 구성 ---------
    on_cols = [COL_SHIP_TO, COL_ITEM]
    if pairs_have_loc and (COL_LOCATION in df_dummy.columns) and (COL_LOCATION in pairs_df.columns):
        on_cols.append(COL_LOCATION)

    # 두 DF 모두에 존재하는 키만 사용(안전)
    on_cols = [c for c in on_cols if (c in df_dummy.columns) and (c in pairs_df.columns)]
    if not on_cols:
        # 키가 없으면 필터 불가 → 빈 DF 반환(규격 유지)
        return df_dummy.iloc[0:0].copy()

    # --------- dtype 정렬(경량) ---------
    # 대용량 고려: 가능하면 pairs_df 쪽을 df_dummy dtype에 맞춤
    for c in on_cols:
        if df_dummy[c].dtype != pairs_df[c].dtype:
            try:
                pairs_df = pairs_df.assign(**{c: pairs_df[c].astype(df_dummy[c].dtype, copy=False)})
            except Exception:
                # 최후 수단: 문자열 통일
                df_dummy = df_dummy.assign(**{c: df_dummy[c].astype('string')})
                pairs_df = pairs_df.assign(**{c: pairs_df[c].astype('string')})

    # --------- 세미 조인(inner merge)로 필터 ---------
    keys = pairs_df[on_cols].drop_duplicates()
    keys = keys.assign(_hit_=1)
    filtered = (
        df_dummy.merge(keys, on=on_cols, how='inner', sort=False, copy=False)
                .drop(columns=['_hit_'], errors='ignore')
    )
    # 주의: Location은 반환 DF에서 제거하지 않는다(그대로 유지)
    return filtered


# ======================================================
# 유틸: 현행 FCST(>=0) 존재 조합 제외 (Vectorized, Anti-Join)
#   - df_pick   : 페어/MeasureLv 적용 후의 더미 후보
#   - fcst_df   : 현행 FCST DF (STR_DF_IN_*_FCST)
#   - fcst_col  : 현행 FCST 수치 컬럼명 (예: 'S/In FCST(GI)_AP2', 'S/Out FCST_AP2', 'Flooring FCST')
#   - time_col  : 시간축 컬럼 (COL_PW 또는 COL_WEEK)
#   - pairs_have_loc : salesItemLocation에 Location 포함 여부
#   - 반환: 현행 FCST가 0 이상 존재하는 (ShipTo, Item, [Location], Time) 조합을 제거한 DF
#   - 주의: Location 컬럼이 df_pick에 있더라도 "키로는" pairs_have_loc가 True일 때만 사용
#           (Location 없는 입력은 ShipTo*Item 기준으로 배제하는 스펙에 부합)
# ======================================================
def _exclude_existing_fcst(
    df_pick: pd.DataFrame,
    fcst_df: pd.DataFrame,
    fcst_col: str,
    # time_col: str,
    pairs_have_loc: bool
) -> pd.DataFrame:

    if df_pick is None or df_pick.empty:
        return df_pick if df_pick is not None else pd.DataFrame()
    if (fcst_df is None) or fcst_df.empty or (fcst_col not in fcst_df.columns):
        return df_pick

    # --------- 조인 키 결정 ---------
    on_cols = [COL_SHIP_TO, COL_ITEM]
    if pairs_have_loc and (COL_LOCATION in df_pick.columns) and (COL_LOCATION in fcst_df.columns):
        on_cols.append(COL_LOCATION)
    # on_cols.append(time_col)

    # 두 DF 모두에 존재하는 키만 사용
    on_cols = [c for c in on_cols if (c in df_pick.columns) and (c in fcst_df.columns)]
    # if time_col not in on_cols:
    #     # 시간축 불일치 → 비교 불가
    #     return df_pick

    # --------- dtype 정렬(경량) ---------
    for c in on_cols:
        if df_pick[c].dtype != fcst_df[c].dtype:
            try:
                fcst_df = fcst_df.assign(**{c: fcst_df[c].astype(df_pick[c].dtype, copy=False)})
            except Exception:
                df_pick = df_pick.assign(**{c: df_pick[c].astype('string')})
                fcst_df = fcst_df.assign(**{c: fcst_df[c].astype('string')})

    # --------- 현행 FCST>=0 마스크 (완전 벡터화) ---------
    s = fcst_df[fcst_col]
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors='coerce')
    mask = s.ge(0)  # NaN → False

    if not bool(mask.any()):
        return df_pick

    fcst_keys = fcst_df.loc[mask, on_cols].drop_duplicates()
    fcst_keys = fcst_keys.assign(_hit_=1)

    # --------- Anti-join (left merge 후 미히트만 유지) ---------
    merged = df_pick.merge(fcst_keys, on=on_cols, how='left', sort=False, copy=False)
    out = merged[merged['_hit_'].isna()].drop(columns=['_hit_'])

    # Location 컬럼은 제거하지 않음(원본 유지)
    return out

# ======================================================
# 보조: Measure Lv 접미사/컬럼명 매핑
# ======================================================
# MEASURE_MAP = {'ap1': 'AP1', 'ap2': 'AP2', 'gc': 'GC', 'local': 'Local'}
def _suffix_for(measureLv: str) -> str:
    lv = (measureLv or '').strip().upper()
    return MEASURE_MAP.get(lv, lv)  # fallback: 그대로 사용

def _cols_for_measure(measureLv: str):
    suf = _suffix_for(measureLv)
    # 더미 컬럼
    sin_dummy_col  = f"S/In FCST(GI) Dummy_{suf}"
    sout_dummy_col = f"S/Out FCST Dummy_{suf}"
    floor_dummy_col = "Flooring FCST Dummy"   # Flooring Dummy는 단일 컬럼 가정
    # 현행 FCST 비교 컬럼
    sin_fcst_col   = f"S/In FCST(GI)_{suf}"
    sout_fcst_col  = f"S/Out FCST_{suf}"
    floor_fcst_col = "Flooring FCST"
    return sin_dummy_col, sout_dummy_col, floor_dummy_col, sin_fcst_col, sout_fcst_col, floor_fcst_col


# ======================================================
# Step 1-1) S/In 더미에서 생성할 Sales 선정
#   - process:
#       (1) measureLv에 맞는 더미 컬럼만 남김
#       (2) salesItemLocation 페어로 필터 (Location은 키에서만 사용, 반환 시 유지)
#       (3) 현행 FCST(>=0) 존재 조합 제외(anti-join)
# ======================================================
@_decoration_
def fn_step01_01_pick_sin_dummy(
    df_in_sin_dummy: pd.DataFrame,
    pairs_df: pd.DataFrame,
    pairs_have_loc: bool,
    measureLv: str,
    df_in_sin_fcst: pd.DataFrame
) -> pd.DataFrame:

    sin_dummy_col, _, _, sin_fcst_col, _, _ = _cols_for_measure(measureLv)

    if df_in_sin_dummy is None or df_in_sin_dummy.empty:
        return pd.DataFrame(columns=[
            COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW, sin_dummy_col
        ])

    # (1) measure 컬럼만 남김(필요 컬럼 + 대상 measure)
    base_cols = [COL_SHIP_TO, COL_ITEM, COL_PW]
    if COL_LOCATION in df_in_sin_dummy.columns:
        base_cols.append(COL_LOCATION)
    keep_cols = [c for c in base_cols + [sin_dummy_col] if c in df_in_sin_dummy.columns]
    work = df_in_sin_dummy[keep_cols]

    # (2) 페어 필터
    work = _filter_dummy_by_pairs(work, pairs_df, pairs_have_loc)

    # (3) 현행 FCST 존재 제외
    work = _exclude_existing_fcst(
        work, df_in_sin_fcst, sin_fcst_col, pairs_have_loc
    )

    return work

@_decoration_
def fn_step01_02_build_output_sin_dummy_delete(
    df_pick: pd.DataFrame, 
    measureLv: str,
    out_version: str
) -> pd.DataFrame:
    sin_dummy_col, *_ = _cols_for_measure(measureLv)
    if df_pick is None or df_pick.empty:
        return df_pick if df_pick is not None else pd.DataFrame()
    out = df_pick.copy()
    # NaN 처리: dtype 유지 위해 None 할당 → float/object는 NaN으로 인식
    out[sin_dummy_col] = np.nan
    out.insert(0, COL_VERSION, out_version)
    out[COL_VERSION] = out[COL_VERSION].astype('category')
    return out

# ======================================================
# Step 1-3) S/Out 더미에서 생성할 Sales 선정
# ======================================================
@_decoration_
def fn_step01_03_pick_sout_dummy(
    df_in_sout_dummy: pd.DataFrame,
    pairs_df: pd.DataFrame,
    pairs_have_loc: bool,
    measureLv: str,
    df_in_sout_fcst: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:

    _, sout_dummy_col, _, _, sout_fcst_col, _ = _cols_for_measure(measureLv)

    if df_in_sout_dummy is None or df_in_sout_dummy.empty:
        return pd.DataFrame(columns=[
            COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW, sout_dummy_col
        ])

    base_cols = [COL_SHIP_TO, COL_ITEM, COL_PW]
    if COL_LOCATION in df_in_sout_dummy.columns:
        base_cols.append(COL_LOCATION)
    keep_cols = [c for c in base_cols + [sout_dummy_col] if c in df_in_sout_dummy.columns]
    work = df_in_sout_dummy[keep_cols]

    work = _filter_dummy_by_pairs(work, pairs_df, pairs_have_loc)

    work = _exclude_existing_fcst(
        work, df_in_sout_fcst, sout_fcst_col, pairs_have_loc
    )

    return work

@_decoration_
def fn_step01_04_build_output_sout_dummy_delete(
    df_pick: pd.DataFrame, 
    measureLv: str,
    out_version: str
) -> pd.DataFrame:
    _, sout_dummy_col, *_ = _cols_for_measure(measureLv)
    if df_pick is None or df_pick.empty:
        return df_pick if df_pick is not None else pd.DataFrame()
    out = df_pick.copy()
    out[sout_dummy_col] = np.nan
    out.insert(0, COL_VERSION, out_version)
    out[COL_VERSION] = out[COL_VERSION].astype('category')
    return out

# ======================================================
# Step 1-5) Flooring 더미에서 생성할 Sales 선정
#   - Flooring은 Week 축 사용
#   - 현행 FCST 비교는 STR_DF_IN_FLOOR_FCST의 'Flooring FCST' 기준
# ======================================================
@_decoration_
def fn_step01_05_pick_flooring_dummy(
    df_in_floor_dummy: pd.DataFrame,
    pairs_df: pd.DataFrame,
    pairs_have_loc: bool,
    measureLv: str,  # 스펙상 "존재 시 제외" 판정에 measureLv 문맥 포함 → 비교 DF는 고정 컬럼명
    df_in_floor_fcst: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:

    # floor 더미/FCST 컬럼
    _, _, floor_dummy_col, _, _, floor_fcst_col = _cols_for_measure(measureLv)

    if df_in_floor_dummy is None or df_in_floor_dummy.empty:
        return pd.DataFrame(columns=[
            COL_SHIP_TO, COL_ITEM, COL_WEEK, floor_dummy_col
        ])

    base_cols = [COL_SHIP_TO, COL_ITEM, COL_WEEK]
    if COL_LOCATION in df_in_floor_dummy.columns:
        base_cols.append(COL_LOCATION)
    keep_cols = [c for c in base_cols + [floor_dummy_col] if c in df_in_floor_dummy.columns]
    work = df_in_floor_dummy[keep_cols]

    work = _filter_dummy_by_pairs(work, pairs_df, pairs_have_loc)

    work = _exclude_existing_fcst(
        work, df_in_floor_fcst, floor_fcst_col, pairs_have_loc
    )

    return work


@_decoration_
def fn_step01_06_build_output_flooring_dummy_delete(
    df_pick: pd.DataFrame, 
    measureLv: str,
    out_version: str
) -> pd.DataFrame:
    *_, floor_dummy_col, _, _, _ = _cols_for_measure(measureLv)
    if df_pick is None or df_pick.empty:
        return df_pick if df_pick is not None else pd.DataFrame()
    out = df_pick.copy()
    out[floor_dummy_col] = np.nan
    out.insert(0, COL_VERSION, out_version)
    out[COL_VERSION] = out[COL_VERSION].astype('category')
    return out





def _order_assort_cols(df: pd.DataFrame, assort_col: str, include_location: bool) -> pd.DataFrame:
    base = [COL_VERSION, COL_SHIP_TO, COL_ITEM]
    if include_location and (COL_LOCATION in df.columns):
        base.append(COL_LOCATION)
    base.append(assort_col)
    exist = [c for c in base if c in df.columns]
    other = [c for c in df.columns if c not in exist]
    return df[exist + other]

# ======================================================
# 보조: measureLv별 Assortment 컬럼명
#  - _suffix_for(measureLv) 는 Step01에서 사용하던 동일 함수 사용 (AP1/AP2/GC/Local → 대문자)
# ======================================================
def _assort_cols_for_measure(measureLv: str):
    suf = _suffix_for(measureLv)  # 'AP1'|'AP2'|'GC'|'Local'
    sin_dummy_col   = f"S/In FCST(GI) Dummy_{suf}"
    sout_dummy_col  = f"S/Out FCST Dummy_{suf}"
    floor_dummy_col = "Flooring FCST Dummy"

    sin_assort_col   = f"S/In FCST(GI) Assortment_{suf}"
    sout_assort_col  = f"S/Out FCST Assortment_{suf}"
    floor_assort_col = "Flooring FCST Assortment"
    return (sin_dummy_col, sout_dummy_col, floor_dummy_col,
            sin_assort_col, sout_assort_col, floor_assort_col)

# ======================================================
# Step 2-1) S/In FCST(GI) Dummy → Assortment
#   입력: df_step01_01_sin_pick (Step1-1 결과; PW 축 포함)
#   출력: [ShipTo, Item, (Location)], S/In FCST(GI) Assortment_APx
#   처리: Dummy→Assortment rename, (0→1 변환), Time(PW) 축 제거(집계), dtype 정리
# ======================================================
@_decoration_
def fn_step02_01_build_sin_assortment(
    df_sin_pick: pd.DataFrame,
    measureLv: str
) -> pd.DataFrame:    
    sin_dummy_col, *_ , sin_assort_col, _, _ = _assort_cols_for_measure(measureLv)

    if df_sin_pick is None or df_sin_pick.empty or (sin_dummy_col not in df_sin_pick.columns):
        cols = [COL_VERSION,COL_SHIP_TO, COL_ITEM]
        if COL_LOCATION in (df_sin_pick.columns if df_sin_pick is not None else []):
            cols.append(COL_LOCATION)
        cols.append(sin_assort_col)
        return pd.DataFrame(columns=cols)

    # 필요 컬럼만 취득
    dim_cols = [COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if COL_LOCATION in df_sin_pick.columns else [])
    use_cols = [c for c in (dim_cols + [sin_dummy_col, COL_PW]) if c in df_sin_pick.columns]
    work = df_sin_pick[use_cols].copy()

    # Rename Dummy → Assortment
    work.rename(columns={sin_dummy_col: sin_assort_col}, inplace=True)

    # 25.11.13 변경
    #   - Dummy 값이 1이 아닌 0으로 들어오는 구조
    #   - Assortment 관점에서는 "행이 존재하면 1" 로 간주
    #   - 즉, 0/기타 non-null → 1, NaN은 그대로 유지
    if sin_assort_col in work.columns:
        work[sin_assort_col] = np.where(work[sin_assort_col].notna(), 1, np.nan)

    # Time 축 제거 및 중복 제거 대신,
    # ultra_fast_groupby_numpy_general 로 ShipTo*Item*(Location) 단위 max 집계
    work = ultra_fast_groupby_numpy_general(
        df=work,
        key_cols=dim_cols,
        aggs={
            sin_assort_col: 'max'
        }
    )

    if COL_VERSION not in work.columns:
        work.insert(0, COL_VERSION, Version)
    # dtype 정리
    _coerce_dims(work, [COL_VERSION,*dim_cols])
    _to_float32(work, [sin_assort_col])

    return work

# ======================================================
# Step 2-2) S/Out FCST Dummy → Assortment
#   입력:
#     - df_sout_pick : Step1-3 결과 (ShipTo*Item*(Location)*PW, Dummy 포함)
#     - measureLv    : 'ap1'/'ap2'/'gc'/'local'
#     - df_item_mst  : Item → Product Group 매핑 테이블          # 25.11.24 변경
#     - df_sout_simul_master : Sell-Out Simul Master (ShipTo/SalesStd5 * Product Group) # 25.11.24 변경
#     - df_sdd       : Sales Std5 → ShipTo 맵핑(SDD), 있으면 사용  # 25.11.24 변경
#
#   출력:
#     - [ShipTo, Item, (Location)], S/Out FCST Assortment_APx
#
#   처리:
#     (1) Dummy→Assortment rename, (0→1 변환)
#     (2) Time(PW) 축 제거(집계), dtype 정리
#         - ShipTo*Item*(Location) 기준 max 집계로 Time(PW) 축 제거
#     (3) Sell-Out Simul Master 기반 필터링 (25.11.24 추가)
#         - ShipTo * Product Group 단위로 S/Out Master Status가 'CON' 인 조합만 유지
#         - 그 외(값이 'CON'이 아니거나 존재하지 않음)는 Assortment를 NaN 으로 변환 

# ======================================================
@_decoration_
def fn_step02_02_build_sout_assortment(
    df_sout_pick        : pd.DataFrame,
    measureLv           : str,
    df_item_mst         : pd.DataFrame = None,   # 25.11.24 추가
    df_sout_simul_master: pd.DataFrame = None,   # 25.11.24 추가
    df_sdd              : pd.DataFrame = None,   # 25.11.24 추가
    **kwargs
) -> pd.DataFrame:
    
    _, sout_dummy_col, _, _, sout_assort_col, _ = _assort_cols_for_measure(measureLv)

    if df_sout_pick is None or df_sout_pick.empty or (sout_dummy_col not in df_sout_pick.columns):
        cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM]
        if df_sout_pick is not None and COL_LOCATION in df_sout_pick.columns:
            cols.append(COL_LOCATION)
        cols.append(sout_assort_col)
        return pd.DataFrame(columns=cols)

    dim_cols = [COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if COL_LOCATION in df_sout_pick.columns else [])
    use_cols = [c for c in (dim_cols + [sout_dummy_col, COL_PW]) if c in df_sout_pick.columns]
    work = df_sout_pick[use_cols].copy()

    # (1) Rename Dummy → Assortment
    work.rename(columns={sout_dummy_col: sout_assort_col}, inplace=True)

    # 25.11.13 변경
    #   - Dummy 값 0(선택) / NaN(미선택) 구조 → Assortment는 선택이면 1로 통일
    if sout_assort_col in work.columns:
        work[sout_assort_col] = np.where(work[sout_assort_col].notna(), 1, np.nan)

    # =====================================================================
    # (2) ShipTo*Item*(Location) 단위로 max 집계 → Time(PW) 축 자연스럽게 제거
    # =====================================================================
    work = ultra_fast_groupby_numpy_general(
        df=work,
        key_cols=dim_cols,
        aggs={
            sout_assort_col: 'max'
        }
    )

    # =====================================================================
    # (3) Sell-Out Simul Master 기반 필터링 (25.11.24 추가)
    #   - df_sout_simul_master: [ShipTo(Std5), Product Group, Status]
    #   - df_item_mst: Item.[Item Std1] 를 Product Group 으로 사용
    #   - df_sdd: ShipTo(leaf) → Sales Std4 매핑
    #   - 최종적으로: work 의 (ShipTo(Std4), Product Group) 이
    #                Simul Master 에서 Status='CON' 인 조합에
    #                하나도 해당되지 않으면 Assortment 를 NaN 처리
    # =====================================================================
    if (
        df_item_mst is not None and not df_item_mst.empty
        and df_sout_simul_master is not None and not df_sout_simul_master.empty
        and df_sdd is not None and not df_sdd.empty
    ):
        # 2-1) Item → Product Group 매핑
        #      df_item_mst 의 Item.[Item Std1] 이 Product Group 역할
        if COL_ITEM_STD1 in df_item_mst.columns:
            item_pg = (
                df_item_mst[[COL_ITEM, COL_ITEM_STD1]]
                .drop_duplicates()
                .rename(columns={COL_ITEM_STD1: COL_PROD_GROUP})  # Std1 → Product Group 으로 사용
            )
            work = work.merge(item_pg, on=COL_ITEM, how='left')
        else:
            # Product Group 기반 필터링 불가 → 기존 로직 그대로 반환
            _coerce_dims(work, dim_cols)
            _to_float32(work, [sout_assort_col])
            return work

        # 2-2) ShipTo(Std5) → ShipTo(Std4) 매핑 + Simul Master 결합
        # df_sdd 예:
        #   Sales Std4   Sales Std5   Ship To
        #   400002       5000002      5000002
        #   400002       400002       400002
        # → leaf ShipTo(=Std5) 기준으로 Std4 를 매핑
        if (
            COL_STD4 in df_sdd.columns
            and COL_SHIP_TO in df_sdd.columns
            and COL_SHIP_TO in df_sout_simul_master.columns
            and COL_PROD_GROUP in df_sout_simul_master.columns
            and COL_SOUT_MASTER_STATUS in df_sout_simul_master.columns
        ):
            # SDD: leaf ShipTo → Sales Std4
            #   ShipTo_Leaf(=Sales Std5 레벨 ShipTo) → ShipTo_Std4
            sdd_map = (
                df_sdd[[COL_STD4, COL_SHIP_TO]]
                .drop_duplicates()
                .rename(columns={
                    COL_SHIP_TO: 'ShipTo_Leaf',
                    COL_STD4   : 'ShipTo_Std4'
                })
            )

            # Simul Master: ShipTo_Leaf(=Sales Std5 ShipTo) * Product Group * Status
            simul_raw = df_sout_simul_master[
                [COL_SHIP_TO, COL_PROD_GROUP, COL_SOUT_MASTER_STATUS]
            ].drop_duplicates().rename(columns={COL_SHIP_TO: 'ShipTo_Leaf'})

            # leaf ShipTo → Std4 ShipTo 매핑
            simul_join = simul_raw.merge(sdd_map, on='ShipTo_Leaf', how='left')

            # Status = 'CON' 인 조합만 허용, (ShipTo_Std4, Product Group) 기준
            simul_join = simul_join[simul_join[COL_SOUT_MASTER_STATUS] == 'CON']
            allowed = simul_join[['ShipTo_Std4', COL_PROD_GROUP]].drop_duplicates()

            # work 의 (ShipTo(Std4), Product Group)이 allowed 에 없으면 Assortment 제거
            work = work.merge(
                allowed.assign(__ok__=1),
                left_on =[COL_SHIP_TO, COL_PROD_GROUP],
                right_on=['ShipTo_Std4', COL_PROD_GROUP],
                how='left'
            )
            work.loc[work['__ok__'].isna(), sout_assort_col] = np.nan
            work.drop(columns=['__ok__', 'ShipTo_Std4'], inplace=True)
        else:
            # 필요한 컬럼이 하나라도 없으면 필터링 생략
            pass


    work = work[dim_cols + [sout_assort_col]]
    if COL_VERSION not in work.columns:
        work.insert(0, COL_VERSION, Version)
        
    _coerce_dims(work, [COL_VERSION,*dim_cols])
    _to_float32(work, [sout_assort_col])

    return work

# ======================================================
# Step 2-3) Flooring FCST Dummy → Assortment  (Week 축)
#   입력: df_step01_05_floor_pick (Week 축 포함)
#   출력: [ShipTo, Item], Flooring FCST Assortment
#   * 스펙 예시엔 Location 없음. 더미에 Location이 있더라도 Assortment는 ShipTo*Item 기준 생성 가정.
# ======================================================
@_decoration_
def fn_step02_03_build_flooring_assortment(
    df_floor_pick: pd.DataFrame,
    measureLv: str   # 인터페이스 일관성 유지용(실제 suffix는 없음)
) -> pd.DataFrame:
    
    *_, floor_dummy_col, _, _, floor_assort_col = _assort_cols_for_measure(measureLv)

    if df_floor_pick is None or df_floor_pick.empty or (floor_dummy_col not in df_floor_pick.columns):
        return pd.DataFrame(columns=[COL_VERSION,COL_SHIP_TO, COL_ITEM, floor_assort_col])

    # Flooring은 ShipTo*Item 기준(예시와 동일). Location은 결과에서 사용하지 않음.
    dim_cols = [COL_SHIP_TO, COL_ITEM]
    use_cols = [c for c in (dim_cols + [floor_dummy_col, COL_WEEK]) if c in df_floor_pick.columns]
    work = df_floor_pick[use_cols].copy()

    # Rename Dummy → Assortment
    work.rename(columns={floor_dummy_col: floor_assort_col}, inplace=True)

    # 25.11.13 변경
    #   - Flooring Dummy도 0(선택) / NaN(미선택)으로 들어옴
    #   - Assortment는 ShipTo*Item 단위로 "있으면 1" 로 세우기
    if floor_assort_col in work.columns:
        work[floor_assort_col] = np.where(work[floor_assort_col].notna(), 1, np.nan)

    # ShipTo*Item 기준으로 max 집계 → Week 축 제거
    work = ultra_fast_groupby_numpy_general(
        df=work,
        key_cols=dim_cols,
        aggs={
            floor_assort_col: 'max'
        }
    )

    if COL_VERSION not in work.columns:
        work.insert(0, COL_VERSION, Version)

    _coerce_dims(work, [COL_VERSION,*dim_cols])
    _to_float32(work, [floor_assort_col])
   
    return work

########################################################################################################################
# Step 3 — FCST 값 생성 (상수명 정합 반영 버전)
########################################################################################################################

########################################################################################################################
# 공통 유틸
########################################################################################################################

def _mk_empty(df_cols: list[str], with_cats: list[str] = None, float_cols: list[str] = None) -> pd.DataFrame:
    df = pd.DataFrame(columns=df_cols)
    with_cats = with_cats or []
    float_cols = float_cols or []
    for c in with_cats:
        df[c] = df[c].astype('category')
    for m in float_cols:
        df[m] = df[m].astype('float32')
    return df

def _unique_keys_from_dummy(
    df_in: pd.DataFrame,
    key_cols: list[str],
    dummy_col: str
) -> pd.DataFrame:
    """
    dummy_col 이 NaN/0 이 아닌 행만 필터 후 key 고유 조합 반환
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=key_cols)

    s_val = pd.to_numeric(df_in[dummy_col], errors='coerce')
    # mask  = (s_val.notna()) & (s_val != 0)
    mask  = (s_val.notna())
    if not mask.any():
        return pd.DataFrame(columns=key_cols)

    df = df_in.loc[mask, key_cols].copy(deep=False).drop_duplicates(ignore_index=True)
    # dtype: category
    for c in key_cols:
        if df[c].dtype.name != 'category':
            df[c] = df[c].astype('category')
    return df

def _expand_by_time(
    df_keys: pd.DataFrame,
    df_time: pd.DataFrame,
    time_col: str
) -> pd.DataFrame:
    """
    (키 DF) × (시간 DF) 카티전 곱
    """
    if df_keys is None or df_keys.empty or df_time is None or df_time.empty:
        base_cols = [] if df_keys is None else list(df_keys.columns)
        return pd.DataFrame(columns=[*base_cols, time_col])

    # 시간축 카테고리 보장
    s_time = df_time[time_col]
    if s_time.dtype.name != 'category':
        s_time = s_time.astype('category')

    df_keys = df_keys.reset_index(drop=True)
    tdf = pd.DataFrame({time_col: s_time.values}).reset_index(drop=True)

    n_key = len(df_keys)
    n_t   = len(tdf)
    rep_keys = df_keys.loc[df_keys.index.repeat(n_t)].reset_index(drop=True)
    tile_t   = pd.concat([tdf]*n_key, ignore_index=True)

    out = pd.concat([rep_keys, tile_t], axis=1)
    out[time_col] = out[time_col].astype('category')
    return out

def _inject_version_and_cast(
    df: pd.DataFrame,
    out_version: str,
    cat_cols: list[str],
    float_cols: list[str]
) -> pd.DataFrame:
    if df is None or df.empty:
        cols = [COL_VERSION, *cat_cols, *float_cols]
        return _mk_empty(cols, with_cats=[COL_VERSION, *cat_cols], float_cols=float_cols)

    df.insert(0, COL_VERSION, out_version)
    # category
    for c in [COL_VERSION, *cat_cols]:
        if df[c].dtype.name != 'category':
            df[c] = df[c].astype('category')
    # float32
    for m in float_cols:
        df[m] = pd.to_numeric(df[m], errors='coerce').astype('float32')
    return df

# -------------------------------------------------------------------------------------------------
# NEW: 전역 measureLv 판정 유틸
# -------------------------------------------------------------------------------------------------
def _is_measure(target: str) -> bool:
    """
    전역변수 measureLv 가 target(ap1|ap2|gc|local)와 일치하면 True
    """
    try:
        return (measureLv or '').strip().lower() == target
    except NameError:
        return False

########################################################################################################################
# Step 3-1-1) S/In FCST(GI)_AP1, S/In FCST(BL)_AP1, S/In FCST(GI) New Model 생성 (Dummy_AP1 기준)
########################################################################################################################
@_decoration_
def fn_step03_01_01_build_sin_fcst_ap1(
    df_sin_pick : pd.DataFrame,   # Step 1-1) 결과
    df_time_pw  : pd.DataFrame,   # Time.[Partial Week] 목록
    out_version : str,
    **kwargs
) -> pd.DataFrame:
    """
    출력 스키마:
    [Version, ShipTo, Item, Location, Time.[Partial Week],
     S/In FCST(GI)_AP1, S/In FCST(BL)_AP1, S/In FCST(GI) New Model]
    값은 모두 0.0
    """
    # measureLv가 ap1이 아니면 즉시 빈 DF 반환
    if not _is_measure('ap1'):
        out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW,
                    COL_SIN_GI_AP1, COL_SIN_BL_AP1, COL_SIN_NEW_MODEL]
        return _mk_empty(out_cols, with_cats=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
                         float_cols=[COL_SIN_GI_AP1, COL_SIN_BL_AP1, COL_SIN_NEW_MODEL])
    
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sin_pick, key_cols, COL_SIN_DUMMY_AP1)
    out_cols = [COL_VERSION, *key_cols, COL_PW, COL_SIN_GI_AP1, COL_SIN_BL_AP1, COL_SIN_NEW_MODEL]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PW],
                         float_cols=[COL_SIN_GI_AP1, COL_SIN_BL_AP1, COL_SIN_NEW_MODEL])

    df = _expand_by_time(keys, df_time_pw, COL_PW)
    df[COL_SIN_GI_AP1]   = 0.0
    df[COL_SIN_BL_AP1]   = 0.0
    df[COL_SIN_NEW_MODEL]= 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
        float_cols=[COL_SIN_GI_AP1, COL_SIN_BL_AP1, COL_SIN_NEW_MODEL]
    )
    return df[out_cols]

########################################################################################################################
# Step 3-1-2) S/In FCST(GI)_AP2, S/In FCST(BL)_AP2 생성 (Dummy_AP2 기준)
########################################################################################################################
@_decoration_
def fn_step03_01_02_build_sin_fcst_ap2(
    df_sin_pick: pd.DataFrame,
    df_time_pw : pd.DataFrame,
    out_version: str,
    **kwargs
) -> pd.DataFrame:
    
    if not _is_measure('ap2'):
        out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW,
                    COL_SIN_GI_AP2, COL_SIN_BL_AP2]
        return _mk_empty(out_cols, with_cats=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
                         float_cols=[COL_SIN_GI_AP2, COL_SIN_BL_AP2])
        
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sin_pick, key_cols, COL_SIN_DUMMY_AP2)
    out_cols = [COL_VERSION, *key_cols, COL_PW, COL_SIN_GI_AP2, COL_SIN_BL_AP2]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PW],
                         float_cols=[COL_SIN_GI_AP2, COL_SIN_BL_AP2])

    df = _expand_by_time(keys, df_time_pw, COL_PW)
    df[COL_SIN_GI_AP2] = 0.0
    df[COL_SIN_BL_AP2] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
        float_cols=[COL_SIN_GI_AP2, COL_SIN_BL_AP2]
    )
    return df[out_cols]

########################################################################################################################
# Step 3-1-3) S/In FCST(GI)_GC, S/In FCST(BL)_GC 생성 (Dummy_GC 기준)
########################################################################################################################
@_decoration_
def fn_step03_01_03_build_sin_fcst_gc(
    df_sin_pick: pd.DataFrame,
    df_time_pw : pd.DataFrame,
    out_version: str,
    **kwargs
) -> pd.DataFrame:
    
    if not _is_measure('gc'):
        out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW,
                    COL_SIN_GC, COL_SIN_BL_GC]
        return _mk_empty(out_cols, with_cats=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
                         float_cols=[COL_SIN_GC, COL_SIN_BL_GC])
        
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sin_pick, key_cols, COL_SIN_DUMMY_GC)
    out_cols = [COL_VERSION, *key_cols, COL_PW, COL_SIN_GC, COL_SIN_BL_GC]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PW],
                         float_cols=[COL_SIN_GC, COL_SIN_BL_GC])

    df = _expand_by_time(keys, df_time_pw, COL_PW)
    df[COL_SIN_GC]    = 0.0
    df[COL_SIN_BL_GC] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
        float_cols=[COL_SIN_GC, COL_SIN_BL_GC]
    )
    return df[out_cols]

########################################################################################################################
# Step 3-1-4) S/In FCST(GI)_Local, S/In FCST(BL)_Local 생성 (Dummy_Local 기준)
########################################################################################################################
@_decoration_
def fn_step03_01_04_build_sin_fcst_local(
    df_sin_pick: pd.DataFrame,
    df_time_pw : pd.DataFrame,
    out_version: str,
    **kwargs
) -> pd.DataFrame:
        
    if not _is_measure('local'):
        out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW,
                    COL_SIN_LOCAL, COL_SIN_BL_LOCAL]
        return _mk_empty(out_cols, with_cats=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
                         float_cols=[COL_SIN_LOCAL, COL_SIN_BL_LOCAL])
        
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sin_pick, key_cols, COL_SIN_DUMMY_LOCAL)
    out_cols = [COL_VERSION, *key_cols, COL_PW, COL_SIN_LOCAL, COL_SIN_BL_LOCAL]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PW],
                         float_cols=[COL_SIN_LOCAL, COL_SIN_BL_LOCAL])

    df = _expand_by_time(keys, df_time_pw, COL_PW)
    df[COL_SIN_LOCAL]    = 0.0
    df[COL_SIN_BL_LOCAL] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
        float_cols=[COL_SIN_LOCAL, COL_SIN_BL_LOCAL]
    )
    return df[out_cols]

########################################################################################################################
# Step 3-2-1) S/Out FCST_AP1 생성 (Dummy_AP1 기준)
########################################################################################################################
@_decoration_
def fn_step03_02_01_build_sout_fcst_ap1(
    df_sout_pick: pd.DataFrame,   # Step 1-3 결과
    df_time_pw  : pd.DataFrame,
    out_version : str,
    **kwargs
) -> pd.DataFrame:

    if not _is_measure('ap1'):
        out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW, COL_SOUT_AP1]
        return _mk_empty(out_cols, with_cats=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
                         float_cols=[COL_SOUT_AP1])    

    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sout_pick, key_cols, COL_SOUT_DUMMY_AP1)
    out_cols = [COL_VERSION, *key_cols, COL_PW, COL_SOUT_AP1]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PW], float_cols=[COL_SOUT_AP1])

    df = _expand_by_time(keys, df_time_pw, COL_PW)
    df[COL_SOUT_AP1] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
        float_cols=[COL_SOUT_AP1]
    )
    return df[out_cols]

########################################################################################################################
# Step 3-2-2) S/Out FCST_AP2 생성 (Dummy_AP2 기준)
########################################################################################################################
@_decoration_
def fn_step03_02_02_build_sout_fcst_ap2(
    df_sout_pick: pd.DataFrame,
    df_time_pw  : pd.DataFrame,
    out_version : str,
    **kwargs
) -> pd.DataFrame:
    if not _is_measure('ap2'):
        out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW, COL_SOUT_AP2]
        return _mk_empty(out_cols, with_cats=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
                         float_cols=[COL_SOUT_AP2])
        
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sout_pick, key_cols, COL_SOUT_DUMMY_AP2)
    out_cols = [COL_VERSION, *key_cols, COL_PW, COL_SOUT_AP2]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PW], float_cols=[COL_SOUT_AP2])

    df = _expand_by_time(keys, df_time_pw, COL_PW)
    df[COL_SOUT_AP2] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
        float_cols=[COL_SOUT_AP2]
    )
    return df[out_cols]

########################################################################################################################
# Step 3-2-3) S/Out FCST_GC 생성 (Dummy_GC 기준)
########################################################################################################################
@_decoration_
def fn_step03_02_03_build_sout_fcst_gc(
    df_sout_pick: pd.DataFrame,
    df_time_pw  : pd.DataFrame,
    out_version : str,
    **kwargs
) -> pd.DataFrame:
    if not _is_measure('gc'):
        out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW, COL_SOUT_GC]
        return _mk_empty(out_cols, with_cats=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
                         float_cols=[COL_SOUT_GC])
        
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sout_pick, key_cols, COL_SOUT_DUMMY_GC)
    out_cols = [COL_VERSION, *key_cols, COL_PW, COL_SOUT_GC]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PW], float_cols=[COL_SOUT_GC])

    df = _expand_by_time(keys, df_time_pw, COL_PW)
    df[COL_SOUT_GC] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
        float_cols=[COL_SOUT_GC]
    )
    return df[out_cols]

########################################################################################################################
# Step 3-2-4) S/Out FCST_Local 생성 (Dummy_Local 기준)
########################################################################################################################
@_decoration_
def fn_step03_02_04_build_sout_fcst_local(
    df_sout_pick: pd.DataFrame,
    df_time_pw  : pd.DataFrame,
    out_version : str,
    **kwargs
) -> pd.DataFrame:
    if not _is_measure('local'):
        out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW, COL_SOUT_LOCAL]
        return _mk_empty(out_cols, with_cats=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
                         float_cols=[COL_SOUT_LOCAL])
        
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sout_pick, key_cols, COL_SOUT_DUMMY_LOCAL)
    out_cols = [COL_VERSION, *key_cols, COL_PW, COL_SOUT_LOCAL]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PW], float_cols=[COL_SOUT_LOCAL])

    df = _expand_by_time(keys, df_time_pw, COL_PW)
    df[COL_SOUT_LOCAL] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PW],
        float_cols=[COL_SOUT_LOCAL]
    )
    return df[out_cols]

########################################################################################################################
# Step 3-3) Flooring FCST 생성 (Flooring Dummy 기준, 시간축: Time.[Week])
########################################################################################################################
@_decoration_
def fn_step03_03_build_flooring_fcst(
    df_floor_pick: pd.DataFrame,   # Step 1-5) 결과
    df_time_w    : pd.DataFrame,   # Time.[Week] 목록
    out_version  : str,
    **kwargs
) -> pd.DataFrame:
    """
    출력 스키마:
    [Version, ShipTo, Item, Time.[Week], Flooring FCST]
    값은 모두 0.0
    """
    key_cols = [COL_SHIP_TO, COL_ITEM]
    keys = _unique_keys_from_dummy(df_floor_pick, key_cols, COL_FLOORING_DUMMY)
    out_cols = [COL_VERSION, *key_cols, COL_WEEK, COL_FLOORING_FCST]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_WEEK], float_cols=[COL_FLOORING_FCST])

    df = _expand_by_time(keys, df_time_w, COL_WEEK)
    df[COL_FLOORING_FCST] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_WEEK],
        float_cols=[COL_FLOORING_FCST]
    )
    return df[out_cols]

########################################################################################################################
# Step 4 — Estimated Price Local Data 생성  (1107버전)
########################################################################################################################
from datetime import date

# ──────────────────────────────────────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────────────────────────────────────

def _pweek_to_month_str(pw: str) -> str:
    """
    'YYYYWWA/B' → ISO 주차의 월 'YYYYMM' 로 변환
    - 예: '202522A' → 2025년 ISO 22주 월요일 날짜 기준 month='202505'
    - 'A/B' 꼬리는 무시, 앞 6자리(YYYYWW) 사용
    """
    if not isinstance(pw, str) or len(pw) < 6:
        return np.nan
    y = int(pw[:4])
    w = int(pw[4:6])
    try:
        d = date.fromisocalendar(y, w, 1)
    except ValueError:
        return np.nan
    return f"{d.year}{d.month:02d}"

# 1107버전추가: 정수 컬럼 지원(NA 허용을 위해 pandas Nullable Int32 사용)
def _mk_empty_cols(cols: list[str], cat_cols: list[str], float_cols: list[str], int_cols: list[str] = None) -> pd.DataFrame:
    df = pd.DataFrame(columns=cols)
    for c in cat_cols:
        df[c] = df[c].astype('category')
    for c in (float_cols or []):
        df[c] = df[c].astype('float32')
    # 1107버전추가
    for c in (int_cols or []):
        df[c] = pd.Series(pd.array([], dtype="Int32"))
    return df

def _expand_s2_by_pweek(df_keys: pd.DataFrame, df_time_pw: pd.DataFrame) -> pd.DataFrame:
    """
    (ShipTo, Item) × PartialWeek 카티전 곱
    """
    if df_keys is None or df_keys.empty or df_time_pw is None or df_time_pw.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_PW])

    base = df_keys[[COL_SHIP_TO, COL_ITEM]].drop_duplicates().reset_index(drop=True)
    if base[COL_SHIP_TO].dtype.name != 'category':
        base[COL_SHIP_TO] = base[COL_SHIP_TO].astype('category')
    if base[COL_ITEM].dtype.name != 'category':
        base[COL_ITEM] = base[COL_ITEM].astype('category')

    s_pw = df_time_pw[COL_PW]
    if s_pw.dtype.name != 'category':
        s_pw = s_pw.astype('category')
    t = pd.DataFrame({COL_PW: s_pw.values})

    n_b = len(base)
    n_t = len(t)

    rep_b = base.loc[base.index.repeat(n_t)].reset_index(drop=True)
    tile_t = pd.concat([t] * n_b, ignore_index=True)

    df = pd.concat([rep_b, tile_t], axis=1)
    df[COL_PW] = df[COL_PW].astype('category')
    return df

# 1107버전추가: 정수 컬럼(int) 캐스팅 지원
def _inject_version_cast(df: pd.DataFrame, out_version: str, cat_cols: list[str],
                         float_cols: list[str], int_cols: list[str] = None) -> pd.DataFrame:
    if df is None or df.empty:
        cols = [COL_VERSION, *cat_cols, *(float_cols or []), *(int_cols or [])]
        return _mk_empty_cols(cols,
                              cat_cols=[COL_VERSION, *cat_cols],
                              float_cols=float_cols or [],
                              int_cols=int_cols or [])
    if COL_VERSION not in df.columns:
        df.insert(0, COL_VERSION, out_version)
    else:
        df[COL_VERSION] = out_version

    for c in [COL_VERSION, *cat_cols]:
        if df[c].dtype.name != 'category':
            df[c] = df[c].astype('category')

    for m in (float_cols or []):
        df[m] = pd.to_numeric(df[m], errors='coerce').astype('float32')

    # 1107버전추가: Int32(NA 허용)
    for ic in (int_cols or []):
        # 우선 숫자화 → 정수로 변환(NA 허용)
        s = pd.to_numeric(df.get(ic), errors='coerce')
        df[ic] = s.astype('Int32')
    return df

def _safe_left_merge(df_left: pd.DataFrame, df_right: pd.DataFrame, on: list[str], how: str = 'left', suffixes=('', '_r')):
    if df_right is None or df_right.empty:
        return df_left
    return df_left.merge(df_right, how=how, on=on, suffixes=suffixes)

# ──────────────────────────────────────────────────────────────────────────────
# Step 4-0) Estimated Price 생성 대상 (ShipTo*Item) 선정  (변경 없음)
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step04_00_select_price_targets(
    df_step01_01_sin_pick : pd.DataFrame,   # Step 1-1 결과
    df_step01_03_sout_pick: pd.DataFrame,   # Step 1-3 결과
    **kwargs
) -> pd.DataFrame:
    need_cols = [COL_SHIP_TO, COL_ITEM]
    # S/In
    if df_step01_01_sin_pick is None or df_step01_01_sin_pick.empty:
        sin_keys = pd.DataFrame(columns=need_cols)
    else:
        sin_keys = df_step01_01_sin_pick[need_cols].drop_duplicates(ignore_index=True)
    # S/Out
    if df_step01_03_sout_pick is None or df_step01_03_sout_pick.empty:
        sout_keys = pd.DataFrame(columns=need_cols)
    else:
        sout_keys = df_step01_03_sout_pick[need_cols].drop_duplicates(ignore_index=True)

    if sin_keys.empty or sout_keys.empty:
        return pd.DataFrame(columns=need_cols)

    df = sin_keys.merge(sout_keys, on=need_cols, how='inner')
    for c in need_cols:
        if df[c].dtype.name != 'category':
            df[c] = df[c].astype('category')
    return df[need_cols]

# ──────────────────────────────────────────────────────────────────────────────
# Step 4-1) Estimated Price Local + Estimated Price Color 생성
#   우선순위 1 → 9
#   1) Estimated Price Modify_Local
#   2) Estimated Price_Local
#   3) Estimated Price Item Std4_Local              → 사용 시 Color=1
#   4) Estimated Price Item Std3_Local              → 사용 시 Color=1
#   5) Estimated Price Item Std2_Local              → 사용 시 Color=1
#   6) Estimated Price Sales Std2 Item Std4_Local   → 사용 시 Color=1   # 25.11.17 변경
#   7) Estimated Price Sales Std2 Item Std3_Local   → 사용 시 Color=1   # 25.11.17 변경
#   8) Estimated Price Sales Std2 Item Std2_Local   → 사용 시 Color=1   # 25.11.17 변경
#   9) AP Price USD(ShipTo,Item,Month) * ExRate Local(Std3,PW) → 사용 시 Color=1
#
#   Color 규칙:
#     - 1,2단계(Modify / Local 직접 사용) → Color=0
#     - 3~9단계 fallback 사용            → Color=1
#     - 값 미생성(NaN)                  → Color=NA
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step04_01_build_est_price_local(
    df_targets                 : pd.DataFrame,   # Step 4-0 결과 (ShipTo*Item)
    df_time_pw                 : pd.DataFrame,   # Time.[Partial Week]
    df_item_mst                : pd.DataFrame,   # Item Std2/3/4
    df_sdd                     : pd.DataFrame,   # SDD (Sales Std3/Std2 매핑용)
    df_est_price               : pd.DataFrame,   # ShipTo*Item*PW → Mod, Local
    df_ep_std4_local           : pd.DataFrame,   # ItemStd4*ShipTo*PW → 값
    df_ep_std3_local           : pd.DataFrame,   # ItemStd3*ShipTo*PW → 값
    df_ep_std2_local           : pd.DataFrame,   # ItemStd2*ShipTo*PW → 값
    df_ep_sales_std2_std4_local: pd.DataFrame,   # SalesStd2*ItemStd4*PW → 값  # 25.11.17 변경
    df_ep_sales_std2_std3_local: pd.DataFrame,   # SalesStd2*ItemStd3*PW → 값  # 25.11.17 변경
    df_ep_sales_std2_std2_local: pd.DataFrame,   # SalesStd2*ItemStd2*PW → 값  # 25.11.17 변경
    df_ap_price                : pd.DataFrame,   # ShipTo*Item*Month → USD
    df_exrate_local            : pd.DataFrame,   # SalesStd3*PW → 환율
    out_version                : str

) -> pd.DataFrame:
    out_cols = [
        COL_VERSION,
        COL_SHIP_TO,
        COL_ITEM,
        COL_PW,
        COL_EST_PRICE_LOCAL,
        COL_EST_PRICE_COLOR
    ]

    if df_targets is None or df_targets.empty or df_time_pw is None or df_time_pw.empty:
        # 빈 스켈레톤 반환
        return _mk_empty_cols(
            out_cols,
            cat_cols=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PW],
            float_cols=[COL_EST_PRICE_LOCAL],
            int_cols=[COL_EST_PRICE_COLOR]
        )

    # (1) (ShipTo*Item)×PW 확장
    df = _expand_s2_by_pweek(df_targets, df_time_pw)  # [ShipTo, Item, PW]

    # (2) Item Std2/3/4, Sales Std3 / Sales Std2 매핑  # 25.11.17 변경(Std2 추가)
    if df_item_mst is not None and not df_item_mst.empty:
        df = _safe_left_merge(
            df,
            df_item_mst[[COL_ITEM, COL_ITEM_STD2, COL_ITEM_STD3, COL_ITEM_STD4]].drop_duplicates(),
            on=[COL_ITEM]
        )
    else:
        df[COL_ITEM_STD2] = np.nan
        df[COL_ITEM_STD3] = np.nan
        df[COL_ITEM_STD4] = np.nan

    if df_sdd is not None and not df_sdd.empty:
        # Sales Std3 에 더해 Sales Std2 까지 함께 매핑  # 25.11.17 변경
        sdd_cols = [COL_SHIP_TO, COL_STD3]
        if COL_STD2 in df_sdd.columns:
            sdd_cols.append(COL_STD2)
        sdd_map = df_sdd[sdd_cols].drop_duplicates()
        df = _safe_left_merge(df, sdd_map, on=[COL_SHIP_TO])
    else:
        df[COL_STD3] = np.nan
        df[COL_STD2] = np.nan

    # (3) 1~2단계 소스 병합 (ShipTo*Item*PW)
    if df_est_price is not None and not df_est_price.empty:
        use_cols = [
            COL_SHIP_TO,
            COL_ITEM,
            COL_PW,
            COL_EST_PRICE_MOD_LOCAL,
            COL_EST_PRICE_LOCAL
        ]
        use_cols = [c for c in use_cols if c in df_est_price.columns]
        df = _safe_left_merge(df, df_est_price[use_cols], on=[COL_SHIP_TO, COL_ITEM, COL_PW])
        # 누락된 컬럼 방어
        if COL_EST_PRICE_MOD_LOCAL not in df.columns:
            df[COL_EST_PRICE_MOD_LOCAL] = np.nan
        if COL_EST_PRICE_LOCAL not in df.columns:
            df[COL_EST_PRICE_LOCAL] = np.nan
    else:
        df[COL_EST_PRICE_MOD_LOCAL] = np.nan
        df[COL_EST_PRICE_LOCAL]     = np.nan

    # (4) 3~5단계(Std4/3/2) 병합: ShipTo+ItemStdX+PW  # 기존 로직 유지
    if df_ep_std4_local is not None and not df_ep_std4_local.empty:
        m4 = df_ep_std4_local[
            [COL_ITEM_STD4, COL_SHIP_TO, COL_PW, COL_EP_STD4_LOCAL]
        ].drop_duplicates()
        df = _safe_left_merge(df, m4, on=[COL_ITEM_STD4, COL_SHIP_TO, COL_PW])
    else:
        df[COL_EP_STD4_LOCAL] = np.nan

    if df_ep_std3_local is not None and not df_ep_std3_local.empty:
        m3 = df_ep_std3_local[
            [COL_ITEM_STD3, COL_SHIP_TO, COL_PW, COL_EP_STD3_LOCAL]
        ].drop_duplicates()
        df = _safe_left_merge(df, m3, on=[COL_ITEM_STD3, COL_SHIP_TO, COL_PW])
    else:
        df[COL_EP_STD3_LOCAL] = np.nan

    if df_ep_std2_local is not None and not df_ep_std2_local.empty:
        m2 = df_ep_std2_local[
            [COL_ITEM_STD2, COL_SHIP_TO, COL_PW, COL_EP_STD2_LOCAL]
        ].drop_duplicates()
        df = _safe_left_merge(df, m2, on=[COL_ITEM_STD2, COL_SHIP_TO, COL_PW])
    else:
        df[COL_EP_STD2_LOCAL] = np.nan

    # (5) 6~8단계: Sales Std2 + ItemStdX + PW 기반 Estimated Price  # 25.11.17 변경
    if df_ep_sales_std2_std4_local is not None and not df_ep_sales_std2_std4_local.empty:
        s4 = df_ep_sales_std2_std4_local[
            [COL_STD2, COL_ITEM_STD4, COL_PW, COL_EP_SALES_STD4_LOCAL]
        ].drop_duplicates()
        df = _safe_left_merge(df, s4, on=[COL_STD2, COL_ITEM_STD4, COL_PW])
    else:
        df[COL_EP_SALES_STD4_LOCAL] = np.nan

    if df_ep_sales_std2_std3_local is not None and not df_ep_sales_std2_std3_local.empty:
        s3 = df_ep_sales_std2_std3_local[
            [COL_STD2, COL_ITEM_STD3, COL_PW, COL_EP_SALES_STD3_LOCAL]
        ].drop_duplicates()
        df = _safe_left_merge(df, s3, on=[COL_STD2, COL_ITEM_STD3, COL_PW])
    else:
        df[COL_EP_SALES_STD3_LOCAL] = np.nan

    if df_ep_sales_std2_std2_local is not None and not df_ep_sales_std2_std2_local.empty:
        s2 = df_ep_sales_std2_std2_local[
            [COL_STD2, COL_ITEM_STD2, COL_PW, COL_EP_SALES_STD2_LOCAL]
        ].drop_duplicates()
        df = _safe_left_merge(df, s2, on=[COL_STD2, COL_ITEM_STD2, COL_PW])
    else:
        df[COL_EP_SALES_STD2_LOCAL] = np.nan

    # (6) 9단계: AP Price USD * EXRATE Local (SalesStd3, PW)
    df[COL_MONTH] = df[COL_PW].astype(str).map(_pweek_to_month_str)

    if df_ap_price is not None and not df_ap_price.empty and COL_MONTH in df_ap_price.columns:
        ap = df_ap_price[
            [COL_SHIP_TO, COL_ITEM, COL_MONTH, COL_AP_PRICE_USD]
        ].drop_duplicates()
        df = _safe_left_merge(df, ap, on=[COL_SHIP_TO, COL_ITEM, COL_MONTH])
    else:
        df[COL_AP_PRICE_USD] = np.nan

    if df_exrate_local is not None and not df_exrate_local.empty:
        ex = df_exrate_local[
            [COL_STD3, COL_PW, COL_EXRATE_LOCAL]
        ].drop_duplicates()
        df = _safe_left_merge(df, ex, on=[COL_STD3, COL_PW])
    else:
        df[COL_EXRATE_LOCAL] = np.nan

    df['__ap_local'] = (
        pd.to_numeric(df[COL_AP_PRICE_USD], errors='coerce')
        * pd.to_numeric(df[COL_EXRATE_LOCAL], errors='coerce')
    )

    # (7) 우선순위 1→9 coalesce + Color 계산  # 25.11.17 변경
    v1 = pd.to_numeric(df[COL_EST_PRICE_MOD_LOCAL],      errors='coerce')  # 1
    v2 = pd.to_numeric(df[COL_EST_PRICE_LOCAL],          errors='coerce')  # 2
    v3 = pd.to_numeric(df[COL_EP_STD4_LOCAL],            errors='coerce')  # 3
    v4 = pd.to_numeric(df[COL_EP_STD3_LOCAL],            errors='coerce')  # 4
    v5 = pd.to_numeric(df[COL_EP_STD2_LOCAL],            errors='coerce')  # 5
    v6 = pd.to_numeric(df[COL_EP_SALES_STD4_LOCAL],      errors='coerce')  # 6 (Sales Std2 + Std4)
    v7 = pd.to_numeric(df[COL_EP_SALES_STD3_LOCAL],      errors='coerce')  # 7 (Sales Std2 + Std3)
    v8 = pd.to_numeric(df[COL_EP_SALES_STD2_LOCAL],      errors='coerce')  # 8 (Sales Std2 + Std2)
    v9 = pd.to_numeric(df['__ap_local'],                 errors='coerce')  # 9 (AP*EXRATE)

    out_val = v1.copy()
    out_val = out_val.fillna(v2)
    out_val = out_val.fillna(v3)
    out_val = out_val.fillna(v4)
    out_val = out_val.fillna(v5)
    out_val = out_val.fillna(v6)   # 25.11.17 추가
    out_val = out_val.fillna(v7)   # 25.11.17 추가
    out_val = out_val.fillna(v8)   # 25.11.17 추가
    out_val = out_val.fillna(v9)

    # 어떤 단계에서 처음 선택되었는지 마스크 계산
    use1 = v1.notna()
    use2 = v2.notna() & v1.isna()
    use3 = v3.notna() & v1.isna() & v2.isna()
    use4 = v4.notna() & v1.isna() & v2.isna() & v3.isna()
    use5 = v5.notna() & v1.isna() & v2.isna() & v3.isna() & v4.isna()
    use6 = v6.notna() & v1.isna() & v2.isna() & v3.isna() & v4.isna() & v5.isna()
    use7 = v7.notna() & v1.isna() & v2.isna() & v3.isna() & v4.isna() & v5.isna() & v6.isna()
    use8 = v8.notna() & v1.isna() & v2.isna() & v3.isna() & v4.isna() & v5.isna() & v6.isna() & v7.isna()
    use9 = v9.notna() & v1.isna() & v2.isna() & v3.isna() & v4.isna() & v5.isna() & v6.isna() & v7.isna() & v8.isna()

    have_any      = out_val.notna()
    fallback_mask = use3 | use4 | use5 | use6 | use7 | use8 | use9  # 3~9단계
    base_mask     = (use1 | use2) & have_any                         # 1,2단계

    # Color: 기본 NA → 1(3~9) / 0(1~2)
    color = pd.Series(pd.NA, index=out_val.index, dtype="Int32")
    color.loc[base_mask]     = 0
    color.loc[fallback_mask] = 1

    df[COL_EST_PRICE_LOCAL] = out_val.astype('float32')
    df[COL_EST_PRICE_COLOR] = color

    # (8) 출력 스키마 + Version 주입
    out = df[[COL_SHIP_TO, COL_ITEM, COL_PW, COL_EST_PRICE_LOCAL, COL_EST_PRICE_COLOR]].copy(deep=False)
    out = _inject_version_cast(
        out,
        out_version=out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_PW],
        float_cols=[COL_EST_PRICE_LOCAL],
        int_cols=[COL_EST_PRICE_COLOR]  # 1107버전추가
    )

    # 메모리 정리
    del df
    gc.collect()
    return out[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PW, COL_EST_PRICE_LOCAL, COL_EST_PRICE_COLOR]]

# ──────────────────────────────────────────────────────────────────────────────
# Step 4-2) Estimated Price_Local 결측치 하위 Lv 평균으로 보완
#   - 결측에서 자식 평균으로 채운 행은 Color=1 로 지정  # 1107버전추가
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step04_02_fill_missing_price_from_children(
    df_est_local : pd.DataFrame,   # Step 4-1 결과
    df_sdd       : pd.DataFrame,   # SDD (parent→children 구성)
    **kwargs
) -> pd.DataFrame:
    if df_est_local is None or df_est_local.empty:
        return df_est_local

    # parent → children 매핑(dict)
    parent_children = {}
    if df_sdd is not None and not df_sdd.empty:
        cols = [COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6, COL_SHIP_TO]
        sdd = df_sdd[cols].drop_duplicates()
        for _, r in sdd.iterrows():
            child = str(r[COL_SHIP_TO])
            for pcol in [COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6]:
                parent = str(r[pcol])
                parent_children.setdefault(parent, set()).add(child)

    existing_shipto = set(df_est_local[COL_SHIP_TO].astype(str).unique())
    parent_children = {
        p: sorted([c for c in ch if c in existing_shipto])
        for p, ch in parent_children.items()
        if len([c for c in ch if c in existing_shipto]) > 0
    }
    if not parent_children:
        return df_est_local

    map_rows = []
    for p, childs in parent_children.items():
        for c in childs:
            map_rows.append((p, c))
    df_map = pd.DataFrame(map_rows, columns=['__parent', '__child'])

    df_child = df_est_local.rename(columns={COL_SHIP_TO: '__child'})[['__child', COL_ITEM, COL_PW, COL_EST_PRICE_LOCAL]]
    df_par_join = df_map.merge(df_child, on='__child', how='left')  # [parent, child, item, pw, val]

    grp = df_par_join.groupby(['__parent', COL_ITEM, COL_PW], as_index=False)[COL_EST_PRICE_LOCAL].mean()
    grp.rename(columns={'__parent': COL_SHIP_TO, COL_EST_PRICE_LOCAL: '__avg_child_val'}, inplace=True)

    out = df_est_local.merge(grp, on=[COL_SHIP_TO, COL_ITEM, COL_PW], how='left')

    need = out[COL_EST_PRICE_LOCAL].isna() & out['__avg_child_val'].notna()
    if need.any():
        out.loc[need, COL_EST_PRICE_LOCAL] = out.loc[need, '__avg_child_val'].astype('float32')
        # 1107버전추가: 자식 평균으로 채운 경우 Color=1
        if COL_EST_PRICE_COLOR not in out.columns:
            out[COL_EST_PRICE_COLOR] = pd.Series(pd.NA, index=out.index, dtype="Int32")
        out.loc[need, COL_EST_PRICE_COLOR] = 1

    out.drop(columns=['__avg_child_val'], inplace=True)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Step 4-4) Estimated Price Local Output 포맷 (Color 포함)  # 1107버전추가
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step04_04_format_est_price_output(
    df_est_local: pd.DataFrame,
    out_version : str,
    **kwargs
) -> pd.DataFrame:
    cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PW, COL_EST_PRICE_LOCAL, COL_EST_PRICE_COLOR]  # 1107버전추가
    if df_est_local is None or df_est_local.empty:
        return _mk_empty_cols(cols,
                              cat_cols=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PW],
                              float_cols=[COL_EST_PRICE_LOCAL],
                              int_cols=[COL_EST_PRICE_COLOR])  # 1107버전추가

    df = df_est_local.copy(deep=False)
    if COL_VERSION not in df.columns:
        df.insert(0, COL_VERSION, out_version)
    else:
        df[COL_VERSION] = out_version

    for c in [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PW]:
        if df[c].dtype.name != 'category':
            df[c] = df[c].astype('category')

    df[COL_EST_PRICE_LOCAL] = pd.to_numeric(df[COL_EST_PRICE_LOCAL], errors='coerce').astype('float32')

    # 1107버전추가: Color 정수화(Int32)
    if COL_EST_PRICE_COLOR not in df.columns:
        df[COL_EST_PRICE_COLOR] = pd.Series(pd.NA, index=df.index, dtype="Int32")
    else:
        df[COL_EST_PRICE_COLOR] = pd.to_numeric(df[COL_EST_PRICE_COLOR], errors='coerce').astype('Int32')

    return df[cols]

########################################################################################################################
# Step 5) Split Ratio Data 생성 — 1107버전 (measureLv 가드만 명시 추가)
########################################################################################################################

# ──────────────────────────────────────────────────────────────────────────────
# [공통] Split Ratio 빌더 (S/In, S/Out 공용) — 변경 없음
# ──────────────────────────────────────────────────────────────────────────────
def _empty_split_ratio_schema(with_loc: bool, meas_col: str) -> pd.DataFrame:
    cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if with_loc else []) + [COL_PW, meas_col]
    return pd.DataFrame(columns=cols)

def _unique_keyframe_from_step3(df_in: pd.DataFrame, *, with_loc: bool) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if with_loc else []) + [COL_PW])
    need_cols = [COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if with_loc else []) + [COL_PW]
    use_cols  = [c for c in need_cols if c in df_in.columns]
    df = df_in.loc[:, use_cols].copy(deep=False)
    return df.drop_duplicates(ignore_index=True)

def _map_shipto_to_std2(df_sdd: pd.DataFrame) -> pd.DataFrame:
    if df_sdd is None or df_sdd.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_STD2])
    return df_sdd[[COL_SHIP_TO, COL_STD2]].drop_duplicates()

def _map_item_to_std1(df_item_mst: pd.DataFrame) -> pd.DataFrame:
    if df_item_mst is None or df_item_mst.empty:
        return pd.DataFrame(columns=[COL_ITEM, COL_ITEM_STD1])
    return df_item_mst[[COL_ITEM, COL_ITEM_STD1]].drop_duplicates()

def _build_split_ratio_generic(
    df_base_step3: pd.DataFrame,
    df_item_mst: pd.DataFrame,
    df_sdd: pd.DataFrame,
    df_ratio_in: pd.DataFrame,   # AP1/AP2/GC/Local 각각의 인풋
    *,
    meas_col: str,               # 최종 산출 컬럼명(예: COL_SIN_SR_AP1)
    with_loc: bool,              # S/In=True, S/Out=True
    out_version: str
) -> pd.DataFrame:
    # 0) 빈 입력 방어
    if df_base_step3 is None or df_base_step3.empty:
        return _empty_split_ratio_schema(with_loc, meas_col)

    # 1) 키 스켈레톤
    key_df = _unique_keyframe_from_step3(df_base_step3, with_loc=with_loc)
    if key_df.empty:
        return _empty_split_ratio_schema(with_loc, meas_col)

    # 2) ShipTo→Std2, Item→Std1 매핑
    map_std2 = _map_shipto_to_std2(df_sdd)
    map_std1 = _map_item_to_std1(df_item_mst)
    df = key_df.merge(map_std2, on=COL_SHIP_TO, how='left')
    df = df.merge(map_std1, on=COL_ITEM,    how='left')

    # 3) Ratio 인풋 조인
    if df_ratio_in is None or df_ratio_in.empty:
        return _empty_split_ratio_schema(with_loc, meas_col)

    need_cols = [COL_STD2, COL_ITEM_STD1, COL_PW, meas_col]
    if any(c not in df_ratio_in.columns for c in need_cols):
        return _empty_split_ratio_schema(with_loc, meas_col)

    rmap = df_ratio_in.loc[:, need_cols].copy(deep=False)
    rmap[meas_col] = pd.to_numeric(rmap[meas_col], errors='coerce').astype('float32')
    df = df.merge(rmap, on=[COL_STD2, COL_ITEM_STD1, COL_PW], how='left')

    # 4) 값 없는 행은 생성하지 않음
    df = df[df[meas_col].notna()].copy(deep=False)
    if df.empty:
        return _empty_split_ratio_schema(with_loc, meas_col)

    # 5) 스키마/타입/Version
    if COL_VERSION not in df.columns:
        df.insert(0, COL_VERSION, out_version)
    else:
        df[COL_VERSION] = out_version
    for c in [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PW] + ([COL_LOCATION] if with_loc else []):
        if c in df.columns:
            df[c] = df[c].astype('category')
    df[meas_col] = df[meas_col].astype('float32')

    out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if with_loc else []) + [COL_PW, meas_col]
    return df[out_cols]

# ──────────────────────────────────────────────────────────────────────────────
# measureLv 전역 가드 — 1107버전추가
# ──────────────────────────────────────────────────────────────────────────────
def _meas_lower() -> str:
    try:
        return (measureLv or '').strip().lower()
    except NameError:
        return ''

# ──────────────────────────────────────────────────────────────────────────────
# Step 5-1) S/In FCST(GI) Split Ratio_AP1 — 1107버전추가: measureLv 가드
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step05_01_build_sin_sr_ap1(
    df_step03_ap1: pd.DataFrame,       # Step 3-1-1 결과 (AP1)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_ap1    : pd.DataFrame,       # DF_IN_SIN_SR_AP1
    out_version  : str
) -> pd.DataFrame:
    if _meas_lower() != 'ap1':  # 1107버전추가
        return _empty_split_ratio_schema(True, COL_SIN_SR_AP1)
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_ap1,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_ap1,
        meas_col      = COL_SIN_SR_AP1,
        with_loc      = True,
        out_version   = out_version
    )

# ──────────────────────────────────────────────────────────────────────────────
# Step 5-2) S/In FCST(GI) Split Ratio_AP2 — 1107버전추가: measureLv 가드
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step05_02_build_sin_sr_ap2(
    df_step03_ap2: pd.DataFrame,       # Step 3-1-2 결과 (AP2)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_ap2    : pd.DataFrame,       # DF_IN_SIN_SR_AP2
    out_version  : str
) -> pd.DataFrame:
    if _meas_lower() != 'ap2':  # 1107버전추가
        return _empty_split_ratio_schema(True, COL_SIN_SR_AP2)
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_ap2,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_ap2,
        meas_col      = COL_SIN_SR_AP2,
        with_loc      = True,
        out_version   = out_version
    )

# ──────────────────────────────────────────────────────────────────────────────
# Step 5-3) S/In FCST(GI) Split Ratio_GC — 1107버전추가: measureLv 가드
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step05_03_build_sin_sr_gc(
    df_step03_gc : pd.DataFrame,       # Step 3-1-3 결과 (GC)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_gc     : pd.DataFrame,       # DF_IN_SIN_SR_GC
    out_version  : str
) -> pd.DataFrame:
    if _meas_lower() != 'gc':  # 1107버전추가
        return _empty_split_ratio_schema(True, COL_SIN_SR_GC)
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_gc,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_gc,
        meas_col      = COL_SIN_SR_GC,
        with_loc      = True,
        out_version   = out_version
    )

# ──────────────────────────────────────────────────────────────────────────────
# Step 5-4) S/In FCST(GI) Split Ratio_Local — 1107버전추가: measureLv 가드
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step05_04_build_sin_sr_local(
    df_step03_local: pd.DataFrame,     # Step 3-1-4 결과 (Local)
    df_item_mst    : pd.DataFrame,
    df_sdd         : pd.DataFrame,
    df_sr_local    : pd.DataFrame,     # DF_IN_SIN_SR_LOCAL
    out_version    : str
) -> pd.DataFrame:
    if _meas_lower() != 'local':  # 1107버전추가
        return _empty_split_ratio_schema(True, COL_SIN_SR_LOCAL)
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_local,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_local,
        meas_col      = COL_SIN_SR_LOCAL,
        with_loc      = True,
        out_version   = out_version
    )

# ──────────────────────────────────────────────────────────────────────────────
# Step 5-5) S/Out FCST Split Ratio_AP1 — 1107버전추가: measureLv 가드
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step05_05_build_sout_sr_ap1(
    df_step03_ap1: pd.DataFrame,       # Step 3-2-1 결과 (AP1)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_ap1    : pd.DataFrame,       # DF_IN_SOUT_SR_AP1
    out_version  : str
) -> pd.DataFrame:
    if _meas_lower() != 'ap1':  # 1107버전추가
        return _empty_split_ratio_schema(True, COL_SOUT_SR_AP1)
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_ap1,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_ap1,
        meas_col      = COL_SOUT_SR_AP1,
        with_loc      = True,
        out_version   = out_version
    )

# ──────────────────────────────────────────────────────────────────────────────
# Step 5-6) S/Out FCST Split Ratio_AP2 — 1107버전추가: measureLv 가드
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step05_06_build_sout_sr_ap2(
    df_step03_ap2: pd.DataFrame,       # Step 3-2-2 결과 (AP2)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_ap2    : pd.DataFrame,       # DF_IN_SOUT_SR_AP2
    out_version  : str
) -> pd.DataFrame:
    if _meas_lower() != 'ap2':  # 1107버전추가
        return _empty_split_ratio_schema(True, COL_SOUT_SR_AP2)
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_ap2,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_ap2,
        meas_col      = COL_SOUT_SR_AP2,
        with_loc      = True,
        out_version   = out_version
    )

# ──────────────────────────────────────────────────────────────────────────────
# Step 5-7) S/Out FCST Split Ratio_GC — 1107버전추가: measureLv 가드
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step05_07_build_sout_sr_gc(
    df_step03_gc : pd.DataFrame,       # Step 3-2-3 결과 (GC)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_gc     : pd.DataFrame,       # DF_IN_SOUT_SR_GC
    out_version  : str
) -> pd.DataFrame:
    if _meas_lower() != 'gc':  # 1107버전추가
        return _empty_split_ratio_schema(True, COL_SOUT_SR_GC)
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_gc,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_gc,
        meas_col      = COL_SOUT_SR_GC,
        with_loc      = True,
        out_version   = out_version
    )

# ──────────────────────────────────────────────────────────────────────────────
# Step 5-8) S/Out FCST Split Ratio_Local — 1107버전추가: measureLv 가드
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step05_08_build_sout_sr_local(
    df_step03_local: pd.DataFrame,     # Step 3-2-4 결과 (Local)
    df_item_mst    : pd.DataFrame,
    df_sdd         : pd.DataFrame,
    df_sr_local    : pd.DataFrame,     # DF_IN_SOUT_SR_LOCAL
    out_version    : str
) -> pd.DataFrame:
    if _meas_lower() != 'local':  # 1107버전추가
        return _empty_split_ratio_schema(True, COL_SOUT_SR_LOCAL)
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_local,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_local,
        meas_col      = COL_SOUT_SR_LOCAL,
        with_loc      = True,
        out_version   = out_version
    )

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
            # 파라메터추가 2025.11.07
            salesItemLocation = '400001:RF29BB8600QLAA^400002:RF29BB8600QLAA'
            measureLv = 'ap2'
            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------

            # input_folder_name  = str_instance       
            input_folder_name  = "PYSalesProductASNDelta"     
            output_folder_name = str_instance
            
            # ------
            # str_input_dir = f'Input/{input_folder_name}'
            str_input_dir = f'Input/{input_folder_name}/PYSalesProductASNDeltaB2C'
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
        logger.Note(p_note=f'salesItemLocation  : {salesItemLocation}', p_log_level=LOG_LEVEL.debug())
        logger.Note(p_note=f'measureLv          : {measureLv}', p_log_level=LOG_LEVEL.debug())


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

        # =====================================================================================
        # Main: Step01 실행(요청 포맷 준수: dict_log + fn_log_dataframe 사용)
        # =====================================================================================
        # 사전 준비
        # Version = "CWV_DP"
        
        ################################################################################################################
        # (옵션) 현행 FCST 입력 삭제: STR_DF_IN_SIN_FCST / STR_DF_IN_SOUT_FCST / STR_DF_IN_FLOOR_FCST
        ################################################################################################################
        # 프로그램 시작부/Step01 실행 전 어느 시점에든 안전하게 비워둠
        # input_dataframes.pop(STR_DF_IN_SIN_FCST,  None)
        # input_dataframes.pop(STR_DF_IN_SOUT_FCST, None)
        # input_dataframes.pop(STR_DF_IN_FLOOR_FCST, None)

        ################################################################################################################
        # Step 1-0) Sales 선정 (salesItemLocation 파싱 → eStore 제외 → SDD 유효 ShipTo만)
        ################################################################################################################
        dict_log = {'p_step_no': 10, 'p_step_desc': 'Step 1-0) Sales 선정', 'p_df_name': None}
        df_step01_00_sales_pairs, pairs_have_loc = fn_step01_00_select_sales(
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),
            input_dataframes.get(DF_IN_ESTORE, pd.DataFrame()),
            salesItemLocation,
            **dict_log
        )
        fn_log_dataframe(df_step01_00_sales_pairs, 'df_step01_00_sales_pairs')   # (ShipTo, Item, [Location?])

        ################################################################################################################
        # Step 1-1) S/In 더미에서 생성할 Sales 선정
        ################################################################################################################
        dict_log = {'p_step_no': 11, 'p_step_desc': 'Step 1-1) S/In 더미에서 생성할 Sales 선정', 'p_df_name': None}
        df_step01_01_sin_pick = fn_step01_01_pick_sin_dummy(
            input_dataframes.get(DF_IN_SIN_DUMMY, pd.DataFrame()),
            df_step01_00_sales_pairs,
            pairs_have_loc,
            measureLv,
            input_dataframes.get(DF_IN_SIN_FCST, pd.DataFrame()),   # (옵션) 현행 FCST 비교
            **dict_log
        )
        fn_log_dataframe(df_step01_01_sin_pick, 'df_step01_01_sin_pick')

        ################################################################################################################
        # Step 1-2) S/In 더미 삭제용 Output
        ################################################################################################################
        dict_log = {'p_step_no': 12, 'p_step_desc': 'Step 1-2) S/In 더미 삭제용 Output', 'p_df_name': None}
        df_step01_02_out_sin = fn_step01_02_build_output_sin_dummy_delete(
            df_step01_01_sin_pick,
            measureLv,
            Version,
            **dict_log
        )
        fn_log_dataframe(df_step01_02_out_sin, 'df_step01_02_Output_SIn_Dummy')
        Output_SIn_Dummy                    = df_step01_02_out_sin
        # output_dataframes[DF_OUT_SIN_DUMMY] = df_step01_02_out_sin

        ################################################################################################################
        # Step 1-3) S/Out 더미에서 생성할 Sales 선정
        ################################################################################################################
        dict_log = {'p_step_no': 13, 'p_step_desc': 'Step 1-3) S/Out 더미에서 생성할 Sales 선정', 'p_df_name': None}
        df_step01_03_sout_pick = fn_step01_03_pick_sout_dummy(
            input_dataframes.get(DF_IN_SOUT_DUMMY, pd.DataFrame()),
            df_step01_00_sales_pairs,
            pairs_have_loc,
            measureLv,
            input_dataframes.get(DF_IN_SOUT_FCST, pd.DataFrame()),  # (옵션) 현행 FCST 비교
            **dict_log
        )
        fn_log_dataframe(df_step01_03_sout_pick, 'df_step01_03_sout_pick')

        ################################################################################################################
        # Step 1-4) S/Out 더미 삭제용 Output
        ################################################################################################################
        dict_log = {'p_step_no': 14, 'p_step_desc': 'Step 1-4) S/Out 더미 삭제용 Output', 'p_df_name': None}
        df_step01_04_out_sout = fn_step01_04_build_output_sout_dummy_delete(
            df_step01_03_sout_pick,
            measureLv,
            Version,
            **dict_log
        )
        fn_log_dataframe(df_step01_04_out_sout, 'df_step01_04_Output_SOut_Dummy')
        Output_SOut_Dummy                    = df_step01_04_out_sout
        # output_dataframes[DF_OUT_SOUT_DUMMY] = df_step01_04_out_sout

        ################################################################################################################
        # Step 1-5) Flooring 더미에서 생성할 Sales 선정
        ################################################################################################################
        dict_log = {'p_step_no': 15, 'p_step_desc': 'Step 1-5) Flooring 더미에서 생성할 Sales 선정', 'p_df_name': None}
        df_step01_05_floor_pick = fn_step01_05_pick_flooring_dummy(
            input_dataframes.get(DF_IN_FLOORING_DUMMY, pd.DataFrame()),
            df_step01_00_sales_pairs,
            pairs_have_loc,
            measureLv,
            input_dataframes.get(DF_IN_FLOOR_FCST, pd.DataFrame()),  # (옵션) 현행 FCST 비교
            **dict_log
        )
        fn_log_dataframe(df_step01_05_floor_pick, 'df_step01_05_floor_pick')

        ################################################################################################################
        # Step 1-6) Flooring 더미 삭제용 Output
        ################################################################################################################
        dict_log = {'p_step_no': 16, 'p_step_desc': 'Step 1-6) Flooring 더미 삭제용 Output', 'p_df_name': None}
        df_step01_06_out_floor = fn_step01_06_build_output_flooring_dummy_delete(
            df_step01_05_floor_pick,
            measureLv,
            Version,
            **dict_log
        )
        fn_log_dataframe(df_step01_06_out_floor, 'df_step01_06_Output_Flooring_Dummy')
        Output_Flooring_Dummy = df_step01_06_out_floor
        # output_dataframes[DF_OUT_FLOORING_DUMMY] = df_step01_06_out_floor

        ################################################################################################################
        # Step 1-7) 모두 빈 경우 종료 플래그
        ################################################################################################################
        all_empty_step01 = (
            Output_SIn_Dummy.empty
            and Output_SOut_Dummy.empty
            and Output_Flooring_Dummy.empty
        )

        if all_empty_step01:
            # 1) Dummy FCST 입력 자체가 모두 없는 경우
            df_in_sin_dummy      = input_dataframes.get(DF_IN_SIN_DUMMY,      pd.DataFrame())
            df_in_sout_dummy     = input_dataframes.get(DF_IN_SOUT_DUMMY,     pd.DataFrame())
            df_in_floor_dummy    = input_dataframes.get(DF_IN_FLOORING_DUMMY, pd.DataFrame())

            no_dummy_all = (
                (df_in_sin_dummy is None  or df_in_sin_dummy.empty)
                and (df_in_sout_dummy is None or df_in_sout_dummy.empty)
                and (df_in_floor_dummy is None or df_in_floor_dummy.empty)
            )

            # 2) FCST 값이 존재하는지 체크
            df_in_sin_fcst   = input_dataframes.get(DF_IN_SIN_FCST,   pd.DataFrame())
            df_in_sout_fcst  = input_dataframes.get(DF_IN_SOUT_FCST,  pd.DataFrame())
            df_in_floor_fcst = input_dataframes.get(DF_IN_FLOOR_FCST, pd.DataFrame())

            has_fcst_any = (
                (df_in_sin_fcst is not None and not df_in_sin_fcst.empty)
                or (df_in_sout_fcst is not None and not df_in_sout_fcst.empty)
                or (df_in_floor_fcst is not None and not df_in_floor_fcst.empty)
            )

            # ───────────────────────────────────────────────────────────
            # 분기:
            #   1) Dummy 입력이 아예 없어서 아무것도 못 만든 경우
            #   2) Dummy는 있었지만(또는 있을 수 있지만) 이미 FCST가 있어 생성 안 된 경우
            #   3) 그 외 기타 케이스 → 기존 에러 유지
            # ───────────────────────────────────────────────────────────
            if no_dummy_all:
                # 1. Dummy FCST에 값이 없어서 값을 생성하지 않는 경우
                #    -> There is no association information.
                raise Exception('There is no association information.')
            elif has_fcst_any:
                # 2. FCST 값이 있어서 S/In 과 S/Out 을 모두 생성하지 않는 경우
                #    -> Forecast value already exists.
                raise Exception('Forecast value already exists.')
            else:
                # 기타: salesItemLocation에 해당 Sales가 없거나,
                #       SDD/eStore 필터링에서 다 걸러지는 등
                raise Exception('Step 1-2,Step 1-4,Step 1-6 is empty.')        

        
        ################################################################################################################
        # Step 2-1) S/In Assortment
        ################################################################################################################
        dict_log = {'p_step_no': 21, 'p_step_desc': 'Step 2-1) S/In FCST(GI) Assortment', 'p_df_name': None}
        df_step02_01_sin_assort = fn_step02_01_build_sin_assortment(
            df_step01_01_sin_pick,   # ← Step1-1 결과 사용(삭제용 Output 아님)
            measureLv,
            **dict_log
        )
        fn_log_dataframe(df_step02_01_sin_assort, 'df_step02_01_Output_SIn_Assortment')
        Output_SIn_Assortment = df_step02_01_sin_assort  # 명시적 변수로 전달(o9 호환)

        ################################################################################################################
        # Step 2-2) S/Out Assortment
        ################################################################################################################
        dict_log = {
            'p_step_no' : 22,
            'p_step_desc': 'Step 2-2) S/Out FCST Assortment',
            'p_df_name' : None
        }
        df_step02_02_sout_assort = fn_step02_02_build_sout_assortment(
            df_step01_03_sout_pick,                           # Step1-3 결과
            measureLv,
            input_dataframes.get(DF_IN_ITEM_MST, pd.DataFrame()),          # 25.11.24 추가
            input_dataframes.get(DF_IN_SOUT_SIMUL_MASTER, pd.DataFrame()), # 25.11.24 추가
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),               # 25.11.24 추가
            **dict_log
        )
        fn_log_dataframe(df_step02_02_sout_assort, 'df_step02_02_Output_SOut_Assortment')
        Output_SOut_Assortment = df_step02_02_sout_assort

        ################################################################################################################
        # Step 2-3) Flooring Assortment
        ################################################################################################################
        dict_log = {'p_step_no': 23, 'p_step_desc': 'Step 2-3) Flooring FCST Assortment', 'p_df_name': None}
        df_step02_03_floor_assort = fn_step02_03_build_flooring_assortment(
            df_step01_05_floor_pick,  # ← Step1-5 결과
            measureLv,
            **dict_log
        )
        fn_log_dataframe(df_step02_03_floor_assort, 'df_step02_03_Output_Flooring_Assortment')
        Output_Flooring_Assortment = df_step02_03_floor_assort

        ################################################################################################################
        # Step 3 — FCST 값 생성 (상수명 정합 반영 버전)
        ################################################################################################################

        ################################################################################################################
        # Step 3-1-1) S/In FCST(GI)_AP1, BL_AP1, New Model
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 31,
            'p_step_desc': 'Step 3-1-1) S/In FCST(GI)_AP1 & BL_AP1 & NewModel',
            'p_df_name'  : None
        }
        df_output_Sell_In_FCST_GI_AP1 = fn_step03_01_01_build_sin_fcst_ap1(
            df_step01_01_sin_pick,
            input_dataframes[DF_IN_TIME_PW],
            Version,
            **dict_log
        )
        df_step03_01_sin_fcst_ap1 = df_output_Sell_In_FCST_GI_AP1
        fn_log_dataframe(df_output_Sell_In_FCST_GI_AP1, f'df_step03_01_01_{DF_OUT_SIN_GI_AP1}')

        ################################################################################################################
        # Step 3-1-2) S/In FCST(GI)_AP2, BL_AP2
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 32,
            'p_step_desc': 'Step 3-1-2) S/In FCST(GI)_AP2 & BL_AP2',
            'p_df_name'  : None
        }
        df_output_Sell_In_FCST_GI_AP2 = fn_step03_01_02_build_sin_fcst_ap2(
            df_step01_01_sin_pick,
            input_dataframes[DF_IN_TIME_PW],
            Version,
            **dict_log
        )
        df_step03_02_sin_fcst_ap2 = df_output_Sell_In_FCST_GI_AP2
        fn_log_dataframe(df_output_Sell_In_FCST_GI_AP2, f'df_step03_01_02_{DF_OUT_SIN_GI_AP2}')

        ################################################################################################################
        # Step 3-1-3) S/In FCST(GI)_GC, BL_GC
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 33,
            'p_step_desc': 'Step 3-1-3) S/In FCST(GI)_GC & BL_GC',
            'p_df_name'  : None
        }
        df_output_Sell_In_FCST_GI_GC = fn_step03_01_03_build_sin_fcst_gc(
            df_step01_01_sin_pick,
            input_dataframes[DF_IN_TIME_PW],
            Version,
            **dict_log
        )
        df_step03_03_sin_fcst_gc = df_output_Sell_In_FCST_GI_GC
        fn_log_dataframe(df_output_Sell_In_FCST_GI_GC, f'df_step03_01_03_{DF_OUT_SIN_GI_GC}')

        ################################################################################################################
        # Step 3-1-4) S/In FCST(GI)_Local, BL_Local
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 34,
            'p_step_desc': 'Step 3-1-4) S/In FCST(GI)_Local & BL_Local',
            'p_df_name'  : None
        }
        df_output_Sell_In_FCST_GI_Local = fn_step03_01_04_build_sin_fcst_local(
            df_step01_01_sin_pick,
            input_dataframes[DF_IN_TIME_PW],
            Version,
            **dict_log
        )
        df_step03_04_sin_fcst_local = df_output_Sell_In_FCST_GI_Local
        fn_log_dataframe(df_output_Sell_In_FCST_GI_Local, f'df_step03_01_04_{DF_OUT_SIN_GI_LOCAL}')

        ################################################################################################################
        # Step 3-2-1) S/Out FCST_AP1
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 35,
            'p_step_desc': 'Step 3-2-1) S/Out FCST_AP1',
            'p_df_name'  : None
        }
        df_output_Sell_Out_FCST_AP1 = fn_step03_02_01_build_sout_fcst_ap1(
            df_step01_03_sout_pick,
            input_dataframes[DF_IN_TIME_PW],
            Version,
            **dict_log
        )
        df_step03_05_sout_fcst_ap1 = df_output_Sell_Out_FCST_AP1
        fn_log_dataframe(df_output_Sell_Out_FCST_AP1, f'df_step03_02_01_{DF_OUT_SOUT_AP1}')

        ################################################################################################################
        # Step 3-2-2) S/Out FCST_AP2
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 36,
            'p_step_desc': 'Step 3-2-2) S/Out FCST_AP2',
            'p_df_name'  : None
        }
        df_output_Sell_Out_FCST_AP2 = fn_step03_02_02_build_sout_fcst_ap2(
            df_step01_03_sout_pick,
            input_dataframes[DF_IN_TIME_PW],
            Version,
            **dict_log
        )
        df_step03_06_sout_fcst_ap2 = df_output_Sell_Out_FCST_AP2
        fn_log_dataframe(df_output_Sell_Out_FCST_AP2, f'step03_02_02_{DF_OUT_SOUT_AP2}')

        ################################################################################################################
        # Step 3-2-3) S/Out FCST_GC
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 37,
            'p_step_desc': 'Step 3-2-3) S/Out FCST_GC',
            'p_df_name'  : None
        }
        df_output_Sell_Out_FCST_GC = fn_step03_02_03_build_sout_fcst_gc(
            df_step01_03_sout_pick,
            input_dataframes[DF_IN_TIME_PW],
            Version,
            **dict_log
        )
        df_step03_07_sout_fcst_gc = df_output_Sell_Out_FCST_GC
        fn_log_dataframe(df_output_Sell_Out_FCST_GC, f'step03_02_03_{DF_OUT_SOUT_GC}')

        ################################################################################################################
        # Step 3-2-4) S/Out FCST_Local
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 38,
            'p_step_desc': 'Step 3-2-4) S/Out FCST_Local',
            'p_df_name'  : None
        }
        df_output_Sell_Out_FCST_Local = fn_step03_02_04_build_sout_fcst_local(
            df_step01_03_sout_pick,
            input_dataframes[DF_IN_TIME_PW],
            Version,
            **dict_log
        )
        df_step03_08_sout_fcst_local = df_output_Sell_Out_FCST_Local
        fn_log_dataframe(df_output_Sell_Out_FCST_Local, f'step03_02_04_{DF_OUT_SOUT_LOCAL}')

        ################################################################################################################
        # Step 3-3) Flooring FCST
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 39,
            'p_step_desc': 'Step 3-3) Flooring FCST',
            'p_df_name'  : None
        }
        df_output_Flooring_FCST = fn_step03_03_build_flooring_fcst(
            df_step01_05_floor_pick,
            input_dataframes[DF_IN_TIME_W],
            Version,
            **dict_log
        )
        df_step03_09_flooring_fcst = df_output_Flooring_FCST
        fn_log_dataframe(df_output_Flooring_FCST, f'step03_03_{DF_OUT_FLOORING_FCST}')


        ################################################################################################################
        # Step 3-4) df_output_BO_FCST  추후스펙. 빈 dataframe
        ################################################################################################################
        COL_VIRTUAL_BO_ID               = 'DP Virtual BO ID.[Virtual BO ID]'
        COL_BO_ID                       = 'DP BO ID.[BO ID]'
        COL_BO_FCST                     = 'BO FCST'
        df_output_BO_FCST = pd.DataFrame(columns=[
            COL_VERSION, COL_ITEM, COL_SHIP_TO, COL_LOCATION,
            COL_VIRTUAL_BO_ID, COL_BO_ID, COL_PW, COL_BO_FCST])
        fn_log_dataframe(df_output_BO_FCST, f'step03_04_{DF_OUT_BO_FCST}')
        
        ################################################################################################################
        # Step 4-0) Estimated Price 대상 ShipTo*Item 선정
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 40,
            'p_step_desc': 'Step 4-0) Estimated Price 대상 선정 (S/In∩S/Out)',
            'p_df_name'  : None
        }
        df_step04_00_targets = fn_step04_00_select_price_targets(
            df_step01_01_sin_pick,
            df_step01_03_sout_pick,
            **dict_log
        )
        fn_log_dataframe(df_step04_00_targets, 'df_step04_00_targets')
        
        ################################################################################################################
        # Step 4-1) Estimated Price Local 생성 (우선순위 1→9 적용)   # 25.11.17 변경
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 41,
            'p_step_desc': 'Step 4-1) Estimated Price Local 생성',
            'p_df_name'  : None
        }
        df_step04_01_est_local = fn_step04_01_build_est_price_local(
            df_step04_00_targets,
            input_dataframes[DF_IN_TIME_PW],
            input_dataframes.get(DF_IN_ITEM_MST,  pd.DataFrame()),
            input_dataframes.get(DF_IN_SDD,       pd.DataFrame()),
            input_dataframes.get(DF_IN_EST_PRICE, pd.DataFrame()),
            input_dataframes.get(DF_IN_EP_STD4_LOCAL,               pd.DataFrame()),
            input_dataframes.get(DF_IN_EP_STD3_LOCAL,               pd.DataFrame()),
            input_dataframes.get(DF_IN_EP_STD2_LOCAL,               pd.DataFrame()),
            # 25.11.17 추가: Sales Std2 기반 Estimated Price 3단계
            input_dataframes.get(DF_IN_EP_SALES_STD2_ITEM_STD4_LOCAL, pd.DataFrame()),
            input_dataframes.get(DF_IN_EP_SALES_STD2_ITEM_STD3_LOCAL, pd.DataFrame()),
            input_dataframes.get(DF_IN_EP_SALES_STD2_ITEM_STD2_LOCAL, pd.DataFrame()),
            # 기존 AP / EXRATE
            input_dataframes.get(DF_IN_AP_PRICE,      pd.DataFrame()),
            input_dataframes.get(DF_IN_EXRATE_LOCAL,  pd.DataFrame()),
            Version,
            **dict_log
        )
        fn_log_dataframe(df_step04_01_est_local, 'df_step04_01_est_local')

        ################################################################################################################
        # Step 4-2) Estimated Price Local 결측 보완(하위 Lv 평균)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 42,
            'p_step_desc': 'Step 4-2) Estimated Price Local 결측 하위Lv 평균 보완',
            'p_df_name'  : None
        }
        df_step04_02_est_local_filled = fn_step04_02_fill_missing_price_from_children(
            df_step04_01_est_local,
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),
            **dict_log
        )
        fn_log_dataframe(df_step04_02_est_local_filled, 'df_step04_02_est_local_filled')

        ################################################################################################################
        # Step 4-4) Estimated Price Local Output 포맷
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 44,
            'p_step_desc': 'Step 4-4) Estimated Price Local Output 포맷',
            'p_df_name'  : None
        }
        df_step04_04_output_est_local = fn_step04_04_format_est_price_output(
            df_step04_02_est_local_filled,
            Version,
            **dict_log
        )
        # o9 output 변수 지정  ← 17. (Output 4) df_output_Estimated_Price_Local
        df_output_Estimated_Price_Local = df_step04_04_output_est_local
        fn_log_dataframe(df_step04_04_output_est_local, f'df_step04_04_{DF_OUT_EST_PRICE_LOCAL}')

        ################################################################################################################
        # Step 5-1) S/In FCST(GI) Split Ratio_AP1
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 51,
            'p_step_desc': 'Step 5-1) S/In FCST(GI) Split Ratio_AP1',
            'p_df_name'  : None
        }
        df_step05_01_sin_sr_ap1 = fn_step05_01_build_sin_sr_ap1(
            df_step03_01_sin_fcst_ap1,                                # Step 3-1-1 결과
            input_dataframes.get(DF_IN_ITEM_MST, pd.DataFrame()),
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),
            input_dataframes.get(DF_IN_SIN_SR_AP1, pd.DataFrame()),
            Version,
            **dict_log
        )
        df_output_Sell_In_FCST_GI_Split_Ratio_AP1 = df_step05_01_sin_sr_ap1     # o9 Output 매핑
        fn_log_dataframe(df_step05_01_sin_sr_ap1, f'df_step05_01_{DF_OUT_SIN_SR_AP1}')
        
        ################################################################################################################
        # Step 5-2) S/In FCST(GI) Split Ratio_AP2
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 52,
            'p_step_desc': 'Step 5-2) S/In FCST(GI) Split Ratio_AP2',
            'p_df_name'  : None
        }
        df_step05_02_sin_sr_ap2 = fn_step05_02_build_sin_sr_ap2(
            df_step03_02_sin_fcst_ap2,                                # Step 3-1-2 결과
            input_dataframes.get(DF_IN_ITEM_MST, pd.DataFrame()),
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),
            input_dataframes.get(DF_IN_SIN_SR_AP2, pd.DataFrame()),
            Version,
            **dict_log
        )
        df_output_Sell_In_FCST_GI_Split_Ratio_AP2 = df_step05_02_sin_sr_ap2
        fn_log_dataframe(df_step05_02_sin_sr_ap2, f'df_step05_02_{DF_OUT_SIN_SR_AP2}')

        ################################################################################################################
        # Step 5-3) S/In FCST(GI) Split Ratio_GC
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 53,
            'p_step_desc': 'Step 5-3) S/In FCST(GI) Split Ratio_GC',
            'p_df_name'  : None
        }
        df_step05_03_sin_sr_gc = fn_step05_03_build_sin_sr_gc(
            df_step03_03_sin_fcst_gc,                                  # Step 3-1-3 결과
            input_dataframes.get(DF_IN_ITEM_MST, pd.DataFrame()),
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),
            input_dataframes.get(DF_IN_SIN_SR_GC, pd.DataFrame()),
            Version,
            **dict_log
        )
        df_output_Sell_In_FCST_GI_Split_Ratio_GC = df_step05_03_sin_sr_gc
        fn_log_dataframe(df_step05_03_sin_sr_gc, f'df_step05_03_{DF_OUT_SIN_SR_GC}')

        ################################################################################################################
        # Step 5-4) S/In FCST(GI) Split Ratio_Local
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 54,
            'p_step_desc': 'Step 5-4) S/In FCST(GI) Split Ratio_Local',
            'p_df_name'  : None
        }
        df_step05_04_sin_sr_local = fn_step05_04_build_sin_sr_local(
            df_step03_04_sin_fcst_local,                               # Step 3-1-4 결과
            input_dataframes.get(DF_IN_ITEM_MST, pd.DataFrame()),
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),
            input_dataframes.get(DF_IN_SIN_SR_LOCAL, pd.DataFrame()),
            Version,
            **dict_log
        )
        df_output_Sell_In_FCST_GI_Split_Ratio_Local = df_step05_04_sin_sr_local
        fn_log_dataframe(df_step05_04_sin_sr_local, f'df_step05_04_{DF_OUT_SIN_SR_LOCAL}')

        ################################################################################################################
        # Step 5-5) S/Out FCST Split Ratio_AP1
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 55,
            'p_step_desc': 'Step 5-5) S/Out FCST Split Ratio_AP1',
            'p_df_name'  : None
        }
        df_step05_05_sout_sr_ap1 = fn_step05_05_build_sout_sr_ap1(
            df_step03_05_sout_fcst_ap1,                            # Step 3-2-1 결과
            input_dataframes.get(DF_IN_ITEM_MST, pd.DataFrame()),
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),
            input_dataframes.get(DF_IN_SOUT_SR_AP1, pd.DataFrame()),
            Version,
            **dict_log
        )
        df_output_Sell_Out_FCST_Split_Ratio_AP1 = df_step05_05_sout_sr_ap1
        fn_log_dataframe(df_step05_05_sout_sr_ap1, f'df_step05_05_{DF_OUT_SOUT_SR_AP1}')

        ################################################################################################################
        # Step 5-6) S/Out FCST Split Ratio_AP2
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 56,
            'p_step_desc': 'Step 5-6) S/Out FCST Split Ratio_AP2',
            'p_df_name'  : None
        }
        df_step05_06_sout_sr_ap2 = fn_step05_06_build_sout_sr_ap2(
            df_step03_06_sout_fcst_ap2,                            # Step 3-2-2 결과
            input_dataframes.get(DF_IN_ITEM_MST, pd.DataFrame()),
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),
            input_dataframes.get(DF_IN_SOUT_SR_AP2, pd.DataFrame()),
            Version,
            **dict_log
        )
        df_output_Sell_Out_FCST_Split_Ratio_AP2 = df_step05_06_sout_sr_ap2
        fn_log_dataframe(df_step05_06_sout_sr_ap2, f'df_step05_06_{DF_OUT_SOUT_SR_AP2}')

        ################################################################################################################
        # Step 5-7) S/Out FCST Split Ratio_GC
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 57,
            'p_step_desc': 'Step 5-7) S/Out FCST Split Ratio_GC',
            'p_df_name'  : None
        }
        df_step05_07_sout_sr_gc = fn_step05_07_build_sout_sr_gc(
            df_step03_07_sout_fcst_gc,                             # Step 3-2-3 결과
            input_dataframes.get(DF_IN_ITEM_MST, pd.DataFrame()),
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),
            input_dataframes.get(DF_IN_SOUT_SR_GC, pd.DataFrame()),
            Version,
            **dict_log
        )
        df_output_Sell_Out_FCST_Split_Ratio_GC = df_step05_07_sout_sr_gc
        fn_log_dataframe(df_step05_07_sout_sr_gc, f'df_step05_07_{DF_OUT_SOUT_SR_GC}')

        ################################################################################################################
        # Step 5-8) S/Out FCST Split Ratio_Local
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 58,
            'p_step_desc': 'Step 5-8) S/Out FCST Split Ratio_Local',
            'p_df_name'  : None
        }
        df_step05_08_sout_sr_local = fn_step05_08_build_sout_sr_local(
            df_step03_08_sout_fcst_local,                          # Step 3-2-4 결과
            input_dataframes.get(DF_IN_ITEM_MST, pd.DataFrame()),
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),
            input_dataframes.get(DF_IN_SOUT_SR_LOCAL, pd.DataFrame()),
            Version,
            **dict_log
        )
        df_output_Sell_Out_FCST_Split_Ratio_Local = df_step05_08_sout_sr_local
        fn_log_dataframe(df_step05_08_sout_sr_local, f'df_step05_08_{DF_OUT_SOUT_SR_LOCAL}')

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
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')
        