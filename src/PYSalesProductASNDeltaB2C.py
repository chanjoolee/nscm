# -*- coding: utf-8 -*-
# Auto-generated from Confluence pages by _build_PYSalesProductASNDeltaB2C_from_confluence.py
# Source pages: 124977153, 124977160, 125075467, 124977181, 125075489
# Generated: 2025-11-06 19:58:48



########################################################################################################################
# Begin Confluence Page 1/5 — ID 124977153 — 01. Source1 V2 PYSalesProductASNDeltaB2C (v3)
########################################################################################################################

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
COL_PWEEK                   = 'Time.[Partial Week]'
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
DF_OUT_FLOORING_ASSORT          = 'Output_Flooring_Assortment'

DF_OUT_SIN_GI_AP1               = 'df_output_Sell_In_FCST_GI_AP1'
DF_OUT_SIN_GI_AP2               = 'df_output_Sell_In_FCST_GI_AP2'
DF_OUT_SIN_GI_GC                = 'df_output_Sell_In_FCST_GI_GC'
DF_OUT_SIN_GI_LOCAL             = 'df_output_Sell_In_FCST_GI_Local'

DF_OUT_SOUT_AP1                 = 'df_output_Sell_Out_FCST_AP1'
DF_OUT_SOUT_AP2                 = 'df_output_Sell_Out_FCST_AP2'
DF_OUT_SOUT_GC                  = 'df_output_Sell_Out_FCST_GC'
DF_OUT_SOUT_LOCAL               = 'df_output_Sell_Out_FCST_Local'

DF_OUT_FLOORING_FCST            = 'df_output_Flooring_FCST'
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


########################################################################################################################
# Begin Confluence Page 2/5 — ID 124977160 — 02. Source1 V2 PYSalesProductASNDeltaB2C (v3)
########################################################################################################################


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

#####################################################
#################### Start Step Functions  ##########
#####################################################
# ======================================================
# Step 01: 생성할 Dummy 선별 (+ 삭제용 Output 구성)
# ======================================================

# ---------- (재확인) 컬럼 상수 ----------
# COL_VERSION   = 'Version.[Version Name]'
# COL_SHIP_TO   = 'Sales Domain.[Ship To]'
# COL_ITEM      = 'Item.[Item]'
# COL_LOCATION  = 'Location.[Location]'
# COL_PWEEK     = 'Time.[Partial Week]'
# COL_WEEK      = 'Time.[Week]'

# COL_SIN_DUMMY_AP1   = 'S/In FCST(GI) Dummy_AP1'
# COL_SIN_DUMMY_AP2   = 'S/In FCST(GI) Dummy_AP2'
# COL_SIN_DUMMY_GC    = 'S/In FCST(GI) Dummy_GC'
# COL_SIN_DUMMY_LOCAL = 'S/In FCST(GI) Dummy_Local'
SIN_DUMMY_COLS = [COL_SIN_DUMMY_AP1, COL_SIN_DUMMY_AP2, COL_SIN_DUMMY_GC, COL_SIN_DUMMY_LOCAL]

# COL_SOUT_DUMMY_AP1   = 'S/Out FCST Dummy_AP1'
# COL_SOUT_DUMMY_AP2   = 'S/Out FCST Dummy_AP2'
# COL_SOUT_DUMMY_GC    = 'S/Out FCST Dummy_GC'
# COL_SOUT_DUMMY_LOCAL = 'S/Out FCST Dummy_Local'
SOUT_DUMMY_COLS = [COL_SOUT_DUMMY_AP1, COL_SOUT_DUMMY_AP2, COL_SOUT_DUMMY_GC, COL_SOUT_DUMMY_LOCAL]

# COL_FLOORING_DUMMY = 'Flooring FCST Dummy'

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

def _any_positive_or_notnull(
    df: pd.DataFrame,
    cols: list[str]
) -> pd.Series:
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return pd.Series(False, index=df.index)
    mask_notnull = df[valid_cols].notna().any(axis=1)
    mask_gt0 = (pd.DataFrame({c: pd.to_numeric(df[c], errors='coerce') for c in valid_cols}) > 0).any(axis=1)
    return mask_notnull | mask_gt0

def _empty_like_sin_dummy(
    out_version: str
) -> pd.DataFrame:
    cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK] + SIN_DUMMY_COLS
    df = pd.DataFrame(columns=cols)
    _coerce_dims(df, [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK])
    _to_float32(df, SIN_DUMMY_COLS)
    if out_version:
        df[COL_VERSION] = df[COL_VERSION].astype('category')
    return df
def _empty_like_sout_dummy(
    out_version: str
) -> pd.DataFrame:
    cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK] + SOUT_DUMMY_COLS
    df = pd.DataFrame(columns=cols)
    _coerce_dims(df, [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK])
    _to_float32(df, SOUT_DUMMY_COLS)
    return df
def _empty_like_flooring_dummy(
    out_version: str
) -> pd.DataFrame:
    cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_WEEK, COL_FLOORING_DUMMY]
    df = pd.DataFrame(columns=cols)
    _coerce_dims(df, [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_WEEK])
    _to_float32(df, [COL_FLOORING_DUMMY])
    return df

########################################################################################################################
# Step 1-0) Sales 선정 (SDD에서 eStore ShipTo 제거)
########################################################################################################################
@_decoration_
def fn_step01_00_select_sales(
    df_sdd   : pd.DataFrame,
    df_estore: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Step 1-0) Sales 선정
    ----------------------------------------------------------
    • load: df_in_Sales_Domain_Dimension, df_in_Sales_Domain_Estore
    • eStore ShipTo( df_in_Sales_Domain_Estore[Ship To] )를 기준으로 SDD에서 제외
    • 반환: 필터링된 df_in_Sales_Domain_Dimension (원본 컬럼 유지)    기대 컬럼 (존재 확인):
      - Sales Domain.[Sales Std1..6], Sales Domain.[Ship To]
    """
    if df_sdd is None or df_sdd.empty:
        raise Exception('[Step 1-0] Input df_in_Sales_Domain_Dimension is empty.')

    level_cols = [
        COL_STD1,
        COL_STD2,
        COL_STD3,
        COL_STD4,
        COL_STD5,
        COL_STD6,
        COL_SHIP_TO
    ]
    missing = [c for c in level_cols if c not in df_sdd.columns]
    if missing:
        raise KeyError(f"[Step 1-0] Missing required columns in SDD: {missing}")

    # eStore ShipTo 집합
    estore_set: set[str] = set()
    if df_estore is not None and not df_estore.empty:
        if COL_SHIP_TO not in df_estore.columns:
            raise KeyError(f"[Step 1-0] Missing '{COL_SHIP_TO}' in df_in_Sales_Domain_Estore.")
        estore_set = set(df_estore[COL_SHIP_TO].astype(str).unique().tolist())

    # eStore 제외 필터
    df_sel = df_sdd.loc[
        ~df_sdd[COL_SHIP_TO].astype(str).isin(estore_set)
    ].copy(deep=False)

    # dtype 최소 정리 (레벨/ShipTo만 category 보장)
    for c in level_cols:
        if df_sel[c].dtype.name != 'category':
            df_sel[c] = df_sel[c].astype('category')

    # 메모리 정리
    del df_sdd, df_estore
    gc.collect()

    return df_sel

########################################################################################################################
# Step 1-1) S/In 더미에서 생성할 Sales 선정
########################################################################################################################
@_decoration_
def fn_step01_01_pick_sin_dummy(
    df_sin: pd.DataFrame,
    ship_to_set: Set[str]
) -> pd.DataFrame:
    """
    Step 1-1) S/In 더미에서 생성할 Sales 선정
    ----------------------------------------------------------
    • ShipTo ∈ ship_to_set
    • 더미 measure 중 하나라도 값 존재(>0 또는 not null)
    • 반환: [ShipTo, Item, Location, PartialWeek, 4 measures]
    """
    need = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK] + SIN_DUMMY_COLS
    if df_sin is None or df_sin.empty:
        return pd.DataFrame(columns=need)
    miss = [c for c in need if c not in df_sin.columns]
    if miss:
        raise KeyError(f"[Step 1-1] Missing columns in df_sin: {miss}")

    df = df_sin[need].copy(deep=False)
    df = df[df[COL_SHIP_TO].astype(str).isin({str(x) for x in ship_to_set})]
    df = df[_any_positive_or_notnull(df, SIN_DUMMY_COLS)]
    _coerce_dims(df, [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK])
    _to_float32(df, SIN_DUMMY_COLS)
    return df

########################################################################################################################
# Step 1-2) S/In 더미 Output 구성(삭제용: measure NULL)
########################################################################################################################
@_decoration_
def fn_step01_02_build_output_sin_dummy_delete(
    df_pick: pd.DataFrame,
    out_version: str
) -> pd.DataFrame:
    """
    Step 1-2) S/In 더미 Output 구성(삭제용)
    ----------------------------------------------------------
    • measure 4개 전부 NaN
    • Version 주입 + dtype/category 정리
    """
    if df_pick is None or df_pick.empty:
        return _empty_like_sin_dummy(out_version)

    df = df_pick[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK]].copy(deep=False)
    for c in SIN_DUMMY_COLS:
        df[c] = np.nan

    df.insert(0, COL_VERSION, out_version)
    cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK] + SIN_DUMMY_COLS
    df = df[cols]

    _coerce_dims(df, [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK])
    _to_float32(df, SIN_DUMMY_COLS)
    return df

########################################################################################################################
# Step 1-3) S/Out 더미에서 생성할 Sales 선정
########################################################################################################################
@_decoration_
def fn_step01_03_pick_sout_dummy(
    df_sout: pd.DataFrame,
    ship_to_set: Set[str]
) -> pd.DataFrame:
    """
    Step 1-3) S/Out 더미에서 생성할 Sales 선정
    ----------------------------------------------------------
    • ShipTo ∈ ship_to_set
    • 더미 measure 중 하나라도 값 존재
    """
    need = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK] + SOUT_DUMMY_COLS
    if df_sout is None or df_sout.empty:
        return pd.DataFrame(columns=need)
    miss = [c for c in need if c not in df_sout.columns]
    if miss:
        raise KeyError(f"[Step 1-3] Missing columns in df_sout: {miss}")

    df = df_sout[need].copy(deep=False)
    df = df[df[COL_SHIP_TO].astype(str).isin({str(x) for x in ship_to_set})]
    df = df[_any_positive_or_notnull(df, SOUT_DUMMY_COLS)]
    _coerce_dims(df, [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK])
    _to_float32(df, SOUT_DUMMY_COLS)
    return df

########################################################################################################################
# Step 1-4) S/Out 더미 Output 구성(삭제용: measure NULL)
########################################################################################################################
@_decoration_
def fn_step01_04_build_output_sout_dummy_delete(
    df_pick: pd.DataFrame,
    out_version: str
) -> pd.DataFrame:
    """
    Step 1-4) S/Out 더미 Output 구성(삭제용)
    ----------------------------------------------------------
    • measure 4개 전부 NaN
    • Version 주입 + dtype/category 정리
    """
    if df_pick is None or df_pick.empty:
        return _empty_like_sout_dummy(out_version)

    df = df_pick[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK]].copy(deep=False)
    for c in SOUT_DUMMY_COLS:
        df[c] = np.nan

    df.insert(0, COL_VERSION, out_version)
    cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK] + SOUT_DUMMY_COLS
    df = df[cols]

    _coerce_dims(df, [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK])
    _to_float32(df, SOUT_DUMMY_COLS)
    return df

########################################################################################################################
# Step 1-5) Flooring 더미에서 생성할 Sales 선정
########################################################################################################################
@_decoration_
def fn_step01_05_pick_flooring_dummy(
    df_floor: pd.DataFrame,
    ship_to_set: Set[str]
) -> pd.DataFrame:
    """
    Step 1-5) Flooring 더미에서 생성할 Sales 선정
    ----------------------------------------------------------
    • ShipTo ∈ ship_to_set
    • Flooring Dummy 값 존재
    • 반환: [ShipTo, Item, Week, Flooring Dummy]
    """
    need = [COL_SHIP_TO, COL_ITEM, COL_WEEK, COL_FLOORING_DUMMY]
    if df_floor is None or df_floor.empty:
        return pd.DataFrame(columns=need)
    miss = [c for c in need if c not in df_floor.columns]
    if miss:
        raise KeyError(f"[Step 1-5] Missing columns in df_floor: {miss}")

    df = df_floor[need].copy(deep=False)
    df = df[df[COL_SHIP_TO].astype(str).isin({str(x) for x in ship_to_set})]
    df = df[_any_positive_or_notnull(df, [COL_FLOORING_DUMMY])]
    _coerce_dims(df, [COL_SHIP_TO, COL_ITEM, COL_WEEK])
    _to_float32(df, [COL_FLOORING_DUMMY])
    return df

########################################################################################################################
# Step 1-6) Flooring 더미 Output 구성(삭제용: measure NULL)
########################################################################################################################
@_decoration_
def fn_step01_06_build_output_flooring_dummy_delete(
    df_pick: pd.DataFrame,
    out_version: str
) -> pd.DataFrame:
    """
    Step 1-6) Flooring 더미 Output 구성(삭제용)
    ----------------------------------------------------------
    • Flooring Dummy = NaN
    • Version 주입 + dtype/category 정리
    """
    if df_pick is None or df_pick.empty:
        return _empty_like_flooring_dummy(out_version)

    df = df_pick[[COL_SHIP_TO, COL_ITEM, COL_WEEK]].copy(deep=False)
    df[COL_FLOORING_DUMMY] = np.nan

    df.insert(0, COL_VERSION, out_version)
    cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_WEEK, COL_FLOORING_DUMMY]
    df = df[cols]

    _coerce_dims(df, [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_WEEK])
    _to_float32(df, [COL_FLOORING_DUMMY])
    return df

########################################################################################################################
# Step 2-1) S/In FCST(GI) Dummy → Assortment Output  (NumPy ultra-fast groupby 사용)
########################################################################################################################
@_decoration_
def fn_step02_01_build_sin_assortment(
    df_sin_pick : pd.DataFrame,   # Step 1-1 결과 (ShipTo*Item*Location*PartialWeek + Dummy 4종)
    out_version: str,
    **kwargs
) -> pd.DataFrame:
    """
    입력  : [Sales Domain.[Ship To], Item.[Item], Location.[Location], Time.[Partial Week],
            S/In FCST(GI) Dummy_AP1, _AP2, _GC, _Local]
    처리  : Time 제거, 더미 4종을 ShipTo*Item*Location 기준으로 OR 집계(max) → Assortment로 rename
           * 초대용량 대비: ultra_fast_groupby_numpy_general 사용 (NumPy reduceat)
           * NaN 전파 방지: 더미를 집계 전 0/1 플래그로 변환
    출력  : [Version.[Version Name], Sales Domain.[Ship To], Item.[Item], Location.[Location],
            S/In FCST(GI) Assortment_AP1, _AP2, _GC, _Local]
    dtype : 차원 category, measure float32(값 1.0 또는 NaN)
    """
    # ── 0) 방어/스키마 ────────────────────────────────────────────────────────
    out_cols = [
        COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION,
        COL_SIN_ASSORT_AP1, COL_SIN_ASSORT_AP2, COL_SIN_ASSORT_GC, COL_SIN_ASSORT_LOCAL
    ]
    if df_sin_pick is None or df_sin_pick.empty:
        df_out = pd.DataFrame(columns=out_cols)
        for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION):
            df_out[c] = df_out[c].astype('category')
        for m in (COL_SIN_ASSORT_AP1, COL_SIN_ASSORT_AP2, COL_SIN_ASSORT_GC, COL_SIN_ASSORT_LOCAL):
            df_out[m] = df_out[m].astype('float32')
        return df_out    
    
    need = [
        COL_SHIP_TO, COL_ITEM, COL_LOCATION,
        COL_SIN_DUMMY_AP1, COL_SIN_DUMMY_AP2, COL_SIN_DUMMY_GC, COL_SIN_DUMMY_LOCAL
    ]
    miss = [c for c in need if c not in df_sin_pick.columns]
    if miss:
        raise KeyError(f"[Step 2-1] Required columns missing in df_sin_pick: {miss}")

    # ── 1) 필요 컬럼만 복사 + 더미 → 0/1 플래그로 변환 ───────────────────────
    use = df_sin_pick[need].copy(deep=False)

    # 숫자/문자 섞여 있어도 안전하게: notna & != 0 → 1(참), 그 외 0(거짓)
    for m in (COL_SIN_DUMMY_AP1, COL_SIN_DUMMY_AP2, COL_SIN_DUMMY_GC, COL_SIN_DUMMY_LOCAL):
        v = pd.to_numeric(use[m], errors='coerce')
        use[m] = ((v.notna()) & (v != 0)).astype(np.int8)

    # ── 2) 초고속 그룹바이(OR = max) ─────────────────────────────────────────
    grp_flags = ultra_fast_groupby_numpy_general(
        df=use,
        key_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION],
        aggs={
            COL_SIN_DUMMY_AP1  : 'max',
            COL_SIN_DUMMY_AP2  : 'max',
            COL_SIN_DUMMY_GC   : 'max',
            COL_SIN_DUMMY_LOCAL: 'max',
        },
    )

    # ── 3) 더미 → 어소트먼트 rename + 0→NaN, 1→1.0 ─────────────────────────
    grp_flags.rename(columns={
        COL_SIN_DUMMY_AP1  : COL_SIN_ASSORT_AP1,
        COL_SIN_DUMMY_AP2  : COL_SIN_ASSORT_AP2,
        COL_SIN_DUMMY_GC   : COL_SIN_ASSORT_GC,
        COL_SIN_DUMMY_LOCAL: COL_SIN_ASSORT_LOCAL,
    }, inplace=True)

    for m in (COL_SIN_ASSORT_AP1, COL_SIN_ASSORT_AP2, COL_SIN_ASSORT_GC, COL_SIN_ASSORT_LOCAL):
        grp_flags[m] = grp_flags[m].astype('float32')
        grp_flags[m] = grp_flags[m].where(grp_flags[m] > 0, np.nan)  # 0 → NaN

    # ── 4) Version 주입 + dtype 정리 ────────────────────────────────────────
    grp_flags.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION):
        if grp_flags[c].dtype.name != 'category':
            grp_flags[c] = grp_flags[c].astype('category')

    return grp_flags[out_cols]


########################################################################################################################
# Begin Confluence Page 3/5 — ID 125075467 — 03. Source1 V2 PYSalesProductASNDeltaB2C (v3)
########################################################################################################################


########################################################################################################################
# Step 2-2) S/Out FCST Dummy → Assortment Output  (NumPy ultra-fast groupby 사용)
########################################################################################################################
@_decoration_
def fn_step02_02_build_sout_assortment(
    df_sout_pick: pd.DataFrame,   # Step 1-3 결과 (ShipTo*Item*Location*PartialWeek + Dummy 4종)
    out_version: str,
    **kwargs
) -> pd.DataFrame:
    """
    입력  : [Sales Domain.[Ship To], Item.[Item], Location.[Location], Time.[Partial Week],
            S/Out FCST Dummy_AP1, _AP2, _GC, _Local]
    처리  : Time 제거, 더미 4종을 ShipTo*Item*Location 기준으로 OR 집계(max) → Assortment로 rename
           * 초대용량 대비: ultra_fast_groupby_numpy_general 사용
           * NaN 전파 방지: 더미를 집계 전 0/1 플래그로 변환
    출력  : [Version.[Version Name], Sales Domain.[Ship To], Item.[Item], Location.[Location],
            S/Out FCST Assortment_AP1, _AP2, _GC, _Local]
    dtype : 차원 category, measure float32(값 1.0 또는 NaN)
    """
    out_cols = [
        COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION,
        COL_SOUT_ASSORT_AP1, COL_SOUT_ASSORT_AP2, COL_SOUT_ASSORT_GC, COL_SOUT_ASSORT_LOCAL
    ]
    if df_sout_pick is None or df_sout_pick.empty:
        df_out = pd.DataFrame(columns=out_cols)
        for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION):
            df_out[c] = df_out[c].astype('category')
        for m in (COL_SOUT_ASSORT_AP1, COL_SOUT_ASSORT_AP2, COL_SOUT_ASSORT_GC, COL_SOUT_ASSORT_LOCAL):
            df_out[m] = df_out[m].astype('float32')
        return df_out

    need = [
        COL_SHIP_TO, COL_ITEM, COL_LOCATION,
        COL_SOUT_DUMMY_AP1, COL_SOUT_DUMMY_AP2, COL_SOUT_DUMMY_GC, COL_SOUT_DUMMY_LOCAL
    ]
    miss = [c for c in need if c not in df_sout_pick.columns]
    if miss:
        raise KeyError(f"[Step 2-2] Required columns missing in df_sout_pick: {miss}")

    use = df_sout_pick[need].copy(deep=False)
    for m in (COL_SOUT_DUMMY_AP1, COL_SOUT_DUMMY_AP2, COL_SOUT_DUMMY_GC, COL_SOUT_DUMMY_LOCAL):
        v = pd.to_numeric(use[m], errors='coerce')
        use[m] = ((v.notna()) & (v != 0)).astype(np.int8)

    grp_flags = ultra_fast_groupby_numpy_general(
        df=use,
        key_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION],
        aggs={
            COL_SOUT_DUMMY_AP1  : 'max',
            COL_SOUT_DUMMY_AP2  : 'max',
            COL_SOUT_DUMMY_GC   : 'max',
            COL_SOUT_DUMMY_LOCAL: 'max',
        },
    )

    grp_flags.rename(columns={
        COL_SOUT_DUMMY_AP1  : COL_SOUT_ASSORT_AP1,
        COL_SOUT_DUMMY_AP2  : COL_SOUT_ASSORT_AP2,
        COL_SOUT_DUMMY_GC   : COL_SOUT_ASSORT_GC,
        COL_SOUT_DUMMY_LOCAL: COL_SOUT_ASSORT_LOCAL,
    }, inplace=True)

    for m in (COL_SOUT_ASSORT_AP1, COL_SOUT_ASSORT_AP2, COL_SOUT_ASSORT_GC, COL_SOUT_ASSORT_LOCAL):
        grp_flags[m] = grp_flags[m].astype('float32')
        grp_flags[m] = grp_flags[m].where(grp_flags[m] > 0, np.nan)

    grp_flags.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION):
        if grp_flags[c].dtype.name != 'category':
            grp_flags[c] = grp_flags[c].astype('category')

    return grp_flags[out_cols]

########################################################################################################################
# Step 2-3) Flooring FCST Dummy → Flooring Assortment Output  (NumPy ultra-fast groupby 사용)
########################################################################################################################
@_decoration_
def fn_step02_03_build_flooring_assortment(
    df_floor_pick: pd.DataFrame,  # Step 1-5 결과 (ShipTo*Item*Week + Flooring FCST Dummy)
    out_version : str,
    **kwargs
) -> pd.DataFrame:
    """
    입력  : [Sales Domain.[Ship To], Item.[Item], Time.[Week], Flooring FCST Dummy]
    처리  : Time 삭제, ShipTo*Item 기준 OR 집계(max), 'Flooring FCST Assortment' 로 rename
           * 초대용량 대비: ultra_fast_groupby_numpy_general 사용
           * NaN 전파 방지: 더미를 집계 전 0/1 플래그로 변환
    출력  : [Version.[Version Name], Sales Domain.[Ship To], Item.[Item], Flooring FCST Assortment]
    dtype : 차원 category, measure float32(값 1.0 또는 NaN)
    """
    out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_FLOORING_ASSORT]
    if df_floor_pick is None or df_floor_pick.empty:
        df_out = pd.DataFrame(columns=out_cols)
        for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM):
            df_out[c] = df_out[c].astype('category')
        df_out[COL_FLOORING_ASSORT] = df_out[COL_FLOORING_ASSORT].astype('float32')
        return df_out

    need = [COL_SHIP_TO, COL_ITEM, COL_FLOORING_DUMMY]
    miss = [c for c in need if c not in df_floor_pick.columns]
    if miss:
        raise KeyError(f"[Step 2-3] Required columns missing in df_floor_pick: {miss}")

    use = df_floor_pick[need].copy(deep=False)
    v = pd.to_numeric(use[COL_FLOORING_DUMMY], errors='coerce')
    use[COL_FLOORING_DUMMY] = ((v.notna()) & (v != 0)).astype(np.int8)

    grp_flags = ultra_fast_groupby_numpy_general(
        df=use,
        key_cols=[COL_SHIP_TO, COL_ITEM],
        aggs={COL_FLOORING_DUMMY: 'max'},
    )

    grp_flags.rename(columns={COL_FLOORING_DUMMY: COL_FLOORING_ASSORT}, inplace=True)
    grp_flags[COL_FLOORING_ASSORT] = grp_flags[COL_FLOORING_ASSORT].astype('float32')
    grp_flags[COL_FLOORING_ASSORT] = grp_flags[COL_FLOORING_ASSORT].where(grp_flags[COL_FLOORING_ASSORT] > 0, np.nan)

    grp_flags.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM):
        if grp_flags[c].dtype.name != 'category':
            grp_flags[c] = grp_flags[c].astype('category')

    return grp_flags[out_cols]

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
    mask  = (s_val.notna()) & (s_val != 0)
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
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sin_pick, key_cols, COL_SIN_DUMMY_AP1)
    out_cols = [COL_VERSION, *key_cols, COL_PWEEK, COL_SIN_GI_AP1, COL_SIN_BL_AP1, COL_SIN_NEW_MODEL]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PWEEK],
                         float_cols=[COL_SIN_GI_AP1, COL_SIN_BL_AP1, COL_SIN_NEW_MODEL])

    df = _expand_by_time(keys, df_time_pw, COL_PWEEK)
    df[COL_SIN_GI_AP1]   = 0.0
    df[COL_SIN_BL_AP1]   = 0.0
    df[COL_SIN_NEW_MODEL]= 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK],
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
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sin_pick, key_cols, COL_SIN_DUMMY_AP2)
    out_cols = [COL_VERSION, *key_cols, COL_PWEEK, COL_SIN_GI_AP2, COL_SIN_BL_AP2]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PWEEK],
                         float_cols=[COL_SIN_GI_AP2, COL_SIN_BL_AP2])

    df = _expand_by_time(keys, df_time_pw, COL_PWEEK)
    df[COL_SIN_GI_AP2] = 0.0
    df[COL_SIN_BL_AP2] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK],
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
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sin_pick, key_cols, COL_SIN_DUMMY_GC)
    out_cols = [COL_VERSION, *key_cols, COL_PWEEK, COL_SIN_GC, COL_SIN_BL_GC]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PWEEK],
                         float_cols=[COL_SIN_GC, COL_SIN_BL_GC])

    df = _expand_by_time(keys, df_time_pw, COL_PWEEK)
    df[COL_SIN_GC]    = 0.0
    df[COL_SIN_BL_GC] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK],
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
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sin_pick, key_cols, COL_SIN_DUMMY_LOCAL)
    out_cols = [COL_VERSION, *key_cols, COL_PWEEK, COL_SIN_LOCAL, COL_SIN_BL_LOCAL]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PWEEK],
                         float_cols=[COL_SIN_LOCAL, COL_SIN_BL_LOCAL])

    df = _expand_by_time(keys, df_time_pw, COL_PWEEK)
    df[COL_SIN_LOCAL]    = 0.0
    df[COL_SIN_BL_LOCAL] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK],
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
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sout_pick, key_cols, COL_SOUT_DUMMY_AP1)
    out_cols = [COL_VERSION, *key_cols, COL_PWEEK, COL_SOUT_AP1]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PWEEK], float_cols=[COL_SOUT_AP1])

    df = _expand_by_time(keys, df_time_pw, COL_PWEEK)
    df[COL_SOUT_AP1] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK],
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
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sout_pick, key_cols, COL_SOUT_DUMMY_AP2)
    out_cols = [COL_VERSION, *key_cols, COL_PWEEK, COL_SOUT_AP2]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PWEEK], float_cols=[COL_SOUT_AP2])

    df = _expand_by_time(keys, df_time_pw, COL_PWEEK)
    df[COL_SOUT_AP2] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK],
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
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sout_pick, key_cols, COL_SOUT_DUMMY_GC)
    out_cols = [COL_VERSION, *key_cols, COL_PWEEK, COL_SOUT_GC]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PWEEK], float_cols=[COL_SOUT_GC])

    df = _expand_by_time(keys, df_time_pw, COL_PWEEK)
    df[COL_SOUT_GC] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK],
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
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    keys = _unique_keys_from_dummy(df_sout_pick, key_cols, COL_SOUT_DUMMY_LOCAL)
    out_cols = [COL_VERSION, *key_cols, COL_PWEEK, COL_SOUT_LOCAL]
    if keys.empty:
        return _mk_empty(out_cols, with_cats=[COL_VERSION, *key_cols, COL_PWEEK], float_cols=[COL_SOUT_LOCAL])

    df = _expand_by_time(keys, df_time_pw, COL_PWEEK)
    df[COL_SOUT_LOCAL] = 0.0

    df = _inject_version_and_cast(
        df, out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK],
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
# Step 4 — Estimated Price Local Data 생성
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
    # ISO week 의 월요일 날짜
    try:
        d = date.fromisocalendar(y, w, 1)
    except ValueError:
        return np.nan
    return f"{d.year}{d.month:02d}"

def _mk_empty_cols(cols: list[str], cat_cols: list[str], float_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(columns=cols)
    for c in cat_cols:
        df[c] = df[c].astype('category')
    for c in float_cols:
        df[c] = df[c].astype('float32')
    return df

def _expand_s2_by_pweek(df_keys: pd.DataFrame, df_time_pw: pd.DataFrame) -> pd.DataFrame:
    """
    (ShipTo, Item) × PartialWeek 카티전 곱
    """
    if df_keys is None or df_keys.empty or df_time_pw is None or df_time_pw.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_PWEEK])

    base = df_keys[[COL_SHIP_TO, COL_ITEM]].drop_duplicates().reset_index(drop=True)
    if base[COL_SHIP_TO].dtype.name != 'category':
        base[COL_SHIP_TO] = base[COL_SHIP_TO].astype('category')
    if base[COL_ITEM].dtype.name != 'category':
        base[COL_ITEM] = base[COL_ITEM].astype('category')

    s_pw = df_time_pw[COL_PWEEK]
    if s_pw.dtype.name != 'category':
        s_pw = s_pw.astype('category')
    t = pd.DataFrame({COL_PWEEK: s_pw.values})

    n_b = len(base)
    n_t = len(t)

    rep_b = base.loc[base.index.repeat(n_t)].reset_index(drop=True)
    tile_t = pd.concat([t] * n_b, ignore_index=True)

    df = pd.concat([rep_b, tile_t], axis=1)
    df[COL_PWEEK] = df[COL_PWEEK].astype('category')
    return df

def _inject_version_cast(df: pd.DataFrame, out_version: str, cat_cols: list[str], float_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        cols = [COL_VERSION, *cat_cols, *float_cols]
        return _mk_empty_cols(cols, cat_cols=[COL_VERSION, *cat_cols], float_cols=float_cols)
    df.insert(0, COL_VERSION, out_version)
    for c in [COL_VERSION, *cat_cols]:
        if df[c].dtype.name != 'category':
            df[c] = df[c].astype('category')
    for m in float_cols:
        df[m] = pd.to_numeric(df[m], errors='coerce').astype('float32')
    return df

def _safe_left_merge(df_left: pd.DataFrame, df_right: pd.DataFrame, on: list[str], how: str = 'left', suffixes=('', '_r')):
    if df_right is None or df_right.empty:
        # 오른쪽이 비어있으면 그냥 왼쪽 반환
        return df_left
    return df_left.merge(df_right, how=how, on=on, suffixes=suffixes)

# ──────────────────────────────────────────────────────────────────────────────
# Step 4-0) Estimated Price 생성 대상 (ShipTo*Item) 선정
#   - Step 1-1 S/In Dummy pick, Step 1-3 S/Out Dummy pick 을 각각 ShipTo*Item 으로 묶은 뒤 교집합
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
# Step 4-1) Estimated Price Local 생성 (우선순위 1→6 적용)
#   우선순위 체인
#     1) Estimated Price Modify_Local
#     2) Estimated Price_Local
#     3) Estimated Price Item Std4_Local (ShipTo, ItemStd4, PartialWeek)
#     4) Estimated Price Item Std3_Local (ShipTo, ItemStd3, PartialWeek)
#     5) Estimated Price Item Std2_Local (ShipTo, ItemStd2, PartialWeek)
#     6) Action Plan Price_USD(ShipTo,Item,Month) * Exchange Rate_Local(SalesStd3,PartialWeek)
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step04_01_build_est_price_local(
    df_targets        : pd.DataFrame,   # Step 4-0 결과 (ShipTo*Item)
    df_time_pw        : pd.DataFrame,   # Time.[Partial Week]
    df_item_mst       : pd.DataFrame,   # Item Std2/3/4
    df_sdd            : pd.DataFrame,   # SDD (Sales Std3 매핑용)
    df_est_price      : pd.DataFrame,   # ShipTo*Item*PW → Mod, Local
    df_ep_std4_local  : pd.DataFrame,   # ItemStd4*ShipTo*PW → 값
    df_ep_std3_local  : pd.DataFrame,   # ItemStd3*ShipTo*PW → 값
    df_ep_std2_local  : pd.DataFrame,   # ItemStd2*ShipTo*PW → 값
    df_ap_price       : pd.DataFrame,   # ShipTo*Item*Month → USD
    df_exrate_local   : pd.DataFrame,   # SalesStd3*PW → 환율
    out_version       : str,
    **kwargs
) -> pd.DataFrame:
    # ── 0) 방어 ─────────────────────────────────────────────────────────────
    out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_EST_PRICE_LOCAL]
    if df_targets is None or df_targets.empty or df_time_pw is None or df_time_pw.empty:
        return _mk_empty_cols(out_cols,
                              cat_cols=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK],
                              float_cols=[COL_EST_PRICE_LOCAL])

    # ── 1) (ShipTo*Item)×PW 확장 ───────────────────────────────────────────
    df = _expand_s2_by_pweek(df_targets, df_time_pw)  # [ShipTo, Item, PW]

    # ── 2) 보조 키 컬럼 주입: Item Std2/3/4, Sales Std3 ────────────────────
    # Item Std2/3/4
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

    # Sales Std3 (ShipTo → Std3)
    if df_sdd is not None and not df_sdd.empty:
        sdd_map = df_sdd[[COL_SHIP_TO, COL_STD3]].drop_duplicates()
        df = _safe_left_merge(df, sdd_map, on=[COL_SHIP_TO])
    else:
        df[COL_STD3] = np.nan

    # ── 3) 1,2 단계 소스 Merge ─────────────────────────────────────────────
    # (1) df_est_price : ShipTo*Item*PW
    if df_est_price is not None and not df_est_price.empty:
        use_cols = [COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_EST_PRICE_MOD_LOCAL, COL_EST_PRICE_LOCAL]
        df = _safe_left_merge(df, df_est_price[use_cols], on=[COL_SHIP_TO, COL_ITEM, COL_PWEEK])
    else:
        df[COL_EST_PRICE_MOD_LOCAL] = np.nan
        df[COL_EST_PRICE_LOCAL]     = np.nan

    # ── 4) 3,4,5 단계(Std4/Std3/Std2) 소스 Merge ───────────────────────────
    # Std4
    if df_ep_std4_local is not None and not df_ep_std4_local.empty:
        m4 = df_ep_std4_local[[COL_ITEM_STD4, COL_SHIP_TO, COL_PWEEK, COL_EP_STD4_LOCAL]].drop_duplicates()
        df = _safe_left_merge(df, m4, on=[COL_ITEM_STD4, COL_SHIP_TO, COL_PWEEK])
    else:
        df[COL_EP_STD4_LOCAL] = np.nan

    # Std3
    if df_ep_std3_local is not None and not df_ep_std3_local.empty:
        m3 = df_ep_std3_local[[COL_ITEM_STD3, COL_SHIP_TO, COL_PWEEK, COL_EP_STD3_LOCAL]].drop_duplicates()
        df = _safe_left_merge(df, m3, on=[COL_ITEM_STD3, COL_SHIP_TO, COL_PWEEK])
    else:
        df[COL_EP_STD3_LOCAL] = np.nan

    # Std2
    if df_ep_std2_local is not None and not df_ep_std2_local.empty:
        m2 = df_ep_std2_local[[COL_ITEM_STD2, COL_SHIP_TO, COL_PWEEK, COL_EP_STD2_LOCAL]].drop_duplicates()
        df = _safe_left_merge(df, m2, on=[COL_ITEM_STD2, COL_SHIP_TO, COL_PWEEK])
    else:
        df[COL_EP_STD2_LOCAL] = np.nan

    # ── 5) 6 단계(AP USD * EXRATE) ─────────────────────────────────────────
    # PartialWeek → Month
    df[COL_MONTH] = df[COL_PWEEK].astype(str).map(_pweek_to_month_str)

    # AP USD (ShipTo*Item*Month)
    if df_ap_price is not None and not df_ap_price.empty and COL_MONTH in df_ap_price.columns:
        ap = df_ap_price[[COL_SHIP_TO, COL_ITEM, COL_MONTH, COL_AP_PRICE_USD]].drop_duplicates()
        df = _safe_left_merge(df, ap, on=[COL_SHIP_TO, COL_ITEM, COL_MONTH])
    else:
        df[COL_AP_PRICE_USD] = np.nan

    # EXRATE (Std3*PW)
    if df_exrate_local is not None and not df_exrate_local.empty:
        ex = df_exrate_local[[COL_STD3, COL_PWEEK, COL_EXRATE_LOCAL]].drop_duplicates()
        df = _safe_left_merge(df, ex, on=[COL_STD3, COL_PWEEK])
    else:
        df[COL_EXRATE_LOCAL] = np.nan

    df['__ap_local'] = pd.to_numeric(df[COL_AP_PRICE_USD], errors='coerce') * pd.to_numeric(df[COL_EXRATE_LOCAL], errors='coerce')

    # ── 6) 우선순위 1→6 coalesce ──────────────────────────────────────────
    coalesce_cols = [
        COL_EST_PRICE_MOD_LOCAL,
        COL_EST_PRICE_LOCAL,
        COL_EP_STD4_LOCAL,
        COL_EP_STD3_LOCAL,
        COL_EP_STD2_LOCAL,
        '__ap_local'
    ]
    # 첫 컬럼부터 순차 적용
    out_val = None
    for c in coalesce_cols:
        v = pd.to_numeric(df[c], errors='coerce')
        out_val = v if out_val is None else out_val.fillna(v)

    df[COL_EST_PRICE_LOCAL] = out_val.astype('float32')

    # ── 7) 출력 스키마 + Version 주입 ─────────────────────────────────────
    out = df[[COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_EST_PRICE_LOCAL]].copy(deep=False)
    out = _inject_version_cast(
        out,
        out_version=out_version,
        cat_cols=[COL_SHIP_TO, COL_ITEM, COL_PWEEK],
        float_cols=[COL_EST_PRICE_LOCAL]
    )

    # 메모리 정리
    del df
    gc.collect()
    return out[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_EST_PRICE_LOCAL]]

# ──────────────────────────────────────────────────────────────────────────────
# Step 4-2) Estimated Price_Local 결측치 하위 Lv 평균으로 보완 (선택/고급)
#   - SDD를 이용해 parent ShipTo → children ShipTo 목록 구성
#   - parent,item,pw 의 값이 NaN 이면 children 동일 item,pw 평균으로 채움
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
    # 각 행에서 (Std1..Std6, ShipTo)을 읽어, 각 StdX 값을 parent 로 보고 ShipTo 를 child 로 등록
    # (실사용시, children 중 df_est_local 에 존재하는 ShipTo 만 유지)
    parent_children = {}
    if df_sdd is not None and not df_sdd.empty:
        cols = [COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6, COL_SHIP_TO]
        sdd = df_sdd[cols].drop_duplicates()
        for _, r in sdd.iterrows():
            child = str(r[COL_SHIP_TO])
            for pcol in [COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6]:
                parent = str(r[pcol])
                parent_children.setdefault(parent, set()).add(child)

    # df_est_local 에 존재하는 ShipTo 로 children 필터링
    existing_shipto = set(df_est_local[COL_SHIP_TO].astype(str).unique())
    parent_children = {
        p: sorted([c for c in ch if c in existing_shipto])
        for p, ch in parent_children.items()
        if len([c for c in ch if c in existing_shipto]) > 0
    }
    if not parent_children:
        return df_est_local

    # 평균값 산출용 DF (child rows)
    # parent 를 확장하여 (parent, child) 쌍 테이블 만들고 child 데이터를 조인 후 parent,item,pw 로 groupby 평균
    map_rows = []
    for p, childs in parent_children.items():
        for c in childs:
            map_rows.append((p, c))
    df_map = pd.DataFrame(map_rows, columns=['__parent', '__child'])

    df_child = df_est_local.rename(columns={COL_SHIP_TO: '__child'})[['__child', COL_ITEM, COL_PWEEK, COL_EST_PRICE_LOCAL]]
    df_par_join = df_map.merge(df_child, on='__child', how='left')  # [parent, child, item, pw, val]

    grp = df_par_join.groupby(['__parent', COL_ITEM, COL_PWEEK], as_index=False)[COL_EST_PRICE_LOCAL].mean()
    grp.rename(columns={'__parent': COL_SHIP_TO, COL_EST_PRICE_LOCAL: '__avg_child_val'}, inplace=True)

    # 결측치 채우기
    out = df_est_local.merge(grp, on=[COL_SHIP_TO, COL_ITEM, COL_PWEEK], how='left')
    need = out[COL_EST_PRICE_LOCAL].isna() & out['__avg_child_val'].notna()
    if need.any():
        out.loc[need, COL_EST_PRICE_LOCAL] = out.loc[need, '__avg_child_val'].astype('float32')

    out.drop(columns=['__avg_child_val'], inplace=True)
    return out


########################################################################################################################
# Begin Confluence Page 4/5 — ID 124977181 — 04. Source1 V2 PYSalesProductASNDeltaB2C (v3)
########################################################################################################################


# ──────────────────────────────────────────────────────────────────────────────
# Step 4-4) Estimated Price Local Output 포맷
#   - 스키마/타입 고정
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step04_04_format_est_price_output(
    df_est_local: pd.DataFrame,
    out_version : str,
    **kwargs
) -> pd.DataFrame:
    cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_EST_PRICE_LOCAL]
    if df_est_local is None or df_est_local.empty:
        return _mk_empty_cols(cols,
                              cat_cols=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK],
                              float_cols=[COL_EST_PRICE_LOCAL])

    df = df_est_local.copy(deep=False)
    if COL_VERSION not in df.columns:
        df.insert(0, COL_VERSION, out_version)

    for c in [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK]:
        if df[c].dtype.name != 'category':
            df[c] = df[c].astype('category')

    df[COL_EST_PRICE_LOCAL] = pd.to_numeric(df[COL_EST_PRICE_LOCAL], errors='coerce').astype('float32')

    return df[cols]

########################################################################################################################
# Step 5) Split Ratio Data 생성	
########################################################################################################################
########################################################################################################################
# [공통] Split Ratio 빌더 (S/In, S/Out 공용)
########################################################################################################################
def _empty_split_ratio_schema(with_loc: bool, meas_col: str) -> pd.DataFrame:
    cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if with_loc else []) + [COL_PWEEK, meas_col]
    return pd.DataFrame(columns=cols)

def _unique_keyframe_from_step3(
    df_in: pd.DataFrame,
    *,
    with_loc: bool
) -> pd.DataFrame:
    """
    Step3 결과에서 Split Ratio 적용을 위한 '키 스켈레톤' 생성
      - 필요 컬럼만 취해 중복 제거 (ShipTo*Item*(Location)*PW)
      - 빈 입력 시 빈 DF 반환
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if with_loc else []) + [COL_PWEEK])

    need_cols = [COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if with_loc else []) + [COL_PWEEK]
    use_cols  = [c for c in need_cols if c in df_in.columns]
    df = df_in.loc[:, use_cols].copy(deep=False)
    return df.drop_duplicates(ignore_index=True)

def _map_shipto_to_std2(df_sdd: pd.DataFrame) -> pd.DataFrame:
    """
    SDD에서 ShipTo ↔ Std2 매핑만 추출 (중복 제거)
    """
    if df_sdd is None or df_sdd.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_STD2])
    m = df_sdd[[COL_SHIP_TO, COL_STD2]].drop_duplicates()
    return m

def _map_item_to_std1(df_item_mst: pd.DataFrame) -> pd.DataFrame:
    """
    Item Master에서 Item ↔ Item Std1 매핑 추출
    """
    if df_item_mst is None or df_item_mst.empty:
        return pd.DataFrame(columns=[COL_ITEM, COL_ITEM_STD1])
    m = df_item_mst[[COL_ITEM, COL_ITEM_STD1]].drop_duplicates()
    return m

def _build_split_ratio_generic(
    df_base_step3: pd.DataFrame,
    df_item_mst: pd.DataFrame,
    df_sdd: pd.DataFrame,
    df_ratio_in: pd.DataFrame,   # AP1/AP2/GC/Local 각각의 인풋
    *,
    meas_col: str,               # 최종 산출 컬럼명(예: COL_SIN_SR_AP1)
    with_loc: bool,              # S/In=True, S/Out=True (둘 다 Location 존재), Flooring용 없음
    out_version: str
) -> pd.DataFrame:
    """
    공통 Split Ratio 생성:
      1) Step3 베이스에서 ShipTo*Item*(Loc)*PW 키 추출
      2) ShipTo→Std2, Item→Std1 매핑 조인
      3) (Std2, Std1, PW) 로 df_ratio_in 과 매칭하여 ratio 값 취득
      4) 값 없는 행은 생성하지 않음 (삭제)
      5) 스키마/타입/Version 정리
    """
    # ── 0) 빈 입력 방어 ──────────────────────────────────────────────────────
    if df_base_step3 is None or df_base_step3.empty:
        return _empty_split_ratio_schema(with_loc, meas_col)

    # ── 1) 키 스켈레톤 ─────────────────────────────────────────────────────
    key_df = _unique_keyframe_from_step3(df_base_step3, with_loc=with_loc)
    if key_df.empty:
        return _empty_split_ratio_schema(with_loc, meas_col)

    # ── 2) ShipTo→Std2, Item→Std1 매핑 ─────────────────────────────────────
    map_std2 = _map_shipto_to_std2(df_sdd)
    map_std1 = _map_item_to_std1(df_item_mst)

    df = key_df.merge(map_std2, on=COL_SHIP_TO, how='left')
    df = df.merge(map_std1, on=COL_ITEM,    how='left')

    # 매핑 실패(Std2/Std1 결측)은 이후 조인 실패로 자연스레 삭제됨

    # ── 3) Ratio 인풋 준비 & 조인 ──────────────────────────────────────────
    # 인풋은 (Std2, Std1, PW, Ratio) 그레인
    if df_ratio_in is None or df_ratio_in.empty:
        return _empty_split_ratio_schema(with_loc, meas_col)

    need_cols = [COL_STD2, COL_ITEM_STD1, COL_PWEEK, meas_col]
    miss_cols = [c for c in need_cols if c not in df_ratio_in.columns]
    if miss_cols:
        # 필요한 컬럼이 없다면 빈 스키마 반환
        return _empty_split_ratio_schema(with_loc, meas_col)

    base_cols = [COL_STD2, COL_ITEM_STD1, COL_PWEEK]
    rmap = df_ratio_in.loc[:, need_cols].copy(deep=False)

    # 숫자화
    rmap[meas_col] = pd.to_numeric(rmap[meas_col], errors='coerce').astype('float32')

    df = df.merge(rmap, on=base_cols, how='left')

    # ── 4) 값 없는 행은 생성하지 않음(삭제) ────────────────────────────────
    df = df[df[meas_col].notna()].copy(deep=False)

    # ── 5) 스키마 정리 ──────────────────────────────────────────────────────
    if df.empty:
        return _empty_split_ratio_schema(with_loc, meas_col)

    # 출력 컬럼
    out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM] + ([COL_LOCATION] if with_loc else []) + [COL_PWEEK, meas_col]
    if COL_VERSION not in df.columns:
        df.insert(0, COL_VERSION, out_version)
    else:
        df[COL_VERSION] = out_version

    # dtype 정리
    for c in [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK] + ([COL_LOCATION] if with_loc else []):
        if c in df.columns:
            df[c] = df[c].astype('category')

    df[meas_col] = df[meas_col].astype('float32')

    # 최종 컬럼 순서
    return df[out_cols]

########################################################################################################################
# Step 5-1) S/In FCST(GI) Split Ratio_AP1
########################################################################################################################
@_decoration_
def fn_step05_01_build_sin_sr_ap1(
    df_step03_ap1: pd.DataFrame,       # Step 3-1-1 결과 (AP1)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_ap1    : pd.DataFrame,       # DF_IN_SIN_SR_AP1
    out_version  : str
) -> pd.DataFrame:
    """
    • Step 3-1 AP1 결과의 ShipTo*Item*Location*PW 스켈레톤을 기준으로
      (Sales Std2, Item Std1, Partial Week) 매핑 → AP1 Split Ratio 생성
    • 값이 없는 조합은 생성하지 않음
    """
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_ap1,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_ap1,
        meas_col      = COL_SIN_SR_AP1,
        with_loc      = True,
        out_version   = out_version
    )
########################################################################################################################
# Step 5-2) S/In FCST(GI) Split Ratio_AP2
########################################################################################################################
@_decoration_
def fn_step05_02_build_sin_sr_ap2(
    df_step03_ap2: pd.DataFrame,       # Step 3-1-2 결과 (AP2)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_ap2    : pd.DataFrame,       # DF_IN_SIN_SR_AP2
    out_version  : str
) -> pd.DataFrame:
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_ap2,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_ap2,
        meas_col      = COL_SIN_SR_AP2,
        with_loc      = True,
        out_version   = out_version
    )
########################################################################################################################
# Step 5-3) S/In FCST(GI) Split Ratio_GC
########################################################################################################################
@_decoration_
def fn_step05_03_build_sin_sr_gc(
    df_step03_gc : pd.DataFrame,       # Step 3-1-3 결과 (GC)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_gc     : pd.DataFrame,       # DF_IN_SIN_SR_GC
    out_version  : str
) -> pd.DataFrame:
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_gc,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_gc,
        meas_col      = COL_SIN_SR_GC,
        with_loc      = True,
        out_version   = out_version
    )
########################################################################################################################
# Step 5-4) S/In FCST(GI) Split Ratio_Local
########################################################################################################################
@_decoration_
def fn_step05_04_build_sin_sr_local(
    df_step03_local: pd.DataFrame,     # Step 3-1-4 결과 (Local)
    df_item_mst    : pd.DataFrame,
    df_sdd         : pd.DataFrame,
    df_sr_local    : pd.DataFrame,     # DF_IN_SIN_SR_LOCAL
    out_version    : str
) -> pd.DataFrame:
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_local,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_local,
        meas_col      = COL_SIN_SR_LOCAL,
        with_loc      = True,
        out_version   = out_version
    )
########################################################################################################################
# Step 5-5) S/Out FCST Split Ratio_AP1
########################################################################################################################
@_decoration_
def fn_step05_05_build_sout_sr_ap1(
    df_step03_ap1: pd.DataFrame,       # Step 3-2-1 결과 (AP1)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_ap1    : pd.DataFrame,       # DF_IN_SOUT_SR_AP1
    out_version  : str
) -> pd.DataFrame:
    """
    • Step 3-2 AP1 결과의 ShipTo*Item*Location*PW 스켈레톤 기준
    • (Sales Std2, Item Std1, Partial Week) 매핑 → AP1 Split Ratio 생성
    • 값 미존재 시 행 생성 안함 (삭제)
    """
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_ap1,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_ap1,
        meas_col      = COL_SOUT_SR_AP1,
        with_loc      = True,
        out_version   = out_version
    )

########################################################################################################################
# Step 5-6) S/Out FCST Split Ratio_AP2
########################################################################################################################
@_decoration_
def fn_step05_06_build_sout_sr_ap2(
    df_step03_ap2: pd.DataFrame,       # Step 3-2-2 결과 (AP2)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_ap2    : pd.DataFrame,       # DF_IN_SOUT_SR_AP2
    out_version  : str
) -> pd.DataFrame:
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_ap2,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_ap2,
        meas_col      = COL_SOUT_SR_AP2,
        with_loc      = True,
        out_version   = out_version
    )
########################################################################################################################
# Step 5-7) S/Out FCST Split Ratio_GC
########################################################################################################################
@_decoration_
def fn_step05_07_build_sout_sr_gc(
    df_step03_gc : pd.DataFrame,       # Step 3-2-3 결과 (GC)
    df_item_mst  : pd.DataFrame,
    df_sdd       : pd.DataFrame,
    df_sr_gc     : pd.DataFrame,       # DF_IN_SOUT_SR_GC
    out_version  : str
) -> pd.DataFrame:
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_gc,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_gc,
        meas_col      = COL_SOUT_SR_GC,
        with_loc      = True,
        out_version   = out_version
    )
########################################################################################################################
# Step 5-8) S/Out FCST Split Ratio_Local
########################################################################################################################
@_decoration_
def fn_step05_08_build_sout_sr_local(
    df_step03_local: pd.DataFrame,     # Step 3-2-4 결과 (Local)
    df_item_mst    : pd.DataFrame,
    df_sdd         : pd.DataFrame,
    df_sr_local    : pd.DataFrame,     # DF_IN_SOUT_SR_LOCAL
    out_version    : str
) -> pd.DataFrame:
    return _build_split_ratio_generic(
        df_base_step3 = df_step03_local,
        df_item_mst   = df_item_mst,
        df_sdd        = df_sdd,
        df_ratio_in   = df_sr_local,
        meas_col      = COL_SOUT_SR_LOCAL,
        with_loc      = True,
        out_version   = out_version
    )

########################################################################################################################
# Spec2: Parameter utilities (no argparse, o9 plugin style)
########################################################################################################################
def parse_sales_item_location_str(s: str) -> pd.DataFrame:
    cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    if not s:
        return pd.DataFrame(columns=cols)
    rows = []
    for token in str(s).split('^'):
        token = token.strip()
        if not token:
            continue
        parts = [p.strip() for p in token.split(':')]
        if len(parts) >= 2:
            ship_to, item = parts[0], parts[1]
            loc = parts[2] if len(parts) >= 3 and parts[2] else '-'
            rows.append({COL_SHIP_TO: ship_to, COL_ITEM: item, COL_LOCATION: loc})
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype('category')
    return df

def fn_step01_00_parse_sales_item_location(sales_item_location: str, df_estore: pd.DataFrame) -> pd.DataFrame:
    df = parse_sales_item_location_str(sales_item_location)
    if df.empty:
        return df
    if df_estore is not None and not df_estore.empty and COL_SHIP_TO in df_estore.columns:
        estore_set = set(df_estore[COL_SHIP_TO].astype(str).unique().tolist())
        df = df[~df[COL_SHIP_TO].astype(str).isin(estore_set)].copy(deep=False)
    return df

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

        # =====================================================================================
        # Main: Step01 실행(요청 포맷 준수: dict_log + fn_log_dataframe 사용)
        # =====================================================================================
        # 사전 준비
        # Version = "CWV_DP"


        ################################################################################################################
        # Step 1-0) Sales 선정 (SDD - eStore ShipTo)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 10,
            'p_step_desc': 'Step 1-0) Sales 선정',
            'p_df_name'  : None
        }
        if 'salesItemLocation' in locals() and salesItemLocation:
            df_step01_00_sales_sdd = fn_step01_00_parse_sales_item_location(
                salesItemLocation,
                input_dataframes.get(DF_IN_ESTORE, pd.DataFrame())
            )
        else:
            df_step01_00_sales_sdd = fn_step01_00_select_sales(
                input_dataframes[DF_IN_SDD],
                input_dataframes.get(DF_IN_ESTORE, pd.DataFrame()),
                **dict_log
            )
        fn_log_dataframe(df_step01_00_sales_sdd, 'df_step01_00_sales_sdd')
        
        # 이후 단계에서 사용할 ShipTo 집합
        ship_to_set = set(df_step01_00_sales_sdd[COL_SHIP_TO].astype(str).unique().tolist())

        ################################################################################################################
        # Step 1-1) S/In 더미에서 생성할 Sales 선정
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 11,
            'p_step_desc': 'Step 1-1) S/In 더미에서 생성할 Sales 선정',
            'p_df_name'  : None
        }
        df_step01_01_sin_pick = fn_step01_01_pick_sin_dummy(
            input_dataframes.get(DF_IN_SIN_DUMMY, pd.DataFrame()),
            ship_to_set,
            **dict_log
        )
        fn_log_dataframe(df_step01_01_sin_pick, 'df_step01_01_sin_pick')

        ################################################################################################################
        # Step 1-2) S/In 더미 Output 구성(삭제용)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 12,
            'p_step_desc': 'Step 1-2) S/In 더미 Output 구성(삭제용)',
            'p_df_name'  : None
        }
        df_step01_02_out_sin_dummy = fn_step01_02_build_output_sin_dummy_delete(
            df_step01_01_sin_pick,
            Version,
            **dict_log
        )
        # o9에서 인식할 수 있도록 output을 정의한다. <== 1. (Output 1-2) Output_SIn_Dummy	
        Output_SIn_Dummy = df_step01_02_out_sin_dummy
        fn_log_dataframe(df_step01_02_out_sin_dummy, f'df_step01_02_{DF_OUT_SIN_DUMMY}')

        ################################################################################################################
        # Step 1-3) S/Out 더미에서 생성할 Sales 선정
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 13,
            'p_step_desc': 'Step 1-3) S/Out 더미에서 생성할 Sales 선정',
            'p_df_name'  : None
        }
        df_step01_03_sout_pick = fn_step01_03_pick_sout_dummy(
            input_dataframes.get(DF_IN_SOUT_DUMMY, pd.DataFrame()),
            ship_to_set,
            **dict_log
        )
        fn_log_dataframe(df_step01_03_sout_pick, 'df_step01_03_sout_pick')

        ################################################################################################################
        # Step 1-4) S/Out 더미 Output 구성(삭제용)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 14,
            'p_step_desc': 'Step 1-4) S/Out 더미 Output 구성(삭제용)',
            'p_df_name'  : None
        }
        df_step01_04_out_sout_dummy = fn_step01_04_build_output_sout_dummy_delete(
            df_step01_03_sout_pick,
            Version,
            **dict_log
        )
        # o9에서 인식할 수 있도록 output을 정의한다.  ← 2. (Output 1-4) Output_SOut_Dummy
        Output_SOut_Dummy = df_step01_04_out_sout_dummy
        fn_log_dataframe(df_step01_04_out_sout_dummy, f'df_step01_04_{DF_OUT_SOUT_DUMMY}')

        ################################################################################################################
        # Step 1-5) Flooring 더미에서 생성할 Sales 선정
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 15,
            'p_step_desc': 'Step 1-5) Flooring 더미에서 생성할 Sales 선정',
            'p_df_name'  : None
        }
        df_step01_05_flooring_pick = fn_step01_05_pick_flooring_dummy(
            input_dataframes.get(DF_IN_FLOORING_DUMMY, pd.DataFrame()),
            ship_to_set,
            **dict_log
        )
        
        fn_log_dataframe(df_step01_05_flooring_pick, 'df_step01_05_floor_pick')

        ################################################################################################################
        # Step 1-6) Flooring 더미 Output 구성(삭제용)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 16,
            'p_step_desc': 'Step 1-6) Flooring 더미 Output 구성(삭제용)',
            'p_df_name'  : None
        }
        df_step01_06_out_flooring_dummy = fn_step01_06_build_output_flooring_dummy_delete(
            df_step01_05_flooring_pick,
            Version,
            **dict_log
        )
        # o9에서 인식할 수 있도록 output을 정의한다.  ← 3. (Output 1-5) Output_Flooring_Dummy
        Output_Flooring_Dummy = df_step01_06_out_flooring_dummy
        fn_log_dataframe(df_step01_06_out_flooring_dummy, f'df_step01_06_{DF_OUT_FLOORING_DUMMY}')
        
        ################################################################################################################
        # Step 1-7) 모두 빈 경우 종료 플래그
        ################################################################################################################
        all_empty_step01 = (
            df_step01_02_out_sin_dummy.empty
            and df_step01_04_out_sout_dummy.empty
            and df_step01_06_out_flooring_dummy.empty
        )
        # 필요 시 이 플래그를 체크하여 조기 종료 처리(상위 메인 제어부에서)
        if all_empty_step01:
            raise Exception('Step 1-2,Step 1-4,Step 1-6 is empty.')
        
        ################################################################################################################
        # Step 2-1) S/In 더미 → Assortment Output
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 21,
            'p_step_desc': 'Step 2-1) S/In 더미 → Assortment Output',
            'p_df_name'  : None
        }
        df_step02_01_out_sin_assort = fn_step02_01_build_sin_assortment(
            df_step01_01_sin_pick,  # Step 1-1 결과
            Version,
            **dict_log
        )
        # o9 인식용 출력 바인딩  ← 4. (Output 2-1) Output_SIn_Assortment
        Output_SIn_Assortment = df_step02_01_out_sin_assort
        fn_log_dataframe(df_step02_01_out_sin_assort, f'df_step02_01_{DF_OUT_SIN_ASSORT}')
        
        ################################################################################################################
        # Step 2-2) S/Out 더미 → Assortment Output
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 22,
            'p_step_desc': 'Step 2-2) S/Out 더미 → Assortment Output',
            'p_df_name'  : None
        }
        df_step02_02_out_sout_assort = fn_step02_02_build_sout_assortment(
            df_step01_03_sout_pick,  # Step 1-3 결과
            Version,
            **dict_log
        )
        # o9 인식용 출력 바인딩  ← 5. (Output 2-2) Output_SOut_Assortment
        Output_SOut_Assortment = df_step02_02_out_sout_assort
        fn_log_dataframe(df_step02_02_out_sout_assort, f'df_step02_02_{DF_OUT_SOUT_ASSORT}')

        ################################################################################################################
        # Step 2-3) Flooring 더미 → Flooring Assortment Output
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 23,
            'p_step_desc': 'Step 2-3) Flooring 더미 → Flooring Assortment Output',
            'p_df_name'  : None
        }
        df_step02_03_out_flooring_assort = fn_step02_03_build_flooring_assortment(
            df_step01_05_flooring_pick,  # Step 1-5 결과
            Version,
            **dict_log
        )
        # o9 인식용 출력 바인딩  ← 6. (Output 2-3) Output_Flooring_Assortment
        Output_Flooring_Assortment = df_step02_03_out_flooring_assort
        fn_log_dataframe(df_step02_03_out_flooring_assort, f'df_step02_03_{DF_OUT_FLOORING_ASSORT}')


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
        df_step03_01_ap1 = df_output_Sell_In_FCST_GI_AP1
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
        df_step03_02_ap2 = df_output_Sell_In_FCST_GI_AP2
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
        df_step03_03_gc = df_output_Sell_In_FCST_GI_GC
        fn_log_dataframe(df_output_Sell_In_FCST_GI_GC, f'df_step03_01_03_{DF_OUT_SIN_GI_GC}')


########################################################################################################################
# Begin Confluence Page 5/5 — ID 125075489 — 05. Source1 V2 PYSalesProductASNDeltaB2C (v2)
########################################################################################################################


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
        df_step03_04_local = df_output_Sell_In_FCST_GI_Local
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
        df_step03_05_sout_ap1 = df_output_Sell_Out_FCST_AP1
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
        df_step03_06_sout_ap2 = df_output_Sell_Out_FCST_AP2
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
        df_step03_07_sout_gc = df_output_Sell_Out_FCST_GC
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
        df_step03_08_sout_local = df_output_Sell_Out_FCST_Local
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
            df_step01_05_flooring_pick,
            input_dataframes[DF_IN_TIME_W],
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Flooring_FCST, f'step03_03_{DF_OUT_FLOORING_FCST}')


        ################################################################################################################
        # Step 3-4) df_output_BO_FCST  추후스펙. 빈 dataframe
        ################################################################################################################
        COL_VIRTUAL_BO_ID               = 'Virtual BO ID.[Virtual BO ID]'
        COL_BO_ID                       = 'BO ID.[BO ID]'
        COL_BO_FCST                     = 'BO FCST'
        df_output_BO_FCST = pd.DataFrame([
            COL_VERSION, COL_ITEM, COL_SHIP_TO, COL_LOCATION,
            COL_VIRTUAL_BO_ID, COL_BO_ID, COL_PWEEK, COL_BO_FCST])
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
        # Step 4-1) Estimated Price Local 생성 (우선순위 1→6 적용)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 41,
            'p_step_desc': 'Step 4-1) Estimated Price Local 생성',
            'p_df_name'  : None
        }
        df_step04_01_est_local = fn_step04_01_build_est_price_local(
            df_step04_00_targets,
            input_dataframes[DF_IN_TIME_PW],
            input_dataframes.get(DF_IN_ITEM_MST, pd.DataFrame()),
            input_dataframes.get(DF_IN_SDD, pd.DataFrame()),
            input_dataframes.get(DF_IN_EST_PRICE, pd.DataFrame()),
            input_dataframes.get(DF_IN_EP_STD4_LOCAL, pd.DataFrame()),
            input_dataframes.get(DF_IN_EP_STD3_LOCAL, pd.DataFrame()),
            input_dataframes.get(DF_IN_EP_STD2_LOCAL, pd.DataFrame()),
            input_dataframes.get(DF_IN_AP_PRICE, pd.DataFrame()),
            input_dataframes.get(DF_IN_EXRATE_LOCAL, pd.DataFrame()),
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
            df_step03_01_ap1,                                # Step 3-1-1 결과
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
            df_step03_02_ap2,                                # Step 3-1-2 결과
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
            df_step03_03_gc,                                  # Step 3-1-3 결과
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
            df_step03_04_local,                               # Step 3-1-4 결과
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
            df_step03_05_sout_ap1,                            # Step 3-2-1 결과
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
            df_step03_06_sout_ap2,                            # Step 3-2-2 결과
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
            df_step03_07_sout_gc,                             # Step 3-2-3 결과
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
            df_step03_08_sout_local,                          # Step 3-2-4 결과
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
        
