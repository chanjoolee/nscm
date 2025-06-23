import os,sys
import time,datetime,shutil
import inspect
import traceback
import pandas as pd
from NSCMCommon import NSCMCommon as common
# from typing_extensions import Literal
import glob
import numpy as np
from typing import Collection, Tuple,Union,Dict
import re
import gc

########################################################################################################################
# Local 개발 시에 필요한 공통 변수 선언
########################################################################################################################
# o9에 저장된 instanceName
str_instance = 'PYForecastB2BLockAndRolling'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"
is_local = common.gfn_get_isLocal()
is_print = True
flag_csv = True
flag_exception = True
# Global variable for max_week
max_week = None
current_partial_week = None
max_week_normalized = None
current_week_normalized = None

v_chunk_size = 100000

########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################
logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
LOG_LEVEL = common.G_log_level

########################################################################################################################
# 컬럼상수
########################################################################################################################
# -*- coding: utf-8 -*-
"""
Constant definitions for PYForecastB2BLockAndRolling V3
=======================================================• DataFrame name constants  – prefix  STR_
• Column    name constants  – prefix  COL_
• Colour-band tokens        – prefix  COLOR_
"""

# ─────────────────────────────────────────────
# ⬥ 원본 Input DataFrame 키
# ─────────────────────────────────────────────
STR_DF_IN_BO_FCST               = 'df_in_BO_FCST'
STR_DF_IN_TOTAL_BOD_LT          = 'df_In_Total_BOD_LT'
STR_DF_IN_MAX_PW                = 'df_In_MAX_PartialWeek'
STR_DF_IN_TIME_PW               = 'df_in_Time_Partial_Week'
STR_DF_IN_SALES_DOMAIN_DIM      = 'df_in_Sales_Domain_Dimension'
STR_DF_IN_MST_RTS_EOS           = 'df_in_MST_RTS_EOS'
STR_DF_IN_SALES_PRODUCT_ASN     = 'df_in_Sales_Product_ASN'

# ─────────────────────────────────────────────
# ⬥ 파생 / 중간 DataFrame 키
# ─────────────────────────────────────────────
STR_DF_FN_RTS_EOS               = 'df_fn_rts_eos'
STR_DF_FN_RTS_EOS_PW            = 'df_fn_rts_eos_pw'
STR_DF_FN_SALES_ASN_WEEK        = 'df_fn_sales_product_asn_item_week'
STR_DF_FN_BO_FCST_ASN           = 'df_fn_bo_fcst_asn'
STR_DF_FN_TOTAL_BOD_LT          = 'df_fn_total_bod_lt'
STR_DF_FN_SHIPTO_DIM            = 'df_fn_shipto_dim'

# ─────────────────────────────────────────────
# ⬥ 최종 Output
# ─────────────────────────────────────────────
STR_DF_OUT_DEMAND               = 'out_Demand'

# ─────────────────────────────────────────────
#  컬럼 이름 (Cube attributes / measures)
# ─────────────────────────────────────────────
COL_VERSION                     = 'Version.[Version Name]'
COL_SHIP_TO                     = 'Sales Domain.[Ship To]'
COL_ITEM                        = 'Item.[Item]'
COL_LOC                         = 'Location.[Location]'

COL_STD1                        = 'Sales Domain.[Sales Std1]'
COL_STD2                        = 'Sales Domain.[Sales Std2]'
COL_STD3                        = 'Sales Domain.[Sales Std3]'
COL_STD4                        = 'Sales Domain.[Sales Std4]'
COL_STD5                        = 'Sales Domain.[Sales Std5]'
COL_STD6                        = 'Sales Domain.[Sales Std6]'

COL_VIRTUAL_BO_ID               = 'DP Virtual BO ID.[Virtual BO ID]'
COL_BO_ID                       = 'DP BO ID.[BO ID]'

COL_TIME_PW                     = 'Time.[Partial Week]'
COL_BO_FCST                     = 'BO FCST'
COL_BO_FCST_LOCK                = 'BO FCST.Lock'
COL_BO_FCST_COLOR_COND          = 'BO FCST Color Condition'
COL_BO_TOTAL_BOD_LT             = 'BO Total BOD LT'

# ───────── RTS / EOS 원본 필드 ─────────
RTS_STATUS                      = 'RTS_STATUS'
RTS_INIT_DATE                   = 'RTS_INIT_DATE'
RTS_DEV_DATE                    = 'RTS_DEV_DATE'
RTS_COM_DATE                    = 'RTS_COM_DATE'

EOS_STATUS                      = 'EOS_STATUS'
EOS_INIT_DATE                   = 'EOS_INIT_DATE'
EOS_CHG_DATE                    = 'EOS_CHG_DATE'
EOS_COM_DATE                    = 'EOS_COM_DATE'

# ───────── Step01-2 Helper 주차 필드 ─────
RTS_PARTIAL_WEEK                = 'RTS_PARTIAL_WEEK'
EOS_PARTIAL_WEEK                = 'EOS_PARTIAL_WEEK'

RTS_WEEK                        = 'RTS_WEEK'
RTS_WEEK_MINUS_1                = 'RTS_WEEK_MINUS_1'
RTS_WEEK_PLUS_3                 = 'RTS_WEEK_PLUS_3'
MAX_RTS_CURRENTWEEK             = 'MAX_RTS_CURRENTWEEK'

EOS_WEEK                        = 'EOS_WEEK'
EOS_WEEK_MINUS_1                = 'EOS_WEEK_MINUS_1'
EOS_WEEK_MINUS_4                = 'EOS_WEEK_MINUS_4'

RTS_INITIAL_WEEK                = 'RTS_INITIAL_WEEK'
EOS_INITIAL_WEEK                = 'EOS_INITIAL_WEEK'
MIN_EOSINI_MAXWEEK              = 'MIN_EOSINI_MAXWEEK'
MIN_EOS_MAXWEEK                 = 'MIN_EOS_MAXWEEK'

# ───────── Step01-4 Helper ──────────────
CURRENT_ROW_WEEK                = 'CURRENT_ROW_WEEK'   # numeric form of PW

# ─────────────────────────────────────────────
#  Colour palette (Lock/Condition bands)
# ─────────────────────────────────────────────
COLOR_WHITE      = '14_WHITE'
COLOR_DARKBLUE   = '15_DARKBLUE'
COLOR_LIGHTBLUE  = '10_LIGHTBLUE'
COLOR_LIGHTRED   = '11_LIGHTRED'
COLOR_DARKRED    = '16_DARKRED'
COLOR_GRAY       = '19_GRAY'
COLOR_DGRAY_RED  = '18_DGRAY_RED'
# COLOR_DGRAY_RED = '18_DGRAY_RED'          # 신규 색상


################################################################################################################
# Start of Util Functions
################################################################################################################
def fn_log_dataframe(df_p_source: pd.DataFrame, str_p_source_name: str) -> None:
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
        logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1)
        if is_local and not df_p_source.empty and flag_csv:
            # 로컬 Debugging 시 csv 파일 출력
            df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)
    else:
        # 최종 Output 테이블인 경우에는 무조건 로그 출력
        if is_output:
            logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1)
            if is_local and not df_p_source.empty:
                # 로컬 Debugging 시 csv 파일 출력
                df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)

def parse_args():
    # Extract arguments from sys.argv
    args = {}
    for arg in sys.argv[1:]:
        if ':' in arg:
            key, value = arg.split(':', 1)  # Split only on the first ':'
            args[key.strip()] = value.strip()
        else:
            print(f"Warning: Argument '{arg}' does not contain a ':' separator.")
    return args

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


def gfn_is_partial_week(week_str: str) -> bool:
    """
    Check if the input week string represents a partial week by determining if the dates span across different months.

    :param week_str: The week string in the format 'YYYYWW'
    :return: True if it's a partial week, False otherwise.
    """
    year = int(week_str[:4])
    week = int(week_str[4:])
    # Calculate the start date of the week using a workaround
    jan_4 = datetime.datetime(year, 1, 4)
    start_date = jan_4 + datetime.timedelta(days=(week - 1) * 7 - jan_4.weekday())
    # Get all dates in the week
    dates_in_week = [start_date + datetime.timedelta(days=i) for i in range(7)]
    # Get the set of months from these dates
    months = {date.month for date in dates_in_week}
    # If there is more than one unique month, it's a partial week
    return len(months) > 1



def gfn_get_partial_week_days(week_str: str) -> dict:
    """
    Get the number of days for each part of a partial week ('A' and 'B').

    :param week_str: The week string in the format 'YYYYWW'
    :return: A dictionary with the number of days for 'A' and 'B'
    """
    year = int(week_str[:4])
    week = int(week_str[4:])
    jan_4 = datetime.datetime(year, 1, 4)
    start_date = jan_4 + datetime.timedelta(days=(week - 1) * 7 - jan_4.weekday())
    dates_in_week = [start_date + datetime.timedelta(days=i) for i in range(7)]
    
    days_count = {'A': 0, 'B': 0}
    first_month = dates_in_week[0].month
    
    for date in dates_in_week:
        if date.month == first_month:
            days_count['A'] += 1
        else:
            days_count['B'] += 1
    
    return days_count


def set_input_output_folder(is_local, args):
    global str_input_dir, str_output_dir
    
    if is_local:
        if args.get('input_folder_name') is not None:
            str_input_dir = f"Input/{args.get('input_folder_name')}"
        if args.get('output_folder_name') is not None:
            str_output_dir = f"Output/{args.get('output_folder_name')}"
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y%m%d_%H_%M")
            str_output_dir = f"{str_output_dir}_{formatted_time}"
        # Ensure the input and output directories exist
        os.makedirs(str_input_dir, exist_ok=True)
        os.makedirs(str_output_dir, exist_ok=True)


def normalize_week(week_str):
    """Convert a week string with potential suffixes to an integer for comparison."""
    # Remove any non-digit characters (e.g., 'A' or 'B') and convert to integer
    try:

        return ''.join(filter(str.isdigit, week_str))
    except Exception as e:
        logger.Note(p_note=f"week_str: {week_str}", p_log_level=LOG_LEVEL.error())


@_decoration_
def fn_output_formatter(df_p_source: pd.DataFrame, str_p_out_version: str) -> pd.DataFrame:
    """
    최종 Output 형태로 정리
    :param df_p_source: 주차별로 가공하여 group by 후 sum을 구한 in_Demand
    :param str_p_out_version: Param_OUT_VERSION
    :return: DataFrame
    """
    # 함수명
    str_my_name = inspect.stack()[0][3]
    # Return 변수
    df_return = pd.DataFrame()

    # 입력 파라미터가 비어 있는 경우 비어 있는 DataFrame을 리턴
    if df_p_source.empty:
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(df_p_source)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    # 입력 파라미터(str_p_out_version)가 비어 있는 경우 경고 메시지를 출력 후 빈 데이터 프레임 리턴
    if str_p_out_version is None or str_p_out_version.strip() == '':
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(str_p_out_version)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    df_return = df_p_source.copy(deep=False)
    df_return[COL_VERSION] = str_p_out_version

    columns_to_return = [
        COL_VERSION,
        COL_ITEM,
        COL_SHIP_TO,
        COL_LOC,
        COL_VIRTUAL_BO_ID,
        COL_BO_ID,
        COL_TIME_PW,
        COL_BO_FCST,
        COL_BO_FCST_LOCK,
        COL_BO_FCST_COLOR_COND
    ]

    df_return = df_return[columns_to_return]

    return df_return

# ────────────────────────────────────────────────────────────────
# Step 99 : 최종 Output 정리 (경량 버전)
# ────────────────────────────────────────────────────────────────
@_decoration_
def _fn_output_formatter(
        df_src         : pd.DataFrame,   # 이전 단계 결과 (in-place X)
        out_version    : str             # 예: 'CWV_DP'
    ) -> pd.DataFrame:
    """
    • Version 컬럼만 추가한 뒤 필요한 9개 컬럼 순서로 Slice
    • 깊은 복사(deep=True) → 제거
    • 빈 DF 또는 버전 문자열 문제는 같은 로직 유지
    """
    # ── 0) 유효성 체크 ─────────────────────────────────────────
    if df_src.empty:
        logger.Note('[fn_output_formatter] 입력 DF가 비어 있습니다.', LOG_LEVEL.warning())
        return pd.DataFrame() 
        if not out_version or not out_version.strip():
            logger.Note('[fn_output_formatter] 버전 문자열이 없습니다.', LOG_LEVEL.warning())
        return pd.DataFrame()

    # ── 1) 얕은 복사(view) 후 Version 컬럼만 추가 ─────────────
    df_out = df_src.copy(deep=False)
    #  ➜ inplace 로 하면 df_src 도 변하므로 새 객체에 할당
    df_out = df_out.assign(**{COL_VERSION: out_version})

    # ── 2) 필요한 컬럼만 Slice (재배치 겸 마무리) ──────────────
    return df_out.loc[:, [
        COL_VERSION,
        COL_ITEM,
        COL_SHIP_TO,
        COL_LOC,
        COL_VIRTUAL_BO_ID,
        COL_BO_ID,
        COL_TIME_PW,
        COL_BO_FCST,
        COL_BO_FCST_LOCK
    ]]

@_decoration_
def fn_set_header() -> pd.DataFrame:
    """
    MediumWeight로 실행 시 발생할 수 있는 Live Server에서의 오류를 방지하기 위해 Header만 있는 Output 테이블을 만든다.
    :return: DataFrame
        """
    df_return = pd.DataFrame()

    # out_Demand
    df_return = pd.DataFrame(
        {
            COL_VERSION         : [],
            COL_ITEM            : [],
            COL_SHIP_TO         : [],
            COL_LOC             : [],
            COL_VIRTUAL_BO_ID   : [],
            COL_BO_ID           : [],
            COL_TIME_PW         : [],
            COL_BO_FCST         : [],
            COL_BO_FCST_LOCK    : [],
            COL_BO_FCST_COLOR_COND  : []

        }
    )

    return df_return

def fn_convert_type(df: pd.DataFrame, startWith: str, type):
    for column in df.columns:
        if column.startswith(startWith):
            df[column] = df[column].astype(type)

def fn_convert_type_equal(df: pd.DataFrame, column: str, type):
    df[column] = df[column].astype(type)
            

@_decoration_
def fn_process_in_df_ms_back():
    if is_local: 
    # 로컬인 경우 Output 폴더를 정리한다.
        for file in os.scandir(str_output_dir):
            os.remove(file.path)

        # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
        file_pattern = f"{os.getcwd()}/{str_input_dir}/*.csv" 
        csv_files = glob.glob(file_pattern)

        file_to_df_mapping = {
            "df_in_BO_FCST.csv"                 : STR_DF_IN_BO_FCST      ,
            "df_in_Total_BOD_LT.csv"            : STR_DF_IN_TOTAL_BOD_LT,
            "df_In_MAX_PartialWeek.csv"         : STR_DF_IN_MAX_PW  
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
                if file_name.startswith(keyword.split('.')[0]):
                    input_dataframes[frame_name] = df
                    mapped = True
                    break

    else:
        # o9 에서 
        input_dataframes[STR_DF_IN_BO_FCST]             = df_in_BO_FCST
        input_dataframes[STR_DF_IN_TOTAL_BOD_LT]        = df_in_Total_BOD_LT
        input_dataframes[STR_DF_IN_MAX_PW]              = df_In_MAX_PartialWeek

    fn_convert_type(input_dataframes[STR_DF_IN_BO_FCST], 'Sales Domain', str)
    input_dataframes[STR_DF_IN_BO_FCST][COL_BO_FCST].fillna(0, inplace=True)
    fn_convert_type_equal(input_dataframes[STR_DF_IN_BO_FCST], COL_BO_FCST, 'int32')
    fn_convert_type_equal(input_dataframes[STR_DF_IN_BO_FCST], COL_BO_FCST_LOCK, bool)
    input_dataframes[STR_DF_IN_TOTAL_BOD_LT][COL_BO_TOTAL_BOD_LT].fillna(0, inplace=True)
    fn_convert_type_equal(input_dataframes[STR_DF_IN_TOTAL_BOD_LT], COL_BO_TOTAL_BOD_LT, 'int32')

# ────────────────────────────────────────────────────────────────
# Util Function : read inputs (+ memory down-cast) – V3 + TimePW
# ----------------------------------------------------------------
@_decoration_
def fn_process_in_df_mst() -> None:
    """
    7 개 입력 테이블 로드 & 메모리 최적화
    --------------------------------------------------------------
    • 'Sales Domain*' 열  → str → category
    • Time.[Partial Week] → ordered category
    • object             → unordered category
    • int64/float64      → down-cast
    • BO_FCST / BOD_LT   → 결측치·dtype 보정
    """
    # 1) ───── 파일 매핑 + 로드 ─────────────────────────────────
    file_to_df = {
        # 기존 6개
        "df_in_BO_FCST.csv"               : STR_DF_IN_BO_FCST,
        "df_in_Total_BOD_LT.csv"          : STR_DF_IN_TOTAL_BOD_LT,
        "df_In_MAX_PartialWeek.csv"       : STR_DF_IN_MAX_PW,
        "df_in_MST_RTS_EOS.csv"           : STR_DF_IN_MST_RTS_EOS,
        "df_in_Sales_Product_ASN.csv"     : STR_DF_IN_SALES_PRODUCT_ASN,
        "df_in_Sales_Domain_Dimension.csv": STR_DF_IN_SALES_DOMAIN_DIM,
        # 신규 7번째
        "df_in_Time_Partial_Week.csv"     : STR_DF_IN_TIME_PW,
    }

    if is_local:
        # 폴더 정리
        for f in os.scandir(str_output_dir):
            os.remove(f.path)

        # CSV 읽기
        for p in glob.glob(f"{os.getcwd()}/{str_input_dir}/*.csv"):
            df_tmp = _read_csv_fallback(p)
            key    = next((k for k in file_to_df
                           if os.path.basename(p).startswith(k.split('.')[0])), None)
            if key:
                input_dataframes[file_to_df[key]] = df_tmp.copy(deep=False)
    else:
        # 서버(o9) 모드 — 이미 메모리에 존재
        input_dataframes.update({
            STR_DF_IN_BO_FCST           : df_in_BO_FCST,
            STR_DF_IN_TOTAL_BOD_LT      : df_in_Total_BOD_LT,
            STR_DF_IN_MAX_PW            : df_In_MAX_PartialWeek,
            STR_DF_IN_MST_RTS_EOS       : df_in_MST_RTS_EOS,
            STR_DF_IN_SALES_PRODUCT_ASN : df_in_Sales_Product_ASN,
            STR_DF_IN_SALES_DOMAIN_DIM  : df_in_Sales_Domain_Dimension,
            STR_DF_IN_TIME_PW           : df_in_Time_Partial_Week,
        })

    # 2) ───── 공통 타입 표준화 & 메모리 다운캐스트 ─────────────
    ORDERED_CAT_COLS   = {COL_TIME_PW}          # 'Time.[Partial Week]'
    SALES_DOMAIN_START = "Sales Domain"         # 프리픽스 매칭

    for df in input_dataframes.values():

        # 2-A) 'Sales Domain*' 열 → str → category
        for col in [c for c in df.columns if c.startswith(SALES_DOMAIN_START)]:
            df[col] = df[col].astype(str).astype("category")

        # 2-B) object 열 → category (ordered 예외)
        for col in df.select_dtypes(include="object"):
            if col in ORDERED_CAT_COLS:
                df[col] = pd.Categorical(df[col],
                                          categories=sorted(df[col].unique()),
                                          ordered=True)
            else:
                df[col] = df[col].astype("category")

        # 2-C) 수치형 down-cast
        ints   = df.select_dtypes(include="int64").columns
        floats = df.select_dtypes(include="float64").columns
        if len(ints):
            df[ints]   = df[ints].apply(pd.to_numeric, downcast="integer")
        if len(floats):
            df[floats] = df[floats].apply(pd.to_numeric, downcast="float")

    # 3) ───── BO_FCST / BOD_LT 결측치 + dtype 보정 ────────────
    if STR_DF_IN_BO_FCST in input_dataframes:
        df_bo = input_dataframes[STR_DF_IN_BO_FCST]
        df_bo[COL_BO_FCST].fillna(0, inplace=True)
        fn_convert_type_equal(df_bo, COL_BO_FCST, "int32")
        fn_convert_type_equal(df_bo, COL_BO_FCST_LOCK, bool)

    if STR_DF_IN_TOTAL_BOD_LT in input_dataframes:
        df_lt = input_dataframes[STR_DF_IN_TOTAL_BOD_LT]
        df_lt[COL_BO_TOTAL_BOD_LT].fillna(0, inplace=True)
        fn_convert_type_equal(df_lt, COL_BO_TOTAL_BOD_LT, "int32")
    
    if is_local:
        input_path = f'{str_output_dir}/input'
        os.makedirs(input_path,exist_ok=True)
        for input_file in input_dataframes:
            input_dataframes[input_file].to_csv(input_path + "/"+input_file+".csv", encoding="UTF8", index=False)

# ── Helper : 인코딩 자동 판별 CSV 로더 ─────────────────────────
def _read_csv_fallback(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Encoding not supported: {path}")

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
        '%m/%d/%Y'    # ④ 04/16/2025
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

def gfn_is_date_parsing(p_date_str: str) -> bool:
    try:
        return bool(date_parser.parse(p_date_str))
    except ValueError:
        return False
def gfn_is_date_matching(p_date_str: Union[str, datetime.datetime], p_format) -> bool:
    try:
        if isinstance(p_date_str, str):
            return bool(datetime.datetime.strptime(p_date_str, p_format))
        else:
            return bool(datetime.datetime.strftime(p_date_str, p_format))
    except ValueError:
        return False

def gfn_to_date(p_str_datetype: str, p_format: str, p_week_day=1, p_day_delta=0) -> datetime:
    """
    string -> datetime
    ex) gfn_to_date('2024-W01', '%Y-W%W') -> datetime(2024-01-01 00:00:00)
        gfn_to_date('2024-M01', '%Y-M%m') -> datetime(2024-01-01 00:00:00)
        gfn_to_date('20240101', '%Y%m%d') -> datetime(2024-01-01 00:00:00)
        gfn_to_date('2024.01.01', '%Y.%m.%d') -> datetime(2024-01-01 00:00:00)
        gfn_to_date('2024.01.01 03:09:09', '%Y.%m.%d %H:%M:%S') -> datetime(2024-01-01 03:09:09)

    :param p_str_datetype:
    :param p_format:
    :param p_week_day:
    :param p_day_delta:
    :return:
    """
    result = None
    str_msg = ''
    if r'%W' in p_format and gfn_is_date_matching(p_str_datetype, p_format):
        year, week = None, None
        all_char = re.sub(r'[^0-9]', '', p_str_datetype)
        if len(all_char) == 6:
            year = int(all_char[:4])
            week = int(all_char[4:])
        elif len(all_char) == 5:
            year = int(all_char[:4])
            week = int(all_char[-1:])
        else:
            str_msg = f'''Error : week format string not matching
            common function : gfn_to_date -> gfn_is_date_matching
            param    : ({p_str_datetype}, {p_format}, {p_week_day})
            '''
            raise Exception(str_msg)

        # result = datetime.datetime.fromisocalendar(year, week, p_week_day)
        result = datetime.datetime.strptime(f"{year:04d}{week:02d}{p_week_day:d}", "%G%V%u")  # .date()

    elif r'%m' in p_format and r'%d' not in p_format and gfn_is_date_matching(p_str_datetype, p_format):
        year, month = None, None
        all_char = re.sub(r'[^0-9]', '', p_str_datetype)
        if len(all_char) == 6:
            year = all_char[:4]
            month = all_char[4:]
        elif len(all_char) == 5:
            year = all_char[:4]
            month = all_char[-1:]
        else:
            str_msg = f'''Error : month format string not matching
            common function : gfn_to_date -> gfn_is_date_matching
            param    : ({p_str_datetype})
            '''
            raise Exception(str_msg)
        str_datetype = '-'.join([year, month, '01'])

        result = datetime.datetime.strptime(str_datetype, '%Y-%m-%d')
    else:
        if gfn_is_date_parsing(p_str_datetype):
            if gfn_is_date_matching(p_date_str=p_str_datetype, p_format=p_format):
                result = datetime.datetime.strptime(p_str_datetype, p_format)
            else:
                str_msg = f'''Error : format string not matching
                common function : gfn_to_date -> gfn_is_date_matching
                param    : ({p_str_datetype}, {p_format}, {p_week_day})
                '''
                raise Exception(str_msg)
        else:
            str_msg = f'''Error : format string not parsing
            common function : gfn_to_date -> gfn_is_date_parsing
            param    : ({p_str_datetype}, {p_format}, {p_week_day})
            '''
            raise Exception(str_msg)

    if p_day_delta == 0:
        return result
    else:
        return result + datetime.timedelta(days=p_day_delta)

def sanitize_pw(x: object) -> str:
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
            - patialweek 로 변환
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
            # return dt_obj.strftime('%Y/%m/%d')              # zero-padding 포함
            # dt = datetime.datetime.strptime(x_str, fmt)
            # return common.gfn_get_partial_week(dt_obj, True)
            year = str(dt_obj.isocalendar()[0])
            week = str(dt_obj.isocalendar()[1]).zfill(2)
            month = dt_obj.strftime('%m')

            # iyyyiw
            iyyyiw = common.G_STR_WEEK_FORMAT.replace('%Y', year).replace('%W', week)

            # start date
            monday_mm = gfn_to_date(p_str_datetype=iyyyiw, p_format='%Y%W', p_week_day=1).strftime('%m')
            # end date
            sunday_mm = gfn_to_date(p_str_datetype=iyyyiw, p_format='%Y%W', p_week_day=7).strftime('%m')
            p_bool_FI_week = True
            if (month == monday_mm) & (month == sunday_mm):
                if p_bool_FI_week:
                    result = iyyyiw + 'A'
                else:
                    result = iyyyiw
            elif (month == monday_mm) & (month != sunday_mm):
                result = iyyyiw + 'A'
            elif (month != monday_mm) & (month == sunday_mm):
                result = iyyyiw + 'B'
            else:
                result = iyyyiw
            return result

    except Exception:
        pass       # fall-through → 실패 처리
    return ''       # 파싱 실패

# 벡터라이즈 버전
v_sanitize_pw = np.vectorize(sanitize_pw, otypes=[object])
# ════════════════════════════════════════════════════════════
# Ultra-fast column-wise sanitiser (caches by unique value)
# ════════════════════════════════════════════════════════════
def _sanitize_pw_series(s: pd.Series) -> pd.Series:
    """
    Vectorised 'YYYYWWA/B' converter for an *entire* Series.    
    • Work in three cheap string ops (split/replace/len check)
    • Parse each unique token once, memoise in dict -> 100× fewer strptime calls
    """
    # 0) Null-guard & early exit
    s = s.astype('string')
    mask_null = s.isna() | (s.str.strip() == '')
    if mask_null.all():
        return pd.Series('', index=s.index, dtype='string')

    # 1) pre-clean (keep date token only, normalise delimiter)
    s_clean = (
        s.str.split(r'\s+|T', n=1, expand=True)  # drop time / 'T' part
          .get(0)
          .str.replace('-', '/', regex=False)
          .fillna('')
    )

    # 2) build lookup dict by unique token
    unique_vals = s_clean.unique()
    mapping = {}

    for tok in unique_vals:
        if not tok:                      # null/blank
            mapping[tok] = ''
            continue
        parts = tok.split('/')
        try:
            if len(parts) == 3:
                # YYYY/MM/DD  or  M/D/YYYY
                if len(parts[0]) == 4:           # YYYY first
                    y, m, d = parts
                else:                            # M/D/YYYY
                    m, d, y = parts
                dt_obj = datetime.datetime(int(y), int(m), int(d))
                year  = str(dt_obj.isocalendar()[0])
                week  = str(dt_obj.isocalendar()[1]).zfill(2)
                iyyyiw = common.G_STR_WEEK_FORMAT.replace('%Y', year)\
                                                .replace('%W', week)

                # decide A/B suffix
                month     = dt_obj.strftime('%m')
                monday_mm = gfn_to_date(iyyyiw, '%Y%W', 1).strftime('%m')
                sunday_mm = gfn_to_date(iyyyiw, '%Y%W', 7).strftime('%m')
                if month == monday_mm == sunday_mm:
                    suffix = 'A'        # whole week in one month
                elif month == monday_mm:
                    suffix = 'A'
                elif month == sunday_mm:
                    suffix = 'B'
                else:
                    suffix = ''
                mapping[tok] = iyyyiw + suffix
            else:
                mapping[tok] = ''
        except Exception:                # any issue -> blank
            mapping[tok] = ''

    # 3) vectorised map + null re-insert
    out = s_clean.map(mapping)
    out[mask_null] = ''
    return out.astype('category')

# ← 기존 프로젝트 공용 모듈# ════════════════════════════════════════════════════════════
# 1) _pw_key_arr : Partial-Week Series → int32 YYYYWW 배열
#    - '202506A' , '202506B'  →  202506
#    - null / '' / NaN       →        -1
# ════════════════════════════════════════════════════════════
def _pw_key_arr(s: pd.Series) -> np.ndarray:
    """
    Parameters
    ----------
    s : pandas.Series
        Partial-Week 문자열(또는 category) 컬럼

    Returns
    -------
    np.ndarray[int32]
        YYYYWW 정수 배열, 유효하지 않은 값은 -1
    """
    s2 = s.astype('string')                               # StringDtype → C-backend
    yww = (
        pd.to_numeric(
            s2.str.slice(0, 6),        # 'YYYYWW'
            errors='coerce'
        )
        .fillna(-1)
        .astype('int32')
    )
    return yww.to_numpy()


# ════════════════════════════════════════════════════════════
# 2) _build_week_shift_dict : 주차 ±n 캐시 테이블 생성
#    { shift : { YYYYWW_int : YYYYWW_shifted_int } }
# ════════════════════════════════════════════════════════════
def _build_week_shift_dict(
        unique_pws: np.ndarray,
        shifts: list[int]
    ) -> dict[int, dict[int, int]]:
    """
    unique_pws : np.ndarray[str]
        중복 제거된 Partial-Week 문자열
    shifts : list[int]
        주차 오프셋 (예: [-1, +3, -4])

    Returns
    -------
    dict
        예) { -1 : { 202506 : 202505, ... },
              +3 : { 202506 : 202509, ... } }
    """
    wk_dict = {sh: {} for sh in shifts}

    for pw in unique_pws:
        pw_str = str(pw)
        if len(pw_str) < 6:
            continue
        base_key = int(pw_str[:6])       # YYYYWW → int

        for sh in shifts:
            tgt_pw   = common.gfn_add_week(pw_str[:6], sh)      # 'YYYYWW'
            tgt_key  = int(tgt_pw) if tgt_pw else -1
            wk_dict[sh][base_key] = tgt_key

    return wk_dict


# ════════════════════════════════════════════════════════════
# 3) _safe_lookup : dict 기반 벡터화 조회기 생성
#    • mapping 에 key 없으면 기본값 -1
# ════════════════════════════════════════════════════════════
def _safe_lookup(mapping: dict[int, int]):
    """
    Returns
    -------
    np.vectorize object
        배열 단위로 호출 가능 :  safe_fn(np_array_of_keys)
    """
    return np.vectorize(lambda k: mapping.get(k, -1), otypes=[int])

################################################################################################################
# End of Util Functions
################################################################################################################
################################################################################################################
# Start of Step Functions
################################################################################################################

# ────────────────────────────────────────────────────────────────
# Step 01-1  : Load + Clean RTS/EOS master
#   • Source : input_dataframes[STR_DF_IN_MST_RTS_EOS]
#   • Drop   : Version, RTS_ISVALID
#   • Save   : output_dataframes[STR_DF_FN_RTS_EOS]
# ────────────────────────────────────────────────────────────────
@_decoration_
def fn_step01_1_load_rts_eos() -> pd.DataFrame:
    """
    Step 01-1
    ----------
    1. 원본 `df_in_MST_RTS_EOS` 를 로드
    2. 불필요한 열 제거
        • Version.[Version Name]
        • RTS_ISVALID
    3. 메모리 절약을 위해 `copy(deep=False)` 뷰 반환
    4. `output_dataframes[STR_DF_FN_RTS_EOS]` 에 등록
    """
    # 1) 원본 로드 ───────────────────────────────────────────────
    df_src = input_dataframes[STR_DF_IN_MST_RTS_EOS]    
    # 2) 컬럼 제거 ──────────────────────────────────────────────
    cols_to_drop = [
        c for c in df_src.columns
        if c.startswith('Version') or c.upper() == 'RTS_ISVALID'
    ]
    df_rts_eos = df_src.drop(columns=cols_to_drop, errors='ignore').copy(deep=False)

    # 3) 결과 등록 & 로깅 ────────────────────────────────────────
    output_dataframes[STR_DF_FN_RTS_EOS] = df_rts_eos
    fn_log_dataframe(df_rts_eos, f'step01_1_{STR_DF_FN_RTS_EOS}')

    # 4) (선택) DuckDB 검사 – 로컬 전용, 서버(o9)에서는 주석 유지
    """
    import duckdb
    duckdb.register('rts_eos', df_rts_eos)
    print(duckdb.sql('SELECT COUNT(*) AS rows, '
                     'COUNT(DISTINCT "COL_ITEM") AS items '
                     'FROM rts_eos').to_df())
    """

    return df_rts_eos

@_decoration_
def fn_step01_2_enrich_rts_eos(
        df_rts_eos: pd.DataFrame,
        current_week_normalized: str,   # 'YYYYWW'
        max_week_normalized: str        # 'YYYYWW'
    ) -> pd.DataFrame:
    """
    Step 01-2  (대용량 대응 · in-place)
    ---------------------------------
    1. RTS_/EOS_ 6개 날짜 → Partial-Week 문자열로 변환
    2. RTS_PARTIAL_WEEK / EOS_PARTIAL_WEEK 계산
    3. Helper 주차 필드 12개를 한 번에 생성
       ▸ ±1·+3·-4 주차 : common.gfn_add_week()
       ▸ int(YYYYWW)   : _pw_key_arr()
        └ RTS_INITIAL_WEEK  ← NEW
        └ EOS_INITIAL_WEEK  ← NEW

    4. 중간 배열/딕셔너리 즉시 삭제 → 메모리 최소화
    """
    # import gc    
    # ── 1)  날짜 → Partial-Week 문자열 ──────────────────────────
    date_cols = [
        RTS_INIT_DATE, RTS_DEV_DATE, RTS_COM_DATE,
        EOS_INIT_DATE, EOS_CHG_DATE, EOS_COM_DATE
    ]
    for c in date_cols:
        df_rts_eos[c] = _sanitize_pw_series(df_rts_eos[c])

    # ── 2)  대표 Partial-Week 결정  (Numpy vectorised) ─────────
    rts_pw = np.where(
        df_rts_eos[RTS_STATUS] == 'COM',
        df_rts_eos[RTS_COM_DATE],
        np.where(
            (df_rts_eos[RTS_DEV_DATE].notna()) & (df_rts_eos[RTS_DEV_DATE] != '') ,
            df_rts_eos[RTS_DEV_DATE],
            df_rts_eos[RTS_INIT_DATE]
        )
    )
    df_rts_eos[RTS_PARTIAL_WEEK] = pd.Categorical(rts_pw)

    eos_pw = np.where(
        df_rts_eos[EOS_STATUS] == 'COM',
        np.where(
            (df_rts_eos[EOS_COM_DATE].notna()) & (df_rts_eos[EOS_COM_DATE] != ''),
            df_rts_eos[EOS_COM_DATE],
            np.where(
                (df_rts_eos[EOS_CHG_DATE].notna()) & (df_rts_eos[EOS_CHG_DATE] != ''),
                df_rts_eos[EOS_CHG_DATE],
                df_rts_eos[EOS_INIT_DATE]
            ),
        ),
        np.where(
            (df_rts_eos[EOS_CHG_DATE].notna()) & (df_rts_eos[EOS_CHG_DATE] != ''),
            df_rts_eos[EOS_CHG_DATE],
            df_rts_eos[EOS_INIT_DATE]
        )
    )
    df_rts_eos[EOS_PARTIAL_WEEK] = pd.Categorical(eos_pw)

    # ── 3)  Helper-주차 컬럼 대량 생성 ─────────────────────────
    rts_key = _pw_key_arr(df_rts_eos[RTS_PARTIAL_WEEK])
    eos_key = _pw_key_arr(df_rts_eos[EOS_PARTIAL_WEEK])
    
    # NEW: RTS / EOS 최초 주차
    df_rts_eos[RTS_INITIAL_WEEK] = _pw_key_arr(df_rts_eos[RTS_INIT_DATE])
    df_rts_eos[EOS_INITIAL_WEEK] = _pw_key_arr(df_rts_eos[EOS_INIT_DATE])


    # 3-A  기본 컬럼
    df_rts_eos[RTS_WEEK] = rts_key
    df_rts_eos[EOS_WEEK] = eos_key

    # 3-B  ± 주차 계산을 위한 캐시 딕셔너리
    uniq_pw = np.unique(np.concatenate([rts_pw, eos_pw]))
    shift_map = _build_week_shift_dict(uniq_pw, shifts=[-1, +3, -4])
    get_m1 = _safe_lookup(shift_map[-1])
    get_p3 = _safe_lookup(shift_map[+3])
    get_m4 = _safe_lookup(shift_map[-4])

    df_rts_eos[RTS_WEEK_MINUS_1]  = get_m1(rts_key)
    df_rts_eos[RTS_WEEK_PLUS_3]   = get_p3(rts_key)
    df_rts_eos[EOS_WEEK_MINUS_1]  = get_m1(eos_key)
    df_rts_eos[EOS_WEEK_MINUS_4]  = get_m4(eos_key)

    # 3-C  Max / Min 파생
    cw_int = int(current_week_normalized)
    mw_int = int(max_week_normalized)
    df_rts_eos[MAX_RTS_CURRENTWEEK] = np.maximum(rts_key, cw_int)
    df_rts_eos[MIN_EOSINI_MAXWEEK]  = np.minimum(
        _pw_key_arr(df_rts_eos[EOS_INITIAL_WEEK]) if EOS_INITIAL_WEEK in df_rts_eos
                                                   else eos_key,
        mw_int
    )
    df_rts_eos[MIN_EOS_MAXWEEK]  = np.minimum(
        _pw_key_arr(df_rts_eos[EOS_WEEK]) if EOS_WEEK in df_rts_eos
                                                   else eos_key,
        mw_int
    )

    # ── 4) 메모리 정리 ────────────────────────────────────────
    del rts_pw, eos_pw, rts_key, eos_key, uniq_pw, shift_map
    gc.collect()

    # DuckDB 검증(로컬) – 주석
    """
    import duckdb, random
    duckdb.register('rts', df_rts_eos[[COL_ITEM, COL_SHIP_TO, RTS_WEEK, EOS_WEEK]])
    sample = duckdb.sql('SELECT * FROM rts USING SAMPLE 5').to_df()
    fn_log_dataframe(sample, 'dbg_step01_2_sample')
    """

    return df_rts_eos


@_decoration_
def fn_step01_4_build_rts_eos_pw(
        df_rts_eos: pd.DataFrame,
        df_time_pw: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Step 01-4
    ----------
    1. df_rts_eos (Item×Ship-To) 와 df_time_pw (Partial-Week 마스터)
       Cartesian 확장 → df_fn_rts_eos_pw 생성
    2. 컬럼 추가
       • COL_TIME_PW            : 주차 문자열
       • CURRENT_ROW_WEEK       : int32 YYYYWW  (A/B 구분無)
       • COL_BO_FCST_LOCK       : True
       • COL_BO_FCST_COLOR_COND : '19_GRAY'
    3. 메모리 절약
       • 뷰 기반 `copy(deep=False)`
       • category dtype 사용
       • 중간 NumPy 배열 즉시 삭제
    """    
    # import gc

    # ── 1) 기본 차수 검사 ───────────────────────────────────────
    n_rts = len(df_rts_eos)
    n_pw  = len(df_time_pw)
    if n_rts == 0 or n_pw == 0:
        raise ValueError("RTS/EOS 또는 Time_Partial_Week 테이블이 비어 있습니다.")

    # ── 2) NumPy 교차 확장 ──────────────────────────────────────
    # ① RTS 행 인덱스 반복, ② 주차 배열 타일
    rts_idx = np.repeat(np.arange(n_rts, dtype='int32'), n_pw)
    pw_vals = np.tile(df_time_pw[COL_TIME_PW].to_numpy(dtype='object'), n_rts)

    df_grid = (
        df_rts_eos[[COL_SHIP_TO,COL_ITEM]].iloc[rts_idx].reset_index(drop=True, inplace=False)
            .assign(**{
                COL_TIME_PW           : pw_vals,
                COL_BO_FCST_LOCK      : True,
                COL_BO_FCST_COLOR_COND: COLOR_GRAY
            })
    )

    # ── 3) 보조 컬럼 · dtype 최적화 ──────────────────────────────
    df_grid[COL_BO_FCST_LOCK]       = df_grid[COL_BO_FCST_LOCK].astype('bool')
    df_grid[COL_BO_FCST_COLOR_COND] = pd.Categorical(
        df_grid[COL_BO_FCST_COLOR_COND],
        categories=[COLOR_GRAY]
    )

    # CURRENT_ROW_WEEK : ‘YYYYWW’ → int32
    df_grid[CURRENT_ROW_WEEK] = _pw_key_arr(df_grid[COL_TIME_PW]).astype('int32')

    # 주차 문자열도 category 로 축소
    df_grid[COL_TIME_PW] = df_grid[COL_TIME_PW].astype('category')

    # ── 4) 중간 배열 메모리 해제 ────────────────────────────────
    del rts_idx, pw_vals; gc.collect()

    # ── 5) output_dataframes 에 최초 1회 등록 ──────────────────
    output_dataframes[STR_DF_FN_RTS_EOS_PW] = df_grid
    fn_log_dataframe(df_grid, f'step01_4_{STR_DF_FN_RTS_EOS_PW}')

    return df_grid    

@_decoration_
def fn_step01_5_set_lock_values(
        df_grid: pd.DataFrame,          # df_fn_rts_eos_pw (in-place 수정)
        df_rts:  pd.DataFrame,          # df_fn_rts_eos  (helper 컬럼 보유)
        current_week_normalized: str,   # 'YYYYWW'
        max_week_normalized:     str    # 'YYYYWW'
    ) -> pd.DataFrame:
    """
    Step 01-5  – 5-Band Lock / Colour (Spec V3)
    ------------------------------------------
    WHITE      : row ≥ CW  AND  max_RTS ≤ row ≤ min(EOS_ini, maxWeek)
    DARKBLUE   : row ≥ CW  AND  RTS_init ≤ row ≤ RTS-1
    LIGHTBLUE  : row ≥ CW  AND  RTS      ≤ row ≤ RTS+3
    LIGHTRED   : row ≥ CW  AND  EOS-4    ≤ row ≤ EOS-1
    DARKRED    : row ≥ CW  AND  EOS      ≤ row ≤ maxWeek
    """    
    # import gc

    # 검증을 위해
    # df_grid_copy = df_grid.copy(deep=True)

    # ────────────────────────────────
    # 1) Helper 배열 구성 – index lookup
    # ────────────────────────────────
    key_cols = [COL_SHIP_TO, COL_ITEM]          # 두 DF 공통 PK
    df_lkp   = (df_rts
                .set_index(key_cols, drop=False)
                .loc[:, [
                    MAX_RTS_CURRENTWEEK, MIN_EOS_MAXWEEK,
                    RTS_INITIAL_WEEK, RTS_WEEK, RTS_WEEK_MINUS_1, RTS_WEEK_PLUS_3,
                    EOS_INITIAL_WEEK, EOS_WEEK, EOS_WEEK_MINUS_1, EOS_WEEK_MINUS_4
                ]])
    # positions: df_grid → df_lkp 매핑
    pos = df_lkp.index.get_indexer(
        df_grid[key_cols].itertuples(index=False, name=None)
    )
    valid = pos >= 0

    # NumPy 배열로 빼내기 (int32)
    to_arr = lambda col: pd.to_numeric(df_lkp[col], errors='coerce').astype('int32').to_numpy()

    max_rts          = np.full(len(df_grid), -1, dtype='int32')
    min_eos_maxweek  = np.full_like(max_rts, -1)

    rts_init         = np.full_like(max_rts, -1)
    rts_week         = np.full_like(max_rts, -1)
    rts_m1           = np.full_like(max_rts, -1)
    rts_p3           = np.full_like(max_rts, -1)

    eos_week         = np.full_like(max_rts, -1)
    eos_m1           = np.full_like(max_rts, -1)
    eos_m4           = np.full_like(max_rts, -1)

    # 채우기
    cols_src = [
        (max_rts,         MAX_RTS_CURRENTWEEK),
        (min_eos_maxweek, MIN_EOS_MAXWEEK),
        (rts_init,        RTS_INITIAL_WEEK),
        (rts_week,        RTS_WEEK),
        (rts_m1,          RTS_WEEK_MINUS_1),
        (rts_p3,          RTS_WEEK_PLUS_3),
        (eos_week,        EOS_WEEK),
        (eos_m1,          EOS_WEEK_MINUS_1),
        (eos_m4,          EOS_WEEK_MINUS_4),
    ]
    for arr, col in cols_src:
        src = to_arr(col)
        arr[valid] = src[pos[valid]]
    logger.debug("fn_step01_5_set_lock_values. 1) Helper 배열 구성 – index lookup")
    # ────────────────────────────────
    # 2) 행 주차, 기준 주차 int 변환
    # ────────────────────────────────
    row_week = df_grid[CURRENT_ROW_WEEK].to_numpy(dtype='int32')
    cw_int   = int(current_week_normalized)
    mw_int   = int(max_week_normalized)

    # ────────────────────────────────
    # 3) 색상 / Lock 초기화
    # ────────────────────────────────
    colour = np.full(len(df_grid), COLOR_GRAY, dtype=object)  # 초기값 Gray (추후 WHITE 재할당)
    lock   = np.ones(len(df_grid), dtype='bool')              # 초기값 True
    logger.debug("fn_step01_5_set_lock_values. 3) 색상 / Lock 초기화")
    # ────────────────────────────────
    # 4) White  (먼저 적용)
    # ────────────────────────────────
    cond_white = (
        (row_week >= cw_int) &
        (row_week >= max_rts) &
        (row_week <= min_eos_maxweek)
    )
    colour[cond_white] = COLOR_WHITE
    lock[cond_white]   = False
    logger.debug("fn_step01_5_set_lock_values. 4) White  (먼저 적용)")

    """
    # validation White
        import duckdb
        duckdb.register('df_rts',df_rts)
        duckdb.register('df_grid',df_grid)
        
        query = f'''
            select 
                {current_week_normalized} as "cur week",
                a."{RTS_WEEK}" ,                
                a."{MAX_RTS_CURRENTWEEK}" ,
                a."{EOS_WEEK}" ,
                {max_week_normalized} as "max week",
                a."{MIN_EOS_MAXWEEK}" 
            from df_rts a
            where a."{COL_SHIP_TO}" == '300116'
        '''
        duckdb.sql(query).show()
        '''
        # 300116 White
        ┌──────────┬──────────┬─────────────────────┬──────────┬──────────┬─────────────────┐
        │ cur week │ RTS_WEEK │ MAX_RTS_CURRENTWEEK │ EOS_WEEK │ max week │ MIN_EOS_MAXWEEK │
        │  int32   │  int32   │        int32        │  int32   │  int32   │      int32      │
        ├──────────┼──────────┼─────────────────────┼──────────┼──────────┼─────────────────┤
        │   202506 │   202249 │              202506 │   202704 │   202606 │          202606 │
        └──────────┴──────────┴─────────────────────┴──────────┴──────────┴─────────────────┘

        300116 의 경우  해당되도록 수정했음.
        (row_week >= cw_int)            202506 >= 202249  : True
        (row_week >= max_rts) &         202506 >= 202506  : True
        (row_week <= min_eos_maxweek)   202506 <  202606  : True
        '''
    """
    # ────────────────────────────────
    # 5) 나머지 4개의 밴드
    # ────────────────────────────────
    # mask = ~cond_white  # 이미 White 면 제외
    mask = (row_week >= cw_int)

    # DarkBlue
    cond = mask & (row_week >= rts_init) & (row_week <= rts_m1)
    colour[cond] = COLOR_DARKBLUE
    lock[cond]   = True
    # mask &= ~cond
    
    """
    # validation DarkBlue
        import duckdb
        duckdb.register('df_rts',df_rts)
        duckdb.register('df_grid',df_grid)
        
        query = f'''
            select 
                {current_week_normalized} as "cur week",
                a."{RTS_WEEK}" ,
                a."{RTS_INITIAL_WEEK}" ,
                a."{RTS_WEEK_MINUS_1}" 
            from df_rts a
            where a."{COL_SHIP_TO}" == '300116'
        '''
        duckdb.sql(query).show()
        '''
        # 300116 DarkBlue
        ┌──────────┬──────────┬──────────────────┬──────────────────┐
        │ cur week │ RTS_WEEK │ RTS_INITIAL_WEEK │ RTS_WEEK_MINUS_1 │
        │  int32   │  int32   │      int32       │      int32       │
        ├──────────┼──────────┼──────────────────┼──────────────────┤
        │   202506 │   202249 │           202202 │           202248 │
        └──────────┴──────────┴──────────────────┴──────────────────┘

        300116 의 경우  해당하지 않는다.
        (row_week >= rts_init)          202506 >= 202202  : 대부분 True
        (row_week <= rts_m1)            202506 <= 202248  : False
        '''
    # """

    # LightBlue
    cond = mask & (row_week >= rts_week) & (row_week <= rts_p3)
    colour[cond] = COLOR_LIGHTBLUE
    lock[cond]   = False
    # mask &= ~cond
    """
    # validation LightBlue
        import duckdb
        duckdb.register('df_rts',df_rts)
        duckdb.register('df_grid',df_grid)
        
        query = f'''
            select 
                {current_week_normalized} as "cur week",
                a."{RTS_WEEK}" ,
                a."{RTS_WEEK_PLUS_3}" 
            from df_rts a
            where a."{COL_SHIP_TO}" == '300116'
        '''
        duckdb.sql(query).show()
        '''
        # 300116 LightBlue
        ┌──────────┬──────────┬─────────────────┐
        │ cur week │ RTS_WEEK │ RTS_WEEK_PLUS_3 │
        │  int32   │  int32   │      int32      │
        ├──────────┼──────────┼─────────────────┤
        │   202506 │   202249 │          202252 │
        └──────────┴──────────┴─────────────────┘

        300116 의 경우  해당하지 않는다.
        (row_week >= rts_week)          202506 >= 202249  : 대부분 True
        (row_week <= rts_p3)            202506 <= 202252  : False

        # 300136 의 경우는 데이타가 있음.
        ┌──────────┬──────────┬─────────────────┐
        │ cur week │ RTS_WEEK │ RTS_WEEK_PLUS_3 │
        │  int32   │  int32   │      int32      │
        ├──────────┼──────────┼─────────────────┤
        │   202506 │   202507 │          202510 │
        └──────────┴──────────┴─────────────────┘
        (row_week >= rts_week)          202506 >= 202507  : True
        (row_week <= rts_p3)            202506 <= 202510  : True
        '''
    """

    # LightRed
    cond = mask & (row_week >= eos_m4) & (row_week <= eos_m1)
    colour[cond] = COLOR_LIGHTRED
    lock[cond]   = False
    # mask &= ~cond
    """
    # validation LightRed
        import duckdb
        duckdb.register('df_rts',df_rts)
        duckdb.register('df_grid',df_grid)
        
        query = f'''
            select 
                {current_week_normalized} as "cur week",
                a."{EOS_WEEK}" ,
                a."{EOS_WEEK_MINUS_4}" ,
                a."{EOS_WEEK_MINUS_1}" 
            from df_rts a
            where a."{COL_SHIP_TO}" == '300116'
        '''
        duckdb.sql(query).show()
        
        '''
        # 300116 LightRed
        ┌──────────┬──────────┬──────────────────┬──────────────────┐
        │ cur week │ EOS_WEEK │ EOS_WEEK_MINUS_4 │ EOS_WEEK_MINUS_1 │
        │  int32   │  int32   │      int32       │      int32       │
        ├──────────┼──────────┼──────────────────┼──────────────────┤
        │   202506 │   202704 │           202653 │           202703 │
        └──────────┴──────────┴──────────────────┴──────────────────┘

        300116 의 경우  해당하지 않는다.
        (row_week >= eos_m4)            202506 >= 202653  : False
        (row_week <= eos_m1)            202506 <= 202703  : True

        '''
    """

    # DarkRed
    cond = mask & (row_week >= eos_week) & (row_week <= mw_int)
    colour[cond] = COLOR_DARKRED
    lock[cond]   = True
    
    """
    # validation DarkRed
        import duckdb
        duckdb.register('df_rts',df_rts)
        duckdb.register('df_grid',df_grid)
        
        query = f'''
            select 
                {current_week_normalized} as "cur week",
                a."{EOS_WEEK}" ,
                {max_week_normalized} as max_week
            from df_rts a
            where a."{COL_SHIP_TO}" == '300116'
        '''
        duckdb.sql(query).show()
        
        '''
        # 300116 DarkRed
        ┌──────────┬──────────┬──────────┐
        │ cur week │ EOS_WEEK │ max_week │
        │  int32   │  int32   │  int32   │
        ├──────────┼──────────┼──────────┤
        │   202506 │   202704 │   202606 │
        └──────────┴──────────┴──────────┘

        300116 의 경우  해당하지 않는다.
        (row_week >= eos_week)          202506 >= 202704  : False
        (row_week <= mw_int)            202506 <= 202606  : True

        '''
    # """

    # ────────────────────────────────
    # 6) DataFrame 반영
    # ────────────────────────────────
    df_grid[COL_BO_FCST_LOCK] = lock
    df_grid[COL_BO_FCST_COLOR_COND] = pd.Categorical(
        colour,
        categories=[COLOR_WHITE, COLOR_DARKBLUE, COLOR_LIGHTBLUE, COLOR_LIGHTRED, COLOR_DARKRED, COLOR_GRAY,COLOR_DGRAY_RED]
    )
    logger.debug("fn_step01_5_set_lock_values. 5) 나머지 4개의 밴드")
    # ────────────────────────────────
    # 7) 중간 배열 메모리 해제
    # ────────────────────────────────
    del (max_rts, min_eos_maxweek, rts_init, rts_week, rts_m1, rts_p3,
         eos_week, eos_m1, eos_m4, colour, lock, cols_src, df_lkp, pos)
    gc.collect()

    """
        ═══════════════════════════════════════════════════════════════
        DUCKDB VALIDATION  –  Step 01-4 vs Step 01-5
        • 확인 항목
            ① 색상/Lock 업데이트 행 수
            ② 밴드별(Colour) 건수
            ③ 특정 Ship-To·Item 의 week-by-week 변화
        ═══════════════════════════════════════════════════════════════
        import duckdb
        # 1) DuckDB 에 데이터프레임 등록
        duckdb.register('pre',  df_grid_copy)        # Step 01-4 결과 (Gray+True)
        duckdb.register('post', df_fn_rts_eos_pw)    # Step 01-5 후 (색상/Lock 적용)

        # 2) 전체 변경 건수
        q_diff = duckdb.sql(f'''
            SELECT  COUNT(*) AS total_rows,
                    SUM( pre.{COL_BO_FCST_COLOR_COND} <> post.{COL_BO_FCST_COLOR_COND} ) AS colour_changed,
                    SUM( pre.{COL_BO_FCST_LOCK} <> post.{COL_BO_FCST_LOCK} )             AS lock_changed
            FROM    pre
            JOIN    post
                ON pre.{COL_SHIP_TO} = post.{COL_SHIP_TO}
                AND pre.{COL_ITEM}    = post.{COL_ITEM}
                AND pre.{COL_TIME_PW} = post.{COL_TIME_PW}
        ''').to_df()
        fn_log_dataframe(q_diff, 'dbg_step01_5_diff_count')

        # 3) 밴드별 건수
        q_band = duckdb.sql(f'''
            SELECT  {COL_BO_FCST_COLOR_COND}  AS colour,
                    COUNT(*)                 AS rows
            FROM    post
            GROUP  BY 1
            ORDER  BY 2 DESC
        ''').to_df()
        fn_log_dataframe(q_band, 'dbg_step01_5_band_dist')

        # 4) 특정 Ship-To · Item 추적 (예: shipto = '300114', item = 'RF65DG90BDSGTL')
        shipto = '300114'
        item   = 'RF65DG90BDSGTL'
        q_trace = duckdb.sql(f'''
            SELECT  p.{COL_TIME_PW}                 AS week,
                    p.{COL_BO_FCST_COLOR_COND}      AS colour_pre,
                    n.{COL_BO_FCST_COLOR_COND}      AS colour_post,
                    p.{COL_BO_FCST_LOCK}            AS lock_pre,
                    n.{COL_BO_FCST_LOCK}            AS lock_post
            FROM    pre p
            JOIN    post n
                ON n.{COL_SHIP_TO} = p.{COL_SHIP_TO}
                AND n.{COL_ITEM}    = p.{COL_ITEM}
                AND n.{COL_TIME_PW} = p.{COL_TIME_PW}
            WHERE   p.{COL_SHIP_TO} = '{shipto}'
            AND   p.{COL_ITEM}    = '{item}'
            ORDER  BY week
        ''').to_df()
        fn_log_dataframe(q_trace, 'dbg_step01_5_trace_shipto_item')
        """

    return df_grid

@_decoration_
def fn_step01_6_load_shipto_dimension(
        df_dim: pd.DataFrame          # df_in_Sales_Domain_Dimension
    ) -> pd.DataFrame:
    """
    Step 01-6  (patched)
    --------------------
    Ship-To(LV5/6 코드) 기준 계층 LUT 생성.    레벨 매핑 규칙
    ─────────────
        Std1 == ShipTo  → LV2
        Std2 == ShipTo  → LV3
        Std3 == ShipTo  → LV4
        Std4 == ShipTo  → LV5
        Std5 == ShipTo  → LV6
        Std6 == ShipTo  → LV7
        (어느 것도 매칭되지 않으면 LV2)

    반환 컬럼
    ────────
        • COL_SHIP_TO
        • COL_STD1 … COL_STD6
        • LV_CODE  (int8)

    메모리
    ──────
        • category 캐스팅
        • 중간 bool / int 배열 삭제 후 gc.collect()
    """
    # import gc

    # 1) 필요한 열만 선택 & category 캐스팅 ------------------------
    use_cols = [
        COL_SHIP_TO, COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6
    ]
    df_lut = (
        df_dim[use_cols]
        .copy(deep=False)
        .astype({c: 'category' for c in use_cols})
    )

    # 2) 레벨 계산 (벡터라이즈) ------------------------------------
    # 초기값 LV2
    lv_code = np.full(len(df_lut), 2, dtype='int8')

    # 각 Std 열이 Ship-To 와 동일하면 해당 레벨로 갱신
    std_cols_levels = [
        (COL_STD6, 7),
        (COL_STD5, 6),
        (COL_STD4, 5),
        (COL_STD3, 4),
        (COL_STD2, 3),
        (COL_STD1, 2)
    ]
    ship_vals = df_lut[COL_SHIP_TO].to_numpy()

    for col, lvl in std_cols_levels:
        match = df_lut[col].to_numpy() == ship_vals
        lv_code[match] = lvl

    df_lut['LV_CODE'] = lv_code

    # 3) 결과 등록 & 로그 -----------------------------------------
    # STR_DF_FN_SHIPTO_DIM = 'df_fn_shipto_dim'
    # output_dataframes[STR_DF_FN_SHIPTO_DIM] = df_lut
    # fn_log_dataframe(df_lut, STR_DF_FN_SHIPTO_DIM)

    # 4) 불필요 객체 메모리 해제 -----------------------------------
    del ship_vals, lv_code, match, df_dim
    gc.collect()

    return df_lut


@_decoration_
def fn_step01_7_prepare_asn_week(
        df_asn: pd.DataFrame,               # df_in_Sales_Product_ASN
        df_time_pw: pd.DataFrame,           # df_in_Time_Partial_Week
        current_week_normalized: str        # 'YYYYWW'
    ) -> pd.DataFrame:
    """
    Step 01-7  (patched)
    --------------------
    • ASN × Partial-Week Cartesian 확장
    • 규칙
        · row_week <  currentWeek → LOCK=True , COLOR=19_GRAY
        · row_week >= currentWeek → LOCK=False, COLOR=14_WHITE
    • CURRENT_ROW_WEEK(int) 컬럼 포함
    """    
    # import gc

    # 1) 교차 확장 ---------------------------------------------------------
    n_asn = len(df_asn)
    n_pw  = len(df_time_pw)
    asn_idx = np.repeat(np.arange(n_asn, dtype='int32'), n_pw)
    pw_vals = np.tile(df_time_pw[COL_TIME_PW].to_numpy(dtype='object'), n_asn)

    base_cols = [COL_SHIP_TO, COL_ITEM, COL_LOC]
    df_grid = (
        df_asn[base_cols].iloc[asn_idx]
              .reset_index(drop=True, inplace=False)
              .assign(**{COL_TIME_PW: pw_vals})
    )

    # 2) CURRENT_ROW_WEEK(int32) 계산 -------------------------------------
    row_week = _pw_key_arr(df_grid[COL_TIME_PW]).astype('int32')
    df_grid[CURRENT_ROW_WEEK] = row_week

    # 3) Lock / Color 초기화 (스펙 반영) -----------------------------------
    cur_int = int(current_week_normalized)
    past_mask = row_week < cur_int

    df_grid[COL_BO_FCST_LOCK] = past_mask          # bool
    df_grid[COL_BO_FCST_COLOR_COND] = pd.Categorical(
        np.where(past_mask, COLOR_GRAY, COLOR_WHITE),
        categories=[COLOR_WHITE, COLOR_GRAY]
    )

    # dtype 최적화
    df_grid[COL_TIME_PW] = df_grid[COL_TIME_PW].astype('category')

    # 4) 결과 등록 ---------------------------------------------------------
    # output_dataframes[STR_DF_FN_SALES_ASN_WEEK] = df_grid
    # fn_log_dataframe(df_grid, f'step01_7_{STR_DF_FN_SALES_ASN_WEEK}')

    # 5) 중간 배열 해제 ----------------------------------------------------
    del asn_idx, pw_vals, row_week, past_mask, df_asn
    gc.collect()

    return df_grid

@_decoration_
def fn_step01_8_apply_rts_colour_to_asn(
        df_asn_week : pd.DataFrame,      # df_fn_sales_product_asn_item_week  (in-place)
        df_rts_pw   : pd.DataFrame,      # df_fn_rts_eos_pw  (LV2·LV3 혼재)
        df_ship_dim : pd.DataFrame       # df_fn_shipto_dim  (LV_CODE, Std1~6)
    ) -> pd.DataFrame:
    """
    LV2(Std1) → LV3(Std2) 순서로 Lock / Color 복사
    ----------------------------------------------
    1) Ship-To → Std1·Std2 매핑
    2) RTS 측 Ship-To 레벨(LV2/LV3) 분리 lookup
    3) LV2 먼저 복사  →  LV3 덮어쓰기
    """    
    # import gc

    # ── 0) 컬러 컬럼을 object 로 임시 변환 (새 색상 허용) ──────────
    df_asn_week[COL_BO_FCST_COLOR_COND] = df_asn_week[COL_BO_FCST_COLOR_COND].astype(object)

    # ── 1) Ship-To 매핑 dict 생성 --------------------------------
    dim        = df_ship_dim.set_index(COL_SHIP_TO)
    map_std1   = dim[COL_STD1].to_dict()        # Std1 → LV2 코드
    map_std2   = dim[COL_STD2].to_dict()        # Std2 → LV3 코드
    lv_code    = dim['LV_CODE'].to_dict()       # Ship-To → 레벨번호

    df_asn_week['_LV2'] = df_asn_week[COL_SHIP_TO].map(map_std1)
    df_asn_week['_LV3'] = df_asn_week[COL_SHIP_TO].map(map_std2)

    # ── 2) RTS 쪽 Ship-To 레벨 부여 & lookup 테이블 제작 ---------
    df_rts_pw['_LV_CODE'] = df_rts_pw[COL_SHIP_TO].map(lv_code).astype('int8')

    key_cols = ['_SHIP_STD', COL_ITEM, COL_TIME_PW]

    def _make_lookup(src: pd.DataFrame, lv: int):
        return (src.query('_LV_CODE == @lv')
                   .rename(columns={COL_SHIP_TO: '_SHIP_STD'})
                   .set_index(key_cols, drop=False)[
                       [COL_BO_FCST_LOCK, COL_BO_FCST_COLOR_COND]])

    lkp_lv2 = _make_lookup(df_rts_pw, 2)
    lkp_lv3 = _make_lookup(df_rts_pw, 3)

    # ── 3) 복사 헬퍼 (DataFrame.loc 사용) ------------------------
    def _copy(df_dst: pd.DataFrame, ship_series: pd.Series, lkp: pd.DataFrame):
        idx_tuples = list(zip(ship_series, df_dst[COL_ITEM], df_dst[COL_TIME_PW]))
        pos        = lkp.index.get_indexer(idx_tuples)
        valid_idx  = np.flatnonzero(pos >= 0)           # 행 인덱스 (ASN)
        if len(valid_idx):
            src_pos = pos[valid_idx]                    # RTS 행 위치
            df_dst.loc[valid_idx, COL_BO_FCST_LOCK]        = lkp[COL_BO_FCST_LOCK].to_numpy()[src_pos]
            df_dst.loc[valid_idx, COL_BO_FCST_COLOR_COND]  = lkp[COL_BO_FCST_COLOR_COND].to_numpy()[src_pos]
            test = 1
            """
            import duckdb
            # numpy 배열을 DataFrame 으로 래핑

            duckdb.register('lkp', lkp)
            duckdb.register('df_dst', df_dst.loc[valid_idx])
            duckdb.sql(f'SELECT "{COL_BO_FCST_COLOR_COND}", COUNT(*) AS rows from lkp GROUP BY 1 ORDER BY 2 DESC').show()
            duckdb.sql(f'SELECT "{COL_BO_FCST_COLOR_COND}", COUNT(*) AS rows from df_dst GROUP BY 1 ORDER BY 2 DESC').show()
            """

    # ── 4) LV2 적용 후 LV3 덮어쓰기 ------------------------------
    _copy(df_asn_week, df_asn_week['_LV2'], lkp_lv2)
    _copy(df_asn_week, df_asn_week['_LV3'], lkp_lv3)

    # ── 5) 컬러 컬럼을 카테고리로 재캐스팅 ------------------------
    df_asn_week[COL_BO_FCST_COLOR_COND] = pd.Categorical(
        df_asn_week[COL_BO_FCST_COLOR_COND],
        categories=[COLOR_WHITE, COLOR_GRAY,
                    COLOR_DARKBLUE, COLOR_LIGHTBLUE,
                    COLOR_LIGHTRED, COLOR_DARKRED]
    )

    # ── 6) 임시 열 및 중간 객체 정리 -----------------------------
    df_asn_week.drop(columns=['_LV2', '_LV3'], inplace=True)
    del map_std1, map_std2, lkp_lv2, lkp_lv3; gc.collect()

    """
    ══════════════════════════════════════════════════════════════
    DUCKDB VALIDATION – 색상 반영 확인
    ══════════════════════════════════════════════════════════════
    import duckdb, os
    duckdb.register('asn', df_asn_week)
    duckdb.register('rts', df_rts_pw)
    duckdb.register('dim', df_ship_dim)

    # 1) RTS 컬러 종류
    duckdb.sql(f'SELECT DISTINCT "{COL_BO_FCST_COLOR_COND}" FROM rts').show()

    # 2) ASN 컬러 종류
    duckdb.sql(f'SELECT DISTINCT "{COL_BO_FCST_COLOR_COND}" FROM asn').show()

    # 3) LV2·LV3 매칭 성공 행 로그 CSV
    for lv, std_col, name in [(2, COL_STD1, 'lv2_hits'),
                            (3, COL_STD2, 'lv3_hits')]:
        q = duckdb.sql(f'''
            SELECT a.*
            FROM   asn a
            JOIN   rts r
            ON   r."{COL_SHIP_TO}" IN (
                    SELECT "{std_col}" FROM dim
                    WHERE "{COL_SHIP_TO}" = a."{COL_SHIP_TO}"
                )
            AND   r."{COL_ITEM}"     = a."{COL_ITEM}"
            AND   r."{COL_TIME_PW}"  = a."{COL_TIME_PW}"
            AND   r."_LV_CODE"       = {lv}
        ''').to_df()
        fn_log_dataframe(q, f'chk_step01_8_{name}')
        # q.to_csv(os.path.join(str_output_dir, f'{name}.csv'), index=False)
    """
    return df_asn_week

# ────────────────────────────────────────────────────────────────
# Step 02-1 : ASN-Week → BO-Colour Grid (초기값 세팅)
# ────────────────────────────────────────────────────────────────
@_decoration_
def fn_step02_1_build_bo_color_grid(
        df_asn_week: pd.DataFrame      # Step 01-8 결과
    ) -> pd.DataFrame:
    """
    출력 컬럼
    --------
    Ship-To · Item · Loc · Partial-Week  +  
    BO FCST(0 초기) / BO FCST.Lock / BO FCST Color Condition  
    DP Virtual BO ID = '-' , DP BO ID = '-'
    """    
    # ── 1) 뷰 복사 ────────────────────────────────────────────
    # df_grid = df_asn_week.copy(deep=False)
    # 메모리의 낭비방지를 위해서 copy를 안함. df_grid 와 df_asn_week 은 같은 instance 임.
    df_grid = df_asn_week

    # ── 2) BO 전용 컬럼 추가 ─────────────────────────────────
    df_grid[COL_BO_FCST]       = 0
    df_grid[COL_VIRTUAL_BO_ID] = '-'          # category 로 변환은 Step 02-2 이후
    df_grid[COL_BO_ID]         = '-'

    # ── 3) 결과 등록 & 로그 ──────────────────────────────────
    # output_dataframes[STR_DF_FN_BO_FCST_ASN] = df_grid
    # fn_log_dataframe(df_grid, STR_DF_FN_BO_FCST_ASN)

    return df_grid

# ────────────────────────────────────────────────────────────────
# Step 02-2 : BO FCST 소스 반영 (Lock / Colour 덮어쓰기)
# ────────────────────────────────────────────────────────────────
@_decoration_
def fn_step02_2_apply_grid_to_bo(
        df_grid   : pd.DataFrame,      # Step 02-1 인스턴스 (in-place)
        df_bo_src : pd.DataFrame       # df_in_BO_FCST  (컬러 없음)
    ) -> pd.DataFrame:
    """
    규칙
    ----
    • PK = (Ship-To, Item, Loc, Partial-Week)
    • df_bo_src 행이 있으면
        └ BO FCST, BO FCST.Lock 덮어쓰기
        └ Color = df_grid 컬러 그대로 (소스 컬러 Null)
    • df_grid 행이 없고 df_bo_src 만 있는 PK → 신규 행 추가
        └ Color = '' (나중 단계에서 색칠)
    • Virtual/BO ID 가 ‘-’ 가 아닌 소스 행은 건드리지 않음
    """
    # import gc    
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW]

    # 1) 소스 BO 테이블 전처리 ------------------------------------------
    df_bo = (
        df_bo_src
        .copy(deep=False)
        .assign(**{
            COL_BO_FCST_COLOR_COND: ''                     # 컬러 Null 보강
        })
    )

    # 2) 두 테이블 outer merge -----------------------------------------
    merged = (
        df_grid.merge(
            df_bo,
            on=key_cols,
            how='outer',
            suffixes=('_grid', '_bo'),
            indicator=True
        )
    )

    # 3) 가공 로직 ------------------------------------------------------
    # (a) BO FCST
    merged[COL_BO_FCST] = merged[f'{COL_BO_FCST}_bo'].fillna(
                              merged[f'{COL_BO_FCST}_grid']
                          ).astype('int32', copy=False)

    # (b) Lock (Virtual/BO ID = '-' 조건). 
    cond_overwrite = (
        (merged[COL_VIRTUAL_BO_ID + '_bo'] == '-') &
        (merged[COL_BO_ID          + '_bo'] == '-')
    )
    merged[COL_BO_FCST_LOCK] = np.where(
        cond_overwrite & merged[f'{COL_BO_FCST_LOCK}_grid'].notna(),    # grid가 우선되어야 한다.
        merged[f'{COL_BO_FCST_LOCK}_grid'],                             # grid가 우선되어야 한다.
        merged[f'{COL_BO_FCST_LOCK}_bo']
    ).astype('bool', copy=False)

    # (c) Colour : 소스 컬러가 없으므로 grid 컬러 유지
    # merged[COL_BO_FCST_COLOR_COND] = merged[f'{COL_BO_FCST_COLOR_COND}_grid']
    # 아래는 기본 컬러를 적용할 때
    merged[COL_BO_FCST_COLOR_COND] = np.where(
        merged[f'{COL_BO_FCST_COLOR_COND}_grid'].notna(),
        merged[f'{COL_BO_FCST_COLOR_COND}_grid'],
        COLOR_GRAY
    )

    # (d) Virtual / BO ID : 소스 값이 우선
    for c in (COL_VIRTUAL_BO_ID, COL_BO_ID):
        merged[c] = merged[f'{c}_bo'].fillna(merged[f'{c}_grid']).astype('string')

    # 4) 컬럼 정렬 & dtype ---------------------------------------------
    keep = key_cols + [
        COL_VIRTUAL_BO_ID, COL_BO_ID,
        COL_BO_FCST, COL_BO_FCST_LOCK, COL_BO_FCST_COLOR_COND
    ]
    df_out = merged[keep]

    df_out[COL_BO_FCST_COLOR_COND] = pd.Categorical(
        df_out[COL_BO_FCST_COLOR_COND],
        categories=[COLOR_WHITE, COLOR_GRAY,
                    COLOR_DARKBLUE, COLOR_LIGHTBLUE,
                    COLOR_LIGHTRED, COLOR_DARKRED]
    )

    # 5) 결과 등록 & 메모리 정리 ----------------------------------------
    # output_dataframes[STR_DF_FN_BO_FCST_ASN] = df_out
    # fn_log_dataframe(df_out, STR_DF_FN_BO_FCST_ASN)

    del merged, df_bo
    gc.collect()

    """
    import duckdb
    duckdb.register('bo', df_out)
    # 덮어쓰기 잘 됐는지 샘플 확인
    duckdb.sql(f'''
    SELECT "{COL_BO_FCST_LOCK}", "{COL_BO_FCST_COLOR_COND}", COUNT(*) AS rows
    FROM   bo
    GROUP  BY 1,2 ORDER BY 3 DESC
    ''').show()
    """
    return df_out
# ────────────────────────────────────────────────────────────────
# Step 03-1 : Total BOD LT → Week 변환  (SPEC V3)
# ────────────────────────────────────────────────────────────────
@_decoration_
def fn_step03_1_prepare_total_bod_lt(
        df_total_src      : pd.DataFrame,    # df_in_Total_BOD_LT
        cur_week_norm_str : str              # e.g. '202506'
    ) -> pd.DataFrame:
    """
    Item·Loc 단위 BOD LT(일수) → ‘YYYYWWA’ 주차로 계산
        week_str = gfn_add_week(cur_week,  (BOD_LT // 7))
    반환 컬럼
        COL_ITEM, COL_LOC, COL_BO_TOTAL_BOD_LT
    """    
    # import gc
    # 1) 필요한 열만 복사 (view)
    use_cols = [COL_ITEM, COL_LOC, COL_BO_TOTAL_BOD_LT]
    df_ret   = df_total_src[use_cols].copy(deep=False)

    # 2) (일수//7) 오프셋 계산  (int32)
    offset_weeks = (df_ret[COL_BO_TOTAL_BOD_LT] // 7).astype('int32').to_numpy()

    # 3) gfn_add_week 벡터화
    add_week_vec = np.vectorize(common.gfn_add_week, otypes=[object])
    new_week = add_week_vec(cur_week_norm_str, offset_weeks)

    # 4) 컬럼 치환
    df_ret[COL_BO_TOTAL_BOD_LT] = pd.Categorical(new_week)

    # 5) 중간 배열 메모리 해제
    del offset_weeks, new_week; gc.collect()
    return df_ret


# ═══════════════════════════════════════════════════════════════
# Step 03-2 : Lock / Colour 재조정  (SPEC V3 완전 반영)
# ═══════════════════════════════════════════════════════════════
@_decoration_
def fn_step03_2_apply_lock_conditions(
        df_grid      : pd.DataFrame,   # BO-Grid (in-place)
        df_bod_map   : pd.DataFrame,   # Step 03-1 결과
        df_rts_eos    : pd.DataFrame,      # Step 01-2 결과  (STD Ship-To 기준 RTS/EOS 주차)
        df_ship_dim   : pd.DataFrame,      # Ship-To Dim (LV_CODE, Std1·Std2)
        cur_week_int : int,            # ex) 202506
        max_week_int : int             # ex) 202606
    ) -> pd.DataFrame:
    """
    cond1 : VBO='-'  & BO_ID='-'      (일반 BO)
    cond2 : VBO!='-' & BO_ID='-'      (Virtual BO)
    cond3 : VBO!='-' & BO_ID!='-'     (Real BO)
    """

    # COLOR_DGRAY_RED = '18_DGRAY_RED'

    # ── 1) BOD_LT 주차 look-up ─────────────────────────────────
    logger.debug("Step 03-2. 1) BOD_LT 주차 look-up")
    bod_idx = (df_bod_map
               .set_index([COL_ITEM, COL_LOC])[COL_BO_TOTAL_BOD_LT])
    bod_pos = bod_idx.index.get_indexer(
        df_grid[[COL_ITEM, COL_LOC]].itertuples(index=False, name=None)
    )
    bod_arr = np.where(
        bod_pos >= 0,
        bod_idx.to_numpy(str)[bod_pos].astype('int32'),
        cur_week_int # changed from max_week_int
    )

    # ── 2) 행별 주차(int) & 기본 마스크 ────────────────────────
    logger.debug("Step 03-2. 2) 행별 주차(int) & 기본 마스크")
    row_week = pd.to_numeric(
        df_grid[COL_TIME_PW].str.replace(r'\D', '', regex=True),
        errors='coerce'
    ).fillna(0).astype('int32').to_numpy()

    cond1 = (df_grid[COL_VIRTUAL_BO_ID] == '-') & (df_grid[COL_BO_ID] == '-')
    cond2 = (df_grid[COL_VIRTUAL_BO_ID] != '-') & (df_grid[COL_BO_ID] == '-')
    cond3 = (df_grid[COL_VIRTUAL_BO_ID] != '-') & (df_grid[COL_BO_ID] != '-')

    # ---------------------------------------------------------
    # ── 3) cond1 : 일반 BO – Lock True, 색상 조건부 18_DGRAY_RED ────────────
    # ---------------------------------------------------------
    logger.debug("Step 03-2. 3) cond1 : 일반 BO – Lock True, 색상 조건부 18_DGRAY_RED")
    # START:  Item·Ship-To 레벨에서 “RTS/EOS 주차 존재 여부” 벡터 준비
    # ① level-5 Ship-To → (Std1, Std2) 매핑 dict
    map_lv2 = df_ship_dim.set_index(COL_SHIP_TO)[COL_STD1].to_dict()
    map_lv3 = df_ship_dim.set_index(COL_SHIP_TO)[COL_STD2].to_dict()    # ② df_rts_eos 를 (Std Ship-To, Item) multi-index 로
    rts_idx = (
        df_rts_eos
        .assign(_SHIP_STD = df_rts_eos[COL_SHIP_TO])        # 이미 Std2 / Std3 수준
        .set_index(['_SHIP_STD', COL_ITEM])
        [['RTS_WEEK', 'EOS_WEEK']]
    )

    # ③ df_grid 의 각 행 → rts_idx 위치 찾기
    ship_std = df_grid[COL_SHIP_TO].map(map_lv3).fillna(
                   df_grid[COL_SHIP_TO].map(map_lv2)         # LV3 매핑 실패 → LV2 시도
               )
    pos      = rts_idx.index.get_indexer(
                   zip(ship_std, df_grid[COL_ITEM])
               )
    valid    = pos >= 0
    # ④ RTS/EOS 정보 존재 여부 → bool 배열
    has_rts_eos = np.zeros(len(df_grid), dtype='bool')
    if valid.any():
        rts_eos_arr = rts_idx.to_numpy(copy=False)[pos[valid]]   # [[RTS_WEEK, EOS_WEEK], …]
        # 각 칼럼(0=RTS_WEEK, 1=EOS_WEEK) 별로 ― (값이 존재하면서 '') 도 아님
        
        # -------------------
        # current conditon. commented
        # ------------------- 
        # col_rts = pd.Series(rts_eos_arr[:, 0], dtype='string')
        # col_eos = pd.Series(rts_eos_arr[:, 1], dtype='string')
        
        # has_rts = col_rts.notna() & (col_rts != '')
        # has_eos = col_eos.notna() & (col_eos != '')

        # # 둘 중 하나라도 존재하면 True
        # has_rts_eos[valid] = has_rts.to_numpy() | has_eos.to_numpy()
        
        # ------------------- 
        # Below is to do. change to range. I have to convert to numpy()?
        # ------------------- 
        # col_rts = pd.Series(rts_eos_arr[:, 0], dtype='int32')
        # col_eos = pd.Series(rts_eos_arr[:, 1], dtype='int32')
        col_rts = pd.to_numeric(rts_eos_arr[:, 0], errors='coerce')   # float64 + NaN
        col_eos = pd.to_numeric(rts_eos_arr[:, 1], errors='coerce')
        # # Current Week ~ RTS ~ LeadTime Week 
        # has_rts = col_rts.notna() & (col_rts >= cur_week_int ) & (col_rts <= bod_arr)
        # # Current Week ~ EOS ~ LeadTimeWeek
        # has_eos = col_eos.notna() & (col_eos >= cur_week_int ) & (col_eos <= bod_arr)

        # # 둘 중 하나라도 존재하면 True
        # # has_rts_eos[valid] = has_rts.to_numpy() | has_eos.to_numpy()
        # has_rts_eos[valid] = (has_rts | has_eos).to_numpy(dtype=bool)
        mask_rts = (~np.isnan(col_rts)) & (col_rts >= cur_week_int) & (col_rts <= bod_arr)
        mask_eos = (~np.isnan(col_eos)) & (col_eos >= cur_week_int) & (col_eos <= bod_arr)
        has_rts_eos[valid] = mask_rts | mask_eos

    # END:  Item·Ship-To 레벨에서 “RTS/EOS 주차 존재 여부” 벡터 준비
    
    m1_base = cond1 & (row_week >= cur_week_int) & (row_week <= bod_arr)
    # 3-A) Lock : 무조건 True
    df_grid.loc[m1_base, COL_BO_FCST_LOCK] = True
    # 3-B) Colour : RTS/EOS 색(화이트·블루·레드 계열)이 없는 행만 대상
    col  = COL_BO_FCST_COLOR_COND
    # (i) 필요하면 카테고리 등록
    if COLOR_DGRAY_RED not in df_grid[col].cat.categories:
        df_grid[col] = df_grid[col].cat.add_categories([COLOR_DGRAY_RED])
    # (ii) 덮어쓸 행 - 기존 색이 GRAY('19_GRAY') 또는 ''(빈 문자열)
    m1_colour = (
        m1_base                                             # 기본 cond1 범위
        & df_grid[col].isin([COLOR_GRAY, COLOR_WHITE,''])   # 현재 색상 GRAY ,WHITE, ''
        & ~has_rts_eos                                      # RTS/EOS 존재하지 않음
    )
    df_grid.loc[m1_colour, col] = COLOR_DGRAY_RED

    # ---------------------------------------------------------
    # ── 4) cond2 : Virtual BO – 최대주차 행 Lock True (없으면 생성) ─
    # ---------------------------------------------------------
    logger.debug("Step 03-2. 4) cond2 : Virtual BO – 최대주차 행 Lock True (없으면 생성)")
    m2 = cond2 & (row_week == max_week_int)
    df_grid.loc[m2, COL_BO_FCST_LOCK] = True

    # ▸ 그룹별 max 주차 존재 여부
    grp_cols = [COL_ITEM, COL_LOC, COL_SHIP_TO, COL_VIRTUAL_BO_ID, COL_BO_ID]
    mask2 = cond2.to_numpy(dtype=bool, copy=False)
    cond2_df = df_grid[mask2]
    # max 주차 여부 벡터
    flag_arr = (row_week[mask2] == max_week_int)
    has_max = (
        cond2_df.assign(flag=flag_arr)               # ← NumPy 배열 사용
                .groupby(grp_cols, sort=False)['flag']
                .any()
                .reset_index(name='has_max')
    )
    need_grp = has_max[~has_max['has_max']]
    
    """
    validation duckdb
    import duckdb
    # cond2_df = cond2_df.assign(flag=flag_arr) 
    cond2_df = cond2_df.assign(row_week=row_week[mask2]) 
    duckdb.register('cond2_df',cond2_df)
    query = f'''
        select 
            "{COL_ITEM}", 
            "{COL_LOC}", 
            "{COL_SHIP_TO}", 
            "{COL_VIRTUAL_BO_ID}", 
            "{COL_BO_ID}",
            max(row_week)
        from cond2_df
        group by 1,2,3,4,5
    '''
    duckdb.sql(query).show()
    ┌────────────────┬─────────────────────┬────────────────────────┬──────────────────────────────────┬──────────────────┬───────────────┐
    │  Item.[Item]   │ Location.[Location] │ Sales Domain.[Ship To] │ DP Virtual BO ID.[Virtual BO ID] │ DP BO ID.[BO ID] │ max(row_week) │
    │    varchar     │       varchar       │        varchar         │             varchar              │     varchar      │     int32     │
    ├────────────────┼─────────────────────┼────────────────────────┼──────────────────────────────────┼──────────────────┼───────────────┤
    │ LH015IEACFS/GO │ S356                │ 5002458                │ VBO_00000001                     │ -                │        202606 │
    │ LH015IEACFS/GO │ S356                │ 5014319                │ VBO_00000004                     │ -                │        202606 │
    │ LH015IEACFS/GO │ S356                │ 5003114                │ VBO_00000002                     │ -                │        202606 │
    │ LH015IEACFS/GO │ S311                │ 5006201                │ VBO_00000003                     │ -                │        202606 │
    └────────────────┴─────────────────────┴────────────────────────┴──────────────────────────────────┴──────────────────┴───────────────┘
    # 모두 max가 있으므로 아래의 로직은 안돌 것이다.
    """

    if not need_grp.empty:
        logger.debug("Step 03-2. 4) cond2 : 없으면 생성")
        latest = (
            cond2_df.sort_values(grp_cols + [COL_TIME_PW],
                                 ascending=[True]*len(grp_cols)+[False])
                    .drop_duplicates(grp_cols, keep='first')
                    .merge(need_grp.drop(columns='has_max'),
                           on=grp_cols, how='inner')
        )
        suffixes = ['A', 'B'] if gfn_is_partial_week(str(max_week_int)) else ['A']
        new_rows = []
        for sfx in suffixes:
            add = latest.copy()
            add[COL_TIME_PW]         = f'{max_week_int}{sfx}'
            add[COL_BO_FCST_LOCK]    = True
            new_rows.append(add)
        df_grid = pd.concat([df_grid, *new_rows], ignore_index=True)

    # ---------------------------------------------------------
    # 5) cond3 : Real BO – 범위 False / 이후 True
    # ---------------------------------------------------------
    logger.debug("Step 03-2. 5) cond3 : Real BO – 범위 False / 이후 True")
    mask3 = cond3.to_numpy(dtype=bool, copy=False)          # (전체 길이) cond3 를 ndarray 로
    cond3_df = df_grid[mask3]                               # cond3 에 해당하는 행만 서브셋# ── Debug 메시지

    if cond3.any():                                         # cond3 행이 하나라도 있으면
        logger.debug("Step 03-2. 5) cond3 : cond3.any()")
        # ① Item|Loc 를 하나의 문자열 key 로 만들어 (cond3 행 수) Series 생성
        key_ser  = (
            df_grid.loc[cond3, [COL_ITEM, COL_LOC]]
                .astype(str)
                .agg('|'.join, axis=1)                   # 예: 'RF18A5101SR/AA|S311WC13'
        )                                                   # length == cond3 행수

        # ② cond3 행마다 가장 이른 주차(min_week) 계산  (groupby 로 중복 제거)
        min_week = (
            pd.Series(row_week[mask3], index=key_ser)       # index=key, value=row_week
            .groupby(level=0)                             # key 별
            .min()                                        # 최소 주차
        )                                                   # length == 고유 key 수

        # ③ cond3 원본 순서대로 “해당 key 의 최소주차” 배열 만들기
        start = np.take(                                    # length == cond3 행수
            min_week.to_numpy(),                            # [min_wk_key1, min_wk_key2 …]
            min_week.index.get_indexer(key_ser)             # cond3 행 → index 위치 매핑
        )

        # ── **BUGFIX** ───────────────────────────────────────
        start_full = np.full(len(df_grid), cur_week_int, dtype='int32')  # 전체 길이로 채움
        start_full[mask3] = start                                        # cond3 위치만 대입
        # ----------------------------------------------------

        # ④ “범위” =  max(cur_week, start) ~ bod_arr  구간 → False
        #    이후 주차                                → True
        m3_range = (cond3 &                                             # Real BO
                    (row_week >= np.maximum(cur_week_int,
                                            start_full)) &             # 시작
                    (row_week <= bod_arr))                              # 끝

        # 범위 안 = False
        df_grid.loc[m3_range,               COL_BO_FCST_LOCK] = False
        # 범위 밖 & cond3 = True
        df_grid.loc[cond3 & ~m3_range,      COL_BO_FCST_LOCK] = True

    # ── 6) 색상 공백 → 기본 WHITE, 카테고리 재설정 ───────────────
    df_grid.loc[df_grid[COL_BO_FCST_COLOR_COND] == '', COL_BO_FCST_COLOR_COND] = COLOR_WHITE
    df_grid[COL_BO_FCST_COLOR_COND] = pd.Categorical(
        df_grid[COL_BO_FCST_COLOR_COND],
        categories=[COLOR_WHITE, COLOR_GRAY, COLOR_DGRAY_RED,
                    COLOR_DARKBLUE, COLOR_LIGHTBLUE,
                    COLOR_LIGHTRED, COLOR_DARKRED]
    )
    logger.debug("Step 03-2. 6) 색상 공백 → 기본 WHITE, 카테고리 재설정")
    gc.collect()
    return df_grid


# ────────────────────────────────────────────────────────────────
# Step 04 – Virtual-BO(가상 BO) 생성
# ────────────────────────────────────────────────────────────────
@_decoration_
def fn_step04_generate_virtual_bo(
        df_grid          : pd.DataFrame,      # ← Step 03-2 결과 (in-place 추가)
        df_bod_map       : pd.DataFrame,      # ← Step 03-1 결과 (Item·Loc별 LT-Week)
        cur_week_norm    : str,               # e.g. '202506'
        max_week_norm    : str,               # e.g. '202606'
        df_bo_src        : pd.DataFrame       # ← df_in_BO_FCST  (VBO max 파악용)
    ) -> pd.DataFrame:
    """
    ① 기존 BO( VBO='-', BO_ID='-' ) 중 **BO FCST>0** 이 있는 Item·Ship-To·Loc
       그룹마다 **새 VBO** 를 하나씩 만든다.    ② 새 VBO ID는
         • df_in_BO_FCST + df_grid 전체에서 찾은 Max ID 뒤에 +1  
         • 모두 ‘-’ 뿐이면 `VBO_00000000` 부터 시작
         • 포맷 : `'VBO_' + 8자리 숫자`

    ③ 생성 규칙 (그룹 g, LT-Week=W_LT, Horizon=CW~MaxWeek)
         • Time.[Partial Week] =CW~MaxWeek 전구간 복제
         • 새 VBO 행은 BO FCST.Lock = True, Colour는 '' 유지
         • 행 W_LT 의 BO FCST = 기존 W_LT 의 값, Lock=False
         • W_LT 가 partial-week 이면 A/B 모두 생성
    """
    # ───── 1. 새 VBO 시퀀스 번호 ──────────────────────────────
    def _extract_seq(vbo: str) -> int:
        m = re.match(r'VBO_(\d{8})$', vbo)
        return int(m.group(1)) if m else -1                    # ‘-’ → -1

    all_vbos = pd.concat([
        df_bo_src[COL_VIRTUAL_BO_ID],
        df_grid[COL_VIRTUAL_BO_ID]
    ], ignore_index=True)

    max_seq = all_vbos.apply(_extract_seq).max()        # -1 → 존재 X
    max_seq = max_seq if max_seq >= 0 else 0            # 모두 ‘-’ ⇒ 0
    seq_gen = (max_seq + i for i in range(1, 10**9))    # 무한 generator

    fmt_vbo = lambda s: f"VBO_{s:08d}"
    # → 중간 Series 메모리 해제
    del all_vbos; gc.collect()
    # ───── 2. LT-Week lookup (Item, Loc → W_LT) ─────────────
    lt_map = df_bod_map.set_index([COL_ITEM, COL_LOC])[COL_BO_TOTAL_BOD_LT]

    # ───── 3. 대상(Basic BO, FCST>0) 그룹 추출 ───────────────
    base_mask = (
        (df_grid[COL_VIRTUAL_BO_ID] == '-') &
        (df_grid[COL_BO_ID]         == '-') # &
        # (df_grid[COL_BO_FCST]       > 0)
    )
    base_df = df_grid[base_mask]

    grp_cols = [COL_ITEM, COL_SHIP_TO, COL_LOC]
    groups   = base_df.groupby(grp_cols, sort=False)

    # → base_df 자체는 이후 재사용하지 않으므로 즉시 삭제
    del base_df; gc.collect()

    new_rows_all = []

    for gkey, gdf in groups:
        

        # (b) LT-Week (문자열, ‘YYYYWW[A/B]’)
        try:
            w_lt = lt_map.loc[(gkey[0], gkey[2])]
        except KeyError:                  
            # LT 없으면 당주주차 사용
            w_lt = cur_week_norm + 'A'

        # A/B 주차 보정
        suffixes_lt = ['A', 'B'] if gfn_is_partial_week(str(w_lt)[:6]) else ['A']
        w_lt_full = [f"{str(w_lt)[:6]}{sfx}" for sfx in suffixes_lt]

        # (c) 원본 BO_FCST (W_LT 에 있는 경우)
        mask_lt = (
            (gdf[COL_TIME_PW].isin(w_lt_full))  &
            (gdf[COL_BO_FCST] > 0) 
        )
        fcst_lt = (
            gdf.loc[mask_lt, COL_BO_FCST]
            #    .max()           # 여러 행 존재 가능 → 최대 사용
        )
        # fcst_lt = 0 if pd.isna(fcst_lt) else int(fcst_lt)
        # fn_log_dataframe(gdf,'_'.join(list(gkey)).replace('/',''))
        if mask_lt.any():
            
            # (a) 새 VBO ID
            new_vbo = fmt_vbo(next(seq_gen))
            # (d) 새 행 복제
            add = gdf.copy()
            add[COL_VIRTUAL_BO_ID]   = new_vbo
            add[COL_BO_ID]           = '-'
            add[COL_BO_FCST]         = 0
            add[COL_BO_FCST_LOCK]    = True         # 기본 Lock True
            # Colour는 그대로 두면 '' / GRAY 유지

            # LT-행(들)만 BO_FCST & Lock 업데이트
            # m_lt = (add[COL_TIME_PW].isin(w_lt_full))
            add.loc[mask_lt, COL_BO_FCST]      = fcst_lt
            add.loc[mask_lt, COL_BO_FCST_LOCK] = False

            # ─────────────────────────────────────────────────────────────
            # 마지막 주차가 partial-week(‘A/B’) 인지 검사하고
            # 누락된 suffix 가 있으면 복제하여 추가
            # ─────────────────────────────────────────────────────────────
            # add 의 마지막(Time.[Partial Week]) 주차
            last_pw   = add[COL_TIME_PW].iloc[-1]
            base_yyyyww = last_pw[:6]                     # ‘YYYYWW’
            if gfn_is_partial_week(base_yyyyww):          # partial-week 여부
                # 현재 add 가 담고 있는 suffix 집합
                have_sfx = set(add[COL_TIME_PW].str[-1])
                # ‘A’ / ‘B’ 중 빠진 suffix 만 골라서 새 행 생성
                for miss_sfx in {'A', 'B'} - have_sfx:
                    new_row            = add.iloc[-1].copy(deep=True)
                    new_row[COL_TIME_PW]       = f'{base_yyyyww}{miss_sfx}'

                    # just before pd.concat
                    bool_cols = [COL_BO_FCST_LOCK]          # any columns you know are boolean
                    new_row[bool_cols] = new_row[bool_cols].astype('bool', copy=False)
                    add = pd.concat([add, new_row.to_frame().T], ignore_index=True)

            new_rows_all.append(add)
            del add; gc.collect()

        # → loop 내부에서 사용 끝난 gdf·add 속히 해제
        del gdf; gc.collect()

    # groups 객체 해제
    del groups; gc.collect()

    if new_rows_all:
        df_grid = pd.concat([df_grid, *new_rows_all], ignore_index=True)
        # 조각(DataFrame) 리스트 삭제
        del new_rows_all; gc.collect()


    # ───── 4. 타입 & 정렬 정리 ────────────────────────────────
    df_grid[COL_VIRTUAL_BO_ID]  = df_grid[COL_VIRTUAL_BO_ID].astype('category')
    df_grid[COL_BO_FCST]        = df_grid[COL_BO_FCST].fillna(0).astype('int32')
    df_grid.sort_values(
        by=[COL_ITEM, COL_SHIP_TO, COL_LOC, COL_VIRTUAL_BO_ID, COL_TIME_PW],
        inplace=True,
        ignore_index=True
    )

    # lt_map 역시 더 이상 필요 없음
    del lt_map; gc.collect()

    return df_grid


################################################################################################################
# End of Step Functions
################################################################################################################

################################################################################################################
# Start of Main
################################################################################################################
if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    # Output 테이블 선언
    out_Demand = pd.DataFrame()
    output_dataframes = {}
    input_dataframes = {}
    try:
        if is_local:
            Version = 'CWV_DP'
            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------

            # input_folder_name  = str_instance
            # output_folder_name = str_instance
            # input_folder_name  = 'PYForecastB2BLockAndRolling_o9'
            # output_folder_name = 'PYForecastB2BLockAndRolling_o9'
            # input_folder_name  = 'PYForecastB2BLockAndRolling_o9_0605'
            # output_folder_name = 'PYForecastB2BLockAndRolling_o9_0605'
            input_folder_name  = 'PYForecastB2BLockAndRolling_o9_LH015IEACFS'
            output_folder_name = 'PYForecastB2BLockAndRolling_o9_LH015IEACFS'

            # # -- COL_VIRTUAL_BO_ID 이 - 으로만 구성된 경우
            # input_folder_name  = 'PYForecastB2BLockAndRolling_bo_'
            # output_folder_name = 'PYForecastB2BLockAndRolling_bo_'
            
            # ------
            str_input_dir = f'Input/{input_folder_name}'
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
            CurrentPartialWeek = '202506A'

        # --------------------------------------------------------------------------
        # df_input 체크 시작
        # --------------------------------------------------------------------------
        logger.Note(p_note='df_input 체크 시작', p_log_level=LOG_LEVEL.debug())
        fn_process_in_df_mst()
         # 입력 변수 중 데이터가 없는 경우 경고 메시지를 출력한다.
        for in_df in input_dataframes:
            fn_check_input_table(input_dataframes[in_df], in_df, '1')


        df_max_partial_week = input_dataframes[STR_DF_IN_MAX_PW]
        max_week = df_max_partial_week[COL_TIME_PW].max()
        # max_week_normalized = normalize_week(max_week)
        max_week_normalized = normalize_week(max_week)
        current_week_normalized = normalize_week(CurrentPartialWeek)


        ################################################################################################################
        # Start of processing
        ################################################################################################################
       

        ################################################################################################################
        # Step 01-1 ─ RTS/EOS 마스터 로드 & 기본 클렌징
        ################################################################################################################
        dict_log = {
            'p_step_no' : 101,
            'p_step_desc': 'Step 01-1 – load / clean RTS·EOS master'
        }
        df_fn_rts_eos = fn_step01_1_load_rts_eos(**dict_log)
        # ── 4) 메모리 정리 ──
        del input_dataframes[STR_DF_IN_MST_RTS_EOS]
        gc.collect()

        # 이후 단계에서 바로 df_fn_rts_eos 사용
        # 예) df_fn_rts_eos = fn_step01_2_enrich_rts_eos(df_fn_rts_eos, ...)
        ################################################################################################################
        # Step 01-2 ─ RTS/EOS 날짜 → Partial-Week 변환 + Helper 주차 파생
        ################################################################################################################
        dict_log = {
            'p_step_no' : 102,
            'p_step_desc': 'Step 01-2 – enrich RTS·EOS with helper weeks'
        }
        df_fn_rts_eos = fn_step01_2_enrich_rts_eos(
            df_fn_rts_eos,
            current_week_normalized,   # 예: '202506'
            max_week_normalized,       # 예: '202552'
            **dict_log
        )
        fn_log_dataframe(df_fn_rts_eos, f'step01_2_{STR_DF_FN_RTS_EOS}')
        # fn_log_dataframe(output_dataframes[STR_DF_FN_RTS_EOS], f'step01_2_output_{STR_DF_FN_RTS_EOS}')
        # 로그 함수는 내부에서 호출됨; 필요 시 추가 디버그 출력만 작성

        ################################################################################################################
        # Step 01-4 – RTS × Partial-Week 그리드
        ################################################################################################################
        dict_log = {
            'p_step_no' : 104,
            'p_step_desc': 'Step 01-4 – build RTS·EOS partial-week grid'
        }
        df_fn_rts_eos_pw = fn_step01_4_build_rts_eos_pw(
            df_fn_rts_eos,
            input_dataframes[STR_DF_IN_TIME_PW],
            **dict_log
        )
        # df_fn_rts_eos_pw 는 이후 Step 01-5 함수에서 in-place 로 수정될 예정

        ################################################################################################################
        # Step 01-5 – Lock/Colour 적용
        ################################################################################################################
        dict_log = {
            'p_step_no': 105,
            'p_step_desc': 'Step 01-5 – apply 5-band lock & colour'
        }
        df_fn_rts_eos_pw = fn_step01_5_set_lock_values(
            df_fn_rts_eos_pw,
            df_fn_rts_eos,
            current_week_normalized,
            max_week_normalized,
            **dict_log
        )
        fn_log_dataframe(df_fn_rts_eos_pw, f'step01_5_{STR_DF_FN_RTS_EOS_PW}')
        # fn_check_input_table(df_fn_rts_eos_pw, f'step01_5_{STR_DF_FN_RTS_EOS_PW}','0')
        # df_fn_rts_eos_pw 는 같은 인스턴스이므로 output_dataframes 재등록 불필요

        ################################################################################################################
        # Step 01-6 – Ship-To 차원 LUT 구축
        ################################################################################################################
        dict_log = {
            'p_step_no': 106,
            'p_step_desc': 'Step 01-6 – load Ship-To dimension LUT'
        }
        df_fn_shipto_dim = fn_step01_6_load_shipto_dimension(
            input_dataframes[STR_DF_IN_SALES_DOMAIN_DIM],
            **dict_log
        )
        output_dataframes[STR_DF_FN_SHIPTO_DIM] = df_fn_shipto_dim
        fn_log_dataframe(df_fn_shipto_dim, STR_DF_FN_SHIPTO_DIM)
        # 이후 단계에서 df_fn_shipto_dim 을 참조 (LV 변환용)

        ################################################################################################################
        # Step 01-7 – ASN item-week grid  (Lock/Color 초기화 포함)
        ################################################################################################################
        dict_log = {
            'p_step_no': 107,
            'p_step_desc': 'Step 01-7 – build ASN item-week grid'
        }
        df_fn_sales_product_asn_item_week = fn_step01_7_prepare_asn_week(
            input_dataframes[STR_DF_IN_SALES_PRODUCT_ASN],
            input_dataframes[STR_DF_IN_TIME_PW],
            current_week_normalized,
            **dict_log
        )
        output_dataframes[STR_DF_FN_SALES_ASN_WEEK] = df_fn_sales_product_asn_item_week
        fn_log_dataframe(df_fn_sales_product_asn_item_week, f'step01_7_{STR_DF_FN_SALES_ASN_WEEK}')
        # Step 01-8 에서 df_fn_sales_product_asn_item_week 를 in-place 로 갱신할 예정

        ################################################################################################################
        # Step 01-8 – RTS 색상/Lock → ASN-Week 복사
        ################################################################################################################
        dict_log = {
            'p_step_no': 108,
            'p_step_desc': 'Step 01-8 – propagate lock/colour (Ship-To level map)'
        }
        df_fn_sales_product_asn_item_week = fn_step01_8_apply_rts_colour_to_asn(
            df_fn_sales_product_asn_item_week,          # Step 01-7
            df_fn_rts_eos_pw,                           # Step 01-5
            output_dataframes[STR_DF_FN_SHIPTO_DIM],    # Step 01-6  df_fn_shipto_dim
            **dict_log
        )
        fn_log_dataframe(df_fn_sales_product_asn_item_week, f'step01_8_{STR_DF_FN_SALES_ASN_WEEK}')
        # 같은 인스턴스이므로 output_dataframes 재등록 불필요

        ################################################################################################################
        # Step 02-1 – BO Grid 생성
        ################################################################################################################
        dict_log = {'p_step_no':201, 'p_step_desc':'Step 02-1 – build BO grid'}
        df_fn_sales_product_asn_item_week = fn_step02_1_build_bo_color_grid(
            df_fn_sales_product_asn_item_week, 
            **dict_log
        )
        # output_dataframes[STR_DF_FN_BO_FCST_ASN] = df_fn_sales_product_asn_item_week
        fn_log_dataframe(df_fn_sales_product_asn_item_week, f'step02_1_{STR_DF_FN_SALES_ASN_WEEK}')
        # fn_check_input_table(df_fn_sales_product_asn_item_week,f'step02_1_{STR_DF_FN_SALES_ASN_WEEK}','1')
        ################################################################################################################
        # Step 02-2 – BO 테이블 적용
        ################################################################################################################

        dict_log = {'p_step_no':202, 'p_step_desc':'Step 02-2 – merge BO FCST'}
        df_fn_bo_fcst_asn = fn_step02_2_apply_grid_to_bo(
            df_fn_sales_product_asn_item_week,                     # 02-1 인스턴스 (in-place 수정)
            input_dataframes[STR_DF_IN_BO_FCST],   # BO 원본
            **dict_log
        )
        output_dataframes[STR_DF_FN_BO_FCST_ASN] = df_fn_bo_fcst_asn
        fn_log_dataframe(df_fn_bo_fcst_asn, f'step02_2_{STR_DF_FN_BO_FCST_ASN}')

        ################################################################################################################
        # Step 03-1  : prepare BOD LT
        ################################################################################################################
        dict_log = {'p_step_no':301, 'p_step_desc':'Step 03-1 – prepare BOD LT'}
        df_fn_total_bod_lt = fn_step03_1_prepare_total_bod_lt(
            input_dataframes[STR_DF_IN_TOTAL_BOD_LT],
            current_week_normalized,      # 문자열 '202506'
            **dict_log
        )
        fn_log_dataframe(df_fn_total_bod_lt, f'step03_1_{STR_DF_FN_TOTAL_BOD_LT}')
        
        ################################################################################################################
        # Step 03-2  : apply Lock/Color rules
        ################################################################################################################
        dict_log = {'p_step_no':302, 'p_step_desc':'Step 03-2 – apply lock rules'}
        df_fn_bo_fcst_asn = fn_step03_2_apply_lock_conditions(
            df_fn_bo_fcst_asn,
            df_fn_total_bod_lt,
            df_fn_rts_eos,                      # Step 01-2 결과  (STD Ship-To 기준 RTS/EOS 주차)
            df_fn_shipto_dim,                   # Ship-To Dim (LV_CODE, Std1·Std2)
            int(current_week_normalized),
            int(max_week_normalized),
            **dict_log
        )
        fn_log_dataframe(df_fn_bo_fcst_asn, f'step03_2_{STR_DF_FN_BO_FCST_ASN}')
        # output_dataframes[STR_DF_FN_BO_FCST_ASN] = df_fn_bo_fcst_asn


        ##################################################################
        # Step 04 – 가상 BO 생성
        ##################################################################
        dict_log = {
            'p_step_no'  : 401,
            'p_step_desc': 'Step 04 – generate Virtual BO',
        }
        df_fn_bo_fcst_asn = fn_step04_generate_virtual_bo(
            df_fn_bo_fcst_asn,           # ← Step 03-2 결과 (in-place)
            df_fn_total_bod_lt,          # ← Step 03-1 결과
            current_week_normalized,     # e.g. '202506'
            max_week_normalized,         # e.g. '202606'
            input_dataframes[STR_DF_IN_BO_FCST],   # 원본 BO 테이블
            **dict_log
        )
        fn_log_dataframe(df_fn_bo_fcst_asn, f'step04_{STR_DF_FN_BO_FCST_ASN}')

        ################################################################################################################
        # 최종 Output 정리
        ################################################################################################################
        dict_log = {
            'p_step_no': 900,
            'p_step_desc': '최종 Output 정리 - out_Demand',
            'p_df_name': 'out_Demand'
        }
        out_Demand = fn_output_formatter(df_fn_bo_fcst_asn, Version, **dict_log)
        # fn_log_dataframe(out_Demand, f'out_Demand')

        # validation
        """
            import duckdb
            duckdb.register('df_in_BO_FCST',input_dataframes[STR_DF_IN_BO_FCST])
            duckdb.register('df_fn_total_bod_lt',df_fn_total_bod_lt)
            duckdb.register('df_fn_bo_fcst_asn',df_fn_bo_fcst_asn)
            
            duckdb.sql(f'''
                select * from df_fn_bo_fcst_asn 
                where  "{COL_LOC}"= 'S356' 
                and "{COL_TIME_PW}" like '202510%' 
                and "{COL_BO_FCST}" > 0  
            ''').show()

            duckdb.sql(f'''
                select * from df_in_BO_FCST 
                where 1=1
                -- and "{COL_TIME_PW}" like '202510%'
                and "{COL_BO_FCST}" > 0  
            ''').show()
            
            del df_fn_bo_fcst_asn
            gc.collect()
        """
        # for df in output_dataframes:
        #     fn_log_dataframe(output_dataframes[df], df)


    except Exception as e:
        trace_msg = traceback.format_exc()
        logger.Note(p_note=trace_msg, p_log_level=LOG_LEVEL.debug())
        logger.Error()
        if flag_exception:
            raise Exception(e)
        else:
            logger.info(f'{str_instance} exit - {time.strftime("%Y-%m-%d - %H:%M:%S")}')

    finally:
        # MediumWeight 실행 시 Header 없는 빈 데이터프레임이 Output이 되는 경우 오류가 발생함.
        # 이 오류를 방지하기 위해 Output이 빈 경우을 체크하여 Header를 만들어 줌.
        if out_Demand.empty:
            out_Demand = fn_set_header()
            fn_log_dataframe(out_Demand, 'out_Demand')

        if is_local:
            log_file_name = common.G_PROGRAM_NAME.replace('py', 'log')
            log_file_name = f'log/{log_file_name}'

            shutil.copyfile(log_file_name, os.path.join(str_output_dir, os.path.basename(log_file_name)))

            # prografile copy
            program_path = f"{os.getcwd()}/NSCM_DP_UI_Develop/{str_instance}.py"
            shutil.copyfile(program_path, os.path.join(str_output_dir, os.path.basename(program_path)))

            # # task.json copy
            # task_path = f"{os.getcwd()}/.vscode/tasks.json"
            # shutil.copyfile(task_path, os.path.join(str_output_dir, os.path.basename(task_path)))

            # log
            # input_path = f'{str_output_dir}/input'
            # os.makedirs(input_path,exist_ok=True)
            # for input_file in input_dataframes:
            #     input_dataframes[input_file].to_csv(input_path + "/"+input_file+".csv", encoding="UTF8", index=False)

            # # log
            # output_path = f'{str_output_dir}/output'
            # os.makedirs(output_path,exist_ok=True)
            # for output_file in output_dataframes:
            #     output_dataframes[output_file].to_csv(output_path + "/"+output_file+".csv", encoding="UTF8", index=False)

        # logger.info(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')
        logger.Finish()
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::') # 25.05.12 need warning Log by Logger Issue
        
################################################################################################################
# End of Main
################################################################################################################       
