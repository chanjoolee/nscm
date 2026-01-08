"""

"""

import os,sys
import time,datetime,shutil
import inspect
import traceback
import pandas as pd
from NSCMCommon import NSCMCommon as common
from NSCMCommon import VDCommon as vdCommon
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
str_instance = 'PYCreateDataNewSalesProductASNEstoreModify'
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

########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################
logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
LOG_LEVEL = common.G_log_level

########################################################################################################################
#  CONSTANTS ─ PYCreateDataNewSalesProductASNEstoreModify
#  ▸ naming rule :  STR_  = dataframe-key,  COL_ = column-name  (see prompt.txt)  
######################################################################################################################### ──────────────────────────────────────────────────────────────────
# 0)  Program / I/O 환경
# ──────────────────────────────────────────────────────────────────
# STR_INSTANCE          = 'PYCreateDataNewSalesProductASNEstoreModify'
# STR_INPUT_DIR         = f'Input/{STR_INSTANCE}'
# STR_OUTPUT_DIR        = f'Output/{STR_INSTANCE}'

# IS_LOCAL              = common.gfn_get_isLocal()     # NSCMCommon helper
# V_CHUNK_SIZE          = 100_000                      # 대용량 처리 기본 청크

# ──────────────────────────────────────────────────────────────────
# 1)  Input DataFrame keys
# ──────────────────────────────────────────────────────────────────
STR_DF_IN_TIME                      = 'df_in_Time'
STR_DF_IN_ESTORE_SHIPTO             = 'df_in_Sales_Domain_Estore'
STR_DF_IN_SALES_PRODUCT_ASN_DELTA   = 'df_in_Sales_Product_ASN_Delta'

# ──────────────────────────────────────────────────────────────────
# 2)  Derived / Intermediate
# ──────────────────────────────────────────────────────────────────
STR_DF_FN_SALES_ASN_DELTA           = 'df_fn_Sales_Product_ASN_Delta'   # Step-01 결과

# ──────────────────────────────────────────────────────────────────
# 3)  Output DataFrame keys
# ──────────────────────────────────────────────────────────────────
STR_DF_OUT_USER_GI_RATIO            = 'df_output_Sell_In_User_Modify_GI_Ratio'
STR_DF_OUT_ISSUE_GI_RATIO           = 'df_output_Sell_In_Issue_Modify_GI_Ratio'

# ──────────────────────────────────────────────────────────────────
# 4)  Common column names   (spec ↔ ERD 일치) 
# ──────────────────────────────────────────────────────────────────
COL_VERSION          = 'Version.[Version Name]'           # category  ('CWV_DP')
COL_SHIP_TO          = 'Sales Domain.[Ship To]'           # category  (Std-5 / Std-6)
COL_ITEM             = 'Item.[Item]'                      # category  (SKU)
COL_LOC              = 'Location.[Location]'              # category
COL_TIME_WK          = 'Time.[Week]'                      # int32  (YYYYWW)
COL_TIME_PW          = 'Time.[Partial Week]'              # category  (YYYYWW[A/B])
COL_PG_ASN_DELTA     = 'Sales Product ASN Delta'          # category  ('Y' / 'N')

# ──────────────────────────────────────────────────────────────────
# 5)  Measure columns – USER
# ──────────────────────────────────────────────────────────────────
COL_SIN_USER_LONG    = 'S/In User Modify GI Ratio(Long Tail)'
COL_SIN_USER_W7      = 'S/In User Modify GI Ratio(W+7)'
COL_SIN_USER_W6      = 'S/In User Modify GI Ratio(W+6)'
COL_SIN_USER_W5      = 'S/In User Modify GI Ratio(W+5)'
COL_SIN_USER_W4      = 'S/In User Modify GI Ratio(W+4)'
COL_SIN_USER_W3      = 'S/In User Modify GI Ratio(W+3)'
COL_SIN_USER_W2      = 'S/In User Modify GI Ratio(W+2)'
COL_SIN_USER_W1      = 'S/In User Modify GI Ratio(W+1)'
COL_SIN_USER_W0      = 'S/In User Modify GI Ratio(W+0)'

LIST_USER_GI_MEAS = [
    COL_SIN_USER_LONG, COL_SIN_USER_W7, COL_SIN_USER_W6, COL_SIN_USER_W5,
    COL_SIN_USER_W4,  COL_SIN_USER_W3, COL_SIN_USER_W2, COL_SIN_USER_W1,
    COL_SIN_USER_W0
]

# ──────────────────────────────────────────────────────────────────
# 6)  Measure columns – ISSUE   (USER → ISSUE rename)
# ──────────────────────────────────────────────────────────────────
COL_SIN_ISSUE_LONG   = 'S/In Issue Modify GI Ratio(Long Tail)'
COL_SIN_ISSUE_W7     = 'S/In Issue Modify GI Ratio(W+7)'
COL_SIN_ISSUE_W6     = 'S/In Issue Modify GI Ratio(W+6)'
COL_SIN_ISSUE_W5     = 'S/In Issue Modify GI Ratio(W+5)'
COL_SIN_ISSUE_W4     = 'S/In Issue Modify GI Ratio(W+4)'
COL_SIN_ISSUE_W3     = 'S/In Issue Modify GI Ratio(W+3)'
COL_SIN_ISSUE_W2     = 'S/In Issue Modify GI Ratio(W+2)'
COL_SIN_ISSUE_W1     = 'S/In Issue Modify GI Ratio(W+1)'
COL_SIN_ISSUE_W0     = 'S/In Issue Modify GI Ratio(W+0)'

LIST_ISSUE_GI_MEAS = [
    COL_SIN_ISSUE_LONG, COL_SIN_ISSUE_W7, COL_SIN_ISSUE_W6, COL_SIN_ISSUE_W5,
    COL_SIN_ISSUE_W4,  COL_SIN_ISSUE_W3, COL_SIN_ISSUE_W2, COL_SIN_ISSUE_W1,
    COL_SIN_ISSUE_W0
]

# ──────────────────────────────────────────────────────────────────
# 7)  Fixed literal
# ──────────────────────────────────────────────────────────────────
# CONST_VERSION        = 'CWV_DP'                # Version value for outputs
F_ZERO_FLOAT32       = np.float32(0.0)         # 0.0 with target dtype


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
        # if is_local and not df_p_source.empty and flag_csv:
        if is_local and flag_csv:
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

########################################################################################################################
# ┌───────────────────────────────────────────────────────────────────────────────┐
# │  STEP-00 · LOAD INPUT CSVs → input_dataframes                                │
# └───────────────────────────────────────────────────────────────────────────────┘
#  • 목적 : 로컬 개발 환경에서 CSV → pandas 로 읽어 input_dataframes 에 적재
#  • 대상 : Time / eStore Ship-To / Sales-Product-ASN-Delta  3개 입력 테이블
#  • 참고 : o9 서버에서는 df_in_* 글로벌 변수가 이미 존재하므로 mapping 없이 바로 할당
########################################################################################################################
# ▶ DATAFRAME-NAME CONSTANTS (spec 과 1:1 대응)
# STR_DF_IN_TIME                       = 'df_in_Time'
# STR_DF_IN_ESTORE_SHIPTO              = 'df_in_Sales_Domain_Estore'
# STR_DF_IN_SALES_PRODUCT_ASN_DELTA    = 'df_in_Sales_Product_ASN_Delta'# ▶ COLUMN-NAME CONSTANTS  (자주 사용하는 것만 선언)
# COL_TIME_WK      = 'COL_TIME_WK'            # df_in_Time
# COL_SHIP_TO      = 'COL_SHIP_TO'            # 공통 Ship-To
# COL_VERSION      = 'COL_VERSION'            # ‘CWV_DP’ 등
# COL_PG_ASN_DELTA = 'COL_PG_ASN_DELTA'       # ‘Y’ / ‘N’



@_decoration_
def fn_process_in_df_mst() -> None:
    """
    3 개 입력 테이블 로드 & 메모리 최적화
    --------------------------------------------------------------
    • 'Sales Domain*' 열  → str → category
    • Time.[Partial Week] → ordered category
    • object             → unordered category
    • int64/float64      → down-cast
    • BO_FCST / BOD_LT   → 결측치·dtype 보정
    """
    # 1) ───── 파일 매핑 + 로드 ─────────────────────────────────
    file_to_df = {
        'df_in_Time'                    : STR_DF_IN_TIME,
        'df_in_Sales_Domain_Estore'     : STR_DF_IN_ESTORE_SHIPTO,
        'df_in_Sales_Product_ASN_Delta' : STR_DF_IN_SALES_PRODUCT_ASN_DELTA
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
        input_dataframes[STR_DF_IN_TIME]                    = df_in_Time
        input_dataframes[STR_DF_IN_ESTORE_SHIPTO]           = df_in_Sales_Domain_Estore
        input_dataframes[STR_DF_IN_SALES_PRODUCT_ASN_DELTA] = df_in_Sales_Product_ASN_Delta

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


    
    if is_local:
        input_path = f'{str_output_dir}/input'
        os.makedirs(input_path,exist_ok=True)
        for input_file in input_dataframes:
            input_dataframes[input_file].to_csv(input_path + "/"+input_file+".csv", encoding="UTF8", index=False)

"""
# ═══════════════════════════════════════════════════════════════════════════════
# DUCKDB QUICK CHECK (OPTIONAL – LOCAL DEBUG)
# ═══════════════════════════════════════════════════════════════════════════════
# import duckdb, os, glob
# for df_name, df in input_dataframes.items():
#     duckdb.register(df_name, df)
# print(duckdb.query('SELECT COUNT(*) FROM df_in_Sales_Product_ASN_Delta').fetchone())
"""


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


@_decoration_
def fn_step01_filter_asn_delta() -> pd.DataFrame:
    """
    Step-01 : 필터링 & 중복제거  ➜ df_fn_Sales_Product_ASN_Delta
    -----------------------------------------------------------------
      1) COL_PG_ASN_DELTA == 'Y' 행만 선택
      2) COL_SHIP_TO 가 df_in_Sales_Domain_Estore 에 존재하는 행만 유지
      3) Version.[Version Name] 컬럼 삭제
      4) (COL_SHIP_TO, COL_ITEM, COL_LOC) 기준 중복 제거
      5) 결과를 output_dataframes[STR_DF_FN_SALES_ASN_DELTA] 에 저장
         · 결과 row 가 0 ⇒ False 반환 (프로그램 상위 단계에서 Early-Exit 용)
         · 결과 row >0 ⇒ True  반환
    -----------------------------------------------------------------
    Returns
    -------
    bool
        데이터가 존재하면 True, 없으면 False
    """

    # ── (1) 원본 DF 로드 ─────────────────────────────────────────────
    df_delta   = input_dataframes[STR_DF_IN_SALES_PRODUCT_ASN_DELTA]
    df_estore  = input_dataframes[STR_DF_IN_ESTORE_SHIPTO]

    # ── (2) 필터 : ASN Delta == 'Y'  &  Ship-To ∈ eStore ───────────
    mask_y     = (df_delta[COL_PG_ASN_DELTA] == 'Y')
    estore_set = set(df_estore[COL_SHIP_TO].unique())
    mask_es    = df_delta[COL_SHIP_TO].isin(estore_set)

    df_filtered = df_delta.loc[mask_y & mask_es,
                               [COL_SHIP_TO, COL_ITEM, COL_LOC]].drop_duplicates()

    # ── (3) 결과 저장 ───────────────────────────────────────────────
    # output_dataframes[STR_DF_FN_SALES_ASN_DELTA] = df_filtered

    # ── (4) 로그 & 메모리 정리 ─────────────────────────────────────
    row_cnt = len(df_filtered)
    logger.Note(f"[Step01] ASN Δ after filter: {row_cnt} rows")

    del df_delta, df_estore, mask_y, mask_es
    gc.collect()

    return df_filtered

# ──────────────────────────────────────────────────────────────────────────────
# UTIL · 빈 스키마 DataFrame 생성
# ──────────────────────────────────────────────────────────────────────────────
def _empty_user_gi_df() -> pd.DataFrame:
    """
    스펙/ERD 와 동일한 컬럼·dtype을 갖는
    빈 df_output_Sell_In_User_Modify_GI_Ratio DataFrame 반환
    --------------------------------------------------------------------------
    • 컬럼 순서 : Version, Ship-To, Item, Loc, Time.[Week], 9×Measure
    • dtype
        - 문자열 계열         → category
        - Time.[Week]         → int32
        - Measure(9개)        → float32
    """
    cols = ([COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_WK]
            + LIST_USER_GI_MEAS)    
    df = pd.DataFrame(columns=cols)

    # dtype 지정
    cat_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC]
    df[cat_cols]      = df[cat_cols].astype('category')
    df[COL_TIME_WK]   = df[COL_TIME_WK].astype(np.int32)
    for m in LIST_USER_GI_MEAS:
        df[m] = df[m].astype(np.float32)

    return df

# ┌──────────────────────────────────────────────────────────────────────┐
# │  STEP-02-1 · USER  GI Ratio 생성                                    │
# └──────────────────────────────────────────────────────────────────────┘
@_decoration_
def fn_step02_1_create_user_modify_ratio(
        df_base : pd.DataFrame,           # ← step-01 결과 (ShipTo·Item·Loc unique)
        df_time : pd.DataFrame            # ← df_in_Time  (COL_TIME_PW 포함)
) -> pd.DataFrame:
    """
    Returns
    -------
    pd.DataFrame
        df_output_Sell_In_User_Modify_GI_Ratio
        (빈 DF 스키마: ShipTo·Item·Loc·PartialWeek = PK, 9 measure = float32)
    """
    # logger.Note(f"[{dict_log['p_step_no']}] {dict_log['p_step_desc']}")    
    # ── (0) base DF 없으면 빈 스키마 DF 반환 ─────────────────────────
    if df_base.empty:
        logger.Note("[Step-02-1] base DF empty → return empty user-GI DF")
        return _empty_user_gi_df()

    # ── (1) Cartesian join : (ShipTo,Item,Loc) × Partial-Week─────────
    df_cross = (
        df_base.assign(_k=1)
               .merge(df_time[[COL_TIME_WK]].assign(_k=1), on="_k")
               .drop("_k", axis=1)
               .reset_index(drop=True)
    )

    # ── (2) Version / dtype / category 최적화 ────────────────────────
    # # Version 파라미터 (로컬·서버 공통)
    # try:                        # o9 환경 : 이미 주입
    #     Version
    # except NameError:           # 로컬 : 기본값
    #     Version = 'CWV_DP'

    df_cross[COL_VERSION] = Version
    # 문자열 열 → category
    for col in [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC]:
        df_cross[col] = df_cross[col].astype("category")

    # ── (3) 9개 Measure 컬럼 float32 0.0 세팅 ───────────────────────
    for col in LIST_USER_GI_MEAS:
        df_cross[col] = F_ZERO_FLOAT32
    df_cross = df_cross.astype({c: np.float32 for c in LIST_USER_GI_MEAS})

    # ── (4) 컬럼 순서 맞춤 ──────────────────────────────────────────
    col_order = ([COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_WK]
                 + LIST_USER_GI_MEAS)
    df_cross = df_cross[col_order]

    logger.Note(f"[Step-02-1] user-GI rows created = {len(df_cross):,}")
    return df_cross


# ──────────────────────────────────────────────────────────────────────────────
# UTIL · 빈 Issue-GI DataFrame
# ──────────────────────────────────────────────────────────────────────────────
def _empty_issue_gi_df() -> pd.DataFrame:
    """
    df_output_Sell_In_Issue_Modify_GI_Ratio 스키마와 동일한 빈 DataFrame
    """
    cols = ([COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_WK]
            + LIST_ISSUE_GI_MEAS)
    df = pd.DataFrame(columns=cols)

    # dtype 지정
    cat_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC]
    df[cat_cols]    = df[cat_cols].astype("category")
    df[COL_TIME_WK] = df[COL_TIME_WK].astype(np.int32)
    for m in LIST_ISSUE_GI_MEAS:
        df[m] = df[m].astype(np.float32)

    return df
    
# ┌──────────────────────────────────────────────────────────────────────┐
# │  STEP-02-2 · ISSUE  GI Ratio 생성                                   │
# └──────────────────────────────────────────────────────────────────────┘
@_decoration_
def fn_step02_2_create_issue_modify_ratio(
        df_base : pd.DataFrame,          # ← step-01 결과 (ShipTo·Item·Loc unique)
        df_time : pd.DataFrame           # ← df_in_Time (COL_TIME_WK 포함)
) -> pd.DataFrame:
    """
    Returns
    -------
    pd.DataFrame
        df_output_Sell_In_Issue_Modify_GI_Ratio
        (빈 DF 스키마: ShipTo·Item·Loc·Week = PK, 9 measure = float32)
    """
    # logger.Note(f"[{dict_log['p_step_no']}] {dict_log['p_step_desc']}")    
    # (0) base DF 없으면 빈 스키마 DF 반환
    if df_base.empty:
        logger.Note("[Step-02-2] base DF empty → return empty issue-GI DF")
        return _empty_issue_gi_df()

    # (1) Cartesian join : (ShipTo,Item,Loc) × Week
    df_cross = (
        df_base.assign(_k=1)
               .merge(df_time[[COL_TIME_WK]].assign(_k=1), on="_k")
               .drop("_k", axis=1)
               .reset_index(drop=True)
    )

    # (2) Version / dtype / category 최적화
    # try:            # o9 환경: 플러그인에서 주입
    #     Version
    # except NameError:
    #     Version = 'CWV_DP'        # 로컬 기본값

    df_cross[COL_VERSION] = Version
    for col in [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC]:
        df_cross[col] = df_cross[col].astype("category")

    # (3) 9개 Measure 컬럼 float32 0.0 세팅
    for col in LIST_ISSUE_GI_MEAS:
        df_cross[col] = F_ZERO_FLOAT32
    df_cross = df_cross.astype({c: np.float32 for c in LIST_ISSUE_GI_MEAS})

    # (4) 컬럼 순서 맞춤
    col_order = ([COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_WK]
                 + LIST_ISSUE_GI_MEAS)
    df_cross = df_cross[col_order]

    logger.Note(f"[Step-02-2] issue-GI rows created = {len(df_cross):,}")
    return df_cross
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
            input_folder_name  = 'PYCreateDataNewSalesProductASNEstoreModify'
            output_folder_name = 'PYCreateDataNewSalesProductASNEstoreModify'

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
            # CurrentPartialWeek = '202506A'

        # vdLog 초기화
        log_path = os.path.dirname(__file__) if is_local else ""
        vdCommon.gfn_pyLog_start(Version, str_instance, logger, is_local, log_path)
        # --------------------------------------------------------------------------
        # df_input 체크 시작
        # --------------------------------------------------------------------------
        logger.Note(p_note='df_input 체크 시작', p_log_level=LOG_LEVEL.debug())
        fn_process_in_df_mst()
         # 입력 변수 중 데이터가 없는 경우 경고 메시지를 출력한다.
        for in_df in input_dataframes:
            fn_check_input_table(input_dataframes[in_df], in_df, '1')




        ################################################################################################################
        # Start of processing
        ################################################################################################################
       

        ################################################################################################################
        # Step 01-1 ─ 필터링 & 중복제거  ➜ df_fn_Sales_Product_ASN_Delta
        ################################################################################################################
        dict_log = {
            'p_step_no' : 101,
            'p_step_desc': '필터링 & 중복제거  ➜ df_fn_Sales_Product_ASN_Delta'
        }
        df_fn_Sales_Product_ASN_Delta = fn_step01_filter_asn_delta(**dict_log)
        fn_log_dataframe(df_fn_Sales_Product_ASN_Delta, f'step01_1_{STR_DF_FN_SALES_ASN_DELTA}')
        # 이후로직에서는 df_fn_Sales_Product_ASN_Delta 의 length를 체크한다.


        ############################################################################################################
        # Step 02-1  ·  S/In User Modify GI Ratio  ➜ df_output_Sell_In_User_Modify_GI_Ratio
        ############################################################################################################
        dict_log = {
            'p_step_no' : 201,
            'p_step_desc': 'User Modify GI Ratio 생성'
        }
        df_output_Sell_In_User_Modify_GI_Ratio = fn_step02_1_create_user_modify_ratio(
            df_fn_Sales_Product_ASN_Delta,             # Step-01 결과
            input_dataframes[STR_DF_IN_TIME],          # 주차 테이블
            **dict_log
        )
        # 로그 · CSV 저장
        fn_log_dataframe(df_output_Sell_In_User_Modify_GI_Ratio,
                        f"{STR_DF_OUT_USER_GI_RATIO}")

        ############################################################################################################
        # Step-02-2 · S/In Issue Modify GI Ratio  ➜ df_output_Sell_In_Issue_Modify_GI_Ratio
        ############################################################################################################
        dict_log = {
            'p_step_no' : 202,
            'p_step_desc': 'Issue Modify GI Ratio 생성'
        }
        df_output_Sell_In_Issue_Modify_GI_Ratio = fn_step02_2_create_issue_modify_ratio(
            df_fn_Sales_Product_ASN_Delta,            # Step-01 결과
            input_dataframes[STR_DF_IN_TIME],         # 주차 테이블 (COL_TIME_WK)
            **dict_log
        )

        fn_log_dataframe(df_output_Sell_In_Issue_Modify_GI_Ratio,
                        f"{STR_DF_OUT_ISSUE_GI_RATIO}")
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
