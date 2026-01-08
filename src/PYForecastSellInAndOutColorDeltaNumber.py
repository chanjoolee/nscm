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
str_instance = 'PYForecastSellInAndOutColorDeltaNumber'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"

is_print = True
flag_csv = True
flag_exception = True

########################################################################################################################
# 컬럼상수
########################################################################################################################
# ── column constants ──────────────────────────────────────────────────────────────────────────────────────────
COL_VERSION             = 'Version.[Version Name]'
COL_ITEM                = 'Item.[Item]'
COL_PT                  = 'Item.[ProductType]'  
COL_GBM                 = 'Item.[Item GBM]'
COL_PG                  = 'Item.[Product Group]'
COL_LV_CODE             = 'ShipTo_Level'
COL_RTS_EOS_SHIPTO      = 'RTS_EOS_ShipTo'
COL_SHIP_TO             = 'Sales Domain.[Ship To]'
COL_STD1                = 'Sales Domain.[Sales Std1]'
COL_STD2                = 'Sales Domain.[Sales Std2]'
COL_STD3                = 'Sales Domain.[Sales Std3]' 
COL_STD4                = 'Sales Domain.[Sales Std4]'
COL_STD5                = 'Sales Domain.[Sales Std5]'
COL_STD6                = 'Sales Domain.[Sales Std6]'


COL_LOC                         = 'Location.[Location]'
COL_CLASS                       = 'ITEMCLASS Class'
COL_TIME_PW                     = 'Time.[Partial Week]'
COL_SIN_FCST_GC_LOCK            = 'S/In FCST(GI)_GC_Lock'
COL_SIN_FCST_COLOR_COND         = 'S/In FCST Color Condition Number'
COL_SIN_FCST_AP2_LOCK           = 'S/In FCST(GI)_AP2_Lock'
COL_SIN_FCST_AP1_LOCK           = 'S/In FCST(GI)_AP1_Lock'
COL_LOCK_COND                   = 'Lock Condition'              # (=구 GC.Lock)

# Salse_Product_ASN       = 'Sales Product ASN'    
COL_ASN_FLAG                    = 'Sales Product ASN'
COL_TATTERM                     = 'ITEMTAT TATTERM'
COL_TAT_SET                     = 'ITEMTAT TATTERM_SET'
COL_RULE_GC                     = 'FORECAST_RULE GC FCST'
COL_RULE_AP2                    = 'FORECAST_RULE AP2 FCST'
COL_RULE_AP1                    = 'FORECAST_RULE AP1 FCST'   
COL_RULE_CUST                   = 'FORECAST_RULE CUST FCST'
COL_RULE_ISVALID                = 'FORECAST_RULE ISVALID'

COL_SOUT_FCST_GC_LOCK         = 'S/Out FCST_GC_Lock'
COL_SOUT_FCST_AP2_LOCK        = 'S/Out FCST_AP2_Lock'
COL_SOUT_FCST_AP1_LOCK        = 'S/Out FCST_AP1_Lock'
COL_SOUT_FCST_COLOR_COND      = 'S/Out FCST Color Condition Number'
COL_SOUT_FCST_NOT_EXISTS      = 'S/Out Fcst Not Exist Flag'


COL_CURRENT_ROW_WEEK         = 'ROW_WK_NORM'
COL_CURRENT_ROW_WEEK_PLUS_8  = 'CURRENTWEEK_NORMALIZED_PLUS_8'   
COL_RTS_INIT_DATE            = 'RTS_INIT_DATE'
COL_RTS_DEV_DATE             = 'RTS_DEV_DATE'
COL_RTS_COM_DATE             = 'RTS_COM_DATE'
COL_RTS_WEEK                 = 'RTS_WEEK_NORMALIZED'
COL_RTS_PARTIAL_WEEK         = 'RTS_PARTIAL_WEEK'
COL_RTS_INITIAL_WEEK         = 'RTS_INITIAL_WEEK_NORMALIZED'
COL_RTS_WEEK_MINUST_1        = 'RTS_WEEK_NORMALIZED_MINUST_1'
COL_RTS_WEEK_PLUS_3          = 'RTS_WEEK_NORMALIZED_PLUS_3'
COL_MAX_RTS_CURRENTWEEK      = 'MAX_RTS_CURRENTWEEK'
COL_RTS_ISVALID              = 'RTS_ISVALID'
COL_RTS_STATUS               = 'RTS_STATUS'
COL_EOS_INIT_DATE            = 'EOS_INIT_DATE'
COL_EOS_CHG_DATE             = 'EOS_CHG_DATE'
COL_EOS_COM_DATE             = 'EOS_COM_DATE'
COL_EOS_WEEK                 = 'EOS_WEEK_NORMALIZED'
COL_EOS_PARTIAL_WEEK         = 'EOS_PARTIAL_WEEK'
COL_EOS_WEEK_MINUS_1         = 'EOS_WEEK_NORMALIZED_MINUS_1'
COL_EOS_WEEK_MINUS_3         = 'EOS_WEEK_NORMALIZED_MINUS_3' # 추가 1125
COL_EOS_WEEK_MINUS_4         = 'EOS_WEEK_NORMALIZED_MINUS_4'
COL_EOS_INITIAL_WEEK         = 'EOS_INITIAL_WEEK_NORMALIZED'
COL_MIN_EOSINI_MAXWEEK       = 'MIN_EOSINI_MAXWEEK'
COL_MIN_EOS_MAXWEEK          = 'MIN_EOS_MAXWEEK'
COL_EOS_STATUS               = 'EOS_STATUS'
COL_HA_EOP_FLAG              = 'HA_EOP_FLAG'
COL_ESTOREACCOUNT            = 'EStoreAccount'

# 
COL_RTS_ISVALID_DELTA        = 'RTS_ISVALID Delta'
COL_RTS_STATUS_DELTA         = 'RTS_ISVALID Delta'
COL_RTS_INIT_DATE_DELTA      = 'RTS_INIT_DATE Delta'    
COL_RTS_DEV_DATE_DELTA       = 'RTS_DEV_DATE Delta'    
COL_RTS_COM_DATE_DELTA       = 'RTS_COM_DATE Delta'    
COL_EOS_STATUS_DELTA         = 'EOS_STATUS Delta'    
COL_EOS_INIT_DATE_DELTA      = 'EOS_INIT_DATE Delta'    
COL_EOS_CHG_DATE_DELTA       = 'EOS_CHG_DATE Delta'              
COL_EOS_COM_DATE_DELTA       = 'EOS_COM_DATE Delta'    

# Step 10-1 / 10-2 색상 상수
COLOR_LIGHTBLUE             = '0'
COLOR_LIGHTRED              = '1'
COLOR_YELLOW                = '2'
COLOR_GREEN                 = '3'
COLOR_WHITE                 = '4'
COLOR_DGRAY_REDB            = '5'
COLOR_DGRAY_RED             = '6'
COLOR_DARKBLUE              = '7'
COLOR_DARKRED               = '8'
COLOR_GRAY                  = '9'


# ───────────────────────────────────────────────────────────────
# CONSTANT STRING VARIABLES FOR DATAFRAME NAMES
# ───────────────────────────────────────────────────────────────
# input
# ───── I/O DataFrame handles (spec 이름 준수) ────
STR_DF_RTS_EOS            = 'df_in_MST_RTS_EOS'
STR_DF_RTS_EOS_DELTA      = 'df_in_MST_RTS_EOS_Delta'

# STR_DF_RULE               = 'df_in_Forecast_Rule'
STR_DF_DIM                = 'df_in_Sales_Domain_Dimension'
STR_DF_ASN                = 'df_in_Sales_Product_ASN'
STR_DF_TIME               = 'df_in_Time_Partial_Week'
STR_DF_ITEMTAT            = 'df_in_Item_TAT'
STR_DF_ITEMCLASS          = 'df_in_Item_CLASS'
STR_DF_ITEMMST            = 'df_in_Item_Master'
STR_DF_ESTORE             = 'df_in_Sales_Domain_Estore'
STR_DF_NO_SELL_OUT        = 'df_in_SELLOUTFCST_NOTEXIST'

# middle
STR_DF_FN_RTS_EOS           = 'df_fn_RTS_EOS'
STR_DF_FN_ASN_PW            = 'df_fn_ASN_Group_Week' # Changed  From STR_DF_FN_RTS_EOS_WK =  'df_fn_RTS_EOS_Week'
STR_DF_FN_ASN_ITEM          = 'df_fn_Sales_Product_ASN_Item'
STR_DF_FN_ASN_GROUP         = 'df_fn_Sales_Product_ASN_Group'
# STR_DF_FN_ASN_ITEM_WK       = 'df_fn_Sales_Product_ASN_Item_Week'

# output
STR_DF_OUT_SIN              = 'df_output_Sell_In_FCST_Color_Condition'
STR_DF_OUT_SOUT             = 'df_output_Sell_Out_FCST_Color_Condition'
STR_DF_OUT_RTS_EOS          = 'df_output_RTS_EOS'



########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################
logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
# fn_set_local_logfile()
LOG_LEVEL = common.G_log_level

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
        logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=20)
        # if is_local and not df_p_source.empty and flag_csv:
        if is_local and flag_csv:
            # 로컬 Debugging 시 csv 파일 출력
            df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)
    else:
        # 최종 Output 테이블인 경우에는 무조건 로그 출력
        if is_output:
            logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=20)
            if is_local and not df_p_source.empty:
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
        # logger.Note(p_note=f'Call gfn_pyLog_detail', p_log_level=LOG_LEVEL.debug())
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

def fn_convert_type(df: pd.DataFrame, startWith: str, target_type):
    """
    특정 prefix 로 시작하는 컬럼들을 지정된 타입으로 변환.
    - target_type 이 str 인 경우:
        • float64 / datetime64 / category 상관없이
        • pandas의 nullable string dtype("string") 으로 변환
        • NaN / NaT 는 <NA> 로 유지 (문자열 'nan' 으로 바뀌지 않음)
    - target_type 이 'bool' 또는 bool 인 경우:
        • pandas nullable boolean("boolean") 으로 변환
    - 그 외:
        • 기존처럼 astype(target_type, errors='ignore')
    """
    for column in df.columns:
        if not column.startswith(startWith):
            continue

        # ---- 문자열로 변환 (nullable string) ----
        if target_type is str:
            # 이미 string / category / datetime / float 모두 OK
            # -> "string" 으로 변환하면 NaN/NaT 가 <NA>로 유지됨
            try:
                df[column] = df[column].astype("string")
            except TypeError:
                # 혹시 extension array 등에서 문제가 나면 한 번 감싸서 재시도
                df[column] = pd.Series(df[column], copy=False).astype("string")

        # ---- bool 처리 (nullable boolean) ----
        elif target_type == "bool" or target_type is bool:
            # 문자열 'True'/'False', 1/0 등 들어오는 경우도 있으니
            # 우선 pandas 가 알아서 처리하도록 하고, NaN 은 <NA> 유지
            df[column] = df[column].astype("boolean")

        # ---- 그 외 타입은 기존 로직 유지 ----
        else:
            df[column] = df[column].astype(target_type, errors="ignore")

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

################################################################################################################──────────
#  공통 타입 변환  (❌ `global` 사용 금지)
#  호출 측에서 `input_dataframes` 를 인자로 넘겨준다.
################################################################################################################──────────
def fn_prepare_input_types(dict_dfs: dict) -> None:
    """
    dict_dfs :  { <df_name> : pandas.DataFrame, ... }

    • object → string(nullable) → category      (숫자·문자 혼재 대비)
      (NaN/None 는 <NA> 로 유지됨)
    • string(dtype) → category
    • float/int → fillna(0) → int32             (값이 실수면 round 후 변환)

    **주의** : dict 내부의 DataFrame 을 *제자리*에서 변환하므로 반환값은 없다.
    """
    if not dict_dfs:
        return

    for df_name, df in dict_dfs.items():
        if df is None or df.empty:
            continue

        # 1) pandas StringDtype 컬럼 : 이미 nullable string 이므로 바로 category 로
        str_cols = df.select_dtypes(include=["string"]).columns
        for col in str_cols:
            # 여기서는 astype("string") 절대 사용 금지
            df[col] = df[col].astype("category")

        # 2) object 컬럼 : nullable string 을 거쳐 category 로
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            # astype("string") 은 NaN/None → <NA> 로 유지해 줌
            df[col] = df[col].astype("string").astype("category")

        # 3) numeric 컬럼 : 기존 로직 유지
        num_cols = df.select_dtypes(
            include=["float64", "float32", "int64", "int32", "int"]
        ).columns
        for col in num_cols:
            df[col].fillna(0, inplace=True)
            try:
                df[col] = df[col].astype("int32")
            except ValueError:
                df[col] = df[col].round().astype("int32")


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
            'df_in_Sales_Domain_Dimension.csv'          :      STR_DF_DIM           ,
            'df_in_Time_Partial_Week.csv'               :      STR_DF_TIME          ,
            'df_in_Item_CLASS.csv'                      :      STR_DF_ITEMCLASS     ,
            'df_in_Item_TAT.csv'                        :      STR_DF_ITEMTAT       ,
            'df_in_MST_RTS_EOS_Delta.csv'               :      STR_DF_RTS_EOS_DELTA ,
            'df_in_MST_RTS_EOS.csv'                     :      STR_DF_RTS_EOS ,
            'df_in_Sales_Product_ASN.csv'               :      STR_DF_ASN           ,
            'df_in_Item_Master.csv'                     :      STR_DF_ITEMMST       ,
            'df_in_SELLOUTFCST_NOTEXIST.csv'            :      STR_DF_NO_SELL_OUT            
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
                if file_name == keyword.split('.')[0]:
                    input_dataframes[frame_name] = df
                    mapped = True
                    break
    else:
        # o9 에서 
        input_dataframes[STR_DF_DIM]            = df_in_Sales_Domain_Dimension
        # input_dataframes[STR_DF_ESTORE]         = df_in_Sales_Domain_Estore
        input_dataframes[STR_DF_TIME]           = df_in_Time_Partial_Week
        input_dataframes[STR_DF_ITEMCLASS]      = df_in_Item_CLASS
        input_dataframes[STR_DF_ITEMTAT]        = df_in_Item_TAT
        input_dataframes[STR_DF_RTS_EOS_DELTA]  = df_in_MST_RTS_EOS_Delta
        input_dataframes[STR_DF_RTS_EOS]        = df_in_MST_RTS_EOS
        input_dataframes[STR_DF_ASN]            = df_in_Sales_Product_ASN
        # input_dataframes[STR_DF_RULE]           = df_in_Forecast_Rule
        input_dataframes[STR_DF_ITEMMST]        = df_in_Item_Master
        input_dataframes[STR_DF_NO_SELL_OUT]    = df_in_SELLOUTFCST_NOTEXIST

    # type convert : str ==> category, int ==> int32
    # df_in_Sales_Domain_Dimension
    fn_convert_type(input_dataframes[STR_DF_DIM], 'Sales Domain', str)    

    # df_in_Sales_Domain_Estore
    # fn_convert_type(input_dataframes[STR_DF_ESTORE], 'Sales Domain', str)

    # df_in_Time_Partial_Week
    fn_convert_type(input_dataframes[STR_DF_TIME], COL_TIME_PW , str)

    # df_in_Item_CLASS
    fn_convert_type(input_dataframes[STR_DF_ITEMCLASS], COL_VERSION , str)
    fn_convert_type(input_dataframes[STR_DF_ITEMCLASS], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_ITEMCLASS], 'Location', str)
    fn_convert_type(input_dataframes[STR_DF_ITEMCLASS], 'Item', str)
    fn_convert_type(input_dataframes[STR_DF_ITEMCLASS], 'ITEMCLASS', str)
    
    # df_in_Item_TAT
    fn_convert_type(input_dataframes[STR_DF_ITEMTAT], COL_VERSION , str)
    fn_convert_type(input_dataframes[STR_DF_ITEMTAT], COL_ITEM, str)
    fn_convert_type(input_dataframes[STR_DF_ITEMTAT], COL_LOC, str)

    # df_in_MST_RTS_EOS Delta
    fn_convert_type(input_dataframes[STR_DF_RTS_EOS_DELTA], COL_VERSION , str)
    fn_convert_type(input_dataframes[STR_DF_RTS_EOS_DELTA], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_RTS_EOS_DELTA], COL_ITEM , str)
    fn_convert_type(input_dataframes[STR_DF_RTS_EOS_DELTA], 'RTS_' , str)
    fn_convert_type(input_dataframes[STR_DF_RTS_EOS_DELTA], 'EOS_' , str)

    
    # df_in_MST_RTS_EOS
    fn_convert_type(input_dataframes[STR_DF_RTS_EOS], COL_VERSION , str)
    fn_convert_type(input_dataframes[STR_DF_RTS_EOS], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_RTS_EOS], COL_ITEM , str)
    fn_convert_type(input_dataframes[STR_DF_RTS_EOS], 'RTS_' , str)
    fn_convert_type(input_dataframes[STR_DF_RTS_EOS], 'EOS_' , str)

    # df_in_Sales_Product_ASN
    fn_convert_type(input_dataframes[STR_DF_ASN], COL_VERSION , str)
    fn_convert_type(input_dataframes[STR_DF_ASN], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_ASN], COL_ITEM, str)
    fn_convert_type(input_dataframes[STR_DF_ASN], COL_LOC, str)
    fn_convert_type(input_dataframes[STR_DF_ASN], COL_ASN_FLAG, str)


    # df_in_SELLOUTFCST_NOTEXIST
    fn_convert_type(input_dataframes[STR_DF_NO_SELL_OUT], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_NO_SELL_OUT], COL_SOUT_FCST_NOT_EXISTS, 'bool')

    fn_prepare_input_types(input_dataframes)

def analyze_by_rbql() :
    asn_df = input_dataframes[STR_DF_ASN]
    master_df = input_dataframes[STR_DF_ITEMMST]

    my_quey = f"""
        select 
            a['{COL_SHIP_TO}'] as shipto,
            a['{COL_ITEM}'] as item,
            a['{COL_LOC}'] as location,
            b['{COL_PT}'] as item_type,
            b['{COL_GBM}'] as item_gbm,
            b['{COL_PG}'] as product_group
        Join b on a['{COL_ITEM}'] == b['{COL_ITEM}']
        where b['{COL_PT}'] == 'BAS'

    """

    result = rbql.query_pandas_dataframe(
        query_text=my_quey,
        input_dataframe=asn_df,
        join_dataframe=master_df
    )


def analyze_by_duckdb():
    # import duckdb
    # Retrieve your DataFrames
    asn_df    = input_dataframes[STR_DF_ASN]
    master_df = input_dataframes[STR_DF_ITEMMST]
    dim_df    = input_dataframes[STR_DF_DIM]      

    # Register each DataFrame as a DuckDB table
    duckdb.register('asn_table', asn_df)
    duckdb.register('master_table', master_df)
    duckdb.register('dim_table', dim_df)

    # Build a SQL query referencing them by table aliases a (asn_table) and b (master_table)
    my_query = f"""
    SELECT
        a['{COL_SHIP_TO}'] AS shipto,
        c['{COL_STD1}'] AS DomainLv2,
        a['{COL_ITEM}']          AS item,
        a['{COL_LOC}']  AS location,
        b['{COL_PT}']          AS item_type,
        b['{COL_GBM}']          AS item_gbm,
        b['{COL_PG}']      AS product_group
    FROM asn_table AS a
    JOIN master_table AS b
      ON a['{COL_ITEM}'] == b['{COL_ITEM}']
    JOIN dim_table AS c
        ON a['{COL_SHIP_TO}'] == c['{COL_SHIP_TO}']
    WHERE b['{COL_PT}']  = 'BAS'
    """

    # Execute the DuckDB query in-memory and fetch as a pandas DataFrame
    result_df = duckdb.query(my_query).to_df()

    return result_df


def analyze_by_duckdb_from_output():

    # from re import X
    # import os,sys,json,shutil,io,zipfile
    # import time
    # import datetime
    # import inspect
    # import traceback
    # import pandas as pd
    # from NSCMCommon import NSCMCommon as common
    # # from typing_extensions import Literal
    # import glob
    # import numpy as np
    # # import rbql
    # import duckdb

    v_base_dir  = "C:\workspace\Output\PYForecastMeasureLockColor_SHA_REF_20250410_14_08"
    v_output_dir  = f"{v_output_dir}/output"
    v_input_dir = f"{v_base_dir}/input"
    
    def read_csv_with_fallback(filepath):
        encodings = ['utf-8-sig', 'utf-8', 'cp949']
        
        for enc in encodings:
            try:
                return pd.read_csv(filepath, encoding=enc)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Unable to read file {filepath} with tried encodings.")

    input_pattern = f"{v_input_dir}/*.csv"
    input_csv_files = glob.glob(input_pattern)
    for file in input_csv_files:
        file_name = file.split("/")[-1].split("\\")[-1].split(".")[0]
        df = read_csv_with_fallback(file)
        duckdb.register(file_name, df)

    output_pattern = f"{v_output_dir}/*.csv"
    output_csv_files = glob.glob(output_pattern)
    for file in output_csv_files:
        file_name = file.split("/")[-1].split("\\")[-1].split(".")[0]
        df = read_csv_with_fallback(file)
        duckdb.register(file_name, df)

    # # Retrieve your DataFrames]
    # file = f"{v_output_dir}/input/df_in_Sales_Product_ASN"
    # df = read_csv_with_fallback(file)
    # duckdb.register(str_df_in_Sales_Product_ASN, df)
    my_query = f"""
    SELECT
        a['{COL_SHIP_TO}'] AS shipto,
        c['{COL_STD1}'] AS DomainLv2,
        a['{COL_ITEM}']          AS item,
        a['{COL_LOC}']  AS location,
        b['{COL_PT}']          AS item_type,
        b['{COL_GBM}']          AS item_gbm,
        b['{COL_PG}']      AS product_group
    FROM asn_table AS a
    JOIN master_table AS b
      ON a['{COL_ITEM}'] == b['{COL_ITEM}']
    JOIN dim_table AS c
        ON a['{COL_SHIP_TO}'] == c['{COL_SHIP_TO}']
    WHERE b['{COL_PT}']  = 'BAS'
    """

    # Execute the DuckDB query in-memory and fetch as a pandas DataFrame
    result_df = duckdb.query(my_query).to_df()

    return result_df




# 1. Sanitize date string
# def sanitize_date_string(x):
#     if pd.isna(x):
#         return ''
#     x = str(x).strip()
#     for token in ['PM', 'AM', '오전', '오후']:
#         if token in x:
#             x = x.split(token)[0].strip()
#     return x[:10]  # Keep only 'YYYY/MM/DD'

def sanitize_date_string(x: object) -> object:
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
        ④ 실패 시 <NA> 리턴
    """
    # ① 결측치는 그대로 NA 로
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()
    if s == '':
        return pd.NA

    # ② 공백(혹은 T) 이후 time 문자열 제거
    s = re.split(r'\s+|T', s, maxsplit=1)[0]

    # ③ 구분자 통일
    s = s.replace('-', '/')

    # ④ 날짜 포맷 판별·정규화
    parts = s.split('/')
    try:
        if len(parts) == 3:
            # case-A : YYYY/MM/DD
            if len(parts[0]) == 4:
                y, m, d = parts
            # case-B : M/D/YYYY  또는  MM/DD/YYYY
            else:
                m, d, y = parts
            dt_obj = datetime.datetime(int(y), int(m), int(d))  # 유효성 체크
            return dt_obj.strftime('%Y/%m/%d')  # zero-padding 포함
    except Exception:
        pass

    # ⑤ 파싱 실패도 NA 로
    return pd.NA


# 벡터라이즈 버전 (결과는 object 이지만 안에 값은 str 또는 pd.NA)
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
        return pd.NA
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


# ──────────────────────────────────────────────────────────────────────────────
# STEP-00 : Ship-To Dimension LUT
#           (LV 코드 + 6 개 Std 컬럼 → 빠른 dict lookup 용)
# ──────────────────────────────────────────────────────────────────────────────
# * 2 → LV2(Std1) … 7 → LV7(Std6)
# * 메모리 절감 : category 캐스팅 & 중간 ndarray 즉시 해제
# * 반환 DF 컬럼
#     ─ COL_SHIP_TO  (PK)
#     ─ COL_STD1 … COL_STD6
#     ─ COL_LV_CODE  (int8)
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def step00_load_shipto_dimension(
        df_dim: pd.DataFrame
) -> pd.DataFrame:
    """
    Ship-To ⇢ LV_CODE & 6×Std 컬럼 LUT 생성.    Parameters
    ----------
    df_dim : pd.DataFrame
        `df_in_Sales_Domain_Dimension` 원본.

    Returns
    -------
    pd.DataFrame
        최소 컬럼 + category 캐스팅 + LV_CODE.
    """
    
    # ── 0) 필요한 컬럼만 선택 & category 캐스팅 ─────────────────────────────
    USE_COLS = [
        COL_SHIP_TO,
        COL_STD1, COL_STD2, COL_STD3,
        COL_STD4, COL_STD5, COL_STD6
    ]
    dim = (
        df_dim[USE_COLS]
        .copy(deep=False)
        .astype({c: 'category' for c in USE_COLS})
    )

    # ── 1) LV_CODE 계산 (벡터라이즈) ──────────────────────────────────────
    #    기본값 LV2; 뒤에서 매칭되면 overwrite
    lv_code = np.full(len(dim), 2, dtype='int8')
    ship_np = dim[COL_SHIP_TO].to_numpy()

    # (Std6 → LV7) 역순으로 매칭하여 “가장 하위” 코드가 남도록
    for col, lv in [
        (COL_STD6, 7),
        (COL_STD5, 6),
        (COL_STD4, 5),
        (COL_STD3, 4),
        (COL_STD2, 3),
        (COL_STD1, 2),
    ]:
        match = dim[col].to_numpy() == ship_np
        lv_code[match] = lv
        # del match                                           # 메모리 즉시 해제
        # gc.collect()

    dim[COL_LV_CODE] = lv_code

    # ── 2) 후처리 & 메모리 정리 ──────────────────────────────────────────
    del ship_np, lv_code, df_dim
    gc.collect()

    return dim


# ── 전처리 : 주차 카테고리를 'YYYYWW' 숫자순으로 정렬 & ordered 지정 ──
def _order_week_cat(sr_cat: pd.Series) -> pd.Series:
    if sr_cat.dtype.name != 'category' or sr_cat.cat.ordered:
        return sr_cat            # 이미 ordered 면 그대로
    cats = sr_cat.cat.categories.astype("string")
    # '202447' 같은 숫자부로 sort → category 재생성
    ordered_cats = pd.Series(cats).sort_values(key=lambda s: s.astype(int)).to_numpy()
    return sr_cat.cat.set_categories(ordered_cats, ordered=True)
    


################################################################################################################
#################### Start Step Functions  ##########
################################################################################################################

# ── 필요 상수 예시 (이미 상단에 선언돼 있다고 가정) ─────────────────
# COL_VERSION = 'Version.[Version Name]'
# COL_ITEM    = 'Item.[Item]'
# COL_SHIP_TO = 'Sales Domain.[Ship To]'
# COL_RTS_ISVALID   = 'RTS_ISVALID'
# COL_RTS_STATUS    = 'RTS_STATUS'
# COL_RTS_COM_DATE  = 'RTS_COM_DATE'
# COL_RTS_DEV_DATE  = 'RTS_DEV_DATE'
# COL_RTS_INIT_DATE = 'RTS_INIT_DATE'
# COL_EOS_STATUS    = 'EOS_STATUS'
# COL_EOS_CHG_DATE  = 'EOS_CHG_DATE'
# COL_EOS_COM_DATE  = 'EOS_COM_DATE'
# COL_EOS_INIT_DATE = 'EOS_INIT_DATE'


################################################################################################################
# Step 1-0) df_output_RTS_EOS_Delta 생성 (2025.12.19 Delta Null 처리 로직 추가)
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────
# STEP 01-0 : Output_RTS_EOS_Delta 생성
#   - df_in_MST_RTS_EOS_Delta 를 copy
#   - (Version, Item, Ship To) 제외 Measure 전부 Null 처리
# ──────────────────────────────────────────────────────────────────────────
@_decoration_
def step01_0_build_output_rts_eos_delta(
        df_in_rts_eos_delta: pd.DataFrame,   # df_in_MST_RTS_EOS_Delta
        out_version: str,
        **kwargs
) -> pd.DataFrame:
    """
    Output_RTS_EOS_Delta
      - input(df_in_MST_RTS_EOS_Delta) 를 그대로 복사
      - Version/Item/ShipTo 를 제외한 모든 Measure 컬럼을 pd.NA 로 Null 처리
      - 컬럼명(Delta suffix 포함)은 유지
    """
    # 빈 입력 방어
    if df_in_rts_eos_delta is None or df_in_rts_eos_delta.empty:
        return pd.DataFrame(columns=[
            COL_VERSION, COL_ITEM, COL_SHIP_TO,
            COL_RTS_ISVALID_DELTA,
            COL_RTS_STATUS_DELTA, COL_RTS_INIT_DATE_DELTA, COL_RTS_DEV_DATE_DELTA, COL_RTS_COM_DATE_DELTA,
            COL_EOS_STATUS_DELTA, COL_EOS_INIT_DATE_DELTA, COL_EOS_CHG_DATE_DELTA, COL_EOS_COM_DATE_DELTA,
        ])

    # 0) Copy (가벼운 복사)
    df_out = df_in_rts_eos_delta.copy()

    # 1) Version 강제 세팅(입력에 Version이 있어도 out_version으로 덮어씀: 스펙 Output 보존 목적)
    df_out[COL_VERSION] = out_version

    # 2) (Version, Item, ShipTo) 제외 전부 Null 처리
    keep_cols = {COL_VERSION, COL_ITEM, COL_SHIP_TO}
    null_cols = [c for c in df_out.columns if c not in keep_cols]
    for i, c in enumerate(df_out.columns):
        if c in null_cols:
            df_out.isetitem(i, pd.NA)

    # 3) dtype 정리(최소한의 카테고리 보장)
    df_out = df_out.astype({
        COL_VERSION: "category",
        COL_ITEM: "category",
        COL_SHIP_TO: "category",
    }, errors="ignore")

    gc.collect()
    return df_out




# =====================================================
# STEP 01 — RTS/EOS Delta 전처리 (Global 사용 없음)
#   • df_fn_RTS_EOS       : 이후 Step02 투입용(Version, RTS_ISVALID 제거)
#   • df_output_RTS_EOS   : 스펙 Output(3) 보존본(Version 포함)
#   • Delta 가 비어있는 RTS/EOS 블록은 df_in_MST_RTS_EOS 에서 보완
# =====================================================
@_decoration_
def step01_preprocess_rts_eos_delta(
    df_rts_delta: pd.DataFrame,
    df_rts_full: pd.DataFrame,          # 🔹 신규: RTS/EOS 전체 정보. 2025.12.09
    version: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Delta 입력(df_in_MST_RTS_EOS_Delta)을 전처리한다.
      1) ' Delta' 접미사 제거(Measure 이름 정규화)
      2) Output(3)용 원본 사본 df_output_RTS_EOS 생성 (Version 포함)
      3) 작업용 df_fn_RTS_EOS 생성 (Version, RTS_ISVALID 제거)

    Returns
    -------
    (df_fn_RTS_EOS, df_output_RTS_EOS)
    """

    if df_rts_delta.empty:
        # 빈 입력은 그대로 두 개의 빈 DF 반환
        empty_cols = [
            COL_VERSION, COL_ITEM, COL_SHIP_TO,
            COL_RTS_STATUS, COL_RTS_INIT_DATE, COL_RTS_DEV_DATE, COL_RTS_COM_DATE,
            COL_EOS_STATUS, COL_EOS_INIT_DATE, COL_EOS_CHG_DATE, COL_EOS_COM_DATE,
            COL_RTS_ISVALID
        ]
        df_empty = pd.DataFrame(columns=empty_cols)
        return df_empty.drop(columns=[COL_VERSION, COL_RTS_ISVALID], errors='ignore'), df_empty

    # 1) 대상 Delta 컬럼 집합
    delta_bases = {
        'RTS_ISVALID'  : COL_RTS_ISVALID,
        'RTS_STATUS'   : COL_RTS_STATUS,
        'RTS_COM_DATE' : COL_RTS_COM_DATE,
        'RTS_DEV_DATE' : COL_RTS_DEV_DATE,
        'RTS_INIT_DATE': COL_RTS_INIT_DATE,
        'EOS_STATUS'   : COL_EOS_STATUS,
        'EOS_CHG_DATE' : COL_EOS_CHG_DATE,
        'EOS_COM_DATE' : COL_EOS_COM_DATE,
        'EOS_INIT_DATE': COL_EOS_INIT_DATE,
    }

    # 1-1) 사용할 컬럼만 슬림화 (+존재하는 Delta 컬럼만 선별)
    use_cols = [c for c in (COL_VERSION, COL_ITEM, COL_SHIP_TO) if c in df_rts_delta.columns]
    for base in delta_bases.keys():
        colname = f"{base} Delta"
        if colname in df_rts_delta.columns:
            use_cols.append(colname)

    df_use = df_rts_delta.loc[:, use_cols].copy(deep=False)

    # 2) ' Delta' 접미사 제거 rename → 정규화된 컬럼명으로
    rename_map = {f"{k} Delta": v for k, v in delta_bases.items() if f"{k} Delta" in df_use.columns}
    df_norm = df_use.rename(columns=rename_map)

    # ─────────────────────────────
    # 2-1) 원본(df_rts_full) 기반으로 RTS/EOS 블록 보완
    #       · 키: (Item, ShipTo)
    #       · RTS_STATUS 가 NaN 인 행  → RTS 블록 전체를 full 값으로 채움
    #       · EOS_STATUS 가 NaN 인 행  → EOS 블록 전체를 full 값으로 채움
    # ─────────────────────────────
    if df_rts_full is not None and not df_rts_full.empty:
        # 원본에서 필요한 컬럼만 사용
        base_cols = [COL_ITEM, COL_SHIP_TO,
                     COL_RTS_STATUS, COL_RTS_INIT_DATE, COL_RTS_DEV_DATE, COL_RTS_COM_DATE,
                     COL_EOS_STATUS, COL_EOS_INIT_DATE, COL_EOS_CHG_DATE, COL_EOS_COM_DATE]
        base_cols = [c for c in base_cols if c in df_rts_full.columns]

        df_base = df_rts_full.loc[:, base_cols].copy(deep=False)

        # Delta(=df_norm) 와 Full(=df_base) 를 Item×ShipTo 로 머지
        df_merged = df_norm.merge(
            df_base,
            on=[COL_ITEM, COL_SHIP_TO],
            how='left',
            suffixes=('', '_BASE')
        )

        # ---- ① 타입 정리 : 문자열/날짜/카테고리 → nullable string 로 통일 ----
        fill_cols = [
            COL_RTS_STATUS, COL_RTS_INIT_DATE, COL_RTS_DEV_DATE, COL_RTS_COM_DATE,
            COL_EOS_STATUS, COL_EOS_INIT_DATE, COL_EOS_CHG_DATE, COL_EOS_COM_DATE,
        ]
        for col in fill_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].astype("string")
            base_col = f"{col}_BASE"
            if base_col in df_merged.columns:
                df_merged[base_col] = df_merged[base_col].astype("string")

        # ---- ② RTS 블록 보완 : RTS_STATUS 가 비어있으면 RTS_* 전부 base 에서 복사 ----
        if COL_RTS_STATUS in df_merged.columns:
            mask_rts_null = df_merged[COL_RTS_STATUS].isna()
            rts_cols = [
                COL_RTS_STATUS,
                COL_RTS_INIT_DATE,
                COL_RTS_DEV_DATE,
                COL_RTS_COM_DATE,
            ]
            for col in rts_cols:
                base_col = f"{col}_BASE"
                if col in df_merged.columns and base_col in df_merged.columns:
                    cur  = df_merged[col]
                    base = df_merged[base_col]
                    # NaN 인 곳만 base 로 대체
                    df_merged[col] = cur.where(~mask_rts_null, base)

        # ---- ③ EOS 블록 보완 : EOS_STATUS 가 비어있으면 EOS_* 전부 base 에서 복사 ----
        if COL_EOS_STATUS in df_merged.columns:
            mask_eos_null = df_merged[COL_EOS_STATUS].isna()
            eos_cols = [
                COL_EOS_STATUS,
                COL_EOS_INIT_DATE,
                COL_EOS_CHG_DATE,
                COL_EOS_COM_DATE,
            ]
            for col in eos_cols:
                base_col = f"{col}_BASE"
                if col in df_merged.columns and base_col in df_merged.columns:
                    cur  = df_merged[col]
                    base = df_merged[base_col]
                    df_merged[col] = cur.where(~mask_eos_null, base)

        # ---- ④ *_BASE 컬럼 제거 ----
        drop_base_cols = [c for c in df_merged.columns if c.endswith("_BASE")]
        df_work = df_merged.drop(columns=drop_base_cols)
    else:
        # df_in_MST_RTS_EOS 가 없으면 Delta 정보만 사용
        df_work = df_norm



    # =====================================================================
    # 3) Output(3)용 DF 구성 (Version 포함, 스펙 보존용)
    # =====================================================================
    df_output_RTS_EOS = df_work.copy(deep=False)

    if COL_VERSION in df_output_RTS_EOS.columns:
        df_output_RTS_EOS[COL_VERSION] = pd.Categorical.from_codes(
            np.zeros(len(df_output_RTS_EOS), dtype='int8'),
            categories=[version]
        )
    else:
        # 입력에 Version이 없으면 새로 삽입
        ver_cat = pd.Categorical.from_codes(np.zeros(len(df_output_RTS_EOS), dtype='int8'), categories=[version])
        df_output_RTS_EOS.insert(0, COL_VERSION, ver_cat)

    # 4) 작업용 DF 구성 : Version, RTS_ISVALID 제거
    drop_cols = [COL_VERSION, COL_RTS_ISVALID]
    df_fn_RTS_EOS = df_work.drop(columns=[c for c in drop_cols if c in df_norm.columns], errors='ignore')

    # 5) dtype 최적화 (메모리 절감)
    for tgt in (df_fn_RTS_EOS, df_output_RTS_EOS):
        if COL_ITEM in tgt.columns:
            tgt[COL_ITEM] = tgt[COL_ITEM].astype('category')
        if COL_SHIP_TO in tgt.columns:
            tgt[COL_SHIP_TO] = tgt[COL_SHIP_TO].astype('category')
        for st_col in (COL_RTS_STATUS, COL_EOS_STATUS):
            if st_col in tgt.columns:
                tgt[st_col] = tgt[st_col].astype('category')

        fn_convert_type(tgt, 'RTS_' , 'category')
        fn_convert_type(tgt, 'EOS_' , 'category')

    # 컬럼 순서 정리(가독성)
    ordered = [
        COL_ITEM, COL_SHIP_TO,
        COL_RTS_STATUS, COL_RTS_INIT_DATE, COL_RTS_DEV_DATE, COL_RTS_COM_DATE,
        COL_EOS_STATUS, COL_EOS_INIT_DATE, COL_EOS_CHG_DATE, COL_EOS_COM_DATE,
    ]
    df_fn_RTS_EOS = df_fn_RTS_EOS.reindex(columns=[c for c in ordered if c in df_fn_RTS_EOS.columns])

    out_cols = [COL_VERSION] + [c for c in ordered if c in df_output_RTS_EOS.columns]
    df_output_RTS_EOS = df_output_RTS_EOS.reindex(columns=out_cols)

    return df_fn_RTS_EOS, df_output_RTS_EOS


################################################################################################################
# Step 2 : Step1의 Result에 Time을 Partial Week 으로 변환
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────────
# STEP-02 : convert date-columns → Partial-Week
#            + 주차 계산 Helper 컬럼 생성
# ──────────────────────────────────────────────────────────────────────────────
# 입력  :  Step-01 결과 DF  (STR_DF_FN_RTS_EOS)
# 반환  :  동일 DF (날짜‧Helper 컬럼 반영)
# 사용  :  to_partial_week_datetime  /  common.gfn_add_week 벡터버전
# ──────────────────────────────────────────────────────────────────────────────
# ── Safe partial-week vectoriser ──────────────────────────────────────────────
def _to_pw_or_nan(x):
    """
    • 정상 →  'YYYYWWS' (A/B 포함)
    • 빈칸/NaN/파싱불가 → np.nan     (pandas 에서 결측치로 인식)
    """
    if x is None or (isinstance(x, str) and not x.strip()) or pd.isna(x):
        return np.nan
    try:
        return to_partial_week_datetime(x)
    except Exception:          # 파싱 실패 → NaN
        return np.nan

v_to_partial_week = np.vectorize(_to_pw_or_nan, otypes=[object])

# -------------- 0) 공통 헬퍼 --------------
@functools.lru_cache(maxsize=None)
def _pw_cached(date_str: str) -> str:
    """날짜 → Partial-Week (캐싱) 1개만 변환"""
    return to_partial_week_datetime(date_str)
    
def _series_to_pw(s: pd.Series) -> pd.Series:
    """중복 제거 + 캐시 변환 + NA 는 그대로 유지"""
    # ① 문자열 dtype으로 정규화 (pd.NA 유지)
    s_str = s.astype('string')

    # ② NA 제외하고 유니크 값만 LUT 생성
    uniq = s_str.dropna().unique()
    lut = {u: _pw_cached(u) for u in uniq}

    # ③ NA 는 na_action='ignore' 로 그대로 두고, 나머지만 map
    out = s_str.map(lut, na_action='ignore')

    # 혹시 _pw_cached 가 '' 를 돌려줄 수 있다면, 여기서 한 번 더 방어해도 됨
    # out = out.replace('', pd.NA)

    # 🔴 여기서 float64 NaN → string <NA> 로 강제 변환
    out = out.astype('string')


    return out

@functools.lru_cache(maxsize=None)
def _add_week_cached(yyyyww: str, offset: int) -> str:
    """YYYYWW  + n  → YYYYWW (A/B 없는 형태) 캐싱"""
    if not yyyyww:                # '', NaN -> ''
        return ''
    try:
        return common.gfn_add_week(yyyyww, offset)
    except Exception:
        return ''

def _bulk_add_week(base_series: pd.Series, offset: int) -> pd.Series:
    """
    · base_series: 'YYYYWW' 문자열/카테고리 (NaN 포함)
    · offset     : ±n
    캐시-사전으로 한 번에 치환
    """
    base_str = base_series.astype('string')      # NaN → <NA>
    uniq     = base_str.dropna().unique()        # 중복 제거
    lut = {u: _add_week_cached(u, offset) for u in uniq}
    out = base_str.map(lut, na_action='ignore')

    # 🔴 여기서 float64 NaN → string <NA> 로 강제 변환
    out = out.astype('string')
    return out


@_decoration_
def step02_convert_date_to_partial_week(
        df_rts: pd.DataFrame,
        current_week: str
) -> pd.DataFrame:
    """
    RTS/EOS 각 날짜 컬럼을 ‘YYYYWWS’(A/B 포함) 로 변환하고,
    후속 Step 에서 쓰일 Helper 컬럼을 한 번에 생성한다.
    """
    tgt = df_rts.copy()                         # 안전 사본    
    # 1) 날짜 → Partial-Week
    date_cols = [
        COL_RTS_INIT_DATE, COL_RTS_DEV_DATE, COL_RTS_COM_DATE,
        COL_EOS_INIT_DATE, COL_EOS_CHG_DATE, COL_EOS_COM_DATE
    ]
    for col in date_cols:
        tgt[col] = v_sanitize_date_string(tgt[col].to_numpy())
        tgt[col] = _series_to_pw(tgt[col])


    # 2) RTS/EOS 기준 Partial-Week 산출 (기존 그대로)
    tgt[COL_RTS_PARTIAL_WEEK] = np.where(
        tgt[COL_RTS_STATUS] == 'COM',
        tgt[COL_RTS_COM_DATE],
        np.where(
            tgt[COL_RTS_DEV_DATE].notna(),
            tgt[COL_RTS_DEV_DATE],
            tgt[COL_RTS_INIT_DATE]
        )
    )
    
    # EOS :   INI → CHG_DATE  /  그외 → COM_DATE
    tgt[COL_EOS_PARTIAL_WEEK] = np.where(
        tgt[COL_EOS_STATUS] == 'COM',
        np.where(
            tgt[COL_EOS_COM_DATE].notna(),
            tgt[COL_EOS_COM_DATE],
            np.where(
                tgt[COL_EOS_CHG_DATE].notna(),
                tgt[COL_EOS_CHG_DATE],
                tgt[COL_EOS_INIT_DATE]
            ),
        ),
        np.where(
            tgt[COL_EOS_CHG_DATE].notna(),
            tgt[COL_EOS_CHG_DATE],
            tgt[COL_EOS_INIT_DATE]
        )
    )

    # 3) ‘숫자부’(A/B 제거) & ±n 주차 계산
    if not tgt.empty:
        # 🔹 항상 string 으로 캐스팅 후 slice
        tgt[COL_RTS_WEEK] = tgt[COL_RTS_PARTIAL_WEEK].astype("string").str.slice(stop=6)
        tgt[COL_EOS_WEEK] = tgt[COL_EOS_PARTIAL_WEEK].astype("string").str.slice(stop=6)
    else:
        tgt[COL_RTS_WEEK] = tgt[COL_RTS_PARTIAL_WEEK]
        tgt[COL_EOS_WEEK] = tgt[COL_EOS_PARTIAL_WEEK]

    # _bulk_add_week 는 기존대로
    tgt[COL_RTS_WEEK_MINUST_1] = _bulk_add_week(tgt[COL_RTS_WEEK], -1)
    tgt[COL_RTS_WEEK_PLUS_3]  = _bulk_add_week(tgt[COL_RTS_WEEK],  3)
    tgt[COL_EOS_WEEK_MINUS_1] = _bulk_add_week(tgt[COL_EOS_WEEK], -1)
    tgt[COL_EOS_WEEK_MINUS_3] = _bulk_add_week(tgt[COL_EOS_WEEK], -3)
    tgt[COL_EOS_WEEK_MINUS_4] = _bulk_add_week(tgt[COL_EOS_WEEK], -4)

    # ❌ 이 줄은 타입 문제를 유발하므로 삭제하거나 맨 끝으로 옮기는 걸 추천
    # tgt.replace({'': np.nan}, inplace=True)

    # 4) RTS/EOS 초기 주차 & 기타 Helper
    if not tgt.empty:
        tgt[COL_RTS_INITIAL_WEEK] = tgt[COL_RTS_INIT_DATE].astype("string").str.slice(stop=6)
        tgt[COL_EOS_INITIAL_WEEK] = tgt[COL_EOS_INIT_DATE].astype("string").str.slice(stop=6)
    else:
        tgt[COL_RTS_INITIAL_WEEK] = tgt[COL_RTS_INIT_DATE]
        tgt[COL_EOS_INITIAL_WEEK] = tgt[COL_EOS_INIT_DATE]

    cur_ww = current_week[:6]
    cur_ww_int = int(cur_ww)

    df_time_pw = input_dataframes.get(STR_DF_TIME)
    max_week = df_time_pw[COL_TIME_PW].astype("string").max()
    max_week_int = int(max_week[:6])

    if not tgt.empty:
        # 🔹 MAX_RTS_CURRENTWEEK (int 비교)
        rts_week_int = (
            pd.to_numeric(tgt[COL_RTS_WEEK].astype("string"), errors="coerce")
              .fillna(0)
              .astype("int32")
        )
        max_rts_int = np.where(rts_week_int < cur_ww_int, cur_ww_int, rts_week_int)
        tgt[COL_MAX_RTS_CURRENTWEEK] = max_rts_int.astype(str)
    else:
        tgt.insert(loc=len(tgt.columns), column=COL_MAX_RTS_CURRENTWEEK, value=pd.NA)

    if not tgt.empty:
        # 🔹 MIN_EOSINI_MAXWEEK
        eos_ini_int = (
            pd.to_numeric(tgt[COL_EOS_INITIAL_WEEK].astype("string"), errors="coerce")
              .fillna(max_week_int)
              .astype("int32")
        )
        min_eos_ini_int = np.where(eos_ini_int < max_week_int, eos_ini_int, max_week_int)
        tgt[COL_MIN_EOSINI_MAXWEEK] = min_eos_ini_int.astype(str)

        # 🔹 MIN_EOS_MAXWEEK
        eos_week_int = (
            pd.to_numeric(tgt[COL_EOS_WEEK].astype("string"), errors="coerce")
              .fillna(max_week_int)
              .astype("int32")
        )
        min_eos_int = np.where(eos_week_int < max_week_int, eos_week_int, max_week_int)
        tgt[COL_MIN_EOS_MAXWEEK] = min_eos_int.astype(str)
    else:
        tgt.insert(loc=len(tgt.columns), column=COL_MIN_EOSINI_MAXWEEK, value=pd.NA)
        tgt.insert(loc=len(tgt.columns), column=COL_MIN_EOS_MAXWEEK, value=pd.NA)

    # 이후 Ship-To 확장/카테고리 변환 부분은 기존 그대로 유지

    # ── Ship-To Level 추가 ───────────────────────────────────────────────
    # df_fn_shipto_dim : step00_load_shipto_dimension 결과
    #   · PK = COL_SHIP_TO
    #   · 값  = COL_LV_CODE  (2~7)

    # 20250731: 필요없음
    # lvl_map = df_fn_shipto_dim.set_index(COL_SHIP_TO)[COL_LV_CODE]

    # tgt[COL_LV_CODE] = (
    #     tgt[COL_SHIP_TO]          # Lv2/Lv3 Ship-To
    #     .map(lvl_map)           # fast vectorised lookup
    #     .fillna(2)              # 매핑 실패 ⇒ LV2 로 보정
    #     .astype('int8')         # 메모리 절감
    # )

    # ── Ship-To Lv-2 → Lv-3(Std2) 확장 ─────────────────────────────────
    # dim_df    : step00 결과 (df_fn_shipto_dim)
    # COL_LV_CODE = 2 → Lv-2(Std1) , 3 → Lv-3(Std2)
    dim_df = df_fn_shipto_dim
    # ① parent-child 매핑 테이블 (Lv-3 의 Std1 = 부모 Lv-2 코드)
    df_map = (
        dim_df.loc[dim_df[COL_LV_CODE] == 3, [COL_STD1, COL_SHIP_TO]]
            .rename(columns={COL_STD1: 'PARENT_LV2',   # 부모
                            COL_SHIP_TO: 'LV3_SHIP_TO'})   # 자식
            .astype('category')
    )

    # ② tgt 중 Lv-2 행만 분리
    mask_lv2 = tgt[COL_SHIP_TO].isin(
        dim_df.loc[dim_df[COL_LV_CODE] == 2, COL_SHIP_TO]
    )
    df_lv2   = tgt.loc[mask_lv2]
    df_rest  = tgt.loc[~mask_lv2]

    # ③ (Lv-2 행 ⨯ 매핑) left-join → 자식 Lv-3 가상 행 생성
    df_lv2_expanded = (
        df_lv2
        .merge(df_map, left_on=COL_SHIP_TO, right_on='PARENT_LV2', how='left')
    )

    # ④ 자식이 있는 경우 LV3_SHIP_TO 사용, 없으면 기존 Lv-2 유지
    df_lv2_expanded[COL_SHIP_TO] = (
        df_lv2_expanded['LV3_SHIP_TO']
        .fillna(df_lv2_expanded[COL_SHIP_TO])
    )

    # ⑤ 불필요한 helper 컬럼 제거
    df_lv2_expanded.drop(columns=['PARENT_LV2', 'LV3_SHIP_TO'], inplace=True)

    # ⑥ Lv-2 fan-out 결과 + 나머지 → 최종 tgt
    tgt = pd.concat([df_rest, df_lv2_expanded], ignore_index=True)

    # ── 5) 메모리 절감 : object → category ────────────────────────────
    obj_cols = tgt.select_dtypes(include=['object','string']).columns
    tgt[obj_cols] = tgt[obj_cols].astype('category')

    return tgt        #  ➜  STR_DF_FN_RTS_EOS

################################################################################################################
# Step 03-1 : df_in_Sales_Product_ASN 전처리			
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────────
# STEP-03-1 : Sales Product ASN 전처리  → df_fn_Sales_Product_ASN_Item
#           (Lv6/Lv7 Ship-To × Item × Location 단위)
# ──────────────────────────────────────────────────────────────────────────────

# 가정: 아래 상수들은 모듈 상단에 이미 정의되어 있습니다.
# COL_SHIP_TO = 'Sales Domain.[Ship To]'
# COL_ITEM    = 'Item.[Item]'
# COL_LOC     = 'Location.[Location]'
# COL_STD5    = 'Sales Domain.[Sales Std5]'
# COL_GBM     = 'Item.[Item GBM]'
# COL_PG      = 'Item.[Product Group]'
# COL_CLASS   = 'ITEMCLASS Class'
# COL_TATTERM = 'ITEMTAT TATTERM'
# COL_TAT_SET = 'ITEMTAT TATTERM_SET'
# COL_HA_EOP_FLAG = 'HA_EOP_FLAG'

@_decoration_
def step03_1_prepare_asn_item_delta(
    df_asn:        pd.DataFrame,  # df_in_Sales_Product_ASN
    df_dim:        pd.DataFrame,  # df_in_Sales_Domain_Dimension (Std 매핑용)
    df_item_mst:   pd.DataFrame,  # df_in_Item_Master (GBM/PG)
    df_item_class: pd.DataFrame,  # df_in_Item_CLASS (HA-EOP 판단)
    df_item_tat:   pd.DataFrame,  # df_in_Item_TAT (TATTERM/SET)
    df_rts_step2:  pd.DataFrame,  # Step02 결과(PartialWeek 변환된 RTS/EOS) → 존재 Item 필터용
) -> pd.DataFrame:
    """
    Delta Step 3-1: Sales Product ASN 전처리
      • Version, 'Sales Product ASN' 제거 → (ShipTo, Item, Location)만 사용
      • **Step02 결과에 존재하는 Item만 유지**  ← Delta 차별점
      • Ship-To → Std5 매핑
      • Item Master 로 GBM/PG 매핑 (후속 Step-08 용)
      • ItemClass('X') 기준으로 HA_EOP_FLAG 계산  (Std5×Location×Item 매칭, 하위 LV 포함)
      • Item TAT (Item×Location) : TATTERM / TATTERM_SET merge → NaN=0 → int
    반환 컬럼 예시:
      [Ship To, Item, Location, Std5, GBM, PG, HA_EOP_FLAG, ITEMTAT TATTERM, ITEMTAT TATTERM_SET]
    """
    # ── 0) 최소 컬럼만 추출 & category 캐스팅
    if df_asn.empty:
        return pd.DataFrame(columns=[
            COL_SHIP_TO, COL_ITEM, COL_LOC, COL_STD5, COL_GBM, COL_PG,
            COL_HA_EOP_FLAG, COL_TATTERM, COL_TAT_SET
        ])

    asn_use = (
        df_asn[[COL_SHIP_TO, COL_ITEM, COL_LOC]]
        .copy(deep=False)
        .astype('category')
    )

    # ── 1) Step02 결과에 존재하는 Item만 유지 (Delta 스펙 차이점)
    if not df_rts_step2.empty and (COL_ITEM in df_rts_step2.columns):
        valid_items = set(df_rts_step2[COL_ITEM].dropna().unique())
        asn_use = asn_use.loc[asn_use[COL_ITEM].isin(valid_items)]
    if asn_use.empty:
        # 필터 결과 없으면 스켈레톤 반환
        return pd.DataFrame(columns=[
            COL_SHIP_TO, COL_ITEM, COL_LOC, COL_STD5, COL_GBM, COL_PG,
            COL_HA_EOP_FLAG, COL_TATTERM, COL_TAT_SET
        ])

    # ── 2) Ship-To → Std5 매핑 (하위 LV 포함 판단의 기준)
    asn_use = asn_use.merge(
        df_dim[[COL_SHIP_TO, COL_STD5]].astype({COL_SHIP_TO: 'category', COL_STD5: 'category'}),
        on=COL_SHIP_TO, how='left', copy=False
    )

    # ── 3) Item Master : GBM / PG 매핑 (후속 Step-08 에서 GBM 사용)
    if not df_item_mst.empty:
        asn_use = asn_use.merge(
            df_item_mst[[COL_ITEM, COL_GBM, COL_PG]].astype('category'),
            on=COL_ITEM, how='left', copy=False
        )
    else:
        asn_use[COL_GBM] = asn_use[COL_GBM].astype('category') if COL_GBM in asn_use else 'unknown'
        asn_use[COL_PG]  = asn_use[COL_PG].astype('category')  if COL_PG  in asn_use else 'unknown'

    # ── 4) HA_EOP_FLAG 계산 (ItemClass == 'X')
    #     ItemClass 의 (Sales Std5, Location, Item) 조합이 있으면 True
    if not df_item_class.empty:
        x_flag = (
            df_item_class.loc[df_item_class[COL_CLASS] == 'X',
                              [COL_ITEM, COL_SHIP_TO, COL_LOC]]
            .rename(columns={COL_SHIP_TO: COL_STD5})   # 표준화: Std5 기준으로 조인
            .drop_duplicates()
            .astype('category')
            .assign(__X__=True)
        )
        asn_use = asn_use.merge(
            x_flag, on=[COL_ITEM, COL_STD5, COL_LOC], how='left', copy=False
        )
        asn_use[COL_HA_EOP_FLAG] = asn_use['__X__'].fillna(False).astype('bool')
        if '__X__' in asn_use.columns:
            asn_use.drop(columns='__X__', inplace=True)
    else:
        asn_use[COL_HA_EOP_FLAG] = False

    # ── 5) ITEMTAT (Item×Location) : TATTERM / TATTERM_SET
    if not df_item_tat.empty:
        tat_cols = [COL_ITEM, COL_LOC, COL_TATTERM, COL_TAT_SET]
        asn_use = asn_use.merge(
            df_item_tat[tat_cols], on=[COL_ITEM, COL_LOC],
            how='left', copy=False
        )
    # NaN → 0 → int32
    for c in (COL_TATTERM, COL_TAT_SET):
        if c not in asn_use.columns:
            asn_use[c] = 0
    asn_use[[COL_TATTERM, COL_TAT_SET]] = (
        asn_use[[COL_TATTERM, COL_TAT_SET]].fillna(0).astype('int32')
    )

    # ── 6) dtype 정리
    asn_use = asn_use.astype({
        COL_SHIP_TO: 'category', COL_STD5: 'category',
        COL_ITEM: 'category',    COL_LOC:  'category',
        COL_GBM: 'category',     COL_PG:   'category'
    }, errors='ignore')

    # 컬럼 순서 (가독성)
    cols_order = [
        COL_SHIP_TO, COL_STD5, COL_ITEM, COL_LOC,
        COL_GBM, COL_PG, COL_HA_EOP_FLAG, COL_TATTERM, COL_TAT_SET
    ]
    cols_final = [c for c in cols_order if c in asn_use.columns]
    return asn_use[cols_final]

################################################################################################################
# Step 03-2 : df_in_Sales_Product_ASN AP2 * Item 단위로 전환(GroupBy)			
################################################################################################################
# ─────────────────────────────────────────────────────────────────────────
# STEP-08 : AP2(Lv-2/3) × Item 단위 그룹핑
#           df_fn_Sales_Product_ASN_Group 생성
# ─────────────────────────────────────────────────────────────────────────
@_decoration_
def step03_2_group_asn_to_ap2_item_delta(
    df_asn_item: pd.DataFrame,   # ← Step03-1 결과: (ShipTo, Item, Loc, HA_EOP_FLAG, TATTERM, TAT_SET)
    df_dim:      pd.DataFrame,   # ← Ship-To Dimension (COL_SHIP_TO, COL_STD1~6 포함)
    df_rts_step2: pd.DataFrame   # ← Step02 결과(RTS/EOS, ShipTo는 Lv3로 확장·정규화되어 있음)
) -> pd.DataFrame:
    """
    Step03-2 (Delta) : Sales Std2 × Item 단위로 집계
    ------------------------------------------------
    • ShipTo를 Sales Std2(=Lv3)로 정규화 후 (ShipTo_std2, Item) 단위로 그룹핑
      - HA_EOP_FLAG : any(True)  또는 max
      - TATTERM / TAT_SET : max
    • **Step02 결과에 존재하는 (ShipTo_std2, Item) 조합만 유지**
      - (요구사항) ShipTo가 Sales Std1인 경우라도 최종은 Sales Std2로 변환
      - (구현) RTS/EOS Step02 쪽 ShipTo를 다시 Std2로 정규화해 inner-join
    • 반환 dtypes
      - ShipTo, Item : category
      - TATTERM, TAT_SET : int32
      - HA_EOP_FLAG : bool
    """
    # ── 0) ShipTo → Sales Std2 정규화 룩업 준비 ──────────────────────────
    # df_dim 은 각 ShipTo 코드별로 Std 계층을 제공.
    # ShipTo 를 키로 Std2를 바로 붙여서 사용한다. (LV6/7 등도 올바르게 Std2로 귀속)
    std2_lut = df_dim[[COL_SHIP_TO, COL_STD2]].astype({COL_SHIP_TO: "category", COL_STD2: "category"})    # ── 1) ASN Item 테이블의 ShipTo ⇒ Std2로 치환 ───────────────────────
    asn_std2 = (
        df_asn_item
          .merge(std2_lut, on=COL_SHIP_TO, how="left", copy=False, sort=False)
          .rename(columns={COL_STD2: "__STD2__"})
          .copy()
    )
    # 치환: 룩업 성공 시 __STD2__ 사용, 실패 시 원 ShipTo 유지(방어)
    asn_std2[COL_SHIP_TO] = asn_std2["__STD2__"].fillna(asn_std2[COL_SHIP_TO])
    asn_std2.drop(columns=["__STD2__"], inplace=True)

    # ── 2) (ShipTo_std2, Item) 단위 집계 ─────────────────────────────────
    # agg_df = (
    #     asn_std2
    #     .groupby([COL_SHIP_TO, COL_ITEM], sort=False, observed=True)
    #     .agg({
    #         COL_HA_EOP_FLAG: "max",   # any(True) 효과
    #         COL_TATTERM    : "max",
    #         COL_TAT_SET    : "max",
    #     })
    #     .reset_index()
    # )

    agg_df = ultra_fast_groupby_numpy_general(
        df=asn_std2,
        key_cols=[COL_SHIP_TO, COL_ITEM],
        aggs={
            COL_HA_EOP_FLAG   : 'max',
            COL_TATTERM       : 'max',
            COL_TAT_SET       : 'max'
            # COL_ESTOREACCOUNT : 'max'
        }
    )

    # ── 3) Step02 존재 조합만 유지 (필터) ───────────────────────────────
    # Step02 결과의 ShipTo를 다시 한 번 Std2로 정규화(방어용)
    rts_pairs = (
        df_rts_step2[[COL_SHIP_TO, COL_ITEM]]
        .drop_duplicates()
        .merge(std2_lut, on=COL_SHIP_TO, how="left", copy=False, sort=False)
    )
    rts_pairs[COL_SHIP_TO] = rts_pairs[COL_STD2].fillna(rts_pairs[COL_SHIP_TO])
    rts_pairs = (
        rts_pairs[[COL_SHIP_TO, COL_ITEM]]
        .drop_duplicates()
        .astype({COL_SHIP_TO: "category", COL_ITEM: "category"})
    )

    # Inner-join으로 Step02에 없는 페어 제거
    agg_df = (
        agg_df
        .merge(rts_pairs, on=[COL_SHIP_TO, COL_ITEM], how="inner", copy=False, sort=False)
    )

    # ── 4) dtype 캐스팅 & 정리 ──────────────────────────────────────────
    agg_df = agg_df.astype({
        COL_SHIP_TO: "category",
        COL_ITEM   : "category",
    }, errors="ignore")

    # int/bool 보정
    for c in (COL_TATTERM, COL_TAT_SET):
        agg_df[c] = pd.to_numeric(agg_df[c], errors="coerce").fillna(0).astype("int32")
    agg_df[COL_HA_EOP_FLAG] = agg_df[COL_HA_EOP_FLAG].astype("bool", copy=False)

    gc.collect()
    return agg_df

################################################################################################################
# Step 4 : Step3-2(df_fn_ASN_Group_Week)의 Dataframe 에 Partial Week 및 Measure Column 추가. 
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────────
# STEP-04 : RTS/EOS (Lv2·Lv3 × Item) → ( +Partial-Week ) Fan-out
#           · df_in_Time_Partial_Week 과 교차조인
#           · 기본 Color = 19_GRAY
#           · Helper 컬럼 추가
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def step04_build_rts_eos_week(
        df_asn_group: pd.DataFrame,      # ← step03-2 결과 (df_fn_ASN_Group_Week)
        df_time_pw: pd.DataFrame,        # ← df_in_Time_Partial_Week
        current_week: str
) -> pd.DataFrame:
    """
    Step 02 까지 전처리한 RTS/EOS 테이블을
    * Partial-Week 축으로 fan-out  
    * 기본 색상(19_GRAY) & Week-helper 컬럼 세팅
    """
    # ── 준비 ────────────────────────────────────────────────────────────────
    pw_arr  = df_time_pw[COL_TIME_PW].astype('category').to_numpy()   # 53-week 벡터
    n_weeks = pw_arr.size    # fan-out 에 남길 핵심 컬럼
    base_cols = [
        COL_ITEM, 
        COL_SHIP_TO
        # COL_LV_CODE
        # COL_GBM,  
        # COL_PG,
        # COL_RTS_PARTIAL_WEEK, COL_EOS_PARTIAL_WEEK,
        # COL_RTS_WEEK,         COL_EOS_WEEK,
        # COL_RTS_WEEK_MINUST_1, COL_RTS_WEEK_PLUS_3,
        # COL_EOS_WEEK_MINUS_1,  COL_EOS_WEEK_MINUS_4,
        # COL_RTS_INITIAL_WEEK,  COL_EOS_INITIAL_WEEK,
        # COL_MAX_RTS_CURRENTWEEK, COL_MIN_EOSINI_MAXWEEK, COL_MIN_EOS_MAXWEEK
    ]
    core = df_asn_group[base_cols].reset_index(drop=True)

    # ── 1) Lv2/3×Item × 53week 크로스조인 (repeat / tile) ────────────────
    rep_idx           = np.repeat(np.arange(len(core)), n_weeks)
    df_out            = core.iloc[rep_idx].reset_index(drop=True)
    df_out[COL_TIME_PW] = np.tile(pw_arr, len(core))

    # ── 2) Helper 컬럼 ────────────────────────────────────────────────────
    df_out[COL_CURRENT_ROW_WEEK] = df_out[COL_TIME_PW].str.slice(stop=6)

    v_add_week = np.vectorize(common.gfn_add_week, otypes=[object])
    # df_out[COL_CURRENT_ROW_WEEK_PLUS_8] = v_add_week(
    #     df_out[COL_CURRENT_ROW_WEEK], 8
    # )

    # ── 3) 기본 Color 초기화 ──────────────────────────────────────────────
    # df_out[COL_SIN_FCST_COLOR_COND] = None          # Changed From 19_GRAY. 20250731
    df_out[COL_SIN_FCST_COLOR_COND] = pd.NA        # 14_WHITE → 미정(NaN)으로 변경
    df_out[COL_SIN_FCST_COLOR_COND] = (
        df_out[COL_SIN_FCST_COLOR_COND].astype('string').astype('category')
    )

    # ── 4) dtype 최적화 ──────────────────────────────────────────────────
    obj_cols = df_out.select_dtypes(include=['object','string']).columns
    df_out[obj_cols] = df_out[obj_cols].astype('category')

    return df_out


################################################################################################################
# Step 5 : Step4의 df 에 당주주차부터 RTS 와 EOS 반영 및 Color 표시	
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────────
# STEP-05 : RTS / EOS Color 반영
#            – 규칙 5-1 ~ 5-5 (WHITE / DARKBLUE / LIGHTBLUE / LIGHTRED / DARKRED)
#            – 벡터라이즈 & 1-pass overwrite (19_GRAY → WHITE → … → DARKRED)
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def step05_apply_color_rts_eos(
        df_week: pd.DataFrame,          # ← step04 결과 (df_fn_ASN_Group_Week)
        df_rts:  pd.DataFrame,          # ← step02 결과 (df_fn_RTS_EOS)
        current_week: str,              # ‘202447A’ 등
        apply_eos_red: bool = True     # 🔸 추가 - 기본값 True (기존과 동일)
) -> pd.DataFrame:
    """
    df_week 의 기본 색상(19_GRAY)을 RTS/EOS 일정에 따라 덮어쓴다.
    ─ 규칙 요약 ───────────────────────────────────────────────────────────────
        · 5-1  WHITE      : max(RTS_WEEK, CW) ~ min(EOS-1, MAX_WEEK) => 적용안함 0731
        · 5-2  DARKBLUE   : RTS_INIT_WEEK ~ RTS_WEEK-1
        · 5-3  LIGHTBLUE  : RTS_WEEK ~ RTS_WEEK+3
        · 5-4  LIGHTRED   : EOS_WEEK-4 ~ EOS_WEEK-1
        · 5-5  DARKRED    : EOS_WEEK ~
      (CW = Current Week ‘YYYYWW’ 형식)
    순서는 “밝은 색 ➞ 진한 색” 으로 덮어쓰며, NaN / '' 는 무시.
    비교는 모두 'YYYYWW' → int32 로 변환해서 수행한다.
    """

    # ────────────────────── 0. RTS/EOS helper 붙이기 ─────────────────────
    cols_rts = [
        COL_ITEM, COL_SHIP_TO,
        COL_RTS_WEEK,          COL_RTS_INITIAL_WEEK,
        COL_RTS_WEEK_MINUST_1, COL_RTS_WEEK_PLUS_3,
        COL_EOS_WEEK,          COL_EOS_INITIAL_WEEK,
        COL_EOS_WEEK_MINUS_1,
        COL_EOS_WEEK_MINUS_3, # 1125 추가 
        COL_EOS_WEEK_MINUS_4,
        COL_MIN_EOS_MAXWEEK,   COL_MAX_RTS_CURRENTWEEK,
        COL_RTS_DEV_DATE      # ★ 추가: DEV 기준 시작 주차 계산용
    ]
    df_join = df_week.merge(
        df_rts[cols_rts],
        on=[COL_ITEM, COL_SHIP_TO],
        how='left',
        copy=False
    ) 
    # categor 등록
    _ALL_COLORS = [
        COLOR_GRAY, COLOR_WHITE,
        COLOR_DARKBLUE, COLOR_LIGHTBLUE,
        COLOR_LIGHTRED, COLOR_DARKRED,
        COLOR_DGRAY_RED, COLOR_DGRAY_REDB,
        COLOR_GREEN
    ]
    if not isinstance(df_join[COL_SIN_FCST_COLOR_COND].dtype, pd.CategoricalDtype):
        df_join[COL_SIN_FCST_COLOR_COND] = df_join[COL_SIN_FCST_COLOR_COND].astype('category')

    # 아직 등록되지 않은 색상만 추가
    missing = [c for c in _ALL_COLORS if c not in df_join[COL_SIN_FCST_COLOR_COND].cat.categories]
    if missing:
        df_join[COL_SIN_FCST_COLOR_COND] = df_join[COL_SIN_FCST_COLOR_COND].cat.add_categories(missing) 

    # ────────────────────── 1. 모든 week → int32 배열 변환 ────────────────
    def _week_to_int(s: pd.Series) -> np.ndarray:
        """'YYYYWW' 문자열/카테고리 → int32 (결측치 0)."""
        return (
            pd.to_numeric(s.astype("string"), errors='coerce')   # NaN → NaN
              .fillna(0)
              .astype('int32')
              .to_numpy()
        )

    wk_now      = _week_to_int(df_join[COL_CURRENT_ROW_WEEK])
    rts_week    = _week_to_int(df_join[COL_RTS_WEEK])
    rts_init    = _week_to_int(df_join[COL_RTS_INITIAL_WEEK])
    rts_m1      = _week_to_int(df_join[COL_RTS_WEEK_MINUST_1])
    rts_p3      = _week_to_int(df_join[COL_RTS_WEEK_PLUS_3])

    eos_week    = _week_to_int(df_join[COL_EOS_WEEK])
    eos_init    = _week_to_int(df_join[COL_EOS_INITIAL_WEEK])
    eos_m1      = _week_to_int(df_join[COL_EOS_WEEK_MINUS_1])
    eos_m3      = _week_to_int(df_join[COL_EOS_WEEK_MINUS_3]) # 1125 추가
    eos_m4      = _week_to_int(df_join[COL_EOS_WEEK_MINUS_4])

    rts_max_cw  = _week_to_int(df_join[COL_MAX_RTS_CURRENTWEEK])
    eos_cut_max = _week_to_int(df_join[COL_MIN_EOS_MAXWEEK])

    # ★ 추가: RTS_DEV_DATE(PartialWeek) 기준 Week int (YYYYWW)
    # Step02에서 COL_RTS_DEV_DATE 는 'YYYYWWS' 형식(Partial Week)이므로 앞 6자리만 사용
    # 1-1) RTS_DEV_DATE 를 string 으로 캐스팅 후 앞 6자리만 사용
    rts_dev_str = (
        df_join[COL_RTS_DEV_DATE]
        .astype("string")    # float64 / NaN / empty 모두 안전
        .str.slice(stop=6)
    )
    rts_dev = _week_to_int(rts_dev_str)

    cur_ww_int  = int(current_week[:6])          # ex) '202447A' → 202447


    # ────────────────────── 2. Mask 계산 & 색상 덮어쓰기 ────────────────
    # 5-0 GRAY : 당주주차 이상 ~ RTS_INIT_DATE 미만 구간 
    mask_gray = (
        (wk_now >= cur_ww_int) &
        (wk_now < rts_init)
    )
    df_join.loc[mask_gray, COL_SIN_FCST_COLOR_COND] = COLOR_GRAY

    # 5-1 WHITE (사용 안함 – 기존 주석 유지)
    # rts_max = np.where(rts_week < cur_ww_int, cur_ww_int, rts_week)
    # mask_white = (wk_now >= rts_max) & (wk_now <= eos_cut_max)
    # df_join.loc[mask_white, COL_SIN_FCST_COLOR_COND] = COLOR_WHITE

    # ★ 변경: DARKBLUE 시작 주차 = max(RTS_INIT_WEEK, RTS_DEV_WEEK)
    #   - RTS_DEV_DATE 가 존재하면 DEV 기준으로 시작
    #   - DEV 가 없거나 INIT 보다 앞선 이상값이면 INIT 유지(데이터 방어)
    rts_start_dblue = np.maximum(rts_init, rts_dev)

    # 5-2 DARKBLUE
    mask_dblue = (
        (wk_now >= cur_ww_int) &
        (wk_now >= rts_start_dblue) & (wk_now <= rts_m1)
    )
    df_join.loc[mask_dblue, COL_SIN_FCST_COLOR_COND] = COLOR_DARKBLUE

    # 5-3 LIGHTBLUE
    mask_lblue = (
        (wk_now >= cur_ww_int) &
        (wk_now >= rts_week) & (wk_now <= rts_p3)
    )
    df_join.loc[mask_lblue, COL_SIN_FCST_COLOR_COND] = COLOR_LIGHTBLUE

    if apply_eos_red:                 # 🔸 옵션에 따라 RED 계열 생략
        # NaN → 0 으로 변환된 eos_week 가운데 0 은 ‘EOS 정보 없음’ 이므로 제외
        valid_eos = eos_week > 0                                   # 🔹 추가 
        # 5-4 LIGHTRED
        mask_lred = (
            valid_eos &                                           # ← 추가
            (wk_now >= cur_ww_int) &
            (wk_now >= eos_m3) & (wk_now <= eos_week)   # 1125 변경
        )
        df_join.loc[mask_lred, COL_SIN_FCST_COLOR_COND] = COLOR_LIGHTRED

        # 5-5 DARKRED
        mask_dred = (
            valid_eos &                                           # ← 추가
            (wk_now >= cur_ww_int) &
            (wk_now > eos_week)     # 1125 변경
        )
        df_join.loc[mask_dred, COL_SIN_FCST_COLOR_COND] = COLOR_DARKRED

    # ────────────────────── 3. 정리 ─────────────────────────────────────
    # 필요 없는 RTS/EOS helper 컬럼 제거 (메모리 절감)
    df_out = df_join.drop(
        columns=[c for c in cols_rts if c not in (COL_ITEM, COL_SHIP_TO)]
    )
    df_out[[COL_SHIP_TO]] =  df_out[[COL_SHIP_TO]].astype('category')
    return df_out

################################################################################################################
# Step 6 : 무선 BAS 제품 8주 구간 13_GREEN UPDATE		
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────────
# STEP-06 : Wireless BAS 모델 – 당주 포함 8 주 구간 13_GREEN 적용
# ──────────────────────────────────────────────────────────────────────────────
#  ☑ 대상  : Item.[ProductType] == 'BAS'  and  Item.[Item GBM] == 'MOBILE'
#  ☑ 범위  : current_week ~ current_week+7  (8 주)
#  ☑ 규칙  : 기존 색상이 19_GRAY 가 *아닌* 행 → 13_GREEN 으로 덮어쓰기
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def step06_apply_green_for_wireless_bas(
        df_week: pd.DataFrame,          # ← step05 결과 (df_fn_ASN_Group_Week)
        df_item_mst: pd.DataFrame,      # ← df_in_Item_Master
        current_week: str
) -> pd.DataFrame:

    # ── 0) Wireless BAS Item 목록 준비 ───────────────────────────────────
    mask_bas = (
        (df_item_mst[COL_PT]  == 'BAS') &
        (df_item_mst[COL_GBM] == 'MOBILE')
    )
    bas_items = df_item_mst.loc[mask_bas, COL_ITEM].unique()
    del df_item_mst, mask_bas
    gc.collect()

    if len(bas_items) == 0:          # 대상 Item 없으면 그대로 반환
        return df_week

    bas_set = set(bas_items)         # lookup O(1)

    # ── 1) 주차 → int32 배열 변환 ────────────────────────────────────────
    # cur_ww_int = int(current_week[:6])
    week_int   = (
        pd.to_numeric(df_week[COL_CURRENT_ROW_WEEK].astype("string"), errors='coerce')
          .fillna(0).astype('int32').to_numpy()
    )
    cw_6      = current_week[:6]                    # '202447A' → '202447'
    cw_int    = int(cw_6)
    cw_p7_int = int(common.gfn_add_week(cw_6, 7))   # 년/주 경계 안전 처리

    # ── 2) 색상 카테고리 확보 (GREEN) ────────────────────────────────────
    if COLOR_GREEN not in df_week[COL_SIN_FCST_COLOR_COND].cat.categories:
        df_week[COL_SIN_FCST_COLOR_COND] = (
            df_week[COL_SIN_FCST_COLOR_COND]
              .cat.add_categories([COLOR_GREEN])
        )

    # ── 3) 마스크 계산 & 덮어쓰기 ────────────────────────────────────────
    #   • Item 이 bas_set 에 존재
    #   • 주차가 [CW, CW+7]
    #   • 현재 색상이 GRAY 가 아님
    mask_item  = df_week[COL_ITEM].isin(bas_set).to_numpy()
    mask_week  = (week_int >= cw_int) & (week_int <= cw_p7_int)
    mask_color = (df_week[COL_SIN_FCST_COLOR_COND] != COLOR_GRAY).to_numpy()

    mask_green = mask_item & mask_week & mask_color
    df_week.loc[mask_green, COL_SIN_FCST_COLOR_COND] = COLOR_GREEN

    # ── 4) 메모리 정리 ──────────────────────────────────────────────────
    del bas_items, bas_set, week_int, mask_item, mask_week, mask_color, mask_green
    gc.collect()

    return df_week


################################################################################################################
# Step-07  : HA EOP management model 조건 반영	
################################################################################################################
# ─────────────────────────────────────────────────────────────────────────
# STEP-07 : HA-EOP Management 모델 12_YELLOW 반영
#           (df_fn_ASN_Group_Week ← df_fn_Sales_Product_ASN_Group)
# ─────────────────────────────────────────────────────────────────────────
@_decoration_
def step07_apply_ha_eop_yellow(
        df_week:      pd.DataFrame,
        df_rts:       pd.DataFrame,
        df_asn_group: pd.DataFrame,
        current_week: str
) -> pd.DataFrame:
    """
    조건
    ▸ df_asn_group.HA_EOP_FLAG == True  (AP2×Item)
    ▸ Case-1  EOS_WEEK > CW   →  [CW, EOS-4]  = 12_YELLOW
      Case-2  EOS 없음/지남   →  [CW, END]     = 12_YELLOW
    미리 LIGHTRED/DARKRED 가 칠해진 구간은 건드리지 않는다.
    """ 
    targets = (
        df_asn_group.loc[df_asn_group[COL_HA_EOP_FLAG],[COL_SHIP_TO, COL_ITEM]]
        .drop_duplicates()
        .astype('category')
        .assign(__PAIR__=True)
    )
    # 1) 최소 조인 (EOS 계산용 컬럼만)
    cols_need = [COL_ITEM, COL_SHIP_TO, COL_EOS_WEEK, COL_EOS_WEEK_MINUS_4]
    wk_join = df_week.merge(
        df_rts[cols_need], on=[COL_ITEM, COL_SHIP_TO], how='left', copy=False
    )
    # 대상 페어 플래그 머지 (apply 제거)
    wk_join = wk_join.merge(
        targets, on=[COL_SHIP_TO, COL_ITEM], how='left', copy=False
    )
    mask_pair = wk_join['__PAIR__'].notna().to_numpy()
    wk_join.drop(columns='__PAIR__', inplace=True)    
    
    # ── 2) week → int32 배열
    def _w2i(s): 
        return pd.to_numeric(s.astype("string"), errors='coerce').fillna(0).astype('int32').to_numpy()
    wk_now = _w2i(wk_join[COL_CURRENT_ROW_WEEK])
    eos_w  = _w2i(wk_join[COL_EOS_WEEK])
    eos_m4 = _w2i(wk_join[COL_EOS_WEEK_MINUS_4])
    cw     = int(current_week[:6])

    # ── 3) 색상 카테고리(YELLOW) 확보 ──────────────────────────────────
    if COLOR_YELLOW not in wk_join[COL_SIN_FCST_COLOR_COND].cat.categories:
        wk_join[COL_SIN_FCST_COLOR_COND] = wk_join[COL_SIN_FCST_COLOR_COND].cat.add_categories([COLOR_YELLOW])
    # ── 4) 마스크 계산
    # 마스크 (RED는 보호)
    darker = (wk_join[COL_SIN_FCST_COLOR_COND] == COLOR_LIGHTRED) | (wk_join[COL_SIN_FCST_COLOR_COND] == COLOR_DARKRED)
    has_future_eos = (eos_w > cw)
    mask_y1 = mask_pair & has_future_eos & (wk_now >= cw) & (wk_now < eos_m4)
    mask_y2 = mask_pair & ~has_future_eos & (wk_now >= cw)
    mask_y  = (mask_y1 | mask_y2) & (~darker.to_numpy())

    wk_join.loc[mask_y, COL_SIN_FCST_COLOR_COND] = COLOR_YELLOW
    df_out = wk_join.drop(columns=[COL_EOS_WEEK, COL_EOS_WEEK_MINUS_4])
    gc.collect()
    return df_out

################################################################################################################
# Step-08  : VD, SHA Lead Time 구간	
################################################################################################################
################################################################################################################
# Step-08-1  : Lead Time 구간 DARKGRAY RED UPDATE	
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────
# STEP-08-1 : VD / SHA  Lead-Time (18_DGRAY_RED) 갱신
# ──────────────────────────────────────────────────────────────────────────
@_decoration_
def step08_1_apply_vd_leadtime(
        df_week:      pd.DataFrame,    # ← step09 결과  df_fn_ASN_Group_Week
        df_asn_item:  pd.DataFrame,    # ← step07 결과  df_fn_Sales_Product_ASN_Item
        df_asn_group: pd.DataFrame,    # ← step08 결과  df_fn_Sales_Product_ASN_Group
        current_week: str
) -> pd.DataFrame:
    """
    대상
      · Item GBM이 'VD' 또는 'SHA'
      · ITEMTAT TATTERM > 0  (AP2×Item 단위)
    규칙
      CW(포함) ~ CW+TATTERM-1 구간이 WHITE(14) 이면 18_DGRAY_RED 로 덮어씀
      0731: WHITE(14) 조건은 적용안함
    """    
    # ── 0) VD/SHA + TATTERM > 0  목록을 “DataFrame” 으로 준비 ──────────
    # 0-1) Item → GBM 매핑 Series
    gbm_series = (
        df_asn_item[[COL_ITEM, COL_GBM]]
        .drop_duplicates(subset=[COL_ITEM])
        .set_index(COL_ITEM)[COL_GBM]
    )

    # 0-2) df_asn_group 에 GBM 붙이고 조건 필터
    df_tat = (
        df_asn_group
          .loc[df_asn_group[COL_TATTERM] > 0,
               [COL_SHIP_TO, COL_ITEM, COL_TATTERM]]
          .assign(**{COL_GBM: lambda d: d[COL_ITEM].map(gbm_series)})
          .loc[lambda d: d[COL_GBM].isin(['VD', 'SHA'])]
          .astype({COL_TATTERM: 'int32',
                   COL_SHIP_TO: 'category',
                   COL_ITEM   : 'category'})
    )
    if df_tat.empty:                         # 대상 없음 ➜ 그대로 반환
        return df_week

    # ── 1) (ShipTo,Item)→TATTERM “정렬된 ndarray” 만들기 ────────────────
    mi_week = pd.MultiIndex.from_arrays(
        [df_week[COL_SHIP_TO], df_week[COL_ITEM]],
        names=[COL_SHIP_TO, COL_ITEM]
    )
    tat_arr = (
        df_tat.set_index([COL_SHIP_TO, COL_ITEM])[COL_TATTERM]
              .reindex(mi_week, fill_value=0)        # 길이 = len(df_week)
              .to_numpy('int16')
    )

    # ── 2) TATTERM 값이 있는 행만 end-week 계산 (유니크 값 → 1회 호출) ──
    cw_6  = current_week[:6]             # '202447'
    uniq  = np.unique(tat_arr[tat_arr > 0])
    # {Δweek:int → end_week_int:int} 룩업
    end_map = {
        d: int(common.gfn_add_week(cw_6, int(d) - 1)) for d in uniq
    }
    # 전체 배열로 확장
    end_week_int = np.fromiter(
        (end_map.get(d, 0) for d in tat_arr),
        dtype='int32',
        count=len(tat_arr)
    )

    # ── 3) 비교·색상 덮어쓰기 ──────────────────────────────────────────
    wk_now_int = (
        pd.to_numeric(df_week[COL_CURRENT_ROW_WEEK].astype("string"),
                      errors='coerce').fillna(0).astype('int32').to_numpy()
    )
    cw_int = int(cw_6)

    if COLOR_DGRAY_RED not in df_week[COL_SIN_FCST_COLOR_COND].cat.categories:
        df_week[COL_SIN_FCST_COLOR_COND] = (
            df_week[COL_SIN_FCST_COLOR_COND]
              .cat.add_categories([COLOR_DGRAY_RED])
        )

    mask = (
        (tat_arr > 0) &
        (wk_now_int >= cw_int) &
        (wk_now_int <= end_week_int) 
        & (df_week[COL_SIN_FCST_COLOR_COND].isna())   # WHITE → NaN 체크
        # & (df_week[COL_SIN_FCST_COLOR_COND] == COLOR_WHITE)  # ==> Nan 인경우
    )
    df_week.loc[mask, COL_SIN_FCST_COLOR_COND] = COLOR_DGRAY_RED
    fn_log_dataframe(df_week.loc[df_week[COL_SIN_FCST_COLOR_COND] == COLOR_DGRAY_RED],f'step08_1_{COLOR_DGRAY_RED}')
    gc.collect()
    return df_week


################################################################################################################
# Step-08-2  : SET Lead Time 구간 DARKGRAY REDB  UPDATE
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────────
# STEP-08-2 : SET Lead-Time 구간 17_DGRAY_REDB 갱신
#             (VD/SHA 모델 + AP2×Item 단위 ITEMTAT TAT_SET 사용)
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def step08_2_apply_set_leadtime(
        df_week:      pd.DataFrame,    # ← step10-1 이후 df_fn_ASN_Group_Week
        df_asn_item:  pd.DataFrame,    # ← step07 결과 df_fn_Sales_Product_ASN_Item
        df_asn_group: pd.DataFrame,    # ← step08 결과 df_fn_Sales_Product_ASN_Group
        current_week: str
) -> pd.DataFrame:
    """
    VD/SHA 모델 중  ▸ ITEMTAT_TAT_SET > 0  ▸ CW ~ CW+TAT_SET-1
    ──────────────────────────────────────────────────────────────
      · 현재 색상이 WHITE(14)  → 17_DGRAY_REDB  ==> 0731: WHITE(14) 조건은 적용안함
      · 현재 색상이 18_DGRAY_RED → 17_DGRAY_REDB 로 승격
    고속화 버전 : gfn_add_week 호출 횟수 = unique(TAT_SET) 개수
    """    
    # ── 0) VD/SHA + TAT_SET>0 목록 (행 수 ≪ df_week) ────────────────────
    gbm_map = (
        df_asn_item[[COL_ITEM, COL_GBM]]
        .drop_duplicates(subset=[COL_ITEM])
        .set_index(COL_ITEM)[COL_GBM]
    )
    df_set = (
        df_asn_group
          .loc[df_asn_group[COL_TAT_SET] > 0,
               [COL_SHIP_TO, COL_ITEM, COL_TAT_SET]]
          .assign(**{COL_GBM: lambda d: d[COL_ITEM].map(gbm_map)})
          .loc[lambda d: d[COL_GBM].isin(['VD', 'SHA'])]
          .astype({COL_TAT_SET: 'int16',
                   COL_SHIP_TO: 'category',
                   COL_ITEM   : 'category'})
    )
    if df_set.empty:
        return df_week

    # ── 1) (ShipTo,Item) → TAT_SET ndarray (len = len(df_week)) ────────
    idx_week = pd.MultiIndex.from_arrays(
        [df_week[COL_SHIP_TO], df_week[COL_ITEM]],
        names=[COL_SHIP_TO, COL_ITEM]
    )
    tatset_arr = (
        df_set.set_index([COL_SHIP_TO, COL_ITEM])[COL_TAT_SET]
              .reindex(idx_week, fill_value=0)
              .to_numpy('int16')
    )

    # ── 2) end-week 계산 : unique Δweek마다 1회만 gfn_add_week 호출 ─────
    cw_6   = current_week[:6]
    cw_int = int(cw_6)
    uniq   = np.unique(tatset_arr[tatset_arr > 0])
    end_map = {d: int(common.gfn_add_week(cw_6, int(d) - 1)) for d in uniq}
    end_week_int = np.fromiter(
        (end_map.get(d, 0) for d in tatset_arr),
        dtype='int32',
        count=len(tatset_arr)
    )

    # ── 3) 비교용 배열 준비 ─────────────────────────────────────────────
    wk_now_int = (
        pd.to_numeric(df_week[COL_CURRENT_ROW_WEEK].astype("string"),
                      errors='coerce').fillna(0)
          .astype('int32').to_numpy()
    )

    if COLOR_DGRAY_REDB not in df_week[COL_SIN_FCST_COLOR_COND].cat.categories:
        df_week[COL_SIN_FCST_COLOR_COND] = (
            df_week[COL_SIN_FCST_COLOR_COND]
              .cat.add_categories([COLOR_DGRAY_REDB])
        )

    # ── 4-A) WHITE → 17_DGRAY_REDB ────────────────────────────────────
    # mask_white = (
    #     (tatset_arr > 0) &
    #     (wk_now_int >= cw_int) &
    #     (wk_now_int <= end_week_int)
    #     & (df_week[COL_SIN_FCST_COLOR_COND] == COLOR_WHITE)
    # )
    # fn_log_dataframe(df_week.loc[mask_white],f'step10_2_{COLOR_DGRAY_REDB}_신규')
    # df_week.loc[mask_white, COL_SIN_FCST_COLOR_COND] = COLOR_DGRAY_REDB
    
    # ── 4-B) 18_DGRAY_RED → 17_DGRAY_REDB 승격 ────────────────────────
    mask_promote = (
        (tatset_arr > 0) &
        (wk_now_int >= cw_int) &
        (wk_now_int <= end_week_int) &
        (df_week[COL_SIN_FCST_COLOR_COND] == COLOR_DGRAY_RED)
    )
    fn_log_dataframe(df_week.loc[mask_promote],f'step08_2_{COLOR_DGRAY_REDB}_승격')
    df_week.loc[mask_promote, COL_SIN_FCST_COLOR_COND] = COLOR_DGRAY_REDB

    fn_log_dataframe(df_week.loc[df_week[COL_SIN_FCST_COLOR_COND] == COLOR_DGRAY_REDB],f'step08_2_{COLOR_DGRAY_REDB}')
    gc.collect()
    return df_week

################################################################################################################
# Step-09  : MX Sellout FCST 없는 모델 당주 이후 미래구간 GRAY UPDATE
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────────
# STEP-09 : MX  Sell-out Forecast  없는  모델  GRAY  업데이트
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def step09_apply_gray_no_sellout(
        df_week:        pd.DataFrame,
        df_dim:         pd.DataFrame,    # (사용 안함: df_week가 이미 Std2)
        df_no_sout:     pd.DataFrame,
        current_week:   str
) -> pd.DataFrame:
    """
    (Lv-2 Ship-To ≒ STD2, Item) 단위로  `S/Out Fcst Not Exist Flag` 가 True 인 모델은  
    Current Week(CW) 포함 이후 주차를 모두 19_GRAY 로 덮어쓴다.
    """    
    # df_week.COL_SHIP_TO 가 이미 Lv3(=Std2) 이므로 그대로 사용
    flags = (
        df_no_sout.loc[df_no_sout[COL_SOUT_FCST_NOT_EXISTS], [COL_STD2, COL_ITEM]]
        .drop_duplicates()
        .rename(columns={COL_STD2: COL_SHIP_TO})
        .astype('category')
        .assign(__FLAG__=True)
    )
    df_week = df_week.merge(
        flags, on=[COL_SHIP_TO, COL_ITEM], how='left', sort=False, copy=False
    )    
    
    if COLOR_GRAY not in df_week[COL_SIN_FCST_COLOR_COND].cat.categories:
        df_week[COL_SIN_FCST_COLOR_COND] = df_week[COL_SIN_FCST_COLOR_COND].cat.add_categories([COLOR_GRAY])

    wk_now = pd.to_numeric(df_week[COL_CURRENT_ROW_WEEK].astype("string"), errors='coerce').fillna(0).astype('int32').to_numpy()
    cw     = int(current_week[:6])
    mask   = df_week['__FLAG__'].notna().to_numpy() & (wk_now >= cw)

    df_week.loc[mask, COL_SIN_FCST_COLOR_COND] = COLOR_GRAY
    df_week.drop(columns='__FLAG__', inplace=True)

    # 메모리 절감 : object → category
    obj_cols = df_week.select_dtypes(include=['object','string']).columns
    df_week[obj_cols] = df_week[obj_cols].astype('category')
    gc.collect()
    return df_week

################################################################################################################
# Step-10  : Output 구성. df_output_Sell_In_FCST_Color_Condition
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────────
# STEP-10 : df_output_Sell_In_FCST_Color_Condition
#           • Version
#           • Sales Std2  (Lv-2 코드)
#           • Item
#           • Partial Week
#           • S/In FCST Color Condition
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def step10_build_sell_in_output(
        df_week:   pd.DataFrame,
        df_dim:    pd.DataFrame,   # (사용 안함)
        version:   str
) -> pd.DataFrame:
    # df_week: ShipTo=Std2 이미 사용 중 → 그대로 복사 없이 뽑기
    df_out = df_week[[COL_SHIP_TO, COL_ITEM, COL_TIME_PW, COL_SIN_FCST_COLOR_COND]].copy(deep=False)
    df_out.rename(columns={COL_SHIP_TO: COL_STD2}, inplace=True)    # Version 단일 카테고리(복사 최소화)
    ver_cat = pd.Categorical.from_codes(
        np.zeros(len(df_out), dtype='int8'), categories=[version]
    )
    df_out.insert(0, COL_VERSION, ver_cat)

    # 필요한 컬럼들만 카테고리 보장 (이미 대부분 category)
    df_out = df_out.astype({
        COL_VERSION:'category', COL_STD2:'category', COL_ITEM:'category',
        COL_TIME_PW:'category', COL_SIN_FCST_COLOR_COND:'category'
    }, errors='ignore')
    gc.collect()
    return df_out

################################################################################################################
# Step-13  : Output 구성. df_output_Sell_Out_FCST_Color_Condition
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────────
# STEP-13 : df_output_Sell_Out_FCST_Color_Condition
#           • eStoreAccount = True 인 (Std2, Item) 조합만 남김
#           • 컬럼명 S/In → S/Out 으로 변경
#           사용안함. 20250718
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def step13_build_sell_out_output(
        df_sin:       pd.DataFrame,   # ← step12 결과  df_output_Sell_In_FCST_Color_Condition
        df_asn_group: pd.DataFrame    # ← step08 결과  df_fn_Sales_Product_ASN_Group
) -> pd.DataFrame:
    """
    Filter Sell-In 결과 → eStore 대상 모델만 남기고
    컬럼 'S/In FCST Color Condition' → 'S/Out FCST Color Condition' 으로 교체
    """    
    # ── 0) eStore (Std2 / Item) 집합 ─────────────────────────────────────
    estore_pairs = (
        df_asn_group
          .loc[df_asn_group[COL_ESTOREACCOUNT],  # True 만
               [COL_SHIP_TO, COL_ITEM]]
          .drop_duplicates()
          .astype({COL_SHIP_TO: "category", COL_ITEM: "category"})
    )

    # ── 1) Sell-In ↔ eStore 매칭 (Std2 = Ship-To Lv-2) ─────────────────
    #   df_sin.COL_STD2  ⇔  df_asn_group.COL_SHIP_TO
    df_out = (
        df_sin
          .merge(estore_pairs,
                 left_on =[COL_STD2, COL_ITEM],
                 right_on=[COL_SHIP_TO, COL_ITEM],
                 how="inner",          # eStore 대상만 남김
                 copy=False)
          .drop(columns=[COL_SHIP_TO])  # 조인용 컬럼 제거
    )

    if df_out.empty:                     # eStore 대상이 없으면 빈 DF 반환
        gc.collect()
        return df_out

    # ── 2) 컬럼명 변경  (In → Out) ──────────────────────────────────────
    df_out.rename(
        columns={COL_SIN_FCST_COLOR_COND: COL_SOUT_FCST_COLOR_COND},
        inplace=True
    )

    # ── 3) dtype 최적화 ────────────────────────────────────────────────
    cat_cols = [COL_VERSION, COL_STD2, COL_ITEM,
                COL_TIME_PW, COL_SOUT_FCST_COLOR_COND]
    df_out[cat_cols] = df_out[cat_cols].astype("category")

    gc.collect()
    return df_out


################################################################################################################
# Step-11  : Output 구성. df_output_Sell_Out_FCST_Color_Condition. Concat. Repeat step4 to step10-1
################################################################################################################
# ──────────────────────────────────────────────────────────────────────
# STEP-11 : non-eStore Sell-Out 색상 계산 + Step13 결과와 병합
# ──────────────────────────────────────────────────────────────────────
@_decoration_
def step11_extend_sell_out_output(
        df_rts:         pd.DataFrame,
        df_time_pw:     pd.DataFrame,
        df_dim:         pd.DataFrame,
        df_item_mst:    pd.DataFrame,
        df_item_class:  pd.DataFrame,
        df_item_tat:    pd.DataFrame,
        df_asn:         pd.DataFrame,
        current_week:   str,
        version:        str
) -> pd.DataFrame:
    df_asn_item  = df_fn_Sales_Product_ASN_Item
    df_asn_group = df_fn_Sales_Product_ASN_Group    
    
    # 4~8-1 : 동일 (RED 제외한 5, GREEN, 08-1)
    df_asn_week = step04_build_rts_eos_week(df_asn_group, df_time_pw, current_week)
    df_asn_week = step05_apply_color_rts_eos(df_asn_week, df_rts, current_week, False)
    df_asn_week = step06_apply_green_for_wireless_bas(df_asn_week, df_item_mst, current_week)
    df_asn_week = step08_1_apply_vd_leadtime(df_asn_week, df_asn_item, df_asn_group, current_week)

    # ShipTo(=Std2) 그대로 사용 + 컬럼명 교체 + Version 단일 카테고리
    df_new = df_asn_week[[COL_SHIP_TO, COL_ITEM, COL_TIME_PW, COL_SIN_FCST_COLOR_COND]].copy(deep=False)
    df_new.rename(columns={
        COL_SHIP_TO: COL_STD2,
        COL_SIN_FCST_COLOR_COND: COL_SOUT_FCST_COLOR_COND
    }, inplace=True)

    ver_cat = pd.Categorical.from_codes(
        np.zeros(len(df_new), dtype='int8'), categories=[version]
    )
    df_new.insert(0, COL_VERSION, ver_cat)

    df_new = df_new.astype({
        COL_VERSION:'category', COL_STD2:'category',
        COL_ITEM:'category', COL_TIME_PW:'category',
        COL_SOUT_FCST_COLOR_COND:'category'
    }, errors='ignore')

    gc.collect()
    return df_new

####################################
############ Start Main  ###########
####################################
if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()
    try:
        pd.options.mode.copy_on_write = True
    except Exception:
        pass

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
            input_folder_name  = f'PYForecastSellInAndOutColor/PYForecastSellInAndOutColorDelta'
            output_folder_name = f'PYForecastSellInAndOutColorDeltaNumber'
            # # ME
            # input_folder_name  = f'PYForecastSellInAndOutColor/input_tenant_0829'
            # output_folder_name = f'PYForecastSellInAndOutColor_input_tenant2'
            
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
            CurrentPartialWeek = '202447A'
            # CurrentPartialWeek = '202518B'
        
        # vdLog 초기화
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
        logger.Note(p_note=f'CurrentPartialWeek : {CurrentPartialWeek}', p_log_level=LOG_LEVEL.debug())



        ################################################################################################################
        # Step 00 – Ship-To 차원 LUT 구축
        ################################################################################################################
        dict_log = {
            'p_step_no': 00,
            'p_step_desc': 'Step 00 – load Ship-To dimension LUT'
        }
        df_fn_shipto_dim = step00_load_shipto_dimension(
            input_dataframes[STR_DF_DIM],
            **dict_log
        )
        fn_log_dataframe(df_fn_shipto_dim, f'step00_df_fn_shipto_dim')

        # ── Delta 입력 상수 예시 ─────────────────────────────
        # STR_DF_RTS_EOS_DELTA = 'df_in_MST_RTS_EOS_Delta'# (중략) input_dataframes[...] 로 테이블 로딩까지 끝난 상태라고 가정
        # Step 1-0 : df_output_RTS_EOS_Delta 생성
        dict_log = {
            "p_step_no": 1.0,
            "p_step_desc": "Step 01-0 – build df_output_RTS_EOS_Delta (nullify measures)"
        }
        df_output_RTS_EOS_Delta = step01_0_build_output_rts_eos_delta(
            input_dataframes[STR_DF_RTS_EOS_DELTA],   # df_in_MST_RTS_EOS_Delta
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_RTS_EOS_Delta, "step01_0_df_output_RTS_EOS_Delta")

        # Step 1-1 : (기존 Step1) df_in_MST_RTS_EOS_Delta 전처리 → df_fn_RTS_EOS 생성
        # (기존 함수명 그대로 유지)
        df_fn_RTS_EOS, df_output_RTS_EOS = step01_preprocess_rts_eos_delta(
            input_dataframes[STR_DF_RTS_EOS_DELTA],   # Delta
            input_dataframes[STR_DF_RTS_EOS],         # Full
            Version
        )
        # =====================================================
        # STEP 01 — RTS/EOS Delta 전처리 (Global 사용 없음)
        #   • df_fn_RTS_EOS       : 이후 Step02 투입용(Version, RTS_ISVALID 제거)
        #   • df_output_RTS_EOS   : 스펙 Output(3) 보존본(Version 포함)
        # =====================================================
        dict_log = {
            'p_step_no': 1, 
            'p_step_desc': 'Step 01 – RTS/EOS Delta preprocessing'
        }  # (옵션) 기록용

        df_fn_RTS_EOS, df_output_RTS_EOS = step01_preprocess_rts_eos_delta(
            input_dataframes[STR_DF_RTS_EOS_DELTA],   # Delta
            input_dataframes[STR_DF_RTS_EOS],         # Full
            Version
        )

        # (원하면) 로그/CSV 출력
        fn_log_dataframe(df_fn_RTS_EOS,     'step01_df_fn_RTS_EOS')
        fn_log_dataframe(df_output_RTS_EOS, 'step01_df_output_RTS_EOS')

        # 이후 Step 02에서 df_fn_RTS_EOS 사용:
        # df_fn_RTS_EOS = step02_convert_date_to_partial_week(df_fn_RTS_EOS, CurrentPartialWeek, **dict_log)

        ################################################################################################################
        # Step 2 : Step1의 Result에 Time을 Partial Week 으로 변환
        ################################################################################################################
        dict_log = {
            'p_step_no' : 2,
            'p_step_desc': 'Step 02 – convert RTS/EOS dates to partial week'
        }
        df_fn_RTS_EOS = step02_convert_date_to_partial_week(
            df_fn_RTS_EOS,                 # Step-01 결과 DF
            CurrentPartialWeek,
            **dict_log
        )
        fn_log_dataframe(df_fn_RTS_EOS, f'step02_{STR_DF_FN_RTS_EOS}')


        ################################################################################################################
        # Step 3 : Sales Product ASN 으로 모수 Data 생성
        ################################################################################################################
        ################################################################################################################
        # Step 03-1 : df_in_Sales_Product_ASN 전처리			
        ################################################################################################################
        dict_log = {
            'p_step_no' : 3.1,
            'p_step_desc': 'Step 03-1 – prepare ASN Item table'
        }
        # ===== Main: Step03-1 호출부 (Delta) =====
        # 준비물:
        #   - input_dataframes[STR_DF_ASN]         : df_in_Sales_Product_ASN
        #   - input_dataframes[STR_DF_DIM]         : df_in_Sales_Domain_Dimension
        #   - input_dataframes[STR_DF_ITEMMST]     : df_in_Item_Master
        #   - input_dataframes[STR_DF_ITEMCLASS]   : df_in_Item_CLASS
        #   - input_dataframes[STR_DF_ITEMTAT]     : df_in_Item_TAT
        #   - df_fn_RTS_EOS_step2                  : Step02 결과(Partial Week 변환된 RTS/EOS)logger.Note(p_note='Step 03-1 – prepare ASN Item (Delta)', p_log_level=LOG_LEVEL.debug())

        df_fn_Sales_Product_ASN_Item = step03_1_prepare_asn_item_delta(
            input_dataframes[STR_DF_ASN],
            input_dataframes[STR_DF_DIM],
            input_dataframes[STR_DF_ITEMMST],
            input_dataframes[STR_DF_ITEMCLASS],
            input_dataframes[STR_DF_ITEMTAT],
            df_fn_RTS_EOS ,      # ← Step02 결과를 그대로 전달 (Delta 핵심)
            **dict_log
        )

        fn_log_dataframe(df_fn_Sales_Product_ASN_Item, f'step03_1_{STR_DF_FN_ASN_ITEM}')

        ################################################################################################################
        # Step 03-2 : df_in_Sales_Product_ASN  AP2 * Item 단위로 전환 (Delta)
        ################################################################################################################
        dict_log = {
            'p_step_no' : 3.2,
            'p_step_desc': 'Step 03-2 – group ASN to AP2×Item (Delta)'
        }

        df_fn_Sales_Product_ASN_Group = step03_2_group_asn_to_ap2_item_delta(
            df_fn_Sales_Product_ASN_Item,   # ← Step03-1 결과
            df_fn_shipto_dim,               # ← Step00 결과(차원 LUT)
            df_fn_RTS_EOS,                  # ← Step02 결과(RTS/EOS, Partial Week 변환본)
            **dict_log
        )

        # (선택) 레지스트리 저장/로그
        # input_dataframes[STR_DF_FN_ASN_GROUP] = df_fn_Sales_Product_ASN_Group
        fn_log_dataframe(df_fn_Sales_Product_ASN_Group, f'step03_2_{STR_DF_FN_ASN_GROUP}')
        ################################################################################################################
        # Step 4 : Step3-2(df_fn_Sales_Product_ASN_Group)의 Dataframe 에 Partial Week 및 Measure Column 추가. 
        ################################################################################################################
        ################################################################################################################
        # STEP 04 – add Partial-Week & base colour
        ################################################################################################################
        dict_log = {
            'p_step_no' : 4,
            'p_step_desc': 'Step 04 – add PartialWeek & base colour'
        }
        df_fn_ASN_Group_Week = step04_build_rts_eos_week(
            df_fn_Sales_Product_ASN_Group,           # ← step03-2 결과
            input_dataframes[STR_DF_TIME],           # Time.PW 테이블
            CurrentPartialWeek,
            **dict_log
        )
        fn_log_dataframe(df_fn_ASN_Group_Week, f'step04_{STR_DF_FN_ASN_PW}')

        # 이후 Step 들이 참조할 수 있도록 registry 에 저장
        input_dataframes[STR_DF_FN_ASN_PW] = df_fn_ASN_Group_Week

        ################################################################################################################
        # Step 5 : Step4의 df 에 당주주차부터 RTS 와 EOS 반영 및 Color 표시	
        ################################################################################################################
        ################################################################################################################
        # Step 05 – RTS/EOS Color 설정
        ################################################################################################################
        dict_log = {
            'p_step_no' : 5,
            'p_step_desc': 'Step 05 – set Color by RTS/EOS'
        }
        df_fn_ASN_Group_Week = step05_apply_color_rts_eos(
            df_fn_ASN_Group_Week,        # ← step04 결과
            df_fn_RTS_EOS,             # ← step02 결과
            CurrentPartialWeek,
            **dict_log                 # 데코레이터에서 Step 로그 처리
        )
        fn_log_dataframe(df_fn_ASN_Group_Week, f'step05_{STR_DF_FN_ASN_PW}')

        ################################################################################################################
        # Step 6 : 무선 BAS 제품 8주 구간 13_GREEN UPDATE		
        ################################################################################################################
        ################################################################################################################
        # STEP 06 – Wireless BAS 8 Week GREEN
        ################################################################################################################
        dict_log = {
            'p_step_no' : 6,
            'p_step_desc': 'Step 06 – Wireless BAS 8-week GREEN'
        }
        df_fn_ASN_Group_Week = step06_apply_green_for_wireless_bas(
            df_fn_ASN_Group_Week,        # ← step05 이후 최신 버전
            input_dataframes[STR_DF_ITEMMST],
            CurrentPartialWeek,        # '202447A' 등
            **dict_log
        )
        fn_log_dataframe(df_fn_ASN_Group_Week, f'step06_{STR_DF_FN_ASN_PW}')



        ################################################################################################################
        # Step-07  : HA EOP management model 조건 반영	
        ################################################################################################################
        ################################################################################################
        # STEP 07 – HA-EOP Yellow update
        ################################################################################################
        dict_log = {
            'p_step_no' : 7,
            'p_step_desc': 'Step 07 – apply HA-EOP 12_YELLOW'
        }
        df_fn_ASN_Group_Week = step07_apply_ha_eop_yellow(
            df_fn_ASN_Group_Week,    # ← step06 결과 (갱신)
            df_fn_RTS_EOS,         # ← step02 결과
            df_fn_Sales_Product_ASN_Group,  # ← step08 결과
            CurrentPartialWeek,    # ‘202447A’ 등
            **dict_log
        )
        fn_log_dataframe(df_fn_ASN_Group_Week, f'step07_{STR_DF_FN_ASN_PW}')

        ################################################################################################################
        # Step-08  : VD, SHA Lead Time 구간	
        ################################################################################################################
        ################################################################################################################
        # Step-08-1  : Lead Time 구간 DARKGRAY RED UPDATE	
        ################################################################################################################
        ################################################################################################
        # STEP 08-1 – VD/SHA Lead-Time (18_DGRAY_RED)
        ################################################################################################
        dict_log = {
            'p_step_no' : 8.1,
            'p_step_desc': 'Step 08-1 – VD/SHA Lead-Time DGRAY_RED'
        }
        df_fn_ASN_Group_Week = step08_1_apply_vd_leadtime(
            df_fn_ASN_Group_Week,
            df_fn_Sales_Product_ASN_Item,    # step07 (GBM 보유)
            df_fn_Sales_Product_ASN_Group,   # step08            
            CurrentPartialWeek,
            **dict_log
        )
        fn_log_dataframe(df_fn_ASN_Group_Week, f'step08_1_{STR_DF_FN_ASN_PW}')

        ################################################################################################################
        # Step-08-2  : SET Lead Time 구간 DARKGRAY REDB  UPDATE
        ################################################################################################################
        ################################################################################################
        # Step 08-2 – SET Lead-Time (17_DGRAY_REDB)
        ################################################################################################
        dict_log = {
            'p_step_no' : 10.2,
            'p_step_desc': 'Step 10-2 – apply SET Lead-Time (17_DGRAY_REDB)'
        }
        df_fn_ASN_Group_Week = step08_2_apply_set_leadtime(
            df_fn_ASN_Group_Week,              # ← step08-1 결과
            df_fn_Sales_Product_ASN_Item,    # ← step03-1 결과
            df_fn_Sales_Product_ASN_Group,   # ← step03-2 결과
            CurrentPartialWeek,
            **dict_log
        )
        fn_log_dataframe(df_fn_ASN_Group_Week, f'step08_2_{STR_DF_FN_ASN_PW}')

        ################################################################################################################
        # Step-09  : MX Sellout FCST 없는 모델 당주 이후 미래구간 GRAY UPDATE
        ################################################################################################################
        dict_log = {
            'p_step_no' : 9,
            'p_step_desc': 'Step 9 – apply GRAY for models w/o Sell-out FCST'
        }
        df_fn_ASN_Group_Week = step09_apply_gray_no_sellout(
            df_fn_ASN_Group_Week,     # step08-2 결과
            df_fn_shipto_dim,       # step00 결과
            input_dataframes[STR_DF_NO_SELL_OUT],   # df_in_SELLOUTFCST_NOTEXIST
            CurrentPartialWeek,
            **dict_log
        )
        fn_log_dataframe(df_fn_ASN_Group_Week, f'step09_{STR_DF_FN_ASN_PW}')


        ################################################################################################################
        # Step-10  : Output 구성. df_output_Sell_In_FCST_Color_Condition
        ################################################################################################################
        ###############################################################################
        # Step 10 – Sell-In Output
        ###############################################################################
        dict_log = {
            'p_step_no' : 10,
            'p_step_desc': 'Step 10 – build Sell-In FCST Color Condition output'
        }
        df_output_Sell_In_FCST_Color_Condition = step10_build_sell_in_output(
            df_fn_ASN_Group_Week,    # ← step09 결과
            df_fn_shipto_dim,      # ← step00 결과 (Ship-To ↔ Std2 LUT)
            Version,               # 전역 Version 문자열
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_In_FCST_Color_Condition,
                        f'step10_{STR_DF_OUT_SIN}')

        ################################################################################################################
        # Step-13  : Output 구성. df_output_Sell_Out_FCST_Color_Condition
        ################################################################################################################
        ###############################################################################
        # Step 13 – Sell-Out Output
        ###############################################################################
        # dict_log = {
        #     "p_step_no" : 13,
        #     "p_step_desc": "Step 13 – build Sell-Out FCST Color Condition output"
        # }
        # df_output_Sell_Out_FCST_Color_Condition = step13_build_sell_out_output(
        #     df_output_Sell_In_FCST_Color_Condition,   # ← step12
        #     df_fn_Sales_Product_ASN_Group,            # ← step08
        #     **dict_log
        # )
        # fn_log_dataframe(df_output_Sell_Out_FCST_Color_Condition,
        #                 f"step13_{STR_DF_OUT_SOUT}")


        ################################################################################################################
        # Step-11  : Output 구성. df_output_Sell_Out_FCST_Color_Condition. Concat. Repeat step4 to step10-1
        ################################################################################################################
        ###############################################################################
        # Step 11 – merge non-eStore Sell-Out
        ###############################################################################
        logger.debug("")
        logger.Note(p_note=f'Start Step13', p_log_level=LOG_LEVEL.debug())
        
        del df_fn_ASN_Group_Week
        # del df_fn_Sales_Product_ASN_Item 
        # del df_fn_Sales_Product_ASN_Group
        gc.collect()
        
        dict_log = {
            "p_step_no": 11,
            "p_step_desc": "Step 11 – extend Sell-Out FCST Color Condition (non-eStore)"
        }
        df_output_Sell_Out_FCST_Color_Condition = step11_extend_sell_out_output(
            df_fn_RTS_EOS,          # step02
            input_dataframes[STR_DF_TIME],
            df_fn_shipto_dim,
            input_dataframes[STR_DF_ITEMMST],
            input_dataframes[STR_DF_ITEMCLASS],
            input_dataframes[STR_DF_ITEMTAT],
            input_dataframes[STR_DF_ASN],
            # input_dataframes[STR_DF_ESTORE],
            CurrentPartialWeek,
            Version,
            # df_output_Sell_Out_FCST_Color_Condition,   # Step-13 output
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_Out_FCST_Color_Condition,
                        f"step11_{STR_DF_OUT_SOUT}")


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
        