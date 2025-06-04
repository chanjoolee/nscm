from re import X
import os,sys,json,shutil,io,zipfile
import time
import datetime
import inspect
import traceback
import pandas as pd
from pandas.core.resample import T
from NSCMCommon import NSCMCommon as common
# from typing_extensions import Literal
import glob
import numpy as np
from typing import Collection, Tuple,Union,Dict
import re
# import rbql
# import duckdb

########################################################################################################################
# Local 개발 시에 필요한 공통 변수 선언
########################################################################################################################
# o9에 저장된 instanceName
is_local = common.gfn_get_isLocal()
str_instance = 'PYForecastSellInAndEstoreSellOutLockColor'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"

is_print = True
flag_csv = True
flag_exception = True

########################################################################################################################
# 컬럼상수
########################################################################################################################
Version_Name = 'Version.[Version Name]'
Item_Item = 'Item.[Item]'
# Item_Type = 'Item.[Item Type]'  
Item_Type = 'Item.[ProductType]'  
Item_GBM = 'Item.[Item GBM]'
Product_Group = 'Item.[Product Group]'
Item_Lv = 'Item_Lv'
RTS_EOS_ShipTo          = 'RTS_EOS_ShipTo'
ForecastRuleShipto      = 'ForecastRuleShipto'
Sales_Domain_ShipTo     = 'Sales Domain.[Ship To]'
Sales_Domain_LV2        = 'Sales Domain.[Sales Domain Lv2]'
Sales_Domain_LV3        = 'Sales Domain.[Sales Domain Lv3]'
Sales_Domain_LV4        = 'Sales Domain.[Sales Domain Lv4]' 
Sales_Domain_LV5        = 'Sales Domain.[Sales Domain Lv5]'
Sales_Domain_LV6        = 'Sales Domain.[Sales Domain Lv6]'
Sales_Domain_LV7        = 'Sales Domain.[Sales Domain Lv7]'

# df_SELLOUTFCST_NOTEXIST
Sales_Std1              = 'Sales Domain.[Sales Std1]'
Sales_Std2              = 'Sales Domain.[Sales Std2]'
Sales_Std3              = 'Sales Domain.[Sales Std3]' 
Sales_Std4              = 'Sales Domain.[Sales Std4]'
Sales_Std5              = 'Sales Domain.[Sales Std5]'
Sales_Std6              = 'Sales Domain.[Sales Std6]'
SOut_Fcst_Not_exist     = 'S/Out Fcst Not Exist Flag'

Location_Location       = 'Location.[Location]'
Item_Class              = 'ITEMCLASS Class'
Partial_Week            = 'Time.[Partial Week]'
SIn_FCST_GC_LOCK                = 'S/In FCST(GI)_GC_Lock'
# SIn_FCST_GC_LOCK                = 'Lock Condition'
SIn_FCST_Color_Condition        = 'S/In FCST Color Condition'
SIn_FCST_AP2_LOCK               = 'S/In FCST(GI)_AP2_Lock'
SIn_FCST_AP1_LOCK               = 'S/In FCST(GI)_AP1_Lock'
# ───── 새 컬럼 / Step Renames ─────────────────────
Lock_Condition            = 'Lock Condition'              # (=구 GC.Lock)
Color_Condition           = 'S/In FCST Color Condition'   # 유지
# Step 10-1 / 10-2 색상 상수
COLOR_DGRAY_RED           = '18_DGRAY_RED'
COLOR_DGRAY_REDB          = '17_DGRAY_REDB'
COLOR_WHITE               = '14_WHITE'
COLOR_LIGHTRED            = '11_LIGHTRED'
COLOR_DARKRED             = '16_DARKRED'
COLOR_YELLOW              = '12_YELLOW'
COLOR_GRAY                = '19_GRAY'
COLOR_GREEN               = '13_GREEN'
COLOR_DARKBLUE            = '15_DARKBLUE'
COLOR_LIGHTBLUE           = '10_LIGHTBLUE'


# Salse_Product_ASN       = 'Sales Product ASN'    
Salse_Product_ASN       = 'Sales Product ASN'
ITEMTAT_TATTERM         = 'ITEMTAT TATTERM'
ITEMTAT_TATTERM_SET     = 'ITEMTAT TATTERM_SET'
FORECAST_RULE_GC_FCST        = 'FORECAST_RULE GC FCST'
FORECAST_RULE_AP2_FCST       = 'FORECAST_RULE AP2 FCST'
FORECAST_RULE_AP1_FCST       = 'FORECAST_RULE AP1 FCST'   
FORECAST_RULE_CUST      = 'FORECAST_RULE CUST FCST'
# ----------------------------------------------------------------
# New column constants for step13
# ----------------------------------------------------------------
SOut_FCST_GC_LOCK         = 'S/Out FCST_GC_Lock'
SOut_FCST_AP2_LOCK        = 'S/Out FCST_AP2_Lock'
SOut_FCST_AP1_LOCK        = 'S/Out FCST_AP1_Lock'
SOut_FCST_Color_Condition = 'S/Out FCST Color Condition'


# ----------------------------------------------------------------
# Helper column constants for step02
# ----------------------------------------------------------------
CURRENT_ROW_WEEK                    = 'current_row_partial_week_normalized'
CURRENT_ROW_WEEK_PLUS_8             = 'CURRENTWEEK_NORMALIZED_PLUS_8'   

RTS_INIT_DATE                       = 'RTS_INIT_DATE'
RTS_DEV_DATE                        = 'RTS_DEV_DATE'
RTS_COM_DATE                        = 'RTS_COM_DATE'

RTS_WEEK                            = 'RTS_WEEK_NORMALIZED'
RTS_PARTIAL_WEEK                    = 'RTS_PARTIAL_WEEK'
RTS_INITIAL_WEEK                    = 'RTS_INITIAL_WEEK_NORMALIZED'
RTS_WEEK_MINUST_1                   = 'RTS_WEEK_NORMALIZED_MINUST_1'
RTS_WEEK_PLUS_3                     = 'RTS_WEEK_NORMALIZED_PLUS_3'
MAX_RTS_CURRENTWEEK                 = 'MAX_RTS_CURRENTWEEK'
RTS_STATUS                          = 'RTS_STATUS'

EOS_INIT_DATE                       = 'EOS_INIT_DATE'
EOS_CHG_DATE                        = 'EOS_CHG_DATE'
EOS_COM_DATE                        = 'EOS_COM_DATE'

EOS_WEEK                            = 'EOS_WEEK_NORMALIZED'
EOS_PARTIAL_WEEK                    = 'EOS_PARTIAL_WEEK'
# EOS_WEEK_MINUS_1                    = 'EOS_WEEK_NORMALIZED_MINUS_1'
EOS_WEEK_MINUS_1         = 'EOS_WEEK_NORMALIZED_MINUS_1'
EOS_WEEK_MINUS_4         = 'EOS_WEEK_NORMALIZED_MINUS_4'
EOS_INITIAL_WEEK         = 'EOS_INITIAL_WEEK_NORMALIZED'
MIN_EOSINI_MAXWEEK                  = 'MIN_EOSINI_MAXWEEK'
MIN_EOS_MAXWEEK                  = 'MIN_EOS_MAXWEEK'
EOS_STATUS                          = 'EOS_STATUS'
# ───────────────────────────────────────────────────────────────
# CONSTANT STRING VARIABLES FOR DATAFRAME NAMES
# ───────────────────────────────────────────────────────────────
# input
str_df_in_Sales_Domain_Dimension        = 'df_in_Sales_Domain_Dimension'
str_df_in_Sales_Domain_Estore           = 'df_in_Sales_Domain_Estore'
str_df_in_Time_Partial_Week             = 'df_in_Time_Partial_Week'
str_df_in_Item_CLASS                    = 'df_in_Item_CLASS'
str_df_in_Item_TAT                      = 'df_in_Item_TAT'
# str_df_in_MST_EOS                       = 'df_in_MST_EOS'
# str_df_in_MST_RTS                       = 'df_in_MST_RTS'
str_df_in_MST_RTS_EOS                   = 'df_in_MST_RTS_EOS'
str_df_in_Sales_Product_ASN             = 'df_in_Sales_Product_ASN'
str_df_in_Forecast_Rule                 = 'df_in_Forecast_Rule'
str_df_in_Item_Master                   = 'df_in_Item_Master'
str_df_in_SELLOUTFCST_NOTEXIST          = 'df_in_SELLOUTFCST_NOTEXIST'
# ───── I/O DataFrame handles (spec 이름 준수) ────
STR_DF_RTS_EOS            = 'df_in_MST_RTS_EOS'
STR_DF_RULE               = 'df_in_Forecast_Rule'
STR_DF_DIM                = 'df_in_Sales_Domain_Dimension'
STR_DF_ASN                = 'df_in_Sales_Product_ASN'
STR_DF_TIME               = 'df_in_Time_Partial_Week'
STR_DF_ITEMTAT            = 'df_in_Item_TAT'
STR_DF_ITEMCLASS          = 'df_in_Item_CLASS'
STR_DF_ITEMMST            = 'df_in_Item_Master'
STR_DF_ESTORE             = 'df_in_Sales_Domain_Estore'
STR_DF_NO_SELL_OUT        = 'df_in_SELLOUTFCST_NOTEXIST'

# middle
str_df_fn_RTS_EOS                           = 'df_fn_RTS_EOS'
str_df_fn_RTS_EOS_Week                      = 'df_fn_RTS_EOS_Week'
str_df_fn_Sales_Product_ASN_Item            = 'df_fn_Sales_Product_ASN_Item'
str_df_fn_Sales_Product_ASN_Item_Week       = 'df_fn_Sales_Product_ASN_Item_Week'
str_df_fn_Forcast_in                        = 'df_fn_Forcast_in'
str_df_fn_Forcast_out                       = 'df_fn_Forcast_out'




########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################
logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
# fn_set_local_logfile()
LOG_LEVEL = common.G_log_level

########################################################################################################################
# Start Function Of Utils
########################################################################################################################
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

        (Sales_Domain_LV7 ,7),
        (Sales_Domain_LV6, 6), (Sales_Domain_LV5, 5), (Sales_Domain_LV4, 4),
        (Sales_Domain_LV3, 3), (Sales_Domain_LV2, 2)
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
    dim_idx = df_dim.set_index(Sales_Domain_ShipTo)
    lv_cols = [Sales_Domain_LV2, Sales_Domain_LV3, Sales_Domain_LV4,
               Sales_Domain_LV5, Sales_Domain_LV6, Sales_Domain_LV7]
    lv_arrs = dim_idx[lv_cols].to_numpy(dtype=object)
    return dim_idx.index, lv_arrs
# -------------------------------------------------


@_decoration_
def fn_process_in_df_mst():

    if is_local: 
        # 로컬인 경우 Output 폴더를 정리한다.
        for file in os.scandir(str_output_dir):
            os.remove(file.path)

        # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
        file_pattern = f"{os.getcwd()}/{str_input_dir}/*.csv" 
        csv_files = glob.glob(file_pattern)

        # file_to_df_mapping = {
        #     'df_in_Sales_Domain_Dimension.csv'          :      str_df_in_Sales_Domain_Dimension     ,
        #     'df_in_Sales_Domain_Estore.csv'             :      str_df_in_Sales_Domain_Estore        ,
        #     'df_in_Time_Partial Week.csv'               :      str_df_in_Time_Partial_Week          ,
        #     'MST_ITEMCLASS.csv'                         :      str_df_in_Item_CLASS                 ,
        #     'MST_ITEMTAT.csv'                           :      str_df_in_Item_TAT                   ,
        #     'df_in_MST_RTS_EOS.csv'                     :      str_df_in_MST_RTS_EOS                ,
        #     'MST_SALESPRODUCT.csv'                      :      str_df_in_Sales_Product_ASN          ,
        #     'df_in_Forecast_Rule.csv'                   :      str_df_in_Forecast_Rule              ,
        #     'VUI_ITEMATTB.csv'                          :      str_df_in_Item_Master                ,
        #     'df_in_SELLOUTFCST_NOTEXIST.csv'            :      str_df_in_SELLOUTFCST_NOTEXIST            
        # }

        file_to_df_mapping = {
            'df_in_Sales_Domain_Dimension.csv'          :      str_df_in_Sales_Domain_Dimension     ,
            'df_in_Sales_Domain_Estore.csv'             :      str_df_in_Sales_Domain_Estore        ,
            'df_in_Time_Partial_Week.csv'               :      str_df_in_Time_Partial_Week          ,
            'df_in_Item_CLASS.csv'                      :      str_df_in_Item_CLASS                 ,
            'df_in_Item_TAT.csv'                        :      str_df_in_Item_TAT                   ,
            'df_in_MST_RTS_EOS.csv'                     :      str_df_in_MST_RTS_EOS                ,
            'df_in_Sales_Product_ASN.csv'               :      str_df_in_Sales_Product_ASN          ,
            'df_in_Forecast_Rule.csv'                   :      str_df_in_Forecast_Rule              ,
            'df_in_Item_Master.csv'                     :      str_df_in_Item_Master                ,
            'df_in_SELLOUTFCST_NOTEXIST.csv'            :      str_df_in_SELLOUTFCST_NOTEXIST            
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
        input_dataframes[str_df_in_Sales_Domain_Dimension]  = df_in_Sales_Domain_Dimension
        input_dataframes[str_df_in_Sales_Domain_Estore]     = df_in_Sales_Domain_Estore
        input_dataframes[str_df_in_Time_Partial_Week]       = df_in_Time_Partial_Week
        input_dataframes[str_df_in_Item_CLASS]              = df_in_Item_CLASS
        input_dataframes[str_df_in_Item_TAT]                = df_in_Item_TAT
        input_dataframes[str_df_in_MST_RTS_EOS]             = df_in_MST_RTS_EOS
        input_dataframes[str_df_in_Sales_Product_ASN]       = df_in_Sales_Product_ASN
        input_dataframes[str_df_in_Forecast_Rule]           = df_in_Forecast_Rule
        input_dataframes[str_df_in_Item_Master]             = df_in_Item_Master
        input_dataframes[str_df_in_SELLOUTFCST_NOTEXIST]    = df_in_SELLOUTFCST_NOTEXIST

    fn_convert_type(input_dataframes[str_df_in_MST_RTS_EOS], 'Sales Domain', str)

    fn_convert_type(input_dataframes[str_df_in_Item_CLASS], 'Sales Domain', str)
    fn_convert_type(input_dataframes[str_df_in_Item_CLASS], 'Location', str)
    fn_convert_type(input_dataframes[str_df_in_Item_CLASS], 'Item', str)
    fn_convert_type(input_dataframes[str_df_in_Item_CLASS], 'ITEMCLASS', str)

    fn_convert_type(input_dataframes[str_df_in_Sales_Product_ASN], 'Sales Domain', str)
    fn_convert_type(input_dataframes[str_df_in_Sales_Product_ASN], 'Location', str)

    fn_convert_type(input_dataframes[str_df_in_Sales_Domain_Dimension], 'Sales Domain', str)
    fn_convert_type(input_dataframes[str_df_in_Sales_Domain_Estore], 'Sales Domain', str)
    fn_convert_type(input_dataframes[str_df_in_Forecast_Rule], 'Sales Domain', str)

    input_dataframes[str_df_in_Forecast_Rule][FORECAST_RULE_GC_FCST].fillna(0, inplace=True)
    input_dataframes[str_df_in_Forecast_Rule][FORECAST_RULE_AP2_FCST].fillna(0, inplace=True)
    input_dataframes[str_df_in_Forecast_Rule][FORECAST_RULE_AP1_FCST].fillna(0, inplace=True)
    input_dataframes[str_df_in_Forecast_Rule][FORECAST_RULE_CUST].fillna(0, inplace=True)

    fn_convert_type(input_dataframes[str_df_in_Forecast_Rule], FORECAST_RULE_GC_FCST, 'int32')
    fn_convert_type(input_dataframes[str_df_in_Forecast_Rule], FORECAST_RULE_AP2_FCST, 'int32')
    fn_convert_type(input_dataframes[str_df_in_Forecast_Rule], FORECAST_RULE_AP1_FCST, 'int32')
    fn_convert_type(input_dataframes[str_df_in_Forecast_Rule], FORECAST_RULE_CUST, 'int32')

    fn_convert_type(input_dataframes[str_df_in_SELLOUTFCST_NOTEXIST], 'Sales Domain', str)



def analyze_by_rbql() :
    asn_df = input_dataframes[str_df_in_Sales_Product_ASN]
    master_df = input_dataframes[str_df_in_Item_Master]

    my_quey = f"""
        select 
            a['{Sales_Domain_ShipTo}'] as shipto,
            a['{Item_Item}'] as item,
            a['{Location_Location}'] as location,
            b['{Item_Type}'] as item_type,
            b['{Item_GBM}'] as item_gbm,
            b['{Product_Group}'] as product_group
        Join b on a['{Item_Item}'] == b['{Item_Item}']
        where b['{Item_Type}'] == 'BAS'

    """

    result = rbql.query_pandas_dataframe(
        query_text=my_quey,
        input_dataframe=asn_df,
        join_dataframe=master_df
    )


def analyze_by_duckdb():
    # import duckdb
    # Retrieve your DataFrames
    asn_df    = input_dataframes[str_df_in_Sales_Product_ASN]
    master_df = input_dataframes[str_df_in_Item_Master]
    dim_df   = input_dataframes[str_df_in_Sales_Domain_Dimension]      

    # Register each DataFrame as a DuckDB table
    duckdb.register('asn_table', asn_df)
    duckdb.register('master_table', master_df)
    duckdb.register('dim_table', dim_df)

    # Build a SQL query referencing them by table aliases a (asn_table) and b (master_table)
    my_query = f"""
    SELECT
        a['{Sales_Domain_ShipTo}'] AS shipto,
        c['{Sales_Domain_LV2}'] AS DomainLv2,
        a['{Item_Item}']          AS item,
        a['{Location_Location}']  AS location,
        b['{Item_Type}']          AS item_type,
        b['{Item_GBM}']          AS item_gbm,
        b['{Product_Group}']      AS product_group
    FROM asn_table AS a
    JOIN master_table AS b
      ON a['{Item_Item}'] == b['{Item_Item}']
    JOIN dim_table AS c
        ON a['{Sales_Domain_ShipTo}'] == c['{Sales_Domain_ShipTo}']
    WHERE b['{Item_Type}']  = 'BAS'
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
        a['{Sales_Domain_ShipTo}'] AS shipto,
        c['{Sales_Domain_LV2}'] AS DomainLv2,
        a['{Item_Item}']          AS item,
        a['{Location_Location}']  AS location,
        b['{Item_Type}']          AS item_type,
        b['{Item_GBM}']          AS item_gbm,
        b['{Product_Group}']      AS product_group
    FROM asn_table AS a
    JOIN master_table AS b
      ON a['{Item_Item}'] == b['{Item_Item}']
    JOIN dim_table AS c
        ON a['{Sales_Domain_ShipTo}'] == c['{Sales_Domain_ShipTo}']
    WHERE b['{Item_Type}']  = 'BAS'
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

# 4.1 Convert to partial week with error-checking. use datttime
# def to_partial_week_datetime(x):
#     try:
#         if x is not None and x != '':
#             # If x is not already a Python datetime, try to convert it
#             dt = pd.to_datetime(x)
#             dt = datetime.datetime.strptime(str(x),'%Y/%m/%d')
#             return common.gfn_get_partial_week(dt, True)
#         else: 
#             return ''
#     except Exception as e:
#         try:
#             dt = datetime.datetime.strptime(str(x),'%Y-%m-%d')
#             return common.gfn_get_partial_week(dt, True) 
#         except Exception as e1:
#             print("Error in to_partial_week with value:", x, "Error:", e1)
#             return ''

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
########################################################################################################################
# End Function Of Utils
########################################################################################################################

########################################################################################################################
# Start Function Of Steps
########################################################################################################################
# ───── STEP 01 – RTS/EOS 전처리 ───────────────────
@_decoration_
def fn_step01_load_rts_eos(df_rts_eos: pd.DataFrame,
                        shipto_idx: pd.Index,
                        level_arr:  np.ndarray) -> pd.DataFrame:
    """
    * merge 필요 없음 – 입력 그대로 사용
    * Ship-To → Item_lv(2/3) 계산
    """
    # vectorized level
    pos  = shipto_idx.get_indexer(df_rts_eos[Sales_Domain_ShipTo].to_numpy())
    lv   = np.where(pos >= 0, level_arr[pos], np.nan)
    df   = df_rts_eos.assign(Item_Lv=lv.astype('int8'))
    # ▲TODO: 날짜 → PartialWeek 변환 등 spec Step 02 로직 이곳에 포함
    return df


# common.gfn_get_partial_week(dt: datetime, use_AB=True) 는 그대로 사용# ── 백업용 개별 파서 ──────────────────────────────────────────────────
_FMT_TRIES = ('%Y/%m/%d', '%Y-%m-%d', '%m-%d-%Y', '%m/%d/%Y')

def _try_strptime(arr_str: np.ndarray) -> np.ndarray:
    """
    넘겨받은 str 배열(길이 k)에 대해 여러 포맷 순차 시도.
    성공한 곳은 datetime, 실패는 None 유지.
    """
    out = np.array([None]*len(arr_str), dtype=object)
    mask = np.full(len(arr_str), True,  bool)          # 아직 미변환
    for fmt in _FMT_TRIES:
        if not mask.any():
            break                                       # 모두 변환됨
        try:
            parsed = np.vectorize(
                lambda s: datetime.datetime.strptime(s, fmt), otypes=[object]
            )(arr_str[mask])
            ok = np.fromiter((p is not None for p in parsed), bool)
            out_idx         = np.where(mask)[0][ok]
            out[out_idx]    = parsed[ok]
            mask[out_idx]   = False                      # 변환 완료
        except Exception:
            continue
    return out                                           # 실패분 None 유지

# ── 메인 벡터 함수 ────────────────────────────────────────────────────
def vec_to_partial_week(col: pd.Series) -> pd.Series:
    """
    • 여러 포맷/널/공백 섞인 열 → partial-week 문자열 열  
    • 순서  
        1) sanitize (기존 v_sanitize_date_string)  
        2) `pd.to_datetime` (errors='coerce') → 대부분 해결  
        3) 남은 NaT 는 strptime 백업 시도  
        4) datetime → partial-week (NumPy vector)  
    """
    col = v_sanitize_date_string(col.astype(str))
    # 1차: pandas 내부 파서
    dt = pd.to_datetime(col, errors='coerce', dayfirst=False, format=None)

    # 아직 NaT 인 값 인덱스
    nat_mask = dt.isna().to_numpy()
    if nat_mask.any():
        # 원본 문자열 배열
        str_arr = col.to_numpy(dtype=object)
        # 백업 파싱
        parsed_arr = _try_strptime(str_arr[nat_mask])
        # 성공한 것만 업데이트
        dt_arr      = dt.to_numpy(dtype=object)
        dt_arr[nat_mask] = parsed_arr
        dt = pd.Series(dt_arr, index=col.index)

    # 최종 datetime64 배열
    dt64 = pd.to_datetime(dt, errors='coerce').to_numpy('datetime64[ns]')
    out  = np.full(dt64.shape, '', dtype=object)

    ok = dt64 != np.datetime64('NaT', 'ns')
    if ok.any():
        sec  = (dt64[ok] - np.datetime64('1970-01-01', 'ns')
               ) // np.timedelta64(1, 's')
        py_dt = np.vectorize(datetime.datetime.utcfromtimestamp, otypes=[object])(sec)
        out[ok] = np.vectorize(common.gfn_get_partial_week, otypes=[object])(py_dt, True)

    return pd.Series(out, index=col.index, dtype=object)

@_decoration_
def fn_step02_convert_date_to_partial_week() -> pd.DataFrame:
    """
    Step 2 : Step1의 Result에 Time을 Partial Week 으로 변환
    """
    df_fn_RTS_EOS = output_dataframes[str_df_fn_RTS_EOS]

    columns_to_convert = [
        RTS_INIT_DATE,
        RTS_DEV_DATE,
        RTS_COM_DATE,
        EOS_INIT_DATE,
        EOS_CHG_DATE,
        EOS_COM_DATE
    ]

    # Step 2: Convert to datetime and partial week
    for col in columns_to_convert:
        df_fn_RTS_EOS[col] = v_sanitize_date_string(df_fn_RTS_EOS[col])
        # df_return[col] = safe_strptime(df_return[col])
        # df_return[col] = to_partial_week(df_return[Item_Item],df_return[Sales_Domain_ShipTo],df_return[col])
        # df_return[col] = to_partial_week_datetime(df_return[col])
        df_fn_RTS_EOS[col] = df_fn_RTS_EOS[col].apply(lambda x: to_partial_week_datetime(x) if pd.notna(x) else '')
        df_fn_RTS_EOS[col].astype(str)

    # if speed up . use below.
    # df_fn_RTS_EOS[columns_to_convert] = df_fn_RTS_EOS[columns_to_convert].apply(vec_to_partial_week)
    # df_fn_RTS_EOS[COLS] = df_fn_RTS_EOS[COLS].astype(str)

    # Step 3: RTS_PARTIAL_WEEK
    df_fn_RTS_EOS[RTS_PARTIAL_WEEK] = np.where(
        df_fn_RTS_EOS[RTS_STATUS] == 'COM',
        df_fn_RTS_EOS[RTS_COM_DATE],
        np.where(
            df_fn_RTS_EOS[RTS_DEV_DATE].notna(),
            df_fn_RTS_EOS[RTS_DEV_DATE],
            df_fn_RTS_EOS[RTS_INIT_DATE]
        )
    )

    # Step 4: Item Level
    # df_fn_RTS_EOS[Item_Lv] = np.where(
    #     df_fn_RTS_EOS[Sales_Domain_ShipTo].astype(str).str.startswith("3"),
    #     3,
    #     2
    # )

    # Step 5: EOS_PARTIAL_WEEK
    df_fn_RTS_EOS[EOS_PARTIAL_WEEK] = np.where(
        df_fn_RTS_EOS[EOS_STATUS] == 'COM',
        np.where(
            df_fn_RTS_EOS[EOS_COM_DATE].notna(),
            df_fn_RTS_EOS[EOS_COM_DATE],
            np.where(
                df_fn_RTS_EOS[EOS_CHG_DATE].notna(),
                df_fn_RTS_EOS[EOS_CHG_DATE],
                df_fn_RTS_EOS[EOS_INIT_DATE]
            ),
        ),
        np.where(
            df_fn_RTS_EOS[EOS_CHG_DATE].notna(),
            df_fn_RTS_EOS[EOS_CHG_DATE],
            df_fn_RTS_EOS[EOS_INIT_DATE]
        )
    )

    # Step 6: Normalized week values
    df_fn_RTS_EOS[RTS_WEEK] = df_fn_RTS_EOS[RTS_PARTIAL_WEEK].astype(str).str.replace(r'\D', '', regex=True)
    df_fn_RTS_EOS[RTS_WEEK_MINUST_1] = df_fn_RTS_EOS[RTS_WEEK].apply(lambda x: common.gfn_add_week(x, -1) if pd.notna(x) and x != '' else '')
    df_fn_RTS_EOS[RTS_WEEK_PLUS_3] = df_fn_RTS_EOS[RTS_WEEK].apply(lambda x: common.gfn_add_week(x, 3) if pd.notna(x) and x != '' else '')
    df_fn_RTS_EOS[MAX_RTS_CURRENTWEEK] = df_fn_RTS_EOS[RTS_WEEK].apply(lambda x: max(x, current_week_normalized) if pd.notna(x) and x != '' else '')

    df_fn_RTS_EOS[EOS_WEEK] = df_fn_RTS_EOS[EOS_PARTIAL_WEEK].astype(str).str.replace(r'\D', '', regex=True)
    df_fn_RTS_EOS[EOS_WEEK_MINUS_1] = df_fn_RTS_EOS[EOS_WEEK].apply(lambda x: common.gfn_add_week(x, -1) if pd.notna(x) and x != '' else ''  )
    df_fn_RTS_EOS[EOS_WEEK_MINUS_4] = df_fn_RTS_EOS[EOS_WEEK].apply(lambda x: common.gfn_add_week(x, -4) if pd.notna(x) and x != '' else '')

    df_fn_RTS_EOS[RTS_INITIAL_WEEK] = df_fn_RTS_EOS[RTS_INIT_DATE].astype(str).str.replace(r'\D', '', regex=True)
    df_fn_RTS_EOS[EOS_INITIAL_WEEK] = df_fn_RTS_EOS[EOS_INIT_DATE].astype(str).str.replace(r'\D', '', regex=True)
    df_fn_RTS_EOS[MIN_EOSINI_MAXWEEK] = df_fn_RTS_EOS[EOS_INITIAL_WEEK].apply(lambda x: min(x, max_week_normalized) if pd.notna(x) and x != '' else '')
    df_fn_RTS_EOS[MIN_EOS_MAXWEEK] = df_fn_RTS_EOS[EOS_WEEK].apply(lambda x: min(x, max_week_normalized) if pd.notna(x) and x != '' else '')

    convert_to_int = [
        RTS_WEEK,
        RTS_WEEK_MINUST_1,
        RTS_WEEK_PLUS_3,
        MAX_RTS_CURRENTWEEK,
        EOS_WEEK,
        EOS_WEEK_MINUS_1,
        EOS_WEEK_MINUS_4,
        RTS_INITIAL_WEEK,
        EOS_INITIAL_WEEK,
        MIN_EOSINI_MAXWEEK
    ]

    for col in convert_to_int:
        df_fn_RTS_EOS[col] = df_fn_RTS_EOS[col].replace('','0')
        df_fn_RTS_EOS[col].fillna(0,inplace=True)
        df_fn_RTS_EOS[col] = df_fn_RTS_EOS[col].astype('int32')


    # # Step 6: Normalized week values (digit-only strings)
    # df_return['RTS_WEEK_NORMALIZED'] = df_return['RTS_PARTIAL_WEEK'].astype(str).str.replace(r'\D', '', regex=True)
    # df_return['EOS_WEEK_NORMALIZED'] = df_return['EOS_PARTIAL_WEEK'].astype(str).str.replace(r'\D', '', regex=True)# Use vectorized add_week
    # df_return['RTS_WEEK_NORMALIZED_MINUST_1'] = v_add_week(df_return['RTS_WEEK_NORMALIZED'], -1)
    # df_return['RTS_WEEK_NORMALIZED_PLUS_3'] = v_add_week(df_return['RTS_WEEK_NORMALIZED'], 3)

    # df_return['EOS_WEEK_NORMALIZED_MINUS_1'] = v_add_week(df_return['EOS_WEEK_NORMALIZED'], -1)
    # df_return['EOS_WEEK_NORMALIZED_MINUS_4'] = v_add_week(df_return['EOS_WEEK_NORMALIZED'], -4)

    # # Use vectorized max
    # df_return['MAX_RTS_CURRENTWEEK'] = v_max_week(df_return['RTS_WEEK_NORMALIZED'])

    # # Normalize initial week values
    # df_return['RTS_INITIAL_WEEK_NORMALIZED'] = df_return['RTS_INIT_DATE'].astype(str).str.replace(r'\D', '', regex=True)
    # df_return['EOS_INITIAL_WEEK_NORMALIZED'] = df_return['EOS_INIT_DATE'].astype(str).str.replace(r'\D', '', regex=True)

    # Use np.minimum for min
    # df_return['MIN_EOSINI_MAXWEEK'] = np.minimum(df_return['EOS_INITIAL_WEEK_NORMALIZED'], max_week_normalized)


    return df_fn_RTS_EOS


@_decoration_
def fn_step03_join_rts_eos() -> pd.DataFrame:
    """
    Step 3. df_in_MST_RTS 와 df_in_MST_EOS의 ITEM * ShipTo로 Inner Join하여 새로운 DF 생성 ( Output Data )
        첫번째 생성된 DF 에서 컬럼을 복사
    """
    fn_step03_join_rts_eos = output_dataframes[str_df_fn_RTS_EOS]
    df_origin = fn_step03_join_rts_eos.copy(deep=True)
    df_return = df_origin[[Item_Item,Sales_Domain_ShipTo]]

    return df_return


@_decoration_
def fn_step04_add_partialweek_measurecolumn() -> pd.DataFrame:
    """
    Step 4  : Step3의 df에 Partial Week 및 Measure Column 추가
    """
    df_fn_RTS_EOS =  output_dataframes[str_df_fn_RTS_EOS]
    df_in_Time_Partial_Week = input_dataframes[str_df_in_Time_Partial_Week]
    df_fn_RTS_EOS['key'] = 1
    df_in_Time_Partial_Week['key'] = 1

    # Perform the merge on the temporary key
    df_fn_RTS_EOS_Week = pd.merge(
        df_fn_RTS_EOS[[
            Item_Item,
            Sales_Domain_ShipTo,
            'key'
        ]],
        df_in_Time_Partial_Week,
        on='key'
    )
    df_fn_RTS_EOS_Week[Item_Lv] = df_fn_RTS_EOS_Week[Sales_Domain_ShipTo].apply(lambda x: 3 if x.startswith("3") else 2)
    df_fn_RTS_EOS_Week[CURRENT_ROW_WEEK] = df_fn_RTS_EOS_Week[Partial_Week].apply(normalize_week)
    df_fn_RTS_EOS_Week[CURRENT_ROW_WEEK_PLUS_8] = df_fn_RTS_EOS_Week[CURRENT_ROW_WEEK].apply(lambda x : common.gfn_add_week(x, 8))
    
    df_fn_RTS_EOS_Week[Lock_Condition] = True
    df_fn_RTS_EOS_Week[SIn_FCST_Color_Condition] = COLOR_GRAY
    df_fn_RTS_EOS_Week = df_fn_RTS_EOS_Week.drop(columns=['key']).reset_index(drop=True)

    # df_fn_RTS_EOS.drop(columns=['key'],inplace=True).reset_index(inplace=True)
    df_in_Time_Partial_Week.drop(columns=['key'],inplace=True)
    df_in_Time_Partial_Week.reset_index(inplace=True)


    # expanded_rows = []
    # for index, row in df_03_joined_rts_eos.iterrows():
    #     for time_value in input_dataframes[str_df_in_Time_Partial_Week]['Time.[Partial Week]']:
    #         new_row = row.to_dict()
    #         new_row['Time.[Partial Week]'] = time_value
    #         new_row['S/In FCST(GI)_GC.Lock'] = 'True'  # Placeholder value, replace with actual logic if needed
    #         new_row['S/In FCST Color Condition'] = 'GRAY'
    #         expanded_rows.append(new_row)
    
    
    # df_return = pd.DataFrame(expanded_rows)

    convert_to_int = [
        CURRENT_ROW_WEEK,
        CURRENT_ROW_WEEK_PLUS_8
    ]

    for col in convert_to_int:
        df_fn_RTS_EOS_Week[col] = df_fn_RTS_EOS_Week[col].replace('','0')
        df_fn_RTS_EOS_Week[col].fillna(0,inplace=True)
        df_fn_RTS_EOS_Week[col] = df_fn_RTS_EOS_Week[col].astype('int32')

    return df_fn_RTS_EOS_Week



@_decoration_
def fn_step05_set_lock_values():
    """
    Step 05: Forecast Measure Lock Color using array-based lookup
    
    This function relies on the following global variables:
      - output_dataframes (dict)
      - current_week_normalized (str)
      - max_week_normalized (str)

    Process:
      1. Load df_fn_RTS_EOS_Week from output_dataframes['df_04_partialweek_measurecolumn'].
      2. Load df_fn_RTS_EOS (the lookup DataFrame) from output_dataframes['df_02_date_to_partial_week'].
      3. Set up an index on df_fn_RTS_EOS for quick array-based lookups.
      4. For each needed column (e.g. 'RTS_WEEK_NORMALIZED_MINUST_1'), create
         a NumPy array that aligns with df_fn_RTS_EOS_Week's rows.
      5. Build your condition arrays in a vectorized way, and apply color/lock
         updates in df_fn_RTS_EOS_Week directly.
    
    Returns:
      pd.DataFrame: updated df_fn_RTS_EOS_Week with 'SIn_FCST(GI)_GC.Lock'
                    and 'SIn_FCST Color Condition' set.
    """

    # -----------------------------
    # 1. Load the partial-week DataFrame
    # -----------------------------
    df_week = output_dataframes[str_df_fn_RTS_EOS_Week]

    # Keep only the columns needed for logic (plus the GC columns we want to set)
    needed_cols_week = [
        Sales_Domain_ShipTo,
        Item_Item,
        Item_Lv,
        CURRENT_ROW_WEEK,
        CURRENT_ROW_WEEK_PLUS_8,
        Lock_Condition,
        SIn_FCST_Color_Condition
    ]
    # df_week = df_week[needed_cols_week]

    # Convert CURRENT_ROW_WEEK to int for numeric comparisons
    df_week[CURRENT_ROW_WEEK] = (
        df_week[CURRENT_ROW_WEEK]
        .fillna('0')
        .astype(int)
    )

    # -----------------------------
    # 2. Load the lookup DataFrame (df_fn_RTS_EOS)
    # -----------------------------
    df_lookup = output_dataframes[str_df_fn_RTS_EOS]
    # The column names in your lookup can remain raw strings,
    # or you can define more constants if desired. Here we keep
    # them as strings or minimal new constants:
    needed_cols_lookup = [
        Sales_Domain_ShipTo,
        Item_Item,
        Item_Lv,
        MAX_RTS_CURRENTWEEK,
        RTS_WEEK,
        MIN_EOSINI_MAXWEEK,
        EOS_WEEK,
        EOS_WEEK_MINUS_4,
        RTS_WEEK_MINUST_1,
        RTS_WEEK_PLUS_3
    ]
    # df_lookup = df_lookup[needed_cols_lookup]

    # -----------------------------
    # 3. Set the index in df_lookup for array-based lookups
    # -----------------------------
    # df_lookup.set_index([Sales_Domain_ShipTo, Item_Item, Item_Lv],inplace=True)  # 이렇게 해도 무방
    df_lookup_idx = df_lookup.set_index(
        [Sales_Domain_ShipTo, Item_Item, Item_Lv], 
        drop=False       # 필요하면 유지
    )

    # We'll match on (Sales_Domain_Ship_To, Item_Item, Item_lv) from df_week
    multi_index_week = pd.MultiIndex.from_arrays(
        [
            df_week[Sales_Domain_ShipTo],
            df_week[Item_Item],
            df_week[Item_Lv]
        ],
        names=[Sales_Domain_ShipTo, Item_Item, Item_Lv]
    )

    # positions: integer array the same length as df_week
    positions = df_lookup_idx.index.get_indexer(multi_index_week)
    valid_mask = (positions != -1)

    # -----------------------------
    # 4. Build arrays from df_lookup columns
    # -----------------------------
    max_rts_array           = pd.to_numeric(df_lookup[MAX_RTS_CURRENTWEEK], errors='coerce').fillna(0).astype(int)
    rts_week_array          = pd.to_numeric(df_lookup[RTS_WEEK], errors='coerce').fillna(0).astype(int)
    rts_init_array          = pd.to_numeric(df_lookup[RTS_INITIAL_WEEK], errors='coerce').fillna(0).astype(int)
    min_eos_maxweek_arr     = pd.to_numeric(df_lookup[MIN_EOS_MAXWEEK], errors='coerce').fillna(0).astype(int)
    eos_week_array          = pd.to_numeric(df_lookup[EOS_WEEK], errors='coerce').fillna(0).astype(int)
    eos_week_minus1_array   = pd.to_numeric(df_lookup[EOS_WEEK_MINUS_1], errors='coerce').fillna(0).astype(int)
    eos_week_minus4_array   = pd.to_numeric(df_lookup[EOS_WEEK_MINUS_4], errors='coerce').fillna(0).astype(int)
    rts_week_minus1_array   = pd.to_numeric(df_lookup[RTS_WEEK_MINUST_1], errors='coerce').fillna(0).astype(int)
    rts_week_plus3_array    = pd.to_numeric(df_lookup[RTS_WEEK_PLUS_3], errors='coerce').fillna(0).astype(int)

    n = len(df_week)
    arr_max_rts            = np.full(n, np.nan, dtype=int)
    arr_rts_week           = np.full(n, np.nan, dtype=int)
    arr_rts_init           = np.full(n, np.nan, dtype=int)
    arr_min_eos_maxweek = np.full(n, np.nan, dtype=int)
    arr_eos_week           = np.full(n, np.nan, dtype=int)
    arr_eos_week_minus1    = np.full(n, np.nan, dtype=int)
    arr_eos_week_minus4    = np.full(n, np.nan, dtype=int)
    arr_rts_week_minus1    = np.full(n, np.nan, dtype=int)
    arr_rts_week_plus3     = np.full(n, np.nan, dtype=int)

    # Fill valid rows
    arr_max_rts[valid_mask]            = max_rts_array[positions[valid_mask]]
    arr_rts_week[valid_mask]           = rts_week_array[positions[valid_mask]]
    arr_rts_init[valid_mask]           = rts_init_array[positions[valid_mask]]
    arr_min_eos_maxweek[valid_mask]    = min_eos_maxweek_arr[positions[valid_mask]]
    arr_eos_week[valid_mask]           = eos_week_array[positions[valid_mask]]
    arr_eos_week_minus1[valid_mask]    = eos_week_minus1_array[positions[valid_mask]]
    arr_eos_week_minus4[valid_mask]    = eos_week_minus4_array[positions[valid_mask]]
    arr_rts_week_minus1[valid_mask]    = rts_week_minus1_array[positions[valid_mask]]
    arr_rts_week_plus3[valid_mask]     = rts_week_plus3_array[positions[valid_mask]]

    # -----------------------------
    # 5. Vectorized condition checks
    # -----------------------------
    row_partial = df_week[CURRENT_ROW_WEEK].to_numpy(dtype=int)
    curr_week_int = int(current_week_normalized)
    max_week_int  = int(max_week_normalized)

    # Step 5-1: White color
    cond_white = (
        (row_partial >= curr_week_int) &
        (arr_max_rts <= row_partial) &
        (row_partial <= arr_min_eos_maxweek)
    )
    df_week.loc[cond_white, Lock_Condition] = False
    df_week.loc[cond_white, SIn_FCST_Color_Condition] = COLOR_WHITE

    # Step 5-2: Dark Blue
    cond_darkblue = (
        (row_partial >= curr_week_int) &
        (arr_rts_init <= row_partial) &
        (row_partial <= arr_rts_week_minus1)
    )
    df_week.loc[cond_darkblue, Lock_Condition] = True
    df_week.loc[cond_darkblue, SIn_FCST_Color_Condition] = COLOR_DARKBLUE

    # Step 5-3: Light Blue
    cond_lightblue = (
        (row_partial >= curr_week_int) &
        (arr_rts_week <= row_partial) &
        (row_partial <= arr_rts_week_plus3)
    )
    df_week.loc[cond_lightblue, Lock_Condition] = False
    df_week.loc[cond_lightblue, SIn_FCST_Color_Condition] = COLOR_LIGHTBLUE

    # Step 5-4: Light Red
    cond_lightred = (
        (row_partial >= curr_week_int) &
        (arr_eos_week_minus4 <= row_partial) &
        (row_partial <= (arr_eos_week_minus1))
    )
    df_week.loc[cond_lightred, Lock_Condition] = False
    df_week.loc[cond_lightred, SIn_FCST_Color_Condition] = COLOR_LIGHTRED

    # Step 5-5: Dark Red
    cond_darkred = (
        (row_partial >= curr_week_int) &
        (arr_eos_week <= row_partial) &
        (row_partial <= max_week_int)
    )
    df_week.loc[cond_darkred, Lock_Condition] = True
    df_week.loc[cond_darkred, SIn_FCST_Color_Condition] = COLOR_DARKRED

    # -----------------------------
    # 6. Return
    # -----------------------------
    # return df_week.reset_index(drop=True)

    """
        # 이단계에서는 True False 가 확실히 됨.
        import duckdb
        duckdb.register('df_week', df_week)
        query = f'''
            select * from df_week 
            where "{Item_Item}" = 'SM-A546UZKBXAG'
            and "{Partial_Week}" >= '202501'
            and "{Partial_Week}" <= '202521'
            -- and "{Partial_Week}" = '300114'
            -- and "{Partial_Week}" = '408351'
            -- and "{Partial_Week}" = '5006941'
        '''
        duckdb.query(query).show()
        ┌────────────────┬────────────────────────┬─────────────────────┬─────────┬─────────────────────────────────────┬───────────────────────────────┬────────────────┬───────────────────────────┐
        │  Item.[Item]   │ Sales Domain.[Ship To] │ Time.[Partial Week] │ Item_Lv │ current_row_partial_week_normalized │ CURRENTWEEK_NORMALIZED_PLUS_8 │ Lock Condition │ S/In FCST Color Condition │
        │    varchar     │        varchar         │       varchar       │  int64  │                int32                │             int32             │    boolean     │          varchar          │
        ├────────────────┼────────────────────────┼─────────────────────┼─────────┼─────────────────────────────────────┼───────────────────────────────┼────────────────┼───────────────────────────┤
        │ SM-A546UZKBXAG │ 300114                 │ 202505B             │       3 │                              202505 │                        202513 │ true           │ 19_GRAY                   │
        │ SM-A546UZKBXAG │ 300114                 │ 202506A             │       3 │                              202506 │                        202514 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202507A             │       3 │                              202507 │                        202515 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202508A             │       3 │                              202508 │                        202516 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202509A             │       3 │                              202509 │                        202517 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202509B             │       3 │                              202509 │                        202517 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202510A             │       3 │                              202510 │                        202518 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202511A             │       3 │                              202511 │                        202519 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202512A             │       3 │                              202512 │                        202520 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202513A             │       3 │                              202513 │                        202521 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202514A             │       3 │                              202514 │                        202522 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202514B             │       3 │                              202514 │                        202522 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202515A             │       3 │                              202515 │                        202523 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202516A             │       3 │                              202516 │                        202524 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202517A             │       3 │                              202517 │                        202525 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202518A             │       3 │                              202518 │                        202526 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202518B             │       3 │                              202518 │                        202526 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202519A             │       3 │                              202519 │                        202527 │ false          │ 14_WHITE                  │
        │ SM-A546UZKBXAG │ 300114                 │ 202520A             │       3 │                              202520 │                        202528 │ false          │ 14_WHITE                  │
        ├────────────────┴────────────────────────┴─────────────────────┴─────────┴─────────────────────────────────────┴───────────────────────────────┴────────────────┴───────────────────────────┤
        │ 19 rows                                                                                                                                                                          8 columns │
        └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

    """
    
    return df_week

@_decoration_
def fn_step06_addcolumn_green_for_wireless_bas_array_based():
    """
    Step 6 (array-based): Add or update 'S/In FCST Color Condition' to '13_GREEN'
    for 'wireless BAS' items, using pre-existing CURRENT_ROW_WEEK_PLUS_8
    to avoid calling gfn_add_week. No merges required.
    """

    # ------------------------------------------------------------------
    # 1) Load the main DataFrame we want to update, e.g. df_week
    #    Suppose it has these columns:
    #       CURRENT_ROW_WEEK          (int)
    #       CURRENT_ROW_WEEK_PLUS_8   (int)
    #       'S/In FCST Color Condition'
    #       'Item.[Item]'  -> We'll use array-based get_indexer with df_in_Item_Master
    # ------------------------------------------------------------------
    df_week = output_dataframes[str_df_fn_RTS_EOS_Week]  # or 'df_04_partialweek_measurecolumn', etc.

    # Ensure CURRENT_ROW_WEEK is int
    df_week[CURRENT_ROW_WEEK] = df_week[CURRENT_ROW_WEEK].fillna('0').astype(int)
    df_week[CURRENT_ROW_WEEK_PLUS_8] = df_week[CURRENT_ROW_WEEK_PLUS_8].fillna('0').astype(int)

    # ------------------------------------------------------------------
    # 2) Load & index your Item Master for array-based lookups
    # ------------------------------------------------------------------
    df_item_master = input_dataframes[str_df_in_Item_Master]

    # Example column references (adapt to your code):
    #   'Item.[Item]' -> Item_Item
    #   'Item.[Item Type]' -> Item_Type
    #   'Item.[Item GBM]'  -> Item_GBM
    #
    # Make Item_Item the index so we can do get_indexer lookups
    df_item_master.set_index(Item_Item, inplace=True)

    # We want to look up each row's item in df_week
    item_array_in_week = df_week[Item_Item].to_numpy()

    # get_indexer returns the positions in df_item_master for each item in item_array_in_week
    positions = df_item_master.index.get_indexer(item_array_in_week)
    valid_mask = (positions != -1)

    # ------------------------------------------------------------------
    # 3) Create arrays for item attributes from df_item_master
    # ------------------------------------------------------------------
    item_type_array = df_item_master[Item_Type].to_numpy(dtype=str)
    item_gbm_array  = df_item_master[Item_GBM].to_numpy(dtype=str)
    item_group_array  = df_item_master[Product_Group].to_numpy(dtype=str)

    n = len(df_week)
    out_item_type = np.full(n, '', dtype=object)
    out_item_gbm  = np.full(n, '', dtype=object)
    out_item_group  = np.full(n, '', dtype=object)

    out_item_type[valid_mask] = item_type_array[positions[valid_mask]]
    out_item_gbm[valid_mask]  = item_gbm_array[positions[valid_mask]]
    out_item_group[valid_mask]  = item_group_array[positions[valid_mask]]

    # # Optionally store them back to df_week
    # df_week[Item_Type] = out_item_type
    df_week[Item_GBM]  = out_item_gbm
    df_week[Product_Group]  = out_item_group

    # ------------------------------------------------------------------
    # 4) Build your mask using CURRENT_ROW_WEEK and CURRENT_ROW_WEEK_PLUS_8
    # ------------------------------------------------------------------
    row_partial_array = df_week[CURRENT_ROW_WEEK].to_numpy()
    row_partial_plus8_array = df_week[CURRENT_ROW_WEEK_PLUS_8].to_numpy()

    # Let's say "green" range is from current_week_normalized up to CURRENT_ROW_WEEK_PLUS_8
    # (adjust logic as needed)
    curr_week_int = int(current_week_normalized)

    mask_wireless_bas = (
        (out_item_type == 'BAS') &
        (out_item_gbm  == 'MOBILE') &
        (row_partial_array >= curr_week_int) &
        (row_partial_array <= row_partial_plus8_array) &
        (df_week[SIn_FCST_Color_Condition] != COLOR_GRAY)
    )

    # ------------------------------------------------------------------
    # 5) Apply the color update in-place
    # ------------------------------------------------------------------
    df_week.loc[mask_wireless_bas, SIn_FCST_Color_Condition] = COLOR_GREEN

    # If you no longer need the item type columns, you can drop them
    # df_week.drop(columns=['Item.[Item Type]', 'Item.[Item GBM]'], inplace=True)

    # ------------------------------------------------------------------
    # 6) Return the updated DataFrame
    # ------------------------------------------------------------------
    df_item_master.reset_index(inplace=True)

    column_returns = [
        Sales_Domain_ShipTo ,
        Item_Item ,
        Item_Lv ,
        Item_GBM ,
        Product_Group ,
        Partial_Week ,
        CURRENT_ROW_WEEK ,
        Lock_Condition ,
        SIn_FCST_Color_Condition
    ]

    return df_week[column_returns]

@_decoration_
def step07_prepare_asn(df_asn: pd.DataFrame,
                       df_itemmst: pd.DataFrame,
                       df_dim: pd.DataFrame,
                       df_time: pd.DataFrame,
                       current_partial_week: str) -> pd.DataFrame:
    """
    df_in_Sales_Product_ASN 전처리.

    • ASN 원본 → (Version, Sales_Product_ASN) 제거.
    • Ship-To → Lv2/Lv3/Lv6 vector 추가.
    • Item Master 로 Item_GBM / Product_Group 매핑.
    • df_time 과 Cross-join → 모든 Partial Week 생성.
    • 주차 < CurrentWeek → Lock=True·19_GRAY / 그 이상 → False·14_WHITE
    ------------------------------------------------------------------------------
    """

    # ── 0) 준비 ──────────────────────────────────────
    cur_norm = int(normalize_week(current_partial_week))
    weeks    = df_time[Partial_Week].to_numpy(dtype=object)         # e.g. 53개
    n_weeks  = len(weeks)    # ── 1) ASN 정리 ──────────────────────────────────
    df_asn = df_asn.drop(columns=[Version_Name, Salse_Product_ASN], errors='ignore')

    # 1-1) ItemMaster join (vectorised)
    item_idx = df_itemmst.set_index(Item_Item)[[Item_GBM, Product_Group]]
    pos_it   = item_idx.index.get_indexer(df_asn[Item_Item])
    itm_gbm  = np.where(pos_it >= 0, item_idx[Item_GBM].to_numpy()[pos_it], None)
    itm_grp  = np.where(pos_it >= 0, item_idx[Product_Group].to_numpy()[pos_it], None)
    df_asn   = df_asn.assign(**{Item_GBM:itm_gbm, Product_Group:itm_grp})

    # 1-2) Ship-To level 파생(Lv2/Lv3/Lv6) -------------
    dim_idx = df_dim.set_index(Sales_Domain_ShipTo)
    lv2 = dim_idx[Sales_Domain_LV2].to_numpy()
    lv3 = dim_idx[Sales_Domain_LV3].to_numpy()
    lv6 = dim_idx[Sales_Domain_LV6].to_numpy()
    pos  = dim_idx.index.get_indexer(df_asn[Sales_Domain_ShipTo].to_numpy())
    df_asn = df_asn.assign(
        **{Sales_Domain_LV2: lv2[pos],
           Sales_Domain_LV3: lv3[pos],
           Sales_Domain_LV6: lv6[pos]}
    )

    # ── 2) Cross-join with Weeks (repeat/tiling) ──────
    base_n   = len(df_asn)
    df_rep   = df_asn.loc[df_asn.index.repeat(n_weeks)].reset_index(drop=True)
    df_rep[Partial_Week] = np.tile(weeks, base_n)

    # ── 3) Lock / Color 초기화 ─────────────────────────
    wk_norm  = np.fromiter((int(normalize_week(w)) for w in df_rep[Partial_Week]),
                            dtype='int32')
    past_mask       = wk_norm < cur_norm
    df_rep[Lock_Condition]   = past_mask        # bool
    df_rep[Color_Condition]  = np.where(past_mask, COLOR_GRAY, COLOR_WHITE)

    return df_rep

@_decoration_
def fn_step08_match_rts(df_asn_week: pd.DataFrame,
                        df_rts_week: pd.DataFrame,
                        df_dim: pd.DataFrame) -> pd.DataFrame:
    """
    ASN만 있는 경우 조건 추가
    Sales Product ASN에 Partial Week 에 따른 Lock 값과, Color 값 적용 ( Step 7에 Step 6를 적용 )

    • df_asn_week  : step07_prepare_asn 결과 (Lv6/Lv7 + 주차 Fan-out, 기본 Lock=False·14_WHITE)
    • df_rts_week  : step05/06 결과 (Lv2/Lv3 주차별 Lock·Color 완료본)
    ------------------------------------------------------------------------------
       Lv7 또는 Lv6 → 부모 Lv3/2 를 계산해서 df_rts_week 의 Lock,Color 를 *벡터*로
       가져온 뒤 override.
       - 매칭 실패 → Lock/Color 그대로(Default)
       - 매칭 성공 → df_rts_week 의 값으로 덮어쓰고 RTS_EOS_ShipTo 컬럼에 부모코드 기록
    """
    tgt           = df_asn_week.copy()
    dim_idx       = df_dim.set_index(Sales_Domain_ShipTo)
    lv2_arr, lv3_arr = (dim_idx[Sales_Domain_LV2].to_numpy(),
                        dim_idx[Sales_Domain_LV3].to_numpy())
    pos_dim       = dim_idx.index.get_indexer(tgt[Sales_Domain_ShipTo])
    has_dim       = pos_dim >= 0    # ── RTS/EOS side  MultiIndex  ────────────────────────────────────────
    rts_idx = df_rts_week.set_index(
        [Item_Item, Sales_Domain_ShipTo, Partial_Week],
        verify_integrity=False)

    rts_lock  = rts_idx[Lock_Condition].to_numpy()
    rts_color = rts_idx[Color_Condition].to_numpy()

    # ── 준비 : 부모코드 두 개 벡터 ─────────────────────────────────────────
    parent_lv2 = np.where(has_dim, lv2_arr[pos_dim], None)
    parent_lv3 = np.where(has_dim, lv3_arr[pos_dim], None)

    # ── look-up Lv3 → Lv2 우선순위로 get_indexer ─────────────────────────
    def fetch(col_arr):
        """returns ndarray[object|None] with rts value aligned to tgt rows"""
        pos = rts_idx.index.get_indexer(
            np.stack([tgt[Item_Item], col_arr, tgt[Partial_Week]], axis=1))
        hit = pos >= 0
        out = np.full(len(tgt), None, dtype=object)
        out[hit] = col_arr[hit]          # store parent ship-to for later
        return hit, pos, out

    hit3, pos3, ship3 = fetch(parent_lv3)
    # Lv3 가 못찾은 row 중 Lv2 재시도
    mask2 = ~hit3
    hit2, pos2, ship2 = fetch(parent_lv2)

    # ── 덮어쓰기 (Logical OR hit) ────────────────────────────────────────
    hit_all     = hit3 | hit2
    rts_pos     = np.where(hit3, pos3, pos2)       # pick the matched index
    src_lock    = np.where(hit_all, rts_lock[rts_pos], tgt[Lock_Condition])
    src_color   = np.where(hit_all, rts_color[rts_pos], tgt[Color_Condition])
    parent_ship = np.where(hit3, ship3, ship2)

    tgt.loc[hit_all, Lock_Condition]   = src_lock[hit_all]
    tgt.loc[hit_all, Color_Condition]  = src_color[hit_all]
    tgt[RTS_EOS_ShipTo] = parent_ship   # NaN for unmatched

    return tgt


# ────────────────────────────────────────────────────────────────────
# Step-09  :  HA-EOP 12_YELLOW / Lock False   (전량 벡터화 - v2)
# ────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step09_apply_itemclass(
    df_fcst    : pd.DataFrame,   # ← step-08 (ASN-Item-Week) 결과
    df_itemcls : pd.DataFrame,   # ← df_in_Item_CLASS (Item_Class == 'X')
    df_rts_eos : pd.DataFrame,   # ← step-02 (EOS_WEEK*, …)
    df_rts_week: pd.DataFrame,   # ← step-04 (CURRENT_ROW_WEEK 보유)
    df_dim     : pd.DataFrame,   # ← df_in_Sales_Domain_Dimension
    cur_partial: str             # ← 예) '202447A'
) -> pd.DataFrame:    
    cur_norm = int(normalize_week(cur_partial))

    # ──────────────────────────────────────────────────────────────
    # 1)  Lv-6 Ship-To 벡터 가져오기  (df_fcst 와 동일 순서)
    # ──────────────────────────────────────────────────────────────
    dim_idx = df_dim.set_index(Sales_Domain_ShipTo)
    lv6_vec = dim_idx[Sales_Domain_LV6] \
                .reindex(df_fcst[Sales_Domain_ShipTo]) \
                .to_numpy(dtype=object)                        # len(df_fcst)

    # ──────────────────────────────────────────────────────────────
    # 2)  ITEM CLASS(X) 룩업-테이블  (Item, Lv-6 Ship-To, Location)
    #     → set -> MultiIndex 로 전환  (get_indexer 용)
    # ──────────────────────────────────────────────────────────────
    df_iclass_x = df_itemcls.loc[df_itemcls[Item_Class] == 'X',
                                 [Item_Item, Sales_Domain_ShipTo, Location_Location]]

    if df_iclass_x.empty:
        return df_fcst           # Item_Class=='X' 데이터가 없으면 그대로 반환

    iclass_index = (
        df_iclass_x
        .drop_duplicates()
        .set_index([Item_Item, Sales_Domain_ShipTo, Location_Location])
        .index
    )

    # ──────────────────────────────────────────────────────────────
    # 3)  df_fcst 의 (Item, Lv-6 Ship-To, Location) 를 한꺼번에 매칭
    # ──────────────────────────────────────────────────────────────
    mi_fcst = pd.MultiIndex.from_arrays(
        [
            df_fcst[Item_Item].to_numpy(),
            lv6_vec,                                   # lv-6 코드
            df_fcst[Location_Location].to_numpy()
        ],
        names=[Item_Item, Sales_Domain_ShipTo, Location_Location]
    )

    pos_iclass = iclass_index.get_indexer(mi_fcst)
    valid_icls = pos_iclass >= 0                     # ITEM CLASS(X) 가 존재하는 행

    # ──────────────────────────────────────────────────────────────
    # 4)  CURRENT_ROW_WEEK  (step-04) 빠른 벡터-룩업
    # ──────────────────────────────────────────────────────────────
    wk_idx = df_rts_week.set_index(
        [Item_Item, Sales_Domain_ShipTo, Partial_Week]
    )

    pos_wk = wk_idx.index.get_indexer(
        pd.MultiIndex.from_arrays(
            [
                df_fcst[Item_Item],
                df_fcst[RTS_EOS_ShipTo],       # Lv-2/3 Ship-To
                df_fcst[Partial_Week]
            ]
        )
    )
    valid_wk = pos_wk >= 0
    wk_norm  = np.full(len(df_fcst), 0, dtype='int32')
    if valid_wk.any():
        wk_norm[valid_wk] = wk_idx[CURRENT_ROW_WEEK] \
                                .to_numpy(dtype='int32')[pos_wk[valid_wk]]

    # ──────────────────────────────────────────────────────────────
    # 5)  EOS_WEEK / EOS_WEEK-4 벡터-룩업
    # ──────────────────────────────────────────────────────────────
    eo_idx = df_rts_eos.set_index([Item_Item, Sales_Domain_ShipTo])
    pos_eo = eo_idx.index.get_indexer(
        pd.MultiIndex.from_arrays(
            [df_fcst[Item_Item], df_fcst[RTS_EOS_ShipTo]]
        )
    )
    valid_eo = pos_eo >= 0

    eos_week   = np.full(len(df_fcst), 999_999, dtype='int32')
    eos_m4week = np.full(len(df_fcst), 999_999, dtype='int32')
    if valid_eo.any():
        eo_vals = eo_idx[[EOS_WEEK, EOS_WEEK_MINUS_4]].to_numpy(dtype='int32')
        eos_week  [valid_eo] = eo_vals[pos_eo[valid_eo], 0]
        eos_m4week[valid_eo] = eo_vals[pos_eo[valid_eo], 1]

    # ──────────────────────────────────────────────────────────────
    # 6)  최종 조건 (YELLOW & Lock=False)
    #     - ITEM CLASS(X) 매칭 + WEEK 조건 + 기존 색상 필터
    # ──────────────────────────────────────────────────────────────
    wk_ge_curr = wk_norm >= cur_norm

    cond_a = (
        valid_icls & wk_ge_curr &
        (eos_week != 999_999) & (wk_norm <= eos_m4week)
    )
    cond_b = (
        valid_icls & wk_ge_curr &
        (eos_week == 999_999)                 # EOS 미존재
    )

    yellow_mask = (cond_a | cond_b) & \
                  ~df_fcst[Color_Condition].isin([COLOR_LIGHTRED, COLOR_DARKRED])

    # ──────────────────────────────────────────────────────────────
    # 7)  업데이트
    # ──────────────────────────────────────────────────────────────
    df_fcst.loc[yellow_mask, Color_Condition] = COLOR_YELLOW
    df_fcst.loc[yellow_mask, Lock_Condition ] = False

    return df_fcst

@_decoration_
def step10_1_vd_leadtime(
    df: pd.DataFrame,
    df_tat: pd.DataFrame,
    df_rts_week:pd.DataFrame,          # ← step04 결과  (CURRENT_ROW_WEEK 보유)
    current_partial_week: str
) -> pd.DataFrame:
    """
    ITEMTAT_TATTERM (LT) 만큼 : Lock=False, Color=18_DGRAY_RED  
    단, 기존 Color 가 14_WHITE 인 곳만 덮어쓴다.
    """
    cur_norm = int(normalize_week(current_partial_week))    # ── 1) 대상(GMB=VD|SHA) 필터 ───────────────────────
    vd_mask = df[Item_GBM].isin(['VD', 'SHA'])
    if not vd_mask.any():
        return df

    # ── 2) LT 벡터 만들기 (Item,Location index) ────────
    tat_idx = df_tat.set_index([Item_Item, Location_Location])[ITEMTAT_TATTERM]
    pos_tat = tat_idx.index.get_indexer(
        list(zip(df[Item_Item], df[Location_Location]))
    )
    tatterm = np.where(pos_tat >= 0, tat_idx.to_numpy()[pos_tat], 0).astype('int16')

    # ── 3) Week mask 계산 ──────────────────────────────
    # wk_norm = np.fromiter((int(normalize_week(w)) for w in df[Partial_Week]),
    #                        dtype='int32')

    wk_idx = df_rts_week.set_index([Item_Item, Sales_Domain_ShipTo, Partial_Week])
    # wk_idx = df_rts_week.index
    pos_wk = wk_idx.index.get_indexer(
        pd.MultiIndex.from_arrays([
            df[Item_Item],
            df[RTS_EOS_ShipTo],     # Lv2/3 Ship-To
            df[Partial_Week]
        ])
    )
    valid_wk     = pos_wk >= 0
    wk_norm      = np.full(len(df), 0, dtype='int32')
    wk_norm[valid_wk] = wk_idx[CURRENT_ROW_WEEK].to_numpy(dtype='int32')[pos_wk[valid_wk]]

    within_lt = (wk_norm - cur_norm) < tatterm
    apply_mask = vd_mask & (wk_norm >= cur_norm) & within_lt & (df[Color_Condition] == COLOR_WHITE)

    # ── 4) Update ─────────────────────────────────────
    df.loc[apply_mask, Lock_Condition]  = False
    df.loc[apply_mask, Color_Condition] = COLOR_DGRAY_RED
    return df


# ────────────────────────────────────────────────────────────────
# Step-10-2 : VD/SHA LT-SET 범위 → 17_DGRAY_REDB, Lock=False
#             (기존 18_DGRAY_RED 승격 / 14_WHITE 신규)
# ────────────────────────────────────────────────────────────────
@_decoration_
def step10_2_vd_set_leadtime(
    df: pd.DataFrame,
    df_tat: pd.DataFrame,
    df_rts_week: pd.DataFrame,        # ← step-04 결과 (CURRENT_ROW_WEEK 보유)
    current_partial_week: str
) -> pd.DataFrame:
    """
    • GBM = 'VD' or 'SHA' 인 Item 에 한하여
        ─ Color 가 18_DGRAY_RED → 17_DGRAY_REDB 승격
        ─ Color 가 14_WHITE     → 17_DGRAY_REDB 신규 지정, Lock=False
    • LT(ITEMTAT_TATTERM_SET) 범위 안에서만 적용
    """
    cur_norm = int(normalize_week(current_partial_week))    # ── 0) VD·SHA 필터(추가) ──────────────────────────────
    vd_mask = df[Item_GBM].isin(['VD', 'SHA'])
    if not vd_mask.any():
        return df                              # 해당 GBM 이 없으면 그대로 반환

    # ── 1) LT-SET 벡터 (Item, Location) ─────────────────
    tat_idx = (
        df_tat
        .set_index([Item_Item, Location_Location])[ITEMTAT_TATTERM_SET]
    )
    pos_tat = tat_idx.index.get_indexer(
        list(zip(df[Item_Item], df[Location_Location]))
    )
    tatterm_set = np.where(pos_tat >= 0,
                           tat_idx.to_numpy()[pos_tat],
                           0).astype('int16')

    # ── 2) CURRENT_ROW_WEEK 벡터-lookup ─────────────────
    wk_idx = df_rts_week.set_index(
        [Item_Item, Sales_Domain_ShipTo, Partial_Week]
    )
    pos_wk = wk_idx.index.get_indexer(
        pd.MultiIndex.from_arrays([
            df[Item_Item],
            df[RTS_EOS_ShipTo],          # Lv2/3 Ship-To
            df[Partial_Week]
        ])
    )
    valid_wk = pos_wk >= 0
    wk_norm  = np.full(len(df), 0, dtype='int32')
    if valid_wk.any():
        wk_norm[valid_wk] = wk_idx[CURRENT_ROW_WEEK] \
                                .to_numpy(dtype='int32')[pos_wk[valid_wk]]

    # ── 3) 범위 조건 계산 ────────────────────────────────
    within_set = (wk_norm - cur_norm) < tatterm_set
    base_mask  = vd_mask & (wk_norm >= cur_norm) & within_set

    mask_white = base_mask & (df[Color_Condition] == COLOR_WHITE)
    mask_dgray = base_mask & (df[Color_Condition] == COLOR_DGRAY_RED)

    # ── 4) 업데이트 ─────────────────────────────────────
    df.loc[mask_dgray, Color_Condition] = COLOR_DGRAY_REDB          # 승격
    df.loc[mask_white,  Color_Condition] = COLOR_DGRAY_REDB         # 신규
    df.loc[mask_white,  Lock_Condition]  = False

    return df

@_decoration_
def step11_apply_no_sellout(
    df: pd.DataFrame,
    df_no_sellout: pd.DataFrame,
    df_dim: pd.DataFrame,
    df_rts_week:pd.DataFrame,          # ← step04 결과  (CURRENT_ROW_WEEK 보유)
    current_partial_week: str
) -> pd.DataFrame:
    """
    Sell-out FCST 없는 3Lv(Std2) → 하위 6/7Lv 전체  
    현재 ~ 미래 Lock=True, Color=19_GRAY
    """
    cur_norm = int(normalize_week(current_partial_week))

    # ──────────────────────────────────────────────────────────────
    # 1)  Lv-3 Ship-To 벡터 가져오기  (df 와 동일 순서)
    # ──────────────────────────────────────────────────────────────
    dim_idx = df_dim.set_index(Sales_Domain_ShipTo)
    lv3_vec = dim_idx[Sales_Domain_LV3] \
                .reindex(df[Sales_Domain_ShipTo]) \
                .to_numpy(dtype=object)                        # len(df_fcst)

    # ──────────────────────────────────────────────────────────────
    # 2)  df_no_sellout 룩업-테이블  (Item, Lv-3 Item , Ship-To )
    #     → set -> MultiIndex 로 전환  (get_indexer 용)
    # ──────────────────────────────────────────────────────────────
    no_sellout_index = (
        df_no_sellout
        .drop_duplicates()
        .set_index([Item_Item, Sales_Std2])
        .index
    )

    # ──────────────────────────────────────────────────────────────
    # 3)  df 의 (Item, Lv-6 Ship-To) 를 한꺼번에 매칭
    # ──────────────────────────────────────────────────────────────
    mi_df = pd.MultiIndex.from_arrays(
        [
            df[Item_Item].to_numpy(),
            lv3_vec                                   # lv-3 코드
        ],
        names=[Item_Item, Sales_Domain_ShipTo]
    )

    pos_no_sellout = no_sellout_index.get_indexer(mi_df)
    valid_no_sellout = pos_no_sellout >= 0                     # ITEM CLASS(X) 가 존재하는 행

    # ──────────────────────────────────────────────────────────────
    # 4)  CURRENT_ROW_WEEK  (step-04) 빠른 벡터-룩업
    # ──────────────────────────────────────────────────────────────
    wk_idx = df_rts_week.set_index(
        [Item_Item, Sales_Domain_ShipTo, Partial_Week]
    )

    pos_wk = wk_idx.index.get_indexer(
        pd.MultiIndex.from_arrays(
            [
                df[Item_Item],
                df[RTS_EOS_ShipTo],       # Lv-2/3 Ship-To
                df[Partial_Week]
            ]
        )
    )
    valid_wk = pos_wk >= 0
    wk_norm  = np.full(len(df), 0, dtype='int32')
    if valid_wk.any():
        wk_norm[valid_wk] = wk_idx[CURRENT_ROW_WEEK] \
                                .to_numpy(dtype='int32')[pos_wk[valid_wk]]
    


    # ──────────────────────────────────────────────────────────────
    # 5)  최종 조건 (GRAY & Lock=True)
    #     - ShipTo,Item 
    # ──────────────────────────────────────────────────────────────
    apply_mask = valid_no_sellout & (wk_norm >= cur_norm)

    df.loc[apply_mask, [Lock_Condition]] = True
    df.loc[apply_mask, [Color_Condition]] = COLOR_GRAY
    return df

# ════════════════════════════════════════════════════════════
#  STEP-12  : Sell-In Forecast-Rule 적용   (GC / AP2 / AP1 / Color)
#  1.  core 로직 (Sell-In / Sell-Out 공용)
# ════════════════════════════════════════════════════════════
# -----------------------------------------------------------
#  step12_core  ─ Forecast-Rule 공용 엔진  (Sell-IN / Sell-OUT)
# -----------------------------------------------------------
def step12_core(df_src        : pd.DataFrame,
                rule_dict     : Dict[tuple[str,str], tuple[int,int,int]],
                ship_idx      : pd.Index,
                lv_codes_arr  : np.ndarray,               # shape(n_shipto, 6) – Lv2~Lv7
                sell_type     : str = "IN"                 # "IN" or "OUT"
               ) -> Tuple[pd.DataFrame, pd.DataFrame,
                          pd.DataFrame, pd.DataFrame]:
    """
    Forecast-Rule 적용 → GC/AP2/AP1 LOCK & COLOR  Raw DataFrame 4개 리턴
    (Version 컬럼은 wrapper(step12/13) 에서 삽입)
    """
    assert sell_type in ("IN", "OUT")

    # ── 0) Ship-To → Lv2~7 vector LUT ───────────────────────────
    ship_src   = df_src[Sales_Domain_ShipTo].to_numpy(dtype=object)
    pos_ship   = ship_idx.get_indexer(ship_src)          # -1 ↔ not-found
    valid_ship = pos_ship >= 0

    def pick_dom(pos: int, lv: int) -> str:
        """Lv2~7 Ship-To 코드 반환 (없으면 '')"""
        return lv_codes_arr[pos, lv-2]

    # ── 1) 컬럼 매핑 (Sell-IN / OUT) ─────────────────────────────
    if sell_type == "IN":
        lock_cols = {"GC": SIn_FCST_GC_LOCK,
                     "AP2": SIn_FCST_AP2_LOCK,
                     "AP1": SIn_FCST_AP1_LOCK}
        color_col = SIn_FCST_Color_Condition
    else:
        lock_cols = {"GC": SOut_FCST_GC_LOCK,
                     "AP2": SOut_FCST_AP2_LOCK,
                     "AP1": SOut_FCST_AP1_LOCK}
        color_col = SOut_FCST_Color_Condition

    rows_gc, rows_ap2, rows_ap1, rows_col = [], [], [], []
    color_key_set: set[tuple] = set()        # 중복 차단용

    # ── 2) ndarray 슬라이스 (iloc 없이 최대 속도) ────────────────
    pg_arr   = df_src[Product_Group      ].to_numpy(dtype=object)
    itm_arr  = df_src[Item_Item          ].to_numpy(dtype=object)
    loc_arr  = df_src[Location_Location  ].to_numpy(dtype=object)
    wk_arr   = df_src[Partial_Week       ].to_numpy(dtype=object)
    gc_lock_arr = df_src[Lock_Condition  ].to_numpy(dtype=bool)
    col_arr     = df_src[SIn_FCST_Color_Condition].to_numpy(dtype=object)  # IN 컬러 재사용

    # ── 3) 메인 루프 ────────────────────────────────────────────
    n = len(df_src)
    for i in range(n):
        if not valid_ship[i]:
            continue
        pos   = pos_ship[i]
        pg    = pg_arr[i]

        # Forecast-Rule key 는 Lv2/Lv3 로만 존재
        for base_lv in (2, 3):
            dom_lv23 = pick_dom(pos, base_lv)
            if not dom_lv23:
                continue
            rule = rule_dict.get((pg, dom_lv23))
            if rule is None:
                continue

            gc_lv, ap2_lv, ap1_lv = rule

            def add_lock_row(lst, lv: int, lock_bool: bool):
                """해당 레벨 Ship-To 로 LOCK-ROW + COLOR-ROW(중복방지) 삽입"""
                new_dom = pick_dom(pos, lv)
                if not new_dom:
                    return
                # ── LOCK 리스트 추가 ──
                lst.append((itm_arr[i], new_dom, loc_arr[i], wk_arr[i], None, lock_bool))
                # ── COLOR 리스트 (중복 check) ──
                key = (itm_arr[i], new_dom, loc_arr[i], wk_arr[i])
                if key not in color_key_set:
                    rows_col.append((*key, col_arr[i]))
                    color_key_set.add(key)

            if 2 <= gc_lv <= 7:
                add_lock_row(rows_gc,  gc_lv,  gc_lock_arr[i])
            if 2 <= ap2_lv <= 7:
                add_lock_row(rows_ap2, ap2_lv, gc_lock_arr[i])
            if 2 <= ap1_lv <= 7:
                add_lock_row(rows_ap1, ap1_lv, gc_lock_arr[i])

    # ── 4) DataFrame 생성 & 컬럼 정리 ────────────────────────────
    def _mk(row_list, lock_name: str | None) -> pd.DataFrame:
        if not row_list:          # 비어있을 수 있음
            return pd.DataFrame(columns=[
                Item_Item, Sales_Domain_ShipTo, Location_Location,
                Partial_Week,
                *( [lock_name, f'{lock_name}.Lock'] if lock_name else [color_col] )
            ])

        if lock_name:      # GC/AP2/AP1
            cols = [Item_Item, Sales_Domain_ShipTo, Location_Location,
                    Partial_Week, lock_name, f'{lock_name}.Lock']
            df = pd.DataFrame(row_list, columns=cols)
        else:              # COLOR
            cols = [Item_Item, Sales_Domain_ShipTo, Location_Location,
                    Partial_Week, color_col]
            df = pd.DataFrame(row_list, columns=cols).drop_duplicates(
                    subset=[Item_Item, Sales_Domain_ShipTo,
                            Location_Location, Partial_Week])
        return df.reset_index(drop=True)

    return (_mk(rows_gc,  lock_cols["GC"]),
            _mk(rows_ap2, lock_cols["AP2"]),
            _mk(rows_ap1, lock_cols["AP1"]),
            _mk(rows_col, None))


# ════════════════════════════════════════════════════════════
#  2.  SELL-IN Wrapper  (Version 열 삽입, 컬럼 순서 정리)
# ════════════════════════════════════════════════════════════
@_decoration_
def step12_build_sellin_outputs(
    df_src      : pd.DataFrame,
    version     : str,
    rule_dict   : dict,
    ship_idx    : pd.Index,
    lv_codes_arr: np.ndarray
) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:    

    df_gc_raw, df_ap2_raw, df_ap1_raw, df_col_raw = step12_core(
        df_src, rule_dict, ship_idx, lv_codes_arr, sell_type="IN")

    def _finish(df, last_col):
        df.insert(0, Version_Name, version)
        if SIn_FCST_Color_Condition == last_col :
            return df[[Version_Name, Item_Item, Sales_Domain_ShipTo,
                    Location_Location, Partial_Week, last_col]]
        else:
            return df[[Version_Name, Item_Item, Sales_Domain_ShipTo,
                    Location_Location, Partial_Week, last_col,f'{last_col}.Lock']]

    return (_finish(df_gc_raw,  SIn_FCST_GC_LOCK),
            _finish(df_ap2_raw, SIn_FCST_AP2_LOCK),
            _finish(df_ap1_raw, SIn_FCST_AP1_LOCK),
            _finish(df_col_raw, SIn_FCST_Color_Condition))


# ════════════════════════════════════════════════════════════
#  3.  SELL-OUT Wrapper (eStore Ship-To 필터 포함)
# ════════════════════════════════════════════════════════════
@_decoration_
def step13_build_sellout_outputs(
    df_src        : pd.DataFrame,
    version       : str,
    estore_set    : Collection[str],
    rule_dict     : dict,
    ship_idx      : pd.Index,
    lv_codes_arr  : np.ndarray
) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:        
                                
    df_est = df_src.loc[df_src[Sales_Domain_ShipTo].isin(estore_set)].reset_index(drop=True)

    df_gc_raw, df_ap2_raw, df_ap1_raw, df_col_raw = step12_core(
        df_est, rule_dict, ship_idx, lv_codes_arr, sell_type="OUT")

    def _finish(df, last_col):
        df.insert(0, Version_Name, version)
        if SOut_FCST_Color_Condition == last_col :
            return df[[Version_Name, Item_Item, Sales_Domain_ShipTo,
                    Location_Location, Partial_Week, last_col]]
        else:
            return df[[Version_Name, Item_Item, Sales_Domain_ShipTo,
                    Location_Location, Partial_Week, last_col,f'{last_col}.Lock']]

    return (_finish(df_gc_raw,  SOut_FCST_GC_LOCK),
            _finish(df_ap2_raw, SOut_FCST_AP2_LOCK),
            _finish(df_ap1_raw, SOut_FCST_AP1_LOCK),
            _finish(df_col_raw, SOut_FCST_Color_Condition))

########################################################################################################################
# End Function Of Steps
########################################################################################################################

########################################################################################################################
# Start Main
########################################################################################################################
if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    # Output 테이블 선언
    out_Demand = pd.DataFrame()
    out_sellin = pd.DataFrame()
    out_sellout = pd.DataFrame()
    output_dataframes = {}
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
            # input_folder_name  = f'{str_instance}_SHA_REF'
            # output_folder_name = f'{str_instance}_SHA_REF'
            # input_folder_name  = f'PYForecastSellInAndEstoreSellOutLockColor_0513_o9_data_local'
            # output_folder_name = f'PYForecastSellInAndEstoreSellOutLockColor_0513_o9_data_local'
            input_folder_name  = f'PYForecastSellInAndEstoreSellOutLockColor_0523_o9_SM_A546UZKBXAG'
            output_folder_name = f'PYForecastSellInAndEstoreSellOutLockColor_0523_o9_SM_A546UZKBXAG'
            
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

        df_dim    = input_dataframes[str_df_in_Sales_Domain_Dimension]
        df_time   = input_dataframes[str_df_in_Time_Partial_Week]
        df_rule   = input_dataframes[str_df_in_Forecast_Rule]
        df_itemmst= input_dataframes[str_df_in_Item_Master]  
        


        df_eos = input_dataframes.get(str_df_in_Time_Partial_Week)
        max_week = df_eos[Partial_Week].max()
        max_week_normalized = normalize_week(max_week)
        current_week_normalized = normalize_week(CurrentPartialWeek)


        # 입력 변수 확인
        if Version is None or Version.strip() == '':
            Version = 'CWV_DP'
        # 입력 변수 확인용 로그
        logger.Note(p_note=f'Version : {Version}', p_log_level=LOG_LEVEL.debug())


        # 2) build LUT once
        ship_idx, lvl_arr, ship_lv_dict = build_shipto_level_lut(
            input_dataframes[STR_DF_DIM])  

        dim_df = input_dataframes[STR_DF_DIM].set_index(Sales_Domain_ShipTo).reindex(ship_idx)
        lv_cols = [Sales_Domain_LV2, Sales_Domain_LV3, Sales_Domain_LV4,
                Sales_Domain_LV5, Sales_Domain_LV6, Sales_Domain_LV7]
        lv_codes_arr = dim_df[lv_cols].to_numpy(dtype=object)   # shape(len(ship_idx),6)  

        # estore_set 역시 전역에 만들어 둡니다.
        estore_set = set(input_dataframes[STR_DF_ESTORE][Sales_Domain_ShipTo].unique())

        # ── B)  Forecast-Rule dict  (벡터 · 반복문 X) ──
        rule_pg   = input_dataframes[STR_DF_RULE][Product_Group       ].to_numpy(dtype=object)
        rule_dom  = input_dataframes[STR_DF_RULE][Sales_Domain_ShipTo].to_numpy(dtype=object)
        rule_vals = input_dataframes[STR_DF_RULE][[
                        FORECAST_RULE_GC_FCST, FORECAST_RULE_AP2_FCST,
                        FORECAST_RULE_AP1_FCST]].to_numpy(dtype='int32')

        rule_dict = {(pg, dom): tuple(vals)
             for pg, dom, vals in zip(rule_pg, rule_dom, rule_vals)}



        ################################################################################################################
        # Step 1. df_in_MST_RTS 와 df_in_MST_EOS 를 Item 과 ShipTo를 기준으로 inner Join
        ################################################################################################################
        dict_log = {
            'p_step_no': 10,
            'p_step_desc': 'Step 1  : df_in_MST_RTS 와 df_in_MST_EOS 를 Item 과 ShipTo를 기준으로 inner Join',
            # 'p_df_name' : 'df_01_joined_rts_eos'
        }
        df_rtsEos = fn_step01_load_rts_eos(
            input_dataframes[STR_DF_RTS_EOS], 
            ship_idx, 
            lvl_arr 
            ,**dict_log
        )
        # fn_check_input_table(df_01_joined_rts_eos, 'df_01_joined_rts_eos', '0')
        output_dataframes[str_df_fn_RTS_EOS] = df_rtsEos
        fn_log_dataframe(df_rtsEos, f'df_01_{str_df_fn_RTS_EOS}')

        ################################################################################################################
        # Step 2  : Step1의 Result에 Time을 Partial Week 으로 변환
        ################################################################################################################
        dict_log = {
            'p_step_no': 20,
            'p_step_desc': 'Step 2  : Step1의 Result에 Time을 Partial Week 으로 변환 ',
            # 'p_df_name' : 'df_02_date_to_partial_week'
        }
        df_fn_RTS_EOS = fn_step02_convert_date_to_partial_week(**dict_log)
        # print for test  
        output_dataframes[str_df_fn_RTS_EOS] = df_fn_RTS_EOS   
        fn_log_dataframe(df_fn_RTS_EOS, f'df_02_{str_df_fn_RTS_EOS}')
        ################################################################################################################
        # Step 3:  df_in_MST_RTS 와 df_in_MST_EOS의 ITEM * ShipTo로 Inner Join하여 새로운 DF 생성 ( Output Data )
        ################################################################################################################
        dict_log = {
            'p_step_no': 30,
            'p_step_desc': 'Step 3  : df_in_MST_RTS 와 df_in_MST_EOS의 ITEM * ShipTo로 Inner Join하여 새로운 DF 생성 ( Output Data ) ' ,
            # 'p_df_name' : 'df_03_joined_rts_eos'
        }
        # df_03_joined_rts_eos = fn_step03_join_rts_eos(**dict_log)
        # # fn_check_input_table(df_03_joined_rts_eos, 'df_03_joined_rts_eos', '0')
        # output_dataframes["df_03_joined_rts_eos"] = df_03_joined_rts_eos
        # if is_local:
        #     fn_log_dataframe(df_03_joined_rts_eos, 'df_03_joined_rts_eos')

        ################################################################################################################
        # Step 4  : Step3의 df에 Partial Week 및 Measure Column 추가
        ################################################################################################################
        dict_log = {
            'p_step_no': 40,
            'p_step_desc': 'Step 4  : Step3의 df에 Partial Week 및 Measure Column 추가 ' ,
            # 'p_df_name' : 'df_04_partialweek_measurecolumn'
        }
        df_fn_RTS_EOS_Week = fn_step04_add_partialweek_measurecolumn(**dict_log)
        # fn_check_input_table(df_04_partialweek_measurecolumn, 'df_04_partialweek_measurecolumn', '0')
        # print for test  
        output_dataframes[str_df_fn_RTS_EOS_Week] = df_fn_RTS_EOS_Week
        fn_log_dataframe(df_fn_RTS_EOS_Week, f'df_04_{str_df_fn_RTS_EOS_Week}')

        ################################################################################################################
        # Step 5:  Step4의 df에 당주주차부터 RTS 와 EOS 반영 및 Color 표시
        ################################################################################################################
        dict_log = {
            'p_step_no': 50,
            'p_step_desc': 'Step 5  : Step4의 df에 당주주차부터 RTS 와 EOS 반영 및 Color 표시 ' ,
            # 'p_df_name' : 'df_05_set_lock_values'
        }
        df_fn_RTS_EOS_Week = fn_step05_set_lock_values(**dict_log)
        fn_log_dataframe(df_fn_RTS_EOS_Week, f'df_05_{str_df_fn_RTS_EOS_Week}')
        output_dataframes[str_df_fn_RTS_EOS_Week] = df_fn_RTS_EOS_Week

        ################################################################################################################
        # Step 6:  Step5의 df에 Item Master 정보 추가( Item Type, Item GBM, Item Product Group) 및 Color 조건 업데이트(무선 BAS 제품 8주 구간 GREEN UPDATE)
        ################################################################################################################
        dict_log = {
            'p_step_no': 60,
            'p_step_desc': 'Step 6  : Step5의 df에 Item Master 정보 추가 및 Color 조건 업데이트 ' ,
            # 'p_df_name' : 'df_06_addcolumn_green_for_wireless_bas'
        }
        df_fn_RTS_EOS_Week = fn_step06_addcolumn_green_for_wireless_bas_array_based(**dict_log)
        fn_log_dataframe(df_fn_RTS_EOS_Week, f'df_06_{str_df_fn_RTS_EOS_Week}')
        output_dataframes[str_df_fn_RTS_EOS_Week] = df_fn_RTS_EOS_Week
        df_rtsWk = df_fn_RTS_EOS_Week

        ################################################################################################################
        # Step 7:  df_in_Sales_Product_ASN 전처리
        # lvl 을 구성한다.
        ################################################################################################################
        dict_log = {
            'p_step_no': 70,
            'p_step_desc': 'Step 7  : df_in_Sales_Product_ASN 전처리 ' ,
            # 'p_df_name' : 'df_07_join_sales_product_asn_to_lvl'
        }
        # df_asn_item_week = fn_step07_join_sales_product_asn_to_lvl_by_merge(**dict_log)
        df_asn_item_week = step07_prepare_asn(
            input_dataframes[STR_DF_ASN],
            input_dataframes[STR_DF_ITEMMST],
            input_dataframes[STR_DF_DIM],
            input_dataframes[STR_DF_TIME],
            CurrentPartialWeek,
            **dict_log
        )

        # fn_check_input_table(df_07_join_sales_product_asn_to_lvl, 'df_07_join_sales_product_asn_to_lvl', '0')
        # print for test  
        output_dataframes[str_df_fn_Sales_Product_ASN_Item_Week] = df_asn_item_week  
        fn_log_dataframe(df_asn_item_week, f'df_07_{str_df_fn_Sales_Product_ASN_Item_Week}')

        ################################################################################################################
        # Step 8:  ASN만 있는 경우 조건 추가
        ################################################################################################################
        dict_log = {
            'p_step_no': 80,
            'p_step_desc': 'Step 8  : ASN만 있는 경우 조건 추가' 
        }
        # df_08_add_weeks_to_dimention = fn_step08_add_weeks_to_dimention_vector_join_chunk('df_07_join_sales_product_asn_to_lvl',**dict_log)
        # df_fn_Sales_Product_ASN_Item_Week = fn_step08_create_asn_item_week_and_match_locks(output_dataframes[str_df_fn_Sales_Product_ASN_Item])
        df_asn_item_week = fn_step08_match_rts(df_asn_item_week, df_rtsWk, df_dim , **dict_log)
        fn_log_dataframe(df_asn_item_week,f'df_08_{str_df_fn_Sales_Product_ASN_Item_Week}')
        output_dataframes[str_df_fn_Sales_Product_ASN_Item_Week] = df_asn_item_week  

        """
        import duckdb
        duckdb.register('df_asn_item_week', df_asn_item_week)
        # 여기서 14_WHITE 가 된다.
        query = f'''
            select * from df_asn_item_week 
            where "{Item_Item}" = 'SM-A546UZKBXAG'
            and "{Location_Location}" = 'S341'
            and "{Partial_Week}" >= '202501'
            and "{Partial_Week}" <= '202521'
            
        '''
        duckdb.query(query).show()
        """

        ################################################################################################################
        # Step 09:   Step8의 df에 Item CLASS 정보 필터링
        ################################################################################################################
        dict_log = {
            'p_step_no': 90,
            'p_step_desc': 'Step 9  : Step8의 df에 Item CLASS 정보 필터링 ' 
        }

        df_asn_item_week = fn_step09_apply_itemclass(
            df_asn_item_week,
            input_dataframes[str_df_in_Item_CLASS],
            output_dataframes[str_df_fn_RTS_EOS],          # step02 완료분
            output_dataframes[str_df_fn_RTS_EOS_Week],     # step04 완료분
            input_dataframes[str_df_in_Sales_Domain_Dimension],
            CurrentPartialWeek,
            **dict_log
        )
        fn_log_dataframe(df_asn_item_week,f'df_09_{str_df_fn_Sales_Product_ASN_Item_Week}')
        output_dataframes[str_df_fn_Sales_Product_ASN_Item_Week] = df_asn_item_week  

        """
        import duckdb
        duckdb.register('df_asn_item_week', df_asn_item_week)
        # 여기서 14_WHITE 가 된다.
        query = f'''
            select * from df_asn_item_week 
            where "{Item_Item}" = 'SM-A546UZKBXAG'
            and "{Location_Location}" = 'S341'
            and "{Partial_Week}" >= '202501'
            and "{Partial_Week}" <= '202521'
            
        '''
        duckdb.query(query).show()
        """

        ################################################################################################################
        # Step 101:   Step9의 df에 Item TAT 정보 필터링 및 Lock
        ################################################################################################################
        dict_log = {
            'p_step_no': 101,
            'p_step_desc': 'Step 10  : Step9의 df에 Item TAT 정보 필터링 및 Lock ' 
        }
        df_asn_item_week = step10_1_vd_leadtime(
            df_asn_item_week, 
            input_dataframes[str_df_in_Item_TAT], 
            output_dataframes[str_df_fn_RTS_EOS_Week],     # step04 완료분
            CurrentPartialWeek,
            **dict_log
        )
        fn_log_dataframe(df_asn_item_week,f'df_10.1_{str_df_fn_Sales_Product_ASN_Item_Week}')
        output_dataframes[str_df_fn_Sales_Product_ASN_Item_Week] = df_asn_item_week  


        ################################################################################################################
        # Step 102:   Step10-1의 df에 TATTERM_SET  정보 필터링 및 Lock
        ################################################################################################################
        dict_log = {
            'p_step_no': 102,
            'p_step_desc': 'Step 102  : Step10-1의 df에 TATTERM_SET  정보 필터링 및 Lock ' 
        }
        df_asn_item_week = step10_2_vd_set_leadtime(
            df_asn_item_week, 
            input_dataframes[str_df_in_Item_TAT], 
            output_dataframes[str_df_fn_RTS_EOS_Week],     # step04 완료분
            CurrentPartialWeek,
            **dict_log
        )
        fn_log_dataframe(df_asn_item_week,f'df_10.2_{str_df_fn_Sales_Product_ASN_Item_Week}')
        output_dataframes[str_df_fn_Sales_Product_ASN_Item_Week] = df_asn_item_week  

        """
        import duckdb
        duckdb.register('df_asn_item_week', df_asn_item_week)
        # 여기서 14_WHITE 가 된다.
        query = f'''
            select * from df_asn_item_week 
            where "{Item_Item}" = 'SM-A546UZKBXAG'
            and "{Location_Location}" = 'S341'
            and "{Partial_Week}" >= '202501'
            and "{Partial_Week}" <= '202521'
            
        '''
        duckdb.query(query).show()
        """

        ################################################################################################################
        # Step11 Sell-out 미존재 Lock 적용 ────────────────────────────
        ################################################################################################################
        dict_log = {
            'p_step_no': 110,
            'p_step_desc': 'Step 11  : MX Sellout FCST 없는 모델 당주 이후 미래구간 GRAY UPDATE		 ' 
        }
        df_asn_item_week = step11_apply_no_sellout(
            df_asn_item_week,
            input_dataframes[str_df_in_SELLOUTFCST_NOTEXIST],
            df_dim, 
            output_dataframes[str_df_fn_RTS_EOS_Week],     # step04 완료분
            CurrentPartialWeek,
            **dict_log
        )
        fn_log_dataframe(df_asn_item_week,f'df_11_{str_df_fn_Sales_Product_ASN_Item_Week}')
        output_dataframes[str_df_fn_Sales_Product_ASN_Item_Week] = df_asn_item_week

        """
        import duckdb
        duckdb.register('df_asn_item_week', df_asn_item_week)

        # 결과치를 보려고 할때
        query = f'''
            select * from df_asn_item_week 
            where 1=1
            -- and "{Item_Item}" = 'SM-A546UZKBXAG'
            -- and "{Location_Location}" = 'S341'
            -- and "{Item_Item}" = 'GT-I8200RWPZVV'
            and "{Item_Item}" = 'LH75QHCEBGCXXF'
            and "{Location_Location}" = 'S627'
            and "{Partial_Week}" >= '202501'
            and "{Partial_Week}" <= '202521'
            
        '''
        duckdb.query(query).show()

        # distinct item
        query = f'''
            select distinct "{Item_Item}" from df_asn_item_week 
        '''
        duckdb.query(query).show()

        # distinct ShipTo, Item For Some Item
        query = f'''
            select distinct 
                "{Sales_Domain_ShipTo}", 
                "{Item_Item}" 
            from df_asn_item_week 
            where "{Item_Item}" = 'LH75QHCEBGCXXF'
        '''
        duckdb.query(query).show()
        # and I added 300498,LH75QHCEBGCXXF to no_sellout

        duckdb.register('no_ex', input_dataframes[str_df_in_SELLOUTFCST_NOTEXIST])
        query = f'''
            select * from no_ex 
        '''
        duckdb.query(query).show()

        # dim
        duckdb.register('df_dim', input_dataframes[str_df_in_Sales_Domain_Dimension])
        query = f'''
            select * from df_dim
            where "{Sales_Domain_ShipTo}" = '5014052'
        '''
        duckdb.query(query).show()


        """
        ################################################################################################################
        # Step 12:   Forecast Rule에 따른 Data 생성
        ################################################################################################################
        dict_log = {
            'p_step_no': 120,
            'p_step_desc': 'Step 12  :  Forecast Rule에 따른 Data 생성' 
        }
        
        # ── LUT & eStore 준비 (초기에 1회) ─────────────────────────────
        (
            df_output_Sell_In_FCST_GI_GC_Lock,
            df_output_Sell_In_FCST_GI_AP2_Lock,
            df_output_Sell_In_FCST_GI_AP1_Lock,
            df_output_Sell_In_FCST_Color_Condition
        ) = step12_build_sellin_outputs(
            df_asn_item_week,
            Version,
            rule_dict,
            ship_idx,
            lv_codes_arr,
            **dict_log
        )
        output_dataframes['df_output_Sell_In_FCST_GI_GC_Lock'       ] = df_output_Sell_In_FCST_GI_GC_Lock
        output_dataframes['df_output_Sell_In_FCST_GI_AP2_Lock'      ] = df_output_Sell_In_FCST_GI_AP2_Lock
        output_dataframes['df_output_Sell_In_FCST_GI_AP1_Lock'      ] = df_output_Sell_In_FCST_GI_AP1_Lock
        output_dataframes['df_output_Sell_In_FCST_Color_Condition'  ] = df_output_Sell_In_FCST_Color_Condition
        fn_log_dataframe(df_output_Sell_In_FCST_GI_GC_Lock,         f'df_12_df_output_Sell_In_FCST_GI_GC_Lock')
        fn_log_dataframe(df_output_Sell_In_FCST_GI_AP2_Lock,        f'df_12_df_output_Sell_In_FCST_GI_AP2_Lock')
        fn_log_dataframe(df_output_Sell_In_FCST_GI_AP1_Lock,        f'df_12_df_output_Sell_In_FCST_GI_AP1_Lock')
        fn_log_dataframe(df_output_Sell_In_FCST_Color_Condition,    f'df_12_df_output_Sell_In_FCST_Color_Condition')
        
        
        ################################################################################################################
        # Step 13:   E-store 로직 추가
        ################################################################################################################
        dict_log = {
            'p_step_no': 130,
            'p_step_desc': 'Step 13  : E-store 로직 추가' 
        }

        # ── D)  STEP-13  (Sell-OUT 4 개 DF) ──
        (
            df_output_Sell_Out_FCST_GC_Lock,
            df_output_Sell_Out_FCST_AP2_Lock,
            df_output_Sell_Out_FCST_AP1_Lock,
            df_output_Sell_Out_FCST_Color_Condition
        ) = step13_build_sellout_outputs(
            df_asn_item_week,
            Version,
            estore_set,
            rule_dict,
            ship_idx,
            lv_codes_arr,
            **dict_log
        )

        output_dataframes['df_output_Sell_Out_FCST_GC_Lock'          ] = df_output_Sell_Out_FCST_GC_Lock
        output_dataframes['df_output_Sell_Out_FCST_AP2_Lock'         ] = df_output_Sell_Out_FCST_AP2_Lock
        output_dataframes['df_output_Sell_Out_FCST_AP1_Lock'         ] = df_output_Sell_Out_FCST_AP1_Lock
        output_dataframes['df_output_Sell_Out_FCST_Color_Condition'  ] = df_output_Sell_Out_FCST_Color_Condition
        fn_log_dataframe(df_output_Sell_Out_FCST_GC_Lock,           f'df_13_df_output_Sell_Out_FCST_GC_Lock')
        fn_log_dataframe(df_output_Sell_Out_FCST_AP2_Lock,          f'df_13_df_output_Sell_Out_FCST_AP2_Lock')
        fn_log_dataframe(df_output_Sell_Out_FCST_AP1_Lock,          f'df_13_df_output_Sell_Out_FCST_AP1_Lock')
        fn_log_dataframe(df_output_Sell_Out_FCST_Color_Condition,   f'df_13_df_output_Sell_Out_FCST_Color_Condition')
        
        """
            # 이단계에서는 True False 가 확실히 됨.
            import duckdb
            duckdb.register('df_ap2', df_output_Sell_In_FCST_GI_AP2_Lock)
            duckdb.register('df_color', df_output_Sell_In_FCST_Color_Condition)
            query = f'''
                select * from df_ap2 
                where "{Item_Item}" = 'SM-A546UZKBXAG'
                and "{Partial_Week}" >= '202501'
                and "{Partial_Week}" <= '202521'
                -- and "{Partial_Week}" = '300114'
                -- and "{Partial_Week}" = '408351'
                -- and "{Partial_Week}" = '5006941'
            '''
            duckdb.query(query).show()

            # 여기서 19_GRAY 가 된다.
            query = f'''
                select * from df_color 
                where "{Item_Item}" = 'SM-A546UZKBXAG'
                and "{Location_Location}" = 'S341'
                and "{Partial_Week}" >= '202501'
                and "{Partial_Week}" <= '202521'
                -- and "{Partial_Week}" = '300114'
                -- and "{Partial_Week}" = '408351'
                -- and "{Partial_Week}" = '5006941'
            '''
            duckdb.query(query).show()
            ┌────────────────────────┬────────────────┬────────────────────────┬─────────────────────┬─────────────────────┬───────────────────────────┐
            │ Version.[Version Name] │  Item.[Item]   │ Sales Domain.[Ship To] │ Location.[Location] │ Time.[Partial Week] │ S/In FCST Color Condition │
            │        varchar         │    varchar     │        varchar         │       varchar       │       varchar       │          varchar          │
            ├────────────────────────┼────────────────┼────────────────────────┼─────────────────────┼─────────────────────┼───────────────────────────┤
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5002469                │ S341                │ 202505B             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5002469                │ S341                │ 202506A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5002469                │ S341                │ 202507A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5002469                │ S341                │ 202508A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5002469                │ S341                │ 202509A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5002469                │ S341                │ 202509B             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5002469                │ S341                │ 202510A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5002469                │ S341                │ 202511A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5002469                │ S341                │ 202512A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5002469                │ S341                │ 202513A             │ 19_GRAY                   │
            │   ·                    │       ·        │    ·                   │  ·                  │    ·                │    ·                      │
            │   ·                    │       ·        │    ·                   │  ·                  │    ·                │    ·                      │
            │   ·                    │       ·        │    ·                   │  ·                  │    ·                │    ·                      │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5018867                │ S341                │ 202513A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5018867                │ S341                │ 202514A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5018867                │ S341                │ 202514B             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5018867                │ S341                │ 202515A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5018867                │ S341                │ 202516A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5018867                │ S341                │ 202517A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5018867                │ S341                │ 202518A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5018867                │ S341                │ 202518B             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5018867                │ S341                │ 202519A             │ 19_GRAY                   │
            │ CWV_DP                 │ SM-A546UZKBXAG │ 5018867                │ S341                │ 202520A             │ 19_GRAY                   │
            ├────────────────────────┴────────────────┴────────────────────────┴─────────────────────┴─────────────────────┴───────────────────────────┤
            │ 988 rows (20 shown)                                                                                                            6 columns │
            └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

        """


    except Exception as e:
        trace_msg = traceback.format_exc()
        logger.Note(p_note=trace_msg, p_log_level=LOG_LEVEL.debug())
        logger.Error()
        if flag_exception:
            raise Exception(e)
        else:
            logger.info(f'{str_instance} exit - {time.strftime("%Y-%m-%d - %H:%M:%S")}')


    finally:
        # # MediumWeight 실행 시 Header 없는 빈 데이터프레임이 Output이 되는 경우 오류가 발생함.
        # # 이 오류를 방지하기 위해 Output이 빈 경우을 체크하여 Header를 만들어 줌.
        # if out_sellin.empty:
        #     out_sellin = fn_set_header_in()
        #     # fn_log_dataframe(out_sellin, 'out_sellin')
        #     if is_local:
        #         out_sellin.to_csv(f'{os.getcwd()}/' + str_output_dir + '/'+'out_sellin.csv', encoding='UTF8', index=False)

        # if out_sellout.empty:
        #     out_sellout = fn_set_header_out()
        #     # fn_log_dataframe(out_sellout, 'out_sellout')
        #     if is_local:
        #         out_sellout.to_csv(f'{os.getcwd()}/' + str_output_dir + '/'+'out_sellout.csv', encoding='UTF8', index=False)
            

        
        if is_local:
            log_file_name = common.G_PROGRAM_NAME.replace('py', 'log')
            log_file_name = f'log/{log_file_name}'

            shutil.copyfile(log_file_name, os.path.join(str_output_dir, os.path.basename(log_file_name)))

            # prografile copy
            program_path = f"{os.getcwd()}/NSCM_DP_UI_Develop/{str_instance}.py"
            shutil.copyfile(program_path, os.path.join(str_output_dir, os.path.basename(program_path)))

            # log
            input_path = f'{str_output_dir}/input'
            os.makedirs(input_path,exist_ok=True)
            for input_file in input_dataframes:
                input_dataframes[input_file].to_csv(input_path + "/"+input_file+".csv", encoding="UTF8", index=False)

            # log
            output_path = f'{str_output_dir}/output'
            os.makedirs(output_path,exist_ok=True)
            for output_file in output_dataframes:
                output_dataframes[output_file].to_csv(output_path + "/"+output_file+".csv", encoding="UTF8", index=False)

        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::') # 25.05.12 need warning Log by Logger Issue
        logger.Finish()
        
       
########################################################################################################################
# End Main
########################################################################################################################
