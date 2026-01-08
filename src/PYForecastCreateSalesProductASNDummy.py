"""
* 프로그램명	
	PYForecastCreateSalesProductASNDummy


* 목적	
	- SalesProductASN 의 기본 Data가 6/7 Lv로 되어 있어서, Dummy Data가 보이기 위해서 Dummy에 대한 ASN 생성 필요
	- Sell-Out Data를 위한 Location = '-' 인 Data 생성 필요
	- PY 실행 전 o9에서 ASN의 초기 Leaf Data를 제외하고 삭제 필요


* 변경이력	
	2025.03.28 전창민 최초 작성

* Script Parameter


* Input Tables (*)

    - (Input 1) Sales Domian Master 정보		
        df_in_Sales_Domain_Dimension	
            Select (
            * [Sales Domain].[Sales LV2]
            * [Sales Domain].[Sales LV3] 
            * [Sales Domain].[Sales LV4] 
            * [Sales Domain].[Sales LV5] 
            * [Sales Domain].[Sales LV6] 
            * [Sales Domain].[Sales LV7] 
            * [Sales Domain].[Ship To] )

            Sales Domain.[Sales LV2]	Sales Domain.[Sales LV3]	Sales Domain.[Sales LV4]	Sales Domain.[Sales LV5]	Sales Domain.[Sales LV6]	Sales Domain.[Sales LV7]	Sales Domain.[Ship To]
            203	203	203	203	203	203	203
            203	300114	300114	300114	300114	300114	300114
            203	300114	A300114	A300114	A300114	A300114	A300114
            203	300114	A300114	400362	400362	400362	400362
            203	300114	A300114	400362	5002453	5002453	5002453
            203	300114	A300114	400362	5002453	A5002453	A5002453
            203	300114	A300114	400362	5003074	5003074	5003074
            203	300114	A300114	400362	5003074	A5003074	A5003074
            203	300114	A300114	400362	5005569	5005569	5005569
            203	300114	A300114	400362	5005569	A5005569	A5005569
            203	300114	A300114	400362	5007280	5007280	5007280
            203	300114	A300114	400362	5007280	A5007280	A5007280
            203	300114	A300114	408273	408273	408273	408273
            203	300114	A300114	408273	5006941	5006941	5006941
            203	300114	A300114	408273	5006941	A5006941	A5006941
            203	300114	A300114	408273	5019692	5019692	5019692
            203	300114	A300114	408273	5019692	A5019692	A5019692

    - (Input 2) Sales Product ASN 정보					
        df_in_Sales_Product_ASN				
            Select ([Version].[Version Name]			
            * [Sales Domain].[Ship To] 			
            * Item.[Item]			
            * [Location].[Location] )  on row, 			
            ( { Measure.[Sales Product ASN] } ) on column			
            where { (Measure.[Sales Product ASN] == "Y" ) } ;			
                        
        Version.[Version Name]	Sales Domain.[Ship To]	Item.[Item]	Location.[Location]	Sales Product ASN
        CWV_DP	A5002453	RF29BB8600QLAA	S377	Y
        CWV_DP	A5002453	RF29BB8600QLAA	S376	Y
        CWV_DP	5006941	RF29BB8600QLAA	S377	Y
        CWV_DP	5006941	RF29BB8600QLAA	S376	Y

* Output Tables (*)
    - (Output 1)		
        Output_Sales_Product_ASN	
            Select ([Version].[Version Name]
            * [Sales Domain].[Ship To] 
            * Item.[Item]
            * [Location].[Location] )  on row, 
            ( { Measure.[Sales Product ASN] } ) on column
            where { (Measure.[Sales Product ASN] == "Y" ) } ;

* Flow Summary




"""

from re import X
import os,sys,json,shutil,io,zipfile
import time
import datetime
import inspect
import traceback
import pandas as pd
from NSCMCommon import NSCMCommon as common
from NSCMCommon import VDCommon as vdCommon
# from typing_extensions import Literal
import glob
import numpy as np
# import rbql
# import duckdb

########################################################################################################################
# Local 개발 시에 필요한 공통 변수 선언
########################################################################################################################
# o9에 저장된 instanceName
is_local = common.gfn_get_isLocal()
str_instance = 'PYForecastCreateSalesProductASNDummy'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"

is_print = True
flag_csv = True
flag_exception = True
# Global variable for max_week
max_week = None
CurrentPartialWeek = None
max_week_normalized = None
current_week_normalized = None

input_dataframes = {}
output_dataframes = {}
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
Sales_Domain_Std1        = 'Sales Domain.[Sales Std1]'
Sales_Domain_Std2        = 'Sales Domain.[Sales Std2]'
Sales_Domain_Std3        = 'Sales Domain.[Sales Std3]' 
Sales_Domain_Std4        = 'Sales Domain.[Sales Std4]'
Sales_Domain_Std5        = 'Sales Domain.[Sales Std5]'
Sales_Domain_Std6        = 'Sales Domain.[Sales Std6]'
Location_Location       = 'Location.[Location]'
Item_Class              = 'ITEMCLASS Class'
Partial_Week            = 'Time.[Partial Week]'
SIn_FCST_GC_LOCK                = 'S/In FCST(GI)_GC.Lock'
SIn_FCST_Color_Condition        = 'S/In FCST Color Condition'
SIn_FCST_AP2_LOCK               = 'S/In FCST(GI)_AP2.Lock'
SIn_FCST_AP1_LOCK               = 'S/In FCST(GI)_AP1.Lock'


# Salse_Product_ASN       = 'Sales Product ASN'    
Salse_Product_ASN       = 'Sales Product ASN'
ITEMTAT_TATTERM         = 'ITEMTAT TATTERM'
ITEMTAT_TATTERM_SET     = 'ITEMTAT TATTERM_SET'
FORECAST_RULE_GC_FCST        = 'FORECAST_RULE GC FCST'
FORECAST_RULE_AP2_FCST       = 'FORECAST_RULE AP2 FCST'
FORECAST_RULE_AP1_FCST       = 'FORECAST_RULE AP1 FCST'   
FORECAST_RULE_CUST      = 'FORECAST_RULE CUST FCST'
# ----------------------------------------------------------------
# 
# ----------------------------------------------------------------
SOut_FCST_GC_LOCK         = 'S/Out FCST_GC.Lock'
SOut_FCST_AP2_LOCK        = 'S/Out FCST_AP2.Lock'
SOut_FCST_AP1_LOCK        = 'S/Out FCST_AP1.Lock'
SOut_FCST_Color_Condition = 'S/Out FCST Color Condition'
# ----------------------------------------------------------------
# 
# ----------------------------------------------------------------
SOut_FCST_GC_ASS = 'S/Out FCST Assortment_GC'
SOut_FCST_AP2_ASS = 'S/Out FCST Assortment_AP2'
SOut_FCST_AP1_ASS = 'S/Out FCST Assortment_AP1'
SIn_FCST_GC_ASS = 'S/In FCST(GI) Assortment_GC'
SIn_FCST_AP2_ASS = 'S/In FCST(GI) Assortment_AP2'
SIn_FCST_AP1_ASS = 'S/In FCST(GI) Assortment_AP1'
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
# ----------------------------------------------------------------
# Data Frame Name
# ----------------------------------------------------------------
str_df_in_Sales_Domain_Dimension                = 'df_in_Sales_Domain_Dimension'
str_df_in_Sales_Product_ASN                     = 'df_in_Sales_Product_ASN'
str_df_fn_Sales_Product_ASN                     = 'df_fn_Sales_Product_ASN'
str_df_fn_Sales_Product_ASN_Dummy               = 'df_fn_Sales_Product_ASN_Dummy'
str_df_fn_Sales_Product_ASN_lv6                 = 'df_fn_Sales_Product_ASN_lv6'
str_df_fn_Sales_Product_ASN_lv7                 = 'df_fn_Sales_Product_ASN_lv7'
str_df_fn_Sales_Product_ASN_lv7_to_lv6          = 'df_fn_Sales_Product_ASN_lv7_to_lv6'
str_df_fn_Sales_Product_ASN_Concat              = 'df_fn_Sales_Product_ASN_Concat'
str_df_fn_Sales_Product_ASN_Concat_dim          = 'df_fn_Sales_Product_ASN_Concat_dim'
str_df_fn_Sales_Product_ASN_Lv2_to_Lv5          = 'df_fn_Sales_Product_ASN_Lv2_to_Lv5'
str_df_fn_Sales_Product_ASN_Lv2_to_Lv6          = 'df_fn_Sales_Product_ASN_Lv2_to_Lv6'
str_df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy    = 'df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy'
str_df_fn_Sales_Product_ASN_Final               = 'df_fn_Sales_Product_ASN_Final'
str_Output_Sales_Product_ASN                    = 'Output_Sales_Product_ASN'



########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################

logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
# fn_set_local_logfile()
LOG_LEVEL = common.G_log_level

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
        if is_local and not df_p_source.empty and flag_csv:
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
def fn_set_header() ->None:
    """
    MediumWeight로 실행 시 발생할 수 있는 Live Server에서의 오류를 방지하기 위해 Header만 있는 Output 테이블을 만든다.
    :return: DataFrame
        """
    df_return = pd.DataFrame()

    # out_Demand
    df_return = pd.DataFrame(
        {
            Version_Name: [], 
            Sales_Domain_ShipTo: [], 
            Item_Item: [], 
            Location_Location: [],
            Partial_Week: [] ,
            SIn_FCST_GC_LOCK : [],
            SIn_FCST_AP2_LOCK : [],
            SIn_FCST_AP1_LOCK : [],
            SIn_FCST_Color_Condition : []
        })

    return df_return


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

def initialize_max_week(is_local, args):
    global max_week, CurrentPartialWeek, max_week_normalized, current_week_normalized,chunk_size_step08 , chunk_size_step09, chunk_size_step10 , chunk_size_step11, chunk_size_step12, chunk_size_step13
    if is_local:
        # Read from MST_MODELEOS_TEST.csv
        df_eos = input_dataframes.get('df_in_Time_Partial_Week')
        max_week = df_eos['Time.[Partial Week]'].max()  # Assuming EOS_COM_DATE represents 최대주차
    else:
        # Get from command-line arguments
        max_week = args.get('max_week')

    # Initialize max_week_normalized. may be 202653A
    max_week_normalized = normalize_week(max_week)
    # max_week_normalized = int(max_week_normalized, 16) # convert to int32 from current str.

    # Initialize CurrentPartialWeek
    CurrentPartialWeek = common.gfn_get_partial_week(p_datetime=datetime.datetime.now())
    CurrentPartialWeek = '202447'
    if args.get('CurrentPartialWeek') is not None:
        CurrentPartialWeek = args.get('CurrentPartialWeek')
    # Initialize current_week_normalized
    current_week_normalized = normalize_week(CurrentPartialWeek)
    # current_week_normalized = int(current_week_normalized, 16)

    if args.get('chunk_size_step08') is not None:
        chunk_size_step08 = int(args.get('chunk_size_step08'))
    if args.get('chunk_size_step09') is not None:
        chunk_size_step09 = int(args.get('chunk_size_step09'))
    if args.get('chunk_size_step10') is not None:
        chunk_size_step10 = int(args.get('chunk_size_step10'))
    if args.get('chunk_size_step11') is not None:
        chunk_size_step11 = int(args.get('chunk_size_step11'))
    if args.get('chunk_size_step12') is not None:
        chunk_size_step12 = int(args.get('chunk_size_step12'))
    if args.get('chunk_size_step13') is not None:
        chunk_size_step13 = int(args.get('chunk_size_step13'))

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


####################################################################
# Input Data Frame 처리
####################################################################
@_decoration_
def fn_process_in_df_mst():
    """
    Input Data Frame 처리
    """

    if is_local: 
        import csv
        # 로컬인 경우 Output 폴더를 정리한다.
        for file in os.scandir(str_output_dir):
            os.remove(file.path)

        # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
        file_pattern = f"{os.getcwd()}/{str_input_dir}/*.csv" 
        csv_files = glob.glob(file_pattern)

        file_to_df_mapping = {
            "df_in_Sales_Domain_Dimension.csv"  :      str_df_in_Sales_Domain_Dimension   ,
            "df_in_Sales_Product_ASN.csv"       :      str_df_in_Sales_Product_ASN             
        }

        def read_csv_with_fallback(filepath):
            encodings = ['utf-8-sig', 'utf-8', 'cp949']
            
            for enc in encodings:
                try:
                    with open(filepath,'r',newline='',encoding=enc) as f:
                        sample = f.read(1024)
                        f.seek(0)   
                        try:                 
                            dialect = csv.Sniffer().sniff(sample,delimiters=[',','\t'])
                            delimeter = dialect.delimiter
                        except csv.Error :
                            delimeter = ',' if ',' in sample else '\t'
                        return pd.read_csv(filepath,delimiter=delimeter, encoding=enc)
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
        input_dataframes[str_df_in_Sales_Domain_Dimension] = df_in_Sales_Domain_Dimension
        input_dataframes[str_df_in_Sales_Product_ASN] = df_in_Sales_Product_ASN


    # 입력 변수 중 데이터가 없는 경우 에러를 출력한다.
    for df in input_dataframes:
        fn_check_input_table(input_dataframes[df], df, '0')

    fn_convert_type(input_dataframes[str_df_in_Sales_Product_ASN], 'Sales Domain', str)
    fn_convert_type(input_dataframes[str_df_in_Sales_Product_ASN], 'Location', str)

    fn_convert_type(input_dataframes[str_df_in_Sales_Domain_Dimension], 'Sales Domain', str)

########################################################################################################################
# Step 01 : Sales Product ASN 전처리 (Base · Dummy · Lv6/Lv7 분리)
########################################################################################################################
########################################################################################################################
# Step 01 : Sales Product ASN 전처리 (Base · Dummy · Lv6/Lv7 분리) – FIXED
########################################################################################################################
@_decoration_
def fn_step01() -> None:
    # 0) LOAD
    df_in = input_dataframes.get(str_df_in_Sales_Product_ASN, pd.DataFrame())
    if df_in.empty:
        logger.Note('[Step01] 입력 df_in_Sales_Product_ASN 이 비어 있습니다.', LOG_LEVEL.warning())
        return    # 1) Base
    df_base = (df_in
               .drop(columns=[c for c in (Version_Name, Salse_Product_ASN) if c in df_in.columns])
               [[Sales_Domain_ShipTo, Item_Item, Location_Location]])
    output_dataframes[str_df_fn_Sales_Product_ASN] = df_base

    # 2) Sell-Out Dummy(-)
    df_dummy = (df_base
                .drop(columns=[Location_Location])
                .drop_duplicates()
                .assign(**{Location_Location: '-'})
                [[Sales_Domain_ShipTo, Item_Item, Location_Location]])
    output_dataframes[str_df_fn_Sales_Product_ASN_Dummy] = df_dummy

    # 3) Lv6 / Lv7 분리 (🆕 rule: LV7 ≠ LV6 이어야 진짜 LV7)
    dim = input_dataframes.get(str_df_in_Sales_Domain_Dimension, pd.DataFrame())
    if dim.empty:
        logger.Note('[Step01] Dimension 테이블이 없어 Lv6/Lv7 분리 생략.', LOG_LEVEL.warning())
        return

    dim_sub = dim[[Sales_Domain_Std5, Sales_Domain_Std6, Sales_Domain_ShipTo]]
    joined  = df_base.merge(dim_sub, on=Sales_Domain_ShipTo, how='left')

    # ── TRUE Lv7 : Ship-To == LV7  &  LV7 ≠ LV6  & LV7 notnull
    mask_lv7 = (
        (joined[Sales_Domain_ShipTo] == joined[Sales_Domain_Std6]) &
        (joined[Sales_Domain_Std6].notna()) &
        (joined[Sales_Domain_Std6] != joined[Sales_Domain_Std5])
    )
    # ── Lv6 : Ship-To == LV6   (나머지 포함)
    mask_lv6 = (joined[Sales_Domain_ShipTo] == joined[Sales_Domain_Std5])

    output_dataframes[str_df_fn_Sales_Product_ASN_lv7] = (
        joined.loc[mask_lv7, [Sales_Domain_ShipTo, Item_Item, Location_Location]]
              .drop_duplicates()
    )
    output_dataframes[str_df_fn_Sales_Product_ASN_lv6] = (
        joined.loc[mask_lv6, [Sales_Domain_ShipTo, Item_Item, Location_Location]]
              .drop_duplicates()
    )

########################################################################################################################
# Step 02 : LV7 Ship-To → LV6 Aggregation
########################################################################################################################
@_decoration_
def fn_step02() -> None:
    df_lv7 = output_dataframes.get(str_df_fn_Sales_Product_ASN_lv7, pd.DataFrame())
    dim    = input_dataframes .get(str_df_in_Sales_Domain_Dimension, pd.DataFrame())
    if df_lv7.empty or dim.empty:
        logger.Note('[Step02] Lv7에 대한 Aggregation 대상 또는 Dimension 이 없습니다.', LOG_LEVEL.warning())
        # df_lv7 이 Empty 인 경우에 대한 Case 처리 (25.05.08)
        output_dataframes[str_df_fn_Sales_Product_ASN_lv7_to_lv6] = df_lv7
        return
    df_lv6 = (df_lv7
              .merge(dim[[Sales_Domain_ShipTo, Sales_Domain_Std5]], on=Sales_Domain_ShipTo, how='left')
              .assign(**{Sales_Domain_ShipTo: lambda d: d[Sales_Domain_Std5]})
              .drop(columns=[Sales_Domain_Std5])
              .drop_duplicates())
    output_dataframes[str_df_fn_Sales_Product_ASN_lv7_to_lv6] = df_lv6


########################################################################################################################
# Step 03 : LV6 Concat → Dimension Join → LV2~LV5 Frame   ★PATCHED
########################################################################################################################
@_decoration_
def fn_step03() -> None:
    df_lv6_orig = output_dataframes.get(str_df_fn_Sales_Product_ASN_lv6,        pd.DataFrame())
    df_lv7_lv6  = output_dataframes.get(str_df_fn_Sales_Product_ASN_lv7_to_lv6, pd.DataFrame())
    dim         = input_dataframes .get(str_df_in_Sales_Domain_Dimension,       pd.DataFrame())    
    if dim.empty or (df_lv6_orig.empty and df_lv7_lv6.empty):
        logger.Note('[Step03] 입력 데이터가 부족합니다.', LOG_LEVEL.warning())
        # return

    # ①  LV-6 Concat
    df_concat = (pd.concat([df_lv6_orig, df_lv7_lv6], ignore_index=True)
                   .drop_duplicates())
    output_dataframes[str_df_fn_Sales_Product_ASN_Concat] = df_concat

    # ②  Dimension Join (LV2~LV7 컬럼 부착)
    dim_cols = [Sales_Domain_Std1, Sales_Domain_Std2, Sales_Domain_Std3,
                Sales_Domain_Std4, Sales_Domain_Std5, Sales_Domain_Std6,
                Sales_Domain_ShipTo]
    df_join = df_concat.merge(dim[dim_cols], on=Sales_Domain_ShipTo, how='left')
    output_dataframes[str_df_fn_Sales_Product_ASN_Concat_dim] = df_join

    # ③  LV2~LV5 프레임 (중복 컬럼 제거)  ★FIX
    frames = []
    for col in (Sales_Domain_Std4, Sales_Domain_Std3, Sales_Domain_Std2, Sales_Domain_Std1):
        df_tmp = (df_join[[col, Item_Item, Location_Location]]           # 원 Ship-To 열 제외
                  .rename(columns={col: Sales_Domain_ShipTo})
                  .dropna(subset=[Sales_Domain_ShipTo])                  # 상위 코드 없는 행 제거
                  .drop_duplicates())
        frames.append(df_tmp)

    output_dataframes[str_df_fn_Sales_Product_ASN_Lv2_to_Lv5] = (
        pd.concat(frames, ignore_index=True).drop_duplicates())

########################################################################################################################
# Step 04 : LV2~LV6 데이터 통합
########################################################################################################################
@_decoration_
def fn_step04() -> None:
    df_lv6      = output_dataframes.get(str_df_fn_Sales_Product_ASN_Concat,     pd.DataFrame())
    df_lv2_lv5  = output_dataframes.get(str_df_fn_Sales_Product_ASN_Lv2_to_Lv5, pd.DataFrame())    
    if df_lv6.empty and df_lv2_lv5.empty:
        logger.Note('[Step04] 통합할 데이터가 없습니다.', LOG_LEVEL.warning())
        return

    output_dataframes[str_df_fn_Sales_Product_ASN_Lv2_to_Lv6] = (
        pd.concat([df_lv6, df_lv2_lv5], ignore_index=True).drop_duplicates())

########################################################################################################################
# Step 05 : Sell-Out Dummy(-) 생성 (LV2~LV6)  ★PATCHED
########################################################################################################################
@_decoration_
def fn_step05() -> None:
    df_lv2_lv6 = output_dataframes.get(str_df_fn_Sales_Product_ASN_Lv2_to_Lv6, pd.DataFrame())
    if df_lv2_lv6.empty:
        logger.Note('[Step05] LV2~LV6 데이터가 없습니다 – Dummy 생략.', LOG_LEVEL.warning())
    
    df_dummy = (df_lv2_lv6
        .drop(columns=[Location_Location])
        .drop_duplicates()
        .assign(**{Location_Location: '-'})
        .reset_index(drop=True)                                 # ←★ index 제거
        [[Sales_Domain_ShipTo, Item_Item, Location_Location]]
    )

    output_dataframes[str_df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy] = df_dummy

########################################################################################################################
# Step 06 : 최종 Sales Product ASN 결과 집계
########################################################################################################################
@_decoration_
def fn_step06() -> None:
    parts = [
        output_dataframes.get(str_df_fn_Sales_Product_ASN_Lv2_to_Lv6,       pd.DataFrame()),
        output_dataframes.get(str_df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy, pd.DataFrame()),
        output_dataframes.get(str_df_fn_Sales_Product_ASN_Dummy,            pd.DataFrame())
    ]
    df_final = pd.concat(parts, ignore_index=True).drop_duplicates()
    if df_final.empty:
        logger.Note('[Step06] 합칠 데이터가 없습니다.', LOG_LEVEL.warning())
    
    df_final[Salse_Product_ASN] = 'Y'
    output_dataframes[str_df_fn_Sales_Product_ASN_Final] = df_final[
        [Sales_Domain_ShipTo, Item_Item, Location_Location, Salse_Product_ASN]
    ]

@_decoration_
def fn_output_formatter(Param_OUT_VERSION: str) -> None:
    """

    """

    # Define column orders for each DataFrame, 
    # matching your ERD and including the relevant measure columns.

    # 1) Output_Sales_Product_ASN
    required_cols = [
        Version_Name,
        Sales_Domain_ShipTo,
        Item_Item,
        Location_Location,
        Salse_Product_ASN
    ]
    

    # A helper function that checks if df is empty, 
    # ensures necessary columns exist, sets the version name, and reorders columns.
    def align_and_set_version(
        df: pd.DataFrame, col_order: list, version_str: str
    ) -> pd.DataFrame:
        if df is None or df.empty:
            df = pd.DataFrame(columns=col_order)

        # Ensure all needed columns exist
        for c in col_order:
            if c not in df.columns:
                df[c] = None

        # Insert version name
        df[Version_Name] = version_str

        # Reorder columns exactly
        df = df[col_order]
        return df

    # For convenience, create a list of (dataframe_name, required_cols).
    # We'll loop through them, apply align_and_set_version,
    # and store back into output_dataframes.

    sin_dfs = [
        (str_df_fn_Sales_Product_ASN_Final, str_Output_Sales_Product_ASN , required_cols, )
    ]

    # Process Sell-In
    for df_name_origin, df_name_to, col_order in sin_dfs:
        df_temp = output_dataframes.get(df_name_origin, pd.DataFrame())
        df_temp = align_and_set_version(df_temp, col_order, Param_OUT_VERSION)
        output_dataframes[df_name_to] = df_temp


    # (Optionally) log or verify the final columns
    # fn_log_dataframe(output_dataframes['Output_SIn_Assortment_GC'], 'Final SIn GC after format')
    # ...

####################################
############ Start Main  ###########
####################################
if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    try:
        ################################################################################################################
        # 전처리 : 모듈 내에서 사용될 데이터에 대한 정합성 체크 및 데이터 선 가공
        ################################################################################################################
        
        if is_local:
            Version = 'CWV_DP'
            # input_folder_name  = str_instance           
            input_folder_name  = f'{str_instance}'
            output_folder_name = f'{str_instance}'
            
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
        
        # vdLog 초기화
        log_path = os.path.dirname(__file__) if is_local else ""
        vdCommon.gfn_pyLog_start(Version, str_instance, logger, is_local, log_path)
        # df_input 처리
        fn_process_in_df_mst()
        for in_df in input_dataframes:
            # 로그출력
            fn_log_dataframe(input_dataframes[in_df], in_df)

        # 입력 변수 확인용 로그
        logger.Note(p_note=f'Param_OUT_VERSION : {Version}', p_log_level=LOG_LEVEL.debug())

        # 입력 변수 확인
        if Version is None or Version.strip() == '':
            Version = 'CWV_DP'

        
        # 입력 변수 확인


        ################################################################################################################
        # Step 01) ASN 전처리
        ################################################################################################################
        dict_log = {'p_step_no': 10, 'p_step_desc': 'Step 01) ASN 전처리', 'p_df_name': None}
        fn_step01(**dict_log)
        fn_log_dataframe(output_dataframes[str_df_fn_Sales_Product_ASN],f'fn_step01_{str_df_fn_Sales_Product_ASN}')
        fn_log_dataframe(output_dataframes[str_df_fn_Sales_Product_ASN_Dummy],f'fn_step01_{str_df_fn_Sales_Product_ASN_Dummy}')
        fn_log_dataframe(output_dataframes[str_df_fn_Sales_Product_ASN_lv6],f'fn_step01_{str_df_fn_Sales_Product_ASN_lv6}')
        fn_log_dataframe(output_dataframes[str_df_fn_Sales_Product_ASN_lv7],f'fn_step01_{str_df_fn_Sales_Product_ASN_lv7}')
        
        ################################################################################################################
        # Step 02) LV7 → LV6
        ################################################################################################################
        dict_log = {'p_step_no': 20, 'p_step_desc': 'Step 02) LV7 → LV6', 'p_df_name': None}
        fn_step02(**dict_log)
        fn_log_dataframe(output_dataframes[str_df_fn_Sales_Product_ASN_lv7_to_lv6],f'fn_step02_{str_df_fn_Sales_Product_ASN_lv7_to_lv6}')


        ################################################################################################################
        # Step 03) Hierarchy Build
        ################################################################################################################
        dict_log = {'p_step_no': 30, 'p_step_desc': 'Step 03) Hierarchy Build', 'p_df_name': None}
        fn_step03(**dict_log)
        fn_log_dataframe(output_dataframes[str_df_fn_Sales_Product_ASN_Concat],f'fn_step03_{str_df_fn_Sales_Product_ASN_Concat}')
        fn_log_dataframe(output_dataframes[str_df_fn_Sales_Product_ASN_Concat_dim],f'fn_step03_{str_df_fn_Sales_Product_ASN_Concat_dim}')
        fn_log_dataframe(output_dataframes[str_df_fn_Sales_Product_ASN_Lv2_to_Lv5],f'fn_step03_{str_df_fn_Sales_Product_ASN_Lv2_to_Lv5}')

        ################################################################################################################
        # Step 04) LV2~LV6 Concat
        ################################################################################################################
        dict_log = {'p_step_no': 40, 'p_step_desc': 'Step 04) LV2~LV6 Concat', 'p_df_name': None}
        fn_step04(**dict_log)
        fn_log_dataframe(output_dataframes[str_df_fn_Sales_Product_ASN_Lv2_to_Lv6],f'fn_step04_{str_df_fn_Sales_Product_ASN_Lv2_to_Lv6}')

        ################################################################################################################
        # Step 05) Sell-Out Dummy
        ################################################################################################################
        dict_log = {'p_step_no': 50, 'p_step_desc': 'Step 05) Sell-Out Dummy', 'p_df_name': None}
        fn_step05(**dict_log)
        fn_log_dataframe(output_dataframes[str_df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy],f'fn_step05_{str_df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy}')

        ################################################################################################################
        # Step 06) Finalize
        ################################################################################################################
        dict_log = {'p_step_no': 60, 'p_step_desc': 'Step 06) Finalize', 'p_df_name': None}
        fn_step06(**dict_log)
        fn_log_dataframe(output_dataframes[str_df_fn_Sales_Product_ASN_Final],f'fn_step06_{str_df_fn_Sales_Product_ASN_Final}')


        

        ################################################################################################################
        # Formatter:  Add Version Name 
        ################################################################################################################
        dict_log = {
            'p_step_no': 900,
            'p_step_desc': '최종 Output 정리 - out_Sellout'
        }
        fn_output_formatter(Version,**dict_log)

        # ════════════════ log data  ════════════════
        fn_log_dataframe(output_dataframes[str_Output_Sales_Product_ASN], str_Output_Sales_Product_ASN)
        
        Output_Sales_Product_ASN = output_dataframes[str_Output_Sales_Product_ASN]
        

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
        fn_set_header()

        if is_local:
            log_file_name = common.G_PROGRAM_NAME.replace('py', 'log')
            log_file_name = f'log/{log_file_name}'

            shutil.copyfile(log_file_name, os.path.join(str_output_dir, os.path.basename(log_file_name)))

            # prografile copy
            program_path = f"{os.getcwd()}/NSCM_DP_UI_Develop/{str_instance}.py"
            shutil.copyfile(program_path, os.path.join(str_output_dir, os.path.basename(program_path)))

        logger.Finish()
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')