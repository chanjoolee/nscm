from re import X
import os,sys,json,shutil,io,zipfile
import time
import datetime
import inspect
import traceback
import pandas as pd
from NSCMCommon import NSCMCommon as common
# from typing_extensions import Literal
import glob
import numpy as np

########################################################################################################################
# Local 개발 시에 필요한 공통 변수 선언
########################################################################################################################
# o9에 저장된 instanceName
is_local = common.gfn_get_isLocal()
str_instance = 'PYForecastMeasureLockColor'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"

is_print = True
flag_csv = True
flag_exception = True
# Global variable for max_week
max_week = None
current_partial_week = None
max_week_normalized = None
current_week_normalized = None
# read size of chunk. file location Output/df_07_join_sales_product_asn_to_lvl.csv
chunk_size_step08 = None
chunk_size_step09 = None
chunk_size_step10 = None
chunk_size_step11 = None
chunk_size_step12 = None
chunk_size_step13 = None
files_step08 = []
files_step09 = []
files_step10 = []
files_step11 = []
files_step12 = []
files_step13 = []
files_step14 = []

dfs_step08 = []
dfs_step09 = []
dfs_step10 = []
dfs_step11 = []
dfs_step12 = []
dfs_step13 = []
dfs_step14 = []

########################################################################################################################
# 컬럼상수
########################################################################################################################

Version_Name = 'Version.[Version Name]'
Item_Item = 'Item.[Item]'
Item_Type = 'Item.[Item Type]'  
Item_GBM = 'Item.[Item GBM]'
Item_Product_Group = 'Item.[Product Group]'
Item_Lv = 'Item_Lv'
Sales_Domain_Ship_To    = 'Sales Domain.[Ship To]'
Sales_Domain_LV2        = 'Sales Domain.[Sales Domain LV2]'
Sales_Domain_LV3        = 'Sales Domain.[Sales Domain LV3]'
Sales_Domain_LV4        = 'Sales Domain.[Sales Domain LV4]' 
Sales_Domain_LV5        = 'Sales Domain.[Sales Domain LV5]'
Sales_Domain_LV6        = 'Sales Domain.[Sales Domain LV6]'
Sales_Domain_LV7        = 'Sales Domain.[Sales Domain LV7]'
Location_Location       = 'Location.[Location]'
Item_Class              = 'ITEMCLASS Class'
Partial_Week            = 'Time.[Partial Week]'
SIn_FCST_GC_LOCK        = 'S/In FCST(GI)_GC.Lock'
SIn_FCST_Color_Condition        = 'S/In FCST Color Condition'

CURRENT_ROW_WEEK        = 'current_row_partial_week_normalized'
CURRENT_ROW_WEEK_PLUS_8 = 'CURRENTWEEK_NORMALIZED_PLUS_8'

EOS_WEEK_NORMALIZED             =  "EOS_WEEK_NORMALIZED"
EOS_WEEK_NORMALIZED_MINUS_1     = "EOS_WEEK_NORMALIZED_MINUS_1"
Salse_Product_ASN       = 'Sales Product ASN'

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
def fn_set_header() -> pd.DataFrame:
    """
    MediumWeight로 실행 시 발생할 수 있는 Live Server에서의 오류를 방지하기 위해 Header만 있는 Output 테이블을 만든다.
    :return: DataFrame
        """
    df_return = pd.DataFrame()

    # out_Demand
    df_return = pd.DataFrame(
            {'Version.[Version Name]': [], 'Item.[Item]': [], 'Location.[Location]': [], 'Time.[Week]': [],
             'W Quantity Max Target': []})

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

def initialize_max_week(is_local, args):
    global max_week, current_partial_week, max_week_normalized, current_week_normalized,chunk_size_step08 , chunk_size_step09, chunk_size_step10 , chunk_size_step11, chunk_size_step12, chunk_size_step13
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

    # Initialize current_partial_week
    current_partial_week = common.gfn_get_partial_week(p_datetime=datetime.datetime.now())
    current_partial_week = '202447'
    if args.get('current_partial_week') is not None:
        current_partial_week = args.get('current_partial_week')
    # Initialize current_week_normalized
    current_week_normalized = normalize_week(current_partial_week)
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
            df[column] = df[column].astype(type)

@_decoration_
def fn_process_in_df_mst():
    global input_dataframes, csv_files

    # 로컬인 경우 Output 폴더를 정리한다.
    for file in os.scandir(str_output_dir):
        os.remove(file.path)

    # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
    file_pattern = f"{os.getcwd()}/{str_input_dir}/*.csv" 
    csv_files = glob.glob(file_pattern)

    file_to_df_mapping = {
        "df_in_Sales_Domain_Dimension.csv"   :      "df_in_Sales_Domain_Dimension"    ,
        "df_in_Time_Partial Week.csv"        :      "df_in_Time_Partial_Week"              ,
        "MST_ITEMCLASS.csv"                  :      "df_in_Item_CLASS"                 ,
        "MST_ITEMTAT.csv"                    :      "df_in_Item_TAT"                          ,
        "MST_MODELEOS.csv"                   :      "df_in_MST_EOS"                    ,
        "MST_MODELRTS.csv"                   :      "df_in_MST_RTS"                    ,
        "MST_SALESPRODUCT.csv"               :      "df_in_Sales_Product_ASN"              ,
        "MTA_FORECASTRULE.csv"               :      "df_in_Forecast_Rule"                  ,
        "VUI_ITEMATTB.csv"                   :      "df_in_Item_Master"                
    }

    # Read all CSV files into a dictionary of DataFrames
    for file in csv_files:
        df = pd.read_csv(file)
        file_name = file.split("/")[-1].split("\\")[-1].split(".")[0]
        # df['SourceFile'] = file_name
        # df.set_index('SourceFile',inplace=True)
        mapped = False
        for keyword, frame_name in file_to_df_mapping.items():
            if file_name.startswith(keyword.split('.')[0]):
                input_dataframes[frame_name] = df
                mapped = True
                break
        # if not mapped:
        #     input_dataframes[file_name] = df

    
    # input_dataframes['df_in_BO_FCST']['BO FCST'].fillna(0, inplace=True)
    # fn_convert_type(input_dataframes['df_in_BO_FCST'], 'BO FCST', 'int32')
    # fn_convert_type(input_dataframes['df_in_BO_FCST'], 'BO FCST.Lock', bool)
    # input_dataframes['df_in_Total_BOD_LT']['BO Total BOD LT'].fillna(0, inplace=True)
    # fn_convert_type(input_dataframes['df_in_Total_BOD_LT'], 'BO Total BOD LT', 'int32')

    fn_convert_type(input_dataframes['df_in_MST_RTS'], 'Sales Domain', str)
    fn_convert_type(input_dataframes['df_in_MST_EOS'], 'Sales Domain', str)

    fn_convert_type(input_dataframes['df_in_Item_CLASS'], 'Sales Domain', str)
    fn_convert_type(input_dataframes['df_in_Item_CLASS'], 'Location', str)
    fn_convert_type(input_dataframes['df_in_Item_CLASS'], 'Item', str)
    fn_convert_type(input_dataframes['df_in_Item_CLASS'], 'ITEMCLASS', str)

    fn_convert_type(input_dataframes['df_in_Sales_Product_ASN'], 'Sales Domain', str)
    fn_convert_type(input_dataframes['df_in_Sales_Product_ASN'], 'Location', str)

    fn_convert_type(input_dataframes['df_in_Sales_Domain_Dimension'], 'Sales Domain', str)

@_decoration_
def fn_step01_join_rts_eos() -> pd.DataFrame:
    """
    Step 1. df_in_MST_RTS 와 df_in_MST_EOS 를 Item 과 ShipTo를 기준으로 inner Join
    """
    df_in_MST_RTS = input_dataframes.get('df_in_MST_RTS')
    df_filtered_rts_isvalid_y = df_in_MST_RTS[df_in_MST_RTS['RTS_ISVALID']=='Y']
    df_return = pd.merge(
        left=df_filtered_rts_isvalid_y,
        right=input_dataframes.get('df_in_MST_EOS'),
        on=['Item.[Item]','Sales Domain.[Ship To]'],
        how='inner',
        suffixes=('_RTS','_EOS')
    )
    df_return = df_return.drop(columns=['Version.[Version Name]_RTS'])
    df_return = df_return.drop(columns=['Version.[Version Name]_EOS'])

    df_return['Item_Lv'] =  np.where(
        df_return['Sales Domain.[Ship To]'].str.startswith("3"),3,2
    )

    return df_return


@_decoration_
def fn_step02_convert_date_to_partial_week() -> pd.DataFrame:
    """
    Step 2  : Step1의 Result에 Time을 Partial Week 으로 변환 
    """
    df_01_joined_rts_eos = output_dataframes['df_01_joined_rts_eos']
    df_return = df_01_joined_rts_eos.copy(deep=True)
    columns_to_convert = [
        'RTS_INIT_DATE',
        'RTS_DEV_DATE',
        'RTS_COM_DATE',
        'EOS_INIT_DATE',
        'EOS_CHG_DATE',
        'EOS_COM_DATE'
    ]

    columns_to_int = [
        'RTS_WEEK_NORMALIZED',
        'current_row_partial_week_normalized',
        'EOS_WEEK_NORMALIZED',
        'EOS_WEEK_NORMALIZED_MINUS_1'
    ]

    # Convert date columns to partial week
    for col in columns_to_convert:
        df_return[col] = pd.to_datetime(df_return[col])
        df_return[col] = df_return[col].apply(lambda x: common.gfn_get_partial_week(x,True) if pd.notna(x) else '')
        df_return[col].astype(str)
    
    # Create new columns related to RTS with partial week
    df_return['RTS_PARTIAL_WEEK'] = df_return.apply(
        lambda row: row['RTS_COM_DATE'] if row['RTS_STATUS'] == 'COM' else (
            row['RTS_DEV_DATE'] if pd.notna(row['RTS_DEV_DATE']) else row['RTS_INIT_DATE']
        ), axis=1
    )
    df_return[Item_Lv] = df_return[Sales_Domain_Ship_To].apply(lambda x: 3 if x.startswith("3") else 2)

    # Create new columns related to EOS with partial week
    df_return['EOS_PARTIAL_WEEK'] = df_return.apply(
        lambda row: row['EOS_COM_DATE'] if row['EOS_STATUS'] == 'COM' else (
            row['EOS_CHG_DATE'] if pd.notna(row['EOS_CHG_DATE']) else row['EOS_INIT_DATE']
        ), axis=1
    )

    # There is partial week have suffix 'A' or 'B'. 
    # And we need to add columns to removed suffix 
    # Create new columns related to RTS with normalized week    
    df_return['RTS_WEEK_NORMALIZED'] = df_return['RTS_PARTIAL_WEEK'].apply(lambda x: ''.join(filter(str.isdigit, str(x))))
    df_return['RTS_WEEK_NORMALIZED_MINUST_1'] = df_return['RTS_WEEK_NORMALIZED'].apply(lambda x: common.gfn_add_week(x, -1))
    df_return['RTS_WEEK_NORMALIZED_PLUS_3'] = df_return['RTS_WEEK_NORMALIZED'].apply(lambda x: common.gfn_add_week(x, 3))
    df_return['MAX_RTS_CURRENTWEEK'] = df_return['RTS_WEEK_NORMALIZED'].apply(lambda x: max(x, current_week_normalized))
    # Create new columns related to EOS with normalized week
    df_return['EOS_WEEK_NORMALIZED'] = df_return['EOS_PARTIAL_WEEK'].apply(lambda x: ''.join(filter(str.isdigit, str(x))))
    df_return['EOS_WEEK_NORMALIZED_MINUS_1'] = df_return['EOS_WEEK_NORMALIZED'].apply(lambda x: common.gfn_add_week(x, -1))
    df_return['EOS_WEEK_NORMALIZED_MINUS_4'] = df_return['EOS_WEEK_NORMALIZED'].apply(lambda x: common.gfn_add_week(x, -4))
    # Create new columns related to RTS with normalized initial week
    df_return['RTS_INITIAL_WEEK_NORMALIZED'] = df_return['RTS_INIT_DATE'].apply(lambda x: ''.join(filter(str.isdigit, str(x))))
    # Create new columns related to EOS with normalized initial week
    df_return['EOS_INITIAL_WEEK_NORMALIZED'] = df_return['EOS_INIT_DATE'].apply(lambda x: ''.join(filter(str.isdigit, str(x))))
    df_return['MIN_EOSINI_MAXWEEK'] = df_return['EOS_INITIAL_WEEK_NORMALIZED'].apply(lambda x: min(x,max_week_normalized))

    return df_return



@_decoration_
def fn_step03_join_rts_eos() -> pd.DataFrame:
    """
    Step 3. df_in_MST_RTS 와 df_in_MST_EOS의 ITEM * ShipTo로 Inner Join하여 새로운 DF 생성 ( Output Data )
        첫번째 생성된 DF 에서 컬럼을 복사
    """
    fn_step03_join_rts_eos = output_dataframes['df_01_joined_rts_eos']
    df_origin = fn_step03_join_rts_eos.copy(deep=True)
    df_return = df_origin[['Item.[Item]','Sales Domain.[Ship To]']]

    return df_return


@_decoration_
def fn_step04_add_partialweek_measurecolumn() -> pd.DataFrame:
    """
    Step 4  : Step3의 df에 Partial Week 및 Measure Column 추가
    """
    df_03_joined_rts_eos =  output_dataframes['df_03_joined_rts_eos']
    df_in_Time_Partial_Week = input_dataframes['df_in_Time_Partial_Week']
    df_03_joined_rts_eos['key'] = 1
    df_in_Time_Partial_Week['key'] = 1

    # Perform the merge on the temporary key
    df_return = pd.merge(
        df_03_joined_rts_eos,
        df_in_Time_Partial_Week,
        on='key'
    )
    df_return[Item_Lv] = df_return[Sales_Domain_Ship_To].apply(lambda x: 3 if x.startswith("3") else 2)
    df_return[CURRENT_ROW_WEEK] = df_return[Partial_Week].apply(normalize_week)
    df_return[CURRENT_ROW_WEEK_PLUS_8] = df_return[CURRENT_ROW_WEEK].apply(lambda x : common.gfn_add_week(x, 8))
    
    df_return[SIn_FCST_GC_LOCK] = 'True'
    df_return[SIn_FCST_Color_Condition] = '19_GRAY'
    df_return = df_return.drop(columns=['key']).reset_index(drop=True)

    # expanded_rows = []
    # for index, row in df_03_joined_rts_eos.iterrows():
    #     for time_value in input_dataframes['df_in_Time_Partial_Week']['Time.[Partial Week]']:
    #         new_row = row.to_dict()
    #         new_row['Time.[Partial Week]'] = time_value
    #         new_row['S/In FCST(GI)_GC.Lock'] = 'True'  # Placeholder value, replace with actual logic if needed
    #         new_row['S/In FCST Color Condition'] = 'GRAY'
    #         expanded_rows.append(new_row)
    
    
    # df_return = pd.DataFrame(expanded_rows)

    return df_return



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
    # Declare global variables
    global output_dataframes
    global current_week_normalized
    global max_week_normalized

    # -----------------------------
    # 1. Load the partial-week DataFrame
    # -----------------------------
    df_week = output_dataframes['df_04_partialweek_measurecolumn']

    # Keep only the columns needed for logic (plus the GC columns we want to set)
    needed_cols_week = [
        Sales_Domain_Ship_To,
        Item_Item,
        Item_Lv,
        CURRENT_ROW_WEEK,
        CURRENT_ROW_WEEK_PLUS_8,
        SIn_FCST_GC_LOCK,
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
    df_lookup = output_dataframes['df_02_date_to_partial_week']
    # The column names in your lookup can remain raw strings,
    # or you can define more constants if desired. Here we keep
    # them as strings or minimal new constants:
    needed_cols_lookup = [
        Sales_Domain_Ship_To,
        Item_Item,
        Item_Lv,
        'MAX_RTS_CURRENTWEEK',
        'RTS_WEEK_NORMALIZED',
        'MIN_EOSINI_MAXWEEK',
        'EOS_WEEK_NORMALIZED',
        'EOS_WEEK_NORMALIZED_MINUS_4',
        'RTS_WEEK_NORMALIZED_MINUST_1',
        'RTS_WEEK_NORMALIZED_PLUS_3'
    ]
    # df_lookup = df_lookup[needed_cols_lookup]

    # -----------------------------
    # 3. Set the index in df_lookup for array-based lookups
    # -----------------------------
    df_lookup.set_index([Sales_Domain_Ship_To, Item_Item, Item_Lv], inplace=True)

    # We'll match on (Sales_Domain_Ship_To, Item_Item, Item_lv) from df_week
    multi_index_week = pd.MultiIndex.from_arrays(
        [
            df_week[Sales_Domain_Ship_To],
            df_week[Item_Item],
            df_week[Item_Lv]
        ],
        names=[Sales_Domain_Ship_To, Item_Item, Item_Lv]
    )

    # positions: integer array the same length as df_week
    positions = df_lookup.index.get_indexer(multi_index_week)
    valid_mask = (positions != -1)

    # -----------------------------
    # 4. Build arrays from df_lookup columns
    # -----------------------------
    max_rts_array          = df_lookup['MAX_RTS_CURRENTWEEK'].to_numpy(dtype=int)
    rts_week_array         = df_lookup['RTS_WEEK_NORMALIZED'].to_numpy(dtype=int)
    min_eosini_maxweek_arr = df_lookup['MIN_EOSINI_MAXWEEK'].to_numpy(dtype=int)
    eos_week_array         = df_lookup['EOS_WEEK_NORMALIZED'].to_numpy(dtype=int)
    eos_week_minus4_array  = df_lookup['EOS_WEEK_NORMALIZED_MINUS_4'].to_numpy(dtype=int)
    rts_week_minus1_array  = df_lookup['RTS_WEEK_NORMALIZED_MINUST_1'].to_numpy(dtype=int)
    rts_week_plus3_array   = df_lookup['RTS_WEEK_NORMALIZED_PLUS_3'].to_numpy(dtype=int)

    n = len(df_week)
    arr_max_rts            = np.full(n, np.nan, dtype=int)
    arr_rts_week           = np.full(n, np.nan, dtype=int)
    arr_min_eosini_maxweek = np.full(n, np.nan, dtype=int)
    arr_eos_week           = np.full(n, np.nan, dtype=int)
    arr_eos_week_minus4    = np.full(n, np.nan, dtype=int)
    arr_rts_week_minus1    = np.full(n, np.nan, dtype=int)
    arr_rts_week_plus3     = np.full(n, np.nan, dtype=int)

    # Fill valid rows
    arr_max_rts[valid_mask]            = max_rts_array[positions[valid_mask]]
    arr_rts_week[valid_mask]           = rts_week_array[positions[valid_mask]]
    arr_min_eosini_maxweek[valid_mask] = min_eosini_maxweek_arr[positions[valid_mask]]
    arr_eos_week[valid_mask]           = eos_week_array[positions[valid_mask]]
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
        (row_partial <= arr_min_eosini_maxweek)
    )
    df_week.loc[cond_white, SIn_FCST_GC_LOCK] = False
    df_week.loc[cond_white, SIn_FCST_Color_Condition] = '14_WHITE'

    # Step 5-2: Dark Blue
    cond_darkblue = (
        (row_partial >= curr_week_int) &
        (arr_rts_week_minus1 <= row_partial) &
        (row_partial <= arr_rts_week_minus1)
    )
    df_week.loc[cond_darkblue, SIn_FCST_GC_LOCK] = True
    df_week.loc[cond_darkblue, SIn_FCST_Color_Condition] = '15_DARKBLUE'

    # Step 5-3: Light Blue
    cond_lightblue = (
        (row_partial >= curr_week_int) &
        (arr_rts_week <= row_partial) &
        (row_partial <= arr_rts_week_plus3)
    )
    df_week.loc[cond_lightblue, SIn_FCST_GC_LOCK] = False
    df_week.loc[cond_lightblue, SIn_FCST_Color_Condition] = '10_LIGHTBLUE'

    # Step 5-4: Light Red
    cond_lightred = (
        (row_partial >= curr_week_int) &
        (arr_eos_week_minus4 <= row_partial) &
        (row_partial <= (arr_eos_week_minus4 + 3))
    )
    df_week.loc[cond_lightred, SIn_FCST_GC_LOCK] = False
    df_week.loc[cond_lightred, SIn_FCST_Color_Condition] = '11_LIGHTRED'

    # Step 5-5: Dark Red
    cond_darkred = (
        (row_partial >= curr_week_int) &
        (arr_eos_week <= row_partial) &
        (row_partial <= max_week_int)
    )
    df_week.loc[cond_darkred, SIn_FCST_GC_LOCK] = True
    df_week.loc[cond_darkred, SIn_FCST_Color_Condition] = '16_DARKRED'

    # -----------------------------
    # 6. Return
    # -----------------------------
    # return df_week.reset_index(drop=True)
    return df_week

@_decoration_
def fn_step06_addcolumn_green_for_wireless_bas_array_based():
    """
    Step 6 (array-based): Add or update 'S/In FCST Color Condition' to '13_GREEN'
    for 'wireless BAS' items, using pre-existing CURRENT_ROW_WEEK_PLUS_8
    to avoid calling gfn_add_week. No merges required.
    """

    global output_dataframes
    global input_dataframes
    global current_week_normalized

    # ------------------------------------------------------------------
    # 1) Load the main DataFrame we want to update, e.g. df_week
    #    Suppose it has these columns:
    #       CURRENT_ROW_WEEK          (int)
    #       CURRENT_ROW_WEEK_PLUS_8   (int)
    #       'S/In FCST Color Condition'
    #       'Item.[Item]'  -> We'll use array-based get_indexer with df_in_Item_Master
    # ------------------------------------------------------------------
    df_week = output_dataframes['df_05_set_lock_values']  # or 'df_04_partialweek_measurecolumn', etc.

    # Ensure CURRENT_ROW_WEEK is int
    df_week[CURRENT_ROW_WEEK] = df_week[CURRENT_ROW_WEEK].fillna('0').astype(int)
    df_week[CURRENT_ROW_WEEK_PLUS_8] = df_week[CURRENT_ROW_WEEK_PLUS_8].fillna('0').astype(int)

    # ------------------------------------------------------------------
    # 2) Load & index your Item Master for array-based lookups
    # ------------------------------------------------------------------
    df_item_master = input_dataframes['df_in_Item_Master']

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

    n = len(df_week)
    out_item_type = np.full(n, '', dtype=object)
    out_item_gbm  = np.full(n, '', dtype=object)

    out_item_type[valid_mask] = item_type_array[positions[valid_mask]]
    out_item_gbm[valid_mask]  = item_gbm_array[positions[valid_mask]]

    # # Optionally store them back to df_week
    # df_week['Item.[Item Type]'] = out_item_type
    # df_week['Item.[Item GBM]']  = out_item_gbm

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
        (df_week[SIn_FCST_Color_Condition] != '19_GRAY')
    )

    # ------------------------------------------------------------------
    # 5) Apply the color update in-place
    # ------------------------------------------------------------------
    df_week.loc[mask_wireless_bas, SIn_FCST_Color_Condition] = '13_GREEN'

    # If you no longer need the item type columns, you can drop them
    # df_week.drop(columns=['Item.[Item Type]', 'Item.[Item GBM]'], inplace=True)

    # ------------------------------------------------------------------
    # 6) Return the updated DataFrame
    # ------------------------------------------------------------------
    return df_week


@_decoration_
def fn_step07_join_sales_product_asn_to_lvl_by_merge() -> pd.DataFrame:
    """
    Step 7: Join df_in_Sales_Product_ASN with df_in_Sales_Domain_Dimension on specified keys
    and reorder columns. Add 'Item.[Product Group]' by merging with df_in_Item_Master.
    """

    df_in_Sales_Product_ASN = input_dataframes['df_in_Sales_Product_ASN']
    df_in_Sales_Domain_Dimension = input_dataframes['df_in_Sales_Domain_Dimension']
    df_in_Item_Master = input_dataframes['df_in_Item_Master']

    # Merge df_in_Item_Master to get 'Item.[Product Group]'
    df_fn_Sales_Product_ASN_Item = df_in_Sales_Product_ASN.merge(
        df_in_Item_Master[[Item_Item, Item_GBM, Item_Product_Group]],
        on=Item_Item,
        how='inner'
    )

    df_fn_Sales_Product_ASN_Item = df_fn_Sales_Product_ASN_Item.merge(
        df_in_Sales_Domain_Dimension[[
            Sales_Domain_Sales_Domain_LV2, 
            Sales_Domain_Sales_Domain_LV3,
            Sales_Domain_Sales_Domain_LV6,
            Sales_Domain_Ship_To]],
        on=Sales_Domain_Ship_To,
        how='inner'
    )

    return_columns = [
        Sales_Domain_Sales_Domain_LV2,
        Sales_Domain_Sales_Domain_LV3,
        Sales_Domain_Sales_Domain_LV6,
        Sales_Domain_Ship_To,
        Item_Item,
        Item_GBM,
        Item_Product_Group,
        Location_Location,
        Salse_Product_ASN
    ]
    return df_fn_Sales_Product_ASN_Item[return_columns]


@_decoration_
def fn_step08_add_weeks_to_dimention_vector_join_chunk(df_07_file:  str) :
    """
    Step 8 (Vectorized): Join df_07_join_sales_product_asn and df_06_process_color_conditions based on custom logic using vectorized operations.   df_06_addcolumn_green_for_wireless_bas
    """
    is_header = True
    # df_06_copy = df_06_addcolumn_green_for_wireless_bas.copy(deep=True)
    

    # Convert columns to string
    df_06_copy['Sales Domain.[Ship To]'] = df_06_copy['Sales Domain.[Ship To]'].astype(str)
    # columns_to_int32_df06 = [
    #     'RTS_WEEK_NORMALIZED',
    #     'EOS_WEEK_NORMALIZED',
    #     'EOS_WEEK_NORMALIZED_MINUS_1',
    #     'current_row_partial_week_normalized'
    # ]
    # df_06_copy[columns_to_int32_df06] = df_06_copy[columns_to_int32_df06].astype('int32')
    
    file_index = 1
    # with open(f"{os.getcwd()}/{str_output_dir}/df_08_add_weeks_to_dimention.csv","w", encoding='utf-8') as f_out:
    # for df_07_chunk in pd.read_csv(f"{os.getcwd()}/{str_output_dir}/{df_07_filename}.csv", chunksize=v_chunk_size):
    dict_log_80 = {
        'p_step_no': 801,
        'p_step_desc': 'Step 8  : Step7의 df에 조건 정보 추가. lvl 정보에 주차정보를 추가한다.' 
    }
    
    dict_log_90 = {
        'p_step_no': 90,
        'p_step_desc': 'Step 9  : Step8의 df에 Item CLASS 정보 필터링 ' 
    }
    dict_log_100 = {
        'p_step_no': 100,
        'p_step_desc': 'Step 10  : Step9의 df에 Item TAT 정보 필터링 및 Lock ' 
    }
    dict_log_110 = {
        'p_step_no': 110,
        'p_step_desc': 'Step 11  : Step10의 df에 Item TAT 정보 필터링 및 Lock.  DARKGRAY ' 
    }
    dict_log_120 = {
        'p_step_no': 120,
        'p_step_desc': 'Step 12  :  Item.[Item GBM] 삭제, Lock Measure Column 추가' 
    }
    dict_log_130 = {
        'p_step_no': 130,
        'p_step_desc': 'Step 13  : Set locks based on the master data from df_in_Forecast_Rule' 
    }
    dict_log_140 = {
        'p_step_no': 140,
        'p_step_desc': 'Step 14  : Add a new column Version.[Version Name] with value CurrentWorkingView to df_13_set_locks_based_on_forecast_rule, and reorder columns as specified.' 
    }

    if chunk_size_step08 is not None :
        ############################
        # Chunk 사용.
        ############################
        for df_07_chunk in pd.read_csv(f"{os.getcwd()}/{str_output_dir}/{df_07_file}.csv", 
                    chunksize=chunk_size_step08 ,
                    dtype={
                        'Sales Domain.[Sales Domain LV2]': str,
                        'Sales Domain.[Sales Domain LV3]': str,
                        'Sales Domain.[Sales Domain LV4]': str,
                        'Sales Domain.[Sales Domain LV5]': str,
                        'Sales Domain.[Sales Domain LV6]': str,
                        'Sales Domain.[Sales Domain LV7]': str,
                        'Sales Domain.[Ship To]': str
                    }
                    # , low_memory=False
                ):

            logger.info(f'reading chunk {df_07_chunk.index[0]:,} - {df_07_chunk.index[-1]:,}')

            file_index += 1
            
            df_08 = fn_step08_add_weeks_to_dimention_vector_join_chunk_sub(df_07_chunk, file_index, **dict_log_80)
            df_09 = fn_step09_filter_itemclass_to_yellow_loc(df_08,input_dataframes['df_in_Item_CLASS'],**dict_log_90)
            df_10 = fn_step10_apply_item_tat_lock_join(df_09,input_dataframes['df_in_Item_TAT'],input_dataframes['df_in_Item_Master'],**dict_log_100)
            df_11 = fn_step11_apply_item_tatset_lock_join(df_10,input_dataframes['df_in_Item_TAT'],input_dataframes['df_in_Item_Master'],**dict_log_110)
            df_12 = fn_step12_del_gbm_add_ap2_ap1(df_11,**dict_log_120)
            df_13 = fn_step13_set_locks_based_on_forecast_rule(df_12,input_dataframes['df_in_Forecast_Rule'],**dict_log_130)

            dfs_step08.append({"key": f"df_08_{str(file_index).zfill(3)}", "value": df_08})
            dfs_step09.append({"key": f"df_09_{str(file_index).zfill(3)}", "value": df_09})
            dfs_step10.append({"key": f"df_10_{str(file_index).zfill(3)}", "value": df_10})
            dfs_step11.append({"key": f"df_11_{str(file_index).zfill(3)}", "value": df_11})
            dfs_step12.append({"key": f"df_12_{str(file_index).zfill(3)}", "value": df_12})
            dfs_step13.append({"key": f"df_12_{str(file_index).zfill(3)}", "value": df_13})
    else:
        ############################
        # Chunk 사용안함.
        ############################
        df_08 = fn_step08_add_weeks_to_dimention_vector_join_chunk_sub(output_dataframes["df_07_join_sales_product_asn_to_lvl"], df_06_copy, file_index, **dict_log_80)
        output_dataframes["df_08"] = df_08
        fn_check_input_table(df_08, 'df_08', '0')
        fn_log_dataframe(df_08,'df_08')

        df_09 = fn_step09_filter_itemclass_to_yellow_loc(df_08,input_dataframes['df_in_Item_CLASS'],**dict_log_90)
        output_dataframes["df_90"] = df_09
        fn_check_input_table(df_09, 'df_90', '0')
        fn_log_dataframe(df_09,'df_90')

        df_10 = fn_step10_apply_item_tat_lock_join(df_09,input_dataframes['df_in_Item_TAT'],input_dataframes['df_in_Item_Master'],**dict_log_100)
        output_dataframes["df_10"] = df_10
        fn_check_input_table(df_10, 'df_10', '0')
        fn_log_dataframe(df_09,'df_10')
        
        df_11 = fn_step11_apply_item_tatset_lock_join(df_10,input_dataframes['df_in_Item_TAT'],input_dataframes['df_in_Item_Master'],**dict_log_110)
        output_dataframes["df_11"] = df_11
        fn_check_input_table(df_11, 'df_11', '0')
        fn_log_dataframe(df_11,'df_11')
        
        df_12 = fn_step12_del_gbm_add_ap2_ap1(df_11,**dict_log_120)
        output_dataframes["df_12"] = df_12
        fn_check_input_table(df_12, 'df_12', '0')
        fn_log_dataframe(df_12,'df_12')
        
        df_13 = fn_step13_set_locks_based_on_forecast_rule(df_12,input_dataframes['df_in_Forecast_Rule'],**dict_log_130)
        output_dataframes["df_13"] = df_13
        fn_check_input_table(df_13, 'df_13', '0')
        fn_log_dataframe(df_13,'df_13')

@_decoration_
def fn_step08_add_weeks_to_dimention_vector_join_chunk_sub(df_07_chunk, file_index) -> pd.DataFrame:
    df_06_addcolumn_green_for_wireless_bas = output_dataframes["df_06_addcolumn_green_for_wireless_bas"]
    df_06 = df_06_addcolumn_green_for_wireless_bas
    df_06.reset_index(inplace=True)
    
    duplicated_columns = df_06.columns[df_06.columns.isin(df_07_chunk.columns)].tolist()
    unduplicated_columns = [col for col in df_06.columns if col not in df_07_chunk.columns]
    # df_06_copy_to_merge = df_06[list(set(unduplicated_columns + ['Item.[Item GBM]','Item.[Product Group]','Item.[Item]', 'Sales Domain.[Ship To]']))]
    df_06_copy_to_merge = df_06_copy_to_merge.rename(columns={'Sales Domain.[Ship To]': 'Sales Domain.[Sales Domain LV2]'})

    merged_df_lv2 = pd.merge(
        df_07_chunk[
            'Sales Domain.[Ship To]',
            'Item.[Item GBM]',
            'Item.[Product Group]',
            'Item.[Item]', 
            'Sales Domain.[Sales Domain LV2]',
            'Sales Domain.[Sales Domain LV3]',
            'Sales Domain.[Sales Domain LV6]',
            'Location.[Location]',
            'Sales Product ASN'
        ],
        df_06_copy_to_merge[
            'Item.[Item GBM]',
            'Item.[Product Group]',
            'Item.[Item]', 
            # Renamed for merge from 'Sales Domain.[Ship To]'
            'Sales Domain.[Sales Domain LV2]',
            'Time.[Partial Week]',
            # 'current_row_partial_week_normalized',
            # 'CURRENTWEEK_NORMALIZED_PLUS_8',
            # 'RTS_WEEK_NORMALIZED',
            # 'EOS_WEEK_NORMALIZED',
            # 'EOS_WEEK_NORMALIZED_MINUS_1',
            'S/In FCST(GI)_GC.Lock',
            'S/In FCST Color Condition'
            
        ],
        left_on=['Item.[Item GBM]','Item.[Product Group]','Item.[Item]', 'Sales Domain.[Sales Domain LV2]'],
        right_on=['Item.[Item GBM]','Item.[Product Group]','Item.[Item]', 'Sales Domain.[Sales Domain LV2]'],
        how='inner',
        suffixes=('', '_y')
    )
    fn_use_x_after_join(merged_df_lv2)
    merged_df_lv2['Item_Lv'] = 2
    merged_df_lv2 = merged_df_lv2[
        'Sales Domain.[Ship To]',
        'Sales Domain.[Sales Domain LV2]',
        'Sales Domain.[Sales Domain LV3]',
        'Sales Domain.[Sales Domain LV6]',
        'Item.[Item]',
        'Item_Lv',
        # 'Item.[Product Group]',
        'Location.[Location]',
        'Sales Product ASN',
        'Time.[Partial Week]',
        # 'current_row_partial_week_normalized',
        # 'CURRENTWEEK_NORMALIZED_PLUS_8',
        # 'RTS_WEEK_NORMALIZED',
        # 'EOS_WEEK_NORMALIZED',
        # 'EOS_WEEK_NORMALIZED_MINUS_1',
        'S/In FCST(GI)_GC.Lock',
        'S/In FCST Color Condition'
    ]
    
    df_06_copy_to_merge = df_06_copy_to_merge.rename(columns={'Sales Domain.[Sales Domain LV2]': 'Sales Domain.[Sales Domain LV3]'})
    merged_df_lv3 = pd.merge(
        df_07_chunk[
            'Sales Domain.[Ship To]',
            'Item.[Item GBM]',
            'Item.[Product Group]',
            'Item.[Item]', 
            'Sales Domain.[Sales Domain LV2]',
            'Sales Domain.[Sales Domain LV3]',
            'Sales Domain.[Sales Domain LV6]',
            'Location.[Location]',
            'Sales Product ASN'
        ],
        df_06_copy_to_merge[
            'Item.[Item GBM]',
            'Item.[Product Group]',
            'Item.[Item]', 
            # Renamed for merge from 'Sales Domain.[Ship To]'
            'Sales Domain.[Sales Domain LV3]',
            'Time.[Partial Week]',
            # 'current_row_partial_week_normalized',
            # 'CURRENTWEEK_NORMALIZED_PLUS_8',
            # 'RTS_WEEK_NORMALIZED',
            # 'EOS_WEEK_NORMALIZED',
            # 'EOS_WEEK_NORMALIZED_MINUS_1',
            'S/In FCST(GI)_GC.Lock',
            'S/In FCST Color Condition'
            
        ],
        left_on=['Item.[Item GBM]','Item.[Product Group]','Item.[Item]', 'Sales Domain.[Sales Domain LV3]'],
        right_on=['Item.[Item GBM]','Item.[Product Group]','Item.[Item]', 'Sales Domain.[Sales Domain LV3]'],
        how='inner',
        suffixes=('', '_y')
    )
    fn_use_x_after_join(merged_df_lv3)
    merged_df_lv2['Item_Lv'] = 3
    merged_df_lv3 = merged_df_lv3[
        'Sales Domain.[Ship To]',
        'Sales Domain.[Sales Domain LV2]',
        'Sales Domain.[Sales Domain LV3]',
        'Sales Domain.[Sales Domain LV6]',
        'Item.[Item]',
        'Item_Lv',
        # 'Item.[Product Group]',
        'Location.[Location]',
        'Sales Product ASN',
        'Time.[Partial Week]',
        # 'current_row_partial_week_normalized',
        # 'CURRENTWEEK_NORMALIZED_PLUS_8',
        # 'RTS_WEEK_NORMALIZED',
        # 'EOS_WEEK_NORMALIZED',
        # 'EOS_WEEK_NORMALIZED_MINUS_1',
        'S/In FCST(GI)_GC.Lock',
        'S/In FCST Color Condition'
    ]

    df_return = pd.concat([merged_df_lv2, merged_df_lv3], ignore_index=True)
    df_return.reset_index(inplace=True)
    df_return.loc[
        df_return['Sales Product ASN'] == 'N',
        ['S/In FCST(GI)_GC.Lock','S/In FCST Color Condition']
    ] = [True, '19_GRAY']


   

    if chunk_size_step08 is not None:
        # df_full.to_csv(f_out, encoding="UTF8", index=False, header=is_header)
        file_name = f"df_08_{str(file_index).zfill(3)}"
        # fn_log_dataframe(df_full,file_name)
        fn_check_input_table(df_return, file_name, '0')
        files_step08.append(file_name)

    return df_return

@_decoration_
def fn_step09_filter_itemclass_to_yellow_loc(df_08: pd.DataFrame, df_in_Item_CLASS: pd.DataFrame) -> pd.DataFrame:
    """
    Step 9: Filter df_p_source based on conditions from df_in_Item_CLASS. df_p_source is df_08_add_weeks_to_dimention.
            And Set 'S/In FCST Color Condition' to YELLOW if it is within the range of EOS Week.
            In df_in_Item_CLASS, there is declared yello based on 
        
    """
    df_fn_RTS_EOS = output_dataframes['df_02_date_to_partial_week']
    df_fn_RTS_EOS_Week = output_dataframes['df_02_date_to_partial_week']
    df_fn_Sales_Product_ASN_Item_Week = df_08

    #########################################
    # Level2
    #########################################
    ##### lookup for df_fn_RTS_EOS
    lookup_rts_eos = df_fn_RTS_EOS.set_index([Item_Item,Item_Lv,Sales_Domain_Ship_To])

    for sales_domain_level in [Sales_Domain_LV2 , Sales_Domain_LV3]:
        positions_rts_eos = lookup_rts_eos.index.get_indexer(pd.MultiIndex.from_arrays([
            df_fn_Sales_Product_ASN_Item_Week[Item_Item],
            df_fn_Sales_Product_ASN_Item_Week[Item_Lv],
            df_fn_Sales_Product_ASN_Item_Week[sales_domain_level],
        ]))
        valid_mask_rts_eos = (positions_rts_eos != -1)

        # Pull the columns we need
        EOS_WEEK_NORMALIZED_array = lookup_rts_eos[EOS_WEEK_NORMALIZED].to_numpy()
        EOS_WEEK_NORMALIZED_MINUS_1_array = lookup_rts_eos[EOS_WEEK_NORMALIZED_MINUS_1].to_numpy()

        # Build arrays for each row of df_fn_Sales_Product_ASN_Item_Week
        fetched_eos_week = np.full(len(df_fn_Sales_Product_ASN_Item_Week), np.nan, dtype=object)
        fetched_eos_minus_1 = np.full(len(df_fn_Sales_Product_ASN_Item_Week), np.nan, dtype=object)

        fetched_eos_week[valid_mask_rts_eos] = EOS_WEEK_NORMALIZED_array[positions_rts_eos[valid_mask_rts_eos]]
        fetched_eos_minus_1[valid_mask_rts_eos] = EOS_WEEK_NORMALIZED_MINUS_1_array[positions_rts_eos[valid_mask_rts_eos]]

        ##### lookup for df_fn_RTS_EOS_Week
        lookup_week = df_fn_RTS_EOS_Week.set_index([Item_Item,Item_Lv,Sales_Domain_Ship_To,Partial_Week])

        positions_week = lookup_week.index.get_indexer(pd.MultiIndex.from_arrays([
            df_fn_Sales_Product_ASN_Item_Week[Item_Item],
            df_fn_Sales_Product_ASN_Item_Week[Item_Lv],
            df_fn_Sales_Product_ASN_Item_Week[sales_domain_level],
            df_fn_Sales_Product_ASN_Item_Week[Partial_Week],
        ]))

        valid_mask_week = (positions_week != -1)
        # Pull the columns we need
        current_row_week_array = lookup_week["current_row_partial_week_normalized"].to_numpy()

        # Build arrays for each row of df_fn_Sales_Product_ASN_Item_Week
        fetched_current_row_week = np.full(len(df_fn_Sales_Product_ASN_Item_Week), np.nan, dtype=object)
        fetched_current_row_week[valid_mask_week] = current_row_week_array[positions_week[valid_mask_week]]

        # Assuming merged_df_lv_2_to_7 is already defined in the context
        condition_1 = (
            (pd.Series(fetched_eos_week).astype(int) >= int(max_week_normalized)) & 
            (pd.Series(fetched_current_row_week).astype(int) >= int(max_week_normalized)) & 
            (pd.Series(fetched_current_row_week).astype(int) >= pd.Series(fetched_eos_minus_1).astype(int)) &
            (df_fn_Sales_Product_ASN_Item_Week['S/In FCST Color Condition'] != '11_LIGHTRED') &
            (df_fn_Sales_Product_ASN_Item_Week['S/In FCST Color Condition'] != '16_DARKRED')
        )

        condition_2 = (
            (pd.Series(fetched_eos_week).astype(int) < int(max_week_normalized)) & 
            (pd.Series(fetched_current_row_week).astype(int) >= int(max_week_normalized)) & 
            (pd.Series(fetched_current_row_week).astype(int) <= pd.Series(fetched_eos_week).astype(int)) &
            (df_fn_Sales_Product_ASN_Item_Week['S/In FCST Color Condition'] != '11_LIGHTRED') &
            (df_fn_Sales_Product_ASN_Item_Week['S/In FCST Color Condition'] != '16_DARKRED')
        )


        df_fn_Sales_Product_ASN_Item_Week.loc[condition_1, [SIn_FCST_GC_LOCK, SIn_FCST_Color_Condition]] = [False, 'YELLOW']
        df_fn_Sales_Product_ASN_Item_Week.loc[condition_2, [SIn_FCST_GC_LOCK, SIn_FCST_Color_Condition]] = [False, 'YELLOW']

    
    # return merged_df_lv_2_to_7, df_merged_full
    return df_fn_Sales_Product_ASN_Item_Week


@_decoration_
def fn_step10_apply_item_tat_lock_join(df_09_itemclass_to_yellow_all: pd.DataFrame, df_in_Item_TAT: pd.DataFrame, df_in_Item_Master: pd.DataFrame) -> pd.DataFrame:
    """
    Step 10: Filter df_09_itemclass_to_yellow_all based on conditions from df_in_Item_TAT and additional hardcoded conditions.
    """

    df_09 = df_09_itemclass_to_yellow_all.copy(deep=True)
    df_09 = df_09_itemclass_to_yellow_all
    

    # Merge df_in_Item_TAT with df_in_Item_Master to get 'Item.[Item GBM]' and filter by 'Item.[Item GBM]' == 'VD'
    df_tat_item = pd.merge(df_in_Item_TAT, df_in_Item_Master[['Item.[Item]', 'Item.[Item GBM]']], on='Item.[Item]', how='inner')
    df_tat_item = df_tat_item[df_tat_item['Item.[Item GBM]'] == 'VD']

    df_tat_item['CURWEEK_MINUS_TATTERM'] = df_tat_item['ITEMTAT TATTERM'].apply(lambda x: common.gfn_add_week(current_week_normalized, int(x) -1))
    columns_to_int = ['ITEMTAT TATTERM','ITEMTAT TATTERM_SET' ,'CURWEEK_MINUS_TATTERM']
    df_tat_item[columns_to_int] = df_tat_item[columns_to_int].astype('int32')

    # # convert string to merge
    # columns_to_str_09 = [
    #     'Sales Domain.[Sales Domain LV2]',
    #     'Sales Domain.[Sales Domain LV3]',
    #     'Sales Domain.[Sales Domain LV4]',
    #     'Sales Domain.[Sales Domain LV5]',
    #     'Sales Domain.[Sales Domain LV6]',
    #     'Sales Domain.[Sales Domain LV7]',
    #     'Sales Domain.[Ship To]',
    #     # 'Item.[Item]',
    #     # 'Item.[Item GBM]',
    #     # 'Item.[Product Group]',
    #     # 'Location.[Location]',
    #     'SalesProductASN_ISVALID',
    #     'Time.[Partial Week]',
    #     # 'S/In FCST(GI)_GC.Lock',
    #     # 'S/In FCST Color Condition',
    #     'RTS_WEEK_NORMALIZED',
    #     'current_row_partial_week_normalized',
    #     'EOS_WEEK_NORMALIZED'
    # ]
    # for col in columns_to_str_09:
    #     df_09[col] = df_09[col].astype(str)

    columns_to_str_tat = [
        'Location.[Location]',
        'Item.[Item]'
    ]
    for col in columns_to_str_tat:
        df_tat_item[col] = df_tat_item[col].astype(str)


    
    # Merge df_09 with df_tat_item on specified columns
    #   - 'Item.[Item]', 'Item.[Item GBM]', 'Item.[Product Group]', 'Location.[Location]'
    #   - 'inner' join, meaning intersection of two dataframes
    merged_tat_09 = pd.merge(
        df_09,
        df_tat_item,
        left_on=[
            'Item.[Item]',
            'Item.[Item GBM]',
            'Location.[Location]'
        ],
        right_on=[
            'Item.[Item]',
            'Item.[Item GBM]',
            'Location.[Location]'
        ],
        how='inner'  # Choose 'left', 'right', 'inner', or 'outer' depending on your needs
    )
    # Rename columns with '_x' suffix to their original names
    fn_use_x_after_join(merged_tat_09)
    # fn_log_dataframe(merged_df_lv_6,'df_09_merged_df_lv_6')


    condition_tat = (
        (merged_tat_09['current_row_partial_week_normalized'] >= int(current_week_normalized)) &
        (merged_tat_09['current_row_partial_week_normalized'] <= merged_tat_09['CURWEEK_MINUS_TATTERM'])
    )
    merged_tat_09.loc[condition_tat,['S/In FCST(GI)_GC.Lock', 'S/In FCST Color Condition']] = [True, 'DGREY_RED']

    df_merged_full  = pd.merge(
        df_09,
        merged_tat_09,
        left_on=[
            'Sales Domain.[Sales Domain LV2]', 
            'Sales Domain.[Sales Domain LV3]', 
            'Sales Domain.[Sales Domain LV4]', 
            'Sales Domain.[Sales Domain LV5]', 
            'Sales Domain.[Sales Domain LV6]', 
            'Sales Domain.[Sales Domain LV7]',
            'Sales Domain.[Ship To]',
            'Location.[Location]', 
            'Item.[Item]',
            'Item.[Item GBM]',
            'Item.[Product Group]',
            'Location.[Location]',
            'Time.[Partial Week]'

        ],
        right_on=[
            'Sales Domain.[Sales Domain LV2]', 
            'Sales Domain.[Sales Domain LV3]', 
            'Sales Domain.[Sales Domain LV4]', 
            'Sales Domain.[Sales Domain LV5]', 
            'Sales Domain.[Sales Domain LV6]',
            'Sales Domain.[Sales Domain LV7]',
            'Sales Domain.[Ship To]',
            'Location.[Location]', 
            'Item.[Item]',
            'Item.[Item GBM]',
            'Item.[Product Group]',
            'Location.[Location]',
            'Time.[Partial Week]'
        ],
        how='left'  # Choose 'left', 'right', 'inner', or 'outer' depending on your needs
    )

    # Update 'S/In FCST(GI)_GC.Lock_x' where 'S/In FCST(GI)_GC.Lock_y' is not NaN and values are different
    condition_lock = (~df_merged_full['S/In FCST(GI)_GC.Lock_y'].isna()) & (df_merged_full['S/In FCST(GI)_GC.Lock_x'] != df_merged_full['S/In FCST(GI)_GC.Lock_y'])
    df_merged_full.loc[condition_lock, 'S/In FCST(GI)_GC.Lock_x'] = df_merged_full.loc[condition_lock, 'S/In FCST(GI)_GC.Lock_y']
    condition_color = (~df_merged_full['S/In FCST Color Condition_y'].isna()) & (df_merged_full['S/In FCST Color Condition_x'] != df_merged_full['S/In FCST Color Condition_y'])
    df_merged_full.loc[condition_color, 'S/In FCST Color Condition_x'] = df_merged_full.loc[condition_lock, 'S/In FCST Color Condition_y']

    fn_use_x_after_join(df_merged_full)

    # return merged_tat_09, df_merged_full
    return df_merged_full

@_decoration_
def fn_step11_apply_item_tatset_lock_join(df_10_apply_item_tat_lock_all: pd.DataFrame, df_in_Item_TAT: pd.DataFrame, df_in_Item_Master: pd.DataFrame) -> pd.DataFrame:
    """
    Step 11: Similar to Step 10 but uses column 'ITEMTAT TATTERM_SET'.
    """

    df_10 = df_10_apply_item_tat_lock_all.copy(deep=True)
    

    # Merge df_in_Item_TAT with df_in_Item_Master to get 'Item.[Item GBM]' and filter by 'Item.[Item GBM]' == 'VD'
    df_tat_item = pd.merge(df_in_Item_TAT, df_in_Item_Master[['Item.[Item]', 'Item.[Item GBM]']], on='Item.[Item]', how='inner')
    df_tat_item = df_tat_item[df_tat_item['Item.[Item GBM]'] == 'VD']

    df_tat_item['CURWEEK_MINUS_TATTERMSET'] = df_tat_item['ITEMTAT TATTERM_SET'].apply(lambda x: common.gfn_add_week(current_week_normalized, int(x) -1))
    columns_to_int = ['ITEMTAT TATTERM','ITEMTAT TATTERM_SET' ,'CURWEEK_MINUS_TATTERMSET']
    df_tat_item[columns_to_int] = df_tat_item[columns_to_int].astype('int32')
    
    # # convert string to merge
    # columns_to_str_10 = [
    #     'Sales Domain.[Sales Domain LV2]',
    #     'Sales Domain.[Sales Domain LV3]',
    #     'Sales Domain.[Sales Domain LV4]',
    #     'Sales Domain.[Sales Domain LV5]',
    #     'Sales Domain.[Sales Domain LV6]',
    #     'Sales Domain.[Sales Domain LV7]',
    #     'Sales Domain.[Ship To]',
    #     # 'Item.[Item]',
    #     # 'Item.[Item GBM]',
    #     # 'Item.[Product Group]',
    #     # 'Location.[Location]',
    #     'SalesProductASN_ISVALID',
    #     'Time.[Partial Week]',
    #     # 'S/In FCST(GI)_GC.Lock',
    #     # 'S/In FCST Color Condition',
    #     'RTS_WEEK_NORMALIZED',
    #     'current_row_partial_week_normalized',
    #     'EOS_WEEK_NORMALIZED'
    # ]
    # for col in columns_to_str_10:
    #     df_10[col] = df_10[col].astype(str)

    columns_to_str_tat = [
        'Location.[Location]',
        'Item.[Item]'
    ]
    for col in columns_to_str_tat:
        df_tat_item[col] = df_tat_item[col].astype(str)


    
    # Merge df_09 with df_tat_item on specified columns
    #   - 'Item.[Item]', 'Item.[Item GBM]', 'Item.[Product Group]', 'Location.[Location]'
    #   - 'inner' join, meaning intersection of two dataframes
    merged_tat_10 = pd.merge(
        df_10,
        df_tat_item,
        left_on=[
            'Item.[Item]',
            'Item.[Item GBM]',
            # 'Item.[Product Group]',
            'Location.[Location]'
        ],
        right_on=[
            'Item.[Item]',
            'Item.[Item GBM]',
            # 'Item.[Product Group]',
            'Location.[Location]'
        ],
        how='inner'  # Choose 'left', 'right', 'inner', or 'outer' depending on your needs
    )
    # Rename columns with '_x' suffix to their original names
    fn_use_x_after_join(merged_tat_10)
    # fn_log_dataframe(merged_df_lv_6,'df_09_merged_df_lv_6')


    condition_tat_color = (
        (merged_tat_10['current_row_partial_week_normalized'] >= int(current_week_normalized)) &
        (merged_tat_10['current_row_partial_week_normalized'] <= merged_tat_10['CURWEEK_MINUS_TATTERMSET']) &
        (merged_tat_10['S/In FCST Color Condition'] == 'DGREY_RED')
    )
    condition_tat_lock = (
        (merged_tat_10['current_row_partial_week_normalized'] >= int(current_week_normalized)) &
        (merged_tat_10['current_row_partial_week_normalized'] <= merged_tat_10['CURWEEK_MINUS_TATTERMSET']) 
    )
    merged_tat_10.loc[condition_tat_color,'S/In FCST Color Condition'] = 'DGREY_REDB'
    merged_tat_10.loc[condition_tat_lock,'S/In FCST(GI)_GC.Lock'] = True
    

    df_merged_full  = pd.merge(
        df_10,
        merged_tat_10,
        left_on=[
            'Sales Domain.[Sales Domain LV2]', 
            'Sales Domain.[Sales Domain LV3]', 
            'Sales Domain.[Sales Domain LV4]', 
            'Sales Domain.[Sales Domain LV5]', 
            'Sales Domain.[Sales Domain LV6]', 
            'Sales Domain.[Sales Domain LV7]',
            'Sales Domain.[Ship To]',
            'Location.[Location]', 
            'Item.[Item]',
            'Item.[Item GBM]',
            'Item.[Product Group]',
            'Location.[Location]',
            'Time.[Partial Week]'

        ],
        right_on=[
            'Sales Domain.[Sales Domain LV2]', 
            'Sales Domain.[Sales Domain LV3]', 
            'Sales Domain.[Sales Domain LV4]', 
            'Sales Domain.[Sales Domain LV5]', 
            'Sales Domain.[Sales Domain LV6]',
            'Sales Domain.[Sales Domain LV7]',
            'Sales Domain.[Ship To]',
            'Location.[Location]', 
            'Item.[Item]',
            'Item.[Item GBM]',
            'Item.[Product Group]',
            'Location.[Location]',
            'Time.[Partial Week]'
        ],
        how='left'  # Choose 'left', 'right', 'inner', or 'outer' depending on your needs
    )

    # Update 'S/In FCST(GI)_GC.Lock_x' where 'S/In FCST(GI)_GC.Lock_y' is not NaN and values are different
    condition_lock = (~df_merged_full['S/In FCST(GI)_GC.Lock_y'].isna()) & (df_merged_full['S/In FCST(GI)_GC.Lock_x'] != df_merged_full['S/In FCST(GI)_GC.Lock_y'])
    df_merged_full.loc[condition_lock, 'S/In FCST(GI)_GC.Lock_x'] = df_merged_full.loc[condition_lock, 'S/In FCST(GI)_GC.Lock_y']
    condition_color = (~df_merged_full['S/In FCST Color Condition_y'].isna()) & (df_merged_full['S/In FCST Color Condition_x'] != df_merged_full['S/In FCST Color Condition_y'])
    df_merged_full.loc[condition_color, 'S/In FCST Color Condition_x'] = df_merged_full.loc[condition_lock, 'S/In FCST Color Condition_y']

    fn_use_x_after_join(df_merged_full)

    # return merged_tat_10, df_merged_full
    return df_merged_full


@_decoration_
def fn_step12_del_gbm_add_ap2_ap1(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Step 12: Drop 'Item.[Item GBM]' and add 'S/In FCST(GI)_AP2.Lock', 'S/In FCST(GI)_AP1.Lock' columns.
    These new columns will have the same value as 'S/In FCST(GI)_GC.Lock'.
    Reorder columns as specified.
    """
    df_return = df_input.copy(deep=True)
    # Drop the 'Item.[Item GBM]' column
    df_return = df_return.drop(columns=['Item.[Item GBM]'])
    
    # Add new columns with the same value as 'S/In FCST(GI)_GC.Lock'
    df_return['S/In FCST(GI)_AP2.Lock'] = df_return['S/In FCST(GI)_GC.Lock']
    df_return['S/In FCST(GI)_AP1.Lock'] = df_return['S/In FCST(GI)_GC.Lock']
    
    # Reorder columns as specified by the user
    column_order = [
        'Sales Domain.[Sales Domain LV2]', 'Sales Domain.[Sales Domain LV3]', 'Sales Domain.[Sales Domain LV4]',
        'Sales Domain.[Sales Domain LV5]', 'Sales Domain.[Sales Domain LV6]', 'Sales Domain.[Sales Domain LV7]',
        'Sales Domain.[Ship To]', 'Item.[Product Group]', 'Item.[Item]', 'Location.[Location]',
        'Time.[Partial Week]', 'S/In FCST(GI)_GC.Lock', 'S/In FCST(GI)_AP2.Lock',
        'S/In FCST(GI)_AP1.Lock', 'S/In FCST Color Condition'
    ]
    df_return = df_return[column_order]
    
    return df_return


@_decoration_
def fn_step13_set_locks_based_on_forecast_rule(df_12_del_gbm_add_ap2_ap1: pd.DataFrame, df_in_Forecast_Rule: pd.DataFrame) -> pd.DataFrame:
    """
    Step 13: Set locks based on the master data from df_in_Forecast_Rule.
    """
    
    # df_result = df_12_del_gbm_add_ap2_ap1.copy(deep=True)
    # df_result.reset_index(drop=True, inplace=True)
    df_result = df_12_del_gbm_add_ap2_ap1

    # convert 
    df_result['Sales Domain.[Ship To]'] = df_result['Sales Domain.[Ship To]'].astype(str)
    df_result['Sales Domain.[Sales Domain LV2]'] = df_result['Sales Domain.[Sales Domain LV2]'].astype(str)
    df_result['Sales Domain.[Sales Domain LV3]'] = df_result['Sales Domain.[Sales Domain LV3]'].astype(str)
    
    # Convert 'Sales Domain.[Ship To]' columns to string for compatibility
    df_in_Forecast_Rule['Sales Domain.[Ship To]'] = df_in_Forecast_Rule['Sales Domain.[Ship To]'].astype(str)
    df_07_join_sales_product_asn_to_lvl['Sales Domain.[Ship To]'] = df_07_join_sales_product_asn_to_lvl['Sales Domain.[Ship To]'].astype(str)

    # Filter df_in_Forecast_Rule by 'Sales Domain.[Ship To]' and 'Item.[Product Group]'
    filtered_forecast_rule = df_in_Forecast_Rule[df_in_Forecast_Rule['Sales Domain.[Ship To]'].isin(df_07_join_sales_product_asn_to_lvl['Sales Domain.[Ship To]']) &
                                        df_in_Forecast_Rule['Item.[Product Group]'].isin(df_07_join_sales_product_asn_to_lvl['Item.[Product Group]'])]

    def apply_lock_for_2(forecast_rule, lock_column):
        if pd.notna(forecast_rule):
            forecast_rule = int(forecast_rule)
            if forecast_rule == 2:
                # Lock based on LV2 and Product Group, excluding '2xx'
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                              (df_result['Item.[Product Group]'] == product_group) &
                              (~df_result['Sales Domain.[Ship To]'].str.startswith('2')),
                              lock_column] = True
                # df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                #               (df_result['Item.[Product Group]'] == product_group) &
                #               (df_result['Sales Domain.[Ship To]'].str.startswith('2')),
                #               lock_column] = True
            elif forecast_rule == 3:
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                              (df_result['Item.[Product Group]'] == product_group) &
                              (~df_result['Sales Domain.[Ship To]'].str.startswith('3')),
                              lock_column] = True
            elif forecast_rule == 4:
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                              (df_result['Item.[Product Group]'] == product_group) &
                              (~df_result['Sales Domain.[Ship To]'].str.startswith('A3')),
                              lock_column] = True
            elif forecast_rule == 5:
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                              (df_result['Item.[Product Group]'] == product_group) &
                              (~df_result['Sales Domain.[Ship To]'].str.startswith('4')),
                              lock_column] = True
            elif forecast_rule == 6:
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                              (df_result['Item.[Product Group]'] == product_group) &
                              (~df_result['Sales Domain.[Ship To]'].str.startswith('5')),
                              lock_column] = True
            elif forecast_rule == 7:
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                              (df_result['Item.[Product Group]'] == product_group) &
                              (~df_result['Sales Domain.[Ship To]'].str.startswith('A5')),
                              lock_column] = True
        else:
            df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                          (df_result['Sales Domain.[Ship To]'] == lv2) &
                          (df_result['Item.[Product Group]'] == product_group),
                          lock_column] = True
            df_result.loc[(df_result['Sales Domain.[Sales Domain LV3]'] == ship_to) &
                          (df_result['Item.[Product Group]'] == product_group),
                          lock_column] = True

    def apply_lock_for_3(forecast_rule, lock_column):
        if pd.notna(forecast_rule):
            forecast_rule = int(forecast_rule)
            if forecast_rule == 2:
                # Lock based on LV3 and Product Group
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV3]'] == ship_to) &
                              (df_result['Item.[Product Group]'] == product_group),
                              lock_column] = True
            elif forecast_rule == 3:
                # Lock based on LV2 or LV3, excluding certain conditions
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                              (df_result['Sales Domain.[Ship To]'] == lv2) &
                              (df_result['Item.[Product Group]'] == product_group),
                              lock_column] = True
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV3]'] == ship_to) &
                              (df_result['Item.[Product Group]'] == product_group) &
                              (~df_result['Sales Domain.[Ship To]'].str.startswith('3')),
                              lock_column] = True
            elif forecast_rule == 4:
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                              (df_result['Sales Domain.[Ship To]'] == lv2) &
                              (df_result['Item.[Product Group]'] == product_group),
                              lock_column] = True
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV3]'] == ship_to) &
                              (df_result['Item.[Product Group]'] == product_group) &
                              (~df_result['Sales Domain.[Ship To]'].str.startswith('A3')),
                              lock_column] = True
            elif forecast_rule == 5:
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                              (df_result['Sales Domain.[Ship To]'] == lv2) &
                              (df_result['Item.[Product Group]'] == product_group),
                              lock_column] = True
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV3]'] == ship_to) &
                              (df_result['Item.[Product Group]'] == product_group) &
                              (~df_result['Sales Domain.[Ship To]'].str.startswith('4')),
                              lock_column] = True
            elif forecast_rule == 6:
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                              (df_result['Sales Domain.[Ship To]'] == lv2) &
                              (df_result['Item.[Product Group]'] == product_group),
                              lock_column] = True
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV3]'] == ship_to) &
                              (df_result['Item.[Product Group]'] == product_group) &
                              (~df_result['Sales Domain.[Ship To]'].str.startswith('5')),
                              lock_column] = True
            elif forecast_rule == 7:
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                              (df_result['Sales Domain.[Ship To]'] == lv2) &
                              (df_result['Item.[Product Group]'] == product_group),
                              lock_column] = True
                df_result.loc[(df_result['Sales Domain.[Sales Domain LV3]'] == ship_to) &
                              (df_result['Item.[Product Group]'] == product_group) &
                              (~df_result['Sales Domain.[Ship To]'].str.startswith('A5')),
                              lock_column] = True
        else:
            df_result.loc[(df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        #   (df_result['Sales Domain.[Ship To]'] == lv2) &
                          (df_result['Item.[Product Group]'] == product_group),
                          lock_column] = True
            df_result.loc[(df_result['Sales Domain.[Sales Domain LV3]'] == ship_to) &
                          (df_result['Item.[Product Group]'] == product_group),
                          lock_column] = True

    for _, rule_row in filtered_forecast_rule.iterrows():
        ship_to = str(rule_row['Sales Domain.[Ship To]'])
        product_group = rule_row['Item.[Product Group]']
        gc_fcst = rule_row['FORECAST_RULE GC FCST']
        ap2_fcst = rule_row['FORECAST_RULE AP2 FCST']
        ap1_fcst = rule_row['FORECAST_RULE AP1 FCST']
        
        lv2_data = df_07_join_sales_product_asn_to_lvl.loc[
            (df_07_join_sales_product_asn_to_lvl['Sales Domain.[Ship To]'] == ship_to) &
            (df_07_join_sales_product_asn_to_lvl['Item.[Product Group]'] == product_group),
            'Sales Domain.[Sales Domain LV2]'
        ]
        if lv2_data.empty:
            continue
        lv2 = str(lv2_data.iloc[0])
        
        # Apply lock logic for each lock column
        if ship_to.startswith('3'):
            apply_lock_for_3(gc_fcst, 'S/In FCST(GI)_GC.Lock')
            apply_lock_for_3(ap2_fcst, 'S/In FCST(GI)_AP2.Lock')
            apply_lock_for_3(ap1_fcst, 'S/In FCST(GI)_AP1.Lock')
        elif ship_to.startswith('2'):
            apply_lock_for_2(gc_fcst, 'S/In FCST(GI)_GC.Lock')
            apply_lock_for_2(ap2_fcst, 'S/In FCST(GI)_AP2.Lock')
            apply_lock_for_2(ap1_fcst, 'S/In FCST(GI)_AP1.Lock')

    return df_result

@_decoration_
def fn_step13_set_locks_based_on_forecast_rule_vector(df_12_del_gbm_add_ap2_ap1: pd.DataFrame, df_in_Forecast_Rule: pd.DataFrame) -> pd.DataFrame:
    """
    Step 13: Set locks based on the master data from df_in_Forecast_Rule.
    """
    # df_result = df_12_del_gbm_add_ap2_ap1.copy(deep=True)
    df_result = df_12_del_gbm_add_ap2_ap1

    # Convert columns to string for compatibility
    df_result['Sales Domain.[Ship To]'] = df_result['Sales Domain.[Ship To]'].astype(str)
    df_result['Sales Domain.[Sales Domain LV2]'] = df_result['Sales Domain.[Sales Domain LV2]'].astype(str)
    df_result['Sales Domain.[Sales Domain LV3]'] = df_result['Sales Domain.[Sales Domain LV3]'].astype(str)
    
    df_in_Forecast_Rule['Sales Domain.[Ship To]'] = df_in_Forecast_Rule['Sales Domain.[Ship To]'].astype(str)
    
    # Filter df_in_Forecast_Rule by 'Sales Domain.[Ship To]' and 'Item.[Product Group]'
    filtered_forecast_rule = df_in_Forecast_Rule[
        df_in_Forecast_Rule['Sales Domain.[Ship To]'].isin(df_result['Sales Domain.[Ship To]']) &
        df_in_Forecast_Rule['Item.[Product Group]'].isin(df_result['Item.[Product Group]'])
    ]

    # Vectorized lock application
    for _, row in filtered_forecast_rule.iterrows():
        lv2 = row['Sales Domain.[Sales Domain LV2]']
        product_group = row['Item.[Product Group]']
        gc_fcst = row['FORECAST_RULE GC FCST']
        ap2_fcst = row['FORECAST_RULE AP2 FCST']
        ap1_fcst = row['FORECAST_RULE AP1 FCST']
        
        lv2_data = df_07_join_sales_product_asn_to_lvl.loc[
            (df_07_join_sales_product_asn_to_lvl['Sales Domain.[Ship To]'] == row['Sales Domain.[Ship To]']) &
            (df_07_join_sales_product_asn_to_lvl['Item.[Product Group]'] == product_group),
            'Sales Domain.[Sales Domain LV2]'
        ]
        if lv2_data.empty:
            continue
        lv2 = str(lv2_data.iloc[0])
        
        # Apply lock logic for each lock column
        if row['Sales Domain.[Ship To]'].startswith('3'):
            if pd.notna(gc_fcst):
                gc_fcst = int(gc_fcst)
                if gc_fcst == 2:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('2'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_GC.Lock'] = True
                elif gc_fcst == 3:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('3'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_GC.Lock'] = True
                # Add more conditions as needed
            if pd.notna(ap2_fcst):
                ap2_fcst = int(ap2_fcst)
                if ap2_fcst == 2:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('2'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_AP2.Lock'] = True
                elif ap2_fcst == 3:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('3'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_AP2.Lock'] = True
                # Add more conditions as needed
            if pd.notna(ap1_fcst):
                ap1_fcst = int(ap1_fcst)
                if ap1_fcst == 2:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('2'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_AP1.Lock'] = True
                elif ap1_fcst == 3:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('3'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_AP1.Lock'] = True
                # Add more conditions as needed
        elif row['Sales Domain.[Ship To]'].startswith('2'):
            if pd.notna(gc_fcst):
                gc_fcst = int(gc_fcst)
                if gc_fcst == 2:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('2'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_GC.Lock'] = True
                elif gc_fcst == 3:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('3'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_GC.Lock'] = True
                # Add more conditions as needed
            if pd.notna(ap2_fcst):
                ap2_fcst = int(ap2_fcst)
                if ap2_fcst == 2:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('2'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_AP2.Lock'] = True
                elif ap2_fcst == 3:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('3'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_AP2.Lock'] = True
                # Add more conditions as needed
            if pd.notna(ap1_fcst):
                ap1_fcst = int(ap1_fcst)
                if ap1_fcst == 2:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('2'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_AP1.Lock'] = True
                elif ap1_fcst == 3:
                    mask = (
                        (df_result['Sales Domain.[Sales Domain LV2]'] == lv2) &
                        (df_result['Item.[Product Group]'] == product_group) &
                        (~df_result['Sales Domain.[Ship To]'].str.startswith('3'))
                    )
                    df_result.loc[mask, 'S/In FCST(GI)_AP1.Lock'] = True
                # Add more conditions as needed

    
    return df_result

@_decoration_
def fn_step14_add_version(df_13: pd.DataFrame) -> pd.DataFrame:
    """
    Step 14: Add a new column 'Version.[Version Name]' with value 'CurrentWorkingView' to df_13_set_locks_based_on_forecast_rule, 
             and reorder columns as specified.
    """
    df_result = df_13.copy(deep=True)
    df_result['Version.[Version Name]'] = 'CurrentWorkingView'
    
    # Reorder columns
    columns_order = ['Version.[Version Name]', 'Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]',
                 'S/In FCST(GI)_GC.Lock', 'S/In FCST(GI)_AP2.Lock', 'S/In FCST(GI)_AP1.Lock', 'S/In FCST Color Condition']
    df_result = df_result[columns_order]
    
    return df_result



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

    # 입력 파라미터(str_p_out_version)가 비어 있는 경우 경고 메시지 출력 후 빈 데이터 프레임 리턴
    if str_p_out_version is None or str_p_out_version.strip() == '':
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(str_p_out_version)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    df_return = df_p_source.copy(deep=True)
    df_return['Version.[Version Name]'] = str_p_out_version

    df_return = df_return[
        ['Version.[Version Name]', 'Item.[Item]', 'Location.[Location]', 'Time.[Week]', 'W Quantity Max Target']]

    return df_return




if __name__ == '__main__':
    args = parse_args()
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    # Output 테이블 선언
    out_Demand = pd.DataFrame()
    try:
        ################################################################################################################
        # 전처리 : 모듈 내에서 사용될 데이터에 대한 정합성 체크 및 데이터 선 가공
        ################################################################################################################
        Param_OUT_VERSION = args.get('Param_OUT_VERSION')
        Param_Exception_Flag = args.get('Param_Exception_Flag')
        input_dataframes = {}
        if is_local:
            set_input_output_folder(is_local, args)
            fn_process_in_df_mst()

        #  Set max_week
        initialize_max_week(is_local, args)
        # 입력 변수 확인
        if Param_OUT_VERSION is None or Param_OUT_VERSION.strip() == '':
            Param_OUT_VERSION = 'CurrentWorkingView'
        # 입력 변수 확인용 로그
        logger.Note(p_note=f'Param_OUT_VERSION : {Param_OUT_VERSION}', p_log_level=LOG_LEVEL.debug())

        # 입력 변수 확인
        if Param_Exception_Flag is None or Param_Exception_Flag.strip() == '':
            flag_exception = True
        elif Param_Exception_Flag == 'S':
            flag_exception = False
        # 입력 변수 확인용 로그
        logger.Note(p_note=f'Param_Exception_Flag : {Param_Exception_Flag}', p_log_level=LOG_LEVEL.debug())

        # 입력 변수 중 데이터가 없는 경우 경고 메시지를 출력한다.
        for in_Demand in input_dataframes:
            fn_check_input_table(input_dataframes[in_Demand], in_Demand, '0')


        output_dataframes = {}
        ################################################################################################################
        # Step 1. df_in_MST_RTS 와 df_in_MST_EOS 를 Item 과 ShipTo를 기준으로 inner Join
        ################################################################################################################
        dict_log = {
            'p_step_no': 10,
            'p_step_desc': 'Step 1  : df_in_MST_RTS 와 df_in_MST_EOS 를 Item 과 ShipTo를 기준으로 inner Join',
            'p_df_name' : 'df_01_joined_rts_eos'
        }
        df_01_joined_rts_eos = fn_step01_join_rts_eos(**dict_log)
        # fn_check_input_table(df_01_joined_rts_eos, 'df_01_joined_rts_eos', '0')
        output_dataframes["df_01_joined_rts_eos"] = df_01_joined_rts_eos
        fn_log_dataframe(df_01_joined_rts_eos, 'df_01_joined_rts_eos')

        ################################################################################################################
        # Step 2  : Step1의 Result에 Time을 Partial Week 으로 변환
        ################################################################################################################
        dict_log = {
            'p_step_no': 20,
            'p_step_desc': 'Step 2  : Step1의 Result에 Time을 Partial Week 으로 변환 ',
            'p_df_name' : 'df_02_date_to_partial_week'
        }
        df_02_date_to_partial_week = fn_step02_convert_date_to_partial_week(**dict_log)
        # print for test  
        output_dataframes["df_02_date_to_partial_week"] = df_02_date_to_partial_week        
        fn_log_dataframe(df_02_date_to_partial_week, 'df_02_date_to_partial_week')
        ################################################################################################################
        # Step 3:  df_in_MST_RTS 와 df_in_MST_EOS의 ITEM * ShipTo로 Inner Join하여 새로운 DF 생성 ( Output Data )
        ################################################################################################################
        dict_log = {
            'p_step_no': 30,
            'p_step_desc': 'Step 3  : df_in_MST_RTS 와 df_in_MST_EOS의 ITEM * ShipTo로 Inner Join하여 새로운 DF 생성 ( Output Data ) ' ,
            'p_df_name' : 'df_03_joined_rts_eos'
        }
        df_03_joined_rts_eos = fn_step03_join_rts_eos(**dict_log)
        # fn_check_input_table(df_03_joined_rts_eos, 'df_03_joined_rts_eos', '0')
        output_dataframes["df_03_joined_rts_eos"] = df_03_joined_rts_eos
        fn_log_dataframe(df_03_joined_rts_eos, 'df_03_joined_rts_eos')

        ################################################################################################################
        # Step 4  : Step3의 df에 Partial Week 및 Measure Column 추가
        ################################################################################################################
        dict_log = {
            'p_step_no': 40,
            'p_step_desc': 'Step 4  : Step3의 df에 Partial Week 및 Measure Column 추가 ' ,
            'p_df_name' : 'df_04_partialweek_measurecolumn'
        }
        df_04_partialweek_measurecolumn = fn_step04_add_partialweek_measurecolumn(**dict_log)
        # fn_check_input_table(df_04_partialweek_measurecolumn, 'df_04_partialweek_measurecolumn', '0')
        # print for test  
        output_dataframes["df_04_partialweek_measurecolumn"] = df_04_partialweek_measurecolumn
        fn_log_dataframe(df_04_partialweek_measurecolumn, 'df_04_partialweek_measurecolumn')

        ################################################################################################################
        # Step 5:  Step4의 df에 당주주차부터 RTS 와 EOS 반영 및 Color 표시
        ################################################################################################################
        dict_log = {
            'p_step_no': 50,
            'p_step_desc': 'Step 5  : Step4의 df에 당주주차부터 RTS 와 EOS 반영 및 Color 표시 ' ,
            'p_df_name' : 'df_05_set_lock_values'
        }
        df_05_set_lock_values = fn_step05_set_lock_values(**dict_log)
        # fn_check_input_table(df_05_set_lock_values, 'df_05_set_lock_values', '0')
        # print for test  
        output_dataframes["df_05_set_lock_values"] = df_05_set_lock_values
        fn_log_dataframe(df_05_set_lock_values, 'df_05_set_lock_values')

        ################################################################################################################
        # Step 6:  Step5의 df에 Item Master 정보 추가( Item Type, Item GBM, Item Product Group) 및 Color 조건 업데이트(무선 BAS 제품 8주 구간 GREEN UPDATE)
        ################################################################################################################
        dict_log = {
            'p_step_no': 60,
            'p_step_desc': 'Step 6  : Step5의 df에 Item Master 정보 추가 및 Color 조건 업데이트 ' ,
            'p_df_name' : 'df_06_addcolumn_green_for_wireless_bas'
        }
        df_06_addcolumn_green_for_wireless_bas = fn_step06_addcolumn_green_for_wireless_bas_array_based(**dict_log)
        output_dataframes["df_06_addcolumn_green_for_wireless_bas"] = df_06_addcolumn_green_for_wireless_bas
        fn_log_dataframe(df_06_addcolumn_green_for_wireless_bas, 'df_06_addcolumn_green_for_wireless_bas')

        ################################################################################################################
        # Step 7:  Step6의 df에 Sales Product ASN 정보 추가
        # lvl 을 구성한다.
        ################################################################################################################
        dict_log = {
            'p_step_no': 70,
            'p_step_desc': 'Step 7  : Step6의 df에 Sales Product ASN 정보 추가 ' ,
            'p_df_name' : 'df_07_join_sales_product_asn_to_lvl'
        }
        df_07_join_sales_product_asn_to_lvl = fn_step07_join_sales_product_asn_to_lvl_by_merge(**dict_log)
        # fn_check_input_table(df_07_join_sales_product_asn_to_lvl, 'df_07_join_sales_product_asn_to_lvl', '0')
        # print for test  
        output_dataframes["df_07_join_sales_product_asn_to_lvl"] = df_07_join_sales_product_asn_to_lvl  
        fn_log_dataframe(df_07_join_sales_product_asn_to_lvl, 'df_07_join_sales_product_asn_to_lvl')

        ################################################################################################################
        # Step 8:  Step7의 df에 조건 정보 추가. lvl 정보에 주차정보를 추가한다.
        # df_08_join_dataframes = df_07_join_sales_product_asn.join(df_06_addcolumn_green_for_wireless_bas)
        ################################################################################################################
        dict_log = {
            'p_step_no': 80,
            'p_step_desc': 'Step 8  : Step7의 df에 조건 정보 추가. lvl 정보에 주차정보를 추가한다. chunk 사용. step13 까지 처리' 
        }
        df_08_add_weeks_to_dimention = fn_step08_add_weeks_to_dimention_vector_join_chunk('df_07_join_sales_product_asn_to_lvl',**dict_log)

        ################################################################################################################
        # Step 9:  Step8의 df에 Item CLASS 정보 필터링
        ################################################################################################################
        dict_log_90 = {
            'p_step_no': 90,
            'p_step_desc': 'Step 9  : Step8의 df에 Item CLASS 정보 필터링 ' 
        }


        
        ################################################################################################################
        # Step 10:  Step9의 df에 Item TAT 정보 필터링 및 Lock
        ################################################################################################################
        dict_log_100 = {
            'p_step_no': 100,
            'p_step_desc': 'Step 10  : Step9의 df에 Item TAT 정보 필터링 및 Lock ' 
        }
        
        ################################################################################################################
        # Step 11:  Step10의 df에 Item TAT 정보 필터링 및 Lock
        ################################################################################################################
        dict_log_110 = {
            'p_step_no': 110,
            'p_step_desc': 'Step 11  : Step10의 df에 Item TAT 정보 필터링 및 Lock.  DARKGRAY ' 
        }
        # ################################################################################################################
        # Step 12: Modify DataFrame by dropping and adding columns
        # Delete :  Item.[Item GBM]  
        # Add : S/In FCST(GI)_AP2.Lock,S/In FCST(GI)_AP1.Lock
        ################################################################################################################
        dict_log_120 = {
            'p_step_no': 120,
            'p_step_desc': 'Step 12  :  Item.[Item GBM] 삭제, Lock Measure Column 추가' 
        }
        ################################################################################################################
        # Step 13: Set locks based on the master data from df_in_Forecast_Rule
        ################################################################################################################
        dict_log = {
            'p_step_no': 130,
            'p_step_desc': 'Step 13  : Set locks based on the master data from df_in_Forecast_Rule' 
        }

        ################################################################################################################
        # Step 14: Add a new column 'Version.[Version Name]' with value 'CurrentWorkingView' to df_13_set_locks_based_on_forecast_rule, 
        #          and reorder columns as specified.
        # ################################################################################################################
        dict_log = {
            'p_step_no': 140,
            'p_step_desc': 'Step 14  : Add a new column Version.[Version Name] with value CurrentWorkingView to df_13_set_locks_based_on_forecast_rule, and reorder columns as specified.' 
        }

        if chunk_size_step08 is not None :
            file_index = 0
            for df in dfs_step13:
                df_14_add_version = fn_step14_add_version(df['value'], **dict_log)
                dfs_step14.append({"key": f"df_14_{str(file_index).zfill(3)}", "value": df_14_add_version})
                file_index = file_index + 1

            del dfs_step08, dfs_step09, dfs_step10, dfs_step11, dfs_step12, dfs_step13, output_dataframes
            out_Demand = pd.concat([df['value'] for df in dfs_step14], ignore_index=True)
        else:
            out_Demand = output_dataframes["df_13"]
        # output_dataframes["out_Demand"] = out_Demand 
        # fn_log_dataframe(out_Demand, 'out_Demand')
        fn_check_input_table(out_Demand, 'out_Demand', '0')
        
        # save to csv file
        csv_file_path = os.path.join(os.getcwd(),str_output_dir, 'out_Demand.csv')
        # out_Demand.to_csv(csv_file_path, index=False)
        
        # # save to zip file by splited 30Mbytes
        # zip_file_path = os.path.join(os.getcwd(),str_output_dir, 'out_Demand.zip')
        # with zipfile.ZipFile(zip_file_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        #     file_size = os.path.getsize(csv_file_path)
        #     chunk_size = 30 * 1024 * 1024
        #     if file_size > chunk_size:
        #         chunk_index = 0
        #         with open(csv_file_path, 'rb') as f:
        #             while True:
        #                 chunk = f.read(chunk_size)
        #                 if not chunk:
        #                     break
        #                 zip_file.writestr(f'out_Demand_{str(chunk_index).zfill(3)}.csv', chunk)
        #                 chunk_index = chunk_index + 1
        #     else:
        #         zip_file.write(csv_file_path, arcdata=out_Demand.to_csv(index=False))

        ################################################################################################################
        # 최종 Output 정리
        ################################################################################################################
        # dict_log = {
        #     'p_step_no': 900,
        #     'p_step_desc': '최종 Output 정리 - out_Demand',
        #     'p_df_name': 'out_Demand'
        # }
        # out_Demand = fn_output_formatter(df_demand_aggr, Param_OUT_VERSION, **dict_log)
        
        # for df in output_dataframes:
        #     # fn_log_dataframe(output_dataframes[df], df)
        #     output_dataframes[df].to_csv(str_output_dir + "/"+df+".csv", encoding="UTF8", index=False)

        

    except Exception as e:
        trace_msg = traceback.format_exc()
        logger.Note(p_note=trace_msg, p_log_level=LOG_LEVEL.debug())
        logger.Error()
        if flag_exception:
            raise Exception(e)
        else:
            logger.info(f'{str_instance} exit - {time.strftime("%Y-%m-%d - %H:%M:%S")}')

    else:
        logger.Finish()

    finally:
        # MediumWeight 실행 시 Header 없는 빈 데이터프레임이 Output이 되는 경우 오류가 발생함.
        # 이 오류를 방지하기 위해 Output이 빈 경우을 체크하여 Header를 만들어 줌.
        if out_Demand.empty:
            out_Demand = fn_set_header()
            fn_log_dataframe(out_Demand, 'out_Demand')

        log_file_name = common.G_PROGRAM_NAME.replace('py', 'log')
        log_file_name = f'log/{log_file_name}'
        shutil.copyfile(log_file_name, os.path.join(str_output_dir, os.path.basename(log_file_name)))


        logger.info(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')