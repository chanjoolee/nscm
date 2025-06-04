import os,sys
import time,datetime,shutil
import inspect
import traceback
import pandas as pd
from NSCMCommon import NSCMCommon as common
# from typing_extensions import Literal
import glob

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
# ── column constants ──────────────────────────────────────────────────────────────────────────────────────────
COL_VERSION         = 'Version.[Version Name]'
COL_ITEM            = 'Item.[Item]'
COL_SHIP_TO         = 'Sales Domain.[Ship To]'
COL_LOC             = 'Location.[Location]'
COL_VIRTUAL_BO_ID   = 'DP Virtual BO ID.[Virtual BO ID]'
COL_BO_ID           = 'DP BO ID.[BO ID]'

COL_TIME_PW             = 'Time.[Partial Week]'
COL_CURRENT_ROW_WEEK    = 'WEEK_NUM'
COL_BO_FCST             = 'BO FCST' 
COL_BO_FCST_LOCK        = 'BO FCST.Lock'          
COL_BO_TOTAL_BOD_LT     = 'BO Total BOD LT'  
COL_BO_TOTAL_BOD_LT_NOM     = 'BO Total BOD LT NORMALIZED'  
# ───────────────────────────────────────────────────────────────
# CONSTANT STRING VARIABLES FOR DATAFRAME NAMES
# ───────────────────────────────────────────────────────────────
# input
STR_DF_IN_BO_FCST       = 'df_in_BO_FCST'
STR_DF_IN_TOTAL_BOD_LT  = 'df_in_Total_BOD_LT'
STR_DF_IN_MAX_PW        = 'df_In_MAX_PartialWeek'
# middle
STR_DF_STEP01_ADDED_WEEK    = 'df_fn_step01_added_week'
STR_DF_STEP03_LOCK          = 'df_fn_step03_lock'
STR_DF_STEP04_CREATED       = 'df_out_step04_created'


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

    df_return = df_p_source.copy(deep=True)
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
        COL_BO_FCST_LOCK
    ]

    df_return = df_return[columns_to_return]

    return df_return

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
            COL_BO_FCST_LOCK    : []    

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
def fn_process_in_df_mst():
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
################################################################################################################
# End of Util Functions
################################################################################################################

################################################################################################################
# Start of Step Functions
################################################################################################################
@_decoration_
def step01_process_total_bod_lt():
    """
    Step 1: Process the df_in_Total_BOD_LT dataframe to add week information.
    Uses global variables for input dataframe and current partial week.
    :return: Processed dataframe with added week information.
    """
    
    # Copy the input dataframe
    df_return = input_dataframes[STR_DF_IN_TOTAL_BOD_LT].copy()
    
    # Drop the 'Version.[Version Name]' column
    df_return.drop(columns=[COL_VERSION], inplace=True)
    
    
    # Calculate the new 'BO Total BOD LT' value
    def extract_week(row):
        return common.gfn_add_week(current_week_normalized, row[COL_BO_TOTAL_BOD_LT] // 7) + 'A'
    
    df_return[COL_BO_TOTAL_BOD_LT] = df_return.apply(extract_week, axis=1)
    
    # Add suffix 'A' or 'B' if present in the original dataframe
    # df_return[COL_BO_TOTAL_BOD_LT] = df_return[COL_BO_TOTAL_BOD_LT].astype(str) + df_in_Total_BOD_LT[COL_BO_TOTAL_BOD_LT].str.extract(r'([AB])$', expand=False).fillna('')
    
    return df_return

@_decoration_
def step02_find_max_virtual_bo_id():
    """
    Step 2: Find the maximum 'DP Virtual BO ID' in df_in_BO_FCST.
    Uses global variable for input dataframe.
    :return: Maximum 'DP Virtual BO ID'.
    """
    df_in_bo_fcst = input_dataframes[STR_DF_IN_BO_FCST]

    # Extract 'DP Virtual BO ID' column and find the maximum value
    max_virtual_bo_id = df_in_bo_fcst[COL_VIRTUAL_BO_ID].max()
    # VBO 가 모두 '-' 인 경우 Default = 'VBO_00000000'
    if ( max_virtual_bo_id == '-') :
        max_virtual_bo_id = 'VBO_00000000'
    
    return max_virtual_bo_id

@_decoration_
def step03_apply_lock_conditions_back():
    """
    Step 3: Apply lock conditions to df_in_BO_FCST.
    Uses global variables for input dataframe and Virtual_BO_ID_MAX.
    :return: Dataframe with lock conditions applied.
    """
    df_in_bo_fcst = input_dataframes[STR_DF_IN_BO_FCST]
    df_step01 = output_dataframes[STR_DF_STEP01_ADDED_WEEK]
    
    # Copy the input dataframe
    df_return = df_in_bo_fcst.copy()
    
    # Join with df_out_step01_added_week
    df_return = df_return.merge(df_step01, 
                                left_on=[COL_ITEM, COL_LOC],
                                right_on=[COL_ITEM, COL_LOC],
                                # 일치하지 않는 조건이 있기 때문에 innner 로 한다.
                                how='inner', 
                                suffixes=('', '_added'))
    
    # Normalize COL_TIME_PW and COL_BO_TOTAL_BOD_LT
    # df_return[COL_CURRENT_ROW_WEEK] = df_return[COL_TIME_PW].apply(normalize_week)
    # df_return[COL_BO_TOTAL_BOD_LT_NOM] = df_return[COL_BO_TOTAL_BOD_LT].apply(normalize_week)
    df_return[COL_CURRENT_ROW_WEEK]     = df_return[COL_TIME_PW].str.replace(r'\D', '', regex=True)
    df_return[COL_BO_TOTAL_BOD_LT_NOM]  = df_return[COL_BO_TOTAL_BOD_LT].str.replace(r'\D', '', regex=True)

    # for test temp
    # df_return.to_csv(str_output_dir + "/df_return_000.csv", encoding="UTF8", index=True)

    # Apply lock conditions

    ############################################################################################
    # For rows where 'DP Virtual BO ID' == '-' and 'DP BO ID' == '-'
    ############################################################################################
    df_return.loc[(df_return[COL_VIRTUAL_BO_ID] == '-') & (df_return[COL_BO_ID] == '-'), COL_BO_FCST_LOCK] = \
        (df_return[COL_CURRENT_ROW_WEEK] >= current_week_normalized) & \
        (df_return[COL_CURRENT_ROW_WEEK] <= df_return[COL_BO_TOTAL_BOD_LT_NOM])

    df_return.loc[(df_return[COL_VIRTUAL_BO_ID] == '-') & (df_return[COL_BO_ID] == '-') & \
                  (df_return[COL_CURRENT_ROW_WEEK] == max_week_normalized), COL_BO_FCST_LOCK] = False
    # csv_name = "df_out_step03_01_lock_equal_equal"
    # fn_log_dataframe(csv_name, df_return)


    ############################################################################################
    # For rows where 'DP Virtual BO ID' != '-' and 'DP BO ID' == '-'
    ############################################################################################
    df_return.loc[(df_return[COL_VIRTUAL_BO_ID] != '-') & (df_return[COL_BO_ID] == '-') & \
                  (df_return[COL_CURRENT_ROW_WEEK] == max_week_normalized), COL_BO_FCST_LOCK] = True

    # csv_name = "df_out_step03_01_lock_not_equal"
    # fn_log_dataframe(csv_name, df_return)

    ############################################################################################
    # For rows where 'DP Virtual BO ID' != '-' and 'DP BO ID' != '-'
    ############################################################################################
    df_return.loc[(df_return[COL_VIRTUAL_BO_ID] != '-') & (df_return[COL_BO_ID] != '-') & \
                  (df_return[COL_CURRENT_ROW_WEEK] == df_return[COL_BO_TOTAL_BOD_LT_NOM]), COL_BO_FCST_LOCK] = False
    df_return.loc[(df_return[COL_VIRTUAL_BO_ID] != '-') & (df_return[COL_BO_ID] != '-') & \
                  (df_return[COL_CURRENT_ROW_WEEK] == max_week_normalized), COL_BO_FCST_LOCK] = True
    # csv_name = "df_out_step03_01_lock_not_not"
    # fn_log_dataframe(csv_name, df_return)

    # Group by 'Item.[Item]', 'Sales Domain.[Ship To]', 'Location.[Location]'
    # Extract unique combinations from df_return in the order they appear
    custom_order = df_return.drop_duplicates(subset=[
        COL_ITEM, 
        COL_SHIP_TO, 
        COL_LOC,
        COL_VIRTUAL_BO_ID,
        COL_BO_ID
    ])[
        [COL_ITEM, COL_SHIP_TO, COL_LOC, 
        COL_VIRTUAL_BO_ID, COL_BO_ID]
    ].apply(tuple, axis=1).tolist()

    grouped = df_return.groupby([
        COL_ITEM, 
        COL_SHIP_TO, 
        COL_LOC,
        COL_VIRTUAL_BO_ID,
        COL_BO_ID
    ])

    # Convert the groups to a list of tuples
    group_list = list(grouped)

    # Sort the groups based on the custom order
    sorted_groups = sorted(group_list, key=lambda x: custom_order.index(x[0]) if x[0] in custom_order else float('inf'))
    
    ############################################################################################
    # Iterate over each group
    # Add Data there is no data in 'Time.[Partial Week]_NORMALIZED' == MaxPartialWeek_normalized
    ############################################################################################
    added_index = 0
    for name, group in sorted_groups:
        # Check if there is no data for the current group
        if not group[(group[COL_CURRENT_ROW_WEEK] == max_week_normalized)].any().any():
            latest_row = group.sort_values(by=COL_TIME_PW, ascending=False).iloc[0]
            latest_index = latest_row.name
            
            fcst_lock = True
            if (latest_row[COL_VIRTUAL_BO_ID] == '-') & (latest_row[COL_BO_ID] == '-'):
                fcst_lock = False
            elif (latest_row[COL_VIRTUAL_BO_ID] != '-') & (latest_row[COL_BO_ID] == '-'):
                fcst_lock = True
            else:
                fcst_lock = True

            if gfn_is_partial_week(max_week_normalized):
                # Add two rows for partial week
                new_row_a = latest_row.copy()
                new_row_a[COL_TIME_PW] = max_week_normalized + 'A'
                new_row_a[COL_CURRENT_ROW_WEEK] = max_week_normalized
                new_row_a[COL_BO_FCST_LOCK] = fcst_lock
                df_return = pd.concat([df_return.iloc[:latest_index + 1 + added_index], pd.DataFrame([new_row_a]), df_return.iloc[latest_index + 1 + added_index:]]).reset_index(drop=True)
                added_index += 1

                new_row_b = latest_row.copy()
                new_row_b[COL_TIME_PW] = max_week_normalized + 'B'
                new_row_b[COL_CURRENT_ROW_WEEK] = max_week_normalized
                new_row_b[COL_BO_FCST_LOCK] = fcst_lock
                df_return = pd.concat([df_return.iloc[:latest_index + 1 + added_index], pd.DataFrame([new_row_b]), df_return.iloc[latest_index + 1 + added_index:]]).reset_index(drop=True)
                added_index += 1
            else:
                # Add one row for non-partial week
                new_row = latest_row.copy()
                new_row[COL_TIME_PW] = max_week_normalized + 'A'
                new_row[COL_CURRENT_ROW_WEEK] = max_week_normalized
                new_row[COL_BO_FCST_LOCK] = fcst_lock
                df_return = pd.concat([df_return.iloc[:latest_index + 1 + added_index], pd.DataFrame([new_row]), df_return.iloc[latest_index + 1 + added_index:]]).reset_index(drop=True)
                added_index += 1
            
            # # Convert the tuple to a string with underscores
            # file_name = "_".join(map(str, name)).replace("/", "_").replace("-", "_")

            # # Use the file name in the path
            # fn_log_dataframe(f"df_out_step03_lock_{file_name}", df_return)
    return df_return

# --------------------------------------------------------------------------------------------
# Step 3 : Apply BO-Lock rules + create missing rows for max_week_normalized (vectorized)
# --------------------------------------------------------------------------------------------
@_decoration_
def step03_apply_lock_conditions() -> pd.DataFrame:
    """
    Step 3 – ① 기본 Lock 규칙 적용
             ② 그룹별로 max_week_normalized 주차가 없으면 ‘A’(+‘B’) 행을 한꺼번에 추가
       • 대상 DF : df_in_BO_FCST  (input)  ⋈  df_fn_step01_added_week  (output of Step 1)
       • group key : [Item, Ship-To, Location, Virtual BO ID, BO ID]
       • 성능 최적화를 위해 group-loop 대신 판다스/넘파이 벡터 연산 사용
    """
    # ──────────────────────────────────────────────────────────────────────────
    # 1) 준비 : Join + helper column
    # ──────────────────────────────────────────────────────────────────────────
    df_bo   = input_dataframes[STR_DF_IN_BO_FCST]                # 원본 Forecast
    df_bod  = output_dataframes[STR_DF_STEP01_ADDED_WEEK]        # Step 1 결과 (BO Total BOD LT)
    
    df_ret = (
        df_bo
        .merge(
            df_bod[[COL_ITEM, COL_LOC, COL_BO_TOTAL_BOD_LT]],
            on=[COL_ITEM, COL_LOC],
            how='inner',
            suffixes=('', '_added')
        )
    )
    # 숫자형 Week (col ‘202506A’ → ‘202506’)
    df_ret[COL_CURRENT_ROW_WEEK]    = df_ret[COL_TIME_PW].str.replace(r'\D', '', regex=True)
    df_ret[COL_BO_TOTAL_BOD_LT_NOM] = df_ret[COL_BO_TOTAL_BOD_LT].str.replace(r'\D', '', regex=True)
    # ──────────────────────────────────────────────────────────────────────────
    # 2) 기본 Lock 규칙 (broadcast)  ― 기존 if-else 유지하되 벡터 적용
    # ──────────────────────────────────────────────────────────────────────────
    cur_week = current_week_normalized
    bod_nom  = df_ret[COL_BO_TOTAL_BOD_LT_NOM]
    # Case ①  Virtual BO ID = '-', BO ID = '-'
    m1 = (df_ret[COL_VIRTUAL_BO_ID] == '-') & (df_ret[COL_BO_ID] == '-')
    df_ret.loc[m1, COL_BO_FCST_LOCK] = (
        (df_ret[COL_CURRENT_ROW_WEEK] >= cur_week) &
        (df_ret[COL_CURRENT_ROW_WEEK] <= bod_nom)
    )
    df_ret.loc[m1 & (df_ret[COL_CURRENT_ROW_WEEK] == max_week_normalized),
               COL_BO_FCST_LOCK] = False
    # Case ②  Virtual BO ID ≠ '-', BO ID = '-'
    m2 = (df_ret[COL_VIRTUAL_BO_ID] != '-') & (df_ret[COL_BO_ID] == '-')
    df_ret.loc[m2 & (df_ret[COL_CURRENT_ROW_WEEK] == max_week_normalized),
               COL_BO_FCST_LOCK] = True
    # Case ③  Virtual BO ID ≠ '-', BO ID ≠ '-'
    m3 = (df_ret[COL_VIRTUAL_BO_ID] != '-') & (df_ret[COL_BO_ID] != '-')
    df_ret.loc[m3 & (df_ret[COL_CURRENT_ROW_WEEK] == bod_nom),
               COL_BO_FCST_LOCK] = False
    df_ret.loc[m3 & (df_ret[COL_CURRENT_ROW_WEEK] == max_week_normalized),
               COL_BO_FCST_LOCK] = True
    # ──────────────────────────────────────────────────────────────────────────
    # 3) 그룹별 max_week_normalized 행 존재 여부 파악 (벡터)
    # ──────────────────────────────────────────────────────────────────────────
    grp_cols = [COL_ITEM, COL_SHIP_TO, COL_LOC, COL_VIRTUAL_BO_ID, COL_BO_ID]
    # has_max = True / False per group
    has_max = (
        df_ret
        .assign(flag = df_ret[COL_CURRENT_ROW_WEEK] == max_week_normalized)
        .groupby(grp_cols, sort=False)['flag']
        .any()
        .reset_index(name='has_max')
    )
    groups_need = has_max[~has_max['has_max']].drop(columns='has_max')
    if groups_need.empty:
        return df_ret         # 모든 그룹에 이미 존재 → 그대로 반환
    # ──────────────────────────────────────────────────────────────────────────
    # 4) 각 그룹의 ‘최신’(가장 큰 Week) 행을 추출 (1 pass, no loop)
    # ──────────────────────────────────────────────────────────────────────────
    df_sorted = df_ret.sort_values(by=[*grp_cols, COL_TIME_PW], ascending=[True,True,True,True,True,False])
    latest_rows = (
        df_sorted
        .drop_duplicates(subset=grp_cols, keep='first')          # 첫 행 = 최신 주차
        .merge(groups_need, on=grp_cols, how='inner')            # missing group 만 추림
        .reset_index(drop=True)
    )
    # ──────────────────────────────────────────────────────────────────────────
    # 5) 새 행 생성 (‘A’, ‘B’ suffix) ― 벡터 broadcast
    # ──────────────────────────────────────────────────────────────────────────
    suffixes = ['A', 'B'] if gfn_is_partial_week(max_week_normalized) else ['A']
    new_rows_all = []
    for sfx in suffixes:                # 두 개뿐이라  O(2) 반복은 무시 가능한 비용
        add = latest_rows.copy()
        add[COL_TIME_PW]          = max_week_normalized + sfx
        add[COL_CURRENT_ROW_WEEK] = max_week_normalized
        # Lock 결정 (vector)
        cond_no_bo = (add[COL_VIRTUAL_BO_ID] == '-') & (add[COL_BO_ID] == '-')
        add[COL_BO_FCST_LOCK] = True
        add.loc[cond_no_bo, COL_BO_FCST_LOCK] = False
        new_rows_all.append(add)
    df_new = pd.concat(new_rows_all, ignore_index=True)
    # ──────────────────────────────────────────────────────────────────────────
    # 6) 기존 DF + 신규 DF concat → 완료
    # ──────────────────────────────────────────────────────────────────────────
    df_final = pd.concat([df_ret, df_new], ignore_index=True)
    # (선택) 정렬 유지 : 필요한 경우 group key + 주차로 정렬
    df_final = df_final.sort_values(by=[*grp_cols, COL_TIME_PW]).reset_index(drop=True)
    return df_final
    """
    # ═══════════════════════════════════════════════════════════════════════
    #  DUCKDB Quick Check (LOCAL ONLY) – step03 vectorization validity
    # ═══════════════════════════════════════════════════════════════════════
    import duckdb, os
    duckdb.register('df_step03', df_final)
    qry = f'''
        SELECT {COL_ITEM}, {COL_SHIP_TO}, {COL_LOC},
               MIN({COL_TIME_PW}) AS min_pw, MAX({COL_TIME_PW}) AS max_pw
        FROM  df_step03
        GROUP BY {COL_ITEM}, {COL_SHIP_TO}, {COL_LOC}
        LIMIT 20;
    '''
    df_chk = duckdb.query(qry).to_df()
    fn_log_dataframe(df_chk, 'vectorize_step03_check')
    """


@_decoration_
def step04_create_virtual_bo():
    """
    Step 4: Create virtual BOs based on df_out_step03_lock.
    Uses global variables for input dataframe and Virtual_BO_ID_MAX.
    :return: Dataframe with virtual BOs created.
    """

    # Identify data for creating virtual BOs
    df_step03 = output_dataframes[STR_DF_STEP03_LOCK]
    tobe_added_data = df_step03[(df_step03[COL_VIRTUAL_BO_ID] == '-') & 
                                         (df_step03[COL_BO_ID] == '-')].copy()
    
    def _get_new_virtual_bo_id(x):
        import re
        # re.search(r'VBO_(\D*)(\d+)', Virtual_BO_ID_MAX)
        # VBO_(\D*)(\d+)
        # VBO_ : literal string
        # (\D*) : not digit (0 or more time)
        # (\d+) : digit (1 or more time)
        match = re.search(r'(\D*)(\d+)', Virtual_BO_ID_MAX)
        if match:
            return f"{match.group(1)}{int(match.group(2)) + x + 1:08}"
        else:
            raise ValueError(f"Virtual_BO_ID_MAX {Virtual_BO_ID_MAX} is not a valid id")
    # Process by group
    tobe_added_data[COL_VIRTUAL_BO_ID] = tobe_added_data.groupby(
        [COL_ITEM, COL_SHIP_TO, COL_LOC]
    ).ngroup().apply(lambda x: _get_new_virtual_bo_id(x))


    
    tobe_added_data[COL_BO_ID] = '-'
    
    # Set 'BO FCST' and 'BO FCST.Lock'
    def _get_bo_fcst(row):
        bo_fcst = row[COL_BO_FCST]
        time_normalized = row[COL_CURRENT_ROW_WEEK]
        bo_total_bod_lt_normalized = row[COL_BO_TOTAL_BOD_LT_NOM]
        if time_normalized == bo_total_bod_lt_normalized:
            return int(bo_fcst)  # Ensure the return value is an integer
        else:
            return 0
    
    tobe_added_data[COL_BO_FCST] = tobe_added_data.apply(_get_bo_fcst, axis=1)
    tobe_added_data[COL_BO_FCST_LOCK] = tobe_added_data.apply(
        lambda row: False if row[COL_CURRENT_ROW_WEEK] == row[COL_BO_TOTAL_BOD_LT_NOM] else True,
        axis=1
    )
    
    # Add tobe_added_data to df_out_step03_lock
    df_step04 = pd.concat([df_step03, tobe_added_data], ignore_index=True)
    
    return df_step04

# Example usage
# dict_log = {
#     'p_step_no': 40,
#     'p_step_desc': 'Step 4: Create Virtual BOs',
#     'p_df_name': 'df_out_step04_created'
# }
# df_out_step04_created = step04_create_virtual_bo(**dict_log)
# output_dataframes['df_out_step04_created'] = df_out_step04_created
# fn_log_dataframe(df_out_step04_created, 'df_out_step04_created')


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

            # -- COL_VIRTUAL_BO_ID 이 - 으로만 구성된 경우
            input_folder_name  = 'PYForecastB2BLockAndRolling_bo_'
            output_folder_name = 'PYForecastB2BLockAndRolling_bo_'
            
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
        # # Example usage
        # dict_log = {
        #     'p_step_no': 30,
        #     'p_step_desc': 'Step 3: Adjust for negative values',
        #     'p_df_name': 'df_out_step03_adjust'
        # }
        # df_out_step03_adjust = step03_adjust_negative_values(**dict_log)
        # output_dataframes['df_out_step03_adjust'] = df_out_step03_adjust
        # fn_log_dataframe(df_out_step03_adjust, 'df_out_step03_adjust')

        dict_log = {
            'p_step_no': 10,
            'p_step_desc': 'Step 1: Process Total BOD LT'
            # 'p_df_name': 'df_out_step01_added_week'
        }
        df_fn_step01_added_week = step01_process_total_bod_lt(**dict_log)
        output_dataframes[STR_DF_STEP01_ADDED_WEEK] = df_fn_step01_added_week
        fn_log_dataframe(df_fn_step01_added_week, STR_DF_STEP01_ADDED_WEEK)

        dict_log = {
            'p_step_no': 20,
            'p_step_desc': 'Step 2: Find Max Virtual BO ID',
            # 'p_df_name': 'Virtual_BO_ID_MAX'
        }
        Virtual_BO_ID_MAX = step02_find_max_virtual_bo_id(**dict_log)
        # output_dataframes['Virtual_BO_ID_MAX'] = Virtual_BO_ID_MAX
        # fn_log_dataframe(Virtual_BO_ID_MAX, 'Virtual_BO_ID_MAX')

        dict_log = {
            'p_step_no': 30,
            'p_step_desc': 'Step 3: Apply Lock Conditions'
            # 'p_df_name': 'df_out_step03_lock'
        }
        df_step03_lock = step03_apply_lock_conditions(**dict_log)
        output_dataframes[STR_DF_STEP03_LOCK] = df_step03_lock
        fn_log_dataframe(df_step03_lock, STR_DF_STEP03_LOCK)

        dict_log = {
            'p_step_no': 40,
            'p_step_desc': 'Step 4: Create Virtual BOs'
            # 'p_df_name': 'df_out_step04_created'
        }
        df_step04_created = step04_create_virtual_bo(**dict_log)
        output_dataframes[STR_DF_STEP04_CREATED] = df_step04_created
        fn_log_dataframe(df_step04_created, STR_DF_STEP04_CREATED)

        ################################################################################################################
        # 최종 Output 정리
        ################################################################################################################
        dict_log = {
            'p_step_no': 900,
            'p_step_desc': '최종 Output 정리 - out_Demand',
            'p_df_name': 'out_Demand'
        }
        out_Demand = fn_output_formatter(df_step04_created, Version, **dict_log)
        
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
            input_path = f'{str_output_dir}/input'
            os.makedirs(input_path,exist_ok=True)
            for input_file in input_dataframes:
                input_dataframes[input_file].to_csv(input_path + "/"+input_file+".csv", encoding="UTF8", index=False)

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
