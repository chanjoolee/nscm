"""
    * 프로그램명
        - PYForecastVDPanelMonthInput

    * 내용
        - VD 패널 패널확정/미확정이 겹치는 달의 월총량 변경시 주차배분 방식

    * Panel 확정/미확정 구간	
        - VD에서 제공할 Total BOD L/T 정보를 사용하면 됨. (Netting 에서 사용할 주차 정보와 동일)
        - Total BOD L/T 은 3-digit으로 구성
        - 맨 앞자리가 1- or 2- 이면 패널 홪겅 구간, 3- 이면 미확정 구간
        Total BOD L/T은 NSCM 에서만 구현될 것이므로, VD Tenant로부터 I/F 받을 예정
        * Grain 불명확함


    * 변경이력
        - 2025.02.27 /  / Create

    * Script Params
        - Version : 기본값 : CurrentWorkingView, 호출시에, 결과데이터를 Version Name을 별도로 지정하는데에 사용.


    * Input Tables
        - Input 1: df_in_S/In_FCST(GI)_AP2 ==> df_in_SIn_FCST_GI_AP2			
            - query				
                Select ([Version].[Version Name]
                * [Sales Domain].[Ship To]
                * [Item].[Item]
                * [Location].[Location]
                * [Time].[Week] ) on row, 
                ( { Measure.[S/In FCST(GI)_AP2] } ) on column;
                                
            
            - Case 1
                - filename : df_In_FCST(GI)_AP2_1
                - data				
                    Version.[Version Name]	Sales Domain.[Ship To]	Item.[Item]	Loaction.[Location]	Time.[Week]	S/In FCST(GI)_AP2
                    CurrentWorkingView	A5000001	P1	S001	202436	100
                    CurrentWorkingView	A5000001	P1	S001	202437	100
                    CurrentWorkingView	A5000001	P1	S001	202438	100
                    CurrentWorkingView	A5000001	P1	S001	202439	100
                    CurrentWorkingView	A5000001	P1	S001	202440	100
            - Case 2   
                - filename : df_In_FCST(GI)_AP2_2
                - data
                    Version.[Version Name]	Sales Domain.[Ship To]	Item.[Item]	Loaction.[Location]	Time.[Week]	S/In FCST(GI)_AP2                     
                    CurrentWorkingView	A5000001	P2	S001	202436	80
                    CurrentWorkingView	A5000001	P2	S001	202437	80
                    CurrentWorkingView	A5000001	P2	S001	202438	80
                    CurrentWorkingView	A5000001	P2	S001	202439	80
                    CurrentWorkingView	A5000001	P2	S001	202440	80

        - Input 2:  Total BOD L/T 정보					
            - df_in_Total_BOD_LT	
                - query			
                    Select ([Version].[Version Name]
                    * [Item].[Item]
                    * [Location].[Location]
                    * [Time].[Week] ) on row, 
                    ( { Measure.[Total BOD L/T] } ) on column;	
                            
            - Case 1:
                - filename: df_in_Total BOD_LT_1	
                - data		
                    Version.[Version Name]	Item.[Item]	Loaction.[Location]	Time.[Week]	Total BOD L/T
                    CurrentWorkingView	P1	S001	202436	100
                    CurrentWorkingView	P1	S001	202437	100
                    CurrentWorkingView	P1	S001	202438	200
                    CurrentWorkingView	P1	S001	202439	300
                    CurrentWorkingView	P1	S001	202440	
            - Case 2: 
                - filename: df_in_Total BOD_LT_2
                - data
                    Version.[Version Name]	Item.[Item]	Loaction.[Location]	Time.[Week]	Total BOD L/T             
                    CurrentWorkingView	P2	S001	202436	100
                    CurrentWorkingView	P2	S001	202437	100
                    CurrentWorkingView	P2	S001	202438	200
                    CurrentWorkingView	P2	S001	202439	300
                    CurrentWorkingView	P2	S001	202440	

    * Output Tables
        - out_Demand
            - query
                Select ([Version].[Version Name]
                * [Sales Domain].[Ship To] 
                * [Item].[Item] 
                * [Location].[Location] 
                * [Time].[Week] ) on row, 
                ( { Measure.[S/In FCST(GI)_AP2] } ) on column;


            - column
                [Version].[Version Name]
                [Sales Domain].[Ship To] 
                [Item].[Item] 
                [Location].[Location] 
                [Time].[Week]
                Measure.[S/In FCST(GI)_AP2] 
                    - comment: Lock과 음수를 고려한 월총량 분배

    * INPUT Parameter
        - MonthTotalValue : 600

    * Flow Summary
        - Step1 : set df_in_SIn_FCST_GI_AP2 based on df_in_Total BOD LT
            - input: 
                - df_in_SIn_FCST_GI_AP2
                    this can be called like input_dataframes['df_in_SIn_FCST_GI_AP2']
                - df_in_Total_BOD_LT
                    this can be called like input_dataframes['df_in_Total_BOD_LT']
            - process
                - load data
                    - read df_in_SIn_FCST_GI_AP2
                    - read df_in_Total_BOD_LT
                - drop column 'Version.[Version Name]' in df_in_SIn_FCST_GI_AP2
                - drop column 'Version.[Version Name]' in df_in_Total_BOD_LT
                - join df_in_SIn_FCST_GI_AP2 with df_in_Total_BOD_LT
                    - join on ['Item.[Item]','Loaction.[Location]','Time.[Week]']
                    - data will be like below
                        Version.[Version Name]	Sales Domain.[Ship To]	Item.[Item]	Loaction.[Location]	Time.[Week]	S/In FCST(GI)_AP2	Total BOD L/T
                        CurrentWorkingView	A5000001	P1	S001	202436	100	100
                        CurrentWorkingView	A5000001	P1	S001	202437	100	100
                        CurrentWorkingView	A5000001	P1	S001	202438	100	200
                        CurrentWorkingView	A5000001	P1	S001	202439	100	300
                        CurrentWorkingView	A5000001	P1	S001	202440	100	

                - return  joind df
            - output: 
                - dataframe name : df_out_step01_join

        - Step2 : MonthTotalValue 를 unLock 구간에 월 총량 분배
            - input: 
                - df_out_step01_join
            - process
                - load data
                    - read df_out_step01_join
                - determine  unlock range
                    - if 'Total BOD L/T' is start with 1 or 2 , then row is lock range
                    - else row is unlock range
                - determine value to be allocated
                    - allocateSum = MonthTotalValue - sum of 'S/In FCST(GI)_AP2' in lock range
                        - MonthTotalValue is read from parameter
                    - sumOfAp2_in_unlock_range = sum of 'S/In FCST(GI)_AP2' in unlock range
                - allocate value to unlock range
                    - logic is below. but don't use for loop. use apply method with lambda function in pandas
                    - for each row
                        - if unlock range
                            - allocateValue = allocateSum * (row['S/In FCST(GI)_AP2'] / sumOfAp2_in_unlock_range)
                            - set 'S/In FCST(GI)_AP2' = allocateValue
                        - else
                            - do nothing
                
                        
                - return df_out_step01_join
            - output: 
                - dataframe name : df_out_step02_allocate

        - Step3 : 음수인 경우 패널 확정 주차에서 차감
            - concept
                - if allocateSum is less than 0, then set 'S/In FCST(GI)_AP2' = 0
                - and value of minus is added to 'S/In FCST(GI)_AP2' in previous week , until 'S/In FCST(GI)_AP2' >= 0
            - input: 
                - df_out_step02_allocate
            - process
                - load data
                    - read df_out_step02_allocate
                    - datad will be like below
                        Version.[Version Name]	Sales Domain.[Ship To]	Item.[Item]	Loaction.[Location]	Time.[Week]	S/In FCST(GI)_AP2	Total BOD L/T
                        CurrentWorkingView	A5000001	P2	S001	202436	80	100
                        CurrentWorkingView	A5000001	P2	S001	202437	80	100
                        CurrentWorkingView	A5000001	P2	S001	202438	80	200
                        CurrentWorkingView	A5000001	P2	S001	202439	-20	300
                        CurrentWorkingView	A5000001	P2	S001	202440	-20	300
                - logic is below. but don't use for loop. if possible, use apply method with lambda function in pandas
                - find row have negative value
                - if there is no negative value
                    - return df_out_step02_allocate
                - do loop rows have negative value in order of 'Time.[Week]' descending
                - set value of negative row to zero in biggest row
                - add value of negative row to previous week until 'S/In FCST(GI)_AP2' >= 0
                - data will be like below after process
                    Version.[Version Name]	Sales Domain.[Ship To]	Item.[Item]	Loaction.[Location]	Time.[Week]	S/In FCST(GI)_AP2	Total BOD L/T
                    CurrentWorkingView	A5000001	P2	S001	202436	80	100
                    CurrentWorkingView	A5000001	P2	S001	202437	80	100
                    CurrentWorkingView	A5000001	P2	S001	202438	40	200
                    CurrentWorkingView	A5000001	P2	S001	202439	0	300
                    CurrentWorkingView	A5000001	P2	S001	202440	0	300
                - convert type of 'S/In FCST(GI)_AP2' to int
                - return df_out_step02_allocate
            - output: 
                - dataframe name : df_out_step03_adjust
            

    * Validation :

    * Execution
        EXEC plugin instance [PYForecastVDPanelMonthInput]	
            for measures {Measure.[S/In FCST(GI)_AP2]}	
        using scope ([Version].[Version Name].[CurrentWorkingView] * [Sales Domain].[Ship To] .[ {{ InputShipTo  }} ] * [Item].[Item].[ {{ InputItem }} ] * [Location].[Location].[ {{ InputLocation  }} ]  * [Time].[Week].{{#filter InputMonthtoWeek}}	
        using arguments {	
                        (ExecutionMode, "MediumWeight")	
                    , (MonthTotalValue, {{ InputMonthTotalValue }} )	
                        }
        ;
"""

import os,sys
import time,datetime,shutil
import inspect
import traceback
import pandas as pd
from NSCMCommon import NSCMCommon as common
from NSCMCommon import VDCommon as vdCommon
import glob

########################################################################################################################
# Local 개발 시에 필요한 공통 변수 선언
########################################################################################################################
# o9에 저장된 instanceName
str_instance = 'PYForecastVDPanelMonthInput'
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
# 컬럼상수
########################################################################################################################
# ── column constants ──────────────────────────────────────────────────────────────────────────────────────────
COL_VERSION         = 'Version.[Version Name]'
COL_ITEM            = 'Item.[Item]'
COL_SHIP_TO         = 'Sales Domain.[Ship To]'
COL_LOC             = 'Location.[Location]'
COL_TIME_WK         = 'Time.[Week]'
COL_TIME_PW         = 'Time.[Partial Week]'
COL_TOTAL_BOD_LT    = 'VD Total BOD LT'  
COL_SIN_FCST_AP2    = 'S/In FCST(GI)_AP2'

# ───────────────────────────────────────────────────────────────
# CONSTANT STRING VARIABLES FOR DATAFRAME NAMES
# ───────────────────────────────────────────────────────────────
# input
STR_DF_IN_FCST              = 'df_in_SIn_FCST_GI_AP2'
STR_DF_IN_TOTAL_BOD_LT      = 'df_in_Total_BOD_LT'
# middle    
STR_DF_STEP01_JOIN          = 'df_step01_join'
STR_DF_STEP02_ALLOCATE      = 'df_step02_allocate'
STR_DF_STEP03_ADJUST        = 'df_step03_adjust'
# out
STR_OUT_DEMAND              = 'out_Demand'

################  Start of Functions  ################
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
    return ''.join(filter(str.isdigit, week_str))





################  End of Functions  ################




@_decoration_
def fn_output_formatter(df_p_source: pd.DataFrame, str_p_out_version: str) -> pd.DataFrame:
    """
    최종 Output 형태로 정리
    :param df_p_source: 주차별로 가공하여 group by 후 sum을 구한 in_Demand
    :param str_p_out_version: Version
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
        COL_SHIP_TO,
        COL_ITEM,
        COL_LOC,
        COL_TIME_WK,
        COL_SIN_FCST_AP2
        # COL_TOTAL_BOD_LT
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
            COL_SHIP_TO         : [],
            COL_ITEM            : [],
            COL_LOC             : [],
            COL_TIME_WK         : [],
            COL_SIN_FCST_AP2    : []
            # COL_TOTAL_BOD_LT  : []
        }
    )

    return df_return



def fn_convert_type(df: pd.DataFrame, startWith: str, type):
    for column in df.columns:
        if column.startswith(startWith):
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
            "df_in_SIn_FCST_GI_AP2.csv"     :    STR_DF_IN_FCST       ,
            "df_in_Total_BOD_LT.csv"        :    STR_DF_IN_TOTAL_BOD_LT    
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

            for keyword, frame_name in file_to_df_mapping.items():
                if file_name.startswith(keyword.split('.')[0]):
                    input_dataframes[frame_name] = df
                    break

        fn_convert_type(input_dataframes[STR_DF_IN_FCST], COL_TIME_WK, str)
        fn_convert_type(input_dataframes[STR_DF_IN_TOTAL_BOD_LT], COL_TIME_WK, str)
        input_dataframes[STR_DF_IN_FCST][COL_SIN_FCST_AP2].fillna(0, inplace=True)
        fn_convert_type(input_dataframes[STR_DF_IN_FCST], COL_SIN_FCST_AP2, 'int32')
        input_dataframes[STR_DF_IN_TOTAL_BOD_LT][COL_TOTAL_BOD_LT].fillna(0, inplace=True)
        fn_convert_type(input_dataframes[STR_DF_IN_TOTAL_BOD_LT], COL_TOTAL_BOD_LT, 'int32')
        logger.info("loaded dataframes")
    else:
        # o9 에서 
        input_dataframes[STR_DF_IN_FCST]            = df_in_SIn_FCST_GI_AP2
        input_dataframes[STR_DF_IN_TOTAL_BOD_LT]    = df_in_Total_BOD_LT


@_decoration_
def step01_join_dataframes() -> pd.DataFrame:
    """
    Step 1: Load and join data from df_in_SIn_FCST_GI_AP2 and df_in_Total_BOD_LT.
    """
    # Load data
    df_in_fcst = input_dataframes[STR_DF_IN_FCST]
    df_in_bod_lt = input_dataframes[STR_DF_IN_TOTAL_BOD_LT]

    # Drop unnecessary columns
    df_in_fcst = df_in_fcst.drop(columns=[COL_VERSION])
    df_in_bod_lt = df_in_bod_lt.drop(columns=[COL_VERSION])

    # Join dataframes
    df_out_step01_join = pd.merge(
        df_in_fcst, df_in_bod_lt, 
        on=[COL_ITEM, COL_LOC, COL_TIME_WK], 
        how='left')
    df_out_step01_join[COL_TOTAL_BOD_LT].fillna(0, inplace=True)
    fn_convert_type(df_out_step01_join, COL_TOTAL_BOD_LT, 'int32')
    return df_out_step01_join

@_decoration_
def step02_allocate_month_total_value_back(month_total_value: int) -> pd.DataFrame:
    """
    Step 2: Allocate MonthTotalValue to the unlock range.
    """
    logger.Note(p_note='Step 2', p_log_level=LOG_LEVEL.debug())
    # Load data
    df_out_step01_join = output_dataframes[STR_DF_STEP01_JOIN]

    # Determine unlock range
    logger.Note(p_note='Determine unlock range', p_log_level=LOG_LEVEL.debug())
    lock_condition = df_out_step01_join[COL_TOTAL_BOD_LT].astype(str).str.startswith(('1', '2'))
    unlock_condition = ~lock_condition

    # Determine values to allocate
    logger.Note(p_note='Determine values to allocate', p_log_level=LOG_LEVEL.debug())
    df_out_step01_join[COL_SIN_FCST_AP2] = df_out_step01_join[COL_SIN_FCST_AP2].round().astype(int)
    allocate_sum = month_total_value - df_out_step01_join.loc[lock_condition, COL_SIN_FCST_AP2].sum()
    logger.Note(p_note=f'allocate_sum: {allocate_sum}', p_log_level=LOG_LEVEL.debug())
    sum_of_ap2_in_unlock_range = df_out_step01_join.loc[unlock_condition, COL_SIN_FCST_AP2].sum()
    logger.Note(p_note=f'unlock_condition: {unlock_condition}', p_log_level=LOG_LEVEL.debug())

    # Allocate values to unlock range
    def allocate(row):
        if unlock_condition.loc[row.name]:
            allocate_value = allocate_sum * (row[COL_SIN_FCST_AP2] / sum_of_ap2_in_unlock_range)
            return allocate_value
        return row[COL_SIN_FCST_AP2]

    
    df_out_step01_join[COL_SIN_FCST_AP2] = df_out_step01_join.apply(allocate, axis=1)

    # Convert to integers
    logger.Note(p_note='Convert to integers', p_log_level=LOG_LEVEL.debug())
    df_out_step01_join[COL_SIN_FCST_AP2] = df_out_step01_join[COL_SIN_FCST_AP2].round().astype(int)

    # Adjust to match allocate_sum
    logger.Note(p_note='Adjust to match allocate_sum', p_log_level=LOG_LEVEL.debug())
    difference = allocate_sum - df_out_step01_join.loc[unlock_condition, COL_SIN_FCST_AP2].sum()
    if difference != 0:
        # Adjust the first unlock row to account for the rounding difference
        first_unlock_index = df_out_step01_join.loc[unlock_condition].index[0]
        df_out_step01_join.at[first_unlock_index, COL_SIN_FCST_AP2] += difference

    return df_out_step01_join

@_decoration_
def step02_allocate_month_total_value(month_total_value) -> pd.DataFrame:
    """
    Step 2 – MonthTotalValue를 *미확정(unlock)* 구간에 비례-재분배
      • month_total_value : 정수·실수·문자열 모두 허용(내부에서 숫자로 변환)
      • Lock 판정  →  Total BOD L/T 컬럼이 ‘1*’ 또는 ‘2*’ 로 시작 → 확정
                     그 외(‘3*’ 또는 0/NaN)                  → 미확정
      • 분배 로직
          1. locked_sum = 확정구간 FCST 합계
          2. allocate_sum = MonthTotalValue − locked_sum
          3. unlock 비율 = 기존 FCST / unlock FCST 총합  
             (총합이 0이면 균등 분배)
          4. 배열 연산으로 신값 산출 → round()·astype(int) 
          5. 반올림 오차(diff) 1줄에 보정
    """
    df = output_dataframes[STR_DF_STEP01_JOIN].copy(deep=True)    
    # ── 1) 숫자형 보장 ─────────────────────────────────────────────────────
    month_total_value = pd.to_numeric(month_total_value, errors='coerce')
    if pd.isna(month_total_value):
        raise ValueError('MonthTotalValue must be numeric.')

    df[COL_SIN_FCST_AP2] = (
        pd.to_numeric(df[COL_SIN_FCST_AP2], errors='coerce')
        .fillna(0)
        .astype(float)
    )
    df[COL_TOTAL_BOD_LT] = (
        pd.to_numeric(df[COL_TOTAL_BOD_LT], errors='coerce')
        .fillna(0)
        .astype(int)
    )

    # ── 2) Lock / Unlock 판정 ────────────────────────────────────────────
    lock_mask   = df[COL_TOTAL_BOD_LT].astype(str).str.startswith(('1', '2'))
    unlock_mask = ~lock_mask

    locked_sum  = df.loc[lock_mask,   COL_SIN_FCST_AP2].sum()
    allocate_sum = month_total_value - locked_sum        # <— 문제 위치 해결

    # ── 3) 분배 대상이 없으면 그대로 반환 ────────────────────────────────
    if (allocate_sum == 0) or (unlock_mask.sum() == 0):
        return df

    base_unlock_sum = df.loc[unlock_mask, COL_SIN_FCST_AP2].sum()

    # ── 4) 비례(or 균등) 재분배 ─────────────────────────────────────────
    if base_unlock_sum == 0:
        # 모든 unlock 값이 0 → 균등 분배
        equal_share = allocate_sum // unlock_mask.sum()
        df.loc[unlock_mask, COL_SIN_FCST_AP2] = equal_share
        remainder = int(allocate_sum - equal_share * unlock_mask.sum())
    else:
        # 기존 비율로 분배
        ratio = df.loc[unlock_mask, COL_SIN_FCST_AP2] / base_unlock_sum
        new_values = (ratio * allocate_sum).round().astype(int)
        df.loc[unlock_mask, COL_SIN_FCST_AP2] = new_values
        remainder = int(allocate_sum - new_values.sum())

    # ── 5) 반올림 오차 1행에 보정 ───────────────────────────────────────
    if remainder != 0:
        first_idx = df.index[unlock_mask][0]
        df.at[first_idx, COL_SIN_FCST_AP2] += remainder

    # 최종 정수형 캐스팅
    df[COL_SIN_FCST_AP2] = df[COL_SIN_FCST_AP2].astype(int)
    return df


@_decoration_
def step03_adjust_negative_values() -> pd.DataFrame:
    """
    Step 3 (벡터라이즈 버전)
      1)  Step 2 결과(df_step02_allocate) 로드
      2)  (Ship-To, Item, Location) 단위로 주차 오름차순 정렬
      3)  각 그룹을 NumPy 배열로 변환해 **역방향 누적-보정** 알고리즘 수행
            · 뒤쪽 주차에서 생긴 음수(부족분)를 앞쪽 주차에서 차감
            · 모든 주차의 S/In FCST(GI)_AP2 ≥ 0 보장
      4)  보정된 값을 DataFrame 에 반영하여 반환
    --------------------------------------------------------------------------
    벡터라이즈 포인트
      • iterrows / loc 루프 제거 –> 그룹 별 NumPy 배열에서만 O(n) 연산
      • 월 5~6 주차 × 수만 그룹까지도 Python 루프 병목 없이 처리
    """
    # ── 1) 데이터 로드 ──────────────────────────────────────────────────────
    df_src = output_dataframes[STR_DF_STEP02_ALLOCATE].copy(deep=True)    # 주차 정렬용 보조 컬럼 (숫자 YYYYWW)
    df_src['_WK_INT'] = (
        df_src[COL_TIME_WK]
        .str.extract(r'(\d{6})')         # '202447A' → '202447'
        .astype(int)
    )

    # ── 2) 그룹별 보정 함수 ────────────────────────────────────────────────
    def _fix_negative(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values('_WK_INT').copy()
        fcst = g[COL_SIN_FCST_AP2].to_numpy(dtype=int)

        # 역방향 누적-보정 ── O(주차)
        deficit = 0
        for i in range(len(fcst) - 1, -1, -1):          # 뒤→앞
            val = fcst[i] + deficit
            if val < 0:
                deficit = val          # 부족분 carry over
                fcst[i] = 0
            else:
                fcst[i] = val
                deficit = 0

        g[COL_SIN_FCST_AP2] = fcst
        return g

    # ── 3) 그룹별 적용 (vectorised apply) ──────────────────────────────────
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOC]
    df_fixed = (
        df_src
        .groupby(key_cols, sort=False, group_keys=False)
        .apply(_fix_negative)
        .drop(columns=['_WK_INT'])
    )

    # 정수형 유지
    df_fixed[COL_SIN_FCST_AP2] = df_fixed[COL_SIN_FCST_AP2].astype(int)

    return df_fixed

if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    # Output 테이블 선언
    out_Demand = pd.DataFrame()
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
            input_folder_name  = 'PYForecastVDPanelMonthInput_o9'
            output_folder_name = 'PYForecastVDPanelMonthInput_o9'
            
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
            CurrentPartialWeek  = '202447A'
            MonthTotalValue     = 600     
            
        # vdLog 초기화
        log_path = os.path.dirname(__file__) if is_local else ""
        vdCommon.gfn_pyLog_start(Version, str_instance, logger, is_local, log_path)
        # --------------------------------------------------------------------------
        # df_input 체크
        # --------------------------------------------------------------------------
        logger.Note(p_note='df_input 체크 시작', p_log_level=LOG_LEVEL.debug())
        fn_process_in_df_mst()
         # 입력 변수 중 데이터가 없는 경우 경고 메시지를 출력한다.
        for in_df in input_dataframes:
            fn_check_input_table(input_dataframes[in_df], in_df, '1')


        # current_week_normalized = normalize_week(CurrentPartialWeek)

        # --------------------------------------------------------------------------
        # Check value Of Script Params
        # --------------------------------------------------------------------------
        logger.Note(p_note=f'[Script Params] Version            : {Version}', p_log_level=LOG_LEVEL.info())
        logger.Note(p_note=f'[Script Params] MonthTotalValue    : {MonthTotalValue}', p_log_level=LOG_LEVEL.info())


        ################################################################################################################
        # Start of processing
        ################################################################################################################
        dict_log = {
            'p_step_no': 10,
            'p_step_desc': 'Step 1 : set df_in_SIn_FCST_GI_AP2 based on df_in_Total BOD LT'
        }
        df_step01_join = step01_join_dataframes(**dict_log)
        output_dataframes[STR_DF_STEP01_JOIN] = df_step01_join
        fn_log_dataframe(df_step01_join, STR_DF_STEP01_JOIN)

        dict_log = {
            'p_step_no': 20,
            'p_step_desc': 'Step 2 : MonthTotalValue 를 unLock 구간에 월 총량 분배'
        }
        df_step02_allocate = step02_allocate_month_total_value(MonthTotalValue, **dict_log)
        output_dataframes[STR_DF_STEP02_ALLOCATE] = df_step02_allocate
        fn_log_dataframe(df_step02_allocate, STR_DF_STEP02_ALLOCATE)

        dict_log = {
            'p_step_no': 30,
            'p_step_desc': 'Step 3: Adjust for negative values'
        }
        df_step03_adjust = step03_adjust_negative_values(**dict_log)
        output_dataframes[STR_DF_STEP03_ADJUST] = df_step03_adjust
        fn_log_dataframe(df_step03_adjust, STR_DF_STEP03_ADJUST)

        ################################################################################################################
        # 최종 Output 정리
        ################################################################################################################
        dict_log = {
            'p_step_no': 900,
            'p_step_desc': '최종 Output 정리 - out_Demand'
        }
        out_Demand = fn_output_formatter(df_step03_adjust, Version, **dict_log)
        fn_log_dataframe(out_Demand,STR_OUT_DEMAND)

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
                input_dataframes[input_file].to_csv(input_path + "/"+input_file.replace('/','') +".csv", encoding="UTF8", index=False)

            # # log
            # output_path = f'{str_output_dir}/output'
            # os.makedirs(output_path,exist_ok=True)
            # for output_file in output_dataframes:
            #     output_dataframes[output_file].to_csv(output_path + "/"+output_file+".csv", encoding="UTF8", index=False)

        # logger.info(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')
        logger.Finish()
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::') # 25.05.12 need warning Log by Logger Issue
        
        
