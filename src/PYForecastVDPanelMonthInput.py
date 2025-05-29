import os,sys
import time,datetime,shutil
import inspect
import traceback
import pandas as pd
from NSCMCommon import NSCMCommon as common
import glob
import numpy as np

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
COL_TOTAL_BOD_LT    = 'Total BOD L/T'  
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
            "df_In_FCST(GI)_AP2.csv" :      STR_DF_IN_FCST       ,
            "df_in_Total_BOD_LT.csv"   :    STR_DF_IN_TOTAL_BOD_LT    
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
        input_dataframes[STR_DF_IN_TOTAL_BOD_LT]    = df_in_Forecast_Rule


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
    return df_out_step01_join

@_decoration_
def step02_allocate_month_total_value(month_total_value: int) -> pd.DataFrame:
    """
    Step 2: Allocate MonthTotalValue to the unlock range.
    """
    # Load data
    df_out_step01_join = output_dataframes[STR_DF_STEP01_JOIN]

    # Determine unlock range
    lock_condition = df_out_step01_join[COL_TOTAL_BOD_LT].astype(str).str.startswith(('1', '2'))
    unlock_condition = ~lock_condition

    # Determine values to allocate
    allocate_sum = month_total_value - df_out_step01_join.loc[lock_condition, COL_SIN_FCST_AP2].sum()
    sum_of_ap2_in_unlock_range = df_out_step01_join.loc[unlock_condition, COL_SIN_FCST_AP2].sum()

    # Allocate values to unlock range
    def allocate(row):
        if unlock_condition.loc[row.name]:
            allocate_value = allocate_sum * (row[COL_SIN_FCST_AP2] / sum_of_ap2_in_unlock_range)
            return allocate_value
        return row[COL_SIN_FCST_AP2]

    df_out_step01_join[COL_SIN_FCST_AP2] = df_out_step01_join.apply(allocate, axis=1)

    # Convert to integers
    df_out_step01_join[COL_SIN_FCST_AP2] = df_out_step01_join[COL_SIN_FCST_AP2].round().astype(int)

    # Adjust to match allocate_sum
    difference = allocate_sum - df_out_step01_join.loc[unlock_condition, COL_SIN_FCST_AP2].sum()
    if difference != 0:
        # Adjust the first unlock row to account for the rounding difference
        first_unlock_index = df_out_step01_join.loc[unlock_condition].index[0]
        df_out_step01_join.at[first_unlock_index, COL_SIN_FCST_AP2] += difference

    return df_out_step01_join


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

            input_folder_name  = str_instance
            output_folder_name = str_instance
            
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
            MonthTotalValue     = 200     
            
        
        # --------------------------------------------------------------------------
        # df_input 체크 시작
        # --------------------------------------------------------------------------
        logger.Note(p_note='df_input 체크 시작', p_log_level=LOG_LEVEL.debug())
        fn_process_in_df_mst()
         # 입력 변수 중 데이터가 없는 경우 경고 메시지를 출력한다.
        for in_df in input_dataframes:
            fn_check_input_table(input_dataframes[in_df], in_df, '1')


        current_week_normalized = normalize_week(CurrentPartialWeek)


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
        

