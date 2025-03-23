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
str_instance = 'PYForecastVDPanelMonthInput'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"
is_local = common.gfn_get_isLocal()
is_print = True
flag_csv = True
flag_exception = True
# Global variable for max_week
# max_week = None
CurrentPartialWeek = None
max_week_normalized = None
CurrentPartialWeek_normalized = None

v_chunk_size = 100000

########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################
logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
LOG_LEVEL = common.G_log_level


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
    df_return['Version.[Version Name]'] = str_p_out_version

    columns_to_return = [
        'Version.[Version Name]',
        'Item.[Item]',
        'Sales Domain.[Ship To]',
        'Location.[Location]',
        'DP Virtual BO ID.[Virtual BO ID]',
        'DP BO ID.[BO ID]',
        'Time.[Partial Week]',
        'BO FCST',
        'BO FCST.Lock'
    ]

    df_return = df_return[columns_to_return]

    return df_return

def fn_convert_type(df: pd.DataFrame, startWith: str, type):
    for column in df.columns:
        if column.startswith(startWith):
            df[column] = df[column].astype(type)

def find_parent_level(domain_value):
    # Load the CSV file into a DataFrame
    df = input_dataframes['df_in_Sales_Domain_Dimension']
    
    # Iterate over each row to find the parent
    levels = ['Sales Domain.[Sales Domain LV2]', 'Sales Domain.[Sales Domain LV3]', 
              'Sales Domain.[Sales Domain LV4]', 'Sales Domain.[Sales Domain LV5]', 
              'Sales Domain.[Sales Domain LV6]', 'Sales Domain.[Sales Domain LV7]']
    for index, row in df.iterrows():
        # Check each level to find the domain_value
        for level_index, level in enumerate(levels):
            if domain_value == row[level]:
                # Return the parent value from the previous level
                if level_index > 0:  # Ensure there's a parent level
                    parent_level = levels[level_index - 1]
                    return row[parent_level]
    
    # Return None if no parent is found
    return None

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
def fn_process_in_df_mst():
    global input_dataframes, csv_files

    # 로컬인 경우 Output 폴더를 정리한다.
    for file in os.scandir(str_output_dir):
        os.remove(file.path)

    # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
    file_pattern = f"{os.getcwd()}/{str_input_dir}/*.csv" 
    csv_files = glob.glob(file_pattern)

    file_to_df_mapping = {
        "df_in_SIn_Dummy_202415.csv" :      "df_in_SIn"       ,
        "df_in_Forecast_Rule_AP2.csv"   :      "df_in_Forecast_Rule",
        "df_in_Sales_Domain_Dimension.csv" : "df_in_Sales_Domain_Dimension"  
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


    fn_convert_type(input_dataframes['df_in_Sales_Domain_Dimension'], 'Sales Domain', str)
    fn_convert_type(input_dataframes['df_in_Forecast_Rule'], 'Sales Domain', str)
    fn_convert_type(input_dataframes['df_in_SIn'], 'Sales Domain', str)
    # fn_convert_type(input_dataframes['df_in_SIn'], 'Sales Domain', str)

    input_dataframes['df_in_Forecast_Rule']['FORECAST_RULE AP2 FCST'].fillna(0, inplace=True)
    fn_convert_type(input_dataframes['df_in_Forecast_Rule'], 'FORECAST_RULE AP2 FCST', 'int32')
    input_dataframes['df_in_SIn']['S/In FCST(GI)_AP2'].fillna(0, inplace=True)
    fn_convert_type(input_dataframes['df_in_SIn'], 'S/In FCST(GI)_AP2', 'int32')

    # Call the function after loading dataframes
    # fn_pre_process_forecast_rules(input_dataframes['df_in_Forecast_Rule'])


@_decoration_
def fn_step01_remove_duplicate_forecast_rule():
    """Step 1: Forecast Rule Table 2LV 중복 제거 전처리
    For rows where 'FORECAST_RULE AP2 FCST' equals 2, if 'Sales Domain.[Ship To]' starts with '3', then update 
    the value using mapping from df_sales_domain_master. Remove duplicates.
    
    Parameters:
        df_forecast_rule: DataFrame of forecast rules.
        df_sales_domain_master: DataFrame containing Sales Domain Master mapping. Expected columns: 'SalesDomainShipTo' and 'ParentLevel'.
        p_step_no: Step number identifier.
        p_step_desc: Step description log.
        
    Returns:
        Processed DataFrame with duplicate rows removed.
    """
    df_processed = input_dataframes['df_in_Forecast_Rule'].copy()
    
    mask = df_processed["FORECAST_RULE AP2 FCST"] == 2
    
    # Apply conversion only for rows where 'Sales Domain.[Ship To]' starts with '3'
    condition = df_processed["Sales Domain.[Ship To]"].astype(str).str.startswith("3") & mask
    df_processed.loc[condition, "Sales Domain.[Ship To]"] = df_processed.loc[condition, "Sales Domain.[Ship To]"].apply(find_parent_level)
    
    # Remove duplicate rows based on all columns
    df_processed = df_processed.drop_duplicates()
    return df_processed

@_decoration_
def fn_step02_expand_sin_to_lv2_lv7():
    """
    Step 2: Enriched SIN data to SIN_DUMMY_SD.
    Process:
      - Load data from input and output dataframes
      - Join df_in_SIn and df_in_Sales_Domain_Dimension on the 'Sales Domain.[Ship To]' column using INNER JOIN

    Parameters:
      df_in_SIn: DataFrame loaded from the input dataframe 'df_in_SIn'.
      df_in_Sales_Domain_Dimension: DataFrame loaded from the output dataframe 'df_in_Sales_Domain_Dimension'.
      dict_log: Optional dictionary with logging details (e.g. 'p_step_no', 'p_step_desc').

    Returns:
      DataFrame: The enriched SIN dataframe, named df_step02_expand_sin_to_lv2_lv7.
    """

    df_in_Sales_Domain_Dimension = input_dataframes['df_in_Sales_Domain_Dimension']
    df_in_SIn = input_dataframes['df_in_SIn']
    if dict_log is not None:
        # Log step details
        print(f"Step {dict_log.get('p_step_no')}: {dict_log.get('p_step_desc')}")

    # Perform inner join on 'Sales Domain.[Ship To]'
    df_step02_expand_sin_to_lv2_lv7 = df_in_SIn.merge(df_in_Sales_Domain_Dimension, on="Sales Domain.[Ship To]", how="inner")

    return_columns = [
        "Version.[Version Name]",
        "Sales Domain.[Sales Domain LV2]",
        "Sales Domain.[Sales Domain LV3]",
        "Sales Domain.[Sales Domain LV4]",
        "Sales Domain.[Sales Domain LV5]",
        "Sales Domain.[Sales Domain LV6]",
        "Sales Domain.[Sales Domain LV7]",
        "Sales Domain.[Ship To]",
        "Location.[Location]",
        "Item.[Product Group]",
        "Item.[Item]",
        "Time.[Planning Month]",
        "Time.[Week]",
        "S/In FCST(GI)_AP2"
    ]
    return df_step02_expand_sin_to_lv2_lv7[return_columns]


@_decoration_
def fn_step03_1_sin_lv2():
    """Step3-1: Inner join for LV2 branch"""
    df_sin = output_dataframes['df_step02_expand_sin_to_lv2_lv7']
    df_forecast = output_dataframes['df_step01_remove_duplicate_forecast_rule']
    
    # Filter forecast rule data where Ship To starts with '2'
    filter_df_forecast = df_forecast[df_forecast['Sales Domain.[Ship To]'].astype(str).str.startswith('2')]
    
    # Perform inner join
    return_df = df_sin.merge(
        filter_df_forecast, 
        left_on=['Sales Domain.[Sales Domain LV2]', 'Item.[Product Group]'], 
        right_on=['Sales Domain.[Ship To]', 'Item.[Product Group]'], 
        how='inner'
    )
    
    # Set GBRULE column based on Forecast Rule level
    return_df.loc[return_df['FORECAST_RULE AP2 FCST'] == 2, 'GBRULE'] = return_df['Sales Domain.[Sales Domain LV2]']
    return_df.loc[return_df['FORECAST_RULE AP2 FCST'] == 3, 'GBRULE'] = return_df['Sales Domain.[Sales Domain LV3]']
    return_df.loc[return_df['FORECAST_RULE AP2 FCST'] == 4, 'GBRULE'] = return_df['Sales Domain.[Sales Domain LV4]']
    return_df.loc[return_df['FORECAST_RULE AP2 FCST'] == 5, 'GBRULE'] = return_df['Sales Domain.[Sales Domain LV5]']
    return_df.loc[return_df['FORECAST_RULE AP2 FCST'] == 6, 'GBRULE'] = return_df['Sales Domain.[Sales Domain LV6]']
    return_df.loc[return_df['FORECAST_RULE AP2 FCST'] == 7, 'GBRULE'] = return_df['Sales Domain.[Sales Domain LV7]']
    
    # Drop unused columns
    return_df.drop(columns=[
        'Sales Domain.[Sales Domain LV2]', 'Sales Domain.[Sales Domain LV3]', 'Sales Domain.[Sales Domain LV4]',
        'Sales Domain.[Sales Domain LV5]', 'Sales Domain.[Sales Domain LV6]', 'Sales Domain.[Sales Domain LV7]'
    ], inplace=True)
    
    return return_df


@_decoration_
def fn_step03_2_sin_lv3():
    """Step3-2: Inner join for LV3 branch"""
    df_sin = output_dataframes['df_step02_expand_sin_to_lv2_lv7']
    df_forecast = output_dataframes['df_step01_remove_duplicate_forecast_rule'].drop(columns=['Version.[Version Name]'])
    
    # Filter forecast rule data where Ship To starts with '3'
    filter_df_forecast = df_forecast[df_forecast['Sales Domain.[Ship To]'].astype(str).str.startswith('3')]
    
    # Perform inner join
    return_df = df_sin.merge(
        filter_df_forecast, 
        left_on=['Sales Domain.[Sales Domain LV3]', 'Item.[Product Group]'], 
        right_on=['Sales Domain.[Ship To]', 'Item.[Product Group]'], 
        how='inner'
    )
    
    # Set GBRULE column based on Forecast Rule level
    return_df.loc[return_df['FORECAST_RULE AP2 FCST'] == 3, 'GBRULE'] = return_df['Sales Domain.[Sales Domain LV3]']
    return_df.loc[return_df['FORECAST_RULE AP2 FCST'] == 4, 'GBRULE'] = return_df['Sales Domain.[Sales Domain LV4]']
    return_df.loc[return_df['FORECAST_RULE AP2 FCST'] == 5, 'GBRULE'] = return_df['Sales Domain.[Sales Domain LV5]']
    return_df.loc[return_df['FORECAST_RULE AP2 FCST'] == 6, 'GBRULE'] = return_df['Sales Domain.[Sales Domain LV6]']
    return_df.loc[return_df['FORECAST_RULE AP2 FCST'] == 7, 'GBRULE'] = return_df['Sales Domain.[Sales Domain LV7]']
    
    # Drop unused columns
    return_df.drop(columns=[
        'Sales Domain.[Sales Domain LV2]', 'Sales Domain.[Sales Domain LV3]', 'Sales Domain.[Sales Domain LV4]',
        'Sales Domain.[Sales Domain LV5]', 'Sales Domain.[Sales Domain LV6]', 'Sales Domain.[Sales Domain LV7]'
    ], inplace=True)
    
    return return_df


@_decoration_
def fn_step03_3_concat_lv2_lv3():
    """Step3-3: Concatenate results from Step3-1 and Step3-2"""
    df_lv2 = output_dataframes['df_out_step03_1_sin_lv2']
    df_lv3 = output_dataframes['df_out_step03_2_sin_lv3']
    
    return pd.concat([df_lv2, df_lv3], ignore_index=True)


@_decoration_
def fn_step03_4_group_sum():
    """Step3-4: Group by GBRULE and sum the values"""
    df_grouped = output_dataframes['df_out_step03_3_sin_lv2_lv3'].drop(columns=['Item.[Product Group]'])
    
    return df_grouped.groupby([
        'Version.[Version Name]',
        'GBRULE',
        'Location.[Location]',
        'Item.[Item]',
        'Time.[Week]'
    ], as_index=False).sum()

@_decoration_
def fn_step04_vd_sellin_ap2_rolling():
    """
    Step 4: VD SellIn AP2 Rolling 로직 적용 및 S/In FCST(GI)_AP2(Rolling ADJ) Measure 생성
    """
    global output_dataframes, CurrentPartialWeek_normalized
    
    # Load data
    df_out_step03_4_sin_lv2_lv3 = output_dataframes['df_out_step03_4_sin_lv2_lv3']

    # Group data by GBRULE, Location.[Location], Item.[Item]
    grouped = df_out_step03_4_sin_lv2_lv3.groupby(['GBRULE', 'Location.[Location]', 'Item.[Item]'])

    # Result array
    return_array = []

    for (gbrule, location, item), group_df in grouped:
        # Split data into current and previous versions
        df_currnt = group_df[group_df['Version.[Version Name]'] == 'CurrentWorkingView']
        df_previous = group_df[group_df['Version.[Version Name]'] != 'CurrentWorkingView']

        # Find the current week row
        row_currentweek_in_current = df_currnt[df_currnt['Time.[Week]'] == CurrentPartialWeek_normalized]

        # Find the previous week row
        previous_week = common.gfn_add_week(CurrentPartialWeek_normalized, -1)
        row_lastweek_in_current = df_currnt[df_currnt['Time.[Week]'] == previous_week]
        row_lastweek_in_previous = df_previous[df_previous['Time.[Week]'] == previous_week]

        # Ensure all required rows exist
        if (
            not row_currentweek_in_current.empty and
            not row_lastweek_in_current.empty and
            not row_lastweek_in_previous.empty
        ):
            row_currentweek_in_current = row_currentweek_in_current.iloc[0]
            row_lastweek_in_current = row_lastweek_in_current.iloc[0]
            row_lastweek_in_previous = row_lastweek_in_previous.iloc[0]

            # Condition check
            if (
                row_lastweek_in_current['Time.[Planning Month]'] == row_currentweek_in_current['Time.[Planning Month]']
                and row_lastweek_in_previous['S/In FCST(GI)_AP2'] != row_lastweek_in_current['S/In FCST(GI)_AP2']
            ):
                # Create new row with adjusted rolling forecast
                new_row = {
                    'Version.[Version Name]': 'CurrentWorkingView',
                    'Sales Domain.[Ship To]': gbrule,
                    'Item.[Item]': row_currentweek_in_current['Item.[Item]'],
                    'Location.[Location]': row_currentweek_in_current['Location.[Location]'],
                    'Time.[Week]': row_currentweek_in_current['Time.[Week]'],
                    'S/In FCST(GI)_AP2(Rolling ADJ)': row_currentweek_in_current['S/In FCST(GI)_AP2']
                    - row_lastweek_in_previous['S/In FCST(GI)_AP2']
                    - row_lastweek_in_current['S/In FCST(GI)_AP2']
                }
                return_array.append(new_row)

    # Convert return array to DataFrame
    df_out_step04 = pd.DataFrame(return_array)

    return df_out_step04


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
        CurrentPartialWeek = args.get('CurrentPartialWeek')
        CurrentPartialWeek_normalized = normalize_week(CurrentPartialWeek)
        # MaxPartialWeek = ""
        # MaxPartialWeek_normalized = ""
        # # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
        Param_OUT_VERSION = args.get('Param_OUT_VERSION')
        Param_Exception_Flag = args.get('Param_Exception_Flag')
        input_dataframes = {}
        if is_local:
            set_input_output_folder(is_local, args)
            fn_process_in_df_mst()
        # initialize_max_week(is_local, args)
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
            'p_step_no': 100, 
            'p_step_desc': 'Step 1: Forecast Rule Table 2LV 중복 제거 전처리',
            'p_df_name': 'df_step01_remove_duplicate_forecast_rule'
        }
        df_step01_remove_duplicate_forecast_rule = fn_step01_remove_duplicate_forecast_rule(**dict_log)
        output_dataframes['df_step01_remove_duplicate_forecast_rule'] = df_step01_remove_duplicate_forecast_rule

        dict_log = {
            'p_step_no': 200, 
            'p_step_desc': 'Step 2: Enriched SIN data to SIN_DUMMY_SD',
            'p_df_name': 'df_step02_expand_sin_to_lv2_lv7'
        }
        df_step02_expand_sin_to_lv2_lv7 = fn_step02_expand_sin_to_lv2_lv7(**dict_log)
        output_dataframes['df_step02_expand_sin_to_lv2_lv7'] = df_step02_expand_sin_to_lv2_lv7

        dict_log = {
            'p_step_no': 301,
            'p_step_desc': 'Step3-1: Process LV2 branch',
            'p_df_name': 'df_out_step03_1_sin_lv2'
        }
        df_out_step03_1_sin_lv2 = fn_step03_1_sin_lv2(**dict_log)
        output_dataframes['df_out_step03_1_sin_lv2'] = df_out_step03_1_sin_lv2


        dict_log = {
            'p_step_no': 302,
            'p_step_desc': 'Step3-2: Process LV3 branch',
            'p_df_name': 'df_out_step03_2_sin_lv3'
        }
        df_out_step03_2_sin_lv3 = fn_step03_2_sin_lv3(**dict_log)
        output_dataframes['df_out_step03_2_sin_lv3'] = df_out_step03_2_sin_lv3


        dict_log = {
            'p_step_no': 303,
            'p_step_desc': 'Step3-3: Concatenate LV2 and LV3',
            'p_df_name': 'df_out_step03_3_sin_lv2_lv3'
        }
        df_out_step03_3_sin_lv2_lv3 = fn_step03_3_concat_lv2_lv3(**dict_log)
        output_dataframes['df_out_step03_3_sin_lv2_lv3'] = df_out_step03_3_sin_lv2_lv3


        dict_log = {
            'p_step_no': 304,
            'p_step_desc': 'Step3-4: Group by GBRULE and sum',
            'p_df_name': 'df_out_step03_4_sin_lv2_lv3'
        }
        df_out_step03_4_sin_lv2_lv3 = fn_step03_4_group_sum(**dict_log)
        output_dataframes['df_out_step03_4_sin_lv2_lv3'] = df_out_step03_4_sin_lv2_lv3


        dict_log = {
            'p_step_no': 400,
            'p_step_desc': 'Step 4: VD SellIn AP2 Rolling 로직 적용 및 S/In FCST(GI)_AP2(Rolling ADJ) Measure 생성',
            'p_df_name': 'df_out_step04'
        }
        df_out_step04 = fn_step04_vd_sellin_ap2_rolling(**dict_log)
        output_dataframes['df_out_step04'] = df_out_step04

        ################################################################################################################
        # 최종 Output 정리
        ################################################################################################################
        # dict_log = {
        #     'p_step_no': 900,
        #     'p_step_desc': '최종 Output 정리 - out_Demand',
        #     'p_df_name': 'out_Demand'
        # }
        # out_Demand = fn_output_formatter(df_out_step04_created, Param_OUT_VERSION, **dict_log)
        
        # # for df in output_dataframes:
        # #     fn_log_dataframe(output_dataframes[df], df)


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