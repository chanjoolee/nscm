"""
			* 프로그램명				
				PYDPMakeActualAndInventoryForecastLevel			
							
							
			* 목적				
				- FCST Rule(입력 레벨)을 고려한 실적 및 재고 생성 로직 개발 W-9 ~ W-1을 전체로 하면서, CE_Offline만 W-2로 적용			
				- Actual 			
				- S/In(GI) GC AP2 AP1			
				- S/In(BL) GC AP2 AP1			
				- S/Out GC AP2 AP1			
				- 구성시 고려점	- S/In 은 전주 실적까지 존재		
					- S/Out은 MX는 전주, CE는 전전주 실적까지 존재		
					- 처음에는 1년 Data 생성		
					- 이후에는 8주 Data에 대해서 매주 Batch로 작업		
							
				- 재고			
					- 사용 Measure		
						- Channel Inv	
						- (Sum) Channl Inv_Inc floor	
						- 재고는 6Lv(Account) 로 들어옴	
					- 생성 Measure		
						- (Simul). Channel Inv GC AP2 AP1	
						- (Simul). Channel Inv_Inc floor GC AP2 AP1	
					- 구성시 고려점		
						- 전주 재고 값을 가져와서 당주 재고 값을 계산한다.	
						- 처음에는 1년 Data 생성	
						- 이후에는 Data에 대해서 매주 Batch로 작업	
							
							
							
			* 변경이력				
				2025.04.08 전창민 최초 작성			
				2025.05.23 AMT 실적 9개 Measure 추가 생성			
				2025.07.07 AP0 Measure 추가			
				2025.07.16 기존 Measure 값 Update 로직 추가	

                25.04.29
                    1. o9 에서 REF Filter를 걸기 위해서는 Input Query 에서 Product Group을 같이 조회를 해야함.
                    -> Input QUery 에서 Product Group, Item GBM 추가 및 전처리에서 PG 붙이는 과정 제거
                    -> 전처리 과정에서 Item Merge 삭제



                25.04.30
                    1. Actual 가 없는 경우 ,FCST의 실적 부분을 덮어씌울 수 없음.
                    -> 실적 없는 주차에 대해서 0으로 채워주는 로직 필요
                    -> 전처리 과정의 N-2) 에서 진행

                    2. Time 조건 필요
                    S/Out 의 CE : -9 ~ -2
                    S/Out 의 MX : -8 ~ -1
                    나머지 8주차 update

                    --- 임의 개발 완료 ---

                25.05.23 
                    (S/In, S/out 의 AMT 값 추가)
                    1. Input1, Input2, Input3, Input4 에서 AMT_USD Measure 추가
                    2. Step1, Step2 전처리 과정 변경 ( Measure 추가)
                    3. Step5 내에서 Measure 추가되었고, 각각 AP1, AP2, GC 처리 필요
                    4. Step 6 에 최종 Measure 명칭 추가


                    (E-store 조건추가됨에 따라 S/Out CE Data 분기 필요


                25.05.26
                    * S/In GI, S/In BL, S/Out , Ch Inv, Ch Inv_Infloor * 4 = 20개의 Input을 받아서 null처리 필요
                    - Input 10 ~ input 28 추가
                    이후 생성한 Table 과 Merge하여, 값이 있는 경우 값 입력.
                    - Step 7)  Input 과 Output Data를 Merge하여 최종 Data 구성


                25.07.15
                    - Input Script Parameter 정리
                    - AP0 Input Data 추가
                    - Input Data time Filter 적용 
                    - Input 1,2 에서 Item GBM Column 삭제
                    - Input Data 3,4,5 합치고 이후 전처리 진행

                25.08.06
                    - 



		Script Parameter		
				
				
				
			(Input 1) Version	
				CWV_DP

"""

from re import X
from turtle import rt
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
import gc

########################################################################################################################
# Local 개발 시에 필요한 공통 변수 선언
########################################################################################################################
# o9에 저장된 instanceName
is_local = common.gfn_get_isLocal()
str_instance = 'PYDPMakeActualAndInventoryForecastLevelAll'
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

########################################################################################################################
# 컬럼상수
########################################################################################################################

COL_VERSION     = 'Version.[Version Name]'
COL_ITEM        = 'Item.[Item]'
COL_GBM         = 'Item.[Item GBM]'
COL_PG          = 'Item.[Product Group]'
COL_SHIP_TO     = 'Sales Domain.[Ship To]'
COL_STD1        = 'Sales Domain.[Sales Std1]'
COL_STD2        = 'Sales Domain.[Sales Std2]'
COL_STD3        = 'Sales Domain.[Sales Std3]' 
COL_STD4        = 'Sales Domain.[Sales Std4]'
COL_STD5        = 'Sales Domain.[Sales Std5]'
COL_STD6        = 'Sales Domain.[Sales Std6]'
COL_LOC         = 'Location.[Location]'
COL_TIME_PW     = 'Time.[Partial Week]'
COL_TIME_WK     = 'Time.[Week]'                     # <-- df_in_Sell_Out_Actual 의 S/Out Actual 원본에 존재 

COL_CURRENT_ROW_WEEK = 'WEEK_NUM'

# ── column constants ────────────────────────────────────────────────
COL_SIN_ACT_GI             = 'S/In Actual(GI)'
COL_SIN_ACT_GI_AMT         = 'S/In Actual(GI) AMT_USD'
COL_SIN_ACT_BL             = 'S/In Actual(BL)'
COL_SIN_ACT_BL_AMT         = 'S/In Actual(BL) AMT_USD'
COL_SOUT_ACT               = 'S/Out Actual'
COL_SOUT_ACT_AMT           = 'S/Out Actual AMT_USD'

# Forecast (GI/BL, AP1/2/GC/Local)
COL_SIN_FCST_GI_AP1        = 'S/In FCST(GI)_AP1'
COL_SIN_FCST_GI_AP1_AMT    = 'S/In FCST(GI) AMT_USD_AP1'
COL_SIN_FCST_GI_AP2        = 'S/In FCST(GI)_AP2'
COL_SIN_FCST_GI_AP2_AMT    = 'S/In FCST(GI) AMT_USD_AP2'
COL_SIN_FCST_GI_GC         = 'S/In FCST(GI)_GC'
COL_SIN_FCST_GI_GC_AMT     = 'S/In FCST(GI) AMT_USD_GC'
COL_SIN_FCST_GI_LOCAL      = 'S/In FCST(GI)_Local'
COL_SIN_FCST_GI_LOCAL_AMT  = 'S/In FCST(GI) AMT_USD_Local'
COL_SIN_FCST_BL_AP1        = 'S/In FCST(BL)_AP1'
COL_SIN_FCST_BL_AP1_AMT    = 'S/In FCST(BL) AMT_USD_AP1'
COL_SIN_FCST_BL_AP2        = 'S/In FCST(BL)_AP2'
COL_SIN_FCST_BL_AP2_AMT    = 'S/In FCST(BL) AMT_USD_AP2'
COL_SIN_FCST_BL_GC         = 'S/In FCST(BL)_GC'
COL_SIN_FCST_BL_GC_AMT     = 'S/In FCST(BL) AMT_USD_GC'
COL_SIN_FCST_BL_LOCAL      = 'S/In FCST(BL)_Local'
COL_SIN_FCST_BL_LOCAL_AMT  = 'S/In FCST(BL) AMT_USD_Local'
COL_SOUT_FCST_AP1          = 'S/Out FCST_AP1'
COL_SOUT_FCST_AP1_AMT      = 'S/Out FCST AMT_USD_AP1'
COL_SOUT_FCST_AP2          = 'S/Out FCST_AP2'
COL_SOUT_FCST_AP2_AMT      = 'S/Out FCST AMT_USD_AP2'
COL_SOUT_FCST_GC           = 'S/Out FCST_GC'
COL_SOUT_FCST_GC_AMT       = 'S/Out FCST AMT_USD_GC'
COL_SOUT_FCST_LOCAL        = 'S/Out FCST_Local'
COL_SOUT_FCST_LOCAL_AMT    = 'S/Out FCST AMT_USD_Local'

# Inventory
COL_CH_INV                 = 'Channel Inv'
COL_CH_INV_SUM_FLOOR       = '(Sum)Channel Inv_Inc Floor'
COL_CH_INV_AP1             = '(Simul)Ch Inv_AP1'
COL_CH_INV_AP2             = '(Simul)Ch Inv_AP2'
COL_CH_INV_GC              = '(Simul)Ch Inv_GC'
COL_CH_INV_LOCAL           = '(Simul)Ch Inv_Local'
COL_CH_INV_FLR_AP1         = '(Simul)Ch Inv_Inc Floor_AP1'
COL_CH_INV_FLR_AP2         = '(Simul)Ch Inv_Inc Floor_AP2'
COL_CH_INV_FLR_GC          = '(Simul)Ch Inv_Inc Floor_GC'
COL_CH_INV_FLR_LOCAL       = '(Simul)Ch Inv_Inc Floor_Local'

# Others
COL_GBRULE                 = 'GBRULE'
COL_FCST_RULE_GC           = 'FORECAST_RULE GC FCST'
COL_FCST_RULE_AP2          = 'FORECAST_RULE AP2 FCST'
COL_FCST_RULE_AP1          = 'FORECAST_RULE AP1 FCST'
COL_FCST_RULE_AP0          = 'FORECAST_RULE AP0 FCST'
COL_FCST_RULE_CUST         = 'FORECAST_RULE CUST FCST'
COL_FCST_RULE_VALID        = 'FORECAST_RULE ISVALID'

# -------------
COL_SIMUL_CH_INV_AP1             = '(Simul). Channel Inv AP1'
COL_SIMUL_CH_INV_AP2             = '(Simul). Channel Inv AP2'
COL_SIMUL_CH_INV_GC              = '(Simul). Channel Inv GC'
# -------------    
COL_SIMUL_CH_INV_FLR_AP1         = '(Simul)Ch Inv_Inc Floor_AP1'
COL_SIMUL_CH_INV_FLR_AP2         = '(Simul)Ch Inv_Inc Floor_AP2'
COL_SIMUL_CH_INV_FLR_GC          = '(Simul)Ch Inv_Inc Floor_GC'

# 25.09.10 추가 – P4W
COL_SIM_P4W_CH_INV              = '(Simul)P4W Ch Inv'
COL_SIM_P4W_CH_INV_FLOOR        = '(Simul)P4W Ch Inv_Inc Floor'


# ── dataframe name constants ────────────────────────────────────────
# input
STR_DF_IN_SI_ACT_GI              = 'df_in_Sell_In_Actual_GI'
STR_DF_IN_SI_ACT_BL              = 'df_in_Sell_In_Actual_BL'
STR_DF_IN_SO_ACT                 = 'df_in_Sell_Out_Actual'
STR_DF_IN_CH_INV                 = 'df_in_Channel_Inv'
STR_DF_IN_CH_INV_FLOOR           = 'df_in_Channel_Inv_Inc_Floor'
STR_DF_IN_SALES_DOMAIN_DIM       = 'df_in_Sales_Domain_Dimension'
STR_DF_IN_FCST_RULE              = 'df_in_Forecast_Rule'
STR_DF_IN_WEEK                   = 'df_in_Week'
STR_DF_IN_PARTIALWEEK            = 'df_in_PartialWeek'
# forecast inputs (GI / BL)
STR_DF_IN_SI_FCST_GI_AP1         = 'df_in_Sell_In_FCST_GI_AP1'
STR_DF_IN_SI_FCST_GI_AP2         = 'df_in_Sell_In_FCST_GI_AP2'
STR_DF_IN_SI_FCST_GI_GC          = 'df_in_Sell_In_FCST_GI_GC'
STR_DF_IN_SI_FCST_GI_LOCAL       = 'df_in_Sell_In_FCST_GI_Local'
STR_DF_IN_SI_FCST_BL_AP1         = 'df_in_Sell_In_FCST_BL_AP1'
STR_DF_IN_SI_FCST_BL_AP2         = 'df_in_Sell_In_FCST_BL_AP2'
STR_DF_IN_SI_FCST_BL_GC          = 'df_in_Sell_In_FCST_BL_GC'
STR_DF_IN_SI_FCST_BL_LOCAL       = 'df_in_Sell_In_FCST_BL_Local'
# forecast inputs (S/Out)
STR_DF_IN_SO_FCST_AP1            = 'df_in_Sell_Out_FCST_AP1'
STR_DF_IN_SO_FCST_AP2            = 'df_in_Sell_Out_FCST_AP2'
STR_DF_IN_SO_FCST_GC             = 'df_in_Sell_Out_FCST_GC'
STR_DF_IN_SO_FCST_LOCAL          = 'df_in_Sell_Out_FCST_Local'
# inventory sim inputs
STR_DF_IN_CH_INV_AP1             = 'df_in_Channel_Inv_AP1'
STR_DF_IN_CH_INV_AP2             = 'df_in_Channel_Inv_AP2'
STR_DF_IN_CH_INV_GC              = 'df_in_Channel_Inv_GC'
STR_DF_IN_CH_INV_LOCAL           = 'df_in_Channel_Inv_Local'
STR_DF_IN_CH_INV_FLR_AP1         = 'df_in_Channel_Inv_Floor_AP1'
STR_DF_IN_CH_INV_FLR_AP2         = 'df_in_Channel_Inv_Floor_AP2'
STR_DF_IN_CH_INV_FLR_GC          = 'df_in_Channel_Inv_Floor_GC'
STR_DF_IN_CH_INV_FLR_LOCAL       = 'df_in_Channel_Inv_Floor_Local'
STR_DF_IN_SALES_DOMAIN_ESTORE    = 'df_in_Sales_Domain_Estore'
# intermediate
STR_DF_STEP05_BUILD              = 'df_fn_step05_built_data'
STR_DF_STEP07_MERGED             = 'df_fn_step07_merged'
# output
STR_DF_OUT_SI_FCST_GI_AP1        = 'df_output_Sell_In_FCST_GI_AP1'
STR_DF_OUT_SI_FCST_GI_AP2        = 'df_output_Sell_In_FCST_GI_AP2'
STR_DF_OUT_SI_FCST_GI_GC         = 'df_output_Sell_In_FCST_GI_GC'
STR_DF_OUT_SI_FCST_GI_LOCAL      = 'df_output_Sell_In_FCST_GI_Local'
STR_DF_OUT_SI_FCST_BL_AP1        = 'df_output_Sell_In_FCST_BL_AP1'
STR_DF_OUT_SI_FCST_BL_AP2        = 'df_output_Sell_In_FCST_BL_AP2'
STR_DF_OUT_SI_FCST_BL_GC         = 'df_output_Sell_In_FCST_BL_GC'
STR_DF_OUT_SI_FCST_BL_LOCAL      = 'df_output_Sell_In_FCST_BL_Local'
STR_DF_OUT_SO_FCST_AP1           = 'df_output_Sell_Out_FCST_AP1'
STR_DF_OUT_SO_FCST_AP2           = 'df_output_Sell_Out_FCST_AP2'
STR_DF_OUT_SO_FCST_GC            = 'df_output_Sell_Out_FCST_GC'
STR_DF_OUT_SO_FCST_LOCAL         = 'df_output_Sell_Out_FCST_Local'
STR_DF_OUT_CH_INV_AP1            = 'df_output_Channel_Inv_AP1'
STR_DF_OUT_CH_INV_AP2            = 'df_output_Channel_Inv_AP2'
STR_DF_OUT_CH_INV_GC             = 'df_output_Channel_Inv_GC'
STR_DF_OUT_CH_INV_LOCAL          = 'df_output_Channel_Inv_Local'
STR_DF_OUT_CH_INV_FLR_AP1        = 'df_output_Channel_Inv_Floor_AP1'
STR_DF_OUT_CH_INV_FLR_AP2        = 'df_output_Channel_Inv_Floor_AP2'
STR_DF_OUT_CH_INV_FLR_GC         = 'df_output_Channel_Inv_Floor_GC'
STR_DF_OUT_CH_INV_FLR_LOCAL      = 'df_output_Channel_Inv_Floor_Local'

# 25.09.10 추가 Output
DF_OUT_SIM_P4W_CH_INV       = 'df_output_Channel_Inv_P4W'           # Measure: (Simul)P4W Ch Inv
DF_OUT_SIM_P4W_CH_INV_FLOOR = 'df_output_Channel_Inv_Floor_P4W'     # Measure: (Simul)P4W Ch Inv_Inc Floor

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

def fn_log_dataframe(df_p_source: pd.DataFrame, str_p_source_name: str,int_p_row_num: int = 20) -> None:
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
        logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=int_p_row_num)
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


################################################################################################################──────────
#  공통 타입 변환  (❌ `global` 사용 금지)
#  호출 측에서 `input_dataframes` 를 인자로 넘겨준다.
################################################################################################################──────────
def fn_prepare_input_types(dict_dfs: dict) -> None:
    """
    dict_dfs :  { <df_name> : pandas.DataFrame, ... }
    
    • object  → str → category                (숫자·문자 혼재 대비)
    • float/int → fillna(0) → int32           (값이 실수면 round 후 변환)    **주의** : dict 내부의 DataFrame 을 *제자리*에서 변환하므로 반환값은 없다.
    """
    if not dict_dfs:        # 빈 dict 방어
        return

    for df_name, df in dict_dfs.items():
        if df.empty:
            continue

        # 1) object 컬럼 : str → category
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df[col] = df[col].astype(str).astype("category")

        # 2) numeric 컬럼 : fillna → int32
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
            'df_in_Sell_In_Actual_GI.csv'				: STR_DF_IN_SI_ACT_GI				        ,	# S/In Actual (GI) + USD AMT
            'df_in_Sell_In_Actual_BL.csv'               : STR_DF_IN_SI_ACT_BL                       ,	# S/In Actual (BL) + USD AMT
            'df_in_Sell_Out_Actual.csv'                 : STR_DF_IN_SO_ACT                          ,	# S/Out Actual + USD AMT
            'df_in_Channel_Inv.csv'                     : STR_DF_IN_CH_INV                          ,	# Channel Inv (Qty)
            'df_in_Channel_Inv_Inc_Floor.csv'           : STR_DF_IN_CH_INV_FLOOR                    ,	# (Sum) Channel Inv Inc Floor
            'df_in_Sales_Domain_Dimension.csv'          : STR_DF_IN_SALES_DOMAIN_DIM                ,	# Ship‑to 7‑Level 기준표
            'df_in_Forecast_Rule.csv'                   : STR_DF_IN_FCST_RULE                       ,	# Forecast Rule (GC/AP2/AP1/AP0)
            'df_in_Week.csv'                            : STR_DF_IN_WEEK                            ,	# Week (‑9 ~ ‑1) 리스트
            'df_in_PartialWeek.csv'                     : STR_DF_IN_PARTIALWEEK                     ,	# Partial Week (‑9 ~ ‑1) 리스트
            'df_in_Sell_In_FCST_GI_AP1.csv'             : STR_DF_IN_SI_FCST_GI_AP1                  ,	# S/In FCST (GI) AP1 + USD
            'df_in_Sell_In_FCST_GI_AP2.csv'             : STR_DF_IN_SI_FCST_GI_AP2                  ,	# S/In FCST (GI) AP2 + USD
            'df_in_Sell_In_FCST_GI_GC.csv'              : STR_DF_IN_SI_FCST_GI_GC                   ,	# S/In FCST (GI) GC  + USD
            'df_in_Sell_In_FCST_BL_AP1.csv'             : STR_DF_IN_SI_FCST_BL_AP1                  ,	# S/In FCST (BL) AP1 + USD
            'df_in_Sell_In_FCST_BL_AP2.csv'             : STR_DF_IN_SI_FCST_BL_AP2                  ,	# S/In FCST (BL) AP2 + USD
            'df_in_Sell_In_FCST_BL_GC.csv'              : STR_DF_IN_SI_FCST_BL_GC                   ,	# S/In FCST (BL) GC  + USD
            'df_in_Sell_Out_FCST_AP1.csv'               : STR_DF_IN_SO_FCST_AP1                     ,	# S/Out FCST AP1 + USD
            'df_in_Sell_Out_FCST_AP2.csv'               : STR_DF_IN_SO_FCST_AP2                     ,	# S/Out FCST AP2 + USD
            'df_in_Sell_Out_FCST_GC.csv'                : STR_DF_IN_SO_FCST_GC                      ,	# S/Out FCST GC  + USD
            'df_in_Channel_Inv_AP1.csv'                 : STR_DF_IN_CH_INV_AP1                      ,	# (Simul) Ch Inv AP1
            'df_in_Channel_Inv_AP2.csv'                 : STR_DF_IN_CH_INV_AP2                      ,	# (Simul) Ch Inv AP2
            'df_in_Channel_Inv_GC.csv'                  : STR_DF_IN_CH_INV_GC                       ,	# (Simul) Ch Inv GC
            'df_in_Channel_Inv_Floor_AP1.csv'           : STR_DF_IN_CH_INV_FLR_AP1                  ,	# (Simul) Ch Inv Inc Floor AP1
            'df_in_Channel_Inv_Floor_AP2.csv'           : STR_DF_IN_CH_INV_FLR_AP2                  ,	# (Simul) Ch Inv Inc Floor AP2
            'df_in_Channel_Inv_Floor_GC.csv'            : STR_DF_IN_CH_INV_FLR_GC                   ,	# (Simul) Ch Inv Inc Floor GC
            'df_in_Sell_In_FCST_GI_Local.csv'           : STR_DF_IN_SI_FCST_GI_LOCAL                ,	# S/In FCST (GI) Local (AP0)
            'df_in_Sell_In_FCST_BL_Local.csv'           : STR_DF_IN_SI_FCST_BL_LOCAL                ,	# S/In FCST (BL) Local (AP0)
            'df_in_Sell_Out_FCST_Local.csv'             : STR_DF_IN_SO_FCST_LOCAL                   ,	# S/Out FCST Local (AP0)
            'df_in_Channel_Inv_Local.csv'               : STR_DF_IN_CH_INV_LOCAL                    ,	# (Simul) Ch Inv Local
            'df_in_Channel_Inv_Floor_Local.csv'         : STR_DF_IN_CH_INV_FLR_LOCAL                ,	# (Simul) Ch Inv Inc Floor Local
            'df_in_Sales_Domain_Estore.csv'             : STR_DF_IN_SALES_DOMAIN_ESTORE             	# eStore Ship‑to 목록
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
                # if file_name.startswith(keyword.split('.')[0]):
                if file_name == keyword.split('.')[0]:
                    input_dataframes[frame_name] = df
                    mapped = True
                    break
    else:
        input_dataframes[STR_DF_IN_SI_ACT_GI		  ]   = df_in_Sell_In_Actual_GI
        input_dataframes[STR_DF_IN_SI_ACT_BL          ]   = df_in_Sell_In_Actual_BL
        input_dataframes[STR_DF_IN_SO_ACT             ]   = df_in_Sell_Out_Actual
        input_dataframes[STR_DF_IN_CH_INV             ]   = df_in_Channel_Inv
        input_dataframes[STR_DF_IN_CH_INV_FLOOR       ]   = df_in_Channel_Inv_Inc_Floor
        input_dataframes[STR_DF_IN_SALES_DOMAIN_DIM   ]   = df_in_Sales_Domain_Dimension
        input_dataframes[STR_DF_IN_FCST_RULE          ]   = df_in_Forecast_Rule
        input_dataframes[STR_DF_IN_WEEK               ]   = df_in_Week
        input_dataframes[STR_DF_IN_PARTIALWEEK        ]   = df_in_PartialWeek
        input_dataframes[STR_DF_IN_SI_FCST_GI_AP1     ]   = df_in_Sell_In_FCST_GI_AP1
        input_dataframes[STR_DF_IN_SI_FCST_GI_AP2     ]   = df_in_Sell_In_FCST_GI_AP2
        input_dataframes[STR_DF_IN_SI_FCST_GI_GC      ]   = df_in_Sell_In_FCST_GI_GC
        input_dataframes[STR_DF_IN_SI_FCST_BL_AP1     ]   = df_in_Sell_In_FCST_BL_AP1
        input_dataframes[STR_DF_IN_SI_FCST_BL_AP2     ]   = df_in_Sell_In_FCST_BL_AP2
        input_dataframes[STR_DF_IN_SI_FCST_BL_GC      ]   = df_in_Sell_In_FCST_BL_GC
        input_dataframes[STR_DF_IN_SO_FCST_AP1        ]   = df_in_Sell_Out_FCST_AP1
        input_dataframes[STR_DF_IN_SO_FCST_AP2        ]   = df_in_Sell_Out_FCST_AP2
        input_dataframes[STR_DF_IN_SO_FCST_GC         ]   = df_in_Sell_Out_FCST_GC
        input_dataframes[STR_DF_IN_CH_INV_AP1         ]   = df_in_Channel_Inv_AP1
        input_dataframes[STR_DF_IN_CH_INV_AP2         ]   = df_in_Channel_Inv_AP2
        input_dataframes[STR_DF_IN_CH_INV_GC          ]   = df_in_Channel_Inv_GC
        input_dataframes[STR_DF_IN_CH_INV_FLR_AP1     ]   = df_in_Channel_Inv_Floor_AP1
        input_dataframes[STR_DF_IN_CH_INV_FLR_AP2     ]   = df_in_Channel_Inv_Floor_AP2
        input_dataframes[STR_DF_IN_CH_INV_FLR_GC      ]   = df_in_Channel_Inv_Floor_GC
        input_dataframes[STR_DF_IN_SI_FCST_GI_LOCAL   ]   = df_in_Sell_In_FCST_GI_Local
        input_dataframes[STR_DF_IN_SI_FCST_BL_LOCAL   ]   = df_in_Sell_In_FCST_BL_Local
        input_dataframes[STR_DF_IN_SO_FCST_LOCAL      ]   = df_in_Sell_Out_FCST_Local
        input_dataframes[STR_DF_IN_CH_INV_LOCAL       ]   = df_in_Channel_Inv_Local
        input_dataframes[STR_DF_IN_CH_INV_FLR_LOCAL   ]   = df_in_Channel_Inv_Floor_Local
        input_dataframes[STR_DF_IN_SALES_DOMAIN_ESTORE]   = df_in_Sales_Domain_Estore



    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ① category 로 캐스팅할 컬럼
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CATEGORY_CAST_MAP = {
        STR_DF_IN_SI_ACT_GI        : [COL_SHIP_TO, COL_PG,  COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SI_ACT_BL        : [COL_SHIP_TO, COL_PG,  COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SO_ACT           : [COL_SHIP_TO, COL_PG,  COL_ITEM, COL_LOC, COL_TIME_WK],
        STR_DF_IN_CH_INV           : [COL_SHIP_TO,           COL_ITEM,           COL_TIME_WK],
        STR_DF_IN_CH_INV_FLOOR     : [COL_SHIP_TO,           COL_ITEM,           COL_TIME_WK],
        STR_DF_IN_SALES_DOMAIN_DIM : [COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6, COL_SHIP_TO],
        STR_DF_IN_FCST_RULE        : [COL_SHIP_TO, COL_PG],
        STR_DF_IN_WEEK             : [COL_TIME_WK],
        STR_DF_IN_PARTIALWEEK      : [COL_TIME_PW],

        # ── Forecast Sell-In (GI / BL) ────────────────────────────
        STR_DF_IN_SI_FCST_GI_AP1   : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SI_FCST_GI_AP2   : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SI_FCST_GI_GC    : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SI_FCST_GI_LOCAL : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SI_FCST_BL_AP1   : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SI_FCST_BL_AP2   : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SI_FCST_BL_GC    : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SI_FCST_BL_LOCAL : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],

        # ── Forecast Sell-Out ────────────────────────────────────
        STR_DF_IN_SO_FCST_AP1      : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SO_FCST_AP2      : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SO_FCST_GC       : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
        STR_DF_IN_SO_FCST_LOCAL    : [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],

        # ── Inventory simulation inputs ───────────────────────────
        STR_DF_IN_CH_INV_AP1       : [COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
        STR_DF_IN_CH_INV_AP2       : [COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
        STR_DF_IN_CH_INV_GC        : [COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
        STR_DF_IN_CH_INV_LOCAL     : [COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
        STR_DF_IN_CH_INV_FLR_AP1   : [COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
        STR_DF_IN_CH_INV_FLR_AP2   : [COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
        STR_DF_IN_CH_INV_FLR_GC    : [COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
        STR_DF_IN_CH_INV_FLR_LOCAL : [COL_SHIP_TO, COL_ITEM, COL_TIME_WK],

        # ── eStore 목록 ───────────────────────────────────────────
        STR_DF_IN_SALES_DOMAIN_ESTORE : [COL_SHIP_TO],
    }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ② int32 로 캐스팅할 컬럼 (모두 ‘값’ 계열)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    INT32_CAST_MAP = {
        STR_DF_IN_FCST_RULE        : [COL_FCST_RULE_GC, COL_FCST_RULE_AP2,COL_FCST_RULE_AP1,COL_FCST_RULE_AP0],
        STR_DF_IN_SI_ACT_GI        : [COL_SIN_ACT_GI,         COL_SIN_ACT_GI_AMT],
        STR_DF_IN_SI_ACT_BL        : [COL_SIN_ACT_BL,         COL_SIN_ACT_BL_AMT],
        STR_DF_IN_SO_ACT           : [COL_SOUT_ACT,           COL_SOUT_ACT_AMT],
        STR_DF_IN_CH_INV           : [COL_CH_INV],
        STR_DF_IN_CH_INV_FLOOR     : [COL_CH_INV_SUM_FLOOR],

        # Forecast Sell-In (GI / BL)
        STR_DF_IN_SI_FCST_GI_AP1   : [COL_SIN_FCST_GI_AP1,    COL_SIN_FCST_GI_AP1_AMT],
        STR_DF_IN_SI_FCST_GI_AP2   : [COL_SIN_FCST_GI_AP2,    COL_SIN_FCST_GI_AP2_AMT],
        STR_DF_IN_SI_FCST_GI_GC    : [COL_SIN_FCST_GI_GC,     COL_SIN_FCST_GI_GC_AMT],
        STR_DF_IN_SI_FCST_GI_LOCAL : [COL_SIN_FCST_GI_LOCAL,  COL_SIN_FCST_GI_LOCAL_AMT],
        STR_DF_IN_SI_FCST_BL_AP1   : [COL_SIN_FCST_BL_AP1,    COL_SIN_FCST_BL_AP1_AMT],
        STR_DF_IN_SI_FCST_BL_AP2   : [COL_SIN_FCST_BL_AP2,    COL_SIN_FCST_BL_AP2_AMT],
        STR_DF_IN_SI_FCST_BL_GC    : [COL_SIN_FCST_BL_GC,     COL_SIN_FCST_BL_GC_AMT],
        STR_DF_IN_SI_FCST_BL_LOCAL : [COL_SIN_FCST_BL_LOCAL,  COL_SIN_FCST_BL_LOCAL_AMT],

        # Forecast Sell-Out
        STR_DF_IN_SO_FCST_AP1      : [COL_SOUT_FCST_AP1,      COL_SOUT_FCST_AP1_AMT],
        STR_DF_IN_SO_FCST_AP2      : [COL_SOUT_FCST_AP2,      COL_SOUT_FCST_AP2_AMT],
        STR_DF_IN_SO_FCST_GC       : [COL_SOUT_FCST_GC,       COL_SOUT_FCST_GC_AMT],
        STR_DF_IN_SO_FCST_LOCAL    : [COL_SOUT_FCST_LOCAL,    COL_SOUT_FCST_LOCAL_AMT],

        # Inventory simulation inputs
        STR_DF_IN_CH_INV_AP1       : [COL_CH_INV_AP1],
        STR_DF_IN_CH_INV_AP2       : [COL_CH_INV_AP2],
        STR_DF_IN_CH_INV_GC        : [COL_CH_INV_GC],
        STR_DF_IN_CH_INV_LOCAL     : [COL_CH_INV_LOCAL],
        STR_DF_IN_CH_INV_FLR_AP1   : [COL_CH_INV_FLR_AP1],
        STR_DF_IN_CH_INV_FLR_AP2   : [COL_CH_INV_FLR_AP2],
        STR_DF_IN_CH_INV_FLR_GC    : [COL_CH_INV_FLR_GC],
        STR_DF_IN_CH_INV_FLR_LOCAL : [COL_CH_INV_FLR_LOCAL],
    }

    for df_name in CATEGORY_CAST_MAP:
        df = input_dataframes.get(df_name)
        if df is not None:
            for col in CATEGORY_CAST_MAP[df_name]:
                df[col] = df[col].astype('object')

    for df_name in INT32_CAST_MAP:
        df = input_dataframes.get(df_name)
        if df is not None:
            for col in INT32_CAST_MAP[df_name]:
                df[col].fillna(0, inplace=True)
                df[col] = df[col].astype('int32')

   
    fn_prepare_input_types(input_dataframes)

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

################################################################################################################
#################### Start Step Functions  ##########
################################################################################################################
  



########################################################################################################################
# Step 01 : S/In Actual(GI) · S/In Actual(BL) 전처리
########################################################################################################################
@_decoration_
def fn_step01_preprocess_sell_in_actual(
        df_si_act_gi: pd.DataFrame,
        df_si_act_bl: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 01) S/In Actual(GI·BL) Pre-Processing
    ----------------------------------------------------------
    • Version 컬럼 삭제              : COL_VERSION
    • 타입 변환 없음                 : 선행 단계에서 완료
    • 반환                           : (GI DataFrame, BL DataFrame)
    """

    # ── ① GI 처리 ────────────────────────────────────────────
    df_gi = df_si_act_gi.copy()                 # 원본 보존
    if COL_VERSION in df_gi.columns:            # 안전 방어
        df_gi.drop(columns=[COL_VERSION], inplace=True)

    # ── ② BL 처리 ────────────────────────────────────────────
    df_bl = df_si_act_bl.copy()
    if COL_VERSION in df_bl.columns:
        df_bl.drop(columns=[COL_VERSION], inplace=True)

    # ── ③ 메모리 즉시 해제 ──────────────────────────────────
    del df_si_act_gi, df_si_act_bl
    gc.collect()

    return df_gi, df_bl


########################################################################################################################
# Step 02 : S/Out Actual · S/Out FCST 전처리
########################################################################################################################
@_decoration_
def fn_step02_preprocess_sell_out(
        # ── inputs ───────────────────────────────────────────────────────────────────────────────────────────────────
        df_so_act         : pd.DataFrame,            # input_dataframes[STR_DF_IN_SO_ACT]
        df_week           : pd.DataFrame,            # input_dataframes[STR_DF_IN_WEEK]
        df_pw             : pd.DataFrame,            # input_dataframes[STR_DF_IN_PARTIALWEEK]
        df_estore         : pd.DataFrame,            # input_dataframes[STR_DF_IN_SALES_DOMAIN_ESTORE]
        df_so_fcst_ap1    : pd.DataFrame,            # input_dataframes[STR_DF_IN_SO_FCST_AP1]
        df_so_fcst_ap2    : pd.DataFrame,            # input_dataframes[STR_DF_IN_SO_FCST_AP2]
        df_so_fcst_gc     : pd.DataFrame,            # input_dataframes[STR_DF_IN_SO_FCST_GC]
        df_so_fcst_local  : pd.DataFrame             # input_dataframes[STR_DF_IN_SO_FCST_LOCAL]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Step 02) S/Out Actual & FCST Pre-Processing
    ----------------------------------------------------------
    ① S/Out Actual 전처리
       • COL_VERSION 제거
       • 조건에 따라 Week –1 or –9 행 drop
       • Time.[Week] → Time.[Partial Week]  (+ 'A')
       • COL_GBM(=Item.[Item GBM]) 제거
    ② S/Out FCST(AP1, AP2, GC, Local) 전처리
       • COL_VERSION & COL_GBM 제거
       • 조건에 따라 Partial Week –1 or –9 행 drop
    ③ 모든 연산은 vectorized, 불필요 객체 즉시 gc
    """    
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # 0) 사전 정보 : Week·Partial-Week 기준점 계산
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    week_list   = df_week[COL_TIME_WK].astype(str).tolist()              # ex) ['202449', …, '202505']
    pw_list     = df_pw[COL_TIME_PW].astype(str).tolist()                # ex) ['202449A', …, '202505A']
    w_m9, w_m1  = week_list[0], week_list[-1]
    pw_m9, pw_m1 = pw_list[0], pw_list[-1]
    w_m9_base   = w_m9                                                 # ex) '202449'
    pw_m9_base  = pw_m9[:6]                                            # ex) '202449'
    pw_m1_base  = pw_m1[:6]                                            # ex) '202505'
    estore_set  = set(df_estore[COL_SHIP_TO].astype(str))



    # ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Helper : 그룹별(Ship-to·Item·Loc) W-1 / W-9 Drop-mask 생성
    # ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def _build_drop_mask(
        df: pd.DataFrame,
        col_week: str,
        last_w: str,
        first_w_base: str,
        *,
        is_partial: bool = False,
    ) -> pd.Series:
        """
        그룹(Ship-to, Item, Location) 단위로
        • VD / SHA & 非-eStore :  W-1 존재 → W-1 삭제, 없으면 → W-9 삭제
        • 그 외                 :  항상 W-9 삭제
        반환값 : Boolean Series (drop 대상 → True)
        """
        # ── NEW : df 가 비면 바로 종료 ───────────────────────────────────────────
        if df.empty:
            # df.index 와 동일한 인덱스를 가진 길이 0 Series
            return pd.Series([], dtype=bool, index=df.index)

        # ── 전처리 ───────────────────────────────────────────────────────────────
        key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOC]
        gbm_flag      = df[COL_GBM].isin(['VD', 'SHA'])
        estore_flag   = df[COL_SHIP_TO].astype(str).isin(estore_set)
        cond_special  = gbm_flag & ~estore_flag              # VD/SHA & 非-eStore
        week_vals     = df[col_week].astype(str)    # Partial-Week ⇒ 앞 6자리(Week)만 비교
        week_key = week_vals.str[:6] if is_partial else week_vals

        # ── 그룹 키 생성 ────────────────────────────────────────────────────────
        grp_key = (
            df[key_cols]
            .astype(str)
            .agg('‖'.join, axis=1)           # 빠른 문자열 결합 구분자
        )

        # 그룹별 week summary : min / max / W-1 존재 여부 ------------------------
        grp_tbl = (
            pd.DataFrame({
                'grp_key' : grp_key,
                'week'    : week_key,
                'is_last' : week_key.eq(last_w),
            })
            .groupby('grp_key')
            .agg(
                min_w      = ('week', 'min'),
                has_last_w = ('is_last', 'any')
            )
        )

        # grp_key → 속성 map (vectorize용) --------------------------------------
        grp_min_map      = grp_tbl['min_w'].to_dict()
        grp_has_last_map = grp_tbl['has_last_w'].to_dict()

        # 벡터라이즈 조회
        min_w       = grp_key.map(grp_min_map)
        has_last_w  = grp_key.map(grp_has_last_map)

        # ── 삭제 조건 ────────────────────────────────────────────────────────────
        # ① VD/SHA & 非-eStore ---------------------------------------------------
        #    • W-1 존재 → W-1 삭제
        #    • W-1 없음 → 그룹의 min_w 삭제
        drop_special_last  =  cond_special & has_last_w & week_key.eq(last_w)
        # drop_special_first =  cond_special & ~has_last_w & week_key.eq(min_w) & (first_w_base != '202301')
        drop_special_first = (cond_special & ~has_last_w
                      & week_key.eq(first_w_base)          # ★
                      & (first_w_base != '202301'))

        # ② 그 외 그룹  →  항상 min_w 삭제  (first_w_base == '202301' 은 제외) ---
        drop_others_first  = (
            ~cond_special
            & week_key.eq(min_w)
            & (first_w_base != '202301')
        )

        # 최종 mask --------------------------------------------------------------
        # return drop_special_last | drop_special_first | drop_others_first
        return drop_special_last | drop_special_first
        
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    # 1) ───────── S/Out Actual 전처리 ────────────────────────────────────────────────────────────────────────────
    df_act = df_so_act.copy()

    # ① Version 컬럼 삭제
    if COL_VERSION in df_act.columns:
        df_act.drop(columns=[COL_VERSION], inplace=True)

    # for debug
    # fn_log_dataframe(df_act,f'step_02_df_act_beforeFilter',len(df_act))

    # ② 행 Drop 규칙 적용 (Week)
    mask_drop_act = _build_drop_mask(
        df_act, COL_TIME_WK, w_m1, w_m9_base, is_partial=False
    )
    df_act = df_act.loc[~mask_drop_act].copy()
    # for debug
    # fn_log_dataframe(df_act,f'step_02_df_act_afterFilter',len(df_act))

    # ③ Week → Partial Week (+ 'A') 변환
    df_act[COL_TIME_PW] = df_act[COL_TIME_WK].astype(str).str.cat(['A']*len(df_act))
    df_act.drop(columns=[COL_TIME_WK], inplace=True)

    # ④ Item.[Item GBM] 컬럼 삭제
    if COL_GBM in df_act.columns:
        df_act.drop(columns=[COL_GBM], inplace=True)

    # ⑤ 메모리 최적화
    gc.collect()

    # 2) ───────── S/Out FCST 전처리 공통 루틴 ────────────────────────────────────────────────────────────────────
    def _process_fcst(df_fcst: pd.DataFrame) -> pd.DataFrame:
        """AP1·AP2·GC·Local 4 개 DataFrame 에 동일 로직 적용"""
        df = df_fcst.copy()

        # Version 삭제
        # if COL_VERSION in df.columns:
        #     df.drop(columns=[COL_VERSION], inplace=True)

        # 행 Drop 규칙 (Partial-Week)
        mask_drop = _build_drop_mask(
            df, COL_TIME_PW, pw_m1, pw_m9_base, is_partial=True
        )
        df = df.loc[~mask_drop].copy()

        # Item.[Item GBM] 삭제
        if COL_GBM in df.columns:
            df.drop(columns=[COL_GBM], inplace=True)

        return df

    
    df_fcst_ap1   = _process_fcst(df_so_fcst_ap1)
    df_fcst_ap2   = _process_fcst(df_so_fcst_ap2)
    df_fcst_gc    = _process_fcst(df_so_fcst_gc)
    df_fcst_local = _process_fcst(df_so_fcst_local)

    # ── 불필요 객체 해제 ─────────────────────────────────────────────────────────────
    # del (df_so_act, df_week, df_pw, df_estore,
    #      df_so_fcst_ap1, df_so_fcst_ap2, df_so_fcst_gc, df_so_fcst_local
    #     )
    gc.collect()

    # 반환 : (Actual, AP1, AP2, GC, Local)
    return df_act, df_fcst_ap1, df_fcst_ap2, df_fcst_gc, df_fcst_local    


########################################################################################################################
# Step 03 : Channel Inv 전처리
########################################################################################################################
@_decoration_
def fn_step03_preprocess_channel_inv(
        df_ch_inv: pd.DataFrame            # input_dataframes[STR_DF_IN_CH_INV]
) -> pd.DataFrame:
    """
    Step 03) Channel Inv Pre-Processing
    ----------------------------------------------------------
    • COL_VERSION 삭제
    • 타입 변환 없음   (선행 단계 완료)
    • 반환            : 정제된 Channel Inv DataFrame
    """
    # ── ① 전처리 ────────────────────────────────────────────────────────────
    df = df_ch_inv.copy()                    # 원본 보존    # Version 컬럼 제거 (안전 방어)
    if COL_VERSION in df.columns:
        df.drop(columns=[COL_VERSION], inplace=True)

    # ── ② 메모리 즉시 해제 ─────────────────────────────────────────────────
    del df_ch_inv
    gc.collect()

    # ── ③ 결과 반환 ───────────────────────────────────────────────────────
    return df

########################################################################################################################
# Step 04 : (Sum) Channel Inv_Inc Floor 전처리
########################################################################################################################
@_decoration_
def fn_step04_preprocess_channel_inv_floor(
        df_ch_inv_floor: pd.DataFrame      # input_dataframes[STR_DF_IN_CH_INV_FLOOR]
) -> pd.DataFrame:
    """
    Step 04) (Sum) Channel Inv_Inc Floor Pre-Processing
    ----------------------------------------------------------
    • COL_VERSION 삭제
    • 타입 변환 없음   (선행 단계에서 완료)
    • 반환            : 정제된 (Sum)Channel Inv_Inc Floor DataFrame
    """
    # ── ① Version 컬럼 제거 ────────────────────────────────────────────────
    df = df_ch_inv_floor.copy()            # 원본 보존
    if COL_VERSION in df.columns:          # 안전 방어
        df.drop(columns=[COL_VERSION], inplace=True)    # ── ② 메모리 즉시 해제 ────────────────────────────────────────────────
    del df_ch_inv_floor
    gc.collect()

    # ── ③ 결과 반환 ───────────────────────────────────────────────────────
    return df

########################################################################################################################
# Step 05 : Forecast-Rule 기준  Actual / FCST / Inventory Level Build (vectorised, +AP0)
########################################################################################################################
@_decoration_
def fn_step05_make_actual_and_inv_fcst_level(
        # ── inputs ───────────────────────────────────────────────────────────────────────────────────────────────────
        df_si_gi : pd.DataFrame,   # GI Actual
        df_si_bl : pd.DataFrame,   # BL Actual
        df_so_act: pd.DataFrame,   # S/Out Actual
        df_ci    : pd.DataFrame,   # Channel Inv
        df_cif   : pd.DataFrame,   # (Sum) Ch Inv_Inc Floor
        df_rule  : pd.DataFrame,   # Forecast Rule
        df_dim   : pd.DataFrame    # Sales-Domain Dimension
) -> dict[str, pd.DataFrame]:
    """
    Step 05)  Forecast-Rule Level 별  Actual / FCST / Inventory 데이터 생성
    ───────────────────────────────────────────────────────────────────────────
    • **GC / AP2 / AP1 / Local(AP0)**  4 tag × { S/In(GI,BL) · S/Out · Inv · Inv_Floor }
    • 수량(Qty) + 금액(AMT)  →  SUM 집계
    • 100 % vectorised   (Python loop ≒ 단  ❬6 회❭ for-level 루프만)
    • 반환 :  { '<OUTPUT_KEY>' : DataFrame, … } 형태의 dict
    """
    if df_rule.empty or df_dim.empty:
        logger.Note('[Step05] Forecast Rule 또는 Sales-Domain Dimension 이 비어 있어 단계를 건너뜁니다.',
                    LOG_LEVEL.warning())
        return {}    

    # ─────────────────────────────────────────────────────────────────────────
    # 0)  Ship-to 계층 & Forecast-Rule look-up (table → dict)
    # ─────────────────────────────────────────────────────────────────────────
    STD_COLS = [COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6]   # Std1 ~ Std6 ⇒ Lv-2 ~ Lv-7
    # 0-1) Ship-to 코드가 **가장 처음(=가장 상위 Std) 등장한 level** 을 저장
    #      ex) code가 Std2, Std3 둘 다 있더라도  ➜  level = 3(Std2)
    ship_level: dict[str, int] = {}
    for lv, col in enumerate(STD_COLS, start=2):          # Std1 ⇒ Lv-2 …
        for code in df_dim[col].dropna().unique():
            ship_level.setdefault(str(code), lv)          # set once, 이후 덮어쓰지 않음

    # 0-2) Ship-to → 각 Std 조상 매핑  (기존 그대로)        
    dim_idx  = df_dim.set_index(COL_SHIP_TO)
    LV_MAP   = {lv: dim_idx[f'Sales Domain.[Sales Std{lv}]'].to_dict()
                for lv in range(1, 7)}                                        # lv=1 ⇒ Std1(=Lv-2) …

    parent_of = lambda arr, lv: np.vectorize(LV_MAP[lv].get, otypes=[object])(arr)

    # <───────────────────  AP0 추가 관련 핵심 변경 ①  ───────────────────>
    # 0-3) Forecast-Rule dict : (PG, ShipTo) → (gc_lv, ap2_lv, ap1_lv, ap0_lv)
    for col in (COL_FCST_RULE_GC,
                COL_FCST_RULE_AP2,
                COL_FCST_RULE_AP1,
                COL_FCST_RULE_AP0):                                           # ← AP0 컬럼 포함
        df_rule[col] = df_rule[col].fillna(0).astype('int32')

    RULE = {(pg, ship): (gc, ap2, ap1, ap0)
            for pg, ship, gc, ap2, ap1, ap0 in zip(df_rule[COL_PG],
                                                   df_rule[COL_SHIP_TO],
                                                   df_rule[COL_FCST_RULE_GC],
                                                   df_rule[COL_FCST_RULE_AP2],
                                                   df_rule[COL_FCST_RULE_AP1],
                                                   df_rule[COL_FCST_RULE_AP0])}
    # >─────────────────────────────────────────────────────────────────────<

    # ─────────────────────────────────────────────────────────────────────────
    # 1)  상위 Ship-to 중복 제거 (벡터라이즈)
    # ─────────────────────────────────────────────────────────────────────────
    ancestors = (
        df_dim[STD_COLS].fillna('')
              .apply(lambda r: {c for c in r if c}, axis=1)
              .to_dict())     # { leaf ShipTo : {Std1…Std6} }

    def remove_upper_shipto(df: pd.DataFrame,
                            *,
                            loc_flag: bool,
                            time_col: str) -> pd.DataFrame:
        """그룹(Item,[Loc],Time) 내에서 ‘하위 Ship-to’만 남기고 상위 Std 삭제"""
        if df.empty:
            return df

        grp_cols = [COL_ITEM]
        if loc_flag:
            grp_cols.append(COL_LOC)
        grp_cols.append(time_col)

        def _filter(block: pd.DataFrame) -> pd.DataFrame:
            ships = block[COL_SHIP_TO].to_numpy()
            union_anc = set().union(*(ancestors.get(s, set()) for s in ships))
            return block.loc[~block[COL_SHIP_TO].isin(union_anc)]

        return (block if (n := len(df)) == 0 else
                df.groupby(grp_cols, group_keys=False, sort=False)
                  .apply(_filter)
                  .reset_index(drop=True))

    # ─────────────────────────────────────────────────────────────────────────
    # 2) Core Builder  (vectorised, +Local tag)
    # ─────────────────────────────────────────────────────────────────────────
    def build_level(df_src     : pd.DataFrame,
                    qty_col    : str,
                    amt_col    : str | None,
                    loc_flag   : bool,
                    time_col   : str,
                    out_prefix : str) -> dict[str, pd.DataFrame]:
        
        # 컬럼명 변환
        def _rename(base: str, _tag: str) -> str:
            if base == COL_CH_INV:
                return f'(Simul)Ch Inv_{_tag}'
            if base == COL_CH_INV_SUM_FLOOR:
                return f'(Simul)Ch Inv_Inc Floor_{_tag}'
            if 'Actual' in base:
                return base.replace('Actual', 'FCST') + f'_{_tag}'
            return f'{base}_{_tag}'
        
        # if df_src.empty:
        #     # # 아래 코드를 응용하여 컬럼을 추가하여 빈 DF 를 return 하고 싶다.
        #     # g_cols = [COL_SHIP_TO] + ([COL_LOC] if loc_flag else []) + [COL_ITEM, time_col]
        #     # meas   = [_rename(qty_col, tag)] + ([_rename(amt_col, tag)] if amt_col else [])
        #     # empty  = pd.DataFrame(columns=g_cols + meas)
        #     # frames[tag] = empty
        #     # fn_log_dataframe(empty,f'df_step05_empty_{tag}')
        #     return {f'df_output_{out_prefix}_{t}': pd.DataFrame()        # 4 tag 모두 빈 DF
        #         for t in ('GC', 'AP2', 'AP1', 'Local')}

        # for debug
        # if ('Sell_Out_FCST' == out_prefix) : 
        #     fn_log_dataframe(df_src,f'step_05_{out_prefix}_originData',len(df_src))

        # ── vector slice ───────────────────────────────────────────────────
        ship = df_src[COL_SHIP_TO].astype(str).to_numpy()
        pg   = df_src[COL_PG]      .to_numpy()
        itm  = df_src[COL_ITEM]    .to_numpy()
        qty  = df_src[qty_col]     .to_numpy()
        time = df_src[time_col].astype(str).to_numpy()
        loc  = df_src[COL_LOC].to_numpy() if loc_flag else None
        amt  = df_src[amt_col].to_numpy() if amt_col else None

        # <───────────────────  AP0 추가 관련 핵심 변경 ②  ───────────────────>
        TAGS      = ('GC', 'AP2', 'AP1', 'Local')               # ← Local 태그 추가
        tag_lvmat = {tag: np.zeros_like(qty, dtype='int8') for tag in TAGS}
        # >────────────────────────────────────────────────────────────────────<

        # Std1~Std6 (6 회) 루프만 사용 – 데이터 행 수(n) 무관
        for lv in range(1, 7):
            parent = parent_of(ship, lv)          # ⚡︎ 벡터라이즈. df_in_Sales_Domain_Dimension 에 정의된 ShipTo
            mask   = parent != None
            if not mask.any():
                continue

            # for debug
            # if ~mask.any():
            #     fn_log_dataframe(df_src.loc[~mask],f'step_05_{lv}_haveNotParent')

            # RULE 벡터 look-up
            rule_vec = np.array(
                [RULE.get((pg[i], parent[i]), (0, 0, 0, 0))   # ← (gc,ap2,ap1,ap0)
                 for i in np.flatnonzero(mask)],
                dtype='int8')

            gc_lv, ap2_lv, ap1_lv, ap0_lv = rule_vec.T

            idx = np.flatnonzero(mask)
            tag_lvmat['GC']   [idx] = np.where(gc_lv  != 0, gc_lv,  tag_lvmat['GC']   [idx])
            tag_lvmat['AP2']  [idx] = np.where(ap2_lv != 0, ap2_lv, tag_lvmat['AP2']  [idx])
            tag_lvmat['AP1']  [idx] = np.where(ap1_lv != 0, ap1_lv, tag_lvmat['AP1']  [idx])
            tag_lvmat['Local'][idx] = np.where(ap0_lv != 0, ap0_lv, tag_lvmat['Local'][idx])
        # >────────────────────────────────────────────────────────────────────<
 
        # Ship-to level 배열 (vectorised look-up)
        ship_lv_arr = np.fromiter((ship_level.get(s, 99) for s in ship), dtype='int8')

        # ── Ship-to 변환 + GroupBy SUM ───────────────────────────────────────
        frames: dict[str, pd.DataFrame] = {}
        for tag in TAGS:
            lv_arr = tag_lvmat[tag]

            # valid  = (lv_arr >= 2) & (lv_arr <= 7)
            # ★ “Forecast Rule level ≤ 본 Ship-to level” 필터 추가
            valid = (lv_arr >= 2) & (lv_arr <= 7) & (lv_arr <= ship_lv_arr)

            # # for debug
            # if (out_prefix == 'Sell_Out_FCST') & (tag == 'GC'):
            #     logger.debug(f'[lv_arr] {lv_arr} ')
            #     logger.debug(f'[valid] {valid}')

            # ── Ship-to 매핑이 전혀 없을 경우 : 빈 DF “껍데기”를 넣어 둠 ────
            if not valid.any():
                # 컬럼 스케치 (join key + measure)
                g_cols = [COL_SHIP_TO] + ([COL_LOC] if loc_flag else []) + [COL_ITEM, time_col]
                meas   = [_rename(qty_col, tag)] + ([_rename(amt_col, tag)] if amt_col else [])
                empty  = pd.DataFrame(columns=g_cols + meas)
                frames[tag] = empty
                # fn_log_dataframe(empty,f'df_step05_empty_{tag}')
                continue
            # ───────────────────────────────────────────────────────────────

            # Ship-to 변환 (ancestor 코드 찾기)
            tgt_ship = np.fromiter(
                (LV_MAP[l-1].get(s) if 2 <= l <= 7 else None
                 for s, l in zip(ship[valid], lv_arr[valid])),
                dtype=object)

            ok = tgt_ship != None
            if not ok.any():                        
                # 변환불가 ⇒ 빈 DF 삽입
                g_cols = [COL_SHIP_TO] + ([COL_LOC] if loc_flag else []) + [COL_ITEM, time_col]
                meas   = [_rename(qty_col, tag)] + ([_rename(amt_col, tag)] if amt_col else [])
                empty  = pd.DataFrame(columns=g_cols + meas)
                frames[tag] = empty
                # fn_log_dataframe(empty,f'df_step05_empty_{tag}')
                continue

            rows = {
                COL_SHIP_TO   : tgt_ship[ok],
                COL_ITEM      : itm [valid][ok],
                time_col      : time[valid][ok],
                qty_col       : qty [valid][ok],
            }
            if loc_flag:
                rows[COL_LOC] = loc[valid][ok]
            if amt_col:
                rows[amt_col] = amt[valid][ok]

            df_tag = pd.DataFrame(rows)

            # for debug
            # if (out_prefix == 'Sell_Out_FCST') & (tag == 'GC'):
            #     fn_log_dataframe(df_tag,f'step_05_{out_prefix}_{tag}_beforeGroup',len(df_tag))

            # Group-by SUM
            g_cols = [COL_SHIP_TO] + ([COL_LOC] if loc_flag else []) + [COL_ITEM, time_col]
            agg_d  = {qty_col: 'sum'}
            if amt_col:
                agg_d[amt_col] = 'sum'

            # df_tag = (df_tag.groupby(g_cols, as_index=False, sort=False)
            #                   .agg(agg_d))
            
            df_tag = ultra_fast_groupby_numpy_general(
                df=df_tag,
                key_cols=g_cols,
                aggs=agg_d
            )
            # for debug
            # if (out_prefix == 'Sell_Out_FCST') & (tag == 'GC'):
            #     fn_log_dataframe(df_tag,f'step_05_{out_prefix}_{tag}_afterGroup',len(df_tag))


            # 상위 Ship-to 제거
            # df_tag = remove_upper_shipto(df_tag, loc_flag=loc_flag, time_col=time_col)



            df_tag.rename(columns={qty_col: _rename(qty_col, tag)}, inplace=True)
            if amt_col:
                df_tag.rename(columns={amt_col: _rename(amt_col, tag)}, inplace=True)

            # 🆕  object → category  (메모리 70~80 % 절감)
            cat_cols = [COL_SHIP_TO, COL_ITEM, time_col]
            if loc_flag:
                cat_cols.append(COL_LOC)
            df_tag[cat_cols] = df_tag[cat_cols].astype('category')

            frames[tag] = df_tag

        return {f'df_output_{out_prefix}_{tag}': df for tag, df in frames.items()}

    # ─────────────────────────────────────────────────────────────────────────
    # 3)  Build 실행
    # ─────────────────────────────────────────────────────────────────────────
    out_dict: dict[str, pd.DataFrame] = {}

    # Sell-In (GI / BL)
    out_dict |= build_level(df_si_gi, COL_SIN_ACT_GI, COL_SIN_ACT_GI_AMT,
                            True,  COL_TIME_PW, 'Sell_In_FCST_GI')
    out_dict |= build_level(df_si_bl, COL_SIN_ACT_BL, COL_SIN_ACT_BL_AMT,
                            True,  COL_TIME_PW, 'Sell_In_FCST_BL')

    # Sell-Out (Actual → FCST 레벨)
    out_dict |= build_level(df_so_act, COL_SOUT_ACT,  COL_SOUT_ACT_AMT,
                            True,  COL_TIME_PW, 'Sell_Out_FCST')

    # Inventory (Week, LOC 無)
    out_dict |= build_level(df_ci,  COL_CH_INV,           None,
                            False, COL_TIME_WK, 'Channel_Inv')
    out_dict |= build_level(df_cif, COL_CH_INV_SUM_FLOOR, None,
                            False, COL_TIME_WK, 'Channel_Inv_Floor')

    # ─────────────────────────────────────────────────────────────────────────
    # 4) 메모리 정리 & 반환
    # ─────────────────────────────────────────────────────────────────────────
    del (df_si_gi, df_si_bl, df_so_act, df_ci, df_cif, df_rule, df_dim)
    gc.collect()

    return out_dict

########################################################################################################################
# Step 06 : 최종 Output DataFrame 에 Version 컬럼 추가
########################################################################################################################
@_decoration_
def fn_step06_append_version(
        dict_lv_df : dict[str, pd.DataFrame],      # ← Step 05 반환 20 개 DataFrame dict
        version_str: str = 'CWV_DP'                # 고정값
) -> dict[str, pd.DataFrame]:
    """
    Step 06) Output Data 에 Version.[Version Name] 컬럼 부여
    ───────────────────────────────────────────────────────────────────────────
    • 입력 : { '<KEY>' : DataFrame, … }   (Step05 결과)
    • 작업 : 모든 DataFrame 에
             ‣ ① Version.[Version Name] = <version_str>
             ‣ ② 컬럼 순서 : Version 컬럼을 가장 앞에 배치
    • 반환 : 동일 key 구조의 dict  (in-place X, 새 DataFrame 복사)
    """
    if not dict_lv_df:
        logger.Note('[Step06] 입력 dict 가 비어 있어 단계를 건너뜁니다.',
                    LOG_LEVEL.warning())
        return {}    
    
    out_dict: dict[str, pd.DataFrame] = {}

    for name, df in dict_lv_df.items():
        # if df.empty:
        #     out_dict[name] = df                         # 그대로 전달
        #     continue

        # ── 복사 후 Version 컬럼 추가 (vectorised) ─────────────────────────
        df_out = df.copy()
        df_out.insert(0, COL_VERSION, version_str)      # 가장 앞에 삽입

        # 🆕  object → category  (메모리 70~80 % 절감)
        cat_cols = [COL_VERSION]
        df_out[cat_cols] = df_out[cat_cols].astype('category')

        out_dict[name] = df_out

    # 메모리 정리 ------------------------------------------------------------
    del dict_lv_df
    gc.collect()

    return out_dict

########################################################################################################################
# Step 07 :  Input × Output Merge → 최종 테이블 생성. 사용안함.
########################################################################################################################
@_decoration_
def fn_step07_merge_input_output(
        input_dfs : dict[str, pd.DataFrame],      # step-00 에서 채운 input_dataframes
        output_dfs: dict[str, pd.DataFrame]       # step-06 까지 만들어진 dict_step06
) -> dict[str, pd.DataFrame]:
    """
    Step 07)  Input 과 Output 을 OUTER JOIN 하여 ‘최종’ 데이터셋 생성
    ───────────────────────────────────────────────────────────────────────────
    • 목적 : **누락(Key 미존재) 행이 없도록 보장**  
      - input 에만 있는 행 → 출력 측 measure 는 NaN  
      - output 에만 있는 행 → input 측 measure 는 NaN  
    • rule : *항상 Output 측 값을 신뢰*  
      - merge 후 **input measure 컬럼은 버리고** output measure 만 남긴다.  
      - 즉, output 값이 NaN 이면 input 값이 있어도 NaN 으로 유지된다.
    • 반환 : { '<최종 DF 이름>' : DataFrame, … }  (20 개)
    """
    # ─────────────────────────────────────────────────────────────────────────
    # 0)  Merge 세트 정의  (FCST 17 종 + Inv 3 종 = 20)
    #     ▸ 이름·측정컬럼·조인키를 자료구조로 작성해 반복문으로 처리
    # ─────────────────────────────────────────────────────────────────────────    
     
    # ① FCST류 17 종  ───────────────────────────────────────────────────────
    FCST_TABLES: list[tuple[str, str, list[str], list[str]]] = [
        #  in-DF name,         out-DF name,                     join-cols,                                      measure-cols
        (STR_DF_IN_SI_FCST_GI_AP1,   STR_DF_OUT_SI_FCST_GI_AP1,   # Sell-In GI AP1
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SIN_FCST_GI_AP1, COL_SIN_FCST_GI_AP1_AMT]),

        (STR_DF_IN_SI_FCST_GI_AP2,   STR_DF_OUT_SI_FCST_GI_AP2,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SIN_FCST_GI_AP2, COL_SIN_FCST_GI_AP2_AMT]),

        (STR_DF_IN_SI_FCST_GI_GC,    STR_DF_OUT_SI_FCST_GI_GC,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SIN_FCST_GI_GC, COL_SIN_FCST_GI_GC_AMT]),

        (STR_DF_IN_SI_FCST_GI_LOCAL, STR_DF_OUT_SI_FCST_GI_LOCAL,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SIN_FCST_GI_LOCAL, COL_SIN_FCST_GI_LOCAL_AMT]),

        (STR_DF_IN_SI_FCST_BL_AP1,   STR_DF_OUT_SI_FCST_BL_AP1,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SIN_FCST_BL_AP1, COL_SIN_FCST_BL_AP1_AMT]),

        (STR_DF_IN_SI_FCST_BL_AP2,   STR_DF_OUT_SI_FCST_BL_AP2,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SIN_FCST_BL_AP2, COL_SIN_FCST_BL_AP2_AMT]),

        (STR_DF_IN_SI_FCST_BL_GC,    STR_DF_OUT_SI_FCST_BL_GC,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SIN_FCST_BL_GC, COL_SIN_FCST_BL_GC_AMT]),

        (STR_DF_IN_SI_FCST_BL_LOCAL, STR_DF_OUT_SI_FCST_BL_LOCAL,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SIN_FCST_BL_LOCAL, COL_SIN_FCST_BL_LOCAL_AMT]),

        (STR_DF_IN_SO_FCST_AP1,      STR_DF_OUT_SO_FCST_AP1,      # Sell-Out
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SOUT_FCST_AP1, COL_SOUT_FCST_AP1_AMT]),

        (STR_DF_IN_SO_FCST_AP2,      STR_DF_OUT_SO_FCST_AP2,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SOUT_FCST_AP2, COL_SOUT_FCST_AP2_AMT]),

        (STR_DF_IN_SO_FCST_GC,       STR_DF_OUT_SO_FCST_GC,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SOUT_FCST_GC, COL_SOUT_FCST_GC_AMT]),

        (STR_DF_IN_SO_FCST_LOCAL,    STR_DF_OUT_SO_FCST_LOCAL,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_TIME_PW],
         [COL_SOUT_FCST_LOCAL, COL_SOUT_FCST_LOCAL_AMT]),
    ]

    # ② Inventory류 3 종  ──────────────────────────────────────────────────
    INV_TABLES: list[tuple[str, str, list[str], list[str]]] = [
        (STR_DF_IN_CH_INV_AP1,       STR_DF_OUT_CH_INV_AP1,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
         [COL_CH_INV_AP1]),

        (STR_DF_IN_CH_INV_AP2,       STR_DF_OUT_CH_INV_AP2,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
         [COL_CH_INV_AP2]),

        (STR_DF_IN_CH_INV_GC,        STR_DF_OUT_CH_INV_GC,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
         [COL_CH_INV_GC]),

        (STR_DF_IN_CH_INV_LOCAL,     STR_DF_OUT_CH_INV_LOCAL,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
         [COL_CH_INV_LOCAL]),

        (STR_DF_IN_CH_INV_FLR_AP1,   STR_DF_OUT_CH_INV_FLR_AP1,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
         [COL_CH_INV_FLR_AP1]),

        (STR_DF_IN_CH_INV_FLR_AP2,   STR_DF_OUT_CH_INV_FLR_AP2,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
         [COL_CH_INV_FLR_AP2]),

        (STR_DF_IN_CH_INV_FLR_GC,    STR_DF_OUT_CH_INV_FLR_GC,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
         [COL_CH_INV_FLR_GC]),

        (STR_DF_IN_CH_INV_FLR_LOCAL, STR_DF_OUT_CH_INV_FLR_LOCAL,
         [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK],
         [COL_CH_INV_FLR_LOCAL]),
    ]

    MERGE_SET = FCST_TABLES + INV_TABLES

    # ─────────────────────────────────────────────────────────────────────────
    # 1)  루프 처리
    # ─────────────────────────────────────────────────────────────────────────
    merged_dict: dict[str, pd.DataFrame] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # (NEW)  helper : 카테고리 키 정렬
    # ─────────────────────────────────────────────────────────────────────────
    def _align_categories(df_left: pd.DataFrame,
                        df_right: pd.DataFrame,
                        join_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        양쪽 DF 의 join key 가 모두 category 면 → categories union 으로 통일.
        그렇지 않으면 object(str) 로 임시 다운-캐스트.
        """
        # fn_log_dataframe(df_left,'step07_align_categories_df_left')
        # fn_log_dataframe(df_right,'step07_align_categories_df_right')
        for col in join_cols:
            if (pd.api.types.is_categorical_dtype(df_left[col]) and
                pd.api.types.is_categorical_dtype(df_right[col])):            # 두 카테고리를 합집합으로 통일
                new_cat = pd.api.types.union_categoricals([df_left[col], df_right[col]]).categories
                df_left [col] = df_left [col].cat.set_categories(new_cat)
                df_right[col] = df_right[col].cat.set_categories(new_cat)

            else:
                # dtype 이 다르면 object(str) 로 임시 변환
                df_left [col] = df_left [col].astype('object')
                df_right[col] = df_right[col].astype('object')

        return df_left, df_right

    for in_name, out_name, keys, meas_cols in MERGE_SET:
        logger.Note(f'in_name : {in_name},  out_name : {out_name},',LOG_LEVEL.debug())
        df_in  = input_dfs .get(in_name,  pd.DataFrame())
        df_out = output_dfs.get(out_name, pd.DataFrame())

        # Sell OUt 인 경우 step02 에서 전처리한 것을 사용한다.
        if in_name == STR_DF_IN_SO_FCST_AP2:
            df_in = df_fn_Sell_Out_FCST_AP2
        if in_name == STR_DF_IN_SO_FCST_AP1:
            df_in = df_fn_Sell_Out_FCST_AP1
        if in_name == STR_DF_IN_SO_FCST_GC:
            df_in = df_fn_Sell_Out_FCST_GC
        if in_name == STR_DF_IN_SO_FCST_LOCAL:
            df_in = df_fn_Sell_Out_FCST_Local

        # if df_in.empty and df_out.empty:
        #     continue      # 둘 다 비어 있으면 skip (경고 로그는 데코레이터에서)

        # (1) 카테고리 키 정렬 ← ★ 추가
        df_in, df_out = _align_categories(df_in, df_out, keys)

        # --- (1)  컬럼 구분용 접미사 ------------------------------------------------
        df_in  = df_in .rename(columns={c: f'{c}_in'   for c in meas_cols})
        df_out = df_out.rename(columns={c: f'{c}_out'  for c in meas_cols})

        # --- (2)  OUTER JOIN --------------------------------------------------------
        df_merged = df_in.merge(df_out, how='outer', on=keys, sort=False)

        # --- (3)  input measure drop  ----------------------------------------------
        df_merged.drop(columns=[f'{c}_in' for c in meas_cols], inplace=True)

        # --- (4)  접미사 제거 → 최종 칼럼명 정리 ------------------------------------
        df_merged.rename(columns={f'{c}_out': c for c in meas_cols}, inplace=True)

        # --- (5)  문자열 → category 캐스팅 (메모리 감소) -----------------------------
        obj_cols = df_merged.select_dtypes(include='object').columns
        if obj_cols.any():
            df_merged[obj_cols] = df_merged[obj_cols].astype('category')

        merged_dict[out_name] = df_merged

    # ─────────────────────────────────────────────────────────────────────────
    # 2)  메모리 정리 & 반환
    # ─────────────────────────────────────────────────────────────────────────
    del output_dfs
    gc.collect()
    return merged_dict

########################################################################################################################
# Step 08 : P4W Channel-Inventory 생성 (AP2 기반 복제)
########################################################################################################################
@_decoration_
def fn_step08_make_p4w_channel_inventory(
        # ── inputs ───────────────────────────────────────────────────────────────────────────────────────────────────
        df_out_ci_ap2     : pd.DataFrame,   # df_output_Channel_Inv_AP2
        df_out_ci_flr_ap2 : pd.DataFrame    # df_output_Channel_Inv_Floor_AP2
) -> dict[str, pd.DataFrame]:
    """
    Step 08)  (Simul)P4W Ch Inv / (Simul)P4W Ch Inv_Inc Floor 생성
    ───────────────────────────────────────────────────────────────────────────
    • 입력:  AP2 결과 테이블 2종
        - df_output_Channel_Inv_AP2
        - df_output_Channel_Inv_Floor_AP2
    • 처리:  각 DF를 그대로 복제하되, Measure 컬럼명을 아래로 교체
        - '(Simul)Ch Inv_AP2'          → '(Simul)P4W Ch Inv'
        - '(Simul)Ch Inv_Inc Floor_AP2'→ '(Simul)P4W Ch Inv_Inc Floor'
      (※ 과거 명명 '(Simul). Channel Inv AP2' 등도 자동 호환)
    • 반환:  { 'df_output_Channel_Inv_P4W' , 'df_output_Channel_Inv_floor_P4W' }
    """
    # ── 입력 검증 ────────────────────────────────────────────────────────────────────────────────────────────────────
    if df_out_ci_ap2 is None or df_out_ci_flr_ap2 is None:
        logger.Note('[Step08] AP2 기반 Channel-Inv 입력이 None 입니다. 단계를 건너뜁니다.',
                    LOG_LEVEL.warning())
        return {}
    # ── 호환 가능한 원본 Measure 후보 (우선순위: 신규 → 과거) ───────────────────────────────────────────────────────
    MEAS_INV_CANDIDATES  = [COL_CH_INV_AP2,       COL_SIMUL_CH_INV_AP2]        # '(Simul)Ch Inv_AP2' / '(Simul). Channel Inv AP2'
    MEAS_FLR_CANDIDATES  = [COL_CH_INV_FLR_AP2,   COL_SIMUL_CH_INV_FLR_AP2]    # '(Simul)Ch Inv_Inc Floor_AP2'

    # ── 공용 유틸 ───────────────────────────────────────────────────────────────────────────────────────────────────
    def _rename_measure(df: pd.DataFrame, candidates: list[str], new_name: str) -> pd.DataFrame:
        """candidates 중 존재하는 첫 컬럼을 new_name 으로 변경하여 반환"""
        if df.empty:
            # 빈 DF도 스펙 컬럼으로 “껍데기” 구성
            cols_pref = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK]
            cols_exist = [c for c in cols_pref if c in df.columns]
            return pd.DataFrame(columns=cols_exist + [new_name])

        for old in candidates:
            if old in df.columns:
                return df.rename(columns={old: new_name})
        raise KeyError(
            f"[Step08] 기대 Measure {candidates} 가 존재하지 않습니다. columns={list(df.columns)}"
        )

    def _reorder(df: pd.DataFrame, prefer: list[str]) -> pd.DataFrame:
        """출력 스펙 정렬 - 선호 컬럼 먼저, 나머지 유지"""
        exist  = [c for c in prefer if c in df.columns]
        others = [c for c in df.columns if c not in exist]
        return df[exist + others]

    def _cast_categories(df: pd.DataFrame, keys: list[str]) -> None:
        for c in keys:
            if c in df.columns:
                df[c] = df[c].astype('category')

    # ── P4W Ch Inv 생성 ────────────────────────────────────────────────────────────────────────────────────────────
    df_p4w_inv = _rename_measure(
        df_out_ci_ap2.copy(),
        MEAS_INV_CANDIDATES,
        COL_SIM_P4W_CH_INV
    )
    df_p4w_inv = _reorder(
        df_p4w_inv,
        [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK, COL_SIM_P4W_CH_INV]
    )
    _cast_categories(df_p4w_inv, [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK])

    # ── P4W Ch Inv_Inc Floor 생성 ──────────────────────────────────────────────────────────────────────────────────
    df_p4w_flr = _rename_measure(
        df_out_ci_flr_ap2.copy(),
        MEAS_FLR_CANDIDATES,
        COL_SIM_P4W_CH_INV_FLOOR
    )
    df_p4w_flr = _reorder(
        df_p4w_flr,
        [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK, COL_SIM_P4W_CH_INV_FLOOR]
    )
    _cast_categories(df_p4w_flr, [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_WK])

    # ── 메모리 정리 & 반환 ─────────────────────────────────────────────────────────────────────────────────────────
    out_dict: dict[str, pd.DataFrame] = {
        DF_OUT_SIM_P4W_CH_INV      : df_p4w_inv,
        DF_OUT_SIM_P4W_CH_INV_FLOOR: df_p4w_flr
    }

    del (df_out_ci_ap2, df_out_ci_flr_ap2, df_p4w_inv, df_p4w_flr)
    gc.collect()

    return out_dict

########################################################################################################################
# 공통  Output Formatter
########################################################################################################################
@_decoration_
def fn_output_formatter(df_src      : pd.DataFrame,
                        erd_columns : list,
                        out_version : str
                        ) -> pd.DataFrame:
    """
    • 목적  : ERD 정의 순서에 맞춰 컬럼을 정렬하고,
             Version.[Version Name] 컬럼(Param_OUT_VERSION 값) 을 선두에 삽입한 결과 DF 반환  
    • 사용법:  
        df_out = fn_output_formatter(
                     df_src       = output_dataframes['df_output_Sell_In_FCST_GI_AP1'],
                     erd_columns  = [
                         Version_Name, Sales_Domain_ShipTo, Item_Item,
                         Location_Location, SIn_FCST_AP1_GI
                     ],
                     out_version  = Param_OUT_VERSION )
    • 파라미터
        df_src      : 원본 DataFrame   (None 또는 empty 허용)  
        erd_columns : ERD 순서대로 정렬된 컬럼 리스트 (Version_Name 포함 X)  
        out_version : Version 값을 채울 문자열 (Param_OUT_VERSION)  
    • 결과
        • 반환값: 정렬/보정된 DataFrame  
        • 비어 있으면 빈 DF 생성 후 반환 ‑‐> 호출 측에서 그대로 저장 가능
    """
    # ------------------------------------------------------------------
    # 1. 빈 DF 대응
    # ------------------------------------------------------------------
    if df_src is None or df_src.empty:
        df_fmt = pd.DataFrame({col: [] for col in [COL_VERSION] + erd_columns})
        return df_fmt    
    # ------------------------------------------------------------------
    # 2. Version 컬럼 삽입 / 값 세팅
    # ------------------------------------------------------------------
    df = df_src.copy()
    df[COL_VERSION] = out_version        # 존재 시 덮어씀, 없으면 생성
    # 컬럼 순서를 위해 앞으로 이동
    cols = [c for c in df.columns if c != COL_VERSION]
    df = df[[COL_VERSION] + cols]

    # ------------------------------------------------------------------
    # 3. ERD 컬럼 정렬 · 누락 컬럼 추가
    # ------------------------------------------------------------------
    # ERD 순서 + 원본 존재 컬럼 교집합
    ordered_cols = [COL_VERSION] + [c for c in erd_columns if c in df.columns]
    # ERD 에 정의됐지만 DF 에 없는 컬럼 보강 (빈 값)
    for miss_col in erd_columns:
        if miss_col not in df.columns:
            df[miss_col] = np.nan
            ordered_cols.append(miss_col)

    # 최종 정렬
    df_fmt = df[ordered_cols]

    return df_fmt


####################################
############ Start Main  ###########
####################################
if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    # Output 테이블 선언
    out_sellin = pd.DataFrame()
    out_sellout = pd.DataFrame()
    input_dataframes = {}
    output_dataframes = {}
    try:
        ################################################################################################################
        # 전처리 : 모듈 내에서 사용될 데이터에 대한 정합성 체크 및 데이터 선 가공
        ################################################################################################################
       
        # ----------------------------------------------------
        # parse_args 대체
        # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
        # ----------------------------------------------------
        if is_local:
            Version = 'CWV_DP'

            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------
            # input_folder_name  = str_instance
            # output_folder_name = str_instance
            # ---
            # input_folder_name  = 'PYDPMakeActualAndInventoryForecastLevel/0715_AP0'
            # output_folder_name = 'PYDPMakeActualAndInventoryForecastLevel_0715_AP0'
            # ---
            # input_folder_name  = 'PYDPMakeActualAndInventoryForecastLevel/0715_AP0_o9'
            # output_folder_name = 'PYDPMakeActualAndInventoryForecastLevel_0715_AP0_o9'
            # ---
            # input_folder_name  = 'PYDPMakeActualAndInventoryForecastLevel/0715_AP0_Local'
            # output_folder_name = 'PYDPMakeActualAndInventoryForecastLevel_0715_AP0_Local'
            # # ---  본데이타 ShipTo 의 Lv 보다 Forecast Lv 이 낮은경우
            # input_folder_name  = 'PYDPMakeActualAndInventoryForecastLevel/0715_AP0_300139_v1'
            # output_folder_name = 'PYDPMakeActualAndInventoryForecastLevel_0715_AP0_300139_v1'
            # ---  0910
            input_folder_name  = 'PYDPMakeActualAndInventoryForecastLevel/0910_step08'
            output_folder_name = 'PYDPMakeActualAndInventoryForecastLevel_0910_step08'
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
        # --------------------------------------------------------------------------
        # df_input 체크 시작
        # --------------------------------------------------------------------------
        logger.Note(p_note='df_input 체크 시작', p_log_level=LOG_LEVEL.debug())
        fn_process_in_df_mst()

        # 초기 MST 데이타 체크. 입력 변수 중 데이터가 없는 경우 경고 메시지를 출력한다.
        for in_df in input_dataframes:
            # 로그출력
            fn_log_dataframe(input_dataframes[in_df], in_df)
        
        logger.Note(p_note=f'Parameter Check', p_log_level=LOG_LEVEL.debug())
        logger.Note(p_note=f'Version            : {Version}', p_log_level=LOG_LEVEL.debug())
        logger.Note(p_note=f'CurrentPartialWeek : {CurrentPartialWeek}', p_log_level=LOG_LEVEL.debug())

        # --------------------------------------------------------------------------
        # df_input 체크 종료
        # --------------------------------------------------------------------------
        # 입력 변수 확인
        if Version is None or Version.strip() == '':
            Version = 'CWV_DP'
        logger.Note(p_note=f'VERSION : {Version}', p_log_level=LOG_LEVEL.debug())

################################################################################################################
#################### Start Call Main  ##########
################################################################################################################
  

        
        ################################################################################################################
        # Step 01. S/In Actual(GI·BL) 전처리
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 10,
            'p_step_desc': 'Step 01) S/In Actual(GI·BL) 전처리',
            'p_df_name'  : None
        }
        df_fn_Sell_In_Actual_GI, df_fn_Sell_In_Actual_BL = (
            fn_step01_preprocess_sell_in_actual(
                input_dataframes[STR_DF_IN_SI_ACT_GI],
                input_dataframes[STR_DF_IN_SI_ACT_BL],
                **dict_log
            )
        )

        fn_log_dataframe(df_fn_Sell_In_Actual_GI, 'step_01_df_fn_Sell_In_Actual_GI')
        fn_log_dataframe(df_fn_Sell_In_Actual_BL, 'step_01_df_fn_Sell_In_Actual_BL')
        ################################################################################################################
        # Step 02  : S/Out Actual 전처리
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 20,
            'p_step_desc': 'Step 02) S/Out Actual · FCST 전처리',
            'p_df_name'  : None
        }

        (df_fn_Sell_Out_Actual,
        df_fn_Sell_Out_FCST_AP1,
        df_fn_Sell_Out_FCST_AP2,
        df_fn_Sell_Out_FCST_GC,
        df_fn_Sell_Out_FCST_Local) = fn_step02_preprocess_sell_out(
            input_dataframes[STR_DF_IN_SO_ACT],
            input_dataframes[STR_DF_IN_WEEK],
            input_dataframes[STR_DF_IN_PARTIALWEEK],
            input_dataframes[STR_DF_IN_SALES_DOMAIN_ESTORE],
            input_dataframes[STR_DF_IN_SO_FCST_AP1],
            input_dataframes[STR_DF_IN_SO_FCST_AP2],
            input_dataframes[STR_DF_IN_SO_FCST_GC],
            input_dataframes[STR_DF_IN_SO_FCST_LOCAL],
            **dict_log        # _decoration_ 에서 로그 처리
        )
        
        fn_log_dataframe(df_fn_Sell_Out_Actual,     'step_02_df_fn_Sell_Out_Actual')
        fn_log_dataframe(df_fn_Sell_Out_FCST_AP1,   'step_02_df_fn_Sell_Out_FCST_AP1')
        fn_log_dataframe(df_fn_Sell_Out_FCST_AP2,   'step_02_df_fn_Sell_Out_FCST_AP2')
        fn_log_dataframe(df_fn_Sell_Out_FCST_GC,    'step_02_df_fn_Sell_Out_FCST_GC')
        fn_log_dataframe(df_fn_Sell_Out_FCST_Local, 'step_02_df_fn_Sell_Out_FCST_Local')

        ################################################################################################################
        # Step 03:  Channel Inv 전처리
        ################################################################################################################
        dict_log = {
            'p_step_no' : 30,
            'p_step_desc': 'Step 03) Channel Inv 전처리',
            'p_df_name' : 'step_03_df_fn_Channel_Inv'
        }
        df_fn_Channel_Inv = fn_step03_preprocess_channel_inv(
            input_dataframes[STR_DF_IN_CH_INV],
            **dict_log
        )

        ################################################################################################################
        # Step 04 : (Sum) Channel Inv_Inc Floor 전처리
        ################################################################################################################
        dict_log = {
            'p_step_no' : 40,
            'p_step_desc': 'Step 04) (Sum) Channel Inv_Inc Floor 전처리',
            'p_df_name' : 'step_04_df_fn_Channel_Inv_Inc_Floor'
        }
        df_fn_Channel_Inv_Inc_Floor = fn_step04_preprocess_channel_inv_floor(
            input_dataframes[STR_DF_IN_CH_INV_FLOOR],
            **dict_log
        )


        # ################################################################################################################
        # Step 05 : Forecast‑Rule 레벨별 실적·재고 생성
        # ################################################################################################################
        dict_log = {
            'p_step_no'  : 50,
            'p_step_desc': 'Step 05) Forecast-Rule Level Build',
            'p_df_name'  : None                # 반환값이 dict ⇒ 개별 DF 로깅은 아래 loop
        }
        dict_step05 = fn_step05_make_actual_and_inv_fcst_level(
            df_fn_Sell_In_Actual_GI,          # ← Step 01 결과
            df_fn_Sell_In_Actual_BL,          # ← Step 01 결과
            df_fn_Sell_Out_Actual,            # ← Step 02 결과
            df_fn_Channel_Inv,                # ← Step 03 결과
            df_fn_Channel_Inv_Inc_Floor,      # ← Step 04 결과
            input_dataframes[STR_DF_IN_FCST_RULE],
            input_dataframes[STR_DF_IN_SALES_DOMAIN_DIM],
            **dict_log
        )

        # ── 개별 DataFrame 로그 & (옵션) 로컬 저장 ─────────────────────────────────────
        for key, df in dict_step05.items():
            fn_log_dataframe(df, f'step_05_{key}')
        

        ########################################################################################################################
        # Step 06) Version 컬럼 추가
        ########################################################################################################################
        dict_log = {
            'p_step_no' : 60,
            'p_step_desc': 'Step 06) Version 컬럼 추가',
            'p_df_name' : None          # 여러 DF → 개별 로그에서 처리
        }
        dict_step06 = fn_step06_append_version(
            dict_step05,                 # ← Step 05 결과 dict
            Version,
            **dict_log
        )

        # 결과 DataFrame 들 로깅
        for key, df in dict_step06.items():
            fn_log_dataframe(df, f'step_06_{key}')


        # 호출안함 Step 07

        
        ########################################################################################################################
        # 최종 output
        ########################################################################################################################
        df_output_Sell_In_FCST_GI_AP1		= dict_step06[STR_DF_OUT_SI_FCST_GI_AP1  ]			
        df_output_Sell_In_FCST_GI_AP2       = dict_step06[STR_DF_OUT_SI_FCST_GI_AP2  ]
        df_output_Sell_In_FCST_GI_GC        = dict_step06[STR_DF_OUT_SI_FCST_GI_GC   ]
        df_output_Sell_In_FCST_GI_Local     = dict_step06[STR_DF_OUT_SI_FCST_GI_LOCAL]
        df_output_Sell_In_FCST_BL_AP1       = dict_step06[STR_DF_OUT_SI_FCST_BL_AP1  ]
        df_output_Sell_In_FCST_BL_AP2       = dict_step06[STR_DF_OUT_SI_FCST_BL_AP2  ]
        df_output_Sell_In_FCST_BL_GC        = dict_step06[STR_DF_OUT_SI_FCST_BL_GC   ]
        df_output_Sell_In_FCST_BL_Local     = dict_step06[STR_DF_OUT_SI_FCST_BL_LOCAL]
        df_output_Sell_Out_FCST_AP1         = dict_step06[STR_DF_OUT_SO_FCST_AP1     ]
        df_output_Sell_Out_FCST_AP2         = dict_step06[STR_DF_OUT_SO_FCST_AP2     ]
        df_output_Sell_Out_FCST_GC          = dict_step06[STR_DF_OUT_SO_FCST_GC      ]
        df_output_Sell_Out_FCST_Local       = dict_step06[STR_DF_OUT_SO_FCST_LOCAL   ]
        df_output_Channel_Inv_AP1           = dict_step06[STR_DF_OUT_CH_INV_AP1      ]
        df_output_Channel_Inv_AP2           = dict_step06[STR_DF_OUT_CH_INV_AP2      ]
        df_output_Channel_Inv_GC            = dict_step06[STR_DF_OUT_CH_INV_GC       ]
        df_output_Channel_Inv_Local         = dict_step06[STR_DF_OUT_CH_INV_LOCAL    ]
        df_output_Channel_Inv_Floor_AP1     = dict_step06[STR_DF_OUT_CH_INV_FLR_AP1  ]
        df_output_Channel_Inv_Floor_AP2     = dict_step06[STR_DF_OUT_CH_INV_FLR_AP2  ]
        df_output_Channel_Inv_Floor_GC      = dict_step06[STR_DF_OUT_CH_INV_FLR_GC   ]
        df_output_Channel_Inv_Floor_Local   = dict_step06[STR_DF_OUT_CH_INV_FLR_LOCAL]   

        #######################################################################################################################
        # Step 08 : P4W Channel-Inventory 생성 (AP2 → P4W)
        #######################################################################################################################
        dict_log = {
            'p_step_no'  : 80,
            'p_step_desc': 'Step 08) P4W Channel-Inventory Build',
            'p_df_name'  : None
        }

        # ※ Step07에서 AP2 결과를 변경/보정했다면, 그 결과를 입력으로 넘기세요.
        #    (아래 예시는 Step05 결과를 직접 사용)
        dict_step08 = fn_step08_make_p4w_channel_inventory(
            df_output_Channel_Inv_AP2       ,
            df_output_Channel_Inv_Floor_AP2 ,
            **dict_log
        )

        # ── 로깅 (옵션) ─────────────────────────────────────────────────────────────────────────────────────────────────────
        for key, df in dict_step08.items():
            fn_log_dataframe(df, f'step_08_{key}')

        df_output_Channel_Inv_P4W         = dict_step08[DF_OUT_SIM_P4W_CH_INV   ]
        df_output_Channel_Inv_Floor_P4W   = dict_step08[DF_OUT_SIM_P4W_CH_INV_FLOOR]   
            
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
        # fn_set_header()

        if is_local:
            log_file_name = common.G_PROGRAM_NAME.replace('py', 'log')
            log_file_name = f'log/{log_file_name}'

            shutil.copyfile(log_file_name, os.path.join(str_output_dir, os.path.basename(log_file_name)))

            # prografile copy
            program_path = f"{os.getcwd()}/NSCM_DP_UI_Develop/{str_instance}.py"
            shutil.copyfile(program_path, os.path.join(str_output_dir, os.path.basename(program_path)))


        logger.Finish()
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')