
import os,sys
import time,datetime,shutil
import inspect
import traceback
import pandas as pd
from NSCMCommon import NSCMCommon as common
from NSCMCommon import VDCommon as vdCommon
# from typing_extensions import Literal
import glob
import re
import numpy as np
import gc

########################################################################################################################
# Local 개발 시에 필요한 공통 변수 선언
########################################################################################################################
# o9에 저장된 instanceName
str_instance = 'PYForecastVDSellOutAP1Rolling'
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
# ───────────────────────────────────────────────────────────────
# CONSTANT STRING VARIABLES FOR COLUMN NAMES  (Sell-OUT  AP1)
# ───────────────────────────────────────────────────────────────
COL_VERSION          = 'Version.[Version Name]'
COL_ITEM             = 'Item.[Item]'
COL_SHIP_TO          = 'Sales Domain.[Ship To]'
COL_LOC              = 'Location.[Location]'
COL_PYEAR            = 'Time.[Planning Year]'
COL_PMONTH           = 'Time.[Planning Month]'
COL_WEEK             = 'Time.[Week]'
COL_FCST_AP1         = 'S/Out FCST_AP1'
COL_FCST_AP1_ROLL    = 'S/Out FCST_AP1(Rolling ADJ)'
COL_GBRULE           = 'GBRULE'                       
# ← **원본 df_in_Sout 에 존재**
# ───────────────────────────────────────────────────────────────
# CONSTANT STRING VARIABLES FOR DATAFRAME HANDLES
# ───────────────────────────────────────────────────────────────
# input
STR_DF_IN_SOUT       = 'df_in_Sout'
STR_DF_IN_TIME       = 'df_in_Time'
# step
STR_DF_STEP01_ROLL   = 'df_fn_step01_rolling'
# output
STR_DF_OUT_DEMAND    = 'out_Demand'


################  Start of Functions  ################
# ----------------------------------------------------------------------------------
# Helper 함수
# ----------------------------------------------------------------------------------
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
    logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=20)

    if df_p_source.empty:
        if str_p_cond == '0':
            # 테이블이 비어 있는 경우 raise Exception
            raise Exception(f'[Exception] Input table({str_p_source_name}) is empty.')
        else:
            # 테이블이 비어 있는 경우 Warning log
            logger.Note(p_note=f'Input table({str_p_source_name}) is empty.', p_log_level=LOG_LEVEL.warning())


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

# ──────────────────────────────────────────────────────────────────────────────
# HELPER : 주차 → int (ex. '202415A' → 202415)
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_week(week_str):
    """Convert a week string with potential suffixes to an integer for comparison."""
    # Remove any non-digit characters (e.g., 'A' or 'B') and convert to integer
    try:

        return int(''.join(filter(str.isdigit, week_str)))
    except Exception as e:
        logger.Note(p_note=f"week_str: {week_str}", p_log_level=LOG_LEVEL.error())


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
            COL_ITEM         : [],
            COL_LOC             : [],
            COL_WEEK : [],
            COL_FCST_AP1_ROLL : []

        }
    )

    return df_return

def fn_convert_type(df: pd.DataFrame, startWith: str, type):
    for column in df.columns:
        if column.startswith(startWith):
            df[column] = df[column].astype(type)

def fn_convert_type_equal(df: pd.DataFrame, column: str, type):
    df[column] = df[column].astype(type)
            
# ──────────────────────────────────────────────────────────────────────────────
#  STEP-00 : 공통 타입 변환  (❌ `global` 사용 금지)
#           호출 측에서 `input_dataframes` 를 인자로 넘겨준다.
# ──────────────────────────────────────────────────────────────────────────────
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
            "df_in_Sout.csv" : STR_DF_IN_SOUT       ,
            "df_in_Time.csv" : STR_DF_IN_TIME 
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
        input_dataframes[STR_DF_IN_SOUT]        = df_in_Sout
        input_dataframes[STR_DF_IN_TIME]        = df_in_Time
    
    # input_dataframes = {...}  # 이미 채워진 전역 dict
    fn_convert_type(input_dataframes[STR_DF_IN_SOUT],'Sales Domain.',str)
    fn_convert_type(input_dataframes[STR_DF_IN_SOUT],'Time.','int32')
    fn_convert_type(input_dataframes[STR_DF_IN_TIME],'Time.','int32')
    fn_prepare_input_types(input_dataframes)

def fn_fill_missing_weeks_year_end(
    df_src: pd.DataFrame,          # 특정 Version 의 원본 Forecast
    df_time: pd.DataFrame,         # Time Master
    current_week: str | int,
    version: str
) -> pd.DataFrame:
    """
    ▸ current_week  ~ 해당 연도의 **최대 주차**(df_time 기준)까지
      Ship-To × Item × Loc × Week 그리드를 생성하고
      Forecast 값이 없으면 0 으로 채운다.
    """
    # ── 1) 기준 정보 ───────────────────────────────────────────────────
    wk_curr  = _normalize_week(str(current_week))          # ex) '202417'
    year_str = str(wk_curr)[:4]                       # '2024'    # df_time 에서 같은 연도의 최대 주차를 추출
    wk_max = (
        df_time.loc[df_time[COL_WEEK].astype(str).str[:4] == year_str, COL_WEEK]
        .astype(int).max()
    )

    # 대상 주차 리스트 (df_time 에 존재하는 Week 만 사용)
    week_list = (
        df_time.loc[
            # (df_time[COL_WEEK].astype(int) >= wk_curr) &
            (df_time[COL_WEEK].astype(int) <= wk_max)
        ][COL_WEEK]
        .astype(int).sort_values().unique()
    )

    # ── 2) cross-join 그리드 ──────────────────────────────────────────
    key_cols  = [COL_SHIP_TO, COL_ITEM, COL_LOC]
    base_keys = df_src[key_cols].drop_duplicates()

    grid = base_keys.merge(
        pd.DataFrame({COL_WEEK: week_list}),
        how="cross"
    )
    grid[COL_VERSION] = version

    # ── 3) Forecast 병합 & 0-채움 ─────────────────────────────────────
    out = (
        grid.merge(
            df_src[[COL_VERSION, *key_cols, COL_WEEK, COL_FCST_AP1]],
            how="left"
        )
        .fillna({COL_FCST_AP1: 0})
    )
    out[COL_FCST_AP1] = out[COL_FCST_AP1].astype("int32")
    return out

# ************************************************************************************
# 본격적인 Step 함수
# ************************************************************************************


# ──────────────────────────────────────────────────────────────────────────────
# STEP-01 : VD SellOut AP1 Rolling 로직 적용 및 S/Out FCST_AP1(Rolling ADJ) Measure 생성
# VD Sell-Out AP1 Rolling (Vectorized)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# STEP-01 : VD Sell-Out AP1 Rolling  (Vectorized + 메모리 최적화)
# ──────────────────────────────────────────────────────────────────────────────
# import gc                                     # ← ➊ GC 사용@_decoration_
@_decoration_
def fn_step01_vd_sellout_ap1_rolling(
    df_sout: pd.DataFrame,
    df_time: pd.DataFrame,
    current_week: str | int,
    version: str,
    prev_version: str | None = None,
) -> pd.DataFrame:

    # ── 0) 주차·Version 준비 ──────────────────────────────────────────────
    wk_curr = _normalize_week(current_week)
    wk_3    = int(common.gfn_add_week(str(wk_curr)[:6], -3))
    wk_2    = int(common.gfn_add_week(str(wk_curr)[:6], -2))

    if prev_version is None:
        cand = (
            df_sout.loc[df_sout[COL_WEEK] == wk_curr, COL_VERSION]
            .unique().tolist()
        )
        cand = [v for v in cand if v != version]
        prev_version = cand[0] if cand else None

    # ── 1) Week 그리드 확장 ───────────────────────────────────────────────
    df_curr_full = fn_fill_missing_weeks_year_end(
        df_sout[df_sout[COL_VERSION] == version],
        df_time, wk_curr, version
    )

    if prev_version:
        df_prev_full = fn_fill_missing_weeks_year_end(
            df_sout[df_sout[COL_VERSION] == prev_version],
            df_time, wk_curr, prev_version
        )
    else:
        df_prev_full = df_curr_full.copy()
        df_prev_full[COL_VERSION]  = "NA"
        df_prev_full[COL_FCST_AP1] = 0

    # ── 2) Pivot (행 : Ship-To×Item×Loc, 열 : Week) ──────────────────────
    grp_cols  = [COL_SHIP_TO, COL_ITEM, COL_LOC]
    week_cols = sorted(df_curr_full[COL_WEEK].unique())
    idx_curr  = week_cols.index(wk_curr)

    # column 을 category 로 바꿔 메모리 ↓
    for c in grp_cols:
        df_curr_full[c] = df_curr_full[c].astype("category")
        df_prev_full[c] = df_prev_full[c].astype("category")
    logger.Note('category 로 변경', p_log_level=LOG_LEVEL.debug())
    # def _build_pivot(df_src: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     observed=True 로 **중복 조합만** 유지 → pivot_table 보다 메모리 절약
    #     """
    #     g = (
    #         df_src
    #         .groupby(grp_cols + [COL_WEEK], observed=True)[COL_FCST_AP1]
    #         .first()
    #     )
    #     return (
    #         g.unstack(fill_value=0)               # (G, W)   ※ 중복 없는 값
    #          .reindex(columns=week_cols, fill_value=0)
    #          .astype("int32")
    #     )

    # def _build_pivot(df_src) -> pd.DataFrame:
    #     return (
    #         df_src
    #         .pivot_table(index=grp_cols,
    #                      columns=COL_WEEK,
    #                      values=COL_FCST_AP1,
    #                      aggfunc='first',
    #                      fill_value=0)
    #         .reindex(columns=week_cols, fill_value=0)
    #         .astype("int32")
    #     )

    # ──────────────── fix: _build_pivot + index-align ────────────────
    def _build_pivot(
        df_src: pd.DataFrame,
        week_cols: list[int],
    ) -> pd.DataFrame:
        """
        (Ship-To × Item × Loc) 행, Week 열의 int32 Matrix 를 만들어 돌려준다.
        - groupby(..., observed=True)   : 실제로 존재하는 조합만 집계 → 메모리 ↓
        - .first()                     : (동일 그룹, 동일 Week) 이 1행이므로 ‘첫 값’ = 유일 값
        - .unstack(fill_value=0)       : Week 컬럼을 wide 포맷으로 전개. 없는 Week → 0
        """
        g = (
            df_src
            .groupby(grp_cols + [COL_WEEK], observed=True, sort=False)[COL_FCST_AP1]
            .first()                                # ← pandas.Series, index = MultiIndex(G, W)
        )
        return (
            g.unstack(fill_value=0)                 # ← DataFrame, index = G, columns = Week
            .reindex(columns=week_cols, fill_value=0)
            .astype("int32")
        )
    # ------------------------------------------------------------------
    # pv_curr = _build_pivot(df_curr_full)
    # pv_prev = _build_pivot(df_prev_full)
    pv_curr = _build_pivot(df_curr_full, week_cols)
    pv_prev = _build_pivot(df_prev_full, week_cols)
    fn_check_input_table(pv_curr,'pv_curr_01','1')
    fn_check_input_table(pv_prev,'pv_prev_01','1')
    logger.Note('pivot 성공', p_log_level=LOG_LEVEL.debug())

    # **두 pivot 의 행 집합이 다르면 shape 미스매치 발생** → union 후 0-채움
    all_idx = pv_curr.index.union(pv_prev.index)
    pv_curr = pv_curr.reindex(all_idx, fill_value=0)
    pv_prev = pv_prev.reindex(all_idx, fill_value=0)
    fn_check_input_table(pv_curr,'pv_curr_02','1')
    fn_check_input_table(pv_prev,'pv_prev_02','1')
    logger.Note('pivot 미스매치 조정', p_log_level=LOG_LEVEL.debug())

    # 큰 DataFrame 해제 ➜ GC
    del df_curr_full, df_prev_full
    gc.collect()                                  # ← ➋

    # ── 3) Δ(week-3, week-2) Vector 계산 ──────────────────────────────────    
    pm_map  = df_time.set_index(COL_WEEK)[COL_PMONTH].astype(int).to_dict()
    same_m3 = int(pm_map.get(wk_3) == pm_map.get(wk_curr))
    same_m2 = int(pm_map.get(wk_2) == pm_map.get(wk_curr))

    # 25.08.21 : 동월인지를 체크하지는 않는다.
    same_m2 = 1 
    delta_total = (
          same_m3 * (pv_prev[wk_3].values - pv_curr[wk_3].values)
        + same_m2 * (pv_prev[wk_2].values - pv_curr[wk_2].values)
    ).astype("int32")

    pv_curr.iloc[:, pv_curr.columns.get_loc(wk_curr)] += delta_total
    logger.Note('delta 계산', p_log_level=LOG_LEVEL.debug())
    # 더 이상 안쓰는 pv_prev 해제
    del pv_prev, delta_total
    gc.collect()                                  # ← ➌

    # ── 4) 음수 carry-over  (Week loop × 그룹 벡터) ───────────────────────
    M = pv_curr.values  # NumPy view (int32)

    for c in range(idx_curr, len(week_cols) - 1):
        neg_mask = M[:, c] < 0
        if not neg_mask.any():
            continue
        carry           = M[neg_mask, c]
        M[neg_mask, c]  = 0
        M[neg_mask, c+1] += carry                # 다음 주차에 전가

    pv_curr.iloc[:, :] = M                      # 변경값 commit

    del M
    gc.collect()                                  # ← ➍
    logger.Note('carry-over', p_log_level=LOG_LEVEL.debug())
    # ── 5) Long 형식 복원 & 반환 ─────────────────────────────────────────
    df_return = (
        pv_curr
        .reset_index()
        .melt(id_vars=grp_cols,
              var_name=COL_WEEK,
              value_name=COL_FCST_AP1_ROLL)
    )
    del pv_curr
    gc.collect()                                  # ← ➎

    df_return[COL_VERSION] = version
    fn_convert_type(df_return, "Time.", "int32")
    fn_prepare_input_types({"df_return": df_return})

    return df_return[[COL_VERSION, *grp_cols, COL_WEEK, COL_FCST_AP1_ROLL]]

@_decoration_
def fn_output_formatter(
    df_roll: pd.DataFrame,          # ← step-01 결과(S/Out FCST_AP1(Rolling ADJ) 컬럼 포함)
    df_time: pd.DataFrame,          # ← Time master
    current_week: str | int,        # ex) '202417'
    version: str                    # ex) 'CWV_DP'
) -> pd.DataFrame:
    """
    • step-01 산출 DF(df_roll) 에서 누락 주차(당월 기준)를 채우고,
      Rolling 값이 없으면 0 으로 보강한다.
    • 반환 컬럼 순서:
        Version / Ship-To / Item / Loc / Week / S/Out FCST_AP1(Rolling ADJ)
    """
    logger.Note('1) 기준 정보', p_log_level=LOG_LEVEL.warning())
    if df_roll.empty:
        logger.Note('[fn_output_formatter] 시작', p_log_level=LOG_LEVEL.debug())
        return pd.DataFrame()    
    # ── 1) 기준 정보 ──────────────────────────────────────────────────────
    wk_curr = _normalize_week(current_week)
    year_str = str(wk_curr)[:4]   
    cur_pyear  = int(df_time.loc[df_time[COL_PYEAR] == int(year_str), COL_PYEAR].iloc[0])

    # ― 당월 주차 리스트 (int32, 오름차순)
    year_weeks: list[int] = (
        df_time.loc[
            (df_time[COL_PYEAR] == cur_pyear) 
            # & (df_time[COL_WEEK] >= wk_curr)
            , 
            COL_WEEK
        ]
        .astype(int).sort_values().tolist()
    )
    logger.Note('1) 기준 정보', p_log_level=LOG_LEVEL.debug())

    # ── 2) Version + 당월 key cross-join ───────────────────────────────
    key_cols  = [COL_SHIP_TO, COL_ITEM, COL_LOC]
    base_keys = (
        df_roll.loc[df_roll[COL_VERSION] == version, key_cols]
        .drop_duplicates()
    )

    week_df   = pd.DataFrame({COL_WEEK: year_weeks})
    full_grid = base_keys.merge(week_df, how='cross')
    full_grid[COL_VERSION] = version

    logger.Note('2) Version + 당월 key cross-join', p_log_level=LOG_LEVEL.debug())
    # ── 3) merge + 0-채움 ────────────────────────────────────────────────
    result = (
        full_grid.merge(
            df_roll[[COL_VERSION, *key_cols, COL_WEEK, COL_FCST_AP1_ROLL]],
            how='left'
        )
        .fillna({COL_FCST_AP1_ROLL: 0})
    )
    logger.Note('3) merge + 0-채움', p_log_level=LOG_LEVEL.debug())

    # 현재주차 이후
    mask = (
        (result[COL_WEEK] >= wk_curr)
    )
    result = result.loc[mask]
    

    # 타입 보강
    result[COL_WEEK]          = result[COL_WEEK].astype('int32')
    result[COL_FCST_AP1_ROLL] = result[COL_FCST_AP1_ROLL].astype('int32')

    # 최종 컬럼 순서
    final_cols = [
        COL_VERSION,
        COL_SHIP_TO,
        COL_ITEM,
        COL_LOC,
        COL_WEEK,
        COL_FCST_AP1_ROLL
    ]
    df_return = result[final_cols]

    fn_prepare_input_types({'df_return': df_return})
    return df_return



if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    # Output 테이블 선언
    out_Demand = pd.DataFrame()
    input_dataframes = {}
    try:
        if is_local:

            Version = 'CWV_DP'
            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------
            input_folder_name  = 'PYForecastVDSellOutAP1Rolling_개발요청서_0625'
            output_folder_name = 'PYForecastVDSellOutAP1Rolling_개발요청서_0625'
            # # o9에서의 소량테스트
            # input_folder_name  = 'PYForecastVDSellOutAP1Rolling_개발요청서_0625_o9_0708'
            # output_folder_name = 'PYForecastVDSellOutAP1Rolling_개발요청서_0625_o9_0708'
            # # o9에서의 대량테스트
            # input_folder_name  = 'PYForecastVDSellOutAP1Rolling_개발요청서_0625_o9_0708_8M'
            # output_folder_name = 'PYForecastVDSellOutAP1Rolling_개발요청서_0625_o9_0708_8M'
            
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
            CurrentPartialWeek = '202417A'

        logger.Note(p_note=f'Parameter Check', p_log_level=LOG_LEVEL.debug())
        logger.Note(p_note=f'Version            : {Version}', p_log_level=LOG_LEVEL.debug())
        logger.Note(p_note=f'CurrentPartialWeek : {CurrentPartialWeek}', p_log_level=LOG_LEVEL.debug())

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
            fn_log_dataframe(input_dataframes[in_df], in_df)

        current_week_normalized = _normalize_week(CurrentPartialWeek)

        # ──────────────────────────────────────────────────────────────────────────────
        # STEP-01 : VD SellOut AP1 Rolling 로직 적용 및 S/Out FCST_AP1(Rolling ADJ) Measure 생성
        # ──────────────────────────────────────────────────────────────────────────────
        dict_log = {
            'p_step_no': 100,
            'p_step_desc': 'Step 01 – VD SellOut AP1 Rolling 로직 적용 및 S/Out FCST_AP1(Rolling ADJ) Measure 생성'
        }
        df_step01_roll = fn_step01_vd_sellout_ap1_rolling(
            input_dataframes[STR_DF_IN_SOUT],
            input_dataframes[STR_DF_IN_TIME],
            CurrentPartialWeek,
            Version,
            **dict_log
        )
        fn_log_dataframe(df_step01_roll, f'step01_{STR_DF_STEP01_ROLL}')

        # ──────────────────────────────────────────────────────────────────────────────
        # STEP-02 : Output formatter
        # ──────────────────────────────────────────────────────────────────────────────
        dict_log = {
            'p_step_no'  : 900,
            'p_step_desc': 'Step 02 – Output formatter'
        }
        out_Demand = fn_output_formatter(
            df_step01_roll,                         # ← step-01 결과
            input_dataframes[STR_DF_IN_TIME],       # Time 테이블
            CurrentPartialWeek,                     # ex: '202417'
            Version,                                # ex: 'CWV_DP'
            **dict_log
        )
        fn_log_dataframe(out_Demand, STR_DF_OUT_DEMAND)

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

        # logger.info(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')
        logger.Finish()
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::') # 25.05.12 need warning Log by Logger Issue
        
        
