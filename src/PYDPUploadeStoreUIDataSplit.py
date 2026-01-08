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
from typing import Collection, Tuple,Union,Dict, Set
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
str_instance = 'PYDPUploadeStoreUIDataSplit'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"

is_print = True
flag_csv = True
flag_exception = True

# ======================================================
# 컬럼 상수 (Columns)
# ======================================================


# Version / Item / Location / Ship To / Week / Partial Week
COL_VERSION      = 'Version.[Version Name]'
COL_ITEM         = 'Item.[Item]'
COL_LOCATION     = 'Location.[Location]'
COL_SHIP_TO      = 'Sales Domain.[Ship To]'
COL_WEEK         = 'Time.[Week]'
COL_PW           = 'Time.[Partial Week]'

# (이미 있던 것들 – 필요시 유지)
COL_STD1         = 'Sales Domain.[Sales Std1]'
COL_STD2         = 'Sales Domain.[Sales Std2]'
COL_STD3         = 'Sales Domain.[Sales Std3]'
COL_STD4         = 'Sales Domain.[Sales Std4]'
COL_STD5         = 'Sales Domain.[Sales Std5]'
COL_STD6         = 'Sales Domain.[Sales Std6]'
COL_ITEM_GBM     = 'Item.[Item GBM]'
COL_ITEM_STD1    = 'Item.[Item Std1]'
COL_ITEM_STD2    = 'Item.[Item Std2]'
COL_ITEM_STD3    = 'Item.[Item Std3]'
COL_ITEM_STD4    = 'Item.[Item Std4]'

# 헤더 테이블(df_in_h)
COL_DP_DATA_SEQ          = 'DP Data Seq.[DP Data Seq]'
COL_UPLOAD_HEADER_DIM    = 'Upload Header_Dimension'
COL_UPLOAD_HEADER_MEAS   = 'Upload Header_Measure'

# 디테일 테이블(df_in_d)
COL_DP_DATA_SEQ_DETAIL   = 'DP Data Seq Detail.[DP Data Seq Detail]'
COL_UPLOAD_DATA_DATA     = 'Upload Data_Data'

# PM (AP2)
COL_SOUT_FCST_AP2            = 'S/Out FCST_AP2'
COL_SIN_FCST_GI_AP2          = 'S/In FCST(GI)_AP2'
COL_SOUT_FCST_MODIFY_AP2     = 'S/Out FCST Modify_AP2'

# KAM (AP1)
COL_SOUT_FCST_AP1            = 'S/Out FCST_AP1'
COL_SIN_FCST_GI_AP1          = 'S/In FCST(GI)_AP1'
COL_SOUT_FCST_MODIFY_AP1     = 'S/Out FCST Modify_AP1'

# ======================================================
# Ratio 컬럼 상수 (df_in_ratio)
# ======================================================
# S/In FCST(GI) Split Ratio
COL_SIN_FCST_GI_SR_AP1   = 'S/In FCST(GI) Split Ratio_AP1'
COL_SIN_FCST_GI_SR_AP2   = 'S/In FCST(GI) Split Ratio_AP2'

# S/Out FCST Split Ratio
COL_SOUT_FCST_SR_AP1     = 'S/Out FCST Split Ratio_AP1'
COL_SOUT_FCST_SR_AP2     = 'S/Out FCST Split Ratio_AP2'


# 기타
COL_SIN_FCST_GI_AP2_W0       = 'S/In FCST(GI)_AP2(W+0)'
COL_SIN_FCST_GI_AP2_W1       = 'S/In FCST(GI)_AP2(W+1)'
COL_SIN_FCST_GI_AP2_W2       = 'S/In FCST(GI)_AP2(W+2)'
COL_SIN_FCST_GI_AP2_W3       = 'S/In FCST(GI)_AP2(W+3)'
COL_SIN_FCST_GI_AP2_W4       = 'S/In FCST(GI)_AP2(W+4)'
COL_SIN_FCST_GI_AP2_W5       = 'S/In FCST(GI)_AP2(W+5)'
COL_SIN_FCST_GI_AP2_W6       = 'S/In FCST(GI)_AP2(W+6)'
COL_SIN_FCST_GI_AP2_W7       = 'S/In FCST(GI)_AP2(W+7)'
COL_SIN_FCST_GI_AP2_LONGTAIL = 'S/In FCST(GI)_AP2(Long Tail)'

COL_SOUT_FCST_AP2_ADJ        = 'S/Out FCST_AP2(ADJ)'

# ======================================================
# 데이터프레임 상수
# ======================================================
# ---------- INPUT DF KEYS ----------
DF_IN_H       = 'df_in_h'       # header 정보 (Upload Header_Dimension / Upload Header_Measure)
DF_IN_D       = 'df_in_d'       # detail 정보 (Upload Data_Data)
DF_IN_RATIO   = 'df_in_ratio'   # ratio 정보 (Split Ratio AP1/AP2)
DF_IN_WEEK    = 'df_in_week'    # Week ↔ Partial Week 매핑

# ---------- OUTPUT DF KEYS ----------
DF_OUT            = 'df_out'             # 최종 결과 (Partial Week 기준)
DF_OUT_LOG        = 'df_out_log'         # 로그 (요약)
DF_OUT_LOG_DETAIL = 'df_out_logDetail'   # 로그 (상세)

########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################
logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
# fn_set_local_logfile()
LOG_LEVEL = common.G_log_level

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
        # if is_local and not df_p_source.empty and flag_csv:
        if is_local and flag_csv:
            # 로컬 Debugging 시 csv 파일 출력
            df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)
    else:
        # 최종 Output 테이블인 경우에는 무조건 로그 출력
        if is_output:
            logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=20)
            # if is_local and not df_p_source.empty:
            if is_local:
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

def fn_convert_type(df: pd.DataFrame, startWith: str, type):
    for column in df.columns:
        if column.startswith(startWith):
            df[column] = df[column].astype(type,errors='ignore')


################################################################################################################──────────
#  공통 타입 변환  (❌ `global` 사용 금지)
#  호출 측에서 `input_dataframes` 를 인자로 넘겨준다.
################################################################################################################──────────
def fn_prepare_input_types(dict_dfs: dict) -> None:
    if not dict_dfs:
        return    

    for df_name, df in dict_dfs.items():
        if df.empty:
            continue

        # 1) object → str → category
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df[col] = df[col].astype(str).astype("category")

        # 2) 정수만 int32로, 실수는 유지
        int_cols = df.select_dtypes(include=["int64", "int32", "Int64", "Int32", "int"]).columns
        for col in int_cols:
            df[col].fillna(0, inplace=True)
            df[col] = df[col].astype("int32")

        float_cols = df.select_dtypes(include=["float64", "float32"]).columns
        # 필요 시 공통 결측 처리만. 반올림/형변환은 각 도메인 함수(가격 등)에서 수행
        for col in float_cols:
            df[col].fillna(np.nan, inplace=True)

@_decoration_
def fn_process_in_df_mst():
    """
    PYSalesProductASNDeltaB2C: 입력 DF 적재 + 타입 표준화
      - 차원(prefix): "Version.", "Sales Domain", "Item.", "Location.", "Time." → category
      - 측정치(더미/가격/환율/Ratio/AP 가격): float32 (NaN 보존)
      - 진짜 정수 코드값만 int32
    전역 dict: input_dataframes 에 적재
    """    
    
    # -----------------------------
    # 0) 파일명 ↔ DF_KEY 매핑
    #    (로컬에서 CSV를 읽을 때 파일명은 아래 키와 동일해야 함: <키>.csv)
    # -----------------------------

    file_to_df_mapping = {
        f'{DF_IN_H     }.csv': DF_IN_H,
        f'{DF_IN_D     }.csv': DF_IN_D,
        f'{DF_IN_RATIO }.csv': DF_IN_RATIO,
        f'{DF_IN_WEEK  }.csv': DF_IN_WEEK,
    }

    def read_csv_with_fallback(filepath: str) -> pd.DataFrame:
        for enc in ('utf-8-sig', 'utf-8', 'cp949'):
            try:
                return pd.read_csv(filepath, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to read file {filepath} with tried encodings.")

    # -----------------------------
    # 1) 로컬 / o9 분기
    # -----------------------------
    if is_local:
        # 출력 폴더 정리
        for file in os.scandir(str_output_dir):
            try:
                os.remove(file.path)
            except Exception:
                pass

        # 입력 CSV 적재
        for file in glob.glob(f"{os.getcwd()}/{str_input_dir}/*.csv"):
            df = read_csv_with_fallback(file)
            file_name = os.path.splitext(os.path.basename(file))[0]
            for fname, df_key in file_to_df_mapping.items():
                if file_name == os.path.splitext(fname)[0]:
                    input_dataframes[df_key] = df
                    break
    else:
        # o9 런타임: 외부에서 주입된 변수 바인딩
        input_dataframes[DF_IN_H     ] = df_in_h
        input_dataframes[DF_IN_D     ] = df_in_d
        input_dataframes[DF_IN_RATIO ] = df_in_ratio
        input_dataframes[DF_IN_WEEK  ] = df_in_week

    # -----------------------------
    # 2) 차원 컬럼: category 로 통일
    # -----------------------------
    dim_prefixes = ("Version.", "Sales Domain", "Item.", "Location.", "Time.")
    for key, df in list(input_dataframes.items()):
        if df is None or df.empty:
            continue
        # prefix 매칭되는 컬럼은 우선 str로 만든 뒤 category 로
        for p in dim_prefixes:
            fn_convert_type(df, p, str)
        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = df[c].astype(str).astype("category")

    # -----------------------------
    # 3) 측정치 컬럼: float32 로 통일(결측 보존)
    # -----------------------------
    # 더미/가격/환율/AP가격/Ratio 들을 커버하는 시작 문자열들
    meas_starts = (
        "S/In FCST",              # e.g. S/In FCST(GI) Dummy_*, S/In FCST(GI)_AP1 ...
        "S/Out FCST",             # e.g. S/Out FCST Dummy_*, S/Out FCST_AP1 ...
        "Flooring FCST",          # e.g. Flooring FCST Dummy
        "Estimated Price",        # e.g. Estimated Price Modify_Local, Estimated Price_Local
        "Exchange Rate",          # e.g. Exchange Rate_Local
        "Action Plan Price",      # e.g. Action Plan Price_USD
        "Split Ratio"             # 안전빵: 혹시 접두가 'Split Ratio' 로만 오는 경우
    )

    def _cast_measures_to_float32(df: pd.DataFrame):
        if df is None or df.empty:
            return
        cols = [c for c in df.columns if c.startswith(meas_starts)]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    for key, df in list(input_dataframes.items()):
        _cast_measures_to_float32(df)

    # -----------------------------
    # 4) 정수 코드값만 int32, float 는 유지
    #    (공통 유틸: object→category, int→int32, float는 NaN 보존)
    # -----------------------------
    fn_prepare_input_types(input_dataframes)

    # -----------------------------
    # 5) 주요 DF 간단 로그(선택)
    # -----------------------------


    
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


# ======================================================
# 공용: 차원형(category) 캐스팅 + 컬럼 순서 정리 유틸
# ======================================================
def _coerce_dims(
    df: pd.DataFrame,
    cols: list[str]
) -> None:
    for c in cols:
        if c in df.columns and df[c].dtype.name != 'category':
            df[c] = df[c].astype('category')

def _to_float32(
    df: pd.DataFrame,
    cols: list[str]
) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')


#####################################################
#################### Start Step Functions  ##########
#####################################################

# ======================================================
# Step 1) df_in_d 를 split 한다. (^ 분해 + 헤더로 컬럼명 구성)
#   입력 : df_in_h, df_in_d
#   출력 : df_step_01  (헤더 기준 컬럼명 적용된 DataFrame)
# ======================================================
@_decoration_
def fn_step01_split_upload_data(
    df_in_h: pd.DataFrame,
    df_in_d: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    df_in_d.Upload Data_Data 를 '^' 구분자로 split 하고,
    df_in_h 의 Upload Header_Dimension / Upload Header_Measure 를 이용해 컬럼명을 구성한다.
    - DP Data Seq.[DP Data Seq] 기준으로 헤더를 찾아 사용.
    """

    # # 필수 Input 체크
    # fn_check_input_table(df_in_h, DF_IN_H, '0')
    # fn_check_input_table(df_in_d, DF_IN_D, '1')

    if df_in_d.empty:
        return pd.DataFrame()

    # 컬럼 존재 여부 체크
    required_cols_d = [COL_DP_DATA_SEQ, COL_UPLOAD_DATA_DATA]
    for c in required_cols_d:
        if c not in df_in_d.columns:
            raise Exception(f'[fn_step01_split_upload_data] Required column not found in df_in_d: {c}')

    required_cols_h = [COL_DP_DATA_SEQ, COL_UPLOAD_HEADER_DIM, COL_UPLOAD_HEADER_MEAS]
    for c in required_cols_h:
        if c not in df_in_h.columns:
            raise Exception(f'[fn_step01_split_upload_data] Required column not found in df_in_h: {c}')

    list_df_parts: list[pd.DataFrame] = []

    # DP Data Seq 별로 분리 처리 (AP1, AP2 등)
    for seq_val, df_grp in df_in_d.groupby(COL_DP_DATA_SEQ, sort=False):
        df_header = df_in_h[df_in_h[COL_DP_DATA_SEQ] == seq_val]

        if df_header.empty:
            logger.Note(
                p_note=f'[fn_step01_split_upload_data] No header row for DP Data Seq = {seq_val}, skip this group.',
                p_log_level=LOG_LEVEL.warning()
            )
            continue

        # 헤더 문자열 (Dimension / Measure)
        dimension_str = str(df_header.iloc[0][COL_UPLOAD_HEADER_DIM]) if COL_UPLOAD_HEADER_DIM in df_header.columns else ''
        measure_str   = str(df_header.iloc[0][COL_UPLOAD_HEADER_MEAS]) if COL_UPLOAD_HEADER_MEAS in df_header.columns else ''

        dim_cols = dimension_str.split('^') if dimension_str else []
        mea_cols = measure_str.split('^') if measure_str else []
        target_cols = dim_cols + mea_cols

        if not target_cols:
            logger.Note(
                p_note=f'[fn_step01_split_upload_data] Empty header (dimension/measure) for DP Data Seq = {seq_val}, skip.',
                p_log_level=LOG_LEVEL.warning()
            )
            continue

        # Upload Data_Data split
        s = df_grp[COL_UPLOAD_DATA_DATA].fillna('')
        df_split = s.astype(str).str.split('^', expand=True)

        n_target = len(target_cols)
        n_actual = df_split.shape[1]

        # 컬럼 개수 맞추기 (부족하면 빈 문자열로 채우고, 넘치면 잘라냄)
        if n_actual < n_target:
            for i in range(n_target - n_actual):
                df_split[n_actual + i] = ''
        elif n_actual > n_target:
            df_split = df_split.iloc[:, :n_target]

        df_split.columns = target_cols

        list_df_parts.append(df_split)

    if not list_df_parts:
        return pd.DataFrame(columns=[
            COL_VERSION, COL_ITEM, COL_LOCATION, COL_SHIP_TO, COL_WEEK
        ])

    df_step_01 = pd.concat(list_df_parts, ignore_index=True)

    # 차원 컬럼은 category 로 캐스팅
    _coerce_dims(
        df_step_01,
        [COL_VERSION, COL_ITEM, COL_LOCATION, COL_SHIP_TO, COL_WEEK]
    )

    # --------------------------------------------------
    # Upload Header_Measure 에 해당하는 컬럼들을 float32 로 캐스팅
    # --------------------------------------------------
    # df_in_h 는 이미 현재 DP Data Seq 로 필터된 상태라고 가정
    # (여러 row 인 경우 첫 번째 row 사용)
    measure_str = df_in_h.iloc[0]['Upload Header_Measure']
    measure_cols = measure_str.split('^')

    for col in measure_cols:
        if col in df_step_01.columns:
            df_step_01[col] = (
                pd.to_numeric(df_step_01[col], errors='coerce')
                  .astype('float32')
            )

    return df_step_01



# ======================================================
# Step 2) df_step_01 전처리 – Version 컬럼 삭제
#   입력 : df_step_01
#   출력 : df_step_02  (Version.[Version Name] 제거)
# ======================================================
@_decoration_
def fn_step02_drop_version(
    df_step_01: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Step 1 결과에서 [Version].[Version Name] 컬럼을 삭제한다.
    """

    if df_step_01 is None or df_step_01.empty:
        return df_step_01.copy() if df_step_01 is not None else pd.DataFrame()

    df_step_02 = df_step_01.copy()

    if COL_VERSION in df_step_02.columns:
        df_step_02 = df_step_02.drop(columns=[COL_VERSION])

    # 차원 컬럼 재정리 (Version 제거 후 나머지)
    _coerce_dims(
        df_step_02,
        [COL_ITEM, COL_LOCATION, COL_SHIP_TO, COL_WEEK]
    )

    return df_step_02

# ======================================================
# Step 3) Ratio 처리 (FCST)
#   - 대상 컬럼 (있는 경우만 처리)
#       S/Out FCST_AP2
#       S/In FCST(GI)_AP2
#       S/Out FCST_AP1
#       S/In FCST(GI)_AP1
#
#   - df_in_week 로 Week ↔ Partial Week 확장
#   - df_in_ratio 에서 ratio 를 찾은 경우:
#       • partial week 1개 → 아무 작업 안 함 (원 값 유지)
#       • partial week 2개 이상 → ratio 비율대로 분배 + 소수점 처리
#   - df_in_ratio 에서 값을 찾지 못한 경우:
#       • partial week 1개 → 아무 작업 안 함
#       • partial week 2개 이상 → 값 n분의1 로 동일 분배
#           (주로 2개이므로 50:50, 소수점은 동일 규칙으로 반올림)
#
#   - target column 들은 최종적으로 int32 로 캐스팅
# ======================================================
@_decoration_
def fn_step03_apply_ratio_fcst(
    df_step_02: pd.DataFrame,
    df_in_week: pd.DataFrame,
    df_in_ratio: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:

    # --------------------------------------------------
    # Input 체크
    # --------------------------------------------------
    # fn_check_input_table(df_step_02, 'df_step_02', '1')
    # fn_check_input_table(df_in_week, 'df_in_week', '1')
    # fn_check_input_table(df_in_ratio, 'df_in_ratio', '1')

    if df_step_02 is None or df_step_02.empty:
        return pd.DataFrame()

    # --------------------------------------------------
    # Week ↔ Partial Week 매핑 생성
    # --------------------------------------------------
    df_week_map = df_in_week[[COL_WEEK, COL_PW]].drop_duplicates()

    # Step2 결과에 Partial Week 붙여서 row 확장
    df_expanded = df_step_02.merge(
        df_week_map,
        on=COL_WEEK,
        how='left'
    )

    # ratio 테이블에도 Week 붙이기
    df_ratio = df_in_ratio.merge(
        df_week_map,
        on=COL_PW,
        how='left'
    )

    # --------------------------------------------------
    # 대상 FCST ↔ Ratio 컬럼 매핑
    # --------------------------------------------------
    target_fcst_cols = [
        COL_SOUT_FCST_AP2,
        COL_SIN_FCST_GI_AP2,
        COL_SOUT_FCST_AP1,
        COL_SIN_FCST_GI_AP1,
    ]
    target_fcst_cols = [c for c in target_fcst_cols if c in df_expanded.columns]

    if not target_fcst_cols:
        logger.Note(
            p_note='[fn_step03_apply_ratio_fcst] No target FCST columns found in df_step_02. Return expanded as-is.',
            p_log_level=LOG_LEVEL.warning()
        )
        # Partial Week 만 붙인 결과 리턴
        df_expanded = df_expanded.copy()
        _coerce_dims(df_expanded, [COL_ITEM, COL_LOCATION, COL_SHIP_TO, COL_WEEK, COL_PW])
        return df_expanded

    map_fcst_to_ratio = {
        COL_SOUT_FCST_AP2:   COL_SOUT_FCST_SR_AP2,
        COL_SIN_FCST_GI_AP2: COL_SIN_FCST_GI_SR_AP2,
        COL_SOUT_FCST_AP1:   COL_SOUT_FCST_SR_AP1,
        COL_SIN_FCST_GI_AP1: COL_SIN_FCST_GI_SR_AP1,
    }

    # ratio 테이블 필수 컬럼만 남기기
    ratio_needed_cols = [
        COL_ITEM,
        COL_LOCATION,
        COL_SHIP_TO,
        COL_WEEK,
        COL_PW,
    ] + list(set(map_fcst_to_ratio.values()))

    df_ratio = df_ratio[ratio_needed_cols].copy()

    # --------------------------------------------------
    # 그룹 기준 컬럼 (Week 레벨)
    # --------------------------------------------------
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK]

    df_result = df_expanded.copy()

    # --------------------------------------------------
    # 그룹별 분배 함수 (ratio 존재/부재 모두 처리)
    # --------------------------------------------------
    def _apply_ratio_or_equal_split(group: pd.DataFrame, fcst_col: str, ratio_col: str) -> pd.DataFrame:
        """
        group: 동일 (ShipTo, Item, Location, Week) 내의 여러 Partial Week 행
        fcst_col: 대상 FCST 컬럼
        ratio_col: 대응 Ratio 컬럼
        """
        base_vals = pd.to_numeric(group[fcst_col], errors='coerce')
        # base 값: 동일 Week 내에서 하나라고 가정 → 첫 번째 유효값
        if base_vals.notna().any():
            base_val = float(base_vals.dropna().iloc[0])
        else:
            # 전부 NaN 이면 그대로 NaN 유지
            group[fcst_col] = np.nan
            return group

        # base 값이 0 이거나 음수인 경우: 분배 의미가 없으므로 그대로 (0 또는 그 값) 복제
        if base_val == 0:
            group[fcst_col] = 0
            return group

        n = len(group)
        # partial week 가 하나인 경우: 아무런 행동을 하지 않음 (원값 그대로)
        if n == 1:
            group[fcst_col] = int(round(base_val))
            return group

        # ratio 값 가져오기
        r = pd.to_numeric(group[ratio_col], errors='coerce')
        r = r.fillna(0.0).to_numpy(dtype='float64')

        sum_r = r.sum()
        if sum_r > 0:
            # ratio 를 정규화해서 사용
            ratio_vec = r / sum_r
        else:
            # ★ df_in_ratio 에서 값을 찾지 못한 경우
            #   - partial week 2개인 경우: 50:50
            #   - 2개 이상인 경우: n분의1 로 균등분배
            ratio_vec = np.full(n, 1.0 / n, dtype='float64')

        # 연속값 분배
        raw_vals = base_val * ratio_vec

        # --------------------------------------------------
        # 소수점 처리
        #   - 소수점 부분 > 0.5 → 올림(ceil)
        #   - 소수점 부분 < 0.5 → 내림(floor)
        #   - (0.5 인 경우는 현실적으로 드물지만, 합계 보존을 위해 올림 쪽에 포함)
        # --------------------------------------------------
        floor_vals = np.floor(raw_vals)
        frac = raw_vals - floor_vals

        base_int = int(round(base_val))
        floor_sum = int(floor_vals.sum())
        diff = base_int - floor_sum

        # diff > 0 인 만큼, frac 이 큰 순서대로 +1
        idx_order = np.argsort(-frac)  # 내림차순
        adjusted = floor_vals.copy()

        for i in range(len(adjusted)):
            if diff <= 0:
                break
            adjusted[idx_order[i]] += 1
            diff -= 1

        # 혹시 diff < 0 이 되는 케이스는 매우 드물지만 방어적으로 처리
        # (이 경우 frac 이 작은 순서대로 -1)
        if diff < 0:
            idx_order_rev = np.argsort(frac)  # frac 작은 순
            for i in range(len(adjusted)):
                if diff >= 0:
                    break
                if adjusted[idx_order_rev[i]] > 0:
                    adjusted[idx_order_rev[i]] -= 1
                    diff += 1

        group[fcst_col] = adjusted.astype('int32')
        return group

    # --------------------------------------------------
    # FCST 컬럼별로 분배 적용
    # --------------------------------------------------
    for fcst_col in target_fcst_cols:
        ratio_col = map_fcst_to_ratio.get(fcst_col)
        if ratio_col is None or ratio_col not in df_ratio.columns:
            logger.Note(
                p_note=f'[fn_step03_apply_ratio_fcst] Ratio column for {fcst_col} not found. Skip.',
                p_log_level=LOG_LEVEL.warning()
            )
            continue

        # df_result + ratio join
        df_tmp = df_result[key_cols + [COL_PW, fcst_col]].merge(
            df_ratio[key_cols + [COL_PW, ratio_col]],
            on=key_cols + [COL_PW],
            how='left'
        )

        # 그룹별 분배 적용
        df_tmp = (
            df_tmp
            .groupby(key_cols, group_keys=False)
            .apply(lambda g: _apply_ratio_or_equal_split(g, fcst_col, ratio_col))
        )

        # 분배 결과를 df_result 에 반영
        df_result[fcst_col] = df_tmp[fcst_col].values

    # 차원형 컬럼 category 캐스팅
    _coerce_dims(df_result, [COL_ITEM, COL_LOCATION, COL_SHIP_TO, COL_WEEK, COL_PW])

    # 최종 결과 리턴
    return df_result


# ======================================================
# Step 4) Modify 컬럼 후처리
#   - 대상 컬럼(있는 경우만 처리)
#       S/Out FCST Modify_AP2
#       S/Out FCST Modify_AP1
#   - 여기서는 값을 그대로 사용하되,
#       • 원래 값이 NaN → NaN 유지
#       • 값이 0        → NaN 으로 변환
#   - df_in_ratio, df_in_week 는 사용하지 않음
# ======================================================
@_decoration_
def fn_step04_cleanup_modify(
    df_step_03: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Step 3 결과에서 Modify 컬럼(AP1/AP2)에 대해
      - 0 값을 NaN 으로 바꾸고
      - NaN 은 그대로 유지한다.
    Ratio / Week 정보는 사용하지 않는다.
    """

    # fn_check_input_table(df_step_03, 'df_step_03', '1')

    if df_step_03 is None or df_step_03.empty:
        return pd.DataFrame()

    modify_target_cols = [
        COL_SOUT_FCST_MODIFY_AP2,
        COL_SOUT_FCST_MODIFY_AP1
    ]
    present_modify_cols = [c for c in modify_target_cols if c in df_step_03.columns]

    if not present_modify_cols:
        logger.Note(
            p_note='[fn_step04_cleanup_modify] No Modify target columns found in df_step_03. Return as-is.',
            p_log_level=LOG_LEVEL.warning()
        )
        return df_step_03.copy()

    df_step_04 = df_step_03.copy()

    for modify_col in present_modify_cols:
        # 숫자로 변환 (NaN 유지)
        vals = pd.to_numeric(df_step_04[modify_col], errors='coerce')

        # 값이 0 인 경우 NaN 으로 변환
        zero_mask = vals == 0
        vals[zero_mask] = np.nan

        # float32 로 캐스팅 (NaN 유지)
        df_step_04[modify_col] = vals.astype('float32')

    # 차원형 컬럼 category 캐스팅 (필요 시)
    _coerce_dims(
        df_step_04,
        [COL_ITEM, COL_LOCATION, COL_SHIP_TO, COL_WEEK, COL_PW]
    )

    return df_step_04


# ======================================================
# Step 5) Version 추가
#   - Step 4 결과에 Version 컬럼을 맨 앞에 추가
# ======================================================
@_decoration_
def fn_step05_add_version(
    df_step_04: pd.DataFrame,
    Version: str,
    **kwargs
) -> pd.DataFrame:
    """
    Step 4 결과에 Version 컬럼을 추가하고, 컬럼 순서를 정리한다.
    - COL_VERSION 컬럼 값은 모두 입력 파라미터 Version 으로 세팅
    """

    if df_step_04 is None or df_step_04.empty:
        return pd.DataFrame()

    df_out = df_step_04.copy()

    # Version 컬럼 추가/세팅
    if COL_VERSION in df_out.columns:
        df_out[COL_VERSION] = Version
    else:
        df_out.insert(0, COL_VERSION, Version)

    # 차원형 컬럼 category 캐스팅
    _coerce_dims(
        df_out,
        [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK, COL_PW]
    )

    # 컬럼 순서 정리
    #   row axis 참고:
    #   [Version].[Version Name]
    #   * [Sales Domain].[Ship To]
    #   * [Item].[Item]
    #   * [Location].[Location]
    #   * [Time].[Partial Week]
    dim_order = [
        COL_VERSION,
        COL_SHIP_TO,
        COL_ITEM,
        COL_LOCATION,
        COL_PW,
        COL_WEEK,  # Week 는 있으면 뒤에 붙인다
    ]
    dim_existing = [c for c in dim_order if c in df_out.columns]
    other_cols = [c for c in df_out.columns if c not in dim_existing]

    df_out = df_out[dim_existing + other_cols]

    return df_out


####################################
############ Start Main  ###########
####################################
if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    input_dataframes = {}
    try:
        ################################################################################################################
        # 전처리 : 모듈 내에서 사용될 데이터에 대한 정합성 체크 및 데이터 선 가공
        ################################################################################################################
        
        if is_local:
            Version = 'CWV_DP'
            # 파라메터추가 2025.11.07
            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------

            # input_folder_name  = str_instance       
            input_folder_name  = "PYDPUploadeStoreUIDataSplit"     
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
            # CurrentPartialWeek = '202447A'

        # --------------------------------------------------------------------------    
        # vdLog 초기화
        # --------------------------------------------------------------------------
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


        ############ To do : 여기 아래에 Step Function 들을 Call 하는 코드 구현. ########

        ################################################################################################################
        # Step 01 – df_in_d 를 '^' 로 split + df_in_h 헤더로 컬럼명 구성
        ################################################################################################################
        dict_log = {
            'p_step_no': 1,
            'p_step_desc': 'Step 01 – split df_in_d by ^ using df_in_h header',
            'p_df_name': 'df_step_01_split'
        }
        df_step_01 = fn_step01_split_upload_data(
            input_dataframes[DF_IN_H],
            input_dataframes[DF_IN_D],
            **dict_log
        )
        # fn_log_dataframe(df_step_01, 'df_step_01')

        ################################################################################################################
        # Step 02 – Version.[Version Name] 컬럼 삭제
        ################################################################################################################
        dict_log = {
            'p_step_no': 2,
            'p_step_desc': 'Step 02 – drop Version.[Version Name] from Step 1 result',
            'p_df_name': 'df_step_02_preprocessed'
        }
        df_step_02 = fn_step02_drop_version(
            df_step_01,
            **dict_log
        )
        # fn_log_dataframe(df_step_02, 'df_step_02')

        # (임시) 현재는 Step 2 결과를 df_out 으로 설정
        # 이후 Step 3, Step 4 구현 시 여기서 df_out 을 교체할 예정.
        df_out = df_step_02

        ################################################################################################################
        # Step 03 – FCST Ratio 처리 (AP1/AP2)
        ################################################################################################################
        dict_log = {
            'p_step_no': 3,
            'p_step_desc': 'Step 03 – apply Split Ratio to FCST(AP1/AP2)',
            'p_df_name': 'df_step_03_fcst_ratio'
        }
        df_step_03 = fn_step03_apply_ratio_fcst(
            df_step_02,
            input_dataframes[DF_IN_WEEK],
            input_dataframes[DF_IN_RATIO],
            **dict_log
        )

        ################################################################################################################
        # Step 04 – Modify 컬럼 후처리 (0 → NaN)
        ################################################################################################################
        dict_log = {
            'p_step_no': 4,
            'p_step_desc': 'Step 04 – cleanup Modify(AP1/AP2) columns (0 → NaN)',
            'p_df_name': 'df_step_04_modify_cleanup'
        }
        df_step_04 = fn_step04_cleanup_modify(
            df_step_03,
            **dict_log
        )

        ################################################################################################################
        # Step 05 – Version 컬럼 추가
        ################################################################################################################
        dict_log = {
            'p_step_no': 5,
            'p_step_desc': 'Step 05 – add Version column and finalize df_out',
            'p_df_name': 'df_out'
        }
        df_out = fn_step05_add_version(
            df_step_04,
            Version,
            **dict_log
        )

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

        df_out_log = vdCommon.gfn_getLog()
        df_out_logDetail = vdCommon.gfn_getLogDetail()
        
        logger.Finish()
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')
        