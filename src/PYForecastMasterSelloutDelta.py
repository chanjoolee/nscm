import os
import sys
import json
import shutil
import io
import zipfile
import time
import datetime
import inspect
import traceback
from typing import Dict, Tuple, List, Optional

import pandas as pd
import numpy as np
import gc

from NSCMCommon import NSCMCommon as common
from NSCMCommon import VDCommon as vdCommon
import glob
from pandas.api.types import CategoricalDtype

########################################################################################################################
# Global / Instance 설정
########################################################################################################################
is_local = common.gfn_get_isLocal()
str_instance = 'PYForecastMasterSelloutDelta'

is_print = True
flag_csv = True
flag_exception = True

########################################################################################################################
# 컬럼 상수
########################################################################################################################
COL_VERSION = 'Version.[Version Name]'
COL_ITEM    = 'Item.[Item]'
COL_PG      = 'Item.[Product Group]'
COL_SHIP_TO = 'Sales Domain.[Ship To]'
COL_STD1    = 'Sales Domain.[Sales Std1]'
COL_STD2    = 'Sales Domain.[Sales Std2]'
COL_STD3    = 'Sales Domain.[Sales Std3]'
COL_STD4    = 'Sales Domain.[Sales Std4]'
COL_STD5    = 'Sales Domain.[Sales Std5]'
COL_STD6    = 'Sales Domain.[Sales Std6]'
COL_LOC     = 'Location.[Location]'

# ASN / Forecast Rule
COL_ASN_FLAG   = 'Sales Product ASN'
COL_FR_GC      = 'FORECAST_RULE GC FCST'
COL_FR_AP2     = 'FORECAST_RULE AP2 FCST'
COL_FR_AP1     = 'FORECAST_RULE AP1 FCST'
COL_FR_AP0     = 'FORECAST_RULE AP0 FCST'

# Sell-In / Sell-Out Assortment 수량 컬럼
COL_SOUT_ASS_FLAG  = 'S/Out Assortment Flag'

COL_SOUT_ASS_GC    = 'S/Out FCST Assortment_GC'
COL_SOUT_ASS_AP2   = 'S/Out FCST Assortment_AP2'
COL_SOUT_ASS_AP1   = 'S/Out FCST Assortment_AP1'
COL_SOUT_ASS_LOCAL = 'S/Out FCST Assortment_Local'

COL_SIN_ASS_GC     = 'S/In FCST Assortment_GC'
COL_SIN_ASS_AP2    = 'S/In FCST Assortment_AP2'
COL_SIN_ASS_AP1    = 'S/In FCST Assortment_AP1'
COL_SIN_ASS_LOCAL  = 'S/In FCST Assortment_Local'

# Ship-to LV 코드
COL_LV_CODE = 'LV_CODE'

# Sell-Out Simul Master 관련
COL_MASTER_STATUS         = 'S/Out Master Status'
COL_MASTER_STATUS_DELTA   = 'S/Out Master Status Delta'
COL_MASTER_CUTOFF         = 'S/Out Master Cutoff'
COL_MASTER_CUTOFF_DELTA   = 'S/Out Master Cutoff Delta'

# ----------------------------------------------------------------
# df_in_Sell_Out_Simul_Master & Delta 관련 컬럼 (2025-12-18 Spec 변경)
# ----------------------------------------------------------------

# [2025-12-18 Spec 변경] Delta Init/Up Time 추가
COL_MASTER_INIT_TIME_DELTA = 'S/Out Master Delta Init Time'
COL_MASTER_UP_TIME_DELTA   = 'S/Out Master Delta Up Time'
COL_MASTER_INIT_TIME       = 'S/Out Master Init Time'
COL_MASTER_UP_TIME         = 'S/Out Master Up Time'




########################################################################################################################
# DF Name 상수
########################################################################################################################
# Input DF
STR_DF_IN_SALES_DOMAIN_DIM        = 'df_in_Sales_Domain_Dimension'
STR_DF_IN_ESTORE                  = 'df_in_Sales_Domain_Estore'
STR_DF_IN_SALES_PRODUCT_ASN       = 'df_in_Sales_Product_ASN'
STR_DF_IN_FORECAST_RULE           = 'df_in_Forecast_Rule'
STR_DF_IN_ITEM_MASTER             = 'df_in_Item_Master'
STR_DF_IN_SELL_OUT_SIMUL_MASTER   = 'df_in_Sell_Out_Simul_Master'
STR_DF_IN_SELL_OUT_SIMUL_MASTER_D = 'df_in_Sell_Out_Simul_Master_Delta'

# 내부/중간 DF
STR_DF_FN_SALES_PRODUCT_ASN       = 'df_fn_Sales_Product_ASN'
STR_DF_FN_SHIPTO_DIM              = 'df_fn_shipto_dim'

# Output DF 이름
STR_DF_OUT_SIN_GC                 = 'Output_SIn_Assortment_GC'
STR_DF_OUT_SIN_AP2                = 'Output_SIn_Assortment_AP2'
STR_DF_OUT_SIN_AP1                = 'Output_SIn_Assortment_AP1'
STR_DF_OUT_SIN_LOCAL              = 'Output_SIn_Assortment_Local'

STR_DF_OUT_SOUT_GC                = 'Output_SOut_Assortment_GC'
STR_DF_OUT_SOUT_AP2               = 'Output_SOut_Assortment_AP2'
STR_DF_OUT_SOUT_AP1               = 'Output_SOut_Assortment_AP1'
STR_DF_OUT_SOUT_LOCAL             = 'Output_SOut_Assortment_Local'

STR_DF_OUT_SIMUL_MASTER           = 'Output_Sell_Out_Simul_Master'
STR_DF_OUT_SIMUL_MASTER_DELTA     = 'Output_Sell_Out_Simul_Master_Delta'

########################################################################################################################
# Logger 설정
########################################################################################################################
logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
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
    Input Table 체크
      - str_p_cond = '0' → 비어 있으면 Exception
      - str_p_cond = '1' → 비어 있으면 Warning
    """
    logger.PrintDF(
        p_df=df_p_source,
        p_df_name=str_p_source_name,
        p_log_level=LOG_LEVEL.debug(),
        p_format=1,
        p_row_num=20
    )

    if df_p_source.empty:
        if str_p_cond == '0':
            raise Exception(f'[Exception] Input table({str_p_source_name}) is empty.')
        else:
            logger.Note(
                p_note=f'Input table({str_p_source_name}) is empty.',
                p_log_level=LOG_LEVEL.warning()
            )


# ───────────────────────────────────────────────────────────────
# Typed casting helper
# ───────────────────────────────────────────────────────────────
def cast_cols(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    int_cols: Optional[List[str]] = None,
    bool_cols: Optional[List[str]] = None,
) -> None:
    """
    In-place dtype casting  
      • cat_cols → category  
      • int_cols → Int32  
      • bool_cols → boolean
    """
    if df is None or df.empty:
        return

    if cat_cols:
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype(str).astype('category')
    if int_cols:
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int32')
    if bool_cols:
        for c in bool_cols:
            if c in df.columns:
                df[c] = df[c].astype('boolean', errors='ignore')


########################################################################################################################
# Ultra-fast groupby (일반 버전) – CreateSellInAndSellOutAssortment에서 사용한 버전 재사용
########################################################################################################################
def ultra_fast_groupby_numpy_general(
    df: pd.DataFrame,
    key_cols: List[str],
    aggs: Dict[str, str],
    *,
    cast_key_to_category: bool = True,
    treat_YN_as_bool: bool = True,
) -> pd.DataFrame:
    """
    NumPy 기반 초고속 groupby (lexsort + reduceat 패턴)
    지원 집계: 'sum','max','min','any','all','first','last','count'
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[*key_cols, *aggs.keys()])

    key_vals: List[np.ndarray] = []
    key_codes: List[np.ndarray] = []

    for col in key_cols:
        s = df[col]
        if s.dtype.name != 'category':
            s = s.astype('category')
        key_vals.append(s.to_numpy())
        key_codes.append(s.cat.codes.to_numpy(np.int64))

    order = np.lexsort(tuple(key_codes[::-1]))
    if order.size == 0:
        return pd.DataFrame(columns=[*key_cols, *aggs.keys()])

    codes_sorted = np.vstack([kc[order] for kc in key_codes]).T
    change = np.any(np.diff(codes_sorted, axis=0) != 0, axis=1)
    first_idx_sorted = np.concatenate(([0], np.flatnonzero(change) + 1))
    end_idx_sorted   = np.empty_like(first_idx_sorted)
    end_idx_sorted[:-1] = first_idx_sorted[1:]
    end_idx_sorted[-1]  = codes_sorted.shape[0]

    rep_rows_sorted = order[first_idx_sorted]
    rep_order = np.argsort(rep_rows_sorted, kind='mergesort')
    rep_rows  = rep_rows_sorted[rep_order]

    result = pd.DataFrame({col: key_vals[i][rep_rows] for i, col in enumerate(key_cols)})

    def _as_ndarray(x):
        return x.to_numpy() if isinstance(x, pd.Series) else np.asarray(x)

    def _is_numeric_dtype(dt) -> bool:
        return (
            np.issubdtype(dt, np.integer)
            or np.issubdtype(dt, np.unsignedinteger)
            or np.issubdtype(dt, np.floating)
            or np.issubdtype(dt, np.bool_)
        )

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

    for tgt_col, how in aggs.items():
        how_l = how.lower()
        col_sorted = _as_ndarray(df[tgt_col])[order]

        restore_YN = False
        if how_l in ('any', 'all', 'max', 'min'):
            if np.issubdtype(col_sorted.dtype, np.bool_):
                arr = col_sorted.view(np.int8)
            elif treat_YN_as_bool and _is_YN_array(col_sorted):
                arr = (col_sorted == 'Y').astype(np.int8, copy=False)
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

        out_vals = out_vals_sorted[rep_order]

        if restore_YN:
            out_vals = np.where(out_vals >= 1, 'Y', 'N')

        result[tgt_col] = out_vals

    if cast_key_to_category and not result.empty:
        result[key_cols] = result[key_cols].astype('category')

    return result



def is_categorical_dtype(arr_or_dtype) -> bool:
    """
    pandas 객체 / dtype 이 categorical 타입인지 여부를 확인하는 헬퍼 함수.
    - Series, Index, numpy dtype, 문자열 모두 안전하게 처리
    """

    # 1) Series / Index 인 경우 → dtype 먼저 꺼냄
    if hasattr(arr_or_dtype, "dtype"):
        dtype = arr_or_dtype.dtype
    else:
        dtype = arr_or_dtype

    # 2) 이미 CategoricalDtype 인 경우
    if isinstance(dtype, CategoricalDtype):
        return True

    # 3) 문자열 'category' 같은 경우
    if str(dtype) == "category":
        return True

    # 4) 나머지는 pandas_dtype로 한 번 더 판단 (dtype-like만 들어감)
    try:
        dtype2 = pd.api.types.pandas_dtype(dtype)
    except TypeError:
        return False
    return isinstance(dtype2, CategoricalDtype) or str(dtype2) == "category"

########################################################################################################################
# STEP 0 – Sell_Out_Simul_Master Delta 처리
########################################################################################################################

# ======================================================
# Step 0-1) Sell_Out_Simul_Master Delta 반영
#   - df_in_Sell_Out_Simul_Master 에 Delta Status 를 반영
#   반환: df_step00_01_Sell_Out_Simul_Master
# ======================================================
@_decoration_
def fn_step00_01_apply_simul_delta(
    df_master: pd.DataFrame,
    df_delta: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Sell-Out Simul Master 에 Delta 의 Status 를 반영
      key: (Version, ShipTo, ProductGroup) 존재 시 Version 포함, 없으면 (ShipTo, ProductGroup)
    """
    if df_master is None or df_master.empty or df_delta is None or df_delta.empty:
        # Delta 없으면 원본 그대로 사용
        return df_master.copy() if df_master is not None else pd.DataFrame()

    df_m = df_master.copy()
    df_d = df_delta.copy()

    has_ver = (COL_VERSION in df_m.columns) and (COL_VERSION in df_d.columns)
    key_cols = [COL_SHIP_TO, COL_PG]
    if has_ver:
        key_cols = [COL_VERSION, COL_SHIP_TO, COL_PG]

    use_cols_delta = [c for c in key_cols + [COL_MASTER_STATUS_DELTA] if c in df_d.columns]
    df_d = df_d[use_cols_delta].copy()

    # merge로 Delta Status 반영
    df_merged = df_m.merge(
        df_d,
        on=key_cols,
        how='left',
        suffixes=('', '_delta')
    )

    if COL_MASTER_STATUS_DELTA in df_merged.columns:
        # 1) Delta 쪽에 값이 있는 row만 업데이트 대상
        mask = df_merged[COL_MASTER_STATUS_DELTA].notna()
        # 2) 만약 원본에 Status 컬럼이 없다면 생성
        if COL_MASTER_STATUS not in df_merged.columns:
            df_merged[COL_MASTER_STATUS] = pd.NA
        # 3) 둘 다 카테고리일 경우, 카테고리 unon 후 재설정
        if (
            is_categorical_dtype(df_merged[COL_MASTER_STATUS]) and
            is_categorical_dtype(df_merged[COL_MASTER_STATUS_DELTA])
        ):
            union_cats = df_merged[COL_MASTER_STATUS].cat.categories.union(
                df_merged[COL_MASTER_STATUS_DELTA].cat.categories
            )
            df_merged[COL_MASTER_STATUS] = df_merged[COL_MASTER_STATUS].cat.set_categories(union_cats)
            df_merged[COL_MASTER_STATUS_DELTA] = df_merged[COL_MASTER_STATUS_DELTA].cat.set_categories(union_cats)
        # 4) 실제 값 덮어쓰기
        df_merged.loc[mask, COL_MASTER_STATUS] = df_merged.loc[mask, COL_MASTER_STATUS_DELTA]
        # 5) Delta 컬럼은 더 이상 필요 없으니 삭제v
        df_merged.drop(columns=[COL_MASTER_STATUS_DELTA], inplace=True)

    return df_merged


# ======================================================
# Step 0-2) Sell_Out_Simul_Master Output 생성
#   - [2025-12-18 Spec 변경]
#     Delta 테이블(df_in_Sell_Out_Simul_Master_Delta)의 4개 Measure 이름을
#     본 Measure 이름으로 변경하여 Output_Sell_Out_Simul_Master 생성
#
#   입력 예시:
#     Version.[Version Name],Sales Domain.[Ship To],Item.[Product Group],
#     S/Out Master Cutoff Delta,S/Out Master Status Delta,
#     S/Out Master Delta Init Time,S/Out Master Delta Up Time
#     CWV_DP,CH018113,REF,TEST,TEST,TEST,TEST
#
#   결과:
#     Version.[Version Name],Sales Domain.[Ship To],Item.[Product Group],
#     S/Out Master Cutoff,S/Out Master Status,
#     S/Out Master Init Time,S/Out Master Up Time
#     CWV_DP,CH018113,REF,TEST,TEST,TEST,TEST
# ======================================================
@_decoration_
def fn_step00_02_build_output_simul_master(
    df_delta: pd.DataFrame,
    out_version: str,
    **kwargs
) -> pd.DataFrame:
    """
    Output_Sell_Out_Simul_Master (Delta row만 포함)

    [역할]
      - Delta 입력(df_in_Sell_Out_Simul_Master_Delta)의 Measure 이름을
        "정식 Master Measure" 이름으로 바꿔서 출력용 테이블을 만든다.
      - 값은 그대로 유지한다. (Delta 테이블이 사실상 Master의 최신 상태를 들고 있음)
      - 컬럼 구성:
          Version, ShipTo, Product Group,
          S/Out Master Cutoff, S/Out Master Status,
          S/Out Master Init Time, S/Out Master Up Time
    """

    # ─────────────────────────────────────────────────────
    # 0) 입력이 비어 있는 경우: 스키마만 맞춰서 빈 DF 반환
    # ─────────────────────────────────────────────────────
    if df_delta is None or df_delta.empty:
        return pd.DataFrame(
            columns=[
                COL_VERSION,
                COL_SHIP_TO,
                COL_PG,
                COL_MASTER_CUTOFF,
                COL_MASTER_STATUS,
                COL_MASTER_INIT_TIME,
                COL_MASTER_UP_TIME,
            ]
        )

    # ─────────────────────────────────────────────────────
    # 1) 사용할 컬럼 리스트 구성
    #    - df_delta 에 실제로 존재하는 컬럼만 사용
    # ─────────────────────────────────────────────────────
    use_cols: list[str] = []
    for c in [
        COL_VERSION,
        COL_SHIP_TO,
        COL_PG,
        COL_MASTER_CUTOFF_DELTA,
        COL_MASTER_STATUS_DELTA,
        COL_MASTER_INIT_TIME_DELTA,   # [2025-12-18 Spec 변경] 추가
        COL_MASTER_UP_TIME_DELTA,     # [2025-12-18 Spec 변경] 추가
    ]:
        if c in df_delta.columns:
            use_cols.append(c)

    # 최소한 키 3개는 있어야 정상 Output 생성 가능
    mandatory = [COL_VERSION, COL_SHIP_TO, COL_PG]
    if any(c not in use_cols for c in mandatory):
        return pd.DataFrame(
            columns=[
                COL_VERSION,
                COL_SHIP_TO,
                COL_PG,
                COL_MASTER_CUTOFF,
                COL_MASTER_STATUS,
                COL_MASTER_INIT_TIME,
                COL_MASTER_UP_TIME,
            ]
        )

    # ─────────────────────────────────────────────────────
    # 2) 사용 컬럼만 복사 (view 이슈 방지)
    # ─────────────────────────────────────────────────────
    df_out = df_delta[use_cols].copy()

    # ─────────────────────────────────────────────────────
    # 3) Delta 컬럼 → 본 Measure 컬럼으로 rename
    #    - [2025-12-18 Spec 변경] Init/Up Time 추가
    # ─────────────────────────────────────────────────────
    rename_map: dict[str, str] = {}

    if COL_MASTER_CUTOFF_DELTA in df_out.columns:
        rename_map[COL_MASTER_CUTOFF_DELTA] = COL_MASTER_CUTOFF
    if COL_MASTER_STATUS_DELTA in df_out.columns:
        rename_map[COL_MASTER_STATUS_DELTA] = COL_MASTER_STATUS
    if COL_MASTER_INIT_TIME_DELTA in df_out.columns:
        rename_map[COL_MASTER_INIT_TIME_DELTA] = COL_MASTER_INIT_TIME
    if COL_MASTER_UP_TIME_DELTA in df_out.columns:
        rename_map[COL_MASTER_UP_TIME_DELTA] = COL_MASTER_UP_TIME

    if rename_map:
        df_out.rename(columns=rename_map, inplace=True)

    # ─────────────────────────────────────────────────────
    # 4) Version 값 통일 (out_version 로 강제 세팅)
    # ─────────────────────────────────────────────────────
    df_out[COL_VERSION] = out_version

    # ─────────────────────────────────────────────────────
    # 5) 컬럼 순서 정렬 + 누락 컬럼 보완
    # ─────────────────────────────────────────────────────
    col_order = [
        COL_VERSION,
        COL_SHIP_TO,
        COL_PG,
        COL_MASTER_CUTOFF,
        COL_MASTER_STATUS,
        COL_MASTER_INIT_TIME,
        COL_MASTER_UP_TIME,
    ]

    for c in col_order:
        if c not in df_out.columns:
            df_out[c] = pd.NA  # 값이 없으면 결측으로 채움

    df_out = df_out[col_order]

    # ─────────────────────────────────────────────────────
    # 6) dtype 정리
    #    - 모두 category 로 캐스팅해 메모리 절감 + 코드 일관성 유지
    # ─────────────────────────────────────────────────────
    cast_cols(
        df_out,
        cat_cols=[
            COL_VERSION,
            COL_SHIP_TO,
            COL_PG,
            COL_MASTER_CUTOFF,
            COL_MASTER_STATUS,
            COL_MASTER_INIT_TIME,
            COL_MASTER_UP_TIME,
        ],
        int_cols=[],
        bool_cols=[],
    )

    return df_out

# ======================================================
# Step 0-3) Sell_Out_Simul_Master_Delta Output 생성
#   - [2025-12-18 Spec 변경]
#     Delta 입력과 같은 키를 사용하되,
#     Measure 값은 모두 "Null(pd.NA)" 로 초기화
#   - 컬럼 이름은 다음과 같이 사용:
#       S/Out Master Cutoff
#       S/Out Master Status
#       S/Out Master Delta Init Time
#       S/Out Master Delta Up Time
#
#   결과 예시:
#     Version.[Version Name],Sales Domain.[Ship To],Item.[Product Group],
#     S/Out Master Cutoff,S/Out Master Status,
#     S/Out Master Delta Init Time,S/Out Master Delta Up Time
#     CWV_DP,CH018113,REF,,,,
#
#   CSV 에서는 ,, (빈문자열) 로 떨어지게 하고 싶으므로,
#   - 값: pd.NA
#   - dtype: category
#   로 세팅하고, cast_cols 에 Delta 컬럼을 넣지 않는다.
# ======================================================
@_decoration_
def fn_step00_03_build_output_simul_master_delta(
    df_delta: pd.DataFrame,
    out_version: str,
    **kwargs
) -> pd.DataFrame:
    """
    Output_Sell_Out_Simul_Master_Delta 생성 함수.

    [역할]
      - 기준키(Version, ShipTo, Product Group)는 Delta 테이블에서 가져온다.
      - Measure 컬럼은 "정식 Master 컬럼명"을 사용하되,
        값은 모두 pd.NA 로 초기화한다.
      - Delta Output 은 "아직 반영되지 않은 변경 후보"를 표현하는 용도이며,
        CSV 상에서는 빈 칸(,,)으로 보여야 한다.
    """

    # ─────────────────────────────────────────────────────
    # 0) 입력이 비어 있는 경우: 스키마만 맞춰서 빈 DF 반환
    # ─────────────────────────────────────────────────────
    if df_delta is None or df_delta.empty:
        return pd.DataFrame(
            columns=[
                COL_VERSION,
                COL_SHIP_TO,
                COL_PG,
                COL_MASTER_CUTOFF,
                COL_MASTER_STATUS,
                COL_MASTER_INIT_TIME_DELTA,
                COL_MASTER_UP_TIME_DELTA,
            ]
        )

    # ─────────────────────────────────────────────────────
    # 1) 사용할 키 컬럼만 체크 (Version, ShipTo, Product Group)
    # ─────────────────────────────────────────────────────
    use_cols: list[str] = []
    for c in [COL_VERSION, COL_SHIP_TO, COL_PG]:
        if c in df_delta.columns:
            use_cols.append(c)

    mandatory = [COL_VERSION, COL_SHIP_TO, COL_PG]
    if any(c not in use_cols for c in mandatory):
        # 키가 제대로 없으면 스키마만 맞춰서 빈 DF 반환
        return pd.DataFrame(
            columns=[
                COL_VERSION,
                COL_SHIP_TO,
                COL_PG,
                COL_MASTER_CUTOFF,
                COL_MASTER_STATUS,
                COL_MASTER_INIT_TIME_DELTA,
                COL_MASTER_UP_TIME_DELTA,
            ]
        )

    # ─────────────────────────────────────────────────────
    # 2) 기준키만 복사 (view 이슈 방지를 위해 copy)
    # ─────────────────────────────────────────────────────
    df_out = df_delta[use_cols].copy()

    # ─────────────────────────────────────────────────────
    # 3) Version 값 통일 (out_version 으로 강제 세팅)
    # ─────────────────────────────────────────────────────
    df_out[COL_VERSION] = out_version

    # ─────────────────────────────────────────────────────
    # 4) Measure 컬럼을 "정식 Master 컬럼명"으로 생성하고,
    #    값은 모두 pd.NA + dtype='category' 로 초기화
    #
    #    이렇게 해야:
    #      - 로그 PrintDF 에서는 'nan' 이 보이더라도,
    #      - to_csv(na_rep='') 기본값 기준으로 CSV 에서는 빈 칸으로 출력됨.
    # ─────────────────────────────────────────────────────
    idx = df_out.index

    # Cutoff (Delta 값은 버리고, 전부 결측으로)
    df_out[COL_MASTER_CUTOFF] = pd.Series(
        data=pd.NA,
        index=idx,
        dtype="category",
    )

    # Status (마찬가지로 전부 결측)
    df_out[COL_MASTER_STATUS] = pd.Series(
        data=pd.NA,
        index=idx,
        dtype="category",
    )

    # [2025-12-18 Spec 변경] Delta Init/Up Time 컬럼 추가
    df_out[COL_MASTER_INIT_TIME_DELTA] = pd.Series(
        data=pd.NA,
        index=idx,
        dtype="category",
    )
    df_out[COL_MASTER_UP_TIME_DELTA] = pd.Series(
        data=pd.NA,
        index=idx,
        dtype="category",
    )

    # ─────────────────────────────────────────────────────
    # 5) 컬럼 순서 정렬 + 누락 컬럼 보완
    # ─────────────────────────────────────────────────────
    col_order = [
        COL_VERSION,
        COL_SHIP_TO,
        COL_PG,
        COL_MASTER_CUTOFF,
        COL_MASTER_STATUS,
        COL_MASTER_INIT_TIME_DELTA,
        COL_MASTER_UP_TIME_DELTA,
    ]

    for c in col_order:
        if c not in df_out.columns:
            df_out[c] = pd.NA

    df_out = df_out[col_order]

    # ─────────────────────────────────────────────────────
    # 6) 공통 cast_cols 적용
    #    - Delta 측 Measure 컬럼은 cast_cols(cat_cols=...) 에 넣지 않는다!
    #      (넣으면 astype(str) 때문에 <NA> → 'nan' 문자열로 바뀌어 버림)
    # ─────────────────────────────────────────────────────
    cast_cols(
        df_out,
        cat_cols=[COL_VERSION, COL_SHIP_TO, COL_PG],
        int_cols=[],
        bool_cols=[],
    )

    return df_out

# ======================================================
# Step 0-4) Sell_Out_Simul_Master_Delta 의 Product Group 목록 추출
#   반환: np.ndarray or None
# ======================================================
@_decoration_
def fn_step00_04_extract_product_group_list(
    df_delta: pd.DataFrame,
    **kwargs
) -> Optional[np.ndarray]:
    """
    Delta 테이블에서 Product Group 목록을 추출
      - 이후 Step-02 의 Sales Product ASN 필터에 사용
    """
    if df_delta is None or df_delta.empty or (COL_PG not in df_delta.columns):
        return None

    arr_pg = pd.unique(df_delta[COL_PG].astype(str))
    return arr_pg


########################################################################################################################
# STEP 1 – Ship-To Dimension LUT (CreateSellInAndSellOutAssortment 의 step01_load_shipto_dimension 재사용)
########################################################################################################################
@_decoration_
def step01_load_shipto_dimension(
    df_dim: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Ship-To ⇢ LV_CODE & Std1~6 LUT 생성
    """
    USE_COLS = [
        COL_SHIP_TO,
        COL_STD1, COL_STD2, COL_STD3,
        COL_STD4, COL_STD5, COL_STD6
    ]
    dim = (
        df_dim[USE_COLS]
        .copy(deep=False)
        .astype({c: 'category' for c in USE_COLS})
    )

    lv_code = np.full(len(dim), 2, dtype='int8')
    ship_np = dim[COL_SHIP_TO].to_numpy()

    for col, lv in [
        (COL_STD6, 7),
        (COL_STD5, 6),
        (COL_STD4, 5),
        (COL_STD3, 4),
        (COL_STD2, 3),
        (COL_STD1, 2),
    ]:
        match = dim[col].to_numpy() == ship_np
        lv_code[match] = lv

    dim[COL_LV_CODE] = lv_code

    del ship_np, lv_code, df_dim
    gc.collect()

    return dim


########################################################################################################################
# STEP 2 – Sales Product ASN 전처리 (+ Product Group 필터)
########################################################################################################################
@_decoration_
def step02_preprocess_asn(
    df_asn     : pd.DataFrame,   # df_in_Sales_Product_ASN
    df_item    : pd.DataFrame,   # df_in_Item_Master
    df_master  : pd.DataFrame,   # Step-0-1 결과 Sell_Out_Simul_Master
    df_est     : pd.DataFrame,   # df_in_Sales_Domain_Estore
    df_dim_lut : pd.DataFrame,   # Step-01 Ship-To LUT
    arr_pg     : Optional[np.ndarray] = None,
    **kwargs
) -> pd.DataFrame:
    """
    ▶ 출력 컬럼
       ─ COL_SHIP_TO
       ─ COL_ITEM
       ─ COL_LOC
       ─ COL_PG
       ─ COL_ASN_FLAG
       ─ COL_SOUT_ASS_FLAG
      + Product Group 필터 (arr_pg 가 주어진 경우 그 PG만 남김)
    """
    use_cols = [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_ASN_FLAG]
    asn = (
        df_asn[use_cols]
        .copy(deep=False)
        .astype({
            COL_SHIP_TO  : 'category',
            COL_ITEM     : 'category',
            COL_LOC      : 'category',
            COL_ASN_FLAG : 'category'
        })
    )

    # 기본 SOUT_ASS_FLAG = True
    asn[COL_SOUT_ASS_FLAG] = True

    # Item Master → Product Group 주입
    item_small = (
        df_item[[COL_ITEM, COL_PG]]
        .copy(deep=False)
        .astype({COL_PG: 'category'})
    )
    asn = asn.merge(item_small, on=COL_ITEM, how='left')

    # Ship-To → Std5 (LV-6) 매핑
    std5_map = df_dim_lut.set_index(COL_SHIP_TO)[COL_STD5].to_dict()
    asn['_STD5'] = asn[COL_SHIP_TO].map(std5_map).astype('category')

    # Simul Master (CON) 조건
    df_master_con = df_master.loc[df_master[COL_MASTER_STATUS] == 'CON']
    master_set = set(
        zip(
            df_master_con[COL_SHIP_TO].astype(str),
            df_master_con[COL_PG].astype(str)
        )
    )

    key_pairs = np.core.defchararray.add(
        asn['_STD5'].astype(str).to_numpy(dtype='U'),
        asn[COL_PG].astype(str).to_numpy(dtype='U')
    )
    mask_in_master = np.in1d(
        key_pairs,
        np.fromiter((a + b for a, b in master_set), dtype='<U32')
    )
    asn.loc[~mask_in_master, COL_SOUT_ASS_FLAG] = False

    # E-Store Ship-To 제외
    est_set = set(df_est[COL_SHIP_TO].astype(str))
    mask_estore = asn[COL_SHIP_TO].astype(str).isin(est_set)
    asn.loc[mask_estore, COL_SOUT_ASS_FLAG] = False

    # 임시 컬럼 정리
    asn.drop(columns=['_STD5'], inplace=True)

    # ▶ Product Group 필터 (Delta 의 PG 만 남김)
    if arr_pg is not None and len(arr_pg) > 0:
        pg_filter = set(str(x) for x in arr_pg)
        asn = asn[asn[COL_PG].astype(str).isin(pg_filter)].copy()

    return asn


########################################################################################################################
# STEP 3 – Sell-In Assortments (GC / AP2 / AP1 / Local-AP0)
#   - PYForecastCreateSellInAndSellOutAssortment 의 step03_create_sellin_assortments 동일 구조
########################################################################################################################
@_decoration_
def _fast_groupby_with_category(df_tag: pd.DataFrame, qty_col: str, **kwargs) -> pd.DataFrame:
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOC):
        if df_tag[c].dtype.name != "category":
            df_tag[c] = df_tag[c].astype("category")

    out = (
        df_tag
        .groupby([COL_SHIP_TO, COL_ITEM, COL_LOC], as_index=False, sort=False, observed=True)
        [[COL_ASN_FLAG, COL_SOUT_ASS_FLAG]]
        .max()
    )
    out[qty_col] = np.int8(1)
    return out[[COL_SHIP_TO, COL_ITEM, COL_LOC, qty_col, COL_ASN_FLAG, COL_SOUT_ASS_FLAG]]


@_decoration_
def _ultra_fast_groupby_numpy(df_tag: pd.DataFrame, qty_col: str, **kwargs) -> pd.DataFrame:
    def _as_cat_codes(s: pd.Series):
        if s.dtype.name != "category":
            s = s.astype("category")
        return s.cat.codes.to_numpy(), s.cat.categories

    s_code, s_cats = _as_cat_codes(df_tag[COL_SHIP_TO])
    i_code, i_cats = _as_cat_codes(df_tag[COL_ITEM])
    l_code, l_cats = _as_cat_codes(df_tag[COL_LOC])

    S = len(s_cats) if len(s_cats) > 0 else 1
    I = len(i_cats) if len(i_cats) > 0 else 1

    gid = s_code.astype(np.int64) + S * (i_code.astype(np.int64) + I * l_code.astype(np.int64))

    asn = df_tag[COL_ASN_FLAG]
    if asn.dtype == object or str(asn.dtype).startswith("category"):
        asn_bin = (asn == 'Y').to_numpy(dtype=np.int8)
    else:
        asn_bin = (asn.astype(str) == 'Y').to_numpy(dtype=np.int8)

    sout_bin = df_tag[COL_SOUT_ASS_FLAG].to_numpy().astype(np.int8)

    order = np.argsort(gid, kind='mergesort')
    gid_sorted = gid[order]

    uniq_gid, first_idx = np.unique(gid_sorted, return_index=True)
    asn_sorted  = asn_bin[order]
    sout_sorted = sout_bin[order]

    end_idx = np.empty_like(first_idx)
    end_idx[:-1] = first_idx[1:]
    end_idx[-1]  = gid_sorted.size

    grp_asn  = np.zeros_like(first_idx, dtype=np.int8)
    grp_sout = np.zeros_like(first_idx, dtype=np.int8)
    for k in range(first_idx.size):
        grp_asn[k]  = asn_sorted [first_idx[k]:end_idx[k]].max()
        grp_sout[k] = sout_sorted[first_idx[k]:end_idx[k]].max()

    rep_rows = order[first_idx]
    ship_vals = df_tag[COL_SHIP_TO].to_numpy()[rep_rows]
    item_vals = df_tag[COL_ITEM   ].to_numpy()[rep_rows]
    loc_vals  = df_tag[COL_LOC    ].to_numpy()[rep_rows]

    out = pd.DataFrame({
        COL_SHIP_TO      : ship_vals,
        COL_ITEM         : item_vals,
        COL_LOC          : loc_vals,
        COL_ASN_FLAG     : np.where(grp_asn  == 1, 'Y', 'N'),
        COL_SOUT_ASS_FLAG: grp_sout.astype(bool),
    })
    out[qty_col] = np.int8(1)
    out[[COL_SHIP_TO, COL_ITEM, COL_LOC]] = out[[COL_SHIP_TO, COL_ITEM, COL_LOC]].astype('category')

    return out[[COL_SHIP_TO, COL_ITEM, COL_LOC, qty_col, COL_ASN_FLAG, COL_SOUT_ASS_FLAG]]

# ──────────────────────────────────────────────────────────────────────────────
# STEP-03 ▸ Sell-In Assortments (GC / AP2 / AP1 / Local-AP0)
#   - 입력 : df_fn_sales_product_asn  (Step-02 결과; S/Out Assortment Flag 포함)
#           df_in_Forecast_Rule      (GC / AP2 / AP1 / AP0 레벨 정보)
#           df_fn_shipto_dim         (Step-01 Ship-To LUT : Std1~6 + LV_CODE)
#   - 출력 : dict[str, pd.DataFrame]
#           {
#             STR_DF_OUT_SIN_GC   : [ShipTo, Item, Loc, S/In Assort_GC,   ASN, Flag],
#             STR_DF_OUT_SIN_AP2  : [ShipTo, Item, Loc, S/In Assort_AP2,  ASN, Flag],
#             STR_DF_OUT_SIN_AP1  : [ShipTo, Item, Loc, S/In Assort_AP1,  ASN, Flag],
#             STR_DF_OUT_SIN_LOCAL: [ShipTo, Item, Loc, S/In Assort_Local,ASN, Flag],
#           }
#
#   ⚠ 중요 :
#     - 여기서는 **S/Out Assortment Flag = False 도 절대 제거하지 않음**.
#       (Flag는 그대로 유지해서 Step04에서 조건에 따라 사용 가능하도록 함)
#     - groupby / 집계는 모두 ultra_fast_groupby_numpy_general 로 처리.
#       (기존 _ultra_fast_groupby_numpy 대신 사용)
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step03_00_create_sellin_assortments(
    df_fn_sales_product_asn: pd.DataFrame,   # Step-02 결과
    df_in_Forecast_Rule:     pd.DataFrame,   # Forecast Rule (GC/AP2/AP1/AP0)
    df_fn_shipto_dim:        pd.DataFrame,   # Ship-To LUT (Std1~6 + LV_CODE)
    **kwargs
) -> dict[str, pd.DataFrame]:
    fn_name = 'fn_step03_00_create_sellin_assortments'
    # ── 0) 기본 가드 : 하나라도 비어 있으면 4개 빈 DF 반환 ─────────────────────
    if (
        df_fn_sales_product_asn is None or df_fn_sales_product_asn.empty or
        df_in_Forecast_Rule     is None or df_in_Forecast_Rule.empty     or
        df_fn_shipto_dim        is None or df_fn_shipto_dim.empty
    ):
        def _empty(q_col: str) -> pd.DataFrame:
            return pd.DataFrame(
                columns=[COL_SHIP_TO, COL_ITEM, COL_LOC, q_col, COL_ASN_FLAG, COL_SOUT_ASS_FLAG]
            )

        return {
            STR_DF_OUT_SIN_GC   : _empty(COL_SIN_ASS_GC),
            STR_DF_OUT_SIN_AP2  : _empty(COL_SIN_ASS_AP2),
            STR_DF_OUT_SIN_AP1  : _empty(COL_SIN_ASS_AP1),
            STR_DF_OUT_SIN_LOCAL: _empty(COL_SIN_ASS_LOCAL),
        }

    # ── 1) Ship-To 계층 정보 준비 (Std1~6, LV_CODE LUT) ──────────────────────
    STD_COLS = [COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6]

    # Ship-To 를 index 로 갖는 LUT
    dim_idx = df_fn_shipto_dim.set_index(COL_SHIP_TO)

    # LV_MAP : lv(1~6) → {ShipTo: Std_lv}
    #   - lv=1 → Std1, lv=2 → Std2 … lv=6 → Std6
    LV_MAP: dict[int, dict] = {
        lv: dim_idx[STD_COLS[lv - 1]].to_dict()
        for lv in range(1, 7)
    }

    # SHIP_LV : ShipTo → LV_CODE (현재 Ship-To 레벨)
    SHIP_LV: dict = dim_idx[COL_LV_CODE].to_dict()
    if is_local:
        logger.debug(f'{fn_name}: SHIP_LV')
        logger.debug(SHIP_LV)

    # ── 2) Forecast Rule 인덱스 준비 : (Product Group, ShipTo) → (GC/AP2/AP1/AP0) ─
    df_rule = df_in_Forecast_Rule.copy(deep=False)

    # 2-1) 룰 레벨 컬럼 NA → 0, dtype=int8 로 정리
    for c in (COL_FR_GC, COL_FR_AP2, COL_FR_AP1, COL_FR_AP0):
        if c in df_rule.columns:
            df_rule[c] = df_rule[c].fillna(0).astype('int8')
        else:
            # 컬럼이 없다면 0으로 생성
            df_rule[c] = np.int8(0)

    # 2-2) (PG, ShipTo) 인덱스로 레벨 벡터 조회용 MultiIndex DataFrame 생성
    rule_idx = (
        df_rule
        .assign(
            **{
                COL_PG     : df_rule[COL_PG].astype(str),
                COL_SHIP_TO: df_rule[COL_SHIP_TO].astype(str),
            }
        )
        .set_index([COL_PG, COL_SHIP_TO])
        [[COL_FR_GC, COL_FR_AP2, COL_FR_AP1, COL_FR_AP0]]
        .astype('int8')
    )
    fn_log_dataframe(rule_idx,f'{fn_name}:rule_idx')

    # ────────────────────────────────────────────────────────────────────────
    # 3) Core Builder
    #    - df_fn_sales_product_asn 전체를 한 번에 처리
    #    - Ship-To / Product-Group / Rule / LV_MAP 를 모두 벡터화해서 처리
    # ────────────────────────────────────────────────────────────────────────
    def _build_level(df_src: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        주어진 ASN/Flag DF(df_src)에 대해 GC/AP2/AP1/Local 각각의
        "Sell-In Assortment" DataFrame 을 생성.
        """

        # ── 3-1) 카테고리 코드 & 고유 Ship/PG 목록 추출 ────────────────────
        ship_cat = df_src[COL_SHIP_TO]
        if ship_cat.dtype.name != 'category':
            ship_cat = ship_cat.astype('category')
        ship_codes   = ship_cat.cat.codes.to_numpy()           # (N,)
        ships_unique = ship_cat.cat.categories.to_numpy()      # (U,)

        pg_cat = df_src[COL_PG]
        if pg_cat.dtype.name != 'category':
            pg_cat = pg_cat.astype('category')
        pg_codes   = pg_cat.cat.codes.to_numpy()               # (N,)
        pg_unique  = pg_cat.cat.categories.to_numpy()          # (PgU,)

        # 실제 값 슬라이스 (Item, Loc, ASN, S/Out Flag)
        item_vals = df_src[COL_ITEM].to_numpy()
        loc_vals  = df_src[COL_LOC ].to_numpy()
        asn_vals  = df_src[COL_ASN_FLAG].to_numpy()
        sout_vals = df_src[COL_SOUT_ASS_FLAG].to_numpy()

        N = ship_codes.size
        U = ships_unique.size
        if is_local:
            logger.debug(f'{fn_name}. unique of shipto')
            logger.debug(ships_unique)

        # ── 3-2) anc_table : [레벨(0~7), 고유ShipIndex] → 조상 Ship ──────────
        #   - index 0,1 은 사용하지 않고 2~7 만 의미 있음
        anc_table = np.empty((8, U), dtype=object)
        anc_table[0] = None
        anc_table[1] = None
        if is_local:
            logger.debug(f'{fn_name}. anc_table Start:')
            logger.debug(anc_table)

        for lv in range(2, 8):
            mapper = LV_MAP[lv - 1]  # lv=2 → Std1, lv=3 → Std2, ...
            anc_table[lv] = np.fromiter(
                (mapper.get(s) for s in ships_unique),
                dtype=object,
                count=U
            )
            if is_local:
                logger.debug(f'{fn_name}. anc_table level {lv}:')
                logger.debug(anc_table)

        # ── 3-3) 본 Ship-To 의 LV_CODE 벡터 ────────────────────────────────
        ship_lv_unique = np.fromiter(
            (SHIP_LV.get(s, 99) for s in ships_unique),
            dtype=np.int8,
            count=U
        )
        ship_lv_arr = ship_lv_unique[ship_codes]  # (N,)
        if is_local:
            logger.debug(f'{fn_name}. 원래소스 asn 의 shipto 레벨. ship_lv_unique')
            logger.debug(ship_lv_unique)
        
            logger.debug(f'{fn_name}. 원래소스 asn 의 shipto 레벨. ship_lv_arr')
            logger.debug(ship_lv_arr)


        # ── 3-4) 태그별 레벨 저장용 배열 초기화 ─────────────────────────────
        TAGS = ('GC', 'AP2', 'AP1', 'Local')
        tag_lvmat: dict[str, np.ndarray] = {
            t: np.zeros(N, dtype=np.int8) for t in TAGS
        }
        if is_local:
            logger.debug(f'{fn_name}. tag_lvmat 초기화')
            logger.debug(tag_lvmat)


        # ── 3-5) 6개 레벨(Std1~Std6)에 대해 Rule 적용 (lv 루프) ─────────────
        #   - lv = 1~6
        #   - anc_table[lv] : (U,) → ship_codes 로 indexing 해서 parent Ship 벡터 생성
        for lv in range(1, 7):
            # parent : 현재 row 의 Ship-To 가 lv 레벨에서 가리키는 조상 Ship 코드
            parent = anc_table[lv][ship_codes]       # (N,), object (ShipTo string or None)
            mask   = parent != None
            if not mask.any():
                continue

            idx = np.flatnonzero(mask)               # 유효 row index (M,)

            parent_sub  = parent[mask]               # (M,)
            pg_code_sub = pg_codes[mask]             # (M,)

            # ── parent_sub factorize → [0..P-1] + uniques ───────────────
            parent_codes_sub, parent_uniques = pd.factorize(
                parent_sub,
                sort=False
            )
            P = int(parent_uniques.size)
            if is_local:
                logger.debug(f'{fn_name}. parent_codes_sub lv:{lv}')
                logger.debug(parent_codes_sub)
                logger.debug(f'{fn_name}. parent_uniques lv:{lv}')
                logger.debug(parent_uniques)

            # 단일 정수 key = pg_code * P + parent_code
            pair_key = (
                pg_code_sub.astype(np.int64) * P +
                parent_codes_sub.astype(np.int64)
            )
            uniq_key, inv = np.unique(pair_key, return_inverse=True)
            uniq_pg_codes   = (uniq_key // P).astype(np.int64)
            uniq_parent_idx = (uniq_key %  P).astype(np.int64)

            # 문자열로 복원 (K,)
            uniq_pg_str   = pg_unique[uniq_pg_codes]
            uniq_parent_s = parent_uniques[uniq_parent_idx]

            # (PG, ShipTo) MultiIndex 로 RULE 조회
            mi = pd.MultiIndex.from_arrays([uniq_pg_str, uniq_parent_s])
            if is_local:
                logger.debug(f'{fn_name}. index(PG, ShipTo)  lv:{lv}')
                logger.debug(mi)

            # reindex 후 NA → 0, int8 로 변환
            rule_mat = (
                rule_idx
                .reindex(mi)
                .fillna(0)
                .astype('int8')
                .to_numpy(copy=False)         # (K, 4) : [GC, AP2, AP1, AP0]
            )
            if is_local:
                logger.debug(f'{fn_name}. rule_mat lv:{lv}')
                logger.debug(rule_mat)
            

            # inv 를 통해 다시 (M,4)로 펼치기
            rv = rule_mat[inv]
            gc_lv, ap2_lv, ap1_lv, ap0_lv = rv.T
            if is_local:
                logger.debug(f'{fn_name}. gc_lv lv:{lv}')
                logger.debug(gc_lv)
                logger.debug(f'{fn_name}. ap2_lv lv:{lv}')
                logger.debug(ap2_lv)
                logger.debug(f'{fn_name}. ap1_lv lv:{lv}')
                logger.debug(ap1_lv)
                logger.debug(f'{fn_name}. ap0_lv lv:{lv}')
                logger.debug(ap0_lv)


            # 하위 레벨 매칭이 상위 레벨 매칭을 덮어쓰도록, 0이 아닌 값만 업데이트
            tag_lvmat['GC']   [idx] = np.where(gc_lv  != 0, gc_lv,  tag_lvmat['GC']   [idx])
            tag_lvmat['AP2']  [idx] = np.where(ap2_lv != 0, ap2_lv, tag_lvmat['AP2']  [idx])
            tag_lvmat['AP1']  [idx] = np.where(ap1_lv != 0, ap1_lv, tag_lvmat['AP1']  [idx])
            tag_lvmat['Local'][idx] = np.where(ap0_lv != 0, ap0_lv, tag_lvmat['Local'][idx])

        # ── 3-6) 태그별 최종 Sell-In Assortment DF 생성 ─────────────────────
        qty_name = {
            'GC'   : COL_SIN_ASS_GC,
            'AP2'  : COL_SIN_ASS_AP2,
            'AP1'  : COL_SIN_ASS_AP1,
            'Local': COL_SIN_ASS_LOCAL,
        }

        frames: dict[str, pd.DataFrame] = {}

        for tag in TAGS:
            lv_arr = tag_lvmat[tag]

            # 유효 조건 :
            #   - 요청 레벨이 2~7 사이
            #   - 요청 레벨 ≤ Ship-To 실제 LV_CODE
            valid = (
                (lv_arr >= 2) &
                (lv_arr <= 7) &
                (lv_arr <= ship_lv_arr)
            )

            if not valid.any():
                # 해당 태그에 대한 결과가 없을 때도 동일 스키마 빈 DF 유지
                frames[tag] = pd.DataFrame(
                    columns=[
                        COL_SHIP_TO, COL_ITEM, COL_LOC,
                        qty_name[tag], COL_ASN_FLAG, COL_SOUT_ASS_FLAG
                    ]
                )
                continue

            # 안전하게 : 유효하지 않은 row 는 lv=0 으로 설정하여 anc_table[0] → None
            safe_lv = lv_arr.copy()
            safe_lv[~valid] = 0

            # anc_table[lv, ship_code] 로 조상 Ship-To 벡터 생성 후 valid 로 필터
            tgt_ship = anc_table[safe_lv, ship_codes]   # (N,)
            tgt_ship = tgt_ship[valid]                  # (n_valid,)

            # 태그별 임시 DF (groupby 전 버퍼)
            df_tag = pd.DataFrame({
                COL_SHIP_TO      : tgt_ship,
                COL_ITEM         : item_vals[valid],
                COL_LOC          : loc_vals [valid],
                COL_ASN_FLAG     : asn_vals [valid],
                COL_SOUT_ASS_FLAG: sout_vals[valid],
            })

            q_col = qty_name[tag]

            # ────────────────────────────────────────────────────────────
            # ultra_fast_groupby_numpy_general 로
            # (ShipTo, Item, Loc) 단위로 MAX 집계
            #   - Sales Product ASN     : 'max' → 'Y'/'N' 플래그 유지
            #   - S/Out Assortment Flag : 'max' → True/False 유지
            #   - qty 컬럼은 집계 후 1로 세팅 (int8)
            # ────────────────────────────────────────────────────────────
            df_tag = ultra_fast_groupby_numpy_general(
                df=df_tag,
                key_cols=[COL_SHIP_TO, COL_ITEM, COL_LOC],
                aggs={
                    COL_ASN_FLAG     : 'max',
                    COL_SOUT_ASS_FLAG: 'max',
                }
            )

            # qty = 1 (int8) 고정. Step4에서 조정한다.
            df_tag[q_col] = np.int8(1)

            # category 
            df_tag[COL_ASN_FLAG] = df_tag[COL_ASN_FLAG].astype(str).astype('category')

            # 최종 컬럼 순서 정리
            df_tag = df_tag[
                [COL_SHIP_TO, COL_ITEM, COL_LOC, q_col, COL_ASN_FLAG, COL_SOUT_ASS_FLAG]
            ]

            frames[tag] = df_tag

        return frames  # {'GC': df, 'AP2': df, 'AP1': df, 'Local': df}

    # ── 4) Core Builder 실행 & 결과 dict 재구성 ─────────────────────────────
    frames = _build_level(df_fn_sales_product_asn)

    df_out: dict[str, pd.DataFrame] = {
        STR_DF_OUT_SIN_GC   : frames['GC'],
        STR_DF_OUT_SIN_AP2  : frames['AP2'],
        STR_DF_OUT_SIN_AP1  : frames['AP1'],
        STR_DF_OUT_SIN_LOCAL: frames['Local'],
    }

    # 메모리 정리 (선택 사항)
    del df_fn_sales_product_asn, df_in_Forecast_Rule, df_fn_shipto_dim, frames
    gc.collect()

    return df_out

# ─────────────────────────────────────────────────────────────────────────────
# STEP-04 ▸ Sell-Out Assortments (GC / AP2 / AP1 / Local-AP0)
# -----------------------------------------------------------------------------
#  • 입력  : Step-03 결과 dict
#       {
#         STR_DF_OUT_SIN_GC   : [ShipTo, Item, Loc, S/In Assort_GC,   ASN, Flag(int8)],
#         STR_DF_OUT_SIN_AP2  : [ShipTo, Item, Loc, S/In Assort_AP2,  ASN, Flag(int8)],
#         STR_DF_OUT_SIN_AP1  : [ShipTo, Item, Loc, S/In Assort_AP1,  ASN, Flag(int8)],
#         STR_DF_OUT_SIN_LOCAL: [ShipTo, Item, Loc, S/In Assort_Local,ASN, Flag(int8)],
#       }
#
#  • 출력  : Sell-Out Assortment dict
#       {
#         STR_DF_OUT_SOUT_GC    : [ShipTo, Item, Loc='-', ASN, Flag(int8), S/Out Assort_GC(float32)],
#         STR_DF_OUT_SOUT_AP2   : [ShipTo, Item, Loc='-', ASN, Flag(int8), S/Out Assort_AP2(float32)],
#         STR_DF_OUT_SOUT_AP1   : [ShipTo, Item, Loc='-', ASN, Flag(int8), S/Out Assort_AP1(float32)],
#         STR_DF_OUT_SOUT_LOCAL : [ShipTo, Item, Loc='-', ASN, Flag(int8), S/Out Assort_Local(float32)],
#       }
#
#  • 주요 설계 포인트
#     1) **S/Out Assortment Flag == 0 (False) 인 행도 절대 삭제하지 않음**
#        - 더 이상 FLAG=True 로 필터링하지 않음
#        - ShipTo+Item 단위 집계 후에도, Flag 그룹결과가 0이면 그대로 0 유지
#
#     2) Sell-Out Assortment 수량 컬럼(S/Out FCST Assortment_*) 타입
#        - dtype: float32
#        - 값   : Flag == 1 → 1.0
#                 Flag == 0 → NaN  (Null 허용)
#
#     3) groupby 는 모두 ultra_fast_groupby_numpy_general 로 처리
#        - key_cols = [COL_SHIP_TO, COL_ITEM]
#        - aggs = { qty_in_col: 'sum', COL_ASN_FLAG: 'max', COL_SOUT_ASS_FLAG: 'max' }
#
#     4) Location 은 Sell-Out 레벨에서 모두 '-' 로 고정
#        - ShipTo + Item 레벨까지 Location 축을 제거하고, 최종에 '-' 세팅
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step04_00_create_sellout_assortments(
    df_sin_dict: dict[str, pd.DataFrame],
    **kwargs
) -> dict[str, pd.DataFrame]:

    # ─────────────────────────────────────────────────────────────────────
    # 0) 공통 Empty DataFrame 생성 헬퍼
    #    - Downstream에서 스키마 의존하므로, 항상 동일한 헤더 유지
    # ─────────────────────────────────────────────────────────────────────
    def _empty_sout(q_col: str) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                COL_SHIP_TO,          # Sales Domain.[Ship To]
                COL_ITEM,             # Item.[Item]
                COL_LOC,              # Location.[Location]  ('-' 고정)
                COL_ASN_FLAG,         # Sales Product ASN
                COL_SOUT_ASS_FLAG,    # S/Out Assortment Flag (int8)
                q_col                 # S/Out FCST Assortment_* (float32, nullable)
            ]
        )

    # ─────────────────────────────────────────────────────────────────────
    # 1) 개별 태그(GC/AP2/AP1/Local)를 Sell-Out 으로 변환하는 헬퍼
    #    - 입력 : Step03 의 한 태그용 DF
    #    - 출력 : Sell-Out용 DF (ShipTo+Item 레벨, Loc='-')
    # ─────────────────────────────────────────────────────────────────────
    def _build_sout_one(
        df_in: pd.DataFrame,
        qty_in_col: str,    # Step-03 의 S/In Assortment 컬럼명 (int8)
        qty_out_col: str    # Step-04 의 S/Out Assortment 컬럼명 (float32 로 생성)
    ) -> pd.DataFrame:

        # 1-1) 입력이 비어 있으면 스키마만 있는 빈 DF 반환
        if df_in is None or df_in.empty:
            return _empty_sout(qty_out_col)

        # 1-2) 필요한 컬럼만 복사 (불필요 컬럼 제거로 메모리 절감)
        #      - Location 은 groupby 키에서 제외 (나중에 '-' 로 세팅)
        use_cols = [
            COL_SHIP_TO,
            COL_ITEM,
            qty_in_col,
            COL_ASN_FLAG,
            COL_SOUT_ASS_FLAG
        ]
        work = df_in[use_cols].copy(deep=False)

        # 1-3) 그룹 키 / 집계 정의
        #      - key_cols : ShipTo + Item
        #      - qty_in_col        : 'sum'  → 중복 제거 목적 (나중에 실제 값은 사용하지 않음)
        #      - Sales Product ASN : 'max'  → Y/N 중 하나라도 Y면 Y
        #      - S/Out Flag(int8)  : 'max'  → 하나라도 1이면 1, 전부 0이면 0
        g_cols = [COL_SHIP_TO, COL_ITEM]
        agg_dict = {
            qty_in_col        : 'sum',
            COL_ASN_FLAG      : 'max',
            COL_SOUT_ASS_FLAG : 'max'
        }

        # 1-4) ultra_fast_groupby_numpy_general 로 ShipTo+Item 단위 집계
        #      - 이 시점에서 S/Out Assortment Flag == 0 인 ShipTo+Item 도 절대 사라지지 않음.
        agg_df = ultra_fast_groupby_numpy_general(
            df=work,
            key_cols=g_cols,
            aggs=agg_dict
        )
        # agg_df 컬럼 예시:
        #   Sales Domain.[Ship To], Item.[Item], qty_in_col(sum), Sales Product ASN(max), S/Out Assortment Flag(max)
        
        # category 
        agg_df[COL_ASN_FLAG] = agg_df[COL_ASN_FLAG].astype(str).astype('category')
        
        # 1-5) Location 은 Sell-Out 레벨에서 '-' 로 고정
        agg_df[COL_LOC] = '-'

        # 1-6) S/Out Assortment Flag 를 다시 int8 로 정리
        #      (ultra_fast_groupby_numpy_general 결과가 int64 등으로 업캐스트 될 수 있으므로)
        if COL_SOUT_ASS_FLAG in agg_df.columns:
            agg_df[COL_SOUT_ASS_FLAG] = (
                pd.to_numeric(agg_df[COL_SOUT_ASS_FLAG], errors='coerce')
                  .fillna(0)
                  .astype('int8')
            )
        else:
            # 안전장치 : 혹시라도 컬럼이 누락되면 0으로 생성
            agg_df[COL_SOUT_ASS_FLAG] = np.int8(0)

        # 1-7) Sell-Out Assortment 수량 컬럼 생성 (float32 + Null 허용)
        #      - Flag == 1 → 1.0
        #      - Flag == 0 → NaN
        flag_arr = agg_df[COL_SOUT_ASS_FLAG].to_numpy()
        vals = np.where(flag_arr == 1, 1.0, np.nan)
        agg_df[qty_out_col] = vals.astype('float32')   # dtype=float32, NaN 허용

        # 1-8) 더 이상 필요 없는 qty_in_col 은 제거
        if qty_in_col in agg_df.columns:
            agg_df.drop(columns=[qty_in_col], inplace=True)

        # 1-9) 컬럼 순서 정리
        agg_df = agg_df[
            [
                COL_SHIP_TO,
                COL_ITEM,
                COL_LOC,
                COL_ASN_FLAG,
                COL_SOUT_ASS_FLAG,
                qty_out_col
            ]
        ]

        # 1-10) 차원 컬럼은 category 로 캐스팅 (메모리 절감)
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOC):
            if c in agg_df.columns:
                agg_df[c] = agg_df[c].astype('category')

        return agg_df

    # ─────────────────────────────────────────────────────────────────────
    # 2) 태그별(GC/AP2/AP1/Local)로 Sell-Out DF 생성
    # ─────────────────────────────────────────────────────────────────────
    df_sin_gc    = df_sin_dict.get(STR_DF_OUT_SIN_GC)
    df_sin_ap2   = df_sin_dict.get(STR_DF_OUT_SIN_AP2)
    df_sin_ap1   = df_sin_dict.get(STR_DF_OUT_SIN_AP1)
    df_sin_local = df_sin_dict.get(STR_DF_OUT_SIN_LOCAL)

    df_sout_gc = _build_sout_one(
        df_in=df_sin_gc,
        qty_in_col=COL_SIN_ASS_GC,
        qty_out_col=COL_SOUT_ASS_GC
    )
    df_sout_ap2 = _build_sout_one(
        df_in=df_sin_ap2,
        qty_in_col=COL_SIN_ASS_AP2,
        qty_out_col=COL_SOUT_ASS_AP2
    )
    df_sout_ap1 = _build_sout_one(
        df_in=df_sin_ap1,
        qty_in_col=COL_SIN_ASS_AP1,
        qty_out_col=COL_SOUT_ASS_AP1
    )
    df_sout_local = _build_sout_one(
        df_in=df_sin_local,
        qty_in_col=COL_SIN_ASS_LOCAL,
        qty_out_col=COL_SOUT_ASS_LOCAL
    )

    # ─────────────────────────────────────────────────────────────────────
    # 3) 결과 dict 구성 후 반환
    # ─────────────────────────────────────────────────────────────────────
    out_dict: dict[str, pd.DataFrame] = {
        STR_DF_OUT_SOUT_GC    : df_sout_gc,
        STR_DF_OUT_SOUT_AP2   : df_sout_ap2,
        STR_DF_OUT_SOUT_AP1   : df_sout_ap1,
        STR_DF_OUT_SOUT_LOCAL : df_sout_local,
    }

    return out_dict


########################################################################################################################
# STEP 5 – 최종 Output Formatter (Version / 컬럼 정렬)
#   - PYForecastMasterSelloutDelta 에서는 Sell-In Assortment(4개)를 Output 에 포함하지 않음
#   - Output 은
#       1) Sell-Out Assortment 4개
#       2) Sell-Out Simul Master 2개
#   - 추가 요구사항
#       • Version.[Version Name] 은 category 타입이어야 한다.
#       • Sell-Out Assortment 4개:
#           - Version / ShipTo / Item / Location : category
#           - S/Out FCST Assortment_*           : float32 (Null 허용)
########################################################################################################################
@_decoration_
def fn_output_formatter(
    out_version          : str,
    df_sout_dict         : Dict[str, pd.DataFrame],
    df_out_simul_master  : pd.DataFrame,
    df_out_simul_delta   : pd.DataFrame,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Sell-Out 4개 + Simul Master 2개 결과 DF 를
      - Version 컬럼 주입
      - 컬럼 순서 정렬
      - dtype 정리(category / float32)
    까지 수행하여 하나의 dict 로 반환하는 Formatter.
    """

    # ─────────────────────────────────────────────────────────────────────
    # 0) Output 테이블별 최종 컬럼 순서 정의
    #    - 이 순서대로만 Output 을 만들어 downstream 에 공급한다.
    # ─────────────────────────────────────────────────────────────────────
    SOUT_ORDER = {
        STR_DF_OUT_SOUT_GC    : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SOUT_ASS_GC],
        STR_DF_OUT_SOUT_AP2   : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SOUT_ASS_AP2],
        STR_DF_OUT_SOUT_AP1   : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SOUT_ASS_AP1],
        STR_DF_OUT_SOUT_LOCAL : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SOUT_ASS_LOCAL],
    }
    SIMUL_ORDER = {
        STR_DF_OUT_SIMUL_MASTER: [
            COL_VERSION, COL_SHIP_TO, COL_PG,
            COL_MASTER_CUTOFF, COL_MASTER_STATUS
        ],
        STR_DF_OUT_SIMUL_MASTER_DELTA: [
            COL_VERSION, COL_SHIP_TO, COL_PG,
            COL_MASTER_CUTOFF_DELTA, COL_MASTER_STATUS_DELTA
        ]
    }

    # 최종 반환 dict
    out_all: Dict[str, pd.DataFrame] = {}

    # ─────────────────────────────────────────────────────────────────────
    # 1) 공통 dtype 정리용 헬퍼 함수들
    # ─────────────────────────────────────────────────────────────────────
    def _cast_dim_to_category(df: pd.DataFrame, cols: list[str]) -> None:
        """
        dim 용 컬럼들을 category 로 캐스팅
        (빈 DF 여도 astype('category') 가능하므로, schema 통일에 유리)
        """
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype('category')
        

    def _cast_measure_to_float32(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        수량/measure 컬럼을 float32 로 캐스팅.
        - NaN 포함 가능 (nullable)
        - df 가 slice/view 여도 안전하게 동작하도록 항상 copy() 후 캐스팅
        """
        # 대상 컬럼이 없으면 그대로 반환
        if col not in df.columns:
            return df

        # 1) 항상 복사본을 만들어서 작업 (SettingWithCopyWarning 방지)
        df = df.copy()

        # 2) float32 로 캐스팅
        df[col] = df[col].astype('float32')

        return df
    
    # ─────────────────────────────────────────────────────────────────────
    # 2) Sell-Out Assortment 4개 Output 정리
    #    - 각 DF 를 가져와서:
    #        1) 비어 있으면 헤더만 가진 DF 생성
    #        2) 필요한 컬럼 없으면 NaN 으로 생성
    #        3) Version 값 주입
    #        4) 컬럼 순서 강제 정렬 (나머지 컬럼은 자동 Drop)
    #        5) dim 컬럼 category 캐스팅
    #        6) Assortment measure 컬럼 float32 캐스팅
    # ─────────────────────────────────────────────────────────────────────
    for df_name, col_order in SOUT_ORDER.items():
        # 2-1) Source DF 가져오기 (없으면 빈 DF)
        df = df_sout_dict.get(df_name, pd.DataFrame())

        # 2-2) 완전히 비어 있으면, schema 부터 맞춘 빈 DF 생성
        if df.empty:
            df = pd.DataFrame(columns=col_order)

        # 2-3) 필요한 컬럼이 없으면 신규 생성 (NaN 채움)
        #      - 이후에 Version 은 실제 값으로 덮어쓰고,
        #      - dim/measure dtype 은 별도 캐스팅으로 정리한다.
        for col in col_order:
            if col not in df.columns:
                df[col] = np.nan

        # 2-4) Version 값 주입
        df[COL_VERSION] = out_version

        # 2-5) 컬럼 순서 강제 정렬
        #      - col_order 에 없는 컬럼은 자동으로 Drop 된다.
        df = df[col_order].copy()

        # 2-6) dim 컬럼을 category 로 맞추기
        #      - 요구사항: Version.[Version Name] 도 category 이어야 함
        _cast_dim_to_category(
            df,
            cols=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC]
        )

        # # 2-7) Assortment measure 컬럼을 float32 로 맞추기
        # qty_col = col_order[-1]   # 마지막 컬럼이 항상 S/Out FCST Assortment_* 임
        # df = _cast_measure_to_float32(df, qty_col)

        # 2-8) 정리된 DF 를 최종 dict 에 저장
        out_all[df_name] = df

    # ─────────────────────────────────────────────────────────────────────
    # 3) Simul Master 2개 Output 정리
    #    - df_out_simul_master / df_out_simul_delta 를 받아 동일 패턴으로 처리
    # ─────────────────────────────────────────────────────────────────────
    df_simul = df_out_simul_master.copy() if df_out_simul_master is not None else pd.DataFrame()
    df_delta = df_out_simul_delta.copy()  if df_out_simul_delta is not None else pd.DataFrame()

    for df_name, col_order in SIMUL_ORDER.items():
        # 3-1) 어떤 DF 를 사용할지 선택
        if df_name == STR_DF_OUT_SIMUL_MASTER:
            df = df_simul
        else:
            df = df_delta

        # 3-2) 비어 있으면 헤더만 있는 DF 생성
        if df.empty:
            df = pd.DataFrame(columns=col_order)

        # 3-3) 필요한 컬럼이 없으면 NaN 으로 생성
        for col in col_order:
            if col not in df.columns:
                df[col] = np.nan

        # 3-4) Version 값 주입
        df[COL_VERSION] = out_version

        # 3-5) 컬럼 순서 강제 정렬
        df = df[col_order].copy()

        # 3-6) dim 컬럼들(category) 정리
        #      - Version, ShipTo, Product Group 은 category 로 통일
        _cast_dim_to_category(
            df,
            cols=[COL_VERSION, COL_SHIP_TO, COL_PG]
        )

        # 3-7) Status 컬럼들도 category 로 맞춰주면, Delta/기준 모두 일관적
        if COL_MASTER_STATUS in df.columns:
            df[COL_MASTER_STATUS] = df[COL_MASTER_STATUS].astype('category')
        if COL_MASTER_STATUS_DELTA in df.columns:
            df[COL_MASTER_STATUS_DELTA] = df[COL_MASTER_STATUS_DELTA].astype('category')

        # 3-8) 필요시 Cutoff 관련 컬럼은 숫자/문자 그대로 두어도 되지만,
        #      원하시면 아래처럼 object→category 또는 numeric 으로 통일 가능
        #  예)
        # if COL_MASTER_CUTOFF in df.columns:
        #     df[COL_MASTER_CUTOFF] = df[COL_MASTER_CUTOFF].astype('category')

        # 3-9) 최종 dict 에 저장
        out_all[df_name] = df

    # ─────────────────────────────────────────────────────────────────────
    # 4) 모든 Output DF 를 담은 dict 반환
    # ─────────────────────────────────────────────────────────────────────
    return out_all

################################################################################################################
# Start Main
################################################################################################################
if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    try:
        ################################################################################################################
        # 전처리 : 모듈 내에서 사용될 데이터에 대한 정합성 체크 및 데이터 선 가공
        ################################################################################################################
        
        if is_local:
            Version = 'CWV_DP'
            input_folder_name = str_instance
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
        
        
        # vdLog 초기화
        log_path = os.path.dirname(__file__) if is_local else ""
        vdCommon.gfn_pyLog_start(Version, str_instance, logger, is_local, log_path)
        # --------------------------------------------------------------------------
        # df_input 체크 시작
        # --------------------------------------------------------------------------
        logger.Note(p_note='df_input 체크 시작', p_log_level=LOG_LEVEL.debug())
        if is_local: 
            import csv
            # 로컬인 경우 Output 폴더를 정리한다.
            for file in os.scandir(str_output_dir):
                os.remove(file.path)

            # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
            file_pattern = f"{os.getcwd()}/{str_input_dir}/*.csv" 
            csv_files = glob.glob(file_pattern)

            file_to_df_mapping = {
                "df_in_Sales_Domain_Dimension.csv"          :      STR_DF_IN_SALES_DOMAIN_DIM           ,
                "df_in_Sales_Domain_Estore.csv"             :      STR_DF_IN_ESTORE                     ,  
                "df_in_Sales_Product_ASN.csv"               :      STR_DF_IN_SALES_PRODUCT_ASN          ,       
                "df_in_Forecast_Rule.csv"                   :      STR_DF_IN_FORECAST_RULE              ,      
                "df_in_Item_Master.csv"                     :      STR_DF_IN_ITEM_MASTER                ,          
                f"{STR_DF_IN_SELL_OUT_SIMUL_MASTER}.csv"    :      STR_DF_IN_SELL_OUT_SIMUL_MASTER      ,            
                f"{STR_DF_IN_SELL_OUT_SIMUL_MASTER_D}.csv"  :      STR_DF_IN_SELL_OUT_SIMUL_MASTER_D                                  
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
                    
                    if file_name == os.path.splitext(keyword)[0]:
                        if STR_DF_IN_SALES_DOMAIN_DIM == frame_name: 
                            df_in_Sales_Domain_Dimension = df
                            break
                        if STR_DF_IN_ESTORE == frame_name: 
                            df_in_Sales_Domain_Estore = df
                            break
                        if STR_DF_IN_SALES_PRODUCT_ASN == frame_name: 
                            df_in_Sales_Product_ASN = df
                            break
                        if STR_DF_IN_FORECAST_RULE == frame_name: 
                            df_in_Forecast_Rule = df
                            break
                        if STR_DF_IN_ITEM_MASTER == frame_name: 
                            df_in_Item_Master = df
                            break
                        if STR_DF_IN_SELL_OUT_SIMUL_MASTER == frame_name: 
                            df_in_Sell_Out_Simul_Master = df
                            break

                        if STR_DF_IN_SELL_OUT_SIMUL_MASTER_D == frame_name: 
                            df_in_Sell_Out_Simul_Master_Delta = df
                            break

        


        # ----------------------------------------------------------------------
        # dtype 캐스팅 (메모리 절감 + 일관성)
        # ----------------------------------------------------------------------
        cast_cols(
            df_in_Sales_Domain_Dimension,
            cat_cols=[COL_SHIP_TO, COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6],
        )
        cast_cols(
            df_in_Sales_Domain_Estore,
            cat_cols=[COL_SHIP_TO]
        )
        cast_cols(
            df_in_Item_Master,
            cat_cols=[COL_ITEM, COL_PG]
        )
        cast_cols(
            df_in_Sales_Product_ASN,
            cat_cols=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_ASN_FLAG]
        )
        cast_cols(
            df_in_Forecast_Rule,
            cat_cols=[COL_VERSION, COL_PG, COL_SHIP_TO],
            int_cols=[COL_FR_GC, COL_FR_AP2, COL_FR_AP1, COL_FR_AP0]
        )
        cast_cols(
            df_in_Sell_Out_Simul_Master,
            cat_cols=[COL_VERSION, COL_SHIP_TO, COL_PG, COL_MASTER_STATUS]
        )
        cast_cols(
            df_in_Sell_Out_Simul_Master_Delta,
            cat_cols=[COL_VERSION, COL_SHIP_TO, COL_PG, COL_MASTER_CUTOFF_DELTA, COL_MASTER_STATUS_DELTA]
        )

        # ----------------------------------------------------------------------
        # Input 체크
        # ----------------------------------------------------------------------

        fn_log_dataframe(df_in_Sales_Domain_Dimension,  STR_DF_IN_SALES_DOMAIN_DIM    ) 
        fn_log_dataframe(df_in_Sales_Domain_Estore,     STR_DF_IN_ESTORE              )
        fn_log_dataframe(df_in_Sales_Product_ASN,       STR_DF_IN_SALES_PRODUCT_ASN   )
        fn_log_dataframe(df_in_Forecast_Rule,           STR_DF_IN_FORECAST_RULE       )
        fn_log_dataframe(df_in_Item_Master,             STR_DF_IN_ITEM_MASTER         )
        fn_log_dataframe(df_in_Sell_Out_Simul_Master,       STR_DF_IN_SELL_OUT_SIMUL_MASTER  )
        fn_log_dataframe(df_in_Sell_Out_Simul_Master_Delta, STR_DF_IN_SELL_OUT_SIMUL_MASTER_D)

        # --------------------------------------------------------------------------
        # df_input 체크 종료
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # 입력 변수 확인
        # --------------------------------------------------------------------------
        if Version is None or Version.strip() == '':
            Version = 'CWV_DP'

        # --------------------------------------------------------------------------
        # 입력 변수 확인용 로그
        logger.Note(p_note=f'Param_OUT_VERSION : {Version}', p_log_level=LOG_LEVEL.debug())
        # --------------------------------------------------------------------------

        
        ################################################################################################################
        # Step 0 – Sell_Out_Simul_Master Delta 처리
        ################################################################################################################
        # 0-1) Delta Status 반영
        dict_log = {
            'p_step_no' : 1,
            'p_step_desc': 'Step 0-1 Apply Sell_Out_Simul_Master Delta'
        }
        df_step00_01_simul_master = fn_step00_01_apply_simul_delta(
            df_in_Sell_Out_Simul_Master,
            df_in_Sell_Out_Simul_Master_Delta,
            **dict_log
        )
        fn_log_dataframe(df_step00_01_simul_master, 'step00_01_df_simul_master')

        # 0-2) Output_Sell_Out_Simul_Master
        dict_log = {
            'p_step_no' : 2,
            'p_step_desc': 'Step 0-2 Build Output_Sell_Out_Simul_Master'
        }
        df_step00_02_out_simul_master = fn_step00_02_build_output_simul_master(
            df_in_Sell_Out_Simul_Master_Delta,
            Version,
            **dict_log
        )
        fn_log_dataframe(df_step00_02_out_simul_master, 'step00_02_Output_Sell_Out_Simul_Master')
        

        # 0-3) Output_Sell_Out_Simul_Master_Delta
        dict_log = {
            'p_step_no' : 3,
            'p_step_desc': 'Step 0-3 Build Output_Sell_Out_Simul_Master_Delta'
        }
        df_step00_03_out_simul_delta = fn_step00_03_build_output_simul_master_delta(
            df_in_Sell_Out_Simul_Master_Delta,
            Version,
            **dict_log
        )
        fn_log_dataframe(df_step00_03_out_simul_delta, f'step00_03_Output_Sell_Out_Simul_Master_Delta')

        # 0-4) Product Group 목록 추출
        dict_log = {
            'p_step_no' : 4,
            'p_step_desc': 'Step 0-4 – Extract Product Group list from Delta'
        }
        arr_step00_04_pg = fn_step00_04_extract_product_group_list(
            df_in_Sell_Out_Simul_Master_Delta,
            **dict_log
        )

        ################################################################################################################
        # Step 1 – Ship-To 차원 LUT 구축
        ################################################################################################################
        dict_log = {
            'p_step_no' : 10,
            'p_step_desc': 'Step 1 – load Ship-To dimension LUT'
        }
        df_fn_shipto_dim = step01_load_shipto_dimension(
            df_in_Sales_Domain_Dimension,
            **dict_log
        )
        fn_log_dataframe(df_fn_shipto_dim, 'step01_df_fn_shipto_dim')

        ################################################################################################################
        # Step 2 – Sales Product ASN 전처리 (+ Product Group 필터)
        ################################################################################################################
        dict_log = {
            'p_step_no' : 20,
            'p_step_desc': 'Step 2 – Sales Product ASN preprocess (with Simul Master & PG filter)'
        }
        df_fn_sales_product_asn = step02_preprocess_asn(
            df_in_Sales_Product_ASN,
            df_in_Item_Master,
            df_step00_01_simul_master,
            df_in_Sales_Domain_Estore,
            df_fn_shipto_dim,
            arr_step00_04_pg,
            **dict_log
        )
        fn_log_dataframe(df_fn_sales_product_asn, f'step02_{STR_DF_FN_SALES_PRODUCT_ASN}')

        ################################################################################################################
        # Step 3 – Sell-In Assortments (GC / AP2 / AP1 / Local-AP0)
        ################################################################################################################
        dict_log = {
            'p_step_no' : 30,
            'p_step_desc': 'Step 3 – Create Sell-In Assortments (GC/AP2/AP1/Local-AP0)'
        }
        df_sin_dict = fn_step03_00_create_sellin_assortments(
            df_fn_sales_product_asn,
            df_in_Forecast_Rule,
            df_fn_shipto_dim,
            **dict_log
        )
        for name, df in df_sin_dict.items():
            fn_log_dataframe(df, f'step03_{name}')

        ################################################################################################################
        # Step 4 – Sell-Out Assortments from Sell-In
        ################################################################################################################
        dict_log = {
            'p_step_no' : 40,
            'p_step_desc': 'Step 4 – Create Sell-Out Assortments (from Sell-In)'
        }
        df_sout_dict = fn_step04_00_create_sellout_assortments(
            df_sin_dict,
            **dict_log
        )
        for name, df in df_sout_dict.items():
            fn_log_dataframe(df, f'step04_{name}')

        ################################################################################################################
        # Step 5 – Formatter: Version 삽입 및 최종 Output 생성
        ################################################################################################################
        dict_log = {
            'p_step_no' : 50,
            'p_step_desc': 'Step 5 – Final Formatter (Version & column order)'
        }

        # 1) Formatter 호출
        #    - df_sout_dict         : Step-04 결과 (Sell-Out Assortment 4개 dict)
        #    - df_step00_02_out_simul_master : Step-0-2 결과 Simul Master 기준 테이블
        #    - df_step00_03_out_simul_delta  : Step-0-3 결과 Simul Master Delta 테이블
        df_final_all = fn_output_formatter(
            Version,                    # out_version (예: 'CWV_DP')
            df_sout_dict,               # Sell-Out Assortment dict
            df_step00_02_out_simul_master,
            df_step00_03_out_simul_delta,
            **dict_log
        )

        # 2) 최종 Output DataFrame 들에 대한 로그 출력
        #    - step05_Output_SOut_*, step05_Output_Sell_Out_Simul_* 형식으로 CSV/로그 남김
        for name, df_out in df_final_all.items():
            fn_log_dataframe(df_out, f'step05_{name}')

        # 3) MediumWeight 에서 참조할 변수명으로 바인딩
        #    - Output_* 이름은 o9 / 상위 프로세스에서 사용하는 Output 명세와 맞추기 위함
        Output_SOut_Assortment_GC       = df_final_all[STR_DF_OUT_SOUT_GC]
        Output_SOut_Assortment_AP2      = df_final_all[STR_DF_OUT_SOUT_AP2]
        Output_SOut_Assortment_AP1      = df_final_all[STR_DF_OUT_SOUT_AP1]
        Output_SOut_Assortment_Local    = df_final_all[STR_DF_OUT_SOUT_LOCAL]

        Output_Sell_Out_Simul_Master       = df_final_all[STR_DF_OUT_SIMUL_MASTER]
        Output_Sell_Out_Simul_Master_Delta = df_final_all[STR_DF_OUT_SIMUL_MASTER_DELTA]

    except Exception as e:
        trace_msg = traceback.format_exc()
        logger.Note(p_note=trace_msg, p_log_level=LOG_LEVEL.debug())
        logger.Error()
        if flag_exception:
            raise Exception(e)
        else:
            logger.info(f'{str_instance} exit - {time.strftime("%Y-%m-%d - %H:%M:%S")}')

    # else:
    #     logger.Finish()

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
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::') # 25.05.12 need warning Log by Logger Issue
        