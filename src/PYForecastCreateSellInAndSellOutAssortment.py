from re import X
from threading import local
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
str_instance = 'PYForecastCreateSellInAndSellOutAssortment'
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
COL_VERSION = 'Version.[Version Name]'
COL_ITEM    = 'Item.[Item]'
COL_PG      = 'Item.[Product Group]'
COL_SHIP_TO     = 'Sales Domain.[Ship To]'
COL_STD1        = 'Sales Domain.[Sales Std1]'
COL_STD2        = 'Sales Domain.[Sales Std2]'
COL_STD3        = 'Sales Domain.[Sales Std3]' 
COL_STD4        = 'Sales Domain.[Sales Std4]'
COL_STD5        = 'Sales Domain.[Sales Std5]'
COL_STD6        = 'Sales Domain.[Sales Std6]'
COL_LOC       = 'Location.[Location]'
Partial_Week            = 'Time.[Partial Week]'
SIn_FCST_GC_LOCK                = 'S/In FCST(GI)_GC.Lock'
SIn_FCST_Color_Condition        = 'S/In FCST Color Condition'
SIn_FCST_AP2_LOCK               = 'S/In FCST(GI)_AP2.Lock'
SIn_FCST_AP1_LOCK               = 'S/In FCST(GI)_AP1.Lock'

# Measure
COL_SOUT_ACT_P12 = 'S/Out Actual'     # df_in_Sell_Out_Actual_NoeStore_Past12 의 수량 컬럼 (실제 값은 Step6에서 안 씀)

# Salse_Product_ASN       = 'Sales Product ASN'    
COL_ASN_FLAG       = 'Sales Product ASN'
ITEMTAT_TATTERM         = 'ITEMTAT TATTERM'
ITEMTAT_TATTERM_SET     = 'ITEMTAT TATTERM_SET'
COL_FR_GC        = 'FORECAST_RULE GC FCST'
COL_FR_AP2       = 'FORECAST_RULE AP2 FCST'
COL_FR_AP1       = 'FORECAST_RULE AP1 FCST'  
COL_FR_AP0       = 'FORECAST_RULE AP0 FCST'  

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
COL_SOUT_ASS_FLAG    = 'S/Out Assortment Flag'

COL_SOUT_ASS_GC      = 'S/Out FCST Assortment_GC'
COL_SOUT_ASS_AP2     = 'S/Out FCST Assortment_AP2'
COL_SOUT_ASS_AP1     = 'S/Out FCST Assortment_AP1'
COL_SOUT_ASS_LOCAL   = 'S/Out FCST Assortment_Local'
COL_SIN_ASS_GC       = 'S/In FCST(GI) Assortment_GC'
COL_SIN_ASS_AP2      = 'S/In FCST(GI) Assortment_AP2'
COL_SIN_ASS_AP1      = 'S/In FCST(GI) Assortment_AP1'
COL_SIN_ASS_LOCAL    = 'S/In FCST(GI) Assortment_Local'

COL_LV_CODE     = 'LV_CODE'

# ----------------------------------------------------------------
# df_in_Sell_Out_Simul_Master
# ----------------------------------------------------------------
COL_MASTER_STATUS                     = 'S/Out Master Status'

# ───────────────────────────────────────────────────────────────
# CONSTANT STRING VARIABLES FOR DATAFRAME NAMES
# ───────────────────────────────────────────────────────────────
STR_DF_IN_SALES_DOMAIN_DIM              = 'df_in_Sales_Domain_Dimension'
STR_DF_IN_ESTORE                        = 'df_in_Sales_Domain_Estore'
STR_DF_IN_SALES_PRODUCT_ASN             = 'df_in_Sales_Product_ASN'
STR_DF_IN_FORECAST_RULE                 = 'df_in_Forecast_Rule'
STR_DF_IN_ITEM_MASTER                   = 'df_in_Item_Master'
STR_DF_IN_SELL_OUT_SIMUL_MASTER         = 'df_in_Sell_Out_Simul_Master'

STR_DF_FN_SALES_PRODUCT_ASN             = 'df_fn_Sales_Product_ASN'

STR_DF_OUT_SIN_GC            = 'Output_SIn_Assortment_GC'
STR_DF_OUT_SIN_AP2           = 'Output_SIn_Assortment_AP2'
STR_DF_OUT_SIN_AP1           = 'Output_SIn_Assortment_AP1'
STR_DF_OUT_SIN_LOCAL         = 'Output_SIn_Assortment_Local'

STR_DF_OUT_SOUT_GC            = 'Output_SOut_Assortment_GC'
STR_DF_OUT_SOUT_AP2           = 'Output_SOut_Assortment_AP2'
STR_DF_OUT_SOUT_AP1           = 'Output_SOut_Assortment_AP1'
STR_DF_OUT_SOUT_LOCAL         = 'Output_SOut_Assortment_Local'  # NEW

# Input-7: S/Out Actual No eStore Past12
STR_DF_IN_SOUT_ACT_NOESTORE_P12 = 'df_in_Sell_Out_Actual_NoeStore_Past12'

########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################

logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
# fn_set_local_logfile()
LOG_LEVEL = common.G_log_level

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


def initialize_max_week(is_local, args):
    global max_week, CurrentPartialWeek, max_week_normalized, current_week_normalized,chunk_size_step08 , chunk_size_step09, chunk_size_step10 , chunk_size_step11, chunk_size_step12, chunk_size_step13
    if is_local:
        # Read from MST_MODELEOS_TEST.csv
        df_eos = df_in_Time_Partial_Week
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


# ───────────────────────────────────────────────────────────────
# 0) HELPER – typed-casting with category for high-cardinality str
# ───────────────────────────────────────────────────────────────
def cast_cols(
    df: pd.DataFrame,
    cat_cols: list[str] = None,
    int_cols: list[str] = None,
    bool_cols: list[str] = None,
) -> None:
    """
    In-place dtype casting  
      • cat_cols → category  
      • int_cols → Int32 (nullable)  
      • bool_cols → boolean (pandas nullable)      **주의**  
    category 로 바꾼 컬럼을 `get_indexer` 에 사용할 때는  
    `df[col].cat.codes` 또는 `df[col].astype(str)` 로 변환해서 사용해야
    정확히 매칭됩니다.
    """
    if cat_cols:
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype("string").astype('category')
    if int_cols:
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int32')
    if bool_cols:
        for c in bool_cols:
            if c in df.columns:
                df[c] = df[c].astype('boolean',errors='ignore')


################################################################################################################
# Start Step Functions
################################################################################################################
# ──────────────────────────────────────────────────────────────────────────────
# STEP-01 : Ship-To Dimension LUT
#           (LV 코드 + 6 개 Std 컬럼 → 빠른 dict lookup 용)
# ──────────────────────────────────────────────────────────────────────────────
# * 2 → LV2(Std1) … 7 → LV7(Std6)
# * 메모리 절감 : category 캐스팅 & 중간 ndarray 즉시 해제
# * 반환 DF 컬럼
#     ─ COL_SHIP_TO  (PK)
#     ─ COL_STD1 … COL_STD6
#     ─ COL_LV_CODE  (int8)
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def step01_load_shipto_dimension(
        df_dim: pd.DataFrame
) -> pd.DataFrame:
    """
    Ship-To ⇢ LV_CODE & 6×Std 컬럼 LUT 생성.    Parameters
    ----------
    df_dim : pd.DataFrame
        `df_in_Sales_Domain_Dimension` 원본.

    Returns
    -------
    pd.DataFrame
        최소 컬럼 + category 캐스팅 + LV_CODE.
    """
    
    # ── 0) 필요한 컬럼만 선택 & category 캐스팅 ─────────────────────────────
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

    # ── 1) LV_CODE 계산 (벡터라이즈) ──────────────────────────────────────
    #    기본값 LV2; 뒤에서 매칭되면 overwrite
    lv_code = np.full(len(dim), 2, dtype='int8')
    ship_np = dim[COL_SHIP_TO].to_numpy()

    # (Std6 → LV7) 역순으로 매칭하여 “가장 하위” 코드가 남도록
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
        # del match                                           # 메모리 즉시 해제
        # gc.collect()

    dim[COL_LV_CODE] = lv_code

    # ── 2) 후처리 & 메모리 정리 ──────────────────────────────────────────
    del ship_np, lv_code, df_dim
    gc.collect()

    return dim


# ───────────────────────────────────────────────────────────────────────────────
# STEP-02 :  Sales-Product-ASN 전처리
#            • SOUT_ASS_FLAG 계산
#            • Product-Group 주입
#            • E-Store / Simul-Master 조건 반영
#            • 메모리 절감을 위해 category 사용
# ───────────────────────────────────────────────────────────────────────────────
# 반환 : df_fn_sales_product_asn  (다음 Step 들이 바로 사용)
# ───────────────────────────────────────────────────────────────────────────────
@_decoration_
def step02_preprocess_asn(
        df_asn:      pd.DataFrame,        # df_in_Sales_Product_ASN
        df_item:     pd.DataFrame,        # df_in_Item_Master
        df_master:   pd.DataFrame,        # df_in_Sell_Out_Simul_Master   (CON 목록)
        df_est:      pd.DataFrame,        # df_in_Sales_Domain_Estore
        df_dim_lut:  pd.DataFrame        # step01_load_shipto_dimension 결과 (Std5 매핑용)
) -> pd.DataFrame:
    """
    ▶ 출력 컬럼
       ─ COL_SHIP_TO
       ─ COL_ITEM
       ─ COL_LOC
       ─ COL_PG                (Product Group)
       ─ COL_ASN_FLAG          (Y/N)
       ─ COL_SOUT_ASS_FLAG     (bool)
    """
    # ──────────────────────────────────────────────────────────────────
    # 0) START : 필요한 컬럼만 keep  ➜ copy(deep=False)  (메모리 ↓)
    # ──────────────────────────────────────────────────────────────────
    use_cols = [COL_SHIP_TO, COL_ITEM, COL_LOC, COL_ASN_FLAG]
    asn = (
        df_asn[use_cols]
        .copy(deep=False)                       # ⬅︎ shallow copy
        .astype({
            COL_SHIP_TO:   'category',
            COL_ITEM:      'category',
            COL_LOC:       'category',
            COL_ASN_FLAG:  'category'           # ‘Y’ / ‘N’
        })
    )    
    # ──────────────────────────────────────────────────────────────────
    # 1) 기본 SOUT_ASS_FLAG = True
    # ──────────────────────────────────────────────────────────────────
    asn[COL_SOUT_ASS_FLAG] = True

    # ──────────────────────────────────────────────────────────────────
    # 2) Item Master join → Product Group 주입  (category 로 캐스팅)
    # ──────────────────────────────────────────────────────────────────
    item_small = (
        df_item[[COL_ITEM, COL_PG]]
        .copy(deep=False)
        .astype({COL_PG: 'category'})
    )
    asn = asn.merge(item_small, on=COL_ITEM, how='left')

    # ──────────────────────────────────────────────────────────────────
    # 3) Ship-To 를 Std5(Lv-6) 코드로 변환  → Simul Master lookup
    # ──────────────────────────────────────────────────────────────────
    std5_map = df_dim_lut.set_index(COL_SHIP_TO)[COL_STD5].to_dict()
    asn['_STD5'] = asn[COL_SHIP_TO].map(std5_map).astype('category')

    # Sell-Out Simul Master (CON) 집합  ➜ 벡터라이즈 비교용
    df_master_con = df_master.loc[df_master[COL_MASTER_STATUS] == 'CON']
    master_set = set(
        zip(
            df_master_con[COL_SHIP_TO].astype("string"),
            df_master_con[COL_PG].astype("string")
        )
    )

    # build boolean mask
    key_pairs = np.core.defchararray.add(
        asn['_STD5'].astype("string").to_numpy(dtype='U'), # ✓ '<U…' 로 변환
        asn[COL_PG].astype("string").to_numpy(dtype='U')
    )

    mask_in_master = np.in1d(
        key_pairs,
        np.fromiter(
            (a + b for a, b in master_set), dtype='<U32'
        )
    )

    # mask_has_con = np.fromiter(
    #     (k in master_set for k in key_pairs),                 # 제너레이터 → O(1) 메모리
    #     dtype=bool,
    #     count=key_pairs.size
    # )
    # 조건 : Simul-Master 에 존재하지 않으면 False
    asn.loc[~mask_in_master, COL_SOUT_ASS_FLAG] = False

    # ──────────────────────────────────────────────────────────────────
    # 4) E-Store Ship-To ➜ SOUT_ASS_FLAG = False
    # ──────────────────────────────────────────────────────────────────
    est_set = set(df_est[COL_SHIP_TO].astype("string"))
    mask_estore = asn[COL_SHIP_TO].astype("string").isin(est_set)
    asn.loc[mask_estore, COL_SOUT_ASS_FLAG] = False

    # ──────────────────────────────────────────────────────────────────
    # 5) 정리 : 임시 컬럼 삭제 / dtype 최적화
    # ──────────────────────────────────────────────────────────────────
    asn.drop(columns=['_STD5'], inplace=True)

    # ‘Sales Product ASN’ (COL_ASN_FLAG) 는 이후 MAX(Y/N) 집계에 쓰므로
    # category 로 둔 채 그대로 유지한다.

    # ──────────────────────────────────────────────────────────────────
    # 6) 반환
    # ──────────────────────────────────────────────────────────────────
    return asn

# ──────────────────────────────────────────────────────────────────────────────
# STEP-03  ▸  Sell-In Assortments (GC / AP2 / AP1 / Local-AP0)
# ------------------------------------------------------------------------------
# * 입력  : df_fn_Sales_Product_ASN   (Step-02 전처리 결과 – SOUT_ASS_FLAG 포함)
#           df_in_Forecast_Rule       (AP0 컬럼 추가됨)
#           df_fn_shipto_dim          (Step-01 – LV_CODE + Std1~6 LUT)
# * 출력  : Output_SIn_Assortment_[GC | AP2 | AP1 | Local]
#           (Sales_Product_ASN / SOUT_ASS_FLAG  컬럼은 Step-04에서 사용되므로 그대로 둔다)
# * 주요 변경점
#   ① AP0(Local) 룰 처리  → ‘S/In FCST(GI) Assortment_Local’ 생성
#   ② df_fn_Sales_Product_ASN 사용  → SOUT_ASS_FLAG·Sales_Product_ASN 컬럼 존재
#   ③ SOUT_ASS_FLAG( bool ) 은 groupby 시 **max()** 로 집계,  
#      Sales_Product_ASN( 'Y'/'N' ) 은 **max()** 로 집계해 ’Y’ 유지
#   ④ Ship-To 조상 제거 로직은 그대로 유지 (ancestors set 활용)
#   ⑤ 메모리 절감  → category 캐스팅 & 불필요 객체 즉시 `del` + `gc.collect()`
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# STEP-03  ▸  Sell-In Assortments (GC / AP2 / AP1 / Local-AP0)  — vectorised
# ------------------------------------------------------------------------------
# • 입력  : df_asn  (Step-02 결과; S/Out Assortment Flag, Sales Product ASN 포함)
#          df_rule (Forecast-Rule; GC/AP2/AP1/AP0 레벨 코드)
#          df_ship_dim (Ship-To ⇢ Std1~6 + LV_CODE)
# • 출력  : { STR_DF_OUT_SIN_GC, STR_DF_OUT_SIN_AP2, STR_DF_OUT_SIN_AP1, STR_DF_OUT_SIN_LOCAL }
#            각각의 DF 컬럼 = [ShipTo, Item, Loc, qty_col(=1 고정), Sales Product ASN(max), S/Out Assortment Flag(max)]
# • 설계  : PYDPMakeActualAndInventoryForecastLevel.Step05 구조를 그대로 따르되,
#          “상위 Ship-to 중복 제거” 단계는 생략(Builder 집계로 충분)
# • 메모리: 최종 join key(object) → category 캐스팅
# ──────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Q: groupby 가 느려요. 무엇을 바꾸면 빨라지나요?
# A: 아래 2단계로 고쳐 쓰면 대부분 빨라집니다.
#    (1) key 컬럼은 str이 아니라 **category** 로 묶어서 groupby
#    (2) 정말 큰 데이터(수천만 행)에서는 NumPy reduce(완전 벡터)로 집계
# ─────────────────────────────────────────────────────────────
# 1) Pandas만으로 빠르게: "groupby 전 key를 category로"
#    - 지금 코드에서 .astype(str) 로 바꾼 뒤 groupby 하셨는데,
#      문자열 groupby 는 해시 비용이 커져 느립니다.
#    - category(== 내부적으로 int 코드) 로 맞추면 groupby 가 훨씬 빨라집니다.
#    - Confluence/모바일 붙여넣기 때문에 str 이 필요하다면, **집계 후** 최종 단계에서만 str로 바꾸세요.
@_decoration_
def _fast_groupby_with_category(df_tag: pd.DataFrame, qty_col: str) -> pd.DataFrame:
    # ① key를 category로 (이미 category면 pass)
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOC):
        if df_tag[c].dtype.name != "category":
            df_tag[c] = df_tag[c].astype("category")

    # ② 불필요한 assign(1) → 집계 후에만 넣기
    out = (
        df_tag
        .groupby([COL_SHIP_TO, COL_ITEM, COL_LOC], as_index=False, sort=False, observed=True)
        [[COL_ASN_FLAG, COL_SOUT_ASS_FLAG]]
        .max()  # 'Y'/'N' 은 category/object여도 'max'가 'Y' 유지, bool은 True 유지
    )

    # ③ qty=1 고정, 메모리 절약형 정수
    out[qty_col] = np.int8(1)

    # ④ (필요하다면) 여기서만 str 로 바꾸기
    # out[[COL_SHIP_TO, COL_ITEM, COL_LOC]] = out[[COL_SHIP_TO, COL_ITEM, COL_LOC]].astype(str)

    return out[[COL_SHIP_TO, COL_ITEM, COL_LOC, qty_col, COL_ASN_FLAG, COL_SOUT_ASS_FLAG]]


# 2) 더 빠르게: 완전 벡터 NumPy 집계 (groupby 없이 수십~수백 ms/백만행)
#    - 아이디어:
#       * key 3개(ShipTo, Item, Loc)를 모두 category 코드로 변환
#       * 3중 코드를 하나의 group id로 압축 (gid = s + S*(i + I*l))
#       * 'Y'→1 / 'N'→0,  bool→0/1 로 변환 후, 각 gid에 대해 최대값을 np.maximum.at 로 계산
#       * 유니크 gid 별 첫 번째 레코드 인덱스를 뽑아 결과를 재구성
#    - 전제:
#       * COL_ASN_FLAG 가 'Y'/'N' 이라면 아래 매핑 사용
#       * key 3개가 이미 category 라면 그대로 cat.codes 사용
@_decoration_
def _ultra_fast_groupby_numpy(df_tag: pd.DataFrame, qty_col: str) -> pd.DataFrame:
    # ① key → category codes
    def _as_cat_codes(s: pd.Series):
        if s.dtype.name != "category":
            s = s.astype("category")
        return s.cat.codes.to_numpy(), s.cat.categories

    s_code, s_cats = _as_cat_codes(df_tag[COL_SHIP_TO])
    i_code, i_cats = _as_cat_codes(df_tag[COL_ITEM])
    l_code, l_cats = _as_cat_codes(df_tag[COL_LOC])

    S = len(s_cats) if len(s_cats) > 0 else 1
    I = len(i_cats) if len(i_cats) > 0 else 1

    # ② 단일 그룹 id
    gid = s_code.astype(np.int64) + S * (i_code.astype(np.int64) + I * l_code.astype(np.int64))

    # ③ 'Y'/'N' → 1/0, bool → 1/0
    asn = df_tag[COL_ASN_FLAG]
    if asn.dtype == object or str(asn.dtype).startswith("category"):
        asn_bin = (asn == 'Y').to_numpy(dtype=np.int8)
    else:
        # 이미 'Y'/'N' 대신 True/False 등으로 들어왔다면 적절히 변환
        asn_bin = (asn.astype("string") == 'Y').to_numpy(dtype=np.int8)

    sout_bin = df_tag[COL_SOUT_ASS_FLAG].to_numpy().astype(np.int8)

    # ④ gid 기준 정렬 후 run-length 인덱스
    order = np.argsort(gid, kind='mergesort')            # 안정 정렬 (첫 인덱스 유지)
    gid_sorted = gid[order]

    # 고유 그룹과 그 시작 위치
    uniq_gid, first_idx = np.unique(gid_sorted, return_index=True)

    # ⑤ 각 그룹의 최대값 계산 (np.maximum.reduceat 유사 패턴)
    asn_sorted  = asn_bin [order]
    sout_sorted = sout_bin[order]

    # 각 구간 끝 인덱스
    end_idx = np.empty_like(first_idx)
    end_idx[:-1] = first_idx[1:]
    end_idx[-1]  = gid_sorted.size

    # 그룹별 최대 (0/1 이므로 max == any)
    # 방법 A) 파편화 없이 한 번에
    grp_asn  = np.zeros_like(first_idx, dtype=np.int8)
    grp_sout = np.zeros_like(first_idx, dtype=np.int8)
    for k in range(first_idx.size):
        grp_asn [k] = asn_sorted [first_idx[k]:end_idx[k]].max()
        grp_sout[k] = sout_sorted[first_idx[k]:end_idx[k]].max()

    # ⑥ 대표(첫) 레코드에서 key 복원
    rep_rows = order[first_idx]   # 각 그룹의 대표 행 인덱스 (원래 df_tag 내 위치)
    ship_vals = df_tag[COL_SHIP_TO].to_numpy()[rep_rows]
    item_vals = df_tag[COL_ITEM   ].to_numpy()[rep_rows]
    loc_vals  = df_tag[COL_LOC    ].to_numpy()[rep_rows]

    # ⑦ 결과 DF
    out = pd.DataFrame({
        COL_SHIP_TO      : ship_vals,
        COL_ITEM         : item_vals,
        COL_LOC          : loc_vals,
        COL_ASN_FLAG     : np.where(grp_asn  == 1, 'Y', 'N'),
        COL_SOUT_ASS_FLAG: grp_sout.astype(bool),
    })
    out[qty_col] = np.int8(1)

    # ⑧ 메모리 절감 (원하시면 유지)
    out[[COL_SHIP_TO, COL_ITEM, COL_LOC]] = out[[COL_SHIP_TO, COL_ITEM, COL_LOC]].astype('category')

    return out[[COL_SHIP_TO, COL_ITEM, COL_LOC, qty_col, COL_ASN_FLAG, COL_SOUT_ASS_FLAG]]


# ──────────────────────────────────────────────────────────────────────────────
# Ultra-fast, NumPy-only groupby (generalised)
# ------------------------------------------------------------------------------
# • 목표: 거대한 DF에서 pandas.groupby 대신 **완전 벡터라이즈**로 집계
# • 키 컬럼 개수 제한 없음 (2, 3, 4…)
# • 지원 집계: 'sum', 'max', 'min', 'any', 'all', 'first', 'last', 'count'
#     - 'max'/'min'은 bool 또는 'Y'/'N' (문자) 플래그에도 안전하게 동작
# • 반환: key + 집계결과 컬럼들
# • 팁: 반환 후 qty=1 고정, Location='-' 등은 호출부에서 .assign 로 추가
# ──────────────────────────────────────────────────────────────────────────────
# @_decoration_
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

# ──────────────────────────────────────────────────────────────────────────────
# 사용 예시 ①  (Step-04 패턴: (ShipTo, Item) 기준, qty_in_col='sum', 플래그 'max')
# ------------------------------------------------------------------------------
# df_sout = ultra_fast_groupby_numpy_general(
#     df=df_filtered,                                      # S/Out Assortment Flag == True 로 필터된 DF
#     key_cols=[COL_SHIP_TO, COL_ITEM],                    # 그룹 키
#     aggs={
#         qty_in_col: 'sum',                               # 여러 row → 합계 (아래서 1로 리셋)
#         COL_ASN_FLAG: 'max',                             # 'Y'/'N' → 'Y' 유지
#         COL_SOUT_ASS_FLAG: 'max',                        # bool → True 유지
#     },
# )
# df_sout = df_sout.assign(**{qty_out_col: 1, COL_LOC: '-'})  # qty=1, Loc='-'
# df_sout[[COL_SHIP_TO, COL_ITEM, COL_LOC]] = df_sout[[COL_SHIP_TO, COL_ITEM, COL_LOC]].astype('category')


# ──────────────────────────────────────────────────────────────────────────────
# 사용 예시 ②  (Step-03 패턴: (ShipTo, Item, Loc) 기준, 플래그 max만 계산하고 qty=1 부여)
# ------------------------------------------------------------------------------
# df_tag = ultra_fast_groupby_numpy_general(
#     df=df_tag,                                          # 변환된 조상 ShipTo / Item / Loc 행들
#     key_cols=[COL_SHIP_TO, COL_ITEM, COL_LOC],
#     aggs={
#         COL_ASN_FLAG     : 'max',                       # 'Y'/'N'
#         COL_SOUT_ASS_FLAG: 'max',                       # bool
#     },
# )
# df_tag[q_col] = np.int8(1)

# ──────────────────────────────────────────────────────────────────────────────
# STEP-03 ▸ Sell-In Assortments (GC / AP2 / AP1 / Local-AP0) — FULLY VECTORIZED (no np.vectorize / no per-row loops)
# ------------------------------------------------------------------------------
# 핵심 최적화 포인트
#   1) parent_of(ship, lv)  계산을 30M 행에 직접 적용하지 않음
#      → ship을 factorize → "고유 Ship 목록(U개)"에만 조상(Std1~6)을 미리 계산(anc_table)
#      → 전체 행은 인덱싱으로 O(N) 한 번에 조회: anc_table[lv, ship_codes]
#   2) RULE 룩업도 30M 행에 직접 dict-get 하지 않음
#      → (pg, parent) 쌍을 "mask된 행"에서만 factorize → 고유 pair(K개) 추출
#      → df_rule (미리 MultiIndex) 로 고유 pair만 reindex 조회 → inv 인덱스로 원복
#   3) 목표 Ship 변환도 anc_table로 한 번에 처리: tgt = anc_table[lv_arr, ship_codes]
#   4) 메모리 절감: category 적극 사용, int8
#   5) 상위 Ship-to 중복 제거 단계는 불필요 (Builder 집계에서 자연히 정리)
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def step03_create_sellin_assortments(
        df_asn      : pd.DataFrame,   # Step-02 결과 (S/Out Flag, Sales Product ASN 포함)
        df_rule     : pd.DataFrame,   # Forecast-Rule (GC/AP2/AP1/AP0)
        df_ship_dim : pd.DataFrame    # Ship-To LUT (Std1~6 + LV_CODE)
) -> dict[str, pd.DataFrame]:    # ────────────────────────────────────────────────────────────────────
    # 0) 기본 가드
    # ────────────────────────────────────────────────────────────────────
    if df_asn.empty or df_rule.empty or df_ship_dim.empty:
        def _empty(qc: str) -> pd.DataFrame:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOC, qc, COL_ASN_FLAG, COL_SOUT_ASS_FLAG])
        return {
            STR_DF_OUT_SIN_GC   : _empty(COL_SIN_ASS_GC),
            STR_DF_OUT_SIN_AP2  : _empty(COL_SIN_ASS_AP2),
            STR_DF_OUT_SIN_AP1  : _empty(COL_SIN_ASS_AP1),
            STR_DF_OUT_SIN_LOCAL: _empty(COL_SIN_ASS_LOCAL),
        }

    # ────────────────────────────────────────────────────────────────────
    # 1) Ship 계층 & RULE 인덱스 준비
    # ────────────────────────────────────────────────────────────────────
    STD_COLS = [COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6]
    dim_idx  = df_ship_dim.set_index(COL_SHIP_TO)

    # LV_MAP: lv=1..6 → Ship → Std{lv}
    LV_MAP = {lv: dim_idx[STD_COLS[lv-1]].to_dict() for lv in range(1, 7)}
    # SHIP_LV: Ship → LV_CODE
    SHIP_LV = dim_idx[COL_LV_CODE].to_dict()

    # RULE index: (PG, Ship) → (gc, ap2, ap1, ap0)
    for c in (COL_FR_GC, COL_FR_AP2, COL_FR_AP1, COL_FR_AP0):
        df_rule[c] = df_rule[c].fillna(0).astype('int8')
    rule_idx = (df_rule
                .assign(**{COL_PG: df_rule[COL_PG].astype("string"),
                           COL_SHIP_TO: df_rule[COL_SHIP_TO].astype("string")})
                .set_index([COL_PG, COL_SHIP_TO])
                [[COL_FR_GC, COL_FR_AP2, COL_FR_AP1, COL_FR_AP0]]
                .astype('int8'))

    # ────────────────────────────────────────────────────────────────────
    # 2) Core Builder (완전 벡터)
    # ────────────────────────────────────────────────────────────────────
    def build_level(df_src: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # 2-1) 카테고리 코드를 사용해 고유/전체 분리
        ship_cat = df_src[COL_SHIP_TO]
        if ship_cat.dtype.name != 'category':
            ship_cat = ship_cat.astype('category')
        ship_codes   = ship_cat.cat.codes.to_numpy()                 # (N,)
        ships_unique = ship_cat.cat.categories.to_numpy()            # (U,)

        pg_cat = df_src[COL_PG]
        if pg_cat.dtype.name != 'category':
            pg_cat = pg_cat.astype('category')
        pg_codes   = pg_cat.cat.codes.to_numpy()                     # (N,)
        pg_unique  = pg_cat.cat.categories.to_numpy()                # (PgU,)

        item_vals = df_src[COL_ITEM].to_numpy()                      # (N,)
        loc_vals  = df_src[COL_LOC ].to_numpy()                      # (N,)
        asn_vals  = df_src[COL_ASN_FLAG].to_numpy()
        sout_vals = df_src[COL_SOUT_ASS_FLAG].to_numpy()

        N, U = ship_codes.size, ships_unique.size

        # 2-2) anc_table: 요청 레벨(행) × 고유 Ship(열) → 조상 Ship 문자열
        #      인덱스 0,1 은 사용 안함(None), 2..7만 유효
        anc_table = np.empty((8, U), dtype=object)
        anc_table[0] = None
        anc_table[1] = None
        for lv in range(2, 8):
            mapper = LV_MAP[lv-1]
            anc_table[lv] = np.fromiter((mapper.get(s) for s in ships_unique), dtype=object, count=U)

        # 2-3) 본 Ship 레벨 (요청 레벨 ≤ 본 레벨 필터용)
        ship_lv_unique = np.fromiter((SHIP_LV.get(s, 99) for s in ships_unique), dtype=np.int8, count=U)
        ship_lv_arr    = ship_lv_unique[ship_codes]

        # 2-4) 태그별 레벨 매트릭스
        TAGS      = ('GC', 'AP2', 'AP1', 'Local')
        tag_lvmat = {t: np.zeros(N, dtype=np.int8) for t in TAGS}

        # 2-5) RULE 벡터 룩업(6회 루프) — 고유 pair만 df_rule에서 한 번에 조회
        for lv in range(1, 7):
            # parent for all rows at this lv (문자열)
            parent = anc_table[lv][ship_codes]                      # (N,)
            mask   = parent != None
            if not mask.any():
                continue

            # (pg, parent) pair → 고유 key
            idx = np.flatnonzero(mask)
            parent_sub  = parent[mask]                              # (M,)
            pg_code_sub = pg_codes[mask]                            # (M,)

            # parent_sub factorize → [0..P-1] + uniques (길이 P)
            parent_codes_sub, parent_uniques = pd.factorize(parent_sub, sort=False)
            P = int(parent_uniques.size)

            # unique pair key (정수): pg_code * P + parent_code
            pair_key = pg_code_sub.astype(np.int64) * P + parent_codes_sub.astype(np.int64)
            uniq_key, inv = np.unique(pair_key, return_inverse=True)
            uniq_pg_codes    = (uniq_key // P).astype(np.int64)     # (K,)
            uniq_parent_idx  = (uniq_key %  P).astype(np.int64)     # (K,)

            # 문자열로 복원 (K,)
            uniq_pg_str   = pg_unique[uniq_pg_codes]
            uniq_parent_s = parent_uniques[uniq_parent_idx]

            # RULE 인덱스 조회 (K,) → (K,4) 레벨 행렬
            mi = pd.MultiIndex.from_arrays([uniq_pg_str, uniq_parent_s])
            # reindex 후 결측은 0
            rule_mat = (rule_idx.reindex(mi)
                                  .fillna(0)
                                  .astype('int8')
                                  .to_numpy(copy=False))            # (K,4) int8

            # 행 수준 룩업 (inv로 복원) → (M,4)
            rv = rule_mat[inv]                                      # (M,4)
            gc_lv, ap2_lv, ap1_lv, ap0_lv = rv.T

            # 하위 매칭이 상위 매칭 덮어쓰기 (비0만 갱신)
            tag_lvmat['GC']   [idx] = np.where(gc_lv  != 0, gc_lv,  tag_lvmat['GC']   [idx])
            tag_lvmat['AP2']  [idx] = np.where(ap2_lv != 0, ap2_lv, tag_lvmat['AP2']  [idx])
            tag_lvmat['AP1']  [idx] = np.where(ap1_lv != 0, ap1_lv, tag_lvmat['AP1']  [idx])
            tag_lvmat['Local'][idx] = np.where(ap0_lv != 0, ap0_lv, tag_lvmat['Local'][idx])

        # 2-6) 태그별 결과 DataFrame 생성 (완전 벡터)
        qty_name = {
            'GC'   : COL_SIN_ASS_GC,
            'AP2'  : COL_SIN_ASS_AP2,
            'AP1'  : COL_SIN_ASS_AP1,
            'Local': COL_SIN_ASS_LOCAL,
        }

        frames: dict[str, pd.DataFrame] = {}

        for tag in TAGS:
            lv_arr = tag_lvmat[tag]
            # 유효 행: 2..7 & 요청 ≤ 본 레벨
            valid = (lv_arr >= 2) & (lv_arr <= 7) & (lv_arr <= ship_lv_arr)
            if not valid.any():
                frames[tag] = pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOC,
                                                    qty_name[tag], COL_ASN_FLAG, COL_SOUT_ASS_FLAG])
                continue

            # 목표 Ship 벡터 변환 (row-wise 2D 인덱싱)
            # 안전하게: 무효는 0행(None) 참조
            safe_lv = lv_arr.copy()
            safe_lv[~valid] = 0
            tgt_ship = anc_table[safe_lv, ship_codes]               # (N,)
            tgt_ship = tgt_ship[valid]                              # (n_valid,)

            df_tag = pd.DataFrame({
                COL_SHIP_TO      : tgt_ship,
                COL_ITEM         : item_vals[valid],
                COL_LOC          : loc_vals [valid],
                COL_ASN_FLAG     : asn_vals [valid],
                COL_SOUT_ASS_FLAG: sout_vals[valid],
            })

            q = qty_name[tag]

            # # groupby 전에 key를 문자열로 맞춰 카테고리 충돌 방지(모바일/Confluence 호환)
            # df_tag[COL_SHIP_TO] = df_tag[COL_SHIP_TO].astype(str)
            # df_tag[COL_ITEM]    = df_tag[COL_ITEM].astype(str)
            # df_tag[COL_LOC]     = df_tag[COL_LOC].astype(str)

            # df_tag = (
            #     df_tag
            #     .groupby([COL_SHIP_TO, COL_ITEM, COL_LOC], as_index=False, sort=False, observed=True)
            #     .agg({
            #         COL_ASN_FLAG     : 'max',   # 'Y'/'N' → 'Y' 유지
            #         COL_SOUT_ASS_FLAG: 'max',   # bool   → True 유지
            #     })
            #     .assign(**{q: 1})
            # )

            # 초대형이면 NumPy
            dict_log_sub = {
                'p_step_no'  : 310,
                'p_step_desc': f'Step-03  Groupby {tag}'
            }
            df_tag = _ultra_fast_groupby_numpy(df_tag, q, **dict_log_sub)

            # # ──────────────────────────────────────────────────────────────────────────────
            # # 사용 예시 ②  (Step-03 패턴: (ShipTo, Item, Loc) 기준, 플래그 max만 계산하고 qty=1 부여)
            # # ------------------------------------------------------------------------------
            # df_tag = ultra_fast_groupby_numpy_general(
            #     df=df_tag,                                          # 변환된 조상 ShipTo / Item / Loc 행들
            #     key_cols=[COL_SHIP_TO, COL_ITEM, COL_LOC],
            #     aggs={
            #         COL_ASN_FLAG     : 'max',                       # 'Y'/'N'
            #         COL_SOUT_ASS_FLAG: 'max',                       # bool
            #     }
            # )
            # df_tag[q] = np.int8(1)            



            frames[tag] = df_tag

        return {
            STR_DF_OUT_SIN_GC   : frames['GC'],
            STR_DF_OUT_SIN_AP2  : frames['AP2'],
            STR_DF_OUT_SIN_AP1  : frames['AP1'],
            STR_DF_OUT_SIN_LOCAL: frames['Local'],
        }

    # ────────────────────────────────────────────────────────────────────
    # 3) 실행 & 반환
    # ────────────────────────────────────────────────────────────────────
    out_dict = build_level(df_asn)

    del df_asn, df_rule, df_ship_dim
    gc.collect()
    return out_dict

# ─────────────────────────────────────────────────────────────────────────────
#  STEP-04  ▸  Sell-Out Assortments  (GC / AP2 / AP1 / Local-AP0)
# -----------------------------------------------------------------------------
#   • 입력  : Step-03 의 4개 Sell-In DataFrame (dict 로 전달)
#   • 로직  :
#       1) SOUT_ASS_FLAG == False 행 **제거**
#       2) Ship-To & Item 기준 groupby
#            – qty     : sum  → 다시 1 로 치환
#            – ASN_FLAG: max  ( 'Y'>'N' )
#            – SOUT_FLAG: max ( True>False )
#       3) Location ‘-’ 컬럼 추가
#   • 출력  : 4개 Sell-Out DataFrame 을 dict 로 반환
# ─────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# STEP-04  ▸  Sell-Out Assortments (GC / AP2 / AP1 / Local-AP0)
# ------------------------------------------------------------------------------
# • Step-03 결과(df_sin_dict)에서 S/Out 전개용 행만 추리고, Ship-To·Item 단위로 묶어
#   수량(=1)을 세팅, Location 은 모두 '-' 로 통일.
# • ⚠️ pandas groupby + categorical 조합에서 카테고리의 "미관측 조합" 때문에
#   카테고리의 데카르트 곱이 생겨 IndexError/Length mismatch 가 날 수 있음.
#   → 해결: groupby(observed=True) + 그룹 키를 일시적으로 문자열(object)로 변환.
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def step04_create_sellout_assortments(
        df_sin_dict : dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Ultra-fast aggregator (NumPy-based) to replace the Pandas groupby in Step-04
    # -----------------------------------------------------------------------------
    # • Key: (Sales Domain.[Ship To], Item.[Item])  — Location 은 Step-04 규칙대로 '-' 로 고정
    # • Agg:  qty_out_col = 1 (중복 제거 목적의 집계)
    #         COL_ASN_FLAG      = max('Y'/'N')  → 그룹 내 하나라도 'Y'면 'Y'
    #         COL_SOUT_ASS_FLAG = max(bool)     → 그룹 내 하나라도 True면 True
    # • 내부 동작:
    #     1) 문자열 키(ShipTo + \x1f + Item)로 벡터 생성
    #     2) pandas.factorize (C-최적화 해시)로 ‘그룹코드’ 산출 (정렬 無)
    #     3) np.bincount 로 각 그룹의 max 집계 계산
    #     4) 유니크 키를 분해해 ShipTo, Item 복원 + Location='-'
    # • 주의:
    #     - 호출 전, 이미 df = df_in.loc[df_in[COL_SOUT_ASS_FLAG]] 처럼 True 필터가 적용되어 있어야 함
    #     - 키 컬럼이 categorical 인 경우에도 astype(str) 로 안전하게 처리
    #     - 리턴 시 키 컬럼은 category 로 환원(메모리 절감)
    # ─────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────
    # Ultra-fast aggregator (NumPy-based) — fixed for np.char.add TypeError
    # -----------------------------------------------------------------------------
    # 핵심 수정:
    #   • ship / item 을 반드시 "NumPy 유니코드 배열(dtype='U')" 로 강제 변환
    #     → np.char.add 가 object dtype 에서 발생시키는
    #       "string operation on non-string array" 를 방지
    #   • 대량데이터 안전성(결측치, 카테고리) 고려
    # 사용처:
    #   df = _ultra_fast_groupby_numpy(df, qty_in_col=qty_in_col, qty_out_col=qty_out_col)
    # ─────────────────────────────────────────────────────────────────────────────
    def _ultra_fast_groupby_numpy(
        df: pd.DataFrame,
        *,
        qty_in_col: str,
        qty_out_col: str,
        to_category: bool = True,   # 결과 키를 category 로 환원할지 여부 (메모리 절감)
    ) -> pd.DataFrame:
        # 빈 입력 대비
        if df is None or df.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOC,
                                        COL_ASN_FLAG, COL_SOUT_ASS_FLAG, qty_out_col])    # 1) 벡터 추출 — 반드시 NumPy "유니코드"로 강제 변환 (dtype='U')
        #    (object → U 로 바꾸면 np.char.* 연산이 안전하게 동작)
        s_ship = df[COL_SHIP_TO]
        s_item = df[COL_ITEM]

        # 결측이 있으면 먼저 채움 (astype(str)만 쓰면 'nan' 문자열이 생길 수 있음)
        if s_ship.hasnans:
            s_ship = s_ship.fillna('')
        if s_item.hasnans:
            s_item = s_item.fillna('')

        # pandas → numpy 유니코드 배열
        ship = np.asarray(s_ship.astype("string").to_numpy(copy=False), dtype=np.str_)
        item = np.asarray(s_item.astype("string").to_numpy(copy=False), dtype=np.str_)

        # 'Y'/'N' → 1/0 (카테고리여도 안전)
        asn_is_y = (np.asarray(df[COL_ASN_FLAG].astype("string").to_numpy(copy=False), dtype=np.str_) == 'Y').astype(np.int8)
        # bool → 1/0
        sout_1_0 = df[COL_SOUT_ASS_FLAG].to_numpy(dtype=np.int8, copy=False)

        # 2) 2-키를 단일 문자열 키로 결합 (벡터화)
        sep = np.str_('\x1f')   # Unit Separator
        key = np.char.add(np.char.add(ship, sep), item)   # ← dtype='U' 이므로 안전

        # 3) 해시 기반 factorize (정렬 無)
        codes, uniques = pd.factorize(key, sort=False)
        ng = int(codes.max()) + 1 if codes.size else 0
        if ng == 0:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOC,
                                        COL_ASN_FLAG, COL_SOUT_ASS_FLAG, qty_out_col])

        # 4) 그룹별 max 계산 (np.bincount 로 O(n))
        asn_cnt  = np.bincount(codes, weights=asn_is_y, minlength=ng)
        sout_cnt = np.bincount(codes, weights=sout_1_0, minlength=ng)

        asn_out  = np.where(asn_cnt  > 0, 'Y', 'N')
        sout_out = (sout_cnt > 0)

        # 5) 유니크 키 분해 (ShipTo, Item) + Location ‘-’
        parts  = np.char.partition(np.asarray(uniques, dtype=np.str_), sep)
        ship_u = parts[:, 0]
        item_u = parts[:, 2]
        loc_u  = np.repeat('-', ng)

        # 6) 결과 DF 구성
        out = pd.DataFrame({
            COL_SHIP_TO      : ship_u,
            COL_ITEM         : item_u,
            COL_LOC          : loc_u,
            COL_ASN_FLAG     : asn_out,
            COL_SOUT_ASS_FLAG: sout_out,
            qty_out_col      : np.ones(ng, dtype=np.int8),
        })

        # 7) 메모리 절감 (옵션)
        if to_category:
            out[[COL_SHIP_TO, COL_ITEM, COL_LOC]] = out[[COL_SHIP_TO, COL_ITEM, COL_LOC]].astype('category')

        return out

    def _ultra_fast_groupby_polars(df_in: pd.DataFrame,
                                qty_in_col: str,
                                qty_out_col: str) -> pd.DataFrame:
        """
        Ship-To + Item 단위 집계를 Polars로 초고속 수행 후, Pandas로 되돌려줍니다.
        - 입력 df_in 은 이미 COL_SOUT_ASS_FLAG == True 로 필터된 상태가 이상적입니다.
        - groupby key: [COL_SHIP_TO, COL_ITEM]  (Location 제외)
        - agg: {qty_in_col: sum, COL_ASN_FLAG: max, COL_SOUT_ASS_FLAG: max}
        - post: qty_out_col = 1, COL_LOC = '-'
        - 반환: Pandas DataFrame (키는 category 캐스팅, qty_out_col=int8)
        """
        try:
            import polars as pl
        except ImportError as e:
            raise RuntimeError("polars 가 설치되어 있지 않습니다. `pip install polars` 후 다시 실행하세요.") from e

        # 필요한 컬럼만 슬라이스 (불필요한 전송 최소화)
        need_cols = [COL_SHIP_TO, COL_ITEM, qty_in_col, COL_ASN_FLAG, COL_SOUT_ASS_FLAG]
        df = df_in[need_cols].copy()

        # groupby 안정성을 위해 문자열 보장 (카테고리 → str)
        df[COL_SHIP_TO] = df[COL_SHIP_TO].astype("string")
        df[COL_ITEM]    = df[COL_ITEM].astype("string")

        # Pandas → Polars
        pl_df = pl.from_pandas(df, include_index=False)

        # dtype 정리 (명시적 캐스팅: 문자열/불리언/정수)
        pl_df = pl_df.with_columns([
            pl.col(COL_SHIP_TO).cast(pl.Utf8),
            pl.col(COL_ITEM).cast(pl.Utf8),
            pl.col(COL_SOUT_ASS_FLAG).cast(pl.Boolean),
            # qty_in_col 이 존재하지 않을 수 있는 상황 방지 (없으면 0으로 생성)
        ])
        if qty_in_col not in pl_df.columns:
            pl_df = pl_df.with_columns(pl.lit(0).alias(qty_in_col))
        # 'Y'/'N' 문자열 max 유지: Utf8 그대로 max 가능 (Y > N)

        # ── Polars groupby ─────────────────────────────────────────────────────
        # sum + max 집계
        agged = (
            pl_df
            .group_by([COL_SHIP_TO, COL_ITEM])
            .agg([
                pl.col(qty_in_col).sum().alias(qty_in_col),
                pl.col(COL_ASN_FLAG).max().alias(COL_ASN_FLAG),
                pl.col(COL_SOUT_ASS_FLAG).max().alias(COL_SOUT_ASS_FLAG),
            ])
            # 고정 컬럼 추가
            .with_columns([
                pl.lit('-').alias(COL_LOC),
                pl.lit(1).alias(qty_out_col),
            ])
            # 출력 컬럼 순서 정리
            .select([COL_SHIP_TO, COL_ITEM, COL_LOC,
                    COL_ASN_FLAG, COL_SOUT_ASS_FLAG, qty_out_col])
        )

        # Polars → Pandas
        out = agged.to_pandas(use_pyarrow_extension_array=False)

        # 메모리 절감 (키는 category, qty는 int8)
        out[qty_out_col] = out[qty_out_col].astype('int8')
        out[[COL_SHIP_TO, COL_ITEM, COL_LOC]] = out[[COL_SHIP_TO, COL_ITEM, COL_LOC]].astype('category')

        return out

    # ── 헬퍼 ────────────────────────────────────────────────────────────────
    def _to_sout(df_in: pd.DataFrame,
                 qty_in_col: str,
                 qty_out_col: str) -> pd.DataFrame:

        if df_in is None or df_in.empty:
            # 빈 경우에도 downstream 병합이 편하도록 스키마 유지
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOC,
                                         COL_ASN_FLAG, COL_SOUT_ASS_FLAG, qty_out_col])

        # 1) SOUT_ASS_FLAG == True 만 유지
        df = df_in.loc[df_in[COL_SOUT_ASS_FLAG]].copy()
        if df.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOC,
                                         COL_ASN_FLAG, COL_SOUT_ASS_FLAG, qty_out_col])

        # 3) Ship-To & Item groupby (Location 제외)
        df = _ultra_fast_groupby_numpy(df, qty_in_col=qty_in_col, qty_out_col=qty_out_col)

        # 2) Polars 초고속 groupby
        # df = _ultra_fast_groupby_polars(df, qty_in_col, qty_out_col)
        return df


    # ── 각각 처리 ───────────────────────────────────────────────────────────
    out_dict: dict[str, pd.DataFrame] = {
        STR_DF_OUT_SOUT_GC    : _to_sout(df_sin_dict.get(STR_DF_OUT_SIN_GC),
                                        COL_SIN_ASS_GC,    COL_SOUT_ASS_GC),
        STR_DF_OUT_SOUT_AP2   : _to_sout(df_sin_dict.get(STR_DF_OUT_SIN_AP2),
                                        COL_SIN_ASS_AP2,   COL_SOUT_ASS_AP2),
        STR_DF_OUT_SOUT_AP1   : _to_sout(df_sin_dict.get(STR_DF_OUT_SIN_AP1),
                                        COL_SIN_ASS_AP1,   COL_SOUT_ASS_AP1),
        STR_DF_OUT_SOUT_LOCAL : _to_sout(df_sin_dict.get(STR_DF_OUT_SIN_LOCAL),
                                        COL_SIN_ASS_LOCAL, COL_SOUT_ASS_LOCAL),
    }

    return out_dict

###############################################################################
#  📌  CONSTANTS ― Step-05  (E-Store Sell-Out)
###############################################################################
# ── column constants ─────────────────────────────────────────────────────────
# 이미 앞 단계에서 선언된 컬럼/DF 상수를 그대로 재사용합니다.
# (여기선 참조만 표기)
#   COL_SHIP_TO, COL_ITEM, COL_LOC
#   COL_ASN_FLAG, COL_SOUT_ASS_FLAG
#   COL_SIN_ASS_GC … COL_SIN_ASS_LOCAL
#   COL_SOUT_ASS_GC … COL_SOUT_ASS_LOCAL
#   COL_STD1 … COL_STD6, COL_FR_GC … COL_FR_AP0
#
# ── DataFrame-name constants ────────────────────────────────────────────────
#   STR_DF_OUT_SOUT_GC  … STR_DF_OUT_SOUT_LOCAL
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
#  STEP-05 ▸  E-Store Sell-Out Assortments
#            (GC / AP2 / AP1 / Local-AP0)  +  Step-04 결과와 병합
# -----------------------------------------------------------------------------
#   • 입력
#       df_asn          : Step-02 결과  (df_fn_Sales_Product_ASN)
#       df_rule         : Forecast-Rule (AP0 포함)
#       df_ship_dim     : Step-01 Ship-To LUT
#       df_est          : df_in_Sales_Domain_Estore
#       df_sout_dict    : Step-04 Sell-Out 결과 dict   (중복 제거용 concat)
#   • 로직
#       1) ASN ∩  E-Store Ship-To 필터
#       2) Step-03 와 동일한 RULE 매칭 로직으로 “Sell-Out” row‐buffer 생성
#          (단, qty 컬럼은 SOUT 이름으로 바로 생성)
#       3) groupby(Ship-To+Item) + Location ‘-’ 추가
#       4) Step-04 결과와 concat → 최종 dict 반환
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  STEP-05 ▸  E-Store Sell-Out Assortments (GC/AP2/AP1/Local-AP0) — vectorised
# -----------------------------------------------------------------------------
#   • 입력
#       df_asn       : Step-02 결과 (df_fn_Sales_Product_ASN; ASN + flags 포함)
#       df_rule      : Forecast-Rule (GC/AP2/AP1/AP0)
#       df_ship_dim  : Step-01 Ship-To LUT (Std1~6 + LV_CODE)
#       df_est       : E-Store Ship-To 목록 (df_in_Sales_Domain_Estore)
#       df_sout_dict : Step-04 Sell-Out 결과 dict (병합 대상)
#   • 로직
#       0) E-Store Ship-To 필터
#       1) RULE / LV_MAP / SHIP_LV 준비
#       2) Core Builder (vectorised, +Local tag)
#          - 벡터 슬라이스 생성 (ship/item/pg/asn/sout)
#          - Std1~Std6 6회 루프(행 수 비례 無): 태그별 최종 레벨 결정
#          - “요청 레벨 ≤ 본 Ship-to 레벨” 필터
#          - Ship-to → 조상 Ship-to 벡터 변환 → (ShipTo, Item) 집계
#            * groupby 전에 문자열로 변환(카테고리 이슈 회피), qty=1 고정
#       3) Step-04 결과와 concat(+중복제거) 후 반환
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def step05_create_sellout_assortments_estore(
        df_asn       : pd.DataFrame,            # Step-02 결과 (df_fn_Sales_Product_ASN; ASN + flags 포함)
        df_rule      : pd.DataFrame,            # Forecast-Rule (GC/AP2/AP1/AP0)
        df_ship_dim  : pd.DataFrame,            # Step-01 Ship-To LUT (Std1~6 + LV_CODE)
        df_est       : pd.DataFrame,            # E-Store Ship-To 목록 (df_in_Sales_Domain_Estore)
        df_sout_dict : dict[str, pd.DataFrame], # Step-04 Sell-Out 결과 dict (병합 대상)
        **kwargs
) -> dict[str, pd.DataFrame]:    
    
    # ────────────────────────────────────────────────────────────────────
    # 0) E-Store Ship-To 필터
    # ────────────────────────────────────────────────────────────────────
    est_set = set(df_est[COL_SHIP_TO].astype("string"))
    df_es   = df_asn[df_asn[COL_SHIP_TO].astype("string").isin(est_set)].copy()
    if df_es.empty:
        return df_sout_dict

    # ────────────────────────────────────────────────────────────────────
    # 1) RULE / LV_MAP / SHIP_LV 준비
    # ────────────────────────────────────────────────────────────────────
    STD_COLS = [COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6]
    dim_idx  = df_ship_dim.set_index(COL_SHIP_TO)
    LV_MAP   = {lv: dim_idx[STD_COLS[lv-1]].to_dict() for lv in range(1, 7)}   # lv=1⇒Std1(=Lv-2) …
    parent_of = lambda arr, lv: np.vectorize(LV_MAP[lv].get, otypes=[object])(arr)
    SHIP_LV  = dim_idx[COL_LV_CODE].to_dict()                                  # {'300114':3, 'A300114':4, …}

    for c in (COL_FR_GC, COL_FR_AP2, COL_FR_AP1, COL_FR_AP0):
        df_rule[c] = df_rule[c].fillna(0).astype('int8')
    RULE = {
        (str(pg), str(st)): (int(gc), int(ap2), int(ap1), int(ap0))
        for pg, st, gc, ap2, ap1, ap0 in zip(
            df_rule[COL_PG], df_rule[COL_SHIP_TO],
            df_rule[COL_FR_GC], df_rule[COL_FR_AP2],
            df_rule[COL_FR_AP1], df_rule[COL_FR_AP0]
        )
    }

    # ────────────────────────────────────────────────────────────────────
    # 2) Core Builder (vectorised)
    # ────────────────────────────────────────────────────────────────────
    def build_level(df_src: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # ── vector slices ────────────────────────────────────────────────
        ship = df_src[COL_SHIP_TO].astype("string").to_numpy()
        item = df_src[COL_ITEM].to_numpy()
        loc  = df_src[COL_LOC].to_numpy()
        pg   = df_src[COL_PG].astype("string").to_numpy()
        asn  = df_src[COL_ASN_FLAG].to_numpy()
        sout = df_src[COL_SOUT_ASS_FLAG].to_numpy()

        N = ship.size
        if N == 0:
            return {
                'GC':    pd.DataFrame(),
                'AP2':   pd.DataFrame(),
                'AP1':   pd.DataFrame(),
                'Local': pd.DataFrame(),
            }

        TAGS      = ('GC', 'AP2', 'AP1', 'Local')
        tag_lvmat = {t: np.zeros(N, dtype='int8') for t in TAGS}

        # Std1~Std6 (6회)만 Python 루프 — 행 수(n)과 무관
        for lv in range(1, 7):
            # parent = np.vectorize(LV_MAP[lv].get, otypes=[object])(ship)  # Ship → lv 조상
            parent = parent_of(ship, lv)
            mask   = parent != None
            if not mask.any():
                continue

            idx = np.flatnonzero(mask)
            rule_vec = np.array(
                [RULE.get((pg[i], parent[i]), (0, 0, 0, 0)) for i in idx],
                dtype='int8'
            )
            gc_lv, ap2_lv, ap1_lv, ap0_lv = rule_vec.T

            # 하위 매칭이 상위 매칭을 덮어씀(비0값 업데이트)
            tag_lvmat['GC']   [idx] = np.where(gc_lv  != 0, gc_lv,  tag_lvmat['GC']   [idx])
            tag_lvmat['AP2']  [idx] = np.where(ap2_lv != 0, ap2_lv, tag_lvmat['AP2']  [idx])
            tag_lvmat['AP1']  [idx] = np.where(ap1_lv != 0, ap1_lv, tag_lvmat['AP1']  [idx])
            tag_lvmat['Local'][idx] = np.where(ap0_lv != 0, ap0_lv, tag_lvmat['Local'][idx])

        # 본 Ship-to 의 현재 레벨 (룩업 벡터)
        ship_lv_arr = np.fromiter((SHIP_LV.get(s, 99) for s in ship), dtype='int8')

        qty_name = {
            'GC'   : COL_SOUT_ASS_GC,
            'AP2'  : COL_SOUT_ASS_AP2,
            'AP1'  : COL_SOUT_ASS_AP1,
            'Local': COL_SOUT_ASS_LOCAL,
        }

        frames: dict[str, pd.DataFrame] = {}
        for tag in TAGS:
            lv_arr = tag_lvmat[tag]
            # 요청 레벨 2..7 이고, “요청 레벨 ≤ 본 Ship-to 레벨” 만 유효
            valid = (lv_arr >= 2) & (lv_arr <= 7) & (lv_arr <= ship_lv_arr)
            if not valid.any():
                frames[tag] = pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOC,
                                                    qty_name[tag], COL_ASN_FLAG, COL_SOUT_ASS_FLAG])
                continue

            # Ship-to → 조상 Ship-to 로 벡터 변환
            tgt_ship = np.fromiter(
                (LV_MAP[l-1].get(s) for s, l in zip(ship[valid], lv_arr[valid])),
                dtype=object
            )
            ok = tgt_ship != None
            if not ok.any():
                frames[tag] = pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOC,
                                                    qty_name[tag], COL_ASN_FLAG, COL_SOUT_ASS_FLAG])
                continue

            df_tag = pd.DataFrame({
                COL_SHIP_TO      : tgt_ship[ok],
                COL_ITEM         : item[valid][ok],
                COL_LOC          : loc [valid][ok],
                COL_ASN_FLAG     : asn [valid][ok],
                COL_SOUT_ASS_FLAG: sout[valid][ok],
            })

            # groupby 前: 카테고리→문자열 (모바일/Confluence 이슈 회피)
            df_tag[COL_SHIP_TO] = df_tag[COL_SHIP_TO].astype("string")
            df_tag[COL_ITEM]    = df_tag[COL_ITEM].astype("string")
            df_tag[COL_LOC]     = df_tag[COL_LOC].astype("string")

            q = qty_name[tag]
            
            # # (ShipTo, Item) 집계 → qty=1 리셋
            # df_tag = (
            #     df_tag
            #     # .assign(**{q: 1})
            #     .groupby([COL_SHIP_TO, COL_ITEM,COL_LOC], as_index=False, sort=False)
            #     .agg({
            #         # q: 'sum', 
            #         COL_ASN_FLAG: 'max', 
            #         COL_SOUT_ASS_FLAG: 'max'
            #     })
            #     .assign(**{q: 1})
            # )

            dict_log_sub = {
                'p_step_no'  : 510,
                'p_step_desc': f'Step-05  Groupby {tag}'
            }
            #    - 권장(1): pandas category groupby
            # df_tag = _fast_groupby_with_category(df_tag, q, **dict_log_sub)
            
            #    - 권장(2): 초대형이면 NumPy 버전으로 더 밀어붙이기
            df_tag = _ultra_fast_groupby_numpy(df_tag, q, **dict_log_sub)


            # Location ‘-’ 고정 추가 + 컬럼 순서 정리
            # df_tag[COL_LOC] = '-'
            # df_tag = df_tag[[COL_SHIP_TO, COL_ITEM, COL_LOC, q, COL_ASN_FLAG, COL_SOUT_ASS_FLAG]]

            # # 메모리 절감: key 컬럼 category, qty=int8
            # df_tag[[COL_SHIP_TO, COL_ITEM, COL_LOC]] = df_tag[[COL_SHIP_TO, COL_ITEM, COL_LOC]].astype('category')
            # df_tag[q] = df_tag[q].astype('int8')

            frames[tag] = df_tag

        return frames  # {'GC': df, 'AP2': df, 'AP1': df, 'Local': df}

    frames = build_level(df_es)

    # ────────────────────────────────────────────────────────────────────
    # 3) Step-04 결과와 concat (ShipTo+Item+Loc 기준 중복 제거, qty=1)
    # ────────────────────────────────────────────────────────────────────
    def _concat_prev(df_new: pd.DataFrame, prev_key: str, qty_col: str) -> pd.DataFrame:
        prev = df_sout_dict.get(prev_key, pd.DataFrame())
        if prev.empty:
            return df_new

        gcols = [COL_SHIP_TO, COL_ITEM, COL_LOC]
        df = (pd.concat([prev, df_new], ignore_index=True)
                .astype({COL_SHIP_TO: str, COL_ITEM: str, COL_LOC: str})
                .drop_duplicates(subset=gcols)
                .assign(**{qty_col: 1}))
        # 재캐스팅(메모리)
        df[gcols]   = df[gcols].astype('category')
        df[qty_col] = df[qty_col].astype('int8')
        return df

    df_final_gc    = _concat_prev(frames['GC'],    STR_DF_OUT_SOUT_GC,    COL_SOUT_ASS_GC)
    df_final_ap2   = _concat_prev(frames['AP2'],   STR_DF_OUT_SOUT_AP2,   COL_SOUT_ASS_AP2)
    df_final_ap1   = _concat_prev(frames['AP1'],   STR_DF_OUT_SOUT_AP1,   COL_SOUT_ASS_AP1)
    df_final_local = _concat_prev(frames['Local'], STR_DF_OUT_SOUT_LOCAL, COL_SOUT_ASS_LOCAL)

    # ────────────────────────────────────────────────────────────────────
    # 4) dict 갱신 & 반환
    # ────────────────────────────────────────────────────────────────────
    df_sout_dict.update({
        STR_DF_OUT_SOUT_GC    : df_final_gc,
        STR_DF_OUT_SOUT_AP2   : df_final_ap2,
        STR_DF_OUT_SOUT_AP1   : df_final_ap1,
        STR_DF_OUT_SOUT_LOCAL : df_final_local
    })

    del (df_asn, df_rule, df_ship_dim, df_est, df_es, frames)
    gc.collect()
    return df_sout_dict

################################################################################################################
# Step 6:  S/Out Actual(Non eStore, Past12) 기반 S/Out Assortment 추가
#          (Step-02,03,04 를 재사용하여 df_sout_dict 확장)
################################################################################################################
@_decoration_
def step06_create_sellout_assortments_from_actual(
        df_in_Sell_Out_Actual_NoeStore_Past12: pd.DataFrame,  # Input-7
        df_sout_dict: dict[str, pd.DataFrame],                # 기존 Step-04+05 결과 (in/out)
        df_item: pd.DataFrame,                                # df_in_Item_Master
        df_master: pd.DataFrame,                              # df_in_Sell_Out_Simul_Master
        df_est: pd.DataFrame,                                 # df_in_Sales_Domain_Estore
        df_ship_dim: pd.DataFrame,                            # Step01 Ship-To LUT (df_fn_shipto_dim)
        df_rule: pd.DataFrame,                                # df_in_Forecast_Rule
        **kwargs
) -> dict[str, pd.DataFrame]:

    # ───────────────────────────────────────────────────────────────────
    # Guard: Input-7 이 비어 있으면 기존 df_sout_dict 그대로 반환
    # ───────────────────────────────────────────────────────────────────
    if df_in_Sell_Out_Actual_NoeStore_Past12 is None or df_in_Sell_Out_Actual_NoeStore_Past12.empty:
        return df_sout_dict

    # ───────────────────────────────────────────────────────────────────
    # Step 6-1) df_in_Sell_Out_Actual_NoeStore_Past12 → ASN 유사 DF 생성
    #          (Step-02 / 03 / 04 를 그대로 재사용하기 위한 형태)
    # ───────────────────────────────────────────────────────────────────
    use_cols = [COL_SHIP_TO, COL_ITEM, COL_LOC]

    # 혹시 컬럼이 누락되어 있으면 에러
    missing = [c for c in use_cols if c not in df_in_Sell_Out_Actual_NoeStore_Past12.columns]
    if missing:
        raise KeyError(f"[step06] df_in_Sell_Out_Actual_NoeStore_Past12 에 필수 컬럼이 없습니다: {missing}")

    # 6-1-1) ASN-like DF 구성
    df_asn_like = (
        df_in_Sell_Out_Actual_NoeStore_Past12[use_cols]
        .copy(deep=False)
    )

    # Sales Product ASN 컬럼 추가 (스펙: Null → pd.NA)
    # dtype 을 'string' 으로 두면 이후 Step02 에서 category 로 캐스팅해도 안전
    df_asn_like[COL_ASN_FLAG] = pd.Series(pd.NA, index=df_asn_like.index, dtype="string")

    # ───────────────────────────────────────────────────────────────────
    # Step 6-1-2) Item Master를 이용해 Product Group 주입
    #             (Step-02 대신 가볍게 PG만 merge)
    # ───────────────────────────────────────────────────────────────────
    # df_asn_like : [ShipTo, Item, Loc, Sales Product ASN(<NA>)] 상태

    # 1) 기본 ASN-like DF 스키마 + dtype 정리
    df_asn_step2 = (
        df_asn_like[[COL_SHIP_TO, COL_ITEM, COL_LOC, COL_ASN_FLAG]]
        .copy(deep=False)
        .astype({
            COL_SHIP_TO : 'category',
            COL_ITEM    : 'category',
            COL_LOC     : 'category',
            COL_ASN_FLAG: 'category',   # 'Y'/'N'/NA
        })
    )

    # 2) Item Master에서 Product Group 가져오기
    item_small = (
        df_item[[COL_ITEM, COL_PG]]
        .copy(deep=False)
        .astype({COL_PG: 'category'})
    )

    df_asn_step2 = df_asn_step2.merge(item_small, on=COL_ITEM, how='left')

    # 3) S/Out Assortment Flag는 스펙상 "Actual이 있는 Item은 모두 대상" 이므로 무조건 True
    df_asn_step2[COL_SOUT_ASS_FLAG] = True

    if df_asn_step2.empty:
        # 실제로 Assortment 대상이 없으면 기존 결과 그대로
        return df_sout_dict
    
    # ───────────────────────────────────────────────────────────────────
    # Step 6-1-3) Step-03 호출 : S/In Assortment 생성
    # ───────────────────────────────────────────────────────────────────
    dict_log_3 = {
        'p_step_no'  : 62,
        'p_step_desc': 'Step-06-1 : Build S/In assortments from S/Out Actual (No eStore Past12)'
    }
    df_sin_from_actual = step03_create_sellin_assortments(
        df_asn_step2,
        df_rule,
        df_ship_dim,
        **dict_log_3
    )

    # df_sin_from_actual: dict[str, DataFrame]
    #  - STR_DF_OUT_SIN_GC
    #  - STR_DF_OUT_SIN_AP2
    #  - STR_DF_OUT_SIN_AP1
    #  - STR_DF_OUT_SIN_LOCAL

    # ───────────────────────────────────────────────────────────────────
    # Step 6-1-4) Step-04 호출 : S/Out Assortment 생성
    # ───────────────────────────────────────────────────────────────────
    dict_log_4 = {
        'p_step_no'  : 63,
        'p_step_desc': 'Step-06-1 : Build S/Out assortments from S/In (Actual-based)'
    }
    df_sout_from_actual = step04_create_sellout_assortments(
        df_sin_from_actual,
        **dict_log_4
    )
    # df_sout_from_actual: dict[str, DataFrame]
    #  - STR_DF_OUT_SOUT_GC
    #  - STR_DF_OUT_SOUT_AP2
    #  - STR_DF_OUT_SOUT_AP1
    #  - STR_DF_OUT_SOUT_LOCAL

    # ───────────────────────────────────────────────────────────────────
    # Step 6-2) Step-06-1 결과(df_sout_from_actual) 와
    #           기존 Step-04+05 결과(df_sout_dict) concat
    #           (ShipTo+Item+Loc 기준 중복 제거, qty=1 유지)
    # ───────────────────────────────────────────────────────────────────

    # in-place 업데이트 (원래 dict 를 계속 재사용하는 패턴 유지)
    result_dict: dict[str, pd.DataFrame] = df_sout_dict

    qty_col_map = {
        STR_DF_OUT_SOUT_GC   : COL_SOUT_ASS_GC,
        STR_DF_OUT_SOUT_AP2  : COL_SOUT_ASS_AP2,
        STR_DF_OUT_SOUT_AP1  : COL_SOUT_ASS_AP1,
        STR_DF_OUT_SOUT_LOCAL: COL_SOUT_ASS_LOCAL,
    }
    key_cols = [COL_SHIP_TO, COL_ITEM, COL_LOC]

    for key, qty_col in qty_col_map.items():
        df_base = result_dict.get(key, pd.DataFrame())
        df_new  = df_sout_from_actual.get(key, pd.DataFrame())

        if df_new is None or df_new.empty:
            continue

        if df_base is None or df_base.empty:
            # 기존에 아무것도 없으면 그대로 사용
            result_dict[key] = df_new
            continue

        # concat 후, 중복 key 는 FLAG 는 max, qty 는 1 유지
        merged = ultra_fast_groupby_numpy_general(
            df=pd.concat([df_base, df_new], ignore_index=True),
            key_cols=key_cols,
            aggs={
                qty_col          : 'max',   # 이미 1 이지만 혹시 모를 중복 대비
                COL_ASN_FLAG     : 'max',   # 'Y'/'N' → 'Y' 우선
                COL_SOUT_ASS_FLAG: 'max',   # bool  → True 우선
            }
        )

        merged[qty_col] = np.int8(1)
        merged[key_cols] = merged[key_cols].astype('category')

        result_dict[key] = merged

    return result_dict

# ─────────────────────────────────────────────────────────────────────────────
# STEP-07  ▸  최종 Output Formatter
#   • Version 컬럼 삽입 & 순서 정렬
#   • 필요 없는 중간 플래그(col) 제거
#   • 빈 DF 는 헤더만 갖춘 빈 DF 로 생성
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_output_formatter(
        out_version : str,
        df_sin_dict : dict[str, pd.DataFrame],
        df_sout_dict: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """
    Sell-In · Sell-Out 8개 결과 DF 를 정렬/클린징 후 하나의 dict 로 반환.
    """
    # ── 0) 준비 – 컬럼 순서 템플릿 ──────────────────────────────────────
    SIN_ORDER = {
        STR_DF_OUT_SIN_GC    : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SIN_ASS_GC],
        STR_DF_OUT_SIN_AP2   : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SIN_ASS_AP2],
        STR_DF_OUT_SIN_AP1   : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SIN_ASS_AP1],
        STR_DF_OUT_SIN_LOCAL : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SIN_ASS_LOCAL],
    }
    SOUT_ORDER = {
        STR_DF_OUT_SOUT_GC    : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SOUT_ASS_GC],
        STR_DF_OUT_SOUT_AP2   : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SOUT_ASS_AP2],
        STR_DF_OUT_SOUT_AP1   : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SOUT_ASS_AP1],
        STR_DF_OUT_SOUT_LOCAL : [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOC, COL_SOUT_ASS_LOCAL],
    }    # 결과 dict (Sell-In+Sell-Out 통합)
    out_all: dict[str, pd.DataFrame] = {}

    # ── 1) Sell-In 정렬 ────────────────────────────────────────────────
    for df_name, col_order in SIN_ORDER.items():
        df = df_sin_dict.get(df_name, pd.DataFrame())

        # 없는 경우: 헤더만 생성
        if df.empty:
            df = pd.DataFrame(columns=col_order)

        # 필요 컬럼 강제 생성 → 순서 맞춘 뒤 나머지 컬럼 drop
        for col in col_order:
            if col not in df.columns:
                df[col] = np.nan

        df[COL_VERSION] = out_version
        df = df[col_order]          # hard-reorder (불필요 컬럼 자동 drop)
        out_all[df_name] = df

    # ── 2) Sell-Out 정렬 ───────────────────────────────────────────────
    for df_name, col_order in SOUT_ORDER.items():
        df = df_sout_dict.get(df_name, pd.DataFrame())

        if df.empty:
            df = pd.DataFrame(columns=col_order)

        for col in col_order:
            if col not in df.columns:
                df[col] = np.nan

        df[COL_VERSION] = out_version
        df = df[col_order]
        out_all[df_name] = df

    return out_all

################################################################################################################
# End Step Functions
################################################################################################################


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
            # set_input_output_folder(is_local, args)

            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------
            input_folder_name = 'PYForecastCreateSellInAndSellOutAssortment/input_ap07'
            output_folder_name = 'PYForecastCreateSellInAndSellOutAssortment_1210_Past12'
            
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
                "df_in_Sales_Domain_Dimension.csv"                  : STR_DF_IN_SALES_DOMAIN_DIM    ,
                "df_in_Sales_Domain_Estore.csv"                     : STR_DF_IN_ESTORE    ,
                "df_in_Sales_Product_ASN.csv"                       : STR_DF_IN_SALES_PRODUCT_ASN              ,
                "df_in_Forecast_Rule.csv"                           : STR_DF_IN_FORECAST_RULE                  ,
                "df_in_Item_Master.csv"                             : STR_DF_IN_ITEM_MASTER           ,
                "df_in_Sell_Out_Simul_Master.csv"                   : STR_DF_IN_SELL_OUT_SIMUL_MASTER     ,
                "df_in_Sell_Out_Actual_NoeStore_Past12.csv"         : STR_DF_IN_SOUT_ACT_NOESTORE_P12        
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
                        if STR_DF_IN_SOUT_ACT_NOESTORE_P12 == frame_name: 
                            df_in_Sell_Out_Actual_NoeStore_Past12 = df
                            break

        

        # fn_convert_type(df_in_Sales_Product_ASN, 'Sales Domain', str)
        # fn_convert_type(df_in_Sales_Product_ASN, 'Location', str)

        # fn_convert_type(df_in_Sales_Domain_Dimension, 'Sales Domain', str)
        # fn_convert_type(df_in_Sales_Domain_Estore, 'Sales Domain', str)
        # fn_convert_type(df_in_Forecast_Rule, 'Sales Domain', str)

        # df_in_Forecast_Rule[COL_FR_GC].fillna(0, inplace=True)
        # df_in_Forecast_Rule[COL_FR_AP2].fillna(0, inplace=True)
        # df_in_Forecast_Rule[COL_FR_AP1].fillna(0, inplace=True)
        # df_in_Forecast_Rule[COL_FR_AP0].fillna(0, inplace=True)
        # # df_in_Forecast_Rule[FORECAST_RULE_CUST].fillna(0, inplace=True)

        # fn_convert_type(df_in_Forecast_Rule, COL_FR_GC, 'int32')
        # fn_convert_type(df_in_Forecast_Rule, COL_FR_AP2, 'int32')
        # fn_convert_type(df_in_Forecast_Rule, COL_FR_AP1, 'int32')
        # fn_convert_type(df_in_Forecast_Rule, COL_FR_AP0, 'int32')
        # # fn_convert_type(df_in_Forecast_Rule, FORECAST_RULE_CUST, 'int32')

        # ───────────────────────────────────────────────────────────────
        # 1) COLUMN LISTS (필요 시 추가)
        # ───────────────────────────────────────────────────────────────
        STR_COLS_SHIP = [
            COL_SHIP_TO, COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6
        ]
        STR_COLS_ITEM = [COL_ITEM, COL_LOC, COL_PG]
        INT_COLS_FR   = [COL_FR_GC, COL_FR_AP2, COL_FR_AP1, COL_FR_AP0]
        BOOL_COLS     = [COL_SOUT_ASS_FLAG]

        # ───────────────────────────────────────────────────────────────
        # 2) APPLY TO EACH INPUT DF   (메모리 절감용)
        # ───────────────────────────────────────────────────────────────
        cast_cols(
            df_in_Sales_Domain_Dimension,
            cat_cols=STR_COLS_SHIP
        )
        cast_cols(
            df_in_Sales_Domain_Estore,
            cat_cols=[COL_SHIP_TO]
        )
        cast_cols(
            df_in_Forecast_Rule,
            cat_cols=[COL_VERSION,COL_PG, COL_SHIP_TO],
            int_cols=INT_COLS_FR
        )

        cast_cols(
            df_in_Item_Master,
            cat_cols=[COL_ITEM, COL_PG]
        )

        cast_cols(
            df_in_Sales_Product_ASN,
            cat_cols=[COL_VERSION,COL_SHIP_TO, COL_ITEM, COL_LOC,COL_ASN_FLAG],
            bool_cols=[],                    # 아직 없음
        )

        cast_cols(
            df_in_Sell_Out_Simul_Master,     # NEW
            cat_cols=[COL_VERSION, COL_SHIP_TO, COL_PG,COL_MASTER_STATUS],
            # bool_cols=[COL_MASTER_STATUS] if COL_MASTER_STATUS in df_in_Sell_Out_Simul_Master.columns else None
            bool_cols=[],
        )

        cast_cols(
            df_in_Sell_Out_Actual_NoeStore_Past12,     # NEW
            cat_cols=[COL_SHIP_TO, COL_ITEM,COL_LOC],
            bool_cols=[],
            int_cols=[COL_SOUT_ACT_P12]
        )

        
        fn_log_dataframe(df_in_Sales_Domain_Dimension,            STR_DF_IN_SALES_DOMAIN_DIM            )  
        fn_log_dataframe(df_in_Sales_Domain_Estore,               STR_DF_IN_SALES_DOMAIN_DIM            )  
        fn_log_dataframe(df_in_Sales_Product_ASN,                 STR_DF_IN_SALES_PRODUCT_ASN           )  
        fn_log_dataframe(df_in_Forecast_Rule,                     STR_DF_IN_FORECAST_RULE               ) 
        fn_log_dataframe(df_in_Item_Master,                       STR_DF_IN_ITEM_MASTER                 ) 
        fn_log_dataframe(df_in_Sell_Out_Simul_Master,             STR_DF_IN_SELL_OUT_SIMUL_MASTER       ) 
        fn_log_dataframe(df_in_Sell_Out_Actual_NoeStore_Past12,   STR_DF_IN_SOUT_ACT_NOESTORE_P12    )

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
        # Step 01 – Ship-To 차원 LUT 구축
        ################################################################################################################
        dict_log = {
            'p_step_no': 100,
            'p_step_desc': 'Step 01-6 – load Ship-To dimension LUT'
        }
        df_fn_shipto_dim = step01_load_shipto_dimension(df_in_Sales_Domain_Dimension)
        fn_log_dataframe(df_fn_shipto_dim, f'step01_df_fn_shipto_dim')

        ################################################################################################################
        # Step 2  : Sales Product ASN 전처리
        ################################################################################################################
        df_fn_sales_product_asn = step02_preprocess_asn(
            df_in_Sales_Product_ASN,
            df_in_Item_Master,
            df_in_Sell_Out_Simul_Master,
            df_in_Sales_Domain_Estore,
            df_fn_shipto_dim              # ← Step-01 LUT
        )
        fn_log_dataframe(df_fn_sales_product_asn, f'step02_{STR_DF_FN_SALES_PRODUCT_ASN}')
        
        ################################################################################################################
        # Step 3: ▸  Sell-In Assortments (GC / AP2 / AP1 / Local-AP0)
        ################################################################################################################
        dict_log = {
            'p_step_no': 30,
            'p_step_desc': 'Step 3  : Forecast Rule에 따른 Data 생성 (GC,AP2,AP1, Local-AP0 에 대해서 각각 진행) ' 
        }
        df_sin_dict = step03_create_sellin_assortments(
            df_fn_sales_product_asn,      # step-02 결과
            df_in_Forecast_Rule,          # 원본 Forecast-Rule
            df_fn_shipto_dim,             # step-01 LUT
            **dict_log
        )
        for df_in in df_sin_dict:
            fn_log_dataframe(df_sin_dict[df_in],f'step03_{df_in}')

        ###############################################################################
        #  Step 4 : Sell-Out Assortments from Sell-In (GC/AP2/AP1/Local)
        ###############################################################################
        dict_log = {
            'p_step_no' : 40,
            'p_step_desc': 'Step 4 : Sell-Out Assortments from Sell-In (GC/AP2/AP1/Local)'
        }
        df_sout_dict = step04_create_sellout_assortments(
            df_sin_dict,          # ← Step-03 결과 dict
            **dict_log            # ← 데코레이터용 Step 로그
        )
        #  로그 / CSV 덤프
        for name, df in df_sout_dict.items():
            fn_log_dataframe(df, f'step04_{name}')


        ################################################################################################################
        # Step 5:  Forecast Rule에 따른 E-store S/Out Assortment Column 및 Data 생성 (GC,AP2,AP1에 대해서 각각 진행)
        #          concat with Step-04
        ################################################################################################################
        dict_log = {
            'p_step_no' : 50,
            'p_step_desc': 'Step-05 : E-Store Sell-Out (concat with Step-04)'
        }
        df_sout_dict = step05_create_sellout_assortments_estore(
            df_fn_sales_product_asn,    # Step-02 결과
            df_in_Forecast_Rule,        # Forecast-Rule
            df_fn_shipto_dim,           # Ship-To LUT
            df_in_Sales_Domain_Estore,  # E-Store 목록
            df_sout_dict,               # Step-04 결과 → in/out
            **dict_log
        )
        # 로그 출력
        for name, df in df_sout_dict.items():
            fn_log_dataframe(df, f'step05_{name}')

        ################################################################################################################
        # Step 6:  S/Out Actual(Non eStore, Past12) 기반 S/Out Assortment 추가
        #          (Step-02,03,04 를 재사용하여 df_sout_dict 확장)
        ################################################################################################################
        # df_in_Sell_Out_Actual_NoeStore_Past12 = input_dataframes[STR_DF_IN_SOUT_ACT_NOESTORE_P12]

        dict_log = {
            'p_step_no' : 60,
            'p_step_desc': 'Step-06 : S/Out Actual(No eStore Past12) 기반 Assortment 추가'
        }
        df_sout_dict = step06_create_sellout_assortments_from_actual(
            df_in_Sell_Out_Actual_NoeStore_Past12,
            df_sout_dict,
            df_in_Item_Master,
            df_in_Sell_Out_Simul_Master,
            df_in_Sales_Domain_Estore,
            df_fn_shipto_dim,
            df_in_Forecast_Rule,
            **dict_log
        )
        # 로그 출력
        for name, df in df_sout_dict.items():
            fn_log_dataframe(df, f'step06_{name}')


        ################################################################################################################
        # Formatter:  Add Version Name 
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 900,
            'p_step_desc': 'Step-06  Final formatter'
        }
        df_final = fn_output_formatter(
            Version,          # ← 'CWV_DP' 등
            df_sin_dict,      # ← Step-03 결과
            df_sout_dict,     # Step-04 + Step-05 반영된 최종 Sell-Out dict
            **dict_log
        )

        # Logging helper – 필요시
        for name, df_out in df_final.items():
            fn_log_dataframe(df_out, name)
            # globals()[name] = df_out

        
        Output_SIn_Assortment_GC        = df_final[STR_DF_OUT_SIN_GC]
        Output_SIn_Assortment_AP2       = df_final[STR_DF_OUT_SIN_AP2]
        Output_SIn_Assortment_AP1       = df_final[STR_DF_OUT_SIN_AP1]
        Output_SIn_Assortment_Local     = df_final[STR_DF_OUT_SIN_LOCAL]
        Output_SOut_Assortment_GC       = df_final[STR_DF_OUT_SOUT_GC]
        Output_SOut_Assortment_AP2      = df_final[STR_DF_OUT_SOUT_AP2]
        Output_SOut_Assortment_AP1      = df_final[STR_DF_OUT_SOUT_AP1]
        Output_SOut_Assortment_Local    = df_final[STR_DF_OUT_SOUT_LOCAL]
        

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

            # # input log
            # input_path = f'{str_output_dir}/input'
            # os.makedirs(input_path,exist_ok=True)
                
            # df_in_Sales_Domain_Dimension.to_csv(input_path + "/" + STR_DF_IN_SALES_DOMAIN_DIM+".csv", encoding="UTF8", index=False)
            # df_in_Sales_Domain_Estore.to_csv(input_path + "/" + STR_DF_IN_ESTORE+".csv", encoding="UTF8", index=False)
            # df_in_Sales_Product_ASN.to_csv(input_path + "/" + STR_DF_IN_SALES_PRODUCT_ASN+".csv", encoding="UTF8", index=False)
            # df_in_Forecast_Rule.to_csv(input_path + "/" + STR_DF_IN_FORECAST_RULE+".csv", encoding="UTF8", index=False)
            # df_in_Item_Master.to_csv(input_path + "/" + STR_DF_IN_ITEM_MASTER+".csv", encoding="UTF8", index=False)

            # # output log
            # output_path = f'{str_output_dir}/output'
            # os.makedirs(output_path,exist_ok=True)
            # for output_file in df_final:
            #     df_final[output_file].to_csv(output_path + "/" + output_file+".csv", encoding="UTF8", index=False)

        # logger.info(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')
        logger.Finish()
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::') # 25.05.12 need warning Log by Logger Issue
        