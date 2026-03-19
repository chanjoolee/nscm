#############################################################
# PYNettingODSD.py (refactored without globals())
# - is_local(True)  : Local PC (CSV read / file copy)
# - is_local(False) : o9 server (df_in_* injected by plugin)
#############################################################

import traceback as tb
import os, shutil, glob
import time
import importlib
import datetime
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from NSCMCommon import NSCMCommon as common
from NSCMCommon import VDCommon as vdCommon

from daily_netting.pre_process.DA.ODS_DA_CONST import Netting_Plan_ODS_W as PLANODS
from daily_netting.pre_process.DA.ODS_DA_CONST import Netting_IF_BOD as BOD


# ======================================================
# Environment
# ======================================================
is_local = common.gfn_get_isLocal()  # True: 내 PC, False: o9 서버
str_instance = 'PYNettingODSD'

is_print = True
flag_csv = True

# ======================================================
# logger 설정
# ======================================================
logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
LOG_LEVEL = common.G_log_level



# ======================================================
# 데이터프레임 상수 (Input)
# ======================================================
DF_IN_DIVISION										  = 'df_in_Division'
DF_IN_ITEM                                            = 'df_in_Item'
DF_IN_LOCATION                                        = 'df_in_Location'
DF_IN_NETTING_DEMAND_TYPE_CONVERSION_RULE_UI          = 'df_in_Netting_Demand_Type_Conversion_Rule_UI'
DF_IN_NETTING_EXTRACT_RULE_UI                         = 'df_in_Netting_Extract_Rule_UI'
DF_IN_NETTING_FCST_RULE_UI                            = 'df_in_Netting_FCST_Rule_UI'
DF_IN_NETTING_IF_ASSOCIATION                          = 'df_in_Netting_IF_Association'
DF_IN_NETTING_IF_BOD                                  = 'df_in_Netting_IF_BOD'
DF_IN_NETTING_IF_DA_PREFERENCE_RANK                   = 'df_in_Netting_IF_DA_Preference_Rank'
DF_IN_NETTING_IF_CUSTOMER_RANK                        = 'df_in_Netting_IF_Customer_Rank'
DF_IN_NETTING_IF_DF_ITEM_SITE                         = 'df_in_Netting_IF_DF_Item_Site'
DF_IN_NETTING_IF_DF_ALLOC_MAP                         = 'df_in_Netting_IF_DF_Alloc_Map'
DF_IN_NETTING_IF_CODE_MAP                             = 'df_in_Netting_IF_Code_Map'
DF_IN_NETTING_IF_PRODUCT_RANK                         = 'df_in_Netting_IF_Product_Rank'
DF_IN_NETTING_IF_SALES_BOM_MAP                        = 'df_in_Netting_IF_Sales_BOM_Map'
DF_IN_NETTING_SITE                                    = 'df_in_Netting_Site'
DF_IN_NETTING_IF_VD_PREFERENCE_RANK                   = 'df_in_Netting_IF_VD_Preference_Rank'
DF_IN_NETTING_ITEM                                    = 'df_in_Netting_Item'
DF_IN_NETTING_ITEM_ATTRIBUTE                          = 'df_in_Netting_Item_Attribute'
DF_IN_NETTING_ITEM_ATTB2                              = 'df_in_Netting_Item_Attb2'
DF_IN_NETTING_ITEM_LEVEL                              = 'df_in_Netting_Item_Level'
DF_IN_NETTING_PLAN_ODS_D                              = 'df_in_Netting_Plan_ODS_D'
DF_IN_NETTING_PLAN_PARAMETER_UI                       = 'df_in_Netting_Plan_Parameter_UI'
DF_IN_NETTING_PRE_ALLOCATION_RULE_UI                  = 'df_in_Netting_Pre_Allocation_Rule_UI'
DF_IN_NETTING_PRIORITY_RANK_ORDER_UI                  = 'df_in_Netting_Priority_Rank_Order_UI'
DF_IN_NETTING_PRIORITY_RANK_UI                        = 'df_in_Netting_Priority_Rank_UI'
DF_IN_NETTING_PRIORITY_RANK_DETAIL_UI                 = 'df_in_Netting_Priority_Rank_Detail_UI'
DF_IN_NETTING_PRIORITY_RULE_ASN_UI                    = 'df_in_Netting_Priority_Rule_ASN_UI'
DF_IN_NETTING_PRIORITY_RULE_MASTER_UI                 = 'df_in_Netting_Priority_Rule_Master_UI'
DF_IN_NETTING_PRIORITY_SWAP_UI                        = 'df_in_Netting_Priority_Swap_UI'
DF_IN_NETTING_SALES                                   = 'df_in_Netting_Sales'
DF_IN_NETTING_SALES_LEVEL                             = 'df_in_Netting_Sales_Level'
DF_IN_SALES_DOMAIN                                    = 'df_in_Sales_Domain'
DF_IN_SELECT_ITEM_SECTION_ATTRIBUTE                   = 'df_in_SELECT_Item_Section_Attribute'
DF_IN_TIME                                            = 'df_in_Time'
DF_IN_NETTING_IF_FCST_PLAN_D                          = 'df_in_Netting_IF_FCST_Plan_D'
DF_IN_SIG_RESULT_GI_N                                 = 'df_in_SIG_Result_GI_N'
DF_IN_NETTING_MEASURE_COPY_RULE_UI                    = 'df_in_Netting_Measure_Copy_Rule_UI'
DF_IN_NETTING_IF_RTF_NETTING                          = 'df_in_Netting_IF_RTF_Netting'
DF_IN_ITEM_MX                                         = 'df_in_Item_mx'
DF_IN_SIG_PLAN_MASTER                                 = 'df_in_SIG_Plan_Master'

# ======================================================
# 추가된 데이터프레임 상수
# ======================================================
DF_IN_NETTING_LP_PLAN_BATCH                           = 'df_in_Netting_LP_Plan_Batch'
DF_IN_ITEM_BAS                                        = 'df_in_Item_BAS'
DF_IN_NETTING_IF_RB_MASTER                            = 'df_in_Netting_IF_RB_Master'
DF_IN_NETTING_IF_CUSTOMER_MODEL_MAP                   = 'df_in_Netting_IF_Customer_Model_Map'
DF_IN_NETTING_IF_MX_PREFERENCE_RANK                   = 'df_in_Netting_IF_MX_Preference_Rank'
DF_IN_NETTING_IF_SALES_ORDER_NC                       = 'df_in_Netting_IF_Sales_Order_NC'                   # ExceptDmdODSD

start_time = time.time()

# 로컬에서만 쓰는 변수 기본값(플러그인 주입 변수와 충돌 없음)
local_using_df = []


def fn_log_dataframe(df_p_source: pd.DataFrame, str_p_source_name: str, int_p_row_num: int = 20) -> None:
    """
    Dataframe 로그 출력 조건 지정 함수
    """
    is_output = False
    if str_p_source_name.startswith('out_'):
        is_output = True

    if is_print:
        logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1, p_row_num=int_p_row_num)
        if is_local and flag_csv:
            df_p_source.to_csv(str_output_dir + "/" + str_p_source_name + ".csv", encoding="UTF8", index=False)
    else:
        if is_output:
            logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1, p_row_num=20)
            if is_local:
                df_p_source.to_csv(str_output_dir + "/" + str_p_source_name + ".csv", encoding="UTF8", index=False)


def _decoration_(func):
    def wrapper(*args, **kwargs):
        tm_start = time.time()
        result = func(*args)  # kwargs는 decorator에서만 사용(기존 방식 유지)
        tm_end = time.time()

        logger.Note(p_note=f'[{func.__name__}] Total time is {tm_end - tm_start:.5f} sec.', p_log_level=LOG_LEVEL.debug())

        _step_no = kwargs.get('p_step_no')
        _step_desc = kwargs.get('p_step_desc')
        vdCommon.gfn_pyLog_detail(_step_desc)

        _df_name = kwargs.get('p_df_name')
        _warn_desc = kwargs.get('p_warn_desc')
        _exception_flag = kwargs.get('p_exception_flag')

        if _step_no is not None and _step_desc is not None:
            logger.Step(p_step_no=_step_no, p_step_desc=_step_desc)

        if _warn_desc is not None:
            if type(result) == pd.DataFrame and result.empty:
                if _exception_flag is not None:
                    if _exception_flag == 0:
                        logger.Note(p_note=_warn_desc, p_log_level=LOG_LEVEL.warning())
                    elif _exception_flag == 1:
                        raise Exception(_warn_desc)

        if _df_name is not None:
            fn_log_dataframe(result, _df_name)

        return result
    return wrapper


# ======================================================
# dtype 표준화 (너 코드 그대로 유지)
# ======================================================
DATE_COLS = {
    BOD.EFFENDDATE,
    BOD.EFFSTARTDATE,
    PLANODS.EFFSTARTDATE,
    'Netting Close Date D'
}

CATEGORY_PREFIXES = (
    "Version.",
    "Netting Division.",
    "Netting Sales.[Sales ID]",
    "Sales Domain",
    "Netting LP Plan Batch.[Sales Domain",
    "Item",
    "Netting Item",
    "Location",
    "Netting Code Map",
    "Netting DA Preference Rank Value",
    "Netting Plan Week",
    "Time.",
    "To Time."
)

FLOAT_PREFIXES = (
    'Netting Code Map Number',
)

INT_PREFIXES = (
    # 'Netting Measure Copy Rule Time',
    # 'Netting BOD Priority',
    # 'Netting BOD Transit Time',
    # 'Netting Sales BOM Map Priority',
    # 'Netting Plan Horizon D',
    # 'Netting Plan Parameter Alloc Retention Level',
    # 'Netting Priority Rank Order',
    # 'Netting Priority Rank Digit',
    # 'Netting Priority Rank Default Value',
    # 'Netting Priority Rank Detail Order',
    # 'Netting Priority Rule Master Start Bucket',
    # 'Netting Priority Rule Seq.[Rule Sequence]',
    # 'Netting Priority Rule Master End Bucket',
    # 'Netting FCST Plan',
    # 'Netting Plan Parameter RTF Retention Level',
    # 'Netting BOD End Bucket',
    # 'Netting BOD Lead Time',
    # 'Netting Measure Copy Rule Time Fence',
    # 'Netting Extract Rule Time Fence',
    # 'Netting BOD Shipping Lead Time',
    # 'Netting GC Extract Rule',
    # 'Netting AP2 Extract Rule',
    # 'Netting AP1 Extract Rule',
    # 'Netting Account Extract Rule',

)

def _to_datetime_safe(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_categorical_dtype(s):
        s = s.astype("string")
    else:
        s = s.astype("string")
    s = s.replace({"": pd.NA, "NaT": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_datetime(s, errors="coerce")

def _should_category(s: pd.Series, max_unique_ratio: float = 0.7, min_rows: int = 1000) -> bool:
    n = len(s)
    if n == 0:
        return False
    if n < min_rows:
        return True
    u = s.nunique(dropna=False)
    return (u / n) <= max_unique_ratio

def fn_prepare_input_types(dict_dfs: dict) -> None:
    if not dict_dfs:
        return

    date_cols_global = set(DATE_COLS)

    for df_name, df in dict_dfs.items():
        if df is None or df.empty:
            continue

        cols = df.columns

        prefix_int_cols = [c for c in cols if c.startswith(INT_PREFIXES) and c not in date_cols_global]
        for c in prefix_int_cols:
            df[c] = df[c].fillna(0)
            df[c] = df[c].astype("int32")
            
        prefix_float_cols = [c for c in cols if c.startswith(FLOAT_PREFIXES) and c not in date_cols_global and c not in prefix_int_cols]
        for c in prefix_float_cols:
            if pd.api.types.is_categorical_dtype(df[c]):
                continue
            df[c] = df[c].astype("float32")

        date_cols = list(date_cols_global.intersection(cols))
        for c in date_cols:
            df[c] = _to_datetime_safe(df[c])

        prefix_cols = [c for c in cols if c.startswith(CATEGORY_PREFIXES) and c not in date_cols_global and c not in prefix_float_cols and c not in prefix_int_cols]
        for c in prefix_cols:
            if pd.api.types.is_categorical_dtype(df[c]):
                continue
            s = df[c].astype("string")
            df[c] = s.astype("category") if _should_category(s) else s

        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        obj_cols = [c for c in obj_cols if c not in date_cols_global and c not in prefix_cols]
        for c in obj_cols:
            s = df[c].astype("string")
            df[c] = s.astype("category") if _should_category(s) else s

        int_cols = df.select_dtypes(include=["int64", "int32", "Int64", "Int32", "int"]).columns
        for c in int_cols:
            if pd.api.types.is_integer_dtype(df[c]) and str(df[c].dtype).startswith("Int"):
                df[c] = df[c].astype("Int32")
            else:
                if df[c].isna().any():
                    df[c] = df[c].astype("Int32")
                else:
                    df[c] = df[c].astype("int32")

        float_cols = df.select_dtypes(include=["float64", "float32"]).columns
        for c in float_cols:
            df[c] = df[c].astype("float32")


# ======================================================
# Local CSV helpers
# ======================================================
def _read_csv_with_fallback(filepath: str) -> pd.DataFrame:
    encodings = ['utf-8-sig', 'utf-8', 'cp949']
    file_stem = os.path.basename(filepath).split(".")[0]

    for enc in encodings:
        try:
            if file_stem == DF_IN_NETTING_LP_PLAN_BATCH:
                return pd.read_csv(filepath, encoding=enc, keep_default_na=False, na_filter=False)
            else:
                return pd.read_csv(filepath, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Unable to read file {filepath} with tried encodings.")

LP_SCENARIO = "Netting LP Plan Batch.[Scenario]"
LP_TENANT = "Netting LP Plan Batch.[Tenant]"

def _sanitize_lp_plan_batch_o9(df: pd.DataFrame) -> None:
    """
    o9 주입 경로에서 'NA'가 결측으로 해석되는 케이스를 복원.
    + 타입도 로컬과 유사하게 맞춤.
    """
    if df is None or df.empty:
        return

    # 1) Scenario/Tenant: 결측을 'NA'로 복원 (원래 결측이 없다는 전제)
    for c in (LP_SCENARIO,LP_TENANT):
        if c in df.columns:
            s = df[c].astype("string")          # category/object 상관없이 string으로 통일
            s = s.fillna("NA")                  # o9가 결측으로 바꿔버린 'NA' 복원
            df[c] = s.astype("category")

# ======================================================
# Inbound 처리 (globals() 없이!)
# ======================================================
@_decoration_
def fn_process_in_df_mst():

    if is_local:
        # 로컬: Output 폴더 정리
        for file in os.scandir(str_output_dir):
            try:
                # os.remove(file.path)
                shutil.rmtree(file.path)
            except Exception:
                pass

        file_pattern = f"{str_input_dir}/*.csv"
        csv_files = glob.glob(file_pattern)

        for file in csv_files:
            file_name = Path(file).stem  # 확장자 제외

            if len(local_using_df) > 0 and file_name not in local_using_df:
                continue

            # "df_in_*.csv"만 취급 (필요한 것만 자동 적재)
            if not file_name.startswith("df_in_"):
                continue

            df = _read_csv_with_fallback(file)
            input_dataframes[file_name] = df

    else:
        # o9 서버: df_in_*는 플러그인이 주입하므로 "선언/초기화 금지"
        # 여기서는 '참조만' 한다 (기존 방식 그대로).
        input_dataframes[DF_IN_DIVISION]                                  = df_in_Division
        input_dataframes[DF_IN_ITEM]                                      = df_in_Item
        input_dataframes[DF_IN_LOCATION]                                  = df_in_Location
        input_dataframes[DF_IN_NETTING_DEMAND_TYPE_CONVERSION_RULE_UI]    = df_in_Netting_Demand_Type_Conversion_Rule_UI
        input_dataframes[DF_IN_NETTING_EXTRACT_RULE_UI]                   = df_in_Netting_Extract_Rule_UI
        input_dataframes[DF_IN_NETTING_FCST_RULE_UI]                      = df_in_Netting_FCST_Rule_UI
        input_dataframes[DF_IN_NETTING_IF_ASSOCIATION]                    = df_in_Netting_IF_Association
        input_dataframes[DF_IN_NETTING_IF_BOD]                            = df_in_Netting_IF_BOD
        input_dataframes[DF_IN_NETTING_IF_DA_PREFERENCE_RANK]             = df_in_Netting_IF_DA_Preference_Rank
        input_dataframes[DF_IN_NETTING_IF_CUSTOMER_RANK]                  = df_in_Netting_IF_Customer_Rank
        input_dataframes[DF_IN_NETTING_IF_DF_ITEM_SITE]                   = df_in_Netting_IF_DF_Item_Site
        input_dataframes[DF_IN_NETTING_IF_DF_ALLOC_MAP]                   = df_in_Netting_IF_DF_Alloc_Map
        input_dataframes[DF_IN_NETTING_IF_CODE_MAP]                       = df_in_Netting_IF_Code_Map
        input_dataframes[DF_IN_NETTING_IF_PRODUCT_RANK]                   = df_in_Netting_IF_Product_Rank
        input_dataframes[DF_IN_NETTING_IF_SALES_BOM_MAP]                  = df_in_Netting_IF_Sales_BOM_Map
        input_dataframes[DF_IN_NETTING_SITE]                              = df_in_Netting_Site
        input_dataframes[DF_IN_NETTING_IF_VD_PREFERENCE_RANK]             = df_in_Netting_IF_VD_Preference_Rank
        input_dataframes[DF_IN_NETTING_ITEM]                              = df_in_Netting_Item
        input_dataframes[DF_IN_NETTING_ITEM_ATTRIBUTE]                    = df_in_Netting_Item_Attribute
        input_dataframes[DF_IN_NETTING_ITEM_ATTB2]                        = df_in_Netting_Item_Attb2
        input_dataframes[DF_IN_NETTING_ITEM_LEVEL]                        = df_in_Netting_Item_Level
        input_dataframes[DF_IN_NETTING_PLAN_ODS_D]                        = df_in_Netting_Plan_ODS_D
        input_dataframes[DF_IN_NETTING_PLAN_PARAMETER_UI]                 = df_in_Netting_Plan_Parameter_UI
        input_dataframes[DF_IN_NETTING_PRE_ALLOCATION_RULE_UI]            = df_in_Netting_Pre_Allocation_Rule_UI
        input_dataframes[DF_IN_NETTING_PRIORITY_RANK_ORDER_UI]            = df_in_Netting_Priority_Rank_Order_UI
        input_dataframes[DF_IN_NETTING_PRIORITY_RANK_UI]                  = df_in_Netting_Priority_Rank_UI
        input_dataframes[DF_IN_NETTING_PRIORITY_RANK_DETAIL_UI]           = df_in_Netting_Priority_Rank_Detail_UI
        input_dataframes[DF_IN_NETTING_PRIORITY_RULE_ASN_UI]              = df_in_Netting_Priority_Rule_ASN_UI
        input_dataframes[DF_IN_NETTING_PRIORITY_RULE_MASTER_UI]           = df_in_Netting_Priority_Rule_Master_UI
        input_dataframes[DF_IN_NETTING_PRIORITY_SWAP_UI]                  = df_in_Netting_Priority_Swap_UI
        input_dataframes[DF_IN_NETTING_SALES]                             = df_in_Netting_Sales
        input_dataframes[DF_IN_NETTING_SALES_LEVEL]                       = df_in_Netting_Sales_Level
        input_dataframes[DF_IN_SALES_DOMAIN]                              = df_in_Sales_Domain
        input_dataframes[DF_IN_SELECT_ITEM_SECTION_ATTRIBUTE]             = df_in_SELECT_Item_Section_Attribute
        input_dataframes[DF_IN_TIME]                                      = df_in_Time
        input_dataframes[DF_IN_NETTING_IF_FCST_PLAN_D]                    = df_in_Netting_IF_FCST_Plan_D
        input_dataframes[DF_IN_SIG_RESULT_GI_N]                           = df_in_SIG_Result_GI_N
        input_dataframes[DF_IN_NETTING_MEASURE_COPY_RULE_UI]              = df_in_Netting_Measure_Copy_Rule_UI
        input_dataframes[DF_IN_NETTING_IF_RTF_NETTING]                    = df_in_Netting_IF_RTF_Netting
        input_dataframes[DF_IN_ITEM_MX]                                   = df_in_Item_mx
        input_dataframes[DF_IN_SIG_PLAN_MASTER]                           = df_in_SIG_Plan_Master

        # 추가된 부분
        input_dataframes[DF_IN_NETTING_LP_PLAN_BATCH]                     = df_in_Netting_LP_Plan_Batch
        input_dataframes[DF_IN_ITEM_BAS]                                  = df_in_Item_BAS
        input_dataframes[DF_IN_NETTING_IF_RB_MASTER]                      = df_in_Netting_IF_RB_Master
        input_dataframes[DF_IN_NETTING_IF_CUSTOMER_MODEL_MAP]             = df_in_Netting_IF_Customer_Model_Map
        input_dataframes[DF_IN_NETTING_IF_MX_PREFERENCE_RANK]             = df_in_Netting_IF_MX_Preference_Rank
        input_dataframes[DF_IN_NETTING_IF_SALES_ORDER_NC]                 = df_in_Netting_IF_Sales_Order_NC                 # ExceptDmdODSD

    # o9에서만 NA 복원 처리
    if (not is_local) and (DF_IN_NETTING_LP_PLAN_BATCH in input_dataframes):
        _sanitize_lp_plan_batch_o9(input_dataframes[DF_IN_NETTING_LP_PLAN_BATCH])

    fn_prepare_input_types(input_dataframes)


# ======================================================
# 실행 요약(df_result)용 실행계획
# ======================================================
@dataclass(frozen=True)
class ExecSpec:
    step_no: int
    step_desc: str
    output_dataframe_nm: str  # 알아보기 쉽게 번호를 먹인 output_dataframe
    output_var_key: str       # 실제 output_dataframe
    module_import_path: str
    python_file: str
    function_nm: str
    needs_update_inputtable: bool = False


def _build_exec_plan(lv_DivisionName: str) -> list[ExecSpec]:
    return [
        ExecSpec(6,  "do_Create_ODS_06", "6_df_out_Netting_Plan_Param_ODS_D",   "df_out_Netting_Plan_Param_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}PlanParamODSD",
                 f"PYNetting{lv_DivisionName}PlanParamODSD.PY", "create_Plan_Param_ODS"),
        ExecSpec(11, "do_Create_ODS_11", "11_df_out_Netting_Item_ODS_D",        "df_out_Netting_Item_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}ItemODSD",
                 f"PYNetting{lv_DivisionName}ItemODSD.PY", "create_Item_ODS", True),
        ExecSpec(12, "do_Create_ODS_12", "12_df_out_Netting_Sales_ODS_D",       "df_out_Netting_Sales_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}SalesODSD",
                 f"PYNetting{lv_DivisionName}SalesODSD.PY", "create_Netting_Sales_ODS", True),
        ExecSpec(13, "do_Create_ODS_13", "13_df_out_Netting_Site_ODS_D",        "df_out_Netting_Site_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}SiteODSD",
                 f"PYNetting{lv_DivisionName}SiteODSD.PY", "create_Netting_Site_ODS", True),
        ExecSpec(14, "do_Create_ODS_14", "14_df_out_Netting_Week_Bucket_ODS_D", "df_out_Netting_Week_Bucket_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}WeekBucketODSD",
                 f"PYNetting{lv_DivisionName}WeekBucketODSD.PY", "create_Week_Bucket_ODS", True),
        ExecSpec(15, "do_Create_ODS_15", "15_df_out_Netting_Month_Bucket_ODS_D","df_out_Netting_Month_Bucket_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}MonthBucketODSD",
                 f"PYNetting{lv_DivisionName}MonthBucketODSD.PY", "create_Month_Bucket_ODS", True),
        ExecSpec(16, "do_Create_ODS_16", "16_df_out_Netting_BOD_ODS_D",         "df_out_Netting_BOD_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}BODODSD",
                 f"PYNetting{lv_DivisionName}BODODSD.PY", "create_Netting_BOD_ODS"),
        ExecSpec(17, "do_Create_ODS_17", "17_df_out_Netting_FCST_Rule_ODS_D",   "df_out_Netting_FCST_Rule_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}FCSTRuleODSD",
                 f"PYNetting{lv_DivisionName}FCSTRuleODSD.PY", "create_FCST_Rule_ODS"),
        ExecSpec(18, "do_Create_ODS_18", "18_df_out_Netting_Extract_Rule_ODS_D","df_out_Netting_Extract_Rule_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}ExtractRuleODSD",
                 f"PYNetting{lv_DivisionName}ExtractRuleODSD.PY", "create_Extract_Rule_ODS"),
        ExecSpec(19, "do_Create_ODS_19", "19_df_out_Netting_Measure_Copy_Rule_ODS_D","df_out_Netting_Measure_Copy_Rule_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}MeasureCopyRuleODSD",
                 f"PYNetting{lv_DivisionName}MeasureCopyRuleODSD.PY", "create_Netting_Measure_Copy_Rule_ODS"),
        ExecSpec(20, "do_Create_ODS_20", "20_df_out_Netting_Demand_Type_Conversion_Rule_ODS_D","df_out_Netting_Demand_Type_Conversion_Rule_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}DemandTypeConversionRuleODSD",
                 f"PYNetting{lv_DivisionName}DemandTypeConversionRuleODSD.PY", "create_Netting_Demand_Type_Conversion_Rule_ODS"),
        ExecSpec(21, "do_Create_ODS_21", "21_df_out_Netting_Measure_Type_Conversion_Rule_ODS_D","df_out_Netting_Measure_Type_Conversion_Rule_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}MeasureTypeConversionRuleODSD",
                 f"PYNetting{lv_DivisionName}MeasureTypeConversionRuleODSD.PY", "create_Netting_Measure_Type_Conversion_Rule_ODS"),
        ExecSpec(22, "do_Create_ODS_22", "22_df_out_Netting_Retention_Rule_ODS_D","df_out_Netting_Retention_Rule_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}RetentionRuleODSD",
                 f"PYNetting{lv_DivisionName}RetentionRuleODSD.PY", "create_Netting_Retention_Rule_ODS"),
        ExecSpec(23, "do_Create_ODS_23", "23_df_out_Netting_Retention_Type_ODS_D","df_out_Netting_Retention_Type_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}RetentionTypeODSD",
                 f"PYNetting{lv_DivisionName}RetentionTypeODSD.PY", "create_Netting_Retention_Type_ODS"),
        ExecSpec(24, "do_Create_ODS_24", "24_df_out_Netting_Pre_Allocation_Rule_ODS_D","df_out_Netting_Pre_Allocation_Rule_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}PreAllocationRuleODSD",
                 f"PYNetting{lv_DivisionName}PreAllocationRuleODSD.PY", "create_Netting_Pre_Allocation_Rule_ODS"),
        ExecSpec(25, "do_Create_ODS_25", "25_df_out_Netting_Priority_Rule_Master_ODS_D","df_out_Netting_Priority_Rule_Master_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}PriorityRuleMasterODSD",
                 f"PYNetting{lv_DivisionName}PriorityRuleMasterODSD.PY", "create_Priority_Rule_Master_ODS"),
        ExecSpec(26, "do_Create_ODS_26", "26_df_out_Netting_Priority_Rank_ODS_D","df_out_Netting_Priority_Rank_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}PriorityRankODSD",
                 f"PYNetting{lv_DivisionName}PriorityRankODSD.PY", "create_Priority_Rank_ODS"),
        ExecSpec(27, "do_Create_ODS_27", "27_df_out_Netting_Priority_Rank_Detail_ODS_D","df_out_Netting_Priority_Rank_Detail_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}PriorityRankDetailODSD",
                 f"PYNetting{lv_DivisionName}PriorityRankDetailODSD.PY", "create_Priority_Rank_Detail_ODS"),
        ExecSpec(28, "do_Create_ODS_28", "28_df_out_Netting_Priority_Swap_ODS_D","df_out_Netting_Priority_Swap_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}PrioritySwapODSD",
                 f"PYNetting{lv_DivisionName}PrioritySwapODSD.PY", "create_Netting_Priority_Swap_ODS"),
        ExecSpec(29, "do_Create_ODS_29", "29_df_out_Netting_Priority_Rule_ASN_ODS_D","df_out_Netting_Priority_Rule_ASN_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}PriorityRuleASNODSD",
                 f"PYNetting{lv_DivisionName}PriorityRuleASNODSD.PY", "create_Priority_Rule_ASN_ODS"),
        ExecSpec(30, "do_Create_ODS_30", "30_df_out_Netting_Preference_Rank_ODS_D","df_out_Netting_Preference_Rank_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}PreferenceRankODSD",
                 f"PYNetting{lv_DivisionName}PreferenceRankODSD.PY", "create_Netting_PreferenceRank_ODS"),
        ExecSpec(31, "do_Create_ODS_31", "31_df_out_Netting_Customer_Rank_ODS_D","df_out_Netting_Customer_Rank_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}CustomerRankODSD",
                 f"PYNetting{lv_DivisionName}CustomerRankODSD.PY", "create_Customer_Rank_ODS"),
        ExecSpec(32, "do_Create_ODS_32", "32_df_out_Netting_Product_Rank_ODS_D","df_out_Netting_Product_Rank_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}ProductRankODSD",
                 f"PYNetting{lv_DivisionName}ProductRankODSD.PY", "create_Product_Rank_ODS"),
        ExecSpec(33, "do_Create_ODS_33", "33_df_out_Netting_FCST_Plan_ODS_D",    "df_out_Netting_FCST_Plan_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}FCSTPlanODSD",
                 f"PYNetting{lv_DivisionName}FCSTPlanODSD.PY", "create_Forecst_Plan_ODS"),
        ExecSpec(34, "do_Create_ODS_34", "34_df_out_Netting_Except_Demand_ODS_D",    "df_out_Netting_Except_Demand_ODS_D",
                 f"daily_netting.pre_process.{lv_DivisionName}.PYNetting{lv_DivisionName}ExceptDmdODSD",
                 f"PYNetting{lv_DivisionName}ExceptDmdODSD.PY", "create_Netting_Excpt_Dmd_ODS"),
    ]


def _run_exec_plan(exec_plan: list[ExecSpec], dic_InputTable: dict):
    dict_outputs: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    executed_src_files: set[Path] = set()

    for spec in exec_plan:
        mod = importlib.import_module(spec.module_import_path)
        func = getattr(mod, spec.function_nm)

        dict_log = {
            "p_step_no": spec.step_no,
            "p_step_desc": spec.step_desc,
            "p_df_name": spec.output_dataframe_nm,
        }

        df_out = func(**dic_InputTable, **dict_log)
        if df_out is None:
            df_out = pd.DataFrame()

        dict_outputs[spec.output_var_key] = df_out

        # 실행된 파일 path (로컬일 때 copy용)
        try:
            src = Path(mod.__file__)
            if src.exists():
                executed_src_files.add(src)
        except Exception:
            pass

        rows.append({
            "output_dataframe_nm": spec.output_dataframe_nm,
            "python_file": spec.python_file,
            "function_nm": spec.function_nm,
            "row_cnt": int(len(df_out)),
        })

        # 11~15는 다음 step input으로도 사용
        if spec.needs_update_inputtable:
            if spec.step_no == 11:
                dic_InputTable["df_in_Netting_Item_ODS_D"] = df_out
            elif spec.step_no == 12:
                dic_InputTable["df_in_Netting_Sales_ODS_D"] = df_out
            elif spec.step_no == 13:
                dic_InputTable["df_in_Netting_Site_ODS_D"] = df_out
            elif spec.step_no == 14:
                dic_InputTable["df_in_Netting_Week_Bucket_ODS_D"] = df_out
            elif spec.step_no == 15:
                dic_InputTable["df_in_Netting_Month_Bucket_ODS_D"] = df_out

    df_result = pd.DataFrame(rows).reset_index(drop=True)
    return dict_outputs, df_result, executed_src_files


def _copy_executed_modules(executed_src_files: set[Path], str_output_dir: str) -> None:
    logger.Note(p_note='실행된모듈들: ')
    copied = set()
    for src in sorted(executed_src_files, key=lambda p: p.name):
        if src.name in copied:
            continue
        try:
            shutil.copyfile(src, Path(str_output_dir) / src.name)
            logger.Note(p_note=f'{src.name}')
            copied.add(src.name)
        except Exception as e:
            logger.Note(p_note=f"[WARN] copy failed: {src} / {e}")

    
def _copy_inbound_dir_to_output(str_input_dir: str, str_output_dir: str) -> None:
    """
    is_local 인 경우:
    - str_input_dir 폴더 전체를
    - str_output_dir/inbound 로 폴더명 변경하여 복사
    """
    src = Path(str_input_dir)
    dst = Path(str_output_dir) / "inbound"

    if not src.exists():
        logger.Note(p_note=f"[WARN] inbound copy skipped: input dir not found -> {src}")
        return

    # 기존 inbound 폴더가 있으면 제거 후 다시 복사
    if dst.exists():
        try:
            shutil.rmtree(dst)
        except Exception as e:
            logger.Note(p_note=f"[WARN] failed to remove existing inbound dir: {dst} / {e}")

    try:
        # python 3.8+ (dirs_exist_ok 지원)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        logger.Note(p_note=f"[OK] inbound copied: {src} -> {dst}")
    except TypeError:
        # (아주 구버전 대비) dirs_exist_ok 미지원인 경우
        shutil.copytree(src, dst)
        logger.Note(p_note=f"[OK] inbound copied: {src} -> {dst}")
    except Exception as e:
        logger.Note(p_note=f"[WARN] inbound copy failed: {src} -> {dst} / {e}")


####################################
############ Start Main  ###########
####################################
if __name__ == '__main__':
    logger.info(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    input_dataframes = {}
    executed_src_files = set()
    df_result = pd.DataFrame()

    try:
        # ----------------------------------------------------------
        # (중요) Version/Division은 o9 플러그인이 주입할 수 있으므로
        #        전역/상단에서 절대 초기화(=할당)하지 않는다.
        #        로컬일 때만 명시적으로 세팅한다.
        # ----------------------------------------------------------
        if is_local:
            # --------------------
            # 운영
            # --------------------
            # Version = '202606_DP_MX_AS_WE'
            # Division = 'D001'
            # str_input_dir = r'C:\Netting\o9Data\PYNettingODSD\Prod\DPNS\202606_DP_MX_AS_WE\download_20260204-124753'

            
            # Version = '202606_DP_MX_EU_WE'
            # Division = 'D001'
            # str_input_dir = r'C:\Netting\o9Data\PYNettingODSD\Prod\DPNS\202606_DP_MX_EU_WE\download_20260204-145427'
            
            # Version = '202606_DP_VD_EU_WE'
            # Division = 'D002'
            # str_input_dir = r'C:\Netting\o9Data\PYNettingODSD\Prod\DPNS\202606_DP_VD_EU_WE\download_20260204-132100'

            # Version = '202606_DP_DA_AS_WE'
            # Division = 'D003'
            # str_input_dir = r'C:\Netting\o9Data\PYNettingODSD\Prod\DPNS\202606_DP_DA_AS_WE\download_20260204-132837'

            # Version = '202606_DP_DA_EU_FR'
            # Division = 'D003'
            # str_input_dir = r'C:\Netting\o9Data\PYNettingODSD\Prod\DPNS\202606_DP_DA_EU_FR\download_20260206-131742'

            # Version = '202606_DP_MX_AS_FR'
            # Division = 'D001'
            # str_input_dir = r'C:\Netting\o9Data\PYNettingODSD\Prod\DPNS\202606_DP_MX_AS_FR\download_20260206-161923'

            Version = '202606_DP_DA_US_FR'
            Division = 'D003'
            str_input_dir = r'C:\Netting\o9Data\PYNettingODSD\Prod\DPNS\202606_DP_DA_US_FR\download_20260209-101823'
            

            # --------------------
            # QA
            # --------------------
            # # QA 202606_DP_MX_AS_MO
            # Version = '202606_DP_MX_AS_MO'
            # Division = 'D001'
            # str_input_dir = r'C:\Netting\o9Data\PYNettingODSD\QA\DPNS\202606_DP_MX_AS_MO\download_20260204-154645'

            # Version = '202606_DP_VD_AS_MO'
            # Division = 'D002'
            # str_input_dir = r'C:\Netting\o9Data\PYNettingODSD\QA\DPNS\202606_DP_VD_AS_MO\download_20260204-160114'

            map_division = {'D001': 'MX', 'D002': 'VD', 'D003': 'DA'}

            div_nm = map_division.get(Division, '')
            output_folder_name = str_instance

            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y%m%d")
            str_output_dir = f'Output/{output_folder_name}_{Version}_{formatted_time}'
            # str_output_dir = f'Output/{output_folder_name}_{div_nm}_{formatted_time}'


            os.makedirs(str_input_dir, exist_ok=True)
            os.makedirs(str_output_dir, exist_ok=True)

            # 필요 시 로컬에서 사용할 df만 제한
            local_using_df = [
                # 'df_in_Netting_Plan_Parameter_UI',
                # 'df_in_Netting_Item',
            ]

        # ----------------------------------------------------------
        # Parameter 체크 (o9에서는 주입된 Version/Division을 그대로 참조)
        # ----------------------------------------------------------
        try:
            lv_ParamVersion = Version
        except NameError:
            lv_ParamVersion = ""

        try:
            lv_ParamDivision = Division
        except NameError:
            lv_ParamDivision = ""

        # ----------------------------------------------------------
        # DivisionName
        # ----------------------------------------------------------
        if lv_ParamDivision == 'D001':
            lv_DivisionName = 'MX'
        elif lv_ParamDivision == 'D002':
            lv_DivisionName = 'VD'
        elif lv_ParamDivision == 'D003':
            lv_DivisionName = 'DA'
        else:
            # o9에서 Division이 꼭 온다는 전제면 여기서 raise 해도 됨
            lv_DivisionName = 'VD'

        # ----------------------------------------------------------
        # vdLog 초기화
        # ----------------------------------------------------------
        log_path = os.path.dirname(__file__) if is_local else ""
        vdCommon.gfn_pyLog_start(lv_ParamVersion, str_instance, logger, is_local, log_path)

        # ----------------------------------------------------------
        # df_input 적재
        # ----------------------------------------------------------
        logger.Note(p_note='df_input 체크 시작', p_log_level=LOG_LEVEL.debug())
        fn_process_in_df_mst(p_step_no=0, p_step_desc="fn_process_in_df_mst", p_df_name=None)

        # 입력 DF 간단 출력
        for in_df in input_dataframes:
            try:
                logger.PrintDF(input_dataframes[in_df], in_df, p_row_num=20)
            except Exception:
                logger.Note(p_note=f"{in_df} 를 표현하는데 문제가 있습니다.")
                logger.Note(p_note=input_dataframes[in_df].dtypes)

        # ----------------------------------------------------------
        # Input Table dictionary 생성
        # ----------------------------------------------------------
        dic_InputTable = {
            'lv_ParamDivision': lv_ParamDivision,
            'lv_ParamVersion':  lv_ParamVersion,
            'lv_VersionName':   lv_ParamVersion,
        }

        # input_dataframes를 dic_InputTable에 그대로 넣는다
        # (기존 모듈들이 df_in_XXX 키로 참조하므로 그대로 유지)
        for k, v in input_dataframes.items():
            dic_InputTable[k] = v

        # ----------------------------------------------------------
        # 실행계획 / 실행
        # ----------------------------------------------------------
        exec_plan = _build_exec_plan(lv_DivisionName)
        dict_outputs, df_result, executed_src_files = _run_exec_plan(exec_plan, dic_InputTable)

        # o9 의 후속에서 인식할 수 있어야 한다.
        df_out_Netting_Plan_Param_ODS_D                  	= dict_outputs.get('df_out_Netting_Plan_Param_ODS_D                  '.strip(),pd.DataFrame())
        df_out_Netting_Item_ODS_D                           = dict_outputs.get('df_out_Netting_Item_ODS_D                        '.strip(),pd.DataFrame())
        df_out_Netting_Sales_ODS_D                          = dict_outputs.get('df_out_Netting_Sales_ODS_D                       '.strip(),pd.DataFrame())
        df_out_Netting_Site_ODS_D                           = dict_outputs.get('df_out_Netting_Site_ODS_D                        '.strip(),pd.DataFrame())
        df_out_Netting_Week_Bucket_ODS_D                    = dict_outputs.get('df_out_Netting_Week_Bucket_ODS_D                 '.strip(),pd.DataFrame())
        df_out_Netting_Month_Bucket_ODS_D                   = dict_outputs.get('df_out_Netting_Month_Bucket_ODS_D                '.strip(),pd.DataFrame())
        df_out_Netting_BOD_ODS_D                            = dict_outputs.get('df_out_Netting_BOD_ODS_D                         '.strip(),pd.DataFrame())
        df_out_Netting_FCST_Rule_ODS_D                      = dict_outputs.get('df_out_Netting_FCST_Rule_ODS_D                   '.strip(),pd.DataFrame())
        df_out_Netting_Extract_Rule_ODS_D                   = dict_outputs.get('df_out_Netting_Extract_Rule_ODS_D                '.strip(),pd.DataFrame())
        df_out_Netting_Measure_Copy_Rule_ODS_D              = dict_outputs.get('df_out_Netting_Measure_Copy_Rule_ODS_D           '.strip(),pd.DataFrame())
        df_out_Netting_Demand_Type_Conversion_Rule_ODS_D    = dict_outputs.get('df_out_Netting_Demand_Type_Conversion_Rule_ODS_D '.strip(),pd.DataFrame())
        df_out_Netting_Measure_Type_Conversion_Rule_ODS_D   = dict_outputs.get('df_out_Netting_Measure_Type_Conversion_Rule_ODS_D'.strip(),pd.DataFrame())
        df_out_Netting_Retention_Rule_ODS_D                 = dict_outputs.get('df_out_Netting_Retention_Rule_ODS_D              '.strip(),pd.DataFrame())
        df_out_Netting_Retention_Type_ODS_D                 = dict_outputs.get('df_out_Netting_Retention_Type_ODS_D              '.strip(),pd.DataFrame())
        df_out_Netting_Pre_Allocation_Rule_ODS_D            = dict_outputs.get('df_out_Netting_Pre_Allocation_Rule_ODS_D         '.strip(),pd.DataFrame())
        df_out_Netting_Priority_Rule_Master_ODS_D           = dict_outputs.get('df_out_Netting_Priority_Rule_Master_ODS_D        '.strip(),pd.DataFrame())
        df_out_Netting_Priority_Rank_ODS_D                  = dict_outputs.get('df_out_Netting_Priority_Rank_ODS_D               '.strip(),pd.DataFrame())
        df_out_Netting_Priority_Rank_Detail_ODS_D           = dict_outputs.get('df_out_Netting_Priority_Rank_Detail_ODS_D        '.strip(),pd.DataFrame())
        df_out_Netting_Priority_Swap_ODS_D                  = dict_outputs.get('df_out_Netting_Priority_Swap_ODS_D               '.strip(),pd.DataFrame())
        df_out_Netting_Priority_Rule_ASN_ODS_D              = dict_outputs.get('df_out_Netting_Priority_Rule_ASN_ODS_D           '.strip(),pd.DataFrame())
        df_out_Netting_Preference_Rank_ODS_D                = dict_outputs.get('df_out_Netting_Preference_Rank_ODS_D             '.strip(),pd.DataFrame())
        df_out_Netting_Customer_Rank_ODS_D                  = dict_outputs.get('df_out_Netting_Customer_Rank_ODS_D               '.strip(),pd.DataFrame())
        df_out_Netting_Product_Rank_ODS_D                   = dict_outputs.get('df_out_Netting_Product_Rank_ODS_D                '.strip(),pd.DataFrame())
        df_out_Netting_FCST_Plan_ODS_D                      = dict_outputs.get('df_out_Netting_FCST_Plan_ODS_D                   '.strip(),pd.DataFrame())
        df_out_Netting_Except_Demand_ODS_D                  = dict_outputs.get('df_out_Netting_Except_Demand_ODS_D               '.strip(),pd.DataFrame())

        

        end_time = time.time()
        logger.Note("NETTING_ODSD_EXEC_TIME: {:.6f}".format(end_time - start_time))

    except Exception as e:
        logger.error(tb.format_exc())
        raise e

    finally:
        # ----------------------------------------------------------
        # 로컬에서만 파일 복사 수행 (o9에서는 파일시스템 접근 금지/불필요)
        # ----------------------------------------------------------
        if is_local:
            try:
                log_file_name = common.G_PROGRAM_NAME.replace('py', 'log')
                log_file_name = f'log/{log_file_name}'
                shutil.copyfile(log_file_name, os.path.join(str_output_dir, os.path.basename(log_file_name)))
            except Exception as e:
                logger.Note(p_note=f"[WARN] log copy failed: {e}")

            try:
                _copy_executed_modules(executed_src_files, str_output_dir)
            except Exception as e:
                logger.Note(p_note=f"[WARN] module copy failed: {e}")

            try:
                base_dir = Path(f"{os.getcwd()}") / "NettingTest"
                program_path = base_dir / f"{str_instance}.py"
                if program_path.exists():
                    shutil.copyfile(program_path, os.path.join(str_output_dir, program_path.name))
            except Exception as e:
                logger.Note(p_note=f"[WARN] program copy failed: {e}")

            # ... (기존 log copy / module copy / program copy)
            # ✅ inbound 폴더 복사 추가
            _copy_inbound_dir_to_output(str_input_dir, str_output_dir)

        # ----------------------------------------------------------
        # df_result 요약 로그 (요청사항)
        # ----------------------------------------------------------
        fn_log_dataframe(df_result, "df_result", int_p_row_num=200)

        logger.Finish()
        logger.Note('Netting ODSD END', 30)
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')
