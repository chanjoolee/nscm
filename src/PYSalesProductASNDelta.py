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
from typing import Collection, Tuple,Union,Dict
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
str_instance = 'PYSalesProductASNDelta'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"

is_print = True
flag_csv = True
flag_exception = True

########################################################################################################################
# 컬럼상수
########################################################################################################################
# ── core dims/time ───────────────────────────────────────────────────────────────────────────────────────────
COL_VERSION                     = 'Version.[Version Name]'
COL_SHIP_TO                     = 'Sales Domain.[Ship To]'
COL_STD1                        = 'Sales Domain.[Sales Std1]'
COL_STD2                        = 'Sales Domain.[Sales Std2]'
COL_STD3                        = 'Sales Domain.[Sales Std3]'
COL_STD4                        = 'Sales Domain.[Sales Std4]'
COL_STD5                        = 'Sales Domain.[Sales Std5]'
COL_STD6                        = 'Sales Domain.[Sales Std6]'
COL_ITEM                        = 'Item.[Item]'
COL_ITEM_GBM                    = 'Item.[Item GBM]'
COL_ITEM_STD1                   = 'Item.[Item Std1]'
COL_ITEM_STD3                   = 'Item.[Item Std3]'
COL_PG                          = 'Item.[Product Group]'
COL_LOCATION                    = 'Location.[Location]'
COL_WEEK                        = 'Time.[Week]'
COL_PWEEK                       = 'Time.[Partial Week]'
COL_MONTH                       = 'Time.[Month]'
COL_VIRTUAL_BO_ID               = 'Virtual BO ID.[Virtual BO ID]'
COL_BO_ID                       = 'BO ID.[BO ID]'
# ── ASN ──────────────────────────────────────────────────────────────────────────────────────────────────────
COL_SALES_PRODUCT_ASN           = 'Sales Product ASN'
COL_SALES_PRODUCT_ASN_DELTA     = 'Sales Product ASN Delta'

# ── Assortment (S/In) ───────────────────────────────────────────────────────────────────────────────────────
COL_SIN_ASSORT_GC               = 'S/In FCST(GI) Assortment_GC'
COL_SIN_ASSORT_AP2              = 'S/In FCST(GI) Assortment_AP2'
COL_SIN_ASSORT_AP1              = 'S/In FCST(GI) Assortment_AP1'
COL_SIN_ASSORT_LOCAL            = 'S/In FCST(GI) Assortment_Local'

# ── Assortment (S/Out) ──────────────────────────────────────────────────────────────────────────────────────
COL_SOUT_ASSORT_GC              = 'S/Out FCST Assortment_GC'
COL_SOUT_ASSORT_AP2             = 'S/Out FCST Assortment_AP2'
COL_SOUT_ASSORT_AP1             = 'S/Out FCST Assortment_AP1'
COL_SOUT_ASSORT_LOCAL           = 'S/Out FCST Assortment_Local'

# ── DSR (S/In) ───────────────────────────────────────────────────────────────────────────────────────────────
COL_SIN_DSR_GC                  = 'S/In FCST(GI) DSR_GC'
COL_SIN_DSR_AP2                 = 'S/In FCST(GI) DSR_AP2'
COL_SIN_DSR_AP1                 = 'S/In FCST(GI) DSR_AP1'
COL_SIN_DSR_LOCAL               = 'S/In FCST(GI) DSR_Local'

# ── DSR (S/Out) ──────────────────────────────────────────────────────────────────────────────────────────────
COL_SOUT_DSR_GC                 = 'S/Out FCST DSR_GC'
COL_SOUT_DSR_AP2                = 'S/Out FCST DSR_AP2'
COL_SOUT_DSR_AP1                = 'S/Out FCST DSR_AP1'
COL_SOUT_DSR_LOCAL              = 'S/Out FCST DSR_Local'

# ── FCST series ──────────────────────────────────────────────────────────────────────────────────────────────
COL_SIN_FCST_AP1                = 'S/In FCST(GI)_AP1'
COL_SIN_FCST_AP2                = 'S/In FCST(GI)_AP2'
COL_SIN_FCST_GC                 = 'S/In FCST(GI)_GC'
COL_SIN_FCST_LOCAL              = 'S/In FCST(GI)_Local'
COL_SOUT_FCST_AP1               = 'S/Out FCST_AP1'
COL_SOUT_FCST_AP2               = 'S/Out FCST_AP2'
COL_SOUT_FCST_GC                = 'S/Out FCST_GC'
COL_SOUT_FCST_LOCAL             = 'S/Out FCST_Local'
# 스펙 내 표기가 혼재되어 둘 다 정의(필요 시 하나만 사용)
COL_SIN_FCST_NEW_MODEL          = 'S/In FCST(GI) New Model'
COL_SIN_FCST_NEW_MODE           = 'S/In FCST(GI) New Mode'

# ── BO / Flooring ────────────────────────────────────────────────────────────────────────────────────────────
COL_BO_FCST                     = 'BO FCST'
COL_FLOORING_FCST               = 'Flooring_FCST'

# ── Price / FX ───────────────────────────────────────────────────────────────────────────────────────────────
COL_EST_PRICE_MODIFY_LOCAL      = 'Estimated Price Modify_Local'
COL_EST_PRICE_LOCAL             = 'Estimated Price_Local'
COL_AP_PRICE_USD                = 'Action Plan Price_USD'
COL_EXCHANGE_RATE_LOCAL         = 'Exchange Rate_Local'

# ── Forecast Rule ────────────────────────────────────────────────────────────────────────────────────────────
COL_FRULE_GC_FCST               = 'FORECAST_RULE GC FCST'
COL_FRULE_AP2_FCST              = 'FORECAST_RULE AP2 FCST'
COL_FRULE_AP1_FCST              = 'FORECAST_RULE AP1 FCST'
COL_FRULE_AP0_FCST              = 'FORECAST_RULE AP0 FCST'
COL_FRULE_ISVALID               = 'FORECAST_RULE ISVALID'
COL_SOUT_MASTER_STATUS          = 'S/Out Master Status'

# ── Split Ratio (S/In) ───────────────────────────────────────────────────────────────────────────────────────
COL_SIN_SPLIT_AP1               = 'S/In FCST(GI) Split Ratio_AP1'
COL_SIN_SPLIT_AP2               = 'S/In FCST(GI) Split Ratio_AP2'
COL_SIN_SPLIT_GC                = 'S/In FCST(GI) Split Ratio_GC'
COL_SIN_SPLIT_LOCAL             = 'S/In FCST(GI) Split Ratio_Local'

# ── Split Ratio (S/Out) ──────────────────────────────────────────────────────────────────────────────────────
COL_SOUT_SPLIT_AP1              = 'S/Out FCST Split Ratio_AP1'
COL_SOUT_SPLIT_AP2              = 'S/Out FCST Split Ratio_AP2'
COL_SOUT_SPLIT_GC               = 'S/Out FCST Split Ratio_GC'
COL_SOUT_SPLIT_LOCAL            = 'S/Out FCST Split Ratio_Local'

# ── Stretch Plan ─────────────────────────────────────────────────────────────────────────────────────────────
COL_SIN_STRETCH_ASSORT          = 'S/In Stretch Plan Assortment'
COL_SOUT_STRETCH_ASSORT         = 'S/Out Stretch Plan Assortment'
COL_SIN_STRETCH_SPLIT           = 'S/In Stretch Plan Split Ratio'
COL_SOUT_STRETCH_SPLIT          = 'S/Out Stretch Plan Split Ratio'

# ── eStore GI Ratio (User / Issue / Final / BestFit / User Item / Issue Item) ────────────────────────────────
COL_SIN_USER_GI_LT              = 'S/In User GI Ratio(Long Tail)'
COL_SIN_USER_GI_W0              = 'S/In User GI Ratio(W+0)'
COL_SIN_USER_GI_W1              = 'S/In User GI Ratio(W+1)'
COL_SIN_USER_GI_W2              = 'S/In User GI Ratio(W+2)'
COL_SIN_USER_GI_W3              = 'S/In User GI Ratio(W+3)'
COL_SIN_USER_GI_W4              = 'S/In User GI Ratio(W+4)'
COL_SIN_USER_GI_W5              = 'S/In User GI Ratio(W+5)'
COL_SIN_USER_GI_W6              = 'S/In User GI Ratio(W+6)'
COL_SIN_USER_GI_W7              = 'S/In User GI Ratio(W+7)'

COL_SIN_ISSUE_GI_LT             = 'S/In Issue GI Ratio(Long Tail)'
COL_SIN_ISSUE_GI_W0             = 'S/In Issue GI Ratio(W+0)'
COL_SIN_ISSUE_GI_W1             = 'S/In Issue GI Ratio(W+1)'
COL_SIN_ISSUE_GI_W2             = 'S/In Issue GI Ratio(W+2)'
COL_SIN_ISSUE_GI_W3             = 'S/In Issue GI Ratio(W+3)'
COL_SIN_ISSUE_GI_W4             = 'S/In Issue GI Ratio(W+4)'
COL_SIN_ISSUE_GI_W5             = 'S/In Issue GI Ratio(W+5)'
COL_SIN_ISSUE_GI_W6             = 'S/In Issue GI Ratio(W+6)'
COL_SIN_ISSUE_GI_W7             = 'S/In Issue GI Ratio(W+7)'

COL_SIN_FINAL_GI_LT             = 'S/In Final GI Ratio(Long Tail)'
COL_SIN_FINAL_GI_W0             = 'S/In Final GI Ratio(W+0)'
COL_SIN_FINAL_GI_W1             = 'S/In Final GI Ratio(W+1)'
COL_SIN_FINAL_GI_W2             = 'S/In Final GI Ratio(W+2)'
COL_SIN_FINAL_GI_W3             = 'S/In Final GI Ratio(W+3)'
COL_SIN_FINAL_GI_W4             = 'S/In Final GI Ratio(W+4)'
COL_SIN_FINAL_GI_W5             = 'S/In Final GI Ratio(W+5)'
COL_SIN_FINAL_GI_W6             = 'S/In Final GI Ratio(W+6)'
COL_SIN_FINAL_GI_W7             = 'S/In Final GI Ratio(W+7)'

COL_SIN_BESTFIT_GI_LT           = 'S/In BestFit GI Ratio(Long Tail)'
COL_SIN_BESTFIT_GI_W0           = 'S/In BestFit GI Ratio(W+0)'
COL_SIN_BESTFIT_GI_W1           = 'S/In BestFit GI Ratio(W+1)'
COL_SIN_BESTFIT_GI_W2           = 'S/In BestFit GI Ratio(W+2)'
COL_SIN_BESTFIT_GI_W3           = 'S/In BestFit GI Ratio(W+3)'
COL_SIN_BESTFIT_GI_W4           = 'S/In BestFit GI Ratio(W+4)'
COL_SIN_BESTFIT_GI_W5           = 'S/In BestFit GI Ratio(W+5)'
COL_SIN_BESTFIT_GI_W6           = 'S/In BestFit GI Ratio(W+6)'
COL_SIN_BESTFIT_GI_W7           = 'S/In BestFit GI Ratio(W+7)'

COL_SIN_USER_ITEM_GI_LT         = 'S/In User Item GI Ratio(Long Tail)'
COL_SIN_USER_ITEM_GI_W0         = 'S/In User Item GI Ratio(W+0)'
COL_SIN_USER_ITEM_GI_W1         = 'S/In User Item GI Ratio(W+1)'
COL_SIN_USER_ITEM_GI_W2         = 'S/In User Item GI Ratio(W+2)'
COL_SIN_USER_ITEM_GI_W3         = 'S/In User Item GI Ratio(W+3)'
COL_SIN_USER_ITEM_GI_W4         = 'S/In User Item GI Ratio(W+4)'
COL_SIN_USER_ITEM_GI_W5         = 'S/In User Item GI Ratio(W+5)'
COL_SIN_USER_ITEM_GI_W6         = 'S/In User Item GI Ratio(W+6)'
COL_SIN_USER_ITEM_GI_W7         = 'S/In User Item GI Ratio(W+7)'

COL_SIN_ISSUE_ITEM_GI_LT        = 'S/In Issue Item GI Ratio(Long Tail)'
COL_SIN_ISSUE_ITEM_GI_W0        = 'S/In Issue Item GI Ratio(W+0)'
COL_SIN_ISSUE_ITEM_GI_W1        = 'S/In Issue Item GI Ratio(W+1)'
COL_SIN_ISSUE_ITEM_GI_W2        = 'S/In Issue Item GI Ratio(W+2)'
COL_SIN_ISSUE_ITEM_GI_W3        = 'S/In Issue Item GI Ratio(W+3)'
COL_SIN_ISSUE_ITEM_GI_W4        = 'S/In Issue Item GI Ratio(W+4)'
COL_SIN_ISSUE_ITEM_GI_W5        = 'S/In Issue Item GI Ratio(W+5)'
COL_SIN_ISSUE_ITEM_GI_W6        = 'S/In Issue Item GI Ratio(W+6)'
COL_SIN_ISSUE_ITEM_GI_W7        = 'S/In Issue Item GI Ratio(W+7)'

# ── Modify GI Ratio ──────────────────────────────────────────────────────────────────────────────────────────
COL_SIN_USER_MOD_GI_LT          = 'S/In User Modify GI Ratio(Long Tail)'
COL_SIN_USER_MOD_GI_W0          = 'S/In User Modify GI Ratio(W+0)'
COL_SIN_USER_MOD_GI_W1          = 'S/In User Modify GI Ratio(W+1)'
COL_SIN_USER_MOD_GI_W2          = 'S/In User Modify GI Ratio(W+2)'
COL_SIN_USER_MOD_GI_W3          = 'S/In User Modify GI Ratio(W+3)'
COL_SIN_USER_MOD_GI_W4          = 'S/In User Modify GI Ratio(W+4)'
COL_SIN_USER_MOD_GI_W5          = 'S/In User Modify GI Ratio(W+5)'
COL_SIN_USER_MOD_GI_W6          = 'S/In User Modify GI Ratio(W+6)'
COL_SIN_USER_MOD_GI_W7          = 'S/In User Modify GI Ratio(W+7)'

COL_SIN_ISSUE_MOD_GI_LT         = 'S/In Issue Modify GI Ratio(Long Tail)'
COL_SIN_ISSUE_MOD_GI_W0         = 'S/In Issue Modify GI Ratio(W+0)'
COL_SIN_ISSUE_MOD_GI_W1         = 'S/In Issue Modify GI Ratio(W+1)'
COL_SIN_ISSUE_MOD_GI_W2         = 'S/In Issue Modify GI Ratio(W+2)'
COL_SIN_ISSUE_MOD_GI_W3         = 'S/In Issue Modify GI Ratio(W+3)'
COL_SIN_ISSUE_MOD_GI_W4         = 'S/In Issue Modify GI Ratio(W+4)'
COL_SIN_ISSUE_MOD_GI_W5         = 'S/In Issue Modify GI Ratio(W+5)'
COL_SIN_ISSUE_MOD_GI_W6         = 'S/In Issue Modify GI Ratio(W+6)'
COL_SIN_ISSUE_MOD_GI_W7         = 'S/In Issue Modify GI Ratio(W+7)'


########################################################################################################################
# 데이터프레임 상수 (Input)
########################################################################################################################
DF_IN_SALES_DOMAIN_DIMENSION                 = 'df_in_Sales_Domain_Dimension'         # (Input 1)
DF_IN_TIME                                   = 'df_in_Time'                           # (Input 2)
DF_IN_ITEM_MASTER_LED_SIGNAGE                = 'df_in_Item_Master_LED_SIGNAGE'        # (Input 3)
DF_IN_ITEM_MASTER                            = 'df_in_Item_Master'                    # (Input 3)
DF_IN_SALES_DOMAIN_ESTORE                    = 'df_in_Sales_Domain_Estore'            # (Input 4)
DF_IN_FORECAST_RULE                          = 'df_in_Forecast_Rule'                  # (Input 5)
DF_IN_SALES_PRODUCT_ASN_DELTA                = 'df_in_Sales_Product_ASN_Delta'        # (Input 6)
DF_IN_SALES_PRODUCT_ASN                      = 'df_in_Sales_Product_ASN'              # (Input 7)
DF_IN_SIN_ASSORTMENT                         = 'df_in_SIn_Assortment'                 # (Input 8)
DF_IN_SOUT_ASSORTMENT                        = 'df_in_SOut_Assortment'                # (Input 9)
DF_IN_ESTIMATED_PRICE                        = 'df_in_Estimated_Price'                # (Input 10)
DF_IN_ACTION_PLAN_PRICE                      = 'df_in_Action_Plan_Price'              # (Input 11)
DF_IN_EXCHANGE_RATE_LOCAL                    = 'df_in_Exchange_Rate_Local'            # (Input 12)
DF_IN_SIN_SPLIT_AP1                          = 'df_in_Sell_In_FCST_GI_Split_Ratio_AP1'    # (Input 16)
DF_IN_SIN_SPLIT_AP2                          = 'df_in_Sell_In_FCST_GI_Split_Ratio_AP2'    # (Input 17)
DF_IN_SIN_SPLIT_GC                           = 'df_in_Sell_In_FCST_GI_Split_Ratio_GC'     # (Input 18)
DF_IN_SIN_SPLIT_LOCAL                        = 'df_in_Sell_In_FCST_GI_Split_Ratio_Local'  # (Input 18-dup)

DF_IN_SOUT_SPLIT_AP1                         = 'df_in_Sell_Out_FCST_Split_Ratio_AP1'      # (Input 19)
DF_IN_SOUT_SPLIT_AP2                         = 'df_in_Sell_Out_FCST_Split_Ratio_AP2'      # (Input 20)
DF_IN_SOUT_SPLIT_GC                          = 'df_in_Sell_Out_FCST_Split_Ratio_GC'       # (Input 21)
DF_IN_SOUT_SPLIT_LOCAL                       = 'df_in_Sell_Out_FCST_Split_Ratio_Local'    # (Input 21-dup)

DF_IN_SIN_STRETCH_SPLIT                      = 'df_in_Sell_In_Stretch_Plan_Split_Ratio'   # (Input 22)
DF_IN_SOUT_STRETCH_SPLIT                     = 'df_in_Sell_Out_Stretch_Plan_Split_Ratio'  # (Input 23)
DF_IN_SIN_STRETCH_ASSORT                     = 'df_in_Sell_In_Stretch_Plan_Assortment'    # (Input 29)
DF_IN_SOUT_STRETCH_ASSORT                    = 'df_in_Sell_Out_Stretch_Plan_Assortment'   # (Input 29-dup)

DF_IN_SOUT_SIMUL_MASTER                      = 'df_in_Sell_Out_Simul_Master'              # (Input 30)

DF_IN_SIN_USER_GI_RATIO                      = 'df_in_Sell_In_User_GI_Ratio'              # (Input 31)
DF_IN_SIN_ISSUE_GI_RATIO                     = 'df_in_Sell_In_Issue_GI_Ratio'             # (Input 32)
DF_IN_SIN_FINAL_GI_RATIO                     = 'df_in_Sell_In_Final_GI_Ratio'             # (Input 33)
DF_IN_SIN_BESTFIT_GI_RATIO                   = 'df_in_Sell_In_BestFit_GI_Ratio'           # (Input 34)
DF_IN_SIN_USER_ITEM_GI_RATIO                 = 'df_in_Sell_In_User_Item_GI_Ratio'         # (Input 35)
DF_IN_SIN_ISSUE_ITEM_GI_RATIO                = 'df_in_Sell_In_Issue_Item_GI_Ratio'        # (Input 36)

########################################################################################################################
# 데이터프레임 상수 (Output)
########################################################################################################################
# (Output 1-x) Sales Product ASN / Delta
OUT_SALES_PRODUCT_ASN                        = 'Output_Sales_Product_ASN'               # (Output 1-6)
OUT_SALES_PRODUCT_ASN_DELTA                  = 'Output_Sales_Product_ASN_Delta'         # (Output 1-7)

# (Output 2-x) Assortment
OUT_SIN_ASSORTMENT_GC                        = 'Output_SIn_Assortment_GC'               # (Output 2-1)
OUT_SIN_ASSORTMENT_AP2                       = 'Output_SIn_Assortment_AP2'              # (Output 2-2)
OUT_SIN_ASSORTMENT_AP1                       = 'Output_SIn_Assortment_AP1'              # (Output 2-3)
OUT_SIN_ASSORTMENT_LOCAL                     = 'Output_SIn_Assortment_Local'            # (Output 2-4)
OUT_SOUT_ASSORTMENT_GC                       = 'Output_SOut_Assortment_GC'              # (Output 2-5)
OUT_SOUT_ASSORTMENT_AP2                      = 'Output_SOut_Assortment_AP2'             # (Output 2-6)
OUT_SOUT_ASSORTMENT_AP1                      = 'Output_SOut_Assortment_AP1'             # (Output 2-7)
OUT_SOUT_ASSORTMENT_LOCAL                    = 'Output_SOut_Assortment_Local'           # (Output 2-8)

# (Output 3-x) DSR
OUT_SIN_DSR_GC                               = 'Output_SIn_DSR_GC'                      # (Output 3-1)
OUT_SIN_DSR_AP2                              = 'Output_SIn_DSR_AP2'                     # (Output 3-2)
OUT_SIN_DSR_AP1                              = 'Output_SIn_DSR_AP1'                     # (Output 3-3)
OUT_SIN_DSR_LOCAL                            = 'Output_SIn_DSR_Local'                   # (Output 3-4)
OUT_SOUT_DSR_GC                              = 'Output_SOut_DSR_GC'                     # (Output 3-5)
OUT_SOUT_DSR_AP2                             = 'Output_SOut_DSR_AP2'                    # (Output 3-6)
OUT_SOUT_DSR_AP1                             = 'Output_SOut_DSR_AP1'                    # (Output 3-7)
OUT_SOUT_DSR_LOCAL                           = 'Output_SOut_DSR_Local'                  # (Output 3-8)

# (Output 4-x) FCST detail
DF_OUT_SIN_FCST_AP1                          = 'df_output_Sell_In_FCST_GI_AP1'          # (Output 4-1)
DF_OUT_SIN_FCST_AP2                          = 'df_output_Sell_In_FCST_GI_AP2'          # (Output 4-2)
DF_OUT_SIN_FCST_GC                           = 'df_output_Sell_In_FCST_GI_GC'           # (Output 4-3)
DF_OUT_SIN_FCST_LOCAL                        = 'df_output_Sell_In_FCST_GI_Local'        # (Output 4-4)
DF_OUT_SOUT_FCST_AP1                         = 'df_output_Sell_Out_FCST_AP1'            # (Output 4-5)
DF_OUT_SOUT_FCST_AP2                         = 'df_output_Sell_Out_FCST_AP2'            # (Output 4-6)
DF_OUT_SOUT_FCST_GC                          = 'df_output_Sell_Out_FCST_GC'             # (Output 4-7)
DF_OUT_SOUT_FCST_LOCAL                       = 'df_output_Sell_Out_FCST_Local'          # (Output 4-8)
DF_OUT_SIN_FCST_NEW_MODEL                    = 'df_output_Sell_In_FCST_GI_New_Model'    # (Output 4-9)
DF_OUT_FLOORING_FCST                         = 'df_output_Flooring_FCST'                # (Output 4-10)
DF_OUT_BO_FCST                               = 'df_output_BO_FCST'                      # (Output 4-11)

# (Output 5) Estimated Price
DF_OUT_ESTIMATED_PRICE_LOCAL                 = 'df_output_Estimated_Price_Local'        # (Output 5)

# (Output 6-x) Split Ratio / Stretch Plan
DF_OUT_SIN_SPLIT_AP1                         = 'df_output_Sell_In_FCST_GI_Split_Ratio_AP1'   # (Output 6-1)
DF_OUT_SIN_SPLIT_AP2                         = 'df_output_Sell_In_FCST_GI_Split_Ratio_AP2'   # (Output 6-2)
DF_OUT_SIN_SPLIT_GC                          = 'df_output_Sell_In_FCST_GI_Split_Ratio_GC'    # (Output 6-3)
DF_OUT_SIN_SPLIT_LOCAL                       = 'df_output_Sell_In_FCST_GI_Split_Ratio_Local' # (Output 6-4)
DF_OUT_SOUT_SPLIT_AP1                        = 'df_output_Sell_Out_FCST_Split_Ratio_AP1'     # (Output 6-5)
DF_OUT_SOUT_SPLIT_AP2                        = 'df_output_Sell_Out_FCST_Split_Ratio_AP2'     # (Output 6-6)
DF_OUT_SOUT_SPLIT_GC                         = 'df_output_Sell_Out_FCST_Split_Ratio_GC'      # (Output 6-7)
DF_OUT_SOUT_SPLIT_LOCAL                      = 'df_output_Sell_Out_FCST_Split_Ratio_Local'   # (Output 6-8)
DF_OUT_SIN_STRETCH_SPLIT                     = 'df_output_Sell_In_Stretch_Plan_Split_Ratio'  # (Output 6-9)
DF_OUT_SIN_STRETCH_ASSORT                    = 'df_output_Sell_In_Stretch_Plan_Assortment'   # (Output 6-10)
DF_OUT_SOUT_STRETCH_SPLIT                    = 'df_output_Sell_Out_Stretch_Plan_Split_Ratio' # (Output 6-11)
DF_OUT_SOUT_STRETCH_ASSORT                   = 'df_output_Sell_Out_Stretch_Plan_Assortment'  # (Output 6-12)

# (Output 7-x) eStore GI Ratio (raw)
DF_OUT_SIN_USER_GI_RATIO                     = 'df_output_Sell_In_User_GI_Ratio'        # (Output 7-1)
DF_OUT_SIN_ISSUE_GI_RATIO                    = 'df_output_Sell_In_Issue_GI_Ratio'       # (Output 7-2)
DF_OUT_SIN_FINAL_GI_RATIO                    = 'df_output_Sell_In_Final_GI_Ratio'       # (Output 7-3)
DF_OUT_SIN_BESTFIT_GI_RATIO                  = 'df_output_Sell_In_BestFit_GI_Ratio'     # (Output 7-4)
DF_OUT_SIN_USER_ITEM_GI_RATIO                = 'df_output_Sell_In_User_Item_GI_Ratio'   # (Output 7-5)
DF_OUT_SIN_ISSUE_ITEM_GI_RATIO               = 'df_output_Sell_In_Issue_Item_GI_Ratio'  # (Output 7-6)

# (Output 8-x) eStore GI Ratio (modify)
DF_OUT_SIN_USER_MODIFY_GI_RATIO              = 'df_output_Sell_In_User_Modify_GI_Ratio'   # (Output 8-1)
DF_OUT_SIN_ISSUE_MODIFY_GI_RATIO             = 'df_output_Sell_In_Issue_Modify_GI_Ratio'  # (Output 8-2)


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

# ───── Ship-To → Level LUT ───────────────────────
def build_shipto_level_lut(df_dim: pd.DataFrame):
    """
    Return (pd.Index, np.ndarray[int32], dict) for fast level lookup.
    """
    COL_LVS = [

        (COL_STD6 ,7),
        (COL_STD5, 6), (COL_STD4, 5), (COL_STD3, 4),
        (COL_STD2, 3), (COL_STD1, 2)
    ]
    lut = {}
    for col, lv in COL_LVS:
        lut.update({code: lv for code in df_dim[col].dropna().unique()})
    idx = pd.Index(lut.keys(), dtype=object)
    arr = np.fromiter(lut.values(), dtype='int32')
    return idx, arr, lut

def build_shipto_dim_arrays(df_dim: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
    """
    Returns
    -------
    dim_idx : Index(level-7 ShipTo)
    lv_arrs : ndarray shape(n,6) [LV2 … LV7]
              (컬럼순 : 2,3,4,5,6,7)
    """
    dim_idx = df_dim.set_index(COL_SHIP_TO)
    lv_cols = [COL_STD1, COL_STD2, COL_STD3,
               COL_STD4, COL_STD5, COL_STD6]
    lv_arrs = dim_idx[lv_cols].to_numpy(dtype=object)
    return dim_idx.index, lv_arrs
# -------------------------------------------------


def round_half_up_to_2(series: pd.Series) -> pd.Series:
    """
    반올림 규칙: Half-Up (3번째 자리에서 5 이상이면 올림).
    벡터화 연산으로 빠르게 처리. NaN은 그대로 유지.
    """
    s = pd.to_numeric(series, errors='coerce')  # 문자열/빈값 → NaN
    mask = s.notna()
    s2 = s.copy()
    abs_s = np.abs(s2[mask].to_numpy(dtype='float64'))
    # Half-Up: sign * floor(|x|*100 + 0.5) / 100
    rounded = (np.sign(s2[mask]) * np.floor(abs_s * 100.0 + 0.5)) / 100.0
    s2.loc[mask] = rounded
    return s2

################################################################################################################──────────
#  공통 타입 변환  (❌ `global` 사용 금지)
#  호출 측에서 `input_dataframes` 를 인자로 넘겨준다.
################################################################################################################──────────
def _fn_prepare_input_types(dict_dfs: dict) -> None:
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
            include=["float64", "float32", "int64", "int32","int"]
        ).columns
        for col in num_cols:
            df[col].fillna(0, inplace=True)
            try:
                df[col] = df[col].astype("int32")
            except ValueError:
                df[col] = df[col].round().astype("int32")

def _fn_prepare_input_type(df: pd.DataFrame) -> None:
    if df.empty:
        return

    # 1) object 컬럼 : str → category
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).astype("category")

    # 2) numeric 컬럼 : fillna → int32
    num_cols = df.select_dtypes(
        include=["float64", "float32", "int64", "int32","int"]
    ).columns
    for col in num_cols:
        df[col].fillna(0, inplace=True)
        try:
            df[col] = df[col].astype("int32")
        except ValueError:
            df[col] = df[col].round().astype("int32")

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

    if is_local: 
        # 로컬인 경우 Output 폴더를 정리한다.
        for file in os.scandir(str_output_dir):
            os.remove(file.path)

        # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
        file_pattern = f"{os.getcwd()}/{str_input_dir}/*.csv" 
        csv_files = glob.glob(file_pattern)


        file_to_df_mapping = {
            f'{DF_IN_SALES_DOMAIN_DIMENSION 	}.csv' : 	DF_IN_SALES_DOMAIN_DIMENSION   			,
            f'{DF_IN_TIME                   	}.csv' :    DF_IN_TIME                              ,
            f'{DF_IN_ITEM_MASTER_LED_SIGNAGE	}.csv' :    DF_IN_ITEM_MASTER_LED_SIGNAGE           ,
            f'{DF_IN_ITEM_MASTER            	}.csv' :    DF_IN_ITEM_MASTER                       ,
            f'{DF_IN_SALES_DOMAIN_ESTORE    	}.csv' :    DF_IN_SALES_DOMAIN_ESTORE               ,
            f'{DF_IN_FORECAST_RULE          	}.csv' :    DF_IN_FORECAST_RULE                     ,
            f'{DF_IN_SALES_PRODUCT_ASN_DELTA	}.csv' :    DF_IN_SALES_PRODUCT_ASN_DELTA           ,
            f'{DF_IN_SALES_PRODUCT_ASN      	}.csv' :    DF_IN_SALES_PRODUCT_ASN                 ,
            f'{DF_IN_SIN_ASSORTMENT         	}.csv' :    DF_IN_SIN_ASSORTMENT                    ,
            f'{DF_IN_SOUT_ASSORTMENT        	}.csv' :    DF_IN_SOUT_ASSORTMENT                   ,
            f'{DF_IN_ESTIMATED_PRICE        	}.csv' :    DF_IN_ESTIMATED_PRICE                   ,
            f'{DF_IN_ACTION_PLAN_PRICE      	}.csv' :    DF_IN_ACTION_PLAN_PRICE                 ,
            f'{DF_IN_EXCHANGE_RATE_LOCAL    	}.csv' :    DF_IN_EXCHANGE_RATE_LOCAL               ,
            f'{DF_IN_SIN_SPLIT_AP1          	}.csv' :    DF_IN_SIN_SPLIT_AP1                     ,
            f'{DF_IN_SIN_SPLIT_AP2          	}.csv' :    DF_IN_SIN_SPLIT_AP2                     ,
            f'{DF_IN_SIN_SPLIT_GC           	}.csv' :    DF_IN_SIN_SPLIT_GC                      ,
            f'{DF_IN_SIN_SPLIT_LOCAL        	}.csv' :    DF_IN_SIN_SPLIT_LOCAL                   ,
            f'{DF_IN_SOUT_SPLIT_AP1         	}.csv' :    DF_IN_SOUT_SPLIT_AP1                    ,
            f'{DF_IN_SOUT_SPLIT_AP2         	}.csv' :    DF_IN_SOUT_SPLIT_AP2                    ,
            f'{DF_IN_SOUT_SPLIT_GC          	}.csv' :    DF_IN_SOUT_SPLIT_GC                     ,
            f'{DF_IN_SOUT_SPLIT_LOCAL       	}.csv' :    DF_IN_SOUT_SPLIT_LOCAL                  ,
            f'{DF_IN_SIN_STRETCH_SPLIT      	}.csv' :    DF_IN_SIN_STRETCH_SPLIT                 ,
            f'{DF_IN_SOUT_STRETCH_SPLIT     	}.csv' :    DF_IN_SOUT_STRETCH_SPLIT                ,
            f'{DF_IN_SIN_STRETCH_ASSORT     	}.csv' :    DF_IN_SIN_STRETCH_ASSORT                ,
            f'{DF_IN_SOUT_STRETCH_ASSORT    	}.csv' :    DF_IN_SOUT_STRETCH_ASSORT               ,
            f'{DF_IN_SOUT_SIMUL_MASTER      	}.csv' :    DF_IN_SOUT_SIMUL_MASTER                 ,
            f'{DF_IN_SIN_USER_GI_RATIO      	}.csv' :    DF_IN_SIN_USER_GI_RATIO                 ,
            f'{DF_IN_SIN_ISSUE_GI_RATIO     	}.csv' :    DF_IN_SIN_ISSUE_GI_RATIO                ,
            f'{DF_IN_SIN_FINAL_GI_RATIO     	}.csv' :    DF_IN_SIN_FINAL_GI_RATIO                ,
            f'{DF_IN_SIN_BESTFIT_GI_RATIO   	}.csv' :    DF_IN_SIN_BESTFIT_GI_RATIO              ,
            f'{DF_IN_SIN_USER_ITEM_GI_RATIO 	}.csv' :    DF_IN_SIN_USER_ITEM_GI_RATIO            ,
            f'{DF_IN_SIN_ISSUE_ITEM_GI_RATIO    }.csv' :    DF_IN_SIN_ISSUE_ITEM_GI_RATIO                       	# eStore Ship‑to 목록
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
        # o9 에서 
        input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION 	] = df_in_Sales_Domain_Dimension         	
        input_dataframes[DF_IN_TIME                     ] = df_in_Time                           	
        input_dataframes[DF_IN_ITEM_MASTER_LED_SIGNAGE  ] = df_in_Item_Master_LED_SIGNAGE 
        input_dataframes[DF_IN_ITEM_MASTER              ] = df_in_Item_Master      	
        input_dataframes[DF_IN_SALES_DOMAIN_ESTORE      ] = df_in_Sales_Domain_Estore            	
        input_dataframes[DF_IN_FORECAST_RULE            ] = df_in_Forecast_Rule                  	
        input_dataframes[DF_IN_SALES_PRODUCT_ASN_DELTA  ] = df_in_Sales_Product_ASN_Delta        	
        input_dataframes[DF_IN_SALES_PRODUCT_ASN        ] = df_in_Sales_Product_ASN              	
        input_dataframes[DF_IN_SIN_ASSORTMENT           ] = df_in_SIn_Assortment                 	
        input_dataframes[DF_IN_SOUT_ASSORTMENT          ] = df_in_SOut_Assortment                	
        input_dataframes[DF_IN_ESTIMATED_PRICE          ] = df_in_Estimated_Price                	
        input_dataframes[DF_IN_ACTION_PLAN_PRICE        ] = df_in_Action_Plan_Price              	
        input_dataframes[DF_IN_EXCHANGE_RATE_LOCAL      ] = df_in_Exchange_Rate_Local            	
        input_dataframes[DF_IN_SIN_SPLIT_AP1            ] = df_in_Sell_In_FCST_GI_Split_Ratio_AP1
        input_dataframes[DF_IN_SIN_SPLIT_AP2            ] = df_in_Sell_In_FCST_GI_Split_Ratio_AP2   
        input_dataframes[DF_IN_SIN_SPLIT_GC             ] = df_in_Sell_In_FCST_GI_Split_Ratio_GC    
        input_dataframes[DF_IN_SIN_SPLIT_LOCAL          ] = df_in_Sell_In_FCST_GI_Split_Ratio_Local 
        input_dataframes[DF_IN_SOUT_SPLIT_AP1           ] = df_in_Sell_Out_FCST_Split_Ratio_AP1     
        input_dataframes[DF_IN_SOUT_SPLIT_AP2           ] = df_in_Sell_Out_FCST_Split_Ratio_AP2     
        input_dataframes[DF_IN_SOUT_SPLIT_GC            ] = df_in_Sell_Out_FCST_Split_Ratio_GC      
        input_dataframes[DF_IN_SOUT_SPLIT_LOCAL         ] = df_in_Sell_Out_FCST_Split_Ratio_Local   
        input_dataframes[DF_IN_SIN_STRETCH_SPLIT        ] = df_in_Sell_In_Stretch_Plan_Split_Ratio  
        input_dataframes[DF_IN_SOUT_STRETCH_SPLIT       ] = df_in_Sell_Out_Stretch_Plan_Split_Ratio 
        input_dataframes[DF_IN_SIN_STRETCH_ASSORT       ] = df_in_Sell_In_Stretch_Plan_Assortment   
        input_dataframes[DF_IN_SOUT_STRETCH_ASSORT      ] = df_in_Sell_Out_Stretch_Plan_Assortment  
        input_dataframes[DF_IN_SOUT_SIMUL_MASTER        ] = df_in_Sell_Out_Simul_Master             
        input_dataframes[DF_IN_SIN_USER_GI_RATIO        ] = df_in_Sell_In_User_GI_Ratio             
        input_dataframes[DF_IN_SIN_ISSUE_GI_RATIO       ] = df_in_Sell_In_Issue_GI_Ratio            
        input_dataframes[DF_IN_SIN_FINAL_GI_RATIO       ] = df_in_Sell_In_Final_GI_Ratio            
        input_dataframes[DF_IN_SIN_BESTFIT_GI_RATIO     ] = df_in_Sell_In_BestFit_GI_Ratio          
        input_dataframes[DF_IN_SIN_USER_ITEM_GI_RATIO   ] = df_in_Sell_In_User_Item_GI_Ratio        
        input_dataframes[DF_IN_SIN_ISSUE_ITEM_GI_RATIO  ] = df_in_Sell_In_Issue_Item_GI_Ratio       


    # type convert : str ==> category, int ==> int32
    fn_convert_type(input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION], 'Sales Domain', str)    

    fn_convert_type(input_dataframes[DF_IN_TIME], 'Time.', str)  

    fn_convert_type(input_dataframes[DF_IN_SALES_DOMAIN_ESTORE], 'Sales Domain.', str)  

    fn_convert_type(input_dataframes[DF_IN_FORECAST_RULE], 'Sales Domain', str) 
    fn_convert_type(input_dataframes[DF_IN_FORECAST_RULE], 'FORECAST_RULE', 'int32') 
    # _fn_prepare_input_types({f'DF_IN_FORECAST_RULE':input_dataframes[DF_IN_FORECAST_RULE]})
    _fn_prepare_input_type(input_dataframes[DF_IN_FORECAST_RULE])
    #
    fn_convert_type(input_dataframes[DF_IN_SALES_PRODUCT_ASN_DELTA], 'Sales Domain', str)  
    fn_convert_type(input_dataframes[DF_IN_SALES_PRODUCT_ASN], 'Sales Domain', str)     

    fn_convert_type(input_dataframes[DF_IN_SIN_ASSORTMENT], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_ASSORTMENT], 'S/In FCST(GI)', 'int32') 
    _fn_prepare_input_type(input_dataframes[DF_IN_SIN_ASSORTMENT])

    fn_convert_type(input_dataframes[DF_IN_SOUT_ASSORTMENT], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_ASSORTMENT], 'S/Out FCST Assortment', 'int32') 
    _fn_prepare_input_type(input_dataframes[DF_IN_SOUT_ASSORTMENT])
    #
    fn_convert_type(input_dataframes[DF_IN_ESTIMATED_PRICE], 'Sales Domain', str)    
    # # 가격 관련 컬럼명 상수 세트
    # PRICE_COLS_EST = [COL_EST_PRICE_LOCAL, COL_EST_PRICE_MODIFY_LOCAL]
    # # ... (CSV 매핑 및 input_dataframes 채운 뒤)    # -----------------------------
    # # Estimated Price: 2자리 Half-Up 반올림 → (선택) float32 다운캐스트
    # # -----------------------------
    # if DF_IN_ESTIMATED_PRICE in input_dataframes:
    #     df_price = input_dataframes[DF_IN_ESTIMATED_PRICE]

    #     for col in PRICE_COLS_EST:
    #         if col in df_price.columns:
    #             # 1) 반올림 (Half-Up)
    #             s_rounded = round_half_up_to_2(df_price[col])

    #             # 2) dtype 결정
    #             #    - 계산 안정성 우선: float64 유지
    #             #    - 메모리 우선: float32로 다운캐스트
    #             # 여기서는 메모리 절약을 원하시면 아래 라인 사용:
    #             # df_price[col] = s_rounded.astype("float32")

    #             # 메모리보다 계산 안정성(추가 곱셈/합계)이 중요하면:
    #             df_price[col] = s_rounded.astype("float64")

    #     input_dataframes[DF_IN_ESTIMATED_PRICE] = df_price

    fn_convert_type(input_dataframes[DF_IN_ESTIMATED_PRICE], COL_EST_PRICE_MODIFY_LOCAL, 'float64') 
    fn_convert_type(input_dataframes[DF_IN_ESTIMATED_PRICE], COL_EST_PRICE_LOCAL, 'float64') 
 
    fn_convert_type(input_dataframes[DF_IN_ACTION_PLAN_PRICE], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_ACTION_PLAN_PRICE], COL_MONTH, str)
    fn_convert_type(input_dataframes[DF_IN_ACTION_PLAN_PRICE], COL_AP_PRICE_USD, 'float64')

    fn_convert_type(input_dataframes[DF_IN_EXCHANGE_RATE_LOCAL], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_EXCHANGE_RATE_LOCAL], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_EXCHANGE_RATE_LOCAL], 'Exchange', 'float64')

    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_AP1], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_AP1], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_AP1], 'S/In FCST(GI)', 'float64')

    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_AP2], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_AP2], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_AP2], 'S/In FCST(GI)', 'float64')

    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_GC], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_GC], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_GC], 'S/In FCST(GI)', 'float64')

    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_LOCAL], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_LOCAL], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_SPLIT_LOCAL], 'S/In FCST(GI)', 'float64')

    # df_in_Sell_Out_FCST_Split_Ratio_AP1
    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_AP1], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_AP1], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_AP1], 'S/Out FCST', 'float64')

    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_AP2], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_AP2], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_AP2], 'S/Out FCST', 'float64')

    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_GC], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_GC], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_GC], 'S/Out FCST', 'float64')

    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_LOCAL], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_LOCAL], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_SPLIT_LOCAL], 'S/Out FCST', 'float64')

    fn_convert_type(input_dataframes[DF_IN_SIN_STRETCH_SPLIT], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_STRETCH_SPLIT], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_STRETCH_SPLIT], 'S/In Stretch', 'float64')

    fn_convert_type(input_dataframes[DF_IN_SOUT_STRETCH_SPLIT], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_STRETCH_SPLIT], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_STRETCH_SPLIT], 'S/Out Stretch', 'float64')

    # df_in_Sell_In_Stretch_Plan_Assortment
    fn_convert_type(input_dataframes[DF_IN_SIN_STRETCH_ASSORT], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_STRETCH_ASSORT], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_STRETCH_ASSORT], 'S/In Stretch Plan', 'float64')
    # df_in_Sell_Out_Stretch_Plan_Assortment
    fn_convert_type(input_dataframes[DF_IN_SOUT_STRETCH_ASSORT], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_STRETCH_ASSORT], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SOUT_STRETCH_ASSORT], 'S/Out Stretch Plan', 'float64')
    # df_in_Sell_Out_Simul_Master
    fn_convert_type(input_dataframes[DF_IN_SOUT_SIMUL_MASTER], 'Sales Domain', str)
    # df_in_Sell_In_User_GI_Ratio : DF_IN_SIN_USER_GI_RATIO
    fn_convert_type(input_dataframes[DF_IN_SIN_USER_GI_RATIO], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_USER_GI_RATIO], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_USER_GI_RATIO], 'S/In', 'float64')
    # df_in_Sell_In_Issue_GI_Ratio
    fn_convert_type(input_dataframes[DF_IN_SIN_ISSUE_GI_RATIO], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_ISSUE_GI_RATIO], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_ISSUE_GI_RATIO], 'S/In', 'float64')
    # df_in_Sell_In_Final_GI_Ratio
    fn_convert_type(input_dataframes[DF_IN_SIN_FINAL_GI_RATIO], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_FINAL_GI_RATIO], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_FINAL_GI_RATIO], 'S/In', 'float64')
    # df_in_Sell_In_BestFit_GI_Ratio
    fn_convert_type(input_dataframes[DF_IN_SIN_BESTFIT_GI_RATIO], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_BESTFIT_GI_RATIO], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_BESTFIT_GI_RATIO], 'S/In', 'float64')
    # df_in_Sell_In_User_Item_GI_Ratio
    fn_convert_type(input_dataframes[DF_IN_SIN_USER_ITEM_GI_RATIO], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_USER_ITEM_GI_RATIO], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_USER_ITEM_GI_RATIO], 'S/In', 'float64')
    # df_in_Sell_In_Issue_Item_GI_Ratio
    fn_convert_type(input_dataframes[DF_IN_SIN_ISSUE_ITEM_GI_RATIO], 'Sales Domain', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_ISSUE_ITEM_GI_RATIO], 'Time', str)
    fn_convert_type(input_dataframes[DF_IN_SIN_ISSUE_ITEM_GI_RATIO], 'S/In', 'float64')





    fn_prepare_input_types(input_dataframes)


# 1. Sanitize date string
# def sanitize_date_string(x):
#     if pd.isna(x):
#         return ''
#     x = str(x).strip()
#     for token in ['PM', 'AM', '오전', '오후']:
#         if token in x:
#             x = x.split(token)[0].strip()
#     return x[:10]  # Keep only 'YYYY/MM/DD'

def sanitize_date_string(x: object) -> str:
    """
    * 입력 예
        12/4/2020 12:00:00 AM
        2025-02-03 12:00:00 AM
        2019-09-16
        ''
    * 처리
        ① 공백 앞(= time 부분) 제거  
        ② `-` → `/` 통일  
        ③ 자리수에 따라  
            - YYYY/MM/DD  → 그대로  
            - M/D/YYYY    → 0-padding 후 YYYY/MM/DD 로 변환  
        ④ 실패 시 '' 리턴
    """
    if pd.isna(x) or str(x).strip() == '':
        return ''

    s = str(x).strip()

    # ① 공백(혹은 T) 이후 time 문자열 제거
    s = re.split(r'\s+|T', s, maxsplit=1)[0]

    # ② 구분자 통일
    s = s.replace('-', '/')

    # ③ 날짜 포맷 판별·정규화
    parts = s.split('/')
    try:
        if len(parts) == 3:
            # case-A : YYYY/MM/DD
            if len(parts[0]) == 4:
                y, m, d = parts
            # case-B : M/D/YYYY  또는  MM/DD/YYYY
            else:
                m, d, y = parts
            dt_obj = datetime.datetime(int(y), int(m), int(d))    # 유효성 체크
            return dt_obj.strftime('%Y/%m/%d')              # zero-padding 포함
    except Exception:
        pass       # fall-through → 실패 처리
    return ''       # 파싱 실패

# 벡터라이즈 버전
v_sanitize_date_string = np.vectorize(sanitize_date_string, otypes=[object])



# v_sanitize_date_string = np.vectorize(sanitize_date_string)

# 2. Validate date
@np.vectorize
def is_valid_date(x):
    try:
        if pd.isna(x) or x == '':
            return True
        datetime.datetime.strptime(str(x), '%Y/%m/%d')
        return True
    except:
        return False

# 3. Convert to datetime
@np.vectorize
def safe_strptime(x):
    try:
        return datetime.datetime.strptime(str(x), '%Y/%m/%d') if pd.notna(x) and x != '' else None
    except:
        return None

# 4. Convert to partial week with error-checking
@np.vectorize
def to_partial_week(item,shipto,x):
    try:
        if x is not None and x != '':
            # If x is not already a Python datetime, try to convert it
            if not isinstance(x, datetime.datetime):
                # This conversion uses pandas to ensure we get a proper Python datetime
                x = pd.to_datetime(x).to_pydatetime()
            # Convert Python datetime to numpy.datetime64 with seconds precision
            np_dt = np.datetime64(x, 's')
            seconds = (np_dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
            dt_utc = datetime.datetime.utcfromtimestamp(seconds)
            return common.gfn_get_partial_week(dt_utc, True)
        else: 
            return ''
    except Exception as e:
        print("Error in to_partial_week with value:", item, shipto, x, "Error:", e)
        return ''


def to_partial_week_datetime(x: Union[str, datetime.date, datetime.datetime]) -> str:
    """
    Robust date-string → 'YYYYWWA/B' converter.
    1. try ``pandas.to_datetime`` (handles *most* inputs fast, incl. numpy64)
    2. fallback to explicit ``strptime`` with the four formats above
    3. log & *raise* if none succeed
    Returns empty-string for ``None`` / '' / NaN.
    """
    _DATE_FMTS = (
        '%Y/%m/%d',   # ① 2025/04/16
        '%Y-%m-%d',   # ② 2025-04-16
        '%m-%d-%Y',   # ③ 04-16-2025
        '%m/%d/%Y',   # ④ 04/16/2025
    )

    if x is None or (isinstance(x, str) and not x.strip()) or pd.isna(x):
        return ''
    # ---------- 1) pandas fast-path ----------
    try:
        dt = pd.to_datetime(x, errors='raise').to_pydatetime()
        return common.gfn_get_partial_week(dt, True)
    except Exception as e_fast:        # noqa: BLE001
        last_exc = e_fast   # remember last exception for logging
    # ---------- 2) explicit strptime fallbacks ----------
    x_str = str(x).strip()
    for fmt in _DATE_FMTS:
        try:
            dt = datetime.datetime.strptime(x_str, fmt)
            return common.gfn_get_partial_week(dt, True)
        except ValueError as exc:
            last_exc = exc              # keep most recent for message
            continue
    # ---------- 3) give up ----------
    msg = f"[to_partial_week_datetime] un-parsable date: {x!r} – last error: {last_exc}"
    logger.Note(p_note=msg, p_log_level=LOG_LEVEL.error())   # or logger.error(...)
    raise ValueError(msg)



def to_add_week(row):
    try:
        if x is not None and x != '':
            # If x is not already a Python datetime, try to convert it
            dt = common.gfn_add_week(x, -1)
            return common.gfn_get_partial_week(dt, True)
        else: 
            return ''
    except Exception as e:
        print("Error in to_partial_week with value:", item, shipto, x, "Error:", e)
        return ''

@np.vectorize
def is_valid_add_week(x):
    try:
        dt = common.gfn_add_week(x, -1)
        return True
    except:
        return False

# ──────────────────────────────────────────────────────────────────────────────
# STEP-00 : Ship-To Dimension LUT
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
def step00_load_shipto_dimension(
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


# ── 전처리 : 주차 카테고리를 'YYYYWW' 숫자순으로 정렬 & ordered 지정 ──
def _order_week_cat(sr_cat: pd.Series) -> pd.Series:
    if sr_cat.dtype.name != 'category' or sr_cat.cat.ordered:
        return sr_cat            # 이미 ordered 면 그대로
    cats = sr_cat.cat.categories.astype(str)
    # '202447' 같은 숫자부로 sort → category 재생성
    ordered_cats = pd.Series(cats).sort_values(key=lambda s: s.astype(int)).to_numpy()
    return sr_cat.cat.set_categories(ordered_cats, ordered=True)
    


################################################################################################################
#################### Start Step Functions  ##########
################################################################################################################
# To do : 여기 아래에 Step Function 들을 구현.
########################################################################################################################
# Step 1-1 : Sales Product ASN Delta 전처리
########################################################################################################################
@_decoration_
def fn_step01_01_preprocess_sales_product_asn_delta(
        df_asn_delta: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 1-1) Sales Product ASN Delta 전처리
    ----------------------------------------------------------
    • 입력 미존재 시 즉시 Exception 발생(프로그램 종료)
    • Version 컬럼(COL_VERSION) 제거
    • 'Sales Product ASN Delta' → 'Sales Product ASN' 컬럼명 변경
    • 반환 컬럼 순서 : [Ship To, Item, Location, Sales Product ASN]
    • dtype : 모두 category 로 캐스팅
    """    # ── 0) 입력 방어 : 공백 데이터면 즉시 종료 ───────────────────────────────
    if df_asn_delta is None or df_asn_delta.empty:
        raise Exception('[Step 1-1] Input 6 (df_in_Sales_Product_ASN_Delta) is empty. Program terminated.')

    # ── 1) 필요한 컬럼 존재 확인 ────────────────────────────────────────────
    REQ_COLS = [
        COL_SHIP_TO,         # 'Sales Domain.[Ship To]'
        COL_ITEM,            # 'Item.[Item]'
        COL_LOCATION,        # 'Location.[Location]'
        COL_SALES_PRODUCT_ASN_DELTA  # 'Sales Product ASN Delta'
    ]
    missing = [c for c in REQ_COLS if c not in df_asn_delta.columns]
    if missing:
        raise KeyError(f"[Step 1-1] Required columns missing in df_asn_delta: {missing}")

    # ── 2) 사본 생성 & 최소 컬럼만 선별 ──────────────────────────────────────
    use_cols = [COL_VERSION] + REQ_COLS if COL_VERSION in df_asn_delta.columns else REQ_COLS
    df = df_asn_delta.loc[:, use_cols].copy(deep=False)

    # ── 3) Version 컬럼 제거 ────────────────────────────────────────────────
    if COL_VERSION in df.columns:
        df.drop(columns=[COL_VERSION], inplace=True)

    # ── 4) 컬럼명 변경 : Delta → ASN ────────────────────────────────────────
    df.rename(columns={COL_SALES_PRODUCT_ASN_DELTA: COL_SALES_PRODUCT_ASN}, inplace=True)

    # ── 5) dtype 캐스팅 (메모리 절감) ───────────────────────────────────────
    #     전부 category 로 맞춰 대용량 처리 대비
    cast_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
    for c in cast_cols:
        # 이미 category 여도 재캐스팅 비용은 작고, object → category 로 바꾸면 메모리 절감 효과 큼
        df[c] = df[c].astype('category')

    # ── 6) 컬럼 순서 정렬 ───────────────────────────────────────────────────
    df = df[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]]

    # ── 7) 원본 즉시 해제(대용량 대비) ──────────────────────────────────────
    del df_asn_delta
    gc.collect()

    return df

########################################################################################################################
# Step 1-2 : Sales Product ASN 전처리  (기본 ASN에서 Delta와 겹치는 (ShipTo, Item, Location) 제거)
########################################################################################################################
@_decoration_
def fn_step01_02_preprocess_sales_product_asn(
        df_asn_base: pd.DataFrame,              # Input 7 : df_in_Sales_Product_ASN (원본)
        df_step01_01_delta: pd.DataFrame        # Step 1-1 결과 : (ShipTo, Item, Location, Sales Product ASN)
) -> pd.DataFrame:
    """
    Step 1-2) Sales Product ASN 전처리
    ----------------------------------------------------------
    • Version 컬럼(COL_VERSION) 삭제 (기본 ASN 표에서만)
    • 기본 ASN 에서 ASN Delta(1-1 결과)와 (ShipTo, Item, Location) 키가 겹치는 행 제거
      - 목적 : 1-3에서 Delta와 합칠 때 Y→N, Y→Y 충돌 제거
    • 반환 스키마 : [Sales Domain.[Ship To], Item.[Item], Location.[Location], Sales Product ASN]
    • 모든 컬럼 dtype : category
    • for-loop 없이 벡터화, 대용량/메모리 고려
    """    # ── 0) 입력 방어 ────────────────────────────────────────────────────────
    if df_asn_base is None or df_asn_base.empty:
        raise Exception('[Step 1-2] Input 7 (df_in_Sales_Product_ASN) is empty.')
    if df_step01_01_delta is None or df_step01_01_delta.empty:
        # 1-1에서 이미 빈 값이면 종료하도록 했지만, 재확인
        raise Exception('[Step 1-2] Step 1-1 result is empty. Program terminated.')

    # ── 1) 필수 컬럼 확인 ───────────────────────────────────────────────────
    REQ_BASE = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
    missing_base = [c for c in REQ_BASE + [COL_VERSION] if c not in df_asn_base.columns and c != COL_VERSION]
    if missing_base:
        raise KeyError(f"[Step 1-2] Required columns missing in df_asn_base: {missing_base}")

    REQ_DELTA_KEYS = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    missing_delta = [c for c in REQ_DELTA_KEYS if c not in df_step01_01_delta.columns]
    if missing_delta:
        raise KeyError(f"[Step 1-2] Required key columns missing in Step 1-1 result: {missing_delta}")

    # ── 2) 기본 ASN 사본 생성 및 Version 제거 ───────────────────────────────
    use_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
    # Version 컬럼이 있으면 드롭
    if COL_VERSION in df_asn_base.columns:
        df_asn = df_asn_base.drop(columns=[COL_VERSION]).loc[:, use_cols].copy(deep=False)
    else:
        df_asn = df_asn_base.loc[:, use_cols].copy(deep=False)

    # ── 3) (ShipTo, Item, Location) 키로 안티조인 (Delta와 겹치는 기본 ASN 제거) ─
    # 카테고리 간 카테고리-세트 불일치로 인한 merge 비용/오류를 피하기 위해
    # 문자열 키로 벡터화 비교(mask) 수행 (Delta 쪽 고유키 set만 생성 → 메모리 최소화)
    # left key
    k1 = df_asn[COL_SHIP_TO].astype('object').astype(str)
    k2 = df_asn[COL_ITEM].astype('object').astype(str)
    k3 = df_asn[COL_LOCATION].astype('object').astype(str)
    key_left = k1 + '|' + k2 + '|' + k3

    # right unique key set
    r1 = df_step01_01_delta[COL_SHIP_TO].astype('object').astype(str)
    r2 = df_step01_01_delta[COL_ITEM].astype('object').astype(str)
    r3 = df_step01_01_delta[COL_LOCATION].astype('object').astype(str)
    right_keys = (r1 + '|' + r2 + '|' + r3).unique()
    right_key_set = set(right_keys)   # Delta는 상대적으로 작다는 전제에서 set 사용

    # 안티조인 마스크
    mask_keep = ~key_left.isin(right_key_set)
    df_out = df_asn.loc[mask_keep, use_cols].copy(deep=False)

    # ── 4) dtype 카테고리 캐스팅 ────────────────────────────────────────────
    for c in use_cols:
        df_out[c] = df_out[c].astype('category')

    # ── 5) 중간 변수 및 원본 참조 해제 ───────────────────────────────────────
    del (df_asn_base, df_step01_01_delta, df_asn,
         k1, k2, k3, key_left, r1, r2, r3, right_keys, right_key_set)
    gc.collect()

    return df_out


########################################################################################################################
# Step 1-3 : Sales Product ASN 구성 (ShipTo × Item × Location)
########################################################################################################################
@_decoration_
def fn_step01_03_build_sales_product_asn(
        df_step01_01_delta: pd.DataFrame,          # Step 1-1 결과 : [ShipTo, Item, Location, Sales Product ASN]
        df_step01_02_base_filtered: pd.DataFrame   # Step 1-2 결과 : [ShipTo, Item, Location, Sales Product ASN]
) -> pd.DataFrame:
    """
    Step 1-3) Sales Product ASN 구성 (ShipTo × Item × Location)
    ----------------------------------------------------------
    • 입력
        - df_step01_01_delta        : 1-1 결과 (Delta 전처리, 'Sales Product ASN' 으로 정규화, Y/N 포함)
        - df_step01_02_base_filtered: 1-2 결과 (기본 ASN 중 Delta와 겹치지 않는 행만 남김, 주로 Y)
    • 처리
        - 두 DataFrame을 같은 스키마로 정렬하여 유니온(concat)
        - (안전) 중복 키(ShipTo,Item,Location) 제거 (원칙적으로 1-2에서 제거되어 없어야 함)
        - 모든 컬럼을 category로 캐스팅
    • 반환 스키마(Version 없이)
        [Sales Domain.[Ship To], Item.[Item], Location.[Location], Sales Product ASN]
    • 성능/메모리
        - for-loop 없음, concat + drop_duplicates 사용
        - concat 중 카테고리 범주가 다르면 일시적으로 object로 오를 수 있으므로, 마지막에 재-cast(category)
    """    
    
    # ── 0) 방어 코드 ───────────────────────────────────────────────────────
    if df_step01_01_delta is None or df_step01_01_delta.empty:
        # 설계상 1-1에서 빈 경우 프로그램 종료이므로 여기서도 방어적으로 예외 처리
        raise Exception('[Step 1-3] Step 1-1 result (ASN Delta) is empty. Program terminated.')

    # 1-2 결과는 비어 있어도 유니온의 의미상 문제 없음(전부 Delta로만 구성)
    if df_step01_02_base_filtered is None:
        df_step01_02_base_filtered = pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN])

    # ── 1) 공통 스키마 정렬 ────────────────────────────────────────────────
    USE_COLS = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]

    missing_11 = [c for c in USE_COLS if c not in df_step01_01_delta.columns]
    if missing_11:
        raise KeyError(f"[Step 1-3] Step 1-1 result missing columns: {missing_11}")

    missing_12 = [c for c in USE_COLS if c not in df_step01_02_base_filtered.columns]
    if missing_12:
        raise KeyError(f"[Step 1-3] Step 1-2 result missing columns: {missing_12}")

    df_delta = df_step01_01_delta.loc[:, USE_COLS].copy(deep=False)
    df_base  = df_step01_02_base_filtered.loc[:, USE_COLS].copy(deep=False)

    # ── 2) 유니온(행 결합) ────────────────────────────────────────────────
    #  - 서로 다른 카테고리 범주가 있으면 일시적으로 object로 승격될 수 있음 → 아래에서 재-cast
    df_union = pd.concat([df_delta, df_base], axis=0, ignore_index=True, copy=False)

    # ── 3) (안전) 중복 키 제거 ─────────────────────────────────────────────
    #  원칙적으로 1-2에서 Delta와 겹치는 키를 제거했으므로 중복이 없어야 함.
    #  혹시 모를 중복 대비로 subset 키 기준 고유화.
    df_union.drop_duplicates(subset=[COL_SHIP_TO, COL_ITEM, COL_LOCATION], keep='last', inplace=True)

    # ── 4) 값 정규화 & dtype 캐스팅 ───────────────────────────────────────
    #  'Sales Product ASN' 값은 {Y,N}만 허용(스펙). 혹시 다른 값이 있으면 문자열로 변환 후 카테고리화.
    if df_union[COL_SALES_PRODUCT_ASN].dtype.name != 'category':
        df_union[COL_SALES_PRODUCT_ASN] = df_union[COL_SALES_PRODUCT_ASN].astype('object').astype(str)
    # 최종 컬럼을 category로 캐스팅
    for c in USE_COLS:
        df_union[c] = df_union[c].astype('category')

    # ── 5) 메모리 정리 ────────────────────────────────────────────────────
    del (df_step01_01_delta, df_step01_02_base_filtered, df_delta, df_base)
    gc.collect()

    return df_union

########################################################################################################################
# Step 1-4 : Sales Product ASN No Location  (ordered categorical + groupby max)
########################################################################################################################
@_decoration_
def fn_step01_04_sales_product_asn_no_location(
        df_step01_02_base_filtered: pd.DataFrame   # Step 1-2 결과 : [ShipTo, Item, Location, Sales Product ASN ('Y'|'N')]
) -> pd.DataFrame:
    """
    Step 1-4) Sales Product ASN No Location
    ----------------------------------------------------------
    • 입력 : Step 1-2 결과 (ShipTo×Item×Location, 'Y'|'N')
    • 처리 :
        1) Location 제거 → (ShipTo, Item) 기준 groupby
        2) 'Y'/'N'을 ordered categorical(['N','Y'])로 변환 후 max 집계
           (→ 하나라도 Y가 있으면 Y, 아니면 N)
        3) 이후 1-5에서 anti-join의 기준으로 쓰이므로, 최종적으로 'Y'인 조합만 남긴다.
    • 반환 (모두 category) :
        [Sales Domain.[Ship To], Item.[Item], Sales Product ASN]
    """
    # ── 0) 방어 코드 ───────────────────────────────────────────────────────
    REQ_COLS = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
    missing = [c for c in REQ_COLS if c not in df_step01_02_base_filtered.columns]
    if missing:
        raise KeyError(f"[Step 1-4] Step 1-2 result missing columns: {missing}")    
    
    if df_step01_02_base_filtered.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN])

    # ── 1) 필요한 컬럼만 사용 ──────────────────────────────────────────────
    df = df_step01_02_base_filtered.loc[:, [COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN]].copy(deep=False)

    # ── 2) Y/N 정규화 → ordered categorical(['N','Y']) ────────────────────
    asn_norm = (
        df[COL_SALES_PRODUCT_ASN]
        .astype('object').astype(str).str.strip().str.upper()
    )
    # 비정상값/공백/NaN → 'N'
    asn_norm = asn_norm.where(asn_norm.isin(['Y', 'N']), 'N')
    ASN_CAT = pd.CategoricalDtype(categories=['N', 'Y'], ordered=True)
    df[COL_SALES_PRODUCT_ASN] = pd.Categorical(asn_norm, dtype=ASN_CAT)

    # ── 3) (ShipTo, Item) 기준 max 집계 ────────────────────────────────────
    gb_keys = [COL_SHIP_TO, COL_ITEM]
    df_agg = (
        df.groupby(gb_keys, sort=False, observed=True)
          .agg({COL_SALES_PRODUCT_ASN: 'max'})   # ordered categorical → 'Y' > 'N'
          .reset_index()
    )

    # ── 4) 'Y' 인 조합만 남김 (1-5 anti-join의 기준이 되므로) ──────────────
    # df_out = df_agg[df_agg[COL_SALES_PRODUCT_ASN] == 'Y'][gb_keys + [COL_SALES_PRODUCT_ASN]].copy()
    df_out = df_agg

    # ── 5) dtype 정리 (메모리 절감: category 유지) ─────────────────────────
    for c in gb_keys:
        df_out[c] = df_out[c].astype('category')
    # COL_SALES_PRODUCT_ASN 은 이미 ordered categorical(['N','Y'])

    # ── 6) 메모리 정리 ────────────────────────────────────────────────────
    del df_step01_02_base_filtered, df, df_agg
    gc.collect()

    return df_out

########################################################################################################################
# Step 1-5 : Sales Product ASN Delta No Location  (ordered categorical + groupby max)
########################################################################################################################
@_decoration_
def fn_step01_05_sales_product_asn_delta_no_location(
        df_step01_01_asn_delta: pd.DataFrame,      # Step 1-1 결과 : [ShipTo, Item, Location, Sales Product ASN ('Y'|'N')]
        df_step01_04_no_location: pd.DataFrame     # Step 1-4 결과 : [ShipTo, Item, (Sales Product ASN)]
) -> pd.DataFrame:
    """
    Step 1-5) Sales Product ASN Delta No Location
    ----------------------------------------------------------
    • 입력
        - Step 1-1 전처리 결과(Delta) : [ShipTo, Item, Location, Sales Product ASN('Y'|'N')]
        - Step 1-4 결과(No Location) : [ShipTo, Item, (Sales Product ASN)]
    • 처리
        1) (ShipTo, Item)로 groupby, 'Y'/'N'은 ordered categorical(['N','Y'])로 변환 후 max 집계
           (→ 하나라도 Y가 있으면 Y, 아니면 N)
        2) Step 1-4에 존재하는 (ShipTo, Item) 조합은 제거 (anti-join)
    • 반환 (모두 category)
        [Sales Domain.[Ship To], Item.[Item], Sales Product ASN]
    """    
    # ── 0) 방어 코드 ───────────────────────────────────────────────────────
    req_cols_11 = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
    miss_11 = [c for c in req_cols_11 if c not in df_step01_01_asn_delta.columns]
    if miss_11:
        raise KeyError(f"[Step 1-5] Step 1-1 result missing columns: {miss_11}")

    req_cols_14 = [COL_SHIP_TO, COL_ITEM]
    miss_14 = [c for c in req_cols_14 if c not in df_step01_04_no_location.columns]
    if miss_14:
        raise KeyError(f"[Step 1-5] Step 1-4 result missing columns: {miss_14}")

    if df_step01_01_asn_delta.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN])

    # ── 1) 필요한 컬럼만 사용 + Y/N 정규화 → ordered categorical(['N','Y']) ──
    use_cols = [COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN]
    df = df_step01_01_asn_delta.loc[:, use_cols].copy(deep=False)

    # 공백/NaN/기타 → 'N', 대문자화
    asn = (
        df[COL_SALES_PRODUCT_ASN]
        .astype('object').astype(str).str.strip().str.upper()
    )
    asn = asn.where(asn.isin(['Y', 'N']), 'N')
    ASN_CAT = pd.CategoricalDtype(categories=['N', 'Y'], ordered=True)
    df[COL_SALES_PRODUCT_ASN] = pd.Categorical(asn, dtype=ASN_CAT)

    # ── 2) (ShipTo, Item) 기준 max 집계 ───────────────────────────────────
    gb_keys = [COL_SHIP_TO, COL_ITEM]
    df_agg = (
        df.groupby(gb_keys, sort=False, observed=True)
          .agg({COL_SALES_PRODUCT_ASN: 'max'})       # ordered cat → 'Y' > 'N'
          .reset_index()
    )
    # 필요 시 문자열로 바꾸고 싶다면:
    # df_agg[COL_SALES_PRODUCT_ASN] = df_agg[COL_SALES_PRODUCT_ASN].astype(str)

    # ── 3) Step 1-4 에 존재하는 (ShipTo, Item) 제거 (anti-join) ────────────
    df14_keys = (
        df_step01_04_no_location
        .loc[:, gb_keys]
        .drop_duplicates(ignore_index=True)
    )

    df_out = (
        df_agg.merge(df14_keys, on=gb_keys, how='left', indicator=True)
              .loc[lambda x: x['_merge'] == 'left_only', gb_keys + [COL_SALES_PRODUCT_ASN]]
              .copy()
    )

    # ── 4) dtype 정리 (메모리 절감: category 유지) ─────────────────────────
    for c in gb_keys:
        df_out[c] = df_out[c].astype('category')
    # COL_SALES_PRODUCT_ASN 은 이미 ordered categorical(['N','Y'])

    # ── 5) 메모리 정리 ────────────────────────────────────────────────────
    del df_step01_01_asn_delta, df_step01_04_no_location, df, df_agg, df14_keys
    gc.collect()

    return df_out

########################################################################################################################
# Step 1-6 : Sales Product ASN output  (from Input 6: df_in_Sales_Product_ASN_Delta)
########################################################################################################################
@_decoration_
def fn_step01_06_output_sales_product_asn(
        df_in_asn_delta: pd.DataFrame,   # Input 6 : [Version, ShipTo, Item, Location, Sales Product ASN Delta (Y/N)]
        version: str                     # 예: 'CWV_DP'
) -> pd.DataFrame:
    """
    Step 1-6) Sales Product ASN output
    ----------------------------------------------------------
    • 입력  : df_in_Sales_Product_ASN_Delta (Input 6)
      - 컬럼 : [Version, ShipTo, Item, Location, Sales Product ASN Delta]
    • 처리  :
      - 컬럼명 변경 : 'Sales Product ASN Delta' → 'Sales Product ASN'
      - 값 변환     : 'N' → ''(빈값), 그 외는 'Y'만 유지
      - Version 값  : 함수 인자(version)로 덮어씀
      - 출력 스키마 : [Version, ShipTo, Item, Location, Sales Product ASN]
      - dtype       : category
    • 반환  : Output_Sales_Product_ASN 스키마의 DataFrame
    """
    # ── 0) 방어 코드 ───────────────────────────────────────────────────────
    REQ_COLS = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN_DELTA]
    missing = [c for c in REQ_COLS if c not in df_in_asn_delta.columns]
    if missing:
        raise KeyError(f"[Step 1-6] Input 6 missing columns: {missing}")    
    
    if df_in_asn_delta.empty:
        cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
        return pd.DataFrame(columns=cols)

    # ── 1) 필요한 컬럼만 복사 ──────────────────────────────────────────────
    df = df_in_asn_delta.loc[:, REQ_COLS].copy(deep=False)

    # ── 2) 컬럼명 변경 : Delta → 표준 컬럼 ─────────────────────────────────
    df.rename(columns={COL_SALES_PRODUCT_ASN_DELTA: COL_SALES_PRODUCT_ASN}, inplace=True)

    # ── 3) 값 변환 : 'Y' 유지, 나머지는 ''(빈값) ───────────────────────────
    norm = (
        df[COL_SALES_PRODUCT_ASN]
        .astype('object').astype(str).str.strip().str.upper()
    )
    df[COL_SALES_PRODUCT_ASN] = norm.where(norm.eq('Y'), '')

    # ── 4) Version 덮어쓰기 & 출력 스키마 정렬 ─────────────────────────────
    df[COL_VERSION] = version
    df_out = df[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]]

    # ── 5) dtype 정리 (메모리 절감: category) ─────────────────────────────
    for c in [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]:
        df_out[c] = df_out[c].astype('category')

    # ── 6) 메모리 정리 ────────────────────────────────────────────────────
    del df_in_asn_delta, df, norm
    gc.collect()

    return df_out

########################################################################################################################
# Step 1-7 : Sales Product ASN Delta output  (from Input 6: df_in_Sales_Product_ASN_Delta)
########################################################################################################################
@_decoration_
def fn_step01_07_output_sales_product_asn_delta(
        df_in_asn_delta: pd.DataFrame,   # Input 6 : [Version, ShipTo, Item, Location, Sales Product ASN Delta (Y/N)]
        version: str                     # 예: 'CWV_DP'
) -> pd.DataFrame:
    """
    Step 1-7) Sales Product ASN Delta output
    ----------------------------------------------------------
    • 입력  : df_in_Sales_Product_ASN_Delta (Input 6)
      - 컬럼 : [Version, ShipTo, Item, Location, Sales Product ASN Delta]
    • 처리  :
      - 값 변환     : 'Sales Product ASN Delta' 컬럼을 전체 ''(빈값)로 설정
      - Version 값  : 함수 인자(version)로 덮어씀
      - 출력 스키마 : [Version, ShipTo, Item, Location, Sales Product ASN Delta]
      - dtype       : category
    • 반환  : Output_Sales_Product_ASN_Delta 스키마의 DataFrame
    """
    # ── 0) 방어 코드 ───────────────────────────────────────────────────────
    REQ_COLS = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN_DELTA]
    missing = [c for c in REQ_COLS if c not in df_in_asn_delta.columns]
    if missing:
        raise KeyError(f"[Step 1-7] Input 6 missing columns: {missing}")    
        
    if df_in_asn_delta.empty:
        # 빈 DF를 지정 스키마로 반환
        df_empty = pd.DataFrame(columns=REQ_COLS)
        for c in REQ_COLS:
            df_empty[c] = df_empty[c].astype('category')
        return df_empty

    # ── 1) 필요한 컬럼만 복사 ──────────────────────────────────────────────
    df = df_in_asn_delta.loc[:, REQ_COLS].copy(deep=False)

    # ── 2) 값 변환 : 전체 ''(빈값)으로 설정 ────────────────────────────────
    #  ※ 스펙 '전체 Null 적용'을 카테고리 호환을 위해 ''로 대체(이후 시스템에서 Null로 처리될 수 있음)
    df[COL_SALES_PRODUCT_ASN_DELTA] = ''

    # ── 3) Version 덮어쓰기 & 출력 스키마 정렬 ─────────────────────────────
    df[COL_VERSION] = version
    df_out = df[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN_DELTA]]

    # ── 4) dtype 정리 (메모리 절감: category) ─────────────────────────────
    for c in [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN_DELTA]:
        df_out[c] = df_out[c].astype('category')

    # ── 5) 메모리 정리 ────────────────────────────────────────────────────
    del df_in_asn_delta, df
    gc.collect()

    return df_out


########################################################################################################################
# Step 2-1) Assortment 전처리
########################################################################################################################
@_decoration_
def fn_step02_01_preprocess_assortment(
        df_in_sin_assort: pd.DataFrame,          # Input 8 : df_in_SIn_Assortment
        df_in_sout_assort: pd.DataFrame,         # Input 9 : df_in_SOut_Assortment
        df_step01_01_asn_delta: pd.DataFrame     # Step 1-1 결과 : [ShipTo, Item, Location, Sales Product ASN]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 2-1) Assortment 전처리
    ----------------------------------------------------------
    • 필터 기준                          : Step 1-1 결과에 존재하는 Item만 사용 (ShipTo/Loc 무관, Item 단독 기준)
    • 공통 처리                          :
        - Version.[Version Name] 컬럼 삭제
        - 필요한 컬럼만 유지 (ShipTo, Item, Location, 각 Assortment Measure 4종)
        - 차원 컬럼(dtype) 최소화 (category 캐스팅)
    • 반환                               : (S/In 전처리 DF, S/Out 전처리 DF)
    • 스키마
        - S/In : [ShipTo, Item, Location, S/In FCST(GI) Assortment_AP1, _AP2, _GC, _Local]
        - S/Out: [ShipTo, Item, Location, S/Out FCST Assortment_AP1, _AP2, _GC, _Local]
    """    # ── 0) 방어 코드 ───────────────────────────────────────────────────────
    if df_step01_01_asn_delta is None or df_step01_01_asn_delta.empty:
        # 1-1 단계에서 이미 종료하도록 되어 있으나, 안전상 빈 DF 반환
        empty_si = pd.DataFrame(columns=[
            COL_SHIP_TO, COL_ITEM, COL_LOCATION,
            COL_SIN_ASSORT_AP1, COL_SIN_ASSORT_AP2, COL_SIN_ASSORT_GC, COL_SIN_ASSORT_LOCAL
        ])
        empty_so = pd.DataFrame(columns=[
            COL_SHIP_TO, COL_ITEM, COL_LOCATION,
            COL_SOUT_ASSORT_AP1, COL_SOUT_ASSORT_AP2, COL_SOUT_ASSORT_GC, COL_SOUT_ASSORT_LOCAL
        ])
        for c in [COL_SHIP_TO, COL_ITEM, COL_LOCATION]:
            empty_si[c] = empty_si[c].astype('category')
            empty_so[c] = empty_so[c].astype('category')
        return empty_si, empty_so

    # Step 1-1 에서 존재하는 Item 목록 (Item 기준 필터)
    items_from_delta = pd.Index(df_step01_01_asn_delta[COL_ITEM].astype('object'))

    # 각 테이블에서 반드시 있어야 할 컬럼
    NEED_DIM = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    NEED_SIN_MEAS  = [COL_SIN_ASSORT_AP1,  COL_SIN_ASSORT_AP2,  COL_SIN_ASSORT_GC,  COL_SIN_ASSORT_LOCAL]
    NEED_SOUT_MEAS = [COL_SOUT_ASSORT_AP1, COL_SOUT_ASSORT_AP2, COL_SOUT_ASSORT_GC, COL_SOUT_ASSORT_LOCAL]

    # 컬럼 체크 유틸
    def _check_cols(df: pd.DataFrame, need_cols: list, tag: str):
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise KeyError(f"[Step 2-1] {tag} missing columns: {missing}")

    _check_cols(df_in_sin_assort,  NEED_DIM + NEED_SIN_MEAS,  "S/In Assortment")
    _check_cols(df_in_sout_assort, NEED_DIM + NEED_SOUT_MEAS, "S/Out Assortment")

    # 공통 전처리 유틸
    def _prep(df_src: pd.DataFrame, meas_cols: list, tag: str) -> pd.DataFrame:
        use_cols = NEED_DIM + meas_cols
        # 필요한 컬럼만 슬라이싱 (+Version은 있으면 제거)
        df = df_src.loc[:, ([COL_VERSION] if COL_VERSION in df_src.columns else []) + use_cols].copy(deep=False)

        # Version 삭제
        if COL_VERSION in df.columns:
            df.drop(columns=[COL_VERSION], inplace=True)

        # Item 기준 필터링 (벡터연산)
        df = df[df[COL_ITEM].isin(items_from_delta)]

        # 차원 컬럼 category 캐스팅 (메모리 절감)
        for c in NEED_DIM:
            df[c] = df[c].astype('category')

        return df

    # ── 1) S/In 전처리 ───────────────────────────────────────────────────
    df_si = _prep(df_in_sin_assort, NEED_SIN_MEAS,  "S/In")

    # ── 2) S/Out 전처리 ──────────────────────────────────────────────────
    df_so = _prep(df_in_sout_assort, NEED_SOUT_MEAS, "S/Out")

    # ── 3) 메모리 정리 ───────────────────────────────────────────────────
    del df_in_sin_assort, df_in_sout_assort, df_step01_01_asn_delta
    gc.collect()

    return df_si, df_so


########################################################################################################################
# 🔧 Alias 상수  (Step 2-2에서 참조되는 미정의 상수 보완)
########################################################################################################################
# S/In Assortment 컬럼
COL_SIN_ASS_GC     = COL_SIN_ASSORT_GC
COL_SIN_ASS_AP2    = COL_SIN_ASSORT_AP2
COL_SIN_ASS_AP1    = COL_SIN_ASSORT_AP1
COL_SIN_ASS_LOCAL  = COL_SIN_ASSORT_LOCAL

# S/Out Assortment 컬럼
COL_SOUT_ASS_GC    = COL_SOUT_ASSORT_GC
COL_SOUT_ASS_AP2   = COL_SOUT_ASSORT_AP2
COL_SOUT_ASS_AP1   = COL_SOUT_ASSORT_AP1
COL_SOUT_ASS_LOCAL = COL_SOUT_ASSORT_LOCAL

# 반환 dict용 키 (Step 2 계열은 OUT_* 이름을 그대로 키로 사용해 충돌 방지)
STR_DF_OUT_SIN_GC     = OUT_SIN_ASSORTMENT_GC
STR_DF_OUT_SIN_AP2    = OUT_SIN_ASSORTMENT_AP2
STR_DF_OUT_SIN_AP1    = OUT_SIN_ASSORTMENT_AP1
STR_DF_OUT_SIN_LOCAL  = OUT_SIN_ASSORTMENT_LOCAL
STR_DF_OUT_SOUT_GC    = OUT_SOUT_ASSORTMENT_GC
STR_DF_OUT_SOUT_AP2   = OUT_SOUT_ASSORTMENT_AP2
STR_DF_OUT_SOUT_AP1   = OUT_SOUT_ASSORTMENT_AP1
STR_DF_OUT_SOUT_LOCAL = OUT_SOUT_ASSORTMENT_LOCAL

########################################################################################################################
# Step 2-2) Sales Product ASN(1-3) 로 Assortment Measure 구성 (GC/AP2/AP1/Local)
#  - Vectorised 방식 (fn_step05_make_actual_and_inv_fcst_level 참조)
########################################################################################################################
@_decoration_
def fn_step02_02_build_assortments(
        df_step01_03_asn_all      : pd.DataFrame,  # [ShipTo, Item, Location, Sales Product ASN (Y/N)]
        df_in_Forecast_Rule       : pd.DataFrame,  # [PG, ShipTo, RULE_GC/AP2/AP1/AP0, ISVALID]
        df_in_Item_Master         : pd.DataFrame,  # [Item, Item Std1]  ※ Item Std1 ↔ Product Group 매칭
        df_in_Sales_Domain_Dimension: pd.DataFrame,# [ShipTo, Sales Std1~6]
        df_in_Sales_Domain_Estore : pd.DataFrame   # [ShipTo]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    반환 : (sin_dict, sout_dict)
      • sin_dict : { OUT_SIn_Assortment_GC/AP2/AP1/Local : DF }
      • sout_dict: { OUT_SOut_Assortment_GC/AP2/AP1/Local: DF }  (기본 '-' + E-Store 상세 결합)
    컬럼 규칙
      • S/In : [ShipTo, Item, Location, Sales Product ASN, S/In FCST(GI) Assortment_XX]
      • S/Out: [ShipTo, Item, Location, Sales Product ASN, S/Out FCST Assortment_XX]
      • ASN='N' → Assortment_XX = NaN
    """    
    # ────────────────────────────────────────────────────────────────────────
    # 0) 방어 및 준비
    # ────────────────────────────────────────────────────────────────────────
    need_asn_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]
    miss = [c for c in need_asn_cols if c not in df_step01_03_asn_all.columns]
    if miss:
        raise KeyError(f"[Step 2-2] Step 1-3 result missing columns: {miss}")

    # Forecast-Rule 필수
    need_rule = [COL_PG, COL_SHIP_TO, COL_FRULE_GC_FCST, COL_FRULE_AP2_FCST,
                 COL_FRULE_AP1_FCST, COL_FRULE_AP0_FCST]
    miss = [c for c in need_rule if c not in df_in_Forecast_Rule.columns]
    if miss:
        raise KeyError(f"[Step 2-2] Forecast Rule missing columns: {miss}")

    # Sales-Domain Dimension 필수
    need_dim = [COL_SHIP_TO, COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6]
    miss = [c for c in need_dim if c not in df_in_Sales_Domain_Dimension.columns]
    if miss:
        raise KeyError(f"[Step 2-2] Sales-Domain Dimension missing columns: {miss}")

    # Item Master 필수
    need_im = [COL_ITEM, COL_ITEM_STD1]
    miss = [c for c in need_im if c not in df_in_Item_Master.columns]
    if miss:
        raise KeyError(f"[Step 2-2] Item Master missing columns: {miss}")

    # E-Store 목록
    if COL_SHIP_TO not in df_in_Sales_Domain_Estore.columns:
        raise KeyError(f"[Step 2-2] E-Store missing column: {COL_SHIP_TO}")

    # ────────────────────────────────────────────────────────────────────────
    # 1) ASN + PG 매핑 (Item.Std1 ↔ ForecastRule.PG)
    # ────────────────────────────────────────────────────────────────────────
    df_asn = df_step01_03_asn_all[need_asn_cols].copy(deep=False)
    pg_map = df_in_Item_Master.set_index(COL_ITEM)[COL_ITEM_STD1].astype(str).to_dict()
    df_asn[COL_PG] = df_asn[COL_ITEM].map(pg_map)

    # 유효 PG 만 남김
    df_asn = df_asn[~df_asn[COL_PG].isna()].copy()
    for c in [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PG, COL_SALES_PRODUCT_ASN]:
        df_asn[c] = df_asn[c].astype('category')

    # ────────────────────────────────────────────────────────────────────────
    # 2) RULE / DIM vector 준비 (fn_step05_make_actual_and_inv_fcst_level 방식)
    # ────────────────────────────────────────────────────────────────────────
    # Rule 유효값만
    df_rule = df_in_Forecast_Rule.copy()
    if COL_FRULE_ISVALID in df_rule.columns:
        df_rule = df_rule[df_rule[COL_FRULE_ISVALID].astype(str).str.upper().eq('Y')].copy()

    for col in (COL_FRULE_GC_FCST, COL_FRULE_AP2_FCST, COL_FRULE_AP1_FCST, COL_FRULE_AP0_FCST):
        df_rule[col] = df_rule[col].fillna(0).astype('int8')

    RULE: dict[tuple[str, str], tuple[int, int, int, int]] = {
        (str(pg), str(ship)): (gc, ap2, ap1, ap0)
        for pg, ship, gc, ap2, ap1, ap0 in zip(
            df_rule[COL_PG].astype(str),
            df_rule[COL_SHIP_TO].astype(str),
            df_rule[COL_FRULE_GC_FCST],
            df_rule[COL_FRULE_AP2_FCST],
            df_rule[COL_FRULE_AP1_FCST],
            df_rule[COL_FRULE_AP0_FCST],
        )
    }

    dim = df_in_Sales_Domain_Dimension.copy()
    dim_idx = dim.set_index(COL_SHIP_TO)
    LV_MAP = {
        lv: dim_idx[col].astype(str).to_dict()
        for lv, col in enumerate([COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6], start=1)
    }

    # ship_level : Ship코드가 처음 등장한(가장 상위) 레벨 (Lv-2 … Lv-7 ↔ Std1 … Std6)
    ship_level: dict[str, int] = {}
    for lv, col in enumerate([COL_STD1, COL_STD2, COL_STD3, COL_STD4, COL_STD5, COL_STD6], start=2):
        for code in dim[col].dropna().astype(str).unique():
            ship_level.setdefault(code, lv)

    def parent_of(arr: np.ndarray, lv: int) -> np.ndarray:
        return np.vectorize(LV_MAP[lv].get, otypes=[object])(arr)

    # ────────────────────────────────────────────────────────────────────────
    # 3) Core: vectorised 룰 매칭 → 4개 Tag buffer 생성 (S/In용)
    # ────────────────────────────────────────────────────────────────────────
    ship_np = df_asn[COL_SHIP_TO].astype(str).to_numpy()
    item_np = df_asn[COL_ITEM].to_numpy()
    loc_np  = df_asn[COL_LOCATION].to_numpy()
    pg_np   = df_asn[COL_PG].astype(str).to_numpy()
    asn_np  = df_asn[COL_SALES_PRODUCT_ASN].astype(str).str.upper().to_numpy()

    TAGS = ('GC', 'AP2', 'AP1', 'Local')
    tag_lvmat = {t: np.zeros_like(ship_np, dtype='int8') for t in TAGS}

    for lv in range(1, 7):  # Std1..Std6
        parent = parent_of(ship_np, lv)              # Ship → Std-lv 코드
        mask = parent != None
        if not mask.any():
            continue

        idx = np.flatnonzero(mask)
        rule_vec = np.array(
            [RULE.get((pg_np[i], parent[i]), (0, 0, 0, 0)) for i in idx],
            dtype='int8'
        )
        gc_lv, ap2_lv, ap1_lv, ap0_lv = rule_vec.T
        tag_lvmat['GC']   [idx] = np.where(gc_lv  != 0, gc_lv,  tag_lvmat['GC']   [idx])
        tag_lvmat['AP2']  [idx] = np.where(ap2_lv != 0, ap2_lv, tag_lvmat['AP2']  [idx])
        tag_lvmat['AP1']  [idx] = np.where(ap1_lv != 0, ap1_lv, tag_lvmat['AP1']  [idx])
        tag_lvmat['Local'][idx] = np.where(ap0_lv != 0, ap0_lv, tag_lvmat['Local'][idx])

    ship_lv_arr = np.fromiter((ship_level.get(s, 99) for s in ship_np), dtype='int8')

    def _build_sin(tag: str, qty_col: str) -> pd.DataFrame:
        lv_arr = tag_lvmat[tag]
        valid  = (lv_arr >= 2) & (lv_arr <= 7) & (lv_arr <= ship_lv_arr)
        if not valid.any():
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])

        tgt_ship = np.fromiter(
            (LV_MAP[l-1].get(s) if 2 <= l <= 7 else None
             for s, l in zip(ship_np[valid], lv_arr[valid])),
            dtype=object
        )
        ok = tgt_ship != None
        if not ok.any():
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])

        rows = {
            COL_SHIP_TO           : tgt_ship[ok],
            COL_ITEM              : item_np[valid][ok],
            COL_LOCATION          : loc_np [valid][ok],
            COL_SALES_PRODUCT_ASN : asn_np [valid][ok],
            qty_col               : np.ones(ok.sum(), dtype='int8')
        }
        df = pd.DataFrame(rows)

        # groupby (ShipTo, Item, Loc) : qty=sum→1, ASN=max('Y'>'N')
        df = (df.groupby([COL_SHIP_TO, COL_ITEM, COL_LOCATION], as_index=False, sort=False)
                .agg({qty_col: 'sum', COL_SALES_PRODUCT_ASN: 'max'})
                .assign(**{qty_col: 1}))

        # ASN='N' → measure NaN
        df[qty_col] = np.where(df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y'), 1, np.nan)

        # dtype 최적화
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')
        return df

    sin_gc    = _build_sin('GC',    COL_SIN_ASS_GC)
    sin_ap2   = _build_sin('AP2',   COL_SIN_ASS_AP2)
    sin_ap1   = _build_sin('AP1',   COL_SIN_ASS_AP1)
    sin_local = _build_sin('Local', COL_SIN_ASS_LOCAL)

    sin_dict: dict[str, pd.DataFrame] = {
        STR_DF_OUT_SIN_GC    : sin_gc,
        STR_DF_OUT_SIN_AP2   : sin_ap2,
        STR_DF_OUT_SIN_AP1   : sin_ap1,
        STR_DF_OUT_SIN_LOCAL : sin_local,
    }

    def _to_sout_base(df_in: pd.DataFrame, sin_col: str, sout_col: str) -> pd.DataFrame:
        if df_in.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                        COL_SALES_PRODUCT_ASN, sout_col])    
        # 보조 플래그: Y→1, 그 외→0
        df_tmp = df_in[[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN, sin_col]].copy()
        df_tmp['_asn_flag'] = (
            df_tmp[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')
        ).astype('int8')

        # ★ observed=True 로 '실제로 존재하는 그룹'만 집계 (카테고리 데카르트 곱 방지)
        df = (df_tmp.groupby([COL_SHIP_TO, COL_ITEM],
                            as_index=False, sort=False, observed=True)
                    .agg({sin_col: 'sum', '_asn_flag': 'max'}))

        # ASN 복원 및 수량 처리
        df[COL_SALES_PRODUCT_ASN] = np.where(df['_asn_flag'] == 1, 'Y', 'N')
        df.drop(columns=['_asn_flag'], inplace=True)

        df[sout_col] = 1
        df[COL_LOCATION] = '-'
        df[sout_col] = np.where(
            df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y'), 1, np.nan
        )

        # dtype 정리
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')

        return df[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN, sout_col]]  

    sout_base_gc    = _to_sout_base(sin_gc,    COL_SIN_ASS_GC,    COL_SOUT_ASS_GC)
    sout_base_ap2   = _to_sout_base(sin_ap2,   COL_SIN_ASS_AP2,   COL_SOUT_ASS_AP2)
    sout_base_ap1   = _to_sout_base(sin_ap1,   COL_SIN_ASS_AP1,   COL_SOUT_ASS_AP1)
    sout_base_local = _to_sout_base(sin_local, COL_SIN_ASS_LOCAL, COL_SOUT_ASS_LOCAL)

    # ────────────────────────────────────────────────────────────────────────
    # 5) S/Out – E-Store 상세(Location 유지) : df_asn 중 E-Store ShipTo만 재빌드
    # ────────────────────────────────────────────────────────────────────────
    est_set = set(df_in_Sales_Domain_Estore[COL_SHIP_TO].astype(str))
    if est_set:
        mask_es = df_asn[COL_SHIP_TO].astype(str).isin(est_set).to_numpy()
    else:
        mask_es = np.zeros(len(df_asn), dtype=bool)

    def _build_sout_est(tag: str, qty_col: str) -> pd.DataFrame:
        if not mask_es.any():
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])

        lv_arr = tag_lvmat[tag][mask_es]
        ship_v = ship_np[mask_es]; item_v = item_np[mask_es]
        loc_v  = loc_np [mask_es]; asn_v  = asn_np [mask_es]

        ship_lv_v = np.fromiter((ship_level.get(s, 99) for s in ship_v), dtype='int8')
        valid = (lv_arr >= 2) & (lv_arr <= 7) & (lv_arr <= ship_lv_v)
        if not valid.any():
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])

        tgt_ship = np.fromiter(
            (LV_MAP[l-1].get(s) if 2 <= l <= 7 else None
             for s, l in zip(ship_v[valid], lv_arr[valid])),
            dtype=object
        )
        ok = tgt_ship != None
        if not ok.any():
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])

        rows = {
            COL_SHIP_TO           : tgt_ship[ok],
            COL_ITEM              : item_v[valid][ok],
            COL_LOCATION          : loc_v [valid][ok],
            COL_SALES_PRODUCT_ASN : asn_v [valid][ok],
            qty_col               : np.ones(ok.sum(), dtype='int8')
        }
        df = pd.DataFrame(rows)
        df = (df.groupby([COL_SHIP_TO, COL_ITEM, COL_LOCATION], as_index=False, sort=False)
                .agg({qty_col: 'sum', COL_SALES_PRODUCT_ASN: 'max'})
                .assign(**{qty_col: 1}))
        df[qty_col] = np.where(df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y'), 1, np.nan)
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')
        return df

    sout_es_gc    = _build_sout_est('GC',    COL_SOUT_ASS_GC)
    sout_es_ap2   = _build_sout_est('AP2',   COL_SOUT_ASS_AP2)
    sout_es_ap1   = _build_sout_est('AP1',   COL_SOUT_ASS_AP1)
    sout_es_local = _build_sout_est('Local', COL_SOUT_ASS_LOCAL)

    # ────────────────────────────────────────────────────────────────────────
    # 6) S/Out 최종: 기본('-') + E-Store 상세 concat (중복 키 제거)
    # ────────────────────────────────────────────────────────────────────────
    def _concat_sout(base: pd.DataFrame, est: pd.DataFrame, qty_col: str) -> pd.DataFrame:
        if base.empty and est.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, qty_col])
        gcols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
        df = pd.concat([base, est], ignore_index=True)
        df = df.drop_duplicates(subset=gcols, keep='first')
        # 보수적으로 다시 ASN 기준 NaN 적용
        df[qty_col] = np.where(df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y'), 1, np.nan)
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')
        return df

    sout_gc    = _concat_sout(sout_base_gc,    sout_es_gc,    COL_SOUT_ASS_GC)
    sout_ap2   = _concat_sout(sout_base_ap2,   sout_es_ap2,   COL_SOUT_ASS_AP2)
    sout_ap1   = _concat_sout(sout_base_ap1,   sout_es_ap1,   COL_SOUT_ASS_AP1)
    sout_local = _concat_sout(sout_base_local, sout_es_local, COL_SOUT_ASS_LOCAL)

    sout_dict: dict[str, pd.DataFrame] = {
        STR_DF_OUT_SOUT_GC    : sout_gc,
        STR_DF_OUT_SOUT_AP2   : sout_ap2,
        STR_DF_OUT_SOUT_AP1   : sout_ap1,
        STR_DF_OUT_SOUT_LOCAL : sout_local,
    }

    # ────────────────────────────────────────────────────────────────────────
    # 7) 메모리 정리
    # ────────────────────────────────────────────────────────────────────────
    del (df_step01_03_asn_all, df_in_Forecast_Rule, df_in_Item_Master,
         df_in_Sales_Domain_Dimension, df_in_Sales_Domain_Estore,
         df_asn, df_rule, dim, dim_idx,
         ship_np, item_np, loc_np, pg_np, asn_np,
         sin_gc, sin_ap2, sin_ap1, sin_local,
         sout_base_gc, sout_base_ap2, sout_base_ap1, sout_base_local,
         sout_es_gc, sout_es_ap2, sout_es_ap1, sout_es_local)
    gc.collect()

    return sin_dict, sout_dict

########################################################################################################################
# Step 2-3 : S/In Assortment Measure (GC/AP2/AP1/Local) 비교
########################################################################################################################
@_decoration_
def fn_step02_03_compare_sin_assortments(
        df_in_sin_assort: pd.DataFrame,                # Step 2-1 결과: S/In Assortment (Input 8 전처리본)
        sin_dict: dict[str, pd.DataFrame]              # Step 2-2 결과: {STR_DF_OUT_SIN_XXX: df, ...}
) -> dict[str, pd.DataFrame]:
    """
    Step 2-3) S/In Assortment Measure (GC/AP2/AP1/Local) 비교
    ----------------------------------------------------------------
    • 입력
      - df_in_sin_assort : (ShipTo, Item, Location, S/In FCST(GI) Assortment_[GC|AP2|AP1|Local])
      - sin_dict         : Step2-2에서 생성된 4개 S/In dict
                           (각 DF는 [ShipTo, Item, Location, Sales Product ASN, Assortment_xx] 스키마)
    • 처리
      - 키(ShipTo, Item, Location) 기준으로 df_in_sin_assort에 **이미 값이 있는 행**은
        ASN='Y' 라면 삭제(중복 방지)
      - ASN='N' 인 행은 해당 Assortment 컬럼을 NaN으로 강제
    • 반환
      - 동일 키의 dict 4종 (GC/AP2/AP1/Local)
    """    
    # ────────────────────────────────────────────────────────────────────────
    # 0) 안전 체크 & 준비
    # ────────────────────────────────────────────────────────────────────────
    if df_in_sin_assort is None:
        df_in_sin_assort = pd.DataFrame()

    TAGS_INFO = [
        (STR_DF_OUT_SIN_GC,    COL_SIN_ASSORT_GC),
        (STR_DF_OUT_SIN_AP2,   COL_SIN_ASSORT_AP2),
        (STR_DF_OUT_SIN_AP1,   COL_SIN_ASSORT_AP1),
        (STR_DF_OUT_SIN_LOCAL, COL_SIN_ASSORT_LOCAL),
    ]

    out_dict: dict[str, pd.DataFrame] = {}

    # ────────────────────────────────────────────────────────────────────────
    # 1) 각 태그별로: 기존 df_in_sin_assort에서 "이미 값 있는 키" 뽑기
    # ────────────────────────────────────────────────────────────────────────
    exist_key_map: dict[str, set[tuple]] = {}
    for _, col_ass in TAGS_INFO:
        if (not df_in_sin_assort.empty) and (col_ass in df_in_sin_assort.columns):
            exist_keys = (
                df_in_sin_assort.loc[df_in_sin_assort[col_ass].notna(),
                                     [COL_SHIP_TO, COL_ITEM, COL_LOCATION]]
                              .drop_duplicates()
                              .to_records(index=False)
            )
            exist_key_map[col_ass] = set(map(tuple, exist_keys))
        else:
            exist_key_map[col_ass] = set()

    # ────────────────────────────────────────────────────────────────────────
    # 2) 필터 로직 (ASN='Y' & 키가 기존에 존재 → drop, ASN='N' → NaN 강제)
    # ────────────────────────────────────────────────────────────────────────
    def _filter_one(df_in: pd.DataFrame, ass_col: str) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, ass_col])

        # 스키마 방어
        need = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN, ass_col]
        miss = [c for c in need if c not in df_in.columns]
        if miss:
            raise KeyError(f"[Step 2-3] S/In DF에 필수 컬럼이 없습니다: {miss}")

        df = df_in[need].copy(deep=False)

        # ASN 정상화 (object→str→upper)
        asn_y = df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')

        # 기존 값 있는 키 집합
        keyset = exist_key_map.get(ass_col, set())
        if keyset:
            keys = list(zip(df[COL_SHIP_TO].astype(str),
                            df[COL_ITEM].astype(str),
                            df[COL_LOCATION].astype(str)))
            has_old = pd.Series([k in keyset for k in keys], index=df.index)

            # ASN='Y' 이고 기존 값 존재 → 제거
            drop_mask = asn_y & has_old
            if drop_mask.any():
                df = df.loc[~drop_mask].copy()

        # ASN='N' → 해당 Assortment 컬럼 NaN 강제
        asn_y = df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')  # 재평가(필터 후)
        df[ass_col] = np.where(asn_y, 1, np.nan)

        # dtype 정리 (메모리 절감)
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')

        return df

    # ────────────────────────────────────────────────────────────────────────
    # 3) 태그별 처리 & out_dict 구성
    # ────────────────────────────────────────────────────────────────────────
    for name, col_ass in TAGS_INFO:
        df_base = sin_dict.get(name, pd.DataFrame())
        out_dict[name] = _filter_one(df_base, col_ass)

    # ────────────────────────────────────────────────────────────────────────
    # 4) 메모리 정리
    # ────────────────────────────────────────────────────────────────────────
    del df_in_sin_assort, sin_dict
    gc.collect()

    return out_dict

########################################################################################################################
# Step 2-4 : S/Out Assortment Measure (GC/AP2/AP1/Local) 비교
########################################################################################################################
@_decoration_
def fn_step02_04_compare_sout_assortments(
        df_in_sout_assort: pd.DataFrame,            # Step 2-1 결과: S/Out Assortment 전처리본
        sout_dict: dict[str, pd.DataFrame]          # Step 2-2 결과: {STR_DF_OUT_SOUT_xxx: df, ...}
) -> dict[str, pd.DataFrame]:
    """
    Step 2-4) S/Out Assortment Measure (GC/AP2/AP1/Local) 비교
    ----------------------------------------------------------------
    • 입력
      - df_in_sout_assort : (ShipTo, Item, Location, S/Out FCST Assortment_[GC|AP2|AP1|Local])
      - sout_dict         : Step2-2에서 생성된 4개 S/Out dict
                            (각 DF는 [ShipTo, Item, Location, Sales Product ASN, Assortment_xx] 스키마)
    • 처리 (키 = ShipTo, Item, Location)
      - ASN='Y' 이고, df_in_sout_assort에 해당 Assortment 값이 **이미 존재**하면 해당 row 삭제
      - ASN='N' 인 행은 해당 Assortment 컬럼을 NaN 으로 강제
    • 반환
      - 동일 키의 dict 4종 (GC/AP2/AP1/Local)
    """    
    
    if df_in_sout_assort is None:
        df_in_sout_assort = pd.DataFrame()

    TAGS_INFO = [
        (STR_DF_OUT_SOUT_GC,    COL_SOUT_ASSORT_GC),
        (STR_DF_OUT_SOUT_AP2,   COL_SOUT_ASSORT_AP2),
        (STR_DF_OUT_SOUT_AP1,   COL_SOUT_ASSORT_AP1),
        (STR_DF_OUT_SOUT_LOCAL, COL_SOUT_ASSORT_LOCAL),
    ]

    out_dict: dict[str, pd.DataFrame] = {}

    # ────────────────────────────────────────────────────────────────────────
    # 1) 태그별로 "기존 값이 있는 키(ShipTo, Item, Loc)" 집합 준비
    # ────────────────────────────────────────────────────────────────────────
    exist_key_map: dict[str, set[tuple]] = {}
    for _, col_ass in TAGS_INFO:
        if (not df_in_sout_assort.empty) and (col_ass in df_in_sout_assort.columns):
            exist_keys = (
                df_in_sout_assort.loc[df_in_sout_assort[col_ass].notna(),
                                      [COL_SHIP_TO, COL_ITEM, COL_LOCATION]]
                              .drop_duplicates()
                              .to_records(index=False)
            )
            exist_key_map[col_ass] = set(map(tuple, exist_keys))
        else:
            exist_key_map[col_ass] = set()

    # ────────────────────────────────────────────────────────────────────────
    # 2) 필터 로직: ASN='Y' & 기존 값 존재 → drop / ASN='N' → NaN 강제
    # ────────────────────────────────────────────────────────────────────────
    def _filter_one(df_in: pd.DataFrame, ass_col: str) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION,
                                         COL_SALES_PRODUCT_ASN, ass_col])

        need = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN, ass_col]
        miss = [c for c in need if c not in df_in.columns]
        if miss:
            raise KeyError(f"[Step 2-4] S/Out DF에 필수 컬럼이 없습니다: {miss}")

        # 필요한 컬럼만 (얕은 복사) – 카테고리 dtype 유지
        df = df_in[need].copy(deep=False)

        # ASN='Y' 판단
        asn_y = df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')

        # 기존 값 존재하는 키셋
        keyset = exist_key_map.get(ass_col, set())
        if keyset:
            keys = list(zip(df[COL_SHIP_TO].astype(str),
                            df[COL_ITEM].astype(str),
                            df[COL_LOCATION].astype(str)))
            has_old = pd.Series([k in keyset for k in keys], index=df.index)

            # ASN='Y' & 기존값 존재 → drop
            drop_mask = asn_y & has_old
            if drop_mask.any():
                df = df.loc[~drop_mask].copy()

        # 남은 행들에 대해: ASN='N' → NaN, ASN='Y' → 1
        asn_y = df[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')  # 재계산(필터 후)
        df[ass_col] = np.where(asn_y, 1, np.nan)

        # 메모리 절감: 카테고리 캐스팅 (마지막에만)
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN):
            df[c] = df[c].astype('category')

        return df

    # ────────────────────────────────────────────────────────────────────────
    # 3) 태그별 처리 & 결과 dict 구성
    # ────────────────────────────────────────────────────────────────────────
    for df_name, col_ass in TAGS_INFO:
        base_df = sout_dict.get(df_name, pd.DataFrame())
        out_dict[df_name] = _filter_one(base_df, col_ass)

    # ────────────────────────────────────────────────────────────────────────
    # 4) 메모리 정리
    # ────────────────────────────────────────────────────────────────────────
    del df_in_sout_assort, sout_dict
    gc.collect()

    return out_dict


########################################################################################################################
# Step 2-5 : Assortment Measure Output 구성 (8개)
########################################################################################################################
@_decoration_
def fn_step02_05_format_assortment_outputs(
        sin_dict_in : dict[str, pd.DataFrame],   # ← Step 2-3 결과 dict (S/In: GC/AP2/AP1/Local)
        sout_dict_in: dict[str, pd.DataFrame],   # ← Step 2-4 결과 dict (S/Out: GC/AP2/AP1/Local)
        out_version : str                        # ← Version (예: 'CWV_DP')
) -> dict[str, pd.DataFrame]:
    """
    Step 2-5) Assortment Measure Output 구성
    ----------------------------------------------------------
    • Version.[Version Name] = out_version 추가 (하드코딩 금지)
    • Sales Product ASN 컬럼 제거
    • 8개 Output DF 반환 (o9 상위 호출 명세와 동일한 이름)

    반환 키:
      - Output_SIn_Assortment_GC / _AP2 / _AP1 / _Local
      - Output_SOut_Assortment_GC / _AP2 / _AP1 / _Local
    """

    # 0) 입력 딕셔너리 키 → (출력 키, 측정치 컬럼) 매핑
    SIN_MAP = {
        STR_DF_OUT_SIN_GC   : (OUT_SIN_ASSORTMENT_GC,    COL_SIN_ASSORT_GC),
        STR_DF_OUT_SIN_AP2  : (OUT_SIN_ASSORTMENT_AP2,   COL_SIN_ASSORT_AP2),
        STR_DF_OUT_SIN_AP1  : (OUT_SIN_ASSORTMENT_AP1,   COL_SIN_ASSORT_AP1),
        STR_DF_OUT_SIN_LOCAL: (OUT_SIN_ASSORTMENT_LOCAL, COL_SIN_ASSORT_LOCAL),
    }
    SOUT_MAP = {
        STR_DF_OUT_SOUT_GC   : (OUT_SOUT_ASSORTMENT_GC,    COL_SOUT_ASSORT_GC),
        STR_DF_OUT_SOUT_AP2  : (OUT_SOUT_ASSORTMENT_AP2,   COL_SOUT_ASSORT_AP2),
        STR_DF_OUT_SOUT_AP1  : (OUT_SOUT_ASSORTMENT_AP1,   COL_SOUT_ASSORT_AP1),
        STR_DF_OUT_SOUT_LOCAL: (OUT_SOUT_ASSORTMENT_LOCAL, COL_SOUT_ASSORT_LOCAL),
    }

    # 1) 공통 포맷터
    def _format(df_in: pd.DataFrame, meas_col: str) -> pd.DataFrame:
        """
        최종 스키마: [Version, ShipTo, Item, Location, meas_col]
        - ASN 컬럼 제거
        - Version 주입 및 컬럼 순서 고정
        - dtype: dims → category, measure → float32
        """
        col_order = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, meas_col]

        if df_in is None or df_in.empty:
            df_out = pd.DataFrame(columns=col_order)
            for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION):
                df_out[c] = df_out[c].astype('category')
            df_out[meas_col] = df_out[meas_col].astype('float32')
            return df_out

        need = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, meas_col]
        miss = [c for c in need if c not in df_in.columns]
        if miss:
            raise KeyError(f"[Step 2-5] 입력 DF에 필수 컬럼 누락: {miss}")

        # 얕은 복사 + ASN 제거
        use_cols = need + ([COL_SALES_PRODUCT_ASN] if COL_SALES_PRODUCT_ASN in df_in.columns else [])
        df = df_in[use_cols].copy(deep=False)
        if COL_SALES_PRODUCT_ASN in df.columns:
            df.drop(columns=[COL_SALES_PRODUCT_ASN], inplace=True)

        # Version 추가 + 순서 고정
        df[COL_VERSION] = out_version
        df = df[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, meas_col]]

        # dtype 정리
        for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION):
            # 이미 category 라면 유지, 아니라면 변환
            if df[c].dtype.name != 'category':
                df[c] = df[c].astype('category')
        # measure → float32 (1 또는 NaN)
        df[meas_col] = pd.to_numeric(df[meas_col], errors='coerce').astype('float32')

        return df

    # 2) S/In 변환
    out_dict: dict[str, pd.DataFrame] = {}
    for in_key, (out_key, meas_col) in SIN_MAP.items():
        df_src = sin_dict_in.get(in_key, pd.DataFrame())
        out_dict[out_key] = _format(df_src, meas_col)

    # 3) S/Out 변환
    for in_key, (out_key, meas_col) in SOUT_MAP.items():
        df_src = sout_dict_in.get(in_key, pd.DataFrame())
        out_dict[out_key] = _format(df_src, meas_col)

    # 4) 메모리 정리
    del sin_dict_in, sout_dict_in
    gc.collect()

    return out_dict

########################################################################################################################
# Step 03 : DSR 구성 (Location 제거)
########################################################################################################################
@_decoration_
def fn_step03_build_dsr(
        sin_compared_dict : dict[str, pd.DataFrame],   # ← Step 2-3 결과 dict  (S/In 비교 후)
        sout_compared_dict: dict[str, pd.DataFrame],   # ← Step 2-4 결과 dict  (S/Out 비교 후)
        out_version       : str                        # ← Version (예: 'CWV_DP')
) -> dict[str, pd.DataFrame]:
    """
    Step 3) DSR 구성 (GC/AP2/AP1/Local, Location 제거)
    ----------------------------------------------------------
    • 입력
      - sin_compared_dict  : { OUT_SIn_Assortment_* : df } (Step 2-3)
      - sout_compared_dict : { OUT_SOut_Assortment_*: df } (Step 2-4)
    • 처리
      - Location 삭제, (Ship To, Item) 기준 groupby
      - Assortment 컬럼 집계: (NaN, 1) → 1   (max 집계)
      - 컬럼명: Assortment_*  →  DSR_*
      - Version 컬럼 추가, 차원형(category) 캐스팅
    • 반환
      - 8개 DSR Output dict
        { OUT_SIn_DSR_GC, OUT_SIn_DSR_AP2, OUT_SIn_DSR_AP1, OUT_SIn_DSR_LOCAL,
          OUT_SOut_DSR_GC, OUT_SOut_DSR_AP2, OUT_SOut_DSR_AP1, OUT_SOut_DSR_LOCAL }
    """    
    # ───────────────────────────────────────────────────────────────────────
    # 0) 내부 헬퍼: Location 제거 후 (ShipTo,Item) 집계 → DSR
    # ───────────────────────────────────────────────────────────────────────
    def _to_dsr(df_in: pd.DataFrame, assort_col: str, dsr_col: str) -> pd.DataFrame:
        """
        df_in: [ShipTo, Item, (Location), assort_col, ...]
        반환: [Version, ShipTo, Item, dsr_col]
        """
        # 빈 DF 방어
        if df_in is None or df_in.empty:
            return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, dsr_col])

        # 필요한 컬럼만 얕은 복사
        use_cols = [COL_SHIP_TO, COL_ITEM, assort_col]
        df_tmp = df_in.loc[:, [c for c in use_cols if c in df_in.columns]].copy(deep=False)

        # 집계 대상이 없으면 빈 껍데기 반환
        if assort_col not in df_tmp.columns:
            return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, dsr_col])

        # Assortment 값: (1 또는 NaN) → float 로 강제 후 max 집계
        df_tmp[assort_col] = pd.to_numeric(df_tmp[assort_col], errors='coerce')

        df_out = (df_tmp.groupby([COL_SHIP_TO, COL_ITEM], as_index=False, sort=False, observed=True)
                        .agg({assort_col: 'max'}))   # (NaN,1)→1

        # 컬럼명 변경: Assortment_* → DSR_*
        df_out.rename(columns={assort_col: dsr_col}, inplace=True)

        # Version 삽입 + 순서 정리
        df_out.insert(0, COL_VERSION, out_version)

        # category 캐스팅 (메모리 절감)
        for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM):
            df_out[c] = df_out[c].astype('category')

        return df_out[[COL_VERSION, COL_SHIP_TO, COL_ITEM, dsr_col]]

    # ───────────────────────────────────────────────────────────────────────
    # 1) S/In DSR (GC/AP2/AP1/Local)
    # ───────────────────────────────────────────────────────────────────────
    sin_gc    = sin_compared_dict.get(STR_DF_OUT_SIN_GC,    pd.DataFrame())
    sin_ap2   = sin_compared_dict.get(STR_DF_OUT_SIN_AP2,   pd.DataFrame())
    sin_ap1   = sin_compared_dict.get(STR_DF_OUT_SIN_AP1,   pd.DataFrame())
    sin_local = sin_compared_dict.get(STR_DF_OUT_SIN_LOCAL, pd.DataFrame())

    df_out_sin_dsr_gc    = _to_dsr(sin_gc,    COL_SIN_ASSORT_GC,    COL_SIN_DSR_GC)
    df_out_sin_dsr_ap2   = _to_dsr(sin_ap2,   COL_SIN_ASSORT_AP2,   COL_SIN_DSR_AP2)
    df_out_sin_dsr_ap1   = _to_dsr(sin_ap1,   COL_SIN_ASSORT_AP1,   COL_SIN_DSR_AP1)
    df_out_sin_dsr_local = _to_dsr(sin_local, COL_SIN_ASSORT_LOCAL, COL_SIN_DSR_LOCAL)

    # ───────────────────────────────────────────────────────────────────────
    # 2) S/Out DSR (GC/AP2/AP1/Local)
    # ───────────────────────────────────────────────────────────────────────
    sout_gc    = sout_compared_dict.get(STR_DF_OUT_SOUT_GC,    pd.DataFrame())
    sout_ap2   = sout_compared_dict.get(STR_DF_OUT_SOUT_AP2,   pd.DataFrame())
    sout_ap1   = sout_compared_dict.get(STR_DF_OUT_SOUT_AP1,   pd.DataFrame())
    sout_local = sout_compared_dict.get(STR_DF_OUT_SOUT_LOCAL, pd.DataFrame())

    df_out_sout_dsr_gc    = _to_dsr(sout_gc,    COL_SOUT_ASSORT_GC,    COL_SOUT_DSR_GC)
    df_out_sout_dsr_ap2   = _to_dsr(sout_ap2,   COL_SOUT_ASSORT_AP2,   COL_SOUT_DSR_AP2)
    df_out_sout_dsr_ap1   = _to_dsr(sout_ap1,   COL_SOUT_ASSORT_AP1,   COL_SOUT_DSR_AP1)
    df_out_sout_dsr_local = _to_dsr(sout_local, COL_SOUT_ASSORT_LOCAL, COL_SOUT_DSR_LOCAL)

    # ───────────────────────────────────────────────────────────────────────
    # 3) 반환 dict (Output 키는 기존 상수 사용)
    # ───────────────────────────────────────────────────────────────────────
    out_dict: dict[str, pd.DataFrame] = {
        OUT_SIN_DSR_GC   : df_out_sin_dsr_gc,
        OUT_SIN_DSR_AP2  : df_out_sin_dsr_ap2,
        OUT_SIN_DSR_AP1  : df_out_sin_dsr_ap1,
        OUT_SIN_DSR_LOCAL: df_out_sin_dsr_local,

        OUT_SOUT_DSR_GC   : df_out_sout_dsr_gc,
        OUT_SOUT_DSR_AP2  : df_out_sout_dsr_ap2,
        OUT_SOUT_DSR_AP1  : df_out_sout_dsr_ap1,
        OUT_SOUT_DSR_LOCAL: df_out_sout_dsr_local,
    }

    # 메모리 정리
    del (sin_gc, sin_ap2, sin_ap1, sin_local,
         sout_gc, sout_ap2, sout_ap1, sout_local)
    gc.collect()

    return out_dict

########################################################################################################################
# Step 04 : FCST 0 값 데이터 생성 (S/In · S/Out · New Model · Flooring · BO)
########################################################################################################################
@_decoration_
def fn_step04_build_zero_fcst(
        # ── inputs ───────────────────────────────────────────────────────────────────────────────────────────────────
        out_version : str,                             # 예) 'CWV_DP'
        df_time     : pd.DataFrame,                    # (Input 2)  Time: [Time.[Partial Week]], [Time.[Week]]
        # Step 2-5 Output (Assortment 결과; Version 포함, ASN 컬럼 제거된 상태)
        df_out_sin_gc   : pd.DataFrame,                # Output_SIn_Assortment_GC
        df_out_sin_ap2  : pd.DataFrame,                # Output_SIn_Assortment_AP2
        df_out_sin_ap1  : pd.DataFrame,                # Output_SIn_Assortment_AP1
        df_out_sin_local: pd.DataFrame,                # Output_SIn_Assortment_Local
        df_out_sout_gc   : pd.DataFrame,               # Output_SOut_Assortment_GC
        df_out_sout_ap2  : pd.DataFrame,               # Output_SOut_Assortment_AP2
        df_out_sout_ap1  : pd.DataFrame,               # Output_SOut_Assortment_AP1
        df_out_sout_local: pd.DataFrame,               # Output_SOut_Assortment_Local
        # ASN 참조 (ASN='Y' → 0 유지, 'N' → NaN)
        df_step01_03_asn_all         : pd.DataFrame,   # Step 1-3 결과: [ShipTo, Item, Location, Sales Product ASN (Y/N)]
        df_step01_05_asn_delta_noloc : pd.DataFrame,   # Step 1-5 결과: [ShipTo, Item, Sales Product ASN (Y/N)]
        # Item Master
        df_in_item_master            : pd.DataFrame,   # [Item.[Item], Item.[Item GBM]]  (New Model 용; GBM='VD')
        df_in_item_master_led_signage: pd.DataFrame    # [Item.[Item]]  (BO FCST 용)
) -> dict[str, pd.DataFrame]:
    """
    Step 4) FCST 0 값 데이터 생성
    ───────────────────────────────────────────────────────────────────────────
    • S/In FCST(GI):  AP1 / AP2 / GC / Local  → Time.[Partial Week] 확장
    • S/Out FCST   :  AP1 / AP2 / GC / Local  → Time.[Partial Week] 확장
      - Location='-' 행은 (ShipTo, Item)의 No-Location ASN으로 판단
    • New Model (VD 전용) : AP1 조합에서 Item GBM='VD' 만 추출하여 0 생성
    • Flooring FCST : Step 1-5 (No-Location) 조합 × Time.[Week] 0 생성
    • BO FCST      : Step 1-3 ASN 중 LED_SIGNAGE Item만 Partial Week 확장, Virtual/BO ID='-'
    • ASN='N' 은 값 NaN, ASN='Y' 는 값 0
    • 모든 차원 컬럼은 category 로 캐스팅
    • 반환 : 11개 Output DataFrame dict
    """    
    # ────────────────────────────────────────────────────────────────────────
    # 0) 준비: 시간 차원 (Partial Week / Week)
    # ────────────────────────────────────────────────────────────────────────
    pw = (df_time[[COL_PWEEK]].dropna().drop_duplicates().rename(columns={COL_PWEEK: COL_PWEEK}))
    wk = (df_time[[COL_WEEK ]].dropna().drop_duplicates().rename(columns={COL_WEEK : COL_WEEK }))
    if not pw.empty: pw[COL_PWEEK] = pw[COL_PWEEK].astype('category')
    if not wk.empty: wk[COL_WEEK ] = wk[COL_WEEK ].astype('category')

    # ────────────────────────────────────────────────────────────────────────
    # 1) ASN 맵 (Loc / No-Location)
    #    - Loc  : (ShipTo,Item,Location) → 1/0
    #    - NoLoc: (ShipTo,Item)          → 1/0 (max 집계)
    # ────────────────────────────────────────────────────────────────────────
    def _yn_to_flag(s: pd.Series) -> pd.Series:
        return (s.astype(str).str.upper().eq('Y')).astype('int8')

    # 1-1) Loc ASN
    if df_step01_03_asn_all.empty:
        asn_loc = pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, '_asn_flag_loc'])
    else:
        asn_loc = (df_step01_03_asn_all[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]]
                      .copy(deep=False))
        asn_loc['_asn_flag_loc'] = _yn_to_flag(asn_loc[COL_SALES_PRODUCT_ASN])
        asn_loc.drop(columns=[COL_SALES_PRODUCT_ASN], inplace=True)
        asn_loc[COL_SHIP_TO]  = asn_loc[COL_SHIP_TO].astype('category')
        asn_loc[COL_ITEM]     = asn_loc[COL_ITEM].astype('category')
        asn_loc[COL_LOCATION] = asn_loc[COL_LOCATION].astype('category')

    # 1-2) No-Location ASN (Step 1-3 전체기준 max) — S/Out 기본 '-' 판정 보조용
    if df_step01_03_asn_all.empty:
        asn_noloc_all = pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, '_asn_flag_noloc'])
    else:
        _tmp = df_step01_03_asn_all[[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN]].copy(deep=False)
        _tmp['_flag'] = _yn_to_flag(_tmp[COL_SALES_PRODUCT_ASN])
        asn_noloc_all = (_tmp.groupby([COL_SHIP_TO, COL_ITEM], as_index=False, sort=False, observed=True)
                              .agg({'_flag': 'max'})
                              .rename(columns={'_flag': '_asn_flag_noloc'}))

        for c in (COL_SHIP_TO, COL_ITEM):
            asn_noloc_all[c] = asn_noloc_all[c].astype('category')
        del _tmp

    # 1-3) No-Location ASN (Step 1-5 제공) — Flooring 전용
    if df_step01_05_asn_delta_noloc.empty:
        asn_noloc_15 = pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, '_asn_flag'])
    else:
        asn_noloc_15 = df_step01_05_asn_delta_noloc[[COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN]].copy(deep=False)
        asn_noloc_15['_asn_flag'] = _yn_to_flag(asn_noloc_15[COL_SALES_PRODUCT_ASN])
        asn_noloc_15.drop(columns=[COL_SALES_PRODUCT_ASN], inplace=True)
        for c in (COL_SHIP_TO, COL_ITEM):
            asn_noloc_15[c] = asn_noloc_15[c].astype('category')

    # ────────────────────────────────────────────────────────────────────────
    # 2) 공통 헬퍼: (조합 DF) × (시간 DF) → 0/NaN 채우기
    #    asn_mode='loc'|'noloc'|'both'
    #      - 'loc'   : (ShipTo,Item,Location)로 ASN 판단
    #      - 'noloc' : (ShipTo,Item)으로 ASN 판단
    #      - 'both'  : loc 우선, 없으면 noloc로 대체
    # ────────────────────────────────────────────────────────────────────────
    def _expand_with_time(df_combo: pd.DataFrame,
                          time_df: pd.DataFrame,
                          time_col: str,
                          meas_col: str,
                          *,
                          asn_mode: str) -> pd.DataFrame:
        if df_combo.empty or time_df.empty:
            return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, time_col, meas_col])

        base = df_combo[[COL_SHIP_TO, COL_ITEM, COL_LOCATION]].drop_duplicates().copy(deep=False)
        # category 보장
        for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION):
            base[c] = base[c].astype('category')

        # ASN 결합
        if asn_mode == 'loc':
            base = base.merge(asn_loc, how='left', on=[COL_SHIP_TO, COL_ITEM, COL_LOCATION])
            base['_asn_flag'] = base['_asn_flag_loc']
            base.drop(columns=['_asn_flag_loc'], inplace=True)
        elif asn_mode == 'noloc':
            base = base.merge(asn_noloc_all, how='left', on=[COL_SHIP_TO, COL_ITEM])
            base['_asn_flag'] = base['_asn_flag_noloc']
            base.drop(columns=['_asn_flag_noloc'], inplace=True)
        else:  # both
            base = (base.merge(asn_loc,       how='left', on=[COL_SHIP_TO, COL_ITEM, COL_LOCATION])
                        .merge(asn_noloc_all, how='left', on=[COL_SHIP_TO, COL_ITEM]))
            base['_asn_flag'] = np.where(base['_asn_flag_loc'].notna(),
                                         base['_asn_flag_loc'],
                                         base['_asn_flag_noloc'])
            base.drop(columns=['_asn_flag_loc', '_asn_flag_noloc'], inplace=True)

        # 시간 카테고리와 카티전 조인 (key=1)
        base['_k'] = 1
        tdf = time_df.copy(deep=False)
        tdf['_k'] = 1
        out = (base.merge(tdf, on='_k', how='left')
                    .drop(columns=['_k']))

        # 값 채우기: ASN=1 → 0, 그 외 NaN
        out[meas_col] = np.where(out['_asn_flag'] == 1, 0.0, np.nan).astype('float32')
        out.drop(columns=['_asn_flag'], inplace=True)

        # Version 삽입 + category 캐스팅
        out.insert(0, COL_VERSION, out_version)
        for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, time_col):
            out[c] = out[c].astype('category')

        # 컬럼 순서
        return out[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, time_col, meas_col]]

    # ────────────────────────────────────────────────────────────────────────
    # 3) 입력 Assortment에서 (ShipTo,Item,Loc) 조합만 추출
    # ────────────────────────────────────────────────────────────────────────
    def _strip_combo(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty:
            return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION])
        cols = [c for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION) if c in df_in.columns]
        df = df_in[cols].drop_duplicates().copy(deep=False)
        for c in cols:
            df[c] = df[c].astype('category')
        # Location 없으면 '-' 부여 (안전)
        if COL_LOCATION not in df.columns:
            df[COL_LOCATION] = '-'
            df[COL_LOCATION] = df[COL_LOCATION].astype('category')
        return df[[COL_SHIP_TO, COL_ITEM, COL_LOCATION]]

    sin_combo_gc    = _strip_combo(df_out_sin_gc)
    sin_combo_ap2   = _strip_combo(df_out_sin_ap2)
    sin_combo_ap1   = _strip_combo(df_out_sin_ap1)
    sin_combo_local = _strip_combo(df_out_sin_local)

    sout_combo_gc    = _strip_combo(df_out_sout_gc)
    sout_combo_ap2   = _strip_combo(df_out_sout_ap2)
    sout_combo_ap1   = _strip_combo(df_out_sout_ap1)
    sout_combo_local = _strip_combo(df_out_sout_local)

    # ────────────────────────────────────────────────────────────────────────
    # 4) S/In FCST(GI) 0 생성  (ASN 판단: loc)
    # ────────────────────────────────────────────────────────────────────────
    df_out_SI_AP1   = _expand_with_time(sin_combo_ap1,   pw, COL_PWEEK, COL_SIN_FCST_AP1,   asn_mode='loc')
    df_out_SI_AP2   = _expand_with_time(sin_combo_ap2,   pw, COL_PWEEK, COL_SIN_FCST_AP2,   asn_mode='loc')
    df_out_SI_GC    = _expand_with_time(sin_combo_gc,    pw, COL_PWEEK, COL_SIN_FCST_GC,    asn_mode='loc')
    df_out_SI_LOCAL = _expand_with_time(sin_combo_local, pw, COL_PWEEK, COL_SIN_FCST_LOCAL, asn_mode='loc')

    # ────────────────────────────────────────────────────────────────────────
    # 5) S/Out FCST 0 생성  (ASN 판단: Location='-'는 noloc, 그 외 loc → both 로 일괄 처리)
    # ────────────────────────────────────────────────────────────────────────
    df_out_SO_AP1   = _expand_with_time(sout_combo_ap1,   pw, COL_PWEEK, COL_SOUT_FCST_AP1,   asn_mode='both')
    df_out_SO_AP2   = _expand_with_time(sout_combo_ap2,   pw, COL_PWEEK, COL_SOUT_FCST_AP2,   asn_mode='both')
    df_out_SO_GC    = _expand_with_time(sout_combo_gc,    pw, COL_PWEEK, COL_SOUT_FCST_GC,    asn_mode='both')
    df_out_SO_LOCAL = _expand_with_time(sout_combo_local, pw, COL_PWEEK, COL_SOUT_FCST_LOCAL, asn_mode='both')

    # ────────────────────────────────────────────────────────────────────────
    # 6) New Model (VD 전용) – AP1 조합에서 Item GBM='VD'만 확장
    #    • ASN 판단: loc
    # ────────────────────────────────────────────────────────────────────────
    if df_in_item_master.empty or df_out_sin_ap1.empty:
        df_out_SI_NewModel = pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_FCST_NEW_MODE])
    else:
        vd_items = set(df_in_item_master.loc[df_in_item_master[COL_ITEM_GBM].astype(str).str.upper() == 'VD', COL_ITEM].astype(str))
        nm_combo = sin_combo_ap1[sin_combo_ap1[COL_ITEM].astype(str).isin(vd_items)]
        df_out_SI_NewModel = _expand_with_time(nm_combo, pw, COL_PWEEK, COL_SIN_FCST_NEW_MODE, asn_mode='loc')

    # ────────────────────────────────────────────────────────────────────────
    # 7) Flooring FCST 0 생성 – Step 1-5 No-Location 기준 × Week
    # ────────────────────────────────────────────────────────────────────────
    if asn_noloc_15.empty or wk.empty:
        df_out_Flooring = pd.DataFrame(
            columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_WEEK, COL_FLOORING_FCST]
        )
    else:
        base = asn_noloc_15.copy(deep=False)
        for c in (COL_SHIP_TO, COL_ITEM):
            base[c] = base[c].astype('category')    
        
        base['_k'] = 1
        twk = wk.copy(deep=False)
        twk['_k'] = 1

        # ① 카티전 조인 (ShipTo×Item) × Week
        df_out_Flooring = base.merge(twk, on='_k', how='left').drop(columns=['_k'])

        # ② ASN Flag 기준으로 0 / NaN 지정  (※ merge 결과의 _asn_flag 사용!)
        df_out_Flooring[COL_FLOORING_FCST] = np.where(
            df_out_Flooring['_asn_flag'].to_numpy() == 1, 0.0, np.nan
        )
        # ③ 불필요 컬럼 제거 + Version, 순서 정리
        df_out_Flooring.drop(columns=['_asn_flag'], inplace=True)
        df_out_Flooring[COL_VERSION] = out_version
        df_out_Flooring = df_out_Flooring[
            [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_WEEK, COL_FLOORING_FCST]
        ]

        # ④ dtype 정리
        df_out_Flooring[[COL_SHIP_TO, COL_ITEM]] = (
            df_out_Flooring[[COL_SHIP_TO, COL_ITEM]].astype('category')
        )


    # ────────────────────────────────────────────────────────────────────────
    # 8) BO FCST 0 생성 – Step 1-3 ASN 중 LED_SIGNAGE Item만, Virtual/BO ID='-'
    #     (Python 3.10.5 호환 / 길이 불일치 ValueError 방지)
    # ────────────────────────────────────────────────────────────────────────
    if df_in_item_master_led_signage.empty or df_step01_03_asn_all.empty or pw.empty:
        df_out_BO = pd.DataFrame(columns=[
            COL_VERSION, COL_ITEM, COL_SHIP_TO, COL_LOCATION,
            COL_VIRTUAL_BO_ID, COL_BO_ID, COL_PWEEK, COL_BO_FCST
        ])
    else:
        sign_items = set(df_in_item_master_led_signage[COL_ITEM].astype(str))    
        
        # 대상 행 추출
        bo_base = df_step01_03_asn_all[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SALES_PRODUCT_ASN]].copy(deep=False)
        bo_base = bo_base[bo_base[COL_ITEM].astype(str).isin(sign_items)]

        if bo_base.empty:
            df_out_BO = pd.DataFrame(columns=[
                COL_VERSION, COL_ITEM, COL_SHIP_TO, COL_LOCATION,
                COL_VIRTUAL_BO_ID, COL_BO_ID, COL_PWEEK, COL_BO_FCST
            ])
        else:
            # Y/N → flag
            bo_base['_asn_flag'] = _yn_to_flag(bo_base[COL_SALES_PRODUCT_ASN])
            bo_base.drop(columns=[COL_SALES_PRODUCT_ASN], inplace=True)

            # 카티전 조인 (항상 _k 방식 사용)
            L = bo_base.copy(deep=False)
            R = (pw[[COL_PWEEK]].dropna().drop_duplicates().reset_index(drop=True).copy(deep=False))
            L['_k'] = 1
            R['_k'] = 1

            df_out_BO = L.merge(R, on='_k', how='left').drop(columns=['_k'])

            # 길이 불일치 방지: df_out_BO 길이에 맞춰 계산
            df_out_BO[COL_VIRTUAL_BO_ID] = '-'
            df_out_BO[COL_BO_ID]         = '-'
            df_out_BO[COL_BO_FCST]       = np.where(df_out_BO['_asn_flag'].to_numpy() == 1, 0.0, np.nan)
            df_out_BO.drop(columns=['_asn_flag'], inplace=True)

            # Version & dtype & 컬럼 순서
            df_out_BO.insert(0, COL_VERSION, out_version)
            for c in (COL_VERSION, COL_ITEM, COL_SHIP_TO, COL_LOCATION, COL_VIRTUAL_BO_ID, COL_BO_ID, COL_PWEEK):
                df_out_BO[c] = df_out_BO[c].astype('category')

            df_out_BO = df_out_BO[[COL_VERSION, COL_ITEM, COL_SHIP_TO, COL_LOCATION,
                                COL_VIRTUAL_BO_ID, COL_BO_ID, COL_PWEEK, COL_BO_FCST]]    

    # ────────────────────────────────────────────────────────────────────────
    # 9) 반환 dict (Output 4-1 … 4-11)
    # ────────────────────────────────────────────────────────────────────────
    out_dict: dict[str, pd.DataFrame] = {
        DF_OUT_SIN_FCST_AP1  : df_out_SI_AP1,
        DF_OUT_SIN_FCST_AP2  : df_out_SI_AP2,
        DF_OUT_SIN_FCST_GC   : df_out_SI_GC,
        DF_OUT_SIN_FCST_LOCAL: df_out_SI_LOCAL,

        DF_OUT_SOUT_FCST_AP1  : df_out_SO_AP1,
        DF_OUT_SOUT_FCST_AP2  : df_out_SO_AP2,
        DF_OUT_SOUT_FCST_GC   : df_out_SO_GC,
        DF_OUT_SOUT_FCST_LOCAL: df_out_SO_LOCAL,

        DF_OUT_SIN_FCST_NEW_MODEL: df_out_SI_NewModel,
        DF_OUT_FLOORING_FCST     : df_out_Flooring,
        DF_OUT_BO_FCST           : df_out_BO,
    }

    # ────────────────────────────────────────────────────────────────────────
    # 10) 메모리 정리
    # ────────────────────────────────────────────────────────────────────────
    del (pw, wk, asn_loc, asn_noloc_all, asn_noloc_15,
         sin_combo_gc, sin_combo_ap2, sin_combo_ap1, sin_combo_local,
         sout_combo_gc, sout_combo_ap2, sout_combo_ap1, sout_combo_local)
    gc.collect()

    return out_dict


##############################################################################################################
# Step 5) Estimated Price Local Data 생성
##############################################################################################################
@_decoration_
def fn_step05_build_estimated_price_local(
    df_step01_05_asn_delta_no_loc : pd.DataFrame,   # Step 1-5 결과 (No-Location)
    df_in_Time                    : pd.DataFrame,   # Time.[Partial Week], (없으면) Time.[Month] 파생
    df_in_Estimated_Price        : pd.DataFrame,   # EP (Modify/Local) — Ship-To는 Std5 수준
    df_in_Action_Plan_Price      : pd.DataFrame,   # AP Price (USD) by ShipTo, Item, Month
    df_in_Exchange_Rate_Local    : pd.DataFrame,   # FX by Std3, PartialWeek
    df_in_Sales_Domain_Dimension : pd.DataFrame,   # ShipTo → Std3, Std5
    out_version                  : str,            # ex) 'CWV_DP'
    **kwargs
) -> pd.DataFrame:
    """
    Step 5) Estimated Price_Local 생성
      • grid(ShipTo×Item×PartialWeek)를 만든 뒤,
        - ShipTo를 Std5로 승격해 EP와 매칭
        - per-row 규칙: (Modify>0 & EP_Local notnull) → EP_Local
                        else → AP_USD(ShipTo,Item,Month) × FX(Std3,PartialWeek)
      • 최종적으로 Estimated Price_Local==NaN 인 행 제거
    """
    # ───────────────────────────────────────────────────────────────────
    # 0) 입력 컬럼 점검 + Time.Month 보강(fallback)
    # ───────────────────────────────────────────────────────────────────
    req_asn = [COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN]
    req_tim = [COL_PWEEK]
    req_dim = [COL_SHIP_TO, COL_STD3, COL_STD5]
    req_ep  = [COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_EST_PRICE_MODIFY_LOCAL, COL_EST_PRICE_LOCAL]
    req_ap  = [COL_SHIP_TO, COL_ITEM, COL_MONTH, COL_AP_PRICE_USD]
    req_fx  = [COL_STD3, COL_PWEEK, COL_EXCHANGE_RATE_LOCAL]    
    
    def _need(df, cols, name):
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise KeyError(f"[Step5] {name} missing columns: {miss}")

    _need(df_step01_05_asn_delta_no_loc, req_asn, 'df_step01_05_asn_delta_no_loc')
    _need(df_in_Time, req_tim, 'df_in_Time')
    _need(df_in_Sales_Domain_Dimension, req_dim, 'df_in_Sales_Domain_Dimension')
    _need(df_in_Estimated_Price, req_ep, 'df_in_Estimated_Price')
    _need(df_in_Action_Plan_Price, req_ap, 'df_in_Action_Plan_Price')
    _need(df_in_Exchange_Rate_Local, req_fx, 'df_in_Exchange_Rate_Local')

    # Time.Month 없으면 PartialWeek[:6] 로 파생
    if COL_MONTH not in df_in_Time.columns:
        df_time = df_in_Time.copy(deep=False)
        df_time[COL_MONTH] = df_time[COL_PWEEK].astype(str).str[:6].astype('category')
    else:
        df_time = df_in_Time[[COL_PWEEK, COL_MONTH]].copy(deep=False)

    # ───────────────────────────────────────────────────────────────────
    # 1) grid 만들기: (ShipTo×Item) from ASN NOLoc, ASN=='Y'만 + PartialWeek 전체
    # ───────────────────────────────────────────────────────────────────
    asn_y = df_step01_05_asn_delta_no_loc[df_step01_05_asn_delta_no_loc[COL_SALES_PRODUCT_ASN]
                                          .astype(str).str.upper().eq('Y')][[COL_SHIP_TO, COL_ITEM]].drop_duplicates()

    if asn_y.empty or df_time.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_EST_PRICE_LOCAL])

    asn_y = asn_y.copy(deep=False)
    asn_y['_k'] = 1
    t = df_time.copy(deep=False)
    t['_k'] = 1
    grid = (asn_y.merge(t, on='_k', how='left')
                 .drop(columns=['_k'])
           )  # cols: ShipTo, Item, PartialWeek, Month

    # ───────────────────────────────────────────────────────────────────
    # 2) ShipTo 승격(Std5) & Std3 매핑
    #    ※ EP의 ShipTo가 Std5 수준이므로 grid를 Std5로 변환 후 EP와 조인
    # ───────────────────────────────────────────────────────────────────
    dim_idx   = df_in_Sales_Domain_Dimension.set_index(COL_SHIP_TO)
    std5_map  = dim_idx[COL_STD5].astype(str).to_dict()
    std3_map  = dim_idx[COL_STD3].astype(str).to_dict()

    grid['_ship_orig'] = grid[COL_SHIP_TO].astype(str)
    grid[COL_SHIP_TO]  = grid['_ship_orig'].map(std5_map)
    # Std5 매핑 실패 시 원본 유지
    mask_na = grid[COL_SHIP_TO].isna() | (grid[COL_SHIP_TO].astype(str) == '')
    grid.loc[mask_na, COL_SHIP_TO] = grid.loc[mask_na, '_ship_orig']
    grid.drop(columns=['_ship_orig'], inplace=True)

    # Std3는 승격된 ShipTo 기준으로 재계산
    grid[COL_STD3] = grid[COL_SHIP_TO].astype(str).map(std3_map).astype('category')

    # ───────────────────────────────────────────────────────────────────
    # 3) EP(left join, per-row 결정)
    #    키: (ShipTo[Std5], Item, PartialWeek)
    # ───────────────────────────────────────────────────────────────────
    ep = df_in_Estimated_Price[[COL_SHIP_TO, COL_ITEM, COL_PWEEK,
                                COL_EST_PRICE_MODIFY_LOCAL, COL_EST_PRICE_LOCAL]].copy(deep=False)
    # dtype 정리(조인 안정성)
    for c in (COL_SHIP_TO, COL_ITEM, COL_PWEEK):
        grid[c] = grid[c].astype(str)
        ep[c]   = ep[c].astype(str)

    grid = grid.merge(ep, on=[COL_SHIP_TO, COL_ITEM, COL_PWEEK], how='left')

    # ───────────────────────────────────────────────────────────────────
    # 4) AP Price & FX 조인
    # ───────────────────────────────────────────────────────────────────
    ap = df_in_Action_Plan_Price[[COL_SHIP_TO, COL_ITEM, COL_MONTH, COL_AP_PRICE_USD]].copy(deep=False)
    # ShipTo/Item/Month dtype 통일
    for c in (COL_SHIP_TO, COL_ITEM, COL_MONTH):
        ap[c]   = ap[c].astype(str)
        grid[c] = grid[c].astype(str)

    grid = grid.merge(ap, on=[COL_SHIP_TO, COL_ITEM, COL_MONTH], how='left')

    fx = df_in_Exchange_Rate_Local[[COL_STD3, COL_PWEEK, COL_EXCHANGE_RATE_LOCAL]].copy(deep=False)
    fx[COL_STD3] = fx[COL_STD3].astype(str)
    fx[COL_PWEEK]= fx[COL_PWEEK].astype(str)
    grid = grid.merge(fx, on=[COL_STD3, COL_PWEEK], how='left')

    # ───────────────────────────────────────────────────────────────────
    # 5) per-row 계산 로직
    #    • use_ep = (Modify notnull & >0) & (EP_Local notnull)
    #    • else   = AP_USD × FX
    # ───────────────────────────────────────────────────────────────────
    mod  = pd.to_numeric(grid[COL_EST_PRICE_MODIFY_LOCAL], errors='coerce')
    epv  = pd.to_numeric(grid[COL_EST_PRICE_LOCAL], errors='coerce')
    apu  = pd.to_numeric(grid[COL_AP_PRICE_USD], errors='coerce')
    fxr  = pd.to_numeric(grid[COL_EXCHANGE_RATE_LOCAL], errors='coerce')

    use_ep = (mod.notna() & (mod > 0) & epv.notna())
    calc   = apu * fxr

    grid[COL_EST_PRICE_LOCAL] = np.where(use_ep, epv, calc)

    # ───────────────────────────────────────────────────────────────────
    # 6) 정리: NaN 제거 + Version 삽입 + dtype
    # ───────────────────────────────────────────────────────────────────
    grid.dropna(subset=[COL_EST_PRICE_LOCAL], inplace=True)

    grid.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK):
        grid[c] = grid[c].astype('category')

    out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_EST_PRICE_LOCAL]
    df_out = grid[out_cols].copy(deep=False)

    return df_out

##############################################################################################################
# Step 6-1) S/In FCST(GI) Split Ratio_AP1 생성
##############################################################################################################
@_decoration_
def fn_step06_01_build_sin_split_ratio_ap1(
    df_out_SIn_Assortment_AP1      : pd.DataFrame,  # Step2-5 결과: Output_SIn_Assortment_AP1
    df_in_Time                     : pd.DataFrame,  # Time.[Partial Week] (필요 시 파생 Month 無)
    df_in_Sell_In_Split_AP1        : pd.DataFrame,  # df_in_Sell_In_FCST_GI_Split_Ratio_AP1
    df_in_Sales_Domain_Dimension   : pd.DataFrame,  # ShipTo → Sales Std2
    df_in_Item_Master              : pd.DataFrame,  # Item → Item Std1
    out_version                    : str            # 예: 'CWV_DP'
) -> pd.DataFrame:
    """
    Step 6-1) S/In FCST(GI) Split Ratio_AP1 생성
      • Input: Output_SIn_Assortment_AP1 (Assortment=1만 사용)
      • 키: ShipTo×Item×Location 에 모든 Partial Week 결합
      • Lookup: (Sales Std2, Item Std1, Partial Week) → Split Ratio
      • 값이 없으면 생성하지 않음(삭제), Version 추가
      • Return: df_output_Sell_In_FCST_GI_Split_Ratio_AP1
    """
    # ─────────────────────────────────────────────
    # 0) 컬럼 점검
    # ─────────────────────────────────────────────
    need_assort = [COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_SIN_ASSORT_AP1]
    need_time   = [COL_PWEEK]
    need_dim    = [COL_SHIP_TO, COL_STD2]
    need_item   = [COL_ITEM, COL_ITEM_STD1]
    need_split  = [COL_STD2, COL_ITEM_STD1, COL_PWEEK, COL_SIN_SPLIT_AP1]    
    
    def _need(df, cols, tag):
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise KeyError(f"[Step6-1] {tag} missing columns: {miss}")

    _need(df_out_SIn_Assortment_AP1, need_assort, 'Output_SIn_Assortment_AP1')
    _need(df_in_Time, need_time, 'df_in_Time')
    _need(df_in_Sales_Domain_Dimension, need_dim, 'df_in_Sales_Domain_Dimension')
    _need(df_in_Item_Master, need_item, 'df_in_Item_Master')
    _need(df_in_Sell_In_Split_AP1, need_split, 'df_in_Sell_In_FCST_GI_Split_Ratio_AP1')

    # ─────────────────────────────────────────────
    # 1) Assortment=1 필터 + (ShipTo,Item,Location) 유니크
    # ─────────────────────────────────────────────
    ass = df_out_SIn_Assortment_AP1.loc[
        pd.to_numeric(df_out_SIn_Assortment_AP1[COL_SIN_ASSORT_AP1], errors='coerce').eq(1)
    , [COL_SHIP_TO, COL_ITEM, COL_LOCATION]].drop_duplicates()

    if ass.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_AP1])

    # ─────────────────────────────────────────────
    # 2) Partial Week 전개 (cross join)
    # ─────────────────────────────────────────────
    pw = df_in_Time[[COL_PWEEK]].drop_duplicates()
    if pw.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_AP1])

    ass = ass.copy(deep=False); ass['_k']=1
    pw  = pw.copy(deep=False);  pw ['_k']=1
    grid = (ass.merge(pw, on='_k', how='left')
               .drop(columns=['_k']))

    # ─────────────────────────────────────────────
    # 3) Std2, ItemStd1 부착
    # ─────────────────────────────────────────────
    dim = df_in_Sales_Domain_Dimension[[COL_SHIP_TO, COL_STD2]].copy(deep=False)
    itm = df_in_Item_Master[[COL_ITEM, COL_ITEM_STD1]].copy(deep=False)

    # 조인 안정성: 문자열로 맞춤
    for c in (COL_SHIP_TO,):
        grid[c] = grid[c].astype(str)
        dim[c]  = dim[c].astype(str)
    for c in (COL_ITEM,):
        grid[c] = grid[c].astype(str)
        itm[c]  = itm[c].astype(str)
    grid[COL_PWEEK] = grid[COL_PWEEK].astype(str)

    grid = grid.merge(dim, on=COL_SHIP_TO, how='left')
    grid = grid.merge(itm, on=COL_ITEM,   how='left')

    # Std2/ItemStd1 없는 행은 제거 (룰 매칭 불가)
    grid.dropna(subset=[COL_STD2, COL_ITEM_STD1], inplace=True)

    # ─────────────────────────────────────────────
    # 4) Split Ratio(AP1) 조인: (Std2, ItemStd1, PW) 키
    # ─────────────────────────────────────────────
    sp = df_in_Sell_In_Split_AP1[[COL_STD2, COL_ITEM_STD1, COL_PWEEK, COL_SIN_SPLIT_AP1]].copy(deep=False)
    for c in (COL_STD2, COL_ITEM_STD1, COL_PWEEK):
        sp[c]    = sp[c].astype(str)
        grid[c]  = grid[c].astype(str)

    grid = grid.merge(sp, on=[COL_STD2, COL_ITEM_STD1, COL_PWEEK], how='left')

    # 스플릿 값 없는 행은 생성하지 않음(삭제)
    grid.dropna(subset=[COL_SIN_SPLIT_AP1], inplace=True)

    # ─────────────────────────────────────────────
    # 5) Version 추가 + dtype 정리 + 반환
    # ─────────────────────────────────────────────
    grid.insert(0, COL_VERSION, out_version)

    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK):
        grid[c] = grid[c].astype('category')

    out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_AP1]
    df_out = grid[out_cols].copy(deep=False)

    # 이름상 (Output 6-1) 형태: df_output_Sell_In_FCST_GI_Split_Ratio_AP1
    return df_out

# ─────────────────────────────────────────────────────────────────────────────
# Step 6-2) S/In FCST(GI) Split Ratio_AP2 생성
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step06_02_build_sin_split_ratio_ap2(
    df_out_sin_assort_ap2      : pd.DataFrame,  # Step2-5 결과: Output_SIn_Assortment_AP2
    df_in_time                 : pd.DataFrame,  # df_in_Time (Partial Week 풀)
    df_in_sin_split_ratio_ap2  : pd.DataFrame,  # df_in_Sell_In_FCST_GI_Split_Ratio_AP2 (Std2×Item Std1×PW)
    df_in_sales_domain_dim     : pd.DataFrame,  # df_in_Sales_Domain_Dimension (ShipTo→Std2)
    df_in_item_master          : pd.DataFrame,  # df_in_Item_Master (Item→Item Std1)
    out_version                : str,           # Version (예: 'CWV_DP')
    **kwargs
) -> pd.DataFrame:
    """
    • Input
      - Output_SIn_Assortment_AP2 에서 Assortment_AP2=1인 (ShipTo, Item, Location) 조합
      - df_in_Time (Time.[Partial Week])
      - df_in_Sell_In_FCST_GI_Split_Ratio_AP2 (키: Sales Std2, Item Std1, Partial Week)
      - df_in_Sales_Domain_Dimension (ShipTo→Sales Std2)
      - df_in_Item_Master (Item→Item Std1)    • Logic
      1) Assortment_AP2=1만 추출 → (ShipTo, Item, Location) 유니크
      2) Partial Week 전체와 카티전 조인
      3) ShipTo→Std2, Item→Item Std1 매핑
      4) (Std2, Item Std1, PW)로 Split-Ratio 테이블과 LEFT JOIN
      5) Ratio가 null이면 “생성 안함(삭제)”
      6) Version 세팅 및 카테고리 캐스팅

    • Return
      df_output_Sell_In_FCST_GI_Split_Ratio_AP2
      (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_AP2)
    """
    # 필요한 상수 참조
    # COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK
    # COL_STD2, COL_ITEM_STD1, COL_SIN_ASSORT_AP2, COL_SIN_SPLIT_AP2

    # 0) 방어적 체크 & 타임 축 준비
    if (df_out_sin_assort_ap2 is None) or (df_in_time is None) or (df_in_sin_split_ratio_ap2 is None):
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_AP2])

    # 사용 PW 풀
    pw = (df_in_time[[COL_PWEEK]]
          .dropna()
          .drop_duplicates()
          .astype({COL_PWEEK: 'category'}))
    if df_out_sin_assort_ap2.empty or pw.empty or df_in_sin_split_ratio_ap2.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_AP2])

    # 1) Assortment_AP2 = 1 만 선별 → (ShipTo, Item, Location) 유니크
    base_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    need_cols = base_cols + ([COL_SIN_ASSORT_AP2] if COL_SIN_ASSORT_AP2 in df_out_sin_assort_ap2.columns else [])
    base = (df_out_sin_assort_ap2[need_cols]
            .copy(deep=False))
    if COL_SIN_ASSORT_AP2 in base.columns:
        base = base.loc[base[COL_SIN_ASSORT_AP2] == 1, base_cols]
    else:
        # 혹시 컬럼명이 다른 경우(이전 단계에서 rename 누락 등) 대비: 존재 시 1만 사용
        base = base[base_cols]
    base = base.drop_duplicates()

    if base.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_AP2])

    # 2) (ShipTo, Item, Loc) × Partial Week 카티전
    b = base.copy(deep=False)
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION):
        b[c] = b[c].astype('category')
    b['_k'] = 1
    tw = pw.copy(deep=False)
    tw['_k'] = 1
    grid = (b.merge(tw, on='_k', how='left')
              .drop(columns=['_k']))

    # 3) ShipTo→Std2, Item→Item Std1 매핑
    #    (조인용 중복 제거 및 카테고리화)
    ship2std2 = (df_in_sales_domain_dim[[COL_SHIP_TO, COL_STD2]]
                 .drop_duplicates()
                 .astype({COL_SHIP_TO: 'category', COL_STD2: 'category'}))
    itm2std1  = (df_in_item_master[[COL_ITEM, COL_ITEM_STD1]]
                 .drop_duplicates()
                 .astype({COL_ITEM: 'category', COL_ITEM_STD1: 'category'}))

    grid = (grid.merge(ship2std2, on=COL_SHIP_TO, how='left')
                .merge(itm2std1,  on=COL_ITEM,   how='left'))

    # 4) Split-Ratio(AP2) 테이블과 조인: (Std2, Item Std1, PW)
    sr_cols = [COL_STD2, COL_ITEM_STD1, COL_PWEEK, COL_SIN_SPLIT_AP2]
    sr = (df_in_sin_split_ratio_ap2[sr_cols]
          .drop_duplicates()
          .astype({COL_STD2:'category', COL_ITEM_STD1:'category', COL_PWEEK:'category'}))

    grid = grid.merge(sr, on=[COL_STD2, COL_ITEM_STD1, COL_PWEEK], how='left')

    # 5) Ratio 누락 제거 (생성 X)
    grid = grid.loc[grid[COL_SIN_SPLIT_AP2].notna()]

    if grid.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_AP2])

    # 6) Version 지정 및 정렬/캐스팅
    grid.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK):
        grid[c] = grid[c].astype('category')

    # 최종 컬럼 순서
    out = grid[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_AP2]].copy(deep=False)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Step 6-3) S/In FCST(GI) Split Ratio_GC 생성
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step06_03_build_sin_split_ratio_gc(
    df_out_sin_assort_gc      : pd.DataFrame,  # Step2-5 결과: Output_SIn_Assortment_GC
    df_in_time                : pd.DataFrame,  # df_in_Time (Partial Week 풀)
    df_in_sin_split_ratio_gc  : pd.DataFrame,  # df_in_Sell_In_FCST_GI_Split_Ratio_GC (Std2×Item Std1×PW)
    df_in_sales_domain_dim    : pd.DataFrame,  # df_in_Sales_Domain_Dimension (ShipTo→Std2)
    df_in_item_master         : pd.DataFrame,  # df_in_Item_Master (Item→Item Std1)
    out_version               : str,           # Version (예: 'CWV_DP')
    **kwargs
) -> pd.DataFrame:
    """
    Output_SIn_Assortment_GC에서 Assortment_GC=1인 (ShipTo, Item, Location)을 PW 전 기간으로 확장한 뒤
    ShipTo→Sales Std2, Item→Item Std1 매핑하여 (Std2, ItemStd1, PW) 키로 GC Split Ratio를 조인.
    Ratio가 없는 행은 생성하지 않고 제거.    반환 컬럼:
      [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_GC]
    """
    # 방어적 체크
    if (df_out_sin_assort_gc is None) or (df_in_time is None) or (df_in_sin_split_ratio_gc is None):
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_GC])

    # Partial Week 풀
    pw = (df_in_time[[COL_PWEEK]]
          .dropna()
          .drop_duplicates()
          .astype({COL_PWEEK: 'category'}))
    if df_out_sin_assort_gc.empty or pw.empty or df_in_sin_split_ratio_gc.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_GC])

    # 1) Assortment_GC = 1만 선별 → (ShipTo, Item, Location) 유니크
    base_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    need_cols = base_cols + ([COL_SIN_ASSORT_GC] if COL_SIN_ASSORT_GC in df_out_sin_assort_gc.columns else [])
    base = df_out_sin_assort_gc[need_cols].copy(deep=False)

    if COL_SIN_ASSORT_GC in base.columns:
        base = base.loc[base[COL_SIN_ASSORT_GC] == 1, base_cols]
    else:
        base = base[base_cols]  # (방어: 컬럼이 없다면 필터 없이 사용)

    base = base.drop_duplicates()
    if base.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_GC])

    # 2) (ShipTo, Item, Loc) × Partial Week 카티전
    b = base.copy(deep=False)
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION):
        b[c] = b[c].astype('category')
    b['_k'] = 1
    tw = pw.copy(deep=False)
    tw['_k'] = 1
    grid = (b.merge(tw, on='_k', how='left')
              .drop(columns=['_k']))

    # 3) ShipTo→Std2, Item→Item Std1 매핑
    ship2std2 = (df_in_sales_domain_dim[[COL_SHIP_TO, COL_STD2]]
                 .drop_duplicates()
                 .astype({COL_SHIP_TO: 'category', COL_STD2: 'category'}))
    itm2std1  = (df_in_item_master[[COL_ITEM, COL_ITEM_STD1]]
                 .drop_duplicates()
                 .astype({COL_ITEM: 'category', COL_ITEM_STD1: 'category'}))

    grid = (grid.merge(ship2std2, on=COL_SHIP_TO, how='left')
                .merge(itm2std1,  on=COL_ITEM,   how='left'))

    # 4) Split-Ratio(GC) 조인: (Std2, Item Std1, PW)
    sr_cols = [COL_STD2, COL_ITEM_STD1, COL_PWEEK, COL_SIN_SPLIT_GC]
    sr = (df_in_sin_split_ratio_gc[sr_cols]
          .drop_duplicates()
          .astype({COL_STD2:'category', COL_ITEM_STD1:'category', COL_PWEEK:'category'}))

    grid = grid.merge(sr, on=[COL_STD2, COL_ITEM_STD1, COL_PWEEK], how='left')

    # 5) Ratio 누락 제거 (생성 X)
    grid = grid.loc[grid[COL_SIN_SPLIT_GC].notna()]
    if grid.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_GC])

    # 6) Version 지정 및 캐스팅
    grid.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK):
        grid[c] = grid[c].astype('category')

    # 최종 컬럼 순서
    out = grid[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_GC]].copy(deep=False)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Step 6-4) S/In FCST(GI) Split Ratio_Local 생성
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step06_04_build_sin_split_ratio_local(
    df_out_sin_assort_local     : pd.DataFrame,  # Step2-5 결과: Output_SIn_Assortment_Local
    df_in_time                  : pd.DataFrame,  # df_in_Time (Partial Week 풀)
    df_in_sin_split_ratio_local : pd.DataFrame,  # df_in_Sell_In_FCST_GI_Split_Ratio_Local (Std2×Item Std1×PW)
    df_in_sales_domain_dim      : pd.DataFrame,  # df_in_Sales_Domain_Dimension (ShipTo→Std2)
    df_in_item_master           : pd.DataFrame,  # df_in_Item_Master (Item→Item Std1)
    out_version                 : str,           # Version (예: 'CWV_DP')
    **kwargs
) -> pd.DataFrame:
    """
    Output_SIn_Assortment_Local에서 Assortment_Local=1인 (ShipTo, Item, Location)을 PW 전 기간으로 확장하고
    ShipTo→Sales Std2, Item→Item Std1 매핑 후 (Std2, ItemStd1, PW)로 Local Split Ratio를 조인.
    Ratio가 없는 행은 생성하지 않음.    반환 컬럼:
      [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_LOCAL]
    """
    # 방어적 체크 및 Partial Week 풀
    if (df_out_sin_assort_local is None) or (df_in_time is None) or (df_in_sin_split_ratio_local is None):
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_LOCAL])

    pw = (df_in_time[[COL_PWEEK]]
          .dropna()
          .drop_duplicates()
          .astype({COL_PWEEK: 'category'}))
    if df_out_sin_assort_local.empty or pw.empty or df_in_sin_split_ratio_local.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_LOCAL])

    # 1) Assortment_Local = 1만 선별 → (ShipTo, Item, Location) 유니크
    base_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    need_cols = base_cols + ([COL_SIN_ASSORT_LOCAL] if COL_SIN_ASSORT_LOCAL in df_out_sin_assort_local.columns else [])
    base = df_out_sin_assort_local[need_cols].copy(deep=False)

    if COL_SIN_ASSORT_LOCAL in base.columns:
        base = base.loc[base[COL_SIN_ASSORT_LOCAL] == 1, base_cols]
    else:
        base = base[base_cols]

    base = base.drop_duplicates()
    if base.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_LOCAL])

    # 2) (ShipTo, Item, Loc) × Partial Week 카티전
    b = base.copy(deep=False)
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION):
        b[c] = b[c].astype('category')
    b['_k'] = 1
    tw = pw.copy(deep=False)
    tw['_k'] = 1
    grid = (b.merge(tw, on='_k', how='left')
              .drop(columns=['_k']))

    # 3) ShipTo→Std2, Item→Item Std1 매핑
    ship2std2 = (df_in_sales_domain_dim[[COL_SHIP_TO, COL_STD2]]
                 .drop_duplicates()
                 .astype({COL_SHIP_TO: 'category', COL_STD2: 'category'}))
    itm2std1  = (df_in_item_master[[COL_ITEM, COL_ITEM_STD1]]
                 .drop_duplicates()
                 .astype({COL_ITEM: 'category', COL_ITEM_STD1: 'category'}))

    grid = (grid.merge(ship2std2, on=COL_SHIP_TO, how='left')
                .merge(itm2std1,  on=COL_ITEM,   how='left'))

    # 4) Split-Ratio(Local) 조인: (Std2, Item Std1, PW)
    sr_cols = [COL_STD2, COL_ITEM_STD1, COL_PWEEK, COL_SIN_SPLIT_LOCAL]
    sr = (df_in_sin_split_ratio_local[sr_cols]
          .drop_duplicates()
          .astype({COL_STD2:'category', COL_ITEM_STD1:'category', COL_PWEEK:'category'}))

    grid = grid.merge(sr, on=[COL_STD2, COL_ITEM_STD1, COL_PWEEK], how='left')

    # 5) Ratio 누락 제거
    grid = grid.loc[grid[COL_SIN_SPLIT_LOCAL].notna()]
    if grid.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_LOCAL])

    # 6) Version 지정 및 캐스팅
    grid.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK):
        grid[c] = grid[c].astype('category')

    # 최종 컬럼 순서
    out = grid[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, COL_SIN_SPLIT_LOCAL]].copy(deep=False)
    return out


# =========================
# Step 6-5 ~ 6-8: Functions
# =========================
@_decoration_
def fn_step06_05_build_sout_split_ratio_ap1(
    df_out_sout_assort_ap1     : pd.DataFrame,  # Step2-5 결과: Output_SOut_Assortment_AP1
    df_in_time                  : pd.DataFrame,  # df_in_Time (Partial Week 풀)
    df_in_sout_split_ratio_ap1  : pd.DataFrame,  # df_in_Sell_Out_FCST_Split_Ratio_AP1 (Std2×Item Std1×PW)
    df_in_sales_domain_dim      : pd.DataFrame,  # df_in_Sales_Domain_Dimension (ShipTo→Std2)
    df_in_item_master           : pd.DataFrame,  # df_in_Item_Master (Item→Item Std1)
    out_version                 : str            # Version (예: 'CWV_DP')
) -> pd.DataFrame:
    """(S/Out) Split Ratio_AP1 생성"""
    return _build_sout_split_ratio_core(
        base_df=df_out_sout_assort_ap1,
        df_in_time=df_in_time,
        df_split=df_in_sout_split_ratio_ap1,
        df_dim=df_in_sales_domain_dim,
        df_item=df_in_item_master,
        out_version=out_version,
        ratio_col=COL_SOUT_SPLIT_AP1,
        assort_col=COL_SOUT_ASSORT_AP1
    )

@_decoration_
def fn_step06_06_build_sout_split_ratio_ap2(
    df_out_sout_assort_ap2     : pd.DataFrame,  # Step2-5 결과: Output_SOut_Assortment_AP2
    df_in_time                  : pd.DataFrame,  # df_in_Time (Partial Week 풀)
    df_in_sout_split_ratio_ap2  : pd.DataFrame,  # df_in_Sell_Out_FCST_Split_Ratio_AP2 (Std2×Item Std1×PW)
    df_in_sales_domain_dim      : pd.DataFrame,  # df_in_Sales_Domain_Dimension (ShipTo→Std2)
    df_in_item_master           : pd.DataFrame,  # df_in_Item_Master (Item→Item Std1)
    out_version                 : str            # Version (예: 'CWV_DP')
) -> pd.DataFrame:
    """(S/Out) Split Ratio_AP2 생성"""
    return _build_sout_split_ratio_core(
        base_df=df_out_sout_assort_ap2,
        df_in_time=df_in_time,
        df_split=df_in_sout_split_ratio_ap2,
        df_dim=df_in_sales_domain_dim,
        df_item=df_in_item_master,
        out_version=out_version,
        ratio_col=COL_SOUT_SPLIT_AP2,
        assort_col=COL_SOUT_ASSORT_AP2
    )


@_decoration_
def fn_step06_07_build_sout_split_ratio_gc(
    df_out_sout_assort_gc     : pd.DataFrame,  # Step2-5 결과: Output_SOut_Assortment_GC
    df_in_time                 : pd.DataFrame,  # df_in_Time (Partial Week 풀)
    df_in_sout_split_ratio_gc  : pd.DataFrame,  # df_in_Sell_Out_FCST_Split_Ratio_GC (Std2×Item Std1×PW)
    df_in_sales_domain_dim     : pd.DataFrame,  # df_in_Sales_Domain_Dimension (ShipTo→Std2)
    df_in_item_master          : pd.DataFrame,  # df_in_Item_Master (Item→Item Std1)
    out_version                : str            # Version (예: 'CWV_DP')
) -> pd.DataFrame:
    """(S/Out) Split Ratio_GC 생성"""
    return _build_sout_split_ratio_core(
        base_df=df_out_sout_assort_gc,
        df_in_time=df_in_time,
        df_split=df_in_sout_split_ratio_gc,
        df_dim=df_in_sales_domain_dim,
        df_item=df_in_item_master,
        out_version=out_version,
        ratio_col=COL_SOUT_SPLIT_GC,
        assort_col=COL_SOUT_ASSORT_GC
    )


@_decoration_
def fn_step06_08_build_sout_split_ratio_local(
    df_out_sout_assort_local     : pd.DataFrame,  # Step2-5 결과: Output_SOut_Assortment_Local
    df_in_time                    : pd.DataFrame,  # df_in_Time (Partial Week 풀)
    df_in_sout_split_ratio_local  : pd.DataFrame,  # df_in_Sell_Out_FCST_Split_Ratio_Local (Std2×Item Std1×PW)
    df_in_sales_domain_dim        : pd.DataFrame,  # df_in_Sales_Domain_Dimension (ShipTo→Std2)
    df_in_item_master             : pd.DataFrame,  # df_in_Item_Master (Item→Item Std1)
    out_version                   : str            # Version (예: 'CWV_DP')
) -> pd.DataFrame:
    """(S/Out) Split Ratio_Local 생성"""
    return _build_sout_split_ratio_core(
        base_df=df_out_sout_assort_local,
        df_in_time=df_in_time,
        df_split=df_in_sout_split_ratio_local,
        df_dim=df_in_sales_domain_dim,
        df_item=df_in_item_master,
        out_version=out_version,
        ratio_col=COL_SOUT_SPLIT_LOCAL,
        assort_col=COL_SOUT_ASSORT_LOCAL
    )


# -------------------------------------------------------
# 공통 코어 (6-5~6-8에서 공용) — Python 3.10 호환, 카티전 안전
# -------------------------------------------------------
def _build_sout_split_ratio_core(
    base_df    : pd.DataFrame,
    df_in_time : pd.DataFrame,
    df_split   : pd.DataFrame,
    df_dim     : pd.DataFrame,
    df_item    : pd.DataFrame,
    out_version: str,
    *,
    ratio_col  : str,
    assort_col : str
) -> pd.DataFrame:
    """
    • base_df: Output_SOut_Assortment_XX (Assortment=1만 사용)
    • df_split: df_in_Sell_Out_FCST_Split_Ratio_XX  (키: Sales Std2, Item Std1, Partial Week)
    생성 컬럼: [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, ratio_col]
    """
    # 필수 체크
    if (base_df is None) or (df_in_time is None) or (df_split is None) \
       or (df_dim is None) or (df_item is None):
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, ratio_col])

    # Partial Week 풀
    pw = (df_in_time[[COL_PWEEK]]
          .dropna()
          .drop_duplicates()
          .astype({COL_PWEEK: 'category'}))
    if base_df.empty or pw.empty or df_split.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, ratio_col])

    # 1) Assortment==1 선별
    base_cols = [COL_SHIP_TO, COL_ITEM, COL_LOCATION]
    need_cols = base_cols + ([assort_col] if assort_col in base_df.columns else [])
    base = base_df[need_cols].copy(deep=False)

    if assort_col in base.columns:
        base = base.loc[base[assort_col] == 1, base_cols]  # 1만
    else:
        base = base[base_cols]

    base = base.drop_duplicates()
    if base.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, ratio_col])

    # 2) (ShipTo, Item, Loc) × Partial Week 카티전
    b = base.copy(deep=False)
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION):
        b[c] = b[c].astype('category')
    b['_k'] = 1
    tw = pw.copy(deep=False)
    tw['_k'] = 1
    grid = (b.merge(tw, on='_k', how='left')
              .drop(columns=['_k']))

    # 3) ShipTo→Std2, Item→Item Std1 매핑
    ship2std2 = (df_dim[[COL_SHIP_TO, COL_STD2]]
                 .drop_duplicates()
                 .astype({COL_SHIP_TO:'category', COL_STD2:'category'}))
    itm2std1  = (df_item[[COL_ITEM, COL_ITEM_STD1]]
                 .drop_duplicates()
                 .astype({COL_ITEM:'category', COL_ITEM_STD1:'category'}))

    grid = (grid.merge(ship2std2, on=COL_SHIP_TO, how='left')
                .merge(itm2std1,  on=COL_ITEM,   how='left'))

    # 4) Split Ratio 조인 (Std2, ItemStd1, PW)
    sr_cols_present = [c for c in (COL_STD2, COL_ITEM_STD1, COL_PWEEK, ratio_col) if c in df_split.columns]
    sr = (df_split[sr_cols_present]
          .drop_duplicates()
          .astype({COL_STD2:'category', COL_ITEM_STD1:'category', COL_PWEEK:'category'}))

    grid = grid.merge(sr, on=[COL_STD2, COL_ITEM_STD1, COL_PWEEK], how='left')

    # 5) Ratio 값 없는 행 제거
    if ratio_col in grid.columns:
        grid = grid.loc[grid[ratio_col].notna()]
    else:
        # ratio 컬럼 자체가 없으면 전부 생성불가
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, ratio_col])

    if grid.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, ratio_col])

    # 6) Version 추가 및 캐스팅
    grid.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK):
        grid[c] = grid[c].astype('category')

    # 7) 최종 컬럼 순서
    out = grid[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_PWEEK, ratio_col]].copy(deep=False)
    return out


# =========================
# Step 6-9) S/In Stretch Plan Assortment 생성
# =========================
@_decoration_
def fn_step06_09_build_sin_stretch_plan_assort(
    df_step01_05_asn_delta_no_loc     : pd.DataFrame,  # Step 1-5 결과 (No-Location)
    df_in_sin_stretch_plan_assortment : pd.DataFrame,  # df_in_Sell_In_Stretch_Plan_Assortment (기존값 있으면 제거)
    out_version                        : str            # 예: 'CWV_DP'
) -> pd.DataFrame:
    """
    Step 6-9) S/In Stretch Plan Assortment 생성
    • 소스: Step 1-5 결과 (No-Location)
    • 로직:
      - ASN='Y' → 1,  ASN='N' → NULL
      - Sales Product ASN 컬럼 삭제
      - df_in_Sell_In_Stretch_Plan_Assortment에 존재하는 (ShipTo,Item) 키는 출력에서 제거(override)
    • 반환 컬럼: [Version, Ship To, Item, S/In Stretch Plan Assortment]
    """
    out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_SIN_STRETCH_ASSORT]    # 빈 입력 방어
    if df_step01_05_asn_delta_no_loc is None or df_step01_05_asn_delta_no_loc.empty:
        return pd.DataFrame(columns=out_cols)

    # 1) (ShipTo, Item, ASN) 유니크 추출
    need = [c for c in (COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN) if c in df_step01_05_asn_delta_no_loc.columns]
    if not set((COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN)).issubset(need):
        # 필수 컬럼 없으면 빈 DF 반환
        return pd.DataFrame(columns=out_cols)

    base = (df_step01_05_asn_delta_no_loc[need]
            .dropna(subset=[COL_SHIP_TO, COL_ITEM])
            .drop_duplicates()
            .copy(deep=False))

    # ────────────────────────────────────────────────────────────────────────
    # 1) ASN 맵 (Loc / No-Location)
    #    - Loc  : (ShipTo,Item,Location) → 1/0
    #    - NoLoc: (ShipTo,Item)          → 1/0 (max 집계)
    # ────────────────────────────────────────────────────────────────────────
    def _yn_to_flag(s: pd.Series) -> pd.Series:
        return (s.astype(str).str.upper().eq('Y')).astype('int8')

    # 2) ASN → flag → Measure 값 (Y→1, N→NaN)
    base['_asn_flag'] = _yn_to_flag(base[COL_SALES_PRODUCT_ASN])  # Y→1, N/기타→0
    base.drop(columns=[COL_SALES_PRODUCT_ASN], inplace=True, errors='ignore')

    base[COL_SIN_STRETCH_ASSORT] = np.where(base['_asn_flag'] == 1, 1.0, np.nan)
    base.drop(columns=['_asn_flag'], inplace=True)

    # 3) 외부 입력(df_in_Sell_In_Stretch_Plan_Assortment)에 존재하는 키는 제거
    if df_in_sin_stretch_plan_assortment is not None and (not df_in_sin_stretch_plan_assortment.empty):
        # 키는 (ShipTo, Item)만 사용 (No-Location 스펙)
        key_cols = [c for c in (COL_SHIP_TO, COL_ITEM) if c in df_in_sin_stretch_plan_assortment.columns]
        if len(key_cols) == 2:
            ext_key = df_in_sin_stretch_plan_assortment[key_cols].dropna().drop_duplicates()
            # anti-join
            base = (base.merge(ext_key, on=key_cols, how='left', indicator=True)
                        .loc[lambda d: d['_merge'] == 'left_only']
                        .drop(columns=['_merge']))

    if base.empty:
        return pd.DataFrame(columns=out_cols)

    # 4) Version 추가 + dtype 정리
    base.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM):
        base[c] = base[c].astype('category')

    # 5) 최종 컬럼 정렬
    out = base[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_SIN_STRETCH_ASSORT]].copy(deep=False)
    return out


# =========================
# Step 6-10) S/In Stretch Plan Split Ratio
#  - 입력: Step 1-5 결과(df_step01_05_asn_delta_no_loc) 사용
#  - ASN='Y'만 사용 → Partial Week 전개 → ShipTo→Std2, Item→Item Std1 매핑
#  - (Std2, ItemStd1, PW)로 df_in_SIN_stretch_split_ratio 조인
#  - 값 없는 행은 생성하지 않음 (drop)
# =========================
@_decoration_
def fn_step06_10_build_sin_stretch_plan_split_ratio(
    df_step01_05_asn_delta_no_loc : pd.DataFrame,  # ← Step 1-5 결과 (No-Location)
    df_in_Time                    : pd.DataFrame,   # ← Time (Partial Week)
    df_in_SIN_stretch_split_ratio : pd.DataFrame,   # ← df_in_Sell_In_Stretch_Plan_Split_Ratio (키: Std2, ItemStd1, PW)
    df_in_Sales_Domain_Dimension  : pd.DataFrame,   # ← ShipTo → Sales Std2 매핑
    df_in_Item_Master             : pd.DataFrame,   # ← Item → Item Std1 매핑
    out_version                   : str,            # ← 예: 'CWV_DP'
    **kwargs
) -> pd.DataFrame:
    """
    Step 6-10) S/In Stretch Plan Split Ratio
    • Step 1-5 결과 중 'Sales Product ASN' == 'Y' 만 사용 (No-Location)
    • ShipTo×Item 유니크 → Partial Week 전개 → ShipTo→Std2, Item→Item Std1 매핑
    • (Std2, ItemStd1, PartialWeek) 키로 기준 테이블과 조인
    • 매칭되지 않는 행은 생성하지 않음 (drop)
    반환 컬럼: [Version, Ship To, Item, Partial Week, S/In Stretch Plan Split Ratio]
    """
    out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_SIN_STRETCH_SPLIT]    # ── 방어 ───────────────────────────────────────────────────────────────
    if (df_step01_05_asn_delta_no_loc is None) or df_step01_05_asn_delta_no_loc.empty:
        return pd.DataFrame(columns=out_cols)
    if (df_in_Time is None) or df_in_Time.empty:
        return pd.DataFrame(columns=out_cols)
    if (df_in_Sales_Domain_Dimension is None) or df_in_Sales_Domain_Dimension.empty:
        return pd.DataFrame(columns=out_cols)
    if (df_in_Item_Master is None) or df_in_Item_Master.empty:
        return pd.DataFrame(columns=out_cols)
    if (df_in_SIN_stretch_split_ratio is None) or df_in_SIN_stretch_split_ratio.empty:
        return pd.DataFrame(columns=out_cols)

    # ── 1) Step 1-5 결과에서 ASN='Y'만 선별 (No-Location) ───────────────
    cols_15 = [c for c in (COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN)
               if c in df_step01_05_asn_delta_no_loc.columns]
    base = (df_step01_05_asn_delta_no_loc[cols_15]
              .dropna(subset=[COL_SHIP_TO, COL_ITEM])
              .copy(deep=False))
    base = base[base[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')]
    if base.empty:
        return pd.DataFrame(columns=out_cols)

    base = base[[COL_SHIP_TO, COL_ITEM]].drop_duplicates(ignore_index=True)

    # ── 2) Partial Week 전개 ─────────────────────────────────────────────
    pw = (df_in_Time[[COL_PWEEK]]
            .dropna()
            .drop_duplicates())
    if pw.empty:
        return pd.DataFrame(columns=out_cols)

    base['_k'] = 1
    pw['_k']   = 1
    grid = (base.merge(pw, on='_k', how='left')
                .drop(columns=['_k']))

    # ── 3) ShipTo→Std2, Item→Item Std1 매핑 ─────────────────────────────
    dim_map = (df_in_Sales_Domain_Dimension[[COL_SHIP_TO, COL_STD2]]
                  .dropna(subset=[COL_SHIP_TO, COL_STD2])
                  .drop_duplicates())
    grid = grid.merge(dim_map, on=COL_SHIP_TO, how='left')

    itm_map = (df_in_Item_Master[[COL_ITEM, COL_ITEM_STD1]]
                  .dropna(subset=[COL_ITEM, COL_ITEM_STD1])
                  .drop_duplicates())
    grid = grid.merge(itm_map, on=COL_ITEM, how='left')

    # 매핑 실패 제거
    grid = grid.dropna(subset=[COL_STD2, COL_ITEM_STD1, COL_PWEEK])
    if grid.empty:
        return pd.DataFrame(columns=out_cols)

    # ── 4) 기준 Split Ratio 조인 (Std2, ItemStd1, PartialWeek) ──────────
    src_cols = [c for c in (COL_STD2, COL_ITEM_STD1, COL_PWEEK, COL_SIN_STRETCH_SPLIT)
                if c in df_in_SIN_stretch_split_ratio.columns]
    ref = (df_in_SIN_stretch_split_ratio[src_cols]
             .dropna(subset=[COL_STD2, COL_ITEM_STD1, COL_PWEEK])
             .drop_duplicates())

    grid = grid.merge(ref, on=[COL_STD2, COL_ITEM_STD1, COL_PWEEK], how='left')

    # 값 없는 행은 생성하지 않음
    grid = grid.dropna(subset=[COL_SIN_STRETCH_SPLIT])
    if grid.empty:
        return pd.DataFrame(columns=out_cols)

    # ── 5) Version 부여 및 dtypes ───────────────────────────────────────
    grid.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK):
        grid[c] = grid[c].astype('category')

    # ── 6) 최종 컬럼 정렬 ────────────────────────────────────────────────
    out = grid[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_SIN_STRETCH_SPLIT]].copy(deep=False)
    out = out.drop_duplicates(ignore_index=True)
    return out


# =========================
# Step 6-11) S/Out Stretch Plan Assortment
#  - 입력: Step 1-5 결과(No-Location) 사용
#  - Sales Product ASN == 'Y' → 1,  그 외(N/NaN) → NULL
#  - df_in_Sell_Out_Stretch_Plan_Assortment 에 존재하는 (ShipTo, Item) 은 제외(anti-join)
#  - 반환 컬럼: [Version, Ship To, Item, S/Out Stretch Plan Assortment]
# =========================
@_decoration_
def fn_step06_11_build_sout_stretch_plan_assortment(
    df_step01_05_asn_delta_no_loc  : pd.DataFrame,   # ← Step 1-5 결과 (No-Location)
    df_in_SOUT_stretch_assort      : pd.DataFrame,   # ← df_in_Sell_Out_Stretch_Plan_Assortment (사용자 입력, 있으면 제외)
    out_version                    : str            # ← 예: 'CWV_DP'
) -> pd.DataFrame:
    out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_SOUT_STRETCH_ASSORT]    # 방어
    if df_step01_05_asn_delta_no_loc is None or df_step01_05_asn_delta_no_loc.empty:
        return pd.DataFrame(columns=out_cols)

    # 1) Step 1-5 결과에서 ShipTo×Item 단위로 최신 ASN 반영 (여기선 단순 사용)
    cols_15 = [c for c in (COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN)
               if c in df_step01_05_asn_delta_no_loc.columns]
    base = (df_step01_05_asn_delta_no_loc[cols_15]
              .dropna(subset=[COL_SHIP_TO, COL_ITEM])
              .drop_duplicates([COL_SHIP_TO, COL_ITEM], keep='last')
              .copy(deep=False))

    # 2) 사용자 입력(override) 있는 (ShipTo, Item) 제외(anti-join)
    if df_in_SOUT_stretch_assort is not None and not df_in_SOUT_stretch_assort.empty:
        ov_pairs = (df_in_SOUT_stretch_assort[[COL_SHIP_TO, COL_ITEM]]
                      .dropna()
                      .drop_duplicates())
        if not ov_pairs.empty:
            base = (base.merge(ov_pairs.assign(_drop=1),
                               on=[COL_SHIP_TO, COL_ITEM],
                               how='left')
                        .loc[lambda x: x['_drop'].isna()]
                        .drop(columns=['_drop']))
    if base.empty:
        return pd.DataFrame(columns=out_cols)

    # 3) Measure 부여: ASN=='Y' → 1, else NULL
    asn_is_y = base[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')
    base[COL_SOUT_STRETCH_ASSORT] = np.where(asn_is_y, 1, np.nan)

    # 4) Version + 형식 정리
    base.insert(0, COL_VERSION, out_version)
    out = base[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_SOUT_STRETCH_ASSORT]].copy(deep=False)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM):
        out[c] = out[c].astype('category')
    out = out.drop_duplicates(ignore_index=True)
    return out

# =========================
# Step 6-12) S/Out Stretch Plan Split Ratio
#  - 입력: Step 1-5 결과(No-Location) 중 ASN=='Y'만 사용
#  - ShipTo×Item 유니크 → Partial Week 전개
#  - ShipTo→Std2, Item→Item Std1 매핑
#  - (Std2, ItemStd1, PartialWeek) 로 기준 테이블(df_in_Sell_Out_Stretch_Plan_Split_Ratio) 조인
#  - 값 없는 행은 생성하지 않음
#  - 반환 컬럼: [Version, Ship To, Item, Partial Week, S/Out Stretch Plan Split Ratio]
# =========================
@_decoration_
def fn_step06_12_build_sout_stretch_plan_split_ratio(
    df_step01_05_asn_delta_no_loc   : pd.DataFrame,  # ← Step 1-5 결과 (No-Location)
    df_in_Time                      : pd.DataFrame,  # ← Time (Partial Week)
    df_in_SOUT_stretch_split_ratio  : pd.DataFrame,  # ← df_in_Sell_Out_Stretch_Plan_Split_Ratio
    df_in_Sales_Domain_Dimension    : pd.DataFrame,  # ← ShipTo → Sales Std2 매핑
    df_in_Item_Master               : pd.DataFrame,  # ← Item → Item Std1 매핑
    out_version                     : str            # ← 예: 'CWV_DP'
) -> pd.DataFrame:
    out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_SOUT_STRETCH_SPLIT]

    # 방어
    if df_step01_05_asn_delta_no_loc is None or df_step01_05_asn_delta_no_loc.empty:
        return pd.DataFrame(columns=out_cols)
    if df_in_Time is None or df_in_Time.empty:
        return pd.DataFrame(columns=out_cols)
    if df_in_Sales_Domain_Dimension is None or df_in_Sales_Domain_Dimension.empty:
        return pd.DataFrame(columns=out_cols)
    if df_in_Item_Master is None or df_in_Item_Master.empty:
        return pd.DataFrame(columns=out_cols)
    if df_in_SOUT_stretch_split_ratio is None or df_in_SOUT_stretch_split_ratio.empty:
        return pd.DataFrame(columns=out_cols)

    # 1) ASN=='Y'만 선별 (No-Location)
    cols_15 = [c for c in (COL_SHIP_TO, COL_ITEM, COL_SALES_PRODUCT_ASN)
               if c in df_step01_05_asn_delta_no_loc.columns]
    base = (df_step01_05_asn_delta_no_loc[cols_15]
              .dropna(subset=[COL_SHIP_TO, COL_ITEM])
              .copy(deep=False))
    base = base[base[COL_SALES_PRODUCT_ASN].astype(str).str.upper().eq('Y')]
    if base.empty:
        return pd.DataFrame(columns=out_cols)

    base = base[[COL_SHIP_TO, COL_ITEM]].drop_duplicates(ignore_index=True)

    # 2) Partial Week 전개
    pw = (df_in_Time[[COL_PWEEK]]
            .dropna()
            .drop_duplicates())
    if pw.empty:
        return pd.DataFrame(columns=out_cols)

    base['_k'] = 1
    pw['_k']   = 1
    grid = (base.merge(pw, on='_k', how='left')
                .drop(columns=['_k']))

    # 3) ShipTo→Std2, Item→Item Std1 매핑
    dim_map = (df_in_Sales_Domain_Dimension[[COL_SHIP_TO, COL_STD2]]
                  .dropna(subset=[COL_SHIP_TO, COL_STD2])
                  .drop_duplicates())
    grid = grid.merge(dim_map, on=COL_SHIP_TO, how='left')

    itm_map = (df_in_Item_Master[[COL_ITEM, COL_ITEM_STD1]]
                  .dropna(subset=[COL_ITEM, COL_ITEM_STD1])
                  .drop_duplicates())
    grid = grid.merge(itm_map, on=COL_ITEM, how='left')

    # 매핑 실패 제거
    grid = grid.dropna(subset=[COL_STD2, COL_ITEM_STD1, COL_PWEEK])
    if grid.empty:
        return pd.DataFrame(columns=out_cols)

    # 4) 기준 Split Ratio 조인 (Std2, ItemStd1, PartialWeek)
    src_cols = [c for c in (COL_STD2, COL_ITEM_STD1, COL_PWEEK, COL_SOUT_STRETCH_SPLIT)
                if c in df_in_SOUT_stretch_split_ratio.columns]
    ref = (df_in_SOUT_stretch_split_ratio[src_cols]
             .dropna(subset=[COL_STD2, COL_ITEM_STD1, COL_PWEEK])
             .drop_duplicates())

    grid = grid.merge(ref, on=[COL_STD2, COL_ITEM_STD1, COL_PWEEK], how='left')

    # 값 없는 행은 생성하지 않음
    grid = grid.dropna(subset=[COL_SOUT_STRETCH_SPLIT])
    if grid.empty:
        return pd.DataFrame(columns=out_cols)

    # 5) Version + 형식
    grid.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK):
        grid[c] = grid[c].astype('category')

    out = grid[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_PWEEK, COL_SOUT_STRETCH_SPLIT]].copy(deep=False)
    out = out.drop_duplicates(ignore_index=True)
    return out

# =========================================
# Step 7) eStore GI Ratio Data 생성
#  - Python 3.10 호환
#  - 공통 유틸 + 7-0 ~ 7-6
# =========================================# ── 공통: 결측 컬럼 체크(간단버전) ─────────────────────────────────────────
def _check_has_cols(df: pd.DataFrame, req: list[str], tag: str):
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"[Step7] {tag} missing columns: {miss}")

# ── Step 7-0) ASN Delta 원본 정제 (E-Store 필터 + Version 제거) ────────────
@_decoration_
def fn_step07_00_prepare_asn_delta_base(
    df_step01_01_asn_delta    : pd.DataFrame,  # ← Step 1-1 결과
    df_in_Sales_Domain_Estore : pd.DataFrame   # ← E-Store ShipTo 목록
) -> pd.DataFrame:
    """
    반환: ShipTo × Item × Location (E-Store 계정만, Version 제거)
    컬럼: [Sales Domain.[Ship To], Item.[Item], Location.[Location]]
    """
    if df_step01_01_asn_delta is None or df_step01_01_asn_delta.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION])

    # 필요 컬럼 확인
    _check_has_cols(df_step01_01_asn_delta,
        [COL_SHIP_TO, COL_ITEM, COL_LOCATION], 'df_step01_01_asn_delta')
    _check_has_cols(df_in_Sales_Domain_Estore, [COL_SHIP_TO], 'df_in_Sales_Domain_Estore')

    # E-Store 계정만 남김
    estore_set = set(df_in_Sales_Domain_Estore[COL_SHIP_TO].astype(str))
    base = (df_step01_01_asn_delta[[COL_SHIP_TO, COL_ITEM, COL_LOCATION]]
              .dropna(subset=[COL_SHIP_TO, COL_ITEM, COL_LOCATION])
              .copy(deep=False))
    base = base[base[COL_SHIP_TO].astype(str).isin(estore_set)]

    if base.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION])

    # 중복 정리 + dtype 최적화
    base = base.drop_duplicates(ignore_index=True)
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION):
        base[c] = base[c].astype('category')
    return base

# ── 공통: ShipTo×Item×Location × Week 전개 + (Item→Item Std1) 매핑 ─────────
def _step07_expand_with_week_and_std1(
    base_SIL                    : pd.DataFrame,  # [ShipTo, Item, Location]
    df_in_Time                  : pd.DataFrame,  # [Time.[Week]]
    df_in_Item_Master           : pd.DataFrame,  # [Item.[Item], Item.[Item Std1]]
) -> pd.DataFrame:
    if base_SIL is None or base_SIL.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK, COL_ITEM_STD1])

    _check_has_cols(df_in_Time,        [COL_WEEK],       'df_in_Time')
    _check_has_cols(df_in_Item_Master, [COL_ITEM, COL_ITEM_STD1], 'df_in_Item_Master')

    # Week 벡터
    wk = (df_in_Time[[COL_WEEK]].dropna().drop_duplicates())
    if wk.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK, COL_ITEM_STD1])

    # Cross join (카티전 곱) : ShipTo×Item×Loc × Week
    left  = base_SIL.copy(deep=False)
    left['_k']  = 1
    right = wk.copy(deep=False)
    right['_k'] = 1
    grid = (left.merge(right, on='_k', how='left')
                 .drop(columns=['_k']))

    # Item → Item Std1 매핑
    imap = (df_in_Item_Master[[COL_ITEM, COL_ITEM_STD1]]
            .dropna(subset=[COL_ITEM, COL_ITEM_STD1])
            .drop_duplicates())
    grid = grid.merge(imap, on=COL_ITEM, how='left')

    # Std1, Week, Location 모두 있어야 매핑 가능
    grid = grid.dropna(subset=[COL_ITEM_STD1, COL_LOCATION, COL_WEEK])

    # dtype 최적화
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK, COL_ITEM_STD1):
        grid[c] = grid[c].astype('category')

    return grid

# ── 공통: (Std1, Location, Week)로 Ratio 테이블 매핑 & 출력 ────────────────
def _step07_apply_ratio(
    grid_SILW            : pd.DataFrame,  # [ShipTo, Item, Location, Week, ItemStd1]
    ref_ratio            : pd.DataFrame,  # 참조 테이블
    measure_cols         : list[str],     # 가져올 측정치 컬럼들
) -> pd.DataFrame:
    if grid_SILW is None or grid_SILW.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK] + measure_cols)
    if ref_ratio is None or ref_ratio.empty:
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK] + measure_cols)    # --- (추가) 참조테이블 Week 정규화: '202506A' -> '202506', '202509B' -> '202509'
    #     숫자가 아닌 문자를 모두 제거해 조인 키를 맞춥니다.
    ref_ratio = ref_ratio.copy(deep=False)
    if COL_WEEK in ref_ratio.columns:
        ref_ratio[COL_WEEK] = (
            ref_ratio[COL_WEEK]
            .astype(str)
            .str.replace(r'\D+', '', regex=True)   # 비숫자 전부 제거
        )
        # 제거 후 빈 문자열이 되면 NA 처리
        ref_ratio.loc[ref_ratio[COL_WEEK].eq(''), COL_WEEK] = np.nan

    # 필요 컬럼 체크 (정규화 후)
    _check_has_cols(ref_ratio, [COL_ITEM_STD1, COL_LOCATION, COL_WEEK], 'ref_ratio')

    # 참조 테이블 슬라이스
    take_cols = [COL_ITEM_STD1, COL_LOCATION, COL_WEEK] + [c for c in measure_cols if c in ref_ratio.columns]
    ref = (ref_ratio[take_cols]
           .dropna(subset=[COL_ITEM_STD1, COL_LOCATION, COL_WEEK])
           .drop_duplicates())

    if len(take_cols) == 3:
        # measure가 하나도 없으면 결과 없음
        return pd.DataFrame(columns=[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK] + measure_cols)

    # (안전) 그리드 Week도 문자열로 변환 (보통은 이미 '202506' 형태이지만 혹시 몰라 통일)
    grid = grid_SILW.copy(deep=False)
    grid[COL_WEEK] = grid[COL_WEEK].astype(str).str.replace(r'\D+', '', regex=True)

    # inner-join: 매핑 없는 행은 생성하지 않음
    out = grid.merge(ref, on=[COL_ITEM_STD1, COL_LOCATION, COL_WEEK], how='inner')

    # 정리
    out = out[[COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK] + [c for c in measure_cols if c in out.columns]]
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK):
        out[c] = out[c].astype('category')
    out = out.drop_duplicates(ignore_index=True)
    return out

# ── 공통: 최종 포맷팅(Version 추가 + 컬럼 순서) ────────────────────────────
def _step07_finalize(
    df: pd.DataFrame,
    measure_cols: list[str],
    out_version: str
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK] + measure_cols)
    df.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK):
        df[c] = df[c].astype('category')
    cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK] + [c for c in measure_cols if c in df.columns]
    return df[cols]


# ─────────────────────────────────────────────────────────────────────────────
# 7-1) S/In User GI Ratio
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step07_01_build_sin_user_gi_ratio(
    base_asn_delta_estore   : pd.DataFrame,  # ← Step 7-0 결과
    df_in_Time              : pd.DataFrame,
    df_in_Item_Master       : pd.DataFrame,
    df_in_SIN_user_ratio    : pd.DataFrame,  # ← df_in_Sell_In_User_GI_Ratio
    out_version             : str,
    **kwargs
) -> pd.DataFrame:
    measures = [
        COL_SIN_USER_GI_LT, COL_SIN_USER_GI_W7, COL_SIN_USER_GI_W6, COL_SIN_USER_GI_W5,
        COL_SIN_USER_GI_W4, COL_SIN_USER_GI_W3, COL_SIN_USER_GI_W2, COL_SIN_USER_GI_W1, COL_SIN_USER_GI_W0
    ]
    grid = _step07_expand_with_week_and_std1(base_asn_delta_estore, df_in_Time, df_in_Item_Master)
    mapped = _step07_apply_ratio(grid, df_in_SIN_user_ratio, measures)
    return _step07_finalize(mapped, measures, out_version)
# ─────────────────────────────────────────────────────────────────────────────
# 7-2) S/In Issue GI Ratio
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step07_02_build_sin_issue_gi_ratio(
    base_asn_delta_estore   : pd.DataFrame,
    df_in_Time              : pd.DataFrame,
    df_in_Item_Master       : pd.DataFrame,
    df_in_SIN_issue_ratio   : pd.DataFrame,  # ← df_in_Sell_In_Issue_GI_Ratio
    out_version             : str,
    **kwargs
) -> pd.DataFrame:
    measures = [
        COL_SIN_ISSUE_GI_LT, COL_SIN_ISSUE_GI_W7, COL_SIN_ISSUE_GI_W6, COL_SIN_ISSUE_GI_W5,
        COL_SIN_ISSUE_GI_W4, COL_SIN_ISSUE_GI_W3, COL_SIN_ISSUE_GI_W2, COL_SIN_ISSUE_GI_W1, COL_SIN_ISSUE_GI_W0
    ]
    grid = _step07_expand_with_week_and_std1(base_asn_delta_estore, df_in_Time, df_in_Item_Master)
    mapped = _step07_apply_ratio(grid, df_in_SIN_issue_ratio, measures)
    return _step07_finalize(mapped, measures, out_version)
# ─────────────────────────────────────────────────────────────────────────────
# 7-3) S/In Final GI Ratio
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step07_03_build_sin_final_gi_ratio(
    base_asn_delta_estore   : pd.DataFrame,
    df_in_Time              : pd.DataFrame,
    df_in_Item_Master       : pd.DataFrame,
    df_in_SIN_final_ratio   : pd.DataFrame,  # ← df_in_Sell_In_Final_GI_Ratio
    out_version             : str,
    **kwargs
) -> pd.DataFrame:
    measures = [
        COL_SIN_FINAL_GI_LT, COL_SIN_FINAL_GI_W7, COL_SIN_FINAL_GI_W6, COL_SIN_FINAL_GI_W5,
        COL_SIN_FINAL_GI_W4, COL_SIN_FINAL_GI_W3, COL_SIN_FINAL_GI_W2, COL_SIN_FINAL_GI_W1, COL_SIN_FINAL_GI_W0
    ]
    grid = _step07_expand_with_week_and_std1(base_asn_delta_estore, df_in_Time, df_in_Item_Master)
    mapped = _step07_apply_ratio(grid, df_in_SIN_final_ratio, measures)
    return _step07_finalize(mapped, measures, out_version)
# ─────────────────────────────────────────────────────────────────────────────
# 7-4) S/In BestFit GI Ratio
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step07_04_build_sin_bestfit_gi_ratio(
    base_asn_delta_estore     : pd.DataFrame,
    df_in_Time                : pd.DataFrame,
    df_in_Item_Master         : pd.DataFrame,
    df_in_SIN_bestfit_ratio   : pd.DataFrame,  # ← df_in_Sell_In_BestFit_GI_Ratio
    out_version               : str,
    **kwargs
) -> pd.DataFrame:
    measures = [
        COL_SIN_BESTFIT_GI_LT, COL_SIN_BESTFIT_GI_W7, COL_SIN_BESTFIT_GI_W6, COL_SIN_BESTFIT_GI_W5,
        COL_SIN_BESTFIT_GI_W4, COL_SIN_BESTFIT_GI_W3, COL_SIN_BESTFIT_GI_W2, COL_SIN_BESTFIT_GI_W1, COL_SIN_BESTFIT_GI_W0
    ]
    grid = _step07_expand_with_week_and_std1(base_asn_delta_estore, df_in_Time, df_in_Item_Master)
    mapped = _step07_apply_ratio(grid, df_in_SIN_bestfit_ratio, measures)
    return _step07_finalize(mapped, measures, out_version)
# ─────────────────────────────────────────────────────────────────────────────
# 7-5) S/In User Item GI Ratio
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step07_05_build_sin_user_item_gi_ratio(
    base_asn_delta_estore       : pd.DataFrame,
    df_in_Time                  : pd.DataFrame,
    df_in_Item_Master           : pd.DataFrame,
    df_in_SIN_user_item_ratio   : pd.DataFrame,  # ← df_in_Sell_In_User_Item_GI_Ratio
    out_version                 : str,
    **kwargs
) -> pd.DataFrame:
    measures = [
        COL_SIN_USER_ITEM_GI_LT, COL_SIN_USER_ITEM_GI_W7, COL_SIN_USER_ITEM_GI_W6, COL_SIN_USER_ITEM_GI_W5,
        COL_SIN_USER_ITEM_GI_W4, COL_SIN_USER_ITEM_GI_W3, COL_SIN_USER_ITEM_GI_W2, COL_SIN_USER_ITEM_GI_W1, COL_SIN_USER_ITEM_GI_W0
    ]
    grid = _step07_expand_with_week_and_std1(base_asn_delta_estore, df_in_Time, df_in_Item_Master)
    mapped = _step07_apply_ratio(grid, df_in_SIN_user_item_ratio, measures)
    return _step07_finalize(mapped, measures, out_version)
# ─────────────────────────────────────────────────────────────────────────────
# 7-6) S/In Issue Item GI Ratio
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step07_06_build_sin_issue_item_gi_ratio(
    base_asn_delta_estore        : pd.DataFrame,
    df_in_Time                   : pd.DataFrame,
    df_in_Item_Master            : pd.DataFrame,
    df_in_SIN_issue_item_ratio   : pd.DataFrame,  # ← df_in_Sell_In_Issue_Item_GI_Ratio
    out_version                  : str,
    **kwargs
) -> pd.DataFrame:
    measures = [
        COL_SIN_ISSUE_ITEM_GI_LT, COL_SIN_ISSUE_ITEM_GI_W7, COL_SIN_ISSUE_ITEM_GI_W6, COL_SIN_ISSUE_ITEM_GI_W5,
        COL_SIN_ISSUE_ITEM_GI_W4, COL_SIN_ISSUE_ITEM_GI_W3, COL_SIN_ISSUE_ITEM_GI_W2, COL_SIN_ISSUE_ITEM_GI_W1, COL_SIN_ISSUE_ITEM_GI_W0
    ]
    grid = _step07_expand_with_week_and_std1(base_asn_delta_estore, df_in_Time, df_in_Item_Master)
    mapped = _step07_apply_ratio(grid, df_in_SIN_issue_item_ratio, measures)
    return _step07_finalize(mapped, measures, out_version)


# ─────────────────────────────────────────────────────────────────────────────
# 공통: base(ShipTo×Item×Location) × Time.[Week] 그리드 생성 후, 지정된 측정치를 모두 0 으로 채움
# ─────────────────────────────────────────────────────────────────────────────
def _step08_expand_zero(
    base_estore: pd.DataFrame,     # Step7-0 결과 (E-Store 대상) : [ShipTo, Item, Location]
    df_in_Time : pd.DataFrame,     # [Time.[Week]]
    out_version: str,              # Version (예: 'CWV_DP')
    measure_cols: list[str],       # 생성할 측정치 목록
) -> pd.DataFrame:    # 결과 컬럼 스키마
    out_cols = [COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK] + measure_cols

    # 방어로직: 비어있으면 빈 DF 반환(스키마 유지)
    if base_estore is None or base_estore.empty or df_in_Time is None or df_in_Time.empty:
        return pd.DataFrame(columns=out_cols)

    # 필요한 컬럼 확인
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION):
        if c not in base_estore.columns:
            return pd.DataFrame(columns=out_cols)
    if COL_WEEK not in df_in_Time.columns:
        return pd.DataFrame(columns=out_cols)

    # 중복 제거 + 카테고리화(메모리/성능 안정)
    base = (base_estore[[COL_SHIP_TO, COL_ITEM, COL_LOCATION]]
            .dropna()
            .drop_duplicates()
            .copy(deep=False))
    for c in (COL_SHIP_TO, COL_ITEM, COL_LOCATION):
        base[c] = base[c].astype('category')

    # 주차 목록 준비(그대로 사용: 202506A/202509B 같은 Suffix 유지)
    wk = (df_in_Time[[COL_WEEK]]
          .dropna()
          .drop_duplicates()
          .copy(deep=False))
    wk[COL_WEEK] = wk[COL_WEEK].astype('category')

    # Cross-Join: base × week
    base['_k'] = 1
    wk['_k']   = 1
    grid = (base.merge(wk, on='_k', how='left')
                 .drop(columns=['_k'])
                 .reset_index(drop=True))

    # 측정치 0으로 채우고 int32로 캐스팅
    grid[measure_cols] = 0
    grid[measure_cols] = grid[measure_cols].astype('int32')

    # Version 추가 및 타입 정리
    grid.insert(0, COL_VERSION, out_version)
    for c in (COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK):
        grid[c] = grid[c].astype('category')

    # 컬럼 순서 정리
    grid = grid[[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_LOCATION, COL_WEEK] + measure_cols] \
             .drop_duplicates(ignore_index=True)

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Step 8-2) S/In User Modify GI Ratio (전부 0으로 생성)
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step08_02_build_sin_user_modify_gi_ratio(
    df_step07_00_base_estore: pd.DataFrame,  # ← Step7-0 결과 (E-Store 대상)
    df_in_Time: pd.DataFrame,                # ← df_in_Time (Time.[Week])
    Version: str                             # ← 예: 'CWV_DP'
) -> pd.DataFrame:
    measure_cols = [
        'S/In User Modify GI Ratio(Long Tail)',
        'S/In User Modify GI Ratio(W+7)',
        'S/In User Modify GI Ratio(W+6)',
        'S/In User Modify GI Ratio(W+5)',
        'S/In User Modify GI Ratio(W+4)',
        'S/In User Modify GI Ratio(W+3)',
        'S/In User Modify GI Ratio(W+2)',
        'S/In User Modify GI Ratio(W+1)',
        'S/In User Modify GI Ratio(W+0)',
    ]
    return _step08_expand_zero(df_step07_00_base_estore, df_in_Time, Version, measure_cols)

# ─────────────────────────────────────────────────────────────────────────────
# Step 8-3) S/In Issue Modify GI Ratio (전부 0으로 생성, 이름만 다름)
# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step08_03_build_sin_issue_modify_gi_ratio(
    df_step07_00_base_estore: pd.DataFrame,  # ← Step7-0 결과 (E-Store 대상)
    df_in_Time: pd.DataFrame,                # ← df_in_Time (Time.[Week])
    Version: str                             # ← 예: 'CWV_DP'
) -> pd.DataFrame:
    measure_cols = [
        'S/In Issue Modify GI Ratio(Long Tail)',
        'S/In Issue Modify GI Ratio(W+7)',
        'S/In Issue Modify GI Ratio(W+6)',
        'S/In Issue Modify GI Ratio(W+5)',
        'S/In Issue Modify GI Ratio(W+4)',
        'S/In Issue Modify GI Ratio(W+3)',
        'S/In Issue Modify GI Ratio(W+2)',
        'S/In Issue Modify GI Ratio(W+1)',
        'S/In Issue Modify GI Ratio(W+0)',
    ]
    return _step08_expand_zero(df_step07_00_base_estore, df_in_Time, Version, measure_cols)

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
            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------

            input_folder_name  = str_instance           
            output_folder_name = str_instance
            
            # ------
            str_input_dir = f'Input/{input_folder_name}'
            # str_input_dir = f'Input/{input_folder_name}/step5'
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
        # logger.Note(p_note=f'CurrentPartialWeek : {CurrentPartialWeek}', p_log_level=LOG_LEVEL.debug())


        ############ To do : 여기 아래에 Step Function 들을 Call 하는 코드 구현. ########
        # 예시
        # ################################################################################################################
        # # Step 00 – Ship-To 차원 LUT 구축
        # ################################################################################################################
        # dict_log = {
        #     'p_step_no': 00,
        #     'p_step_desc': 'Step 00 – load Ship-To dimension LUT'
        # }
        # df_fn_shipto_dim = step00_load_shipto_dimension(
        #     input_dataframes[STR_DF_DIM],
        #     **dict_log
        # )
        # fn_log_dataframe(df_fn_shipto_dim, f'step00_df_fn_shipto_dim')

        ################################################################################################################
        # Step 1-1) Sales Product ASN Delta 전처리
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 11,
            'p_step_desc': 'Step 1-1) Sales Product ASN Delta 전처리',
            'p_df_name'  : None
        }
        df_step01_01_asn_delta = fn_step01_01_preprocess_sales_product_asn_delta(
            input_dataframes[DF_IN_SALES_PRODUCT_ASN_DELTA],
            **dict_log
        )
        # 필요 시 중간결과 로그/CSV 출력은 데코레이터와 fn_log_dataframe 에서 처리됩니다.
        fn_log_dataframe(df_step01_01_asn_delta, 'df_step01_01_asn_delta')

        ################################################################################################################
        # Step 1-2) Sales Product ASN 전처리
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 12,
            'p_step_desc': 'Step 1-2) Sales Product ASN 전처리',
            'p_df_name'  : None
        }
        df_step01_02_asn_base_filtered = fn_step01_02_preprocess_sales_product_asn(
            input_dataframes[DF_IN_SALES_PRODUCT_ASN],     # 기본 ASN (Input 7)
            df_step01_01_asn_delta,                        # Step 1-1 결과
            **dict_log
        )
        fn_log_dataframe(df_step01_02_asn_base_filtered, 'df_step01_02_asn_base_filtered')

        ################################################################################################################
        # Step 1-3) Sales Product ASN 구성 (ShipTo × Item × Location)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 13,
            'p_step_desc': 'Step 1-3) Sales Product ASN 구성 (ShipTo×Item×Location)',
            'p_df_name'  : None
        }
        df_step01_03_asn_all = fn_step01_03_build_sales_product_asn(
            df_step01_01_asn_delta,                 # 1-1 결과
            df_step01_02_asn_base_filtered,         # 1-2 결과
            **dict_log
        )
        fn_log_dataframe(df_step01_03_asn_all, 'df_step01_03_asn_all')


        ################################################################################################################
        # Step 1-4) Sales Product ASN No Location
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 14,
            'p_step_desc': 'Step 1-4) Sales Product ASN No Location',
            'p_df_name'  : None
        }
        df_step01_04_asn_no_loc = fn_step01_04_sales_product_asn_no_location(
            df_step01_02_asn_base_filtered,   # Step 1-2 결과 DataFrame
            **dict_log
        )
        fn_log_dataframe(df_step01_04_asn_no_loc, 'df_step01_04_asn_no_loc')

        ################################################################################################################
        # Step 1-5) Sales Product ASN Delta No Location
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 15,
            'p_step_desc': 'Step 1-5) Sales Product ASN Delta No Location',
            'p_df_name'  : None
        }
        df_step01_05_asn_delta_no_loc = fn_step01_05_sales_product_asn_delta_no_location(
            df_step01_01_asn_delta,        # Step 1-1 결과
            df_step01_04_asn_no_loc,       # Step 1-4 결과
            **dict_log
        )
        fn_log_dataframe(df_step01_05_asn_delta_no_loc, 'df_step01_05_asn_delta_no_loc')
        
        ################################################################################################################
        # Step 1-6) Sales Product ASN output
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 16,
            'p_step_desc': 'Step 1-6) Sales Product ASN output',
            'p_df_name'  : None
        }
        Output_Sales_Product_ASN = fn_step01_06_output_sales_product_asn(
            input_dataframes[DF_IN_SALES_PRODUCT_ASN_DELTA],    # Input6
            Version,                                            # 예: 'CWV_DP'
            **dict_log
        )
        fn_log_dataframe(Output_Sales_Product_ASN, f'df_step01_06_{OUT_SALES_PRODUCT_ASN}')  # 필요시 CSV도 저장
        
        ################################################################################################################
        # Step 1-7) Sales Product ASN Delta output
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 17,
            'p_step_desc': 'Step 1-7) Sales Product ASN Delta output',
            'p_df_name'  : None
        }
        Output_Sales_Product_ASN_Delta = fn_step01_07_output_sales_product_asn_delta(
            input_dataframes[DF_IN_SALES_PRODUCT_ASN_DELTA],  # Input 6
            Version,                                          # 예: 'CWV_DP'
            **dict_log
        )

        # 필요 시 중간결과 로그/CSV 출력은 데코레이터와 fn_log_dataframe 에서 처리됩니다.
        fn_log_dataframe(Output_Sales_Product_ASN_Delta, f'df_step01_06_{OUT_SALES_PRODUCT_ASN_DELTA}')

        ##############################################################################################################
        # Step 2-1) Assortment 전처리
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 21,
            'p_step_desc': 'Step 2-1) Assortment 전처리',
            'p_df_name'  : None
        }
        df_step02_01_sin_assort, df_step02_01_sout_assort = fn_step02_01_preprocess_assortment(
            input_dataframes[DF_IN_SIN_ASSORTMENT],      # Input 8
            input_dataframes[DF_IN_SOUT_ASSORTMENT],     # Input 9
            df_step01_01_asn_delta,                      # Step 1-1 결과
            **dict_log
        )
        fn_log_dataframe(df_step02_01_sin_assort,  'df_step02_01_sin_assort')
        fn_log_dataframe(df_step02_01_sout_assort, 'df_step02_01_sout_assort')

        ################################################################################################################
        # Step 2-2) Assortment Measure 구성 (GC/AP2/AP1/Local)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 22,
            'p_step_desc': 'Step 2-2) Assortment Measure 구성 (GC/AP2/AP1/Local)',
            'p_df_name'  : None
        }
        sin_dict, sout_dict = fn_step02_02_build_assortments(
            df_step01_03_asn_all,                                # Step 1-3 결과
            input_dataframes[DF_IN_FORECAST_RULE],               # Forecast-Rule
            input_dataframes[DF_IN_ITEM_MASTER],                 # Item Master (Item Std1 ↔ Product Group)
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],      # Sales-Domain Dimension
            input_dataframes[DF_IN_SALES_DOMAIN_ESTORE],         # E-Store Ship-To
            **dict_log
        )
        # 로그/CSV (데코레이터 외 필요 시)
        for name, df in sin_dict.items():
            fn_log_dataframe(df, f'step02_02_{name}')
        for name, df in sout_dict.items():
            fn_log_dataframe(df, f'step02_02_{name}')

        # (선택) 변수로 펼치기 — 이후 단계에서 개별 접근이 필요할 경우
        df_step02_02_SIn_Assortment_GC    = sin_dict[STR_DF_OUT_SIN_GC]
        df_step02_02_SIn_Assortment_AP2   = sin_dict[STR_DF_OUT_SIN_AP2]
        df_step02_02_SIn_Assortment_AP1   = sin_dict[STR_DF_OUT_SIN_AP1]
        df_step02_02_SIn_Assortment_Local = sin_dict[STR_DF_OUT_SIN_LOCAL]

        df_step02_02_SOut_Assortment_GC    = sout_dict[STR_DF_OUT_SOUT_GC]
        df_step02_02_SOut_Assortment_AP2   = sout_dict[STR_DF_OUT_SOUT_AP2]
        df_step02_02_SOut_Assortment_AP1   = sout_dict[STR_DF_OUT_SOUT_AP1]
        df_step02_02_SOut_Assortment_Local = sout_dict[STR_DF_OUT_SOUT_LOCAL]

        ################################################################################################################
        # Step 2-3) S/In Assortment Measure 비교 (GC/AP2/AP1/Local)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 23,
            'p_step_desc': 'Step 2-3) S/In Assortment Measure 비교',
            'p_df_name'  : None
        }
        # Step 2-1 결과 (S/In 전처리) + Step 2-2 결과(S/In dict)를 사용
        df_step02_03_sin_dict = fn_step02_03_compare_sin_assortments(
            df_step02_01_sin_assort,    # ← fn_step02_01_preprocess_assortment 반환 1번째 (S/In)
            sin_dict,   # ← fn_step02_02_build_assortments 반환 dict 중 S/In dict
            **dict_log
        )
        # 필요시 로그
        for k, df_tmp in df_step02_03_sin_dict.items():
            fn_log_dataframe(df_tmp, f'step02_03_{k}')

        ################################################################################################################
        # Step 2-4) S/Out Assortment Measure 비교 (GC/AP2/AP1/Local)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 24,
            'p_step_desc': 'Step 2-4) S/Out Assortment Measure 비교',
            'p_df_name'  : None
        }
        df_step02_04_sout_dict = fn_step02_04_compare_sout_assortments(
            df_step02_01_sout_assort,   # ← fn_step02_01_preprocess_assortment 반환 2번째 (S/Out)
            sout_dict,                  # ← fn_step02_02_build_assortments 반환 dict 중 S/Out dict
            **dict_log
        )

        # 필요 시 로깅/CSV
        for k, df_tmp in df_step02_04_sout_dict.items():
            fn_log_dataframe(df_tmp, f'step02_04_{k}')

        ################################################################################################################
        # Step 2-5) Assortment Measure Output 구성
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 25,
            'p_step_desc': 'Step 2-5) Assortment Measure Output 구성',
            'p_df_name'  : None
        }
        df_step02_05_out = fn_step02_05_format_assortment_outputs(
            df_step02_03_sin_dict,   # Step 2-3 결과 dict
            df_step02_04_sout_dict,  # Step 2-4 결과 dict
            Version,                 # 전역 Version (예: 'CWV_DP')
            **dict_log
        )
        # 8개 Output으로 펼치기
        Output_SIn_Assortment_GC     = df_step02_05_out[OUT_SIN_ASSORTMENT_GC]
        Output_SIn_Assortment_AP2    = df_step02_05_out[OUT_SIN_ASSORTMENT_AP2]
        Output_SIn_Assortment_AP1    = df_step02_05_out[OUT_SIN_ASSORTMENT_AP1]
        Output_SIn_Assortment_Local  = df_step02_05_out[OUT_SIN_ASSORTMENT_LOCAL]
        Output_SOut_Assortment_GC    = df_step02_05_out[OUT_SOUT_ASSORTMENT_GC]
        Output_SOut_Assortment_AP2   = df_step02_05_out[OUT_SOUT_ASSORTMENT_AP2]
        Output_SOut_Assortment_AP1   = df_step02_05_out[OUT_SOUT_ASSORTMENT_AP1]
        Output_SOut_Assortment_Local = df_step02_05_out[OUT_SOUT_ASSORTMENT_LOCAL]

        # 로깅
        for name, df_out in df_step02_05_out.items():
            fn_log_dataframe(df_out, f'step02_05_{name}')           

        ################################################################################################################
        # Step 3) DSR 구성 (Location 제거)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 30,
            'p_step_desc': 'Step 3) DSR 구성 (Location 제거)',
            'p_df_name'  : None
        }
        df_step03_dsr_dict = fn_step03_build_dsr(
            df_step02_03_sin_dict,   # ← Step 2-3 결과(dict)
            df_step02_04_sout_dict,  # ← Step 2-4 결과(dict)
            Version,                  # ← 예: 'CWV_DP'
            **dict_log
        )
        # 필요시 로그/CSV
        for name, df_out in df_step03_dsr_dict.items():
            fn_log_dataframe(df_out, f'step03_{name}')

        # 8개 Output으로 펼치기
        Output_SIn_DSR_GC    = df_step03_dsr_dict[OUT_SIN_DSR_GC]
        Output_SIn_DSR_AP2   = df_step03_dsr_dict[OUT_SIN_DSR_AP2]
        Output_SIn_DSR_AP1   = df_step03_dsr_dict[OUT_SIN_DSR_AP1]
        Output_SIn_DSR_Local = df_step03_dsr_dict[OUT_SIN_DSR_LOCAL]

        Output_SOut_DSR_GC    = df_step03_dsr_dict[OUT_SOUT_DSR_GC]
        Output_SOut_DSR_AP2   = df_step03_dsr_dict[OUT_SOUT_DSR_AP2]
        Output_SOut_DSR_AP1   = df_step03_dsr_dict[OUT_SOUT_DSR_AP1]
        Output_SOut_DSR_Local = df_step03_dsr_dict[OUT_SOUT_DSR_LOCAL]


        ################################################################################################################
        # Step 4) FCST 0 값 데이터 생성 (S/In · S/Out · New Model · Flooring · BO)
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 41,
            'p_step_desc': 'Step 4) FCST 0 값 데이터 생성',
            'p_df_name'  : None
        }
        dict_step04 = fn_step04_build_zero_fcst(
            Version,                                        # out_version
            input_dataframes[DF_IN_TIME],                   # df_time    # ─ S/In · S/Out Assortment 조합 (Step 2-5 Output) ─
            Output_SIn_Assortment_GC,                       # df_out_sin_gc
            Output_SIn_Assortment_AP2,                      # df_out_sin_ap2
            Output_SIn_Assortment_AP1,                      # df_out_sin_ap1
            Output_SIn_Assortment_Local,                    # df_out_sin_local

            Output_SOut_Assortment_GC,                      # df_out_sout_gc
            Output_SOut_Assortment_AP2,                     # df_out_sout_ap2
            Output_SOut_Assortment_AP1,                     # df_out_sout_ap1
            Output_SOut_Assortment_Local,                   # df_out_sout_local

            # ─ ASN/Item 마스터 참조 ─
            df_step01_03_asn_all,                           # Step 1-3 결과 (Loc ASN)
            df_step01_05_asn_delta_no_loc,                  # Step 1-5 결과 (No-Location ASN)
            input_dataframes[DF_IN_ITEM_MASTER],            # Item Master (GBM=VD 필터용)
            input_dataframes[DF_IN_ITEM_MASTER_LED_SIGNAGE],# LED/SIGNAGE 대상 Item 목록
            **dict_log
        )

        # ─ 반환 11개 DF 변수로 풀기 ─
        df_output_Sell_In_FCST_GI_AP1    = dict_step04[DF_OUT_SIN_FCST_AP1]
        df_output_Sell_In_FCST_GI_AP2    = dict_step04[DF_OUT_SIN_FCST_AP2]
        df_output_Sell_In_FCST_GI_GC     = dict_step04[DF_OUT_SIN_FCST_GC]
        df_output_Sell_In_FCST_GI_Local  = dict_step04[DF_OUT_SIN_FCST_LOCAL]

        df_output_Sell_Out_FCST_AP1      = dict_step04[DF_OUT_SOUT_FCST_AP1]
        df_output_Sell_Out_FCST_AP2      = dict_step04[DF_OUT_SOUT_FCST_AP2]
        df_output_Sell_Out_FCST_GC       = dict_step04[DF_OUT_SOUT_FCST_GC]
        df_output_Sell_Out_FCST_Local    = dict_step04[DF_OUT_SOUT_FCST_LOCAL]

        df_output_Sell_In_FCST_GI_New_Model = dict_step04[DF_OUT_SIN_FCST_NEW_MODEL]
        df_output_Flooring_FCST             = dict_step04[DF_OUT_FLOORING_FCST]
        df_output_BO_FCST                   = dict_step04[DF_OUT_BO_FCST]

        # ─ 필요 시 로그/CSV 출력 ─
        for name, df_out in dict_step04.items():
            fn_log_dataframe(df_out, f'step04_{name}')

        # ─────────────────────────────────────────────────────────────────────────────
        # Step 5) Estimated Price Local — 메인 호출 블록
        #   - df_in_Time 에 Time.[Month] 가 없으면 임시 파생(PartialWeek[:6])로 보강
        #     * 캘린더가 특수 매핑이면, 실제 Time LUT에 Month를 넣어서 주는 것을 권장
        # ─────────────────────────────────────────────────────────────────────────────
        dict_log = {
            'p_step_no'  : 50,
            'p_step_desc': 'Step 5) Estimated Price Local 데이터 생성',
            'p_df_name'  : None
        }

        # 2) 함수 호출
        df_output_Estimated_Price_Local = fn_step05_build_estimated_price_local(
            df_step01_05_asn_delta_no_loc,                  # Step 1-5 결과 (No-Location)
            input_dataframes[DF_IN_TIME],                   # Time (PartialWeek + Month 必)
            input_dataframes[DF_IN_ESTIMATED_PRICE],         # Estimated Price (Modify/Local)
            input_dataframes[DF_IN_ACTION_PLAN_PRICE],       # Action Plan Price (USD, by Month)
            input_dataframes[DF_IN_EXCHANGE_RATE_LOCAL],     # Exchange Rate (by Std3, PartialWeek)
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],  # Sales-Domain Dimension (ShipTo → Std3)
            Version,                                         # 전역 Version (예: 'CWV_DP')
            **dict_log
        )

        # 3) 로그/CSV (필요 시)
        fn_log_dataframe(df_output_Estimated_Price_Local, f'step05_{DF_OUT_ESTIMATED_PRICE_LOCAL}')
    
        ##############################################################################################################
        # Step 6-1) S/In FCST(GI) Split Ratio_AP1 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 61,
            'p_step_desc': 'Step 6-1) S/In FCST(GI) Split Ratio_AP1 생성',
            'p_df_name'  : None
        }
        df_output_Sell_In_FCST_GI_Split_Ratio_AP1 = fn_step06_01_build_sin_split_ratio_ap1(
            Output_SIn_Assortment_AP1,                         # ← Step2-5 결과: Output_SIn_Assortment_AP1
            input_dataframes[DF_IN_TIME],                      # ← df_in_Time
            input_dataframes[DF_IN_SIN_SPLIT_AP1],             # ← df_in_Sell_In_FCST_GI_Split_Ratio_AP1
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],    # ← df_in_Sales_Domain_Dimension (ShipTo→Std2)
            input_dataframes[DF_IN_ITEM_MASTER],               # ← df_in_Item_Master (Item→Item Std1)
            Version,                                           # ← 예: 'CWV_DP'
            **dict_log
        )

        # 필요 시 로그/CSV
        fn_log_dataframe(df_output_Sell_In_FCST_GI_Split_Ratio_AP1, f'step06_01_{DF_OUT_SIN_SPLIT_AP1}')
    
        ##############################################################################################################
        # Step 6-2) S/In FCST(GI) Split Ratio_AP2 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 62,
            'p_step_desc': 'Step 6-2) S/In FCST(GI) Split Ratio_AP2 생성',
            'p_df_name'  : None
        }
        df_output_Sell_In_FCST_GI_Split_Ratio_AP2 = fn_step06_02_build_sin_split_ratio_ap2(
            Output_SIn_Assortment_AP2,                         # ← Step2-5 결과: Output_SIn_Assortment_AP2
            input_dataframes[DF_IN_TIME],                      # ← df_in_Time
            input_dataframes[DF_IN_SIN_SPLIT_AP2],             # ← df_in_Sell_In_FCST_GI_Split_Ratio_AP2
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],    # ← df_in_Sales_Domain_Dimension (ShipTo→Std2)
            input_dataframes[DF_IN_ITEM_MASTER],               # ← df_in_Item_Master (Item→Item Std1)
            Version,                                           # ← 예: 'CWV_DP'
            **dict_log
        )
        # 필요 시 로그/CSV
        fn_log_dataframe(df_output_Sell_In_FCST_GI_Split_Ratio_AP2, f'step06_02_{DF_OUT_SIN_SPLIT_AP2}')

        ##############################################################################################################
        # Step 6-3) S/In FCST(GI) Split Ratio_GC 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 63,
            'p_step_desc': 'Step 6-3) S/In FCST(GI) Split Ratio_GC 생성',
            'p_df_name'  : None
        }
        df_output_Sell_In_FCST_GI_Split_Ratio_GC = fn_step06_03_build_sin_split_ratio_gc(
            Output_SIn_Assortment_GC,                         # ← Step2-5 결과: Output_SIn_Assortment_GC
            input_dataframes[DF_IN_TIME],                     # ← df_in_Time
            input_dataframes[DF_IN_SIN_SPLIT_GC],             # ← df_in_Sell_In_FCST_GI_Split_Ratio_GC
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],   # ← df_in_Sales_Domain_Dimension (ShipTo→Std2)
            input_dataframes[DF_IN_ITEM_MASTER],              # ← df_in_Item_Master (Item→Item Std1)
            Version,                                          # ← 예: 'CWV_DP'
            **dict_log
        )
        
        # 필요 시 로그/CSV
        fn_log_dataframe(df_output_Sell_In_FCST_GI_Split_Ratio_GC, f'step06_03_{DF_OUT_SIN_SPLIT_GC}')

        ##############################################################################################################
        # Step 6-4) S/In FCST(GI) Split Ratio_Local 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 64,
            'p_step_desc': 'Step 6-4) S/In FCST(GI) Split Ratio_Local 생성',
            'p_df_name'  : None
        }
        df_output_Sell_In_FCST_GI_Split_Ratio_Local = fn_step06_04_build_sin_split_ratio_local(
            Output_SIn_Assortment_Local,                     # ← Step2-5 결과: Output_SIn_Assortment_Local
            input_dataframes[DF_IN_TIME],                    # ← df_in_Time
            input_dataframes[DF_IN_SIN_SPLIT_LOCAL],         # ← df_in_Sell_In_FCST_GI_Split_Ratio_Local
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],  # ← df_in_Sales_Domain_Dimension (ShipTo→Std2)
            input_dataframes[DF_IN_ITEM_MASTER],             # ← df_in_Item_Master (Item→Item Std1)
            Version,                                         # ← 예: 'CWV_DP'
            **dict_log
        )
        # 필요 시 로그/CSV
        fn_log_dataframe(df_output_Sell_In_FCST_GI_Split_Ratio_Local, f'step06_04_{DF_OUT_SIN_SPLIT_LOCAL}')

        # ============================
        # Step 6-5 ~ 6-8: Main 호출블록
        # ============================##############################################################################################################
        # Step 6-5) S/Out FCST Split Ratio_AP1 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 65,
            'p_step_desc': 'Step 6-5) S/Out FCST Split Ratio_AP1 생성',
            'p_df_name'  : None
        }
        df_output_Sell_Out_FCST_Split_Ratio_AP1 = fn_step06_05_build_sout_split_ratio_ap1(
            Output_SOut_Assortment_AP1,                      # ← Step2-5 결과
            input_dataframes[DF_IN_TIME],                    # ← df_in_Time
            input_dataframes[DF_IN_SOUT_SPLIT_AP1],          # ← df_in_Sell_Out_FCST_Split_Ratio_AP1
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],  # ← ShipTo→Std2
            input_dataframes[DF_IN_ITEM_MASTER],             # ← Item→Item Std1
            Version,                                         # ← 'CWV_DP' 등
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_Out_FCST_Split_Ratio_AP1, f'step06_05_{DF_OUT_SOUT_SPLIT_AP1}')


        ##############################################################################################################
        # Step 6-6) S/Out FCST Split Ratio_AP2 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 66,
            'p_step_desc': 'Step 6-6) S/Out FCST Split Ratio_AP2 생성',
            'p_df_name'  : None
        }
        df_output_Sell_Out_FCST_Split_Ratio_AP2 = fn_step06_06_build_sout_split_ratio_ap2(
            Output_SOut_Assortment_AP2,
            input_dataframes[DF_IN_TIME],
            input_dataframes[DF_IN_SOUT_SPLIT_AP2],
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],
            input_dataframes[DF_IN_ITEM_MASTER],
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_Out_FCST_Split_Ratio_AP2, f'step06_06_{DF_OUT_SOUT_SPLIT_AP2}')


        ##############################################################################################################
        # Step 6-7) S/Out FCST Split Ratio_GC 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 67,
            'p_step_desc': 'Step 6-7) S/Out FCST Split Ratio_GC 생성',
            'p_df_name'  : None
        }
        df_output_Sell_Out_FCST_Split_Ratio_GC = fn_step06_07_build_sout_split_ratio_gc(
            Output_SOut_Assortment_GC,
            input_dataframes[DF_IN_TIME],
            input_dataframes[DF_IN_SOUT_SPLIT_GC],
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],
            input_dataframes[DF_IN_ITEM_MASTER],
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_Out_FCST_Split_Ratio_GC, f'step06_07_{DF_OUT_SOUT_SPLIT_GC}')


        ##############################################################################################################
        # Step 6-8) S/Out FCST Split Ratio_Local 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 68,
            'p_step_desc': 'Step 6-8) S/Out FCST Split Ratio_Local 생성',
            'p_df_name'  : None
        }
        df_output_Sell_Out_FCST_Split_Ratio_Local = fn_step06_08_build_sout_split_ratio_local(
            Output_SOut_Assortment_Local,
            input_dataframes[DF_IN_TIME],
            input_dataframes[DF_IN_SOUT_SPLIT_LOCAL],
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],
            input_dataframes[DF_IN_ITEM_MASTER],
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_Out_FCST_Split_Ratio_Local, f'step06_08_{DF_OUT_SOUT_SPLIT_LOCAL}')

        ##############################################################################################################
        # Step 6-9) S/In Stretch Plan Assortment 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 69,
            'p_step_desc': 'Step 6-9) S/In Stretch Plan Assortment 생성',
            'p_df_name'  : None
        }
        df_output_Sell_In_Stretch_Plan_Assortment = fn_step06_09_build_sin_stretch_plan_assort(
            df_step01_05_asn_delta_no_loc,                 # ← Step 1-5 결과
            input_dataframes[DF_IN_SIN_STRETCH_ASSORT],    # ← df_in_Sell_In_Stretch_Plan_Assortment
            Version,                                       # ← 예: 'CWV_DP'
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_In_Stretch_Plan_Assortment, f'step06_09_{DF_OUT_SIN_STRETCH_ASSORT}')
    
        # =========================
        # Step 6-10) Main 호출 블록 (수정版)
        #  - Step 1-5 결과(df_step01_05_asn_delta_no_loc)를 입력으로 사용!
        # =========================
        ##############################################################################################################
        # Step 6-10) S/In Stretch Plan Split Ratio 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 610,
            'p_step_desc': 'Step 6-10) S/In Stretch Plan Split Ratio 생성',
            'p_df_name'  : None
        }
        df_output_Sell_In_Stretch_Plan_Split_Ratio = fn_step06_10_build_sin_stretch_plan_split_ratio(
            df_step01_05_asn_delta_no_loc,                 # ← ★ Step 1-5 결과 (No-Location) 사용
            input_dataframes[DF_IN_TIME],                  # ← df_in_Time (Partial Week)
            input_dataframes[DF_IN_SIN_STRETCH_SPLIT],     # ← df_in_Sell_In_Stretch_Plan_Split_Ratio
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],# ← ShipTo→Std2
            input_dataframes[DF_IN_ITEM_MASTER],           # ← Item→Item Std1
            Version,                                       # ← 예: 'CWV_DP'
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_In_Stretch_Plan_Split_Ratio, f'step06_10_{DF_OUT_SIN_STRETCH_SPLIT}')

        # ##############################################################################################################
        # Step 6-11) S/Out Stretch Plan Assortment 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 611,
            'p_step_desc': 'Step 6-11) S/Out Stretch Plan Assortment 생성',
            'p_df_name'  : None
        }
        df_output_Sell_Out_Stretch_Plan_Assortment = fn_step06_11_build_sout_stretch_plan_assortment(
            df_step01_05_asn_delta_no_loc,                # ← Step 1-5 결과(No-Location)
            input_dataframes[DF_IN_SOUT_STRETCH_ASSORT],  # ← df_in_Sell_Out_Stretch_Plan_Assortment (override 있으면 제외)
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_Out_Stretch_Plan_Assortment, f'step06_11_{DF_OUT_SOUT_STRETCH_ASSORT}')

        ##############################################################################################################
        # Step 6-12) S/Out Stretch Plan Split Ratio 생성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 612,
            'p_step_desc': 'Step 6-12) S/Out Stretch Plan Split Ratio 생성',
            'p_df_name'  : None
        }
        df_output_Sell_Out_Stretch_Plan_Split_Ratio = fn_step06_12_build_sout_stretch_plan_split_ratio(
            df_step01_05_asn_delta_no_loc,                # ← Step 1-5 결과(No-Location), 이 중 ASN=='Y'만 사용
            input_dataframes[DF_IN_TIME],                 # ← df_in_Time (Partial Week)
            input_dataframes[DF_IN_SOUT_STRETCH_SPLIT],   # ← df_in_Sell_Out_Stretch_Plan_Split_Ratio
            input_dataframes[DF_IN_SALES_DOMAIN_DIMENSION],# ← ShipTo→Std2
            input_dataframes[DF_IN_ITEM_MASTER],          # ← Item→Item Std1
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_Out_Stretch_Plan_Split_Ratio, f'step06_12_{DF_OUT_SOUT_STRETCH_SPLIT}')
    
        # =========================================
        # Step 7) Main 호출 블록
        # =========================================##############################################################################################################
        # Step 7-0) ASN Delta (E-Store) 원본 구성
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 700,
            'p_step_desc': 'Step 7-0) ASN Delta (E-Store) 원본 구성',
            'p_df_name'  : None
        }
        df_step07_00_base_estore = fn_step07_00_prepare_asn_delta_base(
            df_step01_01_asn_delta,                         # ← Step 1-1 결과
            input_dataframes[DF_IN_SALES_DOMAIN_ESTORE],    # ← E-Store ShipTo
            **dict_log
        )
        fn_log_dataframe(df_step07_00_base_estore, 'step07_00_base_estore')

        ##############################################################################################################
        # Step 7-1) S/In User GI Ratio
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 701,
            'p_step_desc': 'Step 7-1) S/In User GI Ratio',
            'p_df_name'  : None
        }
        df_output_Sell_In_User_GI_Ratio = fn_step07_01_build_sin_user_gi_ratio(
            df_step07_00_base_estore,                  # ← Step 7-0 결과
            input_dataframes[DF_IN_TIME],              # ← df_in_Time (Week)
            input_dataframes[DF_IN_ITEM_MASTER],       # ← Item Master (Item→Item Std1)
            input_dataframes[DF_IN_SIN_USER_GI_RATIO], # ← 참조테이블
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_In_User_GI_Ratio, f'step07_01_{DF_OUT_SIN_USER_GI_RATIO}')

        ##############################################################################################################
        # Step 7-2) S/In Issue GI Ratio
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 702,
            'p_step_desc': 'Step 7-2) S/In Issue GI Ratio',
            'p_df_name'  : None
        }
        df_output_Sell_In_Issue_GI_Ratio = fn_step07_02_build_sin_issue_gi_ratio(
            df_step07_00_base_estore,
            input_dataframes[DF_IN_TIME],
            input_dataframes[DF_IN_ITEM_MASTER],
            input_dataframes[DF_IN_SIN_ISSUE_GI_RATIO],
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_In_Issue_GI_Ratio, f'step07_02_{DF_OUT_SIN_ISSUE_GI_RATIO}')

        ##############################################################################################################
        # Step 7-3) S/In Final GI Ratio
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 703,
            'p_step_desc': 'Step 7-3) S/In Final GI Ratio',
            'p_df_name'  : None
        }
        df_output_Sell_In_Final_GI_Ratio = fn_step07_03_build_sin_final_gi_ratio(
            df_step07_00_base_estore,
            input_dataframes[DF_IN_TIME],
            input_dataframes[DF_IN_ITEM_MASTER],
            input_dataframes[DF_IN_SIN_FINAL_GI_RATIO],
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_In_Final_GI_Ratio, f'step07_03_{DF_OUT_SIN_FINAL_GI_RATIO}')

        ##############################################################################################################
        # Step 7-4) S/In BestFit GI Ratio
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 704,
            'p_step_desc': 'Step 7-4) S/In BestFit GI Ratio',
            'p_df_name'  : None
        }
        df_output_Sell_In_BestFit_GI_Ratio = fn_step07_04_build_sin_bestfit_gi_ratio(
            df_step07_00_base_estore,
            input_dataframes[DF_IN_TIME],
            input_dataframes[DF_IN_ITEM_MASTER],
            input_dataframes[DF_IN_SIN_BESTFIT_GI_RATIO],
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_In_BestFit_GI_Ratio, f'step07_04_{DF_OUT_SIN_BESTFIT_GI_RATIO}')

        ##############################################################################################################
        # Step 7-5) S/In User Item GI Ratio
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 705,
            'p_step_desc': 'Step 7-5) S/In User Item GI Ratio',
            'p_df_name'  : None
        }
        df_output_Sell_In_User_Item_GI_Ratio = fn_step07_05_build_sin_user_item_gi_ratio(
            df_step07_00_base_estore,
            input_dataframes[DF_IN_TIME],
            input_dataframes[DF_IN_ITEM_MASTER],
            input_dataframes[DF_IN_SIN_USER_ITEM_GI_RATIO],
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_In_User_Item_GI_Ratio, f'step07_05_{DF_OUT_SIN_USER_ITEM_GI_RATIO}')

        ##############################################################################################################
        # Step 7-6) S/In Issue Item GI Ratio
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 706,
            'p_step_desc': 'Step 7-6) S/In Issue Item GI Ratio',
            'p_df_name'  : None
        }
        df_output_Sell_In_Issue_Item_GI_Ratio = fn_step07_06_build_sin_issue_item_gi_ratio(
            df_step07_00_base_estore,
            input_dataframes[DF_IN_TIME],
            input_dataframes[DF_IN_ITEM_MASTER],
            input_dataframes[DF_IN_SIN_ISSUE_ITEM_GI_RATIO],
            Version,
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_In_Issue_Item_GI_Ratio, f'step07_06_{DF_OUT_SIN_ISSUE_ITEM_GI_RATIO}')

        ##############################################################################################################
        # Step 8-2) S/In User Modify GI Ratio (전부 0 값으로 생성)
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 802,
            'p_step_desc': 'Step 8-2) S/In User Modify GI Ratio 생성 (0 고정)',
            'p_df_name'  : None
        }
        df_output_Sell_In_User_Modify_GI_Ratio = fn_step08_02_build_sin_user_modify_gi_ratio(
            df_step07_00_base_estore,             # ← Step7-0 결과 사용
            input_dataframes[DF_IN_TIME],         # ← df_in_Time (Time.[Week])
            Version,                              # ← 예: 'CWV_DP'
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_In_User_Modify_GI_Ratio, f'step08_02_{DF_OUT_SIN_USER_MODIFY_GI_RATIO}')
        
        ##############################################################################################################
        # Step 8-3) S/In Issue Modify GI Ratio (전부 0 값으로 생성)
        ##############################################################################################################
        dict_log = {
            'p_step_no'  : 803,
            'p_step_desc': 'Step 8-3) S/In Issue Modify GI Ratio 생성 (0 고정)',
            'p_df_name'  : None
        }
        df_output_Sell_In_Issue_Modify_GI_Ratio = fn_step08_03_build_sin_issue_modify_gi_ratio(
            df_step07_00_base_estore,             # ← Step7-0 결과 사용
            input_dataframes[DF_IN_TIME],         # ← df_in_Time (Time.[Week])
            Version,                              # ← 예: 'CWV_DP'
            **dict_log
        )
        fn_log_dataframe(df_output_Sell_In_Issue_Modify_GI_Ratio, f'step08_03_{DF_OUT_SIN_ISSUE_MODIFY_GI_RATIO}')
    
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


        logger.Finish()
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::') # 25.05.12 need warning Log by Logger Issue
        