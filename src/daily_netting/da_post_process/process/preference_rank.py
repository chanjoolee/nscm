
# ## **SHASCM.SP_FN_SOPROMISE_UPD_C_REG**

'''
* 변경이력
 - 2025.04.25 : CYCLETEST 에서만 실행되는 함수 calculate_actual_demand 수정 [데이터 검증 완료]
 - 2025.04.15 : 삼성업데이트 함수 upate_samsungdotcom 부분만 [데이터 검증 완료]

'''

# basic lib
import os
import sys
from typing import Tuple, Optional
from datetime import datetime, date, timedelta
import time
import inspect

# 3rd party lib
import numpy as np
import pandas as pd

# user defined lib
from daily_netting.da_post_process.constant.ods_constant import NettedDemandD as ND_      # EXP_SOPROMISE
from daily_netting.da_post_process.constant.dim_constant import Location as L              # MS_SITE
from daily_netting.da_post_process.constant.dim_constant import Item as I                  # Dim.Item
from daily_netting.da_post_process.constant.general_constant import AccountInfo as AI      # MST_ACNTINFO@SCMTF
from daily_netting.da_post_process.constant.da_constant import GI as GI                    # DYN_GI
from daily_netting.da_post_process.constant.da_constant import Shipment as S               # DYN_SHIPMENT
from daily_netting.da_post_process.utils import NSCMCommon as nscm

LOGGER = nscm.G_Logger
DF = pd.DataFrame

# 전역변수, set_preference_rank_env를 통해 동적으로 값 설정
V_TYPE = None
V_PLANWEEK = None
V_PLANID = None
V_EFFSTARTDATE = None
V_EFFENDDATE = None
V_EFFSTARTDATE_STR = None
V_EFFENDDATE_STR = None
V_CLOSEDATE = None
find_priority_position = None


def set_preference_rank_env(accessor: object) -> None:
    '''동적으로 전역변수를 설정하기 위한 함수'''
    global V_TYPE, V_PLANWEEK, V_PLANID, V_EFFSTARTDATE, V_EFFENDDATE, \
    V_EFFSTARTDATE_STR, V_EFFENDDATE_STR, V_CLOSEDATE, find_priority_position

    # plan 설정
    V_TYPE = accessor.plan_type
    V_PLANWEEK = accessor.plan_week
    V_PLANID = accessor.plan_id
    V_EFFSTARTDATE = accessor.start_date
    V_EFFENDDATE = accessor.end_date
    V_EFFSTARTDATE_STR = V_EFFSTARTDATE.strftime('%Y-%m-%d')
    V_EFFENDDATE_STR = V_EFFENDDATE.strftime('%Y-%m-%d')
    V_CLOSEDATE = accessor.close_date

    # 자리수 확인 함수
    find_priority_position = accessor.find_priority_position

    # 함수 종료 메시지
    fx_name = inspect.currentframe().f_code.co_name     # 함수 이름 확인
    print(f"🐼 The function '{fx_name}' has completed its execution.")


def is_file_type_py():
    '''
    Check if the current file is a Python script (.py).

    This function determines whether the script is being executed as a 
    standalone Python file or in a Jupyter Notebook environment. 

    Returns:
        bool: True if the current file is a Python file (.py), 
              False otherwise (including Jupyter Notebook).
    '''

    fx_name = inspect.currentframe().f_code.co_name  # 현재 함수 이름 가져오기
    # 일반 Python 스크립트의 경우
    if '__file__' in locals():
        current_file = __file__
        result = os.path.splitext(current_file)[1] == '.py'  # .py 파일이면 True 반환
        print(f"🐼 The function '{fx_name}' has completed its execution. Result: {result}")
        return result
    # Jupyter Notebook 환경인지 확인
    elif 'ipykernel' in sys.modules:
        print(f"🐼 The function '{fx_name}' has completed its execution. Result: False")
        return False  # Jupyter Notebook은 .py 파일이 아니므로 False 반환
    else:
        print(f"🐼 The function '{fx_name}' has completed its execution. Result: False")
        return False  # 그 외의 경우는 False 반환
    

# start_pos, next_pos, total_length = find_priority_position('G_R001::1' 'PREFERENCERANK', 'D003')


def find_first_zero_or_negative_index(series):
    '''
    Get the index of the first zero or negative value in a pandas Series.

    Parameters:
    series (pd.Series): A pandas Series containing numerical values.

    Returns:
    int or float: The index of the first zero or negative value if it exists; 
                  otherwise, returns NaN.
    '''
    negative_index = series[series <= 0].index
    if not negative_index.empty:
        return negative_index[0]
    else:
        return np.nan


def adjust_qtypromised_C(df: pd.DataFrame) -> pd.DataFrame:
    '''
    V_REMAINQTY를 SOPROMISE의 QTYPROMISED에서 차감하는 함수
    '''
    sorted_df = df.sort_values(by=[ND_.DEMANDPRIORITY], ascending=[False])
    # sorted_df = df.sort_values(by=[ND_.DEMANDPRIORITY, ND_.SALESORDERID, ND_.SOLINENUM], ascending=[False, True, True])     # 데이터 검증용 !!!

    # 정렬 후 인덱스 재설정 ['index' 컬럼이 아님!!]
    sorted_df.reset_index(drop=True, inplace=True)
                                   
    # v_remainqty = sorted_df.at[0, 'V_REMAINQTY']
    v_remainqty = sorted_df.at[0, 'SALESQTY']
    print(f"■ v_remainqty = {v_remainqty}")

    cnt = 0
    for idx, row in sorted_df.iterrows():
        cnt = cnt + 1

        if v_remainqty > 0:
            v_remainqty = v_remainqty - int(row[ND_.QTYPROMISED])

            # remain_qty = int(row['REMAINQTY'])    # remain_qty - current_qty

            if v_remainqty >= 0:
                sorted_df.at[idx, ND_.QTYPROMISED] = 0
                sorted_df.at[idx, 'UPBY'] = str(sorted_df.at[idx, 'UPBY']) + '::' + 'SALES'
                print(f" 🥨 v_remainqty = {v_remainqty}")
                print(f"{row[ND_.SALESORDERID]}, {row[ND_.SOLINENUM]}, {v_remainqty}, {row[ND_.QTYPROMISED]}")
            else:
                sorted_df.at[idx, ND_.QTYPROMISED] = abs(v_remainqty)
                sorted_df.at[idx, 'UPBY'] = str(sorted_df.at[idx, 'UPBY']) + '::' + 'SALES'
                print(f" 🍄 v_remainqty = {v_remainqty}")
                print(f"{row[ND_.SALESORDERID]}, {row[ND_.SOLINENUM]}, {v_remainqty}, {row[ND_.QTYPROMISED]}")
                v_remainqty = 0

            # 업데이트 타임 및 정보 추가
            sorted_df.at[idx, 'UPDTTM'] =datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sorted_df.at[idx, 'UPBY'] = sorted_df.at[idx, 'UPBY'] + '::' + 'SALES'

    return sorted_df


def calculate_actual_demand(v_type: str, v_planid: str, v_planweek: str, df_exp_sopromise_src: DF,
                            df_dyn_gi: DF, df_dyn_shipment: DF, df_mst_site: DF):
    '''
    실적이 발생한 요구수량(DEMAND) = 요구수량(DEMAND) - 판매수량(SALES ) 처리 [주의: C PLAN만 수행]
    '''

    if v_type == 'CYCLETEST' and v_planid.endswith('C'):
        # 조건이 만족할 때 실행할 코드
        print("V_PLANID가 'C'로 끝나는 조건이 만족되었습니다.")


        ### STEP 2-1: 직거래 유형 필터링
        df_dyn_shipment_not = df_dyn_shipment[
            (df_dyn_shipment[S.SITEID].isin(df_mst_site.loc[df_mst_site[L.SHIPMENTTYPE] == 'DIR_D', L.SITEID]))
        ]

        ### STEP 2-2: df_dyn_gi에서 ITEM, SITEID 기준으로 제외
        df_dyn_gi_new = df_dyn_gi[
            ~(df_dyn_gi[GI.ITEM] + df_dyn_gi[GI.SITEID]).isin(df_dyn_shipment_not[S.ITEM] + df_dyn_shipment_not[S.SITEID])
        ]

        ### STEP 2-3: 병합하여 df_qty_sales 생성

        # df_dyn_gi_new에 대해 GROUP BY 및 SUM 적용
        df_sales_qty_gi = df_dyn_gi_new.groupby([GI.ITEM, GI.SITEID], as_index=False)[GI.GIQTY].sum()
        df_sales_qty_gi.rename(columns={GI.GIQTY: 'SALESQTY'}, inplace=True)

        # df_dyn_shipment에 대해 GROUP BY 및 SUM 적용
        df_sales_qty_shipment = df_dyn_shipment.groupby([S.ITEM, S.SITEID], as_index=False)[S.BILLINGQTY].sum()
        df_sales_qty_shipment.rename(columns={S.SITEID: GI.SITEID, S.BILLINGQTY: 'SALESQTY'}, inplace=True)

        # UNION ALL 적용
        df_sales_qty = pd.concat([df_sales_qty_gi, df_sales_qty_shipment], ignore_index=True)

        # HAVING SUM(SALESQTY) > 0 적용
        df_sales_qty = df_sales_qty.groupby([GI.ITEM, GI.SITEID], as_index=False)['SALESQTY'].sum()
        df_sales_qty = df_sales_qty[df_sales_qty['SALESQTY'].astype(int) > 0]


        ### STEP 3: 벡터라이제이션 이용

        # df_sopromise 생성 : [df_exp_sopromise_src, df_sales_qty] 병합
        df_sopromise = df_exp_sopromise_src.copy()
        df_sopromise = df_sopromise.merge(
            df_sales_qty[df_sales_qty['SALESQTY'].astype(int) > 0],
            left_on=[ND_.ITEMID, ND_.SITEID],
            right_on=[GI.ITEM, GI.SITEID],
            how='inner',
        )

        # df_sopromise 필터링
        df_sopromise = df_sopromise[df_sopromise[ND_.PLANID] == v_planid]
        df_sopromise = df_sopromise[df_sopromise[ND_.QTYPROMISED].astype(int) > 0]
        df_sopromise = df_sopromise[df_sopromise[ND_.PROMISEDDELDATE + '_WEEK'] == v_planweek]
        df_sopromise = df_sopromise[df_sopromise[ND_.SOLINENUM].astype(int) < 400]

        # df_sopromise 정렬 [ITEM, SITEID] 별로 DEMANDPRIORITY DESC 정렬
        df_sopromise.sort_values(
            by=[ND_.ITEMID, ND_.SITEID, ND_.DEMANDPRIORITY],
            ascending=[True, True, False],
            inplace=True,
        )

        # df_sopromise 필터링
        df_sopromise['SALESQTY'] = df_sopromise['SALESQTY'].astype(int)
        df_sopromise = df_sopromise[df_sopromise['SALESQTY'].astype(int) > 0]
        
        # 정렬 후 인덱스 재설정
        df_sopromise.reset_index(drop=True, inplace=True)

        if df_sopromise.shape[0] > 0:   # 해당 레코드가 있을때만,

            print(f"df_sopromise에 레코드가 {df_sopromise.shape[0]}개 검출되었습니다.")

            ####################################################################
            # 🎰 QTY 차감 처리: adjust_qtypromised_C

            # df_sopromise 정렬 [ITEM, SITEID] 별로 CUMSUM_QTY 누적 합산 (Cumulative Sum)
            df_sopromise['CUMSUM_QTY'] = df_sopromise.groupby([ND_.ITEMID, ND_.SITEID])[ND_.QTYPROMISED].cumsum()

            # REMAINQTY 계산
            df_sopromise['REMAINQTY'] = df_sopromise['SALESQTY'].astype(int) - df_sopromise['CUMSUM_QTY'].astype(int)

            # UPBY 초기화
            df_sopromise['UPBY'] = ''

            # QTY 차감 처리: adjust_qtypromised_C
            df_sopromise = df_sopromise.groupby([ND_.ITEMID, ND_.SITEID], group_keys=False).apply(adjust_qtypromised_C)
            ####################################################################


            ### UPDATE:  원본(df_exp_sopromise_src)에 [QTYPROMISED, UPBY]를 df_sopromise 의 ['QTYPROMISED_CALC', UPBY] 값으로 ('index' 컬럼 기준)
            df_exp_sopromise_src.set_index('index', inplace=True)
            df_sopromise.set_index('index', inplace=True)

            update_indice = df_sopromise.index
            df_exp_sopromise_src.loc[update_indice, ND_.QTYPROMISED] = df_sopromise.loc[update_indice, ND_.QTYPROMISED]
            df_exp_sopromise_src['UPBY'] = ''
            df_exp_sopromise_src.loc[update_indice, 'UPBY'] = df_sopromise.loc[update_indice, 'UPBY']

            df_exp_sopromise_src.reset_index(inplace=True)
        
        else:
            print(f"df_sopromise에 레코드가 {df_sopromise.shape[0]}개 검출되었습니다.")

    else:
        print("V_PLANID가 'C'로 끝나는 조건이 아닙니다.")

    return df_exp_sopromise_src #, df_sales_qty, df_sopromise


def upate_samsungdotcom(df_exp_sopromise_src, df_mst_acntinfo, df_vui_itemattb):
    '''
    삼성닷컴 DEMANDPRIORITY 업데이트 
    '''

    # 'index' 컬럼이 없는 경우 reset_index() 실행
    if 'index' not in df_exp_sopromise_src.columns:
        df_exp_sopromise_src.reset_index(inplace=True)

    ### STEP 1-1: SECTION 정보 추가 
    df_exp_sopromise_src_sub = df_exp_sopromise_src.copy().merge(
        df_vui_itemattb,
        left_on=[ND_.ITEMID],
        right_on=[I.ITEM],
        how='left',
    )

    ### 컬럼명 변경: SECTION
    ### df_exp_sopromise_src_sub.rename(columns={I.SECTION: 'SECTION'}, inplace=True)

    # FN_GETSECTION(A.ITEM) 적용: SECTION이 NULL 이면 '-' 로 처리
    df_exp_sopromise_src_sub[I.SECTION] = df_exp_sopromise_src_sub[I.SECTION].fillna('-')

    ### STEP 1-2: MST_ACNTINFO 조건 필터링
    df_mst_acntinfo_filtered = df_mst_acntinfo[
        (df_mst_acntinfo[AI.SECTION].isin(['DA', 'DAS']))
        & (df_mst_acntinfo[AI.CHANNELTYPE].isin(['ONLINE']))
        & (df_mst_acntinfo[AI.GPGNAME].isin(['COM', 'COM_SI', 'COM_DIR']))
    ]

    # SALESID, SECTION이 일치하는 행만 필터링
    df_exp_sopromise_src_sub = df_exp_sopromise_src_sub.merge(
        df_mst_acntinfo_filtered[[AI.SALESID, AI.SECTION]],
        left_on=[ND_.SALESID, I.SECTION],
        right_on=[AI.SALESID, AI.SECTION],
        how='inner',
    )

    ### STEP 1-3: DEMANDPRIORITY 관련 컬럼 업데이트
    # DEMANDPRIORITY 가 NULL이 아니고 첫 번째 자리 숫자가 5보다 큰 경우에만 업데이트
    df_exp_sopromise_src_sub = df_exp_sopromise_src_sub[
        (df_exp_sopromise_src_sub[ND_.DEMANDPRIORITY].notna())  
        & (df_exp_sopromise_src_sub[ND_.DEMANDPRIORITY].str[0].astype(int) > 5)
    ]


    # DEMANDPRIORITY
    # 해당 조건으로 필터링된 데이터가 있는 경우에만 실행.
    if df_exp_sopromise_src_sub.shape[0] > 0:
        # DA의 find_priority_position 가 개발시점에서 정의되지 않아서 사용하지 못함.

        # DEMANDPRIORITY 업데이트
        df_exp_sopromise_src_sub[ND_.DEMANDPRIORITY] = '5' + df_exp_sopromise_src_sub[ND_.DEMANDPRIORITY].str[1:]

        # GLOBALPRIORITY 업데이트
        df_exp_sopromise_src_sub[ND_.GLOBALPRIORITY] = np.where(
            df_exp_sopromise_src_sub[ND_.GLOBALPRIORITY].fillna('0') == '0',
            np.nan,
            '5' + df_exp_sopromise_src_sub[ND_.GLOBALPRIORITY].str[1:]
        )

        # LOCALPRIORITY 업데이트 
        df_exp_sopromise_src_sub[ND_.LOCALPRIORITY] = np.where(
            df_exp_sopromise_src_sub[ND_.LOCALPRIORITY].fillna('0') == 0,
            np.nan,
            '5' + df_exp_sopromise_src_sub[ND_.LOCALPRIORITY].str[1:]
        )

        # PREFERENCERANK 업데이트
        df_exp_sopromise_src_sub[ND_.PREFERENCERANK] = np.where(
            df_exp_sopromise_src_sub[ND_.PREFERENCERANK].fillna('-') == '-',
            np.nan,
            '5'
        )

        # UPBY 업데이트
        # df_exp_sopromise_src_sub['UPBY'] = 'CHG_DEMANDPRIORITY'

        ### UPDATE:  원본(df_exp_sopromise_src)에 [DEMANDPRIORITY, GLOBALPRIORITY, LOCALPRIORITY, PREFERENCERANK, UPBY]를 df_exp_sopromise_src_sub 의 값으로 ('index' 컬럼 기준)
        df_exp_sopromise_src.set_index('index', inplace=True)
        df_exp_sopromise_src_sub.set_index('index', inplace=True)

        update_indice = df_exp_sopromise_src_sub.index
        df_exp_sopromise_src.loc[update_indice, ND_.DEMANDPRIORITY] = df_exp_sopromise_src_sub.loc[update_indice, ND_.DEMANDPRIORITY]
        df_exp_sopromise_src.loc[update_indice, ND_.GLOBALPRIORITY] = df_exp_sopromise_src_sub.loc[update_indice, ND_.GLOBALPRIORITY]
        df_exp_sopromise_src.loc[update_indice, ND_.LOCALPRIORITY] = df_exp_sopromise_src_sub.loc[update_indice, ND_.LOCALPRIORITY]
        df_exp_sopromise_src.loc[update_indice, ND_.PREFERENCERANK] = df_exp_sopromise_src_sub.loc[update_indice, ND_.PREFERENCERANK]
        # df_exp_sopromise_src.loc[update_indice, 'UPBY'] = df_exp_sopromise_src_sub.loc[update_indice, 'UPBY']

        df_exp_sopromise_src.reset_index(inplace=True)

        print(f" {ND_.DEMANDPRIORITY} 업데이트가 실행되었습니다." )
    else:
        print(f" {ND_.DEMANDPRIORITY} 업데이트 대상이 검출되지 않았습니다." )
 
    return df_exp_sopromise_src


def generate_final_sopromise(v_planid, df_exp_sopromise_src):
    '''
    최종 SOPROMISE 생성 
    '''
    # STEP 1: df_exp_sopromise 초기화
    # LI_COLUMNS = [
    #     ND_.SOPROMISEID, ND_.PLANID, ND_.SALESORDERID, ND_.SOLINENUM, 
    #     ND_.ITEMID, ND_.QTYPROMISED, ND_.PROMISEDDELDATE, ND_.SITEID, ND_.SHIPTOID,
    #     ND_.SALESID, ND_.SALESLEVEL, ND_.DEMANDTYPERANK, ND_.WEEKRANK, ND_.CHANNELRANK,
    #     ND_.CUSTOMERRANK, ND_.PRODUCTRANK, ND_.DEMANDPRIORITY, ND_.TIEBREAK, ND_.GBM, 
    #     ND_.GLOBALPRIORITY, ND_.LOCALPRIORITY, ND_.BUSINESSTYPE,
    #     ND_.ROUTING_PRIORITY, ND_.NO_SPLIT, ND_.MAP_SATISFY_SS,
    #     ND_.PREALLOC_ATTRIBUTE, 
    #     ND_.BUILDAHEADTIME, ND_.TIMEUOM, ND_.AP2ID, ND_.GCID,
    #     ND_.MEASURETYPERANK, ND_.PREFERENCERANK, ND_.INITDTTM, ND_.INITBY,
    #     ND_.UPDTTM, ND_.UPBY, ND_.REASONCODE, 'LOCALID'
    # ]
    LI_COLUMNS = [
        ND_.SOPROMISEID, ND_.PLANID, ND_.SALESORDERID, ND_.SOLINENUM, 
        ND_.ITEMID, ND_.QTYPROMISED, ND_.PROMISEDDELDATE, ND_.SITEID, ND_.SHIPTOID,
        ND_.SALESID, ND_.SALESLEVEL, ND_.DEMANDTYPERANK, ND_.WEEKRANK, ND_.CHANNELRANK,
        ND_.CUSTOMERRANK, ND_.PRODUCTRANK, ND_.DEMANDPRIORITY, ND_.TIEBREAK,
        ND_.GLOBALPRIORITY, ND_.LOCALPRIORITY, ND_.BUSINESSTYPE,
        ND_.ROUTING_PRIORITY, ND_.NO_SPLIT, ND_.MAP_SATISFY_SS,
        ND_.PREALLOC_ATTRIBUTE, ND_.BUILDAHEADTIME, ND_.TIMEUOM, ND_.AP2ID, ND_.GCID,
        ND_.MEASURETYPERANK, ND_.PREFERENCERANK, ND_.REASONCODE, ND_.LOCALID,
    ]

    df_exp_sopromise = pd.DataFrame(columns=LI_COLUMNS)

    # STEP 2: PLANID = V_PLANID 필터링
    df_filtered = df_exp_sopromise_src[df_exp_sopromise_src[ND_.PLANID] ==  v_planid]

    # STEP 3: DENSE_RANK() OVER (ORDER BY A.DEMANDPRIORITY)
    # df_filtered = df_filtered.astype({ND_.DEMANDPRIORITY: 'int64'})
    df_filtered.sort_values(by=[ND_.DEMANDPRIORITY], ascending=True, inplace=True)
    # df_filtered[ND_.DEMANDPRIORITY] = df_filtered.groupby([ND_.DEMANDPRIORITY]).cumcount() + 1
    df_filtered[ND_.DEMANDPRIORITY] = df_filtered[ND_.DEMANDPRIORITY].rank(method='dense')    #.astype('int64')

    # STEP 4: BUILDAHEADTIME 고정값 처리
    df_filtered[ND_.BUILDAHEADTIME] = '0'

    # STEP 5: INSERT INTO 구현
    df_exp_sopromise = pd.concat([df_exp_sopromise, df_filtered], ignore_index=True)

    return df_exp_sopromise



################################
##### preference_rank main #####
################################
def do_preference_rank(
        df_exp_sopromise_src: DF, df_dyn_gi: DF, df_dyn_shipment: DF,
        df_mst_site: DF, df_mst_acntinfo: DF, df_vui_itemattb: DF, logger: LOGGER
    ):
    '''
    SHASCM.SP_FN_SOPROMISE_UPD_C_REG
    '''

    V_FILENAME = 'preference_rank.py'
    if is_file_type_py():
        V_FILENAME = {os.path.basename(__file__)}
    else:
        V_FILENAME = V_FILENAME


    # ⏲ 시간측정 시작
    start = time.time()
    print(f"🥑 START: {V_FILENAME}")
    logger.Note(f"Start {V_FILENAME}", 20)



    # reset_index():  원본 index 보존 목적
    df_exp_sopromise_src = df_exp_sopromise_src.reset_index()

    # # ### **Type Casting**
    # df_exp_sopromise_src = df_exp_sopromise_src.astype({
    #     ND_.QTYPROMISED : int,
    #     ND_.PROMISEDDELDATE : 'datetime64[ns]',
    #     # 'IS_PLAN_DATE' : 'int32',
    # })

    # 📆 PROMISEDDELDATE를 isoweek으로 변환 컬럼 추가
    # [df_BUF_SOPROMISE, df_BUF_SOPROMISE_LOCAL] PROMISEDDELDATE를 주차 포맷으로 변경
    isoCalDate = df_exp_sopromise_src[ND_.PROMISEDDELDATE].dt.isocalendar()
    df_exp_sopromise_src[ND_.PROMISEDDELDATE + '_WEEK'] = isoCalDate.year.astype(str) + isoCalDate.week.astype(str).str.zfill(2)


    #### 🎱 Apply buildaheadtime 
    logger.Step(1, f"Start: {'calculate_actual_demand'}")
    # CYCLETEST; _C 인 경우만 실행됨. 
    # df_exp_sopromise_src, df_sales_qty, df_sopromise = calculate_actual_demand(V_TYPE, V_PLANID, V_PLANWEEK, df_exp_sopromise_src, df_dyn_gi, df_dyn_shipment, df_mst_site)
    df_exp_sopromise_src = calculate_actual_demand(V_TYPE, V_PLANID, V_PLANWEEK, df_exp_sopromise_src, df_dyn_gi, df_dyn_shipment, df_mst_site)
    logger.Step(1, f"End: {'calculate_actual_demand'}")

    logger.Step(2, f"Start: {'upate_samsungdotcom'}")
    # 삼성닷컴 업데이트
    df_exp_sopromise_src = upate_samsungdotcom(df_exp_sopromise_src, df_mst_acntinfo, df_vui_itemattb)
    logger.Step(2, f"End: {'upate_samsungdotcom'}")

    logger.Step(3, f"Start: {'generate_final_sopromise'}")
    # 최종 SOPROMISE 생성 
    df_exp_sopromise = generate_final_sopromise(V_PLANID, df_exp_sopromise_src)
    logger.Step(3, f"End: {'generate_final_sopromise'}")

    # ⏲ 수행시간 출력
    print(f"🍤 FINISHED: {V_FILENAME}")
    print(f"⏲ {V_FILENAME}: Execution time: {time.time() - start:.5f} seconds")
    logger.Note(f"End {V_FILENAME}", 20)

    return df_exp_sopromise[ND_.LIST_COLUMN]