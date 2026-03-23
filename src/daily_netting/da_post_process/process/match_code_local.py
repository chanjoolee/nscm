# ## **SHASCM.SP_FN_SOPROMISE_LOCAL_REG**
### MOD @ 2025-04-16

# basic lib
import os
import sys
from datetime import datetime, timedelta
import time
import inspect

# 3rd party lib
import numpy as np
import pandas as pd

# user defined lib
from daily_netting.da_post_process.constant.da_constant import LocalSalesOrder as LSaO     # BUF_SALESORDERLINE_LOCAL
from daily_netting.da_post_process.constant.da_constant import LocalMeasure as LM          # EXP_LOCALMEASURE (@SCMTF)
from daily_netting.da_post_process.constant.da_constant import RegionASN as RASN           # MST_SDMREGIONASN
from daily_netting.da_post_process.constant.da_constant import DASellerMap as DASM         # MTA_SELLERMAP
from daily_netting.da_post_process.constant.da_constant import DAItemSellerMap as DAISM    # V_MTA_SELLERMAP
from daily_netting.da_post_process.constant.dim_constant import NettingSales as NS         # Dim.Netting Sales
from daily_netting.da_post_process.constant.dim_constant import Item as I                  # Dim.Item
from daily_netting.da_post_process.constant.ods_constant import NettedDemandD as ND_      # EXP_SOPROMISE
from daily_netting.da_post_process.utils import NSCMCommon as nscm

LOGGER = nscm.G_Logger
DF = pd.DataFrame

# 전역변수, set_match_code_env 통해 동적으로 값 설정
V_TYPE = None
V_PLANWEEK = None
V_PLANID = None
V_EFFSTARTDATE = None
V_EFFENDDATE = None
V_EFFSTARTDATE_STR = None
V_EFFENDDATE_STR = None
V_CURRENTDATE_STR = None
find_priority_position = None


def set_match_code_local_env(accessor: object) -> None:
    '''동적으로 전역변수를 설정하기 위한 함수'''
    global V_TYPE, V_PLANWEEK, V_PLANID, V_EFFSTARTDATE, V_EFFENDDATE, \
        V_EFFSTARTDATE_STR, V_EFFENDDATE_STR, V_CURRENTDATE_STR, find_priority_position

    # plan 설정
    V_TYPE = accessor.plan_type
    V_PLANWEEK = accessor.plan_week
    V_PLANID = accessor.plan_id
    V_EFFSTARTDATE = pd.Timestamp(accessor.start_date)
    V_EFFENDDATE = pd.Timestamp(accessor.end_date)
    V_EFFSTARTDATE_STR = V_EFFSTARTDATE.strftime('%Y-%m-%d')
    V_EFFENDDATE_STR = V_EFFENDDATE.strftime('%Y-%m-%d')
    V_CURRENTDATE_STR = V_EFFSTARTDATE_STR

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


def adjust_qtypromised_QTY_GAP(df: pd.DataFrame) -> pd.DataFrame:
    '''
    QTY_GAP이 0보다 크면 할당을 진행하는 함수
    QTYPROMISED 오름차순으로 정렬 후 적용
    '''
    sorted_df = df.sort_values(by=[ND_.QTYPROMISED], ascending=[True])
    # sorted_df = df.sort_values(by=[ND_.QTYPROMISED, ND_.SALESORDERID, ND_.SOLINENUM], ascending=[True, True, True]) # 로직 비교 검증용!!

    # 정렬 후 인덱스 재설정 ['index' 컬럼이 아님!!]
    sorted_df.reset_index(drop=True, inplace=True)
    v_qty_gap = sorted_df.at[0, 'QTY_GAP']
    print(f"■ v_qty_gap = {v_qty_gap}")

    cnt = 0
    for idx, row in sorted_df.iterrows():
        cnt = cnt + 1

        if int(v_qty_gap) <= 0:
            #### 5: 
            print('loop break')
            print(f"No.{cnt}:  {row[ND_.SALESORDERID]}, {row[ND_.SOLINENUM]}, {row['V_QTY_GAP']}, {row[ND_.QTYPROMISED]}")
            print('loop break')
            break   # 할당 종료

        if int(v_qty_gap) >= int(row[ND_.QTYPROMISED]):
            #### 6: QTYPROMISED를 0으로 설정
            # sorted_df.at[idx, 'UPBY] = sorted_df.at[idx, 'UPBY] + '::' + f"{row['QTYPROMISED]}-{row['QTY_GAP']}"
            sorted_df.at[idx, ND_.QTYPROMISED] = 0
            sorted_df.at[idx, 'UPDTTM'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")            
            print(f"No.{cnt}:  {row[ND_.SALESORDERID]}, {row[ND_.SOLINENUM]}, {row['V_QTY_GAP']}, {row[ND_.QTYPROMISED]}")
            v_qty_gap = v_qty_gap -  int(row[ND_.QTYPROMISED])

        else:
            #### 7: QTY_GAP 만큼 QTYPROMISED 감소
            # sorted_df.at[idx, 'UPBY] = sorted_df.at[idx, 'UPBY] + '::' + f"{row['QTYPROMISED]}-{row['QTY_GAP']}"
            sorted_df.at[idx, ND_.QTYPROMISED] = int(row[ND_.QTYPROMISED]) - v_qty_gap
            sorted_df.at[idx, 'UPDTTM'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"No.{cnt}:  {row[ND_.SALESORDERID]}, {row[ND_.SOLINENUM]}, {row['V_QTY_GAP']}, {row[ND_.QTYPROMISED]}")
            v_qty_gap = 0  # 할당 완료       

    return sorted_df


def adjust_qtypromised_IND_ORD(df: pd.DataFrame) -> pd.DataFrame:
    '''
    V_REMAINQTY를 SOPROMISE의 QTYPROMISED에서 차감하는 함수
    '''

    sorted_df = df.sort_values(by=[ND_.DEMANDPRIORITY, 'SOPROMISE_PRIORITY'], ascending=[False, False])
    # sorted_df = df.sort_values(by=[ND_.DEMANDPRIORITY, 'SOPROMISE_PRIORITY', ND_.SALESORDERID, ND_.SOLINENUM], ascending=[False, False, True, True]) # 로직 비교 검증용!!

    # 정렬 후 인덱스 재설정 ['index' 컬럼이 아님!!]
    sorted_df.reset_index(drop=True, inplace=True)
                                   
    v_remainqty = sorted_df.at[0, 'V_REMAINQTY']
    print(f"■ v_remainqty = {v_remainqty}")

    cnt = 0
    for idx, row in sorted_df.iterrows():
        cnt = cnt + 1

        if v_remainqty > 0:
            v_remainqty = v_remainqty - int(row[ND_.QTYPROMISED])

            # remain_qty = int(row['REMAINQTY'])    # remain_qty - current_qty

            if v_remainqty >= 0:
                sorted_df.at[idx, ND_.QTYPROMISED] = 0
                sorted_df.at[idx, 'UPBY'] = str(sorted_df.at[idx, 'UPBY']) + '::' + 'LOCAL'
                print(f" 🥨 v_remainqty = {v_remainqty}")
                print(f"{row[ND_.SALESORDERID]}, {row[ND_.SOLINENUM]}, {v_remainqty}, {row[ND_.QTYPROMISED]}")
            else:
                sorted_df.at[idx, ND_.QTYPROMISED] = abs(v_remainqty)
                sorted_df.at[idx, 'UPBY'] = str(sorted_df.at[idx, 'UPBY']) + '::' + 'LOCAL'
                print(f" 🍄 v_remainqty = {v_remainqty}")
                print(f"{row[ND_.SALESORDERID]}, {row[ND_.SOLINENUM]}, {v_remainqty}, {row[ND_.QTYPROMISED]}")
                v_remainqty = 0

            # 업데이트 타임 및 정보 추가
            sorted_df.at[idx, 'UPDTTM'] =datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sorted_df.at[idx, 'UPBY'] = sorted_df.at[idx, 'UPBY'] + '::' + 'LOCAL'

    return sorted_df


def adjust_qtypromised_IND_FCST(df: pd.DataFrame) -> pd.DataFrame:
    '''
    V_REMAINQTY를 SOPROMISE의 QTYPROMISED에서 차감하는 함수
    '''

    sorted_df = df.sort_values(by=[ND_.DEMANDPRIORITY], ascending=False)
    # sorted_df = df.sort_values(by=[ND_.DEMANDPRIORITY, ND_.SALESORDERID, ND_.SOLINENUM], ascending=[False, True, True])

    # 정렬 후 인덱스 재설정 ['index' 컬럼이 아님!!]
    sorted_df.reset_index(drop=True, inplace=True)

    v_remainqty = sorted_df.at[0, 'V_REMAINQTY']
    print(f"■ v_remainqty = {v_remainqty}")

    cnt = 0
    for idx, row in sorted_df.iterrows():
        cnt = cnt + 1

        if v_remainqty > 0:
            v_remainqty = v_remainqty - int(row[ND_.QTYPROMISED])

            if v_remainqty >= 0:
                sorted_df.at[idx, ND_.QTYPROMISED] = 0
                sorted_df.at[idx, 'UPBY'] = str(sorted_df.at[idx, 'UPBY'])+ '::' + 'LOCAL_FCST'
                print(f" 🥨 v_remainqty = {v_remainqty}")
                print(f"No.{cnt}:  {row[ND_.SALESORDERID]}, {row[ND_.SOLINENUM]}, {v_remainqty}, {row[ND_.QTYPROMISED]}")
            else:
                sorted_df.at[idx, ND_.QTYPROMISED] = abs(v_remainqty)
                sorted_df.at[idx, 'UPBY'] = str(sorted_df.at[idx, 'UPBY'])+ '::' + 'LOCAL_FCST'
                print(f" 🍄 v_remainqty = {v_remainqty}")
                print(f"No.{cnt}:  {row[ND_.SALESORDERID]}, {row[ND_.SOLINENUM]}, {v_remainqty}, {row[ND_.QTYPROMISED]}")
                v_remainqty = 0

            # 업데이트 타임 및 정보 추가
            sorted_df.at[idx, 'UPDTTM'] =datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sorted_df.at[idx, 'UPBY'] = str(sorted_df.at[idx, 'UPBY']) + '::' + 'LOCAL_FCST'

    return sorted_df


# [ITEM, SITEID] 별 'REMAINQTY'가 최초로 0 이하의 수가 되는 인덱스 추출
def get_first_zero_or_negative_index(series):
    '''
    [ITEM, SITEID] 별 'REMAINQTY'가 최초로 0 이하의 수가 되는 인덱스 추출

    '''
    zero_or_negative_index = series[series <= 0].index
    if not zero_or_negative_index.empty:
        return zero_or_negative_index[0]
    else:
        return np.nan


def create_netted_demand_table(
        df_exp_sopromise_src: DF, df_da_seller_map: DF, df_dim_item: DF,
        df_region_asn: DF):
    '''
    계산을 위한 중간 테이블 생성 : Netted Demand 중 Local 대상
    '''

    df_1_1 = df_da_seller_map.copy().merge(
        df_dim_item,
        left_on=[DAISM.ITEM],
        right_on=[I.ITEM],
        how='inner',
    )

    # MST_SDMREGIONASN 필터링
    df_1_2 = df_region_asn.copy()
    df_1_2 = df_1_2[df_1_2[RASN.ISVALID] == 'Y']

    # df_1 생성 [📝 병합시 컬럼명 주의!]
    df_1 = df_1_1.merge(
        df_1_2, left_on=[DAISM.SITEID, DAISM.SALESID, I.SECTION],
        right_on=[RASN.SITEID, RASN.REGIONID, RASN.SECTION],
        how='inner',)[[DAISM.SITEID, RASN.SALESID, DAISM.ITEM, DAISM.SALESID]]  # DAISM.SALESID : LOCALID

    # 컬럼명 변경   {DAISM.SALESID : 'LOCALID'}
    df_1.rename(columns={DAISM.SALESID: 'LOCALID'}, inplace=True)

    # df_2 생성
    df_2 = df_exp_sopromise_src.copy().merge(
        df_1,
        left_on=[ND_.SITEID, ND_.SALESID, ND_.ITEMID],
        right_on=[DAISM.SITEID, RASN.SALESID, DAISM.ITEM],
        how='inner',
    )

    # df_BUF_SOPROMISE 생성 ('index' 필드 중복 제거)
    df_BUF_SOPROMISE = df_2.copy().drop_duplicates(subset=[ND_.SALESORDERID, ND_.SOLINENUM, ND_.ITEMID], keep='first')

    # GC 필드
    # df_BUF_SOPROMISE[ND_.GCID] = df_BUF_SOPROMISE['GCID']

    # 사용하지 않는 컬럼 제거
    df_BUF_SOPROMISE = df_BUF_SOPROMISE.drop(columns=[DAISM.SITEID, RASN.SALESID, DAISM.ITEM])
    df_BUF_SOPROMISE['ASSIGNEDQTY'] = ''

    df_BUF_SOPROMISE['UPBY'] = ''    # 검증용 UPBY 생성
    df_BUF_SOPROMISE['UPBY'] = 'CREATED_df_BUF_SOPROMISE'

    # 'index' 컬럼이 있는 경우 reset_index() 로 'index' 컬럼 삭제
    if 'index' in df_BUF_SOPROMISE.columns:
        print('index')
        df_BUF_SOPROMISE = df_BUF_SOPROMISE.set_index('index')
        df_BUF_SOPROMISE.reset_index(drop=True, inplace=True)

    return df_BUF_SOPROMISE


def create_local_sales_order_table(
        df_da_item_seller_map: DF, df_dim_item: DF,
        df_region_asn: DF, df_local_sales_order: DF, df_dim_netting_sales: DF, v_planid: str):
    '''
    계산을 위한 중간 테이블 생성 : Sales Order 중 Local 대상
    '''

    # SECTION 정보 추가
    df_1_1 = df_da_item_seller_map.copy().merge(
        df_dim_item,
        left_on=[DAISM.ITEM],
        right_on=[I.ITEM],
        how='inner',
    )

    # MTA_SELLERMAP에서 TYPE='SELLER'로 필터링
    df_1_1 = df_1_1[df_1_1[DAISM.TYPE] == 'SELLER']

    # MST_SDMREGIONASN 에서 ISVALID='Y'로 필터링
    df_1_2 = df_region_asn.copy()
    df_1_2 = df_1_2[df_1_2[RASN.ISVALID] == 'Y']

    # df_1 생성 [📝 병합시 컬럼명 주의!]
    df_1 = df_1_1.merge(
        df_1_2,
        left_on=[DAISM.SITEID, DAISM.SALESID, I.SECTION],
        right_on=[RASN.SITEID, RASN.REGIONID, RASN.SECTION],
        how='inner',
    )

    # df_1 중복제거: subset=[DAISM.SITEID, DAISM.SALESID, RASN.SALESID]
    df_1 = df_1.drop_duplicates(subset=[DAISM.SITEID, DAISM.SALESID, RASN.SALESID], keep='first')

    # df_1 필드명 변경
    df_1 = df_1[[RASN.SALESID, DAISM.SALESID, DAISM.SITEID]]
    # df_1.rename(columns={RASN.SALESID : 'SALESID', DAISM.SALESID: 'LOCALID', DAISM.SITEID: 'SITEID'}, inplace=True)
    df_1.rename(columns={RASN.SALESID: ND_.SALESID, DAISM.SALESID: ND_.LOCALID, DAISM.SITEID: ND_.SITEID}, inplace=True)

    # SUNDAY 계산
    current_date = pd.to_datetime(V_EFFSTARTDATE_STR)
    dt_sunday = current_date.to_period('W').end_time
    v_sunday = dt_sunday.strftime('%Y-%m-%d')

    # df_2 생성
    df_2 = df_local_sales_order.copy().merge(
        df_1,
        left_on=[LSaO.SALESID, LSaO.LOCALID, LSaO.SITEID],
        right_on=[ND_.SALESID, ND_.LOCALID, ND_.SITEID],
        how='inner',
    )

    # GCID, AP2ID 등의 정보는 없으므로 DIM.Sales 정보에서 Salesid 기준으로 join 하여 가져옴 <<< DIM.Sales에 해당컬럼 없음.
    # GCID, AP2ID 등의 정보는 DIM.Sales Domain 정보에서 Salesid 기준으로 join 하여 가져옴!!!
    # [DIM.Sales Domain → df_dim_netting_sales] 로 대체
    df_2 = df_2.merge(
        df_dim_netting_sales,
        left_on=[LSaO.SALESID],
        right_on=[NS.SALESID],
        how='left',
    )

    # WEEK 필드 계산(생성)
    df_2[ND_.WEEKRANK] = ((pd.to_datetime(
        df_2[LSaO.REQDATE]) - pd.to_datetime(v_sunday)) / np.timedelta64(7, 'D')).astype(int) + 2
    # 주차를 숫자로 형 변환해야. SORTING 정렬이 올바로 되어 RANK 가 정상 생성됨
    df_2[ND_.WEEKRANK] = df_2[ND_.WEEKRANK].astype(int)

    # RANK 생성을 위한 정렬
    df_2 = df_2.sort_values(by=[LSaO.ITEM, LSaO.SITEID, LSaO.TYPE, ND_.WEEKRANK], ascending=[True, True, True, True])

    # RANK() OVER(PARTITION BY ITEM, A.SITEID, TYPE ORDER BY WEEK) 구현
    df_2['RANK'] = df_2.groupby([LSaO.ITEM, LSaO.SITEID, LSaO.TYPE])[ND_.WEEKRANK].rank(method='min', ascending=True).astype(int)

    # 컬럼명 변환
    df_2.rename(columns={NS.AP2ID: ND_.AP2ID, NS.GCID: ND_.GCID}, inplace=True)

    # SALESORDERID 생성
    df_2[ND_.SALESORDERID] = 'COM_ORD::1::' + df_2['RANK'].astype(str) + '::' + df_2[ND_.GCID] + '::' + df_2[ND_.AP2ID] + '::' + df_2[LSaO.SALESID] + '::' + df_2[LSaO.ITEM] + '::' + df_2[LSaO.SITEID] + '::' + df_2[LSaO.TYPE] + '_DM'

    # 나머지 컬럼 초기화
    df_2[ND_.SOPROMISEID] = df_2[LSaO.TYPE] + '_DM'
    df_2[ND_.PLANID] = v_planid
    df_2[ND_.SOLINENUM] = '100'
    df_2[ND_.ITEMID] = df_2[LSaO.ITEM]
    df_2[ND_.QTYPROMISED] = df_2[LSaO.QTY]
    df_2[ND_.PROMISEDDELDATE] = df_2[LSaO.REQDATE]
    df_2[ND_.SITEID] = df_2[LSaO.SITEID]
    df_2[ND_.SHIPTOID] = df_2[LSaO.SITEID]
    df_2[ND_.SALESID] = df_2[LSaO.SALESID]
    df_2[ND_.SALESLEVEL] = 'Account'
    df_2[ND_.DEMANDTYPERANK] = '1'
    df_2[ND_.CHANNELRANK] = None
    df_2[ND_.CUSTOMERRANK] = None
    df_2[ND_.PRODUCTRANK] = None
    df_2[ND_.DEMANDPRIORITY] = None
    df_2[ND_.TIEBREAK] = None
    # df_2[ND_.GBM] = ''     # 이후에 processor.py에서 일괄처리하므로 공란으로 처리함
    df_2[ND_.GLOBALPRIORITY] = None
    df_2[ND_.LOCALPRIORITY] = None
    df_2[ND_.BUSINESSTYPE] = None
    df_2[ND_.ROUTING_PRIORITY] = None
    df_2[ND_.NO_SPLIT] = None
    df_2[ND_.MAP_SATISFY_SS] = None
    df_2[ND_.PREALLOC_ATTRIBUTE] = 'NOPREALLOC'
    df_2[ND_.BUILDAHEADTIME] = None
    df_2[ND_.TIMEUOM] = None
    # df_2['GC'] = df_2['GCID']
    df_2[ND_.MEASURETYPERANK] = '1'
    df_2[ND_.PREFERENCERANK] = '1'
    df_2['INITDTTM'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_2['INITBY'] = 'local_reg.py'  # 'DA_LOCAL_REG_DEV.ipynb'

    # 컬럼순서 리스트 : BUF_SOPROMISE_LOCAL
    LI_COL_BUF_SOPROMISE_LOCAL = [
        ND_.SOPROMISEID, ND_.PLANID, ND_.SALESORDERID, ND_.SOLINENUM, ND_.ITEMID,
        ND_.QTYPROMISED, ND_.PROMISEDDELDATE, ND_.SITEID, ND_.SHIPTOID, ND_.SALESID,
        ND_.LOCALID, ND_.SALESLEVEL, ND_.DEMANDTYPERANK, ND_.WEEKRANK, ND_.CHANNELRANK,
        ND_.CUSTOMERRANK, ND_.PRODUCTRANK, ND_.DEMANDPRIORITY, ND_.TIEBREAK,
        ND_.GLOBALPRIORITY, ND_.LOCALPRIORITY, ND_.BUSINESSTYPE, ND_.ROUTING_PRIORITY, ND_.NO_SPLIT,
        ND_.MAP_SATISFY_SS, ND_.PREALLOC_ATTRIBUTE, ND_.BUILDAHEADTIME, ND_.TIMEUOM, ND_.AP2ID,
        ND_.GCID, ND_.MEASURETYPERANK, ND_.PREFERENCERANK, 'INITDTTM', 'INITBY',
    ]

    # df_BUF_SOPROMISE_LOCAL 생성
    df_BUF_SOPROMISE_LOCAL = df_2.copy()
    df_BUF_SOPROMISE_LOCAL = df_BUF_SOPROMISE_LOCAL[LI_COL_BUF_SOPROMISE_LOCAL]
    df_BUF_SOPROMISE_LOCAL = df_BUF_SOPROMISE_LOCAL.reset_index()

    # TYPE CASTING
    df_BUF_SOPROMISE_LOCAL = df_BUF_SOPROMISE_LOCAL.astype({
        ND_.QTYPROMISED: 'int32',
        ND_.PROMISEDDELDATE: 'datetime64[ns]',
    })

    df_BUF_SOPROMISE_LOCAL['UPBY'] = ''    # 검증용 UPBY 생성
    df_BUF_SOPROMISE_LOCAL['UPBY'] = 'CREATED_df_BUF_SOPROMISE_LOCAL'

    return df_BUF_SOPROMISE_LOCAL



def map_priority_to_com_ord(df_BUF_SOPROMISE: DF, df_BUF_SOPROMISE_LOCAL: DF):
    '''
    강제로 추가된 COM_ORD에 우선순위를 매핑
    '''

    df_1 = df_BUF_SOPROMISE.copy()
    df_1 = df_1[df_1[ND_.SALESORDERID].str.contains('ORD', na=False)]

    # MIN(DEMANDPRIORITY) 산출
    df_1 = df_1.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE], as_index=False)[ND_.DEMANDPRIORITY].min()

    # df_BUF_SOPROMISE_LOCAL 과 df_1을 MERGE
    df_BUF_SOPROMISE_LOCAL = df_BUF_SOPROMISE_LOCAL.merge(
        df_1,
        on=[ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE],
        how='left',
        suffixes=('', '_NEW'),
    )

    # df_BUF_SOPROMISE_LOCAL 의 DEMANDPRIORITY 업데이트 (NULL 인 경우 df_1의 값으로)
    df_BUF_SOPROMISE_LOCAL[ND_.DEMANDPRIORITY] = np.where(
        df_BUF_SOPROMISE_LOCAL[ND_.DEMANDPRIORITY].isna(),
        df_BUF_SOPROMISE_LOCAL[ND_.DEMANDPRIORITY + '_NEW'],
        df_BUF_SOPROMISE_LOCAL[ND_.DEMANDPRIORITY]
    )

    # 불필요한 컬럼 제거
    df_BUF_SOPROMISE_LOCAL.drop(columns=[ND_.DEMANDPRIORITY + '_NEW'], inplace=True)

    return df_BUF_SOPROMISE_LOCAL


def allocate_qty_promised(df_BUF_SOPROMISE: DF, df_BUF_SOPROMISE_LOCAL: DF):
    '''
    Demand 와 Local Demand 간의 Demand Qty Gap 산출하여 QTYPROMISED 할당
    '''

    # [df_BUF_SOPROMISE, df_BUF_SOPROMISE_LOCAL] PROMISEDDELDATE를 주차 포맷으로 변경
    isoCalDate = df_BUF_SOPROMISE[ND_.PROMISEDDELDATE].dt.isocalendar()
    df_BUF_SOPROMISE[ND_.PROMISEDDELDATE + '_WEEK'] = isoCalDate.year.astype(str) + isoCalDate.week.astype(str).str.zfill(2)
    isoCalDate = df_BUF_SOPROMISE_LOCAL[ND_.PROMISEDDELDATE].dt.isocalendar()
    df_BUF_SOPROMISE_LOCAL[ND_.PROMISEDDELDATE + '_WEEK'] = isoCalDate.year.astype(str) + isoCalDate.week.astype(str).str.zfill(2)

    # df_1 생성 (DMD_QTY 계산)
    df_1 = df_BUF_SOPROMISE.copy()
    df_1 = df_1[df_1[ND_.SALESORDERID].str.contains(r"^COM.*ORD", regex=True)]
    df_1[ND_.QTYPROMISED] = df_1[ND_.QTYPROMISED].astype(int).fillna(0)
    df_1 = df_1.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE], as_index=False)[ND_.QTYPROMISED].sum()
    df_1.rename(columns={ND_.QTYPROMISED: 'DMD_QTY'}, inplace=True)
    # df_1['IND_QTY'] = 0   # 어차피 0 값이므로 나중에 merge시 컬럼명 혼란방지를 위해 실행하지 않음

    # df_2 생성 (IND_QTY 계산)
    df_2 = df_BUF_SOPROMISE_LOCAL.copy()
    df_2 = df_2[df_2[ND_.SALESORDERID].str.contains(r"^COM.*ORD", regex=True)]
    df_2[ND_.QTYPROMISED] = df_2[ND_.QTYPROMISED].astype(int).fillna(0)
    df_2 = df_2.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE], as_index=False)[ND_.QTYPROMISED].sum()
    df_2.rename(columns={ND_.QTYPROMISED: 'IND_QTY'}, inplace=True)

    # df_3 생성 (QTY_GAP 계산)
    df_3 = df_1.merge(
        df_2,
        on=[ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE],
        how='outer',
    )
    print(f"df_3.shape[0]: {df_3.shape[0]}")

    # df_3: PROMISEDDELDATE를 주차 포맷으로 변경
    isoCalDate = df_3[ND_.PROMISEDDELDATE].dt.isocalendar()
    df_3[ND_.PROMISEDDELDATE + '_WEEK'] = isoCalDate.year.astype(str) + isoCalDate.week.astype(str).str.zfill(2)

    # df_3: ITEM, SITEID, SALESID, PROMISEDDEDATE (날짜는 iso week 기준으로 주차로 변경) 기준으로 grouping 하여 DMD_QTY, IND_QTY 를 SUM
    df_3 = df_3.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK'], as_index=False)[ ['DMD_QTY', 'IND_QTY']].sum()

    # SUM(DMD_QTY) < SUM(IND_QTY) 대상으로만 발췌
    df_3 = df_3[df_3['DMD_QTY'] < df_3['IND_QTY']]

    # QTY_GAP 컬럼 생성
    df_3['QTY_GAP'] = df_3['IND_QTY'].astype(int) - df_3['DMD_QTY'].astype(int)
    df_3['QTY_GAP'] = df_3['QTY_GAP'].astype(int)

    # df_3 필터링: QTY_GAP이 양수인 경우만 필터링
    df_3 = df_3[df_3['QTY_GAP'] > 0]

    df_local_x = df_BUF_SOPROMISE_LOCAL.copy()
    df_local_x = df_local_x[df_local_x[ND_.SALESORDERID].str.contains(r"^COM.*ORD", regex=True)]

    df_BUF_SOPROMISE_LOCAL_X = df_local_x.merge(
        df_3,
        on=[ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK'],
        how='inner',
        suffixes=('', '_from_df_3')
    )

    # df_BUF_SOPROMISE_LOCAL_X.drop(columns=[ND_.PROMISEDDELDATE + '_from_df_3'], inplace=True)    # 불필요한 접미사 컬럼 제거
    df_BUF_SOPROMISE_LOCAL_X['QTY_GAP'] = df_BUF_SOPROMISE_LOCAL_X['QTY_GAP'].fillna(0).astype(int)
    df_BUF_SOPROMISE_LOCAL_X[ND_.QTYPROMISED] = df_BUF_SOPROMISE_LOCAL_X[ND_.QTYPROMISED] .astype(int)

    df_BUF_SOPROMISE_LOCAL_X = df_BUF_SOPROMISE_LOCAL_X.sort_values(by=ND_.QTYPROMISED, ascending=True)

    # 정렬 후 인덱스 재설정
    df_BUF_SOPROMISE_LOCAL_X.reset_index(drop=True, inplace=True)

    ####################################################################
    # 🎰 QTY 할당 처리: adjust_qtypromised_QTY_GAP
    df_BUF_SOPROMISE_LOCAL_X['CUMSUM_QTY'] = df_BUF_SOPROMISE_LOCAL_X.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK'])[ND_.QTYPROMISED].cumsum()

    # V_QTY_GAP 계산
    df_BUF_SOPROMISE_LOCAL_X['V_QTY_GAP'] = df_BUF_SOPROMISE_LOCAL_X['QTY_GAP'].astype(int) - df_BUF_SOPROMISE_LOCAL_X['CUMSUM_QTY'].astype(int)

    df_BUF_SOPROMISE_LOCAL_X = df_BUF_SOPROMISE_LOCAL_X.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK'], group_keys=False).apply(adjust_qtypromised_QTY_GAP)
    ####################################################################

    # UPDATE: SOPROMISE_LOCAL (df_exp_sopromise_src)
    df_BUF_SOPROMISE_LOCAL.set_index('index', inplace=True)
    df_BUF_SOPROMISE_LOCAL_X.set_index('index', inplace=True)

    df_BUF_SOPROMISE_LOCAL.loc[df_BUF_SOPROMISE_LOCAL_X.index.intersection(df_BUF_SOPROMISE_LOCAL.index), ND_.QTYPROMISED] = df_BUF_SOPROMISE_LOCAL_X.loc[df_BUF_SOPROMISE_LOCAL_X.index.intersection(df_BUF_SOPROMISE_LOCAL.index), ND_.QTYPROMISED]
    df_BUF_SOPROMISE_LOCAL.loc[df_BUF_SOPROMISE_LOCAL_X.index.intersection(df_BUF_SOPROMISE_LOCAL.index), 'UPBY'] = df_BUF_SOPROMISE_LOCAL_X.loc[df_BUF_SOPROMISE_LOCAL_X.index.intersection(df_BUF_SOPROMISE_LOCAL.index), 'UPBY']     # 검증용 UPBY 기록

    df_BUF_SOPROMISE_LOCAL = df_BUF_SOPROMISE_LOCAL.reset_index()

    return df_BUF_SOPROMISE_LOCAL


def deduct_demand_by_ind_com_ord(
        df_da_item_seller_map: DF, df_dim_item: DF, df_region_asn: DF,
        df_local_sales_order: DF, df_dim_netting_sales: DF,
        df_BUF_SOPROMISE: DF):
    '''
    IND_COM_ORD 수량만큼을 Demand에서 차감
    '''

    # SECTION 정보 추가
    df_1_1 = df_da_item_seller_map.copy().merge(
        df_dim_item,
        left_on=[DAISM.ITEM],
        right_on=[I.ITEM],
        how='inner',
    )

    # MTA_SELLERMAP에서 TYPE='SELLER'로 필터링
    df_1_1 = df_1_1[df_1_1[DAISM.TYPE] == 'SELLER']

    # MST_SDMREGIONASN 에서 ISVALID='Y'로 필터링
    df_1_2 = df_region_asn.copy()
    df_1_2 = df_1_2[df_1_2[RASN.ISVALID] == 'Y']

    # df_1 생성 [📝 병합시 컬럼명 주의!]
    df_1 = df_1_1.merge(
        df_1_2,
        left_on=[DAISM.SITEID, DAISM.SALESID, I.SECTION],
        right_on=[RASN.SITEID, RASN.REGIONID, RASN.SECTION],
        how='inner',
    )
    # df_1 중복제거: subset=[DAISM.SITEID, DAISM.SALESID, RASN.SALESID]
    df_1 = df_1.drop_duplicates(subset=[DAISM.SITEID, DAISM.SALESID, RASN.SALESID])

    # df_1 필드명 변경
    df_1 = df_1[[RASN.SALESID, DAISM.SALESID, DAISM.SITEID]]
    df_1.rename(columns={RASN.SALESID: ND_.SALESID, DAISM.SALESID: ND_.LOCALID, DAISM.SITEID: ND_.SITEID}, inplace=True)

    # df_2 생성
    df_2 = df_local_sales_order.copy().merge(
        df_1,
        left_on=[LSaO.SALESID, LSaO.LOCALID, LSaO.SITEID],
        right_on=[ND_.SALESID, ND_.LOCALID, ND_.SITEID],
        how='inner',
    )

    # GCID, AP2ID 등의 정보는 DIM.Sales Domain 정보에서 Salesid 기준으로 join 하여 가져옴!!!    # [6,693 @ 202510_Y]  DB 수량 일치 함.
    df_2 = df_2.merge(
        df_dim_netting_sales,
        left_on=[LSaO.SALESID],
        right_on=[ND_.SALESID],
        how='left',
    )

    # Grouping 및 QTY 합산
    df_2 = df_2.groupby([LSaO.ITEM, LSaO.SITEID, 'AP2ID', LSaO.SALESID, LSaO.WEEK, LSaO.REQDATE], as_index=False)[LSaO.QTY].sum()

    # V_REMAINQTY 초기화
    df_2['V_REMAINQTY'] = df_2[LSaO.QTY].astype(int)

    # df_5 생성 (BUF_SOPROMISE 필터링)
    if 'index' not in df_BUF_SOPROMISE.columns:
        df_BUF_SOPROMISE.reset_index(inplace=True)


    # [df_BUF_SOPROMISE, df_BUF_SOPROMISE_LOCAL] PROMISEDDELDATE를 주차 포맷으로 변경
    isoCalDate = df_BUF_SOPROMISE[ND_.PROMISEDDELDATE].dt.isocalendar()
    df_BUF_SOPROMISE[ND_.PROMISEDDELDATE + '_WEEK'] = isoCalDate.year.astype(str) + isoCalDate.week.astype(str).str.zfill(2)

    df_5 = df_BUF_SOPROMISE.copy()
    df_5 = df_5[df_5[ND_.SALESORDERID].str.contains(r"^COM.*ORD", regex=True)]
    # df_5 = df_5[df_5[ND_.SALESORDERID].str.startswith('COM_ORD')]
    df_5 = df_5[df_5[ND_.QTYPROMISED].astype(int) > 0]

    # df_2와 JOIN (ITEM, SITEID, SALESID, WEEK 기준)
    df_5 = df_5.merge(
        df_2,
        left_on=[ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK'],
        right_on=[LSaO.ITEM, LSaO.SITEID, LSaO.SALESID, LSaO.WEEK],
        how='inner',
        suffixes=('', '_y')
    )

    # df_4 에서 발췌한 데이터의 salesid, siteid 기준으로 존재하는 것만 (inner join) 만족하는 대상으로 필터링
    df_4 = df_1
    df_exists = df_5.merge(
        df_4,
        left_on=[ND_.SALESID, ND_.SITEID],
        right_on=[ND_.SALESID, ND_.SITEID],
        how='inner',
    )

    # df_exists 중복제거
    df_exists = df_exists.drop_duplicates(subset=[ND_.SALESID, ND_.SITEID])

    # Exists 조건 필터링
    df_5 = df_5[
        (df_5[ND_.SALESID] + df_5[ND_.SITEID]).isin(
            df_exists[ND_.SALESID] + df_exists[ND_.SITEID]
        )
    ]

    if df_5.shape[0] > 0:
        # df_4 에서 발췌한 데이터의 salesid, siteid 기준으로 존재하는 것만 (inner join) 만족하는 대상으로 필터링
        df_4 = df_1
        df_exists = df_5.merge(
            df_4,
            left_on=[ND_.SALESID, ND_.SITEID],
            right_on=[ND_.SALESID, ND_.SITEID],
            how='inner',
        )

        # df_exists 중복제거
        df_exists = df_exists.drop_duplicates(subset=[ND_.SALESID, ND_.SITEID])

        # Exists 조건 필터링
        df_5 = df_5[
            (df_5[ND_.SALESID] + df_5[ND_.SITEID]).isin(
                df_exists[ND_.SALESID] + df_exists[ND_.SITEID]
            )
        ]

        if df_5.shape[0] > 0:
            print(f"IND_ORD: {df_5.shape[0]}")
            # DEMANDPRIORITY 및 SOPROMISED 정렬
            priority_order = {
                'GI_DM': 1,
                'DO_DM': 2,
                'CO_DM': 3,
                'NO_DM': 4,
            }
            df_5['SOPROMISE_PRIORITY'] = df_5[ND_.SOPROMISEID].map(priority_order)
            df_5 = df_5.sort_values(by=[ND_.DEMANDPRIORITY, 'SOPROMISE_PRIORITY'], ascending=[False, False])

            # 정렬 후 인덱스 재설정
            df_5.reset_index(drop=True, inplace=True)

            ####################################################################
            # 🎰 QTY 차감 처리: adjust_qtypromised_local
            df_5['CUMSUM_QTY'] = df_5.groupby([ND_.ITEMID, ND_.SITEID, ND_.AP2ID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK', LSaO.REQDATE])[ND_.QTYPROMISED].cumsum()

            # REMAINQTY 계산
            df_5['REMAINQTY'] = df_5['V_REMAINQTY'].astype(int) - df_5['CUMSUM_QTY'].astype(int)

            # QTY 차감 처리: adjust_qtypromised_IND_ORD
            df_5 = df_5.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK', LSaO.REQDATE], group_keys=False).apply(adjust_qtypromised_IND_ORD)
            ####################################################################

            # UPDATE: SOPROMISE (df_exp_sopromise_src)
            df_BUF_SOPROMISE.set_index('index', inplace=True)
            df_5.set_index('index', inplace=True)

            df_BUF_SOPROMISE.loc[df_5.index.intersection(df_BUF_SOPROMISE.index), ND_.QTYPROMISED] = df_5.loc[df_5.index.intersection(df_BUF_SOPROMISE.index), ND_.QTYPROMISED]
            df_BUF_SOPROMISE.loc[df_5.index.intersection(df_BUF_SOPROMISE.index), 'UPBY'] = df_5.loc[df_5.index.intersection(df_BUF_SOPROMISE.index), 'UPBY']     # 검증용 UPBY 기록

            df_BUF_SOPROMISE = df_BUF_SOPROMISE.reset_index()

    return df_BUF_SOPROMISE


def create_local_demand_ith_localid_com_ord(
        df_da_item_seller_map: DF, df_dim_item: DF, df_region_asn: DF,
        df_BUF_SOPROMISE: DF, df_BUF_SOPROMISE_LOCAL: DF):
    '''
    DEMAND 정보에 LOCALID 를 붙여서 Local Demand 생성: COM_ORD
    '''

    # SECTION 정보 추가 
    df_1_1 = df_da_item_seller_map.copy().merge(
        df_dim_item,
        left_on=[DAISM.ITEM],
        right_on=[I.ITEM],
        how='inner',
    )

    # MST_SDMREGIONASN 에서 ISVALID='Y'로 필터링
    df_1_2 = df_region_asn.copy() 
    df_1_2 = df_1_2[df_1_2[RASN.ISVALID]=='Y']


    # df_1 생성 [📝 병합시 컬럼명 주의!]
    df_1 = df_1_1.merge(
        df_1_2,
        left_on=[DAISM.SITEID, DAISM.SALESID, I.SECTION],
        right_on=[RASN.SITEID, RASN.REGIONID, RASN.SECTION],
        how='inner',
    )

    # RN 생성을 위한 정렬
    # df_1 = df_1.sort_values(by=[RASN.SALESID, DAISM.SITEID, DAISM.TYPE, DAISM.SALESID], ascending=[True, True, True, True])

    # RN 생성
    df_1['RN'] = df_1.groupby([RASN.SALESID, DAISM.SITEID], as_index=False)[DAISM.TYPE].rank(method='min', ascending=True).astype(int)


    # 최종 df_1
    df_1 = df_1[[RASN.SALESID, DAISM.SITEID, DAISM.SALESID, 'RN']]

    # df_1 필드명 변경
    df_1.rename(columns={RASN.SALESID : ND_.SALESID, DAISM.SALESID: 'LOCALID', DAISM.SITEID: ND_.SITEID}, inplace=True)

    # df_2 생성
    df_2 = df_1[df_1['RN']==1]

    # df_2 중복 레코드 제거
    df_2 = df_2.drop_duplicates(subset=[ND_.SALESID, ND_.SITEID, 'LOCALID','RN'])

    # df_2 컬럼명 변경
    df_2.rename(columns={'LOCALID': ND_.LOCALID}, inplace=True)

    # df_3 생성: df_BUF_SOPROMISE 필터링: COM_ORD
    df_3 = df_BUF_SOPROMISE.copy()
    df_3 = df_3[
        (df_3[ND_.SALESORDERID].str.startswith('COM_ORD')) &
        (df_3[ND_.QTYPROMISED].astype(int) > 0)
    ]

    # df_2 와 JOIN하여 LOCAL_ID 추가
    df_3 = df_3.merge(
        df_2,
        left_on=[ND_.SALESID, ND_.SITEID],
        right_on=[ND_.SALESID, ND_.SITEID],
        how='inner',
        suffixes=('_from_df_3', '_from_df_2')
    )
    # LOCALID 값은 df_2(~from df_da_item_seller_map 의 값)
    df_3.rename(columns={ND_.LOCALID + '_from_df_2': ND_.LOCALID}, inplace=True)
    df_3.rename(columns={ND_.ITEMID + '_from_df_3': ND_.ITEMID}, inplace=True)

    # index 중복 방지
    df_3 = df_3.drop_duplicates(subset=[ND_.SALESORDERID, ND_.SOLINENUM], keep='first')          # drop_duplicates(subset=['index'], keep='first')

    df_3['INITDTTM'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_3['INITBY'] = ''
    # df_3[ND_.GCID] = df_3['GCID']
    df_3['UPBY'] = df_3['UPBY'] + '::' + 'COM_ORD_INSERTED'

    # INSERT 컬럼 리스트
    COL_TO_INSERT = [
        ND_.SOPROMISEID, ND_.PLANID, ND_.SALESORDERID, ND_.SOLINENUM, ND_.ITEMID, 
        ND_.QTYPROMISED, ND_.PROMISEDDELDATE, ND_.SITEID, ND_.SHIPTOID, ND_.SALESID,
        ND_.LOCALID, ND_.SALESLEVEL, ND_.DEMANDTYPERANK, ND_.WEEKRANK,
        ND_.CUSTOMERRANK, ND_.PRODUCTRANK, ND_.DEMANDPRIORITY, 
        ND_.GLOBALPRIORITY, ND_.LOCALPRIORITY,
        ND_.AP2ID,
        ND_.GCID, ND_.MEASURETYPERANK, ND_.PREFERENCERANK, 'INITDTTM', 'INITBY', 'UPBY',
    ]

    # 기존 df_BUF_SOPROMISE_LOCAL 에 데이터 추가(INSERT)
    # df_BUF_SOPROMISE_LOCAL.set_index('index', inplace=True)

    df_BUF_SOPROMISE_LOCAL = pd.concat(
        [df_BUF_SOPROMISE_LOCAL, df_3[COL_TO_INSERT]],
        ignore_index=True,
    )

    return df_BUF_SOPROMISE_LOCAL


def create_local_demand_ith_localid_new_ord(
        df_da_item_seller_map: DF, df_dim_item: DF, df_region_asn: DF,
        df_BUF_SOPROMISE: DF, df_BUF_SOPROMISE_LOCAL: DF):
    '''
    DEMAND 정보에 LOCALID 를 붙여서 Local Demand 생성: NEW_ORD
    '''

    # SECTION 정보 추가 
    df_1_1 = df_da_item_seller_map.copy().merge(
        df_dim_item,
        left_on=[DAISM.ITEM],
        right_on=[I.ITEM],
        how='inner',
    )

    # MST_SDMREGIONASN 에서 ISVALID='Y'로 필터링
    df_1_2 = df_region_asn.copy()
    df_1_2 = df_1_2[df_1_2[RASN.ISVALID]=='Y']


    # df_1 생성 [📝 병합시 컬럼명 주의!]
    df_1 = df_1_1.merge(
        df_1_2,
        left_on=[DAISM.SITEID, DAISM.SALESID, I.SECTION],
        right_on=[RASN.SITEID, RASN.REGIONID, RASN.SECTION],
        how='inner',
    )

    # 가공컬럼
    df_1['TYPE_SALESID'] = df_1[DAISM.TYPE] + '_' + df_1[DAISM.SALESID]

    # 정렬
    df_1 = df_1.sort_values(by=[DAISM.SITEID, 'TYPE_SALESID'], ascending=True)

    # RN 생성 그루핑
    df_1['RN'] = df_1.groupby([RASN.SALESID, DAISM.SITEID])['TYPE_SALESID'].rank(method='dense', ascending=True)
    ### df_1['RN'] = df_1.groupby([RASN.REGIONID, RASN.SITEID])[DAISM.TYPE].rank(method="dense", ascending=True).astype(int)
    ### df_1['RN'] = df_1.groupby([RASN.REGIONID, RASN.SITEID], as_index=False)[DAISM.TYPE].rank(method="min", ascending=True).astype(int)

    # df_1 필드명 변경
    df_1.rename(columns={RASN.SALESID : ND_.SALESID, DAISM.SALESID: 'LOCALID', DAISM.SITEID: ND_.SITEID}, inplace=True)

    # df_2 생성
    df_2 = df_1[df_1['RN'].astype(int)==1]

    # df_2 중복 레코드 제거
    df_2 = df_2.drop_duplicates(subset=[ND_.SALESID, ND_.SITEID, 'LOCALID','RN'])

    # df_2 컬럼명 변경
    df_2.rename(columns={'LOCALID': ND_.LOCALID}, inplace=True)

    # df_3 생성: df_BUF_SOPROMISE 필터링: NEW_ORD
    df_3 = df_BUF_SOPROMISE.copy()
    df_3 = df_3[
        (df_3[ND_.SALESORDERID].str.startswith('NEW_ORD')) &
        (df_3[ND_.QTYPROMISED].astype(int) > 0)
    ]

    # df_2 와 JOIN하여 LOCAL_ID 추가
    df_3 = df_3.merge(
        df_2,
        left_on=[ND_.SALESID, ND_.SITEID],
        right_on=[ND_.SALESID, ND_.SITEID],
        how='inner',
        suffixes=('_from_df_3', '_from_df_2'),
    )
    # LOCALID 값은 df_2(~from df_da_item_seller_map 의 값)
    df_3.rename(columns={ND_.LOCALID + '_from_df_2': ND_.LOCALID}, inplace=True)
    df_3.rename(columns={ND_.ITEMID + '_from_df_3': ND_.ITEMID}, inplace=True)

    # index 중복 방지
    # df_3 = df_3.copy().drop_duplicates(subset=['index'], keep='first')
    # df_3 = df_3.copy().drop_duplicates(keep='first')
    df_3 = df_3.drop_duplicates(subset=[ND_.SALESORDERID, ND_.SOLINENUM], keep='first')

    df_3['INITDTTM'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_3['INITBY'] = ''
    # df_3[ND_.GCID] = df_3['GCID']
    df_3['UPBY'] = df_3['UPBY'] + '::' + 'NEW_ORD_INSERTED'

    # INSERT 컬럼 리스트
    COL_TO_INSERT = [
        ND_.SOPROMISEID, ND_.PLANID, ND_.SALESORDERID, ND_.SOLINENUM, ND_.ITEMID, 
        ND_.QTYPROMISED, ND_.PROMISEDDELDATE, ND_.SITEID, ND_.SHIPTOID, ND_.SALESID,
        ND_.LOCALID, ND_.SALESLEVEL, ND_.DEMANDTYPERANK, ND_.WEEKRANK,
        ND_.CUSTOMERRANK, ND_.PRODUCTRANK, ND_.DEMANDPRIORITY, 
        ND_.GLOBALPRIORITY, ND_.LOCALPRIORITY,
        ND_.AP2ID,
        ND_.GCID, ND_.MEASURETYPERANK, ND_.PREFERENCERANK, 'INITDTTM', 'INITBY', 'UPBY',
    ]

    # 기존 df_BUF_SOPROMISE_LOCAL 에 데이터 추가(INSERT)
    # df_BUF_SOPROMISE_LOCAL.set_index('index', inplace=True)

    df_BUF_SOPROMISE_LOCAL = pd.concat(
        [df_BUF_SOPROMISE_LOCAL, df_3[COL_TO_INSERT]],
        ignore_index=True,
    )

    return df_BUF_SOPROMISE_LOCAL


def create_local_measure_rate_table(
        df_da_item_seller_map: DF, df_local_measure: DF,
        df_local_sales_order: DF, df_dim_item: DF):
    '''
    Local Measure Rate 중간 테이블 생성
    '''

    # df_WITH_VSELLERMAP 생성
    df_WITH_VSELLERMAP = df_da_item_seller_map.copy()
    df_WITH_VSELLERMAP = df_WITH_VSELLERMAP[[DAISM.SITEID, DAISM.SALESID, DAISM.ITEM, DAISM.TYPE]]

    # df_2: EXP_LOCALMEASURE - 'SELLER' TYPE
    if not df_WITH_VSELLERMAP.empty and not df_local_measure.empty:
        df_2 = df_local_measure.copy().merge(
            df_WITH_VSELLERMAP[df_WITH_VSELLERMAP[DAISM.TYPE] == 'SELLER'],
            left_on=[LM.SITEID, LM.ITEM, LM.LOCALID],
            right_on=[DAISM.SITEID, DAISM.ITEM, DAISM.SALESID],
            how='inner',
        )
        df_2 = df_2.groupby([LM.SALESID, DAISM.ITEM, LM.LOCALID, LM.ISOWEEK, LM.SITEID], as_index=False)[LM.MEASUREQTY].sum()
        df_2.rename(columns={LM.MEASUREQTY: 'IND_FCST'}, inplace=True)
    else:
        df_2 = pd.DataFrame(columns=[LM.SALESID, DAISM.ITEM, LM.LOCALID, LM.ISOWEEK, DAISM.SITEID, 'IND_FCST'])

    # df_3: EXP_LOCALMEASURE - 'HUB' TYPE
    if not df_WITH_VSELLERMAP.empty and not df_local_measure.empty:
        df_3 = df_local_measure.copy().merge(
            df_WITH_VSELLERMAP[df_WITH_VSELLERMAP[DAISM.TYPE] == 'HUB'],
            left_on=[LM.SITEID, LM.ITEM, LM.LOCALID],
            right_on=[DAISM.SITEID, DAISM.ITEM, DAISM.SALESID],
            how='inner',
        )
        df_3 = df_3.groupby([LM.SALESID, DAISM.ITEM, LM.LOCALID, LM.ISOWEEK, LM.SITEID], as_index=False)[LM.MEASUREQTY].sum()
        df_3.rename(columns={LM.MEASUREQTY: 'NOR_FCST'}, inplace=True)
    else:
        df_3 = pd.DataFrame(columns=[LM.SALESID, DAISM.ITEM, LM.LOCALID, LM.ISOWEEK, DAISM.SITEID, 'NOR_FCST'])

    # df_4: BUF_SALESORDERLINE_LOCAL - 'SELLER' TYPE
    if not df_WITH_VSELLERMAP.empty and not df_local_sales_order.empty:
        df_4 = df_local_sales_order.copy()
        df_4 = df_4[df_4[LSaO.QTY].astype(int) > 0]

        df_4 = df_4.merge(
            df_WITH_VSELLERMAP[df_WITH_VSELLERMAP[DAISM.TYPE] == 'SELLER'],
            left_on=[LSaO.SITEID, LSaO.ITEM, LSaO.LOCALID],
            right_on=[DAISM.SITEID, DAISM.ITEM, DAISM.SALESID],
            how='inner',
        )
        df_4 = df_4.groupby([LSaO.SALESID, LSaO.ITEM, LSaO.LOCALID, LSaO.WEEK, LSaO.SITEID], as_index=False)[LSaO.QTY].sum()
        df_4.rename(columns={LSaO.QTY: 'IND_ORD'}, inplace=True)
    else:
        df_4 = pd.DataFrame(columns=[LSaO.SALESID, LSaO.ITEM, LSaO.LOCALID, LSaO.WEEK, LSaO.SITEID, 'IND_ORD'])

    # df_5: BUF_SALESORDERLINE_LOCAL - 'HUB' TYPE
    if not df_WITH_VSELLERMAP.empty and not df_local_sales_order.empty:
        df_5 = df_local_sales_order.copy()
        df_5 = df_5[df_5[LSaO.QTY].astype(int) > 0]

        df_5 = df_5.merge(
            df_WITH_VSELLERMAP[df_WITH_VSELLERMAP[DAISM.TYPE] == 'HUB'],
            left_on=[LSaO.SITEID, LSaO.ITEM, LSaO.LOCALID],
            right_on=[DAISM.SITEID, DAISM.ITEM, DAISM.SALESID],
            how='inner',
        )
        df_5 = df_5.groupby([LSaO.SALESID, LSaO.ITEM, LSaO.LOCALID, LSaO.WEEK, LSaO.SITEID], as_index=False)[LSaO.QTY].sum()
        df_5.rename(columns={LSaO.QTY: 'NOR_ORD'}, inplace=True)
    else:
        df_5 = pd.DataFrame(
            columns=[LSaO.SALESID, LSaO.ITEM, LSaO.LOCALID,
                     LSaO.WEEK, LSaO.SITEID, 'NOR_ORD'])

    # df_6 생성: UNION ALL
    df_6 = pd.concat([df_2, df_3, df_4, df_5], ignore_index=True)

    # SECTION 정보 추가
    if not df_6.empty and not df_dim_item.empty:
        df_6 = df_6.merge(
            df_dim_item,
            left_on=[LSaO.ITEM],
            right_on=[I.ITEM],
            how='left',
        )

    # GROUP BY 및 CASE 처리
    if not df_6.empty:
        df_MTA_LOCALMEASURE_RATE = df_6.groupby([I.SECTION, LSaO.SALESID, LSaO.ITEM, LSaO.SITEID, LSaO.WEEK], as_index=False).agg({
            'IND_FCST': 'sum', 
            'IND_ORD': 'sum', 
            'NOR_FCST': 'sum',
            'NOR_ORD': 'sum', 
        }).fillna(0)
        df_MTA_LOCALMEASURE_RATE = df_MTA_LOCALMEASURE_RATE.astype({
            'IND_FCST': 'int32',
            'IND_ORD': 'int32',
            'NOR_FCST': 'int32',
            'NOR_ORD': 'int32',
        })

        df_MTA_LOCALMEASURE_RATE['RATE'] = 0
        df_MTA_LOCALMEASURE_RATE['TOT_QTY'] = 0

    # CASE 문 처리
        df_MTA_LOCALMEASURE_RATE['IND_QTY'] = np.where(
            (df_MTA_LOCALMEASURE_RATE['IND_FCST'] - df_MTA_LOCALMEASURE_RATE['IND_ORD']) < 0, 
            0,
            df_MTA_LOCALMEASURE_RATE['IND_FCST'] - df_MTA_LOCALMEASURE_RATE['IND_ORD']
        )
        df_MTA_LOCALMEASURE_RATE['NOR_QTY'] = np.where(
            (df_MTA_LOCALMEASURE_RATE['NOR_FCST'] - df_MTA_LOCALMEASURE_RATE['NOR_ORD']) < 0, 
            0,
            df_MTA_LOCALMEASURE_RATE['NOR_FCST'] - df_MTA_LOCALMEASURE_RATE['NOR_ORD']
        )

    return df_MTA_LOCALMEASURE_RATE


def clean_local_measure_rate_data(df_MTA_LOCALMEASURE_RATE: DF):
    '''
    Local Measure Rate 데이터 정제 
    '''
    # df_MTA_LOCALMEASURE_RATE 정제: DELETE 처리
    if not df_MTA_LOCALMEASURE_RATE.empty:
        df_MTA_LOCALMEASURE_RATE = df_MTA_LOCALMEASURE_RATE[
            ~((df_MTA_LOCALMEASURE_RATE['IND_QTY'] + df_MTA_LOCALMEASURE_RATE['NOR_QTY'] == 0) |
              (df_MTA_LOCALMEASURE_RATE['IND_QTY'] == 0)
              )
        ].copy()

    # df_MTA_LOCALMEASURE_RATE UPDATE: RATE 산출
    if not df_MTA_LOCALMEASURE_RATE.empty:
        # RATE 산출 UPDATE
        df_MTA_LOCALMEASURE_RATE['TOT_QTY'] = df_MTA_LOCALMEASURE_RATE['IND_QTY'] + df_MTA_LOCALMEASURE_RATE['NOR_QTY']
        df_MTA_LOCALMEASURE_RATE['RATE'] = np.where(
            # df_MTA_LOCALMEASURE_RATE['TOT_QTY'] == 0, 0, np.round(
            #     df_MTA_LOCALMEASURE_RATE['IND_QTY'] /
            #     df_MTA_LOCALMEASURE_RATE['TOT_QTY'],
            #     2))
            df_MTA_LOCALMEASURE_RATE['TOT_QTY'] == 0, 0, 
                (df_MTA_LOCALMEASURE_RATE['IND_QTY'] /
                df_MTA_LOCALMEASURE_RATE['TOT_QTY']).round(2)
            ).astype(float)

        df_MTA_LOCALMEASURE_RATE = df_MTA_LOCALMEASURE_RATE.astype({'TOT_QTY': 'int32'})

    return df_MTA_LOCALMEASURE_RATE


def create_local_demand_from_localmeasurerate(
        df_da_item_seller_map: DF, df_dim_item: DF, df_region_asn: DF,
        df_BUF_SOPROMISE: DF, df_BUF_SOPROMISE_LOCAL: DF, df_MTA_LOCALMEASURE_RATE: DF):
    '''
    Local Measure 비율로 Local Demand 생성 
    '''

    # df_1 생성
    # SECTION 정보 추가 
    df_1_1 = df_da_item_seller_map.copy().merge(
        df_dim_item,
        left_on=[DAISM.ITEM],
        right_on=[I.ITEM],
        how='inner',
    )

    # MTA_SELLERMAP에서 TYPE='SELLER'로 필터링
    df_1_1 = df_1_1[df_1_1[DAISM.TYPE]=='SELLER']

    # MST_SDMREGIONASN 에서 ISVALID='Y'로 필터링
    df_1_2 = df_region_asn.copy()
    df_1_2 = df_1_2[df_1_2[RASN.ISVALID]=='Y']

    # df_1 생성 [📝 병합시 컬럼명 주의!]
    df_1 = df_1_1.merge(
        df_1_2,
        left_on=[DAISM.SITEID, DAISM.SALESID, I.SECTION],
        right_on=[RASN.SITEID, RASN.REGIONID, RASN.SECTION],
        how='inner',
    )
    # df_1 중복제거: subset=[DAISM.SITEID, DAISM.SALESID, RASN.SALESID]
    df_1 = df_1.drop_duplicates(subset=[DAISM.SITEID, DAISM.SALESID, RASN.SALESID])

    # df_1 필드명 변경
    df_1 = df_1[[RASN.SALESID, DAISM.SALESID, DAISM.SITEID]]
    df_1.rename(columns={RASN.SALESID : ND_.SALESID, DAISM.SALESID: 'LOCALID', DAISM.SITEID: ND_.SITEID}, inplace=True)

    # df_1 컬럼명 변경
    df_1.rename(columns={'LOCALID': ND_.LOCALID}, inplace=True)

    # df_2 생성: df_BUF_SOPROMISE 필터링
    df_2 = df_BUF_SOPROMISE.copy()
    df_2 = df_2[
        (~df_2[ND_.SALESORDERID].str.contains('ORD', na=False)) &
        (df_2[ND_.QTYPROMISED].astype(int) > 0)
    ]

    # df_2와 df_1 JOIN (SALESID, SITEID 기준)
    df_2 =  df_2.merge(
        df_1, 
        left_on=[ND_.SALESID, ND_.SITEID],
        right_on=[ND_.SALESID, ND_.SITEID],
        how='inner',
        suffixes=('_from_df_2', '_from_df_1'),
    )

    # LOCALID 값은 df_1(~from df_da_item_seller_map 의 값)
    df_2.rename(columns={ND_.LOCALID + '_from_df_1': ND_.LOCALID}, inplace=True)
    df_2.rename(columns={ND_.ITEMID + '_from_df_2': ND_.ITEMID}, inplace=True)    

    # 컬럼명 변경
    df_2['TOBE_LOCAL_ID'] = df_2[ND_.LOCALID]
    # df_2.rename(columns={'LOCALID': 'TOBE_LOCAL_ID'}, inplace=True)

    # df_3:  df_2와 df_MTA_LOCALMEASURE_RATE  Outer join
    df_3 = df_2.merge(
        df_MTA_LOCALMEASURE_RATE,
        left_on=[ND_.SALESID, ND_.SITEID, ND_.ITEMID, ND_.PROMISEDDELDATE + '_WEEK'],
        right_on=[LSaO.SALESID, LSaO.SITEID, LSaO.ITEM, LSaO.WEEK],
        how='left',
        suffixes=['', '_RATE']
    )

    # df_3 가공 컬럼
    df_3[ND_.SOLINENUM] = pd.to_numeric(df_3[ND_.SOLINENUM], errors='coerce').fillna(0).astype(int) + 100
    # df_3[ND_.QTYPROMISED] = np.round(df_3[ND_.QTYPROMISED] * df_3['RATE'].fillna(1).astype(int))
    df_3['RATE'] = df_3['RATE'].astype(float)
    df_3[ND_.QTYPROMISED] = np.floor(df_3[ND_.QTYPROMISED] * df_3['RATE'].fillna(1) + 0.5).astype(int)      # 오라클과 동일한 연산을 위해 + 0.5 처리

    df_3['INITDTTM'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_3['INITBY'] = 'local_reg.py'  #'DA_LOCAL_REG_DEV.ipynb'

    # index 중복 방지
    df_3 = df_3.copy().drop_duplicates(subset=['index'], keep='first')

    # INSERT 컬럼 리스트
    COL_TO_INSERT = [
        ND_.SOPROMISEID, ND_.PLANID, ND_.SALESORDERID, ND_.SOLINENUM, ND_.ITEMID, 
        ND_.QTYPROMISED, ND_.PROMISEDDELDATE, ND_.SITEID, ND_.SHIPTOID, ND_.SALESID,
        ND_.LOCALID, ND_.SALESLEVEL, ND_.DEMANDTYPERANK, ND_.WEEKRANK,
        ND_.CUSTOMERRANK, ND_.PRODUCTRANK, ND_.DEMANDPRIORITY,
        ND_.GLOBALPRIORITY, ND_.LOCALPRIORITY,
        ND_.AP2ID,
        ND_.GCID, ND_.MEASURETYPERANK, ND_.PREFERENCERANK, 'INITDTTM', 'INITBY', 'UPBY'
    ]

    # df_3[ND_.GC] = df_3['GCID']    # 컬럼명 매핑
    df_3[ND_.LOCALID] = df_3['TOBE_LOCAL_ID']

    # INSERT
    df_BUF_SOPROMISE_LOCAL = pd.concat(
        [df_BUF_SOPROMISE_LOCAL, df_3[COL_TO_INSERT]],
        ignore_index=True,
    )

    return df_BUF_SOPROMISE_LOCAL


def update_local_demand_from_demand_gap(
        df_da_item_seller_map: DF, df_dim_item: DF, df_region_asn: DF,
        df_BUF_SOPROMISE: DF, df_BUF_SOPROMISE_LOCAL: DF,
        df_MTA_LOCALMEASURE_RATE: DF):
    '''
    ORD 를 제외한 Demand 에 대해 수량 차이만큼을 Local Demand에 반영 
    '''

    # 'index' 컬럼이 있는 경우 reset_index() 로 'index' 컬럼 삭제
    if 'index' in df_BUF_SOPROMISE.columns:
        print('index')
        df_BUF_SOPROMISE = df_BUF_SOPROMISE.set_index('index')
        df_BUF_SOPROMISE.reset_index(drop=True, inplace=True)

    # 'index' 컬럼이 있는 경우 reset_index() 로 'index' 컬럼 삭제
    if 'index' in df_BUF_SOPROMISE_LOCAL.columns:
        print('index')
        df_BUF_SOPROMISE_LOCAL = df_BUF_SOPROMISE_LOCAL.set_index('index')
        df_BUF_SOPROMISE_LOCAL.reset_index(drop=True, inplace=True)


    # DMD: df_BUF_SOPROMISE 에서 'ORD' 제외하고 DMD_QTY 계산
    df_dmd = df_BUF_SOPROMISE.copy()
    df_dmd = df_dmd[
        (~df_dmd[ND_.SALESORDERID].str.contains('ORD', na=False)) &
        (df_dmd[ND_.QTYPROMISED] > 0)
    ]
    # Group By
    df_dmd = df_dmd.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK'], as_index=False)[ ND_.QTYPROMISED].sum()
    df_dmd.rename(columns={ND_.QTYPROMISED: 'DMD_QTY'}, inplace=True)

    # IND_DMD: df_BUF_SOPROMISE_LOCAL 에서 'ORD' 제외하고 IND_DMD_QTY 계산
    isoCalDate = df_BUF_SOPROMISE_LOCAL[ND_.PROMISEDDELDATE].dt.isocalendar()
    df_BUF_SOPROMISE_LOCAL[ND_.PROMISEDDELDATE + '_WEEK'] = isoCalDate.year.astype(str) + isoCalDate.week.astype(str).str.zfill(2)

    df_ind_dmd = df_BUF_SOPROMISE_LOCAL.copy()
    df_ind_dmd = df_ind_dmd[
        (~df_ind_dmd[ND_.SALESORDERID].str.contains('ORD', na=False)) &
        (df_ind_dmd[ND_.QTYPROMISED].astype(int) > 0)
    ]

    # Group By
    df_ind_dmd = df_ind_dmd.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK'], as_index=False)[ND_.QTYPROMISED].sum()
    df_ind_dmd.rename(columns={ND_.QTYPROMISED: 'IND_DMD_QTY'}, inplace=True)

    # df_IND_FCST: DMD, IND_DMD, RATE  JOIN 후 GAP 계산
    df_IND_FCST_0 = df_dmd.merge(
        df_ind_dmd,
        left_on=[ND_.SALESID, ND_.SITEID, ND_.ITEMID, ND_.PROMISEDDELDATE + '_WEEK'],
        right_on=[ND_.SALESID, ND_.SITEID, ND_.ITEMID, ND_.PROMISEDDELDATE + '_WEEK'],
        how='inner',
    )
    print(f"🍔 update_local_demand_from_demand_gap {df_IND_FCST_0.shape[0]}")

    df_IND_FCST = df_IND_FCST_0.merge(
        df_MTA_LOCALMEASURE_RATE,
        left_on=[ND_.SALESID, ND_.SITEID, ND_.ITEMID, ND_.PROMISEDDELDATE + '_WEEK'],
        right_on=[LSaO.SALESID, LSaO.SITEID, LSaO.ITEM, LSaO.WEEK],
        how='inner',
    )
    print(f"🥓 update_local_demand_from_demand_gap {df_IND_FCST.shape[0]}")

    # CALC_QTY 계산
    df_IND_FCST['RATE'] = df_IND_FCST['RATE'].astype(float)
    # df_IND_FCST['CALC_QTY'] = np.round(df_IND_FCST['DMD_QTY'] * df_IND_FCST['RATE'])      # np.round(255 * 07) = 178 로 출력됨. 오라클은 179
    df_IND_FCST['CALC_QTY'] = np.floor(df_IND_FCST['DMD_QTY'] * df_IND_FCST['RATE'] + 0.5).astype(int)      # 오라클과 동일한 연산을 위해 + 0.5 처리
    df_IND_FCST['GAP'] = df_IND_FCST['IND_DMD_QTY'] - df_IND_FCST['CALC_QTY']

    # V_REMAINQTY 초기화
    df_IND_FCST['V_REMAINQTY'] = df_IND_FCST['GAP'].astype(int)

    # GAP이 0이 아닌 경우만 필터링
    df_IND_FCST = df_IND_FCST[df_IND_FCST['GAP'] != 0].reset_index(drop=True)

    # df_exists 생성
    df_X = df_da_item_seller_map.copy().merge(
        df_dim_item,
        left_on=[I.ITEM],
        right_on=[I.ITEM],
        how='inner',
    )

    # MST_SDMREGIONASN 에서 ISVALID='Y'로 필터링
    df_Y = df_region_asn.copy()
    df_Y = df_Y[df_Y[RASN.ISVALID] == 'Y']

    # df_exists 생성 [📝 병합시 컬럼명 주의!]
    df_exists = df_X.merge(
        df_Y,
        left_on=[DAISM.SITEID, DAISM.SALESID, I.SECTION],
        right_on=[RASN.SITEID, RASN.REGIONID, RASN.SECTION],
        how='inner',
    )
    # df_exists 중복제거: subset=[DAISM.SITEID, DAISM.SALESID, RASN.SALESID]
    df_exists = df_exists.drop_duplicates(subset=[DAISM.SITEID, DAISM.SALESID, RASN.SALESID])

    # df_exists 필드명 변경
    df_exists = df_exists[[RASN.SALESID, DAISM.SALESID, DAISM.SITEID]]
    df_exists.rename(columns={RASN.SALESID: ND_.SALESID, DAISM.SALESID: 'LOCALID', DAISM.SITEID: ND_.SITEID}, inplace=True)

    # df_sopromise 생성 (BUF_SOPROMISE_LOCAL 필터링)
    df_BUF_SOPROMISE_LOCAL = df_BUF_SOPROMISE_LOCAL.reset_index()
    df_SOPROMISE = df_BUF_SOPROMISE_LOCAL.copy()
    df_SOPROMISE = df_SOPROMISE[~df_SOPROMISE[ND_.SALESORDERID].str.contains( 'ORD')]
    df_SOPROMISE = df_SOPROMISE[df_SOPROMISE[ND_.QTYPROMISED].astype(int) > 0]
    
    print(f"🍿 update_local_demand_from_demand_gap {df_SOPROMISE.shape[0]}")

    # IND_FCST 와 JOIN (WEEK, SITEID, SALESID, ITEM 기준)
    df_SOPROMISE = df_SOPROMISE.merge(
        df_IND_FCST,
        left_on=[ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK'],
        right_on=[LSaO.ITEM, LSaO.SITEID, LSaO.SALESID, LSaO.WEEK],
        how='inner', 
        suffixes=('', '_indFCST')
    )

    print(f"🍕 update_local_demand_from_demand_gap {df_SOPROMISE.shape[0]}")

    if df_SOPROMISE.shape[0] > 0:
        print('Hey')

        # Exists 조건 필터링: df_exists 에서 발췌한 데이터의 salesid, siteid 기준으로 존재하는 것만 (inner join) 만족하는 대상으로 필터링
        df_SOPROMISE = df_SOPROMISE[
            (df_SOPROMISE[ND_.SALESID] + df_SOPROMISE[ND_.SITEID]).isin(
                df_exists[ND_.SALESID] + df_exists[ND_.SITEID]
            )
        ]

        if df_SOPROMISE.shape[0] > 0:
            # DEMANDPRIORITY 및 SOPROMISED 정렬
            df_SOPROMISE = df_SOPROMISE.sort_values(by=[ND_.DEMANDPRIORITY], ascending=[False])

            ####################################################################
            # 🎰 QTY 차감 처리: adjust_qtypromised_IND_FCST
            df_SOPROMISE['CUMSUM_QTY'] = df_SOPROMISE.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK'])[ND_.QTYPROMISED].cumsum()

            # REMAINQTY 계산
            df_SOPROMISE['REMAINQTY'] = df_SOPROMISE['V_REMAINQTY'].astype(int) - df_SOPROMISE['CUMSUM_QTY'].astype(int)

            # df_SOPROMISE = df_SOPROMISE.apply(adjust_qtypromised_IND_FCST)
            df_SOPROMISE = df_SOPROMISE.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE + '_WEEK'], group_keys=False).apply(adjust_qtypromised_IND_FCST)
            ####################################################################

            # UPDATE: SOPROMISE (df_exp_sopromise_src)
            print(f" df_BUF_SOPROMISE_LOCAL.info: {df_BUF_SOPROMISE_LOCAL.info()}")
            print(f" df_SOPROMISE.info: {df_SOPROMISE.info()}")
            df_BUF_SOPROMISE_LOCAL.set_index('index', inplace=True)
            df_SOPROMISE.set_index('index', inplace=True)

            df_BUF_SOPROMISE_LOCAL.loc[df_SOPROMISE.index.intersection(df_BUF_SOPROMISE.index), ND_.QTYPROMISED] = df_SOPROMISE.loc[df_SOPROMISE.index.intersection(df_BUF_SOPROMISE_LOCAL.index), ND_.QTYPROMISED]
            df_BUF_SOPROMISE_LOCAL.loc[df_SOPROMISE.index.intersection(df_BUF_SOPROMISE.index), 'UPBY'] = df_SOPROMISE.loc[df_SOPROMISE.index.intersection(df_BUF_SOPROMISE_LOCAL.index), 'UPBY']     # 검증용 UPBY 기록
            df_BUF_SOPROMISE_LOCAL = df_BUF_SOPROMISE_LOCAL.reset_index()
            print(f"running @ : {time.time()}")

    return df_BUF_SOPROMISE_LOCAL


def build_local_demand_for_hub(
        df_da_item_seller_map: DF, df_dim_item: DF, df_region_asn: DF,
        df_BUF_SOPROMISE: DF, df_BUF_SOPROMISE_LOCAL: DF):
    '''
    Hub 대상에 대한 Local Demand 구성 
    '''

    # df_1 생성
    # SECTION 정보 추가
    df_1_1 = df_da_item_seller_map.copy().merge(
        df_dim_item,
        left_on=[DAISM.ITEM],
        right_on=[I.ITEM],
        how='inner',
    )

    # MTA_SELLERMAP에서 TYPE='SELLER'로 필터링
    df_1_1 = df_1_1[df_1_1[DAISM.TYPE] == 'HUB']

    # MST_SDMREGIONASN 에서 ISVALID='Y'로 필터링
    df_1_2 = df_region_asn.copy()
    df_1_2 = df_1_2[df_1_2[RASN.ISVALID] == 'Y']

    # df_1 생성 [📝 병합시 컬럼명 주의!]
    df_1 = df_1_1.merge(
        df_1_2,
        left_on=[DAISM.SITEID, DAISM.SALESID, I.SECTION],
        right_on=[RASN.SITEID, RASN.REGIONID, RASN.SECTION],
        how='inner',
    )
    # df_1 중복제거: subset=[DAISM.SITEID, DAISM.SALESID, RASN.SALESID]
    df_1 = df_1.drop_duplicates(subset=[DAISM.SALESID, DAISM.SITEID, RASN.SALESID])

    # df_1 필드명 변경
    df_1 = df_1[[RASN.SALESID, DAISM.SALESID, DAISM.SITEID]]
    df_1.rename(columns={RASN.SALESID : ND_.SALESID, DAISM.SALESID: 'LOCAL_ID', DAISM.SITEID: ND_.SITEID}, inplace=True)

    # df_2 생성: df_BUF_SOPROMISE 필터링
    df_2 = df_BUF_SOPROMISE.copy()
    df_2[ND_.SOLINENUM + '_100'] = df_2[ND_.SOLINENUM].astype(int) - 100

    df_2 = df_2[
        (~df_2[ND_.SALESORDERID].str.contains('ORD', na=False))
        & (df_2[ND_.QTYPROMISED].astype(int) > 0)
    ]
    # df_2를 df_1과 JOIN (SALESID, SITEID 기준)
    df_2 = df_2.merge(
        df_1,
        left_on=[ND_.SALESID, ND_.SITEID],
        right_on=[ND_.SALESID, ND_.SITEID],
        how='inner',
    )
    # 컬럼명 변경
    # df_2.rename(columns={'LOCAL_ID': 'LOCAL_ID'}, inplace=True)
    df_2.rename(columns={'LOCAL_ID': 'TOBE_LOCAL_ID'}, inplace=True)

    # df_3 생성: df_2와 df_BUF_SOPROMISE_LOCAL Outer join
    df_2[ND_.SOLINENUM] = df_2[ND_.SOLINENUM].astype(int)
    df_2_1 = df_BUF_SOPROMISE_LOCAL.copy()
    df_2_1[ND_.SOLINENUM] = df_2_1[ND_.SOLINENUM].astype(int) - 100

    df_3 = df_2.merge(
        df_2_1,
        left_on=[ND_.SALESORDERID, ND_.SOLINENUM],
        right_on=[ND_.SALESORDERID, ND_.SOLINENUM],
        how='left',
        suffixes=('', '_IND')
    )

    # df_3 가공컬럼
    df_3[ND_.QTYPROMISED] = df_3[ND_.QTYPROMISED].astype(int) - df_3[ND_.QTYPROMISED + '_IND'].fillna(0).astype(int)
    df_3[ND_.LOCALID] = df_3['TOBE_LOCAL_ID']
    # df_3[ND_.GCID] = df_3['GCID']

    # 필요한 컬럼만 선택
    # INSERT 컬럼 리스트
    COL_TO_INSERT = [
        ND_.SOPROMISEID, ND_.PLANID, ND_.SALESORDERID, ND_.SOLINENUM, ND_.ITEMID, 
        ND_.QTYPROMISED, ND_.PROMISEDDELDATE, ND_.SITEID, ND_.SHIPTOID, ND_.SALESID,
        ND_.LOCALID, ND_.SALESLEVEL, ND_.DEMANDTYPERANK, ND_.WEEKRANK, ND_.CHANNELRANK,
        ND_.CUSTOMERRANK, ND_.PRODUCTRANK, ND_.DEMANDPRIORITY, ND_.TIEBREAK,
        ND_.GLOBALPRIORITY, ND_.LOCALPRIORITY, ND_.BUSINESSTYPE, ND_.ROUTING_PRIORITY,
        ND_.NO_SPLIT, ND_.MAP_SATISFY_SS, ND_.PREALLOC_ATTRIBUTE, ND_.BUILDAHEADTIME,
        ND_.TIMEUOM, ND_.AP2ID, ND_.GCID, ND_.MEASURETYPERANK, ND_.PREFERENCERANK,  
        'INITDTTM', 'INITBY'
    ]

    df_3_final = df_3[COL_TO_INSERT].copy()
    df_3_final['INITDTTM'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_3_final['INITBY'] = ''

    # INSERT
    df_BUF_SOPROMISE_LOCAL = pd.concat(
        [df_BUF_SOPROMISE_LOCAL, df_3_final[COL_TO_INSERT]],
        ignore_index=True,
    )

    return df_BUF_SOPROMISE_LOCAL



def assemble_inbound_data(
        df_da_item_seller_map: DF, df_dim_item: DF, df_region_asn: DF,
        df_exp_sopromise_src: DF, v_planid: str):
    '''
    inbound (EXP_SOPROMISE_SRC) 에 지금까지 계산한 데이터를 구성
    '''

    # df_1 생성
    # SECTION 정보 추가
    df_1_1 = df_da_item_seller_map.copy().merge(
        df_dim_item,
        left_on=[DAISM.ITEM],
        right_on=[I.ITEM],
        how='inner',
    )

    # MST_SDMREGIONASN 에서 ISVALID='Y'로 필터링
    df_1_2 = df_region_asn.copy()
    df_1_2 = df_1_2[df_1_2[RASN.ISVALID] == 'Y']

    # df_1 생성 [📝 병합시 컬럼명 주의!]
    df_1 = df_1_1.merge(
        df_1_2,
        left_on=[DAISM.SITEID, DAISM.SALESID, I.SECTION],
        right_on=[RASN.SITEID, RASN.REGIONID, RASN.SECTION],
        how='inner',
    )  # [[DAISM.SITEID, RASN.SALESID, DAISM.ITEM]]

    # # df_1 중복제거: subset=[DAISM.SITEID, DAISM.SALESID, RASN.SALESID]
    # df_1 = df_1.drop_duplicates(subset=[DAISM.SALESID, DAISM.SITEID, RASN.SALESID])

    # df_1 필드명 변경
    df_1 = df_1[[RASN.SALESID, DAISM.SALESID, DAISM.SITEID, DAISM.ITEM]]
    df_1.rename(
        columns={RASN.SALESID: ND_.SALESID, DAISM.SALESID: 'LOCALID',
                 DAISM.SITEID: ND_.SITEID, DAISM.ITEM: ND_.ITEMID},
        inplace=True)

    # GC 컬럼 추가
    # df_exp_sopromise_src[ND_.GCID] = df_exp_sopromise_src['GCID']

    # inbound(EXP_SOPROMISE_SRC) 에서 df_1 에 해당하는 siteid, item, salesid 조건은 삭제
    df_to_delete = df_exp_sopromise_src[df_exp_sopromise_src[ND_.PLANID] == v_planid]
    df_to_delete = df_to_delete.merge(
        df_1,
        on=[ND_.SITEID, ND_.ITEMID, ND_.SALESID],
        how='inner',
    )

    # DELETE
    # df_exp_sopromise_src = df_exp_sopromise_src[~df_exp_sopromise_src['index'].isin(df_to_delete['index'])]
    df_exp_sopromise_src = df_exp_sopromise_src[~df_exp_sopromise_src.set_index(
        [ND_.SALESORDERID, ND_.SOLINENUM]).index.isin(df_to_delete.set_index([ND_.SALESORDERID, ND_.SOLINENUM]).index)]

    print(f"Deleted {df_to_delete.shape[0]} from df_exp_sopromise_src")
    return df_exp_sopromise_src



def insert_buf_sopromise_local_to_inbound(
        df_BUF_SOPROMISE_LOCAL: DF, df_exp_sopromise_src: DF, v_planid: str):
    '''
    df_BUF_SOPROMISE_LOCAL 데이터를 inbound 데이터에 insert
    '''

    df_to_insert = df_BUF_SOPROMISE_LOCAL[df_BUF_SOPROMISE_LOCAL[ND_.PLANID] == v_planid]
    df_to_insert['INITBY'] = 'local_reg.py'

    # INSERT 컬럼 리스트
    COL_TO_INSERT = [
        ND_.SOPROMISEID, ND_.PLANID, ND_.SALESORDERID, ND_.SOLINENUM, ND_.ITEMID, 
        ND_.QTYPROMISED, ND_.PROMISEDDELDATE, ND_.SITEID, ND_.SHIPTOID, ND_.SALESID,
        ND_.LOCALID, ND_.SALESLEVEL, ND_.DEMANDTYPERANK, ND_.WEEKRANK, ND_.CHANNELRANK,
        ND_.CUSTOMERRANK, ND_.PRODUCTRANK, ND_.DEMANDPRIORITY, ND_.TIEBREAK,
        ND_.GLOBALPRIORITY, ND_.LOCALPRIORITY, ND_.BUSINESSTYPE, ND_.ROUTING_PRIORITY,
        ND_.NO_SPLIT, ND_.MAP_SATISFY_SS, ND_.PREALLOC_ATTRIBUTE, ND_.BUILDAHEADTIME,
        ND_.TIMEUOM, ND_.AP2ID, ND_.GCID, ND_.MEASURETYPERANK, ND_.PREFERENCERANK,  
        'INITDTTM', 'INITBY', 'UPBY'
    ]

    df_to_insert_final = df_to_insert[COL_TO_INSERT]
    df_exp_sopromise_src = pd.concat(
        [df_exp_sopromise_src, df_to_insert_final],
        ignore_index=True,
    )

    # # 확인용
    # print("🍺🍺 insert_buf_sopromise_local_to_inbound")
    # print(df_exp_sopromise_src.info())

    return df_exp_sopromise_src


def assign_localid_to_missing(
        df_da_item_seller_map: DF, df_dim_item: DF, df_region_asn: DF,
        df_exp_sopromise_src: DF):
    '''
    LOCALID 없는 항목에 대해서 LOCALID 구성    
    '''

    # df_1 생성
    # SECTION 정보 추가
    df_1_1 = df_da_item_seller_map.copy().merge(
        df_dim_item,
        left_on=[DAISM.ITEM],
        right_on=[I.ITEM],
        how='inner',
    )

    # MST_SDMREGIONASN 에서 ISVALID='Y'로 필터링
    df_1_2 = df_region_asn.copy()
    df_1_2 = df_1_2[df_1_2[RASN.ISVALID] == 'Y']

    # df_1 생성 [📝 병합시 컬럼명 주의!]
    df_1 = df_1_1.merge(
        df_1_2,
        left_on=[I.SECTION, DAISM.SITEID, DAISM.SALESID],
        right_on=[RASN.SECTION, RASN.SITEID, RASN.REGIONID],
        how='inner',
    )  # [[DAISM.SITEID, RASN.SALESID, DAISM.ITEM]]       # DAISM.SALESID : LOCALID

    # df_1 필드명 변경
    df_1 = df_1[[RASN.SALESID, DAISM.SALESID, DAISM.SITEID, DAISM.ITEM]]
    df_1.rename(columns={RASN.SALESID: ND_.SALESID, DAISM.SALESID: 'LOCALID', DAISM.SITEID: ND_.SITEID, DAISM.ITEM: ND_.ITEMID},
        inplace=True)

    # df_1 중복제거:
    df_1 = df_1.drop_duplicates(subset=[ND_.SITEID, ND_.ITEMID])

    # V_MTA_SELLERMAP (Netting IF DA Item Seller Map) 로 부터 LOCALID 변수 생성
    df_hub_sales = df_da_item_seller_map[
        (df_da_item_seller_map[DAISM.TYPE] == 'HUB')
        & (df_da_item_seller_map[DAISM.SALESID].str.startswith('R'))
    ]
    V_LOCAL_ID_TMP = df_hub_sales[DAISM.SALESID].min()
    print(f"V_LOCAL_ID_TMP: {V_LOCAL_ID_TMP}")

    # LOCALID 가 NULL인 데이터 필터링
    df_to_update = df_exp_sopromise_src[df_exp_sopromise_src[ND_.LOCALID].isna()]

    # df_1의 SITEID, ITEM 조건으로 필터링
    df_to_update = df_to_update.merge(
        df_1,
        on=[ND_.SITEID, ND_.ITEMID],
        how='inner',
        suffixes=('', '_2'),
    )

    # df_to_update KEY 중복제거
    df_to_update = df_to_update.drop_duplicates(subset=[ND_.SALESORDERID, ND_.SOLINENUM])

    # INDEX 설정 [SALESORDERID, SOLINENUM]
    df_exp_sopromise_src.set_index([ND_.SALESORDERID, ND_.SOLINENUM], inplace=True)
    df_to_update.set_index([ND_.SALESORDERID, ND_.SOLINENUM], inplace=True)

    # LOCALID  업데이트
    # df_exp_sopromise_src.loc[df_to_update.index, 'LOCALID'] = V_LOCAL_ID_TMP
    df_exp_sopromise_src.loc[df_to_update.index,  ND_.LOCALID] = V_LOCAL_ID_TMP

    # RESET INDEX
    df_exp_sopromise_src.reset_index(inplace=True)

    # # 확인용
    # print("🍺🍺 assign_localid_to_missing")
    # print(df_exp_sopromise_src.info())

    return df_exp_sopromise_src


#############################
##### demand_level main #####
#############################
def do_match_code_local(df_exp_sopromise_src: DF, df_da_item_seller_map: DF, df_dim_item: DF, df_region_asn: DF, 
                        df_local_sales_order: DF, df_dim_netting_sales: DF, df_local_measure: DF, logger: LOGGER):
    '''
    SHASCM.SP_DN_ODS_TO_SDB_REG
    '''

    V_FILENAME = 'match_code_local.py'
    if is_file_type_py():
        V_FILENAME = {os.path.basename(__file__)}
    else:
        V_FILENAME = V_FILENAME

    # ⏲ 시간측정 시작
    start = time.time()
    print(f"🥑 START: {V_FILENAME}")
    logger.Note(f"Start {V_FILENAME}", 20)

    ### 🎱 계산을 위한 중간 테이블 생성 : Netted Demand 중 Local 대상
    logger.Step(1, f"Start: {'create_netted_demand_table'}")
    df_BUF_SOPROMISE = create_netted_demand_table(df_exp_sopromise_src, df_da_item_seller_map, df_dim_item, df_region_asn)
    logger.Step(1, f"End: {'create_netted_demand_table'}")

    # print('🐞🐞🐞 1. df_BUF_SOPROMISE.info()')
    # print(df_BUF_SOPROMISE.info())

    ### 🎱 계산을 위한 중간 테이블 생성 : SalesOrder 중 Local 대상
    logger.Step(2, f"Start: {'create_local_sales_order_table'}")
    df_BUF_SOPROMISE_LOCAL = create_local_sales_order_table(df_da_item_seller_map, df_dim_item, df_region_asn, df_local_sales_order, df_dim_netting_sales, V_PLANID)
    logger.Step(2, f"End: {'create_local_sales_order_table'}")

    # print('🐞🐞🐞 2. df_BUF_SOPROMISE_LOCAL.info()')
    # print(df_BUF_SOPROMISE_LOCAL.info())

    ### 🎱 강제로 추가된 COM_ORD에 우선순위를 매핑
    logger.Step(3, f"Start: {'map_priority_to_com_ord'}")
    df_BUF_SOPROMISE_LOCAL = map_priority_to_com_ord(df_BUF_SOPROMISE, df_BUF_SOPROMISE_LOCAL)
    logger.Step(3, f"End: {'map_priority_to_com_ord'}")

    ### 🎱 Demand 와 Local Demand 간의 Demand Qty Gap 산출하여 QTYPROMISED 할당
    logger.Step(4, f"Start: {'allocate_qty_promised'}")
    df_BUF_SOPROMISE_LOCAL = allocate_qty_promised(df_BUF_SOPROMISE, df_BUF_SOPROMISE_LOCAL)
    logger.Step(4, f"End: {'allocate_qty_promised'}")

    ### 🎱 IND_COM_ORD 수량만큼을 Demand에서 차감
    logger.Step(5, f"Start: {'deduct_demand_by_ind_com_ord'}")
    df_BUF_SOPROMISE = deduct_demand_by_ind_com_ord(df_da_item_seller_map, df_dim_item, df_region_asn, df_local_sales_order, df_dim_netting_sales, df_BUF_SOPROMISE)
    logger.Step(5, f"End: {'deduct_demand_by_ind_com_ord'}")

    ### 🎱 DEMAND 정보에 LOCALID 를 붙여서 Local Demand 생성
    logger.Step(6, f"Start: {'create_local_demand_ith_localid_com_ord'}")
    df_BUF_SOPROMISE_LOCAL = create_local_demand_ith_localid_com_ord(df_da_item_seller_map, df_dim_item, df_region_asn, df_BUF_SOPROMISE, df_BUF_SOPROMISE_LOCAL)
    logger.Step(6, f"End: {'create_local_demand_ith_localid_com_ord'}")

    ### 🎱 DEMAND 정보에 LOCALID 를 붙여서 Local Demand 생성
    logger.Step(7, f"Start: {'create_local_demand_ith_localid_new_ord'}")
    df_BUF_SOPROMISE_LOCAL = create_local_demand_ith_localid_new_ord(df_da_item_seller_map, df_dim_item, df_region_asn, df_BUF_SOPROMISE, df_BUF_SOPROMISE_LOCAL)
    logger.Step(7, f"End: {'create_local_demand_ith_localid_new_ord'}")

    ### 🎱 Local Measure Rate 중간 테이블 생성
    logger.Step(8, f"Start: {'create_local_measure_rate_table'}")
    df_MTA_LOCALMEASURE_RATE = create_local_measure_rate_table(df_da_item_seller_map, df_local_measure, df_local_sales_order, df_dim_item)
    logger.Step(8, f"End: {'create_local_measure_rate_table'}")

    ### 🎱 Local Measure Rate 데이터 정제
    logger.Step(9, f"Start: {'clean_local_measure_rate_data'}")
    df_MTA_LOCALMEASURE_RATE = clean_local_measure_rate_data(df_MTA_LOCALMEASURE_RATE)
    logger.Step(9, f"End: {'clean_local_measure_rate_data'}")

    ### 🎱 Local Measure 비율로 Local Demand 생성
    logger.Step(10, f"Start: {'create_local_demand_from_localmeasurerate'}")
    df_BUF_SOPROMISE_LOCAL = create_local_demand_from_localmeasurerate(df_da_item_seller_map, df_dim_item, df_region_asn, df_BUF_SOPROMISE, df_BUF_SOPROMISE_LOCAL, df_MTA_LOCALMEASURE_RATE)
    logger.Step(10, f"End: {'create_local_demand_from_localmeasurerate'}")

    ### 🎱 ORD 를 제외한 Demand 에 대해 수량 차이만큼을 Local Demand에 반영
    logger.Step(11, f"Start: {'update_local_demand_from_demand_gap'}")
    df_BUF_SOPROMISE_LOCAL = update_local_demand_from_demand_gap(df_da_item_seller_map, df_dim_item, df_region_asn, df_BUF_SOPROMISE, df_BUF_SOPROMISE_LOCAL, df_MTA_LOCALMEASURE_RATE)
    logger.Step(11, f"End: {'update_local_demand_from_demand_gap'}")

    ### 🎱 Hub 대상에 대한 Local Demand 구성
    logger.Step(12, f"Start: {'build_local_demand_for_hub'}")
    df_BUF_SOPROMISE_LOCAL = build_local_demand_for_hub(df_da_item_seller_map, df_dim_item, df_region_asn, df_BUF_SOPROMISE, df_BUF_SOPROMISE_LOCAL)
    logger.Step(12, f"End: {'build_local_demand_for_hub'}")

    ### 🎱 inbound (EXP_SOPROMISE_SRC) 에 지금까지 계산한 데이터를 구성
    logger.Step(13, f"Start: {'assemble_inbound_data'}")
    df_exp_sopromise_src = assemble_inbound_data(df_da_item_seller_map, df_dim_item, df_region_asn, df_exp_sopromise_src, V_PLANID)
    logger.Step(13, f"End: {'assemble_inbound_data'}")

    ### 🎱 df_BUF_SOPROMISE_LOCAL 데이터를 inbound 데이터에 insert
    logger.Step(14, f"Start: {'insert_buf_sopromise_local_to_inbound'}")
    df_exp_sopromise_src = insert_buf_sopromise_local_to_inbound(df_BUF_SOPROMISE_LOCAL, df_exp_sopromise_src, V_PLANID)
    logger.Step(14, f"End: {'insert_buf_sopromise_local_to_inbound'}")
    
    ### 🎱 LOCALID 없는 항목에 대해서 LOCALID 구성
    logger.Step(15, f"Start: {'assign_localid_to_missing'}")
    df_exp_sopromise_src = assign_localid_to_missing(df_da_item_seller_map, df_dim_item, df_region_asn, df_exp_sopromise_src)
    logger.Step(15, f"End: {'assign_localid_to_missing'}")


    # ⏲ 수행시간 출력
    print(f"🍤 FINISHED: {V_FILENAME}")
    print(f"⏲ {V_FILENAME}: Execution time: {time.time() - start:.5f} seconds")
    logger.Note(f"End {V_FILENAME}", 20)

    return df_exp_sopromise_src[ND_.LIST_COLUMN]
