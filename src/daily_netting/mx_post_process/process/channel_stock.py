#region import

import os, sys

from collections import OrderedDict
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import time

from daily_netting.mx_post_process.utils import common
from daily_netting.mx_post_process.utils import NSCMCommon
from daily_netting.mx_post_process.constant.dim_constant import NettingSales as NS
from daily_netting.mx_post_process.constant.dim_constant import Item as I
from daily_netting.mx_post_process.constant.ods_constant import NettedDemandD as ND_
from daily_netting.mx_post_process.constant.ods_constant import AP1ShortReasonD as AP1SR_
from daily_netting.mx_post_process.constant.mx_constant import DPGuide as DPG
from daily_netting.mx_post_process.constant.mx_constant import MXItemSellerMap as MXISM
from daily_netting.mx_post_process.constant.mx_constant import ESItemSite as ESIS
from daily_netting.mx_post_process.constant.mx_constant import CustomModelMap as CMM
from daily_netting.mx_post_process.constant.mx_constant import ChannelStockMaster as CSM
from daily_netting.mx_post_process.constant.mx_constant import CodeMap as CM
from daily_netting.mx_post_process.constant.mx_constant import AvailableResourceD as AR_
from daily_netting.mx_post_process.constant.mx_constant import ChannelStockAP2Master as CSAP2M
from daily_netting.mx_post_process.constant.mx_constant import ShortReasonDPGuide as SRDPG

#endregion import

def set_channel_stock_env(accessor: object) -> None:
    global V_PLANID, V_EFFSTARTDATE, V_EFFENDDATE, V_PLANWEEK, V_TYPE, V_NUM, \
        V_DPGUIDE, find_priority_position

    V_PLANID = accessor.plan_id
    V_EFFSTARTDATE = accessor.start_date.strftime('%Y-%m-%d')
    V_EFFENDDATE = accessor.end_date.strftime('%Y-%m-%d')
    V_PLANWEEK = accessor.plan_week
    V_TYPE = accessor.plan_type
    V_NUM = 0
    V_DPGUIDE = 0

    if (V_TYPE == 'VPLAN') :
        V_NUM = 1
    
    # 자리수 확인 함수
    find_priority_position = accessor.find_priority_position

# 해당주차의 마지막 날 구하기 (to에 해당하는 날짜 :일요일)
def get_iso_week_end(iso_year_week):
    iso_year = int(iso_year_week[:4])
    iso_week = int(iso_year_week[4:])
    jan_4 = datetime(iso_year, 1, 4)
    first_monday = jan_4 - timedelta(days=jan_4.weekday())
    week_end = first_monday + timedelta(weeks=iso_week - 1, days=6)
    return week_end.strftime('%Y-%m-%d')

#region class정의
'''
class 정의
'''
class SSConstant:
    # SALESID, SITEID, ITEM, PROMISEDDELDATE, UNFORDQTY, CHSTOCKQTY,  UNFORDSUM, CHSTOCKSUM, RNK
    SALESID_IDX             =  0
    SITEID_IDX              =  1
    ITEM_IDX                =  2
    PROMISEDDELDATE_IDX     =  3
    UNFORDQTY_IDX           =  4
    CHSTOCKQTY_IDX          =  5
    UNFORDSUM_IDX           =  6
    CHSTOCKSUM_IDX          =  7
    RNK_IDX                 =  8

class DpguideConstant:
    #   AP2ID	LEVELID	SALESID	UDAITEM	SWEEK	QTY
    AP2ID_IDX           =  0
    LEVELID_IDX         =  1
    SALESID_IDX         =  2
    UDAITEM_IDX         =  3
    SWEEK_IDX           =  4
    QTY_IDX             =  5
#endregion class정의

#region 전역변수정의
'''
전역변수정의
'''
V_PLANID = None
V_EFFSTARTDATE = None
V_EFFENDDATE = None
V_PLANWEEK = None
V_TYPE = None

V_ALIVE   = 0
V_ROLLING = 0
V_CHSTOCK = 0
V_AVAILQTY2 = 0

V_NUM = 0
V_DPGUIDE = 0

find_priority_position = None
#endregion 전역변수정의

#region Alias정의
'''
Alias 정의
'''
LOGGER = NSCMCommon.G_Logger
DF      = pd.DataFrame
NA      = np.array

DPGUIDE = DpguideConstant
SS      = SSConstant
#endregion Alias정의


def f_dpguide(level_id : str, df_exp_dpguide:DF, df_gui_saleshierarchy:DF, df_vui_itemattb:DF, df_Netting_Custom_Model_Map_ODS_W:DF, df_in_mst_chstock:DF ) -> pd.DataFrame:
    '''
    설명: df_exp_dpguide를 원하는 level_id(sales level)로 groupby해 sum한 후,
    제약구간으로 filtering 한다. (df_EXP_DPGUIDE_DEL와 다른 제약구간을 적용한다.)

    이력:
    / 20205.09.19 곽영빈 프로 요청 - SIG 산출 시 제약구간 반영으로 Netting에서 해당 기능 사용 중지
    '''
    # PRODUCTGROUP 컬럼은 없음!
    df_exp_dpguide = df_exp_dpguide[[DPG.SALESID, DPG.UDAITEM, DPG.SWEEK, DPG.QTY]].copy()
    df_exp_dpguide[DPG.QTY] = df_exp_dpguide[DPG.QTY].fillna(0).astype('int32')
    
    df_gui_saleshierarchy = df_gui_saleshierarchy[[NS.SALESID, NS.LEVELID,  NS.AP2ID,]]
    df_vui_itemattb = df_vui_itemattb[[I.ITEM, I.BASICNAME, ]]
    df_dpguide_01 = df_exp_dpguide.loc[
            ~(df_exp_dpguide[DPG.UDAITEM]+'Y').isin(
                    df_Netting_Custom_Model_Map_ODS_W[CMM.CUSTOMITEM]+
                            df_Netting_Custom_Model_Map_ODS_W[CMM.ISVALID])
    ]

    df_in_gui_sales = df_gui_saleshierarchy.loc[df_gui_saleshierarchy[NS.LEVELID]==level_id]
    df_dpguide_02 = pd.merge(df_dpguide_01, df_in_gui_sales, how='inner',
            left_on=[DPG.SALESID], right_on=[NS.SALESID])
    df_dpguide_03 = pd.merge(df_dpguide_02, df_vui_itemattb, how='inner',
            left_on=[DPG.UDAITEM], right_on=[I.ITEM])

    df_dpguide_04 = df_dpguide_03.groupby(by=[NS.AP2ID,NS.LEVELID,NS.SALESID,I.BASICNAME,DPG.UDAITEM,DPG.SWEEK])[[DPG.QTY]].sum()
    df_dpguide_04 = df_dpguide_04.reset_index()

    # / 20205.09.19 곽영빈 프로 요청 - SIG 산출 시 제약구간 반영으로 Netting에서 해당 기능 사용 중지
    # df_dpguide_05  = pd.merge(df_dpguide_04, df_in_mst_chstock, how='left', left_on=[NS.SALESID,DPG.UDAITEM], right_on=[CSM.SALESID,CSM.ITEM,])
    # df_dpguide_05['STARTWEEK'] = common.convert_dt_to_week(pd.to_datetime(df_dpguide_05[CSM.EFFSTARTDATE].fillna(pd.Timestamp.min)))
    # df_dpguide_05['ENDWEEK'] = common.convert_dt_to_week(pd.to_datetime(df_dpguide_05[CSM.EFFENDDATE].fillna(pd.Timestamp.max)))
    # df_dpguide_06 = df_dpguide_05.loc[df_dpguide_05[DPG.SWEEK].between(df_dpguide_05['STARTWEEK'], df_dpguide_05['ENDWEEK'])]
    # df_dpguide_06 = df_dpguide_06[[NS.AP2ID,NS.LEVELID,NS.SALESID, DPG.UDAITEM, DPG.SWEEK, DPG.QTY, 'STARTWEEK', 'ENDWEEK']]
    df_dpguide_06 = df_dpguide_04[[NS.AP2ID,NS.LEVELID,NS.SALESID, DPG.UDAITEM, DPG.SWEEK, DPG.QTY,]]
    
    df_dpguide_return = df_dpguide_06.sort_values(by=[NS.AP2ID,NS.LEVELID,NS.SALESID,DPG.UDAITEM, DPG.SWEEK,])
    return df_dpguide_return



# 기존 channel_stock 프로그램 처리 전 DPGUIDE 데이터 삭제 처리

def df_EXP_DPGUIDE_DEL(df_in_exp_dpguide
                    , df_in_mst_chstock
                    , df_in_mst_chstock_ap2
                    , df_in_gui_saleshierarchy # AP2ID 추가 필요 7/18
                    , df_in_vui_itemattb ) -> DF:
    '''
    설명: df_in_exp_dpguide를 제약구간으로 filtering 한다.
    f_dpguide에서와 다른 제약구간을 적용한다.

    이력:
    / 20205.09.19 곽영빈 프로 요청 - SIG 산출 시 제약구간 반영으로 Netting에서 해당 기능 사용 중지
    '''
    df_exp_dpguide = pd.merge(df_in_exp_dpguide
                            , df_in_vui_itemattb[[I.ITEM, I.PRODUCTGROUP]]
                            , left_on=[DPG.UDAITEM]
                            , right_on=[I.ITEM])

    df_mst_chstock = pd.merge(df_in_mst_chstock
                            , df_in_gui_saleshierarchy[[NS.SALESID, NS.AP2ID]]
                            , left_on=[CSM.SALESID]
                            , right_on=[NS.SALESID])

    df_mst_chstock_joinBC = pd.merge(df_mst_chstock
                                , df_in_mst_chstock_ap2
                                , left_on=[NS.AP2ID]
                                , right_on=[CSM.SALESID], suffixes=['','_C'])
    
    df_mst_chstock_joinBC = df_mst_chstock_joinBC[[CSM.SALESID, CSM.ITEM, CSM.TOBE, CSAP2M.PRODUCTGROUP, CSAP2M.ASIS]]

    df_exp_dpguide_joinAB = pd.merge(df_exp_dpguide
            , df_mst_chstock_joinBC
            , how='left'
            , left_on=[DPG.SALESID, DPG.UDAITEM]
            , right_on=[CSM.SALESID, CSM.ITEM], suffixes=['', '_B'])

    df_exp_dpguide_target = df_exp_dpguide_joinAB[
        ( df_exp_dpguide_joinAB[I.PRODUCTGROUP] == df_exp_dpguide_joinAB[CSAP2M.PRODUCTGROUP + '_B'] )
    ].reset_index(drop=True)
    
    # SIGENDWEEK 생성
    df_exp_dpguide_target['SIGENDWEEK'] = df_exp_dpguide_target[CSM.TOBE].fillna(df_exp_dpguide_target[CSAP2M.ASIS]).astype(int)

    sysdate = pd.Timestamp.now(tz='Asia/Seoul')
    weekval = ( 1 if sysdate.day_of_week > 1 else 0 ) - 1
    # 'MON', 0,'TUE', 0,'WED', 1,'THU', 1,'FRI', 1,'SAT', 1,'SUN', 1

    df_exp_dpguide_target['SIGENDWEEK'] = sysdate + pd.to_timedelta(
            7 * ( df_exp_dpguide_target['SIGENDWEEK'] + weekval ), unit='D' )

    df_exp_dpguide_target['SIGENDWEEK'] = common.convert_dt_to_week(df_exp_dpguide_target['SIGENDWEEK'])

    df_exp_dpguide_target = df_exp_dpguide_target[
        df_exp_dpguide_target[DPG.SWEEK] > df_exp_dpguide_target['SIGENDWEEK']
    ][[DPG.SALESID, DPG.UDAITEM, DPG.SWEEK]].drop_duplicates()

    df_out_exp_dpguide = df_in_exp_dpguide[
        (~ pd.MultiIndex.from_frame(df_in_exp_dpguide[[DPG.SALESID, DPG.UDAITEM, DPG.SWEEK]]).isin(
            pd.MultiIndex.from_frame(df_exp_dpguide_target[[DPG.SALESID, DPG.UDAITEM, DPG.SWEEK]]))
        )
    ]

    return df_out_exp_dpguide

def do_channel_stock(df_Inbound:DF
                    , df_in_exp_dpguide:DF
                    , df_in_gui_saleshierarchy:DF
                    , df_in_vui_itemattb:DF
                    , df_in_Netting_Seller_Map:DF
                    , df_in_Netting_ES_Item_Site_ODS_W:DF
                    , df_in_Netting_Custom_Model_Map_ODS_W:DF
                    , df_in_mst_chstock:DF
                    , df_in_mst_chstock_ap2:DF
                    , df_in_netting_code_map:DF
                    , df_in_Netting_Available_Resource_ODS_W:DF
                    , df_exp_ap1_shortreason:DF
                    , df_short_reason_dp_guide:DF
                    , logger:LOGGER) -> DF:
    
    global V_ALIVE, V_ROLLING, V_CHSTOCK, V_AVAILQTY2, V_NUM, V_DPGUIDE

    logger.Note('Start MX Channel Stock', 20)

    #region logger.Step(0, 'MX Channel Stock Setting')
    logger.Step(0, 'MX Channel Stock Setting Start')
    # ND_.LIST_COLUMN에 ROWA 값이 있는경우 삭제
    if 'ROWA' in ND_.LIST_COLUMN:
        ND_.LIST_COLUMN.remove('ROWA')
        
    # df_Inbound 컬럼 초기화
    for col in ND_.LIST_COLUMN:
        if not col in df_Inbound.columns:
            df_Inbound[col] = ''

    logger.Step(0, 'MX Channel Stock Setting End')
    #endregion logger.Step(0, 'MX Channel Stock Setting')

    #region logger.Step(1, 'MX Channel Stock 초기설정')
    logger.Step(1, 'MX Channel Stock 초기설정 Start')

    df_Inbound = df_Inbound[ND_.LIST_COLUMN]
    df_Outbound = df_Inbound.copy(deep=True)
    
    # / 20205.09.19 곽영빈 프로 요청 - SIG 산출 시 제약구간 반영으로 Netting에서 해당 기능 사용 중지
    # df_in_exp_dpguide = df_EXP_DPGUIDE_DEL(df_in_exp_dpguide
    #                 , df_in_mst_chstock
    #                 , df_in_mst_chstock_ap2
    #                 , df_in_gui_saleshierarchy
    #                 , df_in_vui_itemattb )
    
    dpguide_ap2 = f_dpguide('3', df_in_exp_dpguide, df_in_gui_saleshierarchy, df_in_vui_itemattb, df_in_Netting_Custom_Model_Map_ODS_W, df_in_mst_chstock)
    dpguide_ap1 = f_dpguide('4', df_in_exp_dpguide, df_in_gui_saleshierarchy, df_in_vui_itemattb, df_in_Netting_Custom_Model_Map_ODS_W, df_in_mst_chstock)
    dpguide_account = f_dpguide('5', df_in_exp_dpguide, df_in_gui_saleshierarchy, df_in_vui_itemattb, df_in_Netting_Custom_Model_Map_ODS_W, df_in_mst_chstock)
    # dpguide_account6 = f_dpguide('6', df_in_exp_dpguide, df_in_gui_saleshierarchy, df_in_vui_itemattb, df_in_Netting_Custom_Model_Map_ODS_W, df_in_mst_chstock)
    # dpguide_account7 = f_dpguide('7', df_in_exp_dpguide, df_in_gui_saleshierarchy, df_in_vui_itemattb, df_in_Netting_Custom_Model_Map_ODS_W, df_in_mst_chstock)

    df_dic = df_Outbound[[ND_.SALESORDERID,ND_.SOLINENUM,]]

    # dict 생성
    dict_salesorder = {}
    for row in df_dic.to_numpy():
        soid = row[0]
        solinenum = int(row[1])
        if soid in dict_salesorder:
            dict_salesorder[soid].add(int(solinenum))
        else:
            dict_salesorder[soid] = set((int(solinenum),))

    logger.Step(1, 'MX Channel Stock 초기설정 End')
    #endregion logger.Step(1, 'MX Channel Stock 초기설정')

    #region logger.Step(2, 'SELL-IN GUIDE 수량 차감')
    logger.Step(2, 'SELL-IN GUIDE 수량 차감 Start')

    df_Outbound1 = df_Outbound.copy(deep=True)
    df_Outbound1['ROWA'] = df_Outbound1.index.astype(str)

    df_Outbound2 = df_Outbound1.copy(deep=True)


    df_Outbound2[ND_.QTYPROMISED] = df_Outbound2[ND_.QTYPROMISED].fillna(0).astype('int32')
    df_Outbound2[ND_.SOLINENUM] = df_Outbound2[ND_.SOLINENUM].fillna(0).astype('int32')
    df_Outbound2 = df_Outbound2.loc[df_Outbound2[ND_.QTYPROMISED]>0]


    # AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
    df_Outbound2 = df_Outbound2.loc[
            ~(df_Outbound2[ND_.SITEID]).isin(
                df_in_netting_code_map.loc[df_in_netting_code_map[CM.CODEMAPKEY].str.startswith('HC_WL_DP_CHSTOCK_EXCEPT')][CM.CODE1]
            )
        ]
    # AND NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID)
    df_MVES_ITEMSITE = df_in_Netting_ES_Item_Site_ODS_W
    df_Outbound2 = df_Outbound2.loc[
            ~(df_Outbound2[ND_.ITEMID] + df_Outbound2[ND_.SALESID] + df_Outbound2[ND_.SITEID]).isin(
                df_MVES_ITEMSITE[ESIS.ITEM]+df_MVES_ITEMSITE[ESIS.SALESID]+df_MVES_ITEMSITE[ESIS.SITEID]
            )
        ]

    df_Outbound2['PROMISEDDELDATE2'] = common.convert_dt_to_week(pd.to_datetime(df_Outbound2[ND_.PROMISEDDELDATE]))
    df_Outbound2['SALESORDERID1'] = df_Outbound2[ND_.SALESORDERID].str.split('::').str[0]
    df_Outbound2['SALESORDERID2'] = np.where(df_Outbound2['SALESORDERID1']=='COM_ORD',1, (np.where(df_Outbound2['SALESORDERID1']=='NEW_ORD',2,3)))

    df_Outbound2 = df_Outbound2.sort_values(
        ['SALESORDERID2', ND_.DEMANDPRIORITY,ND_.ITEMID,ND_.SALESORDERID,ND_.SALESID,ND_.QTYPROMISED, ND_.SOLINENUM,ND_.SITEID],
        ascending=[True,True,True,False,False,False,True,False]
    )

    df_Outbound2[ND_.AP1ID] = np.where(df_Outbound2[ND_.AP1ID].isna(),'',df_Outbound2[ND_.AP1ID])
    df_Outbound2[ND_.AP2ID] = np.where(df_Outbound2[ND_.AP2ID].isna(),'',df_Outbound2[ND_.AP2ID])

    additional_column = ['ROWA', 'PROMISEDDELDATE2']
    df_Outbound2 = df_Outbound2[ND_.LIST_COLUMN + additional_column]
    ND_.ROWA_IDX = df_Outbound2.columns.get_loc('ROWA')
    ND_.PROMISEDDELDATE2_IDX = df_Outbound2.columns.get_loc('PROMISEDDELDATE2')

    # dict_dp_account 생성
    dict_dp_account = OrderedDict()
    for row in df_Outbound2.values:
        key = row[ND_.SALESID_IDX]+'::'+str(row[ND_.PROMISEDDELDATE2_IDX])+'::'+row[ND_.ITEMID_IDX]
        value = row.copy()

        if key in dict_dp_account:
            dict_dp_account[key].append(value)
        else:
            dict_dp_account[key] = [value]


    dict_dpguide_account = OrderedDict()
    for row in dpguide_account.to_numpy():
        key = row[DPGUIDE.SALESID_IDX] + '::' + row[DPGUIDE.SWEEK_IDX] + '::' + row[DPGUIDE.UDAITEM_IDX]
        value = row[DPGUIDE.QTY_IDX]

        if key in dict_dpguide_account:
            dict_dpguide_account[key].append(value)
        else:
            dict_dpguide_account[key] = [value]


    count_using_column = df_Outbound2.shape[1]-1 # PROMISEDDELDATE2 사용 안 함
    list_Outbound_account_loop_insert = []
    list_Outbound_account_loop_update = []
    list_Outbound_account_loop_update2= []

    for key, values in dict_dpguide_account.items():
        list_values = list(values)
        V_DPGUIDE = list_values[0]
        dplist = dict_dp_account.get(key)
        if(dplist != None):
            # --! 가이드 수량 > 0 일때만 수행 !--
            for dp in dplist:

                if(V_DPGUIDE > 0):
                    # --! 가이드 수량 >= Demand값 이면 가이드수량만 차감 !--
                    if(V_DPGUIDE >= int(dp[ND_.QTYPROMISED_IDX])):
                        V_DPGUIDE -= int(dp[ND_.QTYPROMISED_IDX])
                    # --! 가이드 수량 < Demand값 !-- 
                    elif(V_DPGUIDE < int(dp[ND_.QTYPROMISED_IDX])):

                        V_SOLINENUM = dp[ND_.SOLINENUM_IDX]+1
                        # --! INSERT 중복시 예외 처리 !-- 시작
                        set_solinenum = dict_salesorder[dp[ND_.SALESORDERID_IDX]]
                        max_solinenum = max(set_solinenum)
                        
                        if V_SOLINENUM in set_solinenum:
                            V_SOLINENUM = max_solinenum + 1

                        set_solinenum.add(V_SOLINENUM)
                        dict_salesorder[dp[ND_.SALESORDERID_IDX]] = set_solinenum
                        # --! INSERT 중복시 예외 처리 !-- 끝
                        i_qtypromise = int(dp[ND_.QTYPROMISED_IDX]) - V_DPGUIDE
                        i_optioncode = int(np.nan_to_num(dp[ND_.OPTION_CODE_IDX], copy=False))+1

                        row = dp[:count_using_column].copy() # -->.copy()를 빼면 데이터 안맞음
                        row[ND_.ROWA_IDX] = 0
                        row[ND_.SOLINENUM_IDX] = V_SOLINENUM
                        row[ND_.QTYPROMISED_IDX] = i_qtypromise
                        row[ND_.OPTION_CODE_IDX] = i_optioncode
                        row[ND_.AP1ID_IDX] = ''
                        list_Outbound_account_loop_insert.append(row)
                        
                        # --! 가이드 수량값만큼은 일반 DP로 !--

                        list_Outbound_account_loop_update.append(np.array([dp[ND_.ROWA_IDX], V_DPGUIDE]))

                        # --! 모두 소진 했으므로 Guide 0 처리 !-- 
                        V_DPGUIDE = 0
                # --! DP GUIDE 수량이 0으로 들어올 경우 !--
                elif(V_DPGUIDE == 0):
                    # --! 가이드를 모두 소진하여 0 이면 나머지는 전부다 Channel Stock Short !--
                    i_optioncode = int(np.nan_to_num(dp[ND_.OPTION_CODE_IDX], copy=False))+1
                    list_Outbound_account_loop_update2.append(np.array([dp[ND_.ROWA_IDX], i_optioncode, 'C/Stock']))



    # dict_dp_ap1 생성
    dict_dp_ap1 = OrderedDict()
    for row in df_Outbound2.to_numpy():
        key = str(row[ND_.AP1ID_IDX])+'::'+str(row[ND_.PROMISEDDELDATE2_IDX])+'::'+row[ND_.ITEMID_IDX]
        value = row.copy()

        if key in dict_dp_ap1:
            dict_dp_ap1[key].append(value)
        else:
            dict_dp_ap1[key] = [value]

    list_Outbound_ap1_loop_insert = []
    list_Outbound_ap1_loop_update = []
    list_Outbound_ap1_loop_update2= []

    nparr_dpguide_ap1 = dpguide_ap1.to_numpy()
        # AP2ID_IDX           =  0
        # LEVELID_IDX         =  1
        # SALESID_IDX         =  2
        # UDAITEM_IDX         =  3
        # SWEEK_IDX           =  4
        # QTY_IDX             =  5

    for i_row in nparr_dpguide_ap1:
        V_DPGUIDE = i_row[DPGUIDE.QTY_IDX]
        sid = i_row[DPGUIDE.SALESID_IDX] + '::' + i_row[DPGUIDE.SWEEK_IDX] + '::' + i_row[DPGUIDE.UDAITEM_IDX]

        dplist = dict_dp_ap1.get(sid)
        if(dplist != None):
            # --! 가이드 수량 > 0 일때만 수행 !--
            for dp in dplist:

                if(V_DPGUIDE > 0):
                    # --! 가이드 수량 >= Demand값 이면 가이드수량만 차감 !--
                    if(V_DPGUIDE >= int(dp[ND_.QTYPROMISED_IDX])):
                        V_DPGUIDE -= int(dp[ND_.QTYPROMISED_IDX])
                    # --! 가이드 수량 < Demand값 !-- 
                    elif(V_DPGUIDE < int(dp[ND_.QTYPROMISED_IDX])):

                        V_SOLINENUM = dp[ND_.SOLINENUM_IDX]+1
                        
                        # --! INSERT 중복시 예외 처리 !-- 시작
                        set_solinenum = dict_salesorder[dp[ND_.SALESORDERID_IDX]]
                        max_solinenum = max(set_solinenum)
                        
                        if V_SOLINENUM in set_solinenum:
                            V_SOLINENUM = max_solinenum + 1

                        set_solinenum.add(V_SOLINENUM)
                        dict_salesorder[dp[ND_.SALESORDERID_IDX]] = set_solinenum
                        # --! INSERT 중복시 예외 처리 !-- 끝
                        
                        i_qtypromise = int(dp[ND_.QTYPROMISED_IDX]) - V_DPGUIDE
                        i_optioncode = int(np.nan_to_num(dp[ND_.OPTION_CODE_IDX], copy=False))+1
                        
                        row = dp[:count_using_column].copy() # -->.copy()를 빼면 데이터 안맞음
                        row[ND_.ROWA_IDX] = 0
                        row[ND_.SOLINENUM_IDX] = V_SOLINENUM
                        row[ND_.QTYPROMISED_IDX] = i_qtypromise
                        row[ND_.OPTION_CODE_IDX] = i_optioncode
                        row[ND_.AP1ID_IDX] = ''
                        list_Outbound_ap1_loop_insert.append(row)
                        
                        # --! 가이드 수량값만큼은 일반 DP로 !--
                        list_Outbound_ap1_loop_update.append(np.array([dp[ND_.ROWA_IDX], V_DPGUIDE]))

                        # --! 모두 소진 했으므로 Guide 0 처리 !-- 
                        V_DPGUIDE = 0
                # --! DP GUIDE 수량이 0으로 들어올 경우 !--
                elif(V_DPGUIDE == 0):
                    # --! 가이드를 모두 소진하여 0 이면 나머지는 전부다 Channel Stock Short !--
                    i_optioncode = int(np.nan_to_num(dp[ND_.OPTION_CODE_IDX], copy=False))+1

                    list_Outbound_ap1_loop_update2.append(np.array([dp[ND_.ROWA_IDX], int(i_optioncode), 'C/Stock']))

    # dict_dp_ap2 생성
    dict_dp_ap2 = OrderedDict()
    for row in df_Outbound2.to_numpy():
        key = str(row[ND_.AP2ID_IDX])+'::'+str(row[ND_.PROMISEDDELDATE2_IDX])+'::'+row[ND_.ITEMID_IDX]
        value = row.copy()

        if key in dict_dp_ap2:
            dict_dp_ap2[key].append(value)
        else:
            dict_dp_ap2[key] = [value]

    list_Outbound_ap2_loop_insert = []
    list_Outbound_ap2_loop_update = []
    list_Outbound_ap2_loop_update2= []

    nparr_dpguide_ap2 = dpguide_ap2.to_numpy()

    for i_row in nparr_dpguide_ap2:
        V_DPGUIDE = i_row[DPGUIDE.QTY_IDX]
        sid = i_row[DPGUIDE.SALESID_IDX] + '::' + i_row[DPGUIDE.SWEEK_IDX] + '::' + i_row[DPGUIDE.UDAITEM_IDX]
        dplist = dict_dp_ap2.get(sid)
        if(dplist != None):
            for dp in dplist:
                # --! 가이드 수량 > 0 일때만 수행 !--
                if(V_DPGUIDE > 0):
                    # --! 가이드 수량 >= Demand값 이면 가이드수량만 차감 !--
                    if(V_DPGUIDE >= int(dp[ND_.QTYPROMISED_IDX])):
                        V_DPGUIDE -= int(dp[ND_.QTYPROMISED_IDX])
                    # --! 가이드 수량 < Demand값 !-- 
                    elif(V_DPGUIDE < int(dp[ND_.QTYPROMISED_IDX])):

                        V_SOLINENUM = dp[ND_.SOLINENUM_IDX]+1
                        
                        # --! INSERT 중복시 예외 처리 !-- 시작
                        set_solinenum = dict_salesorder[dp[ND_.SALESORDERID_IDX]]
                        max_solinenum = max(set_solinenum)
                        
                        if V_SOLINENUM in set_solinenum:
                            V_SOLINENUM = max_solinenum + 1

                        set_solinenum.add(V_SOLINENUM)
                        dict_salesorder[dp[ND_.SALESORDERID_IDX]] = set_solinenum
                        # --! INSERT 중복시 예외 처리 !-- 끝
                        
                        i_qtypromise = (dp[ND_.QTYPROMISED_IDX]) - V_DPGUIDE
                        i_optioncode = int(np.nan_to_num(dp[ND_.OPTION_CODE_IDX], copy=False))+1
                        
                        row = dp[:count_using_column].copy() # -->.copy()를 빼면 데이터 안맞음
                        row[ND_.ROWA_IDX] = 0
                        row[ND_.SOLINENUM_IDX] = V_SOLINENUM
                        row[ND_.QTYPROMISED_IDX] = i_qtypromise
                        row[ND_.OPTION_CODE_IDX] = i_optioncode
                        row[ND_.AP1ID_IDX] = ''
                        list_Outbound_ap2_loop_insert.append(row)
                        
                        # --! 가이드 수량값만큼은 일반 DP로 !--
                        list_Outbound_ap2_loop_update.append(np.array([dp[ND_.ROWA_IDX], V_DPGUIDE]))

                        # --! 모두 소진 했으므로 Guide 0 처리 !-- 
                        V_DPGUIDE = 0
                # --! DP GUIDE 수량이 0으로 들어올 경우 !--
                elif(V_DPGUIDE == 0):
                    # --! 가이드를 모두 소진하여 0 이면 나머지는 전부다 Channel Stock Short !--
                    i_optioncode = int(np.nan_to_num(dp[ND_.OPTION_CODE_IDX], copy=False))+1

                    list_Outbound_ap2_loop_update2.append(np.array([dp[ND_.ROWA_IDX], int(i_optioncode), 'C/Stock']))
                    

    update_columns = ['ROWA','QTYPROMISED_NEW']
    update2_columns = ['ROWA','OPTION_CODE_NEW','UPBY_NEW']
    df_outbound_account_insert  = pd.DataFrame(data=list_Outbound_account_loop_insert, columns=df_Outbound1.columns, dtype=object)
    df_outbound_account_update  = pd.DataFrame(data=list_Outbound_account_loop_update, columns=update_columns, dtype=object)
    df_outbound_account_update2 = pd.DataFrame(data=list_Outbound_account_loop_update2, columns=update2_columns, dtype=object)

    df_outbound_ap1_insert      = pd.DataFrame(data=list_Outbound_ap1_loop_insert, columns=df_Outbound1.columns, dtype=object)
    df_outbound_ap1_update      = pd.DataFrame(data=list_Outbound_ap1_loop_update, columns=update_columns, dtype=object)
    df_outbound_ap1_update2     = pd.DataFrame(data=list_Outbound_ap1_loop_update2, columns=update2_columns, dtype=object)

    df_outbound_ap2_insert      = pd.DataFrame(data=list_Outbound_ap2_loop_insert, columns=df_Outbound1.columns, dtype=object)
    df_outbound_ap2_update      = pd.DataFrame(data=list_Outbound_ap2_loop_update, columns=update_columns, dtype=object)
    df_outbound_ap2_update2     = pd.DataFrame(data=list_Outbound_ap2_loop_update2, columns=update2_columns, dtype=object)

    df_outbound_account_update  = df_outbound_account_update.astype({'ROWA':'str', 'QTYPROMISED_NEW':'int32'})
    df_outbound_account_update2 = df_outbound_account_update2.astype({'ROWA':'str', 'OPTION_CODE_NEW':'str', 'UPBY_NEW':'str'})
    df_outbound_ap1_update      = df_outbound_ap1_update.astype({'ROWA':'str', 'QTYPROMISED_NEW':'int32'})
    df_outbound_ap1_update2     = df_outbound_ap1_update2.astype({'ROWA':'str', 'OPTION_CODE_NEW':'str', 'UPBY_NEW':'str'})
    df_outbound_ap2_update      = df_outbound_ap2_update.astype({'ROWA':'str', 'QTYPROMISED_NEW':'int32'})
    df_outbound_ap2_update2     = df_outbound_ap2_update2.astype({'ROWA':'str', 'OPTION_CODE_NEW':'str', 'UPBY_NEW':'str'})

    # account merge
    df_outbound_mege = df_Outbound1.merge(df_outbound_account_update, on=['ROWA'], how='left')
    df_outbound_mege.loc[~df_outbound_mege['QTYPROMISED_NEW'].isna(),ND_.QTYPROMISED] = df_outbound_mege['QTYPROMISED_NEW']
    df_outbound_mege = df_outbound_mege.drop(columns=['QTYPROMISED_NEW'])

    df_outbound_mege = df_outbound_mege.merge(df_outbound_account_update2, on=['ROWA'], how='left')
    df_outbound_mege.loc[~df_outbound_mege['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_outbound_mege['OPTION_CODE_NEW']
    df_outbound_mege = df_outbound_mege.drop(columns=['OPTION_CODE_NEW','UPBY_NEW'])

    df_outbound_mege = pd.concat([df_outbound_mege, df_outbound_account_insert], axis=0, ignore_index=True)

    # ap1 merge
    df_outbound_mege2 = df_outbound_mege.merge(df_outbound_ap1_update, on=['ROWA'], how='left')
    df_outbound_mege2.loc[~df_outbound_mege2['QTYPROMISED_NEW'].isna(),ND_.QTYPROMISED] = df_outbound_mege2['QTYPROMISED_NEW']
    df_outbound_mege2 = df_outbound_mege2.drop(columns=['QTYPROMISED_NEW'])

    df_outbound_mege2 = df_outbound_mege2.merge(df_outbound_ap1_update2, on=['ROWA'], how='left')
    df_outbound_mege2.loc[~df_outbound_mege2['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_outbound_mege2['OPTION_CODE_NEW']
    df_outbound_mege2 = df_outbound_mege2.drop(columns=['OPTION_CODE_NEW','UPBY_NEW'])

    df_outbound_mege2 = pd.concat([df_outbound_mege2, df_outbound_ap1_insert], axis=0, ignore_index=True)

    # ap2 merge
    df_outbound_mege3 = df_outbound_mege2.merge(df_outbound_ap2_update, on=['ROWA'], how='left')
    df_outbound_mege3.loc[~df_outbound_mege3['QTYPROMISED_NEW'].isna(),ND_.QTYPROMISED] = df_outbound_mege3['QTYPROMISED_NEW']
    df_outbound_mege3 = df_outbound_mege3.drop(columns=['QTYPROMISED_NEW'])

    df_outbound_mege3 = df_outbound_mege3.merge(df_outbound_ap2_update2, on=['ROWA'], how='left')
    df_outbound_mege3.loc[~df_outbound_mege3['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_outbound_mege3['OPTION_CODE_NEW']
    df_outbound_mege3 = df_outbound_mege3.drop(columns=['OPTION_CODE_NEW','UPBY_NEW'])

    df_outbound_mege3 = pd.concat([df_outbound_mege3, df_outbound_ap2_insert], axis=0, ignore_index=True)

    df_outbound_mege3 = df_outbound_mege3.drop(columns=['ROWA'])


    logger.Step(2, 'SELL-IN GUIDE 수량 차감 End')
    #endregion logger.Step(2, 'SELL-IN GUIDE 수량 차감')

    #region logger.Step(3, '재고(가용량) 감안하여 CH_STOCK SHORT 수량 살림')
    logger.Step(3, '재고(가용량) 감안하여 CH_STOCK SHORT 수량 살림 Start')

    df_demand00 = df_outbound_mege3.copy(deep=True)
    df_demand00['ROWA'] = df_outbound_mege3.index.astype(str)

    # AND NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
    df_demand01 = df_demand00.loc[
            ~(df_demand00[ND_.ITEMID] + df_demand00[ND_.SITEID]).isin(
                df_in_Netting_Seller_Map[MXISM.ITEM]+df_in_Netting_Seller_Map[MXISM.SITEID]
            )
        ]

    # AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
    df_demand01 = df_demand01.loc[
            ~(df_demand01[ND_.SITEID]).isin(
                df_in_netting_code_map.loc[
                    df_in_netting_code_map[CM.CODEMAPKEY].str.startswith('HC_WL_DP_CHSTOCK_EXCEPT')
                ][CM.CODE1]
            )
        ]
    '10/13 곽영빈 프로 요청으로, estore fcst도 가용량 소요량 계산에 포함'
    # # AND NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID)
    # df_MVES_ITEMSITE = df_in_Netting_ES_Item_Site_ODS_W.copy(deep=True)
    # df_demand01 = df_demand01.loc[
    #         ~(df_demand01[ND_.ITEMID] + df_demand01[ND_.SALESID] + df_demand01[ND_.SITEID]).isin(
    #             df_MVES_ITEMSITE[ESIS.ITEM]+df_MVES_ITEMSITE[ESIS.SALESID]+df_MVES_ITEMSITE[ESIS.SITEID]
    #         )
    #     ]
    
    # AND ITEM IN (select UDAITEM from EXP_DPGUIDE)
    df_demand01 = df_demand01.loc[df_demand01[ND_.ITEMID].isin(df_in_exp_dpguide[DPG.UDAITEM])]
    df_demand01 = df_demand01[ND_.LIST_COLUMN + ['ROWA']]

    df_demand01['ROWA']  = df_demand01['ROWA'].astype('str')
    df_demand01[ND_.PROMISEDDELDATE]  = df_demand01[ND_.PROMISEDDELDATE].astype('datetime64[ns]')

    mst_inventory_fne = df_in_Netting_Available_Resource_ODS_W.copy(deep=True)

    # AND NOT EXISTS (SELECT 'X' FROM v_mta_sellermap WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
    mst_inventory_fne = mst_inventory_fne.loc[
            ~(mst_inventory_fne[AR_.ITEM] + mst_inventory_fne[AR_.SITEID]).isin(
                df_in_Netting_Seller_Map[MXISM.ITEM]+df_in_Netting_Seller_Map[MXISM.SITEID]
            )
        ]

    # AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
    mst_inventory_fne = mst_inventory_fne.loc[
            ~(mst_inventory_fne[AR_.SITEID]).isin(
                df_in_netting_code_map.loc[
                    df_in_netting_code_map[CM.CODEMAPKEY].str.startswith('HC_WL_DP_CHSTOCK_EXCEPT')
                ][CM.CODE1]
            )
        ]

    df_demand02 = mst_inventory_fne.copy(deep=True)
    df_demand02[ND_.PLANID] = V_PLANID
    df_demand02[ND_.PROMISEDDELDATE] = df_demand02[AR_.WEEK].apply(get_iso_week_end)
    df_demand02.rename(columns={
        AR_.ITEM: ND_.ITEMID,
        AR_.SITEID: ND_.SITEID,
    }, inplace=True)

    overall_column = ND_.LIST_COLUMN + ['ROWA']
    set_overall_column = set(overall_column)
    set_diff_column = set((
        ND_.PLANID, ND_.ITEMID, ND_.QTYPROMISED, ND_.PROMISEDDELDATE, ND_.SITEID, ND_.WEEKRANK
    ))
    set_empty_string_column = set_overall_column - set_diff_column
    list_empty_string_column = list(set_empty_string_column)

    df_demand02[list_empty_string_column] = ''
    df_demand02[ND_.QTYPROMISED] = '0'
    df_demand02[ND_.WEEKRANK] = df_demand02[AR_.WEEK].str[-2:]
    df_demand02['ROWA'] = np.NaN
    df_demand02 = df_demand02[overall_column]

    df_demand02['ROWA'] = df_demand02['ROWA'].astype('str')
    df_demand02 = df_demand02.astype(df_demand01.dtypes)

    df_demand = pd.concat([df_demand01, df_demand02], axis=0, ignore_index=True)

    # AND EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
    df_demand_sell01 = df_demand00.loc[
            (df_demand00[ND_.ITEMID] + df_demand00[ND_.SITEID]).isin(
                df_in_Netting_Seller_Map[MXISM.ITEM]+df_in_Netting_Seller_Map[MXISM.SITEID]
            )
        ]

    # AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
    df_demand_sell01 = df_demand_sell01.loc[
            ~(df_demand_sell01[ND_.SITEID]).isin(
                df_in_netting_code_map.loc[
                    df_in_netting_code_map[CM.CODEMAPKEY].str.startswith('HC_WL_DP_CHSTOCK_EXCEPT')
                ][CM.CODE1]
            )
        ]
    '10/13 곽영빈 프로 요청으로, estore fcst도 가용량 소요량 계산에 포함'
    # # AND NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID)
    # df_MVES_ITEMSITE = df_in_Netting_ES_Item_Site_ODS_W.copy(deep=True)
    # df_demand_sell01 = df_demand_sell01.loc[
    #         ~(df_demand_sell01[ND_.ITEMID] + df_demand_sell01[ND_.SALESID] + df_demand_sell01[ND_.SITEID]).isin(
    #             df_MVES_ITEMSITE[ESIS.ITEM]+df_MVES_ITEMSITE[ESIS.SALESID]+df_MVES_ITEMSITE[ESIS.SITEID]
    #         )
    #     ]

    # AND ITEM IN (select UDAITEM from EXP_DPGUIDE)    
    df_demand_sell01 = df_demand_sell01.loc[df_demand_sell01[ND_.ITEMID].isin(df_in_exp_dpguide[DPG.UDAITEM])]
    df_demand_sell01 = df_demand_sell01[overall_column]

    df_demand_sell01['ROWA']  = df_demand_sell01['ROWA'].astype('str')
    df_demand_sell01[ND_.PROMISEDDELDATE]  = df_demand_sell01[ND_.PROMISEDDELDATE].astype('datetime64[ns]')


    mst_inventory_fne_sell = df_in_Netting_Available_Resource_ODS_W.copy(deep=True)

    # AND EXISTS (SELECT 'X' FROM v_mta_sellermap WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
    mst_inventory_fne_sell = mst_inventory_fne_sell.loc[
            (mst_inventory_fne_sell[AR_.ITEM] + mst_inventory_fne_sell[AR_.SITEID]).isin(
                df_in_Netting_Seller_Map[MXISM.ITEM]+df_in_Netting_Seller_Map[MXISM.SITEID]
            )
        ]

    # AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
    mst_inventory_fne_sell = mst_inventory_fne_sell.loc[
            ~(mst_inventory_fne_sell[AR_.SITEID]).isin(
                df_in_netting_code_map.loc[
                    df_in_netting_code_map[CM.CODEMAPKEY].str.startswith('HC_WL_DP_CHSTOCK_EXCEPT')
                ][CM.CODE1]
            )
        ]


    df_demand_sell02 = mst_inventory_fne_sell.copy(deep=True)
    df_demand_sell02[ND_.PLANID] = V_PLANID
    df_demand_sell02[ND_.PROMISEDDELDATE] = df_demand_sell02[AR_.WEEK].apply(get_iso_week_end)
    df_demand_sell02.rename(columns={
        AR_.ITEM: ND_.ITEMID,
        AR_.SITEID: ND_.SITEID,
    }, inplace=True)
    df_demand_sell02[list_empty_string_column] = ''
    df_demand_sell02[ND_.QTYPROMISED] = '0'
    df_demand_sell02[ND_.WEEKRANK] = df_demand_sell02[AR_.WEEK].str[-2:]
    df_demand_sell02['ROWA'] = np.NaN
    df_demand_sell02 = df_demand_sell02[overall_column]

    df_demand_sell02['ROWA'] = df_demand_sell02['ROWA'].astype('str')
    df_demand_sell02[ND_.PROMISEDDELDATE]  = df_demand_sell02[ND_.PROMISEDDELDATE].astype('datetime64[ns]')
    df_demand_sell02.astype(df_demand01.dtypes)

    df_demand_sell = pd.concat([df_demand_sell01, df_demand_sell02], axis=0, ignore_index=True)

    mst_inventory_fne_inner = df_in_Netting_Available_Resource_ODS_W.copy(deep=True)

    # demand부분
    df_demand['PROMISEDDELDATE2'] = common.convert_dt_to_week(df_demand[ND_.PROMISEDDELDATE])
    #AND    NOT EXISTS (SELECT ''X'' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = ''Y'')
    df_demand = df_demand.loc[
        ~(df_demand[ND_.ITEMID]+'Y').isin(
            df_in_Netting_Custom_Model_Map_ODS_W[CMM.CUSTOMITEM] + df_in_Netting_Custom_Model_Map_ODS_W[CMM.ISVALID]
        )
    ]

    df_demand_aa = pd.merge(df_demand, mst_inventory_fne_inner, how='left',
                                    left_on =[ND_.SITEID, ND_.ITEMID, 'PROMISEDDELDATE2'],
                                    right_on=[AR_.SITEID, AR_.ITEM, AR_.WEEK,])

    df_demand_aa['AVAILQTY'] = df_demand_aa[AR_.QTY].fillna(0)

    df_demand_aa = df_demand_aa[ND_.LIST_COLUMN + ['ROWA', 'AVAILQTY']]

    df_demand_aa['DEMANDPRIORITY2'] = df_demand_aa[ND_.DEMANDPRIORITY]
    df_demand_aa.loc[df_demand_aa['DEMANDPRIORITY2']=='','DEMANDPRIORITY2'] = '0'
    df_demand_aa['OPTION_CODE2'] = df_demand_aa[ND_.OPTION_CODE]
    df_demand_aa.loc[df_demand_aa['OPTION_CODE2'].isna() ,'OPTION_CODE2'] = '0'
    df_demand_aa.loc[df_demand_aa['OPTION_CODE2']=='','OPTION_CODE2'] = '0'
    df_demand_aa.loc[df_demand_aa['AVAILQTY']=='','AVAILQTY'] = '0'

    df_demand_aa['DEMANDPRIORITY2'] = df_demand_aa['DEMANDPRIORITY2'].astype('int32')
    df_demand_aa[ND_.QTYPROMISED] = df_demand_aa[ND_.QTYPROMISED].astype('int32')
    df_demand_aa['OPTION_CODE2'] = df_demand_aa['OPTION_CODE2'].astype('int32')
    df_demand_aa['OPTION_CODE2'] = df_demand_aa['OPTION_CODE2'].mod(2)
    df_demand_aa['AVAILQTY'] = df_demand_aa['AVAILQTY'].astype('int32')

    # ROW_NUMBER() OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.PROMISEDDELDATE ORDER BY NVL(A.DEMANDPRIORITY,0) ASC ) PART_CNT,
    # 정렬을 위해 임시로 날자타입으로 변경
    df_demand_aa['PROMISEDDELDATE2'] = pd.to_datetime(df_demand_aa[ND_.PROMISEDDELDATE])

    df_demand_aa['PART_CNT'] = (df_demand_aa.sort_values(
        by=['DEMANDPRIORITY2','OPTION_CODE2',ND_.QTYPROMISED,ND_.SALESID], ascending=[True,True,True,True]
    ).groupby([ND_.PLANID, ND_.ITEMID, ND_.SITEID, ND_.PROMISEDDELDATE,]).cumcount() + 1)

    # COUNT(*) OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.PROMISEDDELDATE) PART2_CNT,
    df_demand_aa['PART2_CNT'] = df_demand_aa.groupby(
        by=[ND_.PLANID,ND_.ITEMID,ND_.SITEID,ND_.PROMISEDDELDATE,]
    )[ND_.PLANID].transform('count')
    # ROW_NUMBER() OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID ORDER BY A.PROMISEDDELDATE, NVL(A.DEMANDPRIORITY,0) ASC, MOD(NVL(A.OPTION_CODE,0), 2) ASC, a.qtypromised, a.siteid, a.salesid ) ALL_CNT,

    df_demand_aa['ALL_CNT'] = (df_demand_aa.sort_values(
        by=['PROMISEDDELDATE2','DEMANDPRIORITY2','OPTION_CODE2',ND_.QTYPROMISED,ND_.SITEID,ND_.SALESID,], ascending=[True,True,True,True,True,True]
    ).groupby([ND_.PLANID, ND_.ITEMID, ND_.SITEID,]).cumcount() + 1)
    # NVL(SUM(A.QTYPROMISED) OVER (PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.PROMISEDDELDATE ORDER BY NVL(A.DEMANDPRIORITY,0) ASC ROWS UNBOUNDED PRECEDING ) ,0) SUM_DEMAND 
    df_demand_aa['SUM_DEMAND'] = df_demand_aa.sort_values(
        by=['DEMANDPRIORITY2','OPTION_CODE2',ND_.QTYPROMISED,ND_.SALESID], ascending=[True,True,True,True]
    ).groupby([ND_.PLANID, ND_.ITEMID, ND_.SITEID, ND_.PROMISEDDELDATE,])[ND_.QTYPROMISED].cumsum()  

    df_demand_aa = df_demand_aa.sort_values(by=[ND_.PLANID, ND_.ITEMID, ND_.SITEID, ND_.PROMISEDDELDATE, 'ALL_CNT'])
    df_demand_aa['REMAINAVAIL'] = np.where(df_demand_aa['AVAILQTY'] - df_demand_aa['SUM_DEMAND'] < 0,0,df_demand_aa['AVAILQTY'] - df_demand_aa['SUM_DEMAND'])
    df_demand_aa = df_demand_aa.drop(columns=['PROMISEDDELDATE2'])

    # demand_sell 부분
    df_demand_sell['PROMISEDDELDATE2'] = common.convert_dt_to_week(df_demand_sell[ND_.PROMISEDDELDATE])

    #AND    NOT EXISTS (SELECT ''X'' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = ''Y'')
    df_demand_sell = df_demand_sell.loc[
        ~(df_demand_sell[ND_.ITEMID]+'Y').isin(
            df_in_Netting_Custom_Model_Map_ODS_W[CMM.CUSTOMITEM] + df_in_Netting_Custom_Model_Map_ODS_W[CMM.ISVALID]
        )
    ]

    df_demand_sell_aa = pd.merge(df_demand_sell, mst_inventory_fne_inner, how='left',
                                    left_on =[ND_.SITEID, ND_.ITEMID, ND_.AP2ID, 'PROMISEDDELDATE2'],
                                    right_on=[AR_.SITEID, AR_.ITEM, AR_.SALESID, AR_.WEEK,])

    df_demand_sell_aa['AVAILQTY'] = df_demand_sell_aa[AR_.QTY].fillna(0)

    df_demand_sell_aa = df_demand_sell_aa[ND_.LIST_COLUMN + ['ROWA', 'AVAILQTY']]

    df_demand_sell_aa['DEMANDPRIORITY2'] = df_demand_sell_aa[ND_.DEMANDPRIORITY]
    df_demand_sell_aa.loc[df_demand_sell_aa['DEMANDPRIORITY2']=='','DEMANDPRIORITY2'] = '0'
    df_demand_sell_aa['OPTION_CODE2'] = df_demand_sell_aa[ND_.OPTION_CODE]
    df_demand_sell_aa.loc[df_demand_sell_aa['OPTION_CODE2'].isna() ,'OPTION_CODE2'] = '0'
    df_demand_sell_aa.loc[df_demand_sell_aa['OPTION_CODE2']=='','OPTION_CODE2'] = '0'
    df_demand_sell_aa.loc[df_demand_sell_aa['AVAILQTY']=='','AVAILQTY'] = '0'

    df_demand_sell_aa['DEMANDPRIORITY2'] = df_demand_sell_aa['DEMANDPRIORITY2'].astype('int32')
    df_demand_sell_aa[ND_.QTYPROMISED] = df_demand_sell_aa[ND_.QTYPROMISED].astype('int32')
    df_demand_sell_aa['OPTION_CODE2'] = df_demand_sell_aa['OPTION_CODE2'].astype('int32')
    df_demand_sell_aa['OPTION_CODE2'] = df_demand_sell_aa['OPTION_CODE2'].mod(2)
    df_demand_sell_aa['AVAILQTY'] = df_demand_sell_aa['AVAILQTY'].astype('int32')

    # ROW_NUMBER() OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.AP2ID, A.PROMISEDDELDATE ORDER BY NVL(A.DEMANDPRIORITY,0) ASC ) PART_CNT,

    # 정렬을 위해 임시로 날자타입으로 변경
    df_demand_sell_aa['PROMISEDDELDATE2'] = pd.to_datetime(df_demand_sell_aa[ND_.PROMISEDDELDATE])

    # 에러가 있는것 같아서 임시 변경
    # df_demand_sell_aa['PART_CNT'] = (df_demand_sell_aa.sort_values(by=['DEMANDPRIORITY2'], ascending=[True]).groupby(['PLANID', 'ITEM', 'SITEID', 'AP2ID', 'PROMISEDDELDATE']).cumcount() + 1)
    df_demand_sell_aa['PART_CNT'] = (df_demand_sell_aa.sort_values(
        by=['DEMANDPRIORITY2','OPTION_CODE2',ND_.QTYPROMISED,ND_.SALESID], ascending=[True,True,True,True]
    ).groupby([ND_.PLANID, ND_.ITEMID, ND_.SITEID, ND_.AP2ID, ND_.PROMISEDDELDATE,]).cumcount() + 1)


    # COUNT(*) OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.AP2ID, A.PROMISEDDELDATE) PART2_CNT,
    df_demand_sell_aa['PART2_CNT'] = df_demand_sell_aa.groupby(
        by=[ND_.PLANID, ND_.ITEMID, ND_.SITEID, ND_.AP2ID,ND_.PROMISEDDELDATE,]
    )[ND_.PLANID].transform('count')
    # ROW_NUMBER() OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.AP2ID ORDER BY A.PROMISEDDELDATE, NVL(A.DEMANDPRIORITY,0) ASC, MOD(NVL(A.OPTION_CODE,0), 2) ASC, a.qtypromised, a.siteid, a.salesid ) ALL_CNT,
    df_demand_sell_aa['ALL_CNT'] = (df_demand_sell_aa.sort_values(
        by=['PROMISEDDELDATE2','DEMANDPRIORITY2','OPTION_CODE2',ND_.QTYPROMISED,ND_.SITEID,ND_.SALESID],
        ascending=[True,True,True,True,True,True]
    ).groupby([ND_.PLANID, ND_.ITEMID, ND_.SITEID, ND_.AP2ID,]).cumcount() + 1)
    # NVL(SUM(A.QTYPROMISED) OVER (PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.PROMISEDDELDATE ORDER BY NVL(A.DEMANDPRIORITY,0) ASC ROWS UNBOUNDED PRECEDING ) ,0) SUM_DEMAND 
    # 에러가 있는것 같아서 임시 변경
    df_demand_sell_aa['SUM_DEMAND'] = df_demand_sell_aa.sort_values(
        by=['DEMANDPRIORITY2','OPTION_CODE2',ND_.QTYPROMISED,ND_.SALESID], ascending=[True,True,True,True]
    ).groupby([ND_.PLANID, ND_.ITEMID, ND_.SITEID, ND_.AP2ID, ND_.PROMISEDDELDATE,])[ND_.QTYPROMISED].cumsum()

    df_demand_sell_aa = df_demand_sell_aa.sort_values(
        by=[ND_.PLANID, ND_.ITEMID, ND_.SITEID, ND_.AP2ID, ND_.PROMISEDDELDATE, 'ALL_CNT']
    )
    df_demand_sell_aa['REMAINAVAIL'] = np.where((df_demand_sell_aa['AVAILQTY'] - df_demand_sell_aa['SUM_DEMAND'] < 0),0,(df_demand_sell_aa['AVAILQTY'] - df_demand_sell_aa['SUM_DEMAND']))
    df_demand_sell_aa = df_demand_sell_aa.drop(columns=['PROMISEDDELDATE2'])

    df_S = pd.concat([df_demand_aa, df_demand_sell_aa], axis=0, ignore_index=True)
    df_S = df_S[ND_.LIST_COLUMN + [
        'ROWA', 'AVAILQTY','PART_CNT', 'PART2_CNT', 'ALL_CNT', 'SUM_DEMAND', 'REMAINAVAIL'
    ]]

    ND_.ROWA_IDX = df_S.columns.get_loc('ROWA')
    ND_.AVAILQTY_IDX = df_S.columns.get_loc('AVAILQTY')
    ND_.PART_CNT_IDX = df_S.columns.get_loc('PART_CNT')
    ND_.PART2_CNT_IDX = df_S.columns.get_loc('PART2_CNT')
    ND_.ALL_CNT_IDX = df_S.columns.get_loc('ALL_CNT')
    ND_.SUM_DEMAND_IDX = df_S.columns.get_loc('SUM_DEMAND')
    ND_.REMAINAVAIL_IDX = df_S.columns.get_loc('REMAINAVAIL')

    count_using_column = len(ND_.LIST_COLUMN) + 1 # ROWA

    list_Outbound_loop_update_s = np.empty((0,3), dtype=object)
    list_Outbound_loop_update2_s = np.empty((0,5), dtype=object)
    list_Outbound_loop_insert_s = np.empty((0,count_using_column), dtype=object) 

    # 타입변환
    df_S[ND_.OPTION_CODE] = np.where(df_S[ND_.OPTION_CODE].isna(),0,np.where(df_S[ND_.OPTION_CODE]=='','0',df_S[ND_.OPTION_CODE]))
    df_S[ND_.OPTION_CODE] = df_S[ND_.OPTION_CODE].astype('int32')

    nparr_s = df_S.to_numpy()

    V_REMAINQTY = 0
    V_AVAILQTY = 0

    for s_row in nparr_s:

        # --! 새로운 모델/Site 인 경우 잔량 0처리 !--
        if(s_row[ND_.ALL_CNT_IDX] == 1):
            V_REMAINQTY = 0

        # --! 주차가 바뀌는 경우 앞에 남은 수량은 뒤로 Move !--
        if(s_row[ND_.PART_CNT_IDX] == 1):
            V_AVAILQTY = s_row[ND_.AVAILQTY_IDX] + V_REMAINQTY
        
        # --! 가용량이 Demand 수량 합보다 큰 경우인데 C/Stock 일 경우 전량 살림 !--
        if((V_AVAILQTY >= s_row[ND_.SUM_DEMAND_IDX]) & (s_row[ND_.QTYPROMISED_IDX] > 0) & ((s_row[ND_.OPTION_CODE_IDX] % 2) == 1)) :
            list_Outbound_loop_update_s = np.append(list_Outbound_loop_update_s, np.array([[s_row[ND_.ROWA_IDX], s_row[ND_.OPTION_CODE_IDX]-1, 'C/Stock_AVAIL']]), axis=0)
        else:
            # --! 가용량이 누적합보다는 작으나 일부 물량을 살릴 수 있는 경우 !--  
            if (((V_AVAILQTY - (s_row[ND_.SUM_DEMAND_IDX] - s_row[ND_.QTYPROMISED_IDX])) > 0) & ((s_row[ND_.OPTION_CODE_IDX] % 2) == 1)):

                # --! 부분 수량 보전 !--
                # -- 현재 ROW에서 쓸 수 있는 남은 가용량 = V_AVAILQTY - (ND_.SUM_DEMAND - ND_.QTYPROMISED)
                list_Outbound_loop_update2_s = np.append(list_Outbound_loop_update2_s, 
                    np.array([[s_row[ND_.ROWA_IDX], s_row[ND_.OPTION_CODE_IDX]-1, V_AVAILQTY - (s_row[ND_.SUM_DEMAND_IDX] - s_row[ND_.QTYPROMISED_IDX]), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'C/Stock_AVAIL']]), axis=0)

                V_SOLINENUM = int(s_row[ND_.SOLINENUM_IDX])+1
                
                # --! INSERT 중복시 예외 처리 !-- 시작
                set_solinenum = dict_salesorder[s_row[ND_.SALESORDERID_IDX]]
                max_solinenum = max(set_solinenum)
                        
                if V_SOLINENUM in set_solinenum:
                    V_SOLINENUM = max_solinenum + 1

                set_solinenum.add(V_SOLINENUM)
                dict_salesorder[s_row[ND_.SALESORDERID_IDX]] = set_solinenum
                # --! INSERT 중복시 예외 처리 !-- 끝
                
                i_qtypromise = s_row[ND_.SUM_DEMAND_IDX] - V_AVAILQTY

                row = s_row[:count_using_column].copy()
                row[ND_.ROWA_IDX] = 0
                row[ND_.SOLINENUM_IDX] = V_SOLINENUM
                row[ND_.QTYPROMISED_IDX] = i_qtypromise
                list_Outbound_loop_insert_s = np.append(list_Outbound_loop_insert_s, [row], axis=0)
        
        # --! 주차가 끝나는 시점에서 사용되고 남은 가용량 차주로 Move !--
        if(s_row[ND_.PART_CNT_IDX] == s_row[ND_.PART2_CNT_IDX]):
            if((V_AVAILQTY - s_row[ND_.SUM_DEMAND_IDX]) < 0):
                V_REMAINQTY = 0
            else:
                V_REMAINQTY = V_AVAILQTY - s_row[ND_.SUM_DEMAND_IDX]

    #위에 부분 list_Outbound_loop_insert_s, list_Outbound_loop_update_s, list_Outbound_loop_update2_s 소스 df_demand00에 머지
    df_outbound_loop_insert_s  = pd.DataFrame(list_Outbound_loop_insert_s)
    df_outbound_loop_update_s  = pd.DataFrame(list_Outbound_loop_update_s)
    df_outbound_loop_update2_s = pd.DataFrame(list_Outbound_loop_update2_s)


    df_outbound_loop_insert_s.columns = df_Outbound1.columns
    df_outbound_loop_update_s.columns = ['ROWA','OPTION_CODE_NEW','UPBY_NEW']
    df_outbound_loop_update2_s.columns = ['ROWA','OPTION_CODE_NEW','QTYPROMISED_NEW','UPDTTM_NEW','UPBY_NEW']

    df_outbound_loop_update_s = df_outbound_loop_update_s.astype({'ROWA':str, 'OPTION_CODE_NEW':'str', 'UPBY_NEW':'str'})
    df_outbound_loop_update2_s = df_outbound_loop_update2_s.astype({'ROWA':str, 'OPTION_CODE_NEW':'str', 'QTYPROMISED_NEW':'int32','UPDTTM_NEW':'str','UPBY_NEW':'str'})

    df_demand001 = df_demand00.merge(df_outbound_loop_update_s, on=['ROWA'], how='left')
    df_demand001.loc[~df_demand001['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_demand001['OPTION_CODE_NEW']
    df_demand001 = df_demand001.drop(columns=['OPTION_CODE_NEW','UPBY_NEW'])

    df_demand001 = df_demand001.merge(df_outbound_loop_update2_s, on=['ROWA'], how='left')
    df_demand001.loc[~df_demand001['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_demand001['OPTION_CODE_NEW']
    df_demand001.loc[~df_demand001['QTYPROMISED_NEW'].isna(),ND_.QTYPROMISED] = df_demand001['QTYPROMISED_NEW']
    df_demand001 = df_demand001.drop(columns=['OPTION_CODE_NEW','QTYPROMISED_NEW','UPDTTM_NEW','UPBY_NEW'])

    df_demand001 = pd.concat([df_demand001, df_outbound_loop_insert_s], axis=0, ignore_index=True)
    #concat(insert) 후에 ROWA 다시 설정 2025-06-17
    df_demand001['ROWA'] = df_demand001.index.astype(str)
    logger.Step(3, '재고(가용량) 감안하여 CH_STOCK SHORT 수량 살림 End')
    #endregion logger.Step(3, '재고(가용량) 감안하여 CH_STOCK SHORT 수량 살림')

    #region logger.Step(4, '3,4 레벨 버퍼수량 전량 C/Stock Short')
    logger.Step(4, '3,4 레벨 버퍼수량 전량 C/Stock Short Start')

    # --! 3,4 레벨 버퍼수량 전량 C/Stock Short !--

    df_demand01 = df_demand001.copy(deep=True)

    gui_saleshierarchy = df_in_gui_saleshierarchy.loc[df_in_gui_saleshierarchy[NS.LEVELID]=='5']
    exp_dpguide_aa = pd.merge(df_in_exp_dpguide, gui_saleshierarchy, how='inner',
                    left_on=[DPG.SALESID], right_on=[NS.SALESID])
    df_demand01['PROMISEDDELDATE2'] = common.convert_dt_to_week(pd.to_datetime(df_demand01[ND_.PROMISEDDELDATE]))

    df_demand01 = df_demand01.loc[
        (df_demand01[ND_.AP2ID] + df_demand01['PROMISEDDELDATE2'] + df_demand01[ND_.ITEMID]).isin(
            exp_dpguide_aa[NS.AP2ID] + exp_dpguide_aa[DPG.SWEEK]+ exp_dpguide_aa[DPG.UDAITEM]
        )
    ]


    # AND SALESID NOT LIKE '5%' 
    df_demand01 = df_demand01[~(df_demand01[ND_.SALESID].str.startswith('5'))]

    # AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
    df_demand01 = df_demand01.loc[
            ~(df_demand01[ND_.SITEID]).isin(
                df_in_netting_code_map.loc[
                    df_in_netting_code_map[CM.CODEMAPKEY].str.startswith('HC_WL_DP_CHSTOCK_EXCEPT')
                ][CM.CODE1]
            )
        ]

    #AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y')
    df_demand01 = df_demand01.loc[
        ~(df_demand01[ND_.ITEMID]+'Y').isin(
            df_in_Netting_Custom_Model_Map_ODS_W[CMM.CUSTOMITEM] + df_in_Netting_Custom_Model_Map_ODS_W[CMM.ISVALID]
        )
    ]

    # AND NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID)
    df_MVES_ITEMSITE = df_in_Netting_ES_Item_Site_ODS_W
    df_demand01 = df_demand01.loc[
            ~(df_demand01[ND_.ITEMID] + df_demand01[ND_.SALESID] + df_demand01[ND_.SITEID]).isin(
                df_MVES_ITEMSITE[ESIS.ITEM]+df_MVES_ITEMSITE[ESIS.SALESID]+df_MVES_ITEMSITE[ESIS.SITEID]
            )
        ]

    df_demand01[ND_.OPTION_CODE] = np.where(
        df_demand01[ND_.OPTION_CODE].isna(),
        0,
        np.where(df_demand01[ND_.OPTION_CODE]=='','0',df_demand01[ND_.OPTION_CODE])
    )
    df_demand01[ND_.OPTION_CODE] = df_demand01[ND_.OPTION_CODE].astype('int32')

    df_demand01['OPTION_CODE_NEW'] = df_demand01[ND_.OPTION_CODE] + 1
    df_demand01['UPBY_NEW'] = 'C/Stock_Upper'
    df_demand01 = df_demand01[['ROWA','OPTION_CODE_NEW','UPBY_NEW']]

    df_demand001 = df_demand001.merge(df_demand01, on=['ROWA'], how='left')
    df_demand001.loc[~df_demand001['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_demand001['OPTION_CODE_NEW']
    df_demand001 = df_demand001.drop(columns=['OPTION_CODE_NEW','UPBY_NEW'])

    #concat(insert) 후에 ROWA 다시 설정 2025-06-17
    df_demand001['ROWA'] = df_demand001.index.astype(str)

    logger.Step(4, '3,4 레벨 버퍼수량 전량 C/Stock Short End')
    #endregion logger.Step(4, '3,4 레벨 버퍼수량 전량 C/Stock Short')

    #region logger.Step(5, 'DP_GUDIE가 4레벨일때, 3레벨 전량 Short')
    logger.Step(5, 'DP_GUDIE가 4레벨일때, 3레벨 전량 Short Start')

    # --! DP_GUDIE가 4레벨일때, 3레벨 전량 Short !--
    df_demand02 = df_demand001.copy(deep=True)

    gui_saleshierarchy = df_in_gui_saleshierarchy.loc[df_in_gui_saleshierarchy[NS.LEVELID]=='4']
    exp_dpguide_aa = pd.merge(df_in_exp_dpguide, gui_saleshierarchy, how='inner',
                    left_on=[DPG.SALESID], right_on=[NS.SALESID])
    df_demand02['PROMISEDDELDATE2'] = common.convert_dt_to_week(pd.to_datetime(df_demand02[ND_.PROMISEDDELDATE]))

    df_demand02 = df_demand02.loc[
        (df_demand02[ND_.SALESID] + df_demand02['PROMISEDDELDATE2'] + df_demand02[ND_.ITEMID]).isin(
            exp_dpguide_aa[NS.AP2ID] + exp_dpguide_aa[DPG.SWEEK]+ exp_dpguide_aa[DPG.UDAITEM]
        )
    ]

    # AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
    df_demand02 = df_demand02.loc[
            ~(df_demand02[ND_.SITEID]).isin(
                df_in_netting_code_map.loc[
                    df_in_netting_code_map[CM.CODEMAPKEY].str.startswith('HC_WL_DP_CHSTOCK_EXCEPT')
                ][CM.CODE1]
            )
        ]

    # AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y')
    df_demand02 = df_demand02.loc[
        ~(df_demand02[ND_.ITEMID]+'Y').isin(
            df_in_Netting_Custom_Model_Map_ODS_W[CMM.CUSTOMITEM] + df_in_Netting_Custom_Model_Map_ODS_W[CMM.ISVALID]
        )
    ]
    # AND NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID)
    df_demand02 = df_demand02.loc[
            ~(df_demand02[ND_.ITEMID] + df_demand02[ND_.SALESID] + df_demand02[ND_.SITEID]).isin(
                df_MVES_ITEMSITE[ESIS.ITEM]+df_MVES_ITEMSITE[ESIS.SALESID]+df_MVES_ITEMSITE[ESIS.SITEID]
            )
        ]

    df_demand02[ND_.OPTION_CODE] = np.where(df_demand02[ND_.OPTION_CODE].isna(),0,np.where(df_demand02[ND_.OPTION_CODE]=='','0',df_demand02[ND_.OPTION_CODE]))
    df_demand02[ND_.OPTION_CODE] = df_demand02[ND_.OPTION_CODE].astype('int32')

    df_demand02['OPTION_CODE_NEW'] = df_demand02[ND_.OPTION_CODE] + 1
    df_demand02['UPBY_NEW'] = 'C/Stock_Upper'
    df_demand02 = df_demand02[['ROWA','OPTION_CODE_NEW','UPBY_NEW']]

    df_demand001 = df_demand001.merge(df_demand02, on=['ROWA'], how='left')
    df_demand001.loc[~df_demand001['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_demand001['OPTION_CODE_NEW']
    df_demand001 = df_demand001.drop(columns=['OPTION_CODE_NEW','UPBY_NEW'])

    logger.Step(5, 'DP_GUDIE가 4레벨일때, 3레벨 전량 Short End')
    #endregion logger.Step(5, 'DP_GUDIE가 4레벨일때, 3레벨 전량 Short')

    # 잔여 C/Stock 중 COM_ORD & NEW_ORD인 DEMAND 제외 처리 부분 로직 제외(2025-07-09)
    df_demand002 = df_demand001.copy(deep=True)
    #region logger.Step(6, 'COM_ORD 일 경우 C/Stock 제외 (최종 결과에서 COM_ORD가 짤렸을 경우 다시 살려줌)')
    # logger.Step(6, 'COM_ORD 일 경우 C/Stock 제외 (최종 결과에서 COM_ORD가 짤렸을 경우 다시 살려줌) Start')
    # # --SEA 이외 법인은 구간 상관없이 C/Stock 제외 17.02.02
    # # --SEA는 제외 로직에서 해제 하였으나 VZW 거래선만 적용 HC_WL_DP_COMNEW_CHSTOCK_ACNT 추가 19.07.22

    # df_demand03 = df_demand001.copy(deep=True)

    # # (s_row[S.OPTION_CODE_IDX] % 2) == 1
    # df_demand03['OPTION_CODE2'] = df_demand03[ND_.OPTION_CODE]
    # df_demand03.loc[df_demand03['OPTION_CODE2'].isna() ,'OPTION_CODE2'] = '0'
    # df_demand03.loc[df_demand03['OPTION_CODE2']=='','OPTION_CODE2'] = '0'
    # df_demand03['OPTION_CODE2'] = df_demand03['OPTION_CODE2'].astype('int32')
    # df_demand03['OPTION_CODE2'] = df_demand03['OPTION_CODE2'].mod(2)
    # df_demand03 = df_demand03.loc[df_demand03['OPTION_CODE2'] == 1]


    # # AND A.AP2ID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY = 'HC_WL_DP_COMNEW_CHSTOCK')
    # df_demand03 = df_demand03.loc[
    #         ~(df_demand03[ND_.AP2ID]).isin(
    #             df_in_netting_code_map.loc[
    #                 df_in_netting_code_map[CM.CODEMAPKEY].str.startswith('HC_WL_DP_COMNEW_CHSTOCK')
    #             ][CM.CODE1]
    #         )
    #     ]

    # # AND A.SALESID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY = 'HC_WL_DP_COMNEW_CHSTOCK_ACNT')
    # df_demand03 = df_demand03.loc[
    #         ~(df_demand03[ND_.SALESID]).isin(
    #             df_in_netting_code_map.loc[
    #                 df_in_netting_code_map[CM.CODEMAPKEY].str.startswith('HC_WL_DP_COMNEW_CHSTOCK_ACNT')
    #             ][CM.CODE1]
    #         )
    #     ]

    # # AND FN_EXTRACT(A.SALESORDERID, '::',1) IN ( 'COM_ORD','NEW_ORD')
    # df_demand03['SALESORDERID2'] = df_demand03[ND_.SALESORDERID].str.split('::').str[0]
    # df_demand03 = df_demand03.loc[df_demand03['SALESORDERID2'].isin(['COM_ORD','NEW_ORD'])]

    # # AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y')
    # df_demand03 = df_demand03.loc[
    #     ~(df_demand03[ND_.ITEMID]+'Y').isin(
    #         df_in_Netting_Custom_Model_Map_ODS_W[CMM.CUSTOMITEM] + df_in_Netting_Custom_Model_Map_ODS_W[CMM.ISVALID]
    #     )
    # ]
    # # AND NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID)
    # df_demand03 = df_demand03.loc[
    #         ~(df_demand03[ND_.ITEMID] + df_demand03[ND_.SALESID] + df_demand03[ND_.SITEID]).isin(
    #             df_MVES_ITEMSITE[ESIS.ITEM]+df_MVES_ITEMSITE[ESIS.SALESID]+df_MVES_ITEMSITE[ESIS.SITEID]
    #         )
    #     ]

    # df_demand03[ND_.OPTION_CODE] = df_demand03[ND_.OPTION_CODE].astype('int32')

    # df_demand03['OPTION_CODE_NEW'] = df_demand03[ND_.OPTION_CODE] - 1
    # df_demand03['UPBY_NEW'] = 'C/Stock_COM_ORD'

    # df_demand03 = df_demand03[['ROWA','OPTION_CODE_NEW','UPBY_NEW']]

    # df_demand001 = df_demand001.merge(df_demand03, on=['ROWA'], how='left')
    # df_demand001.loc[~df_demand001['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_demand001['OPTION_CODE_NEW']
    # df_demand001 = df_demand001.drop(columns=['OPTION_CODE_NEW','UPBY_NEW'])

    # # --2018.12.03 수정(V PLAN 시 당주기준으로 PROMISEDDELDATE 변경)

    # df_demand04 = df_demand001.copy(deep=True)

    # # MOD(NVL(OPTION_CODE,0), 2) = 1
    # df_demand04['OPTION_CODE2'] = df_demand04[ND_.OPTION_CODE]
    # df_demand04.loc[df_demand04['OPTION_CODE2'].isna() ,'OPTION_CODE2'] = '0'
    # df_demand04.loc[df_demand04['OPTION_CODE2']=='','OPTION_CODE2'] = '0'
    # df_demand04['OPTION_CODE2'] = df_demand04['OPTION_CODE2'].astype('int32')
    # df_demand04['OPTION_CODE2'] = df_demand04['OPTION_CODE2'].mod(2)
    # df_demand04 = df_demand04.loc[df_demand04['OPTION_CODE2'] == 1]


    # # AND    A.AP2ID = CODE1   
    # # AND    B.CATEGORY = 'HC_WL_DP_COMNEW_CHSTOCK'
    # dfMTA_CODEMAP =  df_in_netting_code_map.loc[
    #     df_in_netting_code_map[CM.CODEMAPKEY].str.startswith('HC_WL_DP_COMNEW_CHSTOCK')
    # ]
    # dfMTA_CODEMAP = dfMTA_CODEMAP[[CM.CODE1,CM.NUM1,]]

    # df_demand04[ND_.PROMISEDDELDATE]  = df_demand04[ND_.PROMISEDDELDATE].astype('datetime64[ns]')

    # df_demand04 = df_demand04.merge(dfMTA_CODEMAP, how='inner',
    #                 left_on=[ND_.AP2ID], right_on=[CM.CODE1])
    # df_demand04[CM.NUM1] = df_demand04[CM.NUM1].astype('float').astype('int32')
    # df_demand04['PROMISEDDELDATE_DELTA'] = datetime.now() + pd.to_timedelta(((df_demand04[CM.NUM1]+V_NUM)*7), unit='D')

    # # AND    PROMISEDDELDATE < SYSDATE + ((NUM1+V_NUM)*7)
    # df_demand04.loc[df_demand04[ND_.PROMISEDDELDATE] < df_demand04['PROMISEDDELDATE_DELTA']]


    # # AND FN_EXTRACT(A.SALESORDERID, '::',1) IN ( 'COM_ORD','NEW_ORD')
    # df_demand04['SALESORDERID2'] = df_demand04[ND_.SALESORDERID].str.split('::').str[0]
    # df_demand04 = df_demand04.loc[df_demand04['SALESORDERID2'].isin(['COM_ORD','NEW_ORD'])]

    # # AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y')
    # df_demand04 = df_demand04.loc[
    #     ~(df_demand04[ND_.ITEMID]+'Y').isin(
    #         df_in_Netting_Custom_Model_Map_ODS_W[CMM.CUSTOMITEM] + df_in_Netting_Custom_Model_Map_ODS_W[CMM.ISVALID]
    #     )
    # ]
    # # AND NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID)
    # df_demand04 = df_demand04.loc[
    #         ~(df_demand04[ND_.ITEMID] + df_demand04[ND_.SALESID] + df_demand04[ND_.SITEID]).isin(
    #             df_MVES_ITEMSITE[ESIS.ITEM]+df_MVES_ITEMSITE[ESIS.SALESID]+df_MVES_ITEMSITE[ESIS.SITEID]
    #         )
    #     ]

    # df_demand04[ND_.OPTION_CODE] = df_demand04[ND_.OPTION_CODE].astype('int32')

    # df_demand04['OPTION_CODE_NEW'] = df_demand04[ND_.OPTION_CODE] - 1
    # df_demand04['UPBY_NEW'] = 'C/Stock_COM_ORD'

    # df_demand04 = df_demand04[['ROWA','OPTION_CODE_NEW','UPBY_NEW']]

    # df_demand001 = df_demand001.merge(df_demand04, on=['ROWA'], how='left')
    # df_demand001.loc[~df_demand001['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_demand001['OPTION_CODE_NEW']
    # df_demand001 = df_demand001.drop(columns=['OPTION_CODE_NEW','UPBY_NEW'])

    # # -- SEA는 제외 로직에서 해제 하였으나 VZW 거래선만 적용 HC_WL_DP_COMNEW_CHSTOCK_ACNT 추가 19.07.22
    # # -- 위 AP2ID와 중복되는 데이터 있으면 안됨

    # df_demand05 = df_demand001.copy(deep=True)

    # # MOD(NVL(OPTION_CODE,0), 2) = 1
    # df_demand05['OPTION_CODE2'] = df_demand05[ND_.OPTION_CODE]
    # df_demand05.loc[df_demand05['OPTION_CODE2'].isna() ,'OPTION_CODE2'] = '0'
    # df_demand05.loc[df_demand05['OPTION_CODE2']=='','OPTION_CODE2'] = '0'
    # df_demand05['OPTION_CODE2'] = df_demand05['OPTION_CODE2'].astype('int32')
    # df_demand05['OPTION_CODE2'] = df_demand05['OPTION_CODE2'].mod(2)
    # df_demand05 = df_demand05.loc[df_demand05['OPTION_CODE2'] == 1]


    # # AND    A.AP2ID = CODE1   
    # # AND    B.CATEGORY = 'HC_WL_DP_COMNEW_CHSTOCK'
    # dfMTA_CODEMAP =  df_in_netting_code_map.loc[
    #     df_in_netting_code_map[CM.CODEMAPKEY].str.startswith('HC_WL_DP_COMNEW_CHSTOCK_ACNT')
    # ]
    # dfMTA_CODEMAP = dfMTA_CODEMAP[[CM.CODE1,CM.NUM1,]]

    # df_demand05[ND_.PROMISEDDELDATE]  = df_demand05[ND_.PROMISEDDELDATE].astype('datetime64[ns]')

    # df_demand05 = df_demand05.merge(dfMTA_CODEMAP, how='inner',
    #                 left_on=[ND_.AP2ID], right_on=[CM.CODE1])
    # df_demand05[CM.NUM1] = df_demand05[CM.NUM1].astype('float').astype('int32')
    # df_demand05['PROMISEDDELDATE_DELTA'] = datetime.now() + pd.to_timedelta(((df_demand05[CM.NUM1]+V_NUM)*7), unit='D')

    # # AND    PROMISEDDELDATE < SYSDATE + ((NUM1+V_NUM)*7)
    # df_demand05.loc[df_demand05[ND_.PROMISEDDELDATE] < df_demand05['PROMISEDDELDATE_DELTA']]


    # # AND FN_EXTRACT(A.SALESORDERID, '::',1) IN ( 'COM_ORD','NEW_ORD')
    # df_demand05['SALESORDERID2'] = df_demand05[ND_.SALESORDERID].str.split('::').str[0]
    # df_demand05 = df_demand05.loc[df_demand05['SALESORDERID2'].isin(['COM_ORD','NEW_ORD'])]

    # # AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y')
    # df_demand05 = df_demand05.loc[
    #     ~(df_demand05[ND_.ITEMID]+'Y').isin(
    #         df_in_Netting_Custom_Model_Map_ODS_W[CMM.CUSTOMITEM] + df_in_Netting_Custom_Model_Map_ODS_W[CMM.ISVALID]
    #     )
    # ]
    # # AND NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID)
    # df_demand05 = df_demand05.loc[
    #         ~(df_demand05[ND_.ITEMID] + df_demand05[ND_.SALESID] + df_demand05[ND_.SITEID]).isin(
    #             df_MVES_ITEMSITE[ESIS.ITEM]+df_MVES_ITEMSITE[ESIS.SALESID]+df_MVES_ITEMSITE[ESIS.SITEID]
    #         )
    #     ]

    # df_demand05[ND_.OPTION_CODE] = df_demand05[ND_.OPTION_CODE].astype('int32')

    # df_demand05['OPTION_CODE_NEW'] = df_demand05[ND_.OPTION_CODE] - 1
    # df_demand05['UPBY_NEW'] = 'C/Stock_COM_ORD'

    # df_demand05 = df_demand05[['ROWA','OPTION_CODE_NEW','UPBY_NEW']]

    # df_demand001 = df_demand001.merge(df_demand05, on=['ROWA'], how='left')
    # df_demand001.loc[~df_demand001['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_demand001['OPTION_CODE_NEW']
    # # df_demand001.loc[~df_demand001['UPBY_NEW'].isna(),'UPBY'] = df_demand001['UPBY_NEW']
    # df_demand001 = df_demand001.drop(columns=['OPTION_CODE_NEW','UPBY_NEW'])

    # logger.Step(6, 'COM_ORD 일 경우 C/Stock 제외 (최종 결과에서 COM_ORD가 짤렸을 경우 다시 살려줌) End')
    #endregion logger.Step(6, 'COM_ORD 일 경우 C/Stock 제외 (최종 결과에서 COM_ORD가 짤렸을 경우 다시 살려줌)')

    # UNF_ORD 살리는 작업 제외 (2025-07-09)
    #region logger.Step(7, 'CH.STOCK으로 잘린 것들 UNF_ORD 수량만큼 살려준다 2016.06.08')
    # logger.Step(7, 'CH.STOCK으로 잘린 것들 UNF_ORD 수량만큼 살려준다 2016.06.08 Start')


    # # AND B.CATEGORY(+) IN ( 'HC_WL_DP_UNFORD_CHSTOCK','HC_WL_DP_UNFORD_CHSTOCK_ACNT')
    # dfMTA_CODEMAP = df_in_netting_code_map.loc[
    #     df_in_netting_code_map[CM.CODEMAPKEY].str.startswith((
    #         'HC_WL_DP_UNFORD_CHSTOCK', 'HC_WL_DP_UNFORD_CHSTOCK_ACNT'
    #     ))
    # ]

    # #    AND DECODE(B.CATEGORY(+),'HC_WL_DP_UNFORD_CHSTOCK' 
    # #                            , REGEXP_SUBSTR(A.SALESORDERID, '[^::]+', 1, 5)
    # #                            ,'HC_WL_DP_UNFORD_CHSTOCK_ACNT'
    # #                            , REGEXP_SUBSTR(A.SALESORDERID, '[^::]+', 1, 6)) = B.CODE1(+)
    # dfMTA_CODEMAP = dfMTA_CODEMAP[[CM.CODEMAPKEY,CM.CODE1,CM.CODE2_NUM,]]

    # df_shortreason = df_exp_ap1_shortreason.copy(deep=True)

    # df_shortreason['GUBUN'] = df_shortreason[AP1SR_.SALESORDERID].str.split('::').str[0]
    # df_shortreason['SALESID'] = df_shortreason[AP1SR_.SALESORDERID].str.split('::').str[4]
    # df_shortreason['SALESID2'] = df_shortreason[AP1SR_.SALESORDERID].str.split('::').str[5]

    # # AND REGEXP_SUBSTR(A.SALESORDERID, '[^::]+', 1, 1) = 'UNF_ORD'
    # df_shortreason = df_shortreason.loc[df_shortreason['GUBUN'] == 'UNF_ORD']

    # # MTA_CODEMAP 정의
    # dfMTA_CODEMAP = df_in_netting_code_map.loc[
    #     df_in_netting_code_map[CM.CODEMAPKEY].str.startswith((
    #         'HC_WL_DP_UNFORD_CHSTOCK', 'HC_WL_DP_UNFORD_CHSTOCK_ACNT'
    #     ))
    # ]
    # dfMTA_CODEMAP = dfMTA_CODEMAP[[CM.CODEMAPKEY,CM.CODE1,CM.CODE2_NUM,]]

    # # #    AND DECODE(B.CATEGORY(+),'HC_WL_DP_UNFORD_CHSTOCK' 
    # # #                            , REGEXP_SUBSTR(A.SALESORDERID, '[^::]+', 1, 5)
    # # #                            ,'HC_WL_DP_UNFORD_CHSTOCK_ACNT'
    # # #                            , REGEXP_SUBSTR(A.SALESORDERID, '[^::]+', 1, 6)) = B.CODE1(+)
    # # dfMTA_CODEMAP = dfMTA_CODEMAP[['CATEGORY','CODE1','CODE2_NUM']]
    # # df_shortreason['JOIN_KEY'] = df_shortreason.apply(lambda row:row['SALESID'] if 'HC_WL_DP_UNFORD_CHSTOCK' in dfMTA_CODEMAP['CATEGORY'].values else row['SALESID2'], axis=1)
    # # df_shortreason = df_shortreason.merge(dfMTA_CODEMAP, left_on='JOIN_KEY', right_on='CODE1', how='left')

    # # 일단 위의 쿼리를 아래 쿼리로 단순화 하여 코딩함 
    # # SELECT A.SITEID, A.ITEM , A.DUEDATE , A.SHORTQTY UNFORDQTY , 0 CHSTOCKQTY
    # #   FROM EXP_AP1_SHORTREASON A, MTA_CODEMAP B
    # #  WHERE A.PLANID = '202509_M'
    # #    AND REGEXP_SUBSTR(A.SALESORDERID, '[^::]+', 1, 5) = B.CODE1(+)
    # #    AND A.DUEDATE < SYSDATE + 7 * NVL(B.CODE2_NUM+0, 8) 

    # df_shortreason = df_shortreason.merge(dfMTA_CODEMAP, how='left',left_on='SALESID', right_on=CM.CODE1 )


    # # # AND A.DUEDATE < SYSDATE + 7 * NVL(B.CODE2_NUM+V_NUM, 8)
    # df_shortreason[CM.CODE2_NUM] = df_shortreason[CM.CODE2_NUM].astype('str')
    # df_shortreason[CM.CODE2_NUM] = df_shortreason.apply(
    #     lambda row:'8' if np.isnan(float(row[CM.CODE2_NUM])) else str(float(row[CM.CODE2_NUM])+V_NUM), axis=1
    # )
    # df_shortreason[CM.CODE2_NUM] = df_shortreason[CM.CODE2_NUM].astype('float').astype('int32')

    # dt_startdate = datetime.combine(date.fromisoformat(V_EFFSTARTDATE),datetime.min.time())

    # # df_shortreason['DUEDATE_DELTA'] = datetime.now() + pd.to_timedelta(((df_shortreason[CM.CODE2_NUM]) *7), unit='D')
    # df_shortreason['DUEDATE_DELTA'] = dt_startdate + pd.to_timedelta(((df_shortreason[CM.CODE2_NUM]) *7), unit='D')

    # df_shortreason[AP1SR_.DUEDATE]  = df_shortreason[AP1SR_.DUEDATE].astype('datetime64[ns]')

    # df_shortreason = df_shortreason.loc[df_shortreason[AP1SR_.DUEDATE] < df_shortreason['DUEDATE_DELTA']]

    # # AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.REQITEM AND ISVALID = 'Y')
    # df_shortreason = df_shortreason.loc[
    #     ~(df_shortreason[AP1SR_.REQITEMID]+'Y').isin(
    #         df_in_Netting_Custom_Model_Map_ODS_W[CMM.CUSTOMITEM] + df_in_Netting_Custom_Model_Map_ODS_W[CMM.ISVALID]
    #     )
    # ]

    # df_shortreason = df_shortreason[['SALESID2',AP1SR_.SITEID,AP1SR_.ITEM,AP1SR_.DUEDATE,AP1SR_.SHORTQTY]]

    # df_shortreason.rename(columns={
    #     'SALESID2':'SALESID', AP1SR_.SHORTQTY:'UNFORDQTY',
    # }, inplace=True)

    # df_shortreason['CHSTOCKQTY'] = 0
    # df_shortreason.columns = ['SALESID', 'SITEID', 'ITEM', 'DUEDATE', 'UNFORDQTY', 'CHSTOCKQTY']




    # df_demandinner = df_demand001.copy(deep=True)

    # # AND MOD(NVL(A.OPTION_CODE,0), 2) = 1
    # df_demandinner['OPTION_CODE2'] = df_demandinner[ND_.OPTION_CODE]
    # df_demandinner.loc[df_demandinner['OPTION_CODE2'].isna() ,'OPTION_CODE2'] = '0'
    # df_demandinner.loc[df_demandinner['OPTION_CODE2']=='','OPTION_CODE2'] = '0'

    # df_demandinner['OPTION_CODE2'] = df_demandinner['OPTION_CODE2'].astype('int32')
    # df_demandinner['OPTION_CODE2'] = df_demandinner['OPTION_CODE2'].mod(2)
    # df_demandinner = df_demandinner.loc[df_demandinner['OPTION_CODE2']==1]

    # # AND A.SALESID LIKE '5%'
    # df_demandinner = df_demandinner[df_demandinner[ND_.SALESID].str.startswith('5')]

    # # # AND B.CATEGORY(+) IN ( 'HC_WL_DP_UNFORD_CHSTOCK','HC_WL_DP_UNFORD_CHSTOCK_ACNT')
    # # dfMTA_CODEMAP = df_in_netting_code_map.loc[
    # #         df_in_netting_code_map['CATEGORY'].isin(['HC_WL_DP_UNFORD_CHSTOCK', 'HC_WL_DP_UNFORD_CHSTOCK_ACNT'])
    # #     ]
    # # dfMTA_CODEMAP = dfMTA_CODEMAP[['CATEGORY','CODE1','CODE2_NUM']]


    # # #    AND DECODE(B.CATEGORY(+),'HC_WL_DP_UNFORD_CHSTOCK' 
    # # #                            , A.AP2ID
    # # #                            ,'HC_WL_DP_UNFORD_CHSTOCK_ACNT'
    # # #                            , A.SALESID) = B.CODE1(+)
    # # df_demandinner['JOIN_KEY'] = df_demandinner.apply(lambda row:row['AP2ID'] if 'HC_WL_DP_UNFORD_CHSTOCK' in dfMTA_CODEMAP['CATEGORY'].values else row['SALESID'], axis=1)
    # # df_demandinner = df_demandinner.merge(dfMTA_CODEMAP, left_on='JOIN_KEY', right_on='CODE1', how='left')

    # df_demandinner = df_demandinner.merge(dfMTA_CODEMAP, how='left',left_on=ND_.AP2ID, right_on=CM.CODE1)


    # # AND A.PROMISEDDELDATE < SYSDATE + 7 * NVL(B.CODE2_NUM+V_NUM, 8)
    # df_demandinner[CM.CODE2_NUM] = df_demandinner[CM.CODE2_NUM].astype('str')
    # df_demandinner[CM.CODE2_NUM] = df_demandinner.apply(
    #     lambda row:'8' if np.isnan(float(row[CM.CODE2_NUM])) else str(float(row[CM.CODE2_NUM])+V_NUM), axis=1
    # )
    # df_demandinner[CM.CODE2_NUM] = df_demandinner[CM.CODE2_NUM].astype('float').astype('int32')

    # df_demandinner['DUEDATE_DELTA'] =  datetime.combine(date.fromisoformat(V_EFFSTARTDATE),datetime.min.time()) + pd.to_timedelta(((df_demandinner[CM.CODE2_NUM]) *7), unit='D')
    # df_demandinner['DUEDATE']  = df_demandinner[ND_.PROMISEDDELDATE].astype('datetime64[ns]')
    # df_demandinner = df_demandinner.loc[df_demandinner['DUEDATE'] < df_demandinner['DUEDATE_DELTA']]

    # # AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.REQITEM AND ISVALID = 'Y')
    # df_demandinner = df_demandinner.loc[
    #     ~(df_demandinner[ND_.ITEMID]+'Y').isin(
    #         df_in_Netting_Custom_Model_Map_ODS_W[CMM.CUSTOMITEM] + df_in_Netting_Custom_Model_Map_ODS_W[CMM.ISVALID]
    #     )
    # ]

    # # AND NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID)
    # df_MVES_ITEMSITE = df_in_Netting_ES_Item_Site_ODS_W
    # df_demandinner = df_demandinner.loc[
    #         ~(df_demandinner[ND_.ITEMID] + df_demandinner[ND_.SALESID] + df_demandinner[ND_.SITEID]).isin(
    #             df_MVES_ITEMSITE[ESIS.ITEM]+df_MVES_ITEMSITE[ESIS.SALESID]+df_MVES_ITEMSITE[ESIS.SITEID]
    #         )
    #     ]

    # df_demandinner['UNFORDQTY'] = 0


    # df_demandinner.rename(columns={
    #     ND_.QTYPROMISED:'CHSTOCKQTY'
    # }, inplace=True)


    # df_demandinner=df_demandinner[[ND_.SALESID, ND_.SITEID, ND_.ITEMID, 'DUEDATE', 'UNFORDQTY', 'CHSTOCKQTY']]
    # df_demandinner.columns = ['SALESID', 'SITEID', 'ITEM', 'DUEDATE', 'UNFORDQTY', 'CHSTOCKQTY']


    # df_demandinnerqq = df_demand001.copy(deep=True)
    # df_demandinnerqq[ND_.PROMISEDDELDATE]  = df_demandinnerqq[ND_.PROMISEDDELDATE].astype('datetime64[ns]')
    # df_demandinnerqq['PROMISEDDELDATE2'] = df_demandinnerqq[ND_.PROMISEDDELDATE].dt.strftime('%Y-%m-%d')
    # df_shortreason['UNFORDQTY']  = df_shortreason['UNFORDQTY'].astype('float32').astype('int32')
    # df_shortreason['CHSTOCKQTY']  = df_shortreason['CHSTOCKQTY'].astype('float32').astype('int32')
    # df_demandinner['UNFORDQTY']  = df_demandinner['UNFORDQTY'].astype('float32').astype('int32')
    # df_demandinner['CHSTOCKQTY']  = df_demandinner['CHSTOCKQTY'].astype('float32').astype('int32')

    # df_ss_inner = pd.concat([df_shortreason, df_demandinner], axis=0, ignore_index=True)

    # df_ss_inner = df_ss_inner.groupby(by=['SALESID','SITEID','ITEM','DUEDATE'])[['UNFORDQTY','CHSTOCKQTY']].sum()
    # df_ss_inner['RNK'] = (df_ss_inner.sort_values(by=['DUEDATE'], ascending=[True]).groupby(['SALESID','SITEID','ITEM']).cumcount() + 1)

    # df_ss_inner = df_ss_inner.reset_index()

    # # SUM(UNFORDQTY) OVER (PARTITION BY SALESID, SITEID, ITEM) UNFORDSUM
    # # SUM(CHSTOCKQTY) OVER (PARTITION BY SALESID, SITEID, ITEM) CHSTOCKSUM

    # df_ss_inner[['UNFORDSUM','CHSTOCKSUM']]= df_ss_inner.groupby(['SALESID', 'SITEID', 'ITEM'])[['UNFORDQTY','CHSTOCKQTY']].transform('sum')

    # df_ss_inner = df_ss_inner.reset_index()

    # df_ss_inner.rename(columns={
    #     'DUEDATE':'PROMISEDDELDATE'
    # }, inplace=True)

    # # WHERE UNFORDSUM <> 0 AND CHSTOCKSUM <> 0
    # df_ss_inner = df_ss_inner.loc[df_ss_inner['UNFORDSUM'] > 0 ]
    # df_ss_inner = df_ss_inner.loc[df_ss_inner['CHSTOCKSUM'] > 0 ]
    # df_ss_inner = df_ss_inner[['SALESID', 'SITEID', 'ITEM', 'PROMISEDDELDATE', 'UNFORDQTY', 'CHSTOCKQTY',  'UNFORDSUM', 'CHSTOCKSUM', 'RNK']]

    # df_demandinnerqq = df_demand001.copy(deep=True)
    # df_demandinnerqq[ND_.PROMISEDDELDATE]  = df_demandinnerqq[ND_.PROMISEDDELDATE].astype('datetime64[ns]')
    # df_demandinnerqq['PROMISEDDELDATE3'] = df_demandinnerqq[ND_.PROMISEDDELDATE].dt.strftime('%Y-%m-%d')

    # # AND MOD(NVL(OPTION_CODE,0), 2) = 1
    # df_demandinnerqq['OPTION_CODE2'] = df_demandinnerqq[ND_.OPTION_CODE]
    # df_demandinnerqq.loc[df_demandinnerqq['OPTION_CODE2'].isna() ,'OPTION_CODE2'] = '0'
    # df_demandinnerqq.loc[df_demandinnerqq['OPTION_CODE2']=='','OPTION_CODE2'] = '0'

    # df_demandinnerqq['OPTION_CODE2'] = df_demandinnerqq['OPTION_CODE2'].astype('int32')
    # df_demandinnerqq['OPTION_CODE2'] = df_demandinnerqq['OPTION_CODE2'].mod(2)
    # df_demandinnerqq = df_demandinnerqq.loc[df_demandinnerqq['OPTION_CODE2']==1]

    # df_demandinnerqq[ND_.QTYPROMISED] = df_demandinnerqq[ND_.QTYPROMISED].astype('int32')

    # ND_.ROWA_IDX = df_demandinnerqq.columns.get_loc('ROWA')
    # count_using_column = len(ND_.LIST_COLUMN) + 1 # ROWA

    # list_Outbound_loop_update_ss  = []

    # list_Outbound_loop_update_qq  = []
    # list_Outbound_loop_update_qq2 = []
    # list_Outbound_loop_insert_qq  = []




    # nparr_ss_inner = df_ss_inner.to_numpy()

    # for ss_row in nparr_ss_inner:

    #     if(int(ss_row[SS.RNK_IDX]) == 1):
    #         V_AVAILQTY2 = ss_row[SS.UNFORDQTY_IDX]
    #     else:
    #         V_AVAILQTY2 = ss_row[SS.UNFORDQTY_IDX] + V_ROLLING

    #     if(V_AVAILQTY2 >= ss_row[SS.CHSTOCKQTY_IDX]):
    #         V_ALIVE = ss_row[SS.CHSTOCKQTY_IDX]
    #     else:
    #         V_ALIVE = V_AVAILQTY2

    #     if((V_AVAILQTY2 - int(ss_row[SS.CHSTOCKQTY_IDX])) < 0):
    #         V_ROLLING = 0
    #     else:
    #         V_ROLLING = V_AVAILQTY2 - int(ss_row[SS.CHSTOCKQTY_IDX])

    #     if((int(ss_row[SS.CHSTOCKQTY_IDX]) - V_AVAILQTY2) < 0):
    #         V_CHSTOCK = 0
    #     else:
    #         V_CHSTOCK = int(ss_row[SS.CHSTOCKQTY_IDX]) - V_AVAILQTY2


    #     if(V_AVAILQTY2 >= int(ss_row[SS.CHSTOCKQTY_IDX])):

    #         list_Outbound_loop_update_ss.append(
    #             np.array([
    #                 V_PLANID, str(1), ss_row[SS.SALESID_IDX], ss_row[SS.SITEID_IDX],
    #                 ss_row[SS.ITEM_IDX], str(ss_row[SS.PROMISEDDELDATE_IDX]), str(0), 'C/Stock_UNF_ORD'
    #             ])
    #         )


    #         if((V_AVAILQTY2 - int(ss_row[SS.CHSTOCKQTY_IDX])) < 0):
    #             V_AVAILQTY2 = 0
    #         else:
    #             V_AVAILQTY2 -= int(ss_row[SS.CHSTOCKQTY_IDX])

    #     elif((V_AVAILQTY2 < int(ss_row[SS.CHSTOCKQTY_IDX])) & (V_AVAILQTY2 > 0)):
    #         #  AND SALESID = SS.SALESID
    #         #  AND SITEID = SS.SITEID
    #         #  AND ITEM = SS.ITEM
    #         #  AND PROMISEDDELDATE = SS.PROMISEDDELDATE
    #         df_demandinnerqq_loop = df_demandinnerqq.loc[df_demandinnerqq[ND_.SALESID] == ss_row[SS.SALESID_IDX]]
    #         df_demandinnerqq_loop = df_demandinnerqq_loop.loc[df_demandinnerqq_loop[ND_.SITEID] == ss_row[SS.SITEID_IDX]]
    #         df_demandinnerqq_loop = df_demandinnerqq_loop.loc[df_demandinnerqq_loop[ND_.ITEMID] == ss_row[SS.ITEM_IDX]]
    #         df_demandinnerqq_loop = df_demandinnerqq_loop.loc[
    #             df_demandinnerqq_loop['PROMISEDDELDATE3'] == ss_row[SS.PROMISEDDELDATE_IDX].strftime('%Y-%m-%d')
    #         ]

    #         # ORDER BY DEMANDPRIORITY, QTYPROMISED
    #         df_demandinnerqq_loop = df_demandinnerqq_loop.sort_values([ND_.DEMANDPRIORITY, ND_.QTYPROMISED], ascending=[True,True])

    #         nparr_qq = df_demandinnerqq_loop.to_numpy()
    #         for qq_row in nparr_qq:
    #             if(V_AVAILQTY2 >= int(qq_row[ND_.QTYPROMISED_IDX])):
    #                 list_Outbound_loop_update_qq.append(np.array([str(V_PLANID), str(1), str(qq_row[ND_.SALESORDERID_IDX]), str(qq_row[ND_.SOLINENUM_IDX]), str(0), str('C/Stock_UNF_ORD_PART1')]))

    #                 # V_AVAILQTY2 := CASE WHEN V_AVAILQTY2 - ND_.QTYPROMISED <0 THEN 0 ELSE V_AVAILQTY2 - ND_.QTYPROMISED END ;
    #                 if(V_AVAILQTY2 - int(qq_row[ND_.QTYPROMISED_IDX]) < 0) :
    #                     V_AVAILQTY2 = 0
    #                 else:
    #                     V_AVAILQTY2 -= int(qq_row[ND_.QTYPROMISED_IDX])
    #             elif((V_AVAILQTY2 < int(qq_row[ND_.QTYPROMISED_IDX])) & (V_AVAILQTY2 > 0)):
    #                 list_Outbound_loop_update_qq2.append(np.array([str(V_PLANID), str(qq_row[ND_.SALESORDERID_IDX]), str(qq_row[ND_.SOLINENUM_IDX]), str(int(qq_row[ND_.QTYPROMISED_IDX])- V_AVAILQTY2), 'C/Stock_UNF_ORD_PART3'+str(V_AVAILQTY2), str(qq_row[ND_.OPTION_CODE_IDX])]))

    #                 V_SOLINENUM = int(qq_row[ND_.SOLINENUM_IDX])+1
    #                 upby = 'C/Stock_UNF_ORD_PART2'
    #                 # --! INSERT 중복시 예외 처리 !-- 시작
    #                 max_solinenum = 0
    #                 b_same_solinenum = False
    #                 set_solinenum = dict_salesorder[qq_row[ND_.SALESORDERID_IDX]]
                    
    #                 for in_solinenum in set_solinenum:
    #                     if(max_solinenum <= in_solinenum):
    #                         max_solinenum = in_solinenum
    #                         upby = 'C/Stock_UNF_ORD_PART2'+str(V_AVAILQTY2)

    #                     if(V_SOLINENUM == in_solinenum):
    #                         #기존값이존재하는경우
    #                         b_same_solinenum = True
                            
    #                 if(b_same_solinenum):
    #                     V_SOLINENUM = max_solinenum + 1
    #                     upby = 'C/Stock_UNF_ORD_PART2'

    #                 set_solinenum.add(V_SOLINENUM)
    #                 dict_salesorder[qq_row[ND_.SALESORDERID_IDX]] = set_solinenum
    #                 # --! INSERT 중복시 예외 처리 !-- 끝
                    
    #                 i_qtypromise = V_AVAILQTY2
    #                 i_optioncode = int(np.nan_to_num(qq_row[ND_.OPTION_CODE_IDX], copy=False))-1

    #                 row = qq_row[:count_using_column].copy()
    #                 row[ND_.ROWA_IDX] = 0
    #                 row[ND_.SOLINENUM_IDX] = V_SOLINENUM
    #                 row[ND_.QTYPROMISED_IDX] = i_qtypromise
    #                 row[ND_.OPTION_CODE_IDX] = i_optioncode
    #                 list_Outbound_loop_insert_qq.append(row)

    #                 # V_AVAILQTY2 := CASE WHEN V_AVAILQTY2 - ND_.QTYPROMISED < 0 THEN 0 ELSE V_AVAILQTY2 - ND_.QTYPROMISED END;
    #                 if(V_AVAILQTY2 - int(qq_row[ND_.QTYPROMISED_IDX]) < 0):
    #                     V_AVAILQTY2 = 0
    #                 else:
    #                     V_AVAILQTY2 -= int(qq_row[ND_.QTYPROMISED_IDX])


    # df_demand001['OPTION_CODE2'] = df_demand001[ND_.OPTION_CODE]
    # df_demand001.loc[df_demand001['OPTION_CODE2'].isna() ,'OPTION_CODE2'] = '0'
    # df_demand001.loc[df_demand001['OPTION_CODE2']=='','OPTION_CODE2'] = '0'

    # df_demand001['OPTION_CODE2'] = df_demand001['OPTION_CODE2'].astype('int32')
    # df_demand001['OPTION_CODE2'] = df_demand001['OPTION_CODE2'].mod(2)
    # df_demand001[ND_.PROMISEDDELDATE]  = df_demand001[ND_.PROMISEDDELDATE].astype('datetime64[ns]')
    # df_demand001[ND_.QTYPROMISED]  = df_demand001[ND_.QTYPROMISED].astype('object')
    # df_demand001[ND_.SOLINENUM] = df_demand001[ND_.SOLINENUM].astype('int32')


    # df_outbound_update_ss = pd.DataFrame(list_Outbound_loop_update_ss)
    # df_outbound_update_ss.columns = [ND_.PLANID,'OPTION_CODE2',ND_.SALESID,ND_.SITEID,ND_.ITEMID,ND_.PROMISEDDELDATE,'OPTION_CODE_NEW','UPBY_NEW']
    # df_outbound_update_ss = df_outbound_update_ss.astype({'OPTION_CODE2':'int32', ND_.PROMISEDDELDATE:'datetime64[ns]'})

    # df_outbound_update_ss = df_outbound_update_ss.drop_duplicates()
    # df_demand001 = df_demand001.drop_duplicates()

    # df_demand001 = df_demand001.merge(
    #     df_outbound_update_ss, how='left',
    #     on=[ND_.PLANID,'OPTION_CODE2',ND_.SALESID,ND_.SITEID,ND_.ITEMID,ND_.PROMISEDDELDATE,]
    # )
    # df_demand001.loc[~df_demand001['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_demand001['OPTION_CODE_NEW']
    # df_demand001 = df_demand001.drop(columns=['OPTION_CODE_NEW','UPBY_NEW'])


    # df_outbound_update_qq = pd.DataFrame(list_Outbound_loop_update_qq)
    # df_outbound_update_qq.columns = [ND_.PLANID,'OPTION_CODE2',ND_.SALESORDERID,ND_.SOLINENUM,'OPTION_CODE_NEW','UPBY_NEW']
    # df_outbound_update_qq = df_outbound_update_qq.astype({'OPTION_CODE2':'int32',ND_.SOLINENUM:'int32'})

    # df_demand001 = df_demand001.merge(
    #     df_outbound_update_qq, how='left', on=[ND_.PLANID,'OPTION_CODE2',ND_.SALESORDERID,ND_.SOLINENUM]
    # )
    # df_demand001.loc[~df_demand001['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_demand001['OPTION_CODE_NEW']
    # df_demand001 = df_demand001.drop(columns=['OPTION_CODE_NEW','UPBY_NEW'])


    # df_outbound_update_qq2 = pd.DataFrame(list_Outbound_loop_update_qq2)
    # df_outbound_update_qq2.columns = [ND_.PLANID,ND_.SALESORDERID,ND_.SOLINENUM,'QTYPROMISED_NEW','UPBY_NEW','OPTION_CODE_NEW']
    # df_outbound_update_qq2 = df_outbound_update_qq2.astype({ND_.SOLINENUM:'int32'})


    # df_demand001a = df_demand001.copy(deep=True)
    # df_demand001a = df_demand001a.merge(df_outbound_update_qq2, how='left', on=[ND_.PLANID,ND_.SALESORDERID,ND_.SOLINENUM])

    # df_demand001a.loc[~df_demand001a['QTYPROMISED_NEW'].isna(),ND_.QTYPROMISED] = df_demand001a['QTYPROMISED_NEW']
    # df_demand001a.loc[~df_demand001a['OPTION_CODE_NEW'].isna(),ND_.OPTION_CODE] = df_demand001a['OPTION_CODE_NEW']
    # df_demand001a.loc[~df_demand001a['UPBY_NEW'].isna(),'UPBY'] = df_demand001a['UPBY_NEW']
    # df_demand001a = df_demand001a.drop(columns=['QTYPROMISED_NEW','OPTION_CODE_NEW','UPBY_NEW'])

    # df_demand001a = df_demand001a.drop(columns=['OPTION_CODE2'])

    # df_outbound_insert_qq = pd.DataFrame(list_Outbound_loop_insert_qq)
    # df_outbound_insert_qq.columns = df_Outbound1.columns
    # df_demand002 = pd.concat([df_demand001a, df_outbound_insert_qq], axis=0, ignore_index=True)

    # logger.Step(7, 'CH.STOCK으로 잘린 것들 UNF_ORD 수량만큼 살려준다 2016.06.08 End')
    #endregion logger.Step(7, 'CH.STOCK으로 잘린 것들 UNF_ORD 수량만큼 살려준다 2016.06.08')

    #region logger.Step(8, 'Demend Priority 설정')
    logger.Step(8, 'Demend Priority 설정 Start')

    df_demand002 = df_demand002.loc[df_demand002[ND_.PLANID] == V_PLANID]
    df_demand002['OPTION_CODE2'] = df_demand002[ND_.OPTION_CODE]
    df_demand002.loc[df_demand002['OPTION_CODE2'].isna() ,'OPTION_CODE2'] = '0'
    df_demand002.loc[df_demand002['OPTION_CODE2']=='','OPTION_CODE2'] = '0'
    df_demand002['OPTION_CODE2'] = df_demand002['OPTION_CODE2'].astype('int32')

    df_demand002['OPTION_CODE2'] = df_demand002['OPTION_CODE2'].mod(2)

    df_priority = df_demand002.copy(deep=True)


    df_priority = df_priority.loc[df_priority['OPTION_CODE2'] == 1]

    start_pos, next_pos, total_length = find_priority_position('G_R001::1', 'DEMANDTYPERANK')
    digit = next_pos - start_pos
    df_priority['DEMANDPRIORITY_NEW'] = (
        df_priority[ND_.DEMANDPRIORITY].str[:start_pos]
        + '9'.zfill(digit)
        + df_priority[ND_.DEMANDPRIORITY].str[next_pos:]
    )
    df_priority = df_priority[[ND_.PLANID, ND_.SALESORDERID, ND_.SOLINENUM, 'OPTION_CODE2','DEMANDPRIORITY_NEW']]
    df_priority[ND_.PLANID] = df_priority[ND_.PLANID].astype('str')
    df_priority[ND_.SALESORDERID] = df_priority[ND_.SALESORDERID].astype('str')
    df_priority[ND_.SOLINENUM] = df_priority[ND_.SOLINENUM].astype('int32')
    df_priority['OPTION_CODE2'] = df_priority['OPTION_CODE2'].astype('int64')
    df_priority['DEMANDPRIORITY_NEW'] = df_priority['DEMANDPRIORITY_NEW'].astype('str')

    df_demand002[ND_.PLANID] = df_demand002[ND_.PLANID].astype('str')
    df_demand002[ND_.SALESORDERID] = df_demand002[ND_.SALESORDERID].astype('str')
    df_demand002[ND_.SOLINENUM] = df_demand002[ND_.SOLINENUM].astype('float').astype('int32')
    df_demand002['OPTION_CODE2'] = df_demand002['OPTION_CODE2'].astype('float').astype('int32')
    df_demand002[ND_.DEMANDPRIORITY] = df_demand002[ND_.DEMANDPRIORITY].astype('str')

    df_demand003 = df_demand002.merge(df_priority, how='left',
                                    on=[ND_.PLANID, 'OPTION_CODE2', ND_.SALESORDERID, ND_.SOLINENUM,])

    df_demand003.loc[~df_demand003['DEMANDPRIORITY_NEW'].isna(),ND_.DEMANDPRIORITY] = df_demand003['DEMANDPRIORITY_NEW']
    df_demand003 = df_demand003.drop(columns=['DEMANDPRIORITY_NEW'])

    df_outbound = df_demand003.astype({ND_.QTYPROMISED:'int32', ND_.PROMISEDDELDATE:'datetime64[ns]'})

    df_chstock_step8 = df_outbound[ND_.LIST_COLUMN]

    logger.Step(8, 'Demend Priority 설정 End')
    #endregion logger.Step(8, 'Demend Priority 설정')

    #region logger.Step(9, 'Channel Short 재분류')
    logger.Step(9, 'Channel Short 재분류 Start')
    
    ## 재분류의 기준이 되는 df_short_reason_dp_guide 전처리

    # 우선순위 정렬 및 중복 제거
    df_short_reason_dp_guide = df_short_reason_dp_guide.sort_values(by=[
        SRDPG.SALESID, SRDPG.ITEM, SRDPG.SITEID, SRDPG.WEEK, SRDPG.DEMANDID
    ])
    df_short_reason_dp_guide = df_short_reason_dp_guide.drop_duplicates(subset=SRDPG.DEMANDID)

    # Promised Del Date 기준으로 week 변경
    df_short_reason_dp_guide[SRDPG.WEEK] = pd.to_datetime(df_short_reason_dp_guide[SRDPG.WEEK] + '7', format='%G%V%u')

    # 빠른 계산을 위해 numpy array로 변환
    nparr_reason_dp_guide_cause = df_short_reason_dp_guide.to_numpy()

    ## demand에 대한 전처리
    # option code의 나머지가 1이고, qty > 0인 것이 재분류 대상
    series_classify_condition = (
        (df_chstock_step8[ND_.OPTION_CODE].astype(float)%2==1)
        & (df_chstock_step8[ND_.QTYPROMISED]>0)
    )

    # 재분류 대상과 비대상 분리
    df_classification_target = df_chstock_step8[series_classify_condition].sort_values(by=[
        ND_.SALESID, ND_.ITEMID, ND_.SITEID, ND_.PROMISEDDELDATE, ND_.DEMANDPRIORITY,
        ND_.QTYPROMISED, ND_.SALESORDERID, ND_.SOLINENUM,
    ])
    df_classification_non_target = df_chstock_step8.loc[~series_classify_condition]

    # 재분류 과정에서 분리된 demand를 담을 list
    list_splited = []

    # 재분류 할 demand numpy array 변환
    nparr_classification_target = df_classification_target.to_numpy()

    ## 재분류
    # 탈출 조건 설정을 위해 각각의 길이를 구하고, idx를 초기화
    len_shreason_dp_guideause = len(nparr_reason_dp_guide_cause)
    len_classification_target = len(nparr_classification_target)
    short_reason_dp_guide_idx = 0
    classification_target_idx = 0

    while short_reason_dp_guide_idx < len_shreason_dp_guideause and classification_target_idx < len_classification_target:
        ct_row = nparr_classification_target[classification_target_idx]
        sc_row = nparr_reason_dp_guide_cause[short_reason_dp_guide_idx]

        # 동일한 sales, item, site, week가 아닐 경우 idx를 넘긴다.
        if ct_row[ND_.SALESID_IDX] < sc_row[SRDPG.SALESID_IDX]:
            classification_target_idx += 1
            continue
        elif ct_row[ND_.SALESID_IDX] > sc_row[SRDPG.SALESID_IDX]:
            short_reason_dp_guide_idx += 1
            continue
        else:
            if ct_row[ND_.ITEMID_IDX] < sc_row[SRDPG.ITEM_IDX]:
                classification_target_idx += 1
                continue
            elif ct_row[ND_.ITEMID_IDX] > sc_row[SRDPG.ITEM_IDX]:
                short_reason_dp_guide_idx += 1
                continue
            else:
                if ct_row[ND_.SITEID_IDX] < sc_row[SRDPG.SITEID_IDX]:
                    classification_target_idx += 1
                    continue
                elif ct_row[ND_.SITEID_IDX] > sc_row[SRDPG.SITEID_IDX]:
                    short_reason_dp_guide_idx += 1
                    continue
                else:
                    if ct_row[ND_.PROMISEDDELDATE_IDX] < sc_row[SRDPG.WEEK_IDX]:
                        classification_target_idx += 1
                        continue
                    elif ct_row[ND_.PROMISEDDELDATE_IDX] > sc_row[SRDPG.WEEK_IDX]:
                        short_reason_dp_guide_idx += 1
                        continue
        
        # 동일한 sales, item, site, week일 경우 재분류를 진행한다.
        if ct_row[ND_.QTYPROMISED_IDX] <= sc_row[SRDPG.SHORTQTY_IDX]:
            ct_row[ND_.REASONCODE_IDX] = sc_row[SRDPG.DEMANDID_IDX]
            sc_row[SRDPG.SHORTQTY_IDX] -= ct_row[ND_.QTYPROMISED_IDX]
            classification_target_idx += 1
            if sc_row[SRDPG.SHORTQTY_IDX] == 0:
                short_reason_dp_guide_idx += 1
        else:
            new_row = ct_row.copy()
            new_row[ND_.REASONCODE_IDX] = sc_row[SRDPG.DEMANDID_IDX]
            new_row[ND_.QTYPROMISED_IDX] = sc_row[SRDPG.SHORTQTY_IDX]
            # 새로운 겹치지 않는 solinenum 생성
            set_solinenum = dict_salesorder[ct_row[ND_.SALESORDERID_IDX]]
            new_solinenum = ct_row[ND_.SOLINENUM_IDX] + 1
            while new_solinenum in set_solinenum:
                new_solinenum += 1
            new_row[ND_.SOLINENUM_IDX] = new_solinenum
            set_solinenum.add(new_solinenum)
            list_splited.append(new_row)
            ct_row[ND_.QTYPROMISED_IDX] -= sc_row[SRDPG.SHORTQTY_IDX]
            sc_row[SRDPG.SHORTQTY_IDX] = 0
            short_reason_dp_guide_idx += 1

    ## 재분류 된 demand를 다시 합친다.
    if len(list_splited) > 0:
        nparr_classified = np.concatenate([nparr_classification_target, np.array(list_splited)], axis=0)
    else:
        nparr_classified = nparr_classification_target
    df_classified = pd.DataFrame(data=nparr_classified, columns=ND_.LIST_COLUMN)
    df_classified = df_classified.astype({
        ND_.SOLINENUM: 'int32',
        ND_.QTYPROMISED: 'int32',
    })
    df_chstock_out = pd.concat([df_classification_non_target, df_classified], ignore_index=True)

    logger.Step(9, 'Channel Short 재분류 End')
    #endregion logger.Step(9, 'Channel Short 재분류')

    return df_chstock_out
