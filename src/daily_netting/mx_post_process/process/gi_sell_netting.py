from collections import OrderedDict

import os, sys

import pandas as pd
import numpy as np
import decimal as dc
import time

from daily_netting.mx_post_process.constant.ods_constant import NettedDemandD as ND_                 # EXP_SOPROMISE
from daily_netting.mx_post_process.constant.dim_constant import Location as MST_SITE
from daily_netting.mx_post_process.constant.mx_constant  import SalesResult as GUI_SALESRESULTSMM
from daily_netting.mx_post_process.constant.mx_constant  import CustomModelMap as MTA_CUSTOMMODELMAP
from daily_netting.mx_post_process.constant.mx_constant  import ESItemSite as MVES_ITEMSITE
from daily_netting.mx_post_process.constant.mx_constant  import AvailableResourceD as MST_INVENTORY_FNE
from daily_netting.mx_post_process.constant.mx_constant  import MXItemSellerMap as V_MTA_SELLERMAP
from daily_netting.mx_post_process.utils.NSCMCommon import G_Logger
from daily_netting.mx_post_process.utils.common import validate, convert_to_iyyyiw

class GiNetSellConstant:
    # 'PLANID', 'SITEID', 'SALESID', 'ITEM', 'WEEK', 'RTF', 'GI', 'EXCEPTQ', 'AVAIL', 'DP', 'CUREXCEPT', 'CURAVAIL', 'GIDP', 'MPDP', 'REMAINGIDP', 'UPDATEYN', 'RK'
    PLANID_IDX      =  0
    SITEID_IDX      =  1
    SALESID_IDX     =  2
    ITEM_IDX        =  3
    WEEK_IDX        =  4
    RTF_IDX         =  5
    GI_IDX          =  6
    EXCEPTQ_IDX     =  7
    AVAIL_IDX       =  8
    DP_IDX          =  9
    CUREXCEPT_IDX   = 10
    CURAVAIL_IDX    = 11
    GIDP_IDX        = 12
    MPDP_IDX        = 13
    REMAINGIDP_IDX  = 14
    UPDATEYN_IDX    = 15
    RK_IDX          = 16

class SiteSellConstant :
    SITEID_IDX = 0
    SALESID_IDX = 1
    ITEM_IDX = 2
    WEEK_IDX = 3
    GIDP_IDX = 4

class RConstant : 
    SITEID_IDX      =  0
    SALESID_IDX     =  1
    ITEM_IDX        =  2
    WEEK_IDX	    =  3
    RTF_IDX         =  4
    GI_IDX          =  5
    DP_IDX          =  6
    GIDP_IDX        =  7
    REMAINGIDP_IDX  =  8
    MPDP_IDX        =  9

'''
전역변수정의
'''
V_PLANID = None
V_PREPLANID = None
V_PLANWEEK = None
V_WEEK1 = None
V_WEEK4 = None
V_PREYEAR = None
V_PREYWEEK = None

V_EXCEPTQ = 0
V_SALESID = ''

find_priority_position = None

'''
Alias 정의
'''
DF = pd.DataFrame
NA = np.array
GINETSELL = GiNetSellConstant
SITESELL = SiteSellConstant
R     = RConstant
LOGGER = G_Logger

def set_gi_sell_env(accessor: object) -> None:
    global V_PLANID, V_PREPLANID, V_PLANWEEK, V_WEEK1, V_WEEK4, V_PREYEAR, \
        V_PREYWEEK, find_priority_position
    
    # plan 설정
    V_PLANID = accessor.plan_id
    V_PLANWEEK = accessor.plan_week
    V_WEEK1 = accessor.plan_week

    if accessor.plan_type == 'VPLAN':
        V_PREPLANID = accessor.start_date.strftime('%G%V')
        V_WEEK4 = (accessor.start_date + pd.Timedelta(weeks=4)).strftime('%G%V')
    else:
        V_PREPLANID = (accessor.start_date - pd.Timedelta(weeks=1)).strftime('%G%V')
        V_WEEK4 = (accessor.start_date + pd.Timedelta(weeks=3)).strftime('%G%V')
    
    V_PREYEAR = V_PREPLANID[0:4]
    V_PREYWEEK = V_PREPLANID[4:6]

    # 자리수 확인 함수
    find_priority_position = accessor.find_priority_position

def do_gi_sell_netting(df_Inbound:DF, df_in_Netting_Seller_Map:DF, df_in_Netting_Available_Resource_ODS_W:DF,
                                df_in_Netting_ES_Item_Site_ODS_W:DF, df_in_Netting_Custom_Model_Map_ODS_W:DF, df_Sales_Result_Smm:DF,
                                df_mst_site:DF, df_MST_GINETING_SALES:DF, logger:LOGGER) -> DF:
    global V_EXCEPTQ, V_SALESID
    
    logger.Note('Start GI Netting Sell', 20)

    df_Outbound = df_Inbound.copy(deep=True)

    #region logger.Step(1, 'Data Summary')
    logger.Step(1, 'Data Summary Start')

    # mst_ginet_sell 생성
    '''
    mst_ginet 구성
    '''
    df_Outbound = df_Inbound.copy(deep=True)

    df_Sales_Result_Smm[GUI_SALESRESULTSMM.QTY_RTF].fillna(0).astype('int32')
    df_Sales_Result_Smm[GUI_SALESRESULTSMM.QTY_GI].fillna(0).astype('int32')

    #AND    EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND SITEID = A.SITEID)
    df_MST_GINNETING_01 = df_Sales_Result_Smm.loc[
            (df_Sales_Result_Smm[GUI_SALESRESULTSMM.ITEM] + df_Sales_Result_Smm[GUI_SALESRESULTSMM.SITEID]).isin(
                df_in_Netting_Seller_Map[V_MTA_SELLERMAP.ITEM]+df_in_Netting_Seller_Map[V_MTA_SELLERMAP.SITEID]
            )
        ]


    #AND    NOT EXISTS (SELECT ''X'' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = ''Y'')
    df_MST_GINNETING_01 = df_MST_GINNETING_01.loc[
            ~(df_MST_GINNETING_01[GUI_SALESRESULTSMM.ITEM]+'Y').isin(
                    df_in_Netting_Custom_Model_Map_ODS_W[MTA_CUSTOMMODELMAP.CUSTOMITEM]+
                            df_in_Netting_Custom_Model_Map_ODS_W[MTA_CUSTOMMODELMAP.ISVALID])
        ]


    df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_RTF] = df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_RTF].fillna(0).astype('int32')
    df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_GI] = df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_GI].fillna(0).astype('int32')

    #GROUP BY ITEM, SITEID--, AP2ID
    #HAVING SUM(DECODE(CATEGORY, ''02RTF'', WEEK'||V_PREYWEEK||',0)) < SUM(DECODE(CATEGORY, ''30GI'' , WEEK'||V_PREYWEEK||',0))
    V_PRE_PLAN = V_PREYEAR + V_PREYWEEK
    df_MST_GINNETING_01 = df_MST_GINNETING_01.loc[df_MST_GINNETING_01[GUI_SALESRESULTSMM.WEEK]==V_PRE_PLAN]


    df_MST_GINNETING_01 = df_MST_GINNETING_01[[GUI_SALESRESULTSMM.SITEID,GUI_SALESRESULTSMM.AP2ID,GUI_SALESRESULTSMM.ITEM,GUI_SALESRESULTSMM.QTY_RTF,GUI_SALESRESULTSMM.QTY_GI]]
    df_MST_GINNETING_01 = df_MST_GINNETING_01.groupby(by=[GUI_SALESRESULTSMM.SITEID,GUI_SALESRESULTSMM.AP2ID,GUI_SALESRESULTSMM.ITEM,])[[GUI_SALESRESULTSMM.QTY_RTF,GUI_SALESRESULTSMM.QTY_GI]].sum()
    df_MST_GINNETING_01['EXCEPT'] = df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_GI] - df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_RTF] 
    df_MST_GINNETING_01[GUI_SALESRESULTSMM.WEEK] = V_PLANWEEK
    df_EXCEPT = df_MST_GINNETING_01.loc[df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_RTF] < df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_GI]].reset_index()


    # -- 1. 가용량 (INVENTORY + INTRANSIT + DISTRIBUTIONORDERS)
    # AND    A.WEEK BETWEEN '||V_WEEK1||' AND '||V_WEEK4||'
    df_MST_GINNETING_02 = df_in_Netting_Available_Resource_ODS_W.loc[df_in_Netting_Available_Resource_ODS_W[MST_INVENTORY_FNE.WEEK].between(V_WEEK1, V_WEEK4)]

    # AND    A.SALESID IS NOT NULL
    df_MST_GINNETING_02 = df_MST_GINNETING_02.loc[~df_MST_GINNETING_02[MST_INVENTORY_FNE.SALESID].isna()]

    # AND    EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
    df_MST_GINNETING_02 = df_MST_GINNETING_02[
            (df_MST_GINNETING_02[MST_INVENTORY_FNE.ITEM] + df_MST_GINNETING_02[MST_INVENTORY_FNE.SITEID]).isin(
                df_in_Netting_Seller_Map[V_MTA_SELLERMAP.ITEM]+df_in_Netting_Seller_Map[V_MTA_SELLERMAP.SITEID]
            )
        ]

    df_MST_GINNETING_02[MST_INVENTORY_FNE.QTY] = df_MST_GINNETING_02[MST_INVENTORY_FNE.QTY].fillna(0).astype('int32')

    df_MST_GINNETING_02.rename(columns={
        MST_INVENTORY_FNE.QTY: 'AVAILQTY',
    }, inplace=True)

    df_MST_GINNETING_02 = df_MST_GINNETING_02[[MST_INVENTORY_FNE.ITEM,MST_INVENTORY_FNE.SITEID,MST_INVENTORY_FNE.SALESID,MST_INVENTORY_FNE.WEEK,'AVAILQTY']]
    df_MST_GINNETING_02['DPQTY'] = 0

    # -- 2. DEMAND
    df_MST_GINNETING_03 = df_Inbound[[ND_.ITEMID, ND_.SITEID, ND_.AP2ID, ND_.PROMISEDDELDATE, ND_.QTYPROMISED]].copy()
    df_MST_GINNETING_03.insert(4,'QTY',0)
    df_MST_GINNETING_03['QTY'] = df_MST_GINNETING_03['QTY'].fillna(0).astype('int32')
    df_MST_GINNETING_03[ND_.QTYPROMISED] = df_MST_GINNETING_03[ND_.QTYPROMISED].astype('float').astype('int32')
    # AND     QTYPROMISED > 0
    df_MST_GINNETING_03 = df_MST_GINNETING_03.loc[df_MST_GINNETING_03[ND_.QTYPROMISED] > 0]

    # AND     TO_CHAR(PROMISEDDELDATE, ''IYYYIW'') BETWEEN '||V_WEEK1||' AND '||V_WEEK4||'
    df_MST_GINNETING_03[ND_.PROMISEDDELDATE] = pd.to_datetime(df_MST_GINNETING_03[ND_.PROMISEDDELDATE]).dt.strftime('%G%V')
    df_MST_GINNETING_03 = df_MST_GINNETING_03.loc[df_MST_GINNETING_03[ND_.PROMISEDDELDATE].between(V_WEEK1, V_WEEK4)]

    # AND     EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
    df_DEMAND = df_MST_GINNETING_03.loc[
        (df_MST_GINNETING_03[ND_.ITEMID] + df_MST_GINNETING_03[ND_.SITEID]).isin(
            df_in_Netting_Seller_Map[V_MTA_SELLERMAP.ITEM]+df_in_Netting_Seller_Map[V_MTA_SELLERMAP.SITEID]
        )
    ]

    df_MST_GINNETING_02 = df_MST_GINNETING_02.rename(columns={
        MST_INVENTORY_FNE.ITEM:ND_.ITEMID,
        MST_INVENTORY_FNE.SITEID:ND_.SITEID,
        MST_INVENTORY_FNE.SALESID:ND_.SALESID,
    })
    df_DEMAND = df_DEMAND.rename(columns={
        ND_.AP2ID: ND_.SALESID,
        ND_.PROMISEDDELDATE:MST_INVENTORY_FNE.WEEK,
        'QTY': 'AVAILQTY',
        ND_.QTYPROMISED: 'DPQTY',
    })

    #-- 1. 가용량 (INVENTORY + INTRANSIT + DISTRIBUTIONORDERS) 
    # union all 
    #-- 2. DEMAND
    df_MST_GINNETING_04 = pd.concat([df_MST_GINNETING_02, df_DEMAND], axis=0, ignore_index=True)

    #groupby
    df_AVAIL = df_MST_GINNETING_04.groupby(by=[ND_.ITEMID, ND_.SITEID, ND_.SALESID, MST_INVENTORY_FNE.WEEK]).sum(numeric_only=True).reset_index()
    df_AVAIL = df_AVAIL.rename(columns={
        MST_INVENTORY_FNE.WEEK:'WEEK'
    })

    df_EXCEPT.rename(columns={
        GUI_SALESRESULTSMM.AP2ID:GUI_SALESRESULTSMM.SALESID,
    }, inplace=True)

    #df_EXCEPT, df_AVAIL inner join
    dfMST_GINETTING_SELL = pd.merge(df_EXCEPT, df_AVAIL, how='inner',
                                left_on=[GUI_SALESRESULTSMM.ITEM, GUI_SALESRESULTSMM.SITEID, GUI_SALESRESULTSMM.SALESID],
                                right_on=[ND_.ITEMID, ND_.SITEID, ND_.SALESID]).reset_index()
    dfMST_GINETTING_SELL = dfMST_GINETTING_SELL.sort_values(by=[ND_.ITEMID, ND_.SITEID, 'WEEK',]).reset_index()
    dfMST_GINETTING_SELL = dfMST_GINETTING_SELL[[ND_.SITEID, ND_.SALESID,ND_.ITEMID,'WEEK',GUI_SALESRESULTSMM.QTY_RTF,GUI_SALESRESULTSMM.QTY_GI,'EXCEPT','AVAILQTY','DPQTY']]
    dfMST_GINETTING_SELL.insert(0, ND_.PLANID, V_PLANID)
    dfMST_GINETTING_SELL.columns = [ND_.PLANID, ND_.SITEID, ND_.SALESID, ND_.ITEMID, 'WEEK', GUI_SALESRESULTSMM.QTY_RTF,GUI_SALESRESULTSMM.QTY_GI, 'EXCEPTQ', 'AVAIL', 'DP']
    dfMST_GINETTING_SELL['WEEK'] = dfMST_GINETTING_SELL['WEEK'].fillna(0).astype('int32')
    dfMST_GINETTING_SELL['RK'] = dfMST_GINETTING_SELL.groupby(by=[ND_.ITEMID, ND_.SITEID, ND_.SALESID])['WEEK'].rank(method='min')
    dfMST_GINETTING_SELL.insert(10, 'CUREXCEPT', dfMST_GINETTING_SELL[['EXCEPTQ']])
    dfMST_GINETTING_SELL.insert(11, 'CURAVAIL', dfMST_GINETTING_SELL[['AVAIL']])
    dfMST_GINETTING_SELL.insert(12, 'GIDP', 0)
    dfMST_GINETTING_SELL.insert(13, 'MPDP', 0)
    dfMST_GINETTING_SELL.insert(14, 'REMAINGIDP', np.nan)
    dfMST_GINETTING_SELL.insert(15, 'UPDATEYN','')
    dfMST_GINETTING_SELL['WEEK'] = dfMST_GINETTING_SELL['WEEK'].fillna(0).astype('int32')
    dfMST_GINETTING_SELL = dfMST_GINETTING_SELL.sort_values(by=[ND_.SITEID, ND_.SALESID, ND_.ITEMID, 'WEEK'])

    # mst_ginet_sales은 외부에서 전달받은 것 사용 -> df_MST_GINETING_SALES

    logger.Step(1, 'Data Summary End')
    #endregion logger.Step(1, 'Data Summary')

    #region logger.Step(2, 'AVAIL QTY ROLLING')
    logger.Step(2, 'AVAIL QTY ROLLING Start')

    ''' 
    2.AVAIL QTY ROLLING
    '''

    M_AVAIL = 0

        # PLANID_IDX      =  0
        # SITEID_IDX      =  1
        # SALESID_IDX     =  2
        # ITEM_IDX        =  3
        # WEEK_IDX        =  4
        # RTF_IDX         =  5
        # GI_IDX          =  6
        # EXCEPTQ_IDX     =  7
        # AVAIL_IDX       =  8
        # DP_IDX          =  9
        # CUREXCEPT_IDX   = 10
        # CURAVAIL_IDX    = 11
        # GIDP_IDX        = 12
        # MPDP_IDX        = 13
        # REMAINGIDP_IDX  = 14
        # UPDATEYN_IDX    = 15
        # RK_IDX          = 16


    dt = np.dtype([("S1", 'object'), 
        ("S2", 'object'),
        ("S3", 'object'),
        ("S4", 'object'), 
        ("S5", 'object'), 
        ("i1", np.int32), 
        ("i2", np.int32), 
        ("i3", np.int32),
        ("i4", np.int32),
        ("i5", np.int32),
        ("i6", np.int32),
        ("i7", np.int32),
        ("i8", np.int32),
        ("i9", np.int32), 
        ("i10", np.float64),
        ("S6", 'object'),
        ("i11", np.int32)])

    nparr_ginet = np.array([tuple(v) for v in dfMST_GINETTING_SELL.values.tolist()], dtype=dt)

    for row in nparr_ginet:
        if row[GINETSELL.RK_IDX] == 1: # IF X.RK = 1 THEN
            if row[GINETSELL.AVAIL_IDX] > row[GINETSELL.DP_IDX]:     # IF X.AVAIL > X.DP THEN
                M_AVAIL = row[GINETSELL.AVAIL_IDX] - row[GINETSELL.DP_IDX] # M_AVAIL := X.AVAIL - X.DP;
            else:
                M_AVAIL = 0
        #--! 첫 행이 아니면 !--
        else:
            #--! CURAVAIL = AVAIL + M_AVAIL !--
            row[GINETSELL.CURAVAIL_IDX] = row[GINETSELL.AVAIL_IDX] + M_AVAIL
        
            if (row[GINETSELL.AVAIL_IDX] + M_AVAIL) > row[GINETSELL.DP_IDX]:
                M_AVAIL = (row[GINETSELL.AVAIL_IDX] + M_AVAIL) - row[GINETSELL.DP_IDX]
            else:
                M_AVAIL = 0

    # return nparr_ginet

    logger.Step(2, 'AVAIL QTY ROLLING End')
    #endregion logger.Step(2, 'AVAIL QTY ROLLING')

    #region logger.Step(3, 'GINETTING OPEN')
    logger.Step(3, 'GINETTING OPEN Start')

    ''' 
    3.GINETTING OPEN
    '''
    M_EXCEPT = 0

    #'PLANID', 'SITEID', 'SALESID', 'ITEM', 'WEEK', 'RTF', 'GI', 'EXCEPTQ', 'AVAIL', 'DP', 'CUREXCEPT', 'CURAVAIL', 'GIDP', 'MPDP', 'REMAINGIDP', 'UPDATEYN'--> 4.site OPEN 시 GIDP값을 가져올때 사용하는 구분자, 'RK'
    for row in nparr_ginet:
        if row[GINETSELL.RK_IDX] == 1: # IF X.RK = 1 THEN
            if (row[GINETSELL.CURAVAIL_IDX] - row[GINETSELL.DP_IDX]) >= 0: # IF X.CURAVAIL - X.DP >=0 THEN
                #--! 전량 MPDP, GIDP 0 !--
                row[GINETSELL.GIDP_IDX] = 0
                row[GINETSELL.MPDP_IDX] = row[GINETSELL.DP_IDX]

                M_EXCEPT = row[GINETSELL.EXCEPTQ_IDX]
            else: 
                #--! DP 가 남았을 경우 !--
                if (row[GINETSELL.EXCEPTQ_IDX] >= (row[GINETSELL.DP_IDX] - row[GINETSELL.CURAVAIL_IDX])):
                    #--! 가용량이 없는 만큼은 GIDP, 가용량이 있는 만큼은 MPDP !--
                    row[GINETSELL.GIDP_IDX] = (row[GINETSELL.DP_IDX] - row[GINETSELL.CURAVAIL_IDX])
                    row[GINETSELL.MPDP_IDX] = row[GINETSELL.DP_IDX] - (row[GINETSELL.DP_IDX] - row[GINETSELL.CURAVAIL_IDX])

                    M_EXCEPT = row[GINETSELL.EXCEPTQ_IDX] - (row[GINETSELL.DP_IDX] - row[GINETSELL.CURAVAIL_IDX])
                
                #--! X.EXCEPT< X.DP - X.CURAVAIL !--
                else:
                    #--!  GIDP = X.EXCEPTQ , MPDP =  DP - (X.EXCEPTQ) !--
                    row[GINETSELL.GIDP_IDX] = row[GINETSELL.EXCEPTQ_IDX]
                    row[GINETSELL.MPDP_IDX] = row[GINETSELL.DP_IDX] - row[GINETSELL.EXCEPTQ_IDX]

                    M_EXCEPT = 0
        #--! 두번째 ROW부터 !--
        else:
            #--! AVAIL 수량 > DP 일 경우 !--
            if (row[GINETSELL.CURAVAIL_IDX] - row[GINETSELL.DP_IDX]) >= 0: # IF X.CURAVAIL - X.DP >=0 THEN
                #--! 전량 MPDP, GIDP 0 !--
                row[GINETSELL.GIDP_IDX] = 0
                row[GINETSELL.MPDP_IDX] = row[GINETSELL.DP_IDX]
                row[GINETSELL.CUREXCEPT_IDX] = M_EXCEPT

                M_EXCEPT = M_EXCEPT
            else:
                #--! CUREXCEPT = M_EXCEPT !--
                row[GINETSELL.CUREXCEPT_IDX] = M_EXCEPT

                #--! DP 가 남았을 경우 !--
                if(M_EXCEPT >= (row[GINETSELL.DP_IDX] - row[GINETSELL.CURAVAIL_IDX])):

                    # --!  GIDP = X.DP - X.CURAVAIL , DP - (X.DP - X.CURAVAIL) !--
                    row[GINETSELL.GIDP_IDX] = row[GINETSELL.DP_IDX] - row[GINETSELL.CURAVAIL_IDX]
                    row[GINETSELL.MPDP_IDX] = row[GINETSELL.DP_IDX] - (row[GINETSELL.DP_IDX] - row[GINETSELL.CURAVAIL_IDX])

                    M_EXCEPT = M_EXCEPT - (row[GINETSELL.DP_IDX] - row[GINETSELL.CURAVAIL_IDX])
                else:
                    row[GINETSELL.GIDP_IDX] = M_EXCEPT
                    row[GINETSELL.MPDP_IDX] = row[GINETSELL.DP_IDX] - M_EXCEPT

                    M_EXCEPT = 0

    # return nparr_ginet


    logger.Step(3, 'GINETTING OPEN End')
    #endregion logger.Step(3, 'GINETTING OPEN')

    #region logger.Step(4, 'SALES OPEN')
    logger.Step(4, 'SALES OPEN Start')

    ''' 
    4.SALES OPEN
    '''
    for col in ND_.LIST_COLUMN:
        if col not in df_Inbound.columns:
            df_Inbound[col] = ''

    df_Inbound = df_Inbound[ND_.LIST_COLUMN]


    df_ginet_sales_update = pd.DataFrame()
    df_ginet_sales_merge = pd.DataFrame()

    # 형전환
    df_Outbound[ND_.QTYPROMISED] = df_Outbound[ND_.QTYPROMISED].astype('float').astype('int32')
    df_Outbound[ND_.SOLINENUM] = df_Outbound[ND_.SOLINENUM].fillna(0).astype('int32')

    V_REMAINGIDP = 0
    V_GIDP = 0

    dfMST_GINETTING_SELL = pd.DataFrame(nparr_ginet)
    dfMST_GINETTING_SELL.columns = ['PLANID', 'SITEID', 'SALESID', 'ITEM', 'WEEK', 'RTF', 'GI', 'EXCEPTQ', 'AVAIL', 'DP', 'CUREXCEPT', 'CURAVAIL', 'GIDP', 'MPDP', 'REMAINGIDP','UPDATEYN', 'RK']
    dfMST_GINETTING_SELL = dfMST_GINETTING_SELL.astype({'RTF':'int32', 'GI':'int32', 'EXCEPTQ':'int32', 'AVAIL':'int32', 'DP':'int32', 'CUREXCEPT':'int32', 'CURAVAIL':'int32', 'GIDP':'int32', 'MPDP':'int32', 'REMAINGIDP':'float64', 'RK':'int32'})

    dfMST_GINETTING_SELL_SITE = pd.DataFrame(nparr_ginet)
    dfMST_GINETTING_SELL_SITE.columns = ['PLANID', 'SITEID', 'SALESID', 'ITEM', 'WEEK', 'RTF', 'GI', 'EXCEPTQ', 'AVAIL', 'DP', 'CUREXCEPT', 'CURAVAIL', 'GIDP', 'MPDP', 'REMAINGIDP', 'UPDATEYN', 'RK']
    dfMST_GINETTING_SELL_SITE = dfMST_GINETTING_SELL_SITE.astype({'RTF':'int32', 'GI':'int32', 'EXCEPTQ':'int32', 'AVAIL':'int32', 'DP':'int32', 'CUREXCEPT':'int32', 'CURAVAIL':'int32', 'GIDP':'int32', 'MPDP':'int32', 'REMAINGIDP':'float64', 'RK':'int32'})

    dfMST_GINETTING_SELL_SITE = dfMST_GINETTING_SELL_SITE.loc[dfMST_GINETTING_SELL_SITE['PLANID'] == V_PLANID]
    dfMST_GINETTING_SELL_SITE = dfMST_GINETTING_SELL_SITE.loc[dfMST_GINETTING_SELL_SITE['GIDP'] > 0]

    df_SITE = dfMST_GINETTING_SELL_SITE.sort_values(by=['SITEID', 'SALESID', 'ITEM', 'WEEK',]).reset_index()
    df_SITE = df_SITE[['SITEID', 'SALESID', 'ITEM', 'WEEK', 'GIDP']]
    df_SITE.columns = ['SITEID', 'SALESID', 'ITEM', 'WEEK', 'GIDP'] 
    df_SITE['WEEK'] = df_SITE['WEEK'].fillna(0).astype('int32')

    df_MST_GINETING_SALES = df_MST_GINETING_SALES.astype({'RTF':'int32', 'GI':'int32', 'EXCEPTQ':'int32', 'SUMEXCEPTQ':'int32', 'RNK':'int32', 'DP':'int32', 'CUREXCEPT':'int32'})

    #--기여도 높은 순으로 NOW DP를 발췌
    df_DP1 = df_Inbound.copy(deep=True)

    # 형전환
    df_DP1[ND_.QTYPROMISED] = df_DP1[ND_.QTYPROMISED].astype('float').astype('int32')
    df_DP1[ND_.SOLINENUM] = df_DP1[ND_.SOLINENUM].fillna(0).astype('int32')
    df_MST_GINETING_SALES['EXCEPTQ'] = df_MST_GINETING_SALES['EXCEPTQ'].fillna(0).astype('int32')
    df_MST_GINETING_SALES['CUREXCEPT'] = df_MST_GINETING_SALES['CUREXCEPT'].fillna(0).astype('int32')
    df_MST_GINETING_SALES['RNK'] = df_MST_GINETING_SALES['RNK'].fillna(0).astype('int32')

    # # WHERE  A.PLANID = V_PLANID
    # # AND    QTYPROMISED>0
    # # AND    A.SOLINENUM < 200
    df_DP1 = df_DP1.loc[(df_DP1[ND_.PLANID] == V_PLANID) & (df_DP1[ND_.QTYPROMISED] > 0) & (df_DP1[ND_.SOLINENUM] < 200) ]


    df_DP1['PROMISEDDELDATE2'] = pd.to_datetime(df_DP1[ND_.PROMISEDDELDATE]).dt.strftime('%G%V')
    df_DP1['PROMISEDDELDATE2'] = df_DP1['PROMISEDDELDATE2'].fillna(0).astype('int32')

    df_DP2 = df_MST_GINETING_SALES.copy(deep=True)


    # #     # AND    RK.PLANID = V_PLANID
    # #     # AND    A.ITEM = RK.ITEM
    # #     # AND    A.SITEID = RK.SITEID
    # #     # AND    A.SALESID = RK.SALESID
    df_DP3 = pd.merge(df_DP1, df_DP2, how='inner',
                                    left_on=[ND_.PLANID,ND_.ITEMID, ND_.SITEID,  ND_.SALESID],
                                    right_on=['PLANID','ITEM', 'SITEID','SALESID'])

    df_MVES_ITEMSITE = df_in_Netting_ES_Item_Site_ODS_W
    df_DP3 = df_DP3.loc[
            ~(df_DP3[ND_.ITEMID] + df_DP3[ND_.SALESID] + df_DP3[ND_.SITEID]).isin(
                df_MVES_ITEMSITE[MVES_ITEMSITE.ITEM]+df_MVES_ITEMSITE[MVES_ITEMSITE.SALESID]+df_MVES_ITEMSITE[MVES_ITEMSITE.SITEID]
            )
        ]


    df_DP3[ND_.DEMANDPRIORITY] = df_DP3[ND_.DEMANDPRIORITY].fillna(0).astype('int32')
    df_DP3[ND_.QTYPROMISED] = df_DP3[ND_.QTYPROMISED].fillna(0).astype('int32')

    # # RANK() OVER (ORDER BY RK.RNK, A.SALESID, A.DEMANDPRIORITY DESC, A.QTYPROMISED DESC, ROWNUM ) RNUM, 
    df_DP3 = df_DP3.reset_index()
    df_DP3.rename(columns={'index':'ROWNUM'}, inplace=True)
    df_DP3_sorted = df_DP3.sort_values(by=['RNK', ND_.SALESID, ND_.DEMANDPRIORITY, ND_.QTYPROMISED,'ROWNUM'], ascending=[True,True,False,False,True])
    # 소스상 df_DP3_sorted['RNUM'] = df_DP3_sorted.rank(method='min', ascending=True).reset_index(drop=True).index+1 가 맞으나 아래와 같이 수정하면 에러건수가 줄어듬_20250613
    df_DP3_sorted['RNUM'] = (df_DP3_sorted.sort_values(by=['RNK', ND_.SALESID, ND_.DEMANDPRIORITY, ND_.QTYPROMISED,'ROWNUM'], ascending=[True,True,False,False,True]).groupby([ND_.ITEMID,ND_.SITEID,'PROMISEDDELDATE2']).cumcount() + 1)
    # df_DP3_sorted['RNUM'] = df_DP3_sorted.rank(method='min', ascending=True).reset_index(drop=True).index+1
    # df_DP3['RNUM'] = (df_DP3.sort_values(by=['RNK', ND_.SALESID, ND_.DEMANDPRIORITY, ND_.QTYPROMISED,'ROWNUM'], ascending=[True,True,False,False,True]).groupby([ND_.ITEMID,ND_.SITEID,'PROMISEDDELDATE2']).cumcount() + 1)

    df_DP3_sorted['UPBY'] = ''
    df_DP = df_DP3_sorted.sort_values([ND_.ITEMID, ND_.SITEID, 'RNK', ND_.SALESID, 'RNUM',ND_.DEMANDPRIORITY, ND_.QTYPROMISED], ascending=[True,True,True,True,True,False,False]).reset_index()


    additional_column = ['RNK', 'EXCEPTQ','CUREXCEPT','PROMISEDDELDATE2','RNUM', 'UPBY']
    df_DP = df_DP[ND_.LIST_COLUMN+additional_column]

    ND_.RNK_IDX = df_DP.columns.get_loc('RNK')
    ND_.EXCEPTQ_IDX = df_DP.columns.get_loc('EXCEPTQ')
    ND_.CUREXCEPT_IDX = df_DP.columns.get_loc('CUREXCEPT')
    ND_.PROMISEDDELDATE2_IDX = df_DP.columns.get_loc('PROMISEDDELDATE2')
    ND_.RNUM_IDX = df_DP.columns.get_loc('RNUM')
    ND_.UPBY_IDX = df_DP.columns.get_loc('UPBY')

    colcount = len(ND_.LIST_COLUMN)

    df_DP = df_DP.astype({'RNK':'int32', 'EXCEPTQ':'int32', 'CUREXCEPT':'int32', ND_.SOLINENUM:'int32', ND_.QTYPROMISED:'int32', 'PROMISEDDELDATE2':'int32', 'RNUM':'int32', 'UPBY':'str'})

    df_DPSalesCurexcept = df_DP[[ND_.SITEID,ND_.ITEMID,ND_.SALESID,'EXCEPTQ','CUREXCEPT']].copy()
    df_DPSalesCurexcept['UPBY'] = ''
    df_DPSalesCurexcept['UPBY'] = df_DPSalesCurexcept['UPBY'].astype('str')
    df_DPSalesCurexcept['UPBY'] = df_DPSalesCurexcept['UPBY'].fillna('')

    # list_Sales_loop_update를 dict로 변경
    dict_dp_sales_curexcept = {}
    for row in df_DPSalesCurexcept.values:
        key = row[0]+'::'+row[1]+'::'+row[2]
        dict_dp_sales_curexcept[key] = [row[0], row[1], row[2], int(row[3]), int(row[4]), row[5]]

    list_Outbound_loop_insert = np.empty((0,colcount), dtype=object)
    list_Outbound_loop_update = np.empty((0,5), dtype=object)
    list_Outbound_loop_update_sol = np.empty((0,5), dtype=object)
    # list_Sales_loop_update = np.empty((0,6), dtype=object)
    list_Ginet_loop_update = np.empty((0,7), dtype=object)

    # -- 기여도 있는 sales 먼저 차감을 위해 for loop 추가  20220919 
    # -- 아래 6가지 경우 발생 = 있고 없고는 경우가 더 많아져서 전부 = 조건 넣고 elseif로 연결
    # -- 1. site.gidp >= qtypromised>= sales.gidp 
    # -- 2. site.gidp >= sales.gidp >= qtypromised
    # -- 3. qtypromised >= site.gidp >= sales.gidp
    # -- 4. qtypromised >=sales.gidp >= site.gidp
    # -- 5. sales.gidp >=qtypromised >= site.gidp
    # -- 6. sales.gidp >= site.gidp >= qtypromised
    # -- WEEK,SITE별 계산된 GIDP 먼저 한껀 불러냄  

    dict_dp_inner = {}
    for row in df_DP.values:
        key = row[ND_.SITEID_IDX]+'::'+row[ND_.AP2ID_IDX]+'::'+row[ND_.ITEMID_IDX]+'::'+str(row[ND_.PROMISEDDELDATE2_IDX])
        value = tuple(row)
        if key in dict_dp_inner:
            dict_dp_inner[key].append(value)
        else:
            dict_dp_inner[key] = [value]


    nparr_site = df_SITE.to_numpy()
    for site_row in nparr_site:
        # --해당 주차 GIDP로 넘어가는 누적 수량 
        V_REMAINGIDP = 0 

        # -- GIDP 남는 값 저장 
        V_GIDP = int(site_row[SITESELL.GIDP_IDX])

        key = site_row[SITESELL.SITEID_IDX] + '::' + site_row[SITESELL.SALESID_IDX] + '::' + site_row[SITESELL.ITEM_IDX] + '::' + str(site_row[SITESELL.WEEK_IDX])

        if(key in dict_dp_inner):
            dptuple = dict_dp_inner.get(key)
            dplist = list(dptuple)

            #for loop 안에 변화값
            dict_dp_sales_curexcept2 = {}

            for dpinner in dplist:
                dp_row = list(dpinner)

                # SALES_CUREXCEPT 값
                row_sales_curexcept = list(dict_dp_sales_curexcept.get(dp_row[ND_.SITEID_IDX]+'::'+dp_row[ND_.ITEMID_IDX]+'::'+dp_row[ND_.SALESID_IDX]))

                # AND    RK.EXCEPTQ > RK.CUREXCEPT
                # if(int(dp_row[ND_.EXCEPTQ_IDX]) > int(dp_row[ND_.CUREXCEPT_IDX])):
                if(int(row_sales_curexcept[3]) > int(row_sales_curexcept[4])):
                
                    # -- 같은 WEEK, SALESID 내에서 남는값 저장
                    # -- CUREXCEPT : 이전주차에서 해당 SALES의 GIDP로 넘어간 수량 
                    if(dp_row[ND_.RNUM_IDX] == 1):
                        #V_EXCEPTQ := DP.EXCEPTQ- DP.CUREXCEPT;
                        # V_EXCEPTQ = int(dp_row[ND_.EXCEPTQ_IDX]) - int(dp_row[ND_.CUREXCEPT_IDX])
                        V_EXCEPTQ = int(row_sales_curexcept[3]) - int(row_sales_curexcept[4])
                        V_SALESID = dp_row[ND_.SALESID_IDX]
                    
                    if(V_SALESID != dp_row[ND_.SALESID_IDX]):
                        # V_EXCEPTQ = int(dp_row[ND_.EXCEPTQ_IDX]) - int(dp_row[ND_.CUREXCEPT_IDX])
                        V_EXCEPTQ = int(row_sales_curexcept[3]) - int(row_sales_curexcept[4])
                        V_SALESID = dp_row[ND_.SALESID_IDX]

                    dicSaleskey = dp_row[ND_.SITEID_IDX]+'::'+dp_row[ND_.ITEMID_IDX]+'::'+dp_row[ND_.SALESID_IDX]

                    # --2,6번 case : 그냥 qtypromised 전부 GIDP로
                    if(((V_GIDP >= V_EXCEPTQ) & (V_EXCEPTQ >= int(dp_row[ND_.QTYPROMISED_IDX])))
                        | ((V_EXCEPTQ >= V_GIDP) & (V_GIDP >= int(dp_row[ND_.QTYPROMISED_IDX])))):
                        # --GI DP로 전량 분류
                            # SALESORDERID = DP.SALESORDERID
                            # AND    PLANID       = DP.PLANID
                            # AND    SOLINENUM    = DP.SOLINENUM;

                        list_Outbound_loop_update_sol = np.append(list_Outbound_loop_update_sol, np.array([[dp_row[ND_.SALESORDERID_IDX], dp_row[ND_.PLANID_IDX], int(dp_row[ND_.SOLINENUM_IDX]), int(dp_row[ND_.SOLINENUM_IDX])+200,str(dp_row[ND_.UPBY_IDX]) + '_1']]), axis=0)

                        if dicSaleskey in dict_dp_sales_curexcept2 :
                            intTempCurexcept = int(dict_dp_sales_curexcept2[dicSaleskey][4])
                            strTempUpby = str(dict_dp_sales_curexcept2[dicSaleskey][5])
                            dict_dp_sales_curexcept2[dicSaleskey] = [dp_row[ND_.SITEID_IDX],dp_row[ND_.ITEMID_IDX],dp_row[ND_.SALESID_IDX],int(row_sales_curexcept[3]),intTempCurexcept + int(dp_row[ND_.QTYPROMISED_IDX]), strTempUpby+'_1']
                        else:
                            dict_dp_sales_curexcept2[dicSaleskey] = [dp_row[ND_.SITEID_IDX],dp_row[ND_.ITEMID_IDX],dp_row[ND_.SALESID_IDX],int(row_sales_curexcept[3]),int(row_sales_curexcept[4]) + int(dp_row[ND_.QTYPROMISED_IDX]), row_sales_curexcept[5]+'_1']

                        V_REMAINGIDP += int(dp_row[ND_.QTYPROMISED_IDX])
                        # print('V_REMAINGIDP:' + str(V_REMAINGIDP))
                        # --남은 기여 수량 
                        V_EXCEPTQ -= int(dp_row[ND_.QTYPROMISED_IDX])
                        # --남은  SITE.GIDP
                        V_GIDP -= int(dp_row[ND_.QTYPROMISED_IDX])

                    # --4,5 case : site.gidp 만큼만 gidp 로 넘김
                    elif(((V_EXCEPTQ >= int(dp_row[ND_.QTYPROMISED_IDX])) & (int(dp_row[ND_.QTYPROMISED_IDX]) >= V_GIDP))
                        |((int(dp_row[ND_.QTYPROMISED_IDX]) >= V_EXCEPTQ) & (V_EXCEPTQ >= V_GIDP))):

                        # -- 남은 SITEGIDP만큼만 넘어간다...
                        list_Outbound_loop_update = np.append(list_Outbound_loop_update, np.array([[dp_row[ND_.SALESORDERID_IDX], dp_row[ND_.PLANID_IDX], int(dp_row[ND_.SOLINENUM_IDX]), int(dp_row[ND_.QTYPROMISED_IDX]) - int(V_GIDP), str(dp_row[ND_.UPBY_IDX]) + '_U2']]), axis=0)

                        # --! 잔량만큼은 SOLINENUM 더해서 INSERT하여 GI Short 처리  !--
                        
                        # /**++SOLINENUM +200 분리++**/
                        # /*++남은잔량만큼 INSERT*/
                        dp_row[ND_.SOLINENUM_IDX] = int(dp_row[ND_.SOLINENUM_IDX])+200
                        dp_row[ND_.QTYPROMISED_IDX] = int(V_GIDP)
                        dp_row[ND_.UPBY_IDX] = str(dp_row[ND_.UPBY_IDX])+'_I2'
                        dp_row_insert = dp_row[:colcount].copy()
                        list_Outbound_loop_insert = np.append(list_Outbound_loop_insert, np.array([dp_row_insert]), axis=0)

                        if dicSaleskey in dict_dp_sales_curexcept2 :
                            intTempCurexcept = int(dict_dp_sales_curexcept2[dicSaleskey][4])
                            strTempUpby = str(dict_dp_sales_curexcept2[dicSaleskey][5])
                            dict_dp_sales_curexcept2[dicSaleskey] = [dp_row[ND_.SITEID_IDX],dp_row[ND_.ITEMID_IDX],dp_row[ND_.SALESID_IDX],int(row_sales_curexcept[3]),intTempCurexcept + int(V_GIDP), strTempUpby+'_2']
                        else:
                            dict_dp_sales_curexcept2[dp_row[ND_.SITEID_IDX]+'::'+dp_row[ND_.ITEMID_IDX]+'::'+dp_row[ND_.SALESID_IDX]] = [dp_row[ND_.SITEID_IDX],dp_row[ND_.ITEMID_IDX],dp_row[ND_.SALESID_IDX],int(row_sales_curexcept[3]),int(row_sales_curexcept[4]) + int(V_GIDP), row_sales_curexcept[5]+'_2']

                        V_REMAINGIDP += int(V_GIDP)
                        # --남은 기여 수량 
                        V_EXCEPTQ -= int(V_GIDP)
                        # --남은  SITE.GIDP
                        V_GIDP = 0

                    # --1,3 case : sales.gidp만큼만 넘긴다.
                    elif(((int(V_GIDP) >= int(dp_row[ND_.QTYPROMISED_IDX])) & (int(dp_row[ND_.QTYPROMISED_IDX]) >= int(V_EXCEPTQ)))
                        |((int(dp_row[ND_.QTYPROMISED_IDX]) >= int(V_GIDP)) & (int(V_GIDP) >= int(V_EXCEPTQ)))):

                        # QTYPROMISED 값이 0이상 만 보이도록 하고 싶을 경우 아래 주석 제거
                        # if(int(V_EXCEPTQ) > 0):

                        # --!  그대로 둔다 !--
                        list_Outbound_loop_update = np.append(list_Outbound_loop_update, np.array([[dp_row[ND_.SALESORDERID_IDX], dp_row[ND_.PLANID_IDX], int(dp_row[ND_.SOLINENUM_IDX]), int(dp_row[ND_.QTYPROMISED_IDX]) - int(V_EXCEPTQ),str(dp_row[ND_.UPBY_IDX]) + '_U3']]), axis=0)

                        # # /*++남은잔량만큼 INSERT*/
                        dp_row[ND_.SOLINENUM_IDX] = int(dp_row[ND_.SOLINENUM_IDX])+200
                        dp_row[ND_.QTYPROMISED_IDX] = int(V_EXCEPTQ)
                        dp_row[ND_.UPBY_IDX] = str(dp_row[ND_.UPBY_IDX])+'_I3'
                        dp_row_insert = dp_row[:colcount].copy()
                        list_Outbound_loop_insert = np.append(list_Outbound_loop_insert, np.array([dp_row_insert]), axis=0)

                        
                        if dicSaleskey in dict_dp_sales_curexcept2 :
                            intTempCurexcept = int(dict_dp_sales_curexcept2[dicSaleskey][4])
                            strTempUpby = str(dict_dp_sales_curexcept2[dicSaleskey][5])
                            dict_dp_sales_curexcept2[dicSaleskey] = [dp_row[ND_.SITEID_IDX],dp_row[ND_.ITEMID_IDX],dp_row[ND_.SALESID_IDX],int(row_sales_curexcept[3]),intTempCurexcept + int(V_EXCEPTQ), strTempUpby+'_3']
                        else:
                            dict_dp_sales_curexcept2[dp_row[ND_.SITEID_IDX]+'::'+dp_row[ND_.ITEMID_IDX]+'::'+dp_row[ND_.SALESID_IDX]] = [dp_row[ND_.SITEID_IDX],dp_row[ND_.ITEMID_IDX],dp_row[ND_.SALESID_IDX],int(row_sales_curexcept[3]),int(row_sales_curexcept[4]) + int(V_EXCEPTQ), row_sales_curexcept[5]+'_3']

                        V_REMAINGIDP += V_EXCEPTQ
                        # --남은  SITE.GIDP
                        V_GIDP -= V_EXCEPTQ
                        # --남은 기여 수량 
                        V_EXCEPTQ = 0

                    if(V_REMAINGIDP >= site_row[SITESELL.GIDP_IDX]):
                        break

            for key, value in dict_dp_sales_curexcept2.items():
                dict_dp_sales_curexcept[key] = value
                
            if(site_row[SITESELL.GIDP_IDX] > 0):
                list_Ginet_loop_update = np.append(list_Ginet_loop_update, 
                    np.array([[V_PLANID, site_row[SITESELL.SITEID_IDX], site_row[SITESELL.SALESID_IDX], site_row[SITESELL.ITEM_IDX], str(site_row[SITESELL.WEEK_IDX]), int(max(0, int(site_row[SITESELL.GIDP_IDX]) - V_REMAINGIDP)), 'Y']]), 
                    axis=0)        


    df_sales_update_temp = DF.from_dict(dict_dp_sales_curexcept, orient='index')

    if(not df_sales_update_temp.empty) :
        df_sales_update_temp.columns = ['SITEID','ITEM','SALESID','EXCEPTQ_NEW','CUREXCEPT_NEW','UPBY_NEW']
        df_sales_update_temp['PLANID'] = V_PLANID
        df_ginet_sales_update = df_sales_update_temp.astype({'PLANID':'str', 'SITEID':'str', 'ITEM':'str', 'SALESID':'str', 'CUREXCEPT_NEW':'int32','EXCEPTQ_NEW':'int32', 'UPBY_NEW':'str'})
        df_ginet_sales_update = df_ginet_sales_update.loc[df_ginet_sales_update['CUREXCEPT_NEW'] > 0]
        df_ginet_sales_update = df_ginet_sales_update[['PLANID','SITEID','ITEM','SALESID','CUREXCEPT_NEW','UPBY_NEW']]


    df_outbound_update      = pd.DataFrame(list_Outbound_loop_update)
    df_outbound_update_sol  = pd.DataFrame(list_Outbound_loop_update_sol)
    df_outbound_insert      = pd.DataFrame(list_Outbound_loop_insert)
    # df_ginet_sales_update   = pd.DataFrame(list_Sales_loop_update)
    df_ginet_update         = pd.DataFrame(list_Ginet_loop_update)

    dfMST_GINETTING_SELL.columns = ['PLANID', 'SITEID','SALESID', 'ITEM', 'WEEK', 'RTF', 'GI', 'EXCEPTQ', 'AVAIL', 'DP', 'CUREXCEPT', 'CURAVAIL', 'GIDP', 'MPDP', 'REMAINGIDP', 'UPDATEYN', 'RK']
    dfMST_GINETTING_SELL = dfMST_GINETTING_SELL.astype({'PLANID':'str', 'SITEID':'str', 'ITEM':'str', 'WEEK':'str', 'RTF':'int32', 
        'GI':'int32', 'EXCEPTQ':'int32', 'AVAIL':'int32', 'DP':'int32', 'CUREXCEPT':'int32', 
        'CURAVAIL':'int32', 'GIDP':'int32', 'MPDP':'int32', 'REMAINGIDP':'float64', 'UPDATEYN':'str', 'RK':'int32'
        })
    df_MST_GINETING_SALES['UPBY'] = ''
    df_MST_GINETING_SALES = df_MST_GINETING_SALES.astype({'UPBY':'str'})

    df_outbound_update.columns = [ND_.SALESORDERID, ND_.PLANID, ND_.SOLINENUM, 'QTYPROMISED_NEW','UPBY_NEW']
    df_outbound_update_sol.columns = [ND_.SALESORDERID, ND_.PLANID, ND_.SOLINENUM, 'SOLINENUM_NEW','UPBY_NEW']
    df_outbound_insert.columns = ND_.LIST_COLUMN
    # df_ginet_sales_update.columns = ['PLANID', 'SITEID', 'ITEM', 'SALESID', 'CUREXCEPT_NEW','UPBY_NEW']
    df_ginet_update.columns = ['PLANID', 'SITEID', 'SALESID', 'ITEM', 'WEEK', 'REMAINGIDP_NEW', 'UPDATEYN_NEW']


    df_outbound_update = df_outbound_update.astype({ND_.SALESORDERID:'str', ND_.PLANID:'str', ND_.SOLINENUM:'int32', 'QTYPROMISED_NEW':'int32', 'UPBY_NEW':'str'})
    df_outbound_update_sol = df_outbound_update_sol.astype({ND_.SALESORDERID:'str', ND_.PLANID:'str', ND_.SOLINENUM:'int32', 'SOLINENUM_NEW':'int32', 'UPBY_NEW':'str'})
    # df_ginet_sales_update = df_ginet_sales_update.astype({'PLANID':'str', 'SITEID':'str', 'ITEM':'str', 'SALESID':'str', 'CUREXCEPT_NEW':'int32', 'UPBY_NEW':'str'})
    df_ginet_update = df_ginet_update.astype({'PLANID':'str', 'SITEID':'str', 'SALESID':'str', 'ITEM':'str', 'WEEK':'str','REMAINGIDP_NEW':'float64', 'UPDATEYN_NEW':'str'})

    df_MST_GINETING_SALES.rename(columns={
        GUI_SALESRESULTSMM.ITEM:'ITEM',
        GUI_SALESRESULTSMM.SALESID:'SALESID', 
        GUI_SALESRESULTSMM.SITEID:'SITEID'
    }, inplace=True)


    if(not df_ginet_sales_update.empty):
        df_ginet_sales_merge = df_MST_GINETING_SALES.merge(df_ginet_sales_update, on=['PLANID', 'SITEID', 'ITEM', 'SALESID'], how='left') 
        df_ginet_sales_merge.loc[~df_ginet_sales_merge['CUREXCEPT_NEW'].isna(),'CUREXCEPT'] = df_ginet_sales_merge['CUREXCEPT_NEW']
        df_ginet_sales_merge.loc[~df_ginet_sales_merge['UPBY_NEW'].isna(),'UPBY'] = df_ginet_sales_merge['UPBY_NEW']
        df_ginet_sales_merge = df_ginet_sales_merge.drop(columns=['CUREXCEPT_NEW','UPBY_NEW'])
        df_MST_GINETING_SALES = df_ginet_sales_merge

    df_outbound_mege = df_Outbound.merge(df_outbound_update, on=[ND_.SALESORDERID, ND_.PLANID, ND_.SOLINENUM], how='left')
    df_outbound_mege.loc[~df_outbound_mege['QTYPROMISED_NEW'].isna(),ND_.QTYPROMISED] = df_outbound_mege['QTYPROMISED_NEW']
    df_outbound_mege.loc[~df_outbound_mege['UPBY_NEW'].isna(),'UPBY'] = df_outbound_mege['UPBY_NEW']
    df_outbound_mege = df_outbound_mege.drop(columns=['QTYPROMISED_NEW','UPBY_NEW'])
    df_outbound_mege = df_outbound_mege.merge(df_outbound_update_sol, on=[ND_.SALESORDERID, ND_.PLANID, ND_.SOLINENUM], how='left')
    df_outbound_mege.loc[~df_outbound_mege['SOLINENUM_NEW'].isna(),ND_.SOLINENUM] = df_outbound_mege['SOLINENUM_NEW']
    df_outbound_mege.loc[~df_outbound_mege['UPBY_NEW'].isna(),'UPBY'] = df_outbound_mege['UPBY_NEW']
    df_outbound_mege = df_outbound_mege.drop(columns=['SOLINENUM_NEW','UPBY_NEW'])

    df_ginet_update2 = df_ginet_update.drop_duplicates()
    # df_ginet_update2 = df_ginet_update2.astype({'WEEK':'str'})

    df_ginet_merge = dfMST_GINETTING_SELL.merge(df_ginet_update2, how='left', on=['PLANID', 'SITEID', 'ITEM', 'WEEK','SALESID']) 

    # df_ginet_merge.loc[~df_ginet_merge['UPDATEYN_NEW'].isna(),'UPDATEYN'] = df_ginet_merge['UPDATEYN_NEW']
    # df_ginet_merge.loc[~df_ginet_merge['REMAINGIDP_NEW'].isna(),'REMAINGIDP'] = df_ginet_merge['GIDP'] - df_ginet_merge['REMAINGIDP_NEW']
    # df_ginet_merge['REMAINGIDP'] = df_ginet_merge['REMAINGIDP'].fillna(0).astype('int32')
    # df_ginet_merge['REMAINGIDP_NEW'] = df_ginet_merge['REMAINGIDP_NEW'].fillna(0).astype('int32')
    # # df_ginet_merge.loc[~df_ginet_merge['REMAINGIDP_NEW'].isna(),'REMAINGIDP'] = df_ginet_merge['REMAINGIDP_NEW']
    # # df_ginet_merge.loc[~df_ginet_merge['UPDATEYN_NEW'].isna(),'UPDATEYN'] = df_ginet_merge['UPDATEYN_NEW']


    # df_ginet_merge['REMAINGIDP'] = df_ginet_merge['REMAINGIDP'].fillna(df_ginet_merge['GIDP']).astype('int32')
    df_ginet_merge.loc[~df_ginet_merge['UPDATEYN_NEW'].isna(),'UPDATEYN'] = df_ginet_merge['UPDATEYN_NEW']
    df_ginet_merge.loc[~df_ginet_merge['REMAINGIDP_NEW'].isna(),'REMAINGIDP'] = df_ginet_merge['REMAINGIDP_NEW']
    df_ginet_merge['REMAINGIDP'] = df_ginet_merge['REMAINGIDP'].fillna(df_ginet_merge['GIDP']).astype('int32')
    # df_ginet_merge['REMAINGIDP'] = np.where(df_ginet_merge['GIDP'] >  df_ginet_merge['REMAINGIDP_NEW'], df_ginet_merge['GIDP'] - df_ginet_merge['REMAINGIDP_NEW'] , 0)


    df_ginet_merge = df_ginet_merge.drop(columns=['REMAINGIDP_NEW'])
    df_ginet_merge = df_ginet_merge.drop(columns=['UPDATEYN_NEW'])

    # df_outbound_meged2 = df_outbound_mege.merge(df_outbound_update, on=['SALESORDERID', 'PLANID', 'SOLINENUM'], how='inner')


    # # df_Outbound_final_1 = pd.concat([df_Outbound, df_outbound_mege], axis=0, ignore_index=True)
    df_Outbound_final_2 = pd.concat([df_outbound_mege, df_outbound_insert], axis=0, ignore_index=True)
    # # df_MST_GINETING_SALES_final_1 = pd.concat([df_MST_GINETING_SALES, df_ginet_sales_merge], axis=0, ignore_index=True)
    # # df_MST_GINETTING_final_1 = pd.concat([dfMST_GINETTING, df_ginet_merge], axis=0, ignore_index=True)

    # , 'CUREXCEPT':'int32', 'RNK':'int32'
    df_MST_GINETING_SALES = df_MST_GINETING_SALES.astype({'RATIOEXCEPTQ':'float64'})
    df_MST_GINETING_SALES['CUREXCEPT'] = df_MST_GINETING_SALES['CUREXCEPT'].fillna(0).astype('int32')
    df_MST_GINETING_SALES['RNK'] = df_MST_GINETING_SALES['RNK'].fillna(0).astype('int32')

    df_MST_GINETING_SALES_final_1 = df_MST_GINETING_SALES
    df_MST_GINETTING_final_1 = df_ginet_merge


    logger.Step(4, 'SALES OPEN End')
    #endregion logger.Step(4, 'SALES OPEN')
    
    #region logger.Step(5, 'site OPEN')
    logger.Step(5, 'site OPEN Start')

    '''
    5.site OPEN
    '''
    colcount = len(ND_.LIST_COLUMN)

    list_Outbound_loop_insert = np.empty((0,colcount), dtype=object)
    list_Outbound_loop_update = np.empty((0,5), dtype=object)


    V_REXCEPT = 0
    V_SOLINENUM = 0

    df_R = df_MST_GINETTING_final_1.loc[df_MST_GINETTING_final_1['PLANID'] == V_PLANID]
    df_R = df_R[['SITEID', 'SALESID', 'ITEM', 'WEEK', 'RTF', 'GI', 'DP', 'GIDP', 'REMAINGIDP', 'MPDP']]
    df_R = df_R.astype({'SITEID':'str', 'SALESID':'str','ITEM':'str', 'WEEK':'str', 'GI':'int32','DP':'int32', 'GIDP':'int32','MPDP':'int32'})
    df_R = df_R[df_R['GIDP'] > 0]
    # df_R['REMAINGIDP'] = df_R['REMAINGIDP'].astype('float').astype('int32')
    df_R['GIDP'] = df_R['REMAINGIDP'].fillna(df_R['GIDP'])
    df_R = df_R.sort_values(['SITEID', 'SALESID', 'ITEM', 'WEEK'], ascending=[True,True,True,True])

    df_Outbound_final_2 = df_Outbound_final_2.reset_index()
    df_Outbound_final_2.rename(columns={'index':'ID'}, inplace=True)

    df_DP1 = df_Outbound_final_2.copy(deep=True)




    # # SOLINENUMMAX 구하기 --> 사용안함
    # df_DP11 = df_DP1.loc[df_DP1['SOLINENUM'] < 300]
    # df_DP12 = df_DP11.groupby('SALESORDERID')[['SOLINENUM']].max().reset_index()
    # df_DP12.rename(columns={'SOLINENUM':'SOLINENUMMAX'}, inplace=True)


    # # 형전환
    df_DP1[ND_.QTYPROMISED] = df_DP1[ND_.QTYPROMISED].astype('float').astype('int32')
    df_DP1[ND_.SOLINENUM] = df_DP1[ND_.SOLINENUM].fillna(0).astype('int32')

    # # WHERE  A.PLANID = V_PLANID
    # # AND    QTYPROMISED>0
    # # AND    A.SOLINENUM < 200
    df_DP1 = df_DP1.loc[(df_DP1[ND_.PLANID]==V_PLANID)&(df_DP1[ND_.QTYPROMISED] > 0)&(df_DP1[ND_.SOLINENUM] < 200)]

    # AND    TO_CHAR(A.PROMISEDDELDATE,'IYYYIW') = SITE.WEEK
    df_DP1['PROMISEDDELDATE2'] = pd.to_datetime(df_DP1[ND_.PROMISEDDELDATE]).dt.strftime('%G%V')
    df_DP1['PROMISEDDELDATE2'] = df_DP1['PROMISEDDELDATE2'].astype('str')


    df_DP2 = df_MST_GINETING_SALES_final_1.copy(deep=True)
    df_DP2 = df_DP2.drop(columns=['UPBY'])


    df_DP2.rename(columns={
        'PLANID':ND_.PLANID,
        'ITEM':ND_.ITEMID,
        'SITEID':ND_.SITEID,
        'SALESID':ND_.SALESID 
    }, inplace=True)


    df_DP3 = pd.merge(df_DP1, df_DP2, how='left',
                                    left_on=[ND_.PLANID,ND_.ITEMID,ND_.SITEID,ND_.SALESID],
                                    right_on=[ND_.PLANID,ND_.ITEMID,ND_.SITEID,ND_.SALESID])
    df_DP3['RNK'] = df_DP3['RNK'].fillna(0)

    additional_column = ['ID','RNK', 'EXCEPTQ','CUREXCEPT','PROMISEDDELDATE2']
    df_DP3 = df_DP3[ND_.LIST_COLUMN+additional_column]

    ND_.ID_IDX = df_DP3.columns.get_loc('ID')
    ND_.RNK_IDX = df_DP3.columns.get_loc('RNK')
    ND_.EXCEPTQ_IDX = df_DP3.columns.get_loc('EXCEPTQ')
    ND_.CUREXCEPT_IDX = df_DP3.columns.get_loc('CUREXCEPT')
    ND_.PROMISEDDELDATE2_IDX = df_DP3.columns.get_loc('PROMISEDDELDATE2')


    #AND    NOT EXISTS (SELECT ''X'' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = ''Y'')
    df_DP3 = df_DP3.loc[
            ~(df_DP3[ND_.ITEMID]+'Y').isin(
                    df_in_Netting_Custom_Model_Map_ODS_W[MTA_CUSTOMMODELMAP.CUSTOMITEM]+
                            df_in_Netting_Custom_Model_Map_ODS_W[MTA_CUSTOMMODELMAP.ISVALID])
        ]

    # AND    NOT EXISTS (SELECT 'X' FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SALESID = A.SALESID AND SITEID = A.SITEID)
    df_DP3 = df_DP3.loc[
            ~(df_DP3[ND_.ITEMID] + df_DP3[ND_.SALESID] + df_DP3[ND_.SITEID]).isin(
                df_in_Netting_ES_Item_Site_ODS_W[MVES_ITEMSITE.ITEM]+df_in_Netting_ES_Item_Site_ODS_W[MVES_ITEMSITE.SALESID]+df_in_Netting_ES_Item_Site_ODS_W[MVES_ITEMSITE.SITEID]
            )
        ]


    # # # RANK() OVER (ORDER BY RK.RNK, A.SALESID, A.DEMANDPRIORITY DESC, A.QTYPROMISED DESC, ROWNUM ) RNUM, 
    # # df_DP3['RNUM'] = df_DP3.groupby(by=['ITEM', 'SITEID', 'RNK', 'SALESID'])['DEMANDPRIORITY','QTYPROMISED'].rank(method='min',ascending=False)
    df_DP3['RNK'] = df_DP3['RNK'].astype('float').astype('int32')

    df_DP = df_DP3.sort_values([ND_.SITEID, ND_.ITEMID, 'PROMISEDDELDATE2', 'RNK', ND_.PROMISEDDELDATE, ND_.DEMANDPRIORITY, ND_.QTYPROMISED], ascending=[True,True,True,True,True,False,False])


    df_Outbound_final_2[ND_.SOLINENUM] = df_Outbound_final_2[ND_.SOLINENUM].fillna(0).astype('int32')
    df_dic = df_Outbound_final_2[[ND_.SALESORDERID,ND_.SOLINENUM]]

    # dict 생성
    dict_salesorder = {}
    for row in df_dic.values:
        soid = row[0]
        solinenum = int(row[1])
        if soid in dict_salesorder:
            dict_salesorder[soid].add(solinenum)
        else:
            dict_salesorder[soid] = set((solinenum,))


    #DP2를 OrderedDict으로 변경
    dict_df_dp = OrderedDict()
    for row in df_DP.values:
        key = row[ND_.ITEMID_IDX]+'::'+row[ND_.SITEID_IDX]+'::'+row[ND_.AP2ID_IDX]+'::'+str(row[ND_.PROMISEDDELDATE2_IDX])
        value = tuple(row)
        if key in dict_df_dp:
            dict_df_dp[key].append(value)
        else:
            dict_df_dp[key] = [value]

    nparr_R = df_R.to_numpy()
        # SITEID_IDX      =  0
        # ITEM_IDX        =  1
        # WEEK_IDX	    =  2
        # RTF_IDX         =  3
        # GI_IDX          =  4
        # DP_IDX          =  5
        # GIDP_IDX        =  6
        # MPDP_IDX        =  7
    for r_row in nparr_R:

        V_REXCEPT = r_row[R.GIDP_IDX]

        key = r_row[R.ITEM_IDX] + '::' + r_row[R.SITEID_IDX] + '::' + r_row[R.SALESID_IDX] + '::' + str(r_row[R.WEEK_IDX])
        dptuple = dict_df_dp.get(key)

        if(dptuple != None):
            dplist = list(dptuple)

            for dpinner in dplist:
                dp = list(dpinner)

                # --! 남은 잔량이 0보다 클 경우만 진행한다. 아니면 다음 대상으로 고고싱 !--
                if(V_REXCEPT > 0):
                    # --! 남은 잔량이 DEMAND보다 많을 경우 !--
                    if(V_REXCEPT >= int(dp[ND_.QTYPROMISED_IDX])):
                        V_SOLINENUM = int(dp[ND_.SOLINENUM_IDX]) + 200
                        
                        max_solinenum = 0
                        b_same_solinenum = False
                        set_solinenum = dict_salesorder[dp[ND_.SALESORDERID_IDX]]

                        if(int(dp[ND_.SOLINENUM_IDX]) >= 100):
                            for in_solinenum in set_solinenum:
                                if(max_solinenum <= int(in_solinenum)):
                                    max_solinenum = int(in_solinenum)

                                if(V_SOLINENUM == in_solinenum):
                                    #기존값이존재하는경우
                                    b_same_solinenum = True
                                    
                            if(b_same_solinenum):
                                V_SOLINENUM = max_solinenum + 1
                        else:
                            for in_solinenum in set_solinenum:
                                if((max_solinenum <= int(in_solinenum))&(int(in_solinenum)<300)):
                                    max_solinenum = int(in_solinenum)

                                if(V_SOLINENUM == in_solinenum):
                                    #기존값이존재하는경우
                                    b_same_solinenum = True
                                    
                            if(b_same_solinenum):
                                V_SOLINENUM = max_solinenum + 1

                        set_solinenum.add(V_SOLINENUM)
                        dict_salesorder[dp[ND_.SALESORDERID_IDX]] = set_solinenum

                        list_Outbound_loop_update = np.append(list_Outbound_loop_update, np.array([[dp[ND_.ID_IDX],dp[ND_.SALESORDERID_IDX], dp[ND_.PLANID_IDX], int(V_SOLINENUM), int(dp[ND_.QTYPROMISED_IDX])]]), axis=0)

                        V_REXCEPT -= int(dp[ND_.QTYPROMISED_IDX])
                    else:
                        # --! 남은 잔량이 DEMAND보다 적을 경우 !--
                        # --! QTYPROMISED - 잔량 뺀 나머지는 그대로 둔다 !--
                        # UPDATE EXP_SOPROMISESRC
                        # SET    QTYPROMISED  = DP.QTYPROMISED - V_REXCEPT
                        # WHERE ROWID = DP.ROWA;

                        V_SOLINENUM = dp[ND_.SOLINENUM_IDX] + 200
                        max_solinenum = 0
                        b_same_solinenum = False
                        set_solinenum = dict_salesorder[dp[ND_.SALESORDERID_IDX]]

                        if(int(dp[ND_.SOLINENUM_IDX]) >= 100):
                            for in_solinenum in set_solinenum:
                                if(max_solinenum <= in_solinenum):
                                    max_solinenum = in_solinenum

                                if(V_SOLINENUM == in_solinenum):
                                    #기존값이존재하는경우
                                    b_same_solinenum = True

                            if(b_same_solinenum):
                                V_SOLINENUM = max_solinenum + 1
                        else:
                            for in_solinenum in set_solinenum:
                                if((max_solinenum <= int(in_solinenum))&(int(in_solinenum)<300)):
                                    max_solinenum = int(in_solinenum)

                                if(V_SOLINENUM == in_solinenum):
                                    #기존값이존재하는경우
                                    b_same_solinenum = True
                                    
                            if(b_same_solinenum):
                                V_SOLINENUM = max_solinenum + 1

                        set_solinenum.add(V_SOLINENUM)
                        dict_salesorder[dp[ND_.SALESORDERID_IDX]] = set_solinenum

                        list_Outbound_loop_update = np.append(list_Outbound_loop_update, np.array([[dp[ND_.ID_IDX],dp[ND_.SALESORDERID_IDX], dp[ND_.PLANID_IDX], int(dp[ND_.SOLINENUM_IDX]), int(dp[ND_.QTYPROMISED_IDX])-V_REXCEPT]]), axis=0)
                        
                        # if(V_SOLINENUM <= dp_row[ND_.SOLINENUMMAX_IDX]):
                        #     V_SOLINENUM = dp_row[ND_.SOLINENUMMAX_IDX]+1

                        dp[ND_.SOLINENUM_IDX] = int(V_SOLINENUM)
                        dp[ND_.QTYPROMISED_IDX] = int(V_REXCEPT)
                        dp_row_insert = dp[:colcount].copy()
                        list_Outbound_loop_insert = np.append(list_Outbound_loop_insert, np.array([dp_row_insert]), axis=0)

                        # dp_row[SITE.QTYPROMISED_IDX] - V_REXCEPT
                        V_REXCEPT = 0

    df_outbound_update_aaa  = pd.DataFrame(list_Outbound_loop_update)
    df_outbound_insert      = pd.DataFrame(list_Outbound_loop_insert)

    df_outbound_update_aaa.columns = ['ID', ND_.SALESORDERID, ND_.PLANID, 'SOLINENUM_NEW', 'QTYPROMISED_NEW']

    df_outbound_update = df_outbound_update_aaa[['ID', 'SOLINENUM_NEW', 'QTYPROMISED_NEW']]
    df_outbound_insert.columns = ND_.LIST_COLUMN

    df_outbound_update = df_outbound_update.astype({'ID':'int32', 'SOLINENUM_NEW':'int32', 'QTYPROMISED_NEW':'int32'})

    df_outbound_mege = df_Outbound_final_2.merge(df_outbound_update, on=['ID'], how='left')
    df_outbound_mege.loc[~df_outbound_mege['QTYPROMISED_NEW'].isna(),ND_.QTYPROMISED] = df_outbound_mege['QTYPROMISED_NEW']
    df_outbound_mege.loc[~df_outbound_mege['SOLINENUM_NEW'].isna(),ND_.SOLINENUM] = df_outbound_mege['SOLINENUM_NEW']
    df_outbound_mege = df_outbound_mege.drop(columns=['QTYPROMISED_NEW','SOLINENUM_NEW'])


    df_Outbound_final_4 = pd.concat([df_outbound_mege, df_outbound_insert], axis=0, ignore_index=True)
    df_Outbound_final_4[ND_.SOLINENUM] = df_Outbound_final_4[ND_.SOLINENUM].fillna(0).astype('float').astype('int32')

    logger.Step(5, 'site OPEN End')
    #endregion logger.Step(5, 'site OPEN')
    
    #region logger.Step(6, 'PRIORITY S SRC')
    logger.Step(6, 'PRIORITY S SRC Start')

    '''
    5.PRIORITY S SRC
    '''

    df_priority = df_Outbound_final_4.loc[df_Outbound_final_4[ND_.PLANID] == V_PLANID]
    df_priority = df_priority[[ND_.PLANID, ND_.SALESORDERID,ND_.SOLINENUM,ND_.SITEID,ND_.ITEMID,ND_.DEMANDPRIORITY]]
    df_priority[ND_.SOLINENUM] = df_priority[ND_.SOLINENUM].fillna(0).astype('float').astype('int32')

    df_priority = df_priority.loc[
            df_priority[ND_.SOLINENUM].between(200, 299)
        ]

    df_priority[ND_.SOLINENUM] = df_priority[ND_.SOLINENUM].astype('str')
    df_Outbound_final_4[ND_.SOLINENUM] = df_Outbound_final_4[ND_.SOLINENUM].astype('str')

    df_priority[ND_.DEMANDPRIORITY] = df_priority[ND_.DEMANDPRIORITY].astype('str')

    # AND     EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
    df_priority = df_priority.loc[
            (df_priority[ND_.ITEMID] + df_priority[ND_.SITEID]).isin(
                df_in_Netting_Seller_Map[V_MTA_SELLERMAP.ITEM]+df_in_Netting_Seller_Map[V_MTA_SELLERMAP.SITEID]
            )
        ]

    df_mst_site = df_mst_site.loc[df_mst_site[MST_SITE.SHIPMENTTYPE] == 'DIR_S']
    df_mst_site = df_mst_site.loc[df_mst_site[MST_SITE.ISVALID] == 'Y']


    df_priority = df_priority.loc[
            ~(df_priority[ND_.SITEID]).isin(df_mst_site[MST_SITE.SITEID])
        ]

    start_pos, next_pos, total_length = find_priority_position('G_R001::1', 'DEMANDTYPERANK')
    digit = next_pos - start_pos
    df_priority['DEMANDPRIORITY_NEW'] = (
        df_priority[ND_.DEMANDPRIORITY].str[0: start_pos]
        + '8'.zfill(digit)
        + df_priority[ND_.DEMANDPRIORITY].str[next_pos:]
    )
    df_priority = df_priority[[ND_.PLANID, ND_.SALESORDERID,ND_.SOLINENUM, 'DEMANDPRIORITY_NEW']]

    df_priority[ND_.PLANID] = df_priority[ND_.PLANID].astype('str')
    df_priority[ND_.SALESORDERID] = df_priority[ND_.SALESORDERID].astype('str')
    df_priority[ND_.SOLINENUM] = df_priority[ND_.SOLINENUM].astype('int32')
    df_priority['DEMANDPRIORITY_NEW'] = df_priority['DEMANDPRIORITY_NEW'].astype('str')

    df_Outbound_final_4[ND_.PLANID] = df_Outbound_final_4[ND_.PLANID].astype('str')
    df_Outbound_final_4[ND_.SALESORDERID] = df_Outbound_final_4[ND_.SALESORDERID].astype('str')
    df_Outbound_final_4[ND_.SOLINENUM] = df_Outbound_final_4[ND_.SOLINENUM].astype('float').astype('int32')
    df_Outbound_final_4[ND_.DEMANDPRIORITY] = df_Outbound_final_4[ND_.DEMANDPRIORITY].astype('str')

    df_Outbound_final_5 = df_Outbound_final_4.merge(df_priority, how='left',
                                    on=[ND_.PLANID, ND_.SALESORDERID,ND_.SOLINENUM])

    df_Outbound_final_5.loc[~df_Outbound_final_5['DEMANDPRIORITY_NEW'].isna(),ND_.DEMANDPRIORITY] = df_Outbound_final_5['DEMANDPRIORITY_NEW']
    df_Outbound_final_5 = df_Outbound_final_5.drop(columns=['DEMANDPRIORITY_NEW'])

    df_Outbound_final_5[ND_.SOLINENUM] = df_Outbound_final_5[ND_.SOLINENUM].astype('float64').astype('int32')
    df_Outbound_final_5[ND_.QTYPROMISED] = df_Outbound_final_5[ND_.QTYPROMISED].astype('float64').astype('int32')

    df_Outbound_final_5 = df_Outbound_final_5[ND_.LIST_COLUMN]
    df_Outbound_final_5 = df_Outbound_final_5.reset_index(drop=True)

    logger.Step(6, 'PRIORITY S SRC End')
    #endregion logger.Step(6, 'PRIORITY S SRC')

    logger.Note('END GI Netting Sell', 20)

    df_Outbound = df_Outbound_final_5

    return df_Outbound[ND_.LIST_COLUMN]


