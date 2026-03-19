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

class GiNetConstant:
    # 'PLANID', 'SITEID', 'ITEM', 'WEEK', 'RTF', 'GI', 'EXCEPTQ', 'AVAIL', 'DP', 'CUREXCEPT', 'CURAVAIL', 'GIDP', 'MPDP', 'REMAINGIDP', 'UPDATEYN', 'RK'
    PLANID_IDX      =  0
    SITEID_IDX      =  1
    ITEM_IDX        =  2
    WEEK_IDX        =  3
    RTF_IDX         =  4
    GI_IDX          =  5
    EXCEPTQ_IDX     =  6
    AVAIL_IDX       =  7
    DP_IDX          =  8
    CUREXCEPT_IDX   =  9
    CURAVAIL_IDX    = 10
    GIDP_IDX        = 11
    MPDP_IDX        = 12
    REMAINGIDP_IDX  = 13
    UPDATEYN_IDX    = 14
    RK_IDX          = 15

class Site1Constant :
    SITEID_IDX = 0
    ITEM_IDX = 1
    WEEK_IDX = 2
    GIDP_IDX = 3

class RConstant :
    SITEID_IDX      =  0
    ITEM_IDX        =  1
    WEEK_IDX	    =  2
    RTF_IDX         =  3
    GI_IDX          =  4
    DP_IDX          =  5
    GIDP_IDX        =  6
    MPDP_IDX        =  7


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
GINET = GiNetConstant
SITE1 = Site1Constant
R     = RConstant
LOGGER = G_Logger


def set_gi_env(accessor: object) -> None:
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

def do_gi_netting(
        df_Inbound:DF, df_in_Netting_Seller_Map:DF, df_in_Netting_Available_Resource_ODS_W:DF, df_in_Netting_ES_Item_Site_ODS_W:DF,
        df_in_Netting_Custom_Model_Map_ODS_W:DF, df_Sales_Result_Smm:DF, df_mst_site:DF, logger: LOGGER,
    ) -> None:
    global V_EXCEPTQ, V_SALESID

    logger.Note('Start GI Netting', 20)
    
    # sub_start_time_1 = time.time()    
    #region logger.Step(1, 'Data Summary')
    logger.Step(1, 'Data Summary Start')
    '''
    mst_ginet 구성
    '''
    df_Outbound = df_Inbound.copy(deep=True)

    df_Sales_Result_Smm[GUI_SALESRESULTSMM.QTY_RTF].fillna(0).astype('int32')
    df_Sales_Result_Smm[GUI_SALESRESULTSMM.QTY_GI].fillna(0).astype('int32')

    #AND    NOT EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND SITEID = A.SITEID)
    df_MST_GINNETING_01 = df_Sales_Result_Smm.loc[
            ~(df_Sales_Result_Smm[GUI_SALESRESULTSMM.ITEM] + df_Sales_Result_Smm[GUI_SALESRESULTSMM.SITEID]).isin(
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


    df_MST_GINNETING_01 = df_MST_GINNETING_01[[GUI_SALESRESULTSMM.SITEID,GUI_SALESRESULTSMM.ITEM,GUI_SALESRESULTSMM.QTY_RTF,GUI_SALESRESULTSMM.QTY_GI]]
    df_MST_GINNETING_01 = df_MST_GINNETING_01.groupby(by=[GUI_SALESRESULTSMM.SITEID,GUI_SALESRESULTSMM.ITEM,])[[GUI_SALESRESULTSMM.QTY_RTF,GUI_SALESRESULTSMM.QTY_GI]].sum()
    df_MST_GINNETING_01['EXCEPT'] = df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_GI] - df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_RTF] 
    df_MST_GINNETING_01[GUI_SALESRESULTSMM.WEEK] = V_PLANWEEK
    df_EXCEPT = df_MST_GINNETING_01.loc[df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_RTF] < df_MST_GINNETING_01[GUI_SALESRESULTSMM.QTY_GI]].reset_index()

    # -- 1. 가용량 (INVENTORY + INTRANSIT + DISTRIBUTIONORDERS)
    # AND    A.WEEK BETWEEN '||V_WEEK1||' AND '||V_WEEK4||'
    df_MST_GINNETING_02 = df_in_Netting_Available_Resource_ODS_W.loc[df_in_Netting_Available_Resource_ODS_W[MST_INVENTORY_FNE.WEEK].between(V_WEEK1, V_WEEK4)]

    # AND    NOT EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
    df_MST_GINNETING_02 = df_MST_GINNETING_02[
            ~(df_MST_GINNETING_02[MST_INVENTORY_FNE.ITEM] + df_MST_GINNETING_02[MST_INVENTORY_FNE.SITEID]).isin(
                df_in_Netting_Seller_Map[V_MTA_SELLERMAP.ITEM]+df_in_Netting_Seller_Map[V_MTA_SELLERMAP.SITEID]
            )
        ]

    df_MST_GINNETING_02[MST_INVENTORY_FNE.QTY] = df_MST_GINNETING_02[MST_INVENTORY_FNE.QTY].fillna(0).astype('int32')

    df_MST_GINNETING_02.rename(columns={
        MST_INVENTORY_FNE.QTY: 'AVAILQTY',
    }, inplace=True)

    df_MST_GINNETING_02 = df_MST_GINNETING_02[[MST_INVENTORY_FNE.ITEM,MST_INVENTORY_FNE.SITEID,MST_INVENTORY_FNE.WEEK,'AVAILQTY']]
    df_MST_GINNETING_02['DPQTY'] = 0

    # -- 2. DEMAND
    df_MST_GINNETING_03 = df_Inbound[[ND_.ITEMID, ND_.SITEID, ND_.PROMISEDDELDATE, ND_.QTYPROMISED]].copy()
    df_MST_GINNETING_03.insert(3,'QTY',0)
    df_MST_GINNETING_03['QTY'] = df_MST_GINNETING_03['QTY'].fillna(0).astype('int32')
    df_MST_GINNETING_03[ND_.QTYPROMISED] = df_MST_GINNETING_03[ND_.QTYPROMISED].astype('float').astype('int32')
    # AND     QTYPROMISED > 0
    df_MST_GINNETING_03 = df_MST_GINNETING_03.loc[df_MST_GINNETING_03[ND_.QTYPROMISED] > 0]

    # AND     TO_CHAR(PROMISEDDELDATE, ''IYYYIW'') BETWEEN '||V_WEEK1||' AND '||V_WEEK4||'
    df_MST_GINNETING_03[ND_.PROMISEDDELDATE] = pd.to_datetime(df_MST_GINNETING_03[ND_.PROMISEDDELDATE]).dt.strftime('%G%V')
    df_MST_GINNETING_03 = df_MST_GINNETING_03.loc[df_MST_GINNETING_03[ND_.PROMISEDDELDATE].between(V_WEEK1, V_WEEK4)]

    # AND     NOT EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
    df_DEMAND = df_MST_GINNETING_03.loc[
        ~(df_MST_GINNETING_03[ND_.ITEMID] + df_MST_GINNETING_03[ND_.SITEID]).isin(
            df_in_Netting_Seller_Map[V_MTA_SELLERMAP.ITEM]+df_in_Netting_Seller_Map[V_MTA_SELLERMAP.SITEID]
        )
    ]

    df_MST_GINNETING_02 = df_MST_GINNETING_02.rename(columns={
        MST_INVENTORY_FNE.ITEM:ND_.ITEMID,
        MST_INVENTORY_FNE.SITEID:ND_.SITEID,
    })
    df_DEMAND = df_DEMAND.rename(columns={
        ND_.PROMISEDDELDATE:MST_INVENTORY_FNE.WEEK,
        'QTY': 'AVAILQTY',
        ND_.QTYPROMISED: 'DPQTY',
    })

    #df_MST_GINNETING_02 union all df_MST_GINNETING_03
    df_MST_GINNETING_04 = pd.concat([df_MST_GINNETING_02, df_DEMAND], axis=0, ignore_index=True)

    #groupby
    df_AVAIL = df_MST_GINNETING_04.groupby(by=[ND_.ITEMID, ND_.SITEID, MST_INVENTORY_FNE.WEEK]).sum(numeric_only=True).reset_index()
    df_AVAIL = df_AVAIL.rename(columns={
        MST_INVENTORY_FNE.WEEK:'WEEK'
    })

    #df_EXCEPT, df_AVAIL inner join
    dfMST_GINETTING = pd.merge(df_EXCEPT, df_AVAIL, how='inner',
                                left_on=[GUI_SALESRESULTSMM.ITEM, GUI_SALESRESULTSMM.SITEID],
                                right_on=[ND_.ITEMID, ND_.SITEID]).reset_index()
    dfMST_GINETTING = dfMST_GINETTING.sort_values(by=[ND_.ITEMID, ND_.SITEID, 'WEEK',]).reset_index()
    dfMST_GINETTING = dfMST_GINETTING[[ND_.SITEID,ND_.ITEMID,'WEEK',GUI_SALESRESULTSMM.QTY_RTF,GUI_SALESRESULTSMM.QTY_GI,'EXCEPT','AVAILQTY','DPQTY']]
    dfMST_GINETTING.insert(0, ND_.PLANID, V_PLANID)
    dfMST_GINETTING.columns = [ND_.PLANID, ND_.SITEID, ND_.ITEMID, 'WEEK', GUI_SALESRESULTSMM.QTY_RTF,GUI_SALESRESULTSMM.QTY_GI, 'EXCEPTQ', 'AVAIL', 'DP']
    dfMST_GINETTING['WEEK'] = dfMST_GINETTING['WEEK'].fillna(0).astype('int32')
    dfMST_GINETTING['RK'] = dfMST_GINETTING.groupby(by=[ND_.ITEMID, ND_.SITEID])['WEEK'].rank(method='min')
    dfMST_GINETTING.insert(9, 'CUREXCEPT', dfMST_GINETTING[['EXCEPTQ']])
    dfMST_GINETTING.insert(10, 'CURAVAIL', dfMST_GINETTING[['AVAIL']])
    dfMST_GINETTING.insert(11, 'GIDP', 0)
    dfMST_GINETTING.insert(12, 'MPDP', 0)
    dfMST_GINETTING.insert(13, 'REMAINGIDP', np.nan)
    dfMST_GINETTING.insert(14, 'UPDATEYN','')
    dfMST_GINETTING['WEEK'] = dfMST_GINETTING['WEEK'].fillna(0).astype('int32')


    '''
    mst_ginet_sales 생성
    '''
    #형전환
    df_Sales_Result_Smm[GUI_SALESRESULTSMM.QTY_RTF].fillna(0).astype('int32')
    df_Sales_Result_Smm[GUI_SALESRESULTSMM.QTY_GI].fillna(0).astype('int32')

    df_MST_GINETING_SALES_01 = df_Sales_Result_Smm.copy(deep=True)

    #AND    NOT EXISTS (SELECT ''X'' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = ''Y'')
    df_MST_GINETING_SALES_01 = df_MST_GINETING_SALES_01.loc[
            ~(df_MST_GINETING_SALES_01[GUI_SALESRESULTSMM.ITEM]+'Y').isin(
                    df_in_Netting_Custom_Model_Map_ODS_W[MTA_CUSTOMMODELMAP.CUSTOMITEM]+
                            df_in_Netting_Custom_Model_Map_ODS_W[MTA_CUSTOMMODELMAP.ISVALID])
        ]

    df_MST_GINETING_SALES_01.insert(0, 'PLANID', V_PLANID)

    df_MST_GINETING_SALES_01[GUI_SALESRESULTSMM.QTY_RTF] = df_MST_GINETING_SALES_01[GUI_SALESRESULTSMM.QTY_RTF].fillna(0).astype('int32')
    df_MST_GINETING_SALES_01[GUI_SALESRESULTSMM.QTY_GI] = df_MST_GINETING_SALES_01[GUI_SALESRESULTSMM.QTY_GI].fillna(0).astype('int32')

    V_PRE_PLAN = V_PREYEAR + V_PREYWEEK
    df_MST_GINETING_SALES_01 = df_MST_GINETING_SALES_01.loc[df_MST_GINETING_SALES_01[GUI_SALESRESULTSMM.WEEK]==V_PRE_PLAN]

    df_MST_GINETING_SALES_01 = df_MST_GINETING_SALES_01[['PLANID',GUI_SALESRESULTSMM.ITEM,GUI_SALESRESULTSMM.SALESID,GUI_SALESRESULTSMM.SITEID,GUI_SALESRESULTSMM.QTY_RTF,GUI_SALESRESULTSMM.QTY_GI]]

    df_MST_GINETING_SALES_01.insert(4, 'WEEK', V_PREPLANID)

    #GROUP BY ITEM, SALESID, SITEID--, AP2ID
    df_MST_GINETING_SALES_01 = df_MST_GINETING_SALES_01.groupby(by=['PLANID',GUI_SALESRESULTSMM.ITEM,GUI_SALESRESULTSMM.SALESID,GUI_SALESRESULTSMM.SITEID,'WEEK',])[[GUI_SALESRESULTSMM.QTY_RTF,GUI_SALESRESULTSMM.QTY_GI]].sum()

    df_MST_GINETING_SALES_01['EXCEPTQ'] = df_MST_GINETING_SALES_01[GUI_SALESRESULTSMM.QTY_GI] - df_MST_GINETING_SALES_01[GUI_SALESRESULTSMM.QTY_RTF] 

    #HAVING SUM(DECODE(CATEGORY, ''02RTF'', WEEK'||V_PREYWEEK||',0)) < SUM(DECODE(CATEGORY, ''30GI'' , WEEK'||V_PREYWEEK||',0))
    df_MST_GINETING_SALES_01 = df_MST_GINETING_SALES_01.loc[df_MST_GINETING_SALES_01[GUI_SALESRESULTSMM.QTY_RTF] < df_MST_GINETING_SALES_01[GUI_SALESRESULTSMM.QTY_GI]]

    df_GI1 = df_MST_GINETING_SALES_01.loc[df_MST_GINETING_SALES_01[GUI_SALESRESULTSMM.QTY_RTF] >= 0].reset_index()

    df_GI2 = df_GI1.groupby(by=['PLANID',GUI_SALESRESULTSMM.ITEM,GUI_SALESRESULTSMM.SITEID,'WEEK',])[['EXCEPTQ']].sum()
    df_GI2 = pd.merge(df_GI1, df_GI2, how='inner',
                            left_on=['PLANID',GUI_SALESRESULTSMM.ITEM,GUI_SALESRESULTSMM.SITEID,'WEEK'],
                            right_on=['PLANID',GUI_SALESRESULTSMM.ITEM,GUI_SALESRESULTSMM.SITEID,'WEEK'])
    df_GI2.columns = ['PLANID', GUI_SALESRESULTSMM.ITEM, GUI_SALESRESULTSMM.SALESID, GUI_SALESRESULTSMM.SITEID, 'WEEK', 'RTF', 'GI', 'EXCEPTQ', 'SUMEXCEPTQ']
    df_GI2.sort_values(by=[GUI_SALESRESULTSMM.ITEM, GUI_SALESRESULTSMM.SITEID, 'WEEK', 'EXCEPTQ']).reset_index()

    df_GI2['RATIOEXCEPTQ'] = df_GI2['EXCEPTQ'] / df_GI2['SUMEXCEPTQ']

    df_GI2['RNK'] = df_GI2.groupby(by=[GUI_SALESRESULTSMM.ITEM,GUI_SALESRESULTSMM.SITEID, 'WEEK'])['EXCEPTQ'].rank(method='min',ascending=False)
    df_GI2['DP'] = 0
    df_GI2['CUREXCEPT'] = 0
    df_MST_GINETING_SALES = df_GI2

    logger.Step(1, 'Data Summary End')
    #endregion logger.Step(1, 'Data Summary')
    # print(f"{time.time() - sub_start_time_1:.5f} sec")
    # sub_start_time_2 = time.time()
    #region logger.Step(2, 'AVAIL QTY ROLLING')
    logger.Step(2, 'AVAIL QTY ROLLING Start')
    ''' 
    2.AVAIL QTY ROLLING
    '''

    M_AVAIL = 0

    dt = np.dtype([("S1", 'object'),
        ("S2", 'object'),
        ("S3", 'object'),
        ("S4", 'object'), 
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
        ("S5", 'object'),
        ("i11", np.int32)])

    nparr_ginet = np.array([tuple(v) for v in dfMST_GINETTING.values.tolist()], dtype=dt)

    for row in nparr_ginet:
        if row[GINET.RK_IDX] == 1: # IF X.RK = 1 THEN
            if row[GINET.AVAIL_IDX] > row[GINET.DP_IDX]:     # IF X.AVAIL > X.DP THEN
                M_AVAIL = row[GINET.AVAIL_IDX] - row[GINET.DP_IDX] # M_AVAIL := X.AVAIL - X.DP;
            else:
                M_AVAIL = 0
        #--! 첫 행이 아니면 !--
        else:
            #--! CURAVAIL = AVAIL + M_AVAIL !--
            row[GINET.CURAVAIL_IDX] = row[GINET.AVAIL_IDX] + M_AVAIL
        
            if (row[GINET.AVAIL_IDX] + M_AVAIL) > row[GINET.DP_IDX]:
                M_AVAIL = (row[GINET.AVAIL_IDX] + M_AVAIL) - row[GINET.DP_IDX]
            else:
                M_AVAIL = 0

    logger.Step(2, 'AVAIL QTY ROLLING End')
    #endregion logger.Step(2, 'AVAIL QTY ROLLING Start')
    # print(f"{time.time() - sub_start_time_2:.5f} sec")
    # sub_start_time_3 = time.time()
    #region logger.Step(3, 'GINETTING OPEN')
    logger.Step(3, 'GINETTING OPEN Start')
    ''' 
    3.GINETTING OPEN
    '''
    M_EXCEPT = 0

    for row in nparr_ginet:
        if row[GINET.RK_IDX] == 1: # IF X.RK = 1 THEN
            if (row[GINET.CURAVAIL_IDX] - row[GINET.DP_IDX]) >= 0: # IF X.CURAVAIL - X.DP >=0 THEN
                #--! 전량 MPDP, GIDP 0 !--
                row[GINET.GIDP_IDX] = 0
                row[GINET.MPDP_IDX] = row[GINET.DP_IDX]

                M_EXCEPT = row[GINET.EXCEPTQ_IDX]
            else: 
                #--! DP 가 남았을 경우 !--
                if (row[GINET.EXCEPTQ_IDX] >= (row[GINET.DP_IDX] - row[GINET.CURAVAIL_IDX])):
                    #--! 가용량이 없는 만큼은 GIDP, 가용량이 있는 만큼은 MPDP !--
                    row[GINET.GIDP_IDX] = (row[GINET.DP_IDX] - row[GINET.CURAVAIL_IDX])
                    row[GINET.MPDP_IDX] = row[GINET.DP_IDX] - (row[GINET.DP_IDX] - row[GINET.CURAVAIL_IDX])

                    M_EXCEPT = row[GINET.EXCEPTQ_IDX] - (row[GINET.DP_IDX] - row[GINET.CURAVAIL_IDX])
                
                #--! X.EXCEPT< X.DP - X.CURAVAIL !--
                else:
                    #--!  GIDP = X.EXCEPTQ , MPDP =  DP - (X.EXCEPTQ) !--
                    row[GINET.GIDP_IDX] = row[GINET.EXCEPTQ_IDX]
                    row[GINET.MPDP_IDX] = row[GINET.DP_IDX] - row[GINET.EXCEPTQ_IDX]

                    M_EXCEPT = 0
        #--! 두번째 ROW부터 !--
        else:
            #--! AVAIL 수량 > DP 일 경우 !--
            if (row[GINET.CURAVAIL_IDX] - row[GINET.DP_IDX]) >= 0: # IF X.CURAVAIL - X.DP >=0 THEN
                #--! 전량 MPDP, GIDP 0 !--
                row[GINET.GIDP_IDX] = 0
                row[GINET.MPDP_IDX] = row[GINET.DP_IDX]
                row[GINET.CUREXCEPT_IDX] = M_EXCEPT

                M_EXCEPT = M_EXCEPT
            else:
                #--! CUREXCEPT = M_EXCEPT !--
                row[GINET.CUREXCEPT_IDX] = M_EXCEPT

                #--! DP 가 남았을 경우 !--
                if(M_EXCEPT >= (row[GINET.DP_IDX] - row[GINET.CURAVAIL_IDX])):

                    # --!  GIDP = X.DP - X.CURAVAIL , DP - (X.DP - X.CURAVAIL) !--
                    row[GINET.GIDP_IDX] = row[GINET.DP_IDX] - row[GINET.CURAVAIL_IDX]
                    row[GINET.MPDP_IDX] = row[GINET.DP_IDX] - (row[GINET.DP_IDX] - row[GINET.CURAVAIL_IDX])

                    M_EXCEPT = M_EXCEPT - (row[GINET.DP_IDX] - row[GINET.CURAVAIL_IDX])
                else:
                    row[GINET.GIDP_IDX] = M_EXCEPT
                    row[GINET.MPDP_IDX] = row[GINET.DP_IDX] - M_EXCEPT

                    M_EXCEPT = 0

    logger.Step(3, 'GINETTING OPEN End')
    #endregion logger.Step(3, 'GINETTING OPEN')
    # print(f"{time.time() - sub_start_time_3:.5f} sec")
    # sub_start_time_4 = time.time()
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

    dfMST_GINETTING = pd.DataFrame(nparr_ginet)
    dfMST_GINETTING.columns = ['PLANID', 'SITEID', 'ITEM', 'WEEK', 'RTF', 'GI', 'EXCEPTQ', 'AVAIL', 'DP', 'CUREXCEPT', 'CURAVAIL', 'GIDP', 'MPDP', 'REMAINGIDP','UPDATEYN', 'RK']
    dfMST_GINETTING = dfMST_GINETTING.astype({'RTF':'int32', 'GI':'int32', 'EXCEPTQ':'int32', 'AVAIL':'int32', 'DP':'int32', 'CUREXCEPT':'int32', 'CURAVAIL':'int32', 'GIDP':'int32', 'MPDP':'int32', 'REMAINGIDP':'float64', 'RK':'int32'})

    dfMST_GINETTING_SITE = pd.DataFrame(nparr_ginet)
    dfMST_GINETTING_SITE.columns = ['PLANID', 'SITEID', 'ITEM', 'WEEK', 'RTF', 'GI', 'EXCEPTQ', 'AVAIL', 'DP', 'CUREXCEPT', 'CURAVAIL', 'GIDP', 'MPDP', 'REMAINGIDP', 'UPDATEYN', 'RK']
    dfMST_GINETTING_SITE = dfMST_GINETTING_SITE.astype({'RTF':'int32', 'GI':'int32', 'EXCEPTQ':'int32', 'AVAIL':'int32', 'DP':'int32', 'CUREXCEPT':'int32', 'CURAVAIL':'int32', 'GIDP':'int32', 'MPDP':'int32', 'REMAINGIDP':'float64', 'RK':'int32'})

    dfMST_GINETTING_SITE = dfMST_GINETTING_SITE.loc[dfMST_GINETTING_SITE['PLANID'] == V_PLANID]
    dfMST_GINETTING_SITE = dfMST_GINETTING_SITE.loc[dfMST_GINETTING_SITE['GIDP'] > 0]

    df_SITE = dfMST_GINETTING_SITE.sort_values(by=['SITEID', 'ITEM', 'WEEK',]).reset_index()
    df_SITE = df_SITE[['SITEID', 'ITEM', 'WEEK', 'GIDP']]
    df_SITE.columns = ['SITEID', 'ITEM', 'WEEK', 'GIDP'] 
    df_SITE['WEEK'] = df_SITE['WEEK'].fillna(0).astype('int32')


    #--기여도 높은 순으로 NOW DP를 발췌
    df_DP1 = df_Inbound.copy(deep=True)

    # 형전환
    df_DP1[ND_.QTYPROMISED] = df_DP1[ND_.QTYPROMISED].astype('float').astype('int32')
    df_DP1[ND_.SOLINENUM] = df_DP1[ND_.SOLINENUM].fillna(0).astype('int32')
    df_MST_GINETING_SALES['EXCEPTQ'] = df_MST_GINETING_SALES['EXCEPTQ'].fillna(0).astype('int32')
    df_MST_GINETING_SALES['CUREXCEPT'] = df_MST_GINETING_SALES['CUREXCEPT'].fillna(0).astype('int32')
    df_MST_GINETING_SALES['RNK'] = df_MST_GINETING_SALES['RNK'].fillna(0).astype('int32')

    df_MST_GINETING_SALES['UPBY'] = ''
    df_MST_GINETING_SALES = df_MST_GINETING_SALES.astype({'UPBY':'str'})

    df_MST_GINETING_SALES.rename(columns={
        GUI_SALESRESULTSMM.ITEM:'ITEM',
        GUI_SALESRESULTSMM.SALESID:'SALESID', 
        GUI_SALESRESULTSMM.SITEID:'SITEID'
    }, inplace=True)


    # # WHERE  A.PLANID = V_PLANID
    # # AND    QTYPROMISED>0
    # # AND    A.SOLINENUM < 200
    df_DP1 = df_DP1.loc[(df_DP1[ND_.PLANID] == V_PLANID) & (df_DP1[ND_.QTYPROMISED] > 0) & (df_DP1[ND_.SOLINENUM] < 200) ]

    df_DP1['PROMISEDDELDATE2'] = pd.to_datetime(df_DP1[ND_.PROMISEDDELDATE]).dt.strftime('%G%V')
    df_DP1['PROMISEDDELDATE2'] = df_DP1['PROMISEDDELDATE2'].fillna(0).astype('int32')


    # AND    RK.EXCEPTQ > RK.CUREXCEPT
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

    df_DPSalesCurexcept = df_DP[[ND_.SITEID,ND_.ITEMID,ND_.SALESID,'EXCEPTQ','CUREXCEPT']].drop_duplicates().copy()
    df_DPSalesCurexcept['UPBY'] = ''
    df_DPSalesCurexcept['UPBY'] = df_DPSalesCurexcept['UPBY'].astype('str')
    df_DPSalesCurexcept['UPBY'] = df_DPSalesCurexcept['UPBY'].fillna('')

    # list_Sales_loop_update를 dict로 변경
    dict_dp_sales_curexcept = {}
    for row in df_DPSalesCurexcept.values:
        # key = SITE, ITEM, SALESID
        key = row[0]+'::'+row[1]+'::'+row[2]
        dict_dp_sales_curexcept[key] = [row[0], row[1], row[2], int(row[3]), int(row[4]), row[5]]

    list_Outbound_loop_insert = np.empty((0,colcount), dtype=object)
    list_Outbound_loop_update = np.empty((0,5), dtype=object)
    list_Outbound_loop_update_sol = np.empty((0,5), dtype=object)
    # list_Sales_loop_update = np.empty((0,6), dtype=object)
    list_Ginet_loop_update = np.empty((0,6), dtype=object)

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
        key = row[ND_.SITEID_IDX]+'::'+row[ND_.ITEMID_IDX]+'::'+str(row[ND_.PROMISEDDELDATE2_IDX])
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
        V_GIDP = int(site_row[SITE1.GIDP_IDX])

        key = site_row[SITE1.SITEID_IDX] + '::' + site_row[SITE1.ITEM_IDX] + '::' + str(site_row[SITE1.WEEK_IDX])

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
                # if(V_EXCEPTQ > 0):

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
                        # dict_dp_sales_curexcept[dp_row[ND_.SITEID_IDX]+'::'+dp_row[ND_.ITEMID_IDX]+'::'+dp_row[ND_.SALESID_IDX]] = [dp_row[ND_.SITEID_IDX],dp_row[ND_.ITEMID_IDX],dp_row[ND_.SALESID_IDX],int(row_sales_curexcept[3]),int(row_sales_curexcept[4]) + int(dp_row[ND_.QTYPROMISED_IDX]), row_sales_curexcept[5]+'_1']

                        V_REMAINGIDP += int(dp_row[ND_.QTYPROMISED_IDX])
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
                        # dict_dp_sales_curexcept[dp_row[ND_.SITEID_IDX]+'::'+dp_row[ND_.ITEMID_IDX]+'::'+dp_row[ND_.SALESID_IDX]] = [dp_row[ND_.SITEID_IDX],dp_row[ND_.ITEMID_IDX],dp_row[ND_.SALESID_IDX],int(row_sales_curexcept[3]),int(row_sales_curexcept[4]) + int(V_GIDP), row_sales_curexcept[5]+'_2']
                        # list_Sales_loop_update = np.append(list_Sales_loop_update, np.array([[V_PLANID, dp_row[DP.SITEID_IDX], dp_row[DP.ITEM_IDX], dp_row[DP.SALESID_IDX], int(dp_row[DP.CUREXCEPT_IDX]) + int(V_GIDP),'_2']]), axis=0)

                        V_REMAINGIDP += int(V_GIDP)
                        # --남은 기여 수량 
                        V_EXCEPTQ -= int(V_GIDP)
                        # --남은  SITE.GIDP
                        V_GIDP = 0

                    # --1,3 case : sales.gidp만큼만 넘긴다.
                    elif(((int(V_GIDP) >= int(dp_row[ND_.QTYPROMISED_IDX])) & (int(dp_row[ND_.QTYPROMISED_IDX]) >= int(V_EXCEPTQ)))
                        |((int(dp_row[ND_.QTYPROMISED_IDX]) >= int(V_GIDP)) & (int(V_GIDP) >= int(V_EXCEPTQ)))):

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
                        # dict_dp_sales_curexcept[dp_row[ND_.SITEID_IDX]+'::'+dp_row[ND_.ITEMID_IDX]+'::'+dp_row[ND_.SALESID_IDX]] = [dp_row[ND_.SITEID_IDX],dp_row[ND_.ITEMID_IDX],dp_row[ND_.SALESID_IDX],int(row_sales_curexcept[3]),int(row_sales_curexcept[4]) + int(V_EXCEPTQ), row_sales_curexcept[5]+'_3']

                        V_REMAINGIDP += int(V_EXCEPTQ)
                        # --남은  SITE.GIDP
                        V_GIDP -= int(V_EXCEPTQ)
                        # --남은 기여 수량 
                        V_EXCEPTQ = 0

                    if(V_REMAINGIDP >= site_row[SITE1.GIDP_IDX]):
                        break

            for key, value in dict_dp_sales_curexcept2.items():
                dict_dp_sales_curexcept[key] = value

                
            if(site_row[SITE1.GIDP_IDX] > 0):
                list_Ginet_loop_update = np.append(list_Ginet_loop_update, 
                    np.array([[V_PLANID, site_row[SITE1.SITEID_IDX], site_row[SITE1.ITEM_IDX], str(site_row[SITE1.WEEK_IDX]), int(max(0, int(site_row[SITE1.GIDP_IDX]) - V_REMAINGIDP)), 'Y']]), 
                    axis=0)        


    df_sales_update_temp = DF.from_dict(dict_dp_sales_curexcept, orient='index')

    if(not df_sales_update_temp.empty) :
        df_sales_update_temp.columns = ['SITEID','ITEM','SALESID','EXCEPTQ_NEW','CUREXCEPT_NEW','UPBY_NEW']
        df_sales_update_temp['PLANID'] = V_PLANID
        df_ginet_sales_update = df_sales_update_temp.astype({'PLANID':'str', 'SITEID':'str', 'ITEM':'str', 'SALESID':'str','EXCEPTQ_NEW':'int32', 'CUREXCEPT_NEW':'int32', 'UPBY_NEW':'str'})
        # df_ginet_sales_update = df_ginet_sales_update.loc[df_ginet_sales_update['CUREXCEPT_NEW'] > 0]
        df_ginet_sales_update = df_ginet_sales_update[['PLANID','SITEID','ITEM','SALESID','CUREXCEPT_NEW','UPBY_NEW']]


    df_outbound_update      = pd.DataFrame(list_Outbound_loop_update)
    df_outbound_update_sol  = pd.DataFrame(list_Outbound_loop_update_sol)
    df_outbound_insert      = pd.DataFrame(list_Outbound_loop_insert)
    # df_ginet_sales_update   = pd.DataFrame(list_Sales_loop_update)
    df_ginet_update         = pd.DataFrame(list_Ginet_loop_update)

    dfMST_GINETTING.columns = ['PLANID', 'SITEID', 'ITEM', 'WEEK', 'RTF', 'GI', 'EXCEPTQ', 'AVAIL', 'DP', 'CUREXCEPT', 'CURAVAIL', 'GIDP', 'MPDP', 'REMAINGIDP', 'UPDATEYN', 'RK']
    dfMST_GINETTING = dfMST_GINETTING.astype({'PLANID':'str', 'SITEID':'str', 'ITEM':'str', 'WEEK':'str', 'RTF':'int32', 
        'GI':'int32', 'EXCEPTQ':'int32', 'AVAIL':'int32', 'DP':'int32', 'CUREXCEPT':'int32', 
        'CURAVAIL':'int32', 'GIDP':'int32', 'MPDP':'int32', 'REMAINGIDP':'float64', 'UPDATEYN':'str', 'RK':'int32'
        })
    # df_MST_GINETING_SALES['UPBY'] = ''
    # df_MST_GINETING_SALES = df_MST_GINETING_SALES.astype({'UPBY':'str'})

    df_outbound_update.columns = [ND_.SALESORDERID, ND_.PLANID, ND_.SOLINENUM, 'QTYPROMISED_NEW','UPBY_NEW']
    df_outbound_update_sol.columns = [ND_.SALESORDERID, ND_.PLANID, ND_.SOLINENUM, 'SOLINENUM_NEW','UPBY_NEW']
    df_outbound_insert.columns = ND_.LIST_COLUMN
    # df_ginet_sales_update.columns = ['PLANID', 'SITEID', 'ITEM', 'SALESID', 'CUREXCEPT_NEW','UPBY_NEW']
    df_ginet_update.columns = ['PLANID', 'SITEID', 'ITEM', 'WEEK', 'REMAINGIDP_NEW', 'UPDATEYN_NEW']


    df_outbound_update = df_outbound_update.astype({ND_.SALESORDERID:'str', ND_.PLANID:'str', ND_.SOLINENUM:'int32', 'QTYPROMISED_NEW':'int32', 'UPBY_NEW':'str'})
    df_outbound_update_sol = df_outbound_update_sol.astype({ND_.SALESORDERID:'str', ND_.PLANID:'str', ND_.SOLINENUM:'int32', 'SOLINENUM_NEW':'int32', 'UPBY_NEW':'str'})
    # df_ginet_sales_update = df_ginet_sales_update.astype({'PLANID':'str', 'SITEID':'str', 'ITEM':'str', 'SALESID':'str', 'CUREXCEPT_NEW':'int32', 'UPBY_NEW':'str'})
    df_ginet_update = df_ginet_update.astype({'PLANID':'str', 'SITEID':'str', 'ITEM':'str', 'WEEK':'str','REMAINGIDP_NEW':'float64', 'UPDATEYN_NEW':'str'})

    # df_ginet_sales_update2 = df_ginet_sales_update.groupby(by=['PLANID','SITEID','ITEM','SALESID'])[['CUREXCEPT_NEW']].sum()

    # df_MST_GINETING_SALES.rename(columns={
    #     GUI_SALESRESULTSMM.ITEM:'ITEM',
    #     GUI_SALESRESULTSMM.SALESID:'SALESID', 
    #     GUI_SALESRESULTSMM.SITEID:'SITEID'
    # }, inplace=True)


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
    df_ginet_merge = dfMST_GINETTING.merge(df_ginet_update2, how='left', on=['PLANID', 'SITEID', 'ITEM', 'WEEK']) 

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
    # print(f"SALES OPEN : {time.time() - sub_start_time_4:.5f} sec")
    # sub_start_time_5 = time.time()
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
    df_R = df_R[['SITEID', 'ITEM', 'WEEK', 'RTF', 'GI', 'DP', 'GIDP', 'REMAINGIDP', 'MPDP']]
    df_R = df_R.astype({'SITEID':'str', 'ITEM':'str', 'WEEK':'str', 'GI':'int32','DP':'int32', 'GIDP':'int32','MPDP':'int32'})
    df_R = df_R[df_R['GIDP'] > 0]
    # df_R['REMAINGIDP'] = df_R['REMAINGIDP'].astype('float').astype('int32')
    df_R['GIDP'] = df_R['REMAINGIDP'].fillna(df_R['GIDP'])
    df_R = df_R.sort_values(['SITEID', 'ITEM', 'WEEK'], ascending=[True,True,True])

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
        key = row[ND_.ITEMID_IDX]+'::'+row[ND_.SITEID_IDX]+'::'+str(row[ND_.PROMISEDDELDATE2_IDX])
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

        key = r_row[R.ITEM_IDX] + '::' + r_row[R.SITEID_IDX] + '::' + str(r_row[R.WEEK_IDX])

        if(key in dict_df_dp):
            dptuple = dict_df_dp.get(key)
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

    # df_Outbound_final_3 = pd.concat([df_Outbound_final_2, df_outbound_mege], axis=0, ignore_index=True)
    df_Outbound_final_4 = pd.concat([df_outbound_mege, df_outbound_insert], axis=0, ignore_index=True)
    df_Outbound_final_4[ND_.SOLINENUM] = df_Outbound_final_4[ND_.SOLINENUM].fillna(0).astype('float').astype('int32')
    # return df_Outbound_final_4


    logger.Step(5, 'site OPEN End')
    #endregion logger.Step(5, 'site OPEN')
    # print(f"site OPEN : {time.time() - sub_start_time_5:.5f} sec")
    # sub_start_time_6 = time.time()
    #region logger.Step(6, 'PRIORITY S SRC')
    logger.Step(6, 'PRIORITY S SRC Start')

    '''
    6.PRIORITY S SRC
    '''



    # AND     NOT EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
    df_priority = df_Outbound_final_4.loc[df_Outbound_final_4[ND_.PLANID] == V_PLANID]
    df_priority = df_priority[[ND_.PLANID, ND_.SALESORDERID,ND_.SOLINENUM,ND_.SITEID,ND_.ITEMID,ND_.DEMANDPRIORITY]]
    df_priority[ND_.SOLINENUM] = df_priority[ND_.SOLINENUM].fillna(0).astype('float').astype('int32')

    df_priority = df_priority.loc[
            df_priority[ND_.SOLINENUM].between(200, 299)
        ]

    df_priority[ND_.SOLINENUM] = df_priority[ND_.SOLINENUM].astype('str')
    df_Outbound_final_4[ND_.SOLINENUM] = df_Outbound_final_4[ND_.SOLINENUM].astype('str')

    df_priority[ND_.DEMANDPRIORITY] = df_priority[ND_.DEMANDPRIORITY].astype('str')

    df_priority = df_priority.loc[
            ~(df_priority[ND_.ITEMID] + df_priority[ND_.SITEID]).isin(
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
        df_priority[ND_.DEMANDPRIORITY].str[: start_pos]
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


    df_Outbound_final_5 = df_Outbound_final_5.reset_index(drop=True)

    # return df_Outbound_final_5

    logger.Step(6, 'PRIORITY S SRC End')
    #endregion logger.Step(6, 'PRIORITY S SRC')
    # print(f"PRIORITY S SRC : {time.time() - sub_start_time_6:.5f} sec")
    logger.Note('END GI Netting', 20)

    return df_Outbound_final_5[ND_.LIST_COLUMN], df_MST_GINETTING_final_1, df_MST_GINETING_SALES_final_1



# # '''
# # main 함수시작
# # '''

# logger = LOGGER('Netting')


# # 임시
# V_PLANID = '202526_P'
# V_PREPLANID = '202525'
# V_PLANWEEK = '202526'
# V_WEEK1 = '202526'
# V_WEEK4 = '202529'
# V_PREYEAR = '2025'
# V_PREYWEEK = '25' 

# df_Outbound = pd.DataFrame()
# df_Outbound_origin = pd.DataFrame()

# #gi_net_sell 전달용
# df_MST_GINETING_SALES = pd.DataFrame()

# df_Inbound = pd.DataFrame()
# df_in_Netting_Seller_Map = pd.DataFrame()
# df_in_Netting_Available_Resource_ODS_W = pd.DataFrame()
# df_in_Netting_ES_Item_Site_ODS_W = pd.DataFrame()
# df_in_Netting_Custom_Model_Map_ODS_W = pd.DataFrame()
# df_Sales_Result_Smm = pd.DataFrame()
# df_mst_site = pd.DataFrame()


# df_Inbound, df_in_Netting_Seller_Map, df_in_Netting_Available_Resource_ODS_W, \
# df_in_Netting_ES_Item_Site_ODS_W, df_in_Netting_Custom_Model_Map_ODS_W, df_Sales_Result_Smm, \
# df_mst_site, df_Outbound_origin = _make_gi_netting_dataframe( \
#                             df_Inbound, df_in_Netting_Seller_Map, df_in_Netting_Available_Resource_ODS_W, \
#                             df_in_Netting_ES_Item_Site_ODS_W, df_in_Netting_Custom_Model_Map_ODS_W, df_Sales_Result_Smm, \
#                             df_mst_site)


# # # 전체 수행시간 (csv 데이터 read , csv 파일생성 포함)
# all_start_time = time.time()

# df_Outbound, df_MST_GINETTING, df_MST_GINETING_SALES = do_gi_netting(df_Inbound, df_in_Netting_Seller_Map, df_in_Netting_Available_Resource_ODS_W, df_in_Netting_ES_Item_Site_ODS_W,
#         df_in_Netting_Custom_Model_Map_ODS_W, df_Sales_Result_Smm, df_mst_site, logger)

# #전체 수행시간
# print(f"{time.time() - all_start_time:.5f} sec")



# # # 중간테이블
# # # df_MST_GINETTING
# # # df_MST_GINETING_SALES
# df_MST_GINETTING = df_MST_GINETTING[['PLANID','SITEID','ITEM','WEEK','RTF','GI','EXCEPTQ','AVAIL','DP','CUREXCEPT','CURAVAIL','GIDP','MPDP']]
# df_MST_GINETTING = df_MST_GINETTING.sort_values(by=['SITEID','ITEM','WEEK'])
# df_MST_GINETTING.to_csv('MST_GINETTING.csv',index=False)
# df_MST_GINETING_SALES.to_csv('MST_GINETTING_SALES.csv',index=False)

# outbound_file_name = f"df_outbound최종_{V_PLANID}_DEV.csv"

# df_Outbound = df_Outbound.sort_values(by=[ND_.SALESORDERID,ND_.SOLINENUM])
# df_Outbound = df_Outbound[[ND_.SALESORDERID,ND_.SOLINENUM,ND_.QTYPROMISED, ND_.DEMANDPRIORITY,ND_.ITEMID, 
#                         ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE, ND_.AP1ID, ND_.AP2ID,
#                         ND_.BUILDAHEADTIME, ND_.BUSINESSTYPE,ND_.CHANNELRANK, ND_.CUSTOMERRANK, ND_.DEMANDTYPERANK, 
#                         ND_.GCID, ND_.GLOBALPRIORITY, ND_.LOCALPRIORITY, ND_.MAP_SATISFY_SS, ND_.MEASURETYPERANK,
#                         ND_.NO_SPLIT, ND_.PLANID,ND_.PREALLOC_ATTRIBUTE, ND_.PREFERENCERANK, ND_.PRODUCTRANK,
#                         ND_.REASONCODE,ND_.ROUTING_PRIORITY,ND_.SOPROMISEID,ND_.SALESLEVEL, ND_.SHIPTOID, 
#                         ND_.TIEBREAK, ND_.TIMEUOM, ND_.WEEKRANK]] 
# df_Outbound.to_csv(outbound_file_name,index=False)

# outbound_origin_file_name = f"df_outbound최종_{V_PLANID}_DB.csv"

# df_Outbound_origin = df_Outbound_origin.sort_values(by=[ND_.SALESORDERID,ND_.SOLINENUM])
# df_Outbound_origin = df_Outbound_origin[[ND_.SALESORDERID,ND_.SOLINENUM,ND_.QTYPROMISED, ND_.DEMANDPRIORITY,ND_.ITEMID, 
#                                         ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE, ND_.AP1ID, ND_.AP2ID,
#                                         ND_.BUILDAHEADTIME, ND_.BUSINESSTYPE,ND_.CHANNELRANK, ND_.CUSTOMERRANK, ND_.DEMANDTYPERANK, 
#                                         ND_.GCID, ND_.GLOBALPRIORITY, ND_.LOCALPRIORITY, ND_.MAP_SATISFY_SS, ND_.MEASURETYPERANK,
#                                         ND_.NO_SPLIT, ND_.PLANID,ND_.PREALLOC_ATTRIBUTE, ND_.PREFERENCERANK, ND_.PRODUCTRANK,
#                                         ND_.REASONCODE,ND_.ROUTING_PRIORITY,ND_.SOPROMISEID,ND_.SALESLEVEL, ND_.SHIPTOID, 
#                                         ND_.TIEBREAK, ND_.TIMEUOM, ND_.WEEKRANK]]
# df_Outbound_origin.to_csv(outbound_origin_file_name,index=False)


# # 데이터 타입을 문자열(str)로 변환
# df_Old_Rename = df_Outbound_origin.astype(str)
# df_New = df_Outbound.astype(str)

# # df_Old_Rename[ND_.OPTION_CODE] = df_Old_Rename[ND_.OPTION_CODE].str.replace('.0', '', regex=False)
# # df_New[ND_.OPTION_CODE] = df_New[ND_.OPTION_CODE].str.replace('.0', '', regex=False)

# # 차집합을 사용하여 차이 검증
# set_Old = set([tuple(row) for row in df_Old_Rename.values])
# set_New = set([tuple(row) for row in df_New.values])

# # OLD 기준에서 NEW 차집합 : set_Old - set_New
# oldBase_Row = set_Old - set_New
# df_OldBase = pd.DataFrame(list(oldBase_Row), columns=df_Old_Rename.columns)

# # NEW 기준에서 OLD 차집합 : set_New - set_Old
# newBase_Row = set_New - set_Old
# df_NewBase = pd.DataFrame(list(newBase_Row), columns=df_Old_Rename.columns)

# # 차집합 결과 UNION 처리
# df_Diff = pd.concat([df_OldBase.assign(Status="OLD"), df_NewBase.assign(Status="NEW")])

# df_Diff = df_Diff.sort_values(by=[ND_.SALESORDERID,ND_.SOLINENUM,'Status']).drop_duplicates()
# df_Diff = df_Diff[['Status',ND_.SALESORDERID,ND_.SOLINENUM,ND_.QTYPROMISED, ND_.DEMANDPRIORITY,ND_.ITEMID, 
#                         ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE, ND_.AP1ID, ND_.AP2ID,
#                         ND_.BUILDAHEADTIME, ND_.BUSINESSTYPE,ND_.CHANNELRANK, ND_.CUSTOMERRANK, ND_.DEMANDTYPERANK, 
#                         ND_.GCID, ND_.GLOBALPRIORITY, ND_.LOCALPRIORITY, ND_.MAP_SATISFY_SS, ND_.MEASURETYPERANK,
#                         ND_.NO_SPLIT, ND_.PLANID,ND_.PREALLOC_ATTRIBUTE, ND_.PREFERENCERANK, ND_.PRODUCTRANK,
#                         ND_.REASONCODE,ND_.ROUTING_PRIORITY,ND_.SOPROMISEID,ND_.SALESLEVEL, ND_.SHIPTOID, 
#                         ND_.TIEBREAK, ND_.TIMEUOM, ND_.WEEKRANK]]
# df_Diff.to_csv(f'ginet_Diff_{V_PLANID}.csv',index=False)

# # 검증결과 출력
# print('====================================================================================')
# print(f'OLD / NEW 전체건수: {len(df_Old_Rename)} / {len(df_New)}')
# print(f'OLD - NEW 차이건수: {len(df_OldBase)}')
# print(f'NEW - OLD 차이건수: {len(df_NewBase)}')
# print('====================================================================================')



# # '''
# # main 함수 끝
# # '''
