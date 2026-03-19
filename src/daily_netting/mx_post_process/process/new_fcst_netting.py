# %%
# 모듈 로딩
import pandas as pd
import numpy as np
import time

#오라클과 파이썬 부동소수점 계산 차이로 추가
import decimal
decimal.getcontext().prec = 40


from daily_netting.mx_post_process.constant.ods_constant import NettedDemandD as ND_    # EXP_SOPROMISE
from daily_netting.mx_post_process.constant.mx_constant import MXItemSellerMap as SM    # V_MTA_SELLERMAP 
from daily_netting.mx_post_process.constant.mx_constant import CustomModelMap as CMM    # MTA_CUSTOMMODELMAP
from daily_netting.mx_post_process.constant.mx_constant import InventoryD as Inv_    # MST_INVENTORY
from daily_netting.mx_post_process.constant.mx_constant import IntransitD as Int_    # MST_INTRANSIT
from daily_netting.mx_post_process.constant.mx_constant import DeliveryPlanD as DP_    # EXP_DELIVERYPLAN
from daily_netting.mx_post_process.constant.mx_constant import ShortReasonD as SR_    # EXP_SHORTREASON
from daily_netting.mx_post_process.constant.mx_constant import DistributionOrdersD as DO_    # MST_DISTRIBUTIONORDERS
from daily_netting.mx_post_process.constant.mx_constant import CodeMap as CM    # MTA_CODEMAP
from daily_netting.mx_post_process.constant.mx_constant import AvailableResourceD as AR_    # MST_INVENTORY_FNE
from daily_netting.mx_post_process.constant.mx_constant import ESItemSite as ESIS    # MVES_ITEMSITE
from daily_netting.mx_post_process.constant.mx_constant import NewFCSTPeriod as NFP    # MTA_NEWFCSTNETTING
from daily_netting.mx_post_process.constant.dim_constant import Item as VIA

from daily_netting.mx_post_process.utils.NSCMCommon import G_Logger
LOGGER = G_Logger
DF = pd.DataFrame

v_planid = None
v_effstartdate = None
v_week1 = None
v_week4 = None
v_horisonweek = None
v_endweek = None

find_priority_position = None

def set_new_fcst_env(accessor: object) -> None:
    '''동적으로 전역변수를 설정하기 위한 함수'''
    global v_planid, v_effstartdate, v_week1, v_week4, v_horisonweek, v_endweek
    global find_priority_position
    
    find_priority_position = accessor.find_priority_position

    #입력 파라미터 및 기준 데이터 정의
    v_planid = accessor.plan_id
    v_effstartdate = accessor.start_date
    v_week1 = accessor.plan_week
    
    if accessor.plan_type == 'VPLAN':
        v_horisonweek = 1
        v_endweek = 7
        v_week4 = (v_effstartdate + pd.Timedelta(days=7*4)).strftime('%G%V')
    else:
        v_horisonweek = 0
        v_endweek = 0
        v_week4 = (v_effstartdate + pd.Timedelta(days=7*3)).strftime('%G%V')


#%%
# [markdown]
# #step 1. 4주이내 감소가 한주라도 있는 fcst에 대해 발라내기
# - site, item, week 기준이며, sales는 가용량때문에 생각하지 않는다..
# - 로직 적용하여 new fcst 분리후, exp_sopromisesrc 버프에서 sales도 감안하여 new fcst로 분리
# - 가용량   = DC + INTRANSIT + 확정선적(공장재고 +공장생산)
# - Realease = boddetail 의 max(frozon + LT) 이며, 최대 21을 넘지는 않는다.


# %%
def create_mst_newfcstnetting(
        v_planid, v_week1, v_week4, v_horisonweek, v_effstartdate, v_endweek
        , df_v_mta_sellermap
        , df_exp_sopromisesrc
        , df_exp_deliveryplan
        , df_exp_shortreason
        , df_mst_inventory
        , df_mst_intransit
        , df_mst_inventory_fne
        , df_mst_distributionorders
        , df_vui_itemattb
        , df_mta_codemap
        , df_mta_newfcstnetting
        , df_mta_custommodelmap
):

    ############# SP_FN_SUMMARYDATA 시작 ##################

    #1. WITH ITEMMASTER AS 시작

    df_sellermap_itemsite = df_v_mta_sellermap[SM.ITEM] + df_v_mta_sellermap[SM.SITEID]

    #ITEMMASTER
    #DATA_SRC
    ### 1-1
    df_exp_sopromisesrc_view = df_exp_sopromisesrc[
        ~(df_exp_sopromisesrc[ND_.ITEMID] + df_exp_sopromisesrc[ND_.SITEID]).isin(df_sellermap_itemsite)]
    df_exp_sopromisesrc_view = df_exp_sopromisesrc_view[[ND_.ITEMID, ND_.SITEID]].drop_duplicates()
    df_exp_sopromisesrc_view = df_exp_sopromisesrc_view.rename(columns={ND_.ITEMID:'ITEM', ND_.SITEID:'SITEID'})
    ### 1-2
    df_exp_deliveryplan_view = df_exp_deliveryplan[
        ~(df_exp_deliveryplan[DP_.ITEM] + df_exp_deliveryplan[DP_.SITEID]).isin(df_sellermap_itemsite)]
    df_exp_deliveryplan_view = df_exp_deliveryplan_view[[DP_.ITEM, DP_.SITEID]].drop_duplicates()
    df_exp_deliveryplan_view = df_exp_deliveryplan_view.rename(columns={DP_.ITEM:'ITEM', DP_.SITEID:'SITEID'})
    df_exp_deliveryplan_view
    ### 1-3
    df_exp_shortreason_view = df_exp_shortreason[df_exp_shortreason[SR_.PROBLEMID] == 1]
    df_exp_shortreason_view = df_exp_shortreason_view[df_exp_shortreason_view[SR_.PROBLEMTYPE] == 'NEW_FCST']
    df_exp_shortreason_view = df_exp_shortreason_view[
        ~(df_exp_shortreason_view[SR_.REQITEM] + df_exp_shortreason_view[SR_.REQSITEID]).isin(df_sellermap_itemsite)]
    df_exp_shortreason_view = df_exp_shortreason_view[[SR_.REQITEM, SR_.REQSITEID]].drop_duplicates()
    df_exp_shortreason_view = df_exp_shortreason_view.rename(columns={SR_.REQITEM:'ITEM', SR_.REQSITEID:'SITEID'})

    ### 1-4
    df_mst_inventory_view = df_mst_inventory[
        ~(df_mst_inventory[Inv_.ITEM] + df_mst_inventory[Inv_.SITEID]).isin(df_sellermap_itemsite)]
    df_mst_inventory_view = df_mst_inventory_view[[Inv_.ITEM, Inv_.SITEID]].drop_duplicates()
    df_mst_inventory_view = df_mst_inventory_view.rename(columns={Inv_.ITEM:'ITEM', Inv_.SITEID:'SITEID'})

    ### 1-5
    df_mst_intransit_view = df_mst_intransit[
        ~(df_mst_intransit[Int_.ITEM] + df_mst_intransit[Int_.TOSITEID]).isin(df_sellermap_itemsite)]
    df_mst_intransit_view = df_mst_intransit_view[[Int_.ITEM, Int_.TOSITEID]].drop_duplicates()
    df_mst_intransit_view = df_mst_intransit_view.rename(columns={Int_.ITEM:'ITEM', Int_.TOSITEID:'SITEID'})

    ### 1-6
    df_mst_distributionorders_view = df_mst_distributionorders[
        ~(df_mst_distributionorders[DO_.ITEM] + df_mst_distributionorders[DO_.TOSITEID]).isin(df_sellermap_itemsite)]
    df_mst_distributionorders_view = df_mst_distributionorders_view[[DO_.ITEM, DO_.TOSITEID]].drop_duplicates()
    df_mst_distributionorders_view = df_mst_distributionorders_view.rename(columns={DO_.ITEM:'ITEM', DO_.TOSITEID:'SITEID'})

    #UNION
    df_w_itemmaster = pd.concat([
        df_exp_sopromisesrc_view
        , df_exp_deliveryplan_view
        , df_exp_shortreason_view
        , df_mst_inventory_view
        , df_mst_intransit_view
        , df_mst_distributionorders_view], ignore_index=True)
    df_w_itemmaster = df_w_itemmaster[['ITEM', 'SITEID']].drop_duplicates(ignore_index=True)

    df_w_itemmaster = pd.merge(df_w_itemmaster, df_vui_itemattb, how='inner', left_on=['ITEM'], right_on=[VIA.ITEM])

    df_w_itemmaster = df_w_itemmaster.rename(columns={VIA.PRODUCTGROUP:'PRODUCTGROUP'})

    df_w_itemmaster = df_w_itemmaster[['PRODUCTGROUP', 'ITEM', 'SITEID']]

    #1. WITH ITEMMASTER AS 끝


    #2. (WITH), HORIZON AS 시작

    #MTA_CODEMAP 'HC_WL_DP_SEA_COMMIT' 조건 필터링
    df_mta_codemap_view = df_mta_codemap[df_mta_codemap[CM.CODEMAPKEY].str.split('::').str[0] =='HC_WL_DP_SEA_COMMIT']

    #MTA_NEWFCSTNETTING A, MTA_CODEMAP B 조인
    df_mta_newfcstnetting_merge = pd.merge(df_mta_newfcstnetting
                                        , df_mta_codemap_view, how='left'
                                        , left_on=[NFP.SITEID, NFP.PRODUCTGROUP]
                                        , right_on=[CM.TXT1, CM.TXT2])

    df_mta_newfcstnetting_merge = df_mta_newfcstnetting_merge[[NFP.SITEID, NFP.PRODUCTGROUP, NFP.ENDWEEK, CM.CODE1]]

    #df_mta_newfcstnetting_merge

    #이전 셀에서 생성한 ITEMMASTER와 조인
    df_w_itemmaster_merge = pd.merge(df_w_itemmaster
                                    , df_mta_newfcstnetting_merge, how='left'
                                    , left_on=['SITEID', 'PRODUCTGROUP']
                                    , right_on=[NFP.SITEID, NFP.PRODUCTGROUP])

    #AND   SUBSTR(ITEM, -3) = NVL(B.CODE1(+), SUBSTR(ITEM, -3)) 처리
    df_w_itemmaster_merge['item_sub'] = df_w_itemmaster_merge['ITEM'].str[-3:]
    df_w_itemmaster_merge = df_w_itemmaster_merge[
            df_w_itemmaster_merge['item_sub'] == df_w_itemmaster_merge[CM.CODE1].fillna(df_w_itemmaster_merge['item_sub'])
        ]

    #결과에서 추가 컬럼 구하기
    #NVL(B.PRODUCTGROUP, A.PRODUCTGROUP) PRODUCTGROUP
    df_w_itemmaster_merge['PRODUCTGROUP'] = df_w_itemmaster_merge[NFP.PRODUCTGROUP].fillna(df_w_itemmaster_merge['PRODUCTGROUP'])

    #NVL(B.SITEID, A.SITEID) SITEID
    df_w_itemmaster_merge['SITEID'] = df_w_itemmaster_merge[NFP.SITEID].fillna(df_w_itemmaster_merge['SITEID'])

    #NVL(B.ENDWEEK, 4)+V_HORISONWEEK ENDHORIZON
    df_w_itemmaster_merge['ENDHORIZON'] = df_w_itemmaster_merge[NFP.ENDWEEK].fillna(4).astype(int) + v_horisonweek

    #TO_CHAR(V_EFFSTARTDATE + 7*(NVL(B.ENDWEEK, 4)-1)+V_ENDWEEK,'IYYYIW') ENDWEEK 
    df_w_itemmaster_merge['ENDWEEK'] = (v_effstartdate + pd.to_timedelta(7 * ((df_w_itemmaster_merge[NFP.ENDWEEK].fillna(4).astype(int) - 1) + v_endweek), unit='D')).dt.strftime('%G%V')

    df_w_horizon = df_w_itemmaster_merge[['PRODUCTGROUP', 'ITEM', 'SITEID', 'ENDHORIZON', 'ENDWEEK']]

    #2. (WITH), HORIZON AS 끝


    #3. (WIth),DATA_SRC AS ( 시작 : UNION ALL 위부분
    df_custommodelmap_itemview = df_mta_custommodelmap[df_mta_custommodelmap[CMM.ISVALID] == 'Y']

    df_exp_deliveryplan_merge = df_exp_deliveryplan[
        ~(df_exp_deliveryplan[DP_.ITEM] + df_exp_deliveryplan[DP_.SITEID]).isin(df_sellermap_itemsite)]

    df_exp_deliveryplan_merge = df_exp_deliveryplan_merge[
            ~df_exp_deliveryplan_merge[DP_.ITEM].isin(df_custommodelmap_itemview[CMM.CUSTOMITEM])
        ]

    #조건 처리 : AND      TO_CHAR(PROMISEDSHIPPINGDATE,'IYYYIW') BETWEEN V_WEEK1  AND  B.ENDWEEK
    #PROMISEDSHIPPINGDATE week로 변환
    isoCalDate = pd.to_datetime(df_exp_deliveryplan_merge[DP_.PROMISEDSHIPPINGDATE]).dt.isocalendar()
    df_exp_deliveryplan_merge['PROMISEDSHIPPINGWEEK'] = isoCalDate.year.astype(str) + isoCalDate.week.astype(str).str.zfill(2)

    df_exp_deliveryplan_merge['CURQTY'] = 0
    df_exp_deliveryplan_merge['WEEKSUPPLY'] = 0

    df_exp_deliveryplan_merge = df_exp_deliveryplan_merge[[DP_.SITEID, DP_.ITEM, 'PROMISEDSHIPPINGWEEK', DP_.QTYOPEN, 'CURQTY', 'WEEKSUPPLY']]
    df_exp_deliveryplan_merge = df_exp_deliveryplan_merge.rename(columns={DP_.ITEM:'ITEM', DP_.SITEID:'SITEID', 'PROMISEDSHIPPINGWEEK':'WEEK', DP_.QTYOPEN:'PREQTY'})

    #3. (WIth),DATA_SRC AS ( 블럭의  UNION ALL 아래 부분

    #sellermap not exists 조건 처리
    df_exp_sopromisesrc_merge = df_exp_sopromisesrc[
        ~(df_exp_sopromisesrc[ND_.ITEMID] + df_exp_sopromisesrc[ND_.SITEID]).isin(df_sellermap_itemsite)]
    #customermap not exists 조건 처리
    df_exp_sopromisesrc_merge = df_exp_sopromisesrc_merge[~df_exp_sopromisesrc_merge[ND_.ITEMID].isin(df_custommodelmap_itemview[CMM.CUSTOMITEM])]

    # #6.1s
    isoCalDate = pd.to_datetime(df_exp_sopromisesrc_merge[ND_.PROMISEDDELDATE]).dt.isocalendar()
    df_exp_sopromisesrc_merge['PROMISEDDELWEEK'] = isoCalDate.year.astype(str) + isoCalDate.week.astype(str).str.zfill(2)

    df_exp_sopromisesrc_merge['PREQTY'] = 0
    df_exp_sopromisesrc_merge['WEEKSUPPLY'] = 0

    df_exp_sopromisesrc_merge = df_exp_sopromisesrc_merge[[
        ND_.SITEID
        , ND_.ITEMID
        , 'PROMISEDDELWEEK'
        , 'PREQTY'
        , ND_.QTYPROMISED
        , 'WEEKSUPPLY']]
    df_exp_sopromisesrc_merge = df_exp_sopromisesrc_merge.rename(columns={
        ND_.SITEID : 'SITEID'
        , ND_.ITEMID : 'ITEM'
        , ND_.QTYPROMISED:'CURQTY'
        , 'PROMISEDDELWEEK':'WEEK'})

    #UNION ALL
    df_w_data_src = pd.concat([df_exp_deliveryplan_merge, df_exp_sopromisesrc_merge])

    #HORIZON 조인
    df_w_data_src = pd.merge(df_w_data_src, df_w_horizon, on=['SITEID', 'ITEM'])
    df_w_data_src = df_w_data_src[df_w_data_src['WEEK'].between(v_week1, df_w_data_src['ENDWEEK'])]

    #3. (WIth),DATA_SRC AS ( 끝


    #-- ###  4-1
    #-- 1. 가용량 (INVENTORY + INTRANSIT + DISTRIBUTIONORDERS)

    # V_MTA_SELLERMAP 및 MST_ITEM의 MultiIndex 생성
    sellermap_index = pd.MultiIndex.from_frame(df_v_mta_sellermap[[SM.ITEM, SM.SITEID]]) #df_sellermap_itemsite

    # Step 2: 서브 쿼리 작성 및 필터링
    # MST_INVENTORY_FNE 테이블 필터링
    df_mst_inventory_fne = df_mst_inventory_fne[
        (~pd.MultiIndex.from_frame(df_mst_inventory_fne[[AR_.ITEM, AR_.SITEID]]).isin(sellermap_index))
    ]


    #조인 : MST_INVENTORY_FNE A, HORIZON B
    df_mst_inventory_fne_merge = pd.merge(df_mst_inventory_fne, df_w_horizon, left_on=[AR_.SITEID, AR_.ITEM], right_on=['SITEID', 'ITEM'])

    #AND A.WEEK BETWEEN V_WEEK1 AND B.ENDWEEK 
    df_mst_inventory_fne_merge = df_mst_inventory_fne_merge[
            df_mst_inventory_fne_merge[AR_.WEEK].between(v_week1, df_mst_inventory_fne_merge['ENDWEEK'])
        ]

    df_mst_inventory_fne_merge['PREQTY'] = 0
    df_mst_inventory_fne_merge['CURQTY'] = 0

    df_mst_inventory_fne_merge = df_mst_inventory_fne_merge[[AR_.SITEID, AR_.ITEM, AR_.WEEK, 'PREQTY', 'CURQTY', AR_.QTY]]
    df_mst_inventory_fne_merge = df_mst_inventory_fne_merge.rename(columns={
        AR_.SITEID:'SITEID', AR_.ITEM:'ITEM', AR_.WEEK:'WEEK', AR_.QTY:'WEEKSUPPLY'})


    #-- ###  4-2
    #DATA_SRC    UNION ALL
    df_w_data_src_inv = pd.concat([df_w_data_src, df_mst_inventory_fne_merge])

    #WHERE  NOT EXISTS (SELECT 'X' FROM MST_ITEM WHERE PRODUCTCODE = 'EBABX' AND  ITEM = A.ITEM)
    df_mst_item_sub = df_vui_itemattb[df_vui_itemattb[VIA.PRODUCTCODE] == 'EBABX']
    df_w_data_src_inv = df_w_data_src_inv[~df_w_data_src_inv['ITEM'].isin(df_mst_item_sub[VIA.ITEM])]

    df_w_data_src_inv = df_w_data_src_inv[['SITEID', 'ITEM', 'WEEK', 'PREQTY', 'CURQTY', 'WEEKSUPPLY']]

    df_w_data_src_inv = df_w_data_src_inv.astype({
        'PREQTY': int,
        'CURQTY': int,
        'WEEKSUPPLY': int,
    })


    #-- ###  4-3
    #GROUP BY PLANID, SITEID, ITEM, WEEK
    df_w_data_src_inv = df_w_data_src_inv.groupby(['SITEID', 'ITEM', 'WEEK']).agg({
        'PREQTY': 'sum',
        'CURQTY': 'sum',
        'WEEKSUPPLY': 'sum'
    }).reset_index()

    #-- ###  4-4 : EXP_SHORTREASON B블럭 처리
    df_exp_shortreason_view = df_exp_shortreason[df_exp_shortreason[SR_.PROBLEMID] == 1]
    df_exp_shortreason_view = df_exp_shortreason_view[df_exp_shortreason_view[SR_.PROBLEMTYPE] == 'NEW_FCST']

    #not exists MST_ITEM 조인 조건 처리
    df_exp_shortreason_view = df_exp_shortreason_view[~df_exp_shortreason_view[SR_.REQITEM].isin(df_mst_item_sub)]

    #not exists V_MTA_SELLERMAP 조인 조건 처리
    df_exp_shortreason_view = df_exp_shortreason_view[
        ~(df_exp_shortreason_view[SR_.REQITEM] + df_exp_shortreason_view[SR_.REQSITEID]).isin(df_sellermap_itemsite)]

    #net exists MTA_CUSTOMMODELMAP 조인 조건 처리
    #프로시저에는 WHERE CUSTOMITEM = A.ITEM인데 잘못된 거라함. REQITEM이 맞다함
    df_exp_shortreason_view = df_exp_shortreason_view[~df_exp_shortreason_view[SR_.REQITEM].isin(df_custommodelmap_itemview[CMM.CUSTOMITEM])]

    df_exp_shortreason_view['WEEK'] = pd.to_datetime(df_exp_shortreason_view[SR_.DUEDATE]).dt.strftime('%G%V')

    df_exp_shortreason_view = df_exp_shortreason_view[[SR_.REQSITEID, SR_.REQITEM, 'WEEK', SR_.SHORTQTY]]

    #그룹핑 및 SUM
    df_exp_shortreason_view = df_exp_shortreason_view.groupby([SR_.REQSITEID, SR_.REQITEM, 'WEEK']).agg({
        SR_.SHORTQTY: 'sum'
    }).reset_index()

    df_exp_shortreason_view = df_exp_shortreason_view.rename(columns={SR_.REQSITEID:'SITEID', SR_.REQITEM:'ITEM', SR_.SHORTQTY:'NFSHORT'})

    #조인 : VUI_ITEMATTB C 
    df_mst_newfcstnetting = pd.merge(df_w_data_src_inv, df_vui_itemattb, left_on=['ITEM'], right_on=[VIA.ITEM])
    #조인 : (EXP_SHORTREASON) B
    df_mst_newfcstnetting = pd.merge(df_mst_newfcstnetting, df_exp_shortreason_view, how='left', on=['SITEID', 'ITEM', 'WEEK'])

    df_mst_newfcstnetting = df_mst_newfcstnetting.rename(columns={VIA.ATTB05:'ATTB05', VIA.BASICNAME:'BASICNAME'})

    #AND    NOT EXISTS (SELECT 'X' FROM MTA_CODEMAP WHERE CATEGORY = 'HC_WL_FN_EXCEPNEWFCST' AND TXT1 = A.SITEID AND TXT2 = C.ATTB05 AND TXT3 = C.BASICNAME )
    #MTA_CODEMAP 'HC_WL_FN_EXCEPNEWFCST' 조건 필터링
    df_mta_codemap_fcstview = df_mta_codemap[df_mta_codemap[CM.CODEMAPKEY].str.split('::').str[0] =='HC_WL_FN_EXCEPNEWFCST']
    idx_codemap = pd.MultiIndex.from_frame(df_mta_codemap_fcstview[[CM.TXT1, CM.TXT2, CM.TXT3]])

    df_mst_newfcstnetting = df_mst_newfcstnetting[
        ~pd.MultiIndex.from_frame(df_mst_newfcstnetting[['SITEID','ATTB05','BASICNAME']]).isin(idx_codemap)
    ]

    df_mst_newfcstnetting['NFSHORT'] = df_mst_newfcstnetting['NFSHORT'].fillna(0).astype(int)
    df_mst_newfcstnetting = df_mst_newfcstnetting[['SITEID', 'ITEM', 'WEEK', 'PREQTY', 'CURQTY', 'NFSHORT', 'WEEKSUPPLY']]

    df_mst_newfcstnetting = df_mst_newfcstnetting.groupby(['SITEID', 'ITEM', 'WEEK']).agg({
        'PREQTY': 'sum',
        'CURQTY': 'sum',
        'NFSHORT': 'sum',
        'WEEKSUPPLY': 'sum'
    }).reset_index()

    #정렬
    df_mst_newfcstnetting = df_mst_newfcstnetting.sort_values(by=['SITEID', 'ITEM', 'WEEK'])

    #컬럼 추가
    df_mst_newfcstnetting['DIFF'] = df_mst_newfcstnetting['CURQTY'] - df_mst_newfcstnetting['PREQTY']
    df_mst_newfcstnetting['EXCEPNFSHORT'] = df_mst_newfcstnetting['PREQTY'] - df_mst_newfcstnetting['NFSHORT']

    #V_PGMNAME               VARCHAR2(30) := 'SP_FN_SUMMARYDATA';
    #INSERT INTO MST_NEWFCSTNETTING
    #(  PLANID, SITEID, ITEM, WEEK, RELEASE, PREQTY, CURQTY, DIFF, NFSHORT, EXCEPNFSHORT, MRTF, ARTF, WEEKSUPPLY, MSUPPLY, ASUPPLY, MPDP, NEWDP, INITDTTM, INITBY )
    df_mst_newfcstnetting['RELEASE'] = 0
    df_mst_newfcstnetting['MRTF'] = 0
    df_mst_newfcstnetting['ARTF'] = 0
    df_mst_newfcstnetting['MSUPPLY'] = 0
    df_mst_newfcstnetting['ASUPPLY'] = 0
    df_mst_newfcstnetting['MPDP'] = 0
    df_mst_newfcstnetting['NEWDP'] = 0

    df_mst_newfcstnetting['PLANID'] = v_planid

    #컬럼 정리
    df_mst_newfcstnetting = df_mst_newfcstnetting[['PLANID', 'SITEID', 'ITEM', 'WEEK', 'PREQTY', 'CURQTY', 'NFSHORT', 'WEEKSUPPLY', 'DIFF', 'EXCEPNFSHORT', 'RELEASE', 'MRTF',
                            'ARTF', 'MSUPPLY', 'ASUPPLY', 'MPDP', 'NEWDP']]

    #AND SITEID LIKE 'S341%'
    df_mst_newfcstnetting_total = df_mst_newfcstnetting[df_mst_newfcstnetting['SITEID'].str.startswith('S341')]

    #df_mst_newfcstnetting_total = df_mst_newfcstnetting_total[['PLANID', 'SITEID', 'ITEM', 'WEEK', 'RELEASE', 'PREQTY', 'CURQTY','DIFF', 'NFSHORT', 'EXCEPNFSHORT', 'MRTF', 'ARTF', 'WEEKSUPPLY', 'MSUPPLY', 'ASUPPLY', 'MPDP', 'NEWDP']]

    df_mst_newfcstnetting_total = df_mst_newfcstnetting_total.groupby(['ITEM', 'WEEK']).agg({
            'RELEASE': 'sum',
            'PREQTY': 'sum',
            'CURQTY': 'sum',
            'DIFF': 'sum',
            'NFSHORT': 'sum',
            'EXCEPNFSHORT': 'sum',
            'MRTF': 'sum',
            'ARTF': 'sum',
            'WEEKSUPPLY': 'sum',
            'MSUPPLY': 'sum',
            'ASUPPLY': 'sum',
            'MPDP': 'sum',
            'NEWDP': 'sum'
    }).reset_index()


    df_mst_newfcstnetting_total['PLANID'] = v_planid
    df_mst_newfcstnetting_total['SITEID'] = 'TOTAL'

    df_mst_newfcstnetting_total = df_mst_newfcstnetting_total[['PLANID', 'SITEID', 'ITEM', 'WEEK', 
            'PREQTY', 'CURQTY', 'NFSHORT', 'WEEKSUPPLY', 'DIFF', 'EXCEPNFSHORT', 'RELEASE', 'MRTF', 'ARTF', 'MSUPPLY', 'ASUPPLY', 'MPDP', 'NEWDP']]


    #중복제거
    df_mst_newfcstnetting_total = df_mst_newfcstnetting_total.drop_duplicates()


    #df_mst_newfcstnetting_total = df_mst_newfcstnetting_total[['SITEID', 'ITEM', 'WEEK']].drop_duplicates()
    #df_mst_newfcstnetting_total = df_mst_newfcstnetting_total[['PLANID', 'SITEID', 'ITEM', 'WEEK', 'PREQTY', 'CURQTY', 'NFSHORT', 'WEEKSUPPLY', 'DIFF', 'EXCEPNFSHORT', 'RELEASE', 'MRTF',
    #                        'ARTF', 'MSUPPLY', 'ASUPPLY', 'MPDP', 'NEWDP']]

    #토탈을 만들어서 붙임
    df_mst_newfcstnetting = pd.concat([df_mst_newfcstnetting, df_mst_newfcstnetting_total], ignore_index=True)

    ##################### SP_FN_SUMMARYDATA 끝 #################


    ############ SP_FN_AVAILNETTING 시작 ################
    df_mst_newfcstnetting['WEEK'] = df_mst_newfcstnetting['WEEK'].astype(int)
    df_mst_newfcstnetting['RK'] = df_mst_newfcstnetting.groupby(['ITEM', 'SITEID'])['WEEK'].rank(method='first').astype(int)
    df_mst_newfcstnetting = df_mst_newfcstnetting.sort_values(by=['SITEID', 'ITEM', 'WEEK'])
    df_mst_newfcstnetting

    nparr_mst_newfcstnetting = df_mst_newfcstnetting.to_numpy()

    #컬럼 인덱스 번호 셋팅
    iRK = df_mst_newfcstnetting.columns.get_loc('RK')
    iWEEKSUPPLY = df_mst_newfcstnetting.columns.get_loc('WEEKSUPPLY')
    iEXCEPNFSHORT = df_mst_newfcstnetting.columns.get_loc('EXCEPNFSHORT')
    iCURQTY = df_mst_newfcstnetting.columns.get_loc('CURQTY')
    iMPDP = df_mst_newfcstnetting.columns.get_loc('MPDP')
    iNEWDP = df_mst_newfcstnetting.columns.get_loc('NEWDP')
    iMRTF = df_mst_newfcstnetting.columns.get_loc('MRTF')
    iARTF = df_mst_newfcstnetting.columns.get_loc('ARTF')
    iMSUPPLY = df_mst_newfcstnetting.columns.get_loc('MSUPPLY')
    iASUPPLY = df_mst_newfcstnetting.columns.get_loc('ASUPPLY')

    #초기화
    S_MRTF = 0
    S_ARTF = 0
    S_MSUPPLY = 0
    S_ASUPPLY = 0
    S_MPDP = 0
    S_NEWDP = 0

    for row in nparr_mst_newfcstnetting:
        if row[iRK] == 1:
            S_MRTF = 0
            S_ARTF = row[iEXCEPNFSHORT]
            S_MSUPPLY = 0
            S_ASUPPLY = row[iWEEKSUPPLY]
            S_MPDP = 0
            S_NEWDP = 0
        else:
            if S_ARTF > 0 :
                if S_MPDP > 0 :
                    if S_ARTF > S_MPDP :
                        S_MRTF = S_ARTF - S_MPDP
                    else: 
                        S_MRTF = 0
                else:
                    S_MRTF = S_ARTF
            else:   
                S_MRTF = 0 
            
            S_ARTF = row[iEXCEPNFSHORT] + S_MRTF
            
            if S_ASUPPLY > 0 :
                if S_MPDP > 0 :
                    if S_ASUPPLY > S_MPDP :
                        S_MSUPPLY = S_ASUPPLY - S_MPDP
                    else: 
                        S_MSUPPLY = 0 
                else:
                    S_MSUPPLY = S_ASUPPLY
            else:   
                S_MSUPPLY = 0
            S_ASUPPLY = row[iWEEKSUPPLY] + S_MSUPPLY

        if row[iCURQTY] > 0 :
            if S_ARTF > 0 :
                if row[iCURQTY] >= S_ARTF :
                    if S_ASUPPLY > 0 :
                        if S_ARTF >= S_ASUPPLY :
                            S_MPDP = S_ARTF
                        else:
                            if S_ASUPPLY >= row[iCURQTY] :
                                S_MPDP = row[iCURQTY]
                            else: 
                                S_MPDP = S_ASUPPLY
                    else:
                        S_MPDP = S_ARTF
                else:
                    S_MPDP = row[iCURQTY]
            else:
                if S_ASUPPLY > 0 :
                    if  row[iCURQTY] >= S_ASUPPLY : 
                        S_MPDP = S_ASUPPLY
                    else:
                        S_MPDP = row[iCURQTY]
                else:
                    S_MPDP = 0
        else:
            S_MPDP = 0

        S_NEWDP = row[iCURQTY] - S_MPDP  

        row[iMPDP]   = S_MPDP
        row[iNEWDP]  = S_NEWDP
        row[iMRTF]   = S_MRTF
        row[iARTF]   = S_ARTF
        row[iMSUPPLY] = S_MSUPPLY
        row[iASUPPLY] = S_ASUPPLY
        #list_newfcstnetting_result.append(row)

    #ndArray -> df
    df_mst_newfcstnetting = pd.DataFrame(nparr_mst_newfcstnetting, columns=df_mst_newfcstnetting.columns)

    #루프 내에서 사용하려고 미리 만들어 놓음
    df_mst_newfcstnetting['ORI_NEWDP'] = 0

    #첫번째 루프문 대상
    df_mst_newfcstnetting_total = df_mst_newfcstnetting[(df_mst_newfcstnetting['SITEID'] == 'TOTAL')][['SITEID','ITEM','WEEK','NEWDP']]
    ndarr_mst_newfcstnetting_total = df_mst_newfcstnetting_total.to_numpy()

    #두번째 루프문 전체 대상 - 첫번째 루프 돌면서 다시한번 더 필터링한다.
    df_mst_newfcstnetting_s341_sub = df_mst_newfcstnetting[df_mst_newfcstnetting['SITEID'].isin(['S341', 'S341WC74'])]
    #SUM(NEWDP) OVER (PARTITION BY ITEM, WEEK) TOTNEWDP 루프 내에서 처리
    #df_mst_newfcstnetting_s341_sub['TOTNEWDP'] = df_mst_newfcstnetting.groupby(['ITEM', 'WEEK'])['NEWDP'].transform('sum')

    ndarr_mst_newfcstnetting_s341_sub = df_mst_newfcstnetting_s341_sub.to_numpy()

    #계산 결과를 저장하기 위한 배열 - 두번째 루프 대상 조건에 해당되지 않는 부분만 남겨두고, 계산된 결과를 추가하려고 생성함.
    df_mst_newfcstnetting_result = df_mst_newfcstnetting[~df_mst_newfcstnetting['SITEID'].isin(['S341', 'S341WC74'])]
    ndarr_mst_newfcstnetting_result = df_mst_newfcstnetting_result.to_numpy()

    # IDX_SITEID = df_mst_newfcstnetting_total.columns.get_loc('SITEID')
    IDX_ITEM = df_mst_newfcstnetting_total.columns.get_loc('ITEM')
    IDX_WEEK = df_mst_newfcstnetting_total.columns.get_loc('WEEK')
    IDX_NEWDP = df_mst_newfcstnetting_total.columns.get_loc('NEWDP')

    ssIDX_SITEID = df_mst_newfcstnetting_s341_sub.columns.get_loc('SITEID')
    ssIDX_ITEM = df_mst_newfcstnetting_s341_sub.columns.get_loc('ITEM')
    ssIDX_WEEK = df_mst_newfcstnetting_s341_sub.columns.get_loc('WEEK')
    ssIDX_NEWDP = df_mst_newfcstnetting_s341_sub.columns.get_loc('NEWDP')
    ssIDX_ORI_NEWDP = df_mst_newfcstnetting_s341_sub.columns.get_loc('ORI_NEWDP')
    #ssIDX_TOTNEWDP = df_mst_newfcstnetting_s341_sub.columns.get_loc('TOTNEWDP')

    TOTNEWDP = 0
    list_newfcstnetting_result = []
    for s_row in ndarr_mst_newfcstnetting_total:
        V_SITE1_NEWDP = 0
        V_SITE2_NEWDP = 0
        
        #ITEM, WEEK 필터링
        ndarr_mst_newfcstnetting_ss = ndarr_mst_newfcstnetting_s341_sub[
            (ndarr_mst_newfcstnetting_s341_sub[:, ssIDX_ITEM] == s_row[IDX_ITEM]) & 
            (ndarr_mst_newfcstnetting_s341_sub[:, ssIDX_WEEK] == s_row[IDX_WEEK]) ]
        
        #SUM(NEWDP) OVER (PARTITION BY ITEM, WEEK) TOTNEWDP
        TOTNEWDP = ndarr_mst_newfcstnetting_ss[:, ssIDX_NEWDP].sum()

        for ss_row in ndarr_mst_newfcstnetting_ss:

            if ss_row[ssIDX_SITEID] == 'S341':
                if s_row[IDX_NEWDP] > 0 and (ss_row[ssIDX_NEWDP] + TOTNEWDP) == 0:
                    V_SITE1_NEWDP = np.ceil(s_row[IDX_NEWDP] / 2)
                elif s_row[IDX_NEWDP] > 0 and (ss_row[ssIDX_NEWDP] + TOTNEWDP) > 0:
                    V_SITE1_NEWDP = round(s_row[IDX_NEWDP] * ss_row[ssIDX_NEWDP] / TOTNEWDP)

                ss_row[ssIDX_ORI_NEWDP] = ss_row[ssIDX_NEWDP]
                ss_row[ssIDX_NEWDP] = V_SITE1_NEWDP
            else:
                if s_row[IDX_NEWDP] > 0 and (ss_row[ssIDX_NEWDP] + TOTNEWDP) == 0:
                    V_SITE2_NEWDP = np.floor(s_row[IDX_NEWDP] / 2)
                elif s_row[IDX_NEWDP] > 0 and ss_row[ssIDX_NEWDP] + TOTNEWDP > 0:
                    V_SITE2_NEWDP = round(s_row[IDX_NEWDP] * ss_row[ssIDX_NEWDP] / TOTNEWDP)

                ss_row[ssIDX_ORI_NEWDP] = ss_row[ssIDX_NEWDP]
                ss_row[ssIDX_NEWDP] = V_SITE2_NEWDP

            list_newfcstnetting_result.append(ss_row)

    ndarr_mst_newfcstnetting_result = np.vstack([ndarr_mst_newfcstnetting_result, list_newfcstnetting_result])
    df_mst_newfcstnetting_result = pd.DataFrame(ndarr_mst_newfcstnetting_result, columns=df_mst_newfcstnetting_result.columns)

    return df_mst_newfcstnetting_result

############ SP_FN_AVAILNETTING 끝 ################



######################################################################




# %%
def do_new_fcst_netting(
    df_v_mta_sellermap
    , df_exp_sopromisesrc
    , df_exp_deliveryplan
    , df_exp_shortreason
    , df_mst_inventory
    , df_mst_intransit
    , df_mst_inventory_fne
    , df_mst_distributionorders
    , df_vui_itemattb
    , df_mta_codemap
    , df_mta_newfcstnetting
    , df_mta_custommodelmap
    , df_mves_itemsite 
    , logger: LOGGER
):

    #실행시작 시간
    #start_time = time.time()
    logger.Note('Start NEW FCST Netting', 20)

    #전체 컬럼 인덱스를 저장해 놓음 - 원본과 동일한 컬럼으로 발췌하기 위해
    idx_exp_sopromisesrc = df_exp_sopromisesrc.columns

    # UPBY 컬럼 추가 - 로직 처리시 데이터 이동 확인용
    if not 'UPBY' in df_exp_sopromisesrc.columns:
        df_exp_sopromisesrc['UPBY'] = np.nan
        df_exp_sopromisesrc['UPBY'] = df_exp_sopromisesrc['UPBY'].astype(str)


    #형변환 및 필요 값 추가
    #object -> float -> int
    df_exp_shortreason[SR_.SHORTQTY] = pd.to_numeric(df_exp_shortreason[SR_.SHORTQTY], errors='coerce').astype(int)
    df_exp_shortreason[SR_.PROBLEMID] = df_exp_shortreason[SR_.PROBLEMID].astype(int)

    #WEEK 컬럼 추가(미리 생성)
    isoCalDate = pd.to_datetime(df_exp_sopromisesrc[ND_.PROMISEDDELDATE]).dt.isocalendar()
    df_exp_sopromisesrc['PROMISEDDELWEEK'] = isoCalDate.year.astype(str) + isoCalDate.week.astype(str).str.zfill(2)
    df_exp_sopromisesrc['PROMISEDDELWEEK'] = df_exp_sopromisesrc['PROMISEDDELWEEK'].astype(int)

    df_exp_sopromisesrc['ORG_QTYPROMISED'] = df_exp_sopromisesrc[ND_.QTYPROMISED].astype(int)

    df_exp_sopromisesrc[ND_.SOLINENUM] = df_exp_sopromisesrc[ND_.SOLINENUM].astype(int)
    df_exp_sopromisesrc[ND_.QTYPROMISED] = df_exp_sopromisesrc[ND_.QTYPROMISED].astype(int)
    df_exp_sopromisesrc[ND_.DEMANDPRIORITY] = df_exp_sopromisesrc[ND_.DEMANDPRIORITY].astype(int)

    df_mst_inventory_fne[AR_.QTY] = df_mst_inventory_fne[AR_.QTY].astype(int)

    df_exp_deliveryplan[DP_.QTYOPEN] = df_exp_deliveryplan[DP_.QTYOPEN].astype(int)

    #백업
    df_exp_sopromisesrcnew = df_exp_sopromisesrc.copy()

    logger.Note('Start Create MST_NEWFCSTNETTING', 20)
    df_mst_newfcstnetting_result = create_mst_newfcstnetting(
            v_planid, v_week1, v_week4, v_horisonweek, v_effstartdate, v_endweek
            , df_v_mta_sellermap
            , df_exp_sopromisesrc
            , df_exp_deliveryplan
            , df_exp_shortreason
            , df_mst_inventory
            , df_mst_intransit
            , df_mst_inventory_fne
            , df_mst_distributionorders
            , df_vui_itemattb
            , df_mta_codemap
            , df_mta_newfcstnetting
            , df_mta_custommodelmap )
    logger.Note('End Create MST_NEWFCSTNETTING', 20)


    ################### SP_FN_NEWFCSTSRC 시작 ###################
    # CUR_OPENNETTING  NEW FCST 대상 발췌
    #sp_fn_acailnetting까지 수행 완료된 df 에서 조건 걸어서 df 새로 생성
    # -- NEWDP > 0 이고, SITEID 가 TOTAL 이 아닌 조건

    df_mst_newfcstnetting_fcst = df_mst_newfcstnetting_result[df_mst_newfcstnetting_result['NEWDP'].astype(int) > 0]
    df_mst_newfcstnetting_fcst = df_mst_newfcstnetting_fcst[df_mst_newfcstnetting_fcst['SITEID'] != 'TOTAL']
    df_mst_newfcstnetting_fcst = df_mst_newfcstnetting_fcst.sort_values(by=['SITEID', 'ITEM', 'WEEK'])

    #NEWDP : 는 NEWFCST 수요로 꼬리표를 붙여주고 싶은 수량
    #CURQTY : 수요 수량

    #1 IF  R.NEWDP >= R.CURQTY AND  R.CURQTY > 0  THEN 조건 대상 발췌
    df_mst_newfcstnetting_update1 = df_mst_newfcstnetting_fcst[
            (df_mst_newfcstnetting_fcst['NEWDP'].astype(int) >= df_mst_newfcstnetting_fcst['CURQTY'].astype(int)) & 
            (df_mst_newfcstnetting_fcst['CURQTY'].astype(int) > 0 )
        ]

    df_exp_sopromisesrc = df_exp_sopromisesrc.astype({ND_.SOLINENUM:int})


    idx_mves_itemsite = pd.MultiIndex.from_frame(df_mves_itemsite[[ESIS.ITEM, ESIS.SITEID, ESIS.SALESID]])
    df_exp_sopromisesrc_update1 = df_exp_sopromisesrc[
        ~pd.MultiIndex.from_frame(df_exp_sopromisesrc[[ND_.ITEMID, ND_.SITEID, ND_.SALESID]]).isin(idx_mves_itemsite)
    ]

    df_exp_sopromisesrc_update1 = df_exp_sopromisesrc_update1.astype({'PROMISEDDELWEEK':int})
    df_mst_newfcstnetting_update1 = df_mst_newfcstnetting_update1.astype({'WEEK':int})

    #업데이트 대상 발췌
    df_exp_sopromisesrc_update1_join = pd.merge(
        df_exp_sopromisesrc_update1, df_mst_newfcstnetting_update1
        , left_on=[ND_.SITEID, ND_.ITEMID, 'PROMISEDDELWEEK'], right_on=['SITEID', 'ITEM', 'WEEK']
        , suffixes=('', '_x'))
    df_exp_sopromisesrc_update1_join = df_exp_sopromisesrc_update1_join.drop(df_exp_sopromisesrc_update1_join.filter(regex='_x$').columns, axis=1)

    midx_exp_src = pd.MultiIndex.from_frame(df_exp_sopromisesrc[[ND_.SALESORDERID, ND_.SOLINENUM]])
    midx_exp_tar = pd.MultiIndex.from_frame(df_exp_sopromisesrc_update1_join[[ND_.SALESORDERID, ND_.SOLINENUM]])

    df_exp_sopromisesrc.loc[midx_exp_src.isin(midx_exp_tar), ND_.SOLINENUM] = df_exp_sopromisesrc[ND_.SOLINENUM] + 99
    df_exp_sopromisesrc.loc[midx_exp_src.isin(midx_exp_tar), 'UPBY'] = 'NEWALL'

    #2
    #NEWDP < CURQTY 이면, 수요 수량 중 일부만 NEWFCST 수요로 변환하고, 일부는 그대로 두어야 
    #이 경우는 어느 수요를 NEWFCST 수요로 만들어야 하는지 우선순위에 따라 변환할 ROW를 찾아주는 로직이 추가됨
    #또한 우선순위까지 다 동일한 경우는 여러 ROW에 대해서 동시에 적용하기 위해 N빵 해주는 로직이 적용

    #ELSIF R.CURQTY > R.NEWDP AND R.NEWDP>0 THEN 조건문 반영
    #-- 가용량보다 DEMAND가 많은 경우 해당 주차 PRIORITY별로 나눠주기
    #-- 동순위위는 N빵
    df_mst_newfcstnetting_update2 = df_mst_newfcstnetting_fcst[
        (df_mst_newfcstnetting_fcst['CURQTY'] > df_mst_newfcstnetting_fcst['NEWDP']) &
        (df_mst_newfcstnetting_fcst['NEWDP'] > 0)
    ]

    df_exp_sopromisesrc_sub2 = df_exp_sopromisesrc[
        (df_exp_sopromisesrc[ND_.QTYPROMISED] > 0) &
        ~pd.MultiIndex.from_frame(df_exp_sopromisesrc[[ND_.ITEMID, ND_.SITEID, ND_.SALESID]]).isin(idx_mves_itemsite)
    ]

    #전체 업데이트 대상에 세부 업데이트 대상만큼 돌아야 하니까 Merge join처리하여 하나로 봄.
    df_mst_newfcstnetting_join = pd.merge(
        df_mst_newfcstnetting_update2, df_exp_sopromisesrc_sub2
        , left_on=['ITEM', 'SITEID', 'WEEK'], right_on=[ND_.ITEMID, ND_.SITEID, 'PROMISEDDELWEEK']
        , suffixes=('', '_x1'))
    df_mst_newfcstnetting_join = df_mst_newfcstnetting_join.drop(df_mst_newfcstnetting_join.filter(regex='_x1$').columns, axis=1)

    df_mst_newfcstnetting_join = df_mst_newfcstnetting_join.astype({ND_.DEMANDPRIORITY:int})

    #1차 정렬 : ORDER BY  A.DEMANDPRIORITY, A.QTYPROMISED, A.SALESID -> 전체적으로 봤을 때 ITEM, SITEID, WEEK에 대한 그룹 조건을 생각하여 추가.
    #df_mst_newfcstnetting_join = df_mst_newfcstnetting_join.sort_values(by=['ITEM', 'SITEID', 'WEEK', 'DEMANDPRIORITY', 'QTYPROMISED', 'SALESID'])

    df_mst_newfcstnetting_join['RK'] = df_mst_newfcstnetting_join.groupby(['ITEM', 'SITEID','WEEK'])[ND_.DEMANDPRIORITY].rank(method='dense', ascending=False).astype(int)

    #정렬을 먼저
    #df_mst_newfcstnetting_join = df_mst_newfcstnetting_join.sort_values(by=['ITEM', 'SITEID','WEEK', 'RK', 'SALESORDERID', 'SOLINENUM'], ascending=[True, True, True, True, True, True])

    df_mst_newfcstnetting_join['ROWNUMBER'] = (df_mst_newfcstnetting_join.sort_values(by=['ITEM', 'SITEID', 'WEEK', 'RK', ND_.QTYPROMISED, ND_.SALESID], ascending=[True, True, True, True, True, True]).groupby(['ITEM', 'SITEID', 'WEEK']).cumcount() + 1)

    #FARE SHARE를 위한 체크값
    #CHK : RK 동일 구간별 순번(역순)
    df_mst_newfcstnetting_join['CHRK'] = (df_mst_newfcstnetting_join.sort_values(by=['ITEM', 'SITEID', 'WEEK', 'RK', 'ROWNUMBER'], ascending=[True, True, True, True, False]).groupby(['ITEM', 'SITEID', 'WEEK', 'RK']).cumcount() + 1)

    #동순위 확인
    #CNT : RK 동일 구간별 개수
    df_mst_newfcstnetting_join['CNT'] = df_mst_newfcstnetting_join.groupby(['ITEM', 'SITEID','WEEK','RK'])[ND_.DEMANDPRIORITY].transform('count')
    #SUBTOTAL : RK 동일 구간별 합
    df_mst_newfcstnetting_join['SUBTOTAL'] = df_mst_newfcstnetting_join.groupby(['ITEM', 'SITEID','WEEK','RK'])[ND_.QTYPROMISED].transform('sum')

    #오라클과 파이썬의 부동소수점 계산 차이로 decimal 사용(에러 건수는 줄었으나 해결 안됨)
    #df_mst_newfcstnetting_join['RATIO'] = (df_mst_newfcstnetting_join['QTYPROMISED']/df_mst_newfcstnetting_join['SUBTOTAL']).astype(str)
    #df_mst_newfcstnetting_join['RATIO'] = df_mst_newfcstnetting_join.apply(lambda row: (decimal.Decimal(str(row[ND_.QTYPROMISED]))/decimal.Decimal(str(row['SUBTOTAL']))), axis=1)
    df_mst_newfcstnetting_join['QTYPROMISED_DEC'] = df_mst_newfcstnetting_join[ND_.QTYPROMISED].apply(decimal.Decimal)
    df_mst_newfcstnetting_join['SUBTOTAL_DEC'] = df_mst_newfcstnetting_join['SUBTOTAL'].apply(decimal.Decimal)
    df_mst_newfcstnetting_join['RATIO'] = df_mst_newfcstnetting_join['QTYPROMISED_DEC']/df_mst_newfcstnetting_join['SUBTOTAL_DEC']
    df_mst_newfcstnetting_join = df_mst_newfcstnetting_join.drop(['QTYPROMISED_DEC', 'SUBTOTAL_DEC'], axis=1)

    #2차 정렬 : ORDER BY RK,DEMANDPRIORITY DESC, ROWNUM, QTYPROMISED ;
    df_mst_newfcstnetting_join = df_mst_newfcstnetting_join.sort_values(by=['ITEM', 'SITEID','WEEK', 'RK', 'ROWNUMBER'], ascending=[True, True, True, True, True])

    #실제 루프를 돌리기
    #FOR j IN mydata.first .. mydata.last LOOP 

    #대상 데이터프레임을 넘파일배열로 변환
    #업데이트, 인써트, 페어쉐어 대상 데이터를 리스트에 저장(나중에 일괄 처리)

    ndarr_mst_newfcstnetting = df_mst_newfcstnetting_join.to_numpy()

    #컬럼 인덱스 정의
    idx_fcst_CNT = df_mst_newfcstnetting_join.columns.get_loc('CNT')
    idx_fcst_CHRK = df_mst_newfcstnetting_join.columns.get_loc('CHRK')
    idx_fcst_ROWNUM = df_mst_newfcstnetting_join.columns.get_loc('ROWNUMBER')
    idx_fcst_SUBTOTAL = df_mst_newfcstnetting_join.columns.get_loc('SUBTOTAL')
    idx_fcst_RATIO = df_mst_newfcstnetting_join.columns.get_loc('RATIO')
    idx_fcst_DEMANDPRIORITY = df_mst_newfcstnetting_join.columns.get_loc(ND_.DEMANDPRIORITY)
    idx_fcst_QTYPROMISED = df_mst_newfcstnetting_join.columns.get_loc(ND_.QTYPROMISED)
    idx_fcst_NEWDP = df_mst_newfcstnetting_join.columns.get_loc('NEWDP')

    idx_fcst_SITEID = df_mst_newfcstnetting_join.columns.get_loc(ND_.SITEID)
    idx_fcst_ITEM = df_mst_newfcstnetting_join.columns.get_loc(ND_.ITEMID)
    idx_fcst_PROMISEDDELWEEK = df_mst_newfcstnetting_join.columns.get_loc('PROMISEDDELWEEK')

    idx_fcst_SALESORDERID = df_mst_newfcstnetting_join.columns.get_loc(ND_.SALESORDERID)
    idx_fcst_SOLINENUM = df_mst_newfcstnetting_join.columns.get_loc(ND_.SOLINENUM)
    idx_fcst_PROMISEDDELDATE = df_mst_newfcstnetting_join.columns.get_loc(ND_.PROMISEDDELDATE)

    idx_fcst_UPBY = df_mst_newfcstnetting_join.columns.get_loc('UPBY')

    #변수 초기화
    L_DEMANDPRIORITY = 0
    R_QTY = 0
    STEP_R_QTY = 0
    FARE_SHARE = 0

    list_insert = []
    list_update = []
    list_FARE_SHARE = []

    dict_exp_sopromisesrc_dtype = df_exp_sopromisesrc.dtypes.to_dict()

    for mydata_row in ndarr_mst_newfcstnetting:
        #-- 첫행이거나 
        if mydata_row[idx_fcst_ROWNUM] == 1:
            L_DEMANDPRIORITY = mydata_row[idx_fcst_DEMANDPRIORITY]
            R_QTY = mydata_row[idx_fcst_NEWDP]
            STEP_R_QTY = mydata_row[idx_fcst_NEWDP]
            FARE_SHARE = 0

        #-- 우선순위 바뀔때
        elif L_DEMANDPRIORITY != mydata_row[idx_fcst_DEMANDPRIORITY]:
            L_DEMANDPRIORITY = mydata_row[idx_fcst_DEMANDPRIORITY]
            R_QTY = STEP_R_QTY
            
            FARE_SHARE = 0

        # 발췌된 여러 row 들에 동일한 우선순위가 없는 경우에 대한 처리 
        # 1. 꼬리표 붙여줘야 할 수량이 수요수량보다 크다면, 
        #   -- 전체 newfcst 꼬리표만 달고 (solinenum + 99)  처리 끝
        # 2. 꼬리표 불여줘야 할 수량이 수요 수량보다 작다면,
        #    -- 일부 수량은 newfcst 꼬리표 달고 (solinenum + 99)  insert 
        #    -- 나머지 수량은 기존 row 에다가 (수요수량 - newfcst 꼬리표 불일 수량) 으로 업데이트

        #동순위 없을때
        if mydata_row[idx_fcst_CNT] == 1:

            #남은 수량> QTYPROMISE   작업필요없고 잔량 처리만
            if R_QTY >= mydata_row[idx_fcst_QTYPROMISED]:
                #update
                update_row = mydata_row.copy()
                #update_row[idx_fcst_SOLINENUM] += 99 #키값 수정되는 경우
                update_row[idx_fcst_UPBY] = 'NEWALL2'
                update_row = np.append(update_row, [update_row[idx_fcst_SOLINENUM] + 99]) #CHG_SOLINENUM
                list_update.append(update_row)
                
                STEP_R_QTY = STEP_R_QTY - mydata_row[idx_fcst_QTYPROMISED]

            #남은 수량 < QTYPROMISE  이면 남은수량만큼 UP0DATE 해줘야 함
            else:
                #insert --무조건 수량 상관없이 update, insert 한다. 나중에 N빵 수량 맞추려면 0이라도 row 있어야 함
                insert_row = mydata_row.copy()
                insert_row[idx_fcst_SOLINENUM] += 99
                insert_row[idx_fcst_QTYPROMISED] = R_QTY
                insert_row[idx_fcst_UPBY] = 'NEWINSERT1'
                list_insert.append(insert_row)

                #update
                update_row = mydata_row.copy()
                update_row[idx_fcst_QTYPROMISED] -= R_QTY
                update_row[idx_fcst_UPBY] = 'NEWUPDATE1'
                #update_row = np.append(update_row, [update_row[idx_fcst_SOLINENUM]]) #이 열이 수정되지 않지만 다른 조건에서 추가되므로 열맞추기 위해 추가
                list_update.append(update_row)

                STEP_R_QTY = STEP_R_QTY - R_QTY

        # 발췌된 여러 row 들에 동일한 우선순위가 있는 경우에 대한 처리 
        # 동순위 없을 때와 로직적으로 거의 비슷하지만,
        # 동순위 처리를 위해 한번 더 select 하여 N빵해주는 부분이 있음
        # 이 부분에 대한 처리는 EOP Netting 을 참고해주시면 되고, 
        # RANK() OVER (PARTITION BY RK ORDER BY ROWNUM DESC , QTYPROMISED ) CHRK,  구문에서 
        # rownum desc 구문에 오류가 있어서, ORDER BY 순서는 QTYPROMISED, SALESID 로 변경 필요 ?

        #동순위 있을 때 
        else:
            #남은 수량> QTYPROMISE   작업필요없고 잔량 처리만
            if R_QTY >= mydata_row[idx_fcst_SUBTOTAL]:
                #update
                update_row = mydata_row.copy()
                #update_row[idx_fcst_SOLINENUM] += 99 #키값 수정
                update_row[idx_fcst_UPBY] = 'NEWALL3'
                update_row = np.append(update_row, [update_row[idx_fcst_SOLINENUM] + 99]) #CHG_SOLINENUM
                list_update.append(update_row)

                STEP_R_QTY = STEP_R_QTY - mydata_row[idx_fcst_QTYPROMISED]

            #남은 수량 < QTYPROMISE  이면 남은수량 가지고 N빵 처리 해줘야 함
            else:

                #----------------------------------------------------------------------------------------
                #오라클과 파이썬의 부동소수점 계산 차이로 맞지 않는 부분
                appRatio = decimal.Decimal(mydata_row[idx_fcst_RATIO])
                appAcnt = decimal.Decimal(str(R_QTY)) * appRatio
                appQty = np.trunc(appAcnt)
                #-----------------------------------------------------------------------------------------

                #insert --무조건 수량 상관없이 update, insert 한다. 나중에 N빵 수량 맞추려면 0이라도 row 있어야 함
                insert_row = mydata_row.copy()
                insert_row[idx_fcst_SOLINENUM] += 99
                insert_row[idx_fcst_QTYPROMISED] = appQty
                insert_row[idx_fcst_UPBY] = 'NEWINSERT2'
                list_insert.append(insert_row)

                #update
                update_row = mydata_row.copy()
                update_row[idx_fcst_QTYPROMISED] -= appQty
                update_row[idx_fcst_UPBY] = 'NEWUPDATE2'
                list_update.append(update_row)

                STEP_R_QTY = STEP_R_QTY - appQty
                FARE_SHARE = FARE_SHARE + appQty


                #--N빵처리후 마지막 ROW에서 남는 수량 1,2,3..FARE SHARE를 위해 다시한번
                if mydata_row[idx_fcst_CHRK] == 1 and  R_QTY > FARE_SHARE:
                    #mydata_T(페어쉐어)를 위한 리스트 생성 : R_QTY, FARE_SHARE 변수 값을 데이터프레임에 같이 넣어 놓고, 나중에 사용하다.
                    fare_share_row = mydata_row.copy()
                    fare_share_row = np.append(fare_share_row, [R_QTY, FARE_SHARE])
                    list_FARE_SHARE.append(fare_share_row)
                    
                    STEP_R_QTY = 0



    #일괄 업데이트, 인써트 반영

    #일단 리스트를 원본(머지된테이블)과 동일하게 컬럼을 맞추어 데이터프레임으로 생성한다.
    df_result_insert = pd.DataFrame(list_insert, columns=df_mst_newfcstnetting_join.columns)

    #업데이트할 때 change solinenum 열이 추가되어서 컬럼 추가 작업 추가
    ndarr_update_col = df_mst_newfcstnetting_join.columns.values
    ndarr_update_col_result = np.append(ndarr_update_col, ['CHG_SOLINENUM']) #앞에서 추가된 열에 대한  컬럼명 추가
    df_result_update = pd.DataFrame(list_update, columns=ndarr_update_col_result)

    #다시 업데이트 대상 테이블과 동일하게 컬럼, 데이터타입 맞춘다.
    dict_exp_sopromisesrc_dtype = df_exp_sopromisesrc.dtypes.to_dict()
    df_result_insert = df_result_insert[df_exp_sopromisesrc.columns].astype(dict_exp_sopromisesrc_dtype)

    #업데이트에 필요한 열만 발췌
    df_result_update = df_result_update[[ND_.SALESORDERID, ND_.SOLINENUM, ND_.QTYPROMISED, 'UPBY', 'CHG_SOLINENUM']]

    #Insert : 추가되는 데이터이므로 그냥 합쳐버림(정렬에 맞춰야 돼는지 다시 확인)
    df_exp_sopromisesrc = pd.concat([df_exp_sopromisesrc, df_result_insert], ignore_index=True)

    #업데이트 : left outer join 후 값 변경 처리
    df_exp_sopromisesrc = pd.merge(
            df_exp_sopromisesrc.astype({ND_.SOLINENUM:int})
            , df_result_update.astype({ND_.SOLINENUM:int})
            , on=[ND_.SALESORDERID, ND_.SOLINENUM], how='left', suffixes=('', '_U'))
    df_exp_sopromisesrc.loc[~df_exp_sopromisesrc['UPBY_U'].isna(), 'UPBY'] = df_exp_sopromisesrc['UPBY_U']
    df_exp_sopromisesrc.loc[~df_exp_sopromisesrc[ND_.QTYPROMISED+'_U'].isna(), ND_.QTYPROMISED] = df_exp_sopromisesrc[ND_.QTYPROMISED+'_U']
    df_exp_sopromisesrc.loc[~df_exp_sopromisesrc['CHG_SOLINENUM'].isna(), ND_.SOLINENUM] = df_exp_sopromisesrc['CHG_SOLINENUM']

    df_exp_sopromisesrc = df_exp_sopromisesrc.drop(df_exp_sopromisesrc.filter(regex='_U$').columns, axis=1)

    #3 --N빵처리후 마지막 ROW에서 남는 수량 1,2,3..FARE SHARE를 위해 다시한번 (세번째 루트 작업 시작)
    #IF MYDATA(J).CHRK = 1 AND  R_QTY > FARE_SHARE THEN

    #FARE_SHARE 하기 위한 리스트를 df로 변경하는 작업
    #컬럼 맞추기(순서, 갯수) R_QTY, FARE_SHARE 컬럼 추가)
    ndarr_fcst_col = df_mst_newfcstnetting_join.columns.values
    ndarr_fcst_col_result = np.append(ndarr_fcst_col, ['R_QTY', 'FARE_SHARE'])

    #기존 데이터프레임과 컬럼 데이터 타입을 맞추려고
    dict_mydata_fare_share_dtype = df_mst_newfcstnetting_join.dtypes.to_dict()

    df_mydata_fare_share = pd.DataFrame(list_FARE_SHARE, columns=ndarr_fcst_col_result).astype(dict_mydata_fare_share_dtype)

    #추가 컬럼 데이터 타입 맞추기(R_QTY, FARE_SHARE)
    df_mydata_fare_share = df_mydata_fare_share.astype({
        'R_QTY' : 'int32',
        'FARE_SHARE' : 'int32',
    })



    #mydata_T select쿼리 작업 : 쿼리상 컬럼은 많지만 사용하는 것만 추가

    #R_QTY, FARE_SHARE 컬럼 추가하기 위해 조인 먼저
    # 페어쉐어 건만 대상이 아니라, 페어쉐어 건의 item, site, week 그룹 포함(쿼리로직상 페어쉐어 그룹 중 위에부터 rownum순으로 잘라가면서 처리됨)
    # 나중에 FOR k IN mydata_T.first .. (R_QTY - FARE_SHARE ) 부분에서 (R_QTY - FARE_SHARE)를 계산하기 편하게 하기 위한 작업
    df_exp_sopromisesrc_join2 = pd.merge(df_exp_sopromisesrc, df_mydata_fare_share, on=[ND_.ITEMID, ND_.SITEID, 'PROMISEDDELWEEK'], suffixes=('', '_x2'))
    df_exp_sopromisesrc_join2 = df_exp_sopromisesrc_join2.drop(df_exp_sopromisesrc_join2.filter(regex='_x2$').columns, axis=1)

    df_exp_sopromisesrc_sub3 = df_exp_sopromisesrc_join2[
        (df_exp_sopromisesrc_join2[ND_.SOLINENUM].astype(int) >= 100) &

        #RATIO별로 FIX된 것 중
        (df_exp_sopromisesrc_join2['UPBY'] == 'NEWINSERT2') &

        ~pd.MultiIndex.from_frame(df_exp_sopromisesrc_join2[[ND_.ITEMID, ND_.SITEID, ND_.SALESID]]).isin(idx_mves_itemsite)
    ]

    #WITH DATA AS
    df_exp_sopromisesrcnew_sub = pd.merge(df_exp_sopromisesrcnew, df_mydata_fare_share, on=[ND_.ITEMID, ND_.SITEID, 'PROMISEDDELWEEK'], suffixes=('', '_x2'))
    df_exp_sopromisesrcnew_sub = df_exp_sopromisesrcnew_sub.drop(df_exp_sopromisesrcnew_sub.filter(regex='_x2$').columns, axis=1)

    df_exp_sopromisesrcnew_sub = df_exp_sopromisesrcnew_sub[[ND_.SALESORDERID, ND_.ITEMID, ND_.SITEID, ND_.QTYPROMISED]]
    df_exp_sopromisesrcnew_sub = df_exp_sopromisesrcnew_sub.groupby([ND_.SALESORDERID, ND_.ITEMID, ND_.SITEID]).agg({
        ND_.QTYPROMISED: 'sum'
    }).reset_index()

    df_exp_sopromisesrc_join3 = pd.merge(df_exp_sopromisesrc_sub3, df_exp_sopromisesrcnew_sub, on=[ND_.SALESORDERID, ND_.ITEMID, ND_.SITEID], suffixes=('', '_B'))

    #ORDER BY  A.DEMANDPRIORITY DESC, A.QTYPROMISED, A.SALESID
    #df_exp_sopromisesrc_join3 = df_exp_sopromisesrc_join3.sort_values(by=['ITEM', 'SITEID', 'PROMISEDDELWEEK', 'DEMANDPRIORITY', 'QTYPROMISED', 'SALESID'], ascending=[True, True, True, False, True, True])


    #BULK COLLECT INTO mydata_T 에서 필요 컬럼 추가 * ITEM, SITEID, WEEK 별 RK or ROWNUMBER 순서 정렬이 잘 되어야함

    #--N빵직전 INS NETTING후
    df_exp_sopromisesrc_join3['PROD'] = df_exp_sopromisesrc_join3.groupby([ND_.ITEMID, ND_.SITEID, ND_.WEEKRANK, ND_.SALESORDERID])[ND_.QTYPROMISED].transform('sum').astype(int)
    #--INS NETTING전
    df_exp_sopromisesrc_join3['DEV'] = df_exp_sopromisesrc_join3.groupby([ND_.ITEMID, ND_.SITEID, ND_.WEEKRANK, ND_.SALESORDERID])[ND_.QTYPROMISED +'_B'].transform('sum').astype(int)

    df_exp_sopromisesrc_join3 = df_exp_sopromisesrc_join3.drop(df_exp_sopromisesrc_join3.filter(regex='_B$').columns, axis=1)

    #--NETTING 전보다 데이터가 작아야만 PLUS 해도 상관없음, 같은거는 PLUS 되면 처음 DEMAND보다 많아지므로 안됨
    df_exp_sopromisesrc_join4 = df_exp_sopromisesrc_join3[
        df_exp_sopromisesrc_join3['DEV'] > df_exp_sopromisesrc_join3['PROD']
    ]


    #ROWNUM 순서 맞추기 위해 쏘팅함 'ITEM', 'SITEID','WEEK', 'RK', 'DEMANDPRIORITY', 'QTYPROMISED'까지 같은 경우까지 커버
    #df_exp_sopromisesrc_join4 = df_exp_sopromisesrc_join4.sort_values(by=['ITEM', 'SITEID','WEEK', 'RK', 'DEMANDPRIORITY', 'QTYPROMISED', 'SALESID'], ascending=[True, True, True, True, False, True, True])

    df_exp_sopromisesrc_join4['RK'] = df_exp_sopromisesrc_join4.groupby([ND_.ITEMID, ND_.SITEID, ND_.WEEKRANK])[ND_.DEMANDPRIORITY].rank(method='dense', ascending=False)
    #ITEM, SITEID, WEEK 그룹별 ROWNUM
    df_exp_sopromisesrc_join4['ROWNUMBER'] = (df_exp_sopromisesrc_join4.sort_values(by=['ITEM', 'SITEID', 'WEEK', 'RK', ND_.QTYPROMISED, ND_.SALESID], ascending=[True, True, True, True, True, True]).groupby(['ITEM', 'SITEID', 'WEEK']).cumcount() + 1)

    # #정렬
    #df_exp_sopromisesrc_join4 = df_exp_sopromisesrc_join4.sort_values(by=[ND_.ITEMID, ND_.SITEID, ND_.WEEKRANK, 'RK', ND_.DEMANDPRIORITY, 'ROWNUMBER', ND_.QTYPROMISED], ascending=[True, True, True, True, True, True, True])
    df_exp_sopromisesrc_join4 = df_exp_sopromisesrc_join4.sort_values(by=[ND_.ITEMID, ND_.SITEID, ND_.WEEKRANK, 'RK', 'ROWNUMBER'], ascending=[True, True, True, True, True])


    #페어쉐어 대상 중 대상별 그룹(ITEM,SITE,WEEK)을 구하고(mydata_T), 그룹별 목록에서 ROWNUM이 (R_QTY - FARE_SHARE)값에 해당하는 데이터만 발췌
    #(페어쉐어 대상 자체를 업뎃하는 게 아니라, 페어쉐어 대상 그룹 중 상위에 있는 데이터를 업뎃하기 때문에 동순위 CHRK=1이 아니라도 처리된다.)
    df_exp_sopromisesrc_join5 = df_exp_sopromisesrc_join4[
        df_exp_sopromisesrc_join4['ROWNUMBER'] <= (df_exp_sopromisesrc_join4['R_QTY'] - df_exp_sopromisesrc_join4['FARE_SHARE'])
    ]

    df_exp_sopromisesrc_join5 = df_exp_sopromisesrc_join5.astype({
        ND_.QTYPROMISED:int,
        ND_.SOLINENUM:int,
    })

    #SALESORDERID, SOLINENUM키가 걸리기 때문에 그냥 업데이트하면됨
    df_exp_sopromisesrc_join5['UPBY'] += '_PLUS'
    df_exp_sopromisesrc_join5[ND_.QTYPROMISED] += 1

    # _MINUS 대상 발췌
    df_exp_sopromisesrc_minus = df_exp_sopromisesrc[
        (df_exp_sopromisesrc[ND_.SOLINENUM] < 100) &
        (df_exp_sopromisesrc[ND_.QTYPROMISED] > 0)
    ]

    df_exp_sopromisesrc_minus = pd.merge(df_exp_sopromisesrc_minus, df_exp_sopromisesrc_join5, on=[ND_.SALESORDERID], suffixes=('', '_M'))
    df_exp_sopromisesrc_minus = df_exp_sopromisesrc_minus[
        (df_exp_sopromisesrc_minus[ND_.PREFERENCERANK] == df_exp_sopromisesrc_minus[ND_.PREFERENCERANK + '_M'])
    ]
    df_exp_sopromisesrc_minus = df_exp_sopromisesrc_minus.drop(df_exp_sopromisesrc_minus.filter(regex='_M$').columns, axis=1)

    # # rownum = 1 조건 추가
    # df_exp_sopromisesrc_minus['rownum'] = df_exp_sopromisesrc_minus.sort_values(by=[ND_.SALESORDERID, ND_.SOLINENUM]
    #                                     ).groupby([ND_.SALESORDERID])[ND_.SOLINENUM].rank(method='min').astype(int)
    # df_exp_sopromisesrc_minus = df_exp_sopromisesrc_minus[
    #     df_exp_sopromisesrc_minus['rownum'] == 1
    # ]

    df_exp_sopromisesrc_minus['UPBY'] += '_MINUS'
    df_exp_sopromisesrc_minus[ND_.QTYPROMISED] -= 1

    df_exp_sopromisesrc_plus = df_exp_sopromisesrc_join5[[ND_.SALESORDERID, ND_.SOLINENUM, ND_.QTYPROMISED, 'UPBY' ]]
    df_exp_sopromisesrc_minus = df_exp_sopromisesrc_minus[[ND_.SALESORDERID, ND_.SOLINENUM, ND_.QTYPROMISED, 'UPBY' ]]

    df_exp_sopromisesrc_plus = df_exp_sopromisesrc_plus.astype({
        ND_.QTYPROMISED:int,
        ND_.SOLINENUM:int,
    })
    df_exp_sopromisesrc_minus = df_exp_sopromisesrc_minus.astype({
        ND_.QTYPROMISED:int,
        ND_.SOLINENUM:int,
    })

    # _PLUS 반영
    df_exp_sopromisesrc = pd.merge(df_exp_sopromisesrc, df_exp_sopromisesrc_plus, how='left', on=[ND_.SALESORDERID, ND_.SOLINENUM], suffixes=('', '_PLUS'))
    df_exp_sopromisesrc.loc[(~df_exp_sopromisesrc['UPBY_PLUS'].isna()) , 'UPBY'] = df_exp_sopromisesrc['UPBY_PLUS']
    df_exp_sopromisesrc.loc[(~df_exp_sopromisesrc[ND_.QTYPROMISED + '_PLUS'].isna()), ND_.QTYPROMISED] = df_exp_sopromisesrc[ND_.QTYPROMISED + '_PLUS']

    df_exp_sopromisesrc = df_exp_sopromisesrc.drop(df_exp_sopromisesrc.filter(regex='_PLUS$').columns, axis=1)

    # _MINUS 반영
    df_exp_sopromisesrc = pd.merge(df_exp_sopromisesrc, df_exp_sopromisesrc_minus, how='left', on=[ND_.SALESORDERID, ND_.SOLINENUM], suffixes=('', '_MINUS'))
    df_exp_sopromisesrc.loc[(~df_exp_sopromisesrc['UPBY_MINUS'].isna()) , 'UPBY'] = df_exp_sopromisesrc['UPBY_MINUS']
    df_exp_sopromisesrc.loc[(~df_exp_sopromisesrc[ND_.QTYPROMISED + '_MINUS'].isna()), ND_.QTYPROMISED] = df_exp_sopromisesrc[ND_.QTYPROMISED + '_MINUS']

    df_exp_sopromisesrc = df_exp_sopromisesrc.drop(df_exp_sopromisesrc.filter(regex='_MINUS$').columns, axis=1)

    #UPDATE_FLAG 조건과 수요 수량 조건, WEEK 조건 3가지로 Delete 수행
    #--필요없는 row는 날린다
    # 조건에 맞는 행 삭제
    df_exp_sopromisesrc_del = df_exp_sopromisesrc[
        (df_exp_sopromisesrc['PROMISEDDELWEEK'].astype(int).between(int(v_week1), int(v_week4))) &
        (df_exp_sopromisesrc['UPBY'].fillna('').str.startswith('NEW')) &
        (df_exp_sopromisesrc[ND_.QTYPROMISED] == 0)
    ]
    df_exp_sopromisesrc = df_exp_sopromisesrc.drop(df_exp_sopromisesrc_del.index)

    # 현재 로직은 4번째 자리의 값을 8로 변경하는 것인데, 
    # 이 부분은 demand type 우선순위에 해당하는 위치를 찾아내서 8 로 변경하는 것으로 로직 수정

    # -- 우선순위는 4주이내 mpdp에 대해서 DEMAND TYPE PRIORITY = 8 만들어줌, 나머진 기존값
    # -- NEW FCST는 SOLINENUM >= 100 인 것으로 구분한다.

    #소수점이 붙어 있는 경우 에러가 나거나, 소수점 문자열 그대로 변경될 수 있어 float -> int -> str 순으로 변경
    df_exp_sopromisesrc[ND_.DEMANDPRIORITY] = df_exp_sopromisesrc[ND_.DEMANDPRIORITY].astype(float).astype(int).astype(str)

    start_pos, next_pos, total_length = find_priority_position('G_R001::1', 'DEMANDTYPERANK')
    digit = next_pos - start_pos
    
    condition = df_exp_sopromisesrc[ND_.SOLINENUM] >= 100
    df_exp_sopromisesrc.loc[condition, ND_.DEMANDPRIORITY] = ( 
            df_exp_sopromisesrc[ND_.DEMANDPRIORITY].str[0: start_pos] 
            + '8'.zfill(digit) 
            + df_exp_sopromisesrc[ND_.DEMANDPRIORITY].str[next_pos:] 
        )
    
    #[데이터 생성 완료]##################################################

    #실행 종료 시간 
    #end_time = time.time()
    #print("코드 실행 시간: {:.6f}초".format(end_time - start_time))

    df_exp_sopromisesrc_result = df_exp_sopromisesrc[idx_exp_sopromisesrc]

    logger.Note('End NEW FCST Netting', 20)


    return df_exp_sopromisesrc_result
