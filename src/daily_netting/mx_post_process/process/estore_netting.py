import pandas as pd
import numpy as np

from daily_netting.mx_post_process.constant.dim_constant import Item as VIA
from daily_netting.mx_post_process.constant.ods_constant import NettedDemandD as ND_    # EXP_SOPROMISE
from daily_netting.mx_post_process.constant.mx_constant import EStoreReplenishment as ESR    # MVUI_ES_REPLENISHMENT or DYN_ES_REPLENISHMENT_CLOSING@SCMTF
from daily_netting.mx_post_process.constant.mx_constant import ESItemSite as ESIS    # MVES_ITEMSITE
from daily_netting.mx_post_process.constant.mx_constant import EStoreTWOS as ESTWOS    # MTA_ESTORE_TWOS_WEEK
from daily_netting.mx_post_process.constant.mx_constant import CodeMap as CM    # MTA_CODEMAP
from daily_netting.mx_post_process.utils import NSCMCommon as nscm
from daily_netting.mx_post_process.utils.common import convert_dt_to_week

DF = pd.DataFrame
LOGGER = nscm.G_Logger

v_effstartdate = None
v_effenddate = None
find_priority_position = None

def set_estore_netting_env(accessor: object) -> None:
    '''동적으로 전역변수를 설정하기 위한 함수'''
    global v_effstartdate, v_effenddate
    global find_priority_position
    
    find_priority_position = accessor.find_priority_position

    #입력 파라미터 및 기준 데이터 정의
    v_effstartdate = accessor.start_date
    v_effenddate = accessor.end_date


def do_estore_netting(
    df_exp_sopromise: DF, df_mta_estore_twos_week: DF, df_vui_itemattb: DF, df_mta_codemap: DF,
    df_mves_itemsite: DF, df_mvui_es_replenishment: DF, logger: LOGGER,
) -> None:
    
    logger.Step(1, 'Start eStore Netting')
    idx_exp_sopromise = df_exp_sopromise.columns
    #필요 컬럼 추가 / 컬럼 타입 조정

    # exp_isoCal = pd.to_datetime(df_exp_sopromise[ND_.PROMISEDDELDATE]).dt.isocalendar()
    # df_exp_sopromise['PROMISEDDELWEEK'] = exp_isoCal.year.astype(str) + exp_isoCal.week.astype(str).str.zfill(2)

    df_exp_sopromise['PROMISEDDELWEEK'] = convert_dt_to_week(pd.to_datetime(df_exp_sopromise[ND_.PROMISEDDELDATE]))

    df_exp_sopromise[ND_.SOLINENUM] = df_exp_sopromise[ND_.SOLINENUM].astype(int)
    df_exp_sopromise[ND_.QTYPROMISED] = df_exp_sopromise[ND_.QTYPROMISED].astype(int)
    # df_exp_sopromise['DEMANDPRIORITY'] = df_exp_sopromise['DEMANDPRIORITY'].astype(int)

    # mvui_isoCal = pd.to_datetime(df_mvui_es_replenishment[ESR.PLANNEDSTARTDATE]).dt.isocalendar()
    # df_mvui_es_replenishment['PLANNEDSTARTWEEK'] = mvui_isoCal.year.astype(str) + mvui_isoCal.week.astype(str).str.zfill(2)

    df_mvui_es_replenishment['PLANNEDSTARTWEEK'] = convert_dt_to_week(
        pd.to_datetime(df_mvui_es_replenishment[ESR.PLANNEDSTARTDATE])
    )


    #### 1 (df_AVAIL 생성)
    midx_exp_sopromise = pd.MultiIndex.from_frame(df_exp_sopromise[[ND_.ITEMID, ND_.SITEID, ND_.SALESID]])
    midx_mvui_es_replenishment = pd.MultiIndex.from_frame(df_mvui_es_replenishment[[ESR.ITEM, ESR.SITEID, ESR.SALESID]])

    df_w_avail = df_mvui_es_replenishment[
        (pd.to_datetime(df_mvui_es_replenishment[ESR.PLANNEDSTARTDATE]).between(v_effstartdate, v_effenddate)) &
        (midx_mvui_es_replenishment.isin(midx_exp_sopromise))
    ]
    df_w_avail = df_w_avail[[ESR.ITEM, ESR.SITEID, ESR.SALESID, ESR.PLANNEDSTARTDATE, 'PLANNEDSTARTWEEK', ESR.REPLENISHMENTQTY]]
    df_w_avail = df_w_avail.rename(columns={ESR.ITEM: ND_.ITEMID, 
                            ESR.SITEID: ND_.SITEID, 
                            ESR.SALESID: ND_.SALESID, 
                            ESR.PLANNEDSTARTDATE: 'ENDDATE',
                            'PLANNEDSTARTWEEK': 'WEEK', 
                            ESR.REPLENISHMENTQTY:'AVAILQTY'})#, inplace=True

    df_w_avail['AVAILQTY'] = pd.to_numeric(df_w_avail['AVAILQTY']).astype(int)
    df_w_avail = df_w_avail.groupby(
        [ND_.ITEMID, ND_.SITEID, ND_.SALESID, 'ENDDATE', 'WEEK',]
    ).agg({'AVAILQTY': 'sum'}).reset_index()

    #### 2-1 (df_DEMAND 생성)

    #OPTION_CODE 없는 경우 나중에 문제가 있어 0 처리
    #df_exp_sopromise['OPTION_CODE'] = 0

    #160472
    df_w_demand = df_exp_sopromise[
        (midx_exp_sopromise.isin(midx_mvui_es_replenishment))
    ]

    #### 2-2 가상 DEMAND 정보를 당주에 생성하는 과정
    df_w_demand_v =  df_w_avail[df_w_avail.columns]

    df_w_demand_v[ND_.QTYPROMISED] = 0
    #PLANNEDENDDATE 월요일 -> PROMISEDDELDATE 일요일
    df_w_demand_v[ND_.PROMISEDDELDATE] = pd.to_datetime(df_w_demand_v['ENDDATE']) + pd.Timedelta(days=6)

    df_w_demand_v[ND_.WEEKRANK] = df_w_demand_v['WEEK'].str[-2:]

    demv_isoCal = df_w_demand_v[ND_.PROMISEDDELDATE].dt.isocalendar()
    df_w_demand_v['PROMISEDDELWEEK'] = demv_isoCal.year.astype(str) + demv_isoCal.week.astype(str).str.zfill(2)

    df_w_demand_v[ND_.OPTION_CODE] = int('10', 2)

    #### 2
    #df 2-1, df 2-2 를 합쳐서 df 2 (df DEMAND 생성 )
    df_w_demand = pd.concat([df_w_demand, df_w_demand_v], ignore_index=True)

    #### 3
    #df_AVAIL 과 df_DEMAND 를 join 하여 계산을 위한 컬럼을 생성함
    #SALESID, SITEID, ITEM, WEEK 4개 컬럼으로 outer join

    df_avail_demand_join = pd.merge(
        df_w_demand, df_w_avail, how='left',
        left_on=[ND_.SALESID, ND_.SITEID, ND_.ITEMID, 'PROMISEDDELWEEK'],
        right_on=[ND_.SALESID, ND_.SITEID, ND_.ITEMID, 'WEEK'], suffixes=('', '_B')
    )

    df_avail_demand_join[ND_.QTYPROMISED] = df_avail_demand_join[ND_.QTYPROMISED].astype(int)

    df_avail_demand_join['AVAILQTY'] = df_avail_demand_join['AVAILQTY_B'].fillna(0)

    df_avail_demand_join = df_avail_demand_join.drop(df_avail_demand_join.filter(regex='_B$').columns, axis=1)

    #정렬이 안되기 때문에 날짜형으로 변환함.
    df_avail_demand_join[ND_.PROMISEDDELDATE] = pd.to_datetime(df_avail_demand_join[ND_.PROMISEDDELDATE])

    #DEMANDPRIORITY가 NULL인 경우 아래로 정렬되어야하는데, fillna(0) 처리하면 위로 올라와서 안됨.
    #df_avail_demand_join['DEMANDPRIORITY'] = df_avail_demand_join['DEMANDPRIORITY'].fillna(0).astype(int)

    #값 생성을 위한 1차 정렬
    df_avail_demand_join = df_avail_demand_join.sort_values(by=[ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE, ND_.DEMANDPRIORITY, ND_.QTYPROMISED])

    #그룹별 rownum
    #df_avail_demand_join['PART_CNT'] = df_avail_demand_join.groupby(['ITEM', 'SITEID', 'SALESID', 'PROMISEDDELDATE'])['DEMANDPRIORITY'].cumcount()+1
    df_avail_demand_join['PART_CNT'] = df_avail_demand_join.groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE]).cumcount()+1 #카운트를 하는 것이므로 1차 정렬 기준으로 유지하여 카운트한다.

    #그룹별 카운트
    df_avail_demand_join['PART2_CNT'] = df_avail_demand_join.groupby(
        [ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE]
    )[ND_.QTYPROMISED].transform('count')

    df_avail_demand_join['ALL_CNT'] = (df_avail_demand_join.sort_values(
        by=[
            ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE,
            ND_.DEMANDPRIORITY, ND_.QTYPROMISED,
        ]
    ).groupby([ND_.ITEMID, ND_.SITEID, ND_.SALESID]).cumcount() + 1)

    #그룹별 합계
    df_avail_demand_join['SUM_DEMAND'] = df_avail_demand_join.groupby(
        [ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE]
    )[ND_.QTYPROMISED].cumsum()

    #### 4
    #df 3 을  ITEM, SITEID, SALESID, PROMISEDDELDATE, ALL_CNT 순으로 정렬
    df_avail_demand_join = df_avail_demand_join.sort_values(by=[ND_.ITEMID, ND_.SITEID, ND_.SALESID, ND_.PROMISEDDELDATE, 'ALL_CNT'])


    #df_exp_sopromise - 딕셔너리로 만들어 바로 처리
    cols_df_exp_sopromisesrc = df_exp_sopromise.columns
    iSALESORDERID = cols_df_exp_sopromisesrc.get_loc(ND_.SALESORDERID)
    iSOLINENUM = cols_df_exp_sopromisesrc.get_loc(ND_.SOLINENUM)

    iOPTION_CODE = cols_df_exp_sopromisesrc.get_loc(ND_.OPTION_CODE)
    iQTYPROMISED = cols_df_exp_sopromisesrc.get_loc(ND_.QTYPROMISED)
    # iUPDTTM = cols_df_exp_sopromisesrc.get_loc('UPDTTM')
    # iUPBY = cols_df_exp_sopromisesrc.get_loc('UPBY')

    data_array = df_exp_sopromise.to_numpy()

    # 딕셔너리 생성
    dict_exp_sopromisesrc = { (row[iSALESORDERID], row[iSOLINENUM]): row for row in data_array }
    dict_exp_sopromisesrc

    # SalesOrderID 별 SoLineNum 리스트(ndArray) - max solinenum을 찾을 때 사용
    dict_salesorder = {}
    for row in data_array:
        if row[iSALESORDERID] in dict_salesorder:
            dict_salesorder[row[iSALESORDERID]].add(row[iSOLINENUM])
        else:
            dict_salesorder[row[iSALESORDERID]] = set((row[iSOLINENUM],))


    #FOR S IN (  LOOP 처리 부분

    #df_avail_demand_join = df_avail_demand_join.astype({'OPTION_CODE':int, 'SOLINENUM':int, 'QTYPROMISED':int, 'AVAILQTY':int})
    df_avail_demand_join = df_avail_demand_join.astype({'AVAILQTY':int})

    df_avail_demand_join[ND_.OPTION_CODE] = df_avail_demand_join[ND_.OPTION_CODE].fillna('0').astype(float).astype(int)

    ndarr_avail_demand_join = df_avail_demand_join.to_numpy()

    cols_avail_demand_join = df_avail_demand_join.columns
    idx_ALL_CNT= cols_avail_demand_join.get_loc('ALL_CNT')
    idx_PART_CNT = cols_avail_demand_join.get_loc('PART_CNT')
    idx_PART2_CNT = cols_avail_demand_join.get_loc('PART2_CNT')
    idx_AVAILQTY = cols_avail_demand_join.get_loc('AVAILQTY')
    idx_SUM_DEMAND = cols_avail_demand_join.get_loc('SUM_DEMAND')
    idx_QTYPROMISED = cols_avail_demand_join.get_loc(ND_.QTYPROMISED)

    idx_OPTION_CODE = cols_avail_demand_join.get_loc(ND_.OPTION_CODE)
    # idx_UPDTTM = cols_avail_demand_join.get_loc('UPDTTM')
    # idx_UPBY = cols_avail_demand_join.get_loc('UPBY')

    idx_SALESORDERID = cols_avail_demand_join.get_loc(ND_.SALESORDERID)
    idx_SOLINENUM = cols_avail_demand_join.get_loc(ND_.SOLINENUM)

    idx_ITEM = cols_avail_demand_join.get_loc(ND_.ITEMID)
    idx_SITEID = cols_avail_demand_join.get_loc(ND_.SITEID)
    idx_SALESID = cols_avail_demand_join.get_loc(ND_.SALESID)

    V_REMAINQTY = 0
    V_AVAILQTY = 0

    for s_row in ndarr_avail_demand_join:

        row = dict_exp_sopromisesrc.get( ( s_row[idx_SALESORDERID], s_row[idx_SOLINENUM] ) )

        #--! 새로운 모델/SALES/SITE 인 경우 잔량 0처리 !--
        if s_row[idx_ALL_CNT] == 1 : 
            V_REMAINQTY = 0;

        #--! 주차가 바뀌는 경우 앞에 남은 수량은 뒤로 Move !--
        if s_row[idx_PART_CNT] == 1 :
            V_AVAILQTY = s_row[idx_AVAILQTY] + V_REMAINQTY

        #--! 가용량이 Demand 수량 합보다 큰 경우 전량 ESOTRE용 DEMAND 표시 !--
        if V_AVAILQTY >= s_row[idx_SUM_DEMAND] and s_row[idx_QTYPROMISED] > 0 :

            # UPDATE
            row[iOPTION_CODE] = s_row[idx_OPTION_CODE] + int('1000000000', 2) #--512
            # row[iUPDTTM] = np.datetime64('now')
            # row[iUPBY] = f'ESTORE_ALL{V_AVAILQTY}'
        
        else :
            #--! 가용량이 누적합보다는 작은 경우, 가용량 만큼만 ESOTRE용 DEMAND 표시 !--           
            if V_AVAILQTY - (s_row[idx_SUM_DEMAND] - s_row[idx_QTYPROMISED]) > 0 and s_row[idx_QTYPROMISED] > 0 :

                # UPDATE
                row[iOPTION_CODE] = s_row[idx_OPTION_CODE] + int('1000000000', 2) #--512
                row[iQTYPROMISED] = V_AVAILQTY - (s_row[idx_SUM_DEMAND] - s_row[idx_QTYPROMISED])
                # row[iUPDTTM] = np.datetime64('now')
                # row[iUPBY] = f'ESTORE_PART{V_AVAILQTY}'

                #--! 잔여 수량 일반 DEMAND !--
                V_SOLINENUM = s_row[idx_SOLINENUM] + 1

                if (s_row[idx_SALESORDERID], V_SOLINENUM ) in dict_exp_sopromisesrc :
                    #print('EXCEPTION WHEN DUP_VAL_ON_INDEX THEN')
                    so_arr = np.array( list( dict_salesorder.get(s_row[idx_SALESORDERID]) ) )
                    V_SOLINENUM = np.max( so_arr ) + 1

                new_row = row.copy() # 위에서 먼저 업데이트를 했기에 데이터가 바뀌어 있다.
                new_num = V_SOLINENUM
                new_row[iSOLINENUM] = new_num
                new_row[iOPTION_CODE] = s_row[idx_OPTION_CODE]
                new_row[iQTYPROMISED] = s_row[idx_SUM_DEMAND] - V_AVAILQTY
                # new_row[iUPDTTM] = np.datetime64('now')
                # new_row[iUPBY] = f'ESTORE_NOT_PART{V_AVAILQTY}'
                dict_exp_sopromisesrc[ (s_row[idx_SALESORDERID], new_num) ] = new_row # INSERT 처리
                dict_salesorder.get(s_row[idx_SALESORDERID]).add(V_SOLINENUM) # 다음을 위해 MAX SOLINENUM 반영해 놓음

        #--! 주차가 끝나는 시점에서 사용되고 남은 가용량은 차주로 Move !--
        if s_row[idx_PART_CNT] == s_row[idx_PART2_CNT] : 
            V_REMAINQTY = 0 if V_AVAILQTY - s_row[idx_SUM_DEMAND] < 0 else V_AVAILQTY - s_row[idx_SUM_DEMAND]


    # 데이터프레임으로 복원
    values_array = np.array(list(dict_exp_sopromisesrc.values()))
    df_restore2_exp_sopromisesrc = pd.DataFrame(values_array, columns=cols_df_exp_sopromisesrc)
    dtype_exp_sopromisesrc = df_exp_sopromise.dtypes.to_dict()
    df_exp_sopromise = df_restore2_exp_sopromisesrc.astype(dtype_exp_sopromisesrc)


    #수요 일자 조정과 우선순위 조정을 위해 중간 계산 df 생성
    #MST_ESTORE_SALESORDER 생성 시작

    df_mves_itemsite_sub = df_mves_itemsite[
        (df_mves_itemsite[ESIS.MODELTYPE] == 'H') &
        (df_mves_itemsite[ESIS.TWOS].astype(int) == np.trunc(df_mves_itemsite[ESIS.TWOS].astype(float)))
    ]
    df_mves_itemsite_sub = df_mves_itemsite_sub[[ESIS.ITEM, ESIS.SITEID, ESIS.SALESID, ESIS.TWOS, ESIS.TARGETDATEEARLINESS]]

    df_mves_itemsite_sub = df_mves_itemsite_sub.groupby([ESIS.ITEM, ESIS.SITEID, ESIS.SALESID]).agg({
        ESIS.TWOS : 'max',
        ESIS.TARGETDATEEARLINESS : 'max'
    }).reset_index()

    df_mves_itemsite_sub = df_mves_itemsite_sub[(df_mves_itemsite_sub[ESIS.TWOS].astype(int) > 0)]

    df_mves_itemsite_sub = df_mves_itemsite_sub.rename(columns={
        ESIS.ITEM : ND_.ITEMID,
        ESIS.SITEID : ND_.SITEID,
        ESIS.SALESID : ND_.SALESID,
        # ESIS.TWOS : 'TWOS',
        # ESIS.TARGETDATEEARLINESS : 'TARGETDATEEARLINESS'
    })

    #JOIN - VUI_ITEMATTB VI,
    df_exp_sopromise_join = pd.merge(df_exp_sopromise, df_vui_itemattb[[VIA.ITEM, VIA.PRODUCTGROUP, VIA.ATTB09]], left_on=[ND_.ITEMID], right_on=[VIA.ITEM], suffixes=('', '_VI'))
    # df_exp_sopromise_join = df_exp_sopromise_join.rename(columns={
    #     ND_.ITEMID : 'ITEM',
    #     ND_.SITEID : 'SITEID',
    #     ND_.SALESID : 'SALESID',
    #     VIA.PRODUCTGROUP : 'PRODUCTGROUP',
    #     VIA.ATTB09 : 'ATTB09'
    # })

    #JOIN - (MVES_ITEMSITE) B
    df_exp_sopromise_join = pd.merge(
        df_exp_sopromise_join, df_mves_itemsite_sub,
        on=[ND_.ITEMID, ND_.SITEID, ND_.SALESID], suffixes=('', '_B')
    )

    #JOIN - MTA_ESTORE_TWOS_WEEK C,
    #AND VI.PRODUCTGROUP = C.PRODUCTGROUP (+)
    #AND VI.ATTB09 = CASE WHEN C.ATTB09(+) ='ALL' THEN VI.ATTB09  ELSE C.ATTB09(+) END
    #AND A.SITEID = CASE WHEN C.SITEID(+) ='ALL' THEN A.SITEID  ELSE C.SITEID(+) END
    #우선 PRODUCTGROUP 조건 사용
    #ATTB09, SITEID 조건의 경우 
    # MTA_ESTORE_TWOS_WEEK (Netting IF eStore TWOS)  의 Item.A9 값이나, location.location 값이 ALL 인 경우는 Join 조건에서 제외하고, 
    # ALL 아 아닌 경우에만 join 함
    df_exp_sopromise_join = pd.merge(
        df_exp_sopromise_join, df_mta_estore_twos_week, how='left',
        left_on=[VIA.PRODUCTGROUP, VIA.ATTB09, ND_.SITEID],
        right_on=[ESTWOS.PRODUCTGROUP, ESTWOS.ATTB09, ESTWOS.SITEID], suffixes=('', '_C')
    )
    #df_exp_sopromise_join = pd.merge(df_exp_sopromise_join, df_mta_estore_twos_week, how='left', left_on=['PRODUCTGROUP'], right_on=[ESTWOS.PRODUCTGROUP], suffixes=('', '_C'))

    # df_exp_sopromise_join = df_exp_sopromise_join[
    #     ( (df_exp_sopromise_join[ESTWOS.ATTB09] == 'ALL') | (df_exp_sopromise_join[VIA.ATTB09] == df_exp_sopromise_join[ESTWOS.ATTB09]) ) &
    #     ( (df_exp_sopromise_join[ESTWOS.SITEID] == 'ALL') | (df_exp_sopromise_join['SITEID'] == df_exp_sopromise_join[ESTWOS.SITEID]) )
    # ]

    #df_exp_sopromise_join

    df_exp_sopromise_join[ND_.PROMISEDDELDATE] = pd.to_datetime(df_exp_sopromise_join[ND_.PROMISEDDELDATE])

    # -  전체 join 된 이후 조건  처리
    #   .. TRUNC((PROMISEDDELDATE -  CURRENTDATE) / 7) < NVL(C.WEEK,25) + B.TWOS  -->    ( PROMISEDDELDATE - V_EFFSTARTDATE ) / 7 하고 소수점 내림  < Netting eStore TWOS Week 가 NULL 이면 25 값 입력 + Netting ES Item Site TWOS
    df_exp_sopromise_join = df_exp_sopromise_join[
        ( (((df_exp_sopromise_join[ND_.PROMISEDDELDATE] - v_effstartdate).dt.days) /7) <
        (df_exp_sopromise_join[ESTWOS.WEEK].fillna(25).astype(int) +  df_exp_sopromise_join[ESIS.TWOS].astype(int)) )
    ]

    #df_exp_sopromise_join['dayCnt'] = (df_exp_sopromise_join[ND_.PROMISEDDELDATE] - v_effstartdate).dt.days
    #df_exp_sopromise_join[
    #    ( np.trunc(df_exp_sopromise_join['dayCnt']/7) > (df_exp_sopromise_join[ESTWOS.WEEK].fillna(25).astype(int) +  df_exp_sopromise_join['TWOS'].astype(int)) )
    #]

    #- 컬럼 생성 
    #   .. WEEK_NEW, PROMISEDDELDATE_START, MAPPINGTYPE, RK 등의 컬럼을 생성
    # - 최종적으로 RK = 1 인 조건으로 필터링하여 df 생성
    # df_exp_sopromise_join = df_exp_sopromise_join[
    #     ( (((pd.to_datetime(df_exp_sopromise_join[ND_.PROMISEDDELDATE]) - v_effstartdate).dt.days) /7) < 
    # (df_exp_sopromise_join[ESTWOS.WEEK].fillna(25) +  df_exp_sopromise_join['TWOS'].astype(int)) )
    # ]

    df_exp_sopromise_join['WEEK_ORG'] = df_exp_sopromise_join[ND_.DEMANDPRIORITY].str[1:3]

    week_new = df_exp_sopromise_join[ND_.DEMANDPRIORITY].astype(int).astype(str).str[1:3].astype(int) - df_exp_sopromise_join[ESIS.TWOS].astype(int)

    df_exp_sopromise_join['WEEK_NEW'] = np.where( week_new < 0, '00', 
    np.where( week_new.astype(str).str.len() == 1, week_new.astype(str).str.zfill(2), week_new.astype(str) )
    )

    minus_twos_date = df_exp_sopromise_join[ND_.PROMISEDDELDATE] - pd.to_timedelta(df_exp_sopromise_join[ESIS.TWOS].astype(int) * 7, unit='days')

    df_exp_sopromise_join['PROMISEDDELDATE_START'] = np.where( df_exp_sopromise_join[ESIS.TARGETDATEEARLINESS] == 'N', df_exp_sopromise_join[ND_.PROMISEDDELDATE], 
    np.where( minus_twos_date < v_effstartdate.to_datetime64() , v_effstartdate.to_datetime64(), minus_twos_date )
    )

    df_exp_sopromise_join['ALLOWWEEK'] = df_exp_sopromise_join[ESTWOS.WEEK]

    df_exp_sopromise_join['MAPPINGTYPE'] = df_exp_sopromise_join[ESTWOS.PRODUCTGROUP].fillna('')+'::'+df_exp_sopromise_join[ESTWOS.ATTB09].fillna('')+'::'+df_exp_sopromise_join[ESTWOS.SITEID].fillna('')

    #ROW_NUMBER () OVER (PARTITION BY SALESORDERID, SOLINENUM ORDER BY C.WEEK DESC) AS RK
    df_exp_sopromise_join['RK'] = df_exp_sopromise_join.groupby([ND_.SALESORDERID, ND_.SOLINENUM])[ESTWOS.WEEK].cumcount(ascending=False)+1

    df_exp_sopromise_join = df_exp_sopromise_join[
        df_exp_sopromise_join['RK'] == 1
    ]

    df_mst_estore_salesorder = df_exp_sopromise_join[[
        ND_.SALESORDERID, ND_.SOLINENUM, ND_.DEMANDPRIORITY, 'WEEK_NEW',
        'WEEK_ORG', ND_.PROMISEDDELDATE, 'PROMISEDDELDATE_START', 'ALLOWWEEK',
        ESIS.TARGETDATEEARLINESS, 'MAPPINGTYPE'
    ]]

    #MTA_CODEMAP 'HC_WL_CHANGE_WEEKPRIORITY' 조건 필터링
    df_mta_codemap_view = df_mta_codemap[df_mta_codemap[CM.CODEMAPKEY].str.split('::').str[0] =='HC_WL_CHANGE_WEEKPRIORITY']

    midx_mves_itemsite = pd.MultiIndex.from_frame(df_mves_itemsite[[ESIS.ITEM, ESIS.SALESID, ESIS.SITEID]])
    midx_mst_estore_salesorder = pd.MultiIndex.from_frame(df_mst_estore_salesorder[[ND_.SALESORDERID, ND_.SOLINENUM]])

    df_exp_sopromise_view = df_exp_sopromise[
        (pd.MultiIndex.from_frame(df_exp_sopromise[[ND_.ITEMID, ND_.SALESID, ND_.SITEID,]]).isin(midx_mves_itemsite))
    ]
    df_exp_sopromise_view = df_exp_sopromise_view[
        (~pd.MultiIndex.from_frame(df_exp_sopromise_view[[ND_.SALESORDERID, ND_.SOLINENUM]]).isin(midx_mst_estore_salesorder))
    ]

    df_exp_sopromise_view = pd.merge(df_exp_sopromise_view, df_mta_codemap_view, left_on=[ND_.SALESID], right_on=[CM.TXT1], suffixes=('', '_B'))

    df_exp_sopromise_view = df_exp_sopromise_view[
        ( (df_exp_sopromise_view[CM.CODE2_TXT] == '-') | (df_exp_sopromise_view[ND_.ITEMID] == df_exp_sopromise_view[CM.CODE2_TXT]) )
    ]

    #df_exp_sopromise_view['DEMANDPRIORITY'] = df_exp_sopromise_view['DEMANDPRIORITY'].astype(int).astype(str)

    df_exp_sopromise_view['WEEK_ORG'] = df_exp_sopromise_view[ND_.DEMANDPRIORITY].str[1:3]

    #week_new = df_exp_sopromise_view['DEMANDPRIORITY'].str[1:3].astype(int) - df_exp_sopromise_view[CM.NUM1].fillna(0).astype(int)

    df_exp_sopromise_view['WEEK_NEW_INT'] = df_exp_sopromise_view['WEEK_ORG'].astype(int)
    df_exp_sopromise_view[CM.NUM1] = pd.to_numeric(df_exp_sopromise_view[CM.NUM1], errors='coerce').astype(int)
    df_exp_sopromise_view['WEEK_NEW_INT'] -= df_exp_sopromise_view[CM.NUM1]

    df_exp_sopromise_view['WEEK_NEW'] = np.where(
        df_exp_sopromise_view['WEEK_NEW_INT'] < 0,
        '00',
        np.where(
            df_exp_sopromise_view['WEEK_NEW_INT'].astype(str).str.len() == 1,
            df_exp_sopromise_view['WEEK_NEW_INT'].astype(str).str.zfill(2),
            df_exp_sopromise_view['WEEK_NEW_INT'].astype(str)
        )
    )
    # df_exp_sopromise_view['WEEK_NEW'] = np.where( week_new < 0, '00', 
    #     np.where( week_new.astype(str).str.len() == 1, week_new.astype(str).str.zfill(2), week_new.astype(str) )
    # )

    #df_exp_sopromise_concat = df_exp_sopromise_view[['SALESORDERID', 'SOLINENUM', 'DEMANDPRIORITY', 'WEEK_NEW', 'WEEK_ORG', 'PROMISEDDELDATE']]
    #df_mst_estore_salesorder = pd.concat([df_mst_estore_salesorder, df_exp_sopromise_concat], ignore_index=True)

    concat_columns = [ND_.SALESORDERID, ND_.SOLINENUM, ND_.DEMANDPRIORITY, 'WEEK_NEW', 'WEEK_ORG', ND_.PROMISEDDELDATE,]

    df_mst_estore_salesorder = pd.concat([df_mst_estore_salesorder, df_exp_sopromise_view[concat_columns]], ignore_index=True)

    #df_exp_sopromise_view[['WEEK_NEW','WEEK_NEW_INT']]

    ###inbound (EXP_SOPROMISE) 의 WEEK, DEMANDPRIORITY, GLOBALPRIORITY, LOCALPRIORITY 값을 업데이트
    df_exp_sopromise = pd.merge(df_exp_sopromise, df_mst_estore_salesorder, how='left', on=[ND_.SALESORDERID, ND_.SOLINENUM], suffixes=('', '_ES'))

    #WEEK_NEW 하나만 있음(두번 돌리면 WEEK_NEW_ES 생김 주의)
    cond_update = ~df_exp_sopromise['WEEK_NEW'].isna()

    df_exp_sopromise.loc[cond_update, ND_.WEEKRANK] = df_exp_sopromise['WEEK_NEW'].fillna(0).astype(int)
    
    start_pos, next_pos, total_length = find_priority_position('G_R001::1', 'WEEKRANK')
    digit = next_pos - start_pos

    df_exp_sopromise.loc[cond_update, ND_.DEMANDPRIORITY] = ( df_exp_sopromise[ND_.DEMANDPRIORITY].str[:start_pos] 
                                                                + df_exp_sopromise['WEEK_NEW'].str.zfill(digit) 
                                                                + df_exp_sopromise[ND_.DEMANDPRIORITY].str[next_pos:] )
    
    df_exp_sopromise.loc[cond_update, ND_.GLOBALPRIORITY] = df_exp_sopromise[ND_.DEMANDPRIORITY]
    df_exp_sopromise.loc[cond_update, ND_.LOCALPRIORITY] = df_exp_sopromise[ND_.DEMANDPRIORITY]

    df_exp_sopromise = df_exp_sopromise.drop(df_exp_sopromise.filter(regex='_ES$').columns, axis=1)

    ###MERGE문 STOCK TRANSIT TIME 이 있는 경우 그 만큼의 WEEK 우선순위, WEEK 값에서 차감해줌
    #안쪽 서브쿼리 B
    df_mves_itemsite[ESIS.STOCKTRANSITTIME] = df_mves_itemsite[ESIS.STOCKTRANSITTIME].astype(int)

    df_mves_itemsite_view = df_mves_itemsite[[ESIS.ITEM, ESIS.SITEID, ESIS.SALESID, ESIS.STOCKTRANSITTIME]]

    df_mves_itemsite_view = df_mves_itemsite_view.groupby([ESIS.ITEM, ESIS.SITEID, ESIS.SALESID]).agg({
        ESIS.STOCKTRANSITTIME:'max'
    }).reset_index()

    df_mves_itemsite_view[ESIS.STOCKTRANSITTIME] = np.round(df_mves_itemsite_view[ESIS.STOCKTRANSITTIME]/7).astype(int)

    #서브 A, B 조인
    df_exp_sopromise_merge_join = pd.merge(df_exp_sopromise, df_mves_itemsite_view, left_on=[ND_.ITEMID, ND_.SITEID, ND_.SALESID], right_on=[ESIS.ITEM, ESIS.SITEID, ESIS.SALESID])

    df_exp_sopromise_merge_join = df_exp_sopromise_merge_join.astype({
        ND_.WEEKRANK:int,
        ESIS.STOCKTRANSITTIME:int
    })

    df_exp_sopromise_merge_join['WEEK_NEW'] = np.where(
        df_exp_sopromise_merge_join[ND_.WEEKRANK] - df_exp_sopromise_merge_join[ESIS.STOCKTRANSITTIME] < 0, 
        0, 
        df_exp_sopromise_merge_join[ND_.WEEKRANK] - df_exp_sopromise_merge_join[ESIS.STOCKTRANSITTIME]).astype(str)


    df_exp_sopromise_merge_join['NEW_DEMANDPRIORITY'] = df_exp_sopromise_merge_join['WEEK_NEW'].str.zfill(digit)

    df_exp_sopromise_merge_join['NEW_LOCALPRIORITY'] = (df_exp_sopromise_merge_join[ND_.LOCALPRIORITY].str[:start_pos] 
                                                        + df_exp_sopromise_merge_join['NEW_DEMANDPRIORITY'] 
                                                        + df_exp_sopromise_merge_join[ND_.LOCALPRIORITY].str[next_pos:])
    
    df_exp_sopromise_merge_join['NEW_GLOBALPRIORITY'] = (df_exp_sopromise_merge_join[ND_.GLOBALPRIORITY].str[:start_pos] 
                                                        + df_exp_sopromise_merge_join['NEW_DEMANDPRIORITY'] 
                                                        + df_exp_sopromise_merge_join[ND_.GLOBALPRIORITY].str[next_pos:])
    
    df_exp_sopromise_merge_join['NEW_DEMANDPRIORITY'] = (df_exp_sopromise_merge_join[ND_.DEMANDPRIORITY].str[:start_pos] 
                                                        + df_exp_sopromise_merge_join['NEW_DEMANDPRIORITY'] 
                                                        + df_exp_sopromise_merge_join[ND_.DEMANDPRIORITY].str[next_pos:])

    ### ON ~ UPATE
    df_exp_sopromise_result = pd.merge(
        df_exp_sopromise,
        df_exp_sopromise_merge_join[[ND_.SALESORDERID, ND_.SOLINENUM, 'WEEK_NEW', 'NEW_DEMANDPRIORITY', 'NEW_GLOBALPRIORITY', 'NEW_LOCALPRIORITY']],
        how='left', on=[ND_.SALESORDERID, ND_.SOLINENUM], suffixes=('', '_U')
    )

    #WEEK = B.NEW_WEEK
    df_exp_sopromise_result.loc[~df_exp_sopromise_result['WEEK_NEW_U'].isna(), ND_.WEEKRANK] = df_exp_sopromise_result['WEEK_NEW_U']

    df_exp_sopromise_result.loc[~df_exp_sopromise_result['NEW_LOCALPRIORITY'].isna(), ND_.LOCALPRIORITY] = df_exp_sopromise_result['NEW_LOCALPRIORITY']
    df_exp_sopromise_result.loc[~df_exp_sopromise_result['NEW_GLOBALPRIORITY'].isna(), ND_.GLOBALPRIORITY] = df_exp_sopromise_result['NEW_GLOBALPRIORITY']
    df_exp_sopromise_result.loc[~df_exp_sopromise_result['NEW_DEMANDPRIORITY'].isna(), ND_.DEMANDPRIORITY] = df_exp_sopromise_result['NEW_DEMANDPRIORITY']

    df_exp_sopromise_result = df_exp_sopromise_result.drop(df_exp_sopromise_result.filter(regex='_U$').columns, axis=1)


    #결과 형식 맞추기
    df_exp_sopromise_result[ND_.WEEKRANK] = df_exp_sopromise_result[ND_.WEEKRANK].str.zfill(2)
    df_exp_sopromise_result[ND_.OPTION_CODE] = pd.to_numeric(df_exp_sopromise_result[ND_.OPTION_CODE], errors='coerce').astype('Int32')
    df_exp_sopromise_result[ND_.QTYPROMISED] = pd.to_numeric(df_exp_sopromise_result[ND_.QTYPROMISED], errors='coerce').astype('Int32')
    

    # demand modify에 들어갈 estore sales order 컬럼 이름 변경
    df_mst_estore_salesorder.rename(columns={
        ND_.SALESORDERID: 'SALESORDERID',
        ND_.SOLINENUM: 'SOLINENUM',
        ND_.DEMANDPRIORITY: 'DEMANDPRIORITY',
        'WEEK_NEW': 'WEEK_NEW',
        'WEEK_ORG': 'WEEK_ORG',
        ND_.PROMISEDDELDATE: 'PROMISEDDELDATE',
        'PROMISEDDELDATE_START': 'PROMISEDDELDATE_START',
        'ALLOWWEEK': 'ALLOWWEEK',
        ESIS.TARGETDATEEARLINESS: 'TARGETDATEEARLINESS',
        'MAPPINGTYPE': 'MAPPINGTYPE',
    }, inplace=True)

    logger.Step(1, 'End eStore Netting')
    return df_exp_sopromise_result[idx_exp_sopromise], df_mst_estore_salesorder


