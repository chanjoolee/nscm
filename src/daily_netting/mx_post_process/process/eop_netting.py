'''
EOP Netting에 사용되는 상수, 함수 등이 정의된 모듈
'''
from math import floor
from itertools import groupby

import pandas as pd
import numpy as np

from daily_netting.mx_post_process.constant.dim_constant import Location as L
from daily_netting.mx_post_process.constant.ods_constant import NettedDemandD as ND_
from daily_netting.mx_post_process.constant.general_constant import BOD as BOD
from daily_netting.mx_post_process.constant.mx_constant import MXItemSellerMap as MXISM
from daily_netting.mx_post_process.constant.mx_constant import CustomModelMap as CMM
from daily_netting.mx_post_process.constant.mx_constant import SalesBOMMap as SBOMM
from daily_netting.mx_post_process.constant.mx_constant import ModelEOP as MEOP
from daily_netting.mx_post_process.constant.mx_constant import InventoryD as Inv_
from daily_netting.mx_post_process.constant.mx_constant import IntransitD as Int_
from daily_netting.mx_post_process.constant.mx_constant import InventorySellD as InvS_
from daily_netting.mx_post_process.constant.mx_constant import IntransitSellD as IntS_
from daily_netting.mx_post_process.constant.mx_constant import EOPDemandD as EOPD_
from daily_netting.mx_post_process.utils.NSCMCommon import G_Logger
from daily_netting.mx_post_process.utils.common import convert_dt_to_week, make_sequential_id

LOGGER = G_Logger
DF = pd.DataFrame

class EOPConstant:
    '''Constant of EOP'''

    SITEID = 'SITEID'              ;    SITEID_IDX = 0
    ITEM = 'ITEM'                  ;    ITEM_IDX = 1
    WEEK = 'WEEK'                  ;    WEEK_IDX = 2
    EOPWEEK = 'EOPWEEK'            ;    EOPWEEK_IDX = 3
    INVQTY = 'INVQTY'              ;    INVQTY_IDX = 4
    REMAINQTY = 'REMAINQTY'        ;    REMAINQTY_IDX = 5
    AVAILQTY = 'AVAILQTY'          ;    AVAILQTY_IDX = 6
    DMDQTY = 'DMDQTY'              ;    DMDQTY_IDX = 7
    RANK = 'RANK'                  ;    RANK_IDX = 8
    SALESID = 'SALESID'            ;    SALESID_IDX = 9
    EOP = 'EOP'

    LIST_COLUMN = [
        SITEID, ITEM, WEEK, EOPWEEK, INVQTY,
        REMAINQTY, AVAILQTY, DMDQTY, RANK,
    ]

# 전역변수, set_eop_env를 통해 동적으로 값 설정
# 상수 클래스
EOP = EOPConstant

# plan value
PLAN_WEEK = None
START_DATE = None
VERSIONNAME = None
PLANID = None

def set_eop_env(accessor: object) -> None:
    '''동적으로 전역변수를 설정하기 위한 함수'''
    global PLAN_WEEK, START_DATE, VERSIONNAME, PLANID

    # plan 설정
    VERSIONNAME = accessor.plan_version
    PLANID = accessor.plan_id
    PLAN_WEEK = accessor.plan_week
    if accessor.plan_type == 'VPLAN':
        START_DATE = accessor.start_date + pd.Timedelta(weeks=1)
    else:
        START_DATE = accessor.start_date

def do_eop_netting(
        df_inbound: DF, df_custom_model_map: DF, df_intransit: DF, df_inventory: DF,
        df_intransit_sell: DF, df_inventory_sell: DF, df_location: DF, df_model_eop: DF,
        df_sales_bom_map: DF, df_seller_map: DF, logger: LOGGER,
    ) -> None:
    '''
    match code를 사용하지 않는 재고로 우선 demand를 살려주고,
    그 후에 match code를 사용하는 재고로 demand를 살려준다.
    '''
    logger.Note('Start EOP Netting', 20)
    logger.Step(1, 'EOP PreProcess Start')
    df_inventory_sum = _make_inventory(df_location, df_inventory, df_seller_map, df_intransit, False)
    df_inventory_sum_sell = _make_inventory(
        df_location, df_inventory_sell, df_seller_map, df_intransit_sell, True
    )
    # 원본 업데이트를 위한 검색용 트리 생성
    dict_search_tree = _make_search_tree(df_inbound)
    # 수량 비교를 위한 수량 백업
    df_inbound['QTYPROMISED_BACKUP'] = df_inbound[ND_.QTYPROMISED]
    logger.Step(1, 'EOP PreProcess End')

    # seller case가 아닌 경우에 대한 계산
    logger.Step(2, 'Make EOP Dataframe Start')
    df_eop = _make_eop_dataframe(
        df_inbound, df_custom_model_map, df_inventory_sum, df_model_eop,
        df_sales_bom_map, df_seller_map, False,
    )
    logger.Step(2, 'Make EOP Dataframe End')

    logger.Step(3, 'EOP Inventory Qty Rolling Start')
    nparr_eop = _roll_eop_qty(df_eop, False)
    logger.Step(3, 'EOP Inventory Qty Rolling End')

    logger.Step(4, 'Cut EOP Demand Start')
    df_inbound = _cut_eop_demand(df_inbound, nparr_eop, dict_search_tree, df_inventory_sum, df_sales_bom_map)
    logger.Step(4, 'Cut EOP Demand End')

    logger.Step(5, 'Make EOP History Start')
    df_history = _make_history(df_inbound, nparr_eop, False)
    logger.Step(5, 'Make EOP History End')

    # seller case 계산
    logger.Step(6, 'Make EOP Sell Dataframe Start')
    df_eop = _make_eop_dataframe(
        df_inbound, df_custom_model_map, df_inventory_sum_sell, df_model_eop,
        df_sales_bom_map, df_seller_map, True,
    )
    logger.Step(6, 'Make EOP Sell Dataframe End')

    logger.Step(7, 'EOP Inventory Sell Qty Rolling Start')
    nparr_eop = _roll_eop_qty(df_eop, True)
    logger.Step(7, 'EOP Inventory Sell Qty Rolling End')
    
    logger.Step(8, 'Cut EOP Sell Demand Start')
    df_demand = _cut_eop_demand_sell(df_inbound, nparr_eop, dict_search_tree)
    logger.Step(8, 'Cut EOP Sell Demand End')

    logger.Step(9, 'Make EOP Sell History Start')
    df_history_sell = _make_history(df_demand, nparr_eop, True)
    logger.Step(9, 'Make EOP Sell History End')

    logger.Step(10, 'Concat EOP History Start')
    df_eop_demand = pd.concat([df_history, df_history_sell], ignore_index=True)
    df_eop_demand[EOPD_.VERSIONNAME] = VERSIONNAME
    df_eop_demand[EOPD_.SEQUENCE] = make_sequential_id('', 0, df_eop_demand)
    df_eop_demand[EOPD_.PLANID] = PLANID
    logger.Step(10, 'Concat EOP History End')

    logger.Note('End EOP Netting', 20)

    return df_demand[ND_.LIST_COLUMN], df_eop_demand

def _make_history(df_inbound: DF, nparr_eop: np.ndarray, is_seller_case: bool) -> DF:
    '''
    eop 대상에 대한 Qty 변화를 저장한다.
    '''

    salesid = []
    if is_seller_case:
        salesid = [EOP.SALESID]
    
    # eop 데이터 정제
    df_eop = pd.DataFrame(data=nparr_eop, columns=EOP.LIST_COLUMN + salesid)
    df_eop = df_eop[[EOP.SITEID, EOP.ITEM, EOP.EOPWEEK, EOP.INVQTY,] + salesid]
    df_eop[EOP.INVQTY] = df_eop[EOP.INVQTY].astype('int32')
    df_eop = df_eop.groupby(by=[EOP.SITEID, EOP.ITEM, EOP.EOPWEEK] + salesid)[[EOP.INVQTY]].sum().reset_index()
    
    df_history = df_inbound[[
        ND_.SALESORDERID, ND_.SOLINENUM, ND_.ITEMID, ND_.QTYPROMISED, 'QTYPROMISED_BACKUP',
        ND_.PROMISEDDELDATE, ND_.SITEID, ND_.SALESID, ND_.DEMANDPRIORITY, ND_.GCID
    ]]
    df_history = df_history[
        (~df_history[ND_.SALESORDERID].str.startswith('UNF_ORD'))
        & (df_history['QTYPROMISED_BACKUP'] > 0)
    ]
    if is_seller_case:
        df_history = pd.merge(
            df_history, df_eop, how='inner',
            left_on=[ND_.SITEID, ND_.ITEMID, ND_.SALESID],
            right_on=[EOP.SITEID, EOP.ITEM, EOP.SALESID]
        )
    else:
        df_history = pd.merge(
            df_history, df_eop, how='inner',
            left_on=[ND_.SITEID, ND_.ITEMID],
            right_on=[EOP.SITEID, EOP.ITEM]
        )
    df_history[[EOPD_.VERSIONNAME, EOPD_.SEQUENCE, EOPD_.PLANID]] = np.nan
    df_history[ND_.PROMISEDDELDATE] = convert_dt_to_week(df_history[ND_.PROMISEDDELDATE])
    df_history = df_history.rename(columns={
        ND_.SALESORDERID: EOPD_.SALESORDERID,
        ND_.SOLINENUM: EOPD_.SOLINENUM,
        ND_.ITEMID: EOPD_.ITEM,
        'QTYPROMISED_BACKUP': EOPD_.QTYPROMISED,
        EOP.INVQTY: EOPD_.INV_QTY,
        ND_.QTYPROMISED: EOPD_.MODIFY_QTY,
        ND_.PROMISEDDELDATE: EOPD_.WEEK,
        EOP.EOPWEEK: EOPD_.EOPWEEK,
        ND_.SITEID: EOPD_.SITEID,
        ND_.SALESID: EOPD_.SALESID,
        ND_.DEMANDPRIORITY: EOPD_.DEMANDPRIORITY,
        ND_.GCID: EOPD_.GC,
    })
    df_history = df_history[EOPD_.LIST_COLUMN]

    return df_history

def _cut_eop_demand(
    df_inbound: DF, nparr_eop: np.ndarray, dict_item_filter: dict,
    df_inventory: DF, df_sales_bom_map: DF,
) -> None:
    QTYPROMISED_BACKUP_IDX = df_inbound.columns.get_loc('QTYPROMISED_BACKUP')
    nparr_demand = df_inbound.to_numpy()

    list_candidate_cutted_demand_row_idx = []
    for eop_row in nparr_eop:
        # 상위 레벨 qty들, 동일한 item, site_id, week 단위로 나누어 sum을 한 수량
        available_qty, demand_qty = eop_row[EOP.AVAILQTY_IDX], eop_row[EOP.DMDQTY_IDX]
        item, site_id = eop_row[EOP.ITEM_IDX], eop_row[EOP.SITEID_IDX]
        # promisedeldate는 + 6일한 일요일 기준
        week = eop_row[EOP.WEEK_IDX] + pd.Timedelta(days=6)
        if demand_qty > available_qty:
            list_demand_row = [] # (priority, promised_qty, row)
            if item not in dict_item_filter: continue
            dict_site_filter = dict_item_filter[item]
            if site_id not in dict_site_filter: continue
            dict_week_filter = dict_site_filter[site_id]
            if week not in dict_week_filter: continue
            dict_sales_filter = dict_week_filter[week]
            for list_row_idx in dict_sales_filter.values():
                for row_idx in list_row_idx:
                    row = nparr_demand[row_idx]
                    if row[ND_.QTYPROMISED_IDX] > 0:
                        list_demand_row.append((
                            row[ND_.DEMANDPRIORITY_IDX],
                            row[ND_.QTYPROMISED_IDX],
                            row[ND_.SALESID_IDX],
                            row_idx,
                        ))
            # 정렬 후 우선순위끼리 모으기
            list_demand_row.sort(key=lambda x: (x[0])) # priority 기준 정렬
            for key, group in groupby(list_demand_row, key=lambda x: x[0]):
                group = list(group)
                group_total_qty = sum([qty for priority, qty, sales_id, row_idx in group])

                if available_qty >= group_total_qty:
                    available_qty -= group_total_qty
                elif available_qty > 0:
                    # 나눠줄 수량이 있지만 필요한 수량보다는 적을 경우.
                    group.sort(key=lambda x: (x[1], x[2])) # qty, sales_id 기준 정렬
                    first_available_qty = available_qty # 첫 수량을 저장
                    for priority, qty, sales_id, row_idx in group:
                        row = nparr_demand[row_idx]
                        new_qty = floor(qty * first_available_qty / group_total_qty)
                        row[ND_.QTYPROMISED_IDX] = new_qty
                        available_qty -= new_qty
                        list_candidate_cutted_demand_row_idx.append(row_idx)
                    for priority, qty, sales_id, row_idx in group:
                        row = nparr_demand[row_idx]
                        if available_qty == 0:
                            break
                        row[ND_.QTYPROMISED_IDX] += 1
                        available_qty -= 1
                    if available_qty != 0:
                        raise ValueError("Not allocated available_qty exists!")
                else: # available_qty = 0인 경우.
                    for priority, qty, sales_id, row_idx in group:
                        row = nparr_demand[row_idx]
                        row[ND_.QTYPROMISED_IDX] = 0
                        list_candidate_cutted_demand_row_idx.append(row_idx)

    # 공용화 재고 생성
    df_rep_inventory_a = df_inventory.copy()
    df_rep_inventory_a['MOD_SITEID'] = df_rep_inventory_a.groupby(EOP.ITEM)[EOP.SITEID].transform('max')
    df_rep_inventory_a[EOP.SITEID] = np.where(df_rep_inventory_a[EOP.SITEID] == '-',
                                            df_rep_inventory_a['MOD_SITEID'],
                                            df_rep_inventory_a[EOP.SITEID])
    df_rep_inventory_a = df_rep_inventory_a[df_rep_inventory_a[EOP.SITEID] != '-']
    df_rep_inventory_a = df_rep_inventory_a.groupby(by=[EOP.SITEID, EOP.ITEM])[[EOP.AVAILQTY]].sum().reset_index()
    df_rep_inventory_a[EOP.WEEK] = PLAN_WEEK

    df_rep_inventory_b = df_sales_bom_map[df_sales_bom_map[SBOMM.ISPOSTPONEMENT] == 'Y']
    df_rep_inventory_b = df_rep_inventory_b[df_rep_inventory_b[SBOMM.STATUS] == 'CON']
    df_rep_inventory_b = df_rep_inventory_b[[SBOMM.SITEID, SBOMM.REPMAINSKU]]
    df_rep_inventory_b = df_rep_inventory_b.drop_duplicates()

    df_rep_inventory = pd.merge(df_rep_inventory_a,
                                df_rep_inventory_b,
                                how='inner',
                                left_on=[EOP.ITEM, EOP.SITEID],
                                right_on=[SBOMM.REPMAINSKU, SBOMM.SITEID])
    df_rep_inventory = df_rep_inventory.groupby(by=[EOP.SITEID, EOP.ITEM, EOP.WEEK])[[EOP.AVAILQTY]].sum().reset_index()

    # dict head to main 생성
    df_bom_dict = df_sales_bom_map[df_sales_bom_map[SBOMM.ISPOSTPONEMENT] == 'Y']
    df_bom_dict = df_bom_dict[df_bom_dict[SBOMM.STATUS] == 'CON']
    dict_head_to_main = {head: main for head, main in zip(df_bom_dict[SBOMM.HEADSKU], df_bom_dict[SBOMM.REPMAINSKU])}

    # 공용화 재고에 index 설정
    df_rep_inventory.set_index(keys=[EOP.SITEID, EOP.ITEM], inplace=True)
    dict_rep_site_item = set(df_rep_inventory.index)

    # qty가 잘린 row만 추출
    list_cutted_demand_row_idx = []
    for demand_row_idx in list_candidate_cutted_demand_row_idx:
        demand_row = nparr_demand[demand_row_idx]
        if demand_row[QTYPROMISED_BACKUP_IDX] > demand_row[ND_.QTYPROMISED_IDX] \
            and demand_row[ND_.ITEMID_IDX] in dict_head_to_main \
            and (demand_row[ND_.SITEID_IDX], dict_head_to_main[demand_row[ND_.ITEMID_IDX]]) in dict_rep_site_item:
            list_cutted_demand_row_idx.append(demand_row_idx)

    # 잘린 demand들에 수량 분배
    list_cutted_demand_row_idx.sort(key=lambda row_idx:
                                (nparr_demand[row_idx][ND_.SITEID_IDX],
                                dict_head_to_main[nparr_demand[row_idx][ND_.ITEMID_IDX]],
                                nparr_demand[row_idx][ND_.PROMISEDDELDATE_IDX],
                                nparr_demand[row_idx][ND_.DEMANDPRIORITY_IDX]))

    prev_rep_item, prev_site_id = '', ''
    for demand_row_idx in list_cutted_demand_row_idx:
        demand_row = nparr_demand[demand_row_idx]
        rep_item = dict_head_to_main[demand_row[ND_.ITEMID_IDX]]
        site_id = demand_row[ND_.SITEID_IDX]
        week = demand_row[ND_.PROMISEDDELDATE_IDX].strftime('%G%V')

        if not (prev_rep_item == rep_item and prev_site_id == site_id):
            inventory_row = df_rep_inventory.loc[(site_id, rep_item)]
            inventory_qty = inventory_row[EOP.AVAILQTY]
            inventory_week = inventory_row[EOP.WEEK]
            prev_rep_item, prev_site_id = rep_item, site_id
        
        if inventory_qty == 0 or week < inventory_week:
            continue

        distribute_qty = min(inventory_qty, demand_row[QTYPROMISED_BACKUP_IDX] - demand_row[ND_.QTYPROMISED_IDX])
        demand_row[ND_.QTYPROMISED_IDX] += distribute_qty
        inventory_qty -= distribute_qty

    df_cutted_demand = pd.DataFrame(data=nparr_demand, columns=ND_.LIST_COLUMN + ['QTYPROMISED_BACKUP']).astype({
        ND_.PROMISEDDELDATE: 'datetime64[ns]',
        ND_.QTYPROMISED: 'int32',
        'QTYPROMISED_BACKUP': 'int32'
    })
    return df_cutted_demand


def _cut_eop_demand_sell(
    df_inbound: DF, nparr_eop: np.ndarray, dict_item_filter: dict
) -> None:
    '''
    seller case에 대해 eop 재고가 없는 demand는 그 수량을 자른다.
    '''
    nparr_demand = df_inbound.to_numpy()

    for eop_row in nparr_eop:
        # 상위 레벨 qty들, 동일한 item, site_id, week 단위로 나누어 sum을 한 수량
        available_qty, demand_qty = eop_row[EOP.AVAILQTY_IDX], eop_row[EOP.DMDQTY_IDX]
        item, site_id = eop_row[EOP.ITEM_IDX], eop_row[EOP.SITEID_IDX]
        # promisedeldate는 + 6일한 일요일 기준
        sales_id, week = eop_row[EOP.SALESID_IDX], eop_row[EOP.WEEK_IDX] + pd.Timedelta(days=6)
        if demand_qty > available_qty:
            list_demand_row = [] # (priority, promised_qty, row)
            if item not in dict_item_filter: continue
            dict_site_filter = dict_item_filter[item]
            if site_id not in dict_site_filter: continue
            dict_week_filter = dict_site_filter[site_id]
            if week not in dict_week_filter: continue
            dict_sales_filter = dict_week_filter[week]
            if sales_id not in dict_sales_filter: continue

            list_row_idx = dict_sales_filter[sales_id]
            for row_idx in list_row_idx:
                row = nparr_demand[row_idx]
                if row[ND_.QTYPROMISED_IDX] > 0:
                    list_demand_row.append((
                        row[ND_.DEMANDPRIORITY_IDX],
                        row[ND_.QTYPROMISED_IDX],
                        row[ND_.SALESID_IDX],
                        row_idx,
                    ))
            # 정렬 후 우선순위끼리 모으기
            list_demand_row.sort(key=lambda x: (x[0])) # priority 기준 정렬
            for key, group in groupby(list_demand_row, key=lambda x: x[0]):
                group = list(group)
                group_total_qty = sum([qty for priority, qty, sales_id, row_idx in group])

                if available_qty >= group_total_qty:
                    available_qty -= group_total_qty
                elif available_qty > 0:
                    # 나눠줄 수량이 있지만 필요한 수량보다는 적을 경우.
                    group.sort(key=lambda x: (x[1], x[2])) # qty, sales_id 기준 정렬
                    first_available_qty = available_qty # 첫 수량을 저장
                    for priority, qty, sales_id, row_idx in group:
                        row = nparr_demand[row_idx]
                        new_qty = floor(qty * first_available_qty / group_total_qty)
                        row[ND_.QTYPROMISED_IDX] = new_qty
                        available_qty -= new_qty
                    for priority, qty, sales_id, row_idx in group:
                        row = nparr_demand[row_idx]
                        if available_qty == 0:
                            break
                        row[ND_.QTYPROMISED_IDX] += 1
                        available_qty -= 1
                    if available_qty != 0:
                        raise ValueError("Not allocated available_qty exists!")
                else: # available_qty = 0인 경우.
                    for priority, qty, sales_id, row_idx in group:
                        row = nparr_demand[row_idx]
                        row[ND_.QTYPROMISED_IDX] = 0

    df_cutted_demand = pd.DataFrame(
        data=nparr_demand, columns=ND_.LIST_COLUMN + ['QTYPROMISED_BACKUP']
    ).astype({
        ND_.PROMISEDDELDATE: 'datetime64[ns]',
        ND_.QTYPROMISED: 'int32',
        'QTYPROMISED_BACKUP': 'int32',
    })

    return df_cutted_demand

def _roll_eop_qty(df_eop: DF, is_seller_case: bool) -> np.ndarray:
    '''
    앞주차의 재고를 고려해, 각 주마다의 가용량을 계산한다.
    '''

    salesid = []
    if is_seller_case:
        salesid = [EOP.SALESID]

    df_eop[EOP.WEEK] = pd.to_datetime(df_eop[EOP.WEEK] + '1', format='%G%V%u')
    df_eop[EOP.RANK] = df_eop.groupby(
        by=[EOP.ITEM, EOP.SITEID] + salesid
    )[EOP.WEEK].rank(method='min')
    df_eop = df_eop[EOP.LIST_COLUMN + salesid]
    df_eop = df_eop.sort_values(by=[EOP.SITEID,] + salesid + [EOP.ITEM, EOP.WEEK])

    nparr_eop = df_eop.to_numpy()

    remain_qty = 0
    for row in nparr_eop:
        if row[EOP.RANK_IDX] == 1.0:
            remain_qty = 0
        else:
            row[EOP.REMAINQTY_IDX] = remain_qty
            row[EOP.AVAILQTY_IDX] = row[EOP.INVQTY_IDX] + remain_qty
        
        remain_qty = max(0, row[EOP.AVAILQTY_IDX] - row[EOP.DMDQTY_IDX])
    
    return nparr_eop

def _make_search_tree(df_inbound: DF) -> dict:
    '''
    필요한 row의 index를 검색하기 위한 트리를 만든다.
    트리는 nested dictionary로 이루어져 있고,
    검색 순서는 item -> siteid -> week -> salesid 순이다.
    '''
    dict_item_filter = {}

    for idx, row in enumerate(df_inbound.to_numpy()):
        item = row[ND_.ITEMID_IDX]
        if item not in dict_item_filter:
            dict_item_filter[item] = {}
        
        dict_site_filter = dict_item_filter[item]
        site_id = row[ND_.SITEID_IDX]
        if site_id not in dict_site_filter:
            dict_site_filter[site_id] = {}
        
        dict_week_filter = dict_site_filter[site_id]
        week = row[ND_.PROMISEDDELDATE_IDX]
        if week not in dict_week_filter:
            dict_week_filter[week] = {}
        
        dict_sales_filter = dict_week_filter[week]
        sales_id = row[ND_.AP2ID_IDX]
        if sales_id not in dict_sales_filter:
            dict_sales_filter[sales_id] = []
        
        list_row_idx = dict_sales_filter[sales_id]
        list_row_idx.append(idx)

    return dict_item_filter

def _filter_seller_map(
        df_to_filter: DF, df_seller_map: DF,
        item_site_column: tuple, is_seller_case: bool
    ) -> DF:
    '''
    is_seller_case = True -> seller_map에 포함되는 행만 걸러낸다.
    is_seller_case = False -> seller_map에 포함되지 않는 행만 걸러낸다.
    '''

    item_name, site_name = item_site_column
    df_seller_map_view = df_seller_map[[MXISM.ITEM, MXISM.SITEID]].drop_duplicates()

    seller_condition = (df_to_filter[item_name] + df_to_filter[site_name]).isin(
                        df_seller_map_view[MXISM.ITEM] + df_seller_map_view[MXISM.SITEID])
    
    if is_seller_case:
        df_to_filter = df_to_filter[seller_condition]
    else:
        df_to_filter = df_to_filter[~seller_condition]
    
    return df_to_filter.copy()

def _make_eop_dataframe(
        df_inbound: DF, df_custom_model_map: DF, df_inventory_sum: DF, df_model_eop: DF,
        df_sales_bom_map: DF, df_seller_map: DF, is_seller_case: bool,
    ) -> DF:
    '''
    eop netting을 위한 재고 계산용 임시 dataframe을 만드는 함수.
    '''

    salesid, ap2id = [], []
    if is_seller_case:
        # seller case인 경우 column을 추가한다.
        salesid = [EOP.SALESID]
        ap2id = [ND_.AP2ID]
    # 공통 사용
    df_custom_model_map_view = df_custom_model_map[
                            df_custom_model_map[CMM.ISVALID] == 'Y'
                        ][[CMM.CUSTOMITEM]].drop_duplicates()
    
    # a
    # a-a, 결과 컬럼 -> ['SITEID', 'ITEM', 'WEEK', 'DMDQTY', 'INVQTY']
    df_eop_netting_a_a = df_inbound[[ND_.SITEID, ND_.ITEMID, ND_.PROMISEDDELDATE, ND_.QTYPROMISED,] + ap2id]
    df_eop_netting_a_a = df_eop_netting_a_a[df_eop_netting_a_a[ND_.QTYPROMISED] > 0]
    df_eop_netting_a_a = _filter_seller_map(df_eop_netting_a_a, df_seller_map, (ND_.ITEMID, ND_.SITEID), is_seller_case)

    df_eop_netting_a_a = df_eop_netting_a_a[~df_eop_netting_a_a[ND_.ITEMID].isin(df_custom_model_map_view[CMM.CUSTOMITEM])]
    df_eop_netting_a_a = df_eop_netting_a_a[[ND_.SITEID, ND_.ITEMID, ND_.PROMISEDDELDATE, ND_.QTYPROMISED,] + ap2id]

    df_eop_netting_a_a = df_eop_netting_a_a.groupby(by=[ND_.SITEID, ND_.ITEMID, ND_.PROMISEDDELDATE] + ap2id).sum().reset_index()
    df_eop_netting_a_a[ND_.PROMISEDDELDATE] = convert_dt_to_week(df_eop_netting_a_a[ND_.PROMISEDDELDATE])
    df_eop_netting_a_a.rename(columns={
        ND_.SITEID: EOP.SITEID,
        ND_.ITEMID: EOP.ITEM,
        ND_.PROMISEDDELDATE: EOP.WEEK,
        ND_.QTYPROMISED: EOP.DMDQTY,
        ND_.AP2ID: EOP.SALESID,
    }, inplace=True)
    df_eop_netting_a_a[EOP.INVQTY] = 0
    df_eop_netting_a_a = df_eop_netting_a_a[[EOP.SITEID, EOP.ITEM, EOP.WEEK, EOP.INVQTY, EOP.DMDQTY,] + salesid]

    # a-b
    # a-b-a
    # a-b-a-a
    df_eop_netting_a_b_a_a = df_inventory_sum

    # a-b-a-b,  결과 컬럼 -> ['Location.[Location]', 'Item.[Item]', 'Sub Location.[Sub Location]', 'Item.[Main SKU]']
    df_eop_netting_a_b_a_b = df_sales_bom_map[df_sales_bom_map[SBOMM.PRIORITY] == 1]
    df_eop_netting_a_b_a_b = df_eop_netting_a_b_a_b[[SBOMM.HEADSITEID, SBOMM.HEADSKU, SBOMM.SITEID, SBOMM.MAINSKU,]]

    # a-b-a-a + a-b-a-b -> a-b-a, 결과 컬럼 -> ['SITEID', 'ITEM', 'AVAILQTY', 'MOD_SITEID']
    df_eop_netting_a_b_a = pd.merge(df_eop_netting_a_b_a_a,
                                    df_eop_netting_a_b_a_b,
                                    how='left',
                                    left_on=EOP.ITEM, right_on=SBOMM.MAINSKU)

    df_eop_netting_a_b_a[EOP.SITEID] = df_eop_netting_a_b_a[SBOMM.HEADSITEID].fillna(df_eop_netting_a_b_a[EOP.SITEID])
    df_eop_netting_a_b_a[EOP.ITEM] = df_eop_netting_a_b_a[SBOMM.HEADSKU].fillna(df_eop_netting_a_b_a[EOP.ITEM])
    df_eop_netting_a_b_a = df_eop_netting_a_b_a[[EOP.SITEID, EOP.ITEM, EOP.AVAILQTY] + salesid]
    df_eop_netting_a_b_a['MOD_SITEID'] = df_eop_netting_a_b_a.groupby(EOP.ITEM)[EOP.SITEID].transform('max')

    # a-b-b  결과 컬럼: ['SITEID', 'ITEM', 'QTYPROMISED',]
    df_eop_netting_a_b_b = df_inbound[[ND_.SITEID, ND_.ITEMID, ND_.QTYPROMISED,] + ap2id]
    df_eop_netting_a_b_b = df_eop_netting_a_b_b[df_eop_netting_a_b_b[ND_.QTYPROMISED] > 0]
    df_eop_netting_a_b_b = df_eop_netting_a_b_b[~df_eop_netting_a_b_b[ND_.ITEMID].isin(df_custom_model_map_view[CMM.CUSTOMITEM])]
    df_eop_netting_a_b_b = df_eop_netting_a_b_b[[ND_.ITEMID, ND_.SITEID] + ap2id].drop_duplicates()

    # a-b-a + a-b-b -> a-b, 결과 컬럼: ['SITEID', 'ITEM', 'WEEK', 'INVQTY', 'DMDQTY']
    df_eop_netting_a_b = df_eop_netting_a_b_a
    df_eop_netting_a_b[EOP.SITEID] = np.where(df_eop_netting_a_b[EOP.SITEID] == '-',
                                            df_eop_netting_a_b['MOD_SITEID'],
                                            df_eop_netting_a_b[EOP.SITEID])
    df_eop_netting_a_b = df_eop_netting_a_b[df_eop_netting_a_b[EOP.SITEID] != '-'].copy()
    df_eop_netting_a_b = pd.merge(df_eop_netting_a_b, df_eop_netting_a_b_b, how='inner',
                                left_on=[EOP.ITEM, EOP.SITEID] + salesid,
                                right_on=[ND_.ITEMID, ND_.SITEID] + ap2id)
    df_eop_netting_a_b = df_eop_netting_a_b.groupby(by=[EOP.SITEID, EOP.ITEM] + salesid)[[EOP.AVAILQTY]].sum().reset_index()
    df_eop_netting_a_b[EOP.WEEK] = PLAN_WEEK
    df_eop_netting_a_b[EOP.DMDQTY] = 0
    df_eop_netting_a_b.rename(columns={
        EOP.AVAILQTY: EOP.INVQTY,
    }, inplace=True)
    df_eop_netting_a_b = df_eop_netting_a_b[[EOP.SITEID, EOP.ITEM, EOP.WEEK, EOP.INVQTY, EOP.DMDQTY,] + salesid]

    # a-a + a-b -> a, 결과 컬럼: ['SITEID', 'ITEM', 'WEEK', 'DMDQTY', 'INVQTY']
    df_eop_netting_a = pd.concat([df_eop_netting_a_a, df_eop_netting_a_b], axis=0, ignore_index=True)

    # b  결과 컬럼 -> ['ITEM', 'EOP']
    # b-b
    df_eop_netting_b_b = df_sales_bom_map[df_sales_bom_map[SBOMM.PRIORITY] == 1]
    df_eop_netting_b_b = df_eop_netting_b_b[[SBOMM.HEADSKU, SBOMM.MAINSKU]].rename(columns={SBOMM.HEADSKU: 'HEADSKU'})

    df_eop_netting_b = pd.merge(df_model_eop, df_eop_netting_b_b, how='left', left_on=MEOP.ITEM, right_on=SBOMM.MAINSKU)
    df_eop_netting_b[EOP.ITEM] = df_eop_netting_b['HEADSKU'].fillna(df_eop_netting_b[MEOP.ITEM])
    df_eop_netting_b[EOP.EOP] = np.where(
                            df_eop_netting_b[MEOP.STATUS] == 'COM',
                            df_eop_netting_b[MEOP.EOP_COM_DATE],
                            np.where(
                                df_eop_netting_b[MEOP.STATUS] == 'INI',
                                df_eop_netting_b[MEOP.EOP_CHG_DATE].fillna(df_eop_netting_b[MEOP.EOP_INIT_DATE]),
                                pd.NaT
                            )
                        )

    df_eop_netting_b[EOP.EOP] = pd.to_datetime(df_eop_netting_b[EOP.EOP])
    df_eop_netting_b = df_eop_netting_b[[EOP.ITEM, EOP.EOP]].groupby(by=[EOP.ITEM]).max().reset_index()

    # 시작 날짜보다 먼저 eop가 되는 거 찾음(생산이 종료 돼서 재고만 줄 수 있는 애들)
    comaprison_start_date = START_DATE.floor('D')
    df_eop_netting_b = df_eop_netting_b[df_eop_netting_b[EOP.EOP] <= comaprison_start_date]

    # a + b -> eop_netting
    # ['SITEID', 'ITEM', 'DMDWEEK', 'EOPWEEK', 'INVQTY', 'REMAINQTY', 'AVAILQTY', 'DMDQTY']
    df_eop_netting = pd.merge(df_eop_netting_a, df_eop_netting_b, how='inner', on=EOP.ITEM)
    df_eop_netting[EOP.EOPWEEK] = convert_dt_to_week(df_eop_netting[EOP.EOP])

    df_eop_netting[EOP.DMDQTY] = df_eop_netting[EOP.DMDQTY].fillna(0).astype('int32')
    df_eop_netting[EOP.INVQTY] = df_eop_netting[EOP.INVQTY].fillna(0).astype('int32')

    df_eop_netting = df_eop_netting.groupby(by=[EOP.SITEID, EOP.ITEM, EOP.WEEK, EOP.EOPWEEK,] + salesid)[[EOP.DMDQTY, EOP.INVQTY,]].sum().reset_index()
    df_eop_netting[EOP.REMAINQTY] = 0
    df_eop_netting[EOP.AVAILQTY] = df_eop_netting[EOP.INVQTY]

    return df_eop_netting[[
                EOP.SITEID, EOP.ITEM, EOP.WEEK, EOP.EOPWEEK, EOP.INVQTY,
                EOP.REMAINQTY, EOP.AVAILQTY, EOP.DMDQTY,
           ] + salesid]

def _make_inventory(
        df_location:DF, df_inventory:DF, df_seller_map:DF,
        df_intransit:DF, is_seller_case:bool
    ) -> DF:
    '''
    재고를 합한 dataframe을 만드는 함수이다.
    일반 재고(inventory)와 운송 중인 재고(intransit)를 합한다.

    is_seller_case = True -> sales_id포한한 sell 컬럼들 사용
    is_seller_case = False -> sales_id 없는 컬럼 사용
    '''
    inv_siteid, inv_item, inv_tositeid = Inv_.SITEID, Inv_.ITEM, Inv_.TOSITEID
    inv_availqty, inv_bohaddqty = Inv_.AVAILQTY, Inv_.BOHADDQTY
    inv_w0bohaddqty, inv_salesid = Inv_.W0BOHADDQTY, ''

    int_tositeid, int_item = Int_.TOSITEID, Int_.ITEM
    int_intransitqty, int_salesid = Int_.INTRANSITQTY, ''

    if is_seller_case:
        inv_siteid, inv_item, inv_tositeid = InvS_.SITEID, InvS_.ITEM, InvS_.TOSITEID
        inv_availqty, inv_bohaddqty = InvS_.AVAILQTY, InvS_.BOHADDQTY
        inv_w0bohaddqty, inv_salesid = InvS_.W0BOHADDQTY, InvS_.SALESID

        int_tositeid, int_item = IntS_.TOSITEID, IntS_.ITEM
        int_intransitqty, int_salesid = IntS_.INTRANSITQTY, IntS_.SALESID
    
    list_inv_init_column = [
                            inv_siteid, inv_item, inv_tositeid,
                            inv_availqty, inv_bohaddqty, inv_w0bohaddqty
                        ]
    list_inv_final_column = [EOP.SITEID, inv_item, EOP.AVAILQTY]
    list_int_init_column = [int_tositeid, int_item, int_intransitqty]
    if is_seller_case:
        list_inv_init_column.append(inv_salesid)
        list_inv_final_column.append(inv_salesid)
        list_int_init_column.append(int_salesid)

    df_inventory_view = df_inventory[list_inv_init_column].rename(columns={
                            inv_siteid: EOP.SITEID,
                        })
    df_inventory_view = pd.merge(
            df_inventory_view, df_location[[L.SITEID, L.TYPE]],
            how='inner', left_on=EOP.SITEID, right_on=L.SITEID
        )
    # MF 공장일 경우 옮겨갈 위치 사용, DC 이미 법인일 경우 현위치 사용
    df_inventory_view[EOP.SITEID] = np.where(
                                        df_inventory_view[L.TYPE] == 'MF',
                                        df_inventory_view[inv_tositeid],
                                        np.where(
                                            df_inventory_view[L.TYPE] == 'DC',
                                            df_inventory_view[EOP.SITEID],
                                            np.nan
                                        )
                                    )
    df_filtered_inventory = _filter_seller_map(
            df_inventory_view, df_seller_map,
            (inv_item, EOP.SITEID), is_seller_case
        )
    df_filtered_inventory[EOP.AVAILQTY] = df_filtered_inventory[inv_availqty] \
                                        + df_filtered_inventory[inv_bohaddqty] \
                                        + df_filtered_inventory[inv_w0bohaddqty]
    df_filtered_inventory = df_filtered_inventory[list_inv_final_column]
    df_filtered_inventory = df_filtered_inventory.rename(columns={
                                inv_item: EOP.ITEM,
                                inv_salesid: EOP.SALESID,
                            })

    df_intransit_view = df_intransit[list_int_init_column]
    df_filtered_intransit = _filter_seller_map(
                                df_intransit_view, df_seller_map,
                                (int_item, int_tositeid), is_seller_case
                            )
    df_filtered_intransit = df_filtered_intransit[list_int_init_column]
    df_filtered_intransit = df_filtered_intransit.rename(columns={
                                int_tositeid: EOP.SITEID,
                                int_item: EOP.ITEM,
                                int_intransitqty: EOP.AVAILQTY,
                                int_salesid: EOP.SALESID,
                            })

    return pd.concat([df_filtered_inventory, df_filtered_intransit], axis=0, ignore_index=True)
