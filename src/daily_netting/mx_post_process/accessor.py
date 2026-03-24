from typing import Tuple

import pandas as pd
import numpy as np

from daily_netting.mx_post_process.constant.dim_constant import NettingSales as NS
from daily_netting.mx_post_process.constant.dim_constant import Item as I
from daily_netting.mx_post_process.constant.dim_constant import Location as L
from daily_netting.mx_post_process.constant.dim_constant import NettingLPPlanBatch as NLPPB
from daily_netting.mx_post_process.constant.ods_constant import PlanD as P_
from daily_netting.mx_post_process.constant.ods_constant import PlanOptionD as PO_
from daily_netting.mx_post_process.constant.ods_constant import PriorityRankD as PR_
from daily_netting.mx_post_process.constant.ods_constant import PreDemandD as PD_
from daily_netting.mx_post_process.constant.ods_constant import AP1DeliveryPlanD as AP1DP_
from daily_netting.mx_post_process.constant.ods_constant import AP1ShortReasonD as AP1SR_
from daily_netting.mx_post_process.constant.ods_constant import NettedDemandD as ND_
from daily_netting.mx_post_process.constant.general_constant import AccountInfo as AI
from daily_netting.mx_post_process.constant.general_constant import BOD as BOD
from daily_netting.mx_post_process.constant.general_constant import SSCalendarD as SSC_
from daily_netting.mx_post_process.constant.general_constant import SalesOrderLP as SOLP
from daily_netting.mx_post_process.constant.mx_constant import DPGuide as DPG
from daily_netting.mx_post_process.constant.mx_constant import MXItemSellerMap as MXISM
from daily_netting.mx_post_process.constant.mx_constant import CustomModelMap as CMM
from daily_netting.mx_post_process.constant.mx_constant import SalesBOMMap as SBOMM
from daily_netting.mx_post_process.constant.mx_constant import ModelEOP as MEOP
from daily_netting.mx_post_process.constant.mx_constant import InventoryD as Inv_
from daily_netting.mx_post_process.constant.mx_constant import IntransitD as Int_
from daily_netting.mx_post_process.constant.mx_constant import InventorySellD as InvS_
from daily_netting.mx_post_process.constant.mx_constant import IntransitSellD as IntS_
from daily_netting.mx_post_process.constant.mx_constant import DeliveryPlanD as DP_
from daily_netting.mx_post_process.constant.mx_constant import ShortReasonD as SR_
from daily_netting.mx_post_process.constant.mx_constant import DistributionOrdersD as DO_
from daily_netting.mx_post_process.constant.mx_constant import DistributionOrdersSellD as DOS_
from daily_netting.mx_post_process.constant.mx_constant import CodeMap as CM
from daily_netting.mx_post_process.constant.mx_constant import AvailableResourceD as AR_
from daily_netting.mx_post_process.constant.mx_constant import ESItemSite as ESIS
from daily_netting.mx_post_process.constant.mx_constant import SalesResult as SR
from daily_netting.mx_post_process.constant.mx_constant import DemandSOPPegging as DSOPP
from daily_netting.mx_post_process.constant.mx_constant import BOMComponent as BOMC
from daily_netting.mx_post_process.constant.mx_constant import ChannelStockMaster as CSM
from daily_netting.mx_post_process.constant.mx_constant import ChannelStockAP2Master as CSAP2M
from daily_netting.mx_post_process.constant.mx_constant import EStoreReplenishment as ESR
from daily_netting.mx_post_process.constant.mx_constant import EStoreTWOS as ESTWOS
from daily_netting.mx_post_process.constant.mx_constant import PackageSite as PS
from daily_netting.mx_post_process.constant.mx_constant import Prebuild as P
from daily_netting.mx_post_process.constant.mx_constant import AgingInventory60 as AI60
from daily_netting.mx_post_process.constant.mx_constant import PrebuildException as PE
from daily_netting.mx_post_process.constant.mx_constant import NewFCSTPeriod as NFP
from daily_netting.mx_post_process.constant.mx_constant import RBMaster as RBM
from daily_netting.mx_post_process.constant.mx_constant import ShortReasonDPGuide as SRDPG
from daily_netting.mx_post_process.utils.common import make_sequential_id

def _select_columns_with_optional_groups(
        df: pd.DataFrame, required_columns: list, optional_column_groups: tuple
    ) -> pd.DataFrame:
    list_selected = list(required_columns)
    set_selected = set(list_selected)

    for column_group in optional_column_groups:
        for column in column_group:
            if column in df.columns and column not in set_selected:
                list_selected.append(column)
                set_selected.add(column)
                break

    for column in df.columns:
        if 'CDC' in str(column).upper() and column not in set_selected:
            list_selected.append(column)
            set_selected.add(column)

    return df[list_selected]

class Accessor:
    '''
    input data, plan value 등에 접근하기 위한 클래스.
    '''
    def __init__(
            self, df_dim_item, df_dim_location, df_dim_netting_sales, df_plan, df_plan_option,
            df_priority_rank, df_pre_demand,
            df_dp_guide, df_short_reason_dp_guide, df_rb_master,
            df_mx_item_seller_map, df_custom_model_map, df_inventory, df_intransit, df_sales_bom_map,
            df_model_eop, df_inventory_sell, df_intransit_sell,
            df_es_item_site, df_available_resource, df_code_map,
            df_delivery_plan, df_short_reason, df_distribution_orders,
            df_distribution_orders_sell, df_new_fcst_period, df_sales_result,
            df_channel_stock_master,
            df_channel_stock_ap2_master, df_estore_twos,
            df_estore_replenishment,
            df_prebuild,
            df_dim_netting_lp_plan_batch,
            df_sales_order_lp,
            is_pre_demand: bool = True
        ) -> None:
        # dimension
        self.df_dim_location = df_dim_location[L.LIST_COLUMN]
        self.df_dim_item = df_dim_item[I.LIST_COLUMN]
        self.df_dim_netting_sales = df_dim_netting_sales[NS.LIST_BASE_COLUMN]
        self._extend_netting_sales() # NS.LIST_COLUMN으로 확장
        self.df_dim_netting_lp_plan_batch = df_dim_netting_lp_plan_batch[NLPPB.LIST_COLUMN]
        # ods
        self.df_plan = df_plan[P_.LIST_COLUMN]
        self.df_plan_option = df_plan_option[PO_.LIST_COLUMN]
        self.df_priority_rank = df_priority_rank[PR_.LIST_COLUMN]
        self.df_pre_demand = df_pre_demand[PD_.LIST_COLUMN]
        # general
        self.df_sales_order_lp = df_sales_order_lp[SOLP.LIST_COLUMN]
        # mx
        self.df_short_reason_dp_guide = df_short_reason_dp_guide[SRDPG.LIST_COLUMN]
        self.df_rb_master = df_rb_master[RBM.LIST_COLUMN]
        self.df_dp_guide = df_dp_guide[DPG.LIST_COLUMN]
        self.df_mx_item_seller_map = df_mx_item_seller_map[MXISM.LIST_COLUMN]
        self.df_custom_model_map = df_custom_model_map[CMM.LIST_COLUMN]
        self.df_sales_bom_map = df_sales_bom_map[SBOMM.LIST_COLUMN]
        self.df_model_eop = df_model_eop[MEOP.LIST_COLUMN]
        self.df_inventory = _select_columns_with_optional_groups(
            df_inventory, Inv_.LIST_COLUMN, Inv_.OPTIONAL_COLUMN_GROUPS
        )
        self.df_intransit = df_intransit[Int_.LIST_COLUMN]
        self.df_inventory_sell = _select_columns_with_optional_groups(
            df_inventory_sell, InvS_.LIST_COLUMN, InvS_.OPTIONAL_COLUMN_GROUPS
        )
        self.df_intransit_sell = df_intransit_sell[IntS_.LIST_COLUMN]
        self.df_code_map = df_code_map[CM.LIST_COLUMN]
        self.df_es_item_site = df_es_item_site[ESIS.LIST_COLUMN]
        self.df_sales_result = df_sales_result[SR.LIST_BASE_COLUMN]
        self._extend_sales_result() # SR.LIST_COLUMN으로 확장
        self.df_channel_stock_master = df_channel_stock_master[CSM.LIST_COLUMN]
        self.df_channel_stock_ap2_master = df_channel_stock_ap2_master[CSAP2M.LIST_COLUMN]
        self.df_estore_replenishment = df_estore_replenishment[ESR.LIST_COLUMN]
        self.df_estore_twos = df_estore_twos[ESTWOS.LIST_COLUMN]
        self.df_prebuild = df_prebuild[P.LIST_COLUMN]
        self.df_new_fcst_period = df_new_fcst_period[NFP.LIST_COLUMN]
        self.df_available_resource = df_available_resource[AR_.LIST_COLUMN]
        self.df_delivery_plan = df_delivery_plan[DP_.LIST_COLUMN]
        self.df_short_reason = _select_columns_with_optional_groups(
            df_short_reason, SR_.LIST_BASE_COLUMN, SR_.OPTIONAL_COLUMN_GROUPS
        )
        self.df_distribution_orders = df_distribution_orders[DO_.LIST_COLUMN]
        self.df_distribution_orders_sell = df_distribution_orders_sell[DOS_.LIST_COLUMN]

        self._convert_empty_type()

        # plan data 세팅
        self._set_plan_data()
        if is_pre_demand:
            self.df_ap1_short_reason, self.df_ap1_delivery_plan, self.df_demand = \
                self._process_pre_demand()
        # find_priority_position 생성
        self.find_priority_position = self._make_find_priority_position()

    def _set_plan_data(self) -> None:
        df_plan = self.df_plan

        self.plan_version: str = df_plan[P_.VERSIONNAME].iloc[0]
        self.plan_id: str = df_plan[P_.PLANID].iloc[0]
        self.plan_type: str = df_plan[P_.PLANTYPE].iloc[0]
        self.plan_week: str = df_plan[P_.PLANWEEK].iloc[0]
        self.plan_horizon: float = df_plan[P_.PLANHORIZON].iloc[0]
        self.division_id: str = df_plan[P_.DIVISIONID].iloc[0]
        self.division_name: str = df_plan[P_.DIVISIONNAME].iloc[0]
        self.start_date: pd.Timestamp = df_plan[P_.STARTDATE].iloc[0]
        self.end_date: pd.Timestamp = df_plan[P_.ENDDATE].iloc[0]

    def _make_find_priority_position(self):
            nparr_priority_rank = self.df_priority_rank.to_numpy()
            dict_priority = {}
            for row in nparr_priority_rank:
                if row[PR_.ISVALID_IDX] != 'Y': continue
                rule_key = '::'.join([row[PR_.RULEID_IDX], row[PR_.RULESEQUENCE_IDX]])
                if rule_key not in dict_priority:
                    dict_priority[rule_key] = dict()
                dict_category = dict_priority[rule_key]
                dict_category[row[PR_.RANKCATEGORY_IDX]] = (row[PR_.DIGIT_IDX], row[PR_.ORDER_IDX])

            def find_priority_position(rule_key, category_to_search):
                if rule_key not in dict_priority:
                    raise ValueError('Unknown rule_key')
                
                list_sorted = sorted(dict_priority[rule_key].items(), key=lambda x: x[1][1])
                start_position = 0
                for category, (digit, order) in list_sorted:
                    next_start_position = start_position + digit
                    if category == category_to_search:
                        break
                    start_position += digit

                total_length = sum([digit for category, (digit, order) in list_sorted])
                
                return int(start_position), int(next_start_position), int(total_length)
            
            return find_priority_position

    def _extend_netting_sales(self) -> None:
        df_dim_netting_sales: pd.DataFrame = self.df_dim_netting_sales.copy()
        df_dim_netting_sales[NS.ACCOUNTID] = np.nan
        df_dim_netting_sales[NS.AP1ID] = np.nan
        df_dim_netting_sales[NS.AP2ID] = np.nan
        df_dim_netting_sales[NS.GCID] = np.nan

        df_dim_netting_sales = df_dim_netting_sales.reindex(columns=NS.LIST_COLUMN)
        nparr_dim_netting_sales = df_dim_netting_sales.to_numpy()
        dict_parent = {row[NS.SALESID_IDX]: row[NS.PARENTSALESID_IDX] for row in nparr_dim_netting_sales}
        for row in nparr_dim_netting_sales:
            level = int(row[NS.LEVELID_IDX])
            if level == 1: continue
            start_pos = NS.ACCOUNTID_IDX + 5 - level
            row[start_pos] = row[NS.SALESID_IDX]
            for pos in range(start_pos, NS.GCID_IDX):
                row[pos + 1] = dict_parent[row[pos]]
        
        df_dim_netting_sales = pd.DataFrame(data=nparr_dim_netting_sales, columns=NS.LIST_COLUMN)
        self.df_dim_netting_sales = df_dim_netting_sales

    def _extend_sales_result(self) -> None:
        df_dim_netting_sales: pd.DataFrame = self.df_dim_netting_sales
        df_sales_result: pd.DataFrame = self.df_sales_result

        df_sales_result = pd.merge(
            df_sales_result, df_dim_netting_sales[[NS.SALESID, NS.AP2ID]],
            left_on=[SR.SALESID], right_on=[NS.SALESID], how='left'
        ).rename(columns={
            NS.AP2ID: SR.AP2ID
        })
        self.df_sales_result = df_sales_result[SR.LIST_COLUMN]

    def _process_pre_demand(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        pre demand를 가공해 Netted Demand, ap1 short reason, ap1 delivery plan을 만든다.
        '''
        # 3개의 결과에 공통으로 적용되는 처리 진행
        df_common = pd.merge(
            self.df_pre_demand[[
                PD_.ITEM, PD_.SALESID, PD_.SITEID, PD_.WEEKBUCKETIDX,
                PD_.AP2SHORTQTY, PD_.AP2SHORTREASON, PD_.CUSTOMERRANK,
                PD_.DEMANDPRIORITY, PD_.DEMANDTYPERANK, PD_.DEMANDTYPE,
                PD_.GCSHORTQTY, PD_.GCSHORTREASON, PD_.GLOBALPRIORITY,
                PD_.LOCALPRIORITY, PD_.MEASURETYPERANK, PD_.ORDERTYPE,
                PD_.PREFERENCERANK, PD_.PRODUCTRANK, PD_.QTYPROMISED,
                PD_.WEEKRANK,
            ]],
            self.df_dim_netting_sales[[NS.SALESID, NS.LEVELID, NS.AP1ID, NS.AP2ID, NS.GCID]],
            how='left', left_on=[PD_.SALESID], right_on=[NS.SALESID]
        )
        df_common[PD_.ITEM] = df_common[PD_.ITEM].str[:-3]
        df_common['PLANID'] = self.plan_id
        df_common['SUNDAY'] = \
            self.start_date + pd.to_timedelta(df_common[PD_.WEEKBUCKETIDX].astype(int) * 7 - 1, unit='D')
        df_common['SOPROMISEID'] = np.where(
            df_common[PD_.DEMANDTYPE].str[-3:] == 'ORD',
            df_common[PD_.ORDERTYPE] + '_DM',
            'FCST_DM'
        )
        df_common['SALESORDERID'] = df_common[PD_.DEMANDTYPE].str.cat(
            [
                df_common[PD_.MEASURETYPERANK], df_common[PD_.WEEKBUCKETIDX],
                df_common[NS.GCID], df_common[NS.AP2ID], df_common[PD_.SALESID],
                df_common[PD_.ITEM], df_common[PD_.SITEID], df_common['SOPROMISEID']
            ], sep='::'
        )

        df_ap1_short_reaon, df_ap1_delivery_plan = self._make_short_reason_delivery_plan(df_common)
        df_demand = self._make_demand(df_common)
        
        return df_ap1_short_reaon, df_ap1_delivery_plan, df_demand

    def _make_short_reason_delivery_plan(self, df_common) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        ap1 short reason과 ap1 delivery plan을 만든다.
        '''

        # ap1 short reason과 ap1 delivery plan 공통 처리
        df_ap2 = df_common[df_common[PD_.AP2SHORTREASON].notna()]
        df_gc = df_common[df_common[PD_.GCSHORTREASON].notna()]
        df_rb_filtered = pd.concat([df_ap2, df_gc], ignore_index=True)

        mask_ap2 = [True] * len(df_ap2) + [False] * len(df_gc)

        df_rb_filtered['SALESORDERID'] = np.where(
            mask_ap2,
            df_rb_filtered['SALESORDERID'] + '_1@PRE',
            df_rb_filtered['SALESORDERID'] + '_2@PRE',
        )

        # ap2, gc로 나눠진 값들을 하나의 컬럼에 모음
        df_rb_filtered[PD_.AP2SHORTQTY] = np.where(
            mask_ap2,
            df_rb_filtered[PD_.AP2SHORTQTY],
            df_rb_filtered[PD_.GCSHORTQTY],
        )
        df_rb_filtered[PD_.AP2SHORTREASON] = np.where(
            mask_ap2,
            df_rb_filtered[PD_.AP2SHORTREASON],
            df_rb_filtered[PD_.GCSHORTREASON],
        )

        # ap1 short reason
        df_ap1_short_reason = df_rb_filtered.copy()
        df_ap1_short_reason.rename(columns={
            # common -> short reason
            'PLANID': AP1SR_.PLANID,
            'SUNDAY': AP1SR_.DUEDATE,
            'SALESORDERID': AP1SR_.SALESORDERID,

            # pre demand -> short reason
            PD_.ITEM: AP1SR_.ITEM,
            PD_.SITEID: AP1SR_.SITEID,
            PD_.AP2SHORTQTY: AP1SR_.SHORTQTY,
            PD_.AP2SHORTREASON: AP1SR_.ORGPROBLEMTYPE,
        }, inplace=True)
        df_ap1_short_reason[AP1SR_.REQITEMID]= df_ap1_short_reason[AP1SR_.ITEM]
        df_ap1_short_reason[AP1SR_.REQSITEID] = df_ap1_short_reason[AP1SR_.SITEID]
        df_ap1_short_reason[AP1SR_.VERSIONNAME] = self.plan_version
        df_ap1_short_reason[AP1SR_.SHORTREASONID] = make_sequential_id('SHR_', 10, df_ap1_short_reason)
        df_ap1_short_reason[AP1SR_.GBM] = self.division_name
        df_ap1_short_reason[AP1SR_.PROBLEMID] = '1'
        df_ap1_short_reason[AP1SR_.PATHID] = 1
        df_ap1_short_reason[[AP1SR_.OPERATION, AP1SR_.BOMNAME, AP1SR_.BODNAME]] = ''
        df_ap1_short_reason[AP1SR_.ITEMREQUEST] = df_ap1_short_reason[AP1SR_.SALESORDERID]
        df_ap1_short_reason[AP1SR_.PROBLEMTYPE] = df_ap1_short_reason[AP1SR_.ORGPROBLEMTYPE]
        df_ap1_short_reason[AP1SR_.NEEDEDQTY] = df_ap1_short_reason[AP1SR_.SHORTQTY]
        df_ap1_short_reason[AP1SR_.PROBLEMDTTM] = df_ap1_short_reason[AP1SR_.DUEDATE]
        df_ap1_short_reason[AP1SR_.RESOURCENAME] = np.nan

        # ap1 delivery plan
        df_ap1_delivery_plan = df_rb_filtered
        df_ap1_delivery_plan.rename(columns={
            # common -> delivery plan
            'PLANID': AP1DP_.PLANID,
            'SUNDAY': AP1DP_.PROMISEDSHIPPINGDATE,
            'SALESORDERID': AP1DP_.SALESORDERID,

            # pre demand -> delivery plan
            PD_.ITEM: AP1DP_.ITEM,
            PD_.SITEID: AP1DP_.SITEID,
            PD_.DEMANDPRIORITY: AP1DP_.PRIORITY,
            PD_.AP2SHORTQTY: AP1DP_.QTYOPEN,
        }, inplace=True)
        df_ap1_delivery_plan = df_ap1_delivery_plan.sort_values(by=[AP1DP_.SALESORDERID, AP1DP_.PRIORITY], ignore_index=True)
        df_ap1_delivery_plan[AP1DP_.SOLINENUM] = df_ap1_delivery_plan.groupby(by=[AP1DP_.SALESORDERID]).cumcount() + 1
        df_ap1_delivery_plan[AP1DP_.PLANNEDSHIPMENTDATE] = df_ap1_delivery_plan[AP1DP_.PROMISEDSHIPPINGDATE]
        df_ap1_delivery_plan[AP1DP_.VERSIONNAME] = self.plan_version
        df_ap1_delivery_plan[AP1DP_.DELIVERYPLANID] = make_sequential_id('DLV_', 10, df_ap1_delivery_plan)
        df_ap1_delivery_plan[AP1DP_.GBM] = self.division_name
        df_ap1_delivery_plan[AP1DP_.QTYPLANNED] = 0
        df_ap1_delivery_plan[[AP1DP_.ES_DEMAND_CODE, AP1DP_.SALESNAME, AP1DP_.SALESLEVEL, AP1DP_.REQDELSTARTDATE]] = np.nan

        return df_ap1_short_reason[AP1SR_.LIST_FULL_COLUMN], df_ap1_delivery_plan[AP1DP_.LIST_COLUMN]

    def _make_demand(self, df_demand) -> pd.DataFrame:
        ''' 공통 처리 이후의 netted demand 가공을 진행한다. '''

        df_demand[ND_.SHIPTOID] = df_demand[PD_.SITEID]
        df_demand[NS.LEVELID] = np.where(df_demand[NS.LEVELID]=='3', 'AP2', df_demand[NS.LEVELID])
        df_demand[NS.LEVELID] = np.where(df_demand[NS.LEVELID]=='4', 'AP1', df_demand[NS.LEVELID])
        df_demand[NS.LEVELID] = np.where(df_demand[NS.LEVELID]=='5', 'Account', df_demand[NS.LEVELID])

        df_demand = pd.merge(
            df_demand, self.df_dim_item[[I.ITEM, I.PRODUCTGROUP, I.SECTION]],
            how='left', left_on=[PD_.ITEM], right_on=[I.ITEM]
        ).drop(columns=[I.ITEM])

        df_demand.rename(columns={
            # common -> Netted Demand
            'PLANID': ND_.PLANID,
            'SUNDAY': ND_.PROMISEDDELDATE,
            'SOPROMISEID': ND_.SOPROMISEID,
            'SALESORDERID': ND_.SALESORDERID,

            # Pre Demand -> Netted Demand
            PD_.SALESID: ND_.SALESID,
            PD_.ITEM: ND_.ITEMID,
            PD_.SITEID: ND_.SITEID,
            PD_.LOCALPRIORITY: ND_.LOCALPRIORITY,
            PD_.GLOBALPRIORITY: ND_.GLOBALPRIORITY,
            PD_.DEMANDPRIORITY: ND_.DEMANDPRIORITY,
            PD_.QTYPROMISED: ND_.QTYPROMISED,
            PD_.DEMANDTYPERANK: ND_.DEMANDTYPERANK,
            PD_.WEEKRANK: ND_.WEEKRANK,
            PD_.CUSTOMERRANK: ND_.CUSTOMERRANK,
            PD_.PRODUCTRANK: ND_.PRODUCTRANK,
            PD_.MEASURETYPERANK: ND_.MEASURETYPERANK,
            PD_.PREFERENCERANK: ND_.PREFERENCERANK,

            # Netting Sales -> Netted Demand
            NS.LEVELID: ND_.SALESLEVEL,
            NS.AP1ID: ND_.AP1ID,
            NS.AP2ID: ND_.AP2ID,
            NS.GCID: ND_.GCID,
            
            # Item -> Netted Demand
            I.PRODUCTGROUP: ND_.PRODUCTGROUP,
            I.SECTION: ND_.SECTION,
        }, inplace=True)

        df_demand[[
            ND_.TIEBREAK, ND_.TIMEUOM, ND_.BUSINESSTYPE, ND_.ROUTING_PRIORITY,
            ND_.NO_SPLIT, ND_.MAP_SATISFY_SS, ND_.PREALLOC_ATTRIBUTE, ND_.BUILDAHEADTIME,
            ND_.REASONCODE, ND_.IS_PLAN_DATE, ND_.MFGWEEK, ND_.SUPPLYWEEK1,
            ND_.SUPPLYWEEK2, ND_.FROZENRANK, ND_.ADJ_WEEKRANK, ND_.ADJ_PRIORITY,
            ND_.MATCH_CODE, ND_.OPTION_CODE, ND_.MP_PRIORITY, ND_.MOD_PRIORITY,
            ND_.NO_PLAN, ND_.SHORT_CODE, ND_.CHANNELRANK, ND_.LOCALID,
        ]] = np.nan
        df_demand[[
            ND_.REQDELENDDATE, ND_.REQDELSTARTDATE, ND_.ORDER_ENTRY_DATE,
        ]] = pd.NaT

        df_demand = df_demand.sort_values(by=[ND_.SALESORDERID, ND_.DEMANDPRIORITY], ignore_index=True)
        df_demand[ND_.SOLINENUM] = df_demand.groupby(by=[ND_.SALESORDERID]).cumcount() + 1
        
        div = self.division_id
        if div == 'D001': # mx 특화 로직
            df_demand[ND_.LOCALPRIORITY] = df_demand[ND_.DEMANDPRIORITY]
        elif div == 'D002': # vd 특화 로직
            df_demand[ND_.TIEBREAK] = df_demand[ND_.WEEKRANK] + df_demand[ND_.DEMANDTYPERANK]
        elif div == 'D003': # da 특화 로직
            pass

        return df_demand[ND_.LIST_COLUMN]
    
    def _convert_empty_type(self):
        '''
        data가 없을 경우,
        o9에서 오는 dataframe은 설정된 타입과 무관하게 object 타입으로 온다.
        이는 이후의 연산에 영향을 미칠 수 있어서 비어있는 dataframe은
        type을 바꿔준다.
        '''
        if self.df_dim_location.empty:
            self.df_dim_location = self.df_dim_location.astype({
                L.ISVALID: bool,
            })
        if self.df_plan.empty:
            self.df_plan = self.df_plan.astype({
                P_.PLANHORIZON: 'float64',
                P_.STARTDATE: 'datetime64[ns]',
                P_.ENDDATE: 'datetime64[ns]',
                P_.CLOSEDATE: 'datetime64[ns]',
            })
        if self.df_priority_rank.empty:
            self.df_priority_rank = self.df_priority_rank.astype({
                PR_.DIGIT: 'float64',
                PR_.ORDER: 'float64',
            })
        if self.df_pre_demand.empty:
            self.df_pre_demand = self.df_pre_demand.astype({
                PD_.QTYPROMISED: 'float64',
                PD_.ACCSHORTQTY: 'float64',
                PD_.AP1SHORTQTY: 'float64',
                PD_.AP2SHORTQTY: 'float64',
                PD_.GCSHORTQTY: 'float64',
                PD_.SHORTQTY: 'float64',
            })
        if self.df_dp_guide.empty:
            self.df_dp_guide = self.df_dp_guide.astype({
                DPG.QTY: 'float64',
            })
        if self.df_short_reason_dp_guide.empty:
            self.df_short_reason_dp_guide = self.df_short_reason_dp_guide.astype({
                SRDPG.SHORTQTY: 'float64',
            })
        if self.df_rb_master.empty:
            self.df_rb_master = self.df_rb_master.astype({
                RBM.FLAG: bool,
            })
        if self.df_inventory.empty:
            dict_inventory_type = {
                Inv_.AVAILQTY: 'float64',
                Inv_.BOHADDQTY: 'float64',
                Inv_.W0BOHADDQTY: 'float64',
            }
            for column in self.df_inventory.columns:
                if 'CDC' in str(column).upper():
                    dict_inventory_type[column] = 'float64'
            self.df_inventory = self.df_inventory.astype(dict_inventory_type)
        if self.df_intransit.empty:
            self.df_intransit = self.df_intransit.astype({
                Int_.INTRANSITQTY: 'float64',
            })
        if self.df_sales_bom_map.empty:
            self.df_sales_bom_map = self.df_sales_bom_map.astype({
                SBOMM.PRIORITY: 'float64',
            })
        if self.df_model_eop.empty:
            self.df_model_eop = self.df_model_eop.astype({
                MEOP.EOP_INIT_DATE: 'datetime64[ns]',
                MEOP.EOP_COM_DATE: 'datetime64[ns]',
                MEOP.EOP_CHG_DATE: 'datetime64[ns]',
            })
        if self.df_inventory_sell.empty:
            dict_inventory_sell_type = {
                InvS_.AVAILQTY: 'float64',
                InvS_.BOHADDQTY: 'float64',
                InvS_.W0BOHADDQTY: 'float64',
            }
            for column in self.df_inventory_sell.columns:
                if 'CDC' in str(column).upper():
                    dict_inventory_sell_type[column] = 'float64'
            self.df_inventory_sell = self.df_inventory_sell.astype(dict_inventory_sell_type)
        if self.df_intransit_sell.empty:
            self.df_intransit_sell = self.df_intransit_sell.astype({
                IntS_.INTRANSITQTY: 'float64',
            })
        if self.df_es_item_site.empty:
            self.df_es_item_site = self.df_es_item_site.astype({
                ESIS.TWOS: 'float64',
                ESIS.STOCKTRANSITTIME: 'float64',
            })
        if self.df_available_resource.empty:
            self.df_available_resource = self.df_available_resource.astype({
                AR_.QTY: 'float64',
            })
        if self.df_code_map.empty:
            self.df_code_map = self.df_code_map.astype({
                CM.NUM1: 'float64',
                CM.NUM2: 'float64',
                CM.NUM3: 'float64',
                CM.DATE1: 'datetime64[ns]',
                CM.DATE2: 'datetime64[ns]',
                CM.DATE3: 'datetime64[ns]',
            })
        if self.df_delivery_plan.empty:
            self.df_delivery_plan = self.df_delivery_plan.astype({
                DP_.QTYOPEN: 'float64',
            })
        if self.df_short_reason.empty:
            self.df_short_reason = self.df_short_reason.astype({
                SR_.PROBLEMID: 'float64',
                SR_.SHORTQTY: 'float64',
            })
        if self.df_distribution_orders.empty:
            self.df_distribution_orders = self.df_distribution_orders.astype({
                DO_.QTY: 'float64',
            })
        if self.df_distribution_orders_sell.empty:
            self.df_distribution_orders_sell = self.df_distribution_orders_sell.astype({
                DOS_.QTY: 'float64',
            })
        if self.df_sales_result.empty:
            self.df_sales_result = self.df_sales_result.astype({
                SR.QTY_RTF: 'float64',
                SR.QTY_GI: 'float64',
            })
        if self.df_channel_stock_master.empty:
            self.df_channel_stock_master = self.df_channel_stock_master.astype({
                CSM.EFFSTARTDATE: 'datetime64[ns]',
                CSM.EFFENDDATE: 'datetime64[ns]',
                CSM.TOBE: 'float64',
            })
        if self.df_channel_stock_ap2_master.empty:
            self.df_channel_stock_ap2_master = self.df_channel_stock_ap2_master.astype({
                CSAP2M.ASIS: 'float64',
            })
        if self.df_estore_replenishment.empty:
            self.df_estore_replenishment = self.df_estore_replenishment.astype({
                ESR.PLANNEDENDDATE: 'datetime64[ns]',
                ESR.REPLENISHMENTQTY: 'float64',
            })
        if self.df_prebuild.empty:
            self.df_prebuild = self.df_prebuild.astype({
                P.BUILDAHEADTIME: 'float64',
            })
        if self.df_sales_order_lp.empty:
            self.df_sales_order_lp = self.df_sales_order_lp.astype({
                SOLP.QTY: 'float64',
            })
