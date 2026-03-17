from typing import Tuple

import pandas as pd

from daily_netting.mx_post_process.constant.ods_constant import NettedDemandD as ND_
from daily_netting.mx_post_process.constant.ods_constant import NettedDemandRBD as NDRB_
from daily_netting.mx_post_process.constant.ods_constant import NettedDemandLogD as NDL_
from daily_netting.mx_post_process.constant.ods_constant import AP1ShortReasonD as AP1SR_
from daily_netting.mx_post_process.constant.ods_constant import NDEMAND_ADD as NSO
from daily_netting.mx_post_process.constant.mx_constant import RBMaster as RBM
from daily_netting.mx_post_process.utils.common import is_log_flag_on, make_sequential_id
from daily_netting.mx_post_process.process.eop_netting import do_eop_netting, set_eop_env
from daily_netting.mx_post_process.process.new_fcst_netting import do_new_fcst_netting, set_new_fcst_env
from daily_netting.mx_post_process.process.new_fcst_sell_netting import do_new_fcst_sell_netting, set_new_fcst_sell_env
from daily_netting.mx_post_process.process.gi_netting import do_gi_netting, set_gi_env
from daily_netting.mx_post_process.process.gi_sell_netting import do_gi_sell_netting, set_gi_sell_env
from daily_netting.mx_post_process.process.channel_stock import do_channel_stock, set_channel_stock_env
from daily_netting.mx_post_process.process.estore_netting import do_estore_netting, set_estore_netting_env
from daily_netting.mx_post_process.process.lp_order_netting import do_lp_order_netting, set_lp_order_netting_env

class Processor:
    def __init__(self, accessor: object, logger):
        self.accessor = accessor
        self.logger = logger
        self.list_demand_log = []

    # 후처리 process들
    def process_eop_netting(self) -> None:
        accessor = self.accessor
    
        set_eop_env(accessor)

        df_demand = accessor.df_demand
        df_custom_model_map = accessor.df_custom_model_map
        df_intransit = accessor.df_intransit
        df_inventory = accessor.df_inventory
        df_intransit_sell = accessor.df_intransit_sell
        df_inventory_sell = accessor.df_inventory_sell
        df_dim_location = accessor.df_dim_location
        df_model_eop = accessor.df_model_eop
        df_sales_bom_map = accessor.df_sales_bom_map
        df_mx_item_seller_map = accessor.df_mx_item_seller_map

        df_processed_demand, df_eop_demand = do_eop_netting(
            df_demand, df_custom_model_map, df_intransit, df_inventory,
            df_intransit_sell, df_inventory_sell, df_dim_location, df_model_eop,
            df_sales_bom_map, df_mx_item_seller_map, self.logger,
        )

        accessor.df_eop_demand = df_eop_demand
        accessor.df_demand = df_processed_demand

        if is_log_flag_on(accessor, 'EOP Netting'):
            self.list_demand_log.append(self.make_demand_log('EOP Netting'))
    
    def process_new_fcst_netting(self) -> None:
        accessor = self.accessor
    
        set_new_fcst_env(accessor)
        
        #인바운드
        df_exp_sopromisesrc = accessor.df_demand
        df_mta_codemap           =  accessor.df_code_map
        df_exp_deliveryplan      = accessor.df_delivery_plan
        df_exp_shortreason      = accessor.df_short_reason
        df_mst_inventory         = accessor.df_inventory
        df_mst_intransit         = accessor.df_intransit
        df_mst_distributionorders   = accessor.df_distribution_orders
        df_v_mta_sellermap      = accessor.df_mx_item_seller_map
        df_mta_custommodelmap   = accessor.df_custom_model_map
        df_mves_itemsite        = accessor.df_es_item_site
        df_mst_inventory_fne    = accessor.df_available_resource
        df_vui_itemattb     = accessor.df_dim_item
        df_mta_newfcstnetting    = accessor.df_new_fcst_period

        df_processed_demand = do_new_fcst_netting(
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
            , self.logger
        )

        accessor.df_demand = df_processed_demand

        #################################################
        # new fcst netting_SELL start
        set_new_fcst_sell_env(accessor)
        df_mst_inventory_sell         = accessor.df_inventory_sell
        df_mst_intransit_sell         = accessor.df_intransit_sell
        df_mst_distributionorders_sell   = accessor.df_distribution_orders_sell
        df_exp_sopromisesrc = accessor.df_demand
        df_mta_codemap           =  accessor.df_code_map
        df_exp_deliveryplan      = accessor.df_delivery_plan
        df_exp_shortreason      = accessor.df_short_reason
        df_v_mta_sellermap      = accessor.df_mx_item_seller_map
        df_mta_custommodelmap   = accessor.df_custom_model_map
        df_mves_itemsite        = accessor.df_es_item_site
        df_mst_inventory_fne    = accessor.df_available_resource
        df_vui_itemattb     = accessor.df_dim_item
        df_mta_newfcstnetting    = accessor.df_new_fcst_period

        df_processed_demand = do_new_fcst_sell_netting(
            df_v_mta_sellermap
            , df_exp_sopromisesrc
            , df_exp_deliveryplan
            , df_exp_shortreason
            , df_mst_inventory_sell
            , df_mst_intransit_sell
            , df_mst_inventory_fne
            , df_mst_distributionorders_sell
            , df_vui_itemattb
            , df_mta_codemap
            , df_mta_newfcstnetting
            , df_mta_custommodelmap
            , df_mves_itemsite
            , self.logger
        )
        accessor.df_demand = df_processed_demand
        # new fcst netting_SELL end
        ################################################
        

        if is_log_flag_on(accessor, 'New FCST Netting'):
            self.list_demand_log.append(self.make_demand_log('New FCST Netting'))
    
    def process_gi_netting(self) -> None:
        accessor = self.accessor
    
        set_gi_env(accessor)
        set_gi_sell_env(accessor)

        df_demand = accessor.df_demand
        df_in_seller_map = accessor.df_mx_item_seller_map
        df_in_available_resource = accessor.df_available_resource
        df_in_es_item_site = accessor.df_es_item_site
        df_in_custom_model_map = accessor.df_custom_model_map 
        df_sales_result = accessor.df_sales_result 
        df_mst_site = accessor.df_dim_location

        df_gi_net_demand, df_MST_GINETTING, df_MST_GINETING_SALES = do_gi_netting(
            df_demand, df_in_seller_map, df_in_available_resource, df_in_es_item_site,
            df_in_custom_model_map, df_sales_result, df_mst_site, self.logger,
        )
        
        df_processed_demand = do_gi_sell_netting(
            df_gi_net_demand, df_in_seller_map, df_in_available_resource, df_in_es_item_site,
            df_in_custom_model_map, df_sales_result, df_mst_site, df_MST_GINETING_SALES, self.logger,
        )

        accessor.df_demand = df_processed_demand

        if is_log_flag_on(accessor, 'GI Netting'):
            self.list_demand_log.append(self.make_demand_log('GI Netting'))

    def process_channel_stock(self) -> None:
        accessor = self.accessor
    
        set_channel_stock_env(accessor)

        df_demand = accessor.df_demand
        df_dp_guide = accessor.df_dp_guide
        df_dim_netting_sales = accessor.df_dim_netting_sales
        df_dim_item = accessor.df_dim_item
        df_mx_item_seller_map = accessor.df_mx_item_seller_map
        df_es_item_site = accessor.df_es_item_site
        df_custom_model_map = accessor.df_custom_model_map
        df_channel_stock_master = accessor.df_channel_stock_master
        df_channel_stock_ap2_master = accessor.df_channel_stock_ap2_master
        df_code_map = accessor.df_code_map
        df_available_resource = accessor.df_available_resource
        df_ap1_short_reason = accessor.df_ap1_short_reason[AP1SR_.LIST_COLUMN]
        df_short_reason_dp_guide = accessor.df_short_reason_dp_guide
        
        df_processed_demand = do_channel_stock(
            df_demand
            , df_dp_guide
            , df_dim_netting_sales
            , df_dim_item
            , df_mx_item_seller_map
            , df_es_item_site
            , df_custom_model_map
            , df_channel_stock_master
            , df_channel_stock_ap2_master
            , df_code_map
            , df_available_resource
            , df_ap1_short_reason
            , df_short_reason_dp_guide
            , self.logger
        )

        accessor.df_demand = df_processed_demand

        if is_log_flag_on(accessor, 'Channel Stock'):
            self.list_demand_log.append(self.make_demand_log('Channel Stock'))
    
    def process_estore_netting(self) -> None:
        accessor = self.accessor
    
        set_estore_netting_env(accessor)

        df_demand = accessor.df_demand
        df_estore_twos = accessor.df_estore_twos
        df_dim_item = accessor.df_dim_item
        df_code_map = accessor.df_code_map
        df_es_item_site = accessor.df_es_item_site
        df_estore_replenishment = accessor.df_estore_replenishment

        df_processed_demand, df_estore_sales_order = do_estore_netting(
            df_demand, df_estore_twos, df_dim_item, df_code_map,
            df_es_item_site, df_estore_replenishment, self.logger,
        )

        accessor.df_demand = df_processed_demand
        accessor.df_estore_sales_order = df_estore_sales_order

        if is_log_flag_on(accessor, 'eStore Netting'):
            self.list_demand_log.append(self.make_demand_log('eStore Netting'))


    def process_lp_order_netting(self) -> None:
        accessor = self.accessor

        set_lp_order_netting_env(accessor)

        df_in_EXP_SOPROMISENCP = accessor.df_demand
        df_in_SALESORDER_LP = accessor.df_sales_order_lp
        df_in_VUI_ITEMATTB = accessor.df_dim_item
        df_in_GUI_SALESHIERARCHY = accessor.df_dim_netting_sales
        df_in_Netting_LP_Plan_Batch = accessor.df_dim_netting_lp_plan_batch

        df_processed_demand = do_lp_order_netting(
                df_in_EXP_SOPROMISENCP
                , df_in_SALESORDER_LP
                , df_in_VUI_ITEMATTB
                , df_in_GUI_SALESHIERARCHY
                , df_in_Netting_LP_Plan_Batch
        )
        
        accessor.df_demand = df_processed_demand

        if is_log_flag_on(accessor, 'Lp_Order_Netting'):
            self.list_demand_log.append(self.make_demand_log('Lp_Order_Netting'))

    def process_release_netting(self) -> None:
        accessor = self.accessor

        set_lp_order_netting_env(accessor)

        df_in_EXP_SOPROMISENCP = accessor.df_demand
        df_in_SALESORDER_LP = accessor.df_sales_order_lp
        df_in_VUI_ITEMATTB = accessor.df_dim_item
        df_in_GUI_SALESHIERARCHY = accessor.df_dim_netting_sales
        df_in_Netting_LP_Plan_Batch = accessor.df_dim_netting_lp_plan_batch

        df_processed_demand = do_lp_order_netting(
                df_in_EXP_SOPROMISENCP
                , df_in_SALESORDER_LP
                , df_in_VUI_ITEMATTB
                , df_in_GUI_SALESHIERARCHY
                , df_in_Netting_LP_Plan_Batch
        )
        
        accessor.df_demand = df_processed_demand

        if is_log_flag_on(accessor, 'Lp_Order_Netting'):
            self.list_demand_log.append(self.make_demand_log('Lp_Order_Netting'))

            
    # output 
    def make_final_demand(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''최종 demand를 '''
        accessor = self.accessor
        df_demand = accessor.df_demand

        df_demand[ND_.VERSIONNAME] = accessor.plan_version
        df_demand[ND_.GBM] = accessor.division_name
        df_demand[ND_.DEMANDID] = make_sequential_id('DMD_', 10, df_demand)

        LIST_FULL_COLUMN = ND_.LIST_FULL_COLUMN + NSO.LIST_COLUMN

        df_demand = df_demand.reindex(columns=LIST_FULL_COLUMN).rename(columns={
            ND_.SALESID: ND_.NEWSALESID,
            ND_.ITEMID: ND_.NEWITEMID,
            ND_.SITEID: ND_.NEWSITEID,
        })
        del accessor.df_demand

        return df_demand

    def make_demand_log(self, plan_step) -> pd.DataFrame:
        accessor = self.accessor
        df_demand_log = accessor.df_demand.copy()

        df_demand_log[NDL_.VERSIONNAME] = accessor.plan_version
        df_demand_log[NDL_.GBM] = accessor.division_name
        df_demand_log[NDL_.DEMANDID] = make_sequential_id('DMD_', 10, df_demand_log)
        df_demand_log[NDL_.PLANSTEP] = plan_step

        list_full_column = [NDL_.VERSIONNAME, NDL_.DEMANDID, NDL_.PLANSTEP] + ND_.LIST_COLUMN
        list_full_column.insert(NDL_.LIST_COLUMN.index(NDL_.GBM), NDL_.GBM)
        df_demand_log = df_demand_log.reindex(columns=list_full_column)
        df_demand_log.columns = NDL_.LIST_COLUMN
        
        return df_demand_log

    def concat_demand_log(self) -> pd.DataFrame:
        '''
        여태까지 모인 로그들을 하나의 output으로 만드는 함수
        '''
        if self.list_demand_log:
            return pd.concat(self.list_demand_log, ignore_index=True)
        else:
            return pd.DataFrame(columns=NDL_.LIST_COLUMN)


