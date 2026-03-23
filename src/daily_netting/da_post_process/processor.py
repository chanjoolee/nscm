import pandas as pd

from daily_netting.da_post_process.constant.ods_constant import NettedDemandD as ND_
from daily_netting.da_post_process.constant.ods_constant import NettedDemandLogD as NDL_
from daily_netting.da_post_process.constant.ods_constant import NDEMAND_ADD as NSO
from daily_netting.da_post_process.process.preference_rank import do_preference_rank, set_preference_rank_env
from daily_netting.da_post_process.process.match_code_local import do_match_code_local, set_match_code_local_env
from daily_netting.da_post_process.process.lp_order_netting import do_lp_order_netting, set_lp_order_netting_env
from daily_netting.da_post_process.utils.common import is_log_flag_on, make_sequential_id

class Processor:
    def __init__(self, accessor: object, logger):
        self.accessor = accessor
        self.logger = logger
        self.list_demand_log = []

    # 후처리 process들
    def process_match_code_local(self) -> None:
        accessor = self.accessor
    
        set_match_code_local_env(accessor)

        df_demand = accessor.df_demand
        df_da_item_seller_map = accessor.df_da_item_seller_map
        df_dim_item = accessor.df_dim_item
        df_region_asn = accessor.df_region_asn
        df_local_sales_order = accessor.df_local_sales_order
        df_dim_netting_sales = accessor.df_dim_netting_sales
        df_local_measure = accessor.df_local_measure

        df_processed_demand = do_match_code_local(
            df_demand, df_da_item_seller_map, df_dim_item, df_region_asn,
            df_local_sales_order, df_dim_netting_sales, df_local_measure,
            self.logger,
        )

        accessor.df_demand = df_processed_demand

        if is_log_flag_on(accessor, 'Match Code Local'):
            self.list_demand_log.append(self.make_demand_log('Match Code Local'))
    
    def process_preference_rank(self) -> None:
        accessor = self.accessor
    
        set_preference_rank_env(accessor)

        df_exp_sopromise_src = accessor.df_demand
        df_dyn_gi = accessor.df_gi
        df_dyn_shipment = accessor.df_shipment
        df_mst_site = accessor.df_dim_location
        df_mst_acntinfo = accessor.df_account_info
        df_vui_itemattb = accessor.df_dim_item
        
        df_processed_demand = do_preference_rank(
            df_exp_sopromise_src, df_dyn_gi, df_dyn_shipment,
            df_mst_site, df_mst_acntinfo, df_vui_itemattb, self.logger,
        )

        accessor.df_demand = df_processed_demand

        if is_log_flag_on(accessor, 'Preference Rank'):
            self.list_demand_log.append(self.make_demand_log('Preference Rank'))


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


    # output 
    def make_final_demand(self) -> pd.DataFrame:
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
