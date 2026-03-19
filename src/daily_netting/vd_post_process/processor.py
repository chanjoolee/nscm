import pandas as pd

from daily_netting.vd_post_process.constant.ods_constant import NettedDemandD as ND_
from daily_netting.vd_post_process.constant.ods_constant import NettedDemandLogD as NDL_
from daily_netting.vd_post_process.constant.ods_constant import NDEMAND_ADD as NSO
from daily_netting.vd_post_process.utils.common import is_log_flag_on, make_sequential_id
from daily_netting.vd_post_process.process.lp_order_netting import do_lp_order_netting, set_lp_order_netting_env
from daily_netting.vd_post_process.process.ncp_split_reg import do_ncp_split_reg, set_ncp_split_reg_env

class Processor:
    def __init__(self, accessor: object, logger):
        self.accessor = accessor
        self.logger = logger
        self.list_demand_log = []

    # 후처리 process들
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
            
            
    def process_ncp_split_reg(self) -> None:
        accessor = self.accessor
        
        set_ncp_split_reg_env(accessor)
        
        df_in_EXP_SOPROMISESRCNCP = accessor.df_demand
        df_in_MST_LOGISTICSSITE = accessor.df_Netting_IF_Logistic_Site # 생성 필요
        df_in_BUF_SALESPLANINVENTORY = accessor.df_Netting_IF_Sales_Plan_Inventory # 생성 필요
        
        df_processed_demand = do_ncp_split_reg(df_in_EXP_SOPROMISESRCNCP
                , df_in_MST_LOGISTICSSITE
                , df_in_BUF_SALESPLANINVENTORY
        )
        
        accessor.df_demand = df_processed_demand

        if is_log_flag_on(accessor, 'NCP Split Reg'):
            self.list_demand_log.append(self.make_demand_log('NCP Split Reg'))


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


