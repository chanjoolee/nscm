import gc
import traceback as tb

import pandas as pd

import netting_util.NSCMCommon as nscm
from daily_netting.mx_post_process.accessor import Accessor
from daily_netting.mx_post_process.processor import Processor

nscm.logging.basicConfig(level=nscm.logging.DEBUG)
logger = nscm.G_Logger('MX Post Process D')
logger.Start()

if nscm.G_IS_Local:
    # local 환경에서 data 가져오기
    import os
    from daily_netting.mx_post_process.constant.dim_constant import NettingSales as NS
    from daily_netting.mx_post_process.constant.dim_constant import Item as I
    from daily_netting.mx_post_process.constant.dim_constant import Location as L
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

    # from mx_post_process.constant.ods_constant import PlanD as P_W
    path_plan = '202609_DP_MX_AS_MO'
    # current_path = os.path.dirname(__file__)
    current_path = 'C:\\Netting\\o9Data\\PYNettingMXPostD\\Prod\\DPNS\\202609_DP_MX_AS_MO\\download_20260312-150111'
    # 데이터 가져오기
    # dim
    df_dim_item =                   pd.read_csv(f'{current_path}/df_dim_item.csv.csv', dtype=object)
    df_dim_location =               pd.read_csv(f'{current_path}/df_dim_location.csv', dtype=object)
    df_dim_netting_sales =          pd.read_csv(f'{current_path}/df_dim_netting_sales.csv', dtype=object)
    df_dim_netting_lp_plan_batch =  pd.read_csv(f'{current_path}/df_dim_netting_lp_plan_batch.csv', dtype=object)

    # ods
    df_plan =                       pd.read_csv(f'{current_path}/df_plan.csv', dtype=object)
    df_plan_option =                pd.read_csv(f'{current_path}/df_plan_option.csv', dtype=object)
    df_priority_rank =              pd.read_csv(f'{current_path}/df_priority_rank.csv', dtype=object)
    df_pre_demand =                 pd.read_csv(f'{current_path}/df_pre_demand.csv', dtype=object)

    # general
    df_sales_order_lp =             pd.read_csv(f'{current_path}/df_sales_order_lp.csv', dtype=object)

    # mx
    df_short_reason_dp_guide =      pd.read_csv(f'{current_path}/df_short_reason_dp_guide.csv', dtype=object)
    df_rb_master =                  pd.read_csv(f'{current_path}/df_rb_master.csv', dtype=object)
    df_dp_guide =                   pd.read_csv(f'{current_path}/df_dp_guide.csv', dtype=object)
    df_mx_item_seller_map =         pd.read_csv(f'{current_path}/df_mx_item_seller_map.csv', dtype=object)
    df_custom_model_map =           pd.read_csv(f'{current_path}/df_custom_model_map.csv', dtype=object)
    df_inventory =                  pd.read_csv(f'{current_path}/df_inventory.csv', dtype=object)
    df_intransit =                  pd.read_csv(f'{current_path}/df_intransit.csv', dtype=object)
    df_sales_bom_map =              pd.read_csv(f'{current_path}/df_sales_bom_map.csv', dtype=object)
    df_model_eop =                  pd.read_csv(f'{current_path}/df_model_eop.csv', dtype=object)
    df_inventory_sell =             pd.read_csv(f'{current_path}/df_inventory_sell.csv', dtype=object)
    df_intransit_sell =             pd.read_csv(f'{current_path}/df_intransit_sell.csv', dtype=object)
    df_es_item_site =               pd.read_csv(f'{current_path}/df_es_item_site.csv', dtype=object)
    df_available_resource =         pd.read_csv(f'{current_path}/df_available_resource.csv', dtype=object)
    df_code_map =                   pd.read_csv(f'{current_path}/df_code_map.csv', dtype=object)
    df_delivery_plan =              pd.read_csv(f'{current_path}/df_delivery_plan.csv', dtype=object)
    df_short_reason =               pd.read_csv(f'{current_path}/df_short_reason.csv', dtype=object)
    df_distribution_orders =        pd.read_csv(f'{current_path}/df_distribution_orders.csv', dtype=object)
    df_distribution_orders_sell =   pd.read_csv(f'{current_path}/df_distribution_orders_sell.csv', dtype=object)
    df_new_fcst_period =            pd.read_csv(f'{current_path}/df_new_fcst_period.csv', dtype=object)
    df_sales_result =               pd.read_csv(f'{current_path}/df_sales_result.csv', dtype=object)
    df_channel_stock_master =       pd.read_csv(f'{current_path}/df_channel_stock_master.csv', dtype=object)
    df_channel_stock_ap2_master =   pd.read_csv(f'{current_path}/df_channel_stock_ap2_master.csv', dtype=object)
    df_estore_twos =                pd.read_csv(f'{current_path}/df_estore_twos.csv', dtype=object)
    df_estore_replenishment =       pd.read_csv(f'{current_path}/df_estore_replenishment.csv', dtype=object)
    df_prebuild =                   pd.read_csv(f'{current_path}/df_prebuild.csv', dtype=object)

    print('csv load 완료')
    print('df type변환 시작')
    # dim
    df_dim_location = df_dim_location.astype({
        L.ISVALID: bool,
    })
    # ods
    df_plan = df_plan.astype({
        P_.PLANHORIZON: 'float64',
        P_.STARTDATE: 'datetime64[ns]',
        P_.ENDDATE: 'datetime64[ns]',
        P_.CLOSEDATE: 'datetime64[ns]',
    })
    df_plan_option = df_plan_option.astype({
    })
    df_priority_rank = df_priority_rank.astype({
        PR_.DIGIT: 'float64',
        PR_.ORDER: 'float64',
    })
    df_pre_demand = df_pre_demand.astype({
        PD_.QTYPROMISED: 'float64',
        PD_.ACCSHORTQTY: 'float64',
        PD_.AP1SHORTQTY: 'float64',
        PD_.AP2SHORTQTY: 'float64',
        PD_.GCSHORTQTY: 'float64',
        PD_.SHORTQTY: 'float64',
    })
    # general
    df_sales_order_lp = df_sales_order_lp.astype({
        SOLP.QTY: 'float64',
    })
    # mx
    df_short_reason_dp_guide = df_short_reason_dp_guide.astype({
        SRDPG.SHORTQTY: 'float64',
    })
    df_rb_master = df_rb_master.astype({
        # RBM.FLAG: bool,
    })
    df_dp_guide = df_dp_guide.astype({
        DPG.QTY: 'float64',
    })
    df_mx_item_seller_map = df_mx_item_seller_map.astype({
    })
    df_custom_model_map = df_custom_model_map.astype({
    })
    df_inventory = df_inventory.astype({
        Inv_.AVAILQTY: 'float64',
        Inv_.BOHADDQTY: 'float64',
        Inv_.W0BOHADDQTY: 'float64',
    })
    df_intransit = df_intransit.astype({
        Int_.INTRANSITQTY: 'float64',
    })
    df_sales_bom_map = df_sales_bom_map.astype({
        SBOMM.PRIORITY: 'float64',
    })
    df_model_eop = df_model_eop.astype({
        MEOP.EOP_INIT_DATE: 'datetime64[ns]',
        MEOP.EOP_COM_DATE: 'datetime64[ns]',
        MEOP.EOP_CHG_DATE: 'datetime64[ns]',
    })
    df_inventory_sell = df_inventory_sell.astype({
        InvS_.AVAILQTY: 'float64',
        InvS_.BOHADDQTY: 'float64',
        InvS_.W0BOHADDQTY: 'float64',
    })
    df_intransit_sell = df_intransit_sell.astype({
        IntS_.INTRANSITQTY: 'float64',
    })
    df_es_item_site = df_es_item_site.astype({
        ESIS.TWOS: 'float64',
        ESIS.STOCKTRANSITTIME: 'float64',
    })
    df_available_resource = df_available_resource.astype({
        AR_.QTY: 'float64',
    })
    df_code_map = df_code_map.astype({
        CM.NUM1: 'float64',
        CM.NUM2: 'float64',
        CM.NUM3: 'float64',
        CM.DATE1: 'datetime64[ns]',
        CM.DATE2: 'datetime64[ns]',
        CM.DATE3: 'datetime64[ns]',
    })
    df_delivery_plan = df_delivery_plan.astype({
        DP_.QTYOPEN: 'float64',
    })
    df_short_reason = df_short_reason.astype({
        SR_.PROBLEMID: 'float64',
        SR_.SHORTQTY: 'float64',
    })
    df_distribution_orders = df_distribution_orders.astype({
        DO_.QTY: 'float64',
    })
    df_distribution_orders_sell = df_distribution_orders_sell.astype({
        DOS_.QTY: 'float64',
    })
    df_new_fcst_period = df_new_fcst_period.astype({
    })
    df_sales_result = df_sales_result.astype({
        SR.QTY_RTF: 'float64',
        SR.QTY_GI: 'float64',
    })
    df_channel_stock_master = df_channel_stock_master.astype({
        CSM.EFFSTARTDATE: 'datetime64[ns]',
        CSM.EFFENDDATE: 'datetime64[ns]',
        CSM.TOBE: 'float64',
    })
    df_channel_stock_ap2_master = df_channel_stock_ap2_master.astype({
        CSAP2M.ASIS: 'float64',
    })
    df_estore_twos = df_estore_twos.astype({
    })
    df_estore_replenishment = df_estore_replenishment.astype({
        ESR.PLANNEDENDDATE: 'datetime64[ns]',
        ESR.REPLENISHMENTQTY: 'float64',
    })
    df_prebuild = df_prebuild.astype({
        P.BUILDAHEADTIME: 'float64',
    })
    print('df_type 변환 완료')

try:
    df_list = [
        ('df_dim_item', df_dim_item), ('df_dim_location', df_dim_location), ('df_dim_netting_sales', df_dim_netting_sales),
        ('df_plan', df_plan), ('df_plan_option', df_plan_option), ('df_priority_rank', df_priority_rank), ('df_pre_demand', df_pre_demand),
        ('df_dp_guide', df_dp_guide), ('df_short_reason_dp_guide', df_short_reason_dp_guide), ('df_rb_master', df_rb_master),
        ('df_mx_item_seller_map', df_mx_item_seller_map), ('df_custom_model_map', df_custom_model_map), ('df_inventory', df_inventory),
        ('df_intransit', df_intransit), ('df_sales_bom_map', df_sales_bom_map), ('df_model_eop', df_model_eop),
        ('df_inventory_sell', df_inventory_sell), ('df_intransit_sell', df_intransit_sell), ('df_es_item_site', df_es_item_site),
        ('df_available_resource', df_available_resource), ('df_code_map', df_code_map), ('df_delivery_plan', df_delivery_plan),
        ('df_short_reason', df_short_reason), ('df_distribution_orders', df_distribution_orders),
        ('df_distribution_orders_sell', df_distribution_orders_sell), ('df_new_fcst_period', df_new_fcst_period), ('df_sales_result', df_sales_result),
        ('df_channel_stock_master', df_channel_stock_master),
        ('df_channel_stock_ap2_master', df_channel_stock_ap2_master), ('df_estore_twos', df_estore_twos),
        ('df_estore_replenishment', df_estore_replenishment),
        ('df_prebuild', df_prebuild),
        ('df_dim_netting_lp_plan_batch', df_dim_netting_lp_plan_batch),
        ('df_sales_order_lp', df_sales_order_lp),
    ]
    for name, df in df_list:
        logger.Note(f'{name} has {len(df)} rows', 20)

    # accessor에 데이터 담기, plan data
    accessor = Accessor(
        df_dim_item, df_dim_location, df_dim_netting_sales, df_plan, df_plan_option,
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
    ) # 모든 df 데이터 가져가기

    # 전역 레퍼런스 삭제, gc 비우기
    del df_dim_item, df_dim_location, df_plan, df_plan_option,
    df_priority_rank, df_pre_demand,
    df_dp_guide, df_short_reason_dp_guide,
    df_rb_master, df_mx_item_seller_map,
    df_custom_model_map, df_inventory, df_intransit, df_sales_bom_map,
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

    gc.collect()

    processor = Processor(accessor, logger)

    # pd.set_option('display.max_columns', None)
    processor.process_eop_netting()
    processor.process_new_fcst_netting()
    processor.process_gi_netting()
    processor.process_channel_stock()
    processor.process_estore_netting()
    processor.process_lp_order_netting()
    processor.process_release_netting()

    df_out_ap1_delivery_plan = accessor.df_ap1_delivery_plan
    df_out_ap1_short_reason = accessor.df_ap1_short_reason
    df_out_eop_demand = accessor.df_eop_demand
    df_out_demand_log = processor.concat_demand_log()
    df_out_demand = processor.make_final_demand()
except Exception as e:
    logger.error(tb.format_exc())
    raise e

logger.Finish()
logger.Note('Meaningless log to avoid log missing problem', 30)

if nscm.G_IS_Local:
    # df_out_eop_demand.to_csv(f'Fact.EOP Demand.csv', index=False)
    # df_out_rb_demand.to_csv(f'Fact.RB.csv', index=False)
    # df_out_demand.to_csv(f'202538_M_estore_Netted_Demand.csv', index=False)
    # df_out_demand_log.to_csv(f'202534_M_Netted_Demand_M2_log.csv', index=False)
    pass
