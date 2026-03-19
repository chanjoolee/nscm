import pandas as pd
import numpy as np

from netting_util.conv_gscm_o9_cols import o92gscm, gscm2o9

from daily_netting.mx_post_process.constant.ods_constant import NettedDemandD as ND_D
from daily_netting.mx_post_process.constant.ods_constant import NDEMAND_ADD as NSO
from daily_netting.mx_post_process.constant.dim_constant import Item as VIA


STR_COMPUTER = 'COMPUTER'
STR_L101 = 'L101'
STR_UNF = 'UNF'


def set_release_netting_env(accessor: object) -> None:
    pass


def do_release_netting(
        df_in_EXP_SOPROMISENCP,
        df_in_VUI_ITEMATTB,
):
    df_in_EXP_SOPROMISENCP = o92gscm(df_in_EXP_SOPROMISENCP, ND_D)
    df_in_EXP_SOPROMISENCP = o92gscm(df_in_EXP_SOPROMISENCP, NSO)
    df_in_VUI_ITEMATTB = o92gscm(df_in_VUI_ITEMATTB, VIA)

    df_exp_sopromisencp = df_in_EXP_SOPROMISENCP.copy()
    df_vui_itemattb = df_in_VUI_ITEMATTB.copy()

    # SQL:
    # UPDATE EXP_SOPROMISESRCNCP A
    #    SET QTYPROMISED = 0
    #  WHERE SALESORDERID LIKE 'UNF%'
    #    AND (ITEM, SITEID) NOT IN (
    #          SELECT ITEM, 'L101'
    #            FROM VUI_ITEMATTB
    #           WHERE SECTION = 'COMPUTER'
    #        )
    #    AND QTYPROMISED > 0
    #
    # 해석:
    # - SECTION='COMPUTER' 인 Item 들을 구한다.
    # - 단, (ITEM in computer_items) AND (SITEID='L101') 인 경우는 UPDATE 제외.
    # - 그 외 UNF% / QTYPROMISED>0 대상은 QTYPROMISED=0 처리.

    sr_computer_item = (
        df_vui_itemattb.loc[
            df_vui_itemattb['SECTION'].eq(STR_COMPUTER) & df_vui_itemattb['ITEM'].notna(),
            'ITEM'
        ]
        .drop_duplicates()
    )
    set_computer_item = set(sr_computer_item.tolist())

    sr_salesorderid = df_exp_sopromisencp['SALESORDERID'].astype('string')
    sr_siteid = df_exp_sopromisencp['SITEID'].astype('string')
    sr_qtypromised = pd.to_numeric(df_exp_sopromisencp['QTYPROMISED'], errors='coerce')

    mask_unf = sr_salesorderid.str.startswith(STR_UNF, na=False)
    mask_qty_positive = sr_qtypromised.gt(0)
    mask_excluded_l101_computer = df_exp_sopromisencp['ITEMID'].isin(set_computer_item) & sr_siteid.eq(STR_L101)

    mask_update = mask_unf & mask_qty_positive & (~mask_excluded_l101_computer)

    df_exp_sopromisencp.loc[mask_update, 'QTYPROMISED'] = 0
    

    df_exp_sopromisencp = gscm2o9(df_exp_sopromisencp, ND_D)
    df_exp_sopromisencp = gscm2o9(df_exp_sopromisencp, NSO)

    list_column = ND_D.LIST_COLUMN + NSO.LIST_COLUMN
    for col in list_column:
        if col not in df_exp_sopromisencp.columns:
            df_exp_sopromisencp[col] = np.nan

    return df_exp_sopromisencp[list_column]
