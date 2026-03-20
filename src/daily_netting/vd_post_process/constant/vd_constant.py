# import DIRNetting as DIRN    # MST_DIR_NETTING
# import BAT as BAT    # MTA_BUILDAHEADTIMESET
# import BATAlt as BATA    # MTA_BUILDAHEADTIMESET_ALT
# import BATItem as BATI    # MTA_BUILDAHEADTIMESET_ITEM
# import LocalSwapOut as LSwO    # MST_LOCALSWAP_OUT
# import VDPreferenceRank as VDPR    # MST_PREFERENCERANK
# import TotalBODLT as TBODLT    # TOTAL BOD L/T (VD Tenant)
# import ProfitRank as ProfR    # Profit Rank (VD Tenant)
# import CreditAlloc as CA    # MTA_CREDIT_ALLOC
# import SalesOrder as SO    # MST_SALESORDER
# import SalesOrderLine as SOL    # ARC_SALESORDERLINE_VD
# import TWOSSales as TWOSS    # MTA_TARGETWOS_SALES

class DIRNetting:
    ''' Constants of Netting IF DIR Netting '''

    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 0
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 1
    SHIPMENTTYPE = 'Location.[Shipment Category]'                         ;    SHIPMENTTYPE_IDX = 2
    TIMEFENCE = 'Netting DIR Netting Time Fence'                          ;    TIMEFENCE_IDX = 3
    ISVALID = 'Netting DIR Netting Valid Flag'                            ;    ISVALID_IDX = 4

    LIST_COLUMN = [
        SECTION, SALESID, SHIPMENTTYPE, TIMEFENCE, ISVALID,
    ]

class BAT:
    ''' Constants of Netting IF BAT '''

    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1
    TYPE = 'Netting BAT Type'                                             ;    TYPE_IDX = 2
    BUILDAHEADTIME = 'Netting BAT Value'                                  ;    BUILDAHEADTIME_IDX = 3

    LIST_COLUMN = [
        SECTION, SITEID, TYPE, BUILDAHEADTIME,
    ]

class BATAlt:
    ''' Constants of Netting IF BAT Alt '''

    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1
    FROMWEEK = 'Time.[Week]'                                              ;    FROMWEEK_IDX = 2
    TOWEEK = 'To Time.[To Week]'                                          ;    TOWEEK_IDX = 3
    TYPE = 'Netting BAT Alt Type'                                         ;    TYPE_IDX = 4
    BUILDAHEADTIME = 'Netting BAT Alt Value'                              ;    BUILDAHEADTIME_IDX = 5

    LIST_COLUMN = [
        SECTION, SITEID, FROMWEEK, TOWEEK, TYPE, 
        BUILDAHEADTIME,
    ]

class BATItem:
    ''' Constants of Netting IF BAT Item '''

    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 0
    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 1
    FROMWEEK = 'Time.[Week]'                                              ;    FROMWEEK_IDX = 2
    TOWEEK = 'To Time.[To Week]'                                          ;    TOWEEK_IDX = 3
    TYPE = 'Netting BAT Item Type'                                        ;    TYPE_IDX = 4
    BUILDAHEADTIME = 'Netting BAT Item Value'                             ;    BUILDAHEADTIME_IDX = 5

    LIST_COLUMN = [
        SITEID, ITEM, FROMWEEK, TOWEEK, TYPE, 
        BUILDAHEADTIME,
    ]

class LocalSwapOut:
    ''' Constants of Netting IF Local Swap Out '''

    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 0
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 1
    FROMWEEK = 'Netting Local Swap Out From Week'                         ;    FROMWEEK_IDX = 2
    TOWEEK = 'Netting Local Swap Out To Week'                             ;    TOWEEK_IDX = 3
    ISVALID = 'Netting Local Swap Out Valid Flag'                         ;    ISVALID_IDX = 4

    LIST_COLUMN = [
        SECTION, SALESID, FROMWEEK, TOWEEK, ISVALID,
    ]

class VDPreferenceRank:
    ''' Constants of Netting IF VD Preference Rank '''

    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 0
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 1
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 2
    PREFERENCERANK = 'Netting VD Preference Rank Value'                   ;    PREFERENCERANK_IDX = 3
    TIMEFENCE = 'Netting VD Preference Rank Time Fence'                   ;    TIMEFENCE_IDX = 4

    LIST_COLUMN = [
        SECTION, SALESID, SITEID, PREFERENCERANK, TIMEFENCE,
    ]

class TotalBODLT:
    ''' Constants of Netting IF Total BOD LT '''

    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1
    WEEK = 'Time.[Week]'                                                  ;    WEEK_IDX = 2
    BOD_LT = 'Netting Total BOD LT Value'                                 ;    BOD_LT_IDX = 3

    LIST_COLUMN = [
        ITEM, SITEID, WEEK, BOD_LT,
    ]

class ProfitRank:
    ''' Constants of Netting IF Profit Rank '''

    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 0
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 1
    WEEK = 'Time.[Week]'                                                  ;    WEEK_IDX = 2
    PROFITRANK = 'Netting Profit Rank Value'                              ;    PROFITRANK_IDX = 3

    LIST_COLUMN = [
        ITEM, SALESID, WEEK, PROFITRANK,
    ]

class CreditAlloc:
    ''' Constants of Netting IF Credit Alloc '''

    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 0
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 1
    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 2
    ISVALID = 'Netting Credit Alloc Valid Flag'                           ;    ISVALID_IDX = 3

    LIST_COLUMN = [
        SITEID, SALESID, SECTION, ISVALID,
    ]

class SalesOrder:
    ''' Constants of Netting IF Sales Order '''

    SO_ID = 'Documents.[OrderlineID]'                                     ;    SO_ID_IDX = 0
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 1
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 2
    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 3
    DMD_TYP = 'Netting Sales Order Type'                                  ;    DMD_TYP_IDX = 4
    DELIVERYQTY = 'Netting Sales Order Delivery Qty'                      ;    DELIVERYQTY_IDX = 5
    CONFIRMQTY = 'Netting Sales Order Confirm Qty'                        ;    CONFIRMQTY_IDX = 6
    REQQTY = 'Netting Sales Order Req Qty'                                ;    REQQTY_IDX = 7
    ATPCHECKTYPE = 'Netting Sales Order ATP Check Type'                   ;    ATPCHECKTYPE_IDX = 8
    DFREGION = 'Netting Sales Order DF Region'                            ;    DFREGION_IDX = 9
    MAD = 'Netting Sales Order MAD'                                       ;    MAD_IDX = 10
    COMPANY = 'Netting Sales Order Company'                               ;    COMPANY_IDX = 11
    RMAD = 'Netting Sales Order RMAD'                                     ;    RMAD_IDX = 12
    ATPDATE = 'Netting Sales Order ATP Date'                              ;    ATPDATE_IDX = 13

    LIST_COLUMN = [
        SO_ID, SALESID, SITEID, ITEM, DMD_TYP, 
        DELIVERYQTY, CONFIRMQTY, REQQTY, ATPCHECKTYPE, DFREGION, 
        MAD, COMPANY, RMAD, ATPDATE,
    ]

class SalesOrderLine:
    ''' Constants of Netting IF Sales Order Line '''

    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 0
    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 1
    WEEK = 'Time.[Week]'                                                  ;    WEEK_IDX = 2
    QTYOPEN = 'Netting Sales Order Line Qty'                              ;    QTYOPEN_IDX = 3

    LIST_COLUMN = [
        SALESID, ITEM, WEEK, QTYOPEN,
    ]

class TWOSSales:
    ''' Constants of Netting IF TWOS Sales '''

    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 0
    PRODUCTGROUP = 'Item.[Product Group]'                                 ;    PRODUCTGROUP_IDX = 1
    ISCHSTOCK = 'Netting TWOS Sales CH Stock Flag'                        ;    ISCHSTOCK_IDX = 2

    LIST_COLUMN = [
        SALESID, PRODUCTGROUP, ISCHSTOCK,
    ]
