# import DADemandManagementExsales as DADME    # MST_DMDMGMT_EXSALES
# import ModelRTS as MRTS    # MST_MODELRTS
# import ModelEOP as MEOP    # MST_MODELEOP
# import DAPreferenceRank as DAPR    # MTA_PRODUCTGROUP
# import DAItemSellerMap as DAISM    # V_MTA_SELLERMAP
# import DASellerMap as DASM    # MTA_SELLERMAP
# import RegionASN as RASN    # MST_SDMREGIONASN
# import LocalMeasure as LM    # EXP_LOCALMEASURE (@SCMTF)
# import LocalSalesOrder as LSaO    # BUF_SALESORDERLINE_LOCAL
# import DemandFluctuation as DF    # MST_DEMAND_FLUCTUATION
# import GI as GI    # DYN_GI
# import Shipment as S    # DYN_SHIPMENT
# import MeasureData as MD    # EXP_MEASUREDATA
# import DAPSIPlan as DAPSIP    # VUI_SALESPSI_PLAN
# import DAPSIResult as DAPSIR    # VUI_SALESPSI_RESULT
# import DAPrebuildMaster as DAPM    # MV_PREBUILDMASTER
# import DACHStockExsales as DACHSE    # MST_CHSTOCK_EXSALES
# import SIGResultGIW as SIGRGI_W    # EXP_DPGUIDE
# import SIGPlanMaster as SIGPM    # MST_DPGUIDEDEF@CHSCM
# import SIGMaster as SIGM    # MST_DPGUIDEPARAM@CHSCM


class DADemandManagementExsales:
    ''' Constants of Netting IF DA Dmdmgnt Exsales '''

    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 0
    PRODUCTGROUP = 'Item.[Product Group]'                                 ;    PRODUCTGROUP_IDX = 1
    ISVALID = 'Netting DA Dmdmgnt Exsales Is Valid'                       ;    ISVALID_IDX = 2

    LIST_COLUMN = [
        SALESID, PRODUCTGROUP, ISVALID,
    ]

class ModelRTS:
    ''' Constants of Netting IF Model RTS '''

    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 0
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 1
    RTS_INIT_DATE = 'Netting Model RTS Init Date'                         ;    RTS_INIT_DATE_IDX = 2
    RTS_COM_DATE = 'Netting Model RTS Com Date'                           ;    RTS_COM_DATE_IDX = 3
    FIRST_SALESDATE = 'Netting Model RTS First Sales Date'                ;    FIRST_SALESDATE_IDX = 4

    LIST_COLUMN = [
        ITEM, SALESID, RTS_INIT_DATE, RTS_COM_DATE, FIRST_SALESDATE, 
    ]

class ModelEOP:
    ''' Constants of Netting IF Model EOP '''

    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1
    STATUS = 'Netting Model EOP Status'                                   ;    STATUS_IDX = 2
    EOP_INIT_DATE = 'Netting Model EOP Init Date'                         ;    EOP_INIT_DATE_IDX = 3
    EOP_COM_DATE = 'Netting Model EOP Com Date'                           ;    EOP_COM_DATE_IDX = 4
    EOP_CHG_DATE = 'Netting Model EOP Chg Date'                           ;    EOP_CHG_DATE_IDX = 5

    LIST_COLUMN = [
        ITEM, SITEID, STATUS, EOP_INIT_DATE, EOP_COM_DATE, 
        EOP_CHG_DATE,
    ]

class DAPreferenceRank:
    ''' Constants of Netting IF DA Preference Rank '''

    PRODUCTGROUP = 'Item.[Product Group]'                                 ;    PRODUCTGROUP_IDX = 0
    PREFERENCERANK = 'Netting DA Preference Rank Value'                   ;    PREFERENCERANK_IDX = 1

    LIST_COLUMN = [
        PRODUCTGROUP, PREFERENCERANK,
    ]

class DAItemSellerMap:
    ''' Constants of Netting IF DA Item Seller Map '''

    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 0
    SALESID = 'Netting Account Group.[Account Group]'                     ;    SALESID_IDX = 1
    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 2
    TYPE = 'Netting DA Item Seller Map Type'                              ;    TYPE_IDX = 3
    WEEK = 'Netting DA Item Seller Map Week'                              ;    WEEK_IDX = 4

    LIST_COLUMN = [
        SITEID, SALESID, ITEM, TYPE, WEEK,
    ]

class DASellerMap:
    ''' Constants of Netting IF DA Seller Map '''

    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1
    SALESID = 'Netting Account Group.[Account Group]'                     ;    SALESID_IDX = 2
    TYPE = 'Netting DA Seller Map Type'                                   ;    TYPE_IDX = 3
    ISVALID = 'Netting DA Seller Map Valid Flag'                          ;    ISVALID_IDX = 4

    LIST_COLUMN = [
        SECTION, SITEID, SALESID, TYPE, ISVALID,
    ]

class RegionASN:
    ''' Constants of Netting IF Region ASN '''

    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1
    REGIONID = 'Netting Account Group.[Account Group]'                    ;    REGIONID_IDX = 2
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 3
    ISVALID = 'Netting Region ASN Valid Flag'                             ;    ISVALID_IDX = 4

    LIST_COLUMN = [
        SECTION, SITEID, REGIONID, SALESID, ISVALID,
    ]

class LocalMeasure:
    ''' Constants of Netting IF Local Measure '''

    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 0
    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 1
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 2
    LOCALID = 'Netting Account Group.[Account Group]'                     ;    LOCALID_IDX = 3
    ISOWEEK = 'Time.[Week]'                                               ;    ISOWEEK_IDX = 4
    MEASUREQTY = 'Netting Local Measure Qty'                              ;    MEASUREQTY_IDX = 5

    LIST_COLUMN = [
        SALESID, ITEM, SITEID, LOCALID, ISOWEEK, 
        MEASUREQTY,
    ]

class LocalSalesOrder:
    ''' Constants of Netting IF Local Sales Order '''

    TYPE = 'Netting Order Type.[Order Type]'                              ;    TYPE_IDX = 0
    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 1
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 2
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 3
    LOCALID = 'Netting Account Group.[Account Group]'                     ;    LOCALID_IDX = 4
    WEEK = 'Time.[Week]'                                                  ;    WEEK_IDX = 5
    REQDATE = 'To Time.[To Day]'                                          ;    REQDATE_IDX = 6
    QTY = 'Netting Local Sales Order Qty'                                 ;    QTY_IDX = 7

    LIST_COLUMN = [
        TYPE, ITEM, SITEID, SALESID, LOCALID, 
        WEEK, REQDATE, QTY,
    ]

class DemandFluctuation:
    ''' Constants of Netting IF Demand Fluctuation '''

    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 0
    PRODUCTGROUP = 'Item.[Product Group]'                                 ;    PRODUCTGROUP_IDX = 1
    SCMTYPE = 'Netting Demand Fluctuation SCM Type'                       ;    SCMTYPE_IDX = 2
    PLAN_MEASURE = 'Netting Demand Fluctuation Plan Measure'              ;    PLAN_MEASURE_IDX = 3
    FROZEN_PERIOD = 'Netting Demand Fluctuation Frozen Period'            ;    FROZEN_PERIOD_IDX = 4
    SHORT_ISVALID = 'Netting Demand Fluctuation Short Valid Flag'         ;    SHORT_ISVALID_IDX = 5
    SHORT_TIMEFENCE = 'Netting Demand Fluctuation Short Fence'            ;    SHORT_TIMEFENCE_IDX = 6
    SHORT_RATE = 'Netting Demand Fluctuation Short Rate'                  ;    SHORT_RATE_IDX = 7
    MID_ISVALID = 'Netting Demand Fluctuation Mid Valid Flag'             ;    MID_ISVALID_IDX = 8
    MID_TIMEFENCE = 'Netting Demand Fluctuation Mid Fence'                ;    MID_TIMEFENCE_IDX = 9
    MID_RATE = 'Netting Demand Fluctuation Mid Rate'                      ;    MID_RATE_IDX = 10
    LONG_ISVALID = 'Netting Demand Fluctuation Long Valid Flag'           ;    LONG_ISVALID_IDX = 11
    LONG_TIMEFENCE = 'Netting Demand Fluctuation Long Fence'              ;    LONG_TIMEFENCE_IDX = 12
    LONG_RATE = 'Netting Demand Fluctuation Long Rate'                    ;    LONG_RATE_IDX = 13

    LIST_COLUMN = [
        SALESID, PRODUCTGROUP, SCMTYPE, PLAN_MEASURE, FROZEN_PERIOD, 
        SHORT_ISVALID, SHORT_TIMEFENCE, SHORT_RATE, MID_ISVALID, MID_TIMEFENCE, 
        MID_RATE, LONG_ISVALID, LONG_TIMEFENCE, LONG_RATE,
    ]

class GI:
    ''' Constants of Netting IF GI '''

    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1
    GIQTY = 'Netting GI Qty'                                              ;    GIQTY_IDX = 2

    LIST_COLUMN = [
        ITEM, SITEID, GIQTY,
    ]

class Shipment:
    ''' Constants of Netting IF Shipment '''

    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1
    BILLINGQTY = 'Netting Shipment Qty'                                   ;    BILLINGQTY_IDX = 2

    LIST_COLUMN = [
        ITEM, SITEID, BILLINGQTY,
    ]

class MeasureData:
    ''' Constants of Netting IF Measure Data '''

    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 0
    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 1
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 2
    ISOWEEK = 'Time.[Week]'                                               ;    ISOWEEK_IDX = 3
    MEASUREQTY = 'Netting Measure Qty'                                    ;    MEASUREQTY_IDX = 4

    LIST_COLUMN = [
        SALESID, ITEM, SITEID, ISOWEEK, MEASUREQTY,
    ]

class DAPSIPlan:
    ''' Constants of Netting IF DA PSI Plan '''

    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1
    WEEK = 'Time.[Week]'                                                  ;    WEEK_IDX = 2
    DEMANDQTY = 'Netting DA PSI Plan Demand Qty'                          ;    DEMANDQTY_IDX = 3
    AGINGQTY = 'Netting DA PSI Plan Aging Qty'                            ;    AGINGQTY_IDX = 4

    LIST_COLUMN = [
        ITEM, SITEID, WEEK, DEMANDQTY, AGINGQTY,
    ]

class DAPSIResult:
    ''' Constants of Netting IF DA PSI Result '''

    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1
    EOHQTY = 'Netting DA PSI Result EOH Qty'                              ;    EOHQTY_IDX = 2

    LIST_COLUMN = [
        ITEM, SITEID, EOHQTY,
    ]

class DAPrebuildMaster:
    ''' Constants of Netting IF DA Prebuild Master '''

    TOSITEID = 'Location.[Location]'                                      ;    TOSITEID_IDX = 0
    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 1
    WEEK = 'Netting ID.[Sequence]'                                        ;    WEEK_IDX = 2
    CATEGORY = 'Netting Item Level.[Item]'                                ;    CATEGORY_IDX = 3
    BUILDAHEAD = 'Netting DA Prebuild BAT'                                ;    BUILDAHEAD_IDX = 4
    PRIORITY = 'Netting DA Prebuild Priority'                             ;    PRIORITY_IDX = 5
    EFFSTARTDATE = 'Netting DA Prebuild Effective Start Date'             ;    EFFSTARTDATE_IDX = 6
    EFFENDDATE = 'Netting DA Prebuild Effective End Date'                 ;    EFFENDDATE_IDX = 7
    TRANSITTIME = 'Netting DA Prebuild Transit Time'                      ;    TRANSITTIME_IDX = 8
    SHIPPINGLEADTIME = 'Netting DA Prebuild SLT'                          ;    SHIPPINGLEADTIME_IDX = 9
    TYPE = 'Netting DA Prebuild Type'                                     ;    TYPE_IDX = 10
    BOMEFFSTARTDATE = 'Netting DA Prebuild BOM Effective Start Date'      ;    BOMEFFSTARTDATE_IDX = 11
    BOMEFFENDDATE = 'Netting DA Prebuild BOM Effective End Date'          ;    BOMEFFENDDATE_IDX = 12
    ERPTYPE = 'Netting DA Prebuild ERP Type'                              ;    ERPTYPE_IDX = 13

    LIST_COLUMN = [
        TOSITEID, ITEM, WEEK, CATEGORY, BUILDAHEAD, 
        PRIORITY, EFFSTARTDATE, EFFENDDATE, TRANSITTIME, SHIPPINGLEADTIME, 
        TYPE, BOMEFFSTARTDATE, BOMEFFENDDATE, ERPTYPE,
    ]

class DACHStockExsales:
    ''' Constants of Netting IF DA CHStock Exsales '''

    PRODUCTGROUP = 'Item.[Product Group]'                                 ;    PRODUCTGROUP_IDX = 0
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 1
    ISVALID = 'Netting DA CHStock Exsales Valid Flag'                     ;    ISVALID_IDX = 2

    LIST_COLUMN = [
        PRODUCTGROUP, SALESID, ISVALID,
    ]

class SIGResultGIW:
    ''' Constants of SIG Result GI W '''

    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 0
    UDAITEM = 'Item.[Item]'                                               ;    UDAITEM_IDX = 1
    SWEEK = 'Time.[Week]'                                                 ;    SWEEK_IDX = 2
    QTY = 'SIG C Result GI W'                                             ;    QTY_IDX = 3

    LIST_COLUMN = [
        SALESID, UDAITEM, SWEEK, QTY,
    ]

class SIGPlanMaster:
    ''' Constants of SIG Plan Master '''

    PRODUCTGROUP = 'Item.[Product Group]'                                  ;    PRODUCTGROUP_IDX = 0
    SALESID = 'Sales Domain.[Sales Domain Lv3]'                            ;    SALESID_IDX = 1
    STATUS = 'SIG Status'                                                  ;    STATUS_IDX = 2
    CONSTWEEK = 'SIG Const Week'                                           ;    CONSTWEEK_IDX = 3
    FROZENPERIOD = 'SIG Frozen Period'                                     ;    FROZENPERIOD_IDX = 4
    PLANMEASURE = 'SIG Plan Measure'                                       ;    PLANMEASURE_IDX = 5

    LIST_COLUMN = [
        PRODUCTGROUP, SALESID, STATUS, CONSTWEEK, FROZENPERIOD,
        PLANMEASURE,
    ]

class SIGMaster:
    ''' Constants of SIG MASTER '''

    ITEMUDA = 'Item.[Item]'                                                ;    ITEMUDA_IDX = 0
    AP2ID = 'Sales Domain.[Ship To]'                                       ;    AP2ID_IDX = 1
    TYPE = 'SIG Category'                                                  ;    TYPE_IDX = 2
    ISVALID = 'SIG Param Is Valid'                                         ;    ISVALID_IDX = 3
    PLANNING = 'SIG Const Planning'                                        ;    PLANNING_IDX = 4
    FROZENPERIOD = 'SIG Param Frozen Period'                               ;    FROZENPERIOD_IDX = 5

    LIST_COLUMN = [
        ITEMUDA, AP2ID, TYPE, ISVALID, PLANNING,
        FROZENPERIOD,
    ]
