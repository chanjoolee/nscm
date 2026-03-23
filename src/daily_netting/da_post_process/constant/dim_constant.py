# import Item as I    # Dim.Item
# import Location as L    # Dim.Location
# import SalesDomain as SD    # Dim.Sales Domain
# import Time as T    # Dim.Time
# import NettingSales as NS    # Dim.Netting Sales

class Location:
    ''' Constants of Location '''

    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 0
    TYPE = 'Location.[Location Type]'                                     ;    TYPE_IDX = 1
    SHIPMENTTYPE = 'Location.[Shipment Category]'                         ;    SHIPMENTTYPE_IDX = 2
    ISVALID = 'Location.[Location Valid Flag]'                            ;    ISVALID_IDX = 3

    LIST_COLUMN = [
        SITEID, TYPE, SHIPMENTTYPE, ISVALID,
    ]

class Item:
    ''' Constants of Item '''

    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 0
    PRODUCTGROUP = 'Item.[Product Group]'                                 ;    PRODUCTGROUP_IDX = 1
    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 2
    SCMTYPE = 'Item.[SCM Type]'                                           ;    SCMTYPE_IDX = 3

    LIST_COLUMN = [
        ITEM, PRODUCTGROUP, SECTION, SCMTYPE
    ]

class NettingSales:
    ''' Constants of Netting Sales '''

    SALESID = 'Netting Sales.[Sales ID]'                                  ;    SALESID_IDX = 0
    PARENTSALESID = 'Netting Sales.[Parent Sales ID]'                     ;    PARENTSALESID_IDX = 1
    LEVELID = 'Netting Sales.[Sales Level]'                               ;    LEVELID_IDX = 2
    ACCOUNTID = 'ACCOUNTID'                                               ;    ACCOUNTID_IDX = 3
    AP1ID = 'AP1ID'                                                       ;    AP1ID_IDX = 4
    AP2ID = 'AP2ID'                                                       ;    AP2ID_IDX = 5
    GCID = 'GCID'                                                         ;    GCID_IDX = 6

    LIST_BASE_COLUMN = [
        SALESID, PARENTSALESID, LEVELID,
    ]

    LIST_COLUMN = [
        SALESID, PARENTSALESID, LEVELID, ACCOUNTID, AP1ID, 
        AP2ID, GCID,
    ]


class SalesDomain:
    ''' Constants of Sales Domain '''

    SHIPTO = 'Sales Domain.[Ship To]'                                     ;    SHIPTO_IDX = 0
    GC = 'Sales Domain.[GC]'                                              ;    GC_IDX = 1
    AP2 = 'Sales Domain.[AP2]'                                            ;    AP2_IDX = 2
    AP1 = 'Sales Domain.[AP1]'                                            ;    AP1_IDX = 3
    ACCOUNT = 'Sales Domain.[Account]'                                    ;    ACCOUNT_IDX = 4
    ISVALID = 'Sales Domain.[Ship To Valid Flag]'                         ;    ISVALID_IDX = 5

    LIST_COLUMN = [
        SHIPTO, GC, AP2, AP1, ACCOUNT, 
        ISVALID,
    ]

class Time:
    ''' Constants of Time '''

    WEEK = 'Time.[Week]'                                                  ;    WEEK_IDX = 0
    WEEKSTARTDAY = 'Time.[Week Start Day]'                                ;    WEEKSTARTDAY_IDX = 1

    LIST_COLUMN = [
        WEEK, WEEKSTARTDAY,
    ]

class NettingLPPlanBatch:
    ''' Constants of Netting LP Plan Batch '''

    SALESID = 'Netting LP Plan Batch.[Sales Domain Lv3]'                  ;    SALESID_IDX = 0
    SCENARIO = 'Netting LP Plan Batch.[Scenario]'                         ;    SCENARIO_IDX = 1
    TENANT = 'Netting LP Plan Batch.[Tenant]'                             ;    TENANT_IDX = 2

    LIST_COLUMN = [
        SALESID, SCENARIO, TENANT,
    ]

