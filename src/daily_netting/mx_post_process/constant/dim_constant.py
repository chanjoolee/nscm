# import Location as L    # Dim.Location
# import Item as I    # Dim.Item
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
    PRODUCTCODE = 'Item.[Product Code]'                                   ;    PRODUCTCODE_IDX = 2
    ATTB05 = 'Item.[A5]'                                                  ;    ATTB05_IDX = 3
    ATTB09 = 'Item.[A9]'                                                  ;    ATTB09_IDX = 4
    ATTB15 = 'Item.[A15]'                                                 ;    ATTB15_IDX = 5
    BASICNAME = 'Item.[Basic Name]'                                       ;    BASICNAME_IDX = 6
    CLASSCATEGORY = 'Item.[Class Category]'                               ;    CLASSCATEGORY_IDX = 7
    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 8

    LIST_COLUMN = [
        ITEM, PRODUCTGROUP, PRODUCTCODE, ATTB05, ATTB09, 
        ATTB15, BASICNAME, CLASSCATEGORY, SECTION,
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

class NettingLPPlanBatch:
    ''' Constants of Netting LP Plan Batch '''

    SALESID = 'Netting LP Plan Batch.[Sales Domain Lv3]'                  ;    SALESID_IDX = 0
    SCENARIO = 'Netting LP Plan Batch.[Scenario]'                         ;    SCENARIO_IDX = 1
    TENANT = 'Netting LP Plan Batch.[Tenant]'                             ;    TENANT_IDX = 2

    LIST_COLUMN = [
        SALESID, SCENARIO, TENANT,
    ]
