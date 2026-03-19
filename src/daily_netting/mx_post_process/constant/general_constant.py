# import DFAllocMap as DFAM    # VDF_ALLOCMAP
# import DFItemSite as DFIS    # BUF_DFITEMSITE
# import CustomerRank as CR    # VUI_CUSTOMERRANK
# import ProductRank as ProdR    # VUI_PRODUCTRANK
# import DemandTypeConv as DTC    # MTA_DEMANDTYPECONV
# import AccountInfo as AI    # MST_ACNTINFO@SCMTF
# import BOD as BOD    # MST_BODDETAIL
# import SSCalendarD as SSC_D    # MST_CALENDAR (MX)
# import Association as A    # MST_ASSOCIATION

class DFAllocMap:
    ''' Constants of Netting IF DF Alloc Map '''

    PRODUCTGROUP = 'Item.[Product Group]'                                 ;    PRODUCTGROUP_IDX = 0
    AP2 = 'Sales Domain.[Ship To]'                                        ;    AP2_IDX = 1
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 2
    DFALLOCATION = 'Netting DF Alloc Map Allocation Flag'                 ;    DFALLOCATION_IDX = 3

    LIST_COLUMN = [
        PRODUCTGROUP, AP2, SITEID, DFALLOCATION,
    ]

class DFItemSite:
    ''' Constants of Netting IF DF Item Site '''

    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1

    LIST_COLUMN = [
        ITEM, SITEID,
    ]

class CustomerRank:
    ''' Constants of Netting IF Customer Rank '''

    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 0
    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 1
    GRADE = 'Netting Customer Rank Grade'                                 ;    GRADE_IDX = 2
    GRADEMOD = 'Netting Customer Rank Grade MOD'                          ;    GRADEMOD_IDX = 3

    LIST_COLUMN = [
        SALESID, SECTION, GRADE, GRADEMOD,
    ]

class ProductRank:
    ''' Constants of Netting IF Product Rank '''

    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 0
    GRADE = 'Netting Product Rank Grade'                                  ;    GRADE_IDX = 1
    GRADEMOD = 'Netting Product Rank Grade MOD'                           ;    GRADEMOD_IDX = 2

    LIST_COLUMN = [
        ITEM, GRADE, GRADEMOD,
    ]

class DemandTypeConv:
    ''' Constants of Netting IF Demand Type Conv '''

    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 0
    ITEMID = 'Item.[Item]'                                                ;    ITEMID_IDX = 1
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 2
    MEASURE = 'Netting Measure Type.[Measure Type]'                       ;    MEASURE_IDX = 3
    SOURCEMEASURE = 'Netting Demand Type.[Demand Type]'                   ;    SOURCEMEASURE_IDX = 4
    TARGETMEASURE = 'Netting Demand Type Conv Target'                     ;    TARGETMEASURE_IDX = 5
    ISVLAID = 'Netting Demand Type Conv Valid Flag'                       ;    ISVLAID_IDX = 6

    LIST_COLUMN = [
        SALESID, ITEMID, SITEID, MEASURE, SOURCEMEASURE, 
        TARGETMEASURE, ISVLAID,
    ]

class AccountInfo:
    ''' Constants of Netting IF Account Info '''

    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 0
    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 1
    CHANNELTYPE = 'Netting Account Info Channel Type'                     ;    CHANNELTYPE_IDX = 2
    GPGNAME = 'Netting Account Info GPG Name'                             ;    GPGNAME_IDX = 3

    LIST_COLUMN = [
        SALESID, SECTION, CHANNELTYPE, GPGNAME,
    ]

class BOD:
    ''' Constants of Netting IF BOD '''

    BODNAME = 'Netting BOD.[BOD Name]'                                    ;    BODNAME_IDX = 0
    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 1
    EFFSTARTDATE = 'Netting BOD Eff Start Date'                           ;    EFFSTARTDATE_IDX = 2
    EFFENDDATE = 'Netting BOD Eff End Date'                               ;    EFFENDDATE_IDX = 3
    TRANSITTIME = 'Netting BOD Transit Time'                              ;    TRANSITTIME_IDX = 4
    SHIPPINGLEADTIME = 'Netting BOD Shipping Lead Time'                   ;    SHIPPINGLEADTIME_IDX = 5
    PRIORITY = 'Netting BOD Priority'                                     ;    PRIORITY_IDX = 6
    PREBULD_PRIORITY = 'Netting BOD Prebuild Priority'                    ;    PREBULD_PRIORITY_IDX = 7

    LIST_COLUMN = [
        BODNAME, ITEM, EFFSTARTDATE, EFFENDDATE, TRANSITTIME, 
        SHIPPINGLEADTIME, PRIORITY, PREBULD_PRIORITY,
    ]

class SSCalendarD:
    ''' Constants of Netting IF SS Calendar W '''

    CALENDARNAME = 'Netting SS Calendar.[Calendar Name]'                  ;    CALENDARNAME_IDX = 0
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 1
    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 2
    PATTERNSEQ = 'Time.[Week]'                                            ;    PATTERNSEQ_IDX = 3
    VALUE = 'Netting SS Calendar Value D'                                 ;    VALUE_IDX = 4
    MUST = 'Netting SS Calendar Must D'                                   ;    MUST_IDX = 5

    LIST_COLUMN = [
        CALENDARNAME, SITEID, ITEM, PATTERNSEQ, VALUE, 
        MUST,
    ]

class Association:
    ''' Constants of Netting IF Association '''

    SECTION = 'Item.[Section]'                                            ;    SECTION_IDX = 0
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 1
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 2
    BASEVALIDITY = 'Netting Association Base Validity'                    ;    BASEVALIDITY_IDX = 3

    LIST_COLUMN = [
        SECTION, SALESID, SITEID, BASEVALIDITY,
    ]

class SalesOrderLP :
    ''' constants of Netting IF Sales Order LP '''

    WORK_SCENARIO_NAME = 'Version.[Version Name]'
    DIVISIONID = 'Netting Division.[Division ID]'
    ITEM = 'Item.[Item]'
    SALESID = 'Sales Domain.[Ship To]'
    SITEID = 'Location.[Location]'
    REQDELENDDATE = 'Time.[Day]'
    QTY = 'Netting Sales Order LP Qty'
    ORDERTYPE = 'Netting Sales Order LP Order Type'
    SONO = 'Netting Sales Order LP SO NO'
    SOITEM = 'Netting Sales Order LP SO Item'
    ORDERREASON = 'Netting Sales Order LP Order Reason'
    SOLDTOPARTY = 'Netting Sales Order LP Sold To Party'
    SHIPTOPARTY = 'Netting Sales Order LP Ship To Party'
    RDD = 'Netting Sales Order LP RDD'
    SOCREATIONDATE = 'Netting Sales Order LP SO Creation Date'
    ORDERTYPEDESC = 'Netting Sales Order LP Order Type Desc'

    LIST_COLUMN = [
        WORK_SCENARIO_NAME, DIVISIONID, ITEM, SALESID, SITEID,
        REQDELENDDATE, QTY, ORDERTYPE, SONO, SOITEM,
        ORDERREASON, SOLDTOPARTY, SHIPTOPARTY, RDD, SOCREATIONDATE,
        ORDERTYPEDESC,
    ]
