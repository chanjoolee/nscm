# import PlanD as P_D
# import PlanOptionD as PO_D
# import PriorityRankD as PR_D
# import PreDemandD as PD_D
# import AP1DeliveryPlanD as AP1DP_D    # EXP_AP1_DELIVERYPLAN
# import AP1ShortReasonD as AP1SR_D    # EXP_AP1_SHORTREASON
# import ArcNettedDemandD as AND_D
# import NettedDemandD as ND_D    # EXP_SOPROMISE
# import NettedDemandLogD as NDL_D

class PlanD:
    ''' Constants of Netting Plan ODS D '''

    VERSIONNAME = 'Version.[Version Name]'                                ;    VERSIONNAME_IDX = 0
    DIVISIONID = 'Netting Division.[Division ID]'                         ;    DIVISIONID_IDX = 1
    DIVISIONNAME = 'Netting Division.[Division Name]'                     ;    DIVISIONNAME_IDX = 2
    PLANHORIZON = 'Netting Plan Horizon D'                                ;    PLANHORIZON_IDX = 3
    PLANID = 'Netting Plan ID D'                                          ;    PLANID_IDX = 4
    PLANTYPE = 'Netting Plan Type D'                                      ;    PLANTYPE_IDX = 5
    STARTDATE = 'Netting Effective Start Date D'                          ;    STARTDATE_IDX = 6
    ENDDATE = 'Netting Effective End Date D'                              ;    ENDDATE_IDX = 7
    PLANWEEK = 'Netting Plan Week D'                                      ;    PLANWEEK_IDX = 8
    CLOSEDATE = 'Netting Close Date D'                                    ;    CLOSEDATE_IDX = 9

    LIST_COLUMN = [
        VERSIONNAME, DIVISIONID, DIVISIONNAME, PLANHORIZON, PLANID, 
        PLANTYPE, STARTDATE, ENDDATE, PLANWEEK, CLOSEDATE,
    ]

class PlanOptionD:
    ''' Constants of Netting Plan Option ODS D '''

    PLANSTEP = 'Netting Plan Option.[Plan Step]'                          ;    PLANSTEP_IDX = 0
    LOGFLAG = 'Netting Plan Log Flag D'                                   ;    LOGFLAG_IDX = 1

    LIST_COLUMN = [
        PLANSTEP, LOGFLAG,
    ]

class PriorityRankD:
    ''' Constants of Netting Priority Rank ODS D '''

    RULEID = 'Netting Priority Rule.[Rule ID]'                            ;    RULEID_IDX = 0
    RULESEQUENCE = 'Netting Priority Rule Seq.[Rule Sequence]'            ;    RULESEQUENCE_IDX = 1
    RANKCATEGORY = 'Netting Priority Rank.[Rank Category]'                ;    RANKCATEGORY_IDX = 2
    DIGIT = 'Netting Priority Rank Digit D'                               ;    DIGIT_IDX = 3
    DEFAULTVALUE = 'Netting Priority Rank Default Value D'                ;    DEFAULTVALUE_IDX = 4
    ISVALID = 'Netting Priority Rank Valid Flag D'                        ;    ISVALID_IDX = 5
    ORDER = 'Netting Priority Rank Order D'                               ;    ORDER_IDX = 6

    LIST_COLUMN = [
        RULEID, RULESEQUENCE, RANKCATEGORY, DIGIT, DEFAULTVALUE, 
        ISVALID, ORDER,
    ]

class PreDemandD:
    ''' Constants of Netting Pre Demand ODS D '''

    ITEM = 'Netting Item.[Item]'                                          ;    ITEM_IDX = 0
    SALESID = 'Netting Sales.[Sales ID]'                                  ;    SALESID_IDX = 1
    SITEID = 'Netting Site.[Site ID]'                                     ;    SITEID_IDX = 2
    DEMANDID = 'Demand.[DemandID]'                                        ;    DEMANDID_IDX = 3
    WEEKBUCKETIDX = 'Netting Week Bucket.[Bucket Idx]'                    ;    WEEKBUCKETIDX_IDX = 4
    MONTHBUCKETIDX = 'Netting Month Bucket.[Bucket Idx]'                  ;    MONTHBUCKETIDX_IDX = 5
    AP1SHORTQTY = 'Netting Pre Demand AP1 Short Qty D'                    ;    AP1SHORTQTY_IDX = 6
    AP1SHORTREASON = 'Netting Pre Demand AP1 Shortreason D'               ;    AP1SHORTREASON_IDX = 7
    AP2SHORTQTY = 'Netting Pre Demand AP2 Short Qty D'                    ;    AP2SHORTQTY_IDX = 8
    AP2SHORTREASON = 'Netting Pre Demand AP2 Shortreason D'               ;    AP2SHORTREASON_IDX = 9
    ACCSHORTQTY = 'Netting Pre Demand Account Short Qty D'                ;    ACCSHORTQTY_IDX = 10
    ACCSHORTREASON = 'Netting Pre Demand Account Shortreason D'           ;    ACCSHORTREASON_IDX = 11
    CUSTOMERRANK = 'Netting Pre Demand Customer Rank D'                   ;    CUSTOMERRANK_IDX = 12
    DEMANDPRIORITY = 'Netting Pre Demand Demand Priority D'               ;    DEMANDPRIORITY_IDX = 13
    DEMANDTYPERANK = 'Netting Pre Demand Demand Type Rank D'              ;    DEMANDTYPERANK_IDX = 14
    DEMANDTYPE = 'Netting Pre Demand Demand Type D'                       ;    DEMANDTYPE_IDX = 15
    GCSHORTQTY = 'Netting Pre Demand GC Short Qty D'                      ;    GCSHORTQTY_IDX = 16
    GCSHORTREASON = 'Netting Pre Demand GC Shortreason D'                 ;    GCSHORTREASON_IDX = 17
    GLOBALPRIORITY = 'Netting Pre Demand Global Priority D'               ;    GLOBALPRIORITY_IDX = 18
    GLOBALRULERANK = 'Netting Pre Demand Global Rule Rank D'              ;    GLOBALRULERANK_IDX = 19
    LOCALPRIORITY = 'Netting Pre Demand Local Priority D'                 ;    LOCALPRIORITY_IDX = 20
    LOCALRULERANK = 'Netting Pre Demand Local Rule Rank D'                ;    LOCALRULERANK_IDX = 21
    MEASURETYPERANK = 'Netting Pre Demand Measure Type Rank D'            ;    MEASURETYPERANK_IDX = 22
    MEASURETYPE = 'Netting Pre Demand Measure Type D'                     ;    MEASURETYPE_IDX = 23
    ORDERTYEPRANK = 'Netting Pre Demand Order Type Rank D'                ;    ORDERTYEPRANK_IDX = 24
    ORDERTYPE = 'Netting Pre Demand Order Type D'                         ;    ORDERTYPE_IDX = 25
    PREFERENCERANK = 'Netting Pre Demand Preference Rank D'               ;    PREFERENCERANK_IDX = 26
    PRODUCTRANK = 'Netting Pre Demand Product Rank D'                     ;    PRODUCTRANK_IDX = 27
    QTYPROMISED = 'Netting Pre Demand Promised Qty D'                     ;    QTYPROMISED_IDX = 28
    SHORTQTY = 'Netting Pre Demand Short Qty D'                           ;    SHORTQTY_IDX = 29
    WEEKRANK = 'Netting Pre Demand Week Rank D'                           ;    WEEKRANK_IDX = 30

    LIST_COLUMN = [
        ITEM, SALESID, SITEID, DEMANDID, WEEKBUCKETIDX, 
        MONTHBUCKETIDX, AP1SHORTQTY, AP1SHORTREASON, AP2SHORTQTY, AP2SHORTREASON, 
        ACCSHORTQTY, ACCSHORTREASON, CUSTOMERRANK, DEMANDPRIORITY, DEMANDTYPERANK, 
        DEMANDTYPE, GCSHORTQTY, GCSHORTREASON, GLOBALPRIORITY, GLOBALRULERANK, 
        LOCALPRIORITY, LOCALRULERANK, MEASURETYPERANK, MEASURETYPE, ORDERTYEPRANK, 
        ORDERTYPE, PREFERENCERANK, PRODUCTRANK, QTYPROMISED, SHORTQTY, 
        WEEKRANK,
    ]

class AP1DeliveryPlanD:
    ''' Constants of Netting AP1 Delivery Plan ODS D '''

    VERSIONNAME = 'Version.[Version Name]'                                ;    VERSIONNAME_IDX = 0
    DELIVERYPLANID = 'Documents.[OrderlineID]'                            ;    DELIVERYPLANID_IDX = 1
    ITEM = 'Item.[Item]'                                                  ;    ITEM_IDX = 2
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 3
    PROMISEDSHIPPINGDATE = 'Time.[Day]'                                   ;    PROMISEDSHIPPINGDATE_IDX = 4
    GBM = 'Netting AP1 Delivery Plan GBM D'                               ;    GBM_IDX = 5
    PLANID = 'Netting AP1 Delivery Plan Plan ID D'                        ;    PLANID_IDX = 6
    SALESORDERID = 'Netting AP1 Delivery Plan Sales Order ID D'           ;    SALESORDERID_IDX = 7
    SOLINENUM = 'Netting AP1 Delivery Plan SO Linenum D'                  ;    SOLINENUM_IDX = 8
    QTYOPEN = 'Netting AP1 Delivery Plan Open Qty D'                      ;    QTYOPEN_IDX = 9
    QTYPLANNED = 'Netting AP1 Delivery Plan Planned Qty D'                ;    QTYPLANNED_IDX = 10
    PRIORITY = 'Netting AP1 Delivery Priority D'                          ;    PRIORITY_IDX = 11
    PLANNEDSHIPMENTDATE = 'Netting AP1 Delivery Plan Planned Shipment Date D';    PLANNEDSHIPMENTDATE_IDX = 12
    ES_DEMAND_CODE = 'Netting AP1 Delivery Plan ES Demand Code D'         ;    ES_DEMAND_CODE_IDX = 13
    SALESNAME = 'Netting AP1 Delivery Plan Sales Name D'                  ;    SALESNAME_IDX = 14
    SALESLEVEL = 'Netting AP1 Delivery Plan Sales Level D'                ;    SALESLEVEL_IDX = 15
    REQDELSTARTDATE = 'Netting AP1 Delivery Plan Req Del Start Date D'    ;    REQDELSTARTDATE_IDX = 16

    LIST_COLUMN = [
        VERSIONNAME, DELIVERYPLANID, ITEM, SITEID, PROMISEDSHIPPINGDATE,
        GBM, PLANID, SALESORDERID, SOLINENUM, QTYOPEN, 
        QTYPLANNED, PRIORITY, PLANNEDSHIPMENTDATE, ES_DEMAND_CODE, SALESNAME, 
        SALESLEVEL, REQDELSTARTDATE,
    ]

class AP1ShortReasonD:
    ''' Constants of Netting AP1 Short Reason ODS D '''

    VERSIONNAME = 'Version.[Version Name]'
    SHORTREASONID = 'Documents.[OrderlineID]'
    REQITEMID = 'Item.[Item]'                                             ;    REQITEMID_IDX = 0
    REQSITEID = 'Location.[Location]'                                     ;    REQSITEID_IDX = 1
    DUEDATE = 'Time.[Day]'                                                ;    DUEDATE_IDX = 2
    GBM = 'Netting AP1 Short Reason GBM D'
    PLANID = 'Netting AP1 Short Reason Plan ID D'                         ;    PLANID_IDX = 3
    ITEMREQUEST = 'Netting AP1 Short Reason Item Request D'               ;    ITEMREQUEST_IDX = 4
    PROBLEMID = 'Netting AP1 Short Reason Problem ID D'                   ;    PROBLEMID_IDX = 5
    SALESORDERID = 'Netting AP1 Short Reason Sales Order ID D'            ;    SALESORDERID_IDX = 6
    SHORTQTY = 'Netting AP1 Short Reason Short Qty D'                     ;    SHORTQTY_IDX = 7
    PATHID = 'Netting AP1 Short Reason Path ID D'                         ;    PATHID_IDX = 8
    ORGPROBLEMTYPE = 'Netting AP1 Short Reason Org Problem Type D'        ;    ORGPROBLEMTYPE_IDX = 9
    PROBLEMTYPE = 'Netting AP1 Short Reason Problem Type D'               ;    PROBLEMTYPE_IDX = 10
    PROBLEMDTTM = 'Netting AP1 Short Reason Problem Date D'               ;    PROBLEMDTTM_IDX = 11
    SITEID = 'Netting AP1 Short Reason Site ID D'                         ;    SITEID_IDX = 12
    ITEM = 'Netting AP1 Short Reason Item D'                              ;    ITEM_IDX = 13
    RESOURCENAME = 'Netting AP1 Short Reason Resouce Name D'              ;    RESOURCENAME_IDX = 14
    NEEDEDQTY = 'Netting AP1 Short Reason Needed Qty D'                   ;    NEEDEDQTY_IDX = 15
    OPERATION = 'Netting AP1 Short Reason Operation D'                    ;    OPERATION_IDX = 16
    BOMNAME = 'Netting AP1 Short Reason BOM Name D'                       ;    BOMNAME_IDX = 17
    BODNAME = 'Netting AP1 Short Reason BOD Name D'                       ;    BODNAME_IDX = 18

    LIST_COLUMN = [
        REQITEMID, REQSITEID, DUEDATE, PLANID, ITEMREQUEST, 
        PROBLEMID, SALESORDERID, SHORTQTY, PATHID, ORGPROBLEMTYPE, 
        PROBLEMTYPE, PROBLEMDTTM, SITEID, ITEM, RESOURCENAME, 
        NEEDEDQTY, OPERATION, BOMNAME, BODNAME,
    ]

    LIST_FULL_COLUMN = [
        VERSIONNAME, SHORTREASONID, REQITEMID, REQSITEID, DUEDATE,
        GBM, PLANID, ITEMREQUEST, PROBLEMID, SALESORDERID,
        SHORTQTY, PATHID, ORGPROBLEMTYPE, PROBLEMTYPE, PROBLEMDTTM,
        SITEID, ITEM, RESOURCENAME, NEEDEDQTY, OPERATION,
        BOMNAME, BODNAME,
    ]

class ArcNettedDemandD:
    ''' Constants of Netting Netted Demand ODS D '''

    NEWITEMID = 'Item.[Item]'
    NEWSITEID = 'Location.[Location]'
    NEWSALESID = 'Sales Domain.[Ship To]'
    ITEMID = 'Netting Item.[Item ID]'                                     ;    ITEMID_IDX = 0
    SITEID = 'Netting Site.[Site ID]'                                     ;    SITEID_IDX = 1
    SALESID = 'Netting Sales.[Sales ID]'                                  ;    SALESID_IDX = 2
    PROMISEDDELDATE = 'Time.[Day]'                                        ;    PROMISEDDELDATE_IDX = 3
    SOPROMISEID = 'Netting Demand Arc SO Promise ID D'                    ;    SOPROMISEID_IDX = 4
    PLANID = 'Netting Demand Arc Plan ID D'                               ;    PLANID_IDX = 5
    SALESORDERID = 'Netting Demand Arc Sales Order ID D'                  ;    SALESORDERID_IDX = 6
    SOLINENUM = 'Netting Demand Arc SO Linenum D'                         ;    SOLINENUM_IDX = 7
    QTYPROMISED = 'Netting Demand Arc Promised Qty D'                     ;    QTYPROMISED_IDX = 8
    SHIPTOID = 'Netting Demand Arc Ship To D'                             ;    SHIPTOID_IDX = 9
    SALESLEVEL = 'Netting Demand Arc Sales Level D'                       ;    SALESLEVEL_IDX = 10
    DEMANDTYPERANK = 'Netting Demand Arc Demand Type Rank D'              ;    DEMANDTYPERANK_IDX = 11
    WEEKRANK = 'Netting Demand Arc Week Rank D'                           ;    WEEKRANK_IDX = 12
    CUSTOMERRANK = 'Netting Demand Arc Customer Rank D'                   ;    CUSTOMERRANK_IDX = 13
    PRODUCTRANK = 'Netting Demand Arc Product Rank D'                     ;    PRODUCTRANK_IDX = 14
    DEMANDPRIORITY = 'Netting Demand Arc Demand Priority D'               ;    DEMANDPRIORITY_IDX = 15
    TIEBREAK = 'Netting Demand Arc Tie Break D'                           ;    TIEBREAK_IDX = 16
    TIMEUOM = 'Netting Demand Arc Time UOM D'                             ;    TIMEUOM_IDX = 17
    GLOBALPRIORITY = 'Netting Demand Arc Global Priority D'               ;    GLOBALPRIORITY_IDX = 18
    LOCALPRIORITY = 'Netting Demand Arc Local Priority D'                 ;    LOCALPRIORITY_IDX = 19
    BUSINESSTYPE = 'Netting Demand Arc Business Type D'                   ;    BUSINESSTYPE_IDX = 20
    ROUTING_PRIORITY = 'Netting Demand Arc Routing Priority D'            ;    ROUTING_PRIORITY_IDX = 21
    NO_SPLIT = 'Netting Demand Arc No Split D'                            ;    NO_SPLIT_IDX = 22
    MAP_SATISFY_SS = 'Netting Demand Arc Map Satisfy SS D'                ;    MAP_SATISFY_SS_IDX = 23
    PREALLOC_ATTRIBUTE = 'Netting Demand Arc Pre Allocation Attribute D'  ;    PREALLOC_ATTRIBUTE_IDX = 24
    BUILDAHEADTIME = 'Netting Demand Arc BAT D'                           ;    BUILDAHEADTIME_IDX = 25
    MEASURETYPERANK = 'Netting Demand Arc Measure Type Rank D'            ;    MEASURETYPERANK_IDX = 26
    PREFERENCERANK = 'Netting Demand Arc Preference Rank D'               ;    PREFERENCERANK_IDX = 27
    REASONCODE = 'Netting Demand Arc Reason Code D'                       ;    REASONCODE_IDX = 28
    IS_PLAN_DATE = 'Netting Demand Arc Plan Date Flag D'                  ;    IS_PLAN_DATE_IDX = 29
    MFGWEEK = 'Netting Demand Arc MFG Week D'                             ;    MFGWEEK_IDX = 30
    SUPPLYWEEK1 = 'Netting Demand Arc Supply Week1 D'                     ;    SUPPLYWEEK1_IDX = 31
    SUPPLYWEEK2 = 'Netting Demand Arc Supply Week2 D'                     ;    SUPPLYWEEK2_IDX = 32
    FROZENRANK = 'Netting Demand Arc Frozen Rank D'                       ;    FROZENRANK_IDX = 33
    ADJ_WEEKRANK = 'Netting Demand Arc Adj Week Rank D'                   ;    ADJ_WEEKRANK_IDX = 34
    ADJ_PRIORITY = 'Netting Demand Arc Adj Priority D'                    ;    ADJ_PRIORITY_IDX = 35
    MATCH_CODE = 'Netting Demand Arc Match Code D'                        ;    MATCH_CODE_IDX = 36
    OPTION_CODE = 'Netting Demand Arc Option Code D'                      ;    OPTION_CODE_IDX = 37
    AP1ID = 'Netting Demand Arc AP1ID D'                                  ;    AP1ID_IDX = 38
    MP_PRIORITY = 'Netting Demand Arc MP Priority D'                      ;    MP_PRIORITY_IDX = 39
    MOD_PRIORITY = 'Netting Demand Arc MOD Priority D'                    ;    MOD_PRIORITY_IDX = 40
    REQDELENDDATE = 'Netting Demand Arc Req Delivery End Date D'          ;    REQDELENDDATE_IDX = 41
    REQDELSTARTDATE = 'Netting Demand Arc Req Delivery Start Date D'      ;    REQDELSTARTDATE_IDX = 42
    NO_PLAN = 'Netting Demand Arc No Plan D'                              ;    NO_PLAN_IDX = 43
    SHORT_CODE = 'Netting Demand Arc Short Code D'                        ;    SHORT_CODE_IDX = 44
    CHANNELRANK = 'Netting Demand Arc Channel Rank D'                     ;    CHANNELRANK_IDX = 45
    LOCALID = 'Netting Demand Arc Local ID D'                             ;    LOCALID_IDX = 46
    GCID = 'Netting Demand Arc GCID D'                                    ;    GCID_IDX = 47
    AP2ID = 'Netting Demand Arc AP2ID D'                                  ;    AP2ID_IDX = 48
    PRODUCTGROUP = 'Netting Demand Arc Product Group D'                   ;    PRODUCTGROUP_IDX = 49
    SECTION = 'Netting Demand Arc Section D'                              ;    SECTION_IDX = 50
    ORDER_ENTRY_DATE = 'Netting Demand Arc Order Entry Date D'            ;    ORDER_ENTRY_DATE_IDX = 51

    LIST_COLUMN = [
        ITEMID, SITEID, SALESID, PROMISEDDELDATE, SOPROMISEID, 
        PLANID, SALESORDERID, SOLINENUM, QTYPROMISED, SHIPTOID, 
        SALESLEVEL, DEMANDTYPERANK, WEEKRANK, CUSTOMERRANK, PRODUCTRANK, 
        DEMANDPRIORITY, TIEBREAK, TIMEUOM, GLOBALPRIORITY, LOCALPRIORITY, 
        BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, 
        BUILDAHEADTIME, MEASURETYPERANK, PREFERENCERANK, REASONCODE, IS_PLAN_DATE, 
        MFGWEEK, SUPPLYWEEK1, SUPPLYWEEK2, FROZENRANK, ADJ_WEEKRANK, 
        ADJ_PRIORITY, MATCH_CODE, OPTION_CODE, AP1ID, MP_PRIORITY, 
        MOD_PRIORITY, REQDELENDDATE, REQDELSTARTDATE, NO_PLAN, SHORT_CODE, 
        CHANNELRANK, LOCALID, GCID, AP2ID, PRODUCTGROUP, 
        SECTION, ORDER_ENTRY_DATE,
    ]

class NettedDemandD:
    ''' Constants of Netting Netted Demand ODS D '''
    # 새로 컬럼 이름이 변경 됨
    NEWITEMID = 'Item.[Item]'
    NEWSITEID = 'Location.[Location]'
    NEWSALESID = 'Sales Domain.[Ship To]'
    VERSIONNAME = 'Version.[Version Name]'
    DEMANDID = 'Demand.[DemandID]'
    ITEMID = 'Netting Item.[Item ID]'                                     ;    ITEMID_IDX = 0
    SITEID = 'Netting Site.[Site ID]'                                     ;    SITEID_IDX = 1
    SALESID = 'Netting Sales.[Sales ID]'                                  ;    SALESID_IDX = 2
    PROMISEDDELDATE = 'Time.[Day]'                                        ;    PROMISEDDELDATE_IDX = 3
    GBM = 'Netting Demand GBM D' # 순서가 Dimension 뒤여야 함
    SOPROMISEID = 'Netting Demand SO Promise ID D'                        ;    SOPROMISEID_IDX = 4
    PLANID = 'Netting Demand Plan ID D'                                   ;    PLANID_IDX = 5
    SALESORDERID = 'Netting Demand Sales Order ID D'                      ;    SALESORDERID_IDX = 6
    SOLINENUM = 'Netting Demand SO Linenum D'                             ;    SOLINENUM_IDX = 7
    QTYPROMISED = 'Netting Demand Promised Qty D'                         ;    QTYPROMISED_IDX = 8
    SHIPTOID = 'Netting Demand Ship To D'                                 ;    SHIPTOID_IDX = 9
    SALESLEVEL = 'Netting Demand Sales Level D'                           ;    SALESLEVEL_IDX = 10
    DEMANDTYPERANK = 'Netting Demand Demand Type Rank D'                  ;    DEMANDTYPERANK_IDX = 11
    WEEKRANK = 'Netting Demand Week Rank D'                               ;    WEEKRANK_IDX = 12
    CUSTOMERRANK = 'Netting Demand Customer Rank D'                       ;    CUSTOMERRANK_IDX = 13
    PRODUCTRANK = 'Netting Demand Product Rank D'                         ;    PRODUCTRANK_IDX = 14
    DEMANDPRIORITY = 'Netting Demand Demand Priority D'                   ;    DEMANDPRIORITY_IDX = 15
    TIEBREAK = 'Netting Demand Tie Break D'                               ;    TIEBREAK_IDX = 16
    TIMEUOM = 'Netting Demand Time UOM D'                                 ;    TIMEUOM_IDX = 17
    GLOBALPRIORITY = 'Netting Demand Global Priority D'                   ;    GLOBALPRIORITY_IDX = 18
    LOCALPRIORITY = 'Netting Demand Local Priority D'                     ;    LOCALPRIORITY_IDX = 19
    BUSINESSTYPE = 'Netting Demand Business Type D'                       ;    BUSINESSTYPE_IDX = 20
    ROUTING_PRIORITY = 'Netting Demand Routing Priority D'                ;    ROUTING_PRIORITY_IDX = 21
    NO_SPLIT = 'Netting Demand No Split D'                                ;    NO_SPLIT_IDX = 22
    MAP_SATISFY_SS = 'Netting Demand Map Satisfy SS D'                    ;    MAP_SATISFY_SS_IDX = 23
    PREALLOC_ATTRIBUTE = 'Netting Demand Pre Allocation Attribute D'      ;    PREALLOC_ATTRIBUTE_IDX = 24
    BUILDAHEADTIME = 'Netting Demand BAT D'                               ;    BUILDAHEADTIME_IDX = 25
    MEASURETYPERANK = 'Netting Demand Measure Type Rank D'                ;    MEASURETYPERANK_IDX = 26
    PREFERENCERANK = 'Netting Demand Preference Rank D'                   ;    PREFERENCERANK_IDX = 27
    REASONCODE = 'Netting Demand Reason Code D'                           ;    REASONCODE_IDX = 28
    IS_PLAN_DATE = 'Netting Demand Plan Date Flag D'                      ;    IS_PLAN_DATE_IDX = 29
    MFGWEEK = 'Netting Demand MFG Week D'                                 ;    MFGWEEK_IDX = 30
    SUPPLYWEEK1 = 'Netting Demand Supply Week1 D'                         ;    SUPPLYWEEK1_IDX = 31
    SUPPLYWEEK2 = 'Netting Demand Supply Week2 D'                         ;    SUPPLYWEEK2_IDX = 32
    FROZENRANK = 'Netting Demand Frozen Rank D'                           ;    FROZENRANK_IDX = 33
    ADJ_WEEKRANK = 'Netting Demand Adj Week Rank D'                       ;    ADJ_WEEKRANK_IDX = 34
    ADJ_PRIORITY = 'Netting Demand Adj Priority D'                        ;    ADJ_PRIORITY_IDX = 35
    MATCH_CODE = 'Netting Demand Match Code D'                            ;    MATCH_CODE_IDX = 36
    OPTION_CODE = 'Netting Demand Option Code D'                          ;    OPTION_CODE_IDX = 37
    AP1ID = 'Netting Demand AP1ID D'                                      ;    AP1ID_IDX = 38
    MP_PRIORITY = 'Netting Demand MP Priority D'                          ;    MP_PRIORITY_IDX = 39
    MOD_PRIORITY = 'Netting Demand MOD Priority D'                        ;    MOD_PRIORITY_IDX = 40
    REQDELENDDATE = 'Netting Demand Req Delivery End Date D'              ;    REQDELENDDATE_IDX = 41
    REQDELSTARTDATE = 'Netting Demand Req Delivery Start Date D'          ;    REQDELSTARTDATE_IDX = 42
    NO_PLAN = 'Netting Demand No Plan D'                                  ;    NO_PLAN_IDX = 43
    SHORT_CODE = 'Netting Demand Short Code D'                            ;    SHORT_CODE_IDX = 44
    CHANNELRANK = 'Netting Demand Channel Rank D'                         ;    CHANNELRANK_IDX = 45
    LOCALID = 'Netting Demand Local ID D'                                 ;    LOCALID_IDX = 46
    GCID = 'Netting Demand GCID D'                                        ;    GCID_IDX = 47
    AP2ID = 'Netting Demand AP2ID D'                                      ;    AP2ID_IDX = 48
    PRODUCTGROUP = 'Netting Demand Product Group D'                       ;    PRODUCTGROUP_IDX = 49
    SECTION = 'Netting Demand Section D'                                  ;    SECTION_IDX = 50
    ORDER_ENTRY_DATE = 'Netting Demand Order Entry Date D'                ;    ORDER_ENTRY_DATE_IDX = 51

    LIST_COLUMN = [
        ITEMID, SITEID, SALESID, PROMISEDDELDATE, SOPROMISEID, 
        PLANID, SALESORDERID, SOLINENUM, QTYPROMISED, SHIPTOID, 
        SALESLEVEL, DEMANDTYPERANK, WEEKRANK, CUSTOMERRANK, PRODUCTRANK, 
        DEMANDPRIORITY, TIEBREAK, TIMEUOM, GLOBALPRIORITY, LOCALPRIORITY, 
        BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, 
        BUILDAHEADTIME, MEASURETYPERANK, PREFERENCERANK, REASONCODE, IS_PLAN_DATE, 
        MFGWEEK, SUPPLYWEEK1, SUPPLYWEEK2, FROZENRANK, ADJ_WEEKRANK, 
        ADJ_PRIORITY, MATCH_CODE, OPTION_CODE, AP1ID, MP_PRIORITY, 
        MOD_PRIORITY, REQDELENDDATE, REQDELSTARTDATE, NO_PLAN, SHORT_CODE, 
        CHANNELRANK, LOCALID, GCID, AP2ID, PRODUCTGROUP, 
        SECTION, ORDER_ENTRY_DATE,
    ]

    LIST_FULL_COLUMN = [
        VERSIONNAME, DEMANDID, ITEMID, SITEID, SALESID,
        PROMISEDDELDATE, GBM, SOPROMISEID, PLANID, SALESORDERID,
        SOLINENUM, QTYPROMISED, SHIPTOID, 
        SALESLEVEL, DEMANDTYPERANK, WEEKRANK, CUSTOMERRANK, PRODUCTRANK, 
        DEMANDPRIORITY, TIEBREAK, TIMEUOM, GLOBALPRIORITY, LOCALPRIORITY, 
        BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, 
        BUILDAHEADTIME, MEASURETYPERANK, PREFERENCERANK, REASONCODE, IS_PLAN_DATE, 
        MFGWEEK, SUPPLYWEEK1, SUPPLYWEEK2, FROZENRANK, ADJ_WEEKRANK, 
        ADJ_PRIORITY, MATCH_CODE, OPTION_CODE, AP1ID, MP_PRIORITY, 
        MOD_PRIORITY, REQDELENDDATE, REQDELSTARTDATE, NO_PLAN, SHORT_CODE, 
        CHANNELRANK, LOCALID, GCID, AP2ID, PRODUCTGROUP, 
        SECTION, ORDER_ENTRY_DATE,
    ]

# Netted Demand 추가되는 컬럼 정의
class NDEMAND_ADD :
    # SO 컬럼
    ORDERREASON = 'Netting Demand Order Reason D'
    ORDERTYPE = 'Netting Demand Order Type D'
    ORDERTYPEDESC = 'Netting Demand Order Type Desc D'
    RDD = 'Netting Demand RDD D'
    SOCREATIONDATE = 'Netting Demand SO Create Date D'
    SOITEM = 'Netting Demand SO Item D'
    SONO = 'Netting Demand SO No D'
    SHIPTOPARTY = 'Netting Demand Ship To Party D'
    SOLDTOPARTY = 'Netting Demand Sold To Party D'
    # 권역 관련 컬럼
    SCENARIO = 'Netting Demand Scenario D'
    TENANT = 'Netting Demand Tenant D'

    LIST_COLUMN = [
        ORDERREASON, ORDERTYPE, ORDERTYPEDESC, RDD, SOCREATIONDATE,
        SOITEM, SONO, SHIPTOPARTY, SOLDTOPARTY, SCENARIO,
        TENANT,
    ]

class NettedDemandRBD:
    ''' Constants of Netting Netted Demand ODS RB D '''

    VERSIONNAME = 'Version.[Version Name]'
    DEMANDID = 'Demand.[DemandID]'
    ITEMID = 'Item.[Item]'
    SITEID = 'Location.[Location]'
    SALESID = 'Sales Domain.[Ship To]'
    PROMISEDDELDATE = 'Time.[Day]'
    GBM = 'Netting Demand RB GBM D'
    SOPROMISEID = 'Netting Demand RB SO Promise ID D'
    PLANID = 'Netting Demand RB Plan ID D'
    SALESORDERID = 'Netting Demand RB Sales Order ID D'
    SOLINENUM = 'Netting Demand RB SO Linenum D'
    QTYPROMISED = 'Netting Demand RB Promised Qty D'
    SHIPTOID = 'Netting Demand RB Ship To D'
    SALESLEVEL = 'Netting Demand RB Sales Level D'
    DEMANDTYPERANK = 'Netting Demand RB Demand Type Rank D'
    WEEKRANK = 'Netting Demand RB Week Rank D'
    CUSTOMERRANK = 'Netting Demand RB Customer Rank D'
    PRODUCTRANK = 'Netting Demand RB Product Rank D'
    DEMANDPRIORITY = 'Netting Demand RB Demand Priority D'
    TIEBREAK = 'Netting Demand RB Tie Break D'
    TIMEUOM = 'Netting Demand RB Time UOM D'
    GLOBALPRIORITY = 'Netting Demand RB Global Priority D'
    LOCALPRIORITY = 'Netting Demand RB Local Priority D'
    BUSINESSTYPE = 'Netting Demand RB Business Type D'
    ROUTING_PRIORITY = 'Netting Demand RB Routing Priority D'
    NO_SPLIT = 'Netting Demand RB No Split D'
    MAP_SATISFY_SS = 'Netting Demand RB Map Satisfy SS D'
    PREALLOC_ATTRIBUTE = 'Netting Demand RB Pre Allocation Attribute D'
    BUILDAHEADTIME = 'Netting Demand RB BAT D'
    MEASURETYPERANK = 'Netting Demand RB Measure Type Rank D'
    PREFERENCERANK = 'Netting Demand RB Preference Rank D'
    REASONCODE = 'Netting Demand RB Reason Code D'
    IS_PLAN_DATE = 'Netting Demand RB Plan Date Flag D'
    MFGWEEK = 'Netting Demand RB MFG Week D'
    SUPPLYWEEK1 = 'Netting Demand RB Supply Week1 D'
    SUPPLYWEEK2 = 'Netting Demand RB Supply Week2 D'
    FROZENRANK = 'Netting Demand RB Frozen Rank D'
    ADJ_WEEKRANK = 'Netting Demand RB Adj Week Rank D'
    ADJ_PRIORITY = 'Netting Demand RB Adj Priority D'
    MATCH_CODE = 'Netting Demand RB Match Code D'
    OPTION_CODE = 'Netting Demand RB Option Code D'
    AP1ID = 'Netting Demand RB AP1ID D'
    MP_PRIORITY = 'Netting Demand RB MP Priority D'
    MOD_PRIORITY = 'Netting Demand RB MOD Priority D'
    REQDELENDDATE = 'Netting Demand RB Req Delivery End Date D'
    REQDELSTARTDATE = 'Netting Demand RB Req Delivery Start Date D'
    NO_PLAN = 'Netting Demand RB No Plan D'
    SHORT_CODE = 'Netting Demand RB Short Code D'
    CHANNELRANK = 'Netting Demand RB Channel Rank D'
    LOCALID = 'Netting Demand RB Local ID D'
    GCID = 'Netting Demand RB GCID D'
    AP2ID = 'Netting Demand RB AP2ID D'
    PRODUCTGROUP = 'Netting Demand RB Product Group D'
    SECTION = 'Netting Demand RB Section D'
    ORDER_ENTRY_DATE= 'Netting Demand RB Order Entry Date D'

    LIST_COLUMN = [
        VERSIONNAME, DEMANDID, ITEMID, SITEID, SALESID,
        PROMISEDDELDATE, GBM, SOPROMISEID, PLANID, SALESORDERID,
        SOLINENUM, QTYPROMISED, SHIPTOID, 
        SALESLEVEL, DEMANDTYPERANK, WEEKRANK, CUSTOMERRANK, PRODUCTRANK, 
        DEMANDPRIORITY, TIEBREAK, TIMEUOM, GLOBALPRIORITY, LOCALPRIORITY, 
        BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, 
        BUILDAHEADTIME, MEASURETYPERANK, PREFERENCERANK, REASONCODE, IS_PLAN_DATE, 
        MFGWEEK, SUPPLYWEEK1, SUPPLYWEEK2, FROZENRANK, ADJ_WEEKRANK, 
        ADJ_PRIORITY, MATCH_CODE, OPTION_CODE, AP1ID, MP_PRIORITY, 
        MOD_PRIORITY, REQDELENDDATE, REQDELSTARTDATE, NO_PLAN, SHORT_CODE, 
        CHANNELRANK, LOCALID, GCID, AP2ID, PRODUCTGROUP, 
        SECTION, ORDER_ENTRY_DATE,
    ]

class NettedDemandLogD:
    ''' Constants of Netting Netted Demand Log ODS D '''

    VERSIONNAME = 'Version.[Version Name]'                                ;    VERSIONNAME_IDX = 0
    DEMANDID = 'Demand.[DemandID]'                                        ;    DEMANDID_IDX = 1
    PLANSTEP = 'Netting Plan Option.[Plan Step]'                          ;    PLANSTEP_IDX = 2
    ITEMID = 'Item.[Item]'                                                ;    ITEMID_IDX = 3
    SITEID = 'Location.[Location]'                                        ;    SITEID_IDX = 4
    SALESID = 'Sales Domain.[Ship To]'                                    ;    SALESID_IDX = 5
    PROMISEDDELDATE = 'Time.[Day]'                                        ;    PROMISEDDELDATE_IDX = 6
    GBM = 'Netting Demand Log GBM D'                                      ;    GBM_IDX = 7
    SOPROMISEID = 'Netting Demand Log SO Promise ID D'                    ;    SOPROMISEID_IDX = 8
    PLANID = 'Netting Demand Log Plan ID D'                               ;    PLANID_IDX = 9
    SALESORDERID = 'Netting Demand Log Sales Order ID D'                  ;    SALESORDERID_IDX = 10
    SOLINENUM = 'Netting Demand Log SO Linenum D'                         ;    SOLINENUM_IDX = 11
    QTYPROMISED = 'Netting Demand Log Promised Qty D'                     ;    QTYPROMISED_IDX = 12
    SHIPTOID = 'Netting Demand Log Ship To D'                             ;    SHIPTOID_IDX = 13
    SALESLEVEL = 'Netting Demand Log Sales Level D'                       ;    SALESLEVEL_IDX = 14
    DEMANDTYPERANK = 'Netting Demand Log Demand Type Rank D'              ;    DEMANDTYPERANK_IDX = 15
    WEEKRANK = 'Netting Demand Log Week Rank D'                           ;    WEEKRANK_IDX = 16
    CUSTOMERRANK = 'Netting Demand Log Customer Rank D'                   ;    CUSTOMERRANK_IDX = 17
    PRODUCTRANK = 'Netting Demand Log Product Rank D'                     ;    PRODUCTRANK_IDX = 18
    DEMANDPRIORITY = 'Netting Demand Log Demand Priority D'               ;    DEMANDPRIORITY_IDX = 19
    TIEBREAK = 'Netting Demand Log Tie Break D'                           ;    TIEBREAK_IDX = 20
    TIMEUOM = 'Netting Demand Log Time UOM D'                             ;    TIMEUOM_IDX = 21
    GLOBALPRIORITY = 'Netting Demand Log Global Priority D'               ;    GLOBALPRIORITY_IDX = 22
    LOCALPRIORITY = 'Netting Demand Log Local Priority D'                 ;    LOCALPRIORITY_IDX = 23
    BUSINESSTYPE = 'Netting Demand Log Business Type D'                   ;    BUSINESSTYPE_IDX = 24
    ROUTING_PRIORITY = 'Netting Demand Log Routing Priority D'            ;    ROUTING_PRIORITY_IDX = 25
    NO_SPLIT = 'Netting Demand Log No Split D'                            ;    NO_SPLIT_IDX = 26
    MAP_SATISFY_SS = 'Netting Demand Log Map Satisfy SS D'                ;    MAP_SATISFY_SS_IDX = 27
    PREALLOC_ATTRIBUTE = 'Netting Demand Log Pre Allocation Attribute D'  ;    PREALLOC_ATTRIBUTE_IDX = 28
    BUILDAHEADTIME = 'Netting Demand Log BAT D'                           ;    BUILDAHEADTIME_IDX = 29
    MEASURETYPERANK = 'Netting Demand Log Measure Type Rank D'            ;    MEASURETYPERANK_IDX = 30
    PREFERENCERANK = 'Netting Demand Log Preference Rank D'               ;    PREFERENCERANK_IDX = 31
    REASONCODE = 'Netting Demand Log Reason Code D'                       ;    REASONCODE_IDX = 32
    IS_PLAN_DATE = 'Netting Demand Log Plan Date Flag D'                  ;    IS_PLAN_DATE_IDX = 33
    MFGWEEK = 'Netting Demand Log MFG Week D'                             ;    MFGWEEK_IDX = 34
    SUPPLYWEEK1 = 'Netting Demand Log Supply Week1 D'                     ;    SUPPLYWEEK1_IDX = 35
    SUPPLYWEEK2 = 'Netting Demand Log Supply Week2 D'                     ;    SUPPLYWEEK2_IDX = 36
    FROZENRANK = 'Netting Demand Log Frozen Rank D'                       ;    FROZENRANK_IDX = 37
    ADJ_WEEKRANK = 'Netting Demand Log Adj Week Rank D'                   ;    ADJ_WEEKRANK_IDX = 38
    ADJ_PRIORITY = 'Netting Demand Log Adj Priority D'                    ;    ADJ_PRIORITY_IDX = 39
    MATCH_CODE = 'Netting Demand Log Match Code D'                        ;    MATCH_CODE_IDX = 40
    OPTION_CODE = 'Netting Demand Log Option Code D'                      ;    OPTION_CODE_IDX = 41
    AP1ID = 'Netting Demand Log AP1ID D'                                  ;    AP1ID_IDX = 42
    MP_PRIORITY = 'Netting Demand Log MP Priority D'                      ;    MP_PRIORITY_IDX = 43
    MOD_PRIORITY = 'Netting Demand Log MOD Priority D'                    ;    MOD_PRIORITY_IDX = 44
    REQDELENDDATE = 'Netting Demand Log Req Delivery Start Date D'        ;    REQDELENDDATE_IDX = 45
    REQDELSTARTDATE = 'Netting Demand Log Req Delivery End Date D'        ;    REQDELSTARTDATE_IDX = 46
    NO_PLAN = 'Netting Demand Log No Plan D'                              ;    NO_PLAN_IDX = 47
    SHORT_CODE = 'Netting Demand Log Short Code D'                        ;    SHORT_CODE_IDX = 48
    CHANNELRANK = 'Netting Demand Log Channel Rank D'                     ;    CHANNELRANK_IDX = 49
    LOCALID = 'Netting Demand Log Local ID D'                             ;    LOCALID_IDX = 50
    GCID = 'Netting Demand Log GCID D'                                    ;    GCID_IDX = 51
    AP2ID = 'Netting Demand Log AP2ID D'                                  ;    AP2ID_IDX = 52
    PRODUCTGROUP = 'Netting Demand Log Product Group D'                   ;    PRODUCTGROUP_IDX = 53
    SECTION = 'Netting Demand Log Section D'                              ;    SECTION_IDX = 54
    ORDER_ENTRY_DATE = 'Netting Demand Log Order Entry Date D'            ;    ORDER_ENTRY_DATE_IDX = 55

    LIST_COLUMN = [
        VERSIONNAME, DEMANDID, PLANSTEP, ITEMID, SITEID,
        SALESID, PROMISEDDELDATE, GBM, SOPROMISEID, PLANID, 
        SALESORDERID, SOLINENUM, QTYPROMISED, SHIPTOID, SALESLEVEL, 
        DEMANDTYPERANK, WEEKRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, 
        TIEBREAK, TIMEUOM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, 
        ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, 
        MEASURETYPERANK, PREFERENCERANK, REASONCODE, IS_PLAN_DATE, MFGWEEK, 
        SUPPLYWEEK1, SUPPLYWEEK2, FROZENRANK, ADJ_WEEKRANK, ADJ_PRIORITY, 
        MATCH_CODE, OPTION_CODE, AP1ID, MP_PRIORITY, MOD_PRIORITY, 
        REQDELENDDATE, REQDELSTARTDATE, NO_PLAN, SHORT_CODE, CHANNELRANK, 
        LOCALID, GCID, AP2ID, PRODUCTGROUP, SECTION,
        ORDER_ENTRY_DATE,
    ]
    