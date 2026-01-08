"""
* 프로그램명
    - PYForecastMeasureLockColor

* 내용
    - MP Engine Basic Input 생성하기 전 실제 엔진에 필요한 Demand를 확정하는 프로그램
    - MP Plan Batch Procedure에서 Call 하는 Program
    - Demand MG 데이터 셋에서 에어컨 디멘트 가공

* 변경이력
    - 2024.10.31 / Kang Gyeonghun / Create

* Script Params
    - Param_OUT_VERSION : 기본값 : CWV_MP, 호출시에, 결과데이터를 Version Name을 별도로 지정하는데에 사용.

* Input Tables
    - in_Demand (데이터필수)
        Select ([Version].[Version Name] * [Item].[Item] * [Location].[Location] * [Sales Domain].[Customer Group]
             * [Demand Type].[Demand Type] * [Time].[Week] * [Demand].[DemandID]) on row
             , ({Measure.[W Demand Quantity to SCS], Measure.[W Demand Build Ahead Limit]}) on column
        where {Measure.[W Demand Build Ahead Limit]>0}
        ;

    - in_Time
        Select ([Version].[Version Name] * [Time].[Week]) on row
             , ({Measure.[Plan Week Master_Plan Current Bucket], Measure.[Plan Week Master_Plan End Bucket]
             , Measure.[Plan Week Master_Plan Start Bucket], Measure.[Plan Week Master_Plan Week]}) on column
        ;

* Output Tables
    - out_Demand
        Select ([Version].[Version Name] * [Item].[Item] * [Location].[Location] * [Time].[Week]) on row
             , ({Measure.[W Quantity Max Target]}) on column
        ;

* Flow Summary

    Step 12
        - Puppose.
            I will create dataframe named like df_fn_Sales_Product_ASN_Item_Week_Forcast based on df_in_Forecast_Rule From df_fn_Sales_Product_ASN_Item_Week.
        - Basic Knowlege
            - how to know what level Sales_Domain_ShipTo is along to
                - if start with '2' , then level is 2.
                - if start with '3' , then level is 3.
                - if start with 'A3' , then level is 4.
                - if start with '4' , then level is 5.
                - if start with '5' , then level is 6.
                - if start with 'A5' , then level is 7.
        - Process
            - create empty dataframe named df_fn_Sales_Product_ASN_Item_Week_Forcast
                - columns
                    string  Sales_Domain_ShipTo    "level declared in df_in_Forecast_Rule(FORECAST_RULE_GC_FCST,FORECAST_RULE_AP2_FCST,FORECAST_RULE_AP1_FCST)"
                    string  Item_Item
                    string  Item_GBM    
                    string  Location_Location  
                    string  Partial_Week 
                    bool    SIn_FCST_GC_LOCK        "see FORECAST_RULE_GC_FCST"
                    bool    SIn_FCST_AP2_LOCK       "see FORECAST_RULE_AP2_FCST"
                    bool    SIn_FCST_AP1_LOCK       "see FORECAST_RULE_AP1_FCST"
                    string  SIn_FCST_Color_Condition     ""  

            - 01. Find data from df_fn_Sales_Product_ASN_Item_Week according to df_in_Forecast_Rule. and insert to df_fn_Sales_Product_ASN_Item_Week_Forcast
                - matching condition is below.
                    - df_in_Forecast_Rule
                        - Sales_Domain_ShipTo
                            this is level 2,3
                        - Product_Group
                    - df_fn_Sales_Product_ASN_Item_Week
                        - Sales_Domain_ShipTo
                            this is leve 7
                        - Product_Group
                - matching solve
                    - note: df_in_Forecast_Rule.Sales_Domain_ShipTo is leve 2 or 3.  and We have to find level2 or level3 from df_in_Sales_Domain_Dimension by using df_fn_Sales_Product_ASN_Item_Week.Sales_Domain_ShipTo
                    
                    - Find Base data and Insert to df_fn_Sales_Product_ASN_Item_Week_Forcast
                        - I will use like sql for convinience of understanding. but we operate data frame. and It is better to use index based lookup
                        - Below found data is base data for df_fn_Sales_Product_ASN_Item_Week_Forcast
                        - If df_in_Forecast_Rule.Sales_Domain_ShipTo is level2
                            Select 
                                week.Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                week.SIn_FCST_GC_LOCK,
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP2_LOCK,
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP1_LOCK,
                                week.SIn_FCST_Color_Condition,

                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV2   
                                and forcast.Product_Group = week.Product_Group
                            where forcast.Sales_Domain_ShipTo = '2%'
                                                 
                                
                        - If df_in_Forecast_Rule.Sales_Domain_ShipTo is level3
                            Select 
                                week.Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                week.SIn_FCST_GC_LOCK,
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP2_LOCK,
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP1_LOCK,
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%'
                    



                
            - 02. Insert or update df_fn_Sales_Product_ASN_Item_Week_Forcast.
                - in below. there is exprission like 'null as '. this is mean this column is not target columm
                - Work For df_in_Forecast_Rule.FORECAST_RULE_GC_FCST
                    - FORECAST_RULE_GC_FCST is range from 2 to 7.
                    - Befor insert check if there is data
                        - keys
                            Sales_Domain_ShipTo
                            Item_Item
                            Location_Location
                            Partial_Week
                        - if there is data , update for column ( SIn_FCST_GC_LOCK, SIn_FCST_AP2_LOCK, SIn_FCST_AP1_LOCK)
                    - Below ex is just for forcast.Sales_Domain_ShipTo start with '3' for my convinience. But also consider forcast.Sales_Domain_ShipTo start with '2'
                    
                    - If FORECAST_RULE_GC_FCST is 7
                        - update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV7 as Sales_Domain_ShipTo,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                week.SIn_FCST_GC_LOCK,  
                                null as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,                              
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 6
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV6 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                week.SIn_FCST_GC_LOCK,
                                null as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    
                    - If FORECAST_RULE_GC_FCST is 5
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV5 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                week.SIn_FCST_GC_LOCK,
                                null as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,
                                week.SIn_FCST_Color_Condition,
                                
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 4
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV4 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                week.SIn_FCST_GC_LOCK,
                                null as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 3
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV3 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                week.SIn_FCST_GC_LOCK,
                                null as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 2
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV2 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                week.SIn_FCST_GC_LOCK,
                                null as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 

                - Work For df_in_Forecast_Rule.FORECAST_RULE_AP2_FCST
                    - FORECAST_RULE_AP2_FCST is range from 2 to 7.
                    - Befor insert check if there is data
                        - keys
                            Sales_Domain_ShipTo
                            Item_Item
                            Location_Location
                            Partial_Week
                        - if there is data , update for column ( SIn_FCST_GC_LOCK, SIn_FCST_AP2_LOCK, SIn_FCST_AP1_LOCK)
                    - Below ex is just for forcast.Sales_Domain_ShipTo start with '3' for my convinience. But also consider forcast.Sales_Domain_ShipTo start with '2'
                    
                    - If FORECAST_RULE_AP2_FCST is 7
                        - update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV7 as Sales_Domain_ShipTo,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,                              
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 6
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV6 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,                         
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    
                    - If FORECAST_RULE_GC_FCST is 5
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV5 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,                         
                                week.SIn_FCST_Color_Condition,
                                
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 4
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV4 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,                         
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 3
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV3 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,                         
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 2
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV2 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP2_LOCK,
                                null as SIn_FCST_AP1_LOCK,                         
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                - Work For df_in_Forecast_Rule.SIn_FCST_AP1_LOCK
                    - SIn_FCST_AP1_LOCK is range from 2 to 7.
                    - Befor insert check if there is data
                        - keys
                            Sales_Domain_ShipTo
                            Item_Item
                            Location_Location
                            Partial_Week
                        - if there is data , update for column ( SIn_FCST_GC_LOCK, SIn_FCST_AP2_LOCK, SIn_FCST_AP1_LOCK)
                    - Below ex is just for forcast.Sales_Domain_ShipTo start with '3' for my convinience. But also consider forcast.Sales_Domain_ShipTo start with '2'
                    
                    - If SIn_FCST_AP1_LOCK is 7
                        - update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV7 as Sales_Domain_ShipTo,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                null as SIn_FCST_AP2_LOCK,
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP1_LOCK,                              
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 6
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV6 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                null as SIn_FCST_AP2_LOCK,
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP1_LOCK,   
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    
                    - If FORECAST_RULE_GC_FCST is 5
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV5 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                null as SIn_FCST_AP2_LOCK,
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP1_LOCK,   
                                week.SIn_FCST_Color_Condition,
                                
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 4
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV4 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                null as SIn_FCST_AP2_LOCK,
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP1_LOCK,   
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 3
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV3 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                null as SIn_FCST_AP2_LOCK,
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP1_LOCK,   
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 
                    - If FORECAST_RULE_GC_FCST is 2
                        - insert or update like below.  also use sql. but we must use index based lookup
                            select 
                                dim.Sales_Domain_LV2 as Sales_Domain_ShipTo ,
                                week.Item_Item ,
                                week.Location_Location,
                                week.Partial_Week,
                                null as SIn_FCST_GC_LOCK,  
                                null as SIn_FCST_AP2_LOCK,
                                week.SIn_FCST_GC_LOCK as SIn_FCST_AP1_LOCK,   
                                week.SIn_FCST_Color_Condition
                            from join df_fn_Sales_Product_ASN_Item_Week week  
                            inner join df_in_Sales_Domain_Dimension dim
                                on week.Sales_Domain_ShipTo = dim.Sales_Domain_ShipTo
                            inner join  df_in_Forecast_Rule forcast 
                                on forcast.Sales_Domain_ShipTo = dim.Sales_Domain_LV3   
                                and forcast.Product_Group = week.Product_Group   
                            where forcast.Sales_Domain_ShipTo = '3%' 

* Validation :

* Execution
    EXEC plugin instance [PY_MG_MaxTargetQtyPlanB_REG]
         for measures {[W Quantity Max Target]}
    using scope ([Version].[Version Name].[CWV_MP] * &PlanningHorizon)
    using arguments {
                    (ExecutionMode, "LightWeight")
                  , (Param_OUT_VERSION, "CWV_MP")
                  , (Param_Exception_Flag, "N")
                  , (MaxSliceTableCells, 200000000)
                  , (MaxConcurrency, 1 )
                  , ([Param.o9_sys_log_level], "DEBUG")
                    }
    ;
"""

from re import X
import os,sys,json,shutil,io,zipfile
import time
import datetime
import inspect
import traceback
import pandas as pd
from NSCMCommon import NSCMCommon as common
# from typing_extensions import Literal
import glob
import numpy as np
# import rbql
import duckdb

########################################################################################################################
# Local 개발 시에 필요한 공통 변수 선언
########################################################################################################################
# o9에 저장된 instanceName
is_local = common.gfn_get_isLocal()
str_instance = 'PYForecastMeasureLockColor'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"

is_print = True
flag_csv = True
flag_exception = True

########################################################################################################################
# 컬럼상수
########################################################################################################################
Version_Name = 'Version.[Version Name]'
Item_Item = 'Item.[Item]'
# Item_Type = 'Item.[Item Type]'  
Item_Type = 'Item.[ProductType]'  
Item_GBM = 'Item.[Item GBM]'
Product_Group = 'Item.[Product Group]'
Item_Lv = 'Item_Lv'
RTS_EOS_ShipTo          = 'RTS_EOS_ShipTo'
ForecastRuleShipto      = 'ForecastRuleShipto'
Sales_Domain_ShipTo     = 'Sales Domain.[Ship To]'
Sales_Domain_LV2        = 'Sales Domain.[Sales Domain Lv2]'
Sales_Domain_LV3        = 'Sales Domain.[Sales Domain Lv3]'
Sales_Domain_LV4        = 'Sales Domain.[Sales Domain Lv4]' 
Sales_Domain_LV5        = 'Sales Domain.[Sales Domain Lv5]'
Sales_Domain_LV6        = 'Sales Domain.[Sales Domain Lv6]'
Sales_Domain_LV7        = 'Sales Domain.[Sales Domain Lv7]'
Location_Location       = 'Location.[Location]'
Item_Class              = 'ITEMCLASS Class'
Partial_Week            = 'Time.[Partial Week]'
SIn_FCST_GC_LOCK                = 'S/In FCST(GI)_GC.Lock'
SIn_FCST_Color_Condition        = 'S/In FCST Color Condition'
SIn_FCST_AP2_LOCK               = 'S/In FCST(GI)_AP2.Lock'
SIn_FCST_AP1_LOCK               = 'S/In FCST(GI)_AP1.Lock'


# Salse_Product_ASN       = 'Sales Product ASN'    
Salse_Product_ASN       = 'Sales Product ASN'
ITEMTAT_TATTERM         = 'ITEMTAT TATTERM'
ITEMTAT_TATTERM_SET     = 'ITEMTAT TATTERM_SET'
FORECAST_RULE_GC_FCST        = 'FORECAST_RULE GC FCST'
FORECAST_RULE_AP2_FCST       = 'FORECAST_RULE AP2 FCST'
FORECAST_RULE_AP1_FCST       = 'FORECAST_RULE AP1 FCST'   
FORECAST_RULE_CUST      = 'FORECAST_RULE CUST FCST'
# ----------------------------------------------------------------
# New column constants for step13
# ----------------------------------------------------------------
SOut_FCST_GC_LOCK         = 'S/Out FCST_GC.Lock'
SOut_FCST_AP2_LOCK        = 'S/Out FCST_AP2.Lock'
SOut_FCST_AP1_LOCK        = 'S/Out FCST_AP1.Lock'
SOut_FCST_Color_Condition = 'S/Out FCST Color Condition'


# ----------------------------------------------------------------
# Helper column constants for step02
# ----------------------------------------------------------------
CURRENT_ROW_WEEK                    = 'current_row_partial_week_normalized'
CURRENT_ROW_WEEK_PLUS_8             = 'CURRENTWEEK_NORMALIZED_PLUS_8'   

RTS_INIT_DATE                       = 'RTS_INIT_DATE'
RTS_DEV_DATE                        = 'RTS_DEV_DATE'
RTS_COM_DATE                        = 'RTS_COM_DATE'

RTS_WEEK                            = 'RTS_WEEK_NORMALIZED'
RTS_PARTIAL_WEEK                    = 'RTS_PARTIAL_WEEK'
RTS_INITIAL_WEEK                    = 'RTS_INITIAL_WEEK_NORMALIZED'
RTS_WEEK_MINUST_1                   = 'RTS_WEEK_NORMALIZED_MINUST_1'
RTS_WEEK_PLUS_3                     = 'RTS_WEEK_NORMALIZED_PLUS_3'
MAX_RTS_CURRENTWEEK                 = 'MAX_RTS_CURRENTWEEK'
RTS_STATUS                          = 'RTS_STATUS'

EOS_INIT_DATE                       = 'EOS_INIT_DATE'
EOS_CHG_DATE                        = 'EOS_CHG_DATE'
EOS_COM_DATE                        = 'EOS_COM_DATE'

EOS_WEEK                            = 'EOS_WEEK_NORMALIZED'
EOS_PARTIAL_WEEK                    = 'EOS_PARTIAL_WEEK'
# EOS_WEEK_MINUS_1                    = 'EOS_WEEK_NORMALIZED_MINUS_1'
EOS_WEEK_MINUS_1         = 'EOS_WEEK_NORMALIZED_MINUS_1'
EOS_WEEK_MINUS_4         = 'EOS_WEEK_NORMALIZED_MINUS_4'
EOS_INITIAL_WEEK         = 'EOS_INITIAL_WEEK_NORMALIZED'
MIN_EOSINI_MAXWEEK                  = 'MIN_EOSINI_MAXWEEK'
MIN_EOS_MAXWEEK                  = 'MIN_EOS_MAXWEEK'
EOS_STATUS                          = 'EOS_STATUS'
# ───────────────────────────────────────────────────────────────
# CONSTANT STRING VARIABLES FOR DATAFRAME NAMES
# ───────────────────────────────────────────────────────────────

str_df_in_Sales_Domain                  = 'df_in_Sales_Domain_Dimension'
str_df_in_Sales_Domain_Estore           = 'df_in_Sales_Domain_Estore'
str_df_in_Time_Partial_Week             = 'df_in_Time_Partial_Week'
str_df_in_Item_CLASS                    = 'df_in_Item_CLASS'
str_df_in_Item_TAT                      = 'df_in_Item_TAT'
str_df_in_MST_EOS                       = 'df_in_MST_EOS'
str_df_in_MST_RTS                       = 'df_in_MST_RTS'
str_df_in_Sales_Product_ASN             = 'df_in_Sales_Product_ASN'
str_df_in_Forecast_Rule                 = 'df_in_Forecast_Rule'
str_df_in_Item_Master                   = 'df_in_Item_Master'

# middle
str_df_fn_RTS_EOS                           = 'df_fn_RTS_EOS'
str_df_fn_RTS_EOS_Week                      = 'df_fn_RTS_EOS_Week'
str_df_fn_Sales_Product_ASN_Item            = 'df_fn_Sales_Product_ASN_Item'
str_df_fn_Sales_Product_ASN_Item_Week       = 'df_fn_Sales_Product_ASN_Item_Week'
str_df_fn_Forcast_in                        = 'df_fn_Forcast_in'
str_df_fn_Forcast_out                       = 'df_fn_Forcast_out'
# out



########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################
logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
# fn_set_local_logfile()
LOG_LEVEL = common.G_log_level

def parse_args():
    # Extract arguments from sys.argv
    args = {}
    for arg in sys.argv[1:]:
        if ':' in arg:
            key, value = arg.split(':', 1)  # Split only on the first ':'
            args[key.strip()] = value.strip()
        else:
            print(f"Warning: Argument '{arg}' does not contain a ':' separator.")
    return args

def fn_log_dataframe(df_p_source: pd.DataFrame, str_p_source_name: str) -> None:
    """
    Dataframe 로그 출력 조건 지정 함수
    :param df_p_source: 로그로 찍을 Dataframe
    :param str_p_source_name: 로그로 찍을 Dataframe 명
    :return: None
    """
    is_output = False
    if str_p_source_name.startswith('out_'):
        is_output = True

    if is_print:
        logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1)
        if is_local and not df_p_source.empty and flag_csv:
            # 로컬 Debugging 시 csv 파일 출력
            df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)
    else:
        # 최종 Output 테이블인 경우에는 무조건 로그 출력
        if is_output:
            logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1)
            if is_local and not df_p_source.empty:
                # 로컬 Debugging 시 csv 파일 출력
                df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)



def _decoration_(func):
    """
    1. 소스 내 함수 실행 시 반복되는 코드를 데코레이터로 변형하여 소스 라인을 줄일 수 있도록 함.
    2. 각 Step을 함수로 실행하는 경우 해당 함수에 뒤따르는 Step log 및 DF 로그, DF 로컬 출력을 데코레이터로 항상 출력하게 함.
    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        # 함수 시작 시각
        tm_start = time.time()
        # 함수 실행
        result = func(*args)
        # 함수 종료 시각
        tm_end = time.time()
        # 함수 실행 시간 로그
        logger.Note(p_note=f'[{func.__name__}] Total time is {tm_end - tm_start:.5f} sec.',
                    p_log_level=LOG_LEVEL.debug())
        # Step log 및 DF 로컬 출력 등을 위한 Keywords 변수 확인
        # Step No
        _step_no = kwargs.get('p_step_no')
        _step_desc = kwargs.get('p_step_desc')
        _df_name = kwargs.get('p_df_name')
        _warn_desc = kwargs.get('p_warn_desc')
        _exception_flag = kwargs.get('p_exception_flag')
        # Step log 관련 변수가 입력된 경우 Step log 출력
        if _step_no is not None and _step_desc is not None:
            logger.Step(p_step_no=_step_no, p_step_desc=_step_desc)
        # Warning 메시지가 있는 경우
        if _warn_desc is not None:
            # 함수 실행 결과가 DF이면서 해당 DF가 비어 있는 경우
            if type(result) == pd.DataFrame and result.empty:
                # Exception flag가 확인되고
                if _exception_flag is not None:
                    # Exception flag가 0이면 Warning 로그 출력, 1이면 Exception 발생시킴
                    if _exception_flag == 0:
                        logger.Note(p_note=_warn_desc, p_log_level=LOG_LEVEL.warning())
                    elif _exception_flag == 1:
                        raise Exception(_warn_desc)
        # DF 명이 있는 경우 로그 및 로컬 출력
        if _df_name is not None:
            fn_log_dataframe(result, _df_name)
        return result
    return wrapper


def fn_check_input_table(df_p_source: pd.DataFrame, str_p_source_name: str, str_p_cond: str) -> None:
    """
    Input Table을 체크한 결과를 로그 또는 Exception으로 표시한다.
    :param df_p_source: Input table
    :param str_p_source_name: Name of Input table
    :param str_p_cond: '0' - Exception, '1' - Warning Log
    :return: None
    """
    # Input Table 로그 출력
    logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1)

    if df_p_source.empty:
        if str_p_cond == '0':
            # 테이블이 비어 있는 경우 raise Exception
            raise Exception(f'[Exception] Input table({str_p_source_name}) is empty.')
        else:
            # 테이블이 비어 있는 경우 Warning log
            logger.Note(p_note=f'Input table({str_p_source_name}) is empty.', p_log_level=LOG_LEVEL.warning())


def fn_get_week(list_p_weeks: list, p_row: any) -> list:
    """
    in_Demand의 행과 Time.[Week] 목록을 받아 Time.[Week] - W Demand Build Ahead Limit<= t < Time.[Week]인 t의 목록을 찾아 리턴
    :param list_p_weeks:
    :param p_row:
    :return:
    """
    int_end = int(list_p_weeks.index(p_row['Time.[Week]']))
    int_start = int_end - int(p_row['W Demand Build Ahead Limit'])
    if int_start < 0:
        int_start = 0

    return list_p_weeks[int_start:int_end]

def fn_use_x_after_join(df_source: pd.DataFrame):
    """
    When join , there is 
    """
    df_source.columns = [col.replace('_x', '') if '_x' in col else col for col in df_source.columns]
    # Drop columns with '_y' suffix
    df_source.drop(columns=[col for col in df_source.columns if '_y' in col], inplace=True)
    # df_source = df_source.loc[:, ~df_source.columns.str.endswith('_y')]

def fn_use_y_after_join(df_source: pd.DataFrame):
    """
    When join , there is 
    """
    df_source.columns = [col.replace('_y', '') if '_y' in col else col for col in df_source.columns]
    # Drop columns with '_y' suffix
    df_source.drop(columns=[col for col in df_source.columns if '_x' in col], inplace=True)

# Remove '_x' and '_y' suffixes, keeping '_x' for specified columns
def customize_column_names(df_source: pd.DataFrame, column_use_y: list):
    # Replace '_y' with '' for columns not in column_use_y
    for col in df_source.columns:
        if '_y' in col:
            for col_y in column_use_y:
                if col_y in col:
                    df_source = df_source.rename(columns={col: col.replace('_y', '')})

    # Drop columns with '_x' suffix
    columns_x_to_drop = []
    for col in df_source.columns:
        if '_x' in col:
            for col_y in column_use_y:
                if col_y in col:
                    columns_x_to_drop.append(col)

    df_source.drop(columns=columns_x_to_drop, inplace=True)
    fn_use_x_after_join(df_source)


@_decoration_
def fn_set_header_in() -> pd.DataFrame:
    """
    MediumWeight로 실행 시 발생할 수 있는 Live Server에서의 오류를 방지하기 위해 Header만 있는 Output 테이블을 만든다.
    :return: DataFrame
        """
    df_return = pd.DataFrame()

    # out_Demand
    df_return = pd.DataFrame(
        {
            Version_Name: [], 
            Sales_Domain_ShipTo: [], 
            Item_Item: [], 
            Location_Location: [],
            Partial_Week: [] ,
            SIn_FCST_GC_LOCK : [],
            SIn_FCST_AP2_LOCK : [],
            SIn_FCST_AP1_LOCK : [],
            SIn_FCST_Color_Condition : []
        })

    return df_return

@_decoration_
def fn_set_header_out() -> pd.DataFrame:
    """
    MediumWeight로 실행 시 발생할 수 있는 Live Server에서의 오류를 방지하기 위해 Header만 있는 Output 테이블을 만든다.
    :return: DataFrame
        """
    df_return = pd.DataFrame()

    # out_Demand
    df_return = pd.DataFrame(
        {
            Version_Name: [], 
            Sales_Domain_ShipTo: [], 
            Item_Item: [], 
            Location_Location: [],
            Partial_Week: [] ,
            SOut_FCST_GC_LOCK : [],
            SOut_FCST_AP2_LOCK : [],
            SOut_FCST_AP1_LOCK : [],
            SOut_FCST_Color_Condition : []
        })

    return df_return


@_decoration_
def fn_make_week_list(df_p_source: pd.DataFrame) -> list:
    """
    전처리 - in_Time 테이블에서 Time.[Week]을 오름차순으로 정렬하여 리스트로 변환 후리턴
    :param df_p_source: in_Time
    :return: DataFrame
    """
    # 함수명
    str_my_name = inspect.stack()[0][3]
    
    # 입력 파라미터가 비어 있는 경우 비어 있는 DataFrame을 리턴
    if df_p_source.empty:
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(df_p_source)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    # 오름차순 정렬 후 'Time.[Week]'를 리스트로 변환
    list_return = df_p_source.sort_values(by='Time.[Week]')['Time.[Week]'].to_list()
    
    return list_return

def normalize_week(week_str):
    """Convert a week string with potential suffixes to an integer for comparison."""
    # Remove any non-digit characters (e.g., 'A' or 'B') and convert to integer
    return ''.join(filter(str.isdigit, week_str))

def is_within(current_week, start_week, end_week):
    """
    Check if the current week is within the range defined by start and end weeks.
    """
    return start_week <= current_week <= end_week

def fn_convert_type(df: pd.DataFrame, startWith: str, type):
    for column in df.columns:
        if column.startswith(startWith):
            df[column] = df[column].astype(type,errors='ignore')

@_decoration_
def fn_process_in_df_mst():

    if is_local: 
        # 로컬인 경우 Output 폴더를 정리한다.
        for file in os.scandir(str_output_dir):
            os.remove(file.path)

        # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
        file_pattern = f"{os.getcwd()}/{str_input_dir}/*.csv" 
        csv_files = glob.glob(file_pattern)

        file_to_df_mapping = {
            'df_in_Sales_Domain_Dimension.csv'          :      str_df_in_Sales_Domain    ,
            'df_in_Sales_Domain_Estore.csv'             :      str_df_in_Sales_Domain_Estore    ,
            'df_in_Time_Partial Week.csv'               :      str_df_in_Time_Partial_Week              ,
            'MST_ITEMCLASS.csv'                         :      str_df_in_Item_CLASS                 ,
            'MST_ITEMTAT.csv'                           :      str_df_in_Item_TAT                          ,
            'MST_MODELEOS.csv'                          :      str_df_in_MST_EOS                    ,
            'MST_MODELRTS.csv'                          :      str_df_in_MST_RTS                    ,
            'MST_SALESPRODUCT.csv'                      :      str_df_in_Sales_Product_ASN              ,
            'df_in_Forecast_Rule.csv'                   :      str_df_in_Forecast_Rule                  ,
            'VUI_ITEMATTB.csv'                          :      str_df_in_Item_Master                
        }

        def read_csv_with_fallback(filepath):
            encodings = ['utf-8-sig', 'utf-8', 'cp949']
            
            for enc in encodings:
                try:
                    return pd.read_csv(filepath, encoding=enc)
                except UnicodeDecodeError:
                    continue
            
            raise ValueError(f"Unable to read file {filepath} with tried encodings.")

        # Read all CSV files into a dictionary of DataFrames
        for file in csv_files:
            df = read_csv_with_fallback(file)
            file_name = file.split("/")[-1].split("\\")[-1].split(".")[0]
            # df['SourceFile'] = file_name
            # df.set_index('SourceFile',inplace=True)
            mapped = False
            for keyword, frame_name in file_to_df_mapping.items():
                if file_name.startswith(keyword.split('.')[0]):
                    input_dataframes[frame_name] = df
                    mapped = True
                    break
    else:
        # o9 에서 
        input_dataframes[str_df_in_Sales_Domain]            = df_in_Sales_Domain
        input_dataframes[str_df_in_Sales_Domain_Estore]     = df_in_Sales_Domain_Estore
        input_dataframes[str_df_in_Time_Partial_Week]       = df_in_Time_Partial_Week
        input_dataframes[str_df_in_Item_CLASS]              = df_in_Item_CLASS
        input_dataframes[str_df_in_Item_TAT]                = df_in_Item_TAT
        input_dataframes[str_df_in_MST_EOS]                 = df_in_MST_EOS
        input_dataframes[str_df_in_MST_RTS]                 = df_in_MST_RTS
        input_dataframes[str_df_in_Sales_Product_ASN]       = df_in_Sales_Product_ASN
        input_dataframes[str_df_in_Forecast_Rule]           = df_in_Forecast_Rule
        input_dataframes[str_df_in_Item_Master]             = df_in_Item_Master

    fn_convert_type(input_dataframes[str_df_in_MST_RTS], 'Sales Domain', str)
    fn_convert_type(input_dataframes[str_df_in_MST_EOS], 'Sales Domain', str)

    fn_convert_type(input_dataframes[str_df_in_Item_CLASS], 'Sales Domain', str)
    fn_convert_type(input_dataframes[str_df_in_Item_CLASS], 'Location', str)
    fn_convert_type(input_dataframes[str_df_in_Item_CLASS], 'Item', str)
    fn_convert_type(input_dataframes[str_df_in_Item_CLASS], 'ITEMCLASS', str)

    fn_convert_type(input_dataframes[str_df_in_Sales_Product_ASN], 'Sales Domain', str)
    fn_convert_type(input_dataframes[str_df_in_Sales_Product_ASN], 'Location', str)

    fn_convert_type(input_dataframes[str_df_in_Sales_Domain], 'Sales Domain', str)
    fn_convert_type(input_dataframes[str_df_in_Sales_Domain_Estore], 'Sales Domain', str)
    fn_convert_type(input_dataframes[str_df_in_Forecast_Rule], 'Sales Domain', str)

    input_dataframes[str_df_in_Forecast_Rule][FORECAST_RULE_GC_FCST].fillna(0, inplace=True)
    input_dataframes[str_df_in_Forecast_Rule][FORECAST_RULE_AP2_FCST].fillna(0, inplace=True)
    input_dataframes[str_df_in_Forecast_Rule][FORECAST_RULE_AP1_FCST].fillna(0, inplace=True)
    input_dataframes[str_df_in_Forecast_Rule][FORECAST_RULE_CUST].fillna(0, inplace=True)

    fn_convert_type(input_dataframes[str_df_in_Forecast_Rule], FORECAST_RULE_GC_FCST, 'int32')
    fn_convert_type(input_dataframes[str_df_in_Forecast_Rule], FORECAST_RULE_AP2_FCST, 'int32')
    fn_convert_type(input_dataframes[str_df_in_Forecast_Rule], FORECAST_RULE_AP1_FCST, 'int32')
    fn_convert_type(input_dataframes[str_df_in_Forecast_Rule], FORECAST_RULE_CUST, 'int32')



def analyze_by_rbql() :
    asn_df = input_dataframes[str_df_in_Sales_Product_ASN]
    master_df = input_dataframes[str_df_in_Item_Master]

    my_quey = f"""
        select 
            a['{Sales_Domain_ShipTo}'] as shipto,
            a['{Item_Item}'] as item,
            a['{Location_Location}'] as location,
            b['{Item_Type}'] as item_type,
            b['{Item_GBM}'] as item_gbm,
            b['{Product_Group}'] as product_group
        Join b on a['{Item_Item}'] == b['{Item_Item}']
        where b['{Item_Type}'] == 'BAS'

    """

    result = rbql.query_pandas_dataframe(
        query_text=my_quey,
        input_dataframe=asn_df,
        join_dataframe=master_df
    )


def analyze_by_duckdb():
    # Retrieve your DataFrames
    asn_df    = input_dataframes[str_df_in_Sales_Product_ASN]
    master_df = input_dataframes[str_df_in_Item_Master]
    dim_df   = input_dataframes[str_df_in_Sales_Domain]      

    # Register each DataFrame as a DuckDB table
    duckdb.register('asn_table', asn_df)
    duckdb.register('master_table', master_df)
    duckdb.register('dim_table', dim_df)

    # Build a SQL query referencing them by table aliases a (asn_table) and b (master_table)
    my_query = f"""
    SELECT
        a['{Sales_Domain_ShipTo}'] AS shipto,
        c['{Sales_Domain_LV2}'] AS DomainLv2,
        a['{Item_Item}']          AS item,
        a['{Location_Location}']  AS location,
        b['{Item_Type}']          AS item_type,
        b['{Item_GBM}']          AS item_gbm,
        b['{Product_Group}']      AS product_group
    FROM asn_table AS a
    JOIN master_table AS b
      ON a['{Item_Item}'] == b['{Item_Item}']
    JOIN dim_table AS c
        ON a['{Sales_Domain_ShipTo}'] == c['{Sales_Domain_ShipTo}']
    WHERE b['{Item_Type}']  = 'BAS'
    """

    # Execute the DuckDB query in-memory and fetch as a pandas DataFrame
    result_df = duckdb.query(my_query).to_df()

    return result_df


def analyze_by_duckdb_from_output():

    # from re import X
    # import os,sys,json,shutil,io,zipfile
    # import time
    # import datetime
    # import inspect
    # import traceback
    # import pandas as pd
    # from NSCMCommon import NSCMCommon as common
    # # from typing_extensions import Literal
    # import glob
    # import numpy as np
    # # import rbql
    # import duckdb

    v_base_dir  = "C:\workspace\Output\PYForecastMeasureLockColor_SHA_REF_20250410_14_08"
    v_output_dir  = f"{v_output_dir}/output"
    v_input_dir = f"{v_base_dir}/input"
    
    def read_csv_with_fallback(filepath):
        encodings = ['utf-8-sig', 'utf-8', 'cp949']
        
        for enc in encodings:
            try:
                return pd.read_csv(filepath, encoding=enc)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Unable to read file {filepath} with tried encodings.")

    input_pattern = f"{v_input_dir}/*.csv"
    input_csv_files = glob.glob(input_pattern)
    for file in input_csv_files:
        file_name = file.split("/")[-1].split("\\")[-1].split(".")[0]
        df = read_csv_with_fallback(file)
        duckdb.register(file_name, df)

    output_pattern = f"{v_output_dir}/*.csv"
    output_csv_files = glob.glob(output_pattern)
    for file in output_csv_files:
        file_name = file.split("/")[-1].split("\\")[-1].split(".")[0]
        df = read_csv_with_fallback(file)
        duckdb.register(file_name, df)

    # # Retrieve your DataFrames]
    # file = f"{v_output_dir}/input/df_in_Sales_Product_ASN"
    # df = read_csv_with_fallback(file)
    # duckdb.register(str_df_in_Sales_Product_ASN, df)
    my_query = f"""
    SELECT
        a['{Sales_Domain_ShipTo}'] AS shipto,
        c['{Sales_Domain_LV2}'] AS DomainLv2,
        a['{Item_Item}']          AS item,
        a['{Location_Location}']  AS location,
        b['{Item_Type}']          AS item_type,
        b['{Item_GBM}']          AS item_gbm,
        b['{Product_Group}']      AS product_group
    FROM asn_table AS a
    JOIN master_table AS b
      ON a['{Item_Item}'] == b['{Item_Item}']
    JOIN dim_table AS c
        ON a['{Sales_Domain_ShipTo}'] == c['{Sales_Domain_ShipTo}']
    WHERE b['{Item_Type}']  = 'BAS'
    """

    # Execute the DuckDB query in-memory and fetch as a pandas DataFrame
    result_df = duckdb.query(my_query).to_df()

    return result_df


@_decoration_
def fn_step01_join_rts_eos() -> pd.DataFrame:
    """
    Step 1. df_in_MST_RTS 와 df_in_MST_EOS 를 Item 과 ShipTo를 기준으로 inner Join
    """
    df_in_MST_RTS = input_dataframes.get(str_df_in_MST_RTS)
    df_filtered_rts_isvalid_y = df_in_MST_RTS[df_in_MST_RTS['RTS_ISVALID']=='Y']
    df_return = pd.merge(
        left=df_filtered_rts_isvalid_y,
        right=input_dataframes.get(str_df_in_MST_EOS),
        on=['Item.[Item]','Sales Domain.[Ship To]'],
        how='inner',
        suffixes=('_RTS','_EOS')
    )
    df_return = df_return.drop(columns=['Version.[Version Name]_RTS'])
    df_return = df_return.drop(columns=['Version.[Version Name]_EOS'])

    df_return['Item_Lv'] =  np.where(
        df_return['Sales Domain.[Ship To]'].str.startswith("3"),3,2
    )

    return df_return

# 1. Sanitize date string
def sanitize_date_string(x):
    if pd.isna(x):
        return ''
    x = str(x).strip()
    for token in ['PM', 'AM', '오전', '오후']:
        if token in x:
            x = x.split(token)[0].strip()
    return x[:10]  # Keep only 'YYYY/MM/DD'

v_sanitize_date_string = np.vectorize(sanitize_date_string)

# 2. Validate date
@np.vectorize
def is_valid_date(x):
    try:
        if pd.isna(x) or x == '':
            return True
        datetime.datetime.strptime(str(x), '%Y/%m/%d')
        return True
    except:
        return False

# 3. Convert to datetime
@np.vectorize
def safe_strptime(x):
    try:
        return datetime.datetime.strptime(str(x), '%Y/%m/%d') if pd.notna(x) and x != '' else None
    except:
        return None

# 4. Convert to partial week with error-checking
@np.vectorize
def to_partial_week(item,shipto,x):
    try:
        if x is not None and x != '':
            # If x is not already a Python datetime, try to convert it
            if not isinstance(x, datetime.datetime):
                # This conversion uses pandas to ensure we get a proper Python datetime
                x = pd.to_datetime(x).to_pydatetime()
            # Convert Python datetime to numpy.datetime64 with seconds precision
            np_dt = np.datetime64(x, 's')
            seconds = (np_dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
            dt_utc = datetime.datetime.utcfromtimestamp(seconds)
            return common.gfn_get_partial_week(dt_utc, True)
        else: 
            return ''
    except Exception as e:
        print("Error in to_partial_week with value:", item, shipto, x, "Error:", e)
        return ''

# 4.1 Convert to partial week with error-checking. use datttime
def to_partial_week_datetime(x):
    try:
        if x is not None and x != '':
            # If x is not already a Python datetime, try to convert it
            dt = datetime.datetime.strptime(str(x),'%Y/%m/%d')
            return common.gfn_get_partial_week(dt, True)
        else: 
            return ''
    except Exception as e:
        try:
            dt = datetime.datetime.strptime(str(x),'%Y-%m-%d')
            return common.gfn_get_partial_week(dt, True) 
        except Exception as e1:
            print("Error in to_partial_week with value:", x, "Error:", e1)
            return ''

def to_add_week(row):
    try:
        if x is not None and x != '':
            # If x is not already a Python datetime, try to convert it
            dt = common.gfn_add_week(x, -1)
            return common.gfn_get_partial_week(dt, True)
        else: 
            return ''
    except Exception as e:
        print("Error in to_partial_week with value:", item, shipto, x, "Error:", e)
        return ''

@np.vectorize
def is_valid_add_week(x):
    try:
        dt = common.gfn_add_week(x, -1)
        return True
    except:
        return False

@_decoration_
def fn_step02_convert_date_to_partial_week() -> pd.DataFrame:
    """
    Step 2 : Step1의 Result에 Time을 Partial Week 으로 변환
    """
    df_fn_RTS_EOS = output_dataframes[str_df_fn_RTS_EOS]

    columns_to_convert = [
        RTS_INIT_DATE,
        RTS_DEV_DATE,
        RTS_COM_DATE,
        EOS_INIT_DATE,
        EOS_CHG_DATE,
        EOS_COM_DATE
    ]

    # # Step 1: Sanitize date strings and filter invalid
    # for col in columns_to_convert:
    #     df_return[col] = v_sanitize_date_string(df_return[col])
    #     valid_mask = is_valid_date(df_return[col])
    #     df_return = df_return[valid_mask]

    # Step 2: Convert to datetime and partial week
    for col in columns_to_convert:
        df_fn_RTS_EOS[col] = v_sanitize_date_string(df_fn_RTS_EOS[col])
        # df_return[col] = safe_strptime(df_return[col])
        # df_return[col] = to_partial_week(df_return[Item_Item],df_return[Sales_Domain_ShipTo],df_return[col])
        # df_return[col] = to_partial_week_datetime(df_return[col])
        df_fn_RTS_EOS[col] = df_fn_RTS_EOS[col].apply(lambda x: to_partial_week_datetime(x) if pd.notna(x) else '')
        df_fn_RTS_EOS[col].astype(str)


    # Step 3: RTS_PARTIAL_WEEK
    df_fn_RTS_EOS[RTS_PARTIAL_WEEK] = np.where(
        df_fn_RTS_EOS[RTS_STATUS] == 'COM',
        df_fn_RTS_EOS[RTS_COM_DATE],
        np.where(
            df_fn_RTS_EOS[RTS_DEV_DATE].notna(),
            df_fn_RTS_EOS[RTS_DEV_DATE],
            df_fn_RTS_EOS[RTS_INIT_DATE]
        )
    )

    # Step 4: Item Level
    df_fn_RTS_EOS[Item_Lv] = np.where(
        df_fn_RTS_EOS[Sales_Domain_ShipTo].astype(str).str.startswith("3"),
        3,
        2
    )

    # Step 5: EOS_PARTIAL_WEEK
    df_fn_RTS_EOS[EOS_PARTIAL_WEEK] = np.where(
        df_fn_RTS_EOS[EOS_STATUS] == 'COM',
        np.where(
            df_fn_RTS_EOS[EOS_COM_DATE].notna(),
            df_fn_RTS_EOS[EOS_COM_DATE],
            np.where(
                df_fn_RTS_EOS[EOS_CHG_DATE].notna(),
                df_fn_RTS_EOS[EOS_CHG_DATE],
                df_fn_RTS_EOS[EOS_INIT_DATE]
            ),
        ),
        np.where(
            df_fn_RTS_EOS[EOS_CHG_DATE].notna(),
            df_fn_RTS_EOS[EOS_CHG_DATE],
            df_fn_RTS_EOS[EOS_INIT_DATE]
        )
    )

    # Step 6: Normalized week values
    df_fn_RTS_EOS[RTS_WEEK] = df_fn_RTS_EOS[RTS_PARTIAL_WEEK].astype(str).str.replace(r'\D', '', regex=True)
    df_fn_RTS_EOS[RTS_WEEK_MINUST_1] = df_fn_RTS_EOS[RTS_WEEK].apply(lambda x: common.gfn_add_week(x, -1) if pd.notna(x) and x != '' else '')
    df_fn_RTS_EOS[RTS_WEEK_PLUS_3] = df_fn_RTS_EOS[RTS_WEEK].apply(lambda x: common.gfn_add_week(x, 3) if pd.notna(x) and x != '' else '')
    df_fn_RTS_EOS[MAX_RTS_CURRENTWEEK] = df_fn_RTS_EOS[RTS_WEEK].apply(lambda x: max(x, current_week_normalized) if pd.notna(x) and x != '' else '')

    df_fn_RTS_EOS[EOS_WEEK] = df_fn_RTS_EOS[EOS_PARTIAL_WEEK].astype(str).str.replace(r'\D', '', regex=True)
    df_fn_RTS_EOS[EOS_WEEK_MINUS_1] = df_fn_RTS_EOS[EOS_WEEK].apply(lambda x: common.gfn_add_week(x, -1) if pd.notna(x) and x != '' else ''  )
    df_fn_RTS_EOS[EOS_WEEK_MINUS_4] = df_fn_RTS_EOS[EOS_WEEK].apply(lambda x: common.gfn_add_week(x, -4) if pd.notna(x) and x != '' else '')

    df_fn_RTS_EOS[RTS_INITIAL_WEEK] = df_fn_RTS_EOS[RTS_INIT_DATE].astype(str).str.replace(r'\D', '', regex=True)
    df_fn_RTS_EOS[EOS_INITIAL_WEEK] = df_fn_RTS_EOS[EOS_INIT_DATE].astype(str).str.replace(r'\D', '', regex=True)
    df_fn_RTS_EOS[MIN_EOSINI_MAXWEEK] = df_fn_RTS_EOS[EOS_INITIAL_WEEK].apply(lambda x: min(x, max_week_normalized) if pd.notna(x) and x != '' else '')
    df_fn_RTS_EOS[MIN_EOS_MAXWEEK] = df_fn_RTS_EOS[EOS_WEEK].apply(lambda x: min(x, max_week_normalized) if pd.notna(x) and x != '' else '')

    convert_to_int = [
        RTS_WEEK,
        RTS_WEEK_MINUST_1,
        RTS_WEEK_PLUS_3,
        MAX_RTS_CURRENTWEEK,
        EOS_WEEK,
        EOS_WEEK_MINUS_1,
        EOS_WEEK_MINUS_4,
        RTS_INITIAL_WEEK,
        EOS_INITIAL_WEEK,
        MIN_EOSINI_MAXWEEK
    ]

    for col in convert_to_int:
        df_fn_RTS_EOS[col] = df_fn_RTS_EOS[col].replace('','0')
        df_fn_RTS_EOS[col].fillna(0,inplace=True)
        df_fn_RTS_EOS[col] = df_fn_RTS_EOS[col].astype('int32')


    # # Step 6: Normalized week values (digit-only strings)
    # df_return['RTS_WEEK_NORMALIZED'] = df_return['RTS_PARTIAL_WEEK'].astype(str).str.replace(r'\D', '', regex=True)
    # df_return['EOS_WEEK_NORMALIZED'] = df_return['EOS_PARTIAL_WEEK'].astype(str).str.replace(r'\D', '', regex=True)# Use vectorized add_week
    # df_return['RTS_WEEK_NORMALIZED_MINUST_1'] = v_add_week(df_return['RTS_WEEK_NORMALIZED'], -1)
    # df_return['RTS_WEEK_NORMALIZED_PLUS_3'] = v_add_week(df_return['RTS_WEEK_NORMALIZED'], 3)

    # df_return['EOS_WEEK_NORMALIZED_MINUS_1'] = v_add_week(df_return['EOS_WEEK_NORMALIZED'], -1)
    # df_return['EOS_WEEK_NORMALIZED_MINUS_4'] = v_add_week(df_return['EOS_WEEK_NORMALIZED'], -4)

    # # Use vectorized max
    # df_return['MAX_RTS_CURRENTWEEK'] = v_max_week(df_return['RTS_WEEK_NORMALIZED'])

    # # Normalize initial week values
    # df_return['RTS_INITIAL_WEEK_NORMALIZED'] = df_return['RTS_INIT_DATE'].astype(str).str.replace(r'\D', '', regex=True)
    # df_return['EOS_INITIAL_WEEK_NORMALIZED'] = df_return['EOS_INIT_DATE'].astype(str).str.replace(r'\D', '', regex=True)

    # Use np.minimum for min
    # df_return['MIN_EOSINI_MAXWEEK'] = np.minimum(df_return['EOS_INITIAL_WEEK_NORMALIZED'], max_week_normalized)


    return df_fn_RTS_EOS


@_decoration_
def fn_step03_join_rts_eos() -> pd.DataFrame:
    """
    Step 3. df_in_MST_RTS 와 df_in_MST_EOS의 ITEM * ShipTo로 Inner Join하여 새로운 DF 생성 ( Output Data )
        첫번째 생성된 DF 에서 컬럼을 복사
    """
    fn_step03_join_rts_eos = output_dataframes[str_df_fn_RTS_EOS]
    df_origin = fn_step03_join_rts_eos.copy(deep=True)
    df_return = df_origin[[Item_Item,Sales_Domain_ShipTo]]

    return df_return


@_decoration_
def fn_step04_add_partialweek_measurecolumn() -> pd.DataFrame:
    """
    Step 4  : Step3의 df에 Partial Week 및 Measure Column 추가
    """
    df_fn_RTS_EOS =  output_dataframes[str_df_fn_RTS_EOS]
    df_in_Time_Partial_Week = input_dataframes[str_df_in_Time_Partial_Week]
    df_fn_RTS_EOS['key'] = 1
    df_in_Time_Partial_Week['key'] = 1

    # Perform the merge on the temporary key
    df_fn_RTS_EOS_Week = pd.merge(
        df_fn_RTS_EOS[[
            Item_Item,
            Sales_Domain_ShipTo,
            'key'
        ]],
        df_in_Time_Partial_Week,
        on='key'
    )
    df_fn_RTS_EOS_Week[Item_Lv] = df_fn_RTS_EOS_Week[Sales_Domain_ShipTo].apply(lambda x: 3 if x.startswith("3") else 2)
    df_fn_RTS_EOS_Week[CURRENT_ROW_WEEK] = df_fn_RTS_EOS_Week[Partial_Week].apply(normalize_week)
    df_fn_RTS_EOS_Week[CURRENT_ROW_WEEK_PLUS_8] = df_fn_RTS_EOS_Week[CURRENT_ROW_WEEK].apply(lambda x : common.gfn_add_week(x, 8))
    
    df_fn_RTS_EOS_Week[SIn_FCST_GC_LOCK] = True
    df_fn_RTS_EOS_Week[SIn_FCST_Color_Condition] = '19_GRAY'
    df_fn_RTS_EOS_Week = df_fn_RTS_EOS_Week.drop(columns=['key']).reset_index(drop=True)

    # df_fn_RTS_EOS.drop(columns=['key'],inplace=True).reset_index(inplace=True)
    df_in_Time_Partial_Week.drop(columns=['key'],inplace=True)
    df_in_Time_Partial_Week.reset_index(inplace=True)


    # expanded_rows = []
    # for index, row in df_03_joined_rts_eos.iterrows():
    #     for time_value in input_dataframes[str_df_in_Time_Partial_Week]['Time.[Partial Week]']:
    #         new_row = row.to_dict()
    #         new_row['Time.[Partial Week]'] = time_value
    #         new_row['S/In FCST(GI)_GC.Lock'] = 'True'  # Placeholder value, replace with actual logic if needed
    #         new_row['S/In FCST Color Condition'] = 'GRAY'
    #         expanded_rows.append(new_row)
    
    
    # df_return = pd.DataFrame(expanded_rows)

    convert_to_int = [
        CURRENT_ROW_WEEK,
        CURRENT_ROW_WEEK_PLUS_8
    ]

    for col in convert_to_int:
        df_fn_RTS_EOS_Week[col] = df_fn_RTS_EOS_Week[col].replace('','0')
        df_fn_RTS_EOS_Week[col].fillna(0,inplace=True)
        df_fn_RTS_EOS_Week[col] = df_fn_RTS_EOS_Week[col].astype('int32')

    return df_fn_RTS_EOS_Week



@_decoration_
def fn_step05_set_lock_values():
    """
    Step 05: Forecast Measure Lock Color using array-based lookup
    
    This function relies on the following global variables:
      - output_dataframes (dict)
      - current_week_normalized (str)
      - max_week_normalized (str)

    Process:
      1. Load df_fn_RTS_EOS_Week from output_dataframes['df_04_partialweek_measurecolumn'].
      2. Load df_fn_RTS_EOS (the lookup DataFrame) from output_dataframes['df_02_date_to_partial_week'].
      3. Set up an index on df_fn_RTS_EOS for quick array-based lookups.
      4. For each needed column (e.g. 'RTS_WEEK_NORMALIZED_MINUST_1'), create
         a NumPy array that aligns with df_fn_RTS_EOS_Week's rows.
      5. Build your condition arrays in a vectorized way, and apply color/lock
         updates in df_fn_RTS_EOS_Week directly.
    
    Returns:
      pd.DataFrame: updated df_fn_RTS_EOS_Week with 'SIn_FCST(GI)_GC.Lock'
                    and 'SIn_FCST Color Condition' set.
    """

    # -----------------------------
    # 1. Load the partial-week DataFrame
    # -----------------------------
    df_week = output_dataframes[str_df_fn_RTS_EOS_Week]

    # Keep only the columns needed for logic (plus the GC columns we want to set)
    needed_cols_week = [
        Sales_Domain_ShipTo,
        Item_Item,
        Item_Lv,
        CURRENT_ROW_WEEK,
        CURRENT_ROW_WEEK_PLUS_8,
        SIn_FCST_GC_LOCK,
        SIn_FCST_Color_Condition
    ]
    # df_week = df_week[needed_cols_week]

    # Convert CURRENT_ROW_WEEK to int for numeric comparisons
    df_week[CURRENT_ROW_WEEK] = (
        df_week[CURRENT_ROW_WEEK]
        .fillna('0')
        .astype(int)
    )

    # -----------------------------
    # 2. Load the lookup DataFrame (df_fn_RTS_EOS)
    # -----------------------------
    df_lookup = output_dataframes[str_df_fn_RTS_EOS]
    # The column names in your lookup can remain raw strings,
    # or you can define more constants if desired. Here we keep
    # them as strings or minimal new constants:
    needed_cols_lookup = [
        Sales_Domain_ShipTo,
        Item_Item,
        Item_Lv,
        MAX_RTS_CURRENTWEEK,
        RTS_WEEK,
        MIN_EOSINI_MAXWEEK,
        EOS_WEEK,
        EOS_WEEK_MINUS_4,
        RTS_WEEK_MINUST_1,
        RTS_WEEK_PLUS_3
    ]
    # df_lookup = df_lookup[needed_cols_lookup]

    # -----------------------------
    # 3. Set the index in df_lookup for array-based lookups
    # -----------------------------
    df_lookup.set_index([Sales_Domain_ShipTo, Item_Item, Item_Lv], inplace=True)

    # We'll match on (Sales_Domain_Ship_To, Item_Item, Item_lv) from df_week
    multi_index_week = pd.MultiIndex.from_arrays(
        [
            df_week[Sales_Domain_ShipTo],
            df_week[Item_Item],
            df_week[Item_Lv]
        ],
        names=[Sales_Domain_ShipTo, Item_Item, Item_Lv]
    )

    # positions: integer array the same length as df_week
    positions = df_lookup.index.get_indexer(multi_index_week)
    valid_mask = (positions != -1)

    # -----------------------------
    # 4. Build arrays from df_lookup columns
    # -----------------------------
    max_rts_array           = pd.to_numeric(df_lookup[MAX_RTS_CURRENTWEEK], errors='coerce').fillna(0).astype(int)
    rts_week_array          = pd.to_numeric(df_lookup[RTS_WEEK], errors='coerce').fillna(0).astype(int)
    rts_init_array          = pd.to_numeric(df_lookup[RTS_INITIAL_WEEK], errors='coerce').fillna(0).astype(int)
    min_eos_maxweek_arr     = pd.to_numeric(df_lookup[MIN_EOS_MAXWEEK], errors='coerce').fillna(0).astype(int)
    eos_week_array          = pd.to_numeric(df_lookup[EOS_WEEK], errors='coerce').fillna(0).astype(int)
    eos_week_minus1_array   = pd.to_numeric(df_lookup[EOS_WEEK_MINUS_1], errors='coerce').fillna(0).astype(int)
    eos_week_minus4_array   = pd.to_numeric(df_lookup[EOS_WEEK_MINUS_4], errors='coerce').fillna(0).astype(int)
    rts_week_minus1_array   = pd.to_numeric(df_lookup[RTS_WEEK_MINUST_1], errors='coerce').fillna(0).astype(int)
    rts_week_plus3_array    = pd.to_numeric(df_lookup[RTS_WEEK_PLUS_3], errors='coerce').fillna(0).astype(int)

    n = len(df_week)
    arr_max_rts            = np.full(n, np.nan, dtype=int)
    arr_rts_week           = np.full(n, np.nan, dtype=int)
    arr_rts_init           = np.full(n, np.nan, dtype=int)
    arr_min_eos_maxweek = np.full(n, np.nan, dtype=int)
    arr_eos_week           = np.full(n, np.nan, dtype=int)
    arr_eos_week_minus1    = np.full(n, np.nan, dtype=int)
    arr_eos_week_minus4    = np.full(n, np.nan, dtype=int)
    arr_rts_week_minus1    = np.full(n, np.nan, dtype=int)
    arr_rts_week_plus3     = np.full(n, np.nan, dtype=int)

    # Fill valid rows
    arr_max_rts[valid_mask]            = max_rts_array[positions[valid_mask]]
    arr_rts_week[valid_mask]           = rts_week_array[positions[valid_mask]]
    arr_rts_init[valid_mask]            = rts_init_array[positions[valid_mask]]
    arr_min_eos_maxweek[valid_mask] = min_eos_maxweek_arr[positions[valid_mask]]
    arr_eos_week[valid_mask]           = eos_week_array[positions[valid_mask]]
    arr_eos_week_minus1[valid_mask]    = eos_week_minus1_array[positions[valid_mask]]
    arr_eos_week_minus4[valid_mask]    = eos_week_minus4_array[positions[valid_mask]]
    arr_rts_week_minus1[valid_mask]    = rts_week_minus1_array[positions[valid_mask]]
    arr_rts_week_plus3[valid_mask]     = rts_week_plus3_array[positions[valid_mask]]

    # -----------------------------
    # 5. Vectorized condition checks
    # -----------------------------
    row_partial = df_week[CURRENT_ROW_WEEK].to_numpy(dtype=int)
    curr_week_int = int(current_week_normalized)
    max_week_int  = int(max_week_normalized)

    # Step 5-1: White color
    cond_white = (
        (row_partial >= curr_week_int) &
        (arr_max_rts <= row_partial) &
        (row_partial <= arr_min_eos_maxweek)
    )
    df_week.loc[cond_white, SIn_FCST_GC_LOCK] = False
    df_week.loc[cond_white, SIn_FCST_Color_Condition] = '14_WHITE'

    # Step 5-2: Dark Blue
    cond_darkblue = (
        (row_partial >= curr_week_int) &
        (arr_rts_init <= row_partial) &
        (row_partial <= arr_rts_week_minus1)
    )
    df_week.loc[cond_darkblue, SIn_FCST_GC_LOCK] = True
    df_week.loc[cond_darkblue, SIn_FCST_Color_Condition] = '15_DARKBLUE'

    # Step 5-3: Light Blue
    cond_lightblue = (
        (row_partial >= curr_week_int) &
        (arr_rts_week <= row_partial) &
        (row_partial <= arr_rts_week_plus3)
    )
    df_week.loc[cond_lightblue, SIn_FCST_GC_LOCK] = False
    df_week.loc[cond_lightblue, SIn_FCST_Color_Condition] = '10_LIGHTBLUE'

    # Step 5-4: Light Red
    cond_lightred = (
        (row_partial >= curr_week_int) &
        (arr_eos_week_minus4 <= row_partial) &
        (row_partial <= (arr_eos_week_minus1))
    )
    df_week.loc[cond_lightred, SIn_FCST_GC_LOCK] = False
    df_week.loc[cond_lightred, SIn_FCST_Color_Condition] = '11_LIGHTRED'

    # Step 5-5: Dark Red
    cond_darkred = (
        (row_partial >= curr_week_int) &
        (arr_eos_week <= row_partial) &
        (row_partial <= max_week_int)
    )
    df_week.loc[cond_darkred, SIn_FCST_GC_LOCK] = True
    df_week.loc[cond_darkred, SIn_FCST_Color_Condition] = '16_DARKRED'

    # -----------------------------
    # 6. Return
    # -----------------------------
    # return df_week.reset_index(drop=True)
    return df_week

@_decoration_
def fn_step06_addcolumn_green_for_wireless_bas_array_based():
    """
    Step 6 (array-based): Add or update 'S/In FCST Color Condition' to '13_GREEN'
    for 'wireless BAS' items, using pre-existing CURRENT_ROW_WEEK_PLUS_8
    to avoid calling gfn_add_week. No merges required.
    """

    # ------------------------------------------------------------------
    # 1) Load the main DataFrame we want to update, e.g. df_week
    #    Suppose it has these columns:
    #       CURRENT_ROW_WEEK          (int)
    #       CURRENT_ROW_WEEK_PLUS_8   (int)
    #       'S/In FCST Color Condition'
    #       'Item.[Item]'  -> We'll use array-based get_indexer with df_in_Item_Master
    # ------------------------------------------------------------------
    df_week = output_dataframes[str_df_fn_RTS_EOS_Week]  # or 'df_04_partialweek_measurecolumn', etc.

    # Ensure CURRENT_ROW_WEEK is int
    df_week[CURRENT_ROW_WEEK] = df_week[CURRENT_ROW_WEEK].fillna('0').astype(int)
    df_week[CURRENT_ROW_WEEK_PLUS_8] = df_week[CURRENT_ROW_WEEK_PLUS_8].fillna('0').astype(int)

    # ------------------------------------------------------------------
    # 2) Load & index your Item Master for array-based lookups
    # ------------------------------------------------------------------
    df_item_master = input_dataframes[str_df_in_Item_Master]

    # Example column references (adapt to your code):
    #   'Item.[Item]' -> Item_Item
    #   'Item.[Item Type]' -> Item_Type
    #   'Item.[Item GBM]'  -> Item_GBM
    #
    # Make Item_Item the index so we can do get_indexer lookups
    df_item_master.set_index(Item_Item, inplace=True)

    # We want to look up each row's item in df_week
    item_array_in_week = df_week[Item_Item].to_numpy()

    # get_indexer returns the positions in df_item_master for each item in item_array_in_week
    positions = df_item_master.index.get_indexer(item_array_in_week)
    valid_mask = (positions != -1)

    # ------------------------------------------------------------------
    # 3) Create arrays for item attributes from df_item_master
    # ------------------------------------------------------------------
    item_type_array = df_item_master[Item_Type].to_numpy(dtype=str)
    item_gbm_array  = df_item_master[Item_GBM].to_numpy(dtype=str)
    item_group_array  = df_item_master[Product_Group].to_numpy(dtype=str)

    n = len(df_week)
    out_item_type = np.full(n, '', dtype=object)
    out_item_gbm  = np.full(n, '', dtype=object)
    out_item_group  = np.full(n, '', dtype=object)

    out_item_type[valid_mask] = item_type_array[positions[valid_mask]]
    out_item_gbm[valid_mask]  = item_gbm_array[positions[valid_mask]]
    out_item_group[valid_mask]  = item_group_array[positions[valid_mask]]

    # # Optionally store them back to df_week
    # df_week[Item_Type] = out_item_type
    df_week[Item_GBM]  = out_item_gbm
    df_week[Product_Group]  = out_item_group

    # ------------------------------------------------------------------
    # 4) Build your mask using CURRENT_ROW_WEEK and CURRENT_ROW_WEEK_PLUS_8
    # ------------------------------------------------------------------
    row_partial_array = df_week[CURRENT_ROW_WEEK].to_numpy()
    row_partial_plus8_array = df_week[CURRENT_ROW_WEEK_PLUS_8].to_numpy()

    # Let's say "green" range is from current_week_normalized up to CURRENT_ROW_WEEK_PLUS_8
    # (adjust logic as needed)
    curr_week_int = int(current_week_normalized)

    mask_wireless_bas = (
        (out_item_type == 'BAS') &
        (out_item_gbm  == 'MOBILE') &
        (row_partial_array >= curr_week_int) &
        (row_partial_array <= row_partial_plus8_array) &
        (df_week[SIn_FCST_Color_Condition] != '19_GRAY')
    )

    # ------------------------------------------------------------------
    # 5) Apply the color update in-place
    # ------------------------------------------------------------------
    df_week.loc[mask_wireless_bas, SIn_FCST_Color_Condition] = '13_GREEN'

    # If you no longer need the item type columns, you can drop them
    # df_week.drop(columns=['Item.[Item Type]', 'Item.[Item GBM]'], inplace=True)

    # ------------------------------------------------------------------
    # 6) Return the updated DataFrame
    # ------------------------------------------------------------------
    df_item_master.reset_index(inplace=True)

    column_returns = [
        Sales_Domain_ShipTo ,
        Item_Item ,
        Item_Lv ,
        Item_GBM ,
        Product_Group ,
        Partial_Week ,
        CURRENT_ROW_WEEK ,
        SIn_FCST_GC_LOCK ,
        SIn_FCST_Color_Condition
    ]

    return df_week[column_returns]


@_decoration_
def fn_step07_join_sales_product_asn_to_lvl_by_merge() -> pd.DataFrame:
    """
    Step 7: Join df_in_Sales_Product_ASN with df_in_Sales_Domain_Dimension on specified keys
    and reorder columns. Add 'Item.[Product Group]' by merging with df_in_Item_Master.
    """

    df_in_Sales_Product_ASN = input_dataframes[str_df_in_Sales_Product_ASN]
    df_in_Sales_Domain_Dimension = input_dataframes[str_df_in_Sales_Domain]
    df_in_Item_Master = input_dataframes[str_df_in_Item_Master]

    # Merge df_in_Item_Master to get 'Item.[Product Group]'
    df_fn_Sales_Product_ASN_Item = df_in_Sales_Product_ASN.merge(
        df_in_Item_Master[[Item_Item, Item_GBM, Product_Group]],
        on=Item_Item,
        how='inner'
    )

    df_fn_Sales_Product_ASN_Item = df_fn_Sales_Product_ASN_Item.merge(
        df_in_Sales_Domain_Dimension[[
            Sales_Domain_LV2, 
            Sales_Domain_LV3,
            Sales_Domain_LV6,
            Sales_Domain_ShipTo]],
        on=Sales_Domain_ShipTo,
        how='inner'
    )

    return_columns = [
        Sales_Domain_LV2, 
        Sales_Domain_LV3,
        Sales_Domain_LV6,
        Sales_Domain_ShipTo,
        Item_Item,
        Item_GBM,
        Product_Group,
        Location_Location,
        Salse_Product_ASN
    ]
    return df_fn_Sales_Product_ASN_Item[return_columns]


@_decoration_
def fn_step08_create_asn_item_week_and_match_locks(df_fn_Sales_Product_ASN_Item: pd.DataFrame) -> pd.DataFrame:
    """
    Step 08 (revised):
      1) Load df_rts_eos_week and df_in_Time_Partial_Week
      2) Cross-join df_fn_Sales_Product_ASN_Item with df_in_Time_Partial_Week => df_fn_Sales_Product_ASN_Item_Week
      3) Initialize (SIn_FCST_GC_LOCK=False, SIn_FCST_Color_Condition='14_WHITE')
      4) For each row (which is at level-7 domain),
         find level-2, level-3 from df_in_Sales_Domain_Dimension.
      5) Use array-based lookup in df_rts_eos_week (which is at level-2 or 3)
         to find matching row for either lv2 or lv3 domain + (Item_GBM, Product_Group, Item_Item, Partial_Week).
      6) If found, copy SIn_FCST_GC_LOCK and SIn_FCST_Color_Condition from df_rts_eos_week into the row.

    Returns:
      df_fn_Sales_Product_ASN_Item_Week with updated SIn_FCST_GC_LOCK / SIn_FCST_Color_Condition
    """

    # ---------------------------------------------------------
    # 1) Load the needed data
    # ---------------------------------------------------------
    df_rts_eos_week = output_dataframes["df_fn_RTS_EOS_Week"]
    df_in_time_partial_week = input_dataframes[str_df_in_Time_Partial_Week]
    df_dim = input_dataframes[str_df_in_Sales_Domain]  # for mapping level-7 => (lv2, lv3)

    # ---------------------------------------------------------
    # 2) Create df_fn_Sales_Product_ASN_Item_Week by cross joining partial weeks
    # ---------------------------------------------------------
    df_main = df_fn_Sales_Product_ASN_Item.copy()
    df_main['join_key'] = 1

    df_partial = df_in_time_partial_week.copy()
    df_partial['join_key'] = 1

    df_fn_Sales_Product_ASN_Item_Week = pd.merge(
        df_main,
        df_partial,
        on='join_key'
    ).drop(columns=['join_key'])

    df_main.drop(columns=['join_key'],inplace=True)
    df_main.reset_index(inplace=True)

    df_partial.drop(columns=['join_key'],inplace=True)
    df_partial.reset_index(inplace=True)


    # ---------------------------------------------------------
    # 3) Initialize lock/color
    # ---------------------------------------------------------
    df_fn_Sales_Product_ASN_Item_Week[SIn_FCST_GC_LOCK] = False
    df_fn_Sales_Product_ASN_Item_Week[SIn_FCST_Color_Condition] = '14_WHITE'
    df_fn_Sales_Product_ASN_Item_Week[RTS_EOS_ShipTo] = ''

    # ---------------------------------------------------------
    # 4) Map domain from level-7 => (lv2, lv3)
    # ---------------------------------------------------------
    # df_dim maps: level-7 domain => columns [Sales_Domain_LV2, Sales_Domain_LV3, ...]
    # We'll do an array-based approach

    # 4a) index the dimension by level-7 domain
    dim_index = df_dim.set_index([Sales_Domain_ShipTo])
    arr_lv2_full = dim_index[Sales_Domain_LV2].to_numpy(dtype=object)
    arr_lv3_full = dim_index[Sales_Domain_LV3].to_numpy(dtype=object)

    # 4b) For each row, we find dimension row via .get_indexer
    domain7_array = df_fn_Sales_Product_ASN_Item_Week[Sales_Domain_ShipTo].to_numpy(dtype=object)
    pos_dim = dim_index.index.get_indexer(domain7_array)
    valid_dim = (pos_dim != -1)

    lv2_array = np.full(len(df_fn_Sales_Product_ASN_Item_Week), None, dtype=object)
    lv3_array = np.full(len(df_fn_Sales_Product_ASN_Item_Week), None, dtype=object)

    lv2_array[valid_dim] = arr_lv2_full[pos_dim[valid_dim]]
    lv3_array[valid_dim] = arr_lv3_full[pos_dim[valid_dim]]

    # ---------------------------------------------------------
    # 5) Build an index for df_rts_eos_week to find lock/color
    # ---------------------------------------------------------
    # We'll assume df_rts_eos_week has columns like:
    #   [Item_GBM, Product_Group, Item_Item, Sales_Domain_ShipTo, Partial_Week, SIn_FCST_GC_LOCK, SIn_FCST_Color_Condition]
    # and 'Sales_Domain_ShipTo' is at level-2 or 3

    # index the rts table
    df_rts_index = df_rts_eos_week.set_index([
        Item_GBM,
        Product_Group,
        Item_Item,
        Sales_Domain_ShipTo,
        Partial_Week
    ])

    lock_full  = df_rts_index[SIn_FCST_GC_LOCK].to_numpy()
    color_full = df_rts_index[SIn_FCST_Color_Condition].to_numpy()

    # We'll retrieve columns from df_fn_Sales_Product_ASN_Item_Week to match
    gbm_arr   = df_fn_Sales_Product_ASN_Item_Week[Item_GBM].to_numpy(dtype=object)
    pg_arr    = df_fn_Sales_Product_ASN_Item_Week[Product_Group].to_numpy(dtype=object)
    item_arr  = df_fn_Sales_Product_ASN_Item_Week[Item_Item].to_numpy(dtype=object)
    part_arr  = df_fn_Sales_Product_ASN_Item_Week[Partial_Week].to_numpy(dtype=object)

    # We'll store updated lock/color in arrays, then assign back
    updated_lock  = df_fn_Sales_Product_ASN_Item_Week[SIn_FCST_GC_LOCK].to_numpy(dtype=bool)
    updated_color = df_fn_Sales_Product_ASN_Item_Week[SIn_FCST_Color_Condition].to_numpy(dtype=object)
    updated_rts_eos_shipto = df_fn_Sales_Product_ASN_Item_Week[RTS_EOS_ShipTo].to_numpy(dtype=object)

    def do_lookup_and_update(domain_array, priority_mask):
        """
        domain_array: array of level-2 or level-3 domains
        priority_mask: we only update rows where the lock/color hasn't been found yet 
                       (or if we want lv2 to override previous, do the logic differently).
        """
        # Build a multiindex to do get_indexer
        positions = df_rts_index.index.get_indexer(
            pd.MultiIndex.from_arrays([
                gbm_arr,
                pg_arr,
                item_arr,
                domain_array,
                part_arr
            ])
        )
        valid = (positions != -1) & priority_mask
        # copy lock/color
        updated_lock[valid]  = lock_full[positions[valid]]
        updated_color[valid] = color_full[positions[valid]]
        updated_rts_eos_shipto[valid] = domain_array[valid]

    # 6) Attempt match with LV2 domain first
    # all rows are initially "False" lock => if you want to fill them, define a mask
    mask_not_matched = np.full(len(df_fn_Sales_Product_ASN_Item_Week), True, dtype=bool)
    do_lookup_and_update(lv2_array, mask_not_matched)

    # Next, attempt match with LV3 domain only on rows that remain "not matched" (if lv2 didn't succeed).
    # We'll consider any row that STILL has lock=False to be unmatched
    mask_not_matched = (updated_lock == False)
    do_lookup_and_update(lv3_array, mask_not_matched)

    # store arrays back
    df_fn_Sales_Product_ASN_Item_Week[SIn_FCST_GC_LOCK] = updated_lock
    df_fn_Sales_Product_ASN_Item_Week[SIn_FCST_Color_Condition] = updated_color
    df_fn_Sales_Product_ASN_Item_Week[RTS_EOS_ShipTo] = updated_rts_eos_shipto

    return df_fn_Sales_Product_ASN_Item_Week


@_decoration_
def fn_step09_filter_itemclass_to_yellow_loc(df_fn_Sales_Product_ASN_Item_Week: pd.DataFrame) -> pd.DataFrame:
    """
    Step 09 (array-based approach):
      1. Determine if (Item_Item, ShipTo, Location_Location) 
         from df_main (df_08) has a matching row in df_in_Item_CLASS. 
         We do NOT use 'Item_Class' column—just the existence of a match.
      2. Also do partial-week logic lookups from df_fn_RTS_EOS or df_fn_RTS_EOS_Week 
         if that is part of your condition for setting color to 'YELLOW'.
      3. If condition is met (found in item_class + partial-week logic), 
         set 'SIn_FCST(GI)_GC_LOCK' = False, 
         'SIn_FCST(GI)_Color_Condition' = 'YELLOW'.

    :param df_08: DataFrame (df_fn_Sales_ProductASN_Item_Week) 
                  e.g. columns: 'Item_Item', 'Sales_Domain_Ship_To' (maybe lv7 or lv2/3?), 
                  'Location_Location', 'Partial_Week',
                  'SIn_FCST(GI)_GC_LOCK', 'SIn_FCST(GI)_Color_Condition', etc.

    :return: updated df_08 with color = 'YELLOW' for matching conditions
    """

    df_main = df_fn_Sales_Product_ASN_Item_Week
    # -------------------------------------------------------
    # A) Find level-6 domain from df_in_Sales_Domain_Dimension
    #    for each row in df_main (which has level-7 domain).
    # -------------------------------------------------------
    # df_main has 'Sales_Domain_ShipTo' at level-7
    # We'll do an array-based approach:
    df_dim = input_dataframes[str_df_in_Sales_Domain]

    # Index the dimension by its level-7 domain => retrieve 'Sales_Domain_LV6'
    dim_index = df_dim.set_index([Sales_Domain_ShipTo])  # we assume this is level-7 domain
    lv6_array_full = dim_index[Sales_Domain_LV6].to_numpy(dtype=object)

    # For each row in df_main, get the dimension row
    domain7_array = df_main[Sales_Domain_ShipTo].to_numpy(dtype=object)
    positions_dim = dim_index.index.get_indexer(domain7_array)
    valid_mask_dim = (positions_dim != -1)

    # We'll build an array of lv6 for each row in df_main
    lv6_array = np.full(len(df_main), None, dtype=object)
    lv6_array[valid_mask_dim] = lv6_array_full[positions_dim[valid_mask_dim]]

    # -------------------------------------------------------
    # 1) Identify which rows have a match in df_in_Item_CLASS
    # -------------------------------------------------------
    # df_in_Item_CLASS has columns: 
    #   Item_Item, Sales_Domain_Ship_To, Location_Location
    # We'll interpret “ShipTo” in df_in_Item_CLASS as the same domain as df_main's 'Sales_Domain_Ship_To' 
    # (You might do lv6 or lv7, adapt if needed.)

    df_iclass = input_dataframes[str_df_in_Item_CLASS]
    # Create an index on (Item_Item, Sales_Domain_Ship_To, Location_Location)
    iclass_index = df_iclass.set_index([Item_Item,Sales_Domain_ShipTo,Location_Location])
    # No 'Item_Class' usage


    # We'll do get_indexer against the same triple from df_main
    # Convert df_main columns to arrays
    item_array  = df_main[Item_Item].to_numpy(dtype=object)
    # shipto_array = df_main[Sales_Domain_ShipTo].to_numpy(dtype=object)
    loc_array  = df_main[Location_Location].to_numpy(dtype=object)

    # Build a MultiIndex for df_main
    multi_idx_iclass = pd.MultiIndex.from_arrays(
        [item_array, lv6_array, loc_array],
        names=[Item_Item,Sales_Domain_LV6,Location_Location]
    )

    # positions_iclass: which row in df_iclass matches each row in df_main
    positions_iclass = iclass_index.index.get_indexer(multi_idx_iclass)
    valid_mask_iclass = (positions_iclass != -1)
    # If True => there's a row in df_in_Item_CLASS for that (item, shipto, location)

    # -------------------------------------------------------
    # 2) Do partial-week logic from df_fn_RTS_EOS or df_fn_RTS_EOS_Week
    #    if needed for YELLOW condition
    # -------------------------------------------------------
    df_fn_RTS_EOS = output_dataframes[str_df_fn_RTS_EOS]
    df_fn_RTS_EOS_Week = output_dataframes[str_df_fn_RTS_EOS_Week]

    # For example, we'll index df_fn_RTS_EOS by (Item_Item, ShipTo) 
    # to get 'EOS_WEEK_NORMALIZED', 'EOS_WEEK_NORMALIZED_MINUS_1'
    df_fn_RTS_EOS.reset_index(inplace=True)
    lookup_rts_eos = df_fn_RTS_EOS.set_index([Item_Item,Sales_Domain_ShipTo])
    positions_rts_eos = lookup_rts_eos.index.get_indexer(
        pd.MultiIndex.from_arrays([
            df_main[Item_Item],
            df_main[RTS_EOS_ShipTo],  # adapt to your domain column name
        ])
    )
    valid_mask_rts_eos = (positions_rts_eos != -1)
    # Convert to arrays
    rts_week_array = lookup_rts_eos[RTS_WEEK].to_numpy(dtype=str)
    eos_week_array = lookup_rts_eos[EOS_WEEK].to_numpy(dtype=str)
    eos_minus1_array = lookup_rts_eos[EOS_WEEK_MINUS_1].to_numpy(dtype=str)

    fetched_rts_week = np.full(len(df_main), np.nan, dtype=int)
    fetched_eos_week = np.full(len(df_main), np.nan, dtype=int)
    fetched_eos_minus1 = np.full(len(df_main), np.nan, dtype=int)

    fetched_rts_week[valid_mask_rts_eos] = rts_week_array[positions_rts_eos[valid_mask_rts_eos]]
    fetched_eos_week[valid_mask_rts_eos] = eos_week_array[positions_rts_eos[valid_mask_rts_eos]]
    fetched_eos_minus1[valid_mask_rts_eos] = eos_minus1_array[positions_rts_eos[valid_mask_rts_eos]]

    # If you also want partial-week, do similarly with df_fn_RTS_EOS_Week:
    lookup_week = df_fn_RTS_EOS_Week.set_index([Item_Item,Sales_Domain_ShipTo,Partial_Week])
    positions_week = lookup_week.index.get_indexer(
        pd.MultiIndex.from_arrays([
            df_main[Item_Item],
            df_main[RTS_EOS_ShipTo],
            df_main[Partial_Week]
        ])
    )
    valid_mask_week = (positions_week != -1)
    row_partial_array_full = lookup_week[CURRENT_ROW_WEEK].to_numpy(dtype=int)

    fetched_partial_array = np.full(len(df_main), np.nan, dtype=int)
    fetched_partial_array[valid_mask_week] = row_partial_array_full[positions_week[valid_mask_week]]

    # Convert to series of int for easy numeric compare
    s_rts_week =  pd.Series(fetched_rts_week, dtype=int)
    s_eos_week = pd.Series(fetched_eos_week, dtype=int)
    s_eos_minus1 = pd.Series(fetched_eos_minus1, dtype=int)
    s_partial = pd.Series(fetched_partial_array, dtype=int)

    # -------------------------------------------------------
    # 3) Build final condition => YELLOW
    # -------------------------------------------------------
    # Condition to check if row is found in item_class (valid_mask_iclass)
    # + partial-week conditions from your snippet:
    cond_1 = (
        valid_mask_iclass &  # must have a row in df_in_Item_CLASS
        (s_eos_week >= int(max_week_normalized)) &
        (s_partial >= int(current_week_normalized)) &
        (s_partial >= s_eos_minus1) &
        (df_main[SIn_FCST_Color_Condition] != '11_LIGHTRED') &
        (df_main[SIn_FCST_Color_Condition] != '16_DARKRED')
    )

    cond_2 = (
        valid_mask_iclass &
        # (s_eos_week >= s_rts_week) &
        (s_eos_week < int(max_week_normalized)) &
        (s_partial >= int(current_week_normalized)) &
        (s_partial <= s_eos_week) &
        (df_main[SIn_FCST_Color_Condition] != '11_LIGHTRED') &
        (df_main[SIn_FCST_Color_Condition] != '16_DARKRED')
    )

    # If condition is True => set lock=False, color=YELLOW
    df_main.loc[cond_1, [SIn_FCST_GC_LOCK,SIn_FCST_Color_Condition]] = [False, '12_YELLOW']
    df_main.loc[cond_2, [SIn_FCST_GC_LOCK,SIn_FCST_Color_Condition]] = [False, '12_YELLOW']

    # -------------------------------------------------------
    # 4) Return updated DataFrame
    # -------------------------------------------------------
    return df_main




@_decoration_
def fn_step10_apply_item_tat(df_fn_sales_asn_item_week: pd.DataFrame) -> pd.DataFrame:
    """
    Step 10: Use array-based index lookups to set lock and color.

    Process:
      1) From df_in_Item_TAT, get ITEMTAT_TATTERM by matching (Item_Item, Location_Location).
      2) From df_fn_RTS_EOS_Week, get current_week_normalized by matching
         (Item_Item, Sales_Domain_Ship_To, Partial_Week).
      3) If current_week_normalized <= current_row_partial_week_normalized <=
         current_week_normalized + ITEMTAT_TATTERM,
         then set SIn_FCST(GI)_GC_LOCK = True, SIn_FCST(GI)_Color_Condition = '18_DGRAY_RED'.

    Parameters:
      df_fn_sales_asn_item_week: The target DataFrame for Step10. No merges, all array-based.

    Returns:
      df_fn_sales_asn_item_week with updated lock & color for matching rows.
    """

    # 1) The "target" DF we want to update
    df_target = df_fn_sales_asn_item_week

    # 2) Lookup #1: df_in_Item_TAT for ITEMTAT_TATTERM
    df_item_tat = input_dataframes[str_df_in_Item_TAT]
    # Build index on (Item_Item, Location_Location)
    tat_index = df_item_tat.set_index([Item_Item, Location_Location])
    tatterm_array_full = tat_index[ITEMTAT_TATTERM].to_numpy(dtype=int)

    # For each row in df_target, get position in tat_index
    item_array  = df_target[Item_Item].to_numpy(dtype=object)
    loc_array   = df_target[Location_Location].to_numpy(dtype=object)
    multi_idx_tat = pd.MultiIndex.from_arrays([item_array, loc_array],
                                              names=[Item_Item,Location_Location])
    positions_tat = tat_index.index.get_indexer(multi_idx_tat)
    valid_mask_tat = (positions_tat != -1)

    # Create arr_tatterm for df_target
    arr_tatterm = np.full(len(df_target), np.nan, dtype=int)
    arr_tatterm[valid_mask_tat] = tatterm_array_full[positions_tat[valid_mask_tat]]

    # 3) Lookup #2: df_fn_RTS_EOS_Week for current_week_normalized
    #    We'll index by (Item_Item, Sales_Domain_Ship_To, Partial_Week)
    df_rts_eos_week = output_dataframes[str_df_fn_RTS_EOS_Week]
    lookup_week = df_rts_eos_week.set_index([Item_Item, Sales_Domain_ShipTo, Partial_Week])
    # lookup_week = df_rts_eos_week.index

    row_partial_array_full = lookup_week[CURRENT_ROW_WEEK].to_numpy(dtype=int)

    multi_idx_week = pd.MultiIndex.from_arrays([
        df_target[Item_Item],
        df_target[RTS_EOS_ShipTo],
        df_target[Partial_Week]
    ])
    positions_week = lookup_week.index.get_indexer(multi_idx_week)
    valid_mask_week = (positions_week != -1)

    fetched_partial_array = np.full(len(df_target), np.nan, dtype=object)
    fetched_partial_array[valid_mask_week] = row_partial_array_full[positions_week[valid_mask_week]]
    # fetched_partial_array = fetched_partial_array.fillna(0)
    # 4) Check for NaNs in fetched_partial_array
    NaN_mask = pd.isna(fetched_partial_array)
    fetched_partial_array[NaN_mask] = 0

    # if NaN_mask.any():
    #     # Let’s log how many rows have NaN and show the first few
    #     nan_count = NaN_mask.sum()
    #     # Optionally build a small DataFrame with relevant columns
    #     df_nan_info = df_target.loc[NaN_mask, [
    #         Item_Item, 
    #         RTS_EOS_ShipTo, 
    #         Partial_Week
    #     ]]
    #     # Log or print
    #     logger.Note(f"[Step10] Found {nan_count} rows with NaN for 'CURRENT_ROW_WEEK'. Some samples:\n{df_nan_info.head(10)}", 
    #                 p_log_level=common.G_log_level.warning())
    #     # If you want to raise an error, uncomment:
    #     # raise ValueError(f"[Step10] {nan_count} rows have NaN 'CURRENT_ROW_WEEK' – can't cast to int")
    # fetched_partial_array[NaN_mask] = 0


    # # 4) We also have the row's partial week in df_target
    # row_partial = df_target[CURRENT_ROW_WEEK].fillna('0').astype(int).to_numpy()
    # Convert to series of int for easy numeric compare
    row_partial = pd.Series(fetched_partial_array,dtype=int)

    # 5) Condition:
    #   current_week_normalized <= row_partial <= current_week_normalized + itemtat_tatterm
    #   plus we need valid_mask_tat & valid_mask_week
    cond = (
        valid_mask_tat & valid_mask_week &
        (row_partial >= int(current_week_normalized)) &
        (row_partial < (int(current_week_normalized) + arr_tatterm))
    )

    # 6) Set lock & color
    df_target.loc[cond, [SIn_FCST_GC_LOCK,SIn_FCST_Color_Condition]] = [True, '18_DGRAY_RED']

    return df_target


@_decoration_
def fn_step11_apply_item_tatset_lock_join(df_fn_sales_asn_item_week: pd.DataFrame) -> pd.DataFrame:
    """
    Step 11: Similar to Step 10 but uses column 'ITEMTAT TATTERM_SET'.
    """

    # 1) The "target" DF we want to update
    df_target = df_fn_sales_asn_item_week

    # 2) Lookup #1: df_in_Item_TAT for ITEMTAT_TATTERM
    df_item_tat = input_dataframes[str_df_in_Item_TAT]
    # Build index on (Item_Item, Location_Location)
    tat_index = df_item_tat.set_index([Item_Item, Location_Location])
    tatterm_array_full = tat_index[ITEMTAT_TATTERM_SET].to_numpy(dtype=int)

    # For each row in df_target, get position in tat_index
    item_array  = df_target[Item_Item].to_numpy(dtype=object)
    loc_array   = df_target[Location_Location].to_numpy(dtype=object)
    multi_idx_tat = pd.MultiIndex.from_arrays([item_array, loc_array],
                                              names=[Item_Item,Location_Location])
    positions_tat = tat_index.index.get_indexer(multi_idx_tat)
    valid_mask_tat = (positions_tat != -1)

    # Create arr_tatterm for df_target
    arr_tatterm = np.full(len(df_target), np.nan, dtype=int)
    arr_tatterm[valid_mask_tat] = tatterm_array_full[positions_tat[valid_mask_tat]]

    # 3) Lookup #2: df_fn_RTS_EOS_Week for current_week_normalized
    #    We'll index by (Item_Item, Sales_Domain_Ship_To, Partial_Week)
    df_rts_eos_week = output_dataframes[str_df_fn_RTS_EOS_Week]
    lookup_week = df_rts_eos_week.set_index([Item_Item, Sales_Domain_ShipTo, Partial_Week])
    # lookup_week = df_rts_eos_week.index

    row_partial_array_full = lookup_week[CURRENT_ROW_WEEK].to_numpy(dtype=int)

    multi_idx_week = pd.MultiIndex.from_arrays([
        df_target[Item_Item],
        df_target[RTS_EOS_ShipTo],
        df_target[Partial_Week]
    ])
    positions_week = lookup_week.index.get_indexer(multi_idx_week)
    valid_mask_week = (positions_week != -1)

    fetched_partial_array = np.full(len(df_target), np.nan, dtype=object)
    fetched_partial_array[valid_mask_week] = row_partial_array_full[positions_week[valid_mask_week]]
    NaN_mask = pd.isna(fetched_partial_array)
    fetched_partial_array[NaN_mask] = 0

    # # 4) We also have the row's partial week in df_target
    # row_partial = df_target[CURRENT_ROW_WEEK].fillna('0').astype(int).to_numpy()
    # Convert to series of int for easy numeric compare
    row_partial = pd.Series(fetched_partial_array, dtype=int)

    # 5) Condition:
    #   current_week_normalized <= row_partial <= current_week_normalized + itemtat_tatterm
    #   plus we need valid_mask_tat & valid_mask_week
    cond_color = (
        valid_mask_tat & valid_mask_week &
        (row_partial >= int(current_week_normalized)) &
        (row_partial < (int(current_week_normalized) + arr_tatterm)) &
        (df_target[SIn_FCST_Color_Condition] == '18_DGRAY_RED')
    )
    cond_lock = (
        valid_mask_tat & valid_mask_week &
        (row_partial >= int(current_week_normalized)) &
        (row_partial < (int(current_week_normalized) + arr_tatterm)) 
    )

    # 6) Set lock & color
    df_target.loc[cond_color, SIn_FCST_Color_Condition] = '17_DGRAY_REDB'
    df_target.loc[cond_lock, SIn_FCST_GC_LOCK] = True

    return df_target


@_decoration_
def fn_step12_apply_forecast_rule(
    df_fn_sales_product_asn_item_week: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 12:
    Builds a new DataFrame (df_fn_Sales_Product_ASN_Item_Week_Forcast) from df_fn_Sales_Product_ASN_Item_Week (df_source),
    referencing df_in_Forecast_Rule (dictionary-based) and df_in_Sales_Domain_Dimension (array-based),
    *without* using merges or iterrows on df_in_Forecast_Rule.

    1) We interpret domain level from df_in_Sales_Domain_Dimension to pick a 
       domain-lv2..7 if needed.
    2) We do dictionary lookups for (ProductGroup, domainStr) => (FORECAST_RULE_GC, AP2, AP1).
    3) We produce up to 3 subrows (GC, AP2, AP1) for each source row if those rules exist,
       or update existing row in the final DF (so we don't insert duplicates).

    Returns:
      A new DataFrame with columns:
       [Sales_Domain_ShipTo, Item_Item, Item_GBM, Location_Location, Partial_Week,
        SIn_FCST_GC_LOCK, SIn_FCST_AP2_LOCK, SIn_FCST_AP1_LOCK, SIn_FCST_Color_Condition]
    """

    df_source = df_fn_sales_product_asn_item_week

    # ----------------------------------------------------------------
    # 0) Prepare final columns
    # ----------------------------------------------------------------
    final_cols = [
        Sales_Domain_ShipTo,
        Item_Item,
        # Item_GBM,
        Location_Location,
        Partial_Week,
        SIn_FCST_GC_LOCK,
        SIn_FCST_AP2_LOCK,
        SIn_FCST_AP1_LOCK,
        SIn_FCST_Color_Condition
        # ,Product_Group,        # stored so we can sort
        # ForecastRuleShipto       # stored so we can sort
    ]

    # We'll build the final rows in a list-of-dicts
    new_rows = []
    # We keep a dictionary to check if a row with
    # (domain, item, location, partialWeek) was already inserted:
    row_map = {}

    # ----------------------------------------------------------------
    # 1) Build dimension index for array-based domain-lv2..7 lookups
    # ----------------------------------------------------------------
    df_dim = input_dataframes[str_df_in_Sales_Domain]
    dim_index = df_dim.set_index([Sales_Domain_ShipTo])

    arr_lv2_full = dim_index[Sales_Domain_LV2].to_numpy(dtype=object)
    arr_lv3_full = dim_index[Sales_Domain_LV3].to_numpy(dtype=object)
    arr_lv4_full = dim_index[Sales_Domain_LV4].to_numpy(dtype=object)
    arr_lv5_full = dim_index[Sales_Domain_LV5].to_numpy(dtype=object)
    arr_lv6_full = dim_index[Sales_Domain_LV6].to_numpy(dtype=object)
    arr_lv7_full = dim_index[Sales_Domain_LV7].to_numpy(dtype=object)

    # get_indexer for df_source:
    domain_src_array = df_source[Sales_Domain_ShipTo].to_numpy(dtype=object)
    positions_dim = dim_index.index.get_indexer(domain_src_array)
    valid_dim_mask = (positions_dim != -1)

    def pick_domain(idx_dim: int, level: int) -> str:
        """Pick dimension-lv2..7 from arrays, given the dimension row index and desired level."""
        if level == 2: return arr_lv2_full[idx_dim]
        elif level == 3: return arr_lv3_full[idx_dim]
        elif level == 4: return arr_lv4_full[idx_dim]
        elif level == 5: return arr_lv5_full[idx_dim]
        elif level == 6: return arr_lv6_full[idx_dim]
        elif level == 7: return arr_lv7_full[idx_dim]
        return None

    # ----------------------------------------------------------------
    # 2) Build a dictionary for df_in_Forecast_Rule:
    #    (ProductGroup, domain) -> (gc_val, ap2_val, ap1_val)
    # ----------------------------------------------------------------
    df_rule = input_dataframes[str_df_in_Forecast_Rule]
    # We'll store them in a dictionary for O(1) lookups
    # that means if df_rule has columns: Product_Group, Sales_Domain_ShipTo, FORECAST_RULE_GC, AP2, AP1, ...
    rule_dict = {}
    for idx_r in range(len(df_rule)):
        row_r = df_rule.iloc[idx_r]
        pg = row_r[Product_Group]
        dom = row_r[Sales_Domain_ShipTo]
        gcv = int(row_r[FORECAST_RULE_GC_FCST])
        ap2v= int(row_r[FORECAST_RULE_AP2_FCST])
        ap1v= int(row_r[FORECAST_RULE_AP1_FCST])
        rule_dict[(pg, dom)] = (gcv, ap2v, ap1v)

    # ----------------------------------------------------------------
    # 3) array-extract from df_source for quick reference
    # ----------------------------------------------------------------
    shipto_array_source  = df_source[Sales_Domain_ShipTo].to_numpy(dtype=object)
    pg_array_source  = df_source[Product_Group].to_numpy(dtype=object)
    item_array_source= df_source[Item_Item].to_numpy(dtype=object)
    # gbm_array_source = df_source[Item_GBM].to_numpy(dtype=object)
    loc_array_source = df_source[Location_Location].to_numpy(dtype=object)
    part_array_source= df_source[Partial_Week].to_numpy(dtype=object)
    gc_lock_array    = df_source[SIn_FCST_GC_LOCK].to_numpy(dtype=bool)
    color_array      = df_source[SIn_FCST_Color_Condition].to_numpy(dtype=object)

    # ----------------------------------------------------------------
    # 4) Single pass over df_source
    # ----------------------------------------------------------------
    for i in range(len(df_source)):
        if not valid_dim_mask[i]:
            continue  # dimension row not found, skip
        dimpos = positions_dim[i]

        # read source data
        shipto_val = shipto_array_source[i]
        pg_val    = pg_array_source[i]
        item_val  = item_array_source[i]
        # gbm_val   = gbm_array_source[i]
        loc_val   = loc_array_source[i]
        part_val  = part_array_source[i]
        gc_lock   = gc_lock_array[i]
        color_val = color_array[i]

        # We want to see if there's a rule row in rule_dict for (pg_val, ???) at lv2 or lv3 or lv7, etc. 
        # Actually, your instructions say you interpret the rule's domain strings as "start with '2' => level=2," etc. 
        # We'll do a simpler approach: 
        # We'll try all domain levels from [2..7], build domain_str=pick_domain(dimpos, lv).
        # Then see if (pg_val, domain_str) is in rule_dict. If yes => we get (gc, ap2, ap1). Then we create or update rows. 
        for d in range(2, 8):
            domain_candidate = pick_domain(dimpos, d)
            if domain_candidate is None:
                continue
            # see if there's a rule for (pg_val, domain_candidate)
            if (pg_val, domain_candidate) not in rule_dict:
                continue
            # we found a rule => retrieve (gc_val, ap2_val, ap1_val)
            gc_val, ap2_val, ap1_val = rule_dict[(pg_val, domain_candidate)]

            # We'll build up to 3 subrows: one for GC, one for AP2, one for AP1, if the rule says 2..7
            # for example, if gc_val=6, we pick dimension-lv6 => a domain for GC row. 
            # So let's define a function to handle "build or update row"
            def build_or_update(domain_level: int, which_lock: str,rule_shipto: str):
                """domain_level in [2..7], which_lock in {'GC','AP2','AP1'}."""
                dom_str = pick_domain(dimpos, domain_level)
                if dom_str is None:
                    return
                # build the key
                my_key = (dom_str, item_val, loc_val, part_val)
                if my_key in row_map:
                    row_idx = row_map[my_key]
                    # update the lock in new_rows[row_idx]
                    if which_lock == 'GC':
                        new_rows[row_idx][SIn_FCST_GC_LOCK] = gc_lock
                    elif which_lock == 'AP2':
                        new_rows[row_idx][SIn_FCST_AP2_LOCK] = gc_lock
                    elif which_lock == 'AP1':
                        new_rows[row_idx][SIn_FCST_AP1_LOCK] = gc_lock
                else:
                    # create a new row
                    row_dict = {
                        Sales_Domain_ShipTo: dom_str,
                        Item_Item: item_val,
                        # Item_GBM: gbm_val,
                        Location_Location: loc_val,
                        Partial_Week: part_val,
                        SIn_FCST_GC_LOCK: True,
                        SIn_FCST_AP2_LOCK: True,
                        SIn_FCST_AP1_LOCK: True,
                        SIn_FCST_Color_Condition: color_val
                        # , Product_Group: pg_val,
                        # ForecastRuleShipto: rule_shipto
                    }
                    if which_lock == 'GC':
                        row_dict[SIn_FCST_GC_LOCK] = gc_lock
                    elif which_lock == 'AP2':
                        row_dict[SIn_FCST_AP2_LOCK] = gc_lock
                    elif which_lock == 'AP1':
                        row_dict[SIn_FCST_AP1_LOCK] = gc_lock
                    new_rows.append(row_dict)
                    row_map[my_key] = len(new_rows) - 1

            # original logic
            row_dict_origin = {
                Sales_Domain_ShipTo: shipto_val,
                Item_Item: item_val,
                # Item_GBM: gbm_val,
                Location_Location: loc_val,
                Partial_Week: part_val,
                SIn_FCST_GC_LOCK: True,
                SIn_FCST_AP2_LOCK: True,
                SIn_FCST_AP1_LOCK: True,
                SIn_FCST_Color_Condition: color_val
                # , Product_Group: pg_val,
                # ForecastRuleShipto: domain_candidate
            }
            new_rows.append(row_dict_origin)
            my_key_origin = (shipto_val, item_val, loc_val, part_val)
            row_map[my_key_origin] = len(new_rows) - 1

            # Now we interpret the numeric domain for GC/AP2/AP1
            # E.g. if gc_val=6 => pick_domain(dimpos,6) => domain-lv6
            if 2 <= gc_val <= 7:
                build_or_update(gc_val, 'GC',domain_candidate)
            if 2 <= ap2_val <= 7:
                build_or_update(ap2_val, 'AP2',domain_candidate)
            if 2 <= ap1_val <= 7:
                build_or_update(ap1_val, 'AP1',domain_candidate)

    # end loop

    # 5) Build final DataFrame
    df_forecast = pd.DataFrame(new_rows, columns=final_cols)

    # # 6) Sort by [Item_Product_Group, 'ForecastRuleShipto']
    # df_forecast.sort_values(by=[Product_Group, ForecastRuleShipto], inplace=True)
    # df_forecast.reset_index(drop=True, inplace=True)

    # # 7) If you don't need 'ForecastRuleShipto' or 'Item.[Product Group]' in final,
    # #    you can drop them:
    # df_forecast.drop(columns=[ForecastRuleShipto, Product_Group], inplace=True)


    return df_forecast

@_decoration_
def fn_step13_create_forecast_estore(
    df_fn_sales_product_asn_item_week: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 13:
      - We filter df_fn_sales_product_asn_item_week by the eStore list in df_in_Sales_Domain_Estore.
      - Then we do the same array-based logic as step12 to produce a forecast table,
        but using new column names:
          SOut_FCST_GC_LOCK,
          SOut_FCST_AP2_LOCK,
          SOut_FCST_AP1_LOCK,
          SOut_FCST_Color_Condition
      - This code skeleton references df_in_Forecast_Rule for domain lookups, and
        df_in_Sales_Domain_Dimension for dimension-lv2..7 picking, just like step12,
        but with an additional filter for df_in_Sales_Domain_Estore.

    Returns:
      A new DataFrame (df_fn_Sales_Product_ASN_Item_Week_Forecast_eStore) with columns:
       [Sales_Domain_ShipTo, Item_Item, Location_Location, Partial_Week,
        SOut_FCST_GC_LOCK, SOut_FCST_AP2_LOCK, SOut_FCST_AP1_LOCK, SOut_FCST_Color_Condition]
    """

    # 0) Load the eStore shipto data for filtering
    df_estore = input_dataframes[str_df_in_Sales_Domain_Estore]
    # We'll create a set of valid eStore shiptos
    estore_shipto_set = set(df_estore[Sales_Domain_ShipTo].unique())

    # 1) Filter df_fn_sales_product_asn_item_week so we only process rows
    #    whose Sales_Domain_ShipTo is in the eStore set
    df_source = df_fn_sales_product_asn_item_week
    df_source = df_source[df_source[Sales_Domain_ShipTo].isin(estore_shipto_set)].reset_index(drop=True)

    # 2) Prepare final columns
    final_cols = [
        Sales_Domain_ShipTo,
        Item_Item,
        Location_Location,
        Partial_Week,
        SOut_FCST_GC_LOCK,
        SOut_FCST_AP2_LOCK,
        SOut_FCST_AP1_LOCK,
        SOut_FCST_Color_Condition
    ]

    # We'll collect new rows in a list-of-dicts
    new_rows = []
    row_map = {}

    # 3) Build dimension index (df_in_Sales_Domain_Dimension)
    df_dim = input_dataframes[str_df_in_Sales_Domain]
    dim_index = df_dim.set_index([Sales_Domain_ShipTo])

    arr_lv2_full = dim_index[Sales_Domain_LV2].to_numpy(dtype=object)
    arr_lv3_full = dim_index[Sales_Domain_LV3].to_numpy(dtype=object)
    arr_lv4_full = dim_index[Sales_Domain_LV4].to_numpy(dtype=object)
    arr_lv5_full = dim_index[Sales_Domain_LV5].to_numpy(dtype=object)
    arr_lv6_full = dim_index[Sales_Domain_LV6].to_numpy(dtype=object)
    arr_lv7_full = dim_index[Sales_Domain_LV7].to_numpy(dtype=object)

    domain_src_array = df_source[Sales_Domain_ShipTo].to_numpy(dtype=object)
    positions_dim = dim_index.index.get_indexer(domain_src_array)
    valid_dim_mask = (positions_dim != -1)

    def pick_domain(dim_pos: int, level: int) -> str:
        if level == 2: return arr_lv2_full[dim_pos]
        elif level == 3: return arr_lv3_full[dim_pos]
        elif level == 4: return arr_lv4_full[dim_pos]
        elif level == 5: return arr_lv5_full[dim_pos]
        elif level == 6: return arr_lv6_full[dim_pos]
        elif level == 7: return arr_lv7_full[dim_pos]
        return None

    # 4) Build dictionary from df_in_Forecast_Rule => (ProductGroup, domain_str) -> (gc_val, ap2_val, ap1_val)
    df_rule = input_dataframes[str_df_in_Forecast_Rule]
    rule_dict = {}
    for idx_r in range(len(df_rule)):
        row_r = df_rule.iloc[idx_r]
        pg_val   = row_r[Product_Group]  # e.g. "REF"
        dom_str  = row_r[Sales_Domain_ShipTo] # e.g. "300114"
        gc_val   = int(row_r[FORECAST_RULE_GC_FCST])
        ap2_val  = int(row_r[FORECAST_RULE_AP2_FCST])
        ap1_val  = int(row_r[FORECAST_RULE_AP1_FCST])
        rule_dict[(pg_val, dom_str)] = (gc_val, ap2_val, ap1_val)

    # 5) Extract arrays from df_source for quick reference
    shipto_array = df_source[Sales_Domain_ShipTo].to_numpy(dtype=object)
    pg_array     = df_source[Product_Group].to_numpy(dtype=object)
    item_array   = df_source[Item_Item].to_numpy(dtype=object)
    loc_array    = df_source[Location_Location].to_numpy(dtype=object)
    part_array   = df_source[Partial_Week].to_numpy(dtype=object)
    # In step12 code, you used a lock array or color array. We'll do the same but rename them:
    # e.g. old code had 'SIn_FCST_GC_LOCK', etc. Now we might just import from df_source
    # or build them if needed. If there's a prior lock, rename as 'GC_lock_array'. 
    # If your step12 code didn't have them, you can skip. For demo, we assume color is in df_source:
    gc_lock_array    = df_source[SIn_FCST_GC_LOCK].to_numpy(dtype=bool)
    color_array  = df_source[SIn_FCST_Color_Condition].to_numpy(dtype=object)

    # We'll assume there's an existing boolean column for GC lock if needed, or just default True
    # We'll do default True for demonstration:
    default_gc_lock = True

    # 6) Single pass over df_source
    for i in range(len(df_source)):
        if not valid_dim_mask[i]:
            continue
        dim_pos = positions_dim[i]

        # read source data
        shipto_val = shipto_array[i]
        pg_val     = pg_array[i]
        item_val   = item_array[i]
        loc_val    = loc_array[i]
        part_val   = part_array[i]
        gc_lock   = gc_lock_array[i]
        color_val  = color_array[i]

        # We'll try dimension-lv2..7
        for d in range(2, 8):
            domain_candidate = pick_domain(dim_pos, d)
            if domain_candidate is None:
                continue
            # check if (pg_val, domain_candidate) is in rule_dict
            if (pg_val, domain_candidate) not in rule_dict:
                continue

            # retrieve (gc_val, ap2_val, ap1_val)
            gc_val, ap2_val, ap1_val = rule_dict[(pg_val, domain_candidate)]

            def build_or_update(domain_level: int, which_lock: str):
                """
                domain_level in [2..7], which_lock in {'GC','AP2','AP1'}.
                We'll pick domain-lvX, then either insert or update 
                a row in new_rows with the new column names for step13.
                """
                new_dom = pick_domain(dim_pos, domain_level)
                if not new_dom:
                    return
                # build key
                my_key = (new_dom, item_val, loc_val, part_val)

                if my_key in row_map:
                    row_idx = row_map[my_key]
                    # update whichever lock
                    if which_lock == 'GC':
                        new_rows[row_idx][SOut_FCST_GC_LOCK] = gc_lock
                    elif which_lock == 'AP2':
                        new_rows[row_idx][SOut_FCST_AP2_LOCK] = gc_lock
                    elif which_lock == 'AP1':
                        new_rows[row_idx][SOut_FCST_AP1_LOCK] = gc_lock
                else:
                    row_dict = {
                        Sales_Domain_ShipTo: new_dom,
                        Item_Item: item_val,
                        Location_Location: loc_val,
                        Partial_Week: part_val,
                        SOut_FCST_GC_LOCK: True,
                        SOut_FCST_AP2_LOCK: True,
                        SOut_FCST_AP1_LOCK: True,
                        SOut_FCST_Color_Condition: color_val
                    }
                    # set the relevant lock = True
                    if which_lock == 'GC':
                        row_dict[SOut_FCST_GC_LOCK] = gc_lock
                    elif which_lock == 'AP2':
                        row_dict[SOut_FCST_AP2_LOCK] = gc_lock
                    elif which_lock == 'AP1':
                        row_dict[SOut_FCST_AP1_LOCK] = gc_lock

                    new_rows.append(row_dict)
                    row_map[my_key] = len(new_rows) - 1
            # original logic
            row_dict_origin = {
                Sales_Domain_ShipTo: shipto_val,
                Item_Item: item_val,
                # Item_GBM: gbm_val,
                Location_Location: loc_val,
                Partial_Week: part_val,
                SOut_FCST_GC_LOCK: True,
                SOut_FCST_AP2_LOCK: True,
                SOut_FCST_AP1_LOCK: True,
                SOut_FCST_Color_Condition: color_val
                # , Product_Group: pg_val,
                # ForecastRuleShipto: domain_candidate
            }
            new_rows.append(row_dict_origin)
            my_key_origin = (shipto_val, item_val, loc_val, part_val)
            row_map[my_key_origin] = len(new_rows) - 1

            # interpret numeric domain for GC/AP2/AP1
            if 2 <= gc_val <= 7:
                build_or_update(gc_val, 'GC')
            if 2 <= ap2_val <= 7:
                build_or_update(ap2_val, 'AP2')
            if 2 <= ap1_val <= 7:
                build_or_update(ap1_val, 'AP1')

    # 7) Build final DataFrame
    df_forecast = pd.DataFrame(new_rows, columns=final_cols)

    # # 8) Sort by [Item_Product_Group, 'ForecastRuleShipto']
    # df_forecast.sort_values(by=[Product_Group, ForecastRuleShipto], inplace=True)
    # df_forecast.reset_index(drop=True, inplace=True)

    # # 9) If you don't need 'ForecastRuleShipto' or 'Item.[Product Group]' in final,
    # #    you can drop them:
    # df_forecast.drop(columns=[ForecastRuleShipto, Product_Group], inplace=True)

    return df_forecast


@_decoration_
def fn_output_formatter_sellin(df_p_source: pd.DataFrame, str_p_out_version: str) -> pd.DataFrame:
    """
    최종 Output 형태로 정리
    :param df_p_source: 주차별로 가공하여 group by 후 sum을 구한 in_Demand
    :param str_p_out_version: Param_OUT_VERSION
    :return: DataFrame
    """
    # 함수명
    str_my_name = inspect.stack()[0][3]
    # Return 변수
    df_return = pd.DataFrame()

    # 입력 파라미터가 비어 있는 경우 비어 있는 DataFrame을 리턴
    if df_p_source.empty:
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(df_p_source)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    # 입력 파라미터(str_p_out_version)가 비어 있는 경우 경고 메시지 출력 후 빈 데이터 프레임 리턴
    if str_p_out_version is None or str_p_out_version.strip() == '':
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(str_p_out_version)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    df_return = df_p_source
    df_return['Version.[Version Name]'] = str_p_out_version

    final_cols = [
        Version_Name,
        Sales_Domain_ShipTo,
        Item_Item,
        Location_Location,
        Partial_Week,
        SIn_FCST_GC_LOCK,
        SIn_FCST_AP2_LOCK,
        SIn_FCST_AP1_LOCK,
        SIn_FCST_Color_Condition
    ]

    df_return = df_return[final_cols]

    return df_return


@_decoration_
def fn_output_formatter_sellout(df_p_source: pd.DataFrame, str_p_out_version: str) -> pd.DataFrame:
    """
    최종 Output 형태로 정리
    :param df_p_source: 주차별로 가공하여 group by 후 sum을 구한 in_Demand
    :param str_p_out_version: Param_OUT_VERSION
    :return: DataFrame
    """
    # 함수명
    str_my_name = inspect.stack()[0][3]
    # Return 변수
    df_return = pd.DataFrame()

    # 입력 파라미터가 비어 있는 경우 비어 있는 DataFrame을 리턴
    if df_p_source.empty:
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(df_p_source)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    # 입력 파라미터(str_p_out_version)가 비어 있는 경우 경고 메시지 출력 후 빈 데이터 프레임 리턴
    if str_p_out_version is None or str_p_out_version.strip() == '':
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(str_p_out_version)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    df_return = df_p_source
    df_return['Version.[Version Name]'] = str_p_out_version

    final_cols = [
        Version_Name,
        Sales_Domain_ShipTo,
        Item_Item,
        Location_Location,
        Partial_Week,
        SOut_FCST_GC_LOCK,
        SOut_FCST_AP2_LOCK,
        SOut_FCST_AP1_LOCK,
        SOut_FCST_Color_Condition
    ]

    df_return = df_return[final_cols]

    return df_return



if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    # Output 테이블 선언
    out_Demand = pd.DataFrame()
    out_sellin = pd.DataFrame()
    out_sellout = pd.DataFrame()
    output_dataframes = {}
    input_dataframes = {}
    try:
        ################################################################################################################
        # 전처리 : 모듈 내에서 사용될 데이터에 대한 정합성 체크 및 데이터 선 가공
        ################################################################################################################
        
        if is_local:
            Version = 'CWV_DP'
            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------

            # input_folder_name  = str_instance
            # output_folder_name = str_instance
            input_folder_name  = f'{str_instance}_SHA_REF'
            output_folder_name = f'{str_instance}_SHA_REF'
            # ------
            str_input_dir = f'Input/{input_folder_name}'
            # ------
            str_output_dir = f'Output/{output_folder_name}'
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y%m%d_%H_%M")
            str_output_dir = f"{str_output_dir}_{formatted_time}"
            # ------
            os.makedirs(str_input_dir, exist_ok=True)
            os.makedirs(str_output_dir, exist_ok=True)


            # ----------------------------------------------------
            # Week
            # ----------------------------------------------------
            CurrentPartialWeek = '202504A'

            

        # --------------------------------------------------------------------------
        # df_input 체크 시작
        # --------------------------------------------------------------------------
        logger.Note(p_note='df_input 체크 시작', p_log_level=LOG_LEVEL.debug())
        fn_process_in_df_mst()

        


        df_eos = input_dataframes.get(str_df_in_Time_Partial_Week)
        max_week = df_eos[Partial_Week].max()
        max_week_normalized = normalize_week(max_week)
        current_week_normalized = normalize_week(CurrentPartialWeek)


        # 입력 변수 확인
        if Version is None or Version.strip() == '':
            Version = 'CWV_DP'
        # 입력 변수 확인용 로그
        logger.Note(p_note=f'Version : {Version}', p_log_level=LOG_LEVEL.debug())


        # 입력 변수 중 데이터가 없는 경우 경고 메시지를 출력한다.
        for in_df in input_dataframes:
            fn_check_input_table(input_dataframes[in_df], in_df, '0')


        
        ################################################################################################################
        # Step 1. df_in_MST_RTS 와 df_in_MST_EOS 를 Item 과 ShipTo를 기준으로 inner Join
        ################################################################################################################
        dict_log = {
            'p_step_no': 10,
            'p_step_desc': 'Step 1  : df_in_MST_RTS 와 df_in_MST_EOS 를 Item 과 ShipTo를 기준으로 inner Join',
            # 'p_df_name' : 'df_01_joined_rts_eos'
        }
        df_fn_RTS_EOS = fn_step01_join_rts_eos(**dict_log)
        # fn_check_input_table(df_01_joined_rts_eos, 'df_01_joined_rts_eos', '0')
        output_dataframes[str_df_fn_RTS_EOS] = df_fn_RTS_EOS
        fn_log_dataframe(df_fn_RTS_EOS, f'df_01_{str_df_fn_RTS_EOS}')

        ################################################################################################################
        # Step 2  : Step1의 Result에 Time을 Partial Week 으로 변환
        ################################################################################################################
        dict_log = {
            'p_step_no': 20,
            'p_step_desc': 'Step 2  : Step1의 Result에 Time을 Partial Week 으로 변환 ',
            # 'p_df_name' : 'df_02_date_to_partial_week'
        }
        df_fn_RTS_EOS = fn_step02_convert_date_to_partial_week(**dict_log)
        # print for test  
        # output_dataframes["df_02_date_to_partial_week"] = df_02_date_to_partial_week   
        fn_log_dataframe(df_fn_RTS_EOS, f'df_02_{str_df_fn_RTS_EOS}')
        ################################################################################################################
        # Step 3:  df_in_MST_RTS 와 df_in_MST_EOS의 ITEM * ShipTo로 Inner Join하여 새로운 DF 생성 ( Output Data )
        ################################################################################################################
        dict_log = {
            'p_step_no': 30,
            'p_step_desc': 'Step 3  : df_in_MST_RTS 와 df_in_MST_EOS의 ITEM * ShipTo로 Inner Join하여 새로운 DF 생성 ( Output Data ) ' ,
            # 'p_df_name' : 'df_03_joined_rts_eos'
        }
        # df_03_joined_rts_eos = fn_step03_join_rts_eos(**dict_log)
        # # fn_check_input_table(df_03_joined_rts_eos, 'df_03_joined_rts_eos', '0')
        # output_dataframes["df_03_joined_rts_eos"] = df_03_joined_rts_eos
        # if is_local:
        #     fn_log_dataframe(df_03_joined_rts_eos, 'df_03_joined_rts_eos')

        ################################################################################################################
        # Step 4  : Step3의 df에 Partial Week 및 Measure Column 추가
        ################################################################################################################
        dict_log = {
            'p_step_no': 40,
            'p_step_desc': 'Step 4  : Step3의 df에 Partial Week 및 Measure Column 추가 ' ,
            # 'p_df_name' : 'df_04_partialweek_measurecolumn'
        }
        df_fn_RTS_EOS_Week = fn_step04_add_partialweek_measurecolumn(**dict_log)
        # fn_check_input_table(df_04_partialweek_measurecolumn, 'df_04_partialweek_measurecolumn', '0')
        # print for test  
        output_dataframes[str_df_fn_RTS_EOS_Week] = df_fn_RTS_EOS_Week
        fn_log_dataframe(df_fn_RTS_EOS_Week, f'df_04_{str_df_fn_RTS_EOS_Week}')

        ################################################################################################################
        # Step 5:  Step4의 df에 당주주차부터 RTS 와 EOS 반영 및 Color 표시
        ################################################################################################################
        dict_log = {
            'p_step_no': 50,
            'p_step_desc': 'Step 5  : Step4의 df에 당주주차부터 RTS 와 EOS 반영 및 Color 표시 ' ,
            # 'p_df_name' : 'df_05_set_lock_values'
        }
        df_fn_RTS_EOS_Week = fn_step05_set_lock_values(**dict_log)
        fn_log_dataframe(df_fn_RTS_EOS_Week, f'df_05_{str_df_fn_RTS_EOS_Week}')

        ################################################################################################################
        # Step 6:  Step5의 df에 Item Master 정보 추가( Item Type, Item GBM, Item Product Group) 및 Color 조건 업데이트(무선 BAS 제품 8주 구간 GREEN UPDATE)
        ################################################################################################################
        dict_log = {
            'p_step_no': 60,
            'p_step_desc': 'Step 6  : Step5의 df에 Item Master 정보 추가 및 Color 조건 업데이트 ' ,
            # 'p_df_name' : 'df_06_addcolumn_green_for_wireless_bas'
        }
        df_fn_RTS_EOS_Week = fn_step06_addcolumn_green_for_wireless_bas_array_based(**dict_log)
        # output_dataframes["df_06_addcolumn_green_for_wireless_bas"] = df_06_addcolumn_green_for_wireless_bas
        fn_log_dataframe(df_fn_RTS_EOS_Week, f'df_06_{str_df_fn_RTS_EOS_Week}')

        ################################################################################################################
        # Step 7:  Step6의 df에 Sales Product ASN 정보 추가
        # lvl 을 구성한다.
        ################################################################################################################
        dict_log = {
            'p_step_no': 70,
            'p_step_desc': 'Step 7  : Step6의 df에 Sales Product ASN 정보 추가 ' ,
            # 'p_df_name' : 'df_07_join_sales_product_asn_to_lvl'
        }
        df_fn_Sales_Product_ASN_Item = fn_step07_join_sales_product_asn_to_lvl_by_merge(**dict_log)
        # fn_check_input_table(df_07_join_sales_product_asn_to_lvl, 'df_07_join_sales_product_asn_to_lvl', '0')
        # print for test  
        output_dataframes[str_df_fn_Sales_Product_ASN_Item] = df_fn_Sales_Product_ASN_Item  
        fn_log_dataframe(df_fn_Sales_Product_ASN_Item, f'df_07_{str_df_fn_Sales_Product_ASN_Item}')

        ################################################################################################################
        # Step 8:  Step7의 df에 조건 정보 추가. lvl 정보에 주차정보를 추가한다.
        # df_08_join_dataframes = df_07_join_sales_product_asn.join(df_06_addcolumn_green_for_wireless_bas)
        ################################################################################################################
        dict_log = {
            'p_step_no': 80,
            'p_step_desc': 'Step 8  : Step7의 df에 조건 정보 추가. lvl 정보에 주차정보를 추가한다.' 
        }
        # df_08_add_weeks_to_dimention = fn_step08_add_weeks_to_dimention_vector_join_chunk('df_07_join_sales_product_asn_to_lvl',**dict_log)
        df_fn_Sales_Product_ASN_Item_Week = fn_step08_create_asn_item_week_and_match_locks(output_dataframes[str_df_fn_Sales_Product_ASN_Item])
        output_dataframes[str_df_fn_Sales_Product_ASN_Item_Week] = df_fn_Sales_Product_ASN_Item_Week
        fn_log_dataframe(df_fn_Sales_Product_ASN_Item_Week,f'df_08_{str_df_fn_Sales_Product_ASN_Item_Week}')

        ################################################################################################################
        # Step 09:   Step8의 df에 Item CLASS 정보 필터링
        ################################################################################################################
        dict_log = {
            'p_step_no': 90,
            'p_step_desc': 'Step 9  : Step8의 df에 Item CLASS 정보 필터링 ' 
        }
        df_fn_Sales_Product_ASN_Item_Week = fn_step09_filter_itemclass_to_yellow_loc(df_fn_Sales_Product_ASN_Item_Week,**dict_log)
        fn_log_dataframe(df_fn_Sales_Product_ASN_Item_Week,f'df_09_{str_df_fn_Sales_Product_ASN_Item_Week}')

        ################################################################################################################
        # Step 10:   Step9의 df에 Item TAT 정보 필터링 및 Lock
        ################################################################################################################
        dict_log = {
            'p_step_no': 100,
            'p_step_desc': 'Step 10  : Step9의 df에 Item TAT 정보 필터링 및 Lock ' 
        }
        df_fn_Sales_Product_ASN_Item_Week = fn_step10_apply_item_tat(df_fn_Sales_Product_ASN_Item_Week,**dict_log)
        fn_log_dataframe(df_fn_Sales_Product_ASN_Item_Week,f'df_10_{str_df_fn_Sales_Product_ASN_Item_Week}')
        
        ################################################################################################################
        # Step 11:   Step10의 df에 Item TAT 정보 필터링 및 Lock.  DARKGRAY
        ################################################################################################################
        dict_log = {
            'p_step_no': 110,
            'p_step_desc': 'Step 11  : Step10의 df에 Item TAT 정보 필터링 및 Lock.  DARKGRAY ' 
        }
        df_fn_Sales_Product_ASN_Item_Week = fn_step11_apply_item_tatset_lock_join(df_fn_Sales_Product_ASN_Item_Week,**dict_log)
        fn_log_dataframe(df_fn_Sales_Product_ASN_Item_Week,f'df_11_{str_df_fn_Sales_Product_ASN_Item_Week}')
        
        ################################################################################################################
        # Step 12:   Forecast Rule에 따른 Data 생성
        ################################################################################################################
        dict_log = {
            'p_step_no': 120,
            'p_step_desc': 'Step 12  :  Forecast Rule에 따른 Data 생성' 
        }
        df_fn_Forcast_in = fn_step12_apply_forecast_rule(df_fn_Sales_Product_ASN_Item_Week,**dict_log)
        output_dataframes[str_df_fn_Forcast_in] = df_fn_Forcast_in
        fn_log_dataframe(df_fn_Forcast_in,f'df_12_{str_df_fn_Forcast_in}')
        
        ################################################################################################################
        # Step 13:   E-store 로직 추가
        ################################################################################################################
        dict_log = {
            'p_step_no': 130,
            'p_step_desc': 'Step 13  : E-store 로직 추가' 
        }
        df_fn_Forcast_out = fn_step13_create_forecast_estore(df_fn_Sales_Product_ASN_Item_Week,**dict_log)
        output_dataframes[str_df_fn_Forcast_out] = df_fn_Forcast_out
        fn_log_dataframe(df_fn_Forcast_out,f'df_13_{str_df_fn_Forcast_out}')


        ################################################################################################################
        # Step 900:  최종 Output 정리 - out_Sellin
        ################################################################################################################
        dict_log_sellin = {
            'p_step_no': 900,
            'p_step_desc': '최종 Output 정리 - out_Sellin',
            'p_df_name': 'out_Sellin'
        }
        out_sellin = fn_output_formatter_sellin(df_fn_Forcast_in, Version, **dict_log_sellin)
        fn_log_dataframe(out_sellin,f'out_sellin')

        ################################################################################################################
        # Step 910:  최종 Output 정리 - out_Sellout
        ################################################################################################################
        dict_log_sellout = {
            'p_step_no': 910,
            'p_step_desc': '최종 Output 정리 - out_Sellout',
            'p_df_name': 'out_Sellout'
        }
        out_sellout = fn_output_formatter_sellout(df_fn_Forcast_out, Version, **dict_log_sellout)
        fn_log_dataframe(out_sellout,f'out_sellout')

    except Exception as e:
        trace_msg = traceback.format_exc()
        logger.Note(p_note=trace_msg, p_log_level=LOG_LEVEL.debug())
        logger.Error()
        if flag_exception:
            raise Exception(e)
        else:
            logger.info(f'{str_instance} exit - {time.strftime("%Y-%m-%d - %H:%M:%S")}')


    finally:
        # MediumWeight 실행 시 Header 없는 빈 데이터프레임이 Output이 되는 경우 오류가 발생함.
        # 이 오류를 방지하기 위해 Output이 빈 경우을 체크하여 Header를 만들어 줌.
        if out_sellin.empty:
            out_sellin = fn_set_header_in()
            # fn_log_dataframe(out_sellin, 'out_sellin')
            if is_local:
                out_sellin.to_csv(f'{os.getcwd()}/' + str_output_dir + '/'+'out_sellin.csv', encoding='UTF8', index=False)

        if out_sellout.empty:
            out_sellout = fn_set_header_out()
            # fn_log_dataframe(out_sellout, 'out_sellout')
            if is_local:
                out_sellout.to_csv(f'{os.getcwd()}/' + str_output_dir + '/'+'out_sellout.csv', encoding='UTF8', index=False)
            

        log_file_name = common.G_PROGRAM_NAME.replace('py', 'log')
        log_file_name = f'log/{log_file_name}'
        if is_local:
            shutil.copyfile(log_file_name, os.path.join(str_output_dir, os.path.basename(log_file_name)))

            # prografile copy
            program_path = f"{os.getcwd()}/NSCM_DP_UI_Develop/{str_instance}.py"
            shutil.copyfile(program_path, os.path.join(str_output_dir, os.path.basename(program_path)))

            # log
            input_path = f'{str_output_dir}/input'
            os.makedirs(input_path,exist_ok=True)
            for input_file in input_dataframes:
                input_dataframes[input_file].to_csv(input_path + "/"+input_file+".csv", encoding="UTF8", index=False)

            # log
            output_path = f'{str_output_dir}/output'
            os.makedirs(output_path,exist_ok=True)
            for output_file in output_dataframes:
                output_dataframes[output_file].to_csv(output_path + "/"+output_file+".csv", encoding="UTF8", index=False)

            # output
            out_sellin.to_csv(output_path + "/out_sellin.csv", encoding="UTF8", index=False)
            out_sellout.to_csv(output_path + "/out_sellout.csv", encoding="UTF8", index=False)

        logger.info(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')
        logger.Finish()