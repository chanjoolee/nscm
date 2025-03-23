::: mermaid
erDiagram
    df_in_MST_RTS {
        string  Version_Name
        string  Item_Item
        string  Sales_Domain_Ship_To    "Lv3 or Lv2"
        string  RTS_STATUS
        string  RTS_ISVALID
        date    RTS_INIT_DATE
        date    RTS_DEV_DATE
        date    RTS_COM_DATE 
    }
    
    df_in_MST_EOS {
        string  Version_Name
        string  Item_Item
        string  Sales_Domain_Ship_To    "Lv3 or Lv2"
        string  EOS_STATUS
        date    EOS_INIT_DATE
        date    EOS_CHG_DATE
        date    EOS_COM_DATE 
    }

 


    %%  Joined Table
    df_fn_RTS_EOS {
        string  Item_Item               "PK"
        string  Sales_Domain_Ship_To    "PK : Lv3 or Lv2"
        int     Item_lv                 "2 or 3"
        string  RTS_STATUS
        date    RTS_INIT_DATE
        date    RTS_DEV_DATE
        date    RTS_COM_DATE 
        string  EOS_STATUS
        date    EOS_INIT_DATE           
        date    EOS_CHG_DATE
        date    EOS_COM_DATE 
        string  RTS_PARTIAL_WEEK                "Step02: Helper Column"
        string  EOS_PARTIAL_WEEK                "Step02: Helper Column"
        string  RTS_WEEK_NORMALIZED             "Step02: Helper Column"
        string  RTS_WEEK_NORMALIZED_MINUST_1    "Step02: Helper Column"
        string  RTS_WEEK_NORMALIZED_PLUS_3      "Step02: Helper Column"
        string  MAX_RTS_CURRENTWEEK             "Step02: Helper Column"
        string  EOS_WEEK_NORMALIZED             "Step02: Helper Column"
        string  EOS_WEEK_NORMALIZED_MINUS_1     "Step02: Helper Column"
        string  EOS_WEEK_NORMALIZED_MINUS_4     "Step02: Helper Column"
        string  RTS_INITIAL_WEEK_NORMALIZED     "Step02: Helper Column"
        string  EOS_INITIAL_WEEK_NORMALIZED     "Step02: Helper Column"
        string  MIN_EOSINI_MAXWEEK              "Step02: Helper Column"

    }

    df_in_Time_Partial_Week {
        string  Partial_Week
    }

    df_in_Item_Master {
        string  Item_Type
        string  Item_GBM
        string  Product_Group
        String  Item_Item
    }

    df_in_Forecast_Rule {
        string  Product_Group
        string  Sales_Domain_Ship_To
        int     FORECAST_RULE_GC_FCST
        int     FORECAST_RULE_AP2_FCST
        int     FORECAST_RULE_AP1_FCST
        string  FORECAST_RULE_ISVALID
    }

    df_in_Sales_Product_ASN {
        summary summary                 "Data is more than 1 milions"
        string  Version_Name
        string  Sales_Domain_Ship_To    "Lv7"
        string  Item_Item
        string  Location_Location
        string  Sales_Product_ASN       "Y or N"
    }

    df_in_Sales_Domain_Dimension {
        string  Sales_Domain_LV2
        string  Sales_Domain_LV3
        string  Sales_Domain_LV4
        string  Sales_Domain_LV5
        string  Sales_Domain_LV6
        string  Sales_Domain_LV7
        string  Sales_Domain_Ship_To
    }

    df_in_Item_CLASS {
        string  Item_Item
        string  Sales_Domain_Ship_To    "lv6"
        string  Location_Location
        string  Item_Class

    }

    df_fn_RTS_EOS_Week {
        string  Sales_Domain_Ship_To    "PK : Lv3 or Lv2"
        string  Item_Item               "PK"
        int     Item_lv                 "2 or 3"
        string  Item_GBM                "Step06: Join With df_in_Item_Master"
        string  Product_Group           "Step06: Join With df_in_Item_Master"
        string  Partial_Week            "PK: Step04"
        string  current_row_partial_week_normalized  "Step04: Added : A나 B를 뺀 주차숫자만 표시. 계산시 사용"
        string  CURRENTWEEK_NORMALIZED_PLUS_8   "Step04: Helper Column"
        bool    SIn_FCST(GI)_GC_LOCK            "Step04: True"
        string  SIn_FCST(GI)_Color_Condition    "Step04: 19_GRAY"
        
    }

    df_fn_Sales_Product_ASN_Item {
        summary summary                 "Data is more than 1 milions"
        string  Sales_Domain_Ship_To    "Lv7"
        string  Sales_Domain_LV2
        string  Sales_Domain_LV3
        string  Sales_Domain_LV6        "For Join with Class"
        string  Item_Item
        string  Item_GBM                "Step:07: Merged with Item Master"
        string  Product_Group           "Step:07: Merged with Item Master"
        string  Location_Location
        string  Sales_Product_ASN
        
    }

    df_fn_Sales_Product_ASN_Item_Week_lv2 {
        string  Sales_Domain_Ship_To   
        string  Sales_Domain_LV2
        string  Sales_Domain_LV3
        string  Sales_Domain_LV6        "For Join with Class"     
        string  Item_Item
        string  Item_GBM    
        string  Location_Location  
        string  Partial_Week                    "주차추가"
        bool    SIn_FCST(GI)_GC_LOCK            ""
        string  SIn_FCST(GI)_Color_Condition    ""         
    }

    df_fn_Sales_Product_ASN_Item_Week_lv3 {
        string  Sales_Domain_Ship_To
        string  Sales_Domain_LV2
        string  Sales_Domain_LV3
        string  Sales_Domain_LV6        "For Join with Class"
        string  Item_Item        
        string  Item_GBM    
        string  Location_Location  
        string  Partial_Week                    "주차추가"
        bool    SIn_FCST(GI)_GC_LOCK            ""
        string  SIn_FCST(GI)_Color_Condition    ""   
    }

    df_fn_Sales_Product_ASN_Item_Week {
        string  Sales_Domain_Ship_To
        string  Sales_Domain_LV2
        string  Sales_Domain_LV3
        string  Sales_Domain_LV6        "For Join with Class"
        string  Item_Item
        int     Item_lv                 "2 or 3"
        string  Item_GBM    
        string  Location_Location  
        string  Partial_Week 
        bool    SIn_FCST(GI)_GC_LOCK            ""
        string  SIn_FCST(GI)_Color_Condition    ""   
    }

    

    df_in_MST_RTS ||--|| df_fn_RTS_EOS : "Step2 Join Item_Item,Sales_Domain_Ship_To"
    df_in_MST_EOS ||--|| df_fn_RTS_EOS : ""

    df_fn_RTS_EOS || -- |{ df_fn_RTS_EOS_Week : "Cross Join in Step04"
    df_in_Time_Partial_Week || -- |{ df_fn_RTS_EOS_Week : "Step05: Cross Join. Adjust Lock,Condition"
    df_in_Item_Master || -- |{ df_fn_RTS_EOS_Week : "Step06: 13_GREEN For BAS MOBILE"

    df_in_Item_Master || -- |{ df_fn_Sales_Product_ASN_Item : ""
    df_in_Sales_Product_ASN || -- |{ df_fn_Sales_Product_ASN_Item   : "Step:07"
    df_in_Sales_Domain_Dimension || -- |{ df_fn_Sales_Product_ASN_Item : "Lv2,3 추가 For Join With df_fn_RTS_EOS_Week"
    df_in_Item_CLASS || -- |{ df_fn_Sales_Product_ASN_Item : ""

    df_fn_RTS_EOS_Week || -- |{ df_fn_Sales_Product_ASN_Item_Week_lv2 : "주차추가. Step08"
    df_fn_Sales_Product_ASN_Item || -- |{ df_fn_Sales_Product_ASN_Item_Week_lv2 : ""

    df_fn_RTS_EOS_Week || -- |{ df_fn_Sales_Product_ASN_Item_Week_lv3 : "주차추가. Step08"
    df_fn_Sales_Product_ASN_Item || -- |{ df_fn_Sales_Product_ASN_Item_Week_lv3 : ""

    df_fn_Sales_Product_ASN_Item_Week_lv2 || -- |{ df_fn_Sales_Product_ASN_Item_Week : ""
    df_fn_Sales_Product_ASN_Item_Week_lv3 || -- |{ df_fn_Sales_Product_ASN_Item_Week : "Item_GBM,Product_Group,Item_Item"
:::

