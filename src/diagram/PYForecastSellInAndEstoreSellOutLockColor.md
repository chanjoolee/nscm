# ERD  GC_Lock
``` mermaid
erDiagram
    df_in_MST_RTS_EOS {
        string  Version_Name
        string  Item_Item
        string  Sales_Domain_ShipTo    "Lv3 or Lv2"
        string  RTS_STATUS
        string  RTS_ISVALID
        date    RTS_INIT_DATE
        date    RTS_DEV_DATE
        date    RTS_COM_DATE 
        date    EOS_INIT_DATE
        date    EOS_CHG_DATE
        date    EOS_COM_DATE 
    }

    %%  Joined Table
    df_fn_RTS_EOS {
        string  Item_Item               "PK"
        string  Sales_Domain_ShipTo    "PK : Lv3 or Lv2"
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



    df_in_Sales_Product_ASN {
        summary summary                 "Data is more than 1 milions"
        string  Version_Name
        string  Sales_Domain_ShipTo    "Lv7 or Lv6"
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
        string  Sales_Domain_ShipTo
    }

    df_in_Item_CLASS {
        string  Item_Item
        string  Sales_Domain_ShipTo    "lv6"
        string  Location_Location
        string  Item_Class

    }

    df_fn_RTS_EOS_Week {
        string  Sales_Domain_ShipTo    "PK : Lv3 or Lv2"
        string  Item_Item               "PK"
        int     Item_lv                 "2 or 3"
        string  Item_GBM                "Step06: Join With df_in_Item_Master"
        string  Product_Group           "Step06: Join With df_in_Item_Master"
        string  Partial_Week            "PK: Step04"
        string  CURRENT_ROW_WEEK  "Step04: Added : A나 B를 뺀 주차숫자만 표시. 계산시 사용"
        string  CURRENTWEEK_NORMALIZED_PLUS_8   "Step04: Helper Column"
        bool    SIn_FCST(GI)_GC_LOCK            "Step04: True"
        string  SIn_FCST(GI)_Color_Condition    "Step04: 19_GRAY"
        
    }

    df_fn_Sales_Product_ASN_Item {
        summary summary                 "Data is more than 1 milions"
        string  Sales_Domain_ShipTo    "Lv6 Lv7"
        string  Sales_Domain_LV2
        string  Sales_Domain_LV3
        string  Sales_Domain_LV6        "For Join with Class"
        string  Item_Item
        string  Item_GBM                "Step:07: Merged with Item Master"
        string  Product_Group           "Step:07: Merged with Item Master"
        string  Location_Location
        string  Sales_Product_ASN
        
    }



    df_in_Item_TAT {
        string  Item_Item
        string  Location_Location
        int     ITEMTAT_TATTERM
        int     ITEMTAT_TATTERM_SET

    }

    df_in_Forecast_Rule {
        string  Product_Group
        string  Sales_Domain_ShipTo    "level2 or level3"
        int     FORECAST_RULE_GC_FCST
        int     FORECAST_RULE_AP2_FCST
        int     FORECAST_RULE_AP1_FCST
        string  FORECAST_RULE_ISVALID
    }

    df_fn_Sales_Product_ASN_Item_Week {
        string  Sales_Domain_ShipTo    "Lv6 Lv7"
        string  Sales_Domain_LV2
        string  Sales_Domain_LV3
        string  Sales_Domain_LV6        "For Join with Class"
        string  Product_Group
        string  Item_Item
        string  RTS_EOS_ShipTo          "level2 or level3"
        int     Item_lv                 "2 or 3"
        string  Item_GBM    
        string  Location_Location  
        string  Partial_Week 
        bool    SIn_FCST_GC_LOCK           ""
        string  SIn_FCST_Color_Condition     ""   
    }


    %% 최종  df_output_Sell_In
    df_output_Sell_In_FCST_GI_GC_Lock {
        string  Version_Name
        string  Item_Item
        string  Sales_Domain_ShipTo    "level declared in df_in_Forecast_Rule(FORECAST_RULE_GC_FCST,FORECAST_RULE_AP2_FCST,FORECAST_RULE_AP1_FCST)"
        string  Location_Location  
        string  Partial_Week 
        bool    SIn_FCST_GC_LOCK        "see FORECAST_RULE_GC_FCST"
    }

    


    %% 최종  df_output_Sell_Out
    df_output_Sell_Out_FCST_GC_Lock {
        string  Version_Name
        string  Item_Item
        string  Sales_Domain_ShipTo    "level declared in df_in_Forecast_Rule(FORECAST_RULE_GC_FCST,FORECAST_RULE_AP2_FCST,FORECAST_RULE_AP1_FCST)"
        string  Location_Location  
        string  Partial_Week 
        bool    SOut_FCST_GC_LOCK        "see FORECAST_RULE_GC_FCST"
    }

    df_in_MST_RTS_EOS ||--|| df_fn_RTS_EOS : "Step2 Join Item_Item,Sales_Domain_ShipTo"

    df_fn_RTS_EOS || -- |{ df_fn_RTS_EOS_Week : "Cross Join in Step04"
    df_in_Time_Partial_Week || -- |{ df_fn_RTS_EOS_Week : "Step05: Cross Join. Adjust Lock,Condition"
    df_in_Item_Master || -- |{ df_fn_RTS_EOS_Week : "Step06: 13_GREEN For BAS MOBILE"

    df_in_Item_Master || -- |{ df_fn_Sales_Product_ASN_Item_Week : ""
    df_in_Sales_Product_ASN || -- |{ df_fn_Sales_Product_ASN_Item_Week   : "Step:07"
    df_in_Sales_Domain_Dimension || -- |{ df_fn_Sales_Product_ASN_Item_Week : "Lv2,3 추가 For Join With df_fn_RTS_EOS_Week"
    df_in_Item_CLASS || -- |{ df_fn_Sales_Product_ASN_Item_Week : ""

    df_fn_RTS_EOS_Week || -- |{ df_fn_Sales_Product_ASN_Item_Week : "Step08"

    df_in_Time_Partial_Week || -- |{ df_fn_Sales_Product_ASN_Item_Week : "Step08: 주차추가"
    

    df_in_Item_TAT || -- |{ df_fn_Sales_Product_ASN_Item_Week : " Item_Item :: Location_Location"
    

    df_in_Sales_Domain_Dimension || -- |{ df_in_Forecast_Rule : ""

    %% 최종  df_output_Sell_In
    df_fn_Sales_Product_ASN_Item_Week || -- |{ df_output_Sell_In_FCST_GI_GC_Lock : ""
    df_in_Forecast_Rule || -- |{ df_output_Sell_In_FCST_GI_GC_Lock : "Product_Group :: Sales_Domain_ShipTo"

    %% 최종  df_output_Sell_Out
    df_fn_Sales_Product_ASN_Item_Week || -- |{ df_output_Sell_Out_FCST_GC_Lock : ""
    df_in_Forecast_Rule || -- |{ df_output_Sell_Out_FCST_GC_Lock : "Product_Group :: Sales_Domain_ShipTo"



```

<br><br><br><br>
# ERD  AP2_Lock
``` mermaid
erDiagram

    df_in_Forecast_Rule {
        string  Product_Group
        string  Sales_Domain_ShipTo    "level2 or level3"
        int     FORECAST_RULE_GC_FCST
        int     FORECAST_RULE_AP2_FCST
        int     FORECAST_RULE_AP1_FCST
        string  FORECAST_RULE_ISVALID
    }

    df_fn_Sales_Product_ASN_Item_Week {
        string  Sales_Domain_ShipTo    "Lv6 Lv7"
        string  Sales_Domain_LV2
        string  Sales_Domain_LV3
        string  Sales_Domain_LV6        "For Join with Class"
        string  Product_Group
        string  Item_Item
        string  RTS_EOS_ShipTo          "level2 or level3"
        int     Item_lv                 "2 or 3"
        string  Item_GBM    
        string  Location_Location  
        string  Partial_Week 
        bool    SIn_FCST_GC_LOCK           ""
        string  SIn_FCST_Color_Condition     ""   
    }


    %% 최종  df_output_Sell_In
    df_output_Sell_In_FCST_GI_AP2_Lock {
        string  Version_Name
        string  Item_Item
        string  Sales_Domain_ShipTo    "level declared in df_in_Forecast_Rule(FORECAST_RULE_GC_FCST,FORECAST_RULE_AP2_FCST,FORECAST_RULE_AP1_FCST)"
        string  Location_Location  
        string  Partial_Week 
        bool    SIn_FCST_AP2_LOCK        "see FORECAST_RULE_AP2_FCST"
    }



    %% 최종  df_output_Sell_Out
    df_output_Sell_Out_FCST_AP2_Lock {
        string  Version_Name
        string  Item_Item
        string  Sales_Domain_ShipTo    "level declared in df_in_Forecast_Rule(FORECAST_RULE_GC_FCST,FORECAST_RULE_AP2_FCST,FORECAST_RULE_AP1_FCST)"
        string  Location_Location  
        string  Partial_Week 
        bool    SOut_FCST_AP2_LOCK        "see FORECAST_RULE_AP2_FCST"
    }



    %% 최종  df_output_Sell_In
    df_fn_Sales_Product_ASN_Item_Week || -- |{ df_output_Sell_In_FCST_GI_AP2_Lock : ""
    df_in_Forecast_Rule || -- |{ df_output_Sell_In_FCST_GI_AP2_Lock : "Product_Group :: Sales_Domain_ShipTo"



    %% 최종  df_output_Sell_Out
    df_fn_Sales_Product_ASN_Item_Week || -- |{ df_output_Sell_Out_FCST_AP2_Lock : ""
    df_in_Forecast_Rule || -- |{ df_output_Sell_Out_FCST_AP2_Lock : "Product_Group :: Sales_Domain_ShipTo"


```

<br><br><br><br>
# ERD  AP1_Lock
``` mermaid
erDiagram

    df_in_Forecast_Rule {
        string  Product_Group
        string  Sales_Domain_ShipTo    "level2 or level3"
        int     FORECAST_RULE_GC_FCST
        int     FORECAST_RULE_AP2_FCST
        int     FORECAST_RULE_AP1_FCST
        string  FORECAST_RULE_ISVALID
    }

    df_fn_Sales_Product_ASN_Item_Week {
        string  Sales_Domain_ShipTo    "Lv6 Lv7"
        string  Sales_Domain_LV2
        string  Sales_Domain_LV3
        string  Sales_Domain_LV6        "For Join with Class"
        string  Product_Group
        string  Item_Item
        string  RTS_EOS_ShipTo          "level2 or level3"
        int     Item_lv                 "2 or 3"
        string  Item_GBM    
        string  Location_Location  
        string  Partial_Week 
        bool    SIn_FCST_GC_LOCK           ""
        string  SIn_FCST_Color_Condition     ""   
    }


    %% 최종  df_output_Sell_In
    df_output_Sell_In_FCST_GI_AP1_Lock {
        string  Version_Name
        string  Item_Item
        string  Sales_Domain_ShipTo    "level declared in df_in_Forecast_Rule(FORECAST_RULE_GC_FCST,FORECAST_RULE_AP2_FCST,FORECAST_RULE_AP1_FCST)"
        string  Location_Location  
        string  Partial_Week 
        bool    SIn_FCST_AP1_LOCK        "see FORECAST_RULE_AP1_FCST"
    }



    %% 최종  df_output_Sell_Out
    df_output_Sell_Out_FCST_AP1_Lock {
        string  Version_Name
        string  Item_Item
        string  Sales_Domain_ShipTo    "level declared in df_in_Forecast_Rule(FORECAST_RULE_GC_FCST,FORECAST_RULE_AP2_FCST,FORECAST_RULE_AP1_FCST)"
        string  Location_Location  
        string  Partial_Week 
        bool    SOut_FCST_AP1_LOCK        "see FORECAST_RULE_AP1_FCST"
    }



    %% 최종  df_output_Sell_In
    df_fn_Sales_Product_ASN_Item_Week || -- |{ df_output_Sell_In_FCST_GI_AP1_Lock : ""
    df_in_Forecast_Rule || -- |{ df_output_Sell_In_FCST_GI_AP1_Lock : "Product_Group :: Sales_Domain_ShipTo"


    %% 최종  df_output_Sell_Out
    df_fn_Sales_Product_ASN_Item_Week || -- |{ df_output_Sell_Out_FCST_AP1_Lock : ""
    df_in_Forecast_Rule || -- |{ df_output_Sell_Out_FCST_AP1_Lock : "Product_Group :: Sales_Domain_ShipTo"
```

<br><br><br><br>
# ERD  _FCST_Color
``` mermaid
erDiagram

    df_in_Forecast_Rule {
        string  Product_Group
        string  Sales_Domain_ShipTo    "level2 or level3"
        int     FORECAST_RULE_GC_FCST
        int     FORECAST_RULE_AP2_FCST
        int     FORECAST_RULE_AP1_FCST
        string  FORECAST_RULE_ISVALID
    }

    df_fn_Sales_Product_ASN_Item_Week {
        string  Sales_Domain_ShipTo    "Lv6 Lv7"
        string  Sales_Domain_LV2
        string  Sales_Domain_LV3
        string  Sales_Domain_LV6        "For Join with Class"
        string  Product_Group
        string  Item_Item
        string  RTS_EOS_ShipTo          "level2 or level3"
        int     Item_lv                 "2 or 3"
        string  Item_GBM    
        string  Location_Location  
        string  Partial_Week 
        bool    SIn_FCST_GC_LOCK           ""
        string  SIn_FCST_Color_Condition     ""   
    }


    %% 최종  df_output_Sell_In

    df_output_Sell_In_FCST_Color_Condition {
        string  Item_Item
        string  Sales_Domain_ShipTo    "level declared in df_in_Forecast_Rule(FORECAST_RULE_GC_FCST,FORECAST_RULE_AP2_FCST,FORECAST_RULE_AP1_FCST)"
        string  Location_Location  
        string  Partial_Week 
        string  SIn_FCST_Color_Condition     ""   
        
    }


    %% 최종  df_output_Sell_Out

    df_output_Sell_Out_FCST_Color_Condition {
        string  Item_Item
        string  Sales_Domain_ShipTo    "level declared in df_in_Forecast_Rule(FORECAST_RULE_GC_FCST,FORECAST_RULE_AP2_FCST,FORECAST_RULE_AP1_FCST)"
        string  Location_Location  
        string  Partial_Week 
        string  SOut_FCST_Color_Condition     ""   
        
    }


    %% 최종  df_output_Sell_In
    df_fn_Sales_Product_ASN_Item_Week || -- |{ df_output_Sell_In_FCST_Color_Condition : ""
    df_in_Forecast_Rule || -- |{ df_output_Sell_In_FCST_Color_Condition : "Product_Group :: Sales_Domain_ShipTo"


    %% 최종  df_output_Sell_Out
    df_fn_Sales_Product_ASN_Item_Week || -- |{ df_output_Sell_Out_FCST_Color_Condition : ""
    df_in_Forecast_Rule || -- |{ df_output_Sell_Out_FCST_Color_Condition : "Product_Group :: Sales_Domain_ShipTo"

```