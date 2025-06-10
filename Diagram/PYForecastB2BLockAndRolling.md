# ERD  GC_Lock
::: mermaid
erDiagram
    df_in_BO_FCST {
        category  COL_VERSION
        category  COL_SHIP_TO           "Std5"
        category  COL_ITEM              
        category  COL_LOC               
        category  COL_VIRTUAL_BO_ID
        category  COL_BO_ID
        category  COL_TIME_PW
        int32     COL_BO_FCST
        bool      COL_BO_FCST_LOCK
    }


    df_In_Total_BOD_LT {
        category  COL_VERSION
        category  COL_ITEM
        category  COL_LOC
        int32     COL_BO_TOTAL_BOD_LT
    }

    df_fn_total_bod_lt {
        category  COL_ITEM
        category  COL_LOC
        category  COL_BO_TOTAL_BOD_LT   "202532A = (182 / 7) + 202506A"
    }

    df_In_MAX_PartialWeek {
        category  COL_TIME_PW
    }

    df_in_Time_Partial_Week {
        category  COL_TIME_PW
    }

    df_in_Sales_Domain_Dimension {
        category  COL_STD1
        category  COL_STD2
        category  COL_STD3
        category  COL_STD4
        category  COL_STD5
        category  COL_STD6
        category  COL_SHIP_TO           "PK"
    }

    df_in_MST_RTS_EOS {
        category  COL_VERSION
        category  COL_ITEM
        category  COL_SHIP_TO    "Std1 or Std2"
        category  RTS_STATUS
        category  RTS_ISVALID
        category  RTS_INIT_DATE
        category  RTS_DEV_DATE
        category  RTS_COM_DATE 
        category  EOS_INIT_DATE
        category  EOS_CHG_DATE
        category  EOS_COM_DATE 
    }

    df_in_Sales_Product_ASN {
        category  COL_VERSION             
        category  COL_SHIP_TO             "PK: Std5 or Std6"
        category  COL_ITEM                "PK" 
        category  COL_LOC                 "PK"
        category  Sales_Product_ASN       "Y or N"
    }

    %%  Joined Table
    df_fn_rts_eos {
        category    COL_ITEM              "PK"
        category    COL_SHIP_TO           "PK : Std1 or Std2"
        category    RTS_STATUS        
        category    RTS_INIT_DATE
        category    RTS_DEV_DATE
        category    RTS_COM_DATE 
        category    EOS_STATUS
        category    EOS_INIT_DATE           
        category    EOS_CHG_DATE
        category    EOS_COM_DATE 
        category    RTS_PARTIAL_WEEK     "Step1-2: Helper Column"
        category    EOS_PARTIAL_WEEK     "Step1-2: Helper Column"
        category    RTS_WEEK             "Step1-2: Helper Column"
        category    RTS_WEEK_MINUS_1     "Step1-2: Helper Column"
        category    RTS_WEEK_PLUS_3      "Step1-2: Helper Column"
        category    MAX_RTS_CURRENTWEEK  "Step1-2: Helper Column"
        category    EOS_WEEK             "Step1-2: Helper Column"
        category    EOS_WEEK_MINUS_1     "Step1-2: Helper Column"
        category    EOS_WEEK_MINUS_4     "Step1-2: Helper Column"
        category    RTS_INITIAL_WEEK     "Step1-2: Helper Column"
        category    EOS_INITIAL_WEEK     "Step1-2: Helper Column"
        category    MIN_EOSINI_MAXWEEK   "Step1-2: Helper Column"

    }






    %% middle df. created in step1-4.
    df_fn_rts_eos_pw {
        category  COL_SHIP_TO                     "PK : Lv3 or Lv2"
        category  COL_ITEM                        "PK"
        category  COL_TIME_PW                     "PK: Step 01-4"
        category  CURRENT_ROW_WEEK                "Step 01-4: Added : A나 B를 뺀 주차숫자만 표시. 계산시 사용"
        bool      COL_BO_FCST_LOCK                "Step 01-4: True"
        category  COL_BO_FCST_COLOR_COND          "Step 01-4: 19_GRAY"
        
    }


    df_fn_sales_product_asn_item_week {
        category  COL_SHIP_TO                     "PK: come from df_in_Sales_Product_ASN"
        category  COL_ITEM                        "PK: come from df_in_Sales_Product_ASN"
        category  COL_LOC                         "PK: come from df_in_Sales_Product_ASN"
        category  COL_TIME_PW                     "PK"
        bool      COL_BO_FCST_LOCK           
        category  COL_BO_FCST_COLOR_COND   
    }


    %% Step2
    df_fn_bo_fcst_asn {
        category  COL_ITEM
        category  COL_SHIP_TO           "Std5"
        category  COL_LOC               
        category  COL_VIRTUAL_BO_ID
        category  COL_BO_ID             
        category  COL_TIME_PW               ""
        int32     COL_BO_FCST               "default 0"
        bool      COL_BO_FCST_LOCK
        category  COL_BO_FCST_COLOR_COND    "default Null"
    }


    %% 최종  df_output_Sell_Out
    out_Demand {
        category      COL_VERSION
        category      COL_ITEM
        category      COL_SHIP_TO    
        category      COL_LOC  
        category      COL_VIRTUAL_BO_ID
        category      COL_BO_ID
        category      COL_TIME_PW 
        int32         COL_BO_FCST
        bool          COL_BO_FCST_LOCK    
        category      COL_BO_FCST_COLOR_COND    
    }

    df_in_Sales_Domain_Dimension || .. o{ df_in_MST_RTS_EOS : ""
    df_in_Sales_Domain_Dimension || .. o{ df_in_Sales_Product_ASN : ""
    df_in_Sales_Domain_Dimension || .. o{ df_in_BO_FCST : ""

    df_in_MST_RTS_EOS ||--|| df_fn_rts_eos : "Step2 Join COL_ITEM,COL_SHIP_TO"

    df_fn_rts_eos || -- |{ df_fn_rts_eos_pw : "Cross Join in Step04"
    
    df_in_Time_Partial_Week || -- |{ df_fn_rts_eos_pw : "Step1-5: Cross Join. Adjust Lock,Condition"

    df_in_Sales_Product_ASN || -- |{ df_fn_sales_product_asn_item_week   : "Step1-7"
    

    df_fn_rts_eos_pw || -- |{ df_fn_sales_product_asn_item_week : "Step1-8"

    df_in_Time_Partial_Week || -- |{ df_fn_sales_product_asn_item_week : "Step1-8: 주차추가"

    %% left join: left is df_fn_sales_product_asn_item_week . join columns: COL_SHIP_TO,COL_ITEM,COL_LOC,COL_TIME_PW
    df_in_BO_FCST || -- |{ df_fn_bo_fcst_asn            : "Step2"
    df_fn_sales_product_asn_item_week || -- |{ df_fn_bo_fcst_asn : "Step2: "


    %% Step3
    df_In_Total_BOD_LT || -- |{ df_fn_total_bod_lt  : "Step3-1"
    df_fn_total_bod_lt || .. o{ df_fn_bo_fcst_asn   : "COL_ITEM , COL_LOC"


    %% 최종  df_output_Sell_Out
    df_fn_bo_fcst_asn || -- |{ out_Demand : ""



:::
