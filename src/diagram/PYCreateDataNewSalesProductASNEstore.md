# ERD  GC_Lock
``` mermaid
erDiagram
    %% Input Tables
    df_in_Sales_Domain_Dimension {
        category  COL_STD1
        category  COL_STD2
        category  COL_STD3
        category  COL_STD4
        category  COL_STD5
        category  COL_STD6
        category  COL_SHIP_TO           "PK"
    }

    df_in_Time {
        category  COL_TIME_WK
    }

    df_in_Sales_Domain_Estore {
        category  COL_SHIP_TO           ""  
    }

    
    
    df_in_Sales_Product_ASN_Delta {
        category  COL_VERSION             
        category  COL_SHIP_TO             "PK: Std5 or Std6"
        category  COL_ITEM                "PK" 
        category  COL_LOC                 "PK"
        category  COL_PG_ASN_DELTA        ""
    }
    %% 여기서 COL_SHIP_TO 은 'A5' 또는 '5' 로 시작한다.


    df_in_Item_Master {
        category    COL_ITEM_GBM
        category    COL_ITEM_STD1
        category    COL_ITEM                "PK"
    }

    df_in_Sell_In_User_GI_Ratio {
        category  COL_VERSION
        category  COL_ITEM_GBM              "PK"
        category  COL_ITEM_STD1             "PK"
        category  COL_LOC                   "PK"
        category  COL_TIME_WK               "PK"      
        float64   COL_SIN_USER_LONG         "0"       
        float64   COL_SIN_USER_W7           "0"
        float64   COL_SIN_USER_W6           "0"
        float64   COL_SIN_USER_W5           "0"
        float64   COL_SIN_USER_W4           "0"
        float64   COL_SIN_USER_W3           "0"
        float64   COL_SIN_USER_W2           "0"
        float64   COL_SIN_USER_W1           "0"
        float64   COL_SIN_USER_W0           "0"
    }


    df_in_Sell_In_Issue_GI_Ratio {
        category  COL_VERSION
        category  COL_ITEM_GBM              "PK"
        category  COL_ITEM_STD1             "PK"
        category  COL_LOC                   "PK"
        category  COL_TIME_WK               "PK"            
        float64   COL_SIN_ISSUE_LONG        "0"       
        float64   COL_SIN_ISSUE_W7          "0"
        float64   COL_SIN_ISSUE_W6          "0"
        float64   COL_SIN_ISSUE_W5          "0"
        float64   COL_SIN_ISSUE_W4          "0"
        float64   COL_SIN_ISSUE_W3          "0"
        float64   COL_SIN_ISSUE_W2          "0"
        float64   COL_SIN_ISSUE_W1          "0"
        float64   COL_SIN_ISSUE_W0          "0"
    }

    df_in_Sell_In_Final_GI_Ratio {
        category  COL_VERSION
        category  COL_ITEM_GBM              "PK"
        category  COL_ITEM_STD1             "PK"
        category  COL_LOC                   "PK"
        category  COL_TIME_WK               "PK"            
        float64   COL_SIN_FINAL_LONG        "0"       
        float64   COL_SIN_FINAL_W7          "0"
        float64   COL_SIN_FINAL_W6          "0"
        float64   COL_SIN_FINAL_W5          "0"
        float64   COL_SIN_FINAL_W4          "0"
        float64   COL_SIN_FINAL_W3          "0"
        float64   COL_SIN_FINAL_W2          "0"
        float64   COL_SIN_FINAL_W1          "0"
        float64   COL_SIN_FINAL_W0          "0"
    }

    df_in_Sell_In_BestFit_GI_Ratio {
        category  COL_VERSION
        category  COL_ITEM_GBM              "PK"
        category  COL_ITEM_STD1             "PK"
        category  COL_LOC                   "PK"
        category  COL_TIME_WK               "PK"            
        float64   COL_SIN_BETST_LONG        "0"       
        float64   COL_SIN_BETST_W7          "0"
        float64   COL_SIN_BETST_W6          "0"
        float64   COL_SIN_BETST_W5          "0"
        float64   COL_SIN_BETST_W4          "0"
        float64   COL_SIN_BETST_W3          "0"
        float64   COL_SIN_BETST_W2          "0"
        float64   COL_SIN_BETST_W1          "0"
        float64   COL_SIN_BETST_W0          "0"
    }

    df_in_Sell_In_User_Item_GI_Ratio {
        category  COL_VERSION
        category  COL_ITEM_GBM              "PK"
        category  COL_ITEM_STD1             "PK"
        category  COL_LOC                   "PK"
        category  COL_TIME_WK               "PK"            
        float64   COL_SIN_USER_ITEM_LONG    "0"       
        float64   COL_SIN_USER_ITEM_W7      "0"
        float64   COL_SIN_USER_ITEM_W6      "0"
        float64   COL_SIN_USER_ITEM_W5      "0"
        float64   COL_SIN_USER_ITEM_W4      "0"
        float64   COL_SIN_USER_ITEM_W3      "0"
        float64   COL_SIN_USER_ITEM_W2      "0"
        float64   COL_SIN_USER_ITEM_W1      "0"
        float64   COL_SIN_USER_ITEM_W0      "0"
    }

    df_in_Sell_In_Issue_Item_GI_Ratio {
        category  COL_VERSION
        category  COL_ITEM_GBM              "PK"
        category  COL_ITEM_STD1             "PK"
        category  COL_LOC                   "PK"
        category  COL_TIME_WK               "PK"            
        float64   COL_SIN_ISSUE_ITEM_LONG   "0"       
        float64   COL_SIN_ISSUE_ITEM_W7     "0"
        float64   COL_SIN_ISSUE_ITEM_W6     "0"
        float64   COL_SIN_ISSUE_ITEM_W5     "0"
        float64   COL_SIN_ISSUE_ITEM_W4     "0"
        float64   COL_SIN_ISSUE_ITEM_W3     "0"
        float64   COL_SIN_ISSUE_ITEM_W2     "0"
        float64   COL_SIN_ISSUE_ITEM_W1     "0"
        float64   COL_SIN_ISSUE_ITEM_W0     "0"
    }

    %% Middle Tables
    df_fn_Sales_Product_ASN_Delta {
        category  COL_VERSION             
        category  COL_SHIP_TO             "PK: Std5 or Std6"
        category  COL_ITEM                "PK" 
        category  COL_LOC                 "PK"
        category  COL_PG_ASN_DELTA        ""
    }


    %% output Tables

    df_output_Sell_In_User_GI_Ratio {
        category  COL_VERSION
        category  COL_SHIP_TO              "PK"
        category  COL_ITEM                 "PK"
        category  COL_LOC                  "PK"
        category  COL_TIME_WK              "PK"            
        float64   COL_SIN_USER_LONG        "0"       
        float64   COL_SIN_USER_W7          "0"
        float64   COL_SIN_USER_W6          "0"
        float64   COL_SIN_USER_W5          "0"
        float64   COL_SIN_USER_W4          "0"
        float64   COL_SIN_USER_W3          "0"
        float64   COL_SIN_USER_W2          "0"
        float64   COL_SIN_USER_W1          "0"
        float64   COL_SIN_USER_W0          "0"
    }

    df_output_Sell_In_Issue_GI_Ratio {
        category  COL_VERSION
        category  COL_SHIP_TO               "PK"
        category  COL_ITEM                  "PK"
        category  COL_LOC                   "PK"
        category  COL_TIME_WK               "PK"            
        float64   COL_SIN_ISSUE_LONG        "0"       
        float64   COL_SIN_ISSUE_W7          "0"
        float64   COL_SIN_ISSUE_W6          "0"
        float64   COL_SIN_ISSUE_W5          "0"
        float64   COL_SIN_ISSUE_W4          "0"
        float64   COL_SIN_ISSUE_W3          "0"
        float64   COL_SIN_ISSUE_W2          "0"
        float64   COL_SIN_ISSUE_W1          "0"
        float64   COL_SIN_ISSUE_W0          "0"
    }

    df_output_Sell_In_Final_GI_Ratio {
        category  COL_VERSION
        category  COL_SHIP_TO               "PK"
        category  COL_ITEM                  "PK"
        category  COL_LOC                   "PK"
        category  COL_TIME_WK               "PK"            
        float64   COL_SIN_FINAL_LONG        "0"       
        float64   COL_SIN_FINAL_W7          "0"
        float64   COL_SIN_FINAL_W6          "0"
        float64   COL_SIN_FINAL_W5          "0"
        float64   COL_SIN_FINAL_W4          "0"
        float64   COL_SIN_FINAL_W3          "0"
        float64   COL_SIN_FINAL_W2          "0"
        float64   COL_SIN_FINAL_W1          "0"
        float64   COL_SIN_FINAL_W0          "0"
    }

    df_output_Sell_In_BestFit_GI_Ratio {
        category  COL_VERSION
        category  COL_SHIP_TO               "PK"
        category  COL_ITEM                  "PK"
        category  COL_LOC                   "PK"
        category  COL_TIME_WK               "PK"              
        float64   COL_SIN_BETST_LONG        "0"       
        float64   COL_SIN_BETST_W7          "0"
        float64   COL_SIN_BETST_W6          "0"
        float64   COL_SIN_BETST_W5          "0"
        float64   COL_SIN_BETST_W4          "0"
        float64   COL_SIN_BETST_W3          "0"
        float64   COL_SIN_BETST_W2          "0"
        float64   COL_SIN_BETST_W1          "0"
        float64   COL_SIN_BETST_W0          "0"
    }

    df_output_Sell_In_User_Item_GI_Ratio {
        category  COL_VERSION
        category  COL_SHIP_TO               "PK"
        category  COL_ITEM                  "PK"
        category  COL_LOC                   "PK"
        category  COL_TIME_WK               "PK"            
        float64   COL_SIN_USER_ITEM_LONG    "0"       
        float64   COL_SIN_USER_ITEM_W7      "0"
        float64   COL_SIN_USER_ITEM_W6      "0"
        float64   COL_SIN_USER_ITEM_W5      "0"
        float64   COL_SIN_USER_ITEM_W4      "0"
        float64   COL_SIN_USER_ITEM_W3      "0"
        float64   COL_SIN_USER_ITEM_W2      "0"
        float64   COL_SIN_USER_ITEM_W1      "0"
        float64   COL_SIN_USER_ITEM_W0      "0"
    }

    df_output_Sell_In_Issue_Item_GI_Ratio {
        category  COL_VERSION
        category  COL_SHIP_TO               "PK"
        category  COL_ITEM                  "PK"
        category  COL_LOC                   "PK"
        category  COL_TIME_WK               "PK"             
        float64   COL_SIN_ISSUE_ITEM_LONG   "0"       
        float64   COL_SIN_ISSUE_ITEM_W7     "0"
        float64   COL_SIN_ISSUE_ITEM_W6     "0"
        float64   COL_SIN_ISSUE_ITEM_W5     "0"
        float64   COL_SIN_ISSUE_ITEM_W4     "0"
        float64   COL_SIN_ISSUE_ITEM_W3     "0"
        float64   COL_SIN_ISSUE_ITEM_W2     "0"
        float64   COL_SIN_ISSUE_ITEM_W1     "0"
        float64   COL_SIN_ISSUE_ITEM_W0     "0"
    }
    


    %% Relations

    df_in_Sales_Product_ASN_Delta   || -- |{ df_fn_Sales_Product_ASN_Delta          : ""
    df_in_Sales_Domain_Estore       || -- |{ df_fn_Sales_Product_ASN_Delta          : ""


    df_in_Time                      || -- |{ df_output_Sell_In_User_GI_Ratio        :   ""   
    df_in_Time                      || -- |{ df_output_Sell_In_Issue_GI_Ratio       :   ""
    df_in_Time                      || -- |{ df_output_Sell_In_Final_GI_Ratio       :   ""
    df_in_Time                      || -- |{ df_output_Sell_In_BestFit_GI_Ratio     :   ""
    df_in_Time                      || -- |{ df_output_Sell_In_User_Item_GI_Ratio   :   ""
    df_in_Time                      || -- |{ df_output_Sell_In_Issue_Item_GI_Ratio  :   ""



    df_fn_Sales_Product_ASN_Delta   || -- |{ df_output_Sell_In_User_GI_Ratio        :   ""     
    df_fn_Sales_Product_ASN_Delta   || -- |{ df_output_Sell_In_Issue_GI_Ratio       :   ""
    df_fn_Sales_Product_ASN_Delta   || -- |{ df_output_Sell_In_Final_GI_Ratio       :   ""
    df_fn_Sales_Product_ASN_Delta   || -- |{ df_output_Sell_In_BestFit_GI_Ratio     :   ""
    df_fn_Sales_Product_ASN_Delta   || -- |{ df_output_Sell_In_User_Item_GI_Ratio   :   ""
    df_fn_Sales_Product_ASN_Delta   || -- |{ df_output_Sell_In_Issue_Item_GI_Ratio  :   ""

    
    df_in_Item_Master   || -- |{ df_output_Sell_In_User_GI_Ratio        :   ""     
    df_in_Item_Master   || -- |{ df_output_Sell_In_Issue_GI_Ratio       :   ""
    df_in_Item_Master   || -- |{ df_output_Sell_In_Final_GI_Ratio       :   ""
    df_in_Item_Master   || -- |{ df_output_Sell_In_BestFit_GI_Ratio     :   ""
    df_in_Item_Master   || -- |{ df_output_Sell_In_User_Item_GI_Ratio   :   ""
    df_in_Item_Master   || -- |{ df_output_Sell_In_Issue_Item_GI_Ratio  :   ""


    

    df_in_Sell_In_User_GI_Ratio         || -- |{ df_output_Sell_In_User_GI_Ratio        :   ""     
    df_in_Sell_In_Issue_GI_Ratio        || -- |{ df_output_Sell_In_Issue_GI_Ratio       :   ""
    df_in_Sell_In_Final_GI_Ratio        || -- |{ df_output_Sell_In_Final_GI_Ratio       :   ""
    df_in_Sell_In_BestFit_GI_Ratio      || -- |{ df_output_Sell_In_BestFit_GI_Ratio     :   ""
    df_in_Sell_In_User_Item_GI_Ratio    || -- |{ df_output_Sell_In_User_Item_GI_Ratio   :   ""
    df_in_Sell_In_Issue_Item_GI_Ratio   || -- |{ df_output_Sell_In_Issue_Item_GI_Ratio  :   ""

```
