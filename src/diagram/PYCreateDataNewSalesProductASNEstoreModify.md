
# ERD  
``` mermaid
erDiagram

    df_in_Sales_Domain_Estore {
        category  COL_SHIP_TO           ""  
    }

    df_in_Time {
        category  COL_TIME_WK
    }
    
    df_in_Sales_Product_ASN_Delta {
        category  COL_VERSION             
        category  COL_SHIP_TO             "PK: Std5 or Std6"
        category  COL_ITEM                "PK" 
        category  COL_LOC                 "PK"
        category  COL_PG_ASN_DELTA        ""
    }
    %% 여기서 COL_SHIP_TO 은 'A5' 또는 '5' 로 시작한다.

    df_fn_Sales_Product_ASN_Delta {
        category  COL_VERSION             
        category  COL_SHIP_TO             "PK: Std5 or Std6"
        category  COL_ITEM                "PK" 
        category  COL_LOC                 "PK"
        category  COL_PG_ASN_DELTA        ""
    }

    df_output_Sell_In_User_Modify_GI_Ratio {
        category  COL_VERSION
        category  COL_SHIP_TO             "PK: Std5 or Std6"
        category  COL_ITEM                "PK" 
        category  COL_LOC                 "PK"
        category  COL_TIME_WK             "PK"      
        float32   COL_SIN_USER_LONG       "0"       
        float32   COL_SIN_USER_W7         "0"
        float32   COL_SIN_USER_W6         "0"
        float32   COL_SIN_USER_W5         "0"
        float32   COL_SIN_USER_W4         "0"
        float32   COL_SIN_USER_W3         "0"
        float32   COL_SIN_USER_W2         "0"
        float32   COL_SIN_USER_W1         "0"
        float32   COL_SIN_USER_W0         "0"
    }


    df_output_Sell_In_Issue_Modify_GI_Ratio {
        category  COL_VERSION
        category  COL_SHIP_TO              "PK: Std5 or Std6"
        category  COL_ITEM                 "PK" 
        category  COL_LOC                  "PK"
        category  COL_TIME_WK              "PK"       
        float32   COL_SIN_ISSUE_LONG       "0"       
        float32   COL_SIN_ISSUE_W7         "0"
        float32   COL_SIN_ISSUE_W6         "0"
        float32   COL_SIN_ISSUE_W5         "0"
        float32   COL_SIN_ISSUE_W4         "0"
        float32   COL_SIN_ISSUE_W3         "0"
        float32   COL_SIN_ISSUE_W2         "0"
        float32   COL_SIN_ISSUE_W1         "0"
        float32   COL_SIN_ISSUE_W0         "0"
    }
    
    df_in_Sales_Domain_Estore       || -- |{ df_fn_Sales_Product_ASN_Delta          : ""
    df_in_Sales_Product_ASN_Delta   || -- |{ df_fn_Sales_Product_ASN_Delta          : ""


    df_in_Time                      || -- |{ df_output_Sell_In_User_Modify_GI_Ratio : ""
    df_fn_Sales_Product_ASN_Delta   || -- |{ df_output_Sell_In_User_Modify_GI_Ratio : ""

    df_in_Time                      || -- |{ df_output_Sell_In_Issue_Modify_GI_Ratio : ""
    df_fn_Sales_Product_ASN_Delta   || -- |{ df_output_Sell_In_Issue_Modify_GI_Ratio : ""

```


# Sequence
``` mermaid
%% PYCreateDataNewSalesProductASNEstoreModify – Sequence (entities only on top)
sequenceDiagram
    %% ──────────────── Entity Participants ────────────────
    participant TimeDF          as df_in_Time
    participant EstoreDF        as df_in_Sales_Domain_Estore
    participant DeltaDF         as df_in_Sales_Product_ASN_Delta
    participant ASN_DF          as df_fn_Sales_Product_ASN_Delta
    participant OutUser         as df_output_Sell_In_User_Modify_GI_Ratio
    participant OutIssue        as df_output_Sell_In_Issue_Modify_GI_Ratio    

    %% ──────────────── Step-1 : 필터링 ────────────────
    DeltaDF->>DeltaDF: fn_step01_filter_asn_delta()<br/>• flag 'Y' 필터
    DeltaDF->>EstoreDF: Ship-To 존재 확인 (inner)
    DeltaDF-->>ASN_DF: 필터링 결과

    alt ASN_DF empty
        Note over ASN_DF: 프로그램 종료 (출력 없음)
    else ASN_DF not empty
        %% ───────────── Step-2 : USER GI Ratio ─────────────
        loop fn_step02_create_user_modify_ratio <br/>•각 ShipTo-Item-Loc in ASN_DF
            ASN_DF->>TimeDF: iterate COL_TIME_WK
            TimeDF-->>OutUser: row append (GI=0, Version='CWV_DP')
        end

        %% ───────────── Step-3 : ISSUE GI Ratio ────────────
        OutUser-->>OutIssue: fn_step03_create_issue_modify_ratio()<br/>• 컬럼 rename
    end

```