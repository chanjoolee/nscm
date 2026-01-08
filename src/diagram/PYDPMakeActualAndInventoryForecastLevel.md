# 1. S/In Actual GI 관련 ERD
``` mermaid
erDiagram
    %% --- 공통 기준 테이블 ----------------------------------------------------
    df_in_Forecast_Rule {
        string  Version_Name
        string  Product_Group
        string  Sales_Domain_ShipTo
        int     FORECAST_RULE_GC_FCST
        int     FORECAST_RULE_AP2_FCST
        int     FORECAST_RULE_AP1_FCST
    }
    df_in_Sales_Domain_Dimension {
        string  Sales_Domain_Std1
        string  Sales_Domain_Std2
        string  Sales_Domain_Std3
        string  Sales_Domain_Std4
        string  Sales_Domain_Std5
        string  Sales_Domain_Std6
        string  Sales_Domain_ShipTo
    }
    df_in_Item_Master {
        string  Item_GBM
        string  Product_Group
        string  Item_Item
    }
    %% --- GI 파이프라인 --------------------------------------------------------
    df_in_Sell_In_Actual_GI {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Partial_Week
        int     SIn_Actual_GI
    }
    df_fn_Sell_In_Actual_GI {
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Item_GBM
        string  Product_Group
        string  Location_Location
        string  Partial_Week
        int     SIn_Actual_GI
    }
    df_output_Sell_In_FCST_GI_AP1 {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Partial_Week
        int     SIn_FCST_AP1_GI
    }
    df_output_Sell_In_FCST_GI_AP2 {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Partial_Week
        int     SIn_FCST_AP2_GI
    }
    df_output_Sell_In_FCST_GI_GC {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Partial_Week
        int     SIn_FCST_GC_GI
    }

    df_in_Item_Master             || -- || df_fn_Sell_In_Actual_GI              : ""
    df_in_Sales_Domain_Dimension  || -- || df_fn_Sell_In_Actual_GI              : ""
    df_in_Sell_In_Actual_GI       || -- || df_fn_Sell_In_Actual_GI              : ""
    df_fn_Sell_In_Actual_GI       || -- || df_output_Sell_In_FCST_GI_AP1        : ""    
    df_fn_Sell_In_Actual_GI       || -- || df_output_Sell_In_FCST_GI_AP2        : ""    
    df_fn_Sell_In_Actual_GI       || -- || df_output_Sell_In_FCST_GI_GC         : ""
    df_in_Forecast_Rule           || .. || df_output_Sell_In_FCST_GI_AP1        : ""    
    df_in_Forecast_Rule           || .. || df_output_Sell_In_FCST_GI_AP2        : ""
    df_in_Forecast_Rule           || .. || df_output_Sell_In_FCST_GI_GC         : ""
```
# 2. S/In Actual BL 관련 ERD
``` mermaid
erDiagram
    %% --- 공통 기준 테이블 ----------------------------------------------------
    df_in_Forecast_Rule {
        string  Version_Name
        string  Product_Group
        string  Sales_Domain_ShipTo
        int     FORECAST_RULE_GC_FCST
        int     FORECAST_RULE_AP2_FCST
        int     FORECAST_RULE_AP1_FCST
    }
    df_in_Sales_Domain_Dimension {
        string  Sales_Domain_Std1
        string  Sales_Domain_Std2
        string  Sales_Domain_Std3
        string  Sales_Domain_Std4
        string  Sales_Domain_Std5
        string  Sales_Domain_Std6
        string  Sales_Domain_ShipTo
    }
    df_in_Item_Master {
        string  Item_GBM
        string  Product_Group
        string  Item_Item
    }
    %% --- BL 파이프라인 --------------------------------------------------------
    df_in_Sell_In_Actual_BL {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Partial_Week
        int     SIn_Actual_BL
    }
    df_fn_Sell_In_Actual_BL {
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Item_GBM
        string  Product_Group
        string  Location_Location
        string  Partial_Week
        int     SIn_Actual_BL
    }
    df_output_Sell_In_FCST_BL_AP1 {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Partial_Week
        int     SIn_FCST_AP1_BL
    }
    df_output_Sell_In_FCST_BL_AP2 {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Partial_Week
        int     SIn_FCST_AP2_BL
    }
    df_output_Sell_In_FCST_BL_GC {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Partial_Week
        int     SIn_FCST_GC_BL
    }
    df_in_Item_Master            || -- || df_fn_Sell_In_Actual_BL               : ""
    df_in_Sales_Domain_Dimension || -- || df_fn_Sell_In_Actual_BL               : ""
    df_in_Sell_In_Actual_BL      || -- || df_fn_Sell_In_Actual_BL               : ""
    df_fn_Sell_In_Actual_BL      || -- || df_output_Sell_In_FCST_BL_AP1         : ""
    df_fn_Sell_In_Actual_BL      || -- || df_output_Sell_In_FCST_BL_AP2         : ""
    df_fn_Sell_In_Actual_BL      || -- || df_output_Sell_In_FCST_BL_GC          : ""
    df_in_Forecast_Rule          || .. || df_output_Sell_In_FCST_BL_AP1         : ""
    df_in_Forecast_Rule          || .. || df_output_Sell_In_FCST_BL_AP2         : ""
    df_in_Forecast_Rule          || .. || df_output_Sell_In_FCST_BL_GC          : ""
```
# 3. Sell‑Out 관련 ERD
``` mermaid
erDiagram
    %% --- 공통 기준 테이블 ----------------------------------------------------
    df_in_Forecast_Rule {
        string  Version_Name
        string  Product_Group
        string  Sales_Domain_ShipTo
        int     FORECAST_RULE_GC_FCST
        int     FORECAST_RULE_AP2_FCST
        int     FORECAST_RULE_AP1_FCST
    }
    df_in_Sales_Domain_Dimension {
        string  Sales_Domain_Std1
        string  Sales_Domain_Std2
        string  Sales_Domain_Std3
        string  Sales_Domain_Std4
        string  Sales_Domain_Std5
        string  Sales_Domain_Std6
        string  Sales_Domain_ShipTo
    }
    df_in_Item_Master {
        string  Item_GBM
        string  Product_Group
        string  Item_Item
    }
    %% --- Sell‑Out 파이프라인 --------------------------------------------------
    df_in_Sell_Out_Actual {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Week                " A주차 적용"
        int     SOut_Actual
    }
    df_fn_Sell_Out_Actual {
        string  Item_Item
        string  Item_GBM
        string  Product_Group
        string  Location_Location
        string  Partial_Week
        int     SOut_Actual
    }
    df_output_Sell_Out_FCST_AP1 {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Partial_Week
        int     SOut_FCST_AP1
    }
    df_output_Sell_Out_FCST_AP2 {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Partial_Week
        int     SOut_FCST_AP2
    }
    df_output_Sell_Out_FCST_GC {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Location_Location
        string  Partial_Week
        int     SOut_FCST_GC
    }
    df_in_Item_Master            || -- || df_fn_Sell_Out_Actual			: ""
    df_in_Sales_Domain_Dimension || -- || df_fn_Sell_Out_Actual  		: ""
    df_in_Sell_Out_Actual     || -- || df_fn_Sell_Out_Actual  		: ""
    df_fn_Sell_Out_Actual     || -- || df_output_Sell_Out_FCST_AP1		: ""
    df_fn_Sell_Out_Actual     || -- || df_output_Sell_Out_FCST_AP2		: ""
    df_fn_Sell_Out_Actual     || -- || df_output_Sell_Out_FCST_GC		: ""
    df_in_Forecast_Rule          || .. || df_output_Sell_Out_FCST_AP1		: ""
    df_in_Forecast_Rule          || .. || df_output_Sell_Out_FCST_AP2		: ""
    df_in_Forecast_Rule          || .. || df_output_Sell_Out_FCST_GC		: ""
```
# 4. Channel Inv 관련 ERD
``` mermaid
erDiagram
    %% --- 공통 기준 테이블 ----------------------------------------------------
    df_in_Forecast_Rule {
        string  Version_Name
        string  Product_Group
        string  Sales_Domain_ShipTo
        int     FORECAST_RULE_GC_FCST
        int     FORECAST_RULE_AP2_FCST
        int     FORECAST_RULE_AP1_FCST
    }
    df_in_Sales_Domain_Dimension {
        string  Sales_Domain_Std1
        string  Sales_Domain_Std2
        string  Sales_Domain_Std3
        string  Sales_Domain_Std4
        string  Sales_Domain_Std5
        string  Sales_Domain_Std6
        string  Sales_Domain_ShipTo
    }
    df_in_Item_Master {
        string  Item_GBM
        string  Product_Group
        string  Item_Item
    }
    %% --- Channel Inv 파이프라인 ----------------------------------------------
    df_in_Channel_Inv {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Week
        int     Channel_Inv
    }
    df_fn_Channel_Inv {
        string  Item_Item
        string  Item_GBM
        string  Product_Group
        string  Week
        int     Channel_Inv
    }
    df_output_Channel_Inv_AP1 {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Week
        int     SOut_FCST_AP1_Channel_Inv
    }
    df_output_Channel_Inv_AP2 {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Week
        int     SOut_FCST_AP2_Channel_Inv
    }
    df_output_Channel_Inv_GC {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Week
        int     SOut_FCST_GC_Channel_Inv
    }
    df_in_Item_Master            || -- || df_fn_Channel_Inv					: ""	
    df_in_Sales_Domain_Dimension || -- || df_fn_Channel_Inv                 : ""
    df_in_Channel_Inv            || -- || df_fn_Channel_Inv                 : ""
    df_fn_Channel_Inv            || -- || df_output_Channel_Inv_AP1         : ""
    df_fn_Channel_Inv            || -- || df_output_Channel_Inv_AP2         : ""
    df_fn_Channel_Inv            || -- || df_output_Channel_Inv_GC          : ""
    df_in_Forecast_Rule          || .. || df_output_Channel_Inv_AP1         : ""
    df_in_Forecast_Rule          || .. || df_output_Channel_Inv_AP2         : ""
    df_in_Forecast_Rule          || .. || df_output_Channel_Inv_GC          : ""
```
# 5. Channel Inv Inc Floor 관련 ERD
``` mermaid
erDiagram
    %% --- 공통 기준 테이블 ----------------------------------------------------
    df_in_Forecast_Rule {
        string  Version_Name
        string  Product_Group
        string  Sales_Domain_ShipTo
        int     FORECAST_RULE_GC_FCST
        int     FORECAST_RULE_AP2_FCST
        int     FORECAST_RULE_AP1_FCST
    }
    df_in_Sales_Domain_Dimension {
        string  Sales_Domain_Std1
        string  Sales_Domain_Std2
        string  Sales_Domain_Std3
        string  Sales_Domain_Std4
        string  Sales_Domain_Std5
        string  Sales_Domain_Std6
        string  Sales_Domain_ShipTo
    }
    df_in_Item_Master {
        string  Item_GBM
        string  Product_Group
        string  Item_Item
    }
    %% --- Channel Inv Inc Floor 파이프라인 -------------------------------------
    df_in_Channel_Inv_Floor {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Week
        int     Channel_Inv_Inc_Floor
    }
    df_fn_Channel_Inv_Floor {
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Item_GBM
        string  Product_Group
        string  Week
        int     Channel_Inv_Inc_Floor
    }
    df_output_Channel_Inv_floor_AP1 {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Week
        int     SOut_FCST_AP1_Channel_Inv_Floor
    }
    df_output_Channel_Inv_floor_AP2 {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Week
        int     SOut_FCST_AP2_Channel_Inv_Floor
    }
    df_output_Channel_Inv_floor_GC {
        string  Version_Name
        string  Sales_Domain_ShipTo
        string  Item_Item
        string  Week
        int     SOut_FCST_GC_Channel_Inv_Floor
    }
    df_in_Item_Master                || -- || df_fn_Channel_Inv_Floor    		: ""
    df_in_Sales_Domain_Dimension     || -- || df_fn_Channel_Inv_Floor           : ""
    df_in_Channel_Inv_Floor      || -- || df_fn_Channel_Inv_Floor           : ""
    df_fn_Channel_Inv_Floor      || -- || df_output_Channel_Inv_floor_AP1       : ""
    df_fn_Channel_Inv_Floor      || -- || df_output_Channel_Inv_floor_AP2       : ""
    df_fn_Channel_Inv_Floor      || -- || df_output_Channel_Inv_floor_GC        : ""
    df_in_Forecast_Rule              || .. || df_output_Channel_Inv_floor_AP1       : ""
    df_in_Forecast_Rule              || .. || df_output_Channel_Inv_floor_AP2       : ""
    df_in_Forecast_Rule              || .. || df_output_Channel_Inv_floor_GC        : ""
```
