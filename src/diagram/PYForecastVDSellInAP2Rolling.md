# ERD
``` mermaid
erDiagram
    df_in_SIn {
        string Version_Name
        string Sales_Domain_Ship_To
        string Location
        string Item_Product_Group
        string Item_Item
        string Time_Planning_Month
        string Time_Week
        int S_In_FCST_GI_AP2
    }
    
    df_in_Forecast_Rule {
        string Version_Namet
        string Item_Product_Group
        string Sales_Domain_Ship_To
        int FORECAST_RULE_AP2_FCST
    }

    df_in_Sales_Domain_Dimension {
        string Sales_Domain_LV2
        string Sales_Domain_LV3
        string Sales_Domain_LV4
        string Sales_Domain_LV5
        string Sales_Domain_LV6
        string Sales_Domain_LV7
        string Sales_Domain_Ship_To
    }

    df_step01_remove_duplicate_forecast_rule {
        string Version_Name
        string Item_Product_Group
        string Sales_Domain_Ship_To
        int FORECAST_RULE_AP2_FCST
    }
    
    df_step02_expand_sin_to_lv2_lv7 {
        string Version_Name
        string Sales_Domain_Ship_To
        string Location
        string Item_Product_Group
        string Item_Item
        string Time_Planning_Month
        string Time_Week
        int S_In_FCST_GI_AP2
        string Sales_Domain_LV2
        string Sales_Domain_LV3
        string Sales_Domain_LV4
        string Sales_Domain_LV5
        string Sales_Domain_LV6
        string Sales_Domain_LV7
    }

    df_out_step03_1_sin_lv2 {
        string Version_Name
        string GBRULE
        string Sales_Domain_Ship_To
        string Location
        string Item_Product_Group
        string Item_Item
        string Time_Planning_Month
        string Time_Week
        int S_In_FCST_GI_AP2
    }

    df_out_step03_2_sin_lv3 {
        string Version_Name
        string GBRULE
        string Sales_Domain_Ship_To
        string Location
        string Item_Product_Group
        string Item_Item
        string Time_Planning_Month
        string Time_Week
        int S_In_FCST_GI_AP2
    }

    df_out_step03_3_sin_lv2_lv3 {
        string Version_Name
        string GBRULE
        string Location
        string Item_Item
        string Time_Week
        int S_In_FCST_GI_AP2
    }

    df_out_step03_4_sin_lv2_lv3 {
        string Version_Name
        string GBRULE
        string Location
        string Item_Item
        string Time_Week
        int S_In_FCST_GI_AP2
    }
    
    df_out_step4 {
        string Version_Name
        string GBRULE
        string Item_Item
        string Location
        string Time_Week
        int S_In_FCST_GI_AP2_Rolling_ADJ
    }

    df_in_SIn ||--|| df_step02_expand_sin_to_lv2_lv7 : contains
    df_in_Forecast_Rule ||--|| df_step01_remove_duplicate_forecast_rule : processed_into
    df_in_Sales_Domain_Dimension ||--|| df_step02_expand_sin_to_lv2_lv7 : enriches
    df_step01_remove_duplicate_forecast_rule ||--|| df_step02_expand_sin_to_lv2_lv7 : joins
    df_step02_expand_sin_to_lv2_lv7 ||--|| df_out_step03_1_sin_lv2 : processes_into
    df_step02_expand_sin_to_lv2_lv7 ||--|| df_out_step03_2_sin_lv3 : processes_into
    df_out_step03_1_sin_lv2 ||--|| df_out_step03_3_sin_lv2_lv3 : concatenates
    df_out_step03_2_sin_lv3 ||--|| df_out_step03_3_sin_lv2_lv3 : concatenates
    df_out_step03_3_sin_lv2_lv3 ||--|| df_out_step03_4_sin_lv2_lv3 : processes_into
    df_out_step03_4_sin_lv2_lv3 ||--|| df_out_step4 : computes_adjustments

```

# Sequence : All
``` mermaid
sequenceDiagram
    participant P as Processor
    participant D as df_in_Sales_Domain_Dimension
    participant F as df_in_Forecast_Rule
    participant SDR as df_in_SIn_SD (Enriched SIN Data)
    participant LV2 as df_in_SIn_SD_Lv2 (LV2 Join)
    participant LV3 as df_in_SIn_SD_Lv3 (LV3 Join)
    participant ALL as df_in_SIn_SD_All (Final Output)

    %% Step 1: Update Forecast Rule SalesDomainShipTo if condition is met.
    Note over P,F: Before enrichment, check Forecast Rule
    P->>F: and SalesDomainShipTo starts with "3"
    P->>D: Lookup parent Level2 value in df_in_Sales_Domain_Dimension based on F.SalesDomainShipTo
    D-->>P: Return parent Level2 value
    P->>F: Update F.SalesDomainShipTo with Level2 value
    
    %% Step 2: Enriched SIN data to df_in_SIn_SD.
    P->>SDR: Obtain enriched df_in_SIn_SD (includes Sales Domain.[Sales Domain LV2] & Sales Domain.[Sales Domain LV3])

    %% Step 3-1: Perform inner join for LV2 branch.
    Note over P,LV2: For LV2 join, the join condition is:<br>df_in_Forecast_Rule.Item.[Product Group] = df_in_SIn_SD.Item.[Product Group]<br>AND df_in_Forecast_Rule.SalesDomainShipTo = df_in_SIn_SD.Sales Domain.[Sales Domain LV2]
    P->>LV2: Inner join df_in_SIn_SD and df_in_Forecast_Rule for LV2 match
    LV2-->>P: Return df_in_SIn_SD_Lv2 with GBRULE set using Sales Domain.[Sales Domain LV2] value

    %% Step 3-1: Perform inner join for LV3 branch.
    Note over P,LV3: For LV3 join, the join condition is:<br>df_in_Forecast_Rule.Item.[Product Group] = df_in_SIn_SD.Item.[Product Group]<br>AND df_in_Forecast_Rule.SalesDomainShipTo = df_in_SIn_SD.Sales Domain.[Sales Domain LV3]
    P->>LV3: Inner join df_in_SIn_SD and df_in_Forecast_Rule for LV3 match
    LV3-->>P: Return df_in_SIn_SD_Lv3 with GBRULE set using Sales Domain.[Sales Domain LV3] value

    %% Step 3-1: Concatenate results from both branch joins.
    P->>ALL: Concatenate df_in_SIn_SD_Lv2 and df_in_SIn_SD_Lv3
    ALL-->>P: Return final merged df_in_SIn_SD_All (with GBRULE determined)
```

## Sequence : Step 1 (Removing Duplicate Forecast Rule)
``` mermaid
sequenceDiagram
    participant User
    participant System
    participant df_in_Forecast_Rule
    participant df_in_Sales_Domain_Dimension
    participant df_step01_remove_duplicate_forecast_rule

    User ->> System: Start Step 1 (Remove Duplicate Forecast Rule)
    System ->> df_in_Forecast_Rule: Load data

    System ->> System: Identify rows where FORECAST_RULE AP2 FCST == 2
    System ->> System: Check if Sales Domain.[Ship To] starts with '3'
    
    alt If condition met
        System ->> df_in_Sales_Domain_Dimension: Find parent level for Sales Domain.[Ship To]
        df_in_Sales_Domain_Dimension ->> System: Return parent level
        System ->> df_in_Forecast_Rule: Update 'Sales Domain.[Ship To]' with parent level
    end
    
    System ->> df_step01_remove_duplicate_forecast_rule: Remove duplicate rows
    System ->> User: Return df_step01_remove_duplicate_forecast_rule

```


## Sequence : Step 2 (Expanding SIN Data with Sales Domain)
``` mermaid
sequenceDiagram
    participant User
    participant System
    participant df_in_Sales_Domain_Dimension
    participant df_in_SIn
    participant df_step02_expand_sin_to_lv2_lv7

    User ->> System: Start Step 2 (Expand SIN Data)
    
    System ->> df_in_Sales_Domain_Dimension: Load Sales Domain data
    System ->> df_in_SIn: Load SIN data

    System ->> df_step02_expand_sin_to_lv2_lv7: Perform INNER JOIN
    df_step02_expand_sin_to_lv2_lv7 ->> System: Joined dataframe based on Sales Domain.[Ship To]

    System ->> df_step02_expand_sin_to_lv2_lv7: Add LV2 to LV7 Sales Domain columns
    System ->> User: Return df_step02_expand_sin_to_lv2_lv7


```

## Sequence : Step 3-1 (LV2 Forecast Rule Processing)
``` mermaid
sequenceDiagram
    participant User
    participant System
    participant df_step02_expand_sin_to_lv2_lv7
    participant df_step01_remove_duplicate_forecast_rule
    participant filter_df_step01_forecast_rule_lv2
    participant return_df

    User ->> System: Start Step 3-1 (LV2 Processing)
    System ->> df_step02_expand_sin_to_lv2_lv7: Load data
    System ->> df_step01_remove_duplicate_forecast_rule: Load data

    System ->> filter_df_step01_forecast_rule_lv2: Filter rows
    filter_df_step01_forecast_rule_lv2 ->> System: Return filtered forecast rules (LV2)

    System ->> return_df: Perform INNER JOIN
    return_df ->> System: Joined dataframe based on 'Sales Domain.[Sales Domain LV2]'

    System ->> return_df: Add GBRULE column based on forecast rule
    System ->> return_df: Drop unnecessary columns
    System ->> User: Return df_out_step03_1_sin_lv2

```

## Sequence : Step 3-2 (LV3 Forecast Rule Processing)
``` mermaid
sequenceDiagram
    participant User
    participant System
    participant df_step02_expand_sin_to_lv2_lv7
    participant df_step01_remove_duplicate_forecast_rule
    participant filter_df_step01_forecast_rule_lv3
    participant return_df

    User ->> System: Start Step 3-2 (LV3 Processing)
    System ->> df_step02_expand_sin_to_lv2_lv7: Load data
    System ->> df_step01_remove_duplicate_forecast_rule: Load data

    System ->> df_step01_remove_duplicate_forecast_rule: Drop 'Version.[Version Name]'
    System ->> filter_df_step01_forecast_rule_lv3: Filter rows
    filter_df_step01_forecast_rule_lv3 ->> System: Return filtered forecast rules (LV3)

    System ->> return_df: Perform INNER JOIN
    return_df ->> System: Joined dataframe based on 'Sales Domain.[Sales Domain LV3]'

    System ->> return_df: Add GBRULE column based on forecast rule
    System ->> return_df: Drop unnecessary columns
    System ->> User: Return df_out_step03_2_sin_lv3

```

## Sequence : Step 3-3 (Concatenation of LV2 and LV3 Data)
``` mermaid
sequenceDiagram
    participant User
    participant System
    participant df_out_step03_1_sin_lv2
    participant df_out_step03_2_sin_lv3
    participant df_out_step03_3_sin_lv2_lv3

    User ->> System: Start Step 3-3 (Concatenation)
    System ->> df_out_step03_1_sin_lv2: Load data
    System ->> df_out_step03_2_sin_lv3: Load data

    System ->> df_out_step03_3_sin_lv2_lv3: Concatenate df_out_step03_1_sin_lv2 & df_out_step03_2_sin_lv3
    System ->> User: Return df_out_step03_3_sin_lv2_lv3


```

## Sequence : Step 4 (Rolling Adjustment)
``` mermaid
sequenceDiagram
    participant User
    participant System
    participant df_out_step03_4_sin_lv2_lv3
    participant grouped_data
    participant df_current
    participant df_previous
    participant return_array
    participant df_out_step4

    User ->> System: Start Step 4 (Rolling Adjustment)
    System ->> df_out_step03_4_sin_lv2_lv3: Load data

    System ->> grouped_data: Group by GBRULE, Location, Item
    System ->> return_array: Initialize empty array

    loop For each group in grouped_data
        System ->> df_current: Extract rows with 'CurrentWorkingView'
        System ->> df_previous: Extract rows with historical version
        
        System ->> df_current: Find current week using CurrentPartialWeek_normalized
        System ->> df_previous: Find last week using common.gfn_add_week(CurrentPartialWeek_normalized, -1)
        
        System ->> df_previous: Get last week's FCST values
        System ->> df_current: Get last week's FCST values

        alt If planning month is same and FCST values differ
            System ->> return_array: Compute adjustment and append to return array
        end
    end

    System ->> df_out_step4: Convert return_array to DataFrame
    System ->> User: Return df_out_step4

```