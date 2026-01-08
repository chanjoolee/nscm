# ERD
```mermaid
erDiagram
    df_in_Sales_Domain_Dimension {
        string  Sales_Domain_Std1
        string  Sales_Domain_Std2
        string  Sales_Domain_Std3
        string  Sales_Domain_Std4
        string  Sales_Domain_Std5
        string  Sales_Domain_Std6
        string  Sales_Domain_ShipTo
    }
    
    df_in_Sin_GI_AP1 {
        string Version_Name
        string Item_Item
        string Time_Week
        string Sales_Std5
        int    Sin_FCST_GI_AP1
    }    
    
    df_fn_Sin_GI_AP1 {
        string Item_Item
        string Time_Week
        string Sales_ShipTo
        int    Sin_FCST_GI_AP1
    }

    df_fn_Sin_GI_AP1_lv6 {
        string Item_Item
        string Time_Week
        string Sales_ShipTo
        int    Sin_FCST_GI_AP1
    }

    df_in_EstimatedPrice_USD {
        string Version_Name
        string Item_Item
        string Time_Weekd
        string Sales_ShipTo
        float  Estimated_Price_USD
    }

    df_fn_EstimatedPrice_USD {
        string Item_Item
        string Time_Week
        string Sales_ShipTo
        float  Estimated_Price_USD
    }

    df_fn_Merged_Lv6_Est {
        string Item_Item
        string Time_Week
        string Sales_ShipTo
        int    Sin_FCST_GI_AP1
        float  Estimated_Price_USD
    }

    df_fn_Dummy_Lv7 {
        string Item_Item
        string Time_Week
        string Sales_ShipTo
        float  Estimated_Price_USD
        string Sales_Std6
        string Sales_Std7
    }

    df_fn_Converted_Lv7 {
        string Item_Item
        string Time_Week
        string Sales_ShipTo
        float  Estimated_Price_USD
    }

    df_fn_Dummy_Lv5 {
        string Item_Item
        string Time_Week
        string Sales_ShipTo
        float  Estimated_Price_USD
    }

    df_out_step07_final {
        string Version_Name
        string Item_Item
        string Time_Week
        string Sales_ShipTo
        float  Estimated_Price_USD
    }

    df_in_Sin_GI_AP1 ||--|| df_fn_Sin_GI_AP1 : Cleans
    df_fn_Sin_GI_AP1 ||--o{ df_fn_Sin_GI_AP1_lv6 : Splits
    df_in_EstimatedPrice_USD ||--|| df_fn_EstimatedPrice_USD : Cleans
    df_fn_Sin_GI_AP1_lv6 ||--|| df_fn_Merged_Lv6_Est : Joins
    df_fn_EstimatedPrice_USD ||--|| df_fn_Merged_Lv6_Est : Joins
    df_fn_Merged_Lv6_Est ||--|| df_fn_Dummy_Lv7 : Expands
    df_fn_Dummy_Lv7 ||--|| df_fn_Converted_Lv7 : Renames
    df_fn_Converted_Lv7 ||--|| df_fn_Dummy_Lv5 : Aggregates
    df_fn_Dummy_Lv5 ||--|| df_out_step07_final : Outputs
```

# PYDPDummyPrice Sequence Diagrams (Function-by-Function)
## Step 01-1: fn_step01_1_preprocess_sin_ap1
```mermaid
sequenceDiagram
    participant df_in_Sin_GI_AP1
    participant df_fn_Sin_GI_AP1
    df_in_Sin_GI_AP1->>fn_step01_1_preprocess_sin_ap1: Read raw AP1 forecast
    fn_step01_1_preprocess_sin_ap1->>df_fn_Sin_GI_AP1: Drop Version & rename ShipTo
```
## Step 01-2: fn_step01_2_split_lv_by_item.
```mermaid
sequenceDiagram
    participant df_fn_Sin_GI_AP1
    participant df_fn_Sin_GI_AP1_lv3
    participant df_fn_Sin_GI_AP1_lv4
    participant df_fn_Sin_GI_AP1_lv5
    participant df_fn_Sin_GI_AP1_lv6
    df_fn_Sin_GI_AP1->>fn_step01_2_split_lv_by_item: Split by Item LV
    fn_step01_2_split_lv_by_item->>df_fn_Sin_GI_AP1_lv3: Filter LV=3
    fn_step01_2_split_lv_by_item->>df_fn_Sin_GI_AP1_lv4: Filter LV=4
    fn_step01_2_split_lv_by_item->>df_fn_Sin_GI_AP1_lv5: Filter LV=5
    fn_step01_2_split_lv_by_item->>df_fn_Sin_GI_AP1_lv6: Filter LV=6
```
## Step 02-1: fn_step02_preprocess_estimated_price.  Drop Version
```mermaid
sequenceDiagram
    participant df_in_EstimatedPrice_USD
    participant df_fn_EstimatedPrice_USD
    df_in_EstimatedPrice_USD->>fn_step02_preprocess_estimated_price: Read raw price
    fn_step02_preprocess_estimated_price->>df_fn_EstimatedPrice_USD: Drop Version
```
## Step 03: fn_step03_merge_lv6_est
```mermaid
sequenceDiagram
    participant df_fn_Sin_GI_AP1_lv6
    participant df_fn_EstimatedPrice_USD
    participant df_fn_Merged_Lv6_Est
    df_fn_Sin_GI_AP1_lv6->>fn_step03_merge_lv6_est: Provide 6Lv forecast
    df_fn_EstimatedPrice_USD->>fn_step03_merge_lv6_est: Provide estimated price
    fn_step03_merge_lv6_est->>df_fn_Merged_Lv6_Est: Merge + fillna(1)
```
## Step 04: fn_step04_generate_dummy_lv7
```mermaid
sequenceDiagram
    participant df_fn_Merged_Lv6_Est
    participant df_in_Sales_Domain_Dimension
    participant df_fn_Dummy_Lv7
    df_fn_Merged_Lv6_Est->>fn_step04_generate_dummy_lv7: Start with merged forecast
    df_in_Sales_Domain_Dimension->>fn_step04_generate_dummy_lv7: Lookup for 7Lv mapping
    fn_step04_generate_dummy_lv7->>df_fn_Dummy_Lv7: Filter + Join result
```
## Step 05: fn_step05_convert_lv7_to_shipto
```mermaid
sequenceDiagram
    participant df_fn_Dummy_Lv7
    participant df_fn_Converted_Lv7
    df_fn_Dummy_Lv7->>fn_step05_convert_lv7_to_shipto: Replace ShipTo with Std6
    fn_step05_convert_lv7_to_shipto->>df_fn_Converted_Lv7: Drop legacy Lv info
```
## Step 06: fn_step06_generate_dummy_lv5
```mermaid
sequenceDiagram
    participant df_fn_Converted_Lv7
    participant df_in_Sales_Domain_Dimension
    participant df_fn_Dummy_Lv5
    df_fn_Converted_Lv7->>fn_step06_generate_dummy_lv5: Use ShipTo to lookup Std4
    df_in_Sales_Domain_Dimension->>fn_step06_generate_dummy_lv5: Provide mapping info
    fn_step06_generate_dummy_lv5->>df_fn_Dummy_Lv5: Replace ShipTo with 5Lv
```
## Step 07: fn_step07_concat_and_finalize
```mermaid
sequenceDiagram
    participant df_fn_Merged_Lv6_Est
    participant df_fn_Converted_Lv7
    participant df_fn_Dummy_Lv5
    participant df_out_step07_final
    df_fn_Merged_Lv6_Est->>fn_step07_concat_and_finalize: Final merge input
    df_fn_Converted_Lv7->>fn_step07_concat_and_finalize: Final merge input
    df_fn_Dummy_Lv5->>fn_step07_concat_and_finalize: Final merge input
    fn_step07_concat_and_finalize->>df_out_step07_final: Add Version + Clean output
```

