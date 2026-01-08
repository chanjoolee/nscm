# ERD
``` mermaid
erDiagram

    df_in_Sales_Domain_Dimension {
        string  Sales_Domain_Lv2
        string  Sales_Domain_Lv3
        string  Sales_Domain_Lv4
        string  Sales_Domain_Lv5
        string  Sales_Domain_Lv6
        string  Sales_Domain_Lv7
        string  Sales_Domain_ShipTo
    }

    df_in_Sales_Product_ASN {
        string  Version_Name
        string  Sales_Domain_ShipTo     "Lv 6,7"
        string  Item_Item
        string  Location_Location
        string  Sales_Product_ASN       "Y or N"
    }

    df_fn_Sales_Product_ASN {
        string  Sales_Domain_ShipTo    "Lv 6,7"
        string  Item_Item
        string  Location_Location
    }

    df_fn_Sales_Product_ASN_Dummy {
        string  Sales_Domain_ShipTo     "Lv 6,7"
        string  Item_Item
        string  Location_Location       "-"
    }

    df_fn_Sales_Product_ASN_lv6 {
        string  Sales_Domain_ShipTo    "Lv 6"
        string  Item_Item
        string  Location_Location
    }


    df_fn_Sales_Product_ASN_lv7 {
        string  Sales_Domain_ShipTo    "Lv 7"
        string  Item_Item
        string  Location_Location
    }

    
    df_fn_Sales_Product_ASN_lv7_to_lv6 {
        string  Sales_Domain_ShipTo    "Lv 6"
        string  Item_Item
        string  Location_Location
    }

    df_fn_Sales_Product_ASN_Concat {
        string  Sales_Domain_ShipTo    "Lv 6"
        string  Item_Item
        string  Location_Location
    }

    df_fn_Sales_Product_ASN_Concat_dim {
        string  Sales_Domain_ShipTo    "Lv 6"
        string  Item_Item
        string  Location_Location
        string  Sales_Domain_Lv2
        string  Sales_Domain_Lv3
        string  Sales_Domain_Lv4
        string  Sales_Domain_Lv5
        string  Sales_Domain_Lv6
        string  Sales_Domain_Lv7
    }

    df_fn_Sales_Product_ASN_Lv2_to_Lv5 {
        string  Sales_Domain_ShipTo    "Lv 2 ~ 5"
        string  Item_Item
        string  Location_Location
    }

    df_fn_Sales_Product_ASN_Lv2_to_Lv6 {
        string  Sales_Domain_ShipTo    "Lv 2 ~ 6"
        string  Item_Item
        string  Location_Location
    }

    df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy {
        string  Sales_Domain_ShipTo    "Lv 2 ~ 6"
        string  Item_Item
        string  Location_Location      "-"   
    }

    df_fn_Sales_Product_ASN_Final {
        string  Sales_Domain_ShipTo    "Lv 2 ~ 6"
        string  Item_Item
        string  Location_Location      "-"   
    }

    Output_Sales_Product_ASN {
        string  Version_Name
        string  Sales_Domain_ShipTo    ""
        string  Item_Item
        string  Location_Location
        string  Salse_Product_ASN        "Y"
    }
    
    %% Step01 ===================================
    df_in_Sales_Product_ASN         || --  |{ df_fn_Sales_Product_ASN : "Step 1-1"
    df_in_Sales_Product_ASN         || --  |{ df_fn_Sales_Product_ASN_Dummy : "Step 1-2"

    %% Step 1-3
    df_fn_Sales_Product_ASN         || --  |{ df_fn_Sales_Product_ASN_lv6 : "Step 1-3"
    df_fn_Sales_Product_ASN         || --  |{ df_fn_Sales_Product_ASN_lv7 : "Step 1-3"

    %% Step 2 : 7Lv Sales Product ASN 을 Sales Lv6 으로 Aggr
    df_fn_Sales_Product_ASN_lv7         || --  |{ df_fn_Sales_Product_ASN_lv7_to_lv6 : "Step 2: 7Lv -> 6Lv"
    df_in_Sales_Domain_Dimension        || --  |{ df_fn_Sales_Product_ASN_lv7_to_lv6 : "Step 2"

    %% Step 3  ===================================
    %% Step 3-1
    df_fn_Sales_Product_ASN_lv6         || --  |{ df_fn_Sales_Product_ASN_Concat : "Step 3-1"
    df_fn_Sales_Product_ASN_lv7_to_lv6         || --  |{ df_fn_Sales_Product_ASN_Concat : "Step 3-1"

    %% Step 3-2
    df_in_Sales_Domain_Dimension    || --  |{ df_fn_Sales_Product_ASN_Concat_dim : "Step 3-2"
    df_fn_Sales_Product_ASN_Concat  || --  |{ df_fn_Sales_Product_ASN_Concat_dim : "Step 3-2"

    %% Step 3-3
    df_fn_Sales_Product_ASN_Concat_dim || --  |{ df_fn_Sales_Product_ASN_Lv2_to_Lv5 : "Step 3-3"

    %% Step 4  ===================================
    df_fn_Sales_Product_ASN_Concat      || --  |{ df_fn_Sales_Product_ASN_Lv2_to_Lv6 : "Step 4: concat"
    df_fn_Sales_Product_ASN_Lv2_to_Lv5  || --  |{ df_fn_Sales_Product_ASN_Lv2_to_Lv6 : "Step 4: concat"

    %% Step 5  ===================================
    df_fn_Sales_Product_ASN_Lv2_to_Lv6  || --  |{ df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy : "Step 5: Location -> -"

    %% step06
    df_fn_Sales_Product_ASN_Lv2_to_Lv6  || --  |{ df_fn_Sales_Product_ASN_Final : "Step 6"
    df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy       || --  |{ df_fn_Sales_Product_ASN_Final : "Step 6"
    df_fn_Sales_Product_ASN_Dummy       || --  |{ df_fn_Sales_Product_ASN_Final : "Step 6"
    

    df_fn_Sales_Product_ASN_Final  || --  |{ Output_Sales_Product_ASN : ""
```

# Sequence

## Step 01 : ASN 전처리
``` mermaid
sequenceDiagram
    autonumber
    participant STEP01
    participant df_in_Sales_Product_ASN
    participant df_in_Sales_Domain_Dimension
    participant df_fn_Sales_Product_ASN
    participant df_fn_Sales_Product_ASN_Dummy
    participant df_fn_Sales_Product_ASN_lv6
    participant df_fn_Sales_Product_ASN_lv7   

    STEP01->>df_in_Sales_Product_ASN: read CSV    
    STEP01->>df_fn_Sales_Product_ASN: Base 전처리
    STEP01->>df_fn_Sales_Product_ASN_Dummy: Dummy “-” 생성
    STEP01->>df_in_Sales_Domain_Dimension: join for level 판정
    STEP01->>df_fn_Sales_Product_ASN_lv6: Ship==LV6 → Lv-6 출력
    STEP01->>df_fn_Sales_Product_ASN_lv7: Ship==LV7 AND LV6≠LV7 → Lv-7 출력
```

## Step 02 : LV7 → LV6 집계
``` mermaid
sequenceDiagram
    autonumber
    participant STEP02
    participant df_fn_Sales_Product_ASN_lv7
    participant df_in_Sales_Domain_Dimension
    participant df_fn_Sales_Product_ASN_lv7_to_lv6

    STEP02->>df_fn_Sales_Product_ASN_lv7: read
    STEP02->>df_in_Sales_Domain_Dimension: join (LV6 lookup)
    STEP02->>df_fn_Sales_Product_ASN_lv7_to_lv6: Ship-To 치환(LV6)
```

## Step 03 : Hierarchy Build (LV6 Concat → Join → LV2-LV5)
``` mermaid
sequenceDiagram
    autonumber
    participant STEP03
    participant df_fn_Sales_Product_ASN_lv6
    participant df_fn_Sales_Product_ASN_lv7_to_lv6
    participant df_in_Sales_Domain_Dimension
    participant df_fn_Sales_Product_ASN_Concat
    participant df_fn_Sales_Product_ASN_Concat_dim
    participant df_fn_Sales_Product_ASN_Lv2_to_Lv5

    STEP03->>df_fn_Sales_Product_ASN_lv6: read
    STEP03->>df_fn_Sales_Product_ASN_lv7_to_lv6: read
    STEP03->>df_fn_Sales_Product_ASN_Concat: Concat (LV6 + LV7→6)
    STEP03->>df_in_Sales_Domain_Dimension: join LV2~LV7
    STEP03->>df_fn_Sales_Product_ASN_Concat_dim: save join 결과
    STEP03->>df_fn_Sales_Product_ASN_Lv2_to_Lv5: group LV2~LV5 프레임
```

## Step 04 : LV2-LV6 통합
``` mermaid
sequenceDiagram
    autonumber
    participant STEP04
    participant df_fn_Sales_Product_ASN_Concat
    participant df_fn_Sales_Product_ASN_Lv2_to_Lv5
    participant df_fn_Sales_Product_ASN_Lv2_to_Lv6

    STEP04->>df_fn_Sales_Product_ASN_Concat: read
    STEP04->>df_fn_Sales_Product_ASN_Lv2_to_Lv5: read
    STEP04->>df_fn_Sales_Product_ASN_Lv2_to_Lv6: Concat & dedup
```

## Step 05 : Sell-Out Dummy(“-”) 생성
``` mermaid
sequenceDiagram
    autonumber
    participant STEP05
    participant df_fn_Sales_Product_ASN_Lv2_to_Lv6
    participant df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy

    STEP05->>df_fn_Sales_Product_ASN_Lv2_to_Lv6: read
    STEP05->>df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy: Location → “-” · reset_index
```

## Step 06 : 최종 Sales Product ASN 집계
``` mermaid
sequenceDiagram
    autonumber
    participant STEP06
    participant df_fn_Sales_Product_ASN_Lv2_to_Lv6
    participant df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy
    participant df_fn_Sales_Product_ASN_Dummy
    participant df_fn_Sales_Product_ASN_Final
    participant Output_Sales_Product_ASN

    STEP06->>df_fn_Sales_Product_ASN_Lv2_to_Lv6: read
    STEP06->>df_fn_Sales_Product_ASN_Lv2_to_Lv6_Dummy: read
    STEP06->>df_fn_Sales_Product_ASN_Dummy: read
    STEP06->>df_fn_Sales_Product_ASN_Final: concat + flag='Y'
    STEP06-->>Output_Sales_Product_ASN: 최종 Export
```