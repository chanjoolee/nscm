# ERD
``` mermaid
erDiagram
    %% =========================================================================
    %% PYForecastCreateSellInAndSellOutAssortment  (UPDATED ERD)
    %% - 반영사항
    %%   1) Step01: Ship-to Dimension LUT(df_fn_shipto_dim) 명시
    %%   2) Step02: ASN 전처리 + PG 조인 + SOUT_ASS_FLAG 처리( SimulMaster CON + eStore 제외 )
    %%   3) Step03: Forecast Rule 적용(S/In Assortment) (GC/AP2/AP1/Local)
    %%   4) Step04: SOUT_ASS_FLAG==True 만 S/Out Assortment로 변환(Location='-')
    %%   5) Step05: eStore 대상 S/Out Assortment 추가 + Step04 결과와 concat (dict 재사용)
    %%   6) Step06: (25.12.05) No-eStore Past12 Actual 기반 S/Out Assortment 추가
    %%              - Input7 로부터 ASN-like → (PG merge) → Step03/04 재사용 → 기존 df_sout_dict와 concat
    %%   7) Step07: 최종 Output 포맷(Version 추가 + 컬럼 삭제 규칙)
    %% - 관계선에 Step명 표기
    %% =========================================================================

    %% =========================
    %% INPUTS
    %% =========================
    df_in_Forecast_Rule {
        category  Version_Name
        category  Product_Group               "PK"
        category  Sales_Domain_ShipTo         "PK: rule 기준 ShipTo(Std1~Std3 레벨 사용)"
        int32     FORECAST_RULE_GC_FCST
        int32     FORECAST_RULE_AP2_FCST
        int32     FORECAST_RULE_AP1_FCST
        int32     FORECAST_RULE_AP0_FCST
        category  FORECAST_RULE_ISVALID
    }

    df_in_Sales_Domain_Dimension {
        category  Sales_Domain_ShipTo         "PK"
        category  Sales_Domain_Std1
        category  Sales_Domain_Std2
        category  Sales_Domain_Std3
        category  Sales_Domain_Std4
        category  Sales_Domain_Std5
        category  Sales_Domain_Std6
        int32     LV_CODE
    }

    df_in_Sales_Product_ASN {
        category  Version_Name
        category  Sales_Domain_ShipTo         "PK: Lv6(std5),Lv7(std6)"
        category  Item_Item                   "PK"
        category  Location_Location           "PK"
        category  Sales_Product_ASN           "Y/N"
    }

    df_in_Item_Master {
        category  Item_Item                   "PK"
        category  Product_Group
        category  Item_GBM
    }

    df_in_Sales_Domain_Estore {
        category  Sales_Domain_ShipTo         "eStore ShipTo list"
    }

    df_in_Sell_Out_Simul_Master {
        category  Sales_Domain_ShipTo         "PK: Product Group 기준 Std5(Lv6) 코드"
        category  Product_Group               "PK"
        category  SOut_Master_Status          "CON"
    }

    %% (25.12.04 추가)
    df_in_Sell_Out_Actual_NoeStore_Past12 {
        category  Sales_Domain_ShipTo
        category  Item_Item
        category  Location_Location           "(-)"
        float64   SOut_Actual
    }

    %% =========================
    %% STEP DERIVED / FACT TABLES
    %% =========================

    %% Step01 결과: Ship-To LUT
    df_fn_shipto_dim {
        category  Sales_Domain_ShipTo         "PK"
        category  Sales_Domain_Std1
        category  Sales_Domain_Std2
        category  Sales_Domain_Std3
        category  Sales_Domain_Std4
        category  Sales_Domain_Std5
        category  Sales_Domain_Std6
        int32     LV_CODE
    }

    %% Step02 결과: ASN 전처리 테이블
    df_fn_Sales_Product_ASN {
        category  Sales_Domain_ShipTo         "PK: Lv6(std5),Lv7(std6)"
        category  Item_Item                   "PK"
        category  Location_Location           "PK"
        category  Sales_Product_ASN           "Y/N/NA"
        category  Product_Group               "rule match key"
        boolean   SOut_Assortment_Flag        "SOUT_ASS_FLAG (True/False)"
    }

    %% Step06용 ASN-like (Input7 기반)
    df_fn_ASN_like_from_Actual {
        category  Sales_Domain_ShipTo
        category  Item_Item
        category  Location_Location
        category  Sales_Product_ASN           "NA"
        category  Product_Group               "from Item_Master"
        boolean   SOut_Assortment_Flag        "True (forced)"
    }

    %% =========================
    %% OUTPUTS (Step07 최종 포맷)
    %% =========================
    Output_SIn_Assortment_GC {
        category  Version_Name
        category  Sales_Domain_ShipTo
        category  Item_Item
        category  Location_Location
        int32     SIn_FCST_GC_ASS             "1"
    }

    Output_SIn_Assortment_AP2 {
        category  Version_Name
        category  Sales_Domain_ShipTo
        category  Item_Item
        category  Location_Location
        int32     SIn_FCST_AP2_ASS            "1"
    }

    Output_SIn_Assortment_AP1 {
        category  Version_Name
        category  Sales_Domain_ShipTo
        category  Item_Item
        category  Location_Location
        int32     SIn_FCST_AP1_ASS            "1"
    }

    Output_SIn_Assortment_Local {
        category  Version_Name
        category  Sales_Domain_ShipTo
        category  Item_Item
        category  Location_Location
        int32     SIn_FCST_ASS_LOCAL          "1"
    }

    Output_SOut_Assortment_GC {
        category  Version_Name
        category  Sales_Domain_ShipTo
        category  Item_Item
        category  Location_Location           "(-)"
        int32     SOut_FCST_GC_ASS            "1"
    }

    Output_SOut_Assortment_AP2 {
        category  Version_Name
        category  Sales_Domain_ShipTo
        category  Item_Item
        category  Location_Location           "(-)"
        int32     SOut_FCST_AP2_ASS           "1"
    }

    Output_SOut_Assortment_AP1 {
        category  Version_Name
        category  Sales_Domain_ShipTo
        category  Item_Item
        category  Location_Location           "(-)"
        int32     SOut_FCST_AP1_ASS           "1"
    }

    Output_SOut_Assortment_Local {
        category  Version_Name
        category  Sales_Domain_ShipTo
        category  Item_Item
        category  Location_Location           "(-)"
        int32     SOut_FCST_ASS_LOCAL         "1"
    }

    %% =========================
    %% RELATIONSHIPS (with Step names)
    %% =========================

    %% Step01: Dimension LUT build
    df_in_Sales_Domain_Dimension ||--|| df_fn_shipto_dim : "Step01: Build Ship-To LUT"

    %% Step02: ASN preprocess
    df_in_Sales_Product_ASN      ||--|| df_fn_Sales_Product_ASN : "Step02: Base ASN rows"
    df_in_Item_Master            ||--|| df_fn_Sales_Product_ASN : "Step02: Join Product Group"
    df_fn_shipto_dim             ||..|| df_fn_Sales_Product_ASN : "Step02: Std5 lookup for SimulMaster key"
    df_in_Sell_Out_Simul_Master  ||..|| df_fn_Sales_Product_ASN : "Step02: Set SOUT_ASS_FLAG by CON"
    df_in_Sales_Domain_Estore    ||..|| df_fn_Sales_Product_ASN : "Step02: eStore => SOUT_ASS_FLAG False"

    %% Step03: Forecast Rule 적용 (Sell-In Assortment)
    df_in_Forecast_Rule          ||..|| df_fn_shipto_dim         : "Step03: Need Std1~Std6 + LV_CODE"
    df_fn_Sales_Product_ASN      ||..|| df_in_Forecast_Rule       : "Step03: Match (Product_Group, Ancestor ShipTo)"
    df_in_Forecast_Rule          ||--|| Output_SIn_Assortment_GC  : "Step03: Build S/In GC"
    df_in_Forecast_Rule          ||--|| Output_SIn_Assortment_AP2 : "Step03: Build S/In AP2"
    df_in_Forecast_Rule          ||--|| Output_SIn_Assortment_AP1 : "Step03: Build S/In AP1"
    df_in_Forecast_Rule          ||--|| Output_SIn_Assortment_Local : "Step03: Build S/In Local(AP0)"
    df_fn_Sales_Product_ASN      ||..|| Output_SIn_Assortment_GC  : "Step03: Source ASN flags"
    df_fn_Sales_Product_ASN      ||..|| Output_SIn_Assortment_AP2 : "Step03: Source ASN flags"
    df_fn_Sales_Product_ASN      ||..|| Output_SIn_Assortment_AP1 : "Step03: Source ASN flags"
    df_fn_Sales_Product_ASN      ||..|| Output_SIn_Assortment_Local : "Step03: Source ASN flags"

    %% Step04: Sell-Out Assortment 생성 (SOUT_ASS_FLAG==True, Location='-')
    Output_SIn_Assortment_GC     ||--|| Output_SOut_Assortment_GC : "Step04: To S/Out (Loc='-')"
    Output_SIn_Assortment_AP2    ||--|| Output_SOut_Assortment_AP2: "Step04: To S/Out (Loc='-')"
    Output_SIn_Assortment_AP1    ||--|| Output_SOut_Assortment_AP1: "Step04: To S/Out (Loc='-')"
    Output_SIn_Assortment_Local  ||--|| Output_SOut_Assortment_Local: "Step04: To S/Out (Loc='-')"

    %% Step05: eStore Sell-Out Assortment 추가 + concat with Step04 dict
    df_in_Sales_Domain_Estore    ||..|| Output_SOut_Assortment_GC : "Step05: eStore filter + add S/Out GC + concat"
    df_in_Sales_Domain_Estore    ||..|| Output_SOut_Assortment_AP2: "Step05: eStore filter + add S/Out AP2 + concat"
    df_in_Sales_Domain_Estore    ||..|| Output_SOut_Assortment_AP1: "Step05: eStore filter + add S/Out AP1 + concat"
    df_in_Sales_Domain_Estore    ||..|| Output_SOut_Assortment_Local: "Step05: eStore filter + add S/Out Local + concat"

    %% Step06: No-eStore Actual 기반 S/Out Assortment 추가 (Step03/04 재사용)
    df_in_Sell_Out_Actual_NoeStore_Past12 ||--|| df_fn_ASN_like_from_Actual : "Step06-1: Build ASN-like (add ASN=NA)"
    df_in_Item_Master                    ||--|| df_fn_ASN_like_from_Actual : "Step06-1: Join Product Group"
    df_fn_ASN_like_from_Actual           ||..|| Output_SIn_Assortment_GC    : "Step06-1: Reuse Step03 (S/In GC)"
    df_fn_ASN_like_from_Actual           ||..|| Output_SIn_Assortment_AP2   : "Step06-1: Reuse Step03 (S/In AP2)"
    df_fn_ASN_like_from_Actual           ||..|| Output_SIn_Assortment_AP1   : "Step06-1: Reuse Step03 (S/In AP1)"
    df_fn_ASN_like_from_Actual           ||..|| Output_SIn_Assortment_Local : "Step06-1: Reuse Step03 (S/In Local)"
    Output_SIn_Assortment_GC             ||..|| Output_SOut_Assortment_GC   : "Step06-1: Reuse Step04 + concat"
    Output_SIn_Assortment_AP2            ||..|| Output_SOut_Assortment_AP2  : "Step06-1: Reuse Step04 + concat"
    Output_SIn_Assortment_AP1            ||..|| Output_SOut_Assortment_AP1  : "Step06-1: Reuse Step04 + concat"
    Output_SIn_Assortment_Local          ||..|| Output_SOut_Assortment_Local: "Step06-1: Reuse Step04 + concat"

    %% Step07: Final formatter (Version 추가 + 컬럼 삭제 규칙 적용)
    Output_SOut_Assortment_GC    ||..|| Output_SOut_Assortment_GC    : "Step07: Add Version, drop Sales Product ASN"
    Output_SOut_Assortment_AP2   ||..|| Output_SOut_Assortment_AP2   : "Step07: Add Version, drop Sales Product ASN"
    Output_SOut_Assortment_AP1   ||..|| Output_SOut_Assortment_AP1   : "Step07: Add Version, drop Sales Product ASN"
    Output_SOut_Assortment_Local ||..|| Output_SOut_Assortment_Local : "Step07: Add Version, drop Sales Product ASN"
    Output_SIn_Assortment_GC     ||..|| Output_SIn_Assortment_GC     : "Step07: Add Version, drop (SOUT flag, ASN)"
    Output_SIn_Assortment_AP2    ||..|| Output_SIn_Assortment_AP2    : "Step07: Add Version, drop (SOUT flag, ASN)"
    Output_SIn_Assortment_AP1    ||..|| Output_SIn_Assortment_AP1    : "Step07: Add Version, drop (SOUT flag, ASN)"
    Output_SIn_Assortment_Local  ||..|| Output_SIn_Assortment_Local  : "Step07: Add Version, drop (SOUT flag, ASN)"

```

 
# 전체 Sequence
 ``` mermaid

%% =========================================================

%% PYForecastCreateSellInAndSellOutAssortment

%% Sequence Diagram (Overall)

%% =========================================================

sequenceDiagram

    autonumber

    actor O9 as o9(Caller)

    participant MAIN as PYForecastCreateSellInAndSellOutAssortment.main

    participant S01 as step01_load_shipto_dimension

    participant S02 as step02_preprocess_asn

    participant S03 as step03_create_sellin_assortments

    participant S04 as step04_create_sellout_assortments

    participant S05 as step05_create_sellout_assortments_estore

    participant S06 as step06_create_sellout_assortments_from_actual

    participant FMT as fn_output_formatter

    participant LOG as fn_log_dataframe

    O9->>MAIN: run(version, inputs...)

    MAIN->>S01: Step01(df_in_Sales_Domain_Dimension)

    S01-->>MAIN: df_fn_shipto_dim

    MAIN->>S02: Step02(df_in_Sales_Product_ASN,\n df_in_Item_Master,\n df_in_Sell_Out_Simul_Master,\n df_in_Sales_Domain_Estore,\n df_fn_shipto_dim)

    S02-->>MAIN: df_fn_Sales_Product_ASN (Ship,Item,Loc,ASN,PG,SOUT_ASS_FLAG)

    MAIN->>S03: Step03(df_fn_Sales_Product_ASN,\n df_in_Forecast_Rule,\n df_fn_shipto_dim)

    S03-->>MAIN: df_sin_dict (GC/AP2/AP1/Local)

    MAIN->>S04: Step04(df_sin_dict)

    S04-->>MAIN: df_sout_dict (GC/AP2/AP1/Local)

    MAIN->>S05: Step05(df_fn_Sales_Product_ASN,\n df_in_Forecast_Rule,\n df_fn_shipto_dim,\n df_in_Sales_Domain_Estore,\n df_sout_dict)

    S05-->>MAIN: df_sout_dict (updated in-place)

    MAIN->>S06: Step06(df_in_Sell_Out_Actual_NoeStore_Past12,\n df_sout_dict,\n df_in_Item_Master,\n df_in_Sell_Out_Simul_Master,\n df_in_Sales_Domain_Estore,\n df_fn_shipto_dim,\n df_in_Forecast_Rule)

    S06-->>MAIN: df_sout_dict (updated in-place)

    MAIN->>FMT: Step07(Version, df_sin_dict, df_sout_dict)

    FMT-->>MAIN: df_final (outputs)

    loop log each output df

        MAIN->>LOG: fn_log_dataframe(df, name)

    end

    MAIN-->>O9: return df_final
```

## Step01 Sequence
``` mermaid
%% =========================================================

%% Step01 Sequence

%% =========================================================


sequenceDiagram

    autonumber

    participant MAIN as main

    participant S01 as step01_load_shipto_dimension

    MAIN->>S01: df_in_Sales_Domain_Dimension

    note over S01: Build ShipTo→Std1~Std6 + LV_CODE LUT

    S01-->>MAIN: df_fn_shipto_dim
```
## Step02 Sequence
``` mermaid
%% =========================================================

%% Step02 Sequence

%% =========================================================

sequenceDiagram

    autonumber

    participant MAIN as main

    participant S02 as step02_preprocess_asn

    participant ITEM as df_in_Item_Master

    participant DIM as df_fn_shipto_dim

    participant MAS as df_in_Sell_Out_Simul_Master

    participant EST as df_in_Sales_Domain_Estore

    MAIN->>S02: df_in_Sales_Product_ASN + ITEM + MAS + EST + DIM

    note over S02: Keep cols(Ship,Item,Loc,ASN)\n+ join PG\n+ derive _STD5 via DIM\n+ SOUT_ASS_FLAG: CON set true else false\n+ eStore ShipTo => false

    S02-->>MAIN: df_fn_Sales_Product_ASN
```
## Step03 Sequence
``` mermaid
%% =========================================================

%% Step03 Sequence

%% =========================================================

sequenceDiagram

    autonumber

    participant MAIN as main

    participant S03 as step03_create_sellin_assortments

    participant LUT as df_fn_shipto_dim

    participant RULE as df_in_Forecast_Rule

    participant UFG as ultra_fast_groupby_numpy_general

    MAIN->>S03: df_fn_Sales_Product_ASN + RULE + LUT

    note over S03: Vector build: ancestor table + RULE reindex\nTag lv 결정(GC/AP2/AP1/Local)\nShipTo를 조상으로 변환

    loop for each TAG (GC, AP2, AP1, Local)

        S03->>UFG: groupby([ShipTo,Item,Loc])\nagg(max flags)\nqty=1

        UFG-->>S03: df_tag

    end

    S03-->>MAIN: df_sin_dict
```
## Step04 Sequence
``` mermaid
%% =========================================================

%% Step04 Sequence

%% =========================================================

sequenceDiagram

    autonumber

    participant MAIN as main

    participant S04 as step04_create_sellout_assortments

    participant UFG as ultra_fast_groupby_numpy_general

    MAIN->>S04: df_sin_dict

    note over S04: For each tag DF:\nfilter SOUT_ASS_FLAG==True\nGroupBy(ShipTo,Item) → Location='-'\nqty=1

    loop tags (GC/AP2/AP1/Local)

        S04->>UFG: groupby([ShipTo,Item])\nagg(max flags)\nLocation='-'\nqty=1

        UFG-->>S04: df_sout_tag

    end

    S04-->>MAIN: df_sout_dict
```

## Step05 Sequence (eStore)
``` mermaid
%% =========================================================

%% Step05 Sequence (eStore)

%% =========================================================

sequenceDiagram

    autonumber

    participant MAIN as main

    participant S05 as step05_create_sellout_assortments_estore

    participant RULE as df_in_Forecast_Rule

    participant LUT as df_fn_shipto_dim

    participant EST as df_in_Sales_Domain_Estore

    participant UFG as ultra_fast_groupby_numpy_general

    MAIN->>S05: df_fn_Sales_Product_ASN + RULE + LUT + EST + df_sout_dict

    note over S05: eStore ShipTo만 대상으로\nS/Out Assortment 생성 후\n기존 df_sout_dict와 concat+dedup

    S05->>UFG: concat+groupby key(ShipTo,Item,Loc)\nagg(max)\nqty=1

    UFG-->>S05: updated df_sout_dict

    S05-->>MAIN: df_sout_dict (reuse)
```
## tep06 Sequence (Actual No-eStore Past12)
``` mermaid
%% =========================================================

%% Step06 Sequence (Actual No-eStore Past12)

%% =========================================================

sequenceDiagram

    autonumber

    participant MAIN as main

    participant S06 as step06_create_sellout_assortments_from_actual

    participant ITEM as df_in_Item_Master

    participant S03 as step03_create_sellin_assortments

    participant S04 as step04_create_sellout_assortments

    participant UFG as ultra_fast_groupby_numpy_general

    MAIN->>S06: df_in_Sell_Out_Actual_NoeStore_Past12 + df_sout_dict + ITEM + ... + df_rule + df_ship_dim

    alt Input7 empty

        S06-->>MAIN: df_sout_dict (no change)

    else Input7 exists

        note over S06: Build ASN-like:\nShip,Item,Loc\nASN=pd.NA\nPG merge from ITEM\nSOUT_ASS_FLAG=True(for Step04 filter)

        S06->>S03: Step03(ASN-like, df_rule, df_ship_dim)

        S03-->>S06: df_sin_from_actual

        S06->>S04: Step04(df_sin_from_actual)

        S04-->>S06: df_sout_from_actual

        S06->>UFG: concat(base + new)\nkey(ShipTo,Item,Loc)\nagg(max)\nqty=1

        UFG-->>S06: df_sout_dict (updated)

        S06-->>MAIN: df_sout_dict (reuse)

    end
```

## Step07 Sequence (Formatter)
``` mermaid
%% =========================================================

%% Step07 Sequence (Formatter)

%% =========================================================

sequenceDiagram

    autonumber

    participant MAIN as main

    participant FMT as fn_output_formatter

    MAIN->>FMT: Version + df_sin_dict + df_sout_dict

    note over FMT: Add Version\nDrop columns per spec:\nSOut: drop Sales Product ASN\nSIn: drop SOUT_ASS_FLAG + Sales Product ASN

    FMT-->>MAIN: df_final (6~8 outputs depending on impl)
```



# Flow Chart
``` mermaid
%% =========================================================
%% Flowchart Step03 (with source link)
%% =========================================================
flowchart TD
    S([Step03 Start]) --> I[Input: df_fn_Sales_Product_ASN + df_in_Forecast_Rule + df_fn_shipto_dim]
    I --> P[Prepare LV_MAP / SHIP_LV / RULE index]
    P --> A[Vector build: Ancestor table 2-7]
    A --> L[Loop lv=1..6 only: build PG-parent unique pairs -> rule lookup -> update tag_lvmat]
    L --> T[Per TAG: valid filter -> ShipTo to ancestor ship vector]
    T --> G[Dedup by ShipTo,Item,Loc: ultra_fast_groupby_numpy_general -> qty=1]
    G --> R([Return df_sin_dict])

    %% 🔗 Step03 소스 링크 (라인 포함)
    %% Git UI에 따라 라인 앵커 포맷이 다를 수 있어:
    %% - GitHub 스타일: #L120-L260
    %% - GitLab 스타일: #L120-260
    click S "https://code.sdsdev.co.kr/cmine/NSCM_DP_UI_Develop/tree/b3221cb2f0ad0ac3a20022650354fb5330588313/PYForecastCreateSellInAndSellOutAssortment.py#L916-L1129" "Open Step03 source (adjust line range if needed)" _blank
    click P "https://code.sdsdev.co.kr/cmine/NSCM_DP_UI_Develop/tree/b3221cb2f0ad0ac3a20022650354fb5330588313/PYForecastCreateSellInAndSellOutAssortment.py#L933-L952"
    click A "https://code.sdsdev.co.kr/cmine/NSCM_DP_UI_Develop/tree/b3221cb2f0ad0ac3a20022650354fb5330588313/PYForecastCreateSellInAndSellOutAssortment.py#L978-L986"
    click L "https://code.sdsdev.co.kr/cmine/NSCM_DP_UI_Develop/tree/b3221cb2f0ad0ac3a20022650354fb5330588313/PYForecastCreateSellInAndSellOutAssortment.py#L996-L1038"
    click T "https://code.sdsdev.co.kr/cmine/NSCM_DP_UI_Develop/tree/b3221cb2f0ad0ac3a20022650354fb5330588313/PYForecastCreateSellInAndSellOutAssortment.py#L1051-L1074"
    click G "https://code.sdsdev.co.kr/cmine/NSCM_DP_UI_Develop/tree/b3221cb2f0ad0ac3a20022650354fb5330588313/PYForecastCreateSellInAndSellOutAssortment.py#L1091-L1109"

```
