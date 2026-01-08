# PYForecastSellInAndOutColorDelta

- 본 문서는 `nscm/src/PYForecastSellInAndOutColorDelta.py` 및 개발요청서 `개발요청서_Forecast_PYForecastSellInAndOutColorDelta_20251219.txt`를 기반으로 작성한다.
- 참고 문서(유사 파이프라인): `PYForecastSellInAndOutColorDeltaNumber.md`

---

# 0) 주의사항 / 소스-요청서 차이점

## 0-1) Delta Null 처리(Output 4) 구현 여부

- 개발요청서에는 `Output 4) df_output_RTS_EOS_Delta (2025.12.19 Delta Null 처리)` 및 `Step 1-0)`이 명시되어 있다.
- **현행 소스(`step01_preprocess_rts_eos_delta`)는**
  - `df_in_MST_RTS_EOS_Delta`의 `* Delta` 컬럼명을 base 컬럼으로 rename 하여
  - `df_output_RTS_EOS`(Output 3 역할로 보이는 DF)를 생성하고,
  - 작업용 `df_fn_RTS_EOS`(Version/RTS_ISVALID 제거)를 반환한다.
- 그러나 개발요청서의 `Step 1-0) 모든 Measure를 null 처리한 df_output_RTS_EOS_Delta`에 해당하는 **별도 함수/출력은 확인되지 않는다**.

## 0-2) Delta 결측 보완(df_in_MST_RTS_EOS로 fill) 구현 여부

- 개발요청서에는 `df_in_MST_RTS_EOS_Delta`의 Delta 값이 비어있을 때 `df_in_MST_RTS_EOS`에서 동일 (Item, ShipTo) 조합의 값을 copy하여 보완하는 로직이 있다.
- **현행 소스의 `step01_preprocess_rts_eos_delta`에는 해당 fill 로직이 없다**.

## 0-3) o9(non-local) 입력 바인딩 주의

- `fn_process_in_df_mst()`에서 non-local(o9) 입력 바인딩이
  - `input_dataframes[STR_DF_RTS_EOS_DELTA] = df_in_MST_RTS_EOS`
  로 되어 있어, 이름상 Delta 키에 Full RTS/EOS가 들어간다.
- 로컬 CSV 로딩 경로에서는 `df_in_MST_RTS_EOS_Delta.csv`가 `STR_DF_RTS_EOS_DELTA`로 매핑되므로 **로컬/운영 동작이 달라질 수 있다**.

---

# 1) 전체 ERD

``` mermaid
%%{init: {'erDiagram':{'useMaxWidth':false}}}%%
erDiagram
  df_in_Sales_Domain_Dimension {
    category SHIP_TO
    category STD1
    category STD2
    category STD3
    category STD4
    category STD5
    category STD6
  }

  df_fn_shipto_dim {
    category SHIP_TO
    category STD1
    category STD2
    category STD3
    category STD4
    category STD5
    category STD6
    int LV_CODE
  }

  %% RTS/EOS Delta input
  df_in_MST_RTS_EOS_Delta {
    category VERSION
    category ITEM
    category SHIP_TO
    category RTS_ISVALID_Delta
    category RTS_STATUS_Delta
    category RTS_INIT_DATE_Delta
    category RTS_DEV_DATE_Delta
    category RTS_COM_DATE_Delta
    category EOS_STATUS_Delta
    category EOS_INIT_DATE_Delta
    category EOS_CHG_DATE_Delta
    category EOS_COM_DATE_Delta
  }

  %% RTS/EOS Full input (요청서 상 존재)
  df_in_MST_RTS_EOS {
    category VERSION
    category ITEM
    category SHIP_TO
    category RTS_ISVALID
    category RTS_STATUS
    category RTS_INIT_DATE
    category RTS_DEV_DATE
    category RTS_COM_DATE
    category EOS_STATUS
    category EOS_INIT_DATE
    category EOS_CHG_DATE
    category EOS_COM_DATE
  }

  %% Step01 output + Step02 derived helper
  df_output_RTS_EOS {
    category VERSION
    category ITEM
    category SHIP_TO
    category RTS_STATUS
    category RTS_INIT_DATE
    category RTS_DEV_DATE
    category RTS_COM_DATE
    category EOS_STATUS
    category EOS_INIT_DATE
    category EOS_CHG_DATE
    category EOS_COM_DATE
  }

  df_fn_RTS_EOS {
    category ITEM
    category SHIP_TO
    category RTS_STATUS
    category RTS_INIT_DATE
    category RTS_DEV_DATE
    category RTS_COM_DATE
    category EOS_STATUS
    category EOS_INIT_DATE
    category EOS_CHG_DATE
    category EOS_COM_DATE

    category RTS_PARTIAL_WEEK
    category EOS_PARTIAL_WEEK

    category RTS_WEEK
    category EOS_WEEK
    category RTS_INITIAL_WEEK
    category EOS_INITIAL_WEEK

    category RTS_WEEK_MINUST_1
    category RTS_WEEK_PLUS_3

    category EOS_WEEK_MINUS_1
    category EOS_WEEK_MINUS_3
    category EOS_WEEK_MINUS_4

    category MAX_RTS_CURRENTWEEK
    category MIN_EOSINI_MAXWEEK
    category MIN_EOS_MAXWEEK
  }

  df_in_Time_Partial_Week {
    category TIME_PW
  }

  df_in_Sales_Product_ASN {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOC
    category ASN_FLAG
  }

  df_in_Item_Master {
    category ITEM
    category PT
    category GBM
    category PG
  }

  df_in_Item_CLASS {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOC
    category CLASS
  }

  df_in_Item_TAT {
    category VERSION
    category ITEM
    category LOC
    int TATTERM
    int TAT_SET
  }

  df_fn_Sales_Product_ASN_Item {
    category SHIP_TO
    category STD5
    category ITEM
    category LOC
    category GBM
    category PG
    bool HA_EOP_FLAG
    int TATTERM
    int TAT_SET
  }

  df_fn_Sales_Product_ASN_Group {
    category SHIP_TO
    category ITEM
    bool HA_EOP_FLAG
    int TATTERM
    int TAT_SET
  }

  df_fn_ASN_Group_Week {
    category SHIP_TO
    category ITEM
    category TIME_PW
    category CURRENT_ROW_WEEK
    category SIN_FCST_COLOR_COND
  }

  df_in_SELLOUTFCST_NOTEXIST {
    category STD2
    category ITEM
    bool SOUT_FCST_NOT_EXISTS
  }

  df_output_Sell_In_FCST_Color_Condition {
    category VERSION
    category STD2
    category ITEM
    category TIME_PW
    category SIN_FCST_COLOR_COND
  }

  df_output_Sell_Out_FCST_Color_Condition {
    category VERSION
    category STD2
    category ITEM
    category TIME_PW
    category SOUT_FCST_COLOR_COND
  }

  %% Relationships (key-level)
  df_in_Sales_Domain_Dimension ||--o{ df_fn_shipto_dim : "SHIP_TO -> LUT"

  df_in_MST_RTS_EOS_Delta ||--o{ df_output_RTS_EOS : "Step01: rename *Delta -> base + VERSION"
  df_in_MST_RTS_EOS_Delta ||--o{ df_fn_RTS_EOS : "Step01 -> Step02"
  df_in_Time_Partial_Week ||--o{ df_fn_RTS_EOS : "Step02: helper calc uses max week"
  df_fn_shipto_dim ||--o{ df_fn_RTS_EOS : "Step02: Lv2 -> Lv3 fan-out"

  df_in_Sales_Product_ASN ||--o{ df_fn_Sales_Product_ASN_Item : "Step03-1"
  df_in_Item_Master ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM -> GBM/PG"
  df_in_Item_CLASS ||--o{ df_fn_Sales_Product_ASN_Item : "(ITEM,STD5,LOC) -> HA_EOP_FLAG"
  df_in_Item_TAT ||--o{ df_fn_Sales_Product_ASN_Item : "(ITEM,LOC) -> TATTERM/TAT_SET"
  df_fn_RTS_EOS ||--o{ df_fn_Sales_Product_ASN_Item : "Delta filter: keep only items present in RTS/EOS"

  df_fn_Sales_Product_ASN_Item ||--o{ df_fn_Sales_Product_ASN_Group : "Step03-2 group by (STD2,ITEM)"
  df_fn_RTS_EOS ||--o{ df_fn_Sales_Product_ASN_Group : "Delta filter: keep only (STD2,ITEM) pairs in RTS/EOS"

  df_in_Time_Partial_Week ||--o{ df_fn_ASN_Group_Week : "Step04 cross join"
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "Step04 base population"
  df_fn_RTS_EOS ||--o{ df_fn_ASN_Group_Week : "Step05 merge and apply colors"
  df_in_SELLOUTFCST_NOTEXIST ||--o{ df_fn_ASN_Group_Week : "Step09 gray override"

  df_fn_ASN_Group_Week ||--o{ df_output_Sell_In_FCST_Color_Condition : "Step10 project + rename SHIP_TO->STD2"
  df_fn_ASN_Group_Week ||--o{ df_output_Sell_Out_FCST_Color_Condition : "Step11 recompute subset (전체 판매처)"
```

---

# 1) 전체 ERD PK

``` mermaid
erDiagram
  df_fn_shipto_dim {
    category SHIP_TO PK
    int LV_CODE
  }

  df_in_Time_Partial_Week {
    category TIME_PW PK
  }

  df_in_MST_RTS_EOS_Delta {
    category VERSION PK
    category ITEM PK
    category SHIP_TO PK
  }

  df_fn_RTS_EOS {
    category ITEM PK
    category SHIP_TO PK
  }

  df_in_Sales_Product_ASN {
    category SHIP_TO PK
    category ITEM PK
    category LOC PK
  }

  df_fn_Sales_Product_ASN_Item {
    category SHIP_TO PK
    category ITEM PK
    category LOC PK
  }

  df_fn_Sales_Product_ASN_Group {
    category SHIP_TO PK
    category ITEM PK
  }

  df_fn_ASN_Group_Week {
    category SHIP_TO PK
    category ITEM PK
    category TIME_PW PK
  }

  df_output_Sell_In_FCST_Color_Condition {
    category VERSION PK
    category STD2 PK
    category ITEM PK
    category TIME_PW PK
  }

  df_output_Sell_Out_FCST_Color_Condition {
    category VERSION PK
    category STD2 PK
    category ITEM PK
    category TIME_PW PK
  }
```

---

# 2) Step별 서브 ERD

## Step 00: Ship-To 차원 LUT 구축

``` mermaid
erDiagram
  df_in_Sales_Domain_Dimension ||--o{ df_fn_shipto_dim : "SHIP_TO -> LV_CODE/STD1..6"
```

## Step 01: RTS/EOS Delta 전처리

``` mermaid
erDiagram
  df_in_MST_RTS_EOS_Delta ||--o{ df_fn_RTS_EOS : "rename '* Delta' -> base + drop VERSION/RTS_ISVALID"
  df_in_MST_RTS_EOS_Delta ||--o{ df_output_RTS_EOS : "Output(3)용 보존본 (VERSION 포함)"
```

## Step 02: Partial Week 변환 + helper

``` mermaid
erDiagram
  df_fn_RTS_EOS ||--o{ df_fn_RTS_EOS : "date->partialWeek + week helper"
  df_in_Time_Partial_Week ||--o{ df_fn_RTS_EOS : "max week (helper bound)"
  df_fn_shipto_dim ||--o{ df_fn_RTS_EOS : "Lv2 -> Lv3 fan-out"
```

## Step 03-1: ASN 전처리(Delta 범위 축소)

``` mermaid
erDiagram
  df_in_Sales_Product_ASN ||--o{ df_fn_Sales_Product_ASN_Item : "SHIP_TO+ITEM+LOC"
  df_in_Sales_Domain_Dimension ||--o{ df_fn_Sales_Product_ASN_Item : "SHIP_TO -> STD5"
  df_in_Item_Master ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM -> GBM/PG"
  df_in_Item_CLASS ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM+LOC+STD5 -> HA_EOP_FLAG"
  df_in_Item_TAT ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM+LOC -> TATTERM/TAT_SET"
  df_fn_RTS_EOS ||--o{ df_fn_Sales_Product_ASN_Item : "filter: keep only items in RTS/EOS"
```

## Step 03-2: Std2×Item 그룹핑(Delta 범위 축소)

``` mermaid
erDiagram
  df_fn_Sales_Product_ASN_Item ||--o{ df_fn_Sales_Product_ASN_Group : "SHIP_TO -> STD2 then group"
  df_in_Sales_Domain_Dimension ||--o{ df_fn_Sales_Product_ASN_Group : "SHIP_TO -> STD2 mapping"
  df_fn_RTS_EOS ||--o{ df_fn_Sales_Product_ASN_Group : "filter: keep only (STD2,ITEM) in RTS/EOS"
```

## Step 04: Week fan-out (Std2×Item × PartialWeek)

``` mermaid
erDiagram
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "cross with TIME_PW"
  df_in_Time_Partial_Week ||--o{ df_fn_ASN_Group_Week : "TIME_PW"
```

## Step 05: RTS/EOS Color 적용

``` mermaid
erDiagram
  df_fn_RTS_EOS ||--o{ df_fn_ASN_Group_Week : "merge by (ITEM,SHIP_TO) then apply masks"
```

## Step 06: Wireless BAS GREEN

``` mermaid
erDiagram
  df_in_Item_Master ||--o{ df_fn_ASN_Group_Week : "(PT=='BAS' & GBM=='MOBILE') within 8w => GREEN"
```

## Step 07: HA-EOP YELLOW

``` mermaid
erDiagram
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "HA_EOP_FLAG targets"
  df_fn_RTS_EOS ||--o{ df_fn_ASN_Group_Week : "EOS_WEEK/EOS_WEEK_MINUS_4"
```

## Step 08-1: VD/SHA Lead-time DGRAY_RED

``` mermaid
erDiagram
  df_fn_Sales_Product_ASN_Item ||--o{ df_fn_ASN_Group_Week : "ITEM->GBM"
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "TATTERM range"
```

## Step 08-2: SET Lead-time DGRAY_REDB

``` mermaid
erDiagram
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "TAT_SET promotes DGRAY_REDB"
```

## Step 09: No Sell-out GRAY override

``` mermaid
erDiagram
  df_in_SELLOUTFCST_NOTEXIST ||--o{ df_fn_ASN_Group_Week : "flagged (STD2,ITEM) => GRAY"
```

## Step 10: Sell-In Output

``` mermaid
erDiagram
  df_fn_ASN_Group_Week ||--o{ df_output_Sell_In_FCST_Color_Condition : "SHIP_TO->STD2 + VERSION"
```

## Step 11: Sell-Out Output(전체 판매처)

``` mermaid
erDiagram
  df_fn_RTS_EOS ||--o{ df_output_Sell_Out_FCST_Color_Condition : "recompute (apply_eos_red=False)"
  df_in_Time_Partial_Week ||--o{ df_output_Sell_Out_FCST_Color_Condition : "cross join"
  df_in_Item_Master ||--o{ df_output_Sell_Out_FCST_Color_Condition : "GREEN 대상"
```

---

# 3) 전체 Sequence Diagram

``` mermaid
sequenceDiagram
  autonumber
  participant Main
  participant IO as fn_process_in_df_mst
  participant S00 as step00_load_shipto_dimension
  participant S01 as step01_preprocess_rts_eos_delta
  participant S02 as step02_convert_date_to_partial_week
  participant S31 as step03_1_prepare_asn_item_delta
  participant S32 as step03_2_group_asn_to_ap2_item_delta
  participant S04 as step04_build_rts_eos_week
  participant S05 as step05_apply_color_rts_eos
  participant S06 as step06_apply_green_for_wireless_bas
  participant S07 as step07_apply_ha_eop_yellow
  participant S81 as step08_1_apply_vd_leadtime
  participant S82 as step08_2_apply_set_leadtime
  participant S09 as step09_apply_gray_no_sellout
  participant S10 as step10_build_sell_in_output
  participant S11 as step11_extend_sell_out_output

  Main->>IO: load inputs to input_dataframes
  IO-->>Main: input_dataframes ready

  Main->>S00: df_in_Sales_Domain_Dimension
  S00-->>Main: df_fn_shipto_dim

  Main->>S01: df_in_MST_RTS_EOS_Delta + Version
  S01-->>Main: df_fn_RTS_EOS + df_output_RTS_EOS

  Main->>S02: df_fn_RTS_EOS + CurrentPartialWeek
  S02-->>Main: df_fn_RTS_EOS (with PW + helper)

  Main->>S31: ASN + DIM + ITEM_MST + ITEM_CLASS + ITEM_TAT + df_fn_RTS_EOS
  S31-->>Main: df_fn_Sales_Product_ASN_Item

  Main->>S32: df_fn_Sales_Product_ASN_Item + df_fn_shipto_dim + df_fn_RTS_EOS
  S32-->>Main: df_fn_Sales_Product_ASN_Group

  Main->>S04: df_fn_Sales_Product_ASN_Group + Time_PW + CurrentPartialWeek
  S04-->>Main: df_fn_ASN_Group_Week

  Main->>S05: df_fn_ASN_Group_Week + df_fn_RTS_EOS + CurrentPartialWeek
  S05-->>Main: df_fn_ASN_Group_Week (colors)

  Main->>S06: df_fn_ASN_Group_Week + Item_Master + CurrentPartialWeek
  S06-->>Main: df_fn_ASN_Group_Week

  Main->>S07: df_fn_ASN_Group_Week + df_fn_RTS_EOS + df_fn_Sales_Product_ASN_Group + CurrentPartialWeek
  S07-->>Main: df_fn_ASN_Group_Week

  Main->>S81: df_fn_ASN_Group_Week + df_fn_Sales_Product_ASN_Item + df_fn_Sales_Product_ASN_Group + CurrentPartialWeek
  S81-->>Main: df_fn_ASN_Group_Week

  Main->>S82: df_fn_ASN_Group_Week + df_fn_Sales_Product_ASN_Item + df_fn_Sales_Product_ASN_Group + CurrentPartialWeek
  S82-->>Main: df_fn_ASN_Group_Week

  Main->>S09: df_fn_ASN_Group_Week + df_in_SELLOUTFCST_NOTEXIST + CurrentPartialWeek
  S09-->>Main: df_fn_ASN_Group_Week

  Main->>S10: df_fn_ASN_Group_Week + Version
  S10-->>Main: df_output_Sell_In_FCST_Color_Condition

  Main->>S11: df_fn_RTS_EOS + Time_PW + DIM + Item inputs + ASN inputs + CurrentPartialWeek + Version
  S11-->>Main: df_output_Sell_Out_FCST_Color_Condition
```

---

# 4) Step별 상세 Sequence

## Step 01: RTS/EOS Delta 전처리

``` mermaid
sequenceDiagram
  autonumber
  participant S01 as step01_preprocess_rts_eos_delta
  participant IN as df_in_MST_RTS_EOS_Delta
  participant OUT1 as df_fn_RTS_EOS
  participant OUT2 as df_output_RTS_EOS

  S01->>IN: select VERSION, ITEM, SHIP_TO + available '* Delta' measure cols
  S01->>S01: rename '* Delta' -> base column names
  S01->>S01: build df_output_RTS_EOS (VERSION injected/normalized)
  S01->>S01: build df_fn_RTS_EOS (drop VERSION, RTS_ISVALID)
  S01-->>OUT1: df_fn_RTS_EOS
  S01-->>OUT2: df_output_RTS_EOS
```

## Step 03-1: ASN 전처리(Delta filter)

``` mermaid
sequenceDiagram
  autonumber
  participant S31 as step03_1_prepare_asn_item_delta
  participant ASN as df_in_Sales_Product_ASN
  participant RTS as df_fn_RTS_EOS
  participant DIM as df_in_Sales_Domain_Dimension
  participant MST as df_in_Item_Master
  participant CLS as df_in_Item_CLASS
  participant TAT as df_in_Item_TAT
  participant OUT as df_fn_Sales_Product_ASN_Item

  S31->>ASN: select (SHIP_TO, ITEM, LOC)
  S31->>RTS: filter keep only ITEM in RTS/EOS scope
  S31->>DIM: merge SHIP_TO -> STD5
  S31->>MST: merge ITEM -> GBM, PG
  S31->>CLS: filter CLASS=='X' then join (ITEM,STD5,LOC) -> HA_EOP_FLAG
  S31->>TAT: join (ITEM,LOC) -> TATTERM/TAT_SET (fillna 0)
  S31-->>OUT: df_fn_Sales_Product_ASN_Item
```

## Step 03-2: Std2×Item 그룹핑(Delta filter)

``` mermaid
sequenceDiagram
  autonumber
  participant S32 as step03_2_group_asn_to_ap2_item_delta
  participant IN as df_fn_Sales_Product_ASN_Item
  participant DIM as df_fn_shipto_dim
  participant RTS as df_fn_RTS_EOS
  participant OUT as df_fn_Sales_Product_ASN_Group

  S32->>DIM: build ShipTo -> Std2 lookup
  S32->>IN: replace SHIP_TO with STD2 (fallback to SHIP_TO when missing)
  S32->>S32: groupby (SHIP_TO(=STD2), ITEM) max(HA_EOP_FLAG, TATTERM, TAT_SET)
  S32->>RTS: build distinct (SHIP_TO,ITEM) pairs after Std2 normalization
  S32->>S32: inner-join to keep only RTS/EOS pairs
  S32-->>OUT: df_fn_Sales_Product_ASN_Group
```

## Step 11: Sell-Out Output(전체 판매처)

``` mermaid
sequenceDiagram
  autonumber
  participant S11 as step11_extend_sell_out_output
  participant RTS as df_fn_RTS_EOS
  participant TIME as df_in_Time_Partial_Week
  participant MST as df_in_Item_Master
  participant OUT as df_output_Sell_Out_FCST_Color_Condition

  S11->>S11: reuse global df_fn_Sales_Product_ASN_Item/Group as base population
  S11->>S11: Step04 (week grid)
  S11->>S11: Step05 apply RTS colors (apply_eos_red=False)
  S11->>S11: Step06 GREEN
  S11->>S11: Step08-1 leadtime DGRAY_RED
  S11->>S11: rename SHIP_TO->STD2, SIN->SOUT
  S11->>S11: inject VERSION
  S11-->>OUT: df_output_Sell_Out_FCST_Color_Condition
```

---

# 5) Flowchart

## 5-1) 전체 Flowchart

``` mermaid
flowchart TD
  A[Start
load df_in_*] --> S00[Step00 ShipTo LUT]
  S00 --> S01[Step01 RTS/EOS Delta preprocess]
  S01 --> S02[Step02 dates->PartialWeek + helper]

  S02 --> S31[Step03-1 ASN item preprocess
Delta ITEM filter]
  S31 --> S32[Step03-2 Std2×Item group
Delta pair filter]

  S32 --> S04[Step04 week fan-out]
  S04 --> S05[Step05 RTS/EOS color]
  S05 --> S06[Step06 BAS GREEN]
  S06 --> S07[Step07 HA-EOP YELLOW]
  S07 --> S81[Step08-1 VD/SHA leadtime]
  S81 --> S82[Step08-2 SET leadtime]
  S82 --> S09[Step09 No-sellout GRAY]

  S09 --> S10[Step10 Build Sell-In output]
  S10 --> S11[Step11 Build Sell-Out output (non-eStore)]
  S11 --> Z[End Outputs]
```

## 5-2) Step03-2(Delta filter) Flowchart

``` mermaid
flowchart TD
  A[ASN Item rows
(ShipTo,Item,Loc,...)] --> B[Map ShipTo -> Std2]
  B --> C[GroupBy (Std2,Item)
max(HA_EOP, TATTERM, TAT_SET)]
  C --> D[Build RTS pairs
from Step02 (ShipTo->Std2 normalize)]
  D --> E[Inner Join
keep only RTS pairs]
  E --> F[ASN Group output]
```

---

# 6) 산출물(Output) 정리

- (Output 1) `df_output_Sell_In_FCST_Color_Condition`
  - Grain: `Version + Sales Std2 + Item + Partial Week`
  - Measure: `S/In FCST Color Condition`
- (Output 2) `df_output_Sell_Out_FCST_Color_Condition`
  - Grain: `Version + Sales Std2 + Item + Partial Week`
  - Measure: `S/Out FCST Color Condition`
- (Output 3) `df_output_RTS_EOS`
  - Grain: `Version + Ship To + Item`
  - Measure: RTS/EOS 상태/일자 컬럼(요청서 기준)
- (Output 4) `df_output_RTS_EOS_Delta`
  - 개발요청서에 명시되어 있으나, **현행 소스에서 별도 output 생성/반환 로직은 확인되지 않는다**.
