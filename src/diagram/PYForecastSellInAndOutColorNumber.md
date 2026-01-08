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
  df_in_MST_RTS_EOS {
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
    category RTS_ISVALID
  }
  df_fn_RTS_EOS {
    category ITEM
    category SHIP_TO
    category RTS_STATUS
    category EOS_STATUS
    category RTS_INIT_DATE
    category RTS_DEV_DATE
    category RTS_COM_DATE
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
    category ITEM
    category LOC
    category STD5
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
    category CURRENT_ROW_WEEK_PLUS_8
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
  df_in_MST_RTS_EOS ||--o{ df_fn_RTS_EOS : "ITEM+SHIP_TO"
  df_in_Time_Partial_Week ||--o{ df_fn_RTS_EOS : "used for MAX week calc"
  df_in_Sales_Product_ASN ||--o{ df_fn_Sales_Product_ASN_Item : "SHIP_TO+ITEM+LOC"
  df_fn_shipto_dim ||--o{ df_fn_Sales_Product_ASN_Item : "SHIP_TO -> STD5"
  df_in_Item_Master ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM -> GBM/PG"
  df_in_Item_CLASS ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM+LOC+STD5 -> HA_EOP_FLAG"
  df_in_Item_TAT ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM+LOC -> TATTERM/TAT_SET"
  df_fn_Sales_Product_ASN_Item ||--o{ df_fn_Sales_Product_ASN_Group : "group by SHIP_TO+ITEM"
  df_fn_shipto_dim ||--o{ df_fn_Sales_Product_ASN_Group : "SHIP_TO -> parent(STD2)"
  df_in_Time_Partial_Week ||--o{ df_fn_ASN_Group_Week : "cross join by TIME_PW"
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "SHIP_TO+ITEM"
  df_fn_RTS_EOS ||--o{ df_fn_ASN_Group_Week : "merge by ITEM+SHIP_TO (color rules)"
  df_in_SELLOUTFCST_NOTEXIST ||--o{ df_fn_ASN_Group_Week : "STD2+ITEM (gray override)"
  df_fn_ASN_Group_Week ||--o{ df_output_Sell_In_FCST_Color_Condition : "projection + rename SHIP_TO->STD2"
  df_fn_ASN_Group_Week ||--o{ df_output_Sell_Out_FCST_Color_Condition : "recompute subset (Step11)"
```
# 1) 전체 ERD PK
``` mermaid
erDiagram
  %% =========================
  %% 1) DIM / TIME
  %% =========================
  df_fn_shipto_dim {
    category SHIP_TO PK
    category STD1
    category STD2
    category STD3
    category STD4
    category STD5
    category STD6
    int      LV_CODE
  }
  df_in_Time_Partial_Week {
    category TIME_PW PK
  }
  %% =========================
  %% 2) RTS/EOS (after Step02)
  %% =========================
  df_fn_RTS_EOS {
    category SHIP_TO PK        "(ITEM, SHIP_TO) 가 논리 PK"
    category ITEM    PK
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
    category MIN_EOS_MAXWEEK
    category RTS_DEV_DATE      "'YYYYWWS' (PartialWeek) -> slice(6) used"
  }
  %% =========================
  %% 3) INPUTS (ASN / ITEM)
  %% =========================
  df_in_Sales_Product_ASN {
    category SHIP_TO PK        "(SHIP_TO, ITEM, LOC) 가 논리 PK"
    category ITEM    PK
    category LOC     PK
  }
  df_in_Item_Master {
    category ITEM PK
    category GBM
    category PG
    category PT
  }
  df_in_Item_CLASS {
    category STD5 PK           "(ITEM, STD5, LOC, CLASS) 중, 필터 CLASS=='X' 후 (ITEM,STD5,LOC) 사용"
    category ITEM PK
    category LOC  PK
    category CLASS
  }
  df_in_Item_TAT {
    category ITEM PK           "(ITEM, LOC) 가 논리 PK"
    category LOC  PK
    int      TATTERM
    int      TAT_SET
  }
  %% =========================
  %% 4) Step03-1 output
  %% =========================
  df_fn_Sales_Product_ASN_Item {
    category SHIP_TO PK        "논리 PK: (SHIP_TO, ITEM, LOC)"
    category ITEM    PK
    category LOC     PK
    category STD5    FK        "FK -> df_fn_shipto_dim.STD5 (via SHIP_TO mapping)"
    category GBM     FK        "FK -> df_in_Item_Master.GBM (via ITEM)"
    category PG      FK        "FK -> df_in_Item_Master.PG  (via ITEM)"
    bool     HA_EOP_FLAG       "derived from df_in_Item_CLASS (CLASS=='X') existence"
    int      TATTERM           "from df_in_Item_TAT"
    int      TAT_SET           "from df_in_Item_TAT"
  }
  %% =========================
  %% 5) Step03-2 output
  %% =========================
  df_fn_Sales_Product_ASN_Group {
    category SHIP_TO PK        "논리 PK: (SHIP_TO, ITEM) ; SHIP_TO는 parent STD2(=Lv3)로 치환됨"
    category ITEM    PK
    bool     HA_EOP_FLAG
    int      TATTERM
    int      TAT_SET
  }
  %% =========================
  %% 6) Week fan-out (Step04+)
  %% =========================
  df_fn_ASN_Group_Week {
    category SHIP_TO PK        "논리 PK: (SHIP_TO, ITEM, TIME_PW)"
    category ITEM    PK
    category TIME_PW PK
    category CURRENT_ROW_WEEK  "TIME_PW[:6]"
    category SIN_FCST_COLOR_COND
  }
  %% =========================
  %% 7) No-sellout flag
  %% =========================
  df_in_SELLOUTFCST_NOTEXIST {
    category STD2 PK           "논리 PK: (STD2, ITEM)"
    category ITEM PK
    bool     SOUT_FCST_NOT_EXISTS
  }
  %% =========================
  %% 8) Outputs
  %% =========================
  df_output_Sell_In_FCST_Color_Condition {
    category VERSION PK        "논리 PK: (VERSION, STD2, ITEM, TIME_PW)"
    category STD2    PK
    category ITEM    PK
    category TIME_PW PK
    category SIN_FCST_COLOR_COND
  }
  df_output_Sell_Out_FCST_Color_Condition {
    category VERSION PK        "논리 PK: (VERSION, STD2, ITEM, TIME_PW)"
    category STD2    PK
    category ITEM    PK
    category TIME_PW PK
    category SOUT_FCST_COLOR_COND
  }
  %% ============================================================
  %% RELATIONSHIPS (엄격 FK 라벨링)
  %% ============================================================
  %% ASN -> ShipTo Dim (SHIP_TO -> STD5 lookup via dim)
  df_fn_shipto_dim ||--o{ df_fn_Sales_Product_ASN_Item : "FK: Sales_Product_ASN_Item.SHIP_TO -> shipto_dim.SHIP_TO (get STD5)"
  %% ASN Item -> Item Master
  df_in_Item_Master ||--o{ df_fn_Sales_Product_ASN_Item : "FK: ASN_Item.ITEM -> Item_Master.ITEM (GBM,PG,PT)"
  %% ASN Item -> Item Class (existence join)
  df_in_Item_CLASS ||--o{ df_fn_Sales_Product_ASN_Item : "FK (existence): (ITEM,STD5,LOC) where CLASS=='X'"
  %% ASN Item -> Item TAT
  df_in_Item_TAT ||--o{ df_fn_Sales_Product_ASN_Item : "FK: (ITEM,LOC) -> (ITEM,LOC) (TATTERM,TAT_SET)"
  %% ASN raw -> ASN Item
  df_in_Sales_Product_ASN ||--o{ df_fn_Sales_Product_ASN_Item : "PK propagate: (SHIP_TO,ITEM,LOC)"
  %% Grouping
  df_fn_Sales_Product_ASN_Item ||--o{ df_fn_Sales_Product_ASN_Group : "GroupBy: (parent(SHIP_TO=STD2), ITEM) ; max(HA_EOP/TAT)"
  %% Week fan-out
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "FK: (SHIP_TO,ITEM) -> (SHIP_TO,ITEM)"
  df_in_Time_Partial_Week      ||--o{ df_fn_ASN_Group_Week : "FK: TIME_PW -> TIME_PW (cross join)"
  %% Color apply (RTS/EOS merge)
  df_fn_RTS_EOS ||--o{ df_fn_ASN_Group_Week : "FK: (ITEM,SHIP_TO) -> (ITEM,SHIP_TO) (color rules)"
  %% No-sellout gray override (STD2 == SHIP_TO)
  df_in_SELLOUTFCST_NOTEXIST ||--o{ df_fn_ASN_Group_Week : "FK: (STD2,ITEM) -> (SHIP_TO,ITEM)"
  %% Outputs
  df_fn_ASN_Group_Week ||--o{ df_output_Sell_In_FCST_Color_Condition  : "Projection: SHIP_TO->STD2 + VERSION"
  df_fn_ASN_Group_Week ||--o{ df_output_Sell_Out_FCST_Color_Condition : "Recompute subset + rename SIN->SOUT + VERSION"
```
## 2) Step별 서브 ERD
## Step 00: Ship-To 차원 LUT 구축
``` mermaid
erDiagram
  df_in_Sales_Domain_Dimension ||--o{ df_fn_shipto_dim : "SHIP_TO -> LV_CODE/STD1..6"
```
## Step 01: RTS/EOS 전처리
``` mermaid
erDiagram
  df_in_MST_RTS_EOS ||--o{ df_fn_RTS_EOS : "drop VERSION/RTS_ISVALID + slim columns"
```
## Step 02: Step1의 Result에 Time을 Partial Week 으로 변환
``` mermaid
erDiagram
  df_fn_shipto_dim ||--o{ df_fn_RTS_EOS : "LV2 fan-out to LV3"
  df_in_Time_Partial_Week ||--o{ df_fn_RTS_EOS : "MAX week for helper calc"
```
## Step 03-1 : df_in_Sales_Product_ASN 전처리
``` mermaid
erDiagram
  df_in_Sales_Product_ASN ||--o{ df_fn_Sales_Product_ASN_Item : "SHIP_TO+ITEM+LOC"
  df_fn_shipto_dim ||--o{ df_fn_Sales_Product_ASN_Item : "SHIP_TO -> STD5"
  df_in_Item_Master ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM -> GBM/PG"
  df_in_Item_CLASS ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM+LOC+STD5 -> HA_EOP_FLAG"
  df_in_Item_TAT ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM+LOC -> TATTERM/TAT_SET"
```
## Step 03-2: AP2(Lv-2/3) × Item 단위 그룹핑. df_fn_Sales_Product_ASN_Group 생성
``` mermaid
erDiagram
  df_fn_Sales_Product_ASN_Item ||--o{ df_fn_Sales_Product_ASN_Group : "group (SHIP_TO->parent STD2), ITEM"
  df_fn_shipto_dim ||--o{ df_fn_Sales_Product_ASN_Group : "SHIP_TO -> STD2 mapping"
```
## Step 04: RTS/EOS (Lv3 × Item) → ( +Partial-Week ) Fan-out. 기본 Color = 19_GRAY
``` mermaid
erDiagram
  df_in_Time_Partial_Week ||--o{ df_fn_ASN_Group_Week : "cross join by TIME_PW"
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "SHIP_TO+ITEM"
```
## Step 05: RTS / EOS Color 반영. 
### - 규칙 5-1 ~ 5-5 (WHITE / DARKBLUE / LIGHTBLUE / LIGHTRED / DARKRED)
### - 벡터라이즈 & 1-pass overwrite (19_GRAY → WHITE → … → DARKRED)
``` mermaid
erDiagram
  df_fn_RTS_EOS ||--o{ df_fn_ASN_Group_Week : "ITEM+SHIP_TO merge (RTS/EOS color)"
```
## Step 06: Wireless BAS 모델 – 당주 포함 8 주 구간 13_GREEN 적용
``` mermaid
erDiagram
  df_in_Item_Master ||--o{ df_fn_ASN_Group_Week : "BAS & MOBILE -> COLOR_GREEN override"
```
## Step 07: HA-EOP Management 모델 12_YELLOW 반영
``` mermaid
erDiagram
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "HA_EOP_FLAG targets"
  df_fn_RTS_EOS ||--o{ df_fn_ASN_Group_Week : "EOS ranges for COLOR_YELLOW"
```
## Step 08-1: VD / SHA  Lead-Time (18_DGRAY_RED) 갱신
``` mermaid
erDiagram
  df_fn_Sales_Product_ASN_Item ||--o{ df_fn_ASN_Group_Week : "GBM (VD/SHA) lookup"
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "TATTERM -> COLOR_DGRAY_RED"
```
## Step 08-2: SET Lead-Time 구간 17_DGRAY_REDB 갱신
``` mermaid
erDiagram
  df_fn_Sales_Product_ASN_Item ||--o{ df_fn_ASN_Group_Week : "GBM (VD/SHA) lookup"
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "TAT_SET promotes COLOR_DGRAY_REDB"
```
## Step 09: MX  Sell-out Forecast  없는  모델  GRAY  업데이트
``` mermaid
erDiagram
  df_in_SELLOUTFCST_NOTEXIST ||--o{ df_fn_ASN_Group_Week : "STD2+ITEM -> COLOR_GRAY override"
```
## Step 10: df_output_Sell_In_FCST_Color_Condition
``` mermaid
erDiagram
  df_fn_ASN_Group_Week ||--o{ df_output_Sell_In_FCST_Color_Condition : "SHIP_TO->STD2 projection"
```
## Step 11: df_output_Sell_Out_FCST_Color_Condition
``` mermaid
erDiagram
  df_fn_RTS_EOS ||--o{ df_output_Sell_Out_FCST_Color_Condition : "recompute (no EOS red)"
  df_in_Time_Partial_Week ||--o{ df_output_Sell_Out_FCST_Color_Condition : "cross join"
  df_in_Item_Master ||--o{ df_output_Sell_Out_FCST_Color_Condition : "COLOR_GREEN 대상"
  df_fn_Sales_Product_ASN_Item ||--o{ df_output_Sell_Out_FCST_Color_Condition : "VD/SHA leadtime"
  df_fn_Sales_Product_ASN_Group ||--o{ df_output_Sell_Out_FCST_Color_Condition : "base population"
```
# 3) 전체 Sequence Diagram
``` mermaid
sequenceDiagram
  autonumber
  participant Main
  participant IO as fn_process_in_df_mst
  participant S00 as step00_load_shipto_dimension
  participant S01 as step01_preprocess_rts_eos
  participant S02 as step02_convert_date_to_partial_week
  participant S31 as step03_1_prepare_asn_item
  participant S32 as step03_2_group_asn_to_ap2_item
  participant S04 as step04_build_rts_eos_week
  participant S05 as step05_apply_color_rts_eos
  participant S06 as step06_apply_green_for_wireless_bas
  participant S07 as step07_apply_ha_eop_yellow
  participant S81 as step08_1_apply_vd_leadtime
  participant S82 as step08_2_apply_set_leadtime
  participant S09 as step09_apply_gray_no_sellout
  participant S10 as step10_build_sell_in_output
  participant S11 as step11_extend_sell_out_output
  Main->>IO: load inputs into input_dataframes + type convert
  IO-->>Main: input_dataframes ready
  Main->>S00: df_in_Sales_Domain_Dimension
  S00-->>Main: df_fn_shipto_dim
  Main->>S01: df_in_MST_RTS_EOS
  S01-->>Main: df_fn_RTS_EOS (pre)
  Main->>S02: df_fn_RTS_EOS + CurrentPartialWeek + df_in_Time_Partial_Week + df_fn_shipto_dim
  S02-->>Main: df_fn_RTS_EOS (with PW + helper)
  Main->>S31: df_in_Sales_Product_ASN + df_fn_shipto_dim + df_in_Item_Master + df_in_Item_CLASS + df_in_Item_TAT
  S31-->>Main: df_fn_Sales_Product_ASN_Item
  Main->>S32: df_fn_Sales_Product_ASN_Item + df_fn_shipto_dim
  S32-->>Main: df_fn_Sales_Product_ASN_Group
  Main->>S04: df_fn_Sales_Product_ASN_Group + df_in_Time_Partial_Week + CurrentPartialWeek
  S04-->>Main: df_fn_ASN_Group_Week
  Main->>S05: df_fn_ASN_Group_Week + df_fn_RTS_EOS + CurrentPartialWeek
  S05-->>Main: df_fn_ASN_Group_Week (colors: GRAY/DBLUE/LBLUE/LRED/DRED)
  Main->>S06: df_fn_ASN_Group_Week + df_in_Item_Master + CurrentPartialWeek
  S06-->>Main: df_fn_ASN_Group_Week (COLOR_GREEN override)
  Main->>S07: df_fn_ASN_Group_Week + df_fn_RTS_EOS + df_fn_Sales_Product_ASN_Group + CurrentPartialWeek
  S07-->>Main: df_fn_ASN_Group_Week (COLOR_YELLOW override)
  Main->>S81: df_fn_ASN_Group_Week + df_fn_Sales_Product_ASN_Item + df_fn_Sales_Product_ASN_Group + CurrentPartialWeek
  S81-->>Main: df_fn_ASN_Group_Week (COLOR_DGRAY_RED)
  Main->>S82: df_fn_ASN_Group_Week + df_fn_Sales_Product_ASN_Item + df_fn_Sales_Product_ASN_Group + CurrentPartialWeek
  S82-->>Main: df_fn_ASN_Group_Week (COLOR_DGRAY_REDB)
  Main->>S09: df_fn_ASN_Group_Week + df_in_SELLOUTFCST_NOTEXIST + CurrentPartialWeek
  S09-->>Main: df_fn_ASN_Group_Week (COLOR_GRAY override)
  Main->>S10: df_fn_ASN_Group_Week + Version
  S10-->>Main: df_output_Sell_In_FCST_Color_Condition
  Main->>S11: df_fn_RTS_EOS + df_in_Time_Partial_Week + df_in_Item_Master + df_fn_Sales_Product_ASN_Item/Group + CurrentPartialWeek + Version
  S11-->>Main: df_output_Sell_Out_FCST_Color_Condition
```
## 4) Step별 상세 Sequence (Step 1 ~ Step 11)
## Step 01: RTS/EOS 전처리
``` mermaid
sequenceDiagram
  autonumber
  participant S01 as step01_preprocess_rts_eos
  participant IN as df_in_MST_RTS_EOS
  participant OUT as df_fn_RTS_EOS
  S01->>IN: read (VERSION, ITEM, SHIP_TO, RTS/EOS dates/status...)
  S01->>S01: drop VERSION, RTS_ISVALID
  S01->>S01: keep USE_COLS + cast category
  S01-->>OUT: df_fn_RTS_EOS (pre)
```
## Step 02: Step1의 Result에 Time을 Partial Week 으로 변환
``` mermaid
sequenceDiagram
  autonumber
  participant S02 as step02_convert_date_to_partial_week
  participant RTS as df_fn_RTS_EOS
  participant TIME as df_in_Time_Partial_Week
  participant DIM as df_fn_shipto_dim
  participant OUT as df_fn_RTS_EOS
  S02->>RTS: sanitize_date_string + to_partial_week_datetime
  S02->>S02: derive RTS_PARTIAL_WEEK / EOS_PARTIAL_WEEK
  S02->>S02: derive RTS_WEEK/EOS_WEEK + +/- week helpers
  S02->>TIME: max(TIME_PW) for helper bounds
  S02->>S02: derive MAX_RTS_CURRENTWEEK, MIN_EOSINI_MAXWEEK, MIN_EOS_MAXWEEK
  S02->>DIM: map LV2 rows -> LV3 fan-out (STD1->SHIP_TO child)
  S02-->>OUT: df_fn_RTS_EOS (final)
```
## Step 03-1 : df_in_Sales_Product_ASN 전처리
``` mermaid
sequenceDiagram
  autonumber
  participant S31 as step03_1_prepare_asn_item
  participant ASN as df_in_Sales_Product_ASN
  participant DIM as df_fn_shipto_dim
  participant MST as df_in_Item_Master
  participant CLS as df_in_Item_CLASS
  participant TAT as df_in_Item_TAT
  participant OUT as df_fn_Sales_Product_ASN_Item
  S31->>ASN: select (SHIP_TO, ITEM, LOC)
  S31->>DIM: merge SHIP_TO -> STD5 (Lv6)
  S31->>MST: merge ITEM -> GBM, PG
  S31->>CLS: filter CLASS=='X' then rename SHIP_TO->STD5
  S31->>S31: merge (ITEM, STD5, LOC) -> HA_EOP_FLAG(bool)
  S31->>TAT: merge (ITEM, LOC) -> TATTERM/TAT_SET (fillna 0, int32)
  S31-->>OUT: df_fn_Sales_Product_ASN_Item
```
## Step 03-2: AP2(Lv-2/3) × Item 단위 그룹핑. df_fn_Sales_Product_ASN_Group 생성
``` mermaid
sequenceDiagram
  autonumber
  participant S32 as step03_2_group_asn_to_ap2_item
  participant IN as df_fn_Sales_Product_ASN_Item
  participant DIM as df_fn_shipto_dim
  participant OUT as df_fn_Sales_Product_ASN_Group
  S32->>DIM: build index + STD2/STD1 arrays
  S32->>IN: map SHIP_TO -> parent_lv3(=STD2) (20250731 rule)
  S32->>S32: ultra_fast_groupby_numpy_general by (SHIP_TO, ITEM)
  S32->>S32: agg HA_EOP_FLAG=max, TATTERM=max, TAT_SET=max
  S32-->>OUT: df_fn_Sales_Product_ASN_Group
```
## Step 04: RTS/EOS (Lv3 × Item) → ( +Partial-Week ) Fan-out. 기본 Color = 19_GRAY
``` mermaid
sequenceDiagram
  autonumber
  participant S04 as step04_build_rts_eos_week
  participant GRP as df_fn_Sales_Product_ASN_Group
  participant TIME as df_in_Time_Partial_Week
  participant OUT as df_fn_ASN_Group_Week
  S04->>GRP: take core (ITEM, SHIP_TO)
  S04->>TIME: read TIME_PW vector
  S04->>S04: cross join (repeat/tile) -> (SHIP_TO, ITEM, TIME_PW)
  S04->>S04: CURRENT_ROW_WEEK = slice(TIME_PW,0,6)
  S04->>S04: init SIN_FCST_COLOR_COND = pd.NA (category + add color categories later)
  S04-->>OUT: df_fn_ASN_Group_Week
```
## Step 05: RTS / EOS Color 반영. 
### - 규칙 5-1 ~ 5-5 (WHITE / DARKBLUE / LIGHTBLUE / LIGHTRED / DARKRED)
### - 벡터라이즈 & 1-pass overwrite (19_GRAY → WHITE → … → DARKRED)
``` mermaid
sequenceDiagram
  autonumber
  participant S05 as step05_apply_color_rts_eos
  participant WK as df_fn_ASN_Group_Week
  participant RTS as df_fn_RTS_EOS
  participant OUT as df_fn_ASN_Group_Week
  S05->>RTS: select helper cols (RTS/EOS week variants, DEV)
  S05->>WK: merge on (ITEM, SHIP_TO)
  S05->>S05: week_to_int arrays
  S05->>S05: mask_GRAY (CW.. < RTS_INIT)
  S05->>S05: mask_DARKBLUE using max(RTS_INITIAL_WEEK, RTS_DEV_WEEK) .. RTS_WEEK_MINUST_1
  S05->>S05: mask_LIGHTBLUE RTS_WEEK .. RTS_WEEK_PLUS_3
  S05->>S05: if apply_eos_red: mask_LIGHTRED (EOS_WEEK_MINUS_3..EOS_WEEK), mask_DARKRED (>EOS_WEEK)
  S05-->>OUT: drop helper join cols -> df_fn_ASN_Group_Week
```
## Step 06: Wireless BAS 모델 – 당주 포함 8 주 구간 13_GREEN 적용
``` mermaid
sequenceDiagram
  autonumber
  participant S06 as step06_apply_green_for_wireless_bas
  participant WK as df_fn_ASN_Group_Week
  participant MST as df_in_Item_Master
  participant OUT as df_fn_ASN_Group_Week
  S06->>MST: filter PT=='BAS' and GBM=='MOBILE' -> bas_items
  S06->>WK: within [CW..CW+7] AND color != COLOR_GRAY -> set COLOR_GREEN
  S06-->>OUT: df_fn_ASN_Group_Week
```
## Step 07: HA-EOP Management 모델 12_YELLOW 반영
``` mermaid
sequenceDiagram
  autonumber
  participant S07 as step07_apply_ha_eop_yellow
  participant WK as df_fn_ASN_Group_Week
  participant RTS as df_fn_RTS_EOS
  participant GRP as df_fn_Sales_Product_ASN_Group
  participant OUT as df_fn_ASN_Group_Week
  S07->>GRP: targets where HA_EOP_FLAG==True (SHIP_TO, ITEM)
  S07->>WK: merge EOS_WEEK/EOS_WEEK_MINUS_4 from RTS on (ITEM, SHIP_TO)
  S07->>WK: merge targets flag on (SHIP_TO, ITEM)
  S07->>S07: 보호: COLOR_LIGHTRED/COLOR_DARKRED 영역 제외
  S07->>S07: Case1 EOS> CW -> [CW, EOS_MINUS_4) = COLOR_YELLOW
  S07->>S07: Case2 EOS 없음/지남 -> [CW, END] = COLOR_YELLOW
  S07-->>OUT: df_fn_ASN_Group_Week
```
## Step 08-1: VD / SHA  Lead-Time (18_DGRAY_RED) 갱신
``` mermaid
sequenceDiagram
  autonumber
  participant S81 as step08_1_apply_vd_leadtime
  participant WK as df_fn_ASN_Group_Week
  participant IT as df_fn_Sales_Product_ASN_Item
  participant GRP as df_fn_Sales_Product_ASN_Group
  participant OUT as df_fn_ASN_Group_Week
  S81->>IT: build ITEM->GBM map (drop_duplicates)
  S81->>GRP: filter TATTERM>0 and GBM in ('VD','SHA')
  S81->>WK: align (SHIP_TO, ITEM) -> tat_arr via MultiIndex reindex
  S81->>S81: compute end_week per unique TATTERM
  S81->>WK: if (CW..end) AND SIN_FCST_COLOR_COND is NA -> set COLOR_DGRAY_RED
  S81-->>OUT: df_fn_ASN_Group_Week
```
## Step 08-2: SET Lead-Time 구간 17_DGRAY_REDB 갱신
``` mermaid
sequenceDiagram
  autonumber
  participant S82 as step08_2_apply_set_leadtime
  participant WK as df_fn_ASN_Group_Week
  participant IT as df_fn_Sales_Product_ASN_Item
  participant GRP as df_fn_Sales_Product_ASN_Group
  participant OUT as df_fn_ASN_Group_Week
  S82->>IT: build ITEM->GBM map
  S82->>GRP: filter TAT_SET>0 and GBM in ('VD','SHA')
  S82->>WK: align (SHIP_TO, ITEM) -> tatset_arr
  S82->>S82: compute end_week per unique TAT_SET
  S82->>WK: promote COLOR_DGRAY_RED -> COLOR_DGRAY_REDB in (CW..end)
  S82-->>OUT: df_fn_ASN_Group_Week
```
## Step 09: MX  Sell-out Forecast  없는  모델  GRAY  업데이트
``` mermaid
sequenceDiagram
  autonumber
  participant S09 as step09_apply_gray_no_sellout
  participant WK as df_fn_ASN_Group_Week
  participant NS as df_in_SELLOUTFCST_NOTEXIST
  participant OUT as df_fn_ASN_Group_Week
  S09->>NS: flags where SOUT_FCST_NOT_EXISTS==True (STD2, ITEM)
  S09->>WK: merge flags on (SHIP_TO, ITEM) using STD2->SHIP_TO rename
  S09->>WK: if flagged and week>=CW -> set COLOR_GRAY
  S09-->>OUT: df_fn_ASN_Group_Week
```
## Step 10: df_output_Sell_In_FCST_Color_Condition
``` mermaid
sequenceDiagram
  autonumber
  participant S10 as step10_build_sell_in_output
  participant WK as df_fn_ASN_Group_Week
  participant OUT as df_output_Sell_In_FCST_Color_Condition
  S10->>WK: project (SHIP_TO, ITEM, TIME_PW, SIN_FCST_COLOR_COND)
  S10->>S10: rename SHIP_TO -> STD2
  S10->>S10: insert VERSION (single-category)
  S10-->>OUT: df_output_Sell_In_FCST_Color_Condition
```
## Step 11: df_output_Sell_Out_FCST_Color_Condition
``` mermaid
sequenceDiagram
  autonumber
  participant S11 as step11_extend_sell_out_output
  participant RTS as df_fn_RTS_EOS
  participant TIME as df_in_Time_Partial_Week
  participant MST as df_in_Item_Master
  participant IT as df_fn_Sales_Product_ASN_Item
  participant GRP as df_fn_Sales_Product_ASN_Group
  participant OUT as df_output_Sell_Out_FCST_Color_Condition
  participant S04 as step04_build_rts_eos_week
  participant S05 as step05_apply_color_rts_eos
  participant S06 as step06_apply_green_for_wireless_bas
  participant S81 as step08_1_apply_vd_leadtime
  S11->>S04: build week grid from GRP x TIME
  S04-->>S11: df_asn_week
  S11->>S05: apply RTS colors (apply_eos_red=False)
  S05-->>S11: df_asn_week
  S11->>S06: wireless BAS GREEN
  S06-->>S11: df_asn_week
  S11->>S81: VD/SHA leadtime (DGRAY_RED)
  S81-->>S11: df_asn_week
  S11->>S11: project (SHIP_TO, ITEM, TIME_PW, SIN_FCST_COLOR_COND)
  S11->>S11: rename SHIP_TO->STD2, SIN_FCST_COLOR_COND->SOUT_FCST_COLOR_COND
  S11->>S11: insert VERSION (single-category)
  S11-->>OUT: df_output_Sell_Out_FCST_Color_Condition
```