# PYForecastSellInAndOutColorDeltaNumber

# 1) 전체 ERD
``` mermaid
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
  df_in_MST_RTS_EOS_Delta {
    category VERSION
    category ITEM
    category SHIP_TO
    category RTS_STATUS_Delta
    category RTS_INIT_DATE_Delta
    category RTS_DEV_DATE_Delta
    category RTS_COM_DATE_Delta
    category EOS_STATUS_Delta
    category EOS_INIT_DATE_Delta
    category EOS_CHG_DATE_Delta
    category EOS_COM_DATE_Delta
    category RTS_ISVALID_Delta
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
    category RTS_DEV_WEEK
    category RTS_WEEK_MINUST_1
    category RTS_WEEK_PLUS_3
    category EOS_WEEK_MINUS_1
    category EOS_WEEK_MINUS_3
    category EOS_WEEK_MINUS_4
    category MAX_RTS_CURRENTWEEK
    category MIN_EOSINI_MAXWEEK
    category MIN_EOS_MAXWEEK
  }
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
    bool ESTOREACCOUNT
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

  df_in_MST_RTS_EOS_Delta ||--o{ df_fn_RTS_EOS : "ITEM+SHIP_TO (base)"
  df_in_MST_RTS_EOS ||--o{ df_fn_RTS_EOS : "ITEM+SHIP_TO (fill when Delta missing)"
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
  %% 2) RTS/EOS Inputs
  %% =========================
  df_in_MST_RTS_EOS_Delta {
    category VERSION PK  "(논리 PK: VERSION, ITEM, SHIP_TO)"
    category ITEM    PK
    category SHIP_TO PK
  }
  df_in_MST_RTS_EOS {
    category VERSION PK  "(논리 PK: VERSION, ITEM, SHIP_TO)"
    category ITEM    PK
    category SHIP_TO PK
  }

  %% =========================
  %% 3) RTS/EOS (after Step02)
  %% =========================
  df_fn_RTS_EOS {
    category SHIP_TO PK        "(ITEM, SHIP_TO) 가 논리 PK"
    category ITEM    PK
    category RTS_WEEK
    category EOS_WEEK
    category RTS_INITIAL_WEEK
    category RTS_DEV_WEEK
    category EOS_INITIAL_WEEK
    category RTS_WEEK_MINUST_1
    category RTS_WEEK_PLUS_3
    category EOS_WEEK_MINUS_1
    category EOS_WEEK_MINUS_3
    category EOS_WEEK_MINUS_4
    category MAX_RTS_CURRENTWEEK
    category MIN_EOS_MAXWEEK
    category RTS_DEV_DATE
  }
  df_output_RTS_EOS {
    category VERSION PK        "(논리 PK: VERSION, ITEM, SHIP_TO)"
    category ITEM    PK
    category SHIP_TO PK
  }

  %% =========================
  %% 4) INPUTS (ASN / ITEM)
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
    category STD5 PK           "(ITEM, STD5, LOC, CLASS) 중, CLASS=='X' 후 (ITEM,STD5,LOC) 사용"
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
  %% 5) Step03-1 output
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
  %% 6) Step03-2 output
  %% =========================
  df_fn_Sales_Product_ASN_Group {
    category SHIP_TO PK        "논리 PK: (SHIP_TO, ITEM) ; SHIP_TO는 parent STD2(=Lv3)로 치환됨"
    category ITEM    PK
    bool     HA_EOP_FLAG
    int      TATTERM
    int      TAT_SET
    bool     ESTOREACCOUNT
  }

  %% =========================
  %% 7) Week fan-out (Step04+)
  %% =========================
  df_fn_ASN_Group_Week {
    category SHIP_TO PK        "논리 PK: (SHIP_TO, ITEM, TIME_PW)"
    category ITEM    PK
    category TIME_PW PK
    category CURRENT_ROW_WEEK
    category SIN_FCST_COLOR_COND
  }

  %% =========================
  %% 8) No-sellout flag
  %% =========================
  df_in_SELLOUTFCST_NOTEXIST {
    category STD2 PK           "논리 PK: (STD2, ITEM)"
    category ITEM PK
    bool     SOUT_FCST_NOT_EXISTS
  }

  %% =========================
  %% 9) Outputs
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
  df_fn_shipto_dim ||--o{ df_fn_Sales_Product_ASN_Item : "FK: ASN_Item.SHIP_TO -> shipto_dim.SHIP_TO (get STD5)"
  df_in_Item_Master ||--o{ df_fn_Sales_Product_ASN_Item : "FK: ASN_Item.ITEM -> Item_Master.ITEM (GBM,PG,PT)"
  df_in_Item_CLASS ||--o{ df_fn_Sales_Product_ASN_Item : "FK (existence): (ITEM,STD5,LOC) where CLASS=='X'"
  df_in_Item_TAT ||--o{ df_fn_Sales_Product_ASN_Item : "FK: (ITEM,LOC) -> (ITEM,LOC) (TATTERM,TAT_SET)"
  df_in_Sales_Product_ASN ||--o{ df_fn_Sales_Product_ASN_Item : "PK propagate: (SHIP_TO,ITEM,LOC)"

  df_fn_Sales_Product_ASN_Item ||--o{ df_fn_Sales_Product_ASN_Group : "GroupBy: (parent(SHIP_TO=STD2), ITEM) ; max(HA_EOP/TAT)"
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "FK: (SHIP_TO,ITEM) -> (SHIP_TO,ITEM)"
  df_in_Time_Partial_Week      ||--o{ df_fn_ASN_Group_Week : "FK: TIME_PW -> TIME_PW (cross join)"

  df_fn_RTS_EOS ||--o{ df_fn_ASN_Group_Week : "FK: (ITEM,SHIP_TO) -> (ITEM,SHIP_TO) (color rules)"
  df_in_SELLOUTFCST_NOTEXIST ||--o{ df_fn_ASN_Group_Week : "FK: (STD2,ITEM) -> (SHIP_TO,ITEM)"

  df_fn_ASN_Group_Week ||--o{ df_output_Sell_In_FCST_Color_Condition  : "Projection: SHIP_TO->STD2 + VERSION"
  df_fn_ASN_Group_Week ||--o{ df_output_Sell_Out_FCST_Color_Condition : "Recompute subset + rename SIN->SOUT + VERSION"
```

## 2) Step별 서브 ERD

## Step 00: Ship-To 차원 LUT 구축
``` mermaid
erDiagram
  df_in_Sales_Domain_Dimension ||--o{ df_fn_shipto_dim : "SHIP_TO -> LV_CODE/STD1..6"
```

## Step 01: RTS/EOS 전처리 (Delta + Full 보완)
``` mermaid
erDiagram
  df_in_MST_RTS_EOS_Delta ||--o{ df_fn_RTS_EOS : "Delta base (rename *Delta -> base)"
  df_in_MST_RTS_EOS ||--o{ df_fn_RTS_EOS : "fill when Delta fields missing"
  df_in_MST_RTS_EOS_Delta ||--o{ df_output_RTS_EOS : "Output(Version 포함) for spec"
```

## Step 02: Step1의 Result에 Time을 Partial Week 으로 변환
``` mermaid
erDiagram
  df_in_Time_Partial_Week ||--o{ df_fn_RTS_EOS : "MAX week for helper calc"
  df_fn_shipto_dim ||--o{ df_fn_RTS_EOS : "LV2 -> LV3 fan-out (if rule enabled)"
```

## Step 03-1 : df_in_Sales_Product_ASN 전처리 (Delta 범위 축소)
``` mermaid
erDiagram
  df_in_Sales_Product_ASN ||--o{ df_fn_Sales_Product_ASN_Item : "SHIP_TO+ITEM+LOC"
  df_fn_shipto_dim ||--o{ df_fn_Sales_Product_ASN_Item : "SHIP_TO -> STD5"
  df_in_Item_Master ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM -> GBM/PG"
  df_in_Item_CLASS ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM+LOC+STD5 -> HA_EOP_FLAG"
  df_in_Item_TAT ||--o{ df_fn_Sales_Product_ASN_Item : "ITEM+LOC -> TATTERM/TAT_SET"
  df_fn_RTS_EOS ||--o{ df_fn_Sales_Product_ASN_Item : "Delta filter: keep only Items in RTS/EOS scope"
```

## Step 03-2: AP2(Lv-2/3) × Item 단위 그룹핑. df_fn_Sales_Product_ASN_Group 생성 (Delta 범위 축소)
``` mermaid
erDiagram
  df_fn_Sales_Product_ASN_Item ||--o{ df_fn_Sales_Product_ASN_Group : "group (SHIP_TO->parent STD2), ITEM"
  df_fn_shipto_dim ||--o{ df_fn_Sales_Product_ASN_Group : "SHIP_TO -> STD2 mapping"
  df_fn_RTS_EOS ||--o{ df_fn_Sales_Product_ASN_Group : "Delta filter: keep only (STD2,ITEM) pairs in RTS/EOS scope"
```

## Step 04: RTS/EOS (Std2 × Item) → ( +Partial-Week ) Fan-out. 기본 Color = NA
``` mermaid
erDiagram
  df_in_Time_Partial_Week ||--o{ df_fn_ASN_Group_Week : "cross join by TIME_PW"
  df_fn_Sales_Product_ASN_Group ||--o{ df_fn_ASN_Group_Week : "SHIP_TO+ITEM"
```

## Step 05: RTS / EOS Color 반영
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

## Step 11: df_output_Sell_Out_FCST_Color_Condition (non-eStore 재계산 경로)
``` mermaid
erDiagram
  df_fn_RTS_EOS ||--o{ df_output_Sell_Out_FCST_Color_Condition : "recompute (apply_eos_red=False)"
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

  Main->>IO: load inputs into input_dataframes + type convert
  IO-->>Main: input_dataframes ready

  Main->>S00: df_in_Sales_Domain_Dimension
  S00-->>Main: df_fn_shipto_dim

  Main->>S01: df_in_MST_RTS_EOS_Delta + df_in_MST_RTS_EOS + Version
  S01-->>Main: df_fn_RTS_EOS (pre) + df_output_RTS_EOS

  Main->>S02: df_fn_RTS_EOS + CurrentPartialWeek
  S02-->>Main: df_fn_RTS_EOS (with PW + helper)

  Main->>S31: df_in_Sales_Product_ASN + df_in_Sales_Domain_Dimension + df_in_Item_Master + df_in_Item_CLASS + df_in_Item_TAT + df_fn_RTS_EOS
  S31-->>Main: df_fn_Sales_Product_ASN_Item

  Main->>S32: df_fn_Sales_Product_ASN_Item + df_fn_shipto_dim + df_fn_RTS_EOS
  S32-->>Main: df_fn_Sales_Product_ASN_Group

  Main->>S04: df_fn_Sales_Product_ASN_Group + df_in_Time_Partial_Week + CurrentPartialWeek
  S04-->>Main: df_fn_ASN_Group_Week

  Main->>S05: df_fn_ASN_Group_Week + df_fn_RTS_EOS + CurrentPartialWeek
  S05-->>Main: df_fn_ASN_Group_Week (colors)

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

  Main->>S11: df_fn_RTS_EOS + df_in_Time_Partial_Week + df_fn_shipto_dim + df_in_Item_Master + df_in_Item_CLASS + df_in_Item_TAT + df_in_Sales_Product_ASN + CurrentPartialWeek + Version
  S11-->>Main: df_output_Sell_Out_FCST_Color_Condition
```

## 4) Step별 상세 Sequence (Step 1 ~ Step 11)

## Step 01: RTS/EOS 전처리 (Delta + Full 보완)
``` mermaid
sequenceDiagram
  autonumber
  participant S01 as step01_preprocess_rts_eos_delta
  participant DEL as df_in_MST_RTS_EOS_Delta
  participant FULL as df_in_MST_RTS_EOS
  participant OUT as df_fn_RTS_EOS
  participant OUT2 as df_output_RTS_EOS

  S01->>DEL: read Delta columns (*_Delta)
  S01->>S01: rename "*_Delta" -> base names
  S01->>S01: drop VERSION / RTS_ISVALID for df_fn_RTS_EOS

  S01->>FULL: read Full columns (fallback source)
  S01->>S01: fill missing RTS/EOS fields in Delta rows from Full by (ITEM, SHIP_TO)

  S01->>S01: create df_output_RTS_EOS with VERSION + filled columns
  S01-->>OUT: df_fn_RTS_EOS (pre)
  S01-->>OUT2: df_output_RTS_EOS
```

## Step 02: Step1의 Result에 Time을 Partial Week 으로 변환
``` mermaid
sequenceDiagram
  autonumber
  participant S02 as step02_convert_date_to_partial_week
  participant RTS as df_fn_RTS_EOS
  participant TIME as df_in_Time_Partial_Week
  participant OUT as df_fn_RTS_EOS

  S02->>RTS: sanitize_date_string + to_partial_week_datetime
  S02->>S02: derive RTS_PARTIAL_WEEK / EOS_PARTIAL_WEEK
  S02->>S02: derive RTS_WEEK/EOS_WEEK + +/- week helpers
  S02->>TIME: max(TIME_PW) for helper bounds
  S02->>S02: derive MAX_RTS_CURRENTWEEK, MIN_EOSINI_MAXWEEK, MIN_EOS_MAXWEEK
  S02-->>OUT: df_fn_RTS_EOS (final)
```

## Step 03-1 : df_in_Sales_Product_ASN 전처리 (Delta 범위 축소)
``` mermaid
sequenceDiagram
  autonumber
  participant S31 as step03_1_prepare_asn_item_delta
  participant ASN as df_in_Sales_Product_ASN
  participant DIM as df_in_Sales_Domain_Dimension
  participant MST as df_in_Item_Master
  participant CLS as df_in_Item_CLASS
  participant TAT as df_in_Item_TAT
  participant RTS as df_fn_RTS_EOS
  participant OUT as df_fn_Sales_Product_ASN_Item

  S31->>ASN: select (SHIP_TO, ITEM, LOC)
  S31->>S31: filter Items using RTS scope (ITEM in df_fn_RTS_EOS)
  S31->>DIM: merge SHIP_TO -> STD5 (Lv6)
  S31->>MST: merge ITEM -> GBM, PG

  S31->>CLS: filter CLASS=='X' then rename SHIP_TO->STD5
  S31->>S31: merge (ITEM, STD5, LOC) -> HA_EOP_FLAG(bool)

  S31->>TAT: merge (ITEM, LOC) -> TATTERM/TAT_SET (fillna 0, int)
  S31-->>OUT: df_fn_Sales_Product_ASN_Item
```

## Step 03-2: AP2(Lv-2/3) × Item 단위 그룹핑. df_fn_Sales_Product_ASN_Group 생성 (Delta 범위 축소)
``` mermaid
sequenceDiagram
  autonumber
  participant S32 as step03_2_group_asn_to_ap2_item_delta
  participant IN as df_fn_Sales_Product_ASN_Item
  participant DIM as df_fn_shipto_dim
  participant RTS as df_fn_RTS_EOS
  participant OUT as df_fn_Sales_Product_ASN_Group

  S32->>DIM: build mapping SHIP_TO -> parent STD2
  S32->>IN: map SHIP_TO to STD2 (parent)
  S32->>S32: groupby (SHIP_TO=STD2, ITEM) (max HA_EOP/TAT)
  S32->>S32: filter pairs using RTS scope (keep only (STD2,ITEM) in df_fn_RTS_EOS)
  S32-->>OUT: df_fn_Sales_Product_ASN_Group
```

## Step 04: RTS/EOS (Std2 × Item) → ( +Partial-Week ) Fan-out. 기본 Color = NA
``` mermaid
sequenceDiagram
  autonumber
  participant S04 as step04_build_rts_eos_week
  participant GRP as df_fn_Sales_Product_ASN_Group
  participant TIME as df_in_Time_Partial_Week
  participant OUT as df_fn_ASN_Group_Week

  S04->>GRP: take core (ITEM, SHIP_TO)
  S04->>TIME: read TIME_PW vector
  S04->>S04: cross join -> (SHIP_TO, ITEM, TIME_PW)
  S04->>S04: CURRENT_ROW_WEEK = TIME_PW[:6]
  S04->>S04: init SIN_FCST_COLOR_COND = pd.NA
  S04-->>OUT: df_fn_ASN_Group_Week
```

## Step 05: RTS / EOS Color 반영
``` mermaid
sequenceDiagram
  autonumber
  participant S05 as step05_apply_color_rts_eos
  participant WK as df_fn_ASN_Group_Week
  participant RTS as df_fn_RTS_EOS
  participant OUT as df_fn_ASN_Group_Week

  S05->>WK: merge RTS helpers on (ITEM, SHIP_TO)
  S05->>S05: wk_int arrays
  S05->>S05: mask_GRAY (CW.. < RTS_INITIAL_WEEK)
  S05->>S05: mask_DARKBLUE using max(RTS_INITIAL_WEEK, RTS_DEV_WEEK) .. RTS_WEEK_MINUST_1
  S05->>S05: mask_LIGHTBLUE RTS_WEEK .. RTS_WEEK_PLUS_3
  S05->>S05: (apply_eos_red=True) mask_LIGHTRED, mask_DARKRED
  S05-->>OUT: drop helper cols -> df_fn_ASN_Group_Week
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
  S06->>WK: if ITEM in bas_items and week in [CW..CW+7] -> set COLOR_GREEN
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
  S07->>WK: merge EOS helpers from RTS on (ITEM, SHIP_TO)
  S07->>WK: merge targets flag on (SHIP_TO, ITEM)
  S07->>S07: 보호: LIGHTRED/DARKRED 제외
  S07->>S07: EOS 기준으로 CW 이후를 COLOR_YELLOW 처리
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

  S81->>IT: build ITEM->GBM map
  S81->>GRP: filter TATTERM>0 and GBM in ('VD','SHA')
  S81->>WK: align (SHIP_TO, ITEM) -> tat_arr
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

## Step 11: df_output_Sell_Out_FCST_Color_Condition (non-eStore 재계산 경로)
``` mermaid
sequenceDiagram
  autonumber
  participant S11 as step11_extend_sell_out_output
  participant RTS as df_fn_RTS_EOS
  participant TIME as df_in_Time_Partial_Week
  participant DIM as df_fn_shipto_dim
  participant MST as df_in_Item_Master
  participant CLS as df_in_Item_CLASS
  participant TAT as df_in_Item_TAT
  participant ASN as df_in_Sales_Product_ASN
  participant OUT as df_output_Sell_Out_FCST_Color_Condition
  participant S31 as step03_1_prepare_asn_item_delta
  participant S32 as step03_2_group_asn_to_ap2_item_delta
  participant S04 as step04_build_rts_eos_week
  participant S05 as step05_apply_color_rts_eos
  participant S06 as step06_apply_green_for_wireless_bas
  participant S81 as step08_1_apply_vd_leadtime

  S11->>S31: rebuild ASN_Item (same logic inputs)
  S31-->>S11: df_asn_item
  S11->>S32: rebuild ASN_Group (same logic)
  S32-->>S11: df_asn_group
  S11->>S04: build week grid from asn_group x TIME
  S04-->>S11: df_asn_week
  S11->>S05: apply RTS colors (apply_eos_red=False)
  S05-->>S11: df_asn_week
  S11->>S06: wireless BAS GREEN
  S06-->>S11: df_asn_week
  S11->>S81: VD/SHA leadtime (DGRAY_RED)
  S81-->>S11: df_asn_week
  S11->>S11: project + rename SHIP_TO->STD2, SIN->SOUT
  S11->>S11: insert VERSION (single-category)
  S11-->>OUT: df_output_Sell_Out_FCST_Color_Condition
```