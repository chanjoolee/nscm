# PYSalesProductASNDeltaeStore (Diagram)

- 본 문서는 `nscm/src/PYSalesProductASNDeltaeStore.py` 소스를 기반으로 작성한다.
- 표기 규칙
  - `df_in_*`  : Input DataFrame (o9에서 입력)
  - `df_step*` : Step 처리 중간 DataFrame (Python 내부)
  - `df_output_*` / `Output_*` : Output DataFrame (o9로 반환)
- 주의
  - 개발요청서에는 `Step 8) E-store Output 정리`가 별도 단계로 존재하나, **현행 소스에서는 Step 7 결과 DF에 `Version` 주입 및 dtype 정리까지 포함**되어 별도 Step8 함수는 존재하지 않는다.

---

# 0) 전체 개요

## 0-1. 목적

- eStore Forecast UI에서 사용자가 버튼으로 확정한 Sales(ShipTo)×Item(×Location) 조합에 대해
  - Dummy 기반으로 생성 대상(Association)을 선정하고
  - Assortment를 생성하고
  - FCST(0값) 생성용 스켈레톤을 만들고
  - Estimated Price(Local) 및 Color를 생성하고
  - Split Ratio를 생성하고
  - eStore Ratio(7-1~7-8) 데이터를 규칙 기반으로 생성한다.

## 0-2. 핵심 파라미터

- `Version`: 출력 Version.[Version Name]
- `salesItemLocation`: `ShipTo:Item` 또는 `ShipTo:Item:Location` 토큰을 `^` 로 연결
- `measureLv`: `ap1|ap2|gc|local`

---

# 1) 전체 ERD

``` mermaid
%%{init: {'erDiagram':{'useMaxWidth':false}}}%%
erDiagram
  %% =========================
  %% INPUTS
  %% =========================
  df_in_sin_fcst_dummy {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_DUMMY_AP1
    float SIN_DUMMY_AP2
    float SIN_DUMMY_GC
    float SIN_DUMMY_LOCAL
  }
  df_in_sout_fcst_dummy {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_DUMMY_AP1
    float SOUT_DUMMY_AP2
    float SOUT_DUMMY_GC
    float SOUT_DUMMY_LOCAL
  }
  df_in_Sales_Domain_Dimension {
    category SHIP_TO
    category STD1
    category STD2
    category STD3
    category STD4
    category STD5
    category STD6
  }
  df_in_Sales_Domain_Estore {
    category SHIP_TO
  }
  df_in_Time_pw {
    category PARTIAL_WEEK
  }
  df_in_Time_w {
    category WEEK
  }
  df_in_Item_Master {
    category ITEM
    category ITEM_STD1
    category ITEM_STD2
    category ITEM_STD3
    category ITEM_STD4
  }

  %% Estimated Price inputs
  df_in_Estimated_Price {
    category SHIP_TO
    category ITEM
    category PARTIAL_WEEK
    float EST_PRICE_MOD_LOCAL
    float EST_PRICE_LOCAL
  }
  df_in_Action_Plan_Price {
    category SHIP_TO
    category ITEM
    category MONTH
    float AP_PRICE_USD
  }
  df_in_Exchange_Rate_Local {
    category STD3
    category PARTIAL_WEEK
    float EXRATE_LOCAL
  }
  df_in_Estimated_Price_Item_Std4_Local {
    category ITEM_STD4
    category SHIP_TO
    category PARTIAL_WEEK
    float EP_STD4_LOCAL
  }
  df_in_Estimated_Price_Item_Std3_Local {
    category ITEM_STD3
    category SHIP_TO
    category PARTIAL_WEEK
    float EP_STD3_LOCAL
  }
  df_in_Estimated_Price_Item_Std2_Local {
    category ITEM_STD2
    category SHIP_TO
    category PARTIAL_WEEK
    float EP_STD2_LOCAL
  }
  df_in_Estimated_Price_Sales_Std2_Item_Std4_Local {
    category STD2
    category ITEM_STD4
    category PARTIAL_WEEK
    float EP_SALES_STD4_LOCAL
  }
  df_in_Estimated_Price_Sales_Std2_Item_Std3_Local {
    category STD2
    category ITEM_STD3
    category PARTIAL_WEEK
    float EP_SALES_STD3_LOCAL
  }
  df_in_Estimated_Price_Sales_Std2_Item_Std2_Local {
    category STD2
    category ITEM_STD2
    category PARTIAL_WEEK
    float EP_SALES_STD2_LOCAL
  }

  %% Split Ratio inputs
  df_in_Sell_In_FCST_GI_Split_Ratio_AP1 {
    category STD2
    category ITEM_STD1
    category PARTIAL_WEEK
    float SIN_SR_AP1
  }
  df_in_Sell_In_FCST_GI_Split_Ratio_AP2 {
    category STD2
    category ITEM_STD1
    category PARTIAL_WEEK
    float SIN_SR_AP2
  }
  df_in_Sell_In_FCST_GI_Split_Ratio_GC {
    category STD2
    category ITEM_STD1
    category PARTIAL_WEEK
    float SIN_SR_GC
  }
  df_in_Sell_In_FCST_GI_Split_Ratio_Local {
    category STD2
    category ITEM_STD1
    category PARTIAL_WEEK
    float SIN_SR_LOCAL
  }
  df_in_Sell_Out_FCST_Split_Ratio_AP1 {
    category STD2
    category ITEM_STD1
    category PARTIAL_WEEK
    float SOUT_SR_AP1
  }
  df_in_Sell_Out_FCST_Split_Ratio_AP2 {
    category STD2
    category ITEM_STD1
    category PARTIAL_WEEK
    float SOUT_SR_AP2
  }
  df_in_Sell_Out_FCST_Split_Ratio_GC {
    category STD2
    category ITEM_STD1
    category PARTIAL_WEEK
    float SOUT_SR_GC
  }
  df_in_Sell_Out_FCST_Split_Ratio_Local {
    category STD2
    category ITEM_STD1
    category PARTIAL_WEEK
    float SOUT_SR_LOCAL
  }

  %% eStore Ratio inputs (ShipTo+ItemStd1+Loc+Week)
  df_in_Sell_In_User_GI_Ratio {
    category SHIP_TO
    category ITEM_STD1
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_Issue_GI_Ratio {
    category SHIP_TO
    category ITEM_STD1
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_Final_GI_Ratio {
    category SHIP_TO
    category ITEM_STD1
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_BestFit_GI_Ratio {
    category SHIP_TO
    category ITEM_STD1
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_User_Item_GI_Ratio {
    category SHIP_TO
    category ITEM_STD1
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_Issue_Item_GI_Ratio {
    category SHIP_TO
    category ITEM_STD1
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_User_Modify_GI_Ratio {
    category SHIP_TO
    category ITEM_STD1
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_Issue_Modify_GI_Ratio {
    category SHIP_TO
    category ITEM_STD1
    category LOCATION
    category WEEK
  }

  %% eStore Ratio inputs (ShipTo+Item+Loc+Week) - 신규 판단용
  df_in_Sell_In_User_GI_Ratio_Item {
    category SHIP_TO
    category ITEM
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_Issue_GI_Ratio_Item {
    category SHIP_TO
    category ITEM
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_Final_GI_Ratio_Item {
    category SHIP_TO
    category ITEM
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_BestFit_GI_Ratio_Item {
    category SHIP_TO
    category ITEM
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_User_Item_GI_Ratio_Item {
    category SHIP_TO
    category ITEM
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_Issue_Item_GI_Ratio_Item {
    category SHIP_TO
    category ITEM
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_User_Modify_GI_Ratio_Item {
    category SHIP_TO
    category ITEM
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_Issue_Modify_GI_Ratio_Item {
    category SHIP_TO
    category ITEM
    category LOCATION
    category WEEK
  }

  %% =========================
  %% STEP OUTPUTS / INTERMEDIATES
  %% =========================
  df_step01_00_sales_pairs {
    category SHIP_TO
    category ITEM
    category LOCATION
  }
  df_step01_01_sin_pick {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_DUMMY_TARGET
  }
  df_step01_03_sout_pick {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_DUMMY_TARGET
  }

  Output_SIn_Dummy {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_DUMMY_TARGET
  }
  Output_SOut_Dummy {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_DUMMY_TARGET
  }

  Output_SIn_Assortment {
    category SHIP_TO
    category ITEM
    category LOCATION
    float SIN_ASSORT_TARGET
  }
  Output_SOut_Assortment {
    category SHIP_TO
    category ITEM
    category LOCATION
    float SOUT_ASSORT_TARGET
  }

  df_output_Sell_In_FCST_GI_AP1 {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_GI_AP1
    float SIN_BL_AP1
    float SIN_NEW_MODEL
  }
  df_output_Sell_In_FCST_GI_AP2 {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_GI_AP2
    float SIN_BL_AP2
  }
  df_output_Sell_In_FCST_GI_GC {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_GI_GC
    float SIN_BL_GC
  }
  df_output_Sell_In_FCST_GI_Local {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_GI_LOCAL
    float SIN_BL_LOCAL
  }

  df_output_Sell_Out_FCST_AP1 {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_AP1
  }
  df_output_Sell_Out_FCST_AP2 {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_AP2
  }
  df_output_Sell_Out_FCST_GC {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_GC
  }
  df_output_Sell_Out_FCST_Local {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_LOCAL
  }

  df_output_Estimated_Price_Local {
    category VERSION
    category SHIP_TO
    category ITEM
    category PARTIAL_WEEK
    float EST_PRICE_LOCAL
    int   EST_PRICE_COLOR
  }

  df_output_Sell_In_FCST_GI_Split_Ratio_AP1 {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_SR_AP1
  }
  df_output_Sell_In_FCST_GI_Split_Ratio_AP2 {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_SR_AP2
  }
  df_output_Sell_In_FCST_GI_Split_Ratio_GC {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_SR_GC
  }
  df_output_Sell_In_FCST_GI_Split_Ratio_Local {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_SR_LOCAL
  }

  df_output_Sell_Out_FCST_Split_Ratio_AP1 {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_SR_AP1
  }
  df_output_Sell_Out_FCST_Split_Ratio_AP2 {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_SR_AP2
  }
  df_output_Sell_Out_FCST_Split_Ratio_GC {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_SR_GC
  }
  df_output_Sell_Out_FCST_Split_Ratio_Local {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_SR_LOCAL
  }

  %% eStore Ratio outputs (7-1~7-8): 공통 형태
  df_output_Sell_In_User_GI_Ratio {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category WEEK
    float USER_GI_LT
    float USER_GI_W7
    float USER_GI_W6
    float USER_GI_W5
    float USER_GI_W4
    float USER_GI_W3
    float USER_GI_W2
    float USER_GI_W1
    float USER_GI_W0
  }

  %% =========================
  %% RELATIONSHIPS (logic)
  %% =========================
  df_in_Sales_Domain_Dimension ||--o{ df_step01_00_sales_pairs : "ShipTo 유효성 필터"
  df_in_Sales_Domain_Estore    ||--o{ df_step01_00_sales_pairs : "eStore ShipTo 세트 기반 필터"

  df_step01_00_sales_pairs ||--o{ df_step01_01_sin_pick : "pairs semi-join"
  df_in_sin_fcst_dummy     ||--o{ df_step01_01_sin_pick : "measureLv 기반 dummy 컬럼 선택"

  df_step01_00_sales_pairs ||--o{ df_step01_03_sout_pick : "pairs semi-join"
  df_in_sout_fcst_dummy    ||--o{ df_step01_03_sout_pick : "measureLv 기반 dummy 컬럼 선택"

  df_step01_01_sin_pick ||--o{ Output_SIn_Dummy : "Step1-2: VERSION 주입 + dummy=NaN"
  df_step01_03_sout_pick ||--o{ Output_SOut_Dummy : "Step1-4: VERSION 주입 + dummy=NaN"

  df_step01_01_sin_pick ||--o{ Output_SIn_Assortment : "Step2-1: PW 제거 + max(1)"
  df_step01_03_sout_pick ||--o{ Output_SOut_Assortment : "Step2-2: PW 제거 + max(1)"

  df_in_Time_pw ||--o{ df_output_Sell_In_FCST_GI_AP1 : "Step3: (ShipTo,Item,Loc)×PW expand"
  df_in_Time_pw ||--o{ df_output_Sell_Out_FCST_AP1   : "Step3: (ShipTo,Item,Loc)×PW expand"

  Output_SIn_Assortment ||--o{ df_output_Estimated_Price_Local : "Step4: 대상 ShipTo*Item 선정(교집합 기반)"
  Output_SOut_Assortment ||--o{ df_output_Estimated_Price_Local : "Step4: 대상 ShipTo*Item 선정(교집합 기반)"

  df_in_Estimated_Price ||--o{ df_output_Estimated_Price_Local : "Step4-1: 우선순위 1~2"
  df_in_Estimated_Price_Item_Std4_Local ||--o{ df_output_Estimated_Price_Local : "Step4-1: 우선순위 3"
  df_in_Estimated_Price_Item_Std3_Local ||--o{ df_output_Estimated_Price_Local : "Step4-1: 우선순위 4"
  df_in_Estimated_Price_Item_Std2_Local ||--o{ df_output_Estimated_Price_Local : "Step4-1: 우선순위 5"
  df_in_Estimated_Price_Sales_Std2_Item_Std4_Local ||--o{ df_output_Estimated_Price_Local : "Step4-1: 우선순위 6"
  df_in_Estimated_Price_Sales_Std2_Item_Std3_Local ||--o{ df_output_Estimated_Price_Local : "Step4-1: 우선순위 7"
  df_in_Estimated_Price_Sales_Std2_Item_Std2_Local ||--o{ df_output_Estimated_Price_Local : "Step4-1: 우선순위 8"
  df_in_Action_Plan_Price ||--o{ df_output_Estimated_Price_Local : "Step4-1: 우선순위 9(AP*EXRATE)"
  df_in_Exchange_Rate_Local ||--o{ df_output_Estimated_Price_Local : "Step4-1: 우선순위 9(AP*EXRATE)"

  df_in_Sales_Domain_Dimension ||--o{ df_output_Sell_In_FCST_GI_Split_Ratio_AP1 : "Step5: ShipTo->Std2"
  df_in_Item_Master ||--o{ df_output_Sell_In_FCST_GI_Split_Ratio_AP1 : "Step5: Item->ItemStd1"
  df_in_Sell_In_FCST_GI_Split_Ratio_AP1 ||--o{ df_output_Sell_In_FCST_GI_Split_Ratio_AP1 : "Step5-1 join"

  df_in_Time_w ||--o{ df_output_Sell_In_User_GI_Ratio : "Step7: targets×week grid"
  df_in_Item_Master ||--o{ df_output_Sell_In_User_GI_Ratio : "Step7: Item->ItemStd1(PG)"
  df_in_Sell_In_User_GI_Ratio_Item ||--o{ df_output_Sell_In_User_GI_Ratio : "Step7: 신규/기존 판정"
  df_in_Sell_In_User_GI_Ratio ||--o{ df_output_Sell_In_User_GI_Ratio : "Step7: 신규 Item이면 ShipTo+PG 평균 사용"
```

---

# 1) 전체 ERD PK

``` mermaid
erDiagram
  df_step01_00_sales_pairs {
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
  }

  df_in_sin_fcst_dummy {
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
    category PARTIAL_WEEK PK
  }
  df_in_sout_fcst_dummy {
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
    category PARTIAL_WEEK PK
  }

  df_step01_01_sin_pick {
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
    category PARTIAL_WEEK PK
    float SIN_DUMMY_TARGET
  }
  df_step01_03_sout_pick {
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
    category PARTIAL_WEEK PK
    float SOUT_DUMMY_TARGET
  }

  Output_SIn_Dummy {
    category VERSION PK
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
    category PARTIAL_WEEK PK
    float SIN_DUMMY_TARGET
  }
  Output_SOut_Dummy {
    category VERSION PK
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
    category PARTIAL_WEEK PK
    float SOUT_DUMMY_TARGET
  }

  Output_SIn_Assortment {
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
    float SIN_ASSORT_TARGET
  }
  Output_SOut_Assortment {
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
    float SOUT_ASSORT_TARGET
  }

  df_output_Sell_In_FCST_GI_AP2 {
    category VERSION PK
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
    category PARTIAL_WEEK PK
    float SIN_GI_AP2
    float SIN_BL_AP2
  }

  df_output_Estimated_Price_Local {
    category VERSION PK
    category SHIP_TO PK
    category ITEM PK
    category PARTIAL_WEEK PK
    float EST_PRICE_LOCAL
    int   EST_PRICE_COLOR
  }

  df_output_Sell_In_FCST_GI_Split_Ratio_AP2 {
    category VERSION PK
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
    category PARTIAL_WEEK PK
    float SIN_SR_AP2
  }

  df_output_Sell_In_User_GI_Ratio {
    category VERSION PK
    category SHIP_TO PK
    category ITEM PK
    category LOCATION PK
    category WEEK PK
  }
```

---

# 2) Step별 서브 ERD (컬럼 포함)

## Step 1-0) Sales 선정 (salesItemLocation 파싱 → eStore/SDD 필터)

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
  df_in_Sales_Domain_Estore {
    category SHIP_TO
  }
  df_step01_00_sales_pairs {
    category SHIP_TO
    category ITEM
    category LOCATION
  }

  df_in_Sales_Domain_Estore ||--o{ df_step01_00_sales_pairs : "filter ShipTo set"
  df_in_Sales_Domain_Dimension ||--o{ df_step01_00_sales_pairs : "filter valid ShipTo"
```

## Step 1-1) S/In Dummy → 생성대상 선별

``` mermaid
erDiagram
  df_in_sin_fcst_dummy {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_DUMMY_AP1
    float SIN_DUMMY_AP2
    float SIN_DUMMY_GC
    float SIN_DUMMY_LOCAL
  }
  df_step01_00_sales_pairs {
    category SHIP_TO
    category ITEM
    category LOCATION
  }
  df_in_sin_fcst {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_FCST_MEASURE
  }
  df_step01_01_sin_pick {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_DUMMY_TARGET
  }

  df_step01_00_sales_pairs ||--o{ df_step01_01_sin_pick : "pairs filter"
  df_in_sin_fcst_dummy ||--o{ df_step01_01_sin_pick : "pick dummy by measureLv"
  df_in_sin_fcst ||--o{ df_step01_01_sin_pick : "exclude existing fcst (>=0)"
```

## Step 1-2) S/In Dummy 삭제용 Output

``` mermaid
erDiagram
  df_step01_01_sin_pick {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_DUMMY_TARGET
  }
  Output_SIn_Dummy {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_DUMMY_TARGET
  }

  df_step01_01_sin_pick ||--o{ Output_SIn_Dummy : "VERSION insert + dummy=NaN"
```

## Step 1-3) S/Out Dummy → 생성대상 선별

``` mermaid
erDiagram
  df_in_sout_fcst_dummy {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_DUMMY_AP1
    float SOUT_DUMMY_AP2
    float SOUT_DUMMY_GC
    float SOUT_DUMMY_LOCAL
  }
  df_step01_00_sales_pairs {
    category SHIP_TO
    category ITEM
    category LOCATION
  }
  df_in_sout_fcst {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_FCST_MEASURE
  }
  df_step01_03_sout_pick {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_DUMMY_TARGET
  }

  df_step01_00_sales_pairs ||--o{ df_step01_03_sout_pick : "pairs filter"
  df_in_sout_fcst_dummy ||--o{ df_step01_03_sout_pick : "pick dummy by measureLv"
  df_in_sout_fcst ||--o{ df_step01_03_sout_pick : "exclude existing fcst (>=0)"
```

## Step 1-4) S/Out Dummy 삭제용 Output

``` mermaid
erDiagram
  df_step01_03_sout_pick {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_DUMMY_TARGET
  }
  Output_SOut_Dummy {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_DUMMY_TARGET
  }

  df_step01_03_sout_pick ||--o{ Output_SOut_Dummy : "VERSION insert + dummy=NaN"
```

## Step 2-1) S/In Assortment 생성

``` mermaid
erDiagram
  df_step01_01_sin_pick {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_DUMMY_TARGET
  }
  Output_SIn_Assortment {
    category SHIP_TO
    category ITEM
    category LOCATION
    float SIN_ASSORT_TARGET
  }

  df_step01_01_sin_pick ||--o{ Output_SIn_Assortment : "drop PW + max(1)"
```

## Step 2-2) S/Out Assortment 생성

``` mermaid
erDiagram
  df_step01_03_sout_pick {
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SOUT_DUMMY_TARGET
  }
  Output_SOut_Assortment {
    category SHIP_TO
    category ITEM
    category LOCATION
    float SOUT_ASSORT_TARGET
  }

  df_step01_03_sout_pick ||--o{ Output_SOut_Assortment : "drop PW + max(1)"
```

## Step 3) FCST 0값 스켈레톤 생성 (measureLv별)

``` mermaid
erDiagram
  df_step01_01_sin_pick {
    category SHIP_TO
    category ITEM
    category LOCATION
  }
  df_in_Time_pw {
    category PARTIAL_WEEK
  }
  df_output_Sell_In_FCST_GI_AP2 {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_GI_AP2
    float SIN_BL_AP2
  }

  df_step01_01_sin_pick ||--o{ df_output_Sell_In_FCST_GI_AP2 : "keys from dummy pick"
  df_in_Time_pw ||--o{ df_output_Sell_In_FCST_GI_AP2 : "expand by PW"
```

## Step 4) Estimated Price Local + Color 생성

``` mermaid
erDiagram
  df_step04_00_targets {
    category SHIP_TO
    category ITEM
  }
  df_in_Time_pw {
    category PARTIAL_WEEK
  }
  df_in_Item_Master {
    category ITEM
    category ITEM_STD2
    category ITEM_STD3
    category ITEM_STD4
  }
  df_in_Sales_Domain_Dimension {
    category SHIP_TO
    category STD2
    category STD3
  }
  df_in_Estimated_Price {
    category SHIP_TO
    category ITEM
    category PARTIAL_WEEK
    float EST_PRICE_MOD_LOCAL
    float EST_PRICE_LOCAL
  }
  df_in_Action_Plan_Price {
    category SHIP_TO
    category ITEM
    category MONTH
    float AP_PRICE_USD
  }
  df_in_Exchange_Rate_Local {
    category STD3
    category PARTIAL_WEEK
    float EXRATE_LOCAL
  }
  df_output_Estimated_Price_Local {
    category VERSION
    category SHIP_TO
    category ITEM
    category PARTIAL_WEEK
    float EST_PRICE_LOCAL
    int   EST_PRICE_COLOR
  }

  df_step04_00_targets ||--o{ df_output_Estimated_Price_Local : "ShipTo*Item x PW"
  df_in_Time_pw ||--o{ df_output_Estimated_Price_Local : "expand"
  df_in_Estimated_Price ||--o{ df_output_Estimated_Price_Local : "priority 1-2"
  df_in_Action_Plan_Price ||--o{ df_output_Estimated_Price_Local : "priority 9"
  df_in_Exchange_Rate_Local ||--o{ df_output_Estimated_Price_Local : "priority 9"
```

## Step 5) Split Ratio 생성

``` mermaid
erDiagram
  df_output_Sell_In_FCST_GI_AP2 {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
  }
  df_in_Item_Master {
    category ITEM
    category ITEM_STD1
  }
  df_in_Sales_Domain_Dimension {
    category SHIP_TO
    category STD2
  }
  df_in_Sell_In_FCST_GI_Split_Ratio_AP2 {
    category STD2
    category ITEM_STD1
    category PARTIAL_WEEK
    float SIN_SR_AP2
  }
  df_output_Sell_In_FCST_GI_Split_Ratio_AP2 {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category PARTIAL_WEEK
    float SIN_SR_AP2
  }

  df_output_Sell_In_FCST_GI_AP2 ||--o{ df_output_Sell_In_FCST_GI_Split_Ratio_AP2 : "keys from step3"
  df_in_Item_Master ||--o{ df_output_Sell_In_FCST_GI_Split_Ratio_AP2 : "Item->Std1"
  df_in_Sales_Domain_Dimension ||--o{ df_output_Sell_In_FCST_GI_Split_Ratio_AP2 : "ShipTo->Std2"
  df_in_Sell_In_FCST_GI_Split_Ratio_AP2 ||--o{ df_output_Sell_In_FCST_GI_Split_Ratio_AP2 : "join by Std2+Std1+PW"
```

## Step 7) eStore Ratio 생성 (공통)

``` mermaid
erDiagram
  df_step07_0_targets {
    category SHIP_TO
    category ITEM
    category LOCATION
  }
  df_in_Time_w {
    category WEEK
  }
  df_in_Item_Master {
    category ITEM
    category ITEM_STD1
  }
  df_in_Sell_In_Final_GI_Ratio {
    category SHIP_TO
    category ITEM_STD1
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_Final_GI_Ratio_Item {
    category SHIP_TO
    category ITEM
    category LOCATION
    category WEEK
  }
  df_in_Sell_In_Final_GI_Ratio_PG {
    category ITEM_STD1
    category LOCATION
    category WEEK
  }
  df_output_Sell_In_Final_GI_Ratio {
    category VERSION
    category SHIP_TO
    category ITEM
    category LOCATION
    category WEEK
  }

  df_step07_0_targets ||--o{ df_output_Sell_In_Final_GI_Ratio : "targets x week"
  df_in_Time_w ||--o{ df_output_Sell_In_Final_GI_Ratio : "week grid"
  df_in_Item_Master ||--o{ df_output_Sell_In_Final_GI_Ratio : "Item->PG"
  df_in_Sell_In_Final_GI_Ratio_Item ||--o{ df_output_Sell_In_Final_GI_Ratio : "exists flags"
  df_in_Sell_In_Final_GI_Ratio ||--o{ df_output_Sell_In_Final_GI_Ratio : "ShipTo+PG source (case1)"
  df_in_Sell_In_Final_GI_Ratio_PG ||--o{ df_output_Sell_In_Final_GI_Ratio : "PG source (case2)"
```

---

# 3) 전체 Sequence Diagram

``` mermaid
sequenceDiagram
  autonumber
  participant Main
  participant IO as fn_process_in_df_mst
  participant S10 as Step1-0 fn_step01_00_select_sales
  participant S11 as Step1-1 fn_step01_01_pick_sin_dummy
  participant S12 as Step1-2 fn_step01_02_build_output_sin_dummy_delete
  participant S13 as Step1-3 fn_step01_03_pick_sout_dummy
  participant S14 as Step1-4 fn_step01_04_build_output_sout_dummy_delete
  participant S21 as Step2-1 fn_step02_01_build_sin_assortment
  participant S22 as Step2-2 fn_step02_02_build_sout_assortment
  participant S3 as Step3 Build FCST outputs
  participant S40 as Step4-0 fn_step04_00_select_price_targets
  participant S41 as Step4-1 fn_step04_01_build_est_price_local
  participant S42 as Step4-2 fn_step04_02_fill_missing_price_from_children
  participant S44 as Step4-4 fn_step04_04_format_est_price_output
  participant S5 as Step5 Build Split Ratio outputs
  participant S70 as Step7-0 fn_step07_0_select_targets
  participant S700 as Step7-00 fn_step07_00_build_pg_ratio_tables
  participant S7 as Step7-1..7-8 fn_step07_build_by_group

  Main->>IO: load all df_in_* into input_dataframes
  IO-->>Main: input_dataframes ready

  Main->>S10: df_in_SDD + df_in_ESTORE + salesItemLocation
  S10-->>Main: df_step01_00_sales_pairs (+ pairs_have_loc)

  Main->>S11: df_in_sin_fcst_dummy + pairs + measureLv + (optional df_in_sin_fcst)
  S11-->>Main: df_step01_01_sin_pick
  Main->>S12: df_step01_01_sin_pick + Version
  S12-->>Main: Output_SIn_Dummy

  Main->>S13: df_in_sout_fcst_dummy + pairs + measureLv + (optional df_in_sout_fcst)
  S13-->>Main: df_step01_03_sout_pick
  Main->>S14: df_step01_03_sout_pick + Version
  S14-->>Main: Output_SOut_Dummy

  Main->>S21: df_step01_01_sin_pick
  S21-->>Main: Output_SIn_Assortment
  Main->>S22: df_step01_03_sout_pick
  S22-->>Main: Output_SOut_Assortment

  Main->>S3: df_step01_*_pick + df_in_Time_pw + Version + measureLv
  S3-->>Main: df_output_Sell_In_FCST_GI_* + df_output_Sell_Out_FCST_*

  Main->>S40: df_step01_01_sin_pick + df_step01_03_sout_pick
  S40-->>Main: df_step04_00_targets(ShipTo*Item)
  Main->>S41: targets + Time_pw + Item_Master + SDD + EstPrice + EP* + APPrice + ExRate + Version
  S41-->>Main: df_step04_01_est_local
  Main->>S42: df_step04_01_est_local + SDD
  S42-->>Main: df_step04_02_est_local_filled
  Main->>S44: df_step04_02_est_local_filled + Version
  S44-->>Main: df_output_Estimated_Price_Local

  Main->>S5: df_step03_* + Item_Master + SDD + SplitRatio df_in_* + Version + measureLv
  S5-->>Main: df_output_Sell_In_*_Split_Ratio_* + df_output_Sell_Out_*_Split_Ratio_*

  Main->>S70: df_step01_01_sin_pick + df_step01_03_sout_pick
  S70-->>Main: df_step07_0_targets(ShipTo*Item*Location)
  Main->>S700: input_dataframes(23~30)
  S700-->>Main: df_in_*_PG tables stored into input_dataframes

  Main->>S7: group_key + targets + Time_w + Item_Master + input_dataframes + Version
  S7-->>Main: df_output_Sell_In_*_GI_Ratio (7-1..7-8)
```

---

# 4) Step별 상세 Sequence

## Step 1-1) Dummy 선별(공통 패턴)

``` mermaid
sequenceDiagram
  autonumber
  participant S as fn_step01_01_pick_sin_dummy
  participant D as df_in_sin_fcst_dummy
  participant P as df_step01_00_sales_pairs
  participant F as df_in_sin_fcst (optional)
  participant O as df_step01_01_sin_pick

  S->>D: select keys + pick dummy column by measureLv
  S->>P: semi-join filter by (ShipTo,Item[,Location])
  S->>F: anti-join exclude where existing FCST >= 0 (optional)
  S-->>O: return filtered rows (keys + dummy measure)
```

## Step 4-1) Estimated Price 우선순위 Coalesce

``` mermaid
sequenceDiagram
  autonumber
  participant S as fn_step04_01_build_est_price_local
  participant T as df_step04_00_targets
  participant PW as df_in_Time_pw
  participant MST as df_in_Item_Master
  participant SDD as df_in_Sales_Domain_Dimension
  participant EP as df_in_Estimated_Price
  participant EPI as EP Std4/3/2 + SalesStd2*Std4/3/2
  participant AP as df_in_Action_Plan_Price
  participant EX as df_in_Exchange_Rate_Local
  participant OUT as df_output_Estimated_Price_Local

  S->>T: keys (ShipTo, Item)
  S->>PW: expand keys x PartialWeek
  S->>MST: map Item -> ItemStd2/3/4
  S->>SDD: map ShipTo -> SalesStd3(+SalesStd2)
  S->>EP: merge (ShipTo,Item,PW) for Modify/Local
  S->>EPI: merge fallback sources (Std4/3/2, SalesStd2+Std4/3/2)
  S->>AP: merge (ShipTo,Item,Month) AP USD
  S->>EX: merge (SalesStd3,PW) ExRate
  S->>S: compute AP*EXRATE fallback
  S->>S: coalesce priority 1..9 + set Color(0 base, 1 fallback, NA)
  S-->>OUT: Version + (ShipTo,Item,PW) + EstimatedPriceLocal + Color
```

## Step 7 공통: 신규/기존 판정 & 값 생성

``` mermaid
sequenceDiagram
  autonumber
  participant S as fn_step07_build_by_group
  participant C as _fn_step07_build_ratio_common
  participant T as df_step07_0_targets
  participant W as df_in_Time_w
  participant MST as df_in_Item_Master
  participant I as df_in_*_Ratio_Item (ShipTo+Item level)
  participant R as df_in_*_Ratio (ShipTo+PG level)
  participant PG as df_in_*_Ratio_PG (PG level)
  participant OUT as df_output_Sell_In_*_GI_Ratio

  S->>C: pass meta(df_in, df_in_item, df_pg)
  C->>T: build (ShipTo,Item,Location) targets
  C->>W: cross join to make targets x Week grid
  C->>MST: map Item -> PG(ItemStd1)
  C->>I: exists_item_in_item, exists_shipto_in_item flags
  C->>C: drop rows where exists_shipto_in_item==True (case3)
  C->>R: if 신규 Item(exists_item_in_item==False) fill from ShipTo+PG+Loc+Week (case1)
  C->>PG: if 기존 Item + 신규 ShipTo fill from PG+Loc+Week (case2)
  C->>C: inject Version, cast, float32
  C-->>OUT: output ratio dataframe
```

---

# 5) Flow Chart

## 5-1) 전체 Flowchart

``` mermaid
flowchart TD
  A[Start
load input_dataframes] --> B[Step 1-0
Parse salesItemLocation
Filter by eStore/SDD]

  B --> C1[Step 1-1
Pick S/In Dummy 대상]
  C1 --> C2[Step 1-2
Build Output_SIn_Dummy(delete)]

  B --> D1[Step 1-3
Pick S/Out Dummy 대상]
  D1 --> D2[Step 1-4
Build Output_SOut_Dummy(delete)]

  C1 --> E1[Step 2-1
Build S/In Assortment]
  D1 --> E2[Step 2-2
Build S/Out Assortment]

  E1 --> F[Step 3
Build FCST skeletons
(measureLv guard)]
  E2 --> F

  F --> G0[Step 4-0
Select price targets
(S/In ∩ S/Out)]
  G0 --> G1[Step 4-1
Build Estimated Price Local
(priority 1..9)]
  G1 --> G2[Step 4-2
Fill missing from children avg]
  G2 --> G4[Step 4-4
Format output]

  F --> H[Step 5
Build Split Ratios
(measureLv guard)]

  C1 --> I0[Step 7-0
Select eStore ratio targets
(S/In ∩ S/Out)]
  D1 --> I0
  I0 --> I00[Step 7-00
Build PG mean tables]
  I00 --> I1[Step 7-1..7-8
Build ratio outputs by group]

  C2 --> Z[End: Outputs
Dummy delete + Assortment + FCST + Price + SplitRatio + eStoreRatio]
  D2 --> Z
  E1 --> Z
  E2 --> Z
  F --> Z
  G4 --> Z
  H --> Z
  I1 --> Z
```

## 5-2) Step 7-1~7-8 Flowchart(공통)

``` mermaid
flowchart TD
  S[Start Step7 group] --> T[Targets = (ShipTo,Item,Loc)
from Step7-0]
  T --> W[Cross Join Week Grid
Targets x Time.[Week]]
  W --> PG[Map Item -> Item Std1(PG)]

  PG --> E1{exists_item_in_item?
(using *_Ratio_Item)}
  E1 -- No (신규 Item) --> C1[Case1 fill
from ShipTo+PG+Loc+Week
(df_in_*_Ratio)]
  E1 -- Yes --> E2{exists_shipto_in_item?
(using *_Ratio_Item)}
  E2 -- Yes --> DROP[Case3: drop row
(do not generate)]
  E2 -- No --> C2[Case2 fill
from PG+Loc+Week
(df_in_*_Ratio_PG)]

  C1 --> OUT[Inject Version
Cast category/float32]
  C2 --> OUT
  DROP --> OUT
```

---

# 6) 산출물(Outputs) 요약

## 6-1) Dummy Delete Outputs

- `Output_SIn_Dummy` (`DF_OUT_SIN_DUMMY`)
  - Grain: `Version + ShipTo + Item + Location + PartialWeek`
  - Measure: 선택된 Dummy 컬럼(값은 삭제용으로 `NaN`)
- `Output_SOut_Dummy` (`DF_OUT_SOUT_DUMMY`)
  - Grain 동일

## 6-2) Assortment Outputs

- `Output_SIn_Assortment` (`DF_OUT_SIN_ASSORT`)
  - Grain: `ShipTo + Item + Location`
- `Output_SOut_Assortment` (`DF_OUT_SOUT_ASSORT`)
  - Grain 동일

## 6-3) FCST 스켈레톤 Outputs (measureLv에 따라 비어있을 수 있음)

- `df_output_Sell_In_FCST_GI_AP1/AP2/GC/Local`
- `df_output_Sell_Out_FCST_AP1/AP2/GC/Local`

## 6-4) Estimated Price Output

- `df_output_Estimated_Price_Local`
  - Grain: `Version + ShipTo + Item + PartialWeek`
  - Measure
    - `Estimated Price_Local`
    - `Estimated Price Color` (Int32, 0=base, 1=fallback, NA=미생성)

## 6-5) Split Ratio Outputs

- Sell-In Split Ratio
  - `df_output_Sell_In_FCST_GI_Split_Ratio_AP1/AP2/GC/Local`
- Sell-Out Split Ratio
  - `df_output_Sell_Out_FCST_Split_Ratio_AP1/AP2/GC/Local`

## 6-6) eStore Ratio Outputs (7-1~7-8)

- `df_output_Sell_In_User_GI_Ratio`
- `df_output_Sell_In_Issue_GI_Ratio`
- `df_output_Sell_In_Final_GI_Ratio`
- `df_output_Sell_In_BestFit_GI_Ratio`
- `df_output_Sell_In_User_Item_GI_Ratio`
- `df_output_Sell_In_Issue_Item_GI_Ratio`
- `df_output_Sell_In_User_Modify_GI_Ratio`
- `df_output_Sell_In_Issue_Modify_GI_Ratio`

(각 output은 동일 Grain: `Version + ShipTo + Item + Location + Week`)
