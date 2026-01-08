"""
* 요구사항
    - EW UI S/In FCST(GI) 값 저장 시, Simulation 값 저장을 위한 Python Plugin
    - UI에서 보여지는 Measure 외 연산

* 변경이력
    - 2025.09.15 / lee kyung young / create
    - 2025.09.18 / lee kyung young / changed / df_in_d 의 Data Input 포멧 변경
    - 2025.09.25 / lee kyung young / changed / (Simul) Ch Inv 와 WOS 계산 로직 변경,
                                               (Simul) Ch Inv의 최종 Grain 변경으로 인한 Measure 이름 교체. (ex. Measuer 명 변경 예시 (Simul)Ch Inv_AP2 -> (Simul)Ch Inv PW_AP2)
                                               (Simul) S/In Guide 생성 로직 추가 및 Floor FCST 처리 로직 추가
    - 2025.10.15 / lee kyung young / changed / S/Out FCST Modify_AP2 생성 추가
    - 2025.10.21 / lee kyung young / changed / S/Out FCST AP1 -> AP2 변경은 오타로. 변경
    - 2025.10.27 / lee kyung young / changed / 1. Region 삭제, Floor Grain 조정, WOS 로직 삭제, (Week Avg)S/Out FCST F4W PW_AP1 저장 내용 추가
    - 2025.11.19 / lee kyung young / changed / 25.11.15 변경점
                                                0) salesLv 신규 Input Parameter 추가
                                                1) AMT 계산에 사용하는 Measure 변경에 따른 로직 변경 : Excahnge Rate -> Estimated Price_Local, Estimated Price_Region
                                                2) GI -> BL 로직 변경
                                                3) Ch Inv Floor 생성 로직 변경
    - 2025.12.08 / lee kyung young / changed / (25.12.06) : Delta 인 S/In FCST에 대해서 S/In FCST(GI) Modify_AP2 Measure 추가 및 값 1 적용

* Script Parameter
    - CurrentWeek = &CurrentWeek

* Input
    - df_in_h = pd.read_csv('./input/dp_fcst_ar/df_in_h.csv')

        Select (
            [DP Data Seq].[DP Data Seq]
        ) on row,
        ({
            Measure.[Upload Header_Dimension],
            Measure.[Upload Header_Measure]
        }) on column
        where {
            [Personnel].[Email],
            [Version].[Version Name]
        };


    - df_in_d = pd.read_csv('./input/dp_fcst_ar/df_in_d.csv')
        Select (
            [DP Data Seq].[DP Data Seq]
            * [DP Data Seq Detail].[DP Data Seq Detail]
        ) on row,
        ({
            Measure.[Upload Data_Data]
        }) on column
        where {
            [Personnel].[Email], [Version].[Version Name]
        };


    - df_in_sellin = pd.read_csv('./input/dp_fcst_ar/df_in_sellin.csv')
        Select (
            [Sales Domain].[Ship To]
            * [Item].[Item]
            * [Location].[Location]
            * [Time].[Partial Week]
        ) on row,
        ({
            Measure.[S/In FCST(GI)_AP2],
            Measure.[S/In FCST(BL)_AP2]
        }) on column
        where {
            [Version].[Version Name]
        };


    - df_in_sellout = pd.read_csv('./input/dp_fcst_ar/df_in_sellout.csv')
        Select (
            [Sales Domain].[Ship To]
            * [Item].[Item]
            * [Location].[Location]
            * [Time].[Partial Week]
        ) on row,
        ({
            Measure.[S/Out FCST_AP2]
        }) on column
        where {
            [Version].[Version Name]
        };


    - df_in_transit_time = pd.read_csv('./input/dp_fcst_ar/df_in_transit_time.csv')
        Select (
            [Sales Domain].[Ship To]
            * [Item].[Product Group]
            * [Location].[Location]
        ) on row,
        ({
            Measure.[DP Master GI BL LT Transit Time W]
        }) on column
        where {
            [Version].[Version Name]
        };


    - df_in_split_ratio = pd.read_csv('./input/dp_fcst_ar/df_in_split_ratio.csv')
        Select (
            [Sales Domain].[Ship To]
            * [Item].[Item]
            * [Location].[Location]
            * [Time].[Partial Week]
        ) on row,
        ({
            Measure.[S/In FCST(GI) Split Ratio_AP2]
        }) on column
        where {
            [Version].[Version Name]
        };


    - df_in_estimated_price_USD = pd.read_csv('./input/dp_fcst_ar/df_in_estimated_price_USD.csv')
        Select (
            [Sales Domain].[Ship To]
            * [Item].[Item]
            * [Time].[Partial Week]
        ) on row,
        ({
            Measure.[Estimated Price_USD],
            Measure.[Estimated Price_Local],
            Measure.[Estimated Price_Region]
        }) on column
        where {
            [Version].[Version Name]
        };


    - * 25.11.15 해당 Input 삭제
      df_in_exchange_rate = pd.read_csv('./input/dp_fcst_ar/df_in_exchange_rate.csv')
        Select (
            [Sales Domain].[Sales Std3]
            * [Time].[Partial Week]
         ) on row,
        ({
            Measure.[Exchange Rate_Local]
            , Measure.[Exchange Rate_Region]
        }) on column
        where {
            [Version].[Version Name]
        };


    + df_in_chinv = pd.read_csv('./input/dp_fcst_ar/df_in_chinv.csv')
        Select (
            [Sales Domain].[Ship To]
            * [Item].[Item]
        ) on row,
        ({
            Measure.[(Simul)Ch Inv PW_AP2],
            Measure.[(Simul)Ch Inv_Inc Floor PW_AP2],
            Measure.[(Simul)P4W Ch Inv PW],
            Measure.[(Simul)P4W Ch Inv_Inc Floor PW]
        }) on column
        where {
            [Version].[Version Name],
            [Time].[Week].filter(#.Key == &currentWeek.element(0).leadoffset(-1).Key)
        };


    - df_in_item = pd.read_csv('./input/dp_fcst_ar/df_in_item.csv')
        Select (
             [Item].[Item GBM]
            * [Item].[Item Std1]
            * [Item].[Item]
        );


    - df_in_sales = pd.read_csv('./input/dp_fcst_ar/df_in_sales.csv')
        Select (
            [Sales Domain].[Sales Std1]
             * [Sales Domain].[Sales Std2]
             * [Sales Domain].[Sales Std3]
             * [Sales Domain].[Sales Std4]
             * [Sales Domain].[Sales Std5]
             * [Sales Domain].[Sales Std6]
             * [Sales Domain].[Ship To]
        );

    + df_in_target_WOS = pd.read_csv('./input/dp_fcst_ar/df_in_target_WOS.csv')
        Select (
            [Sales Domain].[Ship To]
            * [Item].[Item]
            * [Time].[Week]
        ) on row,
        ({
            Measure.[Target Ch WOS]
        }) on column
        where {
            [Version].[Version Name]
        };

    + df_in_floor_fcst = pd.read_csv('./input/dp_fcst_ar/df_in_floor_fcst.csv')
        Select (
            [Sales Domain].[Ship To]
            * [Item].[Item]
            * [Time].[Week]
        ) on row,
        ({
            Measure.[Flooring FCST]
        }) on column
        where {
            [Version].[Version Name]
        };


    - df_in_version = pd.read_csv('./input/dp_fcst_ar/df_in_version.csv')
        select ([Version].[Version Name]
        );


    - df_in_gi_bl_intransit_fcst = pd.read_csv('./input/dp_fcst_ar/df_in_gi_bl_intransit_fcst.csv')
        Select (
            [Sales Domain].[Ship To]
            * [Item].[Item]
            * [Location].[Location]
            * [Time].[Partial Week].filter (#.Key >= &CurrentWeek.element(0).Key && #.Key < &CurrentWeek.element(0).leadoffset(1).Key)
        )on row,
        ({Measure.[GI BL Intransit]}) on column
        where { [Version].[Version Name] };


    - df_in_fcst_gi_assortment = pd.read_csv('./input/dp_fcst_ar/df_in_fcst_gi_assortment.csv')
        Select ([Version].[Version Name]
            * [Item].[Item]
            * [Sales Domain].[Ship To]
            * [Location].[Location]
        ) on row,
        ({
            Measure.[S/In FCST(GI) Assortment_AP2]
        }) on column;

* Output
    - df_output
        df_output_Sellin_gi
                Select (
                    [Version].[Version Name]
                    * [Sales Domain].[Ship To]
                    * [Item].[Item]
                    * [Location].[Location]
                    * [Time].[Partial Week]
                ) on row,
                ({
                    Measure.[S/In FCST(GI)_AP2],
                    Measure.[S/In FCST(GI) AMT_USD_AP2],
                    Measure.[S/In FCST(GI) AMT_Local_AP2],
                    Measure.[(Simul)S/In FCST_AP2]
                }) on column;

        df_output_Sellin_bl
                Select (
                    [Version].[Version Name]
                    * [Sales Domain].[Ship To]
                    * [Item].[Item]
                    * [Location].[Location]
                    * [Time].[Partial Week]
                ) on row,
                ({
                    Measure.[S/In FCST(BL)_AP2],
                    Measure.[S/In FCST(BL) AMT_USD_AP2],
                    Measure.[S/In FCST(BL) AMT_Local_AP2],
                    Measure.[S/In FCST(BL) AMT_Region_AP2]
                }) on column;

        df_output_Sellout
                Select (
                    [Version].[Version Name]
                    * [Sales Domain].[Ship To]
                    * [Item].[Item]
                    * [Location].[Location]
                    * [Time].[Partial Week]
                ) on row,
                ({
                    Measure.[S/Out FCST_AP2],
                    Measure.[S/Out FCST AMT_USD_AP2],
                    Measure.[S/Out FCST AMT_Local_AP2]
                }) on column;

        df_output_chinv
                Select (
                    [Version].[Version Name]
                    * [Sales Domain].[Ship To]
                    * [Item].[Item]
                    * [Time].[Week]
                ) on row,
                ({
                    Measure.[(Simul)Ch Inv PW_AP2],
                    Measure.[(Simul)Ch Inv_Inc Floor PW_AP2],
                    Measure.[(Simul)P4W Ch Inv PW],
                    Measure.[(Simul)P4W Ch Inv_Inc Floor PW],
                    Measure.[(Simul)Target Ch Inv PW_AP2]
                }) on column;

        // 제외처리
        df_output_wos
                Select (
                    [Version].[Version Name]
                    * [Sales Domain].[Ship To]
                    * [Item].[Item]
                    * [Time].[Partial Week]
                ) on row,
                ({
                    Measure.[(Simul)WOS_AP2],
                    Measure.[(Simul)WOS_Inc Floor_AP2],
                    Measure.[(Simul)P4W WOS],
                    Measure.[(Simul)P4W WOS_Inc Floor]
                }) on column;


* Flow Summary
    전처리

    Step 01. S/In FCST(GI) , S/Out FCST Delta값 DataFrame 형태로 변환 및 전체데이터와 합치기, Delta 별도 관리
        - df_in_h와 df_in_d 를 이용해 UI에서 입력된 Delta 값 DataFrame 변환
        ex)
        - df_in_h
        DP Data Seq.[DP Data Seq]	Upload Header_Dimension	Upload Header_Measure
        3	Version.[Version Name]^Sales Domain.[Ship To]^Item.[Item]^Location.[Location]^Time.[Partial Week]	S/In FCST(GI)_AP2
        4	Version.[Version Name]^Sales Domain.[Ship To]^Item.[Item]^Location.[Location]^Time.[Partial Week]	S/Out FCST_AP2

        - df_in_d
        DP Data Seq.[DP Data Seq]	DP Data Seq Detail.[DP Data Seq Detail]	Upload Data_Data
        3	1	408158^RR39C7AF5WW/EF^S406^202521A:2000|202522A:2000|202522B:2000|202523A:2000|202524A:2000|202525A:2000|202526A:2000|202527A:2000|202527B:2000|202528A:2000|202529A:2000|202530A:2000|202531A:2000|202531B:2000|202532A:2000|202533A:2000|202534A:2000|202535A:2000|202536A:2000|202537A:2000|202538A:2000|202539A:2000|202540A:2000|202540B:2000|202541A:2000|202542A:2000|202543A:2000|202544A:2000|202544B:2000|202545A:2000|202546A:2000|202547A:2000|202548A:2000|202549A:2000|202550A:2000|202551A:2000|202552A:2000|202601A:2000|202601B:2000|202602A:2000|202603A:2000|202604A:2000|202605A:2000|202605B:2000|202606A:2000|202607A:2000|202608A:2000|202609A:2000|202609B:2000|202610A:2000|202611A:2000|202612A:2000|202613A:2000|202614A:2000|202614B:2000|202615A:2000|202616A:2000|202617A:2000|202618A:2000|202618B:2000|202619A:2000|202620A:2000|202621A:2000|
        3	2	408158^RR39C7AF5WW/EF^S406WV2H^202521A:2000|202522A:2000|202522B:2000|202523A:2000|202524A:2000|202525A:2000|202526A:2000|202527A:2000|202527B:2000|202528A:2000|202529A:2000|202530A:2000|202531A:2000|202531B:2000|202532A:2000|202533A:2000|202534A:2000|202535A:2000|202536A:2000|202537A:2000|202538A:2000|202539A:2000|202540A:2000|202540B:2000|202541A:2000|202542A:2000|202543A:2000|202544A:2000|202544B:2000|202545A:2000|202546A:2000|202547A:2000|202548A:2000|202549A:2000|202550A:2000|202551A:2000|202552A:2000|202601A:2000|202601B:2000|202602A:2000|202603A:2000|202604A:2000|202605A:2000|202605B:2000|202606A:2000|202607A:2000|202608A:2000|202609A:2000|202609B:2000|202610A:2000|202611A:2000|202612A:2000|202613A:2000|202614A:2000|202614B:2000|202615A:2000|202616A:2000|202617A:2000|202618A:2000|202618B:2000|202619A:2000|202620A:2000|202621A:2000|

        - df_in_sellin S/In FCST(GI) 와 합치기
        - df_in_sellout 이용해서 S/Out FCST 와 합치기

    Step 02. S/Out FCST Data 가공
        - S/Out FCST_AP2 Aggregation (Ship To, Item, Week)
        - S/Out Actual(P4W) 값 생성 : df_in_sellout 에서 S/Out FCST 값 가져와서 P4W 생성
        - (Week Avg)S/Out FCST F4W_AP2 값 생성 : S/Out FCST 값 가져와서 F4W로 만들어야함. 이때 Location은 필요 없음

    Step 03. S/In FCST(BL) 값 연산
        - S/In FCST(GI) 값을 Week 단위로 Aggregation
        - df_in_transit_time 을 이용해 Ship To * Product Group * Location 별로 DP Master GI BL LT Transit Time W 값만큼 이동하여 S/In FCST(BL) 값 연산
        - df_in_Split_Ratio 를 이용해 S/In FCST(BL) 값 Week -> Partial week 연산
        - 소수점 처리 로직

    Step 04. Ch Inv 값 연산
	    - S/In FCST(BL)_AP2 Aggregation (Ship To, Item, Week)
	    - (Simul)Ch Inv PW_AP2, (Simul)Ch Inv_Inc loor_AP2 : Measure.[(Simul)Ch Inv PW_AP2] = if ( SUM(Measure.[(Simul)Ch Inv PW_AP2]@(Time.#.leadoffset(-1)), Measure.[(Copy)S/In FCST(BL)_AP2], Measure.[(Copy)S/Out FCST_AP2] * -1) < 0 ) then 0
                                                                                        else SUM(Measure.[(Simul)Ch Inv PW_AP2]@(Time.#.leadoffset(-1)), Measure.[(Copy)S/In FCST(BL)_AP2], Measure.[(Copy)S/Out FCST_AP2] * -1);"
	    - (Simul)P4W Ch Inv PW, (Simul)P4W Ch Inv_Inc Floor PW : Measure.[(Simul)P4W Ch Inv PW] = if ( SUM(Measure.[(Simul)P4W Ch Inv PW]@(Time.#.leadoffset(-1)), Measure.[(Copy)S/In FCST(BL)_AP2], Measure.[(Copy)S/Out Actual(P4W)] * -1) < 0) then 0else SUM(Measure.[(Simul)P4W Ch Inv PW]@(Time.#.leadoffset(-1)), Measure.[(Copy)S/In FCST(BL)_AP2], Measure.[(Copy)S/Out Actual(P4W)] * -1);

    Step 05. WOS 값 연산
	    - (Input 3) WOS 계산을 위한 정보 <- WOS는 Recurrence가 아니기 때문에 데이터 가져올 필요 X, 매번 새로운 Data 넣어주어야함. Q : Delta만 저장할 것인지. 아니면 52주 매번 전체 저장할 것인지
        - (Simul)WOS_AP2, (Simul)WOS_Inc Floor_AP2 : Measure.[(Simul)WOS_AP2] = Safedivide(Measure.[(Simul)Ch Inv PW_AP2], Measure.[(Week Avg)S/Out FCST F4W_AP2], 0);
        - (Simul)P4W WOS, (Simul)P4W WOS_Inc Floor : Measure.[(Simul)P4W WOS] = Safedivide(Measure.[(Simul)P4W Ch Inv PW], Measure.[(Week Avg)S/Out Actual(P4W) F4W], 0);
        - WOS 는 Week -> Partial Week 구성 (Copy)

    Step 06. AMT 값 연산
        - S/In FCST(GI) Delta 값 AMT 계산 : USD, Local
        - S/In FCST(BL) Delta 값 AMT 계산 : USD, Local, Region
        - S/Out FCST Delta 값 AMT 계산 : USD, Local

    Step 07. Output 생성
        - S/In FCST(GI), AMT_USD, AMT_Local
        - S/In FCST(BL), AMT_USD, AMT_Local, AMT_Region
        - S/Out FCST, AMT_USD, AMT_Local
        - (Simul)Ch Inv PW_AP2, (Simul)Ch Inv_Inc loor_AP2, (Simul)P4W Ch Inv PW, (Simul)P4W Ch Inv_Inc Floor PW
        - (Simul)WOS_AP2, (Simul)WOS_Inc Floor_AP2, (Simul)P4W WOS, (Simul)P4W WOS_Inc Floor


* Execution

    EXEC plugin instance [PYForecastAR]
    for measures {
    Measure.[S/In FCST(GI)_AP2], Measure.[S/In FCST(GI) AMT_USD_AP2], Measure.[S/In FCST(GI) AMT_Local_AP2],
    Measure.[S/In FCST(BL)_AP2], Measure.[S/In FCST(BL) AMT_USD_AP2], Measure.[S/In FCST(BL) AMT_Local_AP2], Measure.[S/In FCST(BL) AMT_Region_AP2],
    Measure.[S/Out FCST_AP2], Measure.[S/Out FCST AMT_USD_AP2], Measure.[S/Out FCST AMT_Local_AP2],
    Measure.[(Simul)Ch Inv PW_AP2], Measure.[(Simul)Ch Inv_Inc Floor PW_AP2], Measure.[(Simul)P4W Ch Inv PW], Measure.[(Simul)P4W Ch Inv_Inc Floor PW],
    Measure.[(Simul)WOS_AP2], Measure.[(Simul)WOS_Inc Floor_AP2], Measure.[(Simul)P4W WOS], Measure.[(Simul)P4W WOS_Inc Floor]
    }
    using scope (
    [Version].[Version Name].[CWV_DP]
    * [Sales Domain].[Sales Std3]
    * [Sales Domain].[Ship To]
    * [Item].[Item Std1]
    * [Item].[Product Group]
    * [Item].[Item]
    * [Location].[Location]
    * [Time].[Partial Week].filter (#.Key >= &CurrentWeek.element(0).leadoffset(-5).Key && #.Key < &CurrentWeek.element(0).leadoffset(57).Key) )
    * [DP Data Seq].[DP Data Seq]
    * [Personnel].[Email]
    using arguments {
    (ExecutionMode, "MediumWeight")
    ,(IncludeNullrows, False)
    ,(CurrentWeek, "202537")
    };

    // 실행 검증 630cell TEST 진행
    EXEC plugin instance [PYForecastAR]
    for measures {
    Measure.[S/In FCST(GI)_AP2], Measure.[S/In FCST(GI) AMT_USD_AP2], Measure.[S/In FCST(GI) AMT_Local_AP2],
    Measure.[S/In FCST(BL)_AP2], Measure.[S/In FCST(BL) AMT_USD_AP2], Measure.[S/In FCST(BL) AMT_Local_AP2], Measure.[S/In FCST(BL) AMT_Region_AP2],
    Measure.[S/Out FCST_AP2], Measure.[S/Out FCST AMT_USD_AP2], Measure.[S/Out FCST AMT_Local_AP2],
    Measure.[(Simul)Ch Inv_AP2], Measure.[(Simul)Ch Inv_Inc Floor_AP2], Measure.[(Simul)P4W Ch Inv], Measure.[(Simul)P4W Ch Inv_Inc Floor],
    Measure.[(Simul)WOS_AP2], Measure.[(Simul)WOS_Inc Floor_AP2], Measure.[(Simul)P4W WOS], Measure.[(Simul)P4W WOS_Inc Floor]
    }
    using scope (
    [Version].[Version Name].[CWV_DP]
    * [Sales Domain].[Sales Std3].[A300980]
    * [Sales Domain].[Ship To].[408158]
    * [Item].[Item Std1].[REF]
    * [Item].[Product Group].[REF]
    * [Item].[Item].filter(#.Name in {"RR39C7AF5WW/EF","BRR29703EWW/EF","BRZ22700EWW/EF","BRB70F26DES0EF","BRB80F30ADS0EF","RR39C7AF5WW/EF","RS90F64EETEF","RS62DG5003S9EF"} )
    * [Location].[Location]
    * [Time].[Partial Week].filter (#.Key >= &CurrentWeek.element(0).leadoffset(-5).Key && #.Key < &CurrentWeek.element(0).leadoffset(57).Key)
    * [DP Data Seq].[DP Data Seq].filter(#.Name in {"3","4"})
    * [Personnel].[Email].[cmine.jun@samsung.com]
    )
    using arguments {
    (ExecutionMode, "MediumWeight")
    ,(IncludeNullrows, False)
    ,(CurrentWeek, &CurrentWeek.element(0).Name)
    };


* Developer Mode

"""
import sys
import traceback

import pandas as pd
import numpy as np

from NSCMCommon import NSCMCommon

pd.options.mode.copy_on_write = True

########################################################################################################################
# logger 설정
########################################################################################################################
logger = NSCMCommon.G_Logger(p_py_name='PYForecastAR.py')
NSCMCommon.gfn_set_local_logfile()


def fn_PrintDF(p_df: pd.DataFrame, p_df_name: str = ''):
    logger.PrintDF(p_df=p_df, p_df_name=p_df_name, p_row_num=20)


########################################################################################################################
# 시작
########################################################################################################################
logger.Start()
try:
    G_IS_Local = NSCMCommon.gfn_get_isLocal()
    if G_IS_Local is True:
        logger.Note(f'{NSCMCommon.datetime.datetime.now()}')

        CurrentWeek = "202529"
        salesLv = "Sales Std4"

        df_in_h = pd.read_csv('../input/dp_fcst_ar/df_in_h.csv')
        df_in_d = pd.read_csv('../input/dp_fcst_ar/df_in_d.csv')
        df_in_sellin = pd.read_csv('../input/dp_fcst_ar/df_in_sellin.csv')
        df_in_sellout = pd.read_csv('../input/dp_fcst_ar/df_in_sellout.csv')
        df_in_transit_time = pd.read_csv('../input/dp_fcst_ar/df_in_transit_time.csv')

        df_in_split_ratio = pd.read_csv('../input/dp_fcst_ar/df_in_split_ratio.csv')
        df_in_estimated_price_USD = pd.read_csv('../input/dp_fcst_ar/df_in_estimated_price_USD.csv')
        # df_in_exchange_rate = pd.read_csv('./input/dp_fcst_ar/df_in_exchange_rate.csv')
        df_in_chinv = pd.read_csv('../input/dp_fcst_ar/df_in_chinv.csv')
        df_in_item = pd.read_csv('../input/dp_fcst_ar/df_in_item.csv')

        df_in_sales = pd.read_csv('../input/dp_fcst_ar/df_in_sales.csv')
        df_in_version = pd.read_csv('../input/dp_fcst_ar/df_in_version.csv')
        df_in_floor_fcst = pd.read_csv('../input/dp_fcst_ar/df_in_floor_fcst.csv')
        df_in_target_WOS = pd.read_csv('../input/dp_fcst_ar/df_in_target_WOS.csv')

        df_in_gi_bl_intransit_fcst = pd.read_csv('../input/dp_fcst_ar/df_in_gi_bl_intransit_fcst.csv')
        df_in_fcst_gi_assortment = pd.read_csv('../input/dp_fcst_ar/df_in_fcst_gi_assortment.csv')

    logger.Note(f"salesLv : {salesLv}")

    logger.Note(f"df_in_h.shape[0] : {df_in_h.shape[0]}")
    logger.Note(f"df_in_d.shape[0] : {df_in_d.shape[0]}")
    logger.Note(f"df_in_sellin.shape[0] : {df_in_sellin.shape[0]}")
    logger.Note(f"df_in_sellout.shape[0] : {df_in_sellout.shape[0]}")
    logger.Note(f"df_in_transit_time.shape[0] : {df_in_transit_time.shape[0]}")

    logger.Note(f"df_in_split_ratio.shape[0] : {df_in_split_ratio.shape[0]}")
    logger.Note(f"df_in_estimated_price_USD.shape[0] : {df_in_estimated_price_USD.shape[0]}")
    # logger.Note(f"df_in_exchange_rate.shape[0] : {df_in_exchange_rate.shape[0]}")
    logger.Note(f"df_in_chinv.shape[0] : {df_in_chinv.shape[0]}")
    logger.Note(f"df_in_item.shape[0] : {df_in_item.shape[0]}")

    logger.Note(f"df_in_sales.shape[0] : {df_in_sales.shape[0]}")
    logger.Note(f"df_in_version.shape[0] : {df_in_version.shape[0]}")

    logger.Note(f"df_in_floor_fcst.shape[0] : {df_in_floor_fcst.shape[0]}")
    logger.Note(f"df_in_target_WOS.shape[0] : {df_in_target_WOS.shape[0]}")

    logger.Note(f"df_in_gi_bl_intransit_fcst.shape[0] : {df_in_gi_bl_intransit_fcst.shape[0]}")
    logger.Note(f"df_in_fcst_gi_assortment.shape[0] : {df_in_fcst_gi_assortment.shape[0]}")

    ####################################################################################################################
    # 전처리
    ####################################################################################################################
    CWV = df_in_version['Version.[Version Name]'].values[0]

    # w+52 생성 : 당주(w0), w1 ~ w52 그래서 생성은 53주를 생성함
    dict_in_week52 = {'week': 'string', 'key': 'int'}
    df_in_mst_week = NSCMCommon.gfn_get_df_mst_week(p_frist_week=CurrentWeek, p_duration_week=53, p_in_out_week_format='%Y%W')
    df_in_mst_week = df_in_mst_week.astype(dtype=dict_in_week52)

    CurrentWeek_52 = df_in_mst_week['week'].values[-1]

    logger.Note(f"CurrentWeek : {CurrentWeek}")
    logger.Note(f"CurrentWeek_52 : {CurrentWeek_52}")

    # salesLv
    col_salesLv = f"Sales Domain.[{salesLv}]"

    logger.Note(f"col_salesLv : {col_salesLv}")

    #######################################################

    dict_in_transit_time = {'Sales Domain.[Ship To]': 'string'
        , 'Item.[Product Group]': 'string'
        , 'Location.[Location]': 'string'
        , 'DP Master GI BL LT Transit Time W': 'int64'}

    dict_in_item = {'Item.[Item GBM]': 'string'
        , 'Item.[Item Std1]': 'string'
        , 'Item.[Item]': 'string'}

    dict_in_split_ratio = {'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Location.[Location]': 'string'
        , 'Time.[Partial Week]': 'string'
        , 'S/In FCST(GI) Split Ratio_AP2': 'float64'}

    dict_in_chinv = {'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , '(Simul)Ch Inv PW_AP2': 'float64'
        , '(Simul)Ch Inv_Inc Floor PW_AP2': 'float64'
        , '(Simul)P4W Ch Inv PW': 'float64'
        , '(Simul)P4W Ch Inv_Inc Floor PW': 'float64'}

    dict_in_estimated_price_USD = {'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Time.[Partial Week]': 'string'
        , 'Estimated Price_USD': 'float64'}

    dict_in_exchange_rate = {'Sales Domain.[Sales Std3]': 'string'
        , 'Time.[Partial Week]': 'string'
        , 'Exchange Rate_Local': 'float64'
        , 'Exchange Rate_Region': 'float64'}

    dict_in_floor_fcst = {'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Time.[Week]': 'string'
        , 'Flooring FCST': 'float64'}

    dict_in_target_WOS = {'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Time.[Week]': 'string'
        , 'Target Ch WOS': 'float64'}

    dict_in_gi_bl_intransit_fcst = {'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Location.[Location]': 'string'
        , 'Time.[Partial Week]': 'string'
        , 'GI BL Intransit': 'float64'}

    dict_in_fcst_gi_assortment = {'Version.[Version Name]': 'string'
        , 'Item.[Item]': 'string'
        , 'Sales Domain.[Ship To]': 'string'
        , 'Location.[Location]': 'string'
        , 'S/In FCST(GI) Assortment_AP2': 'float64'}

    df_in_transit_time = df_in_transit_time.astype(dtype=dict_in_transit_time)
    df_in_item = df_in_item.astype(dtype=dict_in_item)
    df_in_split_ratio = df_in_split_ratio.astype(dtype=dict_in_split_ratio)
    df_in_chinv = df_in_chinv.astype(dtype=dict_in_chinv)

    df_in_estimated_price_USD = df_in_estimated_price_USD.astype(dtype=dict_in_estimated_price_USD)
    # df_in_exchange_rate = df_in_exchange_rate.astype(dtype=dict_in_exchange_rate)

    df_in_sales = df_in_sales.astype('string')
    df_in_version = df_in_version.astype('string')
    df_in_floor_fcst = df_in_floor_fcst.astype(dtype=dict_in_floor_fcst)
    df_in_target_WOS = df_in_target_WOS.astype(dtype=dict_in_target_WOS)

    # sort
    list_sellin_key = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]']
    df_in_sellin = df_in_sellin.sort_values(by=list_sellin_key).reset_index(drop=True)
    df_in_sellout = df_in_sellout.sort_values(by=list_sellin_key).reset_index(drop=True)

    # 중복제거
    df_in_floor_fcst = df_in_floor_fcst.drop_duplicates().reset_index(drop=True)

    df_in_gi_bl_intransit_fcst = df_in_gi_bl_intransit_fcst.astype(dtype=dict_in_gi_bl_intransit_fcst)
    df_in_fcst_gi_assortment = df_in_fcst_gi_assortment.astype(dtype=dict_in_fcst_gi_assortment)

    # End Step log
    logger.Step(0, '전처리')
    fn_PrintDF(p_df=df_in_h, p_df_name=f'df_in_h ({df_in_h.shape[0]})')
    fn_PrintDF(p_df=df_in_d, p_df_name=f'df_in_d ({df_in_d.shape[0]})')
    fn_PrintDF(p_df=df_in_sellin, p_df_name=f'df_in_sellin ({df_in_sellin.shape[0]})')
    fn_PrintDF(p_df=df_in_sellout, p_df_name=f'df_in_sellout ({df_in_sellout.shape[0]})')
    fn_PrintDF(p_df=df_in_transit_time, p_df_name=f'df_in_transit_time ({df_in_transit_time.shape[0]})')

    fn_PrintDF(p_df=df_in_split_ratio, p_df_name=f'df_in_split_ratio ({df_in_split_ratio.shape[0]})')
    fn_PrintDF(p_df=df_in_estimated_price_USD, p_df_name=f'df_in_estimated_price_USD ({df_in_estimated_price_USD.shape[0]})')
    # fn_PrintDF(p_df=df_in_exchange_rate, p_df_name=f'df_in_exchange_rate ({df_in_exchange_rate.shape[0]})')
    fn_PrintDF(p_df=df_in_chinv, p_df_name=f'df_in_chinv ({df_in_chinv.shape[0]})')
    fn_PrintDF(p_df=df_in_item, p_df_name=f'df_in_item ({df_in_item.shape[0]})')

    fn_PrintDF(p_df=df_in_sales, p_df_name=f'df_in_sales ({df_in_sales.shape[0]})')
    fn_PrintDF(p_df=df_in_version, p_df_name=f'df_in_version ({df_in_version.shape[0]})')
    fn_PrintDF(p_df=df_in_floor_fcst, p_df_name=f'df_in_floor_fcst ({df_in_floor_fcst.shape[0]})')
    fn_PrintDF(p_df=df_in_target_WOS, p_df_name=f'df_in_target_WOS ({df_in_target_WOS.shape[0]})')

    fn_PrintDF(p_df=df_in_gi_bl_intransit_fcst, p_df_name=f'df_in_gi_bl_intransit_fcst ({df_in_gi_bl_intransit_fcst.shape[0]})')
    fn_PrintDF(p_df=df_in_fcst_gi_assortment, p_df_name=f'df_in_fcst_gi_assortment ({df_in_fcst_gi_assortment.shape[0]})')

    ####################################################################################################################
    # Step 1) S/In FCST(GI) , S/Out FCST Delta값 DataFrame 형태로 변환 및 전체데이터와 합치기, Delta 별도 관리 (Delta Flag ? )
    # 	    - df_in_h와 df_in_d 를 이용해 UI에서 입력된 Delta 값 DataFrame 변환
    #       ex) df_in_h : Sales Domain.[Ship To]^Item.[Item]^Location.[Location]
    ####################################################################################################################
    df_in_d = df_in_d.loc[df_in_d["Upload Data_Data"] != '-']

    dict_in_h = {}
    for i, row in df_in_h.iterrows():
        logger.Note(f"df_in_h {row['Upload Header_Dimension']}")
        arr_col = row['Upload Header_Dimension'].split('^')
        dict_in_h[row['Upload Header_Measure']] = arr_col + [row['Upload Header_Measure']]

    df_tmp_detail = df_in_d.merge(df_in_h, how='inner', on=['DP Data Seq.[DP Data Seq]'])

    logger.Note(f'Step 1-1)')
    fn_PrintDF(p_df=df_tmp_detail, p_df_name=f'df_tmp_detail ({len(df_tmp_detail)})')

    ###########################################
    # df_tmp_sellin_fcst_gi : S/In FCST(GI)_AP2
    ###########################################
    where_sellin_fcst_gi = df_tmp_detail['Upload Header_Measure'] == 'S/In FCST(GI)_AP2'
    df_tmp_sellin_fcst_gi = df_tmp_detail.loc[where_sellin_fcst_gi].copy(deep=True)
    df_tmp_sellin_fcst_gi = df_tmp_sellin_fcst_gi.reset_index(drop=True)

    irow = 0
    icol = 0
    arr_all_col = None
    for i, row in df_tmp_sellin_fcst_gi.iterrows():
        str_row = row['Upload Data_Data']
        if 'Version.[Version Name]' in row['Upload Header_Dimension']:
            str_row = f"{CWV}^" + str_row
        arr_row = str_row.split('^')

        arr_col = arr_row[:-1]
        arr_week_value = arr_row[-1].split('|')

        # logger.Note(f"arr_row : {arr_row}")
        # logger.Note(f"arr_week_value : {arr_week_value}")

        irow += len(arr_week_value)
        for j, val in enumerate(arr_week_value):
            if val != '':
                arr_row = arr_col + val.split(':')
                # logger.Note(f"arr_row {j} : {arr_row}")

                if (j == 0) & (i == 0):
                    arr_all_col = arr_row
                    icol = len(arr_row)
                else:
                    arr_all_col = np.concatenate((arr_all_col, arr_row), axis=0)
                # logger.Note(f"arr_all_col {j} : {arr_all_col}")
            else:
                irow = irow - 1

    if arr_all_col is None:
        arr_all_col = []
    else:
        if irow > 1:
            arr_all_col = arr_all_col.reshape(irow, icol)
        else:
            arr_all_col = [arr_all_col]

    logger.Note(f"Sellin arr_all_col.reshape : {arr_all_col}")

    list_sellin = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]', 'S/In FCST(GI)_AP2']
    if 'S/In FCST(GI)_AP2' in dict_in_h:
        list_sellin = dict_in_h['S/In FCST(GI)_AP2']

    df_in_sellin_fcst_gi = pd.DataFrame(arr_all_col, columns=list_sellin)
    df_in_sellin_fcst_gi['S/In FCST(GI)_AP2'] = df_in_sellin_fcst_gi['S/In FCST(GI)_AP2'].astype('int64')

    logger.Note(f'Step 1-2)')
    fn_PrintDF(p_df=df_in_sellin_fcst_gi, p_df_name=f'df_in_sellin_fcst_gi ({df_in_sellin_fcst_gi.shape[0]}) - df_in_h와 df_in_d 를 이용해 UI에서 입력된 Delta 값 DataFrame 변환 ["S/In FCST(GI)_AP2"] 검증 완료')

    #########################################
    # bl값 amt 계산 할려면 org를 추적해야함
    # * user delta 값 -> df_in_sellin_fcst_gi -> df_org_sellin    ->
    #                    df_in_sellin         -> df_in_sellin_agg -> df_tmp_sellin_bl
    #########################################
    df_org_sellin = df_in_sellin_fcst_gi.copy(deep=True)
    df_org_sellin['Time.[Week]'] = df_org_sellin['Time.[Partial Week]'].str[:6]

    list_org_groupby = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Week]']
    df_org_sellin = df_org_sellin.groupby(by=list_org_groupby)['S/In FCST(GI)_AP2'].sum().reset_index()

    logger.Note(f'Step 1-3)')
    fn_PrintDF(p_df=df_org_sellin, p_df_name=f'df_org_sellin ({df_org_sellin.shape[0]})')

    ###########################################
    # df_tmp_sellout_fcst   : S/Out FCST_AP2
    ###########################################
    where_sellout_fcst = df_tmp_detail['Upload Header_Measure'] == 'S/Out FCST_AP2'
    df_tmp_sellout_fcst = df_tmp_detail.loc[where_sellout_fcst].copy(deep=True)
    df_tmp_sellout_fcst = df_tmp_sellout_fcst.reset_index(drop=True)

    irow = 0
    icol = 0
    arr_all_col = None
    for i, row in df_tmp_sellout_fcst.iterrows():
        str_row = row['Upload Data_Data']
        if 'Version.[Version Name]' in row['Upload Header_Dimension']:
            str_row = f"{CWV}^" + str_row
        arr_row = str_row.split('^')

        arr_col = arr_row[:-1]
        arr_week_value = arr_row[-1].split('|')

        irow += len(arr_week_value)
        for j, val in enumerate(arr_week_value):
            if val != '':
                arr_row = arr_col + val.split(':')
                if (j == 0) & (i == 0):
                    arr_all_col = arr_row
                    icol = len(arr_row)
                else:
                    arr_all_col = np.concatenate((arr_all_col, arr_row), axis=0)
            else:
                irow = irow - 1

    if arr_all_col is None:
        arr_all_col = []
    else:
        if irow > 1:
            arr_all_col = arr_all_col.reshape(irow, icol)
        else:
            arr_all_col = [arr_all_col]

    logger.Note(f"Selout arr_all_col.reshape : {arr_all_col}")

    list_sellout = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]', 'S/Out FCST_AP2']
    if 'S/Out FCST_AP2' in dict_in_h:
        list_sellout = dict_in_h['S/Out FCST_AP2']

    df_in_sellout_fcst = pd.DataFrame(arr_all_col, columns=list_sellout)
    df_in_sellout_fcst['S/Out FCST_AP2'] = df_in_sellout_fcst['S/Out FCST_AP2'].astype('int64')

    # add 20251016
    df_in_sellout_fcst['S/Out FCST Modify_AP2'] = 1
    df_in_sellout_fcst['S/Out FCST Modify_AP2'] = df_in_sellout_fcst['S/Out FCST Modify_AP2'].astype('int64')

    logger.Note(f'Step 1-4)')
    fn_PrintDF(p_df=df_in_sellout_fcst, p_df_name=f'df_in_sellout_fcst ({df_in_sellout_fcst.shape[0]}) - df_in_h와 df_in_d 를 이용해 UI에서 입력된 Delta 값 DataFrame 변환 ["S/Out FCST_AP2"] 검증 완료')

    ###########################################
    # df_in_sellin : S/In FCST(GI) 와 합치기
    ###########################################
    list_sellin_key = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]']
    df_in_sellin[list_sellin_key] = df_in_sellin[list_sellin_key].astype('string')

    df_in_sellin = df_in_sellin.merge(df_in_sellin_fcst_gi, how='left', on=list_sellin_key, suffixes=('', '_y'))

    df_in_sellin['S/In FCST(GI)_AP2_y'] = df_in_sellin['S/In FCST(GI)_AP2_y'].fillna(value=0)
    df_in_sellin['S/In FCST(GI)_AP2_y'] = df_in_sellin['S/In FCST(GI)_AP2_y'].astype('int64')

    df_in_sellin['S/In FCST(GI)_AP2'] = np.where(df_in_sellin['S/In FCST(GI)_AP2_y'].notna()
                                                 , df_in_sellin['S/In FCST(GI)_AP2_y']
                                                 , df_in_sellin['S/In FCST(GI)_AP2'])

    df_in_sellin = df_in_sellin.drop('S/In FCST(GI)_AP2_y', axis=1)

    logger.Note(f'Step 1-5)')
    fn_PrintDF(p_df=df_in_sellin, p_df_name=f'df_in_sellin ({df_in_sellin.shape[0]})')

    ###########################################
    # df_in_sellout : S/Out FCST 와 합치기
    ###########################################
    list_sellout_key = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]']
    df_in_sellout[list_sellout_key] = df_in_sellout[list_sellout_key].astype('string')

    df_in_sellout = df_in_sellout.merge(df_in_sellout_fcst, how='left', on=list_sellout_key, suffixes=('', '_y'))

    df_in_sellout['S/Out FCST_AP2'] = df_in_sellout['S/Out FCST_AP2'].fillna(value=0)
    df_in_sellout['S/Out FCST_AP2'] = df_in_sellout['S/Out FCST_AP2'].astype('int64')

    df_in_sellout['S/Out FCST_AP2'] = np.where(df_in_sellout['S/Out FCST_AP2_y'].notna()
                                               , df_in_sellout['S/Out FCST_AP2_y']
                                               , df_in_sellout['S/Out FCST_AP2'])

    # End Step log
    logger.Step(1, 'S/In Out FCST data 정비')
    fn_PrintDF(p_df=df_in_sellin, p_df_name=f'df_in_sellin({df_in_sellin.shape[0]}) - df_in_sellin S/In FCST(GI) 와 합치기 검증 완료')
    fn_PrintDF(p_df=df_in_sellout, p_df_name=f'df_in_sellout({df_in_sellout.shape[0]}) - df_in_sellout 이용해서 S/Out FCST 와 합치기 검증 완료')

    ####################################################################################################################
    # Step 2) S/Out FCST Data 가공
    # 	    - S/Out FCST_AP2 Aggregation (Ship To, Item, Week)
    ####################################################################################################################
    df_in_sellout_agg = df_in_sellout.copy(deep=True)
    df_in_sellout_agg['Time.[Week]'] = df_in_sellout_agg['Time.[Partial Week]'].str[:6]

    list_sellout_groupby = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']
    df_in_sellout_agg = df_in_sellout_agg.groupby(by=list_sellout_groupby)['S/Out FCST_AP2'].sum().reset_index()

    logger.Note(f'Step 2-1)')
    fn_PrintDF(p_df=df_in_sellout_agg, p_df_name=f'df_in_sellout_agg ({df_in_sellout_agg.shape[0]})')

    ##############################################################################
    # - S/Out Actual(P4W) 값 생성 : df_in_sellout 에서 S/Out FCST 값 가져와서 P4W 생성
    ##############################################################################
    df_in_sellout_agg_p4w = df_in_sellout_agg.merge(df_in_item, how='inner', on=['Item.[Item]'])

    # list_mx_week
    list_mx_week = df_in_sellout_agg.loc[df_in_sellout_agg['Time.[Week]'] < CurrentWeek, 'Time.[Week]'].unique().tolist()
    list_mx_week = list_mx_week[-4:]
    df_mx_week = pd.DataFrame(list_mx_week, columns=['Time.[Week]'])

    logger.Note(f'Step 2-2)')
    fn_PrintDF(p_df=df_mx_week, p_df_name=f'df_mx_week ({df_mx_week.shape[0]})')

    # list_ce_week
    list_ce_week = df_in_sellout_agg.loc[df_in_sellout_agg['Time.[Week]'] < CurrentWeek, 'Time.[Week]'].unique().tolist()
    list_ce_week = list_ce_week[-5:-1]
    df_ce_week = pd.DataFrame(list_ce_week, columns=['Time.[Week]'])

    logger.Note(f'Step 2-3)')
    fn_PrintDF(p_df=df_ce_week, p_df_name=f'df_ce_week ({df_ce_week.shape[0]})')

    ##################################################
    # -  Item GBM 이 MOBILE(MX) 이면 W-4 ~ W-1 의 AVG 값
    ##################################################
    where_mx = df_in_sellout_agg_p4w['Item.[Item GBM]'] == 'MOBILE'
    df_tmp_sellout_agg_p4w_mx = df_in_sellout_agg_p4w.loc[where_mx].copy(deep=True)

    df_tmp_sellout_agg_p4w_mx = df_tmp_sellout_agg_p4w_mx.merge(df_mx_week, how='inner', on=['Time.[Week]'])

    list_mx_groupby = ['Sales Domain.[Ship To]', 'Item.[Item]']
    df_tmp_sellout_agg_p4w_mx = df_tmp_sellout_agg_p4w_mx.groupby(by=list_mx_groupby)['S/Out FCST_AP2'].mean().reset_index()

    logger.Note(f'Step 2-4)')
    fn_PrintDF(p_df=df_tmp_sellout_agg_p4w_mx, p_df_name=f'df_tmp_sellout_agg_p4w_mx ({df_tmp_sellout_agg_p4w_mx.shape[0]})')

    ##################################################
    # -  Item GBM 이 CE(VD,SHA) 이면 W-5 ~ W-2 의 AVG 값
    ##################################################
    where_ce = (df_in_sellout_agg_p4w['Item.[Item GBM]'] == 'SHA') | (df_in_sellout_agg_p4w['Item.[Item GBM]'] == 'VD')
    df_tmp_sellout_agg_p4w_ce = df_in_sellout_agg_p4w.loc[where_ce].copy(deep=True)

    df_tmp_sellout_agg_p4w_ce = df_tmp_sellout_agg_p4w_ce.merge(df_ce_week, how='inner', on=['Time.[Week]'])

    list_ce_groupby = ['Sales Domain.[Ship To]', 'Item.[Item]']
    df_tmp_sellout_agg_p4w_ce = df_tmp_sellout_agg_p4w_ce.groupby(by=list_ce_groupby)['S/Out FCST_AP2'].mean().reset_index()

    logger.Note(f'Step 2-5)')
    fn_PrintDF(p_df=df_tmp_sellout_agg_p4w_ce, p_df_name=f'df_tmp_sellout_agg_p4w_ce ({df_tmp_sellout_agg_p4w_ce.shape[0]})')

    ##################################################
    # p4w 결과
    ##################################################
    df_out_sellout_agg_p4w = pd.DataFrame(columns=['Sales Domain.[Ship To]', 'Item.[Item]', 'S/Out FCST_AP2'])
    if (df_tmp_sellout_agg_p4w_mx.shape[0] > 0) & (df_tmp_sellout_agg_p4w_ce.shape[0] > 0):
        df_out_sellout_agg_p4w = pd.concat(df_tmp_sellout_agg_p4w_mx, df_tmp_sellout_agg_p4w_ce)
    elif df_tmp_sellout_agg_p4w_mx.shape[0] > 0:
        df_out_sellout_agg_p4w = df_tmp_sellout_agg_p4w_mx.copy(deep=True)
    elif df_tmp_sellout_agg_p4w_ce.shape[0] > 0:
        df_out_sellout_agg_p4w = df_tmp_sellout_agg_p4w_ce.copy(deep=True)

    df_out_sellout_agg_p4w.rename(columns={'S/Out FCST_AP2': 'S/Out Actual(P4W)'}, inplace=True)

    logger.Note(f'Step 2-6)')
    fn_PrintDF(p_df=df_out_sellout_agg_p4w, p_df_name=f'df_out_sellout_agg_p4w ({df_out_sellout_agg_p4w.shape[0]})')

    ########################################################################################
    # - (Week Avg)S/Out FCST F4W_AP2 값 생성 : df_in_sellout 에서 S/Out FCST 값 가져와서 F4W 생성
    # - 당주주차(202537) ~ Time 조회 - 4 (202542)에 대해서만 생성
    #   W + 52 까지생성필요
    ########################################################################################
    # W + 52 까지생성필요
    df_in_sellout_agg_f4w = df_in_sellout_agg.copy(deep=True)

    # - all week = 기존 week + w52
    df_tmp_week1 = df_in_sellout_agg_f4w[['Time.[Week]']].drop_duplicates()
    df_tmp_week2 = df_in_mst_week[['week']]
    df_tmp_week2.rename(columns={'week': 'Time.[Week]'}, inplace=True)

    df_mst_f4w_week = pd.concat([df_tmp_week1, df_tmp_week2])
    df_mst_f4w_week = df_mst_f4w_week.drop_duplicates().reset_index(drop=True)

    # - week 제외한 mst 생성
    df_mst_f4w = df_in_sellout_agg_f4w[['Sales Domain.[Ship To]', 'Item.[Item]']].copy(deep=True).drop_duplicates()

    # - cross join base 생성
    df_base_f4w = df_mst_f4w.merge(df_mst_f4w_week, how='cross')

    # - base 적용
    df_in_sellout_agg_f4w = df_base_f4w.merge(df_in_sellout_agg_f4w, how='left', on=['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]'])

    # 미래 4week avg 생성후 뒤로 shift
    df_in_sellout_agg_f4w['f4w'] = df_in_sellout_agg_f4w['S/Out FCST_AP2'].rolling(window=4).mean()
    df_in_sellout_agg_f4w['f4w'] = df_in_sellout_agg_f4w['f4w'].shift(periods=-4)

    # W + 52 까지 생성 완료
    where_f4w = (df_in_sellout_agg_f4w['Time.[Week]'] >= CurrentWeek) & (df_in_sellout_agg_f4w['Time.[Week]'] <= CurrentWeek_52)
    df_in_sellout_agg_f4w = df_in_sellout_agg_f4w.loc[where_f4w]

    logger.Note(f'Step 2-7)')
    fn_PrintDF(p_df=df_in_sellout_agg_f4w, p_df_name=f'df_in_sellout_agg_f4w ({df_in_sellout_agg_f4w.shape[0]})')

    ##################################################
    # f4w 결과
    ##################################################
    df_out_sellout_agg_f4w = df_in_sellout_agg_f4w.copy(deep=True)
    df_out_sellout_agg_f4w['S/Out FCST_AP2'] = df_out_sellout_agg_f4w['f4w']

    list_sellout_f4w = list_sellout_groupby + ['S/Out FCST_AP2']
    df_out_sellout_agg_f4w = df_out_sellout_agg_f4w[list_sellout_f4w]

    # (Week Avg)S/Out FCST F4W_AP2
    df_out_sellout_agg_f4w.rename(columns={'S/Out FCST_AP2': '(Week Avg)S/Out FCST F4W_AP2'}, inplace=True)

    # End Step log
    logger.Step(2, 'S/Out FCST data 가공')
    fn_PrintDF(p_df=df_in_sellout_agg, p_df_name=f'df_in_sellout_agg ({df_in_sellout_agg.shape[0]})')
    fn_PrintDF(p_df=df_out_sellout_agg_p4w, p_df_name=f'df_out_sellout_agg_p4w ({df_out_sellout_agg_p4w.shape[0]}) - S/Out FCST_AP2 Aggregation (Ship To, Item, Week) 검증 완료')
    fn_PrintDF(p_df=df_out_sellout_agg_f4w, p_df_name=f'df_out_sellout_agg_f4w ({df_out_sellout_agg_f4w.shape[0]}) - S/Out FCST_AP2 Aggregation (Ship To, Item, Week) 검증 완료')

    ####################################################################################################################
    # Step 3) S/In FCST(BL) 값 연산
    # 	    - S/In FCST 값을 Week 단위로 Aggregation
    ####################################################################################################################
    df_in_sellin_agg = df_in_sellin.copy(deep=True)
    df_in_sellin_agg['Time.[Week]'] = df_in_sellin_agg['Time.[Partial Week]'].str[:6]

    list_sellin_groupby = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Week]']
    list_sellin_sum = ['S/In FCST(GI)_AP2', 'S/In FCST(BL)_AP2']
    if not df_in_sellin_agg.empty:
        df_in_sellin_agg = df_in_sellin_agg.groupby(by=list_sellin_groupby)[list_sellin_sum].sum().reset_index()

    logger.Note(f'Step 3-1)')
    fn_PrintDF(p_df=df_in_sellin_agg, p_df_name=f'df_in_sellin_agg ({df_in_sellin_agg.shape[0]}) - S/In FCST 값을 Week 단위로 Aggregation 검증 완료')

    ##################################################################################
    # - df_in_transit_time과 df_in_item 을 이용해 Ship To * Product Group * Location 별로
    #   DP Master GI BL LT Transit Time W 값만큼 이동하여 S/In FCST(BL) 값 연산
    ##################################################################################
    df_in_transit_time = df_in_transit_time.merge(df_in_item, how='inner'
                                                  , left_on=['Item.[Product Group]']
                                                  , right_on=['Item.[Item Std1]'])

    list_transit = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'DP Master GI BL LT Transit Time W']
    df_in_transit_time = df_in_transit_time[list_transit]
    df_in_transit_time['DP Master GI BL LT Transit Time W'] = df_in_transit_time['DP Master GI BL LT Transit Time W'].fillna(0)

    list_transit_sort = ['DP Master GI BL LT Transit Time W', 'Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]']
    df_in_transit_time = df_in_transit_time.sort_values(by=list_transit_sort).reset_index(drop=True)

    logger.Note(f'Step 3-2)')
    fn_PrintDF(p_df=df_in_transit_time, p_df_name=f'df_in_transit_time ({df_in_transit_time.shape[0]})')

    ##################################################################
    # bl w52 까지 생성
    ##################################################################
    df_tmp_sellin_bl = df_in_sellin_agg.copy(deep=True)

    # - all week = 기존 week + w52
    df_tmp_week1 = df_tmp_sellin_bl[['Time.[Week]']].drop_duplicates()
    df_tmp_week2 = df_in_mst_week[['week']]
    df_tmp_week2.rename(columns={'week': 'Time.[Week]'}, inplace=True)

    df_mst_bl_week = pd.concat([df_tmp_week1, df_tmp_week2])
    df_mst_bl_week = df_mst_bl_week.drop_duplicates().reset_index(drop=True)

    # - week 제외한 mst 생성
    df_mst_bl = df_tmp_sellin_bl[['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]']].copy(deep=True).drop_duplicates()

    # - cross join base 생성
    df_base_bl = df_mst_bl.merge(df_mst_bl_week, how='cross')

    # - base 적용
    df_tmp_sellin_bl = df_base_bl.merge(df_tmp_sellin_bl, how='left', on=['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Week]'])
    ##################################################################
    # bl w52 까지 생성 End
    ##################################################################
    # bl 생성
    list_bl_tmp = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]']
    df_tmp_sellin_bl = df_tmp_sellin_bl.merge(df_in_transit_time, how='left', on=list_bl_tmp)
    df_tmp_sellin_bl['DP Master GI BL LT Transit Time W'] = df_tmp_sellin_bl['DP Master GI BL LT Transit Time W'].fillna(0)

    # sort
    list_sort = ['DP Master GI BL LT Transit Time W', 'Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Week]']
    df_tmp_sellin_bl = df_tmp_sellin_bl.sort_values(by=list_sort).reset_index(drop=True)

    # 'S/In FCST(GI)_AP2' -> 'S/In FCST(BL)_AP2'
    df_tmp_sellin_bl['S/In FCST(BL)_AP2'] = df_tmp_sellin_bl['S/In FCST(GI)_AP2']

    # add cumcount
    list_bl = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]']

    df_tmp_sellin_bl['cnt'] = 1
    df_tmp_sellin_bl['cum_count'] = df_tmp_sellin_bl.groupby(by=list_bl)['Time.[Week]'].transform('cumcount')
    df_tmp_sellin_bl['cum_count'] = df_tmp_sellin_bl['cum_count']

    df_tmp_sellin_bl['cum_max'] = df_tmp_sellin_bl.groupby(by=list_bl)['cnt'].transform('sum')

    df_tmp_sellin_bl['str_nan'] = np.nan

    logger.Note(f'Step 3-3)')
    fn_PrintDF(p_df=df_tmp_sellin_bl, p_df_name=f'df_tmp_sellin_bl ({df_tmp_sellin_bl.shape[0]})')
    fn_PrintDF(p_df=df_org_sellin, p_df_name=f'df_org_sellin ({df_org_sellin.shape[0]})')

    _df_org_sellin_check = df_tmp_sellin_bl.loc[df_tmp_sellin_bl['Time.[Week]'] == '202548'].copy(deep=True)
    fn_PrintDF(p_df=_df_org_sellin_check, p_df_name=f'_df_org_sellin_check ({_df_org_sellin_check.shape[0]})')

    ##################################
    # bl amt를 위한 org 추적
    ##################################
    df_org_sellin['org_value'] = 'org'

    list_bl_tmp = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Week]']
    df_tmp_sellin_bl = df_tmp_sellin_bl.merge(df_org_sellin, how='left', on=list_bl_tmp, suffixes=('', '_y'))

    # 마지막 week bl 값 set null
    list_transit_w = df_in_transit_time['DP Master GI BL LT Transit Time W'].unique().tolist()
    iweek = 1
    for i, iweek in enumerate(list_transit_w):
        where_transit_w = ((df_tmp_sellin_bl['cum_count'] >= df_tmp_sellin_bl['cum_max'] - iweek) &
                           (df_tmp_sellin_bl['cum_count'] < df_tmp_sellin_bl['cum_max'])
                           )
        where_transit_w2 = (df_tmp_sellin_bl['DP Master GI BL LT Transit Time W'] == iweek)
        df_tmp_sellin_bl.loc[where_transit_w & where_transit_w2, 'S/In FCST(BL)_AP2'] = np.nan

    # shift
    # list_transit_w = df_in_transit_time['DP Master GI BL LT Transit Time W'].unique().tolist()
    for i, iweek in enumerate(list_transit_w):
        where_transit_w = (df_tmp_sellin_bl['DP Master GI BL LT Transit Time W'] == iweek)
        df_tmp_sellin_bl.loc[where_transit_w, 'S/In FCST(BL)_AP2'] = df_tmp_sellin_bl.loc[where_transit_w].groupby(by=list_bl)['S/In FCST(BL)_AP2'].shift(periods=iweek)
        df_tmp_sellin_bl.loc[where_transit_w, 'org_value'] = df_tmp_sellin_bl.loc[where_transit_w].groupby(by=list_bl)['org_value'].shift(periods=iweek)

    df_tmp_sellin_bl['S/In FCST(BL)_AP2'] = np.where(df_tmp_sellin_bl['S/In FCST(BL)_AP2'].notna()
                                                     , df_tmp_sellin_bl['S/In FCST(BL)_AP2']
                                                     , df_tmp_sellin_bl['S/In FCST(GI)_AP2'])

    list_tmp_bl = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Week]'
        , 'S/In FCST(BL)_AP2', 'DP Master GI BL LT Transit Time W', 'org_value']
    df_tmp_sellin_bl = df_tmp_sellin_bl[list_tmp_bl]

    logger.Note(f'Step 3-4)')
    fn_PrintDF(p_df=df_tmp_sellin_bl,
               p_df_name=f'df_tmp_sellin_bl ({df_tmp_sellin_bl.shape[0]}) - df_in_transit_time과 df_in_item 을 이용해 Ship To * Product Group * Location 별로 DP Master GI BL LT Transit Time W 값만큼 이동하여 S/In FCST(BL) 값 연산 검증 완료')

    _df_org_check = df_tmp_sellin_bl.loc[df_tmp_sellin_bl['org_value'] == 'org'].copy(deep=True)
    fn_PrintDF(p_df=_df_org_check,
               p_df_name=f'_df_org_check ({_df_org_check.shape[0]}) - df_tmp_sellin_bl org_value 체크')

    ##################################################################################
    # - df_in_Split_Ratio 를 이용해 S/In FCST(BL) 값 Week -> Partial week 연산
    ##################################################################################
    # df_in_split_ratio['Time.[Week]'] = df_in_split_ratio['Time.[Partial Week]'].str[:6]

    ##################################################################################
    # 이빨 빠지는것 보정
    ##################################################################################
    df_ab = pd.DataFrame(data=[0, 6], columns=['p_day_delta'])

    df_partial_week = df_tmp_sellin_bl[['Time.[Week]']].copy(deep=True)
    df_partial_week = df_partial_week.drop_duplicates()

    df_partial_week = df_partial_week.merge(df_ab, how='cross')

    df_partial_week['datetime'] = np.nan
    df_partial_week['Time.[Partial Week]'] = np.nan
    if df_partial_week.shape[0] > 0:
        # week -> datetime
        df_partial_week['datetime'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_to_date(p_str_datetype=x['Time.[Week]'], p_format='%Y%W',
                                                                                             p_week_day=1, p_day_delta=x['p_day_delta']), axis=1)
        # datetime -> partial week
        df_partial_week['Time.[Partial Week]'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_get_partial_week(p_datetime=x['datetime'],
                                                                                                                 p_bool_FI_week=True), axis=1)
    df_partial_week = df_partial_week.drop(['p_day_delta', 'datetime'], axis=1)
    df_partial_week = df_partial_week.drop_duplicates().reset_index(drop=True)
    df_partial_week = df_partial_week.astype('string')

    ##############################################
    # add ['Partial Week']
    df_tmp_sellin_bl = df_tmp_sellin_bl.merge(df_partial_week, how='inner', on=['Time.[Week]'])

    list_bl = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Week]']
    df_tmp_sellin_bl = df_tmp_sellin_bl.sort_values(by=list_bl).reset_index(drop=True)

    df_tmp_sellin_bl['cnt'] = 1
    df_tmp_sellin_bl['cum_max'] = df_tmp_sellin_bl.groupby(by=list_bl)['cnt'].transform('sum')

    logger.Note(f'Step 3-5)')
    fn_PrintDF(p_df=df_tmp_sellin_bl, p_df_name=f'df_tmp_sellin_bl')
    ##############################################

    list_bl_key = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]']
    df_out_sellin_bl = df_tmp_sellin_bl.merge(df_in_split_ratio, how='left', on=list_bl_key)

    # 비율 default 0
    df_out_sellin_bl['S/In FCST(GI) Split Ratio_AP2'] = df_out_sellin_bl['S/In FCST(GI) Split Ratio_AP2'].fillna(value=0)

    df_out_sellin_bl['ratio_sum'] = df_out_sellin_bl.groupby(by=list_bl)['S/In FCST(GI) Split Ratio_AP2'].transform('sum')

    # ratio 가 0 or nan 이면 1:1로 분배 될수 있게 1로 설정한다.
    df_out_sellin_bl['S/In FCST(GI) Split Ratio_AP2'] = np.where((df_out_sellin_bl['S/In FCST(GI) Split Ratio_AP2'] == 0) &
                                                                 (df_out_sellin_bl['cum_max'] == 1)
                                                                 , 1
                                                                 , df_out_sellin_bl['S/In FCST(GI) Split Ratio_AP2'])

    # -> 2. S/In FCST(GI) Split Ratio_AP2 0이거나 Null인 경우 -> A/B 주차에 값이 하나라도 있으면 0, 둘다 0 or Null 이면 -> 1
    df_out_sellin_bl['S/In FCST(GI) Split Ratio_AP2'] = np.where((df_out_sellin_bl['S/In FCST(GI) Split Ratio_AP2'] == 0) &
                                                                 (df_out_sellin_bl['cum_max'] == 2) & (df_out_sellin_bl['ratio_sum'] == 0.0)
                                                                 , 1
                                                                 , df_out_sellin_bl['S/In FCST(GI) Split Ratio_AP2'])

    logger.Note(f'Step 3-6)')
    fn_PrintDF(p_df=df_out_sellin_bl, p_df_name=f'df_out_sellin_bl ({df_out_sellin_bl.shape[0]}) - df_in_Split_Ratio 를 이용해 S/In FCST(BL) 값 Week -> Partial week 연산 검증 완료')

    ##################################################
    # bl amt를 위한 org_value 추적 df_out_sellin_bl 까지
    # df_org_bl = df_out_sellin_bl.
    ##################################################

    ##################################################################################
    # - 소수점 처리 로직(소수점처리 예시 Sheet 참조)
    ##################################################################################
    list_bl_group = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Week]']

    df_out_sellin_bl['cnt'] = 1
    df_out_sellin_bl['cum_count'] = df_out_sellin_bl.groupby(by=list_bl_group)['Time.[Partial Week]'].transform('cumcount')
    df_out_sellin_bl['cum_max'] = df_out_sellin_bl.groupby(by=list_bl_group)['cnt'].transform('sum')

    df_out_sellin_bl['sum_ratio'] = df_out_sellin_bl.groupby(by=list_bl_group)['S/In FCST(GI) Split Ratio_AP2'].transform('sum')

    # partial week 1W
    where_first = (df_out_sellin_bl['cum_max'] == 2) & (df_out_sellin_bl['cum_count'] == 0)
    df_out_sellin_bl.loc[where_first, 'S/In FCST(BL)_AP2_calc'] = df_out_sellin_bl.loc[where_first, 'S/In FCST(BL)_AP2'] * (
            df_out_sellin_bl.loc[where_first, 'S/In FCST(GI) Split Ratio_AP2'] / df_out_sellin_bl.loc[where_first, 'sum_ratio'])

    df_out_sellin_bl.loc[where_first, 'S/In FCST(BL)_AP2_calc'] = np.round(df_out_sellin_bl.loc[where_first, 'S/In FCST(BL)_AP2_calc'])

    df_out_sellin_bl.loc[where_first, 'S/In FCST(BL)_AP2'] = df_out_sellin_bl.loc[where_first, 'S/In FCST(BL)_AP2_calc']

    logger.Note(f'Step 3-7)')
    fn_PrintDF(p_df=df_out_sellin_bl, p_df_name=f'df_out_sellin_bl ({df_out_sellin_bl.shape[0]})')

    df_out_sellin_bl['S/In FCST(BL)_AP2_calc'] = df_out_sellin_bl['S/In FCST(BL)_AP2_calc'].shift(periods=1)

    logger.Note(f'Step 3-8)')
    fn_PrintDF(p_df=df_out_sellin_bl, p_df_name=f'df_out_sellin_bl ({df_out_sellin_bl.shape[0]})')

    # partial week 2W
    where_second = (df_out_sellin_bl['cum_max'] == 2) & (df_out_sellin_bl['cum_count'] == 1)
    df_out_sellin_bl.loc[where_second, 'S/In FCST(BL)_AP2_calc'] = df_out_sellin_bl.loc[where_second, 'S/In FCST(BL)_AP2'] - df_out_sellin_bl.loc[where_second, 'S/In FCST(BL)_AP2_calc']
    df_out_sellin_bl.loc[where_second, 'S/In FCST(BL)_AP2'] = df_out_sellin_bl.loc[where_second, 'S/In FCST(BL)_AP2_calc']

    logger.Note(f'Step 3-9)')
    fn_PrintDF(p_df=df_out_sellin_bl, p_df_name=f'df_out_sellin_bl ({df_out_sellin_bl.shape[0]})')

    ####################################################################################################################
    # - (25.11.20) GI BL Intransit 값 반영
    # - 조건  1)  S/In FCST(GI)_AP2 의 당주주차의 값이 수정되었을 떄 -> 제외
    # - 조건  2) DP Master GI BL LT Transit Time W == 0 일 때
    # - 조건 1&2 인 경우, 당주주차에 대해서 S/In FCST(BL)_AP2 = S/In FCST(GI)_AP2 + GI BL Intransit 로직을 적용한다.
    # - GI BL Intransit 의 값은 Ship To * Item * Location * Partial Week 가 일치하는 값을 사용한다.
    # * ex) DP Master GI BL LT Transit Time W = 0 이고, 사용자가 GI 202537 주차를 수정했을 때, 당주주차에 대한 연산 시, GI BL Intransit 값을 더해서 BL을 산출한다.
    ####################################################################################################################
    # - 조건  2) DP Master GI BL LT Transit Time W == 0 일 때
    list_gi_bl_intransit = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]']
    df_out_sellin_bl = df_out_sellin_bl.merge(df_in_gi_bl_intransit_fcst, how='left', on=list_gi_bl_intransit)
    df_out_sellin_bl['GI BL Intransit'] = df_out_sellin_bl['GI BL Intransit'].fillna(0).astype('float64')

    df_out_sellin_bl["S/In FCST(BL)_AP2"] = np.where((df_out_sellin_bl['DP Master GI BL LT Transit Time W'] == 0)
                                                     & (df_out_sellin_bl['Time.[Partial Week]'].str.startswith(CurrentWeek))
                                                     , df_out_sellin_bl["S/In FCST(BL)_AP2"] + df_out_sellin_bl["GI BL Intransit"]
                                                     , df_out_sellin_bl["S/In FCST(BL)_AP2"])

    # 컬럼 정리
    list_all_bl = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]', 'S/In FCST(BL)_AP2', 'org_value']
    df_out_sellin_bl = df_out_sellin_bl[list_all_bl]

    # End Step log
    logger.Step(3, 'S/In FCST(BL) 값 연산')
    fn_PrintDF(p_df=df_out_sellin_bl, p_df_name=f'df_out_sellin_bl ({df_out_sellin_bl.shape[0]})')

    ####################################################################################################################
    # Step 4) Ch Inv 값 연산
    # 	    - S/In FCST(BL)_AP2 Aggregation (Ship To, Item, Week)
    ####################################################################################################################
    list_bl_agg = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']
    df_tmp_sellin_bl_agg = df_tmp_sellin_bl.groupby(by=list_bl_agg)['S/In FCST(BL)_AP2'].sum().reset_index()

    if not df_tmp_sellin_bl_agg.empty:
        df_tmp_sellin_bl_agg = df_tmp_sellin_bl_agg.merge(df_in_sellout_agg, how='left', on=list_bl_agg)
    else:
        df_tmp_sellin_bl_agg = df_in_sellout_agg.copy(deep=True)
        df_tmp_sellin_bl_agg['S/In FCST(BL)_AP2'] = np.nan
        df_tmp_sellin_bl_agg['S/In FCST(BL)_AP2'] = df_tmp_sellin_bl_agg['S/In FCST(BL)_AP2'].astype('float64')

    list_bl_chinv = ['Sales Domain.[Ship To]', 'Item.[Item]']
    df_tmp_sellin_bl_agg = df_tmp_sellin_bl_agg.merge(df_in_chinv, how='left', on=list_bl_chinv)

    logger.Note(f'Step 4-1)')
    fn_PrintDF(p_df=df_tmp_sellin_bl_agg, p_df_name=f'df_tmp_sellin_bl_agg ({df_tmp_sellin_bl_agg.shape[0]})')

    ##################################################################################
    # - (Simul)Ch Inv PW_AP2, (Simul)Ch Inv_Inc floor_AP2 : Measure.[(Simul)Ch Inv PW_AP2] =
    # (25.09.25) (Simul)Ch Inv PW_AP2 : Measure.[(Simul)Ch Inv PW_AP2] = Measure.[(Simul)Ch Inv PW_AP2]@(Time.#.leadoffset(-1)) + Measure.[(Copy)S/In FCST(BL)_AP2] - Measure.[(Copy)S/Out FCST_AP2]
    # (25.09.25) (Simul)Ch Inv_Inc Floor PW_AP2 : Measure.[(Simul)Ch Inv_Inc Floor PW_AP2] = Measure.[(Simul)Ch Inv_Inc Floor PW_AP2]@(Time.#.leadoffset(-1)) + Measure.[(Copy)S/In FCST(BL)_AP2] - Measure.[(Copy)S/Out FCST_AP2] + Measure.[Flooring FCST]
    ##################################################################################
    list_simul_ch = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]', 'S/In FCST(BL)_AP2', 'S/Out FCST_AP2',
                     '(Simul)Ch Inv PW_AP2', '(Simul)Ch Inv_Inc Floor PW_AP2']
    df_simul_sellin_bl_agg = df_tmp_sellin_bl_agg[list_simul_ch].copy(deep=True)

    list_simul_week = df_simul_sellin_bl_agg.loc[df_simul_sellin_bl_agg['Time.[Week]'] < CurrentWeek, 'Time.[Week]'].unique().tolist()
    if len(list_simul_week) > 1:
        list_simul_week = list_simul_week[-1]

    if len(list_simul_week) > 1:
        where_simul_nan = df_simul_sellin_bl_agg['Time.[Week]'] < list_simul_week
        df_simul_sellin_bl_agg.loc[where_simul_nan, '(Simul)Ch Inv PW_AP2'] = np.nan
        df_simul_sellin_bl_agg.loc[where_simul_nan, '(Simul)Ch Inv_Inc Floor PW_AP2'] = np.nan

    #################################################################################
    # - 25.11.15 Flooring FCST Lv에 맞는 Data로 Aggregation
    #   df_in_floor_fcst 와 df_in_sales 를 활용해서 salesLv에 맞는 형태로 Aggregation 진행
    #################################################################################
    list_sales_floor_fcst = ['Sales Domain.[Ship To]']
    df_sales_floor_fcst = df_in_sales.merge(df_in_floor_fcst, how='left', on=list_sales_floor_fcst)
    df_sales_floor_fcst['Flooring FCST'] = df_sales_floor_fcst['Flooring FCST'].fillna(0).astype('int64')

    # - Sales Std4 가 들어왔기 때문에 Sales Std4, Item, Week 로 Aggregation 진행 (Sum)
    list_sales_floor_columns = [col_salesLv, 'Item.[Item]', 'Time.[Week]', 'Flooring FCST']
    list_sales_floor_groupby = [col_salesLv, 'Item.[Item]', 'Time.[Week]']

    df_sales_floor_fcst_agg = pd.DataFrame(columns=list_sales_floor_columns)
    if not df_sales_floor_fcst.empty:
        df_sales_floor_fcst_agg = df_sales_floor_fcst.groupby(by=list_sales_floor_groupby)['Flooring FCST'].sum().reset_index()

    # rename
    df_sales_floor_fcst_agg = df_sales_floor_fcst_agg.rename(columns={col_salesLv: "Sales Domain.[Ship To]"})

    # add : df_in_floor_fcst['Flooring FCST']
    list_simul_bl_agg_key = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']
    df_simul_sellin_bl_agg = df_simul_sellin_bl_agg.merge(df_sales_floor_fcst_agg, how='left', on=list_simul_bl_agg_key)
    df_simul_sellin_bl_agg['Flooring FCST'] = df_simul_sellin_bl_agg['Flooring FCST'].fillna(value=0)

    logger.Note(f'Step 4-2)')
    fn_PrintDF(p_df=df_simul_sellin_bl_agg, p_df_name=f'df_simul_sellin_bl_agg ({df_simul_sellin_bl_agg.shape[0]})')

    ##################################################################
    # simul 실행 : ch
    # * (25.09.25)  음수 값 예외 처리 로직 삭제
    ##################################################################
    if not df_simul_sellin_bl_agg.empty:
        arr_simul = [df_simul_sellin_bl_agg['(Simul)Ch Inv PW_AP2'].values[0]]
        for i in range(1, len(df_simul_sellin_bl_agg.index)):
            value = df_simul_sellin_bl_agg['(Simul)Ch Inv PW_AP2'].values[i]
            week = df_simul_sellin_bl_agg['Time.[Week]'].values[i]

            if (week >= CurrentWeek) & (week <= CurrentWeek_52):
                sin = df_simul_sellin_bl_agg['S/In FCST(BL)_AP2'].values[i]
                sout = df_simul_sellin_bl_agg['S/Out FCST_AP2'].values[i]

                if np.isnan(sin):   sin = 0
                if np.isnan(sout):  sout = 0

                value = arr_simul[i - 1] + sin - sout
                arr_simul.append(value)
            else:
                arr_simul.append(value)
        df_simul_sellin_bl_agg['(Simul)Ch Inv PW_AP2'] = arr_simul
        # df_simul_sellin_bl_agg['(Simul)Ch Inv_Inc Floor PW_AP2'] = arr_simul_floor

    ##################################################################################
    # - 25.11.15 Measure.[(Simul)Ch Inv_Inc Floor PW_AP2] = Measure.[(Simul)Ch Inv PW_AP2] + Measure.[Flooring FCST]
    ##################################################################################
    df_simul_sellin_bl_agg['(Simul)Ch Inv_Inc Floor PW_AP2'] = np.where(df_simul_sellin_bl_agg['(Simul)Ch Inv PW_AP2'].isna()
                                                                        , np.nan
                                                                        , df_simul_sellin_bl_agg['(Simul)Ch Inv PW_AP2'] + df_simul_sellin_bl_agg['Flooring FCST'])

    logger.Note(f'Step 4-3)')
    fn_PrintDF(p_df=df_simul_sellin_bl_agg, p_df_name=f'df_simul_sellin_bl_agg ({df_simul_sellin_bl_agg.shape[0]}) - (Simul)Ch Inv PW_AP2, (Simul)Ch Inv_Inc floor_AP2 검증 완료')

    ##################################################################################
    # - (Simul)P4W Ch Inv PW, (Simul)P4W Ch Inv_Inc Floor PW : Measure.[(Simul)P4W Ch Inv PW] =
    # (25.09.25) (Simul)P4W Ch Inv PW : Measure.[(Simul)P4W Ch Inv PW] = Measure.[(Simul)P4W Ch Inv PW]@(Time.#.leadoffset(-1)) + Measure.[(Copy)S/In FCST(BL)_AP2] - Measure.[(Copy)S/Out Actual(P4W)]
    # (25.09.25) (Simul)P4W Ch Inv_Inc Floor PW : Measure.[(Simul)P4W Ch Inv_Inc Floor PW] = Measure.[(Simul)P4W Ch Inv_Inc Floor PW]@(Time.#.leadoffset(-1)) + Measure.[(Copy)S/In FCST(BL)_AP2] - Measure.[(Copy)S/Out Actual(P4W)] + Measure.[Flooring FCST]
    ##################################################################################
    list_simul_p4w = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]', 'S/In FCST(BL)_AP2',
                      '(Simul)P4W Ch Inv PW', '(Simul)P4W Ch Inv_Inc Floor PW']
    df_simul_sellin_p4w_agg = df_tmp_sellin_bl_agg[list_simul_p4w].copy(deep=True)

    df_simul_sellin_p4w_agg = df_simul_sellin_p4w_agg.merge(df_out_sellout_agg_p4w, how='left',
                                                            on=['Sales Domain.[Ship To]', 'Item.[Item]'])

    list_simul_p4w = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]', 'S/In FCST(BL)_AP2', 'S/Out Actual(P4W)',
                      '(Simul)P4W Ch Inv PW', '(Simul)P4W Ch Inv_Inc Floor PW']
    df_simul_sellin_p4w_agg = df_simul_sellin_p4w_agg[list_simul_p4w]

    # * 당주주차부터 ~ W+52 까지의 값 동일
    where_simul_nan = (df_simul_sellin_p4w_agg['Time.[Week]'] < CurrentWeek)
    df_simul_sellin_p4w_agg.loc[where_simul_nan, 'S/Out Actual(P4W)'] = np.nan

    # * W-1 주차의 값 (202536)
    # * 당주주차부터 ~ W+52 까지 연산 필요함.
    list_simul_week = df_simul_sellin_p4w_agg.loc[df_simul_sellin_p4w_agg['Time.[Week]'] < CurrentWeek, 'Time.[Week]'].unique().tolist()
    if len(list_simul_week) > 1:
        list_simul_week = list_simul_week[-1]

    if len(list_simul_week) > 1:
        where_simul_nan = (df_simul_sellin_p4w_agg['Time.[Week]'] < list_simul_week)
        df_simul_sellin_p4w_agg.loc[where_simul_nan, '(Simul)P4W Ch Inv PW'] = np.nan
        df_simul_sellin_p4w_agg.loc[where_simul_nan, '(Simul)P4W Ch Inv_Inc Floor PW'] = np.nan

    # add : df_in_floor_fcst['Flooring FCST']
    list_simul_p4w_agg_key = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']
    df_simul_sellin_p4w_agg = df_simul_sellin_p4w_agg.merge(df_sales_floor_fcst_agg, how='left', on=list_simul_p4w_agg_key)

    df_simul_sellin_p4w_agg['Flooring FCST'] = df_simul_sellin_p4w_agg['Flooring FCST'].fillna(value=0)

    logger.Note(f'Step 4-4)')
    fn_PrintDF(p_df=df_simul_sellin_p4w_agg, p_df_name=f'df_simul_sellin_p4w_agg ({df_simul_sellin_p4w_agg.shape[0]})')

    ##################################################################
    # simul 실행 : p4w
    # * (25.09.25)  음수 값 예외 처리 로직 삭제
    ##################################################################
    if not df_simul_sellin_p4w_agg.empty:
        arr_simul = [df_simul_sellin_p4w_agg['(Simul)P4W Ch Inv PW'].values[0]]
        for i in range(1, len(df_simul_sellin_p4w_agg.index)):
            value = df_simul_sellin_p4w_agg['(Simul)P4W Ch Inv PW'].values[i]
            week = df_simul_sellin_p4w_agg['Time.[Week]'].values[i]

            if (week >= CurrentWeek) & (week <= CurrentWeek_52):
                sin = df_simul_sellin_bl_agg['S/In FCST(BL)_AP2'].values[i]
                sout = df_simul_sellin_p4w_agg['S/Out Actual(P4W)'].values[i]

                if np.isnan(sin):   sin = 0
                if np.isnan(sout):  sout = 0

                value = arr_simul[i - 1] + sin - sout
                arr_simul.append(value)
            else:
                arr_simul.append(value)
        df_simul_sellin_p4w_agg['(Simul)P4W Ch Inv PW'] = arr_simul
        # df_simul_sellin_p4w_agg['(Simul)P4W Ch Inv_Inc Floor PW'] = arr_simul_floor

    ##################################################################################
    # - 25.11.15 Measure.[(Simul)Ch Inv_Inc Floor PW_AP2] = Measure.[(Simul)Ch Inv PW_AP2] + Measure.[Flooring FCST]
    ##################################################################################
    df_simul_sellin_p4w_agg['(Simul)P4W Ch Inv_Inc Floor PW'] = np.where(df_simul_sellin_p4w_agg['(Simul)P4W Ch Inv PW'].isna()
                                                                         , np.nan
                                                                         , df_simul_sellin_p4w_agg['(Simul)P4W Ch Inv PW'] + df_simul_sellin_p4w_agg['Flooring FCST'])

    logger.Note(f'Step 4-5)')
    fn_PrintDF(p_df=df_simul_sellin_bl_agg, p_df_name=f'df_simul_sellin_bl_agg ({df_simul_sellin_bl_agg.shape[0]})')
    fn_PrintDF(p_df=df_simul_sellin_p4w_agg, p_df_name=f'df_simul_sellin_p4w_agg ({df_simul_sellin_p4w_agg.shape[0]}) - (Simul)P4W Ch Inv PW, (Simul)P4W Ch Inv_Inc Floor PW 검증 완료')

    ##################################################################
    # (25.09.25) simul merge 및 partial week 변환
    ##################################################################
    df_ab = pd.DataFrame(data=[0, 6], columns=['p_day_delta'])

    df_partial_week = df_simul_sellin_p4w_agg[['Time.[Week]']].copy(deep=True)
    df_partial_week = df_partial_week.drop_duplicates()

    df_partial_week = df_partial_week.merge(df_ab, how='cross')

    df_partial_week['datetime'] = np.nan
    df_partial_week['Time.[Partial Week]'] = np.nan
    if df_partial_week.shape[0] > 0:
        # week -> datetime
        df_partial_week['datetime'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_to_date(p_str_datetype=x['Time.[Week]'], p_format='%Y%W',
                                                                                             p_week_day=1, p_day_delta=x['p_day_delta']), axis=1)
        # datetime -> partial week
        df_partial_week['Time.[Partial Week]'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_get_partial_week(p_datetime=x['datetime'],
                                                                                                                 p_bool_FI_week=True), axis=1)
    df_partial_week = df_partial_week.drop(['p_day_delta', 'datetime'], axis=1)
    df_partial_week = df_partial_week.drop_duplicates().reset_index(drop=True)
    df_partial_week = df_partial_week.astype('string')

    # base
    df_simul_sellin_pw_agg = df_simul_sellin_p4w_agg.merge(df_partial_week, how='inner', on=['Time.[Week]'])

    # add
    list_simul_bl = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']
    df_simul_sellin_pw_agg = df_simul_sellin_pw_agg.merge(df_simul_sellin_bl_agg, how='inner', on=list_simul_bl, suffixes=('', '_y'))

    list_out_pw = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]'
        , '(Simul)Ch Inv PW_AP2', '(Simul)Ch Inv_Inc Floor PW_AP2', '(Simul)P4W Ch Inv PW', '(Simul)P4W Ch Inv_Inc Floor PW']
    df_simul_sellin_pw_agg = df_simul_sellin_pw_agg[list_out_pw]
    df_simul_sellin_pw_agg = df_simul_sellin_pw_agg.sort_values(by=['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]']).reset_index(drop=True)

    # End Step log
    logger.Step(4, 'Ch Inv 값 연산')
    fn_PrintDF(p_df=df_simul_sellin_pw_agg, p_df_name=f'df_simul_sellin_pw_agg ({df_simul_sellin_pw_agg.shape[0]})')

    '''(20251027) WOS 내용 삭제 처리
    ####################################################################################################################
    # Step 5) WOS 값 연산
    # 	(Input 3) WOS 계산을 위한 정보 <- WOS는 Recurrence가 아니기 때문에 데이터 가져올 필요 X, 매번 새로운 Data 넣어주어야함. Q : Delta만 저장할 것인지. 아니면 52주 매번 전체 저장할 것인지
    # 	- WOS 는 Week -> Partial Week 구성 (Copy)
    #   - (25.09.25) WOS가 Inf 로 떨어지는 것에 대해서 SafeDivide -> 0 처리 필요
    # 	- (25.09.25)  (Simul)WOS_AP2, (Simul)WOS_Inc Floor_AP2 : Measure.[(Simul)WOS_AP2] = if ( Measure.[(Simul)Ch Inv PW_AP2] < 0 ) then 0 else Safedivide(Measure.[(Simul)Ch Inv PW_AP2], Measure.[(Week Avg)S/Out FCST F4W_AP2], 0);
    ####################################################################################################################
    df_wos_sellin_bl_agg = df_simul_sellin_bl_agg.merge(df_out_sellout_agg_f4w, how='left',
                                                        on=['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]'])

    where_w52 = (df_wos_sellin_bl_agg['Time.[Week]'] >= CurrentWeek) & (df_wos_sellin_bl_agg['Time.[Week]'] <= CurrentWeek_52)
    df_wos_sellin_bl_agg = df_wos_sellin_bl_agg.loc[where_w52]

    # '(Simul)WOS_AP2' = '(Simul)Ch Inv PW_AP2' / '(Week Avg)S/Out FCST F4W_AP2'
    df_wos_sellin_bl_agg['(Simul)WOS_AP2'] = np.where(df_wos_sellin_bl_agg['(Simul)Ch Inv PW_AP2'] < 0
                                                      , 0
                                                      , np.where(df_wos_sellin_bl_agg['(Week Avg)S/Out FCST F4W_AP2'].isna() | (df_wos_sellin_bl_agg['(Week Avg)S/Out FCST F4W_AP2'] == 0)
                                                                 , 0
                                                                 , df_wos_sellin_bl_agg['(Simul)Ch Inv PW_AP2'] / df_wos_sellin_bl_agg['(Week Avg)S/Out FCST F4W_AP2'])
                                                      )

    # '(Simul)WOS_Inc Floor_AP2' = '(Simul)Ch Inv_Inc Floor PW_AP2' / '(Week Avg)S/Out FCST F4W_AP2'
    df_wos_sellin_bl_agg['(Simul)WOS_Inc Floor_AP2'] = np.where(df_wos_sellin_bl_agg['(Simul)Ch Inv_Inc Floor PW_AP2'] < 0
                                                                , 0
                                                                , np.where(df_wos_sellin_bl_agg['(Week Avg)S/Out FCST F4W_AP2'].isna() | (df_wos_sellin_bl_agg['(Week Avg)S/Out FCST F4W_AP2'] == 0)
                                                                           , 0
                                                                           , df_wos_sellin_bl_agg['(Simul)Ch Inv_Inc Floor PW_AP2'] / df_wos_sellin_bl_agg['(Week Avg)S/Out FCST F4W_AP2'])
                                                                )

    logger.Note(f'Step 5-1)')
    fn_PrintDF(p_df=df_wos_sellin_bl_agg, p_df_name=f'df_wos_sellin_bl_agg ({df_wos_sellin_bl_agg.shape[0]}) - (Simul)WOS_AP2, (Simul)WOS_Inc Floor_AP2 검증 완료')

    ##########################################
    # * WOS 는 Week -> Partial Week 구성 (Copy)
    ##########################################
    # Partial Week
    # df_partial_week = df_in_split_ratio[['Time.[Week]', 'Time.[Partial Week]']].copy(deep=True)
    # df_partial_week = df_partial_week.drop_duplicates()

    df_ab = pd.DataFrame(data=[0, 6], columns=['p_day_delta'])

    df_partial_week = df_wos_sellin_bl_agg[['Time.[Week]']].copy(deep=True)
    df_partial_week = df_partial_week.drop_duplicates()

    df_partial_week = df_partial_week.merge(df_ab, how='cross')

    df_partial_week['datetime'] = np.nan
    df_partial_week['Time.[Partial Week]'] = np.nan
    if df_partial_week.shape[0] > 0:
        # week -> datetime
        df_partial_week['datetime'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_to_date(p_str_datetype=x['Time.[Week]'], p_format='%Y%W',
                                                                                        p_week_day=1, p_day_delta=x['p_day_delta']), axis=1)
        # datetime -> partial week
        df_partial_week['Time.[Partial Week]'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_get_partial_week(p_datetime=x['datetime'],
                                                                                                                 p_bool_FI_week=True), axis=1)

    df_partial_week = df_partial_week.drop(['p_day_delta', 'datetime'], axis=1)
    df_partial_week = df_partial_week.drop_duplicates().reset_index(drop=True)
    df_partial_week = df_partial_week.astype('string')

    df_out_wos_sellin_bl_agg = df_wos_sellin_bl_agg.merge(df_partial_week, how='inner', on=['Time.[Week]'])

    list_out_wos = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]', '(Simul)WOS_AP2', '(Simul)WOS_Inc Floor_AP2']
    df_out_wos_sellin_bl_agg = df_out_wos_sellin_bl_agg[list_out_wos]
    df_out_wos_sellin_bl_agg = df_out_wos_sellin_bl_agg.sort_values(by=['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]']).reset_index(drop=True)

    logger.Note(f'Step 5-2)')
    fn_PrintDF(p_df=df_out_wos_sellin_bl_agg, p_df_name=f'df_wos_sellin_bl_agg ({df_out_wos_sellin_bl_agg.shape[0]})')

    ##############################################
    # - (Simul)P4W WOS, (Simul)P4W WOS_Inc Floor : Measure.[(Simul)P4W WOS] = Safedivide(Measure.[(Simul)P4W Ch Inv PW], Measure.[(Week Avg)S/Out Actual(P4W) F4W], 0);
    # - (25.09.25)  (Simul)WOS_AP2, (Simul)WOS_Inc Floor_AP2 :
    #   Measure.[(Simul)WOS_AP2] = if ( Measure.[(Simul)Ch Inv PW_AP2] < 0 ) then 0
    #                              else Safedivide(Measure.[(Simul)Ch Inv PW_AP2], Measure.[(Week Avg)S/Out FCST F4W_AP2], 0);
    ##############################################
    list_wos_p4w = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]'
                    , '(Simul)P4W Ch Inv PW', '(Simul)P4W Ch Inv_Inc Floor PW', 'S/Out Actual(P4W)']
    df_wos_sellin_p4w_agg = df_simul_sellin_p4w_agg[list_wos_p4w].copy(deep=True)

    where_w52 = (df_wos_sellin_p4w_agg['Time.[Week]'] >= CurrentWeek) & (df_wos_sellin_p4w_agg['Time.[Week]'] <= CurrentWeek_52)
    df_wos_sellin_p4w_agg = df_wos_sellin_p4w_agg.loc[where_w52]

    # '(Simul)P4W WOS' = '(Simul)P4W Ch Inv PW' / 'S/Out Actual(P4W)'
    df_wos_sellin_p4w_agg['(Simul)P4W WOS'] = np.where(df_wos_sellin_p4w_agg['(Simul)P4W Ch Inv PW'] < 0
                                                       , 0
                                                       , np.where(df_wos_sellin_p4w_agg['S/Out Actual(P4W)'].isna() | (df_wos_sellin_p4w_agg['S/Out Actual(P4W)'] == 0)
                                                                  , 0
                                                                  , df_wos_sellin_p4w_agg['(Simul)P4W Ch Inv PW'] / df_wos_sellin_p4w_agg['S/Out Actual(P4W)'])
                                                       )

    # '(Simul)P4W WOS_Inc Floor' = '(Simul)P4W Ch Inv_Inc Floor PW' / 'S/Out Actual(P4W)'
    df_wos_sellin_p4w_agg['(Simul)P4W WOS_Inc Floor'] = np.where(df_wos_sellin_p4w_agg['(Simul)P4W Ch Inv_Inc Floor PW'] < 0
                                                                 , 0
                                                                 , np.where(df_wos_sellin_p4w_agg['S/Out Actual(P4W)'].isna() | (df_wos_sellin_p4w_agg['S/Out Actual(P4W)'] == 0)
                                                                            , 0
                                                                            , df_wos_sellin_p4w_agg['(Simul)P4W Ch Inv_Inc Floor PW'] / df_wos_sellin_p4w_agg['S/Out Actual(P4W)'])
                                                                 )

    logger.Note(f'Step 5-3)')
    fn_PrintDF(p_df=df_wos_sellin_p4w_agg, p_df_name=f'df_wos_sellin_p4w_agg ({df_wos_sellin_p4w_agg.shape[0]})')

    # * WOS 는 Week -> Partial Week 구성 (Copy)
    df_out_wos_sellin_p4w_agg = df_wos_sellin_p4w_agg.merge(df_partial_week, how='inner', on=['Time.[Week]'])

    list_out_wos_p4w = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]'
        , '(Simul)P4W WOS', '(Simul)P4W WOS_Inc Floor']
    df_out_wos_sellin_p4w_agg = df_out_wos_sellin_p4w_agg[list_out_wos_p4w]
    df_out_wos_sellin_p4w_agg = df_out_wos_sellin_p4w_agg.sort_values(by=['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]']).reset_index(drop=True)
    '''

    # End Step log
    logger.Step(5, 'WOS 값 연산 (20251027) 삭제 처리됨')
    # fn_PrintDF(p_df=df_out_wos_sellin_bl_agg, p_df_name=f'df_out_wos_sellin_bl_agg ({df_out_wos_sellin_bl_agg.shape[0]})')
    # fn_PrintDF(p_df=df_out_wos_sellin_p4w_agg, p_df_name=f'df_out_wos_sellin_p4w_agg ({df_out_wos_sellin_p4w_agg.shape[0]})')

    ####################################################################################################################
    # Step 6) (Simul)S/In FCST_AP2 연산
    # 	- Target Ch WOS 값 전처리
    ####################################################################################################################
    # * S/Out Delta가 있는 경우에 로직 수행                               -> df_in_sellout_fcst
    # * Step2의 S/Out FCST_AP2 Aggregation (Ship To, Item, Week) 된 값 -> df_in_sellout_agg
    #   과 (Week Avg)S/Out FCST F4W_AP2의 값을 가져온다.                 -> df_out_sellout_agg_f4w
    # * Ship To, Item, Week에 매칭 되는 Target CH WOS 값 가져오기

    df_in_sellout_fcst['Time.[Week]'] = df_in_sellout_fcst['Time.[Partial Week]'].str[:6]
    df_tmp_sellout_delta = df_in_sellout_fcst[['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']].copy(deep=True)
    df_tmp_sellout_delta = df_tmp_sellout_delta.drop_duplicates().reset_index()

    list_wos_target = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']
    df_out_wos_target = df_in_sellout_agg.merge(df_tmp_sellout_delta, how='left', on=list_wos_target)
    df_out_wos_target = df_out_wos_target.merge(df_out_sellout_agg_f4w, how='left', on=list_wos_target)
    df_out_wos_target = df_out_wos_target.merge(df_in_target_WOS, how='left', on=list_wos_target)

    logger.Note(f'Step 6-1)')
    fn_PrintDF(p_df=df_in_sellout_agg, p_df_name=f'df_in_sellout_agg ({df_in_sellout_agg.shape[0]})')
    fn_PrintDF(p_df=df_tmp_sellout_delta, p_df_name=f'df_tmp_sellout_delta ({df_tmp_sellout_delta.shape[0]})')

    fn_PrintDF(p_df=df_out_sellout_agg_f4w, p_df_name=f'df_out_sellout_agg_f4w ({df_out_sellout_agg_f4w.shape[0]})')
    fn_PrintDF(p_df=df_in_target_WOS, p_df_name=f'df_in_target_WOS ({df_in_target_WOS.shape[0]})')

    fn_PrintDF(p_df=df_out_wos_target, p_df_name=f'df_out_wos_target ({df_out_wos_target.shape[0]})')

    ############################################################
    # - Target Ch Inv 값 구하기
    ############################################################
    # * Target Ch WOS에서 S/Out Delta의 Ship To, Item
    # * (Simul)Target Ch Inv PW_AP2 = Target Ch WOS * (Week Avg)S/Out FCST F4W_AP2
    # * Target Ch WOS 가 Null 이면 0 처리
    # * 실제로 W+0 ~ W + 52 까지 값 연산
    df_out_wos_target['(Simul)Target Ch Inv PW_AP2'] = np.where(df_out_wos_target['Target Ch WOS'].isna()
                                                                , 0
                                                                , df_out_wos_target['Target Ch WOS'] * df_out_wos_target['(Week Avg)S/Out FCST F4W_AP2']
                                                                )

    logger.Note(f'Step 6-2)')
    fn_PrintDF(p_df=df_out_wos_target, p_df_name=f'df_out_wos_target ({df_out_wos_target.shape[0]})')

    ############################################################
    # - Target Ch Inv 값 PW 구하기
    ############################################################
    # * PW 으로 변환 시 W 값을 그대로 Copy
    list_out_wos_target_pw = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]',
                              '(Week Avg)S/Out FCST F4W_AP2', 'S/Out FCST_AP2',
                              'Target Ch WOS', '(Simul)Target Ch Inv PW_AP2']
    if df_out_wos_target.empty:
        df_out_wos_target_pw = pd.DataFrame(columns=list_out_wos_target_pw)
    else:
        df_ab = pd.DataFrame(data=[0, 6], columns=['p_day_delta'])

        df_partial_week = df_out_wos_target[['Time.[Week]']].copy(deep=True)
        df_partial_week = df_partial_week.drop_duplicates()

        df_partial_week = df_partial_week.merge(df_ab, how='cross')

        # week -> datetime
        df_partial_week['datetime'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_to_date(p_str_datetype=x['Time.[Week]'], p_format='%Y%W',
                                                                                             p_week_day=1, p_day_delta=x['p_day_delta']), axis=1)
        # datetime -> partial week
        df_partial_week['Time.[Partial Week]'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_get_partial_week(p_datetime=x['datetime'],
                                                                                                                 p_bool_FI_week=True), axis=1)
        df_partial_week = df_partial_week.drop(['p_day_delta', 'datetime'], axis=1)
        df_partial_week = df_partial_week.drop_duplicates().reset_index(drop=True)
        df_partial_week = df_partial_week.astype('string')

        df_out_wos_target_pw = df_out_wos_target.merge(df_partial_week, how='inner', on=['Time.[Week]'])

        df_out_wos_target_pw = df_out_wos_target_pw[list_out_wos_target_pw]
        df_out_wos_target_pw = df_out_wos_target_pw.sort_values(by=['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]']).reset_index(drop=True)

    logger.Note(f'Step 6-3)')
    fn_PrintDF(p_df=df_out_wos_target_pw, p_df_name=f'df_out_wos_target_pw ({df_out_wos_target_pw.shape[0]})')

    ############################################################
    # - (Simul) S/In FCST_AP2 값 연산
    ############################################################
    # * Step 4에서 계산된 (Simul)Ch Inv PW_AP2 값 추가 (W-1 값까지)
    # * (Simul)S/In FCST_AP2 = (Simul)Target Ch Inv PW_AP2 + S/Out FCST_AP2 - (Simul)Ch Inv PW_AP2(-1)
    # * 실제로 W+0 ~ W + 52 까지 값 연산
    # * (Simul)Target Ch Inv PW_AP2 는 Step7 Output 구성시 필요함

    # step 4) 결과값 -> partial week -> week
    df_tmp_simul_sellin_agg = df_simul_sellin_pw_agg.copy(deep=True)
    df_tmp_simul_sellin_agg['Time.[Week]'] = df_tmp_simul_sellin_agg['Time.[Partial Week]'].str[:6]

    list_tmp_sellin_agg = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]', '(Simul)Ch Inv PW_AP2']
    df_tmp_simul_sellin_agg = df_tmp_simul_sellin_agg[list_tmp_sellin_agg]
    df_tmp_simul_sellin_agg = df_tmp_simul_sellin_agg.drop_duplicates().reset_index(drop=True)

    # step 6) target 결과값 partial week -> week -> left outer join -> 연산후 -> W+0 ~ W + 52 까지
    df_tmp_out_wos_target = df_out_wos_target_pw.copy(deep=True)
    df_tmp_out_wos_target['Time.[Week]'] = df_tmp_out_wos_target['Time.[Partial Week]'].str[:6]

    df_tmp_out_wos_target = df_tmp_out_wos_target.drop(['Time.[Partial Week]'], axis=1)
    df_tmp_out_wos_target = df_tmp_out_wos_target.drop_duplicates().reset_index(drop=True)

    # step 4) left outer join step 6) -> 연산후 -> W+0 ~ W + 52 까지
    list_out_simul_key = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']
    df_out_simul_sellin_agg = df_tmp_simul_sellin_agg.merge(df_tmp_out_wos_target, how='left', on=list_out_simul_key)
    df_out_simul_sellin_agg['(Simul)S/In FCST_AP2'] = np.nan

    ##################################################################
    # (Simul)S/In FCST_AP2 계산 :
    ##################################################################
    if not df_out_simul_sellin_agg.empty:
        arr_simul = [df_out_simul_sellin_agg['(Simul)S/In FCST_AP2'].values[0]]
        for i in range(1, len(df_out_simul_sellin_agg.index)):
            value = df_out_simul_sellin_agg['(Simul)S/In FCST_AP2'].values[i]
            week = df_out_simul_sellin_agg['Time.[Week]'].values[i]

            if (week >= CurrentWeek) & (week <= CurrentWeek_52):
                chinv = df_out_simul_sellin_agg['(Simul)Target Ch Inv PW_AP2'].values[i]
                sout = df_out_simul_sellin_agg['S/Out FCST_AP2'].values[i]
                chinv_pre = df_out_simul_sellin_agg['(Simul)Ch Inv PW_AP2'].values[i - 1]

                if np.isnan(chinv):      chinv = 0
                if np.isnan(sout):       sout = 0
                if np.isnan(chinv_pre):  chinv_pre = 0

                value = chinv + sout - chinv_pre
                arr_simul.append(value)
            else:
                arr_simul.append(np.nan)
        df_out_simul_sellin_agg['(Simul)S/In FCST_AP2'] = arr_simul

    logger.Note(f'Step 6-4)')
    fn_PrintDF(p_df=df_out_simul_sellin_agg, p_df_name=f'df_out_simul_sellin_agg ({df_out_simul_sellin_agg.shape[0]}) - (Simul) S/In FCST_AP2 값 연산')

    ############################################################
    # - (Simul) S/In FCST_AP2  Location 및 Partial Week 분배 연산
    ############################################################
    list_sellin_fcst = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]', '(Simul)S/In FCST_AP2']
    # where_sellin_fcst = df_out_simul_sellin_agg['(Simul)S/In FCST_AP2'].notna()
    where_sellin_fcst = (df_out_simul_sellin_agg['Time.[Week]'] >= CurrentWeek) & (df_out_simul_sellin_agg['Time.[Week]'] <= CurrentWeek_52)
    df_out_simul_sellin_pw = df_out_simul_sellin_agg[list_sellin_fcst].loc[where_sellin_fcst].copy(deep=True)

    fn_PrintDF(p_df=df_in_split_ratio, p_df_name=f'df_in_split_ratio ({df_in_split_ratio.shape[0]}) 전')

    # Partial Week -> Week -> distinct -> left outer join
    df_tmp_split_ratio = df_in_split_ratio.copy(deep=True)
    df_tmp_split_ratio['Time.[Week]'] = df_tmp_split_ratio['Time.[Partial Week]'].str[:6]

    # 중복발생으로 type을 지정함
    list_simul_sellin_pw = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']
    df_out_simul_sellin_pw[list_simul_sellin_pw] = df_out_simul_sellin_pw[list_simul_sellin_pw].astype('string')

    if not df_tmp_split_ratio.empty:
        list_out_simul_key = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']
        df_out_simul_sellin_pw = df_out_simul_sellin_pw.merge(df_tmp_split_ratio, how='left', on=list_out_simul_key)

    else:
        df_ab = pd.DataFrame(data=[0, 6], columns=['p_day_delta'])

        df_partial_week = df_out_simul_sellin_pw[['Time.[Week]']].copy(deep=True)
        df_partial_week = df_partial_week.drop_duplicates()

        df_partial_week = df_partial_week.merge(df_ab, how='cross')

        df_partial_week['datetime'] = np.nan
        df_partial_week['Time.[Partial Week]'] = np.nan
        if df_partial_week.shape[0] > 0:
            # week -> datetime
            df_partial_week['datetime'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_to_date(p_str_datetype=x['Time.[Week]'], p_format='%Y%W',
                                                                                                 p_week_day=1, p_day_delta=x['p_day_delta']), axis=1)
            # datetime -> partial week
            df_partial_week['Time.[Partial Week]'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_get_partial_week(p_datetime=x['datetime'],
                                                                                                                     p_bool_FI_week=True), axis=1)

        df_partial_week = df_partial_week.drop(['p_day_delta', 'datetime'], axis=1)
        df_partial_week = df_partial_week.drop_duplicates().reset_index(drop=True)
        df_partial_week = df_partial_week.astype('string')

        df_out_simul_sellin_pw = df_out_simul_sellin_pw.merge(df_partial_week, how='inner', on=['Time.[Week]'])
        df_out_simul_sellin_pw['S/In FCST(GI) Split Ratio_AP2'] = 1

        # location 없을 경우 df_in_fcst_gi_assortment 사용함
        df_out_simul_sellin_pw = df_out_simul_sellin_pw.merge(df_in_fcst_gi_assortment, how='left', on=['Sales Domain.[Ship To]', 'Item.[Item]'])

    df_out_simul_sellin_pw = df_out_simul_sellin_pw.drop_duplicates().reset_index(drop=True)

    fn_PrintDF(p_df=df_out_simul_sellin_pw, p_df_name=f'df_out_simul_sellin_pw ({df_out_simul_sellin_pw.shape[0]}) 전')

    # partial week is not nan
    where_delete = df_out_simul_sellin_pw['Time.[Partial Week]'].notna()
    df_out_simul_sellin_pw = df_out_simul_sellin_pw.loc[where_delete]

    fn_PrintDF(p_df=df_out_simul_sellin_pw, p_df_name=f'df_out_simul_sellin_pw ({df_out_simul_sellin_pw.shape[0]}) 후')

    # sort
    list_out_simul_pw = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]', 'Location.[Location]']
    df_out_simul_sellin_pw = df_out_simul_sellin_pw.sort_values(by=list_out_simul_pw).reset_index(drop=True)

    # -->>>>>>>>>>>>>>>>>>>> ratio
    list_out_simul_pw_all = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]',
                             '(Simul)S/In FCST_AP2', 'S/In FCST(GI) Split Ratio_AP2']
    df_out_simul_sellin_pw = df_out_simul_sellin_pw[list_out_simul_pw_all]

    # cnt, cum_count, cum_max
    df_out_simul_sellin_pw['Time.[Week]'] = df_out_simul_sellin_pw['Time.[Partial Week]'].str[:6]

    list_out_simul_pw_cum = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']

    df_out_simul_sellin_pw['cnt'] = 1
    df_out_simul_sellin_pw['cum_count'] = df_out_simul_sellin_pw.groupby(by=list_out_simul_pw_cum)['Time.[Partial Week]'].transform('cumcount')
    df_out_simul_sellin_pw['cum_max'] = df_out_simul_sellin_pw.groupby(by=list_out_simul_pw_cum)['cnt'].transform('sum')
    df_out_simul_sellin_pw['ratio_sum'] = df_out_simul_sellin_pw.groupby(by=list_out_simul_pw_cum)['S/In FCST(GI) Split Ratio_AP2'].transform('sum')

    # Initial Float Value, Cumulative error, Final Value 추가
    df_out_simul_sellin_pw['Initial Float Value'] = df_out_simul_sellin_pw['(Simul)S/In FCST_AP2'] * df_out_simul_sellin_pw['S/In FCST(GI) Split Ratio_AP2'] / df_out_simul_sellin_pw['ratio_sum']
    df_out_simul_sellin_pw['Cumulative error'] = np.where(df_out_simul_sellin_pw['cum_count'] == 0
                                                          , 0
                                                          , np.nan)
    df_out_simul_sellin_pw['Final Value'] = np.round(df_out_simul_sellin_pw['Initial Float Value'])
    df_out_simul_sellin_pw['Cumulative error'] = df_out_simul_sellin_pw['Final Value'] - df_out_simul_sellin_pw['Initial Float Value']

    fn_PrintDF(p_df=df_out_simul_sellin_pw, p_df_name=f'df_out_simul_sellin_pw ({df_out_simul_sellin_pw.shape[0]}) 소수점 계산 전')
    ##################################################################
    # 소수점 계산 :
    ##################################################################
    if not df_out_simul_sellin_pw.empty:
        arr_simul = [df_out_simul_sellin_pw['Final Value'].values[0]]
        arr_error = [df_out_simul_sellin_pw['Cumulative error'].values[0]]
        for i in range(1, len(df_out_simul_sellin_pw.index)):
            value_init = df_out_simul_sellin_pw['Initial Float Value'].values[i]
            value = df_out_simul_sellin_pw['Final Value'].values[i]
            week = df_out_simul_sellin_pw['Time.[Week]'].values[i]

            cum_count = df_out_simul_sellin_pw['cum_count'].values[i]
            cum_max = df_out_simul_sellin_pw['cum_max'].values[i]

            if (week >= CurrentWeek) & (week <= CurrentWeek_52):
                if cum_count == 0:
                    value = np.round(df_out_simul_sellin_pw['Initial Float Value'].values[i])
                    error = arr_error[i - 1] + value - value_init
                else:
                    if arr_error[i - 1] == 0:
                        value = np.round(df_out_simul_sellin_pw['Initial Float Value'].values[i])
                    elif arr_error[i - 1] > 0:
                        value = np.floor(df_out_simul_sellin_pw['Initial Float Value'].values[i])
                    elif arr_error[i - 1] < 0:
                        value = np.ceil(df_out_simul_sellin_pw['Initial Float Value'].values[i])
                    error = arr_error[i - 1] + value - value_init

                arr_error.append(error)
                arr_simul.append(value)
            else:
                arr_simul.append(np.nan)
        df_out_simul_sellin_pw['Final Value'] = arr_simul
        df_out_simul_sellin_pw['Cumulative error'] = arr_error
    ###########################################################
    # 소수점 계산 : End
    ###########################################################
    fn_PrintDF(p_df=df_out_simul_sellin_pw, p_df_name=f'df_out_simul_sellin_pw ({df_out_simul_sellin_pw.shape[0]}) 소수점 계산 후')

    df_out_simul_sellin_pw['Verify Value'] = df_out_simul_sellin_pw.groupby(by=list_out_simul_pw_cum)['Final Value'].transform('sum')

    df_out_simul_sellin_step6 = df_out_simul_sellin_pw.copy(deep=True)
    df_out_simul_sellin_step6['(Simul)S/In FCST_AP2'] = df_out_simul_sellin_step6['Final Value']

    # End Step log
    logger.Step(6, '(Simul)S/In FCST_AP2 연산')
    fn_PrintDF(p_df=df_out_simul_sellin_pw, p_df_name=f'df_out_simul_sellin_pw ({df_out_simul_sellin_pw.shape[0]}) - (Simul) S/In FCST_AP2  Location 및 Partial Week 분배 연산')
    fn_PrintDF(p_df=df_out_simul_sellin_step6, p_df_name=f'df_out_simul_sellin_step6 ({df_out_simul_sellin_step6.shape[0]})')

    ####################################################################################################################
    # Step 7) AMT 값 연산
    # 	- S/In FCST(GI) Delta 값 AMT 계산 : USD, Local
    #   - (25.12.06) : Delta 인 S/In FCST에 대해서 S/In FCST(GI) Modify_AP2 Measure 추가 및 값 1 적용
    ####################################################################################################################
    # price_USD
    df_out_amt_sellin_fcst_gi = df_in_sellin_fcst_gi.merge(df_in_estimated_price_USD, how='left',
                                                           on=['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]'])

    # calc
    df_out_amt_sellin_fcst_gi['S/In FCST(GI) AMT_USD_AP2'] = df_out_amt_sellin_fcst_gi['S/In FCST(GI)_AP2'] * df_out_amt_sellin_fcst_gi['Estimated Price_USD']
    df_out_amt_sellin_fcst_gi['S/In FCST(GI) AMT_Local_AP2'] = df_out_amt_sellin_fcst_gi['S/In FCST(GI)_AP2'] * df_out_amt_sellin_fcst_gi['Estimated Price_Local']
    df_out_amt_sellin_fcst_gi['S/In FCST(GI) Modify_AP2'] = 1

    logger.Note(f'Step 7-1)')
    fn_PrintDF(p_df=df_out_amt_sellin_fcst_gi, p_df_name=f'df_out_amt_sellin_fcst_gi ({df_out_amt_sellin_fcst_gi.shape[0]}) - S/In FCST(GI) Delta 값 AMT 계산 : USD, Local 검증 완료')

    #######################################################
    # - S/In FCST(BL) Delta 값 AMT 계산 : USD, Local, Region
    #######################################################
    where_org = (df_out_sellin_bl['org_value'] == 'org')
    df_out_amt_sellin_bl = df_out_sellin_bl.loc[where_org].copy(deep=True)

    # price_USD
    df_out_amt_sellin_bl = df_out_amt_sellin_bl.merge(df_in_estimated_price_USD, how='left',
                                                      on=['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]'])

    # calc
    df_out_amt_sellin_bl['S/In FCST(BL) AMT_USD_AP2'] = df_out_amt_sellin_bl['S/In FCST(BL)_AP2'] * df_out_amt_sellin_bl['Estimated Price_USD']
    df_out_amt_sellin_bl['S/In FCST(BL) AMT_Local_AP2'] = df_out_amt_sellin_bl['S/In FCST(BL)_AP2'] * df_out_amt_sellin_bl['Estimated Price_Local']
    df_out_amt_sellin_bl['S/In FCST(BL) AMT_Region_AP2'] = df_out_amt_sellin_bl['S/In FCST(BL)_AP2'] * df_out_amt_sellin_bl['Estimated Price_Region']

    df_out_amt_sellin_bl = df_out_amt_sellin_bl.drop('org_value', axis=1)

    logger.Note(f'Step 7-2)')
    fn_PrintDF(p_df=df_out_amt_sellin_bl, p_df_name=f'df_out_amt_sellin_bl ({df_out_amt_sellin_bl.shape[0]}) - S/In FCST(BL) Delta 값 AMT 계산 : USD, Local, Region 검증 완료')

    #######################################################
    # - S/Out FCST Delta 값 AMT 계산 : USD, Local
    #######################################################
    # price_USD
    df_out_amt_sellout_fcst = df_in_sellout_fcst.merge(df_in_estimated_price_USD, how='left',
                                                       on=['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]'])

    # calc
    df_out_amt_sellout_fcst['S/Out FCST AMT_USD_AP2'] = df_out_amt_sellout_fcst['S/Out FCST_AP2'] * df_out_amt_sellout_fcst['Estimated Price_USD']
    df_out_amt_sellout_fcst['S/Out FCST AMT_Local_AP2'] = df_out_amt_sellout_fcst['S/Out FCST_AP2'] * df_out_amt_sellout_fcst['Estimated Price_Local']

    #######################################################
    # (20251027) add
    # - (Week Avg)S/Out FCST F4W PW_AP2 생성 (Grain : Item, ShipTo, Location, Partial Week, Version)
    # * Step 2에서 만든 (Week Avg)S/Out FCST F4W_AP1 값 사용
    # - Week -> Partial Week 으로 전환 (동일값 Copy)
    # - Location.[Location] = '-' 추가
    #######################################################
    df_out_sellout_agg_f4w_pw = df_out_sellout_agg_f4w.copy(deep=True)

    # * PW 으로 변환 시 W 값을 그대로 Copy
    list_out_sellout_agg_f4w_pw = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]',
                                   '(Week Avg)S/Out FCST F4W PW_AP2']
    if df_out_sellout_agg_f4w.empty:
        df_out_sellout_agg_f4w_pw = pd.DataFrame(columns=list_out_sellout_agg_f4w_pw)
    else:
        df_ab = pd.DataFrame(data=[0, 6], columns=['p_day_delta'])

        df_partial_week = df_out_sellout_agg_f4w[['Time.[Week]']].copy(deep=True)
        df_partial_week = df_partial_week.drop_duplicates()

        df_partial_week = df_partial_week.merge(df_ab, how='cross')

        # week -> datetime
        df_partial_week['datetime'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_to_date(p_str_datetype=x['Time.[Week]'], p_format='%Y%W',
                                                                                             p_week_day=1, p_day_delta=x['p_day_delta']), axis=1)
        # datetime -> partial week
        df_partial_week['Time.[Partial Week]'] = df_partial_week.apply(lambda x: NSCMCommon.gfn_get_partial_week(p_datetime=x['datetime'],
                                                                                                                 p_bool_FI_week=True), axis=1)
        df_partial_week = df_partial_week.drop(['p_day_delta', 'datetime'], axis=1)
        df_partial_week = df_partial_week.drop_duplicates().reset_index(drop=True)
        df_partial_week = df_partial_week.astype('string')

        df_out_sellout_agg_f4w_pw = df_out_sellout_agg_f4w.merge(df_partial_week, how='inner', on=['Time.[Week]'])
        df_out_sellout_agg_f4w_pw['Location.[Location]'] = '-'
        df_out_sellout_agg_f4w_pw.rename(columns={'(Week Avg)S/Out FCST F4W_AP2': '(Week Avg)S/Out FCST F4W PW_AP2'}, inplace=True)

        df_out_sellout_agg_f4w_pw = df_out_sellout_agg_f4w_pw[list_out_sellout_agg_f4w_pw]
        df_out_sellout_agg_f4w_pw = df_out_sellout_agg_f4w_pw.sort_values(by=list_out_sellout_agg_f4w_pw).reset_index(drop=True)

    logger.Note(f'Step 7-3) (Week Avg)S/Out FCST F4W PW_AP2 생성')
    fn_PrintDF(p_df=df_out_sellout_agg_f4w_pw, p_df_name=f'df_out_sellout_agg_f4w_pw ({df_out_sellout_agg_f4w_pw.shape[0]})')

    # End Step log
    logger.Step(7, 'AMT 값 연산')
    fn_PrintDF(p_df=df_out_amt_sellin_fcst_gi, p_df_name=f'df_out_amt_sellin_fcst_gi ({df_out_amt_sellin_fcst_gi.shape[0]})')
    fn_PrintDF(p_df=df_out_amt_sellin_bl, p_df_name=f'df_out_amt_sellin_bl ({df_out_amt_sellin_bl.shape[0]})')
    fn_PrintDF(p_df=df_out_amt_sellout_fcst, p_df_name=f'df_out_amt_sellout_fcst ({df_out_amt_sellout_fcst.shape[0]}) - S/Out FCST Delta 값 AMT 계산 : USD, Local 검증 완료')

    ####################################################################################################################
    # Step 8) output 생성
    ####################################################################################################################
    dict_output_gi = {'Version.[Version Name]': 'string'
        , 'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Location.[Location]': 'string'
        , 'Time.[Partial Week]': 'string'
        , 'S/In FCST(GI)_AP2': 'float64'
        , 'S/In FCST(GI) AMT_USD_AP2': 'float64'
        , 'S/In FCST(GI) AMT_Local_AP2': 'float64'
        , 'S/In FCST(GI) Modify_AP2': 'int'}

    dict_output_simul = {'Version.[Version Name]': 'string'
        , 'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Location.[Location]': 'string'
        , 'Time.[Partial Week]': 'string'
        , '(Simul)S/In FCST_AP2': 'float64'}

    dict_output_bl = {'Version.[Version Name]': 'string'
        , 'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Location.[Location]': 'string'
        , 'Time.[Partial Week]': 'string'
        , 'S/In FCST(BL)_AP2': 'float64'
        , 'S/In FCST(BL) AMT_USD_AP2': 'float64'
        , 'S/In FCST(BL) AMT_Local_AP2': 'float64'
        , 'S/In FCST(BL) AMT_Region_AP2': 'float64'}

    dict_output_sellout = {'Version.[Version Name]': 'string'
        , 'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Location.[Location]': 'string'
        , 'Time.[Partial Week]': 'string'
        , 'S/Out FCST_AP2': 'float64'
        , 'S/Out FCST AMT_USD_AP2': 'float64'
        , 'S/Out FCST AMT_Local_AP2': 'float64'
        , 'S/Out FCST Modify_AP2': 'int64'}

    dict_output_chinv = {'Version.[Version Name]': 'string'
        , 'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Time.[Partial Week]': 'string'
        , '(Simul)Ch Inv PW_AP2': 'float64'
        , '(Simul)Ch Inv_Inc Floor PW_AP2': 'float64'
        , '(Simul)P4W Ch Inv PW': 'float64'
        , '(Simul)P4W Ch Inv_Inc Floor PW': 'float64'
        , '(Simul)Target Ch Inv PW_AP2': 'float64'}

    dict_output_wos = {'Version.[Version Name]': 'string'
        , 'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Time.[Partial Week]': 'string'
        , '(Simul)WOS_AP2': 'float64'
        , '(Simul)WOS_Inc Floor_AP2': 'float64'
        , '(Simul)P4W WOS': 'float64'
        , '(Simul)P4W WOS_Inc Floor': 'float64'}

    dict_output_sellout_f4w = {'Version.[Version Name]': 'string'
        , 'Sales Domain.[Ship To]': 'string'
        , 'Item.[Item]': 'string'
        , 'Location.[Location]': 'string'
        , 'Time.[Partial Week]': 'string'
        , '(Week Avg)S/Out FCST F4W PW_AP2': 'float64'}

    ########################################################
    # (Output 1) (25.09.25) (Simul)S/In FCST_AP2 추가
    list_gi = ['Version.[Version Name]', 'Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]',
               'S/In FCST(GI)_AP2', 'S/In FCST(GI) AMT_USD_AP2', 'S/In FCST(GI) AMT_Local_AP2', 'S/In FCST(GI) Modify_AP2']
    df_out_amt_sellin_fcst_gi['Version.[Version Name]'] = CWV

    df_output_Sellin_gi = df_out_amt_sellin_fcst_gi[list_gi]

    df_output_Sellin_gi = df_output_Sellin_gi.astype(dtype=dict_output_gi)

    ########################################################
    # (Output 1-1) (25.11.20) (Simul)S/In FCST_AP1 분리
    list_simul = ['Version.[Version Name]', 'Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]',
                  '(Simul)S/In FCST_AP2']
    df_out_simul_sellin_step6['Version.[Version Name]'] = CWV

    where_out_simul = (df_out_simul_sellin_step6['Time.[Week]'] >= CurrentWeek) & (df_out_simul_sellin_step6['Time.[Week]'] <= CurrentWeek_52)
    df_out_simul_sellin_step6 = df_out_simul_sellin_step6.loc[where_out_simul]

    df_output_Sellin_simul = df_out_simul_sellin_step6[list_simul]

    df_output_Sellin_simul = df_output_Sellin_simul.astype(dtype=dict_output_simul)

    ########################################################
    # (Output 2)
    list_bl = ['Version.[Version Name]', 'Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]',
               'S/In FCST(BL)_AP2', 'S/In FCST(BL) AMT_USD_AP2', 'S/In FCST(BL) AMT_Local_AP2', 'S/In FCST(BL) AMT_Region_AP2']
    df_out_amt_sellin_bl['Version.[Version Name]'] = CWV

    df_output_Sellin_bl = df_out_amt_sellin_bl[list_bl]
    df_output_Sellin_bl = df_output_Sellin_bl.astype(dtype=dict_output_bl)

    ########################################################
    # (Output 3)
    list_sellout = ['Version.[Version Name]', 'Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]',
                    'S/Out FCST_AP2', 'S/Out FCST AMT_USD_AP2', 'S/Out FCST AMT_Local_AP2', 'S/Out FCST Modify_AP2']
    df_out_amt_sellout_fcst['Version.[Version Name]'] = CWV

    df_output_Sellout = df_out_amt_sellout_fcst[list_sellout]
    df_output_Sellout = df_output_Sellout.astype(dtype=dict_output_sellout)

    ########################################################
    # (Output 4) (25.09.25) Partial Week 변경
    df_output_chinv = df_simul_sellin_bl_agg.merge(df_simul_sellin_p4w_agg, how='left'
                                                   , on=['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]'])

    list_ch = ['Version.[Version Name]', 'Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]',
               '(Simul)Ch Inv PW_AP2', '(Simul)Ch Inv_Inc Floor PW_AP2', '(Simul)P4W Ch Inv PW',
               '(Simul)P4W Ch Inv_Inc Floor PW', '(Simul)Target Ch Inv PW_AP2']
    df_output_chinv['Version.[Version Name]'] = CWV

    # (Simul)Target Ch Inv PW_AP2 추가 및 pw 변경
    df_out_wos_target_pw['Time.[Week]'] = df_out_wos_target_pw['Time.[Partial Week]'].str[:6]

    where_out_wos = (df_out_wos_target_pw['Time.[Week]'] >= CurrentWeek) & (df_out_wos_target_pw['Time.[Week]'] <= CurrentWeek_52)
    df_out_wos_target_pw = df_out_wos_target_pw.loc[where_out_wos]

    list_chinv = ['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Week]']
    df_output_chinv = df_output_chinv.merge(df_out_wos_target_pw, how='left', on=list_chinv, suffixes=('', '_y'))

    df_output_chinv = df_output_chinv.loc[df_output_chinv['Time.[Partial Week]'].notna()]

    df_output_chinv = df_output_chinv[list_ch]
    df_output_chinv = df_output_chinv.astype(dtype=dict_output_chinv)

    '''(20251027) WOS 내용 삭제 처리
    ########################################################
    # (Output 5)
    df_output_wos = df_out_wos_sellin_bl_agg.merge(df_out_wos_sellin_p4w_agg, how='left'
                                                   , on=['Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]'])

    list_wos = ['Version.[Version Name]', 'Sales Domain.[Ship To]', 'Item.[Item]', 'Time.[Partial Week]',
               '(Simul)WOS_AP2', '(Simul)WOS_Inc Floor_AP2', '(Simul)P4W WOS', '(Simul)P4W WOS_Inc Floor']
    df_output_wos['Version.[Version Name]'] = CWV

    df_output_wos = df_output_wos[list_wos]
    df_output_wos = df_output_wos.astype(dtype=dict_output_wos)
    '''

    ########################################################
    # (Output 6)
    df_output_Sellout_f4w = df_out_sellout_agg_f4w_pw.copy(deep=True)

    list_f4w_pw = ['Version.[Version Name]', 'Sales Domain.[Ship To]', 'Item.[Item]', 'Location.[Location]', 'Time.[Partial Week]',
                   '(Week Avg)S/Out FCST F4W PW_AP2']
    df_output_Sellout_f4w['Version.[Version Name]'] = CWV
    df_output_Sellout_f4w = df_output_Sellout_f4w[list_f4w_pw]
    df_output_Sellout_f4w = df_output_Sellout_f4w.astype(dtype=dict_output_sellout_f4w)

    # End Step log
    logger.Step(8, 'output 생성')
    fn_PrintDF(p_df=df_output_Sellin_gi, p_df_name=f'df_output_Sellin_gi ({df_output_Sellin_gi.shape[0]})')
    fn_PrintDF(p_df=df_output_Sellin_simul, p_df_name=f'df_output_Sellin_simul ({df_output_Sellin_simul.shape[0]})')
    fn_PrintDF(p_df=df_output_Sellin_bl, p_df_name=f'df_output_Sellin_bl ({df_output_Sellin_bl.shape[0]})')
    fn_PrintDF(p_df=df_output_Sellout, p_df_name=f'df_output_Sellout ({df_output_Sellout.shape[0]})')
    fn_PrintDF(p_df=df_output_chinv, p_df_name=f'df_output_chinv ({df_output_chinv.shape[0]})')
    # fn_PrintDF(p_df=df_output_wos, p_df_name=f'df_output_wos ({df_output_wos.shape[0]})')
    fn_PrintDF(p_df=df_output_Sellout_f4w, p_df_name=f'df_output_Sellout_f4w ({df_output_Sellout_f4w.shape[0]})')

    # End Step log
    logger.Step(9, 'Finish')

except Exception as e:
    trace_msg = traceback.format_exc()
    logger.Note(p_note=trace_msg, p_log_level=NSCMCommon.G_log_level.debug())
    logger.Error()
    raise Exception(e)

else:
    logger.Finish()
