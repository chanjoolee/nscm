"""
    (25.04.16) Sell-Out 변경점
        S/In 로직 이란 PYForecastSellInAndEstoreSellOutLockColor 을 말함.
        
        1. Input 변경 
        - Sales Product ASN 추가

        - Time 입력 구간 변경
            - 당월에 해당하는 주차부터 ~ 당주주차+52
            - 로직상 변경점 없음 (실적구간 Lock)
            - 당주주차 정보 Input으로 넣어주는 것으로 변경

        - Sell-Out Master 변경
            - Status 정보만 받아옴.


        2 Sales Product ASN 정보 추가에 따른 Step 1 변경
        - S/In 과 비슷하게 Color 반영한 Data 구성으로 로직 변경
        - 기존 Step 1-3) , Step1-4) 삭제 및 변경

        3. Step 1-4)에서 추가되는 Lock  명칭을 Lock Condition 으로 변경

        4. Step 2) S/In 로직의 Step6) 와 동일 로직 사용

        5. Step 3-2) 의 VD Lead Time 반영 로직 변경 (S/In 과 유사하지만, Location 차이)
        - Lock = False 처리로 변경
        - RTS 혹은 EOS가 있는 경우에는 색구간 덮어 씌우지 않음.
        - TATTERM -> ITEMTAT TATTERM 으로 참고 Measure 명칭 변경
        ex) BLUE, BLUE, BLUE, DGRAY_RED, DGRAY_RED 형태로 존재해야함.


        6.  기존 Step 4) -> Step 5) 로 이동 및 Step 4) Sales Product ASN 반영 추가
        - Step 4-1) == S/In Step 7)
        - Step 4-2) == S/In Step 8)
        - 하위 7Lv 뿐만 아니라, 6Lv에 대해서도 처리


        7. Step 5 변경 (구 Step 4)
        - Step 5-2) 삭제
        - Step 5-2) ( 구 Step 4-3) 로직 변경 
            - Lock = True 인 경우, 입력 막아주는 로직 적용

        8. 구 Step 5) 삭제 및 Step 6) 사용, S/In Step 12) 로직 변형해서 사용
        - GC, AP2, AP1 각각에 대해서 진행함.


	개요 (*)		
			
		* 프로그램명	
			PYForecastSellOutLockColor
			
			
		* 목적	
			- Sell Out Simul. Master 를 이용하여, Sell Out Measure 의 Lock 정보 정의
			
			
			
		* 변경이력	
			2025.01.14 전창민 최초 작성
			2025.02.11 Step 4-4) Product Group 에 대한 조건 추가 - * Step 1-3) 의 DataFrame을 사용하여, Item.[Product Group]과 Sales Domain.[Ship To] 를 조건으로 Item.[Item] 값을 추가한다.
			2025.02.11 Step 4-4) Item 이 없는 경우에 대한 조건 추가 Product Group 과 Ship To에 대해서 Mapping 되는 경우가 없다면, 해당 row는 삭제한다.
			2025.03.13 색 명칭 변경
			2025.03.13 Dummy 생성 시 공통만 생성하도록 로직 변경
			2025.03.18 Sell-In 에서 E-store Sell-out 작업 진행으로 실행 구문에서 E-store 제외
			2025.04.15 입력 Lv에 맞는 Dummy만 생성하도록 로직 변경
			
	Script Parameter		
			
		(Input 1) CurrentPartialWeek	
			&CurrentPartialWeek - NameSet 활용

	Input Tables (*)					
						
		(Input 1) Sell-Out Simul. Master 정보 (TO-BE : 7LV)				
			df_in_Sell_Out_Simul_Master			
				Select ([Version].[Version Name]		
				 * [Sales Domain].[Ship To]		
				 * [Item].[Product Group] )  on row, 		
				( {Measure.[S/Out Master Status] } ) on column;		
						
						
			Version.[Version Name]	Sales Domain.[Ship To]	Item.[Product Group]	S/Out Master Status
			CWV_DP	A5006941	APS	CON
			CWV_DP	A5006941	MOBILE	CON
			CWV_DP	A5006941	PC	CON
			CWV_DP	A5019692	MOBILE	CON
			CWV_DP	A5017132	APS	CON
			CWV_DP	A5017132	MOBILE	CON
			CWV_DP	A5017132	PC	CON
			CWV_DP	A5022556	APS	CON
			CWV_DP	A5022556	MOBILE	CON
			CWV_DP	A5022556	PC	CON
			CWV_DP	A5017734	APS	CON
			CWV_DP	A5017734	MOBILE	CON
			CWV_DP	A5003117	APS	CON
			CWV_DP	A5003117	MOBILE	CON
			CWV_DP	A5003117	PC	CON
			CWV_DP	A5002458	APS	CON
			CWV_DP	A5002458	MOBILE	CON
		
        (Input 2) Forecast Rule 정보								
			df_in_Forecast_Rule							
				Select ([Version].[Version Name]						
				 * [Item].[Product Group] 						
				 * [Sales Domain].[Ship To] )  on row, 						
				( { Measure.[FORECAST_RULE GC FCST], Measure.[FORECAST_RULE AP2 FCST], Measure.[FORECAST_RULE AP1 FCST], Measure.[FORECAST_RULE CUST FCST],Measure.[FORECAST_RULE ISVALID] } ) on column;						
										
										
			Version.[Version Name]	Item.[Product Group]	Sales Domain.[Ship To]	FORECAST_RULE GC FCST	FORECAST_RULE AP2 FCST	FORECAST_RULE AP1 FCST	FORECAST_RULE CUST FCST	FORECAST_RULE ISVALID
			CWV_DP	MONITOR	211	5		5		Y
			CWV_DP	DAS	211	5	5			Y
			CWV_DP	OPTICS	300114	5				Y
			CWV_DP	ENTERPRISE	300114	3	4	5		Y
			CWV_DP	SYSTEM	300114	5	5	5		Y
			CWV_DP	HME	300114	3	5	5		Y
			CWV_DP	COMPUTER	300114	3	5	5	5	Y
			CWV_DP	PRT	300114	3	5	5		Y
			CWV_DP	TV	300114	3	5	5	5	Y
			CWV_DP	STB	300114	3	5	5	5	Y
			CWV_DP	DA	300114	3	4	5	5	Y
			CWV_DP	MEM_BRAND	300114	3	5	5	5	Y
			CWV_DP	AV	300114	3	5	5	5	Y
			CWV_DP	MOBILE	300114	5	5	5	5	Y
			CWV_DP	DAS	300114	3	4	5	5	Y
			CWV_DP	MONITOR	300114	3	5	5	5	Y
			CWV_DP	MID	300114	3	5	5		Y

		(Input 3) Sales Domian Master 정보							
			df_in_Sales_Domain_Master						
				Select (					
				 * [Sales Domain].[Sales Domain Lv2]					
				 * [Sales Domain].[Sales Domain Lv3] 					
				 * [Sales Domain].[Sales Domain Lv4] 					
				 * [Sales Domain].[Sales Domain Lv5] 					
				 * [Sales Domain].[Sales Domain Lv6] 					
				 * [Sales Domain].[Sales Domain Lv7] 					
				 * [Sales Domain].[Ship To] )					
									
									
									
			Sales Domain.[Sales Domain Lv2]	Sales Domain.[Sales Domain Lv3]	Sales Domain.[Sales Domain Lv4]	Sales Domain.[Sales Domain Lv5]	Sales Domain.[Sales Domain Lv6]	Sales Domain.[Sales Domain Lv7]	Sales Domain.[Ship To]
			203	203	203	203	203	203	203
			203	300114	300114	300114	300114	300114	300114
			203	300114	A300114	A300114	A300114	A300114	A300114
			203	300114	A300114	400362	400362	400362	400362
			203	300114	A300114	400362	5002453	5002453	5002453
			203	300114	A300114	400362	5002453	A5002453	A5002453
			203	300114	A300114	400362	5003074	5003074	5003074
			203	300114	A300114	400362	5003074	A5003074	A5003074
			203	300114	A300114	400362	5005569	5005569	5005569
			203	300114	A300114	400362	5005569	A5005569	A5005569
			203	300114	A300114	400362	5007280	5007280	5007280
			203	300114	A300114	400362	5007280	A5007280	A5007280
			203	300114	A300114	400362	5013134	5013134	5013134
			203	300114	A300114	400362	5013134	A5013134	A5013134
			203	300114	A300114	408273	408273	408273	408273
			203	300114	A300114	408273	5006941	5006941	5006941
			203	300114	A300114	408273	5006941	A5006941	A5006941
			203	300114	A300114	408273	5019692	5019692	5019692
			203	300114	A300114	408273	5019692	A5019692	A5019692
			211	211	211	211	211	211	211
			211	300227	300227	300227	300227	300227	300227
			211	300227	A300227	A300227	A300227	A300227	A300227
			211	300227	A300227	400144	400144	400144	400144
			211	300227	A300227	400144	A400144	A400144	A400144
			211	300227	A300227	400144	A400144	5002090	5002090
			211	300227	A300227	400144	A400144	5002090	A5002090

		(Input 4) Time Partial Week 정보 ( 올해 ~ 차년 )		
			df_in_Time_Partial Week	
				Select (
				Time.[Partial Week].filter(#.Key >= &CurrentMonth.element(0).Key  && #.Key <= &CurrentWeek.element(0).leadoffset(52).Key )    );
				
				
				
			[Time].[Partial Week] 	
			202449A	
			202450A	
			202451A	
			…	
			202649A	
			202650A	
			202651A	
			202652A	

		(Input 5) RTS 정보								
			df_in_MST_RTS							
				Select ([Version].[Version Name]						
				* [Sales Domain].[Ship To]						
				* [Item].[Item] ) on row, 						
				({						
				    Measure.[RTS_ISVALID], 
                    Measure.[RTS_STATUS], 
                    Measure.[RTS_COM_DATE], 
                    Measure.[RTS_DEV_DATE], 
                    Measure.[RTS_INIT_DATE]						
				    }) on column 						
				where { Measure.[RTS_ISVALID] == "Y"} ;						
										
			Version.[Version Name]	Item.[Item]	Sales Domain.[Ship To]	RTS_ISVALID	RTS_STATUS	RTS_INIT_DATE	RTS_DEV_DATE	RTS_COM_DATE
			CWV_DP	RF29BB8600QLAA	300114	Y	COM	2022-02-28	2022-02-28	2022-03-04
			CWV_DP	SM-S911ULIAATT	300114	Y	COM	2022-12-20	2022-12-13	2023-01-20
			CWV_DP	SM-S911UZKEATT	300114	Y	COM	2022-12-20	2022-12-13	2023-01-20
			CWV_DP	SM-S921UZKAAIO	300114	Y	COM	2023-12-12	2024-01-14	2024-01-14
			CWV_DP	SM-S921UZKAATT	300114	Y	COM	2023-12-12	2024-01-14	2024-01-14
			CWV_DP	SM-S921UZKEATT	300114	Y	COM	2023-12-12	2024-01-14	2024-01-14
			CWV_DP	SM-S921UZVAATT	300114	Y	COM	2023-12-12	2024-01-14	2024-01-14
			CWV_DP	SM-S911NLIEKOC	211	Y	COM	2022-12-16	2022-12-16	2023-01-20
			CWV_DP	SM-S911NLIEKOD	211	Y	COM	2022-12-16	2022-12-16	2023-01-20
			CWV_DP	SM-S911NZGEKOC	211	Y	COM	2022-12-16	2022-12-16	2023-01-20
			CWV_DP	SM-S921NZAEKOC	211	Y	COM	2023-12-15	2024-01-13	2024-01-13
			CWV_DP	LH65WAFWLGCXZA	300114	Y	COM	2025-01-13	2025-01-13	2024-12-09
			CWV_DP	BAS-13_GREEN1	300114	Y	COM	2025-01-13	2025-01-13	2024-12-09
			CWV_DP	HA-EOP-YELLOW1	300114	Y	COM	2025-01-13	2025-01-13	2024-12-09
			CWV_DP	QA65QN70FAJXXZ	300114	Y	COM	2025-01-13	2025-01-13	2024-12-09

		(Input 6) MST_ITEM 정보					
			df_in_Item_Master				
				Select (			
				 * [Item].[ProductType]			
				 * [Item].[Item GBM]			
				 * [Item].[Product Group]			
				 * [Item].[Item]			
				);			
							
			Item.[ProductType]	Item.[Item GBM]	Item.[Product Group]	Item.[Item]	
			BAS	MOBILE	MOBILE	BSM-A032F/32D	
			NOR	VD	MONITOR	LH65WAFWLGCXZA	
			NOR	VD	TV	QA65QN70FAJXXZ	
			NOR	SHA	REF	RF29BB8600QLAA	
			NOR	MOBILE	MOBILE	SM-S911NLIEKOC	
			NOR	MOBILE	MOBILE	SM-S911NLIEKOD	
			NOR	MOBILE	MOBILE	SM-S911NZGEKOC	
			NOR	MOBILE	MOBILE	SM-S911ULIAATT	
			NOR	MOBILE	MOBILE	SM-S911UZKEATT	
			NOR	MOBILE	MOBILE	SM-S921NZAEKOC	
			NOR	MOBILE	MOBILE	SM-S921UZKAAIO	
			NOR	MOBILE	MOBILE	SM-S921UZKAATT	
			NOR	MOBILE	MOBILE	SM-S921UZKEATT	
			NOR	MOBILE	MOBILE	SM-S921UZVAATT	
			NOR	BAS	MOBILE	BAS-13_GREEN1	
			NOR	SHA	HAEOP	HA-EOP-YELLOW1	
							
		(Input 7) MST_ITEMTAT정보					
			df_in_Item_TAT				
				Select ([Version].[Version Name]			
				 * [Item].[Item GBM]			
				 * [Item].[Item]			
				 * [Location].[Location] )  on row, 			
				( { Measure.[ITEMTAT TATTERM]} ) on column;			
							
			Version.[Version Name]	Item.[Item GBM]	Item.[Item]	Location.[Location]	ITEMTAT TATTERM
			CWV_DP	VD	QA65QN70FAJXXZ	S614	11
			CWV_DP	VD	QA65QN70FAULXL	S5A3WCB4	2
			CWV_DP	VD	QA65QN70FAUXGH	S712WELC	12
			CWV_DP	VD	QA65QN70FAUXGH	S712WEOC	12
			CWV_DP	VD	QA65QN70FAUXGH	S712WEPG	12
			CWV_DP	VD	QA65QN70FAUXGH	S713WE3N	12
							
		(Input 8) Sales Product ASN 정보				
			df_in_Sales_Product_ASN			
				Select ([Version].[Version Name]		
				 * [Sales Domain].[Ship To] 		
				 * Item.[Item]		
				 )  on row, 	- Location Column 제거	
				( { Measure.[Sales Product ASN] } ) on column;		
						
			Version.[Version Name]	Sales Domain.[Ship To]	Item.[Item]	Sales Product ASN
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5002453	RF29BB8600QLAA	Y
			CWV_DP	A5006941	SM-S911ULIAATT	Y
			CWV_DP	A5006941	SM-S911UZKEATT	Y
			CWV_DP	A5006941	SM-S921UZKAATT	Y
			CWV_DP	A5006941	SM-S921UZKEATT	Y
			CWV_DP	A5006941	SM-S921UZVAATT	Y
			CWV_DP	A5019692	SM-S921UZKAAIO	Y
			CWV_DP	A5019692	SM-S921UZKEATT	Y
			CWV_DP	A5002090	SM-S911NLIEKOC	Y
			CWV_DP	A5002090	SM-S911NLIEKOD	Y
			CWV_DP	A5002090	SM-S911NZGEKOC	Y
			CWV_DP	A5002090	SM-S921NZAEKOC	Y
			CWV_DP	A5002453	LH65WAFWLGCXZA	Y
			CWV_DP	A5002453	LH65WAFWLGCXZA	Y
			CWV_DP	A5002453	BAS-GREEN1	Y
			CWV_DP	A5002453	HA-EOP-YELLOW1	Y
			CWV_DP	A5002453	QA65QN70FAJXXZ	Y

	Output Tables (*)			
				
		(Output 1)		
			df_output_Sell_Out_FCST_GC_Lock	
				Select ([Version].[Version Name]
				* [Item].[Item] 
				* [Sales Domain].[Ship To] 
				* [Location].[Location]
				* Time.[Partial Week] ) on row, 
				( { Measure.[S/Out FCST_GC.Lock],
				 ) on column;
				
		(Output 2)		
			df_output_Sell_Out_FCST_AP2_Lock	
				Select ([Version].[Version Name]
				* [Item].[Item] 
				* [Sales Domain].[Ship To] 
				* [Location].[Location]
				* Time.[Partial Week] ) on row, 
				( { Measure.[S/Out FCST_AP2.Lock],
				 } ) on column;
				
		(Output 3)		
			df_output_Sell_Out_FCST_AP1_Lock	
				Select ([Version].[Version Name]
				* [Item].[Item] 
				* [Sales Domain].[Ship To] 
				* [Location].[Location]
				* Time.[Partial Week] ) on row, 
				( { Measure.[S/Out FCST_AP1.Lock],
				} ) on column;
				
		(Output 4)		
			df_output_Sell_Out_FCST_Color_Condition	
				Select ([Version].[Version Name]
				* [Item].[Item] 
				* [Sales Domain].[Ship To] 
				* [Location].[Location]
				* Time.[Partial Week] ) on row, 
				( { Measure.[S/Out FCST Color Condition]
				} ) on column;


	주요 로직 (*)								
									
									
			Step 1) df_in_MST_RTS 시간 Table 구성						
                Step 1-1) df_in_MST_RTS 전처리						
                                        
                    Item.[Item]	Sales Domain.[Ship To]	RTS_ISVALID	RTS_STATUS	RTS_INIT_DATE	RTS_DEV_DATE	RTS_COM_DATE
                    RF29BB8600QLAA	300114	Y	COM	2022-02-28	2022-02-28	2022-03-04
                    SM-S911ULIAATT	300114	Y	COM	2022-12-20	2022-12-13	2023-01-20
                    SM-S911UZKEATT	300114	Y	COM	2022-12-20	2022-12-13	2023-01-20
                    SM-S921UZKAAIO	300114	Y	COM	2023-12-12	2024-01-14	2024-01-14
                    SM-S921UZKAATT	300114	Y	COM	2023-12-12	2024-01-14	2024-01-14
                    SM-S921UZKEATT	300114	Y	COM	2023-12-12	2024-01-14	2024-01-14
                    SM-S921UZVAATT	300114	Y	COM	2023-12-12	2024-01-14	2024-01-14
                    SM-S911NLIEKOC	211	Y	COM	2022-12-16	2022-12-16	2023-01-20
                    SM-S911NLIEKOD	211	Y	COM	2022-12-16	2022-12-16	2023-01-20
                    SM-S911NZGEKOC	211	Y	COM	2022-12-16	2022-12-16	2023-01-20
                    SM-S921NZAEKOC	211	Y	COM	2023-12-15	2024-01-13	2024-01-13
                    LH65WAFWLGCXZA	300114	Y	COM	2025-01-13	2025-01-13	2024-12-09
                    BAS-13_GREEN1	300114	Y	COM	2025-01-13	2025-01-13	2024-12-09
                    HA-EOP-YELLOW1	300114	Y	COM	2025-01-13	2025-01-13	2024-12-09
                    QA65QN70FAJXXZ	300114	Y	COM	2025-01-13	2025-01-13	2024-12-09
                    * Version.[Version Name] Column 은 삭제한다						
                                        
                                        
                Step 1-2) Time을 Partial Week 으로 변환						
                                        
                    Item.[Item]	Sales Domain.[Ship To]	RTS_STATUS	RTS_INIT_DATE	RTS_DEV_DATE	RTS_COM_DATE	
                    RF29BB8600QLAA	300114	COM	202209A	202209A	202209B	
                    SM-S911ULIAATT	300114	COM	202251A	202250A	202302A	
                    SM-S921UZKEATT	300114	COM	202349A	202402A	202402A	
                    SM-S921UZKAATT	300114	COM	202349A	202402A	202402A	
                    SM-S921UZKAAIO	300114	COM	202349A	202402A	202402A	
                    SM-S921UZVAATT	300114	COM	202349A	202402A	202402A	
                    SM-S911UZKEATT	300114	COM	202251A	202250A	202302A	
                    LH65WAFWLGCXZA	300114	COM	202503A	202503A	202450A	
                    BAS-13_GREEN1	300114	COM	202503A	202503A	202450A	
                    HA-EOP-YELLOW1	300114	COM	202503A	202503A	202450A	
                    SM-S921NZAEKOC	211	COM	202349A	202401A	202401A	
                    SM-S911NZGEKOC	211	COM	202250A	202250A	202302A	
                    SM-S911NLIEKOD	211	COM	202250A	202250A	202302A	
                    SM-S911NLIEKOC	211	COM	202250A	202250A	202302A	
                    * A/B 주차 고려						
                
                Step 1-3) df_in_MST_RTS_EOS 에서 ITEM * Ship To로 새로운 DF 생성				
                                
                                
                    Item.[Item]	Sales Domain.[Ship To]			
                    RF29BB8600QLAA	300114			
                    SM-S911ULIAATT	300114			
                    SM-S921UZKEATT	300114			
                    SM-S921UZKAATT	300114			
                    SM-S921UZKAAIO	300114			
                    SM-S921UZVAATT	300114			
                    SM-S911UZKEATT	300114			
                    LH65WAFWLGCXZA	300114			
                    BAS-13_GREEN1	300114			
                    SM-S921NZAEKOC	211			
                    SM-S911NZGEKOC	211			
                    SM-S911NLIEKOD	211			
                    SM-S911NLIEKOC	211			
                                    
                                    
                Step 1-4) Step1-3의 df에 Partial Week 및 Measure Column 추가				
                                
                    Item.[Item]	Sales Domain.[Ship To]	Time.[Partial Week]	Lock Condition	S/Out FCST Color Condition
                    RF29BB8600QLAA	300114	202252B	True	19_GRAY
                            …	…	…
                            202653A	True	19_GRAY
                    SM-S911ULIAATT	300114	202252B	True	19_GRAY
                            …	…	…
                            202653A	True	19_GRAY
                    SM-S911NLIEKOC	211	202252B	True	19_GRAY
                            …	…	…
                            202653A	True	19_GRAY
                    * df_in_Time_Partial Week를 활용하여 Time.[Partial Week] Column 추가				
                    * Lock Condition Column 추가, True로 일괄 생성				
                    * S/Out FCST Color Condition Column 추가, 19_GRAY로 일괄 생성	

                Step 1-5) RTS Lock 및 Color 반영						
                                        
                    Item.[Item]	Sales Domain.[Ship To]	Time.[Partial Week]	Lock Condition	S/Out FCST Color Condition		
                    RF29BB8600QLAA	300114	202252B	True	19_GRAY		
                            …	…	…		
                            202445	True	15_DARKBLUE		
                            202446	True	15_DARKBLUE		
                            202447		10_LIGHTBLUE		
                            202448		10_LIGHTBLUE		
                            202449		10_LIGHTBLUE		
                            202450		10_LIGHTBLUE		
                            202451		14_WHITE		
                            202452		14_WHITE		
                            …		14_WHITE		
                            202653A		14_WHITE		
                    * COLOR 반영	* Lock 반영	* 반영 구간		* 예시 반영 구간		
                    14_WHITE	False	* MAX(RTS주차, 당주주차) ~ MIN(최대주차) 		202447 ~ 최대주차		
                    15_DARKBLUE	True	* RTS_INIT_DATE ~ (RTS_DEV_DATE or RTS_COM_DATE) - 1 구간		202445 ~ 202446		
                    10_LIGHTBLUE	False, 당주주차 이전이면 True	* RTS_COM_DATE 포함한 이후 4주 주차 구간		202447,202448,202449,202450		
                    * Example						
                    Item.[Item]	Sales Domain.[Ship To]	RTS_STATUS	RTS_ISVALID	RTS_INIT_DATE	RTS_DEV_DATE	RTS_COM_DATE
                    RF29BB8600QLAA	300114	COM	Y	202445	202446	202447
                                        
                                        
			Step 2) 무선 BAS 제품 8주 구간 13_GREEN UPDATE							
                Item.[ProductType]	Item.[Item GBM]	Item.[Product Group]	Item.[Item]	Sales Domain.[Ship To]	Time.[Partial Week]	Lock Condition	S/Out FCST Color Condition
                BAS	MOBILE	MOBILE	BAS-13_GREEN1	300114	…	True	19_GRAY
                                    202446	True	19_GRAY
                                    202447	False	13_GREEN
                                    202448	False	13_GREEN
                                    202449	False	13_GREEN
                                    202450	False	13_GREEN
                                    202451	False	13_GREEN
                                    202452	False	13_GREEN
                                    202453	False	13_GREEN
                                    202454	False	13_GREEN
                                    …	False	
                                    202647	False	
                                    202648	False	
                                    202649	False	
                                    202650	True	
                                    202651	True	19_GRAY
                                    202652	True	19_GRAY
                                    …	True	19_GRAY
                * df_in_Item_Master를 활용하여							
                * Item.[ProductType] = "BAS"							
                * Item.[Item GBM] = "MOBILE"							
                * 위 두 조건에 해당하는 모든 Item에 대해서 당주 주차부터 8주 구간에 대해, 19_GRAY 가 아니면 13_GREEN 처리 , Lock = False							
                * 해당 Item 이 없는 경우 Skip							

			Step 3) VD Lead Time 구간 DARKGRAY UPDATE , MAX 값 사용				
                Step 3-1) df_in_Item_TAT 의 TATTERM 값 생성				
                                
                    Item.[Item GBM]	Item.[Item]	TATTERM		
                    VD	QA65QN70FAJXXZ	11		
                    VD	QA65QN70FAULXL	2		
                    VD	QA65QN70FAUXGH	12		
                                    
                    * Version.[Version Name] 삭제				
                    * Item.[Item GBM] , Item.[Item] 으로 group by 하여 TATTERM 의 값 MAX로 변경				
                                
                Step 3-2) Step 2) 에  VD Lead Time 구간 DARKGRAY RED UPDATE 				
                    Item.[Item]	Sales Domain.[Ship To]	Time.[Partial Week]	Lock Condition	S/Out FCST Color Condition
                    BAS-GREEN1	300114	…	True	19_GRAY
                            202446	True	19_GRAY
                            202447	True	18_DGRAY_RED
                            202448	True	18_DGRAY_RED
                            202449	True	18_DGRAY_RED
                            202450	False	14_WHITE
                            202451	False	14_WHITE
                            …	False	14_WHITE
                            202504	False	14_WHITE
                            202505	False	14_WHITE
                            202506	False	14_WHITE
                            202507	False	14_WHITE
                            202508	False	14_WHITE
                            202509	False	14_WHITE
                            202510	False	14_WHITE
                            …	False	14_WHITE
                    * Step 3-1) 의 ITEM 애 대한 TATTERM 활용				
                    * GBM = 'VD' || 'SHA' 적용				
                    * 당주 주차 기준 + ITEMTAT TATTERM 까지 18_DGRAY_RED 처리				
                    * 당주 주차 기준 + ITEMTAT TATTERM 까지 Lock=False 처리				
                    * 202447 기준, ITEMTAT TATTERM = 3 이면, 202447,202448,202449 처리			
                    * RTS/EOS가 있는 경우 덮어씌우지 않음. Color로 판단 -> 14_WHITE인 경우에만 적용
	

			Step 4) Sales Product ASN 활용하여 최하위 Lv로 Data 생성						
                Step 4-1) Sales Product ASN 전처리						
                    Sales Domain.[Ship To]	Item.[Product Group]	Item.[Item]	Location.[Location]	Time.[Partial Week]	Lock Condition	S/Out FCST Color Condition
                    A5002453		RF29BB8600QLAA	S359WC18	…	True	19_GRAY
                                    202446	True	19_GRAY
                                    202447	False	14_WHITE
                                    202448	False	14_WHITE
                                    …	False	14_WHITE
                                    202707	False	14_WHITE
                                    202708	False	14_WHITE
                                    202707	False	14_WHITE
                                    202709	False	14_WHITE
                                    202710	False	14_WHITE
                                    202711	False	14_WHITE
                    A5002453		RF29BB8600QLAA	S359WC18	…	…	…
                    A5002453		RF29BB8600QLAA	S377	…	…	…
                    A5002453		RF29BB8600QLAA	S376	…	…	…
                    A5002453		RF29BB8600QLAA	S348WDB1	…	…	…
                    A5002453		RF29BB8600QLAA	S359	…	…	…
                    A5002453		RF29BB8600QLAA	S358	…	…	…
                    A5002453		RF29BB8600QLAA	S356	…	…	…
                    A5002453		RF29BB8600QLAA	S348WDD1	…	…	…
                    A5002453		RF29BB8600QLAA	S348WDD2	…	…	…
                    A5002453		RF29BB8600QLAA	S362	…	…	…
                    A5002453		RF29BB8600QLAA	S360	…	…	…
                    A5002453		RF29BB8600QLAA	S366	…	…	…
                    A5002453		RF29BB8600QLAA	S367	…	…	…
                    A5006941		SM-S911ULIAATT	S341	…	…	…
                    A5006941		SM-S911UZKEATT	S341	…	…	…
                    A5006941		SM-S921UZKAATT	S341	…	…	…
                    A5006941		SM-S921UZKEATT	S341	…	…	…
                    A5006941		SM-S921UZVAATT	S341	…	…	…
                    A5019692		SM-S921UZKAAIO	S341	…	…	…
                    A5019692		SM-S921UZKEATT	S341	…	…	…
                    A5002090		SM-S911NLIEKOC	L999	…	…	…
                    A5002090		SM-S911NLIEKOD	L101	…	…	…
                    A5002090		SM-S911NZGEKOC	L999	…	…	…
                    A5002090		SM-S921NZAEKOC	L999	…	…	…
                    A5002453		LH65WAFWLGCXZA	S358	…	…	…
                    A5002453		LH65WAFWLGCXZA	S356	…	…	…
                    A5002453		BAS-GREEN1	S356	…	…	…
                    A5002453		HA-EOP-YELLOW1	S356	…	…	…

                    - df_in_Sales_Product_ASN 에서 Version.[Version Name], Sales Product ASN 삭제						
                    - df_in_Item_Master 로 Item.[Product Group] 정보 추가						
                    * df_in_Time_Partial Week 으로 Time.[Partial Week] Data 추가						
                    * Lock Condition = False., Color = 14_WHITE 로 생성 						
                    * 당주주차 이전 주차에 대해서는 Lock Condition = True, Color = 19_GRAY 적용						
                                        
                Step 4-2) Sales Product ASN에 Partial Week 에 따른 Lock 값과, Color 값 적용 ( Step 4-1)에 Step 3-2)를 적용 )						
                                        
                    Sales Domain.[Ship To]	Item.[Product Group]	Item.[Item]	Location.[Location]	Time.[Partial Week]	Lock Condition	S/Out FCST Color Condition
                    A5002453		RF29BB8600QLAA	S359WC18	…	True	19_GRAY
                                    202446	True	19_GRAY
                                    202447	False	
                                    202448	False	
                                    …	False	
                                    202707	False	
                                    202708	False	
                                    202707	False	
                                    202709	True	19_GRAY
                                    202710	True	19_GRAY
                                    202711	True	19_GRAY
                    Below is logic of PYForecastSellInAndEstoreSellOutLockColor to refer to
                    * Setp7) 에 Step6) Partial Week에 따른 Lock, Color 값을 붙인다.						
                    * Sales Product ASN 에 존재하는 값에 대해서						
                    * Step 6) 의 Item * Ship To 를 기준으로 2LV 하위의 LV7의 값, 3LV 하위의 LV7의 값에 각각 Partial Week에 따른 Lock, Color 값을 반영한다.						
                    * Step 6) 의 Item * Ship To 를 기준으로 2LV 하위의 LV6의 값, 3LV 하위의 LV6의 값에 각각 Partial Week에 따른 Lock, Color 값을 반영한다. 						
                    * Step 6) 의 Sales Domain.[Ship To] 값이 3으로 시작하는 경우						
                        * df_in_Sales_Domain_Dimension의 Sales Domain.[Sales Domain Lv3] = '300114' 를 찾고, 그에 해당하는 Sales Domain.[Sales Domain Lv7] 의 모든 'A5'로 시작하는 값들에 대해서  Step6) 의 Data 값 update					
                        * df_in_Sales_Domain_Dimension의 Sales Domain.[Sales Domain Lv3] = '300114' 를 찾고, 그에 해당하는 Sales Domain.[Sales Domain Lv6] 의 모든 '5'로 시작하는 값들에 대해서  Step6) 의 Data 값 update 					
                    * Step 6) 의 Sales Domain.[Ship To] 값이 2 로 시작하는 경우						
                        * df_in_Sales_Domain_Dimension의 Sales Domain.[Sales Domain Lv2] 에서 찾고, 그에 해당하는 Sales Domain.[Sales Domain Lv7] 의 모든 'A5'로 시작하는 값들에 대해서  Step6) 의 Data 값 update					
                        * df_in_Sales_Domain_Dimension의 Sales Domain.[Sales Domain Lv2] 에서 찾고, 그에 해당하는 Sales Domain.[Sales Domain Lv6] 의 모든 '5'로 시작하는 값들에 대해서  Step6) 의 Data 값 update					
                                        
									
			Step 5) df_in_Sell_Out_Simul_Master Output Data 구성					
                Step 5-1) df_in_Sell_Out_Simul_Master Lock Column 추가 및 값 구성					
                                    
                    Sales Domain.[Ship To]	Item.[Product Group]	S/Out Master Status			
                    A5006941	APS	CON			
                    A5006941	MOBILE	CON			
                    A5006941	PC	CON			
                    A5019692	MOBILE	CON			
                    A5017132	APS	CON			
                    A5017132	MOBILE	CON			
                    A5017132	PC	CON			
                    A5022556	APS	CON			
                    A5022556	MOBILE	CON			
                    A5022556	PC	CON			
                    A5017734	APS	CON			
                    A5017734	MOBILE	CON			
                    A5003117	APS	CON			
                    A5003117	MOBILE	CON			
                    A5003117	PC	CON			
                    A5002458	APS	CON			
                    A5002458	MOBILE	CON			
                    * Version.[Version Name] Column 은 삭제한다.					
                    * S/Out Master Status = 'CON' 값만 사용한다.					
								
								
                Step 5-2) 입력 가능한 7 Level , Item 에 따른 Lock 과 Color 구성					
                    Item.[Product Group]	Item.[Item]	Sales Domain.[Ship To]	Time.[Partial Week]	Lock Condition	S/Out FCST Color Condition
                    MOBILE	SKU 1	A5002346	…	True	19_GRAY
                                202446	True	19_GRAY
                                202447	True	17_DGREY_RED
                                202448	True	17_DGREY_RED
                                202449	True	17_DGREY_RED
                                202450	False	14_WHITE
                                202451	False	14_WHITE
                                …	False	14_WHITE
                                202504	False	14_WHITE
                                202505	False	14_WHITE
                                202506	False	14_WHITE
                                202507	False	14_WHITE
                                202508	False	14_WHITE
                                202509	False	14_WHITE
                                202510	False	14_WHITE
                                …	False	14_WHITE
                    - Step 5-1) 의 Sales Domain.[Ship To] * Item.[Product Group] 에 존재하지 않는 Step 4-2) Data 에 Lock,Color 반영.
                        Lock = True, Color = 19_GRAY			
                                        
                    * Item.[Product Group] Column 에 경우, Step6)에서 사용하지 않는다면 삭제					
                                        
                    - (25.04.15) Master 확정시 변동사항 : S/Out Simul Master 의 최하위 Lv이 Sales Product ASN 과 동일하다는 전제가 필요함. 그렇지 않은 경우, 6Lv, 7Lv 나눠서 반영해주는 로직이 필요함.					

                Step 5-3) 하위 level , Item 에 따른 Lock 과 Color 구성		
                    - Step 5-2) 를 보완 대체 할 예정
                    - Step 5-1) 의 Sales Domain.[Ship To] * Item.[Product Group] 에 존재하지 않는 Step 4-2) Data 에 Lock,Color 반영.
                        Lock = True, Color = 19_GRAY				
                    - 만약 Step 5-1) 의 Sales Domain.[Ship To] 의 level 이 6인것이 있다면 4-2) 의 Sales Domain.[Ship To] 중 같은 level 또는 하위 level 을 찾아서 Lock,Color 반영.

			Step 6) Forecast Rule에 따른 Data 생성			
                - Forecast Rule의 정보를 활용하여, GC FCST, AP2 FCST, AP1 FCST를 각각 구성함.			
                FORECAST_RULE AP2 FCST 를 예시로 진행			
                            
                Step 6-1) Forecast Rule에서 FORECAST_RULE AP2 FCST 정보 추출			
                    Version.[Version Name]	Item.[Product Group]	Sales Domain.[Ship To]	FORECAST_RULE AP2 FCST
                    CWV_DP	REF	300114	5
                    CWV_DP	REF	300115	5
                    CWV_DP	REF	300116	5
                    CWV_DP	REF	300117	5
                    CWV_DP	REF	300118	5
                    CWV_DP	REF	400362	5
                    * GC FCST, AP1 FCST 에 대해서도 동일하게 진행			
						
                Step 6-2) Lock 및 Color Data 를 Forecast Rule 입력 Lv에 맞게 변환			
                    Step 6-2-1) Sales Domain Data 구성						
                        Sales Domain.[Ship To]	Item.[Item]	Time.[Partial Week]	Lock Condition	S/Out FCST Color Condition	GBRULE	FORECAST_RULE AP2 FCST
                        A5002453	Item A	…	True	19_GRAY	5002453	5
                                202446	True	19_GRAY	5002453	5
                                202447	True	17_DGRAY_REDB	5002453	5
                                202448	True	17_DGRAY_REDB	5002453	5
                                202449	True	18_DGRAY_RED	5002453	5
                                202450	False		5002453	5
                                202451	False		5002453	5
                                …	False		5002453	5
                                202504	False		5002453	5
                                202505	False	11_LIGHTRED	5002453	5
                                202506	False	11_LIGHTRED	5002453	5
                                202507	False	11_LIGHTRED	5002453	5
                                202508	False	11_LIGHTRED	5002453	5
                                202509	True	16_DARKRED	5002453	5
                                202510	True	16_DARKRED	5002453	5
                                …	True	19_GRAY	5002453	5
                        - Sales Domain Master Data 참고						
                        1. Shiip To에 해당하는 값을 Sales Domain Master 에서 찾아, 2Lv 값을 가져오고, 해당 값이 Forecast Rule에 있으면, 해당하는 FORECAST_RULE AP2 FCST의 값과 2LV 값은 GBRULE에 각각 Update						
                        2. Shiip To에 해당하는 값을 Sales Domain Master 에서 찾아, 3Lv 값을 가져오고, 해당 값이 Forecast Rule에 있으면, 해당하는 FORECAST_RULE AP2 FCST의 값과 3LV 값은 GBRULE에 각각 Update						
                        2. Shiip To에 해당하는 값을 Sales Domain Master 에서 찾아, 4Lv 값을 가져오고, 해당 값이 Forecast Rule에 있으면, 해당하는 FORECAST_RULE AP2 FCST의 값과 4LV 값은 GBRULE에 각각 Update						
                        3. Shiip To에 해당하는 값을 Sales Domain Master 에서 찾아, 5Lv 값을 가져오고, 해당 값이 Forecast Rule에 있으면, 해당하는 FORECAST_RULE AP2 FCST의 값과 5LV 값은 GBRULE에 각각 Update						
                        4. Shiip To에 해당하는 값을 Sales Domain Master 에서 찾아, 6Lv 값을 가져오고, 해당 값이 Forecast Rule에 있으면, 해당하는 FORECAST_RULE AP2 FCST의 값과 6LV 값은 GBRULE에 각각 Update						
                        5. Shiip To에 해당하는 값을 Sales Domain Master 에서 찾아, 7Lv 값을 가져오고, 해당 값이 Forecast Rule에 있으면, 해당하는 FORECAST_RULE AP2 FCST의 값과 7LV 값은 GBRULE에 각각 Update						
                        * 기존에 찾은 값에 대해서도 하위 순번에 값이 존재하면 Update 진행 -> Forecast Rule 예외조건 반영						
                        * 해당 로직  MultiIndex/Dataframe Merge 활용 여부는 성능 고려, MultiIndex 사용시 FORECAST_RULE_AP2 FCST 값 안가져와도 될 것으로 보임.						


                    Step 6-2-2) GroupBy 진행					
                        Sales Domain.[Ship To]	Item.[Item]	Time.[Partial Week]	S/Out FCST_AP2.Lock	S/Out FCST Color Condition	
                        5002453	Item A	…	True	19_GRAY	
                                202446	True	19_GRAY	
                                202447	True	17_DGRAY_REDB	
                                202448	True	17_DGRAY_REDB	
                                202449	True	18_DGRAY_RED	
                                202450	False		
                                202451	False		
                                …	False		
                                202504	False		
                                202505	False	11_LIGHTRED	
                                202506	False	11_LIGHTRED	
                                202507	False	11_LIGHTRED	
                                202508	False	11_LIGHTRED	
                                202509	True	16_DARKRED	
                                202510	True	16_DARKRED	
                                …	True	19_GRAY	
                        - GBRULE, Item.[Item], Time.[Partial Week], Lock Condition, S/Out FCST Color Condition  으로 Filtering 진행					
                        - GBRULE, Item.[Item], Time.[Partial Week] 으로 Group by 진행 (MIN)					
                        - Rename : GBRULE -> Sales Domain.[Ship To]					
                        - Rename : Lock Condition -> S/Out FCST_AP2.Lock					
                        * Lock Condition 은 AP1, AP2, GC 에 맞는 값 사용					
                        * df_output_Sell_Out_FCST_GC_Lock 생성	Measure.[S/Out FCST_GC.Lock]				
                        * df_output_Sell_Out_FCST_AP2_Lock 생성	Measure.[S/Out FCST_AP2.Lock]				
                        * df_output_Sell_Out_FCST_AP1_Lock 생성	Measure.[S/Out FCST_AP1.Lock]				
								
								
								
			    Step 6-3) S/Out FCST Color Condition Output 생성					
                    Sales Domain.[Ship To]	Item.[Item]	Location.[Location]	Time.[Partial Week]	S/Out FCST Color Condition	
                    5002453	Item A	S356	…	19_GRAY	
                                202446	19_GRAY	
                                202447	17_DGRAY_REDB	
                                202448	17_DGRAY_REDB	
                                202449	18_DGRAY_RED	
                                202450		
                                202451		
                                …		
                                202504		
                                202505	11_LIGHTRED	
                                202506	11_LIGHTRED	
                                202507	11_LIGHTRED	
                                202508	11_LIGHTRED	
                                202509	16_DARKRED	
                                202510	16_DARKRED	
                                …	19_GRAY	
                    - df_output_Sell_Out_FCST_GC_Lock, df_output_Sell_Out_FCST_AP2_Lock, df_output_Sell_Out_FCST_AP1_Lock 에서 Sales Domain.[Ship To], Item.[Item], Time.[Partial Week], S/Out FCST Color Condition 로 Filtering 이후 Merge 진행					
                    - df_output_Sell_Out_FCST_Color_Condition 으로 생성					
								
                Step 6-4) S/Out FCST Output 생성 (4개)					
                    - df_output_Sell_Out_FCST_Color_Condition					
                    * df_output_Sell_Out_FCST_GC_Lock - S/Out FCST Color Condition Column 제거					
                    * df_output_Sell_Out_FCST_AP2_Lock - S/Out FCST Color Condition Column 제거					
                    * df_output_Sell_Out_FCST_AP1_Lock - S/Out FCST Color Condition Column 제거					
                    * df_output에 대하여 모두 Version.[Version Name] = CWV_DP 추가					
                    * df_output에 대하여 모두 Location.[Location] = '-' 추가					
                                        
                                        
                    최종 output 예시					
                                        
                    Version.[Version Name]	Sales Domain.[Ship To]	Item.[Item]	Location.[Location]	Time.[Partial Week]	S/Out FCST_GC.Lock
                    CWV_DP	300114	PC1	-	202252B	True
                    CWV_DP			-	…	…
                    CWV_DP			-	202445	True
                    CWV_DP			-	202446	True
                    CWV_DP			-	202447	
                    CWV_DP			-	202448	
                    CWV_DP			-	202449	
                    CWV_DP			-	202450	
                    CWV_DP			-	202451	
                    CWV_DP			-	202452	
                    CWV_DP			-	…	…
                    CWV_DP			-	202653A	
                    CWV_DP	5006941	PC2	-	202252B	True
                    CWV_DP			-	…	…
                    CWV_DP			-	202445	True
                    CWV_DP			-	202446	True
                    CWV_DP			-	202447	True
                    CWV_DP			-	202448	True
                    CWV_DP			-	202449	True
                    CWV_DP			-	202450	True
                    CWV_DP			-	202451	True
                    CWV_DP			-	202452	True
                    CWV_DP			-	…	…
                    CWV_DP			-	202653A	True

"""

import os,sys
import time,datetime,shutil
import inspect
import traceback
import pandas as pd
import numpy as np
from NSCMCommon import NSCMCommon as common
from NSCMCommon import VDCommon as vdCommon
# from typing_extensions import Literal
import glob
import gc
import math
from typing import Collection, Tuple,Union, Dict
import re
# below package is not for authorized. but use for anlisys. if used for production, this will be removed
# import duckdb


########################################################################################################################
# Local 개발 시에 필요한 공통 변수 선언
########################################################################################################################
# o9에 저장된 instanceName
is_local = common.gfn_get_isLocal()
str_instance = 'PYForecastSellOutLockColor'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"

is_print = True
flag_csv = True
flag_exception = True



########################################################################################################################
# 컬럼상수
########################################################################################################################
# ── column constants ──────────────────────────────────────────────────────────────────────────────────────────
COL_VERSION          = 'Version.[Version Name]'
COL_ITEM             = 'Item.[Item]'
COL_SHIP_TO          = 'Sales Domain.[Ship To]'
COL_RTS_ISVALID      = 'RTS_ISVALID'
COL_RTS_STATUS       = 'RTS_STATUS'
COL_RTS_INIT_DATE    = 'RTS_INIT_DATE'
COL_RTS_DEV_DATE     = 'RTS_DEV_DATE'
COL_RTS_COM_DATE     = 'RTS_COM_DATE'

COL_PROD_TYPE       = 'Item.[ProductType]'  
COL_ITEM_GBM        = 'Item.[Item GBM]'
COL_ITEM_PG         = 'Item.[Product Group]'
COL_PROD_GRP        = 'Item.[Product Group]'
Item_Lv = 'Item_Lv'
RTS_EOS_ShipTo          = 'RTS_EOS_ShipTo'
ForecastRuleShipto      = 'ForecastRuleShipto'
COL_LV2        = 'Sales Domain.[Sales Domain Lv2]'
COL_LV3        = 'Sales Domain.[Sales Domain Lv3]'
COL_LV4        = 'Sales Domain.[Sales Domain Lv4]' 
COL_LV5        = 'Sales Domain.[Sales Domain Lv5]'
COL_LV6        = 'Sales Domain.[Sales Domain Lv6]'
COL_LV7        = 'Sales Domain.[Sales Domain Lv7]'
COL_LOC        = 'Location.[Location]'
Item_Class              = 'ITEMCLASS Class'
COL_TIME_PW            = 'Time.[Partial Week]'
COL_LOCK_COND          = 'Lock Condition'
Lock                    = 'Lock'
SIn_FCST_GC_LOCK                = 'S/In FCST(GI)_GC.Lock'
SIn_FCST_Color_Condition        = 'S/In FCST Color Condition'
SIn_FCST_AP2_LOCK               = 'S/In FCST(GI)_AP2.Lock'
SIn_FCST_AP1_LOCK               = 'S/In FCST(GI)_AP1.Lock'
#

# Salse_Product_ASN       = 'Sales Product ASN'    
COL_SALES_ASN_FLAG               = 'Sales Product ASN'
COL_RAW_TATTERM                 = 'ITEMTAT TATTERM'
COL_TATTERM                     = 'TATTERM'
COL_CW_PLUS_TATTERM             = 'CURWEEK_PLUS_TATTERM'
ITEMTAT_TATTERM_SET             = 'ITEMTAT TATTERM_SET'
COL_RULE_GC           = 'FORECAST_RULE GC FCST'
COL_RULE_AP2          = 'FORECAST_RULE AP2 FCST'
COL_RULE_AP1          = 'FORECAST_RULE AP1 FCST'   
FORECAST_RULE_CUST              = 'FORECAST_RULE CUST FCST'


# ----------------------------------------------------------------
# df_in_Sell_Out_Simul_Master
# ----------------------------------------------------------------
SOut_FCST_Ref                   = 'S/Out Master FCST(Ref.)' 
COL_MASTER_STATUS                     = 'S/Out Master Status'


# ----------------------------------------------------------------
# column constants for step5
# ----------------------------------------------------------------
SOut_FCST_GC_LOCK         = 'S/Out FCST_GC.Lock'
SOut_FCST_AP2_LOCK        = 'S/Out FCST_AP2.Lock'
SOut_FCST_AP1_LOCK        = 'S/Out FCST_AP1.Lock'
COL_COLOR = 'S/Out FCST Color Condition'


# ----------------------------------------------------------------
# Helper column constants for step01_2
# ----------------------------------------------------------------
CURRENT_ROW_WEEK                    = 'WEEK_NUM'
CURRENT_ROW_WEEK_PLUS_8             = 'CURRENTWEEK_NORMALIZED_PLUS_8'   
FILL_I32         = -2**31       # sentinel for “no RTS” rows  (smallest int32)

COL_RTS_WEEK             = 'RTS_WEEK'            # numeric (yyyyww) part of RTS_PARTIAL_WEEK
COL_RTS_PARTIAL_WEEK     = 'RTS_PARTIAL_WEEK'
COL_RTS_INITIAL_WEEK     = 'RTS_INITIAL_WEEK'    # numeric part of RTS_INIT_DATE
COL_RTS_WEEK_MINUS_1     = 'RTS_WEEK_MINUS_1'    # RTS_WEEK - 1
COL_RTS_WEEK_PLUS_3      = 'RTS_WEEK_PLUS_3'     # RTS_WEEK + 3
COL_MAX_RTS_CURRENTWEEK  = 'MAX_RTS_CURRENTWEEK' # max(RTS_WEEK, current week)
COL_MIN_RTS_MAXWEEK      = 'MIN_RTS_MAXWEEK'     # min(RTS_WEEK, max week in calendar)

EOS_INIT_DATE                       = 'EOS_INIT_DATE'
EOS_CHG_DATE                        = 'EOS_CHG_DATE'
EOS_COM_DATE                        = 'EOS_COM_DATE'

EOS_WEEK                            = 'EOS_WEEK_NORMALIZED'
EOS_PARTIAL_WEEK                    = 'EOS_PARTIAL_WEEK'
# EOS_WEEK_MINUS_1                    = 'EOS_WEEK_NORMALIZED_MINUS_1'
EOS_WEEK_MINUS_1         = 'EOS_WEEK_NORMALIZED_MINUS_1'
EOS_WEEK_MINUS_4         = 'EOS_WEEK_NORMALIZED_MINUS_4'
EOS_INITIAL_WEEK         = 'EOS_INITIAL_WEEK_NORMALIZED'
MIN_EOSINI_MAXWEEK                  = 'MIN_EOSINI_MAXWEEK'
EOS_STATUS                          = 'EOS_STATUS'
# Step 10-1 / 10-2 색상 상수
COLOR_DGRAY_RED           = '18_DGRAY_RED'
COLOR_DGRAY_REDB          = '17_DGRAY_REDB'
COLOR_WHITE               = '14_WHITE'
COLOR_LIGHTRED            = '11_LIGHTRED'
COLOR_DARKRED             = '16_DARKRED'
COLOR_YELLOW              = '12_YELLOW'
COLOR_GRAY                = '19_GRAY'
COLOR_GREEN               = '13_GREEN'
COLOR_DARKBLUE            = '15_DARKBLUE'
COLOR_LIGHTBLUE           = '10_LIGHTBLUE'

# ───────────────────────────────────────────────────────────────
# CONSTANT STRING VARIABLES FOR DATAFRAME NAMES
# ───────────────────────────────────────────────────────────────
# input
STR_DF_IN_SELL_OUT_MASTER         = 'df_in_Sell_Out_Simul_Master'
STR_DF_IN_FORECAST_RULE                 = 'df_in_Forecast_Rule'
STR_DF_IN_SALES_DOMAIN_DIM              = 'df_in_Sales_Domain_Dimension'
STR_DF_IN_TIME_PW                       = 'df_in_Time_Partial_Week'
STR_DF_IN_MST_RTS                       = 'df_in_MST_RTS'
STR_DF_IN_ITEM_MASTER                   = 'df_in_Item_Master'
STR_DF_IN_ITEM_TAT                      = 'df_in_Item_TAT'
STR_DF_IN_SALES_PRODUCT_ASN             = 'df_in_Sales_Product_ASN'
# middle
STR_DF_STEP01_1_RTS_CLEAN               = 'df_step01_1_RTS_clean'
STR_DF_STEP01_2_RTS_PW                  = 'df_step01_2_RTS_PW'      # produced in step 1-2
STR_DF_STEP01_3_RTS_DISTINCT            = 'df_step01_3_RTS_distinct'
STR_DF_STEP01_4_RTS_PW_LOCKCOLOR        = 'df_step01_4_RTS_PW_LockColor'  # result of this step
STR_DF_STEP01_5_RTS_LOCKCOLOR_FINAL     = 'df_step01_5_RTS_LockColor_Final'
STR_DF_STEP02_BAS_13_GREEN              = 'df_step02_BAS_13_GREEN'
STR_DF_STEP03_1_TAT_MAX                 = 'df_step03_1_Item_TATTERM'
STR_DF_STEP03_2_VD_LEAD                 = 'df_step03_2_VD_LEADTIME'
STR_DF_STEP04_1_ASN_GRID                = 'df_step04_1_ASN_GRID'
STR_DF_STEP04_2_ASN_LOCKCOLOR           = 'df_step04_2_ASN_LockColor'
STR_DF_STEP05_1_SO_MASTER_LOCK          = 'df_step05_1_SO_Master_Lock'
STR_DF_STEP05_2_GRID_UPD                = 'df_step05_2_Grid_MasterApplied'
STR_DF_STEP05_3_GRID_HIER               = 'df_step05_3_Grid_MasterHier'

########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################
logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
LOG_LEVEL = common.G_log_level


def fn_log_dataframe(df_p_source: pd.DataFrame, str_p_source_name: str) -> None:
    """
    Dataframe 로그 출력 조건 지정 함수
    :param df_p_source: 로그로 찍을 Dataframe
    :param str_p_source_name: 로그로 찍을 Dataframe 명
    :return: None
    """
    is_output = False
    if str_p_source_name.startswith('out_'):
        is_output = True

    if is_print:
        logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1)
        if is_local and not df_p_source.empty and flag_csv:
            # 로컬 Debugging 시 csv 파일 출력
            df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)
    else:
        # 최종 Output 테이블인 경우에는 무조건 로그 출력
        if is_output:
            logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1)
            if is_local and not df_p_source.empty:
                # 로컬 Debugging 시 csv 파일 출력
                df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)

def parse_args():
    # Extract arguments from sys.argv
    args = {}
    for arg in sys.argv[1:]:
        if ':' in arg:
            key, value = arg.split(':', 1)  # Split only on the first ':'
            args[key.strip()] = value.strip()
        else:
            print(f"Warning: Argument '{arg}' does not contain a ':' separator.")
    return args

def _decoration_(func):
    """
    1. 소스 내 함수 실행 시 반복되는 코드를 데코레이터로 변형하여 소스 라인을 줄일 수 있도록 함.
    2. 각 Step을 함수로 실행하는 경우 해당 함수에 뒤따르는 Step log 및 DF 로그, DF 로컬 출력을 데코레이터로 항상 출력하게 함.
    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        # 함수 시작 시각
        tm_start = time.time()
        # 함수 실행
        result = func(*args)
        # 함수 종료 시각
        tm_end = time.time()
        # 함수 실행 시간 로그
        logger.Note(p_note=f'[{func.__name__}] Total time is {tm_end - tm_start:.5f} sec.',
                    p_log_level=LOG_LEVEL.debug())
        # Step log 및 DF 로컬 출력 등을 위한 Keywords 변수 확인
        # Step No
        _step_no = kwargs.get('p_step_no')
        _step_desc = kwargs.get('p_step_desc')
        vdCommon.gfn_pyLog_detail(_step_desc)
        _df_name = kwargs.get('p_df_name')
        _warn_desc = kwargs.get('p_warn_desc')
        _exception_flag = kwargs.get('p_exception_flag')
        # Step log 관련 변수가 입력된 경우 Step log 출력
        if _step_no is not None and _step_desc is not None:
            logger.Step(p_step_no=_step_no, p_step_desc=_step_desc)
        # Warning 메시지가 있는 경우
        if _warn_desc is not None:
            # 함수 실행 결과가 DF이면서 해당 DF가 비어 있는 경우
            if type(result) == pd.DataFrame and result.empty:
                # Exception flag가 확인되고
                if _exception_flag is not None:
                    # Exception flag가 0이면 Warning 로그 출력, 1이면 Exception 발생시킴
                    if _exception_flag == 0:
                        logger.Note(p_note=_warn_desc, p_log_level=LOG_LEVEL.warning())
                    elif _exception_flag == 1:
                        raise Exception(_warn_desc)
        # DF 명이 있는 경우 로그 및 로컬 출력
        if _df_name is not None:
            fn_log_dataframe(result, _df_name)
        return result
    return wrapper


def fn_check_input_table(df_p_source: pd.DataFrame, str_p_source_name: str, str_p_cond: str) -> None:
    """
    Input Table을 체크한 결과를 로그 또는 Exception으로 표시한다.
    :param df_p_source: Input table
    :param str_p_source_name: Name of Input table
    :param str_p_cond: '0' - Exception, '1' - Warning Log
    :return: None
    """
    # Input Table 로그 출력
    logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1)

    if df_p_source.empty:
        if str_p_cond == '0':
            # 테이블이 비어 있는 경우 raise Exception
            raise Exception(f'[Exception] Input table({str_p_source_name}) is empty.')
        else:
            # 테이블이 비어 있는 경우 Warning log
            logger.Note(p_note=f'Input table({str_p_source_name}) is empty.', p_log_level=LOG_LEVEL.warning())


def fn_get_week(list_p_weeks: list, p_row: any) -> list:
    """
    in_Demand의 행과 Time.[Week] 목록을 받아 Time.[Week] - W Demand Build Ahead Limit<= t < Time.[Week]인 t의 목록을 찾아 리턴
    :param list_p_weeks:
    :param p_row:
    :return:
    """
    int_end = int(list_p_weeks.index(p_row['Time.[Week]']))
    int_start = int_end - int(p_row['W Demand Build Ahead Limit'])
    if int_start < 0:
        int_start = 0

    return list_p_weeks[int_start:int_end]

def fn_use_x_after_join(df_source: pd.DataFrame):
    """
    When join , there is 
    """
    df_source.columns = [col.replace('_x', '') if '_x' in col else col for col in df_source.columns]
    # Drop columns with '_y' suffix
    df_source.drop(columns=[col for col in df_source.columns if '_y' in col], inplace=True)
    # df_source = df_source.loc[:, ~df_source.columns.str.endswith('_y')]

def fn_use_y_after_join(df_source: pd.DataFrame):
    """
    When join , there is 
    """
    df_source.columns = [col.replace('_y', '') if '_y' in col else col for col in df_source.columns]
    # Drop columns with '_y' suffix
    df_source.drop(columns=[col for col in df_source.columns if '_x' in col], inplace=True)

# Remove '_x' and '_y' suffixes, keeping '_x' for specified columns
def customize_column_names(df_source: pd.DataFrame, column_use_y: list):
    # Replace '_y' with '' for columns not in column_use_y
    for col in df_source.columns:
        if '_y' in col:
            for col_y in column_use_y:
                if col_y in col:
                    df_source = df_source.rename(columns={col: col.replace('_y', '')})

    # Drop columns with '_x' suffix
    columns_x_to_drop = []
    for col in df_source.columns:
        if '_x' in col:
            for col_y in column_use_y:
                if col_y in col:
                    columns_x_to_drop.append(col)

    df_source.drop(columns=columns_x_to_drop, inplace=True)
    fn_use_x_after_join(df_source)

@_decoration_
def fn_set_header() -> pd.DataFrame:
    """
    MediumWeight로 실행 시 발생할 수 있는 Live Server에서의 오류를 방지하기 위해 Header만 있는 Output 테이블을 만든다.
    :return: DataFrame
        """
    # out_Demand
    df_return = pd.DataFrame(
        {
            COL_VERSION: [], 
            COL_SHIP_TO: [], 
            COL_ITEM: [], 
            COL_LOC: [],
            COL_TIME_PW: [] ,
            SOut_FCST_GC_LOCK : [],
            SOut_FCST_AP2_LOCK : [],
            SOut_FCST_AP1_LOCK : [],
            COL_COLOR : []
        })

    return df_return


@_decoration_
def fn_make_week_list(df_p_source: pd.DataFrame) -> list:
    """
    전처리 - in_Time 테이블에서 Time.[Week]을 오름차순으로 정렬하여 리스트로 변환 후리턴
    :param df_p_source: in_Time
    :return: DataFrame
    """
    # 함수명
    str_my_name = inspect.stack()[0][3]
    
    # 입력 파라미터가 비어 있는 경우 비어 있는 DataFrame을 리턴
    if df_p_source.empty:
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(df_p_source)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    # 오름차순 정렬 후 'Time.[Week]'를 리스트로 변환
    list_return = df_p_source.sort_values(by='Time.[Week]')['Time.[Week]'].to_list()
    
    return list_return

def normalize_week(week_str):
    """Convert a week string with potential suffixes to an integer for comparison."""
    # Remove any non-digit characters (e.g., 'A' or 'B') and convert to integer
    return ''.join(filter(str.isdigit, week_str))

def is_within(current_week, start_week, end_week):
    """
    Check if the current week is within the range defined by start and end weeks.
    """
    return start_week <= current_week <= end_week


################  Start of Functions  ################


################  End of Functions  ################

@_decoration_
def fn_calculate_new_week(df_p_source: pd.DataFrame, list_p_weeks: list) -> pd.DataFrame:
    """
    Step 1-1. in_Demand의 Time.[Week] - W Demand Build Ahead Limit<= t < Time.[Week]인 t를 NewWeek라는 열의 값으로 주고
              W Demand Quantity to SCS 값 복사하여 W Quantity Max Target 생성
    :param df_p_source: in_Demand의
    :param list_p_weeks: in_Time 테이블의 Time.[Week]를 변환한 리스트
    :return: DataFrame
    """
    # 함수명
    str_my_name = inspect.stack()[0][3]
    # Return 변수
    df_return = pd.DataFrame()
    
    # 입력 파라미터가 비어 있는 경우 비어 있는 DataFrame을 리턴
    if df_p_source.empty:
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(df_p_source)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    # 입력 파라미터가 비어 있는 경우 비어 있는 DataFrame을 리턴
    if len(list_p_weeks) == 0:
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(list_p_weeks)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    # 주차가 in_Time의 첫 주보다 큰 경우만 추출
    list_where_first_week = df_p_source['Time.[Week]'] > list_p_weeks[0]
    df_return = df_p_source.loc[list_where_first_week].copy(deep=True)
    # Time.[Week] - W Demand Build Ahead Limit<= t < Time.[Week]의 t에 해당하는 주차를 리스트로 생성
    df_return['NewWeek'] = df_return.apply(lambda x: fn_get_week(list_p_weeks, x), axis=1)
    # 각 행의 주차 리스트 컬럼의 값을 행으로 치환
    df_return = df_return.explode(column='NewWeek')

    # 처리에 필요한 컬럼만 복사
    df_return = df_return[['Item.[Item]', 'Location.[Location]', 'NewWeek', 'W Demand Quantity to SCS']]
    # 죄총 Output 컬럼명으로 변경
    df_return.rename(columns={'NewWeek': 'Time.[Week]', 'W Demand Quantity to SCS': 'W Quantity Max Target'},
                     inplace=True)

    return df_return


@_decoration_
def fn_calculate_max_target(df_p_source: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1-2. Item.[Item], Location.[Location], NewWeek를 Key로 Group By하여
              W Quantity Max Target을 Sum하여 Output 정리
    :param df_p_source: 주차별로 가공된 in_Demand
    :return: DataFrame
    """
    # 함수명
    str_my_name = inspect.stack()[0][3]
    # Return 변수
    df_return = pd.DataFrame()

    # 입력 파라미터가 비어 있는 경우 비어 있는 DataFrame을 리턴
    if df_p_source.empty:
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(df_p_source)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    # Item.[Item], Location.[Location], NewWeek를 Key로 Group By하여 W Quantity Max Target을 Sum하여 Output 정리
    df_return = df_p_source.groupby(
        by=['Item.[Item]', 'Location.[Location]', 'Time.[Week]']).agg({'W Quantity Max Target': 'sum'}
    ).reset_index()

    return df_return


@_decoration_
def fn_output_formatter(df_p_source: pd.DataFrame, str_p_out_version: str) -> pd.DataFrame:
    """
    최종 Output 형태로 정리
    :param df_p_source: 주차별로 가공하여 group by 후 sum을 구한 in_Demand
    :param str_p_out_version: Param_OUT_VERSION
    :return: DataFrame
    """
    # 함수명
    str_my_name = inspect.stack()[0][3]
    # Return 변수
    df_return = pd.DataFrame()

    # 입력 파라미터가 비어 있는 경우 비어 있는 DataFrame을 리턴
    if df_p_source.empty:
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(df_p_source)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    # 입력 파라미터(str_p_out_version)가 비어 있는 경우 경고 메시지를 출력 후 빈 데이터 프레임 리턴
    if str_p_out_version is None or str_p_out_version.strip() == '':
        logger.Note(p_note=f'[{str_my_name}] 입력으로 받은 데이터(str_p_out_version)가 비어 있습니다.',
                    p_log_level=LOG_LEVEL.warning())
        return df_return

    df_return = df_p_source.copy(deep=True)
    df_return[COL_VERSION] = str_p_out_version

    columns_to_return = [
        COL_VERSION,
        COL_SHIP_TO,
        COL_ITEM,
        COL_LOC,
        COL_TIME_PW,
        SOut_FCST_GC_LOCK,
        SOut_FCST_AP2_LOCK,
        SOut_FCST_AP1_LOCK,
        COL_COLOR
    ]

    df_return = df_return[columns_to_return]

    return df_return


def fn_convert_type(df: pd.DataFrame, startWith: str, type):
    for column in df.columns:
        if column.startswith(startWith):
            df[column] = df[column].astype(type)


def find_parent_level(domain_value):
    # Load the CSV file into a DataFrame
    df = input_dataframes[STR_DF_IN_SALES_DOMAIN_DIM]
    
    # Iterate over each row to find the parent
    levels = ['Sales Domain.[Sales Domain LV2]', 'Sales Domain.[Sales Domain LV3]', 
              'Sales Domain.[Sales Domain LV4]', 'Sales Domain.[Sales Domain LV5]', 
              'Sales Domain.[Sales Domain LV6]', 'Sales Domain.[Sales Domain LV7]']
    for index, row in df.iterrows():
        # Check each level to find the domain_value
        for level_index, level in enumerate(levels):
            if domain_value == row[level]:
                # Return the parent value from the previous level
                if level_index > 0:  # Ensure there's a parent level
                    parent_level = levels[level_index - 1]
                    return row[parent_level]
    
    # Return None if no parent is found
    return None



def fn_pre_process_forecast_rules(df_in_Forecast_Rule: pd.DataFrame):
    """
    step 5 에서 사용하기 위해 전처리
    """
    # Fill NaN values and convert types
    df_in_Forecast_Rule['FORECAST_RULE GC FCST'].fillna(0, inplace=True)
    df_in_Forecast_Rule['FORECAST_RULE AP2 FCST'].fillna(0, inplace=True)
    df_in_Forecast_Rule['FORECAST_RULE AP1 FCST'].fillna(0, inplace=True)
    fn_convert_type(df_in_Forecast_Rule, 'FORECAST_RULE GC FCST', 'int32')
    fn_convert_type(df_in_Forecast_Rule, 'FORECAST_RULE AP2 FCST', 'int32')
    fn_convert_type(df_in_Forecast_Rule, 'FORECAST_RULE AP1 FCST', 'int32')

    def start_with(row, column):
        """
        column: 'GC_JOIN', 'AP2_JOIN', 'AP1_JOIN'
        join_value: join with domain master . some time join with lv2 , some time join with lv3
        """
        join_value = str(row['Sales Domain.[Ship To]'])
        join_column = None

        if str(row['Sales Domain.[Ship To]']).startswith('3'):
            join_column = 'Sales Domain.[Sales Domain LV3]'
        elif str(row['Sales Domain.[Ship To]']).startswith('2'):
            join_column = 'Sales Domain.[Sales Domain LV2]'
        else:
            join_column = None
            join_value = None

        if row[column] == 2:
            start_value = '2'
            if str(row['Sales Domain.[Ship To]']).startswith('3'):
                join_column = 'Sales Domain.[Sales Domain LV2]'
                join_value = find_parent_level(join_value)
        elif row[column] == 3:
            start_value = '3'
        elif row[column] == 4:
            start_value = 'A3'
        elif row[column] == 5:
            start_value = '4'
        elif row[column] == 6:
            start_value = '5'
        elif row[column] == 7:
            start_value = 'A5'
        else:
            start_value = None
            join_value = None
            join_column = None

        return start_value, join_column, join_value

    # Apply the function to create new columns
    df_in_Forecast_Rule[['GC_STARTWITH','GC_JOIN_COL', 'GC_JOIN']] = df_in_Forecast_Rule.apply(lambda row: pd.Series(start_with(row, 'FORECAST_RULE GC FCST')), axis=1)
    df_in_Forecast_Rule[['AP2_STARTWITH', 'AP2_JOIN_COL', 'AP2_JOIN']] = df_in_Forecast_Rule.apply(lambda row: pd.Series(start_with(row, 'FORECAST_RULE AP2 FCST')), axis=1)
    df_in_Forecast_Rule[['AP1_STARTWITH', 'AP1_JOIN_COL','AP1_JOIN']] = df_in_Forecast_Rule.apply(lambda row: pd.Series(start_with(row, 'FORECAST_RULE AP1 FCST')), axis=1)

    return df_in_Forecast_Rule


def sanitize_date_string(x: object) -> str:
    """
    * 입력 예
        12/4/2020 12:00:00 AM
        2025-02-03 12:00:00 AM
        2019-09-16
        ''
    * 처리
        ① 공백 앞(= time 부분) 제거  
        ② `-` → `/` 통일  
        ③ 자리수에 따라  
            - YYYY/MM/DD  → 그대로  
            - M/D/YYYY    → 0-padding 후 YYYY/MM/DD 로 변환  
        ④ 실패 시 '' 리턴
    """
    if pd.isna(x) or str(x).strip() == '':
        return ''

    s = str(x).strip()

    # ① 공백(혹은 T) 이후 time 문자열 제거
    s = re.split(r'\s+|T', s, maxsplit=1)[0]

    # ② 구분자 통일
    s = s.replace('-', '/')

    # ③ 날짜 포맷 판별·정규화
    parts = s.split('/')
    try:
        if len(parts) == 3:
            # case-A : YYYY/MM/DD
            if len(parts[0]) == 4:
                y, m, d = parts
            # case-B : M/D/YYYY  또는  MM/DD/YYYY
            else:
                m, d, y = parts
            dt_obj = datetime.datetime(int(y), int(m), int(d))    # 유효성 체크
            return dt_obj.strftime('%Y/%m/%d')              # zero-padding 포함
    except Exception:
        pass       # fall-through → 실패 처리
    return ''       # 파싱 실패

# 벡터라이즈 버전
v_sanitize_date_string = np.vectorize(sanitize_date_string, otypes=[object])

def to_partial_week_datetime(x: Union[str, datetime.date, datetime.datetime]) -> str:
    """
    Robust date-string → 'YYYYWWA/B' converter.
    1. try ``pandas.to_datetime`` (handles *most* inputs fast, incl. numpy64)
    2. fallback to explicit ``strptime`` with the four formats above
    3. log & *raise* if none succeed
    Returns empty-string for ``None`` / '' / NaN.
    """
    _DATE_FMTS = (
        '%Y/%m/%d',   # ① 2025/04/16
        '%Y-%m-%d',   # ② 2025-04-16
        '%m-%d-%Y',   # ③ 04-16-2025
        '%m/%d/%Y',   # ④ 04/16/2025
    )

    if x is None or (isinstance(x, str) and not x.strip()) or pd.isna(x):
        return ''
    # ---------- 1) pandas fast-path ----------
    try:
        dt = pd.to_datetime(x, errors='raise').to_pydatetime()
        return common.gfn_get_partial_week(dt, True)
    except Exception as e_fast:        # noqa: BLE001
        last_exc = e_fast   # remember last exception for logging
    # ---------- 2) explicit strptime fallbacks ----------
    x_str = str(x).strip()
    for fmt in _DATE_FMTS:
        try:
            dt = datetime.datetime.strptime(x_str, fmt)
            return common.gfn_get_partial_week(dt, True)
        except ValueError as exc:
            last_exc = exc              # keep most recent for message
            continue
    # ---------- 3) give up ----------
    msg = f"[to_partial_week_datetime] un-parsable date: {x!r} – last error: {last_exc}"
    logger.Note(p_note=msg, p_log_level=LOG_LEVEL.error())   # or logger.error(...)
    raise ValueError(msg)

# ───── Ship-To → Level LUT ───────────────────────
def build_shipto_level_lut(df_dim: pd.DataFrame):
    """
    Return (pd.Index, np.ndarray[int32], dict) for fast level lookup.
    """
    COL_LVS = [

        (COL_LV7 ,7),
        (COL_LV6, 6), (COL_LV5, 5), (COL_LV4, 4),
        (COL_LV3, 3), (COL_LV2, 2)
    ]
    lut = {}
    for col, lv in COL_LVS:
        lut.update({code: lv for code in df_dim[col].dropna().unique()})
    idx = pd.Index(lut.keys(), dtype=object)
    arr = np.fromiter(lut.values(), dtype='int32')
    return idx, arr, lut

def build_shipto_dim_arrays(df_dim: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
    """
    Returns
    -------
    dim_idx : Index(level-7 ShipTo)
    lv_arrs : ndarray shape(n,6) [LV2 … LV7]
              (컬럼순 : 2,3,4,5,6,7)
    """
    dim_idx = df_dim.set_index(COL_SHIP_TO)
    lv_cols = [COL_LV2, COL_LV3, COL_LV4,
               COL_LV5, COL_LV6, COL_LV7]
    lv_arrs = dim_idx[lv_cols].to_numpy(dtype=object)
    return dim_idx.index, lv_arrs
# -------------------------------------------------

@_decoration_
def fn_process_in_df_mst():
    if is_local: 
        # 로컬인 경우 Output 폴더를 정리한다.
        for file in os.scandir(str_output_dir):
            os.remove(file.path)

        # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
        file_pattern = f"{os.getcwd()}/{str_input_dir}/*.csv" 
        csv_files = glob.glob(file_pattern)

        # file_to_df_mapping = {
        #     # "MST_PSISIMULATIONDEF.csv"                :      "df_in_Sell_Out_Simul_Master"        ,
        #     "df_in_Sell_Out_Simul_Master.csv"           :      str_df_in_Sell_Out_Simul_Master        ,
        #     "df_in_Sales_Domain_Dimension.csv"          :      str_df_in_Sales_Domain_Dimension          ,
        #     "df_in_Time_Partial Week.csv"               :      str_df_in_Time_Partial_Week            ,
        #     # "MST_ITEMCLASS_X_TEST.csv"                :      "df_in_Item_CLASS"                   ,
        #     "MST_ITEMTAT.csv"                           :      str_df_in_Item_TAT                     ,
        #     # "MST_MODELEOS_TEST.csv"                   :      "df_in_MST_EOS"                      ,
        #     "MST_MODELRTS.csv"                          :      str_df_in_MST_RTS                      ,
        #     # "MST_SALESPRODUCT_Y_TEST_LV7.csv"         :      "df_in_Sales_Product_ASN"            ,
        #     "df_in_Forecast_Rule.csv"                   :      str_df_in_Forecast_Rule                ,
        #     "VUI_ITEMATTB.csv"                          :      str_df_in_Item_Master                
        # }

        file_to_df_mapping = {
            'df_in_Sell_Out_Simul_Master.csv'           :      STR_DF_IN_SELL_OUT_MASTER        ,
            'df_in_Forecast_Rule.csv'                   :      STR_DF_IN_FORECAST_RULE              ,
            'df_in_Sales_Domain_Dimension.csv'          :      STR_DF_IN_SALES_DOMAIN_DIM     ,
            'df_in_Time_Partial_Week.csv'               :      STR_DF_IN_TIME_PW          ,
            'df_in_MST_RTS.csv'                         :      STR_DF_IN_MST_RTS                ,
            'df_in_Item_Master.csv'                     :      STR_DF_IN_ITEM_MASTER                ,
            'df_in_Item_TAT.csv'                        :      STR_DF_IN_ITEM_TAT                   ,
            'df_in_Sales_Product_ASN.csv'               :      STR_DF_IN_SALES_PRODUCT_ASN          
        }

        def read_csv_with_fallback(filepath):
            encodings = ['utf-8-sig', 'utf-8', 'cp949']
            
            for enc in encodings:
                try:
                    return pd.read_csv(filepath, encoding=enc)
                except UnicodeDecodeError:
                    continue
            
            raise ValueError(f"Unable to read file {filepath} with tried encodings.")

        # Read all CSV files into a dictionary of DataFrames
        for file in csv_files:
            df = read_csv_with_fallback(file)
            file_name = file.split("/")[-1].split("\\")[-1].split(".")[0]

            for keyword, frame_name in file_to_df_mapping.items():
                if file_name.startswith(keyword.split('.')[0]):
                    input_dataframes[frame_name] = df
                    break

    else:
        # o9 에서 
        input_dataframes[STR_DF_IN_SELL_OUT_MASTER]        = df_in_Sell_Out_Simul_Master
        input_dataframes[STR_DF_IN_FORECAST_RULE]                = df_in_Forecast_Rule
        input_dataframes[STR_DF_IN_SALES_DOMAIN_DIM]       = df_in_Sales_Domain_Dimension
        input_dataframes[STR_DF_IN_TIME_PW]            = df_in_Time_Partial_Week
        input_dataframes[STR_DF_IN_MST_RTS]                      = df_in_MST_RTS
        input_dataframes[STR_DF_IN_ITEM_MASTER]                  = df_in_Item_Master
        input_dataframes[STR_DF_IN_ITEM_TAT]                     = df_in_Item_TAT
        input_dataframes[STR_DF_IN_SALES_PRODUCT_ASN]            = df_in_Sales_Product_ASN

    fn_convert_type(input_dataframes[STR_DF_IN_MST_RTS], 'Sales Domain', str)

    fn_convert_type(input_dataframes[STR_DF_IN_SALES_DOMAIN_DIM], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_IN_MST_RTS], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_IN_SELL_OUT_MASTER], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_IN_FORECAST_RULE], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_IN_SALES_PRODUCT_ASN], 'Sales Domain', str)

    input_dataframes[STR_DF_IN_FORECAST_RULE][COL_RULE_GC].fillna(0, inplace=True)
    input_dataframes[STR_DF_IN_FORECAST_RULE][COL_RULE_AP2].fillna(0, inplace=True)
    input_dataframes[STR_DF_IN_FORECAST_RULE][COL_RULE_AP1].fillna(0, inplace=True)
    input_dataframes[STR_DF_IN_FORECAST_RULE][FORECAST_RULE_CUST].fillna(0, inplace=True)

    fn_convert_type(input_dataframes[STR_DF_IN_FORECAST_RULE], COL_RULE_GC, 'int32')
    fn_convert_type(input_dataframes[STR_DF_IN_FORECAST_RULE], COL_RULE_AP2, 'int32')
    fn_convert_type(input_dataframes[STR_DF_IN_FORECAST_RULE], COL_RULE_AP1, 'int32')
    fn_convert_type(input_dataframes[STR_DF_IN_FORECAST_RULE], FORECAST_RULE_CUST, 'int32')



def analyze_by_lock_for_item():

    analisys = args.get('Param_analisys') 
    if analisys is None or analisys != 'Y':
        return

    """
    This is for analys of below data
    Version.[Version Name]	Sales Domain.[Ship To]	Item.[Item]	Location.[Location]	Time.[Partial Week]	S/Out FCST_GC.Lock	S/Out FCST_AP2.Lock	S/Out FCST_AP1.Lock	S/Out FCST Color Condition
    CWV_DP	300114	AM009KN4DCH/AA	-	202504A	TRUE	TRUE	TRUE	14_WHITE
    CWV_DP	300114	AM009KN4DCH/AA	-	202505A	TRUE	TRUE	TRUE	14_WHITE
    CWV_DP	300114	AM009KN4DCH/AA	-	202505B	TRUE	TRUE	TRUE	14_WHITE
    CWV_DP	300114	AM009KN4DCH/AA	-	202506A	TRUE	TRUE	TRUE	14_WHITE
    CWV_DP	300114	AM009KN4DCH/AA	-	202507A	TRUE	TRUE	TRUE	14_WHITE
    CWV_DP	300114	AM009KN4DCH/AA	-	202508A	TRUE	TRUE	TRUE	14_WHITE
    CWV_DP	300114	AM009KN4DCH/AA	-	202509A	TRUE	TRUE	TRUE	14_WHITE
    CWV_DP	300114	AM009KN4DCH/AA	-	202509B	TRUE	TRUE	TRUE	14_WHITE
    CWV_DP	300114	AM009KN4DCH/AA	-	202510A	TRUE	TRUE	TRUE	14_WHITE
    CWV_DP	300114	AM009KN4DCH/AA	-	202511A	TRUE	TRUE	TRUE	14_WHITE
    CWV_DP	300114	AM009KN4DCH/AA	-	202512A	TRUE	TRUE	TRUE	14_WHITE
    CWV_DP	300114	AM009KN4DCH/AA	-	202513A	TRUE	TRUE	TRUE	14_WHITE

    01. Find Simulation lock
        AM009KN4DCH is declared in df_in_Item_Master by Item_Item
        find Lock in df_in_Sell_Out_Simul_Master.SOut_FCST_Ref

    """
    # from re import X
    # import os,sys,json,shutil,io,zipfile
    # import time
    # import datetime
    # import inspect
    # import traceback
    # import pandas as pd
    # from NSCMCommon import NSCMCommon as common
    # # from typing_extensions import Literal
    # import glob
    # import numpy as np
    # # import rbql
    # import duckdb

    # v_base_dir  = "C:\workspace\Output\PYForecastSellOutLockColor_SHA_REF_20250411_10_52_forAanal"
    v_base_dir  = "C:\workspace\Output\PYForecastSellOutLockColor_SHA_REF_20250414_14_34_applyFalse_1"
    v_output_dir  = f"{v_base_dir}/output"
    v_input_dir = f"{v_base_dir}/input"
    
    v_df = {}
    def read_csv_with_fallback(filepath):
        encodings = ['utf-8-sig', 'utf-8', 'cp949']
        
        for enc in encodings:
            try:
                return pd.read_csv(filepath, encoding=enc)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Unable to read file {filepath} with tried encodings.")

    input_pattern = f"{v_input_dir}/*.csv"
    input_csv_files = glob.glob(input_pattern)
    for file in input_csv_files:
        file_name = file.split("/")[-1].split("\\")[-1].split(".")[0]
        df = read_csv_with_fallback(file)
        v_df[file_name] = df
        # duckdb.register(file_name, df)

    output_pattern = f"{v_output_dir}/*.csv"
    output_csv_files = glob.glob(output_pattern)
    for file in output_csv_files:
        file_name = file.split("/")[-1].split("\\")[-1].split(".")[0]
        df = read_csv_with_fallback(file)
        v_df[file_name] = df
        # duckdb.register(file_name, df)

    fn_convert_type(v_df[STR_DF_IN_SALES_DOMAIN_DIM], 'Sales Domain', str)
    fn_convert_type(v_df[STR_DF_IN_MST_RTS], 'Sales Domain', str)
    fn_convert_type(v_df[STR_DF_IN_SELL_OUT_MASTER], 'Sales Domain', str)
    fn_convert_type(v_df[STR_DF_IN_FORECAST_RULE], 'Sales Domain', str)
    fn_convert_type(v_df[str_df_fn_RTS], 'Sales Domain', str)
    fn_convert_type(v_df[str_df_fn_RTS_Week], 'Sales Domain', str)
    fn_convert_type(v_df[str_df_fn_RTS_Week_Simul], 'Sales Domain', str)
    fn_convert_type(v_df[str_df_fn_RTS_Week_Simul_Forecast], 'Sales Domain', str)

    for df in v_df:
         duckdb.register(df, v_df[df])
    # # Retrieve your DataFrames]
    # file = f"{v_output_dir}/input/df_in_Sales_Product_ASN"
    # df = read_csv_with_fallback(file)
    # duckdb.register(str_df_in_Sales_Product_ASN, df)


    # -----------------------------------
    # Find lock
    # -----------------------------------
    item = 'AM009KN4DCH/AA'
    product_group = 'A/C_SAC'
    

    # -----------------------------------
    # 00.1.  Find df_fn_Sell_Out_Simul_Master By Item
    # -----------------------------------
    query_dec_of_simul = f"""
    SELECT
        im['{COL_ITEM}']                   AS item,
        sim['{COL_ITEM_PG}']              AS Product_Group,
        sim['{COL_SHIP_TO}']        AS Sales_Domain_ShipTo,
        sim['{SOut_FCST_Ref}']              AS SOut_FCST_Ref,  
        sim['{COL_MASTER_STATUS}']                AS SOut_Status

    FROM df_in_Item_Master AS im
    JOIN df_fn_Sell_Out_Simul_Master AS sim ON 
        sim['{COL_ITEM_PG}'] = im['{COL_ITEM_PG}']
    WHERE 1==1
    and im['{COL_ITEM}']  = '{item}'
    order by im['{COL_ITEM}'],sim['{COL_ITEM_PG}'], sim['{COL_SHIP_TO}']
    limit 15
    """

    result_dec_of_simul = duckdb.query(query_dec_of_simul).to_df()
    print(result_dec_of_simul)

    """
                item Product_Group Sales_Domain_ShipTo SOut_FCST_Ref SOut_Status
    0  AM009KN4DCH/AA       A/C_SAC            A5018757          GSBN         CON
    1  AM009KN4DCH/AA       A/C_SAC            A5020721          GSBN         CON
    2  AM009KN4DCH/AA       A/C_SAC            A5021969         EXCEL         CON
    3  AM009KN4DCH/AA       A/C_SAC            A5022277          GSBN         CON

    """



    # -----------------------------------
    # 00.2.  Find df_fn_Sell_Out_Simul_Master  By Product_Group
    # -----------------------------------

    query_dec_of_simul_pg = f"""
    SELECT
        im['{COL_ITEM}']                   AS item,
        sim['{COL_ITEM_PG}']              AS Product_Group,
        sim['{COL_SHIP_TO}']        AS Sales_Domain_ShipTo,
        sim['{SOut_FCST_Ref}']              AS SOut_FCST_Ref,  
        sim['{COL_MASTER_STATUS}']                AS SOut_Status

    FROM df_in_Item_Master AS im
    JOIN df_fn_Sell_Out_Simul_Master AS sim ON 
        sim['{COL_ITEM_PG}'] = im['{COL_ITEM_PG}']
    WHERE 1==1
    and sim['{COL_ITEM_PG}']  = '{product_group}'
    order by im['{COL_ITEM}'],sim['{COL_ITEM_PG}'], sim['{COL_SHIP_TO}']
    limit 15
    """

    result_dec_of_simul_pg = duckdb.query(query_dec_of_simul_pg).to_df()
    print(result_dec_of_simul_pg)

    """
                item Product_Group Sales_Domain_ShipTo SOut_FCST_Ref SOut_Status
    0   4EEVAKA40K1025       A/C_SAC            A5018757          GSBN         CON
    1   4EEVAKA40K1025       A/C_SAC            A5020721          GSBN         CON
    2   4EEVAKA40K1025       A/C_SAC            A5021969         EXCEL         CON
    3   4EEVAKA40K1025       A/C_SAC            A5022277          GSBN         CON
    4   4EEVAKA40K1050       A/C_SAC            A5018757          GSBN         CON
    5   4EEVAKA40K1050       A/C_SAC            A5020721          GSBN         CON
    6   4EEVAKA40K1050       A/C_SAC            A5021969         EXCEL         CON
    7   4EEVAKA40K1050       A/C_SAC            A5022277          GSBN         CON
    8   4EEVAKA64K1075       A/C_SAC            A5018757          GSBN         CON
    9   4EEVAKA64K1075       A/C_SAC            A5020721          GSBN         CON
    10  4EEVAKA64K1075       A/C_SAC            A5021969         EXCEL         CON
    11  4EEVAKA64K1075       A/C_SAC            A5022277          GSBN         CON
    12  4EEVAKA64K1100       A/C_SAC            A5018757          GSBN         CON
    13  4EEVAKA64K1100       A/C_SAC            A5020721          GSBN         CON
    14  4EEVAKA64K1100       A/C_SAC            A5021969         EXCEL         CON


    """


    # -----------------------------------
    # 01.  Find RTS lock
    # -----------------------------------
    query_item_to_rts_week = f"""
    SELECT
        im['{COL_ITEM}'] AS item,
        r['{COL_ITEM_PG}'] AS Product_Group,
        r_wk['{COL_SHIP_TO}']       AS Sales_Domain_ShipTo,
        r_wk['{COL_TIME_PW}']           AS Partial_Week,
        r_wk['{Lock}']                    AS Lock,
        r_wk['{SIn_FCST_Color_Condition}']  AS Color,
        fr['{COL_RULE_GC}'] AS GC ,
        fr['{COL_RULE_AP2}'] AS AP2 ,
        fr['{COL_RULE_AP1}'] AS AP1
    FROM df_in_Item_Master AS im
    JOIN df_fn_RTS r ON 
        r['{COL_ITEM_PG}'] = im['{COL_ITEM_PG}']
        and r['{COL_ITEM}'] = im['{COL_ITEM}']
    JOIN df_fn_RTS_Week AS r_wk ON
        r_wk['{COL_ITEM_PG}'] = r['{COL_ITEM_PG}']
        and r_wk['{COL_ITEM}'] = r['{COL_ITEM}']
        and r_wk['{COL_SHIP_TO}'] = r['{COL_SHIP_TO}']
    JOIN df_in_Forecast_Rule AS fr ON
        fr['{COL_ITEM_PG}'] = r['{COL_ITEM_PG}']
        and fr['{COL_SHIP_TO}'] = r['{COL_SHIP_TO}'] 
        
    WHERE 1==1
    and im['{COL_ITEM}']  = '{item}'
    and r['{COL_SHIP_TO}'] = '300114'
    -- and r_wk['{COL_TIME_PW}'] in ('202504A','202505A','202506A')
    order by im['{COL_ITEM}'], r_wk['{COL_SHIP_TO}'],r_wk['{COL_TIME_PW}']
    limit 15
    """

    result_item_to_rts_week = duckdb.query(query_item_to_rts_week).to_df()
    print(result_item_to_rts_week)

    """
    결과는 아래와 같다. 
    Lock Flase.
                item Product_Group  Sales_Domain_ShipTo Partial_Week   Lock     Color  GC  AP2  AP1
    0   AM009KN4DCH/AA       A/C_SAC               300114      202504A  False  14_WHITE   3    5    6
    1   AM009KN4DCH/AA       A/C_SAC               300114      202505A  False  14_WHITE   3    5    6
    2   AM009KN4DCH/AA       A/C_SAC               300114      202505B  False  14_WHITE   3    5    6
    3   AM009KN4DCH/AA       A/C_SAC               300114      202506A  False  14_WHITE   3    5    6
    4   AM009KN4DCH/AA       A/C_SAC               300114      202507A  False  14_WHITE   3    5    6
    5   AM009KN4DCH/AA       A/C_SAC               300114      202508A  False  14_WHITE   3    5    6
    6   AM009KN4DCH/AA       A/C_SAC               300114      202509A  False  14_WHITE   3    5    6
    7   AM009KN4DCH/AA       A/C_SAC               300114      202509B  False  14_WHITE   3    5    6
    8   AM009KN4DCH/AA       A/C_SAC               300114      202510A  False  14_WHITE   3    5    6
    9   AM009KN4DCH/AA       A/C_SAC               300114      202511A  False  14_WHITE   3    5    6
    10  AM009KN4DCH/AA       A/C_SAC               300114      202512A  False  14_WHITE   3    5    6
    11  AM009KN4DCH/AA       A/C_SAC               300114      202513A  False  14_WHITE   3    5    6
    12  AM009KN4DCH/AA       A/C_SAC               300114      202514A  False  14_WHITE   3    5    6
    13  AM009KN4DCH/AA       A/C_SAC               300114      202514B  False  14_WHITE   3    5    6
    14  AM009KN4DCH/AA       A/C_SAC               300114      202515A  False  14_WHITE   3    5    6



    그래서 결과는 False 이어야 하는데 


    """

    # -----------------------------------
    # 02.  Find Lock For df_fn_RTS_Week_Simul
    # -----------------------------------

    query_item_to_simul = f"""
    SELECT
        im['{COL_ITEM}'] AS item,
        so['{COL_ITEM_PG}'] AS Product_Group,
        so['{COL_SHIP_TO}']       AS Sales_Domain_ShipTo_Simul,
        so['{SOut_FCST_Ref}']             AS SOut_FCST_Ref,
        sof['{COL_LV2}']          AS Sales_Domain_LV2,
        sof['{COL_LV3}']          AS Sales_Domain_LV3,
        sof['{Lock}']                     AS Lock_of_Simul,
        r_wk_s['{COL_TIME_PW}']           AS Partial_Week,
        -- r_wk['{Lock}']                    AS Lock_of_RTSWeek,
        r_wk_s['{Lock}']                    AS Lock_of_SimulWeek,
    FROM df_in_Item_Master AS im
    JOIN df_in_Sell_Out_Simul_Master AS so
      ON im['{COL_ITEM_PG}'] = so['{COL_ITEM_PG}']
    JOIN df_fn_Sell_Out_Simul_Master AS sof ON
        sof['{COL_SHIP_TO}'] = so['{COL_SHIP_TO}']
        and sof['{COL_ITEM_PG}'] = so['{COL_ITEM_PG}']
    JOIN df_fn_RTS_Week_Simul AS r_wk_s ON
        r_wk_s['{COL_ITEM_PG}'] = sof['{COL_ITEM_PG}']
        and r_wk_s['{COL_SHIP_TO}'] = sof['{COL_SHIP_TO}']
        and r_wk_s['{COL_ITEM}'] = im['{COL_ITEM}']
         
    WHERE 1==1
    and im['{COL_ITEM}']  = '{item}'
    and r_wk_s['{COL_TIME_PW}'] in ('202504A','202505A','202506A')
    order by im['{COL_ITEM}'], so['{COL_SHIP_TO}'],r_wk_s['{COL_TIME_PW}']
    limit 15
    """

    # Execute the DuckDB query in-memory and fetch as a pandas DataFrame
    result_item_to_simul = duckdb.query(query_item_to_simul).to_df()
    print(result_item_to_simul)

    """
                item Product_Group Sales_Domain_ShipTo_Simul SOut_FCST_Ref  Sales_Domain_LV2  Sales_Domain_LV3  Lock_of_Simul Partial_Week  Lock_of_SimulWeek
    0   AM009KN4DCH/AA       A/C_SAC                  A5018757          GSBN               203            300114           True      202504A                1.0
    1   AM009KN4DCH/AA       A/C_SAC                  A5018757          GSBN               203            300114           True      202505A                1.0
    2   AM009KN4DCH/AA       A/C_SAC                  A5018757          GSBN               203            300114           True      202506A                1.0
    3   AM009KN4DCH/AA       A/C_SAC                  A5020721          GSBN               203            300114           True      202504A                1.0
    4   AM009KN4DCH/AA       A/C_SAC                  A5020721          GSBN               203            300114           True      202505A                1.0
    5   AM009KN4DCH/AA       A/C_SAC                  A5020721          GSBN               203            300114           True      202506A                1.0
    6   AM009KN4DCH/AA       A/C_SAC                  A5021969         EXCEL               203            300114          False      202504A                0.0
    7   AM009KN4DCH/AA       A/C_SAC                  A5021969         EXCEL               203            300114          False      202505A                0.0
    8   AM009KN4DCH/AA       A/C_SAC                  A5021969         EXCEL               203            300114          False      202506A                0.0
    9   AM009KN4DCH/AA       A/C_SAC                  A5022277          GSBN               203            300114           True      202504A                1.0
    10  AM009KN4DCH/AA       A/C_SAC                  A5022277          GSBN               203            300114           True      202505A                1.0
    11  AM009KN4DCH/AA       A/C_SAC                  A5022277          GSBN               203            300114           True      202506A                1.0

    위의 결과로 보건데 , 

    """

    # -----------------------------------
    # 03.  Find Lock For df_fn_RTS_Week_Simul_Forecast . For Level3 
    # -----------------------------------

    lv7_val = 'A5021969'
    dim_level = COL_LV7
    query_item_to_forcast = f"""
    SELECT
        im['{COL_ITEM}'] AS item,
        so['{COL_ITEM_PG}'] AS Product_Group,
        so['{COL_SHIP_TO}']       AS Sales_Domain_ShipTo_Simul,
        fcst['{COL_SHIP_TO}']       AS Sales_Domain_ShipTo_Fcst,
        so['{SOut_FCST_Ref}']             AS SOut_FCST_Ref,
        sof['{COL_LV2}']          AS Sales_Domain_LV2,
        sof['{COL_LV3}']          AS Sales_Domain_LV3,
        sof['{Lock}']                     AS Lock_of_Simul,
        r_wk_s['{COL_TIME_PW}']           AS Partial_Week,
        r_wk_s['{Lock}']                    AS Lock_of_SimulWeek,
        fcst['{SOut_FCST_GC_LOCK}']                    AS GC,
        fcst['{SOut_FCST_AP2_LOCK}']                    AS AP2,
        fcst['{SOut_FCST_AP1_LOCK}']                    AS AP1
    FROM df_in_Item_Master AS im
    JOIN df_in_Sell_Out_Simul_Master AS so
      ON im['{COL_ITEM_PG}'] = so['{COL_ITEM_PG}']
    JOIN df_fn_Sell_Out_Simul_Master AS sof ON
        sof['{COL_SHIP_TO}'] = so['{COL_SHIP_TO}']
        and sof['{COL_ITEM_PG}'] = so['{COL_ITEM_PG}']
    JOIN df_fn_RTS_Week_Simul AS r_wk_s ON
        r_wk_s['{COL_ITEM_PG}'] = sof['{COL_ITEM_PG}']
        and r_wk_s['{COL_SHIP_TO}'] = sof['{COL_SHIP_TO}']
        and r_wk_s['{COL_ITEM}'] = im['{COL_ITEM}']
    JOIN df_in_Sales_Domain_Master AS dim  ON
        dim['{COL_SHIP_TO}'] = sof['{COL_SHIP_TO}']
    JOIN df_fn_RTS_Week_Simul_Forecast fcst ON         
        fcst['{COL_SHIP_TO}'] = dim['{dim_level}']
        and fcst['{COL_ITEM}'] = r_wk_s['{COL_ITEM}']
        and fcst['{COL_TIME_PW}'] = r_wk_s['{COL_TIME_PW}']
        
    WHERE 1==1
    and im['{COL_ITEM}']  = '{item}'
    -- and r_wk_s['{COL_TIME_PW}'] in ('202504A','202505A','202506A')
    and so['{COL_SHIP_TO}'] = '{lv7_val}'
    order by im['{COL_ITEM}'], so['{COL_SHIP_TO}'],r_wk_s['{COL_TIME_PW}']
    limit 15
    """

    # Execute the DuckDB query in-memory and fetch as a pandas DataFrame
    result_item_to_forcast = duckdb.query(query_item_to_forcast).to_df()
    print(result_item_to_forcast)

    """
                item Product_Group Sales_Domain_ShipTo_Simul Sales_Domain_ShipTo_Fcst SOut_FCST_Ref  Sales_Domain_LV2  ...  Lock_of_Simul  Partial_Week Lock_of_SimulWeek    GC   AP2   AP1
    0   AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202504A               1.0  True  True  True
    1   AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202505A               1.0  True  True  True
    2   AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202505B               1.0  True  True  True
    3   AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202506A               1.0  True  True  True
    4   AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202507A               1.0  True  True  True
    5   AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202508A               1.0  True  True  True
    6   AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202509A               1.0  True  True  True
    7   AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202509B               1.0  True  True  True
    8   AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202510A               1.0  True  True  True
    9   AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202511A               1.0  True  True  True
    10  AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202512A               1.0  True  True  True
    11  AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202513A               1.0  True  True  True
    12  AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202514A               1.0  True  True  True
    13  AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202514B               1.0  True  True  True
    14  AM009KN4DCH/AA       A/C_SAC                  A5022277                 A5022277          GSBN               203  ...           True       202515A               1.0  True  True  True

    [15 rows x 13 columns]



    """


################################################################################################################
# ──[ 1-1 ]  Pre-process RTS master  ───────────────────────────────────────────────────────────────────────────
# Goal : • keep only rows where RTS_ISVALID == 'Y'
#        • drop the Version column
#        • return a clean dataframe for later steps
################################################################################################################
@_decoration_   # <-- uses the same logging decorator as the rest of the code-base
def fn_step01_1_preprocess_rts():
    """
    Step 1-1 – Clean RTS master (df_in_MST_RTS).

    * keeps only RTS_ISVALID == 'Y'
    * drops the Version column
    * returns the cleaned dataframe
    """
    # read the global input table
    df_source = input_dataframes[STR_DF_IN_MST_RTS].copy(deep=True)

    # 1) filter by validity flag
    df_filtered = df_source[df_source[COL_RTS_ISVALID] == 'Y']

    # 2) drop the Version column if present
    if COL_VERSION in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=[COL_VERSION])

    # (optional) re-order columns for readability
    ordered_cols = [
        COL_ITEM,
        COL_SHIP_TO,
        COL_RTS_ISVALID,
        COL_RTS_STATUS,
        COL_RTS_INIT_DATE,
        COL_RTS_DEV_DATE,
        COL_RTS_COM_DATE,
    ]
    df_filtered = df_filtered[ordered_cols]

    return df_filtered


################################################################################################################
# ──[ 1-2 ]  Convert RTS dates → Partial-Week and helper columns  ──────────────────────────────────────────────
################################################################################################################
# Requires:
#   • df_step01_1_RTS_clean already stored in `output_dataframes`
#   • helpers:  v_sanitize_date_string(…)   to_partial_week_datetime(…)
#   • globals : current_week_normalized, max_week_normalized
#   • common.gfn_add_week(week_str, offset)
################################################################################################################import numpy as np        # (already imported elsewhere in the code-base)
@_decoration_
def fn_step01_2_convert_rts_dates_to_pw():
    """
    Step 1-2 – Convert the three RTS date columns to partial-week codes (YYYYWW[A|B])
    and create helper week-number columns used by later rules.
    """
    # ── source ────────────────────────────────────────────────────────────────────
    df_rts = output_dataframes[STR_DF_STEP01_1_RTS_CLEAN].copy(deep=True)

    # ── 1)  clean & convert each date column ─────────────────────────────────────
    date_cols = [COL_RTS_INIT_DATE, COL_RTS_DEV_DATE, COL_RTS_COM_DATE]

    for col in date_cols:
        # (a) sanitize malformed strings like '0000-00-00'
        df_rts[col] = v_sanitize_date_string(df_rts[col])

        # (b) convert → partial-week (returns '' if NaT)
        df_rts[col] = df_rts[col].apply(
            lambda x: to_partial_week_datetime(x) if pd.notna(x) and x != '' else ''
        ).astype(str)

    # ── 2)  decide the representative RTS partial-week per row ──────────────────
    #       COM  ⇒ use COM date
    #       else if DEV exists ⇒ DEV
    #       else               ⇒ INIT
    df_rts[COL_RTS_PARTIAL_WEEK] = np.where(
        df_rts[COL_RTS_STATUS] == 'COM',
        df_rts[COL_RTS_COM_DATE],
        np.where(
            df_rts[COL_RTS_DEV_DATE] != '',
            df_rts[COL_RTS_DEV_DATE],
            df_rts[COL_RTS_INIT_DATE]
        )
    )

    # ── 3)  helper numeric week strings (strip the A/B tag) ─────────────────────
    df_rts[COL_RTS_WEEK]         = df_rts[COL_RTS_PARTIAL_WEEK].str.replace(r'\D', '', regex=True)
    df_rts[COL_RTS_INITIAL_WEEK] = df_rts[COL_RTS_INIT_DATE].str.replace(r'\D', '', regex=True)

    # ── 4)  boundary-related helpers used in lock/color logic later ─────────────
    df_rts[COL_MAX_RTS_CURRENTWEEK] = df_rts[COL_RTS_WEEK].apply(
        lambda w: max(w, current_week_normalized) if w else ''
    )
    df_rts[COL_MIN_RTS_MAXWEEK] = df_rts[COL_RTS_WEEK].apply(
        lambda w: min(w, max_week_normalized) if w else ''
    )
    df_rts[COL_RTS_WEEK_MINUS_1] = df_rts[COL_RTS_WEEK].apply(
        lambda w: common.gfn_add_week(w, -1) if w else ''
    )
    df_rts[COL_RTS_WEEK_PLUS_3] = df_rts[COL_RTS_WEEK].apply(
        lambda w: common.gfn_add_week(w, 3) if w else ''
    )

    return df_rts


################################################################################################################
# ──[ 1-3 ]  Distinct Item × Ship-To list  ─────────────────────────────────────────────────────────────────────
# Goal : from the RTS data prepared in step 1-2, extract the unique pairs
#        Item.[Item]  ×  Sales Domain.[Ship To]
################################################################################################################# ── dataframe-name constants ───────────────────────────────────────────────────
# ── column constants (already declared earlier in the module) ─────────────────
# COL_ITEM    = 'Item.[Item]'
# COL_SHIP_TO = 'Sales Domain.[Ship To]'

@_decoration_
def fn_step01_3_extract_item_shipto_distinct():
    """
    Step 1-3 – Create a dataframe of distinct (Item, Ship-To) pairs.

    Input : df_step01_2_RTS_PW  (RTS data with partial-week columns)
    Output: df_step01_3_RTS_distinct with two columns:
            • Item.[Item]
            • Sales Domain.[Ship To]
    """
    # ── fetch the source dataframe ─────────────────────────────────────────────
    df_rts_pw = output_dataframes[STR_DF_STEP01_2_RTS_PW].copy(deep=True)

    # ── keep only the two columns we need & drop duplicates ───────────────────
    df_dist = (
        df_rts_pw[[COL_ITEM, COL_SHIP_TO]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # (optional) sort for stable output
    df_dist = df_dist.sort_values(by=[COL_SHIP_TO, COL_ITEM]).reset_index(drop=True)

    return df_dist

################################################################################################################
# ──[ 1-4 ]  Cartesian with Time-calendar → initial Lock / Color frame  ────────────────────────────────────────
# Goal : For every (Item, Ship-To) pair (step 1-3) create rows for **all** partial-weeks in the
#        input calendar.  Add default columns:
#           • Lock_Condition              = True
#           • S/Out FCST Color Condition  = '19_GRAY'
################################################################################################################
# Requires:
#   • df_step01_3_RTS_distinct   – produced in step 1-3
#   • df_in_Time_PW              – 1-column calendar table provided as input
################################################################################################################# ── dataframe-name constants ───────────────────────────────────────────────────
# STR_DF_IN_TIME_PW               = 'df_in_Time_Partial Week'        # input
# STR_DF_STEP01_3_RTS_DISTINCT    = 'df_step01_3_RTS_distinct'       # from previous step
# STR_DF_STEP01_4_RTS_PW_LOCKCOLOR = 'df_step01_4_RTS_PW_LockColor'  # result of this step

# ── column constants ──────────────────────────────────────────────────────────
# COL_TIME_PW            = 'Time.[Partial Week]'
# COL_LOCK_COND          = 'Lock Condition'
# COL_COLOR              = 'S/Out FCST Color Condition'

@_decoration_
def fn_step01_4_cartesian_with_calendar():
    """
    Step 1-4 – Produce the complete (Item, Ship-To, Partial-Week) matrix with
    default lock & color settings.
    """
    # source 1 : distinct Item × Ship-To pairs
    df_pairs = output_dataframes[STR_DF_STEP01_3_RTS_DISTINCT].copy(deep=True)

    # source 2 : full partial-week calendar
    df_time  = input_dataframes[STR_DF_IN_TIME_PW].copy(deep=True)   # one column: Time.[Partial Week]

    # -- Cartesian join --------------------------------------------------------
    # add a dummy key to both, merge on that key, then drop the key
    df_pairs['key'] = 1
    df_time ['key'] = 1
    df_cart = df_pairs.merge(df_time, on='key', how='inner').drop(columns=['key'])

    # -- add default measure columns ------------------------------------------
    df_cart[COL_LOCK_COND] = True
    df_cart[COL_COLOR]     = COLOR_GRAY

    # optional sort (Ship-To, Item, Week)
    df_cart = df_cart.sort_values(
        by=[COL_SHIP_TO, COL_ITEM, COL_TIME_PW]
    ).reset_index(drop=True)

    return df_cart

################################################################################################################
# ──[ 1-5 ]  Apply RTS-based Lock / Color rules  ───────────────────────────────────────────────────────────────
#  * vectorised lookup (no large merge) – memory-friendly
################################################################################################################
# prerequisites already stored in `output_dataframes`
#   • STR_DF_STEP01_4_RTS_PW_LOCKCOLOR   (“calendar grid” with default True/19_GRAY)
#   • STR_DF_STEP01_2_RTS_PW             (RTS helper columns incl. weeks)
################################################################################################################# ── dataframe name constants ─────────────────────────────────────────────────────────────────────────────────
# STR_DF_STEP01_4_RTS_PW_LOCKCOLOR   = 'df_step01_4_RTS_PW_LockColor'
# STR_DF_STEP01_2_RTS_PW             = 'df_step01_2_RTS_PW'
# STR_DF_STEP01_5_RTS_LOCKCOLOR_FINAL= 'df_step01_5_RTS_LockColor_Final'

# ── column constants (re-use previous ones) ─────────────────────────────────────────────────────────────────
# COL_ITEM               = 'Item.[Item]'
# COL_SHIP_TO            = 'Sales Domain.[Ship To]'
# COL_TIME_PW            = 'Time.[Partial Week]'
# COL_LOCK_COND          = 'Lock Condition'
# COL_COLOR              = 'S/Out FCST Color Condition'

# helper columns coming from step 1-2
# COL_MAX_RTS_CURRENTWEEK= 'MAX_RTS_CURRENTWEEK'
# COL_MIN_RTS_MAXWEEK    = 'MIN_RTS_MAXWEEK'
# COL_RTS_INITIAL_WEEK   = 'RTS_INITIAL_WEEK'
# COL_RTS_WEEK_MINUS_1   = 'RTS_WEEK_MINUS_1'
# COL_RTS_WEEK           = 'RTS_WEEK'
# COL_RTS_WEEK_PLUS_3    = 'RTS_WEEK_PLUS_3'

# colour palette
# COLOR_DARKBLUE  = '15_DARKBLUE'
# COLOR_LIGHTBLUE = '10_LIGHTBLUE'
# COLOR_WHITE     = '14_WHITE'

# CURRENT_ROW_WEEK = 'WEEK_NUM'   # temp helper for vectorised comparisons
# FILL_I32         = -2**31       # sentinel for “no RTS” rows  (smallest int32)

@_decoration_
def fn_step01_5_apply_rts_rules() -> pd.DataFrame:
    """
    Step 1-5: take the default grid from step 1-4 and overwrite Lock/Colour
    according to RTS timelines (Dark-blue, Light-blue, White).
    Vectorised - no huge merge; only small lookup arrays are materialised.
    """
    # ------------------------------------------------------------------ 1) load sources
    df_grid  = output_dataframes[STR_DF_STEP01_4_RTS_PW_LOCKCOLOR]      # **read-only**
    df_rts   = output_dataframes[STR_DF_STEP01_2_RTS_PW]                # helper weeks

    # ------------------------------------------------------------------ 2) compress RTS -> unique key
    # keep only the first (any) row per (ShipTo, Item) – weeks already identical
    keep_cols = [
        COL_SHIP_TO, COL_ITEM,
        COL_MAX_RTS_CURRENTWEEK, COL_MIN_RTS_MAXWEEK,
        COL_RTS_INITIAL_WEEK, COL_RTS_WEEK_MINUS_1,
        COL_RTS_WEEK, COL_RTS_WEEK_PLUS_3
    ]
    df_lookup = (
        df_rts[keep_cols]
        .drop_duplicates([COL_SHIP_TO, COL_ITEM])
        .set_index([COL_SHIP_TO, COL_ITEM])
    )

    # make sure numeric week strings → int32  ;  fill Na/'' with sentinel
    for c in df_lookup.columns:
        df_lookup[c] = (
            pd.to_numeric(df_lookup[c].replace('', np.nan), errors='coerce')
              .fillna(FILL_I32).astype(np.int32)
        )

    # ------------------------------------------------------------------ 3) build row-wise aligned lookup arrays
    # key arrays from df_grid
    idxer = df_lookup.index.get_indexer(
        pd.MultiIndex.from_arrays(
            [df_grid[COL_SHIP_TO], df_grid[COL_ITEM]],
            names=[COL_SHIP_TO, COL_ITEM]
        )
    )
    valid = idxer != -1

    # materialise numpy arrays only once
    arr_max_rts     = np.full(len(df_grid), FILL_I32, dtype=np.int32)
    arr_rts_init    = np.full_like(arr_max_rts, FILL_I32)
    arr_rts_week_m1 = np.full_like(arr_max_rts, FILL_I32)
    arr_rts_week    = np.full_like(arr_max_rts, FILL_I32)
    arr_rts_week_p3 = np.full_like(arr_max_rts, FILL_I32)

    arr_max_rts[valid]     = df_lookup[COL_MAX_RTS_CURRENTWEEK].to_numpy()[idxer[valid]]
    arr_rts_init[valid]    = df_lookup[COL_RTS_INITIAL_WEEK]   .to_numpy()[idxer[valid]]
    arr_rts_week_m1[valid] = df_lookup[COL_RTS_WEEK_MINUS_1]   .to_numpy()[idxer[valid]]
    arr_rts_week[valid]    = df_lookup[COL_RTS_WEEK]           .to_numpy()[idxer[valid]]
    arr_rts_week_p3[valid] = df_lookup[COL_RTS_WEEK_PLUS_3]    .to_numpy()[idxer[valid]]

    # ------------------------------------------------------------------ 4) derive numeric week for each calendar row
    week_num = pd.to_numeric(df_grid[COL_TIME_PW].str.replace(r'\D', '', regex=True)).to_numpy(dtype=np.int32)

    curr_week_int = np.int32(current_week_normalized)
    max_week_int  = np.int32(max_week_normalized)

    # ------------------------------------------------------------------ 5) prepare result frame (shallow copy + two mutable cols)
    df_res = df_grid.copy(deep=False)          # metadata only (≃0 bytes)
    df_res = df_res.assign(                   # copy just the two mutable columns
        **{
            COL_LOCK_COND:  df_grid[COL_LOCK_COND].copy(),
            COL_COLOR:      df_grid[COL_COLOR].copy()
        }
    )

    # ------------------------------------------------------------------ 8) WHITE  (max(rts, current) … end-of-calendar) —— lock = False
    mask_white = (
        (week_num >= curr_week_int) &                 # << added condition
        (week_num >= arr_max_rts) &
        (week_num <= max_week_int)
    )
    df_res.loc[mask_white, [COL_COLOR, COL_LOCK_COND]] = [COLOR_WHITE, False]

    # ------------------------------------------------------------------ 6) DARK-BLUE  (init … COM-1)  —— lock = True
    mask_darkblue = (
        (week_num >= curr_week_int) &
        (arr_rts_init <= week_num) &
        (week_num <= arr_rts_week_m1)
    )
    df_res.loc[mask_darkblue, COL_COLOR]     = COLOR_DARKBLUE
    df_res.loc[mask_darkblue, COL_LOCK_COND] = True

    # ------------------------------------------------------------------ 7) LIGHT-BLUE (COM … COM+3) —— lock depends on past/future
    mask_lightblue = (
        (week_num >= curr_week_int) &
        (arr_rts_week <= week_num) &
        (week_num <= arr_rts_week_p3)
    )

    df_res.loc[mask_lightblue, COL_COLOR] = COLOR_LIGHTBLUE
    # past weeks locked, future weeks open
    df_res.loc[mask_lightblue & (week_num <  curr_week_int), COL_LOCK_COND] = True
    df_res.loc[mask_lightblue & (week_num >= curr_week_int), COL_LOCK_COND] = False

    """
        # RTS 시작 이후 4주차에 대한 LIGHTBLUE 적용 X 디버그
        import duckdb
        duckdb.register('df_res', df_res)
        query = f'''
            select * from df_res 
            where "{COL_ITEM}" = 'RF70F29DAQLAA'
            and "{COL_TIME_PW}" >= '202513'
            and "{COL_TIME_PW}" <= '202516'
        '''
        duckdb.query(query).show()
    """

    # ------------------------------------------------------------------ 9) save & return
    # output_dataframes[STR_DF_STEP01_5_RTS_LOCKCOLOR_FINAL] = df_res
    return df_res




################################################################################################################
# ──[ 2 ]  BAS · MOBILE items – 8-week 13_GREEN update  ────────────────────────────────────────────────────────
# Rule (identical to Step 6 in S/In script 2024-04-16)
#   • Find items in **df_in_Item_Master** with
#         Item.[ProductType] == "BAS"    AND   Item.[Item GBM] == "MOBILE".
#   • For every (Item, Ship-To) row in the calendar grid
#         from   current-week  … current-week + 7   (inclusive)
#         where  Colour ≠ "19_GRAY"
#         → set  Colour = "13_GREEN",   Lock = False
#   • If no BAS·MOBILE item exists, the grid is returned unchanged.
################################################################################################################
# DataFrames
#   • df_in_Item_Master                        (item catalogue)
#   • df_step01_5_RTS_LOCKCOLOR_FINAL          (grid after Step 1-5)
#   • df_step02_BAS_13_GREEN                   (result of this step)
################################################################################################################
# Column constants ────────────────────────────────────────────────────────────────────────────────────────────
# COL_PROD_TYPE   = 'Item.[ProductType]'
# COL_ITEM_GBM    = 'Item.[Item GBM]'
# COL_ITEM        = 'Item.[Item]'
# COL_SHIP_TO     = 'Sales Domain.[Ship To]'
# COL_TIME_PW     = 'Time.[Partial Week]'
# COL_LOCK_COND   = 'Lock Condition'
# COL_COLOR       = 'S/Out FCST Color Condition'# Colour literals
# COLOR_GRAY   = '19_GRAY'
# COLOR_GREEN  = '13_GREEN'

# DataFrame-name constants
# STR_DF_IN_ITEM_MASTER               = 'df_in_Item_Master'
# STR_DF_STEP01_5_RTS_LOCKCOLOR_FINAL = 'df_step01_5_RTS_LOCKCOLOR_FINAL'
# STR_DF_STEP02_BAS_13_GREEN          = 'df_step02_BAS_13_GREEN'


@_decoration_
def fn_step02_bas_mobile_green():
    """
    Step 2 – 13_GREEN colouring for BAS·MOBILE items over the next 8 weeks.
    """
    # ── 1) BAS·MOBILE item list ────────────────────────────────────────────────
    df_item_mst = input_dataframes[STR_DF_IN_ITEM_MASTER]           # read-only
    bas_mobile_items = (
        df_item_mst[
            (df_item_mst[COL_PROD_TYPE] == 'BAS') &
            (df_item_mst[COL_ITEM_GBM]  == 'MOBILE')
        ][COL_ITEM]
        .unique()
    )

    if bas_mobile_items.size == 0:               # nothing to do
        return output_dataframes[STR_DF_STEP01_5_RTS_LOCKCOLOR_FINAL]

    # ── 2) working grid  (share memory, own mutable cols) ─────────────────────
    df_grid = output_dataframes[STR_DF_STEP01_5_RTS_LOCKCOLOR_FINAL]
    df_res  = df_grid.copy(deep=False)           # ≈0 bytes
    df_res = df_res.assign(
        **{
            COL_LOCK_COND: df_grid[COL_LOCK_COND].copy(),
            COL_COLOR:     df_grid[COL_COLOR].copy()
        }
    )

    # ── 3) mask for the 8-week window & items ─────────────────────────────────
    curr_week_int = int(current_week_normalized)
    cur_week_plus_7_str = common.gfn_add_week(current_week_normalized, 7)
    cur_week_plus_7 = int(cur_week_plus_7_str)

    week_num = df_res[COL_TIME_PW].str.replace(r'\D', '', regex=True).astype(int)


    mask = (
        df_res[COL_ITEM].isin(bas_mobile_items) &
        (week_num >= curr_week_int) &
        (week_num < cur_week_plus_7) &
        (df_res[COL_COLOR] != COLOR_GRAY)
    )

    # ── 4) apply colour + unlock ──────────────────────────────────────────────
    df_res.loc[mask, COL_COLOR]     = COLOR_GREEN
    df_res.loc[mask, COL_LOCK_COND] = False

    # # Optional DuckDB registration for ad-hoc SQL
    # import duckdb
    # duckdb.register(STR_DF_STEP02_BAS_13_GREEN, df_res)

    return df_res


################################################################################################################
# ──[ 3-1 ]  Create max-TATTERM table  ─────────────────────────────────────────────────────────────────────────
#   • Source      : df_in_Item_TAT               (TAT file from PDM / PLM)
#   • Optional FK : df_in_Item_Master            (filter to valid items -- optional but keeps rows tidy)
#   • Result      : df_step03_1_Item_TATTERM     (max-TAT by (Item GBM, Item))
#
#   Spec:
#     – drop Version.[Version Name]
#     – group-by (Item.[Item GBM] , Item.[Item]) and keep MAX(TATTERM)
#     – add CURWEEK_MINUS_TATTERM  = current-week + (TATTERM-1)  ← use common.gfn_add_week()
################################################################################################################# ── dataframe-name constants ─────────────────────────────────────────────────────────────────────────────────
# STR_DF_IN_ITEM_TAT      = 'df_in_Item_TAT'
# STR_DF_IN_ITEM_MASTER   = 'df_in_Item_Master'          # optional filter
# STR_DF_STEP03_1_TAT_MAX = 'df_step03_1_Item_TATTERM'

# ── column constants ─────────────────────────────────────────────────────────────────────────────────────────
# COL_VERSION   = 'Version.[Version Name]'
# COL_ITEM_GBM  = 'Item.[Item GBM]'
# COL_ITEM      = 'Item.[Item]'
# COL_TATTERM   = 'TATTERM'                    # ← final name we expose

# raw column name in the TAT file (if different)
# COL_RAW_TATTERM = 'ITEMTAT_TATTERM'

# helper column
# COL_CURWEEK_MIN_TAT = 'CURWEEK_MINUS_TATTERM'

@_decoration_
def fn_step03_1_generate_tatterm():
    """
    Step 3-1 – produce a table with the maximum TATTERM per (GBM, Item)
    and compute CURWEEK_MINUS_TATTERM.
    """
    # --- 1)  load & (optionally) filter --------------------------------------------------------------------------------
    df_tat = input_dataframes[STR_DF_IN_ITEM_TAT]

    # if an Item master is present, keep only valid items (this is a no-op if the name is not registered)
    if STR_DF_IN_ITEM_MASTER in input_dataframes:
        valid_items = input_dataframes[STR_DF_IN_ITEM_MASTER][[COL_ITEM_GBM, COL_ITEM]].drop_duplicates()
        df_tat = df_tat.merge(valid_items, on=[COL_ITEM_GBM, COL_ITEM], how='inner', copy=False)

    # --- 2)  remove Version column (saves memory immediately) ----------------------------------------------------------
    if COL_VERSION in df_tat.columns:
        df_tat = df_tat.drop(columns=[COL_VERSION])

    # --- 3)  group-by and take MAX(TATTERM) ---------------------------------------------------------------------------
    df_max = (
        df_tat
        .groupby([COL_ITEM_GBM, COL_ITEM], as_index=False, sort=False)[COL_RAW_TATTERM]
        .max()
        .rename(columns={COL_RAW_TATTERM: COL_TATTERM})
    )

    # ensure integer type (some PLM extracts are strings)
    df_max[COL_TATTERM] = pd.to_numeric(df_max[COL_TATTERM], errors='coerce').fillna(0).astype(int)

    # --- 4)  compute CURWEEK_MINUS_TATTERM (week arithmetic, not simple +7) ------------------------------------------
    df_max[COL_CW_PLUS_TATTERM] = df_max[COL_TATTERM].apply(
        lambda x: common.gfn_add_week(current_week_normalized, x - 1) if x else ''
    )

    # (optional) register in DuckDB for ad-hoc SQL analysis
    # import duckdb; duckdb.register(STR_DF_STEP03_1_TAT_MAX, df_max)

    return df_max

################################################################################################################
# ──[ 3-2 ]  Apply VD/SHA lead-time window → 18_DGRAY_RED  ─────────────────────────────────────────────────────
# Business rule
#   • For GBM in {'VD','SHA'} look up max-TATTERM from Step 3-1.
#   • For every calendar row whose      week ∈ [ current-week … current-week + TATTERM-1 ]
#       and whose current colour == '14_WHITE'     (→ not protected by RTS/EOS)
#       → set  Colour = '18_DGRAY_RED',   Lock = True
# DataFrames
#   • df_step02_BAS_13_GREEN        ← output of Step 2  (working grid)
#   • df_step03_1_Item_TATTERM      ← output of Step 3-1 (max TAT per item)
#   • df_step03_2_VD_LEADTIME       ← result of this step
################################################################################################################
# Uses results from:
#   • Step 2  → STR_DF_STEP02_BAS_13_GREEN      (grid with current colour / lock)
#   • Step 3-1→ STR_DF_STEP03_1_TAT_MAX         (max-TAT lookup per Item GBM + Item)
#   • Item master to fetch the missing Item GBM for each calendar row
################################################################################################################
# # Column / colour constants -------------------------------------------------------------------------------COL_ITEM_GBM      = 'Item.[Item GBM]'
# COL_ITEM          = 'Item.[Item]'
# COL_TIME_PW       = 'Time.[Partial Week]'
# COL_COLOR         = 'S/Out FCST Color Condition'
# COL_LOCK_COND          = 'Lock Condition'
# COL_TATTERM       = 'TATTERM'
# COL_CW_PLUS_TATTERM      = 'CURWEEK_PLUS_TATTERM'      # produced in Step 3-1

# COLOR_WHITE       = '14_WHITE'
# COLOR_DGRAY_RED   = '18_DGRAY_RED'

# # DataFrame-name constants -------------------------------------------------------------------------------
# STR_DF_IN_ITEM_MASTER              = 'df_in_Item_Master'
# STR_DF_STEP02_BAS_13_GREEN         = 'df_step02_BAS_13_GREEN'
# STR_DF_STEP03_1_TAT_MAX            = 'df_step03_1_TAT_MAX'
# STR_DF_STEP03_2_VD_LEAD            = 'df_step03_2_VD_LEAD'      # ← new result


@_decoration_
def fn_step03_2_apply_vd_leadtime():
    """
    Step 3-2 – Paint VD / SHA items 18_DGRAY_RED from current-week
    through current-week + TATTERM – 1  (only if colour is still WHITE).

    Lock flag is set to **False** in the affected rows.
    """
    # ── 1) source DataFrames ──────────────────────────────────────────────────────────
    df_grid = output_dataframes[STR_DF_STEP02_BAS_13_GREEN]      # after Step 2
    df_tat  = output_dataframes[STR_DF_STEP03_1_TAT_MAX]         # (GBM,Item) → TATTERM
    df_item = input_dataframes[STR_DF_IN_ITEM_MASTER]            # to fetch Item GBM

    # ── 2) attach Item GBM to the grid (memory-cheap map) ────────────────────────────
    map_item2gbm = (
        df_item[[COL_ITEM, COL_ITEM_GBM]]
        .drop_duplicates(subset=[COL_ITEM])
        .set_index(COL_ITEM)[COL_ITEM_GBM]
        .to_dict()
    )

    df_res = df_grid.copy(deep=False)           # share base memory
    df_res = df_res.assign(
        **{
            COL_ITEM_GBM: df_grid[COL_ITEM].map(map_item2gbm),
            COL_COLOR:    df_grid[COL_COLOR].copy(),
            COL_LOCK_COND:     df_grid[COL_LOCK_COND].copy()
        }
    )

    # ── 3) build fast lookup arrays from df_tat ──────────────────────────────────────
    # keep only VD / SHA rows – everything else is irrelevant here
    df_tat = df_tat[df_tat[COL_ITEM_GBM].isin(['VD', 'SHA'])]

    # index = (GBM, Item)  ➜  very cheap positional lookup
    df_tat_idx = (
        df_tat
        .set_index([COL_ITEM_GBM, COL_ITEM], verify_integrity=False)
        [[COL_TATTERM, COL_CW_PLUS_TATTERM]]
    )

    pos = df_tat_idx.index.get_indexer(
        pd.MultiIndex.from_arrays([df_res[COL_ITEM_GBM], df_res[COL_ITEM]],
                                  names=[COL_ITEM_GBM, COL_ITEM])
    )
    valid = pos != -1

    # extract TATTERM / CW-MINUS arrays (fill with sentinel for non-matches)
    n        = len(df_res)
    fill_val = -999999
    tat_arr  = np.full(n, fill_val, dtype=int)
    cwm_arr  = np.full(n, fill_val, dtype=int)

    # tat_arr[valid] = df_tat_idx.iloc[pos[valid]][COL_TATTERM].to_numpy(dtype=int)
    # cwm_arr[valid] = df_tat_idx.iloc[pos[valid]][COL_CW_PLUS_TATTERM].to_numpy(dtype=int)

    tat_arr[valid] = df_tat_idx[COL_TATTERM].to_numpy()[pos[valid]]
    cwm_arr[valid] = df_tat_idx[COL_CW_PLUS_TATTERM].to_numpy()[pos[valid]]

    # ── 4) numeric week of each calendar row ─────────────────────────────────────────
    row_week = df_res[COL_TIME_PW].str.replace(r'\D', '', regex=True).astype(int).to_numpy()  # yyyyWW as int
    curr_wk  = int(current_week_normalized)

    # ── 5) build boolean mask for rows to update ─────────────────────────────────────
    cond = (
        df_res[COL_ITEM_GBM].isin(['VD', 'SHA']).to_numpy() &
        (row_week >= curr_wk) &
        (row_week <= cwm_arr) &                        # >= CURWEEK_MINUS_TATTERM
        # (row_week <= curr_wk + tat_arr - 1) &          # <= current + (tat-1)
        (df_res[COL_COLOR].to_numpy() == COLOR_WHITE) &   # don’t override RTS/EOS etc.
        (tat_arr != fill_val)                 # ensure TATTERM was found
    )

    # ── 6) apply colour + lock/unlock ────────────────────────────────────────────────
    df_res.loc[cond, COL_COLOR] = COLOR_DGRAY_RED
    df_res.loc[cond, COL_LOCK_COND]  = False        # <- set True if you follow the example table

    """
        # TATTERM 구간 반대 적용 오류 
        # Item.[Item GBM],Item.[Item],TATTERM,CURWEEK_PLUS_TATTERM
        # SHA,RS23A500ASR/AA,15,202520
        import duckdb
        duckdb.register('df_res', df_res)
        query = f'''
            select * from df_res 
            where "{COL_ITEM}" = 'RS23A500ASR/AA'
            and "{COL_TIME_PW}" >= '202506'
            and "{COL_TIME_PW}" <= '202521'
        '''
        duckdb.query(query).show()

        # 이후구간 
        query = f'''
            select * from df_res 
            where "{COL_ITEM}" = 'RS23A500ASR/AA'
            and "{COL_TIME_PW}" > '202520'
        '''
        duckdb.query(query).show()
    """
    return df_res


################################################################################################################
# ──[ 4-1 ]  Sales-Product-ASN → lowest-level calendar grid  ──────────────────────────────────────────────────
# Goal :
#   1) Start from df_in_Sales_Product_ASN  (Ship-To • Item • Location)
#   2) Add Item.[Product Group] from the item-master
#   3) Cartesian-expand across every partial-week in df_in_Time_PW
#   4) Initialise   • Lock   = False   • Colour = 14_WHITE
#      Rows *before* the current week  → Lock = True  • Colour = 19_GRAY
################################################################################################################
# DataFrames in / out
#   IN : df_in_Sales_Product_ASN         (STR_DF_IN_SALES_PRODUCT_ASN)
#        df_in_Item_Master               (STR_DF_IN_ITEM_MASTER)
#        df_in_Time_PW                   (STR_DF_IN_TIME_PW)
#   OUT: df_step04_1_ASN_GRID            (STR_DF_STEP04_1_ASN_GRID)
################################################################################################################
# Column constants ────────────────────────────────────────────────────────────────────────────────────────────
# COL_VERSION           = 'Version.[Version Name]'
# COL_SALES_ASN_FLAG    = 'Sales Product ASN'           # Y/N column to drop
# COL_ITEM              = 'Item.[Item]'
# COL_ITEM_PG           = 'Item.[Product Group]'
# COL_SHIP_TO           = 'Sales Domain.[Ship To]'
# COL_LOCATION          = 'Location.[Location]'
# COL_TIME_PW           = 'Time.[Partial Week]'
# COL_LOCK_COND         = 'Lock Condition'
# COL_COLOR             = 'S/Out FCST Color Condition'# Colours
# COLOR_GRAY   = '19_GRAY'
# COLOR_WHITE  = '14_WHITE'

# Data-frame name constants
# STR_DF_IN_SALES_PRODUCT_ASN = 'df_in_Sales_Product_ASN'
# STR_DF_IN_ITEM_MASTER       = 'df_in_Item_Master'
# STR_DF_IN_TIME_PW           = 'df_in_Time_PartialWeek'
# STR_DF_STEP04_1_ASN_GRID    = 'df_step04_1_ASN_GRID'

@_decoration_
def fn_step04_1_sales_product_asn_preprocess():
    """
    Step 4-1 – explode Sales-Product-ASN to (ShipTo, Item, Location, Week) grid
    with initial Lock / Colour flags.
    """
    # ── 1) source tables ──────────────────────────────────────────────────────
    df_asn   = input_dataframes [STR_DF_IN_SALES_PRODUCT_ASN].copy()
    df_item  = input_dataframes [STR_DF_IN_ITEM_MASTER]      .copy()
    df_weeks = input_dataframes [STR_DF_IN_TIME_PW]          .copy()

    # ── 2) ASN: drop unused columns & add Product-Group ───────────────────────
    drop_cols = [c for c in (COL_VERSION, COL_SALES_ASN_FLAG) if c in df_asn.columns]
    if drop_cols:
        df_asn.drop(columns=drop_cols, inplace=True)

    # map Item → Product Group (small Series → dict → .map, very cheap)
    pg_map = (
        df_item[[COL_ITEM, COL_ITEM_PG]]
        .drop_duplicates(subset=[COL_ITEM])
        .set_index(COL_ITEM)[COL_ITEM_PG]
        .to_dict()
    )
    df_asn[COL_ITEM_PG] = df_asn[COL_ITEM].map(pg_map)

    # ── 3) Cartesian with all partial-weeks (memory-efficient repeat/tile) ────
    week_list = df_weeks[COL_TIME_PW].tolist()
    n_weeks   = len(week_list)
    n_rows    = len(df_asn)

    df_grid = df_asn.loc[df_asn.index.repeat(n_weeks)].copy(deep=False)  # ≈0 bytes extra
    df_grid[COL_TIME_PW] = np.tile(week_list, n_rows)

    # ── 4) initialise Lock / Colour ───────────────────────────────────────────
    df_grid[COL_LOCK_COND] = False
    df_grid[COL_COLOR]     = COLOR_WHITE

    # rows BEFORE current partial-week → Lock & Grey
    week_num = df_grid[COL_TIME_PW].str.replace(r'\D', '', regex=True).astype(int)
    mask_past = week_num < int(current_week_normalized)

    df_grid.loc[mask_past, COL_LOCK_COND] = True
    df_grid.loc[mask_past, COL_COLOR]     = COLOR_GRAY

    return df_grid



# ═════════════════════════════════════════════════════════════════════════════
#  Step 4-2 · copy Lock / Color from Step 3-2 to ASN grid (Lv6 / Lv7 rows)
#           – vectorised parent-lookup, no row duplication
# ═════════════════════════════════════════════════════════════════════════════
# STR_DF_STEP03_2_VD_LEAD   = 'df_step03_2_VD_LEADTIME'      # Step 3-2 result (Lv2/3)
# STR_DF_STEP04_1_ASN_GRID      = 'df_step04_1_ASN_GRID'         # Step 4-1 result (Lv6/7)
# STR_DF_IN_SALES_DOMAIN_DIM    = 'df_in_Sales_Domain_Dimension' # dimension table
# STR_DF_STEP04_2_ASN_LOCKCOLOR = 'df_step04_2_ASN_LOCKCOLOR'    # ← new output nameCOL_ITEM          = 'Item.[Item]'
# COL_SHIP_TO       = 'Sales Domain.[Ship To]'
# COL_TIME_PW       = 'Time.[Partial Week]'
# COL_LOCK_COND     = 'Lock Condition'
# COL_COLOR         = 'S/Out FCST Color Condition'
# COL_LV2     = 'Sales Domain.[Sales Domain LV2]'
# COL_LV3     = 'Sales Domain.[Sales Domain LV3]'

@_decoration_
def fn_step04_2_apply_rts_to_asn():
    """
    Step 4-2 – propagate Step 3-2 lock / colour decisions down to each ASN grid
    row (Lv6 / Lv7) **without duplicating rows**.

    * For every ASN row find its parent Lv3, then Lv2 (via the dimension table)
      and copy the (Lock, Color) of that parent/Item/Week triplet from
      df_step03_2_VD_LEADTIME.
    * Existing grid rows keep their Lock/Colour if no parent match is found.
    * We only overwrite when the grid colour is still WHITE – RTS/EOS zones or
      already-coloured rows stay untouched.
    """
    # ── 1) source frames ─────────────────────────────────────────────────────
    df_grid = output_dataframes[STR_DF_STEP04_1_ASN_GRID].copy()          # Lv6 / Lv7
    df_dec  = output_dataframes[STR_DF_STEP03_2_VD_LEAD]       # Lv2 / Lv3
    df_dim  = input_dataframes [STR_DF_IN_SALES_DOMAIN_DIM]        # hierarchy

    # ── 2) fast index for decision table  (Item,ShipTo,Week) → (Lock,Color) ──
    df_dec_idx = (
        df_dec
        .set_index([COL_ITEM, COL_SHIP_TO, COL_TIME_PW], verify_integrity=False)
        [[COL_LOCK_COND, COL_COLOR]]
    )
    arr_dec_lock  = df_dec_idx[COL_LOCK_COND].to_numpy(dtype=bool)
    arr_dec_color = df_dec_idx[COL_COLOR].to_numpy(dtype=object)

    # ── 3) parent Lv2 / Lv3 look-ups for every Ship-To in grid ───────────────
    dim_idx = df_dim.set_index(COL_SHIP_TO)
    lv2_arr = dim_idx[COL_LV2].to_numpy(dtype=object)
    lv3_arr = dim_idx[COL_LV3].to_numpy(dtype=object)

    pos_dim   = dim_idx.index.get_indexer(df_grid[COL_SHIP_TO])
    valid_dim = pos_dim != -1
    parent_lv2 = np.where(valid_dim, lv2_arr[pos_dim], None)
    parent_lv3 = np.where(valid_dim, lv3_arr[pos_dim], None)

    # ── 4) positional fetch  (Lv3 first, then Lv2) ───────────────────────────
    idx_lv3 = pd.MultiIndex.from_arrays(
        [df_grid[COL_ITEM], parent_lv3, df_grid[COL_TIME_PW]],
        names=[COL_ITEM, COL_SHIP_TO, COL_TIME_PW]
    )
    pos3 = df_dec_idx.index.get_indexer(idx_lv3)
    hit3 = pos3 != -1

    idx_lv2 = pd.MultiIndex.from_arrays(
        [df_grid[COL_ITEM], parent_lv2, df_grid[COL_TIME_PW]]
    )
    pos2 = df_dec_idx.index.get_indexer(idx_lv2)
    hit2 = (~hit3) & (pos2 != -1)          # only rows not found at Lv3

    hit_any   = hit3 | hit2
    pos_final = np.where(hit3, pos3, pos2) # choose Lv3 if available else Lv2

    # ── 5) apply overrides  (only when current colour = WHITE) ───────────────
    grid_color_arr = df_grid[COL_COLOR].to_numpy(dtype=object)
    grid_lock_arr  = df_grid[COL_LOCK_COND].to_numpy(dtype=bool)

    mask_update = hit_any & (grid_color_arr == COLOR_WHITE)
    # mask_update = hit_any

    grid_color_arr[mask_update] = arr_dec_color[pos_final[mask_update]]
    grid_lock_arr [mask_update] = arr_dec_lock [pos_final[mask_update]]

    # write back (no extra copy)
    df_grid.loc[:, COL_COLOR]     = grid_color_arr
    df_grid.loc[:, COL_LOCK_COND] = grid_lock_arr

    return df_grid



# ─────────────────────────────── constants ────────────────────────────────
# STR_DF_IN_SELL_OUT_MASTER      = 'df_in_Sell_Out_Simul_Master'      # input
# STR_DF_STEP05_1_SO_MASTER_LOCK = 'df_step05_1_SO_Master_Lock'       # outputCOL_VERSION        = 'Version.[Version Name]'
# COL_SHIP_TO        = 'Sales Domain.[Ship To]'
# COL_ITEM_PG       = 'Item.[Product Group]'
# COL_MASTER_STATUS  = 'S/Out Master Status'
# COL_LOCK_COND      = 'Lock Condition'          # ← newly added column
# --------------------------------------------------------------------------

@_decoration_
def fn_step05_1_prepare_sellout_master() -> pd.DataFrame:
    """    Step 5-1 –  Prepare Sell-Out-Simul-Master table for later joins.
      • keep rows with S/Out Master Status == 'CON'
      • drop Version column
      • add constant True in “Lock Condition”
      • drop duplicates on (ShipTo, ProductGroup) – one record per key
    """    # 1) load .............................................................................
    df_src = input_dataframes[STR_DF_IN_SELL_OUT_MASTER].copy(deep=True)

    # 2) filter only ‘CON’ status rows ....................................................
    mask_con = df_src[COL_MASTER_STATUS] == 'CON'
    df_flt   = df_src.loc[mask_con,
                          [COL_SHIP_TO, COL_ITEM_PG, COL_MASTER_STATUS]]

    # 3) add Lock flag & remove Version column ............................................
    # df_flt[COL_LOCK_COND] = True          # confirmed masters always locked

    # 4) ensure uniqueness on (ShipTo, ProductGroup) .....................................
    df_res = (
        df_flt
        .drop_duplicates(subset=[COL_SHIP_TO, COL_ITEM_PG], keep='first')
        .reset_index(drop=True)
    )

    return df_res

# ─────────────────────────────────────────────────────────────────────────────
#  Data-frame names
# ─────────────────────────────────────────────────────────────────────────────
# STR_DF_STEP04_2_ASN_LOCKCOLOR  = 'df_step04_2_ASN_LockColor'       # Step 4-2 grid
# STR_DF_STEP05_1_SO_MASTER_LOCK = 'df_step05_1_SO_Master_Lock'      # Step 5-1 keys
# STR_DF_STEP05_2_GRID_UPD       = 'df_step05_2_Grid_MasterApplied'  # NEW result
# ─────────────────────────────────────────────────────────────────────────────
#  Column & literal constants
# ─────────────────────────────────────────────────────────────────────────────
# COL_SHIP_TO     = 'Sales Domain.[Ship To]'
# COL_ITEM_PG    = 'Item.[Product Group]'
# COL_ITEM        = 'Item.[Item]'
# COL_TIME_PW     = 'Time.[Partial Week]'
# COL_LOCK_COND   = 'Lock Condition'
# COL_COLOR       = 'S/Out FCST Color Condition'

# COLOR_GRAY      = '19_GRAY'

# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step05_2_apply_master_lockcolor() -> pd.DataFrame:
    """    Step 5-2 – For every calendar row (Lv6/7) **not** present in the confirmed
    Sell-Out master (ShipTo × ProductGroup), force:

        Lock Condition              = True
        S/Out FCST Color Condition  = '19_GRAY'

    Pure lookup - no expensive merge – scales to 100 M+ rows.
    """    # 1) read sources (never mutated) ----------------------------------------
    df_grid   = output_dataframes[STR_DF_STEP04_2_ASN_LOCKCOLOR]
    df_master = output_dataframes[STR_DF_STEP05_1_SO_MASTER_LOCK]   # only valid CON rows

    # 2) build a constant-time lookup index on (ShipTo, ProdGrp) ------------
    idx_master = (
        df_master[[COL_SHIP_TO, COL_ITEM_PG]]
        .drop_duplicates()
        .set_index([COL_SHIP_TO, COL_ITEM_PG])
    )

    # 3) positional match for every grid row --------------------------------
    pos = idx_master.index.get_indexer(
        pd.MultiIndex.from_arrays(
            [df_grid[COL_SHIP_TO], df_grid[COL_ITEM_PG]],
            names=[COL_SHIP_TO, COL_ITEM_PG]
        )
    )
    mask_not_in_master = pos == -1          # rows to override

    # 4) create result – share memory except 2 writable cols ----------------
    df_res = df_grid.copy(deep=False)       # ≈0 bytes (view)
    df_res = df_res.assign(
        **{
            COL_LOCK_COND: df_grid[COL_LOCK_COND].copy(),
            COL_COLOR    : df_grid[COL_COLOR].copy()
        }
    )

    df_res.loc[mask_not_in_master, COL_LOCK_COND] = True
    df_res.loc[mask_not_in_master, COL_COLOR]     = COLOR_GRAY

    return df_res

# ─────────────────────────────────────────────────────────────────────────────
#  Data-frame names
# ─────────────────────────────────────────────────────────────────────────────
# STR_DF_STEP04_2_ASN_LOCKCOLOR   = 'df_step04_2_ASN_LockColor'        # Step 4-2 grid
# STR_DF_STEP05_1_SO_MASTER_LOCK  = 'df_step05_1_SO_Master_Lock'       # Step 5-1 keys
# STR_DF_IN_SALES_DOMAIN_DIM      = 'df_in_Sales_Domain_Dimension'     # hierarchy
# STR_DF_STEP05_3_GRID_HIER       = 'df_step05_3_Grid_MasterHier'      # new result
# ─────────────────────────────────────────────────────────────────────────────
#  Columns / literals
# ─────────────────────────────────────────────────────────────────────────────
# COL_SHIP_TO      = 'Sales Domain.[Ship To]'
# COL_PROD_GRP     = 'Item.[Product Group]'
# COL_ITEM         = 'Item.[Item]'
# COL_TIME_PW      = 'Time.[Partial Week]'
# COL_LOCK_COND    = 'Lock Condition'
# COL_COLOR        = 'S/Out FCST Color Condition'

# dimension
# COL_LV6       = 'Sales Domain.[Sales Domain Lv6]'
# COL_LV7       = 'Sales Domain.[Sales Domain Lv7]'

# COLOR_GRAY       = '19_GRAY'

# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step05_3_apply_master_hier_lockcolor() -> pd.DataFrame:
    """    Step 5-3 – override Lock / Color for every calendar row whose
    (ShipTo-Lv6/7 × ProductGroup) **is NOT covered** by the Sell-Out Master,
    *considering Lv-6 keys as parents of all their Lv-7 children*.

    Un-covered rows ⇒ Lock = True, Color = '19_GRAY'.

    Pure positional lookup – O(N) and memory-friendly.
    """
    # ── 1) read sources (never mutated) ────────────────────────────────────
    df_grid   = output_dataframes[STR_DF_STEP04_2_ASN_LOCKCOLOR]
    df_master = output_dataframes[STR_DF_STEP05_1_SO_MASTER_LOCK]
    df_dim    = input_dataframes[STR_DF_IN_SALES_DOMAIN_DIM]

    # ── 2) extend Master keys: add all Lv-7 children for every Lv-6 key ────
    # (i) map Lv-6 → Lv-7
    df_map_lv6_to_lv7 = (
        df_dim[[COL_LV6, COL_LV7]]
        .dropna()
        .drop_duplicates()
    )

    # (ii) rows where Master ShipTo is *exactly* a Lv-6
    df_master_lv6 = df_master.merge(df_map_lv6_to_lv7,
                                    left_on=COL_SHIP_TO,
                                    right_on=COL_LV6,
                                    how='inner',
                                    copy=False)

    # descendant keys
    df_desc_keys = (
        df_master_lv6[[COL_LV7, COL_PROD_GRP]]
        .rename(columns={COL_LV7: COL_SHIP_TO})
    )

    # (iii) union original Master keys + descendant Lv-7 keys
    df_keys = (
        pd.concat([df_master[[COL_SHIP_TO, COL_PROD_GRP]],
                   df_desc_keys],
                  ignore_index=True)
        .drop_duplicates()
        .set_index([COL_SHIP_TO, COL_PROD_GRP])
    )

    # ── 3) positional membership test (constant time) ─────────────────────
    pos = df_keys.index.get_indexer(
        pd.MultiIndex.from_arrays(
            [df_grid[COL_SHIP_TO], df_grid[COL_PROD_GRP]],
            names=[COL_SHIP_TO, COL_PROD_GRP]
        )
    )
    mask_not_in_master = pos == -1          # rows to force-lock

    # ── 4) build result – share memory except mutable cols ─────────────────
    df_res = df_grid.copy(deep=False)
    df_res = df_res.assign(
        **{
            COL_LOCK_COND: df_grid[COL_LOCK_COND].copy(),
            COL_COLOR    : df_grid[COL_COLOR].copy()
        }
    )

    df_res.loc[mask_not_in_master, COL_LOCK_COND] = True
    df_res.loc[mask_not_in_master, COL_COLOR]     = COLOR_GRAY

    return df_res


# ─────────────────────────────────────────────────────────────────────────────
#  Data-frame identifiers
# ─────────────────────────────────────────────────────────────────────────────
STR_DF_STEP05_3_GRID_HIER       = 'df_step05_3_Grid_MasterHier'          # base grid (Lv-6/7)
STR_DF_IN_SALES_DOMAIN_DIM      = 'df_in_Sales_Domain_Dimension'         # Lv-2…Lv-7 lookup
STR_DF_IN_FORECAST_RULE         = 'df_in_Forecast_Rule'                  # rule table
STR_DF_OUT_FCST_GC_LOCK         = 'df_output_Sell_Out_FCST_GC_Lock'
STR_DF_OUT_FCST_AP2_LOCK        = 'df_output_Sell_Out_FCST_AP2_Lock'
STR_DF_OUT_FCST_AP1_LOCK        = 'df_output_Sell_Out_FCST_AP1_Lock'
STR_DF_OUT_FCST_COLOR           = 'df_output_Sell_Out_FCST_Color_Condition'

# ─────────────────────────────────────────────────────────────────────────────
#  Column literals
# ─────────────────────────────────────────────────────────────────────────────
# COL_VERSION        = 'Version.[Version Name]'
# COL_SHIP_TO        = 'Sales Domain.[Ship To]'
# COL_ITEM           = 'Item.[Item]'
# COL_PROD_GRP       = 'Item.[Product Group]'
# COL_LOC            = 'Location.[Location]'
# COL_TIME_PW        = 'Time.[Partial Week]'
# COL_LOCK_COND      = 'Lock Condition'
# COL_COLOR          = 'S/Out FCST Color Condition'

# output measures
MEAS_LOCK_GC   = 'S/Out FCST_GC_Lock'
MEAS_LOCK_AP2  = 'S/Out FCST_AP2_Lock'
MEAS_LOCK_AP1  = 'S/Out FCST_AP1_Lock'
MEAS_COLOR     = COL_COLOR                           # same literal

# forecast-rule value columns in the rule table
# COL_RULE_GC    = 'FORECAST_RULE GC FCST'
# COL_RULE_AP2   = 'FORECAST_RULE AP2 FCST'
# COL_RULE_AP1   = 'FORECAST_RULE AP1 FCST'

# Sales-Domain dimension columns
LV_COLS = [COL_LV2,
           COL_LV3,
           COL_LV4,
           COL_LV5,
           COL_LV6,
           COL_LV7]

# Version   = Version      # constant for all outputs
LOC_PLACEHOLD = '-'           # location column in outputs
# COLOR_GRAY    = '19_GRAY'     # just used for fallback rows, if ever

# ─────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step06_build_sellout_forecast_outputs() -> tuple[pd.DataFrame,
                                                        pd.DataFrame,
                                                        pd.DataFrame,
                                                        pd.DataFrame]:
    """
    Step 6 – Convert the lock / colour grid to the three Forecast-Rule
    aggregation levels (GC / AP2 / AP1) **and** the final colour table.

    Returns 4 DataFrames (already version/location-filled):

        df_gc_lock, df_ap2_lock, df_ap1_lock, df_color_cond
    """
    # ── 0) load immutable sources ──────────────────────────────────────────
    df_src  = output_dataframes[STR_DF_STEP05_3_GRID_HIER]
    df_dim  = input_dataframes [STR_DF_IN_SALES_DOMAIN_DIM]
    df_rule = input_dataframes [STR_DF_IN_FORECAST_RULE]

    # ── 1) Dimension: Ship-To → Lv-2…Lv-7 arrays & index lookup ───────────
    df_dim = df_dim[[COL_SHIP_TO, *LV_COLS]].drop_duplicates()
    ship_idx        = df_dim.set_index(COL_SHIP_TO).index
    lv_codes_arr    = df_dim.set_index(COL_SHIP_TO)[LV_COLS].to_numpy(dtype=object)

    def pick_dom(pos: int, lv: int) -> str | None:
        """return Ship-To code of requested level (2-7) for dimension row `pos`"""
        return lv_codes_arr[pos, lv - 2]

    # ── 2) Build fast rule-dictionary  (key = (PG, domain)) ───────────────
    rule_dict: dict[tuple[str, str], tuple[int, int, int]] = {}
    for pg, dom, gcv, a2v, a1v in df_rule[[COL_PROD_GRP, COL_SHIP_TO,
                                           COL_RULE_GC, COL_RULE_AP2, COL_RULE_AP1]
                                          ].itertuples(index=False, name=None):
        rule_dict[(pg, dom)] = (int(gcv), int(a2v), int(a1v))

    # ── 3) ndarray slices from source grid (no .iloc in loop) ─────────────
    ship_arr  = df_src[COL_SHIP_TO ].to_numpy(dtype=object)
    pg_arr    = df_src[COL_PROD_GRP].to_numpy(dtype=object)
    itm_arr   = df_src[COL_ITEM    ].to_numpy(dtype=object)
    pw_arr    = df_src[COL_TIME_PW ].to_numpy(dtype=object)
    lock_arr  = df_src[COL_LOCK_COND    ].to_numpy(dtype=bool)
    color_arr = df_src[COL_COLOR   ].to_numpy(dtype=object)

    pos_ship  = ship_idx.get_indexer(ship_arr)
    valid_pos = pos_ship >= 0

    # ── 4) output row buffers  (lists → DataFrame) ────────────────────────
    rows_gc, rows_ap2, rows_ap1, rows_col = [], [], [], []
    col_seen: set[tuple] = set()          # deduplicate colour rows

    n = len(df_src)
    for i in range(n):
        if not valid_pos[i]:
            continue
        pos = pos_ship[i]
        pg  = pg_arr[i]

        # candidate rule keys at Lv-2 / Lv-3
        rules = []
        for base_lv in (2, 3):
            dom_lv = pick_dom(pos, base_lv)
            if dom_lv is None:
                continue
            rule = rule_dict.get((pg, dom_lv))
            if rule:
                rules.append(rule)

        if not rules:
            # # still output original row so that colour table is complete
            # key = (itm_arr[i], ship_arr[i], pw_arr[i])
            # if key not in col_seen:
            #     rows_col.append((VERSION_VAL, ship_arr[i], itm_arr[i],
            #                      LOC_PLACEHOLD, pw_arr[i], color_arr[i]))
            #     col_seen.add(key)
            continue

        # # always create ORIGINAL Lv-7 colour row (once)
        # col_key = (itm_arr[i], ship_arr[i], pw_arr[i])
        # if col_key not in col_seen:
        #     rows_col.append((VERSION_VAL, ship_arr[i], itm_arr[i],
        #                      LOC_PLACEHOLD, pw_arr[i], color_arr[i]))
        #     col_seen.add(col_key)

        # helper to push a lock-row and ensure colour exists
        def push_lock(lst, target_lv: int) -> None:
            dom_tgt = pick_dom(pos, target_lv)
            if not dom_tgt:
                return
            lst.append((Version, dom_tgt, itm_arr[i],
                        LOC_PLACEHOLD, pw_arr[i], None,lock_arr[i]))
            # colour row for that domain?
            ck = (itm_arr[i], dom_tgt, pw_arr[i])
            if ck not in col_seen:
                rows_col.append((Version, dom_tgt, itm_arr[i],
                                 LOC_PLACEHOLD, pw_arr[i], color_arr[i]))
                col_seen.add(ck)

        # apply EVERY matching rule (lower rule can override higher)
        for gc_lv, ap2_lv, ap1_lv in rules:
            if 2 <= gc_lv <= 7:
                push_lock(rows_gc,  gc_lv)
            if 2 <= ap2_lv <= 7:
                push_lock(rows_ap2, ap2_lv)
            if 2 <= ap1_lv <= 7:
                push_lock(rows_ap1, ap1_lv)

    # ── 5) materialise DataFrames & drop duplicates ───────────────────────
    def mk_lock_df(rows, meas_name):
        df = pd.DataFrame(rows, columns=[COL_VERSION, COL_SHIP_TO, COL_ITEM,
                                         COL_LOC, COL_TIME_PW, meas_name, f'{meas_name}.Lock'])
        return df.drop_duplicates(
            subset=[COL_VERSION, COL_SHIP_TO, COL_ITEM, COL_TIME_PW])

    df_gc_lock  = mk_lock_df(rows_gc,  MEAS_LOCK_GC )
    df_ap2_lock = mk_lock_df(rows_ap2, MEAS_LOCK_AP2)
    df_ap1_lock = mk_lock_df(rows_ap1, MEAS_LOCK_AP1)

    df_color = pd.DataFrame(rows_col, columns=[COL_VERSION, COL_SHIP_TO,
                                               COL_ITEM, COL_LOC, COL_TIME_PW,
                                               MEAS_COLOR])

    # ── 6) return four outputs ────────────────────────────────────────────
    return df_gc_lock, df_ap2_lock, df_ap1_lock, df_color

if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    out_Demand = pd.DataFrame()
    output_dataframes = {}
    input_dataframes = {}
    try:
        ################################################################################################################
        # 전처리 : 모듈 내에서 사용될 데이터에 대한 정합성 체크 및 데이터 선 가공
        ################################################################################################################
        if is_local:
            Version = 'CWV_DP'
            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------

            # input_folder_name  = str_instance
            # output_folder_name = str_instance
            # input_folder_name  = f'{str_instance}_SHA_REF'
            # output_folder_name = f'{str_instance}_SHA_REF'
            # input_folder_name  = f'PYForecastSellOutLockColor_0516_o9_data_local'
            # output_folder_name = f'PYForecastSellOutLockColor_0516_o9_data_local'
            input_folder_name  = f'PYForecastSellOutLockColor_0523_o9'
            output_folder_name = f'PYForecastSellOutLockColor_0523_o9'
            
            # ------
            str_input_dir = f'Input/{input_folder_name}'
            # ------
            str_output_dir = f'Output/{output_folder_name}'
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y%m%d_%H_%M")
            str_output_dir = f"{str_output_dir}_{formatted_time}"
            # ------
            os.makedirs(str_input_dir, exist_ok=True)
            os.makedirs(str_output_dir, exist_ok=True)

            # ----------------------------------------------------
            # Week
            # ----------------------------------------------------
            CurrentPartialWeek = '202506A'
            
        # vdLog 초기화
        log_path = os.path.dirname(__file__) if is_local else ""
        vdCommon.gfn_pyLog_start(Version, str_instance, logger, is_local, log_path)
        # --------------------------------------------------------------------------
        # df_input 체크 시작
        # --------------------------------------------------------------------------
        logger.Note(p_note='df_input 체크 시작', p_log_level=LOG_LEVEL.debug())
        fn_process_in_df_mst()
        # 입력 변수 중 데이터가 없는 경우 경고 메시지를 출력한다.
        for in_df in input_dataframes:
            fn_check_input_table(input_dataframes[in_df], in_df, '1')

        # --------------------------------------------------------------------------
        # 전역설정. max_week, current_week_normalized
        # --------------------------------------------------------------------------
        df_dim    = input_dataframes[STR_DF_IN_SALES_DOMAIN_DIM]
        df_time   = input_dataframes[STR_DF_IN_TIME_PW]
        df_rule   = input_dataframes[STR_DF_IN_FORECAST_RULE]
        df_itemmst= input_dataframes[STR_DF_IN_ITEM_MASTER]  

        df_eos = input_dataframes.get(STR_DF_IN_TIME_PW)
        max_week = df_eos[COL_TIME_PW].max()
        max_week_normalized = normalize_week(max_week)
        current_week_normalized = normalize_week(CurrentPartialWeek)

        # 입력 변수 확인
        if Version is None or Version.strip() == '':
            Version = 'CWV_DP'
        # 입력 변수 확인용 로그
        logger.Note(p_note=f'Version : {Version}', p_log_level=LOG_LEVEL.debug())

        # 필터
        ParamItemGbm = 'VD'

        # --------------------------------------------------------------------------
        # 전역설정 index. 
        # df_in_Sales_Domain_Dimension
        # df_in_Forecast_Rule
        # --------------------------------------------------------------------------
        # Sales_Domain_Dimension
        ship_idx, lvl_arr, ship_lv_dict = build_shipto_level_lut(
            input_dataframes[STR_DF_IN_SALES_DOMAIN_DIM])  

        dim_df = input_dataframes[STR_DF_IN_SALES_DOMAIN_DIM].set_index(COL_SHIP_TO).reindex(ship_idx)
        lv_cols = [COL_LV2, COL_LV3, COL_LV4,
                COL_LV5, COL_LV6, COL_LV7]
        lv_codes_arr = dim_df[lv_cols].to_numpy(dtype=object)   # shape(len(ship_idx),6)  

        # Forecast-Rule dict  (벡터 · 반복문 X) ──
        rule_pg   = input_dataframes[STR_DF_IN_FORECAST_RULE][COL_ITEM_PG       ].to_numpy(dtype=object)
        rule_dom  = input_dataframes[STR_DF_IN_FORECAST_RULE][COL_SHIP_TO].to_numpy(dtype=object)
        rule_vals = input_dataframes[STR_DF_IN_FORECAST_RULE][[
                        COL_RULE_GC, COL_RULE_AP2,
                        COL_RULE_AP1]].to_numpy(dtype='int32')

        rule_dict = {(pg, dom): tuple(vals)
             for pg, dom, vals in zip(rule_pg, rule_dom, rule_vals)}

        
        ################################################################################################################
        # Start of processing
        ################################################################################################################
        
        ################################################################################################################
        # Step 01-1 : call the helper from the main flow
        ################################################################################################################
        dict_log = {
            'p_step_no' : 101,
            'p_step_desc': 'Step 01-1 – RTS master pre-processing (filter ISVALID=Y, drop Version column)'
        }
        df_step01_1_RTS_clean = fn_step01_1_preprocess_rts(**dict_log)
        # log & store
        fn_log_dataframe(df_step01_1_RTS_clean, STR_DF_STEP01_1_RTS_CLEAN)
        output_dataframes[STR_DF_STEP01_1_RTS_CLEAN] = df_step01_1_RTS_clean

        ################################################################################################################
        # Step 01-2 : call from the main flow
        ################################################################################################################

        dict_log = {
            'p_step_no' : 102,
            'p_step_desc': 'Step 01-2 – Convert RTS dates to partial-week & add helper columns'
        }
        df_step01_2_RTS_PW = fn_step01_2_convert_rts_dates_to_pw(**dict_log)

        # log & register the result
        fn_log_dataframe(df_step01_2_RTS_PW, STR_DF_STEP01_2_RTS_PW)
        output_dataframes[STR_DF_STEP01_2_RTS_PW] = df_step01_2_RTS_PW

        ################################################################################################################
        # Step 01-3 : call the helper from the main flow
        ################################################################################################################
        dict_log = {
            'p_step_no' : 103,
            'p_step_desc': 'Step 01-3 – Extract distinct Item × Ship-To pairs from RTS data'
        }
        df_step01_3_RTS_distinct = fn_step01_3_extract_item_shipto_distinct(**dict_log)

        # log & register
        fn_log_dataframe(df_step01_3_RTS_distinct, STR_DF_STEP01_3_RTS_DISTINCT)
        output_dataframes[STR_DF_STEP01_3_RTS_DISTINCT] = df_step01_3_RTS_distinct

        ################################################################################################################
        # Step 01-4 : call helper from main flow
        ################################################################################################################
        dict_log = {
            'p_step_no' : 104,
            'p_step_desc': 'Step 01-4 – Cartesian Item×ShipTo with calendar; init Lock/Color'
        }
        df_step01_4_RTS_PW_LockColor = fn_step01_4_cartesian_with_calendar(**dict_log)

        # log & store
        fn_log_dataframe(df_step01_4_RTS_PW_LockColor, STR_DF_STEP01_4_RTS_PW_LOCKCOLOR)
        output_dataframes[STR_DF_STEP01_4_RTS_PW_LOCKCOLOR] = df_step01_4_RTS_PW_LockColor

        ################################################################################################################
        # Step 01-5 : call helper from main flow
        ################################################################################################################
        dict_log = {
            'p_step_no' : 105,
            'p_step_desc': 'Step 01-5 – apply RTS rules to derive final Lock/Color'
        }
        df_step01_5_RTS_LockColor_Final = fn_step01_5_apply_rts_rules(**dict_log)

        # log & register
        fn_log_dataframe(df_step01_5_RTS_LockColor_Final, STR_DF_STEP01_5_RTS_LOCKCOLOR_FINAL)
        output_dataframes[STR_DF_STEP01_5_RTS_LOCKCOLOR_FINAL] = df_step01_5_RTS_LockColor_Final

        """
            import duckdb
            for df in input_dataframes:
                duckdb.register(df, input_dataframes[df])

            v_item      = 'NT961XGK-K17/C'
            v_shipto    = '201'
            query_rts =  f'''
                select 
                    df_grid['{COL_ITEM}']               as '{COL_ITEM}',
                    df_grid['{COL_SHIP_TO}']            as '{COL_SHIP_TO}',
                    df_grid['{COL_TIME_PW}']            as '{COL_TIME_PW}',
                    {current_week_normalized}           as CUR_WEEK,
                    df_1_2['{COL_RTS_INITIAL_WEEK}'],
                    df_1_2['{COL_RTS_WEEK_MINUS_1}'],
                    df_1_2['{COL_RTS_WEEK_PLUS_3}']     as '{COL_RTS_WEEK_PLUS_3}',
                    df_1_2['{COL_MAX_RTS_CURRENTWEEK}'] as '{COL_MAX_RTS_CURRENTWEEK}',
                    {max_week_normalized}               as MAX_WEEK,
                    df_grid['{COL_LOCK_COND}'],
                    df_1_5['{COL_LOCK_COND}'],
                    df_grid['{COL_COLOR}'],
                    df_1_5['{COL_COLOR}']
                    
                from '{STR_DF_STEP01_2_RTS_PW}' as df_1_2
                join '{STR_DF_STEP01_4_RTS_PW_LOCKCOLOR}' as df_grid on 
                    df_1_2['{COL_SHIP_TO}'] = df_grid['{COL_SHIP_TO}']
                    and df_1_2['{COL_ITEM}'] = df_grid['{COL_ITEM}']
                join '{STR_DF_STEP01_5_RTS_LOCKCOLOR_FINAL}' as df_1_5 on 
                    df_1_5['{COL_SHIP_TO}']     = df_grid['{COL_SHIP_TO}']
                    and df_1_5['{COL_ITEM}']    = df_grid['{COL_ITEM}'] 
                    and df_1_5['{COL_TIME_PW}'] = df_grid['{COL_TIME_PW}'] 
                where 1=1 
                -- and df_1_2['{COL_SHIP_TO}'] = '{v_shipto}'
                -- and df_1_2['{COL_ITEM}'] = '{v_item}'
                and df_1_5['{COL_COLOR}'] = '{COLOR_DARKBLUE}'
                -- and df_grid['{COL_TIME_PW}'] = '202505B'
            '''
            result_rts = duckdb.query(query_rts).to_df()
            fn_log_dataframe(result_rts, 'result_rts')
        """


        ################################################################################################################
        # Step 02 – invoke from main flow
        ################################################################################################################
        dict_log = {
            'p_step_no' : 200,
            'p_step_desc': 'Step 2 – BAS·MOBILE 8-week 13_GREEN update'
        }
        df_step02_BAS_13_GREEN = fn_step02_bas_mobile_green(**dict_log)

        # log & store
        fn_log_dataframe(df_step02_BAS_13_GREEN, STR_DF_STEP02_BAS_13_GREEN)
        output_dataframes[STR_DF_STEP02_BAS_13_GREEN] = df_step02_BAS_13_GREEN    
        

        ################################################################################################################
        # Step 3-1 – call from main flow
        ################################################################################################################
        dict_log = {
            'p_step_no' : 301,
            'p_step_desc': 'Step 3-1 – generate max-TATTERM table'
        }
        df_step03_1_Item_TATTERM = fn_step03_1_generate_tatterm(**dict_log)

        fn_log_dataframe(df_step03_1_Item_TATTERM, STR_DF_STEP03_1_TAT_MAX)
        output_dataframes[STR_DF_STEP03_1_TAT_MAX] = df_step03_1_Item_TATTERM

        ################################################################################################################
        # Step 3-2 – call from main
        ################################################################################################################
        dict_log = {
            'p_step_no' : 302,
            'p_step_desc': 'Step 3-2 – VD/SHA lead-time window → 18_DGRAY_RED'
        }
        df_step03_2_VD_LEADTIME = fn_step03_2_apply_vd_leadtime(**dict_log)

        fn_log_dataframe(df_step03_2_VD_LEADTIME, STR_DF_STEP03_2_VD_LEAD)
        output_dataframes[STR_DF_STEP03_2_VD_LEAD] = df_step03_2_VD_LEADTIME

        """
            # ──[ DuckDB validation - Step 3-2 ]─────────────────────────────────────────────
            import duckdb
            # Handy aliases ↓ – adjust if you renamed anything
            STR_DF_PRE   = STR_DF_STEP02_BAS_13_GREEN      # after Step 2
            STR_DF_POST  = STR_DF_STEP03_2_VD_LEAD         # after Step 3-2
            STR_DF_TAT   = STR_DF_STEP03_1_TAT_MAX         # (GBM,Item) → TATTERM
            STR_DF_ITEM  = STR_DF_IN_ITEM_MASTER           # to display Item GBM

            duckdb.register(STR_DF_PRE,  output_dataframes[STR_DF_PRE])
            duckdb.register(STR_DF_TAT,  output_dataframes[STR_DF_TAT])
            duckdb.register(STR_DF_POST, output_dataframes[STR_DF_POST])
            duckdb.register(STR_DF_ITEM, input_dataframes[STR_DF_ITEM])

            # target for the quick check
            v_item   = 'RF65DG90BDSGTL'
            v_shipto = '300114'

            analysis_query_step03_2 = f'''
            SELECT
                pre['{COL_ITEM}']                     AS item,
                itm['{COL_ITEM_GBM}']                 AS item_gbm,
                pre['{COL_SHIP_TO}']                  AS shipto,
                pre['{COL_TIME_PW}']                  AS week_pw,
                pre['{COL_COLOR}']                    AS colour_before,
                post['{COL_COLOR}']                   AS colour_after,
                pre['{COL_LOCK_COND}']                AS lock_before,
                post['{COL_LOCK_COND}']               AS lock_after,
                tat['{COL_TATTERM}']                  AS tatterm,
                tat['{COL_CW_PLUS_TATTERM}']          AS cw_plus_tatterm,
                {current_week_normalized}             AS curr_week
            FROM   {STR_DF_PRE}  AS pre
            JOIN   {STR_DF_POST} AS post
                ON  pre['{COL_ITEM}']      = post['{COL_ITEM}']
                AND pre['{COL_SHIP_TO}']   = post['{COL_SHIP_TO}']
                AND pre['{COL_TIME_PW}']   = post['{COL_TIME_PW}']
            LEFT  JOIN {STR_DF_TAT}  AS tat
                ON  tat['{COL_ITEM}']      = pre['{COL_ITEM}']
            LEFT  JOIN {STR_DF_ITEM} AS itm
                ON  itm['{COL_ITEM}']      = pre['{COL_ITEM}']
            WHERE  pre['{COL_ITEM}']    = '{v_item}'
            -- AND  pre['{COL_SHIP_TO}'] = '{v_shipto}'
            ORDER  BY week_pw;
            '''

            df_validation_step03_2 = duckdb.query(analysis_query_step03_2).to_df()
            fn_log_dataframe(df_validation_step03_2, 'analysis_result_step03_2')
        """

        ################################################################################################################
        # Step 4-1 – call from main flow
        ################################################################################################################
        dict_log = {
            'p_step_no' : 401,
            'p_step_desc': 'Step 4-1 – Sales-Product-ASN preprocessing & calendar grid'
        }
        df_step04_1_ASN_GRID = fn_step04_1_sales_product_asn_preprocess(**dict_log)

        fn_log_dataframe(df_step04_1_ASN_GRID, STR_DF_STEP04_1_ASN_GRID)
        output_dataframes[STR_DF_STEP04_1_ASN_GRID] = df_step04_1_ASN_GRID

        #  OPTIONAL DuckDB validation – uncomment when needed
        """
            import duckdb
            duckdb.register('pre_asn', input_dataframes[STR_DF_IN_SALES_PRODUCT_ASN])
            duckdb.register('grid',    output_dataframes[STR_DF_STEP04_1_ASN_GRID])

            v_item   = 'RF29BB8600QLAA'
            v_shipto = 'A5002453'

            q = f'''
            SELECT
                g['{COL_ITEM}']      AS item,
                g['{COL_ITEM_PG}']   AS prod_grp,
                g['{COL_SHIP_TO}']   AS shipto,
                g['{COL_LOCATION}']  AS loc,
                g['{COL_TIME_PW}']   AS week_pw,
                g['{COL_COLOR}']     AS colour,
                g['{COL_LOCK_COND}'] AS lock
            FROM   grid g
            WHERE  g['{COL_ITEM}']    = '{v_item}'
            AND  g['{COL_SHIP_TO}'] = '{v_shipto}'
            ORDER  BY week_pw
            
            '''
            df_dbg_41 = duckdb.query(q).to_df()
            fn_log_dataframe(df_dbg_41, 'analysis_result_step04_1')
        """

        ################################################################################################################
        # Step 4-2 : call from main flow
        ################################################################################################################
        dict_log = {
            'p_step_no'  : 402,
            'p_step_desc': 'Step 4-2 – propagate lock/colour to leaf Ship-To'
        }
        df_step04_2_ASN_LOCKCOLOR = fn_step04_2_apply_rts_to_asn(**dict_log)

        fn_log_dataframe(df_step04_2_ASN_LOCKCOLOR, STR_DF_STEP04_2_ASN_LOCKCOLOR)
        output_dataframes[STR_DF_STEP04_2_ASN_LOCKCOLOR] = df_step04_2_ASN_LOCKCOLOR

        """
            # ════════════════════════ DUCKDB VALIDATION ══════════════════════════
            import duckdb, os
            duckdb.register('grid_pre',  output_dataframes[STR_DF_STEP04_1_ASN_GRID])
            duckdb.register('grid_post', output_dataframes[STR_DF_STEP04_2_ASN_LOCKCOLOR])
            v_item   = 'AC110AX4FHH1PP'
            v_ship   = 'A5002453'
            q = f'''
            SELECT 
                pre['{COL_SHIP_TO}'] AS shipto,
                pre['{COL_ITEM}']  AS item,
                pre['{COL_LOCATION}']  AS location,
                pre['{COL_ITEM_PG}']  AS product_group,
                pre['{COL_TIME_PW}'] AS week,
                pre['{COL_COLOR}']   AS col_before,
                post['{COL_COLOR}']  AS col_after,
                pre['{COL_LOCK_COND}']  AS lock_before,
                post['{COL_LOCK_COND}'] AS lock_after
            FROM   grid_pre  AS pre
            JOIN   grid_post AS post
                ON pre['{COL_ITEM}']      = post['{COL_ITEM}']
                AND pre['{COL_SHIP_TO}']   = post['{COL_SHIP_TO}']
                AND pre['{COL_LOCATION}']   = post['{COL_LOCATION}']
                AND pre['{COL_ITEM_PG}']   = post['{COL_ITEM_PG}']
                AND pre['{COL_TIME_PW}']   = post['{COL_TIME_PW}']
            WHERE 1 = 1 
            -- AND pre['{COL_ITEM}'] = '{v_item}'
            -- AND pre['{COL_SHIP_TO}'] = '{v_ship}'
            -- ORDER BY week;
            '''

            df_chk = duckdb.query(q).to_df()
            fn_log_dataframe(df_chk, 'analysis_result_step04_2')

        """

        ################################################################################################################
        # Step 05-1 :  Build Sell-Out-Master lock table
        ################################################################################################################
        dict_log = {
            'p_step_no': 501,
            'p_step_desc': 'Step 05-1 – df_in_Sell_Out_Simul_Master lock-column preparation'
        }
        df_step05_1_SO_Master_Lock = fn_step05_1_prepare_sellout_master(**dict_log)# log & store
        fn_log_dataframe(df_step05_1_SO_Master_Lock, STR_DF_STEP05_1_SO_MASTER_LOCK)
        output_dataframes[STR_DF_STEP05_1_SO_MASTER_LOCK] = df_step05_1_SO_Master_Lock

        # ────────────────────────── local DuckDB check (comment) ──────────────────────────
        """    
        # ═══════════════════════════════════════════════════════════════════
        #  DUCKDB VALIDATION – quick peek at master-lock result
        # ═══════════════════════════════════════════════════════════════════
        import duckdb
        duckdb.register(STR_DF_STEP05_1_SO_MASTER_LOCK,
                        output_dataframes[STR_DF_STEP05_1_SO_MASTER_LOCK])

        df_chk = duckdb.query(f'''
            SELECT *
            FROM   {STR_DF_STEP05_1_SO_MASTER_LOCK}
        ''')
        # .show()

        fn_log_dataframe(df_chk, 'analysis_result_step05_1')
        """

        ################################################################################################################
        # Step 05-2 : apply Master-based overrides (Lock=True, Color=19_GRAY)
        ################################################################################################################
        # dict_log = {
        #     'p_step_no' : 522,
        #     'p_step_desc': 'Step 05-2 – override Lock/Color for non-Master keys'
        # }
        # df_step05_2_Grid_MasterApplied = fn_step05_2_apply_master_lockcolor(**dict_log)
        # fn_log_dataframe(df_step05_2_Grid_MasterApplied, STR_DF_STEP05_2_GRID_UPD)
        # output_dataframes[STR_DF_STEP05_2_GRID_UPD] = df_step05_2_Grid_MasterApplied 

        # ───────── DuckDB validation snippet (comment-out in production) ────────
        """    
        import duckdb
        duckdb.register('grid_pre',  output_dataframes[STR_DF_STEP04_2_ASN_LOCKCOLOR])
        duckdb.register('grid_post', output_dataframes[STR_DF_STEP05_2_GRID_UPD])

        duckdb.query(f'''
            -- quick sanity: how many rows were forced to GRAY ?
            SELECT COUNT(*) AS overridden
            FROM   grid_post
            WHERE  "{COL_COLOR}" = '{COLOR_GRAY}'
            AND  NOT EXISTS (
                    SELECT 1
                    FROM   {STR_DF_STEP05_1_SO_MASTER_LOCK} m
                    WHERE  m."{COL_SHIP_TO}"  = grid_post."{COL_SHIP_TO}"
                    AND  m."{COL_ITEM_PG}" = grid_post."{COL_ITEM_PG}"
                );
        ''').show()

        df_chk = duckdb.query(f'''
            -- quick sanity: how many rows were forced to GRAY ?
            SELECT *
            FROM   grid_post
            WHERE  "{COL_COLOR}" = '{COLOR_GRAY}'
            AND  NOT EXISTS (
                    SELECT 1
                    FROM   {STR_DF_STEP05_1_SO_MASTER_LOCK} m
                    WHERE  m."{COL_SHIP_TO}"  = grid_post."{COL_SHIP_TO}"
                    AND  m."{COL_ITEM_PG}" = grid_post."{COL_ITEM_PG}"
                )
        ''').to_df()
        # .show()
        fn_log_dataframe(df_chk, 'analysis_result_step05_2')
        """       

        ################################################################################################################
        # Step 05-3 : hierarchical Master override (Lv-6 parents + Lv-7 children)
        ################################################################################################################
        dict_log = {
            'p_step_no' : 533,
            'p_step_desc': 'Step 05-3 – Master lock/colour with Lv-6 hierarchy'
        }
        df_step05_3_Grid_MasterHier = fn_step05_3_apply_master_hier_lockcolor(**dict_log)
        fn_log_dataframe(df_step05_3_Grid_MasterHier, STR_DF_STEP05_3_GRID_HIER)
        output_dataframes[STR_DF_STEP05_3_GRID_HIER] = df_step05_3_Grid_MasterHier

        # ───────── Optional DuckDB validation (comment-out) ────────────────────
        """
            import duckdb
            duckdb.register('grid_pre',                     output_dataframes[STR_DF_STEP04_2_ASN_LOCKCOLOR])
            duckdb.register('grid_post',                    output_dataframes[STR_DF_STEP05_3_GRID_HIER])
            duckdb.register('master',                       output_dataframes[STR_DF_STEP05_1_SO_MASTER_LOCK])
            duckdb.register(STR_DF_IN_SALES_DOMAIN_DIM,     input_dataframes[STR_DF_IN_SALES_DOMAIN_DIM])
            
            # 반영된 갯수
            duckdb.query(f'''
                -- rows now locked because parent Lv-6 not in Master?
                SELECT COUNT(*) AS forced
                FROM grid_post
                WHERE "{COL_COLOR}" = '{COLOR_GRAY}'
                AND NOT EXISTS (
                    SELECT 1
                    FROM master m
                    WHERE m."{COL_PROD_GRP}" = grid_post."{COL_PROD_GRP}"
                        AND (
                            m."{COL_SHIP_TO}" = grid_post."{COL_SHIP_TO}"
                            OR m."{COL_SHIP_TO}" IN (
                                SELECT "{COL_LV6}"
                                FROM   {STR_DF_IN_SALES_DOMAIN_DIM}
                                WHERE  "{COL_LV7}" = grid_post."{COL_SHIP_TO}"
                            )
                        )
                );
            ''').show()

            # 이전단계 
            duckdb.query(f'''
                -- rows now locked because parent Lv-6 not in Master?
                SELECT COUNT(*) AS forced
                FROM grid_pre
                
            ''').show()
        """

        ################################################################################################################
        # Step 06 : Sell-Out Forecast-Rule outputs (GC / AP2 / AP1 / Colour)
        ################################################################################################################
        dict_log = {
            'p_step_no' : 601,
            'p_step_desc': 'Step 06 – Build Sell-Out Forecast outputs from rules'
        }

        (df_output_Sell_Out_FCST_GC_Lock,
        df_output_Sell_Out_FCST_AP2_Lock,
        df_output_Sell_Out_FCST_AP1_Lock,
        df_output_Sell_Out_FCST_Color_Condition) = fn_step06_build_sellout_forecast_outputs(**dict_log)
        # store
        fn_log_dataframe(df_output_Sell_Out_FCST_GC_Lock,  STR_DF_OUT_FCST_GC_LOCK)
        fn_log_dataframe(df_output_Sell_Out_FCST_AP2_Lock, STR_DF_OUT_FCST_AP2_LOCK)
        fn_log_dataframe(df_output_Sell_Out_FCST_AP1_Lock, STR_DF_OUT_FCST_AP1_LOCK)
        fn_log_dataframe(df_output_Sell_Out_FCST_Color_Condition, STR_DF_OUT_FCST_COLOR)

        output_dataframes[STR_DF_OUT_FCST_GC_LOCK]  = df_output_Sell_Out_FCST_GC_Lock
        output_dataframes[STR_DF_OUT_FCST_AP2_LOCK] = df_output_Sell_Out_FCST_AP2_Lock
        output_dataframes[STR_DF_OUT_FCST_AP1_LOCK] = df_output_Sell_Out_FCST_AP1_Lock
        output_dataframes[STR_DF_OUT_FCST_COLOR]    = df_output_Sell_Out_FCST_Color_Condition

        """
            import duckdb
            duckdb.register('df_res', df_output_Sell_Out_FCST_Color_Condition)
            query = f'''
                select * from df_res 
                where "{COL_ITEM}" = 'RS23A500ASR/AA'
                and "{COL_TIME_PW}" >= '202506'
                and "{COL_TIME_PW}" <= '202521'
                -- and "{COL_SHIP_TO}" = '300114'
                -- and "{COL_SHIP_TO}" = '408351'
                and "{COL_SHIP_TO}" = '5019194'
            '''
            duckdb.query(query).show()

            # 이후구간 
            query = f'''
                select * from df_res 
                where "{COL_ITEM}" = 'RS23A500ASR/AA'
                and "{COL_TIME_PW}" > '202520'
            '''
            duckdb.query(query).show()
        """

        ################################################################################################################
        # 최종 Output 정리
        ################################################################################################################
        # dict_log = {
        #     'p_step_no': 900,
        #     'p_step_desc': '최종 Output 정리 - out_Demand',
        #     'p_df_name': 'out_Demand'
        # }
        # out_Demand = fn_output_formatter(df_fn_RTS_Week_Simul_Forecast, Version, **dict_log)
        


    except Exception as e:
        trace_msg = traceback.format_exc()
        logger.Note(p_note=trace_msg, p_log_level=LOG_LEVEL.debug())
        logger.Error()
        if flag_exception:
            raise Exception(e)
        else:
            logger.info(f'{str_instance} exit - {time.strftime("%Y-%m-%d - %H:%M:%S")}')


    finally:
        # MediumWeight 실행 시 Header 없는 빈 데이터프레임이 Output이 되는 경우 오류가 발생함.
        # 이 오류를 방지하기 위해 Output이 빈 경우을 체크하여 Header를 만들어 줌.
        if out_Demand.empty:
            out_Demand = fn_set_header()
            fn_log_dataframe(out_Demand, 'out_Demand')

        
        if is_local:
            log_file_name = common.G_PROGRAM_NAME.replace('py', 'log')
            log_file_name = f'log/{log_file_name}'

            shutil.copyfile(log_file_name, os.path.join(str_output_dir, os.path.basename(log_file_name)))

            # prografile copy
            program_path = f"{os.getcwd()}/NSCM_DP_UI_Develop/{str_instance}.py"
            shutil.copyfile(program_path, os.path.join(str_output_dir, os.path.basename(program_path)))

            # # task.json copy
            # task_path = f"{os.getcwd()}/.vscode/tasks.json"
            # shutil.copyfile(task_path, os.path.join(str_output_dir, os.path.basename(task_path)))

            # log
            input_path = f'{str_output_dir}/input'
            os.makedirs(input_path,exist_ok=True)
            for input_file in input_dataframes:
                input_dataframes[input_file].to_csv(input_path + "/"+input_file+".csv", encoding="UTF8", index=False)

            # # log
            # output_path = f'{str_output_dir}/output'
            # os.makedirs(output_path,exist_ok=True)
            # for output_file in output_dataframes:
            #     output_dataframes[output_file].to_csv(output_path + "/"+output_file+".csv", encoding="UTF8", index=False)

        # logger.info(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::')
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::') # 25.05.12 need warning Log by Logger Issue
        logger.Finish()
        





