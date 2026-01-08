"""
	Python 개발요청서 : PYDPDummyPrice.py			
				
		개요 (*)		
				
			* 프로그램명	
				PYDPDummyPrice
				
				
			* 목적	
				Sales의 상위Hierarchy에서  FCST 값을 입력(AP1 혹은 AP2) 했을 경우, 하위 Ship to 기준으로 Estimated Price_USD 산출하는 로직
				주차별로 돌아가며 당월의 당주차 ~ 부터만 반영한다. Procedure 로 Python 실행 시 Time 옵션 제공 
                    ex) Time.[Partial Week].filter (#.Key >= &CurrentPartialWeek && #.Key <= &CurrentMonth.element(0).leadoffset(1).key)
				S/In Fcst GI AP1 을 기준으로 Dummy Price를 계산
				Forecast 값이 없는 경우 빈 주차 Data가 들어올 수 있지만, 0으로 채우기 때문에, 주차 Data는 반드시 존재함.
				
			* 프로그램 가이드
                - 메모리절약
                    - 사용되는 데이타가 1억건 이상을 가정하기 때문에 메모리를 최소화 한다.
                    - Vectorize 사용.
                        - 데이타용량이 엄청 많은관계로 for 문을 사용하면 안된다. 
                    - category datatype 사용.
                        object 대신 category 를 사용한다.
                    - 이후 process 에서 더이상 사용하지 않는 object 들은 즉시 삭제한다.
                        del df
                        gc.collect()


				
			* 변경이력	
				2025.04.04 전창민 최초 작성
				2025.07.18 Price 값이 있는 경우에만 생성, FCST값 0인 경우 Price가 0이 될떄 예외로직 추가
                (25.07.21)
                    Step 3-1) 에서 Step2-1) 을 기준으로 outer join이 아니라 left outer join 적용
                    - Estimated Price_USD 를 기준으로  로직 진행
                    - Null 값인 경우 0 적용 
                    * Estimated Price_USD = 0 이고, FCST = 1 인 경우 발생하는 로직 문제 제거

                                            
                    Step 3-3-2) 에서 아래 로직 추가
                    FCST가 0이라서 Estimated Price_USD를 계산할 수 없는 경우 기존 Estimated Price_USD 값 사용 


				
		Script Parameter		
				
				Version
				- CWV_DP
				

    ====    Input Tables (*)	===================						
                                        
                                        
                (Input 1) df_in_Sin_GI_AP1					
                    - query					
                        Select (				
                        [Version].[Version Name].[CWV_DP]		// 당주주차 ~ 당월주차
                        * [Item].[Item]				        // USD가 6Lv로 들어오니, FCST 값도 6Lv 들어와서 계산
                        * [Time].[Partial Week].filter (#.Key >= &CurrentPartialWeek.element(0).Key && #.Key <= &CurrentMonth.element(0).leadoffset(1).Key)				
                        * [Sales Domain].[Sales Std5]				
                        ) on row, 				
                        ({Measure.[S/In FCST(GI)_AP1]}) on column;	

                    - data ( 20250718)
                        Version.[Version Name]	Item.[Item]	Time.[Partial Week]	Sales Domain.[Sales Std5]	S/In FCST(GI)_AP1	
                        CWV_DP	SKU 6	202443A	6000001	50	6Lv 로직 적용 Case
                        CWV_DP	SKU 6	202444A	6000001	100	
                        CWV_DP	SKU 6	202445A	6000001	150	
                        CWV_DP	SKU 6	202445B	6000001	200	
                        CWV_DP	SKU 6	202446A	6000001	250	
                        CWV_DP	SKU 6	202447A	6000001	300	
                        CWV_DP	SKU 6	202443A	6000002	25	6Lv 로직 적용 Case
                        CWV_DP	SKU 6	202444A	6000002	50	
                        CWV_DP	SKU 6	202445A	6000002	75	
                        CWV_DP	SKU 6	202445B	6000002	100	
                        CWV_DP	SKU 6	202446A	6000002	125	
                        CWV_DP	SKU 6	202447A	6000002	150	
                        CWV_DP	SKU 6	202443A	6000003	25	6Lv 로직 적용 Case
                        CWV_DP	SKU 6	202444A	6000003	50	
                        CWV_DP	SKU 6	202445A	6000003	75	
                        CWV_DP	SKU 6	202445B	6000003	100	
                        CWV_DP	SKU 6	202446A	6000003	125	
                        CWV_DP	SKU 6	202447A	6000003	150	
                        CWV_DP	SKU 5	202443A	5000001	100	5Lv 로직 적용 Case
                        CWV_DP	SKU 5	202444A	5000001	200	
                        CWV_DP	SKU 5	202445A	5000001	300	
                        CWV_DP	SKU 5	202445B	5000001	400	
                        CWV_DP	SKU 5	202446A	5000001	500	
                        CWV_DP	SKU 5	202447A	5000001	600	
                        CWV_DP	SKU 5	202443A	5000002	100	5Lv 로직 적용 Case
                        CWV_DP	SKU 5	202444A	5000002	200	
                        CWV_DP	SKU 5	202445A	5000002	300	
                        CWV_DP	SKU 5	202445B	5000002	400	
                        CWV_DP	SKU 5	202446A	5000002	500	
                        CWV_DP	SKU 5	202447A	5000002	600	
                        CWV_DP	SKU 4	202443A	4000001	200	4Lv 로직 적용 Case
                        CWV_DP	SKU 4	202444A	4000001	400	
                        CWV_DP	SKU 4	202445A	4000001	600	
                        CWV_DP	SKU 4	202445B	4000001	800	
                        CWV_DP	SKU 4	202446A	4000001	1000	
                        CWV_DP	SKU 4	202447A	4000001	1200	
                        CWV_DP	SKU 4	202443A	4000002	200	4Lv 로직 적용 Case
                        CWV_DP	SKU 4	202444A	4000002	400	
                        CWV_DP	SKU 4	202445A	4000002	600	
                        CWV_DP	SKU 4	202445B	4000002	800	
                        CWV_DP	SKU 4	202446A	4000002	1000	
                        CWV_DP	SKU 4	202447A	4000002	1200	
                        CWV_DP	SKU 3	202443A	3000001	400	3Lv 로직 적용 Case
                        CWV_DP	SKU 3	202444A	3000001	800	
                        CWV_DP	SKU 3	202445A	3000001	1200	
                        CWV_DP	SKU 3	202445B	3000001	1600	
                        CWV_DP	SKU 3	202446A	3000001	2000	
                        CWV_DP	SKU 3	202447A	3000001	2400	
                        CWV_DP	SKU 3	202443A	3000002	400	3Lv 로직 적용 Case
                        CWV_DP	SKU 3	202444A	3000002	800	
                        CWV_DP	SKU 3	202445A	3000002	1200	
                        CWV_DP	SKU 3	202445B	3000002	1600	
                        CWV_DP	SKU 3	202446A	3000002	2000	
                        CWV_DP	SKU 3	202447A	3000002	2400	

                    - data ( 20250725) : 6level 의 데이타만 들어온다. 하위데이타가 들어오는 걸로 설계했었지만 오류임.                                    
                        Version.[Version Name]	Item.[Item]	Time.[Partial Week]	Sales Domain.[Sales Std5]	S/In FCST(GI)_AP1	
                        CWV_DP	SKU 6	202443A	6000001	50	6Lv 로직 적용 Case
                        CWV_DP	SKU 6	202444A	6000001	100	
                        CWV_DP	SKU 6	202445A	6000001	150	
                        CWV_DP	SKU 6	202445B	6000001	200	
                        CWV_DP	SKU 6	202446A	6000001	250	
                        CWV_DP	SKU 6	202447A	6000001	300	
                        CWV_DP	SKU 6	202443A	6000002	25	6Lv 로직 적용 Case
                        CWV_DP	SKU 6	202444A	6000002	50	
                        CWV_DP	SKU 6	202445A	6000002	75	
                        CWV_DP	SKU 6	202445B	6000002	100	
                        CWV_DP	SKU 6	202446A	6000002	125	
                        CWV_DP	SKU 6	202447A	6000002	150	
                        CWV_DP	SKU 6	202443A	6000003	25	6Lv 로직 적용 Case

                (Input 2) df_In_EstimatedPrice_USD  exchange rate 실적					
                    - query				
                        Select (			
                        [Version].[Version Name].[CWV_DP]			
                        * [Item].[Item]			
                        * [Time].[Partial Week].filter (#.Key >= &CurrentPartialWeek.element(0).Key && #.Key <= &CurrentMonth.element(0).leadoffset(1).Key)			
                        * [Sales Domain].[Ship To].filter(#.Name startswith([5]) ) on row, 			
                        ({Measure.[Estimated Price_USD]}) 			
                        on column;			
                                    
                    - data
                        Version.[Version Name]	Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	Estimated Price_USD
                        CWV_DP	SKU 6	202443A	6000001	200
                        CWV_DP	SKU 6	202444A	6000001	200
                        CWV_DP	SKU 6	202445A	6000001	200
                        CWV_DP	SKU 6	202445B	6000001	200
                        CWV_DP	SKU 6	202446A	6000001	200
                        CWV_DP	SKU 6	202447A	6000001	200
                        CWV_DP	SKU 6	202443A	6000002	100
                        CWV_DP	SKU 6	202444A	6000002	100
                        CWV_DP	SKU 6	202445A	6000002	100
                        CWV_DP	SKU 6	202445B	6000002	100
                        CWV_DP	SKU 6	202446A	6000002	100
                        CWV_DP	SKU 6	202447A	6000002	100
                        CWV_DP	SKU 6	202443A	6000003	100
                        CWV_DP	SKU 6	202444A	6000003	100
                        CWV_DP	SKU 6	202445A	6000003	100
                        CWV_DP	SKU 6	202445B	6000003	100
                        CWV_DP	SKU 6	202446A	6000003	100
                        CWV_DP	SKU 6	202447A	6000003	100
                        CWV_DP	SKU 5	202443A	6000001	100
                        CWV_DP	SKU 5	202444A	6000001	100
                        CWV_DP	SKU 5	202445A	6000001	100
                        CWV_DP	SKU 5	202445B	6000001	100
                        CWV_DP	SKU 5	202446A	6000001	100
                        CWV_DP	SKU 5	202447A	6000001	100
                        CWV_DP	SKU 5	202443A	6000002	100
                        CWV_DP	SKU 5	202444A	6000002	100
                        CWV_DP	SKU 5	202445A	6000002	100
                        CWV_DP	SKU 5	202445B	6000002	100
                        CWV_DP	SKU 5	202446A	6000002	100
                        CWV_DP	SKU 5	202447A	6000002	100
                        CWV_DP	SKU 4	202443A	6000001	100
                        CWV_DP	SKU 4	202444A	6000001	100
                        CWV_DP	SKU 4	202445A	6000001	100
                        CWV_DP	SKU 4	202445B	6000001	100
                        CWV_DP	SKU 4	202446A	6000001	100
                        CWV_DP	SKU 4	202447A	6000001	100
                        CWV_DP	SKU 4	202443A	6000002	100
                        CWV_DP	SKU 4	202444A	6000002	100
                        CWV_DP	SKU 4	202445A	6000002	100
                        CWV_DP	SKU 4	202445B	6000002	100
                        CWV_DP	SKU 4	202446A	6000002	100
                        CWV_DP	SKU 4	202447A	6000002	100
                        CWV_DP	SKU 3	202443A	6000001	1000
                        CWV_DP	SKU 3	202444A	6000001	1000
                        CWV_DP	SKU 3	202445A	6000001	1000
                        CWV_DP	SKU 3	202445B	6000001	1000
                        CWV_DP	SKU 3	202446A	6000001	1000
                        CWV_DP	SKU 3	202447A	6000001	1000
                        CWV_DP	SKU 3	202443A	6000002	1000
                        CWV_DP	SKU 3	202444A	6000002	1000
                        CWV_DP	SKU 3	202445A	6000002	1000
                        CWV_DP	SKU 3	202445B	6000002	1000
                        CWV_DP	SKU 3	202446A	6000002	1000
                        CWV_DP	SKU 3	202447A	6000002	1000
                        CWV_DP	SKU 3	202443A	6000003	1000
                        CWV_DP	SKU 3	202444A	6000003	1000
                        CWV_DP	SKU 3	202445A	6000003	1000
                        CWV_DP	SKU 3	202445B	6000003	1000
                        CWV_DP	SKU 3	202446A	6000003	1000
                        CWV_DP	SKU 3	202447A	6000003	1000
                        CWV_DP	SKU 3	202443A	6000009	1000
                        CWV_DP	SKU 3	202444A	6000009	1000
                        CWV_DP	SKU 3	202445A	6000009	1000
                        CWV_DP	SKU 3	202445B	6000009	1000
                        CWV_DP	SKU 3	202446A	6000009	1000
                        CWV_DP	SKU 3	202447A	6000009	1000


                (Input 3) df_in_Sales_Domain_Dimension	
                    - query	
                        Select (
                        [Sales Domain].[Sales Std1]
                        * [Sales Domain].[Sales Std2] 
                        * [Sales Domain].[Sales Std3] 
                        * [Sales Domain].[Sales Std4] 
                        * [Sales Domain].[Sales Std5] 
                        * [Sales Domain].[Sales Std6] 
                        * [Sales Domain].[Ship To] );
                        
                    - data
                        Sales Domain.[Sales Std1]	Sales Domain.[Sales Std2]	Sales Domain.[Sales Std3]	Sales Domain.[Sales Std4]	Sales Domain.[Sales Std5]	Sales Domain.[Sales Std6]	Sales Domain.[Ship To]
                        201	201	201	201	201	201	201
                        201	3000001	3000001	3000001	3000001	3000001	3000001
                        201	3000001	4000001	4000001	4000001	4000001	4000001
                        201	3000001	4000001	5000001	5000001	5000001	5000001
                        201	3000001	4000001	5000001	6000001	6000001	6000001
                        201	3000001	4000001	5000001	6000001	6000001	6000001
                        201	3000001	4000002	4000002	4000002	4000002	4000002
                        201	3000001	4000002	5000002	5000002	5000002	5000002
                        201	3000001	4000002	5000002	6000002	6000002	6000002
                        201	3000001	4000002	5000002	6000002	7000002	7000002
                        201	3000001	4000002	5000002	6000003	6000003	6000003
                        201	3000001	4000002	5000002	6000003	7000003	7000003
                        201	3000001	4000002	5000002	6000003	7000004	7000004
                        201	3000002	3000002	3000002	3000002	3000002	3000002
                        201	3000002	4000009	4000009	4000009	4000009	4000009
                        201	3000002	4000009	5000009	5000009	5000009	5000009
                        201	3000002	4000009	5000009	6000009	6000009	6000009
                        201	3000002	4000009	5000009	6000009	7000009	7000009


    ====    Output Tables (*)   =========
                        
                (Output 1)  df_output_Dummy_EstimatedPrice_USD
                    - query
                        Select ([Version].[Version Name]
                        * [Item].[Item]
                        * [Time].[Partial Week]
                        * [Sales Domain].[Ship To] on row, 
                        ({Measure.[Estimated Price_USD] }) on column;




    ====	주요 로직 (*)	=============				
                                
                Step 1) S/In FCST(GI)_AP1 전처리				
                    Step 1-1) Version.[Version Name] 삭제		
                    - 결과데이타 	
                    Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	S/In FCST(GI)_AP1
                    SKU 6	202443A	6000001	50
                    SKU 6	202444A	6000001	100
                    SKU 6	202445A	6000001	150
                    SKU 6	202445B	6000001	200
                    SKU 6	202446A	6000001	250
                    SKU 6	202447A	6000001	300


                    - Process
                        - load df_in_Sin_GI_AP1
                        - Version.[Version Name] 삭제
                        - rename : Sales Domain.[Sales Std5] -> Sales Domain.[Ship To]	


                    Step 1-2) Lv 별 분리			
                        - 3Lv			
                            Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	S/In FCST(GI)_AP1
                            SKU 3	202443A	3000001	400
                            SKU 3	202444A	3000001	800
                            SKU 3	202445A	3000001	1200
                            SKU 3	202445B	3000001	1600
                            SKU 3	202446A	3000001	2000
                            SKU 3	202447A	3000001	2400
                            SKU 3	202443A	3000002	400
                            SKU 3	202444A	3000002	800
                            SKU 3	202445A	3000002	1200
                            SKU 3	202445B	3000002	1600
                            SKU 3	202446A	3000002	2000
                            SKU 3	202447A	3000002	2400
                                
                        - 4Lv			
                            Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	S/In FCST(GI)_AP1
                            SKU 4	202443A	4000001	200
                            SKU 4	202444A	4000001	400
                            SKU 4	202445A	4000001	600
                            SKU 4	202445B	4000001	800
                            SKU 4	202446A	4000001	1000
                            SKU 4	202447A	4000001	1200
                            SKU 4	202443A	4000002	200
                            SKU 4	202444A	4000002	400
                            SKU 4	202445A	4000002	600
                            SKU 4	202445B	4000002	800
                            SKU 4	202446A	4000002	1000
                            SKU 4	202447A	4000002	1200
                                    
                        - 5Lv			
                            Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	S/In FCST(GI)_AP1
                            SKU 5	202443A	5000001	100
                            SKU 5	202444A	5000001	200
                            SKU 5	202445A	5000001	300
                            SKU 5	202445B	5000001	400
                            SKU 5	202446A	5000001	500
                            SKU 5	202447A	5000001	600
                            SKU 5	202443A	5000002	100
                            SKU 5	202444A	5000002	200
                            SKU 5	202445A	5000002	300
                            SKU 5	202445B	5000002	400
                            SKU 5	202446A	5000002	500
                            SKU 5	202447A	5000002	600
                                    
                                    
                        - 6Lv			
                            Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	S/In FCST(GI)_AP1
                            SKU 6	202443A	6000001	50
                            SKU 6	202444A	6000001	100
                            SKU 6	202445A	6000001	150
                            SKU 6	202445B	6000001	200
                            SKU 6	202446A	6000001	250
                            SKU 6	202447A	6000001	300
                            SKU 6	202443A	6000002	25
                            SKU 6	202444A	6000002	50
                            SKU 6	202445A	6000002	75
                            SKU 6	202445B	6000002	100
                            SKU 6	202446A	6000002	125
                            SKU 6	202447A	6000002	150
                            SKU 6	202443A	6000003	25
                            SKU 6	202444A	6000003	50
                            SKU 6	202445A	6000003	75
                            SKU 6	202445B	6000003	100
                            SKU 6	202446A	6000003	125
                            SKU 6	202447A	6000003	150
                    
                        - level 별로 분리할 수 없다. (20250725)
                            - 개요 
                                - 설계상 처음계획은 6레벨 이하의 데이타가 들어오기로 했지만 이는 설계상의 오류임.
                                - 6 Level 상위인  3,4,5 Level 의 데이타를 만들어 줘야 한다. 
                            - 6 Level 로 3,4,5 Level 만들기
                                - 3lv을 만들어준다.
                                    - add COL_STD2 by using COL_SHIP_TO
                                    - groupby COL_ITEM * COL_TIME_PW * COL_STD2
                                        - agg 
                                            - COL_SIN_FCST_AP1 : sum
                                    - set value to COL_SHIP_TO with COL_STD2
                                    - drop column COL_SHIP_TO

                                - 4lv을 만들어준다.
                                    - add COL_STD3 by using COL_SHIP_TO
                                    - groupby COL_ITEM * COL_TIME_PW * COL_STD3
                                        - agg 
                                            - COL_SIN_FCST_AP1 : sum
                                    - set value to COL_SHIP_TO with COL_STD3
                                    - drop column COL_SHIP_TO
                                - 5lv을 만들어준다.
                                    - add COL_STD4 by using COL_SHIP_TO
                                    - groupby COL_ITEM * COL_TIME_PW * COL_STD4
                                        - agg 
                                            - COL_SIN_FCST_AP1 : sum
                                    - set value to COL_SHIP_TO with COL_STD4
                                    - drop column COL_SHIP_TO
                                
                Step 2) Estimated Price_USD 하위 Data 생성 				
                    Step 2-1) Version.[Version Name] 삭제	
                        - 결과데이타		
                            Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	Estimated Price_USD
                            SKU 6	202443A	6000001	200
                            SKU 6	202444A	6000001	200
                            SKU 6	202445A	6000001	200
                            SKU 6	202445B	6000001	200
                            SKU 6	202446A	6000001	200
                            SKU 6	202447A	6000001	200
                            SKU 6	202443A	6000002	100
                            SKU 6	202444A	6000002	100
                            SKU 6	202445A	6000002	100
                            SKU 6	202445B	6000002	100
                            SKU 6	202446A	6000002	100
                            SKU 6	202447A	6000002	100
                            SKU 6	202443A	6000003	100
                            SKU 6	202444A	6000003	100
                            SKU 6	202445A	6000003	100
                            SKU 6	202445B	6000003	100
                            SKU 6	202446A	6000003	100
                            SKU 6	202447A	6000003	100
                            SKU 5	202443A	6000001	100
                            SKU 5	202444A	6000001	100
                            SKU 5	202445A	6000001	100
                            SKU 5	202445B	6000001	100
                            SKU 5	202446A	6000001	100
                            SKU 5	202447A	6000001	100
                            SKU 5	202443A	6000002	100
                            SKU 5	202444A	6000002	100
                            SKU 5	202445A	6000002	100
                            SKU 5	202445B	6000002	100
                            SKU 5	202446A	6000002	100
                            SKU 5	202447A	6000002	100
                            SKU 4	202443A	6000001	100
                            SKU 4	202444A	6000001	100
                            SKU 4	202445A	6000001	100
                            SKU 4	202445B	6000001	100
                            SKU 4	202446A	6000001	100
                            SKU 4	202447A	6000001	100
                            SKU 4	202443A	6000002	100
                            SKU 4	202444A	6000002	100
                            SKU 4	202445A	6000002	100
                            SKU 4	202445B	6000002	100
                            SKU 4	202446A	6000002	100
                            SKU 4	202447A	6000002	100
                            SKU 3	202443A	6000001	1000
                            SKU 3	202444A	6000001	1000
                            SKU 3	202445A	6000001	1000
                            SKU 3	202445B	6000001	1000
                            SKU 3	202446A	6000001	1000
                            SKU 3	202447A	6000001	1000
                            SKU 3	202443A	6000002	1000
                            SKU 3	202444A	6000002	1000
                            SKU 3	202445A	6000002	1000
                            SKU 3	202445B	6000002	1000
                            SKU 3	202446A	6000002	1000
                            SKU 3	202447A	6000002	1000
                            SKU 3	202443A	6000003	1000
                            SKU 3	202444A	6000003	1000
                            SKU 3	202445A	6000003	1000
                            SKU 3	202445B	6000003	1000
                            SKU 3	202446A	6000003	1000
                            SKU 3	202447A	6000003	1000
                            SKU 3	202443A	6000009	1000
                            SKU 3	202444A	6000009	1000
                            SKU 3	202445A	6000009	1000
                            SKU 3	202445B	6000009	1000
                            SKU 3	202446A	6000009	1000
                            SKU 3	202447A	6000009	1000
                        - Process
                            - load df_In_EstimatedPrice_USD
                            - Version.[Version Name] 삭제
                            - return df_fn_EstPrice_USD

                    Step 2-2) 7Lv(Sta6) Dummy Data 생성	
                        - 결과데이타			
                            Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	Sales Domain.[Sales Std6]	Estimated Price_USD
                            SKU 6	202443A	6000001		200
                            SKU 6	202444A	6000001		200
                            SKU 6	202445A	6000001		200
                            SKU 6	202445B	6000001		200
                            SKU 6	202446A	6000001		200
                            SKU 6	202447A	6000001		200
                            SKU 6	202443A	6000002	7000002	100
                            SKU 6	202444A	6000002	7000002	100
                            SKU 6	202445A	6000002	7000002	100
                            SKU 6	202445B	6000002	7000002	100
                            SKU 6	202446A	6000002	7000002	100
                            SKU 6	202447A	6000002	7000002	100
                            SKU 6	202443A	6000003	7000003	100
                            SKU 6	202444A	6000003	7000003	100
                            SKU 6	202445A	6000003	7000003	100
                            SKU 6	202445B	6000003	7000003	100
                            SKU 6	202446A	6000003	7000003	100
                            SKU 6	202447A	6000003	7000003	100
                            SKU 6	202443A	6000003	7000004	100
                            SKU 6	202444A	6000003	7000004	100
                            SKU 6	202445A	6000003	7000004	100
                            SKU 6	202445B	6000003	7000004	100
                            SKU 6	202446A	6000003	7000004	100
                            SKU 6	202447A	6000003	7000004	100
                            SKU 5	202443A	6000001		100
                            SKU 5	202444A	6000001		100
                            SKU 5	202445A	6000001		100
                            SKU 5	202445B	6000001		100
                            SKU 5	202446A	6000001		100
                            SKU 5	202447A	6000001		100
                            SKU 5	202443A	6000002	7000002	100
                            SKU 5	202444A	6000002	7000002	100
                            SKU 5	202445A	6000002	7000002	100
                            SKU 5	202445B	6000002	7000002	100
                            SKU 5	202446A	6000002	7000002	100
                            SKU 5	202447A	6000002	7000002	100
                            SKU 4	202443A	6000001		100
                            SKU 4	202444A	6000001		100
                            SKU 4	202445A	6000001		100
                            SKU 4	202445B	6000001		100
                            SKU 4	202446A	6000001		100
                            SKU 4	202447A	6000001		100
                            SKU 4	202443A	6000002	7000002	100
                            SKU 4	202444A	6000002	7000002	100
                            SKU 4	202445A	6000002	7000002	100
                            SKU 4	202445B	6000002	7000002	100
                            SKU 4	202446A	6000002	7000002	100
                            SKU 4	202447A	6000002	7000002	100
                            SKU 3	202443A	6000001		100
                            SKU 3	202444A	6000001		100
                            SKU 3	202445A	6000001		100
                            SKU 3	202445B	6000001		100
                            SKU 3	202446A	6000001		100
                            SKU 3	202447A	6000001		100
                            SKU 3	202443A	6000002	7000002	1000
                            SKU 3	202444A	6000002	7000002	1000
                            SKU 3	202445A	6000002	7000002	1000
                            SKU 3	202445B	6000002	7000002	1000
                            SKU 3	202446A	6000002	7000002	1000
                            SKU 3	202447A	6000002	7000002	1000
                            SKU 3	202443A	6000003	7000003	1000
                            SKU 3	202444A	6000003	7000003	1000
                            SKU 3	202445A	6000003	7000003	1000
                            SKU 3	202445B	6000003	7000003	1000
                            SKU 3	202446A	6000003	7000003	1000
                            SKU 3	202447A	6000003	7000003	1000
                            SKU 6	202443A	6000003	7000004	1000
                            SKU 6	202444A	6000003	7000004	1000
                            SKU 6	202445A	6000003	7000004	1000
                            SKU 6	202445B	6000003	7000004	1000
                            SKU 6	202446A	6000003	7000004	1000
                            SKU 6	202447A	6000003	7000004	1000
                            SKU 3	202443A	6000009	7000009	1000
                            SKU 3	202444A	6000009	7000009	1000
                            SKU 3	202445A	6000009	7000009	1000
                            SKU 3	202445B	6000009	7000009	1000
                            SKU 3	202446A	6000009	7000009	1000
                            SKU 3	202447A	6000009	7000009	1000
                        - 개요
                            - df_in_Sales_Domain_Dimension에서 Sales Domain.[Ship To]의 값이 '7' 경우만 추출하여 사용
                        - Process
                            - load df_fn_EstPrice_USD
                            - Step 2-1(df_fn_EstPrice_USD) 의 Sales Domain.[Ship To] 와 df_in_Sales_Domain_Dimension의 Sales Domain.[Sales Std5] 를 비교하여, Sales Domain.[Sales Std6] Data 구성				
                                - 매칭되는 Lv7 Data가 없는 경우 삭제				
                                - 1:n 관계			
                            - return df_fn_Dummy_Lv7

                    Step 2-3) 7Lv Dummy Data 생성	
                        - 결과데이타		
                            Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	Estimated Price_USD
                            SKU 6	202443A	7000002	100
                            SKU 6	202444A	7000002	100
                            SKU 6	202445A	7000002	100
                            SKU 6	202445B	7000002	100
                            SKU 6	202446A	7000002	100
                            SKU 6	202447A	7000002	100
                            SKU 6	202443A	7000003	100
                            SKU 6	202444A	7000003	100
                            SKU 6	202445A	7000003	100
                            SKU 6	202445B	7000003	100
                            SKU 6	202446A	7000003	100
                            SKU 6	202447A	7000003	100
                            SKU 6	202443A	7000004	100
                            SKU 6	202444A	7000004	100
                            SKU 6	202445A	7000004	100
                            SKU 6	202445B	7000004	100
                            SKU 6	202446A	7000004	100
                            SKU 6	202447A	7000004	100
                            SKU 5	202443A	7000002	100
                            SKU 5	202444A	7000002	100
                            SKU 5	202445A	7000002	100
                            SKU 5	202445B	7000002	100
                            SKU 5	202446A	7000002	100
                            SKU 5	202447A	7000002	100
                            SKU 4	202443A	7000002	100
                            SKU 4	202444A	7000002	100
                            SKU 4	202445A	7000002	100
                            SKU 4	202445B	7000002	100
                            SKU 4	202446A	7000002	100
                            SKU 4	202447A	7000002	100
                            SKU 3	202443A	7000002	1000
                            SKU 3	202444A	7000002	1000
                            SKU 3	202445A	7000002	1000
                            SKU 3	202445B	7000002	1000
                            SKU 3	202446A	7000002	1000
                            SKU 3	202447A	7000002	1000
                            SKU 3	202443A	7000003	1000
                            SKU 3	202444A	7000003	1000
                            SKU 3	202445A	7000003	1000
                            SKU 3	202445B	7000003	1000
                            SKU 3	202446A	7000003	1000
                            SKU 3	202447A	7000003	1000
                            SKU 6	202443A	7000004	1000
                            SKU 6	202444A	7000004	1000
                            SKU 6	202445A	7000004	1000
                            SKU 6	202445B	7000004	1000
                            SKU 6	202446A	7000004	1000
                            SKU 6	202447A	7000004	1000
                            SKU 3	202443A	7000009	1000
                            SKU 3	202444A	7000009	1000
                            SKU 3	202445A	7000009	1000
                            SKU 3	202445B	7000009	1000
                            SKU 3	202446A	7000009	1000
                            SKU 3	202447A	7000009	1000

                        - Process
                            - load df_fn_Dummy_Lv7
                            - Sales Domain.[Ship To] 삭제			
                            - rename : Sales Domain.[Sales Std6] -> Sales Domain.[Ship To]	
                            - return df_fn_Converted_Lv7		
                                
                Step 3) 5Lv Dummy Data 생성 (물량가중 처리, 없는 경우 하위값 그대로 사용)					
                    Step 3-1) Step 2-1) 의 Data 와 Step 1-2) Lv6 Data를 Merge		
                    - 결과데이타		
                        Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	Estimated Price_USD	S/In FCST(GI)_AP1
                        SKU 6	202443A	6000001	200	50
                        SKU 6	202444A	6000001	200	100
                        SKU 6	202445A	6000001	200	150
                        SKU 6	202445B	6000001	200	200
                        SKU 6	202446A	6000001	200	250
                        SKU 6	202447A	6000001	200	300
                        SKU 6	202443A	6000002	100	25
                        SKU 6	202444A	6000002	100	50
                        SKU 6	202445A	6000002	100	75
                        SKU 6	202445B	6000002	100	100
                        SKU 6	202446A	6000002	100	125
                        SKU 6	202447A	6000002	100	150
                        SKU 6	202443A	6000003	100	25
                        SKU 6	202444A	6000003	100	50
                        SKU 6	202445A	6000003	100	75
                        SKU 6	202445B	6000003	100	100
                        SKU 6	202446A	6000003	100	125
                        SKU 6	202447A	6000003	100	150
                        SKU 5	202443A	6000001	100	0
                        SKU 5	202444A	6000001	100	0
                        SKU 5	202445A	6000001	100	0
                        SKU 5	202445B	6000001	100	0
                        SKU 5	202446A	6000001	100	0
                        SKU 5	202447A	6000001	100	0
                        SKU 5	202443A	6000002	100	0
                        SKU 5	202444A	6000002	100	0
                        SKU 5	202445A	6000002	100	0
                        SKU 5	202445B	6000002	100	0
                        SKU 5	202446A	6000002	100	0
                        SKU 5	202447A	6000002	100	0
                        SKU 4	202443A	6000001	100	0
                        SKU 4	202444A	6000001	100	0
                        SKU 4	202445A	6000001	100	0
                        SKU 4	202445B	6000001	100	0
                        SKU 4	202446A	6000001	100	0
                        SKU 4	202447A	6000001	100	0
                        SKU 4	202443A	6000002	100	0
                        SKU 4	202444A	6000002	100	0
                        SKU 4	202445A	6000002	100	0
                        SKU 4	202445B	6000002	100	0
                        SKU 4	202446A	6000002	100	0
                        SKU 4	202447A	6000002	100	0
                        SKU 3	202443A	6000001	1000	0
                        SKU 3	202444A	6000001	1000	0
                        SKU 3	202445A	6000001	1000	0
                        SKU 3	202445B	6000001	1000	0
                        SKU 3	202446A	6000001	1000	0
                        SKU 3	202447A	6000001	1000	0
                        SKU 3	202443A	6000002	1000	0
                        SKU 3	202444A	6000002	1000	0
                        SKU 3	202445A	6000002	1000	0
                        SKU 3	202445B	6000002	1000	0
                        SKU 3	202446A	6000002	1000	0
                        SKU 3	202447A	6000002	1000	0
                        SKU 3	202443A	6000003	1000	0
                        SKU 3	202444A	6000003	1000	0
                        SKU 3	202445A	6000003	1000	0
                        SKU 3	202445B	6000003	1000	0
                        SKU 3	202446A	6000003	1000	0
                        SKU 3	202447A	6000003	1000	0
                        SKU 3	202443A	6000009	1000	0
                        SKU 3	202444A	6000009	1000	0
                        SKU 3	202445A	6000009	1000	0
                        SKU 3	202445B	6000009	1000	0
                        SKU 3	202446A	6000009	1000	0
                        SKU 3	202447A	6000009	1000	0
                    - Process 
                        - load df_fn_EstPrice_USD( df_lv6_price ) (Step 2-1)
                        - load df_lv6_fcst ( Step1-2 의 6Lv 데이타)
                        - df_lv5_price 생성
                            - Step 2-1(df_fn_EstPrice_USD) 을 기준으로 df_lv6_fcst 을 Left outer join (변동 기존에는 outer 였다.)		
                            - Null 값인 경우 0 적용	( 변동 : 기존에는 1 이었다.)			

                    Step 3-2) Sales Domain 5Lv Data 구성	
                        - 결과데이타				
                            Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	Estimated Price_USD	S/In FCST(GI)_AP1	Sales Domain.[Sales Std4]
                            SKU 6	202443A	6000001	200	50	5000001
                            SKU 6	202444A	6000001	200	100	5000001
                            SKU 6	202445A	6000001	200	150	5000001
                            SKU 6	202445B	6000001	200	200	5000001
                            SKU 6	202446A	6000001	200	250	5000001
                            SKU 6	202447A	6000001	200	300	5000001
                            SKU 6	202443A	6000002	100	25	5000002
                            SKU 6	202444A	6000002	100	50	5000002
                            SKU 6	202445A	6000002	100	75	5000002
                            SKU 6	202445B	6000002	100	100	5000002
                            SKU 6	202446A	6000002	100	125	5000002
                            SKU 6	202447A	6000002	100	150	5000002
                            SKU 6	202443A	6000003	100	25	5000002
                            SKU 6	202444A	6000003	100	50	5000002
                            SKU 6	202445A	6000003	100	75	5000002
                            SKU 6	202445B	6000003	100	100	5000002
                            SKU 6	202446A	6000003	100	125	5000002
                            SKU 6	202447A	6000003	100	150	5000002
                            SKU 5	202443A	6000001	100	0	5000001
                            SKU 5	202444A	6000001	100	0	5000001
                            SKU 5	202445A	6000001	100	0	5000001
                            SKU 5	202445B	6000001	100	0	5000001
                            SKU 5	202446A	6000001	100	0	5000001
                            SKU 5	202447A	6000001	100	0	5000001
                            SKU 5	202443A	6000002	100	0	5000002
                            SKU 5	202444A	6000002	100	0	5000002
                            SKU 5	202445A	6000002	100	0	5000002
                            SKU 5	202445B	6000002	100	0	5000002
                            SKU 5	202446A	6000002	100	0	5000002
                            SKU 5	202447A	6000002	100	0	5000002
                            SKU 4	202443A	6000001	100	0	5000001
                            SKU 4	202444A	6000001	100	0	5000001
                            SKU 4	202445A	6000001	100	0	5000001
                            SKU 4	202445B	6000001	100	0	5000001
                            SKU 4	202446A	6000001	100	0	5000001
                            SKU 4	202447A	6000001	100	0	5000001
                            SKU 4	202443A	6000002	100	0	5000002
                            SKU 4	202444A	6000002	100	0	5000002
                            SKU 4	202445A	6000002	100	0	5000002
                            SKU 4	202445B	6000002	100	0	5000002
                            SKU 4	202446A	6000002	100	0	5000002
                            SKU 4	202447A	6000002	100	0	5000002
                            SKU 3	202443A	6000001	1000	0	5000001
                            SKU 3	202444A	6000001	1000	0	5000001
                            SKU 3	202445A	6000001	1000	0	5000001
                            SKU 3	202445B	6000001	1000	0	5000001
                            SKU 3	202446A	6000001	1000	0	5000001
                            SKU 3	202447A	6000001	1000	0	5000001
                            SKU 3	202443A	6000002	1000	0	5000002
                            SKU 3	202444A	6000002	1000	0	5000002
                            SKU 3	202445A	6000002	1000	0	5000002
                            SKU 3	202445B	6000002	1000	0	5000002
                            SKU 3	202446A	6000002	1000	0	5000002
                            SKU 3	202447A	6000002	1000	0	5000002
                            SKU 3	202443A	6000003	1000	0	5000002
                            SKU 3	202444A	6000003	1000	0	5000002
                            SKU 3	202445A	6000003	1000	0	5000002
                            SKU 3	202445B	6000003	1000	0	5000002
                            SKU 3	202446A	6000003	1000	0	5000002
                            SKU 3	202447A	6000003	1000	0	5000002
                            SKU 3	202443A	6000009	1000	0	5000009
                            SKU 3	202444A	6000009	1000	0	5000009
                            SKU 3	202445A	6000009	1000	0	5000009
                            SKU 3	202445B	6000009	1000	0	5000009
                            SKU 3	202446A	6000009	1000	0	5000009
                            SKU 3	202447A	6000009	1000	0	5000009

                        - 개요 
                            - df_in_Sales_Domain_Dimension를 사용하여 Sales Std4(Lv5) Data 생성	
                        - Process
                            - 컬럼추가 : Sales Domain.[Sales Std4]  
                                - Step 2-1 (df_fn_EstPrice_USD) 의 Sales Domain.[Ship To] 와 df_in_Sales_Domain_Dimension의 Sales Domain.[Ship To] 를 비교하여, Sales Domain.[Sales Std4] Data 구성					
                                - n:1 관계					
                                        
                    Step 3-3) Item.[Item] * Time.[Partial Week] * Sales Domain.[Sales Std4] (lv5) 로 Groupby 적용						
                        Step 3-3-1) Estimated Amount 추가	
                            - 결과데이타					
                                Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	Estimated Price_USD	S/In FCST(GI)_AP1	Estimated Amount	Sales Domain.[Sales Std4]
                                SKU 6	202443A	6000001	 200 	 50 	 10,000 	5000001
                                SKU 6	202444A	6000001	 200 	 100 	 20,000 	5000001
                                SKU 6	202445A	6000001	 200 	 150 	 30,000 	5000001
                                SKU 6	202445B	6000001	 200 	 200 	 40,000 	5000001
                                SKU 6	202446A	6000001	 200 	 250 	 50,000 	5000001
                                SKU 6	202447A	6000001	 200 	 300 	 60,000 	5000001
                                SKU 6	202443A	6000002	 100 	 25 	 2,500 	5000002
                                SKU 6	202444A	6000002	 100 	 50 	 5,000 	5000002
                                SKU 6	202445A	6000002	 100 	 75 	 7,500 	5000002
                                SKU 6	202445B	6000002	 100 	 100 	 10,000 	5000002
                                SKU 6	202446A	6000002	 100 	 125 	 12,500 	5000002
                                SKU 6	202447A	6000002	 100 	 150 	 15,000 	5000002
                                SKU 6	202443A	6000003	 100 	 25 	 2,500 	5000002
                                SKU 6	202444A	6000003	 100 	 50 	 5,000 	5000002
                                SKU 6	202445A	6000003	 100 	 75 	 7,500 	5000002
                                SKU 6	202445B	6000003	 100 	 100 	 10,000 	5000002
                                SKU 6	202446A	6000003	 100 	 125 	 12,500 	5000002
                                SKU 6	202447A	6000003	 100 	 150 	 15,000 	5000002
                                SKU 5	202443A	6000001	 100 	0 	0 	5000001
                                SKU 5	202444A	6000001	 100 	0 	0 	5000001
                                SKU 5	202445A	6000001	 100 	0 	0 	5000001
                                SKU 5	202445B	6000001	 100 	0 	0 	5000001
                                SKU 5	202446A	6000001	 100 	0 	0 	5000001
                                SKU 5	202447A	6000001	 100 	0 	0 	5000001
                                SKU 5	202443A	6000002	 100 	0 	0 	5000002
                                SKU 5	202444A	6000002	 100 	0 	0 	5000002
                                SKU 5	202445A	6000002	 100 	0 	0 	5000002
                                SKU 5	202445B	6000002	 100 	0 	0 	5000002
                                SKU 5	202446A	6000002	 100 	0 	0 	5000002
                                SKU 5	202447A	6000002	 100 	0 	0 	5000002
                                SKU 4	202443A	6000001	 100 	0 	0 	5000001
                                SKU 4	202444A	6000001	 100 	0 	0 	5000001
                                SKU 4	202445A	6000001	 100 	0 	0 	5000001
                                SKU 4	202445B	6000001	 100 	0 	0 	5000001
                                SKU 4	202446A	6000001	 100 	0 	0 	5000001
                                SKU 4	202447A	6000001	 100 	0 	0 	5000001
                                SKU 4	202443A	6000002	 100 	0 	0 	5000002
                                SKU 4	202444A	6000002	 100 	0 	0 	5000002
                                SKU 4	202445A	6000002	 100 	0 	0 	5000002
                                SKU 4	202445B	6000002	 100 	0 	0 	5000002
                                SKU 4	202446A	6000002	 100 	0 	0 	5000002
                                SKU 4	202447A	6000002	 100 	0 	0 	5000002
                                SKU 3	202443A	6000001	 1,000 	0 	0 	5000001
                                SKU 3	202444A	6000001	 1,000 	0 	0 	5000001
                                SKU 3	202445A	6000001	 1,000 	0 	0 	5000001
                                SKU 3	202445B	6000001	 1,000 	0 	0 	5000001
                                SKU 3	202446A	6000001	 1,000 	0 	0 	5000001
                                SKU 3	202447A	6000001	 1,000 	0 	0 	5000001
                                SKU 3	202443A	6000002	 2,000 	0 	0 	5000002
                                SKU 3	202444A	6000002	 1,000 	0 	0 	5000002
                                SKU 3	202445A	6000002	 1,000 	0 	0 	5000002
                                SKU 3	202445B	6000002	 1,000 	0 	0 	5000002
                                SKU 3	202446A	6000002	 1,000 	0 	0 	5000002
                                SKU 3	202447A	6000002	 1,000 	0 	0 	5000002
                                SKU 3	202443A	6000003	 1,000 	0 	0 	5000002
                                SKU 3	202444A	6000003	 1,000 	0 	0 	5000002
                                SKU 3	202445A	6000003	 1,000 	0 	0 	5000002
                                SKU 3	202445B	6000003	 1,000 	0 	0 	5000002
                                SKU 3	202446A	6000003	 1,000 	0 	0 	5000002
                                SKU 3	202447A	6000003	 1,000 	0 	0 	5000002
                                SKU 3	202443A	6000009	 1,000 	0 	0 	5000009
                                SKU 3	202444A	6000009	 1,000 	0 	0 	5000009
                                SKU 3	202445A	6000009	 1,000 	0 	0 	5000009
                                SKU 3	202445B	6000009	 1,000 	0 	0 	5000009
                                SKU 3	202446A	6000009	 1,000 	0 	0 	5000009
                                SKU 3	202447A	6000009	 1,000 	0 	0 	5000009
                            - Process
                                - 컬럼추가 : Estimated Amount   
                                    - Estimated Amount = Estimated Price_USD * S/In FCST(GI)_AP1						
                                            
                    Step 3-3-2) Item.[Item] * Time.[Partial Week] * Sales Domain.[Sales Std4] 로 Groupby 적용 및 물량가중 계산	
                        - 결과데이타					
                            Item.[Item]	Time.[Partial Week]	Estimated Price_USD	S/In FCST(GI)_AP1	Estimated Amount	Sales Domain.[Sales Std4]	Estimated  Price_USD
                            SKU 6	202443A	 200 	 50 	 10,000 	5000001	 200 
                            SKU 6	202444A	 200 	 100 	 20,000 	5000001	 200 
                            SKU 6	202445A	 200 	 150 	 30,000 	5000001	 200 
                            SKU 6	202445B	 200 	 200 	 40,000 	5000001	 200 
                            SKU 6	202446A	 200 	 250 	 50,000 	5000001	 200 
                            SKU 6	202447A	 200 	 300 	 60,000 	5000001	 200 
                            SKU 6	202443A	 100 	 50 	 5,000 	5000002	 100 
                            SKU 6	202444A	 100 	 100 	 10,000 	5000002	 100 
                            SKU 6	202445A	 100 	 150 	 15,000 	5000002	 100 
                            SKU 6	202445B	 100 	 200 	 20,000 	5000002	 100 
                            SKU 6	202446A	 100 	 250 	 25,000 	5000002	 100 
                            SKU 6	202447A	 100 	 300 	 30,000 	5000002	 100 
                            SKU 5	202443A	 100 	0 	0 	5000001	 100 
                            SKU 5	202444A	 100 	0 	0 	5000001	 100 
                            SKU 5	202445A	 100 	0 	0 	5000001	 100 
                            SKU 5	202445B	 100 	0 	0 	5000001	 100 
                            SKU 5	202446A	 100 	0 	0 	5000001	 100 
                            SKU 5	202447A	 100 	0 	0 	5000001	 100 
                            SKU 5	202443A	 100 	0 	0 	5000002	 100 
                            SKU 5	202444A	 100 	0 	0 	5000002	 100 
                            SKU 5	202445A	 100 	0 	0 	5000002	 100 
                            SKU 5	202445B	 100 	0 	0 	5000002	 100 
                            SKU 5	202446A	 100 	0 	0 	5000002	 100 
                            SKU 5	202447A	 100 	0 	0 	5000002	 100 
                            SKU 4	202443A	 100 	0 	0 	5000001	 100 
                            SKU 4	202444A	 100 	0 	0 	5000001	 100 
                            SKU 4	202445A	 100 	0 	0 	5000001	 100 
                            SKU 4	202445B	 100 	0 	0 	5000001	 100 
                            SKU 4	202446A	 100 	0 	0 	5000001	 100 
                            SKU 4	202447A	 100 	0 	0 	5000001	 100 
                            SKU 4	202443A	 100 	0 	0 	5000002	 100 
                            SKU 4	202444A	 100 	0 	0 	5000002	 100 
                            SKU 4	202445A	 100 	0 	0 	5000002	 100 
                            SKU 4	202445B	 100 	0 	0 	5000002	 100 
                            SKU 4	202446A	 100 	0 	0 	5000002	 100 
                            SKU 4	202447A	 100 	0 	0 	5000002	 100 
                            SKU 3	202443A	 1,000 	0 	0 	5000001	 1,000 
                            SKU 3	202444A	 1,000 	0 	0 	5000001	 1,000 
                            SKU 3	202445A	 1,000 	0 	0 	5000001	 1,000 
                            SKU 3	202445B	 1,000 	0 	0 	5000001	 1,000 
                            SKU 3	202446A	 1,000 	0 	0 	5000001	 1,000 
                            SKU 3	202447A	 1,000 	0 	0 	5000001	 1,000 
                            SKU 3	202443A	 1,500 	0 	0 	5000002	 1,000 
                            SKU 3	202444A	 1,000 	0 	0 	5000002	 1,000 
                            SKU 3	202445A	 1,000 	0 	0 	5000002	 1,000 
                            SKU 3	202445B	 1,000 	0 	0 	5000002	 1,000 
                            SKU 3	202446A	 1,000 	0 	0 	5000002	 1,000 
                            SKU 3	202447A	 1,000 	0 	0 	5000002	 1,000 
                            SKU 3	202443A	 1,000 	0 	0 	5000009	 1,000 
                            SKU 3	202444A	 1,000 	0 	0 	5000009	 1,000 
                            SKU 3	202445A	 1,000 	0 	0 	5000009	 1,000 
                            SKU 3	202445B	 1,000 	0 	0 	5000009	 1,000 
                            SKU 3	202446A	 1,000 	0 	0 	5000009	 1,000 
                            SKU 3	202447A	 1,000 	0 	0 	5000009	 1,000 

                        - Process (변동)
                            - Groupby (Item.[Item] * Time.[Partial Week] * Sales Domain.[Sales Std4])
                                - S/In FCST(GI)_AP1, Estimated Amount 에 대해서 Sum 적용 		
                                - Estimated  Price_USD 에 대해서는 AVG 적용 (25.07.21 변동)  => COL_AVG_PRICE
                                    - FCST 값이 Aggr 이후 0인 경우를 생각하여 AVG 적용
                                    - 컬럼의 중복을 피해서 컬럼명을 AVG_PRICE 라고 한다.
                                        - COL_AVG_PRICE = 'AVG_PRICE'

                            - 신규 Column  Estimated  Price_USD  생성 및 계산  
                                - Estimated  Price_USD  =  Estimated Amount / S/In FCST(GI)_AP1		
                                    - 여기서 Estimated Amount 및 S/In FCST(GI)_AP1 은 Groupby Sum 된 값이다.		
                                - (25.07.21) 신규 Column 인 Estimated Price_USD 값 계산 시, FCST(S/In FCST(GI)_AP1) = 0 인 경우, 기존의 Estimated Price_USD 값 사용		
                                    - 여기서 '기존의 Estimated Price_USD' 이란 Avg(Estimated  Price_USD) 이다.

                    Step 3-3-3) Column 정리		
                        - 결과데이타		
                            Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	Estimated  Price_USD	
                            SKU 6	202443A	5000001	200	
                            SKU 6	202444A	5000001	200	
                            SKU 6	202445A	5000001	200	
                            SKU 6	202445B	5000001	200	
                            SKU 6	202446A	5000001	200	
                            SKU 6	202447A	5000001	200	
                            SKU 6	202443A	5000002	100	
                            SKU 6	202444A	5000002	100	
                            SKU 6	202445A	5000002	100	
                            SKU 6	202445B	5000002	100	
                            SKU 6	202446A	5000002	100	
                            SKU 6	202447A	5000002	100	
                        - Process
                            - rename : Sales Domain.[Sales Std4] -> Sales Domain.[Ship To]				
                                    
                Step 4) 4Lv Dummy Data 생성 (물량가중 처리, 없는 경우 하위값 그대로 사용) (Step 3와 방식은 동일하며, 기준 Column만 변화)					
                    Step 4-1) Step 3-3-3) 의 Data 와 Step 1-2) Lv5 Data를 Merge				
                    Step 4-2) Sales Domain 4Lv Data 구성				
                    Step 4-3) Item.[Item] * Time.[Partial Week] * Sales Domain.[Sales Std3] 로 Groupby 적용				
                    Step 4-3-1) Estimated Amount 추가				
                    Step 4-3-2) Item.[Item] * Time.[Partial Week] * Sales Domain.[Sales Std3] 로 Groupby 적용 및 물량가중 계산				
                    Step 4-3-3) Column 정리				
                                    
                Step 5) 3Lv Dummy Data 생성 (물량가중 처리, 없는 경우 하위값 그대로 사용) (Step 3와 방식은 동일하며, 기준 Column만 변화)					
                Step 6) 2Lv Dummy Data 생성 (물량가중 처리, 없는 경우 하위값 그대로 사용) (Step 3와 방식은 동일하며, 기준 Column만 변화)					
                                    
                                    
                Step 7) 최종 Output 구성	
                    - 결과데이타				
                        Version.[Version Name]	Item.[Item]	Time.[Partial Week]	Sales Domain.[Ship To]	Estimated  Price_USD
                        CWV_DP	SKU 6	202443A	5000001	200
                        CWV_DP	SKU 6	202444A	5000001	200
                        CWV_DP	SKU 6	202445A	5000001	200
                        CWV_DP	SKU 6	202445B	5000001	200
                        CWV_DP	SKU 6	202446A	5000001	200
                        CWV_DP	SKU 6	202447A	5000001	200
                    - Process
                        - Concat : Step 2-1) 과 Step 2-3) 과 Step 3-3-3) 과 Step 4), Step5), Step6)		
                            Step 2-1 	: 6Lv (Std5) 원본파일
                            Step 2-3 	: 7Lv (Std6)
                            Step 3-3-3 	: 5Lv (Std4)
                            Step 4		: 4Lv (Std3)
                            Step 5 		: 3Lv (Std2)
                            Step 6		: 2Lv (Std1)		
                        - Version.[Version Name] = 'CWV_DP' 추가				
                                    
    호출 (*)							
                        
        EXEC plugin instance [PYDPDummyPrice] 
        for measures { Measure.[Estimated  Price_USD] } 					
        using scope (
            [Version].[Version Name].[CWV_DP] 
            * [Sales Domain].[Ship To] 
            * [Item].[Item] 
            *  Time.[Partial Week].filter (#.Key >= &CurrentPartialWeek && #.Key <= &CurrentMonth.element(0).leadoffset(1).key) )					
        using arguments {
            (ExecutionMode, "MediumWeight") 					
        }					
        ;					
                        

    ==============  디버그 ======================

    - 0725 : 데이타가 맞지 않는다는 내용
        - 이찬주 부장님 Dummy Price 로직  Data가 맞지 않는 것 같습니다.

                Version.[Version Name]	Item.[Item]	Sales Domain.[Ship To]	Time.[Partial Week]	Estimated Price_USD	S/In FCST(GI)_AP2	AMT
                CWV_DP	HW-QS700F/ZA	401738	202521A	260	0	0
                CWV_DP	HW-QS700F/ZA	401735	202521A	260	2158	561080
                CWV_DP	HW-QS700F/ZA	401737	202521A	260	38	9880
                CWV_DP	HW-QS700F/ZA	404706	202521A	260	0	0
                CWV_DP	HW-QS700F/ZA	408351	202521A	401	8	3208
                CWV_DP	HW-QS700F/ZA	407881	202521A	260	0	0
                260.5117967	2204	574168
                CWV_DP	HW-QS700F/ZA	401738	202522A	260	1293	336180
                CWV_DP	HW-QS700F/ZA	401735	202522A	260	3	780
                CWV_DP	HW-QS700F/ZA	401737	202522A	260	0	0
                CWV_DP	HW-QS700F/ZA	404706	202522A	260	25	6500
                CWV_DP	HW-QS700F/ZA	408351	202522A	401	37	14837
                CWV_DP	HW-QS700F/ZA	407881	202522A	260	0	0
                263.8416789	1358	358297
                Output						
                Version.[Version Name]	Item.[Item]	Sales Domain.[Ship To]	Time.[Partial Week]	Estimated Price_USD		
                CWV_DP	HW-QS700F/ZA	300114	202521A	283		
                CWV_DP	HW-QS700F/ZA	300114	202522A	283	   

            - HW-QS700F/ZA 의 300114 하위에 대해서 확인했을 때, 300114 Lv에서 283이 나옵니다. 
                제 계산으로는 위와 같이 나올 것 같은데,  로직에서 제가 놓친부분이 있을까요 ?
        - 디버깅이 필요함.
            - 실행문 
                EXEC plugin instance [PYDPDummyPrice]
                for measures { Measure.[Estimated Price_USD] }
                using scope (
                [Version].[Version Name].[CWV_DP] 
                * [Sales Domain].[Ship To] 
                * [Item].[Item].[HW-QS700F/ZA]
                * [Time].[Partial Week].filter (#.Key >= &CurrentPartialWeek.element(0).Key && #.Key < &CurrentMonth.element(0).leadoffset(1).Key) )
                using arguments {
                (ExecutionMode, "MediumWeight")
                ,(IncludeNullrows, False)
                }
                ;

                (Input 1) df_in_Sin_GI_AP1
                    - query					
                        Select (				
                        [Version].[Version Name].[CWV_DP]		// 당주주차 ~ 당월주차
                            * [Item].[Item].[HW-QS700F/ZA]	        // USD가 6Lv로 들어오니, FCST 값도 6Lv 들어와서 계산
                            * [Time].[Partial Week].filter (#.Key >= &CurrentPartialWeek.element(0).Key && #.Key <= &CurrentMonth.element(0).leadoffset(1).Key)				
                            * [Sales Domain].[Sales Std5]				
                        ) on row, 				
                        ({Measure.[S/In FCST(GI)_AP1]}) on column;		

                (Input 2) df_In_EstimatedPrice_USD  exchange rate 실적
                    - query				
                        Select (			
                        [Version].[Version Name].[CWV_DP]			
                        * [Item].[Item].[HW-QS700F/ZA]			
                        * [Time].[Partial Week].filter (#.Key >= &CurrentPartialWeek.element(0).Key && #.Key <= &CurrentMonth.element(0).leadoffset(1).Key)			
                        * [Sales Domain].[Ship To].filter(#.Name startswith([5])) 
                        ) on row, 			
                        ({Measure.[Estimated Price_USD]}) 			
                        on column;	

                (Input 3) df_in_Sales_Domain_Dimension
                    - query	
                        Select (
                        [Sales Domain].[Sales Std1]
                        * [Sales Domain].[Sales Std2] 
                        * [Sales Domain].[Sales Std3] 
                        * [Sales Domain].[Sales Std4] 
                        * [Sales Domain].[Sales Std5] 
                        * [Sales Domain].[Sales Std6] 
                        * [Sales Domain].[Ship To] );        				
	
"""

from re import X
import os,sys,json,shutil,io,zipfile
import time
import datetime
import inspect
import traceback
import pandas as pd
from NSCMCommon import NSCMCommon as common
from NSCMCommon import VDCommon as vdCommon
# from typing_extensions import Literal
import glob
import numpy as np
# import rbql
# import duckdb
import gc, psutil

########################################################################################################################
# Local 개발 시에 필요한 공통 변수 선언
########################################################################################################################
# o9에 저장된 instanceName
is_local = common.gfn_get_isLocal()
str_instance = 'PYDPDummyPrice'
str_input_dir = f"Input/{str_instance}"
str_output_dir = f"Output/{str_instance}"

is_print = True
flag_csv = True
flag_exception = True
# Global variable for max_week
max_week = None
CurrentPartialWeek = None
max_week_normalized = None
current_week_normalized = None


########################################################################################################################
# 컬럼상수
########################################################################################################################
COL_VERSION = 'Version.[Version Name]'
COL_ITEM = 'Item.[Item]'
# Item_Type = 'Item.[Item Type]'  
COL_PT = 'Item.[ProductType]'  
COL_GBM = 'Item.[Item GBM]'
COL_PG = 'Item.[Product Group]'
COL_SHIP_TO     = 'Sales Domain.[Ship To]'
COL_STD1        = 'Sales Domain.[Sales Std1]'
COL_STD2        = 'Sales Domain.[Sales Std2]'
COL_STD3        = 'Sales Domain.[Sales Std3]' 
COL_STD4        = 'Sales Domain.[Sales Std4]'
COL_STD5        = 'Sales Domain.[Sales Std5]'
COL_STD6        = 'Sales Domain.[Sales Std6]'

COL_LOC         = 'Location.[Location]'
COL_TIME_PW     = 'Time.[Partial Week]'

COL_PRICE           = 'Estimated Price_USD'
COL_AVG_PRICE       = 'AVG_PRICE'
COL_AMT             = 'Estimated Amount'
COL_SIN_FCST_AP1    = 'S/In FCST(GI)_AP1'

# ───────────────────────────────────────────────────────────────
# CONSTANT STRING VARIABLES FOR DATAFRAME NAMES
# ───────────────────────────────────────────────────────────────
STR_DF_AP1             = 'df_in_Sin_GI_AP1'
STR_DF_PRICE           = 'df_in_EstimatedPrice_USD'
STR_DF_DIM             = 'df_in_Sales_Domain_Dimension'
STR_DF_FN_AP1          = 'df_fn_Sin_GI_AP1'
DOMAIN_DF_FN_LV        = 'df_fn_Sin_GI_AP1_lv'
STR_DF_FN_AP1_LV3      = 'df_fn_Sin_GI_AP1_lv3'
STR_DF_FN_AP1_LV4      = 'df_fn_Sin_GI_AP1_lv4'
STR_DF_FN_AP1_LV5      = 'df_fn_Sin_GI_AP1_lv5'
STR_DF_FN_AP1_LV6      = 'df_fn_Sin_GI_AP1_lv6'
STR_DF_FN_PRICE        = 'df_fn_EstimatedPrice_USD'
STR_DF_FN_MERGED_LV6   = 'df_fn_Merged_Lv6_Est'
STR_DF_FN_DUM_LV7      = 'df_fn_Dummy_Lv7'
STR_DF_FN_CONVERT_LV7  = 'df_fn_Converted_Lv7'
STR_DF_FN_DUM_LV5      = 'df_fn_Dummy_Lv5'
STR_DF_FN_DUM_LV4      = 'df_fn_Dummy_Lv4'
STR_DF_FN_DUM_LV3      = 'df_fn_Dummy_Lv3'
STR_DF_FN_DUM_LV2      = 'df_fn_Dummy_Lv2'
STR_DF_OUT_PRICE       = 'df_output_Dummy_EstimatedPrice_USD'

# ───────────────────────────────────────────────────────────────
# (추가) 컬럼 상수 0908
# ───────────────────────────────────────────────────────────────
COL_RATE_LOCAL   = 'Exchange Rate_Local'       # Input4 환율 (float)
COL_PRICE_LOCAL  = 'Estimated Price_Local'     # 최종 Local 단가 (float, 2자리 올림)

# ───────────────────────────────────────────────────────────────
# (추가) DataFrame 상수 0908
# ───────────────────────────────────────────────────────────────
STR_DF_RATE_LOCAL            = 'df_in_Exchange_Rate_Local'            # Input4
STR_DF_OUT_PRICE_LOCAL       = 'df_output_Dummy_EstimatedPrice_Local'  # Output(Local)

# ───────────────────────────────────────────────────────────────
# 추가된 컬럼 상수 (Step 09)
# ───────────────────────────────────────────────────────────────
COL_PRICE_LOCAL_MOD = 'Estimated Price Modify_Local'   # Input5 · 수정용 Local 단가# (참고) Step08에서 이미 추가한 상수


# ───────────────────────────────────────────────────────────────
# 추가된 DataFrame 상수 (Step 09)
# ───────────────────────────────────────────────────────────────
STR_DF_LOCAL_MOD            = 'df_in_EstimatedPrice_Modify_Local'   # Input5

########################################################################################################################
# log 설정 : PROGRAM file_name
########################################################################################################################

logger = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
# fn_set_local_logfile()
LOG_LEVEL = common.G_log_level

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
        logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=20)
        if is_local and not df_p_source.empty and flag_csv:
            # 로컬 Debugging 시 csv 파일 출력
            df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)
    else:
        # 최종 Output 테이블인 경우에는 무조건 로그 출력
        if is_output:
            logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=20)
            if is_local and not df_p_source.empty:
                # 로컬 Debugging 시 csv 파일 출력
                df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)



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

def initialize_max_week(is_local, args):
    global max_week, CurrentPartialWeek, max_week_normalized, current_week_normalized,chunk_size_step08 , chunk_size_step09, chunk_size_step10 , chunk_size_step11, chunk_size_step12, chunk_size_step13
    if is_local:
        # Read from MST_MODELEOS_TEST.csv
        df_eos = input_dataframes.get('df_in_Time_Partial_Week')
        max_week = df_eos['Time.[Partial Week]'].max()  # Assuming EOS_COM_DATE represents 최대주차
    else:
        # Get from command-line arguments
        max_week = args.get('max_week')

    # Initialize max_week_normalized. may be 202653A
    max_week_normalized = normalize_week(max_week)
    # max_week_normalized = int(max_week_normalized, 16) # convert to int32 from current str.

    # Initialize CurrentPartialWeek
    CurrentPartialWeek = common.gfn_get_partial_week(p_datetime=datetime.datetime.now())
    CurrentPartialWeek = '202447'
    if args.get('CurrentPartialWeek') is not None:
        CurrentPartialWeek = args.get('CurrentPartialWeek')
    # Initialize current_week_normalized
    current_week_normalized = normalize_week(CurrentPartialWeek)
    # current_week_normalized = int(current_week_normalized, 16)

    if args.get('chunk_size_step08') is not None:
        chunk_size_step08 = int(args.get('chunk_size_step08'))
    if args.get('chunk_size_step09') is not None:
        chunk_size_step09 = int(args.get('chunk_size_step09'))
    if args.get('chunk_size_step10') is not None:
        chunk_size_step10 = int(args.get('chunk_size_step10'))
    if args.get('chunk_size_step11') is not None:
        chunk_size_step11 = int(args.get('chunk_size_step11'))
    if args.get('chunk_size_step12') is not None:
        chunk_size_step12 = int(args.get('chunk_size_step12'))
    if args.get('chunk_size_step13') is not None:
        chunk_size_step13 = int(args.get('chunk_size_step13'))

def normalize_week(week_str):
    """Convert a week string with potential suffixes to an integer for comparison."""
    # Remove any non-digit characters (e.g., 'A' or 'B') and convert to integer
    return ''.join(filter(str.isdigit, week_str))

def is_within(current_week, start_week, end_week):
    """
    Check if the current week is within the range defined by start and end weeks.
    """
    return start_week <= current_week <= end_week

def fn_convert_type(df: pd.DataFrame, startWith: str, type):
    for column in df.columns:
        if column.startswith(startWith):
            df[column] = df[column].astype(type,errors='ignore')

# ───────────────────────────────────────────────────────────────
# 1)  Ship-To → Level 2~7 반환 함수  (벡터라이즈용 dict 기반)
# ───────────────────────────────────────────────────────────────
def fn_build_shipto_level_dict(df_dim: pd.DataFrame) -> dict:
    """
    Sales-Domain Dimension 전체를 스캔하여
    Ship-To 코드 → 레벨(2~7) 을 매핑한 dict 를 한 번에 생성한다.
    """
    STD_COL_LEVEL = [
        (COL_STD1, 2),
        (COL_STD2, 3),
        (COL_STD3, 4),
        (COL_STD4, 5),
        (COL_STD5, 6),
        (COL_STD6, 7),
    ]    
    ship_level = {}
    # 각 Std 컬럼에 등장하는 코드 → 해당 레벨로 dict 업데이트 (벡터라이즈용)
    for col, lv in STD_COL_LEVEL:
        ship_level.update( dict.fromkeys( df_dim[col].dropna().unique(), lv) )

    # leaf(Ship-To) 인덱스까지 포함 : “부모 Std 에 없는 Ship-To” 는 leaf-레벨로 덮어씀
    leaf_level = (
        df_dim[[ c for c, _ in STD_COL_LEVEL ]]
        .notna()
        .sum(axis=1)          # Std1~Std6 중 존재하는 개수(=leaf depth-1)
        .to_numpy() + 1       # leaf 은 depth+1
    )
    ship_level.update(
        dict(zip(df_dim[COL_SHIP_TO].to_numpy(), leaf_level))
    )
    return ship_level

def fn_build_shipto_level_dict_v2() -> dict:
    # ───────────────────────────────────────────────────────────────
    # 0)  전역 LUT   (한 번만 생성)
    # ───────────────────────────────────────────────────────────────
    global SHIPTO_INDEX  , LEVEL_ARRAY , SHIP_LEVEL , input_dataframes
    # ───────────────────────────────────────────────────────────────
    # 0) Ship-To → Level LUT  (중복 없는 dict 기반)
    # ───────────────────────────────────────────────────────────────
    STD_LV_COLS = [
        (COL_STD6, 7),
        (COL_STD5, 6),
        (COL_STD4, 5),
        (COL_STD3, 4),
        (COL_STD2, 3),
        (COL_STD1, 2),
    ]
    df_dim = input_dataframes[STR_DF_DIM].copy()

    # ❶ dict 로 먼저 생성해 중복 제거
    ship_level_dict: dict[str, int] = {}

    for col, lv in STD_LV_COLS:
        ship_level_dict.update( { code: lv for code in df_dim[col].dropna().unique() } )

    # # leaf(Ship-To) 는 항상 최종 깊이+1 이므로 덮어씀
    # leaf_lv = (df_dim[[c for c, _ in STD_LV_COLS]]
    #         .notna()
    #         .sum(axis=1)
    #         .to_numpy(dtype='uint8') + 1)
            
    # ship_level_dict.update(
    #     dict(zip(df_dim[Sales_ShipTo].to_numpy(), leaf_lv))
    # )

    # ❷ dict → Index & NumPy array  (모두 UNIQUE)
    SHIPTO_CODES  = np.fromiter(ship_level_dict.keys(),   dtype=object)
    LEVEL_ARRAY   = np.fromiter(ship_level_dict.values(), dtype='int32')
    SHIPTO_INDEX  = pd.Index(SHIPTO_CODES, dtype='object')   # ✅ 유니크 보장

    # for simple dict. solower than usin LEVEL_ARRAY and SHIPTO_INDEX
    SHIP_LEVEL  = ship_level_dict

def analyze_by_duckdb():
    # Retrieve your DataFrames
    import duckdb
    asn_df    = input_dataframes['df_in_Sales_Product_ASN']
    master_df = input_dataframes['df_in_Item_Master']
    dim_df   = input_dataframes['df_in_Sales_Domain_Dimension']      

    # Register each DataFrame as a DuckDB table
    duckdb.register('asn_table', asn_df)
    duckdb.register('master_table', master_df)
    duckdb.register('dim_table', dim_df)

    # Build a SQL query referencing them by table aliases a (asn_table) and b (master_table)
    my_query = f"""
    SELECT
        a['{COL_SHIP_TO}'] AS shipto,
        c['{COL_STD1}'] AS DomainLv2,
        a['{COL_ITEM}']          AS item,
        a['{COL_LOC}']  AS location,
        b['{COL_PT}']          AS item_type,
        b['{COL_GBM}']          AS item_gbm,
        b['{COL_PG}']      AS product_group
    FROM asn_table AS a
    JOIN master_table AS b
      ON a['{COL_ITEM}'] == b['{COL_ITEM}']
    JOIN dim_table AS c
        ON a['{COL_SHIP_TO}'] == c['{COL_SHIP_TO}']
    WHERE b['{COL_PT}']  = 'BAS'
    """

    # Execute the DuckDB query in-memory and fetch as a pandas DataFrame
    result_df = duckdb.query(my_query).to_df()

    return result_df


def analyze_by_duckdb_from_output():

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
    import duckdb

    v_base_dir  = "C:\workspace\Output\PYForecastMeasureLockColor_SHA_REF_20250410_14_08"
    v_output_dir  = f"{v_output_dir}/output"
    v_input_dir = f"{v_base_dir}/input"
    
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
        duckdb.register(file_name, df)

    output_pattern = f"{v_output_dir}/*.csv"
    output_csv_files = glob.glob(output_pattern)
    for file in output_csv_files:
        file_name = file.split("/")[-1].split("\\")[-1].split(".")[0]
        df = read_csv_with_fallback(file)
        duckdb.register(file_name, df)

    # # Retrieve your DataFrames]
    # file = f"{v_output_dir}/input/df_in_Sales_Product_ASN"
    # df = read_csv_with_fallback(file)
    # duckdb.register('df_in_Sales_Product_ASN', df)
    my_query = f"""
    SELECT
        a['{COL_SHIP_TO}'] AS shipto,
        c['{COL_STD1}'] AS DomainLv2,
        a['{COL_ITEM}']          AS item,
        a['{COL_LOC}']  AS location,
        b['{COL_PT}']          AS item_type,
        b['{COL_GBM}']          AS item_gbm,
        b['{COL_PG}']      AS product_group
    FROM asn_table AS a
    JOIN master_table AS b
      ON a['{COL_ITEM}'] == b['{COL_ITEM}']
    JOIN dim_table AS c
        ON a['{COL_SHIP_TO}'] == c['{COL_SHIP_TO}']
    WHERE b['{COL_PT}']  = 'BAS'
    """

    # Execute the DuckDB query in-memory and fetch as a pandas DataFrame
    result_df = duckdb.query(my_query).to_df()

    return result_df

################################################################################################################──────────
#  공통 타입 변환  (❌ `global` 사용 금지)
#  호출 측에서 `input_dataframes` 를 인자로 넘겨준다.
################################################################################################################──────────
def fn_prepare_input_types(dict_dfs: dict) -> None:
    """
    dict_dfs :  { <df_name> : pandas.DataFrame, ... }
    
    • object  → str → category                (숫자·문자 혼재 대비)
    • float/int → fillna(0) → int32           (값이 실수면 round 후 변환)    **주의** : dict 내부의 DataFrame 을 *제자리*에서 변환하므로 반환값은 없다.
    """
    if not dict_dfs:        # 빈 dict 방어
        return

    for df_name, df in dict_dfs.items():
        if df.empty:
            continue

        # 1) object 컬럼 : str → category
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df[col] = df[col].astype(str).astype("category")

        # 2) numeric 컬럼 : fillna → int32
        num_cols = df.select_dtypes(
            include=["float64", "float32", "int64", "int32", "int"]
        ).columns
        for col in num_cols:
            df[col].fillna(0, inplace=True)
            try:
                if not df[col].dtype in ['int32','int8']:
                    df[col] = df[col].astype("float32")

            except ValueError:
                df[col] = df[col].round().astype("int32")


@_decoration_
def fn_process_in_df_mst():

    if is_local: 
        # 로컬인 경우 Output 폴더를 정리한다.
        for file in os.scandir(str_output_dir):
            os.remove(file.path)

        # 로컬인 경우 파일을 읽어 입력 변수를 정의한다.
        file_pattern = f"{os.getcwd()}/{str_input_dir}/*.csv" 
        csv_files = glob.glob(file_pattern)


        file_to_df_mapping = {
            "df_in_Sales_Domain_Dimension.csv"          :      STR_DF_DIM    ,
            "df_In_EstimatedPrice_USD.csv"              :      STR_DF_PRICE    ,
            "df_in_Sin_GI_AP1.csv"                      :      STR_DF_AP1      ,
            "df_in_Exchange_Rate_Local.csv"             :      STR_DF_RATE_LOCAL , 
            "df_in_EstimatedPrice_Modify_Local.csv"     :      STR_DF_LOCAL_MOD
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
            # df['SourceFile'] = file_name
            # df.set_index('SourceFile',inplace=True)
            mapped = False
            for keyword, frame_name in file_to_df_mapping.items():
                if file_name.startswith(keyword.split('.')[0]):
                    input_dataframes[frame_name] = df
                    mapped = True
                    break
    else:
        # o9 에서 
        input_dataframes[STR_DF_DIM]        = df_in_Sales_Domain_Dimension
        input_dataframes[STR_DF_PRICE]      = df_in_EstimatedPrice_USD
        input_dataframes[STR_DF_AP1]        = df_in_Sin_GI_AP1
        input_dataframes[STR_DF_RATE_LOCAL] = df_in_Exchange_Rate_Local
        input_dataframes[STR_DF_LOCAL_MOD]  = df_in_EstimatedPrice_Modify_Local

    # type convert : str ==> category, int ==> int32
    fn_convert_type(input_dataframes[STR_DF_DIM], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_PRICE], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_AP1], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_RATE_LOCAL], 'Sales Domain', str)
    fn_convert_type(input_dataframes[STR_DF_LOCAL_MOD], 'Sales Domain', str)

    input_dataframes[STR_DF_AP1][COL_SIN_FCST_AP1].fillna(0, inplace=True)
    fn_convert_type(input_dataframes[STR_DF_AP1], COL_SIN_FCST_AP1, 'int32')

    input_dataframes[STR_DF_PRICE][COL_PRICE].fillna(0, inplace=True)
    fn_convert_type(input_dataframes[STR_DF_PRICE], COL_PRICE, 'int32')

    
    fn_prepare_input_types(input_dataframes)


################################################################################################################
############## Start Of Step Function  ###############
################################################################################################################

# ───────────────────────────────────────────────────────────────
# 공통 LUT 생성 (Ship-To → Level)
# ───────────────────────────────────────────────────────────────
def fn_build_shipto_level_lut(df_dim: pd.DataFrame) -> tuple[pd.Index, np.ndarray, dict]:
    """
    Sales Domain Dimension 으로부터  
        • SHIPTO_INDEX   : Ship-To code Index  
        • LEVEL_ARRAY    : 동일 위치의 레벨(int32)  
        • SHIP_LEVEL_DICT: {shipto:level}
    를 반환한다. (중복 0, vectorized lookup 용)
    """
    STD_LV_COLS = [
        (COL_STD6, 7), (COL_STD5, 6), (COL_STD4, 5),
        (COL_STD3, 4), (COL_STD2, 3), (COL_STD1, 2),
    ]
    lut = {}
    for col, lv in STD_LV_COLS:
        lut.update({code: lv for code in df_dim[col].dropna().unique()})    
        shipto_index = pd.Index(lut.keys(), dtype=object)

    level_array  = np.fromiter(lut.values(), dtype='int32')
    return shipto_index, level_array, lut


# ───────────────────────────────────────────────────────────────
# Step 01-1 : Sin AP1 전처리
# ───────────────────────────────────────────────────────────────
@_decoration_
def fn_step01_1_preprocess_sin_ap1(df_sin: pd.DataFrame) -> pd.DataFrame:
    out = (df_sin
           .drop(columns=[COL_VERSION], errors='ignore')
           .rename(columns={COL_STD5: COL_SHIP_TO})
           .copy())

    # 메모리 절감용 dtype 보강
    out[COL_SHIP_TO] = out[COL_SHIP_TO].astype('category')
    # logger.Note(f'[01-1] Mem: {psutil.Process().memory_info().rss>>20} MB',
    #             p_log_level=LOG_LEVEL.debug())
    return out

# ───────────────────────────────────────────────────────────────
# Step 01-2 : 레벨별 DataFrame 분리
# ───────────────────────────────────────────────────────────────
@_decoration_
def _fn_step01_2_split_levels(df_fcst: pd.DataFrame,
                             shipto_index: pd.Index,
                             level_array: np.ndarray) -> dict[int, pd.DataFrame]:
    """
    Ship-To → Level vectorized 매핑 후  
    Lv 3~6 각각 Boolean-mask slice → Dict 반환 {lv: dataframe}
    """

    pos  = shipto_index.get_indexer(df_fcst[COL_SHIP_TO].to_numpy())
    lvl  = np.where(pos >= 0, level_array[pos], np.nan).astype('float32')

    df_fcst = df_fcst.assign(_LEVEL=lvl)

    slices: dict[int, pd.DataFrame] = {}
    for lv in (3, 4, 5, 6):
        part = df_fcst.loc[df_fcst._LEVEL == lv].drop(columns=['_LEVEL'])
        slices[lv] = part
    # --- 원본 df_fcst 는 이후 사용 안 하므로 즉시 메모리 해제 ---
    del df_fcst; gc.collect()

    # logger.Note(f'[01-2] Mem: {psutil.Process().memory_info().rss>>20} MB',
    #             p_log_level=LOG_LEVEL.debug())
    return slices

# ───────────────────────────────────────────────────────────────
# Step 01-2 : 레벨별 DataFrame 분리
# Step 01-2 : 6Lv → 5/4/3Lv Forecast 생성 (2025-07-25 변경)
# ───────────────────────────────────────────────────────────────
@_decoration_
def fn_step01_2_split_levels(
    df_fcst: pd.DataFrame,      # 6Lv  AP1 Forecast (Step 01-1 결과)
    df_dim: pd.DataFrame        # Sales-Domain Dimension  (Ship-To ↔ Std4/3/2 매핑용)
) -> dict[int, pd.DataFrame]:
    """
    • 입력은 **항상 6Lv(Std5)** Forecast 하나뿐임  
    • 6Lv → 5Lv/4Lv/3Lv Forecast 를 **벡터라이즈 & groupby** 로 생성  
      반환 : {3:df_lv3, 4:df_lv4, 5:df_lv5, 6:df_lv6}
    """
    # ── 공통 설정 ───────────────────────────────────────────
    grp_base = [COL_ITEM, COL_TIME_PW]        # 공통 groupby 키
    dtype_cat = 'category'    
    # ───────────────────────────────────────────────────────
    # Lv 6   : 그대로 사용
    # ───────────────────────────────────────────────────────
    df_lv6 = df_fcst.copy()
    df_lv6[COL_SHIP_TO] = df_lv6[COL_SHIP_TO].astype(dtype_cat)

    # ───────────────────────────────────────────────────────
    # Ship-To → Std4 / Std3 / Std2 매핑 dict
    # (한 번만 만들어 재활용)
    # ───────────────────────────────────────────────────────
    dim_map = df_dim.set_index(COL_SHIP_TO)[[COL_STD4, COL_STD3, COL_STD2]]

    map_std4 = dim_map[COL_STD4].to_dict()
    map_std3 = dim_map[COL_STD3].to_dict()
    map_std2 = dim_map[COL_STD2].to_dict()

    # ───────────────────────────────────────────────────────
    # helper : level-aggregation
    # ───────────────────────────────────────────────────────
    def _make_level(df_src: pd.DataFrame,
                    map_dict: dict,
                    tgt_col: str) -> pd.DataFrame:
        """
        • Ship-To → tgt_col 매핑 후  
        • groupby(Item,Week,tgt_col) ⇒ Σ(FCST)  
        • tgt_col → Ship-To 로 치환
        """
        df_tmp = df_src.copy()

        df_tmp[tgt_col] = df_tmp[COL_SHIP_TO].map(map_dict).astype(dtype_cat)

        agg = (df_tmp
               .groupby(grp_base + [tgt_col], sort=False, observed=True)
               [COL_SIN_FCST_AP1]
               .sum()
               .reset_index())

        agg.rename(columns={tgt_col: COL_SHIP_TO}, inplace=True)
        agg[COL_SHIP_TO] = agg[COL_SHIP_TO].astype(dtype_cat)

        # 메모리 해제
        del df_tmp
        gc.collect()
        return agg

    # ── 5Lv  (Std4) ────────────────────────────────────────
    df_lv5 = _make_level(df_lv6, map_std4, COL_STD4)

    # ── 4Lv  (Std3) ────────────────────────────────────────
    df_lv4 = _make_level(df_lv6, map_std3, COL_STD3)

    # ── 3Lv  (Std2) ────────────────────────────────────────
    df_lv3 = _make_level(df_lv6, map_std2, COL_STD2)

    # ── 반환 dict ──────────────────────────────────────────
    out = {3: df_lv3, 4: df_lv4, 5: df_lv5, 6: df_lv6}

    # 중간 객체 해제
    del dim_map, map_std4, map_std3, map_std2
    gc.collect()

    return out

# ───────────────────────────────────────────────────────────────
# Step 02-1 : Estimated Price 전처리
# ───────────────────────────────────────────────────────────────
@_decoration_
def fn_step02_1_preprocess_est_price(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    Version 컬럼 제거 → df_fn_EstimatedPrice_USD
    """
    out = (df_price
           .drop(columns=[COL_VERSION], errors='ignore')
           .copy())
    # out[COL_PRICE] = out[COL_PRICE].astype('int32')
    # logger.Note(f'[02-1] Mem: {psutil.Process().memory_info().rss>>20} MB',
    #             p_log_level=LOG_LEVEL.debug())
    return out

# ───────────────────────────────────────────────────────────────
# Step 02-2 : Lv7 Dummy RAW 생성
# ───────────────────────────────────────────────────────────────
@_decoration_
def fn_step02_2_make_dummy_lv7_raw(df_price: pd.DataFrame,
                                   df_dim:   pd.DataFrame,
                                   ship_level: dict
                                   ) -> pd.DataFrame:
    """
    • Std5 = Ship-To 기준으로 Std6(7Lv) 매핑  
    • Ship-To level==7 만 남김
    """
    dim7 = (df_dim[[COL_STD5, COL_STD6]]
            .dropna(subset=[COL_STD6])
            .loc[lambda d: d[COL_STD6].map(ship_level) == 7])

    raw7 = (df_price
            .merge(dim7, left_on=COL_SHIP_TO, right_on=COL_STD5,
                   how='left', sort=False)
            .dropna(subset=[COL_STD6])
            .drop(columns=[COL_STD5]))

    # 더 이상 쓰지 않는 dim7 즉시 해제
    del dim7; gc.collect()

    # logger.Note(f'[02-2] Mem: {psutil.Process().memory_info().rss>>20} MB',
    #             p_log_level=LOG_LEVEL.debug())
    return raw7


# # ───────────────────────────────────────────────────────────────
# # Step 02-3 : Std6 → Ship-To 치환
# # ───────────────────────────────────────────────────────────────
# @_decoration_
# def fn_step02_3_convert_lv7(df_raw7: pd.DataFrame) -> pd.DataFrame:
#     """
#     Std6 값을 Ship-To 로 승격 & 불필요 컬럼 제거
#     """
#     df_final7 = (df_raw7
#                  .assign(**{COL_SHIP_TO: df_raw7[COL_STD6]})
#                  .drop(columns=[COL_STD6]))
#     return df_final7

# ───────────────────────────────────────────────
# Step 02-3 : Std6 → Ship-To 승격 (7Lv 확정)
# ───────────────────────────────────────────────
@_decoration_
def fn_step02_3_convert_lv7(df_raw7: pd.DataFrame) -> pd.DataFrame:
    """
    • Sales Domain.[Sales Std6] 값을 Ship-To 로 올리고
    • 중간 컬럼 제거 → 7Lv 확정 테이블 반환      반환 컬럼
        └ Item.[Item]  
        └ Time.[Partial Week]  
        └ Sales Domain.[Ship To]  (→ Std6 코드)  
        └ Estimated Price_USD
    """
    # 1) Ship-To 치환 & 컬럼 제거 ― 완전 벡터라이즈
    df_out = (
        df_raw7
        .assign(**{COL_SHIP_TO: df_raw7[COL_STD6].astype('category')})
        .drop(columns=[COL_STD6], errors='ignore')
        .copy()
    )

    # 2) dtype 안전 확보 (int32 · category)
    df_out[COL_PRICE]      = df_out[COL_PRICE].astype('float32', copy=False)

    # 3) 메모리 로그
    # logger.Note(
    #     f"[02-3] Mem: {psutil.Process().memory_info().rss >> 20} MB",
    #     p_log_level=LOG_LEVEL.debug()
    # )

    return df_out

# -----------------------------------------------------------------
# ⚠️  공통 헬퍼 : 레벨별 Dummy Price 생성
#   · 25-07-25 규격 수정
#   · 결과 단가 : 소수 둘째 자리에서 **올림**(ceil) 처리
# -----------------------------------------------------------------
def _make_dummy_price(
        df_price:  pd.DataFrame,      # 상위레벨 Price
        df_fcst:   pd.DataFrame,      # 상위레벨 FCST
        df_dim:    pd.DataFrame,      # Dimension
        src_col:   str,               # 매핑 SOURCE  Ship-To
        tgt_col:   str                # 매핑 TARGET Std(레벨)
    ) -> pd.DataFrame:
    """
    • df_price(LEFT) ⟕ df_fcst → FCST NULL→0  
    • Ship-To ⇒ tgt_col 로 매핑  
    • 그룹별 ΣFCST·ΣAMT·AVG_PRICE → FCST>0 ? AMT/FCST : AVG
    • tgt_col → Ship-To  치환 후 반환
    """
    # 1) Join  &  FCST NULL → 0
    df_m = (
        df_price
        .merge(
            df_fcst[[COL_ITEM, COL_SHIP_TO, COL_TIME_PW, COL_SIN_FCST_AP1]],
            on=[COL_ITEM, COL_SHIP_TO, COL_TIME_PW],
            how='left',
            sort=False
        )
        .fillna({COL_SIN_FCST_AP1: 0})
    )

    # 2) Ship-To → tgt_col(Level) 매핑
    df_m[tgt_col] = df_m[COL_SHIP_TO].map(
        df_dim.set_index(COL_SHIP_TO)[tgt_col]
    ).astype('category')

    # 3) Amount
    #     * Price 가 이미 실수(f32/f64) 라고 가정
    df_m[COL_AMT] = (
        df_m[COL_PRICE].astype('float32')
        * df_m[COL_SIN_FCST_AP1].astype('int32')
    )

    # 4) 그룹 집계  (Σ·AVG)
    grp_cols = [COL_ITEM, tgt_col, COL_TIME_PW]
    df_lv = (df_m
        .groupby(grp_cols, sort=False, observed=True)
        .agg({
            COL_SIN_FCST_AP1: 'sum',
            COL_AMT:          'sum',
            COL_PRICE:        'mean'   # AVG_PRICE
        })
        .rename(columns={COL_PRICE: COL_AVG_PRICE})
        .reset_index()
    )

    # 5) 최종 단가
    fcst  = df_lv[COL_SIN_FCST_AP1].to_numpy(dtype='float32')
    amt   = df_lv[COL_AMT].to_numpy(dtype='float32')
    avg   = df_lv[COL_AVG_PRICE].to_numpy(dtype='float32')

    # df_lv[COL_PRICE] = np.where(
    #     fcst > 0,
    #     (amt / fcst).round().astype('int32'),
    #     avg.astype('int32')
    # )

    # 5-1) 결과 배열을 먼저 준비
    price = np.empty_like(fcst, dtype='float32')
    # 5-2) True 위치(FCST>0)만 나눗셈 수행
    mask = fcst > 0
    price[mask]  = amt[mask] / fcst[mask]

    # 5-3) False 위치(FCST=0)는 Avg 가격 사용
    price[~mask] = avg[~mask]

    #   **소수 둘째 자리에서 올림 처리** 0908: 사용안함
    # price = np.ceil(price * 100) / 100        # e.g. 1.522 → 1.53

    df_lv[COL_PRICE] = price.astype('float32')

    # 6) 열 정리
    df_lv = (
        df_lv
        .rename(columns={tgt_col: COL_SHIP_TO})
        [[COL_ITEM, COL_SHIP_TO, COL_TIME_PW, COL_PRICE]]
    )

    # 메모리 해제
    del df_m, fcst, amt, avg
    gc.collect()
    return df_lv



# ───────────────────────────────────────────────────────────────
# Step 03 : 5Lv Dummy Price
# ───────────────────────────────────────────────────────────────
@_decoration_
def fn_step03_make_lv5_dummy(df_lv6_price: pd.DataFrame,
                             df_lv6_fcst:  pd.DataFrame,
                             df_dim:       pd.DataFrame) -> pd.DataFrame:
    """
    6Lv Price · FCST → 5Lv(Std4) Dummy Price 산출
    (로직은 _make_dummy_price 헬퍼와 100% 동일, 매핑 대상만 Std4)
    """
    return _make_dummy_price(
        df_lv6_price,     # 상위 Price
        df_lv6_fcst,      # 상위 FCST
        df_dim,           # Dimension
        COL_SHIP_TO,      # src_col (사용되지 않지만 시그니처 유지)
        COL_STD4          # tgt_col : Sales Std4 ➜ 5Lv
    )                             
# ───────────────────────────────────────────────────────────────
# Step 04 · 05 · 06 : 4Lv / 3Lv / 2Lv Dummy Price
# ───────────────────────────────────────────────────────────────
@_decoration_
def fn_step04_make_lv4_dummy(df_lv5_price, df_lv5_fcst, df_dim):
    return _make_dummy_price(df_lv5_price, df_lv5_fcst,
                             df_dim, COL_SHIP_TO, COL_STD3)

@_decoration_
def fn_step05_make_lv3_dummy(df_lv4_price, df_lv4_fcst, df_dim):
    return _make_dummy_price(df_lv4_price, df_lv4_fcst,
                             df_dim, COL_SHIP_TO, COL_STD2)

@_decoration_
def fn_step06_make_lv2_dummy(df_lv3_price, df_lv3_fcst, df_dim):
    return _make_dummy_price(df_lv3_price, df_lv3_fcst,
                             df_dim, COL_SHIP_TO, COL_STD1)


################################################################################################################
# Step 07 : 최종 Output 통합
################################################################################################################
@_decoration_
def fn_step07_finalize_output(
        df_lv6: pd.DataFrame, df_lv7: pd.DataFrame,
        df_lv5: pd.DataFrame, df_lv4: pd.DataFrame,
        df_lv3: pd.DataFrame, df_lv2: pd.DataFrame,
        version_name: str
    ) -> pd.DataFrame:
    """
    모든 레벨(2~7) Estimated Price 데이터 concat → Version 컬럼 부여
    반환 : df_output_Dummy_EstimatedPrice_USD
    """
    df_out = (
        pd.concat(
            [df_lv6, df_lv7, df_lv5, df_lv4, df_lv3, df_lv2],
            ignore_index=True
        )
        .assign(**{COL_VERSION: version_name})
        [[COL_VERSION, COL_ITEM, COL_TIME_PW, COL_SHIP_TO, COL_PRICE]]
        .sort_values([COL_ITEM, COL_SHIP_TO, COL_TIME_PW], kind='mergesort')
        .reset_index(drop=True)
        # .astype({COL_PRICE: 'int32'})
    )

    # 단가 전체를 소수 둘째 자리에서 올림
    df_out[COL_PRICE] = np.ceil(df_out[COL_PRICE].astype('float32') * 100) / 100

    # dtype 고정 (float32)  → 메모리 절감
    df_out[COL_PRICE] = df_out[COL_PRICE].astype('float32')

    return df_out

# ───────────────────────────────────────────────────────────────
# Step 08 : USD → Local 변환 (수정본)
#   • 이미 Main에서 fn_build_shipto_level_lut()로 ship_level_dict 를 생성했으므로
#     여기서는 재계산하지 않고 인자로 받은 ship_level_dict 를 그대로 사용.
#   • 환율은 Std3 레벨에서 제공
#       - 하위 Lv(Std4~7/Ship-To) : Ship-To → Std3 매핑 후 해당 환율 적용
#       - 상위 Lv(Std1, Std2)     : 자식 Std3 환율들의 평균(AVG) 적용
#   • 소수점 처리 : "셋째 자리에서 올림" → np.ceil(x*100)/100
#   • 불필요한 재형변환(특히 category 캐스팅) 제거
# ───────────────────────────────────────────────────────────────
@_decoration_
def fn_step08_convert_usd_to_local(
        df_usd:  pd.DataFrame,     # Step 07 결과(모든 레벨 USD 단가)
        df_rate: pd.DataFrame,     # Input4 : df_in_Exchange_Rate_Local (Std3, Week, Rate)
        df_dim:  pd.DataFrame,     # Dimension (Ship-To ↔ Std 계층)
        ship_level_dict: dict      # Main에서 미리 만든 {code: level}
    ) -> pd.DataFrame:
    """
    반환 : df_output_Dummy_EstimatedPrice_Local
            [Version, Item, Partial Week, Ship To, Estimated Price_Local]
    """
    # 0) 환율 원본 정리 (Version 제거만)
    rate_std3_df = (
        df_rate
        .drop(columns=[COL_VERSION], errors='ignore')
        [[COL_STD3, COL_TIME_PW, COL_RATE_LOCAL]]
        .copy()
    )    # 1) 레벨 마스크
    level_codes = df_usd[COL_SHIP_TO].map(ship_level_dict).astype('int8')
    mask_lv2 = (level_codes == 2)
    mask_lv3 = (level_codes == 3)
    mask_low = (level_codes >= 4)   # Std4~7/Ship-To

    base_cols = [COL_VERSION, COL_ITEM, COL_TIME_PW, COL_SHIP_TO, COL_PRICE]

    # 2) 하위 레벨(Std4~7) : Ship-To → Std3 매핑 후 환율 join
    df_low = df_usd.loc[mask_low, base_cols].copy()
    if not df_low.empty:
        ship_to_to_std3 = df_dim.set_index(COL_SHIP_TO)[COL_STD3].to_dict()
        df_low['TMP_STD3'] = df_low[COL_SHIP_TO].map(ship_to_to_std3)
        df_low = (
            df_low.merge(
                rate_std3_df,
                left_on=['TMP_STD3', COL_TIME_PW],
                right_on=[COL_STD3,  COL_TIME_PW],
                how='left',
                sort=False
            )
            .drop(columns=[COL_STD3, 'TMP_STD3'])
        )
        v = (
            df_low[COL_PRICE].astype('float32', copy=False).to_numpy()
            * df_low[COL_RATE_LOCAL].astype('float32', copy=False).to_numpy()
        )
        df_low[COL_PRICE_LOCAL] = (np.ceil(v * 100) / 100).astype('float32')
        df_low = df_low[[COL_VERSION, COL_ITEM, COL_TIME_PW, COL_SHIP_TO, COL_PRICE_LOCAL]]

    # 3) Std2(레벨3) : 자식 Std3 환율 평균(Std2, Week)
    df_lv3 = df_usd.loc[mask_lv3, base_cols].copy()
    if not df_lv3.empty:
        std2_std3 = (
            df_dim[[COL_STD2, COL_STD3]]
            .dropna()
            .drop_duplicates()
        )
        rate_std2 = (
            std2_std3
            .merge(rate_std3_df, on=COL_STD3, how='left', sort=False)
            .groupby([COL_STD2, COL_TIME_PW], sort=False, observed=True)[COL_RATE_LOCAL]
            .mean()
            .reset_index()
        )
        df_lv3 = (
            df_lv3.merge(
                rate_std2,
                left_on=[COL_SHIP_TO, COL_TIME_PW],   # Ship-To가 곧 Std2 코드
                right_on=[COL_STD2,   COL_TIME_PW],
                how='left',
                sort=False
            )
            .drop(columns=[COL_STD2])
        )
        v = (
            df_lv3[COL_PRICE].astype('float32', copy=False).to_numpy()
            * df_lv3[COL_RATE_LOCAL].astype('float32', copy=False).to_numpy()
        )
        df_lv3[COL_PRICE_LOCAL] = (np.ceil(v * 100) / 100).astype('float32')
        df_lv3 = df_lv3[[COL_VERSION, COL_ITEM, COL_TIME_PW, COL_SHIP_TO, COL_PRICE_LOCAL]]

        del std2_std3, rate_std2; gc.collect()

    # 4) Std1(레벨2) : 자식 Std3 환율 평균(Std1, Week)
    df_lv2 = df_usd.loc[mask_lv2, base_cols].copy()
    if not df_lv2.empty:
        std1_std3 = (
            df_dim[[COL_STD1, COL_STD3]]
            .dropna()
            .drop_duplicates()
        )
        rate_std1 = (
            std1_std3
            .merge(rate_std3_df, on=COL_STD3, how='left', sort=False)
            .groupby([COL_STD1, COL_TIME_PW], sort=False, observed=True)[COL_RATE_LOCAL]
            .mean()
            .reset_index()
        )
        df_lv2 = (
            df_lv2.merge(
                rate_std1,
                left_on=[COL_SHIP_TO, COL_TIME_PW],   # Ship-To가 곧 Std1 코드
                right_on=[COL_STD1,   COL_TIME_PW],
                how='left',
                sort=False
            )
            .drop(columns=[COL_STD1])
        )
        v = (
            df_lv2[COL_PRICE].astype('float32', copy=False).to_numpy()
            * df_lv2[COL_RATE_LOCAL].astype('float32', copy=False).to_numpy()
        )
        df_lv2[COL_PRICE_LOCAL] = (np.ceil(v * 100) / 100).astype('float32')
        df_lv2 = df_lv2[[COL_VERSION, COL_ITEM, COL_TIME_PW, COL_SHIP_TO, COL_PRICE_LOCAL]]

        del std1_std3, rate_std1; gc.collect()

    # 5) 합치기 + 정렬(안정 정렬)
    df_local = pd.concat([df_low, df_lv3, df_lv2], ignore_index=True)
    if not df_local.empty:
        df_local = (
            df_local
            .sort_values([COL_ITEM, COL_SHIP_TO, COL_TIME_PW], kind='mergesort')
            .reset_index(drop=True)
        )
        # 결과 타입 고정(필요 최소한)
        df_local[COL_PRICE_LOCAL] = df_local[COL_PRICE_LOCAL].astype('float32', copy=False)

    # 임시 객체 해제
    del rate_std3_df, level_codes, mask_lv2, mask_lv3, mask_low, df_low, df_lv3, df_lv2
    gc.collect()
    return df_local

# ───────────────────────────────────────────────────────────────
# Step 09 : Estimated Price Modify_Local 값 적용 (250916 추가)
#   · df_output_Dummy_EstimatedPrice_Local 을 기반으로
#   · 동일 키(Item, Week, Ship-To)에서 df_in_EstimatedPrice_Modify_Local 값이 있으면 덮어씀
#   · Version 컬럼은 기존(df_local)의 값을 유지
# ───────────────────────────────────────────────────────────────
@_decoration_
def fn_step09_apply_modify_local(
    df_local:  pd.DataFrame,   # Step08 결과: df_output_Dummy_EstimatedPrice_Local
    df_modify: pd.DataFrame    # Input5: df_in_EstimatedPrice_Modify_Local
) -> pd.DataFrame:
    """
    • 키: (Item.[Item], Time.[Partial Week], Sales Domain.[Ship To])
    • modify 값(Estimated Price Modify_Local)이 존재하는 경우, Local 값(Estimated Price_Local)을 해당 값으로 업데이트
    • 반환 컬럼:
        [Version].[Version Name], Item.[Item], Time.[Partial Week], Sales Domain.[Ship To], Estimated Price_Local
    """
    keys = [COL_ITEM, COL_TIME_PW, COL_SHIP_TO]    # 1) Modify 테이블 정리: Version 제거, 결측 Modify 제거, 중복 키 제거
    mod = (
        df_modify
        .drop(columns=[COL_VERSION], errors='ignore')
        .dropna(subset=[COL_PRICE_LOCAL_MOD])
        [keys + [COL_PRICE_LOCAL_MOD]]
        .drop_duplicates(subset=keys, keep='last')
        .copy()
    )

    # 2) Left-merge로 수정값을 붙임 (원본 순서 보존)
    merged = df_local.merge(mod, on=keys, how='left', sort=False)

    # 3) 벡터화 overwrite: modify 값이 있는 위치만 교체
    base = merged[COL_PRICE_LOCAL].to_numpy(dtype='float32', copy=False)
    ov   = merged[COL_PRICE_LOCAL_MOD].to_numpy(copy=False)  # float64일 수 있음
    mask = ~pd.isna(ov)

    out_price = base.copy()
    out_price[mask] = ov[mask].astype('float32', copy=False)

    merged[COL_PRICE_LOCAL] = out_price

    # 4) 최종 컬럼만 반환 (정렬은 유지, dtype 유지)
    out = merged[[COL_VERSION, COL_ITEM, COL_TIME_PW, COL_SHIP_TO, COL_PRICE_LOCAL]].copy()

    # 메모리 해제
    del mod, merged, base, ov, out_price, mask
    gc.collect()
    return out

################################################################################################################
############## Start Of Main ###############
################################################################################################################
if __name__ == '__main__':
    args = parse_args()
    logger.debug(f'[START] {str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    logger.Start()

    input_dataframes = {}
    output_dataframes = {}
    try:
        ################################################################################################################
        # 전처리 : 모듈 내에서 사용될 데이터에 대한 정합성 체크 및 데이터 선 가공
        ################################################################################################################

        # ----------------------------------------------------
        # parse_args 대체
        # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
        # ----------------------------------------------------
        if is_local:
            Version = 'CWV_DP'

            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------
            # input_folder_name = 'PYDPDummyPrice'
            # output_folder_name = 'PYDPDummyPrice'
            # # 0725
            # input_folder_name = 'PYDPDummyPrice/input_20250725'
            # output_folder_name = 'PYDPDummyPrice_0725'
            # 0908
            # # -----
            input_folder_name = 'PYDPDummyPrice/input_0916'
            output_folder_name = 'PYDPDummyPrice_0916'
            # ----
            # input_folder_name = 'PYDPDummyPrice/input_o9_1117'
            # output_folder_name = 'PYDPDummyPrice_o9_1117'
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
        
        # vdLog 초기화
        log_path = os.path.dirname(__file__) if is_local else ""
        vdCommon.gfn_pyLog_start(Version, str_instance, logger, is_local, log_path)
        # --------------------------------------------------------------------------
        # df_input 체크 시작
        # --------------------------------------------------------------------------
        logger.Note(p_note='df_input 체크 시작', p_log_level=LOG_LEVEL.debug())
        fn_process_in_df_mst()
        for in_df in input_dataframes:
            # 로그출력
            fn_log_dataframe(input_dataframes[in_df], in_df)
        # --------------------------------------------------------------------------
        # df_input 체크 종료
        # --------------------------------------------------------------------------
        df_in_price = input_dataframes.get(STR_DF_PRICE,      pd.DataFrame())
        df_in_price_empty = (
            df_in_price.empty
        )
        if df_in_price_empty:
            raise Exception('df_In_EstimatedPrice_USD(Local Currency 별  exchange rate 실적) is Empty')

        # --------------------------------------------------------------------------
        # 입력 변수 확인
        # --------------------------------------------------------------------------
        logger.Note(p_note=f'Parameter Check', p_log_level=LOG_LEVEL.debug())
        logger.Note(p_note=f'Version            : {Version}', p_log_level=LOG_LEVEL.debug())

        # ── LUT 생성 ────────────────────────────────────────────────
        shipto_idx, level_arr, ship_level_dict = fn_build_shipto_level_lut(
            input_dataframes[STR_DF_DIM])
        # ─────────────────────────────────────────────────────────────────
        # ── Step 01-1 ───────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────
        dict_log = {'p_step_no': 110, 'p_step_desc': 'Step 01-1 : Sin AP1 전처리'}
        df_fn_Sin_GI_AP1 = fn_step01_1_preprocess_sin_ap1(
            input_dataframes[STR_DF_AP1], **dict_log)
        output_dataframes[STR_DF_FN_AP1] = df_fn_Sin_GI_AP1
        fn_log_dataframe(df_fn_Sin_GI_AP1, f'fn_step01_1_{STR_DF_FN_AP1}')
        # ─────────────────────────────────────────────────────────────────
        # ── Step 01-2 ───────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────
        dict_log = {'p_step_no': 120, 'p_step_desc': 'Step 01-2 : 레벨별 분리'}
        lv_dfs = fn_step01_2_split_levels(
            df_fn_Sin_GI_AP1, 
            input_dataframes[STR_DF_DIM], # shipto_idx, 
            # level_arr, 
            **dict_log
        )

        for lv, df_lv in lv_dfs.items():
            name = f'{DOMAIN_DF_FN_LV}{lv}'
            output_dataframes[name] = df_lv
            fn_log_dataframe(df_lv, f'fn_step01_2_{name}')

        # ── Step 02-1 ───────────────────────────────────────────────
        dict_log = {'p_step_no': 210, 'p_step_desc': 'Step 02-1 : Estimated Price 전처리'}
        df_fn_EstPrice_USD = fn_step02_1_preprocess_est_price(
            input_dataframes[STR_DF_PRICE], **dict_log)
        output_dataframes[STR_DF_FN_PRICE] = df_fn_EstPrice_USD
        fn_log_dataframe(df_fn_EstPrice_USD, f'fn_step02_1_{STR_DF_FN_PRICE}')
        # ─────────────────────────────────────────────────────────────────
        # ── Step 02-2 ───────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────
        dict_log = {'p_step_no': 220, 'p_step_desc': 'Step 02-2 : Lv7 Dummy RAW'}
        df_fn_Dummy_Lv7 = fn_step02_2_make_dummy_lv7_raw(
            df_fn_EstPrice_USD,
            input_dataframes[STR_DF_DIM],
            ship_level_dict, **dict_log)
        output_dataframes[STR_DF_FN_DUM_LV7] = df_fn_Dummy_Lv7
        fn_log_dataframe(df_fn_Dummy_Lv7, f'fn_step02_2_{STR_DF_FN_DUM_LV7}')
        # ─────────────────────────────────────────────────────────────────
        # ── Step 02-3 ───────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────
        dict_log = {'p_step_no': 230, 'p_step_desc': 'Step 02-3 : Lv7 Ship-To 변환'}
        df_fn_Converted_Lv7 = fn_step02_3_convert_lv7(df_fn_Dummy_Lv7, **dict_log)
        output_dataframes[STR_DF_FN_CONVERT_LV7] = df_fn_Converted_Lv7
        fn_log_dataframe(df_fn_Converted_Lv7, f'fn_step02_3_{STR_DF_FN_CONVERT_LV7}')
        
        # … Step 01 & Step 02 전처리 결과 로드 (이미 완료되었다고 가정) … #
        df_lv6_price  = output_dataframes[STR_DF_FN_PRICE]         # Step 02-1 결과 (6Lv)
        df_lv7_price  = output_dataframes[STR_DF_FN_CONVERT_LV7]        # Step 02-3 결과
        df_lv6_fcst   = output_dataframes[STR_DF_FN_AP1_LV6]       # Step 01-2 결과
        df_lv5_fcst   = output_dataframes[STR_DF_FN_AP1_LV5]
        df_lv4_fcst   = output_dataframes[STR_DF_FN_AP1_LV4]
        df_lv3_fcst   = output_dataframes[STR_DF_FN_AP1_LV3]
        df_lv2_fcst   = output_dataframes[STR_DF_FN_AP1_LV3]       # 2Lv 은 3Lv fcst 와 공유

        df_dim        = input_dataframes[STR_DF_DIM]

        # ─────────────────────────────────────────────────────────────────
        # ── Step 03 ──────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────
        dict_log = {'p_step_no': 300, 'p_step_desc': 'Step 03 : 5Lv Dummy Price 산출'}
        df_lv5_price = fn_step03_make_lv5_dummy(df_lv6_price, df_lv6_fcst, df_dim, **dict_log)
        output_dataframes[STR_DF_FN_DUM_LV5] = df_lv5_price
        fn_log_dataframe(df_lv5_price, f'fn_step03_{STR_DF_FN_DUM_LV5}')

        # ─────────────────────────────────────────────────────────────────
        # ── Step 04 ──────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────
        dict_log = {'p_step_no': 400, 'p_step_desc': 'Step 04 : 4Lv Dummy Price 산출'}
        df_lv4_price = fn_step04_make_lv4_dummy(df_lv5_price, df_lv5_fcst, df_dim, **dict_log)
        output_dataframes[STR_DF_FN_DUM_LV4] = df_lv4_price
        fn_log_dataframe(df_lv4_price, f'fn_step04_{STR_DF_FN_DUM_LV4}')

        # ─────────────────────────────────────────────────────────────────
        # ── Step 05 ──────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────
        dict_log = {'p_step_no': 500, 'p_step_desc': 'Step 05 : 3Lv Dummy Price 산출'}
        df_lv3_price = fn_step05_make_lv3_dummy(df_lv4_price, df_lv4_fcst, df_dim, **dict_log)
        output_dataframes[STR_DF_FN_DUM_LV3] = df_lv3_price
        fn_log_dataframe(df_lv3_price, f'fn_step05_{STR_DF_FN_DUM_LV3}')

        # ─────────────────────────────────────────────────────────────────
        # ── Step 06 ──────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────
        dict_log = {'p_step_no': 600, 'p_step_desc': 'Step 06 : 2Lv Dummy Price 산출'}
        df_lv2_price = fn_step06_make_lv2_dummy(df_lv3_price, df_lv3_fcst, df_dim, **dict_log)
        output_dataframes[STR_DF_FN_DUM_LV2] = df_lv2_price
        fn_log_dataframe(df_lv2_price, f'fn_step06_{STR_DF_FN_DUM_LV2}')

        # ─────────────────────────────────────────────────────────────────
        # ── Step 07 ──────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────
        dict_log = {'p_step_no': 700, 'p_step_desc': 'Step 07 : 최종 Output 통합'}
        df_output_Dummy_EstimatedPrice_USD = fn_step07_finalize_output(
            df_lv6_price, df_lv7_price, df_lv5_price, df_lv4_price,
            df_lv3_price, df_lv2_price, Version, **dict_log)

        # 로그 및 최종 출력
        fn_log_dataframe(df_output_Dummy_EstimatedPrice_USD, f'fn_step07_{STR_DF_OUT_PRICE}')
        # output_dataframes[STR_DF_OUT_PRICE] = df_output_Dummy_EstimatedPrice_USD

        # ################################################################################################################
        # # Formatter:  Add Version Name 
        # ################################################################################################################
        # dict_log = {
        #     'p_step_no': 900,
        #     'p_step_desc': '최종 Output 정리 - out_Sellout'
        # }
        # fn_output_formatter(Version,**dict_log)

        # ─────────────────────────────────────────────────────────────────
        # Step 08 : USD → Local 변환 (Main 호출부)
        #   ※ ship_level_dict 는 이미 Main 앞부분에서
        #     shipto_idx, level_arr, ship_level_dict = fn_build_shipto_level_lut(input_dataframes[STR_DF_DIM])
        #     으로 생성되어 있음.
        # ─────────────────────────────────────────────────────────────────
        dict_log = {
            'p_step_no': 800, 
            'p_step_desc': 'Step 08 : USD → Local 변환'
        }
        df_output_Dummy_EstimatedPrice_Local = fn_step08_convert_usd_to_local(
            df_output_Dummy_EstimatedPrice_USD,   # Step07 결과(USD)
            input_dataframes[STR_DF_RATE_LOCAL],        # Input4 : 환율(Std3, Week)
            input_dataframes[STR_DF_DIM],          # Dimension
            ship_level_dict,                # 이미 계산된 레벨 LUT(dict)
            **dict_log
        )

        # 로그 및 보관
        fn_log_dataframe(df_output_Dummy_EstimatedPrice_Local, f'fn_step08_{STR_DF_OUT_PRICE_LOCAL}')

        # ─────────────────────────────────────────────────────────────────
        # Step 09 : Estimated Price Modify_Local 적용 (Main 호출부)
        #   ※ 본 단계는 LUT가 필요 없습니다. (Step08 결과에 수동 수정값을 덮어쓰기)
        # ─────────────────────────────────────────────────────────────────
        dict_log = {
            'p_step_no': 900,
            'p_step_desc': 'Step 09 : Estimated Price Modify_Local 적용'
        }
        df_output_Dummy_EstimatedPrice_Local = fn_step09_apply_modify_local(
            df_output_Dummy_EstimatedPrice_Local,        # Step08 결과(Local)
            input_dataframes[STR_DF_LOCAL_MOD],          # Input5 : Modify_Local
            **dict_log
        )

        # 로그 및 보관
        fn_log_dataframe(df_output_Dummy_EstimatedPrice_Local, f'fn_step09_{STR_DF_OUT_PRICE_LOCAL}')

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
        # fn_set_header()

        if is_local:
            log_file_name = common.G_PROGRAM_NAME.replace('py', 'log')
            log_file_name = f'log/{log_file_name}'

            shutil.copyfile(log_file_name, os.path.join(str_output_dir, os.path.basename(log_file_name)))

            # prografile copy
            program_path = f"{os.getcwd()}/NSCM_DP_UI_Develop/{str_instance}.py"
            shutil.copyfile(program_path, os.path.join(str_output_dir, os.path.basename(program_path)))

            # # log
            # input_path = f'{str_output_dir}/input'
            # os.makedirs(input_path,exist_ok=True)
            # for input_file in input_dataframes:
            #     input_dataframes[input_file].to_csv(input_path + "/"+input_file+".csv", encoding="UTF8", index=False)

            # # log
            # output_path = f'{str_output_dir}/output'
            # os.makedirs(output_path,exist_ok=True)
            # for output_file in output_dataframes:
            #     output_dataframes[output_file].to_csv(output_path + "/"+output_file+".csv", encoding="UTF8", index=False)

        logger.Finish()
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::') # 25.05.12 need warning Log by Logger Issue