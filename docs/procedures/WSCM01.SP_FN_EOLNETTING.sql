CREATE OR REPLACE PROCEDURE WSCM01.SP_FN_EOLNETTING_NC(IN_TYPE VARCHAR2 DEFAULT 'NCP') IS
/******************************************************************************
   NAME:       WSCM01.SP_FN_EOLNETTING_NC
   PURPOSE:    재고를 감안한 EOP 물량 보정 (EXP_SOPROMISESRCNCP ) 
   REVISIONS:
   Ver     Date        Author   Description
   ------- ----------  -------  ------------------------------------
   1.0     2009.08.07  S.J Choi  신규 버전
   1.1     2016.06.19  EUNJU    POSTPONEMENT 가용량 통합 적용
 ******************************************************************************/
 
    V_Pgmname               Varchar2(30) := 'SP_FN_EOLNETTING';
    V_Loadingtime           Varchar2(2);
    
    V_Retsqlerrm            Varchar2(2000);

    V_Planid                Mst_Plan.Planid %Type;
    V_Planweek              Mst_Plan.Planweek%Type;
    V_Effstartdate          Mst_Plan.Effstartdate %Type;
    V_Effenddate            Mst_Plan.Effenddate %Type;
    
    V_LOGMESSAGE        LONG;
    V_ERRMESSAGE        VARCHAR2(2000);
    V_LOGSEQ            NUMBER(10);
    V_ERRORCNT          INTEGER:= 0;
    V_STARTDATE         DATE;
     
    S_REMAINQTY         NUMBER := 0;
    L_DEMANDPRIORITY    NUMBER := 0;
    STEP_R_QTY          NUMBER := 0;
    R_QTY               NUMBER := 0;
    FARE_SHARE          NUMBER := 0;
    
    V_AVAILQTY             NUMBER;
    V_REMAINQTY            NUMBER;
    
    --2018.12.03 추가
    V_TYPE                 VARCHAR2(10);
     
    TYPE myrecord IS RECORD (
        ROWNUMBER             NUMBER,
        RK                    NUMBER,
        CHRK                  NUMBER,
        ROWIDs                ROWID,
        CNT                   NUMBER,
        SUBTOTAL              NUMBER,
        RATIO                 NUMBER,
        ENTERPRISE            EXP_SOPROMISESRCNCP.ENTERPRISE%TYPE,          
        SOPROMISEID           EXP_SOPROMISESRCNCP.SOPROMISEID%TYPE,       
        PLANID                EXP_SOPROMISESRCNCP.PLANID%TYPE,            
        SALESORDERID          EXP_SOPROMISESRCNCP.SALESORDERID%TYPE,      
        SOLINENUM             EXP_SOPROMISESRCNCP.SOLINENUM%TYPE,        
        ITEM                  EXP_SOPROMISESRCNCP.ITEM%TYPE,              
        QTYPROMISED           EXP_SOPROMISESRCNCP.QTYPROMISED%TYPE,       
        PROMISEDDELDATE       EXP_SOPROMISESRCNCP.PROMISEDDELDATE%TYPE,   
        SITEID                EXP_SOPROMISESRCNCP.SITEID%TYPE,            
        SHIPTOID              EXP_SOPROMISESRCNCP.SHIPTOID%TYPE,          
        SALESID               EXP_SOPROMISESRCNCP.SALESID%TYPE,           
        SALESLEVEL            EXP_SOPROMISESRCNCP.SALESLEVEL%TYPE,        
        DEMANDTYPE            EXP_SOPROMISESRCNCP.DEMANDTYPE%TYPE,        
        WEEK                  EXP_SOPROMISESRCNCP.WEEK%TYPE,              
        CHANNELRANK           EXP_SOPROMISESRCNCP.CHANNELRANK%TYPE,       
        CUSTOMERRANK          EXP_SOPROMISESRCNCP.CUSTOMERRANK%TYPE,      
        PRODUCTRANK           EXP_SOPROMISESRCNCP.PRODUCTRANK%TYPE,       
        DEMANDPRIORITY        EXP_SOPROMISESRCNCP.DEMANDPRIORITY%TYPE,    
        TIEBREAK              EXP_SOPROMISESRCNCP.TIEBREAK%TYPE,          
        GBM                   EXP_SOPROMISESRCNCP.GBM%TYPE,               
        GLOBALPRIORITY        EXP_SOPROMISESRCNCP.GLOBALPRIORITY%TYPE,    
        LOCALPRIORITY         EXP_SOPROMISESRCNCP.LOCALPRIORITY%TYPE,     
        BUSINESSTYPE          EXP_SOPROMISESRCNCP.BUSINESSTYPE%TYPE,      
        ROUTING_PRIORITY      EXP_SOPROMISESRCNCP.ROUTING_PRIORITY%TYPE,  
        NO_SPLIT              EXP_SOPROMISESRCNCP.NO_SPLIT%TYPE,          
        MAP_SATISFY_SS        EXP_SOPROMISESRCNCP.MAP_SATISFY_SS%TYPE,    
        PREALLOC_ATTRIBUTE    EXP_SOPROMISESRCNCP.PREALLOC_ATTRIBUTE%TYPE,
        BUILDAHEADTIME        EXP_SOPROMISESRCNCP.BUILDAHEADTIME%TYPE,    
        TIMEUOM               EXP_SOPROMISESRCNCP.TIMEUOM%TYPE,             
        AP2ID                 EXP_SOPROMISESRCNCP.AP2ID%TYPE,             
        GC                    EXP_SOPROMISESRCNCP.GC%TYPE,               
        MEASURERANK           EXP_SOPROMISESRCNCP.MEASURERANK%TYPE,       
        PREFERENCERANK        EXP_SOPROMISESRCNCP.PREFERENCERANK%TYPE,    
        INITDTTM              EXP_SOPROMISESRCNCP.INITDTTM%TYPE,          
        INITBY                EXP_SOPROMISESRCNCP.INITBY%TYPE,            
        UPDTTM                EXP_SOPROMISESRCNCP.UPDTTM%TYPE,            
        UPBY                  EXP_SOPROMISESRCNCP.UPBY%TYPE,                
        REASONCODE            EXP_SOPROMISESRCNCP.REASONCODE%TYPE    
    );
         
    TYPE mytable  IS TABLE OF myrecord INDEX BY PLS_INTEGER;
    mydata mytable;
    mydata_T mytable;
     
 
-----------------------------------------------------------------------------------------------------------------
     --! WEEK 별 가용량 계산 CURSOR !--
     CURSOR CUR_AVAIL(C_PLANID   VARCHAR)IS
        SELECT  ROWID  ROWIDS, RANK() OVER(PARTITION BY ITEM, SITEID  ORDER BY WEEK ) RK
                ,PLANID, SITEID, ITEM, WEEK, INVQTY , REMAINQTY, AVAILQTY, DMDQTY
        FROM    MST_EOLNETTING
        WHERE   PLAN    = 'EOL'
        AND     PLANID  = C_PLANID
        ORDER BY SITEID, ITEM, WEEK ;
    
    R CUR_AVAIL%ROWTYPE; 
                            
------------------------------------------------------------------------------------------------------------------    
BEGIN
        
    SELECT SEQ_LOG.NEXTVAL INTO V_LOGSEQ FROM DUAL;
    SELECT SYSDATE INTO V_STARTDATE FROM DUAL;
            
    --! get current running planid !--
    Sp_FN_Get_Planid(IN_TYPE,v_planid,v_planweek,v_effstartdate,v_effenddate);
    
    --2018.12.03 수정(V PLAN일 경우 당주기준으로 effstartdate 변경)
--    SELECT TYPE
--    INTO   V_TYPE
--    FROM   MST_PLAN
--    WHERE  ISRUNNING = 'Y'
--    AND    TYPE != 'NCP';
--          
--    IF V_TYPE = 'VPLAN' THEN
--        v_effstartdate := v_effstartdate+7;
--        v_effenddate := v_effenddate+7;
--    END IF;

    --! 백업용 BUF 테이블 TRUNCATE !--
    EXECUTE IMMEDIATE 'ALTER TABLE BUF_SOPROMISESRCNCP_POST TRUNCATE PARTITION PEOL';
           
    EXECUTE IMMEDIATE 'alter session enable parallel DML';
    
    
    --! 작업전에 BUF 테이블에 데이터 백업 !--
    INSERT /*+  append parallel(T  5)  */   INTO BUF_SOPROMISESRCNCP_POST T
            (PLAN, ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED, PROMISEDDELDATE
           , SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK
           , GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS
           , PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK
           , INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE)        
    SELECT /*+ FULL(S) parallel(S  5)  */ 'EOL',ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED, PROMISEDDELDATE, 
           SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, 
           DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, 
           NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, 
           INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE
    FROM   EXP_SOPROMISESRCNCP S --EXP_SOPROMISESRCNCP S
    WHERE  PLANID = V_PLANID;
            
    COMMIT;
    
    EXECUTE IMMEDIATE 'alter session disable parallel DML';
 
    --! STEP 1.EOL NETTING 대상 및 가용량 발췌 !--
    BEGIN
         
        --! EOL NETTING 테이블 데이터 DELETE !--
        DELETE FROM MST_EOLNETTING 
        WHERE  PLANID = V_PLANID
        AND    PLAN = 'EOL';
                
        --! EOL NETTING 테이블에 대상 DEMAND 및 가용량 발췌하여 INSERT !--
        INSERT INTO MST_EOLNETTING
        ( PLAN, PLANID, SITEID, ITEM, WEEK, EOPWEEK, INVQTY , REMAINQTY,  AVAILQTY,  DMDQTY )
        SELECT 'EOL' PLAN, V_PLANID PLANID, A.SITEID, A.ITEM,  A.WEEK DMDWEEK, TO_CHAR(B.EOP,'IYYYIW') EOPWEEK, --TO_CHAR( MAX(B.EOP),'IYYIW') EOPWEEK, 
               NVL(SUM(INVQTY),0) INVQTY, 0 REMAINQTY, NVL(SUM(INVQTY),0) AVAILQTY, SUM(DMDQTY) DMDQTY
        FROM  (SELECT /*+  parallel(A  3) */ 
                      SITEID, ITEM, TO_CHAR(PROMISEDDELDATE, 'IYYYIW') WEEK,  0 INVQTY, SUM(QTYPROMISED) DMDQTY 
               FROM   EXP_SOPROMISESRCNCP A --EXP_SOPROMISESRCNCP A
               WHERE  PLANID = V_PLANID
               AND    QTYPROMISED > 0 
               AND    NOT EXISTS (SELECT /*+ unnest  parallel(V_MTA_SELLERMAP  3)  */ 'X' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
               AND    NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND SALESID = A.SALESID) --E-STORE 20200807 추가
               GROUP BY PLANID, SITEID, ITEM,TO_CHAR(PROMISEDDELDATE, 'IYYYIW')
               UNION ALL
                -- 재고 DATA 발췌 : TOSITEID = '-'로 들어오는 경우 처리를 위한 작업
               SELECT DECODE(SITEID,'-',MOD_SITEID,SITEID) SITEID,
                      ITEM,  V_PLANWEEK WEEK, -- 기존 EOL NETTING도 주차 감안안됨 (MF 같은거땜에:1)
                      SUM(AVAILQTY) INV_QTY , 0 
               FROM (       
                       -- [2011.11.17: BEGIN] 구주포장센터(MTA_SALESBOMMAP) 관련 변경 작업
                       SELECT NVL(B.HEADSITEID,A.SITEID) SITEID,
                              MAX(NVL(B.HEADSITEID,A.SITEID)) OVER (PARTITION BY NVL(B.HEADSKU,A.ITEM)) MOD_SITEID ,
                              NVL(B.HEADSKU,A.ITEM) ITEM,
                              AVAILQTY     
                       FROM (
                             /* 1. 재고(PY, CY, 판매법인재고) */
                             SELECT /*+  leading(A C B)  swap_join_inputs(B) */   
                                  A.PLANID, DECODE(B.TYPE,'MF',A.TOSITEID,'DC',A.SITEID) SITEID, A.ITEM, 
--                                  AVAILQTY+BOHADDQTY+W0BOHADDQTY AVAILQTY
                                 CASE WHEN A.SITEID NOT IN ('L101','L999')
                                       THEN AVAILQTY+BOHADDQTY+W0BOHADDQTY
                                       WHEN A.SITEID IN ('L101','L999') 
                                       THEN CDCAVAILQTY+CDCBOHADDQTY + W0CDCBOHADDQTY
                                  END AVAILQTY
                             FROM MST_INVENTORY A, MST_SITE B, 
                                ( SELECT /*+ leading(A B) full(B) parallel(B 3)  pq_distribute(B  BROADCAST  NONE)  */
                                        A.FROMSITEID, A.TOSITEID, B.ITEM, MIN(B.TRANSITTIME) LT 
                                  FROM MST_BOD A, MST_BODDETAIL B
                                  WHERE A.BODNAME = B.BODNAME
                                  AND   NVL(B.EFFENDDATE,v_effenddate) >= v_effstartdate            
                                  GROUP BY A.FROMSITEID, A.TOSITEID, B.ITEM   
                                ) C
                             WHERE A.PLANID   = V_PLANID
                             AND   A.SITEID   = B.SITEID
                             AND   A.SITEID   = C.FROMSITEID(+)
                             AND   A.TOSITEID = C.TOSITEID(+)
                             AND   A.ITEM     = C.ITEM(+)
                             AND    NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = DECODE(B.TYPE,'MF',A.TOSITEID,'DC',A.SITEID))
                             UNION ALL          
                             /* 2. 향해재고 */      
                             SELECT PLANID, TOSITEID, ITEM, INTRANSITQTY
                             FROM   MST_INTRANSIT A
                             WHERE  PLANID = V_PLANID
                             AND    NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.TOSITEID)
                            ) A, 
                            (
                              SELECT HEADSITEID, HEADSKU, SITEID, MAINSKU
                              FROM MTA_SALESBOMMAP
                              WHERE PRIORITY = 1 AND ROWNUM >= 1
                            ) B
                       WHERE A.ITEM   = B.MAINSKU(+)
                       AND   A.PLANID = V_PLANID 
                       -- [2011.11.17: END] BY S.J.CHOI            
                    )    A
               WHERE  DECODE(SITEID,'-',MOD_SITEID,SITEID)   NOT IN ('-')    
               AND    EXISTS (SELECT 'X'  FROM EXP_SOPROMISESRCNCP--EXP_SOPROMISESRCNCP 
                              WHERE PLANID = V_PLANID AND  QTYPROMISED > 0 AND  ITEM = A.ITEM AND  SITEID = DECODE(A.SITEID,'-',A.MOD_SITEID,A.SITEID)
                              AND  (ITEM, SALESID) NOT IN (SELECT CUSTOMITEM, SALESID FROM MTA_CUSTOMMODELMAP) --E-STORE 20200807 추가
                              )
               GROUP BY DECODE(SITEID,'-',MOD_SITEID,SITEID), ITEM               
               )  A,  
              (
                -- EOP 일자 <= PLAN EFFSTARTDATE 인 모델이 Demand 물량 조정 대상
                SELECT A.ITEM, MAX(EOP) AS EOP
                FROM (
                       SELECT /*+ LEADING(B A) 
                               SWAP_JOIN_INPUTS(B) full(B) 
                               full(A) parallel(A 3) PQ_DISTRIBUTE(A  BROADCAST NONE)  */
                              NVL(DECODE(A.ITEM, B.MAINSKU, B.HEADSKU), A.ITEM) ITEM, A.SITEID AS FROMSITEID,
                              CASE WHEN A.STATUS = 'COM' THEN EOP_COM_DATE
                                   WHEN A.STATUS = 'INI' THEN NVL(EOP_CHG_DATE, EOP_INIT_DATE)
                              END AS EOP
                       FROM MST_MODELEOP  A, (SELECT * FROM MTA_SALESBOMMAP WHERE PRIORITY = 1 AND ROWNUM >=1) B
                       WHERE A.ITEM = B.MAINSKU(+)
                       -- [2011.11.17: END] BY S.J.CHOI
                    ) A
                GROUP BY ITEM
                HAVING MAX(EOP) <= TRUNC(V_EFFSTARTDATE)                                                                             
              ) B
         WHERE A.ITEM    = B.ITEM
         GROUP BY A.SITEID, A.ITEM, A.WEEK, TO_CHAR(B.EOP,'IYYYIW') ;

     DBMS_OUTPUT.PUT_LINE('1. AVAIL QTY Data Load:  '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));
                
    EXCEPTION 
     WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error: check '||SQLCODE||'_'||SQLERRM);
        ROLLBACK;
    END; 
         
         
     --! STEP 2. 남은 invqty 뒷구간으로 넘기기 !--
     OPEN CUR_AVAIL(V_PLANID);
     LOOP
        --! 1. Cursor Fetch !--
        FETCH CUR_AVAIL INTO R;
        EXIT WHEN CUR_AVAIL%NOTFOUND;
        
        
        
        --! S_REMAINQTY : 이전WEEK에서 남겨온 수량 !--
        IF  R.RK = 1 THEN
            --! ITEM, SITE가 첫 시작이면 이전WEEK에서 남겨온 수량 0 !--
            S_REMAINQTY := 0;
            
        ELSE
             --! AVAILQTY = 현재 ROW의 가용량 + 이전WEEK에서 남겨온 수량 !--
            UPDATE MST_EOLNETTING
            SET    REMAINQTY  = S_REMAINQTY
                  , AVAILQTY  = INVQTY + S_REMAINQTY
            WHERE  PLAN       = 'EOL'
            AND    PLANID     = V_PLANID
            AND    ITEM       = R.ITEM
            AND    SITEID     = R.SITEID
            AND    WEEK       = R.WEEK;
        
            COMMIT;
            
        END IF;
        
        
        R.INVQTY    := R.INVQTY  + S_REMAINQTY;
        
        
        -- SRC LOOP도 탈 필요없음
        --! ITEM, SITE, WEEK에 대해 INV수량과 DEMAND 수량이 같으면 계산 더이상 필요없음 !--
        IF R.INVQTY = R.DMDQTY THEN
            
            S_REMAINQTY := 0;
            
        --SRC LOOP도 탈 필요없음
        --! ITEM, SITE, WEEK에 대해 INV > DEMAND : 남은 가용량만 다음주로 미뤄줌 !--
        ELSIF  R.INVQTY > R.DMDQTY THEN
        
            S_REMAINQTY := R.INVQTY - R.DMDQTY;
            
        -- SRC LOOP 필요
        --! ITEM, SITE, WEEK에 대해 INV < DEMAND : 남는 가용량 없음 !--
        ELSE  -- C_AVAILQTY < C_DMDQTY
            S_REMAINQTY := 0;
        
        END IF;
        
        
     END LOOP;
     CLOSE CUR_AVAIL;
     DBMS_OUTPUT.PUT_LINE('2. CUR_AVAIL Data Load:  '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));

     --! STEP 3. NETTING !--
     OPEN CUR_AVAIL(V_PLANID);
     LOOP
        --! 1. Cursor Fetch !--
        FETCH CUR_AVAIL INTO R;
        EXIT WHEN CUR_AVAIL%NOTFOUND;
        
        -- S_REMAINQTY : 이전WEEK에서 남겨온 수량 
        --! EOL Netting 대상이면서 가용량이 존재하지 않으면!--
        IF  R.AVAILQTY = 0 AND  R.DMDQTY > 0  THEN
        
            --! EOL Netting 대상이면서 가용량이 존재하지 않으면 Demand 수량 모두 삭제 !--
            UPDATE EXP_SOPROMISESRCNCP--EXP_SOPROMISESRCNCP
            SET    QTYPROMISED = 0 ,
                   UPBY = 'EOP_NET_0'
            WHERE  PLANID = R.PLANID
            AND    SITEID = R.SITEID
            AND    ITEM   = R.ITEM
            AND    TO_CHAR(PROMISEDDELDATE,'IYYYIW') = R.WEEK;
            
            COMMIT;
        
        --! EOL Netting 대상이면서 가용량보다 DEMAND가 많은 경우 !--    
        ELSIF R.DMDQTY > R.AVAILQTY AND R.AVAILQTY>0 THEN 
        
            --! 가용량보다 DEMAND가 많은 경우 해당 주차 PRIORITY별로 나눠주기, 동순위위는 N빵 !--
            SELECT RANK() OVER (ORDER BY ROWNUM) ROWNUMBER, RK, RANK() OVER (PARTITION BY RK ORDER BY ROWNUM DESC, QTYPROMISED ) CHRK,
               ROWIDs,
               COUNT(*) OVER (PARTITION BY RK ORDER BY DEMANDPRIORITY ,RK RANGE UNBOUNDED PRECEDING) AS CNT,
               SUM(QTYPROMISED) OVER (PARTITION BY RK ORDER BY DEMANDPRIORITY ,RK RANGE UNBOUNDED PRECEDING) AS SUBTOTAL,
               NVL(RATIO_TO_REPORT(QTYPROMISED) OVER (PARTITION BY DEMANDPRIORITY ,RK),0) RATIO,
               ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, 
               QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, 
               DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, 
               GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, 
               NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, 
               INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE
             BULK COLLECT INTO mydata
             FROM ( SELECT   RANK() OVER (PARTITION BY A.ITEM,A.SITEID ORDER BY A.DEMANDPRIORITY ) RK,
                             A.ROWID ROWIDS,
                             ENTERPRISE, SOPROMISEID, A.PLANID, SALESORDERID, SOLINENUM, A.ITEM, 
                             QTYPROMISED, PROMISEDDELDATE, A.SITEID, SHIPTOID, SALESID, SALESLEVEL, 
                             DEMANDTYPE, A.WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, 
                             GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, 
                             NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, 
                             INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE
                   FROM      EXP_SOPROMISESRCNCP A --EXP_SOPROMISESRCNCP A
                   WHERE     A.PLANID             = R.PLANID
                   AND       TO_CHAR(A.PROMISEDDELDATE,'IYYYIW') = R.WEEK
                   AND       A.ITEM               = R.ITEM
                   AND       A.SITEID             = R.SITEID
                   AND       A.QTYPROMISED > 0
                   ORDER BY  A.DEMANDPRIORITY, A.QTYPROMISED, A.SALESID
                 )
             ORDER BY RK,DEMANDPRIORITY , ROWNUM, QTYPROMISED ;
          
             
           IF mydata.COUNT > 0 THEN
               FOR j IN mydata.first .. mydata.last
               LOOP 
                    --! 첫행이거나 !--
                    IF  MYDATA(J).ROWNUMBER = 1 THEN 
                         
                     
                        L_DEMANDPRIORITY := MYDATA(J).DEMANDPRIORITY;
                        
                        R_QTY      := R.AVAILQTY;
                        STEP_R_QTY := R.AVAILQTY;
                        FARE_SHARE := 0 ;
                    --! 우선순위 바뀔때 !--    
                    ELSIF L_DEMANDPRIORITY <> MYDATA(J).DEMANDPRIORITY THEN
 
                        L_DEMANDPRIORITY := MYDATA(J).DEMANDPRIORITY;
                        
                        R_QTY := STEP_R_QTY;
                        FARE_SHARE := 0 ;
                        
                    END IF;
                    
                    --! 동순위 없을때 !--
                    IF MYDATA(J).CNT = 1 THEN
                        --! 남은 수량> QTYPROMISE !--
                        IF R_QTY >= MYDATA(J).QTYPROMISED THEN
                           --! 남은 수량> QTYPROMISE   작업필요없고 잔량 처리만!--
                           STEP_R_QTY := STEP_R_QTY - MYDATA(J).QTYPROMISED ;
                        --! 남은 수량 < QTYPROMISE !--
                        ELSE 
                            --! 남은 수량 < QTYPROMISE  이면 남은수량만큼 UP0DATE 해줘야 함 !--
                           UPDATE EXP_SOPROMISESRCNCP --EXP_SOPROMISESRCNCP
                           SET    QTYPROMISED = R_QTY , 
                                  UPBY       = 'EOP_NET_1'
                           WHERE  SALESORDERID = MYDATA(J).SALESORDERID
                           AND    PLANID       = MYDATA(J).PLANID
                           AND    SOLINENUM    = MYDATA(J).SOLINENUM
                           AND    PROMISEDDELDATE = MYDATA(J).PROMISEDDELDATE;
                            
                           STEP_R_QTY := STEP_R_QTY - R_QTY ;
                           
                        END IF;
                    
                    ELSE
                        --! 동순위 있을 때 !--
                        IF R_QTY >= MYDATA(J).SUBTOTAL THEN
                           
                           --! 남은 수량> QTYPROMISE   작업필요없고 잔량 처리만 !--
                           STEP_R_QTY := STEP_R_QTY - MYDATA(J).QTYPROMISED ;
                        
                        ELSE 
                           --! 남은 수량 < QTYPROMISE  이면 남은수량 가지고 N빵 처리 해줘야 함 !--
                           UPDATE EXP_SOPROMISESRCNCP --EXP_SOPROMISESRCNCP
                           SET    QTYPROMISED = TRUNC(R_QTY * MYDATA(J).RATIO) , 
                                  UPBY        = 'EOP_NET_N'
                           WHERE  SALESORDERID = MYDATA(J).SALESORDERID
                           AND    PLANID       = MYDATA(J).PLANID
                           AND    SOLINENUM    = MYDATA(J).SOLINENUM
                           AND    PROMISEDDELDATE = MYDATA(J).PROMISEDDELDATE;
                            
                           STEP_R_QTY := STEP_R_QTY - TRUNC(R_QTY * MYDATA(J).RATIO) ;
                           FARE_SHARE := FARE_SHARE + TRUNC(R_QTY * MYDATA(J).RATIO) ;
                           
                           --! N빵처리후 마지막 ROW이면 !--
                           IF MYDATA(J).CHRK = 1 THEN
                                --! N빵처리후 마지막 ROW에서 남는 수량 1,2,3..FARE SHARE를 위해 다시한번 !--
                                SELECT RANK() OVER (ORDER BY ROWNUM) ROWNUMBER, RK, RANK() OVER (PARTITION BY RK ORDER BY ROWNUM DESC, QTYPROMISED ) CHRK,
                                   ROWIDs,
                                   COUNT(*) OVER (PARTITION BY RK ORDER BY DEMANDPRIORITY ,RK RANGE UNBOUNDED PRECEDING) AS CNT,
                                   SUM(QTYPROMISED) OVER (PARTITION BY RK ORDER BY DEMANDPRIORITY ,RK RANGE UNBOUNDED PRECEDING) AS SUBTOTAL,
                                   NVL(RATIO_TO_REPORT(QTYPROMISED) OVER (PARTITION BY DEMANDPRIORITY ,RK),0) RATIO,
                                   ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, 
                                   QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, 
                                   DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, 
                                   GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, 
                                   NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, 
                                   INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE
                                 BULK COLLECT INTO mydata_T
                                 FROM ( SELECT   RANK() OVER (PARTITION BY A.ITEM,A.SITEID ORDER BY A.DEMANDPRIORITY ) RK,
                                                 A.ROWID ROWIDS,
                                                 SUM(A.QTYPROMISED) OVER (PARTITION BY A.SALESORDERID) PROD, --N빵직전 EOP NETTING후
                                                 SUM(B.QTYPROMISED) OVER (PARTITION BY A.SALESORDERID) DEV,  --EOL NETTING전
                                                 A.ENTERPRISE , A.SOPROMISEID, A.PLANID, A.SALESORDERID, A.SOLINENUM,    
                                                 A.ITEM ,A.QTYPROMISED ,A.PROMISEDDELDATE ,A.SITEID ,A.SHIPTOID ,A.SALESID,        
                                                 A.SALESLEVEL, A.DEMANDTYPE, A.WEEK, A.CHANNELRANK ,A.CUSTOMERRANK ,A.PRODUCTRANK ,A.DEMANDPRIORITY ,A.TIEBREAK ,A.GBM ,A.GLOBALPRIORITY ,A.LOCALPRIORITY,    
                                                 A.BUSINESSTYPE ,A.ROUTING_PRIORITY ,A.NO_SPLIT ,A.MAP_SATISFY_SS ,A.PREALLOC_ATTRIBUTE ,A.BUILDAHEADTIME ,A.TIMEUOM,        
                                                 A.AP2ID ,A.GC ,A.MEASURERANK ,A.PREFERENCERANK ,A.INITDTTM ,A.INITBY ,A.UPDTTM ,A.UPBY ,A.REASONCODE    
                                       FROM      EXP_SOPROMISESRCNCP A, --EXP_SOPROMISESRCNCP A, 
                                                 BUF_SOPROMISESRCNCP_POST B
                                       WHERE     A.PLANID             = R.PLANID
                                       AND       TO_CHAR(A.PROMISEDDELDATE,'IYYYIW') = R.WEEK
                                       AND       A.ITEM               = R.ITEM
                                       AND       A.SITEID             = R.SITEID
                                       AND       A.UPBY LIKE 'EOP_NET_N%' -- 0 보다 큰놈을 고르면 EOP로 인해 0 되버린거 FARE SHARE 제외되고
                                                                     -- 0 보다 큰놈 조건 안넣으면 원래 DEMAND 0인 놈들이 +1 되버릴수도 있음
                                       AND       B.PLAN               = 'EOL'
                                       AND       A.SALESORDERID       = B.SALESORDERID
                                       AND       A.PLANID             = B.PLANID
                                       AND       A.SOLINENUM          = B.SOLINENUM    
                                       ORDER BY  A.DEMANDPRIORITY, A.QTYPROMISED, A.SALESID
                                     )
                                 WHERE  DEV > PROD  --NETTING 전보다 데이터가 작아야만 PLUS 해도 상관없음, 같은거는 PLUS 되면 처음 DEMAND보다 많아지므로 안됨
                                 ORDER BY RK, DEMANDPRIORITY , ROWNUM, QTYPROMISED ;
          
             
                                   IF mydata_T.COUNT > 0 THEN
                                       FOR k IN mydata_T.first .. (R_QTY - FARE_SHARE )
                                       LOOP
                                       
                                        UPDATE EXP_SOPROMISESRCNCP--EXP_SOPROMISESRCNCP
                                        SET    QTYPROMISED = QTYPROMISED +1
                                               , UPBY = UPBY||'_PLUS'
                                        WHERE  SALESORDERID = mydata_T(K).SALESORDERID
                                        AND    PLANID       = MYDATA_T(K).PLANID
                                        AND    SOLINENUM    = MYDATA_T(K).SOLINENUM;
                                       
                                       END LOOP;
                                   END IF;
                                   
                                   
                                   --! 마지막 N빵 처리하면 잔량이 0 되어야함 !--
                                  STEP_R_QTY := 0; 

                           END IF;
                           
                        END IF;

                    END IF;

               END LOOP;
           END IF;
        
            COMMIT;
            
        END IF;
        
     END LOOP;
     CLOSE CUR_AVAIL;
     DBMS_OUTPUT.PUT_LINE('3. EOL Netting For Loop END :  '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));
     
--    --공용화 재고 분배
--    --! 4-1 REPBULK ITEM INVENTORY 테이블 생성 !--
--    BEGIN 
--        --! 현재 PLANID로 생성되어 있는 잔여 데이터 삭제 !--
--        DELETE FROM MST_REPINVENTORY_EOL     
--        WHERE PLANID = V_PLANID;
--    
--        --! 현재 PLANID로 REPBULK ITEM INVENTORY 데이터 INSERT !--
--        INSERT INTO MST_REPINVENTORY_EOL 
--              (PLANID, SITEID, ITEM, WEEK, QTY, INITDTTM, INITBY)
--        SELECT PLANID, SITEID, ITEM, WEEK, SUM(AVAILQTY) AVAILQTY, SYSDATE, V_Pgmname
--        FROM(
--            SELECT V_PLANID PLANID, A.SITEID, A.ITEM,  A.WEEK , NVL(SUM(AVAILQTY),0) AVAILQTY
--            FROM  (-- 재고 DATA 발췌 : TOSITEID = '-'로 들어오는 경우 처리를 위한 작업
--                   SELECT DECODE(SITEID,'-',MOD_SITEID,SITEID) SITEID,
--                          ITEM, WEEK,
--                          SUM(AVAILQTY) AVAILQTY , 0 
--                   FROM (       
--                           -- [2011.11.17: BEGIN] 구주포장센터(MTA_SALESBOMMAP) 관련 변경 작업
--                           SELECT A.SITEID SITEID,
--                                  MAX(A.SITEID) OVER (PARTITION BY A.ITEM) MOD_SITEID,
--                                  A.ITEM ITEM, V_PLANWEEK WEEK, AVAILQTY     
--                           FROM (
--                                 /* 1. 재고(PY, CY, 판매법인재고) */
--                                 SELECT A.PLANID, DECODE(B.TYPE,'MF',A.TOSITEID,'DC',A.SITEID) SITEID, A.ITEM, V_PLANWEEK WEEK,
--                                      AVAILQTY+BOHADDQTY+W0BOHADDQTY AVAILQTY
--                                 FROM MST_INVENTORY A, MST_SITE B, 
--                                    ( SELECT A.FROMSITEID, A.TOSITEID, B.ITEM, MIN(B.TRANSITTIME) LT 
--                                      FROM MST_BOD A, MST_BODDETAIL B
--                                      WHERE A.BODNAME = B.BODNAME
--                                      AND   NVL(B.EFFENDDATE,V_EFFENDDATE) >= V_EFFSTARTDATE            
--                                      GROUP BY A.FROMSITEID, A.TOSITEID, B.ITEM   
--                                    ) C
--                                 WHERE A.PLANID   = V_PLANID
--                                 AND   A.SITEID   = B.SITEID
--                                 AND   A.SITEID   = C.FROMSITEID(+)
--                                 AND   A.TOSITEID = C.TOSITEID(+)
--                                 AND   A.ITEM     = C.ITEM(+)
--                                 AND    NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = DECODE(B.TYPE,'MF',A.TOSITEID,'DC',A.SITEID))
--                                 UNION ALL          
--                                 /* 2. 향해재고 */      
--                                 SELECT PLANID, TOSITEID, ITEM, TO_CHAR(A.SCMETA,'IYYYIW') WEEK, INTRANSITQTY
--                                 FROM   MST_INTRANSIT A
--                                 WHERE  PLANID = V_PLANID
--                                 AND    NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.TOSITEID)
--                                ) A
--                        )    A
--                   WHERE  DECODE(SITEID,'-',MOD_SITEID,SITEID) NOT IN ('-')    
--                   GROUP BY DECODE(SITEID,'-',MOD_SITEID,SITEID), ITEM, WEEK              
--                   )  A,  
--                  (
--                    SELECT DISTINCT SITEID, REPMAINSKU 
--                    FROM MTA_SALESBOMMAP 
--                    WHERE ISPOSTPONEMENT = 'Y' AND STATUS = 'CON'                                                                      
--                  ) B
--            WHERE A.ITEM    = B.REPMAINSKU
--            AND A.SITEID = B.SITEID
--            GROUP BY A.SITEID, A.ITEM, A.WEEK
--         )
--        GROUP BY PLANID,SITEID, ITEM, WEEK;
--        
--        COMMIT;
--    
--    EXCEPTION
--    
--    WHEN OTHERS THEN
--        DBMS_OUTPUT.PUT_LINE('4-1. MST_EOLNETTING_REPINV insert Error : check '||SQLCODE||'_'||SQLERRM);
--    ROLLBACK;
--    
--    END;
--    
--    TUNE.SHOW('4-1. MST_EOLNETTING_REPINV insert' ||' END['||TO_CHAR(SYSDATE, 'YYYY/MM/DD HH24:MI:SS')||']');
--
--     --! 4-2 EOL NETTING시 잘린 DEMAND만 추출하여 BUF 테이블에 INSERT !--
--    BEGIN
--        --! TEMP 테이블 Truncate !--
--        EXECUTE IMMEDIATE 'TRUNCATE TABLE BUF_DEMAND_EOL';
--        
--        --! TEMP 테이블에 잘린 DEMAND INSERT !--
--        INSERT INTO BUF_DEMAND_EOL 
--              ( PLANID, SALESORDERID, SOLINENUM, ITEM, PRE_QTYPROMISED, QTYPROMISED, WEEK, SITEID, SALESID,
--                DEMANDPRIORITY, GC, REMAINDMD)
--        SELECT /*+ leading(b) use_hash(b a c) */
--               A.PLANID, A.SALESORDERID, A.SOLINENUM, A.ITEM, C.QTYPROMISED PRE_QTYPROMISED, A.QTYPROMISED, TO_CHAR(A.PROMISEDDELDATE, 'IYYYIW') WEEK, A.SITEID, A.SALESID, 
--               A.DEMANDPRIORITY, A.GC, (C.QTYPROMISED - A.QTYPROMISED) REMAINDMD
--        FROM   EXP_SOPROMISESRCNCP A, 
--              (SELECT SITEID, ITEM, EOPWEEK , SUM(INVQTY) INVQTY FROM MST_EOLNETTING WHERE  PLANID = V_PLANID GROUP BY SITEID, ITEM, EOPWEEK ) B, 
--               BUF_SOPROMISESRCNCP_POST C
--        WHERE  A.PLANID       = V_PLANID
--        AND    C.QTYPROMISED  > 0
--        AND    A.SALESORDERID NOT LIKE 'UNF_ORD%'
--        AND    A.ITEM         = B.ITEM
--        AND    A.SITEID       = B.SITEID
--        AND    C.PLAN         = 'EOL'
--        AND    A.PLANID       = C.PLANID 
--        AND    A.SALESORDERID = C.SALESORDERID
--        AND    A.SOLINENUM    = C.SOLINENUM
--        UNION ALL         
--        SELECT PLANID, NULL SALESORDERID, NULL SOLINENUM, B.HEADSKU ITEM, 0 PRE_QTYPROMISED, 0 QTYPROMISED, WEEK, B.HEADSITEID SITEID, NULL SALESID,
--               NULL DEMANDPRIORITY, NULL GC, 0 REMAINDMD
--        FROM MST_REPINVENTORY_EOL A, 
--            (SELECT HEADSKU, HEADSITEID, SITEID, REPMAINSKU 
--             FROM MTA_SALESBOMMAP 
--             WHERE ISPOSTPONEMENT = 'Y' AND STATUS = 'CON') B
--        WHERE PLANID = V_PLANID
--        AND A.ITEM = B.REPMAINSKU
--        AND A.SITEID = B.SITEID
--        AND NOT EXISTS (SELECT 'X' FROM v_mta_sellermap WHERE ITEM = A.ITEM AND SITEID = A.SITEID);    
--        
--        COMMIT;
--        
--    EXCEPTION
--    
--    WHEN OTHERS THEN
--        DBMS_OUTPUT.PUT_LINE('4-2. BUF_DEMAND_EOL insert Error : check '||SQLCODE||'_'||SQLERRM);
--    ROLLBACK;
--    
--    END;
--    
--    TUNE.SHOW('4-2. BUF_DEMAND_EOL Insert' ||' END['||TO_CHAR(SYSDATE, 'YYYY/MM/DD HH24:MI:SS')||']');    
--    
--    --! 4-3 공용화 가용량을 HEADSKU, HEADSITE 기준으로 맵핑하여 DEMAND와 조인하여 MERGE UPDATE !--
--    BEGIN
--        
--        --! TEMP 테이블에 DEMAND별 공용화 가용량 배분현황 MERGE UPDATE !--
--        MERGE INTO BUF_DEMAND_EOL T
--        USING(
--            SELECT A.ROWA, A.PLANID, A.SALESORDERID, A.SOLINENUM, A.ITEM, A.PRE_QTYPROMISED, A.QTYPROMISED, A.WEEK, A.SITEID, A.SALESID, A.DEMANDPRIORITY, A.GC, 
--                   A.REMAINDMD,
--                   A.REPMAINSKU REP_SKU,
--                   A.REPSITEID REP_SITEID,
--                   nvl(B.QTY,0) REP_INVQTY,
--                   ROW_NUMBER() OVER(PARTITION BY A.PLANID, A.REPMAINSKU, A.REPSITEID, A.WEEK ORDER BY NVL(A.DEMANDPRIORITY,0) ASC ) PART_CNT,
--                   COUNT(*) OVER(PARTITION BY A.PLANID, A.REPMAINSKU, A.REPSITEID, A.WEEK) PART2_CNT,
--                   ROW_NUMBER() OVER(PARTITION BY A.PLANID, A.REPMAINSKU, A.REPSITEID ORDER BY A.WEEK, NVL(A.DEMANDPRIORITY,0)  ASC ) ALL_CNT,
--                   NVL(SUM(A.REMAINDMD) OVER (PARTITION BY A.PLANID, A.REPMAINSKU, A.REPSITEID, A.WEEK ORDER BY NVL(A.DEMANDPRIORITY,0) ASC ROWS UNBOUNDED PRECEDING ) ,0) SUM_DEMAND
--            FROM (  SELECT A.ROWID ROWA, B.REPMAINSKU,  B.SITEID REPSITEID, 
--                    A.PLANID, A.SALESORDERID, A.SOLINENUM, A.ITEM, A.PRE_QTYPROMISED, A.QTYPROMISED, A.WEEK, A.SITEID, A.SALESID, A.DEMANDPRIORITY, 
--                    A.GC, A.REP_SKU, A.REMAINDMD, A.REP_INVQTY, A.AVAILQTY, A.FIXQTY, A.PART_CNT, A.PART2_CNT, A.ALL_CNT
--                    FROM BUF_DEMAND_EOL A,  
--                        (SELECT HEADSKU, HEADSITEID, SITEID, REPMAINSKU 
--                         FROM MTA_SALESBOMMAP 
--                         WHERE ISPOSTPONEMENT = 'Y' AND STATUS = 'CON') B
--                    WHERE A.ITEM = B.HEADSKU
--                    AND A.SITEID = B.HEADSITEID
--                  ) A, MST_REPINVENTORY_EOL B
--            WHERE A.PLANID = B.PLANID(+)
--            AND A.REPMAINSKU = B.ITEM(+)
--            AND A.REPSITEID = B.SITEID(+)
--            AND A.WEEK= B.WEEK(+)
--        ) S
--        ON(T.ROWID = S.ROWA)
--        WHEN MATCHED THEN
--        UPDATE SET
--            T.REP_SKU = S.REP_SKU,
--            T.REP_SITEID = S.REP_SITEID,
--            T.REP_INVQTY = S.REP_INVQTY,
--            T.PART_CNT = S.PART_CNT,
--            T.PART2_CNT = S.PART2_CNT,
--            T.ALL_CNT = S.ALL_CNT,
--            T.SUM_DEMAND = S.SUM_DEMAND ; 
--        
--        COMMIT;
--    EXCEPTION
--    
--    WHEN OTHERS THEN
--        DBMS_OUTPUT.PUT_LINE('4-3. BUF_DEMAND_EOL REPINV Merge Update Error : check '||SQLCODE||'_'||SQLERRM);
--    ROLLBACK;
--    
--    END;
--    
--    TUNE.SHOW('4-3. BUF_DEMAND_EOL REPINV Merge Update' ||' END['||TO_CHAR(SYSDATE, 'YYYY/MM/DD HH24:MI:SS')||']');    
--    
--    --! 4-4 REP BULK 가용량을 주차별로 ROLLING 하여 DEMAND에 배분 !--
--    BEGIN
--        --! 변수 초기화 !--
--        V_REMAINQTY := 0;
--        V_AVAILQTY  := 0;
--        
--        --! 공용 가용량을 분배받아야 할 대상 DEMAND만 발췌하여 FOR LOOP 시작 !--
--        FOR S IN (
--            SELECT  ROWID ROWA, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED, WEEK, SITEID, SALESID, DEMANDPRIORITY, GC,
--                    REP_SKU, REP_SITEID, REMAINDMD, SUM_DEMAND, REP_INVQTY, AVAILQTY, FIXQTY, REMAINQTY, PART_CNT, PART2_CNT, ALL_CNT
--            FROM BUF_DEMAND_EOL
--            WHERE REP_SKU IS NOT NULL 
--              AND ALL_CNT IS NOT NULL
--            ORDER BY PLANID, REP_SKU, REP_SITEID, WEEK, ALL_CNT    
--        )
--        LOOP 
--            --! 새로운 모델/Site 인 경우 잔량 0처리(전주에서 넘겨받은 남은 가용량이 없으므로 0 처리) !--
--            IF S.ALL_CNT = 1 THEN 
--                --! 전주에서 넘겨온 수량 0 !--
--                V_REMAINQTY := 0;
--            END IF;     
--                 
--            --! 주차가 바뀌는 경우 앞에 남은 수량은 뒤로 Move !--
--            IF S.PART_CNT = 1 THEN 
--            
--            --! V_AVAILQTY = 전주에서 넘어온 수량 + 배정된 가용량 !--
--            V_AVAILQTY := S.REP_INVQTY + V_REMAINQTY;          
--            END IF;
--       
--            --! 가용량이 Demand 누적합보다 큰 경우 전량 가용량 배분 !--
--            IF V_AVAILQTY >= S.SUM_DEMAND THEN
--            
--            --! 분배정보 Update, FIXQTY = 배정된 가용량, Availqty = 현재 ROW에서 사용가능한 가용량 !--         
--            UPDATE BUF_DEMAND_EOL
--               SET FIXQTY = S.REMAINDMD,
--                   AVAILQTY = V_AVAILQTY - (S.SUM_DEMAND - S.REMAINDMD)
--             WHERE ROWID = S.ROWA;
--
--            --! 배분한 가용량 만큼 DEMAND 수량 살림 !--
--            UPDATE EXP_SOPROMISESRCNCP
--               SET QTYPROMISED = S.QTYPROMISED + S.REMAINDMD
--             WHERE PLANID = S.PLANID
--               AND SALESORDERID = S.SALESORDERID
--               AND SOLINENUM = S.SOLINENUM; 
--                
--            ELSE
--                --! 가용량이 누적합보다는 작으나 일부 물량을 배분할 수 있는 경우 !--        
--                IF V_AVAILQTY - (S.SUM_DEMAND - S.REMAINDMD) > 0 THEN
--                --부분 수량 배분
--                    --! 분배정보 Update, FIXQTY = 배정된 가용량, Availqty = 현재 ROW에서 사용가능한 가용량 !-- 
--                    UPDATE BUF_DEMAND_EOL
--                       SET FIXQTY = V_AVAILQTY - (S.SUM_DEMAND - S.REMAINDMD),
--                           AVAILQTY = V_AVAILQTY - (S.SUM_DEMAND - S.REMAINDMD)
--                     WHERE ROWID = S.ROWA;
--                     
--                     --! 배분한 가용량 만큼 DEMAND 수량 살림 !--
--                    UPDATE EXP_SOPROMISESRCNCP
--                       SET QTYPROMISED = S.QTYPROMISED + (V_AVAILQTY - (S.SUM_DEMAND - S.REMAINDMD))
--                     WHERE PLANID = S.PLANID
--                       AND SALESORDERID = S.SALESORDERID
--                       AND SOLINENUM = S.SOLINENUM; 
--                                     
--                END IF;
--
--            END IF;
--            
--            --! 분배정보 Update,  REMAINQTY = 모두 배분하고도 남은 가용량 !-- 
--            UPDATE BUF_DEMAND_EOL
--               SET REMAINQTY = GREATEST(AVAILQTY - S.REMAINDMD,0)
--             WHERE ROWID = S.ROWA;   
--                         
--            --!주차가 끝나는 시점에서 사용되고 남은 가용량 산출하여 차주로 Move !--
--            IF S.PART_CNT = S.PART2_CNT THEN 
--             V_REMAINQTY := CASE WHEN V_AVAILQTY - S.SUM_DEMAND < 0 THEN 0 ELSE V_AVAILQTY - S.SUM_DEMAND END;
--  
--            END IF;
--          
--        END LOOP;
--       
--        COMMIT;
--           
--    EXCEPTION
--    
--    WHEN OTHERS THEN
--        DBMS_OUTPUT.PUT_LINE('4-3. BUF_DEMAND_EOL REP Rolling Error : check '||SQLCODE||'_'||SQLERRM);
--    ROLLBACK;
--    
--    END;
--    
--    TUNE.SHOW('4-3. BUF_DEMAND_INV_EOL REP Rolling' ||' END['||TO_CHAR(SYSDATE, 'YYYY/MM/DD HH24:MI:SS')||']');             

    
        --! ARC_EOPDEMANDHISTORY에서 해당 Planid 데이터 Delete !--
        DELETE ARC_EOPDEMANDHISTORY_NC
        WHERE PLANID = V_PLANID;  
        
        --! EOL로 잘린 DEMAND를 표시해주기 위한 UI용 데이터 생성 !--
        INSERT INTO ARC_EOPDEMANDHISTORY_NC
        SELECT /*+ leading(b) use_hash(b a c) */
               A.PLANID, A.SALESORDERID, A.SOLINENUM, A.ITEM, C.QTYPROMISED, B.INVQTY, A.QTYPROMISED MODIFY_QTY, 
               TO_CHAR(A.PROMISEDDELDATE, 'IYYYIW') WEEK , B.EOPWEEK, A.SITEID, A.SALESID, A.DEMANDPRIORITY, A.GC, SYSDATE, 'SP_FN_SOPROMISESRC_EOPMODIFY' , NULL, NULL 
        FROM   EXP_SOPROMISESRCNCP A, --EXP_SOPROMISESRCNCP A, 
              (SELECT SITEID, ITEM, EOPWEEK , SUM(INVQTY) INVQTY FROM MST_EOLNETTING WHERE  PLANID = V_PLANID GROUP BY SITEID, ITEM, EOPWEEK ) B, 
               BUF_SOPROMISESRCNCP_POST C
        WHERE  A.PLANID       = V_PLANID
        AND    C.QTYPROMISED  > 0
        AND    A.SALESORDERID NOT LIKE 'UNF_ORD%'
        AND    A.ITEM         = B.ITEM
        AND    A.SITEID       = B.SITEID
        AND    C.PLAN         = 'EOL'
        AND    A.PLANID       = C.PLANID 
        AND    A.SALESORDERID = C.SALESORDERID
        AND    A.SOLINENUM    = C.SOLINENUM;
    
        COMMIT;
     
        V_LOGMESSAGE :=   'INSERT COUNT : '||'0'||CHR(10)
                                    ||'  UPDATE COUNT : '||'0'||CHR(10)
                                ||'  ERROR  COUNT : '; -- LOGMESSAGE
        SP_LOGGING_REG (V_LOGSEQ, V_Pgmname,V_STARTDATE,0,V_LOGMESSAGE);
        
EXCEPTION
    WHEN OTHERS THEN DBMS_OUTPUT.PUT_LINE(SQLCODE||'_'||SQLERRM);
    ROLLBACK;
END;
