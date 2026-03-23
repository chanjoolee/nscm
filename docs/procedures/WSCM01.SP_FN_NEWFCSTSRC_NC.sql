CREATE OR REPLACE PROCEDURE WSCM01.SP_FN_NEWFCSTSRC_NC (IN_TYPE VARCHAR2 DEFAULT 'NCP') IS
/******************************************************************************
   NAME:       WSCM01.SP_FN_NEWFCSTSRC_NC
   PURPOSE:    최종 SRC 데이터 분배   
   REVISIONS:
   VER     DATE        AUTHOR   DESCRIPTION
   ------- ----------  -------  ------------------------------------
   1.0     2019.07.03  HYEMI  신규 버전
 ******************************************************************************/
 
    V_PGMNAME                VARCHAR2(30) := 'SP_FN_NEWFCSTSRC_NC';
    V_LOADINGTIME            VARCHAR2(2);
    V_RETSQLERRM             VARCHAR2(2000);
    
    --V_PLANID               MST_PLAN.PLANID %TYPE;
    V_PLANID                 EXP_SOPROMISESRCNCP.PLANID%TYPE; 
    V_PLANWEEK               MST_PLAN.PLANWEEK%TYPE;
    V_EFFSTARTDATE           MST_PLAN.EFFSTARTDATE %TYPE;
    V_EFFENDDATE             MST_PLAN.EFFENDDATE %TYPE;
    v_ruleplan               VARCHAR(20);
   
    V_PREPLANID              EXP_SOPROMISESRCNCP.PLANID%TYPE; 
    V_WEEK1                  MST_WEEK.WEEK%TYPE; 
    V_WEEK4                  MST_WEEK.WEEK%TYPE;
    
    C1_PLANID                MST_NEWFCSTNETTING.PLANID%TYPE;
    C1_SITEID                MST_NEWFCSTNETTING.SITEID%TYPE;
    C1_ITEM                  MST_NEWFCSTNETTING.ITEM%TYPE;
    C1_WEEK                  MST_NEWFCSTNETTING.WEEK%TYPE; 
    C1_RELEASE               MST_NEWFCSTNETTING.RELEASE%TYPE;
    C1_PREQTY                MST_NEWFCSTNETTING.PREQTY%TYPE; 
    C1_CURQTY                MST_NEWFCSTNETTING.CURQTY%TYPE; 
    C1_DIFF                  MST_NEWFCSTNETTING.DIFF%TYPE; 
    C1_MPDP                  MST_NEWFCSTNETTING.MPDP%TYPE; 
    C1_NEWDP                 MST_NEWFCSTNETTING.NEWDP%TYPE; 
    M_CALCNEWDP              NUMBER; 
    M_PRECALCNEWDP           NUMBER; 
    M_DEMANDPRIORITY         BUF_SOPROMISESRCNCP_POST.DEMANDPRIORITY%TYPE :=0;
    F_NEWDP                  MST_NEWFCSTNETTING.NEWDP%TYPE; 
    C_NEWDP                  MST_NEWFCSTNETTING.NEWDP%TYPE; 
    
  
    
    SMS_COUNT                NUMBER := 0;
    SMS_QTYCOUNT             NUMBER := 0;
    SMS_QTYBEFORE            NUMBER := 0;
    SMS_QTYAFTER             NUMBER := 0;
                    
    
    V_LOGMESSAGE        LONG;
    V_ERRMESSAGE        VARCHAR2(2000);
    V_LOGSEQ            NUMBER(10);
    V_ERRORCNT          INTEGER:=0;
    V_STARTDATE         DATE;

--    TYPE  TYPE_CURSOR       IS REF CURSOR;
--    CUR_FINALNEWFCST        TYPE_CURSOR;

    S_RENCPQTY         NUMBER := 0;
    L_DEMANDPRIORITY    NUMBER := 0;
    STEP_R_QTY          NUMBER := 0;
    R_QTY               NUMBER := 0;
    FARE_SHARE          NUMBER := 0;
    
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
--      2_AP2ID_ORI             EXP_SOPROMISESEW.AP2ID_ORI%TYPE,         
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
    
    
    CURSOR CUR_OPENNETTING(C_PLANID VARCHAR) IS
        SELECT  PLANID, SITEID, ITEM, WEEK, PREQTY, CURQTY, DIFF, MPDP, NEWDP
        FROM    MST_NEWFCSTNETTING   
        WHERE   PLANID = C_PLANID 
        AND     NEWDP  > 0 
        AND     SITEID <> 'TOTAL'
--        AND     ITEM = 'EFC-1J9BBEGSTD'
--        and     siteid = 'L401'
        ORDER BY PLANID, SITEID, ITEM, WEEK;  

    R CUR_OPENNETTING%ROWTYPE; 
        
BEGIN

    SELECT SEQ_LOG.NEXTVAL INTO V_LOGSEQ FROM DUAL;
    SELECT SYSDATE INTO V_STARTDATE FROM DUAL;
    
    --* GET CURRENT RUNNING PLANID
    SP_FN_GET_PLANID('NCP',V_PLANID,V_PLANWEEK,V_EFFSTARTDATE,V_EFFENDDATE);
     
     --2018.12.03 수정(V PLAN일 경우 당주기준으로 effstartdate 변경)
    SELECT TYPE
    INTO   V_TYPE
    FROM   MST_PLAN 
    WHERE  ISRUNNING = 'Y'
    AND    TYPE = 'NCP';
          
    IF V_TYPE = 'VPLAN' THEN
        V_WEEK4     := TO_CHAR(V_EFFSTARTDATE + 7*4, 'IYYYIW');
    ELSE
        V_WEEK4     := TO_CHAR(V_EFFSTARTDATE + 7*3, 'IYYYIW');
    END IF;

     V_WEEK1     := V_PLANWEEK;
--     V_WEEK4     := TO_CHAR(V_EFFSTARTDATE + 7*3, 'IYYYIW');
     
     DBMS_OUTPUT.PUT_LINE('START :    '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));
     DBMS_OUTPUT.PUT_LINE('V_PLANID : '||V_PLANID||'  V_WEEK1:'||V_WEEK1||'  V_WEEK4:'||V_WEEK4);
     
     
      EXECUTE IMMEDIATE 'ALTER TABLE BUF_SOPROMISESRCNCP_POST TRUNCATE PARTITION PNEWFCST';
      
      INSERT INTO BUF_SOPROMISESRCNCP_POST --SALESID(AP2ID)사용할수도 있으므로,,,일단 놔둠, 백업용 테이블
          ( PLAN, ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED, PROMISEDDELDATE
           , SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK
           , GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS
           , PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK
           , INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE )
         SELECT    'NEWFCST', ENTERPRISE, SOPROMISEID, A.PLANID, SALESORDERID, SOLINENUM, A.ITEM, A.QTYPROMISED, PROMISEDDELDATE
                 , A.SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, A.WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY
                 , TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT
                 , MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK
                 , INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE
          FROM     EXP_SOPROMISESRCNCP A
          WHERE    A.PLANID  =  V_PLANID;
          
     COMMIT;
        
    
     
     BEGIN 
     -- STEP 5. EXP_SOPROMISESRCNCP 분리
     -- MST_NEWNETTING ( SITEID, ITEM, WEEK 기준)에서 결정된 FINAL DP, NEWDP에 대해 SRC에 분배작업
     -- SRC에서 우선순위 DESC(낮은순), QTY 수량 큰 순으로 
     -- 우선순위가 동순위일 경우, 페어쉐어 (QTYPROMISED 들어와있는 수량비율대로)
     DBMS_OUTPUT.PUT_LINE('5.FINAL -> START OPEN  '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));
     
         OPEN CUR_OPENNETTING(V_PLANID);
         LOOP
            FETCH CUR_OPENNETTING
                  INTO  R;
            EXIT WHEN CUR_OPENNETTING%NOTFOUND;
            
            -- week 수량이 
            IF  R.NEWDP >= R.CURQTY AND  R.CURQTY > 0  THEN
            
                         
                UPDATE EXP_SOPROMISESRCNCP A
                SET    A.SOLINENUM   = A.SOLINENUM + 99 ,
                       A.UPBY        = 'NEWALL'
                WHERE  A.PLANID = R.PLANID
                AND    A.SITEID = R.SITEID
                AND    A.ITEM   = R.ITEM
                AND    TO_CHAR(A.PROMISEDDELDATE,'IYYYIW') = R.WEEK
                AND    NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID);     -- eStore Netting제외 20220907 추가
                
                COMMIT;
                
            ELSIF R.CURQTY > R.NEWDP AND R.NEWDP>0 THEN 
            
                -- 가용량보다 DEMAND가 많은 경우 해당 주차 PRIORITY별로 나눠주기
                -- 동순위위는 N빵
                SELECT RANK() OVER (ORDER BY RK, ROWNUM ) ROWNUMBER, RK, RANK() OVER (PARTITION BY RK ORDER BY ROWNUM DESC, QTYPROMISED ) CHRK,
                   ROWIDs,
                   COUNT(*) OVER (PARTITION BY RK ORDER BY DEMANDPRIORITY DESC ,RK RANGE UNBOUNDED PRECEDING) AS CNT,
                   SUM(QTYPROMISED) OVER (PARTITION BY RK ORDER BY DEMANDPRIORITY DESC,RK RANGE UNBOUNDED PRECEDING) AS SUBTOTAL,
                   NVL(RATIO_TO_REPORT(QTYPROMISED) OVER (PARTITION BY DEMANDPRIORITY ,RK),0) RATIO,
                   ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, 
                   QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, 
                   DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, 
                   GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, 
                   NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, 
                   INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE
                 BULK COLLECT INTO mydata
                 FROM ( SELECT   RANK() OVER (PARTITION BY A.ITEM,A.SITEID ORDER BY A.DEMANDPRIORITY DESC) RK,
                                 A.ROWID ROWIDS,
                                 ENTERPRISE, SOPROMISEID, A.PLANID, SALESORDERID, SOLINENUM, A.ITEM, 
                                 QTYPROMISED, PROMISEDDELDATE, A.SITEID, SHIPTOID, SALESID, SALESLEVEL, 
                                 DEMANDTYPE, A.WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, 
                                 GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, 
                                 NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, 
                                 INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE
                       FROM      EXP_SOPROMISESRCNCP A
                       WHERE     A.PLANID             = R.PLANID
                       AND       TO_CHAR(A.PROMISEDDELDATE,'IYYYIW') = R.WEEK
                       AND       A.ITEM               = R.ITEM
                       AND       NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID)   -- eStore Netting제외 20220907 추가
                       AND       A.SITEID             = R.SITEID
                       AND       A.QTYPROMISED > 0
                       ORDER BY  A.DEMANDPRIORITY, A.QTYPROMISED, A.SALESID
                     )
                 ORDER BY RK,DEMANDPRIORITY DESC, ROWNUM, QTYPROMISED ;
              
                 
               IF mydata.COUNT > 0 THEN
                   FOR j IN mydata.first .. mydata.last
                   LOOP 
                   
                        IF  MYDATA(J).ROWNUMBER = 1 THEN 
                            -- 첫행이거나 
                         
                            L_DEMANDPRIORITY := MYDATA(J).DEMANDPRIORITY;
                            
                            R_QTY      := R.NEWDP;
                            STEP_R_QTY := R.NEWDP;
                            FARE_SHARE := 0 ;
                            
                        ELSIF L_DEMANDPRIORITY <> MYDATA(J).DEMANDPRIORITY THEN
                            -- 우선순위 바뀔때 
                            
                            L_DEMANDPRIORITY := MYDATA(J).DEMANDPRIORITY;
                            
                            R_QTY := STEP_R_QTY;
                            FARE_SHARE := 0 ;
                            
                        END IF;
                        
                        IF MYDATA(J).CNT = 1 THEN
                           -- 동순위 없을때
                                                 
                            IF R_QTY >= MYDATA(J).QTYPROMISED THEN
                               --남은 수량> QTYPROMISE   작업필요없고 잔량 처리만
                               
                               UPDATE EXP_SOPROMISESRCNCP
                               SET    SOLINENUM   = SOLINENUM + 99 , 
                                      UPBY        = 'NEWALL2'
                               WHERE  SALESORDERID = MYDATA(J).SALESORDERID
                               AND    PLANID       = MYDATA(J).PLANID
                               AND    SOLINENUM    = MYDATA(J).SOLINENUM
                               AND    PROMISEDDELDATE = MYDATA(J).PROMISEDDELDATE;
                               
                               STEP_R_QTY := STEP_R_QTY - MYDATA(J).QTYPROMISED ;
                            
                            ELSE 
                                --남은 수량 < QTYPROMISE  이면 남은수량만큼 UP0DATE 해줘야 함
                               
                                INSERT INTO EXP_SOPROMISESRCNCP  --무조건 수량 상관없이 update, insert 한다. 나중에 N빵 수량 맞추려면 0이라도 row 있어야 함
                                      ( ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED, 
                                        PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, 
                                        DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, 
                                        NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, 
                                        AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE)
                                VALUES
                                      (mydata(j).ENTERPRISE, mydata(j).SOPROMISEID, mydata(j).PLANID, mydata(j).SALESORDERID, /*++*/mydata(j).SOLINENUM + 99/*++*/, mydata(j).ITEM, /*++*/R_QTY/*++*/,
    --                                  (mydata(j).ENTERPRISE, mydata(j).SOPROMISEID, mydata(j).PLANID, mydata(j).SALESORDERID, /*++*/'100'/*++*/, mydata(j).ITEM, /*++*/R_QTY/*++*/, 
                                       mydata(j).PROMISEDDELDATE, mydata(j).SITEID, mydata(j).SHIPTOID, mydata(j).SALESID, mydata(j).SALESLEVEL, mydata(j).DEMANDTYPE, mydata(j).WEEK, mydata(j).CHANNELRANK, mydata(j).CUSTOMERRANK, mydata(j).PRODUCTRANK,          
                                       mydata(j).DEMANDPRIORITY, mydata(j).TIEBREAK, mydata(j).GBM, mydata(j).GLOBALPRIORITY, mydata(j).LOCALPRIORITY, mydata(j).BUSINESSTYPE, mydata(j).ROUTING_PRIORITY,
                                       mydata(j).NO_SPLIT, mydata(j).MAP_SATISFY_SS, mydata(j).PREALLOC_ATTRIBUTE, mydata(j).BUILDAHEADTIME, mydata(j).TIMEUOM, 
                                       mydata(j).AP2ID, mydata(j).GC, mydata(j).MEASURERANK, mydata(j).PREFERENCERANK, mydata(j).INITDTTM, mydata(j).INITBY, mydata(j).UPDTTM, 'NEWINSERT1',mydata(j).REASONCODE);-- mydata(j).REASONCODE);
                                    
                                 
                                UPDATE EXP_SOPROMISESRCNCP
                                SET    QTYPROMISED = QTYPROMISED - R_QTY , 
                                       UPBY        = 'NEWUPDATE1'
                                WHERE  SALESORDERID = MYDATA(J).SALESORDERID
                                AND    PLANID       = MYDATA(J).PLANID
                                AND    SOLINENUM    = MYDATA(J).SOLINENUM
                                AND    PROMISEDDELDATE = MYDATA(J).PROMISEDDELDATE;
                                
                                STEP_R_QTY := STEP_R_QTY - R_QTY ;
                               
                            END IF;
                        
                        ELSE
                          -- 동순위 있을 때 
                            IF R_QTY >= MYDATA(J).SUBTOTAL THEN
                               --남은 수량> QTYPROMISE   작업필요없고 잔량 처리만
                               
                               UPDATE EXP_SOPROMISESRCNCP
                               SET    SOLINENUM   = SOLINENUM+99 , 
                                      UPBY       = 'NEWALL3'
                               WHERE  SALESORDERID = MYDATA(J).SALESORDERID
                               AND    PLANID       = MYDATA(J).PLANID
                               AND    SOLINENUM    = MYDATA(J).SOLINENUM
                               AND    PROMISEDDELDATE = MYDATA(J).PROMISEDDELDATE;
                               
                               STEP_R_QTY := STEP_R_QTY - MYDATA(J).QTYPROMISED ;
                            
                            ELSE 
                               --남은 수량 < QTYPROMISE  이면 남은수량 가지고 N빵 처리 해줘야 함
                               INSERT INTO EXP_SOPROMISESRCNCP  --무조건 수량 상관없이 update, insert 한다. 나중에 N빵 수량 맞추려면 0이라도 row 있어야 함
                                      ( ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED, 
                                        PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, 
                                        DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, 
                                        NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, 
                                        AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE)
                                VALUES
                                      (mydata(j).ENTERPRISE, mydata(j).SOPROMISEID, mydata(j).PLANID, mydata(j).SALESORDERID, /*++*/mydata(j).SOLINENUM + 99/*++*/, mydata(j).ITEM, /*++*/TRUNC(R_QTY * MYDATA(J).RATIO)/*++*/,
    --                                  (mydata(j).ENTERPRISE, mydata(j).SOPROMISEID, mydata(j).PLANID, mydata(j).SALESORDERID, /*++*/'100'/*++*/, mydata(j).ITEM, /*++*/TRUNC(R_QTY * MYDATA(J).RATIO)/*++*/, 
                                       mydata(j).PROMISEDDELDATE, mydata(j).SITEID, mydata(j).SHIPTOID, mydata(j).SALESID, mydata(j).SALESLEVEL, mydata(j).DEMANDTYPE, mydata(j).WEEK, mydata(j).CHANNELRANK, mydata(j).CUSTOMERRANK, mydata(j).PRODUCTRANK,          
                                       mydata(j).DEMANDPRIORITY, mydata(j).TIEBREAK, mydata(j).GBM, mydata(j).GLOBALPRIORITY, mydata(j).LOCALPRIORITY, mydata(j).BUSINESSTYPE, mydata(j).ROUTING_PRIORITY,
                                       mydata(j).NO_SPLIT, mydata(j).MAP_SATISFY_SS, mydata(j).PREALLOC_ATTRIBUTE, mydata(j).BUILDAHEADTIME, mydata(j).TIMEUOM, 
                                       mydata(j).AP2ID, mydata(j).GC, mydata(j).MEASURERANK, mydata(j).PREFERENCERANK, mydata(j).INITDTTM, mydata(j).INITBY, mydata(j).UPDTTM, 'NEWINSERT2', mydata(j).REASONCODE);-- mydata(j).REASONCODE);
                                     
                               
                               UPDATE EXP_SOPROMISESRCNCP
                               SET    QTYPROMISED = QTYPROMISED - TRUNC(R_QTY * MYDATA(J).RATIO) , 
                                      UPBY        = 'NEWUPDATE2'
                               WHERE  SALESORDERID = MYDATA(J).SALESORDERID
                               AND    PLANID       = MYDATA(J).PLANID
                               AND    SOLINENUM    = MYDATA(J).SOLINENUM
                               AND    PROMISEDDELDATE = MYDATA(J).PROMISEDDELDATE;
                                
                               STEP_R_QTY := STEP_R_QTY - TRUNC(R_QTY * MYDATA(J).RATIO) ;
                               FARE_SHARE := FARE_SHARE + TRUNC(R_QTY * MYDATA(J).RATIO) ;
                               
                               --N빵처리후 마지막 ROW에서 남는 수량 1,2,3..FARE SHARE를 위해 다시한번
                               IF MYDATA(J).CHRK = 1 AND  R_QTY > FARE_SHARE THEN
                                
                                    SELECT RANK() OVER  (ORDER BY RK, ROWNUM ) ROWNUMBER, RK, RANK() OVER (PARTITION BY RK ORDER BY ROWNUM DESC , QTYPROMISED ) CHRK,
                                       ROWIDs,
                                       COUNT(*) OVER (PARTITION BY RK ORDER BY DEMANDPRIORITY DESC ,RK RANGE UNBOUNDED PRECEDING) AS CNT,
                                       SUM(QTYPROMISED) OVER (PARTITION BY RK ORDER BY DEMANDPRIORITY DESC,RK RANGE UNBOUNDED PRECEDING) AS SUBTOTAL,
                                       NVL(RATIO_TO_REPORT(QTYPROMISED) OVER (PARTITION BY DEMANDPRIORITY ,RK),0) RATIO,
                                       ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, 
                                       QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, 
                                       DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, 
                                       GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, 
                                       NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, 
                                       INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE
                                     BULK COLLECT INTO mydata_T
                                     FROM ( 
                                           WITH DATA AS
                                           ( SELECT PLANID, SALESORDERID, ITEM, SITEID,  SUM(QTYPROMISED) QTYPROMISED 
                                             FROM BUF_SOPROMISESRCNCP_POST
                                             WHERE PLAN    = 'NEWFCST'
                                             AND   PLANID  = R.PLANID
                                             AND   ITEM    = R.ITEM
                                             AND   SITEID  = R.SITEID
                                             AND   TO_CHAR(PROMISEDDELDATE,'IYYYIW') = R.WEEK
                                             GROUP BY PLANID, SALESORDERID, ITEM, SITEID
                                            )
                                               SELECT   RANK() OVER (PARTITION BY A.ITEM,A.SITEID ORDER BY A.DEMANDPRIORITY DESC) RK,
                                                         A.ROWID ROWIDS,
                                                         SUM(A.QTYPROMISED) OVER (PARTITION BY A.SALESORDERID) PROD, --N빵직전 INS NETTING후
                                                         SUM(B.QTYPROMISED) OVER (PARTITION BY A.SALESORDERID) DEV,  --INS NETTING전
                                                         A.ENTERPRISE , A.SOPROMISEID, A.PLANID, A.SALESORDERID, A.SOLINENUM,    
                                                         A.ITEM ,A.QTYPROMISED ,A.PROMISEDDELDATE ,A.SITEID ,A.SHIPTOID ,A.SALESID,        
                                                         A.SALESLEVEL, A.DEMANDTYPE, A.WEEK, A.CHANNELRANK ,A.CUSTOMERRANK ,A.PRODUCTRANK ,A.DEMANDPRIORITY ,A.TIEBREAK ,A.GBM ,A.GLOBALPRIORITY ,A.LOCALPRIORITY,    
                                                         A.BUSINESSTYPE ,A.ROUTING_PRIORITY ,A.NO_SPLIT ,A.MAP_SATISFY_SS ,A.PREALLOC_ATTRIBUTE ,A.BUILDAHEADTIME ,A.TIMEUOM,        
                                                         A.AP2ID ,A.GC ,A.MEASURERANK ,A.PREFERENCERANK ,A.INITDTTM ,A.INITBY ,A.UPDTTM ,A.UPBY ,A.REASONCODE    
                                               FROM      EXP_SOPROMISESRCNCP A, data B
                                               WHERE     A.PLANID             = R.PLANID
                                               AND       TO_CHAR(A.PROMISEDDELDATE,'IYYYIW') = R.WEEK
                                               AND       A.ITEM               = R.ITEM
                                               AND       A.SITEID             = R.SITEID
                                               AND       A.SOLINENUM          >= 100
                                               AND       A.UPBY               = 'NEWINSERT2'  -- RATIO별로 FIX된 아이들 중
                                               AND       A.PLANID             = B.PLANID
                                               AND       A.SALESORDERID       = B.SALESORDERID
        --                                       AND       A.SOLINENUM          = B.SOLINENUM -- 쪼개진 데이터(100)가 생기므로 SOLINENUM 못건다...
                                               AND       A.ITEM               = B.ITEM
                                               AND       NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID AND SALESID = A.SALESID)   -- eStore Netting제외 20220907 추가
                                               AND       A.SITEID             = B.SITEID  
                                               ORDER BY  A.DEMANDPRIORITY DESC, A.QTYPROMISED, A.SALESID
                                         )
                                     WHERE  DEV > PROD  --NETTING 전보다 데이터가 작아야만 PLUS 해도 상관없음, 같은거는 PLUS 되면 처음 DEMAND보다 많아지므로 안됨
                                     ORDER BY RK, DEMANDPRIORITY , ROWNUM, QTYPROMISED;
                 
                                       IF mydata_T.COUNT > 0 THEN
                                           FOR k IN mydata_T.first .. (R_QTY - FARE_SHARE )
                                           LOOP
                                           
                                            UPDATE EXP_SOPROMISESRCNCP
                                            SET    QTYPROMISED = QTYPROMISED +1
                                                   , UPBY = UPBY||'_PLUS'
                                            WHERE  SALESORDERID = mydata_T(K).SALESORDERID
                                            AND    PLANID       = MYDATA_T(K).PLANID
                                            AND    SOLINENUM    = MYDATA_T(K).SOLINENUM
                                            AND    ROWNUM = 1;
                                            
                                            
                                            UPDATE EXP_SOPROMISESRCNCP
                                            SET    QTYPROMISED = QTYPROMISED -1
                                                   , UPBY      = UPBY||'_MINUS'
                                            WHERE  SALESORDERID = mydata_T(K).SALESORDERID
                                            AND    PLANID       = MYDATA_T(K).PLANID
                                            AND    SOLINENUM    < 100
                                            --VPS 통합물류 때문에 SALESORDERID는 같은데 SOLINEUM이 두개 이상 될 경우 두개다 -1 처리 되므로
                                            --PREFRENCERANK 가 같은 DEMAND에 대해 -1 처리
                                            and    PREFERENCERANK =mydata_T(K).PREFERENCERANK
                                            AND    QTYPROMISED > 0 
                                            AND    ROWNUM = 1;                                        
    --                                        AND    SOLINENUM    = 1;
                                           
                                           END LOOP;
                                       END IF;
                                       
                                       
                                       --마지막 N빵 처리하면 잔량이 0 되어야 
                                      STEP_R_QTY := 0; 
                                
                               
                               END IF;
                               
                            END IF;
                        
                        
                        END IF;
                        
                     
                   
                   
                   END LOOP;
               END IF;
                
                
            
                COMMIT;
                
            END IF;
            
         
         END LOOP;
         CLOSE CUR_OPENNETTING;
         
     EXCEPTION
            WHEN NO_DATA_FOUND THEN
                DBMS_OUTPUT.PUT_LINE('Error: check '||SQLCODE||'_'||SQLERRM);
                DBMS_OUTPUT.PUT_LINE('ERR   R.WEEK : '||R.WEEK );
                DBMS_OUTPUT.PUT_LINE('ERR   R.ITEM : '||R.ITEM );
                DBMS_OUTPUT.PUT_LINE('ERR   R.SITEID : '||R.SITEID );
                ROLLBACK;
            WHEN OTHERS THEN
                DBMS_OUTPUT.PUT_LINE('Error: check '||SQLCODE||'_'||SQLERRM);
                DBMS_OUTPUT.PUT_LINE('ERR   R.WEEK : '||R.WEEK );
                DBMS_OUTPUT.PUT_LINE('ERR   R.ITEM : '||R.ITEM );
                DBMS_OUTPUT.PUT_LINE('ERR   R.SITEID : '||R.SITEID );
                ROLLBACK;
     END;

     COMMIT;
     
     
     --필요없는 row는 날린다..
     BEGIN
         DELETE FROM EXP_SOPROMISESRCNCP
         WHERE  PLANID      = V_PLANID
         AND    W(PROMISEDDELDATE) BETWEEN V_WEEK1 AND V_WEEK4
         AND    UPBY        LIKE 'NEW%'
         AND    QTYPROMISED = 0 ;
     EXCEPTION
        WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('22222' );
            NULL;
     END;
     
     -- 우선순위는 4주이내 mpdp에 대해서 DEMAND TYPE PRIORITY = 8 만들어줌, 나머진 기존값
     -- NEW FCST는 SOLINENUM >= 100 인 것으로 구분한다.
     BEGIN
     
         UPDATE EXP_SOPROMISESRCNCP
         SET    DEMANDPRIORITY = SUBSTR(DEMANDPRIORITY, 1,3)||'8'||SUBSTR(DEMANDPRIORITY, 5,4)
         WHERE  PLANID         = V_PLANID
--         AND    W(PROMISEDDELDATE) BETWEEN V_WEEK1 AND V_WEEK4
-- 브라질 W5 NEW FCST 처리로 인해 W5도 8처리
         AND    SOLINENUM >= 100
         AND    SALESORDERID NOT LIKE 'UNF%'; --220413 UNF는 TYPE RANK가 9순위로 최 후순위어야 함 
         
     EXCEPTION
        WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('33333' );
            NULL;
     END;
     
     COMMIT;
     
     DBMS_OUTPUT.PUT_LINE('END : '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss') );
     -- 1. NEW FCST 수량이 제대로 N빵되어 EXP_SOPROMISESRCNCP에 들어갔는지 확인
     BEGIN
         SELECT COUNT(*) INTO SMS_QTYCOUNT 
         FROM(
             SELECT A.ITEM, A.SITEID, A.WEEK, SUM(A.PROD) , SUM(A.DEV) 
             FROM(
                select siteid, item, TO_CHAR(promiseddeldate,'IYYYIW') WEEK, 
                       qtypromised prod , 0 dev  
                from   EXP_SOPROMISESRCNCP
                where  solinenum >=100
                union all
                select siteid, item, WEEK, 0 prod , NEWDP dev 
                from   MST_NEWFCSTNETTING
                where  planid = V_PLANID
                ) A
                WHERE A.SITEID NOT IN (SELECT 'S341' FROM DUAL
                                     UNION ALL
                                     SELECT 'S341WC74' FROM DUAL
                                     UNION ALL
                                     SELECT 'TOTAL' FROM DUAL)
                AND NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND  SITEID = A.SITEID) 
                AND NOT EXISTS (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
                GROUP BY A.ITEM, A.SITEID, A.WEEK
                HAVING SUM(A.PROD)<> SUM(A.DEV)
              );
         
     EXCEPTION
        WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('44444' );
            NULL;
     END;
     
     IF SMS_QTYCOUNT > 0 THEN
        DBMS_OUTPUT.PUT_LINE('Error: check MODELQTY');
     END IF;
     
    
     DBMS_OUTPUT.PUT_LINE('END : '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss') );
     -- 페어쉐어 하면서 (-)나오면 에러 발생
     BEGIN
         SELECT COUNT(*) INTO SMS_COUNT 
         FROM   EXP_SOPROMISESRCNCP
         WHERE  PLANID   = V_PLANID
         AND    QTYPROMISED < 0;
         
     EXCEPTION
        WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('55555' );
            NULL;
     END;
     
     IF SMS_COUNT > 0 THEN
        DBMS_OUTPUT.PUT_LINE('Error: check PAIRSHARE');
     END IF;
     
     
     
     --2. NEW FCST 로직전과 수량비교
     BEGIN
         SELECT SUM(QTYPROMISED)  INTO SMS_QTYBEFORE 
         FROM   BUF_SOPROMISESRCNCP_POST
         WHERE  PLAN     = 'NEWFCST'
         AND    PLANID   = V_PLANID;
         
         
         
         SELECT SUM(QTYPROMISED)  INTO SMS_QTYAFTER 
         FROM   EXP_SOPROMISESRCNCP
         WHERE  PLANID   = V_PLANID;
         
         
     EXCEPTION
        WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('66666' );
            NULL;
     END;
     
     IF SMS_QTYBEFORE != SMS_QTYAFTER THEN
        DBMS_OUTPUT.PUT_LINE('Error: check NEWFCSTERROR');
     END IF;
     
     V_LOGMESSAGE := V_LOGMESSAGE|| 'ERROR  COUNT : '||0; -- LOGMESSAGE
     
     SP_LOGGING_REG(V_LOGSEQ, V_PGMNAME,V_STARTDATE,V_ERRORCNT,V_LOGMESSAGE);
     
     
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('Error: check '||SQLCODE||'_'||SQLERRM);
        ROLLBACK;
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error: check '||SQLCODE||'_'||SQLERRM);
        ROLLBACK;
END;
