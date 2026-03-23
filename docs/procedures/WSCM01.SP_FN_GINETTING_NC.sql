CREATE OR REPLACE PROCEDURE WSCM01.SP_FN_GINETTING_NC (IN_TYPE VARCHAR2 DEFAULT 'NCP') IS
/******************************************************************************
   NAME:       WSCM01.SP_FN_GINETTING_NC
   PURPOSE:    가용량네팅
   REVISIONS:
   VER     DATE        AUTHOR   DESCRIPTION
   ------- ----------  -------  ------------------------------------
   1.0     2010.12.06  HK       신규 버전
   1.1     2016.06.19  EUNJU    POSTPONEMENT 가용량 통합 적용
 ******************************************************************************/
 
    V_PGMNAME               VARCHAR2(30) := 'SP_FN_GINETTING_NC';
    V_LOADINGTIME           VARCHAR2(2);
    V_RETSQLERRM            VARCHAR2(2000);
    
    
    --V_PLANID              MST_PLAN.PLANID %TYPE;
    V_PLANID                EXP_SOPROMISESRCNCP.PLANID%TYPE; 
    V_PLANWEEK              MST_PLAN.PLANWEEK%TYPE;
    V_EFFSTARTDATE          MST_PLAN.EFFSTARTDATE %TYPE;
    V_EFFENDDATE            MST_PLAN.EFFENDDATE %TYPE;
    V_TYPE                  MST_PLAN.TYPE %TYPE;
   
    V_PREPLANID             EXP_SOPROMISESRCNCP.PLANID%TYPE; 
    V_PREYEAR               MST_WEEK.YEAR%TYPE;
    V_PREYWEEK              MST_WEEK.WEEK%TYPE;
    V_WEEK1                 MST_WEEK.WEEK%TYPE; 
    V_WEEK4                 MST_WEEK.WEEK%TYPE;   
    
    
    C_ROWIDS                ROWID;
    C_RK                    NUMBER;

    C_PLANID                MST_GINETTING.PLANID%TYPE;
    C_SITEID                MST_GINETTING.SITEID%TYPE;
    C_ITEM                  MST_GINETTING.ITEM%TYPE;             
    C_WEEK                  MST_GINETTING.WEEK%TYPE;
    C_EXCEPTQ               MST_GINETTING.EXCEPTQ%TYPE;
    C_AVAIL                 MST_GINETTING.AVAIL%TYPE;                             
    C_DP                    MST_GINETTING.DP%TYPE;
    C_CUREXCEPT             MST_GINETTING.CUREXCEPT%TYPE;
    C_CURAVAIL              MST_GINETTING.CURAVAIL%TYPE;
   

    M_RK                    NUMBER :=0;

    M_AVAIL                 NUMBER :=0;
    M_EXCEPT                NUMBER :=0;
    V_REXCEPT               NUMBER :=0;
    
    V_LOGMESSAGE        LONG;
    V_ERRMESSAGE        VARCHAR2(2000);
    V_LOGSEQ            NUMBER(10);
    V_ERRORCNT          INTEGER:=0;

    V_STARTDATE         DATE;
    D_SQL               VARCHAR(30000);
    
    SMS_COUNT                NUMBER := 0;
    SMS_QTYCOUNT             NUMBER := 0;
    SMS_QTYBEFORE            NUMBER := 0;
    SMS_QTYAFTER             NUMBER := 0;
    SMS_QTYTOTAL             NUMBER := 0;
    
    V_SALESID                MST_GINETTING_SALES.SALESID%TYPE;
    V_EXCEPTQ                NUMBER:=0;
    V_REMAINGIDP             NUMBER:=0;
    V_GIDP                   NUMBER:=0;
    
        
BEGIN
        
     SELECT SEQ_LOG.NEXTVAL INTO V_LOGSEQ FROM DUAL;
     SELECT SYSDATE INTO V_STARTDATE FROM DUAL;
     
      --! GET CURRENT RUNNING PLANID !--
      if in_type = 'NCP' then
          --! NCP 일 경우 PLANID !--
          select planid, effstartdate, effenddate, planweek, type
          into   v_planid, v_effstartdate, v_effenddate, v_planweek, v_type
          from   mst_plan 
          where  isrunning = 'Y'
          and    type = 'NCP';
      else
          --! NCP 아닐 경우 PLANID !--  
          select planid, effstartdate, effenddate, planweek, type
          into   v_planid, v_effstartdate, v_effenddate, v_planweek, v_type
          from   mst_plan 
          where  isrunning = 'Y'
          and    type != 'NCP';
      end if;    

     

     IF V_TYPE = 'VPLAN' THEN    
        --! Vplan 인 경우 당주 GI 사용 !--
         V_PREPLANID := TO_CHAR(V_EFFSTARTDATE, 'IYYYIW');
         V_PREYEAR   := SUBSTR(TO_CHAR(V_EFFSTARTDATE, 'IYYYIW'),0,4);
         V_PREYWEEK  := SUBSTR(TO_CHAR(V_EFFSTARTDATE, 'IYYYIW'),5,2);
         V_WEEK4     := TO_CHAR(V_EFFSTARTDATE + 7*4, 'IYYYIW');    --2018.12.03 수정(V PLAN일 경우 당주 기준으로 변경)
     ELSE
         --! Vplan 아닐 경우 전주 GI 사용 !--
         V_PREPLANID := TO_CHAR(V_EFFSTARTDATE - 7*1, 'IYYYIW');
         V_PREYEAR   := SUBSTR(TO_CHAR(V_EFFSTARTDATE - 7*1, 'IYYYIW'),0,4);
         V_PREYWEEK  := SUBSTR(TO_CHAR(V_EFFSTARTDATE - 7*1, 'IYYYIW'),5,2);
         V_WEEK4     := TO_CHAR(V_EFFSTARTDATE + 7*3, 'IYYYIW');    --2018.12.03 수정(V PLAN일 경우 당주 기준으로 변경)
     END IF;     
     
     V_WEEK1     := V_PLANWEEK;
--     V_WEEK4     := TO_CHAR(V_EFFSTARTDATE + 7*3, 'IYYYIW');    --2018.12.03 수정(V PLAN일 경우 당주 기준으로 변경)

     DBMS_OUTPUT.PUT_LINE('V_PLANID:'||V_PLANID ||'  V_EFFSTARTDATE :'||V_EFFSTARTDATE);
     DBMS_OUTPUT.PUT_LINE('V_PREYWEEK:'||V_PREYWEEK ||'  V_WEEK1:'||V_WEEK1||'  V_WEEK4 :'||V_WEEK4);     
     DBMS_OUTPUT.PUT_LINE('START : '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));
    
    --! 백업 테이블 TURNCATE !--
    EXECUTE IMMEDIATE 'ALTER TABLE BUF_SOPROMISESRCNCP_POST TRUNCATE PARTITION PDPDEC';
      
    --! 백업 테이블에 INSERT !--
    INSERT INTO BUF_SOPROMISESRCNCP_POST --SALESID(AP2ID)사용할수도 있으므로,,,일단 놔둠, 백업용 테이블
          ( PLAN, ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED, PROMISEDDELDATE
           , SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK
           , GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS
           , PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK
           , INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE)
         SELECT    'DPDEC', ENTERPRISE, SOPROMISEID, A.PLANID, SALESORDERID, SOLINENUM, A.ITEM, A.QTYPROMISED, PROMISEDDELDATE
                 , A.SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, A.WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY
                 , TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT
                 , MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK
                 , INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE
          FROM     EXP_SOPROMISESRCNCP A
          WHERE    1=1  --작업전 백업용이므로 해당 조건 삭제 전체 백업 필요 _20220905
--          NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND SITEID = A.SITEID)          
          AND      A.PLANID  =  V_PLANID;
          
     COMMIT;  

     DBMS_OUTPUT.PUT_LINE('1.DATA SUMMARY '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));
    
     --! GI NETTING 계산 과정 테이블 DELETE !--   
     DELETE FROM MST_GINETTING
     WHERE  PLANID = V_PLANID;
     
     
     --! GI NETTING 계산 과정 테이블에 가용량 + DEMAND INSERT !--
      D_SQL:='
      INSERT INTO MST_GINETTING
      (PLANID, SITEID, ITEM, WEEK, RTF, GI, EXCEPTQ, AVAIL, DP, CUREXCEPT, CURAVAIL, INITDTTM, INITBY)
      WITH EXCEPT AS
      (
         SELECT ITEM, SITEID,-- AP2ID, 
                '||V_PLANWEEK||' WEEK,
                SUM(DECODE(CATEGORY, ''02RTF'', WEEK'||V_PREYWEEK||',0)) RTFQTY,  
                SUM(DECODE(CATEGORY, ''30GI'' , WEEK'||V_PREYWEEK||',0)) GIQTY,
                SUM(DECODE(CATEGORY, ''30GI'' , WEEK'||V_PREYWEEK||',0)) - SUM(DECODE(CATEGORY, ''02RTF'', WEEK'||V_PREYWEEK||',0)) EXCEPT  
         FROM   GUI_SALESRESULTSMM A
         WHERE  YEAR = '||V_PREYEAR||'
         AND    CATEGORY IN (''02RTF'',''30GI'')
         AND    NOT EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND SITEID = A.SITEID)
         AND    NOT EXISTS (SELECT ''X'' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = ''Y'') --E-STORE 20200807 추가
         --20220919 estore 제품 후처리 제외시, 계산에는 발췌해서 계산 후 분배시에 제외 하도록 조건 제거  req msjd95
--         AND    NOT EXISTS (SELECT ''X'' FROM MVES_ITEMSITE_NC WHERE ITEM = A.ITEM AND SALESID = A.SALESID AND SITEID = A.SITEID)   --ESTORE 후보충 대상 제외 20210708
         GROUP BY ITEM, SITEID--, AP2ID
         HAVING SUM(DECODE(CATEGORY, ''02RTF'', WEEK'||V_PREYWEEK||',0)) < SUM(DECODE(CATEGORY, ''30GI'' , WEEK'||V_PREYWEEK||',0))
      )   
      , AVAIL AS(
        SELECT ITEM, SITEID, WEEK, SUM(AVAILQTY) AVAILQTY, SUM(DPQTY) DPQTY
        FROM(
            -- 1. 가용량 (INVENTORY + INTRANSIT + DISTRIBUTIONORDERS)
            SELECT A.ITEM, A.SITEID, A.WEEK, NVL(A.QTY,0) AVAILQTY, 0 DPQTY
            FROM MST_INVENTORY_DNE A
            WHERE  A.PLANID   = '''||V_PLANID||'''
            AND    A.WEEK BETWEEN '||V_WEEK1||' AND '||V_WEEK4||'
            AND    NOT EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.SITEID)
            -- 2. DEMAND
            UNION ALL
            SELECT  ITEM, SITEID, TO_CHAR(PROMISEDDELDATE, ''IYYYIW'') WEEK, 0, 
                    QTYPROMISED DPQTY
            FROM    EXP_SOPROMISESRCNCP A
            WHERE   1=1
            AND     QTYPROMISED > 0 --PRE ALLOC 대상아닌..
            AND     TO_CHAR(PROMISEDDELDATE, ''IYYYIW'') BETWEEN '||V_WEEK1||' AND '||V_WEEK4||'
            AND     NOT EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE ITEM = A.ITEM AND SITEID = A.SITEID)   
       ) A
       GROUP BY ITEM, SITEID, WEEK
     )
     SELECT '''||V_PLANID||''' PLANID, A.SITEID, A.ITEM, B.WEEK, 
            A.RTFQTY, A.GIQTY, A.EXCEPT, B.AVAILQTY, B.DPQTY , A.EXCEPT, B.AVAILQTY, SYSDATE, ''INITINSERT''
     FROM   EXCEPT A, AVAIL B
     WHERE  A.ITEM   = B.ITEM
     AND    A.SITEID = B.SITEID'; 
            
     
     DBMS_OUTPUT.PUT_LINE(D_SQL);
     EXECUTE IMMEDIATE D_SQL;
     
      
     -- ! SALESID별 RTF EXP DELETE !--
     DELETE FROM MST_GINETTING_SALES
     WHERE PLANID = V_PLANID;
     
     --salesid별 기여도 판변을 위해 insert 추가  20220919 
     --! SALESID별 RTF EXP 발췌 !--
     D_SQL :='
         INSERT INTO MST_GINETTING_SALES
         (PLANID, SALESID, SITEID, ITEM, WEEK, RTF, GI, EXCEPTQ, SUMEXCEPTQ, RATIOEXCEPTQ, RNK, DP, CUREXCEPT ,INITDTTM, INITBY)
         WITH GI AS(   
             SELECT '''||V_PLANID||''' PLANID, SALESID, SITEID, ITEM, WEEK, RTF, GI, EXCEPTQ, 
                    SUM(EXCEPTQ) OVER (PARTITION BY ITEM, SITEID, WEEK) SUMEXCEPTQ,           
                    RATIO_TO_REPORT(EXCEPTQ) OVER (PARTITION BY ITEM, SITEID, WEEK ) RATIOEXCEPTQ,
                    RANK() OVER (PARTITION BY ITEM, SITEID, WEEK ORDER BY EXCEPTQ DESC) RNK
             FROM(
                SELECT ITEM, SALESID, SITEID,-- AP2ID, 
                       '''||V_PREPLANID||''' WEEK,
                       SUM(DECODE(CATEGORY, ''02RTF'', WEEK'||V_PREYWEEK||',0)) RTF,  
                       SUM(DECODE(CATEGORY, ''30GI'' , WEEK'||V_PREYWEEK||',0)) GI,
                       SUM(DECODE(CATEGORY, ''30GI'' , WEEK'||V_PREYWEEK||',0)) - SUM(DECODE(CATEGORY, ''02RTF'', WEEK'||V_PREYWEEK||',0)) EXCEPTQ  
                FROM   GUI_SALESRESULTSMM A
                WHERE  YEAR = '||V_PREYEAR||'
                AND    CATEGORY IN (''02RTF'',''30GI'')
    --            AND    NOT EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND SITEID = A.SITEID) --_sell에서도 같은 테이블 바라보고 NETTING 할거라 조건 삭제 20220905 
                AND    NOT EXISTS (SELECT ''X'' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = ''Y'') --E-STORE 20200807 추가
                --20220919 estore 제품 후처리 제외시, 계산에는 발췌해서 계산 후 분배시에 제외 하도록 조건 제거  req msjd95
--                AND    NOT EXISTS (SELECT ''X'' FROM MVES_ITEMSITE_NC WHERE ITEM = A.ITEM AND SALESID = A.SALESID AND SITEID = A.SITEID)   --ESTORE 후보충 대상 제외 20210708
                GROUP BY ITEM, SALESID, SITEID--, AP2ID
                HAVING SUM(DECODE(CATEGORY, ''02RTF'', WEEK'||V_PREYWEEK||',0)) < SUM(DECODE(CATEGORY, ''30GI'' , WEEK'||V_PREYWEEK||',0))
                AND    SUM(DECODE(CATEGORY, ''02RTF'', WEEK'||V_PREYWEEK||',0)) >=0 -- 가격재무 - 값 RTF AMT 제외 
                )
         )
--        , DP AS(
--        SELECT A.PLANID, A.SALESID, A.SITEID, A.ITEM, TO_CHAR(A.PROMISEDDELDATE, ''IYYYIW'') DPWEEK, SUM(A.QTYPROMISED) QTY 
--        FROM   EXP_SOPROMISESRCNCP A, GI B
--        WHERE  A.PLANID = '''||V_PLANID||'''
--        AND    A.SITEID = B.SITEID
--        AND    A.ITEM = B.ITEM
--        AND    A.SALESID = B.SALESID
--        AND    A.QTYPROMISED>0
--        AND    TO_CHAR(A.PROMISEDDELDATE, ''IYYYIW'') BETWEEN '''||V_WEEK1||''' AND '''||V_WEEK4||'''
----        AND    NOT EXISTS (SELECT ''X'' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND SITEID = A.SITEID)
--        AND    NOT EXISTS (SELECT ''X'' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = ''Y'') --E-STORE 20200807 추가
--        AND    NOT EXISTS (SELECT ''X'' FROM MVES_ITEMSITE_NC WHERE ITEM = A.ITEM AND SALESID = A.SALESID AND SITEID = A.SITEID)   --ESTORE 후보충 대상 제외 20210708
--        GROUP BY A.PLANID, A.SITEID, A.ITEM, A.SALESID,TO_CHAR(A.PROMISEDDELDATE, ''IYYYIW'')
--        )
        SELECT PLANID, SALESID, SITEID ,ITEM, WEEK, RTF, GI, EXCEPTQ, SUMEXCEPTQ, RATIOEXCEPTQ, RNK, 0 DP ,0, SYSDATE, ''SP_FN_GINETTING''
        FROM   GI 
--        UNION ALL
--        SELECT PLANID, SALESID, SITEID, ITEM, DPWEEK, 0, 0, 0, 0, 0, 0, QTY, SYSDATE, ''SP_FN_GINETTING''
--        FROM   DP 
        ';
     
     DBMS_OUTPUT.PUT_LINE(D_SQL);
     EXECUTE IMMEDIATE D_SQL;
     
     
     COMMIT;
     
     DBMS_OUTPUT.PUT_LINE('2.AVAIL QTY ROLLING '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));
     --! DP와비교하여 AVAIL수량은 남는 가용량에 대해 뒷주차로 ROLLING 한다 !--
     FOR X IN(    
        SELECT RANK() OVER(PARTITION BY ITEM, SITEID  ORDER BY WEEK ) RK,
               PLANID, ITEM, SITEID, WEEK, EXCEPTQ, AVAIL, DP, CURAVAIL
        FROM   MST_GINETTING
        WHERE  PLANID = V_PLANID
        ORDER BY SITEID, ITEM, WEEK
     )
     LOOP
        --! 첫 행이면 !--
        IF X.RK = 1 THEN
            --! X.AVAIL > X.DP 이면 !--
            IF X.AVAIL > X.DP THEN 
                M_AVAIL := X.AVAIL - X.DP;
            ELSE M_AVAIL := 0 ;
            END IF;
        --! 첫 행이 아니면 !--
        ELSE
            --! CURAVAIL = AVAIL + M_AVAIL !--
            UPDATE MST_GINETTING
            SET    CURAVAIL = AVAIL + M_AVAIL
            WHERE  PLANID   = X.PLANID
            AND    ITEM     = X.ITEM
            AND    SITEID   = X.SITEID
            AND    WEEK     = X.WEEK;
            
            
            IF X.AVAIL + M_AVAIL > X.DP THEN 
                M_AVAIL := (X.AVAIL + M_AVAIL) - X.DP;
            ELSE M_AVAIL := 0 ;
            END IF;
            
            
        END IF;
        
     END LOOP;
         
     
     COMMIT;
     
     DBMS_OUTPUT.PUT_LINE('3.GINETTING OPEN '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));
     
     FOR X IN(    
        SELECT RANK() OVER(PARTITION BY ITEM, SITEID  ORDER BY WEEK ) RK,
               PLANID, ITEM, SITEID, WEEK, EXCEPTQ, DP, CURAVAIL ,  CUREXCEPT
        FROM   MST_GINETTING
        WHERE  PLANID = V_PLANID
        ORDER BY SITEID, ITEM, WEEK
     )
     LOOP
     
        IF X.RK = 1 THEN
            
            --! AVAIL 수량 > DP 일 경우 !--
            IF X.CURAVAIL - X.DP >=0 THEN
            
                --! 전량 MPDP, GIDP 0 !--
                UPDATE MST_GINETTING
                SET    GIDP =  0,
                       MPDP =  DP
                WHERE  PLANID = X.PLANID
                AND    ITEM   = X.ITEM
                AND    SITEID = X.SITEID
                AND    WEEK   = X.WEEK;
                
                M_EXCEPT := X.EXCEPTQ;
            --! AVAIL 수량보다 DP가 넘어선 수량에 대해서만 GINETTING !--
            ELSE 
                    --! DP 가 남았을 경우 !--
                    IF X.EXCEPTQ >= X.DP - X.CURAVAIL THEN
                      
                       --! 가용량이 없는 만큼은 GIDP, 가용량이 있는 만큼은 MPDP !--
                       UPDATE MST_GINETTING
                       SET    GIDP =  X.DP - X.CURAVAIL,
                              MPDP =  DP - (X.DP - X.CURAVAIL)
                       WHERE  PLANID = X.PLANID
                       AND    ITEM   = X.ITEM
                       AND    SITEID = X.SITEID
                       AND    WEEK   = X.WEEK;
                       
                        M_EXCEPT := X.EXCEPTQ -(X.DP-X.CURAVAIL);
                       
                    --! X.EXCEPT< X.DP - X.CURAVAIL !--
                    ELSE  
                       
                        --!  GIDP = X.EXCEPTQ , MPDP =  DP - (X.EXCEPTQ) !--
                       UPDATE MST_GINETTING
                       SET    GIDP =  X.EXCEPTQ,
                              MPDP =  DP - (X.EXCEPTQ)
                       WHERE  PLANID = X.PLANID
                       AND    ITEM   = X.ITEM
                       AND    SITEID = X.SITEID
                       AND    WEEK   = X.WEEK;
                       
                       
                       M_EXCEPT := 0;
                        
                    END IF; 
                  
            END IF; 
        
        --! 두번째 ROW부터 !--   
        ELSE  
            --! AVAIL 수량 > DP 일 경우 !--
            IF X.CURAVAIL - X.DP >=0 THEN
            
                --! 전량 MPDP, GIDP 0 !--
                UPDATE MST_GINETTING
                SET    GIDP =  0,
                       MPDP =  DP,
                       CUREXCEPT = M_EXCEPT
                WHERE  PLANID = X.PLANID
                AND    ITEM   = X.ITEM
                AND    SITEID = X.SITEID
                AND    WEEK   = X.WEEK;
                
                
                M_EXCEPT := M_EXCEPT;--X.CUREXCEPT;
            --! AVAIL 수량보다 DP가 넘어선 수량에 대해서만 GINETTING !--  
            ELSE 
                --! CUREXCEPT = M_EXCEPT !--
                UPDATE MST_GINETTING
                SET    CUREXCEPT = M_EXCEPT
                WHERE  PLANID = X.PLANID
                AND    ITEM   = X.ITEM
                AND    SITEID = X.SITEID
                AND    WEEK   = X.WEEK;
                
--                IF X.DP > X.CURAVAIL THEN
                    --! DP 가 남았을 경우 !--
                    IF M_EXCEPT >= X.DP - X.CURAVAIL THEN
                       
                        --!  GIDP = X.DP - X.CURAVAIL , DP - (X.DP - X.CURAVAIL) !--
                       UPDATE MST_GINETTING
                       SET    GIDP =  X.DP - X.CURAVAIL,
                              MPDP =  DP - (X.DP - X.CURAVAIL)
                       WHERE  PLANID = X.PLANID
                       AND    ITEM   = X.ITEM
                       AND    SITEID = X.SITEID
                       AND    WEEK   = X.WEEK;
                       
                       M_EXCEPT := M_EXCEPT -(X.DP-X.CURAVAIL);
                      
                    --! X.EXCEPT< X.DP - X.CURAVAIL !--
                    ELSE
                       --! GIDP =  M_EXCEPT,  MPDP =  DP - (M_EXCEPT) !--
                       UPDATE MST_GINETTING
                       SET    GIDP =  M_EXCEPT,
                              MPDP =  DP - (M_EXCEPT)
                       WHERE  PLANID = X.PLANID
                       AND    ITEM   = X.ITEM
                       AND    SITEID = X.SITEID
                       AND    WEEK   = X.WEEK;
                       
                       
                       M_EXCEPT := 0;
                        
                    END IF; 

            END IF; 
        
        END IF;
        
     END LOOP;
     
     
    COMMIT;
    
    
    DBMS_OUTPUT.PUT_LINE('4.sales OPEN '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));
     
    -- 기여도 있는 sales 먼저 차감을 위해 for loop 추가  20220919 
    -- 아래 6가지 경우 발생 = 있고 없고는 경우가 더 많아져서 전부 = 조건 넣고 elseif로 연결
    -- 1. site.gidp >= qtypromised>= sales.gidp 
    -- 2. site.gidp >= sales.gidp >= qtypromised
    -- 3. qtypromised >= site.gidp >= sales.gidp
    -- 4. qtypromised >=sales.gidp >= site.gidp
    -- 5. sales.gidp >=qtypromised >= site.gidp
    -- 6. sales.gidp >= site.gidp >= qtypromised
    -- WEEK,SITE별 계산된 GIDP 먼저 한껀 불러냄  
    FOR SITE IN (
        --W39 GIDP 651대
        SELECT SITEID, ITEM, WEEK, GIDP 
        FROM   MST_GINETTING
        WHERE  PLANID = V_PLANID 
        AND   NVL(GIDP,0) >0
        ORDER BY SITEID, ITEM, WEEK 
    )
    LOOP
    
        --해당 주차 GIDP로 넘어가는 누적 수량 
        V_REMAINGIDP := 0; 
        
        -- GIDP 남는 값 저장 
        V_GIDP := SITE.GIDP;
        
                        
        
        FOR DP IN (
            --기여도 높은 순으로 NOW DP를 발췌 
            SELECT * FROM (
            SELECT RANK() OVER (ORDER BY RK.RNK, A.SALESID, A.DEMANDPRIORITY DESC, A.QTYPROMISED DESC, ROWNUM ) RNUM, 
                   RK.RNK, RK.EXCEPTQ, RK.CUREXCEPT ,
                   ENTERPRISE, SOPROMISEID, A.PLANID, SALESORDERID, SOLINENUM, A.ITEM, QTYPROMISED, PROMISEDDELDATE, 
                   A.SITEID, SHIPTOID, A.SALESID, SALESLEVEL, DEMANDTYPE, A.WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, 
                   DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, 
                   NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, 
                   PREFERENCERANK, A.INITDTTM, A.INITBY, A.UPDTTM, A.UPBY, REASONCODE, A.OPTION_CODE, A.AP1ID
            FROM   EXP_SOPROMISESRCNCP A, MST_GINETTING_SALES RK
            WHERE  A.PLANID = V_PLANID
            AND    QTYPROMISED>0
            AND    RK.PLANID = V_PLANID
            AND    A.ITEM = RK.ITEM
            AND    A.SITEID = RK.SITEID
            AND    A.SALESID = RK.SALESID
            AND    RK.EXCEPTQ > RK.CUREXCEPT
            AND    TO_CHAR(A.PROMISEDDELDATE,'IYYYIW') = SITE.WEEK
            AND    A.ITEM =SITE.ITEM
            AND    A.SITEID = SITE.SITEID
            AND    A.SOLINENUM < 200
            --20220919 estore 제품 후처리 제외시, 계산에는 발췌해서 계산 후 분배시에 제외 하도록 조건 추가 req msjd95
            AND    NOT EXISTS (SELECT 'X' FROM MVES_ITEMSITE_NC WHERE ITEM = A.ITEM AND SALESID = A.SALESID AND SITEID = A.SITEID) --ESTORE 후보충 대상 제외 20210708
            )
            ORDER BY ITEM, SITEID, RNK, SALESID, RNUM, DEMANDPRIORITY DESC, QTYPROMISED DESC
        )
        LOOP
        
            -- 같은 WEEK, SALESID 내에서 남는값 저장 
            -- CUREXCEPT : 이전주차에서 해당 SALES의 GIDP로 넘어간 수량 
            IF DP.RNUM = 1  THEN 
               V_EXCEPTQ := DP.EXCEPTQ- DP.CUREXCEPT; 
               V_SALESID := DP.SALESID; 
            END IF;
            IF V_SALESID <>  DP.SALESID THEN 
               V_EXCEPTQ := DP.EXCEPTQ- DP.CUREXCEPT; 
               V_SALESID := DP.SALESID;
            END IF;
            
            --2,6번 case : 그냥 qtypromised 전부 GIDP로 
            IF ( V_GIDP >= V_EXCEPTQ AND  V_EXCEPTQ  >= DP.QTYPROMISED )
                OR
               ( V_EXCEPTQ >= V_GIDP AND  V_GIDP  >=  DP.QTYPROMISED )
                THEN 
            
                --GI DP로 전량 분류 
                UPDATE EXP_SOPROMISESRCNCP
                SET    SOLINENUM   = SOLINENUM + 200 , 
                       UPBY        = UPBY||'_1'
                WHERE  SALESORDERID = DP.SALESORDERID
                AND    PLANID       = DP.PLANID
                AND    SOLINENUM    = DP.SOLINENUM;
                
                
                UPDATE MST_GINETTING_SALES
                SET    CUREXCEPT = NVL(CUREXCEPT,0) + DP.QTYPROMISED, UPBY= UPBY||'_1'
                WHERE  PLANID = V_PLANID
                AND    SITEID = DP.SITEID
                AND    ITEM   = DP.ITEM
                AND    SALESID = DP.SALESID; 
                
                V_REMAINGIDP := V_REMAINGIDP + DP.QTYPROMISED;
                --남은 기여 수량 
                V_EXCEPTQ := V_EXCEPTQ-DP.QTYPROMISED;
                --남은  SITE.GIDP
                V_GIDP := V_GIDP - DP.QTYPROMISED;
                
                
            --4,5 case : site.gidp 만큼만 gidp 로 넘김 
            ELSIF  (V_EXCEPTQ >= DP.QTYPROMISED  AND  DP.QTYPROMISED >= V_GIDP) 
                    OR
                   (DP.QTYPROMISED>= V_EXCEPTQ  AND  V_EXCEPTQ >= V_GIDP)
                   THEN   
                   -- 남은 SITEGIDP만큼만 넘어간다...
                   
                   
                --! 잔량  그대로 둔다 !--
                UPDATE EXP_SOPROMISESRCNCP
                SET    QTYPROMISED  = QTYPROMISED - V_GIDP,
                       UPBY = UPBY||'_U2'
                WHERE  SALESORDERID = DP.SALESORDERID
                AND    PLANID       = DP.PLANID
                AND    SOLINENUM    = DP.SOLINENUM;

                
                --! 잔량만큼은 SOLINENUM 더해서 INSERT하여 GI Short 처리  !--
                INSERT INTO EXP_SOPROMISESRCNCP
                (  ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, 
                   ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, 
                   DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, 
                   GBM, GLOBALPRIORITY, LOCALPRIORITY, 
                   BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, 
                   PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, 
                   MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE, AP1ID)
                VALUES
                (  DP.ENTERPRISE, DP.SOPROMISEID, DP.PLANID, DP.SALESORDERID, /**++SOLINENUM +200 분리++**/DP.SOLINENUM + 200, 
                   DP.ITEM, /*++남은잔량만큼 INSERT*/ V_GIDP/*++*/, DP.PROMISEDDELDATE, DP.SITEID, DP.SHIPTOID, DP.SALESID, DP.SALESLEVEL, 
                   DP.DEMANDTYPE, DP.WEEK, DP.CHANNELRANK, DP.CUSTOMERRANK, DP.PRODUCTRANK, DP.DEMANDPRIORITY, DP.TIEBREAK, 
                   DP.GBM, DP.GLOBALPRIORITY, DP.LOCALPRIORITY, 
                   DP.BUSINESSTYPE, DP.ROUTING_PRIORITY, DP.NO_SPLIT, DP.MAP_SATISFY_SS, 
                   DP.PREALLOC_ATTRIBUTE, DP.BUILDAHEADTIME, DP.TIMEUOM, DP.AP2ID, DP.GC, 
                   DP.MEASURERANK, DP.PREFERENCERANK, DP.INITDTTM, DP.INITBY, DP.UPDTTM, DP.UPBY||'_I2', DP.REASONCODE, DP.OPTION_CODE, DP.AP1ID);
                 
                 UPDATE MST_GINETTING_SALES
                 SET    CUREXCEPT = NVL(CUREXCEPT,0) + V_GIDP, UPBY= UPBY||'_2'
                 WHERE  PLANID = V_PLANID
                 AND    SITEID = DP.SITEID
                 AND    ITEM   = DP.ITEM
                 AND    SALESID = DP.SALESID; 
                 
                 
                 
                 V_REMAINGIDP := V_REMAINGIDP + V_GIDP;
                 --남은 기여 수량 
                 V_EXCEPTQ := V_EXCEPTQ - V_GIDP;
                 --남은 SITE.GIDP
                 V_GIDP := V_GIDP - V_GIDP;
                 
                 
--                DBMS_OUTPUT.PUT_LINE('22222 V_GIDP '||V_GIDP);
--                DBMS_OUTPUT.PUT_LINE('22222 V_EXCEPTQ '||V_EXCEPTQ);        
--                DBMS_OUTPUT.PUT_LINE('22222 V_REMAINGIDP'||V_REMAINGIDP);
                
            --1,3 case : sales.gidp만큼만 넘긴다.        
            ELSIF  (V_GIDP >= DP.QTYPROMISED AND DP.QTYPROMISED>= V_EXCEPTQ)
                   OR
                   (DP.QTYPROMISED >= V_GIDP AND V_GIDP>= V_EXCEPTQ)
                    THEN 
                   -- 남은 기여 수량만큼만
                   
--                DBMS_OUTPUT.PUT_LINE('333 V_GIDP '||V_GIDP);
--                DBMS_OUTPUT.PUT_LINE('333 V_EXCEPTQ '||V_EXCEPTQ);        
--                DBMS_OUTPUT.PUT_LINE('333 DP.QTYPROMISED '||DP.QTYPROMISED);      
                   
                --!  그대로 둔다 !--
                UPDATE EXP_SOPROMISESRCNCP
                SET    QTYPROMISED  = QTYPROMISED - V_EXCEPTQ,
                       UPBY = UPBY||'_U3'
                WHERE  SALESORDERID = DP.SALESORDERID
                AND    PLANID       = DP.PLANID
                AND    SOLINENUM    = DP.SOLINENUM;

                
                --! 잔량만큼은 SOLINENUM 더해서 INSERT하여 GI Short 처리  !--
                INSERT INTO EXP_SOPROMISESRCNCP
                (  ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, 
                   ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, 
                   DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, 
                   GBM, GLOBALPRIORITY, LOCALPRIORITY, 
                   BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, 
                   PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, 
                   MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE, AP1ID)
                VALUES
                (  DP.ENTERPRISE, DP.SOPROMISEID, DP.PLANID, DP.SALESORDERID, /**++SOLINENUM +200 분리++**/DP.SOLINENUM + 200, 
                   DP.ITEM, /*++ short처리  INSERT*/ V_EXCEPTQ/*++*/, DP.PROMISEDDELDATE, DP.SITEID, DP.SHIPTOID, DP.SALESID, DP.SALESLEVEL, 
                   DP.DEMANDTYPE, DP.WEEK, DP.CHANNELRANK, DP.CUSTOMERRANK, DP.PRODUCTRANK, DP.DEMANDPRIORITY, DP.TIEBREAK, 
                   DP.GBM, DP.GLOBALPRIORITY, DP.LOCALPRIORITY, 
                   DP.BUSINESSTYPE, DP.ROUTING_PRIORITY, DP.NO_SPLIT, DP.MAP_SATISFY_SS, 
                   DP.PREALLOC_ATTRIBUTE, DP.BUILDAHEADTIME, DP.TIMEUOM, DP.AP2ID, DP.GC, 
                   DP.MEASURERANK, DP.PREFERENCERANK, DP.INITDTTM, DP.INITBY, DP.UPDTTM, DP.UPBY||'_I3', DP.REASONCODE, DP.OPTION_CODE, DP.AP1ID);
                 
                 UPDATE MST_GINETTING_SALES
                 SET    CUREXCEPT = NVL(CUREXCEPT,0) + V_EXCEPTQ, UPBY= UPBY||'_3'
                 WHERE  PLANID = V_PLANID
                 AND    SITEID = DP.SITEID
                 AND    ITEM   = DP.ITEM
                 AND    SALESID = DP.SALESID; 
                 
                 
                      
                
                V_REMAINGIDP := V_REMAINGIDP + V_EXCEPTQ;                 
                --남은 SITE.GIDP
                V_GIDP := V_GIDP - V_EXCEPTQ;     
                --남은 기여 수량 
                V_EXCEPTQ := V_EXCEPTQ - V_EXCEPTQ;
                
--                DBMS_OUTPUT.PUT_LINE('3332 V_GIDP '||V_GIDP);
--                DBMS_OUTPUT.PUT_LINE('3332 V_EXCEPTQ '||V_EXCEPTQ);        
--                DBMS_OUTPUT.PUT_LINE('3332 V_REMAINGIDP'||V_REMAINGIDP);     
                   
                
            END IF;
        
            IF V_REMAINGIDP >= SITE.GIDP THEN 
                GOTO UPDAT_POINT;
            END IF;
        
        END LOOP;
        
        <<UPDAT_POINT>> NULL;
        
        
        -- 기여도 높았던 놈들에게 줄 DP 만큼은 GIDP에서 빼고 남긴REMAINGIDP에서 기여도 없는 놈들을 기존 로직대로 나눠줄 예정 
        UPDATE MST_GINETTING
        SET    REMAINGIDP = GREATEST(GIDP - V_REMAINGIDP,0), UPDTTM = SYSDATE, UPBY = 'REMAINUPDATE'
        WHERE  PLANID = V_PLANID
        AND    ITEM = SITE.ITEM
        AND    SITEID = SITE.SITEID
        AND    WEEK = SITE.WEEK
        AND    NVL(GIDP,0) >0;
        
       
    END LOOP;
     
    
    COMMIT;
   
   DBMS_OUTPUT.PUT_LINE('4.site OPEN '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));
    
   --! FOR LOOP 시작 !--
     FOR R IN (
         SELECT  SITEID,  ITEM, WEEK, 
                 RTF, GI, EXCEPTQ, CUREXCEPT, DP, NVL(REMAINGIDP, GIDP) GIDP, MPDP
         FROM    MST_GINETTING
         WHERE   PLANID  = V_PLANID         
         AND     NVL(GIDP,0) > 0
         ORDER BY SITEID, ITEM,  WEEK
     )
     LOOP
     
         BEGIN
     
             V_REXCEPT := R.GIDP;

--              RTF외 판매수량이 W4 DP보다 크면 전량 SOLIENUM 변경
--              RTF외 판매수량이 W4 DP보다 작으면 우선순위 적은놈, 수량 큰순 SOLIENUM 변경 
--              그러나 굳이 구분할 필요없이 차감해가면서 가도록
             --! DEMAND 우선순위별로 발췌하여 FOR LOOP !--
             FOR DP IN (                
                SELECT A.ROWID ROWA, A.ENTERPRISE, A.SOPROMISEID, A.PLANID, A.SALESORDERID, A.SOLINENUM, A.ITEM, A.QTYPROMISED, A.PROMISEDDELDATE, 
                       A.SITEID, A.SHIPTOID, A.SALESID, A.SALESLEVEL, 
                       A.DEMANDTYPE, A.WEEK, CHANNELRANK, A.CUSTOMERRANK, A.PRODUCTRANK, A.DEMANDPRIORITY, A.TIEBREAK, A.GBM, A.GLOBALPRIORITY, A.LOCALPRIORITY, 
                       A.BUSINESSTYPE, A.ROUTING_PRIORITY, A.NO_SPLIT, A.MAP_SATISFY_SS, A.PREALLOC_ATTRIBUTE, A.BUILDAHEADTIME, A.TIMEUOM, A.AP2ID, A.GC, 
                       A.MEASURERANK, A.PREFERENCERANK, A.INITDTTM, A.INITBY, A.UPDTTM, A.UPBY, A.REASONCODE, A.OPTION_CODE, A.AP1ID
                       ,NVL(RK.RNK,0) RNK
--                       ,RANK() OVER (ORDER BY NVL(RK.RNK,0), A.SALESID, A.DEMANDPRIORITY DESC, A.QTYPROMISED DESC, ROWNUM ) RNUM
                FROM   EXP_SOPROMISESRCNCP A , MST_GINETTING_SALES RK
                WHERE  A.PLANID = V_PLANID
                AND    A.ITEM     = R.ITEM 
                AND    A.SITEID   = R.SITEID 
                AND    TO_CHAR(A.PROMISEDDELDATE, 'IYYYIW') = R.WEEK
                AND    A.QTYPROMISED > 0                   
                AND    A.SOLINENUM < 200 
                -- SALES에 올라왔던 DP는 제외 이미 계산 되었으므로, 200 은 제외하고 대신 분배 대상 못찾으면 남은 dp들도 다 넘기도록 20220919
                AND    A.ITEM  = RK.ITEM(+)
                AND    A.SITEID = RK.SITEID(+)
                AND    A.SALESID = RK.SALESID(+)
                AND    A.PLANID = RK.PLANID(+)
                AND    NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y')
                AND    NOT EXISTS (SELECT 'X' FROM MVES_ITEMSITE_NC WHERE ITEM = A.ITEM AND SALESID = A.SALESID AND SITEID = A.SITEID) --ESTORE 후보충 대상 제외 20210708
                ORDER BY NVL(RNK,0), A.PROMISEDDELDATE, A.DEMANDPRIORITY DESC, A.QTYPROMISED DESC
             ) --nvl(rnk,0) order by 해서 sales 없던 놈들 원래 우선순위 순서대로 할방 받고 그래도 못 찾으면 sales 있던 애들한테 다시 가도록 
             LOOP
             
             
                 BEGIN
                     --! 남은 잔량이 0보다 클 경우만 진행한다. 아니면 다음 대상으로 고고싱 !--     
                     IF V_REXCEPT >0 THEN
                         --! 남은 잔량이 DEMAND보다 많을 경우 !--
                         IF V_REXCEPT >= DP.QTYPROMISED THEN               
                         
                             BEGIN   
                                 --! 전량 GI Short, SOLINENUM + 200 !--
                                 UPDATE EXP_SOPROMISESRCNCP
                                 SET    SOLINENUM    = SOLINENUM + 200
                                 WHERE ROWID = DP.ROWA;
                                 --속도 향상을 위해 rowid 사용                             
                                 
                                 
                                 
                             EXCEPTION WHEN DUP_VAL_ON_INDEX THEN
                             
                                -- MAX(SOLINENUM) 을 구할 순 없다...기여도에서 이미 200을 가진놈이 있으면 200+200 = 400번대가 되어버리기 때문에 20220919
                             
                                 --! 전량 GI Short, SOLINENUM + 200 !--
                                 UPDATE EXP_SOPROMISESRCNCP
                                 SET    SOLINENUM    = SOLINENUM + 205
                                 WHERE ROWID = DP.ROWA;
                                 --속도 향상을 위해 rowid 사용                             
                                 
                                 
                             END ;
                             
                             V_REXCEPT := V_REXCEPT - DP.QTYPROMISED;
                         
                         --! 남은 잔량이 DEMAND보다 적을 경우 !--
                         ELSE
                        
                            BEGIN
                                 --! QTYPROMISED - 잔량 뺀 나머지는 그대로 둔다 !--
                                 UPDATE EXP_SOPROMISESRCNCP
                                 SET    QTYPROMISED  = DP.QTYPROMISED - V_REXCEPT
                                 WHERE ROWID = DP.ROWA;

                                 
                                 --! 잔량만큼은 SOLINENUM 더해서 INSERT하여 GI Short 처리  !--
                                 INSERT INTO EXP_SOPROMISESRCNCP
                                 (  ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, 
                                    ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, 
                                    DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, 
                                    GBM, GLOBALPRIORITY, LOCALPRIORITY, 
                                    BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, 
                                    PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, 
                                    MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE, AP1ID)
                                 VALUES
                                 (  DP.ENTERPRISE, DP.SOPROMISEID, DP.PLANID, DP.SALESORDERID, /**++SOLINENUM +200 분리++**/DP.SOLINENUM+200, 
                                    DP.ITEM, /*++남은잔량만큼 INSERT*/ V_REXCEPT/*++*/, DP.PROMISEDDELDATE, DP.SITEID, DP.SHIPTOID, DP.SALESID, DP.SALESLEVEL, 
                                    DP.DEMANDTYPE, DP.WEEK, DP.CHANNELRANK, DP.CUSTOMERRANK, DP.PRODUCTRANK, DP.DEMANDPRIORITY, DP.TIEBREAK, 
                                    DP.GBM, DP.GLOBALPRIORITY, DP.LOCALPRIORITY, 
                                    DP.BUSINESSTYPE, DP.ROUTING_PRIORITY, DP.NO_SPLIT, DP.MAP_SATISFY_SS, 
                                    DP.PREALLOC_ATTRIBUTE, DP.BUILDAHEADTIME, DP.TIMEUOM, DP.AP2ID, DP.GC, 
                                    DP.MEASURERANK, DP.PREFERENCERANK, DP.INITDTTM, DP.INITBY, DP.UPDTTM, DP.UPBY, DP.REASONCODE, DP.OPTION_CODE, DP.AP1ID);
                                    
                            EXCEPTION WHEN DUP_VAL_ON_INDEX THEN
                            
                                --! QTYPROMISED - 잔량 뺀 나머지는 그대로 둔다 !--
                                 UPDATE EXP_SOPROMISESRCNCP
                                 SET    QTYPROMISED  = DP.QTYPROMISED - V_REXCEPT
                                 WHERE ROWID = DP.ROWA;

                                 
                                 --! 잔량만큼은 SOLINENUM 더해서 INSERT하여 GI Short 처리  !--
                                 INSERT INTO EXP_SOPROMISESRCNCP
                                 (  ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, 
                                    ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, 
                                    DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, 
                                    GBM, GLOBALPRIORITY, LOCALPRIORITY, 
                                    BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, 
                                    PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, 
                                    MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE, AP1ID)
                                 VALUES -- max(solinenume) 구해서 더하면 안됨,,, 이미 200번대가 있어서 400번대 되버림 그냥 +205로 해결 20220919
                                 (  DP.ENTERPRISE, DP.SOPROMISEID, DP.PLANID, DP.SALESORDERID, /**++SOLINENUM +200 분리++**/DP.SOLINENUM+205,   
                                    DP.ITEM, /*++남은잔량만큼 INSERT*/ V_REXCEPT/*++*/, DP.PROMISEDDELDATE, DP.SITEID, DP.SHIPTOID, DP.SALESID, DP.SALESLEVEL, 
                                    DP.DEMANDTYPE, DP.WEEK, DP.CHANNELRANK, DP.CUSTOMERRANK, DP.PRODUCTRANK, DP.DEMANDPRIORITY, DP.TIEBREAK, 
                                    DP.GBM, DP.GLOBALPRIORITY, DP.LOCALPRIORITY, 
                                    DP.BUSINESSTYPE, DP.ROUTING_PRIORITY, DP.NO_SPLIT, DP.MAP_SATISFY_SS, 
                                    DP.PREALLOC_ATTRIBUTE, DP.BUILDAHEADTIME, DP.TIMEUOM, DP.AP2ID, DP.GC, 
                                    DP.MEASURERANK, DP.PREFERENCERANK, DP.INITDTTM, DP.INITBY, DP.UPDTTM, DP.UPBY, DP.REASONCODE, DP.OPTION_CODE, DP.AP1ID);
                                    
                            END;           
                                    
                                 V_REXCEPT :=0;
                                       
                         
                         END IF;
                         
                         
                         COMMIT;
                         
                     END IF;
                 
                 EXCEPTION
                     WHEN NO_DATA_FOUND THEN
                     DBMS_OUTPUT.PUT_LINE('3. Error: check '||SQLCODE||'_'||SQLERRM);
                     DBMS_OUTPUT.PUT_LINE('3. DP.INFO  '||DP.SALESORDERID);
                     ROLLBACK;
                     WHEN OTHERS THEN
                     DBMS_OUTPUT.PUT_LINE('3. Error: check '||SQLCODE||'_'||SQLERRM);
                     DBMS_OUTPUT.PUT_LINE('3. DP.INFO  '||DP.SALESORDERID);
                     ROLLBACK;
                 END;
             END LOOP;
        
         EXCEPTION
             WHEN NO_DATA_FOUND THEN
             DBMS_OUTPUT.PUT_LINE('2. Error: check '||SQLCODE||'_'||SQLERRM);
             DBMS_OUTPUT.PUT_LINE('2. R.INFO  '||R.ITEM||'_'||R.SITEID);
             ROLLBACK;
             WHEN OTHERS THEN
             DBMS_OUTPUT.PUT_LINE('2. Error: check '||SQLCODE||'_'||SQLERRM);
             DBMS_OUTPUT.PUT_LINE('2. R.INFO  '||R.ITEM||'_'||R.SITEID);
             ROLLBACK;
         END;
     
     END LOOP;
     
     COMMIT;
     
     DBMS_OUTPUT.PUT_LINE('5.PRIORITY S SRC :    '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss')); 
     -- MPDP SOLINENUM 1~100
     -- NEW FCST DP SOLINENUME 100~199 TYPE PRIORITY : 8
     -- GI DP SOLINENUM 200~299 TYPE PRIORITY : 1->8 변경작업 아래에서 수행
     -- NEW FCST DP && GI DP SOLINENUM 300~399 TYPE PRIORITY :8
      BEGIN
          
          --! GI DP SOLINENUM 200~299 TYPE PRIORITY : 1->8 변경작업 아래에서 수행 !--
          --2019.09.02 직거래선(DIR_S)인 경우에는 Demand 우선순위 변경없음(엔진 경합으로 나온 결과로 SHORT 재분류. 이중목 부장님)
          UPDATE EXP_SOPROMISESRCNCP A
          SET    DEMANDPRIORITY = SUBSTR(DEMANDPRIORITY, 1,3)||'8'||SUBSTR(DEMANDPRIORITY, 5,4)
          WHERE  PLANID         = V_PLANID
          AND    SOLINENUM BETWEEN 200 AND 299
          AND    NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND SITEID = A.SITEID)
          AND    SITEID NOT IN (SELECT SITEID FROM MST_SITE WHERE SHIPMENTTYPE='DIR_S' AND ISVALID='Y') --2019.09.02 직거래선(DIR_S)제외
          AND    SALESORDERID NOT LIKE 'UNF%'; --220413 UNF는 TYPE RANK가 9순위로 최 후순위어야 함 
          
      EXCEPTION
         WHEN NO_DATA_FOUND THEN
         DBMS_OUTPUT.PUT_LINE('33333' );
             NULL;
      END;
     
      COMMIT;
      
     BEGIN
         --! 1. NEW FCST 수량이 제대로 N빵되어 EXP_SOPROMISESRCNCP 들어갔는지 확인 !--
         SELECT COUNT(*) INTO SMS_QTYCOUNT 
         FROM(
             SELECT ITEM, SITEID, WEEK, SUM(PROD) , SUM(DEV) 
             FROM(
                SELECT SITEID, ITEM, TO_CHAR(PROMISEDDELDATE,'IYYYIW') WEEK, 
                       QTYPROMISED PROD , 0 DEV  
                FROM   EXP_SOPROMISESRCNCP A
                WHERE  SOLINENUM >=200
                AND    NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND SITEID = A.SITEID)
                AND    NOT EXISTS (SELECT 'X' FROM MVES_ITEMSITE_NC WHERE ITEM = A.ITEM   AND SITEID = A.SITEID) --20210723 E-STORE 대상 제외
                UNION ALL
                SELECT SITEID, ITEM, WEEK, 0 PROD , GIDP DEV 
                FROM   MST_GINETTING A
                WHERE  PLANID = V_PLANID
                AND    NOT EXISTS (SELECT 'X' FROM MVES_ITEMSITE_NC WHERE ITEM = A.ITEM   AND SITEID = A.SITEID) --20210723 E-STORE 대상 제외
                AND DP > 0
                )
                GROUP BY ITEM, SITEID, WEEK
                HAVING SUM(PROD)<> SUM(DEV)
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
     
     BEGIN
         --! 페어쉐어 하면서 (-)나오면 에러 발생 !--
         SELECT COUNT(*) INTO SMS_COUNT 
         FROM   EXP_SOPROMISESRCNCP A
         WHERE  PLANID   = V_PLANID
         AND    NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND SITEID = A.SITEID)
         AND    QTYPROMISED < 0;
         
     EXCEPTION
        WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('55555' );
            NULL;
     END;
     
     IF SMS_COUNT > 0 THEN
        DBMS_OUTPUT.PUT_LINE('Error: check PAIRSHARE');
     END IF;
     
     
     BEGIN
         --! 2. NEW FCST 로직전과 수량비교 !--
         SELECT SUM(QTYPROMISED)  INTO SMS_QTYBEFORE 
         FROM   BUF_SOPROMISESRCNCP_POST A
         WHERE  PLAN     = 'DPDEC'
         AND    PLANID   = V_PLANID
         AND    NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND SITEID = A.SITEID);
         
         SELECT SUM(QTYPROMISED)  INTO SMS_QTYAFTER 
         FROM   EXP_SOPROMISESRCNCP A
         WHERE  PLANID   = V_PLANID
         AND    NOT EXISTS (SELECT 'X' FROM V_MTA_SELLERMAP WHERE  ITEM = A.ITEM AND SITEID = A.SITEID);
         
         
     EXCEPTION
        WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('66666' );
            NULL;
     END;
     
     IF SMS_QTYBEFORE != SMS_QTYAFTER THEN
        DBMS_OUTPUT.PUT_LINE('Error: check GINETTINGERROR');
     END IF;
    
        DBMS_OUTPUT.PUT_LINE('END : '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss') );


EXCEPTION
    WHEN NO_DATA_FOUND THEN
        NULL; 
    WHEN OTHERS THEN
       DBMS_OUTPUT.PUT_LINE('Error: check '||SQLCODE||'_'||SQLERRM);
       
       ROLLBACK;
END;
