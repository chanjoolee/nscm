CREATE OR REPLACE PROCEDURE WSCM01.SP_FN_CHANNELSTOCK_NC(IN_TYPE VARCHAR2 DEFAULT 'NCP') IS
/******************************************************************************
   NAME:       WSCM01.SP_FN_CHANNELSTOCK
   PURPOSE:    한국총괄 GC GUIDE 수량 분배
   REVISIONS:
   VER     DATE        AUTHOR   DESCRIPTION
   ------- ----------  -------  ------------------------------------
   1.0     2014.01.15  Eunju    신규 버전
   1.1     2014.01.21  Eunju    ITEM별 수량 차감으로 변경
   1.2     2014.02.04  Eunju    SOLINENUM Check 로직 추가
   2.0     2015.01.26  Eunju    전법인 확산 및 재고로직 추가
   2.1     2015.02.05  Eunju    Bulk 및 INU, XSE 가용량 추가
   2.2     2015.04.23  Eunju    Sell-in Guide 기준을 XID에서 XSE로 변경 Req by.정영훈K
   2.3     2015.06.04  Eunju    Guide와 가용량 차감시 NEWORD 우선순위를 높이고 NEWORD Short 발생시 강제 살림 Req By.양효순K
   2.4     2015.06.19  EunJu    Postponement 공용화 재고 적용 및 가용량 WEEK 별 적용 Req by. 김상남k
******************************************************************************/

    --V_PGMNAME           VARCHAR2(30) := 'SP_FN_CHANNELSTOCK';

    V_PLANID            EXP_SOPROMISESRCNCP.PLANID%TYPE;
    V_PLANWEEK          MST_PLAN.PLANWEEK%TYPE;
    V_EFFSTARTDATE      MST_PLAN.EFFSTARTDATE %TYPE;
    V_EFFENDDATE        MST_PLAN.EFFENDDATE %TYPE;

    V_LOGSEQ            NUMBER(10);
    V_STARTDATE         DATE;

    V_DPGUIDE           INTEGER := 0;

    V_CHECK1            INTEGER := 0;
    V_REMAINQTY         INTEGER := 0;
    V_AVAILQTY          INTEGER := 0;

    V_SOLINENUM         NUMBER := 0;
    V_AVAILQTY2         INTEGER:=0;
    V_ALIVE             INTEGER:=0;
    V_ROLLING           INTEGER:=0;
    V_CHSTOCK           INTEGER:=0;

    --2018.12.03 추가
    V_TYPE              VARCHAR2(10);
    V_NUM               NUMBER := 0;
    V_DFREGION          MST_PLAN.DFREGION%TYPE;

    --! 3레벨 GUIDE !--
    --20170821 CHANNEL STOCK EFF 구간관리 추가 BY HK REQ 전창민PRO
    CURSOR dpguide_ap2 IS
        SELECT GBM, PRODUCTGROUP, MEASURE, AP2ID, LEVELID, SALESID,
               UDAITEM, SWEEK, QTY
        FROM (
            SELECT TO_CHAR(NVL(B.EFFSTARTDATE, DATE'2000-01-01'), 'IYYYIW') STARTWEEK,
                   TO_CHAR(NVL(B.EFFENDDATE, DATE'2999-12-30'), 'IYYYIW') ENDWEEK,
                   A.*
            FROM (
                SELECT A.GBM, A.PRODUCTGROUP, A.MEASURE, B.AP2ID, B.LEVELID, A.SALESID, C.BASICNAME,
                       --20191106 INS/INU, XID/XSE 로직 삭제
--                       DECODE(A.SALESID
--                                ,'300143',REPLACE(A.UDAITEM, 'XSE','XID')
--                                ,'300147',REPLACE(A.UDAITEM, 'INU','INS'), A.UDAITEM ) UDAITEM,
                       A.UDAITEM,
                       A.SWEEK, SUM( A.QTY) QTY
                FROM   EXP_DPGUIDE_NC A, GUI_SALESHIERARCHY B, VUI_ITEMATTB C
                WHERE  A.SALESID = B.SALESID
                AND    B.LEVELID = 3
                --20191106 INS/INU, XID/XSE 로직 삭제
--                AND    DECODE(B.AP2ID
--                                    ,'300143',REPLACE(A.UDAITEM, 'XSE','XID')
--                                    ,'300147',REPLACE(A.UDAITEM, 'INU','INS'), A.UDAITEM ) = C.ITEM
                AND A.UDAITEM = C.ITEM
                AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.UDAITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
                GROUP BY A.GBM, A.PRODUCTGROUP, A.MEASURE, B.AP2ID, B.LEVELID, A.SALESID, C.BASICNAME,
                         --20191106 INS/INU, XID/XSE 로직 삭제
--                         DECODE(A.SALESID
--                                  ,'300143',REPLACE(A.UDAITEM, 'XSE','XID')
--                                  ,'300147',REPLACE(A.UDAITEM, 'INU','INS'), A.UDAITEM ),
                            A.UDAITEM, A.SWEEK
                ) A, MST_CHSTOCK B
            WHERE  A.SALESID   = B.SALESID(+)
            AND    A.UDAITEM   = B.ITEM(+)
        )
        WHERE  SWEEK BETWEEN STARTWEEK AND  ENDWEEK ;

    --! 4레벨 GUIDE !--
    --20170821 CHANNEL STOCK EFF 구간관리 추가 BY HK REQ 전창민PRO
    CURSOR dpguide_ap1 IS
        SELECT GBM, PRODUCTGROUP, MEASURE, AP2ID, LEVELID, SALESID,
               UDAITEM, SWEEK, QTY
        FROM (
            SELECT TO_CHAR(NVL(B.EFFSTARTDATE, DATE'2000-01-01'), 'IYYYIW') STARTWEEK,
                   TO_CHAR(NVL(B.EFFENDDATE, DATE'2999-12-30'), 'IYYYIW') ENDWEEK,
                   A.*
            FROM (
                SELECT A.GBM, A.PRODUCTGROUP, A.MEASURE, B.AP2ID, B.LEVELID, A.SALESID, C.BASICNAME,
                       --20191106 INS/INU, XID/XSE 로직 삭제
--                       DECODE(B.AP2ID
--                                ,'300143',REPLACE(A.UDAITEM, 'XSE','XID')
--                                ,'300147',REPLACE(A.UDAITEM, 'INU','INS'), A.UDAITEM ) UDAITEM,
                       A.UDAITEM,
                       A.SWEEK, SUM( A.QTY) QTY
                FROM   EXP_DPGUIDE_NC A, GUI_SALESHIERARCHY B, VUI_ITEMATTB C
                WHERE  A.SALESID = B.SALESID
                AND    B.LEVELID = 4
                --20191106 INS/INU, XID/XSE 로직 삭제
--                AND    DECODE(B.AP2ID
--                                    ,'300143',REPLACE(A.UDAITEM, 'XSE','XID')
--                                    ,'300147',REPLACE(A.UDAITEM, 'INU','INS'), A.UDAITEM ) = C.ITEM
                AND A.UDAITEM = C.ITEM
                AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.UDAITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
                GROUP BY A.GBM, A.PRODUCTGROUP, A.MEASURE, B.AP2ID, B.LEVELID, A.SALESID, C.BASICNAME,
                         --20191106 INS/INU, XID/XSE 로직 삭제
--                         DECODE(B.AP2ID
--                                  ,'300143',REPLACE(A.UDAITEM, 'XSE','XID')
--                                  ,'300147',REPLACE(A.UDAITEM, 'INU','INS'), A.UDAITEM ),
                        A.UDAITEM, A.SWEEK
                ) A, MST_CHSTOCK B
            WHERE  A.SALESID   = B.SALESID(+)
            AND    A.UDAITEM   = B.ITEM(+)
        )
        WHERE  SWEEK BETWEEN STARTWEEK AND  ENDWEEK ;


    --! 5레벨 GUIDE !--
    --20170821 CHANNEL STOCK EFF 구간관리 추가 BY HK REQ 전창민PRO
    CURSOR dpguide_account IS
        SELECT GBM, PRODUCTGROUP, MEASURE, AP2ID, LEVELID, SALESID,
               UDAITEM, SWEEK, QTY
        FROM (
            SELECT TO_CHAR(NVL(B.EFFSTARTDATE, DATE'2000-01-01'), 'IYYYIW') STARTWEEK,
                   TO_CHAR(NVL(B.EFFENDDATE, DATE'2999-12-30'), 'IYYYIW') ENDWEEK,
                   A.*
            FROM (
                SELECT A.GBM, A.PRODUCTGROUP, A.MEASURE, B.AP2ID, B.LEVELID, A.SALESID, C.BASICNAME,
                       --20191106 INS/INU, XID/XSE 로직 삭제
--                       DECODE(B.AP2ID
--                                ,'300143',REPLACE(A.UDAITEM, 'XSE','XID')
--                                ,'300147',REPLACE(A.UDAITEM, 'INU','INS'), A.UDAITEM ) UDAITEM,
                       A.UDAITEM,
                       A.SWEEK, SUM(A.QTY) QTY
                FROM   EXP_DPGUIDE_NC A, GUI_SALESHIERARCHY B, VUI_ITEMATTB C
                WHERE  A.SALESID = B.SALESID
                AND    B.LEVELID = 5
                --20191106 INS/INU, XID/XSE 로직 삭제
--                AND    DECODE(B.AP2ID
--                                    ,'300143',REPLACE(A.UDAITEM, 'XSE','XID')
--                                    ,'300147',REPLACE(A.UDAITEM, 'INU','INS'), A.UDAITEM ) = C.ITEM
                AND A.UDAITEM = C.ITEM
                AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.UDAITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
                GROUP BY A.GBM, A.PRODUCTGROUP, A.MEASURE, B.AP2ID, B.LEVELID, A.SALESID, C.BASICNAME,
                         --20191106 INS/INU, XID/XSE 로직 삭제
--                         DECODE(B.AP2ID
--                                  ,'300143',REPLACE(A.UDAITEM, 'XSE','XID')
--                                  ,'300147',REPLACE(A.UDAITEM, 'INU','INS'), A.UDAITEM ),
                        A.UDAITEM, A.SWEEK
                 ) A, MST_CHSTOCK B
            WHERE  A.SALESID   = B.SALESID(+)
            AND    A.UDAITEM   = B.ITEM(+)
        )
        WHERE  SWEEK BETWEEN STARTWEEK AND  ENDWEEK ;

BEGIN

    --! LOG 시퀀스 발췌 !--
    SELECT SEQ_LOG.NEXTVAL INTO V_LOGSEQ FROM DUAL;
    --! 현재 시스템 시간 저장 !--
    SELECT SYSDATE INTO V_STARTDATE FROM DUAL;
    --! 현재 PLAN 정보 저장 !--
    SP_FN_GET_PLANID(IN_TYPE,V_PLANID,V_PLANWEEK,V_EFFSTARTDATE,V_EFFENDDATE);

    SELECT DFREGION INTO V_DFREGION
    FROM   MST_PLAN
    WHERE  PLANID = V_PLANID;

--  1. MAIN PLAN에 쉘에서 EXP_DPGUIDE_MAIN에 백업
--  2. CHANNEL STOCK 돌때 수요일 아주이면
--   EXP_DPGUIDE_NC 지우고 _MAIN걸로 갈아끼우고 돌리기...
--   월화는 계속 EXP_DPGUIDE_NC로 돌리기
    IF  SUBSTR(V_PLANID,7,8)= 'WE' AND  V_DFREGION = 'AS' THEN

        EXECUTE IMMEDIATE 'TRUNCATE TABLE EXP_DPGUIDE_NC';


        INSERT INTO EXP_DPGUIDE_NC
        (GBM, PRODUCTGROUP, MEASURE, SALESID, UDAITEM, SWEEK, QTY, INITDTTM, INITBY, UPDTTM, UPBY)
        SELECT GBM, PRODUCTGROUP, MEASURE, SALESID, UDAITEM, SWEEK, QTY, INITDTTM, INITBY, UPDTTM, UPBY FROM EXP_DPGUIDE_BK
        WHERE  PLANID = SUBSTR(V_PLANID,1,6)||'_M';


        COMMIT;

    END IF;

     --2018.12.03 수정(V PLAN 시 당주기준으로 PROMISEDDELDATE 변경)
--     SELECT TYPE
--     INTO   V_TYPE
--     FROM   MST_PLAN
--     WHERE  ISRUNNING = 'Y'
--     AND    TYPE != 'NCP';
--
--     IF V_TYPE = 'VPLAN' THEN
--        V_NUM := 1;
--     END IF;
--
--     --EXP_DPGUIDE_NC BK
--     DELETE FROM EXP_DPGUIDE_NC_BK
--     WHERE SUBSTR(PLANID,0,6) < (SELECT TO_CHAR(SYSDATE-7*8,'IYYYIW') PLANID
--                                 FROM DUAL)
--           OR PLANID=V_PLANID;
--
--     INSERT INTO EXP_DPGUIDE_NC_BK
--     SELECT GBM, PRODUCTGROUP, V_PLANID, MEASURE, SALESID, UDAITEM, SWEEK, QTY, INITDTTM, INITBY, UPDTTM, UPBY
--     FROM EXP_DPGUIDE_NC;
--
--     COMMIT;


    DBMS_OUTPUT.PUT_LINE('START :    '||to_char(sysdate,'yyyy-mm-dd-HH24:mi:ss'));
    DBMS_OUTPUT.PUT_LINE('V_PLANID:'||V_PLANID ||'  V_EFFSTARTDATE :'||V_EFFSTARTDATE);


    BEGIN
    --! 이전 단계가 중복 수행되어 이미 OPTION_CODE가 1인 것이 없는지 체크 !--
    select count(*) into V_CHECK1 from EXP_SOPROMISESRCNCP
    where MOD(NVL(OPTION_CODE,0), 2) = 1;

    --! 이전 단계가 중복 수행되어 이미 OPTION_CODE가 1인 것이 있으면 에러 출력 !--
    IF ( V_CHECK1 > 0) THEN
    DBMS_OUTPUT.PUT_LINE('0. Error : CHSTOCK OPTION_CODE Check');
    END IF;

    END;

    --! STEP1. 데이터 백업 !--
    BEGIN
    --! 버퍼 테이블 비우기 !--
    EXECUTE IMMEDIATE 'ALTER TABLE BUF_SOPROMISESRCNCP_POST TRUNCATE PARTITION PCHL';

    EXECUTE IMMEDIATE 'Alter session ENABLE parallel DML';

    --! CHANNEL STOCK 하기전 데이터 백업!--
    INSERT /*+  APPEND  PARALLEL(T  5)  */  INTO BUF_SOPROMISESRCNCP_POST T
            (PLAN, ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED, PROMISEDDELDATE
           , SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK
           , GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS
           , PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK
           , INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE,OPTION_CODE)
        SELECT /*+ FULL(S)  PARALLEL(S 5) */
                 'CHL', ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED, PROMISEDDELDATE
                , SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK
                , GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS
                , PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK
                , INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE
        FROM EXP_SOPROMISESRCNCP S
        WHERE PLANID = V_PLANID;

    COMMIT;

    EXECUTE IMMEDIATE 'Alter session DISABLE parallel DML';
    --! 예외 처리 !--
    EXCEPTION

    WHEN NO_DATA_FOUND THEN
    DBMS_OUTPUT.PUT_LINE('1. Backup Error : check '||SQLCODE||'_'||SQLERRM);
    ROLLBACK;
    WHEN OTHERS THEN
    DBMS_OUTPUT.PUT_LINE('1. Backup Error : check '||SQLCODE||'_'||SQLERRM);
    ROLLBACK;

    END;

    --! 4레벨(AP1) 데이터 추가 !--
    UPDATE EXP_SOPROMISESRCNCP A
    SET AP1ID = (SELECT AP1ID FROM GUI_SALESHIERARCHY WHERE SALESID = A.SALESID );

    /*-- STEP2.SELL-IN GUIDE 수량 차감--*/
    BEGIN
        --! 1) 5레벨 단위 Sell-in Guide 차감 !--
        FOR i IN dpguide_account LOOP
        --! 가이드 데이터가 없으면 LOOP 빠져나감 !--
        exit WHEN dpguide_account%notfound;
            V_DPGUIDE := i.QTY;
            FOR DP IN (
                SELECT /*+ index( A X2_EXP_SOPROMISESRCNCP) */ A.ROWID ROWA, ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, A.ITEM, QTYPROMISED, PROMISEDDELDATE
                , SITEID, SHIPTOID, A.SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK
                , GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS
                , PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK
                , A.INITDTTM, A.INITBY, A.UPDTTM, A.UPBY, REASONCODE, OPTION_CODE
                FROM EXP_SOPROMISESRCNCP A
                WHERE PLANID = V_PLANID
                    AND A.QTYPROMISED > 0
                    AND A.SALESID = I.SALESID
                    AND TO_CHAR(PROMISEDDELDATE, 'IYYYIW') = I.SWEEK
                    --인도일 경우 INU -> INS로 교체
                    --인도네시아일 경우 XID -> XSE, 가이드 수량이 INS 또는 XSE로만 들어오기 때문
                    --20191106 INS/INU, XID/XSE 로직 삭제
--                    AND DECODE(A.SITEID, 'L5N0',REPLACE(A.ITEM, 'INU','INS')
--                                       , 'S529',REPLACE(A.ITEM, 'XSE', 'XID')
--                                       , 'S529WC27',REPLACE(A.ITEM, 'XSE','XID'), A.ITEM) = I.UDAITEM
                    AND A.ITEM = I.UDAITEM

                    --2020.03.26 SIEL B2B Site : SIEL(DELHI)_B2B(L5N0WBN0)의 CH_CONSTRAINT Short 제외(전창민 프로 요청)
                    AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
                    AND (A.ITEM, A.SITEID, A.SALESID) NOT IN (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE)

                --인도네시아일 경우 XID가 XSE보다 우선순위가 높음, 나머지 법인은 DEMANDPRIORITY 따라감
                --Demand type이 COM_ORD 일 경우 1순위, NEW_ORD 일 경우 2순위로 우선순위 강제 조정 Req By.양효순K
                ORDER BY DECODE(FN_EXTRACT(A.SALESORDERID,'::',1), 'COM_ORD', 1, 'NEW_ORD', 2, 3),
                         --20191106 INS/INU, XID/XSE 로직 삭제
--                         DECODE(A.SITEID, 'S529', ITEM, 'S529WC27', ITEM, DEMANDPRIORITY),
                         DEMANDPRIORITY, ITEM, SALESORDERID DESC, SALESID DESC, QTYPROMISED DESC, SOLINENUM, SITEID DESC
            )
            LOOP
                --! 가이드 수량 > 0 일때만 수행 !--
                IF V_DPGUIDE > 0 THEN
                    BEGIN
                        --! 가이드 수량 >= Demand값 이면 가이드수량만 차감 !--
                        IF V_DPGUIDE >= DP.QTYPROMISED THEN
                            V_DPGUIDE := V_DPGUIDE - DP.QTYPROMISED;
                        --! 가이드 수량 < Demand값 !--
                        ELSIF V_DPGUIDE < DP.QTYPROMISED THEN
                            --DBMS_OUTPUT.PUT_LINE('3. 가이드 수량 < Demand값  : '||DP.QTYPROMISED);
                            BEGIN
                                --! QTY = Demand 값- 가이드수량 (Cnannel Stock Short), OPTION_CODE + 1 (2진수 첫자리) !--
                                INSERT INTO EXP_SOPROMISESRCNCP
                                       ( ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED,
                                       PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK,
                                       DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY,
                                       NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM,
                                       AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE)
                                VALUES
                                       (DP.ENTERPRISE, DP.SOPROMISEID, DP.PLANID, DP.SALESORDERID, /*++*/DP.SOLINENUM + 1/*++*/, DP.ITEM, /*++*/DP.QTYPROMISED-V_DPGUIDE/*++*/,
                                       DP.PROMISEDDELDATE, DP.SITEID, DP.SHIPTOID, DP.SALESID, DP.SALESLEVEL, DP.DEMANDTYPE, DP.WEEK, DP.CHANNELRANK, DP.CUSTOMERRANK, DP.PRODUCTRANK,
                                       DP.DEMANDPRIORITY, DP.TIEBREAK, DP.GBM, DP.GLOBALPRIORITY, DP.LOCALPRIORITY, DP.BUSINESSTYPE, DP.ROUTING_PRIORITY,
                                       DP.NO_SPLIT, DP.MAP_SATISFY_SS, DP.PREALLOC_ATTRIBUTE, DP.BUILDAHEADTIME, DP.TIMEUOM,
                                       DP.AP2ID, DP.GC, DP.MEASURERANK, DP.PREFERENCERANK, DP.INITDTTM, DP.INITBY, DP.UPDTTM, 'C/Stock',DP.REASONCODE,
                                       /*++*/NVL(DP.OPTION_CODE,0)+BIN_TO_NUM(1)/*++*/);

                            --! INSERT 중복시 예외 처리 !--
                            EXCEPTION WHEN DUP_VAL_ON_INDEX THEN

                                --! key 값 에러 발생시 MAX solinenum 을 신규 발췌 !--
                                SELECT MAX(SOLINENUM) + 1
                                INTO V_SOLINENUM
                                FROM EXP_SOPROMISESRCNCP
                                WHERE PLANID = DP.PLANID
                                AND SALESORDERID = DP.SALESORDERID;

                                --!  MAX solinenum으로 INSERT 재시도 !--
                                INSERT INTO EXP_SOPROMISESRCNCP
                                       ( ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED,
                                       PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK,
                                       DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY,
                                       NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM,
                                       AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE)
                                VALUES
                                       (DP.ENTERPRISE, DP.SOPROMISEID, DP.PLANID, DP.SALESORDERID,  /*++ MAX SOLINENUM으로 삽입++*/ V_SOLINENUM /*++*/, DP.ITEM, /*++*/DP.QTYPROMISED-V_DPGUIDE/*++*/,
                                       DP.PROMISEDDELDATE, DP.SITEID, DP.SHIPTOID, DP.SALESID, DP.SALESLEVEL, DP.DEMANDTYPE, DP.WEEK, DP.CHANNELRANK, DP.CUSTOMERRANK, DP.PRODUCTRANK,
                                       DP.DEMANDPRIORITY, DP.TIEBREAK, DP.GBM, DP.GLOBALPRIORITY, DP.LOCALPRIORITY, DP.BUSINESSTYPE, DP.ROUTING_PRIORITY,
                                       DP.NO_SPLIT, DP.MAP_SATISFY_SS, DP.PREALLOC_ATTRIBUTE, DP.BUILDAHEADTIME, DP.TIMEUOM,
                                       DP.AP2ID, DP.GC, DP.MEASURERANK, DP.PREFERENCERANK, DP.INITDTTM, DP.INITBY, DP.UPDTTM, 'C/Stock',DP.REASONCODE,
                                       /*++*/NVL(DP.OPTION_CODE,0)+BIN_TO_NUM(1)/*++*/);

--                                DBMS_OUTPUT.PUT_LINE('2-1-1 CHANNEL STOCK 5 LEVEL SOLINENUM 중복발생 MAX+1 : '||DP.SALESORDERID||'***SOLINNUM'||V_SOLINENUM);
                            END;

                            --! 가이드 수량값만큼은 일반 DP로 !--
                            UPDATE EXP_SOPROMISESRCNCP
                            SET    QTYPROMISED = V_DPGUIDE
                            WHERE  ROWID = DP.ROWA;

                            --! 모두 소진 했으므로 Guide 0 처리 !--
                            V_DPGUIDE := 0;

                            COMMIT;
                        END IF;
                    --! 예외 처리 !--
                    EXCEPTION
                         WHEN NO_DATA_FOUND THEN
                         DBMS_OUTPUT.PUT_LINE('2-1-1 CHANNEL STOCK 5 LEVEL Error (DP_GUIDE > 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-1-1 CHANNEL STOCK 5 LEVEL DP.INFO  '||DP.SALESORDERID);
                         ROLLBACK;
                         WHEN OTHERS THEN
                         DBMS_OUTPUT.PUT_LINE('2-1-1 CHANNEL STOCK 5 LEVEL Error (DP_GUIDE > 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-1-1 CHANNEL STOCK 5 LEVEL DP.INFO  '||DP.SALESORDERID);
                         ROLLBACK;
                    END;

                --! DP GUIDE 수량이 0으로 들어올 경우 !--

                ELSIF V_DPGUIDE = 0 THEN
                    BEGIN
                        --! 가이드를 모두 소진하여 0 이면 나머지는 전부다 Channel Stock Short !--
                        UPDATE EXP_SOPROMISESRCNCP
                        SET    OPTION_CODE =  NVL(OPTION_CODE,0) +BIN_TO_NUM(1)
                               ,UPBY = 'C/Stock'
                        WHERE ROWID = DP.ROWA;

                        COMMIT;
                    --! 예외 처리 !--
                    EXCEPTION
                         WHEN NO_DATA_FOUND THEN
                         DBMS_OUTPUT.PUT_LINE('2-1-2 CHANNEL STOCK 5 LEVEL Error (DP_GUIDE = 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-1-2 CHANNEL STOCK 5 LEVEL DP.INFO  '||DP.SALESORDERID||'**'||DP.PLANID||'**'||DP.SOLINENUM||'**'||DP.PROMISEDDELDATE);
                         ROLLBACK;
                         WHEN OTHERS THEN
                         DBMS_OUTPUT.PUT_LINE('2-1-2 CHANNEL STOCK 5 LEVEL Error (DP_GUIDE = 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-1-2 CHANNEL STOCK 5 LEVEL DP.INFO  '||DP.SALESORDERID||'**'||DP.PLANID||'**'||DP.SOLINENUM||'**'||DP.PROMISEDDELDATE);
                         ROLLBACK;
                    END;
                END IF;
            END LOOP;
        END LOOP;
    END;

    BEGIN
        --! 2) 4레벨 단위 Sell-in Guide 차감 !--
        FOR i IN dpguide_ap1 LOOP
        exit WHEN dpguide_ap1%notfound;
            V_DPGUIDE := i.QTY;
            FOR DP IN (
                SELECT /*+ index( A X2_EXP_SOPROMISESRCNCP) */ A.ROWID ROWA,  ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, A.ITEM, QTYPROMISED, PROMISEDDELDATE
                , SITEID, SHIPTOID, A.SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK
                , GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS
                , PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK
                , A.INITDTTM, A.INITBY, A.UPDTTM, A.UPBY, REASONCODE, OPTION_CODE
                FROM EXP_SOPROMISESRCNCP A
                WHERE PLANID = V_PLANID
                    AND A.QTYPROMISED > 0
                    AND A.AP1ID = I.SALESID
                    AND TO_CHAR(PROMISEDDELDATE, 'IYYYIW') = I.SWEEK
                    --인도일 경우 INU -> INS로 교체
                    --인도네시아일 경우 XID -> XSE, 가이드 수량이 INS 또는 XSE로만 들어오기 때문
                    --20191106 INS/INU, XID/XSE 로직 삭제
--                    AND DECODE(A.SITEID, 'L5N0',REPLACE(A.ITEM, 'INU','INS')
--                                       , 'S529',REPLACE(A.ITEM, 'XSE','XID')
--                                       , 'S529WC27',REPLACE(A.ITEM, 'XSE','XID'), A.ITEM) = I.UDAITEM
                    AND A.ITEM = I.UDAITEM

                    --2020.03.26 SIEL B2B Site : SIEL(DELHI)_B2B(L5N0WBN0)의 CH_CONSTRAINT Short 제외(전창민 프로 요청)
                    AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
                    AND (A.ITEM, A.SITEID, A.SALESID) NOT IN (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE)

                --Demand type이 COM_ORD 일 경우 1순위, NEW_ORD 일 경우 2순위로 우선순위 강제 조정 Req By.양효순K
                --인도네시아일 경우 XID가 XSE보다 우선순위가 높음, 나머지 법인은 DEMANDPRIORITY 따라감
                ORDER BY DECODE(FN_EXTRACT(A.SALESORDERID,'::',1), 'COM_ORD', 1, 'NEW_ORD', 2, 3),
                         --20191106 INS/INU, XID/XSE 로직 삭제
--                         DECODE(A.SITEID, 'S529', ITEM, 'S529WC27', ITEM, DEMANDPRIORITY),
                         DEMANDPRIORITY, ITEM, SALESORDERID DESC, SALESID DESC, QTYPROMISED DESC, SOLINENUM, SITEID DESC
            )
            LOOP
                --! 가이드 수량 > 0 일때만 수행 !--
                IF V_DPGUIDE > 0 THEN
                    BEGIN
                        --! 가이드 수량 >= Demand값 이면 가이드수량만 차감 !--
                        IF V_DPGUIDE >= DP.QTYPROMISED THEN
                            V_DPGUIDE := V_DPGUIDE - DP.QTYPROMISED;
                        --! 가이드 수량 < Demand값 !--
                        ELSIF V_DPGUIDE < DP.QTYPROMISED THEN
                            --DBMS_OUTPUT.PUT_LINE('3. 가이드 수량 < Demand값  : '||DP.QTYPROMISED);
                            BEGIN
                                --! QTY = Demand 값- 가이드수량 (Cnannel Stock Short), OPTION_CODE + 1 (2진수 첫자리) !--
                                INSERT INTO EXP_SOPROMISESRCNCP
                                       ( ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED,
                                       PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK,
                                       DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY,
                                       NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM,
                                       AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE)
                                VALUES
                                       (DP.ENTERPRISE, DP.SOPROMISEID, DP.PLANID, DP.SALESORDERID, /*++*/DP.SOLINENUM + 1/*++*/, DP.ITEM, /*++*/DP.QTYPROMISED-V_DPGUIDE/*++*/,
                                       DP.PROMISEDDELDATE, DP.SITEID, DP.SHIPTOID, DP.SALESID, DP.SALESLEVEL, DP.DEMANDTYPE, DP.WEEK, DP.CHANNELRANK, DP.CUSTOMERRANK, DP.PRODUCTRANK,
                                       DP.DEMANDPRIORITY, DP.TIEBREAK, DP.GBM, DP.GLOBALPRIORITY, DP.LOCALPRIORITY, DP.BUSINESSTYPE, DP.ROUTING_PRIORITY,
                                       DP.NO_SPLIT, DP.MAP_SATISFY_SS, DP.PREALLOC_ATTRIBUTE, DP.BUILDAHEADTIME, DP.TIMEUOM,
                                       DP.AP2ID, DP.GC, DP.MEASURERANK, DP.PREFERENCERANK, DP.INITDTTM, DP.INITBY, DP.UPDTTM, 'C/Stock',DP.REASONCODE,
                                       /*++*/NVL(DP.OPTION_CODE,0)+BIN_TO_NUM(1)/*++*/);

                            --! INSERT 중복시 예외 처리 !--
                            EXCEPTION WHEN DUP_VAL_ON_INDEX THEN

                                --! key 값 에러 발생시 MAX solinenum 을 신규 발췌 !--
                                SELECT MAX(SOLINENUM) + 1
                                INTO V_SOLINENUM
                                FROM EXP_SOPROMISESRCNCP
                                WHERE PLANID = DP.PLANID
                                AND SALESORDERID = DP.SALESORDERID;

                                --!  MAX solinenum으로 INSERT 재시도 !--
                                INSERT INTO EXP_SOPROMISESRCNCP
                                       ( ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED,
                                       PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK,
                                       DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY,
                                       NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM,
                                       AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE)
                                VALUES
                                       (DP.ENTERPRISE, DP.SOPROMISEID, DP.PLANID, DP.SALESORDERID,  /*++ MAX SOLINENUM으로 삽입++*/ V_SOLINENUM /*++*/, DP.ITEM, /*++*/DP.QTYPROMISED-V_DPGUIDE/*++*/,
                                       DP.PROMISEDDELDATE, DP.SITEID, DP.SHIPTOID, DP.SALESID, DP.SALESLEVEL, DP.DEMANDTYPE, DP.WEEK, DP.CHANNELRANK, DP.CUSTOMERRANK, DP.PRODUCTRANK,
                                       DP.DEMANDPRIORITY, DP.TIEBREAK, DP.GBM, DP.GLOBALPRIORITY, DP.LOCALPRIORITY, DP.BUSINESSTYPE, DP.ROUTING_PRIORITY,
                                       DP.NO_SPLIT, DP.MAP_SATISFY_SS, DP.PREALLOC_ATTRIBUTE, DP.BUILDAHEADTIME, DP.TIMEUOM,
                                       DP.AP2ID, DP.GC, DP.MEASURERANK, DP.PREFERENCERANK, DP.INITDTTM, DP.INITBY, DP.UPDTTM, 'C/Stock',DP.REASONCODE,
                                       /*++*/NVL(DP.OPTION_CODE,0)+BIN_TO_NUM(1)/*++*/);

--                                DBMS_OUTPUT.PUT_LINE('2-2-1 CHANNEL STOCK 4 LEVEL SOLINENUM 중복발생 MAX+1 : '||DP.SALESORDERID||'***SOLINNUM'||V_SOLINENUM);
                            END;

                            --! 가이드 수량값만큼은 일반 DP로 !--
                            UPDATE EXP_SOPROMISESRCNCP
                            SET    QTYPROMISED = V_DPGUIDE
                            WHERE  ROWID = DP.ROWA;

                            --! 모두 소진 했으므로 Guide 0 처리 !--
                            V_DPGUIDE := 0;

                            COMMIT;
                        END IF;
                    --! 예외 처리 !--
                    EXCEPTION
                         WHEN NO_DATA_FOUND THEN
                         DBMS_OUTPUT.PUT_LINE('2-2-1 CHANNEL STOCK 4 LEVEL Error (DP_GUIDE > 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-2-1 CHANNEL STOCK 4 LEVEL DP.INFO  '||DP.SALESORDERID);
                         ROLLBACK;
                         WHEN OTHERS THEN
                         DBMS_OUTPUT.PUT_LINE('2-2-1 CHANNEL STOCK 4 LEVEL Error (DP_GUIDE > 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-2-1 CHANNEL STOCK 4 LEVEL DP.INFO  '||DP.SALESORDERID);
                         ROLLBACK;
                    END;

                --! DP GUIDE 수량이 0으로 들어올 경우 !--
                ELSIF V_DPGUIDE = 0 THEN
                    BEGIN
                        --! 가이드를 모두 소진하여 0 이면 나머지는 전부다 Channel Stock Short !--
                        UPDATE EXP_SOPROMISESRCNCP
                        SET    OPTION_CODE =  NVL(OPTION_CODE,0) +BIN_TO_NUM(1)
                               ,UPBY = 'C/Stock'
                        WHERE  ROWID = DP.ROWA;

                        COMMIT;
                    --! 예외 처리 !--
                    EXCEPTION
                         WHEN NO_DATA_FOUND THEN
                         DBMS_OUTPUT.PUT_LINE('2-2-2 CHANNEL STOCK 4 LEVEL Error (DP_GUIDE = 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-2-2 CHANNEL STOCK 4 LEVEL DP.INFO  '||DP.SALESORDERID||'**'||DP.PLANID||'**'||DP.SOLINENUM||'**'||DP.PROMISEDDELDATE);
                         ROLLBACK;
                         WHEN OTHERS THEN
                         DBMS_OUTPUT.PUT_LINE('2-2-2 CHANNEL STOCK 4 LEVEL Error (DP_GUIDE = 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-2-2 CHANNEL STOCK 4 LEVEL DP.INFO  '||DP.SALESORDERID||'**'||DP.PLANID||'**'||DP.SOLINENUM||'**'||DP.PROMISEDDELDATE);
                         ROLLBACK;
                    END;
                END IF;
            END LOOP;
        END LOOP;

    END;


    BEGIN
        --! 3) 3레벨 단위 차감 !--
        FOR i IN dpguide_ap2 LOOP
        exit WHEN dpguide_ap2%notfound;

            V_DPGUIDE := i.QTY;
            FOR DP IN (
                SELECT /*+ index( A X2_EXP_SOPROMISESRCNCP) */ A.ROWID ROWA,  ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, A.ITEM, QTYPROMISED, PROMISEDDELDATE
                , SITEID, SHIPTOID, A.SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK
                , GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS
                , PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK
                , A.INITDTTM, A.INITBY, A.UPDTTM, A.UPBY, REASONCODE, OPTION_CODE
                FROM EXP_SOPROMISESRCNCP A
                WHERE PLANID = V_PLANID
                    AND A.QTYPROMISED > 0
                    AND A.AP2ID = I.SALESID
                    AND TO_CHAR(PROMISEDDELDATE, 'IYYYIW') = I.SWEEK
                    --인도일 경우 INU -> INS로 교체
                    --인도네시아일 경우 XID -> XSE, 가이드 수량이 INS 또는 XSE로만 들어오기 때문
                    --20191106 INS/INU, XID/XSE 로직 삭제
--                    AND DECODE(A.SITEID, 'L5N0',REPLACE(A.ITEM, 'INU','INS')
--                                       , 'S529',REPLACE(A.ITEM, 'XSE','XID')
--                                       , 'S529WC27',REPLACE(A.ITEM, 'XSE','XID'), A.ITEM) = I.UDAITEM
                    AND A.ITEM = I.UDAITEM

                    --2020.03.26 SIEL B2B Site : SIEL(DELHI)_B2B(L5N0WBN0)의 CH_CONSTRAINT Short 제외(전창민 프로 요청)
                    AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
                    AND (A.ITEM, A.SITEID, A.SALESID) NOT IN (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE)

                --Demand type이 COM_ORD 일 경우 1순위, NEW_ORD 일 경우 2순위로 우선순위 강제 조정 Req By.양효순K
                --인도네시아일 경우 XID가 XSE보다 우선순위가 높음, 나머지 법인은 DEMANDPRIORITY 따라감
                ORDER BY DECODE(FN_EXTRACT(A.SALESORDERID,'::',1), 'COM_ORD', 1, 'NEW_ORD', 2, 3),
                         --20191106 INS/INU, XID/XSE 로직 삭제
--                         DECODE(A.SITEID, 'S529', ITEM, 'S529WC27', ITEM, DEMANDPRIORITY),
                         DEMANDPRIORITY, ITEM, SALESORDERID DESC, SALESID DESC, QTYPROMISED DESC, SOLINENUM, SITEID DESC
            )
            LOOP
                --! 가이드 수량 > 0 일때만 수행 !--
                IF V_DPGUIDE > 0 THEN
                    BEGIN
                        --! 가이드 수량 >= Demand값 이면 가이드수량만 차감 !--
                        IF V_DPGUIDE >= DP.QTYPROMISED THEN
                            V_DPGUIDE := V_DPGUIDE - DP.QTYPROMISED;
                        --! 가이드 수량 < Demand값 !--
                        ELSIF V_DPGUIDE < DP.QTYPROMISED THEN
                            --DBMS_OUTPUT.PUT_LINE('3. 가이드 수량 < Demand값  : '||DP.QTYPROMISED);
                            BEGIN
                                --! QTY = Demand 값- 가이드수량 (Cnannel Stock Short), OPTION_CODE + 1 (2진수 첫자리) !--
                                INSERT INTO EXP_SOPROMISESRCNCP
                                       ( ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED,
                                       PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK,
                                       DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY,
                                       NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM,
                                       AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE)
                                VALUES
                                       (DP.ENTERPRISE, DP.SOPROMISEID, DP.PLANID, DP.SALESORDERID, /*++*/DP.SOLINENUM + 1/*++*/, DP.ITEM, /*++*/DP.QTYPROMISED-V_DPGUIDE/*++*/,
                                       DP.PROMISEDDELDATE, DP.SITEID, DP.SHIPTOID, DP.SALESID, DP.SALESLEVEL, DP.DEMANDTYPE, DP.WEEK, DP.CHANNELRANK, DP.CUSTOMERRANK, DP.PRODUCTRANK,
                                       DP.DEMANDPRIORITY, DP.TIEBREAK, DP.GBM, DP.GLOBALPRIORITY, DP.LOCALPRIORITY, DP.BUSINESSTYPE, DP.ROUTING_PRIORITY,
                                       DP.NO_SPLIT, DP.MAP_SATISFY_SS, DP.PREALLOC_ATTRIBUTE, DP.BUILDAHEADTIME, DP.TIMEUOM,
                                       DP.AP2ID, DP.GC, DP.MEASURERANK, DP.PREFERENCERANK, DP.INITDTTM, DP.INITBY, DP.UPDTTM, 'C/Stock',DP.REASONCODE,
                                       /*++*/NVL(DP.OPTION_CODE,0)+BIN_TO_NUM(1)/*++*/);
                            --! INSERT 중복시 예외 처리 !--
                            EXCEPTION WHEN DUP_VAL_ON_INDEX THEN

                                --! key 값 에러 발생시 MAX solinenum 을 신규 발췌 !--
                                SELECT MAX(SOLINENUM) + 1
                                INTO V_SOLINENUM
                                FROM EXP_SOPROMISESRCNCP
                                WHERE PLANID = DP.PLANID
                                AND SALESORDERID = DP.SALESORDERID;

                                --!  MAX solinenum으로 INSERT 재시도 !--
                                INSERT INTO EXP_SOPROMISESRCNCP
                                       ( ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED,
                                       PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK,
                                       DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY,
                                       NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM,
                                       AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE)
                                VALUES
                                       (DP.ENTERPRISE, DP.SOPROMISEID, DP.PLANID, DP.SALESORDERID,  /*++ MAX SOLINENUM으로 삽입++*/ V_SOLINENUM /*++*/, DP.ITEM, /*++*/DP.QTYPROMISED-V_DPGUIDE/*++*/,
                                       DP.PROMISEDDELDATE, DP.SITEID, DP.SHIPTOID, DP.SALESID, DP.SALESLEVEL, DP.DEMANDTYPE, DP.WEEK, DP.CHANNELRANK, DP.CUSTOMERRANK, DP.PRODUCTRANK,
                                       DP.DEMANDPRIORITY, DP.TIEBREAK, DP.GBM, DP.GLOBALPRIORITY, DP.LOCALPRIORITY, DP.BUSINESSTYPE, DP.ROUTING_PRIORITY,
                                       DP.NO_SPLIT, DP.MAP_SATISFY_SS, DP.PREALLOC_ATTRIBUTE, DP.BUILDAHEADTIME, DP.TIMEUOM,
                                       DP.AP2ID, DP.GC, DP.MEASURERANK, DP.PREFERENCERANK, DP.INITDTTM, DP.INITBY, DP.UPDTTM, 'C/Stock',DP.REASONCODE,
                                       /*++*/NVL(DP.OPTION_CODE,0)+BIN_TO_NUM(1)/*++*/);

--                                DBMS_OUTPUT.PUT_LINE('2-2-1 CHANNEL STOCK 3 LEVEL SOLINENUM 중복발생 MAX+1 : '||DP.SALESORDERID||'***SOLINNUM'||V_SOLINENUM);
                            END;

                            --! 가이드 수량값만큼은 일반 DP로 !--
                            UPDATE EXP_SOPROMISESRCNCP
                            SET    QTYPROMISED = V_DPGUIDE
                            WHERE  ROWID = DP.ROWA;

                            --! 모두 소진 했으므로 Guide 0 처리 !--
                            V_DPGUIDE := 0;

                            COMMIT;
                        END IF;
                    --! 예외 처리 !--
                    EXCEPTION
                         WHEN NO_DATA_FOUND THEN
                         DBMS_OUTPUT.PUT_LINE('2-3-1 CHANNEL STOCK 3 LEVEL Error (DP_GUIDE > 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-3-1 CHANNEL STOCK 3 LEVEL DP.INFO  '||DP.SALESORDERID);
                         ROLLBACK;
                         WHEN OTHERS THEN
                         DBMS_OUTPUT.PUT_LINE('2-3-1 CHANNEL STOCK 3 LEVEL Error (DP_GUIDE > 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-3-1 CHANNEL STOCK 3 LEVEL DP.INFO  '||DP.SALESORDERID);
                         ROLLBACK;
                    END;

                --! DP GUIDE 수량이 0으로 들어올 경우 !--
                ELSIF V_DPGUIDE = 0 THEN
                    BEGIN
                        --! 가이드를 모두 소진하여 0 이면 나머지는 전부다 Channel Stock Short !--
                        UPDATE EXP_SOPROMISESRCNCP
                        SET    OPTION_CODE =  NVL(OPTION_CODE,0)+BIN_TO_NUM(1)
                               ,UPBY = 'C/Stock'
                        WHERE  ROWID = DP.ROWA;

                        COMMIT;
                    --! 예외 처리 !--
                    EXCEPTION
                         WHEN NO_DATA_FOUND THEN
                         DBMS_OUTPUT.PUT_LINE('2-3-2 CHANNEL STOCK 3 LEVEL Error (DP_GUIDE = 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-3-2 CHANNEL STOCK 3 LEVEL DP.INFO  '||DP.SALESORDERID||'**'||DP.PLANID||'**'||DP.SOLINENUM||'**'||DP.PROMISEDDELDATE);
                         ROLLBACK;
                         WHEN OTHERS THEN
                         DBMS_OUTPUT.PUT_LINE('2-3-2 CHANNEL STOCK 3 LEVEL Error (DP_GUIDE = 0): check '||SQLCODE||'_'||SQLERRM);
                         DBMS_OUTPUT.PUT_LINE('2-3-2 CHANNEL STOCK 3 LEVEL DP.INFO  '||DP.SALESORDERID||'**'||DP.PLANID||'**'||DP.SOLINENUM||'**'||DP.PROMISEDDELDATE);
                         ROLLBACK;
                    END;
                END IF;
            END LOOP;
        END LOOP;

    END;

    --! STEP3.재고(가용량) 감안하여 CH_STOCK SHORT 수량 살림 !--
     BEGIN
        --! 1) DEMAND와 SITE, ITEM 별 재고 조인하여 우선순위별 정렬 계산 !--
        FOR S IN (
        WITH DEMAND AS ( SELECT ROWID ROWA, PLANID, SALESORDERID, SOLINENUM, ENTERPRISE, SOPROMISEID, ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE, AP1ID
                        FROM EXP_SOPROMISESRCNCP A
                       WHERE PLANID = V_PLANID
                         AND NOT EXISTS (SELECT 'X' FROM v_mta_sellermap WHERE ITEM = A.ITEM AND SITEID = A.SITEID)

                         --2020.03.26 SIEL B2B Site : SIEL(DELHI)_B2B(L5N0WBN0)의 CH_CONSTRAINT Short 제외(전창민 프로 요청)
                         AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
                         AND (A.ITEM, A.SITEID, A.SALESID) NOT IN (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE)

                         --20191106 INS/INU, XID/XSE 로직 삭제
--                         AND DECODE(SITEID  , 'L5N0'    ,REPLACE(ITEM, 'INU','INS')
--                                            , 'S529'    ,REPLACE(ITEM, 'XSE','XID')
--                                            , 'S529WC27',REPLACE(ITEM, 'XSE','XID'), ITEM)
--                              IN (      select REPLACE(REPLACE(UDAITEM,'INU','INS'),'XSE','XID')
--                                        from EXP_DPGUIDE_NC)
                         AND ITEM IN (SELECT UDAITEM
                                      FROM EXP_DPGUIDE_NC)
--                                        (FROM (SELECT DISTINCT CASE WHEN INT =  1 THEN UDAITEM  --INS 인것,,XID인것 정상 , INU나 XSE있어도 됨
--                                                              ELSE REPLACE(REPLACE(UDAITEM,'INS','INU'),'XSE','XID') --INU, XSE도 발췌  REPLACE(REPLACE(UDAITEM,'INS','INU'),'XID','XSE')
--                                                              END UDAITEM FROM EXP_DPGUIDE_NC, COPYT WHERE MEASURE = 'ITEM_FCST' AND INT < 3
--                                             ) A, VUI_ITEMATTB B
--                                        WHERE A.UDAITEM = B.ITEM)
                       UNION ALL
                      SELECT NULL ROWA, PLANID, NULL SALESORDERID, NULL SOLINENUM, NULL ENTERPRISE, NULL SOPROMISEID, ITEM, 0 QTYPROMISED,
                             (SELECT ENDDATE - 1 FROM MST_WEEK WHERE WEEK = A.WEEK) PROMISEDDELDATE, SITEID, NULL SHIPTOID, NULL SALESID, NULL SALESLEVEL, NULL DEMANDTYPE, SUBSTR(WEEK,-2,2) WEEK, NULL CHANNELRANK, NULL CUSTOMERRANK, NULL PRODUCTRANK, NULL DEMANDPRIORITY, NULL TIEBREAK,
                             NULL GBM, NULL GLOBALPRIORITY, NULL LOCALPRIORITY, NULL BUSINESSTYPE, NULL ROUTING_PRIORITY, NULL NO_SPLIT, NULL MAP_SATISFY_SS, NULL PREALLOC_ATTRIBUTE, NULL BUILDAHEADTIME,
                             NULL TIMEUOM, NULL AP2ID, NULL GC, NULL MEASURERANK, NULL PREFERENCERANK, NULL INITDTTM, NULL INITBY, NULL UPDTTM, NULL UPBY, NULL REASONCODE,
                             NULL OPTION_CODE, NULL AP1ID
                        FROM MST_INVENTORY_DNE A
                        WHERE PLANID = V_PLANID
                        AND NOT EXISTS (SELECT 'X' FROM v_mta_sellermap WHERE ITEM = A.ITEM AND SITEID = A.SITEID)

                        --2020.03.26 SIEL B2B Site : SIEL(DELHI)_B2B(L5N0WBN0)의 CH_CONSTRAINT Short 제외(전창민 프로 요청)
                        AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')

                       )
                       ,
        DEMAND_SELL AS ( SELECT ROWID ROWA, PLANID, SALESORDERID, SOLINENUM, ENTERPRISE, SOPROMISEID, ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE, AP1ID
                        FROM EXP_SOPROMISESRCNCP A
                       WHERE PLANID = V_PLANID
                       AND EXISTS (SELECT 'X' FROM v_mta_sellermap WHERE ITEM = A.ITEM AND SITEID = A.SITEID)

                       --2020.03.26 SIEL B2B Site : SIEL(DELHI)_B2B(L5N0WBN0)의 CH_CONSTRAINT Short 제외(전창민 프로 요청)
                       AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')

                       --20191106 INS/INU, XID/XSE 로직 삭제
--                       AND DECODE(SITEID  , 'L5N0'    ,REPLACE(ITEM, 'INU','INS')
--                                          , 'S529'    ,REPLACE(ITEM, 'XSE','XID')
--                                          , 'S529WC27',REPLACE(ITEM, 'XSE','XID'), ITEM)
--                                IN (select REPLACE(REPLACE(UDAITEM,'INU','INS'),'XSE','XID')
--                                        from EXP_DPGUIDE_NC)
                       AND ITEM IN (SELECT UDAITEM
                                    FROM EXP_DPGUIDE_NC)
--                                        (SELECT UDAITEM
--                                    FROM (SELECT DISTINCT CASE WHEN INT =  1 THEN UDAITEM
--                                                          ELSE REPLACE(REPLACE(UDAITEM,'INS','INU'),'XSE','XID')
--                                                          END UDAITEM FROM EXP_DPGUIDE_NC, COPYT WHERE MEASURE = 'ITEM_FCST' AND INT < 3
--                                         ) A, VUI_ITEMATTB B
--                                    WHERE A.UDAITEM = B.ITEM)
                       UNION ALL
                      SELECT NULL ROWA, PLANID, NULL SALESORDERID, NULL SOLINENUM, NULL ENTERPRISE, NULL SOPROMISEID, ITEM, 0 QTYPROMISED,
                             (SELECT ENDDATE - 1 FROM MST_WEEK WHERE WEEK = A.WEEK) PROMISEDDELDATE, SITEID, NULL SHIPTOID, NULL SALESID, NULL SALESLEVEL, NULL DEMANDTYPE, SUBSTR(WEEK,-2,2) WEEK, NULL CHANNELRANK, NULL CUSTOMERRANK, NULL PRODUCTRANK, NULL DEMANDPRIORITY, NULL TIEBREAK,
                             NULL GBM, NULL GLOBALPRIORITY, NULL LOCALPRIORITY, NULL BUSINESSTYPE, NULL ROUTING_PRIORITY, NULL NO_SPLIT, NULL MAP_SATISFY_SS, NULL PREALLOC_ATTRIBUTE, NULL BUILDAHEADTIME,
                             NULL TIMEUOM, SALESID AP2ID, NULL GC, NULL MEASURERANK, NULL PREFERENCERANK, NULL INITDTTM, NULL INITBY, NULL UPDTTM, NULL UPBY, NULL REASONCODE,
                             NULL OPTION_CODE, NULL AP1ID
                        FROM MST_INVENTORY_DNE A
                        WHERE PLANID = V_PLANID
                          AND EXISTS (SELECT 'X' FROM v_mta_sellermap WHERE ITEM = A.ITEM AND SITEID = A.SITEID)

                          --2020.03.26 SIEL B2B Site : SIEL(DELHI)_B2B(L5N0WBN0)의 CH_CONSTRAINT Short 제외(전창민 프로 요청)
                          AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')

                       )
        --통합물류 외
            SELECT * FROM (
              SELECT ROWA, PLANID, SALESORDERID, SOLINENUM, ENTERPRISE, SOPROMISEID, ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK,
                     CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS,
                     PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, REASONCODE, OPTION_CODE , AP1ID, UPBY,
                     AVAILQTY, PART_CNT, PART2_CNT,  ALL_CNT, SUM_DEMAND, GREATEST(AVAILQTY - SUM_DEMAND,0) REMAINAVAIL
                FROM ( SELECT A.ROWA,
                              A.PLANID, A.SALESORDERID, A.SOLINENUM, A.ENTERPRISE, A.SOPROMISEID, A.ITEM, A.QTYPROMISED, A.PROMISEDDELDATE, A.SITEID, A.SHIPTOID, A.SALESID, A.SALESLEVEL, A.DEMANDTYPE, A.WEEK,
                              A.CHANNELRANK, A.CUSTOMERRANK, A.PRODUCTRANK, A.DEMANDPRIORITY, A.TIEBREAK, A.GBM, A.GLOBALPRIORITY, A.LOCALPRIORITY, A.BUSINESSTYPE, A.ROUTING_PRIORITY, A.NO_SPLIT, A.MAP_SATISFY_SS,
                              A.PREALLOC_ATTRIBUTE, A.BUILDAHEADTIME, A.TIMEUOM, A.AP2ID, A.GC, A.MEASURERANK, A.PREFERENCERANK, A.REASONCODE, A.OPTION_CODE , A.AP1ID, A.UPBY,
                              NVL(B.QTY,0) AVAILQTY,
                              ROW_NUMBER() OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.PROMISEDDELDATE ORDER BY NVL(A.DEMANDPRIORITY,0) ASC ) PART_CNT,
                              COUNT(*) OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.PROMISEDDELDATE) PART2_CNT,
                              ROW_NUMBER() OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID ORDER BY A.PROMISEDDELDATE, NVL(A.DEMANDPRIORITY,0) ASC, MOD(NVL(A.OPTION_CODE,0), 2) ASC, a.qtypromised, a.siteid, a.salesid ) ALL_CNT,
                              NVL(SUM(A.QTYPROMISED) OVER (PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.PROMISEDDELDATE ORDER BY NVL(A.DEMANDPRIORITY,0) ASC ROWS UNBOUNDED PRECEDING ) ,0) SUM_DEMAND
                         FROM DEMAND A, MST_INVENTORY_DNE B
                        WHERE A.PLANID = B.PLANID(+)
                          AND A.SITEID = B.SITEID(+)
                          AND A.ITEM   = B.ITEM(+)
                          AND TO_CHAR(A.PROMISEDDELDATE,'IYYYIW') = B.WEEK(+)
                          AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
                          )
               ORDER BY PLANID, ITEM, SITEID, PROMISEDDELDATE, ALL_CNT)
               UNION ALL
            --통합물류
            SELECT * FROM (
              SELECT ROWA, PLANID, SALESORDERID, SOLINENUM, ENTERPRISE, SOPROMISEID, ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK,
                     CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS,
                     PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, REASONCODE, OPTION_CODE , AP1ID, UPBY,
                     AVAILQTY, PART_CNT, PART2_CNT,  ALL_CNT, SUM_DEMAND, GREATEST(AVAILQTY - SUM_DEMAND,0) REMAINAVAIL
                FROM ( SELECT A.ROWA,
                              A.PLANID, A.SALESORDERID, A.SOLINENUM, A.ENTERPRISE, A.SOPROMISEID, A.ITEM, A.QTYPROMISED, A.PROMISEDDELDATE, A.SITEID, A.SHIPTOID, A.SALESID, A.SALESLEVEL, A.DEMANDTYPE, A.WEEK,
                              A.CHANNELRANK, A.CUSTOMERRANK, A.PRODUCTRANK, A.DEMANDPRIORITY, A.TIEBREAK, A.GBM, A.GLOBALPRIORITY, A.LOCALPRIORITY, A.BUSINESSTYPE, A.ROUTING_PRIORITY, A.NO_SPLIT, A.MAP_SATISFY_SS,
                              A.PREALLOC_ATTRIBUTE, A.BUILDAHEADTIME, A.TIMEUOM, A.AP2ID, A.GC, A.MEASURERANK, A.PREFERENCERANK, A.REASONCODE, A.OPTION_CODE , A.AP1ID, A.UPBY,
                              NVL(B.QTY,0) AVAILQTY,
                              ROW_NUMBER() OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.AP2ID, A.PROMISEDDELDATE ORDER BY NVL(A.DEMANDPRIORITY,0) ASC ) PART_CNT,
                              COUNT(*) OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.AP2ID, A.PROMISEDDELDATE) PART2_CNT,
                              ROW_NUMBER() OVER(PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.AP2ID ORDER BY A.PROMISEDDELDATE, NVL(A.DEMANDPRIORITY,0) ASC, MOD(NVL(A.OPTION_CODE,0), 2) ASC, a.qtypromised, a.siteid, a.salesid ) ALL_CNT,
                              NVL(SUM(A.QTYPROMISED) OVER (PARTITION BY A.PLANID, A.ITEM, A.SITEID, A.AP2ID, A.PROMISEDDELDATE ORDER BY NVL(A.DEMANDPRIORITY,0) ASC ROWS UNBOUNDED PRECEDING ) ,0) SUM_DEMAND
                        FROM DEMAND_SELL A, MST_INVENTORY_DNE B
                        WHERE A.PLANID = B.PLANID(+)
                          AND A.SITEID = B.SITEID(+)
                          AND A.AP2ID  = B.SALESID(+)
                          AND A.ITEM   = B.ITEM(+)
                          AND TO_CHAR(A.PROMISEDDELDATE,'IYYYIW') = B.WEEK(+)
                          AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
                          )
               ORDER BY PLANID, ITEM, SITEID, AP2ID, PROMISEDDELDATE, ALL_CNT)
         )
        LOOP
            --! 새로운 모델/Site 인 경우 잔량 0처리 !--
            IF S.ALL_CNT = 1 THEN
            V_REMAINQTY := 0;
            END IF;

            --! 주차가 바뀌는 경우 앞에 남은 수량은 뒤로 Move !--
            IF S.PART_CNT = 1 THEN
            V_AVAILQTY := S.AVAILQTY + V_REMAINQTY;
            END IF;

            --! 가용량이 Demand 수량 합보다 큰 경우인데 C/Stock 일 경우 전량 살림 !--
            IF V_AVAILQTY >= S.SUM_DEMAND AND S.QTYPROMISED > 0 AND  MOD(NVL(S.OPTION_CODE,0), 2) = 1 THEN
                UPDATE EXP_SOPROMISESRCNCP
                SET    OPTION_CODE =  NVL(S.OPTION_CODE,0)-BIN_TO_NUM(1)
                       ,UPBY = 'C/Stock_AVAIL'
                WHERE  ROWID = S.ROWA;
            ELSE
                --! 가용량이 누적합보다는 작으나 일부 물량을 살릴 수 있는 경우 !--
                IF V_AVAILQTY - (S.SUM_DEMAND - S.QTYPROMISED) > 0 AND S.QTYPROMISED > 0 AND  MOD(NVL(S.OPTION_CODE,0), 2) = 1 THEN
                    --! 부분 수량 보전 !--
                    -- 현재 ROW에서 쓸 수 있는 남은 가용량 = V_AVAILQTY - (S.SUM_DEMAND - S.QTYPROMISED)
                    UPDATE EXP_SOPROMISESRCNCP
                       SET OPTION_CODE =  NVL(S.OPTION_CODE,0)-BIN_TO_NUM(1),
                           QTYPROMISED = V_AVAILQTY - (S.SUM_DEMAND - S.QTYPROMISED),
                           UPDTTM = SYSDATE,
                           UPBY   = 'C/Stock_AVAIL'
                     WHERE ROWID = S.ROWA;

                    --! 부분 수량 삭제 !--
                    BEGIN
                        --! 가용량을 제외한 나머지 수량은 그대로 C/Stock Short !--
                        INSERT INTO EXP_SOPROMISESRCNCP (PLANID, SALESORDERID, SOLINENUM, ENTERPRISE, SOPROMISEID, ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK,
                                                      CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS,
                                                      PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, UPDTTM, UPBY, REASONCODE, OPTION_CODE, AP1ID)
                             VALUES (S.PLANID, S.SALESORDERID, /* -- */ S.SOLINENUM + 1 /* -- */, S.ENTERPRISE, S.SOPROMISEID, S.ITEM, /* -- */ S.SUM_DEMAND - V_AVAILQTY /* -- */, S.PROMISEDDELDATE, S.SITEID, S.SHIPTOID, S.SALESID, S.SALESLEVEL, S.DEMANDTYPE, S.WEEK,
                                     S.CHANNELRANK, S.CUSTOMERRANK, S.PRODUCTRANK, S.DEMANDPRIORITY, S.TIEBREAK, S.GBM, S.GLOBALPRIORITY, S.LOCALPRIORITY, S.BUSINESSTYPE, S.ROUTING_PRIORITY, S.NO_SPLIT, S.MAP_SATISFY_SS,
                                     S.PREALLOC_ATTRIBUTE, S.BUILDAHEADTIME, S.TIMEUOM, S.AP2ID, S.GC, S.MEASURERANK, S.PREFERENCERANK, SYSDATE, S.UPBY, S.REASONCODE, S.OPTION_CODE, S.AP1ID);
                    --! SOLINENUM 중복시 예외처리 !--
                    EXCEPTION WHEN DUP_VAL_ON_INDEX THEN

                        --! SOLINENUM 의 MAX값을 찾음 !--
                        SELECT MAX(SOLINENUM) + 1
                          INTO V_SOLINENUM
                          FROM EXP_SOPROMISESRCNCP
                         WHERE PLANID = S.PLANID
                           AND SALESORDERID = S.SALESORDERID;

                        --! MAX(SOLINENUM) 값으로 INSERT 재시도 !--
                        INSERT INTO EXP_SOPROMISESRCNCP (PLANID, SALESORDERID, SOLINENUM, ENTERPRISE, SOPROMISEID, ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK,
                                                      CHANNELRANK, CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS,
                                                      PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, MEASURERANK, PREFERENCERANK, UPDTTM, UPBY, REASONCODE, OPTION_CODE, AP1ID)
                             VALUES (S.PLANID, S.SALESORDERID, /* -- */ V_SOLINENUM /* -- */, S.ENTERPRISE, S.SOPROMISEID, S.ITEM, /* -- */ S.SUM_DEMAND - V_AVAILQTY /* -- */, S.PROMISEDDELDATE, S.SITEID, S.SHIPTOID, S.SALESID, S.SALESLEVEL, S.DEMANDTYPE, S.WEEK,
                                     S.CHANNELRANK, S.CUSTOMERRANK, S.PRODUCTRANK, S.DEMANDPRIORITY, S.TIEBREAK, S.GBM, S.GLOBALPRIORITY, S.LOCALPRIORITY, S.BUSINESSTYPE, S.ROUTING_PRIORITY, S.NO_SPLIT, S.MAP_SATISFY_SS,
                                     S.PREALLOC_ATTRIBUTE, S.BUILDAHEADTIME, S.TIMEUOM, S.AP2ID, S.GC, S.MEASURERANK, S.PREFERENCERANK, SYSDATE, S.UPBY, S.REASONCODE, S.OPTION_CODE, S.AP1ID);

                    END;

                END IF;
            END IF;

            --! 주차가 끝나는 시점에서 사용되고 남은 가용량 차주로 Move !--
            IF S.PART_CNT = S.PART2_CNT THEN
             V_REMAINQTY := CASE WHEN V_AVAILQTY - S.SUM_DEMAND < 0 THEN 0 ELSE V_AVAILQTY - S.SUM_DEMAND END;
            END IF;

        END LOOP;
     --! 예외 처리 !--
    EXCEPTION
        WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('3. CHANNEL STOCK AVAIL Error (DP_GUIDE > 0)'||SQLCODE||'_'||SQLERRM);
        ROLLBACK;
        WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('3. CHANNEL STOCK AVAIL Error (DP_GUIDE > 0)'||SQLCODE||'_'||SQLERRM);
        ROLLBACK;
    END;


    --! 3,4 레벨 버퍼수량 전량 C/Stock Short !--
    BEGIN
        --! DP_GUIDE가 5레벨일때, 3/4레벨 전량 Short !--
        UPDATE /*+ index( A X2_EXP_SOPROMISESRCNCP) */ EXP_SOPROMISESRCNCP A
        SET    OPTION_CODE =  NVL(OPTION_CODE,0) +BIN_TO_NUM(1)
               ,UPBY = 'C/Stock_Upper'
        WHERE PLANID = V_PLANID
        AND (AP2ID,TO_CHAR(PROMISEDDELDATE, 'IYYYIW'),
                                                                      --20191106 INS/INU, XID/XSE 로직 삭제
--                                                                    DECODE(SITEID, 'L5N0',REPLACE(ITEM, 'INU','INS')
--                                                                    , 'S529',REPLACE(ITEM, 'XSE','XID')
--                                                                    , 'S529WC27',REPLACE(ITEM, 'XSE','XID'), ITEM)
            ITEM
            )
            IN ( SELECT B.AP2ID, A.SWEEK,
                --20191106 INS/INU, XID/XSE 로직 삭제
--                DECODE(B.AP2ID , '300143', REPLACE(A.UDAITEM ,'XSE','XID'), '300147',REPLACE(A.UDAITEM, 'INU','INS'), A.UDAITEM)
                  A.UDAITEM
                   FROM EXP_DPGUIDE_NC A, GUI_SALESHIERARCHY B
                  WHERE A.SALESID = B.SALESID
                    AND B.LEVELID = 5)
        AND SALESID NOT LIKE '5%'
        --2020.03.26 SIEL B2B Site : SIEL(DELHI)_B2B(L5N0WBN0)의 CH_CONSTRAINT Short 제외(전창민 프로 요청)
        AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
        AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
        AND (A.ITEM, A.SITEID, A.SALESID) NOT IN (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE);

    --! 예외 처리 !--
    EXCEPTION
        WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('4-1. CHANNEL STOCK 5 LEVEL GUIDE, 3,4 LEVEL Update Error'||SQLCODE||'_'||SQLERRM);
        ROLLBACK;
        WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('4-1. CHANNEL STOCK 5 LEVEL GUIDE, 3,4 LEVEL Update Error'||SQLCODE||'_'||SQLERRM);
        ROLLBACK;
    END;

    BEGIN
        --! DP_GUDIE가 4레벨일때, 3레벨 전량 Short !--
        UPDATE /*+ index(A X2_EXP_SOPROMISESRCNCP) */ EXP_SOPROMISESRCNCP A
        SET    OPTION_CODE =  NVL(OPTION_CODE,0) +BIN_TO_NUM(1)
               ,UPBY = 'C/Stock_Upper'
        WHERE PLANID = V_PLANID
        AND (SALESID,TO_CHAR(PROMISEDDELDATE, 'IYYYIW'),
                                                                    --20191106 INS/INU, XID/XSE 로직 삭제
--                                                                    DECODE(SITEID, 'L5N0',REPLACE(ITEM, 'INU','INS')
--                                                                     , 'S529',REPLACE(ITEM, 'XSE','XID')
--                                                                     , 'S529WC27',REPLACE(ITEM, 'XSE','XID'), ITEM)
            ITEM
            )
            IN ( SELECT B.AP2ID, A.SWEEK,
                        --20191106 INS/INU, XID/XSE 로직 삭제
--                        DECODE(B.AP2ID , '300143', REPLACE(A.UDAITEM ,'XSE','XID'), '300147',REPLACE(A.UDAITEM, 'INU','INS'), A.UDAITEM)
                        A.UDAITEM
                   FROM EXP_DPGUIDE_NC A, GUI_SALESHIERARCHY B
                  WHERE A.SALESID = B.SALESID
                    AND B.LEVELID = 4)
        --2020.03.26 SIEL B2B Site : SIEL(DELHI)_B2B(L5N0WBN0)의 CH_CONSTRAINT Short 제외(전창민 프로 요청)
        AND A.SITEID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY='HC_WL_DP_CHSTOCK_EXCEPT')
        AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
        AND (A.ITEM, A.SITEID, A.SALESID) NOT IN (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE);

    --! 예외 처리 !--
    EXCEPTION
        WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('4-2. CHANNEL STOCK 4 LEVEL GUIDE, 3 LEVEL Update Error'||SQLCODE||'_'||SQLERRM);
        ROLLBACK;
        WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('4-2. CHANNEL STOCK 4 LEVEL GUIDE, 3 LEVEL Update Error'||SQLCODE||'_'||SQLERRM);
        ROLLBACK;
    END;

    COMMIT;


    BEGIN
         --! COM_ORD 일 경우 C/Stock 제외 (최종 결과에서 COM_ORD가 짤렸을 경우 다시 살려줌) !--
         --SEA 이외 법인은 구간 상관없이 C/Stock 제외 17.02.02
         --VZW 예외 로직 운영으로 ACCOUNT별 셋팅 기능 추가 19.07.22
        UPDATE EXP_SOPROMISESRCNCP A
        SET    A.OPTION_CODE =  NVL(OPTION_CODE,0)-BIN_TO_NUM(1)
               ,A.UPBY = 'C/Stock_COM_ORD'
        WHERE MOD(NVL(A.OPTION_CODE,0), 2) = 1
        AND A.AP2ID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY = 'HC_WL_DP_COMNEW_CHSTOCK')
        AND A.SALESID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY = 'HC_WL_DP_COMNEW_CHSTOCK_ACNT')
        AND FN_EXTRACT(A.SALESORDERID, '::',1) IN ( 'COM_ORD','NEW_ORD')
        AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
        AND (A.ITEM, A.SITEID, A.SALESID) NOT IN (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE);

        --SEA 법인은 4주구간만 C/Stock 제외 17.02.02
        --2017-04-03 장재황D SEA, SSA, AF_SOUTH(MAURITIUS, ZAMBIA) 대해 4--> 3W내 제약 제외 요청

        --2018.12.03 수정(V PLAN 시 당주기준으로 PROMISEDDELDATE 변경)
        UPDATE EXP_SOPROMISESRCNCP
        SET    OPTION_CODE =  NVL(OPTION_CODE,0)-BIN_TO_NUM(1)
               ,UPBY = 'C/Stock_COM_ORD'
        WHERE ROWID IN (SELECT A.ROWID ROWIDS
                        FROM   EXP_SOPROMISESRCNCP A, MTA_CODEMAP B
                        --        SET    OPTION_CODE =  NVL(OPTION_CODE,0)-BIN_TO_NUM(1)
                        --               ,UPBY = 'C/Stock_COM_ORD'
                        WHERE  MOD(NVL(OPTION_CODE,0), 2) = 1
                        AND    A.AP2ID = CODE1
                        AND    B.CATEGORY = 'HC_WL_DP_COMNEW_CHSTOCK'
                        AND    PROMISEDDELDATE < SYSDATE + ((NUM1+V_NUM)*7)
                        AND    FN_EXTRACT(SALESORDERID, '::',1) IN ( 'COM_ORD','NEW_ORD')
                        AND    NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
                        AND    (A.ITEM, A.SITEID, A.SALESID) NOT IN (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE)
                        );


        --VZW 예외 로직 운영으로 ACCOUNT별 셋팅 기능 추가 19.07.22
        UPDATE EXP_SOPROMISESRCNCP
        SET    OPTION_CODE =  NVL(OPTION_CODE,0)-BIN_TO_NUM(1)
               ,UPBY = 'C/Stock_COM_ORD'
        WHERE ROWID IN (SELECT A.ROWID ROWIDS
                        FROM   EXP_SOPROMISESRCNCP A, MTA_CODEMAP B
                        --        SET    OPTION_CODE =  NVL(OPTION_CODE,0)-BIN_TO_NUM(1)
                        --               ,UPBY = 'C/Stock_COM_ORD'
                        WHERE  MOD(NVL(OPTION_CODE,0), 2) = 1
                        AND    A.SALESID = CODE1
                        AND    B.CATEGORY = 'HC_WL_DP_COMNEW_CHSTOCK_ACNT'
                        AND    PROMISEDDELDATE < SYSDATE + ((NUM1+V_NUM)*7)
                        AND    FN_EXTRACT(SALESORDERID, '::',1) IN ( 'COM_ORD','NEW_ORD')
                        AND    NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
                        AND    (A.ITEM, A.SITEID, A.SALESID) NOT IN (SELECT ITEM, SITEID, SALESID FROM MVES_ITEMSITE)
                        );

--        --! NEW_ORD 일 경우 C/Stock 제외 (최종 결과에서 NEW_ORD가 짤렸을 경우 다시 살려줌) !--
--         --SEA 이외 법인은 구간 상관없이 C/Stock 제외 17.02.02
--        UPDATE EXP_SOPROMISESRCNCP
--        SET    OPTION_CODE =  NVL(OPTION_CODE,0)-BIN_TO_NUM(1)
--               ,UPBY = 'C/Stock_NEW_ORD'
--        WHERE MOD(NVL(OPTION_CODE,0), 2) = 1
--        AND AP2ID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY = 'HC_WL_DP_COMNEW_CHSTOCK')
--        AND FN_EXTRACT(SALESORDERID, '::',1) = 'NEW_ORD';

--        --SEA 법인은 4주구간만 C/Stock 제외 17.02.02
--         --2017-04-03 장재황D SEA, SSA, AF_SOUTH(MAURITIUS, ZAMBIA) 대해 4--> 3W내 제약 제외 요청
--        UPDATE EXP_SOPROMISESRCNCP
--        SET    OPTION_CODE =  NVL(OPTION_CODE,0)-BIN_TO_NUM(1)
--               ,UPBY = 'C/Stock_NEW_ORD'
--        WHERE MOD(NVL(OPTION_CODE,0), 2) = 1
--        AND AP2ID IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY = 'HC_WL_DP_COMNEW_CHSTOCK')
--        AND PROMISEDDELDATE < SYSDATE + 4*7
--        AND FN_EXTRACT(SALESORDERID, '::',1) = 'NEW_ORD';

        COMMIT;

    --! 예외 처리 !--
    EXCEPTION
         WHEN NO_DATA_FOUND THEN
         DBMS_OUTPUT.PUT_LINE('5. CHANNEL STOCK COM_ORD Error: check '||SQLCODE||'_'||SQLERRM);
         ROLLBACK;
         WHEN OTHERS THEN
         DBMS_OUTPUT.PUT_LINE('5. CHANNEL STOCK COM_ORD Error: check '||SQLCODE||'_'||SQLERRM);
    END;


    -- CH.STOCK으로 잘린 것들 UNF_ORD 수량만큼 살려준다 2016.06.08
    BEGIN
        FOR SS IN( SELECT SALESID, SITEID, ITEM, PROMISEDDELDATE, UNFORDQTY, CHSTOCKQTY,  UNFORDSUM, CHSTOCKSUM, RNK
                     FROM (
                            SELECT SALESID , SITEID , ITEM , DUEDATE PROMISEDDELDATE , UNFORDQTY , CHSTOCKQTY
                                   , SUM(UNFORDQTY) OVER (PARTITION BY SALESID, SITEID, ITEM) UNFORDSUM
                                   , SUM(CHSTOCKQTY) OVER (PARTITION BY SALESID, SITEID, ITEM) CHSTOCKSUM
                                   , RNK
                              FROM (
                                    --UNF_ORD 감안 CH_STOCK 제약 복구시 구간 제어 기능 추가 17.08.17
                                    SELECT SALESID, SITEID, ITEM, DUEDATE, SUM(UNFORDQTY) UNFORDQTY, SUM(CHSTOCKQTY) CHSTOCKQTY
                                                                               , RANK() OVER (PARTITION BY SALESID , SITEID, ITEM ORDER BY DUEDATE ) RNK
                                      FROM (
                                            --2018.12.03 수정(V PLAN 시 당주기준으로 DUEDATE 변경)
                                            SELECT REGEXP_SUBSTR(A.SALESORDERID, '[^::]+', 1, 6) SALESID , A.SITEID, A.ITEM , A.DUEDATE , A.SHORTQTY UNFORDQTY , 0 CHSTOCKQTY
                                              FROM EXP_AP1_SHORTREASON A, MTA_CODEMAP B
                                             WHERE PLANID = V_PLANID
                                               AND REGEXP_SUBSTR(A.SALESORDERID, '[^::]+', 1, 1) = 'UNF_ORD'
                                               AND B.CATEGORY(+) IN ( 'HC_WL_DP_UNFORD_CHSTOCK','HC_WL_DP_UNFORD_CHSTOCK_ACNT')
                                               AND DECODE(B.CATEGORY(+),'HC_WL_DP_UNFORD_CHSTOCK'
                                                                       , REGEXP_SUBSTR(A.SALESORDERID, '[^::]+', 1, 5)
                                                                       ,'HC_WL_DP_UNFORD_CHSTOCK_ACNT'
                                                                       , REGEXP_SUBSTR(A.SALESORDERID, '[^::]+', 1, 6)) = B.CODE1(+)
                                               AND A.DUEDATE < SYSDATE + 7 * NVL(B.CODE2_NUM+V_NUM, 8)
                                               AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.REQITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
                                    --           AND REGEXP_SUBSTR(A.SALESORDERID, '[^::]+', 1, 1) = 'UNF_ORD'
                                    --           AND DUEDATE < SYSDATE + 7*8
                                    --           AND FN_EXTRACT(SALESORDERID, '::', 5) NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY = 'HC_WL_DP_UNFORD_CHSTOCK')   --SEM 법인 제외요청 16.12.22
                                             UNION ALL
                                            SELECT A.SALESID, A.SITEID, A.ITEM, A.PROMISEDDELDATE, 0 UNFORDQTY, A.QTYPROMISED CHSTOCKQTY
                                              FROM EXP_SOPROMISESRCNCP A, MTA_CODEMAP B
                                             WHERE A.PLANID = V_PLANID
                                               AND MOD(NVL(A.OPTION_CODE,0), 2) = 1
                                               AND A.SALESID LIKE '5%'
                                               AND B.CATEGORY(+) IN ( 'HC_WL_DP_UNFORD_CHSTOCK','HC_WL_DP_UNFORD_CHSTOCK_ACNT')
                                               AND DECODE(B.CATEGORY(+),'HC_WL_DP_UNFORD_CHSTOCK'
                                                                       , A.AP2ID
                                                                       ,'HC_WL_DP_UNFORD_CHSTOCK_ACNT'
                                                                       , A.SALESID) = B.CODE1(+)
                                               AND A.PROMISEDDELDATE < SYSDATE + 7 * NVL(B.CODE2_NUM+V_NUM, 8)
                                               AND NOT EXISTS (SELECT 'X' FROM MTA_CUSTOMMODELMAP WHERE CUSTOMITEM = A.ITEM AND ISVALID = 'Y') --E-STORE 20200807 추가
                                    --           AND PROMISEDDELDATE < SYSDATE + 7*8
                                    --           AND AP2ID NOT IN (SELECT CODE1 FROM MTA_CODEMAP WHERE CATEGORY = 'HC_WL_DP_UNFORD_CHSTOCK')  --SEM 법인 제외요청 16.12.22
                                           )
                                    GROUP BY SALESID, SITEID, ITEM, DUEDATE
                                   )
                          )
                  WHERE UNFORDSUM <> 0 AND CHSTOCKSUM <> 0
                )
        LOOP

            -- V_AVAILQTY2  가용량
            -- V_ALIVE      살아남을 수 있는 수량
            -- V_ROLLING    가용량 > CHSTOCK인 경우 차주 가용량으로 넘어가는 수량
            -- V_CHSTOCK    가용량 < CHSTOCK 인 경우 다시 CHSTOCK으로 찍혀야 하는 수량

            IF SS.RNK = 1 THEN
                V_AVAILQTY2  := SS.UNFORDQTY;
            ELSE
                V_AVAILQTY2 := SS.UNFORDQTY + V_ROLLING;
            END IF;

            V_ALIVE     := CASE WHEN V_AVAILQTY2 >= SS.CHSTOCKQTY THEN SS.CHSTOCKQTY ELSE V_AVAILQTY2 END;
            V_ROLLING   := CASE WHEN V_AVAILQTY2 - SS.CHSTOCKQTY < 0 THEN 0 ELSE V_AVAILQTY2 - SS.CHSTOCKQTY END;
            V_CHSTOCK   := CASE WHEN SS.CHSTOCKQTY - V_AVAILQTY2 < 0 THEN 0 ELSE SS.CHSTOCKQTY - V_AVAILQTY2 END;

            --가용량이 CHSTOCK 수량보다 많으면 다 살린다
            IF V_AVAILQTY2 >= SS.CHSTOCKQTY THEN

                UPDATE EXP_SOPROMISESRCNCP
                   SET OPTION_CODE = NVL(OPTION_CODE,0)-BIN_TO_NUM(1)
                       , UPBY = 'C/Stock_UNF_ORD'
                 WHERE PLANID = V_PLANID
                   AND MOD(NVL(OPTION_CODE,0), 2) = 1
                   AND SALESID = SS.SALESID
                   AND SITEID = SS.SITEID
                   AND ITEM = SS.ITEM
                   AND PROMISEDDELDATE = SS.PROMISEDDELDATE;

                V_AVAILQTY2 := CASE WHEN V_AVAILQTY2 - SS.CHSTOCKQTY < 0 THEN 0 ELSE V_AVAILQTY2 - SS.CHSTOCKQTY END;


            ELSIF V_AVAILQTY2 < SS.CHSTOCKQTY AND V_AVAILQTY2 > 0 THEN

               --- 가용량이 CHSTOCKQTY보다 적으면, 순위대로 짤라서 살려준다
               FOR QQ IN (SELECT ROW_NUMBER() OVER (ORDER BY DEMANDPRIORITY, QTYPROMISED) RNK, A.*
                            FROM EXP_SOPROMISESRCNCP A
                           WHERE PLANID = V_PLANID
                             AND SALESID = SS.SALESID
                             AND SITEID = SS.SITEID
                             AND ITEM = SS.ITEM
                             AND PROMISEDDELDATE = SS.PROMISEDDELDATE
                             AND MOD(NVL(OPTION_CODE,0), 2) = 1
                           ORDER BY DEMANDPRIORITY, QTYPROMISED
                          )
                          LOOP

                             IF V_AVAILQTY2 >= QQ.QTYPROMISED THEN

                                UPDATE EXP_SOPROMISESRCNCP
                                   SET OPTION_CODE = NVL(OPTION_CODE,0)-BIN_TO_NUM(1)
                                       , UPBY = 'C/Stock_UNF_ORD_PART1'
                                 WHERE PLANID = V_PLANID
                                   AND SALESORDERID = QQ.SALESORDERID
                                   AND SOLINENUM = QQ.SOLINENUM
                                   AND MOD(NVL(OPTION_CODE,0), 2) = 1;

                                   V_AVAILQTY2 := CASE WHEN V_AVAILQTY2 - QQ.QTYPROMISED <0 THEN 0 ELSE V_AVAILQTY2 - QQ.QTYPROMISED END ;

                             ELSIF V_AVAILQTY2 < QQ.QTYPROMISED AND V_AVAILQTY2 > 0 THEN

                                UPDATE EXP_SOPROMISESRCNCP
                                SET QTYPROMISED = QQ.QTYPROMISED - V_AVAILQTY2
                                    , UPBY = 'C/Stock_UNF_ORD_PART3'||V_AVAILQTY2
                                    , OPTION_CODE = NVL(QQ.OPTION_CODE,0)
                                WHERE PLANID = V_PLANID
                                   AND SALESORDERID = QQ.SALESORDERID
                                   AND SOLINENUM = QQ.SOLINENUM;

                                BEGIN

                                    INSERT INTO EXP_SOPROMISESRCNCP ( ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED,
                                                                   PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK,
                                                                   DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY,
                                                                   NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM,
                                                                   AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE, AP1ID)
                                    VALUES (QQ.ENTERPRISE, QQ.SOPROMISEID, QQ.PLANID, QQ.SALESORDERID, QQ.SOLINENUM + 1, QQ.ITEM, V_AVAILQTY2, QQ.PROMISEDDELDATE, QQ.SITEID, QQ.SHIPTOID
                                            , QQ.SALESID, QQ.SALESLEVEL, QQ.DEMANDTYPE, QQ.WEEK, QQ.CHANNELRANK, QQ.CUSTOMERRANK, QQ.PRODUCTRANK, QQ.DEMANDPRIORITY, QQ.TIEBREAK, QQ.GBM, QQ.GLOBALPRIORITY
                                            , QQ.LOCALPRIORITY, QQ.BUSINESSTYPE, QQ.ROUTING_PRIORITY, QQ.NO_SPLIT, QQ.MAP_SATISFY_SS, QQ.PREALLOC_ATTRIBUTE, QQ.BUILDAHEADTIME, QQ.TIMEUOM, QQ.AP2ID
                                            , QQ.GC, QQ.MEASURERANK, QQ.PREFERENCERANK, QQ.INITDTTM, QQ.INITBY, QQ.UPDTTM, 'C/Stock_UNF_ORD_PART2'||V_AVAILQTY2, QQ.REASONCODE, NVL(QQ.OPTION_CODE,0)-BIN_TO_NUM(1), QQ.AP1ID );

                                EXCEPTION WHEN DUP_VAL_ON_INDEX THEN

                                --! SOLINENUM MAX 값을 찾음 !--
                                    SELECT MAX(SOLINENUM) + 1
                                      INTO V_SOLINENUM
                                      FROM EXP_SOPROMISESRCNCP
                                     WHERE PLANID = QQ.PLANID
                                       AND SALESORDERID = QQ.SALESORDERID;

                                --! SOLINENUM MAX 값으로 INSERT 재시도 !--
                                     INSERT INTO EXP_SOPROMISESRCNCP( ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM, ITEM, QTYPROMISED,
                                                                   PROMISEDDELDATE, SITEID, SHIPTOID, SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK, PRODUCTRANK,
                                                                   DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY,
                                                                   NO_SPLIT, MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM,
                                                                   AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, UPBY, REASONCODE, OPTION_CODE, AP1ID)
                                     VALUES (QQ.ENTERPRISE, QQ.SOPROMISEID, QQ.PLANID, QQ.SALESORDERID, V_SOLINENUM, QQ.ITEM, V_AVAILQTY2, QQ.PROMISEDDELDATE, QQ.SITEID, QQ.SHIPTOID
                                              , QQ.SALESID, QQ.SALESLEVEL, QQ.DEMANDTYPE, QQ.WEEK, QQ.CHANNELRANK, QQ.CUSTOMERRANK, QQ.PRODUCTRANK, QQ.DEMANDPRIORITY, QQ.TIEBREAK, QQ.GBM, QQ.GLOBALPRIORITY
                                              , QQ.LOCALPRIORITY, QQ.BUSINESSTYPE, QQ.ROUTING_PRIORITY, QQ.NO_SPLIT, QQ.MAP_SATISFY_SS, QQ.PREALLOC_ATTRIBUTE, QQ.BUILDAHEADTIME, QQ.TIMEUOM, QQ.AP2ID
                                              , QQ.GC, QQ.MEASURERANK, QQ.PREFERENCERANK, QQ.INITDTTM, QQ.INITBY, QQ.UPDTTM, 'C/Stock_UNF_ORD_PART2', QQ.REASONCODE, NVL(QQ.OPTION_CODE,0)-BIN_TO_NUM(1), QQ.AP1ID );

                                END;

                                V_AVAILQTY2 := CASE WHEN V_AVAILQTY2 - QQ.QTYPROMISED < 0 THEN 0 ELSE V_AVAILQTY2 - QQ.QTYPROMISED END;

                             END IF;

                          END LOOP;

            END IF;

            COMMIT;

        END LOOP;

    EXCEPTION
        WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('5-1. CHANNEL STOCK UNF_ORD Update Error'||SQLCODE||'_'||SQLERRM);
        ROLLBACK;
        WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('5-1. CHANNEL STOCK UNF_ORD Update Error'||SQLCODE||'_'||SQLERRM);
        ROLLBACK;
    END;



     BEGIN
         --! SYNC 때는 CH/STOCK 대상을 강제 SHORT을 내지 못하므로 우선순위를 최하위로 낮추어서 RTF를 받지 못하도록함. MOD(NVL(OPTION_CODE,0), 2) = 1 인 것(CHSTOCK 대상)의 DEMAND 우선순위를 9로 강제로 낮춤, NEWFCST 대상일 경우 8, CHSTOCK 대상일 경우 9 !--
         UPDATE EXP_SOPROMISESRCNCP
         SET    DEMANDPRIORITY = SUBSTR(DEMANDPRIORITY, 1,3)||'9'||SUBSTR(DEMANDPRIORITY, 5,4)
         WHERE  PLANID         = V_PLANID
         AND    MOD(NVL(OPTION_CODE,0), 2) = 1;
     --! 예외 처리 !--
     EXCEPTION
        WHEN NO_DATA_FOUND THEN
         DBMS_OUTPUT.PUT_LINE('6. CHANNEL STOCK DEMANDPRIORITY Update Error: check '||SQLCODE||'_'||SQLERRM);
         ROLLBACK;
         WHEN OTHERS THEN
         DBMS_OUTPUT.PUT_LINE('6. CHANNEL STOCK DEMANDPRIORITY Update Error: check '||SQLCODE||'_'||SQLERRM);
     END;

     COMMIT;
--! 예외 처리 !--
EXCEPTION
    WHEN NO_DATA_FOUND THEN
    DBMS_OUTPUT.PUT_LINE('7. Error : check '||SQLCODE||'_'||SQLERRM);
    ROLLBACK;
    WHEN OTHERS THEN
    DBMS_OUTPUT.PUT_LINE('7. Error : check '||SQLCODE||'_'||SQLERRM);
    ROLLBACK;
END;
