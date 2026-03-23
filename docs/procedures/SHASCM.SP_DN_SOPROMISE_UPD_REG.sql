CREATE OR REPLACE PROCEDURE SHASCM.SP_DN_SOPROMISE_UPD_REG (IN_TYPE VARCHAR2 DEFAULT 'NCP')
/*---------------------------------------------------------------------------------------------
 SYSTEM       : 생활가전 GLOBAL-SCM                                                   
 PROGRAM ID   : SP_DN_SOPROMISE_UPD_REG
 PROGRAM DESC : FNE - FNE - Netting 결과 최종 생성
 CREATE DATE  : 2011.05.10                                                 
 CREATE BY    : 이서연                                                            
 DESCRIPTION  : 1.최종적으로 생성된 DEMAND를 TYPE(MAIN/NCP)에 따라
                  EXP_SOPROMISENCP, EXP_SOPROMISE 으로 RELEASE한다.
                >> SP_FN_TCSOPROMISENC_REG  
                 한국총괄에서는 CDC ORDER 만 반영하므로, RDC에 의한 ORDER DEMAND 를 제거하고 사용한다
                 EXP_SOPROMISENCP_FP = FP+ DEMAND PROFILE 을 위한 TABLE 
---------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------
 S/R NO       : 
 UPDATE DATE  : 2015.07.02
 UPDATE BY    : 김용선                          
 DESCRIPTION  : Demand 테이블들에 LOCALID 추가 관련 수정
---------------------------------------------------------------------------------------------*/
IS

    -- DEFAULT VARIABLE
    V_PROGNAME             VARCHAR2(40) DEFAULT 'SP_DN_SOPROMISE_UPD_REG';
    V_LOGSEQ               NUMBER(10);     
    V_STARTDATE            DATE;
    V_ERRORCNT             INTEGER;
    V_LOGMESSAGE           LONG;
    V_ERRMESSAGE           VARCHAR2(2000);    
    V_INSERTCNT            INTEGER;
    V_PLANID               MST_PLAN.PLANID%TYPE;
    V_PLANWEEK             MST_PLAN.PLANWEEK%TYPE;
    V_EFFSTARTDATE         MST_PLAN.EFFSTARTDATE%TYPE;
    V_EFFENDDATE           MST_PLAN.EFFENDDATE%TYPE;
    V_RULEPLAN             MST_PLAN.INITBY%TYPE;
    V_GBM                  MST_ITEM.GBM%TYPE;   
    NO_PLANID              EXCEPTION;
    
    V_COMMITCNT            INTEGER;
    V_BUILDAHEADTIME       EXP_SOPROMISE.BUILDAHEADTIME%TYPE;
    
    V_CNT                  INTEGER;
   
/****************************************************************************************/
BEGIN

    DBMS_OUTPUT.ENABLE(9999999);              
    SELECT SEQ_LOG.NEXTVAL INTO V_LOGSEQ FROM DUAL;
    SELECT SYSDATE INTO V_STARTDATE FROM DUAL;   
    SELECT FN_GETGBM() INTO V_GBM FROM DUAL; 

    TUNE.CAPTURE;         
    TUNE.SHOW('======================================================================');  
    TUNE.SHOW(V_PROGNAME ||' START['||TO_CHAR(SYSDATE, 'YYYY/MM/DD HH24:MI:SS')||']'); 
    TUNE.SHOW('----------------------------------------------------------------------');   
      
    SP_FN_GET_PLANID(IN_TYPE,V_PLANID,V_PLANWEEK,V_EFFSTARTDATE,V_EFFENDDATE);   
    
    IF V_PLANID IS NULL THEN
        RAISE NO_PLANID; 
    END IF;
    
    V_RULEPLAN := PG_FN_COMMON.GET_RULEPLAN(V_PLANID);       
     
    --! NCP 의 경우 !--
    IF IN_TYPE = 'NCP'
    THEN  

        -- 삼성닷컴 DEMANDPRIORITY 업데이트 
        -- 당주포함 10주간만 Preference Rank 반영 25.03.20 이동은P
        UPDATE EXP_SOPROMISE_SRC_NCP A
           SET A.DEMANDPRIORITY = TO_NUMBER('5'||SUBSTR(A.DEMANDPRIORITY,2)),
               A.GLOBALPRIORITY = DECODE(NVL(GLOBALPRIORITY,0),0,NULL, TO_NUMBER('5'||SUBSTR(GLOBALPRIORITY,2))),
               A.LOCALPRIORITY  = DECODE(NVL(LOCALPRIORITY,0),0,NULL, TO_NUMBER('5'||SUBSTR(LOCALPRIORITY,2))),
               A.PREFERENCERANK = DECODE(NVL(PREFERENCERANK,'-'),'-',NULL, '5'),
               A.UPBY = 'CHG_DEMANDPRIORITY'
         WHERE A.PLANID = V_PLANID
           AND A.DEMANDPRIORITY IS NOT NULL
           AND SUBSTR(A.DEMANDPRIORITY,0,1) > 5
           AND A.SALESID NOT IN (SELECT SALESID FROM GUI_SALESHIERARCHY WHERE GCID = '201')  -- 한총은 제외 
           AND (A.SALESID, FN_GETSECTION(A.ITEM)) IN (SELECT SALESID, SECTION
                                                        FROM MST_ACNTINFO
                                                       WHERE SALESID = A.SALESID
                                                         AND SECTION IN ('DA','DAS')
                                                         AND CHANNELTYPE IN ('ONLINE')
                                                         AND GPGNAME IN ('COM','COM_SI','COM_DIR')
                                                      )
           AND TO_CHAR(A.PROMISEDDELDATE,'IYYYIW') <= TO_CHAR(SYSDATE+7*9,'IYYYIW');           
        
        COMMIT; 
    
    
        EXECUTE IMMEDIATE 'TRUNCATE TABLE EXP_SOPROMISENCP';
        EXECUTE IMMEDIATE 'TRUNCATE TABLE EXP_SOPROMISESRCNCP_FP';      
  
        COMMIT;            
                                              
        --! 2. FP+ DEMAND PROFILER 를 위한 데이터 백업 !--
        INSERT INTO EXP_SOPROMISESRCNCP_FP
                   (ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM,
                    ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID,
                    SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK,
                    CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM,
                    GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE,
                    ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS,
                    PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC,
                    MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM,
                    UPBY, REASONCODE)
        SELECT ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM,
               ITEM, (ASSIGNEDQTY+QTYPROMISED) QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, SALESID,
               SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, CUSTOMERRANK,
               PRODUCTRANK, DENSE_RANK() OVER (ORDER BY DEMANDPRIORITY) DEMANDPRIORITY, TIEBREAK, GBM, GLOBALPRIORITY,
               LOCALPRIORITY, BUSINESSTYPE, ROUTING_PRIORITY, NO_SPLIT,
               MAP_SATISFY_SS, PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM,
               AP2ID, GC, MEASURERANK, PREFERENCERANK, INITDTTM, INITBY,
               UPDTTM, UPBY, REASONCODE
          FROM EXP_SOPROMISE_SRC_NCP
         WHERE PLANID = V_PLANID;          
                            
        INSERT INTO EXP_SOPROMISENCP
                   (ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM,
                    ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID,
                    SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK,
                    CUSTOMERRANK, PRODUCTRANK, DEMANDPRIORITY, TIEBREAK, GBM,
                    GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE,
                    ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS,
                    PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC,
                    MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM,
                    UPBY, REASONCODE, LOCALID)
        SELECT ENTERPRISE, SOPROMISEID, PLANID, SALESORDERID, SOLINENUM,
               ITEM, QTYPROMISED, PROMISEDDELDATE, SITEID, SHIPTOID, 
               SALESID, SALESLEVEL, DEMANDTYPE, WEEK, CHANNELRANK, 
               CUSTOMERRANK, PRODUCTRANK, DENSE_RANK() OVER (ORDER BY DEMANDPRIORITY) DEMANDPRIORITY, TIEBREAK, GBM, 
               GLOBALPRIORITY, LOCALPRIORITY, BUSINESSTYPE, 
               ROUTING_PRIORITY, NO_SPLIT, MAP_SATISFY_SS, 
               PREALLOC_ATTRIBUTE, BUILDAHEADTIME, TIMEUOM, AP2ID, GC, 
               MEASURERANK, PREFERENCERANK, INITDTTM, INITBY, UPDTTM, 
               UPBY, REASONCODE, LOCALID
          FROM EXP_SOPROMISE_SRC_NCP
         WHERE PLANID = V_PLANID;                          
    
    COMMIT;        
           
    END IF;
  
    COMMIT;
   
    TUNE.SHOW('=====================================================================');  
    TUNE.SHOW(V_PROGNAME ||' ENDS OK ['||TO_CHAR(SYSDATE, 'YYYY/MM/DD HH24:MI:SS')||']'); 
    TUNE.SHOW('---------------------------------------------------------------------');
    
    SP_LOGGING_REG(V_LOGSEQ,V_PROGNAME,V_STARTDATE,V_ERRORCNT,SUBSTR(SQLERRM,1,200));
   
EXCEPTION 
WHEN  NO_PLANID THEN
         DBMS_OUTPUT.PUT_LINE('NO RUNNING PLANID : '||SUBSTR(SQLERRM,1,100));
WHEN OTHERS THEN

    V_ERRMESSAGE := SUBSTR(SQLCODE||SQLERRM,1,2000);
    V_ERRORCNT := 1; 
    TUNE.SHOW('=====================================================================');  
    TUNE.SHOW(V_PROGNAME ||' ENDS ERROR : '||V_ERRMESSAGE || ' [' ||TO_CHAR(SYSDATE, 'YYYY/MM/DD HH24:MI:SS')||']'); 
    TUNE.SHOW('---------------------------------------------------------------------');    
    SP_LOGGING_DETAIL_REG(V_PROGNAME,V_STARTDATE,V_ERRMESSAGE,V_LOGSEQ);
    SP_LOGGING_REG(V_LOGSEQ,V_PROGNAME,V_STARTDATE,V_ERRORCNT,V_ERRMESSAGE);

END;
