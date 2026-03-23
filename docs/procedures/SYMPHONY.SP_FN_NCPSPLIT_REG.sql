CREATE OR REPLACE PROCEDURE SYMPHONY."SP_FN_NCPSPLIT_REG" (in_type varchar2 default null) IS
/*----------------------------------------------------------------------------------------
  PROGRAM_ID   : SP_FN_SPLIT_REG
  CREATE DATE  : 2006.09.04
  CREATE BY    : NCP T/F
  DESCRIPTION :
       1.ELS LOCAL재고데이터를 구성한다.
       2.데이터에 대해 재고정보가 있는지 체크하여
         재고가 있으면 ELS site의 수량은 재고만큼 차감하고, LOCAL site로
         데이터를 생성한다.
   cf) ELS재고 Data를 가져오기 위해서는 Dynamic 재고 Closing이 선행되어야 한다.
----------------------------------------------------------------------------------------*/

    v_planid          MST_PLAN.planid %TYPE;
    v_planweek        MST_PLAN.planweek%TYPE;
    v_effstartdate    MST_PLAN.effstartdate %TYPE;
    v_effenddate      MST_PLAN.effenddate %TYPE;
    no_planid         EXCEPTION;

    next_qty          number;
    v_consumeqty      number;
    v_fcstqty         number;

    v_salesid         mst_sales.salesid%TYPE;
    v_item            mst_item.item%TYPE;
    v_inventorydate   date;
    v_progname        varchar2(40) default 'SP_FN_SPLIT_REG';

    -- prealloc단계에서 ap2id는 조정되었다는 대전제를 두고 데이터를 조회한다.
    -- logisticssite에 정의된 salesid는 prealloc의 level과 밀접한 연관성을 가진다.
    -- gc는 prealloc이전단계에서 이미 section으로 전용되어져 구성된 상태임.
    -- subsoid는 localsiteid를 감안한 salesorderid로 변경된 so임.
    CURSOR CUR_BUFFER IS
            select a.planid,  a.salesorderid,  a.solinenum, a.demandpriority,
                   a.salesid, a.item,a.siteid, b.salessiteid localsiteid,
                   a.promiseddeldate, a.ap2id, a.gc, a.qtypromised, a.gbm, a.enterprise,sopromiseid,
                   a.salesorderid||b.salessiteid subsoid,
                   a.demandtype, a.week
            from exp_sopromisesrcncp a, MST_LOGISTICSSITE b
            where a.planid = v_planid
            and   a.gc     = b.section
            and   a.siteid = b.LOGISTICSSITEID
            and   a.ap2id  = b.salesid
            and   a.qtypromised > 0
            order by a.week,a.salesid,a.item,demandpriority ASC;



-- 해당 주차의 LOCAL 창고의 재고+INTRANSIT 물량.
    cursor cur_inventory is
        select a.planid, a.logisticssiteid, a.salessiteid, a.salesid, a.item,
               max(a.inventorydate) inventorydate,
               sum(a.availqty) availqty,
               sum(a.consumeqty) consumeqty
          from buf_salesplaninventory a, mst_item b
         where a.planid        = v_planid
           and a.salesid       = v_salesid
           and a.item          = v_item
           and a.item          = b.item
           and a.inventorydate <=v_inventorydate
           and a.consumeqty > 0
        group by a.planid, a.logisticssiteid, a.salessiteid, a.salesid, a.item
        order by inventorydate;

    -- localsite로 conversion된 new sopromise를 구성한다.
    -- HUB 수량이 차감된 만큼 LOCAL Site로 수량이 추가된다.
    procedure add_sopromise(s_planid         exp_sopromise.planid%type,
                            s_salesorderid   exp_sopromise.salesorderid%type,
                            s_solinenum      exp_sopromise.solinenum%type,
                            s_demandpriority exp_sopromise.DEMANDPRIORITY%type,
                            s_salesid        exp_sopromise.salesid%type,
                            s_item           exp_sopromise.item%type,
                            s_siteid         exp_sopromise.siteid%type,
                            s_promiseddeldate exp_sopromise.promiseddeldate%type,
                            s_qtypromised    exp_sopromise.qtypromised%type,
                            s_gbm            exp_sopromise.gbm%type,
                            s_enterprise     exp_sopromise.enterprise%type,
                            s_sopromiseid    exp_sopromise.sopromiseid%type,
                            s_ap2id          exp_sopromise.ap2id%type,
                            s_gc             exp_sopromise.gc%type,
                            s_demandtype     exp_sopromise.demandtype%type,
                            s_week           exp_sopromise.week%type)
    is
    begin

          insert into exp_sopromisesrcncp
            (planid, salesorderid,solinenum,demandpriority,salesid,demandtype,
             item, siteid, shiptoid, promiseddeldate, qtypromised, gbm, ap2id,gc, WEEK,
             enterprise,sopromiseid, initdttm,initby)
          values
            (s_planid,   s_salesorderid,  s_solinenum,  s_demandpriority, s_salesid, s_demandtype,
             s_item, s_siteid, s_siteid,  s_promiseddeldate, s_qtypromised, s_gbm, s_ap2id, s_gc, S_WEEK,
             s_enterprise, s_sopromiseid, sysdate,    v_progname);

    exception
    when others then
         dbms_output.put_line('add_sopromise : '||SUBSTR(SQLERRM,1,100));
    end;

    -- localsite로 conversion된 new sopromise를 구성한다.
    procedure update_sopromise(s_planid      exp_sopromise.planid%type,
                            s_salesorderid   exp_sopromise.salesorderid%type,
                            s_solinenum      exp_sopromise.solinenum%type,
                            s_qtypromised    exp_sopromise.qtypromised%type)
    is
    begin

          update exp_sopromisesrcncp
          set    qtypromised = s_qtypromised,
                 updttm = sysdate,
                 upby   = v_progname
          where  planid = s_planid
          and    salesorderid = s_salesorderid
          and    solinenum = s_solinenum;

    exception
    when others then
         dbms_output.put_line('update_sopromise : '||SUBSTR(SQLERRM,1,100));
    end;

    -- localsite로 conversion된 new sopromise를 구성한다.
    procedure update_inventory(s_planid    buf_salesplaninventory.planid%type,
                               s_salesid   buf_salesplaninventory.salesid%type,
                               s_item      buf_salesplaninventory.item%type,
                               s_date      buf_salesplaninventory.inventorydate%type,
                               s_qty       buf_salesplaninventory.consumeqty%type)
    is
    begin

          update buf_salesplaninventory
          set    consumeqty= s_qty,
                 updttm = sysdate,
                 upby   = v_progname
          where  planid = s_planid
          and    salesid = s_salesid
          and    item    = s_item
          and    inventorydate <= s_date;

    exception
    when others then
         dbms_output.put_line('update_inventory : '||SUBSTR(SQLERRM,1,100));
    end;
/****************************************************************************************/
BEGIN

    dbms_output.enable(9999999);

     --get current running planid  --------------------------------------------
    Sp_fn_Get_Planid(in_type,v_planid,v_planweek,v_effstartdate,v_effenddate);

    IF v_planid IS NULL THEN RAISE no_planid; /* isrunning planid가 없으면 exception */
    END IF;

    -- delete local inventory data of current planid.
    delete from buf_salesplaninventory
    where planid = v_planid;

    -- make local inventory data from closed inventory.
    insert into buf_salesplaninventory
         ( planid, logisticssiteid, salessiteid, salesid, item, inventorydate, availqty, consumeqty,
           initdttm, initby)
    select planid, logisticssiteid, salessiteid, salesid, item, inventorydate,
           sum(availqty), sum(consumeqty),   sysdate,  v_progname
    from (
            select b.planid, a.logisticssiteid, a.salessiteid, a.salesid, b.item,
                   fn_week_first_date(to_char(onhanddate,'iyyyiw')) inventorydate,
                   sum(b.availqty) availqty,
                   sum(b.availqty) consumeqty
            from  mst_logisticssite a, mst_inventory b, mst_item c
            where a.salessiteid = b.siteid
            and   b.planid      = v_planid
            and   a.gbm         = c.gbm
            and   b.item        = c.item
            and   a.section     = c.section
            and   onhanddate is not null
            group by b.planid, a.logisticssiteid, a.salessiteid, a.salesid, b.item,
                     fn_week_first_date(to_char(onhanddate,'iyyyiw'))
            union all
            select b.planid, a.logisticssiteid, a.salessiteid, a.salesid, b.item,
                   fn_week_first_date(to_char(b.scmeta,'iyyyiw')),
                   sum(b.intransitqty) availqty,
                   sum(b.intransitqty) consumeqty
            from  mst_logisticssite a, mst_intransit b, mst_item c
            where b.planid   = v_planid
            and   b.tositeid = a.salessiteid
            and   a.gbm      = c.gbm
            and   b.item     = c.item
            and   a.section     = c.section
            and   b.scmeta is not null
            group by b.planid, a.logisticssiteid, a.salessiteid, a.salesid, b.item,
                     fn_week_first_date(to_char(b.scmeta,'iyyyiw'))
    ) group by planid, logisticssiteid, salessiteid, salesid, item, inventorydate;

    COMMIT;


     -- BUF_SALESPLANINVENTORY의 DATA를 ARC_SALESPLANINVENTORY에 저장한다.
     delete from arc_salesplaninventory
     where planid = v_planid;

     insert into arc_salesplaninventory
       (planid, logisticssiteid, salessiteid, salesid, item, inventorydate,
        availqty, consumeqty, initdttm, initby, updttm, upby)
     select planid, logisticssiteid, salessiteid, salesid, item, inventorydate,
            availqty, consumeqty, initdttm, initby, updttm, upby
     from   buf_salesplaninventory
     where  planid = v_planid;

     COMMIT;


    --------------------------------------------------------------------------------
    -- DEMAND에 대해서 LOCAL재고가 있는지 체크하여 SPLIT 과정을 진행한다.
    -- DEMAND가 W+7부터 존재하고, LOCAL재고가 W+2에 있는 경우에는
    -- W+2부터 W+7까지의 LOCAL 재고를 소진해줘야 하기 때문에 W+7이전의 모든
    -- LOCAL재고를 합산하도록 로직을 처리한다.
    --------------------------------------------------------------------------------
    FOR c1 IN CUR_BUFFER
    LOOP

        begin
            -- 해당Demand의 local 재고가 있는 지 파악.
            select nvl(sum(consumeqty),0)
            into   v_consumeqty
            from  buf_salesplaninventory
            where planid  =  v_planid
            and   salesid =  c1.ap2id
            and   item    =  c1.item
            and   inventorydate <= c1.promiseddeldate;
        exception
            when no_data_found then
                v_consumeqty := 0;
        end;

        ---------------------------------------------------------
        -- case 1. local 재고가 ELS Fcst 물량 보다 작다면
        ---------------------------------------------------------
        if v_consumeqty <   c1.qtypromised and v_consumeqty != 0
        then

            v_inventorydate   := c1.promiseddeldate;

            begin
               -- 해당 주차의 local 창고의 재고+intransit 물량을 0 으로 변경한다.
                update_inventory(v_planid, c1.ap2id, c1.item, c1.promiseddeldate, 0);

               -- ELS 에 대해서 LOCAL 창고 물량만큼 DEMAND를 차감한다.
                update_sopromise(v_planid, c1.salesorderid, c1.solinenum,
                                 c1.qtypromised-v_consumeqty);

               -- 앞에서 빼준 ELS 물량 만큼 LOCALSITE로 demand를 생성한다.
                add_sopromise(v_planid, c1.subsoid, c1.solinenum, c1.demandpriority, c1.salesid,
                              c1.item,  c1.localsiteid, c1.promiseddeldate, v_consumeqty,
                              c1.gbm,   c1.enterprise, c1.sopromiseid, c1.ap2id, c1.gc, c1.demandtype, c1.week);

            exception
            when others then
                 dbms_output.put_line('case 1 : '||SUBSTR(SQLERRM,1,100));
            END;

        -------------------------------------------------------------
        -- case 2. Local 창고 물량이 더 크다면
        -------------------------------------------------------------
        ELSIF v_consumeqty >=   c1.qtypromised AND v_consumeqty !=0 THEN

                v_salesid         := c1.ap2id;
                v_item            := c1.item;
                v_inventorydate   := c1.promiseddeldate;
                v_fcstqty         := c1.qtypromised;

               -- ELS 물량 만큼 LOCAL SITEID로 INSERT해준다.
               add_sopromise(v_planid, c1.subsoid, c1.solinenum, c1.demandpriority, c1.salesid,
                             c1.item,  c1.localsiteid, c1.promiseddeldate, c1.qtypromised,
                             c1.gbm,   c1.enterprise,  c1.sopromiseid, c1.ap2id, c1.gc, c1.demandtype, c1.week);

               -- ELS Demand를 0으로 변경한다.
               update_sopromise(v_planid, c1.salesorderid, c1.solinenum, 0);

               -- 해당 주차의 buf Table 의 재고 수량을 위에서 계산해 준 만큼 차감해 준다.
               FOR C2 IN CUR_INVENTORY
               LOOP

                    BEGIN
                        --* Fcst 물량이 local 재고 물량보다 크다면 local 재고 물량을 다 소진해 준다.
                        if v_fcstqty > c2.consumeqty
                        then
                                --* 해당 주차까지의 누적재고를 0 으로 변경한다.
                                update_inventory(v_planid, c1.ap2id, c1.item, c2.inventorydate, 0);

                        --* fcst <= local재고인 경우  local재고=local재고-fcst
                        --* 로직의 복잡도 문제로 기존재고data삭제하고 신규로 넣는식으로 처리함.
                        elsif  v_fcstqty <= c2.consumeqty and v_fcstqty <> 0
                        then
                                next_qty := c2.consumeqty - v_fcstqty;

                                delete from buf_salesplaninventory
                                where  planid = v_planid
                                and    salesid = c1.ap2id
                                and    item    = c1.item
                                and    inventorydate <= c2.inventorydate;

                                insert into buf_salesplaninventory
                                  ( planid, logisticssiteid, salessiteid, salesid, item,
                                    inventorydate, availqty,consumeqty,initdttm, initby)
                                values(c2.planid, c2.logisticssiteid, c2.salessiteid, c2.salesid, c2.item,
                                       c2.inventorydate, next_qty, next_qty, sysdate, 'TRANSFER');
                        end if;
                    EXCEPTION
                    WHEN OTHERS THEN
                         dbms_output.put_line('case3 : '||substr(sqlerrm,1,100));
                    END;
               END LOOP;

        END IF;

    END LOOP;

    COMMIT;

    EXCEPTION
    WHEN no_planid THEN
         DBMS_OUTPUT.PUT_LINE('ORA-ERR NO RUNNING PLANID : '||SUBSTR(SQLERRM,1,100));
    WHEN OTHERS THEN
         DBMS_OUTPUT.PUT_LINE(SUBSTR(SQLERRM,1,100));
END; 
