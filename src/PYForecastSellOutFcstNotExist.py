"""
    (25.05.28) S/Out FCST_AP2 <= 0 \| S/Out FCST_GC <= 0

    개요 (*)							
								
		* 프로그램명						
			PYForecastSellOutFcstNotExist					
								
								
		* 목적						
			AP2 와 GC FCST가 없는 특정 Item 선별					
								
								
								
		* 변경이력						
			2025.05.12 최초 작성					
								
								
								
	Script Parameter							
			Version					
			    - CWV_DP					
								
								
								
	Input Tables (*)							
								
		(Input 1) S/out FCST_AP2 값						
			df_in_Sout					
				Select (				
				[Version].[Version Name].[CWV_DP]				
				 * [Sales Domain].[Sales Std2].filter( ~(#.Name in {"300768","300340","300112"}) )				
				 * [Item].[Item]				
				) on row, 				
				({Measure.[S/Out FCST_AP2],Measure.[S/Out FCST_GC]}) on column				
				where { [Item].[Product Group].[MOBILE], [Time].[Partial Week].filter (#.Key >= &CurrentYear.element(0).leadoffset(-1).Key && #.Key < &CurrentYear.element(0).leadoffset(1).Key } ;				
								
								
			Version.[Version Name]	Sales Domain.[Sales Std2]	Item.[Item]	S/Out FCST_AP2	S/Out FCST_GC	
			CWV_DP	300114	SM-G965FZAFTIM		10	
			CWV_DP	300114	SM-T580NZWETPH			
								
								
		(Input 2) Sell-In Lock 필요한 Item 정보						
			df_in_SELLOUTFCST_NOTEXIST					
				Select (				
				[Version].[Version Name].[CWV_DP]				
				* [Item].[Item] 				
				) on row, 				
				({Measure.[S/Out Fcst Item Check Flag]}) on column where { Measure.[S/Out Fcst Item Check Flag] == true };				
								
			Version.[Version Name]	Item.[Item]	S/Out Fcst Item Check Flag			
			CWV_DP	SM-G965FZAFTIM	True			
			CWV_DP	SM-T580NZWETPH	True			
			CWV_DP	SM-T813NZWEBTU	True			
			CWV_DP	SM-A217NZKNSKO	True			
			CWV_DP	SM-A520FZKAXSA	True			
								
								
								
								
								
								
								
	Output Tables (*)							
								
			df_output_SellOut_FCST_Not_Exist					
				Select ([Version].[Version Name]				
				 * [Sales Domain].[Sales Std2]				
				 * [Item].[Item]  )  on row, 				
				( { Measure.[S/Out Fcst Not Exist Flag] } ) on column;				
								
								
								
								
								
								
	주요 로직 (*)							
								
		Step 1) df_in_Sout 조건 처리						
			Version.[Version Name]	Sales Domain.[Sales Std2]	Item.[Item]	S/Out FCST_AP2	S/Out FCST_GC	
			CWV_DP	300114	SM-G965FZAFTIM		10	
			CWV_DP	300114	SM-T580NZWETPH			
			
            - df_in_Sout 에서 df_in_SELLOUTFCST_NOTEXIST 값이 true인 Item 만 남긴다.					
			- Sales Std2 가  ('300768','300340','300112') 인 값을 제거한다.					
			- Nan 인 값에 대해서 0으로 채운다.					
			- S/Out FCST_AP2 == 0 || S/Out FCST_GC == 0 인 경우만 남긴다					
								
								
								
								
		Step 2) 최종 Output 구성						
			Version.[Version Name]	Sales Domain.[Sales Std2]	Item.[Item]	S/Out Fcst Not Exist Flag		
			CWV_DP	300114	SM-T580NZWETPH	True		
			- S/Out FCST_AP2, S/Out FCST_GC 삭제					
			- S/Out Fcst Not Exist Flag = True 추가					
								
								
								

"""
import os, sys, time, datetime, traceback, shutil ,  inspect
import pandas as pd
from NSCMCommon import NSCMCommon as common
from NSCMCommon import VDCommon as vdCommon
import glob
# ──────────────────────────────────────────────────────────────────────────────
#  공통 설정
# ──────────────────────────────────────────────────────────────────────────────
str_instance      = 'PYForecastSellOutFcstNotExist'
str_input_dir     = f'Input/{str_instance}'
str_output_dir    = f'Output/{str_instance}'
is_local          = common.gfn_get_isLocal()
is_print          = True
flag_csv          = True
flag_exception    = True
logger            = common.G_Logger(p_py_name=str_instance)
common.gfn_set_local_logfile()
LOG_LEVEL         = common.G_log_level
# ──────────────────────────────────────────────────────────────────────────────
#  컬럼 상수
# ──────────────────────────────────────────────────────────────────────────────
COL_VERSION            = 'Version.[Version Name]'
COL_STD2               = 'Sales Domain.[Sales Std2]'
COL_ITEM               = 'Item.[Item]'
COL_FCST_AP2           = 'S/Out FCST_AP2'
COL_FCST_GC            = 'S/Out FCST_GC'
COL_CHECK_FLAG         = 'S/Out Fcst Item Check Flag'
COL_NOT_EXIST_FLAG     = 'S/Out Fcst Not Exist Flag'
# ──────────────────────────────────────────────────────────────────────────────
#  DF 상수
# ──────────────────────────────────────────────────────────────────────────────
STR_DF_IN_SOUT         = 'df_in_Sout'
STR_DF_IN_FLAG         = 'df_in_SELLOUTFCST_NOTEXIST'
STR_DF_STEP01_FILTER   = 'df_step01_filter'
STR_DF_OUT_NOT_EXIST   = 'df_output_SellOut_FCST_Not_Exist'
# ──────────────────────────────────────────────────────────────────────────────
#  데코레이터 & 로그 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
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

def fn_check_input_table(df_p_source: pd.DataFrame, str_p_source_name: str, str_p_cond: str,p_row_num=20) -> None:
    """
    Input Table을 체크한 결과를 로그 또는 Exception으로 표시한다.
    :param df_p_source: Input table
    :param str_p_source_name: Name of Input table
    :param str_p_cond: '0' - Exception, '1' - Warning Log
    :return: None
    """
    # Input Table 로그 출력
    logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=p_row_num)

    if df_p_source.empty:
        if str_p_cond == '0':
            # 테이블이 비어 있는 경우 raise Exception
            raise Exception(f'[Exception] Input table({str_p_source_name}) is empty.')
        else:
            # 테이블이 비어 있는 경우 Warning log
            logger.Note(p_note=f'Input table({str_p_source_name}) is empty.', p_log_level=LOG_LEVEL.warning())

def fn_log_dataframe(df_p_source: pd.DataFrame, str_p_source_name: str,p_row_num=20) -> None:
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
        # logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=p_row_num)
        fn_check_input_table(df_p_source=df_p_source,str_p_source_name=str_p_source_name,str_p_cond='1')
        # if is_local and not df_p_source.empty and flag_csv:
        if is_local and flag_csv:
            # 로컬 Debugging 시 csv 파일 출력
            df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)
    else:
        # 최종 Output 테이블인 경우에는 무조건 로그 출력
        if is_output:
            logger.PrintDF(p_df=df_p_source, p_df_name=str_p_source_name, p_log_level=LOG_LEVEL.debug(), p_format=1,p_row_num=p_row_num)
            # if is_local and not df_p_source.empty:
            if is_local:
                # 로컬 Debugging 시 csv 파일 출력
                df_p_source.to_csv(str_output_dir + "/"+str_p_source_name+".csv", encoding="UTF8", index=False)

def fn_convert_type(df: pd.DataFrame, startWith: str, type):
    for column in df.columns:
        if column.startswith(startWith):
            df[column] = df[column].astype(type)


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
            'df_in_Sout.csv'                    : STR_DF_IN_SOUT       ,
            'df_in_SELLOUTFCST_NOTEXIST.csv'    : STR_DF_IN_FLAG    
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
        input_dataframes[STR_DF_IN_SOUT]        = df_in_Sout
        input_dataframes[STR_DF_IN_FLAG]        = df_in_SELLOUTFCST_NOTEXIST


    fn_convert_type(input_dataframes[STR_DF_IN_SOUT], 'Sales Domain', str)
    logger.info("loaded dataframes")


# ──────────────────────────────────────────────────────────────────────────────
#  Step-01 : 조건 필터  (빈/누락 입력에도 안전하게 동작)
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step01_filter_items() -> pd.DataFrame:
    """
    · 제외 STD2(300768,300340,300112) 제거
    · Flag 가 비었거나(0행) / 필수 컬럼 누락이어도 에러 없이 **빈 DF** 반환
    · Flag True 인 Item만 대상으로 AP2/GC <= 0 조건 적용
    """
    # ── 0) 로드
    df_fcst = input_dataframes[STR_DF_IN_SOUT].copy(deep=True)    
    
    # ── 1) Sales Std2 제외 (컬럼 없으면 빈 DF 반환)
    excluded = {"300768", "300340", "300112"}
    if COL_STD2 not in df_fcst.columns:
        logger.Note(f"[Step01] '{COL_STD2}' 컬럼이 없어 빈 DF 반환", LOG_LEVEL.warning())
        return df_fcst.iloc[0:0]
    df_fcst = df_fcst[~df_fcst[COL_STD2].astype(str).isin(excluded)]

    # ── 2) Flag 테이블 안전 처리
    df_flag = input_dataframes.get(STR_DF_IN_FLAG, None)
    # 2-1) 아예 없거나, 빈 DF 이면 즉시 **빈 DF** 반환
    if (df_flag is None) or df_flag.empty:
        logger.Note("[Step01] df_in_SELLOUTFCST_NOTEXIST 이 없거나 비어 있어 빈 DF 반환", LOG_LEVEL.debug())
        return df_fcst.iloc[0:0]
    # 2-2) 필수 컬럼 점검 (없으면 빈 DF 반환)
    if (COL_CHECK_FLAG not in df_flag.columns) or (COL_ITEM not in df_flag.columns):
        logger.Note(f"[Step01] '{COL_CHECK_FLAG}' 또는 '{COL_ITEM}' 컬럼이 없어 빈 DF 반환", LOG_LEVEL.warning())
        return df_fcst.iloc[0:0]

    # ── 3) Flag True 아이템 목록 (0개여도 빈 DF 반환)
    #  - dtype 이 bool이 아닐 수 있으므로 == True 로 비교
    mask_true = (df_flag[COL_CHECK_FLAG] == True)
    if mask_true.sum() == 0:
        logger.Note("[Step01] Flag=True 인 Item 이 없어 빈 DF 반환", LOG_LEVEL.debug())
        return df_fcst.iloc[0:0]
    valid_itm = df_flag.loc[mask_true, COL_ITEM].astype(str).unique()

    # ── 4) 대상 아이템 필터
    if COL_ITEM not in df_fcst.columns:
        logger.Note(f"[Step01] '{COL_ITEM}' 컬럼이 없어 빈 DF 반환", LOG_LEVEL.warning())
        return df_fcst.iloc[0:0]
    df_fcst = df_fcst[df_fcst[COL_ITEM].astype(str).isin(valid_itm)]
    if df_fcst.empty:
        return df_fcst  # 이미 빈 DF

    # ── 5) AP2/GC NaN → 0 후 조건(AP2<=0 OR GC<=0)
    #     (컬럼이 없으면 0으로 생성하여 보수적으로 '없음' 취급 가능)
    for c in (COL_FCST_AP2, COL_FCST_GC):
        if c not in df_fcst.columns:
            df_fcst[c] = 0
    df_fcst[[COL_FCST_AP2, COL_FCST_GC]] = df_fcst[[COL_FCST_AP2, COL_FCST_GC]].fillna(0)

    ap2 = pd.to_numeric(df_fcst[COL_FCST_AP2], errors='coerce').fillna(0)
    gc  = pd.to_numeric(df_fcst[COL_FCST_GC],  errors='coerce').fillna(0)
    mask_zero = (ap2 <= 0) | (gc <= 0)

    # 전체 컬럼 유지(원래 네가 return df_fcst.loc[mask] 형태로 쓰고 있었음)
    return df_fcst.loc[mask_zero]

# ──────────────────────────────────────────────────────────────────────────────
#  Step-02 : Output 포매터
# ──────────────────────────────────────────────────────────────────────────────
@_decoration_
def fn_step02_make_output(df_src: pd.DataFrame, version: str) -> pd.DataFrame:
    df_out = df_src.copy(deep=True)
    df_out[COL_NOT_EXIST_FLAG] = True
    df_out[COL_VERSION]        = version
    return df_out[[COL_VERSION, COL_STD2, COL_ITEM, COL_NOT_EXIST_FLAG]]
# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.debug(f'[START] {str_instance} {time.strftime("%F %T")}')
    logger.Start()
    # ── 가정: 외부에서 input_dataframes dict 제공 ────────────────────────────

    input_dataframes  = {}
    output_dataframes = {}
    try:
        if is_local:
            Version = 'CWV_DP'
            # ----------------------------------------------------
            # parse_args 대체
            # input , output 폴더설정. 작업시마다 History를 남기고 싶으면
            # ----------------------------------------------------

            input_folder_name  = str_instance
            output_folder_name = str_instance
            
            # ------
            str_input_dir = f'Input/{input_folder_name}'            
            # str_input_dir = f'Input/{input_folder_name}/0918'
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
         # 입력 변수 중 데이터가 없는 경우 경고 메시지를 출력한다.
        for in_df in input_dataframes:
            # fn_check_input_table(input_dataframes[in_df], in_df, '1')
            fn_log_dataframe(input_dataframes[in_df],in_df)
        
        ################################################################################################################
        # Start of processing
        ################################################################################################################
        # Step-01
        dict_log = {'p_step_no': 100, 'p_step_desc': 'Step-01  조건필터'}
        df_step01_filter = fn_step01_filter_items(**dict_log)
        fn_log_dataframe(df_step01_filter, STR_DF_STEP01_FILTER)
        output_dataframes[STR_DF_STEP01_FILTER] = df_step01_filter
        
        # Step-02
        dict_log = {'p_step_no': 200, 'p_step_desc': 'Step-02  Output 포맷'}
        df_output_SellOut_FCST_Not_Exist = fn_step02_make_output(df_step01_filter, Version, **dict_log)
        fn_log_dataframe(df_output_SellOut_FCST_Not_Exist, STR_DF_OUT_NOT_EXIST)


    except Exception as e:
        trace_msg = traceback.format_exc()
        logger.Note(p_note=trace_msg, p_log_level=LOG_LEVEL.debug())
        logger.Error()
        if flag_exception:
            raise Exception(e)
        else:
            logger.info(f'{str_instance} exit - {time.strftime("%Y-%m-%d - %H:%M:%S")}')
    finally:
        if is_local:
            log_file_name = common.G_PROGRAM_NAME.replace('py', 'log')
            log_file_name = f'log/{log_file_name}'

            shutil.copyfile(log_file_name, os.path.join(str_output_dir, os.path.basename(log_file_name)))

            # prografile copy
            program_path = f"{os.getcwd()}/NSCM_DP_UI_Develop/{str_instance}.py"
            shutil.copyfile(program_path, os.path.join(str_output_dir, os.path.basename(program_path)))


        logger.Finish()
        logger.warning(f'{str_instance} {time.strftime("%Y-%m-%d - %H:%M:%S")}::: Finish :::') # 25.05.12 need warning Log by Logger Issue
        



"""
# ═══════════════════════════════════════════════════════════════════════════════
#  DuckDB QUICK CHECK (로컬 디버깅용 예시)  –  주석 해제 후 사용
# ═══════════════════════════════════════════════════════════════════════════════
# import duckdb
# duckdb.register('df_out', df_output_SellOut_FCST_Not_Exist)
# print(duckdb.query('SELECT COUNT(*) AS rows, COUNT(DISTINCT "Item.[Item]") AS uniq_item FROM df_out').to_df())
"""
