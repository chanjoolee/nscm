import os
import datetime
import pandas as pd
import psutil

G_VD_SourceVersion = 'VDCommon Version : 250312'
G_logger = None
G_isLocal = None
G_path = None

#######################################################################################################################
# VD Python Log 로직 - Start
#######################################################################################################################
# Log Df 정의
df_log = pd.DataFrame([], columns=["Version.[Version Name]", "Python Batch.[Python Name]", "Time.[Day]", "Sequence.[Sequence]",
                                                "Python Batch Log_Start Time", "Python Batch Log_End Time", "Python Batch Log_Elasped Time", "Python Batch Log_Error Message"])
df_logDetail = pd.DataFrame([], columns=["Version.[Version Name]", "Python Batch.[Python Name]", "Time.[Day]", "Sequence.[Sequence]", "[Sequence ID].[Sequence ID]",
                                                    "Python Batch Log Detail_Desc", "Python Batch Log Detail_Column", "Python Batch Log Detail_Row", "Python Batch Log Detail_DataFrame", "Python Batch Log Detail_Log Time",
                                                    "Python Batch Log Detail_Df Memory", "Python Batch Log Detail_Process Memory", "Python Batch Log Detail_Process Ratio"])
# Log 시작
def gfn_pyLog_start(version, pyName, _logger, _isLocal, _path):
    global G_logger
    G_logger = _logger
    global G_isLocal
    G_isLocal = _isLocal
    global G_path
    G_path = _path

    ymd = datetime.datetime.now().strftime("%Y-%m-%d")
    hms = int(datetime.datetime.now().strftime("%H%M%S"))
    sTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_log.loc[len(df_log)] = [version, pyName, ymd, hms, sTime, "", "", ""]

# Log 종료
def gfn_pyLog_end():
    sTime = datetime.datetime.strptime(df_log.loc[0, "Python Batch Log_Start Time"], "%Y-%m-%d %H:%M:%S")
    eTime = datetime.datetime.now()
    time_diff = eTime - sTime

    df_log.loc[0, "Python Batch Log_End Time"] = eTime.strftime("%Y-%m-%d %H:%M:%S")
    df_log.loc[0, "Python Batch Log_Elasped Time"] = round(time_diff.total_seconds(), 0)

# Log Error
def gfn_pyLog_error(msg):
    df_log.loc[0, "Python Batch Log_Error Message"] = msg

# MG에 Batch Log를 남기는 DF를 만드는 함수
def gfn_pyLog_detail(dfDesc, df=None):
    # Memory 사용량 계산
    pMemory = round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 2)
    aMemory = round(psutil.virtual_memory().available / (1024 ** 2), 2)
    pRatio = round((pMemory / aMemory) * 100, 0)

    # Grain
    version = df_log.loc[0, "Version.[Version Name]"]
    pyName = df_log.loc[0, "Python Batch.[Python Name]"]
    ymd = df_log.loc[0, "Time.[Day]"]
    hms = df_log.loc[0, "Sequence.[Sequence]"]

    if df is None:
        df_logDetail.loc[len(df_logDetail)] = [version, pyName, ymd, hms, str(len(df_logDetail) + 1),
                                                   dfDesc, "", "", "", datetime.datetime.now().strftime("%H-%M-%S"),
                                                   0, pMemory, pRatio]
    else:
        dfMemory = round(df.memory_usage(deep=False).sum() / (1024**2), 2)
        df_logDetail.loc[len(df_logDetail)] = [version, pyName, ymd, hms, str(len(df_logDetail) + 1),
                                                       dfDesc, ",".join(df.columns), str(df.shape[0]), df.head(10).to_csv(index=False), datetime.datetime.now().strftime("%H-%M-%S"),
                                                       dfMemory, pMemory, pRatio]

        if pRatio > 90:
            raise Exception("Memory utilization exceeded 90%")
        
def gfn_getLog():
    return df_log

def gfn_getLogDetail():
    return df_logDetail

# DF를 출력하는 함수
saveDf_idx = 1
def gfn_saveDf(_df, _csvName, _localFlag=False):
    if _localFlag & G_isLocal:
        global saveDf_idx
        _df.to_csv(G_path + "\\" + f"out_{saveDf_idx}_{_csvName}.csv", index=False)
        saveDf_idx = saveDf_idx + 1

    # dfMemory = format(round(_df.memory_usage(deep=False).sum() / (1024 ** 2), 2), ",")
    dfMemory = format(round(_df.to_numpy().nbytes / (1024 ** 2), 2), ",")
    dfDetailInfo = " ( " + format(_df.shape[0], ",") + " Row * " + str(_df.shape[1]) + " Column, " + dfMemory + " Mb )"

    G_logger.PrintDF(p_df=_df, p_df_name=_csvName + dfDetailInfo, p_log_level = 20, p_format=1, p_row_num=10)
    gfn_pyLog_detail(_csvName, _df)

# Step 로그를 남긴다.
def gfn_stepLog(log):
    G_logger.Step(p_step_desc=log)
    # MG 로그를 남긴다.
    gfn_pyLog_detail(log)


#######################################################################################################################
# VD Python Log 로직 - End
#######################################################################################################################
