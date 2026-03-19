import pandas as pd

""" constant 클래스를 참조하여 o9, gscm간 데이터프레임의 컬럼명 일괄 변환 """

def getcols(ca):
    return {k: v for k in dir(ca) if not k.startswith('_') and isinstance(v:=getattr(ca, k), str)}

# gscm컬럼명을 o9컬럼명으로 변환
def gscm2o9(df, ca) -> pd.DataFrame:
    return df.rename(columns=getcols(ca))

# o9컬럼명을 gscm컬럼명으로 변환
def o92gscm(df, ca) -> pd.DataFrame:
    d=getcols(ca)
    rd = dict(zip(d.values(), d.keys()))
    return df.rename(columns=rd)

