from sqlalchemy import create_engine
import pandas as pd
import os


def create_engin(con_str='mysql+mysqlconnector://datatec:0.618@[172.16.103.103]:3306/JYDB'):
    return create_engine(con_str, echo=False)

def read_index_from_sql(secucode, begin_t , end_t):
    eng = create_engin()
    begin = begin_t
    end = end_t
    sql_syntax = '''SELECT T1.OpenPrice,T1.HighPrice,T1.LowPrice,T1.ClosePrice,T1.TurnoverVolume,T1.TurnoverValue , T1.TradingDay
    FROM QT_DailyQuote T1
    Inner Join SecuMain T2 ON T2.InnerCode = T1.InnerCode
    Where T1.TradingDay>='{begin_t}' and T1.TradingDay<='{end_t}'
    	And T2.SecuCode = '{secucode}' And T2.SecuCategory=4
    	Order By T1.TradingDay '''.format(begin_t=begin,end_t=end,secucode=secucode)
    df = pd.read_sql_query(sql_syntax, eng, index_col=["TradingDay"], parse_dates=["TradingDay"])
    df.to_pickle('data/%s.pkl'%secucode)
    return df

def read_index_from_pkl(configs):
    secucode = configs['secucode']
    path = 'data/%s.pkl'% secucode
    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        df=read_index_from_pkl(secucode, configs['start_time'], configs['end_time'])
    return df
