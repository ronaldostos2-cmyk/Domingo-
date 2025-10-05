import pandas as pd

def klines_to_df(klines):
    cols = ['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_av','taker_quote_av','ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df
