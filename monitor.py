import streamlit as st
from dotenv import load_dotenv
import os, pandas as pd, time
from .exchange import BinanceExchange
from .utils import klines_to_df

load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TESTNET = os.getenv('BINANCE_TESTNET','True') in ('True','true','1')
SYMBOL = os.getenv('SYMBOL','BTCUSDT')

exchange = BinanceExchange(API_KEY, API_SECRET, testnet=TESTNET)
st.title('Bot Trader Monitor - Binance Testnet (Transformer)')

interval = st.selectbox('Kline Interval', ['1m','3m','5m','15m','1h'])
limit = st.slider('Candles', 50, 500, 200)

if st.button('Refresh') or st.session_state.get('auto', False):
    try:
        klines = exchange.get_klines(SYMBOL, interval=interval, limit=limit)
        df = klines_to_df(klines)
        st.dataframe(df.tail(20))
        st.line_chart(df['close'])
    except Exception as e:
        st.error('Could not fetch: ' + str(e))

st.markdown('---')
st.write('Balances:')
try:
    acc = exchange.client.get_account()
    bal = pd.DataFrame(acc['balances'])
    bal['free'] = bal['free'].astype(float)
    st.dataframe(bal[bal['free'] > 0.0])
except Exception as e:
    st.error('Could not fetch account: ' + str(e))
