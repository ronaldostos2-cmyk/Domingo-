import os, time, numpy as np
from dotenv import load_dotenv
from .exchange import BinanceExchange
from .utils import klines_to_df
from .ai_model import SelfLearningTransformer
from .risk import RiskManager
from .strategy import Strategy

load_dotenv()

API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TESTNET = os.getenv('BINANCE_TESTNET','True') in ('True','true','1')
SYMBOL = os.getenv('SYMBOL','BTCUSDT')
POLL = int(os.getenv('POLL_INTERVAL', '30'))

exchange = BinanceExchange(API_KEY, API_SECRET, testnet=TESTNET)

# model uses 5 features (ema12, ema26, rsi, macd, macd_hist)
model_agent = SelfLearningTransformer(n_features=5, seq_len=30)

quote = SYMBOL[-4:]
balance = exchange.get_balance(quote) or 1000.0
risk = RiskManager(account_balance=balance, risk_per_trade=float(os.getenv('POSITION_RISK',0.01)))
strategy = Strategy(model_agent=model_agent, risk_manager=risk, symbol=SYMBOL, seq_len=30)

def safe_print(*a, **k):
    try:
        print(*a, **k)
    except Exception:
        pass

def run_loop():
    while True:
        try:
            klines = exchange.get_klines(SYMBOL, interval='1m', limit=200)
            df = klines_to_df(klines)
            decision, conf, uncert = strategy.decide(df)
            safe_print(f"Decision={decision} conf={conf:.3f} uncert={uncert:.4f}")
            price = exchange.get_price(SYMBOL)
            if decision == 'BUY' and conf > 0.55:
                stop_distance = price * 0.01
                base_qty = risk.position_size(price, stop_distance)
                qty = risk.factorize(base_qty, conf)
                if qty > 0:
                    resp = exchange.place_market_order(SYMBOL, 'BUY', qty)
                    safe_print('Order resp', resp)
            elif decision == 'SELL' and conf > 0.55:
                base_asset = SYMBOL.replace('USDT','')
                free = exchange.get_balance(base_asset)
                if free and free > 0:
                    resp = exchange.place_market_order(SYMBOL, 'SELL', free)
                    safe_print('Order resp', resp)

            # Self-labeling: naive label by future 1-candle return (you should replace with better label logic)
            future_return = (df['close'].iloc[-1] - df['open'].iloc[-1]) / (df['open'].iloc[-1] + 1e-9)
            label = 1 if future_return > 0 else 0
            X_seq = strategy.build_features_sequence(df)
            if X_seq is not None:
                model_agent.store_label(X_seq, label)

            # periodically save model
            if model_agent.steps % 200 == 0:
                model_agent.save()

        except Exception as e:
            safe_print('Loop error', e)
        time.sleep(POLL)

if __name__ == '__main__':
    run_loop()
