import numpy as np
from .indicators import ema, rsi, macd

class Strategy:
    def __init__(self, model_agent, risk_manager, symbol: str, seq_len: int = 30):
        self.model_agent = model_agent
        self.risk = risk_manager
        self.symbol = symbol
        self.seq_len = seq_len

    def build_features_sequence(self, df):
        close = df['close']
        f = {}
        f['ema_12'] = ema(close, 12)
        f['ema_26'] = ema(close, 26)
        f['rsi_14'] = rsi(close, 14)
        macd_line, signal_line, hist = macd(close)
        # build dataframe of features
        feat_df = df.copy()
        feat_df['ema_12'] = f['ema_12']
        feat_df['ema_26'] = f['ema_26']
        feat_df['rsi_14'] = f['rsi_14']
        feat_df['macd'] = macd_line
        feat_df['macd_hist'] = hist
        feat_df = feat_df.dropna().tail(self.seq_len)
        if len(feat_df) < self.seq_len:
            return None
        X = feat_df[['ema_12','ema_26','rsi_14','macd','macd_hist']].values
        # normalize simple: percentage differences
        Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        return Xn.astype('float32')

    def decide(self, df):
        X = self.build_features_sequence(df)
        if X is None:
            return 'HOLD', 0.5, 1.0
        # model_agent returns mean prob and std (MC Dropout)
        mean, std = self.model_agent.predict_proba(X.reshape(1, X.shape[0], X.shape[1]))
        m = float(mean[0])
        s = float(std[0])
        if (m > 0.6) and (s < 0.08):
            return 'BUY', m, s
        elif (m < 0.4) and (s < 0.08):
            return 'SELL', 1 - m, s
        else:
            return 'HOLD', max(m, 1-m), s
