import os, math
from binance.client import Client
import time

class BinanceExchange:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client = Client(api_key, api_secret)
        if testnet:
            # python-binance versions differ; this sets testnet endpoint
            try:
                self.client.API_URL = 'https://testnet.binance.vision/api'
            except Exception:
                pass

    def get_symbol_info(self, symbol: str):
        return self.client.get_symbol_info(symbol)

    def _format_amount(self, symbol: str, amount: float):
        info = self.get_symbol_info(symbol)
        step_size = None
        for f in info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size = float(f['stepSize'])
        if step_size and step_size > 0:
            precision = max(0, int(round(-math.log(step_size, 10))))
            fmt = '{:.' + str(precision) + 'f}'
            # floor to step_size multiple
            amt = math.floor(amount / step_size) * step_size
            return float(fmt.format(amt))
        return amount

    def place_market_order(self, symbol: str, side: str, quantity: float):
        q = self._format_amount(symbol, quantity)
        try:
            return self.client.create_order(symbol=symbol, side=side, type='MARKET', quantity=q)
        except Exception as e:
            raise

    def get_klines(self, symbol: str, interval: str = '1m', limit: int = 500):
        raw = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        return raw

    def get_balance(self, asset: str):
        try:
            balances = self.client.get_account()['balances']
            for b in balances:
                if b['asset'] == asset:
                    return float(b['free'])
        except Exception:
            return 0.0
        return 0.0

    def get_price(self, symbol: str):
        try:
            return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        except Exception:
            return 0.0
