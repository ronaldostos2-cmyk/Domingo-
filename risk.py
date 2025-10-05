import math

class RiskManager:
    def __init__(self, account_balance: float, risk_per_trade: float = 0.01):
        self.balance = account_balance
        self.risk_per_trade = risk_per_trade

    def update_balance(self, new_balance: float):
        self.balance = new_balance

    def position_size(self, price: float, stop_distance: float):
        risk_capital = self.balance * self.risk_per_trade
        if stop_distance <= 0:
            return 0.0
        qty = risk_capital / stop_distance
        return max(0, qty)

    def factorize(self, base_qty: float, model_confidence: float, min_qty: float = 0.000001):
        factor = max(0.1, min(3.0, 1 + (model_confidence - 0.5) * 2))
        qty = base_qty * factor
        return max(min_qty, qty)
