import copy
from datetime import datetime

import numpy as np
from pandas import Timestamp, DateOffset


class BasicOption(object):
    def __init__(self):
        self._underlying_close_price = None
        self._dividend = 0.0
        self._expiry_date = None
        self._trade_date = None
        self._forward_price = None
        self._exercise_type = 'european'
        self._zero_rate = None

    def set_exercise_type(self, exercise_type):
        self._exercise_type = exercise_type
        return self

    def set_time_to_maturity(self, time_to_maturity_in_days):
        if self.get_evaluation_date() is None:
            self.set_evaluation_date(Timestamp(datetime.now().strftime('%Y-%m-%d')))
        self.set_maturity_date(self.get_evaluation_date() + DateOffset(days=time_to_maturity_in_days))
        return self

    def set_forward_price(self, forward_price):
        self._forward_price = forward_price
        return self

    def set_underlying_close_price(self, underlying_price):
        self._underlying_close_price = underlying_price
        return self

    def set_zero_rate(self, risk_free_rate):
        self._zero_rate = risk_free_rate
        return self

    def set_dividend(self, dividend):
        self._dividend = float(dividend)
        return self

    def set_evaluation_date(self, evaluation_date):
        self._trade_date = Timestamp(evaluation_date)
        return self

    def set_maturity_date(self, maturity_date):
        self._expiry_date = Timestamp(maturity_date)
        return self

    def get_underlying_close_price(self):
        return self._underlying_close_price

    def get_zero_rate(self):
        return self._zero_rate

    def get_dividend(self):
        return self._dividend

    def get_evaluation_date(self):
        return self._trade_date

    def get_expiry_date(self):
        return self._expiry_date

    def get_discount_bond_price(self):
        return 1.0 / np.exp(self.get_zero_rate() * self.get_time_to_maturity())

    def get_exercise_type(self):
        return self._exercise_type

    def get_duration(self):
        """
        :return: time in days
        """
        return (self.get_expiry_date() - self.get_evaluation_date()).days

    def get_forward_price(self):
        if self._forward_price is None:
            return self.get_underlying_close_price() * np.exp(
                (self.get_zero_rate() - self.get_dividend()) * self.get_time_to_maturity())
        else:
            return self._forward_price

    def get_time_to_maturity(self, annualization_factor=365):
        return self.get_duration() / annualization_factor

    def to_dict(self):
        res = {'underlying_close_price': self._underlying_close_price,
               'dividend': self._dividend,
               'expiry_date': self._expiry_date,
               'trade_date': self._trade_date,
               'forward_price': self._forward_price,
               'zero_rate': self._zero_rate,
               'exercise_type': self._exercise_type}
        return res

    def serialize(self):
        return self.to_dict()

    def deserialize(self, serial_dict):
        self._underlying_close_price = serial_dict['underlying_close_price']
        self._dividend = serial_dict['dividend']
        self._expiry_date = serial_dict['expiry_date']
        self._trade_date = serial_dict['trade_date']
        self._forward_price = serial_dict['forward_price']
        self._zero_rate = serial_dict['zero_rate']
        self._exercise_type = serial_dict['exercise_type']

    def copy(self):
        return copy.deepcopy(self)
