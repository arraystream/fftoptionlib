from .helper import call_to_put


class FourierPricer(object):
    def __init__(self, option=None):
        self.option = option
        self._pricing_engine = None
        self._log_st_process = None

    def set_log_st_process(self, log_st_process):
        self._log_st_process = log_st_process
        return self

    def set_pricing_engine(self, price_engine):
        self._pricing_engine = price_engine
        return self

    def get_pricing_engine(self):
        return self._pricing_engine

    def get_log_st_process(self):
        return self._log_st_process

    def calc_price(self, strike, put_call, put_label='put'):
        call_prices = self._pricing_engine(
            strike=strike,
            r=self.option.get_zero_rate(),
            t=self.option.get_time_to_maturity(),
            q=self.option.get_dividend(),
            S0=self.option.get_underlying_close_price(),
            chf_ln_st=self._log_st_process,
        )
        res = call_to_put(
            call_prices, put_call, strike,
            self.option.get_discount_bond_price(), self.option.get_forward_price(), put_label)
        return res
