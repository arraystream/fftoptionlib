# -*- coding: utf-8 -*-

import abc

from .characteristic_funs import (
    black_schole_log_st_chf,
    poisson_log_st_chf,
    merton_jump_log_st_chf,
    kou_jump_log_st_chf,
    vg_log_st_chf,
    nig_log_st_chf,
    heston_log_st_chf,
    cgmy_log_st_chf,
)
from .fourier_pricer import (
    carr_madan_fft_call_pricer,
    carr_madan_fraction_fft_call_pricer,
)
from .helper import call_to_put, spline_fitting


class FourierPricer(abc.ABC):
    def __init__(self, option=None):
        self.option = option
        self._pricing_engine = None
        self._pricing_engine_kwargs = None
        self._log_st_characteristic_fun = None
        self._log_st_characteristic_fun_model_kwargs = None

    def set_log_st_characteristic_fun(self, log_st_chf, **log_st_chf_kwargs):
        self._log_st_characteristic_fun = log_st_chf
        self._log_st_characteristic_fun_model_kwargs = log_st_chf_kwargs
        return self

    def set_pricing_engine(self, price_engine, **price_engine_kwargs):
        self._pricing_engine = price_engine
        self._pricing_engine_kwargs = price_engine_kwargs
        return self

    def get_pricing_engine(self):
        return self._pricing_engine

    def get_pricing_engine_kwargs(self):
        return self._pricing_engine_kwargs

    def get_log_st_characteristic_fun(self):
        return self._log_st_characteristic_fun

    def get_log_st_characteristic_fun_model_kwargs(self):
        return self._log_st_characteristic_fun_model_kwargs

    @abc.abstractmethod
    def calc_price(self, *args, **kwargs):
        pass


log_st_chf_dict = {
    'black_shole': black_schole_log_st_chf,
    'poisson': poisson_log_st_chf,
    'merton_jump': merton_jump_log_st_chf,
    'kou_jump': kou_jump_log_st_chf,
    'vg': vg_log_st_chf,
    'nig': nig_log_st_chf,
    'heston': heston_log_st_chf,
    'cgmy': cgmy_log_st_chf
}

pricing_engine_dict = {
    'fft': carr_madan_fft_call_pricer,
    'fractional_fft': carr_madan_fraction_fft_call_pricer
}


class CarrMadanFFT(FourierPricer):
    def __init__(self, option=None):
        FourierPricer.__init__(self, option)

    def calc_price(self, strike, put_call, spline_order=3, put_label='put'):
        price_engine = pricing_engine_dict[self._pricing_engine]
        chf = log_st_chf_dict[self.get_log_st_characteristic_fun()]
        sim_strikes, call_prices = price_engine(
            r=self.option.get_zero_rate(),
            t=self.option.get_time_to_maturity(),
            S0=self.option.get_underlying_close_price(),
            chf_ln_st=chf,
            q=self.option.get_dividend(),
            **self.get_pricing_engine_kwargs(),
            **self.get_log_st_characteristic_fun_model_kwargs()
        )
        ffn_prices = spline_fitting(sim_strikes, call_prices, spline_order)(strike)
        ffn_prices = call_to_put(
            ffn_prices, put_call, strike,
            self.option.get_discount_bond_price(), self.option.get_forward_price(), put_label)
        return ffn_prices
