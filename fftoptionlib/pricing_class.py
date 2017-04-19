# -*- coding: utf-8 -*-

import abc

import numpy as np

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
from .cosine_pricer import (
    cosine_vanilla_call,
    interval_a_and_b,
)
from .fourier_pricer import (
    carr_madan_fft_call_pricer,
    carr_madan_fraction_fft_call_pricer,
)
from .helper import call_to_put, spline_fitting
from .moment_generating_funs import (
    cumulants_from_mgf,
    general_log_moneyness_mgf,
    black_schole_log_st_mgf,
    poisson_log_st_mgf,
    merton_jump_log_st_mgf,
    kou_jump_log_st_mgf,
    vg_log_st_mgf,
    nig_log_st_mgf,
    heston_log_st_mgf,
    cgmy_log_st_mgf,
)


class FourierPricer(abc.ABC):
    def __init__(self, option=None):
        self.option = option
        self._pricing_engine = None
        self._pricing_engine_kwargs = None
        self._log_st_characteristic_fun = None
        self._log_st_characteristic_fun_kwargs = None

    def set_log_st_characteristic_fun(self, log_st_chf, **log_st_chf_kwargs):
        self._log_st_characteristic_fun = log_st_chf
        self._log_st_characteristic_fun_kwargs = log_st_chf_kwargs
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

    def get_log_st_characteristic_fun_kwargs(self):
        return self._log_st_characteristic_fun_kwargs

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

log_st_mgf_dict = {
    'black_shole': black_schole_log_st_mgf,
    'poisson': poisson_log_st_mgf,
    'merton_jump': merton_jump_log_st_mgf,
    'kou_jump': kou_jump_log_st_mgf,
    'vg': vg_log_st_mgf,
    'nig': nig_log_st_mgf,
    'heston': heston_log_st_mgf,
    'cgmy': cgmy_log_st_mgf
}

pricing_engine_dict = {
    'fft': carr_madan_fft_call_pricer,
    'fractional_fft': carr_madan_fraction_fft_call_pricer,
    'cosine': cosine_vanilla_call
}


class CarrMadanFFT(FourierPricer):
    def __init__(self, option=None):
        FourierPricer.__init__(self, option)

    def calc_price(self, strike, put_call, spline_order=3, put_label='put'):
        price_engine = pricing_engine_dict[self._pricing_engine]
        chf = log_st_chf_dict[self.get_log_st_characteristic_fun()]

        pricer_kwargs = {}
        pricer_kwargs.update(self.get_pricing_engine_kwargs())
        pricer_kwargs.update(self.get_log_st_characteristic_fun_kwargs())

        sim_strikes, call_prices = price_engine(
            r=self.option.get_zero_rate(),
            t=self.option.get_time_to_maturity(),
            S0=self.option.get_underlying_close_price(),
            chf_ln_st=chf,
            q=self.option.get_dividend(),
            **pricer_kwargs
        )
        ffn_prices = spline_fitting(sim_strikes, call_prices, spline_order)(strike)
        ffn_prices = call_to_put(
            ffn_prices, put_call, strike,
            self.option.get_discount_bond_price(), self.option.get_forward_price(), put_label)
        return ffn_prices


class CosinePricer(FourierPricer):
    def __init__(self, option=None):
        FourierPricer.__init__(self, option)

    def calc_integeral_interval(self, strike, L):
        mgf = log_st_mgf_dict[self.get_log_st_characteristic_fun()]
        c1, c2, c4 = cumulants_from_mgf(
            general_log_moneyness_mgf,
            strike,
            mgf,
            t=self.option.get_time_to_maturity(),
            r=self.option.get_zero_rate(),
            q=self.option.get_dividend(),
            S0=self.option.get_underlying_close_price(),
            **self.get_log_st_characteristic_fun_kwargs())
        a, b = interval_a_and_b(c1, c2, c4, L)
        return a, b

    def _single_cal_price(self, strike, L):
        a, b = self.calc_integeral_interval(strike, L)
        price_engine = pricing_engine_dict[self._pricing_engine]
        chf = log_st_chf_dict[self.get_log_st_characteristic_fun()]

        pricer_kwargs = {}
        pricer_kwargs.update(self.get_pricing_engine_kwargs())
        pricer_kwargs.update(self.get_log_st_characteristic_fun_kwargs())

        call_price = price_engine(
            strike=strike,
            intv_a=a,
            intv_b=b,
            r=self.option.get_zero_rate(),
            t=self.option.get_time_to_maturity(),
            S0=self.option.get_underlying_close_price(),
            chf_ln_st=chf,
            q=self.option.get_dividend(),
            **pricer_kwargs)

        return call_price

    def calc_price(self, strike, put_call, L=10, put_label='put'):
        call_prices = [self._single_cal_price(one_strike, L) for one_strike in strike]
        cosine_prices = call_to_put(
            np.array(call_prices), put_call, strike,
            self.option.get_discount_bond_price(),
            self.option.get_forward_price(), put_label)
        return cosine_prices
