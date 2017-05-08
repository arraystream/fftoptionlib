# -*- coding: utf-8 -*-

import unittest

import numpy as np

from fftoptionlib.engine_class import FFTEngine, FractionFFTEngine, CosineEngine
from fftoptionlib.option_class import BasicOption
from fftoptionlib.pricing_class import FourierPricer
from fftoptionlib.process_class import BlackSchole


class TestPricingClass(unittest.TestCase):
    def setUp(self):
        underlying_price = 36
        maturity_date = '1999-05-17'
        risk_free_rate = 0.06
        evaluation_date = '1999-04-22'
        self.vanilla_option = BasicOption()
        (self.vanilla_option.set_underlying_close_price(underlying_price)
         .set_dividend(0.01)
         .set_maturity_date(maturity_date)
         .set_evaluation_date(evaluation_date)
         .set_zero_rate(risk_free_rate))

    def test_cal_price_fft(self):
        fft_pricer = FourierPricer(self.vanilla_option)
        strike_arr = np.array([5, 10, 30, 36, 50, 60, 100])
        put_call_arr = np.array(['call', 'call', 'put', 'call', 'call', 'put', 'call'])
        exp = np.array(
            [3.09958567e+01, 2.60163625e+01, 8.25753140e-05, 8.12953226e-01,
             8.97449491e-11, 2.37785797e+01, 2.19293560e-85, ]
        )

        volatility = 0.20
        N = 2 ** 15
        d_u = 0.01
        alpha = 1
        fft_pricer.set_log_st_process(BlackSchole(volatility))
        fft_pricer.set_pricing_engine(FFTEngine(N, d_u, alpha, spline_order=3))
        res = fft_pricer.calc_price(strike_arr, put_call_arr, put_label='put')
        np.testing.assert_array_almost_equal(res, exp, 6)

    def test_cal_price_fractional_fft(self):
        fft_pricer = FourierPricer(self.vanilla_option)
        strike_arr = np.array([5, 10, 30, 36, 50, 60, 100])
        put_call_arr = np.array(['call', 'call', 'put', 'call', 'call', 'put', 'call'])
        exp = np.array(
            [3.09958567e+01, 2.60163625e+01, 8.25753140e-05, 8.12953226e-01,
             8.97449491e-11, 2.37785797e+01, 2.19293560e-85, ]
        )
        volatility = 0.20
        N = 2 ** 14
        d_u = 0.01
        d_k = 0.01
        alpha = 1
        fft_pricer.set_log_st_process(BlackSchole(volatility))
        fft_pricer.set_pricing_engine(FractionFFTEngine(N, d_u, d_k, alpha, spline_order=3))
        res = fft_pricer.calc_price(strike_arr, put_call_arr, put_label='put')
        np.testing.assert_array_almost_equal(res, exp, 6)

    def test_cal_price_cosine(self):
        cosine_pricer = FourierPricer(self.vanilla_option)
        strike_arr = np.array([5, 10, 30, 36, 50, 60, 100])
        put_call_arr = np.array(['call', 'call', 'put', 'call', 'call', 'put', 'call'])
        exp = np.array([3.09958567e+01, 2.60163625e+01, 8.25753140e-05, 8.12953226e-01,
                        8.97449491e-11, 2.37785797e+01, 2.19293560e-85, ]
                       )

        volatility = 0.20
        N = 150

        cosine_pricer.set_log_st_process(BlackSchole(volatility))
        cosine_pricer.set_pricing_engine(CosineEngine(N, L=30))
        res = cosine_pricer.calc_price(strike_arr, put_call_arr, put_label='put')
        np.testing.assert_array_almost_equal(res, exp, 6)
