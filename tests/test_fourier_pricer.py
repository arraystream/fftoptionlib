# -*- coding: utf-8 -*-

import unittest

from fftoptionlib.fourier_pricer import carr_madan_fft_call_pricer, carr_madan_fraction_fft_call_pricer
from fftoptionlib.helper import spline_fitting
from fftoptionlib.process_class import BlackSchole, Heston, VarianceGamma


class TestFourierPricer(unittest.TestCase):
    def test_carr_madan_fft_call_pricer_black_shole(self):
        N = 2 ** 15
        d_u = 0.01
        alpha = 1
        S0 = 100
        t = 1
        r = 0.05
        q = 0
        sigma = 0.3
        strike, call_prices = carr_madan_fft_call_pricer(N, d_u, alpha, r, t, S0, q, BlackSchole(sigma).set_type('chf'))
        ffn_pricer = spline_fitting(strike, call_prices, 2)
        exp_res = 19.6974
        res = ffn_pricer(90)
        self.assertAlmostEqual(exp_res, res, 4)

    def test_carr_madan_fraction_fft_call_pricer(self):
        N = 2 ** 12
        d_u = 0.01
        d_k = 0.1
        alpha = 1
        S0 = 100
        t = 1
        r = 0.05
        q = 0
        sigma = 0.3
        strike, call_prices = carr_madan_fraction_fft_call_pricer(N, d_u, d_k, alpha, r, t, S0, q, BlackSchole(sigma).set_type('chf'))
        ffn_pricer = spline_fitting(strike, call_prices, 2)
        exp_res = 19.6974
        res = ffn_pricer(90)
        self.assertAlmostEqual(exp_res, res, 4)

    def test_carr_madan_fft_heston(self):
        N = 2 ** 15
        d_u = 0.01
        alpha = 1

        S0 = 100
        t = 0.5
        k = 2
        r = 0.03
        q = 0
        sigma = 0.5
        theta = 0.04
        V0 = 0.04
        rho = -0.7
        strike, call_prices = carr_madan_fft_call_pricer(N, d_u, alpha, r, t, S0, q, Heston(V0, theta, k, sigma, rho).set_type('chf'))
        ffn_pricer = spline_fitting(strike, call_prices, 2)
        exp_res = 13.2023
        res = ffn_pricer(90)
        self.assertAlmostEqual(exp_res, res, 4)

    def test_carr_madan_fractional_fft_heston(self):
        N = 2 ** 13
        d_u = 0.01
        d_k = 0.01
        alpha = 1

        S0 = 100
        t = 0.5
        k = 2
        r = 0.03
        q = 0
        sigma = 0.5
        theta = 0.04
        V0 = 0.04
        rho = -0.7
        strike, call_prices = carr_madan_fraction_fft_call_pricer(N, d_u, d_k, alpha, r, t, S0, q, Heston(V0, theta, k, sigma, rho).set_type('chf'))
        ffn_pricer = spline_fitting(strike, call_prices, 2)
        exp_res = 13.2023
        res = ffn_pricer(90)
        self.assertAlmostEqual(exp_res, res, 4)

    def test_carr_madan_fft_vg(self):
        N = 2 ** 15
        d_u = 0.01
        alpha = 1

        S0 = 100
        t = 1 / 12
        r = 0.1
        q = 0
        sigma = 0.12
        theta = -0.14
        v = 0.2
        strike, call_prices = carr_madan_fft_call_pricer(N, d_u, alpha, r, t, S0, q, VarianceGamma(theta, v, sigma).set_type('chf'))
        ffn_pricer = spline_fitting(strike, call_prices, 2)
        exp_res = 10.8289
        res = ffn_pricer(90)
        self.assertAlmostEqual(exp_res, res, 4)

    def test_carr_madan_fractional_fft_vg(self):
        N = 2 ** 16
        d_u = 0.01
        d_k = 0.01
        alpha = 1
        S0 = 100
        t = 1 / 12
        r = 0.1
        q = 0
        sigma = 0.12
        theta = -0.14
        v = 0.2
        strike, call_prices = carr_madan_fraction_fft_call_pricer(N, d_u, d_k, alpha, r, t, S0, q, VarianceGamma(theta, v, sigma).set_type('chf'))
        ffn_pricer = spline_fitting(strike, call_prices, 2)
        exp_res = 10.8289
        res = ffn_pricer(90)
        self.assertAlmostEqual(exp_res, res, 4)
