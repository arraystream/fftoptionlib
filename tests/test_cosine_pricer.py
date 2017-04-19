# -*- coding: utf-8 -*-

import unittest

from tslib.fourier_pricing_lib.characteristic_funs import black_schole_log_st_chf, heston_log_st_chf, vg_log_st_chf
from tslib.fourier_pricing_lib.cosin_pricer import interval_a_and_b, cosin_vanilla_call
from tslib.fourier_pricing_lib.moment_generating_funs import cumulants_from_mgf, general_log_moneyness_mgf, \
    black_schole_log_st_mgf, heston_log_st_mgf, vg_log_st_mgf


class TestCosinePricer(unittest.TestCase):
    def test_black_shole(self):
        N = 16
        L = 10
        S0 = 100
        t = 0.1
        r = 0.1
        q = 0
        sigma = 0.25

        strike = 80
        c1, c2, c4 = cumulants_from_mgf(general_log_moneyness_mgf, strike, black_schole_log_st_mgf, t, r, q, S0, sigma)
        intv_a, intv_b = interval_a_and_b(c1, c2, c4, L)
        exp_res = 20.792
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, r, t, S0, black_schole_log_st_chf, q=q, sigma=sigma)
        self.assertAlmostEqual(exp_res, res, 3)
        N = 64
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, r, t, S0, black_schole_log_st_chf, q=q, sigma=sigma)
        exp_res = 20.799
        self.assertAlmostEqual(exp_res, res, 3)

    def test_black_shole_2(self):
        N = 16
        L = 10
        S0 = 100
        t = 0.1
        r = 0.1
        q = 0
        sigma = 0.25

        strike = 100
        c1, c2, c4 = cumulants_from_mgf(general_log_moneyness_mgf, strike, black_schole_log_st_mgf, t, r, q, S0, sigma)
        intv_a, intv_b = interval_a_and_b(c1, c2, c4, L)
        exp_res = 3.659
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, r, t, S0, black_schole_log_st_chf, q=q, sigma=sigma)
        self.assertAlmostEqual(exp_res, res, 3)
        N = 64
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, r, t, S0, black_schole_log_st_chf, q=q, sigma=sigma)
        exp_res = 3.660
        self.assertAlmostEqual(exp_res, res, 3)

    def test_black_shole_3(self):
        N = 16
        L = 10
        S0 = 100
        t = 0.1
        r = 0.1
        q = 0
        sigma = 0.25

        strike = 120
        c1, c2, c4 = cumulants_from_mgf(general_log_moneyness_mgf, strike, black_schole_log_st_mgf, t, r, q, S0, sigma)
        intv_a, intv_b = interval_a_and_b(c1, c2, c4, L)
        exp_res = 0.044
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, r, t, S0, black_schole_log_st_chf, q=q, sigma=sigma)
        self.assertAlmostEqual(exp_res, res, 3)
        N = 64
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, r, t, S0, black_schole_log_st_chf, q=q, sigma=sigma)
        exp_res = 0.045
        self.assertAlmostEqual(exp_res, res, 3)

    def test_heston(self):
        N = 100
        L = 10
        S0 = 100
        t = 0.5
        k = 2
        r = 0.03
        q = 0
        sigma = 0.5
        theta = 0.04
        V0 = 0.04
        rho = -0.7
        strike = 90
        c1, c2, c4 = cumulants_from_mgf(general_log_moneyness_mgf, strike, heston_log_st_mgf, t, r, q, S0, V0, theta, k,
                                        sigma, rho)
        intv_a, intv_b = interval_a_and_b(c1, c2, c4, L)
        exp_res = 13.2023
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, r, t, S0, heston_log_st_chf, q=q, V0=V0,
                                 theta=theta, k=k, sigma=sigma, rho=rho)
        self.assertAlmostEqual(exp_res, res, 3)

    def test_vg(self):
        N = 150
        L = 10
        S0 = 100
        t = 1 / 12
        r = 0.1
        q = 0
        sigma = 0.12
        theta = -0.14
        v = 0.2
        strike = 90
        c1, c2, c4 = cumulants_from_mgf(general_log_moneyness_mgf, strike, vg_log_st_mgf, t, r, q, S0, theta, v, sigma)
        intv_a, intv_b = interval_a_and_b(c1, c2, c4, L)
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, r, t, S0, vg_log_st_chf, q=q, theta=theta, v=v,
                                 sigma=sigma)
        exp_res = 10.8289
        self.assertAlmostEqual(exp_res, res, 4)
