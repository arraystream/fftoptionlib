import unittest

from fftoptionlib.cosine_pricer import (
    cosin_vanilla_call,
    interval_a_and_b,
)
from fftoptionlib.moment_generating_funs import (
    cumulants_from_mgf,
    general_log_moneyness_mgf,
)
from fftoptionlib.process_class import (
    BlackScholes,
    Heston,
    VarianceGamma,
)


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
        c1, c2, c4 = cumulants_from_mgf(general_log_moneyness_mgf, strike, BlackScholes(sigma).set_type('mgf'), t=t, r=r, q=q, S0=S0)
        intv_a, intv_b = interval_a_and_b(c1, c2, c4, L)
        print(intv_a, intv_b)
        exp_res = 20.792
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, t, r, q, S0, BlackScholes(sigma).set_type('chf'))
        self.assertAlmostEqual(exp_res, res, 3)
        N = 64
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, t, r, q, S0, BlackScholes(sigma).set_type('chf'))
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
        c1, c2, c4 = cumulants_from_mgf(general_log_moneyness_mgf, strike, BlackScholes(sigma).set_type('mgf'), t=t, r=r, q=q, S0=S0)
        intv_a, intv_b = interval_a_and_b(c1, c2, c4, L)
        print(intv_a, intv_b)

        exp_res = 3.659
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, t, r, q, S0, BlackScholes(sigma).set_type('chf'))
        self.assertAlmostEqual(exp_res, res, 3)
        N = 64
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, t, r, q, S0, BlackScholes(sigma).set_type('chf'))
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
        c1, c2, c4 = cumulants_from_mgf(general_log_moneyness_mgf, strike, BlackScholes(sigma).set_type('mgf'), t=t, r=r, q=q, S0=S0)
        intv_a, intv_b = interval_a_and_b(c1, c2, c4, L)
        print(intv_a, intv_b)
        exp_res = 0.044
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, t, r, q, S0, BlackScholes(sigma).set_type('chf'))
        self.assertAlmostEqual(exp_res, res, 3)
        N = 64
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, t, r, q, S0, BlackScholes(sigma).set_type('chf'))
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
        c1, c2, c4 = cumulants_from_mgf(general_log_moneyness_mgf, strike, Heston(V0, theta, k, sigma, rho).set_type('mgf'), t=t, r=r, q=q, S0=S0)
        intv_a, intv_b = interval_a_and_b(c1, c2, c4, L)
        print(intv_a, intv_b)
        exp_res = 13.2023
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, t, r, q, S0, Heston(V0, theta, k, sigma, rho).set_type('chf'))
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
        c1, c2, c4 = cumulants_from_mgf(general_log_moneyness_mgf, strike, VarianceGamma(theta, v, sigma).set_type('mgf'), t=t, r=r, q=q, S0=S0)
        intv_a, intv_b = interval_a_and_b(c1, c2, c4, L)
        print(intv_a, intv_b)
        res = cosin_vanilla_call(N, strike, intv_a, intv_b, t, r, q, S0, VarianceGamma(theta, v, sigma).set_type('chf'))
        exp_res = 10.8289
        self.assertAlmostEqual(exp_res, res, 4)
