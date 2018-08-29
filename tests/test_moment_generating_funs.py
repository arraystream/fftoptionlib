import unittest

from fftoptionlib.characteristic_funs import (
    black_schole_log_st_chf,
    merton_jump_log_st_chf,
    poisson_log_st_chf,
    diffusion_chf,
    cpp_normal_chf,
    kou_jump_log_st_chf,
    double_exponential_chf,
    poisson_chf,
    vg_log_st_chf,
    nig_log_st_chf,
)
from fftoptionlib.moment_generating_funs import (
    black_scholes_log_st_mgf,
    merton_jump_log_st_mgf,
    poisson_log_st_mgf,
    diffusion_mgf,
    cpp_normal_mgf,
    kou_jump_log_st_mgf,
    double_exponential_mgf,
    poisson_mgf,
    vg_cgm_mgf,
    vg_mgf,
    parameter_to_cgm,
    vg_log_st_mgf,
    nig_log_st_mgf,
)


class TestMomentGeneratingFunctions(unittest.TestCase):
    def test_black_shole_log_st_mgf(self):
        v, S, t, r, q, sigma = 4, 20.2, 1.5, 0.02, 0.01, 0.2
        res = black_scholes_log_st_mgf(v, t, r, q, S, sigma)
        exp_res = black_schole_log_st_chf(v * (-1j), t, r, q, S, sigma)
        self.assertAlmostEqual(res, exp_res)

    def test_merton_jump_log_st_mgf(self):
        v, S, t, r, q, sigma, jp_lamda, norm_mean, norm_sd = 4, 20.2, 1.5, 0.02, 0.01, 0.2, 0, 0, 0
        res = merton_jump_log_st_mgf(v, t, r, q, S, sigma, jp_lamda, norm_mean, norm_sd)
        exp_res = merton_jump_log_st_chf(-1j * v, t, r, q, S, sigma, jp_lamda, norm_mean, norm_sd)
        self.assertAlmostEqual(res, exp_res)

    def test_kou_jump_log_st_mgf_1(self):
        v, S, t, r, q, sigma, jp_lamda, exp_pos, exp_neg, pos_prob = 4, 20.2, 1.5, 0.02, 0.01, 0.2, 0.1, 0.2, 0.4, 0.3
        res = kou_jump_log_st_mgf(v, t, r, q, S, sigma, jp_lamda, exp_pos, exp_neg, pos_prob)
        exp_res = kou_jump_log_st_chf(-1j * v, t, r, q, S, sigma, jp_lamda, exp_pos, exp_neg, pos_prob)
        self.assertAlmostEqual(res, exp_res)

    def test_merton_jump_log_st_mgf_1(self):
        v, S, t, r, q, sigma, jp_lamda, norm_mean, norm_sd = 4, 20.2, 1.5, 0.02, 0.01, 0.2, 0, 0, 0
        res = merton_jump_log_st_mgf(v, t, r, q, S, sigma, jp_lamda, norm_mean, norm_sd)
        exp_res = merton_jump_log_st_chf(-1j * v, t, r, q, S, sigma, jp_lamda, norm_mean, norm_sd)
        self.assertAlmostEqual(res, exp_res)

    def test_poisson_log_st_mgf(self):
        v, S, t, r, q, jp_lamda = 4, 20.2, 1.5, 0.02, 0.01, 0.2
        res = poisson_log_st_mgf(v, t, r, q, S, jp_lamda)
        exp_res = poisson_log_st_chf(-1j * v, t, r, q, S, jp_lamda)
        self.assertAlmostEqual(res, exp_res)

    def test_diffusion_mgf(self):
        u, t, sigma = 2, 1.5, 0.2
        exp_res = diffusion_chf(-1j * u, t, sigma)
        res = diffusion_mgf(u, t, sigma)
        self.assertAlmostEqual(exp_res, res)

    def test_cpp_normal_mgf(self):
        u, t, jump_rate, norm_m, norm_sig = 2, 1.5, 0.5, 1, 0.6
        exp_res = cpp_normal_chf(-1j * u, t, jump_rate, norm_m, norm_sig)
        res = cpp_normal_mgf(u, t, jump_rate, norm_m, norm_sig)
        self.assertAlmostEqual(exp_res, res)

    def test_double_exponential_mgf(self):
        u, exp_pos, exp_neg, prob_pos = 2, 0.5, 1, 0.6
        exp_res = double_exponential_chf(-1j * u, exp_pos, exp_neg, prob_pos)
        res = double_exponential_mgf(u, exp_pos, exp_neg, prob_pos)
        self.assertAlmostEqual(exp_res, res)

    def test_poisson_mgf(self):
        u, t, jump_rate = 2, 0.5, 0.5
        exp_res = poisson_chf(-1j * u, t, jump_rate)
        res = poisson_mgf(u, t, jump_rate)
        self.assertAlmostEqual(exp_res, res)

    def test_parameter_to_cgm(self):
        u, t, theta, v, sigma = 2, 0.5, 0.5, 0.7, 0.3
        res = vg_mgf(u, t, theta, v, sigma)
        c, g, m = parameter_to_cgm(theta, v, sigma)
        exp_res = vg_cgm_mgf(u, t, c, g, m)
        self.assertAlmostEqual(exp_res, res)

    def test_vg_log_st_mgf(self):
        u, S, t, r, q, theta, v, sigma = 4, 20.2, 1.5, 0.02, 0.01, 0.5, 0.7, 0.3
        exp_res = vg_log_st_chf(-1j * u, t, r, q, S, theta, v, sigma)
        res = vg_log_st_mgf(u, t, r, q, S, theta, v, sigma)
        self.assertAlmostEqual(exp_res, res)

    def test_nig_log_st_mgf(self):
        u, S, t, r, q, a, b, delta = 0.5, 20.2, 1.5, 0.01, 0.01, 2, 0.3, 0.6
        exp_res = nig_log_st_chf(-1j * u, t, r, q, S, a, b, delta)
        res = nig_log_st_mgf(u, t, r, q, S, a, b, delta)
        self.assertAlmostEqual(exp_res, res)
