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
    vasicek_int_rt_chf,
    cir_int_rt_chf,
    vg_cgm_chf,
    vg_chf,
    parameter_to_cgm,
    vg_log_st_chf,
    nig_log_st_chf
)


class TestCharacteristicFunctions(unittest.TestCase):
    def test_black_shole_log_st_chf(self):
        v, S, t, r, q, sigma = 4, 20.2, 1.5, 0.02, 0.01, 0.2
        res = black_schole_log_st_chf(v, t, r, q, S, sigma)
        exp_res = 0.5094287369306635 - 0.3512481287698781j
        self.assertAlmostEqual(res, exp_res)

    def test_merton_jump_log_st_chf(self):
        v, S, t, r, q, sigma, jp_lamda, norm_mean, norm_sd = 4, 20.2, 1.5, 0.02, 0.01, 0.2, 0, 0, 0
        res = merton_jump_log_st_chf(v, t, r, q, S, sigma, jp_lamda, norm_mean, norm_sd)
        exp_res = 0.5094287369306635 - 0.3512481287698781j
        self.assertAlmostEqual(res, exp_res)

    def test_kou_jump_log_st_chf_1(self):
        v, S, t, r, q, sigma, jp_lamda, exp_pos, exp_neg, pos_prob = 4, 20.2, 1.5, 0.02, 0.01, 0.2, 0.1, 0.2, 0.4, 0.3
        res = kou_jump_log_st_chf(v, t, r, q, S, sigma, jp_lamda, exp_pos, exp_neg, pos_prob)
        exp_res = 0.53119859429316973 - 0.046219820879423709j
        self.assertAlmostEqual(res, exp_res)

    def test_merton_jump_log_st_chf_1(self):
        v, S, t, r, q, sigma, jp_lamda, norm_mean, norm_sd = 4, 20.2, 1.5, 0.02, 0.01, 0.2, 0, 0, 0
        res = merton_jump_log_st_chf(v, t, r, q, S, sigma, jp_lamda, norm_mean, norm_sd)
        exp_res = 0.5094287369306635 - 0.3512481287698781j
        self.assertAlmostEqual(res, exp_res)

    def test_poisson_log_st_chf(self):
        v, S, t, r, q, jp_lamda = 4, 20.2, 1.5, 0.02, 0.01, 0.2
        res = poisson_log_st_chf(v, t, r, q, S, jp_lamda)
        exp_res = -0.56792446520231077 - 0.21960657867485839j
        self.assertAlmostEqual(res, exp_res)

    def test_diffusion_chf(self):
        u, t, sigma = 2, 1.5, 0.2
        exp_res = 0.88692043671715748
        res = diffusion_chf(u, t, sigma)
        self.assertAlmostEqual(exp_res, res)

    def test_cpp_normal_chf(self):
        u, t, jump_rate, norm_m, norm_sig = 2, 1.5, 0.5, 1, 0.6
        exp_res = 0.38363681311594483 + 0.13224239725064765j
        res = cpp_normal_chf(u, t, jump_rate, norm_m, norm_sig)
        self.assertAlmostEqual(exp_res, res)

    def test_double_exponential_chf(self):
        u, exp_pos, exp_neg, prob_pos = 2, 0.5, 1, 0.6
        exp_res = 0.11529411764705881 - 0.018823529411764697j
        res = double_exponential_chf(u, exp_pos, exp_neg, prob_pos)
        self.assertAlmostEqual(exp_res, res)

    def test_poisson_chf(self):
        u, t, jump_rate = 2, 0.5, 0.5
        exp_res = 0.6837926548536003 + 0.158176826255608j
        res = poisson_chf(u, t, jump_rate)
        self.assertAlmostEqual(exp_res, res)

    def test_vasicek_int_rt_chf(self):
        u, t, k, theta, sigma, r0 = 2, 0.5, 0.5, 0.6, 0.4, 0.2
        exp_res = 0.84624374285630455 + 0.21255258540192432j
        res = vasicek_int_rt_chf(u, t, k, theta, sigma, r0)
        self.assertAlmostEqual(exp_res, res)

    def test_cir_int_rt_chf(self):
        u, t, k, theta, sigma, r0 = 2, 0.5, 0.5, 0.6, 0.4, 0.2
        exp_res = 0.96746751276624199 + 0.24296559703146423j
        res = cir_int_rt_chf(u, t, k, theta, sigma, r0)
        self.assertAlmostEqual(exp_res, res)

    def test_parameter_to_cgm(self):
        u, t, theta, v, sigma = 2, 0.5, 0.5, 0.7, 0.3
        res = vg_chf(u, t, theta, v, sigma)
        c, g, m = parameter_to_cgm(theta, v, sigma)
        exp_res = vg_cgm_chf(u, t, c, g, m)
        self.assertAlmostEqual(exp_res, res)

    def test_vg_log_st_chf(self):
        u, S, t, r, q, theta, v, sigma = 4, 20.2, 1.5, 0.02, 0.01, 0.5, 0.7, 0.3
        exp_res = -0.21142343124302654 - 0.031095014012680902j
        res = vg_log_st_chf(u, t, r, q, S, theta, v, sigma)
        self.assertAlmostEqual(exp_res, res)

    def test_nig_log_st_chf(self):
        u, S, t, r, q, a, b, delta = 4, 20.2, 1.5, 0.01, 0.01, 2, 0.3, 0.6
        exp_res = -0.039178193810523385 - 0.09858914992954515j
        res = nig_log_st_chf(u, t, r, q, S, a, b, delta)
        self.assertAlmostEqual(exp_res, res)
