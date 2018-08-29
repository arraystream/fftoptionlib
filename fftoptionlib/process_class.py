import abc

from fftoptionlib.characteristic_funs import (
    black_schole_log_st_chf,
    merton_jump_log_st_chf,
    kou_jump_log_st_chf,
    poisson_log_st_chf,
    vg_log_st_chf,
    nig_log_st_chf,
    heston_log_st_chf,
    cgmy_log_st_chf,
)
from fftoptionlib.moment_generating_funs import (
    black_scholes_log_st_mgf,
    merton_jump_log_st_mgf,
    kou_jump_log_st_mgf,
    poisson_log_st_mgf,
    nig_log_st_mgf,
    heston_log_st_mgf,
    cgmy_log_st_mgf,
    vg_log_st_mgf,
)


def chf_and_mgf_switch(chf, mgf, type):
    if type == 'chf':
        return chf
    elif type == 'mgf':
        return mgf
    else:
        raise ValueError('type only accept chf or mgf')


class LogSt(abc.ABC):
    def __init__(self, type=None):
        self._type = type

    def set_type(self, type):
        self._type = type
        return self

    @property
    def type(self):
        return self._type


class BlackScholes(LogSt):
    def __init__(self, sigma):
        super(BlackScholes, self).__init__()
        self.sigma = sigma

    def __call__(self, u, t, r, q, S0):
        return chf_and_mgf_switch(black_schole_log_st_chf, black_scholes_log_st_mgf, self.type)(u, t, r, q, S0, self.sigma)


class MertonJump(LogSt):
    def __init__(self, sigma, jump_rate, norm_m, norm_sig):
        self.sigma = sigma
        self.jump_rate = jump_rate
        self.norm_m = norm_m
        self.norm_sig = norm_sig
        super(MertonJump, self).__init__()

    def __call__(self, u, t, r, q, S0):
        return chf_and_mgf_switch(merton_jump_log_st_chf, merton_jump_log_st_mgf, self.type)(u, t, r, q, S0, self.sigma, self.jump_rate, self.norm_m, self.norm_sig)


class KouJump(LogSt):
    def __init__(self, sigma, jump_rate, exp_pos, exp_neg, prob_pos):
        self.sigma = sigma
        self.jump_rate = jump_rate
        self.exp_pos = exp_pos
        self.exp_neg = exp_neg
        self.prob_pos = prob_pos
        super(KouJump, self).__init__()

    def __call__(self, u, t, r, q, S0):
        return chf_and_mgf_switch(kou_jump_log_st_chf, kou_jump_log_st_mgf, self.type)(u, t, r, q, S0, self.sigma,
                                                                                       self.jump_rate,
                                                                                       self.exp_pos,
                                                                                       self.exp_neg,
                                                                                       self.prob_pos)


class Poisson(LogSt):
    def __init__(self, jump_rate):
        self.jump_rate = jump_rate
        super(Poisson, self).__init__()

    def __call__(self, u, t, r, q, S0):
        return chf_and_mgf_switch(poisson_log_st_chf, poisson_log_st_mgf, self.type)(u, t, r, q, S0, self.jump_rate)


class VarianceGamma(LogSt):
    def __init__(self, theta, v, sigma):
        self.theta = theta
        self.v = v
        self.sigma = sigma
        super(VarianceGamma, self).__init__()

    def __call__(self, u, t, r, q, S0):
        return chf_and_mgf_switch(vg_log_st_chf, vg_log_st_mgf, self.type)(u, t, r, q, S0, self.theta, self.v, self.sigma)


class NIG(LogSt):
    def __init__(self, a, b, delta):
        self.a = a
        self.b = b
        self.delta = delta
        super(NIG, self).__init__()

    def __call__(self, u, t, r, q, S0):
        return chf_and_mgf_switch(nig_log_st_chf, nig_log_st_mgf, self.type)(u, t, r, q, S0, self.a, self.b, self.delta)


class Heston(LogSt):
    def __init__(self, V0, theta, k, sigma, rho):
        self.V0 = V0
        self.theta = theta
        self.k = k
        self.sigma = sigma
        self.rho = rho
        super(Heston, self).__init__()

    def __call__(self, u, t, r, q, S0):
        return chf_and_mgf_switch(heston_log_st_chf, heston_log_st_mgf, self.type)(u, t, r, q, S0, self.V0,
                                                                                   self.theta,
                                                                                   self.k,
                                                                                   self.sigma,
                                                                                   self.rho)


class CGMY(LogSt):
    def __init__(self, c, g, m, y):
        self.c = c
        self.g = g
        self.m = m
        self.y = y
        super(CGMY, self).__init__()

    def __call__(self, u, t, r, q, S0):
        return chf_and_mgf_switch(cgmy_log_st_chf, cgmy_log_st_mgf, self.type)(u, t, r, q, S0, self.c,
                                                                               self.g,
                                                                               self.m,
                                                                               self.y)
