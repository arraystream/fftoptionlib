import numpy as np
from scipy.special import gamma


def black_schole_log_st_chf(u, t, r, q, S0, sigma):
    chf_xt = diffusion_chf
    return general_ln_st_chf(u, t, r, q, S0, chf_xt, sigma=sigma)


def merton_jump_log_st_chf(u, t, r, q, S0, sigma, jump_rate, norm_m, norm_sig):
    chf_xt = diffusion_with_cpp_normal_chf
    return general_ln_st_chf(u, t, r, q, S0, chf_xt, sigma=sigma, jump_rate=jump_rate, norm_m=norm_m, norm_sig=norm_sig)


def kou_jump_log_st_chf(u, t, r, q, S0, sigma, jump_rate, exp_pos, exp_neg, prob_pos):
    chf_xt = diffusion_with_cpp_double_exponential_chf
    return general_ln_st_chf(u, t, r, q, S0, chf_xt, sigma=sigma, jump_rate=jump_rate, exp_pos=exp_pos, exp_neg=exp_neg, prob_pos=prob_pos)


def poisson_log_st_chf(u, t, r, q, S0, jump_rate):
    chf_xt = poisson_chf
    return general_ln_st_chf(u, t, r, q, S0, chf_xt, jump_rate=jump_rate)


def vg_log_st_chf(u, t, r, q, S0, theta, v, sigma):
    chf_xt = vg_chf
    return general_ln_st_chf(u, t, r, q, S0, chf_xt, theta=theta, v=v, sigma=sigma)


def nig_log_st_chf(u, t, r, q, S0, a, b, delta):
    chf_xt = nig_chf
    return general_ln_st_chf(u, t, r, q, S0, chf_xt, a=a, b=b, delta=delta)


def heston_log_st_chf(u, t, r, q, S0, V0, theta, k, sigma, rho):
    dt = np.sqrt((sigma ** 2) * (1j * u + u ** 2) + (k - 1j * rho * sigma * u) ** 2)
    beta = k - 1j * u * rho * sigma
    g = (beta - dt) / (beta + dt)
    D_t = (beta - dt) / (sigma ** 2) * ((1 - np.exp(-dt * t)) / (1 - g * np.exp(-dt * t)))
    C_t = 1j * u * (r - q) * t + k * theta / (sigma ** 2) * (
        (beta - dt) * t - 2 * np.log((1 - g * np.exp(-dt * t)) / (1 - g)))
    return np.exp(C_t + D_t * V0 + 1j * u * np.log(S0))


def cgmy_log_st_chf(u, t, r, q, S0, c, g, m, y):
    chf_xt = cgmy_chf
    return general_ln_st_chf(u, t, r, q, S0, chf_xt, c=c, g=g, m=m, y=y)


def general_ln_st_chf(u, t, r, q, S0, chf_xt, *args, **kwargs):
    martingale_adjust = -(1 / t) * np.log(chf_xt(-1j, t, *args, **kwargs))
    normal_term = 1j * (np.log(S0) + (r - q + martingale_adjust) * t) * u
    ln_st_chf = np.exp(normal_term) * chf_xt(u, t, *args, **kwargs)
    return ln_st_chf


def vasicek_int_rt_chf(u, t, k, theta, sigma, r0):
    mean_int_rt = theta * t + (r0 - theta) * (1 - np.exp(-k * t)) / k
    var_int_rt = (sigma ** 2 / (2 * k ** 3)) * ((1 - np.exp(-k * t)) ** 2) + (sigma ** 2 / (k ** 2)) * (
        t - (1 - np.exp(-k * t)) / k)
    return np.exp(1j * mean_int_rt * u - 0.5 * var_int_rt * (u ** 2))


def cir_int_rt_chf(u, t, k, theta, sigma, r0):
    r = np.sqrt(k ** 2 - 1j * u * 2 * sigma ** 2)
    cosh_fun = np.cosh(r * t / 2)
    sinh_fun = np.sinh(r * t / 2)
    coth_fun = cosh_fun / sinh_fun
    a_t_v = np.exp(t * theta * (k ** 2) / (sigma ** 2)) / (cosh_fun + (k / r) * sinh_fun) ** (
        2 * k * theta / (sigma ** 2))
    b_t_v = 2 * 1j * u / (k + r * coth_fun)
    return a_t_v * np.exp(b_t_v * r0)


def general_log_moneyness_chf(u, strike, chf, *args, **kwargs):
    return np.exp(-1j * u * np.log(strike)) * chf(u, *args, **kwargs)


def norm_chf(u, norm_mean, norm_sig):
    return np.exp(1j * norm_mean * u - 0.5 * (u ** 2) * (norm_sig ** 2))


def double_exponential_chf(u, exp_pos, exp_neg, prob_pos):
    return (1j * (prob_pos * u * exp_pos - u * exp_neg + prob_pos * u * exp_neg) + exp_pos * exp_neg) / (
        (u + 1j * exp_pos) * (u - 1j * exp_neg))


def poisson_chf(u, t, jump_rate):
    return np.exp(t * jump_rate * (np.exp(1j * u) - 1))


def diffusion_chf(u, t, sigma):
    return np.exp(- 0.5 * t * (u ** 2) * (sigma ** 2))


def vg_cgm_chf(u, t, c, g, m):
    return ((g * m) / (g * m + (m - g) * u * 1j + u ** 2)) ** (c * t)


def vg_chf(u, t, theta, v, sigma):
    return (1 - 1j * u * theta * v + 0.5 * (sigma ** 2) * (u ** 2) * v) ** (-t / v)


def cgmy_chf(u, t, c, g, m, y):
    return np.exp(c * t * gamma(-y) * ((m - 1j * u) ** y - m ** y + (g + 1j * u) ** y - g ** y))


def nig_chf(u, t, a, b, delta):
    sqa = np.sqrt(a ** 2 - (b + 1j * u) ** 2)
    sqb = np.sqrt(a ** 2 - b ** 2)
    return np.exp(-delta * t * (sqa - sqb))


def parameter_to_cgm(theta, v, sigma):
    c = 1.0 / v
    a = np.sqrt(0.25 * (theta ** 2) * (v ** 2) + 0.5 * (sigma ** 2) * v)
    b = 0.5 * theta * v
    g = 1 / (a - b)
    m = 1 / (a + b)
    return c, g, m


def cpp_normal_chf(u, t, jump_rate, norm_m, norm_sig):
    return np.exp(t * jump_rate * (norm_chf(u, norm_m, norm_sig) - 1))


def cpp_double_exponential_chf(u, t, jump_rate, exp_pos, exp_neg, prob_pos):
    return np.exp(t * jump_rate * (double_exponential_chf(u, exp_pos, exp_neg, prob_pos) - 1))


def diffusion_with_cpp_normal_chf(u, t, sigma, jump_rate, norm_m, norm_sig):
    return diffusion_chf(u, t, sigma) * cpp_normal_chf(u, t, jump_rate, norm_m, norm_sig)


def diffusion_with_cpp_double_exponential_chf(u, t, sigma, jump_rate, exp_pos, exp_neg, prob_pos):
    return diffusion_chf(u, t, sigma) * cpp_double_exponential_chf(u, t, jump_rate, exp_pos, exp_neg, prob_pos)
