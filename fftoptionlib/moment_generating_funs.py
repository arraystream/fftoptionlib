import autograd.numpy as np
from scipy.special import gamma
from autograd import grad


def black_scholes_log_st_mgf(u, t, r, q, S0, sigma):
    chf_xt = diffusion_mgf
    return general_ln_st_mgf(u, t, r, q, S0, chf_xt, sigma=sigma)


def merton_jump_log_st_mgf(u, t, r, q, S0, sigma, jump_rate, norm_m, norm_sig):
    chf_xt = diffusion_with_cpp_normal_mgf
    return general_ln_st_mgf(u, t, r, q, S0, chf_xt, sigma=sigma, jump_rate=jump_rate, norm_m=norm_m, norm_sig=norm_sig)


def kou_jump_log_st_mgf(u, t, r, q, S0, sigma, jump_rate, exp_pos, exp_neg, prob_pos):
    mgf_xt = diffusion_with_cpp_double_exponential_mgf
    return general_ln_st_mgf(u, t, r, q, S0, mgf_xt, sigma=sigma, jump_rate=jump_rate, exp_pos=exp_pos, exp_neg=exp_neg, prob_pos=prob_pos)


def poisson_log_st_mgf(u, t, r, q, S0, jump_rate):
    mgf_xt = poisson_mgf
    return general_ln_st_mgf(u, t, r, q, S0, mgf_xt, jump_rate=jump_rate)


def vg_log_st_mgf(u, t, r, q, S0, theta, v, sigma):
    mgf_xt = vg_mgf
    return general_ln_st_mgf(u, t, r, q, S0, mgf_xt, theta=theta, v=v, sigma=sigma)


def nig_log_st_mgf(u, t, r, q, S0, a, b, delta):
    mgf_xt = nig_mgf
    return general_ln_st_mgf(u, t, r, q, S0, mgf_xt, a=a, b=b, delta=delta)


def heston_log_st_mgf(u, t, r, q, S0, V0, theta, k, sigma, rho):
    dt = np.sqrt((sigma ** 2) * (u - u ** 2) + (k - rho * sigma * u) ** 2)
    beta = k - u * rho * sigma
    g = (beta - dt) / (beta + dt)
    D_t = (beta - dt) / (sigma ** 2) * ((1 - np.exp(-dt * t)) / (1 - g * np.exp(-dt * t)))
    C_t = u * (r - q) * t + k * theta / (sigma ** 2) * (
        (beta - dt) * t - 2 * np.log((1 - g * np.exp(-dt * t)) / (1 - g)))
    return np.exp(C_t + D_t * V0 + u * np.log(S0))


def cgmy_log_st_mgf(u, t, r, q, S0, c, g, m, y):
    mgf_xt = cgmy_mgf
    return general_ln_st_mgf(u, t, r, q, S0, mgf_xt, c=c, g=g, m=m, y=y)


def general_ln_st_mgf(u, t, r, q, S0, mgf_xt, *args, **kwargs):
    martingale_adjust = -(1 / t) * np.log(mgf_xt(1, t, *args, **kwargs))
    normal_term = (np.log(S0) + (r - q + martingale_adjust) * t) * u
    ln_st_mgf = np.exp(normal_term) * mgf_xt(u, t, *args, **kwargs)
    return ln_st_mgf


def norm_mgf(u, norm_mean, norm_sig):
    return np.exp(norm_mean * u + 0.5 * (u ** 2) * (norm_sig ** 2))


def double_exponential_mgf(u, exp_pos, exp_neg, prob_pos):
    return -((prob_pos * u * exp_pos - u * exp_neg + prob_pos * u * exp_neg) + exp_pos * exp_neg) / (
        (u - exp_pos) * (u + exp_neg))


def poisson_mgf(u, t, jump_rate):
    return np.exp(t * jump_rate * (np.exp(u) - 1))


def diffusion_mgf(u, t, sigma):
    return np.exp(0.5 * t * (u ** 2) * (sigma ** 2))


def vg_cgm_mgf(u, t, c, g, m):
    return ((g * m) / (g * m + (m - g) * u - u ** 2)) ** (c * t)


def vg_mgf(u, t, theta, v, sigma):
    return (1 - u * theta * v + 0.5 * (sigma ** 2) * (-u ** 2) * v) ** (-t / v)


def cgmy_mgf(u, t, c, g, m, y):
    return np.exp(c * t * gamma(-y) * ((m - u) ** y - m ** y + (g + u) ** y - g ** y))


def nig_mgf(u, t, a, b, delta):
    sqa = np.sqrt(a ** 2 - (b + u) ** 2)
    sqb = np.sqrt(a ** 2 - b ** 2)
    return np.exp(-delta * t * (sqa - sqb))


def parameter_to_cgm(theta, v, sigma):
    c = 1.0 / v
    a = np.sqrt(0.25 * (theta ** 2) * (v ** 2) + 0.5 * (sigma ** 2) * v)
    b = 0.5 * theta * v
    g = 1 / (a - b)
    m = 1 / (a + b)
    return c, g, m


def cpp_normal_mgf(u, t, jump_rate, norm_m, norm_sig):
    return np.exp(t * jump_rate * (norm_mgf(u, norm_m, norm_sig) - 1))


def cpp_double_exponential_mgf(u, t, jump_rate, exp_pos, exp_neg, prob_pos):
    return np.exp(t * jump_rate * (double_exponential_mgf(u, exp_pos, exp_neg, prob_pos) - 1))


def diffusion_with_cpp_normal_mgf(u, t, sigma, jump_rate, norm_m, norm_sig):
    return diffusion_mgf(u, t, sigma) * cpp_normal_mgf(u, t, jump_rate, norm_m, norm_sig)


def diffusion_with_cpp_double_exponential_mgf(u, t, sigma, jump_rate, exp_pos, exp_neg, prob_pos):
    return diffusion_mgf(u, t, sigma) * cpp_double_exponential_mgf(u, t, jump_rate, exp_pos, exp_neg, prob_pos)


def general_log_moneyness_mgf(u, strike, mgf, **kwargs):
    return np.exp(-u * np.log(strike)) * mgf(u, **kwargs)


def cumulant_generating_fun(u, mgf, *args, **kwargs):
    return np.log(mgf(u, *args, **kwargs))


def cumulants_from_mgf(mgf, *args, **kwargs):
    def dcgf(val):
        return cumulant_generating_fun(val, mgf, *args, **kwargs)

    d_cgf_1 = grad(dcgf)
    d_cgf_2 = grad(d_cgf_1)
    d_cgf_3 = grad(d_cgf_2)
    d_cgf_4 = grad(d_cgf_3)
    return d_cgf_1(0.0), d_cgf_2(0.0), d_cgf_4(0.0)
