import numpy as np

from fftoptionlib.characteristic_funs import general_log_moneyness_chf


def chi(n, a, b, c, d):
    x = n * np.pi / (b - a)
    dm = (d - a) * x
    cm = (c - a) * x
    return (np.cos(dm) * np.exp(d) - np.cos(cm) * np.exp(c) + x * (
        np.sin(dm) * np.exp(d) - np.sin(cm) * np.exp(c))) / (1.0 + x * x)


def phi(n, a, b, c, d):
    x = n * np.pi / (b - a)
    dm = (d - a) * x
    cm = (c - a) * x
    if isinstance(n, np.ndarray):
        res = np.ones_like(n) * (d - c)
        nonzero_bool = n != 0
        res[nonzero_bool] = (np.sin(dm[nonzero_bool]) - np.sin(cm[nonzero_bool])) / x[nonzero_bool]
        return res
    else:
        if n == 0:
            return d - c
        else:
            return (np.sin(dm) - np.sin(cm)) / x


def a_n(n, intv_a, intv_b, strike, chf, **kwargs):
    xi = n * np.pi / (intv_b - intv_a)
    chf_res = general_log_moneyness_chf(xi, strike, chf, **kwargs) * np.exp(-xi * intv_a * 1j)
    return 2. * chf_res.real / (intv_b - intv_a)


def v_call(K, n, a, b):
    return 2 / (b - a) * K * (chi(n, a, b, 0., b) - phi(n, a, b, 0., b))


def v_put(K, n, a, b):
    return 2 / (b - a) * K * (phi(n, a, b, a, 0.) - chi(n, a, b, a, 0.))


def cosin_vanilla_call(N, strike, intv_a, intv_b, t, r, q, S0, chf_ln_st):
    k_arr = np.arange(N)
    a_arr = a_n(k_arr, intv_a, intv_b, strike, chf_ln_st, t=t, r=r, q=q, S0=S0)
    v_arr = v_call(strike, k_arr, intv_a, intv_b)
    res = (intv_b - intv_a) / 2 * np.exp(-r * t) * (a_arr[0] * v_arr[0] / 2 + np.sum(a_arr[1:] * v_arr[1:]))
    return res


def interval_a_and_b(c1, c2, c4, L):
    c2 = np.abs(c2)
    c4 = np.abs(c4)
    a = c1 - L * np.sqrt(c2 + np.sqrt(c4))
    b = c1 + L * np.sqrt(c2 + np.sqrt(c4))
    return a, b
