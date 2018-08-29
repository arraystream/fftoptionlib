import numpy as np
from scipy.fftpack import fft, ifft


def carr_madan_fft_call_pricer(N, d_u, alpha, r, t, S0, q, chf_ln_st):
    d_k = 2 * np.pi / (N * d_u)
    beta = np.log(S0) - d_k * N / 2
    u_arr = np.arange(N) * d_u
    k_arr = beta + np.arange(N) * d_k
    delta_arr = np.zeros(N)
    delta_arr[0] = 1
    w_arr = d_u / 3 * (3 + (-1) ** (np.arange(N) + 1) - delta_arr)
    call_chf = (np.exp(-r * t) / ((alpha + 1j * u_arr) * (alpha + 1j * u_arr + 1))) * chf_ln_st(
        u_arr - (alpha + 1) * 1j,
        t, r, q=q, S0=S0)
    x_arr = np.exp(-1j * beta * u_arr) * call_chf * w_arr
    fft_prices = (fft(x_arr))
    call_prices = (np.exp(-alpha * k_arr) / np.pi) * fft_prices.real
    return np.exp(k_arr), call_prices


def carr_madan_fraction_fft_call_pricer(N, d_u, d_k, alpha, r, t, S0, q, chf_ln_st):
    rou = (d_u * d_k) / (2 * np.pi)
    beta = np.log(S0) - d_k * N / 2
    u_arr = np.arange(N) * d_u
    k_arr = beta + np.arange(N) * d_k
    delta_arr = np.zeros(N)
    delta_arr[0] = 1
    w_arr = d_u / 3 * (3 + (-1) ** (np.arange(N) + 1) - delta_arr)
    call_chf = (np.exp(-r * t) / ((alpha + 1j * u_arr) * (alpha + 1j * u_arr + 1))) * chf_ln_st(
        u_arr - (alpha + 1) * 1j,
        t, r, q=q, S0=S0)
    x_arr = np.exp(-1j * beta * u_arr) * call_chf * w_arr
    y_arr = np.zeros(2 * N) * 0j
    y_arr[:N] = np.exp(-1j * np.pi * rou * np.arange(N) ** 2) * x_arr
    z_arr = np.zeros(2 * N) * 0j
    z_arr[:N] = np.exp(1j * np.pi * rou * np.arange(N) ** 2)
    z_arr[N:] = np.exp(1j * np.pi * rou * np.arange(N - 1, -1, -1) ** 2)
    ffty = (fft(y_arr))
    fftz = (fft(z_arr))
    fftx = ffty * fftz
    fftpsi = ifft(fftx)
    fft_prices = np.exp(-1j * np.pi * (np.arange(N) ** 2) * rou) * fftpsi[:N]
    call_prices = (np.exp(-alpha * k_arr) / np.pi) * fft_prices.real
    return np.exp(k_arr), call_prices
