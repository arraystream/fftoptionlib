import numpy as np

from fftoptionlib.cosine_pricer import (
    cosin_vanilla_call,
    interval_a_and_b,
)
from fftoptionlib.fourier_pricer import (
    carr_madan_fft_call_pricer,
    carr_madan_fraction_fft_call_pricer,
)
from fftoptionlib.helper import spline_fitting
from fftoptionlib.moment_generating_funs import (
    general_log_moneyness_mgf,
    cumulants_from_mgf,
)


class FFTEngine(object):
    def __init__(self, N, d_u, alpha, spline_order):
        self.N = N
        self.d_u = d_u
        self.alpha = alpha
        self.spline_order = spline_order

    def __call__(self, strike, t, r, q, S0, chf_ln_st):
        sim_strikes, call_prices = carr_madan_fft_call_pricer(self.N, self.d_u, self.alpha, r, t, S0, q, chf_ln_st.set_type('chf'))
        ffn_prices = spline_fitting(sim_strikes, call_prices, self.spline_order)(strike)
        return ffn_prices


class FractionFFTEngine(object):
    def __init__(self, N, d_u, d_k, alpha, spline_order):
        self.N = N
        self.d_u = d_u
        self.d_k = d_k
        self.alpha = alpha
        self.spline_order = spline_order

    def __call__(self, strike, t, r, q, S0, chf_ln_st):
        sim_strikes, call_prices = carr_madan_fraction_fft_call_pricer(self.N, self.d_u, self.d_k, self.alpha, r, t, S0, q, chf_ln_st.set_type('chf'))
        ffn_prices = spline_fitting(sim_strikes, call_prices, self.spline_order)(strike)
        return ffn_prices


class CosineEngine(object):
    def __init__(self, N, L):
        self.N = N
        self.L = L

    def calc_integral_interval(self, strike, L, t, r, q, S0, chf_ln_st):
        c1, c2, c4 = cumulants_from_mgf(general_log_moneyness_mgf,
                                        strike,
                                        chf_ln_st.set_type('mgf'),
                                        t=t,
                                        r=r,
                                        q=q,
                                        S0=S0)
        a, b = interval_a_and_b(c1, c2, c4, L)
        return a, b

    def _single_cal_price(self, strike, t, r, q, S0, chf_ln_st):
        a, b = self.calc_integral_interval(strike, self.L, t, r, q, S0, chf_ln_st)
        call_price = cosin_vanilla_call(
            self.N,
            strike=strike,
            intv_a=a,
            intv_b=b,
            t=t,
            r=r,
            q=q,
            S0=S0,
            chf_ln_st=chf_ln_st.set_type('chf'))
        return call_price

    def __call__(self, strike, t, r, q, S0, chf_ln_st):
        call_prices = np.array([self._single_cal_price(one_strike, t, r, q, S0, chf_ln_st) for one_strike in strike])
        return call_prices
