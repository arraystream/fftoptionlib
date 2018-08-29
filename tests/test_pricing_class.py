import unittest

import numpy as np
import numpy.testing as npt

from fftoptionlib.engine_class import (
    FFTEngine,
    FractionFFTEngine,
    CosineEngine,
)
from fftoptionlib.option_class import BasicOption
from fftoptionlib.pricing_class import FourierPricer
from fftoptionlib.process_class import (
    BlackScholes,
    MertonJump,
    KouJump,
    NIG,
    CGMY,
    Poisson,
)


class TestPricingClass(unittest.TestCase):
    def setUp(self):
        underlying_price = 36
        maturity_date = '1999-05-17'
        risk_free_rate = 0.06
        evaluation_date = '1999-04-22'
        self.vanilla_option = BasicOption()
        (self.vanilla_option.set_underlying_close_price(underlying_price)
         .set_dividend(0.01)
         .set_maturity_date(maturity_date)
         .set_evaluation_date(evaluation_date)
         .set_zero_rate(risk_free_rate))

    def test_cal_price_fft(self):
        fft_pricer = FourierPricer(self.vanilla_option)
        strike_arr = np.array([5, 10, 30, 36, 50, 60, 100])
        put_call_arr = np.array(['call', 'call', 'put', 'call', 'call', 'put', 'call'])
        exp = np.array(
            [3.09958567e+01, 2.60163625e+01, 8.25753140e-05, 8.12953226e-01,
             8.97449491e-11, 2.37785797e+01, 2.19293560e-85, ]
        )

        volatility = 0.20
        N = 2 ** 15
        d_u = 0.01
        alpha = 1
        fft_pricer.set_log_st_process(BlackScholes(volatility))
        fft_pricer.set_pricing_engine(FFTEngine(N, d_u, alpha, spline_order=3))
        res = fft_pricer.calc_price(strike_arr, put_call_arr, put_label='put')
        npt.assert_array_almost_equal(res, exp, 6)

    def test_cal_price_fractional_fft(self):
        fft_pricer = FourierPricer(self.vanilla_option)
        strike_arr = np.array([5, 10, 30, 36, 50, 60, 100])
        put_call_arr = np.array(['call', 'call', 'put', 'call', 'call', 'put', 'call'])
        exp = np.array(
            [3.09958567e+01, 2.60163625e+01, 8.25753140e-05, 8.12953226e-01,
             8.97449491e-11, 2.37785797e+01, 2.19293560e-85, ]
        )
        volatility = 0.20
        N = 2 ** 14
        d_u = 0.01
        d_k = 0.01
        alpha = 1
        fft_pricer.set_log_st_process(BlackScholes(volatility))
        fft_pricer.set_pricing_engine(FractionFFTEngine(N, d_u, d_k, alpha, spline_order=3))
        res = fft_pricer.calc_price(strike_arr, put_call_arr, put_label='put')
        npt.assert_array_almost_equal(res, exp, 6)

    def test_cal_price_cosine(self):
        cosine_pricer = FourierPricer(self.vanilla_option)
        strike_arr = np.array([5, 10, 30, 36, 50, 60, 100])
        put_call_arr = np.array(['call', 'call', 'put', 'call', 'call', 'put', 'call'])
        exp = np.array([3.09958567e+01, 2.60163625e+01, 8.25753140e-05, 8.12953226e-01,
                        8.97449491e-11, 2.37785797e+01, 2.19293560e-85, ]
                       )

        volatility = 0.20
        N = 150

        cosine_pricer.set_log_st_process(BlackScholes(volatility))
        cosine_pricer.set_pricing_engine(CosineEngine(N, L=30))
        res = cosine_pricer.calc_price(strike_arr, put_call_arr, put_label='put')
        npt.assert_array_almost_equal(res, exp, 6)

    def test_cal_price_cosine_2(self):
        cosine_pricer = FourierPricer(self.vanilla_option)
        strike_arr = np.array([5, 10, 30, 36, 50, 60])
        put_call_arr = np.array(['call', 'call', 'put', 'call', 'call', 'put'])
        exp = np.array([3.09958568e+01, 2.60164423e+01, 9.56077953e-02, 8.81357807e-01,
                        1.41769466e-10, 2.37785797e+01]
                       )

        volatility = 0.20
        N = 2000

        cosine_pricer.set_log_st_process(MertonJump(sigma=volatility,
                                                    jump_rate=0.090913148257155449,
                                                    norm_m=-0.91157356544103341,
                                                    norm_sig=7.3383200797618833e-05))
        cosine_pricer.set_pricing_engine(CosineEngine(N, L=30))
        res = cosine_pricer.calc_price(strike_arr, put_call_arr, put_label='put')
        npt.assert_array_almost_equal(res, exp, 6)

    def test_cal_price_cosine_3(self):
        cosine_pricer = FourierPricer(self.vanilla_option)
        strike_arr = np.array([5, 10, 30, 36, 40, 60])
        put_call_arr = np.array(['call', 'call', 'put', 'call', 'call', 'put'])
        exp = np.array([3.09958567e+01, 2.60163625e+01, 1.71886506e-04, 8.75203272e-01,
                        3.55292239e-02, 2.37785797e+01]
                       )

        volatility = 0.20
        N = 2000

        cosine_pricer.set_log_st_process(KouJump(sigma=volatility,
                                                 jump_rate=23.339325557373201,
                                                 exp_pos=59.378410421004197,
                                                 exp_neg=-59.447921334340137,
                                                 prob_pos=-200.08018971817182))
        cosine_pricer.set_pricing_engine(CosineEngine(N, L=30))
        res = cosine_pricer.calc_price(strike_arr, put_call_arr, put_label='put')
        npt.assert_array_almost_equal(res, exp, 6)

    def test_cal_price_cosine_4(self):
        cosine_pricer = FourierPricer(self.vanilla_option)
        strike_arr = np.array([5, 10, 30, 36, 40, 60])
        put_call_arr = np.array(['call', 'call', 'put', 'call', 'call', 'put'])
        exp = np.array([30.99596331, 26.01719432, 0.18730742, 1.51294408, 0.78980451,
                        24.03726193]
                       )

        volatility = 0.20
        N = 2000

        cosine_pricer.set_log_st_process(NIG(a=2,
                                             b=0.3,
                                             delta=0.6))
        cosine_pricer.set_pricing_engine(CosineEngine(N, L=30))
        res = cosine_pricer.calc_price(strike_arr, put_call_arr, put_label='put')
        npt.assert_array_almost_equal(res, exp, 6)

    def test_cal_price_cosine_5(self):
        cosine_pricer = FourierPricer(self.vanilla_option)
        strike_arr = np.array([5, 10, 30, 36, 40, 60])
        put_call_arr = np.array(['call', 'call', 'put', 'call', 'call', 'put'])
        exp = np.array([22.48662613, 17.50712233, -8.50824394, -7.13267131, -7.22087064,
                        16.09965887]
                       )

        volatility = 0.20
        N = 2000

        cosine_pricer.set_log_st_process(Poisson(
            jump_rate=0.339325557373201,
        ))
        cosine_pricer.set_pricing_engine(CosineEngine(N, L=30))
        res = cosine_pricer.calc_price(strike_arr, put_call_arr, put_label='put')
        npt.assert_array_almost_equal(res, exp, 6)

    def test_cal_price_cosine_6(self):
        cosine_pricer = FourierPricer(self.vanilla_option)
        strike_arr = np.array([5, 10, 30, 36, 40, 60])
        put_call_arr = np.array(['call', 'call', 'put', 'call', 'call', 'put'])
        exp = np.array([30.9958673, 26.01656399, 0.15928293, 1.72971868, 0.56756891, 23.82357896])

        volatility = 0.20
        N = 2000

        cosine_pricer.set_log_st_process(CGMY(c=0.1,
                                              g=2,
                                              m=2,
                                              y=1.5))
        cosine_pricer.set_pricing_engine(CosineEngine(N, L=30))
        res = cosine_pricer.calc_price(strike_arr, put_call_arr, put_label='put')
        npt.assert_array_almost_equal(res, exp, 6)
