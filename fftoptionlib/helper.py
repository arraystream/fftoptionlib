import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def spline_fitting(x_data, y_data, order):
    return InterpolatedUnivariateSpline(x_data, y_data, k=order)


def to_array_atleast_1d(*args):
    return [np.atleast_1d(item) for item in args]


def to_array_with_same_dimension(*args):
    new_args = to_array_atleast_1d(*args)
    reference_data = new_args[0]
    ref_len = len(reference_data)
    res = [reference_data]
    for items in new_args[1:]:
        items_len = len(items)
        if items_len != ref_len:
            if items_len == 1:
                res.append(np.repeat(items, ref_len))
            else:
                raise ValueError('arguments either has length 1 or same as reference data')
        else:
            res.append(items)
    return res


def call_to_put(call_prices, put_call_arr, strike_arr, discount_bond_price, forward_price, put_label):
    call_prices, put_call_arr, strike_arr = to_array_with_same_dimension(call_prices, put_call_arr, strike_arr)
    put_bool = put_call_arr == put_label
    call_prices[put_bool] = call_prices[put_bool] - discount_bond_price * (forward_price - strike_arr[put_bool])
    return call_prices
