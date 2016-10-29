import pandas
import numpy as np


# Numerical comparison algorithms

def _step_sim(d, offset=0, origin=0):
    # scale is not an argument

    if offset < 0:
        raise ValueError("The offset must be positive.")

    expr = 'abs(d - origin) <= offset'

    return pandas.eval(expr).astype(np.int64)


def _linear_sim(d, scale, offset=0, origin=0):

    if offset < 0:
        raise ValueError("The offset must be positive.")

    if scale <= 0:
        raise ValueError("The scale must be larger than 0. ")

    d = (abs(d - origin)).clip(offset, offset + 2 * scale)

    expr = '1 - (d-offset)/(2*scale)'

    return pandas.eval(expr)


def _squared_sim(d, scale, offset=0, origin=0):

    if offset < 0:
        raise ValueError("The offset must be positive.")

    if scale <= 0:
        raise ValueError("The scale must be larger than 0. ")

    d = (abs(d - origin)).clip(offset, offset + np.sqrt(2) * scale)
    # solve y=1-ad^2 given y(d=scale)=0.5
    # 1-y = ad^2
    # a = (1-y)/d^2

    # fill y=0.5 and d = scale
    # a = (1-0.5)/scale^2
    # a = 1/(2*scale^2)
    # y = 1 - 1/2*(d/scale)^2
    # d = sqrt(2)*scale is the point where similarity is zero.

    expr = '1 - 1/2*exp(2*log((d-offset)/scale))'

    return pandas.eval(expr)


def _exp_sim(d, scale, offset=0, origin=0):

    if offset < 0:
        raise ValueError("The offset must be positive.")

    if scale <= 0:
        raise ValueError("The scale must be larger than 0. ")

    d = (abs(d - origin)).clip(offset, None)

    # solve y=exp(-x*a) if 1/2 = exp(-x/scale)
    expr = '2**(-(d-offset)/scale)'

    return pandas.eval(expr)


def _gauss_sim(d, scale, offset=0, origin=0):

    if offset < 0:
        raise ValueError("The offset must be positive.")

    if scale <= 0:
        raise ValueError("The scale must be larger than 0. ")

    d = (abs(d - origin)).clip(offset, None)

    # solve y=exp(-x^2*a) if 1/2 = exp(-x^2/scale^2)
    expr = '2**(-((d-offset)/scale)**2)'

    return pandas.eval(expr)
