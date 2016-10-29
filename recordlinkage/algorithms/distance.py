import pandas
import numpy as np


# Numerical distance algorithms
def _1d_distance(s1, s2):

    return pandas.eval("s2-s1")


def _haversine_distance(lat1, lng1, lat2, lng2):

    # degrees to radians conversion
    to_rad = np.deg2rad(1)

    # numeric expression to use with numexpr package
    expr = '2*6371*arcsin(sqrt((sin((lat2*to_rad-lat1*to_rad)/2))**2+cos(lat1*to_rad)*cos(lat2*to_rad)*(sin((lng2*to_rad-lng1*to_rad)/2))**2))'

    return pandas.eval(expr)
