"""
basic inference routines

most functions taken from pandas (https://github.com/pandas-dev/pandas)
License BSD

"""

import collections
import re
from numbers import Number

import numpy
import pandas

string_and_binary_types = (str, bytes)


def is_number(obj):
    return isinstance(obj, (Number, numpy.number))


def is_string_like(obj):
    return isinstance(obj, str)


def _iterable_not_string(x):
    return isinstance(x, collections.Iterable) and not isinstance(x, str)


def is_iterator(obj):
    return hasattr(obj, "__next__")


def is_re(obj):
    return isinstance(obj, re._pattern_type)


def is_re_compilable(obj):
    try:
        re.compile(obj)
    except TypeError:
        return False
    else:
        return True


def is_list_like(arg):
    return hasattr(arg, "__iter__") and not isinstance(arg, string_and_binary_types)


def is_dict_like(arg):
    return hasattr(arg, "__getitem__") and hasattr(arg, "keys")


def is_named_tuple(arg):
    return isinstance(arg, tuple) and hasattr(arg, "_fields")


def is_hashable(arg):
    """Return True if hash(arg) will succeed, False otherwise.

    Some types will pass a test against collections.Hashable but fail when they
    are actually hashed with hash().

    Distinguish between these and other types by trying the call to hash() and
    seeing if they raise TypeError.

    Examples
    --------
    >>> a = ([],)
    >>> isinstance(a, collections.Hashable)
    True
    >>> is_hashable(a)
    False
    """

    # unfortunately, we can't use isinstance(arg, collections.Hashable), which
    # can be faster than calling hash, because numpy scalars on Python 3 fail
    # this test

    # reconsider this decision once this numpy bug is fixed:
    # https://github.com/numpy/numpy/issues/5562

    try:
        hash(arg)
    except TypeError:
        return False
    else:
        return True


def is_sequence(x):
    try:
        iter(x)
        len(x)  # it has a length
        return not isinstance(x, string_and_binary_types)
    except (TypeError, AttributeError):
        return False


def is_pandas_like(x):
    return isinstance(x, (pandas.Series, pandas.DataFrame))


def is_pandas_multiindex(x):
    return isinstance(x, (pandas.MultiIndex))


def is_pandas_2d_multiindex(x):
    return is_pandas_multiindex(x) and x.nlevels == 2


def is_numpy_like(x):
    return isinstance(x, (numpy.ndarray))
