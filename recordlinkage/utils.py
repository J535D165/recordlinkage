import sys

import numpy

from recordlinkage.types import is_list_like, is_numpy_like


# Errors and Exception handlers
class IndexError(Exception):
    """ Error class for errors related to indexing. """
    pass

# Checks and conversions


def listify(x):
    """Make a list of the argument if it is not a list."""

    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


def unique(x):
    """Convert a list in a unique list."""

    return list(set(x))


def _resample(frame, index, level_i):
    """
    Resample a pandas.Series or pandas.DataFrame with one level of a
    MultiIndex.
    """

    data = frame.loc[index.get_level_values(level_i)]
    data.index = index

    return data


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def split_or_pass(v):
    """
    Make a tuple of a single value or return the tuple.

    Example
    -------
    >>> v1, v2 = split_or_tuple(3)
    v1 is 3 and v2 and 3
    >>> v1, v2 = split_or_tuple((3,4))
    v1 is 3 and v2 and 4

    """

    if isinstance(v, (tuple, list)):

        if len(v) != 2:
            raise ValueError(
                'The number of elements in the list of tuple has to be 2. ')

        return tuple(v)

    else:
        return (v, v)


def max_number_of_pairs(*args):
    """Compute the maximum number of pairs."""

    if not args:
        raise ValueError('expected at least one dataframe')

    if len(args) == 1:
        return len(args[0]) * (len(args[0]) - 1) / 2
    else:
        return numpy.prod([len(arg) for arg in args])
