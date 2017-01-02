import sys

import numpy

from recordlinkage.types import is_list_like, is_numpy_like


# Errors and Exception handlers
class IndexError(Exception):
    """ Error class for errors related to indexing. """
    pass

# Checks and conversions


def listify(x):
    """ make a list of the argument """

    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


def unique(x):
    """ return the unique values in a list (return_type list)"""

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

    :Example:

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


def _check_jellyfish(raise_error=True):
    """

    Check if the jellyfish is imported. If it is imported, return True. If not
    succesfully imported, raise if raise_error == True and return false if
    not.

    """
    if 'jellyfish' in sys.modules.keys():
        return True
    else:
        if raise_error:
            raise ImportError(
                "Install the module 'jellyfish' to use the following " +
                "string metrics: 'jaro', 'jarowinkler', 'levenshtein'" +
                " and 'damerau_levenshtein'."
            )
        else:
            return False


def max_number_of_pairs(*args):
    """ Compute the maximum number of pairs """

    if not args:
        raise ValueError('expected at least one dataframe')

    if len(args) == 1:
        return len(args[0]) * (len(args[0]) - 1) / 2
    else:
        return numpy.prod([len(arg) for arg in args])
