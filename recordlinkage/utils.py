import pandas
import numpy


# Errors and Exception handlers
class IndexError(Exception):
    """ Error class for errors related to indexing. """
    pass


class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.
    Based on numpy's VisibleDeprecationWarning.
    """
    pass

# Checks and conversions


def is_label_dataframe(label, df):
    """check column label existance"""

    setdiff = set(label) - set(df.columns.tolist())

    if len(setdiff) == 0:
        return True
    else:
        return False


def get_length(x):
    """Return int or len(x)"""

    try:
        return int(x)
    except Exception:
        return len(x)


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


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def max_number_of_pairs(*args):
    """Compute the maximum number of pairs."""

    if not args:
        raise ValueError('expected at least one dataframe')

    if len(args) == 1:
        return len(args[0]) * (len(args[0]) - 1) / 2
    else:
        return numpy.prod([len(arg) for arg in args])


def multi_index_to_frame(index):
    """
    Replicates MultiIndex.to_frame, which was introduced in pandas 0.21,
    for the sake of backwards compatibility.
    """
    return pandas.DataFrame(index.tolist(), index=index, columns=index.names)
