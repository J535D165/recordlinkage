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


def split_index(index, chunks):
    """Function to split pandas.Index and pandas.MultiIndex objects.

    Split pandas.Index and pandas.MultiIndex objects into chunks.
    numpy.array_split returns incorrect results for MultiIndex
    objects. This function is based on numpy.split_array.

    Parameters
    ----------
    index : pandas.Index, pandas.MultiIndex
        A pandas.Index or pandas.MultiIndex to split into chunks.
    chunks : int
        The number of parts to split the index into.

    Returns
    -------
    list
        A list with chunked pandas.Index or pandas.MultiIndex objects.

    """

    Ntotal = index.shape[0]
    Nsections = int(chunks)
    if Nsections <= 0:
        raise ValueError('number sections must be larger than 0.')
    Neach_section, extras = divmod(Ntotal, Nsections)
    section_sizes = ([0] + extras * [Neach_section + 1] +
                     (Nsections - extras) * [Neach_section])
    div_points = numpy.array(section_sizes).cumsum()

    sub_ind = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_ind.append(index[st:end])

    return sub_ind
