
import pandas

# Errors and Exception handlers


class IndexError(Exception):
    """ Error class for errors related to indexing. """
    pass

# Checks and conversions


def _label_or_column(label_or_column, dataframe):
    """

    This internal function to check if the argument is a column label or a
    pandas.Series or pandas.DataFrame. If the argument is a Series or
    DataFrame, nothing is done.

    """
    try:
        return dataframe[label_or_column]
    except Exception:

        if isinstance(label_or_column, (pandas.Series, pandas.DataFrame)):
            return label_or_column
        else:
            raise ValueError("The label or column has to be a valid label " +
                             "or pandas.Series or pandas.DataFrame. ")


def _resample(frame, index, level_i):
    """
    Resample a pandas.Series or pandas.DataFrame with one level of a
    MultiIndex.
    """

    data = frame.ix[index.get_level_values(level_i)]
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
