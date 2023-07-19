import warnings
from functools import wraps

import numpy
import pandas

import recordlinkage.config as cf


# Errors and Exception handlers
class IndexError(Exception):
    """Error class for errors related to indexing."""

    pass


class LearningError(Exception):
    """Learning error"""


class DeprecationHelper:
    """Deprecation helper for classes and functions.

    Based on https://stackoverflow.com/a/9008509/8727928
    """

    def __init__(self, new_target, msg=None):
        self.new_target = new_target
        self.msg = msg

    def _warn(self):
        from warnings import warn

        if self.msg is None:
            msg = "This class will get deprecated."
        else:
            msg = self.msg

        warn(msg, DeprecationWarning, stacklevel=1)

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        self._warn()
        return getattr(self.new_target, attr)


def return_type_deprecator(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        return_type = kwargs.pop("return_type", None)
        if return_type is not None:
            warnings.warn(
                "The argument 'return_type' is deprecated in the next "
                "version. Use recordlinkage.set_option('classification."
                "return_type', '{}') instead.".format(return_type),
                DeprecationWarning,
                stacklevel=2,
            )
            with cf.option_context("classification.return_type", return_type):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return func_wrapper


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


def listify(x, none_value=[]):
    """Make a list of the argument if it is not a list."""

    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    elif x is None:
        return none_value
    else:
        return [x]


def unique(x):
    """Convert a list in a unique list."""

    return list(set(x))


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def multi_index_to_frame(index):
    """
    Replicates MultiIndex.to_frame, which was introduced in pandas 0.21,
    for the sake of backwards compatibility.
    """
    return pandas.DataFrame(index.tolist(), index=index, columns=index.names)


def index_split(index, chunks):
    """Function to split pandas.Index and pandas.MultiIndex objects.

    Split :class:`pandas.Index` and :class:`pandas.MultiIndex` objects
    into chunks. This function is based on :func:`numpy.array_split`.

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
        raise ValueError("number sections must be larger than 0.")
    Neach_section, extras = divmod(Ntotal, Nsections)
    section_sizes = (
        [0] + extras * [Neach_section + 1] + (Nsections - extras) * [Neach_section]
    )
    div_points = numpy.array(section_sizes).cumsum()

    sub_ind = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_ind.append(index[st:end])

    return sub_ind


def split_index(*args, **kwargs):
    warnings.warn(
        "Function will be removed in the future. Use index_split.", DeprecationWarning
    )

    return index_split(*args, **kwargs)


def frame_indexing(frame, multi_index, level_i, indexing_type="label"):
    """Index dataframe based on one level of MultiIndex.

    Arguments
    ---------
    frame : pandas.DataFrame
        The datafrme to select records from.
    multi_index : pandas.MultiIndex
        A pandas multiindex were one fo the levels is used to sample the
        dataframe with.
    level_i : int, str
        The level of the multiIndex to index on.
    indexing_type : str
        The type of indexing. The value can be 'label' or 'position'.
        Default 'label'.

    """

    if indexing_type == "label":
        data = frame.loc[multi_index.get_level_values(level_i)]
        data.index = multi_index
    elif indexing_type == "position":
        data = frame.iloc[multi_index.get_level_values(level_i)]
        data.index = multi_index
    else:
        raise ValueError("indexing_type needs to be 'label' or 'position'")

    return data


def fillna(series_or_arr, missing_value=0.0):
    """Fill missing values in pandas objects and numpy arrays.

    Arguments
    ---------
    series_or_arr : pandas.Series, numpy.ndarray
        The numpy array or pandas series for which the missing values
        need to be replaced.
    missing_value : float, int, str
        The value to replace the missing value with. Default 0.0.

    Returns
    -------
    pandas.Series, numpy.ndarray
        The numpy array or pandas series with the missing values
        filled.
    """

    if pandas.notnull(missing_value):
        if isinstance(series_or_arr, (numpy.ndarray)):
            series_or_arr[numpy.isnan(series_or_arr)] = missing_value
        else:
            series_or_arr.fillna(missing_value, inplace=True)

    return series_or_arr
