from __future__ import division
from __future__ import unicode_literals

from functools import wraps, partial

import pandas
import numpy as np

from recordlinkage.base import BaseCompareFeature
from recordlinkage.algorithms.distance import _1d_distance
from recordlinkage.algorithms.distance import _haversine_distance
from recordlinkage.algorithms.numeric import _step_sim
from recordlinkage.algorithms.numeric import _linear_sim
from recordlinkage.algorithms.numeric import _squared_sim
from recordlinkage.algorithms.numeric import _exp_sim
from recordlinkage.algorithms.numeric import _gauss_sim
from recordlinkage.algorithms.string import jaro_similarity
from recordlinkage.algorithms.string import jarowinkler_similarity
from recordlinkage.algorithms.string import levenshtein_similarity
from recordlinkage.algorithms.string import damerau_levenshtein_similarity
from recordlinkage.algorithms.string import qgram_similarity
from recordlinkage.algorithms.string import cosine_similarity
from recordlinkage.algorithms.string import smith_waterman_similarity
from recordlinkage.algorithms.string import longest_common_substring_similarity
from recordlinkage.utils import DeprecationHelper


def fillna_decorator(missing_value=np.nan):

    def real_decorator(func):

        @wraps(func)
        def func_wrapper(*args, **kwargs):

            mv = kwargs.pop('missing_value', missing_value)

            result = func(*args, **kwargs)

            # fill missing values if missing_value is not a missing value like
            # NaN or None.
            if pandas.notnull(mv):
                if isinstance(result, (np.ndarray)):
                    result[np.isnan(result)] = mv
                else:
                    result.fillna(mv, inplace=True)

            return result

        return func_wrapper

    return real_decorator


def fillna(series_or_arr, missing_value=0.0):

    if pandas.notnull(missing_value):
        if isinstance(series_or_arr, (np.ndarray)):
            series_or_arr[np.isnan(series_or_arr)] = missing_value
        else:
            series_or_arr.fillna(missing_value, inplace=True)

    return series_or_arr


def _missing(*args):
    """ Missing values.

    Internal function to return the index of record pairs with missing values
    """

    return np.any(
        np.concatenate(
            [np.array(pandas.DataFrame(arg).isnull()) for arg in args],
            axis=1),
        axis=1)


class Exact(BaseCompareFeature):
    """Compare the record pairs exactly.

    This class is used to compare records in an exact way. The similarity
    is 1 in case of agreement and 0 otherwise.

    Parameters
    ----------

    labels_left : str or int
        Field name to compare in left DataFrame.
    labels_right : str or int
        Field name to compare in right DataFrame.
    agree_value : float, str, numpy.dtype
        The value when two records are identical. Default 1. If 'values'
        is passed, then the value of the record pair is passed.
    disagree_value : float, str, numpy.dtype
        The value when two records are not identical.
    missing_value : float, str, numpy.dtype
        The value for a comparison with a missing value. Default 0.

    """

    name = "exact"
    description = "Compare attributes of record pairs."

    def __init__(self, labels_left, labels_right, agree_value=1,
                 disagree_value=0, missing_value=0, label=None):
        super(Exact, self).__init__(labels_left, labels_right, label=label)

        self.agree_value = agree_value
        self.disagree_value = disagree_value
        self.missing_value = missing_value

    def _compute_vectorized(self, s1, s2):

        # Values or agree/disagree
        if self.agree_value == 'value':
            compare = s1.copy()
            compare[s1 != s2] = self.disagree_value

        else:
            compare = pandas.Series(self.disagree_value, index=s1.index)
            compare[s1 == s2] = self.agree_value

        # Only when disagree value is not identical with the missing value
        if self.disagree_value != self.missing_value:
            compare[(s1.isnull() | s2.isnull())] = self.missing_value

        return compare


class String(BaseCompareFeature):
    """Compute the (partial) similarity between strings values.

    This class is used to compare string values. The implemented algorithms
    are: 'jaro','jarowinkler', 'levenshtein', 'damerau_levenshtein', 'qgram'
    or 'cosine'. In case of agreement, the similarity is 1 and in case of
    complete disagreement it is 0. The Python Record Linkage Toolkit uses the
    `jellyfish` package for the Jaro, Jaro-Winkler, Levenshtein and Damerau-
    Levenshtein algorithms.

    Parameters
    ----------
    s1 : str or int
        The name or position of the column in the left DataFrame.
    s2 : str or int
        The name or position of the column in the right DataFrame.
    method : str, default 'levenshtein'
        An approximate string comparison method. Options are ['jaro',
        'jarowinkler', 'levenshtein', 'damerau_levenshtein', 'qgram',
        'cosine', 'smith_waterman', 'lcs']. Default: 'levenshtein'
    threshold : float, tuple of floats
        A threshold value. All approximate string comparisons higher or
        equal than this threshold are 1. Otherwise 0.
    missing_value : numpy.dtype
        The value for a comparison with a missing value. Default 0.
    """

    name = "string"
    description = "Compare string attributes of record pairs."

    def __init__(self, labels_left, labels_right, method=None,
                 threshold=None, missing_value=0.0, label=None):
        super(String, self).__init__(labels_left, labels_right, label=label)

        self.method = method
        self.threshold = threshold
        self.missing_value = missing_value

    def _compute_vectorized(self, s1, s2):

        if self.method == 'jaro':
            str_sim_alg = jaro_similarity
        elif self.method in ['jarowinkler', 'jaro_winkler', 'jw']:
            str_sim_alg = jarowinkler_similarity
        elif self.method == 'levenshtein':
            str_sim_alg = levenshtein_similarity
        elif self.method in ['dameraulevenshtein',
                             'damerau_levenshtein',
                             'dl']:
            str_sim_alg = damerau_levenshtein_similarity
        elif self.method in ['q_gram', 'qgram']:
            str_sim_alg = qgram_similarity
        elif self.method == 'cosine':
            str_sim_alg = cosine_similarity
        elif self.method in ['smith_waterman', 'smithwaterman', 'sw']:
            str_sim_alg = smith_waterman_similarity
        elif self.method in ['longest_common_substring', 'lcs']:
            str_sim_alg = longest_common_substring_similarity
        else:
            raise ValueError(
                "The algorithm '{}' is not known.".format(self.method)
            )

        c = str_sim_alg(s1, s2)
        c = fillna(c, self.missing_value)

        if self.threshold:
            return (c >= self.threshold).astype(np.float64)
        else:
            return c


class Numeric(BaseCompareFeature):
    """Compute the (partial) similarity between numeric values.

    This class is used to compare numeric values. The implemented algorithms
    are: 'step', 'linear', 'exp', 'gauss' or 'squared'. In case of agreement,
    the similarity is 1 and in case of complete disagreement it is 0. The
    implementation is similar with numeric comparing in ElasticSearch, a full-
    text search tool. The parameters are explained in the image below (source
    ElasticSearch, The Definitive Guide)

    .. image:: /images/elas_1705.png
        :width: 100%
        :target: https://www.elastic.co/guide/en/elasticsearch/guide/current/decay-functions.html
        :alt: Decay functions, like in ElasticSearch

    Parameters
    ----------
    s1 : str or int
        The name or position of the column in the left DataFrame.
    s2 : str or int
        The name or position of the column in the right DataFrame.
    method : float
        The metric used. Options 'step', 'linear', 'exp', 'gauss' or
        'squared'. Default 'linear'.
    offset : float
        The offset. See image above.
    scale : float
        The scale of the numeric comparison method. See the image above.
        This argument is not available for the 'step' algorithm.
    origin : float
        The shift of bias between the values. See image above.
    missing_value : numpy.dtype
        The value if one or both records have a missing value on the
        compared field. Default 0.

    Note
    ----
    Numeric comparing can be an efficient way to compare date/time
    variables. This can be done by comparing the timestamps.

    """

    name = "numeric"
    description = "Compare numeric attributes of record pairs."

    def __init__(self, labels_left, labels_right, method='linear',
                 offset=0.0, scale=1.0, origin=0.0, missing_value=0.0,
                 label=None):
        super(Numeric, self).__init__(labels_left, labels_right, label=label)

        self.method = method
        self.offset = offset
        self.scale = scale
        self.origin = origin
        self.missing_value = missing_value

    def _compute_vectorized(self, s1, s2):

        d = _1d_distance(s1, s2)

        if self.method == 'step':
            num_sim_alg = partial(_step_sim, d,
                                  self.offset,
                                  self.origin)
        elif self.method in ['linear', 'lin']:
            num_sim_alg = partial(_linear_sim, d,
                                  self.scale,
                                  self.offset,
                                  self.origin)
        elif self.method == 'squared':
            num_sim_alg = partial(_squared_sim, d,
                                  self.scale,
                                  self.offset,
                                  self.origin)
        elif self.method in ['exp', 'exponential']:
            num_sim_alg = partial(_exp_sim, d,
                                  self.scale,
                                  self.offset,
                                  self.origin)
        elif self.method in ['gauss', 'gaussian']:
            num_sim_alg = partial(_gauss_sim, d,
                                  self.scale,
                                  self.offset,
                                  self.origin)
        else:
            raise ValueError(
                "The algorithm '{}' is not known.".format(self.method))

        c = num_sim_alg()
        c = fillna(c, self.missing_value)
        return c


class Geographic(BaseCompareFeature):
    """Compute the (partial) similarity between WGS84 coordinate values.

    Compare the geometric (haversine) distance between two WGS-
    coordinates. The similarity algorithms are 'step', 'linear', 'exp',
    'gauss' or 'squared'. The similarity functions are the same as in
    :meth:`recordlinkage.comparing.Compare.numeric`

    Parameters
    ----------
    lat1 : str or int
        The name or position of the column in the left DataFrame.
    lng1 : str or int
        The name or position of the column in the left DataFrame.
    lat2 : str or int
        The name or position of the column in the right DataFrame.
    lng2 : str or int
        The name or position of the column in the right DataFrame.
    method : str
        The metric used. Options 'step', 'linear', 'exp', 'gauss' or
        'squared'. Default 'linear'.
    offset : float
        The offset. See Compare.numeric.
    scale : float
        The scale of the numeric comparison method. See Compare.numeric.
        This argument is not available for the 'step' algorithm.
    origin : float
        The shift of bias between the values. See Compare.numeric.
    missing_value : numpy.dtype
        The value for a comparison with a missing value. Default 0.
    """

    name = "geographic"
    description = "Compare geographic attributes of record pairs."

    def __init__(self, labels_left, labels_right, method=None,
                 offset=0.0, scale=1.0, origin=0.0, missing_value=0.0,
                 label=None):
        super(Geographic, self).__init__(labels_left, labels_right,
                                         label=label)

        self.method = method
        self.offset = offset
        self.scale = scale
        self.origin = origin
        self.missing_value = missing_value

    def _compute_vectorized(self, lat1, lng1, lat2, lng2):

        d = _haversine_distance(lat1, lng1, lat2, lng2)

        if self.method == 'step':
            num_sim_alg = partial(_step_sim, d,
                                  self.offset,
                                  self.origin)
        elif self.method in ['linear', 'lin']:
            num_sim_alg = partial(_linear_sim, d,
                                  self.scale,
                                  self.offset,
                                  self.origin)
        elif self.method == 'squared':
            num_sim_alg = partial(_squared_sim, d,
                                  self.scale,
                                  self.offset,
                                  self.origin)
        elif self.method in ['exp', 'exponential']:
            num_sim_alg = partial(_exp_sim, d,
                                  self.scale,
                                  self.offset,
                                  self.origin)
        elif self.method in ['gauss', 'gaussian']:
            num_sim_alg = partial(_gauss_sim, d,
                                  self.scale,
                                  self.offset,
                                  self.origin)
        else:
            raise ValueError(
                "The algorithm '{}' is not known.".format(self.method))

        c = num_sim_alg()
        c = fillna(c, self.missing_value)

        return c


class Date(BaseCompareFeature):
    """Compute the (partial) similarity between date values.

    Parameters
    ----------
    s1 : str or int
        The name or position of the column in the left DataFrame.
    s2 : str or int
        The name or position of the column in the right DataFrame.
    swap_month_day : float
        The value if the month and day are swapped. Default 0.5.
    swap_months : list of tuples
        A list of tuples with common errors caused by the translating of
        months into numbers, i.e. October is month 10. The format of the
        tuples is (month_good, month_bad, value). Default : swap_months =
        [(6, 7, 0.5), (7, 6, 0.5), (9, 10, 0.5), (10, 9, 0.5)]
    missing_value : numpy.dtype
        The value for a comparison with a missing value. Default 0.0.

    """

    name = "date"
    description = "Compare date attributes of record pairs."

    def __init__(self, labels_left, labels_right, missing_value=0.0,
                 swap_month_day=0.5, swap_months='default', errors='coerce',
                 label=None):
        super(Date, self).__init__(labels_left, labels_right, label=label)

        self.missing_value = missing_value
        self.swap_months = swap_months
        self.swap_month_day = swap_month_day
        self.errors = errors

    def _compute_vectorized(self, s1, s2):

        # validate datatypes
        if str(s1.dtype) != 'datetime64[ns]':
            raise ValueError('Left column is not of type datetime64[ns]')

        if str(s2.dtype) != 'datetime64[ns]':
            raise ValueError('Right column is not of type datetime64[ns]')

        c = (s1 == s2).astype(np.int64)  # start with int64

        # The case is which there is a swap_month_day value given.
        if (self.swap_month_day and self.swap_month_day != 0):

            c[(s1.dt.year == s2.dt.year) &
              (s1.dt.month == s2.dt.day) &
              (s1.dt.day == s2.dt.month) &
              (c != 1)] = self.swap_month_day

        if (self.swap_months and self.swap_months != 0):

            if self.swap_months == 'default':
                self.swap_months = [(6, 7, 0.5),
                                    (7, 6, 0.5),
                                    (9, 10, 0.5),
                                    (10, 9, 0.5)]
            else:
                try:
                    if not all([len(x) == 3 for x in self.swap_months]):
                        raise Exception
                except Exception:
                    raise ValueError(
                        'swap_months must be a list of (first month, \
                        second month, value) tuples or lists. ')

            for month1, month2, value in self.swap_months:

                c[(s1.dt.year == s2.dt.year) &
                  (s1.dt.month == month1) & (s2.dt.month == month2) &
                  (s1.dt.day == s2.dt.day) &
                  (c != 1)] = value

        c = pandas.Series(c)
        c[s1.isnull() | s2.isnull()] = self.missing_value
        return c


class Frequency(BaseCompareFeature):
    """Compute the (relative) frequency of each variable.

    Parameters
    ----------
    left_on : str or int
        The name or position of the column in the left DataFrame.
    right_on : str or int
        The name or position of the column in the right DataFrame.
    normalise : bool
        Normalise the outcome. This is needed for good result in many
        classification models. Default True.
    missing_value : numpy.dtype
        The value for a comparison with a missing value. Default 0.0.

    """

    name = "frequency"
    description = "Compute the frequency."

    def __init__(self,
                 left_on=None,
                 right_on=None,
                 normalise=True,
                 missing_value=0.0,
                 label=None):
        super(Frequency, self).__init__(left_on, right_on, label=label)

        self.normalise = normalise
        self.missing_value = missing_value

    def _compute_frequency(self, col):

        # https://github.com/pydata/pandas/issues/3729
        na_value = 'NAN'
        value_count = col.fillna(na_value)

        c = value_count.groupby(by=value_count).transform('count')
        c = c.astype(np.float64)

        if self.normalise:
            c = c / len(col)

        # replace missing values
        c[col.isnull()] = self.missing_value

        return c

    def _compute_vectorized(self, *data):

        result = []

        if isinstance(data, tuple):
            for col in data:
                result_i = self._compute_frequency(col)
                result.append(result_i)
        else:
            result_i = self._compute_frequency(*data)
            result.append(result_i)

        return tuple(result)


class FrequencyA(Frequency):
    """Compute the frequency of a variable in the left dataframe.

    Parameters
    ----------
    on : str or int
        The name or position of the column in the left DataFrame.
    normalise : bool
        Normalise the outcome. This is needed for good result in many
        classification models. Default True.
    missing_value : numpy.dtype
        The value for a comparison with a missing value. Default 0.0.

    """

    name = "frequency"
    description = "Compute the frequency."

    def __init__(self, on=None, normalise=True, missing_value=0.0, label=None):
        super(FrequencyA, self).__init__(
            on,
            None,
            normalise=normalise,
            missing_value=missing_value,
            label=label)


class FrequencyB(Frequency):
    """Compute the frequency of a variable in the right dataframe.

    Parameters
    ----------
    on : str or int
        The name or position of the column in the right DataFrame.
    normalise : bool
        Normalise the outcome. This is needed for good result in many
        classification models. Default True.
    missing_value : numpy.dtype
        The value for a comparison with a missing value. Default 0.0.

    """

    name = "frequency"
    description = "Compute the frequency."

    def __init__(self, on=None, normalise=True, missing_value=0.0, label=None):
        super(FrequencyB, self).__init__(
            None,
            on,
            normalise=normalise,
            missing_value=missing_value,
            label=label)


CompareExact = DeprecationHelper(
    Exact, "This class is renamed into Exact.")
CompareString = DeprecationHelper(
    String, "This class is renamed into String.")
CompareNumeric = DeprecationHelper(
    Numeric, "This class is renamed into Numeric.")
CompareGeographic = DeprecationHelper(
    Geographic, "This class is renamed into Geographic.")
CompareDate = DeprecationHelper(
    Date, "This class is renamed into Date.")
