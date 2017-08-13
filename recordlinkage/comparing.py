from __future__ import division
from __future__ import unicode_literals

import warnings
import multiprocessing as mp

import pandas
import numpy as np

from recordlinkage.base import BaseCompare
from recordlinkage.types import is_list_like
from recordlinkage.algorithms.compare import _compare_exact
from recordlinkage.algorithms.compare import _compare_dates
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


def fillna_decorator(missing_value=np.nan):

    def real_decorator(func):

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


def _check_labels(labels, df):

    labels = [labels] if not is_list_like(labels) else labels
    cols = df.columns.tolist() if isinstance(df, pandas.DataFrame) else df

    # Do some checks
    for label in labels:
        if label not in cols:
            raise KeyError(
                'label [%s] not in dataframe' % label
            )


class Compare(BaseCompare):
    """Compare record pairs with the tools in this class.

    Class to compare the attributes of candidate record pairs. The ``Compare``
    class has several methods to compare data such as string similarity
    measures, numeric metrics and exact comparison methods.

    Parameters
    ----------
    block_size : int
        The maximum size of data blocks. Default 1,000,000.

    Examples
    --------
    In the following example, the record pairs of two historical datasets with
    census data are compared. The datasets are named ``census_data_1980`` and
    ``census_data_1990``. The ``candidate_pairs`` are the record pairs to
    compare. The record pairs are compared on the first name, last name, sex,
    date of birth, address, place, and income.

    >>> comp = recordlinkage.Compare()
    >>> comp.string('first_name', 'name', method='jarowinkler')
    >>> comp.string('lastname', 'lastname', method='jarowinkler')
    >>> comp.exact('dateofbirth', 'dob')
    >>> comp.exact('sex', 'sex')
    >>> comp.string('address', 'address', method='levenshtein')
    >>> comp.exact('place', 'place')
    >>> comp.numeric('income', 'income')

    >>> comp.compute(candidate_pairs, census_data_1980, census_data_1990)

    The attribute ``vectors`` is the DataFrame with the comparison data. It
    can be called whenever you want.

    """

    def exact(self, s1, s2, *args, **kwargs):
        """
        exact(s1, s2, agree_value=1, disagree_value=0, missing_value=0, label=None)

        Compare the record pairs exactly.

        Parameters
        ----------

        s1 : label, pandas.Series
            Series or DataFrame to compare all fields.
        s2 : label, pandas.Series
            Series or DataFrame to compare all fields.
        agree_value : float, str, numpy.dtype
            The value when two records are identical. Default 1. If 'values'
            is passed, then the value of the record pair is passed.
        disagree_value : float, str, numpy.dtype
            The value when two records are not identical.
        missing_value : float, str, numpy.dtype
            The value for a comparison with a missing value. Default 0.
        label : label
            The name of the feature and the name of the column.

        """

        return self._compare_vectorized(_compare_exact, s1, s2, *args, **kwargs)

    def string(self, s1, s2, method='levenshtein', threshold=None, *args, **kwargs):
        """
        string(s1, s2, method='levenshtein', threshold=None, missing_value=0, label=None)

        Compare strings.

        Parameters
        ----------
        s1 : label, pandas.Series
            Series or DataFrame to compare all fields.
        s2 : label, pandas.Series
            Series or DataFrame to compare all fields.
        method : str
            A approximate string comparison method. Options are ['jaro',
            'jarowinkler', 'levenshtein', 'damerau_levenshtein', 'qgram',
            'cosine', 'smith_waterman', 'lcs']. Default: 'levenshtein'
        threshold : float, tuple of floats
            A threshold value. All approximate string comparisons higher or
            equal than this threshold are 1. Otherwise 0.
        missing_value : numpy.dtype
            The value for a comparison with a missing value. Default 0.
        label : label
            The name of the feature and the name of the column.

        """

        @fillna_decorator(0)
        def _string_internal(s1, s2, method, threshold=None, *args, **kwargs):
            """

            Internal function to compute the numeric similarity algorithms.

            """
            if method == 'jaro':
                str_sim_alg = jaro_similarity

            elif method in ['jarowinkler', 'jaro_winkler']:
                str_sim_alg = jarowinkler_similarity

            elif method == 'levenshtein':
                str_sim_alg = levenshtein_similarity

            elif method in ['dameraulevenshtein', 'damerau_levenshtein']:
                str_sim_alg = damerau_levenshtein_similarity

            elif method == 'q_gram' or method == 'qgram':
                str_sim_alg = qgram_similarity

            elif method == 'cosine':
                str_sim_alg = cosine_similarity

            elif method == 'smith_waterman' or method == "smithwaterman":
                str_sim_alg = smith_waterman_similarity

            elif method == 'lcs':
                str_sim_alg = longest_common_substring_similarity

            else:
                raise ValueError(
                    "The algorithm '{}' is not known.".format(method))

            c = str_sim_alg(s1, s2, *args, **kwargs)

            if threshold:
                return (c >= threshold).astype(np.float64)
            else:
                return c

        return self._compare_vectorized(
            _string_internal, s1, s2, method, threshold, *args, **kwargs
        )

    def numeric(self, s1, s2, method='linear', *args, **kwargs):
        """
        numeric(s1, s2, method='linear', offset, scale, origin=0, missing_value=0, label=None)

        Compute the similarity of numeric values.

        This method returns the similarity of two numeric values. The
        implemented algorithms are: 'step', 'linear', 'exp', 'gauss' or
        'squared'. In case of agreement, the similarity is 1 and in case of
        complete disagreement it is 0. The implementation is similar with
        numeric comparing in ElasticSearch, a full-text search tool. The
        parameters are explained in the image below (source ElasticSearch, The
        Definitive Guide)

        .. image:: /images/elas_1705.png
            :width: 100%
            :target: https://www.elastic.co/guide/en/elasticsearch/guide/current/decay-functions.html
            :alt: Decay functions, like in ElasticSearch

        Parameters
        ----------
        s1 : label, pandas.Series
            Series or DataFrame to compare all fields.
        s2 : label, pandas.Series
            Series or DataFrame to compare all fields.
        method : float
            The metric used. Options 'step', 'linear', 'exp', 'gauss' or
            'squared'. Default 'linear'.
        offset : float
            The offset. See image above.
        scale : float
            The scale of the numeric comparison method. See the image above.
            This argument is not available for the 'step' algorithm.
        origin : str
            The shift of bias between the values. See image above.
        missing_value : numpy.dtype
            The value if one or both records have a missing value on the
            compared field. Default 0.
        label : label
            The name of the feature and the name of the column.

        Note
        ----
        Numeric comparing can be an efficient way to compare date/time
        variables. This can be done by comparing the timestamps.

        """

        @fillna_decorator(0)
        def _num_internal(s1, s2, method, *args, **kwargs):
            """Internal function to compute the numeric similarity."""

            # compute the 1D distance between the values
            d = _1d_distance(s1, s2)

            if method == 'step':
                num_sim_alg = _step_sim
            elif method in ['linear', 'lin']:
                num_sim_alg = _linear_sim
            elif method == 'squared':
                num_sim_alg = _squared_sim
            elif method in ['exp', 'exponential']:
                num_sim_alg = _exp_sim
            elif method in ['gauss', 'gaussian']:
                num_sim_alg = _gauss_sim
            else:
                raise ValueError(
                    "The algorithm '{}' is not known.".format(method))

            return num_sim_alg(d, *args, **kwargs)

        return self._compare_vectorized(_num_internal, s1, s2, method, *args, **kwargs)

    def geo(self, lat1, lng1, lat2, lng2, method='linear', *args, **kwargs):
        """
        geo(lat1, lng1, lat2, lng2, method='linear', offset, scale, origin=0, missing_value=0, label=None)

        Compute the similarity of two WGS84 coordinates.

        Compare the geometric (haversine) distance between two WGS-
        coordinates. The similarity algorithms are 'step', 'linear', 'exp',
        'gauss' or 'squared'. The similarity functions are the same as in
        :meth:`recordlinkage.comparing.Compare.numeric`

        Parameters
        ----------
        lat1 : pandas.Series, numpy.array, label/string
            Series with Lat-coordinates
        lng1 : pandas.Series, numpy.array, label/string
            Series with Lng-coordinates
        lat2 : pandas.Series, numpy.array, label/string
            Series with Lat-coordinates
        lng2 : pandas.Series, numpy.array, label/string
            Series with Lng-coordinates
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
        label : label
            The name of the feature and the name of the column.

        """

        @fillna_decorator(0)
        def _num_internal(lat1, lng1, lat2, lng2, method, *args, **kwargs):
            """

            Internal function to compute the numeric similarity algorithms.

            """

            # compute the 1D distance between the values
            d = _haversine_distance(lat1, lng1, lat2, lng2)

            print(d)

            if method == 'step':
                num_sim_alg = _step_sim
            elif method in ['linear', 'lin']:
                num_sim_alg = _linear_sim
            elif method == 'squared':
                num_sim_alg = _squared_sim
            elif method in ['exp', 'exponential']:
                num_sim_alg = _exp_sim
            elif method in ['gauss', 'gaussian']:
                num_sim_alg = _gauss_sim
            else:
                raise ValueError(
                    "The algorithm '{}' is not known.".format(method)
                )

            return num_sim_alg(d, *args, **kwargs)

        return self._compare_vectorized(
            _num_internal, (lat1, lng1), (lat2, lng2),
            method, *args, **kwargs
        )

    def date(self, s1, s2, swap_month_day=0.5, swap_months='default', *args, **kwargs):
        """
        date(self, s1, s2, swap_month_day=0.5, swap_months='default', missing_value=0, label=None)

        Compare two dates.

        Parameters
        ----------
        s1 : pandas.Series, numpy.array, label/string
            Dates. This can be a Series, DatetimeIndex or DataFrame (with
            columns 'year', 'month' and 'day').
        s2 : pandas.Series, numpy.array, label/string
            This can be a Series, DatetimeIndex or DataFrame (with columns
            'year', 'month' and 'day').
        swap_month_day : float
            The value if the month and day are swapped.
        swap_months : list of tuples
            A list of tuples with common errors caused by the translating of
            months into numbers, i.e. October is month 10. The format of the
            tuples is (month_good, month_bad, value). Default : swap_months =
            [(6, 7, 0.5), (7, 6, 0.5), (9, 10, 0.5), (10, 9, 0.5)]
        missing_value : numpy.dtype
            The value for a comparison with a missing value. Default 0.
        label : label
            The name of the feature and the name of the column.

        """

        @fillna_decorator(0)
        def _dummy_compare_dates(s1, s2, *args, **kwargs):

            return _compare_dates(s1, s2, *args, **kwargs)

        return self._compare_vectorized(
            _dummy_compare_dates, s1, s2,
            swap_month_day=swap_month_day, swap_months=swap_months,
            *args, **kwargs
        )


def _missing(*args):
    """ Missing values.

    Internal function to return the index of record pairs with missing values
    """

    return np.any(
        np.concatenate(
            [np.array(pandas.DataFrame(arg).isnull()) for arg in args],
            axis=1),
        axis=1)
