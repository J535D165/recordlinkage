from __future__ import division
from __future__ import unicode_literals

from functools import wraps

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
from recordlinkage.algorithms.string import smith_waterman_similarity
from recordlinkage.algorithms.string import longest_common_substring_similarity


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


def _check_labels(labels, df):

    labels = [labels] if not is_list_like(labels) else labels
    cols = df.columns.tolist() if isinstance(df, pandas.DataFrame) else df

    # Do some checks
    for label in labels:
        if label not in cols:
            raise KeyError(
                'label [%s] not in dataframe' % label
            )


@fillna_decorator(0)
def _string_internal(s1, s2, call_method, threshold=None, *args, **kw):

    c = call_method(s1, s2, *args, **kw)

    if threshold:
        return (c >= threshold).astype(np.float64)
    else:
        return c


@fillna_decorator(0)
def _num_internal(s1, s2, call_method, *args, **kwargs):
    """Internal function to compute the numeric similarity."""

    # compute the 1D distance between the values
    d = _1d_distance(s1, s2)

    return call_method(d, *args, **kwargs)


@fillna_decorator(0)
def _geo_internal(lat1, lng1, lat2, lng2, call_method, *args, **kw):

    # compute the 1D distance between the values
    d = _haversine_distance(lat1, lng1, lat2, lng2)

    return call_method(d, *args, **kw)


@fillna_decorator(0)
def _dates_internal(s1, s2, *args, **kwargs):

    return _compare_dates(s1, s2, *args, **kwargs)


class Compare(BaseCompare):
    """Compare(n_jobs=1, indexing_type='label')

    Class to compare record pairs with efficiently.

    Class to compare the attributes of candidate record pairs. The ``Compare``
    class has methods like ``string``, ``exact`` and ``numeric`` to initialise
    the comparing of the records. The ``compute`` method is used to start the
    actual comparing.

    Parameters
    ----------
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for comparing of record pairs.
        If -1, then the number of jobs is set to the number of cores.
    indexing_type : string, optional (default='label')
        The indexing type. The MultiIndex is used to index the DataFrame(s).
        This can be done with pandas ``.loc`` or with ``.iloc``. Use the value
        'label' to make use of ``.loc`` and 'position' to make use of
        ``.iloc``. The value 'position' is only available when the MultiIndex
        consists of integers. The value 'position' is much faster.

    Example
    -------
    Consider two historical datasets with census data to link. The datasets
    are named ``census_data_1980`` and ``census_data_1990``. The MultiIndex
    ``candidate_pairs`` contains the record pairs to compare. The record pairs
    are compared on the first name, last name, sex, date of birth, address,
    place, and income::

        # initialise class
        comp = recordlinkage.Compare()

        # initialise similarity measurement algorithms
        comp.string('first_name', 'name', method='jarowinkler')
        comp.string('lastname', 'lastname', method='jarowinkler')
        comp.exact('dateofbirth', 'dob')
        comp.exact('sex', 'sex')
        comp.string('address', 'address', method='levenshtein')
        comp.exact('place', 'place')
        comp.numeric('income', 'income')

        # the method .compute() returns the DataFrame with the feature vectors.
        comp.compute(candidate_pairs, census_data_1980, census_data_1990)

    """

    def exact(self, s1, s2, *args, **kwargs):
        """
        exact(s1, s2, agree_value=1, disagree_value=0, missing_value=0, label=None)

        Compare the record pairs exactly.

        This method initialises the exact similarity measurement between
        values. The similarity is 1 in case of agreement and 0 otherwise.

        Parameters
        ----------

        s1 : str or int
            Field name to compare in left DataFrame.
        s2 : str or int
            Field name to compare in right DataFrame.
        agree_value : float, str, numpy.dtype
            The value when two records are identical. Default 1. If 'values'
            is passed, then the value of the record pair is passed.
        disagree_value : float, str, numpy.dtype
            The value when two records are not identical.
        missing_value : float, str, numpy.dtype
            The value for a comparison with a missing value. Default 0.
        label : label
            The label of the column in the resulting dataframe.

        """

        label = kwargs.pop('label', None)

        return self._compare_vectorized(
            _compare_exact, s1, s2, args, kwargs,
            label=label,
            name="exact",
            description="Compare record pairs exactly."
        )

    def string(self, s1, s2, method='levenshtein', threshold=None,
               *args, **kwargs):
        """
        string(s1, s2, method='levenshtein', threshold=None, missing_value=0, label=None)

        Compute the (partial) similarity between strings values.

        This method initialises the similarity measurement between string
        values. The implemented algorithms are: 'jaro','jarowinkler',
        'levenshtein', 'damerau_levenshtein', 'qgram' or 'cosine'. In case of
        agreement, the similarity is 1 and in case of complete disagreement it
        is 0. The Python Record Linkage Toolkit uses the `jellyfish` package
        for the Jaro, Jaro-Winkler, Levenshtein and Damerau-Levenshtein
        algorithms.

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
        label : label
            The label of the column in the resulting dataframe.

        """

        if method == 'jaro':
            str_sim_alg = jaro_similarity

        elif method in ['jarowinkler', 'jaro_winkler', 'jw']:
            str_sim_alg = jarowinkler_similarity

        elif method == 'levenshtein':
            str_sim_alg = levenshtein_similarity

        elif method in ['dameraulevenshtein', 'damerau_levenshtein', 'dl']:
            str_sim_alg = damerau_levenshtein_similarity

        elif method == 'q_gram' or method == 'qgram':
            str_sim_alg = qgram_similarity

        elif method == 'cosine':
            str_sim_alg = cosine_similarity

        elif method in ['smith_waterman', "smithwaterman", "sw"]:
            str_sim_alg = smith_waterman_similarity

        elif method in ['longest_common_substring', 'lcs']:
            str_sim_alg = longest_common_substring_similarity

        else:
            raise ValueError(
                "The algorithm '{}' is not known.".format(method))

        label = kwargs.pop('label', None)
        args = (str_sim_alg, threshold) + args

        return self._compare_vectorized(
            _string_internal, s1, s2, args, kwargs,
            label=label,
            name="string '{}'".format(method),
            description="Compare record pairs on string with '{}'"
                        "algorithm".format(method)
        )

    def numeric(self, s1, s2, method='linear', *args, **kwargs):
        """
        numeric(s1, s2, method='linear', offset, scale, origin=0, missing_value=0, label=None)

        Compute the (partial) similarity between numeric values.

        This method initialises the similarity measurement between numeric
        values. The implemented algorithms are: 'step', 'linear', 'exp',
        'gauss' or 'squared'. In case of agreement, the similarity is 1 and in
        case of complete disagreement it is 0. The implementation is similar
        with numeric comparing in ElasticSearch, a full-text search tool. The
        parameters are explained in the image below (source ElasticSearch, The
        Definitive Guide)

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
        origin : str
            The shift of bias between the values. See image above.
        missing_value : numpy.dtype
            The value if one or both records have a missing value on the
            compared field. Default 0.
        label : label
            The label of the column in the resulting dataframe.

        Note
        ----
        Numeric comparing can be an efficient way to compare date/time
        variables. This can be done by comparing the timestamps.

        """

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

        label = kwargs.pop('label', None)
        args = (num_sim_alg,) + args

        return self._compare_vectorized(
            _num_internal, s1, s2, args, kwargs,
            label=label,
            name="numeric '{}'".format(method),
            description="Compare record pairs on numeric with '{}'"
                        "algorithm".format(method)
        )

    def geo(self, lat1, lng1, lat2, lng2, method='linear', *args, **kwargs):
        """
        geo(lat1, lng1, lat2, lng2, method='linear', offset, scale, origin=0, missing_value=0, label=None)

        Compute the (partial) similarity between WGS84 coordinate values.

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
        label : label
            The label of the column in the resulting dataframe.

        """

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

        label = kwargs.pop('label', None)
        args = (num_sim_alg,) + args

        return self._compare_vectorized(
            _geo_internal, (lat1, lng1), (lat2, lng2), args, kwargs,
            label=label,
            name="geographic '{}'".format(method),
            description="Compare record pairs on geographic with '{}'"
                        "algorithm".format(method)
        )

    def date(self, s1, s2, *args, **kwargs):
        """
        date(self, s1, s2, swap_month_day=0.5, swap_months='default', missing_value=0, label=None)

        Compute the (partial) similarity between date values.

        Parameters
        ----------
        s1 : str or int
            The name or position of the column in the left DataFrame.
        s2 : str or int
            The name or position of the column in the right DataFrame.
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
            The label of the column in the resulting dataframe.
        """

        label = kwargs.pop('label', None)

        return self._compare_vectorized(
            _dates_internal, s1, s2, args, kwargs,
            label=label,
            name="date",
            description="Compare record pairs on dates."
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
