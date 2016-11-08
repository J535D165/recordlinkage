from __future__ import division
from __future__ import unicode_literals

import warnings

import pandas
from pandas.types.inference import is_list_like
import numpy as np

from recordlinkage.utils import _resample
from recordlinkage.algorithms.distance import _1d_distance, _haversine_distance
from recordlinkage.algorithms.numeric import _step_sim, \
    _linear_sim, _squared_sim, _exp_sim, _gauss_sim
from recordlinkage.algorithms.string import jaro_similarity, \
    jarowinkler_similarity, levenshtein_similarity, \
    damerau_levenshtein_similarity, qgram_similarity, cosine_similarity


def fillna_decorator(missing_value=np.nan):

    def real_decorator(func):

        def func_wrapper(*args, **kwargs):

            mv = kwargs.pop('missing_value', missing_value)

            result = func(*args, **kwargs)

            # fill missing values if missing_value is not a missing value like NaN or None.
            if pandas.notnull(mv):
                if isinstance(result, (np.ndarray)):
                    result[np.isnan(result)] = mv
                else:
                    result.fillna(mv, inplace=True)

            return result

        return func_wrapper

    return real_decorator


# def _label_or_column(label_or_column, dataframe):
#     """

#     This internal function to check if the argument is a column label or a
#     pandas.Series or pandas.DataFrame. If the argument is a Series or
#     DataFrame, nothing is done.

#     """
#     try:
#         return dataframe[label_or_column]
#     except Exception:

#         if isinstance(label_or_column, (pandas.Series, pandas.DataFrame)):
#             return label_or_column
#         else:
#             raise ValueError("The label or column has to be a valid label " +
#                              "or pandas.Series or pandas.DataFrame. ")


class CompareCore(object):
    """

    Core class for comparing records.

    """

    def __init__(self, pairs, df_a=None, df_b=None, batch=False):

        # The dataframes
        self.df_a = df_a
        self.df_b = df_b

        # The candidate record pairs
        self.pairs = pairs

        self.batch = batch
        self._batch_functions = []

        # The resulting data
        self.vectors = pandas.DataFrame(index=pairs)

        # self.ndim = self._compute_dimension(pairs)

    def compare(self, comp_func, labels_a, labels_b, *args, **kwargs):
        """

        Core method to compare records. This method takes a function and data
        from both records in the record pair. The data is compared with the
        compare function. The built-in methods also use this function.

        Example:

        .. code-block:: python

            >>> comp = recordlinkage.Compare(PAIRS, DATAFRAME1, DATAFRAME2)
            >>> comp.exact('first_name', 'name')

            >>> # same as
            >>> comp.compare(recordlinkage._compare_exact, 'first_name', 'name')

        :param comp_func: A comparison function. This function can be a
                built-in function or a user defined comparison function.
        :param labels_a: The labels, Series or DataFrame to compare.
        :param labels_b: The labels, Series or DataFrame to compare.
        :param name: The name of the feature and the name of the column.
        :param store: Store the result in the dataframe.

        :type comp_func: function
        :type labels_a: label, pandas.Series, pandas.DataFrame
        :type labels_b: label, pandas.Series, pandas.DataFrame
        :type name: label
        :type store: bool, default True

        :return: The DataFrame Compare.vectors
        :rtype: standardise.DataFrame
        """

        # Add to batch compare functions
        self._batch_functions.append(
            (comp_func, labels_a, labels_b, args, kwargs)
        )

        # Run directly if not batch
        if not self.batch:
            return self.run()

    def run(self):
        """

        Batch method for comparing records. This method excecutes the methods
        called before in one time. This method may decrease the computation
        time. This function works ONLY when ``batch=True`` is set in the class
        ``Compare`` and ``run`` is called in the end.

        Example:

        .. code-block:: python

            >>> # This example is almost 3 times faster than the traditional one.
            >>> comp = recordlinkage.Compare(..., batch=True)
            >>> comp.exact('first_name', 'name')
            >>> comp.exact('surname', 'surname')
            >>> comp.exact('date_of_birth', 'dob')
            >>> comp.run()

        :return: The comparison vectors (Compare.vectors)
        :rtype: standardise.DataFrame
        """

        if not self._batch_functions:
            raise Exception("No batch functions found. \
                Check if batch=True in recordlinkage.Compare")

        # Collect the labels
        labelsA = []
        labelsB = []

        for comp_func, lbls_a, lbls_b, args, kwargs in self._batch_functions:

            if isinstance(lbls_a, (tuple, list)):
                labelsA.extend(lbls_a)
            else:
                labelsA.append(lbls_a)

        for comp_func, lbls_a, lbls_b, args, kwargs in self._batch_functions:

            if isinstance(lbls_b, (tuple, list)):
                labelsB.extend(lbls_b)
            else:
                labelsB.append(lbls_b)

        labelsA = list(set(labelsA))
        labelsB = list(set(labelsB))

        # Make selections of columns
        dataA = _resample(self.df_a[labelsA], self.pairs, 0)
        dataB = _resample(self.df_b[labelsB], self.pairs, 1)

        for comp_func, lbls_a, lbls_b, args, kwargs in self._batch_functions:

            # The name of the comparison
            name = kwargs.pop('name', None)

            # # always true, but if passed then ignored
            # store = kwargs.pop('store', True)
            if 'store' in kwargs.keys():
                warnings.warn("The argument store might be removed \
                    in the next version.", DeprecationWarning)

            # Sample the data and add it to the arguments.
            lbls_b = [lbls_b] if not is_list_like(lbls_b) else lbls_b
            lbls_a = [lbls_a] if not is_list_like(lbls_a) else lbls_a

            args = tuple(dataA.loc[:, da] for da in reversed(lbls_a)) + \
                tuple(dataB.loc[:, db] for db in reversed(lbls_b)) + args

            # Compute the comparison
            c = comp_func(*tuple(args), **kwargs)

            # if a pandas series is returned, overwrite the index. The
            # returned index can be different than the MultiIndex passed to
            # the compare function.
            if isinstance(c, pandas.Series):
                c.index = self.vectors.index

            # append column to Compare.vectors
            name_or_id = name if name else len(self.vectors.columns)
            self.vectors[name_or_id] = c

        # Reset the batch functions
        self._batch_functions = []

        # Return the Series
        if not self.batch:
            return self.vectors[name_or_id].rename(name)
        # Return the DataFrame
        else:
            return self.vectors


class Compare(CompareCore):
    """

    Class to compare the attributes of candidate record pairs. The ``Compare``
    class has several methods to compare data such as string similarity
    measures, numeric metrics and exact comparison methods.

    :param pairs: A MultiIndex of candidate record pairs.
    :param df_a: The first dataframe.
    :param df_b: The second dataframe.

    :type pairs: pandas.MultiIndex
    :type df_a: pandas.DataFrame
    :type df_b: pandas.DataFrame

    :returns: A compare class
    :rtype: recordlinkage.Compare

    :var pairs: The candidate record pairs.
    :var df_a: The first DataFrame.
    :var df_b: The second DataFrame.
    :var vectors: The DataFrame with comparison data.

    :vartype pairs: pandas.MultiIndex
    :vartype df_a: pandas.DataFrame
    :vartype df_b: pandas.DataFrame
    :vartype vectors: pandas.DataFrame

    Example:

    In the following example, the record pairs of two historical datasets with
    census data are compared. The datasets are named ``census_data_1980`` and
    ``census_data_1990``. The ``candidate_pairs`` are the record pairs to
    compare. The record pairs are compared on the first name, last name, sex,
    date of birth, address, place, and income.

    .. code:: python

        >>> comp = recordlinkage.Compare(
            candidate_pairs, census_data_1980, census_data_1990
            )
        >>> comp.string('first_name', 'name', method='jarowinkler')
        >>> comp.string('lastname', 'lastname', method='jarowinkler')
        >>> comp.exact('dateofbirth', 'dob')
        >>> comp.exact('sex', 'sex')
        >>> comp.string('address', 'address', method='levenshtein')
        >>> comp.exact('place', 'place')
        >>> comp.numeric('income', 'income')
        >>> print(comp.vectors.head())

    The attribute ``vectors`` is the DataFrame with the comparison data. It
    can be called whenever you want.

    """

    # def __init__(self, pairs, df_a=None, df_b=None, batch=False):

    #     # The dataframes
    #     self.df_a = df_a
    #     self.df_b = df_b

    #     # The candidate record pairs
    #     self.pairs = pairs

    #     self.batch = batch
    #     self._batch_functions = []

    #     # The resulting data
    #     self.vectors = pandas.DataFrame(index=pairs)

    #     # self.ndim = self._compute_dimension(pairs)

    def exact(self, s1, s2, *args, **kwargs):
        """
        exact(s1, s2, agree_value=1, disagree_value=0, missing_value=0, name=None, store=True)

        Compare the record pairs exactly.

        :param s1: Series or DataFrame to compare all fields.
        :param s2: Series or DataFrame to compare all fields.
        :param agree_value: The value when two records are identical.
                Default 1. If 'values' is passed, then the value of the record
                pair is passed.
        :param disagree_value: The value when two records are not identical.
        :param missing_value: The value for a comparison with a missing value.
                Default 0.
        :param name: The name of the feature and the name of the column.
        :param store: Store the result in the dataframe. Default True

        :type s1: label, pandas.Series
        :type s2: label, pandas.Series
        :type agree_value: float, str, numpy.dtype
        :type disagree_value: float, str, numpy.dtype
        :type missing_value: float, str, numpy.dtype
        :type name: label
        :type store: bool

        :return: A Series with comparison values.
        :rtype: pandas.Series

        """

        return self.compare(_compare_exact, s1, s2, *args, **kwargs)

    def string(self, s1, s2, method='levenshtein', threshold=None, *args, **kwargs):
        """
        string(s1, s2, method='levenshtein', threshold=None, missing_value=0, name=None, store=True)

        Compare strings.

        :param s1: Series or DataFrame to compare all fields.
        :param s2: Series or DataFrame to compare all fields.
        :param method: A approximate string comparison method. Options are
                ['jaro', 'jarowinkler', 'levenshtein', 'damerau_levenshtein',
                'qgram', 'cosine']. Default: 'levenshtein'
        :param threshold: A threshold value. All approximate string
                comparisons higher or equal than this threshold are 1.
                Otherwise 0.
        :param missing_value: The value for a comparison with a missing value.
                Default 0.
        :param name: The name of the feature and the name of the column.
        :param store: Store the result in the dataframe. Default True

        :type s1: label, pandas.Series
        :type s2: label, pandas.Series
        :type method: str
        :type threshold: float, tuple of floats
        :type missing_value: numpy.dtype
        :type name: label
        :type store: bool

        :return: A Series with similarity values. Values equal or between 0
                and 1.
        :rtype: pandas.Series

        .. note::

            The 'jarowinkler', 'jaro', 'levenshtein' and 'damerau_levenshtein'
            algorithms use the package 'jellyfish' for string similarity
            measures. It can be installed with pip (``pip install
            jellyfish``).

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

            elif method in ['dameraulevenshtein', 'dameraulevenshtein']:
                str_sim_alg = damerau_levenshtein_similarity

            elif method == 'q_gram' or method == 'qgram':
                str_sim_alg = qgram_similarity

            elif method == 'cosine':
                str_sim_alg = cosine_similarity

            else:
                raise ValueError("The algorithm '{}' is not known.".format(method))

            c = str_sim_alg(s1, s2, *args, **kwargs)

            if threshold:
                return (c >= threshold).astype(np.float64)
            else:
                return c

        return self.compare(
            _string_internal, s1, s2, method, threshold, *args, **kwargs
        )

    def numeric(self, s1, s2, method='linear', *args, **kwargs):
        """
        numeric(s1, s2, method='linear', offset, scale, origin=0, missing_value=0, name=None, store=True)

        This method returns the similarity between two numeric values. The
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

        :param s1: Series or DataFrame to compare all fields.
        :param s2: Series or DataFrame to compare all fields.
        :param method: The metric used. Options 'step', 'linear', 'exp',
                'gauss' or 'squared'. Default 'linear'.
        :param offset: The offset. See image above.
        :param scale: The scale of the numeric comparison method. See the
                image above. This argument is not available for the 'step'
                algorithm.
        :param origin: The shift of bias between the values. See image
                above.
        :param missing_value: The value if one or both records have a
                missing value on the compared field. Default 0.
        :param name: The name of the feature and the name of the column.
        :param store: Store the result in the dataframe. Default True

        :type s1: label, pandas.Series
        :type s2: label, pandas.Series
        :type offset: float
        :type scale: float
        :type origin: float
        :type method: str
        :type missing_value: numpy.dtype
        :type name: label
        :type store: bool

        :return: A Series with comparison values.
        :rtype: pandas.Series

        .. note::

            Numeric comparing can be an efficient way to compare date/time
            variables. This can be done by comparing the timestamps.

        """

        @fillna_decorator(0)
        def _num_internal(s1, s2, method, *args, **kwargs):
            """

            Internal function to compute the numeric similarity algorithms.

            """

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
                raise ValueError("The algorithm '{}' is not known.".format(method))

            return num_sim_alg(d, *args, **kwargs)

        return self.compare(_num_internal, s1, s2, method, *args, **kwargs)

    def geo(self, lat1, lng1, lat2, lng2, method='linear', *args, **kwargs):
        """
        geo(lat1, lng1, lat2, lng2, method='linear', offset, scale, origin=0, missing_value=0, name=None, store=True)

        Compare the geometric (haversine) distance between two WGS-
        coordinates. The similarity algorithms are 'step', 'linear', 'exp',
        'gauss' or 'squared'. The similarity functions are the same as in
        :meth:`recordlinkage.comparing.Compare.numeric`

        :param lat1: Series with Lat-coordinates
        :param lng1: Series with Lng-coordinates
        :param lat2: Series with Lat-coordinates
        :param lng2: Series with Lng-coordinates
        :param method: The metric used. Options 'step', 'linear', 'exp',
                'gauss' or 'squared'. Default 'linear'.
        :param offset: The offset. See Compare.numeric.
        :param scale: The scale of the numeric comparison method. See
                Compare.numeric. This argument is not available for the
                'step' algorithm.
        :param origin: The shift of bias between the values. See
                Compare.numeric.
        :param missing_value: The value for a comparison with a missing value.
                Default 0.
        :param name: The name of the feature and the name of the column.
        :param store: Store the result in the dataframe. Default True.

        :type lat1: pandas.Series, numpy.array, label/string
        :type lng1: pandas.Series, numpy.array, label/string
        :type lat2: pandas.Series, numpy.array, label/string
        :type lng2: pandas.Series, numpy.array, label/string
        :type method: str
        :type offset: float
        :type scale: float
        :type origin: float
        :type missing_value: numpy.dtype
        :type name: label
        :type store: bool

        :return: A Series with comparison values.
        :rtype: pandas.Series
        """

        @fillna_decorator(0)
        def _num_internal(lat1, lng1, lat2, lng2, method, *args, **kwargs):
            """

            Internal function to compute the numeric similarity algorithms.

            """

            # compute the 1D distance between the values
            d = _haversine_distance(lat1, lng1, lat2, lng2)

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

        return self.compare(
            _num_internal, (lat1, lng1), (lat2, lng2),
            method, *args, **kwargs
        )

    def date(self, s1, s2, swap_month_day=0.5, swap_months='default', *args, **kwargs):
        """
        date(self, s1, s2, swap_month_day=0.5, swap_months='default', missing_value=0, name=None, store=True)

        Compare dates.

        :param s1: Dates. This can be a Series, DatetimeIndex or DataFrame
                (with columns 'year', 'month' and 'day').
        :param s2: This can be a Series, DatetimeIndex or DataFrame
                (with columns 'year', 'month' and 'day').
        :param swap_month_day: The value if the month and day are swapped.
        :param swap_months: A list of tuples with common errors caused by the
                translating of months into numbers, i.e. October is month 10.
                The format of the tuples is (month_good, month_bad, value).
                Default: swap_months = [(6, 7, 0.5), (7, 6, 0.5), (9, 10, 0.5),
                (10, 9, 0.5)]
        :param missing_value: The value for a comparison with a missing value.
                Default 0.
        :param name: The name of the feature and the name of the column.
        :param store: Store the result in the dataframe. Default True.

        :type s1: pandas.Series, numpy.array, label/string
        :type s2: pandas.Series, numpy.array, label/string
        :type swap_month_day: float
        :type swap_months: list of tuples
        :type missing_value: numpy.dtype
        :type name: label
        :type store: bool

        :return: A Series with date comparison values.
        :rtype: pandas.Series
        """

        return self.compare(
            _compare_dates, s1, s2,
            swap_month_day=swap_month_day, swap_months=swap_months,
            *args, **kwargs
        )


def _missing(*args):
    """
    Internal function to return the index of record pairs with missing values
    """

    return np.any(
        np.concatenate(
            [np.array(pandas.DataFrame(arg).isnull()) for arg in args],
            axis=1),
        axis=1)


def _compare_exact(s1, s2, agree_value=1, disagree_value=0, missing_value=0):
    # dtypes can be hard if the passed parameters (agreement, disagreement,
    # missing_value) are of different types.
    # http://chris.friedline.net/2015-12-15-rutgers/lessons/python2/03-data-types-and-format.html

    # Convert to pandas.Series if (numpy) arrays are passed.
    if not isinstance(s1, pandas.Series):
        s1 = pandas.Series(s1, index=s1.index)

    if not isinstance(s2, pandas.Series):
        s2 = pandas.Series(s2, index=s2.index)

    # Values or agree/disagree
    if agree_value == 'value':
        compare = s1.copy()
        compare[s1 != s2] = disagree_value

    else:
        compare = pandas.Series(disagree_value, index=s1.index)
        compare[s1 == s2] = agree_value

    # Only when disagree value is not identical with the missing value
    if disagree_value != missing_value:

        compare[(s1.isnull() | s2.isnull())] = missing_value

    return compare


@fillna_decorator(0)
def _compare_dates(s1, s2, swap_month_day=0.5, swap_months='default',
                   errors='coerce', *args, **kwargs):

    missing_pairs = (s1.isnull() | s2.isnull()).values

    # if isinstance(s1, (pandas.Series)):
    #     s1 = s1.values

    # if isinstance(s2, (pandas.Series)):
    #     s2 = s2.values

    s1_dti = pandas.to_datetime(s1.values, errors=errors, *args, **kwargs)
    s2_dti = pandas.to_datetime(s2.values, errors=errors, *args, **kwargs)

    c = (s1 == s2).astype(np.int64)

    # The case is which there is a swap_month_day value given.
    if (swap_month_day and swap_month_day != 0):

        # if isinstance(swap_month_day, float):
        #     c = c.astype(np.float64)
        # elif isinstance(swap_month_day, int):
        #     c = c.astype(np.int64)
        # else:
        #     c = c.astype(object)

        c[(s1_dti.year == s2_dti.year) &
          (s1_dti.month == s2_dti.day) &
          (s1_dti.day == s2_dti.month) &
          (c != 1)] = swap_month_day

    if (swap_months and swap_months != 0):

        if swap_months == 'default':
            swap_months = [(6, 7, 0.5),
                           (7, 6, 0.5),
                           (9, 10, 0.5),
                           (10, 9, 0.5)]
        else:
            try:
                if not all([len(x) == 3 for x in swap_months]):
                    raise Exception
            except Exception:
                raise ValueError(
                    'swap_months must be a list of (first month, \
                    second month, value) tuples or lists. ')

        for month1, month2, value in swap_months:

            # if isinstance(value, float):
            #     c = c.astype(np.float64)
            # elif isinstance(value, int):
            #     c = c.astype(np.int64)
            # else:
            #     c = c.astype(object)

            c[(s1_dti.year == s2_dti.year) &
              (s1_dti.month == month1) & (s2_dti.month == month2) &
              (s1_dti.day == s2_dti.day) &
              (c != 1)] = value

    c = pandas.Series(c)
    c[missing_pairs] = np.nan

    return c
