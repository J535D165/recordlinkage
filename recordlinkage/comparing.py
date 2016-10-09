from __future__ import division
from __future__ import unicode_literals

import warnings

import pandas
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from recordlinkage.utils import _label_or_column, _resample


def _import_jellyfish():

    try:
        import jellyfish
        return jellyfish

    except ImportError:
        raise ImportError(
            "Install the module 'jellyfish' to use the following " +
            "string metrics: 'jaro', 'jarowinkler', 'levenshtein'" +
            " and 'damerau_levenshtein'."
        )


class Compare(object):
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

        if not self.batch:

            name = kwargs.pop('name', None)
            store = kwargs.pop('store', True)

            # Sample the data and add it to the arguments.
            labels_b = [labels_b] if not isinstance(
                labels_b, (tuple, list)) else labels_b
            labels_a = [labels_a] if not isinstance(
                labels_a, (tuple, list)) else labels_a

            args = tuple(_resample(
                _label_or_column(da, self.df_a),
                self.pairs,
                0) for da in reversed(labels_a)) + \
                tuple(_resample(
                    _label_or_column(db, self.df_b),
                    self.pairs,
                    1) for db in reversed(labels_b)) + args

            c = comp_func(*tuple(args), **kwargs)

            # Strange bug in pandas?
            try:
                # down to numpy.array
                c = c.values
            except AttributeError:
                pass

            # Store the result
            if store:
                # append column
                name_or_id = name if name else len(self.vectors.columns)

                self.vectors[name_or_id] = c

            return pandas.Series(c, index=self.pairs, name=name)

        else:

            # Add to batch compare functions
            self._batch_functions.append(
                (comp_func, labels_a, labels_b, args, kwargs))

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
            raise Exception("No batch functions found. "
                            "Check if batch=True in recordlinkage.Compare")

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

        # Make selections of columns
        dataA = _resample(self.df_a[labelsA], self.pairs, 1)
        dataB = _resample(self.df_b[labelsB], self.pairs, 0)

        for comp_func, lbls_a, lbls_b, args, kwargs in self._batch_functions:

            name = kwargs.pop('name', None)
            # always true, but if passed then ignored
            store = kwargs.pop('store', None)

            # Sample the data and add it to the arguments.
            lbls_b = [lbls_b] if not isinstance(
                lbls_b, (tuple, list)) else lbls_b
            lbls_a = [lbls_a] if not isinstance(
                lbls_a, (tuple, list)) else lbls_a

            args = tuple(dataA.loc[:, da] for da in reversed(lbls_a)) + \
                tuple(dataB.loc[:, db] for db in reversed(lbls_b)) + args

            # Compute the comparison
            c = comp_func(*tuple(args), **kwargs)

            # append column to Compare.vectors
            name_or_id = name if name else len(self.vectors.columns)
            self.vectors[name_or_id] = c

        # Reset the batch functions
        self._batch_functions = []

        return self.vectors

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

    def numeric(self, s1, s2, *args, **kwargs):
        """
        numeric(s1, s2, threshold=None, method='step', missing_value=0, name=None, store=True)

        This method returns the similarity between two numeric values. The
        following algorithms can be used: 'step', 'linear' or 'squared'. These
        functions are defined on the interval (-threshold, threshold). In case
        of agreement, the similarity is 1 and in case of complete disagreement
        it is 0. For linear and squared methods is also partial agreement
        possible.

        :param s1: Series or DataFrame to compare all fields.
        :param s2: Series or DataFrame to compare all fields.
        :param threshold: The threshold size. Can be a tuple with two values
                or a single number.
        :param method: The metric used. Options 'step', 'linear' or 'squared'.
                Default 'step'.
        :param missing_value: The value for a comparison with a missing value.
                Default 0.
        :param name: The name of the feature and the name of the column.
        :param store: Store the result in the dataframe. Default True

        :type s1: label, pandas.Series
        :type s2: label, pandas.Series
        :type threshold: float, tuple of floats
        :type method: 'step', 'linear' or 'squared'
        :type missing_value: numpy.dtype
        :type name: label
        :type store: bool

        :return: A Series with comparison values.
        :rtype: pandas.Series

        """

        return self.compare(_numeric_sim, s1, s2, *args, **kwargs)

    def string(self, s1, s2, *args, **kwargs):
        """
        string(s1, s2, method='levenshtein', threshold=None, missing_value=0, name=None, store=True)

        Compare string values with a similarity approximation.

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

        return self.compare(_string_sim, s1, s2, *args, **kwargs)

    def geo(self, lat1, lng1, lat2, lng2, *args, **kwargs):
        """
        geo(lat1, lng1, lat2, lng2, threshold=None, method='step', missing_value=0, name=None, store=True)

        [Experimental] Compare geometric WGS-coordinates with a tolerance
        [window.

        :param lat1: Series with Lat-coordinates
        :param lng1: Series with Lng-coordinates
        :param lat2: Series with Lat-coordinates
        :param lng2: Series with Lng-coordinates
        :param threshold: The threshold size. Can be a tuple with two values
                or a single number.
        :param method: The metric used. Options 'step', 'linear' or 'squared'.
        :param missing_value: The value for a comparison with a missing value.
                Default 0.
        :param name: The name of the feature and the name of the column.
        :param store: Store the result in the dataframe. Default True.

        :type lat1: pandas.Series, numpy.array, label/string
        :type lng1: pandas.Series, numpy.array, label/string
        :type lat2: pandas.Series, numpy.array, label/string
        :type lng2: pandas.Series, numpy.array, label/string
        :type threshold: float, tuple of floats
        :type method: str
        :type missing_value: numpy.dtype
        :type name: label
        :type store: bool

        :return: A Series with comparison values.
        :rtype: pandas.Series
        """

        return self.compare(
            _geo_sim,
            (lat1, lng1),
            (lat2, lng2),
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

    if agree_value == 'value':
        compare = s1.copy()
        compare[s1 != s2] = disagree_value

    else:
        compare = np.where(s1 == s2, agree_value, disagree_value)

    compare = pandas.Series(compare, index=s1.index)

    # Only when disagree value is not identical with the missing value
    if disagree_value != missing_value:
        compare[_missing(s1, s2)] = missing_value

    return compare


def _numeric_sim(s1, s2, threshold=None, method='step', missing_value=0):

    threshold_left, threshold_right = threshold if isinstance(
        threshold, (list, tuple)) else (-threshold, threshold)

    a = threshold_right + threshold_left
    b = 2 / (threshold_right - threshold_left)

    # numeric step functions
    if method == 'step':
        d = (_linear_distance(s1, s2, a=a, b=b) <= 1).astype(int)

    # numeric linear functions
    elif method == 'linear':
        d = 1 - _linear_distance(s1, s2, a=a, b=b)
        d[d < 0] = 0

    # numeric squared function
    elif method == 'squared':
        d = 1 - _squared_distance(s1, s2, a=a, b=b)
        d[d < 0] = 0

    # numeric haversine (for coordinates)
    elif method == 'haversine':
        lat1, lng1 = s1
        lat2, lng2 = s2
        d = 1 - _haversine_distance(lat1, lng1, lat2, lng2) / threshold
        d[d < 0] = 0
    else:
        raise KeyError('The given algorithm is not found.')

    d.fillna(missing_value, inplace=True)

    return d


def _geo_sim(
        lat1, lng1, lat2, lng2,
        threshold=None, method='step', missing_value=0):

    a = threshold
    b = 1 / threshold

    # numeric step functions
    if method == 'step':
        d = (_haversine_distance(lat1, lng1, lat2, lng2) <= 1).astype(int)

    # numeric linear functions
    elif method == 'linear':
        'abs(((s2-s1)-a)*b)'
        d = 1 - abs((_haversine_distance(lat1, lng1, lat2, lng2) - a) * b)
        d[d < 0] = 0

    # numeric squared function
    elif method == 'squared':
        d = 1 - (_haversine_distance(lat1, lng1, lat2, lng2) - a)**2 * b**2
        d[d < 0] = 0

    else:
        raise KeyError('The given algorithm is not found.')

    d.fillna(missing_value, inplace=True)

    return d


def _string_sim(s1, s2, method='levenshtein', threshold=None, missing_value=0):

    if method == 'jaro':
        approx = jaro_similarity(s1, s2)

    elif method in ['jarowinkler', 'jaro_winkler']:
        approx = jarowinkler_similarity(s1, s2)

    elif method == 'levenshtein':
        approx = levenshtein_similarity(s1, s2)

    elif method in ['dameraulevenshtein', 'damerau_levenshtein']:
        approx = damerau_levenshtein_similarity(s1, s2)

    elif method in ['qgram', 'q_gram']:
        approx = qgram_similarity(s1, s2)

    elif method == 'cosine':
        approx = cosine_similarity(s1, s2)

    else:
        raise ValueError("""Algorithm '{}' not found.""".format(method))

    comp = (approx >= threshold).astype(
        int) if threshold is not None else approx

    # Only for missing values
    comp[_missing(s1, s2)] = missing_value

    return comp


def _linear_distance(s1, s2, a=0, b=1):

    expr = 'abs(((s2-s1)-a)*b)'

    # PANDAS BUG?
    # return pandas.eval(expr, engine=None)

    try:
        return pandas.eval(expr, engine='numexpr')
    except ImportError:
        return pandas.eval(expr, engine='python')


def _squared_distance(s1, s2, a=0, b=1):

    expr = '((s2-s1)-a)**2*b**2'

    # PANDAS BUG?
    # return pandas.eval(expr, engine=None)
    try:
        return pandas.eval(expr, engine='numexpr')
    except ImportError:
        return pandas.eval(expr, engine='python')


def _haversine_distance(lat1, lng1, lat2, lng2):

    # degrees to radians conversion
    to_rad = 1 / 360 * np.pi * 2

    # numeric expression to use with numexpr package
    expr = '2*6371*arcsin(sqrt((sin((lat2*to_rad-lat1*to_rad)/2))**2+cos(lat1*to_rad)*cos(lat2*to_rad)*(sin((lng2*to_rad-lng1*to_rad)/2))**2))'

    # PANDAS BUG?
    # return pandas.eval(expr, engine=None)
    try:
        return pandas.eval(expr, engine='numexpr')
    except ImportError:
        return pandas.eval(expr, engine='python')

################################
#      STRING SIMILARITY       #
################################


def jaro_similarity(s1, s2):

    # Check jellyfish
    jellyfish = _import_jellyfish()

    conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

    def jaro_apply(x):

        try:
            return jellyfish.jaro_distance(x[0], x[1])
        except Exception:
            return np.nan

    return conc.apply(jaro_apply, axis=1)


def jarowinkler_similarity(s1, s2):

    # Check jellyfish
    jellyfish = _import_jellyfish()

    conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

    def jaro_winkler_apply(x):

        try:
            return jellyfish.jaro_winkler(x[0], x[1])
        except Exception:
            return np.nan

    return conc.apply(jaro_winkler_apply, axis=1)


def levenshtein_similarity(s1, s2):

    # Check jellyfish
    jellyfish = _import_jellyfish()

    conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

    def levenshtein_apply(x):

        try:
            return 1 - jellyfish.levenshtein_distance(x[0], x[1]) \
                / np.max([len(x[0]), len(x[1])])
        except Exception:
            return np.nan

    return conc.apply(levenshtein_apply, axis=1)


def damerau_levenshtein_similarity(s1, s2):

    # Check jellyfish
    jellyfish = _import_jellyfish()

    conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

    def damerau_levenshtein_apply(x):

        try:
            return 1 - jellyfish.damerau_levenshtein_distance(x[0], x[1]) \
                / np.max([len(x[0]), len(x[1])])
        except Exception:
            return np.nan

    return conc.apply(damerau_levenshtein_apply, axis=1)


def qgram_similarity(s1, s2, include_wb=True, ngram=(2, 2)):

    if len(s1) != len(s2):
        raise ValueError('Arrays or Series have to be same length.')

    # include word boundaries or not
    analyzer = 'char_wb' if include_wb is True else 'char'

    # The vectorizer
    vectorizer = CountVectorizer(
        analyzer=analyzer, strip_accents='unicode', ngram_range=ngram)

    data = s1.append(s2).fillna('')

    vec_fit = vectorizer.fit_transform(data)

    def _metric_sparse_euclidean(u, v):

        match_ngrams = u.minimum(v).sum(axis=1)
        total_ngrams = np.maximum(u.sum(axis=1), v.sum(axis=1))

        # division by zero is not possible in our case, but 0/0 is possible.
        # Numpy raises a warning in that case.
        return np.true_divide(match_ngrams, total_ngrams).A1

    return _metric_sparse_euclidean(vec_fit[:len(s1)], vec_fit[len(s1):])


def cosine_similarity(s1, s2, include_wb=True, ngram=(2, 2)):

    if len(s1) != len(s2):
        raise ValueError('Arrays or Series have to be same length.')

    # include word boundaries or not
    analyzer = 'char_wb' if include_wb is True else 'char'

    # The vectorizer
    vectorizer = CountVectorizer(
        analyzer=analyzer, strip_accents='unicode', ngram_range=ngram)

    data = s1.append(s2).fillna('')

    vec_fit = vectorizer.fit_transform(data)

    def _metric_sparse_cosine(u, v):

        a = np.sqrt(u.multiply(u).sum(axis=1))
        b = np.sqrt(v.multiply(v).sum(axis=1))

        ab = v.multiply(u).sum(axis=1)

        return np.divide(ab, np.multiply(a, b)).A1

    return _metric_sparse_cosine(vec_fit[:len(s1)], vec_fit[len(s1):])
