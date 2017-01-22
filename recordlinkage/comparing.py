from __future__ import division
from __future__ import unicode_literals

import warnings
import multiprocessing as mp

import pandas
import numpy as np

from recordlinkage.types import (is_list_like,
                                 is_pandas_like,
                                 is_numpy_like)
from recordlinkage.utils import listify
from recordlinkage.algorithms.distance import (_1d_distance,
                                               _haversine_distance)
from recordlinkage.algorithms.numeric import (_step_sim,
                                              _linear_sim,
                                              _squared_sim,
                                              _exp_sim,
                                              _gauss_sim)
from recordlinkage.algorithms.string import (jaro_similarity,
                                             jarowinkler_similarity,
                                             levenshtein_similarity,
                                             damerau_levenshtein_similarity,
                                             qgram_similarity,
                                             cosine_similarity)


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


class CompareCore(object):
    """Core class to compare record pairs.

    Core class to compare the attributes of candidate record pairs. This class
    has no algorithms to compare record pairs except from the basic
    ``compare`` method.

    Parameters
    ----------
    pairs : pandas.MultiIndex
        A MultiIndex of candidate record pairs.
    df_a : pandas.DataFrame
        The first dataframe.
    df_b : pandas.DataFrame
        The second dataframe.
    low_memory : bool
        Reduce the amount of memory used by the Compare class. Default False.
    block_size : int
        The maximum size of data blocks. Default 1,000,000.

    Attributes
    ----------
    pairs : pandas.MultiIndex
        The candidate record pairs.
    df_a : pandas.DataFrame
        The first DataFrame.
    df_b : pandas.DataFrame
        The second DataFrame.
    vectors : pandas.DataFrame
        The DataFrame with comparison data.
    """

    def __init__(self, pairs, df_a=None, df_b=None, low_memory=False,
                 block_size=1000000, njobs=1, **kwargs):

        # The dataframes
        self.df_a = df_a
        self.df_b = df_b if df_b is not None else df_a

        # The candidate record pairs
        self.pairs = pairs

        self.low_memory = low_memory
        self.block_size = block_size
        self.njobs = njobs

        self._df_a_indexed = None
        self._df_b_indexed = None

        # The resulting data
        self.vectors = pandas.DataFrame(index=pairs)

        if 'batch_compare' in kwargs:
            warnings.warn(
                "Compare.run() is deprecated." +
                "Use low_memory=True for fast comparisons.",
                DeprecationWarning
            )

    def _loc2(self, frame, multi_index, level_i):
        """Indexing algorithm for MultiIndex on one level

        This internal function is a modification ``.loc`` indexing method in
        pandas. With this function, one level of a MultiIndex is used to index
        a dataframe with. The multiindex is split into smaller pieces to
        improve performance.

        Arguments
        ---------
        frame : pandas.DataFrame
            The datafrme to select records from.
        multi_index : pandas.MultiIndex
            A pandas multiindex were one fo the levels is used to sample the
            dataframe with.
        level_i : int, str
            The level of the multiIndex to index on.

        """

        if not isinstance(self.block_size, int):
            raise TypeError("block_size must be of type int")

        if self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")

        # Number of blocks
        len_mi = len(multi_index)

        if len_mi > self.block_size:  # More than 1 block

            # Collect parts to concat in the end.
            parts = []
            start_slice = 0

            while start_slice <= len_mi:

                # Set the end of the slice
                end_slice = start_slice + self.block_size

                # Slice the MultiIndex
                m_ind = multi_index[start_slice:end_slice]
                ind = m_ind.get_level_values(level_i)

                # The actual indexing
                index_result = frame.loc[ind]

                # Append to list named parts
                parts.append(index_result)

                # Set new slice start
                start_slice = end_slice

            data = pandas.concat(parts, axis=0, copy=False)

        else:  # Only one block
            data = frame.loc[multi_index.get_level_values(level_i)]

        # Add MultiIndex (Is this step important?)
        data.index = multi_index

        return data

    def compare(self, comp_func, labels_a, labels_b, *args, **kwargs):
        """Compare two records.

        Core method to compare record pairs. This method takes a function and
        data from both records in the record pair. The data is compared with
        the compare function. The built-in methods also use this function.

        Example
        -------

        >>> comp = recordlinkage.Compare(PAIRS, DATAFRAME1, DATAFRAME2)
        >>> comp.exact('first_name', 'name')

        >>> # same as
        >>> comp.compare(recordlinkage._compare_exact, 'first_name', 'name')

        Parameters
        ----------
        comp_func : function
            A comparison function. This function can be a built-in function or
            a user defined comparison function.
        labels_a : label, pandas.Series, pandas.DataFrame
            The labels, Series or DataFrame to compare.
        labels_b : label, pandas.Series, pandas.DataFrame
            The labels, Series or DataFrame to compare.
        name : label
            The name of the feature and the name of the column.
        store : bool, default True
            Store the result in the dataframe.

        Returns
        -------
        pandas.Series
            A pandas series with the result of comparing each record pair.

        """

        if len(self.pairs) == 0:
            raise ValueError(
                "need at least one record pair"
            )

        # the name and store arguments
        name = kwargs.pop('name', None)
        store = kwargs.pop('store', True)

        labels_a = listify(labels_a)
        labels_b = listify(labels_b)

        data_a = []

        for label_a in labels_a:

            # the label is a numpy or pandas object
            if is_numpy_like(label_a) or is_pandas_like(label_a):
                data_a.append(label_a)

            # check requested labels (for better error messages)
            elif label_a not in self.df_a.columns:
                raise KeyError("label '%s' is not found in the first"
                               "dataframe" % label_a)

            else:

                if self.low_memory:

                    df_a_label = self._loc2(self.df_a[label_a], self.pairs, 0)
                    data_a.append(df_a_label)

                # not low memory
                else:
                    if self._df_a_indexed is None:

                        self._df_a_indexed = self._loc2(
                            self.df_a, self.pairs, 0)

                    data_a.append(self._df_a_indexed[label_a])

        data_a = tuple(data_a)

        data_b = []

        for label_b in labels_b:

            # the label is a numpy or pandas object
            if is_numpy_like(label_b) or is_pandas_like(label_b):
                data_b.append(label_b)

            # check requested labels (for better error messages)
            elif label_b not in self.df_b.columns:

                raise KeyError("label '%s' is not found in the second"
                               "dataframe" % label_b)

            else:

                if self.low_memory:

                    df_b_label = self._loc2(self.df_b[label_b], self.pairs, 1)
                    data_b.append(df_b_label)

                # not low memory
                else:
                    if self._df_b_indexed is None:

                        self._df_b_indexed = self._loc2(
                            self.df_b, self.pairs, 1)

                    data_b.append(self._df_b_indexed[label_b])

        data_b = tuple(data_b)


        if self.njobs > 1:

            jobs = []

            chunk_size = np.ceil(self.njobs / len(self.pairs))

            # each job
            for i in range(0, self.njobs):

                # The data arguments
                args_a = tuple(df_a_indexed.loc[i*chunk_size:(i+1)*chunk_size, da] for da in labels_a)
                args_b = tuple(df_b_indexed.loc[i*chunk_size:(i+1)*chunk_size, db] for db in labels_b)

                p = mp.Process(target=comp_func, args=args_a + args_b + args, kwargs=kwargs)
                jobs.append(p)

            for proc in jobs:

                # Start the process
                p.start()
                proc.join()

            # merge parts
            c = pandas.concat(jobs, axis=0, copy=False)

        else:

            # # The data arguments
            # args_a = tuple(df_a_indexed.loc[:, da] for da in labels_a)
            # args_b = tuple(df_b_indexed.loc[:, db] for db in labels_b)

            # Compute the comparison
            c = comp_func(*tuple(data_a + data_b + args), **kwargs)

        # if a pandas series is returned, overwrite the index. The
        # returned index can be different than the MultiIndex passed to
        # the compare function.
        if isinstance(c, pandas.Series):
            c.index = self.vectors.index

        # append column to Compare.vectors
        if store:
            name_or_id = name if name else len(self.vectors.columns)
            self.vectors[name_or_id] = c

        return self.vectors[name_or_id].rename(name)

    def run(self):
        """Run in a batch

        This method is decrecated. Use the comparing.Compare(...,
        low_memory=False) for better performance.

        """

        raise AttributeError("method run() is deprecated")

    def clear_memory(self):
        """Clear memory.

        Clear some memory when low_memory was set to True.
        """

        self._df_a_indexed = None
        self._df_b_indexed = None


class Compare(CompareCore):
    """Compare record pairs with the tools in this class.

    Class to compare the attributes of candidate record pairs. The ``Compare``
    class has several methods to compare data such as string similarity
    measures, numeric metrics and exact comparison methods.

    Parameters
    ----------
    pairs : pandas.MultiIndex
        A MultiIndex of candidate record pairs.
    df_a : pandas.DataFrame
        The first dataframe.
    df_b : pandas.DataFrame
        The second dataframe.
    low_memory : bool
        Reduce the amount of memory used by the Compare class. Default False.
    block_size : int
        The maximum size of data blocks. Default 1,000,000.

    Attributes
    ----------
    pairs : pandas.MultiIndex
        The candidate record pairs.
    df_a : pandas.DataFrame
        The first DataFrame.
    df_b : pandas.DataFrame
        The second DataFrame.
    vectors : pandas.DataFrame
        The DataFrame with comparison data.

    Examples
    --------
    In the following example, the record pairs of two historical datasets with
    census data are compared. The datasets are named ``census_data_1980`` and
    ``census_data_1990``. The ``candidate_pairs`` are the record pairs to
    compare. The record pairs are compared on the first name, last name, sex,
    date of birth, address, place, and income.

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

    def exact(self, s1, s2, *args, **kwargs):
        """
        exact(s1, s2, agree_value=1, disagree_value=0, missing_value=0, name=None, store=True)

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
        name : label
            The name of the feature and the name of the column.
        store : bool
            Store the result in the dataframe. Default True

        Returns
        -------
        pandas.Series
            A pandas series with the result of comparing each record pair.

        """

        return self.compare(_compare_exact, s1, s2, *args, **kwargs)

    def string(self, s1, s2, method='levenshtein', threshold=None, *args, **kwargs):
        """
        string(s1, s2, method='levenshtein', threshold=None, missing_value=0, name=None, store=True)

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
            'cosine']. Default: 'levenshtein'
        threshold : float, tuple of floats
            A threshold value. All approximate string comparisons higher or
            equal than this threshold are 1. Otherwise 0.
        missing_value : numpy.dtype
            The value for a comparison with a missing value. Default 0.
        name : label
            The name of the feature and the name of the column.
        store : bool
            Store the result in the dataframe. Default True

        Returns
        -------
        pandas.Series
            A pandas series with similarity values. Values equal or between 0
            and 1.

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
        name : label
            The name of the feature and the name of the column.
        store : bool
            Store the result in the dataframe. Default True

        Returns
        -------
        pandas.Series
            A pandas series with the result of comparing each record pair.

        Note
        ----
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
        name : label
            The name of the feature and the name of the column.
        store : bool
            Store the result in the dataframe. Default True.

        Returns
        -------
        pandas.Series
            A pandas series with the result of comparing each record pair.

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
        name : label
            The name of the feature and the name of the column.
        store : bool
            Store the result in the dataframe. Default True.

        Returns
        -------
        pandas.Series
            A pandas series with the result of comparing each record pair.

        """

        return self.compare(
            _compare_dates, s1, s2,
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
