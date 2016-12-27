from __future__ import division

from functools import wraps

import pandas
import numpy

from recordlinkage.utils import IndexError, merge_dicts, max_number_of_pairs
from recordlinkage.algorithms.string import qgram_similarity


def check_index_names(func):
    # decorator to prevent index name conflicts. Used in functions like
    # blocking and SNI. Also useful for user defined functions.

    @wraps(func)
    def index_name_checker(df_a, df_b, *args, **kwargs):

        if (df_a.index.name is None or
                df_b.index.name is None or
                df_a.index.name is df_b.index.name or
                df_a.index.name in df_a.columns.tolist() or
                df_b.index.name in df_b.columns.tolist()):

            df_a_index_name = df_a.index.name
            df_b_index_name = df_b.index.name

            # temp update of the index name in each datatframe
            df_a.index.name = _save_label(list(df_a), s='index_x')
            df_b.index.name = _save_label(list(df_b), s='index_y')

            pairs = func(df_a, df_b, *args, **kwargs)

            # update the name of the index
            pairs.names = [df_a_index_name, df_b_index_name]

            df_a.index.name = df_a_index_name
            df_b.index.name = df_b_index_name

            return pairs

        else:

            return func(df_a, df_b, *args, **kwargs)

    return index_name_checker


def _save_label(labels, s='label'):

    i = 0

    while s in list(labels):

        i = i + 1
        s = s + '_' + str(i)

    return s

###########################
#       Algorithms        #
###########################


def _random_large_dedup(df_a, n, random_state=None):

    numpy.random.seed(random_state)

    n_max = max_number_of_pairs(df_a)

    if not isinstance(n, int) or n <= 0 or n > n_max:
        raise ValueError("n must be a integer satisfying 0<n<=%s" % n_max)

    full_index = _fullindex_dedup(df_a)
    sample = numpy.random.choice(
        numpy.arange(len(full_index)), n, replace=False
    )

    return full_index[sample]


def _random_large_link(df_a, df_b, n):

    n_max = max_number_of_pairs(df_a, df_b)

    if not isinstance(n, int) or n <= 0 or n > n_max:
        raise ValueError("n must be a integer satisfying 0<n<=%s" % n_max)

    full_index = _fullindex_link(df_a, df_b)
    sample = numpy.random.choice(
        numpy.arange(len(full_index)), n, replace=False
    )

    return full_index[sample]


def _random_small_link(df_a, df_b, n):

    n_max = max_number_of_pairs(df_a, df_b)

    if not isinstance(n, int) or n <= 0 or n > n_max:
        raise ValueError("n must be a integer satisfying 0<n<=%s" % n_max)

    levels = [df_a.index.values, df_b.index.values]
    names = [df_a.index.name, df_b.index.name]

    # Initialize pandas MultiIndex
    pairs = pandas.MultiIndex(levels=levels, labels=[[], []], names=names)

    # Run as long as the number of pairs is less than the requested number
    # of pairs n.
    while len(pairs) < n:

        # The number of pairs to sample (sample twice as much record pairs
        # because the duplicates are dropped).
        n_sample = (n - len(pairs)) * 2
        sample_a = numpy.random.randint(len(df_a), size=n_sample)
        sample_b = numpy.random.randint(len(df_b), size=n_sample)

        # Make a pandas MultiIndex of the sample above
        pairs_sample = pandas.MultiIndex(
            levels=levels, labels=[sample_a, sample_b], names=names
        )

        pairs = pairs.append(pairs_sample).drop_duplicates()

    return pairs[0:n]


def _eye(df_a, df_b):

    return pandas.MultiIndex.from_arrays(
        [df_a.index.values, df_b[0:len(df_a)].index.values],
        names=[df_a.index.name, df_b.index.name]
    )


def _fullindex_link(df_a, df_b):

    return pandas.MultiIndex.from_product(
        [df_a.index.values, df_b.index.values],
        names=[df_a.index.name, df_b.index.name]
    )


def _fullindex_dedup(df_a):

    return pandas.MultiIndex(
        levels=[df_a.index.values, df_a.index.values],
        labels=numpy.triu_indices(len(df_a.index), k=1),
        names=[df_a.index.name, df_a.index.name],
        verify_integrity=False
    )


@check_index_names
def _blockindex(df_a, df_b, on=None, left_on=None, right_on=None):

    # Index name conflicts do not occur. They are handled in the decorator.

    if on:
        left_on, right_on = on, on

    # Rows with missing values on the on attributes are dropped.
    data_left = df_a.dropna(axis=0, how='any', subset=numpy.unique(left_on))
    data_right = df_b.dropna(axis=0, how='any', subset=numpy.unique(right_on))

    # Join
    pairs = data_left.reset_index().merge(
        data_right.reset_index(),
        how='inner',
        left_on=left_on,
        right_on=right_on,
    ).set_index([df_a.index.name, df_b.index.name])

    return pairs.index


@check_index_names
def _sortedneighbourhood(
        df_a, df_b, column, window=3, sorting_key_values=None,
        block_on=[], block_left_on=[], block_right_on=[]):

    # Check if window is an odd number
    if not isinstance(window, int) or (window < 0) or not bool(window % 2):
        raise ValueError('The given window length is not a positive and odd integer.')

    block_on = [block_on] if type(block_on) != list else block_on
    block_left_on = [block_left_on] if type(
        block_left_on) != list else block_left_on
    block_right_on = [block_right_on] if type(
        block_right_on) != list else block_right_on

    block_left_on, block_right_on = [
        block_on, block_on] if block_on else ([], [])
    keys_left = [column] + block_left_on
    keys_right = [column] + block_right_on

    df_a = df_a[df_a[column].notnull()]  # df_a.dropna(inplace=True)
    df_b = df_b[df_b[column].notnull()]  # df_a.dropna(inplace=True)

    # sorting_key_values is the terminology in Data Matching [Christen, 2012]
    if sorting_key_values is None:

        # Combine the results
        sorting_key_values = numpy.sort(numpy.unique(
            numpy.concatenate([df_a[column].values, df_b[column].values])
        ))

    sorting_key_factors = numpy.arange(len(sorting_key_values))

    data_dict_A = {kl: df_a[kl] for kl in keys_left}
    data_dict_B = {kl: df_b[kl] for kl in keys_right}

    sorted_df_A = pandas.DataFrame(
        merge_dicts(
            data_dict_A,
            {column: df_a[column].replace(
                sorting_key_values, sorting_key_factors),
             df_a.index.name: df_a.index.values}))
    sorted_df_B = pandas.DataFrame(
        {column: df_b[column].replace(
            sorting_key_values, sorting_key_factors),
            df_b.index.name: df_b.index.values})

    pairs_concat = None

    # Internal window size
    _window = int((window - 1) / 2)

    for w in range(-_window, _window + 1):

        df = pandas.DataFrame(
            merge_dicts(
                data_dict_B,
                {
                    column: sorted_df_B[column] + w,
                    df_b.index.name: df_b.index.values
                }
            )
        )

        pairs = sorted_df_A.merge(
            df, left_on=keys_left, right_on=keys_right, how='inner'
        ).set_index(
            [df_a.index.name, df_b.index.name]
        )

        if pairs_concat is None:
            pairs_concat = pairs.index
        else:
            pairs_concat = pairs.index.append(pairs_concat)

    return pairs_concat


def _qgram(df_a, df_b, on=None, left_on=None, right_on=None, threshold=0.8):

    if on:
        left_on, right_on = on, on

        # Rows with missing values on the on attributes are dropped.

    fi = pandas.MultiIndex.from_product(
        [df_a[left_on].dropna(axis=0).index.values,
         df_b[right_on].dropna(axis=0).index.values],
        names=[df_a.index.name, df_b.index.name]
    )

    if len(fi) > 0:
        bool_index = qgram_similarity(
            df_a.loc[fi.get_level_values(0), left_on],
            df_b.loc[fi.get_level_values(1), right_on]
        ) >= threshold

        return fi[bool_index]

    else:
        return fi


class PairsCore(object):
    """

    Core class for making record pairs.

    """

    def __init__(self, df_a, df_b=None, chunks=None, verify_integrity=True):

        self.df_a = df_a
        self.df_b = df_b
        self.chunks = chunks

        self.deduplication = True if df_b is None else False

        if verify_integrity:

            if not self.df_a.index.is_unique:
                raise IndexError('index of DataFrame df_a is not unique.')

            if not self.deduplication and not self.df_b.index.is_unique:
                raise IndexError('index of DataFrame df_b is not unique.')

        self.n_pairs = 0

    @property
    def maximum_number_of_pairs(self):
        """ the maximum number of record pairs """

        if self.deduplication:
            return max_number_of_pairs(self.df_a)
        else:
            return max_number_of_pairs(self.df_a, self.df_b)

    # -- Index methods ------------------------------------------------------

    def index(self, index_func, *args, **kwargs):
        """

        Use a custom function to make record pairs of one or two dataframes.
        Each function should return a pandas.MultiIndex with record pairs.

        :param index_func: An indexing function
        :type index_func: function

        :return: MultiIndex
        :rtype: pandas.MultiIndex
        """

        # linking
        if not self.deduplication:

            pairs = index_func(self.df_a, self.df_b, *args, **kwargs)

        # deduplication
        else:

            if index_func.__name__.endswith('_dedup'):

                pairs = index_func(self.df_a, *args, **kwargs)

            else:

                pairs = index_func(self.df_a, self.df_a.copy(),
                                   *args, **kwargs)
                # Remove all double pairs!
                pairs = pairs[
                    pairs.get_level_values(0) < pairs.get_level_values(1)
                ]

        # If there are no chunks, then use the first item of the generator
        if self.chunks is None:
            return pairs

        # Use the chunks
        else:
            return self._iter_pairs(pairs)

    def _iter_pairs(self, pairs):

        if not isinstance(self.chunks, int):
            raise ValueError('argument chunks needs to be integer type')

        bins = numpy.arange(0, len(pairs), step=self.chunks)

        for b in bins:

            yield pairs[b:b + self.chunks]


class Pairs(PairsCore):
    """
    This class can be used to make record pairs. Multiple indexation methods
    can be used to make a smart selection of record pairs. Indexation methods
    included:

    - Full indexing
    - Blocking
    - Sorted Neighbourhood
    - Random indexing
    - Q-gram indexing

    For more information about indexing and methods to reduce the number of
    record pairs, see Christen 2012 [christen2012]_.

    :param df_a: The first dataframe.
    :param df_b: The second dataframe.
    :param chunks: The chunks to divide the result in.
    :param verify_integrity: Verify the integrity of the dataframes
            (default True).

    :type df_a: pandas.DataFrame
    :type df_b: pandas.DataFrame
    :type chunks: int
    :type verify_integrity: bool

    :returns: Candidate links
    :rtype: pandas.MultiIndex

    :var df_a: The first DataFrame.
    :var df_b: The second DataFrame.
    :var n_pairs: The number of candidate record pairs.
    :var reduction: The reduction ratio.

    :vartype df_a: pandas.DataFrame
    :vartype df_b: pandas.DataFrame
    :vartype n_pairs: int
    :vartype reduction: float

    Example:

    In the following example, the record pairs are made for two historical
    datasets with census data. The datasets are named ``census_data_1980``
    and ``census_data_1990``.

    .. code:: python

            >>> pcl = recordlinkage.Pairs(census_data_1980, census_data_1990)
            >>> pcl.block('first_name')

    .. seealso::

            .. [christen2012] Christen, 2012. Data Matching Concepts and
                    Techniques for Record Linkage, Entity Resolution, and
                    Duplicate Detection

    """

    def full(self):
        """
        Make an index with all possible record pairs. In case of linking two
        dataframes (A and B), the number of pairs is len(A)*len(B). In case of
        deduplicating a dataframe A, the number of pairs is
        len(A)*(len(A)-1)/2.

        :return: The index of the candidate record pairs
        :rtype: pandas.MultiIndex
        """
        if self.deduplication:
            return self.index(_fullindex_dedup)
        else:
            return self.index(_fullindex_link)

    def block(self, on=None, left_on=None, right_on=None):
        """
        Return all record pairs that agree on the passed attribute(s). This
        method is known as *blocking*

        :param on: A column name or a list of column names. These columns are
                used to block on.
        :param left_on: A column name or a list of column names of dataframe
                A. These columns are used to block on.
        :param right_on: A column name or a list of column names of dataframe
                B. These columns are used to block on.

        :type on: label
        :type left_on: label
        :type right_on: label

        :return: The index of the candidate record pairs
        :rtype: pandas.MultiIndex
        """
        return self.index(
            _blockindex,
            on=on, left_on=left_on, right_on=right_on
        )

    def sortedneighbourhood(
        self, on, window=3, sorting_key_values=None, block_on=[],
        block_left_on=[], block_right_on=[]
    ):
        """
        Create a Sorted Neighbourhood index.

        :param on: Specify the on to make a sorted index
        :param window: The width of the window, default is 3
        :param sorting_key_values: A list of sorting key values (optional).
        :param block_on: Additional columns to use standard blocking on
        :param block_left_on: Additional columns in the left dataframe to use
                standard blocking on
        :param block_right_on: Additional columns in the right dataframe to
                use standard blocking on

        :type on: label
        :type window: int
        :type sorting_key_values: array
        :type on: label
        :type left_on: label
        :type right_on: label

        :return: The index of the candidate record pairs
        :rtype: pandas.MultiIndex
        """

        # if self.batch and not sorting_key_values:

        #     if self.deduplication:
        #         sorting_key_values = numpy.sort(numpy.unique(self.df_a[on].values))
        #     else:
        #         sorting_key_values = numpy.sort(numpy.unique(
        #             numpy.concatenate([self.df_a[on].values, self.df_b[on].values])
        #         ))

        return self.index(
            _sortedneighbourhood,
            on, window=window, sorting_key_values=sorting_key_values,
            block_on=block_on, block_left_on=block_left_on,
            block_right_on=block_right_on)

    def random(self, n):
        """
        random(n)

        Make an index of randomly selected record pairs

        :param n: The number of record pairs to return. The integer
                n should satisfy 0 < n <= len(A)*len(B).
        :type n: int

        :return: The index of the candidate record pairs
        :rtype: pandas.MultiIndex
        """
        if self.deduplication:
            return self.index(_random_large_dedup, n)
        else:
            if n < 0.1 * self.maximum_number_of_pairs:
                return self.index(_random_large_link, n)
            else:
                return self.index(_random_small_link, n)

    def qgram(self, *args, **kwargs):
        """
        qgram(on=None, left_on=None, right_on=None, threshold=0.8)

        Use Q-gram string comparing metric to make an index.

        :param on: A column name or a list of column names. These columns are
                used to index on
        :param left_on: A column name or a list of column names of dataframe
                A. These columns are used to index on
        :param right_on: A column name or a list of column names of dataframe
                B. These columns are used to index on
        :param threshold: Record pairs with a similarity above the threshold
                are candidate record pairs. [Default 0.8]

        :type on: label
        :type left_on: label
        :type right_on: label
        :type threshold: float

        :return: The index of the candidate record pairs
        :rtype: pandas.MultiIndex
        """
        return self.index(_qgram, *args, **kwargs)

    def eye(self):
        # Only for internal use

        return self.index(_eye)

    @property
    def reduction(self):
        # """

        # The relative reduction of records pairs as the result of indexing.

        # :param n_pairs: The number of record pairs.

        # :type n_pairs: int

        # :return: Value between 0 and 1
        # :rtype: float
        # """

        return 1 - self.n_pairs / self.maximum_number_of_pairs
