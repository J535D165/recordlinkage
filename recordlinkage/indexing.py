from __future__ import division

from functools import wraps

import pandas
import numpy

from recordlinkage.utils import IndexError, merge_dicts, split_or_pass
from recordlinkage.comparing import qgram_similarity


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

            rightname = 'index_y'
            while rightname in df_b.columns.tolist():
                rightname = rightname + '_'

            leftname = 'index_x'
            while leftname in df_a.columns.tolist():
                leftname = leftname + '_'

            df_a.index.name = leftname
            df_b.index.name = rightname

            pairs = func(df_a, df_b, *args, **kwargs)

            pairs.names = [df_a_index_name, df_b_index_name]

            return pairs

        else:

            return func(df_a, df_b, *args, **kwargs)

    return index_name_checker

###########################
#       Algorithms        #
###########################


def _randomindex(df_a, df_b, n_pairs):

    if type(n_pairs) is not int or n_pairs <= 0:
        raise ValueError("n_pairs must be an positive integer")

    if n_pairs < 0.25 * len(df_a) * len(df_b):

        n_count = 0

        while n_count < n_pairs:

            random_index_a = numpy.random.choice(
                df_a.index.values, n_pairs - n_count)
            random_index_b = numpy.random.choice(
                df_b.index.values, n_pairs - n_count)

            sub_ind = pandas.MultiIndex.from_arrays(
                [random_index_a, random_index_b],
                names=[df_a.index.name, df_b.index.name]
            )

            ind = sub_ind if n_count == 0 else ind.append(sub_ind)
            ind = ind.drop_duplicates()

            n_count = len(ind)

        return ind

    else:

        full_index = _fullindex(df_a, df_b)

        return full_index[
            numpy.random.choice(
                numpy.arange(len(full_index)), n_pairs, replace=False)
        ]


def _eye(df_a, df_b):

    return pandas.MultiIndex.from_arrays(
        [df_a.index.values, df_b.index.values],
        names=[df_a.index.name, df_b.index.name]
    )


def _fullindex(df_a, df_b):

    return pandas.MultiIndex.from_product(
        [df_a.index.values, df_b.index.values],
        names=[df_a.index.name, df_b.index.name]
    )


@check_index_names
def _blockindex(df_a, df_b, on=None, left_on=None, right_on=None):

    # Index name conflicts do not occur. They are handled in the decorator.

    if on:
        left_on, right_on = on, on

    # Rows with missing values on the on attributes are dropped.
    data_left = df_a[left_on].dropna(axis=0)
    data_right = df_b[right_on].dropna(axis=0)

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

    bool_index = (qgram_similarity(df_a.loc[fi.get_level_values(
        0), left_on], df_b.loc[fi.get_level_values(1), right_on]) >= threshold)

    return fi[bool_index]


class Pairs(object):
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

    :type df_a: pandas.DataFrame
    :type df_b: pandas.DataFrame

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

    def __init__(self, df_a, df_b=None, chunks=None):

        self.df_a = df_a
        self.chunks = chunks

        # Linking two datasets
        if df_b is not None:

            self.df_b = df_b
            self.deduplication = False

            if not self.df_a.index.is_unique or not self.df_b.index.is_unique:
                raise IndexError('DataFrame index is not unique.')

        # Deduplication of one dataset
        else:
            self.deduplication = True

            # if self.df_a.index.name == None:
            #   raise IndexError('DataFrame has no index name.')

            if not self.df_a.index.is_unique:
                raise IndexError('DataFrame index is not unique.')

        self.n_pairs = 0

        self._index_factors = None

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

        # If there are no chunks, then use the first item of the generator
        if self.chunks is None or self.chunks == (None, None):

            d = next(self._iterindex(index_func, *args, **kwargs))

            return d

        # Use the chunks
        else:
            return self._iterindex(index_func, *args, **kwargs)

    def full(self, *args, **kwargs):
        """
        full()

        Make an index with all possible record pairs. In case of linking two
        dataframes (A and B), the number of pairs is len(A)*len(B). In case of
        deduplicating a dataframe A, the number of pairs is
        len(A)*(len(A)-1)/2.

        :return: The index of the candidate record pairs
        :rtype: pandas.MultiIndex
        """
        return self.index(_fullindex, *args, **kwargs)

    def block(self, *args, **kwargs):
        """
        block(on=None, left_on=None, right_on=None)

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
        return self.index(_blockindex, *args, **kwargs)

    def sortedneighbourhood(self, *args, **kwargs):
        """
        sortedneighbourhood(on, window=3, sorting_key_values=None, block_on=[], block_left_on=[], block_right_on=[])

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
        return self.index(_sortedneighbourhood, *args, **kwargs)

    def random(self, *args, **kwargs):
        """
        random(n_pairs)

        Make an index of randomly selected record pairs

        :param n_pairs: The number of record pairs to return. The integer
                        n_pairs should satisfy 0 < n_pairs <= len(A)*len(B).
        :type n_pairs: int

        :return: The index of the candidate record pairs
        :rtype: pandas.MultiIndex
        """
        return self.index(_randomindex, *args, **kwargs)

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

    def eye(self, *args, **kwargs):
        # Only for internal use

        return self.index(_eye, *args, **kwargs)

    # -- Iterative index methods ----------------------------------------------

    def _iterindex(self, index_func, *args, **kwargs):
        """

        Iterative function that returns records pairs based on a user-defined
        indexing function. The number of iterations can be adjusted to prevent
        memory problems.

        :param index_func: A user defined indexing function.
        :param chunks: The number of records used to split up the data. First
            arugment of the tuple is the number of records in DataFrame 1 and
            the second argument is the number of records in DataFrame 2 (or 1
            in case of deduplication)

        :type index_func: function
        :type chunks: tuple, int

        :return: The index of the candidate record pairs
        :rtype: pandas.MultiIndex
        """

        blocks = self.make_grid()

        # Reset the number of pairs counter
        self.n_pairs = 0

        for bl0, bl1, bl2, bl3 in blocks:

            # If not deduplication, make pairs of records with one record from
            # the first dataset and one of the second dataset
            if not self.deduplication:

                pairs = index_func(
                    self.df_a[bl0:bl2],
                    self.df_b[bl1:bl3],
                    *args, **kwargs
                )

            # If deduplication, remove the record pairs that are already
            # included. For example: (a1, a1), (a1, a2), (a2, a1), (a2, a2)
            # results in (a1, a2) or (a2, a1)
            elif self.deduplication:

                pairs = index_func(
                    self.df_a[bl0:bl2],
                    self.df_a[bl1:bl3],
                    *args, **kwargs
                )

                # Remove all double pairs!
                pairs = pairs[pairs.get_level_values(
                    0) < pairs.get_level_values(1)]

            self.n_pairs = len(pairs)

            yield pairs

    # -- Tools for indexing ----------------------------------------------

    def make_grid(self):

        # Generate blocks to evaluate
        len_block_a, len_block_b = split_or_pass(self.chunks)

        # If len_block is None, then use the length of the DataFrame
        len_block_a = len_block_a if len_block_a else len(self.df_a)
        len_a = len(self.df_a)

        if not self.deduplication:
            len_block_b = len_block_b if len_block_b else len(self.df_b)
            len_b = len(self.df_b)
        else:
            len_block_b = len_block_b if len_block_b else len(self.df_a)
            len_b = len(self.df_a)

        return [(a, b, a + len_block_a, b + len_block_b)
                for a in numpy.arange(0, len_a, len_block_a)
                for b in numpy.arange(0, len_b, len_block_b)]

    @property
    def reduction(self):
        # """

        # The relative reduction of records pairs as the result of indexing.

        # :param n_pairs: The number of record pairs.

        # :type n_pairs: int

        # :return: Value between 0 and 1
        # :rtype: float
        # """

        if self.deduplication:
            max_pairs = (len(self.df_a) * (len(self.df_b) - 1)) / 2
        else:
            max_pairs = len(self.df_a) * len(self.df_b)

        return 1 - self.n_pairs / max_pairs
