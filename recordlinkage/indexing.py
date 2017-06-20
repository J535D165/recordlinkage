from __future__ import division

import warnings
from functools import wraps

import pandas
import numpy

from recordlinkage.base import BaseIndexator
from recordlinkage.utils import IndexError
from recordlinkage.utils import merge_dicts
from recordlinkage.utils import max_number_of_pairs
from recordlinkage.algorithms.string import qgram_similarity
from recordlinkage.utils import listify
from recordlinkage.measures import reduction_ratio
from recordlinkage.measures import max_pairs
from recordlinkage.algorithms.indexing import \
    random_pairs_with_replacement
from recordlinkage.algorithms.indexing import \
    random_pairs_without_replacement_small_frames
from recordlinkage.algorithms.indexing import \
    random_pairs_without_replacement_large_frames

from recordlinkage import rl_logging as logging



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
        raise ValueError(
            'The given window length is not a positive and odd integer.')

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

    sorted_index = pandas.Series(
        index=sorting_key_values, data=sorting_key_factors)
    sorted_df_A = pandas.DataFrame(
        merge_dicts(
            data_dict_A,
            {column: df_a[column].map(sorted_index),
             df_a.index.name: df_a.index.values}))
    sorted_df_B = pandas.DataFrame(
        {column: df_b[column].map(sorted_index),
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
    """Core class for making record pairs.

    """

    def __init__(self, df_a, df_b=None, chunks=None, verify_integrity=True):

        warnings.warn(
            "indexing api changed, see the documentation for the new format",
            DeprecationWarning
        )

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
        """Main function to make an index of record pairs.

        Use a custom function to make record pairs of one or two dataframes.
        Each function should return a pandas.MultiIndex with record pairs.

        Parameters
        ----------
        index_func: function
            An indexing function

        Returns
        -------
        pandas.MultiIndex
            A pandas multiindex with record pairs. The first position in the
            index is the index of a record in dataframe A and the second
            position is the index of a record in dataframe B.

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
    """Make record pairs.

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

    Parameters
    ----------
    df_a : pandas.DataFrame
        The first dataframe.
    df_b : pandas.DataFrame
        The second dataframe.
    chunks : int
        The chunks to divide the result in.
    verify_integrity : bool
        Verify the integrity of the dataframes
            (default True).

    Attributes
    ----------
    df_a : pandas.DataFrame
        The first DataFrame.
    df_b : pandas.DataFrame
        The second DataFrame.
    n_pairs : int
        The number of candidate record pairs.
    reduction : float
        The reduction ratio.

    Examples
    --------
    In the following example, the record pairs are made for two historical
    datasets with census data. The datasets are named ``census_data_1980``
    and ``census_data_1990``.

    >>> pcl = recordlinkage.Pairs(census_data_1980, census_data_1990)
    >>> pcl.block('first_name')

    References
    ----------
    .. [christen2012] Christen, 2012. Data Matching Concepts and
            Techniques for Record Linkage, Entity Resolution, and
            Duplicate Detection

    """

    def full(self):
        """Make a full index (all possible pairs).

        Make an index with all possible record pairs. In case of linking two
        dataframes (A and B), the number of pairs is len(A)*len(B). In case of
        deduplicating a dataframe A, the number of pairs is
        len(A)*(len(A)-1)/2.

        Returns
        -------
        pandas.MultiIndex
            The index of the candidate record pairs

        """
        if self.deduplication:
            return self.index(_fullindex_dedup)
        else:
            return self.index(_fullindex_link)

    def block(self, on=None, left_on=None, right_on=None):
        """Make a block index.

        Return all record pairs that agree on the passed attribute(s). This
        method is known as *blocking*

        Parameters
        ----------
        on : label
            A column name or a list of column names. These columns are used to
            block on.
        left_on : label
            A column name or a list of column names of dataframe A. These
            columns are used to block on.
        right_on : label
            A column name or a list of column names of dataframe B. These
            columns are used to block on.

        Returns
        -------
        pandas.MultiIndex
            The index of the candidate record pairs

        """
        return self.index(
            _blockindex,
            on=on, left_on=left_on, right_on=right_on
        )

    def sortedneighbourhood(
        self, on, window=3, sorting_key_values=None, block_on=[],
        block_left_on=[], block_right_on=[]
    ):
        """Make a Sorted Neighbourhood index.

        Parameters
        ----------
        on: label
            Specify the on to make a sorted index
        window: int
            The width of the window, default is 3
        sorting_key_values: array
            A list of sorting key values (optional).
        block_on: label
            Additional columns to use standard blocking on
        block_left_on: label
            Additional columns of the left dataframe to use standard blocking
            on.
        block_right_on: label
            Additional columns of the right dataframe to use standard
            blocking on

        Returns
        -------
        pandas.MultiIndex
            The index of the candidate record pairs

        """

        return self.index(
            _sortedneighbourhood,
            on, window=window, sorting_key_values=sorting_key_values,
            block_on=block_on, block_left_on=block_left_on,
            block_right_on=block_right_on)

    def random(self, n):
        """Make an index of randomly selected record pairs.

        Parameters
        ----------
        n : int
            The number of record pairs to return. The integer n should satisfy
            0 < n <= len(A)*len(B).

        Returns
        -------
        pandas.MultiIndex
            The index of the candidate record pairs

        """

        if not isinstance(n, int):
            raise ValueError("an integer is required")

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

        Parameters
        ----------
        on : label
            A column name or a list of column names. These columns are used to
            index on.
        left_on : label
            A column name or a list of column names of dataframe A. These
            columns are used to index on.
        right_on : label
            A column name or a list of column names of dataframe B. These
            columns are used to index on.
        threshold : float
            Record pairs with a similarity above the threshold are candidate
            record pairs. [Default 0.8]

        Returns
        -------
        pandas.MultiIndex
            The index of the candidate record pairs

        """
        return self.index(_qgram, *args, **kwargs)

    def eye(self):
        # Only for internal use

        return self.index(_eye)

    @property
    def reduction(self):

        return 1 - self.n_pairs / self.maximum_number_of_pairs


##############################################################################
#
# This section contains the indexing classes for the new indexing API.
#


class FullIndex(BaseIndexator):
    """FullIndex()
    Class to generate a 'full' index.

    A full index is an index with all possible combinations of record pairs.
    In case of linking, this indexation method generates the cartesian product
    of both DataFrame's. In case of deduplicating DataFrame A, this indexation
    method are the pairs defined by the upper triangular matrix of the A x A.

    Note
    ----
    This indexation method can be slow for large DataFrame's. The number of
    comparisons scales quadratic.
    Also, not all classifiers work well with large numbers of record pairs
    were most of the pairs are distinct.

    """

    def __init__(self, *args, **kwargs):
        super(FullIndex, self).__init__(*args, **kwargs)

    def _link_index(self, df_a, df_b):

        n_max = max_pairs((df_a, df_b))

        if n_max > 1e7:
            logging.warn(
                "The number of record pairs is large. Consider a different "
                "indexation algorithm for better performance. "
            )

        return pandas.MultiIndex.from_product(
            [df_a.index.values, df_b.index.values],
            names=[df_a.index.name, df_b.index.name]
        )

    def _dedup_index(self, df_a):

        n_max = max_pairs((df_a))

        if n_max > 1e7:
            logging.warn(
                "The number of record pairs is large. Consider a different "
                "indexation algorithm for better performance. "
            )

        levels = [df_a.index.values, df_a.index.values]
        labels = numpy.triu_indices(len(df_a.index), k=1)
        names = [df_a.index.name, df_a.index.name]

        return pandas.MultiIndex(
            levels=levels,
            labels=labels,
            names=names,
            verify_integrity=False
        )


class BlockIndex(BaseIndexator):
    """BlockIndex(on=None, left_on=None, right_on=None)
    Make candidate record pairs that agree on one or more variables.

    Returns all record pairs that agree on the given variable(s). This method
    is known as *blocking*. Blocking is an effective way to make a subset of
    the record space (A * B).

    Parameters
    ----------
    on : label, optional
        A column name or a list of column names. These columns are used to
        block on.
    left_on : label, optional
        A column name or a list of column names of dataframe A. These
        columns are used to block on. This argument is ignored when argument
        'on' is given.
    right_on : label, optional
        A column name or a list of column names of dataframe B. These
        columns are used to block on. This argument is ignored when argument
        'on' is given.

    Examples
    --------
    In the following example, the record pairs are made for two historical
    datasets with census data. The datasets are named ``census_data_1980``
    and ``census_data_1990``.

    >>> indexer = recordlinkage.BlockIndex(on='first_name')
    >>> indexer.index(census_data_1980, census_data_1990)

    """

    def __init__(self, on=None, left_on=None, right_on=None,
                 *args, **kwargs):
        super(BlockIndex, self).__init__(*args, **kwargs)

        # variables to block on
        self.on = on
        self.left_on = left_on
        self.right_on = right_on

    def _link_index(self, df_a, df_b):
        # Index name conflicts do not occur. They are handled in the
        # decorator.

        left_on = listify(self.left_on)
        right_on = listify(self.right_on)

        if self.on:
            left_on, right_on = listify(self.on), listify(self.on)

        if not left_on or not right_on:
            raise ValueError("no column labels given")

        if len(left_on) != len(right_on):
            raise ValueError(
                "length of left and right keys needs to be the same"
            )

        blocking_keys = ["blocking_key_%d" % i for i, v in enumerate(left_on)]

        # make a dataset for the data on the left
        data_left = df_a[left_on].dropna(axis=0, how='any', inplace=False)
        data_left.columns = blocking_keys
        data_left['index_x'] = data_left.index

        # make a dataset for the data on the right
        data_right = df_b[right_on].dropna(axis=0, how='any', inplace=False)
        data_right.columns = blocking_keys
        data_right['index_y'] = data_right.index

        # merge the dataframes
        pairs = data_left.merge(
            data_right, how='inner', on=blocking_keys
        ).set_index(['index_x', 'index_y'])

        return pairs.index.rename([df_a.index.name, df_b.index.name])


class SortedNeighbourhoodIndex(BaseIndexator):
    """SortedNeighbourhoodIndex(on=None, left_on=None, right_on=None, window=3, sorting_key_values=None, block_on=[], block_left_on=[], block_right_on=[])
    Make candidate record pairs with the SortedNeighbourhood algorithm.

    This algorithm returns record pairs that agree on the sorting key, but
    also records pairs in their neighbourhood. A large window size results in
    more record pairs. A window size of 1 returns the blocking index.

    The Sorted Neighbourhood Index method is a great method when there is
    relatively large amount of spelling mistakes. Blocking will fail in that
    situation because it excludes to many records on minor spelling mistakes.

    Parameters
    ----------
    on: label
        Specify the on to make a sorted index
    window: int, optional
        The width of the window, default is 3
    sorting_key_values: array, optional
        A list of sorting key values (optional).
    block_on: label
        Additional columns to use standard blocking on
    block_left_on: label
        Additional columns of the left dataframe to use standard blocking
        on.
    block_right_on: label
        Additional columns of the right dataframe to use standard
        blocking on


    Examples
    --------
    In the following example, the record pairs are made for two historical
    datasets with census data. The datasets are named ``census_data_1980``
    and ``census_data_1990``.

    >>> indexer = recordlinkage.SortedNeighbourhoodIndex(on='first_name', w=9)
    >>> indexer.index(census_data_1980, census_data_1990)

    """

    def __init__(self, on=None, left_on=None, right_on=None, window=3,
                 sorting_key_values=None, block_on=[], block_left_on=[],
                 block_right_on=[], *args, **kwargs):
        super(SortedNeighbourhoodIndex, self).__init__(*args, **kwargs)

        # variables to block on
        self.on = on
        self.left_on = left_on
        self.right_on = right_on
        self.window = window
        self.sorting_key_values = sorting_key_values
        self.block_on = block_on
        self.block_left_on = block_left_on
        self.block_right_on = block_right_on

    def _get_sorting_key_values(self, array1, array2):
        """return the sorting key values as a series"""

        concat_arrays = numpy.concatenate([array1, array2])
        unique_values = numpy.unique(concat_arrays)

        return numpy.sort(unique_values)

    def _link_index(self, df_a, df_b):

        # Index name conflicts do not occur. They are handled in the
        # decorator.

        left_on = listify(self.left_on)
        right_on = listify(self.right_on)

        if self.on:
            left_on = listify(self.on)
            right_on = listify(self.on)

        if not left_on or not right_on:
            raise ValueError("no column labels given")

        if len(left_on) != len(right_on):
            raise ValueError(
                "length of left and right keys needs to be the same"
            )

        window = self.window

        # Check if window is an odd number
        if not isinstance(window, int) or (window < 0) or not bool(window % 2):
            raise ValueError(
                'window is not a positive and odd integer')

        # # sorting key is single column
        # if isinstance(self.on, (tuple, list, dict)):
        #     raise ValueError(
        #         "sorting key is not a label")

        # make blocking keys correct

        block_left_on = listify(self.block_left_on)
        block_right_on = listify(self.block_right_on)

        if self.block_on:
            block_left_on = listify(self.block_on)
            block_right_on = listify(self.block_on)

        # drop missing values and columns without relevant information
        data_left = df_a[listify(left_on) + block_left_on].dropna(
            axis=0, how='any', inplace=False
        )
        data_left.columns = ['sorting_key'] + ["blocking_key_%d" %
                                               i for i, v in enumerate(block_left_on)]
        data_left['index_x'] = data_left.index

        data_right = df_b[listify(right_on) + block_right_on].dropna(
            axis=0, how='any', inplace=False
        )
        data_right.columns = ['sorting_key'] + ["blocking_key_%d" %
                                                i for i, v in enumerate(block_right_on)]
        data_right['index_y'] = data_right.index

        # sorting_key_values is the terminology in Data Matching [Christen,
        # 2012]
        if self.sorting_key_values is None:

            self.sorting_key_values = self._get_sorting_key_values(
                data_left['sorting_key'].values,
                data_right['sorting_key'].values
            )

        sorting_key_factors = pandas.Series(
            numpy.arange(len(self.sorting_key_values)),
            index=self.sorting_key_values)

        data_left['sorting_key'] = data_left[
            'sorting_key'].map(sorting_key_factors)
        data_right['sorting_key'] = data_right[
            'sorting_key'].map(sorting_key_factors)

        # Internal window size
        _window = int((window - 1) / 2)

        def merge_lagged(x, y, w):
            """Merge two dataframes with a lag on in the sorting key."""

            y = y.copy()
            y['sorting_key'] = y['sorting_key'] + w

            return x.merge(y, how='inner')

        pairs_concat = [merge_lagged(data_left, data_right, w)
                        for w in range(-_window, _window + 1)]

        pairs = pandas.concat(pairs_concat, axis=0).set_index(
            ['index_x', 'index_y']
        ).index.rename([df_a.index.name, df_b.index.name])

        return pairs


class RandomIndex(BaseIndexator):
    """RandomIndex(n, replace=True, random_state=None)
    Class to generate random pairs of records.

    This class returns random pairs of records with or without replacement.
    Use the random_state parameter to seed the algorithm and reproduce
    results. This way to make record pairs is useful for the training of
    unsupervised learning models for record linkage.

    Parameters
    ----------
    n : int
        The number of record pairs to return. In case replace=False, the
        integer n should be bounded by 0 < n <= n_max where n_max is the
        maximum number of pairs possible.
    replace : bool, optional
        Whether the sample of record pairs is with or without replacement.
        Default: True
    random_state : int or numpy.random.RandomState, optional
        Seed for the random number generator (if int), or numpy RandomState
        object.

    """

    def __init__(self, n, replace=True, random_state=None, *args, **kwargs):
        super(RandomIndex, self).__init__(*args, **kwargs)

        self.n = n
        self.replace = replace
        self.random_state = random_state

    def _link_index(self, df_a, df_b):

        shape = (len(df_a), len(df_b))
        n_max = max_pairs(shape)

        if not isinstance(self.n, int):
            raise ValueError('n must be an integer')

        # with replacement
        if self.replace:

            if n_max == 0:
                raise ValueError(
                    "one of the dataframes is empty")

            pairs = random_pairs_with_replacement(
                self.n, shape, self.random_state)

        # without replacement
        else:

            if self.n <= 0 or self.n > n_max:
                raise ValueError(
                    "n must be a integer satisfying 0<n<=%s" % n_max)

            # large dataframes
            if n_max < 1e6:
                pairs = random_pairs_without_replacement_small_frames(
                    self.n, shape, self.random_state)
            # small dataframes
            else:
                pairs = random_pairs_without_replacement_large_frames(
                    self.n, shape, self.random_state)

        levels = [df_a.index.values, df_b.index.values]
        labels = pairs
        names = [df_a.index.name, df_b.index.name]

        return pandas.MultiIndex(
            levels=levels,
            labels=labels,
            names=names,
            verify_integrity=False
        )

    def _dedup_index(self, df_a):

        shape = (len(df_a),)

        # with replacement
        if self.replace:
            pairs = random_pairs_with_replacement(
                self.n, shape, self.random_state)

        # without replacement
        else:

            n_max = max_pairs(shape)

            if not isinstance(self.n, int) or self.n <= 0 or self.n > n_max:
                raise ValueError(
                    "n must be a integer satisfying 0<n<=%s" % n_max)

            # large dataframes
            if n_max < 1e6:
                pairs = random_pairs_without_replacement_small_frames(
                    self.n, shape, self.random_state)
            # small dataframes
            else:
                pairs = random_pairs_without_replacement_large_frames(
                    self.n, shape, self.random_state)

        levels = [df_a.index.values, df_a.index.values]
        labels = pairs
        names = [df_a.index.name, df_a.index.name]

        return pandas.MultiIndex(
            levels=levels,
            labels=labels,
            names=names,
            verify_integrity=False
        )
