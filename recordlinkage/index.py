from __future__ import division

import pandas
import numpy

from recordlinkage.base import BaseIndexAlgorithm
from recordlinkage.utils import (
    IndexError,
    DeprecationHelper,
    listify)
from recordlinkage.measures import full_index_size
from recordlinkage.algorithms.indexing import (
    random_pairs_with_replacement,
    random_pairs_without_replacement_small_frames,
    random_pairs_without_replacement_large_frames)

from recordlinkage import rl_logging as logging


class Full(BaseIndexAlgorithm):
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

    def __init__(self):
        super(Full, self).__init__()

    def _link_index(self, df_a, df_b):

        n_max = full_index_size((df_a, df_b))

        if n_max > 1e7:
            logging.warn(
                "The number of record pairs is large. Consider a different "
                "indexation algorithm for better performance. "
            )

        return pandas.MultiIndex.from_product(
            [df_a.index.values, df_b.index.values]
        )

    def _dedup_index(self, df_a):

        n_max = full_index_size((df_a))

        if n_max > 1e7:
            logging.warn(
                "The number of record pairs is large. Consider a different "
                "indexation algorithm for better performance. "
            )

        levels = [df_a.index.values, df_a.index.values]
        labels = numpy.triu_indices(len(df_a.index), k=1)

        return pandas.MultiIndex(
            levels=levels,
            labels=labels,
            verify_integrity=False
        )


class Block(BaseIndexAlgorithm):
    """Block(on=None, left_on=None, right_on=None)
    Make candidate record pairs that agree on one or more variables.

    Returns all record pairs that agree on the given variable(s). This
    method is known as *blocking*. Blocking is an effective way to make a
    subset of the record space (A * B).

    Parameters
    ----------
    on : label, optional
        A column name or a list of column names. These column(s) are used
        to block on. When linking two dataframes, the 'on' argument needs
        to be present in both dataframes.
    left_on : label, optional
        A column name or a list of column names of dataframe A. These
        columns are used to block on. This argument is ignored when
        argument 'on' is given.
    right_on : label, optional
        A column name or a list of column names of dataframe B. These
        columns are used to block on. This argument is ignored when
        argument 'on' is given.

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
        super(Block, self).__init__(*args, **kwargs)

        # variables to block on
        self.on = on
        self.left_on = left_on
        self.right_on = right_on

    def __repr__(self):

        class_name = self.__class__.__name__

        if self.on is not None:
            left_on = right_on = self.on
        else:
            left_on, right_on = self.left_on, self.right_on

        return "<{} left_on={!r}, right_on={!r}>".format(
            class_name, left_on, right_on)

    def _link_index(self, df_a, df_b):

        if self.on is not None:
            if self.left_on is not None or self.right_on is not None:
                raise IndexError('Can only pass argument "on" OR "left_on" '
                                 'and "right_on", not a combination of both.')
            left_on = right_on = listify(self.on)
        else:
            if self.left_on is None and self.right_on is None:
                raise IndexError('pass argument "on" OR "left_on" and '
                                 '"right_on" at class initalization.')
            elif self.left_on is None:
                raise IndexError('Argument "left_on" is missing '
                                 'at class initalization.')
            elif self.right_on is None:
                raise IndexError('Argument "right_on" is missing '
                                 'at class initalization.')
            else:
                left_on = listify(self.left_on)
                right_on = listify(self.right_on)

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

        return pairs.index


class SortedNeighbourhood(BaseIndexAlgorithm):
    """SortedNeighbourhood(on=None, left_on=None, right_on=None, window=3, sorting_key_values=None, block_on=[], block_left_on=[], block_right_on=[])
    Make candidate record pairs with the SortedNeighbourhood algorithm.

    This algorithm returns record pairs that agree on the sorting key, but
    also records pairs in their neighbourhood. A large window size results
    in more record pairs. A window size of 1 returns the blocking index.

    The Sorted Neighbourhood Index method is a great method when there is
    relatively large amount of spelling mistakes. Blocking will fail in
    that situation because it excludes to many records on minor spelling
    mistakes.

    Parameters
    ----------
    on : label, optional
        The column name of the sorting key. When linking two dataframes,
        the 'on' argument needs to be present in both dataframes.
    left_on : label, optional
        The column name of the sorting key of the first/left dataframe.
        This argument is ignored when argument 'on' is not None.
    right_on : label, optional
        The column name of the sorting key of the second/right dataframe.
        This argument is ignored when argument 'on' is not None.
    window: int, optional
        The width of the window, default is 3
    sorting_key_values: array, optional
        A list of sorting key values (optional).
    block_on: label
        Additional columns to apply standard blocking on.
    block_left_on: label
        Additional columns in the left dataframe to apply standard
        blocking on.
    block_right_on: label
        Additional columns in the right dataframe to apply standard
        blocking on.

    Examples
    --------
    In the following example, the record pairs are made for two historical
    datasets with census data. The datasets are named ``census_data_1980``
    and ``census_data_1990``.

    >>> indexer = recordlinkage.SortedNeighbourhoodIndex('first_name', window=9)
    >>> indexer.index(census_data_1980, census_data_1990)

    When the sorting key has different names in both dataframes:

    >>> indexer = recordlinkage.SortedNeighbourhoodIndex(
            left_on='first_name', right_on='given_name', window=9
        )
    >>> indexer.index(census_data_1980, census_data_1990)

    """

    def __init__(self, on=None, left_on=None, right_on=None, window=3,
                 sorting_key_values=None, block_on=[], block_left_on=[],
                 block_right_on=[], *args, **kwargs):
        super(SortedNeighbourhood, self).__init__(*args, **kwargs)

        # variables to block on
        self.on = on
        self.left_on = left_on
        self.right_on = right_on
        self.window = window
        self.sorting_key_values = sorting_key_values
        self.block_on = block_on
        self.block_left_on = block_left_on
        self.block_right_on = block_right_on

    def __repr__(self):

        class_name = self.__class__.__name__

        if self.on is not None:
            left_on = right_on = self.on
        else:
            left_on, right_on = self.left_on, self.right_on

        return "<{} left_on={!r}, right_on={!r}>".format(
            class_name, left_on, right_on)

    def _get_sorting_key_values(self, array1, array2):
        """return the sorting key values as a series"""

        concat_arrays = numpy.concatenate([array1, array2])
        unique_values = numpy.unique(concat_arrays)

        return numpy.sort(unique_values)

    def _link_index(self, df_a, df_b):

        if self.on is not None:
            if self.left_on is not None or self.right_on is not None:
                raise IndexError('Can only pass argument "on" OR "left_on" '
                                 'and "right_on", not a combination of both.')
            left_on = right_on = listify(self.on)
        else:
            if self.left_on is None and self.right_on is None:
                raise IndexError('pass argument "on" OR "left_on" and '
                                 '"right_on" at class initalization.')
            elif self.left_on is None:
                raise IndexError('Argument "left_on" is missing '
                                 'at class initalization.')
            elif self.right_on is None:
                raise IndexError('Argument "right_on" is missing '
                                 'at class initalization.')
            else:
                left_on = listify(self.left_on)
                right_on = listify(self.right_on)

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
        data_left.columns = ['sorting_key'] + \
            ["blocking_key_%d" % i for i, v in enumerate(block_left_on)]
        data_left['index_x'] = data_left.index

        data_right = df_b[listify(right_on) + block_right_on].dropna(
            axis=0, how='any', inplace=False
        )
        data_right.columns = ['sorting_key'] + \
            ["blocking_key_%d" % i for i, v in enumerate(block_right_on)]
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
        ).index

        return pairs


class Random(BaseIndexAlgorithm):
    """Random(n, replace=True, random_state=None)
    Class to generate random pairs of records.

    This class returns random pairs of records with or without
    replacement. Use the random_state parameter to seed the algorithm and
    reproduce results. This way to make record pairs is useful for the
    training of unsupervised learning models for record linkage.

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
        Seed for the random number generator (if int), or
        numpy.RandomState object.

    """

    def __init__(self, n, replace=True, random_state=None, *args, **kwargs):
        super(Random, self).__init__(*args, **kwargs)

        self.n = n
        self.replace = replace
        self.random_state = random_state

    def __repr__(self):

        class_name = self.__class__.__name__

        return "<{} n={!r}, replace={!r}>".format(
            class_name, self.n, self.replace)

    def _link_index(self, df_a, df_b):

        shape = (len(df_a), len(df_b))
        n_max = full_index_size(shape)

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

        return pandas.MultiIndex(
            levels=levels,
            labels=labels,
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

            n_max = full_index_size(shape)

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

        return pandas.MultiIndex(
            levels=levels,
            labels=labels,
            verify_integrity=False
        )


FullIndex = DeprecationHelper(
    Full, "This class is moved to recordlinkage.index.Full.")
BlockIndex = DeprecationHelper(
    Block, "This class is moved to recordlinkage.index.Block.")
SortedNeighbourhoodIndex = DeprecationHelper(
    SortedNeighbourhood,
    "This class is moved to recordlinkage.index.SortedNeighbourhood.")
RandomIndex = DeprecationHelper(
    Random, "This class is moved to recordlinkage.index.Random.")
