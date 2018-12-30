from __future__ import division

import warnings

import pandas
import numpy

from recordlinkage.base import BaseIndexAlgorithm
from recordlinkage.utils import DeprecationHelper, listify
from recordlinkage.measures import full_index_size
from recordlinkage.algorithms.indexing import (
    random_pairs_with_replacement,
    random_pairs_without_replacement_small_frames,
    random_pairs_without_replacement_large_frames)

from recordlinkage import rl_logging as logging


class Full(BaseIndexAlgorithm):
    """Class to generate a 'full' index.

    A full index is an index with all possible combinations of record pairs.
    In case of linking, this indexation method generates the cartesian product
    of both DataFrame's. In case of deduplicating DataFrame A, this indexation
    method are the pairs defined by the upper triangular matrix of the A x A.

    Parameters
    ----------
    **kwargs :
        Additional keyword arguments to pass to
        :class:`recordlinkage.base.BaseIndexAlgorithm`.

    Note
    ----
    This indexation method can be slow for large DataFrame's. The number of
    comparisons scales quadratic.
    Also, not all classifiers work well with large numbers of record pairs
    were most of the pairs are distinct.

    """

    def __init__(self, **kwargs):
        super(Full, self).__init__(**kwargs)

        logging.warn(
            "indexing - performance warning "
            "- A full index can result in large number of record pairs.")

    def _link_index(self, df_a, df_b):

        return pandas.MultiIndex.from_product(
            [df_a.index.values, df_b.index.values])

    def _dedup_index(self, df_a):

        levels = [df_a.index.values, df_a.index.values]
        labels = numpy.tril_indices(len(df_a.index), k=-1)

        return pandas.MultiIndex(
            levels=levels, labels=labels, verify_integrity=False)


class Block(BaseIndexAlgorithm):
    """Make candidate record pairs that agree on one or more variables.

    Returns all record pairs that agree on the given variable(s). This
    method is known as *blocking*. Blocking is an effective way to make a
    subset of the record space (A * B).

    Parameters
    ----------
    left_on : label, optional
        A column name or a list of column names of dataframe A. These
        columns are used to block on.
    right_on : label, optional
        A column name or a list of column names of dataframe B. These
        columns are used to block on. If 'right_on' is None, the `left_on`
        value is used. Default None.
    **kwargs :
        Additional keyword arguments to pass to
        :class:`recordlinkage.base.BaseIndexAlgorithm`.

    Examples
    --------
    In the following example, the record pairs are made for two historical
    datasets with census data. The datasets are named ``census_data_1980``
    and ``census_data_1990``.

    >>> indexer = recordlinkage.BlockIndex(on='first_name')
    >>> indexer.index(census_data_1980, census_data_1990)

    """

    def __init__(self, left_on=None, right_on=None, **kwargs):
        on = kwargs.pop('on', None)
        super(Block, self).__init__(**kwargs)

        # variables to block on
        self.left_on = left_on
        self.right_on = right_on

        if on is not None:
            warnings.warn(
                "The argument 'on' is deprecated. Use 'left_on=...' and "
                "'right_on=None' to simulate the behaviour of 'on'.",
                stacklevel=2)
            self.left_on, self.right_on = on, on

    def __repr__(self):

        class_name = self.__class__.__name__
        left_on, right_on = self._get_left_and_right_on()

        return "<{} left_on={!r}, right_on={!r}>".format(
            class_name, left_on, right_on)

    def _get_left_and_right_on(self):

        if self.right_on is None:
            return (self.left_on, self.left_on)
        else:
            return (self.left_on, self.right_on)

    def _link_index(self, df_a, df_b):

        left_on, right_on = self._get_left_and_right_on()
        left_on = listify(left_on)
        right_on = listify(right_on)

        blocking_keys = ["blocking_key_%d" % i for i, v in enumerate(left_on)]

        # make a dataset for the data on the left
        # 1. make a dataframe
        # 2. rename columns
        # 3. add index col
        # 4. drop na (last step to presever index)
        data_left = pandas.DataFrame(df_a[left_on], copy=False)
        data_left.columns = blocking_keys
        data_left['index_x'] = numpy.arange(len(df_a))
        data_left.dropna(axis=0, how='any', subset=blocking_keys, inplace=True)

        # make a dataset for the data on the right
        data_right = pandas.DataFrame(df_b[right_on], copy=False)
        data_right.columns = blocking_keys
        data_right['index_y'] = numpy.arange(len(df_b))
        data_right.dropna(
            axis=0, how='any', subset=blocking_keys, inplace=True)

        # merge the dataframes
        pairs_df = data_left.merge(data_right, how='inner', on=blocking_keys)

        return pandas.MultiIndex(
            levels=[df_a.index.values, df_b.index.values],
            labels=[pairs_df['index_x'].values, pairs_df['index_y'].values],
            verify_integrity=False)


class SortedNeighbourhood(BaseIndexAlgorithm):
    """Make candidate record pairs with the SortedNeighbourhood algorithm.

    This algorithm returns record pairs that agree on the sorting key, but
    also records pairs in their neighbourhood. A large window size results
    in more record pairs. A window size of 1 returns the blocking index.

    The Sorted Neighbourhood Index method is a great method when there is
    relatively large amount of spelling mistakes. Blocking will fail in
    that situation because it excludes to many records on minor spelling
    mistakes.

    Parameters
    ----------
    left_on : label, optional
        The column name of the sorting key of the first/left dataframe.
    right_on : label, optional
        The column name of the sorting key of the second/right dataframe.
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
    **kwargs :
        Additional keyword arguments to pass to
        :class:`recordlinkage.base.BaseIndexAlgorithm`.

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

    def __init__(self,
                 left_on=None,
                 right_on=None,
                 window=3,
                 sorting_key_values=None,
                 block_on=[],
                 block_left_on=[],
                 block_right_on=[],
                 **kwargs):
        on = kwargs.pop('on', None)
        super(SortedNeighbourhood, self).__init__(**kwargs)

        # variables to block on
        self.left_on = left_on
        self.right_on = right_on
        self.window = window
        self.sorting_key_values = sorting_key_values
        self.block_on = block_on
        self.block_left_on = block_left_on
        self.block_right_on = block_right_on

        if on is not None:
            warnings.warn(
                "The argument 'on' is deprecated. Use 'left_on=...' and "
                "'right_on=None' to simulate the behaviour of 'on'.",
                stacklevel=2)
            self.left_on, self.right_on = on, on

    def __repr__(self):

        class_name = self.__class__.__name__
        left_on, right_on = self._get_left_and_right_on()

        return "<{} left_on={!r}, right_on={!r}>".format(
            class_name, left_on, right_on)

    def _get_left_and_right_on(self):

        if self.right_on is None:
            return (self.left_on, self.left_on)
        else:
            return (self.left_on, self.right_on)

    def _get_sorting_key_values(self, array1, array2):
        """return the sorting key values as a series"""

        concat_arrays = numpy.concatenate([array1, array2])
        unique_values = numpy.unique(concat_arrays)

        return numpy.sort(unique_values)

    def _link_index(self, df_a, df_b):

        left_on, right_on = self._get_left_and_right_on()
        left_on = listify(left_on)
        right_on = listify(right_on)

        window = self.window

        # Check if window is an odd number
        if not isinstance(window, int) or (window < 0) or not bool(window % 2):
            raise ValueError('window is not a positive and odd integer')

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

        blocking_keys = ['sorting_key'] + \
            ["blocking_key_%d" % i for i, v in enumerate(block_left_on)]

        # make a dataset for the data on the left
        # 1. make a dataframe
        # 2. rename columns
        # 3. add index col
        # 4. drop na (last step to presever index)
        data_left = pandas.DataFrame(
            df_a[listify(left_on) + block_left_on], copy=False)
        data_left.columns = blocking_keys
        data_left['index_x'] = numpy.arange(len(df_a))
        data_left.dropna(axis=0, how='any', subset=blocking_keys, inplace=True)

        data_right = pandas.DataFrame(
            df_b[listify(right_on) + block_right_on], copy=False)
        data_right.columns = blocking_keys
        data_right['index_y'] = numpy.arange(len(df_b))
        data_right.dropna(
            axis=0, how='any', subset=blocking_keys, inplace=True)

        # sorting_key_values is the terminology in Data Matching [Christen,
        # 2012]
        if self.sorting_key_values is None:

            self.sorting_key_values = self._get_sorting_key_values(
                data_left['sorting_key'].values,
                data_right['sorting_key'].values)

        sorting_key_factors = pandas.Series(
            numpy.arange(len(self.sorting_key_values)),
            index=self.sorting_key_values)

        data_left['sorting_key'] = data_left['sorting_key'].map(
            sorting_key_factors)
        data_right['sorting_key'] = data_right['sorting_key'].map(
            sorting_key_factors)

        # Internal window size
        _window = int((window - 1) / 2)

        def merge_lagged(x, y, w):
            """Merge two dataframes with a lag on in the sorting key."""

            y = y.copy()
            y['sorting_key'] = y['sorting_key'] + w

            return x.merge(y, how='inner')

        pairs_concat = [
            merge_lagged(data_left, data_right, w)
            for w in range(-_window, _window + 1)
        ]

        pairs_df = pandas.concat(pairs_concat, axis=0)

        return pandas.MultiIndex(
            levels=[df_a.index.values, df_b.index.values],
            labels=[pairs_df['index_x'].values, pairs_df['index_y'].values],
            verify_integrity=False)


class Random(BaseIndexAlgorithm):
    """Class to generate random pairs of records.

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
    **kwargs :
        Additional keyword arguments to pass to
        :class:`recordlinkage.base.BaseIndexAlgorithm`.

    """

    def __init__(self, n, replace=True, random_state=None, **kwargs):
        super(Random, self).__init__(**kwargs)

        self.n = n
        self.replace = replace
        self.random_state = random_state

    def __repr__(self):

        class_name = self.__class__.__name__

        return "<{} n={!r}, replace={!r}>".format(class_name, self.n,
                                                  self.replace)

    def _link_index(self, df_a, df_b):

        shape = (len(df_a), len(df_b))
        n_max = full_index_size(shape)

        if not isinstance(self.n, int):
            raise ValueError('n must be an integer')

        # with replacement
        if self.replace:

            if n_max == 0:
                raise ValueError("one of the dataframes is empty")

            pairs = random_pairs_with_replacement(self.n, shape,
                                                  self.random_state)

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
            levels=levels, labels=labels, verify_integrity=False)

    def _dedup_index(self, df_a):

        shape = (len(df_a), )

        # with replacement
        if self.replace:
            pairs = random_pairs_with_replacement(self.n, shape,
                                                  self.random_state)

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
            levels=levels, labels=labels, verify_integrity=False)


class NeighbourhoodBlock(Block):
    '''
    :class:`recordlinkage.index.Block` with extended matching types
        * Proximity in record ranking order (like :class:`SortedNeighbourhood`),
          except multiple orderings (one for each field) are allowed
        * Wildcard matching of null values
        * A limited number of complete field mismatches

    Parameters
    ----------
    left_on : label, optional
        A column name or a list of column names of dataframe A. These
        columns are used for matching records.
    right_on : label, optional
        A column name or a list of column names of dataframe B. These
        columns are used for matching records. If 'right_on' is None,
        the `left_on` value is used. Default None.
    max_nulls: int, optional
        Include record pairs with up to this number of wildcard matches (see
        below).  Default: 0 (no wildcard matching)
    max_non_matches: int, optional
        Include record pairs with up to this number of field mismatches (see
        below).  Default: 0 (no mismatches allowed)
    windows: int, optional
        An integer or list of integers representing the window widths (as in
        :class:`SortedNeighbourhood`).  If fewer are specified than the number
        of keys (in *left_on* and/or *right_on*), the final one is repeated
        for the remaining keys.
    **kwargs :
        Additional keyword arguments to pass to
        :class:`recordlinkage.base.BaseIndexAlgorithm`.

    Special cases of this indexer are equivalent to other indexers in this
    module:
        * :class:`Block`: max_nulls=0, max_non_matches=0, *windows=1
          (the defaults)
        * :class:`SortedNeighbourhood`: max_nulls=0, max_non_matches=0,
          windows=[window value for the sorting key, 1 otherwise]
        * :class:`Full`: max_non_matches >= number of keys

    Example
    -------
    In the following example, the record pairs are made for two historical
    datasets with census data. The datasets are named ``census_data_1980``
    and ``census_data_1990``.  The index includes record pairs with matches
    in (at least) any 3 out of the 5 nominated fields.  Proximity matching is
    allowed in the first two fields, and up to one wildcard match of a missing
    value is also allowed.

    >>> keys = ['first_name', 'surname', 'date_of_birth', 'address', 'ssid']
    >>> windows = [9, 3, 1, 1, 1]
    >>> indexer = recordlinkage.index.NeighbourhoodBlock(keys, windows=windows, max_nulls=1, max_non_matches=2)
    >>> indexer.index(census_data_1980, census_data_1990)
    '''

    def __init__(self, left_on=None, right_on=None, max_nulls=0, max_non_matches=0, windows=1,
                 **kwargs):
        super(NeighbourhoodBlock, self).__init__(left_on=left_on, right_on=right_on, **kwargs)
        self.max_nulls = max_nulls
        self.max_non_matches = max_non_matches
        self.windows = listify(windows)

    def __repr__(self):
        cls = type(self)
        attrs = ['left_on', 'right_on', 'max_nulls',
                 'max_non_matches', 'windows']
        attrs_repr = ', '.join('{}={}'.format(attr, repr(getattr(self, attr)))
                               for attr in attrs)
        return '<{cls.__name__} {attrs_repr}>'.format(**locals())

    _coarsening_factor = 2

    def _index(self, df_a, df_b=None):
        dfs = [df_a, df_a if df_b is None else df_b]

        def split_to_match(a, to_match):
            ndx_bounds = numpy.r_[0, numpy.cumsum([len(x) for x in to_match])]
            assert len(a) == ndx_bounds[-1]
            return [a[start:stop]
                    for start, stop in zip(ndx_bounds, ndx_bounds[1:])]

        def deduped_blocks_and_indices(blocks, indices=None):
            if indices is None:
                indices = [numpy.arange(len(blocks))]
            deduped_blocks, index_tx = numpy.unique(blocks, axis=0,
                                                    return_inverse=True)
            return deduped_blocks, [index_tx[raw_ndx] for raw_ndx in indices]

        def get_normalized_linkage_params():
            def default_on_possibilities():
                yield self.left_on
                yield self.right_on
                yield [c for c in dfs[0].columns
                       if all(c in df.columns
                       for df in dfs)]
            default_on = next(filter(lambda x: x is not None,
                                     default_on_possibilities()))
            key_columns = [listify(side_on or default_on)
                           for side_on in [self.left_on, self.right_on]]
            n_key_cols, *n_key_cols_error = list(set(map(len, key_columns)))
            if n_key_cols_error or (n_key_cols == 0):
                raise IndexError('Invalid blocking keys')
            combined_ranks = numpy.vstack([pandas.concat([df[col] for df, col in zip(dfs, col_grp)]).rank(method='dense', na_option='keep').fillna(0).astype(int).values - 1
                                           for col_grp in zip(*key_columns)]).astype(float).T
            combined_ranks[combined_ranks < 0] = numpy.nan
            blocks, indices = deduped_blocks_and_indices(blocks=combined_ranks, indices=split_to_match(numpy.arange(len(combined_ranks)), dfs))
            n_keys = blocks.shape[1]
            windows = self.windows + self.windows[-1:] * (n_keys-len(self.windows))
            if (len(windows) > n_keys) or not all(isinstance(w, int) and (w > 0) and (w % 2 == 1) for w in windows):
                raise ValueError('Windows must be positive odd integers and the maximum number allowed is the number of blocking keys')
            rank_distance_limits = (numpy.array(windows) // 2).astype(float).reshape((1, -1))
            return blocks, indices, rank_distance_limits

        def many_to_many_join_indices(left_keys, right_keys, key_link):
            joined = pandas.DataFrame(key_link, columns=['left_key', 'right_key'])
            for side, values in [('left', left_keys), ('right', right_keys)]:
                joined = joined.join(pandas.DataFrame({'{side}_ndx'.format(**locals()): numpy.arange(len(values))}, index=values),
                                     how='inner', on='{side}_key'.format(**locals()))
            return joined[['left_ndx', 'right_ndx']].values

        def chain_indices(*index_groups):
            remaining_groups = iter(index_groups)
            result = list(next(remaining_groups))
            for txs in remaining_groups:
                result = [tx[r] for tx, r in zip(txs, result)]
            return result

        def linkage_index_codes(blocks, indices, rank_distance_limits, rank_max=None):
            if rank_max is None:
                rank_max = pandas.DataFrame(blocks).max().values
            if any(len(x) <= 1 for x in indices) or (pandas.Series(rank_max - rank_distance_limits.flatten()).max() in [0, numpy.nan]):
                block_pair_candidates = numpy.vstack([a.flatten() for a in numpy.meshgrid(*indices)]).T
            else:
                coarsened_blocks, (block_tx,) = deduped_blocks_and_indices(blocks=numpy.floor(blocks/self._coarsening_factor))
                coarsened_uniques, coarsened_ndx_tx = zip(*[numpy.unique(block_tx[x], return_inverse=True) for x in indices])
                coarsened_unique_link = linkage_index_codes(
                    blocks=coarsened_blocks,
                    indices=coarsened_uniques,
                    rank_distance_limits=numpy.ceil(rank_distance_limits / self._coarsening_factor),
                    rank_max=numpy.floor(rank_max / self._coarsening_factor)
                )
                coarsened_block_link = numpy.vstack(chain_indices(coarsened_unique_link.T, coarsened_uniques)).T
                block_pair_candidates = many_to_many_join_indices(block_tx, block_tx, key_link=coarsened_block_link)
            if len(block_pair_candidates) > 0:
                block_pair_candidates = numpy.unique(block_pair_candidates, axis=0)
            excess_rank_distances = numpy.abs(blocks[block_pair_candidates.T[0]] - blocks[block_pair_candidates.T[1]]) - rank_distance_limits
            null_counts = numpy.sum(numpy.isnan(excess_rank_distances), axis=1)
            match_counts = numpy.sum(numpy.nan_to_num(excess_rank_distances) <= 0.5, axis=1) - null_counts
            block_pair_accepted = (match_counts + numpy.clip(null_counts, 0, self.max_nulls)) >= (blocks.shape[1] - self.max_non_matches)
            return many_to_many_join_indices(*indices, key_link=block_pair_candidates[block_pair_accepted])

        if any(len(df) == 0 for df in dfs):
            rownum_pairs = numpy.array([], dtype=int).reshape((0, 2))
        else:
            blocks, indices, rank_distance_limits = get_normalized_linkage_params()
            rownum_pairs = linkage_index_codes(blocks, indices, rank_distance_limits)
        if df_b is None:    # dedup index
            rownum_pairs = rownum_pairs[rownum_pairs.T[0] > rownum_pairs.T[1]]
        index = pandas.MultiIndex(levels=[df.index.values for df in dfs], labels=rownum_pairs.T, names=['index_a', 'index_b'])
        return index

    def _link_index(self, df_a, df_b):
        return self._index(df_a, df_b)

    def _dedup_index(self, df_a):
        return self._index(df_a)


FullIndex = DeprecationHelper(
    Full, "class recordlinkage.FullIndex is renamed and moved, "
    "use recordlinkage.index.Full")
BlockIndex = DeprecationHelper(
    Block, "class recordlinkage.BlockIndex is renamed and moved, "
    "use recordlinkage.index.Block")
SortedNeighbourhoodIndex = DeprecationHelper(
    SortedNeighbourhood, "class recordlinkage.SortedNeighbourhoodIndex "
    "is renamed and moved, use recordlinkage.index.SortedNeighbourhood")
RandomIndex = DeprecationHelper(
    Random, "class recordlinkage.RandomIndex is renamed and moved, "
    "use recordlinkage.index.Random")
