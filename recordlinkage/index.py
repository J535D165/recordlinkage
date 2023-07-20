import warnings

import numpy
import pandas

from recordlinkage import rl_logging as logging
from recordlinkage.algorithms.indexing import random_pairs_with_replacement
from recordlinkage.algorithms.indexing import random_pairs_without_replacement
from recordlinkage.algorithms.indexing import (
    random_pairs_without_replacement_low_memory,  # NOQA
)
from recordlinkage.base import BaseIndexAlgorithm
from recordlinkage.measures import full_index_size
from recordlinkage.utils import listify


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
        super().__init__(**kwargs)

        logging.warning(
            "indexing - performance warning "
            "- A full index can result in large number of record pairs."
        )

    def _link_index(self, df_a, df_b):
        return pandas.MultiIndex.from_product([df_a.index.values, df_b.index.values])

    def _dedup_index(self, df_a):
        levels = [df_a.index.values, df_a.index.values]
        codes = numpy.tril_indices(len(df_a.index), k=-1)

        return pandas.MultiIndex(levels=levels, codes=codes, verify_integrity=False)


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
        on = kwargs.pop("on", None)
        super().__init__(**kwargs)

        # variables to block on
        self.left_on = left_on
        self.right_on = right_on

        if on is not None:
            warnings.warn(
                "The argument 'on' is deprecated. Use 'left_on=...' and "
                "'right_on=None' to simulate the behaviour of 'on'.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.left_on, self.right_on = on, on

    def __repr__(self):
        class_name = self.__class__.__name__
        left_on, right_on = self._get_left_and_right_on()

        return f"<{class_name} left_on={left_on!r}, right_on={right_on!r}>"

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
        data_left["index_x"] = numpy.arange(len(df_a))
        data_left.dropna(axis=0, how="any", subset=blocking_keys, inplace=True)

        # make a dataset for the data on the right
        data_right = pandas.DataFrame(df_b[right_on], copy=False)
        data_right.columns = blocking_keys
        data_right["index_y"] = numpy.arange(len(df_b))
        data_right.dropna(axis=0, how="any", subset=blocking_keys, inplace=True)

        # merge the dataframes
        pairs_df = data_left.merge(data_right, how="inner", on=blocking_keys)

        return pandas.MultiIndex(
            levels=[df_a.index.values, df_b.index.values],
            codes=[pairs_df["index_x"].values, pairs_df["index_y"].values],
            verify_integrity=False,
        )


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

    >>> indexer = recordlinkage.SortedNeighbourhoodIndex(
            'first_name', window=9
        )
    >>> indexer.index(census_data_1980, census_data_1990)

    When the sorting key has different names in both dataframes:

    >>> indexer = recordlinkage.SortedNeighbourhoodIndex(
            left_on='first_name', right_on='given_name', window=9
        )
    >>> indexer.index(census_data_1980, census_data_1990)

    """

    def __init__(
        self,
        left_on=None,
        right_on=None,
        window=3,
        sorting_key_values=None,
        block_on=[],
        block_left_on=[],
        block_right_on=[],
        **kwargs,
    ):
        on = kwargs.pop("on", None)
        super().__init__(**kwargs)

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
                DeprecationWarning,
                stacklevel=2,
            )
            self.left_on, self.right_on = on, on

    def __repr__(self):
        class_name = self.__class__.__name__
        left_on, right_on = self._get_left_and_right_on()

        return f"<{class_name} left_on={left_on!r}, right_on={right_on!r}>"

    def _get_left_and_right_on(self):
        if self.right_on is None:
            return (self.left_on, self.left_on)
        else:
            return (self.left_on, self.right_on)

    def _get_sorting_key_values(self, array1, array2):
        """return the sorting key values as a series"""

        concat_arrays = numpy.concatenate([array1, array2])
        return numpy.unique(concat_arrays)  # numpy.unique returns sorted list

    def _link_index(self, df_a, df_b):
        left_on, right_on = self._get_left_and_right_on()
        left_on = listify(left_on)
        right_on = listify(right_on)

        window = self.window

        # Check if window is an odd number
        if not isinstance(window, int) or (window < 0) or not bool(window % 2):
            raise ValueError("window is not a positive and odd integer")

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

        blocking_keys = ["sorting_key"] + [
            "blocking_key_%d" % i for i, v in enumerate(block_left_on)
        ]

        # make a dataset for the data on the left
        # 1. make a dataframe
        # 2. rename columns
        # 3. add index col
        # 4. drop na (last step to presever index)
        data_left = pandas.DataFrame(df_a[listify(left_on) + block_left_on], copy=False)
        data_left.columns = blocking_keys
        data_left["index_x"] = numpy.arange(len(df_a))
        data_left.dropna(axis=0, how="any", subset=blocking_keys, inplace=True)

        data_right = pandas.DataFrame(
            df_b[listify(right_on) + block_right_on], copy=False
        )
        data_right.columns = blocking_keys
        data_right["index_y"] = numpy.arange(len(df_b))
        data_right.dropna(axis=0, how="any", subset=blocking_keys, inplace=True)

        # sorting_key_values is the terminology in Data Matching [Christen,
        # 2012]
        if self.sorting_key_values is None:
            self.sorting_key_values = self._get_sorting_key_values(
                data_left["sorting_key"].values, data_right["sorting_key"].values
            )

        sorting_key_factors = pandas.Series(
            numpy.arange(len(self.sorting_key_values)), index=self.sorting_key_values
        )

        data_left["sorting_key"] = data_left["sorting_key"].map(sorting_key_factors)
        data_right["sorting_key"] = data_right["sorting_key"].map(sorting_key_factors)

        # Internal window size
        _window = int((window - 1) / 2)

        def merge_lagged(x, y, w):
            """Merge two dataframes with a lag on in the sorting key."""

            y = y.copy()
            y["sorting_key"] = y["sorting_key"] + w

            return x.merge(y, how="inner")

        pairs_concat = [
            merge_lagged(data_left, data_right, w) for w in range(-_window, _window + 1)
        ]

        pairs_df = pandas.concat(pairs_concat, axis=0)

        return pandas.MultiIndex(
            levels=[df_a.index.values, df_b.index.values],
            codes=[pairs_df["index_x"].values, pairs_df["index_y"].values],
            verify_integrity=False,
        )


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
        super().__init__(**kwargs)

        self.n = n
        self.replace = replace
        self.random_state = random_state

    def __repr__(self):
        class_name = self.__class__.__name__

        return f"<{class_name} n={self.n!r}, replace={self.replace!r}>"

    def _link_index(self, df_a, df_b):
        shape = (len(df_a), len(df_b))
        n_max = full_index_size(shape)

        if not isinstance(self.n, int):
            raise ValueError("n must be an integer")

        # with replacement
        if self.replace:
            if n_max == 0:
                raise ValueError("one of the dataframes is empty")

            pairs = random_pairs_with_replacement(self.n, shape, self.random_state)

        # without replacement
        else:
            if self.n <= 0 or self.n > n_max:
                raise ValueError("n must be a integer satisfying 0<n<=%s" % n_max)

            # the fraction of pairs in the sample
            frac = self.n / n_max

            # large dataframes
            if n_max < 1e6 or frac > 0.5:
                pairs = random_pairs_without_replacement(
                    self.n, shape, self.random_state
                )
            # small dataframes
            else:
                pairs = random_pairs_without_replacement_low_memory(
                    self.n, shape, self.random_state
                )

        levels = [df_a.index.values, df_b.index.values]
        codes = pairs

        return pandas.MultiIndex(levels=levels, codes=codes, verify_integrity=False)

    def _dedup_index(self, df_a):
        shape = (len(df_a),)

        # with replacement
        if self.replace:
            pairs = random_pairs_with_replacement(self.n, shape, self.random_state)

        # without replacement
        else:
            n_max = full_index_size(shape)

            if not isinstance(self.n, int) or self.n <= 0 or self.n > n_max:
                raise ValueError("n must be a integer satisfying 0<n<=%s" % n_max)

            # large dataframes
            if n_max < 1e6:
                pairs = random_pairs_without_replacement(
                    self.n, shape, self.random_state
                )
            # small dataframes
            else:
                pairs = random_pairs_without_replacement_low_memory(
                    self.n, shape, self.random_state
                )

        levels = [df_a.index.values, df_a.index.values]
        labels = pairs

        return pandas.MultiIndex(levels=levels, codes=labels, verify_integrity=False)
