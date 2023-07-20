import numpy
import pandas

from recordlinkage.index import Block
from recordlinkage.utils import listify


class NeighbourhoodBlock(Block):
    """
    :class:`recordlinkage.index.Block` with extended matching types
        * Proximity in record ranking order (like :class:`SortedNeighbourhood`)
          , except multiple orderings (one for each field) are allowed
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

    Wildcard matching
    -----------------
    Missing values can be treated as wild (ie: matching any other value)
    for a limited number of fields determined by the max_nulls parameter.

    Relationship to other index types
    ---------------------------------
    Special cases of this indexer are equivalent to other index types:
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

    >>> from recordlinkage.contrib.index import NeighbourhoodBlock
    >>> keys = ['first_name', 'surname', 'date_of_birth', 'address', 'ssid']
    >>> windows = [9, 3, 1, 1, 1]
    >>> indexer = NeighbourhoodBlock(
    >>>     keys, windows=windows, max_nulls=1, max_non_matches=2)
    >>> indexer.index(census_data_1980, census_data_1990)
    """

    def __init__(
        self,
        left_on=None,
        right_on=None,
        max_nulls=0,
        max_non_matches=0,
        windows=1,
        **kwargs,
    ):
        super().__init__(left_on=left_on, right_on=right_on, **kwargs)
        self.max_nulls = max_nulls
        self.max_non_matches = max_non_matches
        self.windows = listify(windows)

    def __repr__(self):
        cls = type(self)
        attrs = ["left_on", "right_on", "max_nulls", "max_non_matches", "windows"]
        attrs_repr = ", ".join(f"{attr}={repr(getattr(self, attr))}" for attr in attrs)
        return f"<{cls.__name__} {attrs_repr}>"

    _coarsening_factor = 2

    def _index(self, df_a, df_b=None):
        dfs = [df_a, df_a if df_b is None else df_b]

        def split_to_match(a, to_match):
            ndx_bounds = numpy.r_[0, numpy.cumsum([len(x) for x in to_match])]
            assert len(a) == ndx_bounds[-1]
            return [a[start:stop] for start, stop in zip(ndx_bounds, ndx_bounds[1:])]

        def deduped_blocks_and_indices(blocks, indices=None):
            if indices is None:
                indices = [numpy.arange(len(blocks))]
            deduped_blocks, index_tx = numpy.unique(blocks, axis=0, return_inverse=True)
            return deduped_blocks, [index_tx[raw_ndx] for raw_ndx in indices]

        def get_normalized_linkage_params():
            def default_on_possibilities():
                yield self.left_on
                yield self.right_on
                yield [c for c in dfs[0].columns if all(c in df.columns for df in dfs)]

            default_on = next(
                iter(filter(lambda x: x is not None, default_on_possibilities()))
            )
            key_columns = [
                listify(side_on or default_on)
                for side_on in [self.left_on, self.right_on]
            ]
            key_cols = set(map(len, key_columns))
            n_key_cols = next(iter(key_cols))
            if (len(key_cols) > 1) or (n_key_cols == 0):
                raise IndexError("Invalid blocking keys")
            combined_ranks = (
                numpy.vstack(
                    [
                        pandas.concat([df[col] for df, col in zip(dfs, col_grp)])
                        .rank(method="dense", na_option="keep")
                        .fillna(0)
                        .astype(int)
                        .values
                        - 1
                        for col_grp in zip(*key_columns)
                    ]
                )
                .astype(float)
                .T
            )
            combined_ranks[combined_ranks < 0] = numpy.nan
            blocks, indices = deduped_blocks_and_indices(
                blocks=combined_ranks,
                indices=split_to_match(numpy.arange(len(combined_ranks)), dfs),
            )
            n_keys = blocks.shape[1]
            windows = self.windows + self.windows[-1:] * (n_keys - len(self.windows))
            if (len(windows) > n_keys) or not all(
                isinstance(w, int) and (w > 0) and (w % 2 == 1) for w in windows
            ):
                raise ValueError(
                    "Windows must be positive odd integers and the maximum"
                    "number allowed is the number of blocking keys"
                )
            rank_distance_limits = (
                (numpy.array(windows) // 2).astype(float).reshape((1, -1))
            )
            return blocks, indices, rank_distance_limits

        def many_to_many_join_indices(left_keys, right_keys, key_link):
            joined = pandas.DataFrame(key_link, columns=["left_key", "right_key"])
            for side, values in [("left", left_keys), ("right", right_keys)]:
                joined = joined.join(
                    pandas.DataFrame(
                        {f"{side}_ndx": numpy.arange(len(values))},
                        index=values,
                    ),
                    how="inner",
                    on=f"{side}_key",
                )
            return joined[["left_ndx", "right_ndx"]].values

        def chain_indices(*index_groups):
            remaining_groups = iter(index_groups)
            result = list(next(remaining_groups))
            for txs in remaining_groups:
                result = [tx[r] for tx, r in zip(txs, result)]
            return result

        def linkage_index_codes(blocks, indices, rank_distance_limits, rank_max=None):
            if rank_max is None:
                rank_max = pandas.DataFrame(blocks).max().values
            if any(len(x) <= 1 for x in indices) or (
                pandas.Series(rank_max - rank_distance_limits.flatten()).max()
                in [0, numpy.nan]
            ):
                block_pair_candidates = numpy.vstack(
                    [a.flatten() for a in numpy.meshgrid(*indices)]
                ).T
            else:
                coarsened_blocks, (block_tx,) = deduped_blocks_and_indices(
                    blocks=numpy.floor(blocks / self._coarsening_factor)
                )
                coarsened_uniques, coarsened_ndx_tx = zip(
                    *[numpy.unique(block_tx[x], return_inverse=True) for x in indices]
                )
                coarsened_unique_link = linkage_index_codes(
                    blocks=coarsened_blocks,
                    indices=coarsened_uniques,
                    rank_distance_limits=numpy.ceil(
                        rank_distance_limits / self._coarsening_factor
                    ),
                    rank_max=numpy.floor(rank_max / self._coarsening_factor),
                )
                coarsened_block_link = numpy.vstack(
                    chain_indices(coarsened_unique_link.T, coarsened_uniques)
                ).T
                block_pair_candidates = many_to_many_join_indices(
                    block_tx, block_tx, key_link=coarsened_block_link
                )
            if len(block_pair_candidates) > 0:
                block_pair_candidates = numpy.unique(block_pair_candidates, axis=0)
            excess_rank_distances = (
                numpy.abs(
                    blocks[block_pair_candidates.T[0]]
                    - blocks[block_pair_candidates.T[1]]
                )
                - rank_distance_limits
            )
            null_counts = numpy.sum(numpy.isnan(excess_rank_distances), axis=1)
            match_counts = (
                numpy.sum(numpy.nan_to_num(excess_rank_distances) <= 0.5, axis=1)
                - null_counts
            )
            block_pair_accepted = (
                match_counts + numpy.clip(null_counts, 0, self.max_nulls)
            ) >= (blocks.shape[1] - self.max_non_matches)
            return many_to_many_join_indices(
                *indices, key_link=block_pair_candidates[block_pair_accepted]
            )

        if any(len(df) == 0 for df in dfs):
            rownum_pairs = numpy.array([], dtype=int).reshape((0, 2))
        else:
            blocks, indices, rank_distance_limits = get_normalized_linkage_params()
            rownum_pairs = linkage_index_codes(blocks, indices, rank_distance_limits)
        if df_b is None:  # dedup index
            rownum_pairs = rownum_pairs[rownum_pairs.T[0] > rownum_pairs.T[1]]
        index = pandas.MultiIndex(
            levels=[df.index.values for df in dfs],
            codes=rownum_pairs.T,
            names=["index_a", "index_b"],
        )
        return index

    def _link_index(self, df_a, df_b):
        return self._index(df_a, df_b)

    def _dedup_index(self, df_a):
        return self._index(df_a)
