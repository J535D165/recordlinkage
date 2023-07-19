#!/usr/bin/env python

from operator import eq
from operator import gt

import numpy as np
import pytest

from recordlinkage.contrib.index import NeighbourhoodBlock
from recordlinkage.index import Block
from recordlinkage.index import Full
from recordlinkage.index import SortedNeighbourhood
from tests.test_indexing import TestData


class TestNeighbourhoodBlock(TestData):
    """General unittest for the NeighbourhoodBlocking indexing class."""

    @classmethod
    def setup_class(cls):
        TestData.setup_class()

        def incomplete_df_copy(df, nan_proportion=0.1):
            "copy of DataFrame with some cells set to NaN"
            nan_count = int(round(len(df) * nan_proportion))

            def with_nulls(vals):
                vals = vals.copy()
                vals.iloc[
                    np.random.choice(len(df), size=nan_count, replace=False)
                ] = np.nan
                return vals

            return df.copy() if nan_count <= 0 else df.apply(with_nulls)

        np.random.seed(0)
        cls.incomplete_a = incomplete_df_copy(cls.a)
        cls.incomplete_b = incomplete_df_copy(cls.b)

    def assert_index_comparisons(self, pairwise_comparison, indexers, *args, **kwargs):
        indexes = [ndxr.index(*args, **kwargs) for ndxr in indexers]
        for index1, index2 in zip(indexes, indexes[1:]):
            pairs1, pairs2 = map(set, [index1, index2])
            assert (
                (len(pairs1) == len(index1))
                and (len(pairs2) == len(index2))
                and pairwise_comparison(pairs1, pairs2)
            )

    def test_dedup_vs_full(self):
        indexers = [
            NeighbourhoodBlock(max_non_matches=len(self.a.columns)),
            Full(),
        ]
        self.assert_index_comparisons(eq, indexers, self.a)

    def test_link_vs_full(self):
        indexers = [
            NeighbourhoodBlock(max_non_matches=len(self.a.columns)),
            Full(),
        ]
        self.assert_index_comparisons(eq, indexers, self.a, self.b)

    def test_dedup_single_blocking_key_vs_block(self):
        indexers = [
            NeighbourhoodBlock("var_block10", max_nulls=1),
            NeighbourhoodBlock(
                left_on="var_block10", right_on="var_block10", max_nulls=1
            ),
            Block("var_block10"),
        ]
        self.assert_index_comparisons(eq, indexers, self.a)
        self.assert_index_comparisons(gt, indexers[-2:], self.incomplete_a)

    def test_link_single_blocking_key_vs_block(self):
        indexers = [
            NeighbourhoodBlock("var_arange", max_nulls=1),
            NeighbourhoodBlock(
                left_on="var_arange", right_on="var_arange", max_nulls=1
            ),
            Block("var_arange"),
        ]
        self.assert_index_comparisons(eq, indexers, self.a, self.b)
        self.assert_index_comparisons(
            gt, indexers[-2:], self.incomplete_a, self.incomplete_b
        )

    def test_dedup_multiple_blocking_keys_vs_block(self):
        indexers = [
            NeighbourhoodBlock(["var_single", "var_block10"], max_nulls=1),
            NeighbourhoodBlock(
                left_on=["var_single", "var_block10"],
                right_on=["var_single", "var_block10"],
                max_nulls=1,
            ),
            Block(["var_single", "var_block10"]),
        ]
        self.assert_index_comparisons(eq, indexers, self.a)
        self.assert_index_comparisons(gt, indexers[-2:], self.incomplete_a)

    def test_link_multiple_blocking_keys_vs_block(self):
        indexers = [
            NeighbourhoodBlock(["var_arange", "var_block10"], max_nulls=1),
            NeighbourhoodBlock(
                left_on=["var_arange", "var_block10"],
                right_on=["var_arange", "var_block10"],
                max_nulls=1,
            ),
            Block(["var_arange", "var_block10"]),
        ]
        self.assert_index_comparisons(eq, indexers, self.a, self.b)
        self.assert_index_comparisons(
            gt, indexers[-2:], self.incomplete_a, self.incomplete_b
        )

    @pytest.mark.parametrize("window", [3, 5, 7, 9, 11])
    def test_dedup_single_sorting_key_vs_sortedneighbourhood(self, window):
        indexers = [
            NeighbourhoodBlock("var_arange", max_nulls=1, windows=window),
            NeighbourhoodBlock(
                left_on="var_arange", right_on="var_arange", max_nulls=1, windows=window
            ),
            SortedNeighbourhood("var_arange", window=window),
        ]
        self.assert_index_comparisons(eq, indexers, self.a)
        self.assert_index_comparisons(gt, indexers[-2:], self.incomplete_a)

    @pytest.mark.parametrize("window", [3, 5, 7, 9, 11])
    def test_link_single_sorting_key_vs_sortedneighbourhood(self, window):
        indexers = [
            NeighbourhoodBlock("var_arange", max_nulls=1, windows=window),
            NeighbourhoodBlock(
                left_on="var_arange", right_on="var_arange", max_nulls=1, windows=window
            ),
            SortedNeighbourhood("var_arange", window=window),
        ]
        self.assert_index_comparisons(eq, indexers, self.a, self.b)
        self.assert_index_comparisons(
            gt, indexers[-2:], self.incomplete_a, self.incomplete_b
        )

    @pytest.mark.parametrize("window", [3, 5, 7, 9, 11])
    def test_dedup_with_blocking_vs_sortedneighbourhood(self, window):
        indexers = [
            NeighbourhoodBlock(
                ["var_arange", "var_block10"], max_nulls=1, windows=[window, 1]
            ),
            NeighbourhoodBlock(
                left_on=["var_arange", "var_block10"],
                right_on=["var_arange", "var_block10"],
                max_nulls=1,
                windows=[window, 1],
            ),
            SortedNeighbourhood("var_arange", block_on="var_block10", window=window),
        ]
        self.assert_index_comparisons(eq, indexers, self.a)
        self.assert_index_comparisons(gt, indexers[-2:], self.incomplete_a)

    @pytest.mark.parametrize("window", [3, 5, 7, 9, 11])
    def test_link_with_blocking_vs_sortedneighbourhood(self, window):
        indexers = [
            NeighbourhoodBlock(
                ["var_arange", "var_block10"], max_nulls=1, windows=[window, 1]
            ),
            NeighbourhoodBlock(
                left_on=["var_arange", "var_block10"],
                right_on=["var_arange", "var_block10"],
                max_nulls=1,
                windows=[window, 1],
            ),
            SortedNeighbourhood("var_arange", block_on="var_block10", window=window),
        ]
        self.assert_index_comparisons(eq, indexers, self.a, self.b)
        self.assert_index_comparisons(
            gt, indexers[-2:], self.incomplete_a, self.incomplete_b
        )
