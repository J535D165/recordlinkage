import os
import pickle
import shutil
import tempfile

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import recordlinkage
from recordlinkage.algorithms.indexing import random_pairs_without_replacement
from recordlinkage.algorithms.indexing import (
    random_pairs_without_replacement_low_memory,
)
from recordlinkage.contrib.index import NeighbourhoodBlock
from recordlinkage.index import Block
from recordlinkage.index import Full
from recordlinkage.index import Random
from recordlinkage.index import SortedNeighbourhood
from recordlinkage.types import is_pandas_2d_multiindex


def get_test_algorithms():
    """Return list of algorithms"""
    return [
        Full(),
        Block("var_arange"),
        SortedNeighbourhood("var_arange"),
        Random(10, random_state=100, replace=True),
        Random(10, random_state=100, replace=False),
        NeighbourhoodBlock("var_arange"),
    ]


class TestData:
    """Unittest object to setup test data."""

    @classmethod
    def setup_class(cls):
        n_a = 100
        n_b = 150

        cls.index_a = ["rec_a_%s" % i for i in range(0, n_a)]
        cls.index_b = ["rec_b_%s" % i for i in range(0, n_b)]

        cls.a = pd.DataFrame(
            {
                "var_single": np.repeat([1], n_a),
                "var_arange": np.arange(n_a),
                "var_arange_str": np.arange(n_a),
                "var_block10": np.repeat(np.arange(n_a / 10), 10),
            },
            index=cls.index_a,
        )

        cls.b = pd.DataFrame(
            {
                "var_single": np.repeat([1], n_b),
                "var_arange": np.arange(n_b),
                "var_arange_str": np.arange(n_b),
                "var_block10": np.repeat(np.arange(n_b / 10), 10),
            },
            index=cls.index_b,
        )

        # Create a temporary directory
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def teardown_class(cls):
        # Remove the test directory
        shutil.rmtree(cls.test_dir)


class TestIndexApi(TestData):
    def test_init(self):
        algorithms = Full()
        indexer = recordlinkage.Index(algorithms)
        result = indexer.index(self.a, self.b)

        expected = Full().index(self.a, self.b)

        pdt.assert_index_equal(result, expected)

    def test_add_linking(self):
        indexer1 = Full()
        indexer2 = Block(left_on="var_arange", right_on="var_arange")
        expected = indexer1.index(self.a, self.b).union(indexer2.index(self.a, self.b))

        indexer = recordlinkage.Index()
        indexer.add([Full(), Block(left_on="var_arange", right_on="var_arange")])

        result = indexer.index(self.a, self.b)

        pdt.assert_index_equal(result, expected)

    def test_add_dedup(self):
        indexer1 = Full()
        indexer2 = Block(left_on="var_arange", right_on="var_arange")
        expected = indexer1.index(self.a).union(indexer2.index(self.a))

        indexer = recordlinkage.Index()
        indexer.add([Full(), Block(left_on="var_arange", right_on="var_arange")])

        result = indexer.index(self.a)

        pdt.assert_index_equal(result, expected)


class TestIndexAlgorithmApi(TestData):
    """General unittest for the indexing API."""

    @pytest.mark.parametrize("index_class", get_test_algorithms())
    def test_repr(self, index_class):
        index_str = str(index_class)
        index_repr = repr(index_class)
        assert index_str == index_repr

        start_str = f"<{index_class.__class__.__name__}"
        assert index_str.startswith(start_str)

    @pytest.mark.parametrize("index_class", get_test_algorithms())
    def test_arguments(self, index_class):
        """Test the index method arguments"""

        # The following should work
        index_class.index(self.a)
        index_class.index(self.a, self.b)
        index_class.index(self.a)
        index_class.index([self.a])
        index_class.index((self.a, self.b))
        index_class.index([self.a, self.b])
        index_class.index(x=(self.a, self.b))

    def test_iterative(self):
        """Test the iterative behaviour."""

        # SINGLE STEP
        index_class = Full()
        pairs = index_class.index((self.a, self.b))
        pairs = pd.DataFrame(index=pairs).sort_index()

        # MULTI STEP
        index_class = Full()

        pairs1 = index_class.index((self.a[0:50], self.b))
        pairs2 = index_class.index((self.a[50:100], self.b))

        pairs_split = pairs1.append(pairs2)
        pairs_split = pd.DataFrame(index=pairs_split).sort_index()

        pdt.assert_frame_equal(pairs, pairs_split)
        # note possible to sort MultiIndex, so made a frame out of it.

    @pytest.mark.parametrize("index_class", get_test_algorithms())
    def test_empty_imput_dataframes(self, index_class):
        """Empty DataFrames"""

        # make an empty dataframe with the columns of self.a and self.b
        df_a = pd.DataFrame(columns=self.a.columns.tolist())
        df_b = pd.DataFrame(columns=self.b.columns.tolist())

        from recordlinkage.index import Random

        if not isinstance(index_class, Random):
            # make an index
            pairs = index_class.index((df_a, df_b))

            # check if the MultiIndex has length 0
            assert isinstance(pairs, pd.MultiIndex)
            assert len(pairs) == 0
        else:
            with pytest.raises(ValueError):
                index_class.index((df_a, df_b))

    @pytest.mark.parametrize("index_class", get_test_algorithms())
    def test_error_handling(self, index_class):
        """Test error handling on non-unique index."""

        # make a non_unique index
        df_a = self.a.rename(index={self.a.index[1]: self.a.index[0]}, inplace=False)

        with pytest.raises(ValueError):
            index_class.index(df_a)

    @pytest.mark.parametrize("index_class", get_test_algorithms())
    def test_index_names_dedup(self, index_class):
        index_names = ["dedup", None, "index", int(1)]
        expected = [
            ["dedup_1", "dedup_2"],
            [None, None],
            ["index_1", "index_2"],
            ["1_1", "1_2"],
        ]

        for i, name in enumerate(index_names):
            index_A = pd.Index(self.a.index).rename(name)
            df_A = pd.DataFrame(self.a, index=index_A)

            pairs = index_class.index(df_A)

            assert pairs.names == expected[i]
            assert df_A.index.name == name

    @pytest.mark.parametrize("index_class", get_test_algorithms())
    def test_duplicated_index_names_dedup(self, index_class):
        # make an index for each dataframe with a new index name
        index_a = pd.Index(self.a.index, name="index")
        df_a = pd.DataFrame(self.a, index=index_a)

        # make the index
        pairs = index_class.index(df_a)
        assert pairs.names == ["index_1", "index_2"]

        # check for inplace editing (not the intention)
        assert df_a.index.name == "index"

        # make the index
        index_class.suffixes = ["_a", "_b"]
        pairs = index_class.index(df_a)
        assert pairs.names == ["index_a", "index_b"]

        # check for inplace editing (not the intention)
        assert df_a.index.name == "index"

    @pytest.mark.parametrize("index_class", get_test_algorithms())
    def test_index_names_link(self, index_class):
        # tuples with the name of the first and second index
        index_names = [
            ("index1", "index2"),
            ("index1", None),
            (None, "index2"),
            (None, None),
            (10, "index2"),
            (10, 11),
        ]

        for name_a, name_b in index_names:
            # make an index for each dataframe with a new index name
            index_a = pd.Index(self.a.index, name=name_a)
            df_a = pd.DataFrame(self.a, index=index_a)

            index_b = pd.Index(self.b.index, name=name_b)
            df_b = pd.DataFrame(self.b, index=index_b)

            pairs = index_class.index((df_a, df_b))
            assert pairs.names == [name_a, name_b]

            # check for inplace editing (not the intention)
            assert df_a.index.name == name_a
            assert df_b.index.name == name_b

    @pytest.mark.parametrize("index_class", get_test_algorithms())
    def test_duplicated_index_names_link(self, index_class):
        # make an index for each dataframe with a new index name
        index_a = pd.Index(self.a.index, name="index")
        df_a = pd.DataFrame(self.a, index=index_a)

        index_b = pd.Index(self.b.index, name="index")
        df_b = pd.DataFrame(self.b, index=index_b)

        # make the index
        pairs = index_class.index((df_a, df_b))
        assert pairs.names == ["index_1", "index_2"]

        # check for inplace editing (not the intention)
        assert df_a.index.name == "index"
        assert df_b.index.name == "index"

        # make the index
        index_class.suffixes = ["_a", "_b"]
        pairs = index_class.index((df_a, df_b))
        assert pairs.names == ["index_a", "index_b"]

        # check for inplace editing (not the intention)
        assert df_a.index.name == "index"
        assert df_b.index.name == "index"

    @pytest.mark.parametrize("index_class", get_test_algorithms())
    def test_index_names_pandas023(self, index_class):
        # Pandas changes the behaviour of MultiIndex names.
        # https://github.com/pandas-dev/pandas/pull/18882
        # https://github.com/J535D165/recordlinkage/issues/55
        # This test tests compatibility.

        # make an index for each dataframe with a new index name
        index_a = pd.Index(self.a.index, name="index")
        df_a = pd.DataFrame(self.a, index=index_a)

        index_b = pd.Index(self.b.index, name="index")
        df_b = pd.DataFrame(self.b, index=index_b)

        # make the index
        pairs_link = index_class._link_index(df_a, df_b)

        if pairs_link.names[0] is not None:
            assert pairs_link.names[0] != pairs_link.names[1]

        # make the index
        pairs_dedup = index_class._dedup_index(df_a)

        if pairs_link.names[0] is not None:
            assert pairs_dedup.names[0] != pairs_dedup.names[1]

    @pytest.mark.parametrize("index_class", get_test_algorithms())
    def test_pickle(self, index_class):
        """Test if it is possible to pickle the class."""

        pickle_path = os.path.join(self.test_dir, "pickle_compare_obj.pickle")

        # pickle before indexing
        pickle.dump(index_class, open(pickle_path, "wb"))

        # compute the record pairs
        index_class.index(self.a, self.b)

        # pickle after indexing
        pickle.dump(index_class, open(pickle_path, "wb"))

    @pytest.mark.parametrize("index_class", get_test_algorithms())
    def test_lower_triangular(self, index_class):
        # make an index for each dataframe with a new index name
        index_a = pd.Index(self.a.index, name="index")
        df_a = pd.DataFrame(self.a, index=index_a)
        pairs = index_class.index(df_a)

        # expected
        levels = [df_a.index.values, df_a.index.values]
        codes = np.tril_indices(len(df_a.index), k=-1)

        full_pairs = pd.MultiIndex(levels=levels, codes=codes, verify_integrity=False)

        # all pairs are in the lower triangle of the matrix.
        assert len(pairs.difference(full_pairs)) == 0


class TestFullIndexing(TestData):
    """General unittest for the full indexing class."""

    def test_basic_dedup(self):
        """FULL: Test basic characteristics of full indexing (dedup)."""

        from recordlinkage.index import Full

        # finding duplicates
        index_cl = Full()
        pairs = index_cl.index(self.a)

        assert isinstance(pairs, pd.MultiIndex)
        assert len(pairs) == len(self.a) * (len(self.a) - 1) / 2
        assert pairs.is_unique

    def test_basic_link(self):
        """FULL: Test basic characteristics of full indexing (link)."""

        from recordlinkage.index import Full

        # finding duplicates
        index_cl = Full()
        pairs = index_cl.index((self.a, self.b))

        assert isinstance(pairs, pd.MultiIndex)
        assert len(pairs) == len(self.a) * len(self.b)
        assert pairs.is_unique


class TestBlocking(TestData):
    """General unittest for the block indexing class."""

    def test_single_blocking_key(self):
        """BLOCKING: Test class arguments."""

        # all the following cases return in the same index.

        # situation 1
        index_cl1 = Block("var_arange")
        pairs1 = index_cl1.index((self.a, self.b))

        # situation 2
        index_cl2 = Block("var_arange")
        pairs2 = index_cl2.index((self.a, self.b))

        # situation 3
        index_cl3 = Block(left_on="var_arange", right_on="var_arange")
        pairs3 = index_cl3.index((self.a, self.b))

        # situation 4
        index_cl4 = Block(["var_arange"])
        pairs4 = index_cl4.index((self.a, self.b))

        # situation 5
        index_cl5 = Block(left_on=["var_arange"], right_on=["var_arange"])
        pairs5 = index_cl5.index((self.a, self.b))

        # test
        pdt.assert_index_equal(pairs1, pairs2)
        pdt.assert_index_equal(pairs1, pairs3)
        pdt.assert_index_equal(pairs1, pairs4)
        pdt.assert_index_equal(pairs1, pairs5)

    def test_multiple_blocking_keys(self):
        """BLOCKING: test multiple blocking keys"""

        # all the following cases return in the same index.

        # situation 1
        index_cl1 = Block(["var_arange", "var_block10"])
        pairs1 = index_cl1.index((self.a, self.b))

        # situation 2
        index_cl2 = Block(
            left_on=["var_arange", "var_block10"],
            right_on=["var_arange", "var_block10"],
        )
        pairs2 = index_cl2.index((self.a, self.b))

        # test
        pdt.assert_index_equal(pairs1, pairs2)

    def test_blocking_algorithm_link(self):
        """BLOCKING: test blocking algorithm for linking"""

        # situation 1: eye index
        index_cl1 = Block("var_arange")
        pairs1 = index_cl1.index((self.a, self.b))
        assert len(pairs1) == len(self.a)
        assert pairs1.is_unique

        # situation 2: 10 blocks
        index_cl2 = Block("var_block10")
        pairs2 = index_cl2.index((self.a, self.b))
        assert len(pairs2) == len(self.a) * 10
        assert pairs2.is_unique

        # situation 3: full index
        index_cl3 = Block("var_single")
        pairs3 = index_cl3.index((self.a, self.b))
        assert len(pairs3) == len(self.a) * len(self.b)
        assert pairs3.is_unique

    def test_blocking_algorithm_dedup(self):
        """BLOCKING: test blocking algorithm for deduplication"""

        len_a = len(self.a)

        # situation 1: eye index
        index_cl1 = Block("var_arange")
        pairs1 = index_cl1.index(self.a)
        assert len(pairs1) == 0
        assert pairs1.is_unique

        # situation 2: 10 blocks
        index_cl2 = Block("var_block10")
        pairs2 = index_cl2.index(self.a)
        assert len(pairs2) == (len_a * 10 - len_a) / 2
        assert pairs2.is_unique

        # situation 3: full index
        index_cl3 = Block("var_single")
        pairs3 = index_cl3.index(self.a)
        assert len(pairs3) == (len_a * len_a - len_a) / 2
        assert pairs3.is_unique

    def test_depr_on_argument(self):
        index_cl_new = Block("var_arange")
        pairs_new = index_cl_new.index(self.a)

        with pytest.deprecated_call():
            index_cl_old = Block(on="var_arange")
            pairs_old = index_cl_old.index(self.a)

        pdt.assert_index_equal(pairs_new, pairs_old)


class TestSortedNeighbourhoodIndexing(TestData):
    """General unittest for the sorted neighbourhood indexing class."""

    def test_single_sorting_key(self):
        """SNI: Test class arguments."""

        # all the following cases return in the same index.

        # situation 1
        index_cl1 = SortedNeighbourhood("var_arange")
        pairs1 = index_cl1.index((self.a, self.b))

        # situation 2
        index_cl2 = SortedNeighbourhood("var_arange")
        pairs2 = index_cl2.index((self.a, self.b))

        # situation 3
        index_cl3 = SortedNeighbourhood(left_on="var_arange", right_on="var_arange")
        pairs3 = index_cl3.index((self.a, self.b))

        # situation 4
        index_cl4 = SortedNeighbourhood(["var_arange"])
        pairs4 = index_cl4.index((self.a, self.b))

        # situation 5
        index_cl5 = SortedNeighbourhood(left_on=["var_arange"], right_on=["var_arange"])
        pairs5 = index_cl5.index((self.a, self.b))

        # test
        pdt.assert_index_equal(pairs1, pairs2)
        pdt.assert_index_equal(pairs1, pairs3)
        pdt.assert_index_equal(pairs1, pairs4)
        pdt.assert_index_equal(pairs1, pairs5)

    @pytest.mark.parametrize("window", [3, 5, 7, 9, 11])
    def test_sni_algorithm_link(self, window):
        """SNI: Test the window size (link)."""

        # window = 7 # using paramereized tests instead

        index_class = SortedNeighbourhood("var_arange", window=window)
        pairs = index_class.index((self.a, self.b[0 : len(self.a)]))

        # the expected number of pairs
        window_d = (window - 1) / 2
        len_a = len(self.a)
        n_pairs_expected = len(self.a) + 2 * np.sum(
            np.arange(len_a - 1, len_a - (window_d + 1), -1)
        )

        # test
        print("expected number of pairs: %s" % n_pairs_expected)
        print("number of pairs found: %s" % len(pairs))
        assert len(pairs) == n_pairs_expected

    @pytest.mark.parametrize("window", [3, 5, 7, 9, 11])
    def test_sni_algorithm_dedup(self, window):
        """SNI: Test the window size (dedup)."""

        # window = 7 # using paramereized tests instead

        index_class = SortedNeighbourhood("var_arange", window=window)
        pairs = index_class.index(self.a)

        # the expected number of pairs
        window_d = (window - 1) / 2
        len_a = len(self.a)
        n_pairs_expected = np.sum(np.arange(len_a - 1, len_a - (window_d + 1), -1))

        # test
        assert len(pairs) == n_pairs_expected

    def test_sni_with_blocking_link(self):
        """SNI: Test sni with blocking keys."""

        # sni
        index_class = SortedNeighbourhood("var_arange", window=3, block_on="var_arange")
        pairs = index_class.index((self.a, self.b[0 : len(self.a)]))

        # the length of pairs is length(self.a)
        assert len(pairs) == len(self.a)

    def test_sni_with_blocking_dedup(self):
        """SNI: Test sni with blocking keys."""

        # sni
        index_class = SortedNeighbourhood("var_arange", window=3, block_on="var_arange")
        pairs = index_class.index(self.a)

        print(pairs.values)

        # the length of pairs is 0
        assert len(pairs) == 0

    def test_depr_on_argument(self):
        index_cl_new = SortedNeighbourhood("var_arange")
        pairs_new = index_cl_new.index(self.a)

        with pytest.deprecated_call():
            index_cl_old = SortedNeighbourhood(on="var_arange")
            pairs_old = index_cl_old.index(self.a)

        pdt.assert_index_equal(pairs_new, pairs_old)

    def test_get_sorting_key_values(self):
        index_class = SortedNeighbourhood()
        array1 = [10, 1, 2, 1, 2, 1, 9, 8]
        array2 = [5, 5, 5, 3, 1, 4, 7, 10]
        res = index_class._get_sorting_key_values(array1, array2)
        res_sorted = np.sort(res)
        np.testing.assert_array_equal(res, res_sorted)


class TestRandomIndexing(TestData):
    """General unittest for the random indexing class."""

    def test_random_seed(self):
        """Random: test seeding random algorithm"""

        # TEST IDENTICAL
        index_cl1 = Random(n=1000, random_state=100)
        index_cl2 = Random(n=1000, random_state=100)
        index_cl3 = Random(n=1000, random_state=101)

        pairs1 = index_cl1.index((self.a, self.b))
        pairs2 = index_cl2.index((self.a, self.b))
        pairs3 = index_cl3.index((self.a, self.b))

        # are pairs1 and pairs2 indentical?
        pdt.assert_index_equal(pairs1, pairs2)

        # are pairs1 and pairs3 not indentical? # numpy workaround
        assert not np.array_equal(pairs1.values, pairs3.values)

    def test_random_without_replace(self):
        """Random: test random indexing without replacement"""

        # situation 1: linking
        index_cl1 = Random(n=1000, replace=False, random_state=100)

        pairs1 = index_cl1.index((self.a, self.b))
        assert len(pairs1) == 1000
        assert pairs1.is_unique

        # situation 2: dedup
        index_cl2 = Random(n=1000, replace=False, random_state=100)

        pairs2 = index_cl2.index(self.a)
        assert len(pairs2) == 1000
        assert pairs2.is_unique

    def test_random_with_replace(self):
        """Random: test random indexing with replacement"""

        # situation 1: linking
        index_cl1 = Random(n=1000, replace=True, random_state=100)

        pairs1 = index_cl1.index((self.a, self.b))
        assert len(pairs1) == 1000
        assert not pairs1.is_unique

        # situation 2: dedup
        index_cl2 = Random(n=1000, replace=True, random_state=101)

        pairs2 = index_cl2.index(self.a)
        assert len(pairs2) == 1000
        assert not pairs2.is_unique


@pytest.mark.parametrize(
    "n,shape",
    [(10, (3, 4)), (100, (50, 50)), (1000, (50, 50)), (10000, (10000, 10000))],
)
def test_random_without_replace_large(n, shape):
    """Random: test random indexing without replacement"""

    # Use hypothesis here

    pairs = random_pairs_without_replacement_low_memory(n, shape)

    assert isinstance(pairs, np.ndarray)
    assert len(pairs.shape) == 2
    assert pairs.shape[0] == 2


@pytest.mark.parametrize(
    "n,shape", [(10, (3, 4)), (100, (50, 50)), (1000, (50, 50)), (10000, (1000, 1000))]
)
def test_random_without_replace_small(n, shape):
    """Random: test random indexing without replacement"""

    # Use hypothesis here

    pairs = random_pairs_without_replacement(n, shape)

    assert isinstance(pairs, np.ndarray)
    assert len(pairs.shape) == 2
    assert pairs.shape[0] == 2


@pytest.mark.parametrize(
    "n,shape", [(10, (8,)), (100, (50,)), (1000, (50,)), (10000, (100000,))]
)
def test_random_without_replace_large_dedup(n, shape):
    """Random: test random indexing without replacement"""

    # Use hypothesis here

    pairs = random_pairs_without_replacement_low_memory(n, shape)

    assert isinstance(pairs, np.ndarray)
    assert len(pairs.shape) == 2
    assert pairs.shape[0] == 2


@pytest.mark.parametrize(
    "n,shape", [(10, (8,)), (100, (50,)), (1000, (50,)), (10000, (1000,))]
)
def test_random_without_replace_small_dedup(n, shape):
    """Random: test random indexing without replacement"""

    # Use hypothesis here

    pairs = random_pairs_without_replacement(n, shape)

    assert isinstance(pairs, np.ndarray)
    assert len(pairs.shape) == 2
    assert pairs.shape[0] == 2


def test_low_memory():
    df_a = pd.DataFrame(np.random.rand(1000000, 2))
    df_b = pd.DataFrame(np.random.rand(1000000, 2))

    pairs = Random(10, random_state=100, replace=False).index(df_a, df_b)

    assert is_pandas_2d_multiindex(pairs)
    assert len(pairs) == 10


def test_low_memory_seed():
    df_a = pd.DataFrame(np.random.rand(1000000, 2))
    df_b = pd.DataFrame(np.random.rand(1000000, 2))

    pairs1 = Random(10, random_state=100, replace=False).index(df_a, df_b)
    pairs2 = Random(10, random_state=100, replace=False).index(df_a, df_b)

    pdt.assert_index_equal(pairs1, pairs2)
