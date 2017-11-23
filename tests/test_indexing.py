from __future__ import print_function

import os
import unittest
import tempfile
import shutil
import pickle

import numpy as np
import pandas as pd
import pandas.util.testing as ptm
from parameterized import parameterized, param

import recordlinkage


TEST_INDEXATION_OBJECTS = [
    param(recordlinkage.FullIndex()),
    param(recordlinkage.BlockIndex(on='var_arange')),
    param(recordlinkage.SortedNeighbourhoodIndex(on='var_arange')),
    param(recordlinkage.RandomIndex(10, random_state=100, replace=True)),
    param(recordlinkage.RandomIndex(10, random_state=100, replace=False)),
]


class TestData(unittest.TestCase):
    """Unittest object to setup test data."""

    @classmethod
    def setUpClass(cls):

        n_a = 100
        n_b = 150

        cls.index_a = ['rec_a_%s' % i for i in range(0, n_a)]
        cls.index_b = ['rec_b_%s' % i for i in range(0, n_b)]

        cls.a = pd.DataFrame({
            'var_single': np.repeat([1], n_a),
            'var_arange': np.arange(n_a),
            'var_arange_str': np.arange(n_a),
            'var_block10': np.repeat(np.arange(n_a / 10), 10)
        }, index=cls.index_a)

        cls.b = pd.DataFrame({
            'var_single': np.repeat([1], n_b),
            'var_arange': np.arange(n_b),
            'var_arange_str': np.arange(n_b),
            'var_block10': np.repeat(np.arange(n_b / 10), 10)
        }, index=cls.index_b)

        # Create a temporary directory
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):

        # Remove the test directory
        shutil.rmtree(cls.test_dir)


# tests/test_indexing.py:TestIndexApi
class TestIndexApi(TestData):
    """General unittest for the indexing API."""

    @parameterized.expand(TEST_INDEXATION_OBJECTS)
    def test_arguments(self, index_class):
        """Test the index method arguments"""

        # The following should work
        index_class.index(self.a)
        index_class.index(self.a, self.b)
        index_class.index((self.a))
        index_class.index([self.a])
        index_class.index((self.a, self.b))
        index_class.index([self.a, self.b])
        index_class.index(x=(self.a, self.b))

    def test_iterative(self):
        """Test the iterative behaviour."""

        # SINGLE STEP
        index_class = recordlinkage.FullIndex()
        pairs = index_class.index((self.a, self.b))
        pairs = pd.DataFrame(index=pairs).sort_index()

        # MULTI STEP
        index_class = recordlinkage.FullIndex()

        pairs1 = index_class.index((self.a[0:50], self.b))
        pairs2 = index_class.index((self.a[50:100], self.b))

        pairs_split = pairs1.append(pairs2)
        pairs_split = pd.DataFrame(index=pairs_split).sort_index()

        ptm.assert_frame_equal(pairs, pairs_split)
        # note possible to sort MultiIndex, so made a frame out of it.

    @parameterized.expand(TEST_INDEXATION_OBJECTS)
    def test_empty_imput_dataframes(self, index_class):
        """Empty DataFrames"""

        # make an empty dataframe with the columns of self.a and self.b
        df_a = pd.DataFrame(columns=self.a.columns.tolist())
        df_b = pd.DataFrame(columns=self.b.columns.tolist())

        if not isinstance(index_class, recordlinkage.RandomIndex):
            # make an index
            pairs = index_class.index((df_a, df_b))

            # check if the MultiIndex has length 0
            self.assertIsInstance(pairs, pd.MultiIndex)
            self.assertEqual(len(pairs), 0)
        else:
            with self.assertRaises(ValueError):
                index_class.index((df_a, df_b))

    @parameterized.expand(TEST_INDEXATION_OBJECTS)
    def test_error_handling(self, index_class):
        """Test error handling on non-unique index."""

        # make a non_unique index
        df_a = self.a.rename(
            index={self.a.index[1]: self.a.index[0]}, inplace=False)

        with self.assertRaises(ValueError):
            index_class.index(df_a)

    @parameterized.expand(TEST_INDEXATION_OBJECTS)
    def test_index_names_dedup(self, index_class):
        """Test names of MultiIndex is case of dedup."""

        index_names = ['dedup', None, 'index', int(1), float(1.00)]

        for name in index_names:

            index_A = pd.Index(self.a.index).rename(name)
            df_A = pd.DataFrame(self.a, index=index_A)

            pairs = index_class.index((df_A))

            self.assertEqual(pairs.names, [name, name])
            self.assertEqual(df_A.index.name, name)

    @parameterized.expand(TEST_INDEXATION_OBJECTS)
    def test_index_names_link(self, index_class):
        """Test names of MultiIndex is case of dedup."""

        # tuples with the name of the first and second index
        index_names = [
            ('index1', 'index2'),
            ('index', 'index'),
            ('index1', None),
            (None, 'index2'),
            (None, None),
            (10, 'index2'),
            (10, 11)
        ]

        for name_a, name_b in index_names:

            # make an index for each dataframe with a new index name
            index_a = pd.Index(self.a.index).rename(name_a)
            index_b = pd.Index(self.b.index).rename(name_b)

            df_a = pd.DataFrame(self.a, index=index_a)
            df_b = pd.DataFrame(self.b, index=index_b)

            # make the index
            pairs = index_class.index((df_a, df_b))

            # test the names
            self.assertEqual(pairs.names, [name_a, name_b])

            # check for inplace editing (not the intention)
            self.assertEqual(df_a.index.name, name_a)
            self.assertEqual(df_b.index.name, name_b)

    @parameterized.expand(TEST_INDEXATION_OBJECTS)
    def test_pickle(self, index_class):
        """Test if it is possible to pickle the class."""

        pickle_path = os.path.join(self.test_dir, 'pickle_compare_obj.pickle')

        # pickle before indexing
        pickle.dump(index_class, open(pickle_path, 'wb'))

        # compute the record pairs
        index_class.index(self.a, self.b)

        # pickle after indexing
        pickle.dump(index_class, open(pickle_path, 'wb'))


# tests/test_indexing.py:TestFullIndexing
class TestFullIndexing(TestData):
    """General unittest for the full indexing class."""

    def test_basic_dedup(self):
        """FULL: Test basic characteristics of full indexing (dedup)."""

        # finding duplicates
        index_cl = recordlinkage.FullIndex()
        pairs = index_cl.index(self.a)

        self.assertIsInstance(pairs, pd.MultiIndex)
        self.assertEqual(len(pairs), len(self.a) * (len(self.a) - 1) / 2)
        self.assertTrue(pairs.is_unique)

    def test_basic_link(self):
        """FULL: Test basic characteristics of full indexing (link)."""

        # finding duplicates
        index_cl = recordlinkage.FullIndex()
        pairs = index_cl.index((self.a, self.b))

        self.assertIsInstance(pairs, pd.MultiIndex)
        self.assertEqual(len(pairs), len(self.a) * len(self.b))
        self.assertTrue(pairs.is_unique)


# tests/test_indexing.py:TestBlocking
class TestBlocking(TestData):
    """General unittest for the block indexing class."""

    def test_single_blocking_key(self):
        """BLOCKING: Test class arguments."""

        # all the following cases return in the same index.

        # situation 1
        index_cl1 = recordlinkage.BlockIndex('var_arange')
        pairs1 = index_cl1.index((self.a, self.b))

        # situation 2
        index_cl2 = recordlinkage.BlockIndex(on='var_arange')
        pairs2 = index_cl2.index((self.a, self.b))

        # situation 3
        index_cl3 = recordlinkage.BlockIndex(
            left_on='var_arange', right_on='var_arange')
        pairs3 = index_cl3.index((self.a, self.b))

        # situation 4
        index_cl4 = recordlinkage.BlockIndex(on=['var_arange'])
        pairs4 = index_cl4.index((self.a, self.b))

        # situation 5
        index_cl5 = recordlinkage.BlockIndex(
            left_on=['var_arange'], right_on=['var_arange'])
        pairs5 = index_cl5.index((self.a, self.b))

        # test
        ptm.assert_index_equal(pairs1, pairs2)
        ptm.assert_index_equal(pairs1, pairs3)
        ptm.assert_index_equal(pairs1, pairs4)
        ptm.assert_index_equal(pairs1, pairs5)

    def test_multiple_blocking_keys(self):
        """BLOCKING: test multiple blocking keys"""

        # all the following cases return in the same index.

        # situation 1
        index_cl1 = recordlinkage.BlockIndex(['var_arange', 'var_block10'])
        pairs1 = index_cl1.index((self.a, self.b))

        # situation 2
        index_cl2 = recordlinkage.BlockIndex(
            left_on=['var_arange', 'var_block10'],
            right_on=['var_arange', 'var_block10']
        )
        pairs2 = index_cl2.index((self.a, self.b))

        # test
        ptm.assert_index_equal(pairs1, pairs2)

    def test_blocking_algorithm_link(self):
        """BLOCKING: test blocking algorithm for linking"""

        # situation 1: eye index
        index_cl1 = recordlinkage.BlockIndex(on='var_arange')
        pairs1 = index_cl1.index((self.a, self.b))
        self.assertEqual(len(pairs1), len(self.a))
        self.assertTrue(pairs1.is_unique)

        # situation 2: 10 blocks
        index_cl2 = recordlinkage.BlockIndex(on='var_block10')
        pairs2 = index_cl2.index((self.a, self.b))
        self.assertEqual(len(pairs2), len(self.a) * 10)
        self.assertTrue(pairs2.is_unique)

        # situation 3: full index
        index_cl3 = recordlinkage.BlockIndex(on='var_single')
        pairs3 = index_cl3.index((self.a, self.b))
        self.assertEqual(len(pairs3), len(self.a) * len(self.b))
        self.assertTrue(pairs3.is_unique)

    def test_blocking_algorithm_dedup(self):
        """BLOCKING: test blocking algorithm for deduplication"""

        len_a = len(self.a)

        # situation 1: eye index
        index_cl1 = recordlinkage.BlockIndex(on='var_arange')
        pairs1 = index_cl1.index(self.a)
        self.assertEqual(len(pairs1), 0)
        self.assertTrue(pairs1.is_unique)

        # situation 2: 10 blocks
        index_cl2 = recordlinkage.BlockIndex(on='var_block10')
        pairs2 = index_cl2.index(self.a)
        self.assertEqual(len(pairs2), (len_a * 10 - len_a) / 2)
        self.assertTrue(pairs2.is_unique)

        # situation 3: full index
        index_cl3 = recordlinkage.BlockIndex(on='var_single')
        pairs3 = index_cl3.index(self.a)
        self.assertEqual(len(pairs3), (len_a * len_a - len_a) / 2)
        self.assertTrue(pairs3.is_unique)


# tests/test_indexing.py:TestSortedNeighbourhoodIndexing
class TestSortedNeighbourhoodIndexing(TestData):
    """General unittest for the sorted neighbourhood indexing class."""

    def test_single_sorting_key(self):
        """SNI: Test class arguments."""

        # all the following cases return in the same index.

        # situation 1
        index_cl1 = recordlinkage.SortedNeighbourhoodIndex('var_arange')
        pairs1 = index_cl1.index((self.a, self.b))

        # situation 2
        index_cl2 = recordlinkage.SortedNeighbourhoodIndex(on='var_arange')
        pairs2 = index_cl2.index((self.a, self.b))

        # situation 3
        index_cl3 = recordlinkage.SortedNeighbourhoodIndex(
            left_on='var_arange', right_on='var_arange')
        pairs3 = index_cl3.index((self.a, self.b))

        # situation 4
        index_cl4 = recordlinkage.SortedNeighbourhoodIndex(on=['var_arange'])
        pairs4 = index_cl4.index((self.a, self.b))

        # situation 5
        index_cl5 = recordlinkage.SortedNeighbourhoodIndex(
            left_on=['var_arange'], right_on=['var_arange'])
        pairs5 = index_cl5.index((self.a, self.b))

        # test
        ptm.assert_index_equal(pairs1, pairs2)
        ptm.assert_index_equal(pairs1, pairs3)
        ptm.assert_index_equal(pairs1, pairs4)
        ptm.assert_index_equal(pairs1, pairs5)

    @parameterized.expand([
        (3,),
        (5,),
        (7,),
        (9,),
        (11,)
    ])
    def test_sni_algorithm_link(self, window):
        """SNI: Test the window size (link)."""

        # window = 7 # using paramereized tests instead

        index_class = recordlinkage.SortedNeighbourhoodIndex(
            on='var_arange', window=window)
        pairs = index_class.index((self.a, self.b[0:len(self.a)]))

        # the expected number of pairs
        window_d = (window - 1) / 2
        len_a = len(self.a)
        n_pairs_expected = \
            len(self.a) + \
            2 * np.sum(np.arange(len_a - 1, len_a - (window_d + 1), -1))

        # test
        print ('expected number of pairs: %s' % n_pairs_expected)
        print ('number of pairs found: %s' % len(pairs))
        self.assertEqual(len(pairs), n_pairs_expected)

    @parameterized.expand([
        (3,),
        (5,),
        (7,),
        (9,),
        (11,)
    ])
    def test_sni_algorithm_dedup(self, window):
        """SNI: Test the window size (dedup)."""

        # window = 7 # using paramereized tests instead

        index_class = recordlinkage.SortedNeighbourhoodIndex(
            on='var_arange', window=window)
        pairs = index_class.index((self.a))

        # the expected number of pairs
        window_d = (window - 1) / 2
        len_a = len(self.a)
        n_pairs_expected = \
            np.sum(np.arange(len_a - 1, len_a - (window_d + 1), -1))

        # test
        self.assertEqual(len(pairs), n_pairs_expected)

    def test_sni_with_blocking_link(self):
        """SNI: Test sni with blocking keys."""

        # sni
        index_class = recordlinkage.SortedNeighbourhoodIndex(
            on='var_arange', window=3, block_on='var_arange')
        pairs = index_class.index((self.a, self.b[0:len(self.a)]))

        # the length of pairs is length(self.a)
        self.assertEqual(len(pairs), len(self.a))

    def test_sni_with_blocking_dedup(self):
        """SNI: Test sni with blocking keys."""

        # sni
        index_class = recordlinkage.SortedNeighbourhoodIndex(
            on='var_arange', window=3, block_on='var_arange')
        pairs = index_class.index(self.a)

        print (pairs.values)

        # the length of pairs is 0
        self.assertEqual(len(pairs), 0)


# tests/test_indexing.py:TestRandomIndexing
class TestRandomIndexing(TestData):
    """General unittest for the random indexing class."""

    def test_random_seed(self):
        """Random: test seeding random algorithm"""

        # TEST IDENTICAL
        index_cl1 = recordlinkage.RandomIndex(n=1000, random_state=100)
        index_cl2 = recordlinkage.RandomIndex(n=1000, random_state=100)
        index_cl3 = recordlinkage.RandomIndex(n=1000, random_state=101)

        pairs1 = index_cl1.index((self.a, self.b))
        pairs2 = index_cl2.index((self.a, self.b))
        pairs3 = index_cl3.index((self.a, self.b))

        # are pairs1 and pairs2 indentical?
        ptm.assert_index_equal(pairs1, pairs2)

        # are pairs1 and pairs3 not indentical? # numpy workaround
        self.assertFalse(np.array_equal(pairs1.values, pairs3.values))

    def test_random_without_replace(self):
        """Random: test random indexing without replacement"""

        # situation 1: linking
        index_cl1 = recordlinkage.RandomIndex(
            n=1000, replace=False, random_state=100
        )

        pairs1 = index_cl1.index((self.a, self.b))
        self.assertEqual(len(pairs1), 1000)
        self.assertTrue(pairs1.is_unique)

        # situation 2: dedup
        index_cl2 = recordlinkage.RandomIndex(
            n=1000, replace=False, random_state=100
        )

        pairs2 = index_cl2.index(self.a)
        self.assertEqual(len(pairs2), 1000)
        self.assertTrue(pairs2.is_unique)

    def test_random_with_replace(self):
        """Random: test random indexing with replacement"""

        # situation 1: linking
        index_cl1 = recordlinkage.RandomIndex(
            n=1000, replace=True, random_state=100
        )

        pairs1 = index_cl1.index((self.a, self.b))
        self.assertEqual(len(pairs1), 1000)
        self.assertFalse(pairs1.is_unique)

        # situation 2: dedup
        index_cl2 = recordlinkage.RandomIndex(
            n=1000, replace=True, random_state=101
        )

        pairs2 = index_cl2.index(self.a)
        self.assertEqual(len(pairs2), 1000)
        self.assertFalse(pairs2.is_unique)
