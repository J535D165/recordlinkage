from __future__ import print_function

import unittest

from itertools import combinations_with_replacement, product

import recordlinkage
import numpy as np
import pandas as pd

from recordlinkage import datasets

# Two larger numeric dataframes
df_large_numeric_1 = pd.DataFrame(np.arange(1000))
df_large_numeric_1.index.name = None
df_large_numeric_2 = pd.DataFrame(np.arange(1000))
df_large_numeric_2.index.name = None


# nosetests tests/test_indexing.py:TestIndexApi
class TestIndexApi(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.data_A = {
            'name': ['Bob', 'Anne', 'Micheal', 'Charly B', 'Ana'],
            'age': [40, 45, 69, 90, 70],
            'hometown': ['town 1', 'town 1', 'town 1', 'town 3', 'town 1']
        }
        self.data_B = {
            'name': ['Bob', 'Anne', 'Micheal', 'Charly', 'Ana'],
            'age': [40, 45, 68, 89, 70],
            'hometown': ['town 1', 'town 1', 'town 2', 'town 3', 'town 1']
        }

        self.index = ['rec1', 'rec2', 'rec3', 'rec4', 'rec5']

    def test_instance(self):

        A = pd.DataFrame(self.data_A)
        B = pd.DataFrame(self.data_B)

        index_cl = recordlinkage.Pairs(A, B)

        # index full
        pairs = index_cl.full()
        self.assertIsInstance(pairs, pd.MultiIndex)

        # prevent that the index name is changed
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

        # index block
        pairs = index_cl.block('name')
        self.assertIsInstance(pairs, pd.MultiIndex)
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

        # index sni
        pairs = index_cl.sortedneighbourhood('name', 3)
        self.assertIsInstance(pairs, pd.MultiIndex)
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

        # index eye
        pairs = index_cl.eye()
        self.assertIsInstance(pairs, pd.MultiIndex)
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

        # index random
        pairs = index_cl.random(3)
        self.assertIsInstance(pairs, pd.MultiIndex)
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

        # index qgram
        pairs = index_cl.qgram('name')
        self.assertIsInstance(pairs, pd.MultiIndex)
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

    def test_index_names_different(self):

        index_A = pd.Index(self.index, name='left')
        index_B = pd.Index(self.index, name='right')

        A = pd.DataFrame(self.data_A, index=index_A)
        B = pd.DataFrame(self.data_B, index=index_B)

        # index full
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.full()
        self.assertEqual(pairs.names, ['left', 'right'])
        self.assertEqual(A.index.name, 'left')
        self.assertEqual(B.index.name, 'right')

        # index block
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.block('name')
        self.assertEqual(pairs.names, ['left', 'right'])
        self.assertEqual(A.index.name, 'left')
        self.assertEqual(B.index.name, 'right')

        # index sni
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.sortedneighbourhood('name')
        self.assertEqual(pairs.names, ['left', 'right'])
        self.assertEqual(A.index.name, 'left')
        self.assertEqual(B.index.name, 'right')

        # index eye
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.eye()
        self.assertEqual(pairs.names, ['left', 'right'])
        self.assertEqual(A.index.name, 'left')
        self.assertEqual(B.index.name, 'right')

        # index random
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.random(3)
        self.assertEqual(pairs.names, ['left', 'right'])
        self.assertEqual(A.index.name, 'left')
        self.assertEqual(B.index.name, 'right')

        # index qgram
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.qgram('name')
        self.assertEqual(pairs.names, ['left', 'right'])
        self.assertEqual(A.index.name, 'left')
        self.assertEqual(B.index.name, 'right')

    def test_index_names_equal(self):

        index_A = pd.Index(self.index, name='leftright')
        index_B = pd.Index(self.index, name='leftright')

        A = pd.DataFrame(self.data_A, index=index_A)
        B = pd.DataFrame(self.data_B, index=index_B)

        # index full
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.full()
        self.assertEqual(pairs.names, ['leftright', 'leftright'])
        self.assertEqual(A.index.name, 'leftright')
        self.assertEqual(B.index.name, 'leftright')

        # index block
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.block('name')
        self.assertEqual(pairs.names, ['leftright', 'leftright'])
        self.assertEqual(A.index.name, 'leftright')
        self.assertEqual(B.index.name, 'leftright')

        # index sni
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.sortedneighbourhood('name')
        self.assertEqual(pairs.names, ['leftright', 'leftright'])
        self.assertEqual(A.index.name, 'leftright')
        self.assertEqual(B.index.name, 'leftright')

        # index eye
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.eye()
        self.assertEqual(pairs.names, ['leftright', 'leftright'])
        self.assertEqual(A.index.name, 'leftright')
        self.assertEqual(B.index.name, 'leftright')

        # index random
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.random(3)
        self.assertEqual(pairs.names, ['leftright', 'leftright'])
        self.assertEqual(A.index.name, 'leftright')
        self.assertEqual(B.index.name, 'leftright')

        # index random
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.qgram('name')
        self.assertEqual(pairs.names, ['leftright', 'leftright'])
        self.assertEqual(A.index.name, 'leftright')
        self.assertEqual(B.index.name, 'leftright')

    def test_index_names_none(self):

        index_A = pd.Index(self.index)
        index_B = pd.Index(self.index)

        A = pd.DataFrame(self.data_A, index=index_A)
        B = pd.DataFrame(self.data_B, index=index_B)

        # index full
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.full()
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

        # index block
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.block('name')
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

        # index sni
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.sortedneighbourhood('name')
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

        # index eye
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.eye()
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

        # index random
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.random(3)
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

        # index random
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.qgram('name')
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, None)

    def test_index_names_one_none(self):

        index_A = pd.Index(self.index)
        index_B = pd.Index(self.index)

        A = pd.DataFrame(self.data_A, index=index_A)
        B = pd.DataFrame(self.data_B, index=pd.Index(index_B, name='right'))

        # index full
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.full()
        self.assertEqual(pairs.names, [None, 'right'])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, 'right')

        # index block
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.block('name')
        self.assertEqual(pairs.names, [None, 'right'])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, 'right')

        # index sni
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.sortedneighbourhood('name')
        self.assertEqual(pairs.names, [None, 'right'])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, 'right')

        # index eye
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.eye()
        self.assertEqual(pairs.names, [None, 'right'])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, 'right')

        # index random
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.random(3)
        self.assertEqual(pairs.names, [None, 'right'])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, 'right')

        # index random
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.qgram('name')
        self.assertEqual(pairs.names, [None, 'right'])
        self.assertEqual(A.index.name, None)
        self.assertEqual(B.index.name, 'right')

    def test_dedupe_index_name_none(self):

        index_A = pd.Index(self.index)
        A = pd.DataFrame(self.data_A, index=index_A)

        # index full
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.full()
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)

        # index block
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.block('name')
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)

        # index sni
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.sortedneighbourhood('name')
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)

        # index eye
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.eye()
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)

        # index random
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.random(3)
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)

        # index random
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.qgram('name')
        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(A.index.name, None)

    def test_dedupe_index_name_not_none(self):

        index_A = pd.Index(self.index, name='dedup')
        A = pd.DataFrame(self.data_A, index=index_A)

        # index full
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.full()
        self.assertEqual(pairs.names, ['dedup', 'dedup'])
        self.assertEqual(A.index.name, 'dedup')

        # index block
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.block('name')
        self.assertEqual(pairs.names, ['dedup', 'dedup'])
        self.assertEqual(A.index.name, 'dedup')

        # index sni
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.sortedneighbourhood('name')
        self.assertEqual(pairs.names, ['dedup', 'dedup'])
        self.assertEqual(A.index.name, 'dedup')

        # index eye
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.eye()
        self.assertEqual(pairs.names, ['dedup', 'dedup'])
        self.assertEqual(A.index.name, 'dedup')

        # index random
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.random(3)
        self.assertEqual(pairs.names, ['dedup', 'dedup'])
        self.assertEqual(A.index.name, 'dedup')

        # index random
        index_cl = recordlinkage.Pairs(A)
        pairs = index_cl.qgram('name')
        self.assertEqual(pairs.names, ['dedup', 'dedup'])
        self.assertEqual(A.index.name, 'dedup')

    def test_reduction_ratio(self):

        index_A = pd.Index(self.index)
        index_B = pd.Index(self.index)

        A = pd.DataFrame(self.data_A, index=index_A)
        B = pd.DataFrame(self.data_B, index=index_B)

        index = recordlinkage.Pairs(A, B)
        index.full()

        rr = index.reduction

        self.assertEqual(rr, 0)

    def test_full_iter_index_linking(self):

        dfA, dfB = datasets.load_febrl4()

        index_chucks = recordlinkage.Pairs(dfA, dfB, chunks=(100, 200))
        index = recordlinkage.Pairs(dfA, dfB)

        # Compute pairs in one iteration
        pairs_single = index.full()

        # Compute pairs in iterations
        n_pairs_iter = 0
        for pairs in index_chucks.full():

            print (len(pairs))
            n_pairs_iter += len(pairs)

            # Check if index is unique
            self.assertTrue(pairs.is_unique)

        self.assertEqual(len(pairs_single), n_pairs_iter)

        # Check is number of pairs is correct
        self.assertEqual(n_pairs_iter, len(dfA) * len(dfB))

    def test_full_iter_index_deduplication(self):

        dfA = datasets.load_febrl1()

        # Compute pairs in one iteration
        index = recordlinkage.Pairs(dfA)
        pairs_single = index.full()

        # Compute pairs in iterations
        n_pairs_iter = 0

        index_chucks = recordlinkage.Pairs(dfA, chunks=100)

        for pairs in index_chucks.full():

            print (len(pairs))
            n_pairs_iter += len(pairs)

            # Check if index is unique
            self.assertTrue(pairs.is_unique)

        self.assertEqual(len(pairs_single), n_pairs_iter)

        # Check is number of pairs is correct
        self.assertEqual(n_pairs_iter, (len(dfA) - 1) * len(dfA) / 2)


# nosetests tests/test_indexing.py:TestIndexAlgorithms
class TestIndexAlgorithms(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.data_A = {
            'name': ['Bob', 'Anne', 'Micheal', 'Charly B', 'Ana'],
            'age': [40, 45, 69, 90, 70],
            'hometown': ['town 1', 'town 1', 'town 1', 'town 3', 'town 1']
        }
        self.data_B = {
            'name': ['Bob', 'Anne', 'Micheal', 'Charly', 'Ana'],
            'age': [40, 45, 68, 89, 70],
            'hometown': ['town 1', 'town 1', 'town 2', 'town 3', 'town 1']
        }

        self.index = ['rec1', 'rec2', 'rec3', 'rec4', 'rec5']

        self.fullindex_dedup = list(
            combinations_with_replacement(self.index, 2))
        self.fullindex = list(product(self.index, self.index))

    def test_full_index(self):

        A = pd.DataFrame(
            self.data_A,
            index=pd.Index(self.index, name='index_a'))
        B = pd.DataFrame(
            self.data_B,
            index=pd.Index(self.index, name='index_b'))

        # index full
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.full()

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), len(self.fullindex))

    def test_block_index(self):

        A = pd.DataFrame(self.data_A, index=pd.Index(
            self.index, name='index_a'))
        B = pd.DataFrame(self.data_B, index=pd.Index(
            self.index, name='index_b'))

        # index block
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.block('name')

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), 4)

    def test_sni_index(self):

        A = pd.DataFrame(self.data_A, index=pd.Index(
            self.index, name='index_a'))
        B = pd.DataFrame(self.data_B, index=pd.Index(
            self.index, name='index_b'))

        # index full
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.sortedneighbourhood('name')

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertTrue(len(pairs) <= len(A) * len(B))

    def test_random_index(self):

        A = pd.DataFrame(self.data_A, index=pd.Index(
            self.index, name='index_a'))
        B = pd.DataFrame(self.data_B, index=pd.Index(
            self.index, name='index_b'))

        # index block
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.random(5)

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), 5)

        # index block
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.random(2)

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), 2)

    def test_qgram_index(self):

        A = pd.DataFrame(self.data_A, index=pd.Index(
            self.index, name='index_a'))
        B = pd.DataFrame(self.data_B, index=pd.Index(
            self.index, name='index_b'))

        # index block
        index_cl = recordlinkage.Pairs(A, B)
        pairs = index_cl.qgram('name')

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertTrue(len(pairs) <= len(A) * len(B))

    def test_blocking_special_case_of_sorting(self):

        A = pd.DataFrame(self.data_A, index=pd.Index(
            self.index, name='index_a'))
        B = pd.DataFrame(self.data_B, index=pd.Index(
            self.index, name='index_b'))

        # index block
        index_cl = recordlinkage.Pairs(A, B)

        bl = index_cl.block('name')
        sn = index_cl.sortedneighbourhood('name', window=1)

        print('The number of record pairs found with blocking', len(bl))
        print("The number of record pairs found with sorted " +
              "neighbourhood indexing", len(sn))

        # The length of the union should be the same as the length of bl or sn.
        self.assertEqual(len(bl), len(sn))


# nosetests tests/test_indexing.py:TestIndexOnDatasets
class TestIndexOnDatasets(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.A, self.B = datasets.load_febrl4()

    def test_reduction_ratio(self):

        index = recordlinkage.Pairs(self.A, self.B)
        index.full()

        rr = index.reduction

        self.assertEqual(rr, 0)

    def test_random_index_linking(self):

        index = recordlinkage.Pairs(self.A, self.B)
        pairs = index.random(1000)

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertTrue(len(pairs) <= 1000)

    def test_full_index_linking(self):

        index = recordlinkage.Pairs(self.A, self.B)
        pairs = index.full()

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), len(self.A) * len(self.B))

    def test_block_index_linking(self):

        index = recordlinkage.Pairs(self.A, self.B)
        pairs = index.block('given_name')

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

    def test_qgram_index_linking(self):

        index = recordlinkage.Pairs(self.A, self.B)

        index = recordlinkage.Pairs(self.A[0:100], self.B[0:100])
        pairs = index.qgram('given_name')

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

    def test_sorted_index_linking(self):

        index = recordlinkage.Pairs(self.A, self.B)
        pairs = index.sortedneighbourhood('given_name')

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

    def test_small_random_index(self):

        n_pairs = 10000

        index = recordlinkage.Pairs(self.A, self.B)
        pairs = index.random(n_pairs)

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), n_pairs)

    def test_large_random_index(self):

        n_pairs = 300000

        index = recordlinkage.Pairs(self.A, self.B)
        pairs = index.random(n_pairs)

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), n_pairs)
