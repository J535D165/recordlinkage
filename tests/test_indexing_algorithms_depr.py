from __future__ import print_function

import unittest

from itertools import combinations_with_replacement, product

import recordlinkage as rl
import pandas as pd


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
        index_cl = rl.Pairs(A, B)
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
        index_cl = rl.Pairs(A, B)
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
        index_cl = rl.Pairs(A, B)
        pairs = index_cl.sortedneighbourhood('name')

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertTrue(len(pairs) <= len(A) * len(B))

    def test_sni_index_errors(self):

        A = pd.DataFrame(self.data_A, index=pd.Index(
            self.index, name='index_a'))
        B = pd.DataFrame(self.data_B, index=pd.Index(
            self.index, name='index_b'))

        # index full
        index_cl = rl.Pairs(A, B)

        with self.assertRaises(ValueError):
            index_cl.sortedneighbourhood('name', -3)

        with self.assertRaises(ValueError):
            index_cl.sortedneighbourhood('name', 2)

        with self.assertRaises(ValueError):
            index_cl.sortedneighbourhood('name', 'str')

        with self.assertRaises(ValueError):
            index_cl.sortedneighbourhood('name', 2.5)

    def test_random_index(self):

        A = pd.DataFrame(self.data_A, index=pd.Index(
            self.index, name='index_a'))
        B = pd.DataFrame(self.data_B, index=pd.Index(
            self.index, name='index_b'))

        # index block
        index_cl = rl.Pairs(A, B)
        pairs = index_cl.random(20)

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), 20)

        # index block
        index_cl = rl.Pairs(A, B)
        pairs = index_cl.random(5)

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), 5)

    def test_random_index_errors(self):

        A = pd.DataFrame(self.data_A, index=pd.Index(
            self.index, name='index_a'))
        B = pd.DataFrame(self.data_B, index=pd.Index(
            self.index, name='index_b'))

        # index block
        index_cl = rl.Pairs(A, B)

        # Check if index is unique
        with self.assertRaises(ValueError):
            index_cl.random(-10)

        # Check if index is unique
        with self.assertRaises(ValueError):
            index_cl.random(0)

        # Check if index is unique
        with self.assertRaises(ValueError):
            index_cl.random(5.5)

        # Check if index is unique
        with self.assertRaises(ValueError):
            index_cl.random(5.0)

        # Check if index is unique
        with self.assertRaises(ValueError):
            index_cl.random('str')

    def test_qgram_index(self):

        A = pd.DataFrame(self.data_A, index=pd.Index(
            self.index, name='index_a'))
        B = pd.DataFrame(self.data_B, index=pd.Index(
            self.index, name='index_b'))

        # index block
        index_cl = rl.Pairs(A, B)
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
        index_cl = rl.Pairs(A, B)

        bl = index_cl.block('name')
        sn = index_cl.sortedneighbourhood('name', window=1)

        print('The number of record pairs found with blocking', len(bl))
        print("The number of record pairs found with sorted " +
              "neighbourhood indexing", len(sn))

        # The length of the union should be the same as the length of bl or sn.
        self.assertEqual(len(bl), len(sn))