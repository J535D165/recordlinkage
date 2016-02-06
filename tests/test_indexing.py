from __future__ import print_function

import unittest

import pandas.util.testing as pdt
import recordlinkage
import numpy as np
import pandas as pd

from recordlinkage import datasets

class TestIndexing(unittest.TestCase):

    def test_full_index_unique(self):

        df1 = pd.DataFrame({'name':['Bob', 'Anne', 'Micheal']}, index=pd.Index(['001', '002', '003'], name='index_A'))
        df2 = pd.DataFrame({'name':['Bob', 'Anne', 'Micheal']}, index=pd.Index(['001', '002', '003'], name='index_B'))

        index = recordlinkage.Pairs(df1, df2)
        pairs = index.full()

        index_exp = [('001', '001'), ('001', '002'), ('001', '003'), ('002', '001'), ('002', '002'), ('002', '003'), ('003', '001'), ('003', '002'), ('003', '003')]

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), len(df1)*len(df2))

    def test_block_index_unique(self):

        df1 = pd.DataFrame({'name':['Bob', 'Anne', 'Micheal']}, index=pd.Index(['001', '002', '003'], name='index_A'))
        df2 = pd.DataFrame({'name':['Bob', 'Anne', 'Micheal']}, index=pd.Index(['001', '002', '003'], name='index_B'))

        index = recordlinkage.Pairs(df1, df2)
        pairs = index.block('name')

        index_exp = [('001', '001'), ('002', '002'),('003', '003')]

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), 3)

    def test_sortedneighbourhood_index_unique(self):

        df1 = pd.DataFrame({'name':['Bob', 'Anne', 'Micheal']}, index=pd.Index(['001', '002', '003'], name='index_A'))
        df2 = pd.DataFrame({'name':['Bob', 'Anne', 'Micheal']}, index=pd.Index(['001', '002', '003'], name='index_B'))

        index = recordlinkage.Pairs(df1, df2)
        pairs = index.sortedneighbourhood('name')

        index_exp = [('001', '001'), ('002', '002'),('003', '003')]

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

    def test_random_index_unique(self):

        df1 = pd.DataFrame({'name':['Bob', 'Anne', 'Micheal']}, index=pd.Index(['001', '002', '003'], name='index_A'))
        df2 = pd.DataFrame({'name':['Bob', 'Anne', 'Micheal']}, index=pd.Index(['001', '002', '003'], name='index_B'))

        index = recordlinkage.Pairs(df1, df2)
        pairs = index.random(5)

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

    def test_full_index_linking(self):

        dfA = datasets.load_censusA()
        dfB = datasets.load_censusB()

        index = recordlinkage.Pairs(dfA, dfB)
        pairs = index.full()

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertEqual(len(pairs), len(dfA)*len(dfB))

    def test_block_index_linking(self):

        dfA = datasets.load_censusA()
        dfB = datasets.load_censusB()

        index = recordlinkage.Pairs(dfA, dfB)
        pairs = index.block('last_name')

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

    def test_sorted_index_linking(self):

        dfA = datasets.load_censusA()
        dfB = datasets.load_censusB()

        index = recordlinkage.Pairs(dfA, dfB)
        pairs = index.sortedneighbourhood('last_name')

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

    def test_random_index_linking(self):

        dfA = datasets.load_censusA()
        dfB = datasets.load_censusB()

        index = recordlinkage.Pairs(dfA, dfB)
        pairs = index.random(1000)

        # Check if index is unique
        self.assertTrue(pairs.is_unique)

        # Check is number of pairs is correct
        self.assertTrue(len(pairs) <= 1000)

    def test_blocking_special_case_of_sorting(self):

        dfA = datasets.load_censusA()
        dfB = datasets.load_censusB()

        index = recordlinkage.Pairs(dfA, dfB)
        bl = index.block('last_name')
        sn = index.sortedneighbourhood('last_name', window=1)

        print('The number of record pairs found with blocking', len(bl))
        print('The number of record pairs found with sorted neighbourhood indexing', len(sn))

        self.assertEqual(len(bl), len(sn))

        union = bl.union(sn)
        print('The union of all indices of both methods', len(union))
        
        # The length of the union should be the same as the length of bl or sn.
        self.assertEqual(len(union), len(sn))

    def test_full_iter_index_linking(self):

        dfA = datasets.load_censusA()
        dfB = datasets.load_censusB()

        index = recordlinkage.Pairs(dfA, dfB)
        pairs_single = index.full()

        n_pairs_iter = 0
        for pairs in index.iterfull():
            n_pairs_iter += n_pairs_iter + len(pairs)

            # Check if index is unique
            self.assertTrue(pairs.is_unique)

        self.assertEqual(len(pairs_single), n_pairs_iter)

        # Check is number of pairs is correct
        self.assertEqual(n_pairs_iter, len(dfA)*len(dfB))

    def test_full_iter_index_deduplication(self):

        dfA = datasets.load_censusA()

        index = recordlinkage.Pairs(dfA)
        pairs_single = index.full()

        n_pairs_iter = 0
        for pairs in index.iterfull():
            n_pairs_iter += n_pairs_iter + len(pairs)

            # Check if index is unique
            self.assertTrue(pairs.is_unique)

        self.assertEqual(len(pairs_single), n_pairs_iter)

        # Check is number of pairs is correct
        self.assertEqual(n_pairs_iter, (len(dfA)-1)*len(dfA)/2)

    def test_reduction_ratio(self):

        dfA = datasets.load_censusA()
        dfB = datasets.load_censusB()

        index = recordlinkage.Pairs(dfA, dfB)
        pairs = index.full()

        rr = index.reduction()

        self.assertEqual(rr, 0)





