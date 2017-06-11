from __future__ import print_function

import unittest

import numpy as np
import pandas as pd
from nose_parameterized import parameterized, param

import recordlinkage
from recordlinkage.utils import IndexError

# Two larger numeric dataframes
df_large_numeric_1 = pd.DataFrame(np.arange(1000))
df_large_numeric_1.index.name = None
df_large_numeric_2 = pd.DataFrame(np.arange(1000))
df_large_numeric_2.index.name = None


INDEX_ALGORITHMS = [

    # full
    param('full'),

    # blocking
    param('block', 'name'),
    param('block', left_on='name', right_on='name'),
    param('block', left_on=['name'], right_on=['name']),
    param('block', left_on=['name', 'hometown'],
          right_on=['name', 'hometown']),
    param('block', left_on=['name', 'hometown'], right_on=['name', 'name']),

    # sni
    param('sortedneighbourhood', 'name', 3),
    param('sortedneighbourhood', 'name', 5),
    param('sortedneighbourhood', 'name', 49),
    param('sortedneighbourhood', 'name', 3, block_on='hometown'),

    # dev
    param('eye'),
    param('random', 3),  # random_small
    param('random', 3000),  # random_large

    # qgram
    param('qgram', 'name')

]

dA = pd.DataFrame({
    'name': ['Bob', 'Anne', 'Micheal', 'Charly B', 'Ana'],
    'age': [40, 45, 69, 90, 70],
    'hometown': ['town 1', 'town 1', 'town 1', 'town 3', 'town 1']
})

dB = pd.DataFrame({
    'name': ['Bob', 'Anne', 'Micheal', 'Charly', 'Ana'],
    'age': [40, 45, 68, 89, 70],
    'hometown': ['town 1', 'town 1', 'town 2', 'town 3', 'town 1']
})


# nosetests -v tests/test_indexing.py
class TestIndexApi(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        n_a = 100
        n_b = 150

        self.A = {
            'name': np.random.choice(dA['name'], n_a),
            'age': np.random.choice(dA['age'], n_a),
            'hometown': np.random.choice(dA['hometown'], n_a)
        }

        self.B = {
            'name': np.random.choice(dB['name'], n_b),
            'age': np.random.choice(dB['age'], n_b),
            'hometown': np.random.choice(dB['hometown'], n_b)
        }

        self.index_a = ['rec%s' % i for i in range(0, n_a)]
        self.index_b = ['rec%s' % i for i in range(0, n_b)]

        self.bad_index_a = np.random.choice(
            ['rec1', 'rec1', 'rec3', 'rec4', 'rec5'],
            n_a
        )


# nosetests -v tests/test_indexing.py:TestIndexApiDedup
class TestIndexApiDedup(TestIndexApi):

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_instance_dedup(self, method_to_call, *args, **kwargs):

        df_A = pd.DataFrame(self.A)

        index_cl = recordlinkage.Pairs(df_A)
        pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

        self.assertIsInstance(pairs, pd.MultiIndex)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_empty_dataframes_dedup(self, method_to_call, *args, **kwargs):

        df_A = pd.DataFrame(columns=self.A.keys())

        if method_to_call != 'random':

            index_cl = recordlinkage.Pairs(df_A)
            pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

            self.assertIsInstance(pairs, pd.MultiIndex)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_amount_of_pairs_dedup(self, method_to_call, *args, **kwargs):

        df_A = pd.DataFrame(self.A)

        index_cl = recordlinkage.Pairs(df_A)
        pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

        # Test is the amount of pairs is less than A*B
        self.assertTrue(len(pairs) <= len(df_A) * (len(df_A) - 1) / 2)

    def test_index_unique_dedup(self):

        index_A = pd.Index(self.bad_index_a, name='left')
        A = pd.DataFrame(self.A, index=index_A)

        with self.assertRaises(IndexError):
            recordlinkage.Pairs(A)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_full_iter_index_dedup(self, method_to_call, *args, **kwargs):

        df_A = pd.DataFrame(self.A)

        index_cl = recordlinkage.Pairs(df_A, chunks=100)
        index = recordlinkage.Pairs(df_A)

        # Compute pairs in one iteration
        pairs_single = getattr(index, method_to_call)(*args, **kwargs)

        # Compute pairs in iterations
        n_pairs_iter = 0
        for pairs in getattr(index_cl, method_to_call)(*args, **kwargs):

            print (len(pairs))
            n_pairs_iter += len(pairs)

            # Check if index is unique
            self.assertTrue(pairs.is_unique)

        self.assertEqual(len(pairs_single), n_pairs_iter)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_index_names_none_dedup(self, method_to_call, *args, **kwargs):

        # setup A
        index_A = pd.Index(self.index_a)
        df_A = pd.DataFrame(self.A, index=index_A)

        # Make pairs
        index_cl = recordlinkage.Pairs(df_A)
        pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(df_A.index.name, None)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_index_names_not_none_dedup(self, method_to_call, *args, **kwargs):

        # setup A
        index_A = pd.Index(self.index_a, name='dedup')
        df_A = pd.DataFrame(self.A, index=index_A)

        # Make pairs
        index_cl = recordlinkage.Pairs(df_A)
        pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

        self.assertEqual(pairs.names, ['dedup', 'dedup'])
        self.assertEqual(df_A.index.name, 'dedup')

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_reduction_ratio_dedup(self, method_to_call, *args, **kwargs):

        df_A = pd.DataFrame(self.A)

        index_cl = recordlinkage.Pairs(df_A)
        getattr(index_cl, method_to_call)(*args, **kwargs)

        rr = index_cl.reduction

        self.assertTrue((rr >= 0.0) & (rr <= 1.0))


# nosetests -v tests/test_indexing.py:TestIndexApiLink
class TestIndexApiLink(TestIndexApi):

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_instance_linking(self, method_to_call, *args, **kwargs):

        df_A = pd.DataFrame(self.A)
        df_B = pd.DataFrame(self.B)

        index_cl = recordlinkage.Pairs(df_A, df_B)
        pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

        self.assertIsInstance(pairs, pd.MultiIndex)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_empty_dataframes_linking(self, method_to_call, *args, **kwargs):

        df_A = pd.DataFrame(columns=self.A.keys())
        df_B = pd.DataFrame(columns=self.B.keys())

        if method_to_call != 'random':

            index_cl = recordlinkage.Pairs(df_A, df_B)
            pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

            self.assertIsInstance(pairs, pd.MultiIndex)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_index_names_linking(self, method_to_call, *args, **kwargs):
        """ Check modification of index name """

        df_A = pd.DataFrame(self.A)
        df_B = pd.DataFrame(self.B)

        index_cl = recordlinkage.Pairs(df_A, df_B)
        getattr(index_cl, method_to_call)(*args, **kwargs)

        # prevent that the index name is changed
        self.assertEqual(df_A.index.name, None)
        self.assertEqual(df_B.index.name, None)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_index_names_diff_linking(self, method_to_call, *args, **kwargs):

        # setup A
        index_A = pd.Index(self.index_a, name='left')
        df_A = pd.DataFrame(self.A, index=index_A)

        # setup B
        index_B = pd.Index(self.index_b, name='right')
        df_B = pd.DataFrame(self.B, index=index_B)

        # Make pairs
        index_cl = recordlinkage.Pairs(df_A, df_B)
        pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

        self.assertEqual(pairs.names, ['left', 'right'])
        self.assertEqual(df_A.index.name, 'left')
        self.assertEqual(df_B.index.name, 'right')

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_index_names_eq_linking(self, method_to_call, *args, **kwargs):

        # setup A
        index_A = pd.Index(self.index_a, name='leftright')
        df_A = pd.DataFrame(self.A, index=index_A)

        # setup B
        index_B = pd.Index(self.index_b, name='leftright')
        df_B = pd.DataFrame(self.B, index=index_B)

        # Make pairs
        index_cl = recordlinkage.Pairs(df_A, df_B)
        pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

        self.assertEqual(pairs.names, ['leftright', 'leftright'])
        self.assertEqual(df_A.index.name, 'leftright')
        self.assertEqual(df_B.index.name, 'leftright')

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_index_names_none_linking(self, method_to_call, *args, **kwargs):

        # setup A
        index_A = pd.Index(self.index_a)
        df_A = pd.DataFrame(self.A, index=index_A)

        # setup B
        index_B = pd.Index(self.index_b)
        df_B = pd.DataFrame(self.B, index=index_B)

        # Make pairs
        index_cl = recordlinkage.Pairs(df_A, df_B)
        pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

        self.assertEqual(pairs.names, [None, None])
        self.assertEqual(df_A.index.name, None)
        self.assertEqual(df_B.index.name, None)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_index_names_one_none_linking(self, method_to_call,
                                          *args, **kwargs):

        # setup A
        index_A = pd.Index(self.index_a, name=None)
        df_A = pd.DataFrame(self.A, index=pd.Index(index_A, name=None))

        # setup B
        index_B = pd.Index(self.index_b, name=None)
        df_B = pd.DataFrame(self.B, index=pd.Index(index_B, name='right'))

        # Make pairs
        index_cl = recordlinkage.Pairs(df_A, df_B)
        pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

        print (pairs.names)
        print (df_A.index.name)

        self.assertEqual(pairs.names, [None, 'right'])
        self.assertEqual(df_A.index.name, None)
        self.assertEqual(df_B.index.name, 'right')

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_amount_of_pairs_linking(self, method_to_call, *args, **kwargs):

        df_A = pd.DataFrame(self.A)
        df_B = pd.DataFrame(self.B)

        index_cl = recordlinkage.Pairs(df_A, df_B)
        pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

        # Test is the amount of pairs is less than A*B
        self.assertTrue(len(pairs) <= len(df_A) * len(df_B))

    def test_index_unique_linking(self):

        index_A = pd.Index(self.bad_index_a, name='left')
        index_B = pd.Index(self.index_b, name='right')

        A = pd.DataFrame(self.A, index=index_A)
        B = pd.DataFrame(self.B, index=index_B)

        with self.assertRaises(IndexError):
            recordlinkage.Pairs(A, B)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_pairs_are_unique_linking(self, method_to_call, *args, **kwargs):

        df_A = pd.DataFrame(self.A)
        df_B = pd.DataFrame(self.B)

        index_cl = recordlinkage.Pairs(df_A, df_B)
        pairs = getattr(index_cl, method_to_call)(*args, **kwargs)

        self.assertTrue(pairs.is_unique)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_full_iter_index_linking(self, method_to_call, *args, **kwargs):

        df_A = pd.DataFrame(self.A)
        df_B = pd.DataFrame(self.B)

        index_cl = recordlinkage.Pairs(df_A, df_B, chunks=100)
        index = recordlinkage.Pairs(df_A, df_B)

        # Compute pairs in one iteration
        pairs_single = getattr(index, method_to_call)(*args, **kwargs)

        # Compute pairs in iterations
        n_pairs_iter = 0
        for pairs in getattr(index_cl, method_to_call)(*args, **kwargs):

            print (len(pairs))
            n_pairs_iter += len(pairs)

            # Check if index is unique
            self.assertTrue(pairs.is_unique)

        self.assertEqual(len(pairs_single), n_pairs_iter)

    @parameterized.expand(INDEX_ALGORITHMS)
    def test_reduction_ratio_linking(self, method_to_call, *args, **kwargs):

        df_A = pd.DataFrame(self.A)
        df_B = pd.DataFrame(self.B)

        index_cl = recordlinkage.Pairs(df_A, df_B)
        getattr(index_cl, method_to_call)(*args, **kwargs)

        rr = index_cl.reduction

        self.assertTrue((rr >= 0.0) & (rr <= 1.0))