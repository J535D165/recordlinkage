import unittest

import pandas.util.testing as pdt
import recordlinkage
import numpy as np
import pandas as pd

TEST_DATA_1 = pd.DataFrame([
    [u'Donell', u'Gerlach', 20, u'New York'],
    [np.nan, u'Smit', 17, u'Boston'],
    [u'Kalie', u'Flatley', 33, u'Boston'],
    [u'Kittie', u'Schuster', 27, np.nan],
    [np.nan, np.nan, np.nan, u'South Devyn']
    ],
    columns=['name', 'lastname', 'age', 'place'])

TEST_DATA_1.index.name = 'index_df1'

TEST_DATA_2 = pd.DataFrame([
    [u'Donel', u'Gerleach', 20, u'New York'],
    [np.nan, u'Smith', 17, u'Boston'],
    [u'Kaly', u'Flatley', 33, u'Boston'],
    [u'Kittie', np.nan, 20, np.nan],
    [u'Bob', u'Armstrong', 70, u'Lake Gavinmouth']
    ],
    columns=['name', 'lastname', 'age', 'place'])

TEST_DATA_2.index.name = 'index_df2'

TEST_INDEX_LINKING = pd.MultiIndex.from_arrays(
    [np.arange(len(TEST_DATA_1)), np.arange(len(TEST_DATA_2))], 
    names=[TEST_DATA_1.index.name, TEST_DATA_2.index.name])

TEST_INDEX_DEDUP = pd.MultiIndex.from_arrays(
    [np.arange(len(TEST_DATA_1)), np.arange(len(TEST_DATA_1))], 
    names=[TEST_DATA_1.index.name, TEST_DATA_1.index.name])


class TestCompare(unittest.TestCase):

    def test_link_exact_basic(self):

        comp = recordlinkage.Compare(TEST_INDEX_LINKING, TEST_DATA_1, TEST_DATA_2)

        # Missing values
        result = comp.exact('name', 'name', name='y_name')
        expected = pd.Series([0,0,0,1,0], index=TEST_INDEX_LINKING, name='y_name')

        pdt.assert_series_equal(result, expected)

    def test_link_exact_missing(self):

        comp = recordlinkage.Compare(TEST_INDEX_LINKING, TEST_DATA_1, TEST_DATA_2)

        # Missing values as 0
        result = comp.exact('name', 'name', missing_value=0, name='y_name')
        expected = pd.Series([0,0,0,1,0], index=TEST_INDEX_LINKING, name='y_name')

        pdt.assert_series_equal(result, expected)

        # Missing values as np.nan
        result = comp.exact('name', 'name', missing_value=np.nan, name='y_name')
        expected = pd.Series([0,np.nan,0,1,np.nan], index=TEST_INDEX_LINKING, name='y_name')

        pdt.assert_series_equal(result, expected)

        # Missing values as np.nan
        result = comp.exact('name', 'name', missing_value=9, name='y_name')
        expected = pd.Series([0,9,0,1,9], index=TEST_INDEX_LINKING, name='y_name')

        pdt.assert_series_equal(result, expected)

    def test_link_exact_disagree(self):

        comp = recordlinkage.Compare(TEST_INDEX_LINKING, TEST_DATA_1, TEST_DATA_2)

        # Missing values 0 and disagreement as 2
        result = comp.exact('name', 'name', disagree_value=2, missing_value=0, name='y_name')
        expected = pd.Series([2,0,2,1,0], index=TEST_INDEX_LINKING, name='y_name')

        pdt.assert_series_equal(result, expected)

    def test_dedup_exact_basic(self):
        
        comp = recordlinkage.Compare(TEST_INDEX_DEDUP, TEST_DATA_1, TEST_DATA_1)

        # Missing values
        result = comp.exact('name', 'name', name='y_name')
        expected = pd.Series([1,0,1,1,0], index=TEST_INDEX_DEDUP, name='y_name')

        pdt.assert_series_equal(result, expected)

    def test_numeric(self):

        comp = recordlinkage.Compare(TEST_INDEX_LINKING, TEST_DATA_1, TEST_DATA_2)

        # Missing values
        result = comp.numeric('age', 'age', 2)
        expected = pd.Series([1,1,1,0,0], index=TEST_INDEX_LINKING)

        pdt.assert_series_equal(result, expected)

    def test_geo(self):

        comp = recordlinkage.Compare(TEST_INDEX_LINKING, TEST_DATA_1, TEST_DATA_2)

        # Missing values
        result = comp.geo('age', 'age', 'age', 'age', 2)

        self.assertTrue(result.notnull().any())
        self.assertTrue((result[result.notnull()] >= 0).all())
        self.assertTrue((result[result.notnull()] <= 1).all())

    def test_numeric_batch(self):

        comp = recordlinkage.Compare(TEST_INDEX_DEDUP, TEST_DATA_1, TEST_DATA_2)

        for alg in ['step', 'linear', 'squared']:

            print (alg)

            # Missing values
            result = comp.numeric('age', 'age', 2, method=alg)
            
            print (result)

            self.assertTrue(result.notnull().any())
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())

    def test_fuzzy_does_not_exist(self):
        
        comp = recordlinkage.Compare(TEST_INDEX_DEDUP, TEST_DATA_1, TEST_DATA_1)

        self.assertRaises(ValueError, comp.fuzzy, 'name', 'name', name='y_name', method='unknown_algorithm')

    def test_fuzzy_same_labels(self):
        
        comp = recordlinkage.Compare(TEST_INDEX_DEDUP, TEST_DATA_1, TEST_DATA_2)

        for alg in ['jaro', 'jaro_winkler', 'dameraulevenshtein', 'levenshtein', 'q_gram', 'cosine']:

            print (alg)

            # Missing values
            result = comp.fuzzy('name', 'name', method=alg, missing_value=np.nan) 
            result = comp.fuzzy('name', 'name', alg, missing_value=np.nan) 

            print (result)

            self.assertTrue(result.notnull().any())
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())

    def test_fuzzy_different_labels(self):
        
        comp = recordlinkage.Compare(TEST_INDEX_DEDUP, TEST_DATA_1, TEST_DATA_2)

        for alg in ['jaro', 'jaro_winkler', 'dameraulevenshtein', 'levenshtein', 'q_gram', 'cosine']:

            print (alg)

            # Missing values
            result = comp.fuzzy('name', 'name', method=alg, missing_value=np.nan) # Change in future (should work without method)
            
            print (result)

            self.assertTrue(result.notnull().any())
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())

            # Debug trick
            # if alg == 'q_gram':
            #     rr


