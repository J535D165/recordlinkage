import unittest

import pandas.util.testing as pdt
import recordlinkage
import numpy as np
import pandas as pd

TEST_DATA_1 = pd.DataFrame([
    [u'bob', u'smith', 10, u'new york'],
    [np.nan, u'smith', 10, u'new york'],
    [u'bob2', u'smith', 10, u'new york'],
    [u'bob2', u'smith', 10, u'new york'],
    [u'bob', u'smith', np.nan, u'new york']
    ],
    columns=['name', 'lastname', 'age', 'place'],
    index=pd.Index([1,2,3,4,5], name='index_df1')
    )

TEST_DATA_2 = pd.DataFrame([
    [u'bob', u'smith', 10, u'new york'],
    [np.nan, u'smith', 10, u'new york'],
    [u'bob2', u'smith', 10, u'new york'],
    [u'bob2', u'smith', np.nan, u'new york'],
    [u'bob', u'smith', np.nan, u'new york']
    ],
    columns=['name', 'lastname', 'age', 'place'],
    index=pd.Index([1,2,3,4,5], name='index_df2')
    )

TEST_INDEX_LINKING = pd.MultiIndex.from_arrays(
    [[1,1,2,3,4,5], [2,4,3,5,5,1]], 
    names=[TEST_DATA_1.index.name, TEST_DATA_2.index.name])

TEST_INDEX_DEDUP = pd.MultiIndex.from_arrays(
    [[1,1,2,3,4,5], [2,4,3,5,5,1]], 
    names=[TEST_DATA_1.index.name, TEST_DATA_1.index.name])


class TestCompare(unittest.TestCase):

    def test_link_exact_basic(self):

        comp = recordlinkage.Compare(TEST_INDEX_LINKING, TEST_DATA_1, TEST_DATA_2)

        # Missing values
        result = comp.exact('name', 'name', name='y_name')
        expected = pd.Series([0,0,0,0,0,1], index=TEST_INDEX_LINKING, name='y_name')

        pdt.assert_series_equal(result, expected)

    def test_link_exact_missing(self):

        comp = recordlinkage.Compare(TEST_INDEX_LINKING, TEST_DATA_1, TEST_DATA_2)

        # Missing values as 0
        result = comp.exact('name', 'name', missing_value=0, name='y_name')
        expected = pd.Series([0,0,0,0,0,1], index=TEST_INDEX_LINKING, name='y_name')

        pdt.assert_series_equal(result, expected)

        # Missing values as np.nan
        result = comp.exact('name', 'name', missing_value=np.nan, name='y_name')
        expected = pd.Series([np.nan,0,np.nan,0,0,1], index=TEST_INDEX_LINKING, name='y_name')

        pdt.assert_series_equal(result, expected)

        # Missing values as np.nan
        result = comp.exact('name', 'name', missing_value=9, name='y_name')
        expected = pd.Series([9,0,9,0,0,1], index=TEST_INDEX_LINKING, name='y_name')

        pdt.assert_series_equal(result, expected)

    def test_link_exact_disagree(self):

        comp = recordlinkage.Compare(TEST_INDEX_LINKING, TEST_DATA_1, TEST_DATA_2)

        # Missing values 0 and disagreement as 2
        result = comp.exact('name', 'name', disagree_value=2, missing_value=0, name='y_name')
        expected = pd.Series([0,2,0,2,2,1], index=TEST_INDEX_LINKING, name='y_name')

        pdt.assert_series_equal(result, expected)

    def test_dedup_exact_basic(self):
        
        comp = recordlinkage.Compare(TEST_INDEX_DEDUP, TEST_DATA_1, TEST_DATA_1)

        # Missing values
        result = comp.exact('name', 'name', name='y_name')
        expected = pd.Series([0,0,0,0,0,1], index=TEST_INDEX_DEDUP, name='y_name')

        pdt.assert_series_equal(result, expected)
       
    def test_fuzzy_does_not_exist(self):
        
        comp = recordlinkage.Compare(TEST_INDEX_DEDUP, TEST_DATA_1, TEST_DATA_1)

        self.assertRaises(ValueError, comp.fuzzy, 'name', 'name', name='y_name', method='unknown_algorithm')

    def test_fuzzy_same_labels(self):
        
        comp = recordlinkage.Compare(TEST_INDEX_DEDUP, TEST_DATA_1, TEST_DATA_1)

        for alg in ['jaro', 'jaro_winkler', 'dameraulevenshtein', 'levenshtein', 'q_gram', 'cosine']:

            print (alg)

            # Missing values
            result = comp.fuzzy('name', 'name', method=alg, missing_value=np.nan) # Change in future (should work without method)

            print (result)

            self.assertTrue(result.notnull().any())
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())

    def test_fuzzy_different_labels(self):
        
        comp = recordlinkage.Compare(TEST_INDEX_DEDUP, TEST_DATA_1, TEST_DATA_1)

        for alg in ['jaro', 'jaro_winkler', 'dameraulevenshtein', 'levenshtein', 'q_gram', 'cosine']:

            print (alg)

            # Missing values
            result = comp.fuzzy('name', 'lastname', method=alg, missing_value=np.nan) # Change in future (should work without method)
            
            print (result)

            self.assertTrue(result.notnull().any())
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())



