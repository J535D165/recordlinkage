import unittest

import pandas.util.testing as pdt
import recordlinkage
import numpy as np
import pandas as pd

TEST_DATA_1 = pd.DataFrame([
    ['bob', 'smith', 10, 'new york'],
    [np.nan, 'smith', 10, 'new york'],
    ['bob2', 'smith', 10, 'new york'],
    ['bob2', 'smith', 10, 'new york'],
    ['bob', 'smith', np.nan, 'new york']
    ],
    columns=['name', 'lastname', 'age', 'place'],
    index=pd.Index([1,2,3,4,5], name='index_df1')
    )

TEST_DATA_2 = pd.DataFrame([
    ['bob', 'smith', 10, 'new york'],
    [np.nan, 'smith', 10, 'new york'],
    ['bob2', 'smith', 10, 'new york'],
    ['bob2', 'smith', np.nan, 'new york'],
    ['bob', 'smith', np.nan, 'new york']
    ],
    columns=['name', 'lastname', 'age', 'place'],
    index=pd.Index([1,2,3,4,5], name='index_df2')
    )

TEST_INDEX = pd.MultiIndex.from_tuples(
    zip([1,1,2,3,4,5], [2,4,3,5,5,1]), 
    names=[TEST_DATA_1.index.name, TEST_DATA_2.index.name])


class TestCompare(unittest.TestCase):

    def test_exact_two_series(self):

        comp = recordlinkage.Compare(TEST_INDEX, TEST_DATA_1, TEST_DATA_2)

        # Missing values as 0
        result = comp.exact('name', 'name', missing_value=0, name='y_name')
        expected = pd.Series([0,0,0,0,0,1], index=TEST_INDEX, name='y_name')

        pdt.assert_series_equal(expected, result)

        # # Missing values as np.nan
        result = comp.exact('name', 'name', missing_value=9, name='y_name')
        expected = pd.Series([np.nan,0,np.nan,0,0,1], index=TEST_INDEX, name='y_name')

        print result
        pdt.assert_series_equal(expected, result)

        # pdt.assert_series_equal(expected, result)

        # # Missing values 0 and disagreement as 2
        # result = comp.exact(s1, s2, disagreement_value=2, missing_value=0)
        # expected = pd.Series([1,2,2,1,1,1,0])

        # pdt.assert_series_equal(expected, result)
