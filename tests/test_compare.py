import unittest

import pandas.util.testing as pdt
import recordlinkage
import numpy as np
import pandas as pd

class TestCompare(unittest.TestCase):

    def test_exact_two_series(self):

        comp = recordlinkage.Compare()

        s1 = pd.Series(['mary ann', 'bob1', 'angel1', 'bob', 'mary ann', 'john', np.nan])
        s2 = pd.Series(['mary ann', 'bob2', 'angel2', 'bob', 'mary ann', 'john', np.nan])

        # Missing values as 0
        result = comp.exact(s1, s2, missing_value=0)
        expected = pd.Series([1,0,0,1,1,1,0])

        pdt.assert_series_equal(expected, result)

        # Missing values as np.nan
        result = comp.exact(s1, s2, missing_value=np.nan)
        expected = pd.Series([1,0,0,1,1,1,np.nan])

        pdt.assert_series_equal(expected, result)

        # Missing values 0 and disagreement as 2
        result = comp.exact(s1, s2, disagreement_value=2, missing_value=0)
        expected = pd.Series([1,2,2,1,1,1,0])

        pdt.assert_series_equal(expected, result)
