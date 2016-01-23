import unittest

import pandas.util.testing as pdt

import recordlinkage
from recordlinkage.standardise import StandardSeries, StandardDataFrame

import numpy as np
import pandas as pd

class TestStandardise(unittest.TestCase):

    def test_clean_type(self):

        values = ['Mary-ann', 'Bob :)', 'Angel', 'Bob (alias Billy)', 'Mary  ann', 'John', np.nan]

        s = StandardSeries(values)
        s.clean(inplace=True)

        r = StandardSeries(values)
        r = r.clean(inplace=False)

        # Check if type is correct
        self.assertTrue(isinstance(s, (StandardSeries)))

        # Check if type is correct
        self.assertTrue(isinstance(r, (StandardSeries)))

    def test_clean_inplace(self):

        values = ['Mary-ann', 'Bob :)', 'Angel', 'Bob (alias Billy)', 'Mary  ann', 'John', np.nan]
        expected = ['mary ann', 'bob', 'angel', 'bob', 'mary ann', 'john', np.nan]

        s = StandardSeries(values)
        s.clean(inplace=True)

        # Check if type is correct
        self.assertTrue(isinstance(s, (StandardSeries)))

        # Check if series are identical.
        s_exp = StandardSeries(expected)
        pdt.assert_series_equal(s, s_exp)

    def test_clean(self):

        values = ['Mary-ann', 'Bob :)', 'Angel', 'Bob (alias Billy)', 'Mary  ann', 'John', np.nan]
        expected = ['mary ann', 'bob', 'angel', 'bob', 'mary ann', 'john', np.nan]

        s = StandardSeries(values)
        s = s.clean(lower=True, replace_by_none='[^ \-\_A-Za-z0-9]+', replace_by_whitespace='[\-\_]', remove_brackets=True, inplace=False)

        s_exp = StandardSeries(expected)

        pdt.assert_series_equal(s, s_exp)

    def test_clean_lower(self):

        values = [np.nan, 'LowerHigher', 'HIGHERLOWER']
        expected = [np.nan, 'lowerhigher', 'higherlower']

        s = StandardSeries(values)
        s.clean(lower=True)

        s_exp = StandardSeries(expected)

        pdt.assert_series_equal(s, s_exp)

    def test_clean_brackets(self):

        values = [np.nan, 'bra(cke)ts', 'brackets with (brackets)']
        expected = [np.nan, 'brats', 'brackets with']

        s = StandardSeries(values)
        s.clean(remove_brackets=True)

        s_exp = StandardSeries(expected)

        pdt.assert_series_equal(s, s_exp)

    def test_value_occurence(self):

        values_str = [np.nan, np.nan, 'str1', 'str1', 'str1', 'str1', 'str2', 'str3', 'str3', 'str1']
        expected_str = [2,2,5,5,5,5,1,2,2,5]

        s = StandardSeries(values_str)
        s_occ_str = s.value_occurence()
        s_occ_exp = StandardSeries(expected_str)

        pdt.assert_series_equal(s_occ_str, s_occ_exp)

    def test_encode_soundex(self):

        values = [np.nan, 'John', 'Mary Ann', 'billy', 'Jonathan', 'Gretha', 'Micheal', 'Sjors']
        expected = [np.nan, 'J500', 'M650', 'B400', 'J535', 'G630', 'M240', 'S620']

        s = StandardSeries(values)
        s = s.phonetic('soundex', inplace=False)
        s_exp = StandardSeries(expected)

        pdt.assert_series_equal(s, s_exp)

    def test_encode_nysiis(self):

        values = [np.nan, 'John', 'Mary Ann', 'billy', 'Jonathan', 'Gretha', 'Micheal', 'Sjors']
        expected = [np.nan, 'JAN', 'MARY AN', 'BALY', 'JANATAN', 'GRAT', 'MACAL', 'SJAR']

        s = StandardSeries(values)
        s = s.phonetic('nysiis', inplace=False)
        s_exp = StandardSeries(expected)

        pdt.assert_series_equal(s, s_exp)

