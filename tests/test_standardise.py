import unittest

import pandas.util.testing as pdt

import recordlinkage
from recordlinkage.standardise import clean, value_occurence, phonetic

import numpy as np
import pandas as pd

class TestCleaningStandardise(unittest.TestCase):

    def test_clean(self):

        values = pd.Series(['Mary-ann', 'Bob :)', 'Angel', 'Bob (alias Billy)', 'Mary  ann', 'John', np.nan])
        expected = pd.Series(['mary ann', 'bob', 'angel', 'bob', 'mary ann', 'john', np.nan])

        clean_series = clean(values)

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_parameters(self):

        values = pd.Series(['Mary-ann', 'Bob :)', 'Angel', 'Bob (alias Billy)', 'Mary  ann', 'John', np.nan])
        expected = pd.Series(['mary ann', 'bob', 'angel', 'bob', 'mary ann', 'john', np.nan])

        clean_series = clean(values, lower=True, replace_by_none='[^ \-\_A-Za-z0-9]+', replace_by_whitespace='[\-\_]', remove_brackets=True)
        
        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_lower(self):

        values = pd.Series([np.nan, 'LowerHigher', 'HIGHERLOWER'])
        expected = pd.Series([np.nan, 'lowerhigher', 'higherlower'])

        clean_series = clean(values, lower=True)
        
        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_brackets(self):

        values = pd.Series([np.nan, 'bra(cke)ts', 'brackets with (brackets)'])
        expected = pd.Series([np.nan, 'brats', 'brackets with'])

        clean_series = clean(values, remove_brackets=True)
        
        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_value_occurence(self):

        values = pd.Series([np.nan, np.nan, 'str1', 'str1', 'str1', 'str1', 'str2', 'str3', 'str3', 'str1'])
        expected = pd.Series([2,2,5,5,5,5,1,2,2,5])

        pdt.assert_series_equal(value_occurence(values), expected)

class TestEncodingStandardise(unittest.TestCase):

    def test_encode_soundex(self):

        values = pd.Series([np.nan, 'John', 'Mary Ann', 'billy', 'Jonathan', 'Gretha', 'Micheal', 'Sjors'])
        expected = pd.Series([np.nan, 'J500', 'M650', 'B400', 'J535', 'G630', 'M240', 'S620'])

        phon = phonetic(values, 'soundex')

        pdt.assert_series_equal(phon, expected)

    def test_encode_nysiis(self):

        values = pd.Series([np.nan, 'John', 'Mary Ann', 'billy', 'Jonathan', 'Gretha', 'Micheal', 'Sjors'])
        expected = pd.Series([np.nan, 'JAN', 'MARYAN', 'BALY', 'JANATAN', 'GRAT', 'MACAL', 'SJAR'])

        phon = phonetic(values, 'nysiis')

        pdt.assert_series_equal(phon, expected)