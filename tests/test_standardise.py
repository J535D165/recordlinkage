#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import pandas.util.testing as pdt

from recordlinkage.standardise import clean, phonenumbers, \
    value_occurence, phonetic

import numpy as np
import pandas as pd


class TestCleaningStandardise(unittest.TestCase):

    def test_clean(self):

        values = pd.Series(
            ['Mary-ann',
             'Bob :)',
             'Angel',
             'Bob (alias Billy)',
             'Mary  ann',
             'John',
             np.nan
             ])

        expected = pd.Series(
            ['mary ann',
             'bob',
             'angel',
             'bob',
             'mary ann',
             'john',
             np.nan])

        clean_series = clean(values)

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_unicode(self):

        values = pd.Series(
            [u'Mary-ann',
             u'Bob :)',
             u'Angel',
             u'Bob (alias Billy)',
             u'Mary  ann',
             u'John',
             np.nan
             ])

        expected = pd.Series(
            [u'mary ann',
             u'bob',
             u'angel',
             u'bob',
             u'mary ann',
             u'john',
             np.nan])

        clean_series = clean(values)

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_parameters(self):

        values = pd.Series(
            [u'Mary-ann',
             u'Bob :)',
             u'Angel',
             u'Bob (alias Billy)',
             u'Mary  ann',
             u'John',
             np.nan
             ])

        expected = pd.Series(
            [u'mary ann',
             u'bob',
             u'angel',
             u'bob',
             u'mary ann',
             u'john',
             np.nan])

        clean_series = clean(
            values,
            lowercase=True,
            replace_by_none='[^ \-\_A-Za-z0-9]+',
            replace_by_whitespace='[\-\_]',
            remove_brackets=True
        )

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_lower(self):

        values = pd.Series([np.nan, 'LowerHigher', 'HIGHERLOWER'])
        expected = pd.Series([np.nan, 'lowerhigher', 'higherlower'])

        clean_series = clean(values, lowercase=True)

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_brackets(self):

        values = pd.Series([np.nan, 'bra(cke)ts', 'brackets with (brackets)'])
        expected = pd.Series([np.nan, 'brats', 'brackets with'])

        clean_series = clean(values, remove_brackets=True)

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_accent_stripping(self):

        values = pd.Series(['ősdfésdfë', 'without'])
        expected = pd.Series(['osdfesdfe', 'without'])

        values_unicode = pd.Series([u'ősdfésdfë', u'without'])
        expected_unicode = pd.Series([u'osdfesdfe', u'without'])

        # Check if series are identical.
        pdt.assert_series_equal(
            clean(values, strip_accents='unicode'),
            expected)

        # Check if series are identical.
        pdt.assert_series_equal(
            clean(values, strip_accents='ascii'),
            expected)

        # Check if series are identical.
        pdt.assert_series_equal(
            clean(values_unicode, strip_accents='unicode'),
            expected_unicode)

        # Check if series are identical.
        pdt.assert_series_equal(
            clean(values_unicode, strip_accents='ascii'),
            expected_unicode)

    def test_clean_phonenumbers(self):

        values = pd.Series(
            [np.nan, '0033612345678', '+1 201 123 4567', '+336-123 45678'])
        expected = pd.Series(
            [np.nan, '0033612345678', '+12011234567', '+33612345678'])

        clean_series = phonenumbers(values)

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_value_occurence(self):

        values = pd.Series([np.nan, np.nan, 'str1', 'str1',
                            'str1', 'str1', 'str2', 'str3', 'str3', 'str1'])
        expected = pd.Series([2, 2, 5, 5, 5, 5, 1, 2, 2, 5])

        pdt.assert_series_equal(value_occurence(values), expected)


class TestEncodingStandardise(unittest.TestCase):

    def test_encode_soundex(self):

        values = pd.Series([np.nan, u'John', u'Mary Ann', u'billy',
                            u'Jonathan', u'Gretha', u'Micheal', u'Sjors'])
        expected = pd.Series(
            [np.nan, u'J500', u'M650', u'B400', u'J535',
             u'G630', u'M240', u'S620'])

        phon = phonetic(values, 'soundex')

        pdt.assert_series_equal(phon, expected)

    def test_encode_nysiis(self):

        values = pd.Series([np.nan, u'John', u'Mary Ann', u'billy',
                            u'Jonathan', u'Gretha', u'Micheal', u'Sjors'])
        expected = pd.Series(
            [np.nan, u'JAN', u'MARYAN', u'BALY', u'JANATAN',
             u'GRAT', u'MACAL', u'SJAR'])

        phon = phonetic(values, 'nysiis')

        pdt.assert_series_equal(phon, expected)
