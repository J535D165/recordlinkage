#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from recordlinkage.preprocessing import clean
from recordlinkage.preprocessing import phonenumbers
from recordlinkage.preprocessing import phonetic
from recordlinkage.preprocessing import phonetic_algorithms
from recordlinkage.preprocessing import value_occurence


class TestCleaningStandardise:
    def test_clean(self):
        values = pd.Series(
            [
                "Mary-ann",
                "Bob :)",
                "Angel",
                "Bob (alias Billy)",
                "Mary ann",
                "John",
                np.nan,
            ]
        )

        expected = pd.Series(
            ["mary ann", "bob", "angel", "bob", "mary ann", "john", np.nan]
        )

        clean_series = clean(values)

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

        clean_series_nothing = clean(
            values,
            lowercase=False,
            replace_by_none=False,
            replace_by_whitespace=False,
            strip_accents=False,
            remove_brackets=False,
        )

        # Check if ntohing happend.
        pdt.assert_series_equal(clean_series_nothing, values)

    def test_clean_empty(self):
        """Test the cleaning of an empty Series"""

        # Check empty series
        pdt.assert_series_equal(clean(pd.Series(dtype=object)), pd.Series(dtype=object))

    def test_clean_unicode(self):
        values = pd.Series(
            [
                "Mary-ann",
                "Bob :)",
                "Angel",
                "Bob (alias Billy)",
                "Mary  ann",
                "John",
                np.nan,
            ]
        )

        expected = pd.Series(
            ["mary ann", "bob", "angel", "bob", "mary ann", "john", np.nan]
        )

        clean_series = clean(values)

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_parameters(self):
        values = pd.Series(
            [
                "Mary-ann",
                "Bob :)",
                "Angel",
                "Bob (alias Billy)",
                "Mary  ann",
                "John",
                np.nan,
            ]
        )

        expected = pd.Series(
            ["mary ann", "bob", "angel", "bob", "mary ann", "john", np.nan]
        )

        clean_series = clean(
            values,
            lowercase=True,
            replace_by_none=r"[^ \-\_A-Za-z0-9]+",
            replace_by_whitespace=r"[\-\_]",
            remove_brackets=True,
        )

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_lower(self):
        values = pd.Series([np.nan, "LowerHigher", "HIGHERLOWER"])
        expected = pd.Series([np.nan, "lowerhigher", "higherlower"])

        clean_series = clean(values, lowercase=True)

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_brackets(self):
        values = pd.Series([np.nan, "bra(cke)ts", "brackets with (brackets)"])
        expected = pd.Series([np.nan, "brats", "brackets with"])

        clean_series = clean(values, remove_brackets=True)

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_clean_accent_stripping(self):
        values = pd.Series(["ősdfésdfë", "without"])
        expected = pd.Series(["osdfesdfe", "without"])

        values_unicode = pd.Series(["ősdfésdfë", "without"])
        expected_unicode = pd.Series(["osdfesdfe", "without"])

        # values_callable = pd.Series([u'ősdfésdfë', u'without'])
        # expected_callable = pd.Series([u'ősdfésdfë', u'without'])

        # # Callable.
        # pdt.assert_series_equal(
        #     clean(values_callable, strip_accents=lambda x: x),
        #     expected_callable)

        # Check if series are identical.
        pdt.assert_series_equal(clean(values, strip_accents="unicode"), expected)

        # Check if series are identical.
        pdt.assert_series_equal(clean(values, strip_accents="ascii"), expected)

        # Check if series are identical.
        pdt.assert_series_equal(
            clean(values_unicode, strip_accents="unicode"), expected_unicode
        )

        # Check if series are identical.
        pdt.assert_series_equal(
            clean(values_unicode, strip_accents="ascii"), expected_unicode
        )

        with pytest.raises(ValueError):
            clean(values, strip_accents="unknown_algorithm")

    def test_clean_phonenumbers(self):
        values = pd.Series(
            [np.nan, "0033612345678", "+1 201 123 4567", "+336-123 45678"]
        )
        expected = pd.Series([np.nan, "0033612345678", "+12011234567", "+33612345678"])

        clean_series = phonenumbers(values)

        # Check if series are identical.
        pdt.assert_series_equal(clean_series, expected)

    def test_value_occurence(self):
        values = pd.Series(
            [
                np.nan,
                np.nan,
                "str1",
                "str1",
                "str1",
                "str1",
                "str2",
                "str3",
                "str3",
                "str1",
            ]
        )
        expected = pd.Series([2, 2, 5, 5, 5, 5, 1, 2, 2, 5])

        pdt.assert_series_equal(value_occurence(values), expected)


class TestEncodingStandardise:
    def test_encode_soundex(self):
        values = pd.Series(
            [
                np.nan,
                "John",
                "Mary Ann",
                "billy",
                "Jonathan",
                "Gretha",
                "Micheal",
                "Sjors",
            ]
        )
        expected = pd.Series(
            [np.nan, "J500", "M650", "B400", "J535", "G630", "M240", "S620"]
        )

        phon = phonetic(values, "soundex")

        pdt.assert_series_equal(phon, expected)

    def test_encode_nysiis(self):
        values = pd.Series(
            [
                np.nan,
                "John",
                "Mary Ann",
                "billy",
                "Jonathan",
                "Gretha",
                "Micheal",
                "Sjors",
            ]
        )
        expected = pd.Series(
            [np.nan, "JAN", "MARYAN", "BALY", "JANATAN", "GRAT", "MACAL", "SJAR"]
        )

        phon = phonetic(values, "nysiis")

        pdt.assert_series_equal(phon, expected)

    def test_encode_metaphone(self):
        values = pd.Series(
            [
                np.nan,
                "John",
                "Mary Ann",
                "billy",
                "Jonathan",
                "Gretha",
                "Micheal",
                "Sjors",
            ]
        )
        expected = pd.Series([np.nan, "JN", "MRYN", "BL", "JN0N", "KR0", "MXL", "SJRS"])

        phon = phonetic(values, method="metaphone")

        pdt.assert_series_equal(phon, expected)

    def test_encode_match_rating(self):
        values = pd.Series(
            [
                np.nan,
                "John",
                "Mary Ann",
                "billy",
                "Jonathan",
                "Gretha",
                "Micheal",
                "Sjors",
            ]
        )
        # in jellyfish.match_rating_codex version 0.8.0 results have changed
        expected = pd.Series(
            [np.nan, "JHN", "MRYN", "BLY", "JNTHN", "GRTH", "MCHL", "SJRS"]
        )

        phon = phonetic(values, method="match_rating")

        pdt.assert_series_equal(phon, expected)

    def test_phonetic_does_not_exist(self):
        values = pd.Series(
            [
                np.nan,
                "John",
                "Mary Ann",
                "billy",
                "Jonathan",
                "Gretha",
                "Micheal",
                "Sjors",
            ]
        )

        with pytest.raises(ValueError):
            phonetic(values, "unknown_algorithm")

    def test_list_of_algorithms(self):
        algorithms = phonetic_algorithms

        assert isinstance(algorithms, list)

        assert "soundex" in algorithms
        assert "nysiis" in algorithms
        assert "metaphone" in algorithms
        assert "match_rating" in algorithms
