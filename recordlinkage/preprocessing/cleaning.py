import sys

from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.feature_extraction.text import strip_accents_unicode


def clean(
    s,
    lowercase=True,
    replace_by_none=r"[^ \-\_A-Za-z0-9]+",
    replace_by_whitespace=r"[\-\_]",
    strip_accents=None,
    remove_brackets=True,
    encoding="utf-8",
    decode_error="strict",
):
    """Clean string variables.

    Clean strings in the Series by removing unwanted tokens,
    whitespace and brackets.

    Parameters
    ----------
    s : pandas.Series
        A Series to clean.
    lower : bool, optional
        Convert strings in the Series to lowercase. Default True.
    replace_by_none : str, optional
        The matches of this regular expression are replaced by ''.
    replace_by_whitespace : str, optional
        The matches of this regular expression are replaced by a
        whitespace.
    remove_brackets : bool, optional
        Remove all content between brackets and the bracket
        themselves. Default True.
    strip_accents : {'ascii', 'unicode', None}, optional
        Remove accents during the preprocessing step. 'ascii' is a
        fast method that only works on characters that have an direct
        ASCII mapping. 'unicode' is a slightly slower method that
        works on any characters. None (default) does nothing.
    encoding : str, optional
        If bytes are given, this encoding is used to decode. Default
        is 'utf-8'.
    decode_error : {'strict', 'ignore', 'replace'}, optional
        Instruction on what to do if a byte Series is given that
        contains characters not of the given `encoding`. By default,
        it is 'strict', meaning that a UnicodeDecodeError will be
        raised. Other values are 'ignore' and 'replace'.

    Example
    -------
    >>> import pandas
    >>> from recordlinkage.preprocessing import clean
    >>>
    >>> names = ['Mary-ann',
                'Bob :)',
                'Angel',
                'Bob (alias Billy)',
                None]
    >>> s = pandas.Series(names)
    >>> print(clean(s))
    0    mary ann
    1         bob
    2       angel
    3         bob
    4         NaN
    dtype: object

    Returns
    -------
    pandas.Series:
        A cleaned Series of strings.

    """

    if s.shape[0] == 0:
        return s

    # Lower s if lower is True
    if lowercase is True:
        s = s.str.lower()

    # Accent stripping based on https://github.com/scikit-learn/
    # scikit-learn/blob/412996f/sklearn/feature_extraction/text.py
    # BSD license
    if not strip_accents:
        pass
    elif callable(strip_accents):
        strip_accents_fn = strip_accents
    elif strip_accents == "ascii":
        strip_accents_fn = strip_accents_ascii
    elif strip_accents == "unicode":
        strip_accents_fn = strip_accents_unicode
    else:
        raise ValueError(f"Invalid value for 'strip_accents': {strip_accents}")

    # Remove accents etc
    if strip_accents:

        def strip_accents_fn_wrapper(x):
            if sys.version_info[0] >= 3:
                if isinstance(x, str):
                    return strip_accents_fn(x)
                else:
                    return x
            else:
                if isinstance(x, unicode):  # noqa
                    return strip_accents_fn(x)
                else:
                    return x

        # encoding
        s = s.apply(
            lambda x: x.decode(encoding, decode_error) if type(x) == bytes else x
        )
        s = s.map(lambda x: strip_accents_fn_wrapper(x))

    # Remove all content between brackets
    if remove_brackets is True:
        s = s.str.replace(r"(\[.*?\]|\(.*?\)|\{.*?\})", "", regex=True)

    # Remove the special characters
    if replace_by_none:
        s = s.str.replace(replace_by_none, "", regex=True)

    if replace_by_whitespace:
        s = s.str.replace(replace_by_whitespace, " ", regex=True)

    # Remove multiple whitespaces
    s = s.str.replace(r"\s\s+", " ", regex=True)

    # Strip s
    s = s.str.lstrip().str.rstrip()

    return s


def phonenumbers(s):
    """Clean phonenumbers by removing all non-numbers (except +).

    Parameters
    ----------
    s: pandas.Series
        A Series to clean.

    Returns
    -------
    pandas.Series
        A Series with cleaned phonenumbers.

    """

    # Remove all special tokens
    s = s.astype(object).str.replace("[^0-9+]+", "", regex=True)

    return s


def value_occurence(s):
    """Count the number of times each value occurs.

    This function returns the counts for each row, in contrast with
    `pandas.value_counts <http://pandas.pydata.org/pandas-
    docs/stable/generated/pandas.Series.value_counts.html>`_.

    Returns
    -------
    pandas.Series
        A Series with value counts.

    """

    # https://github.com/pydata/pandas/issues/3729
    value_count = s.fillna("NAN")

    return value_count.groupby(by=value_count).transform("count")
