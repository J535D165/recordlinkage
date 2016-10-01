from __future__ import division
from __future__ import absolute_import
# from __future__ import unicode_literals

from sklearn.feature_extraction.text import strip_accents_ascii, \
    strip_accents_unicode


def clean(s, lowercase=True, replace_by_none=r'[^ \-\_A-Za-z0-9]+',
          replace_by_whitespace=r'[\-\_]', strip_accents=None,
          remove_brackets=True, encoding='utf-8', decode_error='strict'):
    """

    Clean strings in the Series by removing unwanted tokens, whitespace and
    brackets.

    :param s: A Series to clean.
    :param lower: Convert strings in the Series to lowercase. Default True.
    :param replace_by_none: The matches of this regular expression are
            replaced by ''.
    :param replace_by_whitespace: The matches of this regular expression are
            replaced by a whitespace.
    :param remove_brackets: Remove all content between brackets and the
            brackets themselves. Default True.
    :param strip_accents: Remove accents during the preprocessing step.
            'ascii' is a fast method that only works on characters that have
            an direct ASCII mapping.
            'unicode' is a slightly slower method that works on any characters.
            None (default) does nothing.
    :param encoding: If bytes are given, this encoding is used to
            decode.
    :param decode_error: Instruction on what to do if a byte Series is given
            that contains characters not of the given `encoding`. By default,
            it is 'strict', meaning that a UnicodeDecodeError will be raised.
            Other values are 'ignore' and 'replace'.

    :type s: pandas.Series
    :type lower: bool
    :type replace_by_none: str
    :type replace_by_whitespace: str
    :type remove_brackets: bool
    :type strip_accents: {'ascii', 'unicode', None}
    :type encoding: string, default='utf-8'
    :type decode_error: {'strict', 'ignore', 'replace'}

    :return: A cleaned Series of strings.
    :rtype: pandas.Series

    Example:

    .. code:: python

        >>> import pandas
        >>> from recordlinkage.standardise import clean

        >>> name = ['Mary-ann', 'Bob :)', 'Angel', 'Bob (alias Billy)', None]
        >>> s = pandas.Series(names)

        >>> print(clean(s))

        0    mary ann
        1         bob
        2       angel
        3         bob
        4         NaN
        dtype: object


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
    elif strip_accents == 'ascii':
        strip_accents_fn = strip_accents_ascii
    elif strip_accents == 'unicode':
        strip_accents_fn = strip_accents_unicode
    else:
        raise ValueError('Invalid value for "strip_accents": %s' %
                         strip_accents)

    # Remove accents etc
    if strip_accents:

        # encoding
        s = s.apply(
            lambda x: x.decode(encoding, decode_error) if type(x) == bytes else x)
        s = s.map(lambda x: strip_accents_fn(x))

    # Remove all content between brackets
    if remove_brackets is True:
        s = s.str.replace(r'(\[.*?\]|\(.*?\)|\{.*?\})', '')

    # Remove the special characters
    if replace_by_none:
        s = s.str.replace(replace_by_none, '')

    if replace_by_whitespace:
        s = s.str.replace(replace_by_whitespace, ' ')

    # Remove multiple whitespaces
    s = s.str.replace(r'\s\s+', ' ')

    # Strip s
    s = s.str.lstrip().str.rstrip()

    return s


def phonenumbers(s):
    """

    Clean phonenumbers by removing all non-numbers (except +).

    :param s: A Series to clean.
    :type s: pandas.Series

    :return: A Series with cleaned phonenumbers.
    :rtype: pandas.Series
    """

    # Remove all special tokens
    s = s.astype(object).str.replace('[^0-9+]+', '')

    return s


def value_occurence(s):
    """

    Count the number of times each value occurs. This function returns the
    values for each row, in contrast with `pandas.value_counts
    <http://pandas.pydata.org/pandas-
    docs/stable/generated/pandas.Series.value_counts.html>`_.

    :return: A Series with value counts.
    :rtype: pandas.Series
    """

    # https://github.com/pydata/pandas/issues/3729
    value_count = s.fillna('NAN')

    return value_count.groupby(by=value_count).transform('count')
