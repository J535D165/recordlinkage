from __future__ import division
from __future__ import absolute_import

import logging

import numpy as np
import pandas as pd

import itertools

def clean(s, lower=True, replace_by_none='[^ \-\_A-Za-z0-9]+', replace_by_whitespace='[\-\_]', remove_brackets=True):
    """
    Clean strings in the Series by removing unwanted tokens, whitespace and brackets.

    :param s: A Series to clean.
    :param lower: Convert strings in the Series to lowercase. Default True.
    :param replace_by_none: The matches of this regular expression are replaced by ''.
    :param replace_by_whitespace: The matches of this regular expression are replaced by a whitespace.
    :param remove_brackets: Remove all content between brackets and the brackets themselves. Default True.

    :type s: pandas.Series
    :type lower: bool
    :type replace_by_none: str
    :type replace_by_whitespace: str
    :type remove_brackets: bool

    :return: A cleaned Series of strings.
    :rtype: pandas.Series

    Example:
    
    .. code:: python

        >>> import pandas
        >>> from recordlinkage.standardise import clean

        >>> s = pandas.Series(['Mary-ann', 'Bob :)', 'Angel', 'Bob (alias Billy)', None])
        >>> print(clean(s))

        0    mary ann
        1         bob
        2       angel
        3         bob
        4         NaN
        dtype: object


    """

    # Lower s if lower is True
    s = s.str.lower() if lower else s 

    # Remove all content between brackets
    if remove_brackets:
        s = s.str.replace(r'(\[.*?\]|\(.*?\)|\{.*?\})', '')

    # Remove the special characters
    s = s.str.replace(replace_by_none, '')
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
    Count the number of times each value occurs. This function returns the values for each row, in contrast with `pandas.value_counts <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html>`_. 

    :return: A Series with value counts.
    :rtype: pandas.Series
    """

    value_count = s.fillna('NAN')

    return value_count.groupby(by=value_count).transform('count')
