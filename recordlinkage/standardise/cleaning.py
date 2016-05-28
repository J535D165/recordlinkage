from __future__ import division
from __future__ import absolute_import

import logging

import numpy as np
import pandas as pd

import itertools

def clean(s, lower=True, replace_by_none='[^ \-\_A-Za-z0-9]+', replace_by_whitespace='[\-\_]', remove_brackets=True):
    """
    clean(lower=True, replace_by_none='[^ \-\_A-Za-z0-9]+', replace_by_whitespace='[\-\_]', remove_brackets=True)
    
    Clean a pandas Series with strings. Remove unwanted tokens and additional
    whitespace.

    :param s: A pandas.Series to clean.
    :param lower: Convert the strings in lowercase characters.
    :param replace_by_none: A regular expression that is replaced by ''.
    :param replace_by_whitespace: A regular expression that is replaced by a whitespace.
    :param remove_brackets: Remove all content between brackets and the brackets themselves. 

    :type s: pandas.Series
    :type lower: boolean
    :type replace_by_none: string
    :type replace_by_whitespace: string
    :type remove_brackets: boolean

    :return: A cleaned Series of strings.
    :rtype: pandas.Series

    For example:

    >>> s = pandas.Series(['Mary-ann', 'Bob :)', 'Angel', 'Bob (alias Billy)'])
    >>> print(recordlinkage.clean(s))
    mary ann
    bob
    angel
    bob

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

    Clean string formatted phonenumbers. 

    :param s: A pandas.Series to clean.
    :type s: pandas.Series

    :return: A Series with cleaned phonenumbers.
    :rtype: pandas.Series
    """

    # Remove all special tokens
    s = s.astype(object).str.replace('[^0-9+]+', '')

    return s

def value_occurence(s):
    """
    Count the number of times a value occurs. The difference with pandas.value_counts is that this function returns the values for each row. 

    :return: A Series with value counts.
    :rtype: pandas.Series
    """

    value_count = s.fillna('NAN')

    return value_count.groupby(by=value_count).transform('count')
