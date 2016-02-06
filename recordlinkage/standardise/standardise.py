from __future__ import division
from __future__ import absolute_import

from .utils import check_type

import logging

import numpy as np
import pandas as pd

import itertools

def clean(s, lower=True, replace_by_none='[^ \-\_A-Za-z0-9]+', replace_by_whitespace='[\-\_]', remove_brackets=True):
    """
    clean(lower=True, replace_by_none='[^ \-\_A-Za-z0-9]+', replace_by_whitespace='[\-\_]', remove_brackets=True)
    
    Remove special tokens from a series of strings. 

    :param lower: Convert the strings in lowercase characters.
    :param replace_by_none: A regular expression that is replaced by ''.
    :param replace_by_whitespace: A regular expression that is replaced by a whitespace.
    :param remove_brackets: Remove all content between brackets and the brackets themselves. 

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

def clean_phonenumbers(s, country='USA'):
    """ Clean string formatted phonenumbers into string of intergers. 

    :return: A Series with cleaned phonenumbers.
    :rtype: pandas.Series
    """

    s = s.astype(str)

    # Remove all special tokens
    s = s.str.replace('[^0-9]+', '')

    return s

def value_occurence(s):
    """
    Count the number of times a value occurs. The difference with pandas.value_counts is that this function returns the values for each row. 

    :return: A Series with value counts.
    :rtype: pandas.Series
    """

    value_count = s.fillna('NAN')

    return value_count.groupby(by=value_count.values).transform('count')

def phonetic(s, method, encoding='utf-8'):
    """
    phonetic(method, encoding='utf-8')

    Phonetically encode the values in the Series. 

    :param method: The algorithm that is used to phonetically encode the values. The possible options are 'soundex' en 'nysiis'.
    :param encoding: String values need to be in unicode. Default 'utf-8'

    :return: A Series with phonetic encoded values.
    :rtype: pandas.Series
    """

    try:
        import jellyfish
    except ImportError:
        print "Install jellyfish to use string encoding."

    if method == 'soundex':
        return s.str.upper().str.decode(encoding).apply(lambda x: jellyfish.soundex(x) if pd.notnull(x) else np.nan)

    elif method == 'nysiis':
        return s.str.upper().str.decode(encoding).apply(lambda x: jellyfish.nysiis(x) if pd.notnull(x) else np.nan)

    else:
        raise Exception("Phonetic encoding method not found")

    return s

def similar_values(s, threshold=0.8):
    """
    similar_values(threshold=0.8)

    Group strings with high similarities. 

    :param threshold: Two strings with similarity above this threshold are considered to be the same string. The threshold is a value equal or between 0 and 1. Default 0.8. 
    :param inplace: If True, replace the current strings by their cleaned variant. Default: True.

    :return: A Series of strings.
    :rtype: pandas.Series

    """
    try:
        import jellyfish
    except ImportError:
        print "Install jellyfish to use string encoding."

    replace_tuples = []

    for pair in itertools.combinations(self[self.notnull()].astype(unicode).unique(), 2):

        sim = 1-jellyfish.levenshtein_distance(pair[0], pair[1])/np.max([len(pair[0]),len(pair[1])])

        if (sim >= threshold):
            replace_tuples.append(pair)

    # This is not a very clever solution I think. Don't known how to solve it atm: connected_components?
    for pair in replace_tuples: 

        if (sum(self == pair[0]) > sum(self == pair[1])):
            self = pd.Series(self.str.replace(pair[1], pair[0]))
        else:
            self = pd.Series(self.str.replace(pair[0], pair[1]))

    return pd.Series(string)

