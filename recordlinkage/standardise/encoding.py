from __future__ import division
# from __future__ import absolute_import

import logging

import numpy as np
import pandas as pd

import itertools

from .cleaning import *

def phonetic(s, method):
    """
    phonetic(method, encoding='utf-8')

    Phonetically encode the values in the Series. 

    :param method: The algorithm that is used to phonetically encode the values. The possible options are 'soundex' en 'nysiis'.
    :type method: string

    :return: A Series with phonetic encoded values.
    :rtype: pandas.Series
    """


    try:
        import jellyfish
    except ImportError:
        print ("Install jellyfish to use string encoding.")

    s = clean(s, replace_by_none='[^\-\_A-Za-z0-9]+')
 
    if method == 'soundex':
        return s.str.upper().apply(lambda x: jellyfish.soundex(x) if pd.notnull(x) else np.nan)

    elif method == 'nysiis':
        return s.str.upper().apply(lambda x: jellyfish.nysiis(x) if pd.notnull(x) else np.nan)

    else:
        raise Exception("Phonetic encoding method not found")

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
        print ("Install jellyfish to use string encoding.")

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

