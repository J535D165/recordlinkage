from __future__ import division

import warnings

import pandas
import numpy as np
import jellyfish

from sklearn.feature_extraction.text import CountVectorizer


# Ingore zero devision errors in cosine and qgram algorithms
warnings.filterwarnings("ignore")

################################
#      STRING SIMILARITY       #
################################


def jaro_similarity(s1, s2):

    conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

    def jaro_apply(x):

        try:
            return jellyfish.jaro_distance(x[0], x[1])
        except Exception as err:
            if pandas.isnull(x[0]) or pandas.isnull(x[1]):
                return np.nan
            else:
                raise err

    return conc.apply(jaro_apply, axis=1)


def jarowinkler_similarity(s1, s2):

    conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

    def jaro_winkler_apply(x):

        try:
            return jellyfish.jaro_winkler(x[0], x[1])
        except Exception as err:
            if pandas.isnull(x[0]) or pandas.isnull(x[1]):
                return np.nan
            else:
                raise err

    return conc.apply(jaro_winkler_apply, axis=1)


def levenshtein_similarity(s1, s2):

    conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

    def levenshtein_apply(x):

        try:
            return 1 - jellyfish.levenshtein_distance(x[0], x[1]) \
                / np.max([len(x[0]), len(x[1])])
        except Exception as err:
            if pandas.isnull(x[0]) or pandas.isnull(x[1]):
                return np.nan
            else:
                raise err

    return conc.apply(levenshtein_apply, axis=1)


def damerau_levenshtein_similarity(s1, s2):

    conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

    def damerau_levenshtein_apply(x):

        try:
            return 1 - jellyfish.damerau_levenshtein_distance(x[0], x[1]) \
                / np.max([len(x[0]), len(x[1])])
        except Exception as err:
            if pandas.isnull(x[0]) or pandas.isnull(x[1]):
                return np.nan
            else:
                raise err

    return conc.apply(damerau_levenshtein_apply, axis=1)


def qgram_similarity(s1, s2, include_wb=True, ngram=(2, 2)):

    if len(s1) != len(s2):
        raise ValueError('Arrays or Series have to be same length.')

    if len(s1) == len(s2) == 0:
        return []

    # include word boundaries or not
    analyzer = 'char_wb' if include_wb is True else 'char'

    # The vectorizer
    vectorizer = CountVectorizer(
        analyzer=analyzer, strip_accents='unicode', ngram_range=ngram)

    data = s1.append(s2).fillna('')

    vec_fit = vectorizer.fit_transform(data)

    def _metric_sparse_euclidean(u, v):

        match_ngrams = u.minimum(v).sum(axis=1)
        total_ngrams = np.maximum(u.sum(axis=1), v.sum(axis=1))

        # division by zero is not possible in our case, but 0/0 is possible.
        # Numpy raises a warning in that case.
        return np.true_divide(match_ngrams, total_ngrams).A1

    return _metric_sparse_euclidean(vec_fit[:len(s1)], vec_fit[len(s1):])


def cosine_similarity(s1, s2, include_wb=True, ngram=(2, 2)):

    if len(s1) != len(s2):
        raise ValueError('Arrays or Series have to be same length.')

    if len(s1) == len(s2) == 0:
        return []

    # include word boundaries or not
    analyzer = 'char_wb' if include_wb is True else 'char'

    # The vectorizer
    vectorizer = CountVectorizer(
        analyzer=analyzer, strip_accents='unicode', ngram_range=ngram)

    data = s1.append(s2).fillna('')

    vec_fit = vectorizer.fit_transform(data)

    def _metric_sparse_cosine(u, v):

        a = np.sqrt(u.multiply(u).sum(axis=1))
        b = np.sqrt(v.multiply(v).sum(axis=1))

        ab = v.multiply(u).sum(axis=1)

        return np.divide(ab, np.multiply(a, b)).A1

    return _metric_sparse_cosine(vec_fit[:len(s1)], vec_fit[len(s1):])
