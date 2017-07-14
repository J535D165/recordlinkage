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


def smith_waterman(s1, s2, match=5, mismatch=-5, gap_start=-5, gap_continue=-1, norm="mean"):
    """
    string(s1, s2, match=1, mismatch=-1, gap_start=-1, gap_continue=-0.2, norm="mean")

    An implementation of the Smith-Waterman string comparison algorithm.

    Parameters
    ----------
    s1 : label, pandas.Series
        Series or DataFrame to compare all fields.
    s2 : label, pandas.Series
        Series or DataFrame to compare all fields.
    match : float
        The value added to the match score if two characters match.
        Greater than mismatch, gap_start, and gap_continue. Default: 1.
    mismatch : float
        The value added to the match score if two characters do not match.
        Less than match. Default: -1.
    gap_start : float
        The value added to the match score upon encountering the start of
        a gap. Default: -1.
    gap_continue : float
        The value added to the match score for positions where a previously
        started gap is continuing. Default: -0.2.
    norm : str
        The name of the normalization metric to be used. Applied by dividing
        the match score by the normalization metric multiplied by match. One
        of "min", "max",or "mean". "min" will use the minimum string length
        as the normalization metric. "max" and "mean" use the maximum and
        mean string length respectively. Default: "mean""

    Returns
    -------
    pandas.Series
        A pandas series with similarity values. Values equal or between 0
        and 1.
    """
    # Assert that match is greater than or equal to mismatch, gap_start, and gap_continue.
    assert match >= max(mismatch, gap_start, gap_continue), \
        "match must be greater than or equal to mismatch, gap_start, and gap_continue"

    if len(s1) != len(s2):
        raise ValueError('Arrays or Series have to be same length.')

    concat = pandas.concat([s1, s2], axis=1, ignore_index=True)

    def sw_apply(t):
        str1 = t[0]
        str2 = t[1]

        def compute_score():
            # Initialize the score matrix
            m = [[0] * (1 + len(str2)) for i in range(1 + len(str1))]
            # Initialize the trace matrix the initial
            t = [[[] for _ in range(1 + len(str2))] for _ in range(1 + len(str1))]

            highest = 0

            for x in range(1, 1 + len(str1)):
                for y in range(1, 1 + len(str2)):
                    if str1[x-1] == str2[y-1]:
                        diagonal = m[x-1][y-1] + match
                    else:
                        diagonal = m[x-1][y-1] + mismatch

                    if "L" in t[x-1][y]:
                        gap_left = m[x-1][y] + gap_continue
                    else:
                        gap_left = m[x-1][y] + gap_start

                    if "A" in t[x][y-1]:
                        gap_above = m[x][y-1] + gap_continue
                    else:
                        gap_above = m[x][y-1] + gap_start

                    score = max(diagonal, gap_left, gap_above)

                    if score <= 0:
                        score = 0
                    else:
                        if score == diagonal:
                            t[x][y].append("D")
                        if score == gap_above:
                            t[x][y].append("A")
                        if score == gap_left:
                            t[x][y].append("L")

                    if score > highest:
                        highest = score

                    m[x][y] = score

            return highest

        def normalize(score):
            if norm == "min":
                return score/(min(len(str1), len(str2)) * match)
            if norm == "max":
                return score/(max(len(str1), len(str2)) * match)
            if norm == "mean":
                return 2*score/((len(str1) + len(str2)) * match)

        try:
            if len(str1) == 0 or len(str2) == 0:
                return 0
            return normalize(compute_score())
        except Exception as err:
            if pandas.isnull(t[0]) or pandas.isnull(t[1]):
                return 0
            else:
                raise err

    return concat.apply(sw_apply, axis=1)
