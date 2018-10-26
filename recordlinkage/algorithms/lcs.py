from __future__ import division
from __future__ import unicode_literals

import warnings

import pandas
import numpy as np


def memoize(func):
    def memoized_func(*args):
        cache = args[0].cache
        if args[1:] in cache:
            return cache[args[1:]]
        result = func(*args)
        cache[args[1:]] = result
        return result

    return memoized_func


class LCS:
    """
    Code refactored to be a class instead of the single method. Allow to cache intermediate results.

    An implementation of the longest common substring similarity algorithm
    described in Christen, Peter (2012) with caching to save computational resources.

    Attributes
    ----------
    s1 : label, pandas.Series
        Series or DataFrame to compare all fields.
    s2 : label, pandas.Series
        Series or DataFrame to compare all fields.
    norm : str
        The name of the normalization applied to the raw length computed by
        the lcs algorithm. One of "overlap", "jaccard", or "dice". Default:
        "dice""
    min_len: int
        Minimal length of the strings to be compared.
    cache: Dict()
        Intermediate dictionary to cache comparison results.
    """
    cache = dict()

    def __init__(self, s1, s2, norm='dice', min_len=2):
        self.s1 = s1
        self.s2 = s2
        self.norm = norm
        self.min_len = min_len

    @memoize
    def lcs_iteration(self, str1, str2):
        """
        lcs_iteration(str1, str2)

        A helper function implementation of a single iteration longest common substring algorithm,
        adapted from
        https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring.
        but oriented towards the iterative approach described by Christen, Peter (2012).

        Parameters
        ----------
        str1 : string to be compared
        str2: string to be compared

        Returns
        -------
        A tuple of strings and a substring length i.e. ((str, str), int).
        """

        if str1 is np.nan or str2 is np.nan or min(len(str1), len(str2)) < self.min_len:
            longest = 0
            new_str1 = None
            new_str2 = None
        else:
            # Creating a matrix of 0s for preprocessing
            m = [[0] * (1 + len(str2)) for _ in range(1 + len(str1))]

            # Track length of longest substring seen
            longest = 0

            # Track the ending position of this substring in str1 (x) and str2(y)
            x_longest = 0
            y_longest = 0

            # Create matrix of substring lengths
            for x in range(1, 1 + len(str1)):
                for y in range(1, 1 + len(str2)):
                    # Check if the chars match
                    if str1[x - 1] == str2[y - 1]:
                        # add 1 to the diagonal
                        m[x][y] = m[x - 1][y - 1] + 1
                        # Update values if longer than previous longest substring
                        if m[x][y] > longest:
                            longest = m[x][y]
                            x_longest = x
                            y_longest = y
                    else:
                        # If there is no match, start from zero
                        m[x][y] = 0

            # Copy str1 and str2, but subtract the longest common substring
            # for the next iteration.
            new_str1 = str1[0:x_longest - longest] + str1[x_longest:]
            new_str2 = str2[0:y_longest - longest] + str2[y_longest:]

        return (new_str1, new_str2), longest

    def normalize_lcs(self, lcs_value, x):
        """
        normalize_lcs(lcs_value, x)

        A helper function used to normalize the score produced by
        compute_score() to a score between 0 and 1. Applies one of
        the normalization schemes described in in Christen, Peter (2012).
        The normalization method is determined by the norm argument provided
        to the parent, longest_common_substring_similarity function.

        Parameters
        ----------
        lcs_value : Float
            The raw lcs length.
        x : pandas.Series
            A pandas Series containing two strings to be compared.

        Returns
        -------
        Float
            The normalized lcs length.
        """
        if len(x[0]) == 0 or len(x[1]) == 0:
            return 0
        if self.norm == 'overlap':
            return lcs_value / min(len(x[0]), len(x[1]))
        elif self.norm == 'jaccard':
            return lcs_value / (len(x[0]) + len(x[1]) - abs(lcs_value))
        elif self.norm == 'dice':
            return lcs_value * 2 / (len(x[0]) + len(x[1]))
        else:
            warnings.warn('Unrecognized longest common substring normalization. Defaulting to "dice" method.')
            return lcs_value * 2 / (len(x[0]) + len(x[1]))

    def lcs_apply(self, x):
        """
        lcs_apply(x)

        A helper function that is applied to each pair of records
        in s1 and s2. Assigns a similarity score to each pair,
        between 0 and 1. Used by the pandas.apply method.

        Parameters
        ----------
        x : pandas.Series
            A pandas Series containing two strings to be compared.

        Returns
        -------
        Float
            A normalized similarity score.
        """
        if pandas.isnull(x[0]) or pandas.isnull(x[1]):
            return np.nan

        # Compute lcs value with first ordering.
        lcs_acc_1 = 0
        new_x_1 = (x[0], x[1])
        while True:
            # Get new string pair (iter_x) and length (iter_lcs)
            # for this iteration.
            iter_x, iter_lcs = self.lcs_iteration(new_x_1[0], new_x_1[1])
            if iter_lcs < self.min_len:
                # End if the longest substring is below the threshold
                break
            else:
                # Otherwise, accumulate length and start a new iteration
                # with the new string pair.
                new_x_1 = iter_x
                lcs_acc_1 = lcs_acc_1 + iter_lcs

        # Compute lcs value with second ordering.
        lcs_acc_2 = 0
        new_x_2 = (x[1], x[0])
        while True:
            # Get new string pair (iter_x) and length (iter_lcs)
            # for this iteration.
            iter_x, iter_lcs = self.lcs_iteration(new_x_2[0], new_x_2[1])
            if iter_lcs < self.min_len:
                # End if the longest substring is below the threshold
                break
            else:
                # Otherwise, accumulate length and start a new iteration
                # with the new string pair.
                new_x_2 = iter_x
                lcs_acc_2 = lcs_acc_2 + iter_lcs

        # Average the two orderings, since lcs may be sensitive to comparison order.
        return (self.normalize_lcs(lcs_acc_1, x) + self.normalize_lcs(lcs_acc_2, x)) / 2

    def longest_common_substring_similarity(self):
        """
        longest_common_substring_similarity()

        Returns
        -------
        pandas.Series
            A pandas series with normalized similarity values.
        """

        if len(self.s1) != len(self.s2):
            raise ValueError('Arrays or Series have to be same length.')

        if len(self.s1) == len(self.s2) == 0:
            return []

        conc = pandas.Series(list(zip(self.s1, self.s2)))

        return conc.apply(self.lcs_apply)
