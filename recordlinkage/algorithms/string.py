import warnings

import numpy as np
import pandas
from sklearn.feature_extraction.text import CountVectorizer

# Ingore zero devision errors in cosine and qgram algorithms
# warnings.filterwarnings("ignore")

################################
#      STRING SIMILARITY       #
################################


def jaro_similarity(s1, s2):
    conc = pandas.Series(list(zip(s1, s2)))

    from jellyfish import jaro_similarity

    def jaro_apply(x):
        try:
            return jaro_similarity(x[0], x[1])
        except Exception as err:
            if pandas.isnull(x[0]) or pandas.isnull(x[1]):
                return np.nan
            else:
                raise err

    return conc.apply(jaro_apply)


def jarowinkler_similarity(s1, s2):
    conc = pandas.Series(list(zip(s1, s2)))

    from jellyfish import jaro_winkler_similarity

    def jaro_winkler_apply(x):
        try:
            return jaro_winkler_similarity(x[0], x[1])
        except Exception as err:
            if pandas.isnull(x[0]) or pandas.isnull(x[1]):
                return np.nan
            else:
                raise err

    return conc.apply(jaro_winkler_apply)


def levenshtein_similarity(s1, s2):
    conc = pandas.Series(list(zip(s1, s2)))

    from jellyfish import levenshtein_distance

    def levenshtein_apply(x):
        try:
            return 1 - levenshtein_distance(x[0], x[1]) / np.max([len(x[0]), len(x[1])])
        except Exception as err:
            if pandas.isnull(x[0]) or pandas.isnull(x[1]):
                return np.nan
            else:
                raise err

    return conc.apply(levenshtein_apply)


def damerau_levenshtein_similarity(s1, s2):
    conc = pandas.Series(list(zip(s1, s2)))

    from jellyfish import damerau_levenshtein_distance

    def damerau_levenshtein_apply(x):
        try:
            return 1 - damerau_levenshtein_distance(x[0], x[1]) / np.max(
                [len(x[0]), len(x[1])]
            )
        except Exception as err:
            if pandas.isnull(x[0]) or pandas.isnull(x[1]):
                return np.nan
            else:
                raise err

    return conc.apply(damerau_levenshtein_apply)


def qgram_similarity(s1, s2, include_wb=True, ngram=(2, 2)):
    if len(s1) != len(s2):
        raise ValueError("Arrays or Series have to be same length.")

    if len(s1) == len(s2) == 0:
        return []

    # include word boundaries or not
    analyzer = "char_wb" if include_wb is True else "char"

    # prepare data
    data = pandas.concat([s1, s2]).fillna("")

    # The vectorizer
    vectorizer = CountVectorizer(
        analyzer=analyzer, strip_accents="unicode", ngram_range=ngram
    )

    vec_fit = vectorizer.fit_transform(data)

    def _metric_sparse_euclidean(u, v):
        match_ngrams = u.minimum(v).sum(axis=1)
        total_ngrams = np.maximum(u.sum(axis=1), v.sum(axis=1))

        # division by zero is not possible in our case, but 0/0 is possible.
        # Numpy raises a warning in that case.

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.true_divide(match_ngrams, total_ngrams).A1

        return m

    return _metric_sparse_euclidean(vec_fit[: len(s1)], vec_fit[len(s1) :])


def cosine_similarity(s1, s2, include_wb=True, ngram=(2, 2)):
    if len(s1) != len(s2):
        raise ValueError("Arrays or Series have to be same length.")

    if len(s1) == len(s2) == 0:
        return []

    # include word boundaries or not
    analyzer = "char_wb" if include_wb is True else "char"

    # The vectorizer
    vectorizer = CountVectorizer(
        analyzer=analyzer, strip_accents="unicode", ngram_range=ngram
    )

    data = pandas.concat([s1, s2]).fillna("")

    vec_fit = vectorizer.fit_transform(data)

    def _metric_sparse_cosine(u, v):
        a = np.sqrt(u.multiply(u).sum(axis=1))
        b = np.sqrt(v.multiply(v).sum(axis=1))

        ab = v.multiply(u).sum(axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.divide(ab, np.multiply(a, b)).A1

        return m

    return _metric_sparse_cosine(vec_fit[: len(s1)], vec_fit[len(s1) :])


def smith_waterman_similarity(
    s1, s2, match=5, mismatch=-5, gap_start=-5, gap_continue=-1, norm="mean"
):
    """Smith-Waterman string comparison.

    An implementation of the Smith-Waterman string comparison algorithm
    described in Christen, Peter (2012).

    Parameters
    ----------
    s1 : label, pandas.Series
        Series or DataFrame to compare all fields.
    s2 : label, pandas.Series
        Series or DataFrame to compare all fields.
    match : float
        The value added to the match score if two characters match.
        Greater than mismatch, gap_start, and gap_continue. Default: 5.
    mismatch : float
        The value added to the match score if two characters do not match.
        Less than match. Default: -5.
    gap_start : float
        The value added to the match score upon encountering the start of
        a gap. Default: -5.
    gap_continue : float
        The value added to the match score for positions where a previously
        started gap is continuing. Default: -1.
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

    # Assert that match is greater than or equal to mismatch, gap_start, and
    # gap_continue.
    assert match >= max(mismatch, gap_start, gap_continue), (
        "match must be greater than or equal to mismatch, "
        "gap_start, and gap_continue"
    )

    if len(s1) != len(s2):
        raise ValueError("Arrays or Series have to be same length.")

    if len(s1) == len(s2) == 0:
        return []

    concat = pandas.Series(list(zip(s1, s2)))

    def sw_apply(t):
        """
        sw_apply(t)

        A helper function that is applied to each pair of records
        in s1 and s2. Assigns a similarity score to each pair,
        between 0 and 1. Used by the pandas.apply method.

        Parameters
        ----------
        t : pandas.Series
            A pandas Series containing two strings to be compared.

        Returns
        -------
        Float
            A similarity score between 0 and 1.
        """
        str1 = t[0]
        str2 = t[1]

        def compute_score():
            """
            compute_score()

            The helper function that produces the non-normalized
            similarity score between two strings. The scores are
            determined using the Smith-Waterman dynamic programming
            algorithm. The scoring scheme is determined from the
            parameters provided to the parent smith_waterman_similarity()
            function.

            Returns
            -------
            Float
                A score 0 or greater. Indicates similarity between two strings.
            """

            # Initialize the score matrix with 0s

            m = [[0] * (1 + len(str2)) for i in range(1 + len(str1))]

            # Initialize the trace matrix with empty lists
            trace = [[[] for _ in range(1 + len(str2))] for _ in range(1 + len(str1))]

            # Initialize the highest seen score to 0
            highest = 0

            # Iterate through the matrix
            for x in range(1, 1 + len(str1)):
                for y in range(1, 1 + len(str2)):
                    # Calculate Diagonal Score
                    if str1[x - 1] == str2[y - 1]:
                        # If characters match, add the match score to the
                        # diagonal score
                        diagonal = m[x - 1][y - 1] + match
                    else:
                        # If characters do not match, add the mismatch score
                        # to the diagonal score
                        diagonal = m[x - 1][y - 1] + mismatch

                    # Calculate the Left Gap Score
                    if "H" in trace[x - 1][y]:
                        # If cell to the left's score was calculated based on
                        # a horizontal gap, add the gap continuation penalty
                        # to the left score.
                        gap_horizontal = m[x - 1][y] + gap_continue
                    else:
                        # Otherwise, add the gap start penalty to the left
                        # score
                        gap_horizontal = m[x - 1][y] + gap_start

                    # Calculate the Above Gap Score
                    if "V" in trace[x][y - 1]:
                        # If above cell's score was calculated based on a
                        # vertical gap, add the gap continuation penalty to
                        # the above score.
                        gap_vertical = m[x][y - 1] + gap_continue
                    else:
                        # Otherwise, add the gap start penalty to the above
                        # score
                        gap_vertical = m[x][y - 1] + gap_start

                    # Choose the highest of the three scores
                    score = max(diagonal, gap_horizontal, gap_vertical)

                    if score <= 0:
                        # If score is less than 0, boost to 0
                        score = 0
                    else:
                        # If score is greater than 0, determine whether it was
                        # calculated based on a diagonal score, horizontal gap,
                        # or vertical gap. Store D, H, or V in the trace matrix
                        # accordingly.
                        if score == diagonal:
                            trace[x][y].append("D")
                        if score == gap_horizontal:
                            trace[x][y].append("H")
                        if score == gap_vertical:
                            trace[x][y].append("V")

                    # If the cell's score is greater than the highest score
                    # previously present, record the score as the highest.
                    if score > highest:
                        highest = score

                    # Set the cell's score to score
                    m[x][y] = score

            # After iterating through the entire matrix, return the highest
            # score found.
            return highest

        def normalize(score):
            """
            normalize(score)

            A helper function used to normalize the score produced by
            compute_score() to a score between 0 and 1. The method for
            normalization is determined by the norm argument provided
            to the parent, smith_waterman_similarity function.

            Parameters
            ----------
            score : Float
                The score produced by the compute_score() function.

            Returns
            -------
            Float
                A normalized score between 0 and 1.
            """
            if norm == "min":
                # Normalize by the shorter string's length
                return score / (min(len(str1), len(str2)) * match)
            if norm == "max":
                # Normalize by the longer string's length
                return score / (max(len(str1), len(str2)) * match)
            if norm == "mean":
                # Normalize by the mean length of the two strings
                return 2 * score / ((len(str1) + len(str2)) * match)
            else:
                warnings.warn(
                    "Unrecognized longest common substring normalization. "
                    'Defaulting to "mean" method.',
                    stacklevel=2,
                )
                return 2 * score / ((len(str1) + len(str2)) * match)

        try:
            if len(str1) == 0 or len(str2) == 0:
                return 0
            return normalize(compute_score())

        except Exception as err:
            if pandas.isnull(t[0]) or pandas.isnull(t[1]):
                return np.nan
            else:
                raise err

    return concat.apply(sw_apply)


def longest_common_substring_similarity(s1, s2, norm="dice", min_len=2):
    """
    longest_common_substring_similarity(s1, s2, norm='dice', min_len=2)

    An implementation of the longest common substring similarity algorithm
    described in Christen, Peter (2012).

    Parameters
    ----------
    s1 : label, pandas.Series
        Series or DataFrame to compare all fields.
    s2 : label, pandas.Series
        Series or DataFrame to compare all fields.
    norm : str
        The name of the normalization applied to the raw length computed by
        the lcs algorithm. One of "overlap", "jaccard", or "dice". Default:
        "dice""

    Returns
    -------
    pandas.Series
        A pandas series with normalized similarity values.
    """

    if len(s1) != len(s2):
        raise ValueError("Arrays or Series have to be same length.")

    if len(s1) == len(s2) == 0:
        return []

    conc = pandas.Series(list(zip(s1, s2)))

    def lcs_iteration(x):
        """
        lcs_iteration(x)

        A helper function implementation of a single iteration longest common
        substring algorithm, adapted from https://en.wikibooks.org/wiki/Algori
        thm_Implementation/Strings/Longest_common_substring. but oriented
        towards the iterative approach described by Christen, Peter (2012).

        Parameters
        ----------
        x : A series containing the two strings to be compared.

        Returns
        -------
        A tuple of strings and a substring length i.e. ((str, str), int).
        """

        str1 = x[0]
        str2 = x[1]

        if str1 is np.nan or str2 is np.nan or min(len(str1), len(str2)) < min_len:
            longest = 0
            new_str1 = None
            new_str2 = None
        else:
            # Creating a matrix of 0s for preprocessing
            m = [[0] * (1 + len(str2)) for _ in range(1 + len(str1))]

            # Track length of longest substring seen
            longest = 0

            # Track the ending position of this substring in str1 (x) and
            # str2(y)
            x_longest = 0
            y_longest = 0

            # Create matrix of substring lengths
            for x in range(1, 1 + len(str1)):
                for y in range(1, 1 + len(str2)):
                    # Check if the chars match
                    if str1[x - 1] == str2[y - 1]:
                        # add 1 to the diagonal
                        m[x][y] = m[x - 1][y - 1] + 1
                        # Update values if longer than previous longest
                        # substring
                        if m[x][y] > longest:
                            longest = m[x][y]
                            x_longest = x
                            y_longest = y
                    else:
                        # If there is no match, start from zero
                        m[x][y] = 0

            # Copy str1 and str2, but subtract the longest common substring
            # for the next iteration.
            new_str1 = str1[0 : x_longest - longest] + str1[x_longest:]
            new_str2 = str2[0 : y_longest - longest] + str2[y_longest:]

        return (new_str1, new_str2), longest

    def lcs_apply(x):
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
            iter_x, iter_lcs = lcs_iteration(new_x_1)
            if iter_lcs < min_len:
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
            iter_x, iter_lcs = lcs_iteration(new_x_2)
            if iter_lcs < min_len:
                # End if the longest substring is below the threshold
                break
            else:
                # Otherwise, accumulate length and start a new iteration
                # with the new string pair.
                new_x_2 = iter_x
                lcs_acc_2 = lcs_acc_2 + iter_lcs

        def normalize_lcs(lcs_value):
            """
            normalize_lcs(lcs_value)

            A helper function used to normalize the score produced by
            compute_score() to a score between 0 and 1. Applies one of the
            normalization schemes described in in Christen, Peter (2012). The
            normalization method is determined by the norm argument provided
            to the parent, longest_common_substring_similarity function.

            Parameters
            ----------
            lcs_value : Float
                The raw lcs length.

            Returns
            -------
            Float
                The normalized lcs length.
            """
            if len(x[0]) == 0 or len(x[1]) == 0:
                return 0
            if norm == "overlap":
                return lcs_value / min(len(x[0]), len(x[1]))
            elif norm == "jaccard":
                return lcs_value / (len(x[0]) + len(x[1]) - abs(lcs_value))
            elif norm == "dice":
                return lcs_value * 2 / (len(x[0]) + len(x[1]))
            else:
                warnings.warn(
                    "Unrecognized longest common substring normalization. "
                    'Defaulting to "dice" method.',
                    stacklevel=2,
                )
                return lcs_value * 2 / (len(x[0]) + len(x[1]))

        # Average the two orderings, since lcs may be sensitive to comparison
        # order.
        return (normalize_lcs(lcs_acc_1) + normalize_lcs(lcs_acc_2)) / 2

    return conc.apply(lcs_apply)
