# measures.py

from __future__ import division

import numpy
import pandas


def reduction_ratio(n, x):
    """Compute the reduction ratio.

    The reduction ratio is 1 minus the ratio candidate matches and the maximum
    number of pairs possible.

    Parameters
    ----------
    n: int, pandas.MultiIndex
        The number of candidate record pairs or the pandas.MultiIndex with
        record pairs.
    x: a list of pandas.DataFrame objects
        The data used to make the candidate record pairs.

    Returns
    -------
    float
        The reduction ratio.

    """

    n_max = max_pairs(x)

    if isinstance(n, pandas.MultiIndex):
        n = len(n)

    if n > n_max:
        raise ValueError("n has to be smaller of equal n_max")

    return 1 - n / n_max


def recall_score(candidate_links, true_matches):

    return len(true_matches.intersection(candidate_links)) / len(true_matches)


def _get_len(x):
    """Return int or len(x)"""

    return x if isinstance(x, int) else len(x)


def _max_pairs_deduplication(x):
    """Compute the maximum number of record pairs in case of deduplication."""

    return int(x * (x - 1) / 2)


def _max_pairs_linkage(x):
    """Get the maximum number of record pairs in case of linking."""
    return numpy.prod(x)


def max_pairs(shape):
    """Compute the maximum number of record pairs possible."""

    if not isinstance(shape, (tuple, list)):
        n = _max_pairs_deduplication(_get_len(shape))

    elif (isinstance(shape, (tuple, list)) and len(shape) == 1):
        n = _max_pairs_deduplication(_get_len(shape[0]))

    else:
        n = _max_pairs_linkage([_get_len(xi) for xi in shape])

    return n


def true_positives(true_match_index, matches_index):
    """Count the number of True Positives.

    Return the number of correctly classified links, also called the number of
    True Positives (TP).

    Parameters
    ----------
    true_match_index: pandas.MultiIndex
        The golden/true links.
    match_index: pandas.MultiIndex
        The classified links.

    Returns
    -------
    int
        The number of correctly classified links.
    """

    return len(true_match_index & matches_index)


def true_negatives(true_match_index, matches_index, n_pairs):
    """Count the number of True Negatives.

    Return the number of correctly classified non-links, also called the
    number of True Negatives (TN).

    Parameters
    ----------
    true_match_index: pandas.MultiIndex
        The golden/true links.
    match_index: pandas.MultiIndex
        The classified links.
    n_pairs: int
        The number of candidate record pairs.

    Returns
    -------
    int
        The number of correctly classified non-links.

    """

    if not isinstance(n_pairs, (int, float)):
        n_pairs = len(n_pairs)

    return int(n_pairs) - len(true_match_index | matches_index)


def false_positives(true_match_index, matches_index):
    """Count the number of false positives.

    Return the number of predicted links, while the record pairs belong to
    different entities. This values is known as the number of False Positives
    (FP).

    Parameters
    ----------
    true_match_index: pandas.MultiIndex
        The golden/true links.
    match_index: pandas.MultiIndex
        The classified links.

    Returns
    -------
    int
        The number of false positives.

    """

    # The classified matches without the true matches.
    return len(matches_index.difference(true_match_index))


def false_negatives(true_match_index, matches_index):
    """Count the number of False Negatives.

    Return the number of predicted non-links, while the record pairs belong to
    the same entity. This values is known as the number of False Negatives
    (FN).

    Parameters
    ----------
    true_match_index: pandas.MultiIndex
        The golden/true links.
    match_index: pandas.MultiIndex
        The classified links.

    Returns
    -------
    int
        The number of false negatives.

    """
    return len(true_match_index.difference(matches_index))


def confusion_matrix(true_match_index, matches_index, n_pairs):
    """Compute the confusion matrix.

    The confusion matrix is of the following
    form:

    +---------------------+-----------------------+----------------------+
    |                     |  Predicted Positive   | Predicted Negatives  |
    +=====================+=======================+======================+
    | **True Positive**   | True Positives (TP)   | False Negatives (FN) |
    +---------------------+-----------------------+----------------------+
    | **True Negative**   | False Positives (FP)  | True Negatives (TN)  |
    +---------------------+-----------------------+----------------------+

    The confusion matrix is used to compute measures like precision and recall.

    Parameters
    ----------
    true_match_index: pandas.MultiIndex
        The golden/true links.
    match_index: pandas.MultiIndex
        The classified links.
    n_pairs: int
        The number of record pairs analysed.

    Returns
    -------
    numpy.array
        The confusion matrix with TP, TN, FN, FP values.

    """

    # True positives
    tp = true_positives(true_match_index, matches_index)

    # True negatives
    tn = true_negatives(true_match_index, matches_index, n_pairs)

    # False positives
    fp = false_positives(true_match_index, matches_index)

    # False negatives
    fn = false_negatives(true_match_index, matches_index)

    return numpy.array([[tp, fn], [fp, tn]])


def precision(confusion_matrix):
    """Compute the precision.

    The precision is given by tp/(tp+fp).

    Parameters
    ----------
    confusion_matrix: numpy.array
        The matrix with tp, fn, fp, tn values.

    Returns
    -------
    float
        The precision
    """

    v = confusion_matrix[0, 0] \
        / (confusion_matrix[0, 0] + confusion_matrix[1, 0])

    return float(v)


def recall(confusion_matrix):
    """Compute the recall/sensitivity.

    The recall is given by tp/(tp+fn).

    Parameters
    ----------
    confusion_matrix: numpy.array
        The matrix with tp, fn, fp, tn values.

    Returns
    -------
    float
        The recall
    """

    v = confusion_matrix[0, 0] \
        / (confusion_matrix[0, 0] + confusion_matrix[0, 1])

    return float(v)


def accuracy(confusion_matrix):
    """Compute the accuracy.

    The accuracy is given by (tp+tn)/(tp+fp+tn+fn).

    Parameters
    ----------
    confusion_matrix: numpy.array
        The matrix with tp, fn, fp, tn values.

    Returns
    -------
    float
        The accuracy
    """

    v = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) \
        / numpy.sum(confusion_matrix)

    return float(v)


def specificity(confusion_matrix):
    """Compute the specificity.

    The specificity is given by tn/(fp+tn).

    Parameters
    ----------
    confusion_matrix: numpy.array
        The matrix with tp, fn, fp, tn values.

    Returns
    -------
    float
        The specificity

    """

    v = confusion_matrix[1, 1] \
        / (confusion_matrix[1, 1] + confusion_matrix[1, 0])

    return float(v)


def fscore(confusion_matrix):
    """Compute the f_score.

    The fscore is given by 2*(precision*recall)/(precision+recall).

    Parameters
    ----------
    confusion_matrix: float
        The matrix with tp, fn, fp, tn values.

    Returns
    -------
    float
        The fscore

    Note
    ----
    If there are no pairs classified as links, this measure will raise a
    ZeroDivisionError.

     """

    prec = precision(confusion_matrix)
    rec = recall(confusion_matrix)

    return float(2 * prec * rec / (prec + rec))
