# measures.py

from __future__ import division

import numpy


def true_positives(true_match_index, matches_index):
    """

    Return the number of correctly classified links, also called the number of
    True Positives (TP).

    :param true_match_index: The golden/true links.
    :param match_index: The classified links.

    :type true_match_index: pandas.MultiIndex
    :type match_index: pandas.MultiIndex

    :return: The number of correctly classified links.
    :rtype: int
    """

    return len(true_match_index & matches_index)


def true_negatives(true_match_index, matches_index, n_pairs):
    """

    Return the number of correctly classified non-links, also called the
    number of True Negatives (TN).

    :param true_match_index: The golden/true links.
    :param match_index: The classified links.
    :param n_pairs: The number of candidate record pairs.
            The number of record pairs analysed.

    :type true_match_index: pandas.MultiIndex
    :type match_index: pandas.MultiIndex
    :type n_pairs: int


    :return: The number of correctly classified non-links.
    :rtype: int
    """

    if not isinstance(n_pairs, (int, float)):
        n_pairs = len(n_pairs)

    return int(n_pairs) - len(true_match_index | matches_index)


def false_positives(true_match_index, matches_index):
    """

    Return the number of predicted links, while the record pairs belong to
    different entities. This values is known as the number of False Positives
    (FP).

    :param true_match_index: The golden/true links.
    :param match_index: The classified links.

    :type true_match_index: pandas.MultiIndex
    :type match_index: pandas.MultiIndex

    :return: The number of false positives.
    :rtype: int
    """

    # The classified matches without the true matches.
    return len(matches_index.difference(true_match_index))


def false_negatives(true_match_index, matches_index):
    """

    Return the number of predicted non-links, while the record pairs belong to
    the same entity. This values is known as the number of False Negatives
    (FN).

    :param true_match_index: The golden/true links.
    :param match_index: The classified links.

    :type true_match_index: pandas.MultiIndex
    :type match_index: pandas.MultiIndex

    :return: The number of false negatives.
    :rtype: int

    """
    return len(true_match_index.difference(matches_index))


def confusion_matrix(true_match_index, matches_index, n_pairs):
    """

    Compute the confusion matrix. The confusion matrix is of the following
    form:

    +---------------------+-----------------------+----------------------+
    |                     |  Predicted Positive   | Predicted Negatives  |
    +=====================+=======================+======================+
    | **True Positive**   | True Positives (TP)   | False Negatives (FN) |
    +---------------------+-----------------------+----------------------+
    | **True Negative**   | False Positives (FP)  | True Negatives (TN)  |
    +---------------------+-----------------------+----------------------+

    The confusion matrix is used to compute measures like precision and recall.

    :param true_match_index: The golden/true links.
    :param match_index: The classified links.
    :param n_pairs: The number of record pairs analysed.

    :type true_match_index: pandas.MultiIndex
    :type match_index: pandas.MultiIndex
    :type n_pairs: int

    :return: The confusion matrix with TP, TN, FN, FP values.
    :rtype: numpy.array
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
    """ Compute the precision

    The precision is given by tp/(tp+fp).

    :param confusion_matrix: The matrix with tp, fn, fp, tn values.

    :return: The precision
    :rtype: float
    """

    v = confusion_matrix[0, 0] \
        / (confusion_matrix[0, 0] + confusion_matrix[1, 0])

    return float(v)


def recall(confusion_matrix):
    """ Compute the recall/sensitivity

    The recall is given by tp/(tp+fn).

    :param confusion_matrix: The matrix with tp, fn, fp, tn values.

    :return: The recall
    :rtype: float
    """

    v = confusion_matrix[0, 0] \
        / (confusion_matrix[0, 0] + confusion_matrix[0, 1])

    return float(v)


def accuracy(confusion_matrix):
    """ Compute the accuracy

    The accuracy is given by (tp+tn)/(tp+fp+tn+fn).

    :param confusion_matrix: The matrix with tp, fn, fp, tn values.

    :return: The accuracy
    :rtype: float
    """

    v = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) \
        / numpy.sum(confusion_matrix)

    return float(v)


def specificity(confusion_matrix):
    """ Compute the specitivity

    The specitivity is given by tn/(fp+tn).

    :param confusion_matrix: The matrix with tp, fn, fp, tn values.

    :return: The accuracy
    :rtype: float
    """

    v = confusion_matrix[1, 1] \
        / (confusion_matrix[1, 1] + confusion_matrix[1, 0])

    return float(v)


def fscore(confusion_matrix):
    """ Compute the f_score

    The fscore is given by 2*(precision*recall)/(precision+recall).

    :note: If there are no pairs classified as links, this measure
            will raise a ZeroDivisionError.

    :param confusion_matrix: The matrix with tp, fn, fp, tn values.

    :return: The fscore
    :rtype: float
    """

    prec = precision(confusion_matrix)
    rec = recall(confusion_matrix)

    return float(2 * prec * rec / (prec + rec))
