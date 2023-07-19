# measures.py

import numpy
import pandas

from recordlinkage.utils import get_length


def _get_multiindex(x):
    if isinstance(x, (pandas.DataFrame, pandas.Series)):
        return x.index
    elif isinstance(x, pandas.MultiIndex):
        return x
    else:
        raise ValueError(
            "Expected one of: pandas.DataFrame, " "pandas.Series, pandas.MultiIndex"
        )


def _isconfusionmatrix(x):
    if isinstance(x, numpy.ndarray) and x.shape == (2, 2):
        return True
    elif isinstance(x, list) and numpy.ndarray(x).shape == (2, 2):
        return True
    else:
        return False


def reduction_ratio(links_pred, *total):
    """Compute the reduction ratio.

    The reduction ratio is 1 minus the ratio candidate matches and the maximum
    number of pairs possible.

    Parameters
    ----------
    links_pred: int, pandas.MultiIndex
        The number of candidate record pairs or the pandas.MultiIndex with
        record pairs.
    *total: pandas.DataFrame object(s)
        The DataFrames are used to compute the full index size with the
        full_index_size function.

    Returns
    -------
    float
        The reduction ratio.

    """

    n_max = full_index_size(*total)

    if isinstance(links_pred, pandas.MultiIndex):
        links_pred = len(links_pred)

    if links_pred > n_max:
        raise ValueError("n has to be smaller of equal n_max")

    return 1 - links_pred / n_max


def max_pairs(shape):
    """[DEPRECATED] Compute the maximum number of record pairs possible."""

    if not isinstance(shape, (tuple, list)):
        x = get_length(shape)
        n = int(x * (x - 1) / 2)

    elif isinstance(shape, (tuple, list)) and len(shape) == 1:
        x = get_length(shape[0])
        n = int(x * (x - 1) / 2)

    else:
        n = numpy.prod([get_length(xi) for xi in shape])

    return n


def full_index_size(*args):
    """Compute the number of records in a full index.

    Compute the number of records in a full index without building the index
    itself. The result is the maximum number of record pairs possible. This
    function is especially useful in measures like the `reduction_ratio`.

    Deduplication: Given a DataFrame A with length N, the full index size is
    N*(N-1)/2. Linking: Given a DataFrame A with length N and a DataFrame B
    with length M, the full index size is N*M.

    Parameters
    ----------
    *args: int, pandas.MultiIndex, pandas.Series, pandas.DataFrame
        A pandas object or a int representing the length of a dataset to link.
        When there is one argument, it is assumed that the record linkage is
        a deduplication process.

    Examples
    --------

    Use integers:
    >>> full_index_size(10)  # deduplication: 45 pairs
    >>> full_index_size(10, 10)  # linking: 100 pairs

    or pandas objects
    >>> full_index_size(DF)  # deduplication: len(DF)*(len(DF)-1)/2 pairs
    >>> full_index_size(DF, DF)  # linking: len(DF)*len(DF) pairs

    """

    # check if a list or tuple is passed as argument
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = tuple(args[0])

    if len(args) == 1:
        n = get_length(args[0])
        size = int(n * (n - 1) / 2)
    else:
        size = numpy.prod([get_length(arg) for arg in args])

    return size


def true_positives(links_true, links_pred):
    """Count the number of True Positives.

    Returns the number of correctly predicted links, also called the number of
    True Positives (TP).

    Parameters
    ----------
    links_true: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The true (or actual) links.
    links_pred: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The predicted links.

    Returns
    -------
    int
        The number of correctly predicted links.
    """

    links_true = _get_multiindex(links_true)
    links_pred = _get_multiindex(links_pred)

    return len(links_true.intersection(links_pred))


def true_negatives(links_true, links_pred, total):
    """Count the number of True Negatives.

    Returns the number of correctly predicted non-links, also called the
    number of True Negatives (TN).

    Parameters
    ----------
    links_true: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The true (or actual) links.
    links_pred: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The predicted links.
    total: int, pandas.MultiIndex
        The count of all record pairs (both links and non-links). When the
        argument is a pandas.MultiIndex, the length of the index is used.

    Returns
    -------
    int
        The number of correctly predicted non-links.

    """

    links_true = _get_multiindex(links_true)
    links_pred = _get_multiindex(links_pred)

    if isinstance(total, pandas.MultiIndex):
        total = len(total)

    return int(total) - len(links_true.union(links_pred))


def false_positives(links_true, links_pred):
    """Count the number of False Positives.

    Returns the number of incorrect predictions of true non-links. (true non-
    links, but predicted as links). This value is known as the number of False
    Positives (FP).

    Parameters
    ----------
    links_true: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The true (or actual) links.
    links_pred: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The predicted links.

    Returns
    -------
    int
        The number of false positives.

    """

    links_true = _get_multiindex(links_true)
    links_pred = _get_multiindex(links_pred)

    return len(links_pred.difference(links_true))


def false_negatives(links_true, links_pred):
    """Count the number of False Negatives.

    Returns the number of incorrect predictions of true links. (true links,
    but predicted as non-links). This value is known as the number of False
    Negatives (FN).

    Parameters
    ----------
    links_true: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The true (or actual) links.
    links_pred: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The predicted links.

    Returns
    -------
    int
        The number of false negatives.

    """

    links_true = _get_multiindex(links_true)
    links_pred = _get_multiindex(links_pred)

    return len(links_true.difference(links_pred))


def confusion_matrix(links_true, links_pred, total=None):
    """Compute the confusion matrix.

    The confusion matrix is of the following form:

    +----------------------+-----------------------+----------------------+
    |                      |  Predicted Positives  | Predicted Negatives  |
    +======================+=======================+======================+
    | **True Positives**   | True Positives (TP)   | False Negatives (FN) |
    +----------------------+-----------------------+----------------------+
    | **True Negatives**   | False Positives (FP)  | True Negatives (TN)  |
    +----------------------+-----------------------+----------------------+

    The confusion matrix is an informative way to analyse a prediction. The
    matrix can used to compute measures like precision and recall. The count
    of true prositives is [0,0], false negatives is [0,1], true negatives
    is [1,1] and false positives is [1,0].

    Parameters
    ----------
    links_true: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The true (or actual) links.
    links_pred: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The predicted links.
    total: int, pandas.MultiIndex
        The count of all record pairs (both links and non-links). When the
        argument is a pandas.MultiIndex, the length of the index is used. If
        the total is None, the number of True Negatives is not computed.
        Default None.

    Returns
    -------
    numpy.array
        The confusion matrix with TP, TN, FN, FP values.

    Note
    ----
    The number of True Negatives is computed based on the total argument.
    This argument is the number of record pairs of the entire matrix.

    """

    links_true = _get_multiindex(links_true)
    links_pred = _get_multiindex(links_pred)

    tp = true_positives(links_true, links_pred)
    fp = false_positives(links_true, links_pred)
    fn = false_negatives(links_true, links_pred)

    if total is None:
        tn = numpy.nan
    else:
        if isinstance(total, pandas.MultiIndex):
            total = len(total)
        tn = true_negatives(links_true, links_pred, total)

    return numpy.array([[tp, fn], [fp, tn]])


def precision(links_true, links_pred=None):
    """precision(links_true, links_pred)

    Compute the precision.

    The precision is given by TP/(TP+FP).

    Parameters
    ----------
    links_true: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The true (or actual) collection of links.
    links_pred: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The predicted collection of links.

    Returns
    -------
    float
        The precision
    """

    if _isconfusionmatrix(links_true):
        confusion_matrix = links_true

        v = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
    else:
        tp = true_positives(links_true, links_pred)
        fp = false_positives(links_true, links_pred)
        v = tp / (tp + fp)

    return float(v)


def recall(links_true, links_pred=None):
    """recall(links_true, links_pred)

    Compute the recall/sensitivity.

    The recall is given by TP/(TP+FN).

    Parameters
    ----------
    links_true: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The true (or actual) collection of links.
    links_pred: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The predicted collection of links.

    Returns
    -------
    float
        The recall
    """

    if _isconfusionmatrix(links_true):
        confusion_matrix = links_true

        v = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    else:
        tp = true_positives(links_true, links_pred)
        fn = false_negatives(links_true, links_pred)
        v = tp / (tp + fn)

    return float(v)


def accuracy(links_true, links_pred=None, total=None):
    """accuracy(links_true, links_pred, total)

    Compute the accuracy.

    The accuracy is given by (TP+TN)/(TP+FP+TN+FN).

    Parameters
    ----------
    links_true: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The true (or actual) collection of links.
    links_pred: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The predicted collection of links.
    total: int, pandas.MultiIndex
        The count of all record pairs (both links and non-links). When the
        argument is a pandas.MultiIndex, the length of the index is used.

    Returns
    -------
    float
        The accuracy
    """

    if isinstance(total, pandas.MultiIndex):
        total = len(total)

    if _isconfusionmatrix(links_true):
        confusion_matrix = links_true

        v = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / numpy.sum(
            confusion_matrix
        )
    else:
        tp = true_positives(links_true, links_pred)
        tn = true_negatives(links_true, links_pred, total)

        v = (tp + tn) / total

    return float(v)


def specificity(links_true, links_pred=None, total=None):
    """specificity(links_true, links_pred, total)

    Compute the specificity.

    The specificity is given by TN/(FP+TN).

    Parameters
    ----------
    links_true: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The true (or actual) collection of links.
    links_pred: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The predicted collection of links.
    total: int, pandas.MultiIndex
        The count of all record pairs (both links and non-links). When the
        argument is a pandas.MultiIndex, the length of the index is used.

    Returns
    -------
    float
        The specificity

    """

    if _isconfusionmatrix(links_true):
        confusion_matrix = links_true

        v = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1])
    else:
        fp = false_positives(links_true, links_pred)

        if isinstance(total, pandas.MultiIndex):
            total = len(total)
        tn = true_negatives(links_true, links_pred, total)
        v = tn / (fp + tn)

    return float(v)


def fscore(links_true, links_pred=None):
    """fscore(links_true, links_pred)

    Compute the F-score.

    The F-score is given by 2*(precision*recall)/(precision+recall).

    Parameters
    ----------
    links_true: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The true (or actual) collection of links.
    links_pred: pandas.MultiIndex, pandas.DataFrame, pandas.Series
        The predicted collection of links.

    Returns
    -------
    float
        The fscore

    Note
    ----
    If there are no pairs predicted as links, this measure will raise a
    ZeroDivisionError.

    """

    prec = precision(links_true, links_pred)
    rec = recall(links_true, links_pred)

    return float(2 * prec * rec / (prec + rec))
