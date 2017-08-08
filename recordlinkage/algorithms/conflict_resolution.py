from __future__ import division
from __future__ import unicode_literals

import random
import collections
import statistics
from operator import itemgetter

import numpy as np


def remove_missing(x, remove_na_vals, remove_na_meta):
    """
    Removes missing (np.nan) values from a nested tuple of values and metadata values.

    :param x: A tuple of one or two n-tuples.
    :return: A tuple of one or two n-tuples.
    """
    length = len(x[0])
    if length == 0:
        return x
    keep_indices = list(range(length))
    print(length, keep_indices)
    # Remove nan-value indices from list
    for i in range(length-1, -1, -1):
        if remove_na_vals and np.isnan(x[0][i]) or remove_na_meta and np.isnan(x[1][i]):
            keep_indices.pop(i)

    # If no changes are needed, return x as-is.
    if len(keep_indices) == len(x[0]):
        return x
    else:
        getter = itemgetter(*keep_indices)
        return getter(x[0]), getter(x[1])


def bool_getter(x, fun):
    """
    Takes a tuple and a function. It returns a function which returns items
    as positions i such that fun(x[i]) is True.
    :param tuple x: A tuple of values
    :param function fun: A predicate function
    :return: operator.itemgetter
    """
    keep_indices = list(range(len(x)))
    for i in range(len(x)-1, -1, -1):
        if fun(x[i]) is False:
            keep_indices.pop(i)
    getter = itemgetter(*keep_indices)
    return getter


def identity(x, remove_na_vals, remove_na_meta):
    """
    Returns an unchanged value without strategy. (Technically, it is the "first",
    though ordering isn't especially meaningful in this context.) It is intended
    to be used to produce data columns as "resolved columns".

    :param tuple x: A 1-tuple cotaining a value, inside another 1-tuple.
    :return: The unchanged value.
    """
    return x[0][0]


def nullify(x, tie_break, remove_na_vals, remove_na_meta):
    """
    Returns null.

    :param tuple x: A tuple of values.
    :return: numpy.nan
    """
    return np.nan


def choose_first(x, remove_na_vals, remove_na_meta):
    """
    Choose the last value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: The last value.
    """
    vals, = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan
    else:
        return vals[0]


def choose_last(x, remove_na_vals, remove_na_meta):
    """
    Choose the last value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: The last value.
    """
    vals, = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan
    else:
        return vals[-1]


def count(x, remove_na_vals, remove_na_meta):
    """
    Returns the number of unique values.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: The number of unique values.
    """
    vals, = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan
    return len(set(vals))


def choose_min(x, tie_break, remove_na_vals, remove_na_meta):
    """
    Choose the smallest value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: The smallest value.
    """
    vals, = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan
    return min(vals)


def choose_max(x, tie_break, remove_na_vals, remove_na_meta):
    """
    Choose the largest value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: The largest value.
    """
    vals, = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan
    return max(vals)


def choose_shortest(x, tie_break, remove_na_vals, remove_na_meta):
    """
    Choose the shortest value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: The shortest value.
    """
    vals, = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan
    return min(vals, key=len)


def choose_longest(x, tie_break, remove_na_vals, remove_na_meta):
    """
    Choose the longest value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: The longest value.
    """
    vals, = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan
    return max(vals, key=len)


def choose_random(x, remove_na_vals, remove_na_meta):
    """
    Choose a random value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: A random value.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    return random.choice(vals)


def vote(x, tie_break, remove_na_vals, remove_na_meta):
    """
    Returns the most common element.

    :param tuple x: Contains a tuple of values to be resolved.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: A canonical value.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    elif length == 1:
        return vals[0]
    counter = collections.Counter(vals)

    # Handle ties

    return counter.most_common()[0][0]


def group(x, remove_na_vals, remove_na_meta):
    """
    Returns a set of all conflicting values.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: A set of values.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    return set(vals)
    # TODO: Implement choice of collection returned.


def no_gossip(x, remove_na_vals, remove_na_meta):
    """
    Returns values if all items in vals are equal, or np.nan
    otherwise.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: A canonical value or np.nan.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    elif length == 1:
        return vals[0]
    for i in vals:
        if i != vals[0]:
            # Short circuit on inconsistent value
            return np.nan
    # If no inconsistencies are seen, return the first value.
    return vals[0]


def aggregate(x, method, remove_na_vals, remove_na_meta):
    """
    Returns a numerical aggregation of values.

    :param tuple x: Values to _resolve
    :param str method: Aggregation method.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: A numerical aggregation of vals.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    if method == 'sum':
        return sum(vals)
    elif method == 'mean':
        return statistics.mean(vals)
    elif method == 'stdev':
        return statistics.stdev(vals)
    elif method == 'variance':
        return statistics.variance(vals)


def choose(x, trusted, tie_break, remove_na_vals, remove_na_meta):
    """
    Choose a value from a trusted source.

    :param tuple x: Values to _resolve, tuple of value sources
    :param trusted: Trusted source identifier.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: A trusted value.
    """
    vals, meta = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan
    for i in range(len(vals)):
        if meta[i] == trusted:
            # Short circuit for trusted value
            return vals[i]
    # Return nan if no trusted value present
    return np.nan


def annotated_concat(x, remove_na_vals, remove_na_meta):
    """
    Returns a collection of value/metadata pairs.

    :param (tuple, tuple) x: Values to be resolved, and metadata values.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: A list of value/metadata tuples.
    """
    vals, meta = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan
    seen = set()
    concat = []
    for i in range(len(vals)):
        if vals[i] not in seen:
            concat.append((vals[i], meta[i]))
            seen.add(vals[i])
        else:
            continue
    return concat


def choose_metadata_min(x, tie_break, remove_na_vals, remove_na_meta):
    """
    Chooses the value with the smallest corresponding metadata value.

    :param (tuple, tuple) x: Values to be resolved, corresponding metadata.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: A canonical value.
    """
    vals, meta = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan
    m = min(meta)
    i = meta.index(m)
    return vals[i]


def choose_metadata_max(x, tie_break, remove_na_vals, remove_na_meta):
    """
    Chooses the value with the largest corresponding metadata value.

    :param (tuple, tuple) x: Values to be resolved, corresponding metadata.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: A canonical value.
    """
    vals, meta = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan
    m = max(meta)
    i = meta.index(m)
    return vals[i]
