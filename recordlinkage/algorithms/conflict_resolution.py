from __future__ import division
from __future__ import unicode_literals

import random
import collections
from typing import Callable, TypeVar
import numpy as np

import recordlinkage.rl_logging as rl_log

TieBreaker = Callable[[tuple, bool], any]


def safe_isnan(x):
    """
    Check if value is np.nan. Handles non-float values.

    :param x: Some value.
    :return: Bool.
    """
    if isinstance(x, float):
        return np.isnan(x)
    else:
        return False


def tupgetter(*items):
    """
    Identical to operator.itemgetter, but returns single items as 1-tuples instead of atomic values.

    :param items: A collection of indexes.
    :return: A tupgetter function.
    """
    if len(items) == 1:
        item = items[0]
        def g(obj):
            return (obj[item], )
    else:
        def g(obj):
            return tuple(obj[item] for item in items)
    return g


def remove_missing(x, remove_na_vals, remove_na_meta):
    """
    Removes missing (np.nan) values from a nested tuple of values and metadata values.
    Value/metadata pairs are removed as pairs.

    :param x: A tuple of one or two n-tuples.
    :param bool remove_na_vals: Should nan values be excluded?
    :param bool remove_na_meta: Should nan metadata values be excluded?
    :return: A tuple of one or two n-tuples.
    """
    # Assert requirements
    if len(x) == 1 and remove_na_meta:
            raise AssertionError('remove_na_meta is True but no metadata was passed.')

    # Process input
    length = len(x[0])
    if length == 0:
        return x
    keep_indices = list(range(length))

    # Remove nan-value indices from list
    for i in range(length-1, -1, -1):
        if remove_na_vals and safe_isnan(x[0][i]) or remove_na_meta and safe_isnan(x[1][i]):
            keep_indices.pop(i)

    # If no changes are needed, return x as-is.
    if len(keep_indices) == len(x[0]):
        return x
    else:
        getter = tupgetter(*keep_indices)
        if len(x) == 1:
            return (getter(x[0]), )
        else:
            return getter(x[0]), getter(x[1])


def bool_getter(x, fun):
    """
    Takes a tuple and a function. It returns a function which returns items
    as positions i such that fun(x[i]) is True.
    :param tuple x: A tuple of values
    :param function fun: A predicate function
    :return: A tupgetter function.
    """
    keep_indices = list(range(len(x)))
    for i in range(len(x)-1, -1, -1):
        if fun(x[i]) is False:
            keep_indices.pop(i)
    getter = tupgetter(*keep_indices)
    return getter


def identity(x, remove_na_vals):
    """
    Returns an unchanged value without strategy. (Technically, it is the "first",
    though ordering isn't especially meaningful in this context.) It is intended
    to be used to produce data columns as "resolved columns".

    :param tuple x: A 1-tuple cotaining a value, inside another 1-tuple.
    :return: The unchanged value.
    """
    if len(x[0]) == 0:
        return np.nan
    else:
        return x[0][0]


def nullify(x, remove_na_vals):
    """
    Returns null.

    :param tuple x: A tuple of values.
    :param bool remove_na_vals: Included for consistency with tie-breaking conflict resolution functions.
    :return: numpy.nan
    """
    return np.nan


def choose_first(x, remove_na_vals):
    """
    Choose the last value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: The last value.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    else:
        return vals[0]


def choose_last(x, remove_na_vals):
    """
    Choose the last value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: The last value.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    else:
        return vals[-1]


def count(x, remove_na_vals):
    """
    Returns the number of unique values.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: The number of unique values.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    return len(set(vals))


def choose_min(x, remove_na_vals):
    """
    Choose the smallest value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: The smallest value.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    try:
        return min(vals)
    except TypeError:
        rl_log.warn('Conflict resolution warning: '
                    'attempted to choose max between '
                    'type-incompatible values {}. Returning nan.'.format(x))
        return np.nan


def choose_max(x, remove_na_vals):
    """
    Choose the largest value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: The largest value.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    try:
        return max(vals)
    except TypeError:
        rl_log.warn('Conflict resolution warning: '
                    'attempted to choose max between '
                    'type-incompatible values {}. Returning nan.'.format(x))
        return np.nan


def choose_shortest(x, tie_break, remove_na_vals):
    """
    Choose the shortest value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: The shortest value.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    v = min(vals, key=len)
    if tie_break is not None:
        l = len(v)
        index = []
        for i in range(len(vals)):
            if len(vals[i]) == l:
                index.append(i)
        if len(index) > 1:
            return tie_break((tupgetter(*index)(vals), ), False)
        else:
            return v
    else:
        return v


def choose_longest(x, tie_break, remove_na_vals):
    """
    Choose the longest value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: The longest value.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    v = max(vals, key=len)
    if tie_break is not None:
        l = len(v)
        index = []
        for i in range(len(vals)):
            if len(vals[i]) == l:
                index.append(i)
        if len(index) > 1:
            return tie_break((tupgetter(*index)(vals), ), False)
        else:
            return v
    else:
        return v


def choose_shortest_tie_break(x, remove_na_vals):
    """
    Choose the shortest value. Used for tie breaking, adn written in terms of choose_longest.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: The shortest value.
    """
    return choose_shortest(x, None, remove_na_vals)


def choose_longest_tie_break(x, remove_na_vals):
    """
    Choose the longest value. Used for tie breaking, and written in terms of choose_longest.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: The longest value.
    """
    return choose_longest(x, None, remove_na_vals)


def choose_random(x, remove_na_vals):
    """
    Choose a random value.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: A random value.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    return random.choice(vals)


def vote(x, tie_break, remove_na_vals):
    """
    Returns the most common element.

    :param tuple x: Contains a tuple of values to be resolved.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: A canonical value.
    """
    # Process input
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    elif length == 1:
        return vals[0]

    # Count occurrences
    counter = collections.Counter(vals)

    # Check for ties
    count = counter.most_common()[0][1]
    common = [t for t in counter.most_common() if t[1] == count]

    if len(common) > 1:
        # Tie
        return tie_break((tuple([x[0] for x in common]), ), False)
    else:
        # No tie
        return common[0][0]


def group(x, kind, remove_na_vals):
    """
    Returns a set of all conflicting values.

    :param tuple x: Contains a tuple of values to be resolved.
    :param str kind: The type of collection to be returned. One of set, list, tuple.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: A set of values.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    s = set(vals)
    if kind == 'set':
        return s
    elif kind == 'list':
        return list(s)
    elif kind == 'tuple':
        return tuple(s)
    else:
        raise ValueError('Unrecognized collection kind.')


def no_gossip(x, remove_na_vals):
    """
    Returns values if all items in vals are equal, or np.nan
    otherwise.

    :param tuple x: Contains a tuple of values to be resolved.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
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


def aggregate(x, method, remove_na_vals):
    """
    Returns a numerical aggregation of values.

    :param tuple x: Values to _resolve
    :param str method: Aggregation method.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :return: A numerical aggregation of vals.
    """
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    if method == 'sum':
        return sum(vals)
    elif method == 'mean':
        return np.mean(vals)
    elif method == 'stdev':
        return np.std(vals)
    elif method == 'var':
        return np.var(vals)
    else:
        raise ValueError('Unrecognized aggregation method: {}.'.format(method))


def choose_trusted(x, trusted, tie_break_trusted, tie_break_untrusted, remove_na_vals, remove_na_meta):
    """
    Prefers values from a trusted source.

    :param tuple x: Values to _resolve, tuple of value sources
    :param trusted: Trusted source identifier.
    :param function tie_break_trusted: A conflict resolution function to be to break ties between trusted values.
    :param function tie_break_untrusted: A conflict resolution function to be to break ties between trusted values.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: A trusted value.
    """
    # Process input
    vals, meta = remove_missing(x, remove_na_vals, remove_na_meta)
    v_len = len(vals)
    if v_len == 0:
        return np.nan

    # Find trusted value(s)
    trust_vals = bool_getter(meta, lambda v: v == trusted)(vals)
    t_len = len(trust_vals)
    if t_len == 0:
        # No trusted values
        untrusted_vals = bool_getter(meta, lambda v: v != trusted)(vals)
        if len(untrusted_vals) == 0:
            return np.nan
        else:
            return tie_break_untrusted((untrusted_vals, ), False)
    elif t_len == 1:
        # One trusted value
        return trust_vals[0]
    else:
        # Tie break
        return tie_break_trusted((trust_vals, ), False)


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
        return []
    concat = []
    for i in range(len(vals)):
        concat.append((vals[i], meta[i]))
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
    # Process input
    vals, meta = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan

    # Find min metadata value
    m = min(meta)

    # Find corresponding value(s)
    take_vals = bool_getter(meta, lambda v: v == m)(vals)
    t_len = len(take_vals)
    if t_len == 1:
        # No tie
        return take_vals[0]
    else:
        # Tie break
        return tie_break((take_vals, ), False)


def choose_metadata_max(x, tie_break, remove_na_vals, remove_na_meta):
    """
    Chooses the value with the largest corresponding metadata value.

    :param (tuple, tuple) x: Values to be resolved, corresponding metadata.
    :param function tie_break: A conflict resolution function to be used in the case of a tie.
    :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
    :return: A canonical value.
    """
    # Process input
    vals, meta = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan

    # Find min metadata value
    m = max(meta)

    # Find corresponding value(s)
    take_vals = bool_getter(meta, lambda v: v == m)(vals)
    t_len = len(take_vals)
    if t_len == 1:
        # No tie
        return take_vals[0]
    else:
        # Tie break
        return tie_break((take_vals, ), False)
