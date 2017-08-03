from __future__ import division
from __future__ import unicode_literals

import random
import collections
import statistics
import numpy as np


# Conflict Resolution Decorator
def conflict_resolution_function(metadata=None, param=None):
    """
    Used to assert high-level requirements in the
    conflict resolution process.

    :param bool metadata: True/False
    :param bool param: True/False
    :return: function
    """

    def decorate(func):
        func.metadata = metadata
        func.param = param
        return func

    return decorate


# Conflict Resolution Functions

@conflict_resolution_function(metadata=False, param=False)
def identity(x):
    """
    Returns an unchanged value without strategy. (Technically, it is the "first",
    though ordering isn't especially meaningful in this context.) It is intended
    to be used to produce data columns as "resolved columns".

    :param tuple x: A 1-tuple cotaining a value, inside another 1-tuple.
    :return: The unchanged value.
    """
    return x[0][0]


@conflict_resolution_function(metadata=False, param=False)
def count(x):
    """
    Returns the number of unique values.

    :param tuple x: Contains a tuple of values to be resolved.
    :return: The number of unique values.
    """
    vals, = x
    length = len(vals)
    if length == 0:
        return np.nan
    return len(set(vals))


@conflict_resolution_function(metadata=False, param=False)
def choose_min(x):
    """
    Choose the smallest value.
    :param tuple x: Contains a tuple of values to be resolved.
    :return: The smallest value.
    """
    vals, = x
    length = len(vals)
    if length == 0:
        return np.nan
    return min(vals)


@conflict_resolution_function(metadata=False, param=False)
def choose_max(x):
    """
    Choose the largest value.
    :param tuple x: Contains a tuple of values to be resolved.
    :return: The largest value.
    """
    vals, = x
    length = len(vals)
    if length == 0:
        return np.nan
    return max(vals)


@conflict_resolution_function(metadata=False, param=False)
def choose_random(x):
    """
    Choose a random value.
    :param tuple x: Contains a tuple of values to be resolved.
    :return: A random value.
    """
    vals, = x
    length = len(vals)
    if length == 0:
        return np.nan
    return random.choice(vals)


@conflict_resolution_function(metadata=False, param=False)
def vote(x):
    """
    Returns the most common element.
    :param tuple x: Contains a tuple of values to be resolved.
    :return:
    """
    vals, = x
    length = len(vals)
    if length == 0:
        return np.nan
    elif length == 1:
        return vals[0]
    counter = collections.Counter(vals)
    return counter.most_common()[0][0]


@conflict_resolution_function(metadata=False, param=False)
def group(x):
    """
    Returns a set of all conflicting values.
    :param tuple x: Contains a tuple of values to be resolved.
    :return: A set of values.
    """
    vals, = x
    return set(vals)
    # TODO: Implement choice of collection returned.


@conflict_resolution_function(metadata=False, param=False)
def no_gossip(x):
    """
    Returns values if all items in vals are equal, or np.nan
    otherwise.

    :param tuple x: Contains a tuple of values to be resolved.
    :return: A canonical value or np.nan.
    """
    vals, = x
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


@conflict_resolution_function(metadata=False, param=True)
def compute_metric(x, method):
    """
    Returns a numerical aggregation of values.

    :param tuple x: Values to _resolve
    :param str method: Aggregation method.
    :return: A numerical aggregation of vals.
    """
    vals, = x
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


@conflict_resolution_function(metadata=True, param=True)
def choose(x, trusted):
    """
    Choose a value from a trusted source.

    :param tuple x: Values to _resolve, tuple of value sources
    :param trusted: Trusted source identifier.
    :return: A trusted value.
    """
    vals, meta = x
    length = len(vals)
    if length == 0:
        return np.nan
    for i in range(len(vals)):
        if meta[i] == trusted:
            # Short circuit for trusted value
            return vals[i]
    # Return nan if no trusted value present
    return np.nan


@conflict_resolution_function(metadata=True, param=False)
def annotated_concat(x):
    """
    Returns a collection of value/metadata pairs.

    :param (tuple, tuple) x: Values to be resolved, and metadata values.
    :return: A list of value/metadata tuples.
    """
    vals, meta = x
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


@conflict_resolution_function(metadata=True, param=False)
def choose_metadata_min(x):
    """
    Chooses the value with the smallest corresponding metadata value.

    :param (tuple, tuple) x: Values to be resolved, corresponding metadata.
    :return: A canonical value.
    """
    vals, meta = x
    length = len(vals)
    if length == 0:
        return np.nan
    m = min(meta)
    i = meta.index(m)
    return vals[i]


@conflict_resolution_function(metadata=True, param=False)
def choose_metadata_max(x):
    """
    Chooses the value with the largest corresponding metadata value.

    :param (tuple, tuple) x: Values to be resolved, corresponding metadata.
    :return: A canonical value.
    """
    vals, meta = x
    length = len(vals)
    if length == 0:
        return np.nan
    m = max(meta)
    i = meta.index(m)
    return vals[i]
