from __future__ import division
from __future__ import unicode_literals

import random
import collections
import pandas as pd
import numpy as np

import recordlinkage.rl_logging as rl_log


# Note that conflict resolution functions must have parameters in the following order:
#   * First, function-specific or "special" parameters such as "kind" or "metrics"
#   * Second, zero, one, or more tie_break parameters
#   * Third, remove_na_vals
#   * Finally, remove_na_meta if applicable.
# The special-tie-na ordering is handled by FuseCore.resolve, but note that order must
# be respected within these categories (i.e. make sure that you're providing parameters
# in the correct order if there are multiple in a single category.

# A "tie-breaking function" is a special case of conflict-resolution functions
#   which must:
#   * Have a signature like tie_break_fun(x, remove_na_vals)
#   * Use values only – no metadata values
#   * Not require tie-breaking

def assert_tuple_correctness(x, meta=False):
    """
    Check whether the tuple has the correct number of sub-tuples.
    If meta=False x should be in the form tuple(tuple(v1, ..., vn), ),
    otherwise x should be in the form tuple(tuple(v1, ..., vn), tuple(m1, ..., mn))

    :param tuple x: Values to check.
    :param bool meta: Is metadata expected?
    :return None: Raises error if incorrect.
    """
    if meta:
        if len(x) == 1:
            raise AssertionError(
                'Conflict resolution function expected metadata, but did not receive any.'
            )
    else:
        if len(x) > 1:
            raise AssertionError(
                'Conflict resolution function received unexpected metadata.'
            )


def tupgetter(*items):
    """
    Identical to operator.itemgetter, but returns single items as 1-tuples instead of atomic values.
    
    Parameters
    ----------
    items : collection
        A collection of indices.

    Returns
    -------
    Function
        A tupgetter function.
    
    """
    if len(items) == 1:
        item = items[0]

        def g(obj):
            return (obj[item],)
    else:
        def g(obj):
            return tuple(obj[i] for i in items)
    return g


def remove_missing(x, remove_na_vals, remove_na_meta):
    """
    Removes missing (np.nan) values from a nested tuple of values and metadata values.
    Corresponding value/metadata pairs are removed together.
    
    Parameters
    ----------
    x : tuple
        A tuple of one or two n-tuples.
    remove_na_vals : bool
        Should nan values be excluded?
    remove_na_meta : bool
        Should nan metadata values be excluded?

    Returns
    -------
    Tuple
        A tuple of one or two n-tuples.
    
    """
    # Assert requirements
    if len(x) == 1 and remove_na_meta:
        raise AssertionError('remove_na_meta is True but no metadata was passed.')

    # Process input
    length = len(x[0])
    if length == 0:
        return x
    keep_indices = list(range(length))

    def check_null(v):
        if isinstance(v, list):
            return False
        else:
            return pd.isnull(v)

    # Remove nan-value indices from list
    for i in range(length - 1, -1, -1):
        if remove_na_vals and check_null(x[0][i]) or remove_na_meta and check_null(x[1][i]):
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
    in positions i which satisfy: fun(x[i]) == True.
    
    Parameters
    ----------
    x : tuple
        A tuple of values
    fun : function
        A predicate function

    Returns
    -------
    Function
        A tupgetter function.
    
    """
    keep_indices = list(range(len(x)))
    for i in range(len(x) - 1, -1, -1):
        if not fun(x[i]):
            keep_indices.pop(i)
    getter = tupgetter(*keep_indices)
    return getter


def nullify(x, remove_na_vals):
    """
    Returns null.
    
    Parameters
    ----------
    x : tuple
        A tuple of values.
    remove_na_vals : bool
        Included for consistency with tie-breaking conflict resolution functions.

    Returns
    -------
    numpy.nan
    
    """
    return np.nan


def choose_first(x, remove_na_vals):
    """
    Choose the first value.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Any
        The first value.
    
    """
    assert_tuple_correctness(x, False)
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    else:
        return vals[0]


def choose_last(x, remove_na_vals):
    """
    Choose the last value.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Any
        The last value.
    
    """
    assert_tuple_correctness(x, False)
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    else:
        return vals[-1]


def count(x, remove_na_vals):
    """
    Returns the number of unique values.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Int
        The number of unique values.
    
    """
    assert_tuple_correctness(x, False)
    vals, = remove_missing(x, remove_na_vals, False)
    return len(set(vals))


def choose_min(x, remove_na_vals):
    """
    Choose the smallest value.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Any
        The smallest value.
    
    """
    assert_tuple_correctness(x, False)
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    try:
        return min(vals)
    except TypeError:
        rl_log.warn('Conflict resolution warning: '
                    'attempted to choose min between '
                    'type-incompatible values {}. Returning nan.'.format(x))
        return np.nan


def choose_max(x, remove_na_vals):
    """
    Choose the largest value.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Any
        The largest value.
    
    """
    assert_tuple_correctness(x, False)
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
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    tie_break : function
        A conflict resolution function to be used in the case of a tie.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Any
        The shortest value.
    
    """
    assert_tuple_correctness(x, False)
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
            return tie_break((tupgetter(*index)(vals),), False)
        else:
            return v
    else:
        return v


def choose_longest(x, tie_break, remove_na_vals):
    """
    Choose the longest value.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    tie_break : function
        A conflict resolution function to be used in the case of a tie.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Any
        The longest value.
    
    """
    assert_tuple_correctness(x, False)
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
            return tie_break((tupgetter(*index)(vals),), False)
        else:
            return v
    else:
        return v


def choose_shortest_tie_break(x, remove_na_vals):
    """
    Choose the shortest value. Used for tie breaking, written in terms of choose_shortest.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Any
        The shortest value.
    
    """
    assert_tuple_correctness(x, False)
    return choose_shortest(x, None, remove_na_vals)


def choose_longest_tie_break(x, remove_na_vals):
    """
    Choose the longest value. Used for tie breaking, written in terms of choose_longest.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Any
        The longest value.
    
    """
    assert_tuple_correctness(x, False)
    return choose_longest(x, None, remove_na_vals)


def choose_random(x, remove_na_vals):
    """
    Choose a random value.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Any
        A random value.
    
    """
    assert_tuple_correctness(x, False)
    vals, = remove_missing(x, remove_na_vals, False)
    length = len(vals)
    if length == 0:
        return np.nan
    return random.choice(vals)


def vote(x, tie_break, remove_na_vals):
    """
    Returns the most common element.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    tie_break : function
        A conflict resolution function to be used in the case of a tie.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Any
        The most common value.
    
    """
    assert_tuple_correctness(x, False)
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
    max_count = counter.most_common()[0][1]
    common = [t for t in counter.most_common() if t[1] == max_count]

    # If nan values kept, count them and act accordingly;
    # collections.Counter is picky with np.nans.
    if not remove_na_vals:
        nan_count = sum(1 for v in vals if np.isnan(v))
        if nan_count > max_count:
            return np.nan
        elif nan_count == max_count:
            common = [t for t in common if not np.isnan(t[0])]
            common.insert(0, (np.nan, nan_count))

    if len(common) > 1:
        # Tie
        return tie_break((tuple([x[0] for x in common]),), False)
    else:
        # No tie
        return common[0][0]


def group(x, kind, remove_na_vals):
    """
    Returns a set of all conflicting values.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    kind : str
        The type of collection to be returned. One of 'set', 'list', 'tuple'.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Set, List, or Tuple
        A collection of values.
    
    """
    assert_tuple_correctness(x, False)
    vals, = remove_missing(x, remove_na_vals, False)
    if kind == 'set':
        return set(vals)
    elif kind == 'list':
        return list(vals)
    elif kind == 'tuple':
        return tuple(vals)
    else:
        raise ValueError('Unrecognized collection kind.')


def no_gossip(x, remove_na_vals):
    """
    Returns a value if all items in vals are equal, or np.nan
    otherwise.
    
    Parameters
    ----------
    x : tuple
        Contains a tuple of values to be resolved.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Any
        A canonical value or np.nan.
    
    """
    assert_tuple_correctness(x, False)
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
    
    Parameters
    ----------
    x : tuple
        Values to _resolve
    method : str
        Aggregation method. One of 'sum', 'mean', 'stdev', 'var'.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).

    Returns
    -------
    Float
        A numerical aggregation of values.
    
    """
    assert_tuple_correctness(x, False)
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


def choose_trusted(x, trusted, tie_break, tie_break_untrusted, remove_na_vals, remove_na_meta):
    """
    Prefers values from a trusted source.
    
    Parameters
    ----------
    x : tuple
        Values to _resolve, tuple of value sources
    trusted : str
        Trusted source identifier. Values with corresponding metadata equal to this value
        are considered to be "trusted" values.
    tie_break : function
        A conflict resolution function to be to break ties between trusted values.
    tie_break_untrusted : function
        A conflict resolution function to be to break ties between untrusted values.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    remove_na_meta : bool
        If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).

    Returns
    -------
    Any
        A trusted value.

    """
    assert_tuple_correctness(x, True)
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
            return tie_break_untrusted((untrusted_vals,), False)
    elif t_len == 1:
        # One trusted value
        return trust_vals[0]
    else:
        # Tie break
        return tie_break((trust_vals,), False)


def annotated_concat(x, remove_na_vals, remove_na_meta):
    """
    Returns a collection of value/metadata pairs.
    
    Parameters
    ----------
    x : tuple
        Values to be resolved, and metadata values.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    remove_na_meta : bool
        If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).

    Returns
    -------
    List
        A list of value/metadata tuples.
    
    """
    assert_tuple_correctness(x, True)
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
    
    Parameters
    ----------
    x : tuple
        Values to be resolved, corresponding metadata.
    tie_break : function
        A conflict resolution function to be used in the case of a tie.
    remove_na_vals : bool
        If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
    remove_na_meta : bool
        If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).

    Returns
    -------
    Any
        The chosen value.
    
    """
    assert_tuple_correctness(x, True)
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
        return tie_break((take_vals,), False)


def choose_metadata_max(x, tie_break, remove_na_vals, remove_na_meta):
    """
    Chooses the value with the largest corresponding metadata value.
    
    Parameters
    ----------
    x : tuple)
        Values to be resolved, corresponding metadata.
    tie_break : function
        A conflict resolution function to be used in the case of a tie.
    remove_na_vals : bool
        If True, value/metadata pairswill be removed if the value is missing (i.e. np.nan).
    remove_na_meta : bool
        If True, value/metadata pairswill be removed if metadata is missing (i.e. np.nan).

    Returns
    -------
    Any
        The chosen value.
    
    """
    assert_tuple_correctness(x, True)
    # Process input
    vals, meta = remove_missing(x, remove_na_vals, remove_na_meta)
    length = len(vals)
    if length == 0:
        return np.nan

    # Find max metadata value
    m = max(meta)

    # Find corresponding value(s)
    take_vals = bool_getter(meta, lambda v: v == m)(vals)
    t_len = len(take_vals)
    if t_len == 1:
        # No tie
        return take_vals[0]
    else:
        # Tie break
        return tie_break((take_vals,), False)
