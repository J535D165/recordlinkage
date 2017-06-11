"""Algorithms for indexing."""

import numpy as np

from recordlinkage.measures import max_pairs


def _map_triu_1d_on_2d(indices, dims):
    """Map 1d indices on upper triangular matrix in 2d. """

    N = (dims * dims - dims) / 2

    m = np.ceil(np.sqrt(2 * N))
    c = m - np.round(np.sqrt(2 * (N - indices))) - 1
    r = np.mod(indices + (c + 1) * (c + 2) / 2 - 1, m) + 1

    return np.array([c, r], dtype=np.int64)


def _unique_rows_numpy(a):
    """return unique rows"""
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def random_pairs_with_replacement(n, shape, random_state=None):
    """make random record pairs"""

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    n_max = max_pairs(shape)

    if n_max <= 0:
        raise ValueError('n_max must be larger than 0')

    # make random pairs
    indices = random_state.randint(0, n_max, n)

    if len(shape) == 1:
        return _map_triu_1d_on_2d(indices, shape[0])
    else:
        return np.unravel_index(indices, shape)


def random_pairs_without_replacement_small_frames(
        n, shape, random_state=None):

    n_max = max_pairs(shape)

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    if not isinstance(n, int) or n <= 0 or n > n_max:
        raise ValueError("n must be a integer satisfying 0<n<=%s" % n_max)

    # make a sample without replacement
    sample = random_state.choice(
        np.arange(n_max), n, replace=False)

    # return 2d indices
    if len(shape) == 1:
        return _map_triu_1d_on_2d(sample, shape[0])
    else:
        return np.unravel_index(sample, shape)


def random_pairs_without_replacement_large_frames(
        n, shape, random_state=None):
    """Make a sample of random pairs with replacement"""

    n_max = max_pairs(shape)

    sample = np.array([])

    # Run as long as the number of pairs is less than the requested number
    # of pairs n.
    while len(sample) < n:

        # The number of pairs to sample (sample twice as much record pairs
        # because the duplicates are dropped).
        n_sample_size = (n - len(sample)) * 2
        sample = random_state.randint(n_max, size=n_sample_size)

        # concatenate pairs and deduplicate
        pairs_non_unique = np.append(sample, sample)
        sample = _unique_rows_numpy(pairs_non_unique)

    # return 2d indices
    if len(shape) == 1:
        return _map_triu_1d_on_2d(sample[0:n], shape[0])
    else:
        return np.unravel_index(sample[0:n], shape)
