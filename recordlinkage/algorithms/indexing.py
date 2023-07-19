"""Algorithms for indexing."""

import numpy as np

from recordlinkage.measures import full_index_size


def _map_tril_1d_on_2d(indices, dims):
    """Map 1d indices on lower triangular matrix in 2d."""

    N = (dims * dims - dims) / 2

    m = np.ceil(np.sqrt(2 * N))
    c = m - np.round(np.sqrt(2 * (N - indices))) - 1
    r = np.mod(indices + (c + 1) * (c + 2) / 2 - 1, m) + 1

    return np.array([r, c], dtype=np.int64)


def random_pairs_with_replacement(n, shape, random_state=None):
    """make random record pairs"""

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    n_max = full_index_size(shape)

    if n_max <= 0:
        raise ValueError("n_max must be larger than 0")

    # make random pairs
    indices = random_state.randint(0, n_max, n, dtype=np.int64)

    if len(shape) == 1:
        return _map_tril_1d_on_2d(indices, shape[0])
    else:
        return np.array(np.unravel_index(indices, shape))


def random_pairs_without_replacement(n, shape, random_state=None):
    """Return record pairs for dense sample.

    Sample random record pairs without replacement bounded by the
    maximum number of record pairs (based on shape). This algorithm is
    efficient and fast for relative small samples.
    """

    n_max = full_index_size(shape)

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    if not isinstance(n, int) or n <= 0 or n > n_max:
        raise ValueError("n must be a integer satisfying 0<n<=%s" % n_max)

    # make a sample without replacement
    sample = random_state.choice(np.arange(n_max), n, replace=False)

    # return 2d indices
    if len(shape) == 1:
        return _map_tril_1d_on_2d(sample, shape[0])
    else:
        return np.array(np.unravel_index(sample, shape))


def random_pairs_without_replacement_low_memory(n, shape, random_state=None):
    """Make a sample of random pairs with replacement.

    Sample random record pairs without replacement bounded by the
    maximum number of record pairs (based on shape). This algorithm
    consumes low memory and is fast for relatively small samples.
    """

    n_max = full_index_size(shape)

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    if not isinstance(n, int) or n <= 0 or n > n_max:
        raise ValueError("n must be a integer satisfying 0<n<=%s" % n_max)

    sample = np.array([], dtype=np.int64)

    # Run as long as the number of pairs is less than the requested number
    # of pairs n.
    while len(sample) < n:
        # The number of pairs to sample (sample twice as much record pairs
        # because the duplicates are dropped).
        n_sample_size = (n - len(sample)) * 2
        sample_sub = random_state.randint(n_max, size=n_sample_size)

        # concatenate pairs and deduplicate
        pairs_non_unique = np.append(sample, sample_sub)
        sample = np.unique(pairs_non_unique)

    # return 2d indices
    if len(shape) == 1:
        return _map_tril_1d_on_2d(sample[0:n], shape[0])
    else:
        return np.array(np.unravel_index(sample[0:n], shape))
