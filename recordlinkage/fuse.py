from __future__ import division
from __future__ import unicode_literals

import random
import warnings
import collections
import multiprocessing as mp
import pandas as pd
import recordlinkage as rl


#######################################
# Conflict Resolution Decorators
#######################################

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


class FuseCore(object):
    def __init__(self):
        pass

    def resolve(self, fun, vals, meta=None, **kwargs):
        # Integrate values in vals (Series of tuples) using optional
        # meta (Series of tuples) by applying the conflict resolution function fun
        # to the series of tuples.
        pass

    def queue_resolve(self, fun, vals, meta=None, **kwargs):
        # Integrate values in vals (Series of tuples) using optional
        # meta (Series of tuples) by applying the conflict resolution function fun
        # to the series of tuples.
        pass

    def _format_resolve(self, c1, c2, m1=None, m2=None, trans_c=None, trans_m=None, memoize=False, *args, **kwargs):
        # No implementation provided.
        # Override in subclass.
        pass

    # Conflict Resolution Functions

    @conflict_resolution_function(metadata=False, param=False)
    def count(self, vals):
        """
        Returns the number of unique values.
        :param tuple vals: Values to resolve.
        :return: The number of unique values.
        """
        return len(set(vals))

    @conflict_resolution_function(metadata=False, param=False)
    def choose_min(self, vals):
        """
        Choose the smallest value.
        :param tuple vals: Values to resolve.
        :return: The smallest value.
        """
        return min(vals)

    @conflict_resolution_function(metadata=False, param=False)
    def choose_max(self, vals):
        """
        Choose the largest value.
        :param tuple vals: Values to resolve.
        :return: The largest value.
        """
        return max(vals)

    @conflict_resolution_function(metadata=False, param=False)
    def random(self, vals):
        """
        Choose a random value.
        :param tuple vals: Values to resolve.
        :return: A random value.
        """
        random.choice(vals)

    @conflict_resolution_function(metadata=False, param=False)
    def vote(self, vals):
        """
        Returns the most common element.
        :param vals:
        :return:
        """
        counter = collections.Counter(vals)
        return counter.most_common()[0][0]

    @conflict_resolution_function(metadata=False, param=False)
    def group(self, vals):
        pass

    @conflict_resolution_function(metadata=False, param=False)
    def shortest(self, vals):
        pass

    @conflict_resolution_function(metadata=False, param=False)
    def longest(self, vals):
        pass

    @conflict_resolution_function(metadata=False, param=False)
    def no_gossip(self, vals):
        pass

    @conflict_resolution_function(metadata=False, param=True)
    def compute_metric(self, vals, method):
        pass

    @conflict_resolution_function(metadata=True, param=True)
    def choose(self, vals, meta, trusted):
        # Conflict resolution function
        # One realization of trust_your_friends
        pass

    @conflict_resolution_function(metadata=True, param=False)
    def annotated_concat(self, vals, meta):
        pass

    @conflict_resolution_function(metadata=True, param=False)
    def minimize_metadata_value(self, vals, meta):
        pass

    @conflict_resolution_function(metadata=True, param=False)
    def maximize_metadata_value(self, vals, meta):
        pass

    # @conflict_resolution_function
    # @requires_metadata(True)
    # def template_resolution_func(self, vals, meta):
    #     """
    #     Resolve values using optional metadata.
    #     :param vals: A tuple of values
    #     :param meta: An optional tuple of metadata
    #     :return: A canonical value
    #     """
    #     # Assert invariants e.g. type checking
    #     # Resolve conflicts / choose / generate value
    #     # Return

    # Conflict Resolution Realizations

    @conflict_resolution_realization
    def trust_your_friends(self, c1, c2, trusted, method='choose'):
        if method == 'choose':
            return self.queue_resolve(self.choose, self._format_resolve(c1, c2, trust=trusted), trust=trusted)
        elif method == 'highest_quality':
            return self.queue_resolve(self.highest_quality, self._format_resolve(c1, c2, trust=trusted), trust=trusted)
        else:
            warnings.warn('Unrecognized method for trust_your_friends')

    @conflict_resolution_realization
    def no_gossiping(self, c1, c2):
        pass

    @conflict_resolution_realization
    def take_the_information(self, c1, c2):
        pass

    def fuse(self, vectors, df_a, df_b, suffix_a='_a', suffix_b='_b'):
        # Apply refinements to vectors / index
        # Make calls to `resolve` using accumulated metadata
        # Return the fused data frame
        pass


class FuseClusters(FuseCore):
    def __init__(self, method='???'):
        super().__init__()
        self.method = method

    def _find_clusters(self, method):
        pass

    def _format_resolve(self, c1, c2, m1=None, m2=None, trans_c=None, trans_m=None, *args, **kwargs):
        pass


class FuseLinks(FuseCore):
    def __init__(self, unique_a=False, unique_b=False):
        super().__init__()
        self.unique_a = unique_a
        self.unique_b = unique_b

    def _apply_refinement(self):
        pass

    def _format_resolve(self, c1, c2, m1=None, m2=None, trans_c=None, trans_m=None, *args, **kwargs):
        pass
